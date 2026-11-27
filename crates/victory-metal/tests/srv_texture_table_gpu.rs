//! Hardware proof for a converted multi-entry SRV **texture** table plus a
//! sampler table on Apple Silicon.
//!
//! This is the path the Cycle 07 wire extension unlocks: a v3 descriptor-table
//! entry can name a registered texture (transport kind 8) or a registered
//! sampler handle, instead of only a buffer range. One submission:
//!
//! * pitched-upload copies two DIFFERENT 4x4 RGBA8 images into two registered
//!   textures;
//! * a count==2 SRV texture table and a count==1 sampler table are bound through
//!   the reflected argument-buffer ABI;
//! * one fragment draw samples both table entries with the table sampler and
//!   writes a colour derived from every sampled component, so a wrong entry
//!   index or a wrong texture produces different exact pixels.
//!
//! A second submission binds an unregistered texture handle and asserts the
//! host refuses it with `UnknownHandle` before any command buffer exists.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test srv_texture_table_gpu -- --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::bindings::{
    DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, EncoderStats, PipelineAbi, SamplerDesc};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBufferToTexture, DescriptorKind,
    DescriptorTableEntryV3, DescriptorTableV3, DrawState, LoadOp, PipelineRef, PrimitiveTopology,
    ScissorRect, StoreOp, SubmissionV3, Viewport,
};

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const TEXTURE_EDGE: u32 = 4;
const ROW_PITCH: usize = 256;
const PAGE_BYTES: usize = 16 * 1024;

const TARGET: u64 = 0x2000;
const TEXTURE0: u64 = 0x3000;
const TEXTURE1: u64 = 0x3001;
const SAMPLER: u64 = 0x4000;
const UNKNOWN_TEXTURE: u64 = 0xdead_beef;

const ALLOC_TARGET: u64 = 1;
const ALLOC_TEXTURE0: u64 = 2;
const ALLOC_TEXTURE1: u64 = 3;
const ALLOC_UPLOAD: u64 = 4;

/// Exact RGBA8 upload colours: distinct per table entry.
const COLOR0: [u8; 4] = [10, 20, 30, 255];
const COLOR1: [u8; 4] = [40, 50, 60, 255];

/// `a.r`, `b.g`, `a.b + b.r`, so both entries contribute and an index swap or
/// a missing second handle cannot pass.
const TARGET_RGBA: [u8; 4] = [10, 50, 70, 255];

/// Hand-written stand-in for the converter's per-table argument buffer.
///
/// `TextureDescriptor` is the converter's 24-byte `{texture, texture, i64}`
/// descriptor: the host writes the `gpuResourceID` into both handle slots. The
/// second entry reads `duplicate`, so a host that wrote only the first slot
/// would fail. `SamplerDescriptor` is `{sampler, 0, 0}` and is read by the same
/// fragment through the sampler table.
const SRV_TABLE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct SamplerDescriptor {
    sampler state;
    long reserved0;
    long reserved1;
};

struct TableAbi {
    device TextureDescriptor *textures [[id(0)]];
    device SamplerDescriptor *samplers [[id(1)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut srv_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 srv_fragment(constant TableAbi &abi [[buffer(2)]]) {
    float4 a = abi.textures[0].primary.sample(abi.samplers[0].state, float2(0.5, 0.5));
    float4 b = abi.textures[1].duplicate.sample(abi.samplers[0].state, float2(0.5, 0.5));
    return float4(a.r, b.g, a.b + b.r, 1.0);
}
"#;

fn tool_path(name: &str) -> Option<PathBuf> {
    let output = Command::new("xcrun").args(["-f", name]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path =
            std::env::temp_dir().join(format!("victory-{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Compiles `source` into a real `.metallib`.
///
/// The proof fails loudly when the Metal toolchain is missing: a skipped
/// hardware proof is not evidence.
fn compile_metallib(dir: &Path, stem: &str, source_text: &str) -> Vec<u8> {
    let metal = tool_path("metal").expect("the Metal toolchain (metal) must be installed");
    let metallib = tool_path("metallib").expect("the Metal toolchain (metallib) must be installed");
    let source = dir.join(format!("{stem}.metal"));
    let air = dir.join(format!("{stem}.air"));
    let library = dir.join(format!("{stem}.metallib"));
    fs::write(&source, source_text).expect("write MSL source");

    let compiled = Command::new(metal)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&air)
        .output()
        .expect("spawn metal");
    assert!(
        compiled.status.success(),
        "metal -c failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let linked = Command::new(metallib)
        .arg(&air)
        .arg("-o")
        .arg(&library)
        .output()
        .expect("spawn metallib");
    assert!(
        linked.status.success(),
        "metallib failed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    fs::read(&library).expect("read compiled metallib")
}

fn rgba8_desc(width: u32, height: u32, usage: TextureUsage, label: &str) -> TextureDesc {
    TextureDesc {
        width,
        height,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage,
        label: Some(label.to_string()),
    }
}

/// The reflected ABI: SRV texture table then sampler table, in root-parameter
/// order (`set` is the parameter index, as the guest records it).
fn texture_table_abi() -> PipelineAbi {
    let layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                size: 8,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler),
                size: 8,
                register: 0,
                space: 0,
            },
        ],
    )
    .expect("fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

fn color_attachment(allocation_id: u64, clear: [f64; 4]) -> ColorAttachment {
    ColorAttachment {
        allocation_id,
        generation: 1,
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: clear,
    }
}

fn full_viewport() -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: f64::from(EDGE),
        height: f64::from(EDGE),
        near: 0.0,
        far: 1.0,
    }
}

fn full_scissor() -> ScissorRect {
    ScissorRect {
        x: 0,
        y: 0,
        width: EDGE,
        height: EDGE,
    }
}

/// A page-aligned zero-copy host page.
struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn new(len: usize) -> Self {
        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        }
        let mut raw: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
        let rc = unsafe { posix_memalign(&mut raw, 16 * 1024, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());
        // SAFETY: `raw` is a fresh allocation of `len` writable bytes.
        unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, len) };
        Self { ptr: raw, len }
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) {
        assert!(
            offset + bytes.len() <= self.len,
            "host page write out of bounds"
        );
        // SAFETY: `offset + bytes.len() <= self.len` and `self.ptr` is live.
        unsafe {
            let destination = self.ptr.cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
    }

    fn raw(&self) -> *mut u8 {
        self.ptr.cast()
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut c_void);
        }
        // SAFETY: `self.ptr` came from `posix_memalign` and is no longer
        // registered with the transport when this guard drops.
        unsafe { free(self.ptr) };
    }
}

fn assert_uniform(pixels: &[[u8; 4]], expected: [u8; 4], label: &str) {
    assert_eq!(pixels.len(), PIXELS, "{label} pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, expected, "{label} pixel {index}");
    }
}

fn copy_to_texture(
    allocation_id: u64,
    destination_allocation_id: u64,
    offset: u64,
) -> CommandV3 {
    CommandV3::CopyBufferToTexture(CopyBufferToTexture {
        allocation_id,
        generation: 1,
        destination_allocation_id,
        destination_generation: 1,
        offset,
        bytes_per_row: ROW_PITCH as u32,
        bytes_per_image: 0,
        extent: [TEXTURE_EDGE, TEXTURE_EDGE, 1],
        mip_level: 0,
        slice: 0,
        origin: [0, 0, 0],
    })
}

/// Exact texel proof through a count==2 SRV texture table and a sampler table.
#[test]
fn srv_texture_table_samples_both_entries_exactly() {
    let dir = TempDir::new("srv-texture-table");
    let metallib = compile_metallib(dir.path(), "srv_table", SRV_TABLE_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [
        (TEXTURE0, "texture0"),
        (TEXTURE1, "texture1"),
        (TARGET, "target"),
    ] {
        transport
            .register_registry_texture(
                handle,
                rgba8_desc(
                    if handle == TARGET { EDGE } else { TEXTURE_EDGE },
                    if handle == TARGET { EDGE } else { TEXTURE_EDGE },
                    TextureUsage::color_attachment(),
                    label,
                ),
            )
            .expect("registry texture must create");
    }
    transport
        .register_sampler(SAMPLER, SamplerDesc::nearest_clamp())
        .expect("sampler must register");

    // Two pitched 4x4 images, one page apart. Every texel of an image has the
    // same colour; the pitch gap is poison so a tight-pitch copy is visible.
    let mut upload = HostPage::new(PAGE_BYTES);
    let poison = [0xaau8; PAGE_BYTES];
    upload.write(0, &poison);
    for (slot, color) in [COLOR0, COLOR1].into_iter().enumerate() {
        let image = slot * PAGE_BYTES / 2;
        for row in 0..TEXTURE_EDGE as usize {
            for column in 0..TEXTURE_EDGE as usize {
                upload.write(image + row * ROW_PITCH + column * 4, &color);
            }
        }
    }
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
    let upload_id = unsafe { transport.register_zerocopy(upload.raw(), PAGE_BYTES) }
        .expect("upload page must register");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_TEXTURE1, 1), VmBufferId(TEXTURE1));
    allocations.insert((ALLOC_UPLOAD, 1), upload_id);

    let pipeline_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("srv_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x5000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "srv_vertex",
            &pipeline_state,
            texture_table_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    let texture_table = DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Srv,
        bindings: vec![
            DescriptorTableEntryV3::Texture(TEXTURE0),
            DescriptorTableEntryV3::Texture(TEXTURE1),
        ],
    };
    let sampler_table = DescriptorTableV3 {
        set: 1,
        base_register: 0,
        kind: DescriptorKind::Sampler,
        bindings: vec![DescriptorTableEntryV3::Sampler(SAMPLER)],
    };

    let submission = SubmissionV3 {
        sequence: 301,
        list: None,
        commands: vec![
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE0, 0),
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE1, (PAGE_BYTES / 2) as u64),
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(ALLOC_TARGET, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetDescriptorTable(texture_table),
            CommandV3::SetDescriptorTable(sampler_table),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the texture-table submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "texture-table status");
    assert_eq!(
        stats,
        EncoderStats {
            command_buffers: 1,
            compute_encoders: 0,
            render_encoders: 1,
            blit_encoders: 1,
            barriers: 0,
        },
        "one blit segment uploads both images and one render encoder draws"
    );

    transport
        .wait_for_gpu()
        .expect("the texture-table pass should complete");

    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, TARGET_RGBA, "srv texture table target");

    // Exact texel values per entry, reported for the gate evidence.
    assert_eq!(
        pixels[0],
        [
            COLOR0[0],
            COLOR1[1],
            COLOR0[2] + COLOR1[0],
            255
        ],
        "entry 0 supplies R/B, entry 1 supplies G and the B addend"
    );

    let allocations_in_set = transport
        .residency_allocation_count()
        .expect("residency allocation count");
    assert!(
        allocations_in_set >= 3,
        "residency set must hold both sampled textures and the target (got {allocations_in_set})"
    );

    // --- negative proof: an unregistered texture handle is refused before any
    // command buffer exists. Reuse the same pipeline and tables, but name a
    // handle that was never registered.
    let unknown = SubmissionV3 {
        sequence: 302,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(ALLOC_TARGET, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![
                    DescriptorTableEntryV3::Texture(TEXTURE0),
                    DescriptorTableEntryV3::Texture(UNKNOWN_TEXTURE),
                ],
            }),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 1,
                base_register: 0,
                kind: DescriptorKind::Sampler,
                bindings: vec![DescriptorTableEntryV3::Sampler(SAMPLER)],
            }),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (refused, stats) = transport
        .submit_v3_with_stats(&unknown, &allocations)
        .expect("the negative submission is answered, not errored");
    assert_eq!(
        refused.status,
        ReplyStatus::UnknownHandle,
        "an unregistered texture handle must be UnknownHandle"
    );
    assert_eq!(
        stats,
        EncoderStats::default(),
        "the refusal happens before any command buffer or encoder exists"
    );

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");

    eprintln!(
        "srv texture table proof on {}: texture0={:?}, texture1={:?}, sampled={:?}, \
         unknown_handle={:?}, residency={}",
        info.name_str(),
        COLOR0,
        COLOR1,
        pixels[0],
        refused.status,
        allocations_in_set
    );
}
