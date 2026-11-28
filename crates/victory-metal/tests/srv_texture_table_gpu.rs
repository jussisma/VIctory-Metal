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

// ----- Cycle 18B: mixed resource tables -------------------------------------

/// A fragment that reads a texture table partition at `[[id(0)]]` and a buffer
/// view table partition at `[[id(1)]]`. The converter flattens a D3D12 table
/// naming two ranges as two consecutive table parameters, so each partition
/// must land on its own declared slot; a cross-bound partition reads the wrong
/// shape and cannot produce the expected pixel.
const MIXED_RESOURCE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct BufferEntry {
    device const float4 *data;
    long reserved0;
    long reserved1;
};

struct SplitAbi {
    device TextureDescriptor *textures [[id(0)]];
    device BufferEntry *buffers [[id(1)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut split_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 split_fragment(constant SplitAbi &abi [[buffer(2)]]) {
    constexpr sampler smp(filter::nearest, address::clamp_to_edge, coord::normalized);
    float4 tex = abi.textures[0].primary.sample(smp, float2(0.5, 0.5));
    float4 buf = abi.buffers[0].data[0];
    return float4(tex.r, buf.g, tex.b + buf.r, 1.0);
}
"#;

/// The converted layout a two-range D3D12 table produces: a `Texture` table
/// parameter followed by a buffer view table parameter.
fn mixed_resource_abi() -> PipelineAbi {
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
                kind: RootParameterKind::DescriptorTable(
                    DescriptorTableKind::ShaderResourceView,
                ),
                size: 8,
                register: 1,
                space: 0,
            },
        ],
    )
    .expect("mixed-resource fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

/// One mixed wire table binds both partitions at their declared consecutive
/// slots: the draw samples a texture entry and reads a buffer range entry and
/// writes an exact pixel derived from both, and the pass encodes.
#[test]
fn a_mixed_resource_table_binds_both_partitions_exactly() {
    use victory_metal::macos::trace::{
        OUTCOME_SPLIT, descriptor_table_refusal_census, reset_descriptor_table_refusal_census,
    };
    use victory_metal::wire::BufferBinding;

    reset_descriptor_table_refusal_census();
    // The GPU lock is taken by the sibling test in this binary; the two run
    // serially under `--test-threads=1`.
    let dir = TempDir::new("mixed-resource-table");
    let metallib = compile_metallib(dir.path(), "mixed_resource", MIXED_RESOURCE_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [(TEXTURE0, "texture0"), (TARGET, "target")] {
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

    // A 4x4 image of COLOR0 at pitch `ROW_PITCH` and a `float4` buffer range.
    let mut upload = HostPage::new(PAGE_BYTES);
    let poison = [0xaau8; PAGE_BYTES];
    upload.write(0, &poison);
    for row in 0..TEXTURE_EDGE as usize {
        for column in 0..TEXTURE_EDGE as usize {
            upload.write(row * ROW_PITCH + column * 4, &COLOR0);
        }
    }
    let mut buffer_page = HostPage::new(PAGE_BYTES);
    let buffer_values: Vec<u8> = [40.0f32 / 255.0, 50.0 / 255.0, 60.0 / 255.0, 1.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    buffer_page.write(0, &buffer_values);

    // SAFETY: both pages are live, 16 KiB-aligned allocations that outlive the
    // synchronous submission and are unregistered before they drop.
    let upload_id = unsafe { transport.register_zerocopy(upload.raw(), PAGE_BYTES) }
        .expect("upload page must register");
    // SAFETY: same zero-copy contract as the upload page.
    let buffer_id = unsafe { transport.register_zerocopy(buffer_page.raw(), PAGE_BYTES) }
        .expect("buffer page must register");

    const ALLOC_BUFFER: u64 = 0x42;
    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_UPLOAD, 1), upload_id);
    allocations.insert((ALLOC_BUFFER, 1), buffer_id);

    const HANDLE: u64 = 0x5100;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "split_vertex",
            &PipelineState {
                color_formats: vec![TextureFormat::Rgba8Unorm],
                depth_format: None,
                fragment_entry: Some("split_fragment".to_string()),
                ..Default::default()
            }
            .encode(PipelineStage::Graphics),
            mixed_resource_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    // Texture entries first, then the buffer range: the two ranges of one
    // D3D12 descriptor table, flattened as two table parameters.
    let mixed_table = DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Srv,
        bindings: vec![
            DescriptorTableEntryV3::Texture(TEXTURE0),
            DescriptorTableEntryV3::Buffer(BufferBinding {
                allocation_id: ALLOC_BUFFER,
                generation: 1,
                offset: 0,
                length: 16,
            }),
        ],
    };

    let submission = SubmissionV3 {
        sequence: 401,
        list: None,
        commands: vec![
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE0, 0),
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
            CommandV3::SetDescriptorTable(mixed_table),
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
        .expect("a legal mixed resource table must encode");
    assert_eq!(reply.status, ReplyStatus::Ok, "mixed-table status");
    assert_eq!(
        stats,
        EncoderStats {
            command_buffers: 1,
            compute_encoders: 0,
            render_encoders: 1,
            blit_encoders: 1,
            barriers: 0,
        },
        "one blit uploads the image and one render encoder draws"
    );
    assert_eq!(transport.encode_counters().draws, 1, "the draw encodes");
    assert_eq!(
        transport.abi_refusal_count(),
        0,
        "a representable mixed table is not an ABI refusal"
    );
    assert_eq!(
        transport.descriptor_table_mixed_unrepresentable_count(),
        0,
        "two declared slots represent both partitions"
    );
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        0,
        "the table is bound, not inert"
    );
    let census = descriptor_table_refusal_census();
    assert!(
        census.kind_mismatches.is_empty(),
        "both partitions fit their declared kinds: {}",
        census.json_value()
    );
    assert_eq!(census.mix_keys.len(), 1, "{}", census.json_value());
    assert_eq!(census.mix_keys[0].key.mix_class, "buffer+texture");
    assert_eq!(census.mix_keys[0].key.outcome, OUTCOME_SPLIT);

    transport
        .wait_for_gpu()
        .expect("the mixed-table pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    // tex.r = COLOR0.r, buf.g = 50, tex.b + buf.r = 30 + 40.
    assert_uniform(&pixels, [COLOR0[0], 50, COLOR0[2] + 40, 255], "mixed table target");
    assert_eq!(
        pixels[0],
        [10, 50, 70, 255],
        "texture partition supplies R/B, buffer partition supplies G and the B addend"
    );

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");
    transport
        .unregister_buffer(buffer_id)
        .expect("unregister the buffer page");
    eprintln!(
        "mixed resource table proof on {}: target={:?}, draws=1, abi_refusals=0, census={}",
        info.name_str(),
        pixels[0],
        census.json_value()
    );
    reset_descriptor_table_refusal_census();
}

// ----- Cycle 18C: fragment ABI observability and reachability ---------------

/// One fragment table parameter that samples the wire table's first entry.
const SINGLE_SOURCE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct SingleAbi {
    device TextureDescriptor *textures [[id(0)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut single_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 single_fragment(constant SingleAbi &abi [[buffer(2)]]) {
    constexpr sampler smp(filter::nearest, address::clamp_to_edge, coord::normalized);
    return abi.textures[0].primary.sample(smp, float2(0.5, 0.5));
}
"#;

/// A fragment that ignores the wire table, so an inert table cannot abort and
/// the draw's exact colour still proves the pass encoded.
const CONSTANT_FRAGMENT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

vertex VOut constant_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 constant_fragment() {
    return float4(7.0 / 255.0, 11.0 / 255.0, 13.0 / 255.0, 1.0);
}
"#;

/// The translated VS carries no parameter; the PS declares exactly one texture
/// table parameter. A wire table with one texture entry must land on it and the
/// exact sampled pixel proves the fragment source is bound.
#[test]
fn a_single_declared_fragment_texture_table_binds_the_wire_entry_exactly() {
    use victory_metal::macos::trace::{
        FRAGMENT_SOURCE_UNREACHABLE_REASON, reset_abi_binding_telemetry, stage_abi_telemetry,
    };

    reset_abi_binding_telemetry();
    // The GPU lock is taken by the sibling test in this binary; the tests run
    // serially under `--test-threads=1`.
    let dir = TempDir::new("fragment-abi-single-source");
    let metallib = compile_metallib(dir.path(), "single_source", SINGLE_SOURCE_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [(TEXTURE0, "texture0"), (TARGET, "target")] {
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

    let mut upload = HostPage::new(PAGE_BYTES);
    let poison = [0xaau8; PAGE_BYTES];
    upload.write(0, &poison);
    for row in 0..TEXTURE_EDGE as usize {
        for column in 0..TEXTURE_EDGE as usize {
            upload.write(row * ROW_PITCH + column * 4, &COLOR0);
        }
    }
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
    let upload_id = unsafe { transport.register_zerocopy(upload.raw(), PAGE_BYTES) }
        .expect("upload page must register");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_UPLOAD, 1), upload_id);

    let vertex = RootSignatureLayout::from_parameters(ReflectedStage::Vertex, &[])
        .expect("empty vertex layout builds");
    let fragment = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[RootParameterSpec {
            kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
            size: 8,
            register: 0,
            space: 1,
        }],
    )
    .expect("single texture table layout builds");
    let pipeline_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("single_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x5200;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "single_vertex",
            &pipeline_state,
            PipelineAbi::graphics(Some(vertex), Some(fragment)),
        )
        .expect("content-hashed graphics pipeline must register");

    let submission = SubmissionV3 {
        sequence: 501,
        list: None,
        commands: vec![
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE0, 0),
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
                bindings: vec![DescriptorTableEntryV3::Texture(TEXTURE0)],
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
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the single texturable source submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "single-source status");
    assert_eq!(stats.render_encoders, 1, "one render encoder draws");
    assert_eq!(transport.abi_refusal_count(), 0, "the table resolves");
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        0,
        "a declared fragment source is bound, not inert"
    );

    transport
        .wait_for_gpu()
        .expect("the single-source pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, COLOR0, "single declared fragment source");

    let abi = stage_abi_telemetry();
    assert_eq!(abi.stage_mask, 0b11, "vertex and fragment layouts are bound");
    assert_eq!(abi.ps_abi_parameters, 1, "the fragment declares one table");
    assert_eq!(abi.ps_bound_parameters, 1, "the wire table binds it");
    assert_eq!(abi.ps_bound_tables, 1, "the bound parameter is a table");
    assert_eq!(abi.fragment_texture_sources_declared, 1);
    assert_eq!(
        abi.fragment_texture_sources_bound, 1,
        "the fragment texture source is bound"
    );
    assert_eq!(abi.tables_dropped_inert, 0);
    assert_eq!(abi.fragment_source_unreachable, 0);
    assert_ne!(abi.fragment_source_unreachable_reason, FRAGMENT_SOURCE_UNREACHABLE_REASON);

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");
    eprintln!(
        "fragment ABI single-source proof on {}: pixel={:?}, ps_abi={}, ps_bound={}, \
         sources={}/{}",
        info.name_str(),
        pixels[0],
        abi.ps_abi_parameters,
        abi.ps_bound_parameters,
        abi.fragment_texture_sources_bound,
        abi.fragment_texture_sources_declared
    );
}

/// Boot #32's fragment shape: the pipeline declares **no** table parameter, so
/// a wire table carrying a texture source has no root-parameter path. The table
/// stays inertly accepted, the new counter names the missing wire concept, and
/// the draw still encodes its exact colour (no abort).
#[test]
fn a_wire_texture_source_with_no_declared_fragment_parameter_is_inert_and_named() {
    use victory_metal::macos::trace::{
        FRAGMENT_SOURCE_UNREACHABLE_REASON, OUTCOME_INERT, descriptor_table_refusal_census,
        reset_abi_binding_telemetry, reset_descriptor_table_refusal_census, stage_abi_telemetry,
    };

    reset_descriptor_table_refusal_census();
    reset_abi_binding_telemetry();
    let dir = TempDir::new("fragment-abi-unreachable-source");
    let metallib = compile_metallib(dir.path(), "unreachable_source", CONSTANT_FRAGMENT_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [(TEXTURE0, "texture0"), (TARGET, "target")] {
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

    let side = HostPage::new(PAGE_BYTES);
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
    let side_id = unsafe { transport.register_zerocopy(side.raw(), PAGE_BYTES) }
        .expect("side page must register");
    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_UPLOAD, 1), side_id);

    // The Boot #32 shape: one root SRV and no table parameter at all.
    let fragment = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[RootParameterSpec {
            kind: RootParameterKind::ShaderResourceView,
            size: 8,
            register: 0,
            space: 1,
        }],
    )
    .expect("zero-table fragment layout builds");
    assert_eq!(
        PipelineAbi::graphics(None, Some(fragment.clone())).declared_descriptor_table_count(),
        0,
        "the fixture must declare no table"
    );
    let pipeline_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("constant_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x5201;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "constant_vertex",
            &pipeline_state,
            PipelineAbi::graphics(None, Some(fragment)),
        )
        .expect("content-hashed graphics pipeline must register");

    let submission = SubmissionV3 {
        sequence: 502,
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
                bindings: vec![DescriptorTableEntryV3::Texture(TEXTURE0)],
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
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("an unreachable fragment source must not abort the batch");
    assert_eq!(reply.status, ReplyStatus::Ok, "inert-source status");
    assert_eq!(stats.render_encoders, 1, "the draw still encodes");
    assert_eq!(transport.abi_refusal_count(), 0, "inert is not a refusal");
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        1,
        "the table is counted inert"
    );
    assert_eq!(transport.encode_counters().draws, 1, "the draw encodes");

    let census = descriptor_table_refusal_census();
    assert_eq!(census.distinct_keys, 1);
    assert_eq!(census.entries[0].key.outcome, OUTCOME_INERT);
    assert_eq!(census.entries[0].key.stage_mask, 0b10, "fragment layout only");

    let abi = stage_abi_telemetry();
    assert_eq!(abi.ps_abi_parameters, 1, "the fragment declares one root SRV");
    assert_eq!(
        abi.fragment_texture_sources_declared, 0,
        "a root SRV is not a table source"
    );
    assert_eq!(abi.fragment_texture_sources_bound, 0);
    assert_eq!(abi.tables_dropped_inert, 1);
    assert_eq!(abi.ps_tables_dropped_inert, 1);
    assert_eq!(abi.fragment_source_unreachable, 1);
    assert_eq!(
        abi.fragment_source_unreachable_reason,
        FRAGMENT_SOURCE_UNREACHABLE_REASON,
        "the missing wire concept is SetDescriptorHeaps"
    );

    transport
        .wait_for_gpu()
        .expect("the inert-source pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, [7, 11, 13, 255], "inert fragment source");

    transport
        .unregister_buffer(side_id)
        .expect("unregister the side page");
    reset_descriptor_table_refusal_census();
    eprintln!(
        "fragment ABI unreachable-source proof on {}: pixel={:?}, ps_abi={}, \
         sources={}/{}, inert={}, unreachable={}, reason={}",
        info.name_str(),
        pixels[0],
        abi.ps_abi_parameters,
        abi.fragment_texture_sources_bound,
        abi.fragment_texture_sources_declared,
        abi.tables_dropped_inert,
        abi.fragment_source_unreachable,
        abi.fragment_source_unreachable_reason
    );
}
