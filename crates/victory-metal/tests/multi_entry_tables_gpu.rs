//! Hardware proof for ADR-023 multi-entry CBV/SRV/UAV descriptor tables.
//!
//! One `submit_v3_with_stats` submission binds, through the reflected
//! argument-buffer ABI:
//!
//! * a table of 4 CBVs read by index,
//! * a table of 4 SRV textures (4 distinct RGBA8 clears) sampled by index,
//! * a table of 4 UAV buffers written by index,
//! * a root-constants block alongside the three tables.
//!
//! The fragment stage sums every CBV and texture and writes index-derived
//! integers through the UAV table. The test asserts exact target pixels and
//! exact UAV words, then checks the host builder's resolved table extents
//! (count and stride, hence `count * stride`) for each shape.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test multi_entry_tables_gpu -- --ignored --test-threads=1 --nocapture'
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
    DescriptorTableBuilder, DescriptorTableKind, ReflectedStage, RootParameterKind,
    RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, EncoderStats, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::wire::BufferBinding;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DescriptorKind, DescriptorTableV3, DrawState,
    LoadOp, PipelineRef, PrimitiveTopology, RootConstantsV3, ScissorRect, StoreOp, SubmissionV3,
    Viewport,
};

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const TEXTURE_EDGE: u32 = 4;
const TABLE_ENTRIES: usize = 4;

const TARGET: u64 = 0x2000;
const TEXTURES: [u64; TABLE_ENTRIES] = [0x2100, 0x2101, 0x2102, 0x2103];

/// Buffer indices for the CBV and UAV pages (`u32` words at 16-byte entries).
const CBV_BASES: [f32; TABLE_ENTRIES] = [0.0, 1.0, 2.0, 3.0];

/// Expected exact RGBA8 target after summing every CBV and every texture.
const TARGET_RGBA: [u8; 4] = [100, 200, 255, 255];

/// Expected exact UAV words: `(cbv.r, tex.r, cbv.g, tex.g) * 255`, rounded.
const UAV_WORDS: [[u32; 4]; TABLE_ENTRIES] = [
    [0, 4, 0, 8],
    [10, 8, 20, 16],
    [20, 12, 40, 24],
    [30, 16, 60, 32],
];

/// Multi-entry table shader: three descriptor arrays and inline root constants
/// in the converter's top-level argument buffer (`buffer(2)`).
///
/// `TextureEntry` is the converter's 24-byte `{texture, texture, i64}` shape:
/// the host writes the `gpuResourceID` into both pointer slots and the shader
/// samples the first. Address tables are arrays of 8-byte device addresses, one
/// per entry, so each element is a one-pointer struct.
const MULTI_ENTRY_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureEntry {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long id;
};

struct BufferEntry {
    device const float4 *data;
};

struct UavEntry {
    device float4 *data;
};

struct BindAbi {
    device TextureEntry *textures [[id(0)]];
    device BufferEntry *cbvs [[id(1)]];
    device UavEntry *uavs [[id(2)]];
    float4 tint [[id(3)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut multi_entry_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 multi_entry_fragment(constant BindAbi &abi [[buffer(2)]],
                                     float4 position [[position]]) {
    constexpr sampler smp(filter::nearest, address::clamp_to_edge, coord::normalized);
    float4 target = abi.tint;
    for (uint i = 0; i < 4; ++i) {
        float4 cbv = abi.cbvs[i].data[0];
        float4 tex = abi.textures[i].primary.sample(smp, float2(0.5, 0.5));
        target += cbv + tex;
        abi.uavs[i].data[0] = round(float4(cbv.r, tex.r, cbv.g, tex.g) * 255.0);
    }
    return target;
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

/// Compiles `source` into a real `.metallib`, or `None` without Xcode tools.
fn compile_metallib(dir: &Path, stem: &str, source_text: &str) -> Option<Vec<u8>> {
    let metal = tool_path("metal")?;
    let metallib = tool_path("metallib")?;
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
    Some(fs::read(&library).expect("read compiled metallib"))
}

fn rgba8_desc(width: u32, height: u32, usage: TextureUsage) -> TextureDesc {
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
        label: Some("multi-entry-tables".to_string()),
    }
}

/// The reflected ABI: three tables first (their ordinal and parameter index
/// coincide, as the guest records `set` = root parameter slot), then root
/// constants.
fn multi_entry_abi() -> PipelineAbi {
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
                    DescriptorTableKind::ConstantBufferView,
                ),
                size: 8,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(
                    DescriptorTableKind::UnorderedAccessView,
                ),
                size: 8,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: 16,
                register: 0,
                space: 0,
            },
        ],
    )
    .expect("fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

fn float4_bytes(values: [f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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

    /// The `f32` word at `offset`, after the GPU wrote it.
    fn float(&self, offset: usize) -> f32 {
        assert!(offset + 4 <= self.len, "host page read out of bounds");
        // SAFETY: the offset is inside the allocation and 4-byte aligned.
        unsafe { self.ptr.cast::<u8>().add(offset).cast::<f32>().read_unaligned() }
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

/// Registers the four SRV textures and the RGBA8 target.
fn register_textures(transport: &DirectMetalTransport) {
    for handle in TEXTURES {
        transport
            .register_registry_texture(
                handle,
                rgba8_desc(TEXTURE_EDGE, TEXTURE_EDGE, TextureUsage::color_attachment()),
            )
            .expect("SRV texture must create");
    }
    transport
        .register_registry_texture(TARGET, rgba8_desc(EDGE, EDGE, TextureUsage::color_attachment()))
        .expect("target texture must create");
}

/// The host builder produces `count * stride` for each table shape used here.
fn assert_resolved_extents() {
    let mut textures = DescriptorTableBuilder::new(DescriptorTableKind::Texture);
    for index in 0..TABLE_ENTRIES {
        textures
            .push_texture(0x1000 + index as u64 * 8)
            .expect("texture descriptor");
    }
    let textures = textures.binding(0x8000).expect("texture table binding");
    assert_eq!(
        (textures.count, textures.stride),
        (TABLE_ENTRIES as u32, 24),
        "texture table count/stride"
    );
    assert_eq!(
        textures.extent().expect("texture table extent"),
        TABLE_ENTRIES as u64 * 24,
    );

    for kind in [
        DescriptorTableKind::ConstantBufferView,
        DescriptorTableKind::ShaderResourceView,
        DescriptorTableKind::UnorderedAccessView,
    ] {
        let mut addresses = DescriptorTableBuilder::new(kind);
        for index in 0..TABLE_ENTRIES {
            addresses
                .push_address(0x1000 + index as u64 * 16)
                .expect("address descriptor");
        }
        let addresses = addresses.binding(0x8000).expect("address table binding");
        assert_eq!(
            (addresses.count, addresses.stride),
            (TABLE_ENTRIES as u32, 8),
            "address table count/stride"
        );
        assert_eq!(
            addresses.extent().expect("address table extent"),
            TABLE_ENTRIES as u64 * 8,
        );
    }
}

/// Exact-value proof: 4 CBVs + 4 SRV textures + 4 UAVs through three
/// multi-entry descriptor tables and a root-constants block, one submission.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn multi_entry_cbv_srv_uav_tables_are_exact() {
    let dir = TempDir::new("multi-entry-tables");
    let Some(metallib) = compile_metallib(dir.path(), "multi_entry", MULTI_ENTRY_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    register_textures(&transport);
    assert_resolved_extents();

    // CBV page: four distinct float4 values at 16-byte offsets.
    let mut cbv_page = HostPage::new(16 * 1024);
    for (index, base) in CBV_BASES.iter().enumerate() {
        // Multiples of 1/255 so `round(value * 255)` is exact in the shader.
        cbv_page.write(
            index * 16,
            &float4_bytes([10.0 * base / 255.0, 20.0 * base / 255.0, 30.0 * base / 255.0, 0.25]),
        );
    }
    // UAV page: four zeroed float4 slots the fragment stage overwrites.
    let uav_page = HostPage::new(16 * 1024);

    // SAFETY: both pages are live, page-aligned allocations that outlive the
    // synchronous submission and are unregistered before they drop.
    let cbv_id = unsafe { transport.register_zerocopy(cbv_page.raw(), 16 * 1024) }
        .expect("CBV page must register");
    // SAFETY: same zero-copy contract as the CBV page.
    let uav_id = unsafe { transport.register_zerocopy(uav_page.raw(), 16 * 1024) }
        .expect("UAV page must register");

    let mut allocations = HashMap::new();
    for (index, handle) in TEXTURES.iter().enumerate() {
        allocations.insert((index as u64 + 1, 1u32), VmBufferId(*handle));
    }
    allocations.insert((5, 1), VmBufferId(TARGET));
    allocations.insert((6, 1), cbv_id);
    allocations.insert((7, 1), uav_id);

    // Clear the four sampled textures to distinct RGBA8 values. A render pass
    // with four attachments clears them all in one command buffer; the proof
    // submission samples the same registry textures afterwards.
    let clears = vec![
        color_attachment(1, [4.0 / 255.0, 8.0 / 255.0, 12.0 / 255.0, 1.0]),
        color_attachment(2, [8.0 / 255.0, 16.0 / 255.0, 24.0 / 255.0, 1.0]),
        color_attachment(3, [12.0 / 255.0, 24.0 / 255.0, 36.0 / 255.0, 1.0]),
        color_attachment(4, [16.0 / 255.0, 32.0 / 255.0, 48.0 / 255.0, 1.0]),
    ];
    let clear_submission = SubmissionV3 {
        sequence: 201,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: clears,
                depth_stencil: None,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, _) = transport
        .submit_v3_with_stats(&clear_submission, &allocations)
        .expect("the clear pass must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    transport.wait_for_gpu().expect("the clear pass should complete");

    let pipeline_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("multi_entry_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x4000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "multi_entry_vertex",
            &pipeline_state,
            multi_entry_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    let texture_bindings: Vec<BufferBinding> = (1..=4)
        .map(|allocation_id| BufferBinding {
            allocation_id,
            generation: 1,
            offset: 0,
            length: 16,
        })
        .collect();
    let cbv_bindings: Vec<BufferBinding> = (0..TABLE_ENTRIES)
        .map(|index| BufferBinding {
            allocation_id: 6,
            generation: 1,
            offset: index as u64 * 16,
            length: 16,
        })
        .collect();
    let uav_bindings: Vec<BufferBinding> = (0..TABLE_ENTRIES)
        .map(|index| BufferBinding {
            allocation_id: 7,
            generation: 1,
            offset: index as u64 * 16,
            length: 16,
        })
        .collect();

    let submission = SubmissionV3 {
        sequence: 202,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(5, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            // Parameter 3: the root-constants block alongside the tables.
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 3,
                bytes: float4_bytes([0.0, 0.0, 0.0, 0.0]),
            }),
            // Parameter 0: 4 SRV textures sampled by index.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: texture_bindings,
            }),
            // Parameter 1: 4 CBVs read by index.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 1,
                base_register: 1,
                kind: DescriptorKind::Cbv,
                bindings: cbv_bindings,
            }),
            // Parameter 2: 4 UAV buffers written by index.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 2,
                base_register: 2,
                kind: DescriptorKind::Uav,
                bindings: uav_bindings,
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
        .expect("the multi-entry-table submission must execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "multi-entry submission status"
    );
    assert_eq!(
        stats,
        EncoderStats {
            command_buffers: 1,
            compute_encoders: 0,
            render_encoders: 1,
            blit_encoders: 0,
            barriers: 0,
        },
        "one render encoder must cover the pass"
    );

    transport
        .wait_for_gpu()
        .expect("the multi-entry-table pass should complete");

    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, TARGET_RGBA, "multi-entry target");

    for (index, expected) in UAV_WORDS.iter().enumerate() {
        for (word, expected) in expected.iter().enumerate() {
            let actual = uav_page.float(index * 16 + word * 4);
            assert_eq!(
                actual, *expected as f32,
                "UAV {index} word {word} (expected {expected})"
            );
        }
    }

    let native = transport
        .residency_is_native()
        .expect("residency state is readable");
    let allocations_in_set = transport
        .residency_allocation_count()
        .expect("residency allocation count");
    let fallback = transport
        .residency_fallback_reason()
        .expect("residency fallback reason is readable");
    assert!(
        native || fallback.is_some(),
        "a non-native residency path must record its fallback reason"
    );
    assert!(
        allocations_in_set >= 7,
        "residency set must hold the four textures, target, CBV and UAV pages (got {allocations_in_set})"
    );

    transport
        .unregister_buffer(cbv_id)
        .expect("unregister the CBV page");
    transport
        .unregister_buffer(uav_id)
        .expect("unregister the UAV page");

    eprintln!(
        "multi-entry table proof on {}: 4 CBVs + 4 SRV textures + 4 UAVs, \
         target={:?}, uav[1]={:?}, native_residency={}, allocations={}, fallback={:?}",
        info.name_str(),
        pixels[0],
        [
            uav_page.float(16),
            uav_page.float(20),
            uav_page.float(24),
            uav_page.float(28),
        ],
        native,
        allocations_in_set,
        fallback
    );
}
