//! Hardware proof for ADR-057: a wire descriptor-table `set` resolves to a
//! declared table parameter without any `VRAB` plan.
//!
//! The pipeline's translated root signature declares root constants plus two
//! descriptor tables (map slot order: constants, table 0, table 1). The
//! submission sends `SetRootConstants` at title slot 0 and the two tables at
//! wire sets 0 and 1 with **no** root-argument plan section. The deterministic
//! ordinal rule must resolve set 0 to the first table and set 1 to the second,
//! the pass and draw must encode, and `abi_refusals` must stay 0.
//!
//! The two textures carry distinct colours and the fragment reads both, so a
//! host that resolved both sets to one table (or dropped one) cannot produce the
//! asserted pixels.
//!
//! Run directly; the GPU lock is acquired inside `cargo test`:
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test descriptor_table_plan_gpu -- --test-threads=1 --nocapture
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
    DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterMap, RootParameterSlot,
    RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, EncoderStats, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBufferToTexture, DescriptorKind,
    DescriptorTableEntryV3, DescriptorTableV3, DrawState, LoadOp, PipelineRef, PrimitiveTopology,
    RootConstantsV3, ScissorRect, StoreOp, SubmissionV3, Viewport,
};

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const TEXTURE_EDGE: u32 = 4;
const ROW_PITCH: usize = 256;
const PAGE_BYTES: usize = 16 * 1024;

const TARGET: u64 = 0x2000;
const TEXTURE0: u64 = 0x3000;
const TEXTURE1: u64 = 0x3001;

const ALLOC_TARGET: u64 = 1;
const ALLOC_TEXTURE0: u64 = 2;
const ALLOC_TEXTURE1: u64 = 3;
const ALLOC_UPLOAD: u64 = 4;

const COLOR0: [u8; 4] = [10, 20, 30, 255];
const COLOR1: [u8; 4] = [40, 50, 60, 255];

/// `a.r`, `b.g`, `a.b + b.r`: both table entries contribute, so resolving both
/// sets to one table cannot pass.
const TARGET_RGBA: [u8; 4] = [10, 50, 70, 255];

/// Hand-written stand-in for the converter's per-table argument buffer.
///
/// `TextureDescriptor` is the converter's 24-byte `{texture, texture, i64}`
/// shape: the host writes the `gpuResourceID` into both handle slots. The two
/// tables sit at argument offsets 0 and 8, the root constants at 16, matching
/// [`plan_abi`]'s flattened order.
const PLAN_TABLE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct BindAbi {
    device TextureDescriptor *first [[id(0)]];
    device TextureDescriptor *second [[id(1)]];
    float4 tint [[id(2)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut plan_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 plan_fragment(constant BindAbi &abi [[buffer(2)]]) {
    constexpr sampler smp(filter::nearest, address::clamp_to_edge, coord::normalized);
    float4 a = abi.first[0].primary.sample(smp, float2(0.5, 0.5));
    float4 b = abi.second[0].primary.sample(smp, float2(0.5, 0.5));
    return float4(a.r, b.g, a.b + b.r + abi.tint.b, 1.0);
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

/// Compiles `source` into a real `.metallib`; a missing toolchain fails loudly.
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

/// The translated ABI: reflected argument-buffer order is table 0, table 1,
/// root constants; the authoritative map declares the same tables at slots 1
/// and 2 so the wire's set 0/1 is a table ordinal, not a flattened slot.
fn plan_abi() -> PipelineAbi {
    let layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                size: 8,
                register: 0,
                space: 1,
            },
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
                size: 8,
                register: 1,
                space: 1,
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
    let map = RootParameterMap::from_slots(vec![
        RootParameterSlot::new(0, 0, RootParameterKind::Constants, 16),
        RootParameterSlot::new(
            1,
            0,
            RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
            8,
        ),
        RootParameterSlot::new(
            1,
            1,
            RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
            8,
        ),
    ]);
    PipelineAbi::graphics(None, Some(layout)).with_root_parameters(map)
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

/// Plan-less two-table proof: both wire sets resolve by ordinal and the pass
/// plus draw encode with `abi_refusals` 0.
#[test]
fn a_plan_less_descriptor_table_set_resolves_by_ordinal() {
    let dir = TempDir::new("descriptor-table-plan");
    let metallib = compile_metallib(dir.path(), "plan_table", PLAN_TABLE_MSL);
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
        fragment_entry: Some("plan_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x5000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "plan_vertex",
            &pipeline_state,
            plan_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    let zero_tint: Vec<u8> = [0.0f32; 4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let submission = SubmissionV3 {
        sequence: 401,
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
            // Title root parameter 0 is the constants block; sets 0 and 1 are
            // the table ordinals, not flattened root-parameter indices.
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: zero_tint,
            }),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![DescriptorTableEntryV3::Texture(TEXTURE0)],
            }),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 1,
                base_register: 1,
                kind: DescriptorKind::Srv,
                bindings: vec![DescriptorTableEntryV3::Texture(TEXTURE1)],
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

    // No plan argument: `submit_v3_with_stats` carries the frame exactly as a
    // guest that predates `VRAB` would.
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the plan-less table submission must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok, "status");
    assert_eq!(
        stats,
        EncoderStats {
            command_buffers: 1,
            compute_encoders: 0,
            render_encoders: 1,
            blit_encoders: 1,
            barriers: 0,
        },
        "the upload blit plus one render encoder, no refusal before encoding"
    );
    assert_eq!(
        transport.abi_refusal_count(),
        0,
        "both descriptor-table sets resolved without an ABI refusal"
    );

    transport
        .wait_for_gpu()
        .expect("the table pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_eq!(pixels.len(), PIXELS, "target pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, TARGET_RGBA, "target pixel {index}");
    }

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");

    eprintln!(
        "descriptor-table plan proof on {}: texture0={:?}, texture1={:?}, pixels={:?}, \
         abi_refusals={}",
        info.name_str(),
        COLOR0,
        COLOR1,
        pixels[0],
        transport.abi_refusal_count()
    );
}

// ----- Cycle 18C: the content-key depth state is applied, not invented ------

/// Fullscreen triangle at `z = 0.5` and a constant colour, so a depth test's
/// outcome is a whole-target exact pixel.
const DEPTH_STATE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

vertex VOut depth_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.5, 1.0);
    return out;
}

fragment float4 depth_fragment() {
    return float4(7.0 / 255.0, 11.0 / 255.0, 13.0 / 255.0, 1.0);
}
"#;

const DEPTH_TARGET: u64 = 0x2100;
const DEPTH_PLANE: u64 = 0x2200;
const DEPTH_COLOR_HANDLE: u64 = 0x5300;
const DEPTH_NONE_HANDLE: u64 = 0x5301;
const ALLOC_COLOR: u64 = 0x31;
const ALLOC_DEPTH_PLANE: u64 = 0x32;
const DRAWN_DEPTH_RGBA: [u8; 4] = [7, 11, 13, 255];

fn depth32_desc(label: &str) -> TextureDesc {
    TextureDesc {
        width: EDGE,
        height: EDGE,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::depth_attachment(),
        label: Some(label.to_string()),
    }
}

fn stencil_keep() -> victory_metal::wire_v3::StencilFaceState {
    use victory_metal::wire_v3::{CompareFunction, StencilOperation, StencilFaceState};
    StencilFaceState {
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
        func: CompareFunction::Always,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn wire_depth_state(compare: victory_metal::wire_v3::CompareFunction) -> victory_metal::wire_v3::DepthStencilState {
    victory_metal::wire_v3::DepthStencilState {
        depth_enable: true,
        depth_write: true,
        depth_func: compare,
        stencil_enable: false,
        front: stencil_keep(),
        back: stencil_keep(),
    }
}

fn depth_pass(
    sequence: u32,
    pipeline: u64,
    color_clear: [f64; 4],
    depth_clear: f32,
) -> SubmissionV3 {
    SubmissionV3 {
        sequence,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(ALLOC_COLOR, color_clear)],
                depth_stencil: Some(victory_metal::wire_v3::DepthStencilAttachment {
                    allocation_id: ALLOC_DEPTH_PLANE,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    depth_load: LoadOp::Clear,
                    depth_store: StoreOp::Store,
                    clear_depth: depth_clear,
                    stencil_load: LoadOp::DontCare,
                    stencil_store: StoreOp::DontCare,
                    clear_stencil: 0,
                }),
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: pipeline,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    }
}

/// A content-hashed PSO whose carried depth state is `Greater` must apply that
/// comparison, and a PSO that carries none must leave no depth gate. The old
/// invented `Less` default fails both halves: `z=0.5` against a `1.0` clear
/// would pass (writing over the clear), and against `0.0` would fail (leaving
/// the clear).
#[test]
fn the_content_key_depth_state_applies_and_absent_state_leaves_no_gate() {
    use victory_metal::wire_v3::CompareFunction;

    let dir = TempDir::new("depth-state-wire");
    let metallib = compile_metallib(dir.path(), "depth_state", DEPTH_STATE_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [(DEPTH_TARGET, "color"), (DEPTH_PLANE, "depth")] {
        transport
            .register_registry_texture(
                handle,
                if handle == DEPTH_TARGET {
                    rgba8_desc(EDGE, EDGE, TextureUsage::color_attachment(), label)
                } else {
                    depth32_desc(label)
                },
            )
            .expect("registry texture must create");
    }

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_COLOR, 1), VmBufferId(DEPTH_TARGET));
    allocations.insert((ALLOC_DEPTH_PLANE, 1), VmBufferId(DEPTH_PLANE));

    let greater_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: Some(TextureFormat::Depth32Float),
        fragment_entry: Some("depth_fragment".to_string()),
        depth_stencil: Some(wire_depth_state(CompareFunction::Greater)),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);
    transport
        .register_content_hashed_graphics_pipeline(
            DEPTH_COLOR_HANDLE,
            &metallib,
            "depth_vertex",
            &greater_state,
            PipelineAbi::default(),
        )
        .expect("the Greater pipeline must register");

    let absent_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: Some(TextureFormat::Depth32Float),
        fragment_entry: Some("depth_fragment".to_string()),
        depth_stencil: None,
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);
    transport
        .register_content_hashed_graphics_pipeline(
            DEPTH_NONE_HANDLE,
            &metallib,
            "depth_vertex",
            &absent_state,
            PipelineAbi::default(),
        )
        .expect("the no-depth-state pipeline must register");

    let run = |sequence: u32, pipeline: u64, color: [f64; 4], depth: f32| {
        let (reply, stats) = transport
            .submit_v3_with_stats(&depth_pass(sequence, pipeline, color, depth), &allocations)
            .expect("the depth pass must encode");
        assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok, "status");
        assert_eq!(stats.render_encoders, 1, "one render encoder");
        transport.wait_for_gpu().expect("the depth pass completes");
        transport
            .read_registered_texture_rgba8(DEPTH_TARGET)
            .expect("target readback")
    };

    // Greater, depth cleared to 1.0: 0.5 > 1.0 is false, so the clear stays.
    let pixels = run(701u32, DEPTH_COLOR_HANDLE, [1.0, 0.0, 0.0, 1.0], 1.0);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel,
            [255, 0, 0, 255],
            "Greater must reject z=0.5 against a 1.0 clear (pixel {index})"
        );
    }

    // Greater, depth cleared to 0.0: 0.5 > 0.0 is true, so the draw writes.
    let pixels = run(702u32, DEPTH_COLOR_HANDLE, [0.0, 0.0, 0.0, 1.0], 0.0);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel, DRAWN_DEPTH_RGBA,
            "Greater must accept z=0.5 against a 0.0 clear (pixel {index})"
        );
    }

    // No carried depth state, depth cleared to 0.0: no gate, so an invented
    // Less test would reject and leave the clear; the draw must write.
    let pixels = run(703u32, DEPTH_NONE_HANDLE, [0.0, 0.0, 0.0, 1.0], 0.0);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel, DRAWN_DEPTH_RGBA,
            "no carried depth state must leave no depth gate (pixel {index})"
        );
    }

    eprintln!(
        "content-key depth-state proof on {}: greater_reject=clear, greater_accept={:?}, \
         no_state={:?}",
        info.name_str(),
        DRAWN_DEPTH_RGBA,
        pixels[0]
    );
}
