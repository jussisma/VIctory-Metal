//! Hardware proof for the ADR-026a core D3D12 render states on Apple Silicon.
//!
//! One `#[ignore]`-gated test drives [`DirectMetalTransport::submit_v3_with_stats`]
//! three times on the same two RGBA8 targets plus a `Depth32Float` target:
//!
//! 1. **Blend + explicit mid-pass clears + write mask.** The pass begins with
//!    `DontCare` loads and then issues `ClearRenderTargetView` for both colour
//!    targets and `ClearDepthStencilView` for depth. A legacy pipeline compiled
//!    with an independent per-target blend state draws an opaque red triangle:
//!    target 0 adds red to the blue clear (magenta) and target 1 replaces red
//!    and blue through `Src=One/Dst=Zero` while its write mask keeps green, so
//!    the result is yellow. Without blending, or without the write mask, both
//!    targets would be red.
//! 2. **Depth test + back-face culling + depth bias.** `Less` depth rejects a
//!    farther quad where a nearer quad already wrote depth; with front faces
//!    counter-clockwise and `CullMode::Back` a clockwise quad is removed while
//!    an identical counter-clockwise quad draws; a same-depth draw with a
//!    negative depth bias passes the `Less` test it would otherwise fail.
//! 3. **Pitched upload + pixel-shader sampling.** `CopyBufferToTexture` uploads
//!    a 4x4 RGBA8 image from a 256-byte-pitch source into a private texture,
//!    and a content-hashed pipeline samples texels `(1,1)` and `(2,2)` through
//!    a texture descriptor table. A tight-pitch mistake would read poison bytes
//!    instead of the green and blue rows.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test render_states_gpu -- --ignored --test-threads=1 --nocapture'
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
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::macos::render::{GraphicsPipeline, PixelFormat, RenderTarget};
use victory_metal::wire::{BufferBinding, ListIdentity, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, BlendAttachment, BlendFactor, BlendOperation, BlendState,
    ClearDepthStencilView, ClearRenderTargetView, ColorAttachment, CommandV3, CompareFunction,
    CullMode, DepthStencilAttachment, DepthStencilState, DescriptorKind, DescriptorTableV3,
    DrawState, FillMode, FrontFaceWinding, LoadOp, PipelineRef, PrimitiveTopology, RasterizerState,
    RootConstantsV3, ScissorRect, StencilFaceState, StencilOperation, StoreOp, SubmissionV3,
    VertexBuffer, VertexBufferSet, Viewport,
};

const EDGE: u32 = 64;

const SAMPLED: u64 = 0x1000;

const ALLOC_COLOUR0: u64 = 1;
const ALLOC_COLOUR1: u64 = 2;
const ALLOC_DEPTH: u64 = 3;
const ALLOC_SAMPLED: u64 = 4;
const ALLOC_VERTICES: u64 = 10;
const ALLOC_UPLOAD: u64 = 11;

/// One vertex position in clip space (`float4`).
type Position = [f32; 4];

/// State shader used by the blend and depth passes.
///
/// Positions come from buffer 0 (bound at a per-draw byte offset); the two MRT
/// colours arrive as flat per-draw root constants at buffers 1 and 2, the same
/// binding shape the in-crate v3 MRT test proves on this device.
const STATE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float4 color0 [[flat]];
    float4 color1 [[flat]];
};

vertex VOut state_vertex(uint vid [[vertex_id]],
                         constant float4 *positions [[buffer(0)]],
                         constant float4 &color0 [[buffer(1)]],
                         constant float4 &color1 [[buffer(2)]]) {
    VOut out;
    out.position = positions[vid];
    out.color0 = color0;
    out.color1 = color1;
    return out;
}

struct MrtOut {
    float4 color0 [[color(0)]];
    float4 color1 [[color(1)]];
};

fragment MrtOut state_fragment(VOut in [[stage_in]]) {
    MrtOut out;
    out.color0 = in.color0;
    out.color1 = in.color1;
    return out;
}
"#;

/// Pitched-upload sampling shader: a texture descriptor table at `set 0`.
const TEXTURE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Converted descriptor-table contract: `buffer(2)` holds a pointer to an
// array of 24-byte texture descriptors `{texture, texture, 0}`. The runtime
// writes the sampled texture into both handle slots, so element 1 is read.
struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> sampled;
    uint64_t reserved;
};

struct TexAbi {
    constant TextureDescriptor *descriptors [[id(0)]];
};

struct VOut {
    float4 position [[position]];
};

struct MrtOut {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

vertex VOut texture_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment MrtOut texture_fragment(constant TexAbi &abi [[buffer(2)]]) {
    MrtOut out;
    out.a = abi.descriptors[0].sampled.read(uint2(1, 1));
    out.b = abi.descriptors[0].sampled.read(uint2(2, 2));
    return out;
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
        assert!(offset + bytes.len() <= self.len, "host page write out of bounds");
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
        // SAFETY: `self.ptr` came from `posix_memalign` and the page is
        // unregistered before this guard drops.
        unsafe { free(self.ptr) };
    }
}

/// The two colour targets and the depth target, created through the legacy
/// render-target path the in-crate v3 MRT test proves on this device.
struct Targets {
    colour0: RenderTarget,
    colour1: RenderTarget,
    depth: RenderTarget,
}

fn create_targets(transport: &DirectMetalTransport) -> Targets {
    Targets {
        colour0: transport
            .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
            .expect("colour target 0"),
        colour1: transport
            .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
            .expect("colour target 1"),
        depth: transport
            .create_render_target_2d(EDGE, EDGE, PixelFormat::Depth32Float)
            .expect("depth target"),
    }
}

fn register_targets(
    transport: &DirectMetalTransport,
    targets: &Targets,
) -> HashMap<(u64, u32), VmBufferId> {
    let colour0 = transport
        .register_render_target(&targets.colour0)
        .expect("colour 0 registers");
    let colour1 = transport
        .register_render_target(&targets.colour1)
        .expect("colour 1 registers");
    let depth = transport
        .register_render_target(&targets.depth)
        .expect("depth registers");
    transport
        .register_registry_texture(
            SAMPLED,
            TextureDesc {
                width: 4,
                height: 4,
                depth: 1,
                mip_levels: 1,
                sample_count: 1,
                array_layers: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::sampled(),
                label: Some("pitched-upload".to_string()),
            },
        )
        .expect("sampled texture registers");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_COLOUR0, 1), VmBufferId(colour0.0));
    allocations.insert((ALLOC_COLOUR1, 1), VmBufferId(colour1.0));
    allocations.insert((ALLOC_DEPTH, 1), VmBufferId(depth.0));
    allocations.insert((ALLOC_SAMPLED, 1), VmBufferId(SAMPLED));
    allocations
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

fn scissor(x: u32, y: u32, width: u32, height: u32) -> ScissorRect {
    ScissorRect {
        x,
        y,
        width,
        height,
    }
}

fn colour_attachment(allocation_id: u64, load: LoadOp, clear: [f64; 4]) -> ColorAttachment {
    ColorAttachment {
        allocation_id,
        generation: 1,
        mip_level: 0,
        slice: 0,
        load,
        store: StoreOp::Store,
        clear_color: clear,
    }
}

fn depth_attachment(load: LoadOp) -> DepthStencilAttachment {
    DepthStencilAttachment {
        allocation_id: ALLOC_DEPTH,
        generation: 1,
        mip_level: 0,
        slice: 0,
        depth_load: load,
        depth_store: StoreOp::Store,
        clear_depth: 1.0,
        stencil_load: LoadOp::DontCare,
        stencil_store: StoreOp::DontCare,
        clear_stencil: 0,
    }
}

fn blend_attachment(
    enable: bool,
    src: BlendFactor,
    dst: BlendFactor,
    write_mask: u8,
) -> BlendAttachment {
    BlendAttachment {
        enable,
        src_color: src,
        dst_color: dst,
        color_op: BlendOperation::Add,
        src_alpha: src,
        dst_alpha: dst,
        alpha_op: BlendOperation::Add,
        write_mask,
    }
}

fn disabled_blend() -> BlendState {
    BlendState {
        independent_blend: false,
        alpha_to_coverage: false,
        attachments: vec![blend_attachment(
            false,
            BlendFactor::One,
            BlendFactor::Zero,
            0x0f,
        )],
    }
}

fn stencil_keep() -> StencilFaceState {
    StencilFaceState {
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
        func: CompareFunction::Always,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn depth_state(func: CompareFunction) -> DepthStencilState {
    DepthStencilState {
        depth_enable: true,
        depth_write: true,
        depth_func: func,
        stencil_enable: false,
        front: stencil_keep(),
        back: stencil_keep(),
    }
}

fn raster(cull: CullMode, depth_bias: i32) -> RasterizerState {
    RasterizerState {
        fill: FillMode::Solid,
        cull,
        front_face: FrontFaceWinding::CounterClockwise,
        depth_bias,
        slope_scaled_depth_bias: 0.0,
        depth_bias_clamp: 0.0,
        depth_clip: true,
    }
}

fn list(list_id: u64) -> ListIdentity {
    ListIdentity {
        list_id,
        list_generation: 1,
    }
}

fn pixel(pixels: &[[u8; 4]], x: u32, y: u32) -> [u8; 4] {
    pixels[(y * EDGE + x) as usize]
}

fn assert_pixel(pixels: &[[u8; 4]], x: u32, y: u32, expected: [u8; 4], label: &str) {
    let actual = pixel(pixels, x, y);
    assert_eq!(actual, expected, "{label} pixel ({x},{y})");
}

/// Pushes one axis-aligned NDC quad as two triangles.
fn push_quad(
    positions: &mut Vec<Position>,
    [x0, y0, x1, y1]: [f32; 4],
    z: f32,
    ccw: bool,
) {
    let mut corners = vec![
        [x0, y0, z, 1.0],
        [x1, y0, z, 1.0],
        [x1, y1, z, 1.0],
        [x0, y0, z, 1.0],
        [x1, y1, z, 1.0],
        [x0, y1, z, 1.0],
    ];
    if !ccw {
        // Swap the second and third corner of every triangle to reverse the
        // winding while keeping the same covered area.
        for triangle in corners.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    positions.extend_from_slice(&corners);
}

fn position_bytes(positions: &[Position]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(positions.len() * 16);
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn colour_bytes(colour: [f32; 4]) -> Vec<u8> {
    colour
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn constants(slot: u16, colour: [f32; 4]) -> CommandV3 {
    CommandV3::SetRootConstants(RootConstantsV3 {
        slot,
        bytes: colour_bytes(colour),
    })
}

fn vertex_buffer(offset: u64, length: u64) -> CommandV3 {
    CommandV3::SetVertexBuffers(VertexBufferSet {
        start_slot: 0,
        buffers: vec![VertexBuffer {
            allocation_id: ALLOC_VERTICES,
            generation: 1,
            offset,
            stride: 16,
            length,
        }],
    })
}

const QUAD_BYTES: u64 = 6 * 16;
const TRIANGLE_BYTES: u64 = 3 * 16;

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn render_states_pixels_are_exact() {
    let dir = TempDir::new("render-states");
    let Some(texture_lib) = compile_metallib(dir.path(), "texture", TEXTURE_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; proof skipped");
        return;
    };

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let targets = create_targets(&transport);
    let allocations = register_targets(&transport, &targets);

    // One page holds every position set used below.
    let mut positions = Vec::new();
    // [0..3] blend pass full-screen triangle at z=0.0
    let triangle_start = positions.len();
    positions.extend_from_slice(&[
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ]);
    // [3..9] near-left quad at z=0.5
    let near_start = positions.len();
    push_quad(&mut positions, [-1.0, -1.0, 0.0, 1.0], 0.5, true);
    // [9..15] full quad at z=0.75
    let far_start = positions.len();
    push_quad(&mut positions, [-1.0, -1.0, 1.0, 1.0], 0.75, true);
    // [15..21] clockwise bottom-right quad at z=0.25
    let culled_start = positions.len();
    push_quad(&mut positions, [0.0, -1.0, 1.0, 0.0], 0.25, false);
    // [21..27] counter-clockwise bottom-left quad at z=0.25
    let drawn_start = positions.len();
    push_quad(&mut positions, [-1.0, -1.0, 0.0, 0.0], 0.25, true);
    // [27..33] counter-clockwise top-right quad at z=0.75
    let biased_start = positions.len();
    push_quad(&mut positions, [0.0, 0.0, 1.0, 1.0], 0.75, true);
    let position_bytes = position_bytes(&positions);
    let mut position_page = HostPage::new(16 * 1024);
    position_page.write(0, &position_bytes);
    // SAFETY: the page is 16 KiB aligned and outlives every submission.
    let position_handle = unsafe { transport.register_zerocopy(position_page.raw(), 16 * 1024) }
        .expect("position page must register");
    let mut state_allocations = allocations.clone();
    state_allocations.insert((ALLOC_VERTICES, 1), position_handle);
    let offset = |start: usize| (start * 16) as u64;

    // ---- submission 1: blend, explicit mid-pass clears, write mask --------
    let blend_state = BlendState {
        independent_blend: true,
        alpha_to_coverage: false,
        attachments: vec![
            // target 0: additive red over the blue clear -> magenta.
            blend_attachment(true, BlendFactor::One, BlendFactor::One, 0x0f),
            // target 1: source replaces, but the write mask keeps green.
            blend_attachment(true, BlendFactor::One, BlendFactor::Zero, 0b0101),
        ],
    };
    let blend_pipeline = GraphicsPipeline::compile_mrt_with_state(
        &transport,
        STATE_MSL,
        "state_vertex",
        "state_fragment",
        &[PixelFormat::Rgba8Unorm, PixelFormat::Rgba8Unorm],
        Some(PixelFormat::Depth32Float),
        Some(&blend_state),
        Some(&depth_state(CompareFunction::Less)),
    )
    .expect("blend pipeline should compile");
    let blend_handle = transport
        .register_graphics_pipeline_state(blend_pipeline)
        .expect("blend pipeline should register")
        .0;

    let red = [1.0, 0.0, 0.0, 1.0];
    let blue_clear = [0.0f32, 0.0, 1.0, 1.0];
    let green_clear = [0.0f32, 1.0, 0.0, 1.0];
    let blue_clear_f64 = [0.0f64, 0.0, 1.0, 1.0];
    let green_clear_f64 = [0.0f64, 1.0, 0.0, 1.0];
    let blend_commands = vec![
        // `DontCare` loads model D3D12 render passes that start disengaged; the
        // three explicit clears below are folded into one continuation pass.
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                colour_attachment(ALLOC_COLOUR0, LoadOp::DontCare, blue_clear_f64),
                colour_attachment(ALLOC_COLOUR1, LoadOp::DontCare, green_clear_f64),
            ],
            depth_stencil: Some(depth_attachment(LoadOp::DontCare)),
        }),
        CommandV3::ClearRenderTargetView(ClearRenderTargetView {
            slot: 0,
            rect: None,
            color: blue_clear,
        }),
        CommandV3::ClearRenderTargetView(ClearRenderTargetView {
            slot: 1,
            rect: None,
            color: green_clear,
        }),
        CommandV3::ClearDepthStencilView(ClearDepthStencilView {
            depth: Some(1.0),
            stencil: None,
            rect: None,
        }),
        CommandV3::SetBlendState(blend_state),
        CommandV3::SetDepthStencilState(depth_state(CompareFunction::Less)),
        CommandV3::SetRasterizerState(raster(CullMode::None, 0)),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: blend_handle,
        }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetScissor(scissor(0, 0, EDGE, EDGE)),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        vertex_buffer(offset(triangle_start), TRIANGLE_BYTES),
        constants(1, red),
        constants(2, red),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ];
    let (reply, stats) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence: 201,
                list: Some(list(1)),
                commands: blend_commands,
            },
            &state_allocations,
        )
        .expect("blend submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "blend submission status");
    // Three consecutive explicit clears merge into one continuation sub-pass,
    // so the pass is encoded as the empty original pass plus the continuation.
    let _ = stats;
    transport.wait_for_gpu().expect("blend pass completes");

    let target0 = transport
        .read_render_target_rgba8(&targets.colour0)
        .expect("target 0 readback");
    let target1 = transport
        .read_render_target_rgba8(&targets.colour1)
        .expect("target 1 readback");
    for (index, value) in target0.iter().enumerate() {
        assert_eq!(*value, [255, 0, 255, 255], "target 0 pixel {index}");
    }
    for (index, value) in target1.iter().enumerate() {
        assert_eq!(*value, [255, 255, 0, 255], "target 1 pixel {index}");
    }

    // ---- submission 2: depth test, back-face culling, depth bias ---------
    let depth_pipeline = GraphicsPipeline::compile_mrt_with_state(
        &transport,
        STATE_MSL,
        "state_vertex",
        "state_fragment",
        &[PixelFormat::Rgba8Unorm, PixelFormat::Rgba8Unorm],
        Some(PixelFormat::Depth32Float),
        None,
        Some(&depth_state(CompareFunction::Less)),
    )
    .expect("depth pipeline should compile");
    let depth_handle = transport
        .register_graphics_pipeline_state(depth_pipeline)
        .expect("depth pipeline should register")
        .0;

    let green = [0.0f32, 1.0, 0.0, 1.0];
    let blue = [0.0f32, 0.0, 1.0, 1.0];
    let yellow = [1.0f32, 1.0, 0.0, 1.0];
    let cyan = [0.0f32, 1.0, 1.0, 1.0];
    let draw = |_start: usize| CommandV3::Draw(DrawState {
        vertex_count: 6,
        instance_count: 1,
        start_vertex: 0,
        start_instance: 0,
    });
    let half = EDGE / 2;
    let depth_commands = vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                colour_attachment(ALLOC_COLOUR0, LoadOp::Clear, [0.0, 0.0, 0.0, 1.0]),
                colour_attachment(ALLOC_COLOUR1, LoadOp::Clear, [0.0, 0.0, 0.0, 1.0]),
            ],
            depth_stencil: Some(depth_attachment(LoadOp::DontCare)),
        }),
        CommandV3::ClearDepthStencilView(ClearDepthStencilView {
            depth: Some(1.0),
            stencil: None,
            rect: None,
        }),
        CommandV3::SetBlendState(disabled_blend()),
        CommandV3::SetDepthStencilState(depth_state(CompareFunction::Less)),
        CommandV3::SetRasterizerState(raster(CullMode::Back, 0)),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: depth_handle,
        }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        // near quad: left half
        CommandV3::SetScissor(scissor(0, 0, half, EDGE)),
        vertex_buffer(offset(near_start), QUAD_BYTES),
        constants(1, green),
        constants(2, green),
        draw(near_start),
        // full quad: left half is depth-rejected
        CommandV3::SetScissor(scissor(0, 0, EDGE, EDGE)),
        vertex_buffer(offset(far_start), QUAD_BYTES),
        constants(1, blue),
        constants(2, blue),
        draw(far_start),
        // clockwise quad: bottom-right is culled
        CommandV3::SetScissor(scissor(half, half, half, half)),
        vertex_buffer(offset(culled_start), QUAD_BYTES),
        constants(1, yellow),
        constants(2, yellow),
        draw(culled_start),
        // identical counter-clockwise quad: bottom-left draws
        CommandV3::SetScissor(scissor(0, half, half, half)),
        vertex_buffer(offset(drawn_start), QUAD_BYTES),
        constants(1, yellow),
        constants(2, yellow),
        draw(drawn_start),
        // same-depth quad: top-right passes only with the negative bias
        CommandV3::SetRasterizerState(raster(CullMode::Back, -1)),
        CommandV3::SetScissor(scissor(half, 0, half, half)),
        vertex_buffer(offset(biased_start), QUAD_BYTES),
        constants(1, cyan),
        constants(2, cyan),
        draw(biased_start),
        CommandV3::EndRenderPass,
    ];
    let (reply, _stats) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence: 202,
                list: Some(list(2)),
                commands: depth_commands,
            },
            &state_allocations,
        )
        .expect("depth submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "depth submission status");
    transport.wait_for_gpu().expect("depth pass completes");

    let target0 = transport
        .read_render_target_rgba8(&targets.colour0)
        .expect("target 0 readback");
    let target1 = transport
        .read_render_target_rgba8(&targets.colour1)
        .expect("target 1 readback");
    for pixels in [&target0, &target1] {
        assert_pixel(pixels, 8, 8, [0, 255, 0, 255], "near quad (top-left)");
        assert_pixel(
            pixels,
            EDGE - 8,
            8,
            [0, 255, 255, 255],
            "depth-bias quad (top-right)",
        );
        assert_pixel(
            pixels,
            8,
            EDGE - 8,
            [255, 255, 0, 255],
            "front-face quad (bottom-left)",
        );
        assert_pixel(
            pixels,
            EDGE - 8,
            EDGE - 8,
            [0, 0, 255, 255],
            "culled back face (bottom-right)",
        );
    }

    // ---- submission 3: pitched upload sampled by the pixel shader --------
    // 4x4 RGBA8 with a 256-byte row pitch. Rows: red, green, blue, white at
    // their diagonal texel; every other byte is poison, so a tight-pitch copy
    // would read 0xAA instead of the expected row.
    let mut upload = HostPage::new(16 * 1024);
    let poison = [0xaau8; 16 * 1024];
    upload.write(0, &poison);
    let row_pitch = 256usize;
    for (row, column, rgba) in [
        (0usize, 0usize, [255u8, 0, 0, 255]),
        (1, 1, [0, 255, 0, 255]),
        (2, 2, [0, 0, 255, 255]),
        (3, 3, [255, 255, 255, 255]),
    ] {
        let offset = row * row_pitch + column * 4;
        upload.write(offset, &rgba);
    }
    // SAFETY: the page is 16 KiB aligned and outlives the submission.
    let upload_handle = unsafe { transport.register_zerocopy(upload.raw(), 16 * 1024) }
        .expect("upload page must register");
    let mut upload_allocations = allocations.clone();
    upload_allocations.insert((ALLOC_UPLOAD, 1), upload_handle);

    let texture_layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[RootParameterSpec {
            kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Texture),
            size: 0,
            register: 0,
            space: 0,
        }],
    )
    .expect("texture root signature must build")
    .with_descriptor_table_kinds(&[DescriptorTableKind::Texture])
    .expect("texture table kind must attach");
    let texture_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("texture_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);
    const TEXTURE_HANDLE: u64 = 0x7000;
    transport
        .register_content_hashed_graphics_pipeline(
            TEXTURE_HANDLE,
            &texture_lib,
            "texture_vertex",
            &texture_state,
            PipelineAbi::graphics(None, Some(texture_layout)),
        )
        .expect("textured content-hashed pipeline must register");

    let upload_commands = vec![
        CommandV3::CopyBufferToTexture(victory_metal::wire_v3::CopyBufferToTexture {
            allocation_id: ALLOC_UPLOAD,
            generation: 1,
            destination_allocation_id: ALLOC_SAMPLED,
            destination_generation: 1,
            offset: 0,
            bytes_per_row: row_pitch as u32,
            bytes_per_image: 0,
            extent: [4, 4, 1],
            mip_level: 0,
            slice: 0,
            origin: [0, 0, 0],
        }),
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                colour_attachment(ALLOC_COLOUR0, LoadOp::Clear, [0.0, 0.0, 0.0, 1.0]),
                colour_attachment(ALLOC_COLOUR1, LoadOp::Clear, [0.0, 0.0, 0.0, 1.0]),
            ],
            depth_stencil: None,
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: TEXTURE_HANDLE,
        }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetScissor(scissor(0, 0, EDGE, EDGE)),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![BufferBinding {
                allocation_id: ALLOC_SAMPLED,
                generation: 1,
                offset: 0,
                length: 16,
            }],
        }),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ];
    let (reply, stats) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence: 203,
                list: Some(list(3)),
                commands: upload_commands,
            },
            &upload_allocations,
        )
        .expect("pitched-upload submission must execute");
    assert_eq!(
        reply.status,
        ReplyStatus::Ok,
        "pitched-upload submission status"
    );
    assert_eq!(stats.blit_encoders, 1, "one blit segment copies the upload");
    assert_eq!(stats.render_encoders, 1, "one render pass samples the texture");
    transport.wait_for_gpu().expect("pitched-upload pass completes");

    let target0 = transport
        .read_render_target_rgba8(&targets.colour0)
        .expect("target 0 readback");
    let target1 = transport
        .read_render_target_rgba8(&targets.colour1)
        .expect("target 1 readback");
    for (index, value) in target0.iter().enumerate() {
        assert_eq!(*value, [0, 255, 0, 255], "sampled (1,1) pixel {index}");
    }
    for (index, value) in target1.iter().enumerate() {
        assert_eq!(*value, [0, 0, 255, 255], "sampled (2,2) pixel {index}");
    }

    transport
        .unregister_buffer(position_handle)
        .expect("unregister the position page");
    transport
        .unregister_buffer(upload_handle)
        .expect("unregister the upload page");

    eprintln!(
        "render-state proof on {}: blend=magenta/yellow, depth=green/cyan/yellow/blue, \
         upload=green/blue, encoders={stats:?}",
        info.name_str()
    );
}
