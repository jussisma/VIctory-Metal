//! Hardware proof that the D3D12 rasterizer state the title's converted
//! pipelines carry actually keeps a fullscreen triangle and culls a
//! deliberately back-facing one.
//!
//! Boot #36 encoded 154 three-vertex 4K draws and every render target stayed
//! uniformly zero. `state_trace` reported `rasterizer_applied=0` and
//! `cull=unset winding=unset`, so no cull/winding decision was observable. The
//! probes here pin down what the mapping does when it is applied:
//!
//! 1. **No cull** keeps the title's fullscreen triangle
//!    `(-1,-1) (3,-1) (-1,3)`, which covers every pixel. This control proves the
//!    geometry, viewport and fragment output write exact RGBA8 bytes; a zero
//!    target cannot be blamed on coverage.
//! 2. **Cull back, front faces clockwise** — the D3D12 default, and the state a
//!    converted `VPS2` pipeline carries when the title supplies no rasterizer
//!    desc — culls that same triangle (it is counter-clockwise in framebuffer
//!    space) and keeps its reversed (clockwise) twin.
//! 3. **Cull back, front faces counter-clockwise** keeps the original triangle,
//!    proving `FrontCounterClockwise` flips the winding decision and nothing
//!    else.
//!
//! The pipeline is registered through the guest path (`PipelineState::encode`
//! with `rasterizer: Some(..)` -> content-hashed `MTLRenderPipelineState`). The
//! same values are also sent as a wire `SetRasterizerState`, because that is
//! the only path the host applies today: the device render path decodes the
//! pipeline's rasterizer into its content key but does not yet push it to the
//! encoder. The test therefore proves the *state*, not that missing wiring; the
//! state trace's `cull=.../wire` lines name the source that took effect.
//!
//! Run directly (the binary owns `/tmp/victory-gpu.lock` through the shared GPU
//! harness) with `--test-threads=1`:
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test rasterizer_cover_gpu -- --test-threads=1 --nocapture
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
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{PipelineStage, PipelineState, TextureFormat};
use victory_metal::macos::render::{state_trace, PixelFormat};
use victory_metal::wire::{ListIdentity, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CullMode, DrawState, FillMode,
    FrontFaceWinding, LoadOp, PipelineRef, PrimitiveTopology, RasterizerState, ScissorRect,
    StoreOp, SubmissionV3, VertexBuffer, VertexBufferSet, Viewport,
};

/// Attachment edge in texels. Coverage is what matters, not the 4K extent.
const EDGE: u32 = 64;

/// Allocation identity/generation of the colour attachment.
const ALLOC_COLOUR: u64 = 1;
/// Allocation identity/generation of the vertex page.
const ALLOC_VERTICES: u64 = 10;

/// Exact fragment bytes the proof expects: `32/255`, `64/255`, `128/255`, `1`.
const EXACT: [u8; 4] = [32, 64, 128, 255];

/// Converted-pipeline shader: clip-space positions from vertex buffer 0 and a
/// constant fragment colour, so only rasterization can remove the triangle.
const COVER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

struct CoverOut {
    float4 color [[color(0)]];
};

vertex VOut cover_vertex(uint vid [[vertex_id]],
                         constant float4 *positions [[buffer(0)]]) {
    VOut out;
    out.position = positions[vid];
    return out;
}

fragment CoverOut cover_fragment() {
    CoverOut out;
    out.color = float4(32.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 1.0);
    return out;
}
"#;

/// The title's fullscreen triangle: counter-clockwise in framebuffer space, so
/// it is *back-facing* when clockwise is the front face.
const TRIANGLE_CCW: [[f32; 4]; 3] = [
    [-1.0, -1.0, 0.0, 1.0],
    [3.0, -1.0, 0.0, 1.0],
    [-1.0, 3.0, 0.0, 1.0],
];

/// The same coverage with the first two vertices swapped: clockwise in
/// framebuffer space, so it is front-facing under clockwise.
const TRIANGLE_CW: [[f32; 4]; 3] = [
    [3.0, -1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, 3.0, 0.0, 1.0],
];

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

/// A 16 KiB-aligned page whose bytes are visible to Metal zero-copy.
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
        // SAFETY: `self.ptr` came from `posix_memalign`.
        unsafe { free(self.ptr) };
    }
}

fn position_bytes(positions: &[[f32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(positions.len() * 16);
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The rasterizer state a converted pipeline carries for these probes.
fn raster(cull: CullMode, front_face: FrontFaceWinding) -> RasterizerState {
    RasterizerState {
        fill: FillMode::Solid,
        cull,
        front_face,
        depth_bias: 0,
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

/// Registers one content-hashed pipeline whose `VPS2` blob carries `raster`.
fn register_pipeline(
    transport: &DirectMetalTransport,
    library: &[u8],
    handle: u64,
    raster_state: RasterizerState,
) {
    let state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("cover_fragment".to_string()),
        rasterizer: Some(raster_state),
        ..PipelineState::default()
    }
    .encode(PipelineStage::Graphics);
    transport
        .register_content_hashed_graphics_pipeline(
            handle,
            library,
            "cover_vertex",
            &state,
            PipelineAbi::default(),
        )
        .expect("the converted pipeline registers");
}

/// One pass: clear to transparent black, optionally apply `raster_state` over
/// the wire, and draw the three vertices at `vertex_offset`.
#[allow(clippy::too_many_arguments)]
fn draw_triangle(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    sequence: u32,
    list_id: u64,
    pipeline: u64,
    raster_state: Option<RasterizerState>,
    vertex_offset: u64,
) -> ReplyStatus {
    let mut commands = Vec::with_capacity(10);
    commands.push(CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![ColorAttachment {
            allocation_id: ALLOC_COLOUR,
            generation: 1,
            mip_level: 0,
            slice: 0,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        }],
        depth_stencil: None,
    }));
    commands.push(CommandV3::SetGraphicsPipeline(PipelineRef {
        pipeline_id: pipeline,
    }));
    if let Some(raster_state) = raster_state {
        commands.push(CommandV3::SetRasterizerState(raster_state));
    }
    commands.extend_from_slice(&[
        CommandV3::SetViewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(EDGE),
            height: f64::from(EDGE),
            near: 0.0,
            far: 1.0,
        }),
        CommandV3::SetScissor(ScissorRect {
            x: 0,
            y: 0,
            width: EDGE,
            height: EDGE,
        }),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: vec![VertexBuffer {
                allocation_id: ALLOC_VERTICES,
                generation: 1,
                offset: vertex_offset,
                stride: 16,
                length: 3 * 16,
            }],
        }),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ]);
    let (reply, _) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence,
                list: Some(list(list_id)),
                commands,
            },
            allocations,
        )
        .expect("the rasterizer probe submission executes");
    reply.status
}

fn assert_uniform(pixels: &[[u8; 4]], expected: [u8; 4], label: &str) {
    assert_eq!(pixels.len(), (EDGE * EDGE) as usize, "{label}: readback size");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, expected, "{label}: pixel {index}");
    }
}

/// The state the title's converted pipeline carries (D3D12 default): cull back,
/// clockwise front faces. The fullscreen triangle is back-facing under it and
/// must disappear; the reversed triangle must write exact bytes.
#[test]
fn converted_pipeline_culls_the_back_facing_fullscreen_triangle() {
    let dir = TempDir::new("rasterizer-cull");
    let Some(library) = compile_metallib(dir.path(), "rasterizer_cover", COVER_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; proof skipped");
        return;
    };
    let transport = match DirectMetalTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("PARTIAL: no Metal device ({error}); proof skipped");
            return;
        }
    };
    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("colour target");
    let handle = transport
        .register_render_target(&target)
        .expect("colour target registers");
    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_COLOUR, 1), VmBufferId(handle.0));
    let mut vertices = position_bytes(&TRIANGLE_CCW);
    vertices.extend_from_slice(&position_bytes(&TRIANGLE_CW));
    let mut page = HostPage::new(16 * 1024);
    page.write(0, &vertices);
    // SAFETY: the page is 16 KiB aligned and outlives every submission.
    let vertex_handle = unsafe { transport.register_zerocopy(page.raw(), 16 * 1024) }
        .expect("vertex page registers");
    allocations.insert((ALLOC_VERTICES, 1), vertex_handle);

    const BACK_CW_HANDLE: u64 = 0x7200;
    register_pipeline(
        &transport,
        &library,
        BACK_CW_HANDLE,
        raster(CullMode::Back, FrontFaceWinding::Clockwise),
    );
    const BACK_CCW_HANDLE: u64 = 0x7201;
    register_pipeline(
        &transport,
        &library,
        BACK_CCW_HANDLE,
        raster(CullMode::Back, FrontFaceWinding::CounterClockwise),
    );

    // Case 1 (control): cull nothing; the triangle writes every pixel. This is
    // what Boot #36 actually ran with, and it proves coverage/shader output.
    let status = draw_triangle(
        &transport,
        &allocations,
        401,
        71,
        BACK_CW_HANDLE,
        Some(raster(CullMode::None, FrontFaceWinding::Clockwise)),
        0,
    );
    assert_eq!(status, ReplyStatus::Ok, "cull-none control");
    transport.wait_for_gpu().expect("cull-none completes");
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("cull-none readback");
    assert_uniform(&pixels, EXACT, "cull none keeps the fullscreen triangle");

    // Case 2: the converted pipeline's state (back/cw) culls it.
    let status = draw_triangle(
        &transport,
        &allocations,
        402,
        72,
        BACK_CW_HANDLE,
        Some(raster(CullMode::Back, FrontFaceWinding::Clockwise)),
        0,
    );
    assert_eq!(status, ReplyStatus::Ok, "cull-back submission");
    transport.wait_for_gpu().expect("cull-back completes");
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("cull-back readback");
    assert_uniform(
        &pixels,
        [0, 0, 0, 0],
        "back/cw culls the counter-clockwise fullscreen triangle",
    );

    // Case 3: the same state keeps the clockwise twin and writes exact bytes.
    let status = draw_triangle(
        &transport,
        &allocations,
        403,
        73,
        BACK_CW_HANDLE,
        Some(raster(CullMode::Back, FrontFaceWinding::Clockwise)),
        48,
    );
    assert_eq!(status, ReplyStatus::Ok, "front-facing submission");
    transport.wait_for_gpu().expect("front-facing completes");
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("front-facing readback");
    assert_uniform(
        &pixels,
        EXACT,
        "back/cw keeps the clockwise fullscreen triangle",
    );

    // Case 4: FrontCounterClockwise=true flips the decision, nothing else.
    let status = draw_triangle(
        &transport,
        &allocations,
        404,
        74,
        BACK_CCW_HANDLE,
        Some(raster(CullMode::Back, FrontFaceWinding::CounterClockwise)),
        0,
    );
    assert_eq!(status, ReplyStatus::Ok, "counter-clockwise winding submission");
    transport.wait_for_gpu().expect("counter-clockwise completes");
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("counter-clockwise readback");
    assert_uniform(
        &pixels,
        EXACT,
        "back/ccw keeps the counter-clockwise fullscreen triangle",
    );

    // Case 5 (characterises the current host gap): the same pipeline carries
    // back/cw in its `VPS2` blob, but with no wire `SetRasterizerState` the
    // counter-clockwise triangle still writes every pixel. The device render
    // path does not push the pipeline's carried rasterizer to the encoder, so
    // the carried state is inert. This case is the direct evidence for Boot
    // #36's `rasterizer_applied=0`; it must be inverted (all-zero expected) once
    // the pipeline path applies `PipelineState::rasterizer`.
    let status = draw_triangle(&transport, &allocations, 405, 75, BACK_CW_HANDLE, None, 0);
    assert_eq!(status, ReplyStatus::Ok, "no wire rasterizer submission");
    transport.wait_for_gpu().expect("no wire rasterizer completes");
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("no wire rasterizer readback");
    assert_uniform(
        &pixels,
        EXACT,
        "without a wire state the pipeline's carried back/cw is not applied",
    );

    eprintln!(
        "rasterizer cover proof on {}: cull-none exact, back/cw culled, \
         back/cw and back/ccw exact, carried pipeline state inert without wire",
        transport
            .device_info()
            .map_or_else(|_| "Metal device".to_string(), |info| info.name_str())
    );
}

/// The state trace now records the cull/winding decision of every encoded draw
/// and names the source that decided it: `back/wire` after a wire state, never
/// `unset` for a drawn pass.
#[test]
fn state_trace_reports_the_applied_cull_and_winding_with_their_source() {
    let dir = TempDir::new("rasterizer-trace");
    let Some(library) = compile_metallib(dir.path(), "rasterizer_cover", COVER_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; trace skipped");
        return;
    };
    // SAFETY: this test binary runs one test at a time; the flag is read once.
    unsafe {
        std::env::set_var(state_trace::STATE_TRACE_ENV, "1");
    }
    state_trace::reset();
    let transport = match DirectMetalTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("PARTIAL: no Metal device ({error}); trace skipped");
            return;
        }
    };
    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("colour target");
    let handle = transport
        .register_render_target(&target)
        .expect("colour target registers");
    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_COLOUR, 1), VmBufferId(handle.0));
    let vertices = position_bytes(&TRIANGLE_CW);
    let mut page = HostPage::new(16 * 1024);
    page.write(0, &vertices);
    // SAFETY: the page is 16 KiB aligned and outlives every submission.
    let vertex_handle = unsafe { transport.register_zerocopy(page.raw(), 16 * 1024) }
        .expect("vertex page registers");
    allocations.insert((ALLOC_VERTICES, 1), vertex_handle);

    const HANDLE: u64 = 0x7202;
    register_pipeline(
        &transport,
        &library,
        HANDLE,
        raster(CullMode::Back, FrontFaceWinding::Clockwise),
    );

    // First pass: a wire rasterizer applies back/clockwise.
    let status = draw_triangle(
        &transport,
        &allocations,
        405,
        75,
        HANDLE,
        Some(raster(CullMode::Back, FrontFaceWinding::Clockwise)),
        0,
    );
    assert_eq!(status, ReplyStatus::Ok, "traced wire submission");
    transport.wait_for_gpu().expect("traced wire pass completes");

    // Second pass: no wire state, so Metal's default (cull none, clockwise)
    // is in force. A drawn pass must name `default`, never `unset`.
    let status = draw_triangle(&transport, &allocations, 406, 76, HANDLE, None, 0);
    assert_eq!(status, ReplyStatus::Ok, "traced default submission");
    transport.wait_for_gpu().expect("traced default pass completes");

    let report = state_trace::report();
    assert!(report.enabled, "the trace is armed: {report:?}");
    assert_eq!(report.passes, 2, "{report:?}");
    assert_eq!(report.draws, 2, "{report:?}");
    assert_eq!(report.topology[3], 2, "two triangles: {report:?}");
    assert_eq!(
        report.rasterizer_applied, 1,
        "only the wire pass applied a non-default state: {report:?}"
    );
    assert_eq!(
        report.cull, [1, 0, 1],
        "one draw ran with cull none (default) and one with cull back: {report:?}"
    );
    assert_eq!(report.winding, [2, 0], "both draws ran clockwise");
    let line = report.last_pass.expect("the pass summary").line();
    assert!(line.contains("cull=none/default"), "{line}");
    assert!(line.contains("winding=clockwise/default"), "{line}");
    assert!(!line.contains("cull=unset"), "{line}");
    assert!(
        report.pass_nonzero_after_draw > 0,
        "the drawn attachment carries pixels: {report:?}"
    );
    state_trace::reset();

    eprintln!(
        "state trace: draws=2 cull=[none:1,back:1] \
         sources=default+wire (never unset)"
    );
}
