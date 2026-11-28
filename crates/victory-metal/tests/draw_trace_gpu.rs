//! GPU proof for ADR-053: a real offscreen draw makes the traced attachment
//! non-zero, and a clear-only pass is not sampled as drawn output.
//!
//! Both cases force the tracer on through `VICTORY_METAL_DRAW_TRACE=1` (the
//! same environment flag a boot sets) before any device call, then reset the
//! process-wide report. No test-only constructor is added to the library; the
//! flag is the only arming path, so production and proof share it.
//!
//! Run serialised with the binary's own GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test draw_trace_gpu -- --ignored --test-threads=1 --nocapture'
//! ```
//!
//! `--test-threads=1` is required: the two cases share one process-wide trace
//! report and one environment flag.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::render::{GraphicsPipeline, PixelFormat, RenderTarget};
use victory_metal::macos::trace;
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CullMode, DrawState, FillMode, FrontFaceWinding,
    LoadOp, PipelineRef, PrimitiveTopology, RasterizerState, ScissorRect, StoreOp, SubmissionV3,
    Viewport,
};

const EDGE: u32 = 64;
const TARGET: u64 = 1;
const SEQUENCE: u32 = 1;

/// One full-viewport triangle, red with opaque alpha.
const TRACE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

vertex VOut trace_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0),
    };
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    return out;
}

fragment float4 trace_fragment() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#;

fn enable_trace() {
    // SAFETY: both tests run with `--test-threads=1` and write the same value
    // before the first trace call in this test binary.
    unsafe { std::env::set_var(trace::DRAW_TRACE_ENV, "1") };
    trace::reset();
}

fn create_target(
    transport: &DirectMetalTransport,
) -> (RenderTarget, HashMap<(u64, u32), VmBufferId>) {
    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("offscreen RGBA8 target");
    let handle = transport
        .register_render_target(&target)
        .expect("offscreen target registers");
    let mut allocations = HashMap::new();
    allocations.insert((TARGET, 1), VmBufferId(handle.0));
    (target, allocations)
}

fn pipeline(transport: &DirectMetalTransport) -> u64 {
    let pipeline = GraphicsPipeline::compile(
        transport,
        TRACE_MSL,
        "trace_vertex",
        "trace_fragment",
        PixelFormat::Rgba8Unorm,
        None,
    )
    .expect("trace pipeline compiles");
    transport
        .register_graphics_pipeline_state(pipeline)
        .expect("trace pipeline registers")
        .0
}

fn colour_attachment(load: LoadOp, clear: [f64; 4]) -> ColorAttachment {
    ColorAttachment {
        allocation_id: TARGET,
        generation: 1,
        mip_level: 0,
        slice: 0,
        load,
        store: StoreOp::Store,
        clear_color: clear,
    }
}

/// A fragment shader that writes zeros over its whole triangle.
const ZERO_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

vertex VOut zero_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0),
    };
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    return out;
}

fragment float4 zero_fragment() {
    return float4(0.0, 0.0, 0.0, 0.0);
}
"#;

/// A pipeline that clears/loads, draws one full-viewport triangle and ends the
/// pass.
fn draw_commands(pipeline_id: u64, load: LoadOp, clear: [f64; 4]) -> Vec<CommandV3> {
    vec![
        begin_pass(load, clear),
        CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
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
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ]
}

/// Compiles the zero-writing pipeline on a real device.
fn zero_pipeline(transport: &DirectMetalTransport) -> u64 {
    let pipeline = GraphicsPipeline::compile(
        transport,
        ZERO_MSL,
        "zero_vertex",
        "zero_fragment",
        PixelFormat::Rgba8Unorm,
        None,
    )
    .expect("zero pipeline compiles");
    transport
        .register_graphics_pipeline_state(pipeline)
        .expect("zero pipeline registers")
        .0
}

fn begin_pass(load: LoadOp, clear: [f64; 4]) -> CommandV3 {
    CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![colour_attachment(load, clear)],
        depth_stencil: None,
    })
}

fn submit(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: Vec<CommandV3>,
) -> Duration {
    submit_sequence(transport, allocations, commands, SEQUENCE)
}

fn submit_sequence(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: Vec<CommandV3>,
    sequence: u32,
) -> Duration {
    let started = Instant::now();
    let (reply, _stats) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence,
                list: None,
                commands,
            },
            allocations,
        )
        .expect("submission executes");
    let elapsed = started.elapsed();
    assert_eq!(reply.status, ReplyStatus::Ok, "submission status");
    transport.wait_for_gpu().expect("GPU completes");
    elapsed
}

/// A draw into an offscreen RGBA8 target is described by the tracer and its
/// sample reports non-zero bytes from that offscreen attachment.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn a_real_draw_makes_the_offscreen_sample_nonzero() {
    enable_trace();
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let (target, allocations) = create_target(&transport);
    let pipeline_id = pipeline(&transport);

    let commands = vec![
        begin_pass(LoadOp::Clear, [0.0, 0.0, 0.0, 0.0]),
        CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
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
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ];
    let submit_cost = submit(&transport, &allocations, commands);

    let record = trace::last_record().expect("the draw was described");
    assert_eq!(record.kind, "draw");
    assert_eq!(record.count, 3);
    assert_eq!(record.instances, 1);
    let attachment = record.target.expect("the record names its attachment");
    assert!(!attachment.swapchain, "the attachment is offscreen");
    assert_eq!(
        (attachment.width, attachment.height, attachment.format),
        (EDGE, EDGE, "rgba8unorm")
    );
    assert!(
        record.line().contains("rt=offscreen 64x64 rgba8unorm"),
        "{}",
        record.line()
    );

    let sample = trace::last_sample().expect("the drawn pass was sampled");
    assert_eq!(sample.pass_draws, 1, "the pass line names its draw count");
    assert_eq!(sample.target.map(|target| target.swapchain), Some(false));
    assert!(
        sample.nonzero_bytes > 0,
        "a drawn offscreen target must not read back all-zero: {sample:?}"
    );
    assert!(
        sample.distinct_values >= 2,
        "red on black has at least two byte values: {sample:?}"
    );

    let report = trace::report();
    assert!(report.enabled);
    assert!(report.traced_work >= 1, "{report:?}");
    assert_eq!(report.traced_passes, 1, "{report:?}");
    assert!(
        report.last_offscreen_nonzero_bytes > 0,
        "telemetry must carry the non-zero offscreen sample: {report:?}"
    );
    assert!(report.last_offscreen_distinct_values >= 2, "{report:?}");

    // Diagnostic only: this is the whole traced submission (encode + commit +
    // wait + 64x64 readback + summary), not the sampler in isolation.
    eprintln!(
        "draw_trace_gpu: traced_submit_us={} sample={}",
        submit_cost.as_micros(),
        sample.line().replace('\n', " | ")
    );

    // Bounded-cost evidence: a standalone diagnostic readback of the same
    // 64x64 target and the CPU summary of a 1920x1080 buffer. `sample_grid`
    // always reads the same 64x64 grid, so its cost is independent of extent.
    let readback_start = Instant::now();
    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("diagnostic readback");
    let readback_us = readback_start.elapsed().as_micros();
    assert!(pixels.iter().any(|pixel| pixel[0] != 0));
    let wide = vec![0x7f_u8; 1920 * 1080 * 4];
    let summary_start = Instant::now();
    let wide_stats = trace::sample_grid(&wide, 1920, 1080);
    let summary_us = summary_start.elapsed().as_micros();
    assert_eq!(
        wide_stats.sampled_bytes,
        u64::from(trace::RT_SAMPLE_EDGE) * u64::from(trace::RT_SAMPLE_EDGE) * 4
    );
    eprintln!(
        "draw_trace_gpu: standalone_readback_us={readback_us} \
         1080p_grid_summary_us={summary_us}"
    );
    let _ = target;
}

/// A pass that only clears is genuinely non-zero on the GPU, but the tracer
/// must not sample it as drawn work or report it as offscreen content.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn a_clear_only_pass_is_not_sampled_as_drawn_output() {
    enable_trace();
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let (target, allocations) = create_target(&transport);

    // Opaque red clear: the attachment is non-zero even though no draw ran.
    let commands = vec![
        begin_pass(LoadOp::Clear, [1.0, 0.0, 0.0, 1.0]),
        CommandV3::EndRenderPass,
    ];
    let submit_cost = submit(&transport, &allocations, commands);

    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("clear-only target readback");
    assert!(
        pixels.iter().any(|pixel| pixel[0] != 0),
        "the clear really wrote a non-zero colour"
    );

    let report = trace::report();
    assert!(report.enabled);
    assert_eq!(report.traced_passes, 0, "a drawless pass is not sampled");
    assert_eq!(
        report.last_offscreen_nonzero_bytes, 0,
        "drawless output must not be reported as traced content: {report:?}"
    );
    assert!(
        trace::last_sample().is_none(),
        "no pixel summary exists for a drawless pass"
    );

    eprintln!(
        "draw_trace_gpu: clear_only_submit_us={} report={report:?}",
        submit_cost.as_micros()
    );
}

// ---------------------------------------------------------------------------
// ADR-068: "written after last clear" evidence on the real GPU.
//
// Boot #36 reported `non_black_pixels = 0` on a pass the tracer had sampled and
// could not say whether the shader drew zeros, a later clear erased non-zero
// pixels, or the non-zero RT was never shown. These three cases pin each class
// on a 64x64 offscreen target.
// ---------------------------------------------------------------------------

/// A zero clear followed by the red draw is `written-after-clear`: the clear
/// alone accounts for zero bytes and the readback is non-zero.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn a_nonzero_draw_after_a_zero_clear_is_written_after_clear() {
    enable_trace();
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let (target, allocations) = create_target(&transport);
    let pipeline_id = pipeline(&transport);
    submit(
        &transport,
        &allocations,
        draw_commands(pipeline_id, LoadOp::Clear, [0.0, 0.0, 0.0, 0.0]),
    );

    let sample = trace::last_sample().expect("the drawn pass was sampled");
    assert_eq!(sample.draws_since_clear, 1, "{sample:?}");
    assert_eq!(sample.clear_color, Some([0.0, 0.0, 0.0, 0.0]));
    assert_eq!(
        sample.nonzero_bytes_after_clear,
        Some(0),
        "a zero clear writes zero bytes: {sample:?}"
    );
    assert!(
        sample.nonzero_bytes_after_draws() > 0,
        "the red draw must survive the clear: {sample:?}"
    );
    assert_eq!(
        sample.distinct_pixels_after_draws, 1,
        "a full-viewport red draw is one distinct pixel even though it has \
         two distinct byte values: {sample:?}"
    );
    assert!(sample.distinct_values >= 2, "{sample:?}");

    let evidence = trace::rt_write_evidence();
    assert_eq!(evidence.rt_traced, 1, "{evidence:?}");
    assert_eq!(evidence.rt_sampled, 1, "{evidence:?}");
    assert_eq!(evidence.rt_written_after_clear, 1, "{evidence:?}");
    assert_eq!(evidence.rts[0].class, trace::RT_CLASS_WRITTEN);
    let _ = target;
}

/// A non-zero clear followed by a draw that writes zeros is `drew-zeros`: the
/// analytic clear bytes are non-zero and the readback is zero.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn a_zero_draw_over_a_nonzero_clear_is_drew_zeros() {
    enable_trace();
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let (target, allocations) = create_target(&transport);
    let pipeline_id = zero_pipeline(&transport);
    let clear = [0.25, 0.5, 0.75, 1.0];
    submit(
        &transport,
        &allocations,
        draw_commands(pipeline_id, LoadOp::Clear, clear),
    );

    let sample = trace::last_sample().expect("the drawn pass was sampled");
    assert_eq!(sample.draws_since_clear, 1, "{sample:?}");
    assert_eq!(sample.clear_color, Some(clear));
    assert_eq!(
        sample.nonzero_bytes_after_clear,
        Some(sample.sampled_bytes),
        "every channel of the clear is non-zero: {sample:?}"
    );
    assert_eq!(
        sample.nonzero_bytes_after_draws(),
        0,
        "the shader wrote zeros over the clear: {sample:?}"
    );
    assert_eq!(sample.distinct_pixels_after_draws, 1, "{sample:?}");

    let evidence = trace::rt_write_evidence();
    assert_eq!(evidence.rt_traced, 1, "{evidence:?}");
    assert_eq!(
        evidence.rt_written_after_clear, 0,
        "a zero draw is not written output: {evidence:?}"
    );
    assert_eq!(evidence.rts[0].class, trace::RT_CLASS_ZERO);
    let _ = target;
}


// ---------------------------------------------------------------------------
// Cycle 18B: the encoder-state trace (`VICTORY_METAL_STATE_TRACE=1`).
//
// Boot #32 encoded 85 three-vertex 4K draws and the attachment still had one
// distinct value. `draw_trace` describes the work but not the encoder state, so
// the new trace reports topology, attachment load/store, depth state, blend
// census, cull/winding and one bounded after-draw pixel sample of the pass's
// colour attachment. This case proves the accounting on the real GPU at 3840x2160.
// ---------------------------------------------------------------------------

const FOUR_K_WIDTH: u32 = 3840;
const FOUR_K_HEIGHT: u32 = 2160;
const ALLOC_4K: u64 = 0x4A10;

fn state_trace_4k_target(
    transport: &DirectMetalTransport,
) -> (RenderTarget, HashMap<(u64, u32), VmBufferId>) {
    let target = transport
        .create_render_target_2d(FOUR_K_WIDTH, FOUR_K_HEIGHT, PixelFormat::Rgba8Unorm)
        .expect("4K offscreen target");
    let handle = transport
        .register_render_target(&target)
        .expect("4K target registers");
    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_4K, 1), VmBufferId(handle.0));
    (target, allocations)
}

/// The encoder-state trace names topology, load/store, depth, blend and
/// cull/winding for a 4K draw, and its after-draw sample is non-zero.
#[test]
fn state_trace_names_the_encoder_state_of_a_4k_draw() {
    // SAFETY: this test binary runs with `--test-threads=1`, so the process-wide
    // flag and report are written before any device call by this test.
    unsafe {
        std::env::set_var(
            victory_metal::macos::render::state_trace::STATE_TRACE_ENV,
            "1",
        )
    };
    victory_metal::macos::render::state_trace::reset();
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let (target, allocations) = state_trace_4k_target(&transport);
    let pipeline_id = pipeline(&transport);

    let commands = vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![ColorAttachment {
                allocation_id: ALLOC_4K,
                generation: 1,
                mip_level: 0,
                slice: 0,
                load: LoadOp::Clear,
                store: StoreOp::Store,
                // Transparent black, so any non-zero sample byte is the draw's.
                clear_color: [0.0, 0.0, 0.0, 0.0],
            }],
            depth_stencil: None,
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
        CommandV3::SetViewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(FOUR_K_WIDTH),
            height: f64::from(FOUR_K_HEIGHT),
            near: 0.0,
            far: 1.0,
        }),
        CommandV3::SetScissor(ScissorRect {
            x: 0,
            y: 0,
            width: FOUR_K_WIDTH,
            height: FOUR_K_HEIGHT,
        }),
        CommandV3::SetRasterizerState(RasterizerState {
            fill: FillMode::Solid,
            cull: CullMode::None,
            front_face: FrontFaceWinding::Clockwise,
            depth_bias: 0,
            slope_scaled_depth_bias: 0.0,
            depth_bias_clamp: 0.0,
            depth_clip: true,
        }),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ];
    submit(&transport, &allocations, commands);

    let report = victory_metal::macos::render::state_trace::report();
    assert!(report.enabled, "the state trace is armed: {report:?}");
    assert_eq!(report.passes, 1, "{report:?}");
    assert_eq!(report.draws, 1, "{report:?}");
    assert_eq!(report.topology[3], 1, "one triangle: {report:?}");
    assert_eq!(report.color_attachments, 1, "{report:?}");
    assert_eq!(report.color_load[2], 1, "the pass loads with a clear");
    assert_eq!(report.color_store[1], 1, "the pass stores the attachment");
    assert_eq!(report.depth_attachment_passes, 0, "no depth attachment");
    assert_eq!(report.depth_bind[0], 1, "the pass ran with no depth state");
    assert_eq!(
        report.blend_baked_attachments, 0,
        "a pipeline compiled without a blend state bakes no mask: {report:?}"
    );
    assert_eq!(report.blend_write_mask_zero, 0, "{report:?}");
    assert_eq!(report.rasterizer_applied, 1, "{report:?}");
    assert_eq!(report.cull[0], 1, "cull none: {report:?}");
    assert_eq!(report.winding[0], 1, "clockwise: {report:?}");

    let pass = report.last_pass.expect("the pass summary is published");
    assert_eq!(pass.draws, 1, "{pass:?}");
    assert_eq!(pass.topology, "triangle", "{pass:?}");
    assert_eq!(pass.color_load, "clear", "{pass:?}");
    assert_eq!(pass.depth_bind, "none", "{pass:?}");
    assert!(
        pass.line().contains("state_trace_pass") && pass.line().contains("topology=triangle"),
        "{}",
        pass.line()
    );

    assert_eq!(report.pass_sampled_passes, 1, "{report:?}");
    assert_eq!(report.pass_samples_pending, 0, "{report:?}");
    assert_eq!(report.pass_sample_failures, 0, "{report:?}");
    assert_eq!(report.pass_sample_bytes, 128, "two 4x4 RGBA8 blocks");
    assert!(
        report.pass_nonzero_after_draw > 0,
        "the draw wrote colour into the 4K attachment: {report:?}"
    );
    assert!(
        report.pass_sample_distinct_values >= 2,
        "red on transparent black has two byte values: {report:?}"
    );
    assert!(report.pass_corner_nonzero > 0, "{report:?}");
    assert!(report.pass_centre_nonzero > 0, "{report:?}");

    eprintln!(
        "state_trace_gpu: draws={} topology={:?} depth={:?} blend_zero_mask={} \
         sample_nonzero={}/{} distinct={}",
        report.draws,
        report.topology,
        report.depth_bind,
        report.blend_write_mask_zero,
        report.pass_nonzero_after_draw,
        report.pass_sample_bytes,
        report.pass_sample_distinct_values
    );
    let _ = target;
}
