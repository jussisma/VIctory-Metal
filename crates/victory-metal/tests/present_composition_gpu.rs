//! ADR-059 hardware gate: offscreen draw -> copy -> present.
//!
//! Draws a known colour with a real graphics pipeline into an offscreen
//! `rgba8unorm` render target, copies a region of it into the IOSurface-backed
//! texture registered as the present surface, presents that exact texture, and
//! reads the presented bytes back. The gate requires:
//!
//! - the copied-to texture is the registered present surface (identity), and
//!   `PresentPath` hands that same surface to the sink;
//! - the presented bytes are exactly the drawn colour (`nonzero_bytes > 0`,
//!   `non_black_pixels > 0`);
//! - the bounded `present_trace` names a texture-region copy into the source
//!   and no probe/source mismatch.
//!
//! No window is opened and no title process runs. D3DMetal is never touched.
//!
//! The GPU test harness serialises through `/tmp/victory-gpu.lock` internally.
//! Run:
//! ```text
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test present_composition_gpu -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use victory_metal::macos::present::{
    FRAME_PROBE_CHANNEL_THRESHOLD, PixelOrder, PresentCopyKind, PresentPath, RecordingSink,
    SurfaceImage, analyze_pixels, present_trace_report,
};
use victory_metal::macos::render::{GraphicsPipeline, PixelFormat, RenderTarget};
use victory_metal::wire::ListIdentity;
use victory_metal::{DirectMetalTransport, wire_v3};

/// Destination (presented) extent.
const EDGE: u32 = 64;
/// The offscreen render target is deliberately larger than the presented
/// surface, so the copy selects a region rather than aliasing identical sizes.
const OFFSCREEN_EDGE: u32 = 128;
const GENERATION: u32 = 1;
const COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
const HIGHLIGHT: [f32; 4] = [0.75, 0.5, 0.25, 1.0];

/// A fullscreen triangle whose vertex stage carries a flat colour from the
/// wire's root constants into the fragment stage.
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float4 color [[flat]];
};

vertex VOut pc_vertex(uint vid [[vertex_id]],
                      constant float4 &color [[buffer(1)]]) {
    float2 positions[3] = { float2(-1.0, -1.0),
                            float2( 3.0, -1.0),
                            float2(-1.0,  3.0) };
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.color = color;
    return out;
}

fragment float4 pc_fragment(VOut in [[stage_in]]) {
    return in.color;
}
"#;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut c_void;
}

/// `IOSurfaceLock` read-only option.
const IOSURFACE_LOCK_READ_ONLY: u32 = 1;

fn retained_texture(raw: *mut c_void) -> *mut c_void {
    assert!(!raw.is_null(), "present surface texture is nil");
    // SAFETY: `raw` is a live `MTLTexture`; `retain` returns the same object +1.
    unsafe {
        let selector = sel_registerName(c"retain".as_ptr());
        let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(raw, selector)
    }
}

fn wrap_surface_texture(image: &SurfaceImage) -> RenderTarget {
    let retained = retained_texture(image.texture_raw());
    // SAFETY: `retained` is a +1 retained `MTLTexture` over the IOSurface.
    unsafe { RenderTarget::from_retained_raw(retained) }
        .expect("surface texture should wrap as a render target")
}

/// Reads the presented IOSurface's rows (RGBA8, padded pitch).
fn read_surface_rgba(image: &SurfaceImage) -> Vec<u8> {
    let surface = image.iosurface_raw();
    // SAFETY: `surface` is the image's live IOSurface; a read-only lock is the
    // documented way to obtain stable CPU access.
    let rc = unsafe { IOSurfaceLock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "IOSurfaceLock(read-only) failed");
    // SAFETY: the locked surface's base address covers its rows.
    let base = unsafe { IOSurfaceGetBaseAddress(surface) }.cast::<u8>();
    assert!(!base.is_null(), "locked IOSurface has no base address");
    let len = image.bytes_per_row() * image.height() as usize;
    let mut bytes = vec![0u8; len];
    // SAFETY: `base` points to `len` readable bytes and `bytes` is a distinct
    // allocation of exactly that size.
    unsafe { std::ptr::copy_nonoverlapping(base, bytes.as_mut_ptr(), len) };
    // SAFETY: unlocks the same live surface with the matching option.
    let rc = unsafe { IOSurfaceUnlock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "IOSurfaceUnlock failed");
    bytes
}

fn float4_bytes(rgba: [f32; 4]) -> Vec<u8> {
    rgba.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn expected_rgba8(rgba: [f32; 4]) -> [u8; 4] {
    rgba.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[test]
fn an_offscreen_draw_copied_to_the_present_surface_is_non_black() {
    let transport = match DirectMetalTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("present_composition_gpu SKIP no Metal device: {error}");
            return;
        }
    };

    // The offscreen render target is larger than the presented surface.
    let offscreen = transport
        .create_render_target_2d(OFFSCREEN_EDGE, OFFSCREEN_EDGE, PixelFormat::Rgba8Unorm)
        .expect("offscreen rgba8unorm render target");
    let offscreen_id = transport
        .register_render_target(&offscreen)
        .expect("register the offscreen target");

    // The presented surface is an IOSurface-backed rgba8unorm texture.
    let mut image = SurfaceImage::create_rgba8(&transport, EDGE, EDGE)
        .expect("IOSurface-backed rgba8unorm present surface");
    assert_eq!(
        image.pixel_format(),
        70,
        "present surface must be rgba8unorm"
    );
    let surface_id = u64::from(image.surface_id());
    let target = wrap_surface_texture(&image);
    let present_texture_id = transport
        .register_render_target(&target)
        .expect("register the present texture");
    transport
        .register_present_surface(surface_id, &target)
        .expect("register the present surface");

    let pipeline = GraphicsPipeline::compile_mrt(
        &transport,
        MSL,
        "pc_vertex",
        "pc_fragment",
        &[PixelFormat::Rgba8Unorm],
        None,
    )
    .expect("offscreen pipeline should compile");
    let pipeline_id = transport
        .register_graphics_pipeline_state(pipeline)
        .expect("offscreen pipeline should register");

    let commands = vec![
        wire_v3::CommandV3::BeginRenderPass(wire_v3::BeginRenderPass {
            colors: vec![wire_v3::ColorAttachment {
                allocation_id: offscreen_id.0,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
                load: wire_v3::LoadOp::Clear,
                store: wire_v3::StoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 1.0],
            }],
            depth_stencil: None,
        }),
        wire_v3::CommandV3::SetGraphicsPipeline(wire_v3::PipelineRef {
            pipeline_id: pipeline_id.0,
        }),
        wire_v3::CommandV3::SetViewport(wire_v3::Viewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(OFFSCREEN_EDGE),
            height: f64::from(OFFSCREEN_EDGE),
            near: 0.0,
            far: 1.0,
        }),
        wire_v3::CommandV3::SetScissor(wire_v3::ScissorRect {
            x: 0,
            y: 0,
            width: OFFSCREEN_EDGE,
            height: OFFSCREEN_EDGE,
        }),
        wire_v3::CommandV3::SetPrimitiveTopology(wire_v3::PrimitiveTopology::TriangleList),
        wire_v3::CommandV3::SetRootConstants(wire_v3::RootConstantsV3 {
            slot: 1,
            bytes: float4_bytes(COLOR),
        }),
        wire_v3::CommandV3::Draw(wire_v3::DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        // A second colour in the right half makes the frame a two-tone pattern,
        // not a uniform clear that could pass as content by accident.
        wire_v3::CommandV3::SetScissor(wire_v3::ScissorRect {
            x: EDGE / 2,
            y: 0,
            width: EDGE / 2,
            height: EDGE,
        }),
        wire_v3::CommandV3::SetRootConstants(wire_v3::RootConstantsV3 {
            slot: 1,
            bytes: float4_bytes(HIGHLIGHT),
        }),
        wire_v3::CommandV3::Draw(wire_v3::DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        wire_v3::CommandV3::EndRenderPass,
        // Offscreen -> presented surface: a same-format RGBA8 region copy of
        // the destination extent.
        wire_v3::CommandV3::CopyTextureRegion(wire_v3::CopyTextureRegion {
            src: wire_v3::TextureSubresource {
                allocation_id: offscreen_id.0,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
            },
            dst: wire_v3::TextureSubresource {
                allocation_id: present_texture_id.0,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
            },
            src_origin: [0, 0, 0],
            dst_origin: [0, 0, 0],
            size: [EDGE, EDGE, 1],
        }),
        wire_v3::CommandV3::Present(wire_v3::Present {
            surface_id,
            sync_interval: 0,
        }),
    ];
    let submission = wire_v3::SubmissionV3 {
        sequence: 1,
        list: Some(ListIdentity {
            list_id: 1,
            list_generation: 1,
        }),
        commands,
    };
    let mut allocations = std::collections::HashMap::new();
    allocations.insert(
        (offscreen_id.0, GENERATION),
        victory_metal::abi::VmBufferId(offscreen_id.0),
    );
    allocations.insert(
        (present_texture_id.0, GENERATION),
        victory_metal::abi::VmBufferId(present_texture_id.0),
    );
    let reply = transport
        .submit_v3(&submission, &allocations)
        .expect("draw, copy and present must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    transport
        .wait_for_gpu()
        .expect("the draw/copy submission must complete");

    // Present the drawn image through the real headless present path.
    let mut path = PresentPath::new(&transport);
    let mut sink = RecordingSink::new();
    path.present_drawn(&mut image, &mut sink)
        .expect("the copied frame must present");
    assert_eq!(sink.frames().len(), 1, "the sink must observe one frame");
    assert_eq!(
        sink.frames()[0].surface_id,
        image.surface_id(),
        "the presented surface is the copied-to texture"
    );
    assert_eq!(surface_id, u64::from(image.surface_id()));
    path.finish_present().expect("the presented frame finishes");

    // Read the presented IOSurface: exact bytes, not a summary of the source.
    let presented = read_surface_rgba(&image);
    let row_bytes = image.bytes_per_row();
    let base = expected_rgba8(COLOR);
    let highlight = expected_rgba8(HIGHLIGHT);
    let mut pixels = Vec::with_capacity((EDGE * EDGE) as usize);
    for y in 0..EDGE as usize {
        for x in 0..EDGE as usize {
            let at = y * row_bytes + x * 4;
            let pixel = [
                presented[at],
                presented[at + 1],
                presented[at + 2],
                presented[at + 3],
            ];
            let expected = if x >= (EDGE / 2) as usize {
                highlight
            } else {
                base
            };
            assert_eq!(
                pixel, expected,
                "presented pixel ({x}, {y}) is {pixel:?}, expected {expected:?}"
            );
            pixels.push(pixel);
        }
    }
    let nonzero_bytes = pixels
        .iter()
        .flat_map(|pixel| pixel.iter())
        .filter(|byte| **byte != 0)
        .count() as u64;
    assert_eq!(
        nonzero_bytes,
        u64::from(EDGE) * u64::from(EDGE) * 4,
        "every presented byte must be drawn content"
    );
    let sample = analyze_pixels(
        &pixels,
        PixelOrder::Rgba,
        [0, 0, 0, 255],
        FRAME_PROBE_CHANNEL_THRESHOLD,
    );
    assert_eq!(sample.non_black_pixels, u64::from(EDGE) * u64::from(EDGE));
    assert!(sample.non_black_pixels > 0);
    assert!(
        sample.unique_luma_buckets >= 2,
        "a two-tone frame must have at least two colour classes: {sample:?}"
    );
    assert!(
        sample.distinct_sample_values >= 2,
        "a two-tone frame must have at least two exact values: {sample:?}"
    );

    // The bounded trace must name a copy into the presented source and must
    // not report a probe/source mismatch.
    let trace = present_trace_report();
    assert!(trace.frames >= 1, "the present frame was recorded");
    assert!(
        trace.copied_into_source >= 1,
        "a host copy wrote the presented source: {trace:?}"
    );
    assert_eq!(trace.probe_source_mismatch, 0);
    assert!(
        trace.keys.iter().any(|entry| {
            entry.copy_kind == PresentCopyKind::CopyTextureRegion && entry.source_is_presented
        }),
        "trace must name a copy into the presented texture: {trace:?}"
    );
    eprintln!(
        "present_composition_gpu PASS extent={EDGE}x{EDGE} offscreen={OFFSCREEN_EDGE}x{OFFSCREEN_EDGE} \
         nonzero_bytes={nonzero_bytes} non_black_pixels={} distinct_values={} \
         base={base:?} highlight={highlight:?} trace={}",
        sample.non_black_pixels,
        sample.distinct_sample_values,
        trace.log_line()
    );
}
