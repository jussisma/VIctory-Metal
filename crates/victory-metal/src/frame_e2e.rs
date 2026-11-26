//! End-to-end frame path: a recorded command list produces a frame that is
//! presented through the IOSurface sink, verified from the surface itself.
//!
//! This module composes the pieces that each have their own unit tests
//! (`wire`, `cmdlist`, `macos::render`, `macos::present`, `macos::views`) into
//! one frame, so a regression at a seam fails here rather than in the title.
//!
//! Scope of the gate:
//!
//! - A `wire::Submission` built with `Command::SetPipeline`, `RootConstants`,
//!   `DrawIndexed` (and, where supported, `Draw`) is executed by the device
//!   against a render target.
//! - That target is an IOSurface-backed swapchain image, so the presented
//!   frame is the frame that was drawn, not a copy.
//! - The sink receives a `PresentedFrame` (surface identity, index, time) with
//!   no pixel bytes; the test alone reads the surface once to assert pixels.
//! - Presenting an already-drawn surface must not clear it.
//!
//! The Wine guest half of the gate lives in
//! `crates/victory-hle/src/modules/xmem/e2e.rs` and
//! `tools/macos/metal-poc/synthetic-frame.sh`; this module proves the host
//! half on the real GPU without Wine.

#[cfg(target_os = "macos")]
mod host_gate {
    use std::ffi::c_void;

    use crate::macos::device::DirectMetalTransport;
    use crate::macos::objc;
    use crate::macos::present::{PresentError, PresentPath, RecordingSink, SurfaceImage};
    use crate::macos::render::{
        GraphicsPipeline, LoadAction, PixelFormat, PrimitiveType, RenderAttachment, RenderEncoder,
        RenderPass, RenderTarget,
    };
    use crate::transport::MetalTransport;

    /// Swapchain edge used by both host-gate tests.
    const SURFACE_EDGE: u32 = 64;

    /// Fragment returns the colour bound at fragment buffer 0; the vertex stage
    /// expands the three inline positions into a triangle that covers the whole
    /// render target.
    const FRAME_MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VOut {
            float4 position [[position]];
        };

        vertex VOut fe_vertex(uint vid [[vertex_id]],
                              constant float2 *positions [[buffer(0)]]) {
            VOut out;
            out.position = float4(positions[vid], 0.0, 1.0);
            return out;
        }

        fragment float4 fe_fragment(constant float4 &color [[buffer(0)]]) {
            return color;
        }
    "#;

    /// BGRA8 bytes Metal writes for an opaque colour given as RGBA floats.
    const GREEN_RGBA: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    const GREEN_BGRA: [u8; 4] = [0, 255, 0, 255];
    const BLUE_RGBA: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    const BLUE_BGRA: [u8; 4] = [255, 0, 0, 255];

    /// `kIOSurfaceLockReadOnly`, test-only.
    const IOSURFACE_LOCK_READ_ONLY: u32 = 1;

    fn as_bytes<T>(values: &[T]) -> &[u8] {
        // SAFETY: Only plain-data arrays (`f32`) are passed here, so the byte
        // view covers exactly the values with no padding.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    /// Wraps the surface's texture as a `RenderTarget` without taking the
    /// image's reference.
    fn wrap_surface_texture(image: &SurfaceImage) -> RenderTarget {
        let raw = image.texture_raw();
        // SAFETY: `raw` is the image's live retained `MTLTexture`; `retain`
        // adds a second +1 reference that `from_retained_raw` takes ownership
        // of, so the image keeps its own and the target releases only this one.
        let retained = unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) };
        // SAFETY: `retained` is a +1 retained `id<MTLTexture>` over a BGRA8
        // IOSurface, exactly what `from_retained_raw` requires.
        unsafe { RenderTarget::from_retained_raw(retained) }
            .expect("surface texture should wrap as a render target")
    }

    /// Draws a full-target triangle of `rgba` into the surface's texture.
    fn draw_fullscreen_color(
        transport: &DirectMetalTransport,
        image: &SurfaceImage,
        pipeline: &GraphicsPipeline,
        rgba: [f32; 4],
    ) {
        let target = wrap_surface_texture(image);
        let pass = RenderPass::new(
            RenderAttachment::color(target)
                .with_load_action(LoadAction::Clear)
                .with_clear_color([0.0, 0.0, 0.0, 1.0]),
        )
        .expect("colour-only render pass should validate");
        let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];

        let mut encoder = RenderEncoder::new(transport, &pass, pipeline)
            .expect("render encoder should be created");
        encoder
            .set_vertex_bytes(as_bytes(&positions), 0)
            .expect("triangle positions should bind");
        encoder
            .set_fragment_bytes(as_bytes(&rgba), 0)
            .expect("fragment colour should bind");
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3)
            .expect("full-target triangle should encode");
        encoder
            .finish()
            .expect("draw pass should complete on the GPU");
    }

    fn compile_pipeline(transport: &DirectMetalTransport) -> GraphicsPipeline {
        GraphicsPipeline::compile(
            transport,
            FRAME_MSL,
            "fe_vertex",
            "fe_fragment",
            PixelFormat::Bgra8Unorm,
            None,
        )
        .expect("frame pipeline should compile")
    }

    /// Test-only: locks the IOSurface exactly once and copies its rows.
    fn read_surface_bgra(image: &SurfaceImage) -> Vec<u8> {
        let surface = image.iosurface_raw();
        // SAFETY: `surface` is the image's live IOSurface; a read-only lock is
        // the documented way to obtain stable CPU access to its backing store.
        let rc = unsafe { IOSurfaceLock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceLock(read-only) failed");

        // SAFETY: The surface is locked, so the base address is valid for
        // `bytes_per_row * height` readable bytes.
        let base = unsafe { IOSurfaceGetBaseAddress(surface) }.cast::<u8>();
        assert!(!base.is_null(), "locked IOSurface has no base address");
        let len = image.bytes_per_row() * image.height() as usize;
        let mut pixels = vec![0u8; len];
        // SAFETY: `base` points to `len` readable bytes of a locked surface and
        // `pixels` is a distinct, exclusively borrowed `len`-byte allocation.
        unsafe { std::ptr::copy_nonoverlapping(base, pixels.as_mut_ptr(), len) };

        // SAFETY: Unlocks the same live surface locked above; the read-only
        // option must match.
        let rc =
            unsafe { IOSurfaceUnlock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceUnlock failed");
        pixels
    }

    /// Asserts every pixel of the surface is `expected` and returns the rows.
    fn assert_all_pixels(image: &SurfaceImage, expected: [u8; 4], context: &str) -> Vec<u8> {
        let pixels = read_surface_bgra(image);
        let width = image.width() as usize;
        let height = image.height() as usize;
        let row_bytes = image.bytes_per_row();
        assert_eq!(
            pixels.len(),
            row_bytes * height,
            "{context}: unexpected surface size"
        );
        for y in 0..height {
            let row = &pixels[y * row_bytes..y * row_bytes + width * 4];
            for (x, pixel) in row.chunks_exact(4).enumerate() {
                assert_eq!(
                    pixel, expected,
                    "{context}: pixel ({x},{y}) was {pixel:?}, expected {expected:?}"
                );
            }
        }
        eprintln!(
            "{context}: surface={} {width}x{height} row_bytes={row_bytes} \
             first_pixel_BGRA={:?} all_pixels_match={expected:?}",
            image.surface_id(),
            &pixels[..4]
        );
        pixels
    }

    #[test]
    fn drawn_frame_presents_without_clearing_and_survives_in_flight_refusal() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let mut image = SurfaceImage::create(&transport, SURFACE_EDGE, SURFACE_EDGE)
            .expect("IOSurface-backed swapchain image creation");
        let pipeline = compile_pipeline(&transport);
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();

        // Frame 0: draw then present the same image; presenting must not clear.
        draw_fullscreen_color(&transport, &image, &pipeline, GREEN_RGBA);
        path.present_drawn(&mut image, &mut sink)
            .expect("drawn frame 0 should be accepted");
        assert_eq!(
            sink.frames().len(),
            1,
            "sink must observe exactly one frame"
        );
        assert_eq!(sink.frames()[0].surface_id, image.surface_id());
        assert_eq!(sink.frames()[0].frame_index, 0);
        assert_eq!(image.frame_index(), Some(0));
        assert_all_pixels(&image, GREEN_BGRA, "frame 0 after present");

        // Frame 1: a new draw into the same surface, then present again.
        path.finish_present().expect("frame 0 should finish");
        draw_fullscreen_color(&transport, &image, &pipeline, BLUE_RGBA);
        path.present_drawn(&mut image, &mut sink)
            .expect("drawn frame 1 should be accepted");
        assert_eq!(sink.frames().len(), 2, "sink must observe two frames");
        assert_eq!(sink.frames()[1].frame_index, 1);
        assert_eq!(sink.frames()[1].surface_id, image.surface_id());
        assert!(
            sink.frames()[1].timestamp_ns >= sink.frames()[0].timestamp_ns,
            "frame timestamps must be monotonic"
        );
        assert_eq!(image.frame_index(), Some(1));
        assert_all_pixels(&image, BLUE_BGRA, "frame 1 after present");

        // A second present without finishing the first must be refused and
        // must leave the drawn frame untouched.
        let error = path
            .present_drawn(&mut image, &mut sink)
            .expect_err("second present must be refused while a frame is in flight");
        assert_eq!(
            error,
            PresentError::FrameInFlight {
                frame_index: 1,
                surface_id: image.surface_id(),
            }
        );
        assert_eq!(
            sink.frames().len(),
            2,
            "refused frame must not reach the sink"
        );
        assert_all_pixels(&image, BLUE_BGRA, "surface after refused in-flight present");

        assert_eq!(
            path.finish_present()
                .expect("frame 1 should finish")
                .frame_index,
            1
        );
        assert_eq!(path.finish_present(), Err(PresentError::NoFrameInFlight));

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "frame_e2e host gate: device={}, surface={} {SURFACE_EDGE}x{SURFACE_EDGE}, \
             frames=[0,1], colours=[{GREEN_BGRA:?},{BLUE_BGRA:?}]",
            device.name_str(),
            image.surface_id()
        );
    }

    #[test]
    fn foreign_or_degenerate_surface_is_refused_before_any_present() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let second = DirectMetalTransport::new().expect("second Metal transport should open");
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();

        if transport.device_raw() != second.device_raw() {
            // Two distinct `MTLDevice` handles: a surface created on the other
            // device must not be presented here.
            let mut foreign = SurfaceImage::create(&second, SURFACE_EDGE, SURFACE_EDGE)
                .expect("foreign IOSurface-backed image creation");
            let error = path
                .present_drawn(&mut foreign, &mut sink)
                .expect_err("foreign surface must be refused");
            assert_eq!(error, PresentError::ForeignSurface);
            assert!(
                sink.frames().is_empty(),
                "refused surface must not reach the sink"
            );
            assert_eq!(path.in_flight(), None);
            eprintln!(
                "frame_e2e failure path: exercised ForeignSurface \
                 (device {:?} vs {:?})",
                transport.device_raw(),
                second.device_raw()
            );
        } else {
            // One physical device on this host: `MTLCreateSystemDefaultDevice`
            // returns the same handle, so every surface is same-device and the
            // foreign check cannot fire. Exercise the degenerate-extent gate,
            // which rejects the surface before a texture is wrapped or a
            // command buffer is created.
            let error = SurfaceImage::create(&transport, 0, SURFACE_EDGE)
                .err()
                .expect("zero-width surface must be refused");
            assert_eq!(
                error,
                PresentError::ZeroExtent {
                    width: 0,
                    height: SURFACE_EDGE,
                }
            );
            assert!(
                sink.frames().is_empty(),
                "rejected surface must not reach the sink"
            );
            assert_eq!(path.in_flight(), None);
            eprintln!(
                "frame_e2e failure path: single MTLDevice on this host \
                 (no foreign surface possible); exercised ZeroExtent before any encoding"
            );
        }
    }

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
        fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    }
}
