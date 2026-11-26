//! Swapchain present over the native ARM64 Metal presenter.
//!
//! Implements the `VICTORY_BACKEND=metal` present path: the swapchain's
//! back-buffer becomes a host-visible IOSurface presented by
//! `victory_metal::macos::present`, with no legacy host translation call and
//! no CPU readback.
//!
//! # Ownership and dependency direction
//!
//! This is the Win64 guest-side crate; it stays free of the host Metal crate
//! so the guest build does not link a host-only backend. The handoff is
//! therefore a narrow local seam:
//!
//! - [`NativeSwapchain`] models the host swapchain state (surface identity,
//!   extent, format, frame index, in-flight frame, sync interval) and can be
//!   built headlessly.
//! - [`SurfacePresenter`] is implemented by the host adapter around
//!   `victory_metal::macos::present::PresentPath` / `PresentSink`.
//! - [`PresentedSurface`] carries only identity and pacing, never pixels, so
//!   the production path cannot introduce a `getBytes`, a memcpy or a surface
//!   lock.
//!
//! [`SwapchainPresent`] mirrors the two fields of
//! `victory_metal::wire_v3::Present` (`surface_id`, `sync_interval`) so the
//! producer can pass a decoded present command without this crate depending on
//! the host crate.
//!
//! # Backend discipline
//!
//! A surface created by the legacy host path is refused with
//! [`PresentError::BackendMismatch`] when native Metal presentation is
//! selected. There is no silent fallback: the caller receives an explicit
//! error and no frame is presented.

use std::time::Instant;

use crate::backend::GpuBackend;

/// Largest 2D surface edge the native presenter accepts.
///
/// Matches `victory_metal::macos::present`; a larger back-buffer is rejected
/// before any host call.
pub const MAX_SURFACE_DIMENSION: u32 = 16_384;

/// Pixel format of a native Metal swapchain back-buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceFormat {
    /// 8-bit BGRA, the IOSurface format the presenter renders into.
    #[default]
    Bgra8Unorm,
}

impl SurfaceFormat {
    /// Bytes per pixel for this format.
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Bgra8Unorm => 4,
        }
    }

    /// Canonical lowercase format name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bgra8Unorm => "bgra8unorm",
        }
    }
}

/// Frame pacing requested for one present.
///
/// Mirrors `victory_metal::macos::present::SyncInterval`; only immediate (`0`)
/// and vertical sync (`1`) are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncInterval {
    /// Present immediately (sync interval `0`).
    #[default]
    Immediate,
    /// Present on vertical sync (sync interval `1`).
    Vsync,
}

impl SyncInterval {
    /// Validates a wire sync interval.
    ///
    /// Any value other than `0` or `1` is
    /// [`PresentError::UnsupportedSyncInterval`].
    pub fn from_wire(value: u32) -> Result<Self, PresentError> {
        match value {
            0 => Ok(Self::Immediate),
            1 => Ok(Self::Vsync),
            other => Err(PresentError::UnsupportedSyncInterval {
                sync_interval: other,
            }),
        }
    }

    /// The wire representation (`0` or `1`).
    pub const fn as_wire(self) -> u32 {
        match self {
            Self::Immediate => 0,
            Self::Vsync => 1,
        }
    }
}

/// The payload of `victory_metal::wire_v3::Present`, mirrored locally.
///
/// The field names and widths are identical to the wire type so the producer
/// can hand a decoded present straight to [`translate_present`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainPresent {
    /// Swapchain/IOSurface handle.
    pub surface_id: u64,
    /// Sync interval (`0` = immediate, `1` = vsync).
    pub sync_interval: u32,
}

impl SwapchainPresent {
    /// Builds a present payload from decoded wire fields.
    pub const fn new(surface_id: u64, sync_interval: u32) -> Self {
        Self {
            surface_id,
            sync_interval,
        }
    }
}

/// A validated present step for one swapchain surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentStep {
    /// Surface the frame is presented from.
    pub surface_id: u64,
    /// Validated pacing.
    pub sync_interval: SyncInterval,
}

/// A frame handed to a [`SurfacePresenter`].
///
/// Carries identity and pacing only. It deliberately has no pixel bytes so a
/// presenter cannot accidentally introduce a readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentedSurface {
    /// Surface that holds the finished frame.
    pub surface_id: u64,
    /// Monotonic presentation index, starting at `0` for the swapchain.
    pub frame_index: u64,
    /// Nanoseconds since the [`NativeSwapchain`] was created.
    pub timestamp_ns: u64,
    /// Pacing the guest requested for this frame.
    pub sync_interval: SyncInterval,
}

/// Host-side presenter implemented over the native Metal present path.
///
/// The host adapter wraps `victory_metal::macos::present::PresentPath` and
/// forwards the presented surface to its `PresentSink` (or to a windowed
/// `CAMetalLayer`). A refused frame returns an error and takes no ownership;
/// the presenter clears its in-flight state and returns the error unchanged.
pub trait SurfacePresenter {
    /// Accepts one finished frame.
    fn present(&mut self, frame: PresentedSurface) -> Result<(), PresentError>;
}

/// One frame held by the swapchain until the host finishes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InFlightFrame {
    /// Surface that holds the frame.
    pub surface_id: u64,
    /// Presentation index of the frame.
    pub frame_index: u64,
    /// Pacing requested for the frame.
    pub sync_interval: SyncInterval,
}

/// Native Metal swapchain state for one surface.
///
/// Constructible headlessly with [`Self::new_headless`], so tests can drive the
/// whole state machine without a Metal device. Production construction also
/// validates that the surface was created by the Metal path.
#[derive(Debug)]
pub struct NativeSwapchain {
    surface_id: u64,
    width: u32,
    height: u32,
    format: SurfaceFormat,
    frame_index: u64,
    in_flight: Option<InFlightFrame>,
    sync_interval: SyncInterval,
    started: Instant,
}

impl NativeSwapchain {
    /// Registers a surface created by `created_by` for native Metal presentation.
    ///
    /// Refuses a surface from any other backend with
    /// [`PresentError::BackendMismatch`] instead of falling back, and rejects
    /// degenerate or oversized extents before any host call.
    pub fn new(
        surface_id: u64,
        width: u32,
        height: u32,
        format: SurfaceFormat,
        created_by: GpuBackend,
    ) -> Result<Self, PresentError> {
        if created_by != GpuBackend::Metal {
            return Err(PresentError::BackendMismatch {
                surface_id,
                actual: created_by.as_str(),
            });
        }
        Self::validate_extent(width, height)?;
        Ok(Self {
            surface_id,
            width,
            height,
            format,
            frame_index: 0,
            in_flight: None,
            sync_interval: SyncInterval::Immediate,
            started: Instant::now(),
        })
    }

    /// Registers a BGRA8 surface as created by the Metal path.
    ///
    /// Headless: touches no Metal object, so tests run in a windowless and
    /// device-less session.
    pub fn new_headless(surface_id: u64, width: u32, height: u32) -> Result<Self, PresentError> {
        Self::new(
            surface_id,
            width,
            height,
            SurfaceFormat::Bgra8Unorm,
            GpuBackend::Metal,
        )
    }

    fn validate_extent(width: u32, height: u32) -> Result<(), PresentError> {
        if width == 0 || height == 0 {
            return Err(PresentError::ZeroExtent { width, height });
        }
        if width > MAX_SURFACE_DIMENSION || height > MAX_SURFACE_DIMENSION {
            return Err(PresentError::ExtentTooLarge { width, height });
        }
        Ok(())
    }

    /// `IOSurface`/swapchain identity this state belongs to.
    pub fn surface_id(&self) -> u64 {
        self.surface_id
    }

    /// Back-buffer width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Back-buffer height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Back-buffer pixel format.
    pub fn format(&self) -> SurfaceFormat {
        self.format
    }

    /// Index the next presented frame will receive.
    pub fn next_frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Pacing of the most recently presented frame.
    pub fn sync_interval(&self) -> SyncInterval {
        self.sync_interval
    }

    /// The frame currently in flight, if any.
    pub fn in_flight(&self) -> Option<InFlightFrame> {
        self.in_flight
    }

    /// Marks the in-flight frame as finished and releases the surface.
    ///
    /// This is the completion signal the host sends after the window server
    /// has consumed the surface.
    pub fn finish_present(&mut self) -> Result<InFlightFrame, PresentError> {
        self.in_flight.take().ok_or(PresentError::NoFrameInFlight)
    }
}

/// Errors from native swapchain construction and presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentError {
    /// A surface edge was zero.
    ZeroExtent {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    /// A surface edge exceeded the supported host 2D limit.
    ExtentTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    /// Native Metal presentation was selected for a surface from another path.
    BackendMismatch {
        /// Surface that was refused.
        surface_id: u64,
        /// Backend that created the surface.
        actual: &'static str,
    },
    /// A present named a surface this swapchain does not own.
    UnknownSurface {
        /// Surface from the present command.
        requested: u64,
        /// Surface registered with the swapchain.
        known: u64,
    },
    /// The present named a sync interval other than immediate or vsync.
    UnsupportedSyncInterval {
        /// Requested sync interval.
        sync_interval: u32,
    },
    /// A frame is still in flight; a second present would overwrite it.
    FrameInFlight {
        /// In-flight frame index.
        frame_index: u64,
        /// In-flight surface identity.
        surface_id: u64,
    },
    /// No frame was in flight when completion was requested.
    NoFrameInFlight,
    /// The host presenter refused the frame; the frame was not accepted.
    SinkRefused {
        /// Refused frame index.
        frame_index: u64,
        /// Presenter-provided refusal reason.
        reason: String,
    },
}

impl std::fmt::Display for PresentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroExtent { width, height } => {
                write!(
                    f,
                    "surface width and height must be nonzero (got {width}x{height})"
                )
            }
            Self::ExtentTooLarge { width, height } => {
                write!(
                    f,
                    "surface extent {width}x{height} exceeds the supported 2D limit"
                )
            }
            Self::BackendMismatch { surface_id, actual } => write!(
                f,
                "native Metal presentation cannot own surface {surface_id}: it was created by the {actual} path"
            ),
            Self::UnknownSurface { requested, known } => write!(
                f,
                "present names unknown surface {requested}; this swapchain owns {known}"
            ),
            Self::UnsupportedSyncInterval { sync_interval } => write!(
                f,
                "unsupported present sync interval {sync_interval}; only 0 (immediate) and 1 (vsync) are supported"
            ),
            Self::FrameInFlight {
                frame_index,
                surface_id,
            } => write!(
                f,
                "frame {frame_index} on surface {surface_id} is still in flight; finish it before presenting again"
            ),
            Self::NoFrameInFlight => write!(f, "no frame is currently in flight"),
            Self::SinkRefused {
                frame_index,
                reason,
            } => write!(f, "present sink refused frame {frame_index}: {reason}"),
        }
    }
}

impl std::error::Error for PresentError {}

/// Translates a wire present command into a validated [`PresentStep`].
///
/// Rejects a surface the swapchain does not own and any sync interval above
/// `1` with an explicit error; no partial state is left behind.
pub fn translate_present(
    swapchain: &NativeSwapchain,
    present: SwapchainPresent,
) -> Result<PresentStep, PresentError> {
    if present.surface_id != swapchain.surface_id {
        return Err(PresentError::UnknownSurface {
            requested: present.surface_id,
            known: swapchain.surface_id,
        });
    }
    let sync_interval = SyncInterval::from_wire(present.sync_interval)?;
    Ok(PresentStep {
        surface_id: present.surface_id,
        sync_interval,
    })
}

/// Hands one finished frame to the host presenter.
///
/// The guest has already drawn into the surface; this function only assigns
/// the frame identity, records the in-flight frame and delegates to
/// `presenter`. It never reads pixels, copies surface memory or locks the
/// surface. A presenter error clears the in-flight state and is propagated
/// unchanged; the frame index is still consumed so indexes stay monotonic.
pub fn present_surface(
    swapchain: &mut NativeSwapchain,
    step: PresentStep,
    presenter: &mut dyn SurfacePresenter,
) -> Result<PresentedSurface, PresentError> {
    if step.surface_id != swapchain.surface_id {
        return Err(PresentError::UnknownSurface {
            requested: step.surface_id,
            known: swapchain.surface_id,
        });
    }
    if let Some(inflight) = swapchain.in_flight {
        return Err(PresentError::FrameInFlight {
            frame_index: inflight.frame_index,
            surface_id: inflight.surface_id,
        });
    }

    let frame_index = swapchain.frame_index;
    let timestamp_ns = u64::try_from(swapchain.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let frame = PresentedSurface {
        surface_id: step.surface_id,
        frame_index,
        timestamp_ns,
        sync_interval: step.sync_interval,
    };

    swapchain.frame_index = frame_index.wrapping_add(1);
    swapchain.sync_interval = step.sync_interval;
    swapchain.in_flight = Some(InFlightFrame {
        surface_id: frame.surface_id,
        frame_index: frame.frame_index,
        sync_interval: frame.sync_interval,
    });

    if let Err(error) = presenter.present(frame) {
        // The presenter did not take ownership: nothing is in flight.
        swapchain.in_flight = None;
        return Err(error);
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct TestPresenter {
        frames: Vec<PresentedSurface>,
        refuse_next: bool,
    }

    impl TestPresenter {
        fn refuse_next_frame(&mut self) {
            self.refuse_next = true;
        }
    }

    impl SurfacePresenter for TestPresenter {
        fn present(&mut self, frame: PresentedSurface) -> Result<(), PresentError> {
            if self.refuse_next {
                self.refuse_next = false;
                return Err(PresentError::SinkRefused {
                    frame_index: frame.frame_index,
                    reason: "test presenter was asked to refuse this frame".to_string(),
                });
            }
            self.frames.push(frame);
            Ok(())
        }
    }

    fn step(surface_id: u64, sync_interval: SyncInterval) -> PresentStep {
        PresentStep {
            surface_id,
            sync_interval,
        }
    }

    #[test]
    fn headless_swapchain_starts_with_no_frames_in_flight() {
        let swapchain =
            NativeSwapchain::new_headless(11, 640, 360).expect("headless swapchain creation");
        assert_eq!(swapchain.surface_id(), 11);
        assert_eq!(swapchain.width(), 640);
        assert_eq!(swapchain.height(), 360);
        assert_eq!(swapchain.format(), SurfaceFormat::Bgra8Unorm);
        assert_eq!(swapchain.next_frame_index(), 0);
        assert_eq!(swapchain.sync_interval(), SyncInterval::Immediate);
        assert_eq!(swapchain.in_flight(), None);
    }

    #[test]
    fn degenerate_or_oversized_surfaces_are_rejected_headlessly() {
        assert_eq!(
            NativeSwapchain::new_headless(1, 0, 64).err(),
            Some(PresentError::ZeroExtent {
                width: 0,
                height: 64
            })
        );
        assert_eq!(
            NativeSwapchain::new_headless(1, MAX_SURFACE_DIMENSION + 1, 64).err(),
            Some(PresentError::ExtentTooLarge {
                width: MAX_SURFACE_DIMENSION + 1,
                height: 64
            })
        );
    }

    #[test]
    fn a_surface_from_another_backend_is_an_explicit_mismatch() {
        let error = NativeSwapchain::new(7, 64, 64, SurfaceFormat::Bgra8Unorm, GpuBackend::D3d12)
            .expect_err("a non-Metal surface must be refused");
        assert_eq!(
            error,
            PresentError::BackendMismatch {
                surface_id: 7,
                actual: GpuBackend::D3d12.as_str(),
            }
        );
    }

    #[test]
    fn translation_rejects_unknown_surfaces_and_bad_intervals() {
        let swapchain =
            NativeSwapchain::new_headless(11, 64, 64).expect("headless swapchain creation");

        assert_eq!(
            translate_present(&swapchain, SwapchainPresent::new(12, 0)),
            Err(PresentError::UnknownSurface {
                requested: 12,
                known: 11,
            })
        );
        assert_eq!(
            translate_present(&swapchain, SwapchainPresent::new(11, 2)),
            Err(PresentError::UnsupportedSyncInterval { sync_interval: 2 })
        );
        assert_eq!(
            translate_present(&swapchain, SwapchainPresent::new(11, 1)),
            Ok(step(11, SyncInterval::Vsync))
        );
        assert_eq!(
            translate_present(&swapchain, SwapchainPresent::new(11, 0)),
            Ok(step(11, SyncInterval::Immediate))
        );
    }

    #[test]
    fn presents_are_one_in_flight_and_a_refusal_is_explicit() {
        let mut swapchain =
            NativeSwapchain::new_headless(3, 32, 16).expect("headless swapchain creation");
        let mut presenter = TestPresenter::default();

        let frame0 = present_surface(&mut swapchain, step(3, SyncInterval::Vsync), &mut presenter)
            .expect("frame 0 should be accepted");
        assert_eq!(frame0.frame_index, 0);
        assert_eq!(frame0.surface_id, 3);
        assert_eq!(frame0.sync_interval, SyncInterval::Vsync);
        assert_eq!(swapchain.next_frame_index(), 1);
        assert_eq!(
            swapchain.in_flight(),
            Some(InFlightFrame {
                surface_id: 3,
                frame_index: 0,
                sync_interval: SyncInterval::Vsync,
            })
        );

        assert_eq!(
            present_surface(&mut swapchain, step(3, SyncInterval::Vsync), &mut presenter),
            Err(PresentError::FrameInFlight {
                frame_index: 0,
                surface_id: 3,
            }),
            "only one frame may be in flight at a time"
        );
        assert_eq!(
            presenter.frames.len(),
            1,
            "refused frame must not be accepted"
        );

        let finished = swapchain.finish_present().expect("frame 0 should finish");
        assert_eq!(finished.frame_index, 0);
        assert_eq!(
            swapchain.finish_present(),
            Err(PresentError::NoFrameInFlight),
            "finishing twice must be an explicit error"
        );

        let frame1 = present_surface(
            &mut swapchain,
            step(3, SyncInterval::Immediate),
            &mut presenter,
        )
        .expect("frame 1 should be accepted");
        assert_eq!(frame1.frame_index, 1);
        assert_eq!(swapchain.sync_interval(), SyncInterval::Immediate);
        assert_eq!(presenter.frames.len(), 2);
        swapchain.finish_present().expect("frame 1 should finish");

        // A refused presenter returns an explicit error and leaves nothing in
        // flight; the frame index is still consumed, matching the host
        // presenter whose surface already holds the rendered frame.
        presenter.refuse_next_frame();
        let error = present_surface(&mut swapchain, step(3, SyncInterval::Vsync), &mut presenter)
            .expect_err("a refusing presenter must surface an explicit error");
        assert_eq!(
            error,
            PresentError::SinkRefused {
                frame_index: 2,
                reason: "test presenter was asked to refuse this frame".to_string(),
            }
        );
        assert_eq!(swapchain.in_flight(), None);
        assert_eq!(
            presenter.frames.len(),
            2,
            "refused frame must not be recorded"
        );

        let frame3 = present_surface(&mut swapchain, step(3, SyncInterval::Vsync), &mut presenter)
            .expect("present after refusal should succeed");
        assert_eq!(frame3.frame_index, 3, "a refused frame consumes its index");
        assert_eq!(presenter.frames.len(), 3);
    }

    #[test]
    fn unknown_surface_is_refused_at_presentation_time_too() {
        let mut swapchain =
            NativeSwapchain::new_headless(5, 16, 16).expect("headless swapchain creation");
        let mut presenter = TestPresenter::default();
        assert_eq!(
            present_surface(
                &mut swapchain,
                step(6, SyncInterval::Immediate),
                &mut presenter
            ),
            Err(PresentError::UnknownSurface {
                requested: 6,
                known: 5,
            })
        );
        assert_eq!(presenter.frames.len(), 0);
        assert_eq!(swapchain.in_flight(), None);
        assert_eq!(swapchain.next_frame_index(), 0);
    }
}
