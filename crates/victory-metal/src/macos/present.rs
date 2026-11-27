//! Native ARM64 presentation of rendered frames, without D3DMetal and
//! without a CPU readback in the present path.
//!
//! Scope of this module:
//!
//! - A `PresentSink` abstraction the host can drive from a headless test and
//!   from a window-backed host, so presentation has one code path.
//! - An IOSurface-backed swapchain image: the renderer draws into the
//!   surface's `MTLTexture`, and handing the surface to the sink is the only
//!   cross-process step.
//! - Frame identity and pacing: monotonic frame index, one in-flight present
//!   at a time, explicit error when the sink refuses a frame.
//!
//! Reading pixels back is a test-only verification step, never part of the
//! production present path. Production code never messages `getBytes`, never
//! memcpys surface memory and never locks an IOSurface; a test may lock the
//! surface once after presentation (see `SurfaceImage::read_bgra_rows`).
//!
//! # Windowed host
//!
//! A windowed host reuses this whole path. After a surface is finished, the
//! single call site that hands it to the window server is:
//!
//! ```ignore
//! // WINDOWED HOST ATTACH POINT — the only place a surface reaches the
//! // window server. `layer` is a `CAMetalLayer` with `pixelFormat` set to
//! // BGRA8Unorm and `framebufferOnly` set to NO.
//! layer.setIOSurface(image.iosurface_raw());
//! // The next `nextDrawable`/commit presents the surface; the host then
//! // signals completion of the frame to the swapchain.
//! ```
//!
//! No window server object is created by this module, so the headless test
//! path works in a windowless session.

use std::ffi::c_void;
use std::mem;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};

/// `MTLPixelFormatBGRA8Unorm`, matching the IOSurface `'BGRA'` format below.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
/// `MTLPixelFormatRGBA8Unorm`, the AAA frame colour-target format.
const MTL_PIXEL_FORMAT_RGBA8_UNORM: usize = 70;
/// `MTLStorageModeShared`; IOSurface-backed textures are CPU/GPU shared.
const MTL_STORAGE_MODE_SHARED: usize = 0;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLTextureUsageRenderTarget`.
const MTL_TEXTURE_USAGE_RENDER_TARGET: usize = 4;
/// `MTLLoadActionClear`.
const MTL_LOAD_ACTION_CLEAR: usize = 2;
/// `MTLStoreActionStore`.
const MTL_STORE_ACTION_STORE: usize = 1;
/// `MTLCommandBufferStatusCompleted`.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// `'BGRA'` four-character code in the big-endian layout IOSurface expects.
const IOSURFACE_PIXEL_FORMAT_BGRA: i32 = 0x4247_5241;
/// `kIOSurfaceLockReadOnly`. Test-only: production never locks a surface.
#[cfg(test)]
const IOSURFACE_LOCK_READ_ONLY: u32 = 1;
/// Every supported pixel is one BGRA8 sample.
const BYTES_PER_PIXEL: u32 = 4;
/// Row pitch alignment. IOSurface/Metal require a padded pitch for BGRA8, and
/// 64 bytes satisfies the Apple-silicon linear-texture alignment on all
/// devices this renderer targets; the actual pitch is re-read from the surface.
const SURFACE_ROW_ALIGNMENT: usize = 64;
/// Apple's maximum 2D texture edge.
const MAX_SURFACE_DIMENSION: u32 = 16_384;

/// Frame pacing requested for one presented frame.
///
/// This mirrors the guest's present sync interval: `0` is immediate and `1` is
/// vertical sync. The headless presenter validates and forwards the value; a
/// windowed host maps it onto its drawable presentation policy.
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
    /// Only `0` and `1` are supported; any other value is an explicit
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

/// A presented frame, as seen by a [`PresentSink`].
///
/// Carries identity and pacing only: the surface's `IOSurfaceID`, the
/// monotonic frame index, the host timestamp and the requested sync interval.
/// It deliberately carries no pixel bytes, so a sink cannot accidentally
/// introduce a readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentedFrame {
    /// `IOSurfaceGetID` of the surface that holds this frame.
    pub surface_id: u32,
    /// Monotonically increasing presentation index, starting at `0`.
    pub frame_index: u64,
    /// Nanoseconds since the [`PresentPath`] was created (monotonic clock).
    pub timestamp_ns: u64,
    /// Pacing the guest requested for this frame.
    pub sync_interval: SyncInterval,
}

/// A sink that accepts finished surfaces.
///
/// The headless test uses [`RecordingSink`]; a windowed host would set the
/// surface on a `CAMetalLayer` (see the module documentation). A sink that
/// returns an error has not taken ownership of that frame: the presenter
/// clears its in-flight state and propagates the error unchanged. A sink must
/// not retain the frame when it returns `Err`.
pub trait PresentSink {
    /// Accepts one finished frame.
    fn present(&mut self, frame: PresentedFrame) -> Result<(), PresentError>;
}

/// Reference recording sink for tests and headless diagnostics.
///
/// Records every frame it accepts and can refuse the next frame on demand so
/// the refusal path has a documented, deterministic error.
#[derive(Debug, Default)]
pub struct RecordingSink {
    frames: Vec<PresentedFrame>,
    refuse_next: bool,
}

impl RecordingSink {
    /// Creates an empty recording sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames accepted so far, in presentation order.
    pub fn frames(&self) -> &[PresentedFrame] {
        &self.frames
    }

    /// Makes the next [`PresentSink::present`] call fail with
    /// [`PresentError::SinkRefused`] and accept nothing.
    pub fn refuse_next_frame(&mut self) {
        self.refuse_next = true;
    }

    /// The documented refusal error for the next frame.
    fn refusal(frame: PresentedFrame) -> PresentError {
        PresentError::SinkRefused {
            frame_index: frame.frame_index,
            reason: "test sink was asked to refuse this frame".to_string(),
        }
    }
}

impl PresentSink for RecordingSink {
    fn present(&mut self, frame: PresentedFrame) -> Result<(), PresentError> {
        if self.refuse_next {
            self.refuse_next = false;
            return Err(Self::refusal(frame));
        }
        self.frames.push(frame);
        Ok(())
    }
}

/// A `CAMetalLayer` drawable pool driven by the native window presenter.
///
/// This is the only type in the module that talks to the window server. It
/// owns the retained layer and the transport queue, acquires drawables with
/// `nextDrawable` and presents them with `presentDrawable:`. Pixels never
/// pass through the CPU: a finished frame is copied into the drawable's
/// texture with a GPU blit when both textures are BGRA8 and the same size;
/// otherwise the drawable is cleared to a diagnostic colour. Either way the
/// drawable texture is the surface the window server presents.
///
/// The type is deliberately not part of the headless `PresentSink` path: the
/// headless presenter never constructs one, so a windowless session cannot
/// depend on AppKit or on a window server.
pub struct NativeSwapchain {
    layer: ObjcObject,
    queue: ObjcObject,
    current: Option<ObjcObject>,
    serial: u64,
    presented: u64,
    width: u32,
    height: u32,
    /// Opt-in presented-frame probe. `None` (the default) adds no command
    /// buffer, no blit, no readback and no allocation to the present path.
    probe: Option<FrameProbe>,
}

impl NativeSwapchain {
    /// Wraps a retained `CAMetalLayer` and retains `queue` for presentation.
    ///
    /// `queue` is normally the transport's queue, so the blit that fills the
    /// drawable is ordered after the frame's render work once that work has
    /// completed. Returns `None` when the queue cannot be retained.
    pub(crate) fn new(layer: ObjcObject, queue: objc::Id, width: u32, height: u32) -> Option<Self> {
        // SAFETY: `queue` is a live `MTLCommandQueue`; `retain` returns it +1.
        let retained = unsafe { objc::msg_send_id_0(queue, objc::sel(c"retain")) };
        let queue = ObjcObject::from_retained(retained)?;
        // The probe is the only extra state; with the environment unset this is
        // a single environment read and every field below stays `None`.
        let probe = FrameProbe::from_env(layer.as_raw());
        Some(Self {
            layer,
            queue,
            current: None,
            serial: 0,
            presented: 0,
            width,
            height,
            probe,
        })
    }

    /// Sets the layer's drawable pixel format before the next acquisition.
    ///
    /// The window opens with BGRA8 (the documented default); a present surface
    /// may be RGBA8, in which case the host switches the layer so the GPU blit
    /// can copy it without a CPU swizzle. No drawable is in flight between
    /// frames, so changing the format is safe.
    pub(crate) fn set_pixel_format(&mut self, format: usize) {
        // SAFETY: `layer` is a live `CAMetalLayer` and `format` is a valid
        // `MTLPixelFormat` value chosen by the caller.
        unsafe {
            msg_send_void_usize(self.layer.as_raw(), objc::sel(c"setPixelFormat:"), format);
        }
    }

    /// Drawables acquired so far.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Drawables presented so far.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// Acquires the next drawable and retains it until [`Self::present`].
    ///
    /// Returns the monotonic drawable serial, or `None` when the layer cannot
    /// supply a drawable (for example a zero-sized layer).
    pub fn next_drawable(&mut self) -> Option<u64> {
        let _pool = AutoreleasePool::new();
        // SAFETY: `layer` is a live `CAMetalLayer`; `nextDrawable` returns an
        // autoreleased `id<CAMetalDrawable>` or nil.
        let raw = unsafe { objc::msg_send_id_0(self.layer.as_raw(), objc::sel(c"nextDrawable")) };
        if raw.is_null() {
            return None;
        }
        // SAFETY: `raw` is a live drawable; `retain` keeps it alive until the
        // matching release in `present`/`Drop`.
        let retained = unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) };
        self.current = ObjcObject::from_retained(retained);
        self.current.as_ref()?;
        self.serial = self.serial.wrapping_add(1);
        Some(self.serial)
    }

    /// Presents the acquired drawable, optionally copying `source` into it.
    ///
    /// The blit is GPU-only. A format or extent mismatch falls back to a
    /// diagnostic clear so the window still shows live, non-black content
    /// instead of stale memory.
    pub fn present(&mut self, source: Option<objc::Id>) -> Result<(), PresentError> {
        let drawable = self.current.take().ok_or_else(|| {
            PresentError::Metal("no CAMetalLayer drawable was acquired".to_string())
        })?;
        let _pool = AutoreleasePool::new();

        // SAFETY: `drawable` is a live `CAMetalDrawable`; `texture` returns its
        // live `id<MTLTexture>`.
        let target = unsafe { objc::msg_send_id_0(drawable.as_raw(), objc::sel(c"texture")) };
        if target.is_null() {
            return Err(PresentError::Metal("drawable texture is nil".to_string()));
        }

        // SAFETY: `queue` is the retained transport queue; `commandBuffer`
        // returns an autoreleased buffer that lives in this pool.
        let command_buffer =
            unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if command_buffer.is_null() {
            return Err(PresentError::Metal(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }

        let mut filled = false;
        if let Some(source) = source {
            // SAFETY: `source` is a live `MTLTexture` retained by the caller;
            // the accessors only read immutable properties.
            let (source_format, target_format, source_width, source_height) = unsafe {
                (
                    objc::msg_send_usize_0(source, objc::sel(c"pixelFormat")),
                    objc::msg_send_usize_0(target, objc::sel(c"pixelFormat")),
                    objc::msg_send_usize_0(source, objc::sel(c"width")),
                    objc::msg_send_usize_0(source, objc::sel(c"height")),
                )
            };
            // BGRA8 is the layer default; RGBA8 is accepted after
            // `set_pixel_format` switched the layer to match the surface.
            if source_format == target_format
                && matches!(
                    source_format,
                    MTL_PIXEL_FORMAT_BGRA8_UNORM | MTL_PIXEL_FORMAT_RGBA8_UNORM
                )
                && source_width == self.width as usize
                && source_height == self.height as usize
            {
                encode_drawable_blit(command_buffer, source, target, self.width, self.height)?;
                filled = true;
            }
        }
        if !filled {
            encode_drawable_clear(command_buffer, target)?;
        }

        // SAFETY: `command_buffer` is an uncommitted buffer carrying either one
        // blit or one clear pass into the drawable texture; `drawable` is the
        // acquired window-server drawable.
        unsafe {
            objc::msg_send_void_1(
                command_buffer,
                objc::sel(c"presentDrawable:"),
                drawable.as_raw(),
            );
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
        }

        // SAFETY: The command buffer completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: `[command_buffer error]` is null or an autoreleased `NSError *`.
            let error = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
            // SAFETY: `error` is null or a live `NSError *` inside this pool.
            let message = unsafe { objc::nserror_message(error) };
            return Err(PresentError::Metal(format!(
                "drawable present failed (status={status}): {message}"
            )));
        }

        // The drawable now holds exactly the pixels the window server shows:
        // either the GPU blit of `source` or the diagnostic clear. Sampling it
        // after `waitUntilCompleted` is the only point where "presented" and
        // "readable" coincide. With the probe off, `probe` is `None` and this
        // block does nothing.
        if let Some(probe) = self.probe.as_mut() {
            // `filled == false` means the drawable was cleared to the known
            // diagnostic colour, so the probe can use that as its clear value.
            probe.observe(self.queue.as_raw(), target, self.presented, !filled);
        }

        self.presented = self.presented.wrapping_add(1);
        Ok(())
    }
}

/// `MTLOrigin` (`NSUInteger x, y, z`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// Copies a same-format texture into the drawable texture on the GPU.
fn encode_drawable_blit(
    command_buffer: objc::Id,
    source: objc::Id,
    target: objc::Id,
    width: u32,
    height: u32,
) -> Result<(), PresentError> {
    // SAFETY: `command_buffer` is a live uncommitted buffer; `blitCommandEncoder`
    // returns an autoreleased encoder.
    let encoder = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder")) };
    if encoder.is_null() {
        return Err(PresentError::Metal(
            "MTLCommandBuffer.blitCommandEncoder returned null".to_string(),
        ));
    }
    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
    let size = objc::MTLSize {
        width: width as usize,
        height: height as usize,
        depth: 1,
    };
    // SAFETY: Both textures are live, single-slice level-0 BGRA8 textures with
    // exactly `width x height` pixels, which the selector's origin/size
    // arguments describe.
    unsafe {
        type Fn = unsafe extern "C" fn(
            objc::Id,
            objc::Sel,
            objc::Id,
            usize,
            usize,
            MTLOrigin,
            objc::MTLSize,
            objc::Id,
            usize,
            usize,
            MTLOrigin,
        );
        let send: Fn = mem::transmute(objc::msg_send_entry());
        send(
            encoder,
            objc::sel(
                c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            ),
            source,
            0,
            0,
            origin,
            size,
            target,
            0,
            0,
            origin,
        );
        objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
    }
    Ok(())
}

/// Clears the drawable texture to an opaque diagnostic colour.
fn encode_drawable_clear(command_buffer: objc::Id, texture: objc::Id) -> Result<(), PresentError> {
    let class = objc::class(c"MTLRenderPassDescriptor");
    if class.is_null() {
        return Err(PresentError::Metal(
            "MTLRenderPassDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `class` is the descriptor class and `renderPassDescriptor` is its
    // documented factory, returning an autoreleased descriptor.
    let descriptor = unsafe { objc::msg_send_id_0(class, objc::sel(c"renderPassDescriptor")) };
    if descriptor.is_null() {
        return Err(PresentError::Metal(
            "MTLRenderPassDescriptor creation returned null".to_string(),
        ));
    }
    // SAFETY: `descriptor` is a live render pass descriptor; index 0 is a
    // colour attachment slot.
    let attachments = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    let attachment =
        unsafe { msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0) };
    if attachment.is_null() {
        return Err(PresentError::Metal(
            "drawable colour attachment 0 is null".to_string(),
        ));
    }
    // SAFETY: `attachment` is a live colour attachment descriptor.
    unsafe {
        objc::msg_send_void_1(attachment, objc::sel(c"setTexture:"), texture);
        msg_send_void_usize(
            attachment,
            objc::sel(c"setLoadAction:"),
            MTL_LOAD_ACTION_CLEAR,
        );
        msg_send_void_usize(
            attachment,
            objc::sel(c"setStoreAction:"),
            MTL_STORE_ACTION_STORE,
        );
        msg_send_void_clear_color(
            attachment,
            objc::sel(c"setClearColor:"),
            MTLClearColor {
                red: 0.05,
                green: 0.10,
                blue: 0.20,
                alpha: 1.0,
            },
        );
    }
    // SAFETY: `command_buffer` is uncommitted and `descriptor` is a fully
    // populated render pass descriptor targeting the drawable texture.
    let encoder = unsafe {
        objc::msg_send_id_1(
            command_buffer,
            objc::sel(c"renderCommandEncoderWithDescriptor:"),
            descriptor,
        )
    };
    if encoder.is_null() {
        return Err(PresentError::Metal(
            "drawable renderCommandEncoderWithDescriptor: returned null".to_string(),
        ));
    }
    // SAFETY: `encoder` is an active render encoder; ending it records the clear.
    unsafe {
        objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Opt-in presented-frame probe
// ---------------------------------------------------------------------------
//
// A successful `presentDrawable:` and a perfectly blank clear look identical
// from the outside. The probe closes that gap without becoming part of the
// production path: it is off unless `VICTORY_METAL_FRAME_PROBE` says otherwise,
// it samples the drawable only on every Nth presented frame, it copies a fixed
// 8x8 grid of 8x8 blocks into one 64x64 shared staging texture (16 KiB), and it
// never waits on the GPU for longer than `FRAME_PROBE_WAIT`. Exactly one probe
// command buffer may be in flight; further opportunities are counted and
// dropped. The probe never writes pixels to disk: only counters and a CRC reach
// telemetry and the host log.

/// Environment variable that opts into the presented-frame probe.
pub const FRAME_PROBE_ENV: &str = "VICTORY_METAL_FRAME_PROBE";
/// Environment variable that overrides the sampling interval.
pub const FRAME_PROBE_EVERY_ENV: &str = "VICTORY_METAL_FRAME_PROBE_EVERY";
/// Default sampling interval: one probe every 30 presented frames.
pub const FRAME_PROBE_DEFAULT_EVERY: u64 = 30;
/// A channel counts as content when it differs from the clear value by more
/// than this many 8-bit steps (out of 255). The comparison is strict: a
/// difference of exactly 8 is still considered clear.
pub const FRAME_PROBE_CHANNEL_THRESHOLD: u8 = 8;
/// The probe copies an 8x8 grid of blocks.
pub const FRAME_PROBE_GRID: u32 = 8;
/// Each sampled block is 8x8 pixels, so the staging texture is 64x64 pixels
/// (16 KiB at BGRA8) regardless of the drawable size.
pub const FRAME_PROBE_BLOCK: u32 = 8;
/// One probe measures exactly this many pixels (`8 * 8 * 8 * 8`).
pub const FRAME_PROBE_PIXELS: u64 = (FRAME_PROBE_GRID as u64)
    * (FRAME_PROBE_BLOCK as u64)
    * (FRAME_PROBE_GRID as u64)
    * (FRAME_PROBE_BLOCK as u64);
/// How long one `present` may block waiting for the probe blit.
pub const FRAME_PROBE_WAIT: Duration = Duration::from_millis(10);
/// After this long an in-flight probe is abandoned and a named failure is
/// recorded instead of blocking teardown forever.
pub const FRAME_PROBE_MAX_IN_FLIGHT: Duration = Duration::from_secs(2);

/// Process-wide last probe report, read by the host for telemetry.
static FRAME_PROBE_REPORT: OnceLock<Mutex<Option<FrameProbeReport>>> = OnceLock::new();

fn publish_frame_probe_report(report: &FrameProbeReport) {
    let cell = FRAME_PROBE_REPORT.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = cell.lock() {
        *slot = Some(report.clone());
    }
}

/// The most recent presented-frame probe report, or `None` when the probe has
/// never run in this process (the default).
pub fn frame_probe_report() -> Option<FrameProbeReport> {
    FRAME_PROBE_REPORT
        .get()
        .and_then(|cell| cell.lock().ok())
        .and_then(|slot| slot.clone())
}

/// Probe opt-in and sampling interval parsed from the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameProbeConfig {
    /// Sample every `every` presented frames; frame 0 is always sampled.
    pub every: u64,
}

impl FrameProbeConfig {
    /// Parses the two environment values.
    ///
    /// `None` means the probe is off; an unset or invalid `every` falls back to
    /// [`FRAME_PROBE_DEFAULT_EVERY`] so a malformed override can never disable
    /// the explicit opt-in.
    pub fn from_values(enabled: Option<&str>, every: Option<&str>) -> Option<Self> {
        if !enabled.is_some_and(probe_flag_enabled) {
            return None;
        }
        let every = every
            .and_then(parse_probe_every)
            .unwrap_or(FRAME_PROBE_DEFAULT_EVERY);
        Some(Self { every })
    }

    /// Reads the configuration from the process environment.
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var(FRAME_PROBE_ENV).ok();
        let every = std::env::var(FRAME_PROBE_EVERY_ENV).ok();
        Self::from_values(enabled.as_deref(), every.as_deref())
    }
}

fn probe_flag_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "on" | "true" | "yes"
    )
}

/// Parses a positive sampling interval; `0` and non-numbers are rejected.
pub fn parse_probe_every(value: &str) -> Option<u64> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|interval| *interval > 0)
}

/// Whether presented frame `frame_index` should be probed.
///
/// Frame 0 (the first presented frame) and every `every`-th frame afterwards
/// are sampled. `every == 0` is inert rather than a division by zero.
pub fn probe_due(frame_index: u64, every: u64) -> bool {
    every > 0 && frame_index.is_multiple_of(every)
}

/// Byte order of a sampled 4-channel frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelOrder {
    /// Blue, green, red, alpha (`MTLPixelFormatBGRA8Unorm`).
    Bgra,
    /// Red, green, blue, alpha (`MTLPixelFormatRGBA8Unorm`).
    Rgba,
}

/// One probe measurement over a sampled BGRA8/RGBA8 buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameProbeSample {
    /// Pixels with any channel farther than the threshold from the clear value.
    pub non_black_pixels: u64,
    /// Distinct 5-bit luma buckets seen (0..=32).
    pub unique_luma_buckets: u32,
    /// CRC-64/ECMA-182 over the sampled pixels in row-major order.
    pub frame_crc: u64,
    /// Number of pixels in this measurement.
    pub probe_pixels: u64,
}

/// The accumulated probe result reported in telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameProbeReport {
    /// Sampled frames that were actually read back.
    pub sampled_frames: u64,
    /// Pixels measured per sample.
    pub probe_pixels: u64,
    /// Frame index of the last successful sample, if any.
    pub last_sampled_frame: Option<u64>,
    /// Non-black pixels in the last sample.
    pub non_black_pixels: u64,
    /// Distinct 5-bit luma buckets in the last sample.
    pub unique_luma_buckets: u32,
    /// CRC-64/ECMA-182 of the last sample.
    pub frame_crc: u64,
    /// Sample opportunities dropped because a probe was still in flight.
    pub dropped_samples: u64,
    /// True when the last sample came from the diagnostic fallback clear.
    pub fallback_clear: bool,
    /// Sampling interval in presented frames.
    pub probe_every: u64,
    /// First named failure, if any. A set failure stops further sampling.
    pub failure: Option<String>,
}

impl Default for FrameProbeReport {
    fn default() -> Self {
        Self {
            sampled_frames: 0,
            probe_pixels: 0,
            last_sampled_frame: None,
            non_black_pixels: 0,
            unique_luma_buckets: 0,
            frame_crc: 0,
            dropped_samples: 0,
            fallback_clear: false,
            probe_every: FRAME_PROBE_DEFAULT_EVERY,
            failure: None,
        }
    }
}

impl FrameProbeReport {
    /// One stable `frame_probe=...` line for the host log.
    pub fn log_line(&self) -> String {
        let frame = self
            .last_sampled_frame
            .map_or_else(|| "none".to_string(), |index| index.to_string());
        let failure = self.failure.as_deref().unwrap_or("none");
        format!(
            "frame_probe=frame={frame} non_black_pixels={} unique_luma_buckets={} \
             frame_crc={:016x} sampled_frames={} probe_pixels={} dropped_samples={} \
             fallback_clear={} probe_every={} failure={failure}",
            self.non_black_pixels,
            self.unique_luma_buckets,
            self.frame_crc,
            self.sampled_frames,
            self.probe_pixels,
            self.dropped_samples,
            u8::from(self.fallback_clear),
            self.probe_every,
        )
    }
}

/// CRC-64/ECMA-182 (poly `0x42F0E1EBA9EA3693`, init `0`, no reflection, no
/// final XOR). This is the checksum of the sampled pixels only, in the order
/// the staging texture is laid out, so it is stable across processes and
/// sensitive to a single changed channel.
pub fn frame_crc64(pixels: &[[u8; 4]]) -> u64 {
    let mut crc = 0u64;
    for pixel in pixels {
        crc = crc64_ecma_update(crc, pixel);
    }
    crc
}

fn crc64_ecma_update(mut crc: u64, bytes: &[u8]) -> u64 {
    const POLYNOMIAL: u64 = 0x42F0_E1EB_A9EA_3693;
    for &byte in bytes {
        crc ^= u64::from(byte) << 56;
        for _ in 0..8 {
            crc = if crc & (1 << 63) != 0 {
                (crc << 1) ^ POLYNOMIAL
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// The 5-bit luma bucket of one pixel (`0..=31`), using BT.601 integer weights
/// so both byte orders agree for grayscale input.
pub fn luma_bucket(pixel: [u8; 4], order: PixelOrder) -> u32 {
    let (red, green, blue) = match order {
        PixelOrder::Bgra => (pixel[2], pixel[1], pixel[0]),
        PixelOrder::Rgba => (pixel[0], pixel[1], pixel[2]),
    };
    let luma = (77 * u32::from(red) + 150 * u32::from(green) + 29 * u32::from(blue)) >> 8;
    (luma.min(255)) >> 3
}

/// Measures a sampled buffer against `clear`.
///
/// A pixel is non-black when any of its four channels differs from the matching
/// `clear` channel by more than `threshold`. `clear` is expressed in the same
/// byte order as `pixels`.
pub fn analyze_pixels(
    pixels: &[[u8; 4]],
    order: PixelOrder,
    clear: [u8; 4],
    threshold: u8,
) -> FrameProbeSample {
    let mut non_black_pixels = 0u64;
    let mut buckets = 0u32;
    for &pixel in pixels {
        if pixel
            .iter()
            .zip(clear.iter())
            .any(|(&value, &clear)| value.abs_diff(clear) > threshold)
        {
            non_black_pixels += 1;
        }
        buckets |= 1u32 << luma_bucket(pixel, order);
    }
    FrameProbeSample {
        non_black_pixels,
        unique_luma_buckets: buckets.count_ones(),
        frame_crc: frame_crc64(pixels),
        probe_pixels: pixels.len() as u64,
    }
}

/// Convenience wrapper for the BGRA8 case.
pub fn analyze_bgra8(pixels: &[[u8; 4]], clear: [u8; 4], threshold: u8) -> FrameProbeSample {
    analyze_pixels(pixels, PixelOrder::Bgra, clear, threshold)
}

/// Layout of the fixed-size sampling grid for one drawable extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProbeLayout {
    block_width: u32,
    block_height: u32,
    step_x: u32,
    step_y: u32,
    staging_width: u32,
    staging_height: u32,
}

impl ProbeLayout {
    fn new(width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let block_width = FRAME_PROBE_BLOCK.min(width);
        let block_height = FRAME_PROBE_BLOCK.min(height);
        Some(Self {
            block_width,
            block_height,
            step_x: (width / FRAME_PROBE_GRID).max(1),
            step_y: (height / FRAME_PROBE_GRID).max(1),
            staging_width: block_width * FRAME_PROBE_GRID,
            staging_height: block_height * FRAME_PROBE_GRID,
        })
    }

    fn block_origin(&self, grid_x: u32, grid_y: u32, width: u32, height: u32) -> MTLOrigin {
        MTLOrigin {
            x: (grid_x * self.step_x).min(width.saturating_sub(self.block_width)) as usize,
            y: (grid_y * self.step_y).min(height.saturating_sub(self.block_height)) as usize,
            z: 0,
        }
    }

    fn staging_origin(&self, grid_x: u32, grid_y: u32) -> MTLOrigin {
        MTLOrigin {
            x: (grid_x * self.block_width) as usize,
            y: (grid_y * self.block_height) as usize,
            z: 0,
        }
    }
}

/// `MTLRegion` (`MTLOrigin origin, MTLSize size`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MTLRegion {
    origin: MTLOrigin,
    size: objc::MTLSize,
}

/// One shared staging texture reused by every probe.
struct StagingTexture {
    texture: ObjcObject,
    format: usize,
    layout: ProbeLayout,
}

/// The GPU side of the probe; created only when the environment opts in.
struct FrameProbe {
    every: u64,
    device: ObjcObject,
    staging: Option<StagingTexture>,
    /// Retained probe command buffer; at most one exists.
    in_flight: Option<ObjcObject>,
    in_flight_frame: u64,
    in_flight_started: Instant,
    in_flight_fallback_clear: bool,
    report: FrameProbeReport,
    /// Set on the first failure so the probe gives up instead of retrying.
    abandoned: bool,
}

impl FrameProbe {
    /// Builds the probe from the environment, or returns `None` when it is off.
    fn from_env(layer: objc::Id) -> Option<Self> {
        let config = FrameProbeConfig::from_env()?;
        // SAFETY: `layer` is a live `CAMetalLayer`; `device` returns its
        // (borrowed) `MTLDevice`, retained here for the probe's lifetime.
        let device = unsafe { objc::msg_send_id_0(layer, objc::sel(c"device")) };
        if device.is_null() {
            return None;
        }
        // SAFETY: `retain` returns the live device +1.
        let retained = unsafe { objc::msg_send_id_0(device, objc::sel(c"retain")) };
        let device = ObjcObject::from_retained(retained)?;
        // Readback needs the drawable to be more than a render target. This is
        // set only for the opt-in probe; the production layer stays
        // framebuffer-only.
        // SAFETY: `layer` is a live `CAMetalLayer`, and `setFramebufferOnly:`
        // takes a BOOL.
        unsafe {
            objc::msg_send_void_u8(layer, objc::sel(c"setFramebufferOnly:"), 0);
        }
        Some(Self {
            every: config.every,
            device,
            staging: None,
            in_flight: None,
            in_flight_frame: 0,
            in_flight_started: Instant::now(),
            in_flight_fallback_clear: false,
            report: FrameProbeReport {
                probe_every: config.every,
                ..FrameProbeReport::default()
            },
            abandoned: false,
        })
    }

    /// Called once per successful present. Never returns an error and never
    /// blocks for longer than [`FRAME_PROBE_WAIT`].
    fn observe(
        &mut self,
        queue: objc::Id,
        target: objc::Id,
        frame_index: u64,
        fallback_clear: bool,
    ) {
        if self.abandoned {
            return;
        }
        self.harvest_if_complete();
        let due = probe_due(frame_index, self.every);
        if self.in_flight.is_some() {
            if self.in_flight_started.elapsed() > FRAME_PROBE_MAX_IN_FLIGHT {
                self.fail("probe_in_flight_timeout");
            } else if due {
                // One probe at a time: drop this sampling opportunity instead
                // of queuing another command buffer.
                self.report.dropped_samples = self.report.dropped_samples.saturating_add(1);
            }
            return;
        }
        if !due {
            return;
        }
        match self.encode(queue, target, frame_index, fallback_clear) {
            Ok(()) => self.wait_for(FRAME_PROBE_WAIT),
            Err(reason) => self.fail(&reason),
        }
    }

    fn encode(
        &mut self,
        queue: objc::Id,
        target: objc::Id,
        frame_index: u64,
        fallback_clear: bool,
    ) -> Result<(), String> {
        // SAFETY: `target` is the live drawable texture; the accessors only
        // read immutable properties.
        let (format, width, height) = unsafe {
            (
                objc::msg_send_usize_0(target, objc::sel(c"pixelFormat")),
                objc::msg_send_usize_0(target, objc::sel(c"width")),
                objc::msg_send_usize_0(target, objc::sel(c"height")),
            )
        };
        if !matches!(
            format,
            MTL_PIXEL_FORMAT_BGRA8_UNORM | MTL_PIXEL_FORMAT_RGBA8_UNORM
        ) {
            return Err(format!("probe_unsupported_pixel_format_{format}"));
        }
        let width = u32::try_from(width).map_err(|_| "probe_extent_overflow".to_string())?;
        let height = u32::try_from(height).map_err(|_| "probe_extent_overflow".to_string())?;
        let layout = ProbeLayout::new(width, height).ok_or("probe_zero_extent".to_string())?;
        self.ensure_staging(format, layout)?;
        let staging = self.staging.as_ref().ok_or("probe_staging_missing")?;

        // SAFETY: `queue` is the live transport queue; `commandBuffer` returns
        // an autoreleased buffer that lives in the caller's pool.
        let command_buffer = unsafe { objc::msg_send_id_0(queue, objc::sel(c"commandBuffer")) };
        if command_buffer.is_null() {
            return Err("probe_command_buffer_null".to_string());
        }
        // SAFETY: `command_buffer` is live and uncommitted.
        let encoder =
            unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder")) };
        if encoder.is_null() {
            return Err("probe_blit_encoder_null".to_string());
        }
        for grid_y in 0..FRAME_PROBE_GRID {
            for grid_x in 0..FRAME_PROBE_GRID {
                // SAFETY: both textures are live 4-channel textures, the source
                // rectangle lies inside `target`, and the destination rectangle
                // lies inside the fixed staging atlas.
                unsafe {
                    encode_texture_block_copy(
                        encoder,
                        target,
                        layout.block_origin(grid_x, grid_y, width, height),
                        layout.block_width,
                        layout.block_height,
                        staging.texture.as_raw(),
                        layout.staging_origin(grid_x, grid_y),
                    );
                }
            }
        }
        // SAFETY: `encoder` is an active blit encoder and `command_buffer` is
        // an uncommitted buffer with only that encoder attached.
        unsafe {
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
        }
        // SAFETY: retry the just-committed buffer so it can be polled after the
        // autorelease pool of the present call is drained.
        let retained = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"retain")) };
        let retained = ObjcObject::from_retained(retained)
            .ok_or_else(|| "probe_command_buffer_retain_failed".to_string())?;
        self.in_flight = Some(retained);
        self.in_flight_frame = frame_index;
        self.in_flight_started = Instant::now();
        self.in_flight_fallback_clear = fallback_clear;
        Ok(())
    }

    fn ensure_staging(&mut self, format: usize, layout: ProbeLayout) -> Result<(), String> {
        if let Some(staging) = self.staging.as_ref()
            && staging.format == format
            && staging.layout == layout
        {
            return Ok(());
        }
        let texture = create_probe_staging_texture(
            self.device.as_raw(),
            format,
            layout.staging_width,
            layout.staging_height,
        )?;
        self.staging = Some(StagingTexture {
            texture,
            format,
            layout,
        });
        Ok(())
    }

    fn poll_status(&self) -> Option<usize> {
        self.in_flight.as_ref().map(|command_buffer| {
            // SAFETY: `command_buffer` is a live retained `MTLCommandBuffer`.
            unsafe { objc::msg_send_usize_0(command_buffer.as_raw(), objc::sel(c"status")) }
        })
    }

    /// Polls without blocking; harvests a completed probe, records a GPU error,
    /// and leaves a still-running probe in place.
    fn harvest_if_complete(&mut self) {
        if self.in_flight.is_none() {
            return;
        }
        match self.poll_status() {
            Some(MTL_COMMAND_BUFFER_STATUS_COMPLETED) => self.harvest(),
            // `MTLCommandBufferStatusError` is 5.
            Some(5) => self.fail("probe_command_buffer_error"),
            _ => {}
        }
    }

    /// Waits at most `timeout` for the in-flight probe, then returns.
    fn wait_for(&mut self, timeout: Duration) {
        let started = Instant::now();
        loop {
            match self.poll_status() {
                Some(MTL_COMMAND_BUFFER_STATUS_COMPLETED) => {
                    self.harvest();
                    return;
                }
                Some(5) => {
                    self.fail("probe_command_buffer_error");
                    return;
                }
                Some(_) => {}
                None => return,
            }
            if started.elapsed() >= timeout {
                // Still in flight: keep the single command buffer and drop the
                // next opportunity rather than block the render loop.
                return;
            }
            std::thread::sleep(Duration::from_micros(250));
        }
    }

    fn harvest(&mut self) {
        let Some(command_buffer) = self.in_flight.take() else {
            return;
        };
        let Some(staging) = self.staging.as_ref() else {
            self.fail("probe_staging_missing");
            return;
        };
        let width = staging.layout.staging_width;
        let height = staging.layout.staging_height;
        let mut pixels = vec![[0u8; 4]; (width as usize) * (height as usize)];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc::MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            },
        };
        let bytes_per_row = (width as usize) * (BYTES_PER_PIXEL as usize);
        // SAFETY: the command buffer completed, so the shared staging texture
        // holds final bytes; `pixels` is exactly `width * height` pixels and
        // `bytes_per_row` matches its row stride.
        unsafe {
            get_texture_bytes(
                staging.texture.as_raw(),
                pixels.as_mut_ptr().cast(),
                bytes_per_row,
                region,
            );
        }
        let order = if staging.format == MTL_PIXEL_FORMAT_BGRA8_UNORM {
            PixelOrder::Bgra
        } else {
            PixelOrder::Rgba
        };
        // A blitted frame has a guest-defined clear that this probe cannot
        // know, so non-black is measured against opaque black. The diagnostic
        // fallback clear *is* known: use its first pixel so a uniform fallback
        // frame reports zero non-black pixels.
        let clear = if self.in_flight_fallback_clear {
            pixels.first().copied().unwrap_or([0, 0, 0, 255])
        } else {
            [0, 0, 0, 255]
        };
        let sample = analyze_pixels(&pixels, order, clear, FRAME_PROBE_CHANNEL_THRESHOLD);
        self.report.sampled_frames = self.report.sampled_frames.saturating_add(1);
        self.report.probe_pixels = sample.probe_pixels;
        self.report.last_sampled_frame = Some(self.in_flight_frame);
        self.report.non_black_pixels = sample.non_black_pixels;
        self.report.unique_luma_buckets = sample.unique_luma_buckets;
        self.report.frame_crc = sample.frame_crc;
        self.report.fallback_clear = self.in_flight_fallback_clear;
        drop(command_buffer);
        self.publish();
    }

    fn fail(&mut self, reason: &str) {
        self.abandoned = true;
        self.in_flight = None;
        self.report.failure = Some(reason.to_string());
        self.publish();
    }

    fn publish(&self) {
        publish_frame_probe_report(&self.report);
        eprintln!("{}", self.report.log_line());
    }
}

/// Creates the one shared BGRA8/RGBA8 staging texture the probe reads back.
fn create_probe_staging_texture(
    device: objc::Id,
    format: usize,
    width: u32,
    height: u32,
) -> Result<ObjcObject, String> {
    let class = objc::class(c"MTLTextureDescriptor");
    if class.is_null() {
        return Err("probe_texture_descriptor_unavailable".to_string());
    }
    // SAFETY: `class` is `MTLTextureDescriptor`; the selector is the documented
    // 2D factory and `format` is a 4-channel unorm format.
    let descriptor = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let send: Fn = mem::transmute(objc::msg_send_entry());
        send(
            class,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            format,
            width as usize,
            height as usize,
            0,
        )
    };
    if descriptor.is_null() {
        return Err("probe_texture_descriptor_null".to_string());
    }
    // SAFETY: `descriptor` is a live descriptor; shared storage lets the CPU
    // read the staging texture after completion.
    unsafe {
        msg_send_void_usize(
            descriptor,
            objc::sel(c"setStorageMode:"),
            MTL_STORAGE_MODE_SHARED,
        );
    }
    // SAFETY: `device` is the live layer device and `descriptor` is fully
    // populated; `newTextureWithDescriptor:` returns a retained texture (+1).
    let raw =
        unsafe { objc::msg_send_id_1(device, objc::sel(c"newTextureWithDescriptor:"), descriptor) };
    ObjcObject::from_retained(raw).ok_or_else(|| "probe_staging_texture_null".to_string())
}

/// Copies one block from `source` to `destination` on a blit encoder.
///
/// # Safety
///
/// Both textures must be live, have the same 4-channel pixel format, and the
/// source/destination rectangles must lie inside their respective textures.
unsafe fn encode_texture_block_copy(
    encoder: objc::Id,
    source: objc::Id,
    source_origin: MTLOrigin,
    width: u32,
    height: u32,
    destination: objc::Id,
    destination_origin: MTLOrigin,
) {
    type Fn = unsafe extern "C" fn(
        objc::Id,
        objc::Sel,
        objc::Id,
        usize,
        usize,
        MTLOrigin,
        objc::MTLSize,
        objc::Id,
        usize,
        usize,
        MTLOrigin,
    );
    let size = objc::MTLSize {
        width: width as usize,
        height: height as usize,
        depth: 1,
    };
    // SAFETY: the caller guarantees textures, formats and rectangles.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: argument layout matches the 11-argument selector.
    unsafe {
        send(
            encoder,
            objc::sel(
                c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            ),
            source,
            0,
            0,
            source_origin,
            size,
            destination,
            0,
            0,
            destination_origin,
        );
    }
}

/// Issues `getBytes:bytesPerRow:fromRegion:mipmapLevel:` on a shared texture.
///
/// # Safety
///
/// `texture` must be a live shared-storage texture whose GPU work completed,
/// and `destination` must hold at least `bytes_per_row * region.size.height`.
unsafe fn get_texture_bytes(
    texture: objc::Id,
    destination: *mut c_void,
    bytes_per_row: usize,
    region: MTLRegion,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, *mut c_void, usize, MTLRegion, usize);
    // SAFETY: the caller guarantees the texture and buffer bounds.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: argument layout matches `getBytes:bytesPerRow:fromRegion:mipmapLevel:`.
    unsafe {
        send(
            texture,
            objc::sel(c"getBytes:bytesPerRow:fromRegion:mipmapLevel:"),
            destination,
            bytes_per_row,
            region,
            0,
        );
    }
}

/// An opaque clear colour used by the present render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearColor {
    /// Red component in `0.0..=1.0`.
    pub red: f64,
    /// Green component in `0.0..=1.0`.
    pub green: f64,
    /// Blue component in `0.0..=1.0`.
    pub blue: f64,
    /// Alpha component in `0.0..=1.0`.
    pub alpha: f64,
}

impl ClearColor {
    /// Opaque red, a channel-exact value BGRA8 can represent without rounding.
    pub const OPAQUE_RED: Self = Self {
        red: 1.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    };
    /// Opaque green.
    pub const OPAQUE_GREEN: Self = Self {
        red: 0.0,
        green: 1.0,
        blue: 0.0,
        alpha: 1.0,
    };
    /// Opaque blue.
    pub const OPAQUE_BLUE: Self = Self {
        red: 0.0,
        green: 0.0,
        blue: 1.0,
        alpha: 1.0,
    };

    fn validate(self) -> Result<(), PresentError> {
        let components = [self.red, self.green, self.blue, self.alpha];
        if components
            .iter()
            .any(|component| !component.is_finite() || !(0.0..=1.0).contains(component))
        {
            return Err(PresentError::InvalidClearColor);
        }
        Ok(())
    }
}

/// Errors from surface creation and the present path.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PresentError {
    /// A surface edge was zero.
    #[error("surface width and height must be nonzero (got {width}x{height})")]
    ZeroExtent {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    /// A surface edge exceeded the supported Metal 2D limit or overflowed.
    #[error("surface extent {width}x{height} exceeds the supported 2D limit")]
    ExtentTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    /// `IOSurfaceCreate` returned null.
    #[error("IOSurfaceCreate failed for a BGRA8 surface")]
    SurfaceCreationFailed,
    /// `newTextureWithDescriptor:iosurface:plane:` returned null.
    #[error("MTLDevice could not wrap the IOSurface as an MTLTexture")]
    TextureCreationFailed,
    /// The surface was created on a different `MTLDevice` than the presenter.
    #[error("surface belongs to a different Metal device than the presenter")]
    ForeignSurface,
    /// A frame is still in flight; a second present would overwrite it.
    #[error(
        "frame {frame_index} on surface {surface_id} is still in flight; finish it before presenting again"
    )]
    FrameInFlight {
        /// In-flight frame index.
        frame_index: u64,
        /// In-flight surface identity.
        surface_id: u32,
    },
    /// No frame was in flight when completion was requested.
    #[error("no frame is currently in flight")]
    NoFrameInFlight,
    /// Clear colour had a non-finite or out-of-range component.
    #[error("clear colour components must be finite and within 0.0..=1.0")]
    InvalidClearColor,
    /// The present named a sync interval other than immediate or vsync.
    #[error(
        "unsupported present sync interval {sync_interval}; only 0 (immediate) and 1 (vsync) are supported"
    )]
    UnsupportedSyncInterval {
        /// Requested sync interval.
        sync_interval: u32,
    },
    /// The sink refused the frame; the frame was not accepted.
    #[error("present sink refused frame {frame_index}: {reason}")]
    SinkRefused {
        /// Refused frame index.
        frame_index: u64,
        /// Sink-provided refusal reason.
        reason: String,
    },
    /// Metal wiring failed while encoding or completing the present pass.
    #[error("Metal present command failed: {0}")]
    Metal(String),
}

/// An `IOSurface`-backed BGRA8 image rendered into directly by Metal.
///
/// The surface identity (`IOSurfaceID`) and the retained `MTLTexture` are
/// exposed for presentation; pixels are never exposed to production code.
pub struct SurfaceImage {
    surface: CfOwned,
    texture: ObjcObject,
    device: objc::Id,
    surface_id: u32,
    width: u32,
    height: u32,
    bytes_per_row: usize,
    frame_index: Option<u64>,
}

impl SurfaceImage {
    /// Creates a BGRA8 IOSurface and wraps it in a render-target `MTLTexture`.
    ///
    /// The surface is created with the documented `IOSurfaceCreate` property
    /// keys (`IOSurfaceWidth`, `IOSurfaceHeight`, `IOSurfaceBytesPerElement`,
    /// `IOSurfaceBytesPerRow`, `IOSurfacePixelFormat` = `'BGRA'`,
    /// `IOSurfaceAllocSize`). Zero extents, extents over the Metal 2D limit and
    /// any null `IOSurface`/`MTLTexture` are rejected explicitly; no GPU work
    /// happens on the error path.
    pub fn create(
        transport: &DirectMetalTransport,
        width: u32,
        height: u32,
    ) -> Result<Self, PresentError> {
        if width == 0 || height == 0 {
            return Err(PresentError::ZeroExtent { width, height });
        }
        if width > MAX_SURFACE_DIMENSION || height > MAX_SURFACE_DIMENSION {
            return Err(PresentError::ExtentTooLarge { width, height });
        }
        let min_row = usize::try_from(width)
            .ok()
            .and_then(|w| w.checked_mul(BYTES_PER_PIXEL as usize))
            .ok_or(PresentError::ExtentTooLarge { width, height })?;
        let bytes_per_row = min_row.next_multiple_of(SURFACE_ROW_ALIGNMENT);

        let _pool = AutoreleasePool::new();
        let surface = create_bgra8_iosurface(width, height, bytes_per_row)?;

        // SAFETY: `surface` is a live IOSurface within this pool, and the
        // geometry accessors only read immutable surface properties.
        let (actual_width, actual_height, actual_row, surface_id) = unsafe {
            (
                IOSurfaceGetWidth(surface.as_ptr()),
                IOSurfaceGetHeight(surface.as_ptr()),
                IOSurfaceGetBytesPerRow(surface.as_ptr()),
                IOSurfaceGetID(surface.as_ptr()),
            )
        };
        if actual_width != width as usize
            || actual_height != height as usize
            || actual_row < min_row
        {
            return Err(PresentError::SurfaceCreationFailed);
        }

        let texture = create_iosurface_texture(
            transport.device_raw(),
            surface.as_ptr(),
            width,
            height,
        )?;

        Ok(Self {
            surface,
            texture,
            device: transport.device_raw(),
            surface_id,
            width,
            height,
            bytes_per_row: actual_row,
            frame_index: None,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Padded row pitch in bytes, as reported by the IOSurface.
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// `IOSurfaceID`, the cross-process identity of the surface.
    pub fn surface_id(&self) -> u32 {
        self.surface_id
    }

    /// Index of the most recent frame rendered into this image, if any.
    pub fn frame_index(&self) -> Option<u64> {
        self.frame_index
    }

    /// Raw retained `MTLTexture` handle for sibling render modules.
    ///
    /// The texture is owned by this image and valid for its lifetime.
    pub fn texture_raw(&self) -> *mut c_void {
        self.texture.as_raw()
    }

    /// Raw `IOSurfaceRef` for the single windowed-host attach point.
    ///
    /// The reference is owned by this image and valid for its lifetime. A
    /// windowed host must retain it if it outlives the image.
    pub fn iosurface_raw(&self) -> *mut c_void {
        self.surface.as_ptr()
    }

    /// Test-only readback: locks the IOSurface exactly once and copies its rows.
    ///
    /// This is the only memcpy of surface memory in the module and exists only
    /// under `cfg(test)`; it is never compiled into a production build.
    #[cfg(test)]
    fn read_bgra_rows(&self) -> Vec<u8> {
        let surface = self.surface.as_ptr();
        // SAFETY: `surface` is a live IOSurface; a read-only lock is the
        // documented way to obtain stable CPU access to its backing store.
        let rc = unsafe { IOSurfaceLock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceLock(read-only) failed");
        // SAFETY: The surface is locked, so the base address is valid for
        // `bytes_per_row * height` bytes of read-only access.
        let base = unsafe { IOSurfaceGetBaseAddress(surface) }.cast::<u8>();
        assert!(!base.is_null(), "locked IOSurface has no base address");
        let len = self.bytes_per_row * self.height as usize;
        let mut pixels = vec![0u8; len];
        // SAFETY: `base` points to `len` readable bytes of a locked surface and
        // `pixels` is a distinct, exclusively borrowed `len`-byte allocation.
        unsafe { std::ptr::copy_nonoverlapping(base, pixels.as_mut_ptr(), len) };
        // SAFETY: Unlocks the same live surface locked above; the read-only
        // option must match.
        let rc = unsafe { IOSurfaceUnlock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceUnlock failed");
        pixels
    }
}

/// One at a time present path shared by the headless test and a windowed host.
///
/// [`Self::present`] encodes a clear pass into the image, waits for GPU
/// completion, then hands the surface to the sink. The frame stays in flight
/// until [`Self::finish_present`] is called, so a second present cannot
/// overwrite a surface the host has not finished with.
pub struct PresentPath<'a> {
    transport: &'a DirectMetalTransport,
    started: Instant,
    next_frame_index: u64,
    in_flight: Option<PresentedFrame>,
}

impl<'a> PresentPath<'a> {
    /// Creates a present path on `transport`'s device and queue.
    pub fn new(transport: &'a DirectMetalTransport) -> Self {
        Self {
            transport,
            started: Instant::now(),
            next_frame_index: 0,
            in_flight: None,
        }
    }

    /// The frame currently in flight, if any.
    pub fn in_flight(&self) -> Option<PresentedFrame> {
        self.in_flight
    }

    /// Renders `clear` into `image`, waits, and hands the surface to `sink`.
    ///
    /// Returns an explicit [`PresentError::FrameInFlight`] when a previous
    /// frame has not been finished; in that case no encoding happens and the
    /// in-flight surface is untouched. A refused frame consumes its frame
    /// index but leaves the presenter with nothing in flight; a Metal error
    /// consumes nothing.
    ///
    /// See the module documentation for the single windowed-host call site
    /// that hands the finished surface to a `CAMetalLayer`.
    pub fn present(
        &mut self,
        image: &mut SurfaceImage,
        clear: ClearColor,
        sink: &mut dyn PresentSink,
    ) -> Result<(), PresentError> {
        self.present_with_interval(image, clear, SyncInterval::Immediate, sink)
    }

    /// Like [`Self::present`], carrying the guest's requested sync interval.
    ///
    /// The interval is validated and forwarded to the sink on the
    /// [`PresentedFrame`]; it never changes how the surface is rendered or
    /// read, so this path stays readback-free.
    pub fn present_with_interval(
        &mut self,
        image: &mut SurfaceImage,
        clear: ClearColor,
        sync_interval: SyncInterval,
        sink: &mut dyn PresentSink,
    ) -> Result<(), PresentError> {
        if let Some(inflight) = self.in_flight {
            return Err(PresentError::FrameInFlight {
                frame_index: inflight.frame_index,
                surface_id: inflight.surface_id,
            });
        }
        if image.device != self.transport.device_raw() {
            return Err(PresentError::ForeignSurface);
        }
        clear.validate()?;

        let frame_index = self.next_frame_index;
        let timestamp_ns = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        encode_clear_pass(self.transport.queue_raw(), image.texture_raw(), clear)?;

        // GPU work is complete: the surface holds the frame now. A Metal
        // failure above did not consume an index.
        self.next_frame_index = frame_index.wrapping_add(1);
        image.frame_index = Some(frame_index);
        let frame = PresentedFrame {
            surface_id: image.surface_id,
            frame_index,
            timestamp_ns,
            sync_interval,
        };
        self.in_flight = Some(frame);
        if let Err(error) = sink.present(frame) {
            // The sink did not take ownership: nothing is in flight.
            self.in_flight = None;
            return Err(error);
        }
        Ok(())
    }

    /// Presents a frame the caller has already drawn into `image`.
    ///
    /// This path encodes no render pass and reads no pixels: it waits for the
    /// GPU work already submitted against `image`'s texture on this
    /// transport's queue to finish, then hands the surface to `sink`. The wait
    /// is an empty command buffer committed to the same queue and awaited;
    /// Metal executes command buffers on one queue in submission order, so its
    /// completion fences every earlier command buffer that drew into the
    /// surface.
    ///
    /// In-flight and foreign-surface rules match [`Self::present`]: a second
    /// call before [`Self::finish_present`] returns
    /// [`PresentError::FrameInFlight`] without touching the surface.
    pub fn present_drawn(
        &mut self,
        image: &mut SurfaceImage,
        sink: &mut dyn PresentSink,
    ) -> Result<(), PresentError> {
        self.present_drawn_with_interval(image, SyncInterval::Immediate, sink)
    }

    /// Like [`Self::present_drawn`], carrying the guest's sync interval.
    ///
    /// The interval is forwarded on the [`PresentedFrame`] and does not touch
    /// the surface's pixels.
    pub fn present_drawn_with_interval(
        &mut self,
        image: &mut SurfaceImage,
        sync_interval: SyncInterval,
        sink: &mut dyn PresentSink,
    ) -> Result<(), PresentError> {
        if let Some(inflight) = self.in_flight {
            return Err(PresentError::FrameInFlight {
                frame_index: inflight.frame_index,
                surface_id: inflight.surface_id,
            });
        }
        if image.device != self.transport.device_raw() {
            return Err(PresentError::ForeignSurface);
        }

        let frame_index = self.next_frame_index;
        let timestamp_ns = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        wait_for_queue_completion(self.transport.queue_raw())?;

        // The drawn frame is complete in the surface now; hand it over
        // unchanged. A Metal failure above did not consume an index.
        self.next_frame_index = frame_index.wrapping_add(1);
        image.frame_index = Some(frame_index);
        let frame = PresentedFrame {
            surface_id: image.surface_id,
            frame_index,
            timestamp_ns,
            sync_interval,
        };
        self.in_flight = Some(frame);
        if let Err(error) = sink.present(frame) {
            // The sink did not take ownership: nothing is in flight.
            self.in_flight = None;
            return Err(error);
        }
        Ok(())
    }

    /// Marks the in-flight frame as finished and releases the surface.
    ///
    /// This is the completion signal a windowed host sends after the window
    /// server has consumed the surface.
    pub fn finish_present(&mut self) -> Result<PresentedFrame, PresentError> {
        self.in_flight.take().ok_or(PresentError::NoFrameInFlight)
    }
}

/// Owns a +1 CoreFoundation reference (`IOSurfaceRef`).
struct CfOwned(*mut c_void);

impl CfOwned {
    fn from_retained(ptr: *mut c_void) -> Option<Self> {
        if ptr.is_null() { None } else { Some(Self(ptr)) }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for CfOwned {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a live +1 CoreFoundation object (IOSurfaceCreate
        // returns it retained) and is released exactly once here.
        unsafe { CFRelease(self.0) };
    }
}

/// Builds the `CFDictionary` of documented property keys and creates the surface.
fn create_bgra8_iosurface(
    width: u32,
    height: u32,
    bytes_per_row: usize,
) -> Result<CfOwned, PresentError> {
    let dict_class = objc::class(c"NSMutableDictionary");
    if dict_class.is_null() {
        return Err(PresentError::SurfaceCreationFailed);
    }
    // SAFETY: `dict_class` is the `NSMutableDictionary` class and `dictionary`
    // is its documented class factory, returning an autoreleased dictionary
    // that stays alive inside the caller's autorelease pool.
    let dict = unsafe { objc::msg_send_id_0(dict_class, objc::sel(c"dictionary")) };
    if dict.is_null() {
        return Err(PresentError::SurfaceCreationFailed);
    }

    let alloc_size = bytes_per_row
        .checked_mul(height as usize)
        .ok_or(PresentError::ExtentTooLarge { width, height })?;
    let (Ok(width_prop), Ok(height_prop), Ok(row_prop)) = (
        i32::try_from(width),
        i32::try_from(height),
        i32::try_from(bytes_per_row),
    ) else {
        return Err(PresentError::ExtentTooLarge { width, height });
    };

    let properties: [(&str, PropertyValue); 6] = [
        ("IOSurfaceWidth", PropertyValue::Int(width_prop)),
        ("IOSurfaceHeight", PropertyValue::Int(height_prop)),
        (
            "IOSurfaceBytesPerElement",
            PropertyValue::Int(BYTES_PER_PIXEL as i32),
        ),
        ("IOSurfaceBytesPerRow", PropertyValue::Int(row_prop)),
        (
            "IOSurfacePixelFormat",
            PropertyValue::Int(IOSURFACE_PIXEL_FORMAT_BGRA),
        ),
        ("IOSurfaceAllocSize", PropertyValue::Long(alloc_size as i64)),
    ];
    for (key, value) in properties {
        let key = objc::nsstring(key).ok_or(PresentError::SurfaceCreationFailed)?;
        // SAFETY: `value` is a Foundation number factory returning an
        // autoreleased `NSNumber` (toll-free bridged to `CFNumber`).
        let value = unsafe {
            match value {
                PropertyValue::Int(value) => number_with_int(value),
                PropertyValue::Long(value) => number_with_long_long(value),
            }
        };
        if value.is_null() {
            return Err(PresentError::SurfaceCreationFailed);
        }
        // SAFETY: `dict` is a mutable dictionary; both objects are live for the
        // duration of this call and `setObject:forKey:` retains them.
        unsafe { msg_send_void_2(dict, objc::sel(c"setObject:forKey:"), value, key.as_raw()) };
    }

    // SAFETY: `dict` is a valid, autoreleased `NSMutableDictionary`
    // (toll-free bridged to `CFDictionary`) carrying every documented
    // `IOSurfaceCreate` property key; IOSurfaceCreate retains what it needs.
    let surface = unsafe { IOSurfaceCreate(dict) };
    CfOwned::from_retained(surface).ok_or(PresentError::SurfaceCreationFailed)
}

/// A dictionary value: a 32-bit property or a 64-bit allocation size.
enum PropertyValue {
    Int(i32),
    Long(i64),
}

/// Creates the retained render-target `MTLTexture` over an IOSurface plane.
fn create_iosurface_texture(
    device: objc::Id,
    surface: *mut c_void,
    width: u32,
    height: u32,
) -> Result<ObjcObject, PresentError> {
    let class = objc::class(c"MTLTextureDescriptor");
    if class.is_null() {
        return Err(PresentError::TextureCreationFailed);
    }
    // SAFETY: `class` is `MTLTextureDescriptor`, the selector is the documented
    // 2D factory (BGRA8, one mip level) and the receiver is the class object.
    let descriptor = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            class,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            MTL_PIXEL_FORMAT_BGRA8_UNORM,
            width as usize,
            height as usize,
            0,
        )
    };
    if descriptor.is_null() {
        return Err(PresentError::TextureCreationFailed);
    }

    // SAFETY: `descriptor` is a live `MTLTextureDescriptor`; both selectors take
    // the documented `MTLTextureUsage`/`MTLStorageMode` integer values.
    unsafe {
        msg_send_void_usize(
            descriptor,
            objc::sel(c"setUsage:"),
            MTL_TEXTURE_USAGE_SHADER_READ | MTL_TEXTURE_USAGE_RENDER_TARGET,
        );
        msg_send_void_usize(
            descriptor,
            objc::sel(c"setStorageMode:"),
            MTL_STORAGE_MODE_SHARED,
        );
    }

    // SAFETY: `device` is the live transport `MTLDevice`, `surface` is the live
    // IOSurface and `plane` is `0` for the single plane of a BGRA8 surface. The
    // returned texture is retained (+1).
    let raw = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, *mut c_void, usize) -> objc::Id;
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            device,
            objc::sel(c"newTextureWithDescriptor:iosurface:plane:"),
            descriptor,
            surface,
            0,
        )
    };
    ObjcObject::from_retained(raw).ok_or(PresentError::TextureCreationFailed)
}

/// Encodes a clear-only render pass into `texture`, commits and waits.
///
/// A full-screen draw would be encoded through the same render pass descriptor
/// before `endEncoding`; clear-only already proves the surface is a valid
/// render target with no CPU involvement.
fn encode_clear_pass(
    queue: objc::Id,
    texture: objc::Id,
    clear: ClearColor,
) -> Result<(), PresentError> {
    let _pool = AutoreleasePool::new();

    // SAFETY: `queue` is the live transport queue; `commandBuffer` returns an
    // autoreleased `MTLCommandBuffer` that lives for this pool.
    let command_buffer = unsafe { objc::msg_send_id_0(queue, objc::sel(c"commandBuffer")) };
    if command_buffer.is_null() {
        return Err(PresentError::Metal(
            "MTLCommandQueue.commandBuffer returned null".to_string(),
        ));
    }

    let class = objc::class(c"MTLRenderPassDescriptor");
    if class.is_null() {
        return Err(PresentError::Metal(
            "MTLRenderPassDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `class` is the descriptor class and `renderPassDescriptor` is its
    // documented factory, returning an autoreleased descriptor.
    let descriptor =
        unsafe { objc::msg_send_id_0(class, objc::sel(c"renderPassDescriptor")) };
    if descriptor.is_null() {
        return Err(PresentError::Metal(
            "MTLRenderPassDescriptor creation returned null".to_string(),
        ));
    }

    // SAFETY: `descriptor` is a live render pass descriptor; `colorAttachments`
    // returns its attachment array and `objectAtIndexedSubscript:` returns the
    // colour attachment at index 0 (the only one this pass uses).
    let attachments = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    if attachments.is_null() {
        return Err(PresentError::Metal(
            "MTLRenderPassDescriptor.colorAttachments returned null".to_string(),
        ));
    }
    // SAFETY: See above; index 0 is inside a colour attachment array.
    let attachment = unsafe {
        msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0)
    };
    if attachment.is_null() {
        return Err(PresentError::Metal(
            "colour attachment 0 is null".to_string(),
        ));
    }

    // SAFETY: `attachment` is a live colour attachment descriptor; the setters
    // take a live `MTLTexture`, the documented load/store actions and a
    // `MTLClearColor` passed by value.
    unsafe {
        objc::msg_send_void_1(attachment, objc::sel(c"setTexture:"), texture);
        msg_send_void_usize(
            attachment,
            objc::sel(c"setLoadAction:"),
            MTL_LOAD_ACTION_CLEAR,
        );
        msg_send_void_usize(
            attachment,
            objc::sel(c"setStoreAction:"),
            MTL_STORE_ACTION_STORE,
        );
        msg_send_void_clear_color(
            attachment,
            objc::sel(c"setClearColor:"),
            MTLClearColor {
                red: clear.red,
                green: clear.green,
                blue: clear.blue,
                alpha: clear.alpha,
            },
        );
    }

    // SAFETY: `command_buffer` is an uncommitted command buffer and
    // `descriptor` is a fully populated render pass descriptor.
    let encoder = unsafe {
        objc::msg_send_id_1(
            command_buffer,
            objc::sel(c"renderCommandEncoderWithDescriptor:"),
            descriptor,
        )
    };
    if encoder.is_null() {
        return Err(PresentError::Metal(
            "renderCommandEncoderWithDescriptor: returned null".to_string(),
        ));
    }

    // SAFETY: `encoder` is an active render encoder, `command_buffer` is
    // uncommitted, and waiting here means the surface contents are final before
    // any sink or window server can read them. No readback is performed.
    unsafe {
        objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
        objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
        objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
    }

    // SAFETY: The command buffer completed via `waitUntilCompleted`.
    let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
    if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
        // SAFETY: `[command_buffer error]` is null or an autoreleased `NSError *`.
        let error = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
        // SAFETY: `error` is null or a live `NSError *` inside this pool.
        let message = unsafe { objc::nserror_message(error) };
        return Err(PresentError::Metal(format!(
            "present command buffer failed (status={status}): {message}"
        )));
    }
    Ok(())
}

/// Commits an empty command buffer and waits for it, fencing earlier work.
///
/// Encoding nothing is deliberate: `present_drawn` must not touch the surface.
/// Because a single `MTLCommandQueue` runs its command buffers in submission
/// order, this buffer's completion means every earlier buffer on the queue,
/// including the one that drew the frame, has completed.
fn wait_for_queue_completion(queue: objc::Id) -> Result<(), PresentError> {
    let _pool = AutoreleasePool::new();

    // SAFETY: `queue` is the live transport queue; `commandBuffer` returns an
    // autoreleased `MTLCommandBuffer` that lives for this pool.
    let command_buffer = unsafe { objc::msg_send_id_0(queue, objc::sel(c"commandBuffer")) };
    if command_buffer.is_null() {
        return Err(PresentError::Metal(
            "MTLCommandQueue.commandBuffer returned null".to_string(),
        ));
    }

    // SAFETY: `command_buffer` is a live uncommitted buffer with no encoders;
    // committing and waiting on it only orders it after earlier queue work.
    unsafe {
        objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
        objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
    }

    // SAFETY: The command buffer completed via `waitUntilCompleted`.
    let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
    if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
        // SAFETY: `[command_buffer error]` is null or an autoreleased `NSError *`.
        let error = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
        // SAFETY: `error` is null or a live `NSError *` inside this pool.
        let message = unsafe { objc::nserror_message(error) };
        return Err(PresentError::Metal(format!(
            "present fence command buffer failed (status={status}): {message}"
        )));
    }
    Ok(())
}

/// Apple `MTLClearColor` (`double red, green, blue, alpha`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MTLClearColor {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
    fn IOSurfaceGetID(surface: *mut c_void) -> u32;
    fn IOSurfaceGetWidth(surface: *mut c_void) -> usize;
    fn IOSurfaceGetHeight(surface: *mut c_void) -> usize;
    fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
    /// Test-only: CPU access to surface pixels is never part of the present path.
    #[cfg(test)]
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    /// Test-only: see [`IOSurfaceGetBaseAddress`].
    #[cfg(test)]
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    /// Test-only: see [`IOSurfaceGetBaseAddress`].
    #[cfg(test)]
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
}

/// Sends `numberWithInt:` and returns the autoreleased `NSNumber`.
unsafe fn number_with_int(value: i32) -> objc::Id {
    let class = objc::class(c"NSNumber");
    if class.is_null() {
        return std::ptr::null_mut();
    }
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, i32) -> objc::Id;
    // SAFETY: `class` is `NSNumber`, the selector is its documented factory and
    // the argument is the value to box.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Argument layout matches `numberWithInt:`.
    unsafe { f(class, objc::sel(c"numberWithInt:"), value) }
}

/// Sends `numberWithLongLong:` and returns the autoreleased `NSNumber`.
unsafe fn number_with_long_long(value: i64) -> objc::Id {
    let class = objc::class(c"NSNumber");
    if class.is_null() {
        return std::ptr::null_mut();
    }
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, i64) -> objc::Id;
    // SAFETY: `class` is `NSNumber`, the selector is its documented factory and
    // the argument is the value to box.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Argument layout matches `numberWithLongLong:`.
    unsafe { f(class, objc::sel(c"numberWithLongLong:"), value) }
}

/// Sends a selector taking one `NSUInteger` (no object lifetime transfer).
unsafe fn msg_send_void_usize(target: objc::Id, s: objc::Sel, value: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
    // SAFETY: Caller supplies a valid receiver, selector and `NSUInteger` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's argument layout.
    unsafe { f(target, s, value) };
}

/// Sends a selector taking one `NSUInteger` and returning an object.
unsafe fn msg_send_id_usize(target: objc::Id, s: objc::Sel, value: usize) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize) -> objc::Id;
    // SAFETY: Caller supplies a valid receiver, selector and `NSUInteger` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's argument and return layout.
    unsafe { f(target, s, value) }
}

/// Sends a selector taking two object arguments.
unsafe fn msg_send_void_2(target: objc::Id, s: objc::Sel, a0: objc::Id, a1: objc::Id) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, objc::Id);
    // SAFETY: Caller supplies a valid receiver and two live object arguments.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's argument layout.
    unsafe { f(target, s, a0, a1) };
}

/// Sends `setClearColor:` with a 32-byte `MTLClearColor` by value.
unsafe fn msg_send_void_clear_color(target: objc::Id, s: objc::Sel, color: MTLClearColor) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLClearColor);
    // SAFETY: Caller supplies a valid colour attachment descriptor; on arm64 the
    // four doubles are passed in v0-v3 through normal `objc_msgSend`.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's argument layout.
    unsafe { f(target, s, color) };
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::Duration;

    use super::*;

    fn open_transport() -> DirectMetalTransport {
        DirectMetalTransport::new().expect("Metal device should open")
    }

    fn create_surface(transport: &DirectMetalTransport, width: u32, height: u32) -> SurfaceImage {
        SurfaceImage::create(transport, width, height).expect("IOSurface-backed image creation")
    }

    fn channel(value: f64) -> u8 {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    fn expected_bgra(clear: ClearColor) -> [u8; 4] {
        [
            channel(clear.blue),
            channel(clear.green),
            channel(clear.red),
            channel(clear.alpha),
        ]
    }

    fn assert_solid(
        image: &SurfaceImage,
        clear: ClearColor,
        context: &str,
    ) -> Vec<u8> {
        let pixels = image.read_bgra_rows();
        let expected = expected_bgra(clear);
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
            "{context}: surface {} {}x{} row_bytes={} first_pixel_BGRA={:?}",
            image.surface_id(),
            width,
            height,
            row_bytes,
            &pixels[..4]
        );
        pixels
    }

    fn median_ms(samples: &[Duration]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2].as_secs_f64() * 1_000.0
    }

    #[test]
    fn three_frames_reach_the_sink_with_monotonic_indexes_and_distinct_surfaces() {
        let transport = open_transport();
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        let clears = [
            ClearColor::OPAQUE_RED,
            ClearColor::OPAQUE_GREEN,
            ClearColor::OPAQUE_BLUE,
        ];
        let mut images: Vec<SurfaceImage> = (0..3)
            .map(|_| create_surface(&transport, 64, 64))
            .collect();
        let mut latencies = Vec::with_capacity(3);

        for (index, image) in images.iter_mut().enumerate() {
            let started = Instant::now();
            path.present(image, clears[index], &mut sink)
                .expect("present should reach the sink");
            latencies.push(started.elapsed());
            path.finish_present().expect("frame should finish");
        }

        let frames = sink.frames();
        assert_eq!(frames.len(), 3, "sink must observe exactly three frames");
        let indexes: Vec<u64> = frames.iter().map(|frame| frame.frame_index).collect();
        assert_eq!(indexes, vec![0, 1, 2]);
        assert!(
            indexes.windows(2).all(|pair| pair[1] > pair[0]),
            "frame indexes must be strictly increasing: {indexes:?}"
        );

        let surface_ids: HashSet<u32> = frames.iter().map(|frame| frame.surface_id).collect();
        assert_eq!(surface_ids.len(), 3, "each frame must use a distinct surface");
        for (index, image) in images.iter().enumerate() {
            assert_eq!(frames[index].surface_id, image.surface_id());
            assert_eq!(image.frame_index(), Some(index as u64));
        }

        eprintln!(
            "NATIVE-PRESENT synthetic present latency (synthetic, not game FPS): samples={latencies:?} median={:.3}ms",
            median_ms(&latencies)
        );
    }

    #[test]
    fn presented_pixels_are_the_registered_clear_colour_in_every_pixel() {
        let transport = open_transport();
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        let clears = [
            ClearColor::OPAQUE_RED,
            ClearColor::OPAQUE_GREEN,
            ClearColor::OPAQUE_BLUE,
        ];

        for clear in clears {
            let mut image = create_surface(&transport, 64, 64);
            path.present(&mut image, clear, &mut sink)
                .expect("present should reach the sink");
            path.finish_present().expect("frame should finish");
            assert_solid(&image, clear, "clear pass");
        }
    }

    #[test]
    fn second_present_while_in_flight_errors_and_leaves_the_surface_intact() {
        let transport = open_transport();
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        let mut image = create_surface(&transport, 64, 64);

        path.present(&mut image, ClearColor::OPAQUE_RED, &mut sink)
            .expect("first present should succeed");
        let before = assert_solid(&image, ClearColor::OPAQUE_RED, "in-flight frame");

        let error = path
            .present(&mut image, ClearColor::OPAQUE_BLUE, &mut sink)
            .expect_err("second present must be refused while a frame is in flight");
        assert_eq!(
            error,
            PresentError::FrameInFlight {
                frame_index: 0,
                surface_id: image.surface_id(),
            }
        );

        let after = assert_solid(&image, ClearColor::OPAQUE_RED, "after refused overwrite");
        assert_eq!(before, after, "in-flight surface contents must not change");
        assert_eq!(sink.frames().len(), 1, "sink must not see the refused frame");

        assert_eq!(
            path.finish_present().expect("finish in-flight frame").frame_index,
            0
        );
        assert_eq!(
            path.finish_present(),
            Err(PresentError::NoFrameInFlight),
            "finishing twice must be an explicit error"
        );

        path.present(&mut image, ClearColor::OPAQUE_GREEN, &mut sink)
            .expect("present after finish should succeed");
        assert_eq!(sink.frames().len(), 2);
        assert_eq!(sink.frames()[1].frame_index, 1);
        assert_solid(&image, ClearColor::OPAQUE_GREEN, "frame after finish");
    }

    #[test]
    fn refused_frame_reports_a_documented_error_and_leaves_nothing_in_flight() {
        let transport = open_transport();
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        let mut image = create_surface(&transport, 64, 64);

        sink.refuse_next_frame();
        let error = path
            .present(&mut image, ClearColor::OPAQUE_RED, &mut sink)
            .expect_err("refused frame must fail");
        assert!(
            matches!(
                error,
                PresentError::SinkRefused {
                    frame_index: 0,
                    ..
                }
            ),
            "unexpected refusal error: {error:?}"
        );
        assert!(sink.frames().is_empty(), "refused frame must not be recorded");
        assert_eq!(path.in_flight(), None, "refusal must clear in-flight state");
        assert_eq!(path.finish_present(), Err(PresentError::NoFrameInFlight));
        assert_solid(&image, ClearColor::OPAQUE_RED, "refused frame render");

        path.present(&mut image, ClearColor::OPAQUE_GREEN, &mut sink)
            .expect("present after refusal should succeed");
        assert_eq!(sink.frames().len(), 1);
        assert_eq!(
            sink.frames()[0].frame_index, 1,
            "a refused frame consumes its index, so indexes stay monotonic"
        );
    }

    #[test]
    fn two_surfaces_present_distinct_frames_with_interval_and_refusal_discipline() {
        use crate::transport::MetalTransport;

        let transport = open_transport();
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        let mut first = create_surface(&transport, 40, 24);
        let mut second = create_surface(&transport, 40, 24);

        // Frame 0: immediate pace, rendered into the first surface.
        path.present_with_interval(
            &mut first,
            ClearColor::OPAQUE_RED,
            SyncInterval::Immediate,
            &mut sink,
        )
        .expect("frame 0 should reach the sink");
        let in_flight = path.in_flight().expect("frame 0 must be in flight");
        assert_eq!(in_flight.frame_index, 0);
        assert_eq!(in_flight.surface_id, first.surface_id());
        assert_eq!(in_flight.sync_interval, SyncInterval::Immediate);
        let first_rows = assert_solid(&first, ClearColor::OPAQUE_RED, "frame 0");

        // Exactly one frame may be in flight: the second surface is refused
        // before it is encoded, so its contents stay untouched.
        let error = path
            .present_with_interval(
                &mut second,
                ClearColor::OPAQUE_GREEN,
                SyncInterval::Vsync,
                &mut sink,
            )
            .expect_err("a second present must be refused while a frame is in flight");
        assert_eq!(
            error,
            PresentError::FrameInFlight {
                frame_index: 0,
                surface_id: first.surface_id(),
            }
        );
        assert_eq!(second.frame_index(), None, "refused frame must not render");
        assert_eq!(
            sink.frames().len(),
            1,
            "sink must not see the refused frame"
        );
        assert_eq!(
            assert_solid(&first, ClearColor::OPAQUE_RED, "frame 0 after refusal"),
            first_rows,
            "the in-flight surface must not change"
        );

        path.finish_present().expect("frame 0 should finish");

        // Frame 1: vsync pace, rendered into the second surface.
        path.present_with_interval(
            &mut second,
            ClearColor::OPAQUE_GREEN,
            SyncInterval::Vsync,
            &mut sink,
        )
        .expect("frame 1 should reach the sink");
        let in_flight = path.in_flight().expect("frame 1 must be in flight");
        assert_eq!(in_flight.frame_index, 1);
        assert_eq!(in_flight.surface_id, second.surface_id());
        assert_eq!(in_flight.sync_interval, SyncInterval::Vsync);
        let second_rows = assert_solid(&second, ClearColor::OPAQUE_GREEN, "frame 1");
        assert_ne!(
            first_rows, second_rows,
            "the two frames must carry distinct clear colours"
        );
        path.finish_present().expect("frame 1 should finish");

        let frames = sink.frames();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].frame_index, 0);
        assert_eq!(frames[1].frame_index, 1);
        assert_eq!(frames[0].surface_id, first.surface_id());
        assert_eq!(frames[1].surface_id, second.surface_id());
        assert_eq!(frames[0].sync_interval, SyncInterval::Immediate);
        assert_eq!(frames[1].sync_interval, SyncInterval::Vsync);

        // A refused sink is an explicit error; the rendered frame is not
        // dropped from the surface even though the sink took no ownership.
        sink.refuse_next_frame();
        let error = path
            .present_with_interval(
                &mut first,
                ClearColor::OPAQUE_BLUE,
                SyncInterval::Vsync,
                &mut sink,
            )
            .expect_err("a refusing sink must surface an explicit error");
        assert!(
            matches!(error, PresentError::SinkRefused { frame_index: 2, .. }),
            "unexpected refusal error: {error:?}"
        );
        assert_eq!(path.in_flight(), None, "refusal must clear in-flight state");
        assert_eq!(sink.frames().len(), 2, "refused frame must not be recorded");
        assert_solid(&first, ClearColor::OPAQUE_BLUE, "refused frame render");

        // The refused frame consumed its index; presentation continues at 3.
        path.present_with_interval(
            &mut first,
            ClearColor::OPAQUE_BLUE,
            SyncInterval::Vsync,
            &mut sink,
        )
        .expect("present after refusal should succeed");
        assert_eq!(sink.frames()[2].frame_index, 3);
        path.finish_present().expect("frame 3 should finish");

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "native present interval/refusal gate: device={}, surfaces=[{},{}], frames=[0,1], \
             intervals=[immediate,vsync], refused_frame_index=2",
            device.name_str(),
            first.surface_id(),
            second.surface_id()
        );
    }

    #[test]
    fn unsupported_sync_interval_is_rejected_explicitly() {
        assert_eq!(
            SyncInterval::from_wire(2),
            Err(PresentError::UnsupportedSyncInterval { sync_interval: 2 })
        );
        assert_eq!(SyncInterval::from_wire(0), Ok(SyncInterval::Immediate));
        assert_eq!(SyncInterval::from_wire(1), Ok(SyncInterval::Vsync));
        assert_eq!(SyncInterval::Vsync.as_wire(), 1);
    }

    #[test]
    fn zero_sized_surfaces_are_rejected_before_surface_creation() {
        let transport = open_transport();
        for (width, height) in [(0, 64), (64, 0), (0, 0)] {
            let error = SurfaceImage::create(&transport, width, height)
                .err()
                .expect("zero extent must be rejected");
            assert_eq!(error, PresentError::ZeroExtent { width, height });
        }
    }

    #[test]
    fn frame_probe_is_off_by_default_and_the_opt_in_is_explicit() {
        assert_eq!(FrameProbeConfig::from_values(None, None), None);
        for off in ["", "0", "off", "false", "no", "  "] {
            assert_eq!(
                FrameProbeConfig::from_values(Some(off), None),
                None,
                "{off:?} must not enable the probe"
            );
        }
        for on in ["1", "on", "true", "yes", " TRUE "] {
            assert!(
                FrameProbeConfig::from_values(Some(on), None).is_some(),
                "{on:?} must enable the probe"
            );
        }
        assert_eq!(
            FrameProbeConfig::from_values(Some("1"), None),
            Some(FrameProbeConfig {
                every: FRAME_PROBE_DEFAULT_EVERY
            })
        );
        assert_eq!(
            FrameProbeConfig::from_values(Some("1"), Some("5")),
            Some(FrameProbeConfig { every: 5 })
        );
        // A malformed or zero interval falls back to the documented default.
        for invalid in ["0", "-3", "bogus", ""] {
            assert_eq!(
                FrameProbeConfig::from_values(Some("1"), Some(invalid)),
                Some(FrameProbeConfig {
                    every: FRAME_PROBE_DEFAULT_EVERY
                }),
                "{invalid:?}"
            );
        }
        assert_eq!(parse_probe_every(" 7 "), Some(7));
        assert_eq!(parse_probe_every("0"), None);
    }

    #[test]
    fn probe_schedule_samples_frame_zero_and_every_nth_frame() {
        let sampled: Vec<u64> = (0..100).filter(|frame| probe_due(*frame, 30)).collect();
        assert_eq!(sampled, vec![0, 30, 60, 90]);
        assert_eq!(
            sampled.last().copied(),
            Some(90),
            "the report carries the last sampled frame index"
        );
        assert!(probe_due(0, 1));
        assert!(probe_due(1, 1));
        assert!(!probe_due(0, 0), "zero interval must be inert");
        assert!(!probe_due(1, 0));
    }

    #[test]
    fn probe_threshold_counts_only_pixels_beyond_the_clear() {
        let clear = [10, 20, 30, 200];
        let all_clear = vec![clear; 64];
        let sample = analyze_bgra8(&all_clear, clear, FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.non_black_pixels, 0, "an all-clear frame has no content");
        assert_eq!(sample.unique_luma_buckets, 1);
        assert_eq!(sample.probe_pixels, 64);

        // A difference of exactly the threshold is still clear.
        let mut at_threshold = all_clear.clone();
        at_threshold[3][0] = clear[0] + FRAME_PROBE_CHANNEL_THRESHOLD;
        assert_eq!(
            analyze_bgra8(&at_threshold, clear, FRAME_PROBE_CHANNEL_THRESHOLD).non_black_pixels,
            0
        );

        // One step beyond the threshold on any channel counts exactly one pixel.
        for channel in 0..4 {
            let mut one = all_clear.clone();
            one[7][channel] = clear[channel] + FRAME_PROBE_CHANNEL_THRESHOLD + 1;
            let sample = analyze_bgra8(&one, clear, FRAME_PROBE_CHANNEL_THRESHOLD);
            assert_eq!(sample.non_black_pixels, 1, "channel {channel}");
        }
    }

    #[test]
    fn probe_luma_buckets_are_five_bit_and_a_gradient_spans_all_of_them() {
        // Every 8-bit gray level maps to one 5-bit bucket.
        let gradient: Vec<[u8; 4]> = (0..=255u16)
            .map(|value| {
                let level = u8::try_from(value).expect("0..=255 fits u8");
                [level, level, level, 255]
            })
            .collect();
        let sample = analyze_bgra8(&gradient, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.unique_luma_buckets, 32);
        assert_eq!(sample.probe_pixels, 256);

        // Values 0..=7 share bucket 0; 8 starts bucket 1.
        let same_bucket = vec![[0, 0, 0, 255], [7, 7, 7, 255]];
        assert_eq!(
            analyze_bgra8(&same_bucket, [0, 0, 0, 255], 0).unique_luma_buckets,
            1
        );
        let next_bucket = vec![[7, 7, 7, 255], [8, 8, 8, 255]];
        assert_eq!(
            analyze_bgra8(&next_bucket, [0, 0, 0, 255], 0).unique_luma_buckets,
            2
        );
    }

    #[test]
    fn probe_crc_is_stable_sensitive_and_matches_the_ecma182_check_value() {
        // CRC-64/ECMA-182 standard check value for the ASCII bytes "123456789".
        assert_eq!(crc64_ecma_update(0, b"123456789"), 0x6C40_DF5F_0B49_7347);
        assert_eq!(frame_crc64(&[]), 0);

        let frame = vec![[1u8, 2, 3, 4]; 16];
        let same = frame.clone();
        assert_eq!(
            frame_crc64(&frame),
            frame_crc64(&same),
            "the CRC must be deterministic for identical pixels"
        );

        let mut one_bit = frame.clone();
        one_bit[7][2] ^= 1;
        assert_ne!(
            frame_crc64(&frame),
            frame_crc64(&one_bit),
            "a single changed channel must change the CRC"
        );
    }

    #[test]
    fn probe_report_line_names_every_telemetry_quantity() {
        let report = FrameProbeReport {
            sampled_frames: 4,
            probe_pixels: FRAME_PROBE_PIXELS,
            last_sampled_frame: Some(90),
            non_black_pixels: 12,
            unique_luma_buckets: 3,
            frame_crc: 0x0123_4567_89AB_CDEF,
            dropped_samples: 2,
            fallback_clear: false,
            probe_every: 30,
            failure: None,
        };
        let line = report.log_line();
        assert!(line.starts_with("frame_probe="), "{line}");
        for expected in [
            "frame=90",
            "non_black_pixels=12",
            "unique_luma_buckets=3",
            "sampled_frames=4",
            "probe_pixels=4096",
            "dropped_samples=2",
            "fallback_clear=0",
            "probe_every=30",
            "failure=none",
        ] {
            assert!(line.contains(expected), "{line} must contain {expected}");
        }
        assert!(line.contains("frame_crc=0123456789abcdef"));
    }

    #[test]
    #[ignore = "requires a Metal device: exercises the GPU staging-blit probe path"]
    fn frame_probe_gpu_staging_blit_copies_the_presented_texture() {
        let transport = open_transport();
        let mut image = create_surface(&transport, 64, 64);
        let mut path = PresentPath::new(&transport);
        let mut sink = RecordingSink::new();
        path.present(&mut image, ClearColor::OPAQUE_RED, &mut sink)
            .expect("present should reach the sink");
        path.finish_present().expect("frame should finish");

        let layout = ProbeLayout::new(64, 64).expect("64x64 layout");
        let staging = create_probe_staging_texture(
            transport.device_raw(),
            MTL_PIXEL_FORMAT_BGRA8_UNORM,
            layout.staging_width,
            layout.staging_height,
        )
        .expect("probe staging texture");

        // SAFETY: `queue` is the live transport queue and both textures are
        // live 64x64 BGRA8 textures.
        unsafe {
            let command_buffer =
                objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer"));
            assert!(!command_buffer.is_null());
            let encoder =
                objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder"));
            assert!(!encoder.is_null());
            for grid_y in 0..FRAME_PROBE_GRID {
                for grid_x in 0..FRAME_PROBE_GRID {
                    encode_texture_block_copy(
                        encoder,
                        image.texture_raw(),
                        layout.block_origin(grid_x, grid_y, 64, 64),
                        layout.block_width,
                        layout.block_height,
                        staging.as_raw(),
                        layout.staging_origin(grid_x, grid_y),
                    );
                }
            }
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));
        }

        let mut pixels = vec![[0u8; 4]; 64 * 64];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc::MTLSize {
                width: 64,
                height: 64,
                depth: 1,
            },
        };
        // SAFETY: the command buffer completed above; `pixels` holds the whole
        // 64x64 staging texture at a matching row pitch.
        unsafe {
            get_texture_bytes(
                staging.as_raw(),
                pixels.as_mut_ptr().cast(),
                64 * 4,
                region,
            );
        }
        let expected = expected_bgra(ClearColor::OPAQUE_RED);
        let mismatch = pixels.iter().position(|pixel| *pixel != expected);
        assert_eq!(mismatch, None, "staging copy differs at pixel {mismatch:?}");
    }
}
