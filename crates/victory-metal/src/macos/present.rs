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

use crate::macos::blit;
use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};
use crate::macos::trace;

/// `MTLPixelFormatBGRA8Unorm`, matching the IOSurface `'BGRA'` format below.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
/// `MTLPixelFormatRGBA8Unorm`, the AAA frame colour-target format.
const MTL_PIXEL_FORMAT_RGBA8_UNORM: usize = 70;
/// `MTLPixelFormatRGBA16Float`, the title's FP16 present target. It is a
/// displayable `CAMetalLayer` format on Apple silicon, so a 16F back-buffer is
/// copied rather than replaced by the diagnostic clear.
const MTL_PIXEL_FORMAT_RGBA16_FLOAT: usize = 115;
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
/// `'RGBA'` four-character code, matching `MTLPixelFormatRGBA8Unorm` (70).
const IOSURFACE_PIXEL_FORMAT_RGBA: i32 = 0x5247_4241;
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

/// `CGSize` (`CGFloat width, height`; `CGFloat` is `f64` on Apple silicon and
/// x86-64).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

/// Sets a `CAMetalLayer`'s drawable size, the extent of the next drawable.
///
/// # Safety
///
/// `layer` must be a live `CAMetalLayer` that owns no drawable.
unsafe fn set_drawable_size(layer: objc::Id, width: u32, height: u32) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, CGSize);
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `layer` is live and `CGSize` is passed by value as the selector
    // documents.
    unsafe {
        send(
            layer,
            objc::sel(c"setDrawableSize:"),
            CGSize {
                width: f64::from(width),
                height: f64::from(height),
            },
        );
    }
}

/// Whether a 4-channel format can back a `CAMetalLayer` drawable and a blit.
///
/// RGBA16Float is displayable on Apple silicon, so the title's FP16
/// back-buffer is copied instead of silently replaced by the diagnostic clear.
fn drawable_format_supported(format: usize) -> bool {
    matches!(
        format,
        MTL_PIXEL_FORMAT_BGRA8_UNORM | MTL_PIXEL_FORMAT_RGBA8_UNORM | MTL_PIXEL_FORMAT_RGBA16_FLOAT
    )
}

/// Bytes per pixel of a 4-channel format the present path can carry.
///
/// `BGRA8Unorm`/`RGBA8Unorm` are 4 bytes; `RGBA16Float` is 8. This is the
/// multiplier every sample byte offset uses, so an 8-byte format is proven by
/// the same layout maths as the 4-byte one even though the analyser currently
/// only reads 8-bit samples.
fn pixel_format_bytes(format: usize) -> Option<u32> {
    match format {
        MTL_PIXEL_FORMAT_BGRA8_UNORM | MTL_PIXEL_FORMAT_RGBA8_UNORM => Some(4),
        MTL_PIXEL_FORMAT_RGBA16_FLOAT => Some(8),
        _ => None,
    }
}

/// The Metal name of a pixel format the probe reports, for the log line.
fn pixel_format_name(format: usize) -> &'static str {
    match format {
        MTL_PIXEL_FORMAT_BGRA8_UNORM => "bgra8unorm",
        MTL_PIXEL_FORMAT_RGBA8_UNORM => "rgba8unorm",
        MTL_PIXEL_FORMAT_RGBA16_FLOAT => "rgba16float",
        _ => "unknown",
    }
}

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

// ---------------------------------------------------------------------------
// Bounded present trace
// ---------------------------------------------------------------------------
//
// One line per process proves which identity `Present` actually showed and
// whether anything wrote that texture before it was shown. The trace is
// counts only: no pixel payload, no title asset, no dump. It is bounded to
// [`PRESENT_TRACE_MAX_KEYS`] distinct keys so a long boot cannot grow memory.

/// Largest number of distinct present keys the trace remembers.
pub const PRESENT_TRACE_MAX_KEYS: usize = 16;

/// What wrote the presented source texture before this frame was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentCopyKind {
    /// No host copy into the source was observed.
    None,
    /// A texture-region copy (guest `CopyResource` lowers to this).
    CopyTextureRegion,
    /// A multisample resolve.
    ResolveSubresource,
}

impl PresentCopyKind {
    /// The stable telemetry name of this kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CopyTextureRegion => "copy_texture_region",
            Self::ResolveSubresource => "resolve_subresource",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PresentTraceKey {
    copy_kind: PresentCopyKind,
    source_is_presented: bool,
    probe_sampled_presented: bool,
    fallback_clear: bool,
}

/// The immutable descriptor of a texture the present path showed.
///
/// Counts and shapes only: a format, an extent and a usage mask. No content,
/// no title asset name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SurfaceDescriptor {
    format: usize,
    width: usize,
    height: usize,
    usage: usize,
}

/// One presented source texture in the bounded identity census.
///
/// The key is the texture's process-local address, which is counted only; it
/// never reaches telemetry. `received_copy` and `seen_as_pass` answer "did any
/// host copy or traced render pass name this exact object?".
#[derive(Debug, Clone, Copy)]
struct SurfaceCensusEntry {
    texture: usize,
    descriptor: SurfaceDescriptor,
    received_copy: bool,
    seen_as_pass: bool,
    descriptor_mismatch: bool,
}

/// One distinct pass target the draw tracer observed at a present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PassTargetFacts {
    format: &'static str,
    width: u32,
    height: u32,
    /// The pass target is one of the host's registered surfaces.
    swapchain: bool,
    /// The pass target is the exact texture the present showed.
    is_presented_surface: bool,
}

/// One distinct `MTLTexture` object the present path observed with its
/// immutable descriptor.
///
/// Objects are recorded from three places that can reach the same guest
/// texture: the registered present surface, a traced render-pass attachment and
/// a blit copy destination (`blit::note_copy_destination`). Counting distinct
/// process-local addresses per descriptor is how the host proves "one object
/// per allocation"; no address reaches telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextureObjectEntry {
    descriptor: SurfaceDescriptor,
    texture: usize,
    origin: TextureObjectOrigin,
}

/// Where the present path observed one `MTLTexture` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureObjectOrigin {
    /// The registered present surface / presented source.
    Surface,
    /// A traced render-pass colour attachment.
    PassTarget,
    /// A blit copy or resolve destination.
    CopyDestination,
}

/// Distinct `MTLTexture` objects the host remembers for the identity census.
const TEXTURE_OBJECTS_MAX: usize = 32;

#[derive(Debug, Default)]
struct PresentTrace {
    frames: u64,
    copied_into_source: u64,
    probe_source_mismatch: u64,
    fallback_clear: u64,
    keys: Vec<(PresentTraceKey, u64)>,
    dropped_keys: u64,
    /// Copy count observed at the previous present of each source texture, so
    /// a copy from an earlier frame is not counted again.
    source_copies: Vec<(usize, u64)>,
    dropped_sources: u64,
    /// Distinct presented source textures, each with its descriptor and the
    /// copy/pass verdict. Bounded to [`PRESENT_TRACE_MAX_KEYS`].
    surfaces: Vec<SurfaceCensusEntry>,
    /// Distinct pass targets observed at present time, bounded.
    pass_targets: Vec<(PassTargetFacts, u64)>,
    /// A source texture appeared again with a different descriptor.
    surface_registered_twice: u64,
    /// A pass target named the presented texture with a different descriptor,
    /// or a presented texture's descriptor changed while it was shown.
    surface_descriptor_mismatch: u64,
    /// Distinct pass targets that are another registered surface.
    surface_pass_target_other_surface: u64,
    /// Distinct pass targets that are unregistered (offscreen) textures.
    surface_pass_target_offscreen: u64,
    /// Presented textures whose census entry did not fit the bound.
    dropped_surfaces: u64,
    /// Distinct `MTLTexture` objects seen as a surface, pass target or copy
    /// destination, with their descriptor and provenance.
    texture_objects: Vec<TextureObjectEntry>,
    /// Texture objects dropped after the bound was reached.
    dropped_texture_objects: u64,
    /// Presents whose traced pass target had the surface's descriptor but was a
    /// different object (a different allocation, or a stale sampled pass).
    surface_pass_target_same_descriptor_other_object: u64,
    /// Copy events observed at the previous present, the composition marker.
    copy_events_seen: u64,
    /// Presents whose traced pass target was the presented surface object.
    surface_path_pass_target: u64,
    /// Presents whose surface received a copy since the previous present.
    surface_path_copy_dest: u64,
    /// Presents where the pass target shared the surface's guest handle but was
    /// still a distinct `MTLTexture` (the alias was never created).
    surface_path_alias_missing: u64,
    /// Presents where a different allocation with the surface's descriptor was
    /// the pass target and no copy reached the surface.
    surface_path_unrelated: u64,
    /// Copies (region copies and resolves) between the last two presents.
    composition_copies: u64,
    /// Those copies whose source was one of the traced pass targets.
    composition_copy_source_is_pass_target: u64,
    /// Those copies whose destination was the presented surface.
    composition_copy_dest_is_surface: u64,
    /// Presents with no copy at all since the previous present.
    composition_present_without_copy: u64,
}

static PRESENT_TRACE: OnceLock<Mutex<PresentTrace>> = OnceLock::new();

fn present_trace() -> &'static Mutex<PresentTrace> {
    PRESENT_TRACE.get_or_init(|| Mutex::new(PresentTrace::default()))
}

/// One-shot pass-target identity a caller declares for the next present.
///
/// The bounded sampling tracer is opt-in and process-wide; a hardware identity
/// case knows the exact texture it just encoded as the render-pass target and
/// declares it here instead of depending on which pass the sampler happened to
/// pick. The census then compares that object with the presented source and the
/// copy destination without arming the tracer's synchronous readback. Taken
/// (cleared) by the next present so it cannot leak across frames.
static IDENTITY_PASS_TARGET: Mutex<Option<(usize, bool)>> = Mutex::new(None);

/// Declares the object a hardware identity case just used as the pass target.
///
/// Public and hidden; the host never calls it. It is an identity token only
/// (no descriptor is read from it), so a stale value can never be dereferenced.
#[doc(hidden)]
pub fn note_identity_pass_target(texture: usize) {
    note_declared_pass_target(texture, true);
}

/// Declares a pass target that is *not* one of the host's registered surfaces.
///
/// The split a composition conflict names is between the presented surface and
/// an unregistered (offscreen) allocation; this hook records exactly that, so a
/// hardware case can prove the conflict without arming the sampling tracer.
#[doc(hidden)]
pub fn note_offscreen_pass_target(texture: usize) {
    note_declared_pass_target(texture, false);
}

fn note_declared_pass_target(texture: usize, swapchain: bool) {
    if texture == 0 {
        return;
    }
    if let Ok(mut slot) = IDENTITY_PASS_TARGET.lock() {
        *slot = Some((texture, swapchain));
    }
}

/// Takes and clears the declared pass target and its surface classification.
fn take_identity_pass_target() -> Option<(usize, bool)> {
    IDENTITY_PASS_TARGET
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

impl DirectMetalTransport {
    /// Object identity of a texture the transport registered for a pass target.
    ///
    /// Public and hidden for the hardware identity gate; the frame path uses
    /// [`resolve_texture`](crate::macos::device) directly. Two handles that name
    /// one guest allocation must return the same address.
    #[doc(hidden)]
    pub fn registered_texture_object(&self, id: u64) -> Option<usize> {
        let state = self.lock_state().ok()?;
        state
            .textures
            .get(&id)
            .map(|texture| texture.as_raw() as usize)
    }

    /// Object identity of the texture registered as present surface `surface_id`.
    #[doc(hidden)]
    pub fn registered_surface_object(&self, surface_id: u64) -> Option<usize> {
        let state = self.lock_state().ok()?;
        state
            .surfaces
            .get(&surface_id)
            .map(|texture| texture.as_raw() as usize)
    }
}

/// Reads the immutable descriptor of a live `MTLTexture`.
///
/// Only called on real textures: the unit tests hand [`record_present_trace`]
/// synthetic addresses and pass `None` here.
fn surface_descriptor_of(texture: objc::Id) -> Option<SurfaceDescriptor> {
    if texture.is_null() {
        return None;
    }
    // SAFETY: `texture` is a live `MTLTexture`; each accessor only reads an
    // immutable property.
    Some(unsafe {
        SurfaceDescriptor {
            format: objc::msg_send_usize_0(texture, objc::sel(c"pixelFormat")),
            width: objc::msg_send_usize_0(texture, objc::sel(c"width")),
            height: objc::msg_send_usize_0(texture, objc::sel(c"height")),
            usage: objc::msg_send_usize_0(texture, objc::sel(c"usage")),
        }
    })
}

/// The pass targets the draw tracer observed, when it is armed.
///
/// The sampled pass and the most recent described draw are both real traced
/// pass targets, and the surface may be either one depending on which pass the
/// bounded tracer happened to sample. Both are `None` with the tracer off, which
/// the present trace records as "unarmed" rather than as "no pass".
fn observed_pass_targets() -> [Option<trace::TraceTarget>; 2] {
    [
        trace::last_sample().and_then(|sample| sample.target),
        trace::last_record().and_then(|record| record.target),
    ]
}

/// Records one live `MTLTexture` in the bounded object identity census.
///
/// Called with real textures only: the registered present surface, a traced
/// pass attachment and a blit/copy destination. Tests that pass synthetic
/// tokens use [`record_present_trace`] with no descriptor, which skips this.
/// Recording the same pointer twice is idempotent; the first provenance wins.
pub(crate) fn note_texture_object(texture: objc::Id, origin: TextureObjectOrigin) {
    let Some(descriptor) = surface_descriptor_of(texture) else {
        return;
    };
    let key = texture as usize;
    let Ok(mut trace) = present_trace().lock() else {
        return;
    };
    if trace
        .texture_objects
        .iter()
        .any(|entry| entry.texture == key)
    {
        return;
    }
    if trace.texture_objects.len() >= TEXTURE_OBJECTS_MAX {
        trace.dropped_texture_objects = trace.dropped_texture_objects.saturating_add(1);
        return;
    }
    trace.texture_objects.push(TextureObjectEntry {
        descriptor,
        texture: key,
        origin,
    });
}

/// Records a blit/copy destination object for the identity census.
pub(crate) fn note_copy_destination_object(texture: objc::Id) {
    note_texture_object(texture, TextureObjectOrigin::CopyDestination);
}

/// Updates the bounded surface census for one presented source texture.
fn note_surface_census(
    trace: &mut PresentTrace,
    source: objc::Id,
    descriptor: Option<SurfaceDescriptor>,
    pass_target: Option<trace::TraceTarget>,
    copy_total: u64,
) {
    let Some(descriptor) = descriptor else {
        return;
    };
    if source.is_null() {
        return;
    }
    let texture = source as usize;
    let index = trace
        .surfaces
        .iter()
        .position(|entry| entry.texture == texture);
    let index = match index {
        Some(index) => {
            let entry = &mut trace.surfaces[index];
            if entry.descriptor != descriptor {
                trace.surface_registered_twice = trace.surface_registered_twice.saturating_add(1);
                if !entry.descriptor_mismatch {
                    entry.descriptor_mismatch = true;
                    trace.surface_descriptor_mismatch =
                        trace.surface_descriptor_mismatch.saturating_add(1);
                }
                entry.descriptor = descriptor;
            }
            index
        }
        None => {
            if trace.surfaces.len() >= PRESENT_TRACE_MAX_KEYS {
                trace.dropped_surfaces = trace.dropped_surfaces.saturating_add(1);
                return;
            }
            trace.surfaces.push(SurfaceCensusEntry {
                texture,
                descriptor,
                received_copy: false,
                seen_as_pass: false,
                descriptor_mismatch: false,
            });
            trace.surfaces.len() - 1
        }
    };

    if copy_total > 0 {
        trace.surfaces[index].received_copy = true;
    }

    let Some(target) = pass_target else {
        return;
    };
    let facts = PassTargetFacts {
        format: target.format,
        width: target.width,
        height: target.height,
        swapchain: target.swapchain,
        is_presented_surface: target.texture == texture,
    };
    let descriptor = trace.surfaces[index].descriptor;
    let descriptor_matches = facts.format == pixel_format_name(descriptor.format)
        && usize::try_from(facts.width).ok() == Some(descriptor.width)
        && usize::try_from(facts.height).ok() == Some(descriptor.height);
    if facts.is_presented_surface {
        trace.surfaces[index].seen_as_pass = true;
        if !descriptor_matches && !trace.surfaces[index].descriptor_mismatch {
            trace.surfaces[index].descriptor_mismatch = true;
            trace.surface_descriptor_mismatch = trace.surface_descriptor_mismatch.saturating_add(1);
        }
    } else if descriptor_matches {
        // The tracer named a different object with the surface's shape. It can
        // be the pass that intentionally targets another allocation of that
        // shape, or the bounded tracer's stale sample; either way the surface
        // object is not this target, so the verdict stays false and the reason
        // is named instead of being folded into "no pass".
        trace.surface_pass_target_same_descriptor_other_object = trace
            .surface_pass_target_same_descriptor_other_object
            .saturating_add(1);
    }
    if !facts.is_presented_surface {
        if facts.swapchain {
            if !trace
                .pass_targets
                .iter()
                .any(|(seen, _)| seen.swapchain && !seen.is_presented_surface)
            {
                trace.surface_pass_target_other_surface =
                    trace.surface_pass_target_other_surface.saturating_add(1);
            }
        } else if !trace.pass_targets.iter().any(|(seen, _)| !seen.swapchain) {
            trace.surface_pass_target_offscreen =
                trace.surface_pass_target_offscreen.saturating_add(1);
        }
    }
    match trace
        .pass_targets
        .iter_mut()
        .find(|(seen, _)| *seen == facts)
    {
        Some((_, count)) => *count = count.saturating_add(1),
        None => {
            if trace.pass_targets.len() < PRESENT_TRACE_MAX_KEYS {
                trace.pass_targets.push((facts, 1));
            }
        }
    }
}

/// Records one presented frame.
///
/// `source` is the texture `Present` was asked to show, `presented` is the
/// texture the window server actually shows, and `probe_sampled` is the
/// texture the probe read (`None` when it did not sample this frame). The
/// source and presented tokens are object identity, not pixel payload.
/// `descriptor` is the source's immutable shape (read from the live texture,
/// `None` for a synthetic test token) and `pass_target` is the render-pass
/// attachment the draw tracer last observed.
#[cfg(test)]
fn record_present_trace(
    source: objc::Id,
    presented: objc::Id,
    probe_sampled: Option<objc::Id>,
    fallback_clear: bool,
    descriptor: Option<SurfaceDescriptor>,
    pass_target: Option<trace::TraceTarget>,
) {
    record_present_trace_targets(
        source,
        presented,
        probe_sampled,
        fallback_clear,
        source == presented,
        descriptor,
        take_identity_pass_target(),
        &[pass_target],
    );
}

/// [`record_present_trace`] over every pass target the tracer observed.
///
/// `registered_source` is true when `source` is the texture the host registered
/// as this present's surface (the headless image itself, or the window's
/// registered present texture). Together with `source == presented` it is the
/// identity verdict: the texture being shown is the surface host work targeted.
/// `explicit_pass_target` is the one-shot identity target taken by the caller,
/// so the same evidence that chose the sampled object is the evidence the
/// verdict is computed from.
#[allow(clippy::too_many_arguments)]
fn record_present_trace_targets(
    source: objc::Id,
    presented: objc::Id,
    probe_sampled: Option<objc::Id>,
    fallback_clear: bool,
    registered_source: bool,
    descriptor: Option<SurfaceDescriptor>,
    explicit_pass_target: Option<(usize, bool)>,
    pass_targets: &[Option<trace::TraceTarget>],
) {
    // Object identity is recorded before the trace lock is taken: the census
    // reads immutable texture properties and takes the same lock. `descriptor`
    // is `Some` only for a live texture (a synthetic test token passes `None`),
    // so no selector is ever sent to a fake address.
    if descriptor.is_some() && !source.is_null() {
        note_texture_object(source, TextureObjectOrigin::Surface);
    }
    for target in pass_targets.iter().flatten() {
        if target.texture != 0 {
            note_texture_object(target.texture as objc::Id, TextureObjectOrigin::PassTarget);
        }
    }
    if let Some((object, _)) = explicit_pass_target {
        note_texture_object(object as objc::Id, TextureObjectOrigin::PassTarget);
    }
    let copy_total = if source.is_null() {
        0
    } else {
        blit::copy_count_for(source)
    };
    let Ok(mut trace) = present_trace().lock() else {
        return;
    };
    let previous_copies = if source.is_null() {
        0
    } else {
        let key = source as usize;
        match trace
            .source_copies
            .iter()
            .position(|(seen, _)| *seen == key)
        {
            Some(index) => {
                let previous = trace.source_copies[index].1;
                trace.source_copies[index].1 = copy_total;
                previous
            }
            None => {
                if trace.source_copies.len() < PRESENT_TRACE_MAX_KEYS {
                    trace.source_copies.push((key, copy_total));
                } else {
                    trace.dropped_sources = trace.dropped_sources.saturating_add(1);
                }
                0
            }
        }
    };
    // A copy that happened before this frame's previous present is not part of
    // this frame; only a new copy makes the source "written for this frame".
    let copied_since_last_present = copy_total > previous_copies;
    let copy_kind = if copied_since_last_present {
        if source.is_null() {
            PresentCopyKind::None
        } else {
            match blit::copy_kind_for(source) {
                Some(blit::BlitCopyKind::CopyTextureRegion) => PresentCopyKind::CopyTextureRegion,
                Some(blit::BlitCopyKind::ResolveSubresource) => PresentCopyKind::ResolveSubresource,
                None => PresentCopyKind::None,
            }
        }
    } else {
        PresentCopyKind::None
    };
    let key = PresentTraceKey {
        copy_kind,
        // Identity: the source is the texture the host registered for this
        // present, or it is literally the drawable (headless). In the windowed
        // path the source is the registered surface and the drawable is its
        // display copy, so `source == presented` alone would always read 0.
        source_is_presented: !source.is_null() && (source == presented || registered_source),
        probe_sampled_presented: probe_sampled.is_some_and(|sampled| sampled == presented),
        fallback_clear,
    };
    // The composition census: what happened between the previous present and
    // this one. A copy is named by its source and destination object, so the
    // three candidate handoffs (pass target, copy into the surface, neither)
    // are counted rather than inferred.
    let surface_key = source as usize;
    let copy_events_total = blit::copy_event_total();
    let copy_mark = trace.copy_events_seen;
    trace.copy_events_seen = copy_events_total;
    let copies_now = copy_events_total.saturating_sub(copy_mark);
    let copy_events = blit::copy_events_since(copy_mark);
    let mut target_keys: Vec<usize> = pass_targets
        .iter()
        .flatten()
        .map(|target| target.texture)
        .filter(|key| *key != 0)
        .collect();
    if let Some((object, _)) = explicit_pass_target {
        target_keys.push(object);
    }
    let mut copy_dest_is_surface = 0u64;
    let mut copy_source_is_target = 0u64;
    if !source.is_null() {
        for event in &copy_events {
            if event.destination == surface_key {
                copy_dest_is_surface = copy_dest_is_surface.saturating_add(1);
            }
            if target_keys.contains(&event.source) {
                copy_source_is_target = copy_source_is_target.saturating_add(1);
            }
        }
    }
    trace.composition_copies = trace.composition_copies.saturating_add(copies_now);
    trace.composition_copy_dest_is_surface = trace
        .composition_copy_dest_is_surface
        .saturating_add(copy_dest_is_surface);
    trace.composition_copy_source_is_pass_target = trace
        .composition_copy_source_is_pass_target
        .saturating_add(copy_source_is_target);
    if copies_now == 0 {
        trace.composition_present_without_copy =
            trace.composition_present_without_copy.saturating_add(1);
    }
    let target_is_surface = !source.is_null() && target_keys.contains(&surface_key);
    let copy_into_surface =
        copy_dest_is_surface > 0 || (copied_since_last_present && !target_is_surface);
    let mut foreign_same_descriptor = false;
    let mut foreign_same_handle = false;
    if let Some(shape) = descriptor
        && !target_is_surface
    {
        for key in &target_keys {
            if *key == surface_key {
                continue;
            }
            let Some(entry) = trace
                .texture_objects
                .iter()
                .find(|object| object.texture == *key)
            else {
                continue;
            };
            // Shape, not usage: a texture registered first as storage and
            // later widened for an RTV keeps the usage it was created with, so
            // the two objects of one allocation can differ there.
            if entry.descriptor.format != shape.format
                || entry.descriptor.width != shape.width
                || entry.descriptor.height != shape.height
            {
                continue;
            }
            foreign_same_descriptor = true;
            if crate::macos::registry::texture_objects_share_handle(surface_key, *key) {
                foreign_same_handle = true;
            }
        }
    }
    if target_is_surface {
        trace.surface_path_pass_target = trace.surface_path_pass_target.saturating_add(1);
    } else if copy_into_surface {
        trace.surface_path_copy_dest = trace.surface_path_copy_dest.saturating_add(1);
    } else if foreign_same_handle {
        trace.surface_path_alias_missing = trace.surface_path_alias_missing.saturating_add(1);
    } else if foreign_same_descriptor {
        trace.surface_path_unrelated = trace.surface_path_unrelated.saturating_add(1);
    }
    trace.frames = trace.frames.saturating_add(1);
    if copied_since_last_present {
        trace.copied_into_source = trace.copied_into_source.saturating_add(1);
    }
    if probe_sampled.is_some() && !key.probe_sampled_presented {
        trace.probe_source_mismatch = trace.probe_source_mismatch.saturating_add(1);
    }
    if fallback_clear {
        trace.fallback_clear = trace.fallback_clear.saturating_add(1);
    }
    let mut counted_targets: Vec<trace::TraceTarget> = Vec::new();
    for target in pass_targets.iter().flatten() {
        if counted_targets.contains(target) {
            continue;
        }
        counted_targets.push(*target);
        note_surface_census(&mut trace, source, descriptor, Some(*target), copy_total);
    }
    // A declared identity target is compared with the source's own descriptor:
    // same shape and same object is the one-object verdict; a different object
    // with the same shape is exactly the split the census must name.
    if let (Some((object, swapchain)), Some(shape)) = (explicit_pass_target, descriptor) {
        let target = trace::TraceTarget {
            texture: object,
            width: u32::try_from(shape.width).unwrap_or(u32::MAX),
            height: u32::try_from(shape.height).unwrap_or(u32::MAX),
            format: pixel_format_name(shape.format),
            swapchain,
        };
        if !counted_targets.contains(&target) {
            counted_targets.push(target);
            note_surface_census(&mut trace, source, descriptor, Some(target), copy_total);
        }
    }
    if counted_targets.is_empty() {
        note_surface_census(&mut trace, source, descriptor, None, copy_total);
    }
    match trace.keys.iter_mut().find(|(seen, _)| *seen == key) {
        Some((_, count)) => *count = count.saturating_add(1),
        None => {
            if trace.keys.len() < PRESENT_TRACE_MAX_KEYS {
                trace.keys.push((key, 1));
            } else {
                trace.dropped_keys = trace.dropped_keys.saturating_add(1);
            }
        }
    }
    let frames = trace.frames;
    drop(trace);
    // A stable line in the host log without any host-binary edit. One line per
    // 60 presented frames is bounded and enough for a boot digest.
    if frames.is_multiple_of(60) {
        eprintln!("{}", present_trace_report().log_line());
    }
}

/// Snapshot of the bounded present trace, for telemetry and tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PresentTraceReport {
    /// Frames recorded by this process.
    pub frames: u64,
    /// Frames whose source texture received a host copy before it was shown.
    pub copied_into_source: u64,
    /// Times the probe sampled a texture other than the one presented.
    pub probe_source_mismatch: u64,
    /// Frames whose drawable was the diagnostic fallback clear.
    pub fallback_clear: u64,
    /// Per-key counts, at most [`PRESENT_TRACE_MAX_KEYS`] entries.
    pub keys: Vec<PresentTraceEntry>,
    /// Keys dropped after the bound was reached.
    pub dropped_keys: u64,
    /// Source textures whose previous-copy state was not retained after the
    /// bound was reached.
    pub dropped_sources: u64,
    /// Distinct presented source textures and their identity verdict.
    pub surfaces: Vec<PresentSurfaceEntry>,
    /// Distinct pass targets the draw tracer observed at present time.
    pub pass_targets: Vec<PresentPassTargetEntry>,
    /// Distinct presented surfaces that a traced render pass named.
    pub surface_targeted_by_pass: u64,
    /// Distinct presented surfaces that a host copy wrote.
    pub surface_targeted_by_copy: u64,
    /// Times a presented source texture appeared with a different descriptor.
    pub surface_registered_twice: u64,
    /// Descriptor disagreements between the presented texture and a pass
    /// target naming the same object.
    pub surface_descriptor_mismatch: u64,
    /// Distinct pass targets that were *another* registered surface.
    pub surface_pass_targets_other_surface: u64,
    /// Distinct pass targets that were not any registered surface (offscreen).
    pub surface_pass_targets_not_surface: u64,
    /// Presented surface descriptors dropped after the bound was reached.
    pub dropped_surfaces: u64,
    /// Distinct `MTLTexture` objects seen per descriptor by the present path
    /// (surface, traced pass target and blit destination). One object per
    /// descriptor is the identity contract.
    pub texture_objects_per_descriptor: Vec<PresentTextureObjectsEntry>,
    /// The registry's histogram of `MTLTexture` objects per registered handle:
    /// `objects` is the number of distinct objects ever created for one handle,
    /// `allocations` how many handles have that count.
    pub texture_objects_per_allocation: Vec<PresentAllocationObjectsEntry>,
    /// One-word class of the composition path: `none`,
    /// `surface-is-copy-dest`, `surface-alias-not-created` or
    /// `surface-is-unrelated-allocation`.
    pub surface_object_mismatch_reason: &'static str,
    /// Presents whose traced pass target had the surface's descriptor but was a
    /// different object.
    pub surface_pass_target_same_descriptor_other_object: u64,
    /// Presents whose traced pass target was the presented surface object.
    pub surface_path_pass_target: u64,
    /// Presents whose surface received a copy since the previous present.
    pub surface_path_copy_dest: u64,
    /// Presents whose pass target shared the surface's handle but stayed a
    /// distinct object (an alias was never created).
    pub surface_path_alias_missing: u64,
    /// Presents whose different allocation with the surface's descriptor was the
    /// pass target and nothing copied into the surface.
    pub surface_path_unrelated: u64,
    /// Copies (region copies and resolves) observed between presents.
    pub composition_copies: u64,
    /// Those copies whose source was one of the traced pass targets.
    pub composition_copy_source_is_pass_target: u64,
    /// Those copies whose destination was the presented surface.
    pub composition_copy_dest_is_surface: u64,
    /// Presents with no copy at all since the previous present.
    pub composition_present_without_copy: u64,
}

/// Distinct `MTLTexture` objects seen for one descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentTextureObjectsEntry {
    /// Metal pixel-format name.
    pub format: &'static str,
    /// Level-0 width in texels.
    pub width: usize,
    /// Level-0 height in texels.
    pub height: usize,
    /// Distinct process-local texture objects with this descriptor.
    pub objects: u64,
}

/// One row of the registry's objects-per-handle histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentAllocationObjectsEntry {
    /// Distinct objects created for one registered handle.
    pub objects: u64,
    /// Handles with this count.
    pub allocations: u64,
}

/// One presented source texture as the identity census saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentSurfaceEntry {
    /// Metal pixel-format name (`bgra8unorm`, `rgba8unorm`, `rgba16float`...).
    pub format: &'static str,
    /// Level-0 width in texels.
    pub width: usize,
    /// Level-0 height in texels.
    pub height: usize,
    /// `MTLTextureUsage` mask.
    pub usage: usize,
    /// A traced render pass named this exact texture.
    pub targeted_by_pass: bool,
    /// A host copy wrote this texture.
    pub targeted_by_copy: bool,
    /// The texture's descriptor changed, or disagreed with a pass target.
    pub descriptor_mismatch: bool,
}

/// One distinct pass target observed at present time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentPassTargetEntry {
    /// Metal pixel-format name of the attachment.
    pub format: &'static str,
    /// Attachment width in texels.
    pub width: u32,
    /// Attachment height in texels.
    pub height: u32,
    /// The attachment is one of the host's registered surfaces.
    pub swapchain: bool,
    /// The attachment is the exact texture the present showed.
    pub is_presented_surface: bool,
    /// Presents that observed this target.
    pub count: u64,
}

/// One counted present key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentTraceEntry {
    /// Copy kind that wrote the source before this frame.
    pub copy_kind: PresentCopyKind,
    /// The source texture and the presented texture are the same object.
    pub source_is_presented: bool,
    /// The probe sampled the presented texture (not a different one).
    pub probe_sampled_presented: bool,
    /// The presented drawable was the diagnostic fallback clear.
    pub fallback_clear: bool,
    /// Frames with this key.
    pub count: u64,
}

/// Clears the present trace (tests).
#[cfg(test)]
fn reset_present_trace() {
    if let Ok(mut trace) = present_trace().lock() {
        *trace = PresentTrace::default();
    }
}

/// Clears the present trace for an integration test's identity gate.
///
/// Public and hidden so each hardware case starts from a clean census; the
/// host never calls it.
#[doc(hidden)]
pub fn reset_present_trace_for_test() {
    if let Ok(mut trace) = present_trace().lock() {
        *trace = PresentTrace::default();
    }
    if let Ok(mut slot) = IDENTITY_PASS_TARGET.lock() {
        *slot = None;
    }
    blit::reset_copy_trace();
}

/// One-word class of the surface / pass-target object disagreement.
///
/// The reason is derived from observed provenance, never guessed:
/// - `none` — no distinct object shared the surface's descriptor;
/// - `descriptor` — the same object disagreed about its shape;
/// - `rtv-wrapper` — a full or partial view object over one registered handle
///   was seen as the pass target (the registry records both pointers on one
///   handle; a full-resource same-format RTV is now aliased, so a remaining
///   wrapper is a real sub-range/format view);
/// - `copy-destination` — the other object was a blit/copy destination;
/// - `creation-path` — two distinct allocations (two handles) share a
///   descriptor, i.e. the guest registered the same shape under two handles;
/// - `view` — an object pair with no surface/pass/copy provenance.
fn surface_object_mismatch_reason(trace: &PresentTrace) -> &'static str {
    if trace.surface_descriptor_mismatch > 0 {
        return "descriptor";
    }
    // The composition path: a copy into the surface and an unrelated pass
    // target are different, counted outcomes. Identity stays `none`.
    if trace.surface_path_copy_dest > 0 {
        return "surface-is-copy-dest";
    }
    if trace.surface_path_alias_missing > 0 {
        return "surface-alias-not-created";
    }
    if trace.surface_path_unrelated > 0 {
        return "surface-is-unrelated-allocation";
    }
    let objects = &trace.texture_objects;
    for (index, left) in objects.iter().enumerate() {
        for right in &objects[index + 1..] {
            if left.descriptor != right.descriptor || left.texture == right.texture {
                continue;
            }
            let origins = [left.origin, right.origin];
            if origins.contains(&TextureObjectOrigin::CopyDestination)
                && (origins.contains(&TextureObjectOrigin::Surface)
                    || origins.contains(&TextureObjectOrigin::PassTarget))
            {
                return "copy-destination";
            }
            if crate::macos::registry::texture_objects_share_handle(left.texture, right.texture) {
                return "rtv-wrapper";
            }
            if origins.contains(&TextureObjectOrigin::Surface)
                && origins.contains(&TextureObjectOrigin::PassTarget)
            {
                return "creation-path";
            }
            return "view";
        }
    }
    if trace.surface_pass_target_same_descriptor_other_object > 0 {
        return "distinct-object";
    }
    "none"
}

/// The present trace accumulated so far.
pub fn present_trace_report() -> PresentTraceReport {
    let Ok(trace) = present_trace().lock() else {
        return PresentTraceReport::default();
    };
    let surfaces: Vec<PresentSurfaceEntry> = trace
        .surfaces
        .iter()
        .map(|entry| PresentSurfaceEntry {
            format: pixel_format_name(entry.descriptor.format),
            width: entry.descriptor.width,
            height: entry.descriptor.height,
            usage: entry.descriptor.usage,
            targeted_by_pass: entry.seen_as_pass,
            targeted_by_copy: entry.received_copy,
            descriptor_mismatch: entry.descriptor_mismatch,
        })
        .collect();
    let surface_targeted_by_pass = surfaces
        .iter()
        .filter(|entry| entry.targeted_by_pass)
        .count() as u64;
    let surface_targeted_by_copy = surfaces
        .iter()
        .filter(|entry| entry.targeted_by_copy)
        .count() as u64;
    let mut descriptor_objects: Vec<(SurfaceDescriptor, u64)> = Vec::new();
    for entry in &trace.texture_objects {
        match descriptor_objects
            .iter_mut()
            .find(|(descriptor, _)| *descriptor == entry.descriptor)
        {
            Some((_, count)) => *count = count.saturating_add(1),
            None => descriptor_objects.push((entry.descriptor, 1)),
        }
    }
    let texture_objects_per_descriptor = descriptor_objects
        .iter()
        .map(|(descriptor, objects)| PresentTextureObjectsEntry {
            format: pixel_format_name(descriptor.format),
            width: descriptor.width,
            height: descriptor.height,
            objects: *objects,
        })
        .collect();
    let texture_objects_per_allocation = crate::macos::registry::texture_objects_per_allocation()
        .into_iter()
        .map(|(objects, allocations)| PresentAllocationObjectsEntry {
            objects,
            allocations,
        })
        .collect();
    let surface_object_mismatch_reason = surface_object_mismatch_reason(&trace);
    PresentTraceReport {
        frames: trace.frames,
        copied_into_source: trace.copied_into_source,
        probe_source_mismatch: trace.probe_source_mismatch,
        fallback_clear: trace.fallback_clear,
        keys: trace
            .keys
            .iter()
            .map(|(key, count)| PresentTraceEntry {
                copy_kind: key.copy_kind,
                source_is_presented: key.source_is_presented,
                probe_sampled_presented: key.probe_sampled_presented,
                fallback_clear: key.fallback_clear,
                count: *count,
            })
            .collect(),
        dropped_keys: trace.dropped_keys,
        dropped_sources: trace.dropped_sources,
        pass_targets: trace
            .pass_targets
            .iter()
            .map(|(facts, count)| PresentPassTargetEntry {
                format: facts.format,
                width: facts.width,
                height: facts.height,
                swapchain: facts.swapchain,
                is_presented_surface: facts.is_presented_surface,
                count: *count,
            })
            .collect(),
        surface_targeted_by_pass,
        surface_targeted_by_copy,
        surface_registered_twice: trace.surface_registered_twice,
        surface_descriptor_mismatch: trace.surface_descriptor_mismatch,
        surface_pass_targets_other_surface: trace.surface_pass_target_other_surface,
        surface_pass_targets_not_surface: trace.surface_pass_target_offscreen,
        dropped_surfaces: trace.dropped_surfaces,
        surfaces,
        texture_objects_per_descriptor,
        texture_objects_per_allocation,
        surface_object_mismatch_reason,
        surface_pass_target_same_descriptor_other_object: trace
            .surface_pass_target_same_descriptor_other_object,
        surface_path_pass_target: trace.surface_path_pass_target,
        surface_path_copy_dest: trace.surface_path_copy_dest,
        surface_path_alias_missing: trace.surface_path_alias_missing,
        surface_path_unrelated: trace.surface_path_unrelated,
        composition_copies: trace.composition_copies,
        composition_copy_source_is_pass_target: trace.composition_copy_source_is_pass_target,
        composition_copy_dest_is_surface: trace.composition_copy_dest_is_surface,
        composition_present_without_copy: trace.composition_present_without_copy,
    }
}

impl PresentTraceReport {
    /// One stable `present_trace=...` line for the host log.
    ///
    /// Every field is a count or a stable name; no texture payload appears.
    pub fn log_line(&self) -> String {
        let count_for = |kind: PresentCopyKind| {
            self.keys
                .iter()
                .filter(|entry| entry.copy_kind == kind)
                .map(|entry| entry.count)
                .fold(0u64, u64::saturating_add)
        };
        let source_is_presented: u64 = self
            .keys
            .iter()
            .filter(|entry| entry.source_is_presented)
            .map(|entry| entry.count)
            .fold(0u64, u64::saturating_add);
        let probe_sampled_presented: u64 = self
            .keys
            .iter()
            .filter(|entry| entry.probe_sampled_presented)
            .map(|entry| entry.count)
            .fold(0u64, u64::saturating_add);
        format!(
            "present_trace=frames={} copied_into_source={} copy_kind_copy_texture_region={} \
             copy_kind_resolve_subresource={} copy_kind_none={} source_is_presented={} \
             probe_sampled_presented={} probe_source_mismatch={} fallback_clear={} \
             distinct_keys={} dropped_keys={} dropped_sources={} distinct_surfaces={} \
             surface_targeted_by_pass={} surface_targeted_by_copy={} \
             surface_registered_twice={} surface_descriptor_mismatch={} \
             surface_pass_targets_not_surface={} surface_pass_targets_other_surface={} \
             surface_object_mismatch_reason={} \
             surface_pass_target_same_descriptor_other_object={} \
             surface_path_pass_target={} surface_path_copy_dest={} \
             surface_path_alias_missing={} surface_path_unrelated={} \
             composition_copies={} composition_copy_source_is_pass_target={} \
             composition_copy_dest_is_surface={} composition_present_without_copy={} {}",
            self.frames,
            self.copied_into_source,
            count_for(PresentCopyKind::CopyTextureRegion),
            count_for(PresentCopyKind::ResolveSubresource),
            count_for(PresentCopyKind::None),
            source_is_presented,
            probe_sampled_presented,
            self.probe_source_mismatch,
            self.fallback_clear,
            self.keys.len(),
            self.dropped_keys,
            self.dropped_sources,
            self.surfaces.len(),
            self.surface_targeted_by_pass,
            self.surface_targeted_by_copy,
            self.surface_registered_twice,
            self.surface_descriptor_mismatch,
            self.surface_pass_targets_not_surface,
            self.surface_pass_targets_other_surface,
            self.surface_object_mismatch_reason,
            self.surface_pass_target_same_descriptor_other_object,
            self.surface_path_pass_target,
            self.surface_path_copy_dest,
            self.surface_path_alias_missing,
            self.surface_path_unrelated,
            self.composition_copies,
            self.composition_copy_source_is_pass_target,
            self.composition_copy_dest_is_surface,
            self.composition_present_without_copy,
            self.surface_log_fields(),
        )
    }

    /// The bounded surface identity census as stable `name=value` words.
    ///
    /// A descriptor is a format name, an extent and a usage mask: no content
    /// and no raw object address appears.
    fn surface_log_fields(&self) -> String {
        let mut out = String::new();
        for (index, surface) in self.surfaces.iter().enumerate() {
            out.push_str(&format!(
                " surface{index}={}x{}:{}:u{}:pass{}:copy{}",
                surface.width,
                surface.height,
                surface.format,
                surface.usage,
                u8::from(surface.targeted_by_pass),
                u8::from(surface.targeted_by_copy),
            ));
        }
        for (index, target) in self.pass_targets.iter().enumerate() {
            out.push_str(&format!(
                " pass_target{index}={}x{}:{}:{}:present{}",
                target.width,
                target.height,
                target.format,
                if target.swapchain {
                    "surface"
                } else {
                    "offscreen"
                },
                u8::from(target.is_presented_surface),
            ));
        }
        for (index, entry) in self.texture_objects_per_descriptor.iter().enumerate() {
            out.push_str(&format!(
                " texture_objects{index}={}x{}:{}:objects{}",
                entry.width, entry.height, entry.format, entry.objects,
            ));
        }
        for (index, entry) in self.texture_objects_per_allocation.iter().enumerate() {
            out.push_str(&format!(
                " allocation_objects{index}=objects{}:allocations{}",
                entry.objects, entry.allocations,
            ));
        }
        out
    }

    /// The report as the `present_trace` telemetry object (counts only).
    ///
    /// Field names are stable; no texture payload appears.
    pub fn json_value(&self) -> String {
        let count_for = |kind: PresentCopyKind| {
            self.keys
                .iter()
                .filter(|entry| entry.copy_kind == kind)
                .map(|entry| entry.count)
                .fold(0u64, u64::saturating_add)
        };
        let source_is_presented: u64 = self
            .keys
            .iter()
            .filter(|entry| entry.source_is_presented)
            .map(|entry| entry.count)
            .fold(0u64, u64::saturating_add);
        let probe_sampled_presented: u64 = self
            .keys
            .iter()
            .filter(|entry| entry.probe_sampled_presented)
            .map(|entry| entry.count)
            .fold(0u64, u64::saturating_add);
        let surfaces = self
            .surfaces
            .iter()
            .map(|surface| {
                format!(
                    "{{\"format\": \"{}\", \"width\": {}, \"height\": {}, \"usage\": {}, \
                     \"targeted_by_pass\": {}, \"targeted_by_copy\": {}, \
                     \"descriptor_mismatch\": {}}}",
                    surface.format,
                    surface.width,
                    surface.height,
                    surface.usage,
                    surface.targeted_by_pass,
                    surface.targeted_by_copy,
                    surface.descriptor_mismatch,
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let pass_targets = self
            .pass_targets
            .iter()
            .map(|target| {
                format!(
                    "{{\"format\": \"{}\", \"width\": {}, \"height\": {}, \
                     \"swapchain\": {}, \"is_presented_surface\": {}, \"count\": {}}}",
                    target.format,
                    target.width,
                    target.height,
                    target.swapchain,
                    target.is_presented_surface,
                    target.count,
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let texture_objects = self
            .texture_objects_per_descriptor
            .iter()
            .map(|entry| {
                format!(
                    "{{\"format\": \"{}\", \"width\": {}, \"height\": {}, \"objects\": {}}}",
                    entry.format, entry.width, entry.height, entry.objects,
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let allocation_objects = self
            .texture_objects_per_allocation
            .iter()
            .map(|entry| {
                format!(
                    "{{\"objects\": {}, \"allocations\": {}}}",
                    entry.objects, entry.allocations,
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\"frames\": {}, \"copied_into_source\": {}, \
             \"copy_kind_copy_texture_region\": {}, \
             \"copy_kind_resolve_subresource\": {}, \"copy_kind_none\": {}, \
             \"source_is_presented\": {}, \"probe_sampled_presented\": {}, \
             \"probe_source_mismatch\": {}, \"fallback_clear\": {}, \
             \"distinct_keys\": {}, \"dropped_keys\": {}, \"dropped_sources\": {}, \
             \"distinct_surfaces\": {}, \"surface_targeted_by_pass\": {}, \
             \"surface_targeted_by_copy\": {}, \"surface_registered_twice\": {}, \
             \"surface_descriptor_mismatch\": {}, \
             \"surface_pass_targets_not_surface\": {}, \
             \"surface_pass_targets_other_surface\": {}, \"dropped_surfaces\": {}, \
             \"surface_object_mismatch_reason\": \"{}\", \
             \"surface_pass_target_same_descriptor_other_object\": {}, \
             \"surface_path_pass_target\": {}, \"surface_path_copy_dest\": {}, \
             \"surface_path_alias_missing\": {}, \"surface_path_unrelated\": {}, \
             \"composition_copies\": {}, \
             \"composition_copy_source_is_pass_target\": {}, \
             \"composition_copy_dest_is_surface\": {}, \
             \"composition_present_without_copy\": {}, \
             \"surfaces\": [{}], \"pass_targets\": [{}], \
             \"texture_objects_per_descriptor\": [{}], \
             \"texture_objects_per_allocation\": [{}]}}",
            self.frames,
            self.copied_into_source,
            count_for(PresentCopyKind::CopyTextureRegion),
            count_for(PresentCopyKind::ResolveSubresource),
            count_for(PresentCopyKind::None),
            source_is_presented,
            probe_sampled_presented,
            self.probe_source_mismatch,
            self.fallback_clear,
            self.keys.len(),
            self.dropped_keys,
            self.dropped_sources,
            self.surfaces.len(),
            self.surface_targeted_by_pass,
            self.surface_targeted_by_copy,
            self.surface_registered_twice,
            self.surface_descriptor_mismatch,
            self.surface_pass_targets_not_surface,
            self.surface_pass_targets_other_surface,
            self.dropped_surfaces,
            self.surface_object_mismatch_reason,
            self.surface_pass_target_same_descriptor_other_object,
            self.surface_path_pass_target,
            self.surface_path_copy_dest,
            self.surface_path_alias_missing,
            self.surface_path_unrelated,
            self.composition_copies,
            self.composition_copy_source_is_pass_target,
            self.composition_copy_dest_is_surface,
            self.composition_present_without_copy,
            surfaces,
            pass_targets,
            texture_objects,
            allocation_objects,
        )
    }
}

/// Whether `sampled` is the same texture `presented` shows.
///
/// A probe that reads a different texture is a false negative; the caller must
/// name and count it rather than count its pixels as success.
pub fn probe_samples_presented(sampled: objc::Id, presented: objc::Id) -> bool {
    !sampled.is_null() && sampled == presented
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
    /// Set once when a supplied back-buffer could not be copied into the
    /// drawable, so the diagnostic fallback is named once instead of once per
    /// frame.
    fallback_reported: bool,
    /// Tracer progress observed at the previous present (`traced_work`,
    /// `traced_passes`). A traced pass target is this frame's only when the
    /// tracer advanced past these marks; otherwise it is carried evidence from
    /// an earlier pass and is ranked last.
    tracer_marks: (u64, u64),
    /// The presented surface object and its host copy count at the previous
    /// present. A new copy between two presents is a write for this frame.
    source_copies: (usize, u64),
    /// Set once when a presented surface had no write of its own and the frame
    /// was presented empty rather than sampling a foreign allocation.
    refusal_reported: bool,
}

/// Why a windowed present sampled the object it did.
///
/// The window samples the object a pass actually drew. When the presented
/// surface was written for this frame it is that object; when it was not, the
/// object a pass wrote is adopted only with evidence (a declared identity
/// target, a traced pass target, or the registry's own record of the
/// descriptor-identical surface). With no such evidence nothing is sampled:
/// an unrelated allocation is never substituted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresentResolution {
    /// The presented surface object itself was written this frame.
    NamedSurface,
    /// The one-shot identity target a caller declared for this present.
    DeclaredPassTarget,
    /// A pass target the tracer recorded for this present.
    FreshPassTarget,
    /// A pass target the tracer recorded earlier, used only because the
    /// presented surface has no write of its own.
    CarriedPassTarget,
    /// A foreign descriptor-identical pass target was offered but is not one of
    /// the host's registered surfaces (or is stale): nothing is sampled.
    Refused,
}

/// Whether a traced target's attachment has the presented surface's shape.
///
/// Shape, not usage: a texture registered first as storage and later widened
/// for an RTV keeps the usage it was created with.
fn target_matches_shape(target: &trace::TraceTarget, shape: &SurfaceDescriptor) -> bool {
    target.format == pixel_format_name(shape.format)
        && usize::try_from(target.width).ok() == Some(shape.width)
        && usize::try_from(target.height).ok() == Some(shape.height)
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
            fallback_reported: false,
            tracer_marks: (0, 0),
            source_copies: (0, 0),
            refusal_reported: false,
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

    /// Chooses the object this frame's present samples.
    ///
    /// The object a pass actually drew is the presented surface when a pass
    /// named it (or a host copy wrote it) for this frame. Otherwise the
    /// descriptor-identical object a pass wrote is adopted, in this order of
    /// evidence:
    ///
    /// 1. the declared one-shot identity target (exactly this frame);
    /// 2. a traced pass target the tracer recorded since the previous present;
    /// 3. a traced pass target recorded earlier (sticky tracer evidence);
    ///
    /// with the registry's shared-handle/view alias preferred over a plain
    /// descriptor match, then the higher registration ordinal. Registration
    /// order alone never decides the alias: an object the registry never
    /// created is never adopted, and a descriptor mismatch is never aliased.
    /// When a foreign descriptor-identical target exists and none is
    /// adoptable, nothing is sampled — an unrelated allocation is not shown.
    fn resolve_present_source(
        &mut self,
        source: Option<objc::Id>,
        explicit: Option<(usize, bool)>,
        observed: &[Option<trace::TraceTarget>; 2],
    ) -> (Option<objc::Id>, PresentResolution) {
        let Some(named) = source.filter(|texture| !texture.is_null()) else {
            return (None, PresentResolution::NamedSurface);
        };
        let Some(shape) = surface_descriptor_of(named) else {
            return (Some(named), PresentResolution::NamedSurface);
        };
        let named_key = named as usize;

        // Freshness: did the tracer describe work since the previous present?
        let report = trace::report();
        let marks = (report.traced_work, report.traced_passes);
        let fresh = marks != self.tracer_marks;
        self.tracer_marks = marks;

        // A copy into the named surface between the previous present and this
        // one is a host write for this frame, exactly as the trace counts it.
        let copies = blit::copy_count_for(named);
        let copied_since = self.source_copies.0 == named_key && copies > self.source_copies.1;
        self.source_copies = (named_key, copies);

        let named_written = copied_since
            || explicit.is_some_and(|(object, _)| object == named_key)
            || observed
                .iter()
                .flatten()
                .any(|target| target.texture == named_key);
        if named_written {
            return (Some(named), PresentResolution::NamedSurface);
        }

        // Best candidate key: lowest evidence rank, then a shared-handle/view
        // alias, then the newest registration.
        let mut best: Option<(u8, u8, u64, usize, PresentResolution)> = None;
        let mut foreign_same_shape = false;
        let mut consider = |object: usize,
                            swapchain: bool,
                            declared: bool,
                            recorded: Option<&trace::TraceTarget>| {
            if object == 0 || object == named_key {
                return;
            }
            let recorded_shape =
                recorded.is_some_and(|target| target_matches_shape(target, &shape));
            if !swapchain {
                // Not one of the host's registered surfaces. Its address is
                // never dereferenced; a recorded shape is enough to name the
                // split but never enough to alias it.
                foreign_same_shape |= recorded_shape;
                return;
            }
            // SAFETY: a target the host classified as a registered present
            // surface is retained by the transport for the process lifetime, and
            // a declared identity target is a live texture the caller just
            // encoded. Only an object with one of those provenances is
            // dereferenced.
            let Some(candidate_shape) = surface_descriptor_of(object as objc::Id) else {
                return;
            };
            if candidate_shape.format != shape.format
                || candidate_shape.width != shape.width
                || candidate_shape.height != shape.height
            {
                return;
            }
            foreign_same_shape = true;
            let rank = if declared {
                0
            } else if fresh {
                1
            } else {
                2
            };
            let shared = u8::from(!crate::macos::registry::texture_objects_share_handle(
                named_key, object,
            ));
            let order = crate::macos::registry::texture_object_registration_order(object)
                .map_or(0, |order| u64::MAX - order);
            let resolution = match rank {
                0 => PresentResolution::DeclaredPassTarget,
                1 => PresentResolution::FreshPassTarget,
                _ => PresentResolution::CarriedPassTarget,
            };
            let key = (rank, shared, order);
            if best.is_none_or(|(best_rank, best_shared, best_order, ..)| {
                key < (best_rank, best_shared, best_order)
            }) {
                best = Some((key.0, key.1, key.2, object, resolution));
            }
        };
        if let Some((object, swapchain)) = explicit {
            consider(object, swapchain, true, None);
        }
        for target in observed.iter().flatten() {
            consider(target.texture, target.swapchain, false, Some(target));
        }

        match best {
            Some((_, _, _, object, resolution)) => (Some(object as objc::Id), resolution),
            None if foreign_same_shape => {
                if !self.refusal_reported {
                    self.refusal_reported = true;
                    tracing::warn!(
                        surface_width = shape.width,
                        surface_height = shape.height,
                        "VICTORY_BACKEND=metal: the presented surface was not written this frame \
                         and the pass target is not one of the host's registered surfaces; \
                         presenting nothing instead of an unrelated allocation"
                    );
                }
                (None, PresentResolution::Refused)
            }
            None => (Some(named), PresentResolution::NamedSurface),
        }
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
    /// The blit is GPU-only and never scales or converts: a source whose
    /// extent or 4-channel format differs from the drawable first resizes the
    /// `CAMetalLayer` drawable to the source's extent and format (the layer
    /// scales the drawable into the view). This is what makes a 3840x2160
    /// title back-buffer visible in a 1280x720 window instead of the
    /// diagnostic clear. A source the layer cannot display still falls back to
    /// the clear, named once rather than silently.
    ///
    /// `source` is the surface the present named. The object that is actually
    /// copied into the drawable is the object a pass wrote this frame (see
    /// [`Self::resolve_present_source`]): when the named surface was not
    /// written and a descriptor-identical registered surface was, that surface
    /// is shown instead of showing an undrawn buffer. When nothing wrote a
    /// same-shape object, the drawable is cleared rather than sampling a
    /// foreign allocation.
    pub fn present(&mut self, source: Option<objc::Id>) -> Result<(), PresentError> {
        let mut drawable = self.current.take().ok_or_else(|| {
            PresentError::Metal("no CAMetalLayer drawable was acquired".to_string())
        })?;
        let _pool = AutoreleasePool::new();

        // The evidence that chooses the sampled object is taken before the
        // drawable is touched, so it describes this frame's pass and not a
        // drawable transition. `named` stays the surface the present named:
        // copy bookkeeping and the identity verdict are about it even when a
        // different object is shown.
        let explicit_pass_target = take_identity_pass_target();
        let observed = observed_pass_targets();
        let named = source.filter(|texture| !texture.is_null());
        let (sample, resolution) =
            self.resolve_present_source(source, explicit_pass_target, &observed);
        let trace_source = if resolution == PresentResolution::Refused {
            named
        } else {
            sample
        };

        // Read the sampled object's immutable properties once, before the
        // drawable can be released and re-acquired.
        // SAFETY: `sample` is a live `MTLTexture` (the named surface retained by
        // the caller, or a surface the transport retains); these accessors only
        // read immutable properties.
        let source_info: Option<(objc::Id, usize, usize, usize)> = sample.map(|sample| unsafe {
            (
                sample,
                objc::msg_send_usize_0(sample, objc::sel(c"pixelFormat")),
                objc::msg_send_usize_0(sample, objc::sel(c"width")),
                objc::msg_send_usize_0(sample, objc::sel(c"height")),
            )
        });

        // A CAMetalLayer cannot be reconfigured while it owns a drawable, and a
        // blit cannot scale or convert. Release the acquired drawable, take the
        // source's displayable format and extent, then acquire a new one.
        if let Some((_, format, source_width, source_height)) = source_info
            && drawable_format_supported(format)
            && source_width > 0
            && source_height > 0
            && source_width <= MAX_SURFACE_DIMENSION as usize
            && source_height <= MAX_SURFACE_DIMENSION as usize
        {
            // SAFETY: `drawable` is a live CAMetalDrawable and its texture only
            // has its immutable format read here.
            let target_format = unsafe {
                let texture = objc::msg_send_id_0(drawable.as_raw(), objc::sel(c"texture"));
                objc::msg_send_usize_0(texture, objc::sel(c"pixelFormat"))
            };
            if source_width != self.width as usize
                || source_height != self.height as usize
                || format != target_format
            {
                drop(drawable);
                // SAFETY: no drawable is held now; `layer` is the live
                // CAMetalLayer this swapchain owns and the values were
                // validated above.
                unsafe {
                    set_drawable_size(
                        self.layer.as_raw(),
                        source_width as u32,
                        source_height as u32,
                    );
                    msg_send_void_usize(self.layer.as_raw(), objc::sel(c"setPixelFormat:"), format);
                }
                self.width = source_width as u32;
                self.height = source_height as u32;
                self.next_drawable().ok_or_else(|| {
                    PresentError::Metal(
                        "CAMetalLayer.nextDrawable returned nil after a drawable resize"
                            .to_string(),
                    )
                })?;
                drawable = self.current.take().ok_or_else(|| {
                    PresentError::Metal("the resized drawable was not retained".to_string())
                })?;
            }
        }

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
        if let Some((source, source_format, source_width, source_height)) = source_info {
            // SAFETY: `target` is the live drawable texture; the accessor only
            // reads an immutable property.
            let target_format =
                unsafe { objc::msg_send_usize_0(target, objc::sel(c"pixelFormat")) };
            // The drawable was resized/retargeted above when the source was
            // displayable, so a 4-channel source of the same extent blits. A
            // matching format and extent is the only case a blit can express.
            if source_format == target_format
                && drawable_format_supported(source_format)
                && source_width == self.width as usize
                && source_height == self.height as usize
            {
                encode_drawable_blit(command_buffer, source, target, self.width, self.height)?;
                filled = true;
            }
        }
        if !filled {
            if let Some((_, format, width, height)) = source_info
                && !self.fallback_reported
            {
                self.fallback_reported = true;
                tracing::warn!(
                    source_format = format,
                    source_width = width,
                    source_height = height,
                    drawable_width = self.width,
                    drawable_height = self.height,
                    "VICTORY_BACKEND=metal: a presented back-buffer could not be copied \
                     into the drawable; presenting the diagnostic clear"
                );
            }
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
        }
        wait_for_completion_bounded(command_buffer, "drawable present")?;

        // The drawable now holds exactly the pixels the window server shows:
        // either the GPU blit of `source` or the diagnostic clear. Sampling it
        // after `waitUntilCompleted` is the only point where "presented" and
        // "readable" coincide. With the probe off, `probe` is `None` and this
        // block does nothing.
        let mut probe_sampled = None;
        if let Some(probe) = self.probe.as_mut() {
            // `filled == false` means the drawable was cleared to the known
            // diagnostic colour, so the probe can use that as its clear value.
            // Both the presented and the sampled texture are this same
            // drawable; handing the two explicitly makes a mismatch named and
            // counted instead of silently reported as pixels.
            if probe.observe(self.queue.as_raw(), target, target, self.presented, !filled) {
                probe_sampled = Some(target);
            }
        }

        record_present_trace_targets(
            trace_source.unwrap_or(std::ptr::null_mut()),
            target,
            probe_sampled,
            !filled,
            // The named surface is always the texture the host registered for
            // this present, whether or not it is the object sampled.
            named.is_some(),
            trace_source.and_then(surface_descriptor_of),
            explicit_pass_target,
            &observed,
        );

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

/// The opaque diagnostic colour the present path clears the drawable to when a
/// source cannot be blitted. The probe compares a uniform sample against this
/// so a real title clear can be told apart from the diagnostic one.
const DIAGNOSTIC_CLEAR: ClearColor = ClearColor {
    red: 0.05,
    green: 0.10,
    blue: 0.20,
    alpha: 1.0,
};

/// [`DIAGNOSTIC_CLEAR`] after Metal's UNORM8 conversion, in canonical RGBA
/// order: `round(0.05 * 255) = 13`, `round(0.10 * 255) = 26`,
/// `round(0.20 * 255) = 51`, alpha `255`. The BGRA8 byte order of the same
/// pixel is `[51, 26, 13, 255]`; [`probe_diagnostic_clear_rgba`] exposes this
/// and the unit test pins the UNORM8 derivation, so the clear constant and the
/// probe's name cannot drift.
const DIAGNOSTIC_CLEAR_RGBA: [u8; 4] = [13, 26, 51, 255];

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
                red: DIAGNOSTIC_CLEAR.red,
                green: DIAGNOSTIC_CLEAR.green,
                blue: DIAGNOSTIC_CLEAR.blue,
                alpha: DIAGNOSTIC_CLEAR.alpha,
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
// dropped. The probe never writes pixels to disk: counters, a CRC and a compact
// colour summary (min/max/mean RGBA, variance, distinct sample count and a
// `uniform_color` name) reach telemetry and the host log.
//
// What is sampled, precisely (after Wave 13A the present path blits the title
// back-buffer into the acquired `CAMetalDrawable`, so this is the drawable's
// texture, not the IOSurface and not the source):
//
// - texture: `CAMetalDrawable.texture`, the one just presented; its extent and
//   pixel format are resized to the source's when `present` retargets the layer
//   (`BGRA8Unorm` or `RGBA8Unorm`; `RGBA16Float` is displayable but not
//   sampled, and yields `probe_unsupported_pixel_format_115`).
// - region: 8x8 blocks on an 8x8 grid (the block's top-left pixel is the grid
//   point; the blocks are not centred). At 3840x2160 the step is
//   `3840 / 8 = 480` by `2160 / 8 = 270`; the first block starts at `(0, 0)`
//   and the last at `(7 * 480, 7 * 270) = (3360, 1890)`, so the last block
//   spans `x 3360..=3367`, `y 1890..=1897`, inside the texture.
// - byte offsets for the boot's real BGRA8 4K surface (4 bytes/pixel, row
//   stride `3840 * 4 = 15_360` bytes, total `3840 * 2160 * 4 = 33_177_600`):
//   first sample pixel `(0,0)` starts at byte `0`; last sample pixel
//   `(3360,1890)` starts at `(1890 * 3840 + 3360) * 4 = 29_043_840`; the last
//   byte actually read from the source is `((1897 * 3840) + 3367) * 4 + 3 =
//   29_151_391`, so the whole read is inside the surface. For the same extent
//   as `RGBA16Float` (8 bytes/pixel) the row stride doubles to `30_720`, the
//   first byte is still `0`, the last sample starts at `58_087_680`, and the
//   last byte read is `58_302_783` of `66_355_200` total. The analyser reads
//   8-bit samples only; a 16F drawable is named and refused rather than read
//   under the wrong stride, and these 16F numbers are pinned by a unit test so
//   a future 16F path inherits proven offsets.
// - row stride: the source copy is texture-to-texture, so no source row pitch
//   is passed to Metal; the stride the probe names is the source's implied
//   `width * bytes_per_pixel`. The readback stride is the 64-pixel-wide staging
//   atlas: `64 * 4 = 256` bytes per row, 64 rows, `16_384` bytes total.
// - format of the readback: the same 4-byte UNORM format as the drawable, read
//   as `[u8; 4]`; `PixelOrder::Bgra` reorders to canonical RGBA for the report.

/// Environment variable that opts into the presented-frame probe.
pub const FRAME_PROBE_ENV: &str = "VICTORY_METAL_FRAME_PROBE";
/// Environment variable that overrides the sampling interval.
pub const FRAME_PROBE_EVERY_ENV: &str = "VICTORY_METAL_FRAME_PROBE_EVERY";
/// Default sampling interval: one probe every 60 presented frames.
///
/// Cycle 13 measured about 1.39 ms per sampled frame at an interval of 30
/// (about 4.6% of a 30 FPS frame budget); 60 halves that overhead while still
/// sampling the first frame and every second after it. An explicit
/// `VICTORY_METAL_FRAME_PROBE_EVERY` overrides this, and `0` is still rejected.
pub const FRAME_PROBE_DEFAULT_EVERY: u64 = 60;

/// The `uniform_color` value before any sample names a colour.
///
/// [`FrameProbeReport::default`] starts here, and the host serialises it as
/// JSON `null` rather than pretending a colour was measured.
pub const FRAME_PROBE_NO_UNIFORM_COLOR: &str = "none";
/// A colour channel counts as content when it differs from the clear value by
/// more than this many 8-bit steps (out of 255). The comparison is strict: a
/// difference of exactly 8 is still considered clear. Alpha is not compared.
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
///
/// Statistics are expressed in canonical RGBA order (red, green, blue, alpha)
/// regardless of the surface's byte order, so a report reads the same for a
/// BGRA8 drawable and an RGBA8 one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameProbeSample {
    /// Pixels whose **colour** channels are farther than the threshold from the
    /// clear value. Alpha alone never makes a pixel content: a fully
    /// transparent (all-zero) surface is the diagnostic case this field must
    /// not confuse with a drawn frame.
    pub non_black_pixels: u64,
    /// Distinct colour classes seen, `0..=33`, derived in the same per-pixel
    /// pass as [`Self::non_black_pixels`].
    ///
    /// Class `0` is "clear": a pixel within `threshold` of the clear value on
    /// every colour channel. A content pixel contributes class
    /// `1 + (luma >> 3)`, where luma is the BT.601 luma in `0..=255`; adding
    /// one keeps a content class distinct from the clear class even when its
    /// 5-bit luma equals the clear's. Consequently `non_black_pixels == 0`
    /// implies `unique_luma_buckets <= 1`, and any sample that mixes clear and
    /// content pixels reports at least `2`. The two fields can never be
    /// computed from different samples or different channel sets.
    pub unique_luma_buckets: u32,
    /// CRC-64/ECMA-182 over the sampled bytes in row-major order.
    ///
    /// The documented value for an all-zero sample is `0` (init `0`, no final
    /// XOR, zero bytes leave the register at zero). A repeated `0x80` byte, a
    /// two-tone frame and an opaque black clear all checksum non-zero, and the
    /// test suite pins those cases. A 64-bit CRC is a linear map with a
    /// non-empty kernel over a 16 KiB message, so a pathological non-zero
    /// sample can in principle checksum to `0`; [`Self::nonzero_bytes`] removes
    /// that ambiguity (`nonzero_bytes == 0` is the only all-zero sample), and
    /// the min/max/mean and `uniform_color` fields name what was read.
    pub frame_crc: u64,
    /// Sampled bytes that were not `0x00`, `0..=probe_bytes`.
    ///
    /// This is the authoritative all-zero test: `nonzero_bytes == 0` means the
    /// read bytes were all zero regardless of the CRC, so `frame_crc=0` plus
    /// `nonzero_bytes=0` names a transparent/zero surface and never an
    /// uninitialised read or a checksum collision.
    pub nonzero_bytes: u64,
    /// Number of pixels in this measurement.
    pub probe_pixels: u64,
    /// Per-channel minimum, canonical RGBA.
    pub min_rgba: [u8; 4],
    /// Per-channel maximum, canonical RGBA.
    pub max_rgba: [u8; 4],
    /// Per-channel integer mean, canonical RGBA.
    pub mean_rgba: [u8; 4],
    /// Largest per-channel population variance, in 8-bit squared units
    /// (`0..=16256`; a uniform sample is `0`).
    pub variance: u32,
    /// Distinct exact pixel values in the sample, `1..=probe_pixels`.
    pub distinct_sample_values: u32,
    /// The exact pixel when every sampled pixel is identical, else `None`.
    pub uniform_rgba: Option<[u8; 4]>,
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
    /// Distinct colour classes in the last sample (clear class 0, plus one per
    /// content luma bucket), `0..=33`; see [`FrameProbeSample`].
    pub unique_luma_buckets: u32,
    /// CRC-64/ECMA-182 of the last sample.
    pub frame_crc: u64,
    /// Sample opportunities dropped because a probe was still in flight.
    pub dropped_samples: u64,
    /// True when the last sample came from the diagnostic fallback clear.
    pub fallback_clear: bool,
    /// True when the probe was asked to sample a texture other than the one
    /// `Present` showed. A set flag is a named failure, never a pixel success.
    pub probe_source_mismatch: bool,
    /// Sampling interval in presented frames.
    pub probe_every: u64,
    /// First named failure, if any. A set failure stops further sampling.
    pub failure: Option<String>,
    /// Per-channel minimum of the last sample, canonical RGBA.
    pub min_rgba: [u8; 4],
    /// Per-channel maximum of the last sample, canonical RGBA.
    pub max_rgba: [u8; 4],
    /// Per-channel mean of the last sample, canonical RGBA.
    pub mean_rgba: [u8; 4],
    /// Largest per-channel variance of the last sample, 8-bit squared units.
    pub variance: u32,
    /// Distinct exact pixel values in the last sample.
    pub distinct_sample_values: u32,
    /// Name of the last sample's colour: `black`, `transparent_black`,
    /// `swatch_0d1a33`, `mixed`, and so on.
    pub uniform_color: String,
    /// True when the last sample is a uniform colour equal to
    /// [`DIAGNOSTIC_CLEAR_RGBA`] within one 8-bit step, i.e. the frame is the
    /// diagnostic clear rather than a title clear that merely looks similar.
    pub matches_diagnostic_clear: bool,
    /// CPU microseconds spent reading back and analysing the last sample.
    pub sample_cost_us: u64,
    /// `MTLPixelFormat` of the sampled drawable texture (70, 80 or 115).
    pub pixel_format: u32,
    /// The Metal name of [`Self::pixel_format`] (`bgra8unorm`, ...).
    pub pixel_format_name: String,
    /// Width in pixels of the sampled texture (the source extent after the
    /// Wave 13A retarget, not the window size).
    pub sampled_width: u32,
    /// Height in pixels of the sampled texture.
    pub sampled_height: u32,
    /// Implied source row stride in bytes: `sampled_width * bytes_per_pixel`.
    pub source_row_bytes: u64,
    /// Total bytes of the sampled source: `source_row_bytes * sampled_height`.
    pub source_bytes: u64,
    /// X of the first sampled pixel (`0` for every layout).
    pub first_sample_x: u32,
    /// Y of the first sampled pixel (`0` for every layout).
    pub first_sample_y: u32,
    /// Byte offset of the first sampled pixel in the source.
    pub first_sample_byte: u64,
    /// X of the last sampled block's top-left pixel.
    pub last_sample_x: u32,
    /// Y of the last sampled block's top-left pixel.
    pub last_sample_y: u32,
    /// Byte offset of the last sampled block's top-left pixel in the source.
    pub last_sample_byte: u64,
    /// Last byte of the last sampled block, inclusive; the end of the source
    /// byte range the blits actually read.
    pub source_last_byte: u64,
    /// Row stride in bytes of the 64-pixel-wide staging atlas
    /// (`staging_width * bytes_per_pixel`).
    pub staging_row_bytes: u64,
    /// Bytes actually read back from the staging atlas (the whole 16 KiB).
    pub probe_bytes: u64,
    /// Sampled bytes that were not `0x00`.
    pub nonzero_bytes: u64,
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
            probe_source_mismatch: false,
            probe_every: FRAME_PROBE_DEFAULT_EVERY,
            failure: None,
            min_rgba: [0; 4],
            max_rgba: [0; 4],
            mean_rgba: [0; 4],
            variance: 0,
            distinct_sample_values: 0,
            uniform_color: FRAME_PROBE_NO_UNIFORM_COLOR.to_string(),
            matches_diagnostic_clear: false,
            sample_cost_us: 0,
            pixel_format: 0,
            pixel_format_name: "none".to_string(),
            sampled_width: 0,
            sampled_height: 0,
            source_row_bytes: 0,
            source_bytes: 0,
            first_sample_x: 0,
            first_sample_y: 0,
            first_sample_byte: 0,
            last_sample_x: 0,
            last_sample_y: 0,
            last_sample_byte: 0,
            source_last_byte: 0,
            staging_row_bytes: 0,
            probe_bytes: 0,
            nonzero_bytes: 0,
        }
    }
}

impl FrameProbeReport {
    /// One stable `frame_probe=...` line for the host log.
    ///
    /// The pre-existing field order and spelling are unchanged; the new
    /// fidelity fields are appended after `failure=`, so a reader that keys on
    /// a field name keeps working.
    pub fn log_line(&self) -> String {
        let frame = self
            .last_sampled_frame
            .map_or_else(|| "none".to_string(), |index| index.to_string());
        let failure = self.failure.as_deref().unwrap_or("none");
        format!(
            "frame_probe=frame={frame} non_black_pixels={} unique_luma_buckets={} \
             frame_crc={:016x} sampled_frames={} probe_pixels={} dropped_samples={} \
             fallback_clear={} probe_every={} failure={failure} \
             min_rgba={},{},{},{} max_rgba={},{},{},{} mean_rgba={},{},{},{} \
             variance={} distinct_sample_values={} uniform_color={} \
             matches_diagnostic_clear={} sample_cost_us={} \
             pixel_format={} pixel_format_name={} sampled_width={} sampled_height={} \
             source_row_bytes={} source_bytes={} \
             first_sample={},{} first_sample_byte={} last_sample={},{} last_sample_byte={} \
             source_last_byte={} staging_row_bytes={} probe_bytes={} nonzero_bytes={} \
             probe_source_mismatch={}",
            self.non_black_pixels,
            self.unique_luma_buckets,
            self.frame_crc,
            self.sampled_frames,
            self.probe_pixels,
            self.dropped_samples,
            u8::from(self.fallback_clear),
            self.probe_every,
            self.min_rgba[0],
            self.min_rgba[1],
            self.min_rgba[2],
            self.min_rgba[3],
            self.max_rgba[0],
            self.max_rgba[1],
            self.max_rgba[2],
            self.max_rgba[3],
            self.mean_rgba[0],
            self.mean_rgba[1],
            self.mean_rgba[2],
            self.mean_rgba[3],
            self.variance,
            self.distinct_sample_values,
            self.uniform_color,
            u8::from(self.matches_diagnostic_clear),
            self.sample_cost_us,
            self.pixel_format,
            self.pixel_format_name,
            self.sampled_width,
            self.sampled_height,
            self.source_row_bytes,
            self.source_bytes,
            self.first_sample_x,
            self.first_sample_y,
            self.first_sample_byte,
            self.last_sample_x,
            self.last_sample_y,
            self.last_sample_byte,
            self.source_last_byte,
            self.staging_row_bytes,
            self.probe_bytes,
            self.nonzero_bytes,
            u8::from(self.probe_source_mismatch),
        )
    }
}

/// The proven geometry of one sample: what was read, from where, with what
/// stride. Computed from the same [`ProbeLayout`] the GPU blit uses, so the
/// reported numbers cannot drift from the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ProbeGeometry {
    pixel_format: u32,
    width: u32,
    height: u32,
    source_row_bytes: u64,
    source_bytes: u64,
    first_sample_x: u32,
    first_sample_y: u32,
    first_sample_byte: u64,
    last_sample_x: u32,
    last_sample_y: u32,
    last_sample_byte: u64,
    staging_row_bytes: u64,
    probe_bytes: u64,
}

impl ProbeGeometry {
    /// Derives the byte geometry of `layout` for `format`, or `None` when the
    /// format has no known pixel size.
    fn new(layout: ProbeLayout, format: usize) -> Option<Self> {
        let bytes_per_pixel = u64::from(pixel_format_bytes(format)?);
        let width = layout.source_width;
        let height = layout.source_height;
        let source_row_bytes = u64::from(width).checked_mul(bytes_per_pixel)?;
        let source_bytes = source_row_bytes.checked_mul(u64::from(height))?;
        let first = layout.block_origin(0, 0, width, height);
        let last = layout.block_origin(FRAME_PROBE_GRID - 1, FRAME_PROBE_GRID - 1, width, height);
        let pixel_byte = |origin: MTLOrigin| {
            (origin.y as u64)
                .checked_mul(u64::from(width))?
                .checked_add(origin.x as u64)?
                .checked_mul(bytes_per_pixel)
        };
        let staging_row_bytes = u64::from(layout.staging_width).checked_mul(bytes_per_pixel)?;
        let probe_bytes = staging_row_bytes.checked_mul(u64::from(layout.staging_height))?;
        Some(Self {
            pixel_format: u32::try_from(format).ok()?,
            width,
            height,
            source_row_bytes,
            source_bytes,
            first_sample_x: u32::try_from(first.x).ok()?,
            first_sample_y: u32::try_from(first.y).ok()?,
            first_sample_byte: pixel_byte(first)?,
            last_sample_x: u32::try_from(last.x).ok()?,
            last_sample_y: u32::try_from(last.y).ok()?,
            last_sample_byte: pixel_byte(last)?,
            staging_row_bytes,
            probe_bytes,
        })
    }

    /// The byte offset of the last byte of the last sampled block (inclusive).
    fn last_sampled_byte_inclusive(layout: ProbeLayout, format: usize) -> Option<u64> {
        let geometry = Self::new(layout, format)?;
        let bytes_per_pixel = u64::from(pixel_format_bytes(format)?);
        geometry
            .last_sample_byte
            .checked_add(u64::from(layout.block_height - 1) * geometry.source_row_bytes)?
            .checked_add(u64::from(layout.block_width) * bytes_per_pixel)?
            .checked_sub(1)
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
    // Byte-at-a-time table: 4096 sampled pixels are 16 KiB, and the bitwise
    // form would spend most of a sample's cost on the checksum.
    const TABLE: [u64; 256] = {
        let mut table = [0u64; 256];
        let mut index = 0usize;
        while index < 256 {
            let mut value = (index as u64) << 56;
            let mut bit = 0;
            while bit < 8 {
                value = if value & (1 << 63) != 0 {
                    (value << 1) ^ POLYNOMIAL
                } else {
                    value << 1
                };
                bit += 1;
            }
            table[index] = value;
            index += 1;
        }
        table
    };
    for &byte in bytes {
        let slot = ((crc >> 56) as u8 ^ byte) as usize;
        crc = (crc << 8) ^ TABLE[slot];
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

/// The canonical `[r, g, b, a]` view of a stored pixel of either byte order.
fn canonical_rgba(pixel: [u8; 4], order: PixelOrder) -> [u8; 4] {
    match order {
        PixelOrder::Bgra => [pixel[2], pixel[1], pixel[0], pixel[3]],
        PixelOrder::Rgba => pixel,
    }
}

/// UNORM8 conversion of one normalised clear-colour component, the same
/// rounding Metal applies to `MTLClearColor`.
#[cfg(test)]
fn unorm8(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// [`DIAGNOSTIC_CLEAR`] as canonical RGBA bytes.
pub fn probe_diagnostic_clear_rgba() -> [u8; 4] {
    DIAGNOSTIC_CLEAR_RGBA
}

/// Distinct exact pixel values in `pixels`.
///
/// A uniform buffer is the common probe case (clear and diagnostic frames), so
/// it takes a single comparison pass instead of a sort.
fn distinct_sample_values(pixels: &[[u8; 4]]) -> u32 {
    let Some(first) = pixels.first() else {
        return 0;
    };
    if pixels.iter().all(|pixel| pixel == first) {
        return 1;
    }
    let mut values: Vec<u32> = pixels
        .iter()
        .map(|pixel| u32::from_le_bytes(*pixel))
        .collect();
    values.sort_unstable();
    values.dedup();
    u32::try_from(values.len()).unwrap_or(u32::MAX)
}

/// A human-readable name for the colour a probe sample represents.
///
/// A uniform sample names its exact colour (`black`, `white`, `swatch_0d1a33`,
/// ...). A sample with one luma bucket but more than one exact value is named
/// by its mean swatch, and anything with several luma buckets is `mixed`.
pub fn probe_color_name(sample: &FrameProbeSample) -> String {
    if sample.distinct_sample_values == 0 {
        return "no_samples".to_string();
    }
    if let Some(rgba) = sample.uniform_rgba {
        let [red, green, blue, alpha] = rgba;
        if alpha == 0 {
            return if red == 0 && green == 0 && blue == 0 {
                "transparent_black".to_string()
            } else {
                format!("transparent_{red:02x}{green:02x}{blue:02x}")
            };
        }
        return match (red, green, blue) {
            (0, 0, 0) => "black".to_string(),
            (255, 255, 255) => "white".to_string(),
            (255, 0, 0) => "red".to_string(),
            (0, 255, 0) => "green".to_string(),
            (0, 0, 255) => "blue".to_string(),
            _ if red == green && green == blue => {
                format!("grey_{red:02x}{green:02x}{blue:02x}")
            }
            _ => format!("swatch_{red:02x}{green:02x}{blue:02x}"),
        };
    }
    if sample.unique_luma_buckets <= 1 {
        let [red, green, blue, _] = sample.mean_rgba;
        return format!("swatch_{red:02x}{green:02x}{blue:02x}");
    }
    "mixed".to_string()
}

/// Measures a sampled buffer against `clear`.
///
/// A pixel is non-black when any of its **colour** channels differs from the
/// matching `clear` channel by more than `threshold`. `clear` is expressed in
/// the same byte order as `pixels`. Alpha is deliberately excluded: it is not
/// content, and comparing it was the Boot #18b bug in which an all-zero
/// (fully transparent) sample reported `non_black_pixels=4096` with
/// `frame_crc=0`.
///
/// The colour-class count is accumulated in the **same** loop and from the
/// **same** classification as `non_black_pixels`: a clear pixel contributes
/// class `0`, a content pixel contributes class `1 + luma_bucket`, so the two
/// fields can never disagree about which samples they describe.
pub fn analyze_pixels(
    pixels: &[[u8; 4]],
    order: PixelOrder,
    clear: [u8; 4],
    threshold: u8,
) -> FrameProbeSample {
    let mut non_black_pixels = 0u64;
    let mut classes = 0u64;
    let mut nonzero_bytes = 0u64;
    let mut min_rgba = [u8::MAX; 4];
    let mut max_rgba = [0u8; 4];
    let mut sums = [0u64; 4];
    let mut sum_squares = [0u64; 4];
    let clear_rgba = canonical_rgba(clear, order);
    for &pixel in pixels {
        let rgba = canonical_rgba(pixel, order);
        let is_content = rgba
            .iter()
            .take(3)
            .zip(clear_rgba.iter().take(3))
            .any(|(&value, &clear)| value.abs_diff(clear) > threshold);
        if is_content {
            non_black_pixels += 1;
            // `+ 1` keeps a content class distinct from the clear class even
            // when the content's luma bucket equals the clear's. The highest
            // bit is 32 (luma bucket 31 + 1), so a u64 holds every class.
            classes |= 1u64 << (luma_bucket(pixel, order) + 1);
        } else {
            classes |= 1;
        }
        nonzero_bytes += pixel.iter().filter(|&&byte| byte != 0).count() as u64;
        for (channel, &value) in rgba.iter().enumerate() {
            min_rgba[channel] = min_rgba[channel].min(value);
            max_rgba[channel] = max_rgba[channel].max(value);
            let value = u64::from(value);
            sums[channel] += value;
            sum_squares[channel] += value * value;
        }
    }
    let count = pixels.len() as u64;
    let mut mean_rgba = [0u8; 4];
    let mut variance = 0u32;
    if count > 0 {
        for channel in 0..4 {
            mean_rgba[channel] = (sums[channel] / count) as u8;
            // Population variance without floating point:
            // (n * Σx² − (Σx)²) / n².
            let n = u128::from(count);
            let numerator = n * u128::from(sum_squares[channel])
                - u128::from(sums[channel]) * u128::from(sums[channel]);
            let channel_variance = (numerator / (n * n)) as u32;
            variance = variance.max(channel_variance);
        }
    } else {
        min_rgba = [0; 4];
    }
    let distinct = distinct_sample_values(pixels);
    let uniform_rgba = if distinct == 1 {
        pixels.first().map(|pixel| canonical_rgba(*pixel, order))
    } else {
        None
    };
    FrameProbeSample {
        non_black_pixels,
        unique_luma_buckets: u32::try_from(classes.count_ones()).unwrap_or(u32::MAX),
        frame_crc: frame_crc64(pixels),
        nonzero_bytes,
        probe_pixels: count,
        min_rgba,
        max_rgba,
        mean_rgba,
        variance,
        distinct_sample_values: distinct,
        uniform_rgba,
    }
}

/// Convenience wrapper for the BGRA8 case.
pub fn analyze_bgra8(pixels: &[[u8; 4]], clear: [u8; 4], threshold: u8) -> FrameProbeSample {
    analyze_pixels(pixels, PixelOrder::Bgra, clear, threshold)
}

/// Layout of the fixed-size sampling grid for one drawable extent.
///
/// The source copy is texture-to-texture, so no source row pitch is involved;
/// only the block origins and the staging atlas geometry are computed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProbeLayout {
    source_width: u32,
    source_height: u32,
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
            source_width: width,
            source_height: height,
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

    /// Called once per successful present. Returns whether this frame was
    /// sampled. Never returns an error and never blocks for longer than
    /// [`FRAME_PROBE_WAIT`].
    ///
    /// `presented` is the texture the window server shows and `sampled` is the
    /// texture the probe would read. They must be the same object: a probe
    /// reading a different image is a false negative, so it is named and
    /// counted (`probe_source_mismatch`) and produces no pixel sample.
    fn observe(
        &mut self,
        queue: objc::Id,
        presented: objc::Id,
        sampled: objc::Id,
        frame_index: u64,
        fallback_clear: bool,
    ) -> bool {
        if !probe_samples_presented(sampled, presented) {
            self.report.probe_source_mismatch = true;
            self.fail("probe_source_mismatch");
            return false;
        }
        if self.abandoned {
            return false;
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
            return false;
        }
        if !due {
            return false;
        }
        match self.encode(queue, sampled, frame_index, fallback_clear) {
            Ok(()) => {
                self.wait_for(FRAME_PROBE_WAIT);
                true
            }
            Err(reason) => {
                self.fail(&reason);
                false
            }
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
            // RGBA16Float drawables are displayable (Wave 13A blits them) but
            // the probe has no 16F analysis path: fail once, named, instead of
            // sampling bytes under the wrong format.
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
        let sample_started = Instant::now();
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
        self.report.min_rgba = sample.min_rgba;
        self.report.max_rgba = sample.max_rgba;
        self.report.mean_rgba = sample.mean_rgba;
        self.report.variance = sample.variance;
        self.report.distinct_sample_values = sample.distinct_sample_values;
        self.report.uniform_color = probe_color_name(&sample);
        self.report.matches_diagnostic_clear = sample.uniform_rgba.is_some_and(|rgba| {
            rgba.iter()
                .zip(probe_diagnostic_clear_rgba())
                .all(|(value, diagnostic)| value.abs_diff(diagnostic) <= 1)
        });
        self.report.sample_cost_us =
            u64::try_from(sample_started.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.report.pixel_format = u32::try_from(staging.format).unwrap_or(u32::MAX);
        self.report.pixel_format_name = pixel_format_name(staging.format).to_string();
        if let Some(geometry) = ProbeGeometry::new(staging.layout, staging.format) {
            self.report.sampled_width = geometry.width;
            self.report.sampled_height = geometry.height;
            self.report.source_row_bytes = geometry.source_row_bytes;
            self.report.source_bytes = geometry.source_bytes;
            self.report.first_sample_x = geometry.first_sample_x;
            self.report.first_sample_y = geometry.first_sample_y;
            self.report.first_sample_byte = geometry.first_sample_byte;
            self.report.last_sample_x = geometry.last_sample_x;
            self.report.last_sample_y = geometry.last_sample_y;
            self.report.last_sample_byte = geometry.last_sample_byte;
            self.report.source_last_byte =
                ProbeGeometry::last_sampled_byte_inclusive(staging.layout, staging.format)
                    .unwrap_or(0);
            self.report.staging_row_bytes = geometry.staging_row_bytes;
            self.report.probe_bytes = geometry.probe_bytes;
        }
        self.report.nonzero_bytes = sample.nonzero_bytes;
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
    /// `MTLPixelFormat` of the wrapped texture (`BGRA8Unorm` or `RGBA8Unorm`).
    format: usize,
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
        Self::create_with_format(
            transport,
            width,
            height,
            MTL_PIXEL_FORMAT_BGRA8_UNORM,
            IOSURFACE_PIXEL_FORMAT_BGRA,
        )
    }

    /// Creates an `rgba8unorm` IOSurface-backed render target.
    ///
    /// The title's AAA offscreen target is `RGBA8Unorm`; a same-format present
    /// surface lets the offscreen->swapchain copy stay a GPU blit instead of a
    /// swizzle or an implicit conversion. The surface uses the `'RGBA'`
    /// four-character code so the IOSurface and the `MTLTexture` agree.
    pub fn create_rgba8(
        transport: &DirectMetalTransport,
        width: u32,
        height: u32,
    ) -> Result<Self, PresentError> {
        Self::create_with_format(
            transport,
            width,
            height,
            MTL_PIXEL_FORMAT_RGBA8_UNORM,
            IOSURFACE_PIXEL_FORMAT_RGBA,
        )
    }

    /// Shared constructor for a 4-byte-per-pixel IOSurface and its texture.
    fn create_with_format(
        transport: &DirectMetalTransport,
        width: u32,
        height: u32,
        format: usize,
        fourcc: i32,
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
        let surface = create_iosurface(width, height, bytes_per_row, fourcc)?;

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
            format,
        )?;

        Ok(Self {
            surface,
            texture,
            device: transport.device_raw(),
            surface_id,
            width,
            height,
            bytes_per_row: actual_row,
            format,
            frame_index: None,
        })
    }

    /// `MTLPixelFormat` of the wrapped texture (70 or 80).
    pub fn pixel_format(&self) -> usize {
        self.format
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
        let rc =
            unsafe { IOSurfaceUnlock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceUnlock failed");
        pixels
    }

    /// Test-only: locks the IOSurface for writing and stores `bytes` verbatim.
    ///
    /// This lets a hermetic GPU test place a known pixel at a known source
    /// coordinate, then prove the probe's blit reads exactly that byte. The
    /// buffer must be exactly `bytes_per_row * height`; production never calls
    /// this (it is `cfg(test)` and no production code can write surface bytes).
    #[cfg(test)]
    fn write_bgra_rows(&self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            self.bytes_per_row * self.height as usize,
            "write buffer must match the IOSurface layout"
        );
        let surface = self.surface.as_ptr();
        // SAFETY: `surface` is a live IOSurface; option 0 is the read/write
        // lock, and no GPU work is in flight on this image in the test.
        let rc = unsafe { IOSurfaceLock(surface, 0, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceLock(read-write) failed");
        // SAFETY: The surface is locked for writing and the base address is
        // valid for `bytes_per_row * height` writable bytes.
        let base = unsafe { IOSurfaceGetBaseAddress(surface) }.cast::<u8>();
        assert!(!base.is_null(), "locked IOSurface has no base address");
        // SAFETY: `base` is the surface's writable backing store and `bytes` is
        // a distinct, exactly sized allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
        }
        // SAFETY: Unlocks the same live surface with the same option.
        let rc = unsafe { IOSurfaceUnlock(surface, 0, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceUnlock(read-write) failed");
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
        // The headless path presents the drawn image itself, so the source and
        // the presented texture are the same object by construction. The host
        // copy trace names whatever wrote it last (for the title path this is
        // the offscreen->swapchain copy/resolve).
        let texture = image.texture_raw();
        record_present_trace_targets(
            texture,
            texture,
            None,
            false,
            true,
            surface_descriptor_of(texture),
            take_identity_pass_target(),
            &observed_pass_targets(),
        );
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
fn create_iosurface(
    width: u32,
    height: u32,
    bytes_per_row: usize,
    pixel_format: i32,
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
        ("IOSurfacePixelFormat", PropertyValue::Int(pixel_format)),
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
    format: usize,
) -> Result<ObjcObject, PresentError> {
    let class = objc::class(c"MTLTextureDescriptor");
    if class.is_null() {
        return Err(PresentError::TextureCreationFailed);
    }
    // SAFETY: `class` is `MTLTextureDescriptor`, the selector is the documented
    // 2D factory (one mip level) and the receiver is the class object.
    let descriptor = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            class,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            format,
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
        type Fn =
            unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, *mut c_void, usize) -> objc::Id;
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
    let descriptor = unsafe { objc::msg_send_id_0(class, objc::sel(c"renderPassDescriptor")) };
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
    let attachment =
        unsafe { msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0) };
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
    }
    wait_for_completion_bounded(command_buffer, "present command buffer")
}

/// How long the present path may wait for one command buffer.
const PRESENT_COMPLETION_TIMEOUT: Duration = Duration::from_secs(30);
/// Poll interval of a bounded present wait: a sleep, never a busy spin.
const PRESENT_COMPLETION_POLL: Duration = Duration::from_micros(250);

/// Waits for one present command buffer without an unbounded
/// `waitUntilCompleted`.
///
/// Polls the immutable status with a sleep and names a timeout, so a wedged
/// queue returns [`PresentError::Metal`] with the label and elapsed bound
/// instead of hanging the caller. Called from the frame path only with a
/// command buffer that was just committed.
fn wait_for_completion_bounded(
    command_buffer: objc::Id,
    what: &'static str,
) -> Result<(), PresentError> {
    let started = Instant::now();
    loop {
        // SAFETY: `command_buffer` is a live committed `MTLCommandBuffer`; the
        // accessors only read its status and error.
        let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
        if status == MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            return Ok(());
        }
        if status == 5 {
            // SAFETY: `[command_buffer error]` is null or an autoreleased
            // `NSError *` inside the caller's pool.
            let error = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
            // SAFETY: `error` is null or a live `NSError *`.
            let message = unsafe { objc::nserror_message(error) };
            return Err(PresentError::Metal(format!(
                "{what} failed (status={status}): {message}"
            )));
        }
        if started.elapsed() >= PRESENT_COMPLETION_TIMEOUT {
            return Err(PresentError::Metal(format!(
                "{what} timed out after {} ms waiting for the GPU (status={status})",
                PRESENT_COMPLETION_TIMEOUT.as_millis()
            )));
        }
        std::thread::sleep(PRESENT_COMPLETION_POLL);
    }
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
    }
    wait_for_completion_bounded(command_buffer, "present fence command buffer")
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
    use std::time::{Duration, Instant};

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

    fn assert_solid(image: &SurfaceImage, clear: ClearColor, context: &str) -> Vec<u8> {
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
        let mut images: Vec<SurfaceImage> =
            (0..3).map(|_| create_surface(&transport, 64, 64)).collect();
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
        assert_eq!(
            surface_ids.len(),
            3,
            "each frame must use a distinct surface"
        );
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
        assert_eq!(
            sink.frames().len(),
            1,
            "sink must not see the refused frame"
        );

        assert_eq!(
            path.finish_present()
                .expect("finish in-flight frame")
                .frame_index,
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
            matches!(error, PresentError::SinkRefused { frame_index: 0, .. }),
            "unexpected refusal error: {error:?}"
        );
        assert!(
            sink.frames().is_empty(),
            "refused frame must not be recorded"
        );
        assert_eq!(path.in_flight(), None, "refusal must clear in-flight state");
        assert_eq!(path.finish_present(), Err(PresentError::NoFrameInFlight));
        assert_solid(&image, ClearColor::OPAQUE_RED, "refused frame render");

        path.present(&mut image, ClearColor::OPAQUE_GREEN, &mut sink)
            .expect("present after refusal should succeed");
        assert_eq!(sink.frames().len(), 1);
        assert_eq!(
            sink.frames()[0].frame_index,
            1,
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
        // ADR-049: the default sampling interval is 60 presented frames.
        assert_eq!(FRAME_PROBE_DEFAULT_EVERY, 60);
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
        let sampled: Vec<u64> = (0..100)
            .filter(|frame| probe_due(*frame, FRAME_PROBE_DEFAULT_EVERY))
            .collect();
        assert_eq!(
            sampled,
            vec![0, 60],
            "the default cadence samples frame 0 and every 60th frame after it"
        );
        assert_eq!(
            sampled.last().copied(),
            Some(60),
            "the report carries the last sampled frame index"
        );
        let every_thirty: Vec<u64> = (0..100).filter(|frame| probe_due(*frame, 30)).collect();
        assert_eq!(every_thirty, vec![0, 30, 60, 90]);
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
        assert_eq!(
            sample.non_black_pixels, 0,
            "an all-clear frame has no content"
        );
        assert_eq!(sample.unique_luma_buckets, 1);
        assert_eq!(sample.probe_pixels, 64);

        // A difference of exactly the threshold is still clear.
        let mut at_threshold = all_clear.clone();
        at_threshold[3][0] = clear[0] + FRAME_PROBE_CHANNEL_THRESHOLD;
        assert_eq!(
            analyze_bgra8(&at_threshold, clear, FRAME_PROBE_CHANNEL_THRESHOLD).non_black_pixels,
            0
        );

        // One step beyond the threshold on any colour channel counts exactly one
        // pixel. Alpha is not content: Boot #18b's all-zero sample must not read
        // as 4096 non-black pixels just because alpha 0 differs from 255.
        for channel in 0..3 {
            let mut one = all_clear.clone();
            one[7][channel] = clear[channel] + FRAME_PROBE_CHANNEL_THRESHOLD + 1;
            let sample = analyze_bgra8(&one, clear, FRAME_PROBE_CHANNEL_THRESHOLD);
            assert_eq!(sample.non_black_pixels, 1, "channel {channel}");
        }
        let mut alpha_only = all_clear.clone();
        alpha_only[7][3] = 0;
        assert_eq!(
            analyze_bgra8(&alpha_only, clear, FRAME_PROBE_CHANNEL_THRESHOLD).non_black_pixels,
            0,
            "a transparent pixel is clear, not content"
        );
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

        // Values 0..=7 share the same 5-bit luma bucket. Measure them as
        // content (against a mid-grey clear so neither is the clear class) to
        // pin the class mapping itself.
        let same_bucket = vec![[0, 0, 0, 255], [7, 7, 7, 255]];
        assert_eq!(luma_bucket(same_bucket[0], PixelOrder::Bgra), 0);
        assert_eq!(luma_bucket(same_bucket[1], PixelOrder::Bgra), 0);
        assert_eq!(
            analyze_bgra8(&same_bucket, [128, 128, 128, 255], 0).unique_luma_buckets,
            1
        );
        let next_bucket = vec![[7, 7, 7, 255], [8, 8, 8, 255]];
        assert_eq!(
            analyze_bgra8(&next_bucket, [128, 128, 128, 255], 0).unique_luma_buckets,
            2
        );
    }

    #[test]
    fn probe_clear_and_content_buckets_share_one_classification() {
        // A uniform clear of any colour is exactly one class even when its
        // luma sits in the middle of the range (the case where "distinct luma
        // buckets" alone would be meaningless as a content test).
        for clear in [[0u8, 0, 0, 255], [128, 128, 128, 255], [51, 26, 13, 255]] {
            let sample = analyze_bgra8(&vec![clear; 64], clear, FRAME_PROBE_CHANNEL_THRESHOLD);
            assert_eq!(sample.non_black_pixels, 0);
            assert_eq!(sample.unique_luma_buckets, 1);
            assert_eq!(sample.distinct_sample_values, 1);
        }

        // One content pixel that stays inside the clear's own luma bucket still
        // adds a class: the two fields describe the same sample.
        let mut mixed = vec![[10u8, 20, 30, 255]; 64];
        mixed[0] = [10, 20, 40, 255]; // +10 blue > threshold 8
        let sample = analyze_bgra8(&mixed, [10, 20, 30, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.non_black_pixels, 1);
        assert_eq!(
            sample.unique_luma_buckets, 2,
            "a clear class plus any content class is at least two"
        );
    }

    #[test]
    fn probe_crc_is_stable_sensitive_and_matches_the_ecma182_check_value() {
        // CRC-64/ECMA-182 standard check value for the ASCII bytes "123456789".
        assert_eq!(crc64_ecma_update(0, b"123456789"), 0x6C40_DF5F_0B49_7347);
        assert_eq!(frame_crc64(&[]), 0);

        // The documented zero value: the all-zero sample (init 0, no final
        // XOR, zero bytes leave the register at zero). This is exactly Boot
        // #18b's surface, and it is why `frame_crc=0` alone could not be told
        // apart from an uninitialised sample. The new mean/min/max/alpha and
        // `uniform_color` fields name it `transparent_black`.
        let all_zero = vec![[0u8; 4]; FRAME_PROBE_PIXELS as usize];
        assert_eq!(frame_crc64(&all_zero), 0, "all-zero CRC is 0 by definition");

        // A known non-zero pattern: one 0x80 byte repeated.
        let mid_grey = vec![[0x80u8; 4]; 64];
        assert_ne!(
            frame_crc64(&mid_grey),
            0,
            "0x80 bytes must not checksum to 0"
        );
        let sample = analyze_bgra8(&mid_grey, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert!(sample.non_black_pixels > 0, "mid-grey is content");
        assert_ne!(sample.frame_crc, 0, "non-black content must not CRC to 0");

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
            min_rgba: [1, 2, 3, 4],
            max_rgba: [250, 251, 252, 253],
            mean_rgba: [13, 26, 51, 255],
            variance: 900,
            distinct_sample_values: 7,
            uniform_color: "swatch_0d1a33".to_string(),
            matches_diagnostic_clear: true,
            sample_cost_us: 42,
            pixel_format: 80,
            pixel_format_name: "bgra8unorm".to_string(),
            sampled_width: 3840,
            sampled_height: 2160,
            source_row_bytes: 15_360,
            source_bytes: 33_177_600,
            first_sample_x: 0,
            first_sample_y: 0,
            first_sample_byte: 0,
            last_sample_x: 3360,
            last_sample_y: 1890,
            last_sample_byte: 29_043_840,
            source_last_byte: 29_151_391,
            staging_row_bytes: 256,
            probe_bytes: 16_384,
            nonzero_bytes: 16_384,
            probe_source_mismatch: false,
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
            "min_rgba=1,2,3,4",
            "max_rgba=250,251,252,253",
            "mean_rgba=13,26,51,255",
            "variance=900",
            "distinct_sample_values=7",
            "uniform_color=swatch_0d1a33",
            "matches_diagnostic_clear=1",
            "sample_cost_us=42",
            "pixel_format=80",
            "pixel_format_name=bgra8unorm",
            "sampled_width=3840",
            "sampled_height=2160",
            "source_row_bytes=15360",
            "source_bytes=33177600",
            "first_sample=0,0",
            "first_sample_byte=0",
            "last_sample=3360,1890",
            "last_sample_byte=29043840",
            "source_last_byte=29151391",
            "staging_row_bytes=256",
            "probe_bytes=16384",
            "nonzero_bytes=16384",
        ] {
            assert!(line.contains(expected), "{line} must contain {expected}");
        }
        assert!(line.contains("frame_crc=0123456789abcdef"));
    }

    #[test]
    fn probe_all_zero_sample_is_transparent_black_not_content() {
        // Boot #18b's exact evidence: 4096 pixels, all zero bytes. The old
        // analyser counted them all as non-black because alpha 0 differs from
        // the opaque-black reference's 255, while the CRC was legitimately 0.
        let all_zero = vec![[0u8; 4]; FRAME_PROBE_PIXELS as usize];
        let sample = analyze_bgra8(&all_zero, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(
            sample.non_black_pixels, 0,
            "a transparent sample is not content"
        );
        assert_eq!(
            sample.frame_crc, 0,
            "all-zero bytes checksum to the documented 0"
        );
        assert_eq!(sample.unique_luma_buckets, 1);
        assert_eq!(sample.probe_pixels, FRAME_PROBE_PIXELS);
        assert_eq!(sample.distinct_sample_values, 1);
        assert_eq!(sample.uniform_rgba, Some([0, 0, 0, 0]));
        assert_eq!(
            probe_color_name(&sample),
            "transparent_black",
            "the name must separate a zero sample from an opaque black clear"
        );
        assert_eq!(sample.mean_rgba, [0, 0, 0, 0]);
        assert_eq!(sample.max_rgba, [0, 0, 0, 0]);
        assert_eq!(sample.variance, 0);
        assert_eq!(
            sample.nonzero_bytes, 0,
            "all-zero bytes are the authoritative zero case"
        );

        // Opaque black is a different surface: same luma bucket, same RGB
        // minimum, but a name and alpha that tell them apart, and a non-zero
        // CRC because the alpha bytes are non-zero.
        let opaque_black = vec![[0u8, 0, 0, 255]; FRAME_PROBE_PIXELS as usize];
        let sample = analyze_bgra8(&opaque_black, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.non_black_pixels, 0);
        assert_ne!(sample.frame_crc, 0, "opaque black has non-zero alpha bytes");
        assert_eq!(
            sample.nonzero_bytes, FRAME_PROBE_PIXELS,
            "opaque black has one non-zero alpha byte per pixel"
        );
        assert_eq!(probe_color_name(&sample), "black");
        assert_eq!(sample.mean_rgba, [0, 0, 0, 255]);
    }

    #[test]
    fn probe_one_nonzero_pixel_is_counted_and_changes_the_crc() {
        let all_zero = vec![[0u8; 4]; FRAME_PROBE_PIXELS as usize];
        let clean = analyze_bgra8(&all_zero, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(clean.non_black_pixels, 0);
        assert_eq!(clean.frame_crc, 0);
        assert_eq!(clean.nonzero_bytes, 0);

        // One pixel with a colour channel beyond the threshold: counted, and
        // the CRC of the bytes read changes.
        let mut one = all_zero.clone();
        one[17] = [0, 0, 255, 0];
        let sample = analyze_bgra8(&one, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.non_black_pixels, 1);
        assert_eq!(sample.nonzero_bytes, 1);
        assert_ne!(
            sample.frame_crc, 0,
            "one non-zero pixel must change the CRC"
        );
        assert_ne!(
            sample.frame_crc, clean.frame_crc,
            "the CRC is sensitive to the one changed pixel"
        );
        assert_eq!(sample.distinct_sample_values, 2);
        assert_eq!(sample.min_rgba, [0, 0, 0, 0]);
        assert_eq!(sample.max_rgba, [255, 0, 0, 0]);
        assert!(sample.variance > 0, "one 255 channel is measurable spread");
    }

    #[test]
    fn probe_two_halves_have_two_buckets_and_a_nonzero_crc() {
        // The 64x64 staging atlas: black left half, white right half.
        let mut staging = vec![[0u8, 0, 0, 255]; 64 * 64];
        for row in staging.chunks_exact_mut(64) {
            for pixel in &mut row[32..] {
                *pixel = [255, 255, 255, 255];
            }
        }
        let sample = analyze_bgra8(&staging, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert!(
            sample.unique_luma_buckets >= 2,
            "black and white must occupy two luma buckets, got {}",
            sample.unique_luma_buckets
        );
        assert_ne!(sample.frame_crc, 0);
        assert_eq!(sample.distinct_sample_values, 2);
        assert_eq!(sample.min_rgba, [0, 0, 0, 255]);
        assert_eq!(sample.max_rgba, [255, 255, 255, 255]);
        assert!(sample.variance > 0, "a two-tone frame has colour spread");
        assert_eq!(probe_color_name(&sample), "mixed");
        // Half the frame differs from black, all of the white half does.
        assert_eq!(sample.non_black_pixels, 32 * 64);
        // Black pixels contribute one non-zero byte (alpha); white pixels four.
        assert_eq!(
            sample.nonzero_bytes,
            32 * 64 * 1 + 32 * 64 * 4,
            "black alpha bytes plus white colour/alpha bytes"
        );
        assert!(
            sample.non_black_pixels > 0 && sample.unique_luma_buckets >= 2,
            "the content count and the class count describe the same sample"
        );
    }

    #[test]
    fn probe_uniform_mid_grey_is_distinguishable_from_black() {
        let grey = vec![[128u8, 128, 128, 255]; FRAME_PROBE_PIXELS as usize];
        let grey_sample = analyze_bgra8(&grey, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        let black = vec![[0u8, 0, 0, 255]; FRAME_PROBE_PIXELS as usize];
        let black_sample = analyze_bgra8(&black, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);

        assert_eq!(probe_color_name(&grey_sample), "grey_808080");
        assert_eq!(probe_color_name(&black_sample), "black");
        assert_eq!(grey_sample.mean_rgba, [128, 128, 128, 255]);
        assert_eq!(grey_sample.min_rgba, [128, 128, 128, 255]);
        assert_eq!(grey_sample.max_rgba, [128, 128, 128, 255]);
        assert_eq!(grey_sample.variance, 0, "a uniform field has no variance");
        assert_eq!(grey_sample.distinct_sample_values, 1);
        assert_ne!(grey_sample.mean_rgba, black_sample.mean_rgba);
        assert_ne!(grey_sample.frame_crc, black_sample.frame_crc);
        assert!(grey_sample.non_black_pixels > 0);
        assert_eq!(black_sample.non_black_pixels, 0);

        // Every colour field distinguishes the two uniform surfaces; only the
        // geometry fields (format, extent, stride) are shared.
        assert_ne!(grey_sample.min_rgba, black_sample.min_rgba);
        assert_ne!(grey_sample.max_rgba, black_sample.max_rgba);
        assert_ne!(
            grey_sample.nonzero_bytes, black_sample.nonzero_bytes,
            "mid-grey has three non-zero colour bytes per pixel, black one alpha byte"
        );
        assert_eq!(
            grey_sample.unique_luma_buckets,
            black_sample.unique_luma_buckets
        );
        assert_eq!(grey_sample.unique_luma_buckets, 1);
        assert_ne!(
            probe_color_name(&grey_sample),
            probe_color_name(&black_sample)
        );
    }

    #[test]
    fn probe_diagnostic_clear_converts_to_the_documented_rgba() {
        // The documented bytes are only correct if Metal's UNORM8 rounding of
        // the clear constant produces them; pin the derivation.
        assert_eq!(
            [
                unorm8(DIAGNOSTIC_CLEAR.red),
                unorm8(DIAGNOSTIC_CLEAR.green),
                unorm8(DIAGNOSTIC_CLEAR.blue),
                unorm8(DIAGNOSTIC_CLEAR.alpha),
            ],
            DIAGNOSTIC_CLEAR_RGBA
        );
        assert_eq!(probe_diagnostic_clear_rgba(), DIAGNOSTIC_CLEAR_RGBA);
        assert_eq!(probe_diagnostic_clear_rgba(), [13, 26, 51, 255]);
        // The BGRA8 byte order the drawable actually holds.
        assert_eq!(expected_bgra(DIAGNOSTIC_CLEAR), [51, 26, 13, 255]);

        let diagnostic = vec![[51u8, 26, 13, 255]; FRAME_PROBE_PIXELS as usize];
        let sample = analyze_bgra8(&diagnostic, diagnostic[0], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.non_black_pixels, 0);
        assert_eq!(probe_color_name(&sample), "swatch_0d1a33");
        assert_eq!(sample.uniform_rgba, Some([13, 26, 51, 255]));
        assert!(sample.uniform_rgba.is_some_and(|rgba| {
            rgba.iter()
                .zip(probe_diagnostic_clear_rgba())
                .all(|(value, diagnostic)| value.abs_diff(diagnostic) <= 1)
        }));
    }

    /// Copies the probe's 8x8 grid of 8x8 blocks from a full-size source
    /// buffer, byte for byte, exactly as the GPU blit + `getBytes` pair does.
    fn sample_like_the_probe(
        source: &[[u8; 4]],
        width: u32,
        height: u32,
        layout: ProbeLayout,
    ) -> Vec<[u8; 4]> {
        let mut staging = vec![[0u8; 4]; (layout.staging_width * layout.staging_height) as usize];
        for grid_y in 0..FRAME_PROBE_GRID {
            for grid_x in 0..FRAME_PROBE_GRID {
                let source_origin = layout.block_origin(grid_x, grid_y, width, height);
                let staging_origin = layout.staging_origin(grid_x, grid_y);
                for row in 0..layout.block_height as usize {
                    for column in 0..layout.block_width as usize {
                        let source_index =
                            (source_origin.y + row) * width as usize + source_origin.x + column;
                        let staging_index = (staging_origin.y + row)
                            * layout.staging_width as usize
                            + staging_origin.x
                            + column;
                        staging[staging_index] = source[source_index];
                    }
                }
            }
        }
        staging
    }

    #[test]
    fn probe_layout_for_a_4k_source_reads_real_sampled_bytes() {
        let (width, height) = (3840u32, 2160u32);
        let layout = ProbeLayout::new(width, height).expect("4K layout");
        // Two grid steps per 4K axis, each block 8x8, atlas 64x64.
        assert_eq!(layout.step_x, 480);
        assert_eq!(layout.step_y, 270);
        assert_eq!(layout.block_width, 8);
        assert_eq!(layout.block_height, 8);
        assert_eq!(layout.staging_width, 64);
        assert_eq!(layout.staging_height, 64);
        let last = layout.block_origin(7, 7, width, height);
        assert_eq!((last.x, last.y), (3360, 1890));
        assert!(last.x + 8 <= width as usize && last.y + 8 <= height as usize);

        // BGRA8 4K: 4 bytes per pixel, 15 360 bytes per row, 33 177 600 bytes.
        assert_eq!(width as usize * BYTES_PER_PIXEL as usize, 15_360);
        assert_eq!(
            width as usize * height as usize * BYTES_PER_PIXEL as usize,
            33_177_600
        );

        // A source whose sampled pixels are all distinct. Filling only the
        // sampled coordinates would prove the offsets, but a full buffer shows
        // the copy ignores the source row pitch instead of trusting it.
        let mut source = vec![[0u8; 4]; (width * height) as usize];
        for (y, row) in source.chunks_exact_mut(width as usize).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                *pixel = [
                    (x & 0xff) as u8,
                    (y & 0xff) as u8,
                    (((x >> 8) ^ (y >> 8)) & 0xff) as u8,
                    255,
                ];
            }
        }

        let staging = sample_like_the_probe(&source, width, height, layout);
        // Every staging pixel is the source pixel at the computed origin, so no
        // block reads zero padding from a wrong offset.
        for grid_y in 0..FRAME_PROBE_GRID {
            for grid_x in 0..FRAME_PROBE_GRID {
                let source_origin = layout.block_origin(grid_x, grid_y, width, height);
                let staging_origin = layout.staging_origin(grid_x, grid_y);
                for row in 0..8usize {
                    for column in 0..8usize {
                        let source_index =
                            (source_origin.y + row) * width as usize + source_origin.x + column;
                        let staging_index =
                            (staging_origin.y + row) * 64 + staging_origin.x + column;
                        assert_eq!(
                            staging[staging_index], source[source_index],
                            "grid ({grid_x},{grid_y}) pixel ({column},{row})"
                        );
                    }
                }
            }
        }

        let sample = analyze_bgra8(&staging, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.probe_pixels, FRAME_PROBE_PIXELS);
        assert_eq!(sample.distinct_sample_values, 4096);
        assert_eq!(sample.uniform_rgba, None, "a 4K gradient is not uniform");
        assert_ne!(sample.frame_crc, 0);
        assert_eq!(sample.min_rgba[1], 0);
        assert_eq!(sample.max_rgba[3], 255);
        assert_eq!(sample.mean_rgba[3], 255);
        assert_ne!(sample.min_rgba, sample.max_rgba);
        assert!(sample.variance > 0);
        assert!(sample.non_black_pixels > 0);
        assert!(sample.unique_luma_buckets >= 2);
    }

    /// The byte geometry the probe reports for the boot's real surface. These
    /// are the numbers in the module documentation and the `frame_probe=` line:
    /// if an offset here were wrong, the GPU blit would read a different block
    /// (or outside the texture).
    #[test]
    fn probe_geometry_pins_the_4k_bgra8_and_rgba16float_byte_offsets() {
        let (width, height) = (3840u32, 2160u32);
        let layout = ProbeLayout::new(width, height).expect("4K layout");

        let bgra8 =
            ProbeGeometry::new(layout, MTL_PIXEL_FORMAT_BGRA8_UNORM).expect("BGRA8 4K geometry");
        assert_eq!(bgra8.pixel_format, 80);
        assert_eq!((bgra8.width, bgra8.height), (3840, 2160));
        assert_eq!(bgra8.source_row_bytes, 15_360);
        assert_eq!(bgra8.source_bytes, 33_177_600);
        assert_eq!((bgra8.first_sample_x, bgra8.first_sample_y), (0, 0));
        assert_eq!(bgra8.first_sample_byte, 0);
        assert_eq!((bgra8.last_sample_x, bgra8.last_sample_y), (3360, 1890));
        assert_eq!(
            bgra8.last_sample_byte,
            (1890u64 * 3840 + 3360) * 4,
            "last sample starts at (1890 * 3840 + 3360) * 4"
        );
        assert_eq!(bgra8.last_sample_byte, 29_043_840);
        assert_eq!(bgra8.staging_row_bytes, 256);
        assert_eq!(bgra8.probe_bytes, 16_384);

        let last_inclusive =
            ProbeGeometry::last_sampled_byte_inclusive(layout, MTL_PIXEL_FORMAT_BGRA8_UNORM)
                .expect("last read byte");
        assert_eq!(
            last_inclusive,
            ((1890u64 + 7) * 3840 + 3360 + 7) * 4 + 3,
            "last byte of the last 8x8 block"
        );
        assert_eq!(last_inclusive, 29_151_391);
        assert!(
            last_inclusive < bgra8.source_bytes,
            "the read must stay inside the 4K BGRA8 surface"
        );

        // RGBA16Float has the same coordinates and twice the stride; the
        // analyser does not read it (it fails closed), but a future 16F path
        // inherits these proven offsets.
        let rgba16f = ProbeGeometry::new(layout, MTL_PIXEL_FORMAT_RGBA16_FLOAT)
            .expect("RGBA16Float 4K geometry");
        assert_eq!(rgba16f.pixel_format, 115);
        assert_eq!(rgba16f.source_row_bytes, 30_720);
        assert_eq!(rgba16f.source_bytes, 66_355_200);
        assert_eq!(rgba16f.first_sample_byte, 0);
        assert_eq!(rgba16f.last_sample_byte, 58_087_680);
        let last_16f_inclusive =
            ProbeGeometry::last_sampled_byte_inclusive(layout, MTL_PIXEL_FORMAT_RGBA16_FLOAT)
                .expect("last 16F read byte");
        assert_eq!(last_16f_inclusive, 58_302_783);
        assert!(last_16f_inclusive < rgba16f.source_bytes);
        assert_eq!(rgba16f.staging_row_bytes, 512);
        assert_eq!(rgba16f.probe_bytes, 32_768);
    }

    /// The boot's real geometry: a 3840x2160 source in a 1280x720 window. The
    /// probe must use the source extent, never the window's.
    #[test]
    fn probe_boot_geometry_samples_the_4k_source_not_the_720p_window() {
        let (source_w, source_h) = (3840u32, 2160u32);
        let (window_w, window_h) = (1280u32, 720u32);
        let source = ProbeLayout::new(source_w, source_h).expect("source layout");
        let window = ProbeLayout::new(window_w, window_h).expect("window layout");
        assert_eq!((source.step_x, source.step_y), (480, 270));
        assert_eq!((window.step_x, window.step_y), (160, 90));

        let geometry =
            ProbeGeometry::new(source, MTL_PIXEL_FORMAT_BGRA8_UNORM).expect("4K geometry");
        assert_eq!((geometry.width, geometry.height), (3840, 2160));
        assert_eq!(
            (geometry.last_sample_x, geometry.last_sample_y),
            (3360, 1890)
        );

        // The last sampled pixel's byte offset is far past the entire window
        // buffer (1280 * 720 * 4 = 3_686_400 bytes), which is the proof the
        // probe is reading the source, not the window.
        let window_bytes = u64::from(window_w) * u64::from(window_h) * 4;
        assert_eq!(window_bytes, 3_686_400);
        assert!(
            geometry.last_sample_byte > window_bytes,
            "the last sample ({}) must lie beyond the 1280x720 window buffer ({window_bytes})",
            geometry.last_sample_byte
        );
        assert!(geometry.last_sample_byte < geometry.source_bytes);

        // If the layout had been built from the window, the last sample would
        // be a different pixel inside the window buffer; the probe uses the
        // source layout's values, not these.
        let window_geometry =
            ProbeGeometry::new(window, MTL_PIXEL_FORMAT_BGRA8_UNORM).expect("window geometry");
        assert_eq!(
            (window_geometry.last_sample_x, window_geometry.last_sample_y),
            (1120, 630)
        );
        assert_ne!(geometry.last_sample_byte, window_geometry.last_sample_byte);
        assert_eq!(window_geometry.source_row_bytes, 5_120);
    }

    /// Measures the CPU side of one sample: `getBytes` of the 16 KiB atlas plus
    /// the full analysis of its 4096 pixels. The GPU blit is measured by
    /// `frame_probe_4k_sample_cost_is_small` on a real device.
    #[test]
    fn probe_analysis_of_4096_pixels_stays_cheap() {
        let uniform = vec![[13u8, 26, 51, 255]; FRAME_PROBE_PIXELS as usize];
        let mut mixed = vec![[0u8; 4]; FRAME_PROBE_PIXELS as usize];
        for (index, pixel) in mixed.iter_mut().enumerate() {
            *pixel = [
                (index & 0xff) as u8,
                ((index >> 4) & 0xff) as u8,
                ((index * 7) & 0xff) as u8,
                255,
            ];
        }
        let measure = |pixels: &[[u8; 4]], label: &str| -> u128 {
            for _ in 0..3 {
                std::hint::black_box(analyze_bgra8(
                    pixels,
                    [0, 0, 0, 255],
                    FRAME_PROBE_CHANNEL_THRESHOLD,
                ));
            }
            let mut runs = Vec::with_capacity(21);
            for _ in 0..21 {
                let started = Instant::now();
                std::hint::black_box(analyze_bgra8(
                    pixels,
                    [0, 0, 0, 255],
                    FRAME_PROBE_CHANNEL_THRESHOLD,
                ));
                runs.push(started.elapsed().as_nanos());
            }
            runs.sort_unstable();
            let median = runs[runs.len() / 2];
            eprintln!(
                "probe analysis of 4096 {label} pixels: median {:.1} us \
                 (min {:.1} us, max {:.1} us, n=21)",
                median as f64 / 1000.0,
                runs[0] as f64 / 1000.0,
                runs[runs.len() - 1] as f64 / 1000.0,
            );
            median
        };
        let uniform_ns = measure(&uniform, "uniform");
        let mixed_ns = measure(&mixed, "mixed");
        // A very generous bound: the point is that 4096 probes are not a
        // millisecond-scale cost, not to benchmark the test runner.
        assert!(
            uniform_ns < 5_000_000,
            "uniform analysis took {uniform_ns} ns"
        );
        assert!(mixed_ns < 5_000_000, "mixed analysis took {mixed_ns} ns");
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
            let encoder = objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder"));
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
            get_texture_bytes(staging.as_raw(), pixels.as_mut_ptr().cast(), 64 * 4, region);
        }
        let expected = expected_bgra(ClearColor::OPAQUE_RED);
        let mismatch = pixels.iter().position(|pixel| *pixel != expected);
        assert_eq!(mismatch, None, "staging copy differs at pixel {mismatch:?}");
    }

    #[test]
    #[ignore = "requires a Metal device: measures the 4K probe sample cost"]
    fn frame_probe_4k_sample_cost_is_small() {
        let (width, height) = (3840u32, 2160u32);
        let transport = open_transport();
        let image = create_surface(&transport, width, height);
        // Fill the real 4K surface with the diagnostic clear through the same
        // render pass the present path uses.
        encode_clear_pass(transport.queue_raw(), image.texture_raw(), DIAGNOSTIC_CLEAR)
            .expect("4K clear pass");
        let layout = ProbeLayout::new(width, height).expect("4K layout");
        let staging = create_probe_staging_texture(
            transport.device_raw(),
            MTL_PIXEL_FORMAT_BGRA8_UNORM,
            layout.staging_width,
            layout.staging_height,
        )
        .expect("probe staging texture");
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc::MTLSize {
                width: 64,
                height: 64,
                depth: 1,
            },
        };
        let mut pixels = vec![[0u8; 4]; 64 * 64];
        let mut runs = Vec::with_capacity(11);
        for _ in 0..11 {
            let started = Instant::now();
            // SAFETY: both textures are live 4K/64x64 BGRA8 textures, the
            // block rectangles are inside the 4K source and the 64x64 atlas,
            // and `pixels` holds 64 rows of 256 bytes for `getBytes`.
            unsafe {
                let command_buffer =
                    objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer"));
                let encoder = objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder"));
                for grid_y in 0..FRAME_PROBE_GRID {
                    for grid_x in 0..FRAME_PROBE_GRID {
                        encode_texture_block_copy(
                            encoder,
                            image.texture_raw(),
                            layout.block_origin(grid_x, grid_y, width, height),
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
                get_texture_bytes(staging.as_raw(), pixels.as_mut_ptr().cast(), 64 * 4, region);
            }
            runs.push(started.elapsed().as_nanos());
        }
        runs.sort_unstable();
        let median = runs[runs.len() / 2];
        eprintln!(
            "probe 4K sample (64 block blits + 16 KiB readback): median {:.1} us \
             (min {:.1} us, max {:.1} us, n=11)",
            median as f64 / 1000.0,
            runs[0] as f64 / 1000.0,
            runs[runs.len() - 1] as f64 / 1000.0,
        );

        let analysis_started = Instant::now();
        let sample = analyze_bgra8(&pixels, pixels[0], FRAME_PROBE_CHANNEL_THRESHOLD);
        eprintln!(
            "probe 4K analysis (4096 pixels): {:.1} us",
            analysis_started.elapsed().as_nanos() as f64 / 1000.0
        );
        assert_eq!(sample.probe_pixels, FRAME_PROBE_PIXELS);
        assert_eq!(sample.distinct_sample_values, 1);
        assert_eq!(probe_color_name(&sample), "swatch_0d1a33");
        assert_eq!(sample.uniform_rgba, Some([13, 26, 51, 255]));
        assert_eq!(sample.non_black_pixels, 0);
        assert_ne!(
            sample.frame_crc, 0,
            "the diagnostic clear is not all-zero bytes"
        );

        // The real blit's geometry is the same one the report names.
        let geometry =
            ProbeGeometry::new(layout, MTL_PIXEL_FORMAT_BGRA8_UNORM).expect("4K geometry");
        assert_eq!((geometry.width, geometry.height), (3840, 2160));
        assert_eq!(geometry.source_row_bytes, 15_360);
        assert_eq!(geometry.first_sample_byte, 0);
        assert_eq!(geometry.last_sample_byte, 29_043_840);
        assert_eq!(geometry.probe_bytes, 16_384);
        assert_eq!(sample.nonzero_bytes, FRAME_PROBE_PIXELS * 4);
    }

    /// Proves the real GPU staging blit at the boot's 3840x2160 extent reads
    /// the bytes at the computed source offsets: a source with a distinct value
    /// at every coordinate is written into a shared IOSurface, then the same
    /// 64-block copy the probe encodes is compared pixel for pixel. A wrong
    /// offset, a window-sized layout or an out-of-bounds block would show up
    /// as a mismatch and as fewer than 4096 distinct sample values.
    #[test]
    #[ignore = "requires a Metal device: proves the real 4K blit reads the computed offsets"]
    fn frame_probe_gpu_4k_blit_reads_the_computed_source_offsets() {
        let (width, height) = (3840u32, 2160u32);
        let transport = open_transport();
        let image = create_surface(&transport, width, height);
        let row_bytes = image.bytes_per_row();
        let mut source = vec![0u8; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * row_bytes + x * 4;
                source[offset] = (x & 0xff) as u8;
                source[offset + 1] = (y & 0xff) as u8;
                source[offset + 2] = (((x >> 8) ^ (y >> 8)) & 0xff) as u8;
                source[offset + 3] = 0xff;
            }
        }
        image.write_bgra_rows(&source);

        let layout = ProbeLayout::new(width, height).expect("4K layout");
        let staging = create_probe_staging_texture(
            transport.device_raw(),
            MTL_PIXEL_FORMAT_BGRA8_UNORM,
            layout.staging_width,
            layout.staging_height,
        )
        .expect("probe staging texture");

        // SAFETY: both textures are live 4K/64x64 BGRA8 textures, every block
        // rectangle lies inside the source and the 64x64 atlas, and the
        // readback buffer is the whole atlas at its 256-byte row pitch.
        unsafe {
            let command_buffer =
                objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer"));
            assert!(!command_buffer.is_null());
            let encoder = objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder"));
            assert!(!encoder.is_null());
            for grid_y in 0..FRAME_PROBE_GRID {
                for grid_x in 0..FRAME_PROBE_GRID {
                    encode_texture_block_copy(
                        encoder,
                        image.texture_raw(),
                        layout.block_origin(grid_x, grid_y, width, height),
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
        // SAFETY: the command buffer completed and `pixels` holds the atlas.
        unsafe {
            get_texture_bytes(staging.as_raw(), pixels.as_mut_ptr().cast(), 64 * 4, region);
        }

        let mut mismatches = 0usize;
        for grid_y in 0..FRAME_PROBE_GRID {
            for grid_x in 0..FRAME_PROBE_GRID {
                let origin = layout.block_origin(grid_x, grid_y, width, height);
                let staging_origin = layout.staging_origin(grid_x, grid_y);
                for row in 0..8usize {
                    for column in 0..8usize {
                        let source_offset = (origin.y + row) * row_bytes + (origin.x + column) * 4;
                        let staging_index =
                            (staging_origin.y + row) * 64 + staging_origin.x + column;
                        if pixels[staging_index] != source[source_offset..source_offset + 4] {
                            mismatches += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "the GPU copy must read the computed source offsets exactly"
        );

        let sample = analyze_bgra8(&pixels, [0, 0, 0, 255], FRAME_PROBE_CHANNEL_THRESHOLD);
        assert_eq!(sample.probe_pixels, FRAME_PROBE_PIXELS);
        assert_eq!(
            sample.distinct_sample_values, 4096,
            "a coordinate-keyed 4K source must produce 4096 distinct samples"
        );
        assert_ne!(sample.frame_crc, 0);
        // Every alpha byte is 0xff; some coordinate-derived colour bytes are
        // legitimately zero, so the exact count is below 4 per pixel.
        assert!(sample.nonzero_bytes > 0);
        assert!(sample.non_black_pixels > 0);
        assert!(sample.unique_luma_buckets >= 2);
    }

    fn token(value: usize) -> objc::Id {
        value as *mut c_void
    }

    #[test]
    fn probe_samples_presented_requires_the_same_texture() {
        let presented = token(0x1000);
        assert!(probe_samples_presented(presented, presented));
        assert!(!probe_samples_presented(token(0x2000), presented));
        assert!(!probe_samples_presented(std::ptr::null_mut(), presented));
    }

    #[test]
    fn present_trace_names_the_copy_kind_and_probe_identity() {
        reset_present_trace();
        blit::reset_copy_trace();
        let source = token(0x3000);
        let presented = token(0x4000);
        blit::note_copy_destination(blit::BlitCopyKind::CopyTextureRegion, source);
        record_present_trace(source, presented, Some(presented), false, None, None);
        let report = present_trace_report();
        assert_eq!(report.frames, 1);
        assert_eq!(report.copied_into_source, 1);
        assert_eq!(report.probe_source_mismatch, 0);
        let entry = report.keys.first().copied().expect("one key");
        assert_eq!(entry.copy_kind, PresentCopyKind::CopyTextureRegion);
        assert!(entry.probe_sampled_presented);
        assert!(!entry.source_is_presented);
        let line = report.log_line();
        assert!(line.contains("copy_kind_copy_texture_region=1"), "{line}");
        assert!(line.contains("probe_source_mismatch=0"), "{line}");

        record_present_trace(source, presented, Some(token(0x5000)), false, None, None);
        let report = present_trace_report();
        assert_eq!(report.frames, 2);
        assert_eq!(report.probe_source_mismatch, 1);
        assert!(
            report
                .keys
                .iter()
                .any(|entry| !entry.probe_sampled_presented),
            "a different sampled texture is a distinct key"
        );
    }

    #[test]
    fn present_trace_is_bounded_to_sixteen_keys() {
        reset_present_trace();
        blit::reset_copy_trace();
        let mut recorded = 0u64;
        for kind in 0..3usize {
            for source_is_presented in [false, true] {
                for probe_presented in [false, true] {
                    for fallback in [false, true] {
                        let index = recorded as usize;
                        let source = token(0x6000 + index);
                        let presented = if source_is_presented {
                            source
                        } else {
                            token(0x7000 + index)
                        };
                        match kind {
                            1 => blit::note_copy_destination(
                                blit::BlitCopyKind::CopyTextureRegion,
                                source,
                            ),
                            2 => blit::note_copy_destination(
                                blit::BlitCopyKind::ResolveSubresource,
                                source,
                            ),
                            _ => {}
                        }
                        let probe = probe_presented.then_some(presented);
                        record_present_trace(source, presented, probe, fallback, None, None);
                        recorded += 1;
                    }
                }
            }
        }
        assert_eq!(recorded, 24);
        let report = present_trace_report();
        assert_eq!(report.frames, 24);
        assert_eq!(report.keys.len(), PRESENT_TRACE_MAX_KEYS);
        assert_eq!(report.dropped_keys, 24 - PRESENT_TRACE_MAX_KEYS as u64);
    }

    #[test]
    fn a_probe_mismatch_is_a_named_failure_not_a_pixel_success() {
        // The guard the present path uses: a sampler handed a different
        // texture must not publish `non_black_pixels` as success.
        let presented = token(0x8000);
        let sampled = token(0x9000);
        assert!(!probe_samples_presented(sampled, presented));
        let report = FrameProbeReport {
            probe_source_mismatch: true,
            failure: Some("probe_source_mismatch".to_string()),
            ..FrameProbeReport::default()
        };
        assert_eq!(report.non_black_pixels, 0);
        assert!(report.log_line().contains("probe_source_mismatch=1"));
    }
}
