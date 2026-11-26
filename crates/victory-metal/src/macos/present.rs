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
use std::time::Instant;

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};

/// `MTLPixelFormatBGRA8Unorm`, matching the IOSurface `'BGRA'` format below.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
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

/// A presented frame, as seen by a [`PresentSink`].
///
/// Carries identity and pacing only: the surface's `IOSurfaceID`, the
/// monotonic frame index and the host timestamp. It deliberately carries no
/// pixel bytes, so a sink cannot accidentally introduce a readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentedFrame {
    /// `IOSurfaceGetID` of the surface that holds this frame.
    pub surface_id: u32,
    /// Monotonically increasing presentation index, starting at `0`.
    pub frame_index: u64,
    /// Nanoseconds since the [`PresentPath`] was created (monotonic clock).
    pub timestamp_ns: u64,
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
    fn zero_sized_surfaces_are_rejected_before_surface_creation() {
        let transport = open_transport();
        for (width, height) in [(0, 64), (64, 0), (0, 0)] {
            let error = SurfaceImage::create(&transport, width, height)
                .err()
                .expect("zero extent must be rejected");
            assert_eq!(error, PresentError::ZeroExtent { width, height });
        }
    }
}
