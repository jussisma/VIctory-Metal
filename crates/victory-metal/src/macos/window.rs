//! Opt-in native `NSWindow` + `CAMetalLayer` presentation (ADR-020).
//!
//! Headless `IOSurface` presentation stays the default so CI and automated
//! gates never need a window server. Setting `VICTORY_METAL_WINDOW=1` (or a
//! size spec such as `1280x720`) asks the host daemon to present
//! [`crate::macos::present::NativeSwapchain`] drawables on screen instead.
//!
//! # Environment
//!
//! | value | meaning |
//! |---|---|
//! | unset / `0` / `off` / `false` | headless (default) |
//! | `1` / `on` / `true` | open a window at the default size |
//! | `<width>x<height>` | open a window at that drawable size |
//! | `title=<text>` (after `,`) | set the window title |
//!
//! # Status
//!
//! This module is the ADR-020 seam. [`NativeWindow::open`] creates an
//! `NSApplication`/`NSWindow` and a `CAMetalLayer`, and
//! [`NativeWindow::present`] presents the acquired drawable. Nothing here is
//! required by the headless path, and opening a window never loads D3DMetal.

use std::ffi::c_void;
use std::mem;
use std::ptr::null_mut;

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};
use crate::macos::present::{NativeSwapchain, PresentError};

/// `NSApplicationActivationPolicyRegular`.
const NS_ACTIVATION_POLICY_REGULAR: usize = 0;
/// `NSWindowStyleMaskTitled`.
const NS_WINDOW_STYLE_TITLED: usize = 1;
/// `NSWindowStyleMaskClosable`.
const NS_WINDOW_STYLE_CLOSABLE: usize = 2;
/// `NSWindowStyleMaskMiniaturizable`.
const NS_WINDOW_STYLE_MINIATURIZABLE: usize = 4;
/// `NSWindowStyleMaskResizable`.
const NS_WINDOW_STYLE_RESIZABLE: usize = 8;
/// `NSBackingStoreBuffered`.
const NS_BACKING_STORE_BUFFERED: usize = 2;
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
/// `NSEventMaskAny`.
const NS_EVENT_MASK_ANY: usize = usize::MAX;

/// Environment variable that opts into on-screen presentation.
pub const WINDOW_ENV: &str = "VICTORY_METAL_WINDOW";

/// Default drawable size when the environment only says "on".
pub const DEFAULT_WINDOW_WIDTH: u32 = 1280;
/// Default drawable size when the environment only says "on".
pub const DEFAULT_WINDOW_HEIGHT: u32 = 720;

/// Requested window geometry and title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            title: "victory-metal".to_string(),
        }
    }
}

impl WindowConfig {
    /// Parses a `VICTORY_METAL_WINDOW` value.
    ///
    /// Returns `None` for the headless spellings, `Some(Err(..))` for a value
    /// that asks for a window but is malformed, and `Some(Ok(..))` otherwise.
    #[allow(clippy::result_unit_err)]
    pub fn parse(value: &str) -> Option<Result<Self, ()>> {
        let mut parts = value.split(',').map(str::trim);
        let head = parts.next().unwrap_or("").trim();
        let lower = head.to_ascii_lowercase();
        if head.is_empty() || matches!(lower.as_str(), "0" | "off" | "false" | "no") {
            return None;
        }
        let mut config = WindowConfig::default();
        if !matches!(lower.as_str(), "1" | "on" | "true" | "yes") {
            let Some((width, height)) = head.split_once(['x', 'X']) else {
                return Some(Err(()));
            };
            let Ok(width) = width.trim().parse::<u32>() else {
                return Some(Err(()));
            };
            let Ok(height) = height.trim().parse::<u32>() else {
                return Some(Err(()));
            };
            if width == 0 || height == 0 {
                return Some(Err(()));
            }
            config.width = width;
            config.height = height;
        }
        for part in parts {
            if let Some(title) = part.strip_prefix("title=") {
                if title.trim().is_empty() {
                    return Some(Err(()));
                }
                config.title = title.trim().to_string();
            } else if !part.is_empty() {
                return Some(Err(()));
            }
        }
        Some(Ok(config))
    }

    /// Reads the configuration from the process environment.
    pub fn from_env() -> Option<Result<Self, ()>> {
        std::env::var(WINDOW_ENV).ok().and_then(|v| Self::parse(&v))
    }

    /// True when on-screen presentation was requested.
    pub fn requested() -> bool {
        matches!(Self::from_env(), Some(_))
    }
}

/// Failure to open or present through the native window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowError {
    /// No window server session, or the backend is not implemented yet.
    Unsupported(&'static str),
    /// AppKit rejected a call (bad selector, nil object, wrong thread).
    AppKit(String),
    /// Metal or the swapchain refused the drawable.
    Metal(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::Unsupported(why) => write!(f, "native window unsupported: {why}"),
            WindowError::AppKit(why) => write!(f, "AppKit: {why}"),
            WindowError::Metal(why) => write!(f, "Metal: {why}"),
        }
    }
}

impl std::error::Error for WindowError {}

/// An open on-screen presenter.
///
/// The API is intentionally pointer-free: a drawable is identified by a
/// monotonic serial assigned when `CAMetalLayer.nextDrawable` succeeds, so the
/// type stays usable from the transport and from tests.
pub struct NativeWindow {
    config: WindowConfig,
    app: ObjcObject,
    window: ObjcObject,
    _view: ObjcObject,
    _device: ObjcObject,
    swapchain: NativeSwapchain,
    present_texture: Option<ObjcObject>,
    drawable_serial: u64,
    presented: u64,
    open: bool,
}

impl std::fmt::Debug for NativeWindow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeWindow")
            .field("config", &self.config)
            .field("drawable_serial", &self.drawable_serial)
            .field("presented", &self.presented)
            .field("open", &self.open)
            .finish()
    }
}

impl NativeWindow {
    /// Opens a window for `config` using the system default Metal device.
    ///
    /// Fails with [`WindowError::Unsupported`] when AppKit is unavailable, when
    /// the process is not on the main thread or when there is no window server
    /// session. It never panics.
    pub fn open(config: WindowConfig) -> Result<Self, WindowError> {
        let device = objc::create_system_default_device().ok_or(WindowError::Unsupported(
            "no system default Metal device is available",
        ))?;
        // SAFETY: `device` is a live `MTLDevice`; `newCommandQueue` returns a
        // retained queue (+1) owned by this window's swapchain.
        let queue = unsafe { objc::msg_send_id_0(device.as_raw(), objc::sel(c"newCommandQueue")) };
        let queue = ObjcObject::from_retained(queue).ok_or(WindowError::Unsupported(
            "MTLDevice.newCommandQueue returned null",
        ))?;
        Self::open_with(config, device, queue)
    }

    /// Opens a window that presents on `transport`'s queue.
    ///
    /// The queue is retained, so the swapchain's blit is ordered after the
    /// render work the transport already waited for.
    pub fn open_with_transport(
        config: WindowConfig,
        transport: &DirectMetalTransport,
    ) -> Result<Self, WindowError> {
        // SAFETY: `transport.device_raw()` is a live `MTLDevice`; `retain`
        // returns the same device +1 so the window keeps it alive.
        let device_raw =
            unsafe { objc::msg_send_id_0(transport.device_raw(), objc::sel(c"retain")) };
        let device = ObjcObject::from_retained(device_raw).ok_or(WindowError::Unsupported(
            "retaining the transport device returned null",
        ))?;
        // SAFETY: `queue_raw()` is a live `MTLCommandQueue`; `retain` returns
        // the same queue +1 for the swapchain.
        let queue_raw = unsafe { objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"retain")) };
        let queue = ObjcObject::from_retained(queue_raw).ok_or(WindowError::Unsupported(
            "retaining the transport queue returned null",
        ))?;
        Self::open_with(config, device, queue)
    }

    fn open_with(
        config: WindowConfig,
        device: ObjcObject,
        queue: ObjcObject,
    ) -> Result<Self, WindowError> {
        if !window_server_available() {
            return Err(WindowError::Unsupported("no window server session"));
        }
        if !is_main_thread() {
            return Err(WindowError::Unsupported(
                "NSWindow must be created on the main thread",
            ));
        }

        let _pool = AutoreleasePool::new();
        let app_class = objc::class(c"NSApplication");
        if app_class.is_null() {
            return Err(WindowError::Unsupported("NSApplication is unavailable"));
        }
        // SAFETY: `sharedApplication` returns the process-wide application
        // object; `retain` keeps our handle alive for the window's lifetime.
        let app = unsafe { objc::msg_send_id_0(app_class, objc::sel(c"sharedApplication")) };
        let app = retain_object(app).ok_or(WindowError::Unsupported(
            "NSApplication.sharedApplication returned nil",
        ))?;
        // SAFETY: `app` is a live `NSApplication`.
        unsafe {
            msg_send_set_activation_policy(app.as_raw(), NS_ACTIVATION_POLICY_REGULAR);
            objc::msg_send_void_0(app.as_raw(), objc::sel(c"finishLaunching"));
        }

        let width = config.width as f64;
        let height = config.height as f64;
        let rect = CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize { width, height },
        };

        let window_class = objc::class(c"NSWindow");
        if window_class.is_null() {
            return Err(WindowError::Unsupported("NSWindow is unavailable"));
        }
        // SAFETY: `NSWindow alloc` returns a new instance and
        // `initWithContentRect:styleMask:backing:defer:` consumes it.
        let window = unsafe {
            let alloc = objc::msg_send_id_0(window_class, objc::sel(c"alloc"));
            init_window(
                alloc,
                rect,
                NS_WINDOW_STYLE_TITLED
                    | NS_WINDOW_STYLE_CLOSABLE
                    | NS_WINDOW_STYLE_MINIATURIZABLE
                    | NS_WINDOW_STYLE_RESIZABLE,
                NS_BACKING_STORE_BUFFERED,
            )
        };
        let window = ObjcObject::from_retained(window).ok_or(WindowError::AppKit(
            "NSWindow initialization returned nil".to_string(),
        ))?;
        // SAFETY: `window` is a live `NSWindow`; the title string is copied by
        // AppKit and the selector takes a BOOL.
        unsafe {
            msg_send_void_u8(window.as_raw(), objc::sel(c"setReleasedWhenClosed:"), 0);
            if let Some(title) = objc::nsstring(&config.title) {
                objc::msg_send_void_1(window.as_raw(), objc::sel(c"setTitle:"), title.as_raw());
            }
            objc::msg_send_void_0(window.as_raw(), objc::sel(c"center"));
        }

        let layer_class = objc::class(c"CAMetalLayer");
        if layer_class.is_null() {
            return Err(WindowError::Unsupported("CAMetalLayer is unavailable"));
        }
        // SAFETY: `CAMetalLayer alloc` + `init` returns a retained layer.
        let layer = unsafe {
            let alloc = objc::msg_send_id_0(layer_class, objc::sel(c"alloc"));
            objc::msg_send_id_0(alloc, objc::sel(c"init"))
        };
        let layer = ObjcObject::from_retained(layer).ok_or(WindowError::AppKit(
            "CAMetalLayer initialization returned nil".to_string(),
        ))?;
        // SAFETY: `layer` and `device` are live Metal/AppKit objects; the
        // integer and struct setters use the documented property types.
        unsafe {
            objc::msg_send_void_1(layer.as_raw(), objc::sel(c"setDevice:"), device.as_raw());
            msg_send_void_usize(
                layer.as_raw(),
                objc::sel(c"setPixelFormat:"),
                MTL_PIXEL_FORMAT_BGRA8_UNORM,
            );
            msg_send_void_u8(layer.as_raw(), objc::sel(c"setFramebufferOnly:"), 1);
            msg_send_void_usize(layer.as_raw(), objc::sel(c"setMaximumDrawableCount:"), 3);
            msg_send_void_size(
                layer.as_raw(),
                objc::sel(c"setDrawableSize:"),
                CGSize { width, height },
            );
            msg_send_void_f64(layer.as_raw(), objc::sel(c"setContentsScale:"), 1.0);
        }

        let view_class = objc::class(c"NSView");
        if view_class.is_null() {
            return Err(WindowError::Unsupported("NSView is unavailable"));
        }
        // SAFETY: `NSView alloc` + `initWithFrame:` returns a retained view.
        let view = unsafe {
            let alloc = objc::msg_send_id_0(view_class, objc::sel(c"alloc"));
            msg_send_id_rect(alloc, objc::sel(c"initWithFrame:"), rect)
        };
        let view = ObjcObject::from_retained(view).ok_or(WindowError::AppKit(
            "NSView initialization returned nil".to_string(),
        ))?;
        // SAFETY: `view` is a live `NSView`, `layer` a live `CAMetalLayer`; the
        // content view retains the view and the view retains the layer.
        unsafe {
            msg_send_void_u8(view.as_raw(), objc::sel(c"setWantsLayer:"), 1);
            objc::msg_send_void_1(view.as_raw(), objc::sel(c"setLayer:"), layer.as_raw());
            objc::msg_send_void_1(
                window.as_raw(),
                objc::sel(c"setContentView:"),
                view.as_raw(),
            );
            objc::msg_send_void_1(
                window.as_raw(),
                objc::sel(c"makeKeyAndOrderFront:"),
                null_mut(),
            );
            msg_send_void_u8(app.as_raw(), objc::sel(c"activateIgnoringOtherApps:"), 1);
        }

        let swapchain = NativeSwapchain::new(layer, queue.as_raw(), config.width, config.height)
            .ok_or(WindowError::Unsupported(
                "could not retain the presentation command queue",
            ))?;

        Ok(Self {
            config,
            app,
            window,
            _view: view,
            _device: device,
            swapchain,
            present_texture: None,
            drawable_serial: 0,
            presented: 0,
            open: true,
        })
    }

    /// The configuration this window was opened with.
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    /// Drawables acquired so far.
    pub fn drawable_serial(&self) -> u64 {
        self.drawable_serial
    }

    /// Drawables presented so far.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// True while the window has not been closed.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Retains `texture` as the source for the next [`Self::present`].
    ///
    /// The texture is copied into the drawable on the GPU, never through the
    /// CPU. Passing the transport's registered present surface keeps the
    /// presented pixels identical to the headless frame.
    pub fn set_present_surface(
        &mut self,
        transport: &DirectMetalTransport,
        surface_id: u64,
    ) -> Result<(), WindowError> {
        let (surface, format) = {
            let state = transport
                .lock_state()
                .map_err(|error| WindowError::Metal(error.to_string()))?;
            state
                .surfaces
                .get(&surface_id)
                // SAFETY: `surfaces` holds live retained textures; `retain`
                // returns the same object +1 for the window and `pixelFormat`
                // only reads an immutable property.
                .map(|texture| unsafe {
                    let retained = objc::msg_send_id_0(texture.as_raw(), objc::sel(c"retain"));
                    (
                        retained,
                        objc::msg_send_usize_0(retained, objc::sel(c"pixelFormat")),
                    )
                })
                .ok_or_else(|| {
                    WindowError::Metal(format!("present surface {surface_id} is not registered"))
                })?
        };
        let texture = ObjcObject::from_retained(surface).ok_or_else(|| {
            WindowError::Metal("retaining the present surface returned null".to_string())
        })?;
        // A AAA frame is RGBA8, while the layer opens as BGRA8. Switching the
        // layer to the surface format lets the present path stay a GPU blit
        // instead of a CPU swizzle.
        if matches!(format, 70 | 80) {
            self.swapchain.set_pixel_format(format);
        }
        self.present_texture = Some(texture);
        Ok(())
    }

    /// Acquires the next drawable, if the layer can supply one.
    pub fn next_drawable(&mut self) -> Option<u64> {
        if !self.open {
            return None;
        }
        let serial = self.swapchain.next_drawable()?;
        self.drawable_serial = serial;
        Some(serial)
    }

    /// Presents the most recently acquired drawable.
    ///
    /// The drawable's texture is the presented surface: the registered present
    /// texture is blitted into it on the GPU when the formats and extents
    /// match, otherwise the drawable is cleared to a diagnostic colour.
    pub fn present(&mut self) -> Result<(), WindowError> {
        if !self.open {
            return Err(WindowError::Unsupported("the native window is closed"));
        }
        let source = self.present_texture.as_ref().map(ObjcObject::as_raw);
        self.swapchain.present(source).map_err(map_present_error)?;
        self.present_texture = None;
        self.presented = self.presented.wrapping_add(1);
        Ok(())
    }

    /// Pumps one AppKit event cycle so the window stays responsive.
    pub fn pump_events(&mut self) {
        if !self.open {
            return;
        }
        let _pool = AutoreleasePool::new();
        // SAFETY: `app` is a live `NSApplication`; `dateWithTimeIntervalSinceNow:`
        // returns an autoreleased `NSDate` valid inside this pool.
        let now = unsafe {
            type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, f64) -> objc::Id;
            let send: Fn = mem::transmute(objc::msg_send_entry());
            send(
                objc::class(c"NSDate"),
                objc::sel(c"dateWithTimeIntervalSinceNow:"),
                0.0,
            )
        };
        if now.is_null() {
            return;
        }
        // SAFETY: `nextEventMatchingMask:untilDate:inMode:dequeue:` returns an
        // autoreleased event or nil; events are dispatched to `app`.
        unsafe {
            loop {
                let event = next_event(
                    self.app.as_raw(),
                    NS_EVENT_MASK_ANY,
                    now,
                    default_run_loop_mode(),
                );
                if event.is_null() {
                    break;
                }
                objc::msg_send_void_1(self.app.as_raw(), objc::sel(c"sendEvent:"), event);
            }
            objc::msg_send_void_0(self.app.as_raw(), objc::sel(c"updateWindows"));
            if objc::msg_send_u8_0(self.window.as_raw(), objc::sel(c"isVisible")) == 0 {
                self.open = false;
            }
        }
    }

    /// Closes the window and stops presenting. Idempotent and non-blocking.
    pub fn close(&mut self) {
        if !self.open {
            return;
        }
        // SAFETY: `window` is a live `NSWindow`; `orderOut:` takes a sender id
        // (nil is valid) and `close` releases the window server resources.
        unsafe {
            objc::msg_send_void_1(self.window.as_raw(), objc::sel(c"orderOut:"), null_mut());
            objc::msg_send_void_0(self.window.as_raw(), objc::sel(c"close"));
        }
        self.open = false;
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        self.close();
    }
}

fn map_present_error(error: PresentError) -> WindowError {
    WindowError::Metal(error.to_string())
}

/// Retains a borrowed object and wraps it as an owned handle.
fn retain_object(raw: objc::Id) -> Option<ObjcObject> {
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` is a live Objective-C object; `retain` returns it +1.
    let retained = unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) };
    ObjcObject::from_retained(retained)
}

/// Whether AppKit can reach a window server (false in headless CI/ssh).
fn window_server_available() -> bool {
    // SAFETY: `CGSessionCopyCurrentDictionary` returns a +1 CFDictionary or
    // null; the non-null result is released exactly once.
    unsafe {
        let session = CGSessionCopyCurrentDictionary();
        if session.is_null() {
            return false;
        }
        CFRelease(session);
        true
    }
}

fn is_main_thread() -> bool {
    let class = objc::class(c"NSThread");
    if class.is_null() {
        return false;
    }
    // SAFETY: `+[NSThread isMainThread]` takes no arguments and returns BOOL.
    unsafe { objc::msg_send_u8_0(class, objc::sel(c"isMainThread")) != 0 }
}

/// `NSRect`/`CGRect` (`CGPoint origin, CGSize size`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}

/// `CGPoint` (`double x, y`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

/// `CGSize` (`double width, height`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    static NSDefaultRunLoopMode: *mut c_void;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGSessionCopyCurrentDictionary() -> *const c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

fn default_run_loop_mode() -> objc::Id {
    // SAFETY: `NSDefaultRunLoopMode` is an immutable global `NSString *`.
    unsafe { NSDefaultRunLoopMode }
}

unsafe fn init_window(alloc: objc::Id, rect: CGRect, style: usize, backing: usize) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, CGRect, usize, usize, u8) -> objc::Id;
    // SAFETY: `alloc` is a fresh `NSWindow`; `initWithContentRect:...` consumes it.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: Argument layout matches `initWithContentRect:styleMask:backing:defer:`.
    unsafe {
        send(
            alloc,
            objc::sel(c"initWithContentRect:styleMask:backing:defer:"),
            rect,
            style,
            backing,
            0,
        )
    }
}

unsafe fn msg_send_id_rect(target: objc::Id, s: objc::Sel, rect: CGRect) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, CGRect) -> objc::Id;
    // SAFETY: Caller supplies a valid receiver and a `CGRect` argument.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's argument and return layout.
    unsafe { send(target, s, rect) }
}

unsafe fn msg_send_void_size(target: objc::Id, s: objc::Sel, size: CGSize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, CGSize);
    // SAFETY: Caller supplies a valid receiver and a `CGSize` argument.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's argument layout.
    unsafe { send(target, s, size) };
}

unsafe fn msg_send_void_f64(target: objc::Id, s: objc::Sel, value: f64) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, f64);
    // SAFETY: Caller supplies a valid receiver and a `CGFloat` argument.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's argument layout.
    unsafe { send(target, s, value) };
}

unsafe fn msg_send_void_usize(target: objc::Id, s: objc::Sel, value: usize) {
    // SAFETY: Reuses the shared typed `NSUInteger` setter.
    unsafe { objc::msg_send_void_usize(target, s, value) };
}

unsafe fn msg_send_void_u8(target: objc::Id, s: objc::Sel, value: u8) {
    // SAFETY: Reuses the shared typed `BOOL` setter.
    unsafe { objc::msg_send_void_u8(target, s, value) };
}

unsafe fn msg_send_set_activation_policy(app: objc::Id, policy: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize) -> u8;
    // SAFETY: `app` is a live `NSApplication` and `policy` is a valid enum value.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches `setActivationPolicy:` (BOOL return).
    unsafe {
        send(app, objc::sel(c"setActivationPolicy:"), policy);
    }
}

unsafe fn next_event(app: objc::Id, mask: usize, until: objc::Id, mode: objc::Id) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, objc::Id, objc::Id, u8) -> objc::Id;
    // SAFETY: `app` is a live `NSApplication`, `until` a live `NSDate` and
    // `mode` a live run-loop mode string.
    let send: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's argument and return layout.
    unsafe {
        send(
            app,
            objc::sel(c"nextEventMatchingMask:untilDate:inMode:dequeue:"),
            mask,
            until,
            mode,
            1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_spellings_disable_the_window() {
        for value in ["", "0", "off", "false", "no"] {
            assert_eq!(WindowConfig::parse(value), None, "{value}");
        }
    }

    #[test]
    fn on_uses_default_geometry() {
        assert_eq!(WindowConfig::parse("1"), Some(Ok(WindowConfig::default())));
        assert_eq!(WindowConfig::parse("on"), Some(Ok(WindowConfig::default())));
        assert_eq!(WindowConfig::parse("true"), Some(Ok(WindowConfig::default())));
    }

    #[test]
    fn size_and_title_are_parsed() {
        let parsed = WindowConfig::parse("1920x1080,title=Victory").unwrap().unwrap();
        assert_eq!(parsed.width, 1920);
        assert_eq!(parsed.height, 1080);
        assert_eq!(parsed.title, "Victory");
    }

    #[test]
    fn malformed_specs_are_rejected() {
        assert_eq!(WindowConfig::parse("width"), Some(Err(())));
        assert_eq!(WindowConfig::parse("0x10"), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x0"), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x10,title="), Some(Err(())));
        assert_eq!(WindowConfig::parse("10x10,bogus"), Some(Err(())));
    }

    #[test]
    fn headless_defaults_are_the_documented_geometry() {
        let config = WindowConfig::default();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.width, DEFAULT_WINDOW_WIDTH);
        assert_eq!(config.height, DEFAULT_WINDOW_HEIGHT);
        assert_eq!(config.title, "victory-metal");
    }

    #[test]
    fn size_specs_keep_the_default_title_and_clamp_nothing() {
        let parsed = WindowConfig::parse("64x48").unwrap().unwrap();
        assert_eq!(parsed.width, 64);
        assert_eq!(parsed.height, 48);
        assert_eq!(parsed.title, WindowConfig::default().title);

        // Case-insensitive separators and surrounding whitespace are accepted.
        let parsed = WindowConfig::parse(" 640 X 480 , title= Bench ")
            .unwrap()
            .unwrap();
        assert_eq!((parsed.width, parsed.height), (640, 480));
        assert_eq!(parsed.title, "Bench");
    }

    #[test]
    fn open_either_reports_a_real_window_or_unsupported_without_panicking() {
        // In a windowless CI session `open` must return `Unsupported`; in a
        // real session it must return a usable window. Both paths are a pass,
        // and neither may panic.
        match NativeWindow::open(WindowConfig {
            width: 64,
            height: 64,
            title: "victory-metal-window-unit".to_string(),
        }) {
            Ok(mut window) => {
                assert_eq!(window.config().width, 64);
                assert_eq!(window.config().height, 64);
                assert!(window.is_open());
                window.pump_events();
                window.close();
                assert!(!window.is_open());
            }
            Err(WindowError::Unsupported(reason)) => {
                eprintln!("window unit test: no window server session ({reason})");
            }
            Err(other) => panic!("unexpected window open error: {other}"),
        }
    }
}
