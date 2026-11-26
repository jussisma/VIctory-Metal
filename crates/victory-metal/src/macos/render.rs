//! ARM64 Metal 3D render encoding: render passes, graphics pipelines and
//! draw submission.
//!
//! Scope of this module:
//!
//! - `MTLRenderPassDescriptor` with colour and depth/stencil attachments,
//!   load/store actions and clear values.
//! - A render command encoder that binds a graphics pipeline state, vertex
//!   and index buffers, viewport and scissor, root constants and
//!   descriptor-table ranges, then issues indexed and non-indexed draws.
//! - One command buffer per recorded list with an explicit completion and
//!   error path; no implicit CPU readback and no double execution.
//!
//! The Metal FFI lives here (or in private helpers of this module) so the
//! module can grow without editing `objc.rs`, which other modules share.
//!
//! # Ownership model
//!
//! [`RenderTarget`] owns a retained `MTLTexture`. [`RenderPass`] borrows it for
//! the duration of a recorded pass, and [`RenderEncoder`] borrows the pass and
//! pipeline. The encoder creates its `MTLCommandBuffer` lazily on the first
//! encoding call, so argument errors (zero-size attachments, zero vertex
//! counts, nil index buffers) are reported before any GPU object exists.

use std::ffi::c_void;
use std::mem;

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};
use crate::transport::MetalError;

/// `MTLPixelFormatRGBA8Unorm`.
const MTL_PIXEL_FORMAT_RGBA8_UNORM: usize = 70;
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
/// `MTLPixelFormatDepth32Float`.
const MTL_PIXEL_FORMAT_DEPTH32_FLOAT: usize = 252;
/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: usize = 2;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLTextureUsageRenderTarget`.
const MTL_TEXTURE_USAGE_RENDER_TARGET: usize = 4;
/// `MTLCommandBufferStatusCompleted`.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// `MTLLoadActionLoad`.
const MTL_LOAD_ACTION_LOAD: usize = 1;
/// `MTLLoadActionClear`.
const MTL_LOAD_ACTION_CLEAR: usize = 2;
/// `MTLStoreActionStore`.
const MTL_STORE_ACTION_STORE: usize = 1;
/// `MTLStoreActionDontCare`.
const MTL_STORE_ACTION_DONT_CARE: usize = 0;
/// `MTLIndexTypeUInt16`.
const MTL_INDEX_TYPE_UINT16: usize = 0;
/// `MTLIndexTypeUInt32`.
const MTL_INDEX_TYPE_UINT32: usize = 1;
/// `MTLPrimitiveTypeTriangle`.
const MTL_PRIMITIVE_TYPE_TRIANGLE: usize = 3;
/// `MTLPrimitiveTypeTriangleStrip`.
const MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP: usize = 4;
/// Largest `setVertexBytes:length:atIndex:` payload accepted by Metal.
const MAX_INLINE_BYTES: usize = 4096;
/// Largest supported level-0 2D render-target edge.
const MAX_RENDER_TARGET_DIMENSION: u32 = 16_384;

/// Apple `MTLClearColor` (`double red, green, blue, alpha`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct MTLClearColor {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

/// Apple `MTLViewport` (`double originX, originY, width, height, znear, zfar`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct MTLViewport {
    origin_x: f64,
    origin_y: f64,
    width: f64,
    height: f64,
    znear: f64,
    zfar: f64,
}

/// Apple `MTLScissorRect` (`NSUInteger x, y, width, height`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLScissorRect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// Apple `MTLOrigin` (`NSUInteger x, y, z`); only used by the test readback blit.
#[cfg(test)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// Supported attachment pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// `MTLPixelFormatBGRA8Unorm`, the common swapchain/camera format.
    Bgra8Unorm,
    /// `MTLPixelFormatRGBA8Unorm`.
    Rgba8Unorm,
    /// `MTLPixelFormatDepth32Float`, valid only for depth attachments.
    Depth32Float,
}

impl PixelFormat {
    /// Whether this format can back a colour attachment.
    pub const fn is_color(self) -> bool {
        matches!(self, Self::Bgra8Unorm | Self::Rgba8Unorm)
    }

    const fn mtl_pixel_format(self) -> usize {
        match self {
            Self::Bgra8Unorm => MTL_PIXEL_FORMAT_BGRA8_UNORM,
            Self::Rgba8Unorm => MTL_PIXEL_FORMAT_RGBA8_UNORM,
            Self::Depth32Float => MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
        }
    }

    fn from_mtl_pixel_format(value: usize) -> Option<Self> {
        match value {
            MTL_PIXEL_FORMAT_BGRA8_UNORM => Some(Self::Bgra8Unorm),
            MTL_PIXEL_FORMAT_RGBA8_UNORM => Some(Self::Rgba8Unorm),
            MTL_PIXEL_FORMAT_DEPTH32_FLOAT => Some(Self::Depth32Float),
            _ => None,
        }
    }
}

/// Load action for a render attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadAction {
    /// Preserve the attachment's previous contents.
    Load,
    /// Discard previous contents and apply the clear value.
    Clear,
}

impl LoadAction {
    const fn mtl_value(self) -> usize {
        match self {
            Self::Load => MTL_LOAD_ACTION_LOAD,
            Self::Clear => MTL_LOAD_ACTION_CLEAR,
        }
    }
}

/// Store action for a render attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreAction {
    /// Keep the rendered contents in the attachment.
    Store,
    /// Discard the rendered contents at the end of the pass.
    DontCare,
}

impl StoreAction {
    const fn mtl_value(self) -> usize {
        match self {
            Self::Store => MTL_STORE_ACTION_STORE,
            Self::DontCare => MTL_STORE_ACTION_DONT_CARE,
        }
    }
}

/// Primitive topology accepted by the draw entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    /// `MTLPrimitiveTypeTriangle`.
    Triangle,
    /// `MTLPrimitiveTypeTriangleStrip`.
    TriangleStrip,
}

impl PrimitiveType {
    const fn mtl_value(self) -> usize {
        match self {
            Self::Triangle => MTL_PRIMITIVE_TYPE_TRIANGLE,
            Self::TriangleStrip => MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP,
        }
    }
}

/// Index width for indexed draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// `MTLIndexTypeUInt16`.
    Uint16,
    /// `MTLIndexTypeUInt32`.
    Uint32,
}

impl IndexType {
    const fn mtl_value(self) -> usize {
        match self {
            Self::Uint16 => MTL_INDEX_TYPE_UINT16,
            Self::Uint32 => MTL_INDEX_TYPE_UINT32,
        }
    }

    const fn alignment(self) -> usize {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Explicit viewport rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// Left origin in pixels.
    pub origin_x: f64,
    /// Top origin in pixels.
    pub origin_y: f64,
    /// Width in pixels; must be positive.
    pub width: f64,
    /// Height in pixels; must be positive.
    pub height: f64,
    /// Near depth plane, normally `0.0`.
    pub znear: f64,
    /// Far depth plane, normally `1.0`.
    pub zfar: f64,
}

/// Explicit scissor rectangle in attachment pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScissorRect {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels; must be nonzero.
    pub width: u32,
    /// Height in pixels; must be nonzero.
    pub height: u32,
}

/// Owning handle to a retained `MTLTexture` used as a render attachment.
pub struct RenderTarget {
    texture: ObjcObject,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
}

impl RenderTarget {
    /// Wraps a retained texture and queries its geometry and pixel format.
    pub(crate) fn from_objc(texture: ObjcObject) -> Result<Self, MetalError> {
        let raw = texture.as_raw();
        // SAFETY: `raw` is a retained `MTLTexture` owned by `texture`.
        let width = unsafe { objc::msg_send_usize_0(raw, objc::sel(c"width")) };
        // SAFETY: See above.
        let height = unsafe { objc::msg_send_usize_0(raw, objc::sel(c"height")) };
        // SAFETY: See above.
        let pixel_format = unsafe { objc::msg_send_usize_0(raw, objc::sel(c"pixelFormat")) };
        if width == 0 || height == 0 {
            return Err(MetalError::InvalidArg(
                "render attachment width and height must be nonzero",
            ));
        }
        let width = u32::try_from(width)
            .map_err(|_| MetalError::InvalidArg("render attachment width exceeds u32"))?;
        let height = u32::try_from(height)
            .map_err(|_| MetalError::InvalidArg("render attachment height exceeds u32"))?;
        let pixel_format =
            PixelFormat::from_mtl_pixel_format(pixel_format).ok_or(MetalError::NotSupported(
                "render attachment must be BGRA8Unorm, RGBA8Unorm or Depth32Float",
            ))?;
        Ok(Self {
            texture,
            width,
            height,
            pixel_format,
        })
    }

    /// Wraps a caller-owned, retained (+1) `MTLTexture *`.
    ///
    /// # Safety
    ///
    /// `raw` must be null or a valid retained `id<MTLTexture>`. On success this
    /// takes ownership of that +1 reference and releases it when the
    /// [`RenderTarget`] drops; the caller must not release or use the pointer
    /// afterwards.
    pub unsafe fn from_retained_raw(raw: *mut c_void) -> Result<Self, MetalError> {
        if raw.is_null() {
            return Err(MetalError::InvalidArg("render attachment texture is nil"));
        }
        let texture = ObjcObject::from_retained(raw)
            .ok_or(MetalError::InvalidArg("render attachment texture is nil"))?;
        Self::from_objc(texture)
    }

    /// Texture width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Texture height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Attachment pixel format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    pub(crate) fn as_raw(&self) -> objc::Id {
        self.texture.as_raw()
    }
}

impl DirectMetalTransport {
    /// Creates a private, level-0 2D texture usable as a render attachment.
    ///
    /// Colour formats are created with shader-read and render-target usage so
    /// the result can be sampled after the pass; depth formats are created with
    /// render-target usage only. Zero or oversized dimensions fail with
    /// [`MetalError::InvalidArg`] before any Metal object is created.
    pub fn create_render_target_2d(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<RenderTarget, MetalError> {
        if width == 0 || height == 0 {
            return Err(MetalError::InvalidArg(
                "render target width and height must be nonzero",
            ));
        }
        if width > MAX_RENDER_TARGET_DIMENSION || height > MAX_RENDER_TARGET_DIMENSION {
            return Err(MetalError::InvalidArg(
                "render target exceeds the supported 2D dimension limit",
            ));
        }
        let usage = if format.is_color() {
            MTL_TEXTURE_USAGE_SHADER_READ | MTL_TEXTURE_USAGE_RENDER_TARGET
        } else {
            MTL_TEXTURE_USAGE_RENDER_TARGET
        };

        let _pool = AutoreleasePool::new();
        let descriptor =
            texture_2d_descriptor(format.mtl_pixel_format(), width as usize, height as usize)?;
        // SAFETY: `descriptor` is a live `MTLTextureDescriptor` in this pool and
        // the selectors take documented `MTLStorageMode`/`MTLTextureUsage` values.
        unsafe {
            msg_send_void_usize(
                descriptor,
                objc::sel(c"setStorageMode:"),
                MTL_STORAGE_MODE_PRIVATE,
            );
            msg_send_void_usize(descriptor, objc::sel(c"setUsage:"), usage);
        }

        // SAFETY: `device` is a valid `MTLDevice`, `descriptor` is a live
        // descriptor, and `newTextureWithDescriptor:` returns a retained texture.
        let raw = unsafe {
            objc::msg_send_id_1(
                self.device_raw(),
                objc::sel(c"newTextureWithDescriptor:"),
                descriptor,
            )
        };
        let texture = ObjcObject::from_retained(raw).ok_or_else(|| {
            MetalError::InternalError(
                "MTLDevice.newTextureWithDescriptor returned null".to_string(),
            )
        })?;
        RenderTarget::from_objc(texture)
    }
}

/// One colour or depth attachment plus its load/store/clear state.
pub struct RenderAttachment {
    target: RenderTarget,
    load_action: LoadAction,
    store_action: StoreAction,
    clear_color: [f64; 4],
    clear_depth: f64,
}

impl RenderAttachment {
    /// Colour attachment that clears to opaque black by default.
    pub fn color(target: RenderTarget) -> Self {
        Self {
            target,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            clear_depth: 1.0,
        }
    }

    /// Depth attachment that clears to `clear_depth` by default.
    pub fn depth(target: RenderTarget, clear_depth: f64) -> Self {
        Self {
            target,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            clear_depth,
        }
    }

    /// Overrides the load action.
    pub fn with_load_action(mut self, action: LoadAction) -> Self {
        self.load_action = action;
        self
    }

    /// Overrides the store action.
    pub fn with_store_action(mut self, action: StoreAction) -> Self {
        self.store_action = action;
        self
    }

    /// Overrides the RGBA clear colour (only used with [`LoadAction::Clear`]).
    pub fn with_clear_color(mut self, color: [f64; 4]) -> Self {
        self.clear_color = color;
        self
    }

    /// Overrides the depth clear value (only used with [`LoadAction::Clear`]).
    pub fn with_clear_depth(mut self, depth: f64) -> Self {
        self.clear_depth = depth;
        self
    }

    /// The attachment texture.
    pub fn target(&self) -> &RenderTarget {
        &self.target
    }
}

/// A validated colour attachment plus an optional matching depth attachment.
pub struct RenderPass {
    color: RenderAttachment,
    depth: Option<RenderAttachment>,
}

impl RenderPass {
    /// Starts a pass with a colour attachment.
    ///
    /// Rejects a non-colour or nil attachment texture with
    /// [`MetalError::InvalidArg`].
    pub fn new(color: RenderAttachment) -> Result<Self, MetalError> {
        if !color.target.pixel_format.is_color() {
            return Err(MetalError::InvalidArg(
                "colour attachment must be BGRA8Unorm or RGBA8Unorm",
            ));
        }
        Ok(Self { color, depth: None })
    }

    /// Attaches a depth target.
    ///
    /// The depth format must be [`PixelFormat::Depth32Float`] and its
    /// dimensions must match the colour attachment exactly; otherwise
    /// [`MetalError::InvalidArg`] is returned.
    pub fn with_depth(mut self, depth: RenderAttachment) -> Result<Self, MetalError> {
        if depth.target.pixel_format != PixelFormat::Depth32Float {
            return Err(MetalError::InvalidArg(
                "depth attachment must be Depth32Float",
            ));
        }
        if depth.target.width != self.color.target.width
            || depth.target.height != self.color.target.height
        {
            return Err(MetalError::InvalidArg(
                "depth attachment size must match the colour attachment",
            ));
        }
        self.depth = Some(depth);
        Ok(self)
    }

    /// The colour attachment texture.
    pub fn color_target(&self) -> &RenderTarget {
        &self.color.target
    }

    /// The optional depth attachment texture.
    pub fn depth_target(&self) -> Option<&RenderTarget> {
        self.depth.as_ref().map(|depth| &depth.target)
    }
}

/// A compiled `MTLRenderPipelineState` for one vertex/fragment pair.
pub struct GraphicsPipeline {
    state: ObjcObject,
    color_format: PixelFormat,
    depth_format: Option<PixelFormat>,
}

impl GraphicsPipeline {
    /// Compiles `source` and builds a render pipeline state.
    ///
    /// The library is compiled once; both named entry points are resolved from
    /// it. MSL or pipeline-creation failures return
    /// [`MetalError::CompileFailed`] carrying Metal's diagnostic message.
    pub fn compile(
        transport: &DirectMetalTransport,
        source: &str,
        vertex_entry: &str,
        fragment_entry: &str,
        color_format: PixelFormat,
        depth_format: Option<PixelFormat>,
    ) -> Result<Self, MetalError> {
        if source.is_empty() || vertex_entry.is_empty() || fragment_entry.is_empty() {
            return Err(MetalError::InvalidArg(
                "MSL source and both entry point names must be non-empty",
            ));
        }
        if !color_format.is_color() {
            return Err(MetalError::InvalidArg(
                "render pipeline colour format must be BGRA8Unorm or RGBA8Unorm",
            ));
        }
        if let Some(format) = depth_format
            && format != PixelFormat::Depth32Float
        {
            return Err(MetalError::InvalidArg(
                "render pipeline depth format must be Depth32Float",
            ));
        }

        let _pool = AutoreleasePool::new();
        let source_ns = objc::nsstring(source).ok_or(MetalError::InvalidArg(
            "MSL source string allocation failed",
        ))?;
        let mut ns_err = objc::empty_id();
        // SAFETY: `device` is a valid `MTLDevice`, `source_ns` is an immutable
        // `NSString`, `options` is nil, and `ns_err` receives an `NSError *`.
        let lib_raw = unsafe {
            objc::msg_send_id_3_err(
                transport.device_raw(),
                objc::sel(c"newLibraryWithSource:options:error:"),
                source_ns.as_raw(),
                objc::empty_id(),
                &mut ns_err,
            )
        };
        let library = ObjcObject::from_retained(lib_raw).ok_or_else(|| {
            // SAFETY: Metal populated `ns_err` when compilation failed.
            let message = unsafe { objc::nserror_message(ns_err) };
            MetalError::CompileFailed(message)
        })?;

        let vertex = new_library_function(&library, vertex_entry)?;
        let fragment = new_library_function(&library, fragment_entry)?;

        let descriptor_class = objc::class(c"MTLRenderPipelineDescriptor");
        if descriptor_class.is_null() {
            return Err(MetalError::InternalError(
                "MTLRenderPipelineDescriptor class unavailable".to_string(),
            ));
        }
        // SAFETY: `descriptor_class` is the documented class and `new` returns a
        // retained, zero-initialized descriptor.
        let descriptor_raw = unsafe { objc::msg_send_id_0(descriptor_class, objc::sel(c"new")) };
        let descriptor = ObjcObject::from_retained(descriptor_raw).ok_or_else(|| {
            MetalError::InternalError(
                "MTLRenderPipelineDescriptor creation returned null".to_string(),
            )
        })?;

        // SAFETY: All receivers are live Metal objects and the selectors take
        // the documented object/`NSUInteger` arguments.
        unsafe {
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setVertexFunction:"),
                vertex.as_raw(),
            );
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setFragmentFunction:"),
                fragment.as_raw(),
            );
            let attachments =
                objc::msg_send_id_0(descriptor.as_raw(), objc::sel(c"colorAttachments"));
            if attachments.is_null() {
                return Err(MetalError::InternalError(
                    "MTLRenderPipelineDescriptor.colorAttachments is nil".to_string(),
                ));
            }
            let attachment =
                msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0);
            if attachment.is_null() {
                return Err(MetalError::InternalError(
                    "colour attachment descriptor at index 0 is nil".to_string(),
                ));
            }
            msg_send_void_usize(
                attachment,
                objc::sel(c"setPixelFormat:"),
                color_format.mtl_pixel_format(),
            );
            if let Some(format) = depth_format {
                msg_send_void_usize(
                    descriptor.as_raw(),
                    objc::sel(c"setDepthAttachmentPixelFormat:"),
                    format.mtl_pixel_format(),
                );
            }
        }

        let mut ns_err = objc::empty_id();
        // SAFETY: `descriptor` is fully configured and `ns_err` receives an
        // autoreleased `NSError *` on failure.
        let state_raw = unsafe {
            objc::msg_send_id_err(
                transport.device_raw(),
                objc::sel(c"newRenderPipelineStateWithDescriptor:error:"),
                descriptor.as_raw(),
                &mut ns_err,
            )
        };
        let state = ObjcObject::from_retained(state_raw).ok_or_else(|| {
            // SAFETY: Metal populated `ns_err` when pipeline creation failed.
            let message = unsafe { objc::nserror_message(ns_err) };
            MetalError::CompileFailed(message)
        })?;

        Ok(Self {
            state,
            color_format,
            depth_format,
        })
    }
}

/// Resolves one named function from a compiled `MTLLibrary`.
fn new_library_function(library: &ObjcObject, entry: &str) -> Result<ObjcObject, MetalError> {
    let name = objc::nsstring(entry).ok_or(MetalError::InvalidArg(
        "shader entry string allocation failed",
    ))?;
    // SAFETY: `library` is a valid `MTLLibrary` and `name` is a retained
    // `NSString`; `newFunctionWithName:` returns a retained `MTLFunction`.
    let raw = unsafe {
        objc::msg_send_id_1(
            library.as_raw(),
            objc::sel(c"newFunctionWithName:"),
            name.as_raw(),
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        MetalError::NotFound(format!("shader function `{entry}` not found in MTLLibrary"))
    })
}

/// Records one render pass into one command buffer.
///
/// The command buffer and render encoder are created on the first encoding
/// call, so draw/binding validation errors are reported before any GPU work is
/// submitted. Call [`RenderEncoder::finish`] to close, commit and wait.
pub struct RenderEncoder<'a> {
    transport: &'a DirectMetalTransport,
    pass: &'a RenderPass,
    pipeline: &'a GraphicsPipeline,
    pool: Option<AutoreleasePool>,
    cb: objc::Id,
    enc: objc::Id,
    ended: bool,
}

impl<'a> RenderEncoder<'a> {
    /// Prepares an encoder for `pass` using `pipeline`.
    ///
    /// Pipeline and pass formats must agree; a mismatch or nil attachment is
    /// reported as [`MetalError::InvalidArg`] before a command buffer exists.
    pub fn new(
        transport: &'a DirectMetalTransport,
        pass: &'a RenderPass,
        pipeline: &'a GraphicsPipeline,
    ) -> Result<Self, MetalError> {
        if pass.color.target.pixel_format != pipeline.color_format {
            return Err(MetalError::InvalidArg(
                "pipeline colour format does not match the render pass",
            ));
        }
        let pass_depth = pass.depth.as_ref().map(|depth| depth.target.pixel_format);
        if pass_depth != pipeline.depth_format {
            return Err(MetalError::InvalidArg(
                "pipeline depth format does not match the render pass",
            ));
        }
        Ok(Self {
            transport,
            pass,
            pipeline,
            pool: None,
            cb: objc::empty_id(),
            enc: objc::empty_id(),
            ended: false,
        })
    }

    /// Sets the viewport. Width and height must be positive.
    pub fn set_viewport(&mut self, viewport: Viewport) -> Result<(), MetalError> {
        if viewport.width <= 0.0 || viewport.height <= 0.0 {
            return Err(MetalError::InvalidArg(
                "viewport width and height must be positive",
            ));
        }
        self.ensure_encoder()?;
        let viewport = MTLViewport {
            origin_x: viewport.origin_x,
            origin_y: viewport.origin_y,
            width: viewport.width,
            height: viewport.height,
            znear: viewport.znear,
            zfar: viewport.zfar,
        };
        // SAFETY: `self.enc` is a live render encoder and `viewport` matches
        // the `MTLViewport` argument layout.
        unsafe { msg_send_void_viewport(self.enc, objc::sel(c"setViewport:"), viewport) };
        Ok(())
    }

    /// Sets the scissor rectangle. Width and height must be nonzero.
    pub fn set_scissor_rect(&mut self, rect: ScissorRect) -> Result<(), MetalError> {
        if rect.width == 0 || rect.height == 0 {
            return Err(MetalError::InvalidArg(
                "scissor width and height must be nonzero",
            ));
        }
        self.ensure_encoder()?;
        let rect = MTLScissorRect {
            x: rect.x as usize,
            y: rect.y as usize,
            width: rect.width as usize,
            height: rect.height as usize,
        };
        // SAFETY: `self.enc` is a live render encoder and `rect` matches the
        // `MTLScissorRect` argument layout.
        unsafe { msg_send_void_scissor(self.enc, objc::sel(c"setScissorRect:"), rect) };
        Ok(())
    }

    /// Binds an `MTLBuffer` for the vertex stage at `index`.
    ///
    /// # Safety
    ///
    /// `buffer` must be a live `id<MTLBuffer>` (or null) whose backing storage
    /// stays valid until the pass's command buffer completes.
    pub unsafe fn set_vertex_buffer(
        &mut self,
        buffer: *mut c_void,
        offset: usize,
        index: usize,
    ) -> Result<(), MetalError> {
        self.bind_buffer(buffer, offset, index, BufferStage::Vertex)
    }

    /// Binds an `MTLBuffer` for the fragment stage at `index`.
    ///
    /// # Safety
    ///
    /// `buffer` must be a live `id<MTLBuffer>` (or null) whose backing storage
    /// stays valid until the pass's command buffer completes.
    pub unsafe fn set_fragment_buffer(
        &mut self,
        buffer: *mut c_void,
        offset: usize,
        index: usize,
    ) -> Result<(), MetalError> {
        self.bind_buffer(buffer, offset, index, BufferStage::Fragment)
    }

    fn bind_buffer(
        &mut self,
        buffer: *mut c_void,
        offset: usize,
        index: usize,
        stage: BufferStage,
    ) -> Result<(), MetalError> {
        if buffer.is_null() {
            return Err(MetalError::InvalidArg("render buffer is nil"));
        }
        if !offset.is_multiple_of(4) {
            return Err(MetalError::InvalidArg(
                "render buffer offset must be 4-byte aligned",
            ));
        }
        self.ensure_encoder()?;
        let selector = match stage {
            BufferStage::Vertex => objc::sel(c"setVertexBuffer:offset:atIndex:"),
            BufferStage::Fragment => objc::sel(c"setFragmentBuffer:offset:atIndex:"),
        };
        // SAFETY: `self.enc` is a live render encoder, `buffer` is a caller-
        // provided live `MTLBuffer` (non-null, per the check above), and the
        // offset/index match the selector's `NSUInteger` arguments.
        unsafe { msg_send_void_buffer(self.enc, selector, buffer, offset, index) };
        Ok(())
    }

    /// Copies `bytes` into the vertex shader's constant argument slot `index`.
    ///
    /// Metal copies the payload during the call, so `bytes` need not outlive
    /// it. Payloads larger than 4 KiB are rejected.
    pub fn set_vertex_bytes(&mut self, bytes: &[u8], index: usize) -> Result<(), MetalError> {
        self.set_inline_bytes(bytes, index, BufferStage::Vertex)
    }

    /// Copies `bytes` into the fragment shader's constant argument slot `index`.
    ///
    /// Metal copies the payload during the call, so `bytes` need not outlive
    /// it. Payloads larger than 4 KiB are rejected.
    pub fn set_fragment_bytes(&mut self, bytes: &[u8], index: usize) -> Result<(), MetalError> {
        self.set_inline_bytes(bytes, index, BufferStage::Fragment)
    }

    fn set_inline_bytes(
        &mut self,
        bytes: &[u8],
        index: usize,
        stage: BufferStage,
    ) -> Result<(), MetalError> {
        if bytes.is_empty() || bytes.len() > MAX_INLINE_BYTES {
            return Err(MetalError::InvalidArg(
                "inline root constants must be 1..=4096 bytes",
            ));
        }
        self.ensure_encoder()?;
        let selector = match stage {
            BufferStage::Vertex => objc::sel(c"setVertexBytes:length:atIndex:"),
            BufferStage::Fragment => objc::sel(c"setFragmentBytes:length:atIndex:"),
        };
        // SAFETY: `self.enc` is a live render encoder, `bytes` is valid for
        // `bytes.len()` bytes during the call, and Metal copies it immediately.
        unsafe {
            msg_send_void_bytes(self.enc, selector, bytes.as_ptr(), bytes.len(), index);
        }
        Ok(())
    }

    /// Draws `vertex_count` non-indexed vertices starting at `vertex_start`.
    ///
    /// A zero vertex count returns [`MetalError::InvalidArg`] without creating
    /// the command buffer.
    pub fn draw_primitives(
        &mut self,
        primitive: PrimitiveType,
        vertex_start: usize,
        vertex_count: usize,
    ) -> Result<(), MetalError> {
        if vertex_count == 0 {
            return Err(MetalError::InvalidArg("draw vertex count must be nonzero"));
        }
        self.ensure_encoder()?;
        // SAFETY: `self.enc` is a live render encoder and both counts are
        // passed as the selector's `NSUInteger` arguments.
        unsafe {
            msg_send_void_two_usize(
                self.enc,
                objc::sel(c"drawPrimitives:vertexStart:vertexCount:"),
                primitive.mtl_value(),
                vertex_start,
                vertex_count,
            );
        }
        Ok(())
    }

    /// Draws `index_count` indices from `index_buffer`, adding `base_vertex` to
    /// every fetched index.
    ///
    /// A zero index count, a nil index buffer or a misaligned index-buffer
    /// offset returns [`MetalError::InvalidArg`] without creating the command
    /// buffer.
    ///
    /// # Safety
    ///
    /// `index_buffer` must be a live `id<MTLBuffer>` whose backing storage
    /// stays valid until the pass's command buffer completes.
    pub unsafe fn draw_indexed_primitives(
        &mut self,
        primitive: PrimitiveType,
        index_count: usize,
        index_type: IndexType,
        index_buffer: *mut c_void,
        index_buffer_offset: usize,
        base_vertex: isize,
    ) -> Result<(), MetalError> {
        if index_count == 0 {
            return Err(MetalError::InvalidArg("draw index count must be nonzero"));
        }
        if index_buffer.is_null() {
            return Err(MetalError::InvalidArg("index buffer is nil"));
        }
        if !index_buffer_offset.is_multiple_of(index_type.alignment()) {
            return Err(MetalError::InvalidArg(
                "index buffer offset is not aligned to the index type",
            ));
        }
        self.ensure_encoder()?;
        // SAFETY: `self.enc` is a live render encoder, `index_buffer` is a
        // caller-provided live `MTLBuffer` (non-null, checked above), and the
        // scalar arguments match the selector's ABI.
        unsafe {
            msg_send_draw_indexed(
                self.enc,
                objc::sel(c"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                primitive.mtl_value(),
                index_count,
                index_type.mtl_value(),
                index_buffer,
                index_buffer_offset,
                1,
                base_vertex,
                0,
            );
        }
        Ok(())
    }

    /// Closes the encoder, commits the command buffer and waits for completion.
    ///
    /// GPU faults are returned as [`MetalError::InternalError`] with Metal's
    /// command-buffer diagnostic. No pixels are read back here.
    pub fn finish(mut self) -> Result<(), MetalError> {
        self.ensure_encoder()?;
        let cb = self.cb;
        let enc = self.enc;
        // SAFETY: `enc` is a live render encoder that has not been ended yet.
        unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
        self.ended = true;
        // SAFETY: `cb` is a live command buffer with every encoder closed.
        unsafe {
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
        // SAFETY: `cb` completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: `[cb error]` returns null or an autoreleased `NSError *`.
            let error_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
            // SAFETY: `error_raw` is null or a valid `NSError *`.
            let message = unsafe { objc::nserror_message(error_raw) };
            return Err(MetalError::InternalError(format!(
                "Metal render command buffer failed (status={status}): {message}"
            )));
        }
        Ok(())
    }

    /// Creates the pipe (command buffer, pass descriptor, render encoder) on
    /// first use.
    fn ensure_encoder(&mut self) -> Result<(), MetalError> {
        if !self.enc.is_null() {
            return Ok(());
        }
        let pool = AutoreleasePool::new();
        // SAFETY: `queue` is a valid `MTLCommandQueue`; the returned command
        // buffer is autoreleased and stays alive for `pool`'s lifetime.
        let cb =
            unsafe { objc::msg_send_id_0(self.transport.queue_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }
        let descriptor = build_render_pass_descriptor(self.pass)?;
        // SAFETY: `cb` is a live uncommitted command buffer and `descriptor`
        // is a live `MTLRenderPassDescriptor` in this pool.
        let enc = unsafe {
            objc::msg_send_id_1(
                cb,
                objc::sel(c"renderCommandEncoderWithDescriptor:"),
                descriptor,
            )
        };
        if enc.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandBuffer.renderCommandEncoderWithDescriptor returned null".to_string(),
            ));
        }

        let color = &self.pass.color.target;
        let viewport = MTLViewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: f64::from(color.width),
            height: f64::from(color.height),
            znear: 0.0,
            zfar: 1.0,
        };
        let scissor = MTLScissorRect {
            x: 0,
            y: 0,
            width: color.width as usize,
            height: color.height as usize,
        };
        // SAFETY: `enc` is a live render encoder, `pipeline.state` is a valid
        // `MTLRenderPipelineState`, and the default viewport/scissor cover the
        // colour attachment.
        unsafe {
            objc::msg_send_void_1(
                enc,
                objc::sel(c"setRenderPipelineState:"),
                self.pipeline.state.as_raw(),
            );
            msg_send_void_viewport(enc, objc::sel(c"setViewport:"), viewport);
            msg_send_void_scissor(enc, objc::sel(c"setScissorRect:"), scissor);
        }

        self.pool = Some(pool);
        self.cb = cb;
        self.enc = enc;
        Ok(())
    }
}

impl Drop for RenderEncoder<'_> {
    fn drop(&mut self) {
        if !self.enc.is_null() && !self.ended {
            // SAFETY: `self.enc` is a live render encoder created by
            // `ensure_encoder` and not yet ended.
            unsafe { objc::msg_send_void_0(self.enc, objc::sel(c"endEncoding")) };
            self.ended = true;
        }
    }
}

#[derive(Clone, Copy)]
enum BufferStage {
    Vertex,
    Fragment,
}

/// Builds an autoreleased `MTLRenderPassDescriptor` for `pass`.
fn build_render_pass_descriptor(pass: &RenderPass) -> Result<objc::Id, MetalError> {
    let cls = objc::class(c"MTLRenderPassDescriptor");
    if cls.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `cls` is the documented class factory and the returned descriptor
    // is autoreleased into the caller's active pool.
    let descriptor = unsafe { objc::msg_send_id_0(cls, objc::sel(c"renderPassDescriptor")) };
    if descriptor.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor creation returned null".to_string(),
        ));
    }

    // SAFETY: `descriptor` is a live render pass descriptor.
    let colors = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    if colors.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor.colorAttachments is nil".to_string(),
        ));
    }
    // SAFETY: `colors` is the colour attachment array and index 0 exists.
    let color = unsafe { msg_send_id_usize(colors, objc::sel(c"objectAtIndexedSubscript:"), 0) };
    if color.is_null() {
        return Err(MetalError::InternalError(
            "colour attachment descriptor at index 0 is nil".to_string(),
        ));
    }
    configure_color_attachment(color, &pass.color);

    if let Some(depth) = &pass.depth {
        // SAFETY: `descriptor` is a live render pass descriptor.
        let depth_attachment =
            unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"depthAttachment")) };
        if depth_attachment.is_null() {
            return Err(MetalError::InternalError(
                "MTLRenderPassDescriptor.depthAttachment is nil".to_string(),
            ));
        }
        configure_depth_attachment(depth_attachment, depth);
    }

    Ok(descriptor)
}

/// Applies texture/load/store/clear state to an `MTLRenderPassColorAttachmentDescriptor`.
fn configure_color_attachment(raw: objc::Id, attachment: &RenderAttachment) {
    let clear = MTLClearColor {
        red: attachment.clear_color[0],
        green: attachment.clear_color[1],
        blue: attachment.clear_color[2],
        alpha: attachment.clear_color[3],
    };
    // SAFETY: `raw` is a live colour attachment descriptor, the texture is a
    // retained render target, and the selectors take the documented value types.
    unsafe {
        objc::msg_send_void_1(raw, objc::sel(c"setTexture:"), attachment.target.as_raw());
        msg_send_void_usize(
            raw,
            objc::sel(c"setLoadAction:"),
            attachment.load_action.mtl_value(),
        );
        msg_send_void_usize(
            raw,
            objc::sel(c"setStoreAction:"),
            attachment.store_action.mtl_value(),
        );
        msg_send_void_clear_color(raw, objc::sel(c"setClearColor:"), clear);
    }
}

/// Applies texture/load/store/clear state to an `MTLRenderPassDepthAttachmentDescriptor`.
fn configure_depth_attachment(raw: objc::Id, attachment: &RenderAttachment) {
    // SAFETY: `raw` is a live depth attachment descriptor, the texture is a
    // retained depth render target, and the selectors take documented values.
    unsafe {
        objc::msg_send_void_1(raw, objc::sel(c"setTexture:"), attachment.target.as_raw());
        msg_send_void_usize(
            raw,
            objc::sel(c"setLoadAction:"),
            attachment.load_action.mtl_value(),
        );
        msg_send_void_usize(
            raw,
            objc::sel(c"setStoreAction:"),
            attachment.store_action.mtl_value(),
        );
        msg_send_void_f64(raw, objc::sel(c"setClearDepth:"), attachment.clear_depth);
    }
}

/// Creates an autoreleased `MTLTextureDescriptor` for a non-mipmapped 2D texture.
fn texture_2d_descriptor(
    pixel_format: usize,
    width: usize,
    height: usize,
) -> Result<objc::Id, MetalError> {
    let cls = objc::class(c"MTLTextureDescriptor");
    if cls.is_null() {
        return Err(MetalError::InternalError(
            "MTLTextureDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `cls` is the documented class factory and `mipmapped` receives
    // `NO` as a `BOOL` byte.
    let descriptor = unsafe {
        msg_send_id_texture_descriptor(
            cls,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            pixel_format,
            width,
            height,
            0,
        )
    };
    if descriptor.is_null() {
        return Err(MetalError::InternalError(
            "MTLTextureDescriptor creation returned null".to_string(),
        ));
    }
    Ok(descriptor)
}

/// Copies a level-0 2D texture into a CPU-visible buffer and returns its bytes.
#[cfg(test)]
fn read_back_texture(transport: &DirectMetalTransport, target: &RenderTarget) -> Vec<u8> {
    let width = target.width as usize;
    let height = target.height as usize;
    let bytes_per_row = width * 4;
    let len = bytes_per_row * height;

    let _pool = AutoreleasePool::new();
    // SAFETY: `device` is a valid `MTLDevice`; storage mode `0` is shared, so
    // the buffer is CPU-visible, and `newBufferWithLength:options:` returns a
    // retained buffer.
    let buffer_raw = unsafe { objc::msg_send_new_buffer(transport.device_raw(), len, 0) };
    let buffer = ObjcObject::from_retained(buffer_raw)
        .expect("readback MTLBuffer allocation should succeed");

    // SAFETY: `queue` is a valid `MTLCommandQueue`; the command buffer is
    // autoreleased and stays alive for this pool.
    let cb = unsafe { objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer")) };
    assert!(!cb.is_null(), "readback command buffer should exist");
    // SAFETY: `cb` is a live uncommitted command buffer.
    let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder")) };
    assert!(!enc.is_null(), "readback blit encoder should exist");

    // SAFETY: `enc` is a live blit encoder, `target` is a live texture and
    // `buffer` can hold exactly one level-0 2D image with this row pitch.
    unsafe {
        msg_send_copy_texture_to_buffer(
            enc,
            target.as_raw(),
            MTLOrigin { x: 0, y: 0, z: 0 },
            objc::MTLSize {
                width,
                height,
                depth: 1,
            },
            buffer.as_raw(),
            0,
            bytes_per_row,
        );
        objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
        objc::msg_send_void_0(cb, objc::sel(c"commit"));
        objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
    }

    // SAFETY: `cb` completed via `waitUntilCompleted`.
    let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
    assert_eq!(
        status, MTL_COMMAND_BUFFER_STATUS_COMPLETED,
        "readback blit did not complete"
    );

    // SAFETY: `buffer` uses shared storage and `contents` points to `len`
    // readable bytes while the buffer is alive.
    let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
    assert!(!contents.is_null(), "readback buffer has no contents");
    // SAFETY: `contents` is valid for `len` bytes for the buffer's lifetime.
    let bytes = unsafe { std::slice::from_raw_parts(contents.cast::<u8>(), len) };
    bytes.to_vec()
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
}

unsafe fn msg_send_id_usize(target: objc::Id, s: objc::Sel, value: usize) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize) -> objc::Id;
    // SAFETY: Caller supplies a valid receiver/selector returning an object
    // with one `NSUInteger` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, value) }
}

unsafe fn msg_send_void_usize(target: objc::Id, s: objc::Sel, value: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // one `NSUInteger` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, value) }
}

unsafe fn msg_send_void_f64(target: objc::Id, s: objc::Sel, value: f64) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, f64);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // one `double` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, value) }
}

unsafe fn msg_send_void_buffer(
    target: objc::Id,
    s: objc::Sel,
    buffer: objc::Id,
    offset: usize,
    index: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, usize);
    // SAFETY: Caller supplies a valid receiver/selector and a live `MTLBuffer`.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, buffer, offset, index) }
}

unsafe fn msg_send_void_bytes(
    target: objc::Id,
    s: objc::Sel,
    bytes: *const u8,
    len: usize,
    index: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, *const c_void, usize, usize);
    // SAFETY: Caller guarantees `bytes` is valid for `len` bytes during the call.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout; Metal
    // copies the payload before returning.
    unsafe { f(target, s, bytes.cast(), len, index) }
}

unsafe fn msg_send_void_two_usize(
    target: objc::Id,
    s: objc::Sel,
    first: usize,
    second: usize,
    third: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // three `NSUInteger` arguments.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, first, second, third) }
}

#[allow(clippy::too_many_arguments)]
unsafe fn msg_send_draw_indexed(
    target: objc::Id,
    s: objc::Sel,
    primitive: usize,
    index_count: usize,
    index_type: usize,
    index_buffer: objc::Id,
    index_buffer_offset: usize,
    instance_count: usize,
    base_vertex: isize,
    base_instance: usize,
) {
    type Fn = unsafe extern "C" fn(
        objc::Id,
        objc::Sel,
        usize,
        usize,
        usize,
        objc::Id,
        usize,
        usize,
        isize,
        usize,
    );
    // SAFETY: Caller supplies a valid receiver/selector and a live index buffer.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            target,
            s,
            primitive,
            index_count,
            index_type,
            index_buffer,
            index_buffer_offset,
            instance_count,
            base_vertex,
            base_instance,
        )
    }
}

unsafe fn msg_send_void_clear_color(target: objc::Id, s: objc::Sel, clear: MTLClearColor) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLClearColor);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // one `MTLClearColor` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, clear) }
}

unsafe fn msg_send_void_viewport(target: objc::Id, s: objc::Sel, viewport: MTLViewport) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLViewport);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // one `MTLViewport` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, viewport) }
}

unsafe fn msg_send_void_scissor(target: objc::Id, s: objc::Sel, rect: MTLScissorRect) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLScissorRect);
    // SAFETY: Caller supplies a valid receiver/selector returning `void` with
    // one `MTLScissorRect` argument.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, rect) }
}

unsafe fn msg_send_id_texture_descriptor(
    cls: objc::Id,
    s: objc::Sel,
    pixel_format: usize,
    width: usize,
    height: usize,
    mipmapped: u8,
) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
    // SAFETY: Caller supplies the `MTLTextureDescriptor` class and the
    // documented factory selector.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(cls, s, pixel_format, width, height, mipmapped) }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
unsafe fn msg_send_copy_texture_to_buffer(
    enc: objc::Id,
    texture: objc::Id,
    origin: MTLOrigin,
    size: objc::MTLSize,
    buffer: objc::Id,
    buffer_offset: usize,
    bytes_per_row: usize,
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
        usize,
    );
    // SAFETY: Caller supplies a live blit encoder and live Metal objects. The
    // per-image stride is `0` for a single level-0 image.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            enc,
            objc::sel(c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"),
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            buffer_offset,
            bytes_per_row,
            0,
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::transport::MetalTransport;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const PIXEL_COUNT: usize = (WIDTH * HEIGHT) as usize;

    /// Full-screen triangle driven entirely by inline root constants: three
    /// positions in vertex bytes and the fragment colour in fragment bytes.
    const FULLSCREEN_MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VOut {
            float4 position [[position]];
        };

        vertex VOut vm_fullscreen_vertex(uint vid [[vertex_id]],
                                         constant float2 *positions [[buffer(0)]]) {
            VOut out;
            out.position = float4(positions[vid], 0.0, 1.0);
            return out;
        }

        fragment float4 vm_fullscreen_fragment(constant float4 &color [[buffer(0)]]) {
            return color;
        }
    "#;

    /// Indexed draw with a non-zero base vertex: the index buffer addresses the
    /// first three slots, which `baseVertex` shifts onto the real triangle.
    const INDEXED_MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VOut {
            float4 position [[position]];
        };

        vertex VOut vm_indexed_vertex(uint vid [[vertex_id]],
                                      constant float2 *positions [[buffer(0)]]) {
            VOut out;
            out.position = float4(positions[vid], 0.0, 1.0);
            return out;
        }

        fragment float4 vm_indexed_fragment() {
            return float4(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    fn open_transport() -> DirectMetalTransport {
        DirectMetalTransport::new().expect("Metal device should open")
    }

    fn color_target(transport: &DirectMetalTransport) -> RenderTarget {
        transport
            .create_render_target_2d(WIDTH, HEIGHT, PixelFormat::Bgra8Unorm)
            .expect("BGRA8 render target should be created")
    }

    fn as_bytes<T>(values: &[T]) -> &[u8] {
        // SAFETY: Only `f32` and `u16` arrays are passed here; both are plain
        // data with no padding, so the byte view covers exactly the values.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    /// Allocates a shared-storage `MTLBuffer` preloaded with `bytes`.
    fn shared_buffer(transport: &DirectMetalTransport, bytes: &[u8]) -> ObjcObject {
        // SAFETY: `device` is valid; storage mode `0` is shared and the length
        // is at least one byte, so Metal returns a retained CPU-visible buffer.
        let raw =
            unsafe { objc::msg_send_new_buffer(transport.device_raw(), bytes.len().max(4), 0) };
        let buffer = ObjcObject::from_retained(raw).expect("shared MTLBuffer allocation");
        // SAFETY: `buffer` is a live shared-storage buffer with `contents`
        // pointing at least `bytes.len()` writable bytes.
        let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
        assert!(!contents.is_null(), "shared buffer has no contents");
        // SAFETY: `contents` is valid for `bytes.len()` bytes and does not
        // overlap `bytes`, which is a Rust slice.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len());
        }
        buffer
    }

    /// Reads every pixel back as BGRA8 quads.
    fn read_back(transport: &DirectMetalTransport, target: &RenderTarget) -> Vec<[u8; 4]> {
        let bytes = read_back_texture(transport, target);
        assert_eq!(bytes.len(), PIXEL_COUNT * 4, "readback byte count mismatch");
        bytes.as_chunks::<4>().0.to_vec()
    }

    #[test]
    fn full_screen_triangle_writes_expected_bgra_pixels() {
        const GREEN_BGRA: [u8; 4] = [0, 255, 0, 255];

        let transport = open_transport();
        let color = color_target(&transport);
        let pass =
            RenderPass::new(RenderAttachment::color(color).with_clear_color([0.0, 0.0, 0.0, 1.0]))
                .expect("colour-only render pass should validate");
        let pipeline = GraphicsPipeline::compile(
            &transport,
            FULLSCREEN_MSL,
            "vm_fullscreen_vertex",
            "vm_fullscreen_fragment",
            PixelFormat::Bgra8Unorm,
            None,
        )
        .expect("full-screen pipeline should compile");

        let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let green: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

        let mut encoder =
            RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder should be created");
        encoder
            .set_viewport(Viewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: WIDTH as f64,
                height: HEIGHT as f64,
                znear: 0.0,
                zfar: 1.0,
            })
            .expect("viewport should be valid");
        encoder
            .set_scissor_rect(ScissorRect {
                x: 0,
                y: 0,
                width: WIDTH,
                height: HEIGHT,
            })
            .expect("scissor should be valid");
        encoder
            .set_vertex_bytes(as_bytes(&positions), 0)
            .expect("vertex root constants should bind");
        encoder
            .set_fragment_bytes(as_bytes(&green), 0)
            .expect("fragment root constants should bind");
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3)
            .expect("full-screen draw should encode");
        encoder.finish().expect("render pass should complete");

        let pixels = read_back(&transport, pass.color_target());
        let mut mismatches = 0usize;
        let mut first_bad = None;
        for (index, pixel) in pixels.iter().enumerate() {
            if *pixel != GREEN_BGRA {
                mismatches += 1;
                first_bad.get_or_insert((index, *pixel));
            }
        }
        assert_eq!(
            mismatches, 0,
            "expected all {PIXEL_COUNT} pixels green, first mismatch at {first_bad:?}"
        );

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "render full-screen triangle: device={}, pixels={PIXEL_COUNT}, expected={GREEN_BGRA:?}",
            device.name_str()
        );
    }

    #[test]
    fn indexed_draw_base_vertex_hits_centre_not_corner() {
        const RED_BGRA: [u8; 4] = [0, 0, 255, 255];
        const BLACK_BGRA: [u8; 4] = [0, 0, 0, 255];
        const CENTRE: usize = (HEIGHT / 2 * WIDTH + WIDTH / 2) as usize;
        const CORNER: usize = 0;

        let transport = open_transport();
        let color = color_target(&transport);
        let pass = RenderPass::new(RenderAttachment::color(color))
            .expect("colour-only render pass should validate");
        let pipeline = GraphicsPipeline::compile(
            &transport,
            INDEXED_MSL,
            "vm_indexed_vertex",
            "vm_indexed_fragment",
            PixelFormat::Bgra8Unorm,
            None,
        )
        .expect("indexed pipeline should compile");

        // Slot 0 is deliberately unused; indices [0,1,2] + baseVertex 1 select
        // slots 1..=3, forming a triangle that reaches the top-centre and covers
        // the middle pixel while leaving the top-left corner cleared.
        let vertices: [f32; 8] = [0.0, 0.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0];
        let indices: [u16; 3] = [0, 1, 2];
        let vertex_buffer = shared_buffer(&transport, as_bytes(&vertices));
        let index_buffer = shared_buffer(&transport, as_bytes(&indices));

        let mut encoder =
            RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder should be created");
        // SAFETY: Both buffers are live for the whole `encoder` scope, which is
        // consumed by `finish` before either `ObjcObject` drops.
        unsafe {
            encoder
                .set_vertex_buffer(vertex_buffer.as_raw(), 0, 0)
                .expect("vertex buffer should bind");
            encoder
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    indices.len(),
                    IndexType::Uint16,
                    index_buffer.as_raw(),
                    0,
                    1,
                )
                .expect("indexed draw should encode");
        }
        encoder
            .finish()
            .expect("indexed render pass should complete");

        let pixels = read_back(&transport, pass.color_target());
        assert_eq!(
            pixels[CENTRE], RED_BGRA,
            "centre pixel should be covered by the indexed triangle"
        );
        assert_eq!(
            pixels[CORNER], BLACK_BGRA,
            "top-left corner should remain cleared"
        );
        assert_ne!(
            pixels[CENTRE], pixels[CORNER],
            "covered and uncovered pixels must differ"
        );

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "render indexed draw: device={}, centre={:?}, corner={:?}, base_vertex=1",
            device.name_str(),
            pixels[CENTRE],
            pixels[CORNER]
        );
    }

    #[test]
    fn invalid_msl_reports_compile_failed() {
        const BROKEN_MSL: &str = r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex void broken_vertex() { this is not MSL }
        "#;

        let transport = open_transport();
        let error = GraphicsPipeline::compile(
            &transport,
            BROKEN_MSL,
            "broken_vertex",
            "broken_fragment",
            PixelFormat::Bgra8Unorm,
            None,
        )
        .err()
        .expect("invalid MSL must not produce a pipeline");
        match &error {
            MetalError::CompileFailed(message) => {
                assert!(!message.is_empty(), "Metal diagnostic should not be empty");
            }
            other => panic!("expected CompileFailed, got {other:?}"),
        }
        eprintln!("render invalid MSL rejected: {error}");
    }

    #[test]
    fn invalid_render_arguments_rejected_without_command_buffer() {
        let transport = open_transport();

        // Zero-size attachment is rejected before any texture exists.
        let zero_size = transport
            .create_render_target_2d(0, HEIGHT, PixelFormat::Bgra8Unorm)
            .err()
            .expect("zero-size attachment must be rejected");
        assert!(
            matches!(zero_size, MetalError::InvalidArg(_)),
            "unexpected zero-size error: {zero_size:?}"
        );

        // A depth target of a different size must not be attachable.
        let color = color_target(&transport);
        let small_depth = transport
            .create_render_target_2d(WIDTH / 2, HEIGHT / 2, PixelFormat::Depth32Float)
            .expect("depth target should be created");
        let pass = RenderPass::new(RenderAttachment::color(color))
            .expect("colour-only render pass should validate");
        let mismatch = pass
            .with_depth(RenderAttachment::depth(small_depth, 1.0))
            .err()
            .expect("mismatched depth attachment must be rejected");
        assert!(
            matches!(mismatch, MetalError::InvalidArg(_)),
            "unexpected size-mismatch error: {mismatch:?}"
        );

        // Zero vertex count is rejected before the lazy command buffer exists.
        let color = color_target(&transport);
        let pass = RenderPass::new(RenderAttachment::color(color))
            .expect("colour-only render pass should validate");
        let pipeline = GraphicsPipeline::compile(
            &transport,
            FULLSCREEN_MSL,
            "vm_fullscreen_vertex",
            "vm_fullscreen_fragment",
            PixelFormat::Bgra8Unorm,
            None,
        )
        .expect("full-screen pipeline should compile");
        let mut encoder =
            RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder should be created");
        assert!(
            encoder.cb.is_null(),
            "encoder must start without a command buffer"
        );
        let zero_draw = encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 0)
            .expect_err("zero vertex count must be rejected");
        assert!(
            matches!(zero_draw, MetalError::InvalidArg(_)),
            "unexpected zero-vertex error: {zero_draw:?}"
        );
        assert!(
            encoder.cb.is_null(),
            "rejected draw must not create a command buffer"
        );
        encoder
            .finish()
            .expect("empty clear-only pass should complete");
    }

    #[test]
    fn depth_attachment_clear_and_draw_completes() {
        const GREEN_BGRA: [u8; 4] = [0, 255, 0, 255];

        let transport = open_transport();
        let color = color_target(&transport);
        let depth = transport
            .create_render_target_2d(WIDTH, HEIGHT, PixelFormat::Depth32Float)
            .expect("Depth32Float target should be created");
        assert_eq!(depth.pixel_format(), PixelFormat::Depth32Float);
        let pass = RenderPass::new(RenderAttachment::color(color))
            .expect("colour-only render pass should validate")
            .with_depth(RenderAttachment::depth(depth, 1.0).with_clear_depth(1.0))
            .expect("matching depth attachment should validate");

        let pipeline = GraphicsPipeline::compile(
            &transport,
            FULLSCREEN_MSL,
            "vm_fullscreen_vertex",
            "vm_fullscreen_fragment",
            PixelFormat::Bgra8Unorm,
            Some(PixelFormat::Depth32Float),
        )
        .expect("depth-enabled pipeline should compile");

        let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let green: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        let mut encoder =
            RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder should be created");
        encoder
            .set_vertex_bytes(as_bytes(&positions), 0)
            .expect("vertex root constants should bind");
        encoder
            .set_fragment_bytes(as_bytes(&green), 0)
            .expect("fragment root constants should bind");
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3)
            .expect("depth pass draw should encode");
        encoder
            .finish()
            .expect("pass with depth clear 1.0 should complete");

        let pixels = read_back(&transport, pass.color_target());
        assert!(
            pixels.iter().all(|pixel| *pixel == GREEN_BGRA),
            "depth-enabled pass should still write the expected colour"
        );
        assert_eq!(
            pass.depth_target().map(RenderTarget::pixel_format),
            Some(PixelFormat::Depth32Float)
        );

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "render depth pass: device={}, depth_clear=1.0, pixels={PIXEL_COUNT}",
            device.name_str()
        );
    }
}
