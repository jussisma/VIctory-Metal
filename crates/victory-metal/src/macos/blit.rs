//! Texture region copies and subresource resolves (ADR-015).
//!
//! This module owns `copyFromTexture` with an explicit source/destination
//! subresource plus origin/size box, and multisample resolve into a
//! single-sample target.
//!
//! # Resolve selector
//!
//! Metal has no `resolveTexture:sourceTexture:` message on
//! `MTLBlitCommandEncoder` (searched `MTLBlitCommandEncoder.h`); multisample
//! resolve is a render-pass store action. [`encode_resolve`] therefore builds a
//! one-attachment `MTLRenderPassDescriptor` whose colour attachment is the
//! multisample source, sets `resolveTexture` to the destination and
//! `MTLStoreActionStoreAndMultisampleResolve`, and opens/closes an empty render
//! encoder on the caller's command buffer. The result is the same GPU resolve a
//! blit encoder would perform, without a CPU readback.
//!
//! All sizes are texels. Every box is checked against the live texture geometry
//! (`width`, `height`, `depth`, `mipmapLevelCount`) before a selector is sent.

use std::sync::{Mutex, OnceLock};

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::render::{PixelFormat, RenderTarget};
use crate::transport::MetalError;

/// Largest number of distinct destination textures the bounded copy trace
/// remembers. A title can create thousands of textures; the trace keeps the
/// first 16 it sees and counts the rest as dropped, so memory is fixed.
pub(crate) const COPY_TRACE_MAX_KEYS: usize = 16;

/// The copy operation a host blit encoder ran into a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlitCopyKind {
    /// `MTLBlitCommandEncoder` texture-to-texture region copy. Guest
    /// `CopyResource` lowers to this once its subresources are expanded.
    CopyTextureRegion,
    /// Multisample resolve through a render pass store action.
    ResolveSubresource,
}

impl BlitCopyKind {
    #[cfg(test)]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CopyTextureRegion => "copy_texture_region",
            Self::ResolveSubresource => "resolve_subresource",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CopyTraceCount {
    copy_texture_region: u64,
    resolve_subresource: u64,
    last: Option<BlitCopyKind>,
}

#[derive(Debug, Default)]
struct CopyTrace {
    keys: Vec<(usize, CopyTraceCount)>,
    dropped_keys: u64,
}

static COPY_TRACE: OnceLock<Mutex<CopyTrace>> = OnceLock::new();

fn copy_trace() -> &'static Mutex<CopyTrace> {
    COPY_TRACE.get_or_init(|| Mutex::new(CopyTrace::default()))
}

/// Records one host copy into `texture`.
///
/// Counts only: the key is the texture's object address, which is process-local
/// and names no title asset. The trace is bounded to
/// [`COPY_TRACE_MAX_KEYS`] distinct destinations.
pub(crate) fn note_copy_destination(kind: BlitCopyKind, texture: objc::Id) {
    if texture.is_null() {
        return;
    }
    let key = texture as usize;
    let Ok(mut trace) = copy_trace().lock() else {
        return;
    };
    let entry = match trace.keys.iter_mut().find(|(seen, _)| *seen == key) {
        Some((_, counts)) => counts,
        None => {
            if trace.keys.len() >= COPY_TRACE_MAX_KEYS {
                trace.dropped_keys = trace.dropped_keys.saturating_add(1);
                return;
            }
            trace.keys.push((key, CopyTraceCount::default()));
            &mut trace.keys.last_mut().expect("just pushed").1
        }
    };
    match kind {
        BlitCopyKind::CopyTextureRegion => {
            entry.copy_texture_region = entry.copy_texture_region.saturating_add(1);
        }
        BlitCopyKind::ResolveSubresource => {
            entry.resolve_subresource = entry.resolve_subresource.saturating_add(1);
        }
    }
    entry.last = Some(kind);
}

/// The most recent host copy kind into `texture`, if any.
pub(crate) fn copy_kind_for(texture: objc::Id) -> Option<BlitCopyKind> {
    if texture.is_null() {
        return None;
    }
    let key = texture as usize;
    let trace = copy_trace().lock().ok()?;
    trace
        .keys
        .iter()
        .find(|(seen, _)| *seen == key)
        .and_then(|(_, counts)| counts.last)
}

/// Total host copies (region and resolve) into `texture`.
pub(crate) fn copy_count_for(texture: objc::Id) -> u64 {
    if texture.is_null() {
        return 0;
    }
    let key = texture as usize;
    let Ok(trace) = copy_trace().lock() else {
        return 0;
    };
    trace
        .keys
        .iter()
        .find(|(seen, _)| *seen == key)
        .map_or(0, |(_, counts)| {
            counts
                .copy_texture_region
                .saturating_add(counts.resolve_subresource)
        })
}

/// `(copy_texture_region, resolve_subresource, dropped_keys)` totals.
#[cfg(test)]
pub(crate) fn copy_trace_totals() -> (u64, u64, u64) {
    let Ok(trace) = copy_trace().lock() else {
        return (0, 0, 0);
    };
    let mut region = 0u64;
    let mut resolve = 0u64;
    for (_, counts) in &trace.keys {
        region = region.saturating_add(counts.copy_texture_region);
        resolve = resolve.saturating_add(counts.resolve_subresource);
    }
    (region, resolve, trace.dropped_keys)
}

/// Clears the bounded copy trace (tests and a fresh session).
#[cfg(test)]
/// Clears the copy trace for an integration test's identity gate.
///
/// Public and hidden so a hardware case starts with no earlier copy; the host
/// never calls it.
#[doc(hidden)]
pub fn reset_copy_trace_for_test() {
    reset_copy_trace();
}

pub(crate) fn reset_copy_trace() {
    if let Ok(mut trace) = copy_trace().lock() {
        *trace = CopyTrace::default();
    }
}

/// `MTLLoadActionLoad`.
const MTL_LOAD_ACTION_LOAD: usize = 1;
/// `MTLStoreActionStoreAndMultisampleResolve`.
const MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE: usize = 3;
/// `MTLTextureType2DMultisample`.
const MTL_TEXTURE_TYPE_2D_MULTISAMPLE: usize = 4;
/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: usize = 2;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLTextureUsageRenderTarget`.
const MTL_TEXTURE_USAGE_RENDER_TARGET: usize = 4;
/// Largest supported level-0 texture edge.
const MAX_TEXTURE_DIMENSION: u32 = 16_384;

/// Apple `MTLOrigin` (`NSUInteger x, y, z`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// One texture and the subresource a copy or resolve addresses.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextureSubresourceRef {
    /// Registry-held `MTLTexture`.
    pub(crate) texture: objc::Id,
    /// Array slice.
    pub(crate) slice: usize,
    /// Mip level.
    pub(crate) level: usize,
}

/// One explicit texel-box copy between two subresources.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextureRegion {
    /// Source subresource.
    pub(crate) source: TextureSubresourceRef,
    /// Source box origin in texels.
    pub(crate) source_origin: [usize; 3],
    /// Destination subresource.
    pub(crate) destination: TextureSubresourceRef,
    /// Destination box origin in texels.
    pub(crate) destination_origin: [usize; 3],
    /// Box size in texels; every component nonzero.
    pub(crate) size: [usize; 3],
}

impl TextureRegion {
    /// Validates the box against both live textures.
    pub(crate) fn validate(&self) -> Result<(), MetalError> {
        validate_box("source", self.source, self.source_origin, self.size)?;
        validate_box(
            "destination",
            self.destination,
            self.destination_origin,
            self.size,
        )
    }
}

/// Geometry and subresource counts of one live `MTLTexture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextureGeometry {
    width: usize,
    height: usize,
    depth: usize,
    mip_levels: usize,
    slices: usize,
    pixel_format: usize,
    sample_count: usize,
}

fn texture_geometry(texture: objc::Id) -> Result<TextureGeometry, MetalError> {
    if texture.is_null() {
        return Err(MetalError::InvalidArg("texture is nil"));
    }
    // SAFETY: `texture` is a live `MTLTexture` and these accessors only read
    // immutable texture properties.
    let (width, height, depth, mip_levels, slices, pixel_format, sample_count) = unsafe {
        (
            objc::msg_send_usize_0(texture, objc::sel(c"width")),
            objc::msg_send_usize_0(texture, objc::sel(c"height")),
            objc::msg_send_usize_0(texture, objc::sel(c"depth")),
            objc::msg_send_usize_0(texture, objc::sel(c"mipmapLevelCount")),
            objc::msg_send_usize_0(texture, objc::sel(c"arrayLength")),
            objc::msg_send_usize_0(texture, objc::sel(c"pixelFormat")),
            objc::msg_send_usize_0(texture, objc::sel(c"sampleCount")),
        )
    };
    if width == 0 || height == 0 {
        return Err(MetalError::InvalidArg(
            "texture width and height must be nonzero",
        ));
    }
    if mip_levels == 0 || slices == 0 || sample_count == 0 {
        return Err(MetalError::InvalidArg(
            "texture mip, slice and sample counts must be nonzero",
        ));
    }
    Ok(TextureGeometry {
        width,
        height,
        depth: depth.max(1),
        mip_levels,
        slices,
        pixel_format,
        sample_count,
    })
}

/// Level dimensions, halved per mip level and floored at one texel.
fn level_extent(geometry: &TextureGeometry, level: usize) -> [usize; 3] {
    let shift = u32::try_from(level).unwrap_or(u32::MAX);
    let half = |value: usize| value.checked_shr(shift).unwrap_or(0).max(1);
    [
        half(geometry.width),
        half(geometry.height),
        half(geometry.depth),
    ]
}

fn validate_subresource(
    field: &'static str,
    subresource: TextureSubresourceRef,
) -> Result<TextureGeometry, MetalError> {
    let geometry = texture_geometry(subresource.texture)?;
    if subresource.level >= geometry.mip_levels {
        return Err(MetalError::InvalidArg(match field {
            "source" => "source mip level is out of range",
            _ => "destination mip level is out of range",
        }));
    }
    if subresource.slice >= geometry.slices {
        return Err(MetalError::InvalidArg(match field {
            "source" => "source slice is out of range",
            _ => "destination slice is out of range",
        }));
    }
    Ok(geometry)
}

fn validate_box(
    field: &'static str,
    subresource: TextureSubresourceRef,
    origin: [usize; 3],
    size: [usize; 3],
) -> Result<(), MetalError> {
    if size.contains(&0) {
        return Err(MetalError::InvalidArg(
            "copy region width, height and depth must be nonzero",
        ));
    }
    let geometry = validate_subresource(field, subresource)?;
    let extent = level_extent(&geometry, subresource.level);
    for axis in 0..3 {
        let end = origin[axis]
            .checked_add(size[axis])
            .ok_or(MetalError::InvalidArg("copy region box overflows usize"))?;
        if end > extent[axis] {
            return Err(MetalError::InvalidArg(match (field, axis) {
                ("source", 0) => "copy region exceeds the source width",
                ("source", 1) => "copy region exceeds the source height",
                ("source", _) => "copy region exceeds the source depth",
                (_, 0) => "copy region exceeds the destination width",
                (_, 1) => "copy region exceeds the destination height",
                _ => "copy region exceeds the destination depth",
            }));
        }
    }
    Ok(())
}

/// Copies one explicit texel box between two textures on a live blit encoder.
pub(crate) fn copy_texture_region(enc: objc::Id, region: &TextureRegion) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError("blit encoder is nil".to_string()));
    }
    region.validate()?;
    note_copy_destination(BlitCopyKind::CopyTextureRegion, region.destination.texture);
    // The copy destination joins the present path's object identity census, so
    // a surface, pass target and copy target of one guest texture are proven to
    // be the same `MTLTexture`.
    crate::macos::present::note_copy_destination_object(region.destination.texture);
    let origin = |values: [usize; 3]| MTLOrigin {
        x: values[0],
        y: values[1],
        z: values[2],
    };
    let size = MTLSize {
        width: region.size[0],
        height: region.size[1],
        depth: region.size[2],
    };
    // SAFETY: `enc` is a live blit encoder, both textures are registry-held,
    // and every level/slice/origin/size was checked against live geometry.
    unsafe {
        type Fn = unsafe extern "C" fn(
            objc::Id,
            objc::Sel,
            objc::Id,
            usize,
            usize,
            MTLOrigin,
            MTLSize,
            objc::Id,
            usize,
            usize,
            MTLOrigin,
        );
        let send: Fn = std::mem::transmute(objc::msg_send_entry());
        send(
            enc,
            objc::sel(c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
            region.source.texture,
            region.source.slice,
            region.source.level,
            origin(region.source_origin),
            size,
            region.destination.texture,
            region.destination.slice,
            region.destination.level,
            origin(region.destination_origin),
        );
    }
    Ok(())
}

/// One pitched upload-heap to private-texture copy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BufferTextureCopy {
    /// Registry-held source `MTLBuffer`.
    pub(crate) buffer: objc::Id,
    /// Byte offset of the first source texel.
    pub(crate) source_offset: usize,
    /// Source row pitch in bytes.
    pub(crate) bytes_per_row: usize,
    /// Source slice pitch in bytes; zero for a single-slice copy.
    pub(crate) bytes_per_image: usize,
    /// Destination subresource.
    pub(crate) destination: TextureSubresourceRef,
    /// Destination box origin in texels.
    pub(crate) destination_origin: [usize; 3],
    /// Box size in texels; every component nonzero.
    pub(crate) size: [usize; 3],
}

impl BufferTextureCopy {
    /// Validates the copy against the live destination texture geometry.
    pub(crate) fn validate(&self) -> Result<(), MetalError> {
        if self.buffer.is_null() {
            return Err(MetalError::InvalidArg("copy source buffer is nil"));
        }
        if self.bytes_per_row == 0 {
            return Err(MetalError::InvalidArg(
                "copy source row pitch must be nonzero",
            ));
        }
        if self.size.contains(&0) {
            return Err(MetalError::InvalidArg(
                "copy width, height and depth must be nonzero",
            ));
        }
        let rows = self.size[1];
        let row_bytes = self
            .bytes_per_row
            .checked_mul(rows)
            .ok_or(MetalError::InvalidArg("copy source row pitch overflows"))?;
        if self.size[2] > 1 {
            if self.bytes_per_image == 0 {
                return Err(MetalError::InvalidArg(
                    "copy source image pitch must be nonzero for a volume copy",
                ));
            }
            if self.bytes_per_image < row_bytes {
                return Err(MetalError::InvalidArg(
                    "copy source image pitch is smaller than one image",
                ));
            }
        } else if self.bytes_per_image != 0 && self.bytes_per_image < row_bytes {
            return Err(MetalError::InvalidArg(
                "copy source image pitch is smaller than one image",
            ));
        }
        validate_box(
            "destination",
            self.destination,
            self.destination_origin,
            self.size,
        )
    }
}

/// Copies one pitched buffer image into a texture subresource.
///
/// The row pitch is supplied by the caller and validated against the live
/// destination extent by [`BufferTextureCopy::validate`], so a stride mismatch
/// is an explicit error before any encoder call.
pub(crate) fn copy_buffer_to_texture(
    enc: objc::Id,
    copy: &BufferTextureCopy,
) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError("blit encoder is nil".to_string()));
    }
    copy.validate()?;
    crate::macos::present::note_copy_destination_object(copy.destination.texture);
    let origin = MTLOrigin {
        x: copy.destination_origin[0],
        y: copy.destination_origin[1],
        z: copy.destination_origin[2],
    };
    let size = MTLSize {
        width: copy.size[0],
        height: copy.size[1],
        depth: copy.size[2],
    };
    // SAFETY: `enc` is a live blit encoder, the buffer is registry-held for the
    // synchronous submission, the texture is registry-held, and every
    // level/slice/origin/size/pitch was checked before the selector is sent.
    unsafe {
        type Fn = unsafe extern "C" fn(
            objc::Id,
            objc::Sel,
            objc::Id,
            usize,
            usize,
            usize,
            MTLSize,
            objc::Id,
            usize,
            usize,
            MTLOrigin,
        );
        let send: Fn = std::mem::transmute(objc::msg_send_entry());
        send(
            enc,
            objc::sel(c"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
            copy.buffer,
            copy.source_offset,
            copy.bytes_per_row,
            copy.bytes_per_image,
            size,
            copy.destination.texture,
            copy.destination.slice,
            copy.destination.level,
            origin,
        );
    }
    Ok(())
}

/// Validates a multisample resolve against both live textures.
pub(crate) fn validate_resolve(
    source: TextureSubresourceRef,
    destination: TextureSubresourceRef,
) -> Result<(), MetalError> {
    let src = validate_subresource("source", source)?;
    let dst = validate_subresource("destination", destination)?;
    if src.sample_count <= 1 {
        return Err(MetalError::InvalidArg(
            "resolve source must be a multisample texture",
        ));
    }
    if dst.sample_count != 1 {
        return Err(MetalError::InvalidArg(
            "resolve destination must be a single-sample texture",
        ));
    }
    if src.pixel_format != dst.pixel_format {
        return Err(MetalError::InvalidArg(
            "resolve source and destination pixel formats must match",
        ));
    }
    if level_extent(&src, source.level) != level_extent(&dst, destination.level) {
        return Err(MetalError::InvalidArg(
            "resolve source and destination dimensions must match",
        ));
    }
    Ok(())
}

/// Resolves one multisample subresource into a single-sample subresource.
///
/// `command_buffer` is the caller's uncommitted `MTLCommandBuffer`; no encoder
/// may be open when this is called. The resolve is encoded as an empty render
/// pass with a `StoreAndMultisampleResolve` colour attachment.
pub(crate) fn encode_resolve(
    command_buffer: objc::Id,
    source: TextureSubresourceRef,
    destination: TextureSubresourceRef,
) -> Result<(), MetalError> {
    if command_buffer.is_null() {
        return Err(MetalError::InvalidArg("command buffer is nil"));
    }
    validate_resolve(source, destination)?;
    note_copy_destination(BlitCopyKind::ResolveSubresource, destination.texture);
    crate::macos::present::note_copy_destination_object(destination.texture);

    let _pool = AutoreleasePool::new();
    let cls = objc::class(c"MTLRenderPassDescriptor");
    if cls.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `cls` is the documented class factory and the descriptor is
    // autoreleased into the pool above.
    let descriptor = unsafe { objc::msg_send_id_0(cls, objc::sel(c"renderPassDescriptor")) };
    if descriptor.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor creation returned null".to_string(),
        ));
    }
    // SAFETY: `descriptor` is a live render pass descriptor.
    let attachments = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    if attachments.is_null() {
        return Err(MetalError::InternalError(
            "MTLRenderPassDescriptor.colorAttachments is nil".to_string(),
        ));
    }
    // SAFETY: index 0 is below the declared 1..=8 colour attachment limit.
    let attachment =
        unsafe { objc::msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0) };
    if attachment.is_null() {
        return Err(MetalError::InternalError(
            "colour attachment descriptor is nil".to_string(),
        ));
    }
    // SAFETY: both textures are registry-held and validated; the values are
    // documented `MTLLoadAction`/`MTLStoreAction` encodings and subresource
    // indices.
    unsafe {
        objc::msg_send_void_1(attachment, objc::sel(c"setTexture:"), source.texture);
        objc::msg_send_void_1(
            attachment,
            objc::sel(c"setResolveTexture:"),
            destination.texture,
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setLoadAction:"),
            MTL_LOAD_ACTION_LOAD,
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setStoreAction:"),
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
        );
        objc::msg_send_void_usize(attachment, objc::sel(c"setLevel:"), source.level);
        objc::msg_send_void_usize(attachment, objc::sel(c"setSlice:"), source.slice);
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setResolveLevel:"),
            destination.level,
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setResolveSlice:"),
            destination.slice,
        );
    }

    // SAFETY: `command_buffer` is live and uncommitted with no open encoder;
    // `descriptor` is live for this pool.
    let encoder = unsafe {
        objc::msg_send_id_1(
            command_buffer,
            objc::sel(c"renderCommandEncoderWithDescriptor:"),
            descriptor,
        )
    };
    if encoder.is_null() {
        return Err(MetalError::InternalError(
            "MTLCommandBuffer.renderCommandEncoderWithDescriptor returned null".to_string(),
        ));
    }
    // SAFETY: `encoder` is a live render encoder that recorded only the resolve
    // store action.
    unsafe { objc::msg_send_void_0(encoder, objc::sel(c"endEncoding")) };
    Ok(())
}

/// Creates a private multisample colour target for resolve proofs.
///
/// Metal resolves multisample content through a render-pass store action, so
/// producing a multisample source needs a texture type the single-sample
/// [`DirectMetalTransport::create_render_target_2d`] cannot create. This helper
/// keeps that creation in the blit module and out of production paths.
pub fn create_multisample_render_target(
    transport: &DirectMetalTransport,
    width: u32,
    height: u32,
    format: PixelFormat,
    sample_count: usize,
) -> Result<RenderTarget, MetalError> {
    if width == 0 || height == 0 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION
    {
        return Err(MetalError::InvalidArg(
            "multisample target dimensions are zero or oversized",
        ));
    }
    if !format.is_color() {
        return Err(MetalError::InvalidArg(
            "multisample target must use a colour pixel format",
        ));
    }
    if !matches!(sample_count, 2 | 4 | 8) {
        return Err(MetalError::InvalidArg(
            "multisample target sample count must be 2, 4 or 8",
        ));
    }

    let _pool = AutoreleasePool::new();
    let cls = objc::class(c"MTLTextureDescriptor");
    if cls.is_null() {
        return Err(MetalError::InternalError(
            "MTLTextureDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `cls` is the documented class; the selector takes documented
    // `MTLPixelFormat`, `NSUInteger` and `BOOL` arguments.
    let described = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let send: Fn = std::mem::transmute(objc::msg_send_entry());
        send(
            cls,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            format.mtl_pixel_format(),
            width as usize,
            height as usize,
            0,
        )
    };
    // The class factory returns an autoreleased descriptor; it is owned by the
    // pool above, so it must not be adopted as a retained object (doing so
    // over-releases it when the pool drains and crashes).
    if described.is_null() {
        return Err(MetalError::InternalError(
            "MTLTextureDescriptor creation returned null".to_string(),
        ));
    }
    let descriptor = described;
    // SAFETY: `descriptor` is a live texture descriptor and each value is a
    // documented `MTLTextureType`/sample count/storage/usage encoding.
    unsafe {
        objc::msg_send_void_usize(
            descriptor,
            objc::sel(c"setTextureType:"),
            MTL_TEXTURE_TYPE_2D_MULTISAMPLE,
        );
        objc::msg_send_void_usize(descriptor, objc::sel(c"setSampleCount:"), sample_count);
        objc::msg_send_void_usize(
            descriptor,
            objc::sel(c"setStorageMode:"),
            MTL_STORAGE_MODE_PRIVATE,
        );
        objc::msg_send_void_usize(
            descriptor,
            objc::sel(c"setUsage:"),
            MTL_TEXTURE_USAGE_SHADER_READ | MTL_TEXTURE_USAGE_RENDER_TARGET,
        );
    }

    // SAFETY: `transport` owns a valid `MTLDevice` and `descriptor` is a live
    // descriptor; the returned texture is retained (+1).
    let texture = unsafe {
        objc::msg_send_id_1(
            transport.device_raw(),
            objc::sel(c"newTextureWithDescriptor:"),
            descriptor,
        )
    };
    let texture = ObjcObject::from_retained(texture).ok_or_else(|| {
        MetalError::InternalError("MTLDevice.newTextureWithDescriptor returned null".to_string())
    })?;
    RenderTarget::from_objc(texture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_blit_level_extent_halves_and_floors_at_one() {
        let geometry = TextureGeometry {
            width: 64,
            height: 64,
            depth: 1,
            mip_levels: 7,
            slices: 1,
            pixel_format: 70,
            sample_count: 1,
        };
        assert_eq!(level_extent(&geometry, 0), [64, 64, 1]);
        assert_eq!(level_extent(&geometry, 1), [32, 32, 1]);
        assert_eq!(level_extent(&geometry, 6), [1, 1, 1]);
        assert_eq!(level_extent(&geometry, 9), [1, 1, 1]);
    }

    #[test]
    fn macos_blit_rejects_a_nil_texture_before_any_selector() {
        let subresource = TextureSubresourceRef {
            texture: std::ptr::null_mut(),
            slice: 0,
            level: 0,
        };
        assert!(matches!(
            copy_texture_region(
                std::ptr::null_mut(),
                &TextureRegion {
                    source: subresource,
                    source_origin: [0; 3],
                    destination: subresource,
                    destination_origin: [0; 3],
                    size: [1; 3],
                },
            ),
            Err(MetalError::InternalError(_))
        ));
        assert!(matches!(
            validate_resolve(subresource, subresource),
            Err(MetalError::InvalidArg("texture is nil"))
        ));
    }

    #[test]
    fn the_copy_trace_counts_by_kind_and_is_bounded() {
        reset_copy_trace();
        let first = 0x1000 as objc::Id;
        let second = 0x2000 as objc::Id;
        note_copy_destination(BlitCopyKind::CopyTextureRegion, first);
        note_copy_destination(BlitCopyKind::CopyTextureRegion, first);
        note_copy_destination(BlitCopyKind::ResolveSubresource, second);
        assert_eq!(copy_kind_for(first), Some(BlitCopyKind::CopyTextureRegion));
        assert_eq!(
            copy_kind_for(second),
            Some(BlitCopyKind::ResolveSubresource)
        );
        assert_eq!(
            BlitCopyKind::CopyTextureRegion.as_str(),
            "copy_texture_region"
        );
        assert_eq!(copy_trace_totals(), (2, 1, 0));

        reset_copy_trace();
        for index in 0..(COPY_TRACE_MAX_KEYS + 3) {
            note_copy_destination(
                BlitCopyKind::CopyTextureRegion,
                (0x3000 + index) as objc::Id,
            );
        }
        let (region, resolve, dropped) = copy_trace_totals();
        assert_eq!(region, COPY_TRACE_MAX_KEYS as u64);
        assert_eq!(resolve, 0);
        assert_eq!(dropped, 3);
    }
}
