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

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::render::{PixelFormat, RenderTarget};
use crate::transport::MetalError;

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
        validate_box(
            "source",
            self.source,
            self.source_origin,
            self.size,
        )?;
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
    let (
        width,
        height,
        depth,
        mip_levels,
        slices,
        pixel_format,
        sample_count,
    ) = unsafe {
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
    [half(geometry.width), half(geometry.height), half(geometry.depth)]
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
pub(crate) fn copy_texture_region(
    enc: objc::Id,
    region: &TextureRegion,
) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "blit encoder is nil".to_string(),
        ));
    }
    region.validate()?;
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
    let attachment = unsafe {
        objc::msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0)
    };
    if attachment.is_null() {
        return Err(MetalError::InternalError(
            "colour attachment descriptor is nil".to_string(),
        ));
    }
    // SAFETY: both textures are registry-held and validated; the values are
    // documented `MTLLoadAction`/`MTLStoreAction` encodings and subresource
    // indices.
    unsafe {
        objc::msg_send_void_1(
            attachment,
            objc::sel(c"setTexture:"),
            source.texture,
        );
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
    if width == 0 || height == 0 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION {
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
        objc::msg_send_void_usize(
            descriptor,
            objc::sel(c"setSampleCount:"),
            sample_count,
        );
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
}
