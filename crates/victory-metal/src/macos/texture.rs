//! Native private 2D textures with validated buffer-to-texture blits.
//!
//! This is an isolated host-side primitive on [`DirectMetalTransport`], not a
//! guest-facing wire or C ABI feature. It intentionally supports only:
//!
//! * level 0 (no mipmaps),
//! * uncompressed RGBA8/BGRA8 (`4` bytes per pixel),
//! * `MTLStorageModePrivate` GPU textures.
//!
//! Blits consume buffers from the existing no-copy registry ([`VmBufferId`]),
//! run on the transport's single `MTLCommandQueue`, and wait for completion
//! while holding the registry lock. That reuses the real `MTLBuffer` lifetime
//! and queue serial ordering without introducing a second ownership model.

use std::mem;

use crate::abi::VmBufferId;
use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::transport::MetalError;

/// Largest supported 2D texture edge, matching Apple GPU 2D texture limits.
pub const MAX_TEXTURE_DIMENSION_2D: u32 = 16_384;

/// `MTLPixelFormatRGBA8Unorm`.
const MTL_PIXEL_FORMAT_RGBA8_UNORM: usize = 70;
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: usize = 2;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLCommandBufferStatusCompleted`.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// Byte alignment for a blit buffer offset and row pitch (`4` = RGBA8/BGRA8 pixel).
const BLIT_ALIGNMENT: u64 = 4;

/// Apple `MTLOrigin` (`NSUInteger x, y, z`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// Opaque handle to a native private 2D texture.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VmTextureId(pub u64);

/// Supported uncompressed 2D pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat2d {
    /// `MTLPixelFormatRGBA8Unorm`.
    Rgba8Unorm,
    /// `MTLPixelFormatBGRA8Unorm`.
    Bgra8Unorm,
}

impl TextureFormat2d {
    /// Bytes per pixel for the supported formats.
    pub const fn bytes_per_pixel(self) -> u32 {
        4
    }

    const fn mtl_pixel_format(self) -> usize {
        match self {
            Self::Rgba8Unorm => MTL_PIXEL_FORMAT_RGBA8_UNORM,
            Self::Bgra8Unorm => MTL_PIXEL_FORMAT_BGRA8_UNORM,
        }
    }

    fn from_mtl_pixel_format(pixel_format: usize) -> Option<Self> {
        match pixel_format {
            MTL_PIXEL_FORMAT_RGBA8_UNORM => Some(Self::Rgba8Unorm),
            MTL_PIXEL_FORMAT_BGRA8_UNORM => Some(Self::Bgra8Unorm),
            _ => None,
        }
    }
}

/// Level-0 private 2D texture description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc2d {
    /// Texture width in pixels; must be in `1..=MAX_TEXTURE_DIMENSION_2D`.
    pub width: u32,
    /// Texture height in pixels; must be in `1..=MAX_TEXTURE_DIMENSION_2D`.
    pub height: u32,
    /// Uncompressed 4-byte pixel format.
    pub format: TextureFormat2d,
}

impl TextureDesc2d {
    /// Exact bytes in one level-0 row, without any hardware pitch padding.
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.width) * u64::from(self.format.bytes_per_pixel())
    }

    fn validate(&self) -> Result<(), MetalError> {
        if self.width == 0 || self.height == 0 {
            return Err(MetalError::InvalidArg(
                "texture width and height must be nonzero",
            ));
        }
        if self.width > MAX_TEXTURE_DIMENSION_2D || self.height > MAX_TEXTURE_DIMENSION_2D {
            return Err(MetalError::InvalidArg(
                "texture exceeds the supported 2D dimension limit",
            ));
        }
        Ok(())
    }
}

/// Exact buffer region for one level-0 buffer/texture blit.
///
/// `bytes_per_row` may include hardware pitch padding, but it must be 4-byte
/// aligned and at least the exact row size. The checked end offset includes
/// only the copied pixel bytes of the final row, not a full padded row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTextureRegion2d {
    /// Byte offset of the region inside the registered buffer.
    pub buffer_offset: u64,
    /// Distance in bytes between consecutive rows in the buffer.
    pub bytes_per_row: u64,
    /// Destination/source X origin inside the texture.
    pub origin_x: u32,
    /// Destination/source Y origin inside the texture.
    pub origin_y: u32,
    /// Copied width in pixels.
    pub width: u32,
    /// Copied height in pixels.
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
enum BlitDirection {
    BufferToTexture,
    TextureToBuffer,
}

/// Validates a blit region against a texture and a registered buffer length.
fn validate_region(
    desc: &TextureDesc2d,
    region: &BufferTextureRegion2d,
    buffer_len: usize,
) -> Result<(), MetalError> {
    if region.width == 0 || region.height == 0 {
        return Err(MetalError::InvalidArg(
            "blit region width and height must be nonzero",
        ));
    }
    let end_x = region
        .origin_x
        .checked_add(region.width)
        .ok_or(MetalError::InvalidArg("blit region X extent overflows u32"))?;
    let end_y = region
        .origin_y
        .checked_add(region.height)
        .ok_or(MetalError::InvalidArg("blit region Y extent overflows u32"))?;
    if end_x > desc.width || end_y > desc.height {
        return Err(MetalError::InvalidArg(
            "blit region exceeds the texture dimensions",
        ));
    }

    let row_bytes = u64::from(region.width) * u64::from(desc.format.bytes_per_pixel());
    if region.bytes_per_row < row_bytes {
        return Err(MetalError::InvalidArg(
            "blit row pitch is smaller than the exact row size",
        ));
    }
    if !region.bytes_per_row.is_multiple_of(BLIT_ALIGNMENT) {
        return Err(MetalError::InvalidArg(
            "blit row pitch must be 4-byte aligned",
        ));
    }
    if !region.buffer_offset.is_multiple_of(BLIT_ALIGNMENT) {
        return Err(MetalError::InvalidArg(
            "blit buffer offset must be 4-byte aligned",
        ));
    }

    let end = region
        .bytes_per_row
        .checked_mul(u64::from(region.height - 1))
        .and_then(|padded_rows| region.buffer_offset.checked_add(padded_rows))
        .and_then(|last_row| last_row.checked_add(row_bytes))
        .ok_or(MetalError::InvalidArg("blit region end overflows u64"))?;
    if end > buffer_len as u64 {
        return Err(MetalError::InvalidArg(
            "blit region exceeds the registered buffer length",
        ));
    }
    Ok(())
}

/// Reads back the actual texture geometry and validates its pixel format.
fn texture_desc(texture: objc::Id) -> Result<TextureDesc2d, MetalError> {
    // SAFETY: `texture` is a live `MTLTexture` kept alive by the registry lock.
    let width = unsafe { objc::msg_send_usize_0(texture, objc::sel(c"width")) };
    // SAFETY: See above.
    let height = unsafe { objc::msg_send_usize_0(texture, objc::sel(c"height")) };
    // SAFETY: See above.
    let pixel_format = unsafe { objc::msg_send_usize_0(texture, objc::sel(c"pixelFormat")) };
    let format = TextureFormat2d::from_mtl_pixel_format(pixel_format).ok_or(
        MetalError::NotSupported("texture pixel format is not RGBA8/BGRA8"),
    )?;
    let width = u32::try_from(width)
        .map_err(|_| MetalError::InternalError("texture width exceeds u32".to_string()))?;
    let height = u32::try_from(height)
        .map_err(|_| MetalError::InternalError("texture height exceeds u32".to_string()))?;
    Ok(TextureDesc2d {
        width,
        height,
        format,
    })
}

impl DirectMetalTransport {
    /// Creates a private, level-0, uncompressed 2D texture.
    ///
    /// The descriptor is fully validated before Metal sees it. Unsupported
    /// dimensions fail with [`MetalError::InvalidArg`] and perform no GPU work.
    pub fn create_texture_2d(&self, desc: TextureDesc2d) -> Result<VmTextureId, MetalError> {
        desc.validate()?;
        let _pool = AutoreleasePool::new();
        let descriptor = texture_2d_descriptor(
            desc.format.mtl_pixel_format(),
            desc.width as usize,
            desc.height as usize,
        )?;

        // SAFETY: `descriptor` is an autoreleased `MTLTextureDescriptor` and the
        // selectors take the documented `MTLStorageMode`/`MTLTextureUsage` values.
        unsafe {
            msg_send_void_usize(
                descriptor,
                objc::sel(c"setStorageMode:"),
                MTL_STORAGE_MODE_PRIVATE,
            );
            msg_send_void_usize(
                descriptor,
                objc::sel(c"setUsage:"),
                MTL_TEXTURE_USAGE_SHADER_READ,
            );
        }

        // SAFETY: `device` is a valid `MTLDevice`, `descriptor` is a live
        // descriptor in this autorelease pool, and the result is retained (+1).
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

        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.textures.insert(id, texture);
        Ok(VmTextureId(id))
    }

    /// Destroys a texture created by [`Self::create_texture_2d`].
    pub fn destroy_texture_2d(&self, texture: VmTextureId) -> Result<(), MetalError> {
        let mut state = self.lock_state()?;
        state
            .textures
            .remove(&texture.0)
            .map(|_| ())
            .ok_or_else(|| MetalError::NotFound(format!("texture {}", texture.0)))
    }

    /// Retains the `MTLTexture` behind a [`VmTextureId`] for a residency set.
    ///
    /// The retain happens while the registry lock is held, so the returned
    /// object stays valid independently of a later
    /// [`Self::destroy_texture_2d`]. This is an additive host-side accessor: it
    /// does not change the existing texture API.
    pub(crate) fn retain_texture_2d(&self, texture: VmTextureId) -> Result<ObjcObject, MetalError> {
        let state = self.lock_state()?;
        let texture = state
            .textures
            .get(&texture.0)
            .ok_or_else(|| MetalError::NotFound(format!("texture {}", texture.0)))?;
        // SAFETY: `texture` is a live `MTLTexture`; `retain` returns the same
        // object with an additional +1 reference owned by the result.
        let raw = unsafe { objc::msg_send_id_0(texture.as_raw(), objc::sel(c"retain")) };
        ObjcObject::from_retained(raw)
            .ok_or_else(|| MetalError::InternalError("MTLTexture retain returned null".to_string()))
    }

    /// Blits a registered buffer region into level 0 of a private texture.
    ///
    /// The region is validated for format, row pitch, offset and checked exact
    /// end before a command buffer exists. The blit shares the transport queue,
    /// so it is ordered with compute submissions, and the registry lock is held
    /// through completion so the backing `MTLBuffer` cannot be released early.
    pub fn blit_buffer_to_texture_2d(
        &self,
        texture: VmTextureId,
        buffer: VmBufferId,
        region: &BufferTextureRegion2d,
    ) -> Result<(), MetalError> {
        self.blit_2d(texture, buffer, region, BlitDirection::BufferToTexture)
    }

    /// Blits level 0 of a private texture back into a registered buffer region.
    ///
    /// The buffer must use a CPU-visible storage mode for the caller to observe
    /// the pixels. Region checks and queue/lifetime semantics match
    /// [`Self::blit_buffer_to_texture_2d`].
    pub fn blit_texture_to_buffer_2d(
        &self,
        texture: VmTextureId,
        buffer: VmBufferId,
        region: &BufferTextureRegion2d,
    ) -> Result<(), MetalError> {
        self.blit_2d(texture, buffer, region, BlitDirection::TextureToBuffer)
    }

    fn blit_2d(
        &self,
        texture: VmTextureId,
        buffer: VmBufferId,
        region: &BufferTextureRegion2d,
        direction: BlitDirection,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        // Hold the registry lock through completion so the registered buffer
        // wrapper and the private texture stay alive for the whole blit.
        let state = self.lock_state()?;
        let texture_obj = state
            .textures
            .get(&texture.0)
            .ok_or_else(|| MetalError::NotFound(format!("texture {}", texture.0)))?;
        let buffer_obj = state
            .buffers
            .get(&buffer.0)
            .ok_or_else(|| MetalError::NotFound(format!("buffer {}", buffer.0)))?;
        let desc = texture_desc(texture_obj.as_raw())?;
        validate_region(&desc, region, buffer_obj.len)?;

        let size = MTLSize {
            width: region.width as usize,
            height: region.height as usize,
            depth: 1,
        };
        let origin = MTLOrigin {
            x: region.origin_x as usize,
            y: region.origin_y as usize,
            z: 0,
        };
        let buffer_offset = usize::try_from(region.buffer_offset)
            .map_err(|_| MetalError::InvalidArg("blit buffer offset exceeds host usize"))?;
        let bytes_per_row = usize::try_from(region.bytes_per_row)
            .map_err(|_| MetalError::InvalidArg("blit row pitch exceeds host usize"))?;

        // SAFETY: `queue` is a valid `MTLCommandQueue`; the command buffer is
        // autoreleased and stays valid for this pool.
        let cb = unsafe { objc::msg_send_id_0(self.queue_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }
        // SAFETY: `cb` is a valid uncommitted `MTLCommandBuffer`.
        let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder")) };
        if enc.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandBuffer.blitCommandEncoder returned null".to_string(),
            ));
        }

        // SAFETY: `enc` is an active blit encoder; both resources are live
        // registry entries, the geometry was validated above, and the encoded
        // copy stays inside the registered buffer and level-0 texture.
        unsafe {
            let buffer_raw = buffer_obj.buffer.as_raw();
            let texture_raw = texture_obj.as_raw();
            match direction {
                BlitDirection::BufferToTexture => copy_buffer_to_texture(
                    enc,
                    buffer_raw,
                    buffer_offset,
                    bytes_per_row,
                    size,
                    texture_raw,
                    origin,
                ),
                BlitDirection::TextureToBuffer => copy_texture_to_buffer(
                    enc,
                    texture_raw,
                    origin,
                    size,
                    buffer_raw,
                    buffer_offset,
                    bytes_per_row,
                ),
            }
            objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }

        // SAFETY: `cb` completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: `[cb error]` is null or an autoreleased `NSError *`.
            let err_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
            // SAFETY: `err_raw` is null or a valid `NSError *`.
            let msg = unsafe { objc::nserror_message(err_raw) };
            return Err(MetalError::InternalError(format!(
                "Metal blit command buffer failed (status={status}): {msg}"
            )));
        }
        Ok(())
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
    // SAFETY: `cls` is the `MTLTextureDescriptor` class, the selector is the
    // documented class factory, and `mipmapped` receives `NO` as a `BOOL` byte.
    let descriptor = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
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

/// Sends a selector taking one `NSUInteger` (no object lifetime transfer).
unsafe fn msg_send_void_usize(target: objc::Id, s: objc::Sel, value: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
    // SAFETY: Caller supplies a valid receiver, selector, and matching
    // `NSUInteger` return ABI.
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's argument layout.
    unsafe { f(target, s, value) };
}

/// Encodes `copyFromBuffer:...toTexture:...` for a single level-0 2D image.
unsafe fn copy_buffer_to_texture(
    enc: objc::Id,
    buffer: objc::Id,
    buffer_offset: usize,
    bytes_per_row: usize,
    size: MTLSize,
    texture: objc::Id,
    origin: MTLOrigin,
) {
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
    // SAFETY: Caller supplies a valid blit encoder and live Metal objects. For
    // a 2D texture the per-image stride is `0` (single image, level 0).
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Argument layout matches the selector declaration.
    unsafe {
        f(
            enc,
            objc::sel(c"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
            buffer,
            buffer_offset,
            bytes_per_row,
            0,
            size,
            texture,
            0,
            0,
            origin,
        );
    }
}

/// Encodes `copyFromTexture:...toBuffer:...` for a single level-0 2D image.
unsafe fn copy_texture_to_buffer(
    enc: objc::Id,
    texture: objc::Id,
    origin: MTLOrigin,
    size: MTLSize,
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
        MTLSize,
        objc::Id,
        usize,
        usize,
        usize,
    );
    // SAFETY: Caller supplies a valid blit encoder and live Metal objects. For
    // a 2D texture the per-image stride is `0` (single image, level 0).
    let f: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: Argument layout matches the selector declaration.
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
        );
    }
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MetalTransport;

    fn desc(width: u32, height: u32) -> TextureDesc2d {
        TextureDesc2d {
            width,
            height,
            format: TextureFormat2d::Rgba8Unorm,
        }
    }

    fn region(
        buffer_offset: u64,
        bytes_per_row: u64,
        origin_x: u32,
        origin_y: u32,
        width: u32,
        height: u32,
    ) -> BufferTextureRegion2d {
        BufferTextureRegion2d {
            buffer_offset,
            bytes_per_row,
            origin_x,
            origin_y,
            width,
            height,
        }
    }

    #[test]
    fn validates_descriptor_dimensions_and_supported_pixel_formats() {
        assert!(desc(0, 4).validate().is_err());
        assert!(desc(4, 0).validate().is_err());
        assert!(
            desc(MAX_TEXTURE_DIMENSION_2D, MAX_TEXTURE_DIMENSION_2D)
                .validate()
                .is_ok()
        );
        assert!(desc(MAX_TEXTURE_DIMENSION_2D + 1, 4).validate().is_err());
        assert!(desc(4, MAX_TEXTURE_DIMENSION_2D + 1).validate().is_err());

        assert_eq!(TextureFormat2d::Rgba8Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat2d::Bgra8Unorm.bytes_per_pixel(), 4);
        assert_eq!(
            TextureFormat2d::from_mtl_pixel_format(MTL_PIXEL_FORMAT_RGBA8_UNORM),
            Some(TextureFormat2d::Rgba8Unorm)
        );
        assert_eq!(
            TextureFormat2d::from_mtl_pixel_format(MTL_PIXEL_FORMAT_BGRA8_UNORM),
            Some(TextureFormat2d::Bgra8Unorm)
        );
        // RGBA8Unorm_sRGB is a different MTLPixelFormat and is unsupported here.
        assert_eq!(TextureFormat2d::from_mtl_pixel_format(71), None);
        assert_eq!(TextureFormat2d::from_mtl_pixel_format(0), None);
        assert_eq!(desc(9, 3).row_bytes(), 36);
    }

    #[test]
    fn validates_exact_region_extent_including_the_final_row() {
        // 4x4 RGBA8: exact row is 16 bytes, last row needs only 16 bytes.
        let texture = desc(4, 4);
        assert_eq!(texture.row_bytes(), 16);

        let exact = region(0, 16, 0, 0, 4, 4);
        assert!(validate_region(&texture, &exact, 64).is_ok());
        assert!(validate_region(&texture, &exact, 63).is_err());

        let offset = region(8, 16, 0, 0, 4, 4);
        assert!(validate_region(&texture, &offset, 72).is_ok());
        assert!(validate_region(&texture, &offset, 71).is_err());

        // A padded pitch must not be charged for the final row: 16 + 32 + 8 = 56.
        let partial = region(16, 32, 0, 0, 2, 2);
        assert!(validate_region(&texture, &partial, 56).is_ok());
        assert!(validate_region(&texture, &partial, 55).is_err());

        // A sub-rectangle with a nonzero origin stays inside the 4x4 texture.
        let sub = region(0, 16, 1, 1, 3, 3);
        assert!(validate_region(&texture, &sub, 64).is_ok());
    }

    #[test]
    fn rejects_misaligned_pitch_offset_zero_extent_and_overflow() {
        let texture = desc(4, 4);

        assert!(validate_region(&texture, &region(2, 16, 0, 0, 4, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 15, 0, 0, 4, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 20, 0, 0, 4, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 12, 0, 0, 4, 4), 64).is_err());

        assert!(validate_region(&texture, &region(0, 16, 0, 0, 0, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 16, 0, 0, 4, 0), 64).is_err());

        assert!(validate_region(&texture, &region(0, 16, 1, 0, 4, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 16, 0, 1, 4, 4), 64).is_err());
        assert!(validate_region(&texture, &region(0, 16, 0, 0, 5, 1), 64).is_err());

        assert!(validate_region(&texture, &region(u64::MAX, 16, 0, 0, 1, 1), usize::MAX).is_err());
        assert!(validate_region(&texture, &region(0, u64::MAX, 0, 0, 1, 2), usize::MAX).is_err());
    }

    /// Positive GPU path, compiled but intentionally not executed by this task:
    /// the GPU is reserved for another swarm worker. It uses only synthetic
    /// pixels and the public host API, including padded pitch and nonzero buffer
    /// offsets, to prove the no-copy buffer -> private texture -> buffer path.
    #[test]
    #[ignore = "GPU reserved by another worker; NATIVE-TEX compiles this positive path only"]
    fn prepared_private_texture_blit_roundtrip_preserves_synthetic_pattern() {
        use std::ffi::c_void;
        use std::ptr::null_mut;

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const BUFFER_BYTES: usize = 16384;
        const WIDTH: u32 = 8;
        const HEIGHT: u32 = 8;
        const ROW_BYTES: usize = WIDTH as usize * 4;
        const PITCH: usize = 256;
        const OFFSET: usize = 512;
        const SENTINEL: u8 = 0xA5;

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is not registered.
                unsafe { free(self.0) };
            }
        }

        let transport = DirectMetalTransport::new().expect("Metal device should open");

        let mut src_raw: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
        let rc = unsafe { posix_memalign(&mut src_raw, ALIGN, BUFFER_BYTES) };
        assert_eq!(rc, 0, "source posix_memalign failed");
        assert!(!src_raw.is_null());
        let _src_guard = PageGuard(src_raw);

        let mut dst_raw: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
        let rc = unsafe { posix_memalign(&mut dst_raw, ALIGN, BUFFER_BYTES) };
        assert_eq!(rc, 0, "destination posix_memalign failed");
        assert!(!dst_raw.is_null());
        let _dst_guard = PageGuard(dst_raw);

        // SAFETY: Both allocations are live for `BUFFER_BYTES` bytes.
        let src = unsafe { std::slice::from_raw_parts_mut(src_raw.cast::<u8>(), BUFFER_BYTES) };
        // SAFETY: See above.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_raw.cast::<u8>(), BUFFER_BYTES) };
        src.fill(SENTINEL);
        dst.fill(SENTINEL);
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let base = OFFSET + y * PITCH + x * 4;
                src[base] = x as u8;
                src[base + 1] = y as u8;
                src[base + 2] = (x as u8) ^ (y as u8) ^ 0x5A;
                src[base + 3] = 0xFF;
            }
        }

        // SAFETY: Both regions are aligned, page-multiple, and outlive the blits.
        let src_id = unsafe { transport.register_zerocopy(src_raw.cast::<u8>(), BUFFER_BYTES) }
            .expect("source register_zerocopy should succeed");
        // SAFETY: See above.
        let dst_id = unsafe { transport.register_zerocopy(dst_raw.cast::<u8>(), BUFFER_BYTES) }
            .expect("destination register_zerocopy should succeed");

        let texture = transport
            .create_texture_2d(TextureDesc2d {
                width: WIDTH,
                height: HEIGHT,
                format: TextureFormat2d::Rgba8Unorm,
            })
            .expect("private texture creation should succeed");

        let copy = BufferTextureRegion2d {
            buffer_offset: OFFSET as u64,
            bytes_per_row: PITCH as u64,
            origin_x: 0,
            origin_y: 0,
            width: WIDTH,
            height: HEIGHT,
        };
        transport
            .blit_buffer_to_texture_2d(texture, src_id, &copy)
            .expect("buffer -> private texture blit should complete");
        transport
            .blit_texture_to_buffer_2d(texture, dst_id, &copy)
            .expect("private texture -> buffer blit should complete");

        for y in 0..HEIGHT as usize {
            let row = OFFSET + y * PITCH;
            for x in 0..WIDTH as usize {
                let base = row + x * 4;
                assert_eq!(
                    &dst[base..base + 4],
                    &src[base..base + 4],
                    "pixel ({x},{y}) changed through the private texture"
                );
            }
            assert!(
                dst[row + ROW_BYTES..row + PITCH]
                    .iter()
                    .all(|&byte| byte == SENTINEL),
                "row {y} padding was modified"
            );
        }

        transport
            .destroy_texture_2d(texture)
            .expect("texture destruction should succeed");
        transport
            .unregister_buffer(src_id)
            .expect("source unregister_buffer should succeed");
        transport
            .unregister_buffer(dst_id)
            .expect("destination unregister_buffer should succeed");
    }
}
