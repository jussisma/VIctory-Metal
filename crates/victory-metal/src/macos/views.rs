//! Native resource views: formats, depth/stencil surfaces and descriptor
//! binding for the ARM64 Metal renderer.
//!
//! Scope of this module:
//!
//! - The guest format to `MTLPixelFormat` mapping, including the depth and
//!   stencil formats, with typed, sRGB and compressed families covered and an
//!   explicit error for formats Metal cannot represent.
//! - Private 2D colour and depth/stencil texture creation, sampled and
//!   storage usage flags, mip and array levels.
//! - The host-side descriptor shape for SRV/UAV/RTV/DSV use: format, view
//!   dimension, first mip, mip count, first slice, slice count and the
//!   resource handle they resolve against.
//! - Upload paths: zero-copy when the guest range is already page-aligned and
//!   shareable, otherwise a GPU blit from a staging buffer.
//!
//! Metal FFI specific to textures and depth states lives here so `objc.rs`
//! stays a shared, stable helper module.
//!
//! Handles created here live in the same [`crate::macos::device::DeviceState`]
//! texture registry as [`crate::macos::texture`], so
//! [`crate::macos::texture::DirectMetalTransport::destroy_texture_2d`] and the
//! validated buffer blits apply to them unchanged.
//!
//! # Depth formats on Apple GPUs
//!
//! `MTLPixelFormatDepth24Unorm_Stencil8` (255) and
//! `MTLPixelFormatX24_Stencil8` (262) are unavailable on Apple Silicon
//! (`depth24Stencil8PixelFormatSupported` is false) and Metal aborts the
//! process when a descriptor asks for them. Guest `D24_UNORM_S8` therefore
//! resolves to `MTLPixelFormatDepth32Float_Stencil8` and its stencil plane to
//! `MTLPixelFormatX32_Stencil8` on those GPUs; devices that report support
//! keep the 24-bit formats. The substitution is centralized in
//! [`PixelFormat::mtl_pixel_format_on`] and
//! [`PixelFormat::stencil_plane_format`] so no descriptor path can bypass it.

use std::ffi::CStr;
use std::mem;

use thiserror::Error;

use crate::abi::VmBufferId;
use crate::macos::device::{DirectMetalTransport, RegisteredBuffer};
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::texture::{VmTextureId, MAX_TEXTURE_DIMENSION_2D};
use crate::transport::{MetalError, MetalTransport};

/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: usize = 2;
/// `MTLResourceStorageModeShared` for CPU-written staging buffers.
const MTL_RESOURCE_STORAGE_MODE_SHARED: usize = 0;
/// `MTLTextureType2D`.
const MTL_TEXTURE_TYPE_2D: usize = 2;
/// `MTLTextureType2DArray`.
const MTL_TEXTURE_TYPE_2D_ARRAY: usize = 3;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLTextureUsageShaderWrite`.
const MTL_TEXTURE_USAGE_SHADER_WRITE: usize = 2;
/// `MTLTextureUsageRenderTarget` (also the usage bit for depth/stencil targets).
const MTL_TEXTURE_USAGE_RENDER_TARGET: usize = 4;
/// `MTLCommandBufferStatusCompleted`.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// Widest 2D mip chain for a `16384` edge (`log2(16384) + 1`).
const MAX_MIP_LEVELS: u32 = 15;
/// Metal's maximum 2D array length.
const MAX_ARRAY_LAYERS: u32 = 2_048;
/// Minimum alignment of a blit row pitch and buffer offset, matching the
/// 4-byte texel size of the smallest supported pixel format.
const BLIT_ALIGNMENT: u64 = 4;
/// Row pitch of a CPU-written staging buffer; Metal requires the source row
/// pitch of a buffer-to-texture copy to be a multiple of 64 bytes (256 keeps
/// every supported texel size and block size integral).
const STAGING_ROW_ALIGNMENT: u64 = 256;

/// `MTLPixelFormatDepth16Unorm`.
const MTL_PIXEL_FORMAT_DEPTH16_UNORM: usize = 250;
/// `MTLPixelFormatDepth32Float`.
const MTL_PIXEL_FORMAT_DEPTH32_FLOAT: usize = 252;
/// `MTLPixelFormatDepth24Unorm_Stencil8` (unavailable on Apple Silicon).
const MTL_PIXEL_FORMAT_DEPTH24_UNORM_STENCIL8: usize = 255;
/// `MTLPixelFormatDepth32Float_Stencil8`.
const MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8: usize = 260;
/// `MTLPixelFormatX32_Stencil8`.
const MTL_PIXEL_FORMAT_X32_STENCIL8: usize = 261;
/// `MTLPixelFormatX24_Stencil8` (unavailable on Apple Silicon).
const MTL_PIXEL_FORMAT_X24_STENCIL8: usize = 262;

/// Apple `MTLOrigin` (`NSUInteger x, y, z`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// Errors from the native resource layer.
#[derive(Debug, Error)]
pub enum ResourceError {
    /// A guest DXGI format has no entry in the mapping table.
    #[error("DXGI format {0} is not mapped by the native resource layer")]
    UnmappedDxgiFormat(u32),
    /// A texture descriptor failed validation before Metal saw it.
    #[error("texture descriptor rejected: {0}")]
    InvalidDescriptor(String),
    /// A resource view failed validation against its texture.
    #[error("resource view rejected: {0}")]
    InvalidView(String),
    /// A referenced texture handle is not registered.
    #[error("texture {0} not found")]
    UnknownTexture(u64),
    /// An upload request failed validation before any GPU work.
    #[error("texture upload rejected: {0}")]
    InvalidUpload(String),
    /// A Metal call failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// Per-device format capabilities that change the guest mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatSupport {
    /// `[MTLDevice supportsBCTextureCompression]`.
    pub bc_texture_compression: bool,
    /// `[MTLDevice depth24Stencil8PixelFormatSupported]`; false on Apple
    /// Silicon, where 24-bit depth descriptors make Metal abort.
    pub depth24_stencil8: bool,
}

/// Guest DXGI pixel formats mapped by the native renderer.
///
/// The numeric DXGI values are the ones in the Windows SDK
/// (`DXGI_FORMAT_*`); `R8UnormSrgb` and `Rg8UnormSrgb` are Metal-only mirrors
/// because DXGI has no 8-bit single/dual-channel sRGB encoding (they are
/// reachable through [`PixelFormat::from_mtl_pixel_format`] only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// `DXGI_FORMAT_R8_UNORM` / `MTLPixelFormatR8Unorm`.
    R8Unorm,
    /// `MTLPixelFormatR8Unorm_sRGB` (no DXGI equivalent).
    R8UnormSrgb,
    /// `DXGI_FORMAT_R8G8_UNORM` / `MTLPixelFormatRG8Unorm`.
    Rg8Unorm,
    /// `MTLPixelFormatRG8Unorm_sRGB` (no DXGI equivalent).
    Rg8UnormSrgb,
    /// `DXGI_FORMAT_R8G8B8A8_UNORM` / `MTLPixelFormatRGBA8Unorm`.
    Rgba8Unorm,
    /// `DXGI_FORMAT_R8G8B8A8_UNORM_SRGB` / `MTLPixelFormatRGBA8Unorm_sRGB`.
    Rgba8UnormSrgb,
    /// `DXGI_FORMAT_B8G8R8A8_UNORM` / `MTLPixelFormatBGRA8Unorm`.
    Bgra8Unorm,
    /// `DXGI_FORMAT_B8G8R8A8_UNORM_SRGB` / `MTLPixelFormatBGRA8Unorm_sRGB`.
    Bgra8UnormSrgb,
    /// `DXGI_FORMAT_R16_FLOAT` / `MTLPixelFormatR16Float`.
    R16Float,
    /// `DXGI_FORMAT_R16G16_FLOAT` / `MTLPixelFormatRG16Float`.
    Rg16Float,
    /// `DXGI_FORMAT_R16G16B16A16_FLOAT` / `MTLPixelFormatRGBA16Float`.
    Rgba16Float,
    /// `DXGI_FORMAT_R32_FLOAT` / `MTLPixelFormatR32Float`.
    R32Float,
    /// `DXGI_FORMAT_R32G32_FLOAT` / `MTLPixelFormatRG32Float`.
    Rg32Float,
    /// `DXGI_FORMAT_R32G32B32A32_FLOAT` / `MTLPixelFormatRGBA32Float`.
    Rgba32Float,
    /// `DXGI_FORMAT_R32_UINT` / `MTLPixelFormatR32Uint`.
    R32Uint,
    /// `DXGI_FORMAT_R32G32_UINT` / `MTLPixelFormatRG32Uint`.
    Rg32Uint,
    /// `DXGI_FORMAT_R32G32B32A32_UINT` / `MTLPixelFormatRGBA32Uint`.
    Rgba32Uint,
    /// `DXGI_FORMAT_D32_FLOAT` / `MTLPixelFormatDepth32Float`.
    Depth32Float,
    /// `DXGI_FORMAT_D24_UNORM_S8_UINT`; 24-bit depth where supported,
    /// otherwise `MTLPixelFormatDepth32Float_Stencil8`.
    Depth24UnormStencil8,
    /// `DXGI_FORMAT_D16_UNORM` / `MTLPixelFormatDepth16Unorm`.
    Depth16Unorm,
    /// `DXGI_FORMAT_D32_FLOAT_S8X24_UINT` /
    /// `MTLPixelFormatDepth32Float_Stencil8`.
    Depth32FloatStencil8,
    /// `DXGI_FORMAT_BC1_UNORM` / `MTLPixelFormatBC1_RGBA`.
    Bc1Unorm,
    /// `DXGI_FORMAT_BC1_UNORM_SRGB` / `MTLPixelFormatBC1_RGBA_sRGB`.
    Bc1UnormSrgb,
    /// `DXGI_FORMAT_BC3_UNORM` / `MTLPixelFormatBC3_RGBA`.
    Bc3Unorm,
    /// `DXGI_FORMAT_BC3_UNORM_SRGB` / `MTLPixelFormatBC3_RGBA_sRGB`.
    Bc3UnormSrgb,
    /// `DXGI_FORMAT_BC7_UNORM` / `MTLPixelFormatBC7_RGBAUnorm`.
    Bc7Unorm,
    /// `DXGI_FORMAT_BC7_UNORM_SRGB` / `MTLPixelFormatBC7_RGBAUnorm_sRGB`.
    Bc7UnormSrgb,
}

impl PixelFormat {
    /// Every mapped format, in table order.
    pub const ALL: [Self; 27] = [
        Self::R8Unorm,
        Self::R8UnormSrgb,
        Self::Rg8Unorm,
        Self::Rg8UnormSrgb,
        Self::Rgba8Unorm,
        Self::Rgba8UnormSrgb,
        Self::Bgra8Unorm,
        Self::Bgra8UnormSrgb,
        Self::R16Float,
        Self::Rg16Float,
        Self::Rgba16Float,
        Self::R32Float,
        Self::Rg32Float,
        Self::Rgba32Float,
        Self::R32Uint,
        Self::Rg32Uint,
        Self::Rgba32Uint,
        Self::Depth32Float,
        Self::Depth24UnormStencil8,
        Self::Depth16Unorm,
        Self::Depth32FloatStencil8,
        Self::Bc1Unorm,
        Self::Bc1UnormSrgb,
        Self::Bc3Unorm,
        Self::Bc3UnormSrgb,
        Self::Bc7Unorm,
        Self::Bc7UnormSrgb,
    ];

    /// Human-readable DXGI-style name for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::R8Unorm => "R8_UNORM",
            Self::R8UnormSrgb => "R8_UNORM_SRGB",
            Self::Rg8Unorm => "R8G8_UNORM",
            Self::Rg8UnormSrgb => "R8G8_UNORM_SRGB",
            Self::Rgba8Unorm => "R8G8B8A8_UNORM",
            Self::Rgba8UnormSrgb => "R8G8B8A8_UNORM_SRGB",
            Self::Bgra8Unorm => "B8G8R8A8_UNORM",
            Self::Bgra8UnormSrgb => "B8G8R8A8_UNORM_SRGB",
            Self::R16Float => "R16_FLOAT",
            Self::Rg16Float => "R16G16_FLOAT",
            Self::Rgba16Float => "R16G16B16A16_FLOAT",
            Self::R32Float => "R32_FLOAT",
            Self::Rg32Float => "R32G32_FLOAT",
            Self::Rgba32Float => "R32G32B32A32_FLOAT",
            Self::R32Uint => "R32_UINT",
            Self::Rg32Uint => "R32G32_UINT",
            Self::Rgba32Uint => "R32G32B32A32_UINT",
            Self::Depth32Float => "D32_FLOAT",
            Self::Depth24UnormStencil8 => "D24_UNORM_S8_UINT",
            Self::Depth16Unorm => "D16_UNORM",
            Self::Depth32FloatStencil8 => "D32_FLOAT_S8X24_UINT",
            Self::Bc1Unorm => "BC1_UNORM",
            Self::Bc1UnormSrgb => "BC1_UNORM_SRGB",
            Self::Bc3Unorm => "BC3_UNORM",
            Self::Bc3UnormSrgb => "BC3_UNORM_SRGB",
            Self::Bc7Unorm => "BC7_UNORM",
            Self::Bc7UnormSrgb => "BC7_UNORM_SRGB",
        }
    }

    /// The guest `DXGI_FORMAT_*` value, or `None` for the Metal-only sRGB
    /// mirrors.
    pub const fn dxgi(self) -> Option<u32> {
        match self {
            Self::R8Unorm => Some(61),
            Self::R8UnormSrgb => None,
            Self::Rg8Unorm => Some(49),
            Self::Rg8UnormSrgb => None,
            Self::Rgba8Unorm => Some(28),
            Self::Rgba8UnormSrgb => Some(29),
            Self::Bgra8Unorm => Some(87),
            Self::Bgra8UnormSrgb => Some(91),
            Self::R16Float => Some(54),
            Self::Rg16Float => Some(34),
            Self::Rgba16Float => Some(10),
            Self::R32Float => Some(41),
            Self::Rg32Float => Some(16),
            Self::Rgba32Float => Some(2),
            Self::R32Uint => Some(42),
            Self::Rg32Uint => Some(17),
            Self::Rgba32Uint => Some(3),
            Self::Depth32Float => Some(40),
            Self::Depth24UnormStencil8 => Some(45),
            Self::Depth16Unorm => Some(55),
            Self::Depth32FloatStencil8 => Some(20),
            Self::Bc1Unorm => Some(71),
            Self::Bc1UnormSrgb => Some(72),
            Self::Bc3Unorm => Some(77),
            Self::Bc3UnormSrgb => Some(78),
            Self::Bc7Unorm => Some(98),
            Self::Bc7UnormSrgb => Some(99),
        }
    }

    /// Resolves a guest `DXGI_FORMAT_*` value, naming the value on failure.
    pub fn from_dxgi(value: u32) -> Result<Self, ResourceError> {
        match value {
            61 => Ok(Self::R8Unorm),
            49 => Ok(Self::Rg8Unorm),
            28 => Ok(Self::Rgba8Unorm),
            29 => Ok(Self::Rgba8UnormSrgb),
            87 => Ok(Self::Bgra8Unorm),
            91 => Ok(Self::Bgra8UnormSrgb),
            54 => Ok(Self::R16Float),
            34 => Ok(Self::Rg16Float),
            10 => Ok(Self::Rgba16Float),
            41 => Ok(Self::R32Float),
            16 => Ok(Self::Rg32Float),
            2 => Ok(Self::Rgba32Float),
            42 => Ok(Self::R32Uint),
            17 => Ok(Self::Rg32Uint),
            3 => Ok(Self::Rgba32Uint),
            40 => Ok(Self::Depth32Float),
            45 => Ok(Self::Depth24UnormStencil8),
            55 => Ok(Self::Depth16Unorm),
            20 => Ok(Self::Depth32FloatStencil8),
            71 => Ok(Self::Bc1Unorm),
            72 => Ok(Self::Bc1UnormSrgb),
            77 => Ok(Self::Bc3Unorm),
            78 => Ok(Self::Bc3UnormSrgb),
            98 => Ok(Self::Bc7Unorm),
            99 => Ok(Self::Bc7UnormSrgb),
            other => Err(ResourceError::UnmappedDxgiFormat(other)),
        }
    }

    /// The nominal Metal pixel format, ignoring per-device substitutions.
    ///
    /// Prefer [`Self::mtl_pixel_format_on`] for descriptor creation: a 24-bit
    /// depth descriptor aborts on GPUs without 24-bit support.
    pub const fn mtl_pixel_format(self) -> usize {
        match self {
            Self::R8Unorm => 10,
            Self::R8UnormSrgb => 11,
            Self::Rg8Unorm => 30,
            Self::Rg8UnormSrgb => 31,
            Self::Rgba8Unorm => 70,
            Self::Rgba8UnormSrgb => 71,
            Self::Bgra8Unorm => 80,
            Self::Bgra8UnormSrgb => 81,
            Self::R16Float => 25,
            Self::Rg16Float => 65,
            Self::Rgba16Float => 115,
            Self::R32Float => 55,
            Self::Rg32Float => 105,
            Self::Rgba32Float => 125,
            Self::R32Uint => 53,
            Self::Rg32Uint => 103,
            Self::Rgba32Uint => 123,
            Self::Depth32Float => MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
            Self::Depth24UnormStencil8 => MTL_PIXEL_FORMAT_DEPTH24_UNORM_STENCIL8,
            Self::Depth16Unorm => MTL_PIXEL_FORMAT_DEPTH16_UNORM,
            Self::Depth32FloatStencil8 => MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            Self::Bc1Unorm => 130,
            Self::Bc1UnormSrgb => 131,
            Self::Bc3Unorm => 134,
            Self::Bc3UnormSrgb => 135,
            Self::Bc7Unorm => 152,
            Self::Bc7UnormSrgb => 153,
        }
    }

    /// The Metal pixel format a descriptor may use on `support`'s device.
    pub const fn mtl_pixel_format_on(self, support: FormatSupport) -> usize {
        match self {
            Self::Depth24UnormStencil8 if !support.depth24_stencil8 => {
                MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8
            }
            _ => self.mtl_pixel_format(),
        }
    }

    /// The stencil-plane pixel format for a combined depth/stencil format, or
    /// `None` for formats without a stencil plane.
    pub const fn stencil_plane_format(self, support: FormatSupport) -> Option<usize> {
        match self {
            Self::Depth24UnormStencil8 => Some(if support.depth24_stencil8 {
                MTL_PIXEL_FORMAT_X24_STENCIL8
            } else {
                MTL_PIXEL_FORMAT_X32_STENCIL8
            }),
            Self::Depth32FloatStencil8 => Some(MTL_PIXEL_FORMAT_X32_STENCIL8),
            _ => None,
        }
    }

    /// Reverse lookup from a Metal pixel format.
    pub const fn from_mtl_pixel_format(pixel_format: usize) -> Option<Self> {
        match pixel_format {
            10 => Some(Self::R8Unorm),
            11 => Some(Self::R8UnormSrgb),
            30 => Some(Self::Rg8Unorm),
            31 => Some(Self::Rg8UnormSrgb),
            70 => Some(Self::Rgba8Unorm),
            71 => Some(Self::Rgba8UnormSrgb),
            80 => Some(Self::Bgra8Unorm),
            81 => Some(Self::Bgra8UnormSrgb),
            25 => Some(Self::R16Float),
            65 => Some(Self::Rg16Float),
            115 => Some(Self::Rgba16Float),
            55 => Some(Self::R32Float),
            105 => Some(Self::Rg32Float),
            125 => Some(Self::Rgba32Float),
            53 => Some(Self::R32Uint),
            103 => Some(Self::Rg32Uint),
            123 => Some(Self::Rgba32Uint),
            MTL_PIXEL_FORMAT_DEPTH16_UNORM => Some(Self::Depth16Unorm),
            MTL_PIXEL_FORMAT_DEPTH32_FLOAT => Some(Self::Depth32Float),
            MTL_PIXEL_FORMAT_DEPTH24_UNORM_STENCIL8 => Some(Self::Depth24UnormStencil8),
            MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8 => Some(Self::Depth32FloatStencil8),
            130 => Some(Self::Bc1Unorm),
            131 => Some(Self::Bc1UnormSrgb),
            134 => Some(Self::Bc3Unorm),
            135 => Some(Self::Bc3UnormSrgb),
            152 => Some(Self::Bc7Unorm),
            153 => Some(Self::Bc7UnormSrgb),
            _ => None,
        }
    }

    /// Whether this is a depth or depth/stencil format.
    pub const fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth32Float
                | Self::Depth24UnormStencil8
                | Self::Depth16Unorm
                | Self::Depth32FloatStencil8
        )
    }

    /// Whether this format carries a stencil plane.
    pub const fn has_stencil(self) -> bool {
        matches!(self, Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8)
    }

    /// Whether this is a block-compressed colour format.
    pub const fn is_block_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1Unorm
                | Self::Bc1UnormSrgb
                | Self::Bc3Unorm
                | Self::Bc3UnormSrgb
                | Self::Bc7Unorm
                | Self::Bc7UnormSrgb
        )
    }

    /// Uncompressed bytes per pixel (`None` for depth and BC formats).
    pub const fn bytes_per_pixel(self) -> Option<u32> {
        match self {
            Self::R8Unorm | Self::R8UnormSrgb => Some(1),
            Self::Rg8Unorm | Self::Rg8UnormSrgb => Some(2),
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb => Some(4),
            Self::R16Float => Some(2),
            Self::Rg16Float => Some(4),
            Self::Rgba16Float => Some(8),
            Self::R32Float | Self::R32Uint => Some(4),
            Self::Rg32Float | Self::Rg32Uint => Some(8),
            Self::Rgba32Float | Self::Rgba32Uint => Some(16),
            _ => None,
        }
    }

    /// Bytes in one 4x4 BC block (`None` for unconpressed and depth formats).
    pub const fn block_bytes(self) -> Option<u32> {
        match self {
            Self::Bc1Unorm | Self::Bc1UnormSrgb => Some(8),
            Self::Bc3Unorm | Self::Bc3UnormSrgb | Self::Bc7Unorm | Self::Bc7UnormSrgb => Some(16),
            _ => None,
        }
    }

    /// Exact bytes in one row of `width` pixels, without pitch padding.
    ///
    /// `None` for depth/stencil formats (no guest upload layout) and for a
    /// block width that is not a multiple of the 4-pixel block.
    pub fn row_bytes(self, width: u32) -> Option<u64> {
        if let Some(bytes_per_pixel) = self.bytes_per_pixel() {
            return Some(u64::from(width) * u64::from(bytes_per_pixel));
        }
        let block = self.block_bytes()?;
        if width == 0 || !width.is_multiple_of(4) {
            return None;
        }
        Some(u64::from(width / 4) * u64::from(block))
    }

    /// The paired sRGB or linear format, if the pair keeps the same memory
    /// layout (D3D12 allows this cast for SRVs and RTVs).
    pub const fn srgb_pair(self) -> Option<Self> {
        match self {
            Self::R8Unorm => Some(Self::R8UnormSrgb),
            Self::R8UnormSrgb => Some(Self::R8Unorm),
            Self::Rg8Unorm => Some(Self::Rg8UnormSrgb),
            Self::Rg8UnormSrgb => Some(Self::Rg8Unorm),
            Self::Rgba8Unorm => Some(Self::Rgba8UnormSrgb),
            Self::Rgba8UnormSrgb => Some(Self::Rgba8Unorm),
            Self::Bgra8Unorm => Some(Self::Bgra8UnormSrgb),
            Self::Bgra8UnormSrgb => Some(Self::Bgra8Unorm),
            Self::Bc1Unorm => Some(Self::Bc1UnormSrgb),
            Self::Bc1UnormSrgb => Some(Self::Bc1Unorm),
            Self::Bc3Unorm => Some(Self::Bc3UnormSrgb),
            Self::Bc3UnormSrgb => Some(Self::Bc3Unorm),
            Self::Bc7Unorm => Some(Self::Bc7UnormSrgb),
            Self::Bc7UnormSrgb => Some(Self::Bc7Unorm),
            _ => None,
        }
    }
}

/// Whether a raw Metal pixel format is a depth or stencil format.
pub const fn mtl_pixel_format_is_depth_stencil(pixel_format: usize) -> bool {
    matches!(
        pixel_format,
        MTL_PIXEL_FORMAT_DEPTH16_UNORM
            | MTL_PIXEL_FORMAT_DEPTH32_FLOAT
            | MTL_PIXEL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8
            | MTL_PIXEL_FORMAT_X24_STENCIL8
            | MTL_PIXEL_FORMAT_X32_STENCIL8
    )
}

/// Metal texture usage bits the renderer requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureUsage(u8);

impl TextureUsage {
    /// Sampled by shaders (SRV).
    pub const SHADER_READ: Self = Self(1 << 0);
    /// Written by shaders (UAV).
    pub const SHADER_WRITE: Self = Self(1 << 1);
    /// Bound as a render target or depth/stencil target (RTV/DSV).
    pub const RENDER_TARGET: Self = Self(1 << 2);

    /// Raw Metal `MTLTextureUsage` bits.
    pub const fn bits(self) -> usize {
        let mut metal = 0;
        if self.0 & Self::SHADER_READ.0 != 0 {
            metal |= MTL_TEXTURE_USAGE_SHADER_READ;
        }
        if self.0 & Self::SHADER_WRITE.0 != 0 {
            metal |= MTL_TEXTURE_USAGE_SHADER_WRITE;
        }
        if self.0 & Self::RENDER_TARGET.0 != 0 {
            metal |= MTL_TEXTURE_USAGE_RENDER_TARGET;
        }
        metal
    }

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for TextureUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Which plane of a combined depth/stencil format a texture holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthPlane {
    /// One texture with the combined depth/stencil format.
    Combined,
    /// The depth plane (`D24`/`D32` data); for a combined Metal format this is
    /// the same handle as [`Self::Combined`].
    Depth,
    /// A stencil-only texture (`X32_Stencil8`, or `X24_Stencil8` on devices
    /// that support 24-bit depth).
    Stencil,
}

/// Descriptor for a private 2D colour or depth/stencil texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    /// Texture width in pixels; `1..=MAX_TEXTURE_DIMENSION_2D`.
    pub width: u32,
    /// Texture height in pixels; `1..=MAX_TEXTURE_DIMENSION_2D`.
    pub height: u32,
    /// Mapped guest format.
    pub format: PixelFormat,
    /// Mips per array slice; `1..=MAX_MIP_LEVELS`.
    pub mip_levels: u32,
    /// Array slices; `1..=MAX_ARRAY_LAYERS`.
    pub array_layers: u32,
    /// Metal usage bits.
    pub usage: TextureUsage,
    /// Depth/stencil plane selection; colour formats require `Combined`.
    pub depth_plane: DepthPlane,
    /// Whether this descriptor is a depth/stencil surface. Set by the
    /// constructors; keeps a colour descriptor from silently accepting a depth
    /// format and vice versa.
    pub depth_stencil: bool,
}

impl TextureDesc {
    /// A sampled colour texture at level 0.
    pub const fn colour(
        width: u32,
        height: u32,
        format: PixelFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            width,
            height,
            format,
            mip_levels: 1,
            array_layers: 1,
            usage,
            depth_plane: DepthPlane::Combined,
            depth_stencil: false,
        }
    }

    /// A depth/stencil target at level 0, sampled and renderable.
    pub const fn depth(
        width: u32,
        height: u32,
        format: PixelFormat,
        plane: DepthPlane,
    ) -> Self {
        Self {
            width,
            height,
            format,
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ.union_const(TextureUsage::RENDER_TARGET),
            depth_plane: plane,
            depth_stencil: true,
        }
    }

    /// Sets the mip count.
    pub const fn with_mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    /// Sets the array layer count.
    pub const fn with_array_layers(mut self, array_layers: u32) -> Self {
        self.array_layers = array_layers;
        self
    }

    /// Replaces the usage bits.
    pub const fn with_usage(mut self, usage: TextureUsage) -> Self {
        self.usage = usage;
        self
    }
}

impl TextureUsage {
    /// Const-friendly union used by the descriptor constructors.
    const fn union_const(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Properties read back from a live `MTLTexture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth in slices (1 for 2D textures).
    pub depth: u32,
    /// Mips per array slice.
    pub mip_levels: u32,
    /// Array slices.
    pub array_layers: u32,
    /// Raw `MTLPixelFormat`.
    pub pixel_format: usize,
    /// Raw `MTLTextureUsage`.
    pub usage: usize,
    /// Raw `MTLStorageMode`.
    pub storage_mode: usize,
    /// Raw `MTLTextureType`.
    pub texture_type: usize,
    /// Sample count.
    pub sample_count: u32,
}

/// View kind, mirroring the D3D12 descriptor families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewKind {
    /// Shader resource (texture read).
    Srv,
    /// Unordered access (texture write).
    Uav,
    /// Render target.
    Rtv,
    /// Depth/stencil target.
    Dsv,
}

/// View dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewDimension {
    /// 1D texture; the base height must be 1.
    Texture1d,
    /// Single 2D slice.
    Texture2d,
    /// 2D array.
    Texture2dArray,
    /// 3D volume; slices index depth.
    Texture3d,
    /// Cube (6 slices per cube).
    Cube,
    /// Cube array (6 slices per cube).
    CubeArray,
}

/// Host-side descriptor for a texture view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceView {
    /// SRV/UAV/RTV/DSV.
    pub kind: ViewKind,
    /// Format the view exposes.
    pub format: PixelFormat,
    /// View dimensionality.
    pub dimension: ViewDimension,
    /// First mip level of the view.
    pub first_mip: u32,
    /// Number of mip levels in the view; at least 1.
    pub mip_count: u32,
    /// First array slice (or depth slice for 3D) of the view.
    pub first_slice: u32,
    /// Number of array slices; at least 1.
    pub slice_count: u32,
}

impl ResourceView {
    /// A single-mip, single-slice view.
    pub const fn new(kind: ViewKind, format: PixelFormat, dimension: ViewDimension) -> Self {
        Self {
            kind,
            format,
            dimension,
            first_mip: 0,
            mip_count: 1,
            first_slice: 0,
            slice_count: 1,
        }
    }

    /// Sets the mip range.
    pub const fn with_mips(mut self, first_mip: u32, mip_count: u32) -> Self {
        self.first_mip = first_mip;
        self.mip_count = mip_count;
        self
    }

    /// Sets the slice range.
    pub const fn with_slices(mut self, first_slice: u32, slice_count: u32) -> Self {
        self.first_slice = first_slice;
        self.slice_count = slice_count;
        self
    }
}

/// A view resolved against a live texture, with the observed Metal format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedView {
    /// Original descriptor.
    pub view: ResourceView,
    /// Raw `MTLPixelFormat` of the texture the view binds.
    pub metal_format: usize,
    /// Width of the first mip in the view.
    pub width: u32,
    /// Height of the first mip in the view.
    pub height: u32,
    /// Depth (slices) of the first mip for 3D views, otherwise 1.
    pub depth: u32,
}

/// Where an upload's source bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadSource {
    /// Page-aligned memory shared with the guest; eligible for a zero-copy
    /// `MTLBuffer` when the range is page-aligned.
    SharedGuestMemory,
    /// Ordinary host memory that must be copied into a staging buffer.
    HostMemory,
}

/// Which upload path a completed upload used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadPath {
    /// The source range was wrapped with `newBufferWithBytesNoCopy` and
    /// blitted directly.
    ZeroCopy,
    /// The rows were copied into a shared staging buffer and blitted.
    StagingBlit,
}

/// One buffer-to-texture upload request.
#[derive(Debug, Clone, Copy)]
pub struct TextureUpload {
    /// Source bytes; valid for `len` reads for the duration of the upload.
    pub pixels: *const u8,
    /// Bytes valid at `pixels`.
    pub len: usize,
    /// Source row pitch in bytes (at least the exact row size).
    pub bytes_per_row: u64,
    /// First destination X inside the mip.
    pub origin_x: u32,
    /// First destination Y inside the mip.
    pub origin_y: u32,
    /// Copied width in pixels.
    pub width: u32,
    /// Copied height in pixels.
    pub height: u32,
    /// Destination mip level.
    pub level: u32,
    /// Destination array slice.
    pub slice: u32,
    /// Source memory kind.
    pub source: UploadSource,
}

impl TextureUpload {
    /// A tightly checked request over ordinary host memory.
    pub fn new(pixels: &[u8], bytes_per_row: u64, width: u32, height: u32) -> Self {
        Self {
            pixels: pixels.as_ptr(),
            len: pixels.len(),
            bytes_per_row,
            origin_x: 0,
            origin_y: 0,
            width,
            height,
            level: 0,
            slice: 0,
            source: UploadSource::HostMemory,
        }
    }

    /// A request over page-aligned memory shared with the guest.
    pub const fn shared_guest(
        pixels: *const u8,
        len: usize,
        bytes_per_row: u64,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            pixels,
            len,
            bytes_per_row,
            origin_x: 0,
            origin_y: 0,
            width,
            height,
            level: 0,
            slice: 0,
            source: UploadSource::SharedGuestMemory,
        }
    }

    /// Sets the destination origin inside the mip.
    pub const fn with_origin(mut self, origin_x: u32, origin_y: u32) -> Self {
        self.origin_x = origin_x;
        self.origin_y = origin_y;
        self
    }

    /// Sets the destination subresource.
    pub const fn with_subresource(mut self, level: u32, slice: u32) -> Self {
        self.level = level;
        self.slice = slice;
        self
    }
}

/// Validated properties of a descriptor that Metal will accept.
#[derive(Debug, Clone, Copy)]
struct ValidatedTexture {
    mtl_format: usize,
    usage_bits: usize,
    mip_levels: u32,
    array_layers: u32,
}

/// Validates a descriptor against the format table and the device.
fn validate_texture_desc(
    desc: &TextureDesc,
    support: FormatSupport,
) -> Result<ValidatedTexture, ResourceError> {
    let format = desc.format;
    if desc.width == 0 || desc.height == 0 {
        return Err(ResourceError::InvalidDescriptor(
            "width and height must be nonzero".to_string(),
        ));
    }
    if desc.width > MAX_TEXTURE_DIMENSION_2D || desc.height > MAX_TEXTURE_DIMENSION_2D {
        return Err(ResourceError::InvalidDescriptor(format!(
            "{}x{} exceeds the {MAX_TEXTURE_DIMENSION_2D} pixel 2D limit",
            desc.width, desc.height
        )));
    }
    if desc.mip_levels == 0 || desc.mip_levels > MAX_MIP_LEVELS {
        return Err(ResourceError::InvalidDescriptor(format!(
            "mip level count {} is outside 1..={MAX_MIP_LEVELS}",
            desc.mip_levels
        )));
    }
    if desc.array_layers == 0 || desc.array_layers > MAX_ARRAY_LAYERS {
        return Err(ResourceError::InvalidDescriptor(format!(
            "array layer count {} is outside 1..={MAX_ARRAY_LAYERS}",
            desc.array_layers
        )));
    }
    if desc.usage.bits() == 0 {
        return Err(ResourceError::InvalidDescriptor(
            "at least one usage bit is required".to_string(),
        ));
    }

    let is_depth = format.is_depth_stencil();
    if is_depth != desc.depth_stencil {
        return Err(ResourceError::InvalidDescriptor(format!(
            "{} descriptor {} a depth/stencil format",
            format.name(),
            if desc.depth_stencil {
                "requires"
            } else {
                "cannot use"
            }
        )));
    }
    if is_depth {
        if desc.usage.contains(TextureUsage::SHADER_WRITE) {
            return Err(ResourceError::InvalidDescriptor(format!(
                "{} is a depth/stencil format and cannot be a UAV",
                format.name()
            )));
        }
        if !desc.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(ResourceError::InvalidDescriptor(
                "depth/stencil textures need render-target usage".to_string(),
            ));
        }
        if desc.depth_plane != DepthPlane::Combined && !format.has_stencil() {
            return Err(ResourceError::InvalidDescriptor(format!(
                "{} has no stencil plane",
                format.name()
            )));
        }
        if desc.depth_plane == DepthPlane::Stencil
            && (desc.usage.contains(TextureUsage::SHADER_READ)
                || desc.usage.contains(TextureUsage::SHADER_WRITE))
        {
            return Err(ResourceError::InvalidDescriptor(format!(
                "the stencil plane of {} cannot be sampled or written by shaders",
                format.name()
            )));
        }
    } else {
        if desc.depth_plane != DepthPlane::Combined {
            return Err(ResourceError::InvalidDescriptor(
                "colour textures have no depth/stencil plane".to_string(),
            ));
        }
    }

    if format.is_block_compressed() {
        if !support.bc_texture_compression {
            return Err(ResourceError::InvalidDescriptor(format!(
                "{} needs BC texture compression, which this device does not support",
                format.name()
            )));
        }
        if !desc.width.is_multiple_of(4) || !desc.height.is_multiple_of(4) {
            return Err(ResourceError::InvalidDescriptor(format!(
                "{} needs width and height multiples of 4 (got {}x{})",
                format.name(),
                desc.width,
                desc.height
            )));
        }
        if desc.usage.contains(TextureUsage::SHADER_WRITE)
            || desc.usage.contains(TextureUsage::RENDER_TARGET)
        {
            return Err(ResourceError::InvalidDescriptor(format!(
                "{} is block-compressed and cannot be a UAV or render target",
                format.name()
            )));
        }
    }

    let mtl_format = match desc.depth_plane {
        DepthPlane::Combined | DepthPlane::Depth => format.mtl_pixel_format_on(support),
        DepthPlane::Stencil => format.stencil_plane_format(support).ok_or_else(|| {
            ResourceError::InvalidDescriptor(format!(
                "{} has no stencil plane format",
                format.name()
            ))
        })?,
    };

    Ok(ValidatedTexture {
        mtl_format,
        usage_bits: desc.usage.bits(),
        mip_levels: desc.mip_levels,
        array_layers: desc.array_layers,
    })
}

impl DirectMetalTransport {
    /// Reports the device properties that change the guest format mapping.
    pub fn format_support(&self) -> FormatSupport {
        let device = self.device_raw();
        // SAFETY: `self.device_raw()` is a live `MTLDevice`; both selectors are
        // declared by Metal to return `BOOL`.
        let bc = unsafe {
            objc::msg_send_u8_0(device, objc::sel(c"supportsBCTextureCompression"))
        } != 0;
        // SAFETY: See above. The property's getter is
        // `isDepth24Stencil8PixelFormatSupported`; the plain name is not a
        // selector and messaging it raises an Objective-C exception.
        let depth24 = unsafe {
            objc::msg_send_u8_0(device, objc::sel(c"isDepth24Stencil8PixelFormatSupported"))
        } != 0;
        FormatSupport {
            bc_texture_compression: bc,
            depth24_stencil8: depth24,
        }
    }

    /// Creates a private 2D colour, depth or stencil texture.
    ///
    /// The descriptor is fully validated before a Metal descriptor exists, so
    /// zero extents, bad mip/array counts, unsupported formats and invalid
    /// usage combinations fail without allocating a texture. Colour and
    /// depth/stencil formats share one entry point because Metal treats a
    /// depth target as a texture with render-target usage.
    pub fn create_render_texture(&self, desc: TextureDesc) -> Result<VmTextureId, ResourceError> {
        let support = self.format_support();
        let validated = validate_texture_desc(&desc, support)?;
        let _pool = AutoreleasePool::new();
        let descriptor = texture_2d_descriptor(
            validated.mtl_format,
            desc.width as usize,
            desc.height as usize,
            validated.mip_levels > 1,
        )?;

        set_descriptor_usize(
            descriptor,
            c"setStorageMode:",
            MTL_STORAGE_MODE_PRIVATE,
        );
        set_descriptor_usize(descriptor, c"setUsage:", validated.usage_bits);
        set_descriptor_usize(
            descriptor,
            c"setMipmapLevelCount:",
            validated.mip_levels as usize,
        );
        if validated.array_layers > 1 {
            set_descriptor_usize(
                descriptor,
                c"setTextureType:",
                MTL_TEXTURE_TYPE_2D_ARRAY,
            );
            set_descriptor_usize(
                descriptor,
                c"setArrayLength:",
                validated.array_layers as usize,
            );
        } else {
            set_descriptor_usize(descriptor, c"setTextureType:", MTL_TEXTURE_TYPE_2D);
        }

        // SAFETY: `self.device_raw()` is a live `MTLDevice`, `descriptor` is a
        // fully configured descriptor owned by this pool, and the result is
        // retained (+1) or null.
        let raw = unsafe {
            objc::msg_send_id_1(
                self.device_raw(),
                objc::sel(c"newTextureWithDescriptor:"),
                descriptor,
            )
        };
        let texture = ObjcObject::from_retained(raw).ok_or_else(|| {
            MetalError::InternalError(format!(
                "newTextureWithDescriptor returned null for {}",
                desc.format.name()
            ))
        })?;

        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.textures.insert(id, texture);
        Ok(VmTextureId(id))
    }

    /// Reads back the Metal properties of a registered texture.
    pub fn texture_properties(&self, texture: VmTextureId) -> Result<TextureInfo, ResourceError> {
        let state = self.lock_state()?;
        let object = state
            .textures
            .get(&texture.0)
            .ok_or(ResourceError::UnknownTexture(texture.0))?;
        texture_info(object.as_raw())
    }

    /// Resolves a view descriptor against a live texture.
    ///
    /// Every range is checked, never clamped: a view whose mip or slice range
    /// reaches past the texture, whose format/kind combination is illegal or
    /// whose usage the texture does not carry is rejected.
    pub fn resolve_view(
        &self,
        texture: VmTextureId,
        view: &ResourceView,
    ) -> Result<ResolvedView, ResourceError> {
        let info = self.texture_properties(texture)?;
        let support = self.format_support();

        if view.mip_count == 0 {
            return Err(ResourceError::InvalidView(
                "mip count must be at least 1".to_string(),
            ));
        }
        let last_mip = view
            .first_mip
            .checked_add(view.mip_count)
            .ok_or_else(|| ResourceError::InvalidView("mip range overflows u32".to_string()))?;
        if last_mip > info.mip_levels {
            return Err(ResourceError::InvalidView(format!(
                "mips {}..{last_mip} exceed the texture's {} mip levels",
                view.first_mip, info.mip_levels
            )));
        }

        if view.slice_count == 0 {
            return Err(ResourceError::InvalidView(
                "slice count must be at least 1".to_string(),
            ));
        }
        let slices_available = if view.dimension == ViewDimension::Texture3d {
            info.depth
        } else {
            info.array_layers
        };
        let last_slice = view
            .first_slice
            .checked_add(view.slice_count)
            .ok_or_else(|| ResourceError::InvalidView("slice range overflows u32".to_string()))?;
        match view.dimension {
            ViewDimension::Texture1d => {
                if info.height != 1 {
                    return Err(ResourceError::InvalidView(format!(
                        "1D views need height 1, texture height is {}",
                        info.height
                    )));
                }
                if view.first_slice != 0 || view.slice_count != 1 {
                    return Err(ResourceError::InvalidView(
                        "1D views address exactly one slice".to_string(),
                    ));
                }
            }
            ViewDimension::Texture2d => {
                if view.first_slice != 0 || view.slice_count != 1 {
                    return Err(ResourceError::InvalidView(
                        "2D views address exactly one slice".to_string(),
                    ));
                }
            }
            ViewDimension::Texture2dArray | ViewDimension::Texture3d => {
                if last_slice > slices_available {
                    return Err(ResourceError::InvalidView(format!(
                        "slices {}..{last_slice} exceed the texture's {slices_available}",
                        view.first_slice
                    )));
                }
            }
            ViewDimension::Cube | ViewDimension::CubeArray => {
                if !slices_available.is_multiple_of(6) {
                    return Err(ResourceError::InvalidView(format!(
                        "cube views need a multiple of 6 slices, texture has {slices_available}"
                    )));
                }
                if !view.slice_count.is_multiple_of(6) {
                    return Err(ResourceError::InvalidView(format!(
                        "cube views need a multiple of 6 slices, view asks for {}",
                        view.slice_count
                    )));
                }
                if last_slice > slices_available {
                    return Err(ResourceError::InvalidView(format!(
                        "slices {}..{last_slice} exceed the texture's {slices_available}",
                        view.first_slice
                    )));
                }
            }
        }

        let actual_is_depth = mtl_pixel_format_is_depth_stencil(info.pixel_format);
        let view_mtl = view.format.mtl_pixel_format_on(support);

        match view.kind {
            ViewKind::Srv => {
                if !texture_usage_has(info.usage, MTL_TEXTURE_USAGE_SHADER_READ) {
                    return Err(ResourceError::InvalidView(
                        "texture was not created with shader-read usage".to_string(),
                    ));
                }
                if !format_compatible(view_mtl, info.pixel_format) {
                    return Err(ResourceError::InvalidView(format!(
                        "{} cannot view a texture stored as Metal format {}",
                        view.format.name(),
                        info.pixel_format
                    )));
                }
            }
            ViewKind::Uav => {
                if actual_is_depth || view.format.is_depth_stencil() {
                    return Err(ResourceError::InvalidView(
                        "depth/stencil textures cannot be UAVs".to_string(),
                    ));
                }
                if view.format.is_block_compressed() {
                    return Err(ResourceError::InvalidView(
                        "block-compressed textures cannot be UAVs".to_string(),
                    ));
                }
                if !texture_usage_has(info.usage, MTL_TEXTURE_USAGE_SHADER_WRITE) {
                    return Err(ResourceError::InvalidView(
                        "texture was not created with shader-write usage".to_string(),
                    ));
                }
                if view_mtl != info.pixel_format {
                    return Err(ResourceError::InvalidView(format!(
                        "UAV format {} (Metal {view_mtl}) must match the texture format ({})",
                        view.format.name(),
                        info.pixel_format
                    )));
                }
            }
            ViewKind::Rtv => {
                if actual_is_depth || view.format.is_depth_stencil() {
                    return Err(ResourceError::InvalidView(
                        "depth/stencil textures need a DSV, not an RTV".to_string(),
                    ));
                }
                if !texture_usage_has(info.usage, MTL_TEXTURE_USAGE_RENDER_TARGET) {
                    return Err(ResourceError::InvalidView(
                        "texture was not created with render-target usage".to_string(),
                    ));
                }
                if !format_compatible(view_mtl, info.pixel_format) {
                    return Err(ResourceError::InvalidView(format!(
                        "{} cannot view a texture stored as Metal format {}",
                        view.format.name(),
                        info.pixel_format
                    )));
                }
            }
            ViewKind::Dsv => {
                if !actual_is_depth || !view.format.is_depth_stencil() {
                    return Err(ResourceError::InvalidView(
                        "DSVs need a depth/stencil texture and format".to_string(),
                    ));
                }
                if !texture_usage_has(info.usage, MTL_TEXTURE_USAGE_RENDER_TARGET) {
                    return Err(ResourceError::InvalidView(
                        "texture was not created with render-target usage".to_string(),
                    ));
                }
                let stencil = view.format.stencil_plane_format(support);
                if view_mtl != info.pixel_format && stencil != Some(info.pixel_format) {
                    return Err(ResourceError::InvalidView(format!(
                        "{} cannot view a texture stored as Metal format {}",
                        view.format.name(),
                        info.pixel_format
                    )));
                }
            }
        }

        let width = (info.width >> view.first_mip).max(1);
        let height = (info.height >> view.first_mip).max(1);
        let depth = if view.dimension == ViewDimension::Texture3d {
            (info.depth >> view.first_mip).max(1)
        } else {
            1
        };
        Ok(ResolvedView {
            view: *view,
            metal_format: info.pixel_format,
            width,
            height,
            depth,
        })
    }

    /// Uploads guest bytes into a colour texture.
    ///
    /// A `SharedGuestMemory` source whose range is page-aligned is wrapped with
    /// `newBufferWithBytesNoCopy` and blitted without copying; anything else is
    /// copied row by row into a shared staging buffer and blitted. Both paths
    /// wait for the command buffer and leave the pixels in the same
    /// [`VmTextureId`].
    ///
    /// # Safety
    ///
    /// `upload.pixels` must be valid for reads of `upload.len` bytes for the
    /// duration of this call and must not be mutated while it runs. When the
    /// source is `SharedGuestMemory` the caller must also keep the range
    /// mapped and alive until this function returns.
    pub unsafe fn upload_texture_2d(
        &self,
        texture: VmTextureId,
        upload: &TextureUpload,
    ) -> Result<UploadPath, ResourceError> {
        let _pool = AutoreleasePool::new();
        if upload.pixels.is_null() || upload.len == 0 {
            return Err(ResourceError::InvalidUpload(
                "upload source pointer is null or empty".to_string(),
            ));
        }
        let info = self.texture_properties(texture)?;
        let format = PixelFormat::from_mtl_pixel_format(info.pixel_format).ok_or_else(|| {
            ResourceError::InvalidUpload(format!(
                "texture pixel format {} has no guest upload layout",
                info.pixel_format
            ))
        })?;
        validate_upload_region(&info, format, upload)?;

        let page = objc::host_page_size();
        let zero_copy = upload.source == UploadSource::SharedGuestMemory
            && (upload.pixels as usize).is_multiple_of(page)
            && upload.len.is_multiple_of(page);
        if zero_copy {
            // SAFETY: the caller guarantees the range is live and page-aligned
            // for the duration of the call; alignment was checked above.
            let buffer = unsafe {
                MetalTransport::register_zerocopy(self, upload.pixels.cast_mut(), upload.len)
            }?;
            let _guard = RegistrationGuard {
                transport: self,
                id: buffer,
            };
            self.blit_upload(texture, buffer, upload, upload.bytes_per_row)?;
            return Ok(UploadPath::ZeroCopy);
        }

        self.upload_through_staging(texture, upload, format)?;
        Ok(UploadPath::StagingBlit)
    }

    /// Copies the upload rows into a shared staging buffer and blits it.
    fn upload_through_staging(
        &self,
        texture: VmTextureId,
        upload: &TextureUpload,
        format: PixelFormat,
    ) -> Result<(), ResourceError> {
        let row_bytes = format.row_bytes(upload.width).ok_or_else(|| {
            ResourceError::InvalidUpload(format!(
                "{} has no row layout for width {}",
                format.name(),
                upload.width
            ))
        })?;
        let pitch = row_bytes
            .max(BLIT_ALIGNMENT)
            .next_multiple_of(STAGING_ROW_ALIGNMENT);
        let staging_len = pitch
            .checked_mul(u64::from(upload.height))
            .ok_or_else(|| ResourceError::InvalidUpload("staging size overflows u64".to_string()))?;
        let staging_len = usize::try_from(staging_len)
            .map_err(|_| ResourceError::InvalidUpload("staging size exceeds host usize".to_string()))?;
        let source_pitch = usize::try_from(upload.bytes_per_row).map_err(|_| {
            ResourceError::InvalidUpload("source row pitch exceeds host usize".to_string())
        })?;
        let pitch = usize::try_from(pitch)
            .map_err(|_| ResourceError::InvalidUpload("staging pitch exceeds host usize".to_string()))?;
        let row_bytes = usize::try_from(row_bytes)
            .map_err(|_| ResourceError::InvalidUpload("row size exceeds host usize".to_string()))?;

        // SAFETY: `self.device_raw()` is a live `MTLDevice`; a nonzero length
        // yields a retained shared `MTLBuffer` or null.
        let raw = unsafe {
            objc::msg_send_new_buffer(
                self.device_raw(),
                staging_len,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        let buffer = ObjcObject::from_retained(raw).ok_or_else(|| {
            MetalError::InternalError(format!(
                "newBufferWithLength returned null for {staging_len} staging bytes"
            ))
        })?;
        // SAFETY: a live shared buffer exposes `contents` for its whole length.
        let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
        if contents.is_null() {
            return Err(ResourceError::Metal(MetalError::InternalError(
                "staging MTLBuffer.contents returned null".to_string(),
            )));
        }
        // SAFETY: the caller guarantees `upload.pixels` is readable for
        // `upload.len` bytes; the region check bounds every row read inside it,
        // and the staging buffer was allocated for `pitch * height` bytes.
        unsafe {
            for row in 0..upload.height as usize {
                let source = upload.pixels.add(row * source_pitch);
                let destination = contents.cast::<u8>().add(row * pitch);
                std::ptr::copy_nonoverlapping(source, destination, row_bytes);
            }
        }

        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.buffers.insert(
            id,
            RegisteredBuffer {
                buffer,
                len: staging_len,
            },
        );
        drop(state);
        let _guard = RegistrationGuard {
            transport: self,
            id: VmBufferId(id),
        };
        self.blit_upload(texture, VmBufferId(id), upload, pitch as u64)
    }

    /// Encodes, commits and waits for one buffer-to-texture copy.
    fn blit_upload(
        &self,
        texture: VmTextureId,
        buffer: VmBufferId,
        upload: &TextureUpload,
        bytes_per_row: u64,
    ) -> Result<(), ResourceError> {
        let bytes_per_row = usize::try_from(bytes_per_row).map_err(|_| {
            ResourceError::InvalidUpload("blit row pitch exceeds host usize".to_string())
        })?;
        let size = MTLSize {
            width: upload.width as usize,
            height: upload.height as usize,
            depth: 1,
        };
        let origin = MTLOrigin {
            x: upload.origin_x as usize,
            y: upload.origin_y as usize,
            z: 0,
        };

        // Hold the registry lock through completion so neither the texture nor
        // the buffer can be released while the GPU reads or writes them.
        let state = self.lock_state()?;
        let texture_object = state
            .textures
            .get(&texture.0)
            .ok_or(ResourceError::UnknownTexture(texture.0))?;
        let buffer_object = state.buffers.get(&buffer.0).ok_or_else(|| {
            ResourceError::Metal(MetalError::NotFound(format!("buffer {}", buffer.0)))
        })?;

        // SAFETY: `queue_raw()` is a live `MTLCommandQueue`; the command buffer
        // is autoreleased in this pool.
        let command_buffer =
            unsafe { objc::msg_send_id_0(self.queue_raw(), objc::sel(c"commandBuffer")) };
        if command_buffer.is_null() {
            return Err(ResourceError::Metal(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            )));
        }
        // SAFETY: `command_buffer` is a live uncommitted command buffer.
        let encoder =
            unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"blitCommandEncoder")) };
        if encoder.is_null() {
            return Err(ResourceError::Metal(MetalError::InternalError(
                "MTLCommandBuffer.blitCommandEncoder returned null".to_string(),
            )));
        }

        // SAFETY: `encoder` is an active blit encoder, both resources are live
        // registry entries held by `state`, the region was validated against
        // the texture and the buffer, and the copy stays inside both.
        unsafe {
            copy_buffer_to_texture(
                encoder,
                buffer_object.buffer.as_raw(),
                0,
                bytes_per_row,
                size,
                texture_object.as_raw(),
                upload.slice,
                upload.level,
                origin,
            );
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"commit"));
            objc::msg_send_void_0(command_buffer, objc::sel(c"waitUntilCompleted"));

            let status = objc::msg_send_usize_0(command_buffer, objc::sel(c"status"));
            if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
                let error = objc::msg_send_id_0(command_buffer, objc::sel(c"error"));
                let message = objc::nserror_message(error);
                return Err(ResourceError::Metal(MetalError::InternalError(format!(
                    "Metal texture upload failed (status={status}): {message}"
                ))));
            }
        }
        drop(state);
        Ok(())
    }
}

/// Unregisters a temporary buffer however the upload ends.
struct RegistrationGuard<'a> {
    transport: &'a DirectMetalTransport,
    id: VmBufferId,
}

impl Drop for RegistrationGuard<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.transport.unregister_buffer(self.id) {
            tracing::warn!(
                ?error,
                buffer = self.id.0,
                "temporary upload buffer was not unregistered"
            );
        }
    }
}

/// Whether a texture's Metal usage includes `bit`.
const fn texture_usage_has(usage: usize, bit: usize) -> bool {
    usage & bit != 0
}

/// Whether a view's Metal format can address a texture of `stored` format.
fn format_compatible(view_mtl: usize, stored: usize) -> bool {
    if view_mtl == stored {
        return true;
    }
    PixelFormat::from_mtl_pixel_format(stored)
        .and_then(PixelFormat::srgb_pair)
        .is_some_and(|pair| pair.mtl_pixel_format() == view_mtl)
}

/// Validates an upload region against a live texture.
fn validate_upload_region(
    info: &TextureInfo,
    format: PixelFormat,
    upload: &TextureUpload,
) -> Result<(), ResourceError> {
    if format.is_depth_stencil() {
        return Err(ResourceError::InvalidUpload(format!(
            "{} is a depth/stencil format; guest bytes are uploaded to colour planes only",
            format.name()
        )));
    }
    if upload.width == 0 || upload.height == 0 {
        return Err(ResourceError::InvalidUpload(
            "upload width and height must be nonzero".to_string(),
        ));
    }
    if upload.level >= info.mip_levels {
        return Err(ResourceError::InvalidUpload(format!(
            "upload level {} exceeds the texture's {} mip levels",
            upload.level, info.mip_levels
        )));
    }
    if info.texture_type == MTL_TEXTURE_TYPE_2D_ARRAY {
        if upload.slice >= info.array_layers {
            return Err(ResourceError::InvalidUpload(format!(
                "upload slice {} exceeds the texture's {} array layers",
                upload.slice, info.array_layers
            )));
        }
    } else if upload.slice != 0 {
        return Err(ResourceError::InvalidUpload(
            "a non-array texture has only slice 0".to_string(),
        ));
    }

    let mip_width = (info.width >> upload.level).max(1);
    let mip_height = (info.height >> upload.level).max(1);
    let end_x = upload
        .origin_x
        .checked_add(upload.width)
        .ok_or_else(|| ResourceError::InvalidUpload("upload X extent overflows u32".to_string()))?;
    let end_y = upload
        .origin_y
        .checked_add(upload.height)
        .ok_or_else(|| ResourceError::InvalidUpload("upload Y extent overflows u32".to_string()))?;
    if end_x > mip_width || end_y > mip_height {
        return Err(ResourceError::InvalidUpload(format!(
            "upload region {}x{} at ({},{}) exceeds mip {} ({}x{})",
            upload.width,
            upload.height,
            upload.origin_x,
            upload.origin_y,
            upload.level,
            mip_width,
            mip_height
        )));
    }

    let row_bytes = format.row_bytes(upload.width).ok_or_else(|| {
        ResourceError::InvalidUpload(format!(
            "{} has no row layout for width {}",
            format.name(),
            upload.width
        ))
    })?;
    if upload.bytes_per_row < row_bytes {
        return Err(ResourceError::InvalidUpload(format!(
            "source row pitch {} is smaller than the exact {row_bytes}-byte row",
            upload.bytes_per_row
        )));
    }
    if !upload.bytes_per_row.is_multiple_of(BLIT_ALIGNMENT) {
        return Err(ResourceError::InvalidUpload(
            "source row pitch must be 4-byte aligned".to_string(),
        ));
    }
    if format.is_block_compressed()
        && (!upload.width.is_multiple_of(4) || !upload.height.is_multiple_of(4))
    {
        return Err(ResourceError::InvalidUpload(format!(
            "{} uploads need width and height multiples of 4 (got {}x{})",
            format.name(),
            upload.width,
            upload.height
        )));
    }

    // The last row is charged exactly `row_bytes`, not a full padded pitch.
    let end = upload
        .bytes_per_row
        .checked_mul(u64::from(upload.height - 1))
        .and_then(|padded_rows| padded_rows.checked_add(row_bytes))
        .ok_or_else(|| ResourceError::InvalidUpload("upload extent overflows u64".to_string()))?;
    if end > upload.len as u64 {
        return Err(ResourceError::InvalidUpload(format!(
            "upload needs {end} bytes but the source range is {} bytes",
            upload.len
        )));
    }
    Ok(())
}

/// Reads the Metal properties of a live `MTLTexture`.
fn texture_info(texture: objc::Id) -> Result<TextureInfo, ResourceError> {
    let read = |selector: &CStr| {
        // SAFETY: `texture` is a live `MTLTexture` and every selector below is
        // an `NSUInteger` property getter.
        unsafe { objc::msg_send_usize_0(texture, objc::sel(selector)) }
    };
    let to_u32 = |value: usize, field: &'static str| {
        u32::try_from(value).map_err(|_| {
            ResourceError::Metal(MetalError::InternalError(format!(
                "texture {field} exceeds u32"
            )))
        })
    };
    Ok(TextureInfo {
        width: to_u32(read(c"width"), "width")?,
        height: to_u32(read(c"height"), "height")?,
        depth: to_u32(read(c"depth"), "depth")?,
        mip_levels: to_u32(read(c"mipmapLevelCount"), "mipmapLevelCount")?,
        array_layers: to_u32(read(c"arrayLength"), "arrayLength")?,
        pixel_format: read(c"pixelFormat"),
        usage: read(c"usage"),
        storage_mode: read(c"storageMode"),
        texture_type: read(c"textureType"),
        sample_count: to_u32(read(c"sampleCount"), "sampleCount")?,
    })
}

/// Creates an autoreleased `MTLTextureDescriptor` for a 2D texture.
fn texture_2d_descriptor(
    pixel_format: usize,
    width: usize,
    height: usize,
    mipmapped: bool,
) -> Result<objc::Id, ResourceError> {
    let class = objc::class(c"MTLTextureDescriptor");
    if class.is_null() {
        return Err(ResourceError::Metal(MetalError::InternalError(
            "MTLTextureDescriptor class unavailable".to_string(),
        )));
    }
    // SAFETY: `class` is the `MTLTextureDescriptor` class, the selector is the
    // documented class factory, and `mipmapped` receives a `BOOL` byte.
    let descriptor = unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, u8) -> objc::Id;
        let factory: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        factory(
            class,
            objc::sel(c"texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            pixel_format,
            width,
            height,
            u8::from(mipmapped),
        )
    };
    if descriptor.is_null() {
        return Err(ResourceError::Metal(MetalError::InternalError(
            "MTLTextureDescriptor creation returned null".to_string(),
        )));
    }
    Ok(descriptor)
}

/// Sends a descriptor setter taking one `NSUInteger`.
fn set_descriptor_usize(descriptor: objc::Id, selector: &CStr, value: usize) {
    // SAFETY: `descriptor` is a live `MTLTextureDescriptor` in the caller's
    // pool and the selector is a documented `NSUInteger` setter.
    unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
        let setter: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        setter(descriptor, objc::sel(selector), value);
    }
}

/// Encodes `copyFromBuffer:...toTexture:...` for a single 2D image.
unsafe fn copy_buffer_to_texture(
    encoder: objc::Id,
    buffer: objc::Id,
    buffer_offset: usize,
    bytes_per_row: usize,
    size: MTLSize,
    texture: objc::Id,
    slice: u32,
    level: u32,
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
    // SAFETY: The caller supplies a live blit encoder and live Metal objects.
    // For a 2D image the per-image stride is 0 and the slice/level indices
    // were validated against the texture.
    let copy: Fn = unsafe { mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: The argument layout matches the selector declaration.
    unsafe {
        copy(
            encoder,
            objc::sel(c"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
            buffer,
            buffer_offset,
            bytes_per_row,
            0,
            size,
            texture,
            slice as usize,
            level as usize,
            origin,
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
    use crate::macos::texture::BufferTextureRegion2d;

    const PAGE_SIZE: usize = 16 * 1024;

    unsafe extern "C" {
        fn posix_memalign(memptr: *mut *mut std::ffi::c_void, alignment: usize, size: usize) -> i32;
        fn free(ptr: *mut std::ffi::c_void);
    }

    /// A page-aligned host allocation usable by `register_zerocopy`.
    struct AlignedBuffer {
        ptr: *mut u8,
        len: usize,
    }

    impl AlignedBuffer {
        fn new(len: usize) -> Self {
            let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `posix_memalign` writes one page-aligned allocation of
            // `len` bytes into `raw`; the alignment is a power of two.
            let rc = unsafe { posix_memalign(&mut raw, PAGE_SIZE, len) };
            assert_eq!(rc, 0, "posix_memalign failed");
            assert!(!raw.is_null(), "posix_memalign returned null");
            Self {
                ptr: raw.cast::<u8>(),
                len,
            }
        }

        fn as_slice(&self) -> &[u8] {
            // SAFETY: the allocation is live for `len` bytes.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: the allocation is live and uniquely borrowed.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    impl Drop for AlignedBuffer {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` came from `posix_memalign` and is not registered.
            unsafe { free(self.ptr.cast()) };
        }
    }

    fn gradient_pixel(x: u32, y: u32) -> [u8; 4] {
        [x as u8, y as u8, (x as u8) ^ (y as u8), 0xFF]
    }

    fn fill_gradient(bytes: &mut [u8], width: u32, height: u32, pitch: usize) {
        for y in 0..height {
            for x in 0..width {
                let base = y as usize * pitch + x as usize * 4;
                bytes[base..base + 4].copy_from_slice(&gradient_pixel(x, y));
            }
        }
    }

    /// Checks every pixel of a 64x64 RGBA8 readback against the gradient.
    fn assert_gradient_roundtrip(dst: &[u8], pitch: usize) {
        let mut checked = 0usize;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let base = y as usize * pitch + x as usize * 4;
                assert_eq!(
                    &dst[base..base + 4],
                    &gradient_pixel(x, y),
                    "pixel ({x},{y}) changed through the private texture"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 64 * 64, "every pixel was compared");
    }

    fn rgba8_upload_texture(transport: &DirectMetalTransport) -> VmTextureId {
        transport
            .create_render_texture(TextureDesc::colour(
                64,
                64,
                PixelFormat::Rgba8Unorm,
                TextureUsage::SHADER_READ,
            ))
            .expect("RGBA8 upload texture creation should succeed")
    }

    fn read_back(
        transport: &DirectMetalTransport,
        texture: VmTextureId,
        dst: &mut AlignedBuffer,
    ) -> VmBufferId {
        dst.as_mut_slice().fill(0xCD);
        // SAFETY: the destination is page-aligned, page-multiple and outlives
        // the synchronous blit.
        let id = unsafe { transport.register_zerocopy(dst.ptr, dst.len) }
            .expect("readback buffer registration should succeed");
        transport
            .blit_texture_to_buffer_2d(
                texture,
                id,
                &BufferTextureRegion2d {
                    buffer_offset: 0,
                    bytes_per_row: 256,
                    origin_x: 0,
                    origin_y: 0,
                    width: 64,
                    height: 64,
                },
            )
            .expect("private texture -> buffer blit should complete");
        id
    }

    #[test]
    fn maps_every_required_guest_format_and_rejects_unknown_values() {
        let required = [
            (61u32, PixelFormat::R8Unorm),
            (49, PixelFormat::Rg8Unorm),
            (28, PixelFormat::Rgba8Unorm),
            (29, PixelFormat::Rgba8UnormSrgb),
            (87, PixelFormat::Bgra8Unorm),
            (91, PixelFormat::Bgra8UnormSrgb),
            (54, PixelFormat::R16Float),
            (34, PixelFormat::Rg16Float),
            (10, PixelFormat::Rgba16Float),
            (41, PixelFormat::R32Float),
            (16, PixelFormat::Rg32Float),
            (2, PixelFormat::Rgba32Float),
            (42, PixelFormat::R32Uint),
            (17, PixelFormat::Rg32Uint),
            (3, PixelFormat::Rgba32Uint),
            (40, PixelFormat::Depth32Float),
            (45, PixelFormat::Depth24UnormStencil8),
            (55, PixelFormat::Depth16Unorm),
            (20, PixelFormat::Depth32FloatStencil8),
            (71, PixelFormat::Bc1Unorm),
            (72, PixelFormat::Bc1UnormSrgb),
            (77, PixelFormat::Bc3Unorm),
            (78, PixelFormat::Bc3UnormSrgb),
            (98, PixelFormat::Bc7Unorm),
            (99, PixelFormat::Bc7UnormSrgb),
        ];
        for (dxgi, expected) in required {
            assert_eq!(
                PixelFormat::from_dxgi(dxgi).expect("DXGI value should be mapped"),
                expected,
                "DXGI format {dxgi}"
            );
            assert_eq!(expected.dxgi(), Some(dxgi), "{} DXGI value", expected.name());
            let roundtrip = PixelFormat::from_mtl_pixel_format(expected.mtl_pixel_format())
                .expect("nominal Metal format should map back");
            assert_eq!(
                roundtrip.mtl_pixel_format(),
                expected.mtl_pixel_format(),
                "{} Metal roundtrip",
                expected.name()
            );
        }

        assert_eq!(PixelFormat::ALL.len(), 27, "mapped format table size");
        assert_eq!(
            PixelFormat::ALL
                .iter()
                .filter(|format| format.dxgi().is_some())
                .count(),
            25,
            "DXGI-backed entries"
        );

        let error = PixelFormat::from_dxgi(0xFFFF).expect_err("0xFFFF must not be mapped");
        assert!(
            error.to_string().contains("65535"),
            "the error must name the rejected value: {error}"
        );
        assert!(PixelFormat::from_dxgi(0).is_err(), "DXGI_FORMAT_UNKNOWN is unmapped");
        assert_eq!(PixelFormat::from_mtl_pixel_format(9999), None);

        assert!(PixelFormat::Depth32Float.is_depth_stencil());
        assert!(!PixelFormat::Rgba8Unorm.is_depth_stencil());
        assert_eq!(PixelFormat::Rgba8Unorm.row_bytes(7), Some(28));
        assert_eq!(PixelFormat::Bc1Unorm.row_bytes(8), Some(16));
        assert_eq!(PixelFormat::Bc1Unorm.row_bytes(6), None);
        assert_eq!(PixelFormat::Bc3Unorm.row_bytes(8), Some(32));
        assert_eq!(PixelFormat::Depth16Unorm.row_bytes(4), None);
    }

    #[test]
    fn depth_planes_and_descriptor_validation_select_the_right_storage() {
        let legacy = FormatSupport {
            bc_texture_compression: true,
            depth24_stencil8: false,
        };
        let native = FormatSupport {
            bc_texture_compression: true,
            depth24_stencil8: true,
        };

        assert_eq!(
            PixelFormat::Depth24UnormStencil8.mtl_pixel_format_on(legacy),
            MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8
        );
        assert_eq!(
            PixelFormat::Depth24UnormStencil8.mtl_pixel_format_on(native),
            MTL_PIXEL_FORMAT_DEPTH24_UNORM_STENCIL8
        );
        assert_eq!(
            PixelFormat::Depth24UnormStencil8.stencil_plane_format(legacy),
            Some(MTL_PIXEL_FORMAT_X32_STENCIL8)
        );
        assert_eq!(
            PixelFormat::Depth24UnormStencil8.stencil_plane_format(native),
            Some(MTL_PIXEL_FORMAT_X24_STENCIL8)
        );
        assert_eq!(
            PixelFormat::Depth32FloatStencil8.stencil_plane_format(legacy),
            Some(MTL_PIXEL_FORMAT_X32_STENCIL8)
        );
        assert_eq!(PixelFormat::Depth16Unorm.stencil_plane_format(legacy), None);
        assert_eq!(PixelFormat::Depth32Float.stencil_plane_format(legacy), None);

        let valid = TextureDesc::colour(16, 16, PixelFormat::Rgba8Unorm, TextureUsage::SHADER_READ);
        assert!(validate_texture_desc(&valid, legacy).is_ok());
        assert!(validate_texture_desc(&valid.with_mip_levels(5), legacy).is_ok());
        assert!(validate_texture_desc(&valid.with_array_layers(3), legacy).is_ok());

        let rejected = [
            TextureDesc::colour(0, 16, PixelFormat::Rgba8Unorm, TextureUsage::SHADER_READ),
            TextureDesc::colour(16, 0, PixelFormat::Rgba8Unorm, TextureUsage::SHADER_READ),
            TextureDesc::colour(
                MAX_TEXTURE_DIMENSION_2D + 1,
                16,
                PixelFormat::Rgba8Unorm,
                TextureUsage::SHADER_READ,
            ),
            valid.with_mip_levels(0),
            valid.with_mip_levels(MAX_MIP_LEVELS + 1),
            valid.with_array_layers(0),
            valid.with_array_layers(MAX_ARRAY_LAYERS + 1),
            TextureDesc::colour(16, 16, PixelFormat::Rgba8Unorm, TextureUsage::default()),
            TextureDesc::colour(16, 16, PixelFormat::Depth32Float, TextureUsage::SHADER_READ),
            TextureDesc::depth(16, 16, PixelFormat::Rgba8Unorm, DepthPlane::Combined),
            TextureDesc::depth(16, 16, PixelFormat::Depth16Unorm, DepthPlane::Stencil),
            TextureDesc::depth(16, 16, PixelFormat::Depth32Float, DepthPlane::Stencil),
            TextureDesc::depth(16, 16, PixelFormat::Depth32Float, DepthPlane::Combined)
                .with_usage(TextureUsage::SHADER_READ),
            TextureDesc::depth(16, 16, PixelFormat::Depth32Float, DepthPlane::Combined)
                .with_usage(TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET),
            TextureDesc::depth(16, 16, PixelFormat::Depth24UnormStencil8, DepthPlane::Stencil),
        ];
        for (index, desc) in rejected.iter().enumerate() {
            assert!(
                validate_texture_desc(desc, legacy).is_err(),
                "rejected descriptor {index} ({:?}) was accepted",
                desc.format
            );
        }

        let bc_rejected = TextureDesc::colour(6, 4, PixelFormat::Bc3Unorm, TextureUsage::SHADER_READ);
        assert!(validate_texture_desc(&bc_rejected, legacy).is_err());
        let bc_unsupported = TextureDesc::colour(8, 8, PixelFormat::Bc7Unorm, TextureUsage::SHADER_READ);
        assert!(
            validate_texture_desc(
                &bc_unsupported,
                FormatSupport {
                    bc_texture_compression: false,
                    depth24_stencil8: false,
                }
            )
            .is_err()
        );

        let depth24 = TextureDesc::depth(8, 8, PixelFormat::Depth24UnormStencil8, DepthPlane::Combined);
        let validated = validate_texture_desc(&depth24, legacy).expect("D24 has a fallback");
        assert_eq!(validated.mtl_format, MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8);
        let stencil = TextureDesc::depth(8, 8, PixelFormat::Depth24UnormStencil8, DepthPlane::Stencil)
            .with_usage(TextureUsage::RENDER_TARGET);
        let validated = validate_texture_desc(&stencil, legacy).expect("D24 stencil plane");
        assert_eq!(validated.mtl_format, MTL_PIXEL_FORMAT_X32_STENCIL8);
        assert_eq!(validated.usage_bits, MTL_TEXTURE_USAGE_RENDER_TARGET);
    }

    #[test]
    fn creates_every_mapped_colour_and_depth_format_with_expected_metal_properties() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let device = transport.device_info().expect("device info should be available");
        let support = transport.format_support();
        println!(
            "device: {} (unified {}, page {} bytes); BC {}, 24-bit depth {}",
            device.name_str(),
            device.has_unified_memory,
            device.page_size,
            support.bc_texture_compression,
            support.depth24_stencil8
        );

        let mut colour = 0usize;
        let mut depth = 0usize;
        let mut stencil_planes = 0usize;
        for format in PixelFormat::ALL {
            if format.is_block_compressed() && !support.bc_texture_compression {
                continue;
            }
            let (width, height) = if format.is_block_compressed() {
                (8, 8)
            } else {
                (16, 8)
            };
            let usage = if format.is_depth_stencil() {
                TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
            } else if format.is_block_compressed() {
                TextureUsage::SHADER_READ
            } else {
                TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET
            };
            let desc = if format.is_depth_stencil() {
                TextureDesc::depth(width, height, format, DepthPlane::Combined)
            } else {
                TextureDesc::colour(width, height, format, usage)
            };

            let texture = transport
                .create_render_texture(desc)
                .unwrap_or_else(|error| panic!("{}: {error}", format.name()));
            let properties = transport
                .texture_properties(texture)
                .unwrap_or_else(|error| panic!("{}: {error}", format.name()));
            assert_eq!(
                properties.pixel_format,
                format.mtl_pixel_format_on(support),
                "{} Metal pixel format",
                format.name()
            );
            assert_eq!(
                mtl_pixel_format_is_depth_stencil(properties.pixel_format),
                format.is_depth_stencil(),
                "{} depth-ness",
                format.name()
            );
            assert_eq!(
                properties.storage_mode, MTL_STORAGE_MODE_PRIVATE,
                "{} must be a private texture",
                format.name()
            );
            assert_eq!(properties.width, width, "{} width", format.name());
            assert_eq!(properties.height, height, "{} height", format.name());
            assert_eq!(properties.mip_levels, 1, "{} mips", format.name());
            assert_eq!(properties.array_layers, 1, "{} layers", format.name());
            let expected_usage = usage.bits();
            assert_eq!(
                properties.usage & expected_usage,
                expected_usage,
                "{} Metal usage bits (observed {})",
                format.name(),
                properties.usage
            );

            if format.is_depth_stencil() {
                depth += 1;
            } else {
                colour += 1;
            }
            transport
                .destroy_texture_2d(texture)
                .expect("texture destruction should succeed");

            if let Some(plane_format) = format.stencil_plane_format(support) {
                let plane = transport
                    .create_render_texture(
                        TextureDesc::depth(8, 8, format, DepthPlane::Stencil)
                            .with_usage(TextureUsage::RENDER_TARGET),
                    )
                    .unwrap_or_else(|error| panic!("{} stencil plane: {error}", format.name()));
                let properties = transport
                    .texture_properties(plane)
                    .expect("stencil plane properties");
                assert_eq!(
                    properties.pixel_format, plane_format,
                    "{} stencil plane Metal format",
                    format.name()
                );
                assert!(mtl_pixel_format_is_depth_stencil(properties.pixel_format));
                assert_eq!(properties.storage_mode, MTL_STORAGE_MODE_PRIVATE);
                transport
                    .destroy_texture_2d(plane)
                    .expect("stencil plane destruction should succeed");
                stencil_planes += 1;
            }
        }

        assert_eq!(colour + depth, PixelFormat::ALL.len(), "every format created");
        assert_eq!(stencil_planes, 2, "D24 and D32 stencil planes created");
        println!(
            "created {} formats ({colour} colour, {depth} depth) and {stencil_planes} stencil planes",
            colour + depth
        );
    }

    #[test]
    fn staging_upload_of_a_64x64_gradient_roundtrips_exactly() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let texture = rgba8_upload_texture(&transport);

        let mut source = vec![0u8; 64 * 256];
        fill_gradient(&mut source, 64, 64, 256);
        // SAFETY: `source` is live and unmodified until the synchronous upload
        // returns. Ordinary host memory takes the staging path.
        let path = unsafe {
            transport.upload_texture_2d(texture, &TextureUpload::new(&source, 256, 64, 64))
        }
        .expect("staging upload should complete");
        assert_eq!(path, UploadPath::StagingBlit);

        let mut destination = AlignedBuffer::new(PAGE_SIZE);
        let dst_id = read_back(&transport, texture, &mut destination);
        assert_gradient_roundtrip(destination.as_slice(), 256);

        transport
            .unregister_buffer(dst_id)
            .expect("readback unregister should succeed");
        transport
            .destroy_texture_2d(texture)
            .expect("texture destruction should succeed");
        println!("staging path: 4096/4096 pixels exact");
    }

    #[test]
    fn zero_copy_upload_of_a_64x64_gradient_roundtrips_exactly() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let texture = rgba8_upload_texture(&transport);

        let mut source = AlignedBuffer::new(PAGE_SIZE);
        fill_gradient(source.as_mut_slice(), 64, 64, 256);
        // SAFETY: the source is page-aligned, page-multiple, live and untouched
        // until the synchronous upload returns.
        let upload = TextureUpload::shared_guest(source.ptr, source.len, 256, 64, 64);
        let path = unsafe { transport.upload_texture_2d(texture, &upload) }
            .expect("zero-copy upload should complete");
        assert_eq!(path, UploadPath::ZeroCopy);

        let mut destination = AlignedBuffer::new(PAGE_SIZE);
        let dst_id = read_back(&transport, texture, &mut destination);
        assert_gradient_roundtrip(destination.as_slice(), 256);

        transport
            .unregister_buffer(dst_id)
            .expect("readback unregister should succeed");
        transport
            .destroy_texture_2d(texture)
            .expect("texture destruction should succeed");
        println!("zero-copy path: 4096/4096 pixels exact");
    }

    #[test]
    fn view_resolution_rejects_out_of_bounds_and_zero_extent_without_allocating() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let texture = transport
            .create_render_texture(
                TextureDesc::colour(
                    16,
                    16,
                    PixelFormat::Rgba8Unorm,
                    TextureUsage::SHADER_READ
                        | TextureUsage::SHADER_WRITE
                        | TextureUsage::RENDER_TARGET,
                )
                .with_mip_levels(4),
            )
            .expect("view test texture creation should succeed");

        let good = ResourceView::new(ViewKind::Srv, PixelFormat::Rgba8Unorm, ViewDimension::Texture2d)
            .with_mips(2, 2);
        let resolved = transport
            .resolve_view(texture, &good)
            .expect("mips 2..4 of a 4-mip texture are in range");
        assert_eq!((resolved.width, resolved.height, resolved.depth), (4, 4, 1));
        assert_eq!(resolved.metal_format, PixelFormat::Rgba8Unorm.mtl_pixel_format());

        let rtv = ResourceView::new(ViewKind::Rtv, PixelFormat::Rgba8UnormSrgb, ViewDimension::Texture2d);
        transport
            .resolve_view(texture, &rtv)
            .expect("an sRGB RTV cast on a linear RGBA8 texture is legal");

        let before = transport
            .lock_state()
            .expect("state lock")
            .textures
            .len();

        let past_last_mip = ResourceView::new(
            ViewKind::Srv,
            PixelFormat::Rgba8Unorm,
            ViewDimension::Texture2d,
        )
        .with_mips(3, 2);
        let error = transport
            .resolve_view(texture, &past_last_mip)
            .expect_err("mips 3..5 of a 4-mip texture must be rejected");
        assert!(error.to_string().contains("mip"), "{error}");

        let past_last_slice = ResourceView::new(
            ViewKind::Srv,
            PixelFormat::Rgba8Unorm,
            ViewDimension::Texture2dArray,
        )
        .with_slices(0, 2);
        let error = transport
            .resolve_view(texture, &past_last_slice)
            .expect_err("2 slices of a 1-layer texture must be rejected");
        assert!(error.to_string().contains("slices"), "{error}");

        let dsv_on_colour = ResourceView::new(
            ViewKind::Dsv,
            PixelFormat::Rgba8Unorm,
            ViewDimension::Texture2d,
        );
        assert!(transport.resolve_view(texture, &dsv_on_colour).is_err());

        let zero_mips = good.with_mips(0, 0);
        assert!(transport.resolve_view(texture, &zero_mips).is_err());

        let zero_width = transport
            .create_render_texture(TextureDesc::colour(
                0,
                8,
                PixelFormat::Rgba8Unorm,
                TextureUsage::SHADER_READ,
            ))
            .expect_err("zero-width creation must be rejected");
        assert!(zero_width.to_string().contains("nonzero"), "{zero_width}");
        let zero_height = transport
            .create_render_texture(TextureDesc::colour(
                8,
                0,
                PixelFormat::Rgba8Unorm,
                TextureUsage::SHADER_READ,
            ))
            .expect_err("zero-height creation must be rejected");
        assert!(zero_height.to_string().contains("nonzero"), "{zero_height}");

        let after = transport
            .lock_state()
            .expect("state lock")
            .textures
            .len();
        assert_eq!(before, after, "rejected views and descriptors allocated nothing");
        println!("rejected view/descriptor paths allocated no texture ({} live)", after);
    }
}
