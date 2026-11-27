//! Host texture and pipeline registry (ADR-013).
//!
//! This module owns host-side registration of Metal textures
//! (2D/2DArray/Cube/CubeArray/3D, mipmaps, BC1-BC7, depth/stencil,
//! R11G11B10F, R10G10B10A2 and RTV/DSV/SRV/UAV views) and content-hashed
//! graphics/mesh/compute pipeline states created from converted metallibs. It
//! never owns a transport or a session, so a future `submit_v3` can hold one of
//! each registry and resolve wire handles through [`TextureRegistry::get`] and
//! [`PipelineRegistry::get`] without a second ownership model.
//!
//! # Textures
//!
//! [`TextureRegistry::register`] validates the complete descriptor, builds an
//! `MTLTextureDescriptor` with `MTLStorageModePrivate`, calls
//! `newTextureWithDescriptor:` and then re-reads the object's attributes from
//! Metal so a silent driver substitution cannot be reported as success. Every
//! descriptor is rejected explicitly: zero extents, impossible dimension/array
//! combinations, unsupported sample counts, block-compressed textures that are
//! not 4x4 aligned, depth formats used as colour attachments and vice versa.
//! Views are created with
//! `newTextureViewWithPixelFormat:textureType:levels:slices:`; the view kind
//! must match the usage the texture was registered with, and every mip/slice
//! range is checked against the source descriptor before Metal sees it.
//!
//! # Pipelines
//!
//! [`PipelineKey::from_parts`] is a BLAKE3 digest over a length-prefixed
//! canonical encoding of the metallib, entry point, stage and caller state. The
//! key keeps the canonical bytes, so duplicate detection compares bytes rather
//! than trusting a 256-bit digest. [`PipelineKey::from_stages`] is the
//! one-library-per-stage form: it hashes every stage's metallib and entry, so a
//! graphics pipeline whose pixel library changed is a different key even when
//! the vertex library is identical. [`PipelineRegistry`] creates at most one
//! Metal pipeline object per key; repeated registration returns the existing
//! handle and counts the reuse.
//!
//! The byte blob passed as `state` is the seam for the D3D12 state translation.
//! It is parsed with [`PipelineState::decode`] for graphics and mesh pipelines
//! (attachment formats, fragment/object entry points, mesh threadgroup budgets
//! and — from `VPS2` on — blend, depth/stencil and rasterizer state) and is
//! opaque for compute. A `VPS1` blob keeps the pre-`VPS2` meaning: every
//! render-state field is its default. A metallib that does not load, an entry
//! point that does not exist, or an entry point whose
//! `MTLFunction.functionType` does not match the requested stage is always an
//! explicit error: the registry never substitutes a different entry point.
//!
//! # Pipeline state versus encoder state
//!
//! Metal bakes per-attachment blending (enable, factors, operations, write
//! mask) and alpha-to-coverage into `MTLRenderPipelineState`, so those
//! [`PipelineState`] fields are applied to the descriptor this module builds.
//! Cull mode, fill mode, front-face winding and depth bias are
//! `MTLRenderCommandEncoder` state and stay on the Cycle 06 encoder path
//! (`macos::state::apply_rasterizer`). Depth/stencil comparison state is a
//! separate cached `MTLDepthStencilState` bound per draw; the encoded values
//! participate in the content key so two pipelines that differ only in depth
//! test or stencil operations are distinct, while `macos::device` keeps binding
//! its existing content-hashed depth state.
//! [`PipelineRegistry::register_graphics_stages`] and
//! [`PipelineRegistry::register_mesh_stages`] take one [`StageLibrary`] per
//! stage, so a persistent compiler that emits one metallib per stage registers
//! without a combined library.

use std::collections::HashMap;
use std::mem;

use crate::abi::VmBufferId;
use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::transport::MetalError;
use crate::wire_v3::{
    BlendAttachment, BlendFactor, BlendOperation, BlendState, CompareFunction, CullMode,
    DepthStencilState, FillMode, FrontFaceWinding, RasterizerState, StencilFaceState,
    StencilOperation,
};

/// `MTLTextureType1D`.
const MTL_TEXTURE_TYPE_1D: usize = 0;
/// `MTLTextureType2D`.
const MTL_TEXTURE_TYPE_2D: usize = 2;
/// `MTLTextureType2DArray`.
const MTL_TEXTURE_TYPE_2D_ARRAY: usize = 3;
/// `MTLTextureType2DMultisample`.
const MTL_TEXTURE_TYPE_2D_MULTISAMPLE: usize = 4;
/// `MTLTextureType2DMultisampleArray`.
const MTL_TEXTURE_TYPE_2D_MULTISAMPLE_ARRAY: usize = 5;
/// `MTLTextureTypeCube`.
const MTL_TEXTURE_TYPE_CUBE: usize = 6;
/// `MTLTextureTypeCubeArray`.
const MTL_TEXTURE_TYPE_CUBE_ARRAY: usize = 7;
/// `MTLTextureType3D`.
const MTL_TEXTURE_TYPE_3D: usize = 8;

/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: usize = 2;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 1;
/// `MTLTextureUsageShaderWrite`.
const MTL_TEXTURE_USAGE_SHADER_WRITE: usize = 2;
/// `MTLTextureUsageRenderTarget` (colour and depth/stencil attachments).
const MTL_TEXTURE_USAGE_RENDER_TARGET: usize = 4;
/// `MTLTextureUsagePixelFormatView`.
const MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW: usize = 16;

/// `MTLLoadActionClear`.
const MTL_LOAD_ACTION_CLEAR: usize = 2;
/// `MTLStoreActionStore`.
const MTL_STORE_ACTION_STORE: usize = 1;

/// `MTLCommandBufferStatusCompleted`.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;

/// `MTLFunctionTypeVertex`.
const MTL_FUNCTION_TYPE_VERTEX: usize = 1;
/// `MTLFunctionTypeFragment`.
const MTL_FUNCTION_TYPE_FRAGMENT: usize = 2;
/// `MTLFunctionTypeKernel`.
const MTL_FUNCTION_TYPE_KERNEL: usize = 3;
/// `MTLFunctionTypeMesh`.
const MTL_FUNCTION_TYPE_MESH: usize = 7;
/// `MTLFunctionTypeObject`.
const MTL_FUNCTION_TYPE_OBJECT: usize = 8;

/// Apple `NSRange` (`NSUInteger location, length`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NSRange {
    location: usize,
    length: usize,
}

/// Apple `MTLOrigin` (`NSUInteger x, y, z`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MTLOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// Apple `MTLClearColor` (four `double` components).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MTLClearColor {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

/// Errors produced by the host registries.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// The texture descriptor is internally inconsistent or unsupported.
    #[error("invalid texture descriptor: {0}")]
    InvalidDesc(&'static str),
    /// The requested view/subresource range is invalid for the texture.
    #[error("invalid subresource: {0}")]
    InvalidSubresource(&'static str),
    /// The operation is not supported by Metal or by this host.
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
    /// A texture id was registered twice.
    #[error("texture id {0} is already registered")]
    DuplicateTexture(u64),
    /// A texture id is not registered.
    #[error("texture id {0} is not registered")]
    UnknownTexture(u64),
    /// A pipeline id is not registered.
    #[error("pipeline id {0} is not registered")]
    UnknownPipeline(u64),
    /// The pipeline key does not describe the supplied bytes and stage.
    #[error("pipeline key does not match its parts")]
    KeyMismatch,
    /// The metallib could not be loaded by Metal.
    #[error("metallib is not parsable: {0}")]
    InvalidMetallib(String),
    /// The named entry point is missing or has the wrong `MTLFunctionType`.
    #[error("entry point `{0}` was not found or has the wrong stage")]
    EntryPointNotFound(String),
    /// A stage's entry point is missing from that stage's own library.
    ///
    /// The stage is named so a per-stage registration failure is diagnosable:
    /// the vertex library may resolve while the pixel library does not.
    #[error(
        "the {stage} stage entry point `{entry}` was not found or has the wrong type in its library"
    )]
    MissingStageFunction {
        /// Stage whose function could not be resolved.
        stage: &'static str,
        /// Entry point the stage asked for.
        entry: String,
    },
    /// The opaque pipeline state blob is malformed.
    #[error("pipeline state blob is invalid: {0}")]
    InvalidState(&'static str),
    /// A Metal object could not be created.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// Texture dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    /// `MTLTextureType1D`.
    D1,
    /// `MTLTextureType2D`.
    D2,
    /// `MTLTextureType2DArray`.
    D2Array,
    /// `MTLTextureTypeCube` (six 2D faces).
    Cube,
    /// `MTLTextureTypeCubeArray` (a multiple of six 2D faces).
    CubeArray,
    /// `MTLTextureType3D`.
    D3,
}

/// Host texture formats mirrored from D3D12/DXGI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// `MTLPixelFormatBC1_RGBA`.
    Bc1Rgba,
    /// `MTLPixelFormatBC2_RGBA`.
    Bc2Rgba,
    /// `MTLPixelFormatBC3_RGBA`.
    Bc3Rgba,
    /// `MTLPixelFormatBC4_RUnorm`.
    Bc4Rgba,
    /// `MTLPixelFormatBC5_RGUnorm`.
    Bc5Rgba,
    /// `MTLPixelFormatBC6H_RGBUfloat`.
    Bc6hRgbUf16,
    /// `MTLPixelFormatBC7_RGBAUnorm`.
    Bc7Rgba,
    /// `MTLPixelFormatRGBA8Unorm`.
    Rgba8Unorm,
    /// `MTLPixelFormatRGBA8Unorm_sRGB`.
    Rgba8Srgb,
    /// `MTLPixelFormatBGRA8Unorm`.
    Bgra8Unorm,
    /// `MTLPixelFormatRGBA16Float` (DXGI_FORMAT_R16G16B16A16_FLOAT, the
    /// title's FP16 present/render target).
    Rgba16Float,
    /// `MTLPixelFormatRG11B10Float`.
    Rg11g11b10Float,
    /// `MTLPixelFormatRGB10A2Unorm`.
    Rgb10a2Unorm,
    /// `MTLPixelFormatR32Float`.
    R32Float,
    /// `MTLPixelFormatR16Float`.
    R16Float,
    /// `MTLPixelFormatDepth32Float`.
    Depth32Float,
    /// `MTLPixelFormatDepth24Unorm_Stencil8`.
    Depth24Stencil8,
    /// `MTLPixelFormatDepth32Float_Stencil8`.
    Depth32FloatStencil8,
}

impl TextureFormat {
    /// Exact `MTLPixelFormat` value for this host format.
    pub const fn mtl_pixel_format(self) -> usize {
        match self {
            Self::Bc1Rgba => 130,
            Self::Bc2Rgba => 132,
            Self::Bc3Rgba => 134,
            Self::Bc4Rgba => 140,
            Self::Bc5Rgba => 142,
            Self::Bc6hRgbUf16 => 151,
            Self::Bc7Rgba => 152,
            Self::Rgba8Unorm => 70,
            Self::Rgba8Srgb => 71,
            Self::Bgra8Unorm => 80,
            Self::Rgba16Float => 115,
            Self::Rg11g11b10Float => 92,
            Self::Rgb10a2Unorm => 90,
            Self::R32Float => 55,
            Self::R16Float => 25,
            Self::Depth32Float => 252,
            Self::Depth24Stencil8 => 255,
            Self::Depth32FloatStencil8 => 260,
        }
    }

    /// Whether this format is a depth or depth/stencil format.
    pub const fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth32Float | Self::Depth24Stencil8 | Self::Depth32FloatStencil8
        )
    }

    /// Whether this format is block-compressed.
    pub const fn is_block_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1Rgba
                | Self::Bc2Rgba
                | Self::Bc3Rgba
                | Self::Bc4Rgba
                | Self::Bc5Rgba
                | Self::Bc6hRgbUf16
                | Self::Bc7Rgba
        )
    }

    /// Whether this format may back a colour render target.
    pub const fn is_color_render_target(self) -> bool {
        !self.is_depth_stencil() && !self.is_block_compressed()
    }

    /// Stable one-byte code used by [`PipelineState`] encodings.
    pub const fn code(self) -> u8 {
        match self {
            Self::Bc1Rgba => 0,
            Self::Bc2Rgba => 1,
            Self::Bc3Rgba => 2,
            Self::Bc4Rgba => 3,
            Self::Bc5Rgba => 4,
            Self::Bc6hRgbUf16 => 5,
            Self::Bc7Rgba => 6,
            Self::Rgba8Unorm => 7,
            Self::Rgba8Srgb => 8,
            Self::Bgra8Unorm => 9,
            Self::Rgba16Float => 17,
            Self::Rg11g11b10Float => 10,
            Self::Rgb10a2Unorm => 11,
            Self::R32Float => 12,
            Self::R16Float => 13,
            Self::Depth32Float => 14,
            Self::Depth24Stencil8 => 15,
            Self::Depth32FloatStencil8 => 16,
        }
    }

    /// Decodes a stable one-byte format code.
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Bc1Rgba,
            1 => Self::Bc2Rgba,
            2 => Self::Bc3Rgba,
            3 => Self::Bc4Rgba,
            4 => Self::Bc5Rgba,
            5 => Self::Bc6hRgbUf16,
            6 => Self::Bc7Rgba,
            7 => Self::Rgba8Unorm,
            8 => Self::Rgba8Srgb,
            9 => Self::Bgra8Unorm,
            17 => Self::Rgba16Float,
            10 => Self::Rg11g11b10Float,
            11 => Self::Rgb10a2Unorm,
            12 => Self::R32Float,
            13 => Self::R16Float,
            14 => Self::Depth32Float,
            15 => Self::Depth24Stencil8,
            16 => Self::Depth32FloatStencil8,
            _ => return None,
        })
    }

    const fn is_rgba8_storage(self) -> bool {
        matches!(self, Self::Rgba8Unorm | Self::Bgra8Unorm)
    }
}

/// Texture usage flags.
///
/// `depth_stencil` maps to Metal's single `MTLTextureUsageRenderTarget` bit; it
/// is tracked separately here so the descriptor validation can reject a depth
/// format registered as a colour target and a colour format registered as a
/// depth target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureUsage {
    /// The texture is read by a shader (`MTLTextureUsageShaderRead`).
    pub shader_read: bool,
    /// The texture is written by a shader (`MTLTextureUsageShaderWrite`).
    pub shader_write: bool,
    /// The texture is a colour render target (`MTLTextureUsageRenderTarget`).
    pub render_target: bool,
    /// The texture is a depth/stencil attachment (`MTLTextureUsageRenderTarget`).
    pub depth_stencil: bool,
}

impl TextureUsage {
    /// Shader-read-only usage, the common sampled-texture case.
    pub const fn sampled() -> Self {
        Self {
            shader_read: true,
            shader_write: false,
            render_target: false,
            depth_stencil: false,
        }
    }

    /// Colour render target usage (RTV plus optional sampling).
    pub const fn color_attachment() -> Self {
        Self {
            shader_read: true,
            shader_write: false,
            render_target: true,
            depth_stencil: false,
        }
    }

    /// Depth/stencil attachment usage (DSV plus optional sampling).
    pub const fn depth_attachment() -> Self {
        Self {
            shader_read: true,
            shader_write: false,
            render_target: false,
            depth_stencil: true,
        }
    }

    const fn is_empty(self) -> bool {
        !self.shader_read && !self.shader_write && !self.render_target && !self.depth_stencil
    }

    const fn mtl_bits(self) -> usize {
        let mut bits = MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;
        if self.shader_read {
            bits |= MTL_TEXTURE_USAGE_SHADER_READ;
        }
        if self.shader_write {
            bits |= MTL_TEXTURE_USAGE_SHADER_WRITE;
        }
        if self.render_target || self.depth_stencil {
            bits |= MTL_TEXTURE_USAGE_RENDER_TARGET;
        }
        bits
    }
}

/// Complete host texture description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureDesc {
    /// Width in texels (or blocks for block-compressed formats).
    pub width: u32,
    /// Height in texels (or blocks for block-compressed formats).
    pub height: u32,
    /// Depth in slices for [`TextureDimension::D3`]; `1` otherwise.
    pub depth: u32,
    /// Mipmap level count, at least `1`.
    pub mip_levels: u32,
    /// Sample count: `1`, `2`, `4` or `8`.
    pub sample_count: u32,
    /// Array element count; `1` for non-array dimensions and `6` (or a
    /// multiple of `6`) for cube dimensions.
    pub array_layers: u32,
    /// Texture dimensionality.
    pub dimension: TextureDimension,
    /// Pixel format.
    pub format: TextureFormat,
    /// Usage flags.
    pub usage: TextureUsage,
    /// Optional host-side label.
    ///
    /// `MTLTextureDescriptor` exposes no public `label` property on this SDK,
    /// so the label is stored in the registry entry (and passed to the
    /// descriptor only when the runtime implements `setLabel:`). It is never
    /// silently dropped from the descriptor metadata reported by
    /// [`TextureRegistry::get`].
    pub label: Option<String>,
}

impl TextureDesc {
    /// Validates every descriptor field against Metal's texture rules.
    pub fn validate(&self) -> Result<(), RegistryError> {
        use TextureDimension::{Cube, CubeArray, D1, D2, D2Array, D3};
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err(RegistryError::InvalidDesc(
                "width, height and depth must be nonzero",
            ));
        }
        if self.width > crate::macos::texture::MAX_TEXTURE_DIMENSION_2D
            || self.height > crate::macos::texture::MAX_TEXTURE_DIMENSION_2D
        {
            return Err(RegistryError::InvalidDesc(
                "texture exceeds the supported 2D dimension limit",
            ));
        }
        if self.mip_levels == 0 {
            return Err(RegistryError::InvalidDesc("mip_levels must be at least 1"));
        }
        if self.array_layers == 0 {
            return Err(RegistryError::InvalidDesc(
                "array_layers must be at least 1",
            ));
        }
        if !matches!(self.sample_count, 1 | 2 | 4 | 8) {
            return Err(RegistryError::InvalidDesc(
                "sample_count must be 1, 2, 4 or 8",
            ));
        }
        if self.usage.is_empty() {
            return Err(RegistryError::InvalidDesc(
                "at least one usage flag must be set",
            ));
        }

        match self.dimension {
            D1 => {
                if self.height != 1 || self.depth != 1 || self.array_layers != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "1D textures must have height = depth = array_layers = 1",
                    ));
                }
                if self.sample_count != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "1D textures cannot be multisampled",
                    ));
                }
                if self.usage.render_target || self.usage.depth_stencil {
                    return Err(RegistryError::InvalidDesc(
                        "1D textures cannot be render or depth attachments",
                    ));
                }
            }
            D2 => {
                if self.depth != 1 || self.array_layers != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "2D textures must have depth = array_layers = 1",
                    ));
                }
            }
            D2Array => {
                if self.depth != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "2D array textures must have depth = 1",
                    ));
                }
            }
            Cube => {
                if self.width != self.height || self.depth != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "cube textures must be square with depth = 1",
                    ));
                }
                if self.array_layers != 6 {
                    return Err(RegistryError::InvalidDesc(
                        "cube textures must declare exactly 6 array layers",
                    ));
                }
                if self.sample_count != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "cube textures cannot be multisampled",
                    ));
                }
            }
            CubeArray => {
                if self.width != self.height || self.depth != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "cube array textures must be square with depth = 1",
                    ));
                }
                if self.array_layers == 0 || !self.array_layers.is_multiple_of(6) {
                    return Err(RegistryError::InvalidDesc(
                        "cube array textures need a nonzero multiple of 6 array layers",
                    ));
                }
                if self.sample_count != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "cube array textures cannot be multisampled",
                    ));
                }
            }
            D3 => {
                if self.array_layers != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "3D textures must declare exactly 1 array layer",
                    ));
                }
                if self.sample_count != 1 {
                    return Err(RegistryError::InvalidDesc(
                        "3D textures cannot be multisampled",
                    ));
                }
                if self.usage.render_target || self.usage.depth_stencil {
                    return Err(RegistryError::InvalidDesc(
                        "3D textures cannot be render or depth attachments here",
                    ));
                }
            }
        }

        if self.sample_count > 1 {
            if self.mip_levels != 1 {
                return Err(RegistryError::InvalidDesc(
                    "multisampled textures cannot have mipmaps",
                ));
            }
            if !matches!(self.dimension, D2 | D2Array) {
                return Err(RegistryError::InvalidDesc(
                    "only 2D and 2D array textures can be multisampled",
                ));
            }
        }

        if self.format.is_depth_stencil() {
            if self.usage.render_target {
                return Err(RegistryError::InvalidDesc(
                    "a depth format cannot be registered as a colour render target",
                ));
            }
            if self.usage.shader_write {
                return Err(RegistryError::InvalidDesc(
                    "depth formats cannot be shader-written on this path",
                ));
            }
            if !matches!(self.dimension, D2 | D2Array) {
                return Err(RegistryError::InvalidDesc(
                    "depth textures must be 2D or 2D arrays",
                ));
            }
        } else if self.usage.depth_stencil {
            return Err(RegistryError::InvalidDesc(
                "a colour format cannot be registered as a depth attachment",
            ));
        }

        if self.format.is_block_compressed() {
            if self.usage.shader_write || self.usage.render_target || self.usage.depth_stencil {
                return Err(RegistryError::InvalidDesc(
                    "block-compressed textures are read-only samples",
                ));
            }
            if self.sample_count != 1 {
                return Err(RegistryError::InvalidDesc(
                    "block-compressed textures cannot be multisampled",
                ));
            }
            if matches!(self.dimension, D1) {
                return Err(RegistryError::InvalidDesc(
                    "block-compressed textures must be 2D or cube",
                ));
            }
            if !self.width.is_multiple_of(4) || !self.height.is_multiple_of(4) {
                return Err(RegistryError::InvalidDesc(
                    "block-compressed textures need 4x4-aligned extents",
                ));
            }
        }

        if self.mip_levels > self.max_mip_levels() {
            return Err(RegistryError::InvalidDesc(
                "mip_levels exceeds the texture's maximum mip chain",
            ));
        }
        Ok(())
    }

    /// Largest legal mip count for the declared extents.
    pub fn max_mip_levels(&self) -> u32 {
        let widest = self.width.max(self.height).max(self.depth).max(1);
        32 - widest.leading_zeros()
    }

    const fn mtl_texture_type(&self) -> usize {
        match self.dimension {
            TextureDimension::D1 => MTL_TEXTURE_TYPE_1D,
            TextureDimension::D2 if self.sample_count > 1 => MTL_TEXTURE_TYPE_2D_MULTISAMPLE,
            TextureDimension::D2 => MTL_TEXTURE_TYPE_2D,
            TextureDimension::D2Array if self.sample_count > 1 => {
                MTL_TEXTURE_TYPE_2D_MULTISAMPLE_ARRAY
            }
            TextureDimension::D2Array => MTL_TEXTURE_TYPE_2D_ARRAY,
            TextureDimension::Cube => MTL_TEXTURE_TYPE_CUBE,
            TextureDimension::CubeArray => MTL_TEXTURE_TYPE_CUBE_ARRAY,
            TextureDimension::D3 => MTL_TEXTURE_TYPE_3D,
        }
    }
}

/// Kind of view to create over a registered texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureViewKind {
    /// Colour render target view (`MTLTextureUsageRenderTarget`).
    Rtv,
    /// Depth/stencil view (`MTLTextureUsageRenderTarget`).
    Dsv,
    /// Shader resource view (`MTLTextureUsageShaderRead`).
    Srv,
    /// Unordered access view (`MTLTextureUsageShaderWrite`).
    Uav,
}

impl TextureViewKind {
    /// Stable name used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rtv => "RTV",
            Self::Dsv => "DSV",
            Self::Srv => "SRV",
            Self::Uav => "UAV",
        }
    }
}

/// Mip and array-slice range for one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSubresource {
    /// First mip level.
    pub base_mip_level: u32,
    /// Number of mip levels (at least 1).
    pub mip_level_count: u32,
    /// First array layer (0 for non-array textures).
    pub base_array_layer: u32,
    /// Number of array layers (at least 1).
    pub array_layer_count: u32,
}

impl TextureSubresource {
    /// One mip level of one array slice.
    pub const fn single(mip_level: u32, array_layer: u32) -> Self {
        Self {
            base_mip_level: mip_level,
            mip_level_count: 1,
            base_array_layer: array_layer,
            array_layer_count: 1,
        }
    }

    /// Every mip level of one array slice.
    pub const fn all_mips_of(mip_levels: u32, array_layer: u32) -> Self {
        Self {
            base_mip_level: 0,
            mip_level_count: mip_levels,
            base_array_layer: array_layer,
            array_layer_count: 1,
        }
    }
}

/// A validated view creation plan, produced before any Metal call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewPlan {
    pixel_format: usize,
    texture_type: usize,
    levels: NSRange,
    slices: NSRange,
}

impl ViewPlan {
    fn resolve(
        desc: &TextureDesc,
        kind: TextureViewKind,
        sub: TextureSubresource,
    ) -> Result<Self, RegistryError> {
        use TextureDimension::{Cube, CubeArray, D1, D2, D2Array, D3};
        if sub.mip_level_count == 0 || sub.array_layer_count == 0 {
            return Err(RegistryError::InvalidSubresource(
                "mip_level_count and array_layer_count must be at least 1",
            ));
        }
        let mip_end = sub
            .base_mip_level
            .checked_add(sub.mip_level_count)
            .ok_or(RegistryError::InvalidSubresource("mip range overflows"))?;
        if mip_end > desc.mip_levels {
            return Err(RegistryError::InvalidSubresource(
                "mip range exceeds the texture's mip levels",
            ));
        }
        let layer_end = sub
            .base_array_layer
            .checked_add(sub.array_layer_count)
            .ok_or(RegistryError::InvalidSubresource("array range overflows"))?;
        if layer_end > desc.array_layers {
            return Err(RegistryError::InvalidSubresource(
                "array range exceeds the texture's array layers",
            ));
        }
        if matches!(desc.dimension, D3) && (sub.base_array_layer != 0 || sub.array_layer_count != 1)
        {
            return Err(RegistryError::InvalidSubresource(
                "3D views use one slice range and no array layers",
            ));
        }

        let texture_type = match kind {
            TextureViewKind::Rtv | TextureViewKind::Dsv => {
                if sub.mip_level_count != 1 || sub.array_layer_count != 1 {
                    return Err(RegistryError::InvalidSubresource(
                        "render/depth views cover exactly one mip and one array layer",
                    ));
                }
                if matches!(desc.dimension, D1 | D3) {
                    return Err(RegistryError::InvalidSubresource(
                        "render/depth views require a 2D, array or cube texture",
                    ));
                }
                MTL_TEXTURE_TYPE_2D
            }
            TextureViewKind::Srv => match desc.dimension {
                D1 => MTL_TEXTURE_TYPE_1D,
                D2 => MTL_TEXTURE_TYPE_2D,
                D2Array => {
                    if sub.array_layer_count == 1 {
                        MTL_TEXTURE_TYPE_2D
                    } else {
                        MTL_TEXTURE_TYPE_2D_ARRAY
                    }
                }
                Cube => {
                    if sub.array_layer_count == 6 {
                        MTL_TEXTURE_TYPE_CUBE
                    } else if sub.array_layer_count == 1 {
                        MTL_TEXTURE_TYPE_2D
                    } else {
                        return Err(RegistryError::InvalidSubresource(
                            "cube SRV views need 1 face or all 6 faces",
                        ));
                    }
                }
                CubeArray => {
                    if sub.array_layer_count.is_multiple_of(6) {
                        MTL_TEXTURE_TYPE_CUBE_ARRAY
                    } else {
                        return Err(RegistryError::InvalidSubresource(
                            "cube array SRV views need a multiple of 6 layers",
                        ));
                    }
                }
                D3 => MTL_TEXTURE_TYPE_3D,
            },
            TextureViewKind::Uav => {
                if sub.mip_level_count != 1 {
                    return Err(RegistryError::InvalidSubresource(
                        "UAV views cover exactly one mip level",
                    ));
                }
                match desc.dimension {
                    D2 => MTL_TEXTURE_TYPE_2D,
                    D2Array => {
                        if sub.array_layer_count == 1 {
                            MTL_TEXTURE_TYPE_2D
                        } else {
                            MTL_TEXTURE_TYPE_2D_ARRAY
                        }
                    }
                    D3 => MTL_TEXTURE_TYPE_3D,
                    D1 | Cube | CubeArray => {
                        return Err(RegistryError::InvalidSubresource(
                            "UAV views require a 2D, 2D array or 3D texture",
                        ));
                    }
                }
            }
        };

        match kind {
            TextureViewKind::Rtv => {
                if !desc.usage.render_target {
                    return Err(RegistryError::Unsupported(
                        "texture was not registered with render_target usage",
                    ));
                }
                if !desc.format.is_color_render_target() {
                    return Err(RegistryError::Unsupported(
                        "RTV views require a colour, non-block-compressed format",
                    ));
                }
            }
            TextureViewKind::Dsv => {
                if !desc.usage.depth_stencil {
                    return Err(RegistryError::Unsupported(
                        "texture was not registered with depth_stencil usage",
                    ));
                }
                if !desc.format.is_depth_stencil() {
                    return Err(RegistryError::Unsupported(
                        "DSV views require a depth or depth/stencil format",
                    ));
                }
            }
            TextureViewKind::Srv => {
                if !desc.usage.shader_read {
                    return Err(RegistryError::Unsupported(
                        "texture was not registered with shader_read usage",
                    ));
                }
            }
            TextureViewKind::Uav => {
                if !desc.usage.shader_write {
                    return Err(RegistryError::Unsupported(
                        "texture was not registered with shader_write usage",
                    ));
                }
                if desc.format.is_depth_stencil() {
                    return Err(RegistryError::Unsupported(
                        "depth formats cannot be UAVs on this path",
                    ));
                }
            }
        }

        Ok(Self {
            pixel_format: desc.format.mtl_pixel_format(),
            texture_type,
            levels: NSRange {
                location: sub.base_mip_level as usize,
                length: sub.mip_level_count as usize,
            },
            slices: NSRange {
                location: sub.base_array_layer as usize,
                length: sub.array_layer_count as usize,
            },
        })
    }
}

/// Registered texture plus its validated description.
pub struct TextureEntry {
    texture: ObjcObject,
    desc: TextureDesc,
}

impl TextureEntry {
    /// Descriptor this texture was registered with.
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    /// Retained raw `MTLTexture`.
    pub(crate) fn raw(&self) -> objc::Id {
        self.texture.as_raw()
    }
}

/// A retained `MTLTexture` view over part of a registered texture.
pub struct TextureView {
    texture: ObjcObject,
    source_id: u64,
    kind: TextureViewKind,
    format: TextureFormat,
    subresource: TextureSubresource,
}

impl TextureView {
    /// Id of the texture this view was created from.
    pub fn source_id(&self) -> u64 {
        self.source_id
    }

    /// View kind.
    pub fn kind(&self) -> TextureViewKind {
        self.kind
    }

    /// Pixel format of the view (currently always the source format).
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Mip/slice range of the view.
    pub fn subresource(&self) -> TextureSubresource {
        self.subresource
    }

    pub(crate) fn raw(&self) -> objc::Id {
        self.texture.as_raw()
    }
}

/// Host texture registry: one `MTLTexture` per caller-supplied id.
pub struct TextureRegistry {
    device: ObjcObject,
    queue: ObjcObject,
    textures: HashMap<u64, TextureEntry>,
}

impl TextureRegistry {
    /// Retains the transport's device and queue and starts an empty registry.
    pub fn new(transport: &DirectMetalTransport) -> Result<Self, RegistryError> {
        Ok(Self {
            device: retain(transport.device_raw())?,
            queue: retain(transport.queue_raw())?,
            textures: HashMap::new(),
        })
    }

    /// Registers `id` as a real private `MTLTexture`.
    ///
    /// Duplicate ids are rejected before Metal sees the descriptor. The created
    /// object's type, pixel format, extents, mip count, sample count and array
    /// length are read back and compared with the request.
    pub fn register(&mut self, id: u64, desc: TextureDesc) -> Result<(), RegistryError> {
        if id == 0 {
            return Err(RegistryError::InvalidDesc("texture id 0 is reserved"));
        }
        if self.textures.contains_key(&id) {
            return Err(RegistryError::DuplicateTexture(id));
        }
        desc.validate()?;

        let _pool = AutoreleasePool::new();
        let descriptor = new_texture_descriptor(&desc)?;
        let raw = unsafe {
            objc::msg_send_id_1(
                self.device.as_raw(),
                objc::sel(c"newTextureWithDescriptor:"),
                descriptor.as_raw(),
            )
        };
        let texture = ObjcObject::from_retained(raw).ok_or_else(|| {
            RegistryError::Metal(MetalError::InternalError(
                "MTLDevice.newTextureWithDescriptor returned null".to_string(),
            ))
        })?;

        verify_texture_attributes(&texture, &desc)?;
        self.textures.insert(id, TextureEntry { texture, desc });
        Ok(())
    }

    /// Releases the texture registered under `id`.
    pub fn destroy(&mut self, id: u64) -> Result<(), RegistryError> {
        self.textures
            .remove(&id)
            .map(|_| ())
            .ok_or(RegistryError::UnknownTexture(id))
    }

    /// Borrows the entry registered under `id`.
    pub fn get(&self, id: u64) -> Option<&TextureEntry> {
        self.textures.get(&id)
    }

    /// Number of live textures.
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Whether the registry holds no textures.
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }

    /// Raw retained `MTLTexture` for `id` (crate-internal consumers only).
    pub(crate) fn raw(&self, id: u64) -> Option<objc::Id> {
        self.get(id).map(TextureEntry::raw)
    }

    /// Creates a typed view over `subresource` of the texture registered at `id`.
    ///
    /// The kind must match the usage flags the texture was registered with and
    /// the range must fit the source mip chain and array layers. Every failure
    /// is reported before `newTextureViewWithPixelFormat:` is called.
    pub fn create_view(
        &self,
        id: u64,
        kind: TextureViewKind,
        subresource: TextureSubresource,
    ) -> Result<TextureView, RegistryError> {
        let entry = self.get(id).ok_or(RegistryError::UnknownTexture(id))?;
        let plan = ViewPlan::resolve(&entry.desc, kind, subresource)?;
        let _pool = AutoreleasePool::new();
        let raw = unsafe { new_texture_view(entry.raw(), plan) };
        let texture = ObjcObject::from_retained(raw).ok_or_else(|| {
            RegistryError::Metal(MetalError::InternalError(format!(
                "newTextureViewWithPixelFormat returned null for texture {id}"
            )))
        })?;
        Ok(TextureView {
            texture,
            source_id: id,
            kind,
            format: entry.desc.format,
            subresource,
        })
    }

    /// Clears a colour view with a render pass and waits for completion.
    ///
    /// This is a host-side diagnostic that proves the registered texture and
    /// its RTV view are real Metal objects; it is never part of the frame path.
    pub fn clear_color(&self, view: &TextureView, color: [f64; 4]) -> Result<(), RegistryError> {
        if view.kind != TextureViewKind::Rtv {
            return Err(RegistryError::Unsupported(
                "clear_color requires an RTV view",
            ));
        }
        let _pool = AutoreleasePool::new();
        let descriptor = new_render_pass_descriptor(view, color)?;
        let cb = self.new_command_buffer()?;
        let encoder = unsafe {
            objc::msg_send_id_1(
                cb,
                objc::sel(c"renderCommandEncoderWithDescriptor:"),
                descriptor,
            )
        };
        if encoder.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "renderCommandEncoderWithDescriptor: returned null".to_string(),
            )));
        }
        unsafe {
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
        check_completion(cb, "texture clear")
    }

    /// Reads a level-0, slice-0 RGBA8/BGRA8 view back through a blit.
    ///
    /// The returned texels are in the texture's byte order: `[r, g, b, a]` for
    /// RGBA8 and `[b, g, r, a]` for BGRA8. The view must start at mip 0,
    /// layer 0 and cover one mip and one slice.
    pub fn read_view_rgba8(&self, view: &TextureView) -> Result<Vec<[u8; 4]>, RegistryError> {
        if !view.format.is_rgba8_storage() {
            return Err(RegistryError::Unsupported(
                "readback requires an RGBA8 or BGRA8 view",
            ));
        }
        if view.subresource.base_mip_level != 0
            || view.subresource.mip_level_count != 1
            || view.subresource.base_array_layer != 0
            || view.subresource.array_layer_count != 1
        {
            return Err(RegistryError::Unsupported(
                "readback requires mip 0 of array layer 0",
            ));
        }
        let entry = self
            .get(view.source_id)
            .ok_or(RegistryError::UnknownTexture(view.source_id))?;
        let width = entry.desc.width as usize;
        let height = entry.desc.height as usize;
        let bytes_per_row = width
            .checked_mul(4)
            .ok_or(RegistryError::InvalidDesc("row length overflows usize"))?;
        let len = bytes_per_row
            .checked_mul(height)
            .ok_or(RegistryError::InvalidDesc("texture length overflows usize"))?;

        let _pool = AutoreleasePool::new();
        let buffer_raw = unsafe { objc::msg_send_new_buffer(self.device.as_raw(), len, 0) };
        let buffer = ObjcObject::from_retained(buffer_raw).ok_or_else(|| {
            RegistryError::Metal(MetalError::InternalError(
                "readback MTLBuffer allocation returned null".to_string(),
            ))
        })?;
        let cb = self.new_command_buffer()?;
        let encoder = unsafe { objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder")) };
        if encoder.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "blitCommandEncoder returned null".to_string(),
            )));
        }
        unsafe {
            copy_texture_to_buffer(
                encoder,
                view.raw(),
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width,
                    height,
                    depth: 1,
                },
                buffer.as_raw(),
                0,
                bytes_per_row,
            );
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
        check_completion(cb, "texture readback")?;

        let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
        if contents.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "readback buffer has no CPU-visible contents".to_string(),
            )));
        }
        let bytes = unsafe { std::slice::from_raw_parts(contents.cast::<u8>(), len) };
        Ok(bytes
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect())
    }

    fn new_command_buffer(&self) -> Result<objc::Id, RegistryError> {
        let cb = unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            )));
        }
        Ok(cb)
    }
}

/// Pipeline stage of a registered pipeline state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Vertex/fragment render pipeline.
    Graphics,
    /// Mesh/object/fragment render pipeline.
    Mesh,
    /// Compute kernel.
    Compute,
}

impl PipelineStage {
    /// Stable one-byte encoding used by [`PipelineKey`] and [`PipelineState`].
    pub const fn code(self) -> u8 {
        match self {
            Self::Graphics => 0,
            Self::Mesh => 1,
            Self::Compute => 2,
        }
    }

    /// Stable name used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Graphics => "graphics",
            Self::Mesh => "mesh",
            Self::Compute => "compute",
        }
    }

    /// Decodes a stable one-byte stage code.
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Graphics,
            1 => Self::Mesh,
            2 => Self::Compute,
            _ => return None,
        })
    }
}

/// The role one shader function plays in a registered pipeline.
///
/// A version 2 registration carries one library per stage, so the registry
/// resolves every function in its own library instead of assuming one metallib
/// holds both render stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FunctionRole {
    /// Vertex function of a graphics pipeline.
    Vertex,
    /// Fragment (pixel) function of a graphics or mesh pipeline.
    Fragment,
    /// Mesh function of a mesh pipeline.
    Mesh,
    /// Object function of a mesh pipeline that declares an object stage.
    Object,
    /// Kernel function of a compute pipeline.
    Kernel,
}

impl FunctionRole {
    /// Stable one-byte encoding used by [`PipelineKey::from_stages`].
    pub const fn code(self) -> u8 {
        match self {
            Self::Vertex => 0,
            Self::Fragment => 1,
            Self::Mesh => 2,
            Self::Object => 3,
            Self::Kernel => 4,
        }
    }

    /// Stage name used by entry-point and library diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "pixel",
            Self::Mesh => "mesh",
            Self::Object => "object",
            Self::Kernel => "compute",
        }
    }

    /// The `MTLFunctionType` this role must declare.
    const fn mtl_function_type(self) -> usize {
        match self {
            Self::Vertex => MTL_FUNCTION_TYPE_VERTEX,
            Self::Fragment => MTL_FUNCTION_TYPE_FRAGMENT,
            Self::Mesh => MTL_FUNCTION_TYPE_MESH,
            Self::Object => MTL_FUNCTION_TYPE_OBJECT,
            Self::Kernel => MTL_FUNCTION_TYPE_KERNEL,
        }
    }
}

/// One shader function and the `.metallib` that contains it.
///
/// The version 2 wire frame names one ticket (one library file) per stage, so a
/// pipeline may be built from distinct libraries. Nothing here concatenates
/// them or requires them to be equal.
#[derive(Debug, Clone, Copy)]
pub struct StageLibrary<'a> {
    /// Role this function plays in the pipeline.
    pub role: FunctionRole,
    /// The stage's own metallib bytes.
    pub metallib: &'a [u8],
    /// Entry point resolved inside `metallib`.
    pub entry: &'a str,
}

/// Content-addressed pipeline key.
///
/// The key retains the canonical byte encoding as well as the BLAKE3 digest, so
/// registry lookups compare exact bytes and never trust the digest alone.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    canonical: Vec<u8>,
    digest: [u8; 32],
}

impl PipelineKey {
    /// Hashes the length-prefixed canonical encoding of every pipeline input.
    pub fn from_parts(metallib: &[u8], entry: &str, stage: PipelineStage, state: &[u8]) -> Self {
        let mut canonical = Vec::with_capacity(64 + metallib.len() + entry.len() + state.len());
        canonical.extend_from_slice(b"VPSO");
        canonical.extend_from_slice(&1u16.to_le_bytes());
        canonical.push(stage.code());
        push_length_prefixed(&mut canonical, metallib);
        push_length_prefixed(&mut canonical, entry.as_bytes());
        push_length_prefixed(&mut canonical, state);
        let digest = *blake3::hash(&canonical).as_bytes();
        Self { canonical, digest }
    }

    /// Exact canonical bytes hashed by [`Self::digest`].
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }

    /// Hashes one library per stage.
    ///
    /// Unlike [`Self::from_parts`], every stage's role, metallib and entry
    /// point participate, so a changed pixel library is a different key even
    /// when the vertex library is unchanged. Stages are ordered by role before
    /// hashing, so the same stage set listed in another order maps to the same
    /// object.
    pub fn from_stages(stages: &[StageLibrary<'_>], stage: PipelineStage, state: &[u8]) -> Self {
        let mut ordered: Vec<&StageLibrary<'_>> = stages.iter().collect();
        ordered.sort_by_key(|entry| entry.role.code());
        let mut canonical = Vec::with_capacity(
            32 + state.len()
                + ordered
                    .iter()
                    .map(|entry| entry.metallib.len() + entry.entry.len() + 9)
                    .sum::<usize>(),
        );
        canonical.extend_from_slice(b"VPS2");
        canonical.extend_from_slice(&2u16.to_le_bytes());
        canonical.push(stage.code());
        canonical.push(u8::try_from(ordered.len()).expect("stage count exceeds u8"));
        for entry in ordered {
            canonical.push(entry.role.code());
            push_length_prefixed(&mut canonical, entry.metallib);
            push_length_prefixed(&mut canonical, entry.entry.as_bytes());
        }
        push_length_prefixed(&mut canonical, state);
        let digest = *blake3::hash(&canonical).as_bytes();
        Self { canonical, digest }
    }

    /// BLAKE3 digest of the canonical bytes.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Lowercase hexadecimal digest.
    pub fn digest_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.digest {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

fn push_length_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).expect("pipeline key part exceeds u32");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Encoded graphics/mesh pipeline state carried in the `state` blob.
///
/// Compute pipelines treat their `state` bytes as opaque; graphics and mesh
/// pipelines decode them with [`Self::decode`]. The encoding starts with
/// `VPS1` or `VPS2` and is length-checked, so a malformed blob is rejected
/// before Metal sees a descriptor.
///
/// A `VPS1` blob carries only the attachment/entry/mesh shape; it decodes with
/// every render-state field `None`, exactly the state a pipeline had before
/// `VPS2`. A `VPS2` blob appends the D3D12 render state in a fixed order:
/// per-attachment blend, depth/stencil and rasterizer. The render-state enums
/// are the D3D12-valued wire enums, so the guest and the host share one
/// mapping (`victory-gpu/src/native_metal/state.rs`).
///
/// D3D12 logic-op blending and depth bounds have no Metal equivalent and are
/// not encoded; the arithmetic blend path and the encoder-side depth state keep
/// their Cycle 06 behaviour. Depth/stencil and rasterizer values are encoded so
/// they change the content key; only blend is baked into the pipeline
/// descriptor here (see the module note for the pipeline/encoder split).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineState {
    /// Colour attachment formats in slot order (`0..=8`).
    pub color_formats: Vec<TextureFormat>,
    /// Optional depth/stencil attachment format.
    pub depth_format: Option<TextureFormat>,
    /// Fragment entry point (required for graphics and mesh).
    pub fragment_entry: Option<String>,
    /// Object entry point (mesh only).
    pub object_entry: Option<String>,
    /// `maxTotalThreadsPerMeshThreadgroup` source dimensions (mesh only).
    pub mesh_threadgroup: [u32; 3],
    /// Object threadgroup dimensions (mesh only).
    pub object_threadgroup: [u32; 3],
    /// `payloadMemoryLength` (mesh only).
    pub payload_memory_length: u32,
    /// `maxTotalThreadgroupsPerMeshGrid`, `0` for the descriptor default.
    pub max_total_threadgroups_per_mesh_grid: u32,
    /// Per-attachment blend state baked into the render pipeline descriptor.
    pub blend: Option<BlendState>,
    /// Depth/stencil comparison state carried in the content key.
    pub depth_stencil: Option<DepthStencilState>,
    /// Rasterizer state carried in the content key (cycle 06 encoder state).
    pub rasterizer: Option<RasterizerState>,
}

/// Render-state section flag: a [`BlendState`] follows.
const RENDER_SECTION_BLEND: u8 = 1;
/// Render-state section flag: a [`DepthStencilState`] follows.
const RENDER_SECTION_DEPTH_STENCIL: u8 = 2;
/// Render-state section flag: a [`RasterizerState`] follows.
const RENDER_SECTION_RASTERIZER: u8 = 4;
/// Every bit the render-state section defines.
const RENDER_SECTION_KNOWN: u8 = RENDER_SECTION_BLEND
    | RENDER_SECTION_DEPTH_STENCIL
    | RENDER_SECTION_RASTERIZER;

impl PipelineState {
    /// Serialises this state into the versioned `state` blob.
    ///
    /// A state with no blend, depth/stencil or rasterizer field is byte-for-byte
    /// the historical `VPS1` encoding. Any render state appends a `VPS2`
    /// section in one fixed order, so the bytes (and therefore the BLAKE3 key)
    /// change whenever any encoded state changes.
    pub fn encode(&self, stage: PipelineStage) -> Vec<u8> {
        let has_render_state = self.blend.is_some()
            || self.depth_stencil.is_some()
            || self.rasterizer.is_some();
        let mut out = Vec::with_capacity(if has_render_state { 192 } else { 64 });
        out.extend_from_slice(if has_render_state { b"VPS2" } else { b"VPS1" });
        out.push(stage.code());
        out.push(u8::try_from(self.color_formats.len()).expect("colour format count exceeds u8"));
        for format in &self.color_formats {
            out.push(format.code());
        }
        match self.depth_format {
            Some(format) => {
                out.push(1);
                out.push(format.code());
            }
            None => out.push(0),
        }
        let fragment = self.fragment_entry.as_deref().unwrap_or("");
        let object = self.object_entry.as_deref().unwrap_or("");
        out.push(u8::from(!fragment.is_empty()) | (u8::from(!object.is_empty()) << 1));
        push_u16_string(&mut out, fragment);
        push_u16_string(&mut out, object);
        for value in self
            .mesh_threadgroup
            .iter()
            .chain(self.object_threadgroup.iter())
        {
            out.extend_from_slice(
                &u16::try_from(*value)
                    .expect("threadgroup dimension exceeds u16")
                    .to_le_bytes(),
            );
        }
        out.extend_from_slice(
            &u16::try_from(self.payload_memory_length)
                .expect("payload memory length exceeds u16")
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.max_total_threadgroups_per_mesh_grid.to_le_bytes());
        if has_render_state {
            self.encode_render_state(&mut out);
        }
        out
    }

    /// Appends the `VPS2` render-state section.
    fn encode_render_state(&self, out: &mut Vec<u8>) {
        let mut flags = 0u8;
        if self.blend.is_some() {
            flags |= RENDER_SECTION_BLEND;
        }
        if self.depth_stencil.is_some() {
            flags |= RENDER_SECTION_DEPTH_STENCIL;
        }
        if self.rasterizer.is_some() {
            flags |= RENDER_SECTION_RASTERIZER;
        }
        out.push(flags);
        if let Some(blend) = self.blend.as_ref() {
            out.push(u8::from(blend.independent_blend));
            out.push(u8::from(blend.alpha_to_coverage));
            out.push(
                u8::try_from(blend.attachments.len())
                    .expect("blend attachment count exceeds u8"),
            );
            for attachment in &blend.attachments {
                push_blend_attachment(out, attachment);
            }
        }
        if let Some(depth) = self.depth_stencil.as_ref() {
            out.push(u8::from(depth.depth_enable));
            out.push(u8::from(depth.depth_write));
            push_u16(out, depth.depth_func.code());
            out.push(u8::from(depth.stencil_enable));
            push_stencil_face(out, &depth.front);
            push_stencil_face(out, &depth.back);
        }
        if let Some(raster) = self.rasterizer.as_ref() {
            push_u16(out, raster.fill.code());
            push_u16(out, raster.cull.code());
            push_u16(out, raster.front_face.code());
            out.extend_from_slice(&raster.depth_bias.to_le_bytes());
            out.extend_from_slice(&raster.slope_scaled_depth_bias.to_bits().to_le_bytes());
            out.extend_from_slice(&raster.depth_bias_clamp.to_bits().to_le_bytes());
            out.push(u8::from(raster.depth_clip));
        }
    }

    /// Parses a `VPS1` or `VPS2` state blob, requiring it to belong to `stage`.
    ///
    /// The stage byte is part of the encoding, so a mesh blob passed to the
    /// graphics registration path is rejected instead of silently accepted.
    pub fn decode_for(bytes: &[u8], stage: PipelineStage) -> Result<Self, RegistryError> {
        if bytes.get(4).copied() != Some(stage.code()) {
            return Err(RegistryError::InvalidState(
                "state stage does not match the pipeline stage",
            ));
        }
        Self::decode(bytes)
    }

    /// Parses a state blob and validates its bounds.
    ///
    /// `VPS1` is accepted byte-compatibly: every render-state field decodes to
    /// `None`. A truncated or malformed `VPS2` section is a typed
    /// [`RegistryError::InvalidState`], never a panic.
    pub fn decode(bytes: &[u8]) -> Result<Self, RegistryError> {
        let mut reader = BlobReader::new(bytes);
        let version = match reader.take(4)? {
            b"VPS1" => 1,
            b"VPS2" => 2,
            _ => return Err(RegistryError::InvalidState("bad magic")),
        };
        let stage = reader.u8()?;
        if PipelineStage::from_code(stage).is_none() {
            return Err(RegistryError::InvalidState("unknown stage"));
        }
        let color_count = reader.u8()? as usize;
        if color_count > 8 {
            return Err(RegistryError::InvalidState(
                "colour attachment count exceeds 8",
            ));
        }
        let mut color_formats = Vec::with_capacity(color_count);
        for _ in 0..color_count {
            color_formats.push(
                TextureFormat::from_code(reader.u8()?)
                    .ok_or(RegistryError::InvalidState("unknown colour format"))?,
            );
        }
        let depth_format = match reader.u8()? {
            0 => None,
            1 => Some(
                TextureFormat::from_code(reader.u8()?)
                    .ok_or(RegistryError::InvalidState("unknown depth format"))?,
            ),
            _ => return Err(RegistryError::InvalidState("bad depth format flag")),
        };
        let flags = reader.u8()?;
        if flags & !0b11 != 0 {
            return Err(RegistryError::InvalidState("unknown entry-point flags"));
        }
        let fragment_entry = reader.u16_string()?;
        let object_entry = reader.u16_string()?;
        let mut mesh_threadgroup = [0u32; 3];
        let mut object_threadgroup = [0u32; 3];
        for value in mesh_threadgroup.iter_mut() {
            *value = u32::from(reader.u16()?);
        }
        for value in object_threadgroup.iter_mut() {
            *value = u32::from(reader.u16()?);
        }
        let payload_memory_length = u32::from(reader.u16()?);
        let max_total_threadgroups_per_mesh_grid = reader.u32()?;
        let (blend, depth_stencil, rasterizer) = if version >= 2 {
            decode_render_state(&mut reader)?
        } else {
            (None, None, None)
        };
        if !reader.is_empty() {
            return Err(RegistryError::InvalidState("trailing bytes"));
        }
        Ok(Self {
            color_formats,
            depth_format,
            fragment_entry: (flags & 1 != 0).then_some(fragment_entry),
            object_entry: (flags & 2 != 0).then_some(object_entry),
            mesh_threadgroup,
            object_threadgroup,
            payload_memory_length,
            max_total_threadgroups_per_mesh_grid,
            blend,
            depth_stencil,
            rasterizer,
        })
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_blend_attachment(out: &mut Vec<u8>, attachment: &BlendAttachment) {
    out.push(u8::from(attachment.enable));
    push_u16(out, attachment.src_color.code());
    push_u16(out, attachment.dst_color.code());
    push_u16(out, attachment.color_op.code());
    push_u16(out, attachment.src_alpha.code());
    push_u16(out, attachment.dst_alpha.code());
    push_u16(out, attachment.alpha_op.code());
    out.push(attachment.write_mask);
}

fn push_stencil_face(out: &mut Vec<u8>, face: &StencilFaceState) {
    push_u16(out, face.fail.code());
    push_u16(out, face.depth_fail.code());
    push_u16(out, face.pass.code());
    push_u16(out, face.func.code());
    out.push(face.read_mask);
    out.push(face.write_mask);
}

fn decode_render_state(
    reader: &mut BlobReader<'_>,
) -> Result<
    (
        Option<BlendState>,
        Option<DepthStencilState>,
        Option<RasterizerState>,
    ),
    RegistryError,
> {
    let flags = reader.u8()?;
    if flags & !RENDER_SECTION_KNOWN != 0 {
        return Err(RegistryError::InvalidState(
            "unknown render-state section flag",
        ));
    }
    let blend = if flags & RENDER_SECTION_BLEND != 0 {
        Some(decode_blend_state(reader)?)
    } else {
        None
    };
    let depth_stencil = if flags & RENDER_SECTION_DEPTH_STENCIL != 0 {
        Some(decode_depth_stencil_state(reader)?)
    } else {
        None
    };
    let rasterizer = if flags & RENDER_SECTION_RASTERIZER != 0 {
        Some(decode_rasterizer_state(reader)?)
    } else {
        None
    };
    Ok((blend, depth_stencil, rasterizer))
}

fn decode_blend_state(reader: &mut BlobReader<'_>) -> Result<BlendState, RegistryError> {
    let independent_blend = decode_bool(reader, "blend independent flag")?;
    let alpha_to_coverage = decode_bool(reader, "alpha-to-coverage flag")?;
    let count = usize::from(reader.u8()?);
    if count == 0 || count > 8 {
        return Err(RegistryError::InvalidState(
            "blend attachment count is not 1..=8",
        ));
    }
    let mut attachments = Vec::with_capacity(count);
    for _ in 0..count {
        attachments.push(BlendAttachment {
            enable: decode_bool(reader, "blend enable flag")?,
            src_color: decode_blend_factor(reader)?,
            dst_color: decode_blend_factor(reader)?,
            color_op: decode_blend_operation(reader)?,
            src_alpha: decode_blend_factor(reader)?,
            dst_alpha: decode_blend_factor(reader)?,
            alpha_op: decode_blend_operation(reader)?,
            write_mask: reader.u8()?,
        });
    }
    Ok(BlendState {
        independent_blend,
        alpha_to_coverage,
        attachments,
    })
}

fn decode_depth_stencil_state(
    reader: &mut BlobReader<'_>,
) -> Result<DepthStencilState, RegistryError> {
    let depth_enable = decode_bool(reader, "depth enable flag")?;
    let depth_write = decode_bool(reader, "depth write flag")?;
    let depth_func = decode_compare_function(reader)?;
    let stencil_enable = decode_bool(reader, "stencil enable flag")?;
    let front = decode_stencil_face(reader)?;
    let back = decode_stencil_face(reader)?;
    Ok(DepthStencilState {
        depth_enable,
        depth_write,
        depth_func,
        stencil_enable,
        front,
        back,
    })
}

fn decode_stencil_face(reader: &mut BlobReader<'_>) -> Result<StencilFaceState, RegistryError> {
    Ok(StencilFaceState {
        fail: decode_stencil_operation(reader)?,
        depth_fail: decode_stencil_operation(reader)?,
        pass: decode_stencil_operation(reader)?,
        func: decode_compare_function(reader)?,
        read_mask: reader.u8()?,
        write_mask: reader.u8()?,
    })
}

fn decode_rasterizer_state(
    reader: &mut BlobReader<'_>,
) -> Result<RasterizerState, RegistryError> {
    Ok(RasterizerState {
        fill: decode_fill_mode(reader)?,
        cull: decode_cull_mode(reader)?,
        front_face: decode_winding(reader)?,
        depth_bias: reader.u32()? as i32,
        slope_scaled_depth_bias: f32::from_bits(reader.u32()?),
        depth_bias_clamp: f32::from_bits(reader.u32()?),
        depth_clip: decode_bool(reader, "depth-clip flag")?,
    })
}

fn decode_bool(reader: &mut BlobReader<'_>, field: &'static str) -> Result<bool, RegistryError> {
    match reader.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(RegistryError::InvalidState(field)),
    }
}

fn decode_blend_factor(reader: &mut BlobReader<'_>) -> Result<BlendFactor, RegistryError> {
    BlendFactor::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown blend factor code"))
}

fn decode_blend_operation(reader: &mut BlobReader<'_>) -> Result<BlendOperation, RegistryError> {
    BlendOperation::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown blend operation code"))
}

fn decode_compare_function(reader: &mut BlobReader<'_>) -> Result<CompareFunction, RegistryError> {
    CompareFunction::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown comparison function code"))
}

fn decode_stencil_operation(
    reader: &mut BlobReader<'_>,
) -> Result<StencilOperation, RegistryError> {
    StencilOperation::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown stencil operation code"))
}

fn decode_fill_mode(reader: &mut BlobReader<'_>) -> Result<FillMode, RegistryError> {
    FillMode::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown fill mode code"))
}

fn decode_cull_mode(reader: &mut BlobReader<'_>) -> Result<CullMode, RegistryError> {
    CullMode::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown cull mode code"))
}

fn decode_winding(reader: &mut BlobReader<'_>) -> Result<FrontFaceWinding, RegistryError> {
    FrontFaceWinding::from_code(reader.u16()?)
        .map_err(|_| RegistryError::InvalidState("unknown front-face winding code"))
}

fn push_u16_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(
        &u16::try_from(text.len())
            .expect("pipeline state string exceeds u16")
            .to_le_bytes(),
    );
    out.extend_from_slice(text.as_bytes());
}

struct BlobReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BlobReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], RegistryError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(RegistryError::InvalidState("offset overflows"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(RegistryError::InvalidState("truncated blob"))?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, RegistryError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, RegistryError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, RegistryError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u16_string(&mut self) -> Result<String, RegistryError> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| RegistryError::InvalidState("entry point is not UTF-8"))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Handle returned by [`PipelineRegistry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub u64);

/// A registered pipeline object and the metadata resolved from its key.
pub struct PipelineEntry {
    id: PipelineId,
    key: PipelineKey,
    stage: PipelineStage,
    object: ObjcObject,
    state: PipelineState,
    raw_state: Vec<u8>,
    reuse_count: u64,
}

impl PipelineEntry {
    /// Registry id.
    pub fn id(&self) -> PipelineId {
        self.id
    }

    /// Content key this object was created for.
    pub fn key(&self) -> &PipelineKey {
        &self.key
    }

    /// Pipeline stage.
    pub fn stage(&self) -> PipelineStage {
        self.stage
    }

    /// Parsed state (defaults for compute).
    pub fn state(&self) -> &PipelineState {
        &self.state
    }

    /// Exact caller state bytes hashed into the key.
    pub fn raw_state(&self) -> &[u8] {
        &self.raw_state
    }

    /// How many later registrations reused this object.
    pub fn reuse_count(&self) -> u64 {
        self.reuse_count
    }

    /// Raw `MTLComputePipelineState`/`MTLRenderPipelineState` handle.
    pub(crate) fn raw(&self) -> objc::Id {
        self.object.as_raw()
    }
}

/// Content-hashed pipeline registry.
pub struct PipelineRegistry {
    device: ObjcObject,
    queue: ObjcObject,
    pipelines: HashMap<PipelineKey, PipelineEntry>,
    by_id: HashMap<u64, PipelineKey>,
    next_id: u64,
    created: u64,
    reused: u64,
}

impl PipelineRegistry {
    /// Retains the transport's device and queue and starts an empty registry.
    pub fn new(transport: &DirectMetalTransport) -> Result<Self, RegistryError> {
        Ok(Self {
            device: retain(transport.device_raw())?,
            queue: retain(transport.queue_raw())?,
            pipelines: HashMap::new(),
            by_id: HashMap::new(),
            next_id: 1,
            created: 0,
            reused: 0,
        })
    }

    /// Registers (or reuses) a compute pipeline from metallib bytes.
    ///
    /// `state` is opaque for compute; it still participates in the key so two
    /// callers that disagree about compute state never share a pipeline.
    pub fn register_compute(
        &mut self,
        key: PipelineKey,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        self.register(
            key,
            metallib,
            entry,
            state,
            PipelineStage::Compute,
            PipelineState::default(),
        )
    }

    /// Registers (or reuses) a graphics pipeline from metallib bytes.
    ///
    /// `entry` is the vertex function; the fragment function, colour formats
    /// and optional depth format come from the [`PipelineState`] blob.
    pub fn register_graphics(
        &mut self,
        key: PipelineKey,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        let parsed = PipelineState::decode_for(state, PipelineStage::Graphics)?;
        require_entries(&parsed, PipelineStage::Graphics)?;
        self.register(key, metallib, entry, state, PipelineStage::Graphics, parsed)
    }

    /// Registers (or reuses) a mesh pipeline from metallib bytes.
    ///
    /// `entry` is the mesh function; the fragment/object entries and budgets
    /// come from the [`PipelineState`] blob. The library may contain both the
    /// mesh and fragment functions.
    pub fn register_mesh(
        &mut self,
        key: PipelineKey,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        let parsed = PipelineState::decode_for(state, PipelineStage::Mesh)?;
        require_entries(&parsed, PipelineStage::Mesh)?;
        self.register(key, metallib, entry, state, PipelineStage::Mesh, parsed)
    }

    /// Registers (or reuses) a graphics pipeline from one library per stage.
    ///
    /// `stages` must carry exactly the vertex and fragment functions, each with
    /// its own metallib and entry point; the fragment entry must be the one the
    /// [`PipelineState`] blob names. Every function is resolved in its own
    /// library, then one `MTLRenderPipelineState` is created from them.
    pub fn register_graphics_stages(
        &mut self,
        key: PipelineKey,
        stages: &[StageLibrary<'_>],
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        let parsed = PipelineState::decode_for(state, PipelineStage::Graphics)?;
        require_entries(&parsed, PipelineStage::Graphics)?;
        validate_stage_set(
            stages,
            &[FunctionRole::Vertex, FunctionRole::Fragment],
            PipelineStage::Graphics,
            &parsed,
        )?;
        self.register_stages(key, stages, state, PipelineStage::Graphics, parsed)
    }

    /// Registers (or reuses) a mesh pipeline from one library per stage.
    ///
    /// `stages` must carry the mesh and fragment functions and, when the
    /// [`PipelineState`] names an object entry, the object function too. Each
    /// function is resolved in its own library.
    pub fn register_mesh_stages(
        &mut self,
        key: PipelineKey,
        stages: &[StageLibrary<'_>],
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        let parsed = PipelineState::decode_for(state, PipelineStage::Mesh)?;
        require_entries(&parsed, PipelineStage::Mesh)?;
        let roles: &[FunctionRole] = if parsed.object_entry.is_some() {
            &[
                FunctionRole::Mesh,
                FunctionRole::Object,
                FunctionRole::Fragment,
            ]
        } else {
            &[FunctionRole::Mesh, FunctionRole::Fragment]
        };
        validate_stage_set(stages, roles, PipelineStage::Mesh, &parsed)?;
        self.register_stages(key, stages, state, PipelineStage::Mesh, parsed)
    }

    /// Registers (or reuses) a compute pipeline from one library per stage.
    ///
    /// A compute pipeline has one kernel function; the staged form exists so a
    /// frame built from several stage files never has to assume the kernel is
    /// in the first library. `state` stays opaque for compute and still
    /// participates in the key.
    pub fn register_compute_stages(
        &mut self,
        key: PipelineKey,
        stages: &[StageLibrary<'_>],
        state: &[u8],
    ) -> Result<PipelineId, RegistryError> {
        validate_stage_set(
            stages,
            &[FunctionRole::Kernel],
            PipelineStage::Compute,
            &PipelineState::default(),
        )?;
        self.register_stages(
            key,
            stages,
            state,
            PipelineStage::Compute,
            PipelineState::default(),
        )
    }

    /// Creates (or reuses) the object for one per-stage registration.
    fn register_stages(
        &mut self,
        key: PipelineKey,
        stages: &[StageLibrary<'_>],
        state: &[u8],
        stage: PipelineStage,
        parsed_state: PipelineState,
    ) -> Result<PipelineId, RegistryError> {
        if stages.iter().any(|entry| entry.entry.is_empty()) {
            return Err(RegistryError::EntryPointNotFound(String::new()));
        }
        let expected = PipelineKey::from_stages(stages, stage, state);
        if key != expected {
            return Err(RegistryError::KeyMismatch);
        }
        if let Some(existing) = self.pipelines.get_mut(&key) {
            existing.reuse_count += 1;
            self.reused += 1;
            return Ok(existing.id);
        }

        let _pool = AutoreleasePool::new();
        let object = match stage {
            PipelineStage::Graphics => {
                create_graphics_pso_stages(self.device.as_raw(), stages, &parsed_state)?
            }
            PipelineStage::Mesh => {
                create_mesh_pso_stages(self.device.as_raw(), stages, &parsed_state)?
            }
            PipelineStage::Compute => create_compute_pso_stages(self.device.as_raw(), stages)?,
        };
        Ok(self.insert_pipeline(key, stage, object, parsed_state, state))
    }

    fn register(
        &mut self,
        key: PipelineKey,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
        stage: PipelineStage,
        parsed_state: PipelineState,
    ) -> Result<PipelineId, RegistryError> {
        if entry.is_empty() {
            return Err(RegistryError::EntryPointNotFound(String::new()));
        }
        let expected = PipelineKey::from_parts(metallib, entry, stage, state);
        if key != expected {
            return Err(RegistryError::KeyMismatch);
        }
        if let Some(existing) = self.pipelines.get_mut(&key) {
            existing.reuse_count += 1;
            self.reused += 1;
            return Ok(existing.id);
        }

        let _pool = AutoreleasePool::new();
        let object = match stage {
            PipelineStage::Compute => create_compute_pso(self.device.as_raw(), metallib, entry)?,
            PipelineStage::Graphics => {
                create_graphics_pso(self.device.as_raw(), metallib, entry, &parsed_state)?
            }
            PipelineStage::Mesh => {
                create_mesh_pso(self.device.as_raw(), metallib, entry, &parsed_state)?
            }
        };

        Ok(self.insert_pipeline(key, stage, object, parsed_state, state))
    }

    /// Records one created object under its content key.
    fn insert_pipeline(
        &mut self,
        key: PipelineKey,
        stage: PipelineStage,
        object: ObjcObject,
        parsed_state: PipelineState,
        state: &[u8],
    ) -> PipelineId {
        let id = PipelineId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.created += 1;
        let entry = PipelineEntry {
            id,
            key: key.clone(),
            stage,
            object,
            state: parsed_state,
            raw_state: state.to_vec(),
            reuse_count: 0,
        };
        self.by_id.insert(id.0, key.clone());
        self.pipelines.insert(key, entry);
        id
    }

    /// Borrows the pipeline registered under `id`.
    pub fn get(&self, id: PipelineId) -> Option<&PipelineEntry> {
        self.by_id
            .get(&id.0)
            .and_then(|key| self.pipelines.get(key))
    }

    /// Borrows the pipeline registered under a content key.
    pub fn get_by_key(&self, key: &PipelineKey) -> Option<&PipelineEntry> {
        self.pipelines.get(key)
    }

    /// Releases the pipeline registered under `id`.
    pub fn destroy(&mut self, id: PipelineId) -> Result<(), RegistryError> {
        let key = self
            .by_id
            .remove(&id.0)
            .ok_or(RegistryError::UnknownPipeline(id.0))?;
        self.pipelines
            .remove(&key)
            .ok_or(RegistryError::UnknownPipeline(id.0))?;
        Ok(())
    }

    /// Number of live pipeline objects.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Whether the registry holds no pipelines.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Number of pipeline objects actually created.
    pub fn created_count(&self) -> u64 {
        self.created
    }

    /// Number of registrations that reused an existing object.
    pub fn reuse_count(&self) -> u64 {
        self.reused
    }

    /// Encodes one 1D compute dispatch with a pipeline owned by this registry.
    ///
    /// Every buffer must be registered on `transport`; an unknown handle or a
    /// threadgroup larger than the pipeline's maximum is rejected before a
    /// command buffer exists. The transport state lock is held through
    /// synchronous completion, exactly like the device's own dispatch path.
    pub fn dispatch_compute_1d(
        &self,
        transport: &DirectMetalTransport,
        id: PipelineId,
        buffers: &[VmBufferId],
        threads: u32,
        threads_per_group: u32,
    ) -> Result<(), RegistryError> {
        if threads == 0 || threads_per_group == 0 {
            return Err(RegistryError::InvalidDesc(
                "threads and threads_per_group must be nonzero",
            ));
        }
        let entry = self.get(id).ok_or(RegistryError::UnknownPipeline(id.0))?;
        if entry.stage != PipelineStage::Compute {
            return Err(RegistryError::Unsupported(
                "dispatch_compute_1d requires a compute pipeline",
            ));
        }

        let _pool = AutoreleasePool::new();
        let state = transport.lock_state()?;
        let mut bound = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let registered = state.buffers.get(&buffer.0).ok_or_else(|| {
                RegistryError::Metal(MetalError::NotFound(format!("buffer {}", buffer.0)))
            })?;
            bound.push(registered.buffer.as_raw());
        }

        let pso_max = unsafe {
            objc::msg_send_usize_0(entry.raw(), objc::sel(c"maxTotalThreadsPerThreadgroup"))
        }
        .max(1);
        if threads_per_group as usize > pso_max {
            return Err(RegistryError::InvalidDesc(
                "threads_per_group exceeds the pipeline maximum",
            ));
        }

        let cb = unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            )));
        }
        let encoder = unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
        if encoder.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "computeCommandEncoder returned null".to_string(),
            )));
        }
        unsafe {
            objc::msg_send_void_1(encoder, objc::sel(c"setComputePipelineState:"), entry.raw());
            for (slot, buffer) in bound.iter().enumerate() {
                objc::msg_send_set_buffer(encoder, *buffer, 0, slot);
            }
            objc::msg_send_dispatch_threads(
                encoder,
                MTLSize {
                    width: threads as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as usize,
                    height: 1,
                    depth: 1,
                },
            );
            objc::msg_send_void_0(encoder, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
        check_completion(cb, "registry compute dispatch")
    }
}

fn require_entries(state: &PipelineState, stage: PipelineStage) -> Result<(), RegistryError> {
    if state.color_formats.is_empty() || state.color_formats.len() > 8 {
        return Err(RegistryError::InvalidState(
            "graphics/mesh pipelines need 1..=8 colour formats",
        ));
    }
    if state
        .color_formats
        .iter()
        .any(|format| !format.is_color_render_target())
    {
        return Err(RegistryError::InvalidState(
            "colour attachments must be colour, non-block-compressed formats",
        ));
    }
    if let Some(depth) = state.depth_format
        && !depth.is_depth_stencil()
    {
        return Err(RegistryError::InvalidState(
            "depth attachment must be a depth format",
        ));
    }
    if let Some(blend) = state.blend.as_ref() {
        if blend.attachments.is_empty() || blend.attachments.len() > 8 {
            return Err(RegistryError::InvalidState(
                "blend state needs 1..=8 render-target entries",
            ));
        }
        // D3D12 always carries eight entries and uses the first
        // `NumRenderTargets`; a guest state may therefore carry more entries
        // than the pipeline has colour attachments. Fewer is a real error: a
        // target with no entry would silently lose its blend state.
        if blend.independent_blend && blend.attachments.len() < state.color_formats.len() {
            return Err(RegistryError::InvalidState(
                "independent blend needs an entry for every colour attachment",
            ));
        }
    }
    if state.fragment_entry.as_deref().is_none_or(str::is_empty) {
        return Err(RegistryError::InvalidState(
            "graphics/mesh pipelines need a fragment entry point",
        ));
    }
    if stage == PipelineStage::Graphics && state.object_entry.is_some() {
        return Err(RegistryError::InvalidState(
            "graphics pipelines do not have an object entry point",
        ));
    }
    Ok(())
}

fn create_compute_pso(
    device: objc::Id,
    metallib: &[u8],
    entry: &str,
) -> Result<ObjcObject, RegistryError> {
    let library = load_library(device, metallib, "compute")?;
    let function = new_function(&library, entry, MTL_FUNCTION_TYPE_KERNEL)?;
    build_compute_pso(device, function.as_raw(), entry)
}

/// Creates one `MTLComputePipelineState` from an already-resolved kernel.
fn build_compute_pso(
    device: objc::Id,
    function: objc::Id,
    entry: &str,
) -> Result<ObjcObject, RegistryError> {
    let mut err = objc::empty_id();
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newComputePipelineStateWithFunction:error:"),
            function,
            &mut err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        let message = unsafe { objc::nserror_message(err) };
        RegistryError::Metal(MetalError::CompileFailed(format!(
            "compute `{entry}` pipeline: {message}"
        )))
    })
}

/// Creates a compute pipeline from a per-stage library set.
fn create_compute_pso_stages(
    device: objc::Id,
    stages: &[StageLibrary<'_>],
) -> Result<ObjcObject, RegistryError> {
    let kernel_stage = stage_with_role(stages, FunctionRole::Kernel)?;
    let (_kernel_library, kernel) = stage_function(device, kernel_stage)?;
    build_compute_pso(device, kernel.as_raw(), kernel_stage.entry)
}

fn create_graphics_pso(
    device: objc::Id,
    metallib: &[u8],
    vertex_entry: &str,
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let library = load_library(device, metallib, "graphics")?;
    let vertex = new_function(&library, vertex_entry, MTL_FUNCTION_TYPE_VERTEX)?;
    let fragment_entry = state
        .fragment_entry
        .as_deref()
        .ok_or(RegistryError::InvalidState("missing fragment entry point"))?;
    let fragment = new_function(&library, fragment_entry, MTL_FUNCTION_TYPE_FRAGMENT)?;
    build_graphics_pso(
        device,
        vertex.as_raw(),
        fragment.as_raw(),
        vertex_entry,
        fragment_entry,
        state,
    )
}

/// Creates a graphics pipeline from one library per stage.
///
/// The vertex function comes from the vertex stage's library and the fragment
/// function from the pixel stage's library; the two libraries need not be the
/// same file and are never concatenated.
fn create_graphics_pso_stages(
    device: objc::Id,
    stages: &[StageLibrary<'_>],
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let vertex_stage = stage_with_role(stages, FunctionRole::Vertex)?;
    let fragment_stage = stage_with_role(stages, FunctionRole::Fragment)?;
    let (_vertex_library, vertex) = stage_function(device, vertex_stage)?;
    let (_fragment_library, fragment) = stage_function(device, fragment_stage)?;
    build_graphics_pso(
        device,
        vertex.as_raw(),
        fragment.as_raw(),
        vertex_stage.entry,
        fragment_stage.entry,
        state,
    )
}

/// Creates one `MTLRenderPipelineState` from already-resolved functions.
fn build_graphics_pso(
    device: objc::Id,
    vertex: objc::Id,
    fragment: objc::Id,
    vertex_entry: &str,
    fragment_entry: &str,
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let descriptor_class = objc::class(c"MTLRenderPipelineDescriptor");
    if descriptor_class.is_null() {
        return Err(RegistryError::Unsupported(
            "MTLRenderPipelineDescriptor is unavailable on this host",
        ));
    }
    let raw = unsafe { objc::msg_send_id_0(descriptor_class, objc::sel(c"new")) };
    let descriptor = ObjcObject::from_retained(raw).ok_or_else(|| {
        RegistryError::Metal(MetalError::InternalError(
            "MTLRenderPipelineDescriptor creation returned null".to_string(),
        ))
    })?;
    unsafe {
        objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setVertexFunction:"), vertex);
        objc::msg_send_void_1(
            descriptor.as_raw(),
            objc::sel(c"setFragmentFunction:"),
            fragment,
        );
    }
    set_attachment_formats(descriptor.as_raw(), state)?;

    let mut err = objc::empty_id();
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newRenderPipelineStateWithDescriptor:error:"),
            descriptor.as_raw(),
            &mut err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        let message = unsafe { objc::nserror_message(err) };
        RegistryError::Metal(MetalError::CompileFailed(format!(
            "graphics `{vertex_entry}` / `{fragment_entry}` pipeline: {message}"
        )))
    })
}

fn create_mesh_pso(
    device: objc::Id,
    metallib: &[u8],
    mesh_entry: &str,
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let library = load_library(device, metallib, "mesh")?;
    let mesh = new_function(&library, mesh_entry, MTL_FUNCTION_TYPE_MESH)?;
    let fragment_entry = state
        .fragment_entry
        .as_deref()
        .ok_or(RegistryError::InvalidState("missing fragment entry point"))?;
    let fragment = new_function(&library, fragment_entry, MTL_FUNCTION_TYPE_FRAGMENT)?;
    let object = match state.object_entry.as_deref() {
        Some(entry) => Some(new_function(&library, entry, MTL_FUNCTION_TYPE_OBJECT)?),
        None => None,
    };
    build_mesh_pso(
        device,
        mesh.as_raw(),
        fragment.as_raw(),
        object.as_ref().map(ObjcObject::as_raw),
        mesh_entry,
        fragment_entry,
        state,
    )
}

/// Creates a mesh pipeline from one library per stage.
///
/// The mesh, optional object and fragment functions each come from their own
/// stage's library; a frame that names an object entry must carry an object
/// stage, which the role validation has already enforced.
fn create_mesh_pso_stages(
    device: objc::Id,
    stages: &[StageLibrary<'_>],
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let mesh_stage = stage_with_role(stages, FunctionRole::Mesh)?;
    let fragment_stage = stage_with_role(stages, FunctionRole::Fragment)?;
    let (_mesh_library, mesh) = stage_function(device, mesh_stage)?;
    let (_fragment_library, fragment) = stage_function(device, fragment_stage)?;
    let object = match state.object_entry.as_deref() {
        Some(_) => {
            let object_stage = stage_with_role(stages, FunctionRole::Object)?;
            let (library, function) = stage_function(device, object_stage)?;
            Some((library, function))
        }
        None => None,
    };
    build_mesh_pso(
        device,
        mesh.as_raw(),
        fragment.as_raw(),
        object
            .as_ref()
            .map(|(_library, function)| function.as_raw()),
        mesh_stage.entry,
        fragment_stage.entry,
        state,
    )
}

/// Creates one `MTLRenderPipelineState` with a mesh descriptor.
fn build_mesh_pso(
    device: objc::Id,
    mesh: objc::Id,
    fragment: objc::Id,
    object: Option<objc::Id>,
    mesh_entry: &str,
    fragment_entry: &str,
    state: &PipelineState,
) -> Result<ObjcObject, RegistryError> {
    let descriptor_class = objc::class(c"MTLMeshRenderPipelineDescriptor");
    if descriptor_class.is_null() {
        return Err(RegistryError::Unsupported(
            "MTLMeshRenderPipelineDescriptor is unavailable on this host",
        ));
    }
    // SAFETY: `device` is a live MTLDevice; the selector check is a pure query.
    if !unsafe {
        objc::msg_send_responds_to(
            device,
            objc::sel(c"newRenderPipelineStateWithMeshDescriptor:error:"),
        )
    } {
        return Err(RegistryError::Unsupported(
            "MTLDevice does not implement newRenderPipelineStateWithMeshDescriptor:error:",
        ));
    }

    let mesh_threads = threadgroup_product(state.mesh_threadgroup).ok_or(
        RegistryError::InvalidState("mesh threadgroup dimensions overflow"),
    )?;
    if mesh_threads == 0 {
        return Err(RegistryError::InvalidState(
            "mesh threadgroup dimensions must be nonzero",
        ));
    }
    let object_threads = threadgroup_product(state.object_threadgroup).ok_or(
        RegistryError::InvalidState("object threadgroup dimensions overflow"),
    )?;
    if object_threads == 0 {
        return Err(RegistryError::InvalidState(
            "object threadgroup dimensions must be nonzero",
        ));
    }
    let max = unsafe { objc::msg_send_mtl_size_0(device, objc::sel(c"maxThreadsPerThreadgroup")) };
    let device_max = max
        .width
        .saturating_mul(max.height)
        .saturating_mul(max.depth)
        .max(1);
    if mesh_threads > device_max || object_threads > device_max {
        return Err(RegistryError::InvalidState(
            "threadgroup size exceeds the device maximum",
        ));
    }

    let raw = unsafe { objc::msg_send_id_0(descriptor_class, objc::sel(c"new")) };
    let descriptor = ObjcObject::from_retained(raw).ok_or_else(|| {
        RegistryError::Metal(MetalError::InternalError(
            "MTLMeshRenderPipelineDescriptor creation returned null".to_string(),
        ))
    })?;
    unsafe {
        objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setMeshFunction:"), mesh);
        objc::msg_send_void_1(
            descriptor.as_raw(),
            objc::sel(c"setFragmentFunction:"),
            fragment,
        );
        if let Some(object) = object {
            objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setObjectFunction:"), object);
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setMaxTotalThreadsPerObjectThreadgroup:"),
                object_threads,
            );
        }
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setMaxTotalThreadsPerMeshThreadgroup:"),
            mesh_threads,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setPayloadMemoryLength:"),
            state.payload_memory_length as usize,
        );
        if state.max_total_threadgroups_per_mesh_grid != 0 {
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setMaxTotalThreadgroupsPerMeshGrid:"),
                state.max_total_threadgroups_per_mesh_grid as usize,
            );
        }
    }
    set_attachment_formats(descriptor.as_raw(), state)?;

    let mut err = objc::empty_id();
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newRenderPipelineStateWithMeshDescriptor:error:"),
            descriptor.as_raw(),
            &mut err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        let message = unsafe { objc::nserror_message(err) };
        RegistryError::Metal(MetalError::CompileFailed(format!(
            "mesh `{mesh_entry}` / `{fragment_entry}` pipeline: {message}"
        )))
    })
}

/// Rejects a stage set that cannot describe one pipeline of `stage`'s family.
///
/// Every role must appear exactly once, no other role may appear, and the
/// fragment/object entries must be the ones the [`PipelineState`] blob names, so
/// the created functions and the content key cannot disagree.
fn validate_stage_set(
    stages: &[StageLibrary<'_>],
    expected: &[FunctionRole],
    stage: PipelineStage,
    parsed: &PipelineState,
) -> Result<(), RegistryError> {
    if stages.len() != expected.len() {
        return Err(RegistryError::InvalidState(match stage {
            PipelineStage::Graphics => {
                "graphics pipelines need exactly a vertex and a fragment stage"
            }
            PipelineStage::Mesh => {
                "mesh pipelines need a mesh and fragment stage, plus the object stage the state names"
            }
            PipelineStage::Compute => "compute pipelines need exactly one kernel stage",
        }));
    }
    for role in expected {
        if stages.iter().filter(|entry| entry.role == *role).count() != 1 {
            return Err(RegistryError::InvalidState(
                "a pipeline stage is missing or repeated",
            ));
        }
    }
    if stages
        .iter()
        .any(|entry| !expected.contains(&entry.role))
    {
        return Err(RegistryError::InvalidState(
            "a stage does not belong to this pipeline family",
        ));
    }
    for entry in stages {
        let declared = match entry.role {
            FunctionRole::Fragment => Some(parsed.fragment_entry.as_deref()),
            FunctionRole::Object => Some(parsed.object_entry.as_deref()),
            _ => None,
        };
        if let Some(declared) = declared
            && declared != Some(entry.entry)
        {
            return Err(RegistryError::InvalidState(match entry.role {
                FunctionRole::Fragment => {
                    "the fragment stage entry point does not match the pipeline state"
                }
                _ => "the object stage entry point does not match the pipeline state",
            }));
        }
    }
    Ok(())
}

/// Borrows the stage that plays `role`.
fn stage_with_role<'a, 'b: 'a>(
    stages: &'a [StageLibrary<'b>],
    role: FunctionRole,
) -> Result<&'a StageLibrary<'b>, RegistryError> {
    stages
        .iter()
        .find(|entry| entry.role == role)
        .ok_or(RegistryError::InvalidState("a pipeline stage is missing"))
}

/// Resolves one stage's function inside that stage's own library.
///
/// The library is returned alongside the function so the caller can keep it
/// alive until the pipeline descriptor has consumed the function.
fn stage_function(
    device: objc::Id,
    stage: &StageLibrary<'_>,
) -> Result<(ObjcObject, ObjcObject), RegistryError> {
    let library = load_library(device, stage.metallib, stage.role.label())?;
    let function = new_function(&library, stage.entry, stage.role.mtl_function_type()).map_err(
        |error| match error {
            RegistryError::EntryPointNotFound(_) => RegistryError::MissingStageFunction {
                stage: stage.role.label(),
                entry: stage.entry.to_string(),
            },
            other => other,
        },
    )?;
    Ok((library, function))
}

fn set_attachment_formats(
    descriptor: objc::Id,
    state: &PipelineState,
) -> Result<(), RegistryError> {
    let attachments = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    if attachments.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "pipeline descriptor colorAttachments is nil".to_string(),
        )));
    }
    for (index, format) in state.color_formats.iter().enumerate() {
        let (pixel_format, _) = attachment_pixel_format(*format)?;
        let attachment = unsafe {
            objc::msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), index)
        };
        if attachment.is_null() {
            return Err(RegistryError::Metal(MetalError::InternalError(
                "colour attachment descriptor is nil".to_string(),
            )));
        }
        unsafe {
            objc::msg_send_void_usize(
                attachment,
                objc::sel(c"setPixelFormat:"),
                pixel_format,
            );
            // Blending is pipeline state in Metal. The shared
            // (`IndependentBlendEnable = FALSE`) form applies entry 0 to every
            // target, exactly like D3D12. `apply_blend_attachment` is the one
            // D3D12 -> MTLBlendFactor/Operation/mask mapping (macos::state).
            if let Some(blend) = state.blend.as_ref() {
                let entry = if blend.independent_blend {
                    blend.attachments.get(index)
                } else {
                    blend.attachments.first()
                };
                if let Some(entry) = entry {
                    crate::macos::state::apply_blend_attachment(attachment, entry)?;
                }
            }
        }
    }
    if state
        .blend
        .as_ref()
        .is_some_and(|blend| blend.alpha_to_coverage)
    {
        unsafe {
            objc::msg_send_void_u8(
                descriptor,
                objc::sel(c"setAlphaToCoverageEnabled:"),
                1,
            );
        }
    }
    if let Some(format) = state.depth_format {
        let (pixel_format, has_stencil) = attachment_pixel_format(format)?;
        unsafe {
            objc::msg_send_void_usize(
                descriptor,
                objc::sel(c"setDepthAttachmentPixelFormat:"),
                pixel_format,
            );
            if has_stencil {
                objc::msg_send_void_usize(
                    descriptor,
                    objc::sel(c"setStencilAttachmentPixelFormat:"),
                    pixel_format,
                );
            }
        }
    }
    Ok(())
}

/// The `MTLPixelFormat` of one pipeline attachment and whether it carries a
/// stencil plane.
///
/// Only formats that can legitimately back a colour or depth attachment are
/// accepted. RGBA16Float is a colour format (`MTLPixelFormatRGBA16Float`,
/// 115), never depth; block-compressed and depth formats stay out of the
/// colour path exactly as before.
fn attachment_pixel_format(format: TextureFormat) -> Result<(usize, bool), RegistryError> {
    match format {
        TextureFormat::Rgba8Unorm
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Rgba16Float
        | TextureFormat::Depth32Float => Ok((format.mtl_pixel_format(), false)),
        TextureFormat::Depth32FloatStencil8 => Ok((format.mtl_pixel_format(), true)),
        _ => Err(RegistryError::Unsupported(
            "attachment format is not supported by the render pipeline path",
        )),
    }
}

fn threadgroup_product(dims: [u32; 3]) -> Option<usize> {
    (dims[0] as usize)
        .checked_mul(dims[1] as usize)?
        .checked_mul(dims[2] as usize)
}

fn load_library(
    device: objc::Id,
    metallib: &[u8],
    kind: &'static str,
) -> Result<ObjcObject, RegistryError> {
    if metallib.is_empty() {
        return Err(RegistryError::InvalidMetallib(format!(
            "{kind} metallib bytes are empty"
        )));
    }
    let data = objc::dispatch_data_copy(metallib).ok_or_else(|| {
        RegistryError::InvalidMetallib(format!("{kind} metallib copy allocation failed"))
    })?;
    let mut err = objc::empty_id();
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newLibraryWithData:error:"),
            data.as_raw(),
            &mut err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        let message = unsafe { objc::nserror_message(err) };
        RegistryError::InvalidMetallib(format!("{kind} metallib: {message}"))
    })
}

fn new_function(
    library: &ObjcObject,
    entry: &str,
    expected_type: usize,
) -> Result<ObjcObject, RegistryError> {
    if entry.is_empty() {
        return Err(RegistryError::EntryPointNotFound(String::new()));
    }
    let name = objc::nsstring(entry).ok_or_else(|| {
        RegistryError::Metal(MetalError::InvalidArg("entry point name allocation failed"))
    })?;
    let raw = unsafe {
        objc::msg_send_id_1(
            library.as_raw(),
            objc::sel(c"newFunctionWithName:"),
            name.as_raw(),
        )
    };
    let function = ObjcObject::from_retained(raw)
        .ok_or_else(|| RegistryError::EntryPointNotFound(entry.to_string()))?;
    let actual = unsafe { objc::msg_send_usize_0(function.as_raw(), objc::sel(c"functionType")) };
    if actual != expected_type {
        return Err(RegistryError::EntryPointNotFound(entry.to_string()));
    }
    Ok(function)
}

fn retain(raw: objc::Id) -> Result<ObjcObject, RegistryError> {
    if raw.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "cannot retain a null Metal object".to_string(),
        )));
    }
    let retained = unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) };
    ObjcObject::from_retained(retained).ok_or_else(|| {
        RegistryError::Metal(MetalError::InternalError(
            "Metal retain returned null".to_string(),
        ))
    })
}

fn new_texture_descriptor(desc: &TextureDesc) -> Result<ObjcObject, RegistryError> {
    let class = objc::class(c"MTLTextureDescriptor");
    if class.is_null() {
        return Err(RegistryError::Unsupported(
            "MTLTextureDescriptor is unavailable on this host",
        ));
    }
    let raw = unsafe { objc::msg_send_id_0(class, objc::sel(c"new")) };
    let descriptor = ObjcObject::from_retained(raw).ok_or_else(|| {
        RegistryError::Metal(MetalError::InternalError(
            "MTLTextureDescriptor creation returned null".to_string(),
        ))
    })?;
    unsafe {
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setTextureType:"),
            desc.mtl_texture_type(),
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setPixelFormat:"),
            desc.format.mtl_pixel_format(),
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setWidth:"),
            desc.width as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setHeight:"),
            desc.height as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setDepth:"),
            desc.depth as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setMipmapLevelCount:"),
            desc.mip_levels as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setSampleCount:"),
            desc.sample_count as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setArrayLength:"),
            desc.array_layers as usize,
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setUsage:"),
            desc.usage.mtl_bits(),
        );
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setStorageMode:"),
            MTL_STORAGE_MODE_PRIVATE,
        );
        if let Some(label) = desc.label.as_deref()
            && let Some(text) = objc::nsstring(label)
            && objc::msg_send_responds_to(descriptor.as_raw(), objc::sel(c"setLabel:"))
        {
            objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setLabel:"), text.as_raw());
        }
    }
    Ok(descriptor)
}

fn verify_texture_attributes(
    texture: &ObjcObject,
    desc: &TextureDesc,
) -> Result<(), RegistryError> {
    let raw = texture.as_raw();
    let check = |selector, expected: usize| -> Result<(), RegistryError> {
        let actual = unsafe { objc::msg_send_usize_0(raw, objc::sel(selector)) };
        if actual != expected {
            return Err(RegistryError::Metal(MetalError::InternalError(format!(
                "Metal texture attribute `{}` is {actual}, expected {expected}",
                selector.to_string_lossy()
            ))));
        }
        Ok(())
    };
    check(c"textureType", desc.mtl_texture_type())?;
    check(c"pixelFormat", desc.format.mtl_pixel_format())?;
    check(c"width", desc.width as usize)?;
    check(c"height", desc.height as usize)?;
    check(c"depth", desc.depth as usize)?;
    check(c"mipmapLevelCount", desc.mip_levels as usize)?;
    check(c"sampleCount", desc.sample_count as usize)?;
    check(c"arrayLength", desc.array_layers as usize)
}

unsafe fn new_texture_view(texture: objc::Id, plan: ViewPlan) -> objc::Id {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, NSRange, NSRange) -> objc::Id;
    // SAFETY: Caller supplies a live MTLTexture and a plan derived from its
    // descriptor. `NSRange` is two `NSUInteger`s, passed per the platform C ABI.
    let f: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            texture,
            objc::sel(c"newTextureViewWithPixelFormat:textureType:levels:slices:"),
            plan.pixel_format,
            plan.texture_type,
            plan.levels,
            plan.slices,
        )
    }
}

fn new_render_pass_descriptor(
    view: &TextureView,
    color: [f64; 4],
) -> Result<objc::Id, RegistryError> {
    let class = objc::class(c"MTLRenderPassDescriptor");
    if class.is_null() {
        return Err(RegistryError::Unsupported(
            "MTLRenderPassDescriptor is unavailable on this host",
        ));
    }
    // `renderPassDescriptor` returns an autoreleased object owned by the pool.
    let descriptor = unsafe { objc::msg_send_id_0(class, objc::sel(c"renderPassDescriptor")) };
    if descriptor.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "MTLRenderPassDescriptor creation returned null".to_string(),
        )));
    }
    let attachments = unsafe { objc::msg_send_id_0(descriptor, objc::sel(c"colorAttachments")) };
    if attachments.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "MTLRenderPassDescriptor.colorAttachments is nil".to_string(),
        )));
    }
    let attachment =
        unsafe { objc::msg_send_id_usize(attachments, objc::sel(c"objectAtIndexedSubscript:"), 0) };
    if attachment.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "colour attachment descriptor is nil".to_string(),
        )));
    }
    let clear = MTLClearColor {
        red: color[0],
        green: color[1],
        blue: color[2],
        alpha: color[3],
    };
    unsafe {
        objc::msg_send_void_1(attachment, objc::sel(c"setTexture:"), view.raw());
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setLoadAction:"),
            MTL_LOAD_ACTION_CLEAR,
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setStoreAction:"),
            MTL_STORE_ACTION_STORE,
        );
        set_clear_color(attachment, clear);
    }
    Ok(descriptor)
}

unsafe fn set_clear_color(attachment: objc::Id, clear: MTLClearColor) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLClearColor);
    // SAFETY: Caller supplies a live MTLRenderPassColorAttachmentDescriptor and
    // a four-double clear colour, passed per the platform C ABI.
    let f: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `f` matches `setClearColor:`.
    unsafe { f(attachment, objc::sel(c"setClearColor:"), clear) };
}

unsafe fn copy_texture_to_buffer(
    encoder: objc::Id,
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
    // SAFETY: Caller supplies a live blit encoder and live Metal objects; the
    // per-image stride is `0` for a single level-0 image.
    let f: Fn = unsafe { mem::transmute(objc::msg_send_entry()) };
    // SAFETY: Argument layout matches the selector declaration.
    unsafe {
        f(
            encoder,
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

fn check_completion(cb: objc::Id, what: &'static str) -> Result<(), RegistryError> {
    let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
    if status == MTL_COMMAND_BUFFER_STATUS_COMPLETED {
        return Ok(());
    }
    let err = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
    let message = unsafe { objc::nserror_message(err) };
    Err(RegistryError::Metal(MetalError::InternalError(format!(
        "{what} failed (status={status}): {message}"
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> TextureDesc {
        TextureDesc {
            width: 64,
            height: 64,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            array_layers: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::sampled(),
            label: None,
        }
    }

    #[test]
    fn every_declared_format_maps_to_a_distinct_metal_pixel_format() {
        let formats = [
            (TextureFormat::Bc1Rgba, 130),
            (TextureFormat::Bc2Rgba, 132),
            (TextureFormat::Bc3Rgba, 134),
            (TextureFormat::Bc4Rgba, 140),
            (TextureFormat::Bc5Rgba, 142),
            (TextureFormat::Bc6hRgbUf16, 151),
            (TextureFormat::Bc7Rgba, 152),
            (TextureFormat::Rgba8Unorm, 70),
            (TextureFormat::Rgba8Srgb, 71),
            (TextureFormat::Bgra8Unorm, 80),
            (TextureFormat::Rgba16Float, 115),
            (TextureFormat::Rg11g11b10Float, 92),
            (TextureFormat::Rgb10a2Unorm, 90),
            (TextureFormat::R32Float, 55),
            (TextureFormat::R16Float, 25),
            (TextureFormat::Depth32Float, 252),
            (TextureFormat::Depth24Stencil8, 255),
            (TextureFormat::Depth32FloatStencil8, 260),
        ];
        let mut seen = std::collections::HashSet::new();
        for (format, mtl) in formats {
            assert_eq!(format.mtl_pixel_format(), mtl, "{format:?}");
            assert!(seen.insert(mtl), "duplicate MTLPixelFormat {mtl}");
            assert_eq!(TextureFormat::from_code(format.code()), Some(format));
        }
        assert_eq!(TextureFormat::from_code(200), None);
    }

    /// A 2560x1440 RGBA16Float colour target (the title's format-10
    /// back-buffer) is a valid registry texture and a valid 115 attachment;
    /// no GPU is touched.
    #[test]
    fn a_2560x1440_rgba16_float_colour_target_is_accepted() {
        let mut target = desc();
        target.width = 2560;
        target.height = 1440;
        target.format = TextureFormat::Rgba16Float;
        target.usage = TextureUsage::color_attachment();
        assert!(target.validate().is_ok());
        assert!(matches!(
            attachment_pixel_format(target.format),
            Ok((115, false))
        ));
    }

    #[test]
    fn texture_descriptor_rejects_inconsistent_geometry() {
        assert!(desc().validate().is_ok());

        let mut zero = desc();
        zero.width = 0;
        assert!(matches!(
            zero.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut huge = desc();
        huge.width = crate::macos::texture::MAX_TEXTURE_DIMENSION_2D + 1;
        assert!(matches!(
            huge.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut one_d = desc();
        one_d.dimension = TextureDimension::D1;
        one_d.height = 2;
        assert!(matches!(
            one_d.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut cube = desc();
        cube.dimension = TextureDimension::Cube;
        cube.array_layers = 6;
        assert!(cube.validate().is_ok());
        cube.array_layers = 5;
        assert!(matches!(
            cube.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut cube_array = desc();
        cube_array.dimension = TextureDimension::CubeArray;
        cube_array.array_layers = 12;
        assert!(cube_array.validate().is_ok());
        cube_array.array_layers = 7;
        assert!(matches!(
            cube_array.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut volume = desc();
        volume.dimension = TextureDimension::D3;
        volume.depth = 16;
        assert!(volume.validate().is_ok());
        volume.usage = TextureUsage::color_attachment();
        assert!(matches!(
            volume.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut msaa = desc();
        msaa.sample_count = 3;
        assert!(matches!(
            msaa.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));
        msaa.sample_count = 4;
        msaa.mip_levels = 2;
        assert!(matches!(
            msaa.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut no_usage = desc();
        no_usage.usage = TextureUsage::default();
        assert!(matches!(
            no_usage.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut mips = desc();
        mips.mip_levels = 7;
        assert!(mips.validate().is_ok());
        mips.mip_levels = 8;
        assert!(matches!(
            mips.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));
    }

    #[test]
    fn texture_descriptor_rejects_mismatched_format_usage() {
        let mut depth_as_color = desc();
        depth_as_color.format = TextureFormat::Depth32Float;
        depth_as_color.usage = TextureUsage::color_attachment();
        assert!(matches!(
            depth_as_color.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut color_as_depth = desc();
        color_as_depth.usage = TextureUsage::depth_attachment();
        assert!(matches!(
            color_as_depth.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));

        let mut depth = desc();
        depth.format = TextureFormat::Depth32Float;
        depth.usage = TextureUsage::depth_attachment();
        assert!(depth.validate().is_ok());

        let mut bc_small = desc();
        bc_small.format = TextureFormat::Bc1Rgba;
        bc_small.width = 6;
        bc_small.height = 8;
        assert!(matches!(
            bc_small.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));
        bc_small.width = 8;
        assert!(bc_small.validate().is_ok());

        let mut bc_write = desc();
        bc_write.format = TextureFormat::Bc7Rgba;
        bc_write.usage = TextureUsage {
            shader_read: true,
            shader_write: true,
            render_target: false,
            depth_stencil: false,
        };
        assert!(matches!(
            bc_write.validate(),
            Err(RegistryError::InvalidDesc(_))
        ));
    }

    #[test]
    fn view_resolution_rejects_out_of_range_and_wrong_usage() {
        let texture = desc();
        assert!(
            ViewPlan::resolve(
                &texture,
                TextureViewKind::Rtv,
                TextureSubresource::single(0, 0)
            )
            .is_err()
        );
        assert!(
            ViewPlan::resolve(
                &texture,
                TextureViewKind::Srv,
                TextureSubresource::single(0, 0)
            )
            .is_ok()
        );
        assert!(matches!(
            ViewPlan::resolve(
                &texture,
                TextureViewKind::Srv,
                TextureSubresource::single(1, 0)
            ),
            Err(RegistryError::InvalidSubresource(_))
        ));
        assert!(matches!(
            ViewPlan::resolve(
                &texture,
                TextureViewKind::Srv,
                TextureSubresource {
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 1,
                    array_layer_count: 1,
                }
            ),
            Err(RegistryError::InvalidSubresource(_))
        ));

        let mut color = desc();
        color.usage = TextureUsage::color_attachment();
        let rtv = ViewPlan::resolve(
            &color,
            TextureViewKind::Rtv,
            TextureSubresource::single(0, 0),
        )
        .expect("an RTV over a colour target is valid");
        assert_eq!(rtv.texture_type, MTL_TEXTURE_TYPE_2D);
        assert!(matches!(
            ViewPlan::resolve(
                &color,
                TextureViewKind::Dsv,
                TextureSubresource::single(0, 0)
            ),
            Err(RegistryError::Unsupported(_))
        ));

        let mut depth = desc();
        depth.format = TextureFormat::Depth32Float;
        depth.usage = TextureUsage::depth_attachment();
        assert!(
            ViewPlan::resolve(
                &depth,
                TextureViewKind::Dsv,
                TextureSubresource::single(0, 0)
            )
            .is_ok()
        );
        assert!(matches!(
            ViewPlan::resolve(
                &depth,
                TextureViewKind::Rtv,
                TextureSubresource::single(0, 0)
            ),
            Err(RegistryError::Unsupported(_))
        ));

        let mut storage = desc();
        storage.usage = TextureUsage {
            shader_read: true,
            shader_write: true,
            render_target: false,
            depth_stencil: false,
        };
        let uav = ViewPlan::resolve(
            &storage,
            TextureViewKind::Uav,
            TextureSubresource::single(0, 0),
        )
        .expect("a UAV over a shader-written 2D texture is valid");
        assert_eq!(uav.texture_type, MTL_TEXTURE_TYPE_2D);
        assert!(matches!(
            ViewPlan::resolve(
                &storage,
                TextureViewKind::Uav,
                TextureSubresource {
                    base_mip_level: 0,
                    mip_level_count: 2,
                    base_array_layer: 0,
                    array_layer_count: 1,
                }
            ),
            Err(RegistryError::InvalidSubresource(_))
        ));
    }

    #[test]
    fn pipeline_key_is_length_prefixed_and_changes_on_one_byte() {
        let metallib = b"METALLIB".to_vec();
        let state = b"STATE".to_vec();
        let base = PipelineKey::from_parts(&metallib, "main", PipelineStage::Compute, &state);
        assert_eq!(
            base,
            PipelineKey::from_parts(&metallib, "main", PipelineStage::Compute, &state)
        );
        assert_eq!(base.digest_hex().len(), 64);

        let mut byte = metallib.clone();
        byte[0] ^= 1;
        assert_ne!(
            base,
            PipelineKey::from_parts(&byte, "main", PipelineStage::Compute, &state)
        );

        let mut one_state = state.clone();
        one_state[0] ^= 1;
        assert_ne!(
            base,
            PipelineKey::from_parts(&metallib, "main", PipelineStage::Compute, &one_state)
        );
        assert_ne!(
            base,
            PipelineKey::from_parts(&metallib, "maiN", PipelineStage::Compute, &state)
        );
        assert_ne!(
            base,
            PipelineKey::from_parts(&metallib, "main", PipelineStage::Graphics, &state)
        );

        // Length prefixing keeps part boundaries unambiguous.
        let split_a = PipelineKey::from_parts(b"ab", "c", PipelineStage::Compute, b"");
        let split_b = PipelineKey::from_parts(b"a", "bc", PipelineStage::Compute, b"");
        assert_ne!(split_a, split_b);
    }

    fn test_blend(enable: bool, src: BlendFactor, dst: BlendFactor) -> BlendState {
        BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: vec![BlendAttachment {
                enable,
                src_color: src,
                dst_color: dst,
                color_op: BlendOperation::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::Zero,
                alpha_op: BlendOperation::Add,
                write_mask: 0x0f,
            }],
        }
    }

    fn test_depth_stencil() -> DepthStencilState {
        let face = StencilFaceState {
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::IncrSat,
            pass: StencilOperation::Replace,
            func: CompareFunction::LessEqual,
            read_mask: 0x7f,
            write_mask: 0x0f,
        };
        DepthStencilState {
            depth_enable: true,
            depth_write: true,
            depth_func: CompareFunction::Less,
            stencil_enable: true,
            front: face,
            back: face,
        }
    }

    fn test_rasterizer() -> RasterizerState {
        RasterizerState {
            fill: FillMode::Wireframe,
            cull: CullMode::Front,
            front_face: FrontFaceWinding::CounterClockwise,
            depth_bias: -8,
            slope_scaled_depth_bias: 2.5,
            depth_bias_clamp: 0.25,
            depth_clip: false,
        }
    }

    #[test]
    fn pipeline_state_round_trips_and_rejects_malformed_blobs() {
        let state = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm],
            depth_format: Some(TextureFormat::Depth32Float),
            fragment_entry: Some("fs_main".to_string()),
            object_entry: Some("obj_main".to_string()),
            mesh_threadgroup: [32, 1, 1],
            object_threadgroup: [8, 1, 1],
            payload_memory_length: 128,
            max_total_threadgroups_per_mesh_grid: 4,
            blend: Some(test_blend(true, BlendFactor::SrcAlpha, BlendFactor::InvSrcAlpha)),
            depth_stencil: Some(test_depth_stencil()),
            rasterizer: Some(test_rasterizer()),
        };
        let encoded = state.encode(PipelineStage::Mesh);
        assert_eq!(PipelineState::decode(&encoded).unwrap(), state);
        assert_eq!(
            PipelineState::decode_for(&encoded, PipelineStage::Mesh).unwrap(),
            state
        );
        assert!(matches!(
            PipelineState::decode_for(&encoded, PipelineStage::Graphics),
            Err(RegistryError::InvalidState(_))
        ));
        assert_eq!(encoded[..4], *b"VPS2");

        let mut truncated = encoded.clone();
        truncated.pop();
        assert!(matches!(
            PipelineState::decode(&truncated),
            Err(RegistryError::InvalidState(_))
        ));
        // Every prefix of a VPS2 blob is a typed error, never a panic.
        for cut in 0..encoded.len() {
            assert!(
                matches!(
                    PipelineState::decode(&encoded[..cut]),
                    Err(RegistryError::InvalidState(_))
                ),
                "a {cut}-byte VPS2 prefix must be rejected"
            );
        }

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            PipelineState::decode(&trailing),
            Err(RegistryError::InvalidState(_))
        ));

        let mut bad_magic = encoded.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            PipelineState::decode(&bad_magic),
            Err(RegistryError::InvalidState(_))
        ));

        let mut bad_stage = encoded.clone();
        bad_stage[4] = 9;
        assert!(matches!(
            PipelineState::decode(&bad_stage),
            Err(RegistryError::InvalidState(_))
        ));

        // A VPS1 blob is still accepted and means the scaffold's default state.
        let v1 = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            depth_format: Some(TextureFormat::Depth32Float),
            fragment_entry: Some("fs_main".to_string()),
            ..PipelineState::default()
        }
        .encode(PipelineStage::Graphics);
        assert_eq!(v1[..4], *b"VPS1");
        let decoded_v1 = PipelineState::decode(&v1).expect("VPS1 stays decodable");
        assert!(decoded_v1.blend.is_none());
        assert!(decoded_v1.depth_stencil.is_none());
        assert!(decoded_v1.rasterizer.is_none());

        let empty = PipelineState::default();
        let empty_bytes = empty.encode(PipelineStage::Compute);
        assert_eq!(empty_bytes[..4], *b"VPS1");
        assert_eq!(PipelineState::decode(&empty_bytes).unwrap(), empty);
    }

    #[test]
    fn staged_pipeline_key_covers_every_stage_library() {
        let vertex = StageLibrary {
            role: FunctionRole::Vertex,
            metallib: b"vertex metallib",
            entry: "vs_main",
        };
        let pixel_a = StageLibrary {
            role: FunctionRole::Fragment,
            metallib: b"pixel metallib",
            entry: "ps_main",
        };
        let pixel_b = StageLibrary {
            role: FunctionRole::Fragment,
            metallib: b"other pixel metallib",
            entry: "ps_main",
        };
        let state = b"graphics state";
        let with_a =
            PipelineKey::from_stages(&[vertex, pixel_a], PipelineStage::Graphics, state);
        assert_eq!(
            with_a,
            PipelineKey::from_stages(&[pixel_a, vertex], PipelineStage::Graphics, state),
            "stage order does not change the key"
        );
        let with_b =
            PipelineKey::from_stages(&[vertex, pixel_b], PipelineStage::Graphics, state);
        assert_ne!(
            with_a, with_b,
            "a changed pixel library is a different pipeline"
        );
        assert_ne!(
            with_a,
            PipelineKey::from_parts(b"vertex metallib", "vs_main", PipelineStage::Graphics, state),
            "the single-library key covers neither stage's separate library"
        );
        let single = StageLibrary {
            role: FunctionRole::Kernel,
            metallib: b"compute metallib",
            entry: "cs_main",
        };
        assert_ne!(
            PipelineKey::from_stages(&[single], PipelineStage::Compute, state),
            PipelineKey::from_stages(&[single], PipelineStage::Graphics, state),
            "the pipeline family participates in the key"
        );
    }

    #[test]
    fn staged_role_validation_rejects_wrong_sets_and_entries() {
        let state = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            fragment_entry: Some("ps_main".to_string()),
            ..PipelineState::default()
        }
        .encode(PipelineStage::Graphics);
        let parsed = PipelineState::decode_for(&state, PipelineStage::Graphics).unwrap();
        let vertex = StageLibrary {
            role: FunctionRole::Vertex,
            metallib: b"vs",
            entry: "vs_main",
        };
        let pixel = StageLibrary {
            role: FunctionRole::Fragment,
            metallib: b"ps",
            entry: "ps_main",
        };
        let expected = [FunctionRole::Vertex, FunctionRole::Fragment];
        assert!(validate_stage_set(&[vertex, pixel], &expected, PipelineStage::Graphics, &parsed).is_ok());

        let wrong_role = StageLibrary {
            role: FunctionRole::Mesh,
            metallib: b"ms",
            entry: "ms_main",
        };
        assert!(matches!(
            validate_stage_set(&[vertex, wrong_role], &expected, PipelineStage::Graphics, &parsed),
            Err(RegistryError::InvalidState(_))
        ));
        let missing = StageLibrary {
            role: FunctionRole::Fragment,
            metallib: b"ps",
            entry: "ps_other",
        };
        assert!(matches!(
            validate_stage_set(&[vertex, missing], &expected, PipelineStage::Graphics, &parsed),
            Err(RegistryError::InvalidState(_))
        ));
        assert!(matches!(
            validate_stage_set(&[vertex], &expected, PipelineStage::Graphics, &parsed),
            Err(RegistryError::InvalidState(_))
        ));
    }

    #[test]
    fn pipeline_registration_rejects_a_key_that_does_not_match() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let mut registry = PipelineRegistry::new(&transport).expect("registry");
        let key = PipelineKey::from_parts(b"x", "main", PipelineStage::Compute, b"");
        let error = registry
            .register_compute(key, b"y", "main", b"")
            .unwrap_err();
        assert!(matches!(error, RegistryError::KeyMismatch));
        assert!(registry.is_empty());
        assert_eq!(registry.created_count(), 0);
    }

    #[test]
    fn texture_registry_rejects_zero_and_duplicate_ids_without_gpu_objects() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let mut registry = TextureRegistry::new(&transport).expect("registry");
        assert!(matches!(
            registry.register(0, desc()),
            Err(RegistryError::InvalidDesc(_))
        ));
        registry
            .register(7, desc())
            .expect("a valid descriptor must create a real texture");
        assert!(matches!(
            registry.register(7, desc()),
            Err(RegistryError::DuplicateTexture(7))
        ));
        assert!(matches!(
            registry.destroy(8),
            Err(RegistryError::UnknownTexture(8))
        ));
        assert!(registry.get(7).is_some());
        registry
            .destroy(7)
            .expect("registered texture must destroy");
        assert!(registry.get(7).is_none());
    }

    /// Compiles a tiny vertex/fragment metallib, or `None` without Apple's
    /// offline Metal toolchain (the test then skips instead of pretending).
    fn compile_graphics_library() -> Option<Vec<u8>> {
        use std::process::Command;
        let tool = |name: &str| -> Option<std::path::PathBuf> {
            let output = Command::new("xcrun").args(["-f", name]).output().ok()?;
            if !output.status.success() {
                return None;
            }
            let text = String::from_utf8(output.stdout).ok()?;
            let text = text.trim().to_owned();
            (!text.is_empty()).then(|| std::path::PathBuf::from(text))
        };
        let metal = tool("metal")?;
        let metallib = tool("metallib")?;
        let dir = std::env::temp_dir().join(format!(
            "victory-registry-state-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;
        let source = dir.join("state.metal");
        let air = dir.join("state.air");
        let library = dir.join("state.metallib");
        std::fs::write(
            &source,
            r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 vs_main(uint vid [[vertex_id]]) { return float4(0.0, 0.0, 0.0, 1.0); }
fragment float4 fs_main() { return float4(1.0, 0.0, 0.0, 1.0); }
"#,
        )
        .ok()?;
        let compiled = Command::new(metal)
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .output()
            .ok()?;
        if !compiled.status.success() {
            eprintln!(
                "SKIP: metal -c failed: {}",
                String::from_utf8_lossy(&compiled.stderr).trim()
            );
            let _ = std::fs::remove_dir_all(&dir);
            return None;
        }
        let linked = Command::new(metallib)
            .arg(&air)
            .arg("-o")
            .arg(&library)
            .output()
            .ok()?;
        if !linked.status.success() {
            eprintln!(
                "SKIP: metallib failed: {}",
                String::from_utf8_lossy(&linked.stderr).trim()
            );
            let _ = std::fs::remove_dir_all(&dir);
            return None;
        }
        let bytes = std::fs::read(&library).ok();
        let _ = std::fs::remove_dir_all(&dir);
        bytes
    }

    /// ADR-029: the content key covers the render state, so two blend states
    /// create two pipelines and the identical registration is reused.
    #[test]
    fn pipeline_registry_creates_on_blend_change_and_reuses_identical_state() {
        let Some(library) = compile_graphics_library() else {
            eprintln!("SKIP: Apple Metal toolchain not installed");
            return;
        };
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let mut registry = PipelineRegistry::new(&transport).expect("registry");
        let colors = vec![TextureFormat::Rgba8Unorm];
        let make_state = |blend: BlendState| {
            PipelineState {
                color_formats: colors.clone(),
                fragment_entry: Some("fs_main".to_string()),
                blend: Some(blend),
                ..PipelineState::default()
            }
            .encode(PipelineStage::Graphics)
        };
        let additive = BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: vec![BlendAttachment {
                enable: true,
                src_color: BlendFactor::SrcAlpha,
                dst_color: BlendFactor::InvSrcAlpha,
                color_op: BlendOperation::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::Zero,
                alpha_op: BlendOperation::Add,
                write_mask: 0x0f,
            }],
        };
        let disabled = BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: vec![BlendAttachment {
                enable: false,
                src_color: BlendFactor::One,
                dst_color: BlendFactor::Zero,
                color_op: BlendOperation::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::Zero,
                alpha_op: BlendOperation::Add,
                write_mask: 0x0f,
            }],
        };
        let state_a = make_state(additive);
        let key_a = PipelineKey::from_parts(&library, "vs_main", PipelineStage::Graphics, &state_a);
        let first = registry
            .register_graphics(key_a.clone(), &library, "vs_main", &state_a)
            .expect("the blended pipeline registers");
        let reused = registry
            .register_graphics(key_a.clone(), &library, "vs_main", &state_a)
            .expect("the identical registration reuses the pipeline");
        assert_eq!(first, reused, "identical state shares one pipeline id");
        assert_eq!(registry.created_count(), 1);
        assert_eq!(registry.reuse_count(), 1);

        let state_b = make_state(disabled);
        let key_b = PipelineKey::from_parts(&library, "vs_main", PipelineStage::Graphics, &state_b);
        assert_ne!(key_a.digest(), key_b.digest());
        let second = registry
            .register_graphics(key_b, &library, "vs_main", &state_b)
            .expect("the disabled pipeline registers");
        assert_ne!(first, second, "a different blend state is a different PSO id");
        assert_eq!(registry.created_count(), 2);
        assert_eq!(registry.reuse_count(), 1);
    }
}
