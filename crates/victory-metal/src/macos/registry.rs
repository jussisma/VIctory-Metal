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
use crate::macos::vertex_layout::{VertexLayout, VertexLayoutError};
use crate::macos::views::FormatSupport;
use crate::transport::MetalError;
use crate::wire_v3::{
    BlendAttachment, BlendFactor, BlendOperation, BlendState, CompareFunction, CullMode,
    DepthStencilState, FillMode, FrontFaceWinding, RasterizerState, StencilFaceState,
    StencilOperation,
};

// The canonical `MTLTextureType` values, exactly as Metal.framework declares
// them (ADR-042). They are NOT sequential from 2: `Cube` is 5, `CubeArray` 6,
// `3D` 7 and `2DMultisampleArray` 8. Before ADR-042 this table shifted every
// cube/3D constant up by one, so a 3D texture was created as a multisample
// array and the 33x33x33 FP16 volume asserted on the host.
/// `MTLTextureType1D`.
const MTL_TEXTURE_TYPE_1D: usize = 0;
/// `MTLTextureType1DArray`.
const MTL_TEXTURE_TYPE_1D_ARRAY: usize = 1;
/// `MTLTextureType2D`.
const MTL_TEXTURE_TYPE_2D: usize = 2;
/// `MTLTextureType2DArray`.
const MTL_TEXTURE_TYPE_2D_ARRAY: usize = 3;
/// `MTLTextureType2DMultisample`.
const MTL_TEXTURE_TYPE_2D_MULTISAMPLE: usize = 4;
/// `MTLTextureTypeCube`.
const MTL_TEXTURE_TYPE_CUBE: usize = 5;
/// `MTLTextureTypeCubeArray`.
const MTL_TEXTURE_TYPE_CUBE_ARRAY: usize = 6;
/// `MTLTextureType3D`.
const MTL_TEXTURE_TYPE_3D: usize = 7;
/// `MTLTextureType2DMultisampleArray`.
const MTL_TEXTURE_TYPE_2D_MULTISAMPLE_ARRAY: usize = 8;

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

/// `MTLVertexStepFunctionPerVertex`.
const MTL_VERTEX_STEP_FUNCTION_PER_VERTEX: usize = 1;

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
    /// A vertex-layout chunk in the state blob is malformed.
    #[error("vertex layout is invalid: {0}")]
    InvalidVertexLayout(String),
    /// A vertex-layout chunk names a format this host cannot express in Metal.
    #[error("vertex attribute format code {0} is not supported")]
    UnsupportedVertexFormat(u8),
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

/// The canonical `MTLTextureType` for a dimension, sample count and layer
/// count (ADR-042).
///
/// This is the one table both a resource descriptor and a typed view plan go
/// through, so neither can drift from Metal's enum again. `array_layers` is
/// the descriptor's array length for a resource and the view's slice count for
/// a view; a single layer of an arrayable dimension is a plain 2D (or 2D
/// multisample) texture, exactly as `newTextureViewWithPixelFormat:` expects.
/// `D1` with more than one layer is the 1D array Metal also declares.
pub(crate) const fn canonical_texture_type(
    dimension: TextureDimension,
    sample_count: u32,
    array_layers: u32,
) -> usize {
    use TextureDimension::{Cube, CubeArray, D1, D2, D2Array, D3};
    match dimension {
        D1 if array_layers > 1 => MTL_TEXTURE_TYPE_1D_ARRAY,
        D1 => MTL_TEXTURE_TYPE_1D,
        D2 if sample_count > 1 => MTL_TEXTURE_TYPE_2D_MULTISAMPLE,
        D2 => MTL_TEXTURE_TYPE_2D,
        D2Array if sample_count > 1 && array_layers > 1 => MTL_TEXTURE_TYPE_2D_MULTISAMPLE_ARRAY,
        D2Array if sample_count > 1 => MTL_TEXTURE_TYPE_2D_MULTISAMPLE,
        D2Array if array_layers > 1 => MTL_TEXTURE_TYPE_2D_ARRAY,
        D2Array => MTL_TEXTURE_TYPE_2D,
        // A cube array with exactly one cube (six layers) is the `Cube` type;
        // only more than one cube selects `CubeArray`. `validate` refuses
        // non-multiple-of-six layer counts before Metal sees them.
        Cube | CubeArray if array_layers == 6 => MTL_TEXTURE_TYPE_CUBE,
        // A single face of a cube texture is a plain 2D view.
        Cube if array_layers == 1 => MTL_TEXTURE_TYPE_2D,
        Cube | CubeArray if array_layers % 6 == 0 && array_layers > 6 => {
            MTL_TEXTURE_TYPE_CUBE_ARRAY
        }
        Cube | CubeArray => MTL_TEXTURE_TYPE_CUBE,
        D3 => MTL_TEXTURE_TYPE_3D,
    }
}

/// Host texture formats mirrored from D3D12/DXGI.
///
/// `TextureFormat` is a handle into [`FORMAT_TABLE`], the one canonical
/// `DXGI_FORMAT` -> `MTLPixelFormat` table shared by the Metal registry, the
/// native views layer and the guest-side pipeline/registration code
/// (ADR-033). The numeric `DXGI_FORMAT_*` values are the ones in the Windows
/// SDK, cross-checked against `windows` crate 0.62 (`Win32::Graphics::Dxgi::
/// Common`): 62 is `R8_UINT`, 63 `R8_SNORM`, 64 `R8_SINT`, 65 `A8_UNORM`,
/// 57 `R16_UINT`, 87 `B8G8R8A8_UNORM` and 88 `B8G8R8X8_UNORM`.
///
/// The associated constants preserve every name the existing registry code
/// and its callers used. The private index keeps the table authoritative:
/// [`TextureFormat::mtl_pixel_format`], [`TextureFormat::code`] and the family
/// questions all read the same entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureFormat(u16);

/// A DXGI value this build cannot represent on Metal.
///
/// The value is kept so callers can name it; [`Self::name`] returns the
/// canonical `DXGI_FORMAT_*` name instead of a guessed substitute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnsupportedDxgiFormat(pub u32);

impl UnsupportedDxgiFormat {
    /// Canonical name of the refused format.
    pub const fn name(self) -> &'static str {
        match self.0 {
            0 => "DXGI_FORMAT_UNKNOWN",
            5 => "DXGI_FORMAT_R32G32B32_TYPELESS",
            6 => "DXGI_FORMAT_R32G32B32_FLOAT",
            7 => "DXGI_FORMAT_R32G32B32_UINT",
            8 => "DXGI_FORMAT_R32G32B32_SINT",
            68 => "DXGI_FORMAT_R8G8_B8G8_UNORM",
            69 => "DXGI_FORMAT_G8R8_G8B8_UNORM",
            89 => "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM",
            103..=132 => "a video/planar DXGI format",
            _ => "an unmapped DXGI format",
        }
    }
}

impl std::fmt::Display for UnsupportedDxgiFormat {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{} ({}) has no Metal pixel format", self.name(), self.0)
    }
}

/// What a pixel format is used for as an attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatKind {
    /// Colour (including packed and block-compressed formats).
    Colour,
    /// Depth only (`D32_FLOAT`, `D16_UNORM`).
    Depth,
    /// Combined depth and stencil (`D24_UNORM_S8_UINT`,
    /// `D32_FLOAT_S8X24_UINT`).
    DepthStencil,
    /// Stencil-only read view (`X32_STENCIL8`, `X24_STENCIL8`).
    Stencil,
}

impl FormatKind {
    /// Whether the kind carries guest depth state.
    pub const fn is_depth(self) -> bool {
        matches!(self, Self::Depth | Self::DepthStencil)
    }
}

/// View-compatibility family for typeless reinterpretation.
///
/// D3D12 groups formats by element size and category: a typeless resource may
/// be viewed with any format in its family, never across families. The sizes
/// are the whole element sizes Metal's `newTextureViewWithPixelFormat:` also
/// accepts, so `R8G8B8A8_TYPELESS` can be viewed as `R32_UINT` (both 32 bits)
/// but not as `R16G16B16A16_FLOAT` (64 bits) or `D32_FLOAT` (depth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewClass {
    /// A 128-bit colour element (16 bytes).
    Bits128,
    /// A 64-bit colour element (8 bytes).
    Bits64,
    /// A 32-bit colour element (4 bytes).
    Bits32,
    /// A 16-bit colour element (2 bytes).
    Bits16,
    /// An 8-bit colour element (1 byte).
    Bits8,
    /// An 8-byte 4x4 block (`BC1`, `BC4`).
    Block64,
    /// A 16-byte 4x4 block (`BC2`, `BC3`, `BC5`, `BC6H`, `BC7`).
    Block128,
    /// A depth-only plane.
    Depth,
    /// A combined depth/stencil plane.
    DepthStencil,
    /// A stencil-only plane.
    Stencil,
}

/// One row of the canonical format table.
///
/// Every field answers one question the renderer asks of a guest format: the
/// DXGI value, the Metal pixel format, the stable PSO code, the stable kind 8
/// wire code, the depth/stencil kind, the view-compatibility family, the
/// element size, the block size, the sRGB bit, the concrete base of a typeless
/// format, and whether a pixel-format view may reinterpret it.
#[derive(Debug, Clone, Copy)]
pub struct FormatEntry {
    /// Handle this row describes.
    pub format: TextureFormat,
    /// Canonical `DXGI_FORMAT_*` name.
    pub name: &'static str,
    /// Canonical DXGI value; `None` for the Metal-only sRGB mirrors.
    pub dxgi: Option<u32>,
    /// `MTLPixelFormat` value for this row's nominal format.
    pub mtl: usize,
    /// Stable one-byte `VPS1`/`VPS2` format code.
    pub code: u8,
    /// Stable one-byte kind 8 host-registration code.
    pub wire: u8,
    /// View-compatibility family.
    pub view_class: ViewClass,
    /// Depth/stencil kind.
    pub kind: FormatKind,
    /// Uncompressed bytes per pixel (`0` for depth and block formats).
    pub bytes_per_pixel: u8,
    /// Bytes in one 4x4 block (`0` for uncompressed and depth formats).
    pub block_bytes: u8,
    /// Whether this is the sRGB variant of its family.
    pub srgb: bool,
    /// For a `*_TYPELESS` resource format: the concrete format to create with.
    pub typeless_base: Option<TextureFormat>,
    /// Whether a pixel-format view may reinterpret a resource of this family.
    pub view_compatible: bool,
}

macro_rules! format_table {
    ($count:literal; $(
        $idx:literal => $variant:ident ($name:literal): dxgi $dxgi:expr, mtl $mtl:literal,
        code $code:literal, wire $wire:literal, class $class:ident, kind $kind:ident,
        bpp $bpp:literal, block $block:literal, srgb $srgb:literal,
        typeless $typeless:expr, view $view:literal;
    )*) => {
        #[allow(non_upper_case_globals)]
        impl TextureFormat {
            $(pub const $variant: Self = Self($idx);)*

            /// `MTLPixelFormatBC4_RUnorm` (the pre-ADR-033 registry name).
            pub const Bc4Rgba: Self = Self::Bc4RUnorm;
            /// `MTLPixelFormatBC5_RGUnorm` (the pre-ADR-033 registry name).
            pub const Bc5Rgba: Self = Self::Bc5RgUnorm;

            /// Every mapped format, in canonical table order.
            pub const ALL: [Self; $count] = [$(Self($idx)),*];
        }

        /// The canonical `DXGI_FORMAT` -> `MTLPixelFormat` table (ADR-033).
        pub const FORMAT_TABLE: &[FormatEntry] = &[
            $(FormatEntry {
                format: TextureFormat($idx),
                name: $name,
                dxgi: $dxgi,
                mtl: $mtl,
                code: $code,
                wire: $wire,
                view_class: ViewClass::$class,
                kind: FormatKind::$kind,
                bytes_per_pixel: $bpp,
                block_bytes: $block,
                srgb: $srgb,
                typeless_base: $typeless,
                view_compatible: $view,
            }),*
        ];
    };
}

format_table! {
    93;
    0 => Rgba32Float ("DXGI_FORMAT_R32G32B32A32_FLOAT"): dxgi Some(2), mtl 125,
        code 20, wire 12, class Bits128, kind Colour, bpp 16, block 0, srgb false,
        typeless None, view true;
    1 => Rgba32Uint ("DXGI_FORMAT_R32G32B32A32_UINT"): dxgi Some(3), mtl 123,
        code 21, wire 13, class Bits128, kind Colour, bpp 16, block 0, srgb false,
        typeless None, view true;
    2 => Rgba32Sint ("DXGI_FORMAT_R32G32B32A32_SINT"): dxgi Some(4), mtl 124,
        code 22, wire 14, class Bits128, kind Colour, bpp 16, block 0, srgb false,
        typeless None, view true;
    3 => Rgba32Typeless ("DXGI_FORMAT_R32G32B32A32_TYPELESS"): dxgi Some(1), mtl 125,
        code 23, wire 15, class Bits128, kind Colour, bpp 16, block 0, srgb false,
        typeless Some(TextureFormat::Rgba32Float), view true;
    4 => Rgba16Float ("DXGI_FORMAT_R16G16B16A16_FLOAT"): dxgi Some(10), mtl 115,
        code 17, wire 9, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    5 => Rgba16Unorm ("DXGI_FORMAT_R16G16B16A16_UNORM"): dxgi Some(11), mtl 110,
        code 24, wire 16, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    6 => Rgba16Uint ("DXGI_FORMAT_R16G16B16A16_UINT"): dxgi Some(12), mtl 113,
        code 25, wire 17, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    7 => Rgba16Snorm ("DXGI_FORMAT_R16G16B16A16_SNORM"): dxgi Some(13), mtl 112,
        code 26, wire 18, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    8 => Rgba16Sint ("DXGI_FORMAT_R16G16B16A16_SINT"): dxgi Some(14), mtl 114,
        code 27, wire 19, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    9 => Rgba16Typeless ("DXGI_FORMAT_R16G16B16A16_TYPELESS"): dxgi Some(9), mtl 115,
        code 28, wire 20, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless Some(TextureFormat::Rgba16Float), view true;
    10 => Rg32Float ("DXGI_FORMAT_R32G32_FLOAT"): dxgi Some(16), mtl 105,
        code 29, wire 21, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    11 => Rg32Uint ("DXGI_FORMAT_R32G32_UINT"): dxgi Some(17), mtl 103,
        code 30, wire 22, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    12 => Rg32Sint ("DXGI_FORMAT_R32G32_SINT"): dxgi Some(18), mtl 104,
        code 31, wire 23, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless None, view true;
    13 => Rg32Typeless ("DXGI_FORMAT_R32G32_TYPELESS"): dxgi Some(15), mtl 105,
        code 32, wire 24, class Bits64, kind Colour, bpp 8, block 0, srgb false,
        typeless Some(TextureFormat::Rg32Float), view true;
    14 => Depth32FloatStencil8 ("DXGI_FORMAT_D32_FLOAT_S8X24_UINT"): dxgi Some(20), mtl 260,
        code 16, wire 8, class DepthStencil, kind DepthStencil, bpp 0, block 0, srgb false,
        typeless None, view true;
    15 => R32FloatX8X24Typeless ("DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS"): dxgi Some(21), mtl 252,
        code 33, wire 25, class DepthStencil, kind Depth, bpp 0, block 0, srgb false,
        typeless None, view true;
    16 => X32TypelessG8X24Uint ("DXGI_FORMAT_X32_TYPELESS_G8X24_UINT"): dxgi Some(22), mtl 261,
        code 34, wire 26, class Stencil, kind Stencil, bpp 0, block 0, srgb false,
        typeless None, view true;
    17 => R32G8X24Typeless ("DXGI_FORMAT_R32G8X24_TYPELESS"): dxgi Some(19), mtl 260,
        code 35, wire 27, class DepthStencil, kind DepthStencil, bpp 0, block 0, srgb false,
        typeless Some(TextureFormat::Depth32FloatStencil8), view true;
    18 => Rgb10a2Unorm ("DXGI_FORMAT_R10G10B10A2_UNORM"): dxgi Some(24), mtl 90,
        code 11, wire 6, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    19 => Rgb10a2Uint ("DXGI_FORMAT_R10G10B10A2_UINT"): dxgi Some(25), mtl 91,
        code 36, wire 28, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    20 => Rgb10a2Typeless ("DXGI_FORMAT_R10G10B10A2_TYPELESS"): dxgi Some(23), mtl 90,
        code 37, wire 29, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::Rgb10a2Unorm), view true;
    21 => Rg11g11b10Float ("DXGI_FORMAT_R11G11B10_FLOAT"): dxgi Some(26), mtl 92,
        code 10, wire 5, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    22 => Rgba8Unorm ("DXGI_FORMAT_R8G8B8A8_UNORM"): dxgi Some(28), mtl 70,
        code 7, wire 0, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    23 => Rgba8Srgb ("DXGI_FORMAT_R8G8B8A8_UNORM_SRGB"): dxgi Some(29), mtl 71,
        code 8, wire 2, class Bits32, kind Colour, bpp 4, block 0, srgb true,
        typeless None, view true;
    24 => Rgba8Uint ("DXGI_FORMAT_R8G8B8A8_UINT"): dxgi Some(30), mtl 73,
        code 38, wire 30, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    25 => Rgba8Snorm ("DXGI_FORMAT_R8G8B8A8_SNORM"): dxgi Some(31), mtl 72,
        code 39, wire 31, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    26 => Rgba8Sint ("DXGI_FORMAT_R8G8B8A8_SINT"): dxgi Some(32), mtl 74,
        code 40, wire 32, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    27 => Rgba8Typeless ("DXGI_FORMAT_R8G8B8A8_TYPELESS"): dxgi Some(27), mtl 70,
        code 41, wire 33, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::Rgba8Unorm), view true;
    28 => Rg16Float ("DXGI_FORMAT_R16G16_FLOAT"): dxgi Some(34), mtl 65,
        code 42, wire 34, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    29 => Rg16Unorm ("DXGI_FORMAT_R16G16_UNORM"): dxgi Some(35), mtl 60,
        code 43, wire 35, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    30 => Rg16Uint ("DXGI_FORMAT_R16G16_UINT"): dxgi Some(36), mtl 63,
        code 44, wire 36, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    31 => Rg16Snorm ("DXGI_FORMAT_R16G16_SNORM"): dxgi Some(37), mtl 62,
        code 45, wire 37, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    32 => Rg16Sint ("DXGI_FORMAT_R16G16_SINT"): dxgi Some(38), mtl 64,
        code 46, wire 38, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    33 => Rg16Typeless ("DXGI_FORMAT_R16G16_TYPELESS"): dxgi Some(33), mtl 65,
        code 47, wire 39, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::Rg16Float), view true;
    34 => R32Float ("DXGI_FORMAT_R32_FLOAT"): dxgi Some(41), mtl 55,
        code 12, wire 3, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    35 => R32Uint ("DXGI_FORMAT_R32_UINT"): dxgi Some(42), mtl 53,
        code 18, wire 10, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    36 => R32Sint ("DXGI_FORMAT_R32_SINT"): dxgi Some(43), mtl 54,
        code 19, wire 11, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    37 => Depth32Float ("DXGI_FORMAT_D32_FLOAT"): dxgi Some(40), mtl 252,
        code 14, wire 7, class Depth, kind Depth, bpp 0, block 0, srgb false,
        typeless None, view true;
    38 => R32Typeless ("DXGI_FORMAT_R32_TYPELESS"): dxgi Some(39), mtl 55,
        code 48, wire 40, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::R32Float), view true;
    39 => Depth24Stencil8 ("DXGI_FORMAT_D24_UNORM_S8_UINT"): dxgi Some(45), mtl 255,
        code 15, wire 41, class DepthStencil, kind DepthStencil, bpp 0, block 0, srgb false,
        typeless None, view true;
    40 => R24UnormX8Typeless ("DXGI_FORMAT_R24_UNORM_X8_TYPELESS"): dxgi Some(46), mtl 252,
        code 49, wire 42, class DepthStencil, kind Depth, bpp 0, block 0, srgb false,
        typeless None, view true;
    41 => X24TypelessG8Uint ("DXGI_FORMAT_X24_TYPELESS_G8_UINT"): dxgi Some(47), mtl 261,
        code 50, wire 43, class Stencil, kind Stencil, bpp 0, block 0, srgb false,
        typeless None, view true;
    42 => R24G8Typeless ("DXGI_FORMAT_R24G8_TYPELESS"): dxgi Some(44), mtl 260,
        code 51, wire 44, class DepthStencil, kind DepthStencil, bpp 0, block 0, srgb false,
        typeless Some(TextureFormat::Depth24Stencil8), view true;
    43 => Rg8Unorm ("DXGI_FORMAT_R8G8_UNORM"): dxgi Some(49), mtl 30,
        code 52, wire 45, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    44 => Rg8Uint ("DXGI_FORMAT_R8G8_UINT"): dxgi Some(50), mtl 33,
        code 53, wire 46, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    45 => Rg8Snorm ("DXGI_FORMAT_R8G8_SNORM"): dxgi Some(51), mtl 32,
        code 54, wire 47, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    46 => Rg8Sint ("DXGI_FORMAT_R8G8_SINT"): dxgi Some(52), mtl 34,
        code 55, wire 48, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    47 => Rg8Typeless ("DXGI_FORMAT_R8G8_TYPELESS"): dxgi Some(48), mtl 30,
        code 56, wire 49, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless Some(TextureFormat::Rg8Unorm), view true;
    48 => R16Float ("DXGI_FORMAT_R16_FLOAT"): dxgi Some(54), mtl 25,
        code 13, wire 4, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    49 => R16Unorm ("DXGI_FORMAT_R16_UNORM"): dxgi Some(56), mtl 20,
        code 57, wire 50, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    50 => R16Uint ("DXGI_FORMAT_R16_UINT"): dxgi Some(57), mtl 23,
        code 58, wire 51, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    51 => R16Snorm ("DXGI_FORMAT_R16_SNORM"): dxgi Some(58), mtl 22,
        code 59, wire 52, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    52 => R16Sint ("DXGI_FORMAT_R16_SINT"): dxgi Some(59), mtl 24,
        code 60, wire 53, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view true;
    53 => Depth16Unorm ("DXGI_FORMAT_D16_UNORM"): dxgi Some(55), mtl 250,
        code 61, wire 54, class Depth, kind Depth, bpp 0, block 0, srgb false,
        typeless None, view true;
    54 => R16Typeless ("DXGI_FORMAT_R16_TYPELESS"): dxgi Some(53), mtl 25,
        code 62, wire 55, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless Some(TextureFormat::R16Float), view true;
    55 => R8Unorm ("DXGI_FORMAT_R8_UNORM"): dxgi Some(61), mtl 10,
        code 63, wire 56, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless None, view true;
    56 => R8Uint ("DXGI_FORMAT_R8_UINT"): dxgi Some(62), mtl 13,
        code 64, wire 57, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless None, view true;
    57 => R8Snorm ("DXGI_FORMAT_R8_SNORM"): dxgi Some(63), mtl 12,
        code 65, wire 58, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless None, view true;
    58 => R8Sint ("DXGI_FORMAT_R8_SINT"): dxgi Some(64), mtl 14,
        code 66, wire 59, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless None, view true;
    59 => R8Typeless ("DXGI_FORMAT_R8_TYPELESS"): dxgi Some(60), mtl 10,
        code 67, wire 60, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless Some(TextureFormat::R8Unorm), view true;
    60 => A8Unorm ("DXGI_FORMAT_A8_UNORM"): dxgi Some(65), mtl 1,
        code 68, wire 61, class Bits8, kind Colour, bpp 1, block 0, srgb false,
        typeless None, view false;
    61 => Rg9b9e5Float ("DXGI_FORMAT_R9G9B9E5_SHAREDEXP"): dxgi Some(67), mtl 93,
        code 69, wire 62, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view false;
    62 => B5g6r5Unorm ("DXGI_FORMAT_B5G6R5_UNORM"): dxgi Some(85), mtl 40,
        code 70, wire 63, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view false;
    63 => Bgr5a1Unorm ("DXGI_FORMAT_B5G5R5A1_UNORM"): dxgi Some(86), mtl 43,
        code 71, wire 64, class Bits16, kind Colour, bpp 2, block 0, srgb false,
        typeless None, view false;
    64 => Bgra8Unorm ("DXGI_FORMAT_B8G8R8A8_UNORM"): dxgi Some(87), mtl 80,
        code 9, wire 1, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    65 => Bgra8Srgb ("DXGI_FORMAT_B8G8R8A8_UNORM_SRGB"): dxgi Some(91), mtl 81,
        code 72, wire 65, class Bits32, kind Colour, bpp 4, block 0, srgb true,
        typeless None, view true;
    66 => Bgra8Typeless ("DXGI_FORMAT_B8G8R8A8_TYPELESS"): dxgi Some(90), mtl 80,
        code 73, wire 66, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::Bgra8Unorm), view true;
    67 => Bgrx8Unorm ("DXGI_FORMAT_B8G8R8X8_UNORM"): dxgi Some(88), mtl 80,
        code 74, wire 67, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless None, view true;
    68 => Bgrx8Srgb ("DXGI_FORMAT_B8G8R8X8_UNORM_SRGB"): dxgi Some(93), mtl 81,
        code 75, wire 68, class Bits32, kind Colour, bpp 4, block 0, srgb true,
        typeless None, view true;
    69 => Bgrx8Typeless ("DXGI_FORMAT_B8G8R8X8_TYPELESS"): dxgi Some(92), mtl 80,
        code 76, wire 69, class Bits32, kind Colour, bpp 4, block 0, srgb false,
        typeless Some(TextureFormat::Bgrx8Unorm), view true;
    70 => Bc1Rgba ("DXGI_FORMAT_BC1_UNORM"): dxgi Some(71), mtl 130,
        code 0, wire 70, class Block64, kind Colour, bpp 0, block 8, srgb false,
        typeless None, view true;
    71 => Bc1RgbaSrgb ("DXGI_FORMAT_BC1_UNORM_SRGB"): dxgi Some(72), mtl 131,
        code 77, wire 71, class Block64, kind Colour, bpp 0, block 8, srgb true,
        typeless None, view true;
    72 => Bc2Rgba ("DXGI_FORMAT_BC2_UNORM"): dxgi Some(74), mtl 132,
        code 1, wire 72, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    73 => Bc2RgbaSrgb ("DXGI_FORMAT_BC2_UNORM_SRGB"): dxgi Some(75), mtl 133,
        code 78, wire 73, class Block128, kind Colour, bpp 0, block 16, srgb true,
        typeless None, view true;
    74 => Bc3Rgba ("DXGI_FORMAT_BC3_UNORM"): dxgi Some(77), mtl 134,
        code 2, wire 74, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    75 => Bc3RgbaSrgb ("DXGI_FORMAT_BC3_UNORM_SRGB"): dxgi Some(78), mtl 135,
        code 79, wire 75, class Block128, kind Colour, bpp 0, block 16, srgb true,
        typeless None, view true;
    76 => Bc4RUnorm ("DXGI_FORMAT_BC4_UNORM"): dxgi Some(80), mtl 140,
        code 3, wire 76, class Block64, kind Colour, bpp 0, block 8, srgb false,
        typeless None, view true;
    77 => Bc4RSnorm ("DXGI_FORMAT_BC4_SNORM"): dxgi Some(81), mtl 141,
        code 80, wire 77, class Block64, kind Colour, bpp 0, block 8, srgb false,
        typeless None, view true;
    78 => Bc5RgUnorm ("DXGI_FORMAT_BC5_UNORM"): dxgi Some(83), mtl 142,
        code 4, wire 78, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    79 => Bc5RgSnorm ("DXGI_FORMAT_BC5_SNORM"): dxgi Some(84), mtl 143,
        code 81, wire 79, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    80 => Bc6hRgbUf16 ("DXGI_FORMAT_BC6H_UF16"): dxgi Some(95), mtl 151,
        code 5, wire 80, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    81 => Bc6hRgbSf16 ("DXGI_FORMAT_BC6H_SF16"): dxgi Some(96), mtl 150,
        code 82, wire 81, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    82 => Bc7Rgba ("DXGI_FORMAT_BC7_UNORM"): dxgi Some(98), mtl 152,
        code 6, wire 82, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless None, view true;
    83 => Bc7RgbaSrgb ("DXGI_FORMAT_BC7_UNORM_SRGB"): dxgi Some(99), mtl 153,
        code 83, wire 83, class Block128, kind Colour, bpp 0, block 16, srgb true,
        typeless None, view true;
    84 => Bc1Typeless ("DXGI_FORMAT_BC1_TYPELESS"): dxgi Some(70), mtl 130,
        code 84, wire 84, class Block64, kind Colour, bpp 0, block 8, srgb false,
        typeless Some(TextureFormat::Bc1Rgba), view true;
    85 => Bc2Typeless ("DXGI_FORMAT_BC2_TYPELESS"): dxgi Some(73), mtl 132,
        code 85, wire 85, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless Some(TextureFormat::Bc2Rgba), view true;
    86 => Bc3Typeless ("DXGI_FORMAT_BC3_TYPELESS"): dxgi Some(76), mtl 134,
        code 86, wire 86, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless Some(TextureFormat::Bc3Rgba), view true;
    87 => Bc4Typeless ("DXGI_FORMAT_BC4_TYPELESS"): dxgi Some(79), mtl 140,
        code 87, wire 87, class Block64, kind Colour, bpp 0, block 8, srgb false,
        typeless Some(TextureFormat::Bc4RUnorm), view true;
    88 => Bc5Typeless ("DXGI_FORMAT_BC5_TYPELESS"): dxgi Some(82), mtl 142,
        code 88, wire 88, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless Some(TextureFormat::Bc5RgUnorm), view true;
    89 => Bc6hTypeless ("DXGI_FORMAT_BC6H_TYPELESS"): dxgi Some(94), mtl 151,
        code 89, wire 89, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless Some(TextureFormat::Bc6hRgbUf16), view true;
    90 => Bc7Typeless ("DXGI_FORMAT_BC7_TYPELESS"): dxgi Some(97), mtl 152,
        code 90, wire 90, class Block128, kind Colour, bpp 0, block 16, srgb false,
        typeless Some(TextureFormat::Bc7Rgba), view true;
    91 => R8UnormSrgb ("Metal MTLPixelFormatR8Unorm_sRGB"): dxgi None, mtl 11,
        code 91, wire 91, class Bits8, kind Colour, bpp 1, block 0, srgb true,
        typeless None, view true;
    92 => Rg8UnormSrgb ("Metal MTLPixelFormatRG8Unorm_sRGB"): dxgi None, mtl 31,
        code 92, wire 92, class Bits16, kind Colour, bpp 2, block 0, srgb true,
        typeless None, view true;
}

impl TextureFormat {
    /// The canonical table row for this format.
    pub const fn entry(self) -> &'static FormatEntry {
        &FORMAT_TABLE[self.0 as usize]
    }

    /// Human-readable DXGI-style name for diagnostics.
    pub const fn name(self) -> &'static str {
        self.entry().name
    }

    /// The canonical `DXGI_FORMAT_*` value, or `None` for a Metal-only mirror.
    pub const fn dxgi(self) -> Option<u32> {
        self.entry().dxgi
    }

    /// The nominal Metal pixel format.
    ///
    /// For a `*_TYPELESS` resource format this is the concrete base the texture
    /// is created with; typed views select their own family member.
    pub const fn mtl_pixel_format(self) -> usize {
        self.entry().mtl
    }

    /// Stable one-byte `VPS1`/`VPS2` code. Unique per table row.
    pub const fn code(self) -> u8 {
        self.entry().code
    }

    /// Stable one-byte kind 8 host-registration code. Unique per table row.
    pub const fn wire_code(self) -> u8 {
        self.entry().wire
    }

    /// View-compatibility family.
    pub const fn view_class(self) -> ViewClass {
        self.entry().view_class
    }

    /// Depth/stencil kind.
    pub const fn kind(self) -> FormatKind {
        self.entry().kind
    }

    /// Whether this is a depth or combined depth/stencil format.
    pub const fn is_depth_stencil(self) -> bool {
        self.entry().kind.is_depth()
    }

    /// Whether this format carries a stencil plane.
    pub const fn has_stencil(self) -> bool {
        matches!(self.entry().kind, FormatKind::DepthStencil)
    }

    /// Whether this is a block-compressed colour format.
    pub const fn is_block_compressed(self) -> bool {
        self.entry().block_bytes != 0
    }

    /// Whether this format may back a colour render target.
    pub const fn is_color_render_target(self) -> bool {
        matches!(self.entry().kind, FormatKind::Colour) && !self.is_block_compressed()
    }

    /// Whether this is a `*_TYPELESS` resource format.
    pub const fn is_typeless(self) -> bool {
        self.entry().typeless_base.is_some()
    }

    /// For a `*_TYPELESS` format, the concrete format to create the texture
    /// with; `None` for typed formats.
    pub const fn typeless_base(self) -> Option<Self> {
        self.entry().typeless_base
    }

    /// Whether this row is the sRGB variant of its family.
    pub const fn is_srgb(self) -> bool {
        self.entry().srgb
    }

    /// Whether a pixel-format view may reinterpret a resource of this family.
    pub const fn is_view_compatible(self) -> bool {
        self.entry().view_compatible
    }

    /// Uncompressed bytes per pixel (`None` for depth and BC formats).
    pub const fn bytes_per_pixel(self) -> Option<u32> {
        if self.entry().bytes_per_pixel == 0 {
            None
        } else {
            Some(self.entry().bytes_per_pixel as u32)
        }
    }

    /// Bytes in one 4x4 BC block (`None` for uncompressed and depth formats).
    pub const fn block_bytes(self) -> Option<u32> {
        if self.entry().block_bytes == 0 {
            None
        } else {
            Some(self.entry().block_bytes as u32)
        }
    }

    /// Exact bytes in one row of `width` pixels, without pitch padding.
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

    /// Resolves a canonical guest `DXGI_FORMAT_*` value.
    ///
    /// A typeless value resolves to its typeless table row; callers that need
    /// a concrete creation format use [`Self::typeless_base`]. A value Metal
    /// cannot represent is [`UnsupportedDxgiFormat`], named, never guessed.
    pub fn from_dxgi(value: u32) -> Result<Self, UnsupportedDxgiFormat> {
        FORMAT_TABLE
            .iter()
            .find(|entry| entry.dxgi == Some(value))
            .map(|entry| entry.format)
            .ok_or(UnsupportedDxgiFormat(value))
    }

    /// Decodes a stable one-byte `VPS1`/`VPS2` format code.
    pub fn from_code(code: u8) -> Option<Self> {
        FORMAT_TABLE
            .iter()
            .find(|entry| entry.code == code)
            .map(|entry| entry.format)
    }

    /// Decodes a stable one-byte kind 8 host-registration code.
    pub fn from_wire_code(code: u8) -> Option<Self> {
        FORMAT_TABLE
            .iter()
            .find(|entry| entry.wire == code)
            .map(|entry| entry.format)
    }

    /// The Metal pixel format a descriptor may use on `support`'s device.
    pub const fn mtl_pixel_format_on(self, support: FormatSupport) -> usize {
        match self {
            Self::Depth24Stencil8 if !support.depth24_stencil8 => 260,
            Self::Depth24Stencil8 => 255,
            _ => self.mtl_pixel_format(),
        }
    }

    /// The stencil-plane pixel format for a combined depth/stencil format, or
    /// `None` for formats without a stencil plane.
    pub const fn stencil_plane_format(self, support: FormatSupport) -> Option<usize> {
        match self {
            Self::Depth24Stencil8 => Some(if support.depth24_stencil8 { 262 } else { 261 }),
            Self::Depth32FloatStencil8 => Some(261),
            _ => None,
        }
    }

    /// Reverse lookup from a Metal pixel format.
    ///
    /// Several guest formats can share one Metal format (a typeless base and
    /// its typed member, or `B8G8R8X8` and `B8G8R8A8`). The typed, non-typeless
    /// row is preferred so callers get an uploadable format.
    pub fn from_mtl_pixel_format(pixel_format: usize) -> Option<Self> {
        FORMAT_TABLE
            .iter()
            .find(|entry| entry.mtl == pixel_format && !entry.format.is_typeless())
            .or_else(|| FORMAT_TABLE.iter().find(|entry| entry.mtl == pixel_format))
            .map(|entry| entry.format)
    }

    /// The paired sRGB or linear format, when the pair keeps the same memory
    /// layout (D3D12 allows this cast for SRVs and RTVs).
    pub const fn srgb_pair(self) -> Option<Self> {
        Some(match self {
            Self::R8Unorm => Self::R8UnormSrgb,
            Self::R8UnormSrgb => Self::R8Unorm,
            Self::Rg8Unorm => Self::Rg8UnormSrgb,
            Self::Rg8UnormSrgb => Self::Rg8Unorm,
            Self::Rgba8Unorm => Self::Rgba8Srgb,
            Self::Rgba8Srgb => Self::Rgba8Unorm,
            Self::Bgra8Unorm => Self::Bgra8Srgb,
            Self::Bgra8Srgb => Self::Bgra8Unorm,
            Self::Bgrx8Unorm => Self::Bgrx8Srgb,
            Self::Bgrx8Srgb => Self::Bgrx8Unorm,
            Self::Bc1Rgba => Self::Bc1RgbaSrgb,
            Self::Bc1RgbaSrgb => Self::Bc1Rgba,
            Self::Bc2Rgba => Self::Bc2RgbaSrgb,
            Self::Bc2RgbaSrgb => Self::Bc2Rgba,
            Self::Bc3Rgba => Self::Bc3RgbaSrgb,
            Self::Bc3RgbaSrgb => Self::Bc3Rgba,
            Self::Bc7Rgba => Self::Bc7RgbaSrgb,
            Self::Bc7RgbaSrgb => Self::Bc7Rgba,
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
        if matches!(self.format.kind(), FormatKind::Stencil) {
            return Err(RegistryError::InvalidDesc(
                "a stencil-only view format cannot back a texture",
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
        canonical_texture_type(self.dimension, self.sample_count, self.array_layers)
    }
}

/// Whether a typed view of a resource stored as `stored` may use `view`.
///
/// D3D12 typeless reinterpretation stays inside one element-size family; a
/// combined depth/stencil resource may additionally expose its depth or
/// stencil plane. Depth never mixes with colour.
fn view_classes_compatible(stored: TextureFormat, view: TextureFormat) -> bool {
    use FormatKind::{Colour, Depth, DepthStencil, Stencil};
    match (stored.kind(), view.kind()) {
        (DepthStencil, Depth | DepthStencil | Stencil) => true,
        (Depth, Depth) => true,
        (Stencil, Stencil) => true,
        (Colour, Colour) => stored.view_class() == view.view_class(),
        _ => false,
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
    #[cfg(test)]
    fn resolve(
        desc: &TextureDesc,
        kind: TextureViewKind,
        sub: TextureSubresource,
    ) -> Result<Self, RegistryError> {
        Self::resolve_as(
            desc,
            kind,
            sub,
            desc.format,
            FormatSupport {
                bc_texture_compression: true,
                depth24_stencil8: false,
            },
        )
    }

    fn resolve_as(
        desc: &TextureDesc,
        kind: TextureViewKind,
        sub: TextureSubresource,
        format: TextureFormat,
        support: FormatSupport,
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
                // One attachment layer: 2D, or 2D multisample when the source
                // texture is multisampled.
                canonical_texture_type(D2Array, desc.sample_count, 1)
            }
            TextureViewKind::Srv => match desc.dimension {
                D1 => canonical_texture_type(D1, desc.sample_count, sub.array_layer_count),
                D2 => canonical_texture_type(D2, desc.sample_count, sub.array_layer_count),
                D2Array => {
                    canonical_texture_type(D2Array, desc.sample_count, sub.array_layer_count)
                }
                Cube => {
                    if !matches!(sub.array_layer_count, 1 | 6) {
                        return Err(RegistryError::InvalidSubresource(
                            "cube SRV views need 1 face or all 6 faces",
                        ));
                    }
                    canonical_texture_type(Cube, desc.sample_count, sub.array_layer_count)
                }
                CubeArray => {
                    if !sub.array_layer_count.is_multiple_of(6) {
                        return Err(RegistryError::InvalidSubresource(
                            "cube array SRV views need a multiple of 6 layers",
                        ));
                    }
                    canonical_texture_type(CubeArray, desc.sample_count, sub.array_layer_count)
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
                    D2 => canonical_texture_type(D2, desc.sample_count, sub.array_layer_count),
                    D2Array => {
                        canonical_texture_type(D2Array, desc.sample_count, sub.array_layer_count)
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

        if format != desc.format {
            let allowed = desc.format.is_typeless() || desc.format.srgb_pair() == Some(format);
            if !allowed || !view_classes_compatible(desc.format, format) {
                return Err(RegistryError::Unsupported(
                    "view format is not in the texture's typeless family",
                ));
            }
        }

        match kind {
            TextureViewKind::Rtv => {
                if !desc.usage.render_target {
                    return Err(RegistryError::Unsupported(
                        "texture was not registered with render_target usage",
                    ));
                }
                if !format.is_color_render_target() {
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
                if !format.is_depth_stencil() {
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
                if format.is_depth_stencil() || format.is_block_compressed() {
                    return Err(RegistryError::Unsupported(
                        "depth and block-compressed formats cannot be UAVs on this path",
                    ));
                }
            }
        }

        Ok(Self {
            pixel_format: format.mtl_pixel_format_on(support),
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
    support: FormatSupport,
    textures: HashMap<u64, TextureEntry>,
}

impl TextureRegistry {
    /// Retains the transport's device and queue and starts an empty registry.
    pub fn new(transport: &DirectMetalTransport) -> Result<Self, RegistryError> {
        Ok(Self {
            device: retain(transport.device_raw())?,
            queue: retain(transport.queue_raw())?,
            support: transport.format_support(),
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
        let descriptor = new_texture_descriptor(&desc, self.support)?;
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

        verify_texture_attributes(&texture, &desc, self.support)?;
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
        let format = self
            .get(id)
            .ok_or(RegistryError::UnknownTexture(id))?
            .desc
            .format;
        self.create_view_as(id, kind, subresource, format)
    }

    /// Creates a typed view of `format` over `subresource` of texture `id`.
    ///
    /// A typeless source texture may be viewed with any format of its
    /// element-size family (a combined depth/stencil source may expose its
    /// depth or stencil plane); a typed source only accepts its own format or
    /// its sRGB pair. The view is created with
    /// `newTextureViewWithPixelFormat:` before any caller can bind it.
    pub fn create_view_as(
        &self,
        id: u64,
        kind: TextureViewKind,
        subresource: TextureSubresource,
        format: TextureFormat,
    ) -> Result<TextureView, RegistryError> {
        let entry = self.get(id).ok_or(RegistryError::UnknownTexture(id))?;
        let plan = ViewPlan::resolve_as(&entry.desc, kind, subresource, format, self.support)?;
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
            format,
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
    /// Metal vertex layout appended after the state encoding.
    ///
    /// Most state blobs do not carry one: a graphics pipeline whose vertex
    /// function reads no `[[attribute(n)]]` input needs no descriptor. A
    /// version 3 host translation appends a chunk derived from the vertex
    /// stage's DXIL `ISG1` signature, and the descriptor is set from it before
    /// the pipeline state is created.
    pub vertex_layout: Option<VertexLayout>,
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
        if let Some(layout) = self.vertex_layout.as_ref()
            && !layout.is_empty()
        {
            layout.append_to(&mut out);
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
        // A version 3 host translation appends the vertex stage's layout after
        // the state. A blob without one keeps its historical meaning exactly.
        let vertex_layout = if reader.is_empty() {
            None
        } else {
            let remaining = reader.remaining();
            if !remaining.starts_with(&crate::macos::vertex_layout::VERTEX_LAYOUT_MAGIC) {
                // Historical VPS1/VPS2 blobs reject every trailing byte; the
                // only trailing section this host now understands is a chunk
                // that starts with its own magic.
                return Err(RegistryError::InvalidState("trailing bytes"));
            }
            match VertexLayout::decode(remaining) {
                Ok(layout) => Some(layout),
                Err(VertexLayoutError::UnknownFormat(code)) => {
                    return Err(RegistryError::UnsupportedVertexFormat(code));
                }
                Err(error) => {
                    return Err(RegistryError::InvalidVertexLayout(error.to_string()));
                }
            }
        };
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
            vertex_layout,
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

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
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
    if let Some(layout) = state.vertex_layout.as_ref()
        && !layout.is_empty()
    {
        apply_vertex_descriptor(descriptor.as_raw(), layout)?;
    }

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

/// Applies a derived vertex layout to a live `MTLRenderPipelineDescriptor`.
///
/// The descriptor is the one Metal requires before it will accept a vertex
/// function with `[[attribute(n)]]` inputs. Every attribute's format, offset
/// and buffer index and every used buffer's stride are set; nothing is
/// defaulted or dropped. The array subscripts return objects owned by the
/// `MTLVertexDescriptor`, so they are never retained or released here.
fn apply_vertex_descriptor(
    descriptor: objc::Id,
    layout: &VertexLayout,
) -> Result<(), RegistryError> {
    let class = objc::class(c"MTLVertexDescriptor");
    if class.is_null() {
        return Err(RegistryError::Unsupported(
            "MTLVertexDescriptor is unavailable on this host",
        ));
    }
    let raw = unsafe { objc::msg_send_id_0(class, objc::sel(c"new")) };
    let vertex_descriptor = ObjcObject::from_retained(raw).ok_or_else(|| {
        RegistryError::Metal(MetalError::InternalError(
            "MTLVertexDescriptor creation returned null".to_string(),
        ))
    })?;
    let attributes = unsafe {
        objc::msg_send_id_0(vertex_descriptor.as_raw(), objc::sel(c"attributes"))
    };
    if attributes.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "MTLVertexDescriptor attributes is nil".to_string(),
        )));
    }

    let mut layouts: HashMap<u8, u32> = HashMap::new();
    for attribute in layout.attributes() {
        let target = unsafe {
            objc::msg_send_id_usize(
                attributes,
                objc::sel(c"objectAtIndexedSubscript:"),
                usize::from(attribute.index),
            )
        };
        if target.is_null() {
            return Err(RegistryError::InvalidVertexLayout(format!(
                "attribute {} has no Metal attribute descriptor",
                attribute.index
            )));
        }
        // SAFETY: `target` is a live attribute descriptor owned by the
        // `MTLVertexDescriptor`; every value is the documented NSUInteger form.
        unsafe {
            objc::msg_send_void_usize(
                target,
                objc::sel(c"setFormat:"),
                attribute.format.mtl_format(),
            );
            objc::msg_send_void_usize(target, objc::sel(c"setOffset:"), attribute.offset as usize);
            objc::msg_send_void_usize(
                target,
                objc::sel(c"setBufferIndex:"),
                usize::from(attribute.buffer_index),
            );
        }
        layouts
            .entry(attribute.buffer_index)
            .or_insert(attribute.stride);
    }

    let layout_array = unsafe {
        objc::msg_send_id_0(vertex_descriptor.as_raw(), objc::sel(c"layouts"))
    };
    if layout_array.is_null() {
        return Err(RegistryError::Metal(MetalError::InternalError(
            "MTLVertexDescriptor layouts is nil".to_string(),
        )));
    }
    for (buffer_index, stride) in layouts {
        let target = unsafe {
            objc::msg_send_id_usize(
                layout_array,
                objc::sel(c"objectAtIndexedSubscript:"),
                usize::from(buffer_index),
            )
        };
        if target.is_null() {
            return Err(RegistryError::InvalidVertexLayout(format!(
                "buffer {buffer_index} has no Metal vertex layout descriptor"
            )));
        }
        // SAFETY: `target` is a live buffer layout descriptor owned by the
        // `MTLVertexDescriptor`; `stride` is 4-byte aligned and nonzero.
        unsafe {
            objc::msg_send_void_usize(target, objc::sel(c"setStride:"), stride as usize);
            objc::msg_send_void_usize(
                target,
                objc::sel(c"setStepFunction:"),
                MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            );
            objc::msg_send_void_usize(target, objc::sel(c"setStepRate:"), 1);
        }
    }

    // SAFETY: `descriptor` is the live pipeline descriptor under construction
    // and `vertex_descriptor` is a complete MTLVertexDescriptor.
    unsafe {
        objc::msg_send_void_1(
            descriptor,
            objc::sel(c"setVertexDescriptor:"),
            vertex_descriptor.as_raw(),
        );
    }
    Ok(())
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
    if format.is_depth_stencil() {
        return Ok((format.mtl_pixel_format(), format.has_stencil()));
    }
    if format.is_color_render_target() {
        return Ok((format.mtl_pixel_format(), false));
    }
    Err(RegistryError::Unsupported(
        "attachment format is not supported by the render pipeline path",
    ))
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

fn new_texture_descriptor(
    desc: &TextureDesc,
    support: FormatSupport,
) -> Result<ObjcObject, RegistryError> {
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
            desc.format.mtl_pixel_format_on(support),
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
    support: FormatSupport,
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
    check(c"pixelFormat", desc.format.mtl_pixel_format_on(support))?;
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
    use crate::macos::vertex_layout::{
        VERTEX_LAYOUT_HEADER_LEN, VERTEX_LAYOUT_MAGIC, VertexAttribute, VertexFormat,
    };

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
            (TextureFormat::R32Uint, 53),
            (TextureFormat::R32Sint, 54),
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

    /// ADR-042: every canonical `MTLTextureType` value is what Metal.framework
    /// declares, and the one table distinguishes all nine cases. The old
    /// table was shifted from `Cube` on (`Cube` 6, `CubeArray` 7, `3D` 8), so
    /// a 3D resource became a multisample array.
    #[test]
    fn mtl_texture_type_matches_the_canonical_metal_enum() {
        use TextureDimension::{Cube, CubeArray, D1, D2, D2Array, D3};

        assert_eq!(MTL_TEXTURE_TYPE_1D, 0);
        assert_eq!(MTL_TEXTURE_TYPE_1D_ARRAY, 1);
        assert_eq!(MTL_TEXTURE_TYPE_2D, 2);
        assert_eq!(MTL_TEXTURE_TYPE_2D_ARRAY, 3);
        assert_eq!(MTL_TEXTURE_TYPE_2D_MULTISAMPLE, 4);
        assert_eq!(MTL_TEXTURE_TYPE_CUBE, 5);
        assert_eq!(MTL_TEXTURE_TYPE_CUBE_ARRAY, 6);
        assert_eq!(MTL_TEXTURE_TYPE_3D, 7);
        assert_eq!(MTL_TEXTURE_TYPE_2D_MULTISAMPLE_ARRAY, 8);

        // D1, D1Array, D2, D2Array, D2 multisample, 2D multisample array,
        // Cube, CubeArray and D3, in that order.
        assert_eq!(canonical_texture_type(D1, 1, 1), 0);
        assert_eq!(canonical_texture_type(D1, 1, 2), 1);
        assert_eq!(canonical_texture_type(D2, 1, 1), 2);
        assert_eq!(canonical_texture_type(D2Array, 1, 4), 3);
        assert_eq!(canonical_texture_type(D2, 4, 1), 4);
        assert_eq!(canonical_texture_type(D2Array, 4, 4), 8);
        assert_eq!(canonical_texture_type(Cube, 1, 6), 5);
        assert_eq!(canonical_texture_type(CubeArray, 1, 12), 6);
        assert_eq!(canonical_texture_type(D3, 1, 1), 7);

        // A single layer of an arrayable dimension is the plain 2D type, and a
        // six-layer cube array is one cube, never the shifted 3D value.
        assert_eq!(canonical_texture_type(D2Array, 1, 1), 2);
        assert_eq!(canonical_texture_type(CubeArray, 1, 6), 5);
        assert_eq!(canonical_texture_type(D2Array, 4, 1), 4);
    }

    /// Boot #11/#12's 33x33x33 `R16G16B16A16_FLOAT` unordered-access volume
    /// (ADR-042): one mip, one array layer, depth 33, and `MTLTextureType3D`
    /// (7). The old table sent 8 (`MTLTextureType2DMultisampleArray`), which
    /// the host refused/asserted on. No GPU is touched here.
    #[test]
    fn a_33x33x33_fp16_uav_volume_is_a_3d_texture() {
        let mut volume = desc();
        volume.width = 33;
        volume.height = 33;
        volume.depth = 33;
        volume.mip_levels = 1;
        volume.array_layers = 1;
        volume.dimension = TextureDimension::D3;
        volume.format = TextureFormat::Rgba16Float;
        volume.usage = TextureUsage {
            shader_read: true,
            shader_write: true,
            render_target: false,
            depth_stencil: false,
        };

        assert!(volume.validate().is_ok());
        assert_eq!(volume.mtl_texture_type(), MTL_TEXTURE_TYPE_3D);
        assert_eq!(volume.mtl_texture_type(), 7);
        assert_eq!(volume.depth, 33);
        assert_eq!(volume.array_layers, 1);
        let uav = ViewPlan::resolve(
            &volume,
            TextureViewKind::Uav,
            TextureSubresource::single(0, 0),
        )
        .expect("a 33x33x33 FP16 UAV volume must resolve");
        assert_eq!(uav.texture_type, 7);
        assert_eq!(
            uav.levels,
            NSRange {
                location: 0,
                length: 1
            }
        );
        assert_eq!(
            uav.slices,
            NSRange {
                location: 0,
                length: 1
            }
        );
    }

    /// The Boot #2b compute-storage target (1280x720 R32_UINT) is both a
    /// colour attachment (RTV, Metal 53) and a shader-written storage target
    /// (UAV); R32_SINT is its signed sibling (Metal 54). No GPU is touched.
    #[test]
    fn a_1280x720_r32_uint_storage_colour_target_is_accepted() {
        let mut target = desc();
        target.width = 1280;
        target.height = 720;
        target.format = TextureFormat::R32Uint;
        target.usage = TextureUsage::color_attachment();
        assert!(target.validate().is_ok());
        assert!(matches!(
            attachment_pixel_format(target.format),
            Ok((53, false))
        ));

        let mut storage = target.clone();
        storage.usage = TextureUsage {
            shader_read: true,
            shader_write: true,
            render_target: true,
            depth_stencil: false,
        };
        assert!(storage.validate().is_ok());
        assert!(
            ViewPlan::resolve(
                &storage,
                TextureViewKind::Uav,
                TextureSubresource::single(0, 0),
            )
            .is_ok(),
            "an R32_UINT storage target must bind as a UAV"
        );
        assert!(
            ViewPlan::resolve(
                &storage,
                TextureViewKind::Rtv,
                TextureSubresource::single(0, 0),
            )
            .is_ok(),
            "an R32_UINT storage target is also a colour attachment"
        );

        target.format = TextureFormat::R32Sint;
        assert!(target.validate().is_ok());
        assert!(matches!(
            attachment_pixel_format(target.format),
            Ok((54, false))
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
            ..PipelineState::default()
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
    fn pipeline_state_carries_an_appended_vertex_layout() {
        let float4 = VertexFormat::with_components(VertexFormat::FLOAT, 4).unwrap();
        let layout = VertexLayout::from_attributes(vec![
            VertexAttribute {
                index: 0,
                format: float4,
                offset: 0,
                buffer_index: 0,
                stride: 32,
            },
            VertexAttribute {
                index: 1,
                format: float4,
                offset: 16,
                buffer_index: 0,
                stride: 32,
            },
        ])
        .expect("two float4 attributes");
        let state = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            fragment_entry: Some("fs_main".to_string()),
            vertex_layout: Some(layout.clone()),
            ..PipelineState::default()
        };
        let encoded = state.encode(PipelineStage::Graphics);
        let decoded = PipelineState::decode_for(&encoded, PipelineStage::Graphics).unwrap();
        assert_eq!(decoded.vertex_layout, Some(layout));
        // The chunk is at the very end and keeps the VPS1 header.
        assert_eq!(&encoded[..4], b"VPS1");
        let magic = encoded
            .windows(4)
            .position(|window| window == VERTEX_LAYOUT_MAGIC)
            .expect("the encoded state carries a vertex layout chunk");
        assert_eq!(&encoded[magic..magic + 4], &VERTEX_LAYOUT_MAGIC);
        assert_eq!(encoded[magic + 5], 2, "two attributes are encoded");

        // A state without a layout still decodes with no descriptor.
        let plain = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            fragment_entry: Some("fs_main".to_string()),
            ..PipelineState::default()
        }
        .encode(PipelineStage::Graphics);
        assert_eq!(PipelineState::decode(&plain).unwrap().vertex_layout, None);

        // A chunk format outside the supported set is a named RegistryError.
        let mut corrupt = encoded.clone();
        corrupt[magic + VERTEX_LAYOUT_HEADER_LEN + 1] = 0x31;
        assert!(matches!(
            PipelineState::decode(&corrupt),
            Err(RegistryError::UnsupportedVertexFormat(0x31))
        ));
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
    /// ADR-033: one canonical table covers the standard D3D12 / Xbox Series
    /// set, the BC formats and every `*_TYPELESS` family. Every row has a
    /// distinct PSO code and kind 8 wire code; the pre-ADR codes 0-19 keep
    /// their meaning.
    #[test]
    fn canonical_table_covers_the_standard_set_and_round_trips_codes() {
        let required = [
            (2u32, "R32G32B32A32_FLOAT"),
            (10, "R16G16B16A16_FLOAT"),
            (11, "R16G16B16A16_UNORM"),
            (12, "R16G16B16A16_UINT"),
            (13, "R16G16B16A16_SNORM"),
            (14, "R16G16B16A16_SINT"),
            (16, "R32G32_FLOAT"),
            (17, "R32G32_UINT"),
            (18, "R32G32_SINT"),
            (20, "D32_FLOAT_S8X24_UINT"),
            (24, "R10G10B10A2_UNORM"),
            (25, "R10G10B10A2_UINT"),
            (26, "R11G11B10_FLOAT"),
            (28, "R8G8B8A8_UNORM"),
            (29, "R8G8B8A8_UNORM_SRGB"),
            (30, "R8G8B8A8_UINT"),
            (31, "R8G8B8A8_SNORM"),
            (32, "R8G8B8A8_SINT"),
            (34, "R16G16_FLOAT"),
            (35, "R16G16_UNORM"),
            (36, "R16G16_UINT"),
            (37, "R16G16_SNORM"),
            (38, "R16G16_SINT"),
            (40, "D32_FLOAT"),
            (41, "R32_FLOAT"),
            (42, "R32_UINT"),
            (43, "R32_SINT"),
            (45, "D24_UNORM_S8_UINT"),
            (49, "R8G8_UNORM"),
            (50, "R8G8_UINT"),
            (51, "R8G8_SNORM"),
            (52, "R8G8_SINT"),
            (54, "R16_FLOAT"),
            (55, "D16_UNORM"),
            (56, "R16_UNORM"),
            (57, "R16_UINT"),
            (58, "R16_SNORM"),
            (59, "R16_SINT"),
            (61, "R8_UNORM"),
            (62, "R8_UINT"),
            (63, "R8_SNORM"),
            (64, "R8_SINT"),
            (65, "A8_UNORM"),
            (67, "R9G9B9E5_SHAREDEXP"),
            (71, "BC1_UNORM"),
            (72, "BC1_UNORM_SRGB"),
            (74, "BC2_UNORM"),
            (75, "BC2_UNORM_SRGB"),
            (77, "BC3_UNORM"),
            (78, "BC3_UNORM_SRGB"),
            (80, "BC4_UNORM"),
            (81, "BC4_SNORM"),
            (83, "BC5_UNORM"),
            (84, "BC5_SNORM"),
            (85, "B5G6R5_UNORM"),
            (86, "B5G5R5A1_UNORM"),
            (87, "B8G8R8A8_UNORM"),
            (88, "B8G8R8X8_UNORM"),
            (91, "B8G8R8A8_UNORM_SRGB"),
            (93, "B8G8R8X8_UNORM_SRGB"),
            (95, "BC6H_UF16"),
            (96, "BC6H_SF16"),
            (98, "BC7_UNORM"),
            (99, "BC7_UNORM_SRGB"),
        ];
        for (dxgi, name) in required {
            let format = TextureFormat::from_dxgi(dxgi)
                .unwrap_or_else(|error| panic!("DXGI {dxgi}: {error}"));
            assert!(
                format.name().contains(name),
                "DXGI {dxgi} maps to {} ({})",
                format.name(),
                format.mtl_pixel_format()
            );
        }

        // Every typeless family resolves and names its concrete base.
        for dxgi in [1u32, 9, 15, 19, 23, 27, 33, 39, 44, 48, 53, 60, 70, 73, 76, 79, 82, 90, 92, 94, 97] {
            let format = TextureFormat::from_dxgi(dxgi)
                .unwrap_or_else(|error| panic!("typeless DXGI {dxgi}: {error}"));
            assert!(format.is_typeless(), "DXGI {dxgi} must be typeless");
            assert!(
                format.typeless_base().is_some(),
                "DXGI {dxgi} must name a concrete base"
            );
        }

        // Codes 0-19 keep their pre-ADR meaning.
        for (code, expected) in [
            (0u8, TextureFormat::Bc1Rgba),
            (1, TextureFormat::Bc2Rgba),
            (2, TextureFormat::Bc3Rgba),
            (3, TextureFormat::Bc4RUnorm),
            (4, TextureFormat::Bc5RgUnorm),
            (5, TextureFormat::Bc6hRgbUf16),
            (6, TextureFormat::Bc7Rgba),
            (7, TextureFormat::Rgba8Unorm),
            (8, TextureFormat::Rgba8Srgb),
            (9, TextureFormat::Bgra8Unorm),
            (10, TextureFormat::Rg11g11b10Float),
            (11, TextureFormat::Rgb10a2Unorm),
            (12, TextureFormat::R32Float),
            (13, TextureFormat::R16Float),
            (14, TextureFormat::Depth32Float),
            (15, TextureFormat::Depth24Stencil8),
            (16, TextureFormat::Depth32FloatStencil8),
            (17, TextureFormat::Rgba16Float),
            (18, TextureFormat::R32Uint),
            (19, TextureFormat::R32Sint),
        ] {
            assert_eq!(TextureFormat::from_code(code), Some(expected), "code {code}");
            assert_eq!(expected.code(), code, "{} code", expected.name());
        }
        let mut codes = std::collections::HashSet::new();
        let mut wires = std::collections::HashSet::new();
        for format in TextureFormat::ALL {
            assert!(codes.insert(format.code()), "duplicate code for {format:?}");
            assert_eq!(TextureFormat::from_code(format.code()), Some(format));
            assert!(wires.insert(format.wire_code()), "duplicate wire code");
        }

        // Unsupported DXGI values are named refusals, never substitutes.
        for (dxgi, name) in [
            (5u32, "R32G32B32_TYPELESS"),
            (6, "R32G32B32_FLOAT"),
            (7, "R32G32B32_UINT"),
            (8, "R32G32B32_SINT"),
            (68, "R8G8_B8G8_UNORM"),
            (69, "G8R8_G8B8_UNORM"),
            (89, "R10G10B10_XR_BIAS_A2_UNORM"),
        ] {
            let error = TextureFormat::from_dxgi(dxgi).expect_err("must be refused");
            assert!(error.name().contains(name), "DXGI {dxgi}: {error}");
        }
    }

    /// The D24S8 substitution keeps Metal from aborting on Apple Silicon while
    /// devices that report 24-bit support keep the 24-bit descriptor.
    #[test]
    fn d24s8_substitutes_on_devices_without_24_bit_depth() {
        let legacy = FormatSupport {
            bc_texture_compression: true,
            depth24_stencil8: false,
        };
        let native = FormatSupport {
            bc_texture_compression: true,
            depth24_stencil8: true,
        };
        assert_eq!(
            TextureFormat::Depth24Stencil8.mtl_pixel_format_on(legacy),
            260
        );
        assert_eq!(
            TextureFormat::Depth24Stencil8.mtl_pixel_format_on(native),
            255
        );
        assert_eq!(
            TextureFormat::Depth24Stencil8.stencil_plane_format(legacy),
            Some(261)
        );
        assert_eq!(
            TextureFormat::Depth24Stencil8.stencil_plane_format(native),
            Some(262)
        );
        // The typeless D24G8 base is always created with the safe combined
        // format; its depth and stencil views use the plane formats.
        assert_eq!(TextureFormat::R24G8Typeless.mtl_pixel_format(), 260);
        assert_eq!(TextureFormat::R24UnormX8Typeless.mtl_pixel_format(), 252);
        assert_eq!(TextureFormat::X24TypelessG8Uint.mtl_pixel_format(), 261);
    }

    /// Typeless -> typed views are accepted inside one element-size family and
    /// refused across families; typed textures need the sRGB pair or their own
    /// format.
    #[test]
    fn typeless_view_reinterpretation_is_family_limited() {
        let typeless = |format| {
            TextureDesc {
                width: 64,
                height: 64,
                depth: 1,
                mip_levels: 1,
                sample_count: 1,
                array_layers: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsage {
                    shader_read: true,
                    shader_write: true,
                    render_target: true,
                    depth_stencil: false,
                },
                label: None,
            }
        };
        let rgba8 = typeless(TextureFormat::Rgba8Typeless);
        // Inside the 32-bit family: R32_UINT and RGBA8 both resolve.
        assert!(ViewPlan::resolve_as(
            &rgba8,
            TextureViewKind::Uav,
            TextureSubresource::single(0, 0),
            TextureFormat::R32Uint,
            FormatSupport {
                bc_texture_compression: true,
                depth24_stencil8: false,
            },
        )
        .is_ok());
        assert!(ViewPlan::resolve_as(
            &rgba8,
            TextureViewKind::Srv,
            TextureSubresource::single(0, 0),
            TextureFormat::Rgba8Unorm,
            FormatSupport {
                bc_texture_compression: true,
                depth24_stencil8: false,
            },
        )
        .is_ok());
        // Across the 64-bit family and to depth: refused.
        for refused in [
            TextureFormat::Rgba16Float,
            TextureFormat::Depth32Float,
        ] {
            assert!(
                ViewPlan::resolve_as(
                    &rgba8,
                    TextureViewKind::Srv,
                    TextureSubresource::single(0, 0),
                    refused,
                    FormatSupport {
                        bc_texture_compression: true,
                        depth24_stencil8: false,
                    },
                )
                .is_err(),
                "{} must not view a typeless RGBA8 resource",
                refused.name()
            );
        }
    }

    /// The GPU proof required by ADR-033: on the real device, create the
    /// Boot #2 target and every family the table grew, and create the typed
    /// views. Ignored by default; run under `.swarm/bin/gpu-lock` with
    /// `--ignored`.
    #[test]
    #[ignore = "GPU proof; run under .swarm/bin/gpu-lock with --ignored"]
    fn gpu_creates_the_full_format_set_and_typeless_views() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let support = transport.format_support();
        println!(
            "gpu proof: BC {}, 24-bit depth {}",
            support.bc_texture_compression,
            support.depth24_stencil8
        );
        let mut registry = TextureRegistry::new(&transport).expect("texture registry");

        let colour = |format, shader_write, render_target| TextureDesc {
            width: 640,
            height: 360,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            array_layers: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsage {
                shader_read: true,
                shader_write,
                render_target,
                depth_stencil: false,
            },
            label: None,
        };

        // Boot #2: a 640x360 R8Uint UAV + RT texture, exactly the resource
        // the live title failed to register.
        registry
            .register(1, colour(TextureFormat::R8Uint, true, true))
            .expect("640x360 R8Uint UAV+RT creation");
        registry
            .create_view(1, TextureViewKind::Uav, TextureSubresource::single(0, 0))
            .expect("R8Uint UAV view");
        registry
            .create_view(1, TextureViewKind::Rtv, TextureSubresource::single(0, 0))
            .expect("R8Uint RTV view");

        // Typeless RGBA8 viewed as R32Uint and as RGBA8.
        registry
            .register(2, colour(TextureFormat::Rgba8Typeless, true, true))
            .expect("typeless RGBA8 creation");
        registry
            .create_view_as(
                2,
                TextureViewKind::Uav,
                TextureSubresource::single(0, 0),
                TextureFormat::R32Uint,
            )
            .expect("typeless RGBA8 -> R32Uint UAV");
        registry
            .create_view_as(
                2,
                TextureViewKind::Srv,
                TextureSubresource::single(0, 0),
                TextureFormat::Rgba8Unorm,
            )
            .expect("typeless RGBA8 -> RGBA8 SRV");
        assert!(
            registry
                .create_view_as(
                    2,
                    TextureViewKind::Srv,
                    TextureSubresource::single(0, 0),
                    TextureFormat::Rgba16Float,
                )
                .is_err(),
            "cross-family typeless view must be refused"
        );

        // BC1/BC3/BC7 (the device reports BC support on Apple Silicon).
        if support.bc_texture_compression {
            for (id, format) in [
                (3u64, TextureFormat::Bc1Rgba),
                (4, TextureFormat::Bc3Rgba),
                (5, TextureFormat::Bc7Rgba),
            ] {
                let mut desc = colour(format, false, false);
                desc.width = 64;
                desc.height = 64;
                registry.register(id, desc).unwrap_or_else(|error| {
                    panic!("{} creation: {error}", format.name())
                });
                registry
                    .create_view(id, TextureViewKind::Srv, TextureSubresource::single(0, 0))
                    .unwrap_or_else(|error| panic!("{} SRV: {error}", format.name()));
            }
        }

        // R11G11B10F colour target.
        registry
            .register(6, colour(TextureFormat::Rg11g11b10Float, false, true))
            .expect("R11G11B10F target creation");
        registry
            .create_view(6, TextureViewKind::Rtv, TextureSubresource::single(0, 0))
            .expect("R11G11B10F RTV view");

        // D32F and D24S8 targets; D24S8 uses the substitution on Apple.
        let depth = |format| TextureDesc {
            width: 640,
            height: 360,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            array_layers: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsage::depth_attachment(),
            label: None,
        };
        registry
            .register(7, depth(TextureFormat::Depth32Float))
            .expect("D32F target creation");
        registry
            .create_view(7, TextureViewKind::Dsv, TextureSubresource::single(0, 0))
            .expect("D32F DSV view");
        registry
            .register(8, depth(TextureFormat::Depth24Stencil8))
            .expect("D24S8 target creation (substituted on Apple Silicon)");
        registry
            .create_view(8, TextureViewKind::Dsv, TextureSubresource::single(0, 0))
            .expect("D24S8 DSV view");
        assert_eq!(
            registry.get(8).unwrap().desc().format.mtl_pixel_format_on(support),
            if support.depth24_stencil8 { 255 } else { 260 },
            "D24S8 effective Metal format"
        );

        // ADR-042 / Boot #11/#12: the 33x33x33 FP16 unordered-access volume
        // must create as a real `MTLTextureType3D` (7) and take a UAV view.
        // Before the canonical enum fix this descriptor was built as a
        // multisample array and the host asserted.
        let mut volume = colour(TextureFormat::Rgba16Float, true, false);
        volume.width = 33;
        volume.height = 33;
        volume.depth = 33;
        volume.dimension = TextureDimension::D3;
        assert_eq!(volume.mtl_texture_type(), MTL_TEXTURE_TYPE_3D);
        registry
            .register(9, volume)
            .expect("33x33x33 FP16 UAV 3D creation");
        let created = registry.get(9).expect("33x33x33 volume registered");
        assert_eq!(created.desc().dimension, TextureDimension::D3);
        assert_eq!(created.desc().depth, 33);
        assert_eq!(created.desc().array_layers, 1);
        assert_eq!(created.desc().mtl_texture_type(), 7);
        registry
            .create_view(9, TextureViewKind::Uav, TextureSubresource::single(0, 0))
            .expect("33x33x33 FP16 UAV 3D view");
        println!("gpu proof: 33x33x33 FP16 UAV volume created as MTLTextureType3D and viewed");
        println!("gpu proof: every creation and view succeeded ({} textures)", registry.len());
    }

}
