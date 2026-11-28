//! Guest-side D3D12 render-state tracking for the native Metal backend.
//!
//! Owner module reserved by the orchestrator for Cycle 06 (ADR-026): blend,
//! depth-stencil, rasterizer, stencil reference and blend factor state recorded
//! by the `victory-gpu` D3D12 API path and forwarded to the host over Wire v3.
//!
//! The producer must never fall back to D3D12 or D3DMetal after native
//! submission starts: an unsupported state returns an explicit error.
//!
//! # Scope
//!
//! This module only converts and tracks. It does not touch `list.rs` and it
//! emits no submission; a later task wires the entry points to
//! [`GuestRenderState::state_commands`]. Everything here is cross-platform so
//! the conversions compile and test on the macOS host as well as on Windows.
//!
//! D3D12 enumeration values are repeated as `u32` constants with their header
//! values, which keeps the module free of the Windows-only `windows` crate. The
//! checks are exhaustive: a code outside the D3D12 enum is a
//! [`GuestStateError::UnknownEnum`], never a silent default.

use std::fmt;

use victory_metal::wire_v3::{
    BlendAttachment, BlendFactor, BlendOperation, BlendState, CommandV3, CompareFunction,
    CullMode, DepthStencilState, FillMode, FrontFaceWinding, RasterizerState, StencilFaceState,
    StencilOperation, StencilRef,
};

// D3D12_BLEND values (`D3D12_BLEND_ZERO` .. `D3D12_BLEND_INV_SRC1_ALPHA`).
/// `D3D12_BLEND_ZERO`.
pub const D3D12_BLEND_ZERO: u32 = 1;
/// `D3D12_BLEND_ONE`.
pub const D3D12_BLEND_ONE: u32 = 2;
/// `D3D12_BLEND_SRC_COLOR`.
pub const D3D12_BLEND_SRC_COLOR: u32 = 3;
/// `D3D12_BLEND_INV_SRC_COLOR`.
pub const D3D12_BLEND_INV_SRC_COLOR: u32 = 4;
/// `D3D12_BLEND_SRC_ALPHA`.
pub const D3D12_BLEND_SRC_ALPHA: u32 = 5;
/// `D3D12_BLEND_INV_SRC_ALPHA`.
pub const D3D12_BLEND_INV_SRC_ALPHA: u32 = 6;
/// `D3D12_BLEND_DEST_ALPHA`.
pub const D3D12_BLEND_DEST_ALPHA: u32 = 7;
/// `D3D12_BLEND_INV_DEST_ALPHA`.
pub const D3D12_BLEND_INV_DEST_ALPHA: u32 = 8;
/// `D3D12_BLEND_DEST_COLOR`.
pub const D3D12_BLEND_DEST_COLOR: u32 = 9;
/// `D3D12_BLEND_INV_DEST_COLOR`.
pub const D3D12_BLEND_INV_DEST_COLOR: u32 = 10;
/// `D3D12_BLEND_SRC_ALPHA_SAT`.
pub const D3D12_BLEND_SRC_ALPHA_SAT: u32 = 11;
/// `D3D12_BLEND_BLEND_FACTOR`.
pub const D3D12_BLEND_BLEND_FACTOR: u32 = 14;
/// `D3D12_BLEND_INV_BLEND_FACTOR`.
pub const D3D12_BLEND_INV_BLEND_FACTOR: u32 = 15;
/// `D3D12_BLEND_SRC1_COLOR`.
pub const D3D12_BLEND_SRC1_COLOR: u32 = 16;
/// `D3D12_BLEND_INV_SRC1_COLOR`.
pub const D3D12_BLEND_INV_SRC1_COLOR: u32 = 17;
/// `D3D12_BLEND_SRC1_ALPHA`.
pub const D3D12_BLEND_SRC1_ALPHA: u32 = 18;
/// `D3D12_BLEND_INV_SRC1_ALPHA`.
pub const D3D12_BLEND_INV_SRC1_ALPHA: u32 = 19;

// D3D12_BLEND_OP values.
/// `D3D12_BLEND_OP_ADD`.
pub const D3D12_BLEND_OP_ADD: u32 = 1;
/// `D3D12_BLEND_OP_SUBTRACT`.
pub const D3D12_BLEND_OP_SUBTRACT: u32 = 2;
/// `D3D12_BLEND_OP_REV_SUBTRACT`.
pub const D3D12_BLEND_OP_REV_SUBTRACT: u32 = 3;
/// `D3D12_BLEND_OP_MIN`.
pub const D3D12_BLEND_OP_MIN: u32 = 4;
/// `D3D12_BLEND_OP_MAX`.
pub const D3D12_BLEND_OP_MAX: u32 = 5;

// D3D12_COMPARISON_FUNC values.
/// `D3D12_COMPARISON_FUNC_NEVER`.
pub const D3D12_COMPARISON_FUNC_NEVER: u32 = 1;
/// `D3D12_COMPARISON_FUNC_LESS`.
pub const D3D12_COMPARISON_FUNC_LESS: u32 = 2;
/// `D3D12_COMPARISON_FUNC_EQUAL`.
pub const D3D12_COMPARISON_FUNC_EQUAL: u32 = 3;
/// `D3D12_COMPARISON_FUNC_LESS_EQUAL`.
pub const D3D12_COMPARISON_FUNC_LESS_EQUAL: u32 = 4;
/// `D3D12_COMPARISON_FUNC_GREATER`.
pub const D3D12_COMPARISON_FUNC_GREATER: u32 = 5;
/// `D3D12_COMPARISON_FUNC_NOT_EQUAL`.
pub const D3D12_COMPARISON_FUNC_NOT_EQUAL: u32 = 6;
/// `D3D12_COMPARISON_FUNC_GREATER_EQUAL`.
pub const D3D12_COMPARISON_FUNC_GREATER_EQUAL: u32 = 7;
/// `D3D12_COMPARISON_FUNC_ALWAYS`.
pub const D3D12_COMPARISON_FUNC_ALWAYS: u32 = 8;

// D3D12_STENCIL_OP values.
/// `D3D12_STENCIL_OP_KEEP`.
pub const D3D12_STENCIL_OP_KEEP: u32 = 1;
/// `D3D12_STENCIL_OP_ZERO`.
pub const D3D12_STENCIL_OP_ZERO: u32 = 2;
/// `D3D12_STENCIL_OP_REPLACE`.
pub const D3D12_STENCIL_OP_REPLACE: u32 = 3;
/// `D3D12_STENCIL_OP_INCR_SAT`.
pub const D3D12_STENCIL_OP_INCR_SAT: u32 = 4;
/// `D3D12_STENCIL_OP_DECR_SAT`.
pub const D3D12_STENCIL_OP_DECR_SAT: u32 = 5;
/// `D3D12_STENCIL_OP_INVERT`.
pub const D3D12_STENCIL_OP_INVERT: u32 = 6;
/// `D3D12_STENCIL_OP_INCR`.
pub const D3D12_STENCIL_OP_INCR: u32 = 7;
/// `D3D12_STENCIL_OP_DECR`.
pub const D3D12_STENCIL_OP_DECR: u32 = 8;

// D3D12_FILL_MODE values.
/// `D3D12_FILL_MODE_WIREFRAME`.
pub const D3D12_FILL_MODE_WIREFRAME: u32 = 2;
/// `D3D12_FILL_MODE_SOLID`.
pub const D3D12_FILL_MODE_SOLID: u32 = 3;

// D3D12_CULL_MODE values.
/// `D3D12_CULL_MODE_NONE`.
pub const D3D12_CULL_MODE_NONE: u32 = 1;
/// `D3D12_CULL_MODE_FRONT`.
pub const D3D12_CULL_MODE_FRONT: u32 = 2;
/// `D3D12_CULL_MODE_BACK`.
pub const D3D12_CULL_MODE_BACK: u32 = 3;

/// `D3D12_COLOR_WRITE_ENABLE_ALL`.
pub const D3D12_COLOR_WRITE_ENABLE_ALL: u8 = 0x0f;

/// Reasons a recorded D3D12 state cannot be represented on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestStateError {
    /// A D3D12 enumeration code is not defined by the header.
    UnknownEnum {
        /// Name of the enumeration field.
        field: &'static str,
        /// The undefined code.
        code: u32,
    },
    /// A stencil reference or mask value does not fit the wire field.
    OutOfRange {
        /// Name of the field.
        field: &'static str,
        /// The value that did not fit.
        value: u32,
    },
    /// A blend desc carried more than the wire's eight render targets.
    TooManyRenderTargets(usize),
}

impl fmt::Display for GuestStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownEnum { field, code } => write!(f, "undefined {field} code {code}"),
            Self::OutOfRange { field, value } => write!(f, "{field} value {value} out of range"),
            Self::TooManyRenderTargets(count) => {
                write!(f, "{count} render-target blend entries exceeds 8")
            }
        }
    }
}

impl std::error::Error for GuestStateError {}

/// One `D3D12_RENDER_TARGET_BLEND_DESC` with header-valued fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetBlend {
    /// `BlendEnable`.
    pub blend_enable: bool,
    /// `SrcBlend`.
    pub src_blend: u32,
    /// `DestBlend`.
    pub dest_blend: u32,
    /// `BlendOp`.
    pub blend_op: u32,
    /// `SrcBlendAlpha`.
    pub src_blend_alpha: u32,
    /// `DestBlendAlpha`.
    pub dest_blend_alpha: u32,
    /// `BlendOpAlpha`.
    pub blend_op_alpha: u32,
    /// `RenderTargetWriteMask`.
    pub write_mask: u8,
}

impl Default for RenderTargetBlend {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_blend: D3D12_BLEND_ONE,
            dest_blend: D3D12_BLEND_ZERO,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ONE,
            dest_blend_alpha: D3D12_BLEND_ZERO,
            blend_op_alpha: D3D12_BLEND_OP_ADD,
            write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
        }
    }
}

/// `D3D12_BLEND_DESC`: alpha-to-coverage, independent blending and eight
/// per-target entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendDesc {
    /// `AlphaToCoverageEnable`.
    pub alpha_to_coverage: bool,
    /// `IndependentBlendEnable`.
    pub independent_blend: bool,
    /// `RenderTarget[8]`.
    pub render_target: [RenderTargetBlend; 8],
}

impl Default for BlendDesc {
    fn default() -> Self {
        Self {
            alpha_to_coverage: false,
            independent_blend: false,
            render_target: [RenderTargetBlend::default(); 8],
        }
    }
}

/// One `D3D12_DEPTH_STENCILOP_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilOpDesc {
    /// `StencilFailOp`.
    pub fail_op: u32,
    /// `StencilDepthFailOp`.
    pub depth_fail_op: u32,
    /// `StencilPassOp`.
    pub pass_op: u32,
    /// `StencilFunc`.
    pub func: u32,
}

impl Default for StencilOpDesc {
    fn default() -> Self {
        Self {
            fail_op: D3D12_STENCIL_OP_KEEP,
            depth_fail_op: D3D12_STENCIL_OP_KEEP,
            pass_op: D3D12_STENCIL_OP_KEEP,
            func: D3D12_COMPARISON_FUNC_ALWAYS,
        }
    }
}

/// `D3D12_DEPTH_STENCIL_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencilDesc {
    /// `DepthEnable`.
    pub depth_enable: bool,
    /// `DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL`.
    pub depth_write: bool,
    /// `DepthFunc`.
    pub depth_func: u32,
    /// `StencilEnable`.
    pub stencil_enable: bool,
    /// `StencilReadMask`.
    pub stencil_read_mask: u8,
    /// `StencilWriteMask`.
    pub stencil_write_mask: u8,
    /// `FrontFace`.
    pub front_face: StencilOpDesc,
    /// `BackFace`.
    pub back_face: StencilOpDesc,
}

impl Default for DepthStencilDesc {
    fn default() -> Self {
        Self {
            depth_enable: false,
            depth_write: true,
            depth_func: D3D12_COMPARISON_FUNC_ALWAYS,
            stencil_enable: false,
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            front_face: StencilOpDesc::default(),
            back_face: StencilOpDesc::default(),
        }
    }
}

/// `D3D12_RASTERIZER_DESC`. `FrontCounterClockwise` is D3D12's winding boolean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterizerDesc {
    /// `FillMode`.
    pub fill_mode: u32,
    /// `CullMode`.
    pub cull_mode: u32,
    /// `FrontCounterClockwise`.
    pub front_counter_clockwise: bool,
    /// `DepthBias`.
    pub depth_bias: i32,
    /// `DepthBiasClamp`.
    pub depth_bias_clamp: f32,
    /// `SlopeScaledDepthBias`.
    pub slope_scaled_depth_bias: f32,
    /// `DepthClipEnable`.
    pub depth_clip_enable: bool,
}

impl Default for RasterizerDesc {
    fn default() -> Self {
        Self {
            fill_mode: D3D12_FILL_MODE_SOLID,
            cull_mode: D3D12_CULL_MODE_BACK,
            front_counter_clockwise: false,
            depth_bias: 0,
            depth_bias_clamp: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_clip_enable: true,
        }
    }
}

/// Converts one D3D12 blend factor code.
pub fn blend_factor(code: u32) -> Result<BlendFactor, GuestStateError> {
    BlendFactor::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "blend factor",
        code,
    })
}

/// Converts one D3D12 blend operation code.
pub fn blend_operation(code: u32) -> Result<BlendOperation, GuestStateError> {
    BlendOperation::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "blend operation",
        code,
    })
}

/// Converts one D3D12 comparison function code.
pub fn compare_function(code: u32) -> Result<CompareFunction, GuestStateError> {
    CompareFunction::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "comparison function",
        code,
    })
}

/// Converts one D3D12 stencil operation code.
pub fn stencil_operation(code: u32) -> Result<StencilOperation, GuestStateError> {
    StencilOperation::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "stencil operation",
        code,
    })
}

/// Converts one D3D12 fill mode code.
pub fn fill_mode(code: u32) -> Result<FillMode, GuestStateError> {
    FillMode::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "fill mode",
        code,
    })
}

/// Converts one D3D12 cull mode code.
pub fn cull_mode(code: u32) -> Result<CullMode, GuestStateError> {
    CullMode::from_code(code as u16).map_err(|_| GuestStateError::UnknownEnum {
        field: "cull mode",
        code,
    })
}

/// Converts `FrontCounterClockwise` into the wire winding.
pub const fn winding(front_counter_clockwise: bool) -> FrontFaceWinding {
    if front_counter_clockwise {
        FrontFaceWinding::CounterClockwise
    } else {
        FrontFaceWinding::Clockwise
    }
}

/// Converts one `D3D12_RENDER_TARGET_BLEND_DESC`.
pub fn render_target_blend(desc: &RenderTargetBlend) -> Result<BlendAttachment, GuestStateError> {
    Ok(BlendAttachment {
        enable: desc.blend_enable,
        src_color: blend_factor(desc.src_blend)?,
        dst_color: blend_factor(desc.dest_blend)?,
        color_op: blend_operation(desc.blend_op)?,
        src_alpha: blend_factor(desc.src_blend_alpha)?,
        dst_alpha: blend_factor(desc.dest_blend_alpha)?,
        alpha_op: blend_operation(desc.blend_op_alpha)?,
        write_mask: desc.write_mask,
    })
}

/// Converts a full `D3D12_BLEND_DESC`.
///
/// D3D12 always carries eight entries; the wire keeps only the entries the
/// caller needs. With `IndependentBlendEnable` the wire must carry every entry
/// the pass uses, so all eight are emitted; otherwise D3D12 semantics say entry
/// 0 applies everywhere and only entry 0 is emitted.
pub fn blend_state(desc: &BlendDesc) -> Result<BlendState, GuestStateError> {
    if desc.independent_blend {
        let mut attachments = Vec::with_capacity(desc.render_target.len());
        for entry in &desc.render_target {
            attachments.push(render_target_blend(entry)?);
        }
        Ok(BlendState {
            independent_blend: true,
            alpha_to_coverage: desc.alpha_to_coverage,
            attachments,
        })
    } else {
        Ok(BlendState {
            independent_blend: false,
            alpha_to_coverage: desc.alpha_to_coverage,
            attachments: vec![render_target_blend(&desc.render_target[0])?],
        })
    }
}

/// Converts one `D3D12_DEPTH_STENCILOP_DESC`.
pub fn stencil_face(desc: &StencilOpDesc) -> Result<StencilFaceState, GuestStateError> {
    Ok(StencilFaceState {
        fail: stencil_operation(desc.fail_op)?,
        depth_fail: stencil_operation(desc.depth_fail_op)?,
        pass: stencil_operation(desc.pass_op)?,
        func: compare_function(desc.func)?,
        read_mask: 0xff,
        write_mask: 0xff,
    })
}

/// Converts a full `D3D12_DEPTH_STENCIL_DESC`.
pub fn depth_stencil_state(desc: &DepthStencilDesc) -> Result<DepthStencilState, GuestStateError> {
    let front = stencil_face(&desc.front_face)?;
    let back = stencil_face(&desc.back_face)?;
    Ok(DepthStencilState {
        depth_enable: desc.depth_enable,
        depth_write: desc.depth_write,
        depth_func: compare_function(desc.depth_func)?,
        stencil_enable: desc.stencil_enable,
        front: StencilFaceState {
            read_mask: desc.stencil_read_mask,
            write_mask: desc.stencil_write_mask,
            ..front
        },
        back: StencilFaceState {
            read_mask: desc.stencil_read_mask,
            write_mask: desc.stencil_write_mask,
            ..back
        },
    })
}

/// Converts a full `D3D12_RASTERIZER_DESC`.
pub fn rasterizer_state(desc: &RasterizerDesc) -> Result<RasterizerState, GuestStateError> {
    Ok(RasterizerState {
        fill: fill_mode(desc.fill_mode)?,
        cull: cull_mode(desc.cull_mode)?,
        front_face: winding(desc.front_counter_clockwise),
        depth_bias: desc.depth_bias,
        slope_scaled_depth_bias: desc.slope_scaled_depth_bias,
        depth_bias_clamp: desc.depth_bias_clamp,
        depth_clip: desc.depth_clip_enable,
    })
}

/// Render state recorded by the guest and lowered on demand.
///
/// Each setter stores the D3D12 value and returns the converted wire value, so
/// a caller can either keep raw state or forward it immediately. Nothing is
/// emitted until [`GuestRenderState::state_commands`] is asked for the ordered
/// command list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GuestRenderState {
    blend: Option<BlendState>,
    depth_stencil: Option<DepthStencilState>,
    rasterizer: Option<RasterizerState>,
    stencil_ref: Option<u8>,
    blend_factor: Option<[f32; 4]>,
}

impl GuestRenderState {
    /// Creates empty state (no command recorded yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `OMSetBlendState`.
    pub fn set_blend(&mut self, desc: &BlendDesc) -> Result<&BlendState, GuestStateError> {
        let wire = blend_state(desc)?;
        self.blend = Some(wire);
        Ok(self.blend.as_ref().expect("blend was just stored"))
    }

    /// Records `OMSetDepthStencilState`.
    pub fn set_depth_stencil(
        &mut self,
        desc: &DepthStencilDesc,
    ) -> Result<&DepthStencilState, GuestStateError> {
        let wire = depth_stencil_state(desc)?;
        self.depth_stencil = Some(wire);
        Ok(self
            .depth_stencil
            .as_ref()
            .expect("depth-stencil was just stored"))
    }

    /// Records `RSSetState`.
    pub fn set_rasterizer(
        &mut self,
        desc: &RasterizerDesc,
    ) -> Result<&RasterizerState, GuestStateError> {
        let wire = rasterizer_state(desc)?;
        self.rasterizer = Some(wire);
        Ok(self.rasterizer.as_ref().expect("rasterizer was just stored"))
    }

    /// Records `OMSetStencilRef`. D3D12 takes a `UINT`; the wire carries a byte
    /// because the stencil masks are eight bits wide.
    pub fn set_stencil_ref(&mut self, value: u32) -> Result<u8, GuestStateError> {
        let value = u8::try_from(value).map_err(|_| GuestStateError::OutOfRange {
            field: "stencil reference",
            value,
        })?;
        self.stencil_ref = Some(value);
        Ok(value)
    }

    /// Records `OMSetBlendFactor`.
    pub fn set_blend_factor(&mut self, value: [f32; 4]) {
        self.blend_factor = Some(value);
    }

    /// The currently recorded blend state, if any.
    pub fn blend(&self) -> Option<&BlendState> {
        self.blend.as_ref()
    }

    /// The currently recorded depth/stencil state, if any.
    pub fn depth_stencil(&self) -> Option<&DepthStencilState> {
        self.depth_stencil.as_ref()
    }

    /// The currently recorded rasterizer state, if any.
    pub fn rasterizer(&self) -> Option<&RasterizerState> {
        self.rasterizer.as_ref()
    }

    /// The currently recorded stencil reference, if any.
    pub fn stencil_ref(&self) -> Option<u8> {
        self.stencil_ref
    }

    /// The currently recorded blend factor, if any.
    pub fn blend_factor(&self) -> Option<[f32; 4]> {
        self.blend_factor
    }

    /// The wire commands for the recorded state, in a stable order.
    ///
    /// A later producer task appends these to the command list it is lowering;
    /// every recorded value appears exactly once so a redraw after a state
    /// change cannot miss one.
    pub fn state_commands(&self) -> Vec<CommandV3> {
        let mut commands = Vec::new();
        if let Some(blend) = &self.blend {
            commands.push(CommandV3::SetBlendState(blend.clone()));
        }
        if let Some(depth) = self.depth_stencil {
            commands.push(CommandV3::SetDepthStencilState(depth));
        }
        if let Some(rasterizer) = self.rasterizer {
            commands.push(CommandV3::SetRasterizerState(rasterizer));
        }
        if let Some(value) = self.stencil_ref {
            commands.push(CommandV3::SetStencilRef(StencilRef { value }));
        }
        if let Some(value) = self.blend_factor {
            commands.push(CommandV3::SetBlendFactor(
                victory_metal::wire_v3::BlendFactorValue { value },
            ));
        }
        commands
    }
}

// ---------------------------------------------------------------------------
// Guest input layout (ADR-061)
// ---------------------------------------------------------------------------

/// Vertex-buffer slots a D3D12 list tracks.
pub const VERTEX_SLOTS: usize = 32;
/// First Metal attribute index the converter assigns to a layout element.
pub const FIRST_VERTEX_ATTRIBUTE: u8 = 11;
/// Largest Metal attribute index the descriptor holds.
pub const MAX_VERTEX_ATTRIBUTE_INDEX: u8 = 30;

/// Magic that starts the vertex-layout chunk appended to a pipeline state.
pub const VMVD_MAGIC: [u8; 4] = *b"VMVD";
/// Chunk version this guest writes.
pub const VMVD_VERSION: u8 = 2;
const VMVD_HEADER_LEN: usize = 6;
const VMVD_ATTRIBUTE_LEN: usize = 11;
const VMVD_SLOT_TRAILER_LEN: usize = 1;
const VMVD_SLOT_LEN: usize = 10;

/// Why a guest input layout could not be built or encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuestLayoutError {
    /// The input layout has no elements or more than the attribute limit.
    TooManyElements(usize),
    /// An element's Metal index is outside `0..=30`.
    AttributeIndexOutOfRange { element: usize, index: u8 },
    /// An element names a DXGI format the host's vertex formats do not cover.
    UnsupportedFormat { element: usize, format: u32 },
    /// An element's format code is outside the VMVD set.
    UnknownFormatCode(u8),
    /// Two elements claim the same Metal attribute index.
    DuplicateIndex(u8),
    /// An element names a vertex slot outside `0..32`.
    InvalidSlot(u8),
    /// A known stride is zero.
    ZeroStride(u8),
    /// A per-instance slot has a zero step rate.
    ZeroStepRate(u8),
    /// Elements of one slot disagree on per-vertex/per-instance or step rate.
    MixedSlotClass(u8),
}

impl fmt::Display for GuestLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyElements(count) => {
                write!(f, "the guest input layout declares {count} elements")
            }
            Self::AttributeIndexOutOfRange { element, index } => write!(
                f,
                "guest input element {element} needs Metal attribute index {index}, above {MAX_VERTEX_ATTRIBUTE_INDEX}"
            ),
            Self::UnsupportedFormat { element, format } => {
                write!(
                    f,
                    "guest input element {element} has unmappable DXGI format {format}"
                )
            }
            Self::UnknownFormatCode(code) => {
                write!(f, "guest input format code {code} is outside the VMVD set")
            }
            Self::DuplicateIndex(index) => {
                write!(
                    f,
                    "two guest input elements claim Metal attribute index {index}"
                )
            }
            Self::InvalidSlot(slot) => {
                write!(f, "guest input slot {slot} exceeds the vertex slot limit")
            }
            Self::ZeroStride(slot) => write!(f, "guest input slot {slot} has a zero stride"),
            Self::ZeroStepRate(slot) => {
                write!(
                    f,
                    "guest input slot {slot} is per-instance with a zero step rate"
                )
            }
            Self::MixedSlotClass(slot) => {
                write!(
                    f,
                    "guest input slot {slot} mixes per-vertex and per-instance elements"
                )
            }
        }
    }
}

impl std::error::Error for GuestLayoutError {}

/// One guest input element in D3D12 declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestInputElement {
    /// Metal attribute index (`11 + element order`).
    pub index: u8,
    /// VMVD format code.
    pub format: u8,
    /// Resolved `AlignedByteOffset`.
    pub offset: u32,
    /// D3D12 `InputSlot`.
    pub slot: u8,
    /// D3D12 `InputSlotClass == PER_INSTANCE_DATA`.
    pub per_instance: bool,
    /// D3D12 `InstanceDataStepRate`.
    pub step_rate: u32,
}

/// One guest vertex-buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestInputSlot {
    /// D3D12 `InputSlot`.
    pub slot: u8,
    /// The most recent `IASetVertexBuffers` stride, or `None` when unknown.
    pub stride: Option<u32>,
    /// D3D12 `InputSlotClass == PER_INSTANCE_DATA`.
    pub per_instance: bool,
    /// D3D12 `InstanceDataStepRate` (1 for per-vertex).
    pub step_rate: u32,
}

/// The guest's `D3D12_INPUT_LAYOUT_DESC` in VMVD v2 wire form.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestInputLayout {
    elements: Vec<GuestInputElement>,
    slots: Vec<GuestInputSlot>,
}

impl GuestInputLayout {
    /// Builds a layout from elements in declaration order and the most recent
    /// per-slot strides. Element order is preserved: the converter numbers
    /// attributes from [`FIRST_VERTEX_ATTRIBUTE`] in that order.
    pub fn from_elements(
        elements: Vec<GuestInputElement>,
        strides: &[Option<u32>; VERTEX_SLOTS],
    ) -> Result<Self, GuestLayoutError> {
        if elements.is_empty() || elements.len() > usize::from(MAX_VERTEX_ATTRIBUTE_INDEX) + 1 {
            return Err(GuestLayoutError::TooManyElements(elements.len()));
        }
        let mut seen = [false; MAX_VERTEX_ATTRIBUTE_INDEX as usize + 1];
        let mut slots: Vec<GuestInputSlot> = Vec::new();
        for (order, element) in elements.iter().enumerate() {
            if element.index > MAX_VERTEX_ATTRIBUTE_INDEX {
                return Err(GuestLayoutError::AttributeIndexOutOfRange {
                    element: order,
                    index: element.index,
                });
            }
            if seen[usize::from(element.index)] {
                return Err(GuestLayoutError::DuplicateIndex(element.index));
            }
            seen[usize::from(element.index)] = true;
            if element.format == 0 || element.format > 48 {
                return Err(GuestLayoutError::UnknownFormatCode(element.format));
            }
            let slot = usize::from(element.slot);
            if slot >= VERTEX_SLOTS {
                return Err(GuestLayoutError::InvalidSlot(element.slot));
            }
            match slots.iter_mut().find(|slot| slot.slot == element.slot) {
                Some(existing) => {
                    if existing.per_instance != element.per_instance
                        || (element.per_instance && existing.step_rate != element.step_rate)
                    {
                        return Err(GuestLayoutError::MixedSlotClass(element.slot));
                    }
                }
                None => {
                    let stride = strides[slot];
                    if stride == Some(0) {
                        return Err(GuestLayoutError::ZeroStride(element.slot));
                    }
                    if element.per_instance && element.step_rate == 0 {
                        return Err(GuestLayoutError::ZeroStepRate(element.slot));
                    }
                    slots.push(GuestInputSlot {
                        slot: element.slot,
                        stride,
                        per_instance: element.per_instance,
                        step_rate: if element.per_instance {
                            element.step_rate
                        } else {
                            1
                        },
                    });
                }
            }
        }
        Ok(Self { elements, slots })
    }

    /// Whether the guest declared no input elements.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// The declared elements, in D3D12 order.
    pub fn elements(&self) -> &[GuestInputElement] {
        &self.elements
    }

    /// The slots, in D3D12 `InputSlot` order.
    pub fn slots(&self) -> &[GuestInputSlot] {
        &self.slots
    }

    /// Encodes the VMVD v2 chunk. An unknown slot stride is encoded as zero
    /// and counted in [`ia_stride_unknown`]; the host then derives the packed
    /// stride for that slot only.
    pub fn encode(&self) -> Vec<u8> {
        if self.elements.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(
            VMVD_HEADER_LEN
                + VMVD_ATTRIBUTE_LEN * self.elements.len()
                + VMVD_SLOT_TRAILER_LEN
                + VMVD_SLOT_LEN * self.slots.len(),
        );
        out.extend_from_slice(&VMVD_MAGIC);
        out.push(VMVD_VERSION);
        out.push(u8::try_from(self.elements.len()).expect("element count is bounded"));
        for element in &self.elements {
            let stride = self
                .slots
                .iter()
                .find(|slot| slot.slot == element.slot)
                .and_then(|slot| slot.stride)
                .unwrap_or(0);
            out.push(element.index);
            out.push(element.format);
            out.extend_from_slice(&element.offset.to_le_bytes());
            out.push(element.slot);
            out.extend_from_slice(&stride.to_le_bytes());
        }
        let mut slots = self.slots.clone();
        slots.sort_by_key(|slot| slot.slot);
        out.push(u8::try_from(slots.len()).expect("slot count is bounded"));
        let mut unknown = 0u64;
        for slot in &slots {
            out.push(slot.slot);
            let stride = slot.stride.unwrap_or(0);
            if slot.stride.is_none() {
                unknown = unknown.saturating_add(1);
            }
            out.extend_from_slice(&stride.to_le_bytes());
            out.push(u8::from(slot.per_instance));
            let step_rate = if slot.per_instance {
                slot.step_rate.max(1)
            } else {
                1
            };
            out.extend_from_slice(&step_rate.to_le_bytes());
        }
        if unknown > 0 {
            IA_STRIDE_UNKNOWN.fetch_add(unknown, std::sync::atomic::Ordering::Relaxed);
        }
        out
    }

    /// Appends the chunk to a pipeline state blob.
    pub fn append_to(&self, state: &mut Vec<u8>) {
        state.extend_from_slice(&self.encode());
    }
}

/// Maps a DXGI format number to the VMVD format code (a mirror of the host's
/// `VertexFormat`). Vertex formats only: typeless, depth and packed formats
/// have no Metal vertex attribute mapping here and return `None`.
pub fn vertex_format_code(dxgi_format: u32) -> Option<u8> {
    let (base, components) = match dxgi_format {
        2 => (1, 4),   // R32G32B32A32_FLOAT
        6 => (1, 3),   // R32G32B32_FLOAT
        16 => (1, 2),  // R32G32_FLOAT
        41 => (1, 1),  // R32_FLOAT
        3 => (41, 4),  // R32G32B32A32_UINT
        7 => (41, 3),  // R32G32B32_UINT
        17 => (41, 2), // R32G32_UINT
        42 => (41, 1), // R32_UINT
        4 => (45, 4),  // R32G32B32A32_SINT
        8 => (45, 3),  // R32G32B32_SINT
        18 => (45, 2), // R32G32_SINT
        43 => (45, 1), // R32_SINT
        10 => (5, 4),  // R16G16B16A16_FLOAT
        34 => (5, 2),  // R16G16_FLOAT
        50 => (5, 1),  // R16_FLOAT
        11 => (33, 4), // R16G16B16A16_UNORM
        35 => (33, 2), // R16G16_UNORM
        52 => (33, 1), // R16_UNORM
        12 => (25, 4), // R16G16B16A16_UINT
        36 => (25, 2), // R16G16_UINT
        53 => (25, 1), // R16_UINT
        13 => (37, 4), // R16G16B16A16_SNORM
        37 => (37, 2), // R16G16_SNORM
        54 => (37, 1), // R16_SNORM
        14 => (29, 4), // R16G16B16A16_SINT
        38 => (29, 2), // R16G16_SINT
        55 => (29, 1), // R16_SINT
        28 => (17, 4), // R8G8B8A8_UNORM
        45 => (17, 2), // R8G8_UNORM
        57 => (17, 1), // R8_UNORM
        30 => (9, 4),  // R8G8B8A8_UINT
        46 => (9, 2),  // R8G8_UINT
        58 => (9, 1),  // R8_UINT
        31 => (21, 4), // R8G8B8A8_SNORM
        47 => (21, 2), // R8G8_SNORM
        59 => (21, 1), // R8_SNORM
        32 => (13, 4), // R8G8B8A8_SINT
        48 => (13, 2), // R8G8_SINT
        56 => (13, 1), // R8_SINT
        _ => return None,
    };
    Some(base + components - 1)
}

/// The most recent `IASetVertexBuffers` stride per slot.
///
/// A title may create a PSO before it binds a vertex buffer, so these are
/// advisory: a slot that has never been bound stays `None` and its layout
/// encode counts it in [`ia_stride_unknown`] rather than inventing a packed
/// stride silently.
static LAST_VERTEX_STRIDES: std::sync::Mutex<[Option<u32>; VERTEX_SLOTS]> =
    std::sync::Mutex::new([None; VERTEX_SLOTS]);

/// Slots whose stride was unknown when a pipeline layout was encoded.
static IA_STRIDE_UNKNOWN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records the strides of one `IASetVertexBuffers` call; `None` clears a slot
/// (a null view unbinds it). Out-of-range slots and zero strides are ignored,
/// exactly as the wire refuses them.
pub fn note_vertex_buffer_strides(start_slot: u32, strides: &[Option<u32>]) {
    let Ok(start) = usize::try_from(start_slot) else {
        return;
    };
    let mut table = LAST_VERTEX_STRIDES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for (offset, stride) in strides.iter().enumerate() {
        let Some(slot) = start.checked_add(offset) else {
            break;
        };
        let Some(entry) = table.get_mut(slot) else {
            break;
        };
        *entry = stride.filter(|stride| *stride != 0);
    }
}

/// The most recent stride per slot.
pub fn vertex_strides() -> [Option<u32>; VERTEX_SLOTS] {
    *LAST_VERTEX_STRIDES
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

/// How many slots an encoded layout had to send with an unknown stride.
pub fn ia_stride_unknown() -> u64 {
    IA_STRIDE_UNKNOWN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clears the stride table and the unknown-stride counter (tests).
pub fn reset_vertex_layout_telemetry() {
    *LAST_VERTEX_STRIDES
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = [None; VERTEX_SLOTS];
    IA_STRIDE_UNKNOWN.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn independent_targets() -> [RenderTargetBlend; 8] {
        let mut targets = [RenderTargetBlend::default(); 8];
        targets[0] = RenderTargetBlend {
            blend_enable: true,
            src_blend: D3D12_BLEND_SRC_ALPHA,
            dest_blend: D3D12_BLEND_INV_SRC_ALPHA,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ONE,
            dest_blend_alpha: D3D12_BLEND_ONE,
            blend_op_alpha: D3D12_BLEND_OP_MAX,
            write_mask: D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        targets[1] = RenderTargetBlend {
            blend_enable: true,
            src_blend: D3D12_BLEND_ONE,
            dest_blend: D3D12_BLEND_ZERO,
            blend_op: D3D12_BLEND_OP_ADD,
            src_blend_alpha: D3D12_BLEND_ZERO,
            dest_blend_alpha: D3D12_BLEND_ONE,
            blend_op_alpha: D3D12_BLEND_OP_SUBTRACT,
            write_mask: 0b0101,
        };
        targets
    }

    #[test]
    fn blend_factors_convert_all_defined_codes() {
        for (code, expected) in [
            (D3D12_BLEND_ZERO, BlendFactor::Zero),
            (D3D12_BLEND_ONE, BlendFactor::One),
            (D3D12_BLEND_SRC_COLOR, BlendFactor::SrcColor),
            (D3D12_BLEND_INV_SRC_ALPHA, BlendFactor::InvSrcAlpha),
            (D3D12_BLEND_DEST_COLOR, BlendFactor::DestColor),
            (D3D12_BLEND_SRC_ALPHA_SAT, BlendFactor::SrcAlphaSat),
            (D3D12_BLEND_BLEND_FACTOR, BlendFactor::BlendFactor),
            (D3D12_BLEND_INV_BLEND_FACTOR, BlendFactor::InvBlendFactor),
            (D3D12_BLEND_SRC1_COLOR, BlendFactor::Src1Color),
            (D3D12_BLEND_INV_SRC1_ALPHA, BlendFactor::InvSrc1Alpha),
        ] {
            assert_eq!(blend_factor(code), Ok(expected), "code {code}");
        }
        // 12 and 13 are unused by D3D12.
        assert_eq!(
            blend_factor(12),
            Err(GuestStateError::UnknownEnum {
                field: "blend factor",
                code: 12,
            })
        );
    }

    #[test]
    fn blend_operations_and_compare_functions_convert() {
        assert_eq!(blend_operation(1), Ok(BlendOperation::Add));
        assert_eq!(blend_operation(5), Ok(BlendOperation::Max));
        assert_eq!(
            blend_operation(6),
            Err(GuestStateError::UnknownEnum {
                field: "blend operation",
                code: 6,
            })
        );
        assert_eq!(compare_function(2), Ok(CompareFunction::Less));
        assert_eq!(compare_function(8), Ok(CompareFunction::Always));
        assert_eq!(
            compare_function(0),
            Err(GuestStateError::UnknownEnum {
                field: "comparison function",
                code: 0,
            })
        );
    }

    #[test]
    fn stencil_operations_convert() {
        assert_eq!(stencil_operation(1), Ok(StencilOperation::Keep));
        assert_eq!(stencil_operation(4), Ok(StencilOperation::IncrSat));
        assert_eq!(stencil_operation(7), Ok(StencilOperation::Incr));
        assert_eq!(
            stencil_operation(9),
            Err(GuestStateError::UnknownEnum {
                field: "stencil operation",
                code: 9,
            })
        );
    }

    #[test]
    fn cull_winding_and_fill_convert() {
        assert_eq!(fill_mode(2), Ok(FillMode::Wireframe));
        assert_eq!(fill_mode(3), Ok(FillMode::Solid));
        assert_eq!(
            fill_mode(1),
            Err(GuestStateError::UnknownEnum {
                field: "fill mode",
                code: 1,
            })
        );
        assert_eq!(cull_mode(1), Ok(CullMode::None));
        assert_eq!(cull_mode(2), Ok(CullMode::Front));
        assert_eq!(cull_mode(3), Ok(CullMode::Back));
        assert_eq!(winding(false), FrontFaceWinding::Clockwise);
        assert_eq!(winding(true), FrontFaceWinding::CounterClockwise);
    }

    #[test]
    fn independent_blend_keeps_every_target_and_shared_blend_keeps_entry_zero() {
        let targets = independent_targets();
        let independent = BlendDesc {
            alpha_to_coverage: true,
            independent_blend: true,
            render_target: targets,
        };
        let wire = blend_state(&independent).expect("independent blend converts");
        assert!(wire.independent_blend);
        assert!(wire.alpha_to_coverage);
        assert_eq!(wire.attachments.len(), 8);
        assert_eq!(wire.attachments[0].src_color, BlendFactor::SrcAlpha);
        assert_eq!(wire.attachments[1].write_mask, 0b0101);
        assert_eq!(wire.attachments[1].alpha_op, BlendOperation::Subtract);

        let shared = BlendDesc {
            alpha_to_coverage: false,
            independent_blend: false,
            render_target: targets,
        };
        let wire = blend_state(&shared).expect("shared blend converts");
        assert!(!wire.independent_blend);
        assert_eq!(wire.attachments.len(), 1);
        assert_eq!(wire.attachments[0].src_color, BlendFactor::SrcAlpha);
    }

    #[test]
    fn depth_stencil_masks_and_faces_convert() {
        let desc = DepthStencilDesc {
            depth_enable: true,
            depth_write: true,
            depth_func: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            stencil_enable: true,
            stencil_read_mask: 0x7f,
            stencil_write_mask: 0x0f,
            front_face: StencilOpDesc {
                fail_op: D3D12_STENCIL_OP_KEEP,
                depth_fail_op: D3D12_STENCIL_OP_INCR_SAT,
                pass_op: D3D12_STENCIL_OP_REPLACE,
                func: D3D12_COMPARISON_FUNC_EQUAL,
            },
            back_face: StencilOpDesc::default(),
        };
        let wire = depth_stencil_state(&desc).expect("depth state converts");
        assert!(wire.depth_enable && wire.depth_write);
        assert_eq!(wire.depth_func, CompareFunction::LessEqual);
        assert_eq!(wire.front.read_mask, 0x7f);
        assert_eq!(wire.front.write_mask, 0x0f);
        assert_eq!(wire.front.depth_fail, StencilOperation::IncrSat);
        assert_eq!(wire.front.pass, StencilOperation::Replace);
        assert_eq!(wire.front.func, CompareFunction::Equal);
        assert_eq!(wire.back.read_mask, 0x7f);
        assert_eq!(wire.back.fail, StencilOperation::Keep);
    }

    #[test]
    fn rasterizer_bias_and_depth_clip_convert() {
        let desc = RasterizerDesc {
            fill_mode: D3D12_FILL_MODE_WIREFRAME,
            cull_mode: D3D12_CULL_MODE_BACK,
            front_counter_clockwise: true,
            depth_bias: -8,
            depth_bias_clamp: 0.5,
            slope_scaled_depth_bias: 2.0,
            depth_clip_enable: false,
        };
        let wire = rasterizer_state(&desc).expect("rasterizer converts");
        assert_eq!(wire.fill, FillMode::Wireframe);
        assert_eq!(wire.cull, CullMode::Back);
        assert_eq!(wire.front_face, FrontFaceWinding::CounterClockwise);
        assert_eq!(wire.depth_bias, -8);
        assert_eq!(wire.depth_bias_clamp, 0.5);
        assert_eq!(wire.slope_scaled_depth_bias, 2.0);
        assert!(!wire.depth_clip);
    }

    #[test]
    fn tracking_replaces_state_and_lowers_stable_commands() {
        let mut state = GuestRenderState::new();
        assert!(state.state_commands().is_empty());

        state
            .set_blend(&BlendDesc::default())
            .expect("default blend converts");
        state
            .set_rasterizer(&RasterizerDesc::default())
            .expect("default rasterizer converts");
        let first = state.state_commands();
        assert_eq!(first.len(), 2);
        assert!(matches!(first[0], CommandV3::SetBlendState(_)));
        assert!(matches!(first[1], CommandV3::SetRasterizerState(_)));

        // A second set replaces the value instead of appending a command.
        state
            .set_rasterizer(&RasterizerDesc {
                cull_mode: D3D12_CULL_MODE_NONE,
                ..RasterizerDesc::default()
            })
            .expect("rasterizer converts");
        let second = state.state_commands();
        assert_eq!(second.len(), 2);
        let CommandV3::SetRasterizerState(raster) = &second[1] else {
            panic!("expected a rasterizer command");
        };
        assert_eq!(raster.cull, CullMode::None);
    }

    #[test]
    fn stencil_ref_refuses_values_wider_than_a_byte() {
        let mut state = GuestRenderState::new();
        assert_eq!(state.set_stencil_ref(255), Ok(255));
        assert_eq!(
            state.set_stencil_ref(256),
            Err(GuestStateError::OutOfRange {
                field: "stencil reference",
                value: 256,
            })
        );
        assert_eq!(state.stencil_ref(), Some(255));
    }

    #[test]
    fn blend_factor_is_tracked_and_lowered() {
        let mut state = GuestRenderState::new();
        state.set_blend_factor([0.25, 0.5, 0.75, 1.0]);
        let commands = state.state_commands();
        let CommandV3::SetBlendFactor(factor) = &commands[0] else {
            panic!("expected a blend-factor command");
        };
        assert_eq!(factor.value, [0.25, 0.5, 0.75, 1.0]);
    }

    fn element(index: u8, format: u8, offset: u32, slot: u8) -> GuestInputElement {
        GuestInputElement {
            index,
            format,
            offset,
            slot,
            per_instance: false,
            step_rate: 0,
        }
    }

    #[test]
    fn household_dxgi_vertex_formats_map_to_vmvd_codes() {
        assert_eq!(vertex_format_code(2), Some(4), "R32G32B32A32_FLOAT");
        assert_eq!(vertex_format_code(6), Some(3), "R32G32B32_FLOAT");
        assert_eq!(vertex_format_code(16), Some(2), "R32G32_FLOAT");
        assert_eq!(vertex_format_code(41), Some(1), "R32_FLOAT");
        assert_eq!(vertex_format_code(28), Some(20), "R8G8B8A8_UNORM");
        assert_eq!(vertex_format_code(45), Some(18), "R8G8_UNORM");
        assert_eq!(vertex_format_code(11), Some(36), "R16G16B16A16_UNORM");
        assert_eq!(vertex_format_code(10), Some(8), "R16G16B16A16_FLOAT");
        // Packed, depth and typeless formats have no vertex mapping.
        for unsupported in [1, 5, 24, 26, 27, 40, 87, 115] {
            assert_eq!(
                vertex_format_code(unsupported),
                None,
                "format {unsupported}"
            );
        }
    }

    #[test]
    fn a_two_slot_layout_encodes_element_order_slots_and_strides() {
        let mut strides = [None; VERTEX_SLOTS];
        strides[0] = Some(32);
        strides[1] = Some(32);
        let mut colour = element(12, 4, 8, 1);
        colour.per_instance = true;
        colour.step_rate = 2;
        let layout = GuestInputLayout::from_elements(vec![element(11, 2, 0, 0), colour], &strides)
            .expect("two-slot layout");
        let chunk = layout.encode();
        assert_eq!(&chunk[..4], &VMVD_MAGIC);
        assert_eq!(chunk[4], VMVD_VERSION);
        assert_eq!(chunk[5], 2, "two elements");
        // Element order preserved: 11 then 12.
        assert_eq!(chunk[6], 11);
        assert_eq!(chunk[6 + VMVD_ATTRIBUTE_LEN], 12);
        let slots_at = VMVD_HEADER_LEN + VMVD_ATTRIBUTE_LEN * 2;
        assert_eq!(chunk[slots_at], 2, "two slots");
        let first_slot = slots_at + VMVD_SLOT_TRAILER_LEN;
        assert_eq!(chunk[first_slot], 0);
        assert_eq!(
            u32::from_le_bytes(chunk[first_slot + 1..first_slot + 5].try_into().unwrap()),
            32
        );
        assert_eq!(chunk[first_slot + 5], 0, "slot 0 per-vertex");
        let second_slot = first_slot + VMVD_SLOT_LEN;
        assert_eq!(chunk[second_slot], 1);
        assert_eq!(
            u32::from_le_bytes(chunk[second_slot + 1..second_slot + 5].try_into().unwrap()),
            32
        );
        assert_eq!(chunk[second_slot + 5], 1, "slot 1 per-instance");
        assert_eq!(
            u32::from_le_bytes(chunk[second_slot + 6..second_slot + 10].try_into().unwrap()),
            2
        );
        assert_eq!(
            chunk.len(),
            VMVD_HEADER_LEN + VMVD_ATTRIBUTE_LEN * 2 + VMVD_SLOT_TRAILER_LEN + VMVD_SLOT_LEN * 2
        );
    }

    #[test]
    fn an_unknown_stride_is_encoded_as_zero_and_counted() {
        reset_vertex_layout_telemetry();
        let strides = [None; VERTEX_SLOTS];
        let layout = GuestInputLayout::from_elements(vec![element(11, 2, 0, 0)], &strides)
            .expect("one element");
        let chunk = layout.encode();
        // Attribute stride and slot stride are both zero (unknown).
        assert_eq!(
            u32::from_le_bytes(chunk[6 + 7..6 + 11].try_into().unwrap()),
            0
        );
        let slots_at = VMVD_HEADER_LEN + VMVD_ATTRIBUTE_LEN;
        assert_eq!(
            u32::from_le_bytes(chunk[slots_at + 2..slots_at + 6].try_into().unwrap()),
            0
        );
        assert_eq!(ia_stride_unknown(), 1);
        reset_vertex_layout_telemetry();
        assert_eq!(ia_stride_unknown(), 0);
    }

    #[test]
    fn recorded_strides_feed_the_next_layout() {
        reset_vertex_layout_telemetry();
        note_vertex_buffer_strides(0, &[Some(32), Some(48)]);
        let strides = vertex_strides();
        assert_eq!(strides[0], Some(32));
        assert_eq!(strides[1], Some(48));
        note_vertex_buffer_strides(1, &[Some(0), None]);
        let strides = vertex_strides();
        assert_eq!(strides[1], None, "a zero stride is not adopted");
        assert_eq!(strides[0], Some(32));
        reset_vertex_layout_telemetry();
        assert_eq!(vertex_strides()[0], None);
    }

    #[test]
    fn guest_layout_refuses_bad_slots_classes_and_formats() {
        let known = [Some(32); VERTEX_SLOTS];
        let mut per_instance = element(11, 2, 0, 0);
        per_instance.per_instance = true;
        assert_eq!(
            GuestInputLayout::from_elements(vec![per_instance], &known).unwrap_err(),
            GuestLayoutError::ZeroStepRate(0)
        );
        let mut mixed = element(12, 2, 8, 0);
        mixed.per_instance = true;
        mixed.step_rate = 1;
        assert_eq!(
            GuestInputLayout::from_elements(vec![element(11, 2, 0, 0), mixed], &known).unwrap_err(),
            GuestLayoutError::MixedSlotClass(0)
        );
        assert_eq!(
            GuestInputLayout::from_elements(vec![element(31, 2, 0, 0)], &known).unwrap_err(),
            GuestLayoutError::AttributeIndexOutOfRange {
                element: 0,
                index: 31
            }
        );
        assert_eq!(
            GuestInputLayout::from_elements(vec![element(11, 2, 0, 32)], &known).unwrap_err(),
            GuestLayoutError::InvalidSlot(32)
        );
        assert_eq!(
            GuestInputLayout::from_elements(vec![element(11, 0, 0, 0)], &known).unwrap_err(),
            GuestLayoutError::UnknownFormatCode(0)
        );
        let mut zero = [None; VERTEX_SLOTS];
        zero[0] = Some(0);
        assert_eq!(
            GuestInputLayout::from_elements(vec![element(11, 2, 0, 0)], &zero).unwrap_err(),
            GuestLayoutError::ZeroStride(0)
        );
    }
}
