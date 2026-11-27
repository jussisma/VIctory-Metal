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
}
