//! Native Metal pipeline and encoder state for the D3D12 core render states.
//!
//! Owner module reserved by the orchestrator for Cycle 06 (ADR-026): blend
//! state, depth-stencil state (including `OMSetStencilRef`/`OMSetBlendFactor`),
//! rasterizer state (cull mode, winding, fill mode, depth bias), render-target
//! and depth clears, and pitched upload-to-private texture copies.
//!
//! Nothing here falls back to D3D12 or D3DMetal. An unrepresentable state is an
//! explicit error, never a silent default.
//!
//! # Split between pipeline state and encoder state
//!
//! Metal's `MTLRenderPipelineState` bakes per-attachment blending into the
//! pipeline, while cull/winding/fill/depth-bias, the stencil reference and the
//! blend-colour constant are render-encoder state. [`apply_blend_attachment`]
//! is therefore called while the pipeline descriptor is built, and
//! [`apply_rasterizer`], [`apply_stencil_ref`] and [`apply_blend_color`] are
//! called on a live `MTLRenderCommandEncoder`. Depth/stencil comparison state
//! is a separate cached `MTLDepthStencilState` bound per draw.
//!
//! Every wire enum already refused undefined D3D12 codes at decode time; the
//! mapping functions here are total and panic-free.

use std::collections::HashMap;

use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, ObjcObject};
use crate::transport::MetalError;
use crate::wire_v3::{
    BlendAttachment, BlendFactor, BlendOperation, CompareFunction, CullMode, DepthStencilState,
    FillMode, FrontFaceWinding, RasterizerState, StencilOperation,
};

/// `MTLBlendFactorZero`.
const MTL_BLEND_FACTOR_ZERO: usize = 0;
/// `MTLBlendFactorOne`.
const MTL_BLEND_FACTOR_ONE: usize = 1;
/// `MTLBlendFactorSourceColor`.
const MTL_BLEND_FACTOR_SOURCE_COLOR: usize = 2;
/// `MTLBlendFactorOneMinusSourceColor`.
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR: usize = 3;
/// `MTLBlendFactorSourceAlpha`.
const MTL_BLEND_FACTOR_SOURCE_ALPHA: usize = 4;
/// `MTLBlendFactorOneMinusSourceAlpha`.
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA: usize = 5;
/// `MTLBlendFactorDestinationColor`.
const MTL_BLEND_FACTOR_DESTINATION_COLOR: usize = 6;
/// `MTLBlendFactorOneMinusDestinationColor`.
const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR: usize = 7;
/// `MTLBlendFactorDestinationAlpha`.
const MTL_BLEND_FACTOR_DESTINATION_ALPHA: usize = 8;
/// `MTLBlendFactorOneMinusDestinationAlpha`.
const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA: usize = 9;
/// `MTLBlendFactorSourceAlphaSaturated`.
const MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED: usize = 10;
/// `MTLBlendFactorBlendColor`.
const MTL_BLEND_FACTOR_BLEND_COLOR: usize = 11;
/// `MTLBlendFactorOneMinusBlendColor`.
const MTL_BLEND_FACTOR_ONE_MINUS_BLEND_COLOR: usize = 12;
/// `MTLBlendFactorSource1Color`.
const MTL_BLEND_FACTOR_SOURCE1_COLOR: usize = 15;
/// `MTLBlendFactorOneMinusSource1Color`.
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_COLOR: usize = 16;
/// `MTLBlendFactorSource1Alpha`.
const MTL_BLEND_FACTOR_SOURCE1_ALPHA: usize = 17;
/// `MTLBlendFactorOneMinusSource1Alpha`.
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA: usize = 18;

/// `MTLBlendOperationAdd`.
const MTL_BLEND_OPERATION_ADD: usize = 0;
/// `MTLBlendOperationSubtract`.
const MTL_BLEND_OPERATION_SUBTRACT: usize = 1;
/// `MTLBlendOperationReverseSubtract`.
const MTL_BLEND_OPERATION_REVERSE_SUBTRACT: usize = 2;
/// `MTLBlendOperationMin`.
const MTL_BLEND_OPERATION_MIN: usize = 3;
/// `MTLBlendOperationMax`.
const MTL_BLEND_OPERATION_MAX: usize = 4;

/// `MTLCompareFunctionNever`.
const MTL_COMPARE_FUNCTION_NEVER: usize = 0;
/// `MTLCompareFunctionLess`.
const MTL_COMPARE_FUNCTION_LESS: usize = 1;
/// `MTLCompareFunctionEqual`.
const MTL_COMPARE_FUNCTION_EQUAL: usize = 2;
/// `MTLCompareFunctionLessEqual`.
const MTL_COMPARE_FUNCTION_LESS_EQUAL: usize = 3;
/// `MTLCompareFunctionGreater`.
const MTL_COMPARE_FUNCTION_GREATER: usize = 4;
/// `MTLCompareFunctionNotEqual`.
const MTL_COMPARE_FUNCTION_NOT_EQUAL: usize = 5;
/// `MTLCompareFunctionGreaterEqual`.
const MTL_COMPARE_FUNCTION_GREATER_EQUAL: usize = 6;
/// `MTLCompareFunctionAlways`.
const MTL_COMPARE_FUNCTION_ALWAYS: usize = 7;

/// `MTLStencilOperationKeep`.
const MTL_STENCIL_OPERATION_KEEP: usize = 0;
/// `MTLStencilOperationZero`.
const MTL_STENCIL_OPERATION_ZERO: usize = 1;
/// `MTLStencilOperationReplace`.
const MTL_STENCIL_OPERATION_REPLACE: usize = 2;
/// `MTLStencilOperationIncrementClamp`.
const MTL_STENCIL_OPERATION_INCREMENT_CLAMP: usize = 3;
/// `MTLStencilOperationDecrementClamp`.
const MTL_STENCIL_OPERATION_DECREMENT_CLAMP: usize = 4;
/// `MTLStencilOperationInvert`.
const MTL_STENCIL_OPERATION_INVERT: usize = 5;
/// `MTLStencilOperationIncrementWrap`.
const MTL_STENCIL_OPERATION_INCREMENT_WRAP: usize = 6;
/// `MTLStencilOperationDecrementWrap`.
const MTL_STENCIL_OPERATION_DECREMENT_WRAP: usize = 7;

/// `MTLTriangleFillModeFill`.
const MTL_TRIANGLE_FILL_MODE_FILL: usize = 0;
/// `MTLTriangleFillModeLines`.
const MTL_TRIANGLE_FILL_MODE_LINES: usize = 1;

/// `MTLCullModeNone`.
const MTL_CULL_MODE_NONE: usize = 0;
/// `MTLCullModeFront`.
const MTL_CULL_MODE_FRONT: usize = 1;
/// `MTLCullModeBack`.
const MTL_CULL_MODE_BACK: usize = 2;

/// `MTLWindingClockwise`.
const MTL_WINDING_CLOCKWISE: usize = 0;
/// `MTLWindingCounterClockwise`.
const MTL_WINDING_COUNTER_CLOCKWISE: usize = 1;

/// Maps a D3D12 blend factor onto `MTLBlendFactor`.
pub(crate) const fn mtl_blend_factor(factor: BlendFactor) -> usize {
    match factor {
        BlendFactor::Zero => MTL_BLEND_FACTOR_ZERO,
        BlendFactor::One => MTL_BLEND_FACTOR_ONE,
        BlendFactor::SrcColor => MTL_BLEND_FACTOR_SOURCE_COLOR,
        BlendFactor::InvSrcColor => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR,
        BlendFactor::SrcAlpha => MTL_BLEND_FACTOR_SOURCE_ALPHA,
        BlendFactor::InvSrcAlpha => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        BlendFactor::DestColor => MTL_BLEND_FACTOR_DESTINATION_COLOR,
        BlendFactor::InvDestColor => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR,
        BlendFactor::DestAlpha => MTL_BLEND_FACTOR_DESTINATION_ALPHA,
        BlendFactor::InvDestAlpha => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA,
        BlendFactor::SrcAlphaSat => MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
        BlendFactor::BlendFactor => MTL_BLEND_FACTOR_BLEND_COLOR,
        BlendFactor::InvBlendFactor => MTL_BLEND_FACTOR_ONE_MINUS_BLEND_COLOR,
        BlendFactor::Src1Color => MTL_BLEND_FACTOR_SOURCE1_COLOR,
        BlendFactor::InvSrc1Color => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_COLOR,
        BlendFactor::Src1Alpha => MTL_BLEND_FACTOR_SOURCE1_ALPHA,
        BlendFactor::InvSrc1Alpha => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA,
    }
}

/// Maps a D3D12 blend operation onto `MTLBlendOperation`.
pub(crate) const fn mtl_blend_operation(operation: BlendOperation) -> usize {
    match operation {
        BlendOperation::Add => MTL_BLEND_OPERATION_ADD,
        BlendOperation::Subtract => MTL_BLEND_OPERATION_SUBTRACT,
        BlendOperation::RevSubtract => MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
        BlendOperation::Min => MTL_BLEND_OPERATION_MIN,
        BlendOperation::Max => MTL_BLEND_OPERATION_MAX,
    }
}

/// Maps a D3D12 comparison function onto `MTLCompareFunction`.
pub(crate) const fn mtl_compare_function(function: CompareFunction) -> usize {
    match function {
        CompareFunction::Never => MTL_COMPARE_FUNCTION_NEVER,
        CompareFunction::Less => MTL_COMPARE_FUNCTION_LESS,
        CompareFunction::Equal => MTL_COMPARE_FUNCTION_EQUAL,
        CompareFunction::LessEqual => MTL_COMPARE_FUNCTION_LESS_EQUAL,
        CompareFunction::Greater => MTL_COMPARE_FUNCTION_GREATER,
        CompareFunction::NotEqual => MTL_COMPARE_FUNCTION_NOT_EQUAL,
        CompareFunction::GreaterEqual => MTL_COMPARE_FUNCTION_GREATER_EQUAL,
        CompareFunction::Always => MTL_COMPARE_FUNCTION_ALWAYS,
    }
}

/// Maps a D3D12 stencil operation onto `MTLStencilOperation`.
///
/// D3D12's saturating `INCR_SAT`/`DECR_SAT` clamp at the mask boundary and map
/// to Metal's `IncrementClamp`/`DecrementClamp`; the wrapping variants map to
/// `IncrementWrap`/`DecrementWrap`.
pub(crate) const fn mtl_stencil_operation(operation: StencilOperation) -> usize {
    match operation {
        StencilOperation::Keep => MTL_STENCIL_OPERATION_KEEP,
        StencilOperation::Zero => MTL_STENCIL_OPERATION_ZERO,
        StencilOperation::Replace => MTL_STENCIL_OPERATION_REPLACE,
        StencilOperation::IncrSat => MTL_STENCIL_OPERATION_INCREMENT_CLAMP,
        StencilOperation::DecrSat => MTL_STENCIL_OPERATION_DECREMENT_CLAMP,
        StencilOperation::Invert => MTL_STENCIL_OPERATION_INVERT,
        StencilOperation::Incr => MTL_STENCIL_OPERATION_INCREMENT_WRAP,
        StencilOperation::Decr => MTL_STENCIL_OPERATION_DECREMENT_WRAP,
    }
}

/// Maps a D3D12 fill mode onto `MTLTriangleFillMode`.
pub(crate) const fn mtl_fill_mode(mode: FillMode) -> usize {
    match mode {
        FillMode::Wireframe => MTL_TRIANGLE_FILL_MODE_LINES,
        FillMode::Solid => MTL_TRIANGLE_FILL_MODE_FILL,
    }
}

/// Maps a D3D12 cull mode onto `MTLCullMode`.
pub(crate) const fn mtl_cull_mode(mode: CullMode) -> usize {
    match mode {
        CullMode::None => MTL_CULL_MODE_NONE,
        CullMode::Front => MTL_CULL_MODE_FRONT,
        CullMode::Back => MTL_CULL_MODE_BACK,
    }
}

/// Maps the `FrontCounterClockwise` boolean onto `MTLWinding`.
pub(crate) const fn mtl_winding(winding: FrontFaceWinding) -> usize {
    match winding {
        FrontFaceWinding::Clockwise => MTL_WINDING_CLOCKWISE,
        FrontFaceWinding::CounterClockwise => MTL_WINDING_COUNTER_CLOCKWISE,
    }
}

/// Maps a D3D12 colour write mask onto `MTLColorWriteMask`.
///
/// The two encodings use the same four channels in the opposite bit order:
/// D3D12 is `R=1, G=2, B=4, A=8` while Metal is `R=8, G=4, B=2, A=1`.
pub(crate) const fn mtl_write_mask(write_mask: u8) -> usize {
    let mask = (write_mask & 0b1111) as usize;
    ((mask & 0b0001) << 3)
        | ((mask & 0b0010) << 1)
        | ((mask & 0b0100) >> 1)
        | ((mask & 0b1000) >> 3)
}

/// Applies one render target's blend state to an
/// `MTLRenderPipelineColorAttachmentDescriptor`.
///
/// # Safety
///
/// `attachment` must be a live colour attachment descriptor of a pipeline
/// descriptor under construction.
pub(crate) unsafe fn apply_blend_attachment(
    attachment: objc::Id,
    state: &BlendAttachment,
) -> Result<(), MetalError> {
    if attachment.is_null() {
        return Err(MetalError::InternalError(
            "colour attachment descriptor is nil".to_string(),
        ));
    }
    let mask = mtl_write_mask(state.write_mask);
    // SAFETY: the caller guarantees a live attachment descriptor; every value
    // is a documented `MTLBlendFactor`/`MTLBlendOperation`/mask encoding.
    unsafe {
        objc::msg_send_void_u8(attachment, objc::sel(c"setBlendingEnabled:"), u8::from(state.enable));
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setSourceRGBBlendFactor:"),
            mtl_blend_factor(state.src_color),
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setDestinationRGBBlendFactor:"),
            mtl_blend_factor(state.dst_color),
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setRgbBlendOperation:"),
            mtl_blend_operation(state.color_op),
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setSourceAlphaBlendFactor:"),
            mtl_blend_factor(state.src_alpha),
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setDestinationAlphaBlendFactor:"),
            mtl_blend_factor(state.dst_alpha),
        );
        objc::msg_send_void_usize(
            attachment,
            objc::sel(c"setAlphaBlendOperation:"),
            mtl_blend_operation(state.alpha_op),
        );
        objc::msg_send_void_usize(attachment, objc::sel(c"setWriteMask:"), mask);
    }
    Ok(())
}

/// Builds one `MTLDepthStencilState` from the wire depth/stencil state.
pub(crate) fn new_depth_stencil_state(
    transport: &DirectMetalTransport,
    state: &DepthStencilState,
) -> Result<ObjcObject, MetalError> {
    let class = objc::class(c"MTLDepthStencilDescriptor");
    if class.is_null() {
        return Err(MetalError::InternalError(
            "MTLDepthStencilDescriptor class unavailable".to_string(),
        ));
    }
    // SAFETY: `class` is the documented class and `new` returns a retained,
    // zero-initialized descriptor.
    let raw = unsafe { objc::msg_send_id_0(class, objc::sel(c"new")) };
    let descriptor = ObjcObject::from_retained(raw).ok_or_else(|| {
        MetalError::InternalError("MTLDepthStencilDescriptor creation returned null".to_string())
    })?;
    // SAFETY: the descriptor is live and every value is a documented
    // `MTLCompareFunction`/BOOL/stencil encoding.
    unsafe {
        objc::msg_send_void_usize(
            descriptor.as_raw(),
            objc::sel(c"setDepthCompareFunction:"),
            mtl_compare_function(state.depth_func),
        );
        objc::msg_send_void_u8(
            descriptor.as_raw(),
            objc::sel(c"setDepthWriteEnabled:"),
            u8::from(state.depth_write),
        );
        apply_stencil_face(
            descriptor.as_raw(),
            objc::sel(c"frontFaceStencil"),
            state.stencil_enable,
            &state.front,
        )?;
        apply_stencil_face(
            descriptor.as_raw(),
            objc::sel(c"backFaceStencil"),
            state.stencil_enable,
            &state.back,
        )?;
    }
    // SAFETY: `descriptor` is a live depth-stencil descriptor. The device
    // selector returns a retained state or nil.
    let state_raw = unsafe {
        objc::msg_send_id_1(
            transport.device_raw(),
            objc::sel(c"newDepthStencilStateWithDescriptor:"),
            descriptor.as_raw(),
        )
    };
    ObjcObject::from_retained(state_raw).ok_or_else(|| {
        MetalError::InternalError(
            "MTLDevice.newDepthStencilStateWithDescriptor returned nil".to_string(),
        )
    })
}

/// Fills one face descriptor obtained from `depth_descriptor`.
///
/// # Safety
///
/// `depth_descriptor` must be a live `MTLDepthStencilDescriptor` and `face`
/// the selector returning one of its `MTLStencilDescriptor`s.
unsafe fn apply_stencil_face(
    depth_descriptor: objc::Id,
    face: objc::Sel,
    enabled: bool,
    state: &crate::wire_v3::StencilFaceState,
) -> Result<(), MetalError> {
    // SAFETY: the caller guarantees a live depth-stencil descriptor.
    let stencil = unsafe { objc::msg_send_id_0(depth_descriptor, face) };
    if stencil.is_null() {
        return Err(MetalError::InternalError(
            "MTLDepthStencilDescriptor face stencil is nil".to_string(),
        ));
    }
    // A disabled stencil test never writes: keep the comparison always true
    // and both masks zero so an unconfigured stencil plane stays untouched.
    let (func, read_mask, write_mask) = if enabled {
        (
            mtl_compare_function(state.func),
            u32::from(state.read_mask),
            u32::from(state.write_mask),
        )
    } else {
        (MTL_COMPARE_FUNCTION_ALWAYS, 0, 0)
    };
    // SAFETY: `stencil` is a live `MTLStencilDescriptor`; all values are
    // documented encodings.
    unsafe {
        objc::msg_send_void_usize(stencil, objc::sel(c"setStencilCompareFunction:"), func);
        objc::msg_send_void_usize(
            stencil,
            objc::sel(c"setStencilFailureOperation:"),
            mtl_stencil_operation(state.fail),
        );
        objc::msg_send_void_usize(
            stencil,
            objc::sel(c"setDepthFailureOperation:"),
            mtl_stencil_operation(state.depth_fail),
        );
        objc::msg_send_void_usize(
            stencil,
            objc::sel(c"setDepthStencilPassOperation:"),
            mtl_stencil_operation(state.pass),
        );
        msg_send_void_u32(stencil, objc::sel(c"setReadMask:"), read_mask);
        msg_send_void_u32(stencil, objc::sel(c"setWriteMask:"), write_mask);
    }
    Ok(())
}

/// Applies cull, winding, fill and depth bias to a live render encoder.
///
/// # Safety
///
/// `enc` must be a live `MTLRenderCommandEncoder`.
pub(crate) unsafe fn apply_rasterizer(
    enc: objc::Id,
    state: &RasterizerState,
) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    // SAFETY: the caller guarantees a live render encoder.
    unsafe {
        objc::msg_send_void_usize(enc, objc::sel(c"setCullMode:"), mtl_cull_mode(state.cull));
        objc::msg_send_void_usize(
            enc,
            objc::sel(c"setFrontFacingWinding:"),
            mtl_winding(state.front_face),
        );
        objc::msg_send_void_usize(
            enc,
            objc::sel(c"setTriangleFillMode:"),
            mtl_fill_mode(state.fill),
        );
        msg_send_void_depth_bias(
            enc,
            objc::sel(c"setDepthBias:slopeScale:clamp:"),
            state.depth_bias as f32,
            state.slope_scaled_depth_bias,
            state.depth_bias_clamp,
        );
    }
    Ok(())
}

/// Applies the stencil reference value to a live render encoder.
///
/// # Safety
///
/// `enc` must be a live `MTLRenderCommandEncoder`.
pub(crate) unsafe fn apply_stencil_ref(enc: objc::Id, value: u8) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    // SAFETY: the caller guarantees a live render encoder.
    unsafe {
        msg_send_void_u32(
            enc,
            objc::sel(c"setStencilReferenceValue:"),
            u32::from(value),
        );
    }
    Ok(())
}

/// Applies the blend colour constant to a live render encoder.
///
/// # Safety
///
/// `enc` must be a live `MTLRenderCommandEncoder`.
pub(crate) unsafe fn apply_blend_color(enc: objc::Id, color: [f32; 4]) -> Result<(), MetalError> {
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    // SAFETY: the caller guarantees a live render encoder.
    unsafe {
        msg_send_void_color4_f32(
            enc,
            objc::sel(c"setBlendColorRed:green:blue:alpha:"),
            color,
        );
    }
    Ok(())
}

/// Cache of `MTLDepthStencilState` objects keyed by the full wire state.
///
/// The key is the whole [`DepthStencilState`] value, so two states that differ
/// in any compare function, operation, mask or enable flag never alias. The
/// cache lives in `DeviceState` and keeps every state alive for the life of
/// the device.
#[derive(Default)]
pub(crate) struct DepthStencilCache {
    states: HashMap<DepthStencilState, ObjcObject>,
}

impl DepthStencilCache {
    /// Creates an empty cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the cached state for `state`, creating it on first use.
    ///
    /// The returned handle is owned by the cache, which outlives every prepared
    /// segment of a submission while the device state guard is held.
    pub(crate) fn get_or_create(
        &mut self,
        transport: &DirectMetalTransport,
        state: &DepthStencilState,
    ) -> Result<objc::Id, MetalError> {
        if let Some(existing) = self.states.get(state) {
            return Ok(existing.as_raw());
        }
        let object = new_depth_stencil_state(transport, state)?;
        let raw = object.as_raw();
        self.states.insert(*state, object);
        Ok(raw)
    }

    /// Number of distinct states built so far.
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }
}

unsafe fn msg_send_void_u32(target: objc::Id, s: objc::Sel, value: u32) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, u32);
    let f: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: the caller supplies a receiver/selector taking one `uint32_t`.
    unsafe { f(target, s, value) }
}

unsafe fn msg_send_void_depth_bias(
    target: objc::Id,
    s: objc::Sel,
    bias: f32,
    slope_scale: f32,
    clamp: f32,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, f32, f32, f32);
    let f: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: the caller supplies a receiver/selector taking three floats.
    unsafe { f(target, s, bias, slope_scale, clamp) }
}

unsafe fn msg_send_void_color4_f32(target: objc::Id, s: objc::Sel, color: [f32; 4]) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, f32, f32, f32, f32);
    let f: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: the caller supplies a receiver/selector taking four floats.
    unsafe { f(target, s, color[0], color[1], color[2], color[3]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire_v3::{StencilFaceState, StencilRef};

    fn face() -> StencilFaceState {
        StencilFaceState {
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::IncrSat,
            pass: StencilOperation::Replace,
            func: CompareFunction::Equal,
            read_mask: 0xff,
            write_mask: 0x0f,
        }
    }

    fn depth_state() -> DepthStencilState {
        DepthStencilState {
            depth_enable: true,
            depth_write: true,
            depth_func: CompareFunction::Less,
            stencil_enable: true,
            front: face(),
            back: face(),
        }
    }

    #[test]
    fn blend_factors_map_to_metal_values() {
        assert_eq!(mtl_blend_factor(BlendFactor::Zero), 0);
        assert_eq!(mtl_blend_factor(BlendFactor::One), 1);
        assert_eq!(mtl_blend_factor(BlendFactor::SrcColor), 2);
        assert_eq!(mtl_blend_factor(BlendFactor::InvSrcAlpha), 5);
        assert_eq!(mtl_blend_factor(BlendFactor::DestColor), 6);
        assert_eq!(mtl_blend_factor(BlendFactor::SrcAlphaSat), 10);
        assert_eq!(mtl_blend_factor(BlendFactor::BlendFactor), 11);
        assert_eq!(mtl_blend_factor(BlendFactor::Src1Color), 15);
        assert_eq!(mtl_blend_factor(BlendFactor::InvSrc1Alpha), 18);
    }

    #[test]
    fn blend_operations_map_to_metal_values() {
        assert_eq!(mtl_blend_operation(BlendOperation::Add), 0);
        assert_eq!(mtl_blend_operation(BlendOperation::Subtract), 1);
        assert_eq!(mtl_blend_operation(BlendOperation::RevSubtract), 2);
        assert_eq!(mtl_blend_operation(BlendOperation::Min), 3);
        assert_eq!(mtl_blend_operation(BlendOperation::Max), 4);
    }

    #[test]
    fn compare_functions_map_to_metal_values() {
        assert_eq!(mtl_compare_function(CompareFunction::Never), 0);
        assert_eq!(mtl_compare_function(CompareFunction::Less), 1);
        assert_eq!(mtl_compare_function(CompareFunction::Equal), 2);
        assert_eq!(mtl_compare_function(CompareFunction::LessEqual), 3);
        assert_eq!(mtl_compare_function(CompareFunction::Greater), 4);
        assert_eq!(mtl_compare_function(CompareFunction::NotEqual), 5);
        assert_eq!(mtl_compare_function(CompareFunction::GreaterEqual), 6);
        assert_eq!(mtl_compare_function(CompareFunction::Always), 7);
    }

    #[test]
    fn stencil_operations_map_to_metal_values() {
        assert_eq!(mtl_stencil_operation(StencilOperation::Keep), 0);
        assert_eq!(mtl_stencil_operation(StencilOperation::Zero), 1);
        assert_eq!(mtl_stencil_operation(StencilOperation::Replace), 2);
        assert_eq!(mtl_stencil_operation(StencilOperation::IncrSat), 3);
        assert_eq!(mtl_stencil_operation(StencilOperation::DecrSat), 4);
        assert_eq!(mtl_stencil_operation(StencilOperation::Invert), 5);
        assert_eq!(mtl_stencil_operation(StencilOperation::Incr), 6);
        assert_eq!(mtl_stencil_operation(StencilOperation::Decr), 7);
    }

    #[test]
    fn write_mask_translates_d3d12_bits_to_metal_bits() {
        // D3D12 R=1,G=2,B=4,A=8 -> Metal R=8,G=4,B=2,A=1.
        assert_eq!(mtl_write_mask(0b0001), 0b1000);
        assert_eq!(mtl_write_mask(0b0010), 0b0100);
        assert_eq!(mtl_write_mask(0b0100), 0b0010);
        assert_eq!(mtl_write_mask(0b1000), 0b0001);
        // The GPU proof uses R+B (D3D12 0b0101) which must become Metal 0b1010.
        assert_eq!(mtl_write_mask(0b0101), 0b1010);
        assert_eq!(mtl_write_mask(0b1111), 0b1111);
        assert_eq!(mtl_write_mask(0b0000), 0);
        // Only the low four bits are meaningful.
        assert_eq!(mtl_write_mask(0xf5), 0b1010);
    }

    #[test]
    fn rasterizer_and_winding_map_to_metal_values() {
        assert_eq!(mtl_fill_mode(FillMode::Solid), 0);
        assert_eq!(mtl_fill_mode(FillMode::Wireframe), 1);
        assert_eq!(mtl_cull_mode(CullMode::None), 0);
        assert_eq!(mtl_cull_mode(CullMode::Front), 1);
        assert_eq!(mtl_cull_mode(CullMode::Back), 2);
        assert_eq!(mtl_winding(FrontFaceWinding::Clockwise), 0);
        assert_eq!(mtl_winding(FrontFaceWinding::CounterClockwise), 1);
    }

    #[test]
    fn depth_stencil_cache_key_covers_every_field() {
        // The key is the whole descriptor: each single-field change must produce
        // a distinct key, so a cached state can never alias a different one.
        let base = depth_state();
        let mut variants = Vec::new();
        let mut depth_func = base;
        depth_func.depth_func = CompareFunction::Greater;
        variants.push(depth_func);
        let mut depth_write = base;
        depth_write.depth_write = false;
        variants.push(depth_write);
        let mut stencil = base;
        stencil.stencil_enable = false;
        variants.push(stencil);
        let mut front = base;
        front.front.read_mask = 0x0f;
        variants.push(front);
        let mut back = base;
        back.back.write_mask = 0xf0;
        variants.push(back);

        let mut keys = std::collections::HashSet::new();
        assert!(keys.insert(base));
        for variant in variants {
            assert!(keys.insert(variant), "each variant must be a distinct key");
        }
        assert_eq!(keys.len(), 6);
    }

    #[test]
    fn wire_stencil_ref_is_a_byte() {
        assert_eq!(StencilRef { value: 255 }.value, 255);
    }
}
