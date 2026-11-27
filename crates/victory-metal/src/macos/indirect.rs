//! GPU-driven indirect command encoding (ADR-015).
//!
//! This module owns Metal encoding of indirect draws (`drawIndirect`,
//! `drawIndexedIndirect`) and indirect mesh/compute dispatch
//! (`drawMeshThreadgroupsWithIndirectBuffer`,
//! `dispatchThreadgroupsWithIndirectBuffer`) including the optional count
//! buffer. Argument buffers stay GPU-resident: no CPU readback.
//!
//! # Argument layout
//!
//! Metal's indirect argument structs are the ABI the GPU reads, so the wire
//! strides are Metal's, not D3D12's:
//!
//! ```text
//! MTLDrawPrimitivesIndirectArguments          16 bytes
//!   vertexCount:u32, instanceCount:u32, vertexStart:u32, baseInstance:u32
//! MTLDrawIndexedPrimitivesIndirectArguments   20 bytes
//!   indexCount, instanceCount, indexStart, baseVertex:i32, baseInstance
//! MTLDispatchThreadgroupsIndirectArguments    12 bytes
//!   threadgroupsPerGrid:u32[3]
//! ```
//!
//! The indexed record carries a fifth `baseInstance` word that D3D12's
//! `D3D12_DRAW_INDEXED_ARGUMENTS` does not have; a producer that writes only
//! the D3D12 prefix must leave the following word zero.
//!
//! # Count buffer
//!
//! D3D12 clamps the executed commands to `min(MaxCommandCount, count value)`.
//! Metal's legacy indirect draw/dispatch selectors have no count-buffer
//! operand (`drawCount:stride:` and friends are not in
//! `MTLRenderCommandEncoder.h`), and reading the count on the CPU would break
//! the no-readback rule. [`IndirectArguments::count`] is therefore resolved and
//! validated by the caller, then refused with [`MetalError::NotSupported`]
//! before any encoder call. Without a count buffer the executor issues one
//! GPU-read indirect call per argument record, which is exactly D3D12's
//! `MaxCommandCount` semantics.

use crate::macos::objc::{self, MTLSize};
use crate::macos::render::{IndexType, PrimitiveType};
use crate::transport::MetalError;
use crate::wire_v3::MAX_INDIRECT_COMMANDS_V3;

/// Bytes of one `MTLDrawPrimitivesIndirectArguments` record.
pub(crate) const DRAW_ARGUMENT_STRIDE: u64 = 16;
/// Bytes of one `MTLDrawIndexedPrimitivesIndirectArguments` record.
pub(crate) const DRAW_INDEXED_ARGUMENT_STRIDE: u64 = 20;
/// Bytes of one `MTLDispatchThreadgroupsIndirectArguments` record.
pub(crate) const DISPATCH_ARGUMENT_STRIDE: u64 = 12;

/// One GPU-resident indirect argument range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndirectArguments {
    /// Argument `MTLBuffer` resolved from the live registry.
    pub(crate) argument_buffer: objc::Id,
    /// Byte offset of the first record; 4-byte aligned.
    pub(crate) argument_offset: usize,
    /// Records to execute; nonzero and at most `MAX_INDIRECT_COMMANDS_V3`.
    pub(crate) max_count: u32,
    /// Optional GPU count buffer; unsupported by Metal's legacy selectors.
    pub(crate) count: Option<IndirectCount>,
}

/// One GPU-resident count-buffer reference.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndirectCount {
    /// Count `MTLBuffer` resolved from the live registry.
    pub(crate) buffer: objc::Id,
    /// Byte offset of the 32-bit count; 4-byte aligned.
    pub(crate) offset: usize,
}

/// Validates one indirect range before any encoder is touched.
///
/// The caller resolves the argument and count buffers from the live registry
/// first, so an unknown handle is reported as `UnknownHandle` before this
/// function runs. A present count buffer is refused here because no Metal
/// legacy indirect selector can consume it without a CPU readback.
pub(crate) fn validate(arguments: &IndirectArguments) -> Result<(), MetalError> {
    if arguments.argument_buffer.is_null() {
        return Err(MetalError::InvalidArg(
            "indirect argument buffer is nil",
        ));
    }
    if !arguments.argument_offset.is_multiple_of(4) {
        return Err(MetalError::InvalidArg(
            "indirect argument offset must be 4-byte aligned",
        ));
    }
    if arguments.max_count == 0 {
        return Err(MetalError::InvalidArg(
            "indirect max command count must be nonzero",
        ));
    }
    if arguments.max_count > MAX_INDIRECT_COMMANDS_V3 {
        return Err(MetalError::InvalidArg(
            "indirect max command count exceeds the v3 limit",
        ));
    }
    if let Some(count) = arguments.count {
        if count.buffer.is_null() {
            return Err(MetalError::InvalidArg("indirect count buffer is nil"));
        }
        if !count.offset.is_multiple_of(4) {
            return Err(MetalError::InvalidArg(
                "indirect count offset must be 4-byte aligned",
            ));
        }
        return Err(MetalError::NotSupported(
            "Metal's legacy indirect selectors have no count-buffer operand \
             (drawPrimitives:indirectBuffer:indirectBufferOffset:drawCount:stride: \
             does not exist in MTLRenderCommandEncoder.h); ExecuteIndirect count \
             clamping needs an MTLIndirectCommandBuffer execution-range path",
        ));
    }
    Ok(())
}

/// Encodes `max_count` non-indexed indirect draws into `enc`.
///
/// `enc` is a live `MTLRenderCommandEncoder`; the primitive topology and the
/// bound graphics pipeline/vertex buffers are already set by the caller.
pub(crate) fn encode_draw(
    enc: objc::Id,
    primitive: PrimitiveType,
    arguments: &IndirectArguments,
) -> Result<(), MetalError> {
    validate(arguments)?;
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    for offset in record_offsets(arguments, DRAW_ARGUMENT_STRIDE)? {
        // SAFETY: `enc` is a live render encoder, `argument_buffer` is a
        // registry-held buffer, and `offset` was bounds-checked by the caller
        // and advanced within `max_count * stride`.
        unsafe {
            type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize, objc::Id, usize);
            let send: Fn = std::mem::transmute(objc::msg_send_entry());
            send(
                enc,
                objc::sel(c"drawPrimitives:indirectBuffer:indirectBufferOffset:"),
                primitive_code(primitive),
                arguments.argument_buffer,
                offset,
            );
        }
    }
    Ok(())
}

/// Encodes `max_count` indexed indirect draws into `enc`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_draw_indexed(
    enc: objc::Id,
    primitive: PrimitiveType,
    index_type: IndexType,
    index_buffer: objc::Id,
    index_offset: usize,
    arguments: &IndirectArguments,
) -> Result<(), MetalError> {
    validate(arguments)?;
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    if index_buffer.is_null() {
        return Err(MetalError::InvalidArg("indirect index buffer is nil"));
    }
    for offset in record_offsets(arguments, DRAW_INDEXED_ARGUMENT_STRIDE)? {
        // SAFETY: `enc` is a live render encoder, both buffers are
        // registry-held, and every offset was validated.
        unsafe {
            type Fn = unsafe extern "C" fn(
                objc::Id,
                objc::Sel,
                usize,
                usize,
                objc::Id,
                usize,
                objc::Id,
                usize,
            );
            let send: Fn = std::mem::transmute(objc::msg_send_entry());
            send(
                enc,
                objc::sel(
                    c"drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
                ),
                primitive_code(primitive),
                index_type_code(index_type),
                index_buffer,
                index_offset,
                arguments.argument_buffer,
                offset,
            );
        }
    }
    Ok(())
}

/// Encodes `max_count` indirect compute dispatches into `enc`.
pub(crate) fn encode_dispatch_threadgroups(
    enc: objc::Id,
    threads_per_threadgroup: MTLSize,
    arguments: &IndirectArguments,
) -> Result<(), MetalError> {
    validate(arguments)?;
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "compute encoder is nil".to_string(),
        ));
    }
    if threads_per_threadgroup.width == 0
        || threads_per_threadgroup.height == 0
        || threads_per_threadgroup.depth == 0
    {
        return Err(MetalError::InvalidArg(
            "indirect dispatch threadgroup size must be nonzero",
        ));
    }
    for offset in record_offsets(arguments, DISPATCH_ARGUMENT_STRIDE)? {
        // SAFETY: `enc` is a live compute encoder, the buffer is registry-held
        // and `offset` names a complete 12-byte threadgroup record.
        unsafe {
            type Fn =
                unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, MTLSize);
            let send: Fn = std::mem::transmute(objc::msg_send_entry());
            send(
                enc,
                objc::sel(
                    c"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
                ),
                arguments.argument_buffer,
                offset,
                threads_per_threadgroup,
            );
        }
    }
    Ok(())
}

/// Encodes `max_count` indirect mesh-shader dispatches into `enc`.
pub(crate) fn encode_dispatch_mesh(
    enc: objc::Id,
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
    arguments: &IndirectArguments,
) -> Result<(), MetalError> {
    validate(arguments)?;
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "render encoder is nil".to_string(),
        ));
    }
    for offset in record_offsets(arguments, DISPATCH_ARGUMENT_STRIDE)? {
        // SAFETY: `enc` is a live render encoder with the mesh pipeline set;
        // the buffer is registry-held and every offset is a full record.
        unsafe {
            type Fn = unsafe extern "C" fn(
                objc::Id,
                objc::Sel,
                objc::Id,
                usize,
                MTLSize,
                MTLSize,
            );
            let send: Fn = std::mem::transmute(objc::msg_send_entry());
            send(
                enc,
                objc::sel(
                    c"drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
                ),
                arguments.argument_buffer,
                offset,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            );
        }
    }
    Ok(())
}

/// Byte offsets of every argument record, checked for `usize` overflow.
fn record_offsets(
    arguments: &IndirectArguments,
    stride: u64,
) -> Result<std::iter::StepBy<std::ops::Range<usize>>, MetalError> {
    let stride = usize::try_from(stride)
        .map_err(|_| MetalError::InvalidArg("indirect argument stride exceeds usize"))?;
    let max_count = arguments.max_count as usize;
    let base = arguments.argument_offset;
    let end = base
        .checked_add(
            max_count
                .checked_mul(stride)
                .ok_or(MetalError::InvalidArg(
                    "indirect argument range overflows usize",
                ))?,
        )
        .ok_or(MetalError::InvalidArg(
            "indirect argument range overflows usize",
        ))?;
    Ok((base..end).step_by(stride))
}

const fn primitive_code(primitive: PrimitiveType) -> usize {
    match primitive {
        PrimitiveType::Point => 0,
        PrimitiveType::Line => 1,
        PrimitiveType::LineStrip => 2,
        PrimitiveType::Triangle => 3,
        PrimitiveType::TriangleStrip => 4,
    }
}

const fn index_type_code(index_type: IndexType) -> usize {
    match index_type {
        IndexType::Uint16 => 0,
        IndexType::Uint32 => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(offset: usize, max_count: u32) -> IndirectArguments {
        IndirectArguments {
            // A non-null sentinel: every test below stops before a message send.
            argument_buffer: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            argument_offset: offset,
            max_count,
            count: None,
        }
    }

    #[test]
    fn macos_indirect_accepts_aligned_nonzero_ranges() {
        assert!(validate(&arguments(0, 1)).is_ok());
        assert!(validate(&arguments(64, MAX_INDIRECT_COMMANDS_V3)).is_ok());
    }

    #[test]
    fn macos_indirect_rejects_null_unaligned_zero_and_oversized_ranges() {
        let mut null = arguments(0, 1);
        null.argument_buffer = std::ptr::null_mut();
        assert!(matches!(
            validate(&null),
            Err(MetalError::InvalidArg(
                "indirect argument buffer is nil"
            ))
        ));

        assert!(matches!(
            validate(&arguments(2, 1)),
            Err(MetalError::InvalidArg(
                "indirect argument offset must be 4-byte aligned"
            ))
        ));
        assert!(matches!(
            validate(&arguments(0, 0)),
            Err(MetalError::InvalidArg(
                "indirect max command count must be nonzero"
            ))
        ));
        assert!(matches!(
            validate(&arguments(0, MAX_INDIRECT_COMMANDS_V3 + 1)),
            Err(MetalError::InvalidArg(
                "indirect max command count exceeds the v3 limit"
            ))
        ));
    }

    #[test]
    fn macos_indirect_refuses_a_count_buffer_without_reading_it() {
        let mut arguments = arguments(0, 1);
        arguments.count = Some(IndirectCount {
            buffer: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            offset: 0,
        });
        assert!(matches!(
            validate(&arguments),
            Err(MetalError::NotSupported(_))
        ));

        arguments.count = Some(IndirectCount {
            buffer: std::ptr::null_mut(),
            offset: 0,
        });
        assert!(matches!(
            validate(&arguments),
            Err(MetalError::InvalidArg("indirect count buffer is nil"))
        ));
    }

    #[test]
    fn macos_indirect_record_stride_matches_the_metal_abi() {
        assert_eq!(DRAW_ARGUMENT_STRIDE, 16);
        assert_eq!(DRAW_INDEXED_ARGUMENT_STRIDE, 20);
        assert_eq!(DISPATCH_ARGUMENT_STRIDE, 12);
        let offsets: Vec<usize> = record_offsets(&arguments(8, 3), DRAW_ARGUMENT_STRIDE)
            .expect("offsets")
            .collect();
        assert_eq!(offsets, vec![8, 24, 40]);
    }

    #[test]
    fn macos_indirect_production_source_has_no_readback_call() {
        // The host side must never map the argument or count buffer. This is
        // the structural half of the no-readback proof; the GPU test asserts
        // the pixel half.
        let source = include_str!("indirect.rs");
        let code = source.split("#[cfg(test)]").next().expect("production half");
        assert!(!code.contains("getBytes"));
        assert!(!code.contains("didModifyRange"));
        assert!(!code.contains("contents"));
    }
}
