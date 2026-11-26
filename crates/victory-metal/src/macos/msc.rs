//! Typed binder for the verified tiny Metal Shader Converter compute root
//! signature.
//!
//! Metal Shader Converter (`libmetalirconverter.dylib`) lowers a DXIL compute
//! shader with the synthetic root signature below into a Metal function whose
//! only resource is reached through a *top-level global argument buffer*:
//!
//! ```text
//! HLSL: RootFlags(0), UAV(u0)
//! Metal: buffer(2) -> 8-byte device address of the UAV at argument offset 0
//! ```
//!
//! The constants were observed end-to-end by `tools/macos/metal-poc/arm64-dxil.sh`
//! (x86_64 converter process, separate arm64 renderer process). This module does
//! not guess layouts for arbitrary titles: it is an opt-in typed binding for
//! exactly that verified shape. Callers that need another layout must derive and
//! verify it first, then extend this module explicitly.
//!
//! The binder writes a checked [`MscUavDispatch`] against the existing pipeline
//! and registered-buffer tables; see
//! [`crate::macos::device::DirectMetalTransport::dispatch_msc_uav_1d`].

use crate::transport::MetalError;
use crate::wire;

/// Byte length of the synthetic top-level argument buffer.
///
/// Matches the verified harness: a 32-byte buffer with the UAV device address at
/// offset 0. Metal only reads the declared fields, so the slack keeps the shape
/// explicit without changing converted-shader semantics.
pub const MSC_ARGUMENT_BUFFER_LENGTH: usize = 32;

/// Metal buffer argument index of the converted top-level argument buffer.
pub const MSC_DEFAULT_ARGUMENT_BUFFER_INDEX: usize = 2;

/// Byte offset of the UAV device address inside the top-level argument buffer.
pub const MSC_DEFAULT_UAV_ADDRESS_OFFSET: usize = 0;

/// Size of one encoded GPU device address (`MTLGPUAddress`).
pub const MSC_DEVICE_ADDRESS_SIZE: usize = 8;

/// Exclusive upper bound for a Metal buffer argument index.
///
/// Metal exposes buffer argument indices `0..=30` for compute encoders; index 31
/// and above is never a valid `setBuffer:offset:atIndex:` target.
pub const MSC_MAX_BUFFER_ARGUMENT_INDEX: usize = 31;

/// Validated layout of the converted top-level argument buffer.
///
/// Fields are private so a caller cannot construct a layout that violates the
/// device-address alignment or the argument-buffer length; use
/// [`MscUavLayout::new`] or [`MscUavLayout::DEFAULT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MscUavLayout {
    argument_buffer_index: usize,
    address_offset: usize,
}

impl MscUavLayout {
    /// The verified layout from `arm64-dxil.sh`: `buffer(2)`, address at offset 0.
    pub const DEFAULT: Self = Self {
        argument_buffer_index: MSC_DEFAULT_ARGUMENT_BUFFER_INDEX,
        address_offset: MSC_DEFAULT_UAV_ADDRESS_OFFSET,
    };

    /// Builds an explicit layout for a verified converted root signature.
    ///
    /// * `argument_buffer_index` must be a valid Metal buffer argument index
    ///   (`0..31`).
    /// * `address_offset` must be 8-byte aligned and leave room for one device
    ///   address inside [`MSC_ARGUMENT_BUFFER_LENGTH`].
    ///
    /// Returns [`MetalError::InvalidArg`] for any other combination. This binder
    /// intentionally has no fallback and no probing: an invalid layout is
    /// rejected before any Metal object is created.
    pub fn new(argument_buffer_index: usize, address_offset: usize) -> Result<Self, MetalError> {
        if argument_buffer_index >= MSC_MAX_BUFFER_ARGUMENT_INDEX {
            return Err(MetalError::InvalidArg(
                "argument buffer index must be a valid Metal buffer index",
            ));
        }
        if !address_offset.is_multiple_of(MSC_DEVICE_ADDRESS_SIZE) {
            return Err(MetalError::InvalidArg(
                "UAV device address offset must be 8-byte aligned",
            ));
        }
        let Some(end) = address_offset.checked_add(MSC_DEVICE_ADDRESS_SIZE) else {
            return Err(MetalError::InvalidArg(
                "UAV device address offset overflows the argument buffer",
            ));
        };
        if end > MSC_ARGUMENT_BUFFER_LENGTH {
            return Err(MetalError::InvalidArg(
                "UAV device address does not fit in the argument buffer",
            ));
        }
        Ok(Self {
            argument_buffer_index,
            address_offset,
        })
    }

    /// Metal buffer argument index where the argument buffer is bound.
    pub fn argument_buffer_index(&self) -> usize {
        self.argument_buffer_index
    }

    /// Byte offset of the UAV device address inside the argument buffer.
    pub fn address_offset(&self) -> usize {
        self.address_offset
    }

    /// Byte length of the argument buffer this layout describes.
    pub fn argument_buffer_len(&self) -> usize {
        MSC_ARGUMENT_BUFFER_LENGTH
    }
}

impl Default for MscUavLayout {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One opt-in 1D compute dispatch through the converted MSC argument buffer.
///
/// `offset`/`length` are an explicit range inside the *registered backing* of
/// `uav`, resolved by the caller from the win64 allocation/generation mapping.
/// They are validated again against the registry at dispatch time; the binder
/// never trusts a caller-computed end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MscUavDispatch {
    /// Pipeline registered by `compile_dxil`, `compile_msl`, or `load_metallib`.
    pub pipeline: crate::abi::VmPipelineId,
    /// Registered buffer that backs the converted UAV.
    pub uav: crate::abi::VmBufferId,
    /// Byte offset of the UAV range inside the registered backing.
    pub offset: u64,
    /// Byte length of the UAV range.
    pub length: u64,
    /// Total threads in the 1D grid (nonzero).
    pub threads: u32,
    /// Threads per threadgroup (nonzero, within the pipeline maximum).
    pub threads_per_group: u32,
    /// Layout of the converted top-level argument buffer.
    pub layout: MscUavLayout,
}

/// Validates a UAV byte range against the registered backing length.
///
/// Returns `(offset, end)` as host `usize` values. The rules mirror the legacy
/// wire batch contract: nonzero length capped by [`wire::MAX_RESOURCE_LEN`],
/// 4-byte offset alignment, overflow-checked end, and `end <= registered_len`.
pub(crate) fn checked_binding_range(
    offset: u64,
    length: u64,
    registered_len: usize,
) -> Result<(usize, usize), MetalError> {
    if length == 0 {
        return Err(MetalError::InvalidArg("UAV range length must be nonzero"));
    }
    if length > wire::MAX_RESOURCE_LEN {
        return Err(MetalError::InvalidArg(
            "UAV range exceeds the maximum resource length",
        ));
    }
    if !offset.is_multiple_of(crate::macos::device::MTL_BUFFER_OFFSET_ALIGNMENT) {
        return Err(MetalError::InvalidArg(
            "UAV range offset must be 4-byte aligned",
        ));
    }
    let Some(end) = offset.checked_add(length) else {
        return Err(MetalError::InvalidArg("UAV range end overflows"));
    };
    let (Ok(offset), Ok(end)) = (usize::try_from(offset), usize::try_from(end)) else {
        return Err(MetalError::InvalidArg(
            "UAV range does not fit host address space",
        ));
    };
    if end > registered_len {
        return Err(MetalError::InvalidArg(
            "UAV range exceeds the registered backing length",
        ));
    }
    Ok((offset, end))
}

/// Encodes `address` into `arguments` according to `layout`.
///
/// The 8-byte field is written little-endian, matching the Metal ABI on both
/// supported Mac architectures. The rest of the buffer is left untouched so the
/// caller can zero and reuse it deterministically.
pub(crate) fn write_uav_address(
    arguments: &mut [u8],
    layout: MscUavLayout,
    address: u64,
) -> Result<(), MetalError> {
    let end = layout
        .address_offset()
        .checked_add(MSC_DEVICE_ADDRESS_SIZE)
        .ok_or(MetalError::InvalidArg(
            "UAV device address offset overflows",
        ))?;
    if end > arguments.len() {
        return Err(MetalError::InvalidArg(
            "argument buffer slice is shorter than its layout",
        ));
    }
    arguments[layout.address_offset()..end].copy_from_slice(&address.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MSC_ARGUMENT_BUFFER_LENGTH, MSC_DEVICE_ADDRESS_SIZE, MscUavLayout, checked_binding_range,
        write_uav_address,
    };
    use crate::transport::MetalError;
    use crate::wire;

    #[test]
    fn default_layout_matches_verified_harness() {
        let layout = MscUavLayout::DEFAULT;
        assert_eq!(layout.argument_buffer_index(), 2);
        assert_eq!(layout.address_offset(), 0);
        assert_eq!(layout.argument_buffer_len(), 32);
        assert_eq!(MscUavLayout::default(), layout);
    }

    #[test]
    fn layout_rejects_out_of_range_index() {
        assert!(matches!(
            MscUavLayout::new(31, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(usize::MAX, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(MscUavLayout::new(30, 0).is_ok());
    }

    #[test]
    fn layout_rejects_misaligned_or_oversized_address_offset() {
        assert!(matches!(
            MscUavLayout::new(2, 1),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(2, MSC_ARGUMENT_BUFFER_LENGTH - MSC_DEVICE_ADDRESS_SIZE + 1),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(2, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(MscUavLayout::new(2, MSC_ARGUMENT_BUFFER_LENGTH - MSC_DEVICE_ADDRESS_SIZE).is_ok());
    }

    #[test]
    fn range_validation_matches_registry_contract() {
        assert_eq!(
            checked_binding_range(0, 8, 8).expect("in-bounds range"),
            (0, 8)
        );
        assert_eq!(
            checked_binding_range(4, 8, 12).expect("in-bounds offset range"),
            (4, 12)
        );
        assert!(matches!(
            checked_binding_range(0, 0, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(2, 4, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(8, 4, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(0, wire::MAX_RESOURCE_LEN + 1, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(u64::MAX - 3, 8, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn uav_address_is_encoded_little_endian_at_the_layout_offset() {
        let mut arguments = [0u8; MSC_ARGUMENT_BUFFER_LENGTH];
        let address = 0x1122_3344_5566_7788u64;
        write_uav_address(&mut arguments, MscUavLayout::DEFAULT, address)
            .expect("default layout must fit the verified argument buffer");
        assert_eq!(
            &arguments[..MSC_DEVICE_ADDRESS_SIZE],
            &address.to_le_bytes()
        );
        assert!(arguments[MSC_DEVICE_ADDRESS_SIZE..].iter().all(|&b| b == 0));

        let shifted = MscUavLayout::new(2, 8).expect("offset 8 is aligned and fits");
        let mut arguments = [0u8; MSC_ARGUMENT_BUFFER_LENGTH];
        write_uav_address(&mut arguments, shifted, address).expect("shifted layout must fit");
        assert_eq!(&arguments[..8], &[0u8; 8]);
        assert_eq!(&arguments[8..16], &address.to_le_bytes());
    }

    #[test]
    fn address_encoding_rejects_a_short_slice() {
        let mut arguments = [0u8; MSC_DEVICE_ADDRESS_SIZE - 1];
        assert!(matches!(
            write_uav_address(&mut arguments, MscUavLayout::DEFAULT, 1),
            Err(MetalError::InvalidArg(_))
        ));
    }
}
