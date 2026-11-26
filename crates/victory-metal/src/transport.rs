//! Host-OS-agnostic backend transport interface for `victory-metal`.
//!
//! Phase 1 bridges callers in Wine across `__wine_unix_call` using the C ABI
//! (`c_api`), while Phase 2 (native macOS Mach-O host without Wine) can invoke
//! [`MetalTransport`] directly in-process via `DirectMetalTransport`.

use thiserror::Error;

use crate::abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmPipelineId, VmStatus};

/// Errors produced by a [`MetalTransport`] implementation.
#[derive(Debug, Error)]
pub enum MetalError {
    /// Invalid argument passed by the caller.
    #[error("invalid argument: {0}")]
    InvalidArg(&'static str),
    /// Operation unsupported on this OS, CPU architecture, or configuration.
    #[error("not supported: {0}")]
    NotSupported(&'static str),
    /// Metal could not acquire a system default GPU device.
    #[error("no Metal device available")]
    DeviceUnavailable,
    /// Pointer or size was not aligned to the required hardware page boundary.
    #[error("pointer 0x{ptr:x} (len {len}) must be {page_size}-byte aligned and non-empty")]
    AlignmentError {
        /// Base host address.
        ptr: usize,
        /// Region length in bytes.
        len: usize,
        /// Required alignment in bytes.
        page_size: usize,
    },
    /// Shader or pipeline state creation failed.
    #[error("shader/pipeline compilation failed: {0}")]
    CompileFailed(String),
    /// Referenced buffer or pipeline handle was not found.
    #[error("handle not found: {0}")]
    NotFound(String),
    /// Metal command execution or internal synchronization failed.
    #[error("internal Metal error: {0}")]
    InternalError(String),
}

impl MetalError {
    /// Maps a [`MetalError`] to its stable C ABI [`VmStatus`] code.
    pub fn status(&self) -> VmStatus {
        match self {
            Self::InvalidArg(_) => VmStatus::InvalidArg,
            Self::NotSupported(_) => VmStatus::NotSupported,
            Self::DeviceUnavailable => VmStatus::DeviceUnavailable,
            Self::AlignmentError { .. } => VmStatus::AlignmentError,
            Self::CompileFailed(_) => VmStatus::CompileFailed,
            Self::NotFound(_) => VmStatus::NotFound,
            Self::InternalError(_) => VmStatus::InternalError,
        }
    }
}

/// Abstract transport boundary implemented by `DirectMetalTransport` on macOS
/// (and pluggable by remote/Wine bridge shims without changing `victory-metal` core).
pub trait MetalTransport: Send + Sync {
    /// Returns device capabilities and zero-copy page alignment constraints.
    fn device_info(&self) -> Result<VmDeviceInfo, MetalError>;

    /// Wraps page-aligned host memory directly via `newBufferWithBytesNoCopy`.
    ///
    /// # Safety
    /// The memory region `[ptr, ptr + len)` must remain valid, readable/writable,
    /// and unfreed until [`MetalTransport::unregister_buffer`] returns and no
    /// GPU work referencing the returned buffer is still executing.
    unsafe fn register_zerocopy(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError>;

    /// Releases the Metal buffer wrapper associated with `id`.
    /// Does not free the backing host allocation (`deallocator: nil`).
    fn unregister_buffer(&self, id: VmBufferId) -> Result<(), MetalError>;

    /// Compiles a Metal Shading Language (MSL) compute kernel into a pipeline.
    fn compile_msl(&self, source: &str, entry: &str) -> Result<VmPipelineId, MetalError>;

    /// Compiles DXIL bytecode into a Metal compute pipeline via `libmetalirconverter.dylib`.
    fn compile_dxil(
        &self,
        dxil: &[u8],
        entry: &str,
    ) -> Result<(VmPipelineId, VmDxilCompileOutput), MetalError>;

    /// Encodes and synchronously executes a 1D compute dispatch binding `buffers`
    /// sequentially at `[[buffer(0)]]`, `[[buffer(1)]]`, ...
    fn dispatch_compute_1d(
        &self,
        pipeline: VmPipelineId,
        buffers: &[VmBufferId],
        threads: u32,
        threads_per_group: u32,
    ) -> Result<(), MetalError>;
}
