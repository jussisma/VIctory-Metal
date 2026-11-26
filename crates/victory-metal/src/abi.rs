//! Stable C ABI types shared between `victory-metal` and any host transport.
//!
//! Nothing in this module depends on Wine, Windows headers, or macOS frameworks,
//! allowing Phase 1 (Wine unixlib bridge) and Phase 2 (Wine-free native macOS
//! host) to share the exact same ABI types.

use core::fmt;

/// Status codes returned by the `victory-metal` C ABI.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// Caller supplied a null pointer, empty slice, or invalid handle/parameter.
    InvalidArg = 1,
    /// Feature or binary format is not supported in the current environment
    /// (for example, `libmetalirconverter.dylib` on non-`x86_64` or non-macOS).
    NotSupported = 2,
    /// No Metal GPU device could be opened.
    DeviceUnavailable = 3,
    /// Host pointer or length did not satisfy zero-copy page alignment rules.
    AlignmentError = 4,
    /// Shader or pipeline compilation failed.
    CompileFailed = 5,
    /// Requested buffer, library, or pipeline handle was not found.
    NotFound = 6,
    /// Internal Metal execution or host runtime error.
    InternalError = 7,
}

/// Opaque handle to a registered zero-copy or Metal-backed buffer.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VmBufferId(pub u64);

/// Opaque handle to a compiled `MTLLibrary`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VmLibraryId(pub u64);

/// Opaque handle to a compiled compute/render pipeline state.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VmPipelineId(pub u64);

/// Metal device capabilities exposed across the ABI boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmDeviceInfo {
    /// UTF-8 device name, NUL-terminated.
    pub name: [u8; 128],
    /// Non-zero when the GPU and CPU share unified memory (`hasUnifiedMemory`).
    pub has_unified_memory: u8,
    /// Maximum scalar threads allowed per threadgroup for compute shaders.
    pub max_threads_per_threadgroup: u32,
    /// Metal's recommended maximum working set size in bytes.
    pub recommended_max_working_set_size: u64,
    /// Minimum page alignment (bytes) required by `newBufferWithBytesNoCopy`.
    pub page_size: u64,
}

impl Default for VmDeviceInfo {
    fn default() -> Self {
        Self {
            name: [0; 128],
            has_unified_memory: 0,
            max_threads_per_threadgroup: 0,
            recommended_max_working_set_size: 0,
            page_size: 0,
        }
    }
}

impl VmDeviceInfo {
    /// Returns the NUL-terminated device name as UTF-8.
    pub fn name_str(&self) -> String {
        let end = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.name.len());
        String::from_utf8_lossy(&self.name[..end]).into_owned()
    }
}

impl fmt::Display for VmDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (unified={}, max_threads={}, page_size={})",
            self.name_str(),
            self.has_unified_memory != 0,
            self.max_threads_per_threadgroup,
            self.page_size
        )
    }
}

/// Summary metadata returned after converting DXIL into a Metal binary library.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmDxilCompileOutput {
    /// `IRShaderStage` integer returned by `libmetalirconverter.dylib`.
    pub stage: u32,
    /// Size in bytes of the generated `.metallib` container.
    pub metallib_bytes: usize,
}
