//! Dynamic loader and RAII wrapper around Apple's `libmetalirconverter.dylib`.
//!
//! Note: Apple currently ships `libmetalirconverter.dylib` inside
//! `D3DMetal.framework/Versions/A/Resources/` as a **Mach-O `x86_64` binary**,
//! matching Rosetta 2's `x86_64` address space used by CrossOver Wine and by the
//! future native Mach-O host. Calling from an `arm64` test binary returns
//! `MetalError::NotSupported` cleanly.
//!
//! Two entry points share the same conversion core:
//! - [`compile_dxil_to_library`] loads the converted bytes into an `MTLLibrary`
//!   in the current process (requires a Metal device).
//! - [`convert_dxil_to_metallib`] serializes the same bytes to a `Vec<u8>` with
//!   no `MTLDevice`, `MTLLibrary`, or GPU work, for path-in/path-out tooling and
//!   a separate ARM64 loader process.
//!
//! Both paths consume the root signature embedded in the DXIL by the original
//! shader. `IRCompilerSetGlobalRootSignature` is intentionally never resolved or
//! called, so there is no implicit/explicit root-signature override.
//!
//! The DXIL input is passed with ownership `1` (`IR_BYTECODE_OWNERSHIP_COPY`):
//! IRConverter copies the bytes during `IRObjectCreateFromDXIL`, so the borrowed
//! Rust slice may be dropped immediately afterwards. Callers must not rely on
//! IRConverter retaining the pointer.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr::{NonNull, null_mut};
use std::slice;

use crate::abi::VmDxilCompileOutput;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};
use crate::transport::MetalError;

const RTLD_NOW: c_int = 0x2;
const IR_BYTECODE_OWNERSHIP_COPY: c_int = 1;

/// Maximum accepted DXIL input size for [`convert_dxil_to_metallib`].
///
/// Real DXIL modules are typically tens to hundreds of KiB. The bound keeps a
/// malformed or hostile input from asking IRConverter or this process to
/// allocate without limit.
pub const MAX_DXIL_INPUT_BYTES: usize = 16 * 1024 * 1024;

/// Maximum accepted entry-point name length in bytes.
pub const MAX_ENTRY_POINT_BYTES: usize = 256;

/// Maximum accepted serialized metallib output size.
///
/// The bound is checked against both `IRMetalLibGetBytecodeSize` and the mapped
/// `dispatch_data_t` length before any output is written.
pub const MAX_METALLIB_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

#[link(name = "System")]
unsafe extern "C" {
    /// Flattens a possibly discontiguous `dispatch_data_t` into one mapped range.
    ///
    /// The returned object retains the mapping; the buffer pointer is valid only
    /// until that object is released.
    fn dispatch_data_create_map(
        data: *mut c_void,
        buffer: *mut *mut c_void,
        size: *mut usize,
    ) -> *mut c_void;
}

/// Bytecode produced by [`convert_dxil_to_metallib`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetallibOutput {
    /// Serialized metallib bytes. Non-empty and at most
    /// [`MAX_METALLIB_OUTPUT_BYTES`] long.
    pub bytes: Vec<u8>,
    /// Metal shader stage reported by `IRObjectGetMetalIRShaderStage`
    /// (`8` is compute for the verified synthetic kernel).
    pub stage: u32,
}

enum IRCompiler {}
enum IRObject {}
enum IRMetalLibBinary {}
enum IRError {}

type FnIRCompilerCreate = unsafe extern "C" fn() -> *mut IRCompiler;
type FnIRCompilerDestroy = unsafe extern "C" fn(*mut IRCompiler);
type FnIRCompilerSetEntryPointName = unsafe extern "C" fn(*mut IRCompiler, *const c_char);
type FnIRObjectCreateFromDXIL =
    unsafe extern "C" fn(*const u8, usize, c_int) -> *mut IRObject;
type FnIRObjectDestroy = unsafe extern "C" fn(*mut IRObject);
type FnIRObjectGetMetalIRShaderStage = unsafe extern "C" fn(*const IRObject) -> c_int;
type FnIRCompilerAllocCompileAndLink = unsafe extern "C" fn(
    *mut IRCompiler,
    *const c_char,
    *const IRObject,
    *mut *mut IRError,
) -> *mut IRObject;
type FnIRMetalLibBinaryCreate = unsafe extern "C" fn() -> *mut IRMetalLibBinary;
type FnIRMetalLibBinaryDestroy = unsafe extern "C" fn(*mut IRMetalLibBinary);
type FnIRObjectGetMetalLibBinary =
    unsafe extern "C" fn(*const IRObject, c_int, *mut IRMetalLibBinary) -> bool;
type FnIRMetalLibGetBytecodeSize = unsafe extern "C" fn(*const IRMetalLibBinary) -> usize;
type FnIRMetalLibGetBytecodeData = unsafe extern "C" fn(*const IRMetalLibBinary) -> *mut c_void;
type FnIRErrorGetCode = unsafe extern "C" fn(*const IRError) -> u32;
type FnIRErrorDestroy = unsafe extern "C" fn(*mut IRError);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

struct DynLib {
    handle: NonNull<c_void>,
}

// SAFETY: Dynamic library handles resolved read-only from `dlopen` are thread-safe to hold.
unsafe impl Send for DynLib {}
// SAFETY: Symbol lookups do not mutate the shared handle.
unsafe impl Sync for DynLib {}

impl Drop for DynLib {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was returned by a successful `dlopen`.
        unsafe {
            let _ = dlclose(self.handle.as_ptr());
        }
    }
}

struct Api {
    _lib: DynLib,
    compiler_create: FnIRCompilerCreate,
    compiler_destroy: FnIRCompilerDestroy,
    set_entry_point: FnIRCompilerSetEntryPointName,
    from_dxil: FnIRObjectCreateFromDXIL,
    obj_destroy: FnIRObjectDestroy,
    get_stage: FnIRObjectGetMetalIRShaderStage,
    compile_and_link: FnIRCompilerAllocCompileAndLink,
    metallib_create: FnIRMetalLibBinaryCreate,
    metallib_destroy: FnIRMetalLibBinaryDestroy,
    get_metallib: FnIRObjectGetMetalLibBinary,
    metallib_size: FnIRMetalLibGetBytecodeSize,
    metallib_data: FnIRMetalLibGetBytecodeData,
    err_code: FnIRErrorGetCode,
    err_destroy: FnIRErrorDestroy,
}

unsafe fn load_sym<T: Copy>(handle: *mut c_void, name: &CStr) -> Result<T, MetalError> {
    if mem::size_of::<T>() != mem::size_of::<*mut c_void>() {
        return Err(MetalError::InternalError(
            "invalid function pointer size".to_string(),
        ));
    }
    // SAFETY: `handle` is an open `dlopen` handle and `name` is NUL-terminated.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    let Some(ptr) = NonNull::new(sym) else {
        return Err(MetalError::CompileFailed(format!(
            "missing IRConverter symbol `{}`",
            name.to_string_lossy()
        )));
    };
    // SAFETY: Caller specifies exact C symbol signature matching Apple's IRConverter ABI.
    Ok(unsafe { mem::transmute_copy::<*mut c_void, T>(&ptr.as_ptr()) })
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(custom) = std::env::var_os("VICTORY_METAL_IRCONVERTER")
        && !custom.is_empty()
    {
        paths.push(PathBuf::from(custom));
    }
    if let Some(d3dmetal_res) = std::env::var_os("D3DMETAL_RES")
        && !d3dmetal_res.is_empty()
    {
        paths.push(PathBuf::from(d3dmetal_res).join("libmetalirconverter.dylib"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let base = PathBuf::from(home);
        paths.push(
            base.join("Library/Caches/victory-macos/target/wine-cx/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib"),
        );
    }
    paths.push(PathBuf::from(
        "/Library/Application Support/CrossOver/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
    ));
    paths
}

fn load_api(lib: DynLib) -> Result<Api, MetalError> {
    let h = lib.handle.as_ptr();
    // SAFETY: Function signatures match Apple's verified `libmetalirconverter.dylib` C API.
    unsafe {
        Ok(Api {
            compiler_create: load_sym(h, c"IRCompilerCreate")?,
            compiler_destroy: load_sym(h, c"IRCompilerDestroy")?,
            set_entry_point: load_sym(h, c"IRCompilerSetEntryPointName")?,
            from_dxil: load_sym(h, c"IRObjectCreateFromDXIL")?,
            obj_destroy: load_sym(h, c"IRObjectDestroy")?,
            get_stage: load_sym(h, c"IRObjectGetMetalIRShaderStage")?,
            compile_and_link: load_sym(h, c"IRCompilerAllocCompileAndLink")?,
            metallib_create: load_sym(h, c"IRMetalLibBinaryCreate")?,
            metallib_destroy: load_sym(h, c"IRMetalLibBinaryDestroy")?,
            get_metallib: load_sym(h, c"IRObjectGetMetalLibBinary")?,
            metallib_size: load_sym(h, c"IRMetalLibGetBytecodeSize")?,
            metallib_data: load_sym(h, c"IRMetalLibGetBytecodeData")?,
            err_code: load_sym(h, c"IRErrorGetCode")?,
            err_destroy: load_sym(h, c"IRErrorDestroy")?,
            _lib: lib,
        })
    }
}

/// Opens an explicitly configured converter library path.
///
/// This is the deterministic path used by [`convert_dxil_to_metallib`]: the
/// caller names one file, and no environment variable or search fallback can
/// silently select a different library.
fn open_api_from_path(path: &Path) -> Result<Api, MetalError> {
    if !path.is_file() {
        return Err(MetalError::NotSupported(
            "configured IRConverter path is not a regular file",
        ));
    }
    let Ok(cpath) = CString::new(path.to_string_lossy().as_bytes()) else {
        return Err(MetalError::InvalidArg(
            "converter path contains an interior NUL",
        ));
    };
    // SAFETY: `cpath` is NUL-terminated and `handle` is an open `dlopen` handle.
    let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
    let Some(nn) = NonNull::new(handle) else {
        // SAFETY: `dlerror()` returns a valid static/thread-local C string or null after `dlopen`.
        let err_ptr = unsafe { dlerror() };
        let detail = if err_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: `err_ptr` is NUL-terminated per POSIX `dlerror`.
            unsafe { CStr::from_ptr(err_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        if detail.contains("incompatible architecture") {
            return Err(MetalError::NotSupported(
                "libmetalirconverter.dylib is x86_64-only (run under Rosetta 2 x86_64)",
            ));
        }
        return Err(MetalError::NotSupported(
            "failed to load the configured IRConverter library",
        ));
    };
    load_api(DynLib { handle: nn })
}

fn open_api() -> Result<Api, MetalError> {
    let mut last_err = String::new();
    let mut opened = None;
    for path in candidate_paths() {
        if !path.is_file() {
            continue;
        }
        let Ok(cpath) = CString::new(path.to_string_lossy().as_bytes()) else {
            continue;
        };
        // SAFETY: `cpath` is a NUL-terminated path to an existing regular file.
        let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
        if let Some(nn) = NonNull::new(handle) {
            opened = Some(DynLib { handle: nn });
            break;
        }
        // SAFETY: `dlerror()` returns a valid static/thread-local C string or null after `dlopen`.
        let err_ptr = unsafe { dlerror() };
        if !err_ptr.is_null() {
            // SAFETY: `err_ptr` is NUL-terminated per POSIX `dlerror`.
            last_err = unsafe { CStr::from_ptr(err_ptr) }
                .to_string_lossy()
                .into_owned();
        }
    }

    let Some(lib) = opened else {
        if last_err.contains("incompatible architecture") {
            return Err(MetalError::NotSupported(
                "libmetalirconverter.dylib is x86_64-only (run under Rosetta 2 x86_64)",
            ));
        }
        return Err(MetalError::NotSupported(
            "libmetalirconverter.dylib not found (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)",
        ));
    };
    load_api(lib)
}

struct Scoped<T> {
    ptr: *mut T,
    drop_fn: unsafe extern "C" fn(*mut T),
}

impl<T> Drop for Scoped<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was created by the paired IRConverter constructor.
            unsafe { (self.drop_fn)(self.ptr) };
        }
    }
}

fn validate_dxil(dxil: &[u8]) -> Result<(), MetalError> {
    if dxil.is_empty() {
        return Err(MetalError::InvalidArg("DXIL bytecode slice is empty"));
    }
    if dxil.len() > MAX_DXIL_INPUT_BYTES {
        return Err(MetalError::InvalidArg(
            "DXIL input exceeds the 16 MiB limit",
        ));
    }
    Ok(())
}

fn c_entry(entry: &str) -> Result<CString, MetalError> {
    if entry.is_empty() {
        return Err(MetalError::InvalidArg("entry point is empty"));
    }
    if entry.len() > MAX_ENTRY_POINT_BYTES {
        return Err(MetalError::InvalidArg(
            "entry point exceeds the 256-byte limit",
        ));
    }
    CString::new(entry).map_err(|_| MetalError::InvalidArg("entry point contains NUL"))
}

/// Converted IRConverter output that keeps its source library loaded.
struct CompiledBinary<'a> {
    stage: i32,
    // Drop the binary before the linked AIR object (`bin` is declared first).
    bin: Scoped<IRMetalLibBinary>,
    _air: Scoped<IRObject>,
    size_fn: FnIRMetalLibGetBytecodeSize,
    data_fn: FnIRMetalLibGetBytecodeData,
    _api: std::marker::PhantomData<&'a Api>,
}

impl CompiledBinary<'_> {
    fn byte_len(&self) -> usize {
        // SAFETY: `self.bin.ptr` is a valid IRMetalLibBinary for this values' lifetime.
        unsafe { (self.size_fn)(self.bin.ptr) }
    }

    fn dispatch_data(&self) -> Result<*mut c_void, MetalError> {
        // SAFETY: `self.bin.ptr` is valid and IRMetalLibGetBytecodeData returns a
        // borrowed dispatch_data_t tied to the binary's lifetime.
        let data = unsafe { (self.data_fn)(self.bin.ptr) };
        if data.is_null() || self.byte_len() == 0 {
            return Err(MetalError::CompileFailed(
                "IRMetalLibGetBytecodeData returned empty bytecode".to_string(),
            ));
        }
        Ok(data)
    }

    /// Copies the borrowed `dispatch_data_t` into owned bytes without loading a
    /// Metal library or touching a GPU.
    fn serialized_bytes(&self) -> Result<Vec<u8>, MetalError> {
        let expected = self.byte_len();
        if expected == 0 {
            return Err(MetalError::CompileFailed(
                "IRMetalLibGetBytecodeSize returned zero".to_string(),
            ));
        }
        if expected > MAX_METALLIB_OUTPUT_BYTES {
            return Err(MetalError::CompileFailed(format!(
                "metallib output is {expected} bytes; limit is {MAX_METALLIB_OUTPUT_BYTES}"
            )));
        }

        let data = self.dispatch_data()?;
        let mut mapped: *mut c_void = null_mut();
        let mut mapped_len = 0usize;
        // SAFETY: `data` is a valid borrowed dispatch_data_t; the out-pointers are
        // valid stack slots, and the returned object owns the mapped range.
        let flat_raw = unsafe { dispatch_data_create_map(data, &mut mapped, &mut mapped_len) };
        let Some(_flat) = ObjcObject::from_retained(flat_raw) else {
            return Err(MetalError::CompileFailed(
                "dispatch_data_create_map returned null".to_string(),
            ));
        };
        if mapped.is_null() || mapped_len == 0 {
            return Err(MetalError::CompileFailed(
                "dispatch_data_create_map returned an empty range".to_string(),
            ));
        }
        if mapped_len != expected {
            return Err(MetalError::InternalError(format!(
                "IRConverter bytecode size mismatch: reported {expected}, mapped {mapped_len}"
            )));
        }
        // SAFETY: `mapped` points to `mapped_len` readable bytes owned by `_flat`,
        // which stays alive until after this copy.
        let bytes = unsafe { slice::from_raw_parts(mapped.cast::<u8>(), mapped_len) }.to_vec();
        Ok(bytes)
    }
}

fn compile_to_binary<'a>(
    api: &'a Api,
    dxil: &[u8],
    entry_c: &CStr,
) -> Result<CompiledBinary<'a>, MetalError> {
    // SAFETY: `IRCompilerCreate` takes no arguments and returns a newly allocated compiler.
    let compiler_raw = unsafe { (api.compiler_create)() };
    if compiler_raw.is_null() {
        return Err(MetalError::CompileFailed(
            "IRCompilerCreate returned null".to_string(),
        ));
    }
    let compiler = Scoped {
        ptr: compiler_raw,
        drop_fn: api.compiler_destroy,
    };

    // SAFETY: `entry_c` is a valid NUL-terminated string and `compiler.ptr` is non-null.
    unsafe { (api.set_entry_point)(compiler.ptr, entry_c.as_ptr()) };

    // SAFETY: `dxil` slice is valid for `dxil.len()` bytes; ownership `1` instructs
    // IRConverter to copy the bytes before `IRObjectCreateFromDXIL` returns.
    let input_raw =
        unsafe { (api.from_dxil)(dxil.as_ptr(), dxil.len(), IR_BYTECODE_OWNERSHIP_COPY) };
    if input_raw.is_null() {
        return Err(MetalError::CompileFailed(
            "IRObjectCreateFromDXIL returned null".to_string(),
        ));
    }
    let input = Scoped {
        ptr: input_raw,
        drop_fn: api.obj_destroy,
    };

    // SAFETY: `input.ptr` is a valid DXIL `IRObject`.
    let stage = unsafe { (api.get_stage)(input.ptr) };
    let mut err_raw: *mut IRError = null_mut();
    // SAFETY: `compiler.ptr`, `entry_c`, `input.ptr`, and `&mut err_raw` are valid pointers.
    let air_raw = unsafe {
        (api.compile_and_link)(compiler.ptr, entry_c.as_ptr(), input.ptr, &mut err_raw)
    };
    let _err_guard = Scoped {
        ptr: err_raw,
        drop_fn: api.err_destroy,
    };
    if air_raw.is_null() {
        // SAFETY: If non-null, `err_raw` is a valid `IRError` managed by `_err_guard`.
        let code = unsafe { err_raw.as_ref().map_or(0, |e| (api.err_code)(e)) };
        return Err(MetalError::CompileFailed(format!(
            "IRCompilerAllocCompileAndLink failed (IRError code {code})"
        )));
    }
    let air = Scoped {
        ptr: air_raw,
        drop_fn: api.obj_destroy,
    };

    // SAFETY: `IRMetalLibBinaryCreate` returns an empty container freed by `metallib_destroy`.
    let bin_raw = unsafe { (api.metallib_create)() };
    if bin_raw.is_null() {
        return Err(MetalError::CompileFailed(
            "IRMetalLibBinaryCreate returned null".to_string(),
        ));
    }
    let bin = Scoped {
        ptr: bin_raw,
        drop_fn: api.metallib_destroy,
    };

    // SAFETY: `air.ptr` and `bin.ptr` are valid handles from the compiler above.
    let ok = unsafe { (api.get_metallib)(air.ptr, stage, bin.ptr) };
    if !ok {
        return Err(MetalError::CompileFailed(format!(
            "IRObjectGetMetalLibBinary failed for stage {stage}"
        )));
    }

    Ok(CompiledBinary {
        stage,
        bin,
        _air: air,
        size_fn: api.metallib_size,
        data_fn: api.metallib_data,
        _api: std::marker::PhantomData,
    })
}

/// Converts DXIL bytes into a compiled `MTLLibrary` using `libmetalirconverter.dylib`.
pub(crate) fn compile_dxil_to_library(
    device: &ObjcObject,
    dxil: &[u8],
    entry: &str,
) -> Result<(ObjcObject, VmDxilCompileOutput), MetalError> {
    validate_dxil(dxil)?;
    let entry_c = c_entry(entry)?;
    let api = open_api()?;
    let compiled = compile_to_binary(&api, dxil, &entry_c)?;
    let stage = compiled.stage;
    let metallib_bytes = compiled.byte_len();
    let dispatch_data = compiled.dispatch_data()?;

    let _pool = AutoreleasePool::new();
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a valid `MTLDevice` and `dispatch_data` is borrowed
    // from `compiled`, which stays alive for this call.
    let lib_raw = unsafe {
        objc::msg_send_id_err(
            device.as_raw(),
            objc::sel(c"newLibraryWithData:error:"),
            dispatch_data,
            &mut ns_err,
        )
    };
    let lib = ObjcObject::from_retained(lib_raw).ok_or_else(|| {
        // SAFETY: `ns_err` was populated by Metal's `newLibraryWithData:error:` on failure.
        let msg = unsafe { objc::nserror_message(ns_err) };
        MetalError::CompileFailed(msg)
    })?;

    Ok((
        lib,
        VmDxilCompileOutput {
            stage: u32::try_from(stage).unwrap_or(0),
            metallib_bytes,
        },
    ))
}

/// Converts DXIL to serialized metallib bytes without opening a `MTLDevice`.
///
/// `converter_path` is the explicitly configured `libmetalirconverter.dylib`
/// (the x86_64 library shipped under `D3DMetal.framework/.../Resources/`).
/// `entry` is the entry point named in the DXIL module. The root signature must
/// already be embedded in the DXIL; this path never overrides it.
///
/// This is a deterministic, non-GPU operation suitable for a path-in/path-out
/// CLI built for `x86_64-apple-darwin`. It accepts at most
/// [`MAX_DXIL_INPUT_BYTES`] of input and produces at most
/// [`MAX_METALLIB_OUTPUT_BYTES`] of output; entry names are limited to
/// [`MAX_ENTRY_POINT_BYTES`] and must not contain NUL.
pub fn convert_dxil_to_metallib(
    dxil: &[u8],
    entry: &str,
    converter_path: &Path,
) -> Result<MetallibOutput, MetalError> {
    validate_dxil(dxil)?;
    let entry_c = c_entry(entry)?;
    let api = open_api_from_path(converter_path)?;
    let compiled = compile_to_binary(&api, dxil, &entry_c)?;
    let bytes = compiled.serialized_bytes()?;
    Ok(MetallibOutput {
        bytes,
        stage: u32::try_from(compiled.stage).unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_oversized_dxil_before_loading_converter() {
        let missing = Path::new("/definitely/not/a/converter.dylib");
        assert!(matches!(
            convert_dxil_to_metallib(&[], "main", missing),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            convert_dxil_to_metallib(&vec![0; MAX_DXIL_INPUT_BYTES + 1], "main", missing),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn rejects_invalid_entry_names_before_loading_converter() {
        let missing = Path::new("/definitely/not/a/converter.dylib");
        assert!(matches!(
            convert_dxil_to_metallib(&[0x42], "", missing),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            convert_dxil_to_metallib(&[0x42], "a\0b", missing),
            Err(MetalError::InvalidArg(_))
        ));
        let long = "a".repeat(MAX_ENTRY_POINT_BYTES + 1);
        assert!(matches!(
            convert_dxil_to_metallib(&[0x42], &long, missing),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn missing_converter_path_is_not_supported() {
        let missing = Path::new("/definitely/not/a/converter.dylib");
        assert!(matches!(
            convert_dxil_to_metallib(&[0x42], "main", missing),
            Err(MetalError::NotSupported(_))
        ));
    }
}
