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
use std::fs;
use std::mem;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::{NonNull, null_mut};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::abi::VmDxilCompileOutput;
use crate::macos::cache::{
    BindingMetadata, Cached, DiskCache, MAX_CACHED_ROOT_SIGNATURE_BYTES, ReflectionPayload,
    TranslationKey, TranslationOptions, blake3_bytes,
};
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

/// Version string mixed into the translation cache key.
///
/// It covers this crate's conversion protocol rather than the (unversioned)
/// `libmetalirconverter.dylib` file itself. Deployments that replace the dylib
/// must set `VICTORY_IRCONVERTER_VERSION` (or pass an explicit
/// [`ProcessWorker::with_version`]) so cached output is invalidated.
pub const DEFAULT_CONVERTER_VERSION: &str =
    concat!("victory-metal/", env!("CARGO_PKG_VERSION"), "/irconverter-protocol-1");

/// One translation request handed to a [`TranslationBackend`].
///
/// The root signature, options and binding metadata travel with the DXIL so the
/// Rosetta worker and the cache see exactly the same inputs the caller saw.
#[derive(Debug)]
pub struct TranslationRequest<'a> {
    /// Raw DXIL bytes.
    pub dxil: &'a [u8],
    /// Entry point named in the DXIL module.
    pub entry: &'a str,
    /// Raw root signature bytes when the caller has them. The conversion still
    /// consumes the root signature embedded in the DXIL; this value selects the
    /// cache key and is carried into the reflection payload.
    pub root_signature: Option<&'a [u8]>,
    /// Translation options that can change output.
    pub options: &'a TranslationOptions,
    /// Verified binding metadata for the binder.
    pub bindings: &'a [BindingMetadata],
    /// Converter/protocol version mixed into the key.
    pub converter_version: &'a str,
}

/// Raw result produced by a translation backend before cache validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerOutput {
    /// Serialized metallib bytes.
    pub metallib: Vec<u8>,
    /// Reflection metadata describing the result.
    pub reflection: ReflectionPayload,
}

/// A cache-aware translation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationOutcome {
    /// Serialized metallib bytes.
    pub metallib: Vec<u8>,
    /// Reflection metadata the binder consumes.
    pub reflection: ReflectionPayload,
    /// True when the result came from the persistent disk cache.
    pub cache_hit: bool,
}

/// Pluggable DXIL-to-metallib translator.
///
/// Production uses [`ProcessWorker`] (x86_64 Rosetta worker) or
/// [`InProcessConverter`] (legacy same-process `dlopen`). Tests inject counting
/// or failing backends without a converter or GPU.
pub trait TranslationBackend {
    /// Converter/protocol version mixed into the cache key.
    fn converter_version(&self) -> &str;

    /// Translates one request. Implementations must not cache by themselves.
    fn translate(&self, request: &TranslationRequest<'_>) -> Result<WorkerOutput, MetalError>;
}

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
enum IRShaderReflection {}

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
/// Fills a caller-created reflection object for one shader stage.
///
/// The stage argument is the value reported by
/// `IRObjectGetMetalIRShaderStage` (`8` for compute). The reflection is only
/// populated for the linked AIR object, not the input DXIL object.
type FnIRObjectGetReflection =
    unsafe extern "C" fn(*const IRObject, c_int, *mut IRShaderReflection);
type FnIRShaderReflectionCreate = unsafe extern "C" fn() -> *mut IRShaderReflection;
type FnIRShaderReflectionDestroy = unsafe extern "C" fn(*mut IRShaderReflection);
type FnIRShaderReflectionCopyJSONString =
    unsafe extern "C" fn(*const IRShaderReflection) -> *const c_char;
type FnIRShaderReflectionFreeString = unsafe extern "C" fn(*const c_char);

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
    reflection_get: Option<FnIRObjectGetReflection>,
    reflection_create: Option<FnIRShaderReflectionCreate>,
    reflection_destroy: Option<FnIRShaderReflectionDestroy>,
    reflection_json: Option<FnIRShaderReflectionCopyJSONString>,
    reflection_free: Option<FnIRShaderReflectionFreeString>,
}

unsafe fn load_sym_opt<T: Copy>(handle: *mut c_void, name: &CStr) -> Option<T> {
    if mem::size_of::<T>() != mem::size_of::<*mut c_void>() {
        return None;
    }
    // SAFETY: `handle` is an open `dlopen` handle and `name` is NUL-terminated.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    let ptr = NonNull::new(sym)?;
    // SAFETY: Caller specifies the exact C symbol signature matching Apple's
    // IRConverter ABI; this is the optional-symbol form of `load_sym`.
    Some(unsafe { mem::transmute_copy::<*mut c_void, T>(&ptr.as_ptr()) })
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
            reflection_get: load_sym_opt(h, c"IRObjectGetReflection"),
            reflection_create: load_sym_opt(h, c"IRShaderReflectionCreate"),
            reflection_destroy: load_sym_opt(h, c"IRShaderReflectionDestroy"),
            reflection_json: load_sym_opt(h, c"IRShaderReflectionCopyJSONString"),
            reflection_free: load_sym_opt(h, c"IRShaderReflectionFreeString"),
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
    /// Converter reflection JSON captured from the linked AIR object when the
    /// caller asked for it. `None` for the plain bytecode path.
    reflection_json: Option<String>,
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
    with_reflection: bool,
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

    let reflection_json = if with_reflection {
        Some(copy_reflection_json(api, air.ptr, stage)?)
    } else {
        None
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
        reflection_json,
        size_fn: api.metallib_size,
        data_fn: api.metallib_data,
        _api: std::marker::PhantomData,
    })
}

/// Captures Apple's reflection JSON for a linked AIR object.
///
/// `IRObjectGetReflection` only fills the reflection for the *linked* AIR
/// object, and it takes the stage reported by `IRObjectGetMetalIRShaderStage`.
/// The JSON is copied into an owned `String` before the reflection object is
/// destroyed.
fn copy_reflection_json(
    api: &Api,
    air: *const IRObject,
    stage: i32,
) -> Result<String, MetalError> {
    let (
        Some(get_reflection),
        Some(create),
        Some(destroy),
        Some(copy_json),
        Some(free_string),
    ) = (
        api.reflection_get,
        api.reflection_create,
        api.reflection_destroy,
        api.reflection_json,
        api.reflection_free,
    )
    else {
        return Err(MetalError::CompileFailed(
            "IRConverter build does not expose the reflection symbols".to_string(),
        ));
    };

    // SAFETY: `create` is the paired constructor for `IRShaderReflection`.
    let reflection = unsafe { create() };
    if reflection.is_null() {
        return Err(MetalError::CompileFailed(
            "IRShaderReflectionCreate returned null".to_string(),
        ));
    }
    struct ReflectionGuard {
        ptr: *mut IRShaderReflection,
        destroy: FnIRShaderReflectionDestroy,
    }
    impl Drop for ReflectionGuard {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` came from `IRShaderReflectionCreate`.
            unsafe { (self.destroy)(self.ptr) };
        }
    }
    let _guard = ReflectionGuard {
        ptr: reflection,
        destroy,
    };

    // SAFETY: `air` is a live linked AIR object and `reflection` is a live,
    // caller-created reflection for the requested stage.
    unsafe { get_reflection(air, stage, reflection) };

    // SAFETY: `reflection` is a live reflection object; the returned string is
    // owned by the caller and released with `IRShaderReflectionFreeString`.
    let raw = unsafe { copy_json(reflection) };
    if raw.is_null() {
        return Err(MetalError::CompileFailed(
            "IRShaderReflectionCopyJSONString returned null".to_string(),
        ));
    }
    // SAFETY: `raw` is a NUL-terminated string owned by this call.
    let text = unsafe { CStr::from_ptr(raw) }
        .to_str()
        .map_err(|_| {
            MetalError::CompileFailed("IRConverter reflection JSON is not UTF-8".to_string())
        })?
        .to_string();
    // SAFETY: `raw` was handed to the caller by `IRShaderReflectionCopyJSONString`.
    unsafe { free_string(raw) };
    Ok(text)
}

/// Converts DXIL bytes into a compiled `MTLLibrary` using the configured
/// translation backend (Rosetta worker, or the legacy in-process library).
pub(crate) fn compile_dxil_to_library(
    device: &ObjcObject,
    dxil: &[u8],
    entry: &str,
) -> Result<(ObjcObject, VmDxilCompileOutput), MetalError> {
    compile_dxil_to_library_with_layout(
        device,
        dxil,
        entry,
        None,
        &TranslationOptions::default(),
        &[],
    )
}

/// Translation entry point with an explicit root layout, options and binding
/// metadata.
pub(crate) fn compile_dxil_to_library_with_layout(
    device: &ObjcObject,
    dxil: &[u8],
    entry: &str,
    root_signature: Option<&[u8]>,
    options: &TranslationOptions,
    bindings: &[BindingMetadata],
) -> Result<(ObjcObject, VmDxilCompileOutput), MetalError> {
    validate_dxil(dxil)?;
    c_entry(entry)?;
    let backend: Box<dyn TranslationBackend> = match ProcessWorker::from_env() {
        Some(worker) => Box::new(worker),
        None => Box::new(InProcessConverter),
    };
    let outcome = translate_dxil_with(
        global_cache(),
        backend.as_ref(),
        dxil,
        entry,
        0,
        root_signature,
        options,
        bindings,
    )?;

    let _pool = AutoreleasePool::new();
    let dispatch_data = objc::dispatch_data_copy(&outcome.metallib).ok_or_else(|| {
        MetalError::InternalError("dispatch_data_create returned null".to_string())
    })?;
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a valid `MTLDevice`, `dispatch_data` is a retained
    // `dispatch_data_t`, and `ns_err` receives an autoreleased `NSError *`.
    let lib_raw = unsafe {
        objc::msg_send_id_err(
            device.as_raw(),
            objc::sel(c"newLibraryWithData:error:"),
            dispatch_data.as_raw(),
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
            stage: outcome.reflection.stage,
            metallib_bytes: outcome.metallib.len(),
        },
    ))
}

/// Cache-aware translation.
///
/// A valid cache entry is returned without calling `backend` at all. On a miss
/// the backend runs, its output hash and metadata are verified against the key,
/// and only then is the result stored. A backend failure or a verification
/// failure is never cached.
pub fn translate_dxil_with(
    cache: Option<&DiskCache>,
    backend: &dyn TranslationBackend,
    dxil: &[u8],
    entry: &str,
    stage: u32,
    root_signature: Option<&[u8]>,
    options: &TranslationOptions,
    bindings: &[BindingMetadata],
) -> Result<TranslationOutcome, MetalError> {
    validate_dxil(dxil)?;
    c_entry(entry)?;
    if let Some(root) = root_signature {
        if root.len() > MAX_CACHED_ROOT_SIGNATURE_BYTES {
            return Err(MetalError::InvalidArg(
                "root signature exceeds the 1 MiB cache limit",
            ));
        }
    }

    let version = backend.converter_version();
    let key = TranslationKey::new(dxil, root_signature, entry, stage, version, options);

    if let Some(cache) = cache {
        if let Some(cached) = cache.get(&key) {
            if cached_matches_request(&cached.reflection, &key, entry, stage, options, bindings) {
                return Ok(TranslationOutcome {
                    metallib: cached.metallib,
                    reflection: cached.reflection,
                    cache_hit: true,
                });
            }
        }
    }

    let request = TranslationRequest {
        dxil,
        entry,
        root_signature,
        options,
        bindings,
        converter_version: version,
    };
    let output = backend.translate(&request)?;
    verify_output(&output, &key, entry, stage, options, bindings)?;

    if let Some(cache) = cache {
        let cached = Cached {
            metallib: output.metallib.clone(),
            reflection: output.reflection.clone(),
        };
        if let Err(err) = cache.put(&key, &cached) {
            tracing::warn!(
                error = %err,
                dir = %cache.dir().display(),
                "failed to store translated shader in the disk cache"
            );
        }
    }

    Ok(TranslationOutcome {
        metallib: output.metallib,
        reflection: output.reflection,
        cache_hit: false,
    })
}

fn cached_matches_request(
    reflection: &ReflectionPayload,
    key: &TranslationKey,
    entry: &str,
    stage: u32,
    options: &TranslationOptions,
    bindings: &[BindingMetadata],
) -> bool {
    reflection.entry == entry
        && (stage == 0 || reflection.stage == stage)
        && reflection.dxil_hash == key.dxil_hash
        && reflection.root_layout_hash == key.root_layout_hash
        && reflection.converter_version == key.converter_version
        && &reflection.options == options
        && reflection.bindings == bindings
}

fn verify_output(
    output: &WorkerOutput,
    key: &TranslationKey,
    entry: &str,
    stage: u32,
    options: &TranslationOptions,
    bindings: &[BindingMetadata],
) -> Result<(), MetalError> {
    if output.metallib.is_empty() {
        return Err(MetalError::CompileFailed(
            "translation backend produced an empty metallib".to_string(),
        ));
    }
    if output.metallib.len() > MAX_METALLIB_OUTPUT_BYTES {
        return Err(MetalError::CompileFailed(format!(
            "translation backend produced {} bytes; limit is {}",
            output.metallib.len(),
            MAX_METALLIB_OUTPUT_BYTES
        )));
    }
    let reflection = &output.reflection;
    if reflection.stage == 0 {
        return Err(MetalError::CompileFailed(
            "translation backend reported stage 0".to_string(),
        ));
    }
    if stage != 0 && reflection.stage != stage {
        return Err(MetalError::CompileFailed(format!(
            "translation backend reported stage {}; request declared stage {stage}",
            reflection.stage
        )));
    }
    if reflection.entry != entry {
        return Err(MetalError::CompileFailed(
            "translation backend reflection entry does not match the request".to_string(),
        ));
    }
    if reflection.dxil_hash != key.dxil_hash
        || reflection.root_layout_hash != key.root_layout_hash
    {
        return Err(MetalError::CompileFailed(
            "translation backend reflection hashes do not match the request".to_string(),
        ));
    }
    if reflection.metallib_bytes != output.metallib.len() as u64
        || reflection.metallib_hash != blake3_bytes(&output.metallib)
    {
        return Err(MetalError::CompileFailed(
            "translation backend metallib hash or size does not match the output".to_string(),
        ));
    }
    if reflection.converter_version != key.converter_version {
        return Err(MetalError::CompileFailed(
            "translation backend converter version does not match the request".to_string(),
        ));
    }
    if &reflection.options != options {
        return Err(MetalError::CompileFailed(
            "translation backend reflection options do not match the request".to_string(),
        ));
    }
    if reflection.bindings != bindings {
        return Err(MetalError::CompileFailed(
            "translation backend reflection bindings do not match the request".to_string(),
        ));
    }
    Ok(())
}

/// Opens the process-wide disk cache configured by
/// `VICTORY_METAL_SHADER_CACHE`, or `$HOME/Library/Caches/victory-metal/shader-cache`.
///
/// Failure to create the cache directory disables caching (translation still
/// runs) rather than failing compilation.
fn global_cache() -> Option<&'static DiskCache> {
    static CACHE: OnceLock<Option<DiskCache>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let dir = std::env::var_os("VICTORY_METAL_SHADER_CACHE")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        PathBuf::from(home)
                            .join("Library/Caches/victory-metal/shader-cache")
                    })
                })
                .unwrap_or_else(|| std::env::temp_dir().join("victory-metal-shader-cache"));
            match DiskCache::open(&dir) {
                Ok(cache) => Some(cache),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        dir = %dir.display(),
                        "shader disk cache disabled: cannot create directory"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Legacy same-process translator using `dlopen` on
/// `libmetalirconverter.dylib`.
///
/// Only works in an x86_64 process; on arm64 [`open_api`] returns
/// [`MetalError::NotSupported`] and [`ProcessWorker`] must be configured.
pub struct InProcessConverter;

impl TranslationBackend for InProcessConverter {
    fn converter_version(&self) -> &str {
        DEFAULT_CONVERTER_VERSION
    }

    fn translate(&self, request: &TranslationRequest<'_>) -> Result<WorkerOutput, MetalError> {
        let entry_c = c_entry(request.entry)?;
        let api = open_api()?;
        let compiled = compile_to_binary(&api, request.dxil, &entry_c, false)?;
        let metallib = compiled.serialized_bytes()?;
        let stage = u32::try_from(compiled.stage).unwrap_or(0);
        let reflection = ReflectionPayload::from_translation(
            &metallib,
            stage,
            request.entry,
            request.dxil,
            request.root_signature,
            request.converter_version,
            request.options,
            request.bindings,
        );
        Ok(WorkerOutput {
            metallib,
            reflection,
        })
    }
}

/// x86_64 Rosetta worker process that performs DXIL conversion and writes the
/// metallib plus a reflection payload.
#[derive(Debug, Clone)]
pub struct ProcessWorker {
    program: PathBuf,
    converter: PathBuf,
    converter_version: String,
}

impl ProcessWorker {
    /// Builds a worker from an explicit program and converter path.
    pub fn new(program: impl Into<PathBuf>, converter: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            converter: converter.into(),
            converter_version: DEFAULT_CONVERTER_VERSION.to_string(),
        }
    }

    /// Overrides the converter version mixed into the cache key.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.converter_version = version.into();
        self
    }

    /// Resolves the worker from `VICTORY_IRCONVERTER_WORKER` or the
    /// `victory-irconverter` sibling of the current executable.
    ///
    /// Returns `None` when no worker or no converter library is configured, so
    /// the caller can fall back to the legacy in-process path.
    pub fn from_env() -> Option<Self> {
        let program = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                let exe = std::env::current_exe().ok()?;
                let sibling = exe.parent()?.join("victory-irconverter");
                sibling.is_file().then_some(sibling)
            })?;

        let converter = std::env::var_os("VICTORY_METAL_IRCONVERTER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                std::env::var_os("D3DMETAL_RES")
                    .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                    .filter(|path| path.is_file())
            })
            .or_else(|| candidate_paths().into_iter().find(|path| path.is_file()))?;

        let version = std::env::var("VICTORY_IRCONVERTER_VERSION")
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_CONVERTER_VERSION.to_string());

        Some(Self {
            program,
            converter,
            converter_version: version,
        })
    }

    /// Worker program path.
    pub fn program(&self) -> &Path {
        &self.program
    }

    /// Converter library path passed to the worker.
    pub fn converter(&self) -> &Path {
        &self.converter
    }

    fn run(&self, request: &TranslationRequest<'_>) -> Result<WorkerOutput, MetalError> {
        let scratch = ScratchDir::new()?;
        let input = scratch.path().join("input.dxil");
        let output = scratch.path().join("output.metallib");
        let reflection_path = scratch.path().join("reflection.bin");
        let root_path = scratch.path().join("root-signature.bin");

        fs::write(&input, request.dxil).map_err(|err| {
            MetalError::CompileFailed(format!("failed to write worker DXIL input: {err}"))
        })?;
        if let Some(root) = request.root_signature {
            fs::write(&root_path, root).map_err(|err| {
                MetalError::CompileFailed(format!("failed to write worker root signature: {err}"))
            })?;
        }

        let mut command = Command::new(&self.program);
        command
            .arg("--converter")
            .arg(&self.converter)
            .arg("--entry")
            .arg(request.entry)
            .arg("--converter-version")
            .arg(request.converter_version)
            .arg("--reflection")
            .arg(&reflection_path)
            .arg("--target")
            .arg(&request.options.target)
            .arg("--min-os")
            .arg(&request.options.min_os);
        if request.root_signature.is_some() {
            command.arg("--root-signature").arg(&root_path);
        }
        for flag in &request.options.flags {
            command.arg("--flag").arg(flag);
        }
        for binding in request.bindings {
            command.arg("--binding").arg(format!(
                "{}:{}:{}:{}",
                binding.kind.as_str(),
                binding.register,
                binding.metal_index,
                binding.byte_offset
            ));
        }
        command.arg(&input).arg(&output);

        let result = command.output().map_err(|err| {
            MetalError::CompileFailed(format!(
                "failed to spawn IRConverter worker {}: {err}",
                self.program.display()
            ))
        })?;
        if !result.status.success() {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(MetalError::CompileFailed(format!(
                "IRConverter worker failed with {}: {}{}",
                result.status,
                stdout.trim(),
                stderr.trim()
            )));
        }

        let metallib = read_limited(
            &output,
            MAX_METALLIB_OUTPUT_BYTES,
            "worker metallib output",
        )?;
        let reflection_bytes = fs::read(&reflection_path).map_err(|err| {
            MetalError::CompileFailed(format!(
                "IRConverter worker did not write a reflection payload: {err}"
            ))
        })?;
        let reflection = ReflectionPayload::decode(&reflection_bytes).ok_or_else(|| {
            MetalError::CompileFailed(
                "IRConverter worker wrote a malformed reflection payload".to_string(),
            )
        })?;
        Ok(WorkerOutput {
            metallib,
            reflection,
        })
    }
}

impl TranslationBackend for ProcessWorker {
    fn converter_version(&self) -> &str {
        &self.converter_version
    }

    fn translate(&self, request: &TranslationRequest<'_>) -> Result<WorkerOutput, MetalError> {
        self.run(request)
    }
}

fn read_limited(path: &Path, limit: usize, label: &str) -> Result<Vec<u8>, MetalError> {
    let metadata = fs::metadata(path).map_err(|err| {
        MetalError::CompileFailed(format!("failed to stat {label} {}: {err}", path.display()))
    })?;
    if !metadata.is_file() {
        return Err(MetalError::CompileFailed(format!(
            "{label} is not a regular file: {}",
            path.display()
        )));
    }
    let reported = metadata.len();
    if reported == 0 {
        return Err(MetalError::CompileFailed(format!(
            "{label} is empty: {}",
            path.display()
        )));
    }
    if reported > limit as u64 {
        return Err(MetalError::CompileFailed(format!(
            "{label} is {reported} bytes; limit is {limit}: {}",
            path.display()
        )));
    }
    let bytes = fs::read(path).map_err(|err| {
        MetalError::CompileFailed(format!("failed to read {label} {}: {err}", path.display()))
    })?;
    if bytes.is_empty() || bytes.len() > limit {
        return Err(MetalError::CompileFailed(format!(
            "{label} has invalid size {} (limit {limit}): {}",
            bytes.len(),
            path.display()
        )));
    }
    Ok(bytes)
}

static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self, MetalError> {
        let sequence = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "victory-irconverter-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path).map_err(|err| {
            MetalError::CompileFailed(format!(
                "failed to create IRConverter scratch directory {}: {err}",
                path.display()
            ))
        })?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
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
    let compiled = compile_to_binary(&api, dxil, &entry_c, false)?;
    let bytes = compiled.serialized_bytes()?;
    Ok(MetallibOutput {
        bytes,
        stage: u32::try_from(compiled.stage).unwrap_or(0),
    })
}

/// Converted metallib plus Apple's reflection JSON for the linked program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetallibWithReflection {
    /// Serialized metallib bytes and reported stage.
    pub metallib: MetallibOutput,
    /// `IRShaderReflectionCopyJSONString` output for the linked AIR object.
    ///
    /// The JSON lists the converted entry point, stage, root parameters in
    /// `TopLevelArgumentBuffer` (byte offset, size, register slot/space and the
    /// `Constant`/`CBV`/`SRV`/`UAV`/`Table` type) plus `UsedResources`. It is
    /// the only source for the flattened argument-buffer ABI; see
    /// [`crate::macos::bindings::RootSignatureLayout`].
    pub reflection_json: String,
}

/// Converts DXIL and also returns Apple's shader reflection JSON.
///
/// This uses the same x86_64 `libmetalirconverter.dylib` path as
/// [`convert_dxil_to_metallib`]. Reflection is only populated for the linked
/// AIR object, so the call happens between `IRCompilerAllocCompileAndLink` and
/// bytecode serialization. Builds without the `IRShaderReflection*` symbols
/// return [`MetalError::CompileFailed`] rather than a partial layout.
pub fn convert_dxil_to_metallib_with_reflection(
    dxil: &[u8],
    entry: &str,
    converter_path: &Path,
) -> Result<MetallibWithReflection, MetalError> {
    validate_dxil(dxil)?;
    let entry_c = c_entry(entry)?;
    let api = open_api_from_path(converter_path)?;
    let compiled = compile_to_binary(&api, dxil, &entry_c, true)?;
    let bytes = compiled.serialized_bytes()?;
    let stage = u32::try_from(compiled.stage).unwrap_or(0);
    let reflection_json = compiled.reflection_json.ok_or_else(|| {
        MetalError::InternalError("reflection JSON was not captured for this build".to_string())
    })?;
    Ok(MetallibWithReflection {
        metallib: MetallibOutput { bytes, stage },
        reflection_json,
    })
}

/// First existing converter library from the documented search order.
///
/// Mirrors [`candidate_paths`] without opening the library, so callers can
/// decide whether reflection-based layout construction is possible before
/// attempting a conversion.
pub fn converter_library_path() -> Option<PathBuf> {
    candidate_paths().into_iter().find(|path| path.is_file())
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

    use crate::macos::cache::{BindingKind, TempDir, blake3_bytes, root_layout_hash};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::atomic::Ordering as AtomicOrdering;

    #[derive(Default)]
    struct CountingBackend {
        calls: AtomicUsize,
        version: String,
        fail_first: AtomicBool,
    }

    impl CountingBackend {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                version: DEFAULT_CONVERTER_VERSION.to_string(),
                fail_first: AtomicBool::new(false),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(AtomicOrdering::SeqCst)
        }
    }

    impl TranslationBackend for CountingBackend {
        fn converter_version(&self) -> &str {
            &self.version
        }

        fn translate(
            &self,
            request: &TranslationRequest<'_>,
        ) -> Result<WorkerOutput, MetalError> {
            let call = self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            if self.fail_first.load(AtomicOrdering::SeqCst) && call == 0 {
                return Err(MetalError::CompileFailed(
                    "synthetic translation failure".to_string(),
                ));
            }
            let metallib = format!(
                "METALLIB entry={} dxil={} root={}",
                request.entry,
                request.dxil.len(),
                request.root_signature.map_or(0, <[u8]>::len)
            )
            .into_bytes();
            let reflection = ReflectionPayload::from_translation(
                &metallib,
                8,
                request.entry,
                request.dxil,
                request.root_signature,
                request.converter_version,
                request.options,
                request.bindings,
            );
            Ok(WorkerOutput {
                metallib,
                reflection,
            })
        }
    }

    fn sample_options() -> TranslationOptions {
        TranslationOptions::default()
    }

    #[test]
    fn same_key_is_one_backend_invocation() {
        let dir = TempDir::new("ir-same-key");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let backend = CountingBackend::new();
        let options = sample_options();

        let first = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-same",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("first translation");
        assert!(!first.cache_hit);
        assert_eq!(backend.calls(), 1);

        let second = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-same",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("cached translation");
        assert!(second.cache_hit);
        assert_eq!(backend.calls(), 1, "cache hit must not invoke the backend");
        assert_eq!(first.metallib, second.metallib);
        assert_eq!(first.reflection, second.reflection);
    }

    #[test]
    fn different_inputs_get_different_keys_and_recompute() {
        let dir = TempDir::new("ir-different");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let backend = CountingBackend::new();
        let options = sample_options();
        let other_options = TranslationOptions {
            target: "apple-macos".to_string(),
            min_os: "15.0".to_string(),
            flags: vec!["-O3".to_string()],
        };

        let calls = [
            (b"dxil-a".as_slice(), "main", Some(b"root-a".as_slice()), &options),
            (b"dxil-b".as_slice(), "main", Some(b"root-a".as_slice()), &options),
            (b"dxil-a".as_slice(), "other", Some(b"root-a".as_slice()), &options),
            (b"dxil-a".as_slice(), "main", Some(b"root-b".as_slice()), &options),
            (b"dxil-a".as_slice(), "main", None, &options),
            (b"dxil-a".as_slice(), "main", Some(b"root-a".as_slice()), &other_options),
        ];
        for (index, (dxil, entry, root, options)) in calls.iter().enumerate() {
            let outcome = translate_dxil_with(
                Some(&cache),
                &backend,
                dxil,
                entry,
                8,
                *root,
                options,
                &[],
            )
            .unwrap_or_else(|err| panic!("translation {index} failed: {err}"));
            assert!(!outcome.cache_hit, "input {index} must not be a cache hit");
            assert_eq!(backend.calls(), index + 1);
        }
    }

    #[test]
    fn corrupt_entry_is_a_miss_and_is_recomputed() {
        let dir = TempDir::new("ir-corrupt");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let backend = CountingBackend::new();
        let options = sample_options();
        let key = TranslationKey::new(b"dxil-corrupt", Some(b"root"), "main", 8, DEFAULT_CONVERTER_VERSION, &options);
        translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-corrupt",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("first translation");
        assert_eq!(backend.calls(), 1);

        let path = cache.path_for(&key);
        let mut bytes = fs::read(&path).expect("read entry");
        bytes.truncate(bytes.len() - 16);
        fs::write(&path, bytes).expect("truncate entry");
        assert!(cache.get(&key).is_none());

        let outcome = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-corrupt",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("recomputed translation");
        assert!(!outcome.cache_hit);
        assert_eq!(backend.calls(), 2);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn failed_translation_is_never_cached() {
        let dir = TempDir::new("ir-failure");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let backend = CountingBackend::new();
        backend.fail_first.store(true, AtomicOrdering::SeqCst);
        let options = sample_options();

        assert!(
            translate_dxil_with(
                Some(&cache),
                &backend,
                b"dxil-fail",
                "main",
                8,
                Some(b"root"),
                &options,
                &[],
            )
            .is_err()
        );
        assert_eq!(backend.calls(), 1);

        let outcome = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-fail",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("second attempt succeeds");
        assert!(!outcome.cache_hit);
        assert_eq!(backend.calls(), 2);

        let cached = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-fail",
            "main",
            8,
            Some(b"root"),
            &options,
            &[],
        )
        .expect("third attempt is a hit");
        assert!(cached.cache_hit);
        assert_eq!(backend.calls(), 2);
    }

    #[test]
    fn process_worker_is_spawned_once_and_output_is_cached() {
        let dir = TempDir::new("ir-process-worker");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let dxil = b"dxil-process-worker".to_vec();
        let root = b"root-process-worker".to_vec();
        let options = TranslationOptions {
            target: "apple-macos".to_string(),
            min_os: "14.0".to_string(),
            flags: vec!["-O2".to_string()],
        };
        let bindings = [BindingMetadata {
            kind: BindingKind::Uav,
            register: 0,
            metal_index: 2,
            byte_offset: 0,
        }];
        let metallib = b"FAKE-PROCESS-METALLIB".to_vec();
        let reflection = ReflectionPayload::from_translation(
            &metallib,
            8,
            "main",
            &dxil,
            Some(&root),
            DEFAULT_CONVERTER_VERSION,
            &options,
            &bindings,
        );

        let metallib_path = dir.path().join("fixture.metallib");
        let reflection_path = dir.path().join("fixture.reflection");
        let converter_path = dir.path().join("fixture.dylib");
        let counter_path = dir.path().join("invocations");
        fs::write(&metallib_path, &metallib).expect("write metallib fixture");
        fs::write(&reflection_path, reflection.encode()).expect("write reflection fixture");
        fs::write(&converter_path, b"not a real dylib").expect("write converter placeholder");

        let script_path = dir.path().join("worker.sh");
        let script = format!(
            "#!/bin/sh\n\
             count={counter}\n\
             if [ -e \"$count\" ]; then\n\
               echo 'worker invoked more than once' >&2\n\
               exit 17\n\
             fi\n\
             : > \"$count\"\n\
             reflection=''\n\
             want=''\n\
             for arg in \"$@\"; do\n\
               if [ \"$want\" = reflection ]; then reflection=\"$arg\"; want=''; continue; fi\n\
               if [ \"$arg\" = --reflection ]; then want=reflection; fi\n\
             done\n\
             output=''\n\
             for arg in \"$@\"; do output=\"$arg\"; done\n\
             cp {reflection_fixture} \"$reflection\"\n\
             cp {metallib_fixture} \"$output\"\n\
             echo 'IRCONVERT_OK entry=\"main\" stage=8'\n",
            counter = shell_quote(&counter_path),
            reflection_fixture = shell_quote(&reflection_path),
            metallib_fixture = shell_quote(&metallib_path),
        );
        fs::write(&script_path, script).expect("write worker script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755))
                .expect("mark worker executable");
        }

        let backend = ProcessWorker::new(&script_path, &converter_path);
        let first = translate_dxil_with(
            Some(&cache),
            &backend,
            &dxil,
            "main",
            8,
            Some(&root),
            &options,
            &bindings,
        )
        .expect("worker translation");
        assert!(!first.cache_hit);
        assert_eq!(first.metallib, metallib);
        assert!(counter_path.is_file(), "worker script must have run");

        let second = translate_dxil_with(
            Some(&cache),
            &backend,
            &dxil,
            "main",
            8,
            Some(&root),
            &options,
            &bindings,
        )
        .expect("cached translation");
        assert!(second.cache_hit);
        assert_eq!(second.reflection.bindings, bindings);
    }

    fn shell_quote(path: &Path) -> String {
        let text = path.to_string_lossy();
        format!("'{}'", text.replace('\'', "'\\''"))
    }

    #[test]
    fn process_worker_rejects_missing_reflection_payload() {
        let dir = TempDir::new("ir-process-missing-reflection");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let converter_path = dir.path().join("fixture.dylib");
        fs::write(&converter_path, b"not a real dylib").expect("write converter placeholder");
        let script_path = dir.path().join("worker.sh");
        fs::write(
            &script_path,
            "#!/bin/sh\noutput=''\nfor arg in \"$@\"; do output=\"$arg\"; done\nprintf 'METALLIB' > \"$output\"\n",
        )
        .expect("write worker script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755))
                .expect("mark worker executable");
        }
        let backend = ProcessWorker::new(&script_path, &converter_path);
        let options = sample_options();
        let error = translate_dxil_with(
            Some(&cache),
            &backend,
            b"dxil-no-reflection",
            "main",
            8,
            None,
            &options,
            &[],
        )
        .expect_err("missing reflection must fail");
        assert!(matches!(error, MetalError::CompileFailed(_)));
    }

    #[test]
    fn cache_hit_skips_worker_and_reports_reflection_hashes() {
        let dir = TempDir::new("ir-hit-hashes");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let backend = CountingBackend::new();
        let options = sample_options();
        let dxil = b"dxil-hash-check";
        let outcome = translate_dxil_with(
            Some(&cache),
            &backend,
            dxil,
            "main",
            8,
            None,
            &options,
            &[],
        )
        .expect("translation");
        assert_eq!(outcome.reflection.dxil_hash, blake3_bytes(dxil));
        assert_eq!(outcome.reflection.root_layout_hash, root_layout_hash(None));
        assert_eq!(
            outcome.reflection.metallib_hash,
            blake3_bytes(&outcome.metallib)
        );
        assert_eq!(outcome.reflection.metallib_bytes, outcome.metallib.len() as u64);
    }
}
