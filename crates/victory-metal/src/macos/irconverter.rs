//! Dynamic loader and RAII wrapper around Apple's `libmetalirconverter.dylib`.
//!
//! Note: Apple currently ships `libmetalirconverter.dylib` inside
//! `D3DMetal.framework/Versions/A/Resources/` as a **Mach-O `x86_64` binary**,
//! matching Rosetta 2's `x86_64` address space used by CrossOver Wine and by the
//! future native Mach-O host. Calling from an `arm64` test binary returns
//! `MetalError::NotSupported` cleanly.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem;
use std::path::PathBuf;
use std::ptr::{NonNull, null_mut};

use crate::abi::VmDxilCompileOutput;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};
use crate::transport::MetalError;

const RTLD_NOW: c_int = 0x2;
const IR_BYTECODE_OWNERSHIP_COPY: c_int = 1;

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

/// Converts DXIL bytes into a compiled `MTLLibrary` using `libmetalirconverter.dylib`.
pub(crate) fn compile_dxil_to_library(
    device: &ObjcObject,
    dxil: &[u8],
    entry: &str,
) -> Result<(ObjcObject, VmDxilCompileOutput), MetalError> {
    if dxil.is_empty() {
        return Err(MetalError::InvalidArg("DXIL bytecode slice is empty"));
    }
    let entry_c =
        CString::new(entry).map_err(|_| MetalError::InvalidArg("entry point contains NUL"))?;
    let api = open_api()?;

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

    // SAFETY: `dxil` slice is valid for `dxil.len()` bytes; ownership `1` instructs IRConverter to copy.
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
    // SAFETY: `compiler.ptr`, `input.ptr`, and `&mut err_raw` are valid pointers.
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

    // SAFETY: `bin.ptr` holds a valid MetalLib payload until `bin` drops at end of scope.
    let metallib_bytes = unsafe { (api.metallib_size)(bin.ptr) };
    // SAFETY: `dispatch_data_t` stays valid while `bin` is alive for `newLibraryWithData:error:`.
    let dispatch_data = unsafe { (api.metallib_data)(bin.ptr) };
    if dispatch_data.is_null() || metallib_bytes == 0 {
        return Err(MetalError::CompileFailed(
            "IRMetalLibGetBytecodeData returned empty bytecode".to_string(),
        ));
    }

    let _pool = AutoreleasePool::new();
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a valid `MTLDevice` and `dispatch_data` is valid while `bin` is alive.
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
