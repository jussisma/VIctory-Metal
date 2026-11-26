//! Stable C FFI exports wrapping [`DirectMetalTransport`].
//!
//! Phase 1 Wine unixlib bridges (`__wine_unix_call`) and C tools can call
//! these entry points dynamically from `libvictory_metal.dylib`, while Phase 2
//! Rust callers can invoke [`DirectMetalTransport`] directly without Wine.

use crate::abi::{VmBufferId, VmDeviceInfo, VmStatus};
use crate::macos::device::DirectMetalTransport;
use crate::transport::MetalTransport;

/// Opaque engine handle returned to C callers by [`vm_engine_create`].
pub struct VmEngine {
    inner: DirectMetalTransport,
}

/// Creates a [`VmEngine`] bound to the system default Metal GPU.
///
/// # Safety
/// `out_engine` must be a non-null, properly aligned writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vm_engine_create(out_engine: *mut *mut VmEngine) -> VmStatus {
    if out_engine.is_null() {
        return VmStatus::InvalidArg;
    }
    // SAFETY: Caller guarantees `out_engine` is valid and writable.
    unsafe { *out_engine = std::ptr::null_mut() };
    match DirectMetalTransport::new() {
        Ok(inner) => {
            let boxed = Box::new(VmEngine { inner });
            // SAFETY: `out_engine` was validated non-null above.
            unsafe { *out_engine = Box::into_raw(boxed) };
            VmStatus::Ok
        }
        Err(e) => e.status(),
    }
}

/// Destroys an engine handle created by [`vm_engine_create`]. Passing null is a no-op.
///
/// # Safety
/// `engine` must be null or a live pointer previously returned by [`vm_engine_create`]
/// and not yet destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vm_engine_destroy(engine: *mut VmEngine) {
    if !engine.is_null() {
        // SAFETY: Caller transfers ownership back from `Box::into_raw`.
        unsafe { drop(Box::from_raw(engine)) };
    }
}

/// Queries Metal device capabilities and zero-copy page alignment.
///
/// # Safety
/// `engine` must be a valid handle from [`vm_engine_create`]; `out_info` must be
/// non-null and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vm_device_info(
    engine: *const VmEngine,
    out_info: *mut VmDeviceInfo,
) -> VmStatus {
    if engine.is_null() || out_info.is_null() {
        return VmStatus::InvalidArg;
    }
    // SAFETY: Caller guarantees `engine` points to a live `VmEngine`.
    let eng = unsafe { &*engine };
    match eng.inner.device_info() {
        Ok(info) => {
            // SAFETY: Caller guarantees `out_info` is valid and writable.
            unsafe { *out_info = info };
            VmStatus::Ok
        }
        Err(e) => e.status(),
    }
}

/// Wraps page-aligned host memory with `newBufferWithBytesNoCopy`.
///
/// # Safety
/// `engine` must be valid, `out_id` must be non-null/writable, and `[ptr, ptr + len)`
/// must remain valid until [`vm_unregister_buffer`] completes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vm_register_zerocopy(
    engine: *const VmEngine,
    ptr: *mut u8,
    len: usize,
    out_id: *mut VmBufferId,
) -> VmStatus {
    if engine.is_null() || out_id.is_null() {
        return VmStatus::InvalidArg;
    }
    // SAFETY: Caller guarantees `engine` points to a live `VmEngine`.
    let eng = unsafe { &*engine };
    // SAFETY: Caller guarantees `[ptr, ptr + len)` outlives GPU usage up to `vm_unregister_buffer`.
    match unsafe { eng.inner.register_zerocopy(ptr, len) } {
        Ok(id) => {
            // SAFETY: Caller guarantees `out_id` is non-null and writable.
            unsafe { *out_id = id };
            VmStatus::Ok
        }
        Err(e) => e.status(),
    }
}

/// Releases a previously registered zero-copy buffer wrapper (does not free host memory).
///
/// # Safety
/// `engine` must be valid and live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vm_unregister_buffer(
    engine: *const VmEngine,
    id: VmBufferId,
) -> VmStatus {
    if engine.is_null() {
        return VmStatus::InvalidArg;
    }
    // SAFETY: Caller guarantees `engine` points to a live `VmEngine`.
    let eng = unsafe { &*engine };
    match eng.inner.unregister_buffer(id) {
        Ok(()) => VmStatus::Ok,
        Err(e) => e.status(),
    }
}
