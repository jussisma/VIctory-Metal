//! Minimal Objective-C runtime and Apple Metal FFI helpers.
//!
//! Uses direct `libobjc` message dispatch with typed function pointers on both
//! `aarch64` and `x86_64` (including `objc_msgSend_stret` for struct returns on
//! `x86_64`), keeping zero external Objective-C crate dependencies.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem;
use std::ptr::{NonNull, null, null_mut};

pub(crate) type Id = *mut c_void;
pub(crate) type Sel = *mut c_void;
pub(crate) type Class = *mut c_void;

/// Apple `MTLSize` (`NSUInteger width, height, depth`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    #[cfg(target_arch = "x86_64")]
    fn objc_msgSend_stret();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "System")]
unsafe extern "C" {
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> Id;
}

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

/// Returns macOS host `getpagesize()`.
pub(crate) fn host_page_size() -> usize {
    // SAFETY: POSIX `getpagesize()` has no preconditions and returns a positive page size.
    let ps = unsafe { getpagesize() };
    usize::try_from(ps).unwrap_or(4096).max(4096)
}

/// Acquires the system default `MTLDevice` (retained +1).
pub(crate) fn create_system_default_device() -> Option<ObjcObject> {
    // SAFETY: `MTLCreateSystemDefaultDevice` is safe to call on macOS and returns
    // either null or a retained `id<MTLDevice>`.
    let raw = unsafe { MTLCreateSystemDefaultDevice() };
    ObjcObject::from_retained(raw)
}

pub(crate) fn class(name: &CStr) -> Class {
    // SAFETY: `name` is a valid NUL-terminated C string.
    unsafe { objc_getClass(name.as_ptr()) }
}

pub(crate) fn sel(name: &CStr) -> Sel {
    // SAFETY: `name` is a valid NUL-terminated C string.
    unsafe { sel_registerName(name.as_ptr()) }
}

/// RAII wrapper around an Objective-C `@autoreleasepool`.
pub(crate) struct AutoreleasePool {
    token: *mut c_void,
}

impl AutoreleasePool {
    pub(crate) fn new() -> Self {
        // SAFETY: Pushing an autorelease pool is always valid on an OS thread.
        let token = unsafe { objc_autoreleasePoolPush() };
        Self { token }
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        if !self.token.is_null() {
            // SAFETY: `token` came from the matching `objc_autoreleasePoolPush` on this thread.
            unsafe { objc_autoreleasePoolPop(self.token) };
        }
    }
}

/// Owning RAII handle for a retained (+1) Objective-C object.
pub(crate) struct ObjcObject {
    ptr: NonNull<c_void>,
}

// SAFETY: `MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLLibrary`, and
// `MTLComputePipelineState` are documented by Apple Metal as thread-safe to
// share across threads (`Send` + `Sync` handles; synchronized command buffers
// handle ordering). `NSString` instances wrapped here are immutable.
unsafe impl Send for ObjcObject {}
// SAFETY: See `Send` justification above; interior Objective-C calls shown on
// these immutable/thread-safe Metal objects require `&self` safely.
unsafe impl Sync for ObjcObject {}

impl ObjcObject {
    /// Wraps a newly created/retained (+1) Objective-C object pointer.
    pub(crate) fn from_retained(ptr: Id) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    pub(crate) fn as_raw(&self) -> Id {
        self.ptr.as_ptr()
    }
}

impl Drop for ObjcObject {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is an owned retained Objective-C object; sending
        // `release` decrements its retain count symmetrically.
        unsafe { msg_send_void_0(self.ptr.as_ptr(), sel(c"release")) };
    }
}

/// Allocates an immutable `NSString` from UTF-8 bytes without relying on autorelease.
pub(crate) fn nsstring(s: &str) -> Option<ObjcObject> {
    const NS_UTF8_STRING_ENCODING: usize = 4;
    let ns_string_cls = class(c"NSString");
    if ns_string_cls.is_null() {
        return None;
    }
    // SAFETY: `[NSString alloc]` returns a newly allocated `NSString` instance.
    let alloc = unsafe { msg_send_id_0(ns_string_cls, sel(c"alloc")) };
    if alloc.is_null() {
        return None;
    }
    // SAFETY: `initWithBytes:length:encoding:` initializes and consumes `alloc`
    // using UTF-8 bytes (`s.as_ptr()..+s.len()`), returning a retained `NSString`.
    let init = unsafe {
        type Fn = unsafe extern "C" fn(Id, Sel, *const u8, usize, usize) -> Id;
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            alloc,
            sel(c"initWithBytes:length:encoding:"),
            s.as_ptr(),
            s.len(),
            NS_UTF8_STRING_ENCODING,
        )
    };
    ObjcObject::from_retained(init)
}

/// Copies `bytes` into a retained `dispatch_data_t` with system-owned storage.
///
/// Passing the default destructor (a null block) makes libdispatch copy the
/// caller's buffer, so `bytes` only has to be valid for the duration of this
/// call. This is the input expected by `newLibraryWithData:error:`.
pub(crate) fn dispatch_data_copy(bytes: &[u8]) -> Option<ObjcObject> {
    if bytes.is_empty() {
        return None;
    }
    // SAFETY: `bytes` is valid for `bytes.len()` bytes during the call; a null
    // queue plus `DISPATCH_DATA_DESTRUCTOR_DEFAULT` (null) requests the copying
    // path, so no queue or destructor is stored or invoked.
    let raw =
        unsafe { dispatch_data_create(bytes.as_ptr().cast(), bytes.len(), null_mut(), null_mut()) };
    ObjcObject::from_retained(raw)
}

/// Converts an `NSString *` or `NSError *` into a Rust `String` within an active autorelease pool.
pub(crate) unsafe fn nserror_message(err: Id) -> String {
    if err.is_null() {
        return "unknown Metal error".to_string();
    }
    let _pool = AutoreleasePool::new();
    // SAFETY: Caller passes a valid `NSError *`; `localizedDescription` returns an `NSString *`.
    let desc = unsafe { msg_send_id_0(err, sel(c"localizedDescription")) };
    if desc.is_null() {
        return "unknown Metal error".to_string();
    }
    // SAFETY: `desc` is a valid `NSString *`; `UTF8String` returns a NUL-terminated UTF-8 pointer.
    let cstr = unsafe { msg_send_cstr_0(desc, sel(c"UTF8String")) };
    if cstr.is_null() {
        return "unknown Metal error".to_string();
    }
    // SAFETY: `UTF8String` returns a valid NUL-terminated C string valid for the autorelease pool lifetime.
    unsafe { CStr::from_ptr(cstr) }
        .to_string_lossy()
        .into_owned()
}

pub(crate) unsafe fn msg_send_id_0(target: Id, s: Sel) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel) -> Id;
    // SAFETY: Caller supplies a valid Objective-C receiver, selector, and matching return ABI.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

pub(crate) unsafe fn msg_send_id_1(target: Id, s: Sel, a0: Id) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, Id) -> Id;
    // SAFETY: Caller supplies a valid Objective-C receiver, selector, and 1 pointer argument.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0)
    }
}

pub(crate) unsafe fn msg_send_id_err(target: Id, s: Sel, a0: Id, err: *mut Id) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id;
    // SAFETY: Caller supplies a valid Objective-C receiver, selector, object arg, and `NSError **` slot.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0, err)
    }
}

pub(crate) unsafe fn msg_send_id_3_err(
    target: Id,
    s: Sel,
    a0: Id,
    a1: Id,
    err: *mut Id,
) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id;
    // SAFETY: Caller supplies valid receiver, selector, 2 pointer args, and `NSError **` out pointer.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0, a1, err)
    }
}

pub(crate) unsafe fn msg_send_nocopy_buffer(
    device: Id,
    ptr: *mut c_void,
    len: usize,
    options: usize,
) -> Id {
    type Fn = unsafe extern "C" fn(Id, Sel, *mut c_void, usize, usize, *const c_void) -> Id;
    // SAFETY: Caller supplies a valid `MTLDevice` and page-aligned buffer region. Passing null
    // for `deallocator:` matches `deallocator:nil` in Objective-C.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            device,
            sel(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
            ptr,
            len,
            options,
            null(),
        )
    }
}

pub(crate) unsafe fn msg_send_u8_0(target: Id, s: Sel) -> u8 {
    type Fn = unsafe extern "C" fn(Id, Sel) -> u8;
    // SAFETY: Caller supplies a valid Objective-C receiver and selector returning `BOOL` (`u8`).
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

pub(crate) unsafe fn msg_send_u64_0(target: Id, s: Sel) -> u64 {
    type Fn = unsafe extern "C" fn(Id, Sel) -> u64;
    // SAFETY: Caller supplies a valid Objective-C receiver and selector returning `uint64_t`.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

pub(crate) unsafe fn msg_send_usize_0(target: Id, s: Sel) -> usize {
    type Fn = unsafe extern "C" fn(Id, Sel) -> usize;
    // SAFETY: Caller supplies a valid Objective-C receiver and selector returning `NSUInteger`.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

pub(crate) unsafe fn msg_send_cstr_0(target: Id, s: Sel) -> *const c_char {
    type Fn = unsafe extern "C" fn(Id, Sel) -> *const c_char;
    // SAFETY: Caller supplies a valid Objective-C receiver and selector returning `const char *`.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s)
    }
}

pub(crate) unsafe fn msg_send_mtl_size_0(target: Id, s: Sel) -> MTLSize {
    #[cfg(target_arch = "aarch64")]
    {
        type Fn = unsafe extern "C" fn(Id, Sel) -> MTLSize;
        // SAFETY: On `aarch64`, large struct returns pass `x8` via normal `objc_msgSend`.
        unsafe {
            let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
            f(target, s)
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        type Fn = unsafe extern "C" fn(*mut MTLSize, Id, Sel);
        let mut out = MTLSize {
            width: 0,
            height: 0,
            depth: 0,
        };
        // SAFETY: On `x86_64`, 24-byte `MTLSize` return values use `objc_msgSend_stret`.
        unsafe {
            let f: Fn = mem::transmute(objc_msgSend_stret as unsafe extern "C" fn());
            f(&mut out, target, s);
        }
        out
    }
}

pub(crate) unsafe fn msg_send_void_0(target: Id, s: Sel) {
    type Fn = unsafe extern "C" fn(Id, Sel);
    // SAFETY: Caller supplies a valid Objective-C receiver and selector returning `void`.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s);
    }
}

pub(crate) unsafe fn msg_send_void_1(target: Id, s: Sel, a0: Id) {
    type Fn = unsafe extern "C" fn(Id, Sel, Id);
    // SAFETY: Caller supplies a valid Objective-C receiver, selector, and 1 pointer argument.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, s, a0);
    }
}

pub(crate) unsafe fn msg_send_set_buffer(enc: Id, buf: Id, offset: usize, index: usize) {
    type Fn = unsafe extern "C" fn(Id, Sel, Id, usize, usize);
    // SAFETY: Caller supplies a valid `MTLComputeCommandEncoder` and `MTLBuffer`.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(enc, sel(c"setBuffer:offset:atIndex:"), buf, offset, index);
    }
}

pub(crate) unsafe fn msg_send_dispatch_threads(
    enc: Id,
    threads_per_grid: MTLSize,
    threads_per_group: MTLSize,
) {
    type Fn = unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize);
    // SAFETY: Caller supplies a valid `MTLComputeCommandEncoder` and positive `MTLSize` dims.
    unsafe {
        let f: Fn = mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(
            enc,
            sel(c"dispatchThreads:threadsPerThreadgroup:"),
            threads_per_grid,
            threads_per_group,
        );
    }
}

pub(crate) fn copy_cstr_to_array<const N: usize>(src: &CStr, dst: &mut [u8; N]) {
    dst.fill(0);
    let bytes = src.to_bytes();
    let copy_len = bytes.len().min(N.saturating_sub(1));
    dst[..copy_len].copy_from_slice(&bytes[..copy_len]);
}

pub(crate) fn empty_id() -> Id {
    null_mut()
}
