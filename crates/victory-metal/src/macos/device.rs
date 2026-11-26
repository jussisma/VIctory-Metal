//! Direct in-process macOS Metal implementation of [`MetalTransport`].
//!
//! Used both by native macOS callers (Phase 2, Wine-free) and behind `c_api`
//! for Phase 1 Wine unixlib bridges. Zero Wine or Windows code lives here.

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;

use crate::abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmPipelineId};
use crate::macos::irconverter;
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::transport::{MetalError, MetalTransport};
use crate::wire::{self, Command, Reply, ReplyStatus, Submission};

const APPLE_SILICON_NOCOPY_PAGE_SIZE: usize = 16 * 1024;
/// `MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared | MTLResourceHazardTrackingModeDefault`.
const MTL_RESOURCE_STORAGE_MODE_SHARED: usize = 0;
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// Conservative byte alignment for a bound range offset.
///
/// `setBuffer:offset:atIndex:` accepts 4-byte-aligned offsets for general
/// buffers. Shader-declared types may require stricter natural alignment (for
/// example 16 bytes for a `float4`), which the host cannot infer from the wire
/// frame. This entry point therefore enforces the 4-byte ABI minimum and leaves
/// stricter shader-specific alignment to Metal's runtime validation; application
/// layouts should place every typed declaration on its natural alignment.
const MTL_BUFFER_OFFSET_ALIGNMENT: u64 = 4;

struct RegisteredBuffer {
    buffer: ObjcObject,
    /// Byte length reported to Metal at registration time.
    len: usize,
}

struct DeviceState {
    next_id: u64,
    buffers: HashMap<u64, RegisteredBuffer>,
    pipelines: HashMap<u64, ObjcObject>,
}

impl DeviceState {
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }
}

/// Native Apple Metal backend implementing [`MetalTransport`].
pub struct DirectMetalTransport {
    device: ObjcObject,
    queue: ObjcObject,
    info: VmDeviceInfo,
    state: Mutex<DeviceState>,
}

impl DirectMetalTransport {
    /// Opens the system default Metal GPU device and initializes a command queue.
    pub fn new() -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        let device = objc::create_system_default_device().ok_or(MetalError::DeviceUnavailable)?;

        // SAFETY: `device` is a valid `MTLDevice` and `newCommandQueue` returns a retained queue.
        let queue_raw = unsafe { objc::msg_send_id_0(device.as_raw(), objc::sel(c"newCommandQueue")) };
        let queue = ObjcObject::from_retained(queue_raw).ok_or_else(|| {
            MetalError::InternalError("MTLDevice.newCommandQueue returned null".to_string())
        })?;

        let info = query_device_info(&device);

        Ok(Self {
            device,
            queue,
            info,
            state: Mutex::new(DeviceState {
                next_id: 1,
                buffers: HashMap::new(),
                pipelines: HashMap::new(),
            }),
        })
    }

    fn create_compute_pipeline(
        &self,
        lib: &ObjcObject,
        entry: &str,
    ) -> Result<VmPipelineId, MetalError> {
        let _pool = AutoreleasePool::new();
        let entry_ns = objc::nsstring(entry)
            .ok_or(MetalError::InvalidArg("entry name string allocation failed"))?;

        // SAFETY: `lib` is a valid `MTLLibrary` and `entry_ns` is a retained `NSString`.
        let fn_raw = unsafe {
            objc::msg_send_id_1(
                lib.as_raw(),
                objc::sel(c"newFunctionWithName:"),
                entry_ns.as_raw(),
            )
        };
        let func = ObjcObject::from_retained(fn_raw).ok_or_else(|| {
            MetalError::NotFound(format!("shader function `{entry}` not found in MTLLibrary"))
        })?;

        let mut ns_err = objc::empty_id();
        // SAFETY: `device` and `func` are valid Metal objects; `ns_err` receives an autoreleased `NSError *`.
        let pso_raw = unsafe {
            objc::msg_send_id_err(
                self.device.as_raw(),
                objc::sel(c"newComputePipelineStateWithFunction:error:"),
                func.as_raw(),
                &mut ns_err,
            )
        };
        let pso = ObjcObject::from_retained(pso_raw).ok_or_else(|| {
            // SAFETY: `ns_err` was populated by Metal on pipeline creation failure.
            let msg = unsafe { objc::nserror_message(ns_err) };
            MetalError::CompileFailed(msg)
        })?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;
        let id = state.alloc_id();
        state.pipelines.insert(id, pso);
        Ok(VmPipelineId(id))
    }

    /// Executes a decoded version-1 submission as one ordered native batch.
    ///
    /// # Contract
    ///
    /// * `allocations` is a caller-owned snapshot of the allocation registry,
    ///   keyed by `(allocation_id, generation)`. It is a plain `HashMap`, not a
    ///   callback, so no caller code can run while the internal state mutex is
    ///   held; refreshing the snapshot is the caller's responsibility.
    /// * Every command is fully resolved against `allocations`, the pipeline
    ///   table and the registered buffer table before any command buffer exists.
    ///   An unknown pipeline, allocation, stale generation, zero allocation ID or
    ///   generation, or out-of-range binding returns
    ///   [`ReplyStatus::UnknownHandle`] and performs no GPU work.
    /// * A binding range must satisfy `offset % 4 == 0` and
    ///   `offset + length <= registered_length`. `length` is a lifetime/validity
    ///   claim for the registered range, not a shader index bound: host kernels
    ///   are trusted, and Metal validates shader-specific alignment at dispatch.
    /// * `threads_per_group` must not exceed the pipeline's
    ///   `maxTotalThreadsPerThreadgroup`; an explicit width over the limit is
    ///   rejected with [`MetalError::InvalidArg`] before GPU work. Silently
    ///   clamping would change threadgroup indices and shader semantics.
    /// * One `MTLCommandBuffer` holds one compute encoder per command, so Metal's
    ///   hazard tracking orders the dispatches while keeping the batch atomic.
    /// * The state lock is held through synchronous completion, so a registered
    ///   buffer cannot be unregistered while the GPU reads or writes it.
    /// * The returned reply echoes the submission sequence. A GPU fault after
    ///   commit returns [`ReplyStatus::ExecutionFailed`].
    ///
    /// Limits from [`crate::wire`] are re-checked here so a programmatically
    /// constructed [`Submission`] cannot bypass the decoder bounds.
    pub fn submit_batch(
        &self,
        submission: &Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        if submission.sequence == 0 {
            return Err(MetalError::InvalidArg(
                "submission sequence must be nonzero",
            ));
        }
        if submission.commands.len() > wire::MAX_COMMANDS {
            return Err(MetalError::InvalidArg(
                "submission exceeds wire command limit",
            ));
        }
        for command in &submission.commands {
            match command {
                Command::Dispatch1d(dispatch) => {
                    if dispatch.pipeline_id == 0
                        || dispatch.threads == 0
                        || dispatch.threads_per_group == 0
                    {
                        return Err(MetalError::InvalidArg(
                            "dispatch handle and thread counts must be nonzero",
                        ));
                    }
                    if dispatch.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                        return Err(MetalError::InvalidArg(
                            "dispatch exceeds wire binding limit",
                        ));
                    }
                }
            }
        }

        let reply = |status| Reply {
            sequence: submission.sequence,
            status,
        };
        if submission.commands.is_empty() {
            return Ok(reply(ReplyStatus::Ok));
        }

        let _pool = AutoreleasePool::new();
        // Hold the state lock through synchronous completion so registered
        // buffers and pipelines cannot be released while the GPU uses them.
        let state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;

        // Resolve and validate everything before creating or committing a command
        // buffer. This is what makes a stale generation a no-op on GPU memory.
        let mut prepared = Vec::with_capacity(submission.commands.len());
        for command in &submission.commands {
            let Command::Dispatch1d(dispatch) = command;
            let Some(pso) = state.pipelines.get(&dispatch.pipeline_id) else {
                return Ok(reply(ReplyStatus::UnknownHandle));
            };

            let mut bindings = Vec::with_capacity(dispatch.bindings.len());
            for (slot, binding) in dispatch.bindings.iter().enumerate() {
                if binding.allocation_id == 0
                    || binding.generation == 0
                    || binding.length == 0
                    || binding.length > wire::MAX_RESOURCE_LEN
                    || !binding.offset.is_multiple_of(MTL_BUFFER_OFFSET_ALIGNMENT)
                {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                }
                let Some(registered_id) = allocations
                    .get(&(binding.allocation_id, binding.generation))
                    .copied()
                else {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                };
                let Some(registered) = state.buffers.get(&registered_id.0) else {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                };
                let Some(end) = binding.checked_end() else {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                };
                let (Ok(offset), Ok(end)) = (usize::try_from(binding.offset), usize::try_from(end))
                else {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                };
                if end > registered.len {
                    return Ok(reply(ReplyStatus::UnknownHandle));
                }
                bindings.push((registered.buffer.as_raw(), offset, slot));
            }

            // Unlike the legacy single-dispatch API, do not clamp an explicit
            // wire width: that would change threadgroup indices and shader
            // semantics. Reject before the command buffer exists.
            // SAFETY: `pso` is a valid `MTLComputePipelineState`.
            let max_threads = unsafe {
                objc::msg_send_usize_0(pso.as_raw(), objc::sel(c"maxTotalThreadsPerThreadgroup"))
            }
            .max(1);
            let threads_per_group = dispatch.threads_per_group as usize;
            if threads_per_group > max_threads {
                return Err(MetalError::InvalidArg(
                    "dispatch threads_per_group exceeds pipeline threadgroup maximum",
                ));
            }
            prepared.push((pso.as_raw(), dispatch.threads, threads_per_group, bindings));
        }

        // SAFETY: `self.queue` is a valid `MTLCommandQueue`; the returned command
        // buffer is autoreleased and stays alive for this pool's lifetime.
        let cb = unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }

        for (pso, threads, threads_per_group, bindings) in &prepared {
            // SAFETY: `cb` is a valid uncommitted `MTLCommandBuffer`.
            let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
            if enc.is_null() {
                return Err(MetalError::InternalError(
                    "MTLCommandBuffer.computeCommandEncoder returned null".to_string(),
                ));
            }
            // SAFETY: `enc` and every prepared pipeline/buffer are valid Metal
            // objects for the active encoder scope, and offsets were validated.
            unsafe {
                objc::msg_send_void_1(enc, objc::sel(c"setComputePipelineState:"), *pso);
                for (buf_raw, offset, slot) in bindings {
                    objc::msg_send_set_buffer(enc, *buf_raw, *offset, *slot);
                }
                objc::msg_send_dispatch_threads(
                    enc,
                    MTLSize {
                        width: *threads as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: *threads_per_group,
                        height: 1,
                        depth: 1,
                    },
                );
                objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
            }
        }

        // SAFETY: `cb` is a valid command buffer with every encoder closed.
        unsafe {
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }

        // SAFETY: `cb` has completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            return Ok(reply(ReplyStatus::ExecutionFailed));
        }

        Ok(reply(ReplyStatus::Ok))
    }
}

impl MetalTransport for DirectMetalTransport {
    fn device_info(&self) -> Result<VmDeviceInfo, MetalError> {
        Ok(self.info)
    }

    unsafe fn register_zerocopy(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        if ptr.is_null() || len == 0 {
            return Err(MetalError::InvalidArg("buffer pointer is null or len == 0"));
        }
        let page_size = usize::try_from(self.info.page_size).unwrap_or(APPLE_SILICON_NOCOPY_PAGE_SIZE);
        let addr = ptr as usize;
        if !addr.is_multiple_of(page_size) || !len.is_multiple_of(page_size) {
            return Err(MetalError::AlignmentError {
                ptr: addr,
                len,
                page_size,
            });
        }

        // SAFETY: Caller upholds the `register_zerocopy` lifetime contract; alignment & non-zero
        // length were verified above, and `deallocator` is nil so Metal never frees host memory.
        let buf_raw = unsafe {
            objc::msg_send_nocopy_buffer(
                self.device.as_raw(),
                ptr.cast::<c_void>(),
                len,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        let buf = ObjcObject::from_retained(buf_raw).ok_or_else(|| {
            MetalError::InternalError(format!(
                "newBufferWithBytesNoCopy returned null for 0x{addr:x} ({len} bytes)"
            ))
        })?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;
        let id = state.alloc_id();
        state
            .buffers
            .insert(id, RegisteredBuffer { buffer: buf, len });
        Ok(VmBufferId(id))
    }

    fn unregister_buffer(&self, id: VmBufferId) -> Result<(), MetalError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;
        if state.buffers.remove(&id.0).is_none() {
            return Err(MetalError::NotFound(format!("buffer {}", id.0)));
        }
        Ok(())
    }

    fn compile_msl(&self, source: &str, entry: &str) -> Result<VmPipelineId, MetalError> {
        if source.is_empty() || entry.is_empty() {
            return Err(MetalError::InvalidArg("MSL source or entry point is empty"));
        }
        let _pool = AutoreleasePool::new();
        let src_ns = objc::nsstring(source)
            .ok_or(MetalError::InvalidArg("MSL source string allocation failed"))?;

        let mut ns_err = objc::empty_id();
        // SAFETY: `device` is a valid `MTLDevice`; `src_ns` is an immutable `NSString`; `options` is nil.
        let lib_raw = unsafe {
            objc::msg_send_id_3_err(
                self.device.as_raw(),
                objc::sel(c"newLibraryWithSource:options:error:"),
                src_ns.as_raw(),
                objc::empty_id(),
                &mut ns_err,
            )
        };
        let lib = ObjcObject::from_retained(lib_raw).ok_or_else(|| {
            // SAFETY: `ns_err` is populated by Metal when MSL compilation fails.
            let msg = unsafe { objc::nserror_message(ns_err) };
            MetalError::CompileFailed(msg)
        })?;

        self.create_compute_pipeline(&lib, entry)
    }

    fn compile_dxil(
        &self,
        dxil: &[u8],
        entry: &str,
    ) -> Result<(VmPipelineId, VmDxilCompileOutput), MetalError> {
        if entry.is_empty() {
            return Err(MetalError::InvalidArg("shader entry point is empty"));
        }
        let (lib, output) = irconverter::compile_dxil_to_library(&self.device, dxil, entry)?;
        let pipeline = self.create_compute_pipeline(&lib, entry)?;
        Ok((pipeline, output))
    }

    fn dispatch_compute_1d(
        &self,
        pipeline: VmPipelineId,
        buffers: &[VmBufferId],
        threads: u32,
        threads_per_group: u32,
    ) -> Result<(), MetalError> {
        if threads == 0 || threads_per_group == 0 {
            return Err(MetalError::InvalidArg(
                "threads and threads_per_group must be non-zero",
            ));
        }

        let _pool = AutoreleasePool::new();
        // Hold the state lock through synchronous completion so registered buffers
        // cannot be unregistered while the GPU reads/writes them.
        let state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;

        let pso = state
            .pipelines
            .get(&pipeline.0)
            .ok_or_else(|| MetalError::NotFound(format!("pipeline {}", pipeline.0)))?;
        let mut bound = Vec::with_capacity(buffers.len());
        for buf_id in buffers {
            let buf = state
                .buffers
                .get(&buf_id.0)
                .ok_or_else(|| MetalError::NotFound(format!("buffer {}", buf_id.0)))?;
            bound.push(buf.buffer.as_raw());
        }

        // SAFETY: `pso` is a valid `MTLComputePipelineState`.
        let pso_max = unsafe {
            objc::msg_send_usize_0(pso.as_raw(), objc::sel(c"maxTotalThreadsPerThreadgroup"))
        }
        .max(1);
        let tpg = (threads_per_group as usize).min(pso_max);

        // SAFETY: `self.queue` is a valid `MTLCommandQueue`; returns an autoreleased command buffer.
        let cb = unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }
        // SAFETY: `cb` is a valid uncommitted `MTLCommandBuffer`.
        let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
        if enc.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandBuffer.computeCommandEncoder returned null".to_string(),
            ));
        }

        // SAFETY: `enc` and `pso` are valid Metal objects for the active encoder scope.
        unsafe {
            objc::msg_send_void_1(
                enc,
                objc::sel(c"setComputePipelineState:"),
                pso.as_raw(),
            );
            for (slot, buf_raw) in bound.iter().enumerate() {
                objc::msg_send_set_buffer(enc, *buf_raw, 0, slot);
            }
            objc::msg_send_dispatch_threads(
                enc,
                MTLSize {
                    width: threads as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tpg,
                    height: 1,
                    depth: 1,
                },
            );
            objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }

        // SAFETY: `cb` has completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: `[cb error]` returns an autoreleased `NSError *` describing the GPU fault.
            let err_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
            // SAFETY: `err_raw` is either null or a valid `NSError *`.
            let msg = unsafe { objc::nserror_message(err_raw) };
            return Err(MetalError::InternalError(format!(
                "Metal compute command buffer failed (status={status}): {msg}"
            )));
        }

        Ok(())
    }
}

fn query_device_info(device: &ObjcObject) -> VmDeviceInfo {
    let mut info = VmDeviceInfo::default();
    let _pool = AutoreleasePool::new();

    // SAFETY: `device` is a valid `MTLDevice`; `[device name]` returns an autoreleased `NSString *`.
    let name_ns = unsafe { objc::msg_send_id_0(device.as_raw(), objc::sel(c"name")) };
    if !name_ns.is_null() {
        // SAFETY: `name_ns` is a valid `NSString *`; `UTF8String` is NUL-terminated.
        let cstr_ptr = unsafe { objc::msg_send_cstr_0(name_ns, objc::sel(c"UTF8String")) };
        if !cstr_ptr.is_null() {
            // SAFETY: `cstr_ptr` stays valid for the lifetime of `_pool`.
            let cstr = unsafe { CStr::from_ptr(cstr_ptr) };
            objc::copy_cstr_to_array(cstr, &mut info.name);
        }
    }

    // SAFETY: `[device hasUnifiedMemory]` returns a `BOOL` byte.
    let unified = unsafe { objc::msg_send_u8_0(device.as_raw(), objc::sel(c"hasUnifiedMemory")) };
    info.has_unified_memory = u8::from(unified != 0);

    // SAFETY: `[device maxThreadsPerThreadgroup]` returns `MTLSize`.
    let max_threads = unsafe {
        objc::msg_send_mtl_size_0(device.as_raw(), objc::sel(c"maxThreadsPerThreadgroup"))
    };
    info.max_threads_per_threadgroup = u32::try_from(max_threads.width).unwrap_or(u32::MAX);

    // SAFETY: `[device recommendedMaxWorkingSetSize]` returns `uint64_t`.
    info.recommended_max_working_set_size =
        unsafe { objc::msg_send_u64_0(device.as_raw(), objc::sel(c"recommendedMaxWorkingSetSize")) };

    let host_page = objc::host_page_size();
    let nocopy_page = if info.has_unified_memory != 0 {
        host_page.max(APPLE_SILICON_NOCOPY_PAGE_SIZE)
    } else {
        host_page
    };
    info.page_size = nocopy_page as u64;

    info
}
