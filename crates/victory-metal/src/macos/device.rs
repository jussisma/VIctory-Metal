//! Direct in-process macOS Metal implementation of [`MetalTransport`].
//!
//! Used both by native macOS callers (Phase 2, Wine-free) and behind `c_api`
//! for Phase 1 Wine unixlib bridges. Zero Wine or Windows code lives here.

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;

use crate::abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmPipelineId};
use crate::macos::irconverter;
use crate::macos::msc::{self, MscUavDispatch};
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::transport::{MetalError, MetalTransport};
use crate::wire::{self, Command, Reply, ReplyStatus, Submission};

const APPLE_SILICON_NOCOPY_PAGE_SIZE: usize = 16 * 1024;
/// `MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared | MTLResourceHazardTrackingModeDefault`.
const MTL_RESOURCE_STORAGE_MODE_SHARED: usize = 0;
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// `MTLResourceUsageRead`.
const MTL_RESOURCE_USAGE_READ: usize = 1;
/// `MTLResourceUsageWrite`.
const MTL_RESOURCE_USAGE_WRITE: usize = 2;
/// Conservative byte alignment for a bound range offset.
///
/// `setBuffer:offset:atIndex:` accepts 4-byte-aligned offsets for general
/// buffers. Shader-declared types may require stricter natural alignment (for
/// example 16 bytes for a `float4`), which the host cannot infer from the wire
/// frame. This entry point therefore enforces the 4-byte ABI minimum and leaves
/// stricter shader-specific alignment to Metal's runtime validation; application
/// layouts should place every typed declaration on its natural alignment.
pub(crate) const MTL_BUFFER_OFFSET_ALIGNMENT: u64 = 4;

pub(crate) struct RegisteredBuffer {
    pub(crate) buffer: ObjcObject,
    /// Byte length reported to Metal at registration time.
    pub(crate) len: usize,
}

pub(crate) struct DeviceState {
    next_id: u64,
    pub(crate) buffers: HashMap<u64, RegisteredBuffer>,
    pipelines: HashMap<u64, ObjcObject>,
    pub(crate) textures: HashMap<u64, ObjcObject>,
}

impl DeviceState {
    pub(crate) fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }
}

/// One resolved `(MTLBuffer, byte offset, buffer slot)` binding.
type BoundRange = (objc::Id, usize, usize);

/// A command resolved against the live pipeline and buffer registries but not
/// yet encoded into a command buffer.
enum PreparedDispatch {
    OneD {
        pipeline: objc::Id,
        threads: usize,
        threads_per_group: usize,
        bindings: Vec<BoundRange>,
    },
    ThreeD {
        pipeline: objc::Id,
        group_counts: MTLSize,
        threads_per_group: MTLSize,
        bindings: Vec<BoundRange>,
    },
}

/// Resolves and validates every binding against one registry snapshot.
///
/// Returns `None` for any unknown handle, stale generation, zero field or
/// out-of-range span. The caller maps that to `UnknownHandle` before creating a
/// command buffer, so a rejected batch performs no GPU work.
fn resolve_bindings(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    bindings: &[wire::BufferBinding],
) -> Option<Vec<BoundRange>> {
    let mut resolved = Vec::with_capacity(bindings.len());
    for (slot, binding) in bindings.iter().enumerate() {
        if binding.allocation_id == 0
            || binding.generation == 0
            || binding.length == 0
            || binding.length > wire::MAX_RESOURCE_LEN
            || !binding.offset.is_multiple_of(MTL_BUFFER_OFFSET_ALIGNMENT)
        {
            return None;
        }
        let registered_id = *allocations.get(&(binding.allocation_id, binding.generation))?;
        let registered = state.buffers.get(&registered_id.0)?;
        let end = binding.checked_end()?;
        let offset = usize::try_from(binding.offset).ok()?;
        let end = usize::try_from(end).ok()?;
        if end > registered.len {
            return None;
        }
        resolved.push((registered.buffer.as_raw(), offset, slot));
    }
    Some(resolved)
}

fn pipeline_thread_limit(pso: &ObjcObject) -> u64 {
    // SAFETY: `pso` is a valid `MTLComputePipelineState`.
    unsafe { objc::msg_send_usize_0(pso.as_raw(), objc::sel(c"maxTotalThreadsPerThreadgroup")) }
        .max(1) as u64
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
                textures: HashMap::new(),
            }),
        })
    }

    /// Raw retained `MTLDevice` handle for sibling modules that allocate GPU objects.
    pub(crate) fn device_raw(&self) -> objc::Id {
        self.device.as_raw()
    }

    /// Raw retained `MTLCommandQueue` handle.
    ///
    /// Blits submitted through this queue share its serial execution order with
    /// [`DirectMetalTransport::submit_batch`] and `dispatch_compute_1d`.
    pub(crate) fn queue_raw(&self) -> objc::Id {
        self.queue.as_raw()
    }

    /// Locks the allocation registry.
    ///
    /// Callers must hold the guard until any GPU work that references registry
    /// objects has completed, exactly like the existing batch and legacy paths.
    pub(crate) fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, DeviceState>, MetalError> {
        self.state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))
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
    ///   The same resolver validates 1D and 3D commands.
    /// * A 1D `threads_per_group` must not exceed the pipeline's
    ///   `maxTotalThreadsPerThreadgroup`; an explicit width over the limit is
    ///   rejected with [`MetalError::InvalidArg`] before GPU work. Silently
    ///   clamping would change threadgroup indices and shader semantics.
    /// * A 3D command carries nonzero D3D threadgroup counts and nonzero local
    ///   `x`, `y`, `z` dimensions. The checked local product must not exceed the
    ///   pipeline's `maxTotalThreadsPerThreadgroup`; oversized local dimensions
    ///   are rejected with [`MetalError::InvalidArg`] before GPU work. The host
    ///   encodes the exact group grid with
    ///   `dispatchThreadgroups:threadsPerThreadgroup:`, never a reshaped total.
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
                Command::Dispatch3d(dispatch) => {
                    if dispatch.pipeline_id == 0
                        || dispatch.group_counts.contains(&0)
                        || dispatch.threads_per_group.contains(&0)
                    {
                        return Err(MetalError::InvalidArg(
                            "3D dispatch handle and dimensions must be nonzero",
                        ));
                    }
                    if dispatch.checked_total_threads().is_none() {
                        return Err(MetalError::InvalidArg(
                            "3D dispatch thread count overflows u64",
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
            let (pipeline_id, wire_bindings) = match command {
                Command::Dispatch1d(dispatch) => (dispatch.pipeline_id, &dispatch.bindings),
                Command::Dispatch3d(dispatch) => (dispatch.pipeline_id, &dispatch.bindings),
            };
            let Some(pso) = state.pipelines.get(&pipeline_id) else {
                return Ok(reply(ReplyStatus::UnknownHandle));
            };
            let Some(bindings) = resolve_bindings(&state, allocations, wire_bindings) else {
                return Ok(reply(ReplyStatus::UnknownHandle));
            };
            let max_threads = pipeline_thread_limit(pso);

            // Unlike the legacy single-dispatch API, do not clamp an explicit
            // wire width: that would change threadgroup indices and shader
            // semantics. Reject before the command buffer exists.
            match command {
                Command::Dispatch1d(dispatch) => {
                    let threads_per_group = dispatch.threads_per_group as usize;
                    if threads_per_group as u64 > max_threads {
                        return Err(MetalError::InvalidArg(
                            "dispatch threads_per_group exceeds pipeline threadgroup maximum",
                        ));
                    }
                    prepared.push(PreparedDispatch::OneD {
                        pipeline: pso.as_raw(),
                        threads: dispatch.threads as usize,
                        threads_per_group,
                        bindings,
                    });
                }
                Command::Dispatch3d(dispatch) => {
                    // `dispatchThreadgroups:` uses the group grid exactly. Its
                    // local MTLSize must fit the pipeline's hardware total before
                    // any command buffer is created; never clamp or reshape it.
                    let Some(local_threads) = dispatch.checked_threads_per_group() else {
                        return Err(MetalError::InvalidArg(
                            "3D dispatch local thread product overflows u64",
                        ));
                    };
                    if local_threads > max_threads {
                        return Err(MetalError::InvalidArg(
                            "3D dispatch threadgroup exceeds pipeline threadgroup maximum",
                        ));
                    }
                    prepared.push(PreparedDispatch::ThreeD {
                        pipeline: pso.as_raw(),
                        group_counts: MTLSize {
                            width: dispatch.group_counts[0] as usize,
                            height: dispatch.group_counts[1] as usize,
                            depth: dispatch.group_counts[2] as usize,
                        },
                        threads_per_group: MTLSize {
                            width: dispatch.threads_per_group[0] as usize,
                            height: dispatch.threads_per_group[1] as usize,
                            depth: dispatch.threads_per_group[2] as usize,
                        },
                        bindings,
                    });
                }
            }
        }

        // SAFETY: `self.queue` is a valid `MTLCommandQueue`; the returned command
        // buffer is autoreleased and stays alive for this pool's lifetime.
        let cb = unsafe { objc::msg_send_id_0(self.queue.as_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }

        for dispatch in &prepared {
            // SAFETY: `cb` is a valid uncommitted `MTLCommandBuffer`.
            let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
            if enc.is_null() {
                return Err(MetalError::InternalError(
                    "MTLCommandBuffer.computeCommandEncoder returned null".to_string(),
                ));
            }

            let (pipeline, bindings) = match dispatch {
                PreparedDispatch::OneD {
                    pipeline, bindings, ..
                }
                | PreparedDispatch::ThreeD {
                    pipeline, bindings, ..
                } => (*pipeline, bindings),
            };

            // SAFETY: `enc` and every prepared pipeline/buffer are valid Metal
            // objects for the active encoder scope, and offsets were validated.
            unsafe {
                objc::msg_send_void_1(enc, objc::sel(c"setComputePipelineState:"), pipeline);
                for (buf_raw, offset, slot) in bindings {
                    objc::msg_send_set_buffer(enc, *buf_raw, *offset, *slot);
                }
            }
            match dispatch {
                PreparedDispatch::OneD {
                    threads,
                    threads_per_group,
                    ..
                } => {
                    // SAFETY: `enc` is valid and the 1D grid dimensions were
                    // validated as nonzero.
                    unsafe {
                        objc::msg_send_dispatch_threads(
                            enc,
                            MTLSize {
                                width: *threads,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: *threads_per_group,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                }
                PreparedDispatch::ThreeD {
                    group_counts,
                    threads_per_group,
                    ..
                } => {
                    // SAFETY: `enc` is valid; the exact group grid is nonzero
                    // and the local dimensions fit the pipeline's hardware limit.
                    unsafe {
                        objc::msg_send_dispatch_threadgroups(
                            enc,
                            *group_counts,
                            *threads_per_group,
                        );
                    }
                }
            }
            // SAFETY: `enc` is a valid compute encoder awaiting `endEncoding`.
            unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
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

    /// Loads a precompiled Metal library (`.metallib` bytes) and registers the
    /// compute function `entry` in the same pipeline table that
    /// [`Self::submit_batch`] resolves.
    ///
    /// This is the native bridge for offline-compiled libraries, such as the
    /// bytes produced by a separate Rosetta `libmetalirconverter` host or by
    /// Apple's `metal`/`metallib` tools. The byte slice is copied into a
    /// `dispatch_data_t` and handed to `-[MTLDevice newLibraryWithData:error:]`,
    /// so the current process needs no shader source, no x86_64 library and no
    /// filesystem path at load time.
    ///
    /// The returned [`VmPipelineId`] is immediately valid for
    /// [`crate::wire`] submissions through [`Self::submit_batch`]. An unknown
    /// entry returns [`MetalError::NotFound`]; malformed library bytes return
    /// [`MetalError::CompileFailed`] with Metal's diagnostic.
    pub fn load_metallib(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        if metallib.is_empty() || entry.is_empty() {
            return Err(MetalError::InvalidArg(
                "metallib bytes or entry point is empty",
            ));
        }
        let _pool = AutoreleasePool::new();
        let data = objc::dispatch_data_copy(metallib).ok_or_else(|| {
            MetalError::InternalError("dispatch_data_create returned null".to_string())
        })?;

        let mut ns_err = objc::empty_id();
        // SAFETY: `device` is a valid `MTLDevice`, `data` is a retained
        // `dispatch_data_t`, and `ns_err` receives an autoreleased `NSError *`.
        let lib_raw = unsafe {
            objc::msg_send_id_err(
                self.device.as_raw(),
                objc::sel(c"newLibraryWithData:error:"),
                data.as_raw(),
                &mut ns_err,
            )
        };
        let lib = ObjcObject::from_retained(lib_raw).ok_or_else(|| {
            // SAFETY: `ns_err` was populated by Metal when the library failed to load.
            let msg = unsafe { objc::nserror_message(ns_err) };
            MetalError::CompileFailed(msg)
        })?;

        self.create_compute_pipeline(&lib, entry)
    }

    /// Binds one registered UAV for the verified Metal Shader Converter root
    /// signature and executes a 1D compute dispatch.
    ///
    /// This is an explicit opt-in path, separate from
    /// [`MetalTransport::dispatch_compute_1d`] and from wire submissions. It
    /// encodes the **one verified shape** only: the converted shader reads an
    /// 8-byte device address from argument offset 0 of a top-level argument
    /// buffer bound at Metal `buffer(2)`. Callers must have obtained that shape
    /// from a verified conversion, not guessed it. Any other slot or offset is
    /// requested through [`MscUavLayout::new`] with explicit values.
    ///
    /// # Contract
    ///
    /// * `pipeline` must come from the same [`DirectMetalTransport`], typically
    ///   from [`Self::load_metallib`] on bytes produced by the x86_64 converter
    ///   process; `uav` must be a live handle from
    ///   [`MetalTransport::register_zerocopy`].
    /// * `offset`/`length` must describe the UAV range inside the registered
    ///   backing. The range is validated here: nonzero, at most
    ///   [`wire::MAX_RESOURCE_LEN`], 4-byte offset alignment, no overflow, and
    ///   `offset + length <= registered_len`. Invalid ranges fail before any
    ///   Metal object or command buffer is created.
    /// * `threads_per_group` above the pipeline's
    ///   `maxTotalThreadsPerThreadgroup` is rejected with
    ///   [`MetalError::InvalidArg`]; it is never clamped.
    /// * The internal state lock is held through synchronous completion, so the
    ///   registered backing, the pipeline and the argument buffer all outlive
    ///   the GPU work. The argument buffer is a local RAII object dropped only
    ///   after `waitUntilCompleted`.
    /// * Residency: the UAV is declared with
    ///   `useResource:usage:(Read|Write)` on the encoder, as the verified
    ///   harness does. The directly bound argument buffer needs no extra
    ///   declaration.
    ///
    /// An unknown pipeline/buffer returns [`MetalError::NotFound`]; a GPU fault
    /// returns [`MetalError::InternalError`] with Metal's command-buffer
    /// diagnostic.
    pub fn dispatch_msc_uav_1d(&self, request: &MscUavDispatch) -> Result<(), MetalError> {
        if request.threads == 0 || request.threads_per_group == 0 {
            return Err(MetalError::InvalidArg(
                "threads and threads_per_group must be non-zero",
            ));
        }

        let _pool = AutoreleasePool::new();
        // Hold the state lock through synchronous completion so the pipeline and
        // the registered backing cannot be released while the GPU uses them.
        let state = self
            .state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))?;

        let pso = state
            .pipelines
            .get(&request.pipeline.0)
            .ok_or_else(|| MetalError::NotFound(format!("pipeline {}", request.pipeline.0)))?;
        let uav = state
            .buffers
            .get(&request.uav.0)
            .ok_or_else(|| MetalError::NotFound(format!("buffer {}", request.uav.0)))?;

        let (offset, _end) = msc::checked_binding_range(request.offset, request.length, uav.len)?;

        // SAFETY: `pso` is a valid `MTLComputePipelineState`.
        let max_threads = unsafe {
            objc::msg_send_usize_0(pso.as_raw(), objc::sel(c"maxTotalThreadsPerThreadgroup"))
        }
        .max(1);
        if request.threads_per_group as usize > max_threads {
            return Err(MetalError::InvalidArg(
                "dispatch threads_per_group exceeds pipeline threadgroup maximum",
            ));
        }

        // SAFETY: `uav.buffer` is a live `MTLBuffer`; `gpuAddress` is its base device address.
        let base_address =
            unsafe { objc::msg_send_u64_0(uav.buffer.as_raw(), objc::sel(c"gpuAddress")) };
        let device_address = base_address.checked_add(offset as u64).ok_or_else(|| {
            MetalError::InternalError("UAV device address overflowed".to_string())
        })?;

        // The argument buffer is created per dispatch so no stale address can be
        // observed by a later submission. Metal zero-initializes new buffers;
        // zero it again explicitly and then write the single address field.
        // SAFETY: `self.device` is a valid `MTLDevice`; shared storage is CPU-visible.
        let args_raw = unsafe {
            objc::msg_send_new_buffer(
                self.device.as_raw(),
                request.layout.argument_buffer_len(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        let arguments = ObjcObject::from_retained(args_raw).ok_or_else(|| {
            MetalError::InternalError("argument buffer allocation returned null".to_string())
        })?;
        // SAFETY: `arguments` is a live shared-storage `MTLBuffer`; `contents`
        // points to exactly `argument_buffer_len()` writable bytes.
        let contents = unsafe { objc::msg_send_ptr_0(arguments.as_raw(), objc::sel(c"contents")) };
        if contents.is_null() {
            return Err(MetalError::InternalError(
                "argument buffer has no CPU-visible contents".to_string(),
            ));
        }
        // SAFETY: The slice covers the whole freshly created argument buffer.
        let arguments_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                contents.cast::<u8>(),
                request.layout.argument_buffer_len(),
            )
        };
        arguments_bytes.fill(0);
        msc::write_uav_address(arguments_bytes, request.layout, device_address)?;

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

        // SAFETY: `enc`, `pso`, `arguments` and `uav.buffer` are valid Metal
        // objects for the active encoder scope; the range was validated above.
        unsafe {
            objc::msg_send_void_1(enc, objc::sel(c"setComputePipelineState:"), pso.as_raw());
            objc::msg_send_set_buffer(
                enc,
                arguments.as_raw(),
                0,
                request.layout.argument_buffer_index(),
            );
            objc::msg_send_use_resource(
                enc,
                uav.buffer.as_raw(),
                MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
            );
            objc::msg_send_dispatch_threads(
                enc,
                MTLSize {
                    width: request.threads as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: request.threads_per_group as usize,
                    height: 1,
                    depth: 1,
                },
            );
            objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }

        // SAFETY: `cb` has completed via `waitUntilCompleted`; `arguments` stays
        // alive until this function returns.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: `[cb error]` returns an autoreleased `NSError *` describing the GPU fault.
            let err_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
            // SAFETY: `err_raw` is either null or a valid `NSError *`.
            let msg = unsafe { objc::nserror_message(err_raw) };
            return Err(MetalError::InternalError(format!(
                "Metal MSC argument-buffer dispatch failed (status={status}): {msg}"
            )));
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command as ProcessCommand;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::DirectMetalTransport;
    use crate::macos::msc::{MscUavDispatch, MscUavLayout};
    use crate::transport::{MetalError, MetalTransport};
    use crate::wire::{self, BufferBinding, Command, Dispatch1d, Reply, ReplyStatus, Submission};

    /// Original synthetic compute source. Two entry points prove that a named
    /// function inside the precompiled library is selected, not the first one.
    const METALLIB_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void vm_metallib_affine(device uint *data [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 3u + 7u;
}
kernel void vm_metallib_unused(device uint *data [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] + 1u;
}
"#;

    /// ARM64 stand-in for the converted top-level argument buffer contract:
    /// Metal `buffer(2)` holds a `device uint *` device address at offset 0.
    const MSC_ABI_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VmMscUavArgs {
    device uint *uav [[id(0)]];
};
kernel void vm_msc_uav_affine(constant VmMscUavArgs &args [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
    args.uav[tid] = tid * 3u + 7u;
}
"#;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            let path = std::env::temp_dir()
                .join(format!("victory-{label}-{}-{stamp}", std::process::id()));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tool_path(name: &str) -> Option<PathBuf> {
        let output = ProcessCommand::new("xcrun")
            .args(["-f", name])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        let text = text.trim();
        (!text.is_empty()).then(|| PathBuf::from(text))
    }

    /// Compiles the original synthetic source into a `.metallib` with Apple's
    /// offline Metal toolchain. Returns `None` only when the toolchain is not
    /// installed, so a host without Xcode skips instead of pretending to pass.
    fn compile_fixture(dir: &Path) -> Option<PathBuf> {
        compile_source(dir, "victory_metallib_fixture", METALLIB_SOURCE)
    }

    /// Compiles one synthetic MSL source into a `.metallib` with Apple's offline
    /// Metal toolchain, returning `None` when `metal`/`metallib` are unavailable.
    fn compile_source(dir: &Path, stem: &str, source_text: &str) -> Option<PathBuf> {
        let metal = tool_path("metal")?;
        let metallib = tool_path("metallib")?;
        let source = dir.join(format!("{stem}.metal"));
        let air = dir.join(format!("{stem}.air"));
        let library = dir.join(format!("{stem}.metallib"));
        fs::write(&source, source_text).expect("write fixture source");

        let compiled = ProcessCommand::new(metal)
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .output()
            .expect("spawn metal");
        assert!(
            compiled.status.success(),
            "metal -c failed: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );

        let linked = ProcessCommand::new(metallib)
            .arg(&air)
            .arg("-o")
            .arg(&library)
            .output()
            .expect("spawn metallib");
        assert!(
            linked.status.success(),
            "metallib failed: {}",
            String::from_utf8_lossy(&linked.stderr)
        );
        Some(library)
    }

    #[test]
    fn precompiled_metallib_runs_through_wire_batch() {
        let dir = TempDir::new("metallib-e2e");
        let Some(library) = compile_fixture(dir.path()) else {
            eprintln!(
                "skipping precompiled metallib e2e: Apple Metal toolchain (metal/metallib) not found"
            );
            return;
        };
        let metallib = fs::read(&library).expect("read compiled metallib");
        assert!(!metallib.is_empty());

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const BYTES: usize = 65536;
        const COUNT: usize = BYTES / 4;
        const ALLOCATION_ID: u64 = 1;
        const GENERATION: u32 = 1;
        const SEQUENCE: u32 = 1;

        let transport = DirectMetalTransport::new().expect("Metal device should open");

        // CPU-side argument validation is independent of any GPU work.
        assert!(matches!(
            transport.load_metallib(&[], "vm_metallib_affine"),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            transport.load_metallib(&metallib, ""),
            Err(MetalError::InvalidArg(_))
        ));

        let pipeline = transport
            .load_metallib(&metallib, "vm_metallib_affine")
            .expect("precompiled metallib should load and register its entry point");
        assert_ne!(pipeline.0, 0);
        assert!(matches!(
            transport.load_metallib(&metallib, "vm_metallib_missing"),
            Err(MetalError::NotFound(_))
        ));

        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // SAFETY: `[raw, raw + BYTES)` is freshly allocated and u32-aligned.
        let host = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), COUNT) };
        for (index, slot) in host.iter_mut().enumerate() {
            *slot = index as u32 + 1;
        }

        // SAFETY: `raw` points to 64 KiB valid memory that outlives `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let submission = Submission {
            sequence: SEQUENCE,
            commands: vec![Command::Dispatch1d(Dispatch1d {
                pipeline_id: pipeline.0,
                threads: COUNT as u32,
                threads_per_group: 64,
                bindings: vec![BufferBinding {
                    allocation_id: ALLOCATION_ID,
                    generation: GENERATION,
                    offset: 0,
                    length: BYTES as u64,
                }],
            })],
        };

        // Round-trip the real wire codec, then execute the decoded submission.
        let frame = wire::encode_submission(&submission).expect("submission should encode");
        let decoded = wire::decode_submission(&frame).expect("submission should decode");
        assert_eq!(decoded, submission);

        let mut allocations = HashMap::new();
        allocations.insert((ALLOCATION_ID, GENERATION), buf_id);

        let reply = transport
            .submit_batch(&decoded, &allocations)
            .expect("wire batch should complete");
        assert_eq!(
            reply,
            Reply {
                sequence: SEQUENCE,
                status: ReplyStatus::Ok,
            }
        );

        for (index, &value) in host.iter().enumerate() {
            let want = (index as u32 + 1) * 3 + 7;
            assert_eq!(value, want, "mismatch at dword {index}");
        }

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "precompiled metallib e2e: {} bytes, entry=vm_metallib_affine, verified_words={}, device={}",
            metallib.len(),
            COUNT,
            device.name_str()
        );

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");
    }

    /// Hardware proof for [`DirectMetalTransport::dispatch_msc_uav_1d`].
    ///
    /// Not part of the default suite: the ARM64 GPU is a serialized resource
    /// owned by the IPC/shader and mesh gates. When it is free, set
    /// `VICTORY_MSC_METALLIB` to a metallib converted from the synthetic
    /// `RootFlags(0), UAV(u0)` HLSL (entry `main`, `gOut[tid] = tid * 3 + 7`)
    /// to test the real converter output. With the variable unset the test
    /// compiles an ABI-equivalent MSL fixture instead, so it still exercises the
    /// binder and residency declarations without an x86_64 converter process.
    #[test]
    #[ignore = "GPU hardware test; run only when the native ARM64 Metal device is free"]
    fn msc_uav_argument_buffer_hardware_dispatch() {
        const ALIGN: usize = 16384;
        const BYTES: usize = 65536;
        const COUNT: usize = BYTES / 4;

        let dir = TempDir::new("msc-uav-e2e");
        let (metallib, entry) = match std::env::var_os("VICTORY_MSC_METALLIB") {
            Some(path) => (fs::read(&path).expect("read VICTORY_MSC_METALLIB"), "main"),
            None => {
                let Some(library) =
                    compile_source(dir.path(), "victory_msc_abi_fixture", MSC_ABI_SOURCE)
                else {
                    eprintln!(
                        "skipping MSC argument-buffer hardware dispatch: \
                         VICTORY_MSC_METALLIB unset and Apple Metal toolchain not found"
                    );
                    return;
                };
                (
                    fs::read(&library).expect("read compiled metallib"),
                    "vm_msc_uav_affine",
                )
            }
        };
        assert!(!metallib.is_empty());

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        let transport = DirectMetalTransport::new().expect("Metal device should open");

        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // SAFETY: `[raw, raw + BYTES)` is freshly allocated and u32-aligned.
        let host = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), COUNT) };
        host.fill(0);

        // SAFETY: `raw` points to 64 KiB valid memory that outlives `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let pipeline = transport
            .load_metallib(&metallib, entry)
            .expect("MSC metallib should load and register its entry point");

        let request = MscUavDispatch {
            pipeline,
            uav: buf_id,
            offset: 0,
            length: BYTES as u64,
            threads: COUNT as u32,
            threads_per_group: 64,
            layout: MscUavLayout::DEFAULT,
        };
        transport
            .dispatch_msc_uav_1d(&request)
            .expect("MSC argument-buffer dispatch should complete");

        for (index, &value) in host.iter().enumerate() {
            assert_eq!(
                value,
                index as u32 * 3 + 7,
                "mismatch at dword {index} (entry={entry})"
            );
        }

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");

        eprintln!(
            "MSC argument-buffer hardware dispatch: entry={entry}, verified_words={COUNT}, \
             slot={}, uav_offset={}",
            MscUavLayout::DEFAULT.argument_buffer_index(),
            MscUavLayout::DEFAULT.address_offset()
        );
    }
}
