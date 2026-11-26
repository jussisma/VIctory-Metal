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
use crate::macos::render::{
    GraphicsPipeline, IndexType, LoadAction, PixelFormat, PrimitiveType, RenderAttachment,
    RenderEncoder, RenderPass, RenderTarget,
};
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

/// Edge of the lazily created host render target, in pixels.
///
/// The executor renders every graphics submission into this single internal
/// 64x64 RGBA8 target; it is a proof surface for the native path, not a
/// swapchain. Presentation to a window/view is handled by `views`/`present`.
const HOST_TARGET_DIMENSION: u32 = 64;

/// Byte width of the index values consumed by wire `DrawIndexed` commands.
///
/// The wire frame carries no `DXGI_FORMAT` for the index buffer, so the native
/// executor fixes one convention: every `DrawIndexed` index is a 32-bit
/// unsigned integer (`MTLIndexTypeUint32`). A guest that records 16-bit indices
/// must widen them before submitting.
const GRAPHICS_INDEX_SIZE: usize = 4;

/// Largest inline root-constant payload the render encoder accepts.
///
/// This mirrors `MTLCommandBuffer.setVertexBytes:length:atIndex:` (4 KiB).
/// The wire decoder independently caps a decoded `RootConstants` command at
/// [`wire::MAX_ROOT_CONSTANT_BYTES`] (256 bytes); this executor also accepts a
/// programmatically built submission up to the Metal limit.
const MAX_GRAPHICS_ROOT_CONSTANT_BYTES: usize = 4096;

pub(crate) struct RegisteredBuffer {
    pub(crate) buffer: ObjcObject,
    /// Byte length reported to Metal at registration time.
    pub(crate) len: usize,
}

pub(crate) struct DeviceState {
    next_id: u64,
    pub(crate) buffers: HashMap<u64, RegisteredBuffer>,
    /// Compute pipeline states. Compute commands (`Dispatch1d`/`Dispatch3d`)
    /// resolve here.
    pipelines: HashMap<u64, ObjcObject>,
    /// Graphics pipeline states. Graphics commands (`SetPipeline`, draws)
    /// resolve here.
    ///
    /// Both maps share [`DeviceState::alloc_id`], so an id is globally unique
    /// across compute and graphics pipelines: a compute id never names a
    /// graphics pipeline and vice versa. Because the id is unique, a single
    /// lookup order is sufficient — compute commands consult `pipelines` only,
    /// graphics commands consult `render_pipelines` only. Neither lookup can
    /// fall through to the other table and silently accept a mismatched id.
    render_pipelines: HashMap<u64, GraphicsPipeline>,
    /// Lazily created 64x64 RGBA8 host render target for graphics batches.
    host_target: Option<RenderTarget>,
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

/// One resolved vertex-buffer binding `(MTLBuffer, byte offset, vertex slot)`.
type BoundVertex = (objc::Id, usize, usize);

/// One validated indexed draw, fully resolved before any command buffer exists.
struct PreparedDrawIndexed {
    index_count: usize,
    index_buffer: objc::Id,
    index_offset: usize,
    base_vertex: isize,
    /// Bindings from wire slots `1..`; `bindings[1]` is vertex slot 0.
    vertex_bindings: Vec<BoundVertex>,
}

/// One validated non-indexed draw, fully resolved before any command buffer.
struct PreparedDraw {
    vertex_count: usize,
    vertex_start: usize,
    /// Wire binding `i` binds vertex slot `i`.
    vertex_bindings: Vec<BoundVertex>,
}

/// A validated graphics operation, ready to encode.
enum PreparedGraphicsOp {
    /// Inline vertex-stage constants at `slot`.
    RootConstants { slot: usize, bytes: Vec<u8> },
    /// One explicit vertex-buffer binding (from a `DescriptorTable`).
    BindVertex(BoundVertex),
    /// Indexed draw; binds its vertex buffers, then draws.
    DrawIndexed(PreparedDrawIndexed),
    /// Non-indexed draw.
    Draw(PreparedDraw),
}

/// Consecutive graphics operations sharing one pipeline and one render pass.
///
/// A `ResourceBarrier` or a pipeline change closes a segment. Each segment is a
/// separate pass (and therefore command buffer) over the same host target: the
/// first segment clears, later segments load. Metal's command-buffer ordering
/// then makes the segments strictly ordered, which is at least as strong as the
/// hazard tracking a single pass would provide.
struct PreparedGraphicsSegment {
    pipeline_id: u64,
    ops: Vec<PreparedGraphicsOp>,
}

/// Resolves and validates every barrier entry against one registry snapshot.
///
/// Barrier entries are proven to name a live `(allocation_id, generation)` with
/// an in-bounds range before any command buffer exists. The D3D12 state fields
/// are carried verbatim and are not interpreted: Metal orders work through
/// command-buffer/encoder boundaries, so the transition itself performs no GPU
/// command here.
fn resolve_barrier_entries(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    entries: &[wire::BarrierEntry],
) -> Option<()> {
    for entry in entries {
        if entry.allocation_id == 0
            || entry.generation == 0
            || entry.length == 0
            || entry.length > wire::MAX_RESOURCE_LEN
        {
            return None;
        }
        let registered_id = *allocations.get(&(entry.allocation_id, entry.generation))?;
        let registered = state.buffers.get(&registered_id.0)?;
        let end = usize::try_from(entry.checked_end()?).ok()?;
        if end > registered.len {
            return None;
        }
    }
    Some(())
}

/// Retains the host target's `MTLTexture` so a per-segment [`RenderTarget`] can
/// own a +1 reference independently of the [`DeviceState`] copy.
fn retain_render_target(target: &RenderTarget) -> Result<RenderTarget, MetalError> {
    // SAFETY: `target.as_raw()` is a live, retained `MTLTexture`; `retain`
    // returns that same object with an additional +1 reference.
    let raw = unsafe { objc::msg_send_id_0(target.as_raw(), objc::sel(c"retain")) };
    // SAFETY: `raw` is a +1 retained `MTLTexture`; `from_retained_raw` takes
    // ownership of exactly that reference.
    unsafe { RenderTarget::from_retained_raw(raw) }
}

/// Starts a new prepared segment if the previous one was closed.
fn open_graphics_segment(
    segments: &mut Vec<PreparedGraphicsSegment>,
    segment_open: &mut bool,
    pipeline_id: u64,
) {
    if !*segment_open {
        segments.push(PreparedGraphicsSegment {
            pipeline_id,
            ops: Vec::new(),
        });
        *segment_open = true;
    }
}

/// Reads a private render target back through a shared `MTLBuffer` blit.
#[cfg(test)]
fn read_back_render_target(
    transport: &DirectMetalTransport,
    target: &RenderTarget,
) -> Result<Vec<u8>, MetalError> {
    let width = target.width() as usize;
    let height = target.height() as usize;
    let bytes_per_row = width
        .checked_mul(4)
        .ok_or(MetalError::InvalidArg("render target row length overflows usize"))?;
    let len = bytes_per_row
        .checked_mul(height)
        .ok_or(MetalError::InvalidArg("render target byte length overflows usize"))?;

    let _pool = AutoreleasePool::new();
    // SAFETY: `device_raw()` is a valid `MTLDevice`; storage mode `0` is
    // shared, `len` is nonzero, and the result is retained (+1).
    let buffer_raw = unsafe { objc::msg_send_new_buffer(transport.device_raw(), len, 0) };
    let buffer = ObjcObject::from_retained(buffer_raw).ok_or_else(|| {
        MetalError::InternalError("readback MTLBuffer allocation returned null".to_string())
    })?;
    // SAFETY: `queue_raw()` is a valid `MTLCommandQueue`; the command buffer is
    // autoreleased and lives for this pool.
    let cb =
        unsafe { objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer")) };
    if cb.is_null() {
        return Err(MetalError::InternalError(
            "readback command buffer is null".to_string(),
        ));
    }
    // SAFETY: `cb` is a live uncommitted command buffer.
    let enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder")) };
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "readback blit encoder is null".to_string(),
        ));
    }
    // SAFETY: `enc` is a live blit encoder, `target` is a live level-0 texture
    // and `buffer` can hold exactly one image with this row pitch.
    unsafe {
        msg_send_copy_texture_to_buffer(
            enc,
            target.as_raw(),
            HostOrigin { x: 0, y: 0, z: 0 },
            MTLSize {
                width,
                height,
                depth: 1,
            },
            buffer.as_raw(),
            0,
            bytes_per_row,
        );
        objc::msg_send_void_0(enc, objc::sel(c"endEncoding"));
        objc::msg_send_void_0(cb, objc::sel(c"commit"));
        objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
    }
    // SAFETY: `cb` completed via `waitUntilCompleted`.
    let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
    if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
        // SAFETY: `[cb error]` returns null or an autoreleased `NSError *`.
        let error_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
        // SAFETY: `error_raw` is null or a valid `NSError *`.
        let message = unsafe { objc::nserror_message(error_raw) };
        return Err(MetalError::InternalError(format!(
            "host target readback failed (status={status}): {message}"
        )));
    }
    // SAFETY: `buffer` uses shared storage; `contents` is valid for `len`
    // bytes for the buffer's lifetime.
    let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
    if contents.is_null() {
        return Err(MetalError::InternalError(
            "readback buffer has no CPU-visible contents".to_string(),
        ));
    }
    // SAFETY: `contents` points to `len` readable bytes.
    let bytes = unsafe { std::slice::from_raw_parts(contents.cast::<u8>(), len) };
    Ok(bytes.to_vec())
}

/// Apple `MTLOrigin` (`NSUInteger x, y, z`) for the test readback blit.
#[cfg(test)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HostOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// `-[MTLBlitCommandEncoder copyFromTexture:...toBuffer:...]` for a level-0
/// 2D slice.
#[cfg(test)]
unsafe fn msg_send_copy_texture_to_buffer(
    enc: objc::Id,
    texture: objc::Id,
    origin: HostOrigin,
    size: MTLSize,
    buffer: objc::Id,
    buffer_offset: usize,
    bytes_per_row: usize,
) {
    type Fn = unsafe extern "C" fn(
        objc::Id,
        objc::Sel,
        objc::Id,
        usize,
        usize,
        HostOrigin,
        MTLSize,
        objc::Id,
        usize,
        usize,
        usize,
    );
    // SAFETY: Caller supplies a live blit encoder and live Metal objects; the
    // per-image stride is `0` for a single level-0 image.
    let f: Fn = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            enc,
            objc::sel(c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"),
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            buffer_offset,
            bytes_per_row,
            0,
        )
    }
}

#[cfg(test)]
#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
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
                render_pipelines: HashMap::new(),
                host_target: None,
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

    /// Registers one MSL vertex/fragment pair as a graphics pipeline.
    ///
    /// The library is compiled once and both named entry points are resolved
    /// from it. The returned [`VmPipelineId`] comes from the same allocator as
    /// [`Self::compile_msl`]/[`Self::load_metallib`], so graphics and compute
    /// ids never collide and each command family looks in exactly one table.
    ///
    /// The pipeline is created without a depth attachment (`depth_format`
    /// `None`); the host target used by [`Self::submit_batch`] is colour-only
    /// 64x64 RGBA8.
    pub fn register_graphics_pipeline(
        &self,
        msl: &str,
        vertex_entry: &str,
        fragment_entry: &str,
        format: PixelFormat,
    ) -> Result<VmPipelineId, MetalError> {
        let pipeline =
            GraphicsPipeline::compile(self, msl, vertex_entry, fragment_entry, format, None)?;
        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.render_pipelines.insert(id, pipeline);
        Ok(VmPipelineId(id))
    }

    /// Creates the shared 64x64 RGBA8 host render target on first use.
    fn ensure_host_target(&self) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let mut state = self.lock_state()?;
        if state.host_target.is_none() {
            let target = self.create_render_target_2d(
                HOST_TARGET_DIMENSION,
                HOST_TARGET_DIMENSION,
                PixelFormat::Rgba8Unorm,
            )?;
            state.host_target = Some(target);
        }
        Ok(())
    }

    /// Reads the host render target back as RGBA8 pixels, row-major from the
    /// top-left corner.
    ///
    /// Test-only: a blit copies the private texture into a shared `MTLBuffer`,
    /// then the host maps that buffer. The state lock is held through the
    /// synchronous blit, so the target cannot be re-rendered concurrently.
    #[cfg(test)]
    pub(crate) fn host_target_pixels(&self) -> Result<Vec<[u8; 4]>, MetalError> {
        self.ensure_host_target()?;
        let state = self.lock_state()?;
        let target = state.host_target.as_ref().ok_or_else(|| {
            MetalError::InternalError("host render target was not created".to_string())
        })?;
        let bytes = read_back_render_target(self, target)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect())
    }

    /// Executes a submission that contains only graphics and state commands.
    ///
    /// # Command mapping (documented local conventions)
    ///
    /// The wire frame does not carry a topology, index format or shader stage
    /// for these opcodes. The native executor fixes the following conventions
    /// for the recorded order:
    ///
    /// * `SetPipeline` selects the graphics pipeline used by later state and
    ///   draw commands. A draw must name the same pipeline id.
    /// * `RootConstants { slot, bytes }` binds `bytes` as vertex-stage inline
    ///   constants at vertex buffer index `slot` (`setVertexBytes`), 1..=4096
    ///   bytes.
    /// * `DescriptorTable { base_register, bindings }` binds each resolved
    ///   binding as a vertex buffer at index `base_register + i`.
    /// * `DrawIndexed` uses `bindings[0]` as the index buffer and `bindings[1..]`
    ///   as vertex buffers at vertex indices `0..`. Indices are `u32`
    ///   (`MTLIndexTypeUint32`); `index_buffer_offset` is
    ///   `binding.offset + start_index * 4`. `base_vertex` is forwarded as
    ///   `baseVertex`. Only `instance_count == 1` and `start_instance == 0` are
    ///   encodable and are required; other instancing is refused.
    /// * `Draw` binds `bindings[i]` at vertex index `i` and draws
    ///   `vertex_count` triangles from `start_vertex`.
    /// * `ResourceBarrier` validates every entry, then closes the current
    ///   segment. The next command starts a new pass over the same target with
    ///   a load action, so the two segments cannot overlap.
    /// * `DispatchMesh` is refused: no mesh pipeline can be registered on this
    ///   transport, and a batch never reports success for work that did not run.
    ///
    /// Validation is complete before the first [`RenderEncoder`] (and therefore
    /// the first command buffer) exists. An unknown pipeline or allocation
    /// returns [`ReplyStatus::UnknownHandle`] with no GPU work; structural
    /// violations return [`MetalError`].
    fn execute_graphics_batch(
        &self,
        submission: &Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let reply = |status| Reply {
            sequence: submission.sequence,
            status,
        };
        self.ensure_host_target()?;
        let _pool = AutoreleasePool::new();
        let state = self.lock_state()?;

        let mut segments: Vec<PreparedGraphicsSegment> = Vec::new();
        let mut current_pipeline: Option<u64> = None;
        let mut segment_open = false;

        // Validation pass: resolve every handle/range and build immutable
        // segment data. No Metal command buffer is created here.
        for command in &submission.commands {
            match command {
                Command::SetPipeline(set_pipeline) => {
                    if !state.render_pipelines.contains_key(&set_pipeline.pipeline_id) {
                        return Ok(reply(ReplyStatus::UnknownHandle));
                    }
                    if current_pipeline != Some(set_pipeline.pipeline_id) {
                        current_pipeline = Some(set_pipeline.pipeline_id);
                        segment_open = false;
                    }
                }
                Command::RootConstants(constants) => {
                    let Some(pipeline) = current_pipeline else {
                        return Err(MetalError::InvalidArg(
                            "root constants require a preceding SetPipeline",
                        ));
                    };
                    let slot = usize::from(constants.slot);
                    open_graphics_segment(&mut segments, &mut segment_open, pipeline);
                    segments
                        .last_mut()
                        .expect("segment was just opened")
                        .ops
                        .push(PreparedGraphicsOp::RootConstants {
                            slot,
                            bytes: constants.bytes.clone(),
                        });
                }
                Command::DescriptorTable(table) => {
                    let Some(pipeline) = current_pipeline else {
                        return Err(MetalError::InvalidArg(
                            "a descriptor table requires a preceding SetPipeline",
                        ));
                    };
                    let Some(resolved) =
                        resolve_bindings(&state, allocations, &table.bindings)
                    else {
                        return Ok(reply(ReplyStatus::UnknownHandle));
                    };
                    open_graphics_segment(&mut segments, &mut segment_open, pipeline);
                    let segment = segments.last_mut().expect("segment was just opened");
                    for (offset_index, (buffer, offset, _slot)) in resolved.iter().enumerate() {
                        let index = usize::from(table.base_register)
                            .checked_add(offset_index)
                            .ok_or(MetalError::InvalidArg(
                                "descriptor table vertex index overflows usize",
                            ))?;
                        segment
                            .ops
                            .push(PreparedGraphicsOp::BindVertex((*buffer, *offset, index)));
                    }
                }
                Command::DrawIndexed(draw) => {
                    if let Some(selected) = current_pipeline {
                        if selected != draw.pipeline_id {
                            return Err(MetalError::InvalidArg(
                                "indexed draw pipeline does not match the selected pipeline",
                            ));
                        }
                    } else {
                        if !state.render_pipelines.contains_key(&draw.pipeline_id) {
                            return Ok(reply(ReplyStatus::UnknownHandle));
                        }
                        current_pipeline = Some(draw.pipeline_id);
                        segment_open = false;
                    }
                    let Some(resolved) = resolve_bindings(&state, allocations, &draw.bindings)
                    else {
                        return Ok(reply(ReplyStatus::UnknownHandle));
                    };

                    // `bindings[0]` is the index buffer. The 32-bit index
                    // convention plus `start_index` must fit inside the bound
                    // index range; this is the "one byte short" guard.
                    let start_byte = u64::from(draw.start_index)
                        .checked_mul(GRAPHICS_INDEX_SIZE as u64)
                        .ok_or(MetalError::InvalidArg("start_index overflows u64"))?;
                    let index_window = start_byte
                        .checked_add(
                            u64::from(draw.index_count)
                                .checked_mul(GRAPHICS_INDEX_SIZE as u64)
                                .ok_or(MetalError::InvalidArg(
                                    "index count overflows u64",
                                ))?,
                        )
                        .ok_or(MetalError::InvalidArg("indexed draw range overflows u64"))?;
                    if index_window > draw.bindings[0].length {
                        return Err(MetalError::InvalidArg(
                            "indexed draw range exceeds the index binding length",
                        ));
                    }
                    let start_byte = usize::try_from(start_byte).map_err(|_| {
                        MetalError::InvalidArg("start_index byte offset overflows usize")
                    })?;
                    let index_offset = resolved[0].1.checked_add(start_byte).ok_or(
                        MetalError::InvalidArg("indexed draw index offset overflows usize"),
                    )?;

                    let vertex_bindings: Vec<BoundVertex> = resolved[1..]
                        .iter()
                        .enumerate()
                        .map(|(index, (buffer, offset, _slot))| (*buffer, *offset, index))
                        .collect();
                    open_graphics_segment(&mut segments, &mut segment_open, draw.pipeline_id);
                    segments
                        .last_mut()
                        .expect("segment was just opened")
                        .ops
                        .push(PreparedGraphicsOp::DrawIndexed(PreparedDrawIndexed {
                            index_count: draw.index_count as usize,
                            index_buffer: resolved[0].0,
                            index_offset,
                            base_vertex: draw.base_vertex as isize,
                            vertex_bindings,
                        }));
                }
                Command::Draw(draw) => {
                    if let Some(selected) = current_pipeline {
                        if selected != draw.pipeline_id {
                            return Err(MetalError::InvalidArg(
                                "draw pipeline does not match the selected pipeline",
                            ));
                        }
                    } else {
                        if !state.render_pipelines.contains_key(&draw.pipeline_id) {
                            return Ok(reply(ReplyStatus::UnknownHandle));
                        }
                        current_pipeline = Some(draw.pipeline_id);
                        segment_open = false;
                    }
                    let Some(resolved) = resolve_bindings(&state, allocations, &draw.bindings)
                    else {
                        return Ok(reply(ReplyStatus::UnknownHandle));
                    };
                    let vertex_bindings: Vec<BoundVertex> = resolved
                        .iter()
                        .enumerate()
                        .map(|(index, (buffer, offset, _slot))| (*buffer, *offset, index))
                        .collect();
                    open_graphics_segment(&mut segments, &mut segment_open, draw.pipeline_id);
                    segments
                        .last_mut()
                        .expect("segment was just opened")
                        .ops
                        .push(PreparedGraphicsOp::Draw(PreparedDraw {
                            vertex_count: draw.vertex_count as usize,
                            vertex_start: draw.start_vertex as usize,
                            vertex_bindings,
                        }));
                }
                Command::ResourceBarrier(barrier) => {
                    if resolve_barrier_entries(&state, allocations, &barrier.entries).is_none() {
                        return Ok(reply(ReplyStatus::UnknownHandle));
                    }
                    // The barrier ends the current encoder. The next command
                    // opens a new pass (load action) over the same target.
                    segment_open = false;
                }
                Command::Dispatch1d(_) | Command::Dispatch3d(_) | Command::DispatchMesh(_) => {
                    return Err(MetalError::InternalError(
                        "compute command reached the graphics executor".to_string(),
                    ));
                }
            }
        }

        let target = state.host_target.as_ref().ok_or_else(|| {
            MetalError::InternalError("host render target was not created".to_string())
        })?;

        // Encode pass: every handle above was validated, so this phase only
        // creates command buffers for work that will be executed.
        let mut first_segment = true;
        for segment in &segments {
            let pipeline = state
                .render_pipelines
                .get(&segment.pipeline_id)
                .ok_or_else(|| {
                    MetalError::InternalError(
                        "validated graphics pipeline disappeared".to_string(),
                    )
                })?;
            let attachment = RenderAttachment::color(retain_render_target(target)?);
            let attachment = if first_segment {
                attachment
            } else {
                attachment.with_load_action(LoadAction::Load)
            };
            let pass = RenderPass::new(attachment)?;
            let mut encoder = RenderEncoder::new(self, &pass, pipeline)?;

            for op in &segment.ops {
                match op {
                    PreparedGraphicsOp::RootConstants { slot, bytes } => {
                        encoder.set_vertex_bytes(bytes, *slot)?;
                    }
                    PreparedGraphicsOp::BindVertex((buffer, offset, index)) => {
                        // SAFETY: The buffer comes from the locked allocation
                        // registry and stays registered until this synchronous
                        // submit completes; the range was bounds-checked.
                        unsafe {
                            encoder.set_vertex_buffer(*buffer, *offset, *index)?;
                        }
                    }
                    PreparedGraphicsOp::DrawIndexed(draw) => {
                        for (buffer, offset, index) in &draw.vertex_bindings {
                            // SAFETY: Same registry-held buffer guarantee as
                            // `BindVertex`; offset/index were validated.
                            unsafe {
                                encoder.set_vertex_buffer(*buffer, *offset, *index)?;
                            }
                        }
                        // SAFETY: The index buffer is registry-held for the
                        // whole synchronous submit and the 32-bit index window
                        // was validated against the bound range.
                        unsafe {
                            encoder.draw_indexed_primitives(
                                PrimitiveType::Triangle,
                                draw.index_count,
                                IndexType::Uint32,
                                draw.index_buffer,
                                draw.index_offset,
                                draw.base_vertex,
                            )?;
                        }
                    }
                    PreparedGraphicsOp::Draw(draw) => {
                        for (buffer, offset, index) in &draw.vertex_bindings {
                            // SAFETY: Same registry-held buffer guarantee.
                            unsafe {
                                encoder.set_vertex_buffer(*buffer, *offset, *index)?;
                            }
                        }
                        encoder.draw_primitives(
                            PrimitiveType::Triangle,
                            draw.vertex_start,
                            draw.vertex_count,
                        )?;
                    }
                }
            }
            encoder.finish()?;
            first_segment = false;
        }

        Ok(reply(ReplyStatus::Ok))
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
    /// * A submission is classified as *compute* (only `Dispatch1d`/
    ///   `Dispatch3d`), *graphics* (any of `SetPipeline`, `RootConstants`,
    ///   `DescriptorTable`, `DrawIndexed`, `Draw`, `ResourceBarrier`) or
    ///   *empty*. Graphics and state commands are executed by
    ///   [`Self::execute_graphics_batch`] over the shared 64x64 RGBA8 host
    ///   target. Mixing compute and graphics commands in one submission is
    ///   refused with [`MetalError::NotSupported`]; keeping the families
    ///   separate preserves the atomic "validate everything, then execute"
    ///   contract of each path.
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
        let mut has_compute = false;
        let mut has_graphics = false;
        for command in &submission.commands {
            match command {
                Command::Dispatch1d(dispatch) => {
                    has_compute = true;
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
                    has_compute = true;
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
                Command::DrawIndexed(draw) => {
                    has_graphics = true;
                    if draw.pipeline_id == 0
                        || draw.index_count == 0
                        || draw.instance_count == 0
                    {
                        return Err(MetalError::InvalidArg(
                            "indexed draw pipeline, index and instance counts must be nonzero",
                        ));
                    }
                    if draw.bindings.len() < 2
                        || draw.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH
                    {
                        return Err(MetalError::InvalidArg(
                            "indexed draw needs index and vertex bindings within the wire limit",
                        ));
                    }
                    if draw.instance_count != 1 || draw.start_instance != 0 {
                        return Err(MetalError::NotSupported(
                            "instanced indexed draws are not encoded by the native graphics path",
                        ));
                    }
                }
                Command::Draw(draw) => {
                    has_graphics = true;
                    if draw.pipeline_id == 0 || draw.vertex_count == 0 || draw.instance_count == 0 {
                        return Err(MetalError::InvalidArg(
                            "draw pipeline, vertex and instance counts must be nonzero",
                        ));
                    }
                    if draw.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                        return Err(MetalError::InvalidArg(
                            "draw exceeds wire binding limit",
                        ));
                    }
                    if draw.instance_count != 1 || draw.start_instance != 0 {
                        return Err(MetalError::NotSupported(
                            "instanced draws are not encoded by the native graphics path",
                        ));
                    }
                }
                Command::DispatchMesh(_) => {
                    // No mesh pipeline can be registered on this transport, so
                    // a mesh dispatch never has a validated pipeline to run.
                    // Refuse it instead of silently skipping the command.
                    return Err(MetalError::NotSupported(
                        "mesh-shader pipelines are not registered; DispatchMesh is refused",
                    ));
                }
                Command::ResourceBarrier(barrier) => {
                    has_graphics = true;
                    if barrier.entries.is_empty()
                        || barrier.entries.len() > wire::MAX_BARRIERS_PER_COMMAND
                    {
                        return Err(MetalError::InvalidArg(
                            "resource barrier needs 1..=16 entries",
                        ));
                    }
                    for entry in &barrier.entries {
                        if entry.allocation_id == 0
                            || entry.generation == 0
                            || entry.length == 0
                            || entry.length > wire::MAX_RESOURCE_LEN
                        {
                            return Err(MetalError::InvalidArg(
                                "barrier allocation, generation and length must be nonzero and bounded",
                            ));
                        }
                    }
                }
                Command::SetPipeline(set_pipeline) => {
                    has_graphics = true;
                    if set_pipeline.pipeline_id == 0 {
                        return Err(MetalError::InvalidArg(
                            "set-pipeline handle must be nonzero",
                        ));
                    }
                }
                Command::RootConstants(constants) => {
                    has_graphics = true;
                    if constants.bytes.is_empty()
                        || constants.bytes.len() > MAX_GRAPHICS_ROOT_CONSTANT_BYTES
                    {
                        return Err(MetalError::InvalidArg(
                            "root constants must be 1..=4096 bytes",
                        ));
                    }
                }
                Command::DescriptorTable(table) => {
                    has_graphics = true;
                    if table.bindings.is_empty()
                        || table.bindings.len() > wire::MAX_DESCRIPTOR_BINDINGS
                    {
                        return Err(MetalError::InvalidArg(
                            "descriptor table needs 1..=16 bindings",
                        ));
                    }
                }
            }
        }
        if has_compute && has_graphics {
            return Err(MetalError::NotSupported(
                "a submission cannot mix compute dispatches and graphics commands",
            ));
        }

        let reply = |status| Reply {
            sequence: submission.sequence,
            status,
        };
        if submission.commands.is_empty() {
            return Ok(reply(ReplyStatus::Ok));
        }
        if has_graphics {
            return self.execute_graphics_batch(submission, allocations);
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
                _ => {
                    // The classification pass above routes any submission with
                    // a graphics command to `execute_graphics_batch`.
                    return Err(MetalError::InternalError(
                        "graphics command reached the compute executor".to_string(),
                    ));
                }
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
                _ => {
                    return Err(MetalError::InternalError(
                        "graphics command reached the compute executor".to_string(),
                    ));
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
    use crate::macos::render::PixelFormat;
    use crate::transport::{MetalError, MetalTransport};
    use crate::wire::{
        self, BufferBinding, Command, Dispatch1d, DrawIndexed, Reply, ReplyStatus, RootConstants,
        SetPipeline, Submission,
    };

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

    /// MSL used by the native host-target graphics tests.
    ///
    /// Positions come from the vertex buffer bound at slot 0; the flat colour
    /// comes from vertex root constants at slot 1. A green result therefore
    /// proves the `RootConstants` command was applied and that the vertex
    /// buffer from `DrawIndexed` was bound and indexed.
    const HOST_GFX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float4 color;
};

vertex VOut vm_host_vertex(uint vid [[vertex_id]],
                           constant float2 *positions [[buffer(0)]],
                           constant float4 &color [[buffer(1)]]) {
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.color = color;
    return out;
}

fragment float4 vm_host_fragment(VOut in [[stage_in]]) {
    return in.color;
}
"#;

    const HOST_PAGE_BYTES: usize = 16 * 1024;
    const HOST_INDEX_OFFSET: usize = 0;
    const HOST_VERTEX_OFFSET: usize = 16;
    const HOST_INDEX_BYTES: usize = 3 * 4;
    const HOST_VERTEX_BYTES: usize = 6 * 4;
    const HOST_TARGET_PIXELS: usize = 64 * 64;
    const HOST_CENTRE_PIXEL: usize = 32 * 64 + 32;
    const HOST_GREEN_RGBA: [u8; 4] = [0, 255, 0, 255];

    unsafe extern "C" {
        fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        fn free(ptr: *mut c_void);
    }

    /// One 16 KiB page holding `[u32; 3]` indices at offset 0 and three
    /// `float2` positions at offset 16.
    struct HostPage(*mut c_void);

    impl HostPage {
        fn new() -> Self {
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation
            // of `HOST_PAGE_BYTES` bytes into `raw`.
            let rc = unsafe { posix_memalign(&mut raw, 16384, HOST_PAGE_BYTES) };
            assert_eq!(rc, 0, "posix_memalign failed");
            assert!(!raw.is_null());
            let page = Self(raw);

            // SAFETY: `raw` is freshly allocated and writable for
            // `HOST_PAGE_BYTES` bytes; the slice does not outlive `page`.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(raw.cast::<u8>(), HOST_PAGE_BYTES)
            };
            bytes.fill(0);
            for (index, value) in [0u32, 1, 2].iter().enumerate() {
                let at = HOST_INDEX_OFFSET + index * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            for (index, value) in positions.iter().enumerate() {
                let at = HOST_VERTEX_OFFSET + index * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            page
        }

        fn raw(&self) -> *mut u8 {
            self.0.cast()
        }
    }

    impl Drop for HostPage {
        fn drop(&mut self) {
            // SAFETY: `self.0` came from `posix_memalign` and is no longer
            // registered with the transport when this guard drops.
            unsafe { free(self.0) };
        }
    }

    /// `[SetPipeline, RootConstants, DrawIndexed]`, the recorded graphics list
    /// used by the GPU tests.
    fn host_graphics_submission(pipeline_id: u64) -> Submission {
        let mut green = Vec::new();
        for value in [0.0f32, 1.0, 0.0, 1.0] {
            green.extend_from_slice(&value.to_le_bytes());
        }
        Submission {
            sequence: 1,
            commands: vec![
                Command::SetPipeline(SetPipeline { pipeline_id }),
                Command::RootConstants(RootConstants {
                    slot: 1,
                    bytes: green,
                }),
                Command::DrawIndexed(DrawIndexed {
                    pipeline_id,
                    index_count: 3,
                    instance_count: 1,
                    start_index: 0,
                    base_vertex: 0,
                    start_instance: 0,
                    list_id: 1,
                    list_generation: 1,
                    bindings: vec![
                        BufferBinding {
                            allocation_id: 1,
                            generation: 1,
                            offset: HOST_INDEX_OFFSET as u64,
                            length: HOST_INDEX_BYTES as u64,
                        },
                        BufferBinding {
                            allocation_id: 1,
                            generation: 1,
                            offset: HOST_VERTEX_OFFSET as u64,
                            length: HOST_VERTEX_BYTES as u64,
                        },
                    ],
                }),
            ],
        }
    }

    fn host_graphics_pipeline(transport: &DirectMetalTransport) -> crate::abi::VmPipelineId {
        transport
            .register_graphics_pipeline(
                HOST_GFX_MSL,
                "vm_host_vertex",
                "vm_host_fragment",
                PixelFormat::Rgba8Unorm,
            )
            .expect("graphics pipeline should register")
    }

    /// Real GPU proof: a registered graphics pipeline, three vertices and three
    /// u32 indices uploaded through a registered allocation, executed as
    /// `[SetPipeline, RootConstants, DrawIndexed]` into the host target.
    #[test]
    fn graphics_batch_draw_indexed_writes_centre_pixel() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let pipeline = host_graphics_pipeline(&transport);
        assert_ne!(pipeline.0, 0);

        let page = HostPage::new();
        // SAFETY: `page` points to 16 KiB that outlives the registration and
        // every submission in this test.
        let buffer = unsafe { transport.register_zerocopy(page.raw(), HOST_PAGE_BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");
        let mut allocations = HashMap::new();
        allocations.insert((1u64, 1u32), buffer);

        let submission = host_graphics_submission(pipeline.0);
        let frame = wire::encode_submission(&submission).expect("graphics submission should encode");
        let decoded =
            wire::decode_submission(&frame).expect("graphics submission should decode");
        assert_eq!(decoded, submission);

        let reply = transport
            .submit_batch(&decoded, &allocations)
            .expect("graphics batch should complete");
        assert_eq!(
            reply,
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );

        let pixels = transport
            .host_target_pixels()
            .expect("host target should read back");
        assert_eq!(pixels.len(), HOST_TARGET_PIXELS);
        assert_eq!(
            pixels[HOST_CENTRE_PIXEL], HOST_GREEN_RGBA,
            "centre pixel should carry the root-constant green"
        );
        assert!(
            pixels.iter().all(|pixel| *pixel == HOST_GREEN_RGBA),
            "the full-screen indexed triangle should cover every pixel"
        );

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "graphics batch: device={}, pipeline={}, pixels={}, centre={:?}, corner={:?}",
            device.name_str(),
            pipeline.0,
            pixels.len(),
            pixels[HOST_CENTRE_PIXEL],
            pixels[0]
        );

        transport
            .unregister_buffer(buffer)
            .expect("unregister_buffer should succeed");
    }

    /// An unknown graphics pipeline is rejected with `UnknownHandle` and no
    /// command buffer, so the host target keeps its pre-submit contents.
    #[test]
    fn graphics_batch_unknown_pipeline_leaves_target_unchanged() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let before = transport
            .host_target_pixels()
            .expect("host target should read back before the submission");

        let unknown = Submission {
            sequence: 9,
            commands: vec![Command::SetPipeline(SetPipeline {
                pipeline_id: 999_999,
            })],
        };
        let reply = transport
            .submit_batch(&unknown, &HashMap::new())
            .expect("an unknown pipeline should still return a reply");
        assert_eq!(
            reply,
            Reply {
                sequence: 9,
                status: ReplyStatus::UnknownHandle,
            }
        );

        let after = transport
            .host_target_pixels()
            .expect("host target should read back after the submission");
        assert_eq!(
            before, after,
            "an unknown pipeline must not modify the host target"
        );
        eprintln!(
            "graphics unknown pipeline: pixels={} unchanged, first={:?}",
            after.len(),
            after[0]
        );
    }

    /// An index binding one byte shorter than `index_count * 4` is refused
    /// during validation, before any command buffer exists.
    #[test]
    fn graphics_batch_short_index_binding_is_refused_before_encoding() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let pipeline = host_graphics_pipeline(&transport);

        let page = HostPage::new();
        // SAFETY: `page` outlives the registration and the submission below.
        let buffer = unsafe { transport.register_zerocopy(page.raw(), HOST_PAGE_BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");
        let mut allocations = HashMap::new();
        allocations.insert((1u64, 1u32), buffer);

        let before = transport
            .host_target_pixels()
            .expect("host target should read back before the submission");

        let mut submission = host_graphics_submission(pipeline.0);
        let Command::DrawIndexed(draw) = &mut submission.commands[2] else {
            panic!("expected a DrawIndexed command");
        };
        draw.bindings[0].length = (HOST_INDEX_BYTES - 1) as u64;

        let error = transport
            .submit_batch(&submission, &allocations)
            .expect_err("a one-byte-short index binding must be refused");
        assert!(
            matches!(error, MetalError::InvalidArg(_)),
            "unexpected short-index error: {error:?}"
        );

        let after = transport
            .host_target_pixels()
            .expect("host target should read back after the submission");
        assert_eq!(
            before, after,
            "a refused indexed draw must not create or run a command buffer"
        );
        eprintln!("graphics short index binding refused: {error}");

        transport
            .unregister_buffer(buffer)
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
