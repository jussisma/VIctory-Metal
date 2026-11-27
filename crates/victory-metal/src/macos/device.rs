//! Direct in-process macOS Metal implementation of [`MetalTransport`].
//!
//! Used both by native macOS callers (Phase 2, Wine-free) and behind `c_api`
//! for Phase 1 Wine unixlib bridges. Zero Wine or Windows code lives here.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, c_void};
use std::ops::Range;
use std::sync::Mutex;

use crate::abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmPipelineId};
use crate::macos::irconverter;
use crate::macos::mesh::{MeshPipeline, MeshPipelineSpec};
use crate::macos::msc::{self, MscUavDispatch};
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::render::{
    GraphicsPipeline, IndexType, LoadAction, MTL_BARRIER_SCOPE_BUFFERS, MTL_BARRIER_SCOPE_TEXTURES,
    PixelFormat, PrimitiveType, RawColorAttachment, RawDepthAttachment, RenderAttachment,
    RenderEncoder, RenderPass, RenderTarget, ScissorRect as RenderScissorRect, SegmentKind,
    StoreAction, Viewport as RenderViewport, build_mrt_render_pass_descriptor,
    plan_encoder_segments, texture_info,
};
use crate::macos::texture::VmTextureId;
use crate::macos::{blit, indirect};
use crate::transport::{MetalError, MetalTransport};
use crate::wire::{self, Command, Reply, ReplyStatus, Submission};
use crate::wire_v3::{
    BeginRenderPass, CommandV3, DescriptorKind, IndexFormat, LoadOp, PrimitiveTopology, StoreOp,
    SubmissionV3,
};

/// How many executed list identities the v3 entry point remembers.
///
/// Mirrors [`crate::cmdlist::ListLedger`]: one identity executes at most once,
/// and the history is bounded so a long session cannot grow without limit.
const V3_EXECUTED_LIST_CAPACITY: usize = 256;

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
    /// Mesh render pipeline states. v3 `SetMeshPipeline` and `DispatchMesh`
    /// resolve here; ids share [`DeviceState::alloc_id`] with every other
    /// pipeline family.
    mesh_pipelines: HashMap<u64, MeshPipeline>,
    /// Lazily created 64x64 RGBA8 host render target for graphics batches.
    host_target: Option<RenderTarget>,
    pub(crate) textures: HashMap<u64, ObjcObject>,
    /// Swapchain surfaces handed to v3 `Present`, keyed by wire surface handle.
    pub(crate) surfaces: HashMap<u64, ObjcObject>,
    /// v3 list identities already executed or executing.
    executed_lists: HashSet<(u64, u32)>,
    executed_list_order: VecDeque<(u64, u32)>,
}

impl DeviceState {
    /// Records a v3 list identity as executed, evicting the oldest entry.
    fn mark_list_executed(&mut self, list_id: u64, list_generation: u32) {
        let identity = (list_id, list_generation);
        if self.executed_lists.insert(identity) {
            self.executed_list_order.push_back(identity);
        }
        while self.executed_list_order.len() > V3_EXECUTED_LIST_CAPACITY {
            if let Some(oldest) = self.executed_list_order.pop_front() {
                self.executed_lists.remove(&oldest);
            }
        }
    }

    fn list_was_executed(&self, list_id: u64, list_generation: u32) -> bool {
        self.executed_lists.contains(&(list_id, list_generation))
    }
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
///
/// Public because a host-side diagnostic or the frame gate needs to observe
/// what was drawn without a second render path.
fn read_back_render_target(
    transport: &DirectMetalTransport,
    target: &RenderTarget,
) -> Result<Vec<u8>, MetalError> {
    let width = target.width() as usize;
    let height = target.height() as usize;
    let bytes_per_row = width.checked_mul(4).ok_or(MetalError::InvalidArg(
        "render target row length overflows usize",
    ))?;
    let len = bytes_per_row
        .checked_mul(height)
        .ok_or(MetalError::InvalidArg(
            "render target byte length overflows usize",
        ))?;

    let _pool = AutoreleasePool::new();
    // SAFETY: `device_raw()` is a valid `MTLDevice`; storage mode `0` is
    // shared, `len` is nonzero, and the result is retained (+1).
    let buffer_raw = unsafe { objc::msg_send_new_buffer(transport.device_raw(), len, 0) };
    let buffer = ObjcObject::from_retained(buffer_raw).ok_or_else(|| {
        MetalError::InternalError("readback MTLBuffer allocation returned null".to_string())
    })?;
    // SAFETY: `queue_raw()` is a valid `MTLCommandQueue`; the command buffer is
    // autoreleased and lives for this pool.
    let cb = unsafe { objc::msg_send_id_0(transport.queue_raw(), objc::sel(c"commandBuffer")) };
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

/// Apple `MTLOrigin` (`NSUInteger x, y, z`) for the readback blit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HostOrigin {
    x: usize,
    y: usize,
    z: usize,
}

/// `-[MTLBlitCommandEncoder copyFromTexture:...toBuffer:...]` for a level-0
/// 2D slice.
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
        let queue_raw =
            unsafe { objc::msg_send_id_0(device.as_raw(), objc::sel(c"newCommandQueue")) };
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
                mesh_pipelines: HashMap::new(),
                host_target: None,
                textures: HashMap::new(),
                surfaces: HashMap::new(),
                executed_lists: HashSet::new(),
                executed_list_order: VecDeque::new(),
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
        let entry_ns = objc::nsstring(entry).ok_or(MetalError::InvalidArg(
            "entry name string allocation failed",
        ))?;

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

    /// Registers an already-compiled graphics pipeline state.
    ///
    /// This is the MRT/depth counterpart of [`Self::register_graphics_pipeline`]:
    /// the caller compiles with [`GraphicsPipeline::compile_mrt`] so several
    /// colour formats and a depth format are part of the pipeline, then hands
    /// the state to the same table a v3 `SetGraphicsPipeline` resolves.
    pub fn register_graphics_pipeline_state(
        &self,
        pipeline: GraphicsPipeline,
    ) -> Result<VmPipelineId, MetalError> {
        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.render_pipelines.insert(id, pipeline);
        Ok(VmPipelineId(id))
    }

    /// Registers an already-compiled mesh pipeline state.
    ///
    /// The returned [`VmPipelineId`] is what a v3 `SetMeshPipeline` names and
    /// what its following `DispatchMesh` must repeat. Mesh ids share the same
    /// allocator as compute and graphics pipelines, so a handle from one family
    /// can never resolve in another.
    pub fn register_mesh_pipeline(
        &self,
        pipeline: MeshPipeline,
    ) -> Result<VmPipelineId, MetalError> {
        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.mesh_pipelines.insert(id, pipeline);
        Ok(VmPipelineId(id))
    }

    /// Compiles and registers a mesh pipeline from MSL source.
    ///
    /// The fragment stage is required; the object stage is optional. Budgets
    /// and attachment formats are validated by [`MeshPipelineSpec`] before any
    /// library or pipeline object is created.
    pub fn register_mesh_pipeline_msl(
        &self,
        source: &str,
        mesh_entry: &str,
        object_entry: Option<&str>,
        fragment_entry: &str,
        spec: &MeshPipelineSpec,
    ) -> Result<VmPipelineId, MetalError> {
        let pipeline =
            MeshPipeline::compile_msl(self, source, mesh_entry, object_entry, fragment_entry, spec)?;
        self.register_mesh_pipeline(pipeline)
    }

    /// Loads precompiled mesh/object/fragment metallibs and registers the
    /// resulting mesh pipeline.
    ///
    /// This is the native bridge for bytes produced by the Rosetta IRConverter
    /// worker, mirroring [`Self::load_metallib`] for compute.
    #[allow(clippy::too_many_arguments)]
    pub fn register_mesh_pipeline_metallibs(
        &self,
        mesh_library: &[u8],
        mesh_entry: &str,
        object_library: Option<(&[u8], &str)>,
        fragment_library: &[u8],
        fragment_entry: &str,
        spec: &MeshPipelineSpec,
    ) -> Result<VmPipelineId, MetalError> {
        let pipeline = MeshPipeline::from_metallibs(
            self,
            mesh_library,
            mesh_entry,
            object_library,
            fragment_library,
            fragment_entry,
            spec,
        )?;
        self.register_mesh_pipeline(pipeline)
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
    /// Returns the pixels of the shared host render target in row-major
    /// `[r, g, b, a]` order after the last executed graphics list.
    ///
    /// This is a diagnostic readback: it blits the private texture to a
    /// shared buffer and waits. It is never part of the present path.
    pub fn host_target_pixels(&self) -> Result<Vec<[u8; 4]>, MetalError> {
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

    /// Reads a colour render target back as RGBA8 pixels, row-major from the
    /// top-left corner.
    ///
    /// Diagnostic/test helper: it blits the private texture into a shared
    /// buffer and waits, holding the state lock so no other submission can use
    /// the target concurrently. It is never part of the present path.
    pub fn read_render_target_rgba8(
        &self,
        target: &RenderTarget,
    ) -> Result<Vec<[u8; 4]>, MetalError> {
        if !target.pixel_format().is_color() {
            return Err(MetalError::InvalidArg(
                "readback target must use a colour pixel format",
            ));
        }
        let _state = self.lock_state()?;
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
                    if !state
                        .render_pipelines
                        .contains_key(&set_pipeline.pipeline_id)
                    {
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
                    let Some(resolved) = resolve_bindings(&state, allocations, &table.bindings)
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
                                .ok_or(MetalError::InvalidArg("index count overflows u64"))?,
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
                    let index_offset =
                        resolved[0]
                            .1
                            .checked_add(start_byte)
                            .ok_or(MetalError::InvalidArg(
                                "indexed draw index offset overflows usize",
                            ))?;

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
                    MetalError::InternalError("validated graphics pipeline disappeared".to_string())
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
                    if draw.pipeline_id == 0 || draw.index_count == 0 || draw.instance_count == 0 {
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
                        return Err(MetalError::InvalidArg("draw exceeds wire binding limit"));
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

/// Diagnostic encoder inventory for one version 3 submission.
///
/// The counts are observed while encoding, so a test can assert that a mixed
/// command stream really produced one command buffer and one encoder per
/// contiguous segment instead of trusting documentation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncoderStats {
    /// Command buffers created for the submission (0 for an empty plan, else 1).
    pub command_buffers: u32,
    /// `MTLComputeCommandEncoder`s opened.
    pub compute_encoders: u32,
    /// `MTLRenderCommandEncoder`s opened.
    pub render_encoders: u32,
    /// `MTLBlitCommandEncoder`s opened.
    pub blit_encoders: u32,
    /// Explicit `ResourceBarrier` commands executed.
    pub barriers: u32,
}

/// Why preparing a v3 submission stopped before any command buffer existed.
enum PrepareAbort {
    /// A referenced pipeline, allocation, generation or surface is unknown.
    UnknownHandle,
    /// The stream is structurally invalid or uses an unsupported feature.
    Error(MetalError),
}

impl From<MetalError> for PrepareAbort {
    fn from(error: MetalError) -> Self {
        Self::Error(error)
    }
}

/// Thread grid of one prepared compute dispatch.
enum PreparedGrid {
    OneD {
        threads: usize,
        threads_per_group: usize,
    },
    ThreeD {
        group_counts: MTLSize,
        threads_per_group: MTLSize,
    },
}

/// One resolved v3 compute dispatch.
struct PreparedV3Dispatch {
    pipeline: objc::Id,
    grid: PreparedGrid,
    bindings: Vec<BoundRange>,
}

/// One validated compute-segment command.
enum PreparedComputeOp {
    Pipeline(objc::Id),
    Bytes { slot: usize, bytes: Vec<u8> },
    Bind(Vec<BoundRange>),
    Dispatch(PreparedV3Dispatch),
    /// One GPU-driven dispatch; the grid is read by the GPU.
    DispatchIndirect {
        pipeline: objc::Id,
        threads_per_group: MTLSize,
        arguments: indirect::IndirectArguments,
    },
    Barrier,
}

/// One validated render-segment command.
enum PreparedRenderOp {
    Pipeline {
        state: objc::Id,
        depth_state: Option<objc::Id>,
    },
    Viewport(RenderViewport),
    Scissor(RenderScissorRect),
    VertexBuffers(Vec<BoundVertex>),
    VertexBytes {
        slot: usize,
        bytes: Vec<u8>,
    },
    Barrier,
    Draw {
        primitive: PrimitiveType,
        vertex_start: usize,
        vertex_count: usize,
        instance_count: usize,
        base_instance: usize,
    },
    DrawIndexed {
        primitive: PrimitiveType,
        index_count: usize,
        index_type: IndexType,
        index_buffer: objc::Id,
        index_offset: usize,
        base_vertex: isize,
        instance_count: usize,
        base_instance: usize,
    },
    /// One mesh draw with the per-stage threadgroup sizes recorded on the
    /// registered mesh pipeline.
    DispatchMesh {
        group_counts: MTLSize,
        threads_per_object_threadgroup: MTLSize,
        threads_per_mesh_threadgroup: MTLSize,
        bindings: Vec<BoundRange>,
    },
    /// GPU-driven non-indexed instanced draw.
    DrawIndirect {
        primitive: PrimitiveType,
        arguments: indirect::IndirectArguments,
    },
    /// GPU-driven indexed instanced draw.
    DrawIndexedIndirect {
        primitive: PrimitiveType,
        index_type: IndexType,
        index_buffer: objc::Id,
        index_offset: usize,
        arguments: indirect::IndirectArguments,
    },
    /// GPU-driven mesh-shader dispatch.
    DispatchMeshIndirect {
        threads_per_object_threadgroup: MTLSize,
        threads_per_mesh_threadgroup: MTLSize,
        arguments: indirect::IndirectArguments,
    },
}

impl PreparedRenderOp {
    /// Whether this operation needs the indirect-aware render executor.
    fn is_indirect(&self) -> bool {
        matches!(
            self,
            Self::DrawIndirect { .. }
                | Self::DrawIndexedIndirect { .. }
                | Self::DispatchMeshIndirect { .. }
        )
    }
}

/// One validated blit-segment command.
enum PreparedBlitOp {
    CopyBuffer {
        src: objc::Id,
        src_offset: usize,
        dst: objc::Id,
        dst_offset: usize,
        length: usize,
    },
    CopyTexture {
        src: objc::Id,
        src_slice: usize,
        src_level: usize,
        dst: objc::Id,
        dst_slice: usize,
        dst_level: usize,
        width: usize,
        height: usize,
        depth: usize,
    },
    /// Explicit texel-box copy with independent origins.
    CopyTextureRegion(blit::TextureRegion),
    /// Multisample resolve into a single-sample subresource.
    ResolveSubresource {
        src: blit::TextureSubresourceRef,
        dst: blit::TextureSubresourceRef,
    },
    /// A barrier between copies; the blit encoder boundary is the fence.
    Barrier,
}

/// One prepared encoder segment. Descriptors and raw handles were resolved
/// against the registry snapshot before the command buffer existed.
enum PreparedSegment {
    Compute(Vec<PreparedComputeOp>),
    Render {
        descriptor: objc::Id,
        width: u32,
        height: u32,
        ops: Vec<PreparedRenderOp>,
    },
    Blit(Vec<PreparedBlitOp>),
}

fn to_usize(value: u64) -> Result<usize, MetalError> {
    usize::try_from(value).map_err(|_| MetalError::InvalidArg("byte offset exceeds host usize"))
}

fn map_load_op(load: LoadOp) -> LoadAction {
    match load {
        LoadOp::Load => LoadAction::Load,
        LoadOp::Clear => LoadAction::Clear,
        LoadOp::DontCare => LoadAction::DontCare,
    }
}

fn map_store_op(store: StoreOp) -> StoreAction {
    match store {
        StoreOp::Store => StoreAction::Store,
        StoreOp::DontCare => StoreAction::DontCare,
    }
}

fn map_topology(topology: PrimitiveTopology) -> PrimitiveType {
    match topology {
        PrimitiveTopology::PointList => PrimitiveType::Point,
        PrimitiveTopology::LineList => PrimitiveType::Line,
        PrimitiveTopology::LineStrip => PrimitiveType::LineStrip,
        PrimitiveTopology::TriangleList => PrimitiveType::Triangle,
        PrimitiveTopology::TriangleStrip => PrimitiveType::TriangleStrip,
    }
}

fn map_index_format(format: IndexFormat) -> IndexType {
    match format {
        IndexFormat::Uint16 => IndexType::Uint16,
        IndexFormat::Uint32 => IndexType::Uint32,
    }
}

fn index_stride(index_type: IndexType) -> u64 {
    match index_type {
        IndexType::Uint16 => 2,
        IndexType::Uint32 => 4,
    }
}

/// Resolves one wire buffer range against `allocations` and the registry.
///
/// Returns the raw `MTLBuffer`, the byte offset and the declared range length.
/// `None` means an unknown handle, zero field or an out-of-bounds range, which
/// the caller reports as `UnknownHandle` exactly like [`Self::submit_batch`].
fn resolve_buffer_range(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    allocation_id: u64,
    generation: u32,
    offset: u64,
    length: u64,
) -> Option<(objc::Id, usize, u64)> {
    if allocation_id == 0
        || generation == 0
        || length == 0
        || length > wire::MAX_RESOURCE_LEN
        || !offset.is_multiple_of(MTL_BUFFER_OFFSET_ALIGNMENT)
    {
        return None;
    }
    let registered_id = *allocations.get(&(allocation_id, generation))?;
    let registered = state.buffers.get(&registered_id.0)?;
    let end = offset.checked_add(length)?;
    if end > registered.len as u64 {
        return None;
    }
    Some((
        registered.buffer.as_raw(),
        usize::try_from(offset).ok()?,
        length,
    ))
}

/// Resolves one texture handle and reads its geometry.
fn resolve_texture(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    allocation_id: u64,
    generation: u32,
) -> Option<(objc::Id, crate::macos::render::TextureInfo)> {
    if allocation_id == 0 || generation == 0 {
        return None;
    }
    let registered_id = *allocations.get(&(allocation_id, generation))?;
    let texture = state.textures.get(&registered_id.0)?;
    let info = texture_info(texture.as_raw()).ok()?;
    Some((texture.as_raw(), info))
}

/// Resolves a wire binding list to consecutive slots starting at `start_slot`.
fn resolve_binding_slots(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    bindings: &[wire::BufferBinding],
    start_slot: usize,
) -> Result<Vec<BoundRange>, PrepareAbort> {
    let mut resolved = Vec::with_capacity(bindings.len());
    for (index, binding) in bindings.iter().enumerate() {
        let (buffer, offset, _length) = resolve_buffer_range(
            state,
            allocations,
            binding.allocation_id,
            binding.generation,
            binding.offset,
            binding.length,
        )
        .ok_or(PrepareAbort::UnknownHandle)?;
        let slot = start_slot
            .checked_add(index)
            .ok_or(MetalError::InvalidArg("buffer slot overflows usize"))?;
        resolved.push((buffer, offset, slot));
    }
    Ok(resolved)
}

/// Resolves one wire indirect range and its optional count buffer.
///
/// The argument range length is `max_count * stride`, so a range that runs past
/// the registered buffer is an unknown handle exactly like a bad binding. A
/// present count buffer is resolved too (so an unresolved allocation is
/// `UnknownHandle` before any encoder exists); [`indirect::validate`] then
/// refuses it because Metal's legacy indirect selectors cannot consume it.
fn resolve_indirect_arguments(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    arguments: &crate::wire_v3::IndirectArguments,
    stride: u64,
) -> Result<indirect::IndirectArguments, PrepareAbort> {
    let length = arguments
        .byte_len(stride)
        .ok_or(MetalError::InvalidArg("indirect argument range overflows u64"))?;
    let (argument_buffer, argument_offset, _) = resolve_buffer_range(
        state,
        allocations,
        arguments.allocation_id,
        arguments.generation,
        arguments.offset,
        length,
    )
    .ok_or(PrepareAbort::UnknownHandle)?;
    let count = match &arguments.count {
        Some(count) => {
            let (buffer, offset, _) = resolve_buffer_range(
                state,
                allocations,
                count.allocation_id,
                count.generation,
                count.offset,
                4,
            )
            .ok_or(PrepareAbort::UnknownHandle)?;
            Some(indirect::IndirectCount { buffer, offset })
        }
        None => None,
    };
    Ok(indirect::IndirectArguments {
        argument_buffer,
        argument_offset,
        max_count: arguments.max_count,
        count,
    })
}

/// Resolves the bound index buffer for an indirect indexed draw.
///
/// The index count is GPU-side, so only the binding itself can be checked; the
/// GPU clamps every read to the declared index-buffer range at execution.
fn prepare_indirect_indexed_draw(
    index_binding: &Option<(objc::Id, usize, IndexType, u64)>,
) -> Result<(objc::Id, usize, IndexType), PrepareAbort> {
    let Some((buffer, base_offset, index_type, _length)) = *index_binding else {
        return Err(
            MetalError::InvalidArg("indirect indexed draw requires a preceding SetIndexBuffer")
                .into(),
        );
    };
    Ok((buffer, base_offset, index_type))
}

fn validate_viewport(
    viewport: &crate::wire_v3::Viewport,
    width: u32,
    height: u32,
) -> Result<RenderViewport, MetalError> {
    if !(viewport.x.is_finite()
        && viewport.y.is_finite()
        && viewport.width.is_finite()
        && viewport.height.is_finite()
        && viewport.near.is_finite()
        && viewport.far.is_finite())
    {
        return Err(MetalError::InvalidArg("viewport values must be finite"));
    }
    if viewport.x < 0.0 || viewport.y < 0.0 {
        return Err(MetalError::InvalidArg(
            "viewport origin must not be negative",
        ));
    }
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(MetalError::InvalidArg(
            "viewport width and height must be positive",
        ));
    }
    if viewport.x + viewport.width > f64::from(width)
        || viewport.y + viewport.height > f64::from(height)
    {
        return Err(MetalError::InvalidArg(
            "viewport exceeds the render pass attachment extent",
        ));
    }
    Ok(RenderViewport {
        origin_x: viewport.x,
        origin_y: viewport.y,
        width: viewport.width,
        height: viewport.height,
        znear: viewport.near,
        zfar: viewport.far,
    })
}

fn validate_scissor(
    scissor: &crate::wire_v3::ScissorRect,
    width: u32,
    height: u32,
) -> Result<RenderScissorRect, MetalError> {
    if scissor.width == 0 || scissor.height == 0 {
        return Err(MetalError::InvalidArg(
            "scissor width and height must be nonzero",
        ));
    }
    let end_x = scissor
        .x
        .checked_add(scissor.width)
        .ok_or(MetalError::InvalidArg("scissor X extent overflows u32"))?;
    let end_y = scissor
        .y
        .checked_add(scissor.height)
        .ok_or(MetalError::InvalidArg("scissor Y extent overflows u32"))?;
    if end_x > width || end_y > height {
        return Err(MetalError::InvalidArg(
            "scissor exceeds the render pass attachment extent",
        ));
    }
    Ok(RenderScissorRect {
        x: scissor.x,
        y: scissor.y,
        width: scissor.width,
        height: scissor.height,
    })
}

/// Validates and prepares a compute segment.
fn prepare_compute_segment(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    range: Range<usize>,
) -> Result<Vec<PreparedComputeOp>, PrepareAbort> {
    let mut ops = Vec::new();
    for index in range {
        match &commands[index] {
            CommandV3::SetComputePipeline(pipeline) => {
                let pso = state
                    .pipelines
                    .get(&pipeline.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedComputeOp::Pipeline(pso.as_raw()));
            }
            CommandV3::SetPipeline(pipeline) => {
                let pso = state
                    .pipelines
                    .get(&pipeline.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedComputeOp::Pipeline(pso.as_raw()));
            }
            CommandV3::Dispatch1d(dispatch) => {
                if dispatch.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                    return Err(
                        MetalError::InvalidArg("dispatch exceeds wire binding limit").into(),
                    );
                }
                let pso = state
                    .pipelines
                    .get(&dispatch.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                let threads_per_group = dispatch.threads_per_group as usize;
                if threads_per_group as u64 > pipeline_thread_limit(pso) {
                    return Err(MetalError::InvalidArg(
                        "dispatch threads_per_group exceeds pipeline threadgroup maximum",
                    )
                    .into());
                }
                ops.push(PreparedComputeOp::Dispatch(PreparedV3Dispatch {
                    pipeline: pso.as_raw(),
                    grid: PreparedGrid::OneD {
                        threads: dispatch.threads as usize,
                        threads_per_group,
                    },
                    bindings: resolve_binding_slots(state, allocations, &dispatch.bindings, 0)?,
                }));
            }
            CommandV3::Dispatch3d(dispatch) => {
                if dispatch.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                    return Err(
                        MetalError::InvalidArg("dispatch exceeds wire binding limit").into(),
                    );
                }
                let pso = state
                    .pipelines
                    .get(&dispatch.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                let local_threads =
                    dispatch
                        .checked_threads_per_group()
                        .ok_or(MetalError::InvalidArg(
                            "3D dispatch local thread product overflows u64",
                        ))?;
                if local_threads > pipeline_thread_limit(pso) {
                    return Err(MetalError::InvalidArg(
                        "3D dispatch threadgroup exceeds pipeline threadgroup maximum",
                    )
                    .into());
                }
                ops.push(PreparedComputeOp::Dispatch(PreparedV3Dispatch {
                    pipeline: pso.as_raw(),
                    grid: PreparedGrid::ThreeD {
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
                    },
                    bindings: resolve_binding_slots(state, allocations, &dispatch.bindings, 0)?,
                }));
            }
            CommandV3::DispatchMesh(_) => {
                // Mesh draws are render-encoder work; the planner only puts
                // them inside a render pass. Reaching a compute segment means
                // the plan and executor disagree, which is a structural error.
                return Err(MetalError::InvalidArg(
                    "mesh dispatch is only valid inside a render pass",
                )
                .into());
            }
            CommandV3::ExecuteIndirectDispatch(dispatch) => {
                let pso = state
                    .pipelines
                    .get(&dispatch.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if dispatch.threads_per_group.contains(&0) {
                    return Err(
                        MetalError::InvalidArg("indirect dispatch threadgroup must be nonzero")
                            .into(),
                    );
                }
                let local_threads = dispatch.threads_per_group.iter().try_fold(
                    1u64,
                    |product, dimension| product.checked_mul(u64::from(*dimension)),
                );
                let Some(local_threads) = local_threads else {
                    return Err(MetalError::InvalidArg(
                        "indirect dispatch threadgroup product overflows u64",
                    )
                    .into());
                };
                if local_threads > pipeline_thread_limit(pso) {
                    return Err(MetalError::InvalidArg(
                        "indirect dispatch exceeds the pipeline threadgroup maximum",
                    )
                    .into());
                }
                let arguments = resolve_indirect_arguments(
                    state,
                    allocations,
                    &dispatch.arguments,
                    indirect::DISPATCH_ARGUMENT_STRIDE,
                )?;
                indirect::validate(&arguments)?;
                ops.push(PreparedComputeOp::DispatchIndirect {
                    pipeline: pso.as_raw(),
                    threads_per_group: MTLSize {
                        width: dispatch.threads_per_group[0] as usize,
                        height: dispatch.threads_per_group[1] as usize,
                        depth: dispatch.threads_per_group[2] as usize,
                    },
                    arguments,
                });
            }
            CommandV3::SetRootConstants(constants) => {
                if constants.bytes.is_empty()
                    || constants.bytes.len() > crate::wire_v3::MAX_ROOT_CONSTANT_BYTES_V3
                {
                    return Err(
                        MetalError::InvalidArg("root constants must be 1..=256 bytes").into(),
                    );
                }
                ops.push(PreparedComputeOp::Bytes {
                    slot: usize::from(constants.slot),
                    bytes: constants.bytes.clone(),
                });
            }
            CommandV3::SetDescriptorTable(table) => {
                if table.bindings.is_empty() || table.bindings.len() > wire::MAX_DESCRIPTOR_BINDINGS
                {
                    return Err(
                        MetalError::InvalidArg("descriptor table needs 1..=16 bindings").into(),
                    );
                }
                if table.kind == DescriptorKind::Sampler {
                    return Err(MetalError::NotSupported(
                        "sampler descriptor tables are not bound by the native compute path",
                    )
                    .into());
                }
                ops.push(PreparedComputeOp::Bind(resolve_binding_slots(
                    state,
                    allocations,
                    &table.bindings,
                    usize::from(table.base_register),
                )?));
            }
            CommandV3::SetRootBufferView(view) => {
                let (buffer, offset, _length) = resolve_buffer_range(
                    state,
                    allocations,
                    view.allocation_id,
                    view.generation,
                    view.offset,
                    view.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedComputeOp::Bind(vec![(
                    buffer,
                    offset,
                    usize::from(view.slot),
                )]));
            }
            CommandV3::ResourceBarrier(barrier) => {
                if resolve_barrier_entries(state, allocations, &barrier.entries).is_none() {
                    return Err(PrepareAbort::UnknownHandle);
                }
                ops.push(PreparedComputeOp::Barrier);
            }
            _ => {
                return Err(
                    MetalError::InvalidArg("command is not valid in a compute segment").into(),
                );
            }
        }
    }
    Ok(ops)
}

/// Validates and prepares one render segment, including its pass descriptor.
fn prepare_render_segment(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    range: Range<usize>,
    pass: &BeginRenderPass,
) -> Result<(objc::Id, u32, u32, Vec<PreparedRenderOp>), PrepareAbort> {
    let mut colors = Vec::with_capacity(pass.colors.len());
    let mut color_formats = Vec::with_capacity(pass.colors.len());
    let mut extent: Option<(u32, u32)> = None;
    for color in &pass.colors {
        let (texture, info) =
            resolve_texture(state, allocations, color.allocation_id, color.generation)
                .ok_or(PrepareAbort::UnknownHandle)?;
        if !info.format.is_color() {
            return Err(MetalError::InvalidArg(
                "colour attachment must be BGRA8Unorm or RGBA8Unorm",
            )
            .into());
        }
        if u32::from(color.mip_level) >= info.mip_levels || u32::from(color.slice) >= info.slices {
            return Err(MetalError::InvalidArg(
                "colour attachment mip level or slice is out of range",
            )
            .into());
        }
        match extent {
            Some(existing) if existing != (info.width, info.height) => {
                return Err(MetalError::InvalidArg(
                    "all render pass attachments must share one extent",
                )
                .into());
            }
            Some(_) => {}
            None => extent = Some((info.width, info.height)),
        }
        color_formats.push(info.format);
        colors.push(RawColorAttachment {
            texture,
            load: map_load_op(color.load),
            store: map_store_op(color.store),
            clear_color: color.clear_color,
            level: usize::from(color.mip_level),
            slice: usize::from(color.slice),
        });
    }
    let (width, height) = extent.ok_or(MetalError::InvalidArg(
        "render pass needs at least one colour attachment",
    ))?;

    let mut depth_format = None;
    let depth = match &pass.depth_stencil {
        Some(depth) => {
            let (texture, info) =
                resolve_texture(state, allocations, depth.allocation_id, depth.generation)
                    .ok_or(PrepareAbort::UnknownHandle)?;
            if !info.format.is_depth() {
                return Err(MetalError::InvalidArg(
                    "depth attachment must be Depth32Float or Depth32Float_Stencil8",
                )
                .into());
            }
            if (info.width, info.height) != (width, height) {
                return Err(MetalError::InvalidArg(
                    "depth attachment size must match the colour attachments",
                )
                .into());
            }
            if u32::from(depth.mip_level) >= info.mip_levels
                || u32::from(depth.slice) >= info.slices
            {
                return Err(MetalError::InvalidArg(
                    "depth attachment mip level or slice is out of range",
                )
                .into());
            }
            depth_format = Some(info.format);
            Some(RawDepthAttachment {
                texture,
                has_stencil: info.format.has_stencil(),
                depth_load: map_load_op(depth.depth_load),
                depth_store: map_store_op(depth.depth_store),
                clear_depth: f64::from(depth.clear_depth),
                stencil_load: map_load_op(depth.stencil_load),
                stencil_store: map_store_op(depth.stencil_store),
                clear_stencil: depth.clear_stencil,
                level: usize::from(depth.mip_level),
                slice: usize::from(depth.slice),
            })
        }
        None => None,
    };

    let descriptor = build_mrt_render_pass_descriptor(&colors, depth.as_ref())?;

    let mut ops = Vec::new();
    let mut mesh_pipeline: Option<u64> = None;
    let mut current_pipeline = false;
    let mut index_binding: Option<(objc::Id, usize, IndexType, u64)> = None;
    let mut topology = PrimitiveType::Triangle;
    for index in range {
        match &commands[index] {
            CommandV3::BeginRenderPass(_) | CommandV3::EndRenderPass => continue,
            CommandV3::SetGraphicsPipeline(pipeline) => {
                let graphics = state
                    .render_pipelines
                    .get(&pipeline.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if graphics.color_formats() != color_formats.as_slice() {
                    return Err(MetalError::InvalidArg(
                        "graphics pipeline colour formats do not match the render pass",
                    )
                    .into());
                }
                if graphics.depth_format() != depth_format {
                    return Err(MetalError::InvalidArg(
                        "graphics pipeline depth format does not match the render pass",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::Pipeline {
                    state: graphics.state_raw(),
                    depth_state: graphics.depth_state_raw(),
                });
                current_pipeline = true;
                mesh_pipeline = None;
            }
            CommandV3::SetPipeline(pipeline) => {
                let graphics = state
                    .render_pipelines
                    .get(&pipeline.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if graphics.color_formats() != color_formats.as_slice() {
                    return Err(MetalError::InvalidArg(
                        "graphics pipeline colour formats do not match the render pass",
                    )
                    .into());
                }
                if graphics.depth_format() != depth_format {
                    return Err(MetalError::InvalidArg(
                        "graphics pipeline depth format does not match the render pass",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::Pipeline {
                    state: graphics.state_raw(),
                    depth_state: graphics.depth_state_raw(),
                });
                current_pipeline = true;
                mesh_pipeline = None;
            }
            CommandV3::SetMeshPipeline(pipeline) => {
                let mesh = state
                    .mesh_pipelines
                    .get(&pipeline.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if mesh.color_formats() != color_formats.as_slice() {
                    return Err(MetalError::InvalidArg(
                        "mesh pipeline colour formats do not match the render pass",
                    )
                    .into());
                }
                if mesh.depth_format() != depth_format {
                    return Err(MetalError::InvalidArg(
                        "mesh pipeline depth format does not match the render pass",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::Pipeline {
                    state: mesh.state_raw(),
                    depth_state: mesh.depth_state_raw(),
                });
                current_pipeline = true;
                mesh_pipeline = Some(pipeline.pipeline_id);
            }
            CommandV3::SetViewport(viewport) => {
                ops.push(PreparedRenderOp::Viewport(validate_viewport(
                    viewport, width, height,
                )?));
            }
            CommandV3::SetScissor(scissor) => {
                ops.push(PreparedRenderOp::Scissor(validate_scissor(
                    scissor, width, height,
                )?));
            }
            CommandV3::SetPrimitiveTopology(wire_topology) => {
                topology = map_topology(*wire_topology);
            }
            CommandV3::SetVertexBuffers(set) => {
                if set.buffers.is_empty() || set.buffers.len() > crate::wire_v3::MAX_VERTEX_BUFFERS
                {
                    return Err(
                        MetalError::InvalidArg("vertex buffer set needs 1..=16 buffers").into(),
                    );
                }
                for buffer in &set.buffers {
                    if buffer.stride == 0 {
                        return Err(
                            MetalError::InvalidArg("vertex buffer stride must be nonzero").into(),
                        );
                    }
                }
                let bindings = set
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(offset, buffer)| {
                        let (raw, byte_offset, _length) = resolve_buffer_range(
                            state,
                            allocations,
                            buffer.allocation_id,
                            buffer.generation,
                            buffer.offset,
                            buffer.length,
                        )
                        .ok_or(PrepareAbort::UnknownHandle)?;
                        let slot = usize::from(set.start_slot)
                            .checked_add(offset)
                            .ok_or(MetalError::InvalidArg("vertex buffer slot overflows usize"))?;
                        Ok((raw, byte_offset, slot))
                    })
                    .collect::<Result<Vec<_>, PrepareAbort>>()?;
                ops.push(PreparedRenderOp::VertexBuffers(bindings));
            }
            CommandV3::SetIndexBuffer(binding) => {
                let (buffer, offset, length) = resolve_buffer_range(
                    state,
                    allocations,
                    binding.allocation_id,
                    binding.generation,
                    binding.offset,
                    binding.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let index_type = map_index_format(binding.format);
                if !offset.is_multiple_of(index_stride(index_type) as usize) {
                    return Err(MetalError::InvalidArg(
                        "index buffer offset is not aligned to the index type",
                    )
                    .into());
                }
                // The binding is captured by each indexed draw; no encoder
                // call happens until a draw uses it.
                index_binding = Some((buffer, offset, index_type, length));
            }
            CommandV3::SetRootConstants(constants) => {
                if constants.bytes.is_empty()
                    || constants.bytes.len() > crate::wire_v3::MAX_ROOT_CONSTANT_BYTES_V3
                {
                    return Err(
                        MetalError::InvalidArg("root constants must be 1..=256 bytes").into(),
                    );
                }
                ops.push(PreparedRenderOp::VertexBytes {
                    slot: usize::from(constants.slot),
                    bytes: constants.bytes.clone(),
                });
            }
            CommandV3::SetDescriptorTable(table) => {
                if table.bindings.is_empty() || table.bindings.len() > wire::MAX_DESCRIPTOR_BINDINGS
                {
                    return Err(
                        MetalError::InvalidArg("descriptor table needs 1..=16 bindings").into(),
                    );
                }
                if table.kind == DescriptorKind::Sampler {
                    return Err(MetalError::NotSupported(
                        "sampler descriptor tables are not bound by the native graphics path",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::VertexBuffers(resolve_binding_slots(
                    state,
                    allocations,
                    &table.bindings,
                    usize::from(table.base_register),
                )?));
            }
            CommandV3::SetRootBufferView(view) => {
                let (buffer, offset, _length) = resolve_buffer_range(
                    state,
                    allocations,
                    view.allocation_id,
                    view.generation,
                    view.offset,
                    view.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedRenderOp::VertexBuffers(vec![(
                    buffer,
                    offset,
                    usize::from(view.slot),
                )]));
            }
            CommandV3::ResourceBarrier(barrier) => {
                if resolve_barrier_entries(state, allocations, &barrier.entries).is_none() {
                    return Err(PrepareAbort::UnknownHandle);
                }
                ops.push(PreparedRenderOp::Barrier);
            }
            CommandV3::DispatchMesh(dispatch) => {
                let mesh = state
                    .mesh_pipelines
                    .get(&dispatch.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if mesh_pipeline != Some(dispatch.pipeline_id) {
                    return Err(MetalError::InvalidArg(
                        "mesh dispatch requires a selected mesh pipeline in its render pass",
                    )
                    .into());
                }
                if dispatch.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                    return Err(MetalError::InvalidArg(
                        "mesh dispatch exceeds the wire binding limit",
                    )
                    .into());
                }
                let group_counts = dispatch.checked_group_count().ok_or(
                    MetalError::InvalidArg("mesh dispatch group counts overflow u64"),
                )?;
                let limit = mesh.max_total_threadgroups_per_mesh_grid();
                if limit != 0 && group_counts > limit as u64 {
                    return Err(MetalError::InvalidArg(
                        "mesh dispatch exceeds maxTotalThreadgroupsPerMeshGrid",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::DispatchMesh {
                    group_counts: MTLSize {
                        width: dispatch.group_counts[0] as usize,
                        height: dispatch.group_counts[1] as usize,
                        depth: dispatch.group_counts[2] as usize,
                    },
                    threads_per_object_threadgroup: mesh.threads_per_object_threadgroup(),
                    threads_per_mesh_threadgroup: mesh.threads_per_mesh_threadgroup(),
                    bindings: resolve_binding_slots(state, allocations, &dispatch.bindings, 0)?,
                });
            }
            CommandV3::Draw(draw) => {
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                if draw.vertex_count == 0 || draw.instance_count == 0 {
                    return Err(MetalError::InvalidArg(
                        "draw vertex and instance counts must be nonzero",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::Draw {
                    primitive: topology,
                    vertex_start: draw.start_vertex as usize,
                    vertex_count: draw.vertex_count as usize,
                    instance_count: draw.instance_count as usize,
                    base_instance: draw.start_instance as usize,
                });
            }
            CommandV3::DrawInstanced(draw) => {
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                if draw.vertex_count_per_instance == 0 || draw.instance_count == 0 {
                    return Err(MetalError::InvalidArg(
                        "draw vertex and instance counts must be nonzero",
                    )
                    .into());
                }
                ops.push(PreparedRenderOp::Draw {
                    primitive: topology,
                    vertex_start: draw.start_vertex as usize,
                    vertex_count: draw.vertex_count_per_instance as usize,
                    instance_count: draw.instance_count as usize,
                    base_instance: draw.start_instance as usize,
                });
            }
            CommandV3::DrawIndexed(draw) => {
                let (buffer, offset, index_type, index_count) =
                    prepare_indexed_draw(&index_binding, draw.index_count, draw.start_index)?;
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                if draw.instance_count == 0 {
                    return Err(
                        MetalError::InvalidArg("draw instance count must be nonzero").into(),
                    );
                }
                ops.push(PreparedRenderOp::DrawIndexed {
                    primitive: topology,
                    index_count,
                    index_type,
                    index_buffer: buffer,
                    index_offset: offset,
                    base_vertex: draw.base_vertex as isize,
                    instance_count: draw.instance_count as usize,
                    base_instance: draw.start_instance as usize,
                });
            }
            CommandV3::DrawIndexedInstanced(draw) => {
                let (buffer, offset, index_type, index_count) = prepare_indexed_draw(
                    &index_binding,
                    draw.index_count_per_instance,
                    draw.start_index,
                )?;
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                if draw.instance_count == 0 {
                    return Err(
                        MetalError::InvalidArg("draw instance count must be nonzero").into(),
                    );
                }
                ops.push(PreparedRenderOp::DrawIndexed {
                    primitive: topology,
                    index_count,
                    index_type,
                    index_buffer: buffer,
                    index_offset: offset,
                    base_vertex: draw.base_vertex as isize,
                    instance_count: draw.instance_count as usize,
                    base_instance: draw.start_instance as usize,
                });
            }
            CommandV3::ExecuteIndirectDraw(draw) => {
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                let arguments = resolve_indirect_arguments(
                    state,
                    allocations,
                    &draw.arguments,
                    indirect::DRAW_ARGUMENT_STRIDE,
                )?;
                indirect::validate(&arguments)?;
                ops.push(PreparedRenderOp::DrawIndirect {
                    primitive: topology,
                    arguments,
                });
            }
            CommandV3::ExecuteIndirectDrawIndexed(draw) => {
                if !current_pipeline || mesh_pipeline.is_some() {
                    return Err(MetalError::InvalidArg(
                        "draw requires a selected graphics pipeline in its render pass",
                    )
                    .into());
                }
                let (index_buffer, index_offset, index_type) =
                    prepare_indirect_indexed_draw(&index_binding)?;
                let arguments = resolve_indirect_arguments(
                    state,
                    allocations,
                    &draw.arguments,
                    indirect::DRAW_INDEXED_ARGUMENT_STRIDE,
                )?;
                indirect::validate(&arguments)?;
                ops.push(PreparedRenderOp::DrawIndexedIndirect {
                    primitive: topology,
                    index_type,
                    index_buffer,
                    index_offset,
                    arguments,
                });
            }
            CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
                let mesh = state
                    .mesh_pipelines
                    .get(&dispatch.pipeline_id)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                if mesh_pipeline != Some(dispatch.pipeline_id) {
                    return Err(MetalError::InvalidArg(
                        "mesh dispatch requires a selected mesh pipeline in its render pass",
                    )
                    .into());
                }
                let arguments = resolve_indirect_arguments(
                    state,
                    allocations,
                    &dispatch.arguments,
                    indirect::DISPATCH_ARGUMENT_STRIDE,
                )?;
                indirect::validate(&arguments)?;
                ops.push(PreparedRenderOp::DispatchMeshIndirect {
                    threads_per_object_threadgroup: mesh.threads_per_object_threadgroup(),
                    threads_per_mesh_threadgroup: mesh.threads_per_mesh_threadgroup(),
                    arguments,
                });
            }
            _ => {
                return Err(
                    MetalError::InvalidArg("command is not valid in a render segment").into(),
                );
            }
        }
    }
    Ok((descriptor, width, height, ops))
}

/// Resolves the index window of an indexed draw against the bound index buffer.
fn prepare_indexed_draw(
    index_binding: &Option<(objc::Id, usize, IndexType, u64)>,
    index_count: u32,
    start_index: u32,
) -> Result<(objc::Id, usize, IndexType, usize), PrepareAbort> {
    if index_count == 0 {
        return Err(MetalError::InvalidArg("draw index count must be nonzero").into());
    }
    let Some((buffer, base_offset, index_type, length)) = *index_binding else {
        return Err(
            MetalError::InvalidArg("indexed draw requires a preceding SetIndexBuffer").into(),
        );
    };
    let stride = index_stride(index_type);
    let start_byte = u64::from(start_index)
        .checked_mul(stride)
        .ok_or(MetalError::InvalidArg("start_index overflows u64"))?;
    let window = start_byte
        .checked_add(
            u64::from(index_count)
                .checked_mul(stride)
                .ok_or(MetalError::InvalidArg("index count overflows u64"))?,
        )
        .ok_or(MetalError::InvalidArg("indexed draw range overflows u64"))?;
    if window > length {
        return Err(
            MetalError::InvalidArg("indexed draw range exceeds the index binding length").into(),
        );
    }
    let offset = base_offset
        .checked_add(to_usize(start_byte)?)
        .ok_or(MetalError::InvalidArg(
            "indexed draw index offset overflows usize",
        ))?;
    Ok((buffer, offset, index_type, index_count as usize))
}

/// Validates and prepares a blit segment.
fn prepare_blit_segment(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    range: Range<usize>,
) -> Result<Vec<PreparedBlitOp>, PrepareAbort> {
    let mut ops = Vec::new();
    for index in range {
        match &commands[index] {
            CommandV3::CopyBuffer(copy) => {
                if copy.length == 0 || !copy.length.is_multiple_of(4) {
                    return Err(MetalError::InvalidArg(
                        "buffer copy length must be nonzero and 4-byte aligned",
                    )
                    .into());
                }
                let (src, src_offset, _) = resolve_buffer_range(
                    state,
                    allocations,
                    copy.src_allocation_id,
                    copy.src_generation,
                    copy.src_offset,
                    copy.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let (dst, dst_offset, _) = resolve_buffer_range(
                    state,
                    allocations,
                    copy.dst_allocation_id,
                    copy.dst_generation,
                    copy.dst_offset,
                    copy.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedBlitOp::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    length: to_usize(copy.length)?,
                });
            }
            CommandV3::CopyTexture(copy) => {
                if copy.width == 0 || copy.height == 0 || copy.depth == 0 {
                    return Err(MetalError::InvalidArg(
                        "texture copy width, height and depth must be nonzero",
                    )
                    .into());
                }
                let (src, src_info) = resolve_texture(
                    state,
                    allocations,
                    copy.src_allocation_id,
                    copy.src_generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let (dst, dst_info) = resolve_texture(
                    state,
                    allocations,
                    copy.dst_allocation_id,
                    copy.dst_generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                if src_info.format != dst_info.format {
                    return Err(MetalError::InvalidArg(
                        "texture copy source and destination formats must match",
                    )
                    .into());
                }
                if u32::from(copy.src_mip) >= src_info.mip_levels
                    || u32::from(copy.src_slice) >= src_info.slices
                    || u32::from(copy.dst_mip) >= dst_info.mip_levels
                    || u32::from(copy.dst_slice) >= dst_info.slices
                {
                    return Err(MetalError::InvalidArg(
                        "texture copy mip level or slice is out of range",
                    )
                    .into());
                }
                if copy.width > src_info.width
                    || copy.height > src_info.height
                    || copy.width > dst_info.width
                    || copy.height > dst_info.height
                {
                    return Err(MetalError::InvalidArg(
                        "texture copy extent exceeds a texture dimension",
                    )
                    .into());
                }
                ops.push(PreparedBlitOp::CopyTexture {
                    src,
                    src_slice: usize::from(copy.src_slice),
                    src_level: usize::from(copy.src_mip),
                    dst,
                    dst_slice: usize::from(copy.dst_slice),
                    dst_level: usize::from(copy.dst_mip),
                    width: copy.width as usize,
                    height: copy.height as usize,
                    depth: copy.depth as usize,
                });
            }
            CommandV3::CopyTextureRegion(copy) => {
                let (src, src_info) = resolve_texture(
                    state,
                    allocations,
                    copy.src.allocation_id,
                    copy.src.generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let (dst, dst_info) = resolve_texture(
                    state,
                    allocations,
                    copy.dst.allocation_id,
                    copy.dst.generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                if src_info.format != dst_info.format {
                    return Err(MetalError::InvalidArg(
                        "texture copy source and destination formats must match",
                    )
                    .into());
                }
                let region = blit::TextureRegion {
                    source: blit::TextureSubresourceRef {
                        texture: src,
                        slice: usize::from(copy.src.slice),
                        level: usize::from(copy.src.mip_level),
                    },
                    source_origin: [
                        copy.src_origin[0] as usize,
                        copy.src_origin[1] as usize,
                        copy.src_origin[2] as usize,
                    ],
                    destination: blit::TextureSubresourceRef {
                        texture: dst,
                        slice: usize::from(copy.dst.slice),
                        level: usize::from(copy.dst.mip_level),
                    },
                    destination_origin: [
                        copy.dst_origin[0] as usize,
                        copy.dst_origin[1] as usize,
                        copy.dst_origin[2] as usize,
                    ],
                    size: [
                        copy.size[0] as usize,
                        copy.size[1] as usize,
                        copy.size[2] as usize,
                    ],
                };
                region.validate()?;
                ops.push(PreparedBlitOp::CopyTextureRegion(region));
            }
            CommandV3::ResolveSubresource(resolve) => {
                let (src, _) = resolve_texture(
                    state,
                    allocations,
                    resolve.src.allocation_id,
                    resolve.src.generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let (dst, _) = resolve_texture(
                    state,
                    allocations,
                    resolve.dst.allocation_id,
                    resolve.dst.generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let source = blit::TextureSubresourceRef {
                    texture: src,
                    slice: usize::from(resolve.src.slice),
                    level: usize::from(resolve.src.mip_level),
                };
                let destination = blit::TextureSubresourceRef {
                    texture: dst,
                    slice: usize::from(resolve.dst.slice),
                    level: usize::from(resolve.dst.mip_level),
                };
                blit::validate_resolve(source, destination)?;
                ops.push(PreparedBlitOp::ResolveSubresource {
                    src: source,
                    dst: destination,
                });
            }
            CommandV3::ResourceBarrier(barrier) => {
                if resolve_barrier_entries(state, allocations, &barrier.entries).is_none() {
                    return Err(PrepareAbort::UnknownHandle);
                }
                ops.push(PreparedBlitOp::Barrier);
            }
            _ => {
                return Err(
                    MetalError::InvalidArg("command is not valid in a blit segment").into(),
                );
            }
        }
    }
    Ok(ops)
}

/// Validates every handle and bound and builds the executable segment plan.
fn prepare_v3(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    plan: &[crate::macos::render::Segment],
) -> Result<Vec<PreparedSegment>, PrepareAbort> {
    let passes: Vec<&BeginRenderPass> = commands
        .iter()
        .filter_map(|command| match command {
            CommandV3::BeginRenderPass(pass) => Some(pass),
            _ => None,
        })
        .collect();
    let mut prepared = Vec::with_capacity(plan.len());
    for segment in plan {
        match &segment.kind {
            SegmentKind::Compute(range) => prepared.push(PreparedSegment::Compute(
                prepare_compute_segment(state, allocations, commands, range.clone())?,
            )),
            SegmentKind::Render { pass_index, range } => {
                let pass = passes.get(*pass_index).ok_or(MetalError::InternalError(
                    "render segment names a missing BeginRenderPass".to_string(),
                ))?;
                let (descriptor, width, height, ops) =
                    prepare_render_segment(state, allocations, commands, range.clone(), pass)?;
                prepared.push(PreparedSegment::Render {
                    descriptor,
                    width,
                    height,
                    ops,
                });
            }
            SegmentKind::Blit(range) => prepared.push(PreparedSegment::Blit(prepare_blit_segment(
                state,
                allocations,
                commands,
                range.clone(),
            )?)),
        }
    }
    Ok(prepared)
}

impl DirectMetalTransport {
    /// Registers a render-target texture as an attachment allocation.
    ///
    /// The returned handle can be named by a v3 `BeginRenderPass` colour or
    /// depth attachment through an `(allocation_id, generation)` entry in the
    /// submission's allocation snapshot.
    pub fn register_render_target(&self, target: &RenderTarget) -> Result<VmTextureId, MetalError> {
        let texture = target.clone_objc()?;
        let mut state = self.lock_state()?;
        let id = state.alloc_id();
        state.textures.insert(id, texture);
        Ok(VmTextureId(id))
    }

    /// Registers the texture of an IOSurface-backed image as a v3 present sink.
    ///
    /// `Present` only checks that the handle names a registered surface; the
    /// caller has already rendered into the same texture through a render pass,
    /// so no copy happens on the present path.
    pub fn register_present_surface(
        &self,
        surface_id: u64,
        target: &RenderTarget,
    ) -> Result<(), MetalError> {
        if surface_id == 0 {
            return Err(MetalError::InvalidArg(
                "present surface handle must be nonzero",
            ));
        }
        let texture = target.clone_objc()?;
        let mut state = self.lock_state()?;
        state.surfaces.insert(surface_id, texture);
        Ok(())
    }

    /// Executes one version 3 submission on a single `MTLCommandBuffer`.
    ///
    /// See [`Self::submit_v3_with_stats`] for the full contract. This wrapper
    /// discards the diagnostic encoder inventory.
    pub fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        self.submit_v3_with_stats(submission, allocations)
            .map(|(reply, _stats)| reply)
    }

    /// Executes one version 3 submission and reports its encoder inventory.
    ///
    /// # Contract
    ///
    /// * The command stream is first planned by
    ///   [`crate::macos::render::plan_encoder_segments`], so illegal orderings
    ///   are explicit errors before anything is resolved.
    /// * Every pipeline, buffer, texture and surface handle, every bound range
    ///   and every attachment extent/level/slice is validated against the live
    ///   registry *before* the command buffer is created. An unknown handle
    ///   returns [`ReplyStatus::UnknownHandle`] with no GPU work.
    /// * Consecutive dispatches share one `MTLComputeCommandEncoder`,
    ///   consecutive copies share one `MTLBlitCommandEncoder`, and each render
    ///   pass gets one `MTLRenderCommandEncoder`; all of them live on one
    ///   `MTLCommandBuffer`.
    /// * `memoryBarrier` is emitted only for an explicit `ResourceBarrier`
    ///   command, never per dispatch. A barrier between blit copies is realized
    ///   by the blit encoder boundary, because `MTLBlitCommandEncoder` has no
    ///   barrier API.
    /// * A list identity executes at most once: a replay returns
    ///   [`ReplyStatus::AlreadyExecuted`] with no GPU work, mirroring
    ///   [`crate::cmdlist::ListLedger`]. A validation refusal does not consume
    ///   the identity, so the caller may retry a corrected submission.
    /// * `DispatchMesh` is render-encoder work: it must appear inside a
    ///   `BeginRenderPass`/`EndRenderPass` pair and after a `SetMeshPipeline`
    ///   naming a pipeline registered through
    ///   [`Self::register_mesh_pipeline`]. The per-stage threadgroup sizes come
    ///   from that registered pipeline; the wire only carries group counts.
    ///   A mesh dispatch outside a pass, without a mesh pipeline, or naming an
    ///   unknown handle fails before any command buffer exists.
    pub fn submit_v3_with_stats(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<(Reply, EncoderStats), MetalError> {
        if submission.sequence == 0 {
            return Err(MetalError::InvalidArg(
                "submission sequence must be nonzero",
            ));
        }
        if submission.commands.len() > crate::wire_v3::MAX_COMMANDS_V3 {
            return Err(MetalError::InvalidArg(
                "submission exceeds the v3 command limit",
            ));
        }
        if let Some(list) = submission.list
            && (list.list_id == 0 || list.list_generation == 0)
        {
            return Err(MetalError::InvalidArg(
                "list identity fields must be nonzero",
            ));
        }
        let reply = |status| Reply {
            sequence: submission.sequence,
            status,
        };
        let plan = plan_encoder_segments(&submission.commands)?;
        let _pool = AutoreleasePool::new();
        let mut state = self.lock_state()?;
        let list_key = submission
            .list
            .map(|list| (list.list_id, list.list_generation));
        if let Some((list_id, generation)) = list_key
            && state.list_was_executed(list_id, generation)
        {
            return Ok((reply(ReplyStatus::AlreadyExecuted), EncoderStats::default()));
        }
        for command in &submission.commands {
            if let CommandV3::Present(present) = command
                && !state.surfaces.contains_key(&present.surface_id)
            {
                return Ok((reply(ReplyStatus::UnknownHandle), EncoderStats::default()));
            }
        }

        let prepared = match prepare_v3(&state, allocations, &submission.commands, &plan) {
            Ok(prepared) => prepared,
            Err(PrepareAbort::UnknownHandle) => {
                return Ok((reply(ReplyStatus::UnknownHandle), EncoderStats::default()));
            }
            Err(PrepareAbort::Error(error)) => return Err(error),
        };

        let mut stats = EncoderStats::default();
        if prepared.is_empty() {
            if let Some((list_id, generation)) = list_key {
                state.mark_list_executed(list_id, generation);
            }
            return Ok((reply(ReplyStatus::Ok), stats));
        }

        let completed = self.encode_v3_segments(&prepared, &mut stats)?;
        if let Some((list_id, generation)) = list_key {
            state.mark_list_executed(list_id, generation);
        }
        Ok((
            reply(if completed {
                ReplyStatus::Ok
            } else {
                ReplyStatus::ExecutionFailed
            }),
            stats,
        ))
    }

    /// Encodes prepared segments onto one command buffer and waits for it.
    ///
    /// Returns whether the command buffer completed successfully.
    fn encode_v3_segments(
        &self,
        segments: &[PreparedSegment],
        stats: &mut EncoderStats,
    ) -> Result<bool, MetalError> {
        let _pool = AutoreleasePool::new();
        // SAFETY: `queue_raw()` is a valid `MTLCommandQueue`; the command buffer
        // is autoreleased and stays alive for this pool.
        let cb = unsafe { objc::msg_send_id_0(self.queue_raw(), objc::sel(c"commandBuffer")) };
        if cb.is_null() {
            return Err(MetalError::InternalError(
                "MTLCommandQueue.commandBuffer returned null".to_string(),
            ));
        }
        stats.command_buffers = 1;

        for segment in segments {
            match segment {
                PreparedSegment::Compute(ops) => {
                    // SAFETY: `cb` is a live uncommitted command buffer.
                    let enc =
                        unsafe { objc::msg_send_id_0(cb, objc::sel(c"computeCommandEncoder")) };
                    if enc.is_null() {
                        return Err(MetalError::InternalError(
                            "MTLCommandBuffer.computeCommandEncoder returned null".to_string(),
                        ));
                    }
                    stats.compute_encoders += 1;
                    for op in ops {
                        match op {
                            PreparedComputeOp::Pipeline(pipeline) => unsafe {
                                objc::msg_send_void_1(
                                    enc,
                                    objc::sel(c"setComputePipelineState:"),
                                    *pipeline,
                                );
                            },
                            PreparedComputeOp::Bytes { slot, bytes } => unsafe {
                                msg_send_set_bytes(enc, bytes.as_ptr(), bytes.len(), *slot);
                            },
                            PreparedComputeOp::Bind(bindings) => {
                                for (buffer, offset, slot) in bindings {
                                    // SAFETY: registry-held buffer, validated range.
                                    unsafe {
                                        objc::msg_send_set_buffer(enc, *buffer, *offset, *slot);
                                    }
                                }
                            }
                            PreparedComputeOp::Dispatch(dispatch) => {
                                // SAFETY: `enc`, the pipeline and every buffer
                                // are live for this encoder scope; ranges were
                                // bounds-checked and the grid fits the pipeline.
                                unsafe {
                                    objc::msg_send_void_1(
                                        enc,
                                        objc::sel(c"setComputePipelineState:"),
                                        dispatch.pipeline,
                                    );
                                    for (buffer, offset, slot) in &dispatch.bindings {
                                        objc::msg_send_set_buffer(enc, *buffer, *offset, *slot);
                                    }
                                    match &dispatch.grid {
                                        PreparedGrid::OneD {
                                            threads,
                                            threads_per_group,
                                        } => objc::msg_send_dispatch_threads(
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
                                        ),
                                        PreparedGrid::ThreeD {
                                            group_counts,
                                            threads_per_group,
                                        } => objc::msg_send_dispatch_threadgroups(
                                            enc,
                                            *group_counts,
                                            *threads_per_group,
                                        ),
                                    }
                                }
                            }
                            PreparedComputeOp::DispatchIndirect {
                                pipeline,
                                threads_per_group,
                                arguments,
                            } => {
                                // SAFETY: `enc` is a live compute encoder, the
                                // pipeline is registry-held and the argument
                                // buffer was bounds-checked in prepare.
                                unsafe {
                                    objc::msg_send_void_1(
                                        enc,
                                        objc::sel(c"setComputePipelineState:"),
                                        *pipeline,
                                    );
                                }
                                indirect::encode_dispatch_threadgroups(
                                    enc,
                                    *threads_per_group,
                                    arguments,
                                )?;
                            }
                            PreparedComputeOp::Barrier => {
                                // SAFETY: `enc` is a live compute encoder, and
                                // the scope covers every resource class.
                                unsafe {
                                    msg_send_void_usize(
                                        enc,
                                        objc::sel(c"memoryBarrierWithScope:"),
                                        MTL_BARRIER_SCOPE_BUFFERS | MTL_BARRIER_SCOPE_TEXTURES,
                                    );
                                }
                                stats.barriers += 1;
                            }
                        }
                    }
                    // SAFETY: `enc` is a live compute encoder awaiting its end.
                    unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
                }
                PreparedSegment::Render {
                    descriptor,
                    width,
                    height,
                    ops,
                } => {
                    if ops.iter().any(PreparedRenderOp::is_indirect) {
                        encode_indirect_render_segment(
                            cb, *descriptor, *width, *height, ops, stats,
                        )?;
                        continue;
                    }
                    let mut encoder =
                        RenderEncoder::new_with_descriptor(self, cb, *descriptor, *width, *height);
                    encoder.ensure_started()?;
                    stats.render_encoders += 1;
                    for op in ops {
                        match op {
                            PreparedRenderOp::Pipeline { state, depth_state } => {
                                encoder.set_pipeline_state(*state, *depth_state)?;
                            }
                            PreparedRenderOp::Viewport(viewport) => {
                                encoder.set_viewport(*viewport)?;
                            }
                            PreparedRenderOp::Scissor(scissor) => {
                                encoder.set_scissor_rect(*scissor)?;
                            }
                            PreparedRenderOp::VertexBuffers(bindings) => {
                                for (buffer, offset, slot) in bindings {
                                    // SAFETY: registry-held buffer, validated range.
                                    unsafe {
                                        encoder.set_vertex_buffer(*buffer, *offset, *slot)?;
                                    }
                                }
                            }
                            PreparedRenderOp::VertexBytes { slot, bytes } => {
                                encoder.set_vertex_bytes(bytes, *slot)?;
                            }
                            PreparedRenderOp::Barrier => {
                                encoder.memory_barrier(MTL_BARRIER_SCOPE_BUFFERS)?;
                                stats.barriers += 1;
                            }
                            PreparedRenderOp::Draw {
                                primitive,
                                vertex_start,
                                vertex_count,
                                instance_count,
                                base_instance,
                            } => {
                                encoder.draw_primitives_instanced(
                                    *primitive,
                                    *vertex_start,
                                    *vertex_count,
                                    *instance_count,
                                    *base_instance,
                                )?;
                            }
                            PreparedRenderOp::DrawIndexed {
                                primitive,
                                index_count,
                                index_type,
                                index_buffer,
                                index_offset,
                                base_vertex,
                                instance_count,
                                base_instance,
                            } => {
                                // SAFETY: index buffer is registry-held for the
                                // whole synchronous submission and the index
                                // window was validated.
                                unsafe {
                                    encoder.draw_indexed_primitives_instanced(
                                        *primitive,
                                        *index_count,
                                        *index_type,
                                        *index_buffer,
                                        *index_offset,
                                        *instance_count,
                                        *base_vertex,
                                        *base_instance,
                                    )?;
                                }
                            }
                            PreparedRenderOp::DispatchMesh {
                                group_counts,
                                threads_per_object_threadgroup,
                                threads_per_mesh_threadgroup,
                                bindings,
                            } => {
                                for (buffer, offset, slot) in bindings {
                                    // SAFETY: registry-held buffer, validated range.
                                    unsafe {
                                        encoder.set_mesh_buffer(*buffer, *offset, *slot)?;
                                    }
                                }
                                encoder.draw_mesh_threadgroups(
                                    *group_counts,
                                    *threads_per_object_threadgroup,
                                    *threads_per_mesh_threadgroup,
                                )?;
                            }
                            PreparedRenderOp::DrawIndirect { .. }
                            | PreparedRenderOp::DrawIndexedIndirect { .. }
                            | PreparedRenderOp::DispatchMeshIndirect { .. } => {
                                return Err(MetalError::InternalError(
                                    "indirect render op reached the non-indirect executor"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                    encoder.end_encoding();
                }
                PreparedSegment::Blit(ops) => {
                    // SAFETY: `cb` is a live uncommitted command buffer.
                    let mut enc = unsafe { objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder")) };
                    if enc.is_null() {
                        return Err(MetalError::InternalError(
                            "MTLCommandBuffer.blitCommandEncoder returned null".to_string(),
                        ));
                    }
                    stats.blit_encoders += 1;
                    for op in ops {
                        match op {
                            PreparedBlitOp::CopyBuffer {
                                src,
                                src_offset,
                                dst,
                                dst_offset,
                                length,
                            } => {
                                // SAFETY: both buffers are registry-held for the
                                // synchronous submission and both ranges fit.
                                unsafe {
                                    msg_send_copy_buffer(
                                        enc,
                                        *src,
                                        *src_offset,
                                        *dst,
                                        *dst_offset,
                                        *length,
                                    );
                                }
                            }
                            PreparedBlitOp::CopyTexture {
                                src,
                                src_slice,
                                src_level,
                                dst,
                                dst_slice,
                                dst_level,
                                width,
                                height,
                                depth,
                            } => {
                                // SAFETY: both textures are registry-held and
                                // every mip/slice/extent was validated.
                                unsafe {
                                    msg_send_copy_texture(
                                        enc,
                                        *src,
                                        *src_slice,
                                        *src_level,
                                        HostOrigin { x: 0, y: 0, z: 0 },
                                        MTLSize {
                                            width: *width,
                                            height: *height,
                                            depth: *depth,
                                        },
                                        *dst,
                                        *dst_slice,
                                        *dst_level,
                                        HostOrigin { x: 0, y: 0, z: 0 },
                                    );
                                }
                            }
                            PreparedBlitOp::CopyTextureRegion(region) => {
                                blit::copy_texture_region(enc, region)?;
                            }
                            PreparedBlitOp::ResolveSubresource { src, dst } => {
                                // A resolve is a render-pass store action, so
                                // the blit encoder must close first and a fresh
                                // blit encoder resumes any later copies.
                                // SAFETY: `enc` is the live blit encoder opened
                                // above and no work is recorded after this call.
                                unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
                                blit::encode_resolve(cb, *src, *dst)?;
                                stats.render_encoders += 1;
                                // SAFETY: `cb` is the live uncommitted command
                                // buffer used for every segment.
                                enc = unsafe {
                                    objc::msg_send_id_0(cb, objc::sel(c"blitCommandEncoder"))
                                };
                                if enc.is_null() {
                                    return Err(MetalError::InternalError(
                                        "MTLCommandBuffer.blitCommandEncoder returned null"
                                            .to_string(),
                                    ));
                                }
                                stats.blit_encoders += 1;
                            }
                            PreparedBlitOp::Barrier => {
                                // The blit encoder has no barrier API; the
                                // encoder boundary is the fence.
                                stats.barriers += 1;
                            }
                        }
                    }
                    // SAFETY: `enc` is a live blit encoder awaiting its end.
                    unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
                }
            }
        }

        // SAFETY: `cb` has every encoder closed.
        unsafe {
            objc::msg_send_void_0(cb, objc::sel(c"commit"));
            objc::msg_send_void_0(cb, objc::sel(c"waitUntilCompleted"));
        }
        // SAFETY: `cb` completed via `waitUntilCompleted`.
        let status = unsafe { objc::msg_send_usize_0(cb, objc::sel(c"status")) };
        Ok(status == MTL_COMMAND_BUFFER_STATUS_COMPLETED)
    }
}

/// Apple `MTLViewport` for the indirect-aware render executor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HostViewport {
    origin_x: f64,
    origin_y: f64,
    width: f64,
    height: f64,
    znear: f64,
    zfar: f64,
}

/// Apple `MTLScissorRect` for the indirect-aware render executor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HostScissorRect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// Replays a render segment that contains an indirect command.
///
/// [`RenderEncoder`] owns its raw `enc` privately and `render.rs` is outside
/// this task's write scope, so this executor opens the same
/// `MTLRenderPassDescriptor` on the shared command buffer directly and replays
/// the already-validated [`PreparedRenderOp`] stream. Segments without an
/// indirect command stay on [`RenderEncoder`].
fn encode_indirect_render_segment(
    cb: objc::Id,
    descriptor: objc::Id,
    width: u32,
    height: u32,
    ops: &[PreparedRenderOp],
    stats: &mut EncoderStats,
) -> Result<(), MetalError> {
    let _pool = AutoreleasePool::new();
    // SAFETY: `cb` is a live uncommitted command buffer and `descriptor` is a
    // live render pass descriptor owned by the prepared segment.
    let enc = unsafe {
        objc::msg_send_id_1(
            cb,
            objc::sel(c"renderCommandEncoderWithDescriptor:"),
            descriptor,
        )
    };
    if enc.is_null() {
        return Err(MetalError::InternalError(
            "MTLCommandBuffer.renderCommandEncoderWithDescriptor returned null".to_string(),
        ));
    }
    stats.render_encoders += 1;

    // SAFETY: `enc` is a live render encoder; the defaults cover the whole
    // attachment exactly like `RenderEncoder::ensure_encoder`.
    unsafe {
        msg_send_set_viewport(
            enc,
            objc::sel(c"setViewport:"),
            HostViewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: f64::from(width),
                height: f64::from(height),
                znear: 0.0,
                zfar: 1.0,
            },
        );
        msg_send_set_scissor(
            enc,
            objc::sel(c"setScissorRect:"),
            HostScissorRect {
                x: 0,
                y: 0,
                width: width as usize,
                height: height as usize,
            },
        );
    }

    for op in ops {
        match op {
            PreparedRenderOp::Pipeline { state, depth_state } => {
                // SAFETY: both objects are registry-held pipeline states.
                unsafe {
                    objc::msg_send_void_1(enc, objc::sel(c"setRenderPipelineState:"), *state);
                    if let Some(depth_state) = depth_state {
                        objc::msg_send_void_1(
                            enc,
                            objc::sel(c"setDepthStencilState:"),
                            *depth_state,
                        );
                    }
                }
            }
            PreparedRenderOp::Viewport(viewport) => {
                // SAFETY: `enc` is live and the viewport was validated against
                // the attachment extent.
                unsafe {
                    msg_send_set_viewport(
                        enc,
                        objc::sel(c"setViewport:"),
                        HostViewport {
                            origin_x: viewport.origin_x,
                            origin_y: viewport.origin_y,
                            width: viewport.width,
                            height: viewport.height,
                            znear: viewport.znear,
                            zfar: viewport.zfar,
                        },
                    );
                }
            }
            PreparedRenderOp::Scissor(scissor) => {
                // SAFETY: `enc` is live and the scissor was validated.
                unsafe {
                    msg_send_set_scissor(
                        enc,
                        objc::sel(c"setScissorRect:"),
                        HostScissorRect {
                            x: scissor.x as usize,
                            y: scissor.y as usize,
                            width: scissor.width as usize,
                            height: scissor.height as usize,
                        },
                    );
                }
            }
            PreparedRenderOp::VertexBuffers(bindings) => {
                for (buffer, offset, slot) in bindings {
                    // SAFETY: registry-held buffer, validated range.
                    unsafe { msg_send_set_vertex_buffer(enc, *buffer, *offset, *slot) };
                }
            }
            PreparedRenderOp::VertexBytes { slot, bytes } => {
                // SAFETY: `bytes` is valid for the call; Metal copies it.
                unsafe {
                    msg_send_set_vertex_bytes(enc, bytes.as_ptr(), bytes.len(), *slot);
                }
            }
            PreparedRenderOp::Barrier => {
                // SAFETY: `enc` is live; the scope is a documented mask.
                unsafe {
                    msg_send_memory_barrier(
                        enc,
                        MTL_BARRIER_SCOPE_BUFFERS | MTL_BARRIER_SCOPE_TEXTURES,
                    );
                }
                stats.barriers += 1;
            }
            PreparedRenderOp::Draw {
                primitive,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                // SAFETY: every count was validated in prepare.
                unsafe {
                    msg_send_draw_primitives_instanced(
                        enc,
                        *primitive as usize,
                        *vertex_start,
                        *vertex_count,
                        *instance_count,
                        *base_instance,
                    );
                }
            }
            PreparedRenderOp::DrawIndexed {
                primitive,
                index_count,
                index_type,
                index_buffer,
                index_offset,
                base_vertex,
                instance_count,
                base_instance,
            } => {
                // SAFETY: the index window was validated in prepare.
                unsafe {
                    msg_send_draw_indexed_primitives_instanced(
                        enc,
                        *primitive as usize,
                        *index_count,
                        *index_type as usize,
                        *index_buffer,
                        *index_offset,
                        *instance_count,
                        *base_vertex,
                        *base_instance,
                    );
                }
            }
            PreparedRenderOp::DispatchMesh {
                group_counts,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
                bindings,
            } => {
                for (buffer, offset, slot) in bindings {
                    // SAFETY: registry-held buffer, validated range.
                    unsafe { msg_send_set_mesh_buffer(enc, *buffer, *offset, *slot) };
                }
                // SAFETY: the mesh pipeline is selected and the group counts
                // fit the pipeline's declared limits.
                unsafe {
                    msg_send_draw_mesh_threadgroups(
                        enc,
                        *group_counts,
                        *threads_per_object_threadgroup,
                        *threads_per_mesh_threadgroup,
                    );
                }
            }
            PreparedRenderOp::DrawIndirect {
                primitive,
                arguments,
            } => indirect::encode_draw(enc, *primitive, arguments)?,
            PreparedRenderOp::DrawIndexedIndirect {
                primitive,
                index_type,
                index_buffer,
                index_offset,
                arguments,
            } => indirect::encode_draw_indexed(
                enc,
                *primitive,
                *index_type,
                *index_buffer,
                *index_offset,
                arguments,
            )?,
            PreparedRenderOp::DispatchMeshIndirect {
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
                arguments,
            } => indirect::encode_dispatch_mesh(
                enc,
                *threads_per_object_threadgroup,
                *threads_per_mesh_threadgroup,
                arguments,
            )?,
        }
    }

    // SAFETY: `enc` is a live render encoder awaiting its end.
    unsafe { objc::msg_send_void_0(enc, objc::sel(c"endEncoding")) };
    Ok(())
}

/// `-[MTLRenderCommandEncoder setViewport:]`.
unsafe fn msg_send_set_viewport(enc: objc::Id, s: objc::Sel, viewport: HostViewport) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, HostViewport);
    // SAFETY: Caller supplies a live render encoder and a valid `MTLViewport`.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe { send(enc, s, viewport) }
}

/// `-[MTLRenderCommandEncoder setScissorRect:]`.
unsafe fn msg_send_set_scissor(enc: objc::Id, s: objc::Sel, scissor: HostScissorRect) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, HostScissorRect);
    // SAFETY: Caller supplies a live render encoder and a valid `MTLScissorRect`.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe { send(enc, s, scissor) }
}

/// `-[MTLRenderCommandEncoder setVertexBuffer:offset:atIndex:]`.
unsafe fn msg_send_set_vertex_buffer(
    enc: objc::Id,
    buffer: objc::Id,
    offset: usize,
    index: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, usize);
    // SAFETY: Caller supplies a live render encoder and a registry-held buffer.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(c"setVertexBuffer:offset:atIndex:"),
            buffer,
            offset,
            index,
        )
    }
}

/// `-[MTLRenderCommandEncoder setMeshBuffer:offset:atIndex:]`.
unsafe fn msg_send_set_mesh_buffer(
    enc: objc::Id,
    buffer: objc::Id,
    offset: usize,
    index: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, usize);
    // SAFETY: Caller supplies a live render encoder and a registry-held buffer.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(c"setMeshBuffer:offset:atIndex:"),
            buffer,
            offset,
            index,
        )
    }
}

/// `-[MTLRenderCommandEncoder setVertexBytes:length:atIndex:]`.
unsafe fn msg_send_set_vertex_bytes(
    enc: objc::Id,
    bytes: *const u8,
    len: usize,
    index: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, *const c_void, usize, usize);
    // SAFETY: Caller supplies a live render encoder and `bytes` valid for `len`.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(c"setVertexBytes:length:atIndex:"),
            bytes.cast(),
            len,
            index,
        )
    }
}

/// `-[MTLRenderCommandEncoder drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:]`.
unsafe fn msg_send_draw_primitives_instanced(
    enc: objc::Id,
    primitive: usize,
    vertex_start: usize,
    vertex_count: usize,
    instance_count: usize,
    base_instance: usize,
) {
    type Fn =
        unsafe extern "C" fn(objc::Id, objc::Sel, usize, usize, usize, usize, usize);
    // SAFETY: Caller supplies a live render encoder and validated counts.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(c"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
            primitive,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        )
    }
}

/// `-[MTLRenderCommandEncoder drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:]`.
#[allow(clippy::too_many_arguments)]
unsafe fn msg_send_draw_indexed_primitives_instanced(
    enc: objc::Id,
    primitive: usize,
    index_count: usize,
    index_type: usize,
    index_buffer: objc::Id,
    index_offset: usize,
    instance_count: usize,
    base_vertex: isize,
    base_instance: usize,
) {
    type Fn = unsafe extern "C" fn(
        objc::Id,
        objc::Sel,
        usize,
        usize,
        usize,
        objc::Id,
        usize,
        usize,
        isize,
        usize,
    );
    // SAFETY: Caller supplies a live render encoder, a registry-held index
    // buffer and a validated index window.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(
                c"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
            ),
            primitive,
            index_count,
            index_type,
            index_buffer,
            index_offset,
            instance_count,
            base_vertex,
            base_instance,
        )
    }
}

/// `-[MTLRenderCommandEncoder drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:]`.
unsafe fn msg_send_draw_mesh_threadgroups(
    enc: objc::Id,
    group_counts: MTLSize,
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, MTLSize, MTLSize, MTLSize);
    // SAFETY: Caller supplies a live render encoder with a mesh pipeline.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe {
        send(
            enc,
            objc::sel(
                c"drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
            ),
            group_counts,
            threads_per_object_threadgroup,
            threads_per_mesh_threadgroup,
        )
    }
}

/// `-[MTLRenderCommandEncoder memoryBarrierWithScope:]`.
unsafe fn msg_send_memory_barrier(enc: objc::Id, scope: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
    // SAFETY: Caller supplies a live render encoder and a barrier scope mask.
    let send: Fn = unsafe { std::mem::transmute(objc::msg_send_entry()) };
    // SAFETY: `send` matches the selector's documented argument layout.
    unsafe { send(enc, objc::sel(c"memoryBarrierWithScope:"), scope) }
}


unsafe fn msg_send_set_bytes(enc: objc::Id, bytes: *const u8, len: usize, index: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, *const c_void, usize, usize);
    // SAFETY: Caller supplies a live compute encoder and `bytes` valid for
    // `len` bytes during the call; Metal copies the payload immediately.
    let f: Fn = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            enc,
            objc::sel(c"setBytes:length:atIndex:"),
            bytes.cast(),
            len,
            index,
        )
    }
}

/// `-[MTLBlitCommandEncoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:]`.
unsafe fn msg_send_copy_buffer(
    enc: objc::Id,
    src: objc::Id,
    src_offset: usize,
    dst: objc::Id,
    dst_offset: usize,
    size: usize,
) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, objc::Id, usize, usize);
    // SAFETY: Caller supplies a live blit encoder and two live buffers whose
    // ranges were validated against the registry.
    let f: Fn = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            enc,
            objc::sel(c"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        )
    }
}

/// `-[MTLBlitCommandEncoder copyFromTexture:...toTexture:...]` for 2D slices.
#[allow(clippy::too_many_arguments)]
unsafe fn msg_send_copy_texture(
    enc: objc::Id,
    src: objc::Id,
    src_slice: usize,
    src_level: usize,
    src_origin: HostOrigin,
    src_size: MTLSize,
    dst: objc::Id,
    dst_slice: usize,
    dst_level: usize,
    dst_origin: HostOrigin,
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
        HostOrigin,
    );
    // SAFETY: Caller supplies a live blit encoder and two live textures whose
    // formats, mips, slices and extents were validated.
    let f: Fn = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe {
        f(
            enc,
            objc::sel(c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
            src,
            src_slice,
            src_level,
            src_origin,
            src_size,
            dst,
            dst_slice,
            dst_level,
            dst_origin,
        )
    }
}

/// Sends a selector taking one `NSUInteger` and returning `void`.
unsafe fn msg_send_void_usize(target: objc::Id, s: objc::Sel, value: usize) {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, usize);
    // SAFETY: Caller supplies a valid receiver/selector and an `NSUInteger`.
    let f: Fn = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: `f` matches the selector's documented argument layout.
    unsafe { f(target, s, value) }
}

impl MetalTransport for DirectMetalTransport {
    fn device_info(&self) -> Result<VmDeviceInfo, MetalError> {
        Ok(self.info)
    }

    unsafe fn register_zerocopy(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        if ptr.is_null() || len == 0 {
            return Err(MetalError::InvalidArg("buffer pointer is null or len == 0"));
        }
        let page_size =
            usize::try_from(self.info.page_size).unwrap_or(APPLE_SILICON_NOCOPY_PAGE_SIZE);
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
        let src_ns = objc::nsstring(source).ok_or(MetalError::InvalidArg(
            "MSL source string allocation failed",
        ))?;

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
            objc::msg_send_void_1(enc, objc::sel(c"setComputePipelineState:"), pso.as_raw());
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
    info.recommended_max_working_set_size = unsafe {
        objc::msg_send_u64_0(device.as_raw(), objc::sel(c"recommendedMaxWorkingSetSize"))
    };

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

    use super::{DirectMetalTransport, EncoderStats, read_back_render_target};
    use crate::abi::VmBufferId;
    use crate::macos::msc::{MscUavDispatch, MscUavLayout};
    use crate::macos::objc;
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
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(raw.cast::<u8>(), HOST_PAGE_BYTES) };
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
        let frame =
            wire::encode_submission(&submission).expect("graphics submission should encode");
        let decoded = wire::decode_submission(&frame).expect("graphics submission should decode");
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
    // ---- ADR-006: one command buffer, contiguous encoder segments ----------

    use crate::macos::present::SurfaceImage;
    use crate::macos::render::{GraphicsPipeline, RenderTarget};
    use crate::wire_v3::{
        BeginRenderPass, ColorAttachment, CommandV3, CopyBuffer, CopyTexture,
        DepthStencilAttachment, DrawIndexedInstanced, DrawIndexedState, DrawInstanced, DrawState,
        IndexBufferBinding, IndexFormat, LoadOp, PipelineRef, Present, PrimitiveTopology,
        RootConstantsV3, ScissorRect as WireScissor, StoreOp, SubmissionV3, VertexBuffer,
        VertexBufferSet, Viewport as WireViewport,
    };

    const V3_EDGE: u32 = 64;
    const V3_PIXELS: usize = (V3_EDGE * V3_EDGE) as usize;
    const V3_RED: [u8; 4] = [0, 0, 255, 255];
    const V3_GREEN: [u8; 4] = [0, 255, 0, 255];
    const V3_BLUE: [u8; 4] = [255, 0, 0, 255];
    const V3_WHITE: [u8; 4] = [255, 255, 255, 255];
    const V3_CYAN: [u8; 4] = [255, 255, 0, 255];
    const V3_MAGENTA: [u8; 4] = [255, 0, 255, 255];
    const V3_YELLOW: [u8; 4] = [0, 255, 255, 255];
    const V3_PAGE_BYTES: usize = 16 * 1024;
    const V3_INDEX_PAGE_U16_AT: usize = 0;
    const V3_INDEX_PAGE_U32_AT: usize = 16;
    const V3_VERTEX_A_AT: usize = 32;
    const V3_VERTEX_B_AT: usize = 80;
    const V3_VERTEX_C_AT: usize = 128;
    /// Three `float4` positions (16-byte stride, matching MSL `float4`).
    const V3_VERTEX_BYTES: usize = 3 * 16;

    /// 16 KiB-aligned host page used by `register_zerocopy`.
    struct V3Page {
        ptr: *mut c_void,
        len: usize,
    }

    impl V3Page {
        fn new(len: usize) -> Self {
            unsafe extern "C" {
                fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            }
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
            let rc = unsafe { posix_memalign(&mut raw, 16 * 1024, len) };
            assert_eq!(rc, 0, "posix_memalign failed");
            assert!(!raw.is_null());
            // SAFETY: `raw` is a fresh allocation of `len` writable bytes.
            unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, len) };
            Self { ptr: raw, len }
        }

        fn raw(&self) -> *mut u8 {
            self.ptr.cast()
        }

        fn bytes_mut(&mut self) -> &mut [u8] {
            // SAFETY: `self.ptr` is a live allocation of `self.len` bytes.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) }
        }

        fn words(&self) -> &[u32] {
            // SAFETY: The allocation is 16 KiB-aligned and `self.len` is a
            // multiple of 4 for every caller.
            unsafe { std::slice::from_raw_parts(self.ptr.cast::<u32>(), self.len / 4) }
        }
    }

    impl Drop for V3Page {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn free(ptr: *mut c_void);
            }
            // SAFETY: `self.ptr` came from `posix_memalign` and is no longer
            // registered with the transport when this guard drops.
            unsafe { free(self.ptr) };
        }
    }

    /// Retains an IOSurface-backed texture as an owning `RenderTarget`.
    fn retained_surface_target(image: &SurfaceImage) -> RenderTarget {
        // SAFETY: `texture_raw` is a live retained `MTLTexture` owned by `image`;
        // `retain` adds a reference that `RenderTarget` takes ownership of.
        let raw = unsafe { objc::msg_send_id_0(image.texture_raw(), objc::sel(c"retain")) };
        assert!(!raw.is_null(), "surface texture retain returned nil");
        // SAFETY: `raw` is the +1 reference created above.
        unsafe { RenderTarget::from_retained_raw(raw) }.expect("surface texture should wrap")
    }

    fn v3_color_attachment(allocation_id: u64, load: LoadOp) -> ColorAttachment {
        ColorAttachment {
            allocation_id,
            generation: 1,
            mip_level: 0,
            slice: 0,
            load,
            store: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn v3_float4(values: [f32; 4]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// ADR-006 hardware proof: one submission mixes compute, two MRT render
    /// passes with depth and scissor, texture and buffer blits, an explicit
    /// barrier, a checksum compute over the first target's pixels and a present.
    #[test]
    fn v3_mixed_segments_one_command_buffer_mrt_depth_scissor_and_checksum() {
        const V3_MSL: &str = r#"
            #include <metal_stdlib>
            using namespace metal;

            struct VOut {
                float4 position [[position]];
                float4 color0 [[flat]];
                float4 color1 [[flat]];
            };

            vertex VOut vm_v3_vertex(uint vid [[vertex_id]],
                                     constant float4 *positions [[buffer(0)]],
                                     constant float4 &color0 [[buffer(1)]],
                                     constant float4 &color1 [[buffer(2)]]) {
                VOut out;
                out.position = positions[vid];
                out.color0 = color0;
                out.color1 = color1;
                return out;
            }

            struct MrtOut {
                float4 color0 [[color(0)]];
                float4 color1 [[color(1)]];
            };

            fragment MrtOut vm_v3_fragment_mrt(VOut in [[stage_in]]) {
                MrtOut out;
                out.color0 = in.color0;
                out.color1 = in.color1;
                return out;
            }

            fragment float4 vm_v3_fragment_single(VOut in [[stage_in]]) {
                return in.color0;
            }

            kernel void vm_v3_fill(device uint *data [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
                data[tid] = tid + 1u;
            }

            kernel void vm_v3_checksum(device const uint *src [[buffer(0)]],
                                       device uint *out [[buffer(1)]],
                                       uint tid [[thread_position_in_grid]]) {
                if (tid == 0u) {
                    uint sum = 0u;
                    for (uint i = 0u; i < 4096u; ++i) {
                        sum += src[i];
                    }
                    out[0] = sum;
                }
            }
        "#;

        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let mut image0 = SurfaceImage::create(&transport, V3_EDGE, V3_EDGE)
            .expect("first IOSurface target should be created");
        let mut image1 = SurfaceImage::create(&transport, V3_EDGE, V3_EDGE)
            .expect("second IOSurface target should be created");
        let target0 = retained_surface_target(&image0);
        let target1 = retained_surface_target(&image1);
        let depth = transport
            .create_render_target_2d(V3_EDGE, V3_EDGE, PixelFormat::Depth32Float)
            .expect("depth target should be created");
        let staging = transport
            .create_shared_texture_2d(V3_EDGE, V3_EDGE, PixelFormat::Bgra8Unorm)
            .expect("shared staging texture should be created");
        assert_eq!(staging.bytes_per_row, V3_EDGE as usize * 4);
        assert_eq!(staging.len, V3_PAGE_BYTES);

        let t0 = transport
            .register_render_target(&target0)
            .expect("first target should register");
        let t1 = transport
            .register_render_target(&target1)
            .expect("second target should register");
        let depth_id = transport
            .register_render_target(&depth)
            .expect("depth target should register");
        transport
            .register_present_surface(u64::from(image1.surface_id()), &target1)
            .expect("present surface should register");

        // One page holds both index widths and all three vertex sets.
        let mut geometry = V3Page::new(V3_PAGE_BYTES);
        let bytes = geometry.bytes_mut();
        let u16_indices: [u16; 3] = [0, 1, 2];
        for (index, value) in u16_indices.iter().enumerate() {
            let at = V3_INDEX_PAGE_U16_AT + index * 2;
            bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
        }
        let u32_indices: [u32; 3] = [0, 1, 2];
        for (index, value) in u32_indices.iter().enumerate() {
            let at = V3_INDEX_PAGE_U32_AT + index * 4;
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        // A: full-screen triangle at z=0.9. B: centre triangle at z=0.1.
        // C: full-screen triangle at z=0.95, drawn last and rejected by depth.
        let vertices: [(&[f32; 12], usize); 3] = [
            (
                &[
                    -1.0f32, -1.0, 0.9, 1.0, 3.0, -1.0, 0.9, 1.0, -1.0, 3.0, 0.9, 1.0,
                ],
                V3_VERTEX_A_AT,
            ),
            (
                &[
                    -0.5f32, -0.5, 0.1, 1.0, 0.5, -0.5, 0.1, 1.0, 0.0, 0.5, 0.1, 1.0,
                ],
                V3_VERTEX_B_AT,
            ),
            (
                &[
                    -1.0f32, -1.0, 0.95, 1.0, 3.0, -1.0, 0.95, 1.0, -1.0, 3.0, 0.95, 1.0,
                ],
                V3_VERTEX_C_AT,
            ),
        ];
        for (positions, at) in vertices {
            for (index, value) in positions.iter().enumerate() {
                let offset = at + index * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let geometry_id = unsafe { transport.register_zerocopy(geometry.raw(), V3_PAGE_BYTES) }
            .expect("geometry page should register");

        let data = V3Page::new(V3_PAGE_BYTES);
        let data_id = unsafe { transport.register_zerocopy(data.raw(), V3_PAGE_BYTES) }
            .expect("stage-one buffer should register");
        let copy_dst = V3Page::new(V3_PAGE_BYTES);
        let copy_dst_id = unsafe { transport.register_zerocopy(copy_dst.raw(), V3_PAGE_BYTES) }
            .expect("copy destination buffer should register");
        let checksum = V3Page::new(V3_PAGE_BYTES);
        let checksum_id = unsafe { transport.register_zerocopy(checksum.raw(), V3_PAGE_BYTES) }
            .expect("checksum buffer should register");

        let fill_pipeline = transport
            .compile_msl(V3_MSL, "vm_v3_fill")
            .expect("fill kernel should compile");
        let checksum_pipeline = transport
            .compile_msl(V3_MSL, "vm_v3_checksum")
            .expect("checksum kernel should compile");
        let mrt_pipeline = GraphicsPipeline::compile_mrt(
            &transport,
            V3_MSL,
            "vm_v3_vertex",
            "vm_v3_fragment_mrt",
            &[PixelFormat::Bgra8Unorm, PixelFormat::Bgra8Unorm],
            Some(PixelFormat::Depth32Float),
        )
        .expect("MRT pipeline with depth should compile");
        let single_pipeline = GraphicsPipeline::compile_mrt(
            &transport,
            V3_MSL,
            "vm_v3_vertex",
            "vm_v3_fragment_single",
            &[PixelFormat::Bgra8Unorm],
            None,
        )
        .expect("single-target pipeline should compile");
        let mrt_id = transport
            .register_graphics_pipeline_state(mrt_pipeline)
            .expect("MRT pipeline should register");
        let single_id = transport
            .register_graphics_pipeline_state(single_pipeline)
            .expect("single-target pipeline should register");
        assert_ne!(mrt_id.0, 0);
        assert_ne!(single_id.0, 0);

        let vertex = |offset: usize| VertexBuffer {
            allocation_id: 8,
            generation: 1,
            offset: offset as u64,
            stride: 16,
            length: V3_VERTEX_BYTES as u64,
        };
        let viewport = WireViewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(V3_EDGE),
            height: f64::from(V3_EDGE),
            near: 0.0,
            far: 1.0,
        };
        let full_scissor = WireScissor {
            x: 0,
            y: 0,
            width: V3_EDGE,
            height: V3_EDGE,
        };
        let half_scissor = WireScissor {
            x: 0,
            y: 0,
            width: V3_EDGE / 2,
            height: V3_EDGE,
        };
        let u16_index = IndexBufferBinding {
            allocation_id: 8,
            generation: 1,
            offset: V3_INDEX_PAGE_U16_AT as u64,
            length: 6,
            format: IndexFormat::Uint16,
        };
        let u32_index = IndexBufferBinding {
            allocation_id: 8,
            generation: 1,
            offset: V3_INDEX_PAGE_U32_AT as u64,
            length: 12,
            format: IndexFormat::Uint32,
        };
        let constants = |slot: u16, color: [f32; 4]| {
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot,
                bytes: v3_float4(color),
            })
        };

        let mut commands = Vec::new();
        commands.push(CommandV3::Dispatch1d(wire::Dispatch1d {
            pipeline_id: fill_pipeline.0,
            threads: 4096,
            threads_per_group: 64,
            bindings: vec![wire::BufferBinding {
                allocation_id: 5,
                generation: 1,
                offset: 0,
                length: V3_PAGE_BYTES as u64,
            }],
        }));
        commands.push(CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                v3_color_attachment(1, LoadOp::Clear),
                v3_color_attachment(2, LoadOp::Clear),
            ],
            depth_stencil: Some(DepthStencilAttachment {
                allocation_id: 3,
                generation: 1,
                mip_level: 0,
                slice: 0,
                depth_load: LoadOp::Clear,
                depth_store: StoreOp::Store,
                clear_depth: 1.0,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                clear_stencil: 0,
            }),
        }));
        commands.push(CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: mrt_id.0,
        }));
        commands.push(CommandV3::SetViewport(viewport));
        commands.push(CommandV3::SetScissor(full_scissor));
        commands.push(CommandV3::SetPrimitiveTopology(
            PrimitiveTopology::TriangleList,
        ));
        commands.push(CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: vec![vertex(V3_VERTEX_A_AT)],
        }));
        commands.push(CommandV3::SetIndexBuffer(u16_index));
        commands.push(constants(1, [1.0, 0.0, 0.0, 1.0]));
        commands.push(constants(2, [0.0, 1.0, 1.0, 1.0]));
        commands.push(CommandV3::DrawIndexed(DrawIndexedState {
            index_count: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }));
        commands.push(CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: vec![vertex(V3_VERTEX_B_AT)],
        }));
        commands.push(constants(1, [0.0, 1.0, 0.0, 1.0]));
        commands.push(constants(2, [1.0, 0.0, 1.0, 1.0]));
        commands.push(CommandV3::DrawInstanced(DrawInstanced {
            vertex_count_per_instance: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }));
        commands.push(CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: vec![vertex(V3_VERTEX_C_AT)],
        }));
        commands.push(CommandV3::SetIndexBuffer(u32_index));
        commands.push(constants(1, [0.0, 0.0, 1.0, 1.0]));
        commands.push(constants(2, [1.0, 1.0, 1.0, 1.0]));
        commands.push(CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
            index_count_per_instance: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }));
        commands.push(CommandV3::EndRenderPass);
        commands.push(CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![v3_color_attachment(1, LoadOp::Load)],
            depth_stencil: None,
        }));
        commands.push(CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: single_id.0,
        }));
        commands.push(CommandV3::SetViewport(viewport));
        commands.push(CommandV3::SetScissor(half_scissor));
        commands.push(CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: vec![vertex(V3_VERTEX_C_AT)],
        }));
        commands.push(constants(1, [1.0, 1.0, 0.0, 1.0]));
        commands.push(constants(2, [1.0, 1.0, 0.0, 1.0]));
        commands.push(CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }));
        commands.push(CommandV3::EndRenderPass);
        commands.push(CommandV3::CopyTexture(CopyTexture {
            src_allocation_id: 1,
            src_generation: 1,
            src_mip: 0,
            src_slice: 0,
            dst_allocation_id: 4,
            dst_generation: 1,
            dst_mip: 0,
            dst_slice: 0,
            width: V3_EDGE,
            height: V3_EDGE,
            depth: 1,
        }));
        commands.push(CommandV3::CopyBuffer(CopyBuffer {
            src_allocation_id: 5,
            src_generation: 1,
            src_offset: 0,
            dst_allocation_id: 6,
            dst_generation: 1,
            dst_offset: 0,
            length: V3_PAGE_BYTES as u64,
        }));
        commands.push(CommandV3::ResourceBarrier(wire::ResourceBarrier {
            entries: vec![wire::BarrierEntry {
                allocation_id: 4,
                generation: 1,
                offset: 0,
                length: V3_PAGE_BYTES as u64,
                before_state: 0,
                after_state: 1,
            }],
        }));
        commands.push(CommandV3::Dispatch1d(wire::Dispatch1d {
            pipeline_id: checksum_pipeline.0,
            threads: 1,
            threads_per_group: 1,
            bindings: vec![
                wire::BufferBinding {
                    allocation_id: 4,
                    generation: 1,
                    offset: 0,
                    length: V3_PAGE_BYTES as u64,
                },
                wire::BufferBinding {
                    allocation_id: 7,
                    generation: 1,
                    offset: 0,
                    length: 16,
                },
            ],
        }));
        commands.push(CommandV3::Present(Present {
            surface_id: u64::from(image1.surface_id()),
            sync_interval: 1,
        }));

        let mut allocations = HashMap::new();
        allocations.insert((1u64, 1u32), VmBufferId(t0.0));
        allocations.insert((2u64, 1u32), VmBufferId(t1.0));
        allocations.insert((3u64, 1u32), VmBufferId(depth_id.0));
        allocations.insert((4u64, 1u32), staging.buffer_id);
        allocations.insert((5u64, 1u32), data_id);
        allocations.insert((6u64, 1u32), copy_dst_id);
        allocations.insert((7u64, 1u32), checksum_id);
        allocations.insert((8u64, 1u32), geometry_id);

        let submission = SubmissionV3 {
            sequence: 41,
            list: Some(wire::ListIdentity {
                list_id: 5,
                list_generation: 1,
            }),
            commands,
        };

        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &allocations)
            .expect("v3 submission should execute");
        assert_eq!(
            reply,
            Reply {
                sequence: 41,
                status: ReplyStatus::Ok,
            }
        );
        assert_eq!(
            stats,
            EncoderStats {
                command_buffers: 1,
                compute_encoders: 2,
                render_encoders: 2,
                blit_encoders: 1,
                barriers: 1,
            },
            "ADR-006 segmentation must be observable"
        );

        // Stage one wrote exactly 4096 dwords; the CopyBuffer duplicated them.
        for (index, &value) in data.words().iter().enumerate() {
            assert_eq!(value, index as u32 + 1, "fill mismatch at dword {index}");
        }
        assert_eq!(
            copy_dst.words(),
            data.words(),
            "CopyBuffer must duplicate the fill stage on the same command buffer"
        );

        let rows0 =
            read_back_render_target(&transport, &target0).expect("first target should read back");
        let rows1 =
            read_back_render_target(&transport, &target1).expect("second target should read back");
        assert_eq!(rows0.len(), V3_PIXELS * 4);
        assert_eq!(rows1.len(), V3_PIXELS * 4);
        let pixels0: Vec<[u8; 4]> = rows0
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect();
        let pixels1: Vec<[u8; 4]> = rows1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect();
        let at = |x: u32, y: u32| (y * V3_EDGE + x) as usize;

        // Pass two's scissored full-screen triangle covers the left half only.
        for y in 0..V3_EDGE {
            for x in 0..V3_EDGE / 2 {
                assert_eq!(
                    pixels0[at(x, y)],
                    V3_YELLOW,
                    "target 0 ({x},{y}) should be the scissored yellow quad"
                );
            }
        }
        assert_eq!(pixels0[at(63, 0)], V3_RED, "right half keeps pass one");
        assert_eq!(
            pixels0[at(32, 32)],
            V3_GREEN,
            "the nearer centre triangle must win the depth test"
        );
        assert_eq!(pixels0[at(60, 60)], V3_RED);
        assert_eq!(
            pixels1[at(0, 0)],
            V3_CYAN,
            "second target kept its own colour"
        );
        assert_eq!(pixels1[at(63, 63)], V3_CYAN);
        assert_eq!(
            pixels1[at(32, 32)],
            V3_MAGENTA,
            "the second target's centre colour differs from the first target's"
        );
        for (pixels, label) in [(&pixels0, "target 0"), (&pixels1, "target 1")] {
            assert!(
                !pixels
                    .iter()
                    .any(|pixel| *pixel == V3_BLUE || *pixel == V3_WHITE),
                "{label} must not contain the depth-rejected far draw"
            );
        }

        let cpu_checksum = rows0.as_chunks::<4>().0.iter().fold(0u32, |sum, pixel| {
            sum.wrapping_add(u32::from_le_bytes(*pixel))
        });
        assert_eq!(
            checksum.words()[0],
            cpu_checksum,
            "compute must read the copied attachment pixels and checksum them"
        );

        // The list identity is one-shot: a replay is refused with no GPU work.
        let (replay, replay_stats) = transport
            .submit_v3_with_stats(&submission, &allocations)
            .expect("a replayed list should still return a reply");
        assert_eq!(replay.status, ReplyStatus::AlreadyExecuted);
        assert_eq!(replay_stats, EncoderStats::default());

        let device = transport.device_info().expect("device_info should succeed");
        eprintln!(
            "v3 mixed segments: device={}, surface0={}, surface1={}, pixels={V3_PIXELS}, \
             stats={stats:?}, checksum=0x{cpu_checksum:08x}",
            device.name_str(),
            image0.surface_id(),
            image1.surface_id()
        );

        assert_eq!(image0.frame_index(), None);
        assert_eq!(image1.frame_index(), None);
        let _ = (&mut image0, &mut image1);
    }

    #[test]
    fn v3_illegal_ordering_is_rejected_before_any_command_buffer() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let submission = SubmissionV3 {
            sequence: 3,
            list: None,
            commands: vec![
                CommandV3::BeginRenderPass(BeginRenderPass {
                    colors: vec![v3_color_attachment(1, LoadOp::Clear)],
                    depth_stencil: None,
                }),
                CommandV3::Dispatch1d(wire::Dispatch1d {
                    pipeline_id: 1,
                    threads: 1,
                    threads_per_group: 1,
                    bindings: Vec::new(),
                }),
            ],
        };
        let error = transport
            .submit_v3(&submission, &HashMap::new())
            .expect_err("compute inside a pass must be refused");
        assert!(
            matches!(error, MetalError::InvalidArg(message) if message.contains("open render pass")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn v3_unknown_pipeline_returns_unknown_handle_without_encoders() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let submission = SubmissionV3 {
            sequence: 4,
            list: None,
            commands: vec![
                CommandV3::BeginRenderPass(BeginRenderPass {
                    colors: vec![v3_color_attachment(1, LoadOp::Clear)],
                    depth_stencil: None,
                }),
                CommandV3::SetGraphicsPipeline(PipelineRef {
                    pipeline_id: 999_999,
                }),
                CommandV3::Draw(DrawState {
                    vertex_count: 3,
                    instance_count: 1,
                    start_vertex: 0,
                    start_instance: 0,
                }),
                CommandV3::EndRenderPass,
            ],
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &HashMap::new())
            .expect("an unknown pipeline should still return a reply");
        assert_eq!(reply.status, ReplyStatus::UnknownHandle);
        assert_eq!(
            stats,
            EncoderStats::default(),
            "a refused submission must not create a command buffer"
        );
    }

    #[test]
    fn v3_mesh_dispatch_outside_a_render_pass_is_rejected() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let submission = SubmissionV3 {
            sequence: 5,
            list: None,
            commands: vec![CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: 1,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            })],
        };
        let error = transport
            .submit_v3(&submission, &HashMap::new())
            .expect_err("a mesh dispatch without a render pass must be refused");
        assert!(
            matches!(error, MetalError::InvalidArg(message) if message.contains("open render pass")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn v3_unknown_mesh_pipeline_returns_unknown_handle_without_encoders() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let target = transport
            .create_render_target_2d(4, 4, PixelFormat::Rgba8Unorm)
            .expect("mesh target should be created");
        let target_id = transport
            .register_render_target(&target)
            .expect("mesh target should register");
        let mut allocations = HashMap::new();
        allocations.insert((1u64, 1u32), VmBufferId(target_id.0));
        let submission = SubmissionV3 {
            sequence: 6,
            list: None,
            commands: vec![
                CommandV3::BeginRenderPass(BeginRenderPass {
                    colors: vec![v3_color_attachment(1, LoadOp::Clear)],
                    depth_stencil: None,
                }),
                CommandV3::SetMeshPipeline(PipelineRef {
                    pipeline_id: 999_999,
                }),
                CommandV3::DispatchMesh(wire::DispatchMesh {
                    pipeline_id: 999_999,
                    group_counts: [1, 1, 1],
                    bindings: Vec::new(),
                }),
                CommandV3::EndRenderPass,
            ],
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &allocations)
            .expect("an unknown mesh pipeline should still return a reply");
        assert_eq!(reply.status, ReplyStatus::UnknownHandle);
        assert_eq!(
            stats,
            EncoderStats::default(),
            "a refused mesh submission must not create a command buffer"
        );
    }

    #[test]
    fn v3_present_requires_a_registered_surface() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let submission = SubmissionV3 {
            sequence: 6,
            list: None,
            commands: vec![CommandV3::Present(Present {
                surface_id: 123_456,
                sync_interval: 1,
            })],
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &HashMap::new())
            .expect("an unknown surface should return a reply");
        assert_eq!(reply.status, ReplyStatus::UnknownHandle);
        assert_eq!(stats, EncoderStats::default());
    }

    #[test]
    fn v3_validation_refusal_releases_the_list_identity() {
        let transport = DirectMetalTransport::new().expect("Metal device should open");
        let list = Some(wire::ListIdentity {
            list_id: 77,
            list_generation: 2,
        });
        let refused = SubmissionV3 {
            sequence: 1,
            list,
            commands: vec![CommandV3::Dispatch1d(wire::Dispatch1d {
                pipeline_id: 999_999,
                threads: 1,
                threads_per_group: 1,
                bindings: Vec::new(),
            })],
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&refused, &HashMap::new())
            .expect("an unknown pipeline should return a reply");
        assert_eq!(reply.status, ReplyStatus::UnknownHandle);
        assert_eq!(stats.command_buffers, 0);

        // ADR-007: a validation refusal releases the claim, so the same list
        // identity can be submitted again once corrected.
        let empty = SubmissionV3 {
            sequence: 2,
            list,
            commands: Vec::new(),
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&empty, &HashMap::new())
            .expect("the retried list identity should be accepted");
        assert_eq!(reply.status, ReplyStatus::Ok);
        assert_eq!(stats.command_buffers, 0);

        // After successful execution the identity is consumed.
        let (reply, _) = transport
            .submit_v3_with_stats(&empty, &HashMap::new())
            .expect("a replay should return a reply");
        assert_eq!(reply.status, ReplyStatus::AlreadyExecuted);
    }
}
