//! Direct in-process macOS Metal implementation of [`MetalTransport`].
//!
//! Used both by native macOS callers (Phase 2, Wine-free) and behind `c_api`
//! for Phase 1 Wine unixlib bridges. Zero Wine or Windows code lives here.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, c_void};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmPipelineId};
use crate::fence_page::{FencePage, FencePageMapping};
use crate::macos::bindings::{
    ArgumentBufferBuilder, DescriptorTableBuilder, DescriptorTableKind, ReflectedStage,
    RootParameterKind, RootParameterMap, RootParameterSlot, RootSignatureLayout,
};
use crate::macos::irconverter;
use crate::macos::mesh::{MeshPipeline, MeshPipelineSpec};
use crate::macos::msc::{self, MscUavDispatch};
use crate::macos::objc::{self, AutoreleasePool, MTLSize, ObjcObject};
use crate::macos::registry::{
    PipelineKey, PipelineRegistry, PipelineStage, RegistryError, StageLibrary, TextureDesc,
    TextureFormat, TextureRegistry,
};
use crate::macos::render::{
    GraphicsPipeline, IndexType, LoadAction, MTL_BARRIER_SCOPE_BUFFERS, MTL_BARRIER_SCOPE_TEXTURES,
    PixelFormat, PrimitiveType, RawColorAttachment, RawDepthAttachment, RenderAttachment,
    RenderEncoder, RenderPass, RenderTarget, ScissorRect as RenderScissorRect, SegmentKind,
    StoreAction, Viewport as RenderViewport, build_mrt_render_pass_descriptor,
    new_less_depth_state, plan_encoder_segments, texture_info,
};
use crate::macos::residency::ResidencySet;
use crate::macos::retire::{BackpressureBudget, RetireQueue};
use crate::macos::state::{self, DepthStencilCache};
use crate::macos::texture::VmTextureId;
use crate::macos::{blit, indirect};
use crate::root_arguments::{PlanArgumentKind, RootParameterPlan, RootPlanEntry};
use crate::transport::{MetalError, MetalTransport};
use crate::wire::{self, Command, Reply, ReplyStatus, Submission};
use crate::wire_v3::{
    BeginRenderPass, CommandV3, DescriptorKind, IndexFormat, LoadOp, PrimitiveTopology,
    RasterizerState, StoreOp, SubmissionV3,
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

/// Maximum bytes of a GPU-failure reason carried host-side or on a Wire v3
/// error reply.
///
/// The bounded text is stored in telemetry, logged, and (for a kind 5 error)
/// carried in the reply's reason trailer. It is sanitised to one line so a
/// Metal `localizedDescription` cannot flood the log or the JSON record, and it
/// never exceeds [`crate::wire::MAX_REPLY_REASON_BYTES`].
pub(crate) const MAX_GPU_REASON_BYTES: usize = 256;

/// One non-completed command buffer, named by the encoder that built it.
///
/// `sequence` is the wire sequence for a synchronous submission or the fence
/// value for an asynchronous one. `status` is the wire status the guest still
/// receives: a GPU fault never changes the protocol status, it only explains
/// the `ExecutionFailed` refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandBufferFailure {
    /// Wire sequence (synchronous) or completion fence (asynchronous).
    pub sequence: u64,
    /// The wire status the guest receives: always `ExecutionFailed`, unchanged
    /// by this diagnostic work.
    pub status: ReplyStatus,
    /// Command family that built the failed buffer.
    pub kind: &'static str,
    /// First command index of that family in the submission.
    pub index: usize,
    /// `MTLCommandBuffer.status`.
    pub mtl_status: usize,
    /// Bounded, single-line reason, including Metal's own text.
    pub reason: String,
}

/// Cleans a Metal diagnostic into one bounded log/telemetry line.
///
/// Control characters (including newlines from a Metal error) become spaces,
/// repeated whitespace collapses, and the result is truncated on a UTF-8
/// boundary to at most [`MAX_GPU_REASON_BYTES`]. Raw guest memory is never
/// passed here: callers only hand over Metal's own `localizedDescription`.
pub(crate) fn sanitize_gpu_reason(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len().min(MAX_GPU_REASON_BYTES));
    for character in text.chars() {
        if character.is_control() {
            if !cleaned.is_empty() && !cleaned.ends_with(' ') {
                cleaned.push(' ');
            }
            continue;
        }
        if character == ' ' {
            if !cleaned.is_empty() && !cleaned.ends_with(' ') {
                cleaned.push(' ');
            }
            continue;
        }
        cleaned.push(character);
    }
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return "unknown Metal error".to_string();
    }
    truncate_reason(cleaned, MAX_GPU_REASON_BYTES)
}

/// Truncates `text` to at most `max` bytes on a UTF-8 boundary, with an ASCII
/// marker when anything was dropped.
fn truncate_reason(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let marker = "...";
    let budget = max.saturating_sub(marker.len());
    let mut end = budget;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = text[..end].to_string();
    out.push_str(marker);
    out
}

/// Names a non-completed command buffer without changing its wire status.
///
/// The status stays [`ReplyStatus::ExecutionFailed`]; the bounded reason is
/// returned to the caller so a kind 5 host can carry it in the Wire v3 reply
/// reason trailer and record it in telemetry. Only Metal's own text is used.
pub(crate) fn describe_command_buffer_failure(
    sequence: u64,
    mtl_status: usize,
    kind: &'static str,
    index: usize,
    detail: &str,
) -> CommandBufferFailure {
    let prefix = format!("{kind} #{index} (MTLCommandBuffer.status={mtl_status}): ");
    let budget = MAX_GPU_REASON_BYTES.saturating_sub(prefix.len());
    let detail = truncate_reason(&sanitize_gpu_reason(detail), budget);
    CommandBufferFailure {
        sequence,
        status: ReplyStatus::ExecutionFailed,
        kind,
        index,
        mtl_status,
        reason: format!("{prefix}{detail}"),
    }
}

/// [`sanitize_gpu_reason`] for callers that only need the bounded text.
pub(crate) fn bounded_gpu_reason(text: &str) -> String {
    sanitize_gpu_reason(text)
}

/// Adds the command family/index to an error that already carries Metal's own
/// text (and status) and bounds the combined line.
pub(crate) fn annotate_gpu_error(
    sequence: u32,
    kind: &'static str,
    index: usize,
    error: &MetalError,
) -> MetalError {
    let detail = error.to_string();
    let prefix = format!("{kind} #{index}: ");
    let budget = MAX_GPU_REASON_BYTES.saturating_sub(prefix.len());
    let reason = format!(
        "{prefix}{}",
        truncate_reason(&sanitize_gpu_reason(&detail), budget)
    );
    tracing::error!(sequence, reason = %reason, "Metal command buffer failed");
    MetalError::InternalError(reason)
}

/// The command family and first command index of one v3 encoder segment.
fn segment_failure_site(segment: &crate::macos::render::Segment) -> (&'static str, usize) {
    match &segment.kind {
        SegmentKind::Compute(range) => ("dispatch", range.start),
        SegmentKind::Render { range, .. } => ("render pass", range.start),
        SegmentKind::Blit(range) => ("copy", range.start),
    }
}

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
    /// Textures created through the ADR-013 [`TextureRegistry`], keyed by a
    /// transport handle that is also the registry id. A handle here resolves
    /// through the registry, never through the legacy `textures` map.
    registry_textures: HashMap<u64, u64>,
    /// `MTLSamplerState`s named by converted sampler descriptor tables.
    samplers: HashMap<u64, ObjcObject>,
    /// Content-hashed pipelines keyed by the wire handle a `Set*Pipeline`
    /// command names.
    content_pipelines: HashMap<u64, crate::macos::registry::PipelineId>,
    /// Root-signature ABI registered for a content-hashed pipeline handle.
    abis: HashMap<u64, PipelineAbi>,
    /// Lazily created host registries and the persistent residency set.
    texture_registry: Option<TextureRegistry>,
    pipeline_registry: Option<PipelineRegistry>,
    residency: Option<ResidencySet>,
    /// v3 list identities already executed or executing.
    executed_lists: HashSet<(u64, u32)>,
    executed_list_order: VecDeque<(u64, u32)>,
    /// `MTLDepthStencilState` objects built from wire depth/stencil commands.
    depth_states: DepthStencilCache,
    /// ADR-053: opt-in draw trace, armed on the first traced encode.
    trace: Option<crate::macos::trace::TraceState>,
    /// ADR-055: the live title-memory map and its built page table.
    pub(crate) title: crate::macos::title_memory::TitleMemory,
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
    /// Wire index of the first command that opened this segment.
    first_command_index: usize,
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
    first_command_index: usize,
) {
    if !*segment_open {
        segments.push(PreparedGraphicsSegment {
            pipeline_id,
            first_command_index,
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

/// Default number of command buffers kept in flight (ADR-017).
const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;
/// Hard upper bound the environment override is clamped to.
const MAX_FRAMES_IN_FLIGHT: usize = 3;
/// Environment override for the in-flight cap, clamped to `1..=3`.
const FRAMES_IN_FLIGHT_ENV: &str = "VICTORY_METAL_FRAMES_IN_FLIGHT";
/// Bounded spin on the published completion fence before a frame-boundary
/// submission escalates to `waitUntilCompleted` on the oldest buffer only.
const BACKPRESSURE_SPIN: Duration = Duration::from_micros(250);

// `_NSConcreteStackBlock` class object exported by libclosure.
//
// Only the symbol address matters: a stack block literal stores it as `isa`
// and `addCompletedHandler:` copies the literal to the heap before use.
unsafe extern "C" {
    static _NSConcreteStackBlock: *const c_void;
}

/// Layout prefix every Objective-C block literal starts with.
///
/// The captured fields follow the prefix verbatim; `Block_copy` moves exactly
/// `descriptor.size` bytes, so a literal with no Objective-C object captures
/// needs no `copy`/`dispose` helper and no signature.
#[repr(C)]
struct CompletionBlock {
    isa: *const c_void,
    flags: i32,
    reserved: i32,
    invoke: unsafe extern "C" fn(*mut CompletionBlock, objc::Id),
    descriptor: *const CompletionBlockDescriptor,
    /// Transport-owned async state; valid until the transport is dropped.
    state: *const Mutex<AsyncSubmitState>,
    /// Fence value this command buffer was committed with.
    fence: u64,
}

/// Trailing descriptor of a block literal: two reserved words then the size.
#[repr(C)]
struct CompletionBlockDescriptor {
    reserved: usize,
    size: usize,
}

/// Static descriptor shared by every completion block literal.
///
/// A static is required: `Block_copy` keeps the descriptor pointer, so it must
/// not point into a returning stack frame.
static COMPLETION_BLOCK_DESCRIPTOR: CompletionBlockDescriptor = CompletionBlockDescriptor {
    reserved: 0,
    size: std::mem::size_of::<CompletionBlock>(),
};

/// `MTLCommandBuffer.addCompletedHandler:` body (ADR-017).
///
/// Runs on Metal's completion thread after the command buffer finishes. It
/// publishes the submission's fence value into the shared [`FencePage`] and
/// releases retired staging work, so no submitter has to wait.
///
/// # Safety
///
/// `block` must be a live [`CompletionBlock`] copied by Metal whose `state`
/// pointer still names the transport-owned [`AsyncSubmitState`]; `command_buffer`
/// must be the committed `MTLCommandBuffer` the handler was attached to.
unsafe extern "C" fn completion_block_invoke(block: *mut CompletionBlock, command_buffer: objc::Id) {
    // SAFETY: the caller is libdispatch invoking a block literal built by
    // `DirectMetalTransport::commit_async`; `state` outlives every committed
    // buffer because `Drop` waits for the pipeline to drain.
    let (state, fence) = unsafe { (&*(*block).state, (*block).fence) };
    // SAFETY: `command_buffer` is a live `MTLCommandBuffer` at completion time.
    let status = unsafe { objc::msg_send_usize_0(command_buffer, objc::sel(c"status")) };
    let detail = if status == MTL_COMMAND_BUFFER_STATUS_COMPLETED {
        None
    } else {
        // SAFETY: `[command_buffer error]` returns null or an autoreleased
        // `NSError *`; `nserror_message` accepts both.
        let error_raw = unsafe { objc::msg_send_id_0(command_buffer, objc::sel(c"error")) };
        Some(unsafe { objc::nserror_message(error_raw) })
    };
    publish_completion(state, fence, status, detail);
}

/// Records one observed completion and publishes the contiguous watermark.
///
/// A poisoned lock means a panic aborted a previous holder; the publication is
/// dropped rather than resumed with possibly inconsistent state. The
/// diagnostics-only `stall_publication` flag suppresses the record so a lost
/// completion can be reproduced; it is never set by the host.
fn publish_completion(
    state: &Mutex<AsyncSubmitState>,
    fence: u64,
    status: usize,
    detail: Option<String>,
) {
    let failure = {
        let Ok(mut state) = state.lock() else {
            return;
        };
        if state.stall_publication {
            return;
        }
        let failure = if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            let (kind, index) = state
                .contexts
                .get(&fence)
                .copied()
                .unwrap_or(("command buffer", 0));
            let detail = detail.unwrap_or_else(|| "unknown Metal error".to_string());
            let failure = describe_command_buffer_failure(fence, status, kind, index, &detail);
            state.last_failure = Some(failure.reason.clone());
            Some(failure)
        } else {
            None
        };
        state.note_completion(fence, status == MTL_COMMAND_BUFFER_STATUS_COMPLETED);
        failure
    };
    // Log outside the state lock: a tracing subscriber must never be able to
    // re-enter the completion bookkeeping.
    if let Some(failure) = failure {
        tracing::error!(
            fence,
            wire_status = ?failure.status,
            mtl_status = status,
            reason = %failure.reason,
            "asynchronous Metal command buffer did not complete"
        );
    }
}

/// Contiguous completion watermark over committed fence values.
///
/// Fence values are committed in strictly increasing order, but completion
/// handlers may be delivered out of order. Only a contiguous prefix may be
/// published: publishing across a gap would release staging memory a still
/// running command buffer reads.
#[derive(Debug, Default)]
struct CompletionWatermark {
    /// `(fence, done)` in commit order.
    pending: VecDeque<(u64, bool)>,
    completed: u64,
}

impl CompletionWatermark {
    const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            completed: 0,
        }
    }

    fn len(&self) -> usize {
        self.pending.len()
    }

    /// True when no committed buffer is outstanding.
    fn is_idle(&self) -> bool {
        self.pending.is_empty()
    }

    /// Oldest fence still in flight.
    fn front(&self) -> Option<u64> {
        self.pending.front().map(|entry| entry.0)
    }

    /// Newest committed fence.
    fn back(&self) -> Option<u64> {
        self.pending.back().map(|entry| entry.0)
    }

    fn completed(&self) -> u64 {
        self.completed
    }

    /// True when `fence` is committed and not yet marked done.
    fn is_pending(&self, fence: u64) -> bool {
        self.pending
            .iter()
            .any(|entry| entry.0 == fence && !entry.1)
    }

    fn push(&mut self, fence: u64) {
        self.pending.push_back((fence, false));
    }

    /// Marks `fence` complete.
    ///
    /// Returns `false` when the fence is not pending: a completion for a
    /// command buffer this transport never committed (a dropped, duplicated or
    /// forged handler) must publish and release nothing.
    fn mark_done(&mut self, fence: u64) -> bool {
        let Some(entry) = self
            .pending
            .iter_mut()
            .find(|entry| entry.0 == fence && !entry.1)
        else {
            return false;
        };
        entry.1 = true;
        true
    }

    /// Pops the contiguous completed prefix and returns the new watermark.
    ///
    /// `None` means no completed entry is contiguous yet, so nothing may be
    /// published or released.
    fn advance(&mut self) -> Option<u64> {
        let mut advanced = None;
        while let Some((fence, done)) = self.pending.front().copied() {
            if !done {
                break;
            }
            self.pending.pop_front();
            self.completed = fence;
            advanced = Some(fence);
        }
        advanced
    }
}

/// Counters behind [`DirectMetalTransport::async_submit_stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AsyncCounters {
    submitted: u64,
    completed: u64,
    failed: u64,
    rejected: u64,
    ignored_completions: u64,
    in_flight_peak: u64,
    backpressure_waits: u64,
    retire_wait_micros_max: u64,
}

/// Host bookkeeping of the asynchronous submit pipeline (ADR-017).
///
/// The state is shared between the submitting thread(s) and Metal's completion
/// callbacks, so every field is behind the transport's own mutex. It owns no
/// Metal type except the tracked command buffers, which is what keeps a
/// completion handler from observing a deallocated buffer.
struct AsyncSubmitState {
    fence_page: Arc<FencePage>,
    /// Host-mapped page that takes precedence over `fence_page`.
    ///
    /// A native host shares one `MAP_SHARED` page with its guest. The transport
    /// retains the mapping and publishes the contiguous completion watermark
    /// into it from the completion handler, so a guest load observes a sequence
    /// only after the GPU finished that work. The `Arc` keeps the mapping
    /// mapped for as long as any completion handler can still run.
    shared_fence: Option<Arc<FencePageMapping>>,
    cap: usize,
    next_fence: u64,
    watermark: CompletionWatermark,
    /// Retained committed command buffers, keyed by fence, until they complete.
    held: HashMap<u64, ObjcObject>,
    /// Command family and first command index of every committed buffer.
    ///
    /// Metal reports a fault for the whole command buffer, so this is the only
    /// attribution an asynchronous completion handler can name.
    contexts: HashMap<u64, (&'static str, usize)>,
    /// Last nonzero-status command-buffer failure, sanitised and bounded.
    last_failure: Option<String>,
    retire: RetireQueue,
    /// Staged Metal objects pinned by token until their fence completes.
    staged: HashMap<u64, Vec<ObjcObject>>,
    /// Caller pins: token -> caller id, reported by `drain_retired`.
    pins: HashMap<u64, u64>,
    /// Caller ids released since the last `drain_retired`.
    released: Vec<u64>,
    next_token: u64,
    /// Diagnostics-only fault injection: a stalled completion publication is
    /// never recorded, so the bounded retire backpressure path can be proven
    /// without a hung GPU. Always `false` for the host.
    stall_publication: bool,
    counters: AsyncCounters,
}

impl AsyncSubmitState {
    fn new(fence_page: Arc<FencePage>, cap: usize, retire_capacity: usize) -> Self {
        Self {
            fence_page,
            shared_fence: None,
            cap,
            next_fence: 0,
            watermark: CompletionWatermark::new(),
            held: HashMap::new(),
            contexts: HashMap::new(),
            last_failure: None,
            retire: RetireQueue::with_capacity(retire_capacity),
            staged: HashMap::new(),
            pins: HashMap::new(),
            released: Vec::new(),
            next_token: 1,
            stall_publication: false,
            counters: AsyncCounters::default(),
        }
    }

    /// Reserves the next monotonic fence value for a committed submission.
    ///
    /// The value is `max(sequence, previous + 1)`, so the published completion
    /// fence is monotonic even for a caller that replays an old sequence.
    fn reserve_fence(&mut self, sequence: u32) -> u64 {
        let fence = u64::from(sequence).max(self.next_fence + 1);
        self.next_fence = fence;
        fence
    }

    fn alloc_token(&mut self) -> u64 {
        let token = self.next_token;
        self.next_token = self.next_token.wrapping_add(1).max(1);
        token
    }

    /// Tracks one freshly committed command buffer.
    fn note_commit(
        &mut self,
        fence: u64,
        command_buffer: ObjcObject,
        site: (&'static str, usize),
    ) {
        self.watermark.push(fence);
        self.held.insert(fence, command_buffer);
        self.contexts.insert(fence, site);
        self.counters.submitted += 1;
        self.counters.in_flight_peak = self
            .counters
            .in_flight_peak
            .max(self.watermark.len() as u64);
    }

    /// Stages prepared objects and returns the retire token that owns them.
    ///
    /// The token is inserted before the queue push so a completion that covers
    /// the fence can never release an id the release bookkeeping does not know
    /// yet. `None` means nothing was staged and no retire entry is needed.
    ///
    /// This only assigns ownership; the caller pushes the token into the retire
    /// queue with [`DirectMetalTransport::retire_pin`], which may wait on the
    /// GPU and therefore must run without this mutex.
    fn stage_staged(&mut self, staged: Vec<ObjcObject>) -> Option<u64> {
        if staged.is_empty() {
            return None;
        }
        let token = self.alloc_token();
        self.staged.insert(token, staged);
        Some(token)
    }

    /// Registers a caller-visible resource id and returns its retire token.
    ///
    /// Inserted before the queue push for the same reason as [`Self::stage_staged`].
    fn stage_caller_pin(&mut self, caller_id: u64) -> u64 {
        let token = self.alloc_token();
        self.pins.insert(token, caller_id);
        token
    }

    /// Publishes `value` as completed into the page a guest observes.
    ///
    /// The host-mapped page wins when one is attached; the process-local page is
    /// the headless default. Publishing is `fetch_max`, so no replay or stale
    /// handler can lower a value a guest already read.
    fn publish_completed(&self, value: u64) {
        match &self.shared_fence {
            Some(mapping) => {
                mapping.page().publish(value);
            }
            None => {
                self.fence_page.publish(value);
            }
        }
    }

    /// Reads the completed value a guest would observe right now.
    fn completed(&self) -> u64 {
        match &self.shared_fence {
            Some(mapping) => mapping.page().completed(),
            None => self.fence_page.completed(),
        }
    }

    /// Completes a submission that encoded no GPU work.
    ///
    /// The fence is reserved in commit order and marked done at once, so the
    /// contiguous watermark advances without publishing across a command buffer
    /// that is still in flight. No command buffer was committed, so the
    /// submission counters stay untouched.
    fn note_empty_submission(&mut self, sequence: u32) {
        let fence = self.reserve_fence(sequence);
        self.watermark.push(fence);
        let _ = self.watermark.mark_done(fence);
        if let Some(advanced) = self.watermark.advance() {
            self.publish_completed(advanced);
            self.held.retain(|&held, _| held > advanced);
        }
        self.drain();
    }

    /// Marks `fence` complete and advances the contiguous watermark.
    ///
    /// A completion for a fence this transport never committed is ignored: a
    /// dropped, duplicated or forged handler must not publish a value, because
    /// that would release staging work the GPU may still read.
    ///
    /// A buffer that finished with a GPU error still advances the watermark — it
    /// is no longer reading anything — and is counted in `failed` instead; the
    /// reply cannot report it because it was returned before `commit`.
    fn note_completion(&mut self, fence: u64, succeeded: bool) {
        if !self.watermark.mark_done(fence) {
            self.counters.ignored_completions += 1;
            return;
        }
        if succeeded {
            self.counters.completed += 1;
        } else {
            self.counters.failed += 1;
        }
        if let Some(advanced) = self.watermark.advance() {
            self.publish_completed(advanced);
            self.held.retain(|&fence, _| fence > advanced);
            self.contexts.retain(|&fence, _| fence > advanced);
        }
        self.drain();
    }

    /// Releases every retired token at or below the observed watermark.
    fn drain(&mut self) -> usize {
        let completed = self.watermark.completed();
        let released = self.retire.drain_upto(completed);
        let count = released.len();
        for token in released {
            drop(self.staged.remove(&token));
            if let Some(caller_id) = self.pins.remove(&token) {
                self.released.push(caller_id);
            }
        }
        count
    }

    fn stats(&self) -> AsyncSubmitStats {
        AsyncSubmitStats {
            submitted: self.counters.submitted,
            completed: self.counters.completed,
            failed: self.counters.failed,
            rejected: self.counters.rejected,
            ignored_completions: self.counters.ignored_completions,
            in_flight: self.watermark.len() as u64,
            in_flight_peak: self.counters.in_flight_peak,
            frames_in_flight_cap: self.cap as u64,
            retire_depth: self.retire.len(),
            retire_high_watermark: self.retire.high_watermark(),
            retire_capacity: self.retire.capacity(),
            retire_refusals: self.retire.refusals(),
            retire_max_wait_micros: self.counters.retire_wait_micros_max,
            backpressure_waits: self.counters.backpressure_waits,
            completed_fence: self.watermark.completed(),
        }
    }
}

/// Counters for the asynchronous submit pipeline (ADR-017).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsyncSubmitStats {
    /// Command buffers committed by this transport.
    pub submitted: u64,
    /// Committed buffers whose completion handler reported success.
    pub completed: u64,
    /// Committed buffers whose completion handler reported a GPU failure.
    pub failed: u64,
    /// Pre-commit refusals (unknown handle, replay, encoder error).
    pub rejected: u64,
    /// Completions ignored because their fence was not in flight.
    pub ignored_completions: u64,
    /// Command buffers in flight at the moment of the call.
    pub in_flight: u64,
    /// Highest observed in-flight depth.
    pub in_flight_peak: u64,
    /// Configured cap (`VICTORY_METAL_FRAMES_IN_FLIGHT`, 1..=3).
    pub frames_in_flight_cap: u64,
    /// Resources still pinned in the retire queue.
    pub retire_depth: usize,
    /// Highest retire depth ever observed. Never above [`Self::retire_capacity`]
    /// once the transport applies backpressure; a value above it means the hard
    /// bound was bypassed.
    pub retire_high_watermark: usize,
    /// Hard bound of the retire queue.
    pub retire_capacity: usize,
    /// Pins accepted over capacity through the legacy soft path. Always zero for
    /// the transport, which pins through the bounded path.
    pub retire_refusals: u64,
    /// Longest single frame-boundary wait on the oldest command buffer, in
    /// microseconds. This is the only place a submit can block (ADR-017).
    pub retire_max_wait_micros: u64,
    /// Number of waits on the oldest command buffer: the in-flight-cap gate and
    /// the retire-queue backpressure path.
    pub backpressure_waits: u64,
    /// Contiguous completed fence value published in the shared page.
    pub completed_fence: u64,
}

/// A command buffer encoded for one submission but not yet committed.
struct EncodedSubmission {
    command_buffer: ObjcObject,
    /// Staging Metal objects that must stay alive until the fence completes.
    staged: Vec<ObjcObject>,
}

/// Retains one more reference to a committed command buffer.
///
/// The extra reference lets a caller wait on the buffer outside the state lock
/// without racing a completion handler that pops and releases the slot.
fn retain_command_buffer(raw: objc::Id) -> Option<ObjcObject> {
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` is a live retained `MTLCommandBuffer` owned by the async
    // state; `retain` returns it with one extra reference.
    ObjcObject::from_retained(unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) })
}

/// Retains one more reference to a live Metal object.
///
/// Used where an object is owned by a short-lived planner structure but must
/// stay alive until a command buffer that references it completes.
fn retain_object(raw: objc::Id) -> Option<ObjcObject> {
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` is a live retained Metal object; `retain` returns it with
    // one extra reference owned by the returned `ObjcObject`.
    ObjcObject::from_retained(unsafe { objc::msg_send_id_0(raw, objc::sel(c"retain")) })
}

/// Clamps an environment value to the supported in-flight window `1..=3`.
///
/// A missing or unparsable value falls back to [`DEFAULT_FRAMES_IN_FLIGHT`], so
/// a typo can never remove backpressure entirely.
fn clamp_frames_in_flight(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, MAX_FRAMES_IN_FLIGHT))
        .unwrap_or(DEFAULT_FRAMES_IN_FLIGHT)
}

/// Reads the in-flight cap from the environment, clamped to `1..=3`.
fn resolve_frames_in_flight() -> usize {
    let raw = std::env::var(FRAMES_IN_FLIGHT_ENV).ok();
    clamp_frames_in_flight(raw.as_deref())
}

/// Native Apple Metal backend implementing [`MetalTransport`].
pub struct DirectMetalTransport {
    device: ObjcObject,
    queue: ObjcObject,
    info: VmDeviceInfo,
    state: Mutex<DeviceState>,
    async_submit: Arc<Mutex<AsyncSubmitState>>,
    /// ADR-053: exact encoded-work counters for the whole process.
    encode_counters: crate::macos::trace::EncodeCounters,
    /// Root-signature arguments refused because the translated layout does not
    /// declare them or declares a different `(space, register, kind)`.
    abi_refusals: std::sync::atomic::AtomicU64,
    /// Root arguments bound through the legacy wire-slot lookup, because the
    /// submission carried no root-argument plan entry for its slot.
    ///
    /// Counted separately so a boot can tell "every binding resolved by its
    /// translated identity" from "the guest sent no plan and the old lookup
    /// still happened to agree".
    legacy_root_bindings: std::sync::atomic::AtomicU64,
    /// MEMORY_SPACE parameters bound to a default zero-filled buffer.
    memory_space_bound: std::sync::atomic::AtomicU64,
    /// Bytes allocated for those default MEMORY_SPACE buffers.
    memory_space_bytes: std::sync::atomic::AtomicU64,
}

impl DirectMetalTransport {
    /// Opens the system default Metal GPU device and initializes a command queue.
    pub fn new() -> Result<Self, MetalError> {
        Self::open(RetireQueue::DEFAULT_CAPACITY)
    }

    /// Opens the same device with an explicit hard retire-queue bound.
    ///
    /// Diagnostics and hardware tests use this to drive a pipeline whose pins
    /// outnumber the bound; production callers use [`Self::new`]. The capacity
    /// is clamped to at least one by [`RetireQueue::with_capacity`].
    pub fn with_retire_capacity(retire_capacity: usize) -> Result<Self, MetalError> {
        Self::open(retire_capacity)
    }

    fn open(retire_capacity: usize) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        let device = objc::create_system_default_device().ok_or(MetalError::DeviceUnavailable)?;

        // SAFETY: `device` is a valid `MTLDevice` and `newCommandQueue` returns a retained queue.
        let queue_raw =
            unsafe { objc::msg_send_id_0(device.as_raw(), objc::sel(c"newCommandQueue")) };
        let queue = ObjcObject::from_retained(queue_raw).ok_or_else(|| {
            MetalError::InternalError("MTLDevice.newCommandQueue returned null".to_string())
        })?;

        let info = query_device_info(&device);

        let async_submit = Arc::new(Mutex::new(AsyncSubmitState::new(
            Arc::new(FencePage::new()),
            resolve_frames_in_flight(),
            retire_capacity,
        )));

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
                registry_textures: HashMap::new(),
                samplers: HashMap::new(),
                content_pipelines: HashMap::new(),
                abis: HashMap::new(),
                texture_registry: None,
                pipeline_registry: None,
                residency: None,
                executed_lists: HashSet::new(),
                executed_list_order: VecDeque::new(),
                depth_states: DepthStencilCache::new(),
                trace: None,
                title: crate::macos::title_memory::TitleMemory::new(),
            }),
            async_submit,
            encode_counters: crate::macos::trace::EncodeCounters::default(),
            abi_refusals: std::sync::atomic::AtomicU64::new(0),
            legacy_root_bindings: std::sync::atomic::AtomicU64::new(0),
            memory_space_bound: std::sync::atomic::AtomicU64::new(0),
            memory_space_bytes: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// ADR-053: a snapshot of the exact encoded-work counters.
    pub fn encode_counters(&self) -> crate::macos::trace::CounterSnapshot {
        self.encode_counters.snapshot()
    }

    /// `(parameters, bytes)` bound by the default MEMORY_SPACE fallback.
    pub fn memory_space_counters(&self) -> (u64, u64) {
        (
            self.memory_space_bound
                .load(std::sync::atomic::Ordering::Relaxed),
            self.memory_space_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Root-signature arguments refused by the ABI resolvers.
    pub fn abi_refusal_count(&self) -> u64 {
        self.abi_refusals
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn count_abi_refusal(&self) {
        self.abi_refusals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Root arguments bound through the legacy wire-slot lookup, i.e. without a
    /// plan entry for their slot.
    pub fn legacy_root_binding_count(&self) -> u64 {
        self.legacy_root_bindings
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn count_legacy_root_binding(&self) {
        self.legacy_root_bindings
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn count_memory_space(&self, parameters: u64, bytes: u64) {
        self.memory_space_bound
            .fetch_add(parameters, std::sync::atomic::Ordering::Relaxed);
        self.memory_space_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
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
    /// A submission plans, validates and encodes while holding this guard, so a
    /// concurrent registration cannot change the tables mid-frame. Since ADR-017
    /// the guard is **not** held until the GPU finishes: a submission returns
    /// after `commit` and only its staging objects are pinned in the retire
    /// queue. A caller that unregisters a buffer which in-flight work can still
    /// reference must therefore drain the pipeline first
    /// ([`DirectMetalTransport::wait_for_gpu`]).
    pub(crate) fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, DeviceState>, MetalError> {
        self.state
            .lock()
            .map_err(|_| MetalError::InternalError("Metal state mutex poisoned".to_string()))
    }

    fn lock_async(&self) -> Result<std::sync::MutexGuard<'_, AsyncSubmitState>, MetalError> {
        self.async_submit
            .lock()
            .map_err(|_| MetalError::InternalError("submit pipeline mutex poisoned".to_string()))
    }

    /// Borrows the process-local completion page this transport publishes into.
    ///
    /// The default page is a process-local allocation; a native host that
    /// shares a mapped page either installs it as the process page with
    /// [`Self::set_fence_page`] or, preferably, attaches it with
    /// [`Self::attach_shared_fence_page`], which keeps the guest's own mapping
    /// as the publication target.
    pub fn fence_page(&self) -> Arc<FencePage> {
        match self.async_submit.lock() {
            Ok(state) => Arc::clone(&state.fence_page),
            Err(_) => Arc::new(FencePage::new()),
        }
    }

    /// Replaces the process-local completion page later submissions publish into.
    ///
    /// Command buffers already in flight keep publishing into the page that was
    /// current when they were committed, so install a page before submitting.
    pub fn set_fence_page(&self, page: Arc<FencePage>) -> Result<(), MetalError> {
        let mut state = self.lock_async()?;
        state.fence_page = page;
        Ok(())
    }

    /// Attaches the host-mapped completion page a guest observes.
    ///
    /// From this call on, the contiguous completion watermark is published into
    /// `mapping.page()` — the same `MAP_SHARED` file the guest maps — from the
    /// GPU completion handler only. The transport retains the mapping, so it
    /// cannot be unmapped while a completion handler may still run; detach it
    /// with [`Self::detach_shared_fence_page`] (or drop the transport, which
    /// drains the pipeline first).
    ///
    /// Install the page before the first submission: Metal publishes into the
    /// page that was current when a buffer was committed.
    pub fn attach_shared_fence_page(
        &self,
        mapping: Arc<FencePageMapping>,
    ) -> Result<(), MetalError> {
        let mut state = self.lock_async()?;
        state.shared_fence = Some(mapping);
        Ok(())
    }

    /// Detaches the host-mapped completion page.
    ///
    /// The pipeline is drained first, so no completion handler can publish into
    /// the mapping after this returns and the caller may unmap it.
    pub fn detach_shared_fence_page(&self) -> Result<(), MetalError> {
        self.wait_for_gpu()?;
        let mut state = self.lock_async()?;
        state.shared_fence = None;
        Ok(())
    }

    /// Contiguous completion value the attached page publishes right now.
    ///
    /// This is the host-side view of the same atomic a guest loads: it never
    /// covers a command buffer that is still in flight.
    pub fn completed_fence(&self) -> u64 {
        match self.async_submit.lock() {
            Ok(state) => state.completed(),
            Err(_) => 0,
        }
    }

    /// Returns the asynchronous submit counters (ADR-017).
    pub fn async_submit_stats(&self) -> AsyncSubmitStats {
        match self.async_submit.lock() {
            Ok(state) => state.stats(),
            Err(_) => AsyncSubmitStats::default(),
        }
    }

    /// The last asynchronous command-buffer failure, sanitised and bounded.
    ///
    /// The reply for an asynchronous submission is returned before the GPU
    /// finishes, so a runtime fault cannot change it. The completion handler
    /// records the same text here for `telemetry.json` (`last_error`) and logs
    /// it as a tracing error; `None` means no failure has been observed.
    pub fn last_gpu_failure(&self) -> Option<String> {
        self.async_submit
            .lock()
            .ok()
            .and_then(|state| state.last_failure.clone())
    }

    /// Drains every retire entry whose fence value has completed and returns
    /// the caller ids released since the previous call.
    ///
    /// This is the transport-level drain: it runs at the start of every submit
    /// and at a `Present` frame boundary, and is safe to call from any thread.
    pub fn drain_retired(&self) -> Vec<u64> {
        match self.async_submit.lock() {
            Ok(mut state) => {
                state.drain();
                std::mem::take(&mut state.released)
            }
            Err(_) => Vec::new(),
        }
    }

    /// Pins `caller_id` behind the newest committed submission.
    ///
    /// The id is reported by [`Self::drain_retired`] only after the GPU fence of
    /// the newest committed command buffer is observed complete, which is what
    /// lets a caller recycle staging memory without a use-after-free. Returns
    /// the fence value it is pinned behind, or `None` when nothing is in flight
    /// (nothing can still read the resource).
    ///
    /// A full retire queue applies backpressure here: the call may wait on the
    /// oldest committed command buffer, and fails with
    /// [`MetalError::Backpressure`] instead of growing the queue when that fence
    /// never completes.
    pub fn pin_until_complete(&self, caller_id: u64) -> Result<Option<u64>, MetalError> {
        // The pin bookkeeping and the token are registered before the queue
        // push: a completion that covers the fence must never release an id the
        // release path cannot map back to the caller.
        let (fence, token) = {
            let mut state = self.lock_async()?;
            let Some(fence) = state.watermark.back() else {
                return Ok(None);
            };
            let token = state.stage_caller_pin(caller_id);
            (fence, token)
        };
        if let Err(error) = self.retire_pin(fence, token) {
            // The pin was never queued; drop the bookkeeping so a later drain
            // cannot report an id this call refused.
            if let Ok(mut state) = self.async_submit.lock() {
                state.pins.remove(&token);
            }
            return Err(error);
        }
        Ok(Some(fence))
    }

    /// Pins retire `token` behind `fence`, applying hard backpressure.
    ///
    /// The queue capacity is a hard bound: this never pushes over it and never
    /// drops a live pin. When the queue is full, [`Self::relieve_retire`] waits
    /// on the oldest committed command buffer and records its completion; the
    /// retry then has room. At most
    /// [`RetireQueue::MAX_BACKPRESSURE_WAITS`] waits are performed, after which
    /// the pin fails with [`MetalError::Backpressure`].
    ///
    /// # Lock order
    ///
    /// The submit mutex is taken only to attempt the push and inside
    /// [`Self::relieve_retire`] for its snapshot and its recording. No lock is
    /// held across `waitUntilCompleted`: the command buffer's completion handler
    /// takes the same submit mutex, so waiting under it would deadlock.
    fn retire_pin(&self, fence: u64, token: u64) -> Result<(), MetalError> {
        let mut budget = BackpressureBudget::new();
        loop {
            let full = {
                let mut state = self.lock_async()?;
                // A cheap drain first: a completion handler may have published
                // while this pin was between attempts, and the queue only learns
                // about it through a drain.
                state.drain();
                match state.retire.try_push(fence, token) {
                    Ok(()) => return Ok(()),
                    Err(full) => full,
                }
            };
            let oldest = budget.next(full).map_err(|error| {
                MetalError::Backpressure {
                    capacity: error.capacity,
                    oldest: error.oldest,
                    completed: error.completed,
                    waits: error.waits,
                }
            })?;
            // The next iteration retries the push with whatever the wait made
            // visible; if it made none, the budget is one step closer to the
            // typed error above.
            let _ = self.relieve_retire(oldest)?;
        }
    }

    /// Waits on the oldest committed command buffer for `fence` and records the
    /// completion it observes.
    ///
    /// Returns `true` only when the completion watermark advanced or a retire
    /// entry was released, i.e. the wait made progress the pin can use.
    ///
    /// # Lock order
    ///
    /// The submit mutex is released for the `waitUntilCompleted` call and for the
    /// duration of every other GPU wait. `waitUntilCompleted` delivers the
    /// buffer's completion handler before returning, and that handler takes this
    /// same mutex; holding it here would deadlock the pipeline.
    fn relieve_retire(&self, fence: u64) -> Result<bool, MetalError> {
        let (retained, before_completed, before_len) = {
            let state = self.lock_async()?;
            (
                state
                    .held
                    .get(&fence)
                    .and_then(|buffer| retain_command_buffer(buffer.as_raw())),
                state.completed(),
                state.retire.len(),
            )
        };
        let Some(retained) = retained else {
            // The buffer is no longer tracked: either the watermark already
            // advanced past it (then the next push attempt sees the released
            // slot) or the transport never held it. There is nothing to wait on,
            // so report "no progress" and let the budget decide.
            return Ok(false);
        };
        let started = Instant::now();
        // SAFETY: `retained` is a live retained committed `MTLCommandBuffer`, so
        // `waitUntilCompleted` blocks until it is done.
        unsafe {
            objc::msg_send_void_0(retained.as_raw(), objc::sel(c"waitUntilCompleted"))
        };
        let micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        // SAFETY: the buffer completed above, so `status` is final.
        let status = unsafe {
            objc::msg_send_usize_0(retained.as_raw(), objc::sel(c"status"))
        };
        let mut state = self.lock_async()?;
        state.counters.backpressure_waits += 1;
        state.counters.retire_wait_micros_max = state.counters.retire_wait_micros_max.max(micros);
        if state.stall_publication {
            // Diagnostics-only: the observed completion is deliberately lost, so
            // the bounded budget is the only way out of the wait.
            return Ok(false);
        }
        // The completion handler normally recorded this observation; the
        // fallback covers a handler that was never delivered and is exactly what
        // `wait_on_buffer` does for the in-flight-cap gate.
        if state.watermark.is_pending(fence) {
            state.note_completion(fence, status == MTL_COMMAND_BUFFER_STATUS_COMPLETED);
        }
        state.drain();
        Ok(state.completed() > before_completed || state.retire.len() < before_len)
    }

    /// Diagnostics-only: suppresses every completion publication.
    ///
    /// The host never calls this. `retire_backpressure_gpu` uses it to prove
    /// that a lost completion publication surfaces as
    /// [`MetalError::Backpressure`] inside the bounded wait rather than a hang.
    /// Clear it before dropping the transport so teardown can drain.
    #[doc(hidden)]
    pub fn set_stall_completion_publication(&self, stalled: bool) -> Result<(), MetalError> {
        let mut state = self.lock_async()?;
        state.stall_publication = stalled;
        Ok(())
    }

    /// Waits until every committed command buffer has completed.
    ///
    /// This is a frame-boundary/teardown barrier, never part of a normal submit:
    /// it waits on the oldest in-flight buffer first, so the GPU pipeline drains
    /// in commit order without a per-command-list wait.
    pub fn wait_for_gpu(&self) -> Result<(), MetalError> {
        loop {
            let next = {
                let mut state = self.lock_async()?;
                if state.watermark.is_idle() {
                    state.drain();
                    return Ok(());
                }
                let fence = state.watermark.front().expect("non-empty watermark");
                let retained = state
                    .held
                    .get(&fence)
                    .and_then(|buffer| retain_command_buffer(buffer.as_raw()));
                (fence, retained)
            };
            let (fence, retained) = next;
            let Some(retained) = retained else {
                return Err(MetalError::InternalError(
                    "in-flight command buffer is null".to_string(),
                ));
            };
            self.wait_on_buffer(fence, &retained);
        }
    }

    /// Waits on one committed buffer and records its completion.
    ///
    /// `waitUntilCompleted` calls the completion handler before it returns; the
    /// state machine also records the observation itself so a handler that is
    /// never delivered cannot stall the pipeline.
    fn wait_on_buffer(&self, fence: u64, command_buffer: &ObjcObject) {
        let started = Instant::now();
        // SAFETY: `command_buffer` is a live retained committed
        // `MTLCommandBuffer`; `waitUntilCompleted` blocks until it is done.
        unsafe {
            objc::msg_send_void_0(command_buffer.as_raw(), objc::sel(c"waitUntilCompleted"))
        };
        let micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        // SAFETY: `command_buffer` completed above, so `status` is final.
        let status = unsafe {
            objc::msg_send_usize_0(command_buffer.as_raw(), objc::sel(c"status"))
        };
        if let Ok(mut state) = self.async_submit.lock() {
            state.counters.backpressure_waits += 1;
            state.counters.retire_wait_micros_max =
                state.counters.retire_wait_micros_max.max(micros);
            // The completion handler normally recorded this observation
            // already; only an undelivered handler needs the fallback, and the
            // check and the mark share the state lock so it is race-free.
            if state.watermark.is_pending(fence) {
                state.note_completion(fence, status == MTL_COMMAND_BUFFER_STATUS_COMPLETED);
            }
        }
    }

    /// Blocks until the in-flight cap admits one more command buffer.
    ///
    /// A short bounded spin on the published completion value covers the common
    /// case where a frame finishes while the next one is being prepared; only
    /// then does the oldest — never the current — command buffer get an explicit
    /// `waitUntilCompleted`.
    fn await_slot(&self) -> Result<(), MetalError> {
        loop {
            let pending = {
                let state = self.lock_async()?;
                if state.watermark.len() < state.cap {
                    return Ok(());
                }
                let fence = state.watermark.front().expect("cap is at least one");
                (
                    fence,
                    state
                        .held
                        .get(&fence)
                        .and_then(|buffer| retain_command_buffer(buffer.as_raw())),
                    Arc::clone(&state.fence_page),
                    state.shared_fence.clone(),
                )
            };
            let (fence, retained, owned_page, shared_page) = pending;
            let deadline = Instant::now() + BACKPRESSURE_SPIN;
            while Instant::now() < deadline {
                // The shared mapping is retained by the same state snapshot, so
                // reading it here cannot race an unmap.
                let completed = match &shared_page {
                    Some(mapping) => mapping.page().completed(),
                    None => owned_page.completed(),
                };
                if completed >= fence {
                    break;
                }
                std::hint::spin_loop();
            }
            // The page may have been advanced by another publisher; only the
            // state machine decides whether this slot is really free.
            let still_capped = {
                let state = self.lock_async()?;
                state.watermark.len() >= state.cap
                    && state.watermark.front() == Some(fence)
            };
            if !still_capped {
                continue;
            }
            let Some(retained) = retained else {
                return Err(MetalError::InternalError(
                    "in-flight command buffer is null".to_string(),
                ));
            };
            self.wait_on_buffer(fence, &retained);
        }
    }

    /// Commits an encoded submission and registers its completion handler.
    ///
    /// The handler is attached before `commit`, so Metal cannot finish the
    /// buffer before the publication path exists. The reply status is decided by
    /// pre-GPU validation only; GPU status is reported through the shared page
    /// and [`Self::async_submit_stats`].
    ///
    /// The staging pin is applied **before** the commit is registered in the
    /// watermark, because a full retire queue makes that pin wait: registering an
    /// uncommitted buffer first would expose it to a concurrent `wait_on_buffer`.
    /// The fence is reserved first, so the pin is behind a value no published
    /// completion can cover yet.
    fn commit_async(
        &self,
        encoded: EncodedSubmission,
        sequence: u32,
        site: (&'static str, usize),
        trace_wait: bool,
    ) -> Result<(), MetalError> {
        self.await_slot()?;
        let state_ptr = Arc::as_ptr(&self.async_submit);
        let EncodedSubmission {
            command_buffer,
            staged,
        } = encoded;
        let command_buffer_raw = command_buffer.as_raw();
        let (fence, staged_token) = {
            let mut state = self.lock_async()?;
            let fence = state.reserve_fence(sequence);
            let token = state.stage_staged(staged);
            (fence, token)
        };
        if let Some(token) = staged_token
            && let Err(error) = self.retire_pin(fence, token)
        {
            // The pin was never queued; drop the staged objects so they are not
            // retained until a release that will never come.
            if let Ok(mut state) = self.async_submit.lock() {
                state.staged.remove(&token);
            }
            return Err(error);
        }
        {
            let mut state = self.lock_async()?;
            state.note_commit(fence, command_buffer, site);
        }
        let block = CompletionBlock {
            // SAFETY: `_NSConcreteStackBlock` is the libclosure class object; its
            // address is the `isa` every stack block literal carries.
            isa: std::ptr::addr_of!(_NSConcreteStackBlock) as *const c_void,
            // No copy/dispose helper and no signature: the literal captures only
            // plain pointers, so `Block_copy` can memmove it.
            flags: 0,
            reserved: 0,
            invoke: completion_block_invoke,
            descriptor: &COMPLETION_BLOCK_DESCRIPTOR,
            state: state_ptr,
            fence,
        };
        // SAFETY: `command_buffer_raw` is a live retained committed-to-be
        // `MTLCommandBuffer`; the stack literal is valid for this call because
        // `addCompletedHandler:` copies it; `commit` starts execution.
        unsafe {
            objc::msg_send_void_1(
                command_buffer_raw,
                objc::sel(c"addCompletedHandler:"),
                std::ptr::addr_of!(block).cast_mut().cast(),
            );
            objc::msg_send_void_0(command_buffer_raw, objc::sel(c"commit"));
        }
        if trace_wait {
            // ADR-053, opt-in only: the first rendered pass of a traced
            // submission is read back, so its command buffer must complete
            // first. At most MAX_TRACED_PASSES submissions pay this cost.
            unsafe { objc::msg_send_void_0(command_buffer_raw, objc::sel(c"waitUntilCompleted")) };
        }
        Ok(())
    }

    /// ADR-053: samples the first colour attachment of a traced pass.
    fn trace_sample_target(&self, target: &crate::macos::trace::TraceTarget) {
        let Some(texture) = retain_object(target.texture as objc::Id) else {
            eprintln!("draw_trace_rt failure=texture-gone");
            crate::macos::trace::note_failure("texture gone before sample");
            return;
        };
        let Ok(render_target) = RenderTarget::from_objc(texture) else {
            eprintln!("draw_trace_rt failure=texture-unsupported");
            crate::macos::trace::note_failure("attachment format unsupported");
            return;
        };
        match read_back_render_target(self, &render_target) {
            Ok(bytes) => {
                let stats =
                    crate::macos::trace::sample_grid(&bytes, target.width, target.height);
                eprintln!("{}", stats.line());
            }
            Err(error) => {
                eprintln!("draw_trace_rt failure=readback {error}");
                crate::macos::trace::note_failure("target readback failed");
            }
        }
    }

    /// Records one submission that returned before any commit.
    fn note_refusal(&self) {
        if let Ok(mut state) = self.async_submit.lock() {
            state.counters.rejected += 1;
        }
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
        for (command_index, command) in submission.commands.iter().enumerate() {
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
                    open_graphics_segment(&mut segments, &mut segment_open, pipeline, command_index);
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
                    open_graphics_segment(&mut segments, &mut segment_open, pipeline, command_index);
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
                    open_graphics_segment(&mut segments, &mut segment_open, draw.pipeline_id, command_index);
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
                    open_graphics_segment(&mut segments, &mut segment_open, draw.pipeline_id, command_index);
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
            encoder.finish().map_err(|error| {
                annotate_gpu_error(
                    submission.sequence,
                    "render pass",
                    segment.first_command_index,
                    &error,
                )
            })?;
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
    ///   commit returns `Err` with a bounded Metal reason; the IPC host maps
    ///   that to [`ReplyStatus::ExecutionFailed`], so the guest status is
    ///   unchanged while telemetry and tracing name the failure.
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
            // SAFETY: `[cb error]` returns null or an autoreleased `NSError *`.
            let error_raw = unsafe { objc::msg_send_id_0(cb, objc::sel(c"error")) };
            // SAFETY: `error_raw` is null or a valid `NSError *`.
            let detail = unsafe { objc::nserror_message(error_raw) };
            // The batch is one command buffer; the first command is the only
            // attribution Metal's whole-buffer failure allows. The status the
            // guest receives is unchanged (`ExecutionFailed`).
            let failure = describe_command_buffer_failure(
                u64::from(submission.sequence),
                status,
                "dispatch",
                0,
                &detail,
            );
            tracing::error!(
                sequence = failure.sequence,
                wire_status = ?failure.status,
                mtl_status = failure.mtl_status,
                reason = %failure.reason,
                "Metal compute command buffer did not complete"
            );
            return Err(MetalError::InternalError(failure.reason));
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
            let failure = describe_command_buffer_failure(0, status, "dispatch", 0, &msg);
            tracing::error!(
                wire_status = ?failure.status,
                mtl_status = failure.mtl_status,
                reason = %failure.reason,
                "Metal MSC argument-buffer dispatch did not complete"
            );
            return Err(MetalError::InternalError(failure.reason));
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

/// Shader stage a converted root-signature argument buffer is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// `[[stage_in]]` producer: `setVertexBuffer:offset:atIndex:`.
    Vertex,
    /// `[[color(n)]]` producer: `setFragmentBuffer:offset:atIndex:`.
    Fragment,
    /// Metal 3 mesh stage: `setMeshBuffer:offset:atIndex:`.
    Mesh,
    /// Metal 3 object stage: `setObjectBuffer:offset:atIndex:`.
    Object,
    /// `MTLComputeCommandEncoder` kernel: `setBuffer:offset:atIndex:`.
    Compute,
}

impl ShaderStage {
    /// Whether the stage is part of a render encoder.
    pub const fn is_render(self) -> bool {
        !matches!(self, Self::Compute)
    }
}

/// Reflected root-signature ABI of one content-hashed pipeline.
///
/// Each field is the converter-reflection layout of one shader stage. A stage
/// with `None` has no top-level argument buffer, so the executor leaves it
/// untouched. The ABI is registered with the pipeline handle, so a wire
/// `SetGraphicsPipeline`/`SetMeshPipeline`/`SetComputePipeline` automatically
/// selects the matching bindings.
#[derive(Debug, Clone, Default)]
pub struct PipelineAbi {
    /// Vertex-stage layout (graphics pipelines).
    pub vertex: Option<RootSignatureLayout>,
    /// Fragment-stage layout (graphics and mesh pipelines).
    pub fragment: Option<RootSignatureLayout>,
    /// Mesh-stage layout (mesh pipelines).
    pub mesh: Option<RootSignatureLayout>,
    /// Object-stage layout (mesh pipelines).
    pub object: Option<RootSignatureLayout>,
    /// Compute-stage layout (compute pipelines).
    pub compute: Option<RootSignatureLayout>,
    /// Authoritative root-parameter slot map of the translated root signature.
    ///
    /// When present, every wire `SetRootConstants`/`SetRootBufferView`/
    /// `SetDescriptorTable` is resolved through it. When absent (a pipeline
    /// registered without translated signature bytes, as the legacy tests and
    /// pre-v3 paths do), the map is derived from the stage layouts' flattened
    /// order, which is exactly what those callers address.
    pub root_parameters: Option<RootParameterMap>,
}

impl PipelineAbi {
    /// ABI of a vertex/fragment graphics pipeline.
    pub fn graphics(
        vertex: Option<RootSignatureLayout>,
        fragment: Option<RootSignatureLayout>,
    ) -> Self {
        Self {
            vertex,
            fragment,
            ..Self::default()
        }
    }

    /// ABI of a mesh/object/fragment mesh pipeline.
    pub fn mesh(
        mesh: Option<RootSignatureLayout>,
        object: Option<RootSignatureLayout>,
        fragment: Option<RootSignatureLayout>,
    ) -> Self {
        Self {
            mesh,
            object,
            fragment,
            ..Self::default()
        }
    }

    /// ABI of a compute pipeline.
    pub fn compute(layout: RootSignatureLayout) -> Self {
        Self {
            compute: Some(layout),
            ..Self::default()
        }
    }

    /// Attaches the authoritative slot -> `(space, register, kind)` map parsed
    /// from the translated root signature.
    pub fn with_root_parameters(mut self, map: RootParameterMap) -> Self {
        self.root_parameters = Some(map);
        self
    }

    /// The declaration of wire root-parameter `slot`.
    ///
    /// Prefers the authoritative map. A pipeline registered without one (the
    /// legacy constructors) is resolved from the stage layouts' flattened
    /// order; if two stages declare different parameters at the same flattened
    /// index, that slot is ambiguous and resolves to `None` rather than one of
    /// them.
    pub fn root_parameter(&self, slot: usize) -> Option<RootParameterSlot> {
        if let Some(map) = self.root_parameters.as_ref() {
            return map.slot(slot).copied();
        }
        let mut found: Option<RootParameterSlot> = None;
        for layout in self.layouts() {
            if let Some(parameter) = layout.parameters().get(slot) {
                let candidate = RootParameterSlot::new(
                    parameter.space(),
                    parameter.register(),
                    parameter.kind(),
                    parameter.size(),
                );
                match found {
                    Some(previous) if previous != candidate => return None,
                    _ => found = Some(candidate),
                }
            }
        }
        found
    }

    /// The declaration of wire descriptor-table set `set`.
    ///
    /// A wire `DescriptorTableV3.set` is the table's *ordinal* among the
    /// pipeline's descriptor tables, not a flattened root-parameter index
    /// (a signature may declare root constants or root views before it). The
    /// authoritative map is enumerated in slot order; without one, the stage
    /// layouts are walked in registration order and a table declared by two
    /// stages counts once.
    pub fn root_parameter_for_table(&self, set: usize) -> Option<RootParameterSlot> {
        if let Some(map) = self.root_parameters.as_ref() {
            let mut ordinal = 0usize;
            for slot in map.slots() {
                if matches!(slot.kind(), RootParameterKind::DescriptorTable(_)) {
                    if ordinal == set {
                        return Some(*slot);
                    }
                    ordinal += 1;
                }
            }
            // The authoritative map may not describe this table (a signature
            // family whose table the converter reflects differently, or a map
            // that parsed but declared no table at this ordinal). The stage
            // layouts are the same fallback the legacy constructors use, so a
            // valid gate/title pipeline is never dropped for a partial map.
        }
        let mut tables: Vec<RootParameterSlot> = Vec::new();
        for layout in self.layouts() {
            for parameter in layout.parameters() {
                if matches!(parameter.kind(), RootParameterKind::DescriptorTable(_)) {
                    let candidate = RootParameterSlot::new(
                        parameter.space(),
                        parameter.register(),
                        parameter.kind(),
                        parameter.size(),
                    );
                    if !tables.contains(&candidate) {
                        tables.push(candidate);
                    }
                }
            }
        }
        tables.get(set).copied()
    }

    /// Every stage layout this ABI carries, in registration order.
    fn layouts(&self) -> impl Iterator<Item = &RootSignatureLayout> {
        self.vertex
            .iter()
            .chain(self.fragment.iter())
            .chain(self.mesh.iter())
            .chain(self.object.iter())
            .chain(self.compute.iter())
    }

    /// The declaration the translated root signature binds at `(space,
    /// register)` for a root argument of `kind`.
    ///
    /// This is the bridge from the guest's versioned root-argument plan, which
    /// names the translated register identity, to the flattened parameter the
    /// argument buffer is written at. Title root-parameter indices are *not*
    /// translated slots, so a wire `SetRootBufferView` at title index 0 must be
    /// looked up here rather than at flattened index 0.
    ///
    /// `kind` is the plan's argument kind and is coarser than
    /// [`RootParameterKind`]: [`PlanArgumentKind::RootView`] names any of a
    /// root CBV, SRV or UAV, because the wire's `SetRootBufferView` carries no
    /// view kind. The translated signature's root view registers are the
    /// title's root-parameter indices, so `(space, register)` is unique among
    /// them: one index is exactly one view kind. A descriptor table's reflected
    /// field may be `u32::MAX` (converter reflection reports every table that
    /// way); such a field matches any requested value. The first declaration in
    /// slot order wins deterministically, mirroring
    /// [`RootSignatureLayout::parameter_by`].
    pub fn root_parameter_for_argument(
        &self,
        space: u16,
        register: u16,
        kind: PlanArgumentKind,
    ) -> Option<RootParameterSlot> {
        if let Some(map) = self.root_parameters.as_ref() {
            return map
                .slots()
                .iter()
                .copied()
                .find(|slot| root_slot_declares_argument(slot, space, register, kind));
        }
        self.layouts().find_map(|layout| {
            layout.parameters().iter().find_map(|parameter| {
                let slot = RootParameterSlot::new(
                    parameter.space(),
                    parameter.register(),
                    parameter.kind(),
                    parameter.size(),
                );
                root_slot_declares_argument(&slot, space, register, kind).then_some(slot)
            })
        })
    }

    /// Whether no stage declares a root signature.
    pub fn is_empty(&self) -> bool {
        self.vertex.is_none()
            && self.fragment.is_none()
            && self.mesh.is_none()
            && self.object.is_none()
            && self.compute.is_none()
    }

    /// Layouts that apply to a converter-reflected stage, in binding order.
    fn stages_for(&self, stage: ReflectedStage) -> Vec<(ShaderStage, &RootSignatureLayout)> {
        match stage {
            ReflectedStage::Vertex => self
                .vertex
                .iter()
                .map(|layout| (ShaderStage::Vertex, layout))
                .collect(),
            ReflectedStage::Fragment => self
                .fragment
                .iter()
                .map(|layout| (ShaderStage::Fragment, layout))
                .collect(),
            ReflectedStage::Compute => self
                .compute
                .iter()
                .map(|layout| (ShaderStage::Compute, layout))
                .collect(),
            ReflectedStage::Other => Vec::new(),
        }
    }

    /// Mesh-stage layouts (mesh, object, fragment) in Metal binding order.
    fn mesh_stages(&self) -> Vec<(ShaderStage, &RootSignatureLayout)> {
        let mut stages = Vec::new();
        if let Some(layout) = self.mesh.as_ref() {
            stages.push((ShaderStage::Mesh, layout));
        }
        if let Some(layout) = self.object.as_ref() {
            stages.push((ShaderStage::Object, layout));
        }
        if let Some(layout) = self.fragment.as_ref() {
            stages.push((ShaderStage::Fragment, layout));
        }
        stages
    }
}

/// One sampler created for a converted sampler descriptor table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerDesc {
    /// `MTLSamplerMinMagFilter` for minification (`0` nearest, `1` linear).
    pub min_filter: usize,
    /// `MTLSamplerMinMagFilter` for magnification (`0` nearest, `1` linear).
    pub mag_filter: usize,
    /// `MTLSamplerAddressMode` for S (`1` clamp-to-edge).
    pub s_address_mode: usize,
    /// `MTLSamplerAddressMode` for T.
    pub t_address_mode: usize,
    /// `MTLSamplerAddressMode` for R.
    pub r_address_mode: usize,
}

impl SamplerDesc {
    /// Nearest filtering with clamp-to-edge addressing on every axis.
    ///
    /// This is the sampler every exact-texel proof in this crate uses: with a
    /// 1:1 footprint the sampled value is the texel, not a filtered blend.
    pub const fn nearest_clamp() -> Self {
        Self {
            min_filter: 0,
            mag_filter: 0,
            s_address_mode: 1,
            t_address_mode: 1,
            r_address_mode: 1,
        }
    }
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

/// One uploaded converted argument buffer plus its resident resources.
///
/// The buffer is a freshly allocated shared `MTLBuffer`; its bytes were written
/// on the CPU from [`ArgumentBufferBuilder`] output before any encoder existed.
/// No resource data is copied: root views and tables carry 64-bit device
/// addresses / `gpuResourceID`s only.
struct PreparedArgumentBuffer {
    stage: ShaderStage,
    index: usize,
    buffer: ObjcObject,
    /// The converter's implicit resource descriptor heap argument buffer.
    ///
    /// Bound at [`crate::macos::bindings::RESOURCE_DESCRIPTOR_HEAP_BUFFER_INDEX`]
    /// when a title-memory page table names heap indices.
    heap: Option<ObjcObject>,
    /// Converted descriptor tables this argument buffer points at.
    ///
    /// Each table is its own `MTLBuffer`, and `AbiPending` is cleared as soon as
    /// this argument buffer is uploaded — long before the submission is
    /// encoded. Owning the retains here keeps every table alive until
    /// `encode_v3_segments` pins the objects in the retire queue; without it the
    /// GPU (and `useResource:`) reads a released buffer.
    tables: Vec<ObjcObject>,
    /// ADR-055 title-memory buffers this argument buffer references: the page
    /// table, write bitmap, heap argument buffer and every allocation it
    /// names, retained until any in-flight command buffer is done.
    title: Vec<ObjcObject>,
    /// Resources the argument buffer references, declared with `useResource:`.
    resources: Vec<objc::Id>,
}

/// One validated compute-segment command.
enum PreparedComputeOp {
    Pipeline(objc::Id),
    Bytes { slot: usize, bytes: Vec<u8> },
    Bind(Vec<BoundRange>),
    /// One converted top-level argument buffer at its reflected buffer index.
    ArgumentBuffer(PreparedArgumentBuffer),
    Dispatch(PreparedV3Dispatch),
    /// One GPU-driven dispatch; the grid is read by the GPU.
    DispatchIndirect {
        pipeline: objc::Id,
        threads_per_group: MTLSize,
        arguments: indirect::IndirectArguments,
    },
    Barrier,
}

/// Ownership of an optional depth-stencil state used by one prepared draw.
enum DepthState {
    /// State owned by a registered legacy `GraphicsPipeline`.
    Borrowed(objc::Id),
    /// State created for a content-hashed pipeline and owned by the prepare op.
    Owned(ObjcObject),
}

impl DepthState {
    fn raw(&self) -> objc::Id {
        match self {
            Self::Borrowed(raw) => *raw,
            Self::Owned(object) => object.as_raw(),
        }
    }
}

/// One validated render-segment command.
enum PreparedRenderOp {
    Pipeline {
        state: objc::Id,
        depth_state: Option<DepthState>,
    },
    /// Binds a depth/stencil state built from a wire `SetDepthStencilState`.
    /// The id is owned by the device's [`DepthStencilCache`].
    DepthStencil(objc::Id),
    Viewport(RenderViewport),
    Scissor(RenderScissorRect),
    /// Encoder-level rasterizer state (cull, winding, fill, depth bias).
    Rasterizer(RasterizerState),
    /// Stencil reference value (`OMSetStencilRef`).
    StencilRef(u8),
    /// Blend colour constant (`OMSetBlendFactor`).
    BlendColor([f32; 4]),
    VertexBuffers(Vec<BoundVertex>),
    VertexBytes {
        slot: usize,
        bytes: Vec<u8>,
    },
    /// One converted top-level argument buffer at its reflected buffer index.
    ArgumentBuffer(PreparedArgumentBuffer),
    /// Root constants bound at a dedicated reflected Metal buffer index.
    StageBytes {
        stage: ShaderStage,
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
    /// Pitched upload-buffer to texture copy.
    CopyBufferToTexture(blit::BufferTextureCopy),
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
        /// One render pass per sub-pass; an explicit clear inside a pass splits
        /// the pass so the clear becomes the next sub-pass's load action.
        passes: Vec<PreparedRenderPass>,
        width: u32,
        height: u32,
    },
    Blit(Vec<PreparedBlitOp>),
}

/// One `MTLRenderCommandEncoder` sub-pass of a v3 render pass.
struct PreparedRenderPass {
    /// Descriptor owned by the submission's autorelease pool.
    descriptor: objc::Id,
    /// Replayed commands, including the state prefix when the sub-pass follows
    /// an explicit clear.
    ops: Vec<PreparedRenderOp>,
    /// ADR-053: first colour attachment, for the opt-in offscreen-RT sample.
    trace_target: Option<crate::macos::trace::TraceTarget>,
    /// ADR-053: draw-family ops this sub-pass encodes.
    draws: u32,
}

/// Counts the draw-family ops one prepared sub-pass encodes.
fn draw_op_count(ops: &[PreparedRenderOp]) -> u32 {
    ops.iter()
        .filter(|op| {
            matches!(
                op,
                PreparedRenderOp::Draw { .. }
                    | PreparedRenderOp::DrawIndexed { .. }
                    | PreparedRenderOp::DispatchMesh { .. }
                    | PreparedRenderOp::DrawIndirect { .. }
                    | PreparedRenderOp::DrawIndexedIndirect { .. }
                    | PreparedRenderOp::DispatchMeshIndirect { .. }
            )
        })
        .count() as u32
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
/// ADR-052: the handle class preparation reached before it refused.
///
/// The reply status carries no reason, so the failing resolution names itself
/// here; the host folds it into `telemetry.json` as `unknown_handle_site`.
static UNKNOWN_SITE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Names of the bounded refusal sites, indexed by [`UNKNOWN_SITE`].
pub const UNKNOWN_SITES: [&str; 8] = [
    "unknown",
    "buffer",
    "texture",
    "pipeline:compute",
    "pipeline:mesh",
    "pipeline:graphics",
    "sampler",
    "surface",
];

/// Records the resolution site that is about to be attempted.
fn note_unknown_site(site: &'static str) {
    let index = UNKNOWN_SITES
        .iter()
        .position(|candidate| *candidate == site)
        .unwrap_or(0);
    UNKNOWN_SITE.store(index as u32, std::sync::atomic::Ordering::Relaxed);
}

/// The last resolution site that refused an unknown handle.
pub fn last_unknown_site() -> &'static str {
    let index = UNKNOWN_SITE.load(std::sync::atomic::Ordering::Relaxed) as usize;
    UNKNOWN_SITES[index % UNKNOWN_SITES.len()]
}

/// ADR-052: the wire allocation id the last texture refusal named.
static UNKNOWN_HANDLE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The allocation id the last `UnknownHandle` refusal named, or `0`.
pub fn last_unknown_id() -> u64 {
    UNKNOWN_HANDLE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

fn resolve_buffer_range(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    allocation_id: u64,
    generation: u32,
    offset: u64,
    length: u64,
) -> Option<(objc::Id, usize, u64)> {
    note_unknown_site("buffer");
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
///
/// A handle resolves through the legacy `textures` map first and then through
/// the ADR-013 [`TextureRegistry`]; the two handle spaces are disjoint because
/// both draw from [`DeviceState::alloc_id`].
fn resolve_texture(
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    allocation_id: u64,
    generation: u32,
) -> Option<(objc::Id, crate::macos::render::TextureInfo)> {
    note_unknown_site("texture");
    if allocation_id == 0 || generation == 0 {
        note_attachment_miss(format_args!(
            "reference is zero id={allocation_id:#x} generation={generation}"
        ));
        return None;
    }
    let Some(registered_id) = allocations.get(&(allocation_id, generation)).copied() else {
        note_attachment_miss(format_args!(
            "not in the session snapshot id={allocation_id:#x} generation={generation} snapshot={} in_textures={} in_registry={}",
            allocations.len(),
            state.textures.contains_key(&allocation_id),
            state.registry_textures.contains_key(&allocation_id),
        ));
        return None;
    };
    if let Some(texture) = state.textures.get(&registered_id.0) {
        match texture_info(texture.as_raw()) {
            Ok(info) => return Some((texture.as_raw(), info)),
            Err(error) => {
                // SAFETY: a live `MTLTexture`; `pixelFormat` is immutable.
                let mtl = unsafe {
                    objc::msg_send_usize_0(texture.as_raw(), objc::sel(c"pixelFormat"))
                };
                note_attachment_miss(format_args!(
                    "registered texture has no readable geometry id={allocation_id:#x} mtl_format={mtl} error={error}"
                ));
                return None;
            }
        }
    }
    let Some(raw) = resolve_registered_texture(state, registered_id.0) else {
        note_attachment_miss(format_args!(
            "registry handle does not resolve id={allocation_id:#x} generation={generation}"
        ));
        return None;
    };
    match texture_info(raw) {
        Ok(info) => Some((raw, info)),
        Err(error) => {
            note_attachment_miss(format_args!(
                "registry texture has no readable geometry id={allocation_id:#x} error={error}"
            ));
            None
        }
    }
}

/// Bounded host-stderr diagnostic for an attachment resolution failure.
///
/// The host binary does not install a tracing subscriber, so a `tracing` event
/// would be dropped; this names the failed lookup in `host.log` at most 16
/// times per process (ADR-054 diagnosis).
fn note_attachment_miss(message: std::fmt::Arguments<'_>) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NOTES: AtomicU64 = AtomicU64::new(0);
    if NOTES.fetch_add(1, Ordering::Relaxed) < 16 {
        eprintln!("victory-metal-host: attachment {message}");
    }
}

/// Resolves a transport texture handle through the ADR-013 texture registry.
fn resolve_registered_texture(state: &DeviceState, handle: u64) -> Option<objc::Id> {
    let registry_id = *state.registry_textures.get(&handle)?;
    let registry = state.texture_registry.as_ref()?;
    registry.raw(registry_id)
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

/// One validated root-signature binding collected before its argument buffer
/// is uploaded.
enum PendingAbiBinding {
    /// Inline root constants for one declared root parameter.
    Constants {
        slot: RootParameterSlot,
        /// Wire root-parameter index the command named.
        wire_slot: usize,
        bytes: Vec<u8>,
    },
    /// One root CBV/SRV/UAV device address and the resource it names.
    RootView {
        slot: RootParameterSlot,
        /// Wire root-parameter index the command named.
        wire_slot: usize,
        address: u64,
        resource: objc::Id,
        usage: usize,
    },
    /// One converted descriptor array in GPU-visible shared memory.
    Table {
        slot: RootParameterSlot,
        /// Wire descriptor-set index the command named.
        wire_slot: usize,
        buffer: ObjcObject,
        address: u64,
        length: u64,
        count: u32,
        stride: usize,
        resources: Vec<(objc::Id, usize)>,
    },
}

/// Root-signature bindings collected for the active pipeline, in wire order.
#[derive(Default)]
struct AbiPending {
    bindings: Vec<PendingAbiBinding>,
}

impl AbiPending {
    fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }
}

/// Allocates a shared `MTLBuffer` and copies `bytes` into it.
///
/// The copy is a CPU write of *binding metadata* (device addresses, descriptor
/// arrays), never resource data and never a GPU readback. The returned object
/// owns the buffer until the prepared op is dropped after completion.
fn new_argument_buffer(device: objc::Id, bytes: &[u8]) -> Result<ObjcObject, MetalError> {
    let len = bytes.len().max(8);
    // SAFETY: `device` is a live `MTLDevice`; shared storage is CPU-visible.
    let raw = unsafe { objc::msg_send_new_buffer(device, len, 0) };
    let buffer = ObjcObject::from_retained(raw).ok_or_else(|| {
        MetalError::InternalError("argument buffer allocation returned null".to_string())
    })?;
    // SAFETY: `buffer` is a fresh shared `MTLBuffer` of `len` bytes.
    let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
    if contents.is_null() {
        return Err(MetalError::InternalError(
            "argument buffer has no CPU-visible contents".to_string(),
        ));
    }
    // SAFETY: The destination covers `bytes.len() <= len` bytes of a buffer no
    // command buffer references yet.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len());
    }
    Ok(buffer)
}

/// 64-bit GPU address of a live `MTLBuffer`.
fn buffer_gpu_address(buffer: objc::Id) -> u64 {
    // SAFETY: `buffer` is a live `MTLBuffer`; `gpuAddress` returns its base.
    unsafe { objc::msg_send_u64_0(buffer, objc::sel(c"gpuAddress")) }
}

/// 64-bit resource id of a live `MTLTexture`/`MTLSamplerState`.
fn resource_gpu_id(resource: objc::Id) -> u64 {
    // SAFETY: `resource` is a live `MTLResource`; `gpuResourceID` is its id.
    unsafe { objc::msg_send_u64_0(resource, objc::sel(c"gpuResourceID")) }
}

/// Wire table kind a converted range kind requires.
fn table_kind_accepts_wire(kind: DescriptorTableKind, wire_kind: DescriptorKind) -> bool {
    match kind {
        DescriptorTableKind::ConstantBufferView => wire_kind == DescriptorKind::Cbv,
        DescriptorTableKind::ShaderResourceView => wire_kind == DescriptorKind::Srv,
        DescriptorTableKind::UnorderedAccessView => wire_kind == DescriptorKind::Uav,
        DescriptorTableKind::Texture => {
            matches!(wire_kind, DescriptorKind::Srv | DescriptorKind::Uav)
        }
        DescriptorTableKind::Sampler => wire_kind == DescriptorKind::Sampler,
        // Reflection only says "Table". The wire entry variant selects the
        // converter descriptor shape, so every wire kind is expressible here.
        DescriptorTableKind::Unspecified => true,
    }
}

/// The converted descriptor shape one table actually uses.
///
/// The wire entry variant is authoritative where reflection is `Unspecified`
/// (the converter reports every range as `Type=Table`): a buffer entry is the
/// 24-byte `{device_address, 0, 0}` resource descriptor, a texture entry is
/// `{gpuResourceID, gpuResourceID, 0}`, and a sampler entry is
/// `{gpuResourceID, 0, 0}`. An explicitly attached range kind is cross-checked
/// against the entries so a mismatched caller cannot silently pick a shape.
fn descriptor_table_shape(
    table: &crate::wire_v3::DescriptorTableV3,
    kind: DescriptorTableKind,
) -> Result<DescriptorTableKind, MetalError> {
    use crate::wire_v3::DescriptorTableEntryV3;

    let mut entry_shape: Option<DescriptorTableKind> = None;
    for entry in &table.bindings {
        let shape = match entry {
            DescriptorTableEntryV3::Buffer(_) => {
                if table.kind == DescriptorKind::Sampler {
                    return Err(MetalError::InvalidArg(
                        "a sampler table cannot hold buffer entries",
                    ));
                }
                DescriptorTableKind::Unspecified
            }
            DescriptorTableEntryV3::Texture(_) => {
                if !matches!(table.kind, DescriptorKind::Srv | DescriptorKind::Uav) {
                    return Err(MetalError::InvalidArg(
                        "a texture entry needs an SRV or UAV table",
                    ));
                }
                DescriptorTableKind::Texture
            }
            DescriptorTableEntryV3::Sampler(_) => {
                if table.kind != DescriptorKind::Sampler {
                    return Err(MetalError::InvalidArg(
                        "a sampler entry needs a sampler table",
                    ));
                }
                DescriptorTableKind::Sampler
            }
        };
        match entry_shape {
            Some(previous) if previous != shape => {
                return Err(MetalError::InvalidArg(
                    "descriptor table mixes buffer, texture and sampler entries",
                ));
            }
            _ => entry_shape = Some(shape),
        }
    }
    let entry_shape = entry_shape
        .ok_or(MetalError::InvalidArg("descriptor table is empty"))?;
    match kind {
        DescriptorTableKind::Unspecified => Ok(entry_shape),
        DescriptorTableKind::Texture if entry_shape == DescriptorTableKind::Texture => {
            Ok(DescriptorTableKind::Texture)
        }
        DescriptorTableKind::Sampler if entry_shape == DescriptorTableKind::Sampler => {
            Ok(DescriptorTableKind::Sampler)
        }
        DescriptorTableKind::ConstantBufferView
        | DescriptorTableKind::ShaderResourceView
        | DescriptorTableKind::UnorderedAccessView
            if entry_shape == DescriptorTableKind::Unspecified =>
        {
            Ok(kind)
        }
        _ => Err(MetalError::InvalidArg(
            "descriptor table entry does not match the reflected range kind",
        )),
    }
}

/// Converts a wire table into buffer-range bindings for the pre-ABI path.
///
/// Registered texture and sampler handles have no plain `[[buffer(n)]]` slot
/// binding; only a converted argument buffer carries their 24-byte descriptor,
/// so the legacy path refuses them instead of misreading a handle as an
/// address.
fn buffer_table_bindings(
    table: &crate::wire_v3::DescriptorTableV3,
) -> Result<Vec<wire::BufferBinding>, MetalError> {
    let mut bindings = Vec::with_capacity(table.bindings.len());
    for entry in &table.bindings {
        bindings.push(entry.as_buffer().copied().ok_or_else(|| {
            MetalError::NotSupported(
                "registered texture and sampler entries need a converted argument buffer",
            )
        })?);
    }
    Ok(bindings)
}

/// Builds the converted descriptor array for one wire table.
///
/// Buffer entries copy the resolved view address of each `xmem::share`-backed
/// buffer into the converter's 24-byte resource descriptor; texture entries
/// copy `gpuResourceID`s of registered `MTLTexture`s; sampler entries copy
/// `gpuResourceID`s of registered `MTLSamplerState`s. Every entry is resolved
/// against the live maps first, so an unknown handle is
/// [`PrepareAbort::UnknownHandle`] before any encoder exists.
fn build_descriptor_table(
    device: objc::Id,
    state: &DeviceState,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    table: &crate::wire_v3::DescriptorTableV3,
    slot: RootParameterSlot,
) -> Result<PendingAbiBinding, PrepareAbort> {
    use crate::wire_v3::DescriptorTableEntryV3;

    let RootParameterKind::DescriptorTable(kind) = slot.kind() else {
        return Err(MetalError::InvalidArg(
            "descriptor table names a non-table root parameter",
        )
        .into());
    };
    if !table_kind_accepts_wire(kind, table.kind) {
        return Err(MetalError::InvalidArg(
            "descriptor table kind does not match the reflected range kind",
        )
        .into());
    }
    let shape = descriptor_table_shape(table, kind)?;
    let usage = match table.kind {
        DescriptorKind::Uav => MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
        _ => MTL_RESOURCE_USAGE_READ,
    };
    let mut builder = DescriptorTableBuilder::new(shape);
    let mut resources = Vec::with_capacity(table.bindings.len());
    for entry in &table.bindings {
        match entry {
            DescriptorTableEntryV3::Texture(handle)
                if shape == DescriptorTableKind::Texture =>
            {
                let texture =
                    resolve_registered_texture(state, *handle).ok_or(PrepareAbort::UnknownHandle)?;
                builder.push_texture(resource_gpu_id(texture))?;
                resources.push((texture, usage));
            }
            DescriptorTableEntryV3::Sampler(handle) if shape == DescriptorTableKind::Sampler => {
                note_unknown_site("sampler");
                let sampler = state
                    .samplers
                    .get(handle)
                    .map(ObjcObject::as_raw)
                    .ok_or(PrepareAbort::UnknownHandle)?;
                builder.push_sampler(resource_gpu_id(sampler))?;
                // `MTLSamplerState` is not an `MTLResource`: it is referenced
                // through the descriptor and kept alive by `state.samplers`,
                // so it is never passed to `useResource:` (which would fault).
            }
            DescriptorTableEntryV3::Buffer(binding)
                if matches!(
                    shape,
                    DescriptorTableKind::ConstantBufferView
                        | DescriptorTableKind::ShaderResourceView
                        | DescriptorTableKind::UnorderedAccessView
                        | DescriptorTableKind::Unspecified
                ) =>
            {
                let (buffer, offset, _length) = resolve_buffer_range(
                    state,
                    allocations,
                    binding.allocation_id,
                    binding.generation,
                    binding.offset,
                    binding.length,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                let address = buffer_gpu_address(buffer)
                    .checked_add(offset as u64)
                    .ok_or(MetalError::InvalidArg(
                        "descriptor table address overflows u64",
                    ))?;
                builder.push_address(address)?;
                resources.push((buffer, usage));
            }
            _ => {
                return Err(MetalError::InvalidArg(
                    "descriptor table entry does not match its encoding shape",
                )
                .into());
            }
        }
    }
    let buffer = new_argument_buffer(device, builder.bytes())?;
    let address = buffer_gpu_address(buffer.as_raw());
    let table_binding = builder.binding(address)?;
    Ok(PendingAbiBinding::Table {
        slot,
        wire_slot: usize::from(table.set),
        buffer,
        address,
        length: table_binding.length,
        count: table_binding.count,
        stride: table_binding.stride,
        resources,
    })
}

impl PendingAbiBinding {
    /// The root-parameter declaration this wire argument resolved to.
    fn slot(&self) -> RootParameterSlot {
        match self {
            Self::Constants { slot, .. }
            | Self::RootView { slot, .. }
            | Self::Table { slot, .. } => *slot,
        }
    }

    /// The wire slot the command named (root-parameter or descriptor-set index).
    fn wire_slot(&self) -> usize {
        match self {
            Self::Constants { wire_slot, .. }
            | Self::RootView { wire_slot, .. }
            | Self::Table { wire_slot, .. } => *wire_slot,
        }
    }
}

/// Upper bound on bytes the default MEMORY_SPACE fallback may allocate for one
/// submission. A stage declares a handful of 8-byte descriptors; a signature
/// that needs more than this is refused instead of sized.
const MEMORY_SPACE_DEFAULT_BUDGET: usize = 1 << 20;

/// Uploads one argument buffer per stage and appends the bind ops.
///
/// Every pending wire argument is resolved by `(space, register, kind)` through
/// [`RootSignatureLayout::parameter_by`], never by its wire slot. A pending
/// argument no stage declares is a counted refusal. After the title bindings,
/// every MEMORY_SPACE parameter of the stage that received no binding is
/// pointed at a zero-filled shared buffer (one allocation per stage layout, at
/// most [`MEMORY_SPACE_DEFAULT_BUDGET`] bytes per submission).
///
/// The builders write only checked 64-bit device addresses, `gpuResourceID`s
/// and inline root constants. Stage layouts without any encoded parameter are
/// skipped, so a fragment stage that does not read the root signature keeps its
/// own `buffer(2)` untouched.
/// The layout index a pending wire argument names in one stage layout.
///
/// The resolver is `(space, register, kind)`. The wire also carries the flat
/// index the title named, which is exactly the converter's parameter index on
/// the legacy registration path (a gate pipeline registered from supplied
/// reflection has no versioned signature map). When the triple is not declared
/// — or a table's reflected triple is `u32::MAX` — the wire index is the
/// deterministic identity, never a silent drop.
fn abi_parameter_index(
    layout: &RootSignatureLayout,
    binding: &PendingAbiBinding,
) -> Option<usize> {
    let slot = binding.slot();
    let index = binding.wire_slot();
    // Converter reflection reports a descriptor table's Slot/Space as
    // `u32::MAX`, so every table's reflected triple is identical: a triple
    // lookup would bind the first table twice. The wire index is that path's
    // only identity, and it is checked first whenever the layout agrees with it
    // (an unknown `u32::MAX` field, or the exact declared pair).
    if let Some(parameter) = layout.parameters().get(index)
        && abi_kinds_compatible(parameter.kind(), slot.kind())
        && (parameter.space() == slot.space() || parameter.space() == u32::MAX)
        && (parameter.register() == slot.register() || parameter.register() == u32::MAX)
    {
        return Some(index);
    }
    if let Ok(Some(found)) = layout.parameter_by(slot.space(), slot.register(), slot.kind()) {
        return Some(found);
    }
    None
}

/// Whether a wire argument kind can address a reflected parameter kind.
///
/// A table is a table; converter reflection cannot distinguish its range kinds
/// (`Table` is stored as `Unspecified`), so those match any table.
fn abi_kinds_compatible(declared: RootParameterKind, named: RootParameterKind) -> bool {
    match (declared, named) {
        (
            RootParameterKind::DescriptorTable(_),
            RootParameterKind::DescriptorTable(_),
        ) => true,
        (left, right) => left == right,
    }
}

/// Whether one translated declaration binds the `(space, register)` a root
/// argument of `kind` names.
///
/// The plan's root view kind covers all three view kinds, because the wire
/// `SetRootBufferView` carries no view kind; the translated signature's root
/// view registers are title root-parameter indices, so one index is exactly one
/// view kind. A table's reflected `(space, register)` may be `u32::MAX`
/// (converter reflection reports every table that way) and then matches any
/// requested value. A sampler table never names a root parameter: the
/// translated signature's samplers are static samplers.
/// Whether a translated parameter kind can carry a plan argument kind,
/// independent of space and register.
fn root_kind_matches(kind: RootParameterKind, plan: PlanArgumentKind) -> bool {
    match plan {
        PlanArgumentKind::Constants => kind == RootParameterKind::Constants,
        PlanArgumentKind::RootView => matches!(
            kind,
            RootParameterKind::ConstantBufferView
                | RootParameterKind::ShaderResourceView
                | RootParameterKind::UnorderedAccessView
        ),
        PlanArgumentKind::DescriptorTable => matches!(kind, RootParameterKind::DescriptorTable(_)),
        PlanArgumentKind::SamplerTable => false,
    }
}

fn root_slot_declares_argument(
    slot: &RootParameterSlot,
    space: u16,
    register: u16,
    kind: PlanArgumentKind,
) -> bool {
    let table = matches!(slot.kind(), RootParameterKind::DescriptorTable(_));
    let space_matches = slot.space() == u32::from(space) || (table && slot.space() == u32::MAX);
    let register_matches =
        slot.register() == u32::from(register) || (table && slot.register() == u32::MAX);
    if !space_matches || !register_matches {
        return false;
    }
    match kind {
        PlanArgumentKind::Constants => slot.kind() == RootParameterKind::Constants,
        PlanArgumentKind::RootView => matches!(
            slot.kind(),
            RootParameterKind::ConstantBufferView
                | RootParameterKind::ShaderResourceView
                | RootParameterKind::UnorderedAccessView
        ),
        PlanArgumentKind::DescriptorTable => {
            matches!(slot.kind(), RootParameterKind::DescriptorTable(_))
        }
        PlanArgumentKind::SamplerTable => false,
    }
}

/// Resolves one planned wire root argument to the translated parameter it
/// names, or refuses it by name.
///
/// The plan entry is authoritative: a guest that recorded it named a translated
/// register identity, so a layout that does not declare that identity is a
/// counted ABI refusal instead of a binding at the wrong parameter. The entry's
/// kind must also match the command (a `SetRootBufferView` cannot consume a
/// root-constants record).
fn planned_root_parameter(
    transport: &DirectMetalTransport,
    abi: &PipelineAbi,
    entry: &RootPlanEntry,
    kind: PlanArgumentKind,
    mismatched: &'static str,
) -> Result<RootParameterSlot, PrepareAbort> {
    if entry.kind != kind {
        transport.count_abi_refusal();
        return Err(MetalError::InvalidArg(mismatched).into());
    }
    if let Some(slot) = abi.root_parameter_for_argument(entry.space, entry.register, kind) {
        return Ok(slot);
    }
    transport.count_abi_refusal();
    Err(MetalError::InvalidArg(
        "a wire root argument names a translated parameter the root signature does not declare",
    )
    .into())
}

/// Which root argument a `SetRoot...` command names, and the refusals that
/// command's resolution reports.
#[derive(Clone, Copy)]
struct RootArgumentSite {
    /// The translated argument kind the command consumes.
    kind: PlanArgumentKind,
    /// Name of the refusal when no parameter is declared for the wire slot.
    undeclared: &'static str,
    /// Name of the refusal when the plan entry is another argument kind.
    mismatched: &'static str,
}

impl RootArgumentSite {
    const CONSTANTS: Self = Self {
        kind: PlanArgumentKind::Constants,
        undeclared: "root constants name no declared root parameter",
        mismatched: "root constants plan names a non-constant root argument",
    };
    const ROOT_VIEW: Self = Self {
        kind: PlanArgumentKind::RootView,
        undeclared: "root buffer view names no declared root parameter",
        mismatched: "root buffer view plan names a non-view root argument",
    };
}

/// Resolves the wire root-parameter slot `wire_slot` of `site` against the
/// selected pipeline.
///
/// A plan entry for the slot is resolved by its translated `(space, register,
/// kind)`; a plan entry that names no declared parameter is a counted ABI
/// refusal. Without a plan entry the legacy wire-slot lookup is kept, exactly as
/// a guest that predates the plan expects, and counted separately through
/// [`DirectMetalTransport::count_legacy_root_binding`].
fn resolve_wire_root_argument(
    transport: &DirectMetalTransport,
    abi: &PipelineAbi,
    plan: Option<&RootParameterPlan>,
    wire_slot: u16,
    site: RootArgumentSite,
) -> Result<RootParameterSlot, PrepareAbort> {
    if let Some(entry) = plan.and_then(|plan| plan.entry(wire_slot)) {
        return planned_root_parameter(transport, abi, entry, site.kind, site.mismatched);
    }
    // Without a plan entry, a translated parameter is still derivable from the
    // converter's own reflection: its register numbers follow the shader's
    // bindings, and the wire slot is the title's root-parameter index, which
    // names that register. Search the reflected layouts by kind and register
    // (the reflected space can differ from the emitter's `ROOT_CBV_SPACE`
    // convention, as Boot #28 showed: the converter reports the root CBV at
    // space 0). This is narrower than the legacy slot lookup, which reads a
    // translated index as a title index and can land on a constants block.
    if let Some(slot) = abi.layouts().find_map(|layout| {
        layout
            .parameters()
            .iter()
            .find(|parameter| {
                parameter.register() == u32::from(wire_slot)
                    && root_kind_matches(parameter.kind(), site.kind)
            })
            .map(|parameter| {
                RootParameterSlot::new(
                    parameter.space(),
                    parameter.register(),
                    parameter.kind(),
                    parameter.size(),
                )
            })
    }) {
        return Ok(slot);
    }
    // A descriptor table names its base register, not the title's root index,
    // so the register search above cannot map it. When the translated signature
    // declares exactly one parameter of the command's kind, that parameter is
    // the binding: unambiguous, and never a guess between two tables.
    let mut only: Option<RootParameterSlot> = None;
    let mut several = false;
    for layout in abi.layouts() {
        for parameter in layout.parameters() {
            if !root_kind_matches(parameter.kind(), site.kind) {
                continue;
            }
            let slot = RootParameterSlot::new(
                parameter.space(),
                parameter.register(),
                parameter.kind(),
                parameter.size(),
            );
            match &only {
                None => only = Some(slot),
                Some(existing)
                    if existing.space() == slot.space()
                        && existing.register() == slot.register() => {}
                Some(_) => several = true,
            }
        }
    }
    if !several && let Some(slot) = only {
        return Ok(slot);
    }
    transport.count_legacy_root_binding();
    // Bounded stderr (the daemon installs no tracing subscriber): names whether
    // the guest sent a plan at all and how many entries it carried.
    use std::sync::atomic::{AtomicU64, Ordering};
    static NOTES: AtomicU64 = AtomicU64::new(0);
    if NOTES.fetch_add(1, Ordering::Relaxed) < 8 {
        eprintln!(
            "victory-metal-host: legacy root binding wire_slot={wire_slot} plan_present={} plan_entries={}",
            plan.is_some(),
            plan.map_or(0, |plan| plan.len()),
        );
    }
    abi.root_parameter(usize::from(wire_slot)).ok_or_else(|| {
        transport.count_abi_refusal();
        MetalError::InvalidArg(site.undeclared).into()
    })
}


fn flush_abi(
    transport: &DirectMetalTransport,
    state: &mut DeviceState,
    stages: &[(ShaderStage, RootSignatureLayout)],
    pending: &mut AbiPending,
) -> Result<
    (
        Vec<PreparedArgumentBuffer>,
        Vec<(ShaderStage, usize, Vec<u8>)>,
    ),
    PrepareAbort,
> {
    let device = transport.device_raw();
    let mut buffers = Vec::new();
    let mut bytes_ops = Vec::new();
    if pending.is_empty()
        && !stages.iter().any(|(_, layout)| {
            layout
                .parameters()
                .iter()
                .any(|parameter| parameter.space() == victory_shader::emit::MEMORY_SPACE)
        })
    {
        return Ok((buffers, bytes_ops));
    }

    // ADR-055: build the live page table once for this flush. A stage that
    // declares MEMORY_SPACE t0/u0 gets the real buffers; no map (or a map
    // naming an unregistered allocation) leaves the previous zero fallback.
    let needs_title = stages.iter().any(|(_, layout)| {
        layout
            .parameters()
            .iter()
            .any(|parameter| parameter.space() == victory_shader::emit::MEMORY_SPACE)
    });
    let title = if needs_title {
        let DeviceState { title, buffers: registered, .. } = state;
        title.binding(device, registered)?
    } else {
        None
    };

    // Resolve every pending argument against every stage before allocating
    // anything. A wire slot no layout declares is refused by name, never
    // silently dropped.
    for binding in &pending.bindings {
        let slot = binding.slot();
        let mut declared = false;
        for (_stage, layout) in stages {
            if abi_parameter_index(layout, binding).is_some() {
                declared = true;
                break;
            }
        }
        if !declared {
            transport.count_abi_refusal();
            return Err(MetalError::InvalidArg(
                "a wire root argument is not declared by the converted layout",
            )
            .into());
        }
    }

    // Flattened indices each stage received a title binding for.
    let mut bound_per_stage: Vec<Vec<bool>> = Vec::with_capacity(stages.len());
    for (_stage, layout) in stages {
        let mut bound = vec![false; layout.parameters().len()];
        for binding in &pending.bindings {
            let slot = binding.slot();
            if let Some(index) = abi_parameter_index(layout, binding) {
                bound[index] = true;
            }
        }
        bound_per_stage.push(bound);
    }

    // Size the default MEMORY_SPACE buffers first, so an over-budget layout is
    // refused before any Metal allocation exists.
    let mut default_bytes = 0usize;
    for ((_stage, layout), bound) in stages.iter().zip(&bound_per_stage) {
        for (index, parameter) in layout.parameters().iter().enumerate() {
            if parameter.space() == victory_shader::emit::MEMORY_SPACE && !bound[index] {
                default_bytes = default_bytes
                    .checked_add(parameter.size().max(crate::macos::bindings::DEVICE_ADDRESS_BYTES))
                    .ok_or(MetalError::InvalidArg(
                        "memory-space default binding size overflows",
                    ))?;
            }
        }
    }
    if default_bytes > MEMORY_SPACE_DEFAULT_BUDGET {
        transport.count_abi_refusal();
        return Err(MetalError::InvalidArg(
            "memory-space default bindings exceed the 1 MiB budget",
        )
        .into());
    }

    for (stage_index, (stage, layout)) in stages.iter().enumerate() {
        let mut builder = ArgumentBufferBuilder::new(layout);
        let mut resources: Vec<(objc::Id, usize)> = Vec::new();
        let mut tables: Vec<ObjcObject> = Vec::new();
        let mut stage_bytes: Option<(usize, Vec<u8>)> = None;
        for binding in &pending.bindings {
            let slot = binding.slot();
            let Some(index) = abi_parameter_index(layout, binding) else {
                // The parameter belongs to another stage (root signatures are
                // per-stage visible); it is not this stage's argument buffer.
                continue;
            };
            match binding {
                PendingAbiBinding::Constants { bytes, .. } => {
                    if layout.parameters()[index].kind() != RootParameterKind::Constants {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "root constants do not name a constant root parameter",
                        )
                        .into());
                    }
                    match layout.root_constants_metal_index() {
                        Some(metal_index) => stage_bytes = Some((metal_index, bytes.clone())),
                        None => builder.set_root_constants(index, bytes)?,
                    }
                }
                PendingAbiBinding::RootView {
                    address,
                    resource,
                    usage,
                    ..
                } => {
                    match layout.parameters()[index].kind() {
                        RootParameterKind::ConstantBufferView
                        | RootParameterKind::ShaderResourceView
                        | RootParameterKind::UnorderedAccessView => {}
                        _ => {
                            transport.count_abi_refusal();
                            return Err(MetalError::InvalidArg(
                                "root buffer view does not name a buffer root parameter",
                            )
                            .into());
                        }
                    }
                    builder.set_buffer_address(index, *address)?;
                    resources.push((*resource, *usage));
                }
                PendingAbiBinding::Table {
                    buffer,
                    address,
                    length,
                    count,
                    stride,
                    resources: table_resources,
                    ..
                } => {
                    if !matches!(
                        layout.parameters()[index].kind(),
                        RootParameterKind::DescriptorTable(_)
                    ) {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "descriptor table does not name a table root parameter",
                        )
                        .into());
                    }
                    // Re-check the converted extent against the bound range
                    // before the argument buffer points at it.
                    let extent = u64::from(*count)
                        .checked_mul(*stride as u64)
                        .ok_or(MetalError::InvalidArg(
                            "descriptor table converted extent overflows",
                        ))?;
                    if extent > *length {
                        return Err(MetalError::InvalidArg(
                            "descriptor table converted extent exceeds the bound range",
                        )
                        .into());
                    }
                    builder.set_table_address(index, *address)?;
                    resources.push((buffer.as_raw(), MTL_RESOURCE_USAGE_READ));
                    resources.extend(table_resources.iter().copied());
                    // `pending` is cleared by the caller as soon as this
                    // argument buffer is uploaded, so the table buffer must be
                    // retained into the prepared op that will outlive it.
                    tables.push(retain_object(buffer.as_raw()).ok_or_else(|| {
                        MetalError::InternalError(
                            "descriptor table buffer retain returned null".to_string(),
                        )
                    })?);
                }
            }
        }

        // MEMORY_SPACE: the live page table `t0` and write bitmap `u0` when a
        // map is installed, and one zero-filled shared buffer carved into one
        // aligned sub-range per remaining unbound parameter.
        let bound = &bound_per_stage[stage_index];
        let mut unbound: Vec<usize> = Vec::new();
        let mut title_parameters = 0u64;
        let mut title_bytes = 0u64;
        for (index, parameter) in layout.parameters().iter().enumerate() {
            if parameter.space() != victory_shader::emit::MEMORY_SPACE || bound[index] {
                continue;
            }
            if let Some(binding) = &title {
                let view = matches!(
                    parameter.kind(),
                    RootParameterKind::ShaderResourceView
                        | RootParameterKind::UnorderedAccessView
                );
                if view && parameter.register() == 0 {
                    let (address, bytes) = match parameter.kind() {
                        RootParameterKind::ShaderResourceView => {
                            (buffer_gpu_address(binding.table), binding.table_bytes)
                        }
                        _ => (buffer_gpu_address(binding.bitmap), binding.bitmap_bytes),
                    };
                    builder.set_buffer_address(index, address)?;
                    title_parameters += 1;
                    title_bytes = title_bytes
                        .saturating_add(bytes)
                        .saturating_add(binding.heap_bytes);
                    continue;
                }
            }
            unbound.push(index);
        }
        if title_parameters > 0 {
            transport.count_memory_space(title_parameters, title_bytes);
        }
        if !unbound.is_empty() {
            let mut total = 0usize;
            for &index in &unbound {
                total = total
                    .checked_add(
                        layout.parameters()[index]
                            .size()
                            .max(crate::macos::bindings::DEVICE_ADDRESS_BYTES),
                    )
                    .ok_or(MetalError::InvalidArg(
                        "memory-space default binding size overflows",
                    ))?;
            }
            let zero = vec![0u8; total.max(8)];
            let buffer = new_argument_buffer(device, &zero)?;
            let base = buffer_gpu_address(buffer.as_raw());
            let mut offset = 0usize;
            for &index in &unbound {
                let parameter = &layout.parameters()[index];
                if !matches!(
                    parameter.kind(),
                    RootParameterKind::ConstantBufferView
                        | RootParameterKind::ShaderResourceView
                        | RootParameterKind::UnorderedAccessView
                ) {
                    transport.count_abi_refusal();
                    return Err(MetalError::InvalidArg(
                        "a memory-space parameter is not a buffer view",
                    )
                    .into());
                }
                let address = base.checked_add(offset as u64).ok_or(
                    MetalError::InvalidArg("memory-space default address overflows"),
                )?;
                builder.set_buffer_address(index, address)?;
                offset += parameter.size().max(crate::macos::bindings::DEVICE_ADDRESS_BYTES);
            }
            resources.push((buffer.as_raw(), MTL_RESOURCE_USAGE_READ));
            tables.push(buffer);
            transport.count_memory_space(unbound.len() as u64, zero.len() as u64);
        }

        if let Some((index, bytes)) = stage_bytes {
            bytes_ops.push((*stage, index, bytes));
        }
        let mut heap = None;
        let mut title_owned = Vec::new();
        if title_parameters > 0 {
            let binding = title.as_ref().expect("counted title parameters");
            for raw in &binding.resources {
                resources.push((*raw, MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE));
                if let Some(object) = retain_object(*raw) {
                    title_owned.push(object);
                }
            }
            heap = retain_object(binding.heap);
        }
        if builder.encoded_parameters() == 0 {
            continue;
        }
        let buffer = new_argument_buffer(device, builder.bytes())?;
        buffers.push(PreparedArgumentBuffer {
            stage: *stage,
            index: layout.argument_buffer_index(),
            buffer,
            heap,
            tables,
            title: title_owned,
            resources: resources.into_iter().map(|(resource, _)| resource).collect(),
        });
    }
    Ok((buffers, bytes_ops))
}

/// Stage layouts a render-family pipeline binds converter argument buffers for.
///
/// A mesh-family pipeline (mesh/object/fragment) binds its mesh stages; the
/// converter never emits a vertex stage for a mesh PSO, so looking only at
/// vertex/fragment would miss every mesh-stage root parameter.
/// ADR-053: records one encoded draw/dispatch in the opt-in tracer.
///
/// A no-op unless `VICTORY_METAL_DRAW_TRACE` armed the trace and fewer than
/// [`crate::macos::trace::MAX_TRACED_WORK`] calls were described. The line is
/// written to stderr the host log already carries; it contains counts, shapes
/// and the attachment's geometry and format, never resource contents.
#[allow(clippy::too_many_arguments)]
fn trace_encoded_work(
    state: &mut DeviceState,
    kind: &'static str,
    pipeline_id: u64,
    count: u32,
    instances: u32,
    viewport: Option<[u32; 4]>,
    scissor: Option<[u32; 4]>,
    abi: Option<&PipelineAbi>,
    mesh_family: bool,
    target: Option<crate::macos::trace::TraceTarget>,
) {
    let Some(trace) = state.trace.as_mut() else {
        return;
    };
    if trace.saturated() {
        return;
    }
    let (parameters, bytes) = abi_argument_summary(abi, mesh_family);
    let record = crate::macos::trace::DrawRecord {
        ordinal: trace.records.len() + 1,
        kind,
        pipeline_id,
        count,
        instances,
        viewport,
        scissor,
        abi_parameters: parameters,
        abi_bytes: bytes,
        target,
    };
    if let Some(line) = trace.record(record) {
        eprintln!("{line}");
    }
}

/// Counts a pipeline's reflected parameters and argument-buffer bytes.
fn abi_argument_summary(abi: Option<&PipelineAbi>, mesh_family: bool) -> (u32, u32) {
    let Some(abi) = abi else {
        return (0, 0);
    };
    let mut parameters = 0u32;
    let mut bytes = 0u32;
    for (_, layout) in abi_render_stages(abi, mesh_family) {
        parameters = parameters.saturating_add(layout.parameters().len() as u32);
        bytes = bytes.saturating_add(layout.argument_buffer_len() as u32);
    }
    (parameters, bytes)
}

fn abi_render_stages(
    abi: &PipelineAbi,
    mesh_family: bool,
) -> Vec<(ShaderStage, RootSignatureLayout)> {
    let stages: Vec<(ShaderStage, &RootSignatureLayout)> = if mesh_family {
        abi.mesh_stages()
    } else {
        abi.stages_for(ReflectedStage::Vertex)
            .into_iter()
            .chain(abi.stages_for(ReflectedStage::Fragment))
            .collect()
    };
    stages
        .into_iter()
        .map(|(stage, layout)| (stage, layout.clone()))
        .collect()
}

/// Flushes collected root-signature bindings for a render-family pipeline.
fn flush_render_abi(
    transport: &DirectMetalTransport,
    state: &mut DeviceState,
    abi: Option<&PipelineAbi>,
    mesh_family: bool,
    pending: &mut AbiPending,
    ops: &mut Vec<PreparedRenderOp>,
) -> Result<(), PrepareAbort> {
    let Some(abi) = abi else {
        pending.clear();
        return Ok(());
    };
    let stages = abi_render_stages(abi, mesh_family);
    // ADR-058: a translated stage whose MEMORY_SPACE t0/u0 is host-owned has
    // no pending wire binding, so the empty-pending shortcut must not skip the
    // live page-table bind.
    if pending.is_empty() && !crate::macos::memory_space_bind::declares_memory_space(&stages) {
        return Ok(());
    }
    let (buffers, bytes_ops) = flush_abi(transport, state, &stages, pending)?;
    for buffer in buffers {
        ops.push(PreparedRenderOp::ArgumentBuffer(buffer));
    }
    for (stage, slot, bytes) in bytes_ops {
        ops.push(PreparedRenderOp::StageBytes { stage, slot, bytes });
    }
    pending.clear();
    Ok(())
}

/// Flushes collected root-signature bindings for a compute pipeline.
fn flush_compute_abi(
    transport: &DirectMetalTransport,
    state: &mut DeviceState,
    abi: Option<&PipelineAbi>,
    pending: &mut AbiPending,
    ops: &mut Vec<PreparedComputeOp>,
) -> Result<(), PrepareAbort> {
    let Some(abi) = abi else {
        pending.clear();
        return Ok(());
    };
    let stages: Vec<(ShaderStage, RootSignatureLayout)> = abi
        .stages_for(ReflectedStage::Compute)
        .into_iter()
        .map(|(stage, layout)| (stage, layout.clone()))
        .collect();
    if pending.is_empty() && !crate::macos::memory_space_bind::declares_memory_space(&stages) {
        return Ok(());
    }
    let (buffers, bytes_ops) = flush_abi(transport, state, &stages, pending)?;
    for buffer in buffers {
        ops.push(PreparedComputeOp::ArgumentBuffer(buffer));
    }
    for (_stage, slot, bytes) in bytes_ops {
        ops.push(PreparedComputeOp::Bytes { slot, bytes });
    }
    pending.clear();
    Ok(())
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

/// One resolved compute pipeline: raw state and threadgroup ceiling.
struct ResolvedComputePipeline {
    raw: objc::Id,
    thread_limit: u64,
}

/// Resolves a compute handle through the legacy map or the content registry.
fn resolve_compute_pipeline(
    state: &DeviceState,
    id: u64,
) -> Result<ResolvedComputePipeline, PrepareAbort> {
    if let Some(pso) = state.pipelines.get(&id) {
        return Ok(ResolvedComputePipeline {
            raw: pso.as_raw(),
            thread_limit: pipeline_thread_limit(pso),
        });
    }
    note_unknown_site("pipeline:compute");
    let entry = resolve_content_pipeline(state, id, PipelineStage::Compute)?;
    // SAFETY: `entry` owns a live `MTLComputePipelineState`.
    let limit =
        unsafe { objc::msg_send_usize_0(entry.raw(), objc::sel(c"maxTotalThreadsPerThreadgroup")) };
    Ok(ResolvedComputePipeline {
        raw: entry.raw(),
        thread_limit: limit as u64,
    })
}

/// One resolved mesh dispatch: per-stage threadgroup sizes and grid ceiling.
///
/// `DispatchMesh` on the wire names only group counts, so the sizes come from
/// the selected pipeline: a legacy [`MeshPipeline`] carries them directly and a
/// content-hashed mesh PSO carries them in its decoded `PipelineState`.
struct ResolvedMeshPipeline {
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
    max_total_threadgroups_per_mesh_grid: usize,
}

/// A stage threadgroup size Metal accepts.
///
/// A zero dimension in a `PipelineState` blob means "unset" (the descriptor
/// default), and every `drawMeshThreadgroups` dimension must be at least one.
fn mesh_threadgroup_size(dimensions: [u32; 3]) -> MTLSize {
    MTLSize {
        width: dimensions[0].max(1) as usize,
        height: dimensions[1].max(1) as usize,
        depth: dimensions[2].max(1) as usize,
    }
}

/// Resolves a mesh dispatch handle through the legacy map or the content
/// registry.
///
/// A content-hashed mesh PSO registered by
/// [`DirectMetalTransport::register_content_hashed_mesh_pipeline`] is dispatchable
/// exactly like a legacy [`MeshPipeline`]; the per-stage threadgroup sizes come
/// from its `PipelineState`.
fn resolve_mesh_pipeline(
    state: &DeviceState,
    id: u64,
) -> Result<ResolvedMeshPipeline, PrepareAbort> {
    if let Some(mesh) = state.mesh_pipelines.get(&id) {
        return Ok(ResolvedMeshPipeline {
            threads_per_object_threadgroup: mesh.threads_per_object_threadgroup(),
            threads_per_mesh_threadgroup: mesh.threads_per_mesh_threadgroup(),
            max_total_threadgroups_per_mesh_grid: mesh.max_total_threadgroups_per_mesh_grid(),
        });
    }
    let entry = resolve_content_pipeline(state, id, PipelineStage::Mesh)?;
    let declared = entry.state();
    let grid_limit = declared.max_total_threadgroups_per_mesh_grid as usize;
    Ok(ResolvedMeshPipeline {
        threads_per_object_threadgroup: mesh_threadgroup_size(declared.object_threadgroup),
        threads_per_mesh_threadgroup: mesh_threadgroup_size(declared.mesh_threadgroup),
        max_total_threadgroups_per_mesh_grid: grid_limit,
    })
}

/// Validates and prepares a compute segment.
fn prepare_compute_segment(
    state: &mut DeviceState,
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    range: Range<usize>,
    root_plan: Option<&RootParameterPlan>,
) -> Result<Vec<PreparedComputeOp>, PrepareAbort> {
    let device = transport.device_raw();
    if state.trace.is_none() && crate::macos::trace::enabled() {
        state.trace = Some(crate::macos::trace::TraceState::default());
    }
    let mut ops = Vec::new();
    let mut abi: Option<PipelineAbi> = None;
    let mut pending = AbiPending::default();
    for index in range {
        match &commands[index] {
            CommandV3::SetComputePipeline(pipeline) => {
                let pso = resolve_compute_pipeline(state, pipeline.pipeline_id)?;
                pending.clear();
                abi = pipeline_abi(state, pipeline.pipeline_id);
                ops.push(PreparedComputeOp::Pipeline(pso.raw));
            }
            CommandV3::SetPipeline(pipeline) => {
                let pso = resolve_compute_pipeline(state, pipeline.pipeline_id)?;
                pending.clear();
                abi = pipeline_abi(state, pipeline.pipeline_id);
                ops.push(PreparedComputeOp::Pipeline(pso.raw));
            }
            CommandV3::Dispatch1d(dispatch) => {
                if dispatch.bindings.len() > wire::MAX_BINDINGS_PER_DISPATCH {
                    return Err(
                        MetalError::InvalidArg("dispatch exceeds wire binding limit").into(),
                    );
                }
                let pso = resolve_compute_pipeline(state, dispatch.pipeline_id)?;
                let threads_per_group = dispatch.threads_per_group as usize;
                if threads_per_group as u64 > pso.thread_limit {
                    return Err(MetalError::InvalidArg(
                        "dispatch threads_per_group exceeds pipeline threadgroup maximum",
                    )
                    .into());
                }
                flush_compute_abi(transport, state, abi.as_ref(), &mut pending, &mut ops)?;
                transport.encode_counters.bump_dispatch();
                trace_encoded_work(
                    state,
                    "dispatch",
                    dispatch.pipeline_id,
                    dispatch.threads,
                    1,
                    None,
                    None,
                    abi.as_ref(),
                    false,
                    None,
                );
                ops.push(PreparedComputeOp::Dispatch(PreparedV3Dispatch {
                    pipeline: pso.raw,
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
                let pso = resolve_compute_pipeline(state, dispatch.pipeline_id)?;
                let local_threads =
                    dispatch
                        .checked_threads_per_group()
                        .ok_or(MetalError::InvalidArg(
                            "3D dispatch local thread product overflows u64",
                        ))?;
                if local_threads > pso.thread_limit {
                    return Err(MetalError::InvalidArg(
                        "3D dispatch threadgroup exceeds pipeline threadgroup maximum",
                    )
                    .into());
                }
                flush_compute_abi(transport, state, abi.as_ref(), &mut pending, &mut ops)?;
                transport.encode_counters.bump_dispatch();
                trace_encoded_work(
                    state,
                    "dispatch",
                    dispatch.pipeline_id,
                    u32::try_from(local_threads).unwrap_or(u32::MAX),
                    1,
                    None,
                    None,
                    abi.as_ref(),
                    false,
                    None,
                );
                ops.push(PreparedComputeOp::Dispatch(PreparedV3Dispatch {
                    pipeline: pso.raw,
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
                let pso = resolve_compute_pipeline(state, dispatch.pipeline_id)?;
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
                if local_threads > pso.thread_limit {
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
                flush_compute_abi(transport, state, abi.as_ref(), &mut pending, &mut ops)?;
                transport.encode_counters.bump_indirect();
                trace_encoded_work(
                    state,
                    "dispatch_indirect",
                    dispatch.pipeline_id,
                    0,
                    1,
                    None,
                    None,
                    abi.as_ref(),
                    false,
                    None,
                );
                ops.push(PreparedComputeOp::DispatchIndirect {
                    pipeline: pso.raw,
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
                if let Some(active) = abi.as_ref() {
                    let slot = resolve_wire_root_argument(
                        transport,
                        active,
                        root_plan,
                        constants.slot,
                        RootArgumentSite::CONSTANTS,
                    )?;
                    if slot.kind() != RootParameterKind::Constants {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "root constants name a non-constant root parameter",
                        )
                        .into());
                    }
                    pending.bindings.push(PendingAbiBinding::Constants {
                        slot,
                        wire_slot: usize::from(constants.slot),
                        bytes: constants.bytes.clone(),
                    });
                } else {
                    ops.push(PreparedComputeOp::Bytes {
                        slot: usize::from(constants.slot),
                        bytes: constants.bytes.clone(),
                    });
                }
            }
            CommandV3::SetDescriptorTable(table) => {
                if table.bindings.is_empty()
                    || table.bindings.len() > crate::wire_v3::MAX_DESCRIPTOR_BINDINGS_V3
                {
                    return Err(MetalError::InvalidArg(
                        "descriptor table needs 1..=32 bindings",
                    )
                    .into());
                }
                if let Some(active) = abi.as_ref() {
                    // A plan entry names the translated table by its
                    // `(space, base register)`; without one the wire set index
                    // is the flattened parameter index on this path, and a
                    // signature that also declares non-table parameters first
                    // may instead send the table's ordinal. Try the index, then
                    // the ordinal, and only then refuse.
                    let set = usize::from(table.set);
                    let slot = match root_plan.and_then(|plan| plan.entry(table.set)) {
                        Some(entry) => planned_root_parameter(
                            transport,
                            active,
                            entry,
                            PlanArgumentKind::DescriptorTable,
                            "descriptor table plan names a non-table root argument",
                        )?,
                        None => {
                            transport.count_legacy_root_binding();
                            let Some(slot) = active
                                .root_parameter(set)
                                .filter(|slot| {
                                    matches!(slot.kind(), RootParameterKind::DescriptorTable(_))
                                })
                                .or_else(|| active.root_parameter_for_table(set))
                            else {
                                transport.count_abi_refusal();
                                return Err(MetalError::InvalidArg(
                                    "descriptor table names no declared root parameter",
                                )
                                .into());
                            };
                            slot
                        }
                    };
                    if !matches!(slot.kind(), RootParameterKind::DescriptorTable(_)) {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "descriptor table names a non-table root parameter",
                        )
                        .into());
                    }
                    pending.bindings.push(build_descriptor_table(
                        device,
                        state,
                        allocations,
                        table,
                        slot,
                    )?);
                } else {
                    if table.kind == DescriptorKind::Sampler {
                        return Err(MetalError::NotSupported(
                            "sampler descriptor tables are not bound by the native compute path",
                        )
                        .into());
                    }
                    ops.push(PreparedComputeOp::Bind(resolve_binding_slots(
                        state,
                        allocations,
                        &buffer_table_bindings(table)?,
                        usize::from(table.base_register),
                    )?));
                }
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
                if let Some(active) = abi.as_ref() {
                    let slot = resolve_wire_root_argument(
                        transport,
                        active,
                        root_plan,
                        view.slot,
                        RootArgumentSite::ROOT_VIEW,
                    )?;
                    if !matches!(
                        slot.kind(),
                        RootParameterKind::ConstantBufferView
                            | RootParameterKind::ShaderResourceView
                            | RootParameterKind::UnorderedAccessView
                    ) {
                        transport.count_abi_refusal();
                        // Bounded stderr: the host installs no subscriber
                        // (ADR-054/Cycle 16 diagnosis).
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static NOTES: AtomicU64 = AtomicU64::new(0);
                        if NOTES.fetch_add(1, Ordering::Relaxed) < 16 {
                            eprintln!(
                                "victory-metal-host: root view slot={} space={} register={} kind={:?}",
                                view.slot,
                                slot.space(),
                                slot.register(),
                                slot.kind(),
                            );
                        }
                        return Err(MetalError::InvalidArg(
                            "root buffer view names a non-buffer root parameter",
                        )
                        .into());
                    }
                    let address = buffer_gpu_address(buffer)
                        .checked_add(offset as u64)
                        .ok_or(MetalError::InvalidArg(
                            "root buffer view address overflows u64",
                        ))?;
                    pending.bindings.push(PendingAbiBinding::RootView {
                        slot,
                        wire_slot: usize::from(view.slot),
                        address,
                        resource: buffer,
                        usage: MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                    });
                } else {
                    ops.push(PreparedComputeOp::Bind(vec![(
                        buffer,
                        offset,
                        usize::from(view.slot),
                    )]));
                }
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
/// One explicit clear recorded inside a render pass.
#[derive(Clone, Copy)]
enum ClearSpec {
    Color {
        slot: usize,
        color: [f64; 4],
    },
    Depth {
        depth: Option<f32>,
        stencil: Option<u32>,
    },
}

/// Prepares one v3 render pass, splitting it at explicit clears.
///
/// A Metal render encoder has no clear command. An explicit
/// `ClearRenderTargetView`/`ClearDepthStencilView` recorded inside the pass is
/// therefore encoded by closing the current encoder and opening a continuation
/// pass whose load action is `Clear` for the named plane and `Load` for every
/// other attachment. The continuation replays the pass's state (pipeline,
/// viewport, bindings) but not its earlier draws, so draw order is preserved.
/// This is the "fold into the next pass load action" rule; a clear is never
/// silently dropped. Partial-rectangle clears cannot be expressed as a load
/// action and are refused explicitly.
fn prepare_render_segment(
    state: &mut DeviceState,
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    range: Range<usize>,
    pass: &BeginRenderPass,
    root_plan: Option<&RootParameterPlan>,
) -> Result<(Vec<PreparedRenderPass>, u32, u32), PrepareAbort> {
    let device = transport.device_raw();
    let mut colors = Vec::with_capacity(pass.colors.len());
    let mut color_formats = Vec::with_capacity(pass.colors.len());
    let mut extent: Option<(u32, u32)> = None;
    // ADR-053: the pass's first colour attachment is what the tracer samples.
    let mut trace_target: Option<crate::macos::trace::TraceTarget> = None;
    for color in &pass.colors {
        let (texture, info) =
            resolve_texture(state, allocations, color.allocation_id, color.generation).ok_or_else(
                || {
                    UNKNOWN_HANDLE_ID.store(
                        color.allocation_id,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    PrepareAbort::UnknownHandle
                },
            )?;
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
        if trace_target.is_none() {
            trace_target = Some(crate::macos::trace::TraceTarget {
                texture: texture as usize,
                width: info.width,
                height: info.height,
                format: match info.format {
                    PixelFormat::Bgra8Unorm => "bgra8unorm",
                    PixelFormat::Rgba8Unorm => "rgba8unorm",
                    _ => "non_colour",
                },
                swapchain: state.surfaces.values().any(|surface| surface.as_raw() == texture),
            });
        }
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
                    .ok_or_else(|| {
                        UNKNOWN_HANDLE_ID.store(
                            depth.allocation_id,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PrepareAbort::UnknownHandle
                    })?;
            if !info.format.is_depth() {
                return Err(MetalError::InvalidArg(
                    "depth attachment must be a depth format",
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

    // Group clears that are adjacent in the command stream, so two clears of
    // two attachments share one continuation pass.
    let mut groups: Vec<(usize, Vec<ClearSpec>)> = Vec::new();
    for index in range.clone() {
        match &commands[index] {
            CommandV3::ClearRenderTargetView(clear) => {
                if clear.rect.is_some() {
                    return Err(MetalError::InvalidArg(
                        "partial-rect clears are not supported; clear the whole attachment",
                    )
                    .into());
                }
                if usize::from(clear.slot) >= colors.len() {
                    return Err(MetalError::InvalidArg(
                        "clear slot names an attachment the pass does not have",
                    )
                    .into());
                }
                let spec = ClearSpec::Color {
                    slot: usize::from(clear.slot),
                    color: [
                        f64::from(clear.color[0]),
                        f64::from(clear.color[1]),
                        f64::from(clear.color[2]),
                        f64::from(clear.color[3]),
                    ],
                };
                push_clear_group(&mut groups, index, spec);
            }
            CommandV3::ClearDepthStencilView(clear) => {
                if clear.rect.is_some() {
                    return Err(MetalError::InvalidArg(
                        "partial-rect clears are not supported; clear the whole attachment",
                    )
                    .into());
                }
                let raw_depth = depth.as_ref().ok_or(MetalError::InvalidArg(
                    "depth/stencil clear requires a depth attachment in the pass",
                ))?;
                if clear.stencil.is_some() && !raw_depth.has_stencil {
                    return Err(MetalError::InvalidArg(
                        "stencil clear requires a depth-stencil attachment format",
                    )
                    .into());
                }
                let spec = ClearSpec::Depth {
                    depth: clear.depth,
                    stencil: clear.stencil,
                };
                push_clear_group(&mut groups, index, spec);
            }
            _ => {}
        }
    }
    // A group is only a boundary if the clear is the first command of the
    // group after a run of non-clear commands; adjacent groups are merged by
    // `push_clear_group`.
    let groups = merge_clear_groups(commands, range.clone(), groups);
    let folded_clears: u64 = groups.iter().map(|(_, specs)| specs.len() as u64).sum();
    for _ in 0..folded_clears {
        transport.encode_counters.bump_clear();
    }

    let mut passes = Vec::with_capacity(groups.len() + 1);
    let mut cursor = range.start;
    let mut previous_clears: &[ClearSpec] = &[];
    let mut first = true;
    for (clear_at, clears) in &groups {
        let ops = build_render_ops(
            device,
            state,
            transport,
            allocations,
            commands,
            range.start,
            cursor..*clear_at,
            &color_formats,
            depth_format,
            width,
            height,
            trace_target,
            root_plan,
        )?;
        let (descriptor_colors, descriptor_depth) = if first {
            (colors.clone(), depth)
        } else {
            (
                continuation_colors(&colors, previous_clears, false),
                continuation_depth(depth.as_ref(), previous_clears, false),
            )
        };
        let descriptor = build_mrt_render_pass_descriptor(
            &descriptor_colors,
            descriptor_depth.as_ref(),
        )?;
        passes.push(PreparedRenderPass {
            descriptor,
            draws: draw_op_count(&ops),
            ops,
            trace_target,
        });
        cursor = *clear_at;
        previous_clears = clears;
        first = false;
    }
    let final_ops = build_render_ops(
        device,
        state,
        transport,
        allocations,
        commands,
        range.start,
        cursor..range.end,
        &color_formats,
        depth_format,
        width,
        height,
        trace_target,
        root_plan,
    )?;
    let (final_colors, final_depth) = if first {
        (colors, depth)
    } else {
        (
            continuation_colors(&colors, previous_clears, true),
            continuation_depth(depth.as_ref(), previous_clears, true),
        )
    };
    let final_descriptor =
        build_mrt_render_pass_descriptor(&final_colors, final_depth.as_ref())?;
    passes.push(PreparedRenderPass {
        descriptor: final_descriptor,
        draws: draw_op_count(&final_ops),
        ops: final_ops,
        trace_target,
    });
    for _ in &passes {
        transport.encode_counters.bump_render_pass();
    }
    Ok((passes, width, height))
}
/// Records one clear at its command index, merging adjacent clears.
fn push_clear_group(groups: &mut Vec<(usize, Vec<ClearSpec>)>, index: usize, spec: ClearSpec) {
    match groups.last_mut() {
        Some((last_index, specs)) if *last_index == index => specs.push(spec),
        _ => groups.push((index, vec![spec])),
    }
}

/// Merges clear groups that have no non-clear command between them.
fn merge_clear_groups(
    commands: &[CommandV3],
    range: Range<usize>,
    groups: Vec<(usize, Vec<ClearSpec>)>,
) -> Vec<(usize, Vec<ClearSpec>)> {
    let mut merged: Vec<(usize, Vec<ClearSpec>)> = Vec::new();
    for (index, specs) in groups {
        let adjacent = merged.last().is_some_and(|(previous, _)| {
            (*previous..index).all(|position| {
                position < range.start
                    || position >= range.end
                    || matches!(
                        commands[position],
                        CommandV3::ClearRenderTargetView(_)
                            | CommandV3::ClearDepthStencilView(_)
                    )
            })
        });
        if adjacent {
            if let Some((_, previous_specs)) = merged.last_mut() {
                previous_specs.extend(specs);
            }
        } else {
            merged.push((index, specs));
        }
    }
    merged
}

/// Builds the colour attachment list of one sub-pass.
///
/// Sub-pass 0 keeps the pass's recorded load/store actions; a continuation
/// loads every attachment, applies the clears that start it, and stores so the
/// next sub-pass can load it (the final sub-pass keeps the recorded store
/// actions).
fn continuation_colors(
    colors: &[RawColorAttachment],
    clears: &[ClearSpec],
    final_pass: bool,
) -> Vec<RawColorAttachment> {
    let mut out = colors.to_vec();
    for clear in clears {
        if let ClearSpec::Color { slot, color } = clear {
            if let Some(attachment) = out.get_mut(*slot) {
                attachment.load = LoadAction::Clear;
                attachment.clear_color = *color;
            }
        }
    }
    if !final_pass {
        for attachment in &mut out {
            if attachment.load != LoadAction::Clear {
                attachment.load = LoadAction::Load;
            }
            attachment.store = StoreAction::Store;
        }
    } else {
        for attachment in &mut out {
            if attachment.load != LoadAction::Clear {
                attachment.load = LoadAction::Load;
            }
        }
    }
    out
}

/// Builds the optional depth attachment of one sub-pass.
fn continuation_depth(
    depth: Option<&RawDepthAttachment>,
    clears: &[ClearSpec],
    final_pass: bool,
) -> Option<RawDepthAttachment> {
    let mut out = *depth?;
    for clear in clears {
        if let ClearSpec::Depth { depth, stencil } = clear {
            if let Some(value) = depth {
                out.depth_load = LoadAction::Clear;
                out.clear_depth = f64::from(*value);
            }
            if let Some(value) = stencil {
                out.stencil_load = LoadAction::Clear;
                out.clear_stencil = *value;
            }
        }
    }
    if !final_pass {
        if out.depth_load != LoadAction::Clear {
            out.depth_load = LoadAction::Load;
        }
        if out.stencil_load != LoadAction::Clear {
            out.stencil_load = LoadAction::Load;
        }
        out.depth_store = StoreAction::Store;
        out.stencil_store = StoreAction::Store;
    } else {
        if out.depth_load != LoadAction::Clear {
            out.depth_load = LoadAction::Load;
        }
        if out.stencil_load != LoadAction::Clear {
            out.stencil_load = LoadAction::Load;
        }
    }
    Some(out)
}

/// Tracks the last value applied to one encoder-state slot.
///
/// A repeated wire state command whose value is unchanged must not emit a
/// redundant Metal call. This is the whole dedup rule for encoder state; keeping
/// it in one pure type makes the idempotency unit-testable without a GPU.
#[derive(Debug)]
struct StateDedup<T> {
    last: Option<T>,
}

impl<T> Default for StateDedup<T> {
    fn default() -> Self {
        Self { last: None }
    }
}

impl<T: Copy + PartialEq> StateDedup<T> {
    /// Records `next` and reports whether it differs from the last value.
    fn should_emit(&mut self, next: T) -> bool {
        if self.last == Some(next) {
            false
        } else {
            self.last = Some(next);
            true
        }
    }
}

/// Builds the replayed ops of one render sub-pass.
///
/// `state_from` is the first command of the pass; commands in
/// `state_from..emit.start` are replayed as state only (draws and barriers are
/// skipped) so a sub-pass that follows an explicit clear starts from the same
/// pipeline, viewport, buffers and bindings as the pass did. Clear commands are
/// never replayed: each one is the load action of the sub-pass it starts.
#[allow(clippy::too_many_arguments)]
fn build_render_ops(
    device: objc::Id,
    state: &mut DeviceState,
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    state_from: usize,
    emit: Range<usize>,
    color_formats: &[PixelFormat],
    depth_format: Option<PixelFormat>,
    width: u32,
    height: u32,
    trace_target: Option<crate::macos::trace::TraceTarget>,
    root_plan: Option<&RootParameterPlan>,
) -> Result<Vec<PreparedRenderOp>, PrepareAbort> {
    let mut ops = Vec::new();
    let mut mesh_pipeline: Option<u64> = None;
    let mut current_pipeline = false;
    // ADR-053: the id of the selected pipeline, for the trace line.
    let mut current_pipeline_id: u64 = 0;
    let mut index_binding: Option<(objc::Id, usize, IndexType, u64)> = None;
    let mut topology = PrimitiveType::Triangle;
    // ADR-053: the viewport/scissor a traced draw ran with.
    let mut viewport_rect: Option<[u32; 4]> = None;
    let mut scissor_rect: Option<[u32; 4]> = None;
    if state.trace.is_none() && crate::macos::trace::enabled() {
        state.trace = Some(crate::macos::trace::TraceState::default());
    }
    // Root-signature state of the selected pipeline and its collected bindings.
    let mut abi: Option<PipelineAbi> = None;
    let mut pending = AbiPending::default();
    // Last-applied encoder state, so a repeated Set command emits no redundant
    // Metal call inside one sub-pass.
    let mut last_rasterizer: StateDedup<RasterizerState> = StateDedup::default();
    let mut last_stencil_ref: StateDedup<u8> = StateDedup::default();
    let mut last_blend_color: StateDedup<[f32; 4]> = StateDedup::default();
    let mut last_depth_state: StateDedup<objc::Id> = StateDedup::default();
    // D3D12 depth/stencil state is independent of the PSO. Once a wire state is
    // set, pipeline selection must not overwrite it with the pipeline default.
    let mut depth_overridden = false;
    // Blend is baked into the pipeline state; a wire blend state is checked
    // against the selected pipeline so a mismatch is an explicit error.
    let mut current_blend: Option<crate::wire_v3::BlendState> = None;
    let mut selected_blend: Option<Option<crate::wire_v3::BlendState>> = None;
    for index in state_from..emit.end {
        if index < emit.start && is_work_command(&commands[index]) {
            continue;
        }
        match &commands[index] {
            CommandV3::BeginRenderPass(_) | CommandV3::EndRenderPass => continue,
            CommandV3::ClearRenderTargetView(_) | CommandV3::ClearDepthStencilView(_) => continue,
            CommandV3::SetGraphicsPipeline(pipeline) => {
                let mut depth_state = None;
                let state_raw;
                if let Some(graphics) = state.render_pipelines.get(&pipeline.pipeline_id) {
                    if graphics.color_formats() != color_formats {
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
                    check_blend_state(
                        current_blend.as_ref(),
                        PipelineBlend::Legacy(graphics.blend_state()),
                    )?;
                    selected_blend = Some(graphics.blend_state().cloned());
                    state_raw = graphics.state_raw();
                    if !depth_overridden {
                        depth_state = graphics.depth_state_raw().map(DepthState::Borrowed);
                    }
                } else {
                    // Content-hashed registry pipeline: the attachment state is
                    // part of the content key and must match this pass exactly.
                    let entry =
                        resolve_content_pipeline(state, pipeline.pipeline_id, PipelineStage::Graphics)?;
                    check_pipeline_attachments(entry, color_formats, depth_format)?;
                    check_blend_state(current_blend.as_ref(), PipelineBlend::ContentHashed)?;
                    selected_blend = Some(None);
                    state_raw = entry.raw();
                    if !depth_overridden && entry.state().depth_format.is_some() {
                        depth_state = Some(DepthState::Owned(new_less_depth_state(transport)?));
                    }
                }
                pending.clear();
                abi = pipeline_abi(state, pipeline.pipeline_id);
                ops.push(PreparedRenderOp::Pipeline {
                    state: state_raw,
                    depth_state,
                });
                current_pipeline = true;
                mesh_pipeline = None;
                current_pipeline_id = pipeline.pipeline_id;
            }
            CommandV3::SetPipeline(pipeline) => {
                let mut depth_state = None;
                let state_raw;
                if let Some(graphics) = state.render_pipelines.get(&pipeline.pipeline_id) {
                    if graphics.color_formats() != color_formats {
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
                    check_blend_state(
                        current_blend.as_ref(),
                        PipelineBlend::Legacy(graphics.blend_state()),
                    )?;
                    selected_blend = Some(graphics.blend_state().cloned());
                    state_raw = graphics.state_raw();
                    if !depth_overridden {
                        depth_state = graphics.depth_state_raw().map(DepthState::Borrowed);
                    }
                } else {
                    // Content-hashed registry pipeline: the attachment state is
                    // part of the content key and must match this pass exactly.
                    let entry =
                        resolve_content_pipeline(state, pipeline.pipeline_id, PipelineStage::Graphics)?;
                    check_pipeline_attachments(entry, color_formats, depth_format)?;
                    check_blend_state(current_blend.as_ref(), PipelineBlend::ContentHashed)?;
                    selected_blend = Some(None);
                    state_raw = entry.raw();
                    if !depth_overridden && entry.state().depth_format.is_some() {
                        depth_state = Some(DepthState::Owned(new_less_depth_state(transport)?));
                    }
                }
                pending.clear();
                abi = pipeline_abi(state, pipeline.pipeline_id);
                ops.push(PreparedRenderOp::Pipeline {
                    state: state_raw,
                    depth_state,
                });
                current_pipeline = true;
                mesh_pipeline = None;
                current_pipeline_id = pipeline.pipeline_id;
            }
            CommandV3::SetMeshPipeline(pipeline) => {
                let mut depth_state = None;
                let state_raw;
                if let Some(mesh) = state.mesh_pipelines.get(&pipeline.pipeline_id) {
                    if mesh.color_formats() != color_formats {
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
                    check_blend_state(current_blend.as_ref(), PipelineBlend::ContentHashed)?;
                    selected_blend = Some(None);
                    state_raw = mesh.state_raw();
                    if !depth_overridden {
                        depth_state = mesh.depth_state_raw().map(DepthState::Borrowed);
                    }
                } else {
                    let entry =
                        resolve_content_pipeline(state, pipeline.pipeline_id, PipelineStage::Mesh)?;
                    check_pipeline_attachments(entry, color_formats, depth_format)?;
                    check_blend_state(current_blend.as_ref(), PipelineBlend::ContentHashed)?;
                    selected_blend = Some(None);
                    state_raw = entry.raw();
                    if !depth_overridden && entry.state().depth_format.is_some() {
                        depth_state = Some(DepthState::Owned(new_less_depth_state(transport)?));
                    }
                }
                pending.clear();
                abi = pipeline_abi(state, pipeline.pipeline_id);
                ops.push(PreparedRenderOp::Pipeline {
                    state: state_raw,
                    depth_state,
                });
                current_pipeline = true;
                mesh_pipeline = Some(pipeline.pipeline_id);
                current_pipeline_id = pipeline.pipeline_id;
            }
            CommandV3::SetBlendState(blend) => {
                current_blend = Some(blend.clone());
                // The pipeline is validated when it is selected; if one is
                // already bound, check the new state against it now.
                if selected_blend.is_some() {
                    check_blend_state(
                        Some(blend),
                        selected_blend_to_pipeline(selected_blend.as_ref()),
                    )?;
                }
            }
            CommandV3::SetDepthStencilState(depth) => {
                let raw = state
                    .depth_states
                    .get_or_create(transport, depth)?;
                depth_overridden = true;
                if last_depth_state.should_emit(raw) {
                    ops.push(PreparedRenderOp::DepthStencil(raw));
                }
            }
            CommandV3::SetRasterizerState(raster) => {
                if last_rasterizer.should_emit(*raster) {
                    ops.push(PreparedRenderOp::Rasterizer(*raster));
                }
            }
            CommandV3::SetStencilRef(reference) => {
                if last_stencil_ref.should_emit(reference.value) {
                    ops.push(PreparedRenderOp::StencilRef(reference.value));
                }
            }
            CommandV3::SetBlendFactor(factor) => {
                if last_blend_color.should_emit(factor.value) {
                    ops.push(PreparedRenderOp::BlendColor(factor.value));
                }
            }
            CommandV3::SetViewport(viewport) => {
                let validated = validate_viewport(viewport, width, height)?;
                viewport_rect = Some([
                    validated.origin_x as u32,
                    validated.origin_y as u32,
                    validated.width as u32,
                    validated.height as u32,
                ]);
                ops.push(PreparedRenderOp::Viewport(validated));
            }
            CommandV3::SetScissor(scissor) => {
                let validated = validate_scissor(scissor, width, height)?;
                scissor_rect = Some([
                    validated.x as u32,
                    validated.y as u32,
                    validated.width as u32,
                    validated.height as u32,
                ]);
                ops.push(PreparedRenderOp::Scissor(validated));
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
                if let Some(active) = abi.as_ref() {
                    // Converted ABI: resolve the wire slot through the
                    // translated root signature (by the plan's register
                    // identity when the guest sent one), then write the bytes
                    // at the reflected constant slot (inline in the top-level
                    // argument buffer, or a dedicated setBytes index).
                    let slot = resolve_wire_root_argument(
                        transport,
                        active,
                        root_plan,
                        constants.slot,
                        RootArgumentSite::CONSTANTS,
                    )?;
                    if slot.kind() != RootParameterKind::Constants {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "root constants name a non-constant root parameter",
                        )
                        .into());
                    }
                    pending.bindings.push(PendingAbiBinding::Constants {
                        slot,
                        wire_slot: usize::from(constants.slot),
                        bytes: constants.bytes.clone(),
                    });
                } else {
                    ops.push(PreparedRenderOp::VertexBytes {
                        slot: usize::from(constants.slot),
                        bytes: constants.bytes.clone(),
                    });
                }
            }
            CommandV3::SetDescriptorTable(table) => {
                if table.bindings.is_empty()
                    || table.bindings.len() > crate::wire_v3::MAX_DESCRIPTOR_BINDINGS_V3
                {
                    return Err(MetalError::InvalidArg(
                        "descriptor table needs 1..=32 bindings",
                    )
                    .into());
                }
                if let Some(active) = abi.as_ref() {
                    // A plan entry names the translated table by its
                    // `(space, base register)`; without one the wire set index
                    // is the flattened parameter index on this path, and a
                    // signature that also declares non-table parameters first
                    // may instead send the table's ordinal. Try the index, then
                    // the ordinal, and only then refuse.
                    let set = usize::from(table.set);
                    let slot = match root_plan.and_then(|plan| plan.entry(table.set)) {
                        Some(entry) => planned_root_parameter(
                            transport,
                            active,
                            entry,
                            PlanArgumentKind::DescriptorTable,
                            "descriptor table plan names a non-table root argument",
                        )?,
                        None => {
                            transport.count_legacy_root_binding();
                            let Some(slot) = active
                                .root_parameter(set)
                                .filter(|slot| {
                                    matches!(slot.kind(), RootParameterKind::DescriptorTable(_))
                                })
                                .or_else(|| active.root_parameter_for_table(set))
                            else {
                                transport.count_abi_refusal();
                                return Err(MetalError::InvalidArg(
                                    "descriptor table names no declared root parameter",
                                )
                                .into());
                            };
                            slot
                        }
                    };
                    if !matches!(slot.kind(), RootParameterKind::DescriptorTable(_)) {
                        transport.count_abi_refusal();
                        return Err(MetalError::InvalidArg(
                            "descriptor table names a non-table root parameter",
                        )
                        .into());
                    }
                    pending.bindings.push(build_descriptor_table(
                        device,
                        state,
                        allocations,
                        table,
                        slot,
                    )?);
                } else {
                    if table.kind == DescriptorKind::Sampler {
                        return Err(MetalError::NotSupported(
                            "sampler descriptor tables are not bound by the native graphics path",
                        )
                        .into());
                    }
                    ops.push(PreparedRenderOp::VertexBuffers(resolve_binding_slots(
                        state,
                        allocations,
                        &buffer_table_bindings(table)?,
                        usize::from(table.base_register),
                    )?));
                }
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
                if let Some(active) = abi.as_ref() {
                    let slot = resolve_wire_root_argument(
                        transport,
                        active,
                        root_plan,
                        view.slot,
                        RootArgumentSite::ROOT_VIEW,
                    )?;
                    if !matches!(
                        slot.kind(),
                        RootParameterKind::ConstantBufferView
                            | RootParameterKind::ShaderResourceView
                            | RootParameterKind::UnorderedAccessView
                    ) {
                        transport.count_abi_refusal();
                        // Bounded stderr: the host installs no subscriber
                        // (ADR-054/Cycle 16 diagnosis).
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static NOTES: AtomicU64 = AtomicU64::new(0);
                        if NOTES.fetch_add(1, Ordering::Relaxed) < 16 {
                            eprintln!(
                                "victory-metal-host: root view slot={} space={} register={} kind={:?}",
                                view.slot,
                                slot.space(),
                                slot.register(),
                                slot.kind(),
                            );
                        }
                        return Err(MetalError::InvalidArg(
                            "root buffer view names a non-buffer root parameter",
                        )
                        .into());
                    }
                    let address = buffer_gpu_address(buffer)
                        .checked_add(offset as u64)
                        .ok_or(MetalError::InvalidArg(
                            "root buffer view address overflows u64",
                        ))?;
                    pending.bindings.push(PendingAbiBinding::RootView {
                        slot,
                        wire_slot: usize::from(view.slot),
                        address,
                        resource: buffer,
                        usage: MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                    });
                } else {
                    ops.push(PreparedRenderOp::VertexBuffers(vec![(
                        buffer,
                        offset,
                        usize::from(view.slot),
                    )]));
                }
            }
            CommandV3::ResourceBarrier(barrier) => {
                if resolve_barrier_entries(state, allocations, &barrier.entries).is_none() {
                    return Err(PrepareAbort::UnknownHandle);
                }
                ops.push(PreparedRenderOp::Barrier);
            }
            CommandV3::DispatchMesh(dispatch) => {
                let mesh = resolve_mesh_pipeline(state, dispatch.pipeline_id)?;
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
                let limit = mesh.max_total_threadgroups_per_mesh_grid;
                if limit != 0 && group_counts > limit as u64 {
                    return Err(MetalError::InvalidArg(
                        "mesh dispatch exceeds maxTotalThreadgroupsPerMeshGrid",
                    )
                    .into());
                }
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_mesh_dispatch();
                trace_encoded_work(
                    state,
                    "mesh_dispatch",
                    current_pipeline_id,
                    dispatch.group_counts[0],
                    1,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
                ops.push(PreparedRenderOp::DispatchMesh {
                    group_counts: MTLSize {
                        width: dispatch.group_counts[0] as usize,
                        height: dispatch.group_counts[1] as usize,
                        depth: dispatch.group_counts[2] as usize,
                    },
                    threads_per_object_threadgroup: mesh.threads_per_object_threadgroup,
                    threads_per_mesh_threadgroup: mesh.threads_per_mesh_threadgroup,
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_draw();
                trace_encoded_work(
                    state,
                    "draw",
                    current_pipeline_id,
                    draw.vertex_count,
                    draw.instance_count,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_draw();
                trace_encoded_work(
                    state,
                    "draw",
                    current_pipeline_id,
                    draw.vertex_count_per_instance,
                    draw.instance_count,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_draw_indexed();
                trace_encoded_work(
                    state,
                    "draw_indexed",
                    current_pipeline_id,
                    draw.index_count,
                    draw.instance_count,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_draw_indexed();
                trace_encoded_work(
                    state,
                    "draw_indexed",
                    current_pipeline_id,
                    draw.index_count_per_instance,
                    draw.instance_count,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_indirect();
                trace_encoded_work(
                    state,
                    "draw_indirect",
                    current_pipeline_id,
                    0,
                    1,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_indirect();
                trace_encoded_work(
                    state,
                    "draw_indexed_indirect",
                    current_pipeline_id,
                    0,
                    1,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
                ops.push(PreparedRenderOp::DrawIndexedIndirect {
                    primitive: topology,
                    index_type,
                    index_buffer,
                    index_offset,
                    arguments,
                });
            }
            CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
                let mesh = resolve_mesh_pipeline(state, dispatch.pipeline_id)?;
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
                flush_render_abi(
                    transport,
                    state,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    &mut pending,
                    &mut ops,
                )?;
                transport.encode_counters.bump_indirect();
                trace_encoded_work(
                    state,
                    "mesh_dispatch_indirect",
                    current_pipeline_id,
                    0,
                    1,
                    viewport_rect,
                    scissor_rect,
                    abi.as_ref(),
                    mesh_pipeline.is_some(),
                    trace_target,
                );
                ops.push(PreparedRenderOp::DispatchMeshIndirect {
                    threads_per_object_threadgroup: mesh.threads_per_object_threadgroup,
                    threads_per_mesh_threadgroup: mesh.threads_per_mesh_threadgroup,
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
    // A sub-pass can end on state that no draw consumed (for example right
    // before an explicit clear). Emit those bindings so the encoder is in the
    // same state the pass had reached.
    flush_render_abi(
        transport,
        state,
        abi.as_ref(),
        mesh_pipeline.is_some(),
        &mut pending,
        &mut ops,
    )?;
    Ok(ops)
}

/// Whether a command does GPU work rather than only set state.
///
/// A sub-pass that follows an explicit clear replays the pass's state but must
/// not repeat earlier draws, mesh dispatches or barriers.
fn is_work_command(command: &CommandV3) -> bool {
    matches!(
        command,
        CommandV3::Draw(_)
            | CommandV3::DrawIndexed(_)
            | CommandV3::DrawInstanced(_)
            | CommandV3::DrawIndexedInstanced(_)
            | CommandV3::ExecuteIndirectDraw(_)
            | CommandV3::ExecuteIndirectDrawIndexed(_)
            | CommandV3::ExecuteIndirectDispatchMesh(_)
            | CommandV3::DispatchMesh(_)
            | CommandV3::ResourceBarrier(_)
    )
}

/// What a selected pipeline can say about its baked blend state.
#[derive(Clone, Copy)]
enum PipelineBlend<'a> {
    /// A legacy `GraphicsPipeline`, which records the blend state it was
    /// compiled with (`None` when it was compiled without one).
    Legacy(Option<&'a crate::wire_v3::BlendState>),
    /// A content-hashed registry pipeline; the registry compiles it from the
    /// guest `PipelineState` blob, which carries the blend/depth/stencil and
    /// rasterizer state and is part of the content key.
    ContentHashed,
}

/// Reconstructs the pipeline-blend view of a selection recorded as the owned
/// state a legacy pipeline declared (`Some(None)` for content-hashed).
fn selected_blend_to_pipeline(
    selected: Option<&Option<crate::wire_v3::BlendState>>,
) -> PipelineBlend<'_> {
    match selected {
        Some(Some(blend)) => PipelineBlend::Legacy(Some(blend)),
        Some(None) => PipelineBlend::ContentHashed,
        None => PipelineBlend::ContentHashed,
    }
}

/// Refuses a wire blend state the selected pipeline cannot represent.
///
/// Metal bakes per-attachment blending into the pipeline, so a blend state that
/// differs from the one the pipeline was compiled with cannot be applied at
/// draw time. An all-disabled state matches every pipeline (the default) and is
/// always accepted.
fn check_blend_state(
    current: Option<&crate::wire_v3::BlendState>,
    pipeline: PipelineBlend<'_>,
) -> Result<(), MetalError> {
    let Some(current) = current else {
        return Ok(());
    };
    if !current.alpha_to_coverage && current.attachments.iter().all(|entry| !entry.enable) {
        return Ok(());
    }
    match pipeline {
        PipelineBlend::Legacy(Some(baked)) => {
            if *baked == *current {
                Ok(())
            } else {
                Err(MetalError::InvalidArg(
                    "wire blend state does not match the pipeline's compiled blend state",
                ))
            }
        }
        PipelineBlend::Legacy(None) => Err(MetalError::InvalidArg(
            "wire blend state needs a pipeline compiled with the same blend state",
        )),
        // Cycle 07 ADR-029: a content-hashed pipeline is compiled from the
        // guest `PipelineState` blob, and that blob is part of the BLAKE3 key.
        // The blend/depth/stencil/rasterizer state is therefore baked into the
        // pipeline object itself, so a draw-time wire state is accepted here:
        // the registered pipeline is the authority, and a mismatch shows up as
        // wrong pixels in the proof test instead of a blanket refusal.
        PipelineBlend::ContentHashed => Ok(()),
    }
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
            CommandV3::CopyBufferToTexture(copy) => {
                let mip = u32::from(copy.mip_level);
                let slice = u32::from(copy.slice);
                let (texture, info) = resolve_texture(
                    state,
                    allocations,
                    copy.destination_allocation_id,
                    copy.destination_generation,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                if mip >= info.mip_levels || slice >= info.slices {
                    return Err(MetalError::InvalidArg(
                        "buffer-to-texture copy mip level or slice is out of range",
                    )
                    .into());
                }
                if copy.extent[2] != 1 {
                    return Err(MetalError::InvalidArg(
                        "buffer-to-texture copy supports one depth slice per call",
                    )
                    .into());
                }
                let level_width = info.width >> mip.min(31);
                let level_height = info.height >> mip.min(31);
                if copy.origin[0]
                    .checked_add(copy.extent[0])
                    .is_none_or(|end| end > level_width)
                    || copy.origin[1]
                        .checked_add(copy.extent[1])
                        .is_none_or(|end| end > level_height)
                {
                    return Err(MetalError::InvalidArg(
                        "buffer-to-texture copy exceeds the destination extent",
                    )
                    .into());
                }
                let bpp = u64::from(info.format.bytes_per_texel());
                let row_bytes = u64::from(copy.bytes_per_row);
                if row_bytes < u64::from(copy.extent[0]) * bpp {
                    return Err(MetalError::InvalidArg(
                        "copy source row pitch is smaller than one destination row",
                    )
                    .into());
                }
                let rows = u64::from(copy.extent[1]);
                let needed = row_bytes
                    .checked_mul(rows)
                    .ok_or(MetalError::InvalidArg("copy source range overflows"))?;
                let (buffer, offset, _) = resolve_buffer_range(
                    state,
                    allocations,
                    copy.allocation_id,
                    copy.generation,
                    copy.offset,
                    needed,
                )
                .ok_or(PrepareAbort::UnknownHandle)?;
                ops.push(PreparedBlitOp::CopyBufferToTexture(
                    blit::BufferTextureCopy {
                        buffer,
                        source_offset: offset,
                        bytes_per_row: copy.bytes_per_row as usize,
                        bytes_per_image: copy.bytes_per_image as usize,
                        destination: blit::TextureSubresourceRef {
                            texture,
                            slice: slice as usize,
                            level: mip as usize,
                        },
                        destination_origin: [
                            copy.origin[0] as usize,
                            copy.origin[1] as usize,
                            copy.origin[2] as usize,
                        ],
                        size: [
                            copy.extent[0] as usize,
                            copy.extent[1] as usize,
                            copy.extent[2] as usize,
                        ],
                    },
                ));
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

/// Compares a content-hashed pipeline's declared attachment state with a pass.
///
/// A mismatch is an explicit error: the registry never rebuilds the pipeline
/// with a different entry point or attachment layout to make a submission fit.
fn check_pipeline_attachments(
    entry: &crate::macos::registry::PipelineEntry,
    color_formats: &[PixelFormat],
    depth_format: Option<PixelFormat>,
) -> Result<(), PrepareAbort> {
    let declared = entry.state();
    if declared.color_formats.len() != color_formats.len()
        || declared
            .color_formats
            .iter()
            .zip(color_formats)
            .any(|(declared, pass)| declared.mtl_pixel_format() != pass.mtl_pixel_format())
    {
        return Err(MetalError::InvalidArg(
            "content-hashed pipeline colour formats do not match the render pass",
        )
        .into());
    }
    let declared_depth = declared
        .depth_format
        .map(TextureFormat::mtl_pixel_format);
    if declared_depth != depth_format.map(PixelFormat::mtl_pixel_format) {
        return Err(MetalError::InvalidArg(
            "content-hashed pipeline depth format does not match the render pass",
        )
        .into());
    }
    Ok(())
}

/// Resolves one wire handle through the content-hashed pipeline registry.
fn resolve_content_pipeline(
    state: &DeviceState,
    id: u64,
    expected: PipelineStage,
) -> Result<&crate::macos::registry::PipelineEntry, PrepareAbort> {
    note_unknown_site(match expected {
        PipelineStage::Graphics => "pipeline:graphics",
        PipelineStage::Compute => "pipeline:compute",
        PipelineStage::Mesh => "pipeline:mesh",
    });
    let pipeline = state
        .content_pipelines
        .get(&id)
        .ok_or(PrepareAbort::UnknownHandle)?;
    let registry = state
        .pipeline_registry
        .as_ref()
        .ok_or(PrepareAbort::UnknownHandle)?;
    let entry = registry.get(*pipeline).ok_or(PrepareAbort::UnknownHandle)?;
    if entry.stage() != expected {
        return Err(MetalError::InvalidArg(
            "pipeline handle names a different pipeline stage",
        )
        .into());
    }
    Ok(entry)
}

/// Extracts the root-signature ABI registered for `id`, if any.
fn pipeline_abi(state: &DeviceState, id: u64) -> Option<PipelineAbi> {
    state.abis.get(&id).cloned()
}

/// Borrows the device's content-hashed pipeline registry, creating it on first
/// use.
fn content_pipeline_registry<'a>(
    transport: &DirectMetalTransport,
    state: &'a mut DeviceState,
) -> Result<&'a mut PipelineRegistry, MetalError> {
    if state.pipeline_registry.is_none() {
        state.pipeline_registry = Some(PipelineRegistry::new(transport).map_err(registry_error)?);
    }
    state
        .pipeline_registry
        .as_mut()
        .ok_or(MetalError::InternalError(
            "pipeline registry was not created".to_string(),
        ))
}

/// Maps a host-registry failure to the transport error surface.
///
/// A Metal failure created inside the registry keeps its original
/// classification; every validation/cache failure becomes an internal error
/// carrying the registry's message.
fn registry_error(error: RegistryError) -> MetalError {
    match error {
        RegistryError::Metal(inner) => inner,
        other => MetalError::InternalError(other.to_string()),
    }
}

/// Validates every handle and bound and builds the executable segment plan.
fn prepare_v3(
    state: &mut DeviceState,
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    commands: &[CommandV3],
    plan: &[crate::macos::render::Segment],
    root_plan: Option<&RootParameterPlan>,
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
                prepare_compute_segment(
                    state,
                    transport,
                    allocations,
                    commands,
                    range.clone(),
                    root_plan,
                )?,
            )),
            SegmentKind::Render { pass_index, range } => {
                let pass = passes.get(*pass_index).ok_or(MetalError::InternalError(
                    "render segment names a missing BeginRenderPass".to_string(),
                ))?;
                let (passes, width, height) = prepare_render_segment(
                    state,
                    transport,
                    allocations,
                    commands,
                    range.clone(),
                    pass,
                    root_plan,
                )?;
                prepared.push(PreparedSegment::Render {
                    passes,
                    width,
                    height,
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

    /// Registers an ADR-013 registry texture under `id`.
    ///
    /// The texture is created and owned by the persistent [`TextureRegistry`],
    /// then aliased into the `textures` map [`resolve_texture`] consults, so a
    /// `BeginRenderPass` colour/depth attachment that names the handle resolves
    /// to the same object. The `registry_textures` alias lets
    /// [`resolve_registered_texture`] resolve SRV/UAV descriptor-table entries
    /// for the same handle.
    pub fn register_registry_texture(
        &self,
        id: u64,
        desc: TextureDesc,
    ) -> Result<(), MetalError> {
        if id == 0 {
            return Err(MetalError::InvalidArg(
                "registry texture handle must be nonzero",
            ));
        }
        let mut state = self.lock_state()?;
        if state.texture_registry.is_none() {
            state.texture_registry =
                Some(TextureRegistry::new(self).map_err(registry_error)?);
        }
        let raw = {
            let registry = state
                .texture_registry
                .as_mut()
                .ok_or(MetalError::InternalError(
                    "texture registry was not created".to_string(),
                ))?;
            match registry.register(id, desc.clone()) {
                Ok(()) => {}
                // A guest re-registers a handle after a reconnect or a
                // resource re-create. The same descriptor is the same texture:
                // accept it and republish the aliases rather than refusing a
                // handle that already works. A different descriptor is a real
                // conflict and stays named.
                Err(RegistryError::DuplicateTexture(_)) => {
                    let existing = registry.get(id).ok_or(MetalError::InternalError(
                        "texture registry reported a duplicate for an absent handle".to_string(),
                    ))?;
                    if existing.desc() != &desc {
                        return Err(MetalError::InternalError(format!(
                            "texture handle {id:#x} is already registered with a different descriptor"
                        )));
                    }
                }
                Err(error) => return Err(registry_error(error)),
            }
            registry
                .raw(id)
                .ok_or(MetalError::InternalError(
                    "registered texture is not resolvable".to_string(),
                ))?
        };
        // SAFETY: `raw` is a live `MTLTexture` owned by the registry; `retain`
        // returns the same object with an additional +1 reference.
        let retained = ObjcObject::from_retained(unsafe {
            objc::msg_send_id_0(raw, objc::sel(c"retain"))
        })
        .ok_or(MetalError::InternalError(
            "registered texture retain returned null".to_string(),
        ))?;
        state.textures.insert(id, retained);
        state.registry_textures.insert(id, id);
        Ok(())
    }

    /// Registers (or reuses) an `MTLSamplerState` under `id`.
    ///
    /// Converted sampler descriptor tables name the registered handle; a second
    /// registration for the same id is idempotent so a reconverted table cannot
    /// churn sampler objects.
    pub fn register_sampler(&self, id: u64, desc: SamplerDesc) -> Result<(), MetalError> {
        if id == 0 {
            return Err(MetalError::InvalidArg("sampler handle must be nonzero"));
        }
        {
            let state = self.lock_state()?;
            if state.samplers.contains_key(&id) {
                return Ok(());
            }
        }
        let _pool = AutoreleasePool::new();
        let class = objc::class(c"MTLSamplerDescriptor");
        if class.is_null() {
            return Err(MetalError::NotSupported(
                "MTLSamplerDescriptor class is unavailable",
            ));
        }
        // SAFETY: `class` is `MTLSamplerDescriptor`; `new` returns +1.
        let descriptor = ObjcObject::from_retained(unsafe {
            objc::msg_send_id_0(class, objc::sel(c"new"))
        })
        .ok_or(MetalError::InternalError(
            "MTLSamplerDescriptor creation returned null".to_string(),
        ))?;
        // SAFETY: `descriptor` is a live `MTLSamplerDescriptor` and every
        // property takes a documented `NSUInteger` filter/address mode.
        unsafe {
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setMinFilter:"),
                desc.min_filter,
            );
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setMagFilter:"),
                desc.mag_filter,
            );
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setSAddressMode:"),
                desc.s_address_mode,
            );
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setTAddressMode:"),
                desc.t_address_mode,
            );
            objc::msg_send_void_usize(
                descriptor.as_raw(),
                objc::sel(c"setRAddressMode:"),
                desc.r_address_mode,
            );
        }
        // SAFETY: `descriptor` is a live sampler descriptor and the device is
        // live; the returned sampler state is retained (+1).
        let sampler = ObjcObject::from_retained(unsafe {
            objc::msg_send_id_1(
                self.device.as_raw(),
                objc::sel(c"newSamplerStateWithDescriptor:"),
                descriptor.as_raw(),
            )
        })
        .ok_or(MetalError::InternalError(
            "MTLDevice.newSamplerStateWithDescriptor returned null".to_string(),
        ))?;
        let mut state = self.lock_state()?;
        state.samplers.entry(id).or_insert(sampler);
        Ok(())
    }

    /// Registers a content-hashed graphics pipeline under a wire `handle`.
    ///
    /// The content key is derived from the metallib, vertex entry and encoded
    /// [`crate::macos::registry::PipelineState`] blob, so two identical
    /// registrations share one `MTLRenderPipelineState`. The handle is inserted
    /// into the same `content_pipelines` table that `SetGraphicsPipeline`
    /// resolves; `abi` is stored so the executor can bind the converted root
    /// signature automatically.
    pub fn register_content_hashed_graphics_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        vertex_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline(
            handle,
            metallib,
            vertex_entry,
            state,
            PipelineStage::Graphics,
            abi,
        )
    }

    /// Registers a content-hashed mesh pipeline under a wire `handle`.
    pub fn register_content_hashed_mesh_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        mesh_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline(
            handle,
            metallib,
            mesh_entry,
            state,
            PipelineStage::Mesh,
            abi,
        )
    }

    /// Registers a content-hashed graphics pipeline from one library per stage.
    ///
    /// `stages` carries the vertex and pixel functions with the library each one
    /// lives in, so a live `--stdio` pool that emits one metallib per stage is
    /// registrable. Unlike
    /// [`Self::register_content_hashed_graphics_pipeline`] the two functions
    /// need not share a metallib; no bytes are concatenated. The content key
    /// covers every stage's library, so a changed pixel library is a different
    /// pipeline state.
    pub fn register_content_hashed_graphics_pipeline_stages(
        &self,
        handle: u64,
        stages: &[StageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline_stages(
            handle,
            stages,
            state,
            PipelineStage::Graphics,
            abi,
        )
    }

    /// Registers a content-hashed mesh pipeline from one library per stage.
    ///
    /// `stages` carries the mesh, the fragment and (when the state names an
    /// object entry) the object function, each in its own library.
    pub fn register_content_hashed_mesh_pipeline_stages(
        &self,
        handle: u64,
        stages: &[StageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline_stages(
            handle,
            stages,
            state,
            PipelineStage::Mesh,
            abi,
        )
    }

    /// Registers a content-hashed compute pipeline from one library per stage.
    ///
    /// The kernel function is resolved in the stage that names it, never in an
    /// assumed first library.
    pub fn register_content_hashed_compute_pipeline_stages(
        &self,
        handle: u64,
        stages: &[StageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline_stages(
            handle,
            stages,
            state,
            PipelineStage::Compute,
            abi,
        )
    }

    /// Registers a content-hashed compute pipeline under a wire `handle`.
    pub fn register_content_hashed_compute_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        self.register_content_hashed_pipeline(
            handle,
            metallib,
            entry,
            state,
            PipelineStage::Compute,
            abi,
        )
    }

    fn register_content_hashed_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        entry: &str,
        state_bytes: &[u8],
        stage: PipelineStage,
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        if handle == 0 {
            return Err(MetalError::InvalidArg(
                "content pipeline handle must be nonzero",
            ));
        }
        let key = PipelineKey::from_parts(metallib, entry, stage, state_bytes);
        let mut state = self.lock_state()?;
        let registry = content_pipeline_registry(self, &mut state)?;
        let pipeline = match stage {
            PipelineStage::Graphics => {
                registry.register_graphics(key, metallib, entry, state_bytes)
            }
            PipelineStage::Mesh => registry.register_mesh(key, metallib, entry, state_bytes),
            PipelineStage::Compute => {
                registry.register_compute(key, metallib, entry, state_bytes)
            }
        }
        .map_err(registry_error)?;
        state.content_pipelines.insert(handle, pipeline);
        state.abis.insert(handle, abi);
        Ok(())
    }

    /// Registers a content-hashed pipeline created from one library per stage.
    fn register_content_hashed_pipeline_stages(
        &self,
        handle: u64,
        stages: &[StageLibrary<'_>],
        state_bytes: &[u8],
        stage: PipelineStage,
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        if handle == 0 {
            return Err(MetalError::InvalidArg(
                "content pipeline handle must be nonzero",
            ));
        }
        let key = PipelineKey::from_stages(stages, stage, state_bytes);
        let mut state = self.lock_state()?;
        let registry = content_pipeline_registry(self, &mut state)?;
        let pipeline = match stage {
            PipelineStage::Graphics => registry.register_graphics_stages(key, stages, state_bytes),
            PipelineStage::Mesh => registry.register_mesh_stages(key, stages, state_bytes),
            PipelineStage::Compute => registry.register_compute_stages(key, stages, state_bytes),
        }
        .map_err(registry_error)?;
        state.content_pipelines.insert(handle, pipeline);
        state.abis.insert(handle, abi);
        Ok(())
    }

    /// Attaches every live buffer and texture to the persistent residency set.
    ///
    /// The set is created once per transport. On the native `MTLResidencySet`
    /// path the additions are committed and the set is attached to the queue;
    /// the fallback records why it could not create a native set and declares
    /// each allocation per encoder through [`ResidencySet::use_resources`].
    fn sync_residency(&self, state: &mut DeviceState) -> Result<(), MetalError> {
        if state.residency.is_none() {
            let residency =
                ResidencySet::new(self).map_err(|error| MetalError::InternalError(error.to_string()))?;
            residency
                .attach_to_queue(self)
                .map_err(|error| MetalError::InternalError(error.to_string()))?;
            state.residency = Some(residency);
        }
        let residency = state
            .residency
            .as_mut()
            .ok_or(MetalError::InternalError(
                "residency set was not created".to_string(),
            ))?;
        for buffer in state.buffers.values() {
            residency
                .add_buffer_raw(buffer.buffer.as_raw(), buffer.len)
                .map_err(|error| MetalError::InternalError(error.to_string()))?;
        }
        for texture in state.textures.values() {
            residency
                .add_texture_raw(texture.as_raw())
                .map_err(|error| MetalError::InternalError(error.to_string()))?;
        }
        if let Some(registry) = state.texture_registry.as_ref() {
            for id in state.registry_textures.values() {
                residency
                    .add_texture(registry, *id)
                    .map_err(|error| MetalError::InternalError(error.to_string()))?;
            }
        }
        residency.commit();
        Ok(())
    }

    /// Whether a native `MTLResidencySet` backs the persistent residency path.
    pub fn residency_is_native(&self) -> Result<bool, MetalError> {
        let state = self.lock_state()?;
        Ok(state
            .residency
            .as_ref()
            .is_some_and(ResidencySet::is_native))
    }

    /// Exact selector and `file:line` that forced the residency fallback.
    pub fn residency_fallback_reason(&self) -> Result<Option<String>, MetalError> {
        let state = self.lock_state()?;
        Ok(state
            .residency
            .as_ref()
            .and_then(ResidencySet::fallback_reason)
            .map(str::to_string))
    }

    /// Number of allocations recorded by the persistent residency set.
    pub fn residency_allocation_count(&self) -> Result<usize, MetalError> {
        let state = self.lock_state()?;
        Ok(state
            .residency
            .as_ref()
            .map_or(0, ResidencySet::allocation_count))
    }

    /// Reads a registry texture back as RGBA8 pixels through a shared staging
    /// buffer.
    ///
    /// `handle` is the transport handle passed to
    /// [`Self::register_registry_texture`]. The private texture is blitted into
    /// a shared `MTLBuffer` and mapped there; raw `getBytes:` is never issued on
    /// a private texture.
    pub fn read_registered_texture_rgba8(
        &self,
        handle: u64,
    ) -> Result<Vec<[u8; 4]>, MetalError> {
        let _pool = AutoreleasePool::new();
        let state = self.lock_state()?;
        let raw = resolve_registered_texture(&state, handle).ok_or_else(|| {
            MetalError::NotFound(format!("registry texture handle {handle} is not registered"))
        })?;
        let info = texture_info(raw)?;
        if !info.format.is_color() {
            return Err(MetalError::InvalidArg(
                "readback texture must use a colour pixel format",
            ));
        }
        let width = info.width as usize;
        let height = info.height as usize;
        let bytes_per_row = width.checked_mul(4).ok_or(MetalError::InvalidArg(
            "texture row length overflows usize",
        ))?;
        let len = bytes_per_row
            .checked_mul(height)
            .ok_or(MetalError::InvalidArg("texture byte length overflows usize"))?;
        // SAFETY: `device_raw()` is a valid `MTLDevice`; storage mode `0` is
        // shared, `len` is nonzero, and the result is retained (+1).
        let buffer = ObjcObject::from_retained(unsafe {
            objc::msg_send_new_buffer(self.device_raw(), len, 0)
        })
        .ok_or_else(|| {
            MetalError::InternalError("readback MTLBuffer allocation returned null".to_string())
        })?;
        // SAFETY: `queue_raw()` is a valid `MTLCommandQueue`; the command buffer
        // is autoreleased and lives for this pool.
        let cb = unsafe { objc::msg_send_id_0(self.queue_raw(), objc::sel(c"commandBuffer")) };
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
        // SAFETY: `enc` is a live blit encoder, `raw` is a live level-0 texture
        // and `buffer` can hold exactly one image with this row pitch.
        unsafe {
            msg_send_copy_texture_to_buffer(
                enc,
                raw,
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
                "registry texture readback failed (status={status}): {message}"
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
        Ok(bytes
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect())
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

    /// Executes one version 3 submission with the guest's root-argument plan.
    ///
    /// The plan names the translated `(space, register, kind)` each title
    /// root-parameter index feeds; `None` keeps the legacy wire-slot lookup, so
    /// a submission that carried no plan executes exactly as before. See
    /// [`Self::submit_v3_with_stats`] for the execution contract. This wrapper
    /// discards the diagnostic encoder inventory.
    pub fn submit_v3_with_root_plan(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
        plan: Option<&RootParameterPlan>,
    ) -> Result<Reply, MetalError> {
        self.submit_v3_with_stats_and_plan(submission, allocations, plan)
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
    ///   naming either a pipeline registered through
    ///   [`Self::register_mesh_pipeline`] or a content-hashed mesh PSO
    ///   registered through [`Self::register_content_hashed_mesh_pipeline`].
    ///   The per-stage threadgroup sizes come from that pipeline (a legacy
    ///   `MeshPipeline` or the content-hashed `PipelineState`); the wire only
    ///   carries group counts.
    ///   A mesh dispatch outside a pass, without a mesh pipeline, or naming an
    ///   unknown handle fails before any command buffer exists.
    ///
    /// # Asynchronous execution (ADR-017)
    ///
    /// An accepted submission is validated and encoded on the calling thread and
    /// then **committed without waiting**: `ReplyStatus::Ok` means the command
    /// buffer was accepted by the GPU, not that it finished. Every encoder or
    /// validation failure still returns its explicit error *before* `commit`, so
    /// a refusal never publishes a fence value. Completion is published into
    /// [`Self::fence_page`] — or, when the host attached one, into the shared
    /// page given to [`Self::attach_shared_fence_page`] — by an
    /// `addCompletedHandler:` block, which also releases the staging objects
    /// pinned for the submission through the per-transport retire queue. At most
    /// `VICTORY_METAL_FRAMES_IN_FLIGHT` (1..=3, default 3) buffers are in
    /// flight; a full pipeline waits on the *oldest* command buffer only.
    ///
    /// A submission that encodes no GPU work completes immediately through the
    /// same contiguous watermark, so a guest fence never stalls on it.
    ///
    /// A `Present` command is a frame boundary: it drains completed retire
    /// entries, but never waits for its own command buffer.
    pub fn submit_v3_with_stats(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<(Reply, EncoderStats), MetalError> {
        self.submit_v3_with_stats_and_plan(submission, allocations, None)
    }

    /// [`Self::submit_v3_with_stats`] with the guest's root-argument plan.
    ///
    /// Every `SetRootConstants`/`SetRootBufferView`/`SetDescriptorTable` is
    /// resolved through the plan entry for its title slot, so a title
    /// root-parameter index binds the parameter the *translated* root signature
    /// declares at that register rather than the flattened parameter at the same
    /// index. A submission without a plan (or without an entry for a slot) keeps
    /// the legacy wire-slot lookup.
    pub fn submit_v3_with_stats_and_plan(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
        root_plan: Option<&RootParameterPlan>,
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
        let plan = match plan_encoder_segments(&submission.commands) {
            Ok(plan) => plan,
            Err(error) => {
                self.note_refusal();
                return Err(error);
            }
        };
        let _pool = AutoreleasePool::new();
        let mut state = self.lock_state()?;
        // Frame boundary: release staging work the GPU already finished with.
        let _ = self.drain_retired();
        let list_key = submission
            .list
            .map(|list| (list.list_id, list.list_generation));
        if let Some((list_id, generation)) = list_key
            && state.list_was_executed(list_id, generation)
        {
            self.note_refusal();
            return Ok((reply(ReplyStatus::AlreadyExecuted), EncoderStats::default()));
        }
        for command in &submission.commands {
            if let CommandV3::Present(present) = command
                && !state.surfaces.contains_key(&present.surface_id)
            {
                note_unknown_site("surface");
                self.note_refusal();
                return Ok((reply(ReplyStatus::UnknownHandle), EncoderStats::default()));
            }
        }

        UNKNOWN_HANDLE_ID.store(0, std::sync::atomic::Ordering::Relaxed);
        let prepared = match prepare_v3(
            &mut state,
            self,
            allocations,
            &submission.commands,
            &plan,
            root_plan,
        ) {
            Ok(prepared) => prepared,
            Err(PrepareAbort::UnknownHandle) => {
                self.note_refusal();
                return Ok((reply(ReplyStatus::UnknownHandle), EncoderStats::default()));
            }
            Err(PrepareAbort::Error(error)) => {
                self.note_refusal();
                return Err(error);
            }
        };

        let mut stats = EncoderStats::default();
        if prepared.is_empty() {
            if let Some((list_id, generation)) = list_key {
                state.mark_list_executed(list_id, generation);
            }
            // Nothing was encoded, so nothing can still be reading: complete the
            // sequence through the same watermark path a completion handler
            // uses, never across an in-flight command buffer.
            if let Ok(mut async_state) = self.async_submit.lock() {
                async_state.note_empty_submission(submission.sequence);
            }
            self.note_refusal();
            return Ok((reply(ReplyStatus::Ok), stats));
        }

        // ADR-053: the first pass of a traced submission is sampled after its
        // command buffer completes; at most MAX_TRACED_PASSES per process.
        let trace_sample = if crate::macos::trace::enabled() {
            let first = prepared.iter().find_map(|segment| match segment {
                PreparedSegment::Render { passes, .. } => passes
                    .iter()
                    .find(|pass| pass.draws > 0)
                    .and_then(|pass| pass.trace_target),
                _ => None,
            });
            match (first, state.trace.as_mut()) {
                (Some(target), Some(trace)) => {
                    if trace.next_pass() {
                        Some(target)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        // Attach every live buffer and registered texture to the persistent
        // residency set and publish the additions before the queues run.
        self.sync_residency(&mut state)?;
        // Encoding errors (including a null encoder) are reported before any
        // commit, so nothing is published for a rejected submission.
        let encoded = match self.encode_v3_segments(
            prepared,
            &mut stats,
            state.residency.as_ref(),
        ) {
            Ok(encoded) => encoded,
            Err(error) => {
                self.note_refusal();
                return Err(error);
            }
        };
        let has_present = submission
            .commands
            .iter()
            .any(|command| matches!(command, CommandV3::Present(_)));
        // Metal faults the whole command buffer; the first encoder segment is
        // the only attribution an asynchronous completion handler can name.
        let site = plan
            .first()
            .map(segment_failure_site)
            .unwrap_or(("command buffer", 0));
        self.commit_async(encoded, submission.sequence, site, trace_sample.is_some())?;
        if let Some(target) = trace_sample {
            self.trace_sample_target(&target);
        }
        if let Some((list_id, generation)) = list_key {
            state.mark_list_executed(list_id, generation);
        }
        if has_present {
            // Frame boundary: drain what the GPU has already finished with.
            let _ = self.drain_retired();
        }
        Ok((reply(ReplyStatus::Ok), stats))
    }

    /// Encodes prepared segments onto one command buffer and returns it ready to
    /// commit.
    ///
    /// Nothing is committed here: the caller attaches the completion handler and
    /// commits through [`Self::commit_async`], so an encoding error leaves the
    /// GPU untouched. Every owned staging object (a converted argument buffer)
    /// is returned with the command buffer so the caller can pin it in the retire
    /// queue until the fence completes.
    fn encode_v3_segments(
        &self,
        segments: Vec<PreparedSegment>,
        stats: &mut EncoderStats,
        residency: Option<&ResidencySet>,
    ) -> Result<EncodedSubmission, MetalError> {
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
        let mut staged: Vec<ObjcObject> = Vec::new();

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
                    if let Some(residency) = residency {
                        // Native residency is a no-op; the fallback declares
                        // every allocation once per encoder.
                        residency.use_resources(enc);
                    }
                    for op in &ops {
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
                            PreparedComputeOp::ArgumentBuffer(argument_buffer) => {
                                // SAFETY: `enc` is a live compute encoder and the
                                // buffer is owned by this prepared segment.
                                unsafe {
                                    objc::msg_send_set_buffer(
                                        enc,
                                        argument_buffer.buffer.as_raw(),
                                        0,
                                        argument_buffer.index,
                                    );
                                    if let Some(heap) = &argument_buffer.heap {
                                        objc::msg_send_set_buffer(
                                            enc,
                                            heap.as_raw(),
                                            0,
                                            crate::macos::bindings::RESOURCE_DESCRIPTOR_HEAP_BUFFER_INDEX,
                                        );
                                    }
                                    for resource in &argument_buffer.resources {
                                        objc::msg_send_use_resource(
                                            enc,
                                            *resource,
                                            MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                                        );
                                    }
                                }
                                // Staging: the GPU reads it after `commit`.
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
                    // Staging: the GPU reads it after `commit` returns.
                    for op in ops {
                        if let PreparedComputeOp::ArgumentBuffer(argument_buffer) = op {
                            staged.extend(argument_buffer.title);
                            staged.push(argument_buffer.buffer);
                            staged.extend(argument_buffer.heap);
                            staged.extend(argument_buffer.tables);
                        }
                    }
                }
                PreparedSegment::Render {
                    passes,
                    width,
                    height,
                } => {
                    for pass in passes {
                        let PreparedRenderPass { descriptor, ops, .. } = pass;
                        if ops.iter().any(PreparedRenderOp::is_indirect) {
                            encode_indirect_render_segment(
                                cb,
                                descriptor,
                                width,
                                height,
                                &ops,
                                stats,
                                residency,
                            )?;
                            for op in ops {
                                if let PreparedRenderOp::ArgumentBuffer(argument_buffer) = op {
                                    staged.extend(argument_buffer.title);
                                    staged.push(argument_buffer.buffer);
                                    staged.extend(argument_buffer.heap);
                                    staged.extend(argument_buffer.tables);
                                }
                            }
                            continue;
                        }
                        let mut encoder = RenderEncoder::new_with_descriptor(
                            self,
                            cb,
                            descriptor,
                            width,
                            height,
                        );
                        encoder.ensure_started()?;
                        stats.render_encoders += 1;
                        if let Some(residency) = residency {
                            residency.use_resources(encoder.raw_encoder());
                        }
                        for op in &ops {
                            match op {
                                PreparedRenderOp::Pipeline { state, depth_state } => {
                                        encoder.set_pipeline_state(
                                        *state,
                                        depth_state.as_ref().map(DepthState::raw),
                                    )?;
                                }
                                PreparedRenderOp::DepthStencil(raw) => {
                                    encoder.set_depth_stencil_state(*raw)?;
                                }
                                PreparedRenderOp::Rasterizer(raster) => {
                                    encoder.apply_rasterizer_state(raster)?;
                                }
                                PreparedRenderOp::StencilRef(value) => {
                                    encoder.apply_stencil_ref(*value)?;
                                }
                                PreparedRenderOp::BlendColor(color) => {
                                    encoder.apply_blend_color(*color)?;
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
                            PreparedRenderOp::ArgumentBuffer(argument_buffer) => {
                                encoder.bind_argument_buffer(
                                    argument_buffer.stage,
                                    argument_buffer.buffer.as_raw(),
                                    argument_buffer.index,
                                )?;
                                if let Some(heap) = &argument_buffer.heap {
                                    encoder.bind_argument_buffer(
                                        argument_buffer.stage,
                                        heap.as_raw(),
                                        crate::macos::bindings::RESOURCE_DESCRIPTOR_HEAP_BUFFER_INDEX,
                                    )?;
                                }
                                for resource in &argument_buffer.resources {
                                    // SAFETY: the resource is a live Metal object
                                    // named by the argument buffer.
                                    unsafe {
                                        objc::msg_send_use_resource(
                                            encoder.raw_encoder(),
                                            *resource,
                                            MTL_RESOURCE_USAGE_READ
                                            | MTL_RESOURCE_USAGE_WRITE,
                                        );
                                    }
                                }
                                // Staging: the GPU reads it after `commit`.
                            }
                            PreparedRenderOp::StageBytes { stage, slot, bytes } => {
                                encoder.set_stage_bytes(*stage, bytes, *slot)?;
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
                    // Staging: the GPU reads it after `commit` returns.
                    for op in ops {
                        if let PreparedRenderOp::ArgumentBuffer(argument_buffer) = op {
                            staged.extend(argument_buffer.title);
                            staged.push(argument_buffer.buffer);
                            staged.extend(argument_buffer.heap);
                            staged.extend(argument_buffer.tables);
                        }
                    }
                    }
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
                    if let Some(residency) = residency {
                        residency.use_resources(enc);
                    }
                    for op in &ops {
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
                            PreparedBlitOp::CopyBufferToTexture(copy) => {
                                blit::copy_buffer_to_texture(enc, copy)?;
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
                                if let Some(residency) = residency {
                                    residency.use_resources(enc);
                                }
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

        // SAFETY: `cb` has every encoder closed and is still uncommitted; the
        // extra reference keeps it alive past this function's autorelease pool.
        let command_buffer = ObjcObject::from_retained(unsafe {
            objc::msg_send_id_0(cb, objc::sel(c"retain"))
        })
        .ok_or_else(|| {
            MetalError::InternalError("command buffer retain returned null".to_string())
        })?;
        Ok(EncodedSubmission {
            command_buffer,
            staged,
        })
    }
}

impl Drop for DirectMetalTransport {
    /// Drains every committed command buffer before the transport state dies.
    ///
    /// Completion blocks hold a raw pointer to the transport's async state, so
    /// the state must outlive every committed buffer. `waitUntilCompleted`
    /// invokes the pending handlers before it returns, so after this barrier no
    /// handler can still reference freed memory.
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
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
    residency: Option<&ResidencySet>,
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
    if let Some(residency) = residency {
        residency.use_resources(enc);
    }

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
                            depth_state.raw(),
                        );
                    }
                }
            }
            PreparedRenderOp::DepthStencil(raw) => {
                // SAFETY: `raw` is a live cached `MTLDepthStencilState`.
                unsafe {
                    objc::msg_send_void_1(enc, objc::sel(c"setDepthStencilState:"), *raw);
                }
            }
            PreparedRenderOp::Rasterizer(raster) => {
                // SAFETY: `enc` is a live render encoder.
                unsafe { state::apply_rasterizer(enc, raster)? };
            }
            PreparedRenderOp::StencilRef(value) => {
                // SAFETY: `enc` is a live render encoder.
                unsafe { state::apply_stencil_ref(enc, *value)? };
            }
            PreparedRenderOp::BlendColor(color) => {
                // SAFETY: `enc` is a live render encoder.
                unsafe { state::apply_blend_color(enc, *color)? };
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
            PreparedRenderOp::ArgumentBuffer(argument_buffer) => {
                // SAFETY: the buffer is owned by the prepared segment and the
                // stage/index were validated by the layout.
                unsafe {
                    msg_send_set_stage_buffer(
                        enc,
                        argument_buffer.stage,
                        argument_buffer.buffer.as_raw(),
                        0,
                        argument_buffer.index,
                    );
                    for resource in &argument_buffer.resources {
                        objc::msg_send_use_resource(
                            enc,
                            *resource,
                            MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                        );
                    }
                }
            }
            PreparedRenderOp::StageBytes { stage, slot, bytes } => {
                // SAFETY: `bytes` is valid for the call; Metal copies it.
                unsafe {
                    msg_send_set_stage_bytes(enc, *stage, bytes.as_ptr(), bytes.len(), *slot);
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

/// `-[MTLRenderCommandEncoder setVertex|Fragment|Mesh|ObjectBuffer:offset:atIndex:]`.
unsafe fn msg_send_set_stage_buffer(
    enc: objc::Id,
    stage: ShaderStage,
    buffer: objc::Id,
    offset: usize,
    index: usize,
) {
    let selector = match stage {
        ShaderStage::Vertex => objc::sel(c"setVertexBuffer:offset:atIndex:"),
        ShaderStage::Fragment => objc::sel(c"setFragmentBuffer:offset:atIndex:"),
        ShaderStage::Mesh => objc::sel(c"setMeshBuffer:offset:atIndex:"),
        ShaderStage::Object => objc::sel(c"setObjectBuffer:offset:atIndex:"),
        ShaderStage::Compute => objc::sel(c"setBuffer:offset:atIndex:"),
    };
    // SAFETY: Caller supplies a live encoder of the matching family.
    unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, usize, usize);
        let send: Fn = std::mem::transmute(objc::msg_send_entry());
        send(enc, selector, buffer, offset, index);
    }
}

/// `-[MTLRenderCommandEncoder setVertex|Fragment|Mesh|ObjectBytes:length:atIndex:]`.
unsafe fn msg_send_set_stage_bytes(
    enc: objc::Id,
    stage: ShaderStage,
    bytes: *const u8,
    len: usize,
    index: usize,
) {
    let selector = match stage {
        ShaderStage::Vertex => objc::sel(c"setVertexBytes:length:atIndex:"),
        ShaderStage::Fragment => objc::sel(c"setFragmentBytes:length:atIndex:"),
        ShaderStage::Mesh => objc::sel(c"setMeshBytes:length:atIndex:"),
        ShaderStage::Object => objc::sel(c"setObjectBytes:length:atIndex:"),
        ShaderStage::Compute => objc::sel(c"setBytes:length:atIndex:"),
    };
    // SAFETY: Caller supplies a live encoder and `bytes` readable for `len`.
    unsafe {
        type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, *const c_void, usize, usize);
        let send: Fn = std::mem::transmute(objc::msg_send_entry());
        send(enc, selector, bytes.cast(), len, index);
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

    /// A wire descriptor-table `set` is an ordinal among the pipeline's
    /// tables, not a flattened root-parameter index: a signature with root
    /// constants first must still resolve table 0 to the table parameter.
    #[test]
    fn a_descriptor_table_set_resolves_by_ordinal() {
        use crate::macos::bindings::{
            DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec,
            RootSignatureLayout,
        };
        let layout = RootSignatureLayout::from_parameters(
            ReflectedStage::Compute,
            &[
                RootParameterSpec {
                    kind: RootParameterKind::Constants,
                    size: 16,
                    register: 0,
                    space: 0,
                },
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(DescriptorTableKind::UnorderedAccessView),
                    size: 8,
                    register: 0,
                    space: 1,
                },
                RootParameterSpec {
                    kind: RootParameterKind::DescriptorTable(DescriptorTableKind::ShaderResourceView),
                    size: 8,
                    register: 0,
                    space: 1,
                },
            ],
        )
        .expect("three root parameters");
        let abi = super::PipelineAbi::compute(layout);
        let uav = abi.root_parameter_for_table(0).expect("first table");
        assert_eq!(uav.space(), 1);
        assert_eq!(uav.register(), 0);
        assert!(matches!(
            uav.kind(),
            RootParameterKind::DescriptorTable(DescriptorTableKind::UnorderedAccessView)
        ));
        // A flattened-index lookup would have named the constants block; the
        // ordinal lookup must not.
        assert!(matches!(
            abi.root_parameter(0).map(|slot| slot.kind()),
            Some(RootParameterKind::Constants)
        ));
        assert!(abi.root_parameter_for_table(2).is_none());
    }

    /// Encoder state is applied once per change: repeating the same wire state
    /// must not emit a second op, and a later change must be emitted again.
    #[test]
    fn encoder_state_dedup_is_idempotent_and_ordered() {
        let mut dedup: super::StateDedup<u32> = super::StateDedup::default();
        // First use, repeat, change, repeat, return to the first value.
        let sequence = [7u32, 7, 9, 9, 7];
        let emitted: Vec<bool> = sequence
            .iter()
            .map(|value| dedup.should_emit(*value))
            .collect();
        assert_eq!(emitted, vec![true, false, true, false, true]);
    }

    #[test]
    fn encoder_state_dedup_handles_composite_values() {
        let mut dedup: super::StateDedup<[u32; 2]> = super::StateDedup::default();
        assert!(dedup.should_emit([1, 2]));
        assert!(!dedup.should_emit([1, 2]));
        assert!(dedup.should_emit([1, 3]));
        assert!(dedup.should_emit([1, 2]));
    }

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

        // ADR-017: the submission is committed asynchronously, so host memory
        // is only valid after the transport observes GPU completion.
        transport
            .wait_for_gpu()
            .expect("the submitted frame should complete");

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

/// CPU-only tests for the asynchronous submit bookkeeping (ADR-017).
///
/// These exercise the fence watermark, the retire ordering and the in-flight cap
/// without touching Metal, so they run on any host and in the default `--lib`
/// suite.
#[cfg(test)]
mod async_pipeline_tests {
    use super::*;

    /// A completion for a fence this transport never committed must publish and
    /// release nothing: a dropped or duplicated handler is not evidence that the
    /// GPU finished.
    #[test]
    fn uncommitted_completion_never_publishes() {
        let mut watermark = CompletionWatermark::new();
        watermark.push(1);
        watermark.push(2);

        assert!(!watermark.mark_done(9), "fence 9 was never committed");
        assert_eq!(watermark.advance(), None);
        assert_eq!(watermark.completed(), 0);
        assert_eq!(watermark.front(), Some(1));
    }

    #[test]
    fn out_of_order_completion_keeps_the_watermark_contiguous() {
        let mut watermark = CompletionWatermark::new();
        for fence in 1..=3 {
            watermark.push(fence);
        }

        // The newest buffer finishing first must not jump the fence.
        assert!(watermark.mark_done(3));
        assert_eq!(watermark.advance(), None);
        assert_eq!(watermark.completed(), 0);

        // Fence 1 completing while 2 is still running publishes exactly 1.
        assert!(watermark.mark_done(1));
        assert_eq!(watermark.advance(), Some(1));
        assert_eq!(watermark.completed(), 1);

        // 2 follows; 3 was already done, so the rest of the prefix drains too.
        assert!(watermark.mark_done(2));
        assert_eq!(watermark.advance(), Some(3), "1..=3 are contiguous");
        assert_eq!(watermark.completed(), 3);
        assert_eq!(watermark.front(), None);
    }

    #[test]
    fn duplicate_completion_is_ignored() {
        let mut watermark = CompletionWatermark::new();
        watermark.push(4);
        assert!(watermark.mark_done(4));
        assert!(!watermark.mark_done(4));
        assert_eq!(watermark.advance(), Some(4));
        assert_eq!(watermark.completed(), 4);
    }

    #[test]
    fn retired_pins_release_only_after_the_watermark_advances() {
        let mut state = AsyncSubmitState::new(Arc::new(FencePage::new()), 3, 64);
        let fence = state.reserve_fence(7);
        state.watermark.push(fence);
        let token = state.stage_caller_pin(77);
        state
            .retire
            .try_push(fence, token)
            .expect("a fresh queue has room");

        // Nothing is observed complete yet: the pin must stay.
        assert_eq!(state.drain(), 0);
        assert!(state.released.is_empty(), "released before completion");
        assert_eq!(state.retire.len(), 1);

        assert!(state.watermark.mark_done(fence));
        assert_eq!(state.drain(), 0, "advance has not run yet");
        assert!(state.watermark.advance().is_some());
        assert_eq!(state.drain(), 1);
        assert_eq!(state.released, vec![77]);
        assert!(state.retire.is_empty());
    }

    #[test]
    fn reserve_fence_is_monotonic_across_a_replayed_sequence() {
        let mut state = AsyncSubmitState::new(Arc::new(FencePage::new()), 3, 64);
        assert_eq!(state.reserve_fence(1), 1);
        assert_eq!(state.reserve_fence(2), 2);
        // A replay of an older sequence must not move the fence backwards.
        assert_eq!(state.reserve_fence(1), 3);
    }

    /// A submission with no GPU work completes immediately, but never across a
    /// command buffer that is still in flight.
    #[test]
    fn empty_submission_completes_without_jumping_in_flight_work() {
        let page = Arc::new(FencePage::new());
        let mut state = AsyncSubmitState::new(Arc::clone(&page), 3, 64);

        // Fence 1 is committed and still running.
        let in_flight = state.reserve_fence(1);
        state.watermark.push(in_flight);
        assert_eq!(in_flight, 1);

        // An empty submission with sequence 2 must not publish 2 yet.
        state.note_empty_submission(2);
        assert_eq!(page.completed(), 0, "published across an in-flight fence");
        assert_eq!(state.completed(), 0);
        assert_eq!(
            state.watermark.front(),
            Some(1),
            "fence 1 still gates the contiguous watermark"
        );

        // Once fence 1 completes, both are published in commit order.
        state.note_completion(in_flight, true);
        assert_eq!(page.completed(), 2);
        assert_eq!(state.completed(), 2);
        assert!(state.watermark.is_idle());

        // An empty submission with nothing in flight publishes at once.
        state.note_empty_submission(3);
        assert_eq!(page.completed(), 3);
        assert_eq!(state.counters.submitted, 0, "no command buffer was committed");
    }

    #[test]
    fn frames_in_flight_cap_is_clamped_to_the_supported_window() {
        assert_eq!(clamp_frames_in_flight(None), 3);
        assert_eq!(clamp_frames_in_flight(Some("2")), 2);
        assert_eq!(clamp_frames_in_flight(Some("0")), 1);
        assert_eq!(clamp_frames_in_flight(Some("64")), 3);
        assert_eq!(clamp_frames_in_flight(Some("nonsense")), 3);
        assert_eq!(clamp_frames_in_flight(Some(" 1 ")), 1);
    }

    /// A simulated failed command buffer keeps the wire status and carries the
    /// Metal diagnostic text, named by command family and index. No GPU or
    /// `MTLCommandBuffer` is created: the decision is this pure function.
    #[test]
    fn command_buffer_failure_keeps_status_and_carries_metal_text() {
        let failure = describe_command_buffer_failure(
            586,
            4,
            "draw",
            12,
            "Invalid resource (out of bounds write)\nsecond line",
        );
        assert_eq!(
            failure.status,
            ReplyStatus::ExecutionFailed,
            "the guest status must not change"
        );
        assert_eq!(failure.sequence, 586);
        assert_eq!(failure.kind, "draw");
        assert_eq!(failure.index, 12);
        assert!(
            failure.reason.contains("draw #12"),
            "{}",
            failure.reason
        );
        assert!(
            failure.reason.contains("Invalid resource"),
            "{}",
            failure.reason
        );
        assert!(
            failure.reason.contains("out of bounds write"),
            "{}",
            failure.reason
        );
        assert!(
            !failure.reason.contains('\n'),
            "the reason must stay one line: {}",
            failure.reason
        );
        assert!(failure.reason.len() <= MAX_GPU_REASON_BYTES);
    }

    /// The bounded reason never splits a UTF-8 character and never exceeds the
    /// documented ceiling, even for a long Metal error.
    #[test]
    fn gpu_reason_is_sanitised_and_bounded() {
        let long_ascii = "x".repeat(500);
        let reason = bounded_gpu_reason(&long_ascii);
        assert!(reason.len() <= MAX_GPU_REASON_BYTES);
        assert!(reason.ends_with("..."));

        let long_unicode = "é".repeat(500);
        let reason = bounded_gpu_reason(&long_unicode);
        assert!(reason.len() <= MAX_GPU_REASON_BYTES);
        assert!(reason.is_char_boundary(reason.len()));

        assert_eq!(bounded_gpu_reason(""), "unknown Metal error");
        assert_eq!(
            bounded_gpu_reason("line one\n\tline two"),
            "line one line two"
        );
    }
}
