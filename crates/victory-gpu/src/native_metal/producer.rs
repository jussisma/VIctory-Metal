//! D3D12 command-list to Wire v3 producer.
//!
//! Implements the `VICTORY_BACKEND=metal` translation of recorded command-list
//! state (render targets, viewports, PSOs, root arguments, vertex/index
//! buffers, draws, dispatches, barriers, copies, present) into
//! [`SubmissionV3`] batches, the version 3 stateful command stream described
//! by `crates/victory-metal/src/wire_v3.rs`.
//!
//! # Dependency direction
//!
//! `victory-gpu` is the Win64 guest-side crate and reuses the canonical
//! version 3 model from `victory_metal::wire_v3` ([`SubmissionV3`],
//! [`CommandV3`], [`BeginRenderPass`], ...): the producer owns recording and
//! lowering, the codec owns encoding. The IPC transport itself stays in the
//! separate bridge task; this module opens no socket and carries its
//! submissions only through [`WireV3Sink`].
//!
//! # Recording model
//!
//! Every method mirrors one D3D12 graphics-command-list entry point. State
//! calls only update the producer's state; the ordered command stream is
//! lowered when the state is first needed:
//!
//! - [`MetalProducer::set_render_targets`] (`OMSetRenderTargets`) records the
//!   requested attachments but does not emit `BeginRenderPass` yet, so clears
//!   and viewport/scissor calls that follow fold into the pass.
//! - The first draw or mesh dispatch opens the pass. Work never opens a
//!   second pass for the same attachments, so a pipeline, viewport or scissor
//!   change does not split it; changing the render targets or depth attachment
//!   does.
//! - [`MetalProducer::resource_barrier`], copies, compute dispatches and
//!   present close the pass first: the wire forbids those inside a pass.
//! - `flush` closes the pass and returns the submission for the commands
//!   recorded so far, with no sequence assigned; the transport assigns the
//!   session-wide sequence when the submission is enqueued, so a pass never
//!   spans two submissions and an empty or dropped stream consumes nothing.
//!
//! # Clears
//!
//! `ClearRenderTargetView` and `ClearDepthStencilView` name a descriptor's
//! resource, not a currently bound slot: D3D12 lets a title clear a target
//! before it binds one, and lets it clear a target it never draws into. The
//! producer therefore never drops a clear and never refuses one just because
//! the target is not bound yet. It keeps clears by [`ResourceId`] (allocation
//! plus generation) and lowers each one at the first point where the wire can
//! express it in recording order:
//!
//! - A clear for a target that a later `OMSetRenderTargets` binds becomes that
//!   attachment's `LoadOp::Clear`/`StoreOp::Store`, and is removed from the
//!   pending set, so binding the same target again does not clear it twice.
//!   A second clear before the bind replaces the first: the last value is the
//!   target's content when the pass opens.
//! - A clear for a target that is never bound is encoded as its own
//!   single-attachment render pass (`BeginRenderPass` with
//!   `LoadOp::Clear`/`StoreOp::Store`, no draws, `EndRenderPass`) before the
//!   next draw, dispatch, copy, barrier, present or `flush` that would
//!   otherwise be encoded first.
//! - A clear issued while a pass is already open closes that pass (its
//!   attachments are stored), encodes the clear as that standalone pass, and
//!   leaves the same attachment set recorded; the next draw reopens it with
//!   `LoadOp::Load`. Nothing is reordered and nothing is dropped, even when no
//!   draw follows. `EndRenderPass` therefore resets the recorded load actions:
//!   a pass that already ran loads its attachments when it is reopened.
//! - A recorded attachment cleared but never drawn still clears: `flush`
//!   materialises the recorded pass (`BeginRenderPass` + `EndRenderPass`) so
//!   its clear load action executes.
//! - Depth/stencil clears take the same path. The wire requires 1..=8 colour
//!   attachments in every pass, so a depth-only clear uses the first recorded
//!   colour attachment as a load/store anchor (the depth attachment must match
//!   its size). A depth clear with no colour attachment to anchor its pass
//!   cannot be expressed and is refused by name at the point it would have
//!   been encoded; that refusal fails `Close` instead of dropping the clear.
//!
//! # Failure
//!
//! Anything the producer cannot express returns
//! [`ProducerError::Unsupported`] without appending a command. There is no
//! host-D3D12 or D3DMetal call on this path (ADR-008): `list.rs` forwards a
//! refused call to [`MetalList::note_error`] and `Close` returns the failure
//! HRESULT to the title.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::native_metal::bindings::{
    RootArgument, RootArgumentError, RootArgumentKind, MAX_ROOT_ARGUMENTS,
};
use victory_metal::wire::{
    BufferBinding, Dispatch3d, DispatchMesh, ListIdentity, Reply, ReplyStatus, ResourceBarrier,
    WireError, MAX_BINDINGS_PER_DISPATCH,
};
pub use victory_metal::wire::BarrierEntry;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBuffer, CopyBufferToTexture, CopyTexture,
    CopyTextureRegion as WireCopyTextureRegion, DepthStencilAttachment, DescriptorKind,
    DescriptorTableEntryV3, DescriptorTableV3,
    DrawIndexedInstanced, DrawInstanced, ExecuteIndirectDispatch, ExecuteIndirectDispatchMesh,
    ExecuteIndirectDraw, ExecuteIndirectDrawIndexed, IndirectArguments,
    IndexBufferBinding, IndexFormat, LoadOp, PipelineRef, Present, PrimitiveTopology,
    ResolveSubresource as WireResolveSubresource, RootBufferView, RootConstantsV3, ScissorRect,
    StoreOp, TextureSubresource, VertexBuffer, VertexBufferSet, Viewport, MAX_BARRIERS_V3,
    MAX_COLOR_ATTACHMENTS, MAX_COMMANDS_V3, MAX_INDIRECT_COMMANDS_V3, MAX_ROOT_CONSTANT_BYTES_V3,
    MAX_VERTEX_BUFFERS, OP_BEGIN_RENDER_PASS, OP_COPY_BUFFER, OP_COPY_BUFFER_TO_TEXTURE,
    OP_COPY_TEXTURE, OP_COPY_TEXTURE_REGION, OP_DESCRIPTOR_TABLE, OP_DISPATCH_3D, OP_DISPATCH_MESH,
    OP_DRAW_INDEXED_INSTANCED, OP_EXECUTE_INDIRECT_DISPATCH, OP_EXECUTE_INDIRECT_DISPATCH_MESH,
    OP_EXECUTE_INDIRECT_DRAW,
    OP_EXECUTE_INDIRECT_DRAW_INDEXED, OP_PRESENT, OP_RESOURCE_BARRIER,
    OP_ROOT_CONSTANTS, OP_SET_INDEX_BUFFER, OP_SET_PRIMITIVE_TOPOLOGY, OP_SET_ROOT_BUFFER_VIEW,
    OP_SET_SCISSOR, OP_SET_VIEWPORT, OP_SET_VERTEX_BUFFERS,
};
pub use victory_metal::wire_v3::SubmissionV3;

/// Input-assembler vertex-buffer slots a D3D12 list tracks.
pub const MAX_VERTEX_SLOTS: usize = 32;

/// Sequence numbers are session-wide so two lists never collide.
static NEXT_SEQUENCE: AtomicU32 = AtomicU32::new(1);

/// The sequence value that means "not assigned to a transport frame yet".
///
/// A submission leaves [`MetalProducer::flush`] with this placeholder; only
/// the choke points before the wire ([`MetalProducer::submit`] and
/// `NativeQueue::enqueue`) assign a real sequence, exactly once.
pub const UNASSIGNED_SEQUENCE: u32 = 0;

/// A resource's stable identity on the wire.
///
/// The native bridge owns the allocation registry; until it registers a
/// guest range, the guest uses the title allocation's base address as the
/// key with generation `1`. Both are never zero, so a submission is always
/// encodable; the bridge validates the key against its registry before any
/// GPU work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceId {
    /// Allocation registry key, never zero.
    pub allocation_id: u64,
    /// Allocation generation, never zero.
    pub generation: u32,
}

impl ResourceId {
    /// Builds a checked identity.
    pub fn new(allocation_id: u64, generation: u32) -> Result<Self, ProducerError> {
        if allocation_id == 0 || generation == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "resource identity",
            });
        }
        Ok(Self {
            allocation_id,
            generation,
        })
    }

    /// The provisional guest identity of an allocation at `address`.
    pub fn from_guest_address(address: u64) -> Result<Self, ProducerError> {
        Self::new(address, 1)
    }

    /// The wire identity of a guest range the share resolver resolved.
    ///
    /// This is the identity a kind 2 registration and the wire bindings use:
    /// the share allocation id, never the guest address. Callers register the
    /// range first (`native_metal::queue::register_resolved_range`), so the host
    /// validates the same pair the submission names.
    pub fn from_range(
        range: &crate::native_metal::bindings::GuestRange,
    ) -> Result<Self, ProducerError> {
        Self::new(range.allocation_id, range.generation)
    }
}

/// A checked list identity (both fields nonzero).
fn checked_identity(
    list_id: u64,
    list_generation: u32,
) -> Result<ListIdentity, ProducerError> {
    if list_id == 0 || list_generation == 0 {
        return Err(ProducerError::InvalidListIdentity);
    }
    Ok(ListIdentity {
        list_id,
        list_generation,
    })
}

/// What `IASetPrimitiveTopology` asks the native path to do with one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopologyChange {
    /// A topology the Wire v3 stream expresses directly.
    Map(PrimitiveTopology),
    /// `D3D12_PRIMITIVE_TOPOLOGY_UNDEFINED`: leave the recorded topology alone.
    Undefined,
    /// A valid D3D12 value the Metal path cannot express (strips with
    /// adjacency, control-point patch lists, or a reserved enum gap). It is an
    /// accepted, non-fatal hint: the title keeps drawing with the topology it
    /// last set, exactly as D3D12 "no change" state would.
    Named(&'static str),
}

/// `IASetPrimitiveTopology`'s number as a [`TopologyChange`].
///
/// The full D3D12 enum is accepted: `0` is undefined (no change), `1..=5` map
/// to the five Wire v3 topologies, adjacency (`10..=13`), control-point patch
/// lists (`33..=64`) and the reserved gaps are named state hints, never
/// refusals. A value outside `0..=64` is the only refusal.
fn primitive_topology(value: u32) -> Result<TopologyChange, ProducerError> {
    Ok(match value {
        0 => TopologyChange::Undefined,
        1 => TopologyChange::Map(PrimitiveTopology::PointList),
        2 => TopologyChange::Map(PrimitiveTopology::LineList),
        3 => TopologyChange::Map(PrimitiveTopology::LineStrip),
        4 => TopologyChange::Map(PrimitiveTopology::TriangleList),
        5 => TopologyChange::Map(PrimitiveTopology::TriangleStrip),
        10 => TopologyChange::Named("LINELIST_ADJ"),
        11 => TopologyChange::Named("LINESTRIP_ADJ"),
        12 => TopologyChange::Named("TRIANGLELIST_ADJ"),
        13 => TopologyChange::Named("TRIANGLESTRIP_ADJ"),
        33..=64 => TopologyChange::Named("a control-point patch list"),
        6..=9 | 14..=32 => TopologyChange::Named("a reserved D3D12 topology value"),
        _ => {
            return Err(ProducerError::unsupported_opcode(
                "IASetPrimitiveTopology: topology value outside the D3D12 enum",
                OP_SET_PRIMITIVE_TOPOLOGY,
            ));
        }
    })
}

/// Draw-list state that D3D12 allows but Wire v3 has no command for.
///
/// These are hints or legal unbinds, not failures: counting them under their
/// own names keeps a boot's refusal budget for real errors and makes the
/// state changes visible in the log. A counter is logged at powers of two, so
/// the log stays bounded however long a title runs.
pub mod hint_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// `IASetIndexBuffer(null)`: the index buffer was unbound.
    pub const DRAW_LIST_INDEX_UNBIND: usize = 0;
    /// `SetPixelShaderDepthOrder`: a recorded depth-order hint.
    pub const DEPTH_ORDER_HINTS: usize = 1;
    /// `SetShadingRate`: a recorded shading-rate hint.
    pub const SHADING_RATE_HINTS: usize = 2;
    /// A topology D3D12 defines but Wire v3 cannot express.
    pub const TOPOLOGY_HINTS: usize = 3;
    /// `D3D12_PRIMITIVE_TOPOLOGY_UNDEFINED`: no change to the topology.
    pub const TOPOLOGY_UNDEFINED: usize = 4;

    /// The static counter names, in the order of the constants above.
    pub const NAMES: [&str; 5] = [
        "draw_list_index_unbind",
        "depth_order_hints",
        "shading_rate_hints",
        "topology_hints",
        "topology_undefined",
    ];

    static COUNTS: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];

    /// Counts one hint and logs the whole census at powers of two.
    pub(crate) fn note(outcome: usize) -> u64 {
        let count = COUNTS[outcome].fetch_add(1, Relaxed) + 1;
        if count.is_power_of_two() {
            let [index_unbind, depth_order, shading_rate, topology, undefined] =
                std::array::from_fn(|i| COUNTS[i].load(Relaxed));
            tracing::info!(
                draw_list_index_unbind = index_unbind,
                depth_order_hints = depth_order,
                shading_rate_hints = shading_rate,
                topology_hints = topology,
                topology_undefined = undefined,
                "draw-list hint census"
            );
        }
        count
    }

    /// The counters, in [`NAMES`] order.
    pub fn counts() -> [u64; NAMES.len()] {
        std::array::from_fn(|i| COUNTS[i].load(Relaxed))
    }
}

/// Root arguments the producer cannot express, counted under bounded names.
///
/// A root argument that names a space, a sampler table or a constants write the
/// translated root signature cannot address is a refusal, never a silent drop.
/// The counters are logged at powers of two, so a title that repeats one every
/// frame does not fill the log.
pub mod root_argument_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The argument names a register space past
    /// [`bindings::MAX_ROOT_SPACE`](crate::native_metal::bindings::MAX_ROOT_SPACE).
    pub const UNKNOWN_SPACE: usize = 0;
    /// A sampler table is not in the sampler space.
    pub const SAMPLER_NON_SAMPLER_SPACE: usize = 1;
    /// A constants write reaches past the block the translated pipeline declared.
    pub const CONSTANTS_BEYOND_BLOCK: usize = 2;
    /// A root-argument plan is already at its bounded capacity.
    pub const PLAN_FULL: usize = 3;
    /// A root-argument record is otherwise malformed.
    pub const MALFORMED: usize = 4;

    /// The static counter names, in the order of the constants above.
    pub const NAMES: [&str; 5] = [
        "root_argument_unknown_space",
        "root_argument_sampler_non_sampler_space",
        "root_argument_constants_beyond_block",
        "root_argument_plan_full",
        "root_argument_malformed",
    ];

    static COUNTS: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];

    /// Counts one refusal and logs the whole census at powers of two.
    pub(crate) fn note(outcome: usize) -> u64 {
        let count = COUNTS[outcome].fetch_add(1, Relaxed) + 1;
        if count.is_power_of_two() {
            let counts = counts();
            tracing::warn!(?counts, "root-argument refusal census");
        }
        count
    }

    /// The counters, in [`NAMES`] order.
    pub fn counts() -> [u64; NAMES.len()] {
        std::array::from_fn(|i| COUNTS[i].load(Relaxed))
    }
}

/// A DXGI index format number as a version 3 element width.
pub(crate) fn index_format(format: u32) -> Result<IndexFormat, ProducerError> {
    match format {
        57 => Ok(IndexFormat::Uint16),
        42 => Ok(IndexFormat::Uint32),
        _ => Err(ProducerError::unsupported_opcode(
            "IASetIndexBuffer: only R16_UINT and R32_UINT are expressed",
            OP_SET_INDEX_BUFFER,
        )),
    }
}

/// A render target requested by `OMSetRenderTargets`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderTarget {
    /// Texture identity.
    pub resource: ResourceId,
    /// Mip level the view renders into.
    pub mip_level: u16,
    /// Array slice the view renders into.
    pub slice: u16,
    /// Load action at pass start.
    pub load: LoadOp,
    /// Store action at pass end.
    pub store: StoreOp,
    /// Clear color, used when `load == LoadOp::Clear`.
    pub clear_color: [f64; 4],
}

impl RenderTarget {
    /// A target that preserves and stores its contents.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            mip_level: 0,
            slice: 0,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear_color: [0.0; 4],
        }
    }

    /// A target cleared to `clear_color` when the pass opens.
    pub fn cleared(resource: ResourceId, clear_color: [f64; 4]) -> Self {
        Self {
            load: LoadOp::Clear,
            clear_color,
            ..Self::new(resource)
        }
    }

    fn attachment(&self) -> ColorAttachment {
        ColorAttachment {
            allocation_id: self.resource.allocation_id,
            generation: self.resource.generation,
            mip_level: self.mip_level,
            slice: self.slice,
            load: self.load,
            store: self.store,
            clear_color: self.clear_color,
        }
    }
}

/// A depth/stencil target requested by `OMSetRenderTargets`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthTarget {
    /// Texture identity.
    pub resource: ResourceId,
    /// Mip level the view renders into.
    pub mip_level: u16,
    /// Array slice the view renders into.
    pub slice: u16,
    /// Depth load action.
    pub depth_load: LoadOp,
    /// Depth store action.
    pub depth_store: StoreOp,
    /// Depth clear value.
    pub clear_depth: f32,
    /// Stencil load action.
    pub stencil_load: LoadOp,
    /// Stencil store action.
    pub stencil_store: StoreOp,
    /// Stencil clear value.
    pub clear_stencil: u32,
}

impl DepthTarget {
    /// A depth target that preserves and stores its contents.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            mip_level: 0,
            slice: 0,
            depth_load: LoadOp::Load,
            depth_store: StoreOp::Store,
            clear_depth: 1.0,
            stencil_load: LoadOp::Load,
            stencil_store: StoreOp::Store,
            clear_stencil: 0,
        }
    }

    fn attachment(&self) -> DepthStencilAttachment {
        DepthStencilAttachment {
            allocation_id: self.resource.allocation_id,
            generation: self.resource.generation,
            mip_level: self.mip_level,
            slice: self.slice,
            depth_load: self.depth_load,
            depth_store: self.depth_store,
            clear_depth: self.clear_depth,
            stencil_load: self.stencil_load,
            stencil_store: self.stencil_store,
            clear_stencil: self.clear_stencil,
        }
    }
}

/// One `IASetVertexBuffer` view as the title passed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexBufferView {
    /// Guest address of the first element.
    pub address: u64,
    /// Byte length visible to the shader.
    pub size: u32,
    /// Byte stride between elements.
    pub stride: u32,
}

/// Why the producer or its sink refused a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProducerError {
    /// A resource identity or guest address was zero.
    InvalidResource {
        /// Operation that named it.
        operation: &'static str,
    },
    /// The list identity must be `(nonzero, nonzero)`.
    InvalidListIdentity,
    /// The producer cannot express the D3D12 operation.
    Unsupported {
        /// D3D12 operation, for the log and the report.
        operation: &'static str,
        /// Wire opcode that would be needed, when one exists.
        opcode: Option<u16>,
    },
    /// A binding named an allocation the host has not registered.
    ///
    /// The operation is supported; the guest backing is not visible to the
    /// host, so the submission the host would receive could only be refused.
    /// The title sees `E_FAIL`, not `E_NOTIMPL`.
    Unregistered {
        /// Operation whose binding named it.
        operation: &'static str,
    },
    /// A binding's guest range could not be resolved to a share allocation.
    ///
    /// The operation is supported; the guest address names no range the
    /// resolver can pin, so the submission the host would receive could only
    /// be refused. The title sees `E_FAIL`, not `E_NOTIMPL`.
    Unresolved {
        /// Operation whose binding could not be resolved.
        operation: &'static str,
    },
    /// A draw or dispatch ran without a pipeline of that stage.
    MissingPipeline {
        /// Stage the command needs (`graphics`, `compute`, `mesh`).
        stage: &'static str,
    },
    /// A draw was recorded with no render targets bound.
    NoRenderTargets {
        /// Operation that needed them.
        operation: &'static str,
    },
    /// More color attachments than one pass can carry.
    TooManyColorAttachments(usize),
    /// More barriers than one command can carry.
    TooManyBarriers(usize),
    /// The submission command budget is full.
    TooManyCommands(usize),
    /// The session sequence counter wrapped.
    SequenceExhausted,
    /// The sink answered a different sequence.
    ReplySequenceMismatch {
        /// Sequence the producer sent.
        expected: u32,
        /// Sequence the sink answered.
        actual: u32,
    },
    /// The sink completed the submission with a failure status.
    SinkRejected {
        /// Sequence the producer sent.
        sequence: u32,
        /// Status the sink reported.
        status: ReplyStatus,
    },
}

impl ProducerError {
    /// An unsupported operation that needs no specific wire opcode.
    pub fn unsupported(operation: &'static str) -> Self {
        Self::Unsupported {
            operation,
            opcode: None,
        }
    }

    /// A binding whose allocation the host has not registered: `E_FAIL`,
    /// because the operation is supported and the backing is what is missing.
    pub fn unregistered(operation: &'static str) -> Self {
        Self::Unregistered { operation }
    }

    /// A binding whose guest range the resolver could not map: `E_FAIL`,
    /// because the operation is supported and the backing is what is missing.
    pub fn unresolved(operation: &'static str) -> Self {
        Self::Unresolved { operation }
    }

    /// An unsupported operation that would need `opcode`.
    pub fn unsupported_opcode(operation: &'static str, opcode: u16) -> Self {
        Self::Unsupported {
            operation,
            opcode: Some(opcode),
        }
    }

    /// The wire opcode a refused call needed, when there is one.
    ///
    /// The queue records this in the unsupported-call histogram at the moment
    /// of refusal, so a list that fails `Close` still names its opcode.
    pub const fn opcode(&self) -> Option<u16> {
        match self {
            Self::Unsupported { opcode, .. } => *opcode,
            _ => None,
        }
    }

    /// The D3D12 operation a refusal names, when it names one.
    ///
    /// The named-refusal telemetry uses it so calls without a wire opcode are
    /// still countable by name instead of collapsing into one bucket.
    pub const fn operation(&self) -> Option<&'static str> {
        match self {
            Self::InvalidResource { operation }
            | Self::Unsupported { operation, .. }
            | Self::Unregistered { operation }
            | Self::Unresolved { operation }
            | Self::NoRenderTargets { operation } => Some(operation),
            _ => None,
        }
    }

    /// The static submission-refusal reason this variant is counted under.
    ///
    /// The names come from the fixed
    /// [`telemetry::SUBMIT_REFUSAL_NAMES`](crate::native_metal::telemetry::SUBMIT_REFUSAL_NAMES)
    /// table, so a refused submission is never counted without one: a
    /// `ReplySequenceMismatch` is the sequence-mismatch reason and a
    /// `SinkRejected` reply is the host-refused one, everything else keeps its
    /// own `producer_*` name.
    pub const fn submit_refusal_name(&self) -> &'static str {
        use crate::native_metal::telemetry::submit_refusals as names;
        match self {
            Self::InvalidResource { .. } => names::PRODUCER_INVALID_RESOURCE,
            Self::InvalidListIdentity => names::PRODUCER_INVALID_LIST_IDENTITY,
            Self::Unsupported { .. } => names::PRODUCER_UNSUPPORTED,
            Self::Unregistered { .. } => names::PRODUCER_UNREGISTERED,
            Self::Unresolved { .. } => names::PRODUCER_UNRESOLVED,
            Self::MissingPipeline { .. } => names::PRODUCER_MISSING_PIPELINE,
            Self::NoRenderTargets { .. } => names::PRODUCER_NO_RENDER_TARGETS,
            Self::TooManyColorAttachments(_) => names::PRODUCER_TOO_MANY_COLOR_ATTACHMENTS,
            Self::TooManyBarriers(_) => names::PRODUCER_TOO_MANY_BARRIERS,
            Self::TooManyCommands(_) => names::PRODUCER_TOO_MANY_COMMANDS,
            Self::SequenceExhausted => names::PRODUCER_SEQUENCE_EXHAUSTED,
            Self::ReplySequenceMismatch { .. } => names::SEQUENCE_MISMATCH,
            Self::SinkRejected { .. } => names::HOST_REFUSED,
        }
    }

    /// The D3D12 HRESULT the guest entry point returns to the title.
    ///
    /// Unsupported operations are `E_NOTIMPL` (`0x8000_4001`), everything
    /// else `E_FAIL` (`0x8000_4005`); the title never falls back to
    /// D3D12/D3DMetal.
    pub const fn hresult(&self) -> i32 {
        match self {
            Self::Unsupported { .. } => 0x8000_4001_u32 as i32,
            _ => 0x8000_4005_u32 as i32,
        }
    }
}

impl fmt::Display for ProducerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResource { operation } => {
                write!(f, "{operation}: resource identity is zero")
            }
            Self::InvalidListIdentity => write!(f, "list identity must be nonzero"),
            Self::Unsupported { operation, opcode } => match opcode {
                Some(opcode) => write!(f, "{operation} is not expressible (opcode {opcode})"),
                None => write!(f, "{operation} is not expressible"),
            },
            Self::Unregistered { operation } => write!(
                f,
                "{operation}: the binding's allocation is not registered with the host"
            ),
            Self::Unresolved { operation } => write!(
                f,
                "{operation}: the binding could not be resolved to a guest allocation"
            ),
            Self::MissingPipeline { stage } => write!(f, "a {stage} command has no pipeline"),
            Self::NoRenderTargets { operation } => {
                write!(f, "{operation} has no render targets bound")
            }
            Self::TooManyColorAttachments(count) => {
                write!(
                    f,
                    "{count} color attachments exceed {MAX_COLOR_ATTACHMENTS}"
                )
            }
            Self::TooManyBarriers(count) => {
                write!(f, "{count} barriers exceed {MAX_BARRIERS_V3}")
            }
            Self::TooManyCommands(count) => {
                write!(f, "{count} commands exceed {MAX_COMMANDS_V3}")
            }
            Self::SequenceExhausted => write!(f, "the submission sequence wrapped"),
            Self::ReplySequenceMismatch { expected, actual } => {
                write!(f, "reply sequence {actual} for submission {expected}")
            }
            Self::SinkRejected { sequence, status } => {
                write!(f, "submission {sequence} failed: {status:?}")
            }
        }
    }
}

impl std::error::Error for ProducerError {}

/// Where a finished submission goes.
///
/// The real IPC sink is `WIRE-V3-HOST-BRIDGE`'s task; this crate keeps no
/// socket code, so the trait is the only seam.
pub trait WireV3Sink {
    /// Carries one submission to its execution host.
    fn submit(&mut self, submission: &SubmissionV3) -> Result<Reply, ProducerError>;

    /// Carries a submission together with the versioned root-argument plan.
    ///
    /// The default keeps a sink that predates the plan working unchanged: it
    /// forwards the submission and ignores the plan, exactly as an older host
    /// keeps decoding the version 3 frame it already knows. A transport that
    /// does carry the plan encodes it with
    /// [`bindings::encode_root_arguments`](crate::native_metal::bindings::encode_root_arguments).
    fn submit_with_root_arguments(
        &mut self,
        submission: &SubmissionV3,
        _root_arguments: &[RootArgument],
    ) -> Result<Reply, ProducerError> {
        self.submit(submission)
    }
}

/// Sink that keeps every submission, for tests and diagnostics.
#[derive(Debug, Default)]
pub struct RecordingSink {
    submissions: Vec<SubmissionV3>,
    root_arguments: Vec<Vec<RootArgument>>,
    status: Option<ReplyStatus>,
}

impl RecordingSink {
    /// A sink that answers `Ok`.
    pub fn new() -> Self {
        Self::default()
    }

    /// A sink that answers `status` to the next submission.
    pub fn with_status(status: ReplyStatus) -> Self {
        Self {
            submissions: Vec::new(),
            root_arguments: Vec::new(),
            status: Some(status),
        }
    }

    /// The submissions received so far.
    pub fn submissions(&self) -> &[SubmissionV3] {
        &self.submissions
    }

    /// The root-argument plan that arrived with submission `index`, if the
    /// sink was handed one.
    pub fn root_arguments(&self, index: usize) -> Option<&[RootArgument]> {
        self.root_arguments.get(index).map(Vec::as_slice)
    }

    /// Takes the submissions received so far.
    pub fn take(&mut self) -> Vec<SubmissionV3> {
        std::mem::take(&mut self.submissions)
    }
}

impl WireV3Sink for RecordingSink {
    fn submit(&mut self, submission: &SubmissionV3) -> Result<Reply, ProducerError> {
        self.submissions.push(submission.clone());
        self.root_arguments.push(Vec::new());
        Ok(Reply {
            sequence: submission.sequence,
            status: self.status.take().unwrap_or(ReplyStatus::Ok),
        })
    }

    fn submit_with_root_arguments(
        &mut self,
        submission: &SubmissionV3,
        root_arguments: &[RootArgument],
    ) -> Result<Reply, ProducerError> {
        self.submissions.push(submission.clone());
        self.root_arguments.push(root_arguments.to_vec());
        Ok(Reply {
            sequence: submission.sequence,
            status: self.status.take().unwrap_or(ReplyStatus::Ok),
        })
    }
}

/// Viewport/scissor state of the pass being recorded.
#[derive(Debug, Clone, Default)]
struct PassState {
    colors: Vec<ColorAttachment>,
    depth_stencil: Option<DepthStencilAttachment>,
}

/// A clear recorded before a pass can carry it.
///
/// Keyed by [`ResourceId`] in recording order: a later clear of the same
/// resource replaces the earlier one, because only the value at the time the
/// clear executes is observable. Clears of different resources are independent,
/// so their relative order does not change the result.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PendingClear {
    Color {
        resource: ResourceId,
        clear_color: [f64; 4],
    },
    Depth {
        resource: ResourceId,
        clear_depth: Option<f32>,
        clear_stencil: Option<u32>,
    },
}

/// Graphics state kept across draws, emitted in a fixed order.
#[derive(Debug, Default)]
struct GraphicsState {
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    topology: Option<PrimitiveTopology>,
    pipeline: Option<u64>,
    vertex_buffers: [Option<VertexBuffer>; MAX_VERTEX_SLOTS],
    index: Option<IndexBufferBinding>,
    constants: BTreeMap<u16, Vec<u8>>,
    views: BTreeMap<u16, RootBufferView>,
    tables: BTreeMap<(u16, u16), DescriptorTableV3>,
    /// Last `SetPixelShaderDepthOrder` value, kept for a consistent read.
    pixel_shader_depth_order: Option<u32>,
    /// Last `SetShadingRate` value, kept for a consistent read.
    shading_rate: Option<u32>,
}

/// Which parts of [`GraphicsState`] still need a command in the current pass.
#[derive(Debug, Default)]
struct GraphicsDirty {
    viewport: bool,
    scissor: bool,
    topology: bool,
    pipeline: bool,
    vertex_buffers: bool,
    index: bool,
    constants: BTreeSet<u16>,
    views: BTreeSet<u16>,
    tables: BTreeSet<(u16, u16)>,
}

impl GraphicsDirty {
    fn all(state: &GraphicsState) -> Self {
        Self {
            viewport: state.viewport.is_some(),
            scissor: state.scissor.is_some(),
            topology: state.topology.is_some(),
            pipeline: state.pipeline.is_some(),
            vertex_buffers: state.vertex_buffers.iter().any(Option::is_some),
            index: state.index.is_some(),
            constants: state.constants.keys().copied().collect(),
            views: state.views.keys().copied().collect(),
            tables: state.tables.keys().copied().collect(),
        }
    }

    fn any(&self) -> bool {
        self.viewport
            || self.scissor
            || self.topology
            || self.pipeline
            || self.vertex_buffers
            || self.index
            || !self.constants.is_empty()
            || !self.views.is_empty()
            || !self.tables.is_empty()
    }
}

/// Which root signature a list committed (`SetGraphicsRootSignature` or
/// `SetComputeRootSignature`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootSignatureStage {
    /// The graphics root signature.
    Graphics,
    /// The compute root signature.
    Compute,
}

/// The D3D12 command list as a Wire v3 producer.
///
/// One producer records one list generation. [`flush`](Self::flush) yields
/// the submission for the commands recorded since the previous flush; the
/// list identity and session sequence are carried with it.
#[derive(Debug)]
pub struct MetalProducer {
    identity: ListIdentity,
    pass: PassState,
    pass_open: bool,
    pending_clears: Vec<PendingClear>,
    graphics: GraphicsState,
    dirty: GraphicsDirty,
    graphics_root_signature: Option<u64>,
    compute_root_signature: Option<u64>,
    compute_pipeline: Option<u64>,
    mesh_pipeline: Option<u64>,
    compute_constants: BTreeMap<u16, Vec<u8>>,
    compute_views: BTreeMap<u16, RootBufferView>,
    compute_tables: BTreeMap<(u16, u16), DescriptorTableV3>,
    compute_local_size: Option<[u32; 3]>,
    emitted_compute_pipeline: Option<u64>,
    emitted_mesh_pipeline: Option<u64>,
    commands: Vec<CommandV3>,
    /// Every root argument recorded so far, with the register space the
    /// translated root signature binds it in. This is the versioned binding
    /// plan the submission seam carries; the wire's own root commands keep
    /// their existing slot/register shape byte-for-byte.
    root_arguments: Vec<crate::native_metal::bindings::RootArgument>,
    /// Declared roots-constant blocks, per stage: `(graphics, root parameter,
    /// space)` to the translated block's dword count.
    root_constant_blocks: BTreeMap<(bool, u16, u16), u32>,
}

impl MetalProducer {
    /// Starts a producer for one recorded list generation.
    pub fn new(list_id: u64, list_generation: u32) -> Result<Self, ProducerError> {
        Ok(Self {
            identity: checked_identity(list_id, list_generation)?,
            pass: PassState::default(),
            pass_open: false,
            pending_clears: Vec::new(),
            graphics: GraphicsState::default(),
            dirty: GraphicsDirty::default(),
            graphics_root_signature: None,
            compute_root_signature: None,
            compute_pipeline: None,
            mesh_pipeline: None,
            compute_constants: BTreeMap::new(),
            compute_views: BTreeMap::new(),
            compute_tables: BTreeMap::new(),
            compute_local_size: None,
            emitted_compute_pipeline: None,
            emitted_mesh_pipeline: None,
            commands: Vec::new(),
            root_arguments: Vec::new(),
            root_constant_blocks: BTreeMap::new(),
        })
    }

    /// Identity carried by this producer's submissions.
    pub fn identity(&self) -> ListIdentity {
        self.identity
    }

    /// Commands recorded and not yet flushed.
    pub fn commands(&self) -> &[CommandV3] {
        &self.commands
    }

    /// Every root argument recorded so far, in the order the title set it.
    ///
    /// The records carry the register space each argument is bound in, which
    /// the wire's root commands cannot express on their own. Same-slot
    /// arguments in different spaces stay distinct entries.
    pub fn root_arguments(&self) -> &[RootArgument] {
        &self.root_arguments
    }

    /// Encodes [`root_arguments`](Self::root_arguments) as the versioned
    /// `(root_parameter, space, kind, register/offset, size)` plan.
    pub fn encode_root_arguments(&self) -> Result<Vec<u8>, RootArgumentError> {
        crate::native_metal::bindings::encode_root_arguments(&self.root_arguments)
    }

    /// Declares the translated root-constants block a pipeline gives
    /// `(root_parameter, space)` on a stage.
    ///
    /// A later `Set*Root32BitConstants` that reaches past it is refused by
    /// name instead of updating bytes the translated shader never reads.
    pub fn declare_root_constant_block(
        &mut self,
        stage: RootSignatureStage,
        root_parameter: u16,
        space: u16,
        dwords: u32,
    ) {
        self.root_constant_blocks.insert(
            (stage == RootSignatureStage::Graphics, root_parameter, space),
            dwords,
        );
    }

    fn record_root_argument(&mut self, argument: RootArgument) -> Result<(), ProducerError> {
        if self.root_arguments.len() >= MAX_ROOT_ARGUMENTS {
            root_argument_census::note(root_argument_census::PLAN_FULL);
            return Err(ProducerError::unsupported(
                "a root-argument plan is already at its capacity",
            ));
        }
        if let Err(error) = argument.validate() {
            return Err(refuse_root_argument(error));
        }
        self.root_arguments.push(argument);
        Ok(())
    }

    fn check_declared_block(
        &self,
        stage: RootSignatureStage,
        root_parameter: u16,
        space: u16,
        offset: u32,
        words: usize,
    ) -> Result<(), ProducerError> {
        let key = (
            stage == RootSignatureStage::Graphics,
            root_parameter,
            space,
        );
        let Some(&dwords) = self.root_constant_blocks.get(&key) else {
            return Ok(());
        };
        crate::native_metal::bindings::check_constant_block(
            root_parameter,
            space,
            offset,
            words,
            dwords,
        )
        .map_err(refuse_root_argument)
    }

    /// Whether a render pass is open (commands are inside it).
    pub fn has_open_pass(&self) -> bool {
        self.pass_open
    }

    /// Clears everything recorded without producing a submission.
    ///
    /// Used when a refused call has already poisoned the list: `Close`
    /// returns the failure instead of submitting a partial stream.
    pub fn abandon(&mut self) {
        self.commands.clear();
        self.pass_open = false;
        self.pass.colors.clear();
        self.pass.depth_stencil = None;
        self.pending_clears.clear();
        self.emitted_compute_pipeline = None;
        self.emitted_mesh_pipeline = None;
        self.root_arguments.clear();
    }

    fn ensure_room(&self, extra: usize) -> Result<(), ProducerError> {
        let total = self.commands.len().saturating_add(extra);
        if total > MAX_COMMANDS_V3 {
            return Err(ProducerError::TooManyCommands(total));
        }
        Ok(())
    }

    fn push(&mut self, command: CommandV3) -> Result<(), ProducerError> {
        self.ensure_room(1)?;
        self.commands.push(command);
        Ok(())
    }

    // ---------------------------------------------------------------- pass

    /// `OMSetRenderTargets`: records the pass's attachments.
    ///
    /// `BeginRenderPass` is emitted by the first draw, so clears and fixed
    /// function state recorded after this call fold into the pass. Re-binding
    /// the same attachments does not split an open pass; different
    /// attachments (or a different depth target) do.
    ///
    /// A clear recorded before this call for one of the attachments becomes
    /// that attachment's load action (see the module's clears section). A
    /// recorded set that never opened still owes its clears, so they are
    /// encoded before the new attachments replace it.
    ///
    /// An empty `colors` slice unbinds the targets and closes any open pass.
    pub fn set_render_targets(
        &mut self,
        colors: &[RenderTarget],
        depth: Option<DepthTarget>,
    ) -> Result<(), ProducerError> {
        if colors.len() > MAX_COLOR_ATTACHMENTS {
            return Err(ProducerError::TooManyColorAttachments(colors.len()));
        }
        for color in colors {
            ResourceId::new(color.resource.allocation_id, color.resource.generation)?;
        }
        if let Some(depth) = depth {
            ResourceId::new(depth.resource.allocation_id, depth.resource.generation)?;
            if colors.is_empty() {
                return Err(ProducerError::unsupported_opcode(
                    "OMSetRenderTargets: depth without a color attachment",
                    OP_BEGIN_RENDER_PASS,
                ));
            }
        }

        let mut next: Vec<ColorAttachment> = colors.iter().map(RenderTarget::attachment).collect();
        let mut next_depth = depth.map(|depth| depth.attachment());

        if self.pass_open {
            if !same_attachments(
                &self.pass.colors,
                &next,
                &self.pass.depth_stencil,
                &next_depth,
            ) {
                self.close_pass()?;
            } else {
                // Same render targets: the pass continues. Its load/store
                // actions were latched when it opened.
                return Ok(());
            }
        } else {
            // The recorded set is about to be replaced. A clear it carries and
            // never executed still has to happen.
            self.materialize_latent_clear()?;
        }

        self.apply_pending_clears(&mut next, &mut next_depth);
        self.pass.colors = next;
        self.pass.depth_stencil = next_depth;
        Ok(())
    }

    /// Alias of [`set_render_targets`](Self::set_render_targets) named after
    /// the D3D12 call it derives from.
    pub fn begin_render_pass(
        &mut self,
        colors: &[RenderTarget],
        depth: Option<DepthTarget>,
    ) -> Result<(), ProducerError> {
        self.set_render_targets(colors, depth)
    }

    /// Closes the active render pass, if one is open.
    pub fn end_render_pass(&mut self) -> Result<(), ProducerError> {
        self.close_pass()
    }

    fn close_pass(&mut self) -> Result<(), ProducerError> {
        if !self.pass_open {
            return Ok(());
        }
        self.push(CommandV3::EndRenderPass)?;
        self.pass_open = false;
        // The pass stored its attachments. A later pass over the same set loads
        // them instead of repeating the clear its load action already applied.
        for color in &mut self.pass.colors {
            color.load = LoadOp::Load;
        }
        if let Some(depth) = &mut self.pass.depth_stencil {
            depth.depth_load = LoadOp::Load;
            depth.stencil_load = LoadOp::Load;
        }
        Ok(())
    }

    /// `ClearRenderTargetView`: clears `target`, whether or not it is bound.
    ///
    /// A target already recorded for the pass folds the clear into that pass's
    /// load action. A clear inside an open pass closes it and is encoded as its
    /// own pass (see the module's clears section). A target that is not
    /// recorded at all is remembered until a bind applies it or the stream
    /// reaches a point where it is encoded as a standalone clear pass.
    pub fn clear_render_target(
        &mut self,
        target: ResourceId,
        clear_color: [f32; 4],
    ) -> Result<(), ProducerError> {
        ResourceId::new(target.allocation_id, target.generation)?;
        let clear_color = clear_color.map(f64::from);
        let bound = self.pass.colors.iter().position(|color| {
            color.allocation_id == target.allocation_id && color.generation == target.generation
        });
        match bound {
            Some(_) if self.pass_open => {
                // The target is an attachment of the pass that is already
                // recording: close it, encode the clear, and let the next draw
                // reopen the same attachments with `LoadOp::Load`.
                self.close_pass()?;
                self.push_color_clear_pass(target, clear_color)
            }
            Some(index) => {
                let color = &mut self.pass.colors[index];
                color.load = LoadOp::Clear;
                color.store = StoreOp::Store;
                color.clear_color = clear_color;
                Ok(())
            }
            None => {
                if self.pass_open {
                    self.close_pass()?;
                }
                self.record_pending_color(target, clear_color);
                Ok(())
            }
        }
    }

    /// `ClearDepthStencilView`: clears the depth and/or stencil plane of
    /// `target`, whether or not it is bound.
    ///
    /// `clear_depth` and `clear_stencil` follow the D3D12 clear flags. A target
    /// that is the recorded depth attachment folds the clear into the pass's
    /// load actions; a clear inside an open pass closes it and is encoded as
    /// its own pass; any other target is remembered and applied on bind or
    /// encoded later. That later pass needs a colour attachment to anchor it,
    /// because the wire has no depth-only pass.
    pub fn clear_depth_stencil(
        &mut self,
        target: ResourceId,
        clear_depth: bool,
        depth: f32,
        clear_stencil: bool,
        stencil: u8,
    ) -> Result<(), ProducerError> {
        if !clear_depth && !clear_stencil {
            return Ok(());
        }
        ResourceId::new(target.allocation_id, target.generation)?;
        let bound = self.pass.depth_stencil.is_some_and(|attachment| {
            attachment.allocation_id == target.allocation_id
                && attachment.generation == target.generation
        });
        if bound && !self.pass_open {
            let attachment = self
                .pass
                .depth_stencil
                .as_mut()
                .expect("the target is the recorded depth attachment");
            if clear_depth {
                attachment.depth_load = LoadOp::Clear;
                attachment.depth_store = StoreOp::Store;
                attachment.clear_depth = depth;
            }
            if clear_stencil {
                attachment.stencil_load = LoadOp::Clear;
                attachment.stencil_store = StoreOp::Store;
                attachment.clear_stencil = u32::from(stencil);
            }
            return Ok(());
        }
        if bound {
            self.close_pass()?;
            return self.push_depth_clear_pass(
                target,
                clear_depth.then_some(depth),
                clear_stencil.then_some(u32::from(stencil)),
            );
        }
        if self.pass_open {
            self.close_pass()?;
        }
        self.record_pending_depth(
            target,
            clear_depth.then_some(depth),
            clear_stencil.then_some(u32::from(stencil)),
        );
        Ok(())
    }

    // ------------------------------------------------------- fixed state

    /// `RSSetViewports` for the single viewport the wire expresses.
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        near: f32,
        far: f32,
    ) -> Result<(), ProducerError> {
        if !(x.is_finite() && y.is_finite() && width.is_finite() && height.is_finite()) {
            return Err(ProducerError::unsupported_opcode(
                "RSSetViewports: non-finite viewport",
                OP_SET_VIEWPORT,
            ));
        }
        if width <= 0.0 || height <= 0.0 {
            return Err(ProducerError::unsupported_opcode(
                "RSSetViewports: empty viewport",
                OP_SET_VIEWPORT,
            ));
        }
        self.graphics.viewport = Some(Viewport {
            x: f64::from(x),
            y: f64::from(y),
            width: f64::from(width),
            height: f64::from(height),
            near: f64::from(near),
            far: f64::from(far),
        });
        self.dirty.viewport = true;
        Ok(())
    }

    /// `RSSetScissorRects` for the single rectangle the wire expresses.
    pub fn set_scissor_edges(
        &mut self,
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    ) -> Result<(), ProducerError> {
        if left < 0 || top < 0 || right <= left || bottom <= top {
            return Err(ProducerError::unsupported_opcode(
                "RSSetScissorRects: rectangle the wire cannot express",
                OP_SET_SCISSOR,
            ));
        }
        self.graphics.scissor = Some(ScissorRect {
            x: left as u32,
            y: top as u32,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
        });
        self.dirty.scissor = true;
        Ok(())
    }

    /// `IASetPrimitiveTopology` with a D3D12 topology number.
    ///
    /// Every D3D12 value is accepted: the five Wire v3 topologies are recorded
    /// as before, `UNDEFINED` leaves the recorded topology alone, and the
    /// non-expressible values (adjacency, patch lists, reserved gaps) are
    /// counted hints rather than a refusal of the whole list.
    pub fn set_topology_value(&mut self, topology: u32) -> Result<(), ProducerError> {
        match primitive_topology(topology)? {
            TopologyChange::Map(topology) => self.set_topology(topology),
            TopologyChange::Undefined => {
                hint_census::note(hint_census::TOPOLOGY_UNDEFINED);
                Ok(())
            }
            TopologyChange::Named(_) => {
                hint_census::note(hint_census::TOPOLOGY_HINTS);
                Ok(())
            }
        }
    }

    /// `SetPixelShaderDepthOrder`: a hint Wire v3 carries no command for.
    ///
    /// The value is recorded so a later read sees what the title set, and the
    /// call is counted; it never refuses and never drops a draw.
    pub fn set_pixel_shader_depth_order(&mut self, order: u32) -> Result<(), ProducerError> {
        self.graphics.pixel_shader_depth_order = Some(order);
        hint_census::note(hint_census::DEPTH_ORDER_HINTS);
        Ok(())
    }

    /// The last `SetPixelShaderDepthOrder` value, when the title set one.
    pub fn pixel_shader_depth_order(&self) -> Option<u32> {
        self.graphics.pixel_shader_depth_order
    }

    /// `SetShadingRate`: a hint Wire v3 carries no command for.
    ///
    /// Recorded and counted like [`set_pixel_shader_depth_order`], never a
    /// refusal.
    pub fn set_shading_rate(&mut self, flags: u32) -> Result<(), ProducerError> {
        self.graphics.shading_rate = Some(flags);
        hint_census::note(hint_census::SHADING_RATE_HINTS);
        Ok(())
    }

    /// The last `SetShadingRate` value, when the title set one.
    pub fn shading_rate(&self) -> Option<u32> {
        self.graphics.shading_rate
    }

    /// `IASetPrimitiveTopology` with a version 3 topology.
    pub fn set_topology(&mut self, topology: PrimitiveTopology) -> Result<(), ProducerError> {
        self.graphics.topology = Some(topology);
        self.dirty.topology = true;
        Ok(())
    }

    /// `SetPipelineState`: the stage is chosen by the first draw or dispatch
    /// that uses it, as in D3D12.
    pub fn set_pipeline(&mut self, pipeline_id: u64) -> Result<(), ProducerError> {
        if pipeline_id == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "SetPipelineState",
            });
        }
        self.graphics.pipeline = Some(pipeline_id);
        self.dirty.pipeline = true;
        Ok(())
    }

    /// `IASetVertexBuffers`: binds `views` from `start_slot`.
    pub fn set_vertex_buffers_at(
        &mut self,
        start_slot: u32,
        views: &[VertexBufferView],
    ) -> Result<(), ProducerError> {
        let start = usize::try_from(start_slot).map_err(|_| ProducerError::InvalidResource {
            operation: "IASetVertexBuffers",
        })?;
        if start
            .checked_add(views.len())
            .is_none_or(|end| end > MAX_VERTEX_SLOTS)
        {
            return Err(ProducerError::InvalidResource {
                operation: "IASetVertexBuffers",
            });
        }
        for view in views {
            if view.size == 0 || view.stride == 0 {
                return Err(ProducerError::unsupported_opcode(
                    "IASetVertexBuffers: empty buffer or stride",
                    OP_SET_VERTEX_BUFFERS,
                ));
            }
            ResourceId::from_guest_address(view.address)?;
        }
        for (offset, view) in views.iter().enumerate() {
            self.graphics.vertex_buffers[start + offset] = Some(VertexBuffer {
                allocation_id: view.address,
                generation: 1,
                offset: 0,
                stride: view.stride,
                length: u64::from(view.size),
            });
        }
        self.dirty.vertex_buffers = true;
        Ok(())
    }

    /// `IASetVertexBuffers` with null views: unbinds the slots.
    ///
    /// The wire has no unbind, so a clear before the pass opens simply drops
    /// the slots; a clear inside a pass is refused rather than silently
    /// leaving the old buffer bound.
    pub fn clear_vertex_buffers(
        &mut self,
        start_slot: u32,
        count: u32,
    ) -> Result<(), ProducerError> {
        let start = usize::try_from(start_slot).map_err(|_| ProducerError::InvalidResource {
            operation: "IASetVertexBuffers(null)",
        })?;
        let count = usize::try_from(count).map_err(|_| ProducerError::InvalidResource {
            operation: "IASetVertexBuffers(null)",
        })?;
        if start.saturating_add(count) > MAX_VERTEX_SLOTS {
            return Err(ProducerError::unsupported_opcode(
                "IASetVertexBuffers(null): slot out of range",
                OP_SET_VERTEX_BUFFERS,
            ));
        }
        if self.pass_open {
            return Err(ProducerError::unsupported_opcode(
                "IASetVertexBuffers(null) inside a recorded pass",
                OP_SET_VERTEX_BUFFERS,
            ));
        }
        for slot in &mut self.graphics.vertex_buffers[start..start + count] {
            *slot = None;
        }
        self.dirty.vertex_buffers = true;
        Ok(())
    }

    /// `IASetVertexBuffers` with views already resolved to wire bindings.
    ///
    /// The guest resolves each title view's GPU address through the
    /// `GuestRangeResolver` seam (allocation id, generation, exact in-page
    /// offset) and requires the host to have registered that allocation, so the
    /// host's `binding_within_allocation` check is satisfied at execution. The
    /// allocation id is never the address: an unregistered or unresolved range
    /// is refused before anything is recorded.
    pub fn set_vertex_buffers_bound(
        &mut self,
        start_slot: u32,
        views: &[VertexBuffer],
    ) -> Result<(), ProducerError> {
        let start = usize::try_from(start_slot).map_err(|_| ProducerError::InvalidResource {
            operation: "IASetVertexBuffers",
        })?;
        if start
            .checked_add(views.len())
            .is_none_or(|end| end > MAX_VERTEX_SLOTS)
        {
            return Err(ProducerError::InvalidResource {
                operation: "IASetVertexBuffers",
            });
        }
        for view in views {
            if view.length == 0 || view.stride == 0 {
                return Err(ProducerError::unsupported_opcode(
                    "IASetVertexBuffers: empty buffer or stride",
                    OP_SET_VERTEX_BUFFERS,
                ));
            }
            ResourceId::new(view.allocation_id, view.generation)?;
        }
        for (offset, view) in views.iter().enumerate() {
            self.graphics.vertex_buffers[start + offset] = Some(*view);
        }
        self.dirty.vertex_buffers = true;
        Ok(())
    }

    /// `IASetIndexBuffer` with a view already resolved to a wire binding.
    pub fn set_index_buffer_bound(
        &mut self,
        binding: IndexBufferBinding,
    ) -> Result<(), ProducerError> {
        if binding.length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "IASetIndexBuffer: empty buffer",
                OP_SET_INDEX_BUFFER,
            ));
        }
        ResourceId::new(binding.allocation_id, binding.generation)?;
        self.graphics.index = Some(binding);
        self.dirty.index = true;
        Ok(())
    }

    /// `IASetIndexBuffer(null)`.
    ///
    /// D3D12 clears the index-buffer *view*, but the title's state block keeps
    /// using the last non-null binding: after nulling the view it issues
    /// `DrawIndexed` and expects the previously bound buffer to be read. The
    /// producer therefore remembers the last binding instead of dropping it,
    /// and emits no wire command (Wire v3 has no unbind). A draw is refused by
    /// name only when no index buffer was ever bound in this list, which is the
    /// one case where there is genuinely nothing to read. The call is counted
    /// as [`hint_census::DRAW_LIST_INDEX_UNBIND`].
    pub fn unbind_index_buffer(&mut self) -> Result<(), ProducerError> {
        // Keep `self.graphics.index`: a null view is not a buffer release. The
        // binding was already emitted for the current pass (or is re-emitted
        // when the next pass opens), so a following `DrawIndexed` reads it.
        hint_census::note(hint_census::DRAW_LIST_INDEX_UNBIND);
        Ok(())
    }

    /// `IASetIndexBuffer`.
    pub fn set_index_buffer_at(
        &mut self,
        address: u64,
        size: u32,
        format: u32,
    ) -> Result<(), ProducerError> {
        if size == 0 {
            return Err(ProducerError::unsupported_opcode(
                "IASetIndexBuffer: empty buffer",
                OP_SET_INDEX_BUFFER,
            ));
        }
        ResourceId::from_guest_address(address)?;
        self.graphics.index = Some(IndexBufferBinding {
            allocation_id: address,
            generation: 1,
            offset: 0,
            length: u64::from(size),
            format: index_format(format)?,
        });
        self.dirty.index = true;
        Ok(())
    }

    /// `SetGraphicsRoot32BitConstants`: merges the words into the slot.
    ///
    /// The translated graphics root signature puts per-stage user data in
    /// [`USER_DATA_SPACE`](victory_shader::emit::USER_DATA_SPACE); the call
    /// keeps its existing slot behaviour and records that space with it.
    pub fn set_graphics_root_constants(
        &mut self,
        slot: u32,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        self.set_graphics_root_constants_at(
            slot,
            victory_shader::emit::USER_DATA_SPACE as u16,
            values,
            offset,
        )
    }

    /// `SetGraphicsRoot32BitConstants` in an explicit register space.
    ///
    /// The space is the one the translated root signature binds the block in.
    /// It is recorded with the argument and checked against any block a
    /// pipeline declared; the wire command itself keeps the title's slot
    /// exactly as before.
    pub fn set_graphics_root_constants_at(
        &mut self,
        slot: u32,
        space: u16,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        let slot_u16 = u16::try_from(slot).map_err(|_| {
            ProducerError::unsupported_opcode(
                "SetGraphicsRoot32BitConstants: slot out of range",
                OP_ROOT_CONSTANTS,
            )
        })?;
        self.check_declared_block(
            RootSignatureStage::Graphics,
            slot_u16,
            space,
            offset,
            values.len(),
        )?;
        let merge = merge_constants(&self.graphics.constants, slot, values, offset)?;
        self.graphics.constants.insert(slot_u16, merge);
        self.dirty.constants.insert(slot_u16);
        let argument = RootArgument::constants(slot_u16, space, offset, values.len())
            .map_err(refuse_root_argument)?;
        self.record_root_argument(argument)
    }

    /// `SetComputeRoot32BitConstants`: merges the words into the slot.
    ///
    /// Root constants are inline root arguments, not buffer bindings: they are
    /// emitted before every dispatch so each compute segment carries them.
    pub fn set_compute_root_constants(
        &mut self,
        slot: u32,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        self.set_compute_root_constants_at(
            slot,
            victory_shader::emit::USER_DATA_SPACE as u16,
            values,
            offset,
        )
    }

    /// `SetComputeRoot32BitConstants` in an explicit register space.
    pub fn set_compute_root_constants_at(
        &mut self,
        slot: u32,
        space: u16,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        let slot_u16 = u16::try_from(slot).map_err(|_| {
            ProducerError::unsupported_opcode(
                "SetComputeRoot32BitConstants: slot out of range",
                OP_ROOT_CONSTANTS,
            )
        })?;
        self.check_declared_block(
            RootSignatureStage::Compute,
            slot_u16,
            space,
            offset,
            values.len(),
        )?;
        let merge = merge_constants(&self.compute_constants, slot, values, offset)?;
        self.compute_constants.insert(slot_u16, merge);
        let argument = RootArgument::constants(slot_u16, space, offset, values.len())
            .map_err(refuse_root_argument)?;
        self.record_root_argument(argument)
    }

    /// A compute root CBV/SRV/UAV (already resolved to an allocation).
    ///
    /// The guest call names only a GPU virtual address; `list.rs` resolves it
    /// through the `GuestRangeResolver`, pins the backing and refuses an
    /// unresolved or unregistered allocation before this call. A view at slot
    /// `i` is Metal `[[buffer(i)]]`, matching the backend's graphics
    /// convention.
    pub fn set_compute_root_buffer_view(
        &mut self,
        view: RootBufferView,
    ) -> Result<(), ProducerError> {
        self.set_compute_root_buffer_view_at(
            view,
            victory_shader::emit::ROOT_CBV_SPACE as u16,
        )
    }

    /// A compute root view in an explicit register space.
    pub fn set_compute_root_buffer_view_at(
        &mut self,
        view: RootBufferView,
        space: u16,
    ) -> Result<(), ProducerError> {
        ResourceId::new(view.allocation_id, view.generation)?;
        if view.length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "SetComputeRoot*View: empty range",
                OP_SET_ROOT_BUFFER_VIEW,
            ));
        }
        self.compute_views.insert(view.slot, view);
        self.record_root_argument(RootArgument::root_view(view.slot, space, view.length as u32))
    }

    /// A compute descriptor table (already resolved to buffer ranges).
    pub fn set_compute_descriptor_table(
        &mut self,
        table: DescriptorTableV3,
    ) -> Result<(), ProducerError> {
        let space = if table.kind == DescriptorKind::Sampler {
            victory_shader::emit::SAMPLER_SPACE as u16
        } else {
            victory_shader::emit::ROOT_CBV_SPACE as u16
        };
        self.set_compute_descriptor_table_at(table, space)
    }

    /// A compute descriptor table in an explicit register space.
    pub fn set_compute_descriptor_table_at(
        &mut self,
        table: DescriptorTableV3,
        space: u16,
    ) -> Result<(), ProducerError> {
        if table.bindings.is_empty() {
            return Err(ProducerError::unsupported_opcode(
                "SetComputeRootDescriptorTable: empty table",
                OP_DESCRIPTOR_TABLE,
            ));
        }
        for entry in &table.bindings {
            match entry {
                DescriptorTableEntryV3::Buffer(binding) => {
                    ResourceId::new(binding.allocation_id, binding.generation)?;
                    if binding.length == 0 {
                        return Err(ProducerError::unsupported_opcode(
                            "SetComputeRootDescriptorTable: empty range",
                            OP_DESCRIPTOR_TABLE,
                        ));
                    }
                }
                DescriptorTableEntryV3::Texture(handle)
                | DescriptorTableEntryV3::Sampler(handle) => {
                    if *handle == 0 {
                        return Err(ProducerError::unsupported_opcode(
                            "SetComputeRootDescriptorTable: zero resource handle",
                            OP_DESCRIPTOR_TABLE,
                        ));
                    }
                }
            }
        }
        let kind = if table.kind == DescriptorKind::Sampler {
            RootArgumentKind::SamplerTable
        } else {
            RootArgumentKind::DescriptorTable
        };
        let entries = u32::try_from(table.bindings.len()).unwrap_or(u32::MAX);
        let argument = RootArgument::descriptor_table(
            table.set,
            space,
            table.base_register,
            kind,
            entries,
        );
        self.record_root_argument(argument)?;
        self.compute_tables
            .insert((table.set, table.base_register), table);
        Ok(())
    }

    /// A graphics root buffer view (already resolved to an allocation).
    pub fn set_graphics_root_buffer_view(
        &mut self,
        view: RootBufferView,
    ) -> Result<(), ProducerError> {
        self.set_graphics_root_buffer_view_at(
            view,
            victory_shader::emit::ROOT_CBV_SPACE as u16,
        )
    }

    /// A graphics root view in an explicit register space.
    pub fn set_graphics_root_buffer_view_at(
        &mut self,
        view: RootBufferView,
        space: u16,
    ) -> Result<(), ProducerError> {
        ResourceId::new(view.allocation_id, view.generation)?;
        if view.length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "SetGraphicsRootShaderResourceView: empty range",
                OP_SET_ROOT_BUFFER_VIEW,
            ));
        }
        let length = u32::try_from(view.length).unwrap_or(u32::MAX);
        self.graphics.views.insert(view.slot, view);
        self.dirty.views.insert(view.slot);
        self.record_root_argument(RootArgument::root_view(view.slot, space, length))
    }

    /// A graphics descriptor table (already resolved to buffer ranges).
    pub fn set_graphics_descriptor_table(
        &mut self,
        table: DescriptorTableV3,
    ) -> Result<(), ProducerError> {
        let space = if table.kind == DescriptorKind::Sampler {
            victory_shader::emit::SAMPLER_SPACE as u16
        } else {
            victory_shader::emit::ROOT_CBV_SPACE as u16
        };
        self.set_graphics_descriptor_table_at(table, space)
    }

    /// A graphics descriptor table in an explicit register space.
    pub fn set_graphics_descriptor_table_at(
        &mut self,
        table: DescriptorTableV3,
        space: u16,
    ) -> Result<(), ProducerError> {
        if table.bindings.is_empty() {
            return Err(ProducerError::unsupported_opcode(
                "SetGraphicsRootDescriptorTable: empty table",
                OP_DESCRIPTOR_TABLE,
            ));
        }
        for entry in &table.bindings {
            match entry {
                DescriptorTableEntryV3::Buffer(binding) => {
                    ResourceId::new(binding.allocation_id, binding.generation)?;
                    if binding.length == 0 {
                        return Err(ProducerError::unsupported_opcode(
                            "SetGraphicsRootDescriptorTable: empty range",
                            OP_DESCRIPTOR_TABLE,
                        ));
                    }
                }
                DescriptorTableEntryV3::Texture(handle)
                | DescriptorTableEntryV3::Sampler(handle) => {
                    if *handle == 0 {
                        return Err(ProducerError::unsupported_opcode(
                            "SetGraphicsRootDescriptorTable: zero resource handle",
                            OP_DESCRIPTOR_TABLE,
                        ));
                    }
                }
            }
        }
        let kind = if table.kind == DescriptorKind::Sampler {
            RootArgumentKind::SamplerTable
        } else {
            RootArgumentKind::DescriptorTable
        };
        let entries = u32::try_from(table.bindings.len()).unwrap_or(u32::MAX);
        let argument = RootArgument::descriptor_table(
            table.set,
            space,
            table.base_register,
            kind,
            entries,
        );
        self.record_root_argument(argument)?;
        let key = (table.set, table.base_register);
        self.graphics.tables.insert(key, table);
        self.dirty.tables.insert(key);
        Ok(())
    }

    // ------------------------------------------------------------ draws

    /// `DrawInstanced`.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    ) -> Result<(), ProducerError> {
        if vertex_count == 0 || instance_count == 0 {
            // D3D12 records a draw, but it produces nothing.
            return Ok(());
        }
        self.require_graphics_pipeline()?;
        self.open_pass("draw")?;
        self.flush_graphics_state()?;
        self.push(CommandV3::DrawInstanced(DrawInstanced {
            vertex_count_per_instance: vertex_count,
            instance_count,
            start_vertex,
            start_instance,
        }))
    }

    /// `DrawIndexedInstanced`.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    ) -> Result<(), ProducerError> {
        if index_count == 0 || instance_count == 0 {
            return Ok(());
        }
        if self.graphics.index.is_none() {
            // No `IASetIndexBuffer` ever bound a buffer in this list: the draw
            // has nothing to read. `IASetIndexBuffer(null)` after a real bind
            // keeps that bind (see `unbind_index_buffer`), so it does not land
            // here.
            return Err(ProducerError::unsupported_opcode(
                "DrawIndexed: no index buffer was ever bound",
                OP_DRAW_INDEXED_INSTANCED,
            ));
        }
        self.require_graphics_pipeline()?;
        self.open_pass("draw")?;
        self.flush_graphics_state()?;
        self.push(CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
            index_count_per_instance: index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
        }))
    }

    // -------------------------------------------------------- indirect

    /// Validates one GPU-resident argument range, `stride` bytes per record.
    fn checked_indirect(
        arguments: &IndirectArguments,
        stride: u32,
        opcode: u16,
    ) -> Result<(), ProducerError> {
        ResourceId::new(arguments.allocation_id, arguments.generation)?;
        if arguments.max_count == 0 || arguments.max_count > MAX_INDIRECT_COMMANDS_V3 {
            return Err(ProducerError::unsupported_opcode(
                "ExecuteIndirect: max command count outside 1..=256",
                opcode,
            ));
        }
        if !arguments.offset.is_multiple_of(4) {
            return Err(ProducerError::unsupported_opcode(
                "ExecuteIndirect: argument offset is not 4-byte aligned",
                opcode,
            ));
        }
        if u64::from(arguments.max_count)
            .checked_mul(u64::from(stride))
            .is_none()
        {
            return Err(ProducerError::unsupported_opcode(
                "ExecuteIndirect: argument range overflows",
                opcode,
            ));
        }
        if let Some(count) = arguments.count {
            ResourceId::new(count.allocation_id, count.generation)?;
            if !count.offset.is_multiple_of(4) {
                return Err(ProducerError::unsupported_opcode(
                    "ExecuteIndirect: count offset is not 4-byte aligned",
                    opcode,
                ));
            }
        }
        Ok(())
    }

    /// `ExecuteIndirect` of a draw signature (record stride 16).
    pub fn execute_indirect_draw(
        &mut self,
        arguments: IndirectArguments,
    ) -> Result<(), ProducerError> {
        Self::checked_indirect(&arguments, 16, OP_EXECUTE_INDIRECT_DRAW)?;
        self.require_graphics_pipeline()?;
        self.open_pass("draw")?;
        self.flush_graphics_state()?;
        self.push(CommandV3::ExecuteIndirectDraw(ExecuteIndirectDraw {
            arguments,
        }))
    }

    /// `ExecuteIndirect` of an indexed draw signature (record stride 20).
    pub fn execute_indirect_draw_indexed(
        &mut self,
        arguments: IndirectArguments,
    ) -> Result<(), ProducerError> {
        Self::checked_indirect(&arguments, 20, OP_EXECUTE_INDIRECT_DRAW_INDEXED)?;
        self.require_graphics_pipeline()?;
        self.open_pass("draw")?;
        self.flush_graphics_state()?;
        self.push(CommandV3::ExecuteIndirectDrawIndexed(
            ExecuteIndirectDrawIndexed { arguments },
        ))
    }

    /// `ExecuteIndirect` of a mesh dispatch signature (record stride 12).
    ///
    /// The host encodes an indirect mesh dispatch on a live
    /// `MTLRenderCommandEncoder` inside a render pass, exactly like a direct
    /// [`dispatch_mesh`](Self::dispatch_mesh): this call opens the pass when
    /// none is open and keeps an already open pass, so the indirect dispatch
    /// executes between `BeginRenderPass` and the matching `EndRenderPass`.
    /// `SetMeshPipeline` is emitted before it unless it is already current.
    ///
    /// The host drops its graphics pipeline selection at `SetMeshPipeline`, so
    /// the dirty pipeline is flagged here and a following draw re-emits
    /// `SetGraphicsPipeline` into the same pass. A pass already open for other
    /// draws is joined, not split.
    pub fn execute_indirect_dispatch_mesh(
        &mut self,
        pipeline_id: u64,
        arguments: IndirectArguments,
    ) -> Result<(), ProducerError> {
        Self::checked_indirect(&arguments, 12, OP_EXECUTE_INDIRECT_DISPATCH_MESH)?;
        if pipeline_id == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "ExecuteIndirect(mesh)",
            });
        }
        self.open_pass("ExecuteIndirect(mesh)")?;
        if self.emitted_mesh_pipeline != Some(pipeline_id) {
            self.push(CommandV3::SetMeshPipeline(PipelineRef { pipeline_id }))?;
            self.emitted_mesh_pipeline = Some(pipeline_id);
        }
        // The host drops its graphics pipeline selection at `SetMeshPipeline`:
        // a following draw must re-emit `SetGraphicsPipeline` in this pass.
        self.dirty.pipeline = true;
        self.push(CommandV3::ExecuteIndirectDispatchMesh(
            ExecuteIndirectDispatchMesh {
                pipeline_id,
                arguments,
            },
        ))
    }

    /// `ExecuteIndirect` of a compute dispatch signature (record stride 12).
    pub fn execute_indirect_dispatch(
        &mut self,
        arguments: IndirectArguments,
    ) -> Result<(), ProducerError> {
        Self::checked_indirect(&arguments, 12, OP_EXECUTE_INDIRECT_DISPATCH)?;
        let Some(threads_per_group) = self.compute_local_size else {
            return Err(ProducerError::unsupported_opcode(
                "ExecuteIndirect(dispatch): compute threadgroup size not set",
                OP_EXECUTE_INDIRECT_DISPATCH,
            ));
        };
        if threads_per_group.iter().any(|&dimension| dimension == 0) {
            return Err(ProducerError::unsupported_opcode(
                "ExecuteIndirect(dispatch): zero threadgroup size",
                OP_EXECUTE_INDIRECT_DISPATCH,
            ));
        }
        let Some(pipeline_id) = self.compute_pipeline else {
            return Err(ProducerError::MissingPipeline { stage: "compute" });
        };
        self.flush_pending_clears()?;
        self.close_pass()?;
        if self.emitted_compute_pipeline != Some(pipeline_id) {
            self.push(CommandV3::SetComputePipeline(PipelineRef { pipeline_id }))?;
            self.emitted_compute_pipeline = Some(pipeline_id);
        }
        self.flush_compute_state()?;
        self.note_unbound_dispatch();
        self.push(CommandV3::ExecuteIndirectDispatch(ExecuteIndirectDispatch {
            pipeline_id,
            arguments,
            threads_per_group,
        }))
    }

    // -------------------------------------------------------- dispatches

    /// `Dispatch`: the local threadgroup size must be known (set from the
    /// compute pipeline). A dispatch with an unknown size is refused rather
    /// than guessed.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<(), ProducerError> {
        if x == 0 || y == 0 || z == 0 {
            return Ok(());
        }
        let Some(pipeline_id) = self.compute_pipeline else {
            return Err(ProducerError::MissingPipeline { stage: "compute" });
        };
        let Some(threads_per_group) = self.compute_local_size else {
            return Err(ProducerError::unsupported_opcode(
                "Dispatch: compute threadgroup size not set",
                OP_DISPATCH_3D,
            ));
        };
        if threads_per_group.iter().any(|&dimension| dimension == 0) {
            return Err(ProducerError::unsupported_opcode(
                "Dispatch: zero threadgroup size",
                OP_DISPATCH_3D,
            ));
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        if self.emitted_compute_pipeline != Some(pipeline_id) {
            self.push(CommandV3::SetComputePipeline(PipelineRef { pipeline_id }))?;
            self.emitted_compute_pipeline = Some(pipeline_id);
        }
        self.flush_compute_state()?;
        let bindings = self.compute_bindings();
        self.note_unbound_dispatch();
        self.push(CommandV3::Dispatch3d(Dispatch3d {
            pipeline_id,
            group_counts: [x, y, z],
            threads_per_group,
            bindings,
        }))
    }

    /// The local threads per threadgroup of the compute pipeline.
    pub fn set_compute_threadgroup_size(&mut self, size: [u32; 3]) -> Result<(), ProducerError> {
        if size.iter().any(|&dimension| dimension == 0) {
            return Err(ProducerError::unsupported_opcode(
                "SetPipelineState: zero compute threadgroup size",
                OP_DISPATCH_3D,
            ));
        }
        self.compute_local_size = Some(size);
        Ok(())
    }

    /// `DispatchMesh` with the current mesh pipeline.
    ///
    /// The host records a mesh dispatch inside a render pass, so this call
    /// opens the pass like a draw does: `set_render_targets` followed by a
    /// mesh dispatch records `BeginRenderPass`, `SetMeshPipeline`,
    /// `DispatchMesh` and, at `flush`/`Close`, `EndRenderPass` even when no
    /// `Draw` occurs. A pass already open is kept: a later draw joins it.
    ///
    /// Group counts follow the host's rules: every dimension must be nonzero
    /// and their product must fit `u64`. A dispatch with no targets bound or
    /// no mesh pipeline is refused with a named reason.
    pub fn dispatch_mesh(&mut self, x: u32, y: u32, z: u32) -> Result<(), ProducerError> {
        if x == 0 || y == 0 || z == 0 {
            return Err(ProducerError::unsupported_opcode(
                "DispatchMesh: zero group count",
                OP_DISPATCH_MESH,
            ));
        }
        if u64::from(x)
            .checked_mul(u64::from(y))
            .and_then(|groups| groups.checked_mul(u64::from(z)))
            .is_none()
        {
            return Err(ProducerError::unsupported_opcode(
                "DispatchMesh: group counts overflow u64",
                OP_DISPATCH_MESH,
            ));
        }
        let Some(pipeline_id) = self.mesh_pipeline else {
            return Err(ProducerError::MissingPipeline { stage: "mesh" });
        };
        self.open_pass("DispatchMesh")?;
        if self.emitted_mesh_pipeline != Some(pipeline_id) {
            self.push(CommandV3::SetMeshPipeline(PipelineRef { pipeline_id }))?;
            self.emitted_mesh_pipeline = Some(pipeline_id);
        }
        // The host drops its graphics pipeline selection at `SetMeshPipeline`:
        // a following draw must re-emit `SetGraphicsPipeline`.
        self.dirty.pipeline = true;
        self.push(CommandV3::DispatchMesh(DispatchMesh {
            pipeline_id,
            group_counts: [x, y, z],
            bindings: Vec::new(),
        }))
    }

    /// The mesh pipeline selected so far, if any.
    pub fn mesh_pipeline(&self) -> Option<u64> {
        self.mesh_pipeline
    }

    /// Selects the mesh pipeline used by [`dispatch_mesh`](Self::dispatch_mesh).
    pub fn set_mesh_pipeline(&mut self, pipeline_id: u64) -> Result<(), ProducerError> {
        if pipeline_id == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "SetPipelineState(mesh)",
            });
        }
        self.mesh_pipeline = Some(pipeline_id);
        Ok(())
    }

    /// Selects the compute pipeline used by [`dispatch`](Self::dispatch).
    pub fn set_compute_pipeline(&mut self, pipeline_id: u64) -> Result<(), ProducerError> {
        if pipeline_id == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "SetPipelineState(compute)",
            });
        }
        self.compute_pipeline = Some(pipeline_id);
        Ok(())
    }

    /// Records `SetGraphicsRootSignature`/`SetComputeRootSignature`.
    ///
    /// Wire v3 has no root-signature command: the host binds a PSO's own root
    /// signature when that PSO is registered (kind 4/129), so the commit
    /// selects which guest layout the following root arguments describe. It
    /// is kept on the producer instead of dropped, so the choice survives for
    /// a later PSO comparison (a zero pointer unbinds it, as D3D12 allows).
    pub fn commit_root_signature(
        &mut self,
        stage: RootSignatureStage,
        signature: u64,
    ) -> Result<(), ProducerError> {
        let slot = match stage {
            RootSignatureStage::Graphics => &mut self.graphics_root_signature,
            RootSignatureStage::Compute => &mut self.compute_root_signature,
        };
        *slot = (signature != 0).then_some(signature);
        Ok(())
    }

    /// The signature committed for `stage`, `None` when unbound.
    pub const fn root_signature(&self, stage: RootSignatureStage) -> Option<u64> {
        match stage {
            RootSignatureStage::Graphics => self.graphics_root_signature,
            RootSignatureStage::Compute => self.compute_root_signature,
        }
    }

    // ------------------------------------------------------ copies/barriers

    /// `CopyBufferRegion`.
    pub fn copy_buffer(
        &mut self,
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        length: u64,
    ) -> Result<(), ProducerError> {
        if length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "CopyBufferRegion: zero bytes",
                OP_COPY_BUFFER,
            ));
        }
        ResourceId::new(src.allocation_id, src.generation)?;
        ResourceId::new(dst.allocation_id, dst.generation)?;
        if src_offset.checked_add(length).is_none() || dst_offset.checked_add(length).is_none() {
            return Err(ProducerError::unsupported_opcode(
                "CopyBufferRegion: range overflow",
                OP_COPY_BUFFER,
            ));
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::CopyBuffer(CopyBuffer {
            src_allocation_id: src.allocation_id,
            src_generation: src.generation,
            src_offset,
            dst_allocation_id: dst.allocation_id,
            dst_generation: dst.generation,
            dst_offset,
            length,
        }))
    }

    /// `CopyTextureRegion` of one whole mip/slice.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture(
        &mut self,
        src: ResourceId,
        src_mip: u16,
        src_slice: u16,
        dst: ResourceId,
        dst_mip: u16,
        dst_slice: u16,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<(), ProducerError> {
        ResourceId::new(src.allocation_id, src.generation)?;
        ResourceId::new(dst.allocation_id, dst.generation)?;
        if width == 0 || height == 0 || depth == 0 {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: empty extent",
                OP_COPY_TEXTURE,
            ));
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::CopyTexture(CopyTexture {
            src_allocation_id: src.allocation_id,
            src_generation: src.generation,
            src_mip,
            src_slice,
            dst_allocation_id: dst.allocation_id,
            dst_generation: dst.generation,
            dst_mip,
            dst_slice,
            width,
            height,
            depth,
        }))
    }

    /// `CopyTextureRegion` from a guest buffer footprint into a texture.
    ///
    /// The wire carries one pitched copy: `offset` is the first source row,
    /// `bytes_per_row` its pitch and `extent`/`origin` one destination box.
    /// The host encodes a single depth slice per command, so an extent deeper
    /// than one is refused instead of truncated.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer_to_texture(
        &mut self,
        source: ResourceId,
        destination: TextureSubresource,
        offset: u64,
        bytes_per_row: u32,
        extent: [u32; 3],
        origin: [u32; 3],
    ) -> Result<(), ProducerError> {
        ResourceId::new(source.allocation_id, source.generation)?;
        ResourceId::new(destination.allocation_id, destination.generation)?;
        if extent.iter().any(|&extent| extent == 0) {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: empty buffer-upload box",
                OP_COPY_BUFFER_TO_TEXTURE,
            ));
        }
        if extent[2] != 1 {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: a buffer upload of more than one depth slice",
                OP_COPY_BUFFER_TO_TEXTURE,
            ));
        }
        if bytes_per_row == 0 {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: a buffer upload with no row pitch",
                OP_COPY_BUFFER_TO_TEXTURE,
            ));
        }
        if !offset.is_multiple_of(4) {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: a buffer upload offset is not 4-byte aligned",
                OP_COPY_BUFFER_TO_TEXTURE,
            ));
        }
        self.close_pass()?;
        self.push(CommandV3::CopyBufferToTexture(CopyBufferToTexture {
            allocation_id: source.allocation_id,
            generation: source.generation,
            destination_allocation_id: destination.allocation_id,
            destination_generation: destination.generation,
            offset,
            bytes_per_row,
            // One depth slice per command: no image pitch is needed.
            bytes_per_image: 0,
            extent,
            mip_level: destination.mip_level,
            slice: destination.slice,
            origin,
        }))
    }

    /// `CopyTextureRegion` of an explicit texel box between two textures.
    ///
    /// A buffer footprint is never a texture-to-texture box: uploads lower to
    /// [`copy_buffer_to_texture`](Self::copy_buffer_to_texture), and readbacks
    /// are refused by the caller.
    pub fn copy_texture_region(
        &mut self,
        src: TextureSubresource,
        dst: TextureSubresource,
        src_origin: [u32; 3],
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) -> Result<(), ProducerError> {
        ResourceId::new(src.allocation_id, src.generation)?;
        ResourceId::new(dst.allocation_id, dst.generation)?;
        if size.iter().any(|&extent| extent == 0) {
            return Err(ProducerError::unsupported_opcode(
                "CopyTextureRegion: empty extent",
                OP_COPY_TEXTURE_REGION,
            ));
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::CopyTextureRegion(WireCopyTextureRegion {
            src,
            dst,
            src_origin,
            dst_origin,
            size,
        }))
    }

    /// `ResolveSubresource`: a multisample source into a single-sample
    /// target.
    pub fn resolve_subresource(
        &mut self,
        src: TextureSubresource,
        dst: TextureSubresource,
    ) -> Result<(), ProducerError> {
        ResourceId::new(src.allocation_id, src.generation)?;
        ResourceId::new(dst.allocation_id, dst.generation)?;
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::ResolveSubresource(WireResolveSubresource {
            src,
            dst,
        }))
    }

    /// `ResourceBarrier`: closes the pass first, like D3D12 requires.
    pub fn resource_barrier(&mut self, entries: &[BarrierEntry]) -> Result<(), ProducerError> {
        if entries.is_empty() {
            return Ok(());
        }
        if entries.len() > MAX_BARRIERS_V3 {
            return Err(ProducerError::TooManyBarriers(entries.len()));
        }
        for entry in entries {
            ResourceId::new(entry.allocation_id, entry.generation)?;
            if entry.length == 0 || entry.offset.checked_add(entry.length).is_none() {
                return Err(ProducerError::unsupported_opcode(
                    "ResourceBarrier: empty or overflowing range",
                    OP_RESOURCE_BARRIER,
                ));
            }
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::ResourceBarrier(ResourceBarrier {
            entries: entries.to_vec(),
        }))
    }

    /// `Present`, emitted after closing the pass.
    pub fn present(&mut self, surface_id: u64, sync_interval: u32) -> Result<(), ProducerError> {
        if surface_id == 0 {
            return Err(ProducerError::InvalidResource {
                operation: "Present",
            });
        }
        if sync_interval > 1 {
            return Err(ProducerError::unsupported_opcode(
                "Present: sync interval other than 0 or 1",
                OP_PRESENT,
            ));
        }
        self.flush_pending_clears()?;
        self.close_pass()?;
        self.push(CommandV3::Present(Present {
            surface_id,
            sync_interval,
        }))
    }

    /// Records an explicit refusal without touching the stream.
    ///
    /// `list.rs` uses it for entry points the producer cannot express yet
    /// (indirect execution, subresource copies, ...): the refusal is logged
    /// and surfaces as a failure HRESULT at `Close`, never as a fallback.
    pub fn refuse(
        &mut self,
        operation: &'static str,
        opcode: Option<u16>,
    ) -> Result<(), ProducerError> {
        Err(ProducerError::Unsupported { operation, opcode })
    }

    // ------------------------------------------------------------- flush

    /// Closes the pass and returns the submission recorded so far.
    ///
    /// The submission carries no sequence yet ([`UNASSIGNED_SEQUENCE`]):
    /// the transport assigns the next session-wide sequence when it enqueues
    /// the frame, exactly once, so an empty stream or a submission a list
    /// drops before execution consumes nothing and the host's ledger stays
    /// dense. Everything else about the flush (closing pass, latent clears,
    /// pending clears) is unchanged.
    ///
    /// Clears that no draw or bind consumed are encoded here too: a recorded
    /// pass whose clear never ran is materialised, and a clear of a target that
    /// was never bound becomes its own pass.
    pub fn flush(&mut self) -> Result<SubmissionV3, ProducerError> {
        self.ensure_room(0)?;
        self.close_pass()?;
        self.materialize_latent_clear()?;
        self.flush_pending_clears()?;
        let commands = std::mem::take(&mut self.commands);
        self.emitted_compute_pipeline = None;
        self.emitted_mesh_pipeline = None;
        self.dirty = GraphicsDirty::all(&self.graphics);
        Ok(SubmissionV3 {
            sequence: UNASSIGNED_SEQUENCE,
            list: Some(self.identity),
            commands,
        })
    }

    /// Encodes a submission with the canonical version 3 codec.
    pub fn encode(submission: &SubmissionV3) -> Result<Vec<u8>, WireError> {
        victory_metal::wire_v3::encode_submission_v3(submission)
    }

    /// Flushes and carries the submission to `sink`, checking its reply.
    ///
    /// The direct sink path assigns the sequence here, through the same
    /// single helper the native queue uses at enqueue.
    pub fn submit<S: WireV3Sink + ?Sized>(&mut self, sink: &mut S) -> Result<Reply, ProducerError> {
        let mut submission = self.flush()?;
        ensure_sequence(&mut submission)?;
        let reply = sink.submit_with_root_arguments(&submission, &self.root_arguments)?;
        if reply.sequence != submission.sequence {
            return Err(ProducerError::ReplySequenceMismatch {
                expected: submission.sequence,
                actual: reply.sequence,
            });
        }
        if reply.status != ReplyStatus::Ok {
            return Err(ProducerError::SinkRejected {
                sequence: submission.sequence,
                status: reply.status,
            });
        }
        Ok(reply)
    }

    // ----------------------------------------------------------- lowering

    fn require_graphics_pipeline(&self) -> Result<u64, ProducerError> {
        self.graphics
            .pipeline
            .ok_or(ProducerError::MissingPipeline { stage: "graphics" })
    }

    /// Applies the clear of any attachment `next` binds and forgets it.
    fn apply_pending_clears(
        &mut self,
        colors: &mut [ColorAttachment],
        depth: &mut Option<DepthStencilAttachment>,
    ) {
        for color in colors.iter_mut() {
            let Some(index) = self.pending_clears.iter().position(|clear| {
                matches!(
                    clear,
                    PendingClear::Color { resource, .. }
                        if resource.allocation_id == color.allocation_id
                            && resource.generation == color.generation
                )
            }) else {
                continue;
            };
            let PendingClear::Color { clear_color, .. } = self.pending_clears.remove(index) else {
                continue;
            };
            color.load = LoadOp::Clear;
            color.store = StoreOp::Store;
            color.clear_color = clear_color;
        }
        let Some(depth) = depth.as_mut() else {
            return;
        };
        let Some(index) = self.pending_clears.iter().position(|clear| {
            matches!(
                clear,
                PendingClear::Depth { resource, .. }
                    if resource.allocation_id == depth.allocation_id
                        && resource.generation == depth.generation
            )
        }) else {
            return;
        };
        let PendingClear::Depth {
            clear_depth,
            clear_stencil,
            ..
        } = self.pending_clears.remove(index)
        else {
            return;
        };
        if let Some(value) = clear_depth {
            depth.depth_load = LoadOp::Clear;
            depth.depth_store = StoreOp::Store;
            depth.clear_depth = value;
        }
        if let Some(value) = clear_stencil {
            depth.stencil_load = LoadOp::Clear;
            depth.stencil_store = StoreOp::Store;
            depth.clear_stencil = value;
        }
    }

    /// Remembers a colour clear until a bind consumes it; the last one wins.
    fn record_pending_color(&mut self, resource: ResourceId, clear_color: [f64; 4]) {
        self.pending_clears.retain(|clear| {
            !matches!(clear, PendingClear::Color { resource: pending, .. } if *pending == resource)
        });
        self.pending_clears.push(PendingClear::Color {
            resource,
            clear_color,
        });
    }

    /// Remembers a depth/stencil clear until a bind consumes it.
    ///
    /// Clears of the two planes merge, so clearing depth and then stencil
    /// before the bind clears both.
    fn record_pending_depth(
        &mut self,
        resource: ResourceId,
        clear_depth: Option<f32>,
        clear_stencil: Option<u32>,
    ) {
        let mut merged_depth = clear_depth;
        let mut merged_stencil = clear_stencil;
        self.pending_clears.retain(|clear| match clear {
            PendingClear::Depth {
                resource: pending,
                clear_depth,
                clear_stencil,
            } if *pending == resource => {
                merged_depth = merged_depth.or(*clear_depth);
                merged_stencil = merged_stencil.or(*clear_stencil);
                false
            }
            _ => true,
        });
        self.pending_clears.push(PendingClear::Depth {
            resource,
            clear_depth: merged_depth,
            clear_stencil: merged_stencil,
        });
    }

    /// Encodes one standalone, single-attachment colour clear pass.
    fn push_color_clear_pass(
        &mut self,
        resource: ResourceId,
        clear_color: [f64; 4],
    ) -> Result<(), ProducerError> {
        let attachment = ColorAttachment {
            allocation_id: resource.allocation_id,
            generation: resource.generation,
            mip_level: 0,
            slice: 0,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear_color,
        };
        self.push(CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![attachment],
            depth_stencil: None,
        }))?;
        self.push(CommandV3::EndRenderPass)
    }

    /// Encodes one standalone depth/stencil clear pass.
    ///
    /// The wire has no depth-only pass, so the clear rides on a render pass
    /// whose single colour attachment is the first recorded target: it is
    /// loaded and stored unchanged. The depth attachment must match its size,
    /// as it would in the pass the title recorded it for.
    fn push_depth_clear_pass(
        &mut self,
        resource: ResourceId,
        clear_depth: Option<f32>,
        clear_stencil: Option<u32>,
    ) -> Result<(), ProducerError> {
        let Some(anchor) = self.pass.colors.first().copied() else {
            return Err(ProducerError::unsupported_opcode(
                "ClearDepthStencilView: no colour attachment to anchor the clear pass",
                OP_BEGIN_RENDER_PASS,
            ));
        };
        let color = ColorAttachment {
            load: LoadOp::Load,
            store: StoreOp::Store,
            ..anchor
        };
        let depth = DepthStencilAttachment {
            allocation_id: resource.allocation_id,
            generation: resource.generation,
            mip_level: 0,
            slice: 0,
            depth_load: if clear_depth.is_some() {
                LoadOp::Clear
            } else {
                LoadOp::Load
            },
            depth_store: StoreOp::Store,
            clear_depth: clear_depth.unwrap_or(1.0),
            stencil_load: if clear_stencil.is_some() {
                LoadOp::Clear
            } else {
                LoadOp::Load
            },
            stencil_store: StoreOp::Store,
            clear_stencil: clear_stencil.unwrap_or(0),
        };
        self.push(CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color],
            depth_stencil: Some(depth),
        }))?;
        self.push(CommandV3::EndRenderPass)
    }

    /// Encodes every clear that no bind consumed, in recording order.
    ///
    /// Called before any work that would otherwise be encoded first and at
    /// `flush`, so a clear is never moved after a later command. The depth
    /// check runs before this appends anything: a call that cannot be expressed
    /// leaves no partial stream.
    fn flush_pending_clears(&mut self) -> Result<(), ProducerError> {
        if self.pending_clears.is_empty() {
            return Ok(());
        }
        if self.pass.colors.is_empty()
            && self
                .pending_clears
                .iter()
                .any(|clear| matches!(clear, PendingClear::Depth { .. }))
        {
            return Err(ProducerError::unsupported_opcode(
                "ClearDepthStencilView: no colour attachment to anchor the clear pass",
                OP_BEGIN_RENDER_PASS,
            ));
        }
        self.close_pass()?;
        self.ensure_room(self.pending_clears.len().saturating_mul(2))?;
        let pending = std::mem::take(&mut self.pending_clears);
        for clear in pending {
            match clear {
                PendingClear::Color {
                    resource,
                    clear_color,
                } => self.push_color_clear_pass(resource, clear_color)?,
                PendingClear::Depth {
                    resource,
                    clear_depth,
                    clear_stencil,
                } => self.push_depth_clear_pass(resource, clear_depth, clear_stencil)?,
            }
        }
        Ok(())
    }

    /// Encodes the recorded pass when a clear it carries never got to run.
    ///
    /// A recorded attachment keeps `LoadOp::Clear` only while the pass never
    /// opened: `close_pass` resets a pass that ran to `Load`. Opening and
    /// immediately closing the recorded set therefore executes exactly the
    /// clears that are still owed, with no draws.
    fn materialize_latent_clear(&mut self) -> Result<(), ProducerError> {
        if self.pass_open || self.pass.colors.is_empty() {
            return Ok(());
        }
        let color_clear = self
            .pass
            .colors
            .iter()
            .any(|color| color.load == LoadOp::Clear);
        let depth_clear = self.pass.depth_stencil.is_some_and(|depth| {
            depth.depth_load == LoadOp::Clear || depth.stencil_load == LoadOp::Clear
        });
        if !color_clear && !depth_clear {
            return Ok(());
        }
        self.open_pass("Clear (no following draw)")?;
        self.close_pass()
    }

    fn open_pass(&mut self, operation: &'static str) -> Result<(), ProducerError> {
        if self.pass_open {
            return Ok(());
        }
        if self.pass.colors.is_empty() {
            return Err(ProducerError::NoRenderTargets { operation });
        }
        self.ensure_room(1 + graphics_state_count(&self.graphics))?;
        // A clear that no attachment consumed must execute before this pass's
        // work, never after it.
        self.flush_pending_clears()?;
        self.commands
            .push(CommandV3::BeginRenderPass(BeginRenderPass {
                colors: self.pass.colors.clone(),
                depth_stencil: self.pass.depth_stencil,
            }));
        self.pass_open = true;
        // A new encoder starts with no state: everything bound is re-emitted.
        self.dirty = GraphicsDirty::all(&self.graphics);
        self.emitted_mesh_pipeline = None;
        Ok(())
    }

    fn flush_graphics_state(&mut self) -> Result<(), ProducerError> {
        if !self.dirty.any() {
            return Ok(());
        }
        let dirty = std::mem::take(&mut self.dirty);
        self.ensure_room(graphics_state_count(&self.graphics))?;
        if dirty.viewport
            && let Some(viewport) = self.graphics.viewport
        {
            self.commands.push(CommandV3::SetViewport(viewport));
        }
        if dirty.scissor
            && let Some(scissor) = self.graphics.scissor
        {
            self.commands.push(CommandV3::SetScissor(scissor));
        }
        if dirty.topology
            && let Some(topology) = self.graphics.topology
        {
            self.commands
                .push(CommandV3::SetPrimitiveTopology(topology));
        }
        if dirty.pipeline
            && let Some(pipeline_id) = self.graphics.pipeline
        {
            self.commands
                .push(CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }));
            // The host drops its mesh pipeline selection at
            // `SetGraphicsPipeline`: a following mesh dispatch must re-emit it.
            self.emitted_mesh_pipeline = None;
        }
        if dirty.vertex_buffers {
            for set in vertex_buffer_sets(&self.graphics.vertex_buffers) {
                self.commands.push(CommandV3::SetVertexBuffers(set));
            }
        }
        if dirty.index
            && let Some(index) = self.graphics.index
        {
            self.commands.push(CommandV3::SetIndexBuffer(index));
        }
        for slot in dirty.constants {
            if let Some(bytes) = self.graphics.constants.get(&slot) {
                self.commands
                    .push(CommandV3::SetRootConstants(RootConstantsV3 {
                        slot,
                        bytes: bytes.clone(),
                    }));
            }
        }
        for slot in dirty.views {
            if let Some(view) = self.graphics.views.get(&slot) {
                self.commands.push(CommandV3::SetRootBufferView(*view));
            }
        }
        for register in dirty.tables {
            if let Some(table) = self.graphics.tables.get(&register) {
                self.commands
                    .push(CommandV3::SetDescriptorTable(table.clone()));
            }
        }
        Ok(())
    }

    fn flush_compute_state(&mut self) -> Result<(), ProducerError> {
        // Each dispatch may start a new compute segment (a render pass in
        // between splits them), so every current root argument is re-emitted
        // instead of relying on state from an earlier segment.
        let constants: Vec<(u16, Vec<u8>)> = self
            .compute_constants
            .iter()
            .map(|(slot, bytes)| (*slot, bytes.clone()))
            .collect();
        for (slot, bytes) in constants {
            self.push(CommandV3::SetRootConstants(RootConstantsV3 { slot, bytes }))?;
        }
        let views: Vec<RootBufferView> = self.compute_views.values().copied().collect();
        for view in views {
            self.push(CommandV3::SetRootBufferView(view))?;
        }
        let tables: Vec<DescriptorTableV3> = self.compute_tables.values().cloned().collect();
        for table in tables {
            self.push(CommandV3::SetDescriptorTable(table))?;
        }
        Ok(())
    }

    /// The buffer bindings a dispatch carries in `[[buffer(0)]]` order.
    ///
    /// The wire binds this list at `[[buffer(0)]]`, `[[buffer(1)]]`, ..., so
    /// only the contiguous run of root views from slot 0 can ride on the
    /// dispatch. Every view is also emitted as a stateful `SetRootBufferView`
    /// at its own slot, which covers a sparse set (a descriptor table or a
    /// root constant at an earlier slot) without mis-binding.
    fn compute_bindings(&self) -> Vec<BufferBinding> {
        self.compute_views
            .iter()
            .enumerate()
            .take(MAX_BINDINGS_PER_DISPATCH)
            .take_while(|(index, (slot, _))| usize::from(**slot) == *index)
            .map(|(_, (_, view))| BufferBinding {
                allocation_id: view.allocation_id,
                generation: view.generation,
                offset: view.offset,
                length: view.length,
            })
            .collect()
    }

    /// Makes a dispatch with no recorded buffer binding visible in telemetry.
    ///
    /// The producer cannot see a pipeline's declared root layout, so it cannot
    /// prove a kernel expected a buffer; it can prove the recording produced
    /// none. The named counter keeps that from being a silent empty dispatch
    /// (`Dispatch3d.bindings` empty and no compute descriptor table).
    fn note_unbound_dispatch(&self) {
        if self.compute_views.is_empty() && self.compute_tables.is_empty() {
            crate::native_metal::telemetry::note_compute_dispatch_without_bindings();
        }
    }
}

fn graphics_state_count(state: &GraphicsState) -> usize {
    let mut count = 0;
    if state.viewport.is_some() {
        count += 1;
    }
    if state.scissor.is_some() {
        count += 1;
    }
    if state.topology.is_some() {
        count += 1;
    }
    if state.pipeline.is_some() {
        count += 1;
    }
    count += vertex_buffer_sets(&state.vertex_buffers).len();
    if state.index.is_some() {
        count += 1;
    }
    count + state.constants.len() + state.views.len() + state.tables.len()
}

/// Splits bound vertex slots into wire-sized contiguous runs.
fn vertex_buffer_sets(slots: &[Option<VertexBuffer>; MAX_VERTEX_SLOTS]) -> Vec<VertexBufferSet> {
    let mut sets = Vec::new();
    let mut slot = 0;
    while slot < slots.len() {
        if slots[slot].is_none() {
            slot += 1;
            continue;
        }
        let start = slot;
        let mut buffers = Vec::new();
        while slot < slots.len()
            && buffers.len() < MAX_VERTEX_BUFFERS
            && let Some(buffer) = slots[slot]
        {
            buffers.push(buffer);
            slot += 1;
        }
        sets.push(VertexBufferSet {
            start_slot: start as u16,
            buffers,
        });
    }
    sets
}

fn same_attachments(
    colors: &[ColorAttachment],
    next_colors: &[ColorAttachment],
    depth: &Option<DepthStencilAttachment>,
    next_depth: &Option<DepthStencilAttachment>,
) -> bool {
    let same_color = colors.len() == next_colors.len()
        && colors.iter().zip(next_colors).all(|(a, b)| {
            a.allocation_id == b.allocation_id
                && a.generation == b.generation
                && a.mip_level == b.mip_level
                && a.slice == b.slice
        });
    let same_depth = match (depth, next_depth) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.allocation_id == b.allocation_id
                && a.generation == b.generation
                && a.mip_level == b.mip_level
                && a.slice == b.slice
        }
        _ => false,
    };
    same_color && same_depth
}

/// Turns a root-argument refusal into a named producer error and counts it.
///
/// Every case is a call the translated root signature cannot address; the
/// title sees `E_NOTIMPL` with the argument's own operation name instead of a
/// binding that silently reaches the wrong parameter.
fn refuse_root_argument(error: RootArgumentError) -> ProducerError {
    match error {
        RootArgumentError::UnknownSpace { .. } => {
            root_argument_census::note(root_argument_census::UNKNOWN_SPACE);
            ProducerError::unsupported_opcode(
                "Set*Root*: the root argument names an unknown register space",
                OP_ROOT_CONSTANTS,
            )
        }
        RootArgumentError::SamplerNonSamplerSpace { .. } => {
            root_argument_census::note(root_argument_census::SAMPLER_NON_SAMPLER_SPACE);
            ProducerError::unsupported_opcode(
                "Set*RootDescriptorTable: a sampler table is outside the sampler space",
                OP_DESCRIPTOR_TABLE,
            )
        }
        RootArgumentError::ConstantsBeyondBlock { .. } => {
            root_argument_census::note(root_argument_census::CONSTANTS_BEYOND_BLOCK);
            ProducerError::unsupported_opcode(
                "Set*Root32BitConstants: the write is past the declared root-constant block",
                OP_ROOT_CONSTANTS,
            )
        }
        RootArgumentError::TooMany(_) => {
            root_argument_census::note(root_argument_census::PLAN_FULL);
            ProducerError::unsupported("Set*Root*: the root-argument plan is full")
        }
        _ => {
            root_argument_census::note(root_argument_census::MALFORMED);
            ProducerError::unsupported("Set*Root*: the root argument is malformed")
        }
    }
}

fn merge_constants(
    existing: &BTreeMap<u16, Vec<u8>>,
    slot: u32,
    values: &[u32],
    offset: u32,
) -> Result<Vec<u8>, ProducerError> {
    if slot > u32::from(u16::MAX) {
        return Err(ProducerError::unsupported_opcode(
            "SetRoot32BitConstants: slot out of range",
            OP_ROOT_CONSTANTS,
        ));
    }
    if values.is_empty() {
        return Err(ProducerError::unsupported_opcode(
            "SetRoot32BitConstants: no values",
            OP_ROOT_CONSTANTS,
        ));
    }
    let end = (offset as usize)
        .checked_add(values.len())
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProducerError::unsupported_opcode(
            "SetRoot32BitConstants: offset overflow",
            OP_ROOT_CONSTANTS,
        ))?;
    if end > MAX_ROOT_CONSTANT_BYTES_V3 {
        return Err(ProducerError::unsupported_opcode(
            "SetRoot32BitConstants: more than 256 bytes",
            OP_ROOT_CONSTANTS,
        ));
    }
    let mut bytes = existing.get(&(slot as u16)).cloned().unwrap_or_default();
    if bytes.len() < end {
        bytes.resize(end, 0);
    }
    let start = offset as usize * 4;
    for (index, value) in values.iter().enumerate() {
        let at = start + index * 4;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn next_sequence() -> Result<u32, ProducerError> {
    let sequence = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    if sequence == 0 {
        return Err(ProducerError::SequenceExhausted);
    }
    Ok(sequence)
}

/// Assigns the next session-wide sequence to `submission` unless it already
/// carries one.
///
/// This is the only place outside tests where a sequence is handed out.
/// [`MetalProducer::flush`] leaves the field unassigned, so an empty stream or
/// a submission a list drops before execution consumes nothing; both choke
/// points before the wire ([`MetalProducer::submit`] and
/// `NativeQueue::enqueue`) call this, and a submission that was already sent
/// keeps its sequence when it is retried, never renumbering it.
pub fn ensure_sequence(submission: &mut SubmissionV3) -> Result<u32, ProducerError> {
    if submission.sequence != UNASSIGNED_SEQUENCE {
        return Ok(submission.sequence);
    }
    let sequence = next_sequence()?;
    submission.sequence = sequence;
    Ok(sequence)
}

/// Resets the session sequence counter (tests only: it is process-global).
#[cfg(test)]
pub(crate) fn reset_sequences(start: u32) {
    NEXT_SEQUENCE.store(start, Ordering::Relaxed);
}

/// One finished submission and the root-argument plan its producer recorded.
///
/// The plan names the translated `(space, register, kind)` each title
/// root-parameter index feeds, which the wire's root commands cannot express;
/// the bridge encodes it beside the unchanged frame
/// ([`encode_root_arguments`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedSubmission {
    /// The closed list's version 3 submission.
    pub submission: SubmissionV3,
    /// Root arguments recorded while it was recorded, in set order.
    pub root_arguments: Vec<RootArgument>,
}

/// One recorded list on the native path: its producer and the submissions
/// waiting for the bridge.
///
/// `list.rs` creates one when `VICTORY_BACKEND=metal`; `Close` flushes into
/// [`pending`](Self::take) and `Reset` starts the next generation. The
/// queue/bridge task drains the pending submissions; this crate never opens
/// a socket.
#[derive(Debug)]
pub struct MetalList {
    list_id: u64,
    generation: u32,
    producer: MetalProducer,
    pending: Vec<RecordedSubmission>,
    failed: Option<ProducerError>,
    failures: u64,
}

impl MetalList {
    /// Starts a list at generation 1.
    pub fn new(list_id: u64) -> Result<Self, ProducerError> {
        Ok(Self {
            list_id,
            generation: 1,
            producer: MetalProducer::new(list_id, 1)?,
            pending: Vec::new(),
            failed: None,
            failures: 0,
        })
    }

    /// The producer calls record into.
    pub fn producer_mut(&mut self) -> &mut MetalProducer {
        &mut self.producer
    }

    /// Remembers the first refused call for `Close` to return.
    ///
    /// Every refusal is counted and logged; the first eight also carry the
    /// operation name so a run lists what the native path cannot express.
    pub fn note_error(&mut self, error: ProducerError) {
        self.failures += 1;
        if self.failures <= 8 {
            tracing::warn!(%error, "VICTORY_BACKEND=metal: the producer refused a call");
        }
        if self.failed.is_none() {
            self.failed = Some(error);
        }
    }

    /// `Close`: flushes the recorded stream into the pending batch.
    ///
    /// If any call was refused, nothing is submitted and the first refusal
    /// is returned so the guest entry point can return its failure HRESULT.
    /// An empty stream is not kept: it carries no commands and holds no
    /// sequence, so dropping it cannot open a hole in the host's ledger.
    pub fn close(&mut self) -> Result<(), ProducerError> {
        if let Some(error) = self.failed.take() {
            self.producer.abandon();
            return Err(error);
        }
        let submission = self.producer.flush()?;
        if !submission.commands.is_empty() {
            debug_assert_eq!(
                submission.sequence, UNASSIGNED_SEQUENCE,
                "a list closes a submission before the transport assigns its sequence"
            );
            // The plan the list recorded travels with the submission it
            // describes; it is encoded by the bridge, not here.
            let root_arguments = self.producer.root_arguments().to_vec();
            self.pending.push(RecordedSubmission {
                submission,
                root_arguments,
            });
        }
        Ok(())
    }

    /// `Reset`: starts the next list generation and drops submissions the
    /// bridge did not take (the title reset before executing this list).
    ///
    /// The dropped submissions are still unassigned: `Close` never takes a
    /// sequence, the queue does at enqueue, so a reset before execution cannot
    /// desynchronise the host's ledger.
    pub fn reset(&mut self) -> Result<(), ProducerError> {
        if !self.pending.is_empty() {
            tracing::debug!(
                count = self.pending.len(),
                "native submissions dropped by a list reset before the bridge took them"
            );
            debug_assert!(
                self.pending
                    .iter()
                    .all(|recorded| recorded.submission.sequence == UNASSIGNED_SEQUENCE),
                "a pending submission is dropped before the transport assigned it a sequence"
            );
        }
        self.pending.clear();
        self.failed = None;
        let generation = self.generation.wrapping_add(1).max(1);
        self.generation = generation;
        self.producer = MetalProducer::new(self.list_id, generation)?;
        Ok(())
    }

    /// Submissions finished by `Close` and not yet taken by the bridge.
    ///
    /// The root-argument plans are left behind: a caller that only needs the
    /// frames uses this, one that carries the plan uses
    /// [`take_with_root_arguments`](Self::take_with_root_arguments).
    pub fn take(&mut self) -> Vec<SubmissionV3> {
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|recorded| recorded.submission)
            .collect()
    }

    /// Submissions finished by `Close` with the root-argument plan each one
    /// recorded, for the transport that carries it.
    pub fn take_with_root_arguments(&mut self) -> Vec<RecordedSubmission> {
        std::mem::take(&mut self.pending)
    }

    /// The list's current producer generation.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Refused calls since the list was created.
    pub fn failures(&self) -> u64 {
        self.failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GpuBackend;
    use victory_metal::wire_v3::{
        decode_submission_v3, DescriptorKind, IndirectCountBuffer, OP_DESCRIPTOR_TABLE,
        OP_DRAW_INDEXED_INSTANCED, OP_END_RENDER_PASS, OP_EXECUTE_INDIRECT_DISPATCH,
        OP_EXECUTE_INDIRECT_DISPATCH_MESH, OP_EXECUTE_INDIRECT_DRAW,
        OP_EXECUTE_INDIRECT_DRAW_INDEXED, OP_ROOT_CONSTANTS, OP_SET_COMPUTE_PIPELINE,
        OP_SET_GRAPHICS_PIPELINE, OP_SET_MESH_PIPELINE, OP_SET_ROOT_BUFFER_VIEW,
    };

    fn resource(allocation_id: u64) -> ResourceId {
        ResourceId::new(allocation_id, 1).expect("test resource identity")
    }

    fn targets() -> (Vec<RenderTarget>, Option<DepthTarget>) {
        (
            vec![
                RenderTarget::cleared(resource(0x1000), [0.0, 0.0, 0.0, 1.0]),
                RenderTarget::new(resource(0x2000)),
            ],
            Some(DepthTarget::new(resource(0x3000))),
        )
    }

    fn record_draw(producer: &mut MetalProducer) {
        // Every test that records an indexed draw must name an index buffer:
        // an indexed draw with no buffer ever bound is a named refusal now.
        if producer.graphics.index.is_none() {
            producer
                .set_index_buffer_at(0x20_0000, 1024, 57)
                .expect("test index buffer");
        }
        producer
            .draw_indexed(36, 1, 0, 0, 0)
            .expect("indexed draw recorded");
    }

    /// Flushes `producer` and assigns the sequence the transport would, so a
    /// test can encode the frame it just recorded.
    fn flush_for_wire(producer: &mut MetalProducer) -> SubmissionV3 {
        let mut submission = producer.flush().expect("flush");
        ensure_sequence(&mut submission).expect("the transport assigns a sequence");
        submission
    }

    /// The ARM64 host planner (the real encoder-segment rules) accepts the
    /// recorded stream. Only builds where the host exists can check this.
    #[cfg(target_os = "macos")]
    fn assert_host_planner_accepts(commands: &[CommandV3]) {
        victory_metal::macos::render::plan_encoder_segments(commands)
            .expect("the host planner accepts the recorded mesh pass");
    }

    #[test]
    fn recorded_sequence_flushes_in_wire_order() {
        let mut producer = MetalProducer::new(7, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer
            .set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0)
            .expect("viewport");
        producer
            .set_scissor_edges(0, 0, 1920, 1080)
            .expect("scissor");
        producer.set_pipeline(0x900).expect("pipeline");
        producer
            .set_vertex_buffers_at(
                0,
                &[VertexBufferView {
                    address: 0x10_0000,
                    size: 1024,
                    stride: 32,
                }],
            )
            .expect("vertex buffer");
        record_draw(&mut producer);
        producer
            .resource_barrier(&[BarrierEntry {
                allocation_id: 0x2000,
                generation: 1,
                offset: 0,
                length: 64,
                before_state: 0x4,
                after_state: 0x20,
            }])
            .expect("barrier");
        producer
            .copy_buffer(resource(0x4000), 0, resource(0x5000), 0, 256)
            .expect("copy");
        producer.present(3, 1).expect("present");

        let mut submission = producer.flush().expect("flush");
        assert_eq!(
            submission.list,
            Some(ListIdentity {
                list_id: 7,
                list_generation: 1
            })
        );
        assert_eq!(
            submission.sequence, UNASSIGNED_SEQUENCE,
            "flush leaves the sequence to the transport, which assigns it once"
        );
        ensure_sequence(&mut submission).expect("the transport assigns a sequence");
        assert!(submission.sequence >= 1);
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_VIEWPORT,
                OP_SET_SCISSOR,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_VERTEX_BUFFERS,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_RESOURCE_BARRIER,
                OP_COPY_BUFFER,
                OP_PRESENT,
            ]
        );

        match &submission.commands[0] {
            CommandV3::BeginRenderPass(pass) => {
                assert_eq!(pass.colors.len(), 2);
                assert_eq!(pass.colors[0].allocation_id, 0x1000);
                assert_eq!(pass.colors[0].load, LoadOp::Clear);
                assert_eq!(pass.colors[0].clear_color, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(pass.colors[1].allocation_id, 0x2000);
                assert_eq!(pass.colors[1].load, LoadOp::Load);
                assert_eq!(
                    pass.depth_stencil.map(|depth| depth.allocation_id),
                    Some(0x3000)
                );
            }
            other => panic!("first command is {other:?}"),
        }
        match &submission.commands[6] {
            CommandV3::DrawIndexedInstanced(draw) => {
                assert_eq!(draw.index_count_per_instance, 36);
                assert_eq!(draw.instance_count, 1);
            }
            other => panic!("seventh command is {other:?}"),
        }
        match &submission.commands[9] {
            CommandV3::CopyBuffer(copy) => {
                assert_eq!(copy.src_allocation_id, 0x4000);
                assert_eq!(copy.dst_allocation_id, 0x5000);
                assert_eq!(copy.length, 256);
            }
            other => panic!("tenth command is {other:?}"),
        }
        match &submission.commands[10] {
            CommandV3::Present(present) => {
                assert_eq!(present.surface_id, 3);
                assert_eq!(present.sync_interval, 1);
            }
            other => panic!("last command is {other:?}"),
        }
    }

    #[test]
    fn pipeline_changes_do_not_split_the_pass() {
        let mut producer = MetalProducer::new(8, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("first pipeline");
        record_draw(&mut producer);
        producer.set_pipeline(0x901).expect("second pipeline");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_SET_GRAPHICS_PIPELINE,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
    }

    /// A mesh pass the title records as `OMSetRenderTargets` + `DispatchMesh`
    /// with no intervening draw still opens and closes its render pass.
    #[test]
    fn a_mesh_dispatch_opens_and_records_its_render_pass() {
        let mut producer = MetalProducer::new(22, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_mesh_pipeline(0x902).expect("mesh pipeline");
        producer.dispatch_mesh(4, 2, 1).expect("mesh dispatch");
        assert!(
            producer.has_open_pass(),
            "a mesh dispatch opens the pass it needs"
        );

        let mut sink = RecordingSink::new();
        producer.submit(&mut sink).expect("submit");
        assert_eq!(sink.submissions().len(), 1, "one mesh pass, one submission");
        let submission = &sink.submissions()[0];
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_MESH_PIPELINE,
                OP_DISPATCH_MESH,
                OP_END_RENDER_PASS,
            ]
        );
        match &submission.commands[2] {
            CommandV3::DispatchMesh(mesh) => {
                assert_eq!(mesh.pipeline_id, 0x902);
                assert_eq!(mesh.group_counts, [4, 2, 1]);
            }
            other => panic!("the third command is {other:?}"),
        }
        let bytes = MetalProducer::encode(submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), *submission);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A mesh dispatch with no targets to open a pass for is refused, and the
    /// refusal names the operation instead of recording a partial pass.
    #[test]
    fn a_mesh_dispatch_without_render_targets_is_refused_by_name() {
        let mut producer = MetalProducer::new(23, 1).expect("producer");
        producer.set_mesh_pipeline(0x902).expect("mesh pipeline");
        let error = producer
            .dispatch_mesh(1, 1, 1)
            .expect_err("a mesh dispatch needs a render pass");
        assert!(matches!(
            error,
            ProducerError::NoRenderTargets {
                operation: "DispatchMesh"
            }
        ));
        assert!(error.to_string().contains("DispatchMesh"));
        assert!(
            producer.commands().is_empty(),
            "a refusal records no command"
        );
    }

    #[test]
    fn a_mesh_dispatch_without_a_mesh_pipeline_is_refused() {
        let mut producer = MetalProducer::new(24, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        let error = producer
            .dispatch_mesh(1, 1, 1)
            .expect_err("a mesh dispatch needs a mesh pipeline");
        assert!(matches!(
            error,
            ProducerError::MissingPipeline { stage: "mesh" }
        ));
    }

    /// `SetMeshPipeline` clears the host's graphics pipeline selection, so the
    /// draw that follows re-emits `SetGraphicsPipeline` in the same pass.
    #[test]
    fn a_graphics_draw_after_a_mesh_dispatch_keeps_the_pass_open() {
        let mut producer = MetalProducer::new(25, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("graphics pipeline");
        producer.set_mesh_pipeline(0x902).expect("mesh pipeline");
        producer.dispatch_mesh(1, 1, 1).expect("mesh dispatch");
        record_draw(&mut producer);
        assert!(
            producer.has_open_pass(),
            "the draw joins the mesh pass instead of splitting it"
        );

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_MESH_PIPELINE,
                OP_DISPATCH_MESH,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// An indirect mesh dispatch executes inside a render pass: the producer
    /// opens `BeginRenderPass`, emits `SetMeshPipeline` before the indirect
    /// command, and keeps the pass open for a following draw.
    #[test]
    fn an_indirect_mesh_dispatch_opens_the_pass_and_a_draw_joins_it() {
        let mut producer = MetalProducer::new(26, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_mesh_pipeline(0xa00).expect("mesh pipeline");
        producer
            .execute_indirect_dispatch_mesh(0xa00, arguments(3, false))
            .expect("indirect mesh dispatch");
        assert!(
            producer.has_open_pass(),
            "an indirect mesh dispatch opens the pass it executes in"
        );

        producer.set_pipeline(0x900).expect("graphics pipeline");
        record_draw(&mut producer);
        assert!(
            producer.has_open_pass(),
            "the draw joins the indirect mesh pass instead of splitting it"
        );

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_MESH_PIPELINE,
                OP_EXECUTE_INDIRECT_DISPATCH_MESH,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let position = |wanted: u16| {
            opcodes
                .iter()
                .position(|opcode| *opcode == wanted)
                .unwrap_or_else(|| panic!("opcode {wanted} is missing from {opcodes:?}"))
        };
        assert!(
            position(OP_BEGIN_RENDER_PASS) < position(OP_SET_MESH_PIPELINE),
            "the mesh pipeline is selected inside the pass"
        );
        assert!(
            position(OP_SET_MESH_PIPELINE) < position(OP_EXECUTE_INDIRECT_DISPATCH_MESH),
            "SetMeshPipeline must precede ExecuteIndirect"
        );
        assert!(
            position(OP_EXECUTE_INDIRECT_DISPATCH_MESH) < position(OP_DRAW_INDEXED_INSTANCED),
            "the following draw stays in the same pass"
        );
        assert_eq!(
            opcodes
                .iter()
                .filter(|opcode| **opcode == OP_END_RENDER_PASS)
                .count(),
            1,
            "the pass closes once, at flush"
        );
        match &submission.commands[2] {
            CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
                assert_eq!(dispatch.pipeline_id, 0xa00);
                assert_eq!(dispatch.arguments.max_count, 3);
            }
            other => panic!("the third command is {other:?}"),
        }
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// An indirect mesh dispatch into an already open pass joins it: no second
    /// `BeginRenderPass` is recorded.
    #[test]
    fn an_indirect_mesh_dispatch_joins_an_open_pass() {
        let mut producer = MetalProducer::new(27, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("graphics pipeline");
        record_draw(&mut producer);
        assert!(producer.has_open_pass(), "the draw opened the pass");

        producer.set_mesh_pipeline(0xa00).expect("mesh pipeline");
        producer
            .execute_indirect_dispatch_mesh(0xa00, arguments(1, false))
            .expect("indirect mesh dispatch");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes
                .iter()
                .filter(|opcode| **opcode == OP_BEGIN_RENDER_PASS)
                .count(),
            1,
            "the indirect mesh joins the open pass: {opcodes:?}"
        );
        assert_eq!(
            opcodes
                .iter()
                .filter(|opcode| **opcode == OP_END_RENDER_PASS)
                .count(),
            1,
            "one pass, one close: {opcodes:?}"
        );
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// An indirect mesh dispatch with no pipeline handle is refused before it
    /// can open a pass, so no partial stream is recorded.
    #[test]
    fn an_indirect_mesh_dispatch_without_a_pipeline_handle_is_refused() {
        let mut producer = MetalProducer::new(28, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        let error = producer
            .execute_indirect_dispatch_mesh(0, arguments(1, false))
            .expect_err("an indirect mesh dispatch needs a pipeline");
        assert!(matches!(
            error,
            ProducerError::InvalidResource {
                operation: "ExecuteIndirect(mesh)"
            }
        ));
        assert!(!producer.has_open_pass(), "the refusal opens no pass");
        assert!(
            producer.commands().is_empty(),
            "the refusal records no command"
        );
    }

    #[test]
    fn render_target_changes_split_the_pass() {
        let mut producer = MetalProducer::new(9, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);

        let (other, other_depth) = (
            vec![RenderTarget::new(resource(0x6000))],
            Some(DepthTarget::new(resource(0x7000))),
        );
        producer
            .set_render_targets(&other, other_depth)
            .expect("new render targets");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let begins: Vec<&BeginRenderPass> = submission
            .commands
            .iter()
            .filter_map(|command| match command {
                CommandV3::BeginRenderPass(pass) => Some(pass),
                _ => None,
            })
            .collect();
        assert_eq!(begins.len(), 2);
        assert_eq!(begins[0].colors[0].allocation_id, 0x1000);
        assert_eq!(begins[1].colors[0].allocation_id, 0x6000);
    }

    /// The colour attachments of the passes in `commands`, by load action.
    fn passes(commands: &[CommandV3]) -> Vec<&BeginRenderPass> {
        commands
            .iter()
            .filter_map(|command| match command {
                CommandV3::BeginRenderPass(pass) => Some(pass),
                _ => None,
            })
            .collect()
    }

    /// How many encoded passes clear at least one colour attachment.
    fn clear_passes(commands: &[CommandV3]) -> usize {
        passes(commands)
            .into_iter()
            .filter(|pass| pass.colors.iter().any(|color| color.load == LoadOp::Clear))
            .count()
    }

    /// `ClearRenderTargetView` names a descriptor, not a bound slot: a title may
    /// clear before `OMSetRenderTargets`. The clear becomes the pass's load
    /// action and the command list carries no separate clear pass.
    #[test]
    fn a_clear_before_the_bind_clears_only_that_pass() {
        let mut producer = MetalProducer::new(30, 1).expect("producer");
        let clear_color = [0.25, 0.5, 0.75, 1.0];
        producer
            .clear_render_target(resource(0x1000), clear_color)
            .expect("a clear before the bind is expressible");
        assert!(
            producer.commands().is_empty(),
            "a pending clear records no command yet"
        );

        producer
            .set_render_targets(&[RenderTarget::new(resource(0x1000))], None)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes.len(), 1, "the clear folds into the draw's pass");
        assert_eq!(
            clear_passes(&submission.commands),
            1,
            "exactly one clear pass: {opcodes:?}"
        );
        assert_eq!(passes[0].colors.len(), 1);
        assert_eq!(passes[0].colors[0].allocation_id, 0x1000);
        assert_eq!(passes[0].colors[0].load, LoadOp::Clear);
        assert_eq!(passes[0].colors[0].store, StoreOp::Store);
        assert_eq!(passes[0].colors[0].clear_color, clear_color.map(f64::from));
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A clear whose target is never bound and never drawn still takes effect:
    /// it is encoded once as a single-attachment clear pass at flush.
    #[test]
    fn a_clear_with_no_bind_and_no_draw_is_still_encoded_once() {
        let mut producer = MetalProducer::new(31, 1).expect("producer");
        producer
            .clear_render_target(resource(0x1100), [1.0, 0.0, 0.0, 1.0])
            .expect("pending clear");

        let mut submission = producer.flush().expect("flush");
        super::ensure_sequence(&mut submission).expect("sequence");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![OP_BEGIN_RENDER_PASS, OP_END_RENDER_PASS],
            "one standalone clear pass, no draws"
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].colors.len(), 1);
        assert_eq!(passes[0].colors[0].allocation_id, 0x1100);
        assert_eq!(passes[0].colors[0].load, LoadOp::Clear);
        assert_eq!(passes[0].colors[0].store, StoreOp::Store);
        assert_eq!(
            passes[0].colors[0].clear_color,
            [1.0, 0.0, 0.0, 1.0].map(f64::from)
        );
        assert!(passes[0].depth_stencil.is_none());
        assert_eq!(clear_passes(&submission.commands), 1);

        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A clear issued while the target is already in an open pass closes that
    /// pass, encodes the clear as its own pass, and the next draw reopens the
    /// same attachment with `LoadOp::Load`. The clear is never dropped and never
    /// moves after the draw that follows it.
    #[test]
    fn a_clear_inside_an_open_pass_is_encoded_as_its_own_pass() {
        let mut producer = MetalProducer::new(32, 1).expect("producer");
        let target = resource(0x1000);
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);
        assert!(producer.has_open_pass(), "the draw opened the pass");

        producer
            .clear_render_target(target, [0.0, 1.0, 0.0, 1.0])
            .expect("an in-pass clear is expressible");
        assert!(
            !producer.has_open_pass(),
            "the pass closes so the clear keeps its place in the stream"
        );
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ],
            "draw, standalone clear, then the reopened draw"
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[0].colors[0].load, LoadOp::Load);
        assert_eq!(passes[1].colors[0].allocation_id, 0x1000);
        assert_eq!(passes[1].colors[0].load, LoadOp::Clear);
        assert_eq!(passes[1].colors[0].store, StoreOp::Store);
        assert_eq!(
            passes[1].colors[0].clear_color,
            [0.0, 1.0, 0.0, 1.0].map(f64::from)
        );
        assert_eq!(
            passes[2].colors[0].load,
            LoadOp::Load,
            "the reopened pass loads the cleared content"
        );
        assert_eq!(
            clear_passes(&submission.commands),
            1,
            "the clear is encoded once: {opcodes:?}"
        );
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// Clearing and binding the same target twice applies each recorded clear
    /// exactly once: a pending clear is consumed by the bind that uses it and
    /// is not re-emitted as a standalone pass.
    #[test]
    fn clearing_and_binding_the_same_target_twice_clears_once_per_clear() {
        let mut producer = MetalProducer::new(33, 1).expect("producer");
        let target = resource(0x1200);
        producer
            .clear_render_target(target, [1.0, 0.0, 0.0, 1.0])
            .expect("pending red clear");
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("first bind");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);

        producer
            .clear_render_target(target, [0.0, 0.0, 1.0, 1.0])
            .expect("in-pass blue clear");
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("second bind");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[0].colors[0].load, LoadOp::Clear);
        assert_eq!(
            passes[0].colors[0].clear_color,
            [1.0, 0.0, 0.0, 1.0].map(f64::from)
        );
        assert_eq!(passes[1].colors[0].load, LoadOp::Clear);
        assert_eq!(
            passes[1].colors[0].clear_color,
            [0.0, 0.0, 1.0, 1.0].map(f64::from)
        );
        assert_eq!(passes[2].colors[0].load, LoadOp::Load);
        assert_eq!(
            clear_passes(&submission.commands),
            2,
            "one clear per recorded clear, not one per bind: {opcodes:?}"
        );
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A single clear must survive a bind that replaces the recorded set: the
    /// clear is materialised when the second bind arrives, and the draw that
    /// follows loads the cleared content instead of clearing it again.
    #[test]
    fn one_clear_bound_twice_clears_once() {
        let mut producer = MetalProducer::new(37, 1).expect("producer");
        let target = resource(0x1300);
        producer
            .clear_render_target(target, [0.0, 0.5, 1.0, 1.0])
            .expect("pending clear");
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("first bind");
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("second bind");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ],
            "the clear is encoded once, then the draw's pass loads it"
        );
        assert_eq!(
            clear_passes(&submission.commands),
            1,
            "binding the target twice must not clear it twice"
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes[0].colors[0].load, LoadOp::Clear);
        assert_eq!(passes[1].colors[0].load, LoadOp::Load);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// `ClearDepthStencilView` before the bind clears the depth/stencil planes
    /// through the pass's load actions, like the colour clear.
    #[test]
    fn a_depth_clear_before_the_bind_clears_the_pass() {
        let mut producer = MetalProducer::new(34, 1).expect("producer");
        producer
            .clear_depth_stencil(resource(0x3000), true, 0.25, true, 9)
            .expect("a depth clear before the bind is expressible");
        assert!(producer.commands().is_empty());
        producer
            .set_render_targets(
                &[RenderTarget::new(resource(0x1000))],
                Some(DepthTarget::new(resource(0x3000))),
            )
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let depth = passes(&submission.commands)[0]
            .depth_stencil
            .expect("the pass keeps its depth attachment");
        assert_eq!(depth.allocation_id, 0x3000);
        assert_eq!(depth.depth_load, LoadOp::Clear);
        assert_eq!(depth.depth_store, StoreOp::Store);
        assert_eq!(depth.clear_depth, 0.25);
        assert_eq!(depth.stencil_load, LoadOp::Clear);
        assert_eq!(depth.stencil_store, StoreOp::Store);
        assert_eq!(depth.clear_stencil, 9);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A depth/stencil clear inside an open pass closes it and is encoded as
    /// its own pass, anchored on the pass's first colour attachment (the wire
    /// has no depth-only pass).
    #[test]
    fn a_depth_clear_inside_an_open_pass_is_encoded_as_its_own_pass() {
        let mut producer = MetalProducer::new(35, 1).expect("producer");
        producer
            .set_render_targets(
                &[RenderTarget::new(resource(0x1000))],
                Some(DepthTarget::new(resource(0x3000))),
            )
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);
        producer
            .clear_depth_stencil(resource(0x3000), true, 1.0, false, 0)
            .expect("an in-pass depth clear is expressible");
        assert!(!producer.has_open_pass());
        record_draw(&mut producer);

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
            ]
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[1].colors.len(), 1);
        assert_eq!(passes[1].colors[0].allocation_id, 0x1000);
        assert_eq!(
            passes[1].colors[0].load,
            LoadOp::Load,
            "the anchor colour is preserved"
        );
        assert_eq!(passes[1].colors[0].store, StoreOp::Store);
        let depth = passes[1]
            .depth_stencil
            .expect("the clear pass carries the depth target");
        assert_eq!(depth.allocation_id, 0x3000);
        assert_eq!(depth.depth_load, LoadOp::Clear);
        assert_eq!(depth.depth_store, StoreOp::Store);
        assert_eq!(
            depth.stencil_load,
            LoadOp::Load,
            "only the flagged plane clears"
        );
        let reopened = passes[2]
            .depth_stencil
            .expect("the reopened pass keeps its depth attachment");
        assert_eq!(reopened.depth_load, LoadOp::Load);
        assert_eq!(reopened.stencil_load, LoadOp::Load);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// The wire requires 1..=8 colour attachments in every pass, so a depth
    /// clear that has no colour attachment to anchor its pass is refused by
    /// name at the point it would be encoded, not dropped.
    #[test]
    fn a_depth_clear_with_no_colour_attachment_is_refused_by_name() {
        let mut producer = MetalProducer::new(36, 1).expect("producer");
        producer
            .clear_depth_stencil(resource(0x3000), true, 1.0, false, 0)
            .expect("the clear is remembered until the stream needs it");
        assert!(producer.commands().is_empty());

        let error = producer
            .flush()
            .expect_err("the wire has no depth-only pass to carry the clear");
        assert_eq!(
            error,
            ProducerError::Unsupported {
                operation: "ClearDepthStencilView: no colour attachment to anchor the clear pass",
                opcode: Some(OP_BEGIN_RENDER_PASS),
            }
        );
        assert!(
            producer.commands().is_empty(),
            "a refused clear leaves no partial stream"
        );
    }

    /// A clear inside an open pass is not dropped when no draw follows: it is
    /// already encoded as its own pass and only the previous pass had to close.
    #[test]
    fn a_clear_inside_an_open_pass_with_no_following_draw_still_clears() {
        let mut producer = MetalProducer::new(38, 1).expect("producer");
        let target = resource(0x1400);
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);
        producer
            .clear_render_target(target, [0.5, 0.5, 0.5, 1.0])
            .expect("in-pass clear");

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
                OP_SET_INDEX_BUFFER,
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_END_RENDER_PASS,
            ],
            "the clear is encoded even though no draw follows it"
        );
        let passes = passes(&submission.commands);
        assert_eq!(passes[1].colors[0].load, LoadOp::Clear);
        assert_eq!(
            passes[1].colors[0].clear_color,
            [0.5, 0.5, 0.5, 1.0].map(f64::from)
        );
        assert_eq!(clear_passes(&submission.commands), 1);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    /// A recorded attachment cleared but never drawn still clears at flush, and
    /// the pass is emitted exactly once.
    #[test]
    fn a_clear_of_a_bound_target_with_no_draw_is_encoded_once() {
        let mut producer = MetalProducer::new(39, 1).expect("producer");
        let target = resource(0x1500);
        producer
            .set_render_targets(&[RenderTarget::new(target)], None)
            .expect("render targets");
        producer
            .clear_render_target(target, [0.0, 0.0, 0.0, 1.0])
            .expect("recorded clear");

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![OP_BEGIN_RENDER_PASS, OP_END_RENDER_PASS]
        );
        assert_eq!(clear_passes(&submission.commands), 1);
        let passes = passes(&submission.commands);
        assert_eq!(passes[0].colors[0].allocation_id, 0x1500);
        assert_eq!(passes[0].colors[0].load, LoadOp::Clear);
        assert_eq!(passes[0].colors[0].store, StoreOp::Store);
        #[cfg(target_os = "macos")]
        assert_host_planner_accepts(&submission.commands);
    }

    #[test]
    fn an_unsupported_operation_leaves_no_partial_command() {
        let mut producer = MetalProducer::new(10, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        record_draw(&mut producer);
        let before = producer.commands().to_vec();

        let error = producer
            .dispatch(1, 1, 1)
            .expect_err("a dispatch with no compute pipeline is refused");
        assert_eq!(error, ProducerError::MissingPipeline { stage: "compute" });
        let error = producer
            .set_topology_value(65)
            .expect_err("a value outside the D3D12 enum has no topology");
        assert!(matches!(error, ProducerError::Unsupported { .. }));
        let error = producer
            .refuse("ExecuteIndirect", None)
            .expect_err("indirect execution is refused explicitly");
        assert_eq!(
            error,
            ProducerError::Unsupported {
                operation: "ExecuteIndirect",
                opcode: None
            }
        );
        assert_eq!(
            producer.commands(),
            before.as_slice(),
            "a refusal must not append or mutate a command"
        );
        assert_eq!(error.hresult(), 0x8000_4001_u32 as i32);
    }

    #[test]
    fn dispatches_need_the_threadgroup_size_and_emit_it_once_per_submission() {
        let mut producer = MetalProducer::new(11, 1).expect("producer");
        let error = {
            producer.set_pipeline(0x901).expect("pipeline");
            producer
                .set_compute_pipeline(0x901)
                .expect("compute pipeline");
            producer
                .dispatch(2, 3, 4)
                .expect_err("unknown threadgroup size")
        };
        assert!(matches!(error, ProducerError::Unsupported { .. }));

        producer
            .set_compute_threadgroup_size([8, 8, 1])
            .expect("threadgroup size");
        producer.dispatch(2, 3, 4).expect("dispatch");
        producer.dispatch(1, 1, 1).expect("second dispatch");
        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![OP_SET_COMPUTE_PIPELINE, OP_DISPATCH_3D, OP_DISPATCH_3D]
        );
        match &submission.commands[1] {
            CommandV3::Dispatch3d(dispatch) => {
                assert_eq!(dispatch.group_counts, [2, 3, 4]);
                assert_eq!(dispatch.threads_per_group, [8, 8, 1]);
            }
            other => panic!("second command is {other:?}"),
        }
    }

    /// `flush` is sequence-free: the transport (the native queue at enqueue,
    /// or `submit` for the direct sink) assigns the session sequence exactly
    /// once, so an empty flush consumes nothing.
    #[test]
    fn flushes_do_not_consume_sequences() {
        let mut producer = MetalProducer::new(12, 1).expect("producer");
        producer.present(1, 0).expect("present");
        let first = producer.flush().expect("first flush");
        assert_eq!(first.sequence, UNASSIGNED_SEQUENCE);
        assert!(!first.commands.is_empty(), "the first flush carries a frame");
        // A second flush with nothing recorded is still unassigned: it must
        // not take a sequence the host will never see.
        let empty = producer.flush().expect("second flush");
        assert_eq!(empty.sequence, UNASSIGNED_SEQUENCE);
        assert!(empty.commands.is_empty());
        let mut first = first;
        let assigned = ensure_sequence(&mut first).expect("the transport assigns");
        assert!(assigned >= 1);
        // A submission that already carries a sequence keeps it on a retry.
        assert_eq!(
            ensure_sequence(&mut first).expect("an assigned frame is not renumbered"),
            assigned
        );
    }

    #[test]
    fn the_sink_receives_the_submission_and_the_reply_is_checked() {
        let mut producer = MetalProducer::new(13, 1).expect("producer");
        producer.present(9, 0).expect("present");
        let mut sink = RecordingSink::new();
        let reply = producer.submit(&mut sink).expect("submit");
        assert_eq!(reply.status, ReplyStatus::Ok);
        assert_eq!(sink.submissions().len(), 1);
        assert_eq!(sink.submissions()[0].sequence, reply.sequence);

        let mut producer = MetalProducer::new(14, 1).expect("producer");
        producer.present(9, 0).expect("present");
        let mut sink = RecordingSink::with_status(ReplyStatus::AlreadyExecuted);
        let error = producer.submit(&mut sink).expect_err("refused reply");
        assert!(matches!(error, ProducerError::SinkRejected { .. }));
    }

    /// Every `ProducerError` variant names a static submission-refusal reason
    /// from the fixed table: a producer refusal can never be counted bare.
    #[test]
    fn every_producer_error_names_a_submit_refusal_reason() {
        use crate::native_metal::telemetry::submit_refusal_index;

        let variant = |error: &ProducerError| error.submit_refusal_name();
        let errors = [
            ProducerError::InvalidResource { operation: "test" },
            ProducerError::InvalidListIdentity,
            ProducerError::unsupported("test"),
            ProducerError::unsupported_opcode("test", 7),
            ProducerError::unregistered("test"),
            ProducerError::unresolved("test"),
            ProducerError::MissingPipeline { stage: "compute" },
            ProducerError::NoRenderTargets { operation: "test" },
            ProducerError::TooManyColorAttachments(9),
            ProducerError::TooManyBarriers(9),
            ProducerError::TooManyCommands(9),
            ProducerError::SequenceExhausted,
            ProducerError::ReplySequenceMismatch {
                expected: 1,
                actual: 2,
            },
            ProducerError::SinkRejected {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            },
        ];

        let mut names: Vec<&'static str> = errors.iter().map(&variant).collect();
        for name in &names {
            assert!(
                submit_refusal_index(name).is_some(),
                "{name} is not in SUBMIT_REFUSAL_NAMES"
            );
        }
        // Every variant keeps its own name, except the two whose refusal is not
        // a producer-class one: a sequence mismatch and a host-rejected reply.
        assert_eq!(
            variant(&ProducerError::ReplySequenceMismatch {
                expected: 1,
                actual: 2
            }),
            crate::native_metal::telemetry::submit_refusals::SEQUENCE_MISMATCH
        );
        assert_eq!(
            variant(&ProducerError::SinkRejected {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            }),
            crate::native_metal::telemetry::submit_refusals::HOST_REFUSED
        );
        names.retain(|name| !name.starts_with("producer_"));
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names,
            vec![
                crate::native_metal::telemetry::submit_refusals::HOST_REFUSED,
                crate::native_metal::telemetry::submit_refusals::SEQUENCE_MISMATCH,
            ]
        );
    }

    #[test]
    fn a_metal_list_flushes_on_close_and_refuses_after_an_error() {
        let mut list = MetalList::new(15).expect("list");
        list.producer_mut().present(1, 0).expect("present");
        list.close().expect("close");
        let pending = list.take();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].list.map(|list| list.list_id), Some(15));

        list.producer_mut().present(1, 0).expect("present");
        list.note_error(ProducerError::unsupported("ExecuteIndirect"));
        let error = list.close().expect_err("close reports the refusal");
        assert_eq!(error.hresult(), 0x8000_4001_u32 as i32);
        assert!(list.take().is_empty(), "a refused list submits nothing");
        assert_eq!(list.generation(), 1);
        list.reset().expect("reset");
        assert_eq!(list.generation(), 2);
        assert_eq!(list.failures(), 1);
    }

    #[test]
    fn emitted_opcodes_match_the_wire_constants_and_round_trip() {
        use victory_metal::wire_v3::{decode_submission_v3, DescriptorKind};

        let mut producer = MetalProducer::new(21, 1).expect("producer");
        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer
            .set_viewport(0.0, 0.0, 640.0, 360.0, 0.0, 1.0)
            .expect("viewport");
        producer
            .set_scissor_edges(0, 0, 640, 360)
            .expect("scissor");
        producer
            .set_topology(PrimitiveTopology::TriangleList)
            .expect("topology");
        producer.set_pipeline(0x900).expect("pipeline");
        producer
            .set_vertex_buffers_at(
                0,
                &[VertexBufferView {
                    address: 0x10_0000,
                    size: 4096,
                    stride: 32,
                }],
            )
            .expect("vertex buffer");
        producer
            .set_index_buffer_at(0x20_0000, 1024, 57)
            .expect("index buffer");
        producer
            .set_graphics_root_constants(0, &[1, 2, 3, 4], 0)
            .expect("root constants");
        producer
            .set_graphics_root_buffer_view(RootBufferView {
                slot: 1,
                allocation_id: 0x30_0000,
                generation: 1,
                offset: 0,
                length: 64,
            })
            .expect("root view");
        producer
            .set_graphics_descriptor_table(DescriptorTableV3 {
                set: 0,
                base_register: 2,
                kind: DescriptorKind::Cbv,
                bindings: vec![DescriptorTableEntryV3::Buffer(
                    victory_metal::wire::BufferBinding {
                        allocation_id: 0x40_0000,
                        generation: 1,
                        offset: 0,
                        length: 256,
                    },
                )],
            })
            .expect("descriptor table");
        record_draw(&mut producer);
        producer.set_mesh_pipeline(0x902).expect("mesh pipeline");
        producer.dispatch_mesh(1, 1, 1).expect("mesh dispatch");
        producer
            .resource_barrier(&[BarrierEntry {
                allocation_id: 0x2000,
                generation: 1,
                offset: 0,
                length: 64,
                before_state: 0x4,
                after_state: 0x20,
            }])
            .expect("barrier");
        producer
            .copy_buffer(resource(0x4000), 0, resource(0x5000), 0, 256)
            .expect("copy buffer");
        producer
            .copy_texture(resource(0x6000), 0, 0, resource(0x7000), 0, 0, 64, 64, 1)
            .expect("copy texture");
        producer
            .set_compute_threadgroup_size([8, 8, 1])
            .expect("compute threadgroup");
        producer.set_compute_pipeline(0x901).expect("compute pipeline");
        producer.dispatch(2, 3, 4).expect("dispatch");
        producer.present(3, 1).expect("present");

        let submission = flush_for_wire(&mut producer);
        let expected = [
            (CommandV3::BeginRenderPass(BeginRenderPass {
                colors: Vec::new(),
                depth_stencil: None,
            }), OP_BEGIN_RENDER_PASS),
            (CommandV3::SetViewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                near: 0.0,
                far: 0.0,
            }), OP_SET_VIEWPORT),
            (CommandV3::SetScissor(ScissorRect { x: 0, y: 0, width: 0, height: 0 }), OP_SET_SCISSOR),
            (CommandV3::SetPrimitiveTopology(PrimitiveTopology::PointList), OP_SET_PRIMITIVE_TOPOLOGY),
            (CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id: 1 }), OP_SET_GRAPHICS_PIPELINE),
            (CommandV3::SetVertexBuffers(VertexBufferSet { start_slot: 0, buffers: Vec::new() }), OP_SET_VERTEX_BUFFERS),
            (CommandV3::SetIndexBuffer(IndexBufferBinding {
                allocation_id: 1,
                generation: 1,
                offset: 0,
                length: 1,
                format: IndexFormat::Uint16,
            }), OP_SET_INDEX_BUFFER),
            (CommandV3::SetRootConstants(RootConstantsV3 { slot: 0, bytes: Vec::new() }), OP_ROOT_CONSTANTS),
            (CommandV3::SetRootBufferView(RootBufferView {
                slot: 0,
                allocation_id: 1,
                generation: 1,
                offset: 0,
                length: 1,
            }), OP_SET_ROOT_BUFFER_VIEW),
            (CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Cbv,
                bindings: Vec::new(),
            }), OP_DESCRIPTOR_TABLE),
            (CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
                index_count_per_instance: 1,
                instance_count: 1,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }), OP_DRAW_INDEXED_INSTANCED),
            (CommandV3::SetMeshPipeline(PipelineRef { pipeline_id: 1 }), OP_SET_MESH_PIPELINE),
            (CommandV3::DispatchMesh(DispatchMesh {
                pipeline_id: 1,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }), OP_DISPATCH_MESH),
            (CommandV3::EndRenderPass, OP_END_RENDER_PASS),
            (CommandV3::ResourceBarrier(ResourceBarrier { entries: Vec::new() }), OP_RESOURCE_BARRIER),
            (CommandV3::CopyBuffer(CopyBuffer {
                src_allocation_id: 1,
                src_generation: 1,
                src_offset: 0,
                dst_allocation_id: 1,
                dst_generation: 1,
                dst_offset: 0,
                length: 1,
            }), OP_COPY_BUFFER),
            (CommandV3::CopyTexture(CopyTexture {
                src_allocation_id: 1,
                src_generation: 1,
                src_mip: 0,
                src_slice: 0,
                dst_allocation_id: 1,
                dst_generation: 1,
                dst_mip: 0,
                dst_slice: 0,
                width: 1,
                height: 1,
                depth: 1,
            }), OP_COPY_TEXTURE),
            (CommandV3::SetComputePipeline(PipelineRef { pipeline_id: 1 }), OP_SET_COMPUTE_PIPELINE),
            (CommandV3::Dispatch3d(Dispatch3d {
                pipeline_id: 1,
                group_counts: [1, 1, 1],
                threads_per_group: [1, 1, 1],
                bindings: Vec::new(),
            }), OP_DISPATCH_3D),
            (CommandV3::Present(Present { surface_id: 1, sync_interval: 0 }), OP_PRESENT),
        ];

        assert_eq!(
            submission.commands.len(),
            expected.len(),
            "the recorded stream should exercise every emitted command"
        );
        for (command, (shape, opcode)) in submission.commands.iter().zip(expected.iter()) {
            assert_eq!(
                core::mem::discriminant(command),
                core::mem::discriminant(shape),
                "unexpected command {command:?}"
            );
            assert_eq!(
                command.opcode(),
                *opcode,
                "opcode for {command:?} differs from wire_v3"
            );
        }
        // Keep the compile-time link to the constants explicit: an opcode the
        // producer can emit but the wire does not define would not build.
        let bytes = MetalProducer::encode(&submission).expect("encode");
        let decoded = decode_submission_v3(&bytes).expect("decode");
        assert_eq!(decoded, submission, "the frame must round-trip exactly");
    }

    #[test]
    fn the_backend_default_remains_d3d12() {
        assert_eq!(GpuBackend::default(), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(None), GpuBackend::D3d12);
    }

    fn arguments(max_count: u32, counted: bool) -> IndirectArguments {
        IndirectArguments {
            allocation_id: 0x4000,
            generation: 1,
            offset: 0,
            max_count,
            count: counted.then_some(IndirectCountBuffer {
                allocation_id: 0x5000,
                generation: 1,
                offset: 0,
            }),
        }
    }

    /// The exercised indirect set becomes the matching v3 commands, with the
    /// pass opened for draws and the compute group size taken from the
    /// pipeline state.
    #[test]
    fn indirect_draws_and_dispatches_become_their_wire_commands() {
        let mut producer = MetalProducer::new(9, 1).expect("producer");
        let (colors, depth) = targets();
        producer.set_render_targets(&colors, depth).expect("targets");
        producer.set_pipeline(0x900).expect("graphics pipeline");
        producer
            .execute_indirect_draw(arguments(4, false))
            .expect("indirect draw");
        producer
            .execute_indirect_draw_indexed(arguments(2, true))
            .expect("indirect indexed draw");
        assert!(
            producer.has_open_pass(),
            "an indirect draw opens the pass like a draw"
        );

        producer.end_render_pass().expect("close the draw pass");
        producer.set_mesh_pipeline(0xa00).expect("mesh pipeline");
        producer
            .execute_indirect_dispatch_mesh(0xa00, arguments(3, false))
            .expect("indirect mesh dispatch");
        assert!(
            producer.has_open_pass(),
            "an indirect mesh dispatch opens the pass it executes in"
        );

        producer.set_compute_pipeline(0xb00).expect("compute pipeline");
        producer
            .set_compute_threadgroup_size([64, 1, 1])
            .expect("threadgroup size");
        producer
            .execute_indirect_dispatch(arguments(5, false))
            .expect("indirect dispatch");

        let submission = flush_for_wire(&mut producer);
        let indirect: Vec<u16> = submission
            .commands
            .iter()
            .filter_map(|command| {
                matches!(
                    command,
                    CommandV3::ExecuteIndirectDraw(_)
                        | CommandV3::ExecuteIndirectDrawIndexed(_)
                        | CommandV3::ExecuteIndirectDispatchMesh(_)
                        | CommandV3::ExecuteIndirectDispatch(_)
                )
                .then_some(command.opcode())
            })
            .collect();
        assert_eq!(
            indirect,
            vec![
                OP_EXECUTE_INDIRECT_DRAW,
                OP_EXECUTE_INDIRECT_DRAW_INDEXED,
                OP_EXECUTE_INDIRECT_DISPATCH_MESH,
                OP_EXECUTE_INDIRECT_DISPATCH,
            ]
        );
        // Every command round-trips through the canonical codec.
        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
    }

    /// An indirect range outside the wire's limits is refused with the
    /// opcode the histogram names, not clamped or dropped.
    #[test]
    fn an_unexpressible_indirect_range_is_refused_with_its_opcode() {
        let mut producer = MetalProducer::new(10, 1).expect("producer");
        let (colors, depth) = targets();
        producer.set_render_targets(&colors, depth).expect("targets");
        producer.set_pipeline(0x900).expect("graphics pipeline");
        let error = producer
            .execute_indirect_draw(arguments(0, false))
            .expect_err("zero max count is not expressible");
        assert_eq!(error.opcode(), Some(OP_EXECUTE_INDIRECT_DRAW));
        let error = producer
            .execute_indirect_draw(arguments(MAX_INDIRECT_COMMANDS_V3 + 1, false))
            .expect_err("too many commands");
        assert_eq!(error.opcode(), Some(OP_EXECUTE_INDIRECT_DRAW));
        let mut unaligned = arguments(1, false);
        unaligned.offset = 3;
        let error = producer
            .execute_indirect_draw(unaligned)
            .expect_err("unaligned offset");
        assert_eq!(error.opcode(), Some(OP_EXECUTE_INDIRECT_DRAW));
    }

    /// Subresource copies and resolves reach the wire with the mip/slice
    /// identity the caller named.
    #[test]
    fn texture_regions_and_resolves_become_their_wire_commands() {
        let mut producer = MetalProducer::new(11, 1).expect("producer");
        let src = TextureSubresource {
            allocation_id: 0x1000,
            generation: 1,
            mip_level: 2,
            slice: 1,
        };
        let dst = TextureSubresource {
            allocation_id: 0x2000,
            generation: 1,
            mip_level: 0,
            slice: 0,
        };
        producer
            .copy_texture_region(src, dst, [1, 2, 0], [0, 0, 0], [16, 16, 1])
            .expect("copy region");
        producer
            .resolve_subresource(src, dst)
            .expect("resolve");
        let submission = flush_for_wire(&mut producer);
        assert!(matches!(
            submission.commands.as_slice(),
            [
                CommandV3::CopyTextureRegion(copy),
                CommandV3::ResolveSubresource(resolve),
            ] if copy.src == src
                && copy.dst == dst
                && copy.size == [16, 16, 1]
                && resolve.src == src
                && resolve.dst == dst
        ));
        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
    }

    /// Root signature commits are kept per stage: Wire v3 has no command for
    /// them, so they live on the producer for a later PSO comparison instead
    /// of being dropped. A null commit unbinds, as D3D12 allows.
    #[test]
    fn root_signature_commits_are_kept_per_stage() {
        let mut producer = MetalProducer::new(17, 1).expect("producer");
        assert_eq!(producer.root_signature(RootSignatureStage::Graphics), None);
        producer
            .commit_root_signature(RootSignatureStage::Graphics, 0x7000)
            .expect("graphics root signature");
        producer
            .commit_root_signature(RootSignatureStage::Compute, 0x9000)
            .expect("compute root signature");
        assert_eq!(
            producer.root_signature(RootSignatureStage::Graphics),
            Some(0x7000)
        );
        assert_eq!(
            producer.root_signature(RootSignatureStage::Compute),
            Some(0x9000)
        );
        producer
            .commit_root_signature(RootSignatureStage::Graphics, 0)
            .expect("unbind");
        assert_eq!(producer.root_signature(RootSignatureStage::Graphics), None);
        assert_eq!(
            producer.root_signature(RootSignatureStage::Compute),
            Some(0x9000),
            "the other stage is untouched"
        );
        assert!(producer.commands().is_empty(), "a commit emits no command");
    }

    /// A guest buffer footprint becomes one pitched `CopyBufferToTexture`
    /// command: the host receives the source allocation, its offset and row
    /// pitch, and the destination mip/slice/origin for the texture it resolves.
    #[test]
    fn a_buffer_footprint_upload_becomes_a_pitched_copy() {
        let mut producer = MetalProducer::new(18, 1).expect("producer");
        let destination = TextureSubresource {
            allocation_id: 0x2000,
            generation: 1,
            mip_level: 1,
            slice: 2,
        };
        producer
            .copy_buffer_to_texture(
                resource(0x1000),
                destination,
                0x140,
                256,
                [16, 16, 1],
                [8, 4, 0],
            )
            .expect("buffer upload");
        let mut submission = producer.flush().expect("submission");
        super::ensure_sequence(&mut submission).expect("sequence");
        assert!(matches!(
            submission.commands.as_slice(),
            [CommandV3::CopyBufferToTexture(copy)]
                if copy.allocation_id == 0x1000
                    && copy.generation == 1
                    && copy.destination_allocation_id == 0x2000
                    && copy.destination_generation == 1
                    && copy.offset == 0x140
                    && copy.bytes_per_row == 256
                    && copy.bytes_per_image == 0
                    && copy.extent == [16, 16, 1]
                    && copy.origin == [8, 4, 0]
                    && copy.mip_level == 1
                    && copy.slice == 2
        ));
        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
    }

    /// The host copies one depth slice per blit: a deeper box is refused with
    /// the pitched-copy opcode instead of being silently truncated.
    #[test]
    fn a_multi_slice_buffer_upload_is_refused() {
        let mut producer = MetalProducer::new(19, 1).expect("producer");
        let destination = TextureSubresource {
            allocation_id: 0x2000,
            generation: 1,
            mip_level: 0,
            slice: 0,
        };
        let error = producer
            .copy_buffer_to_texture(
                resource(0x1000),
                destination,
                0,
                256,
                [16, 16, 2],
                [0, 0, 0],
            )
            .expect_err("a volume upload");
        assert_eq!(error.opcode(), Some(OP_COPY_BUFFER_TO_TEXTURE));
        assert!(producer.commands().is_empty(), "no partial copy is recorded");
    }

    // ------------------------------------------------- compute root bindings

    fn compute_view(slot: u16, allocation_id: u64) -> RootBufferView {
        RootBufferView {
            slot,
            allocation_id,
            generation: 1,
            offset: 0,
            length: 64,
        }
    }

    fn compute_binding(allocation_id: u64) -> BufferBinding {
        BufferBinding {
            allocation_id,
            generation: 1,
            offset: 0,
            length: 64,
        }
    }

    fn compute_producer(list_id: u64) -> MetalProducer {
        let mut producer = MetalProducer::new(list_id, 1).expect("producer");
        producer.set_compute_pipeline(0xb00).expect("compute pipeline");
        producer
            .set_compute_threadgroup_size([64, 1, 1])
            .expect("threadgroup size");
        producer
    }

    /// A compute root UAV reaches the dispatch as exactly one binding, with
    /// the resolved allocation, generation, offset and length.
    #[test]
    fn a_compute_root_view_produces_one_dispatch_binding() {
        let mut producer = compute_producer(30);
        let mut view = compute_view(0, 0x4000);
        view.offset = 0x80;
        view.length = 0x120;
        producer
            .set_compute_root_buffer_view(view)
            .expect("compute root view");
        producer.dispatch(2, 1, 1).expect("dispatch");

        let submission = flush_for_wire(&mut producer);
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_SET_COMPUTE_PIPELINE,
                OP_SET_ROOT_BUFFER_VIEW,
                OP_DISPATCH_3D,
            ]
        );
        match &submission.commands[1] {
            CommandV3::SetRootBufferView(recorded) => assert_eq!(*recorded, view),
            other => panic!("second command is {other:?}"),
        }
        match &submission.commands[2] {
            CommandV3::Dispatch3d(dispatch) => assert_eq!(
                dispatch.bindings,
                vec![BufferBinding {
                    allocation_id: 0x4000,
                    generation: 1,
                    offset: 0x80,
                    length: 0x120,
                }]
            ),
            other => panic!("third command is {other:?}"),
        }
        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
    }

    /// Two views set in call order produce two bindings in that order, and
    /// each is also emitted at its own slot for a sparse root signature.
    #[test]
    fn two_compute_root_views_produce_two_bindings_in_call_order() {
        let mut producer = compute_producer(31);
        producer
            .set_compute_root_buffer_view(compute_view(0, 0x4000))
            .expect("first view");
        producer
            .set_compute_root_buffer_view(compute_view(1, 0x5000))
            .expect("second view");
        producer.dispatch(1, 1, 1).expect("dispatch");

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_SET_COMPUTE_PIPELINE,
                OP_SET_ROOT_BUFFER_VIEW,
                OP_SET_ROOT_BUFFER_VIEW,
                OP_DISPATCH_3D,
            ]
        );
        let views: Vec<u16> = submission
            .commands
            .iter()
            .filter_map(|command| match command {
                CommandV3::SetRootBufferView(view) => Some(view.slot),
                _ => None,
            })
            .collect();
        assert_eq!(views, vec![0, 1], "stateful views keep their slots");
        match &submission.commands[3] {
            CommandV3::Dispatch3d(dispatch) => assert_eq!(
                dispatch.bindings,
                vec![compute_binding(0x4000), compute_binding(0x5000)]
            ),
            other => panic!("last command is {other:?}"),
        }
    }

    /// A view at a slot after a gap is still bound at its own slot; the
    /// dispatch payload only carries the contiguous run from slot 0 (the wire
    /// binds it at `[[buffer(0)]]`), so nothing is mis-bound.
    #[test]
    fn a_sparse_compute_view_is_bound_at_its_own_slot() {
        let mut producer = compute_producer(32);
        producer
            .set_compute_root_constants(0, &[7], 0)
            .expect("slot 0 constants");
        producer
            .set_compute_root_buffer_view(compute_view(2, 0x6000))
            .expect("slot 2 view");
        producer.dispatch(1, 1, 1).expect("dispatch");

        let submission = producer.flush().expect("flush");
        match &submission.commands[2] {
            CommandV3::SetRootBufferView(view) => assert_eq!(view.slot, 2),
            other => panic!("third command is {other:?}"),
        }
        match &submission.commands[3] {
            CommandV3::Dispatch3d(dispatch) => {
                assert!(
                    dispatch.bindings.is_empty(),
                    "a slot 2 view is not [[buffer(0)]]"
                );
            }
            other => panic!("fourth command is {other:?}"),
        }
    }

    /// Root constants stay inline and are re-emitted before every dispatch,
    /// alongside the bindings.
    #[test]
    fn compute_constants_and_bindings_reach_every_dispatch() {
        let mut producer = compute_producer(33);
        producer
            .set_compute_root_constants(2, &[0x1122_3344, 0x5566_7788], 0)
            .expect("root constants");
        producer
            .set_compute_root_buffer_view(compute_view(0, 0x4000))
            .expect("root view");
        producer.dispatch(1, 1, 1).expect("first dispatch");
        producer.dispatch(1, 1, 1).expect("second dispatch");

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_SET_COMPUTE_PIPELINE,
                OP_ROOT_CONSTANTS,
                OP_SET_ROOT_BUFFER_VIEW,
                OP_DISPATCH_3D,
                OP_ROOT_CONSTANTS,
                OP_SET_ROOT_BUFFER_VIEW,
                OP_DISPATCH_3D,
            ]
        );
        match &submission.commands[1] {
            CommandV3::SetRootConstants(constants) => {
                assert_eq!(constants.slot, 2);
                assert_eq!(
                    constants.bytes,
                    [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]
                );
            }
            other => panic!("second command is {other:?}"),
        }
        for index in [3, 6] {
            match &submission.commands[index] {
                CommandV3::Dispatch3d(dispatch) => {
                    assert_eq!(dispatch.bindings, vec![compute_binding(0x4000)]);
                }
                other => panic!("command {index} is {other:?}"),
            }
        }
    }

    /// A mixed buffer+texture table is recorded as one table and survives the
    /// wire, so a registered resource sits beside a buffer range without
    /// dropping either entry.
    #[test]
    fn a_mixed_descriptor_table_is_recorded_and_round_trips() {
        let mut producer = compute_producer(46);
        let table = DescriptorTableV3 {
            set: 0,
            base_register: 1,
            kind: DescriptorKind::Uav,
            bindings: vec![
                DescriptorTableEntryV3::Buffer(BufferBinding {
                    allocation_id: 0x7000,
                    generation: 1,
                    offset: 16,
                    length: 256,
                }),
                DescriptorTableEntryV3::Texture(0x2100),
            ],
        };
        producer
            .set_compute_descriptor_table(table.clone())
            .expect("a mixed table is accepted");
        producer.dispatch(1, 1, 1).expect("dispatch");

        let submission = flush_for_wire(&mut producer);
        let recorded = submission
            .commands
            .iter()
            .find_map(|command| match command {
                CommandV3::SetDescriptorTable(recorded) => Some(recorded.clone()),
                _ => None,
            })
            .expect("the table command");
        assert_eq!(recorded, table, "both entries reached the producer submission");

        let frame = victory_metal::wire_v3::encode_submission_v3(&submission)
            .expect("a mixed table encodes");
        let decoded = decode_submission_v3(&frame).expect("a mixed table decodes");
        assert_eq!(decoded, submission);
    }

    /// A compute descriptor table is recorded at its root slot before the
    /// dispatch; its ranges are the table's, not dispatch bindings.
    #[test]
    fn a_compute_descriptor_table_precedes_the_dispatch() {
        let mut producer = compute_producer(34);
        let table = DescriptorTableV3 {
            set: 1,
            base_register: 1,
            kind: DescriptorKind::Uav,
            bindings: vec![DescriptorTableEntryV3::Buffer(BufferBinding {
                allocation_id: 0x7000,
                generation: 1,
                offset: 16,
                length: 256,
            })],
        };
        producer
            .set_compute_descriptor_table(table.clone())
            .expect("compute table");
        producer.dispatch(1, 1, 1).expect("dispatch");

        let submission = producer.flush().expect("flush");
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OP_SET_COMPUTE_PIPELINE,
                OP_DESCRIPTOR_TABLE,
                OP_DISPATCH_3D,
            ]
        );
        match &submission.commands[1] {
            CommandV3::SetDescriptorTable(recorded) => assert_eq!(*recorded, table),
            other => panic!("second command is {other:?}"),
        }
        match &submission.commands[2] {
            CommandV3::Dispatch3d(dispatch) => assert!(
                dispatch.bindings.is_empty(),
                "a table's ranges ride on the table, not [[buffer(0)]]"
            ),
            other => panic!("last command is {other:?}"),
        }
    }

    /// A dispatch with no compute root view or table is counted by name, so an
    /// empty-binding dispatch is visible instead of silent.
    #[test]
    fn a_compute_dispatch_without_bindings_is_counted() {
        use crate::native_metal::telemetry;

        let before = telemetry::compute_dispatches_without_bindings();
        let mut producer = compute_producer(35);
        producer.dispatch(1, 1, 1).expect("unbound dispatch");
        producer.flush().expect("flush");
        assert_eq!(
            telemetry::compute_dispatches_without_bindings(),
            before + 1,
            "an unbound dispatch must be counted"
        );

        let before = telemetry::compute_dispatches_without_bindings();
        let mut producer = compute_producer(36);
        producer
            .set_compute_root_buffer_view(compute_view(0, 0x4000))
            .expect("bound view");
        producer.dispatch(1, 1, 1).expect("bound dispatch");
        producer.flush().expect("flush");
        assert_eq!(
            telemetry::compute_dispatches_without_bindings(),
            before,
            "a bound dispatch must not be counted"
        );
    }

    /// The compute dispatch that writes the indirect argument buffer is
    /// ordered before the indirect draw that reads it, with the barrier the
    /// title recorded between them, and both name the same allocation.
    #[test]
    fn a_compute_dispatch_precedes_the_indirect_draw_it_feeds() {
        let arguments = arguments(1, false);
        let mut producer = compute_producer(37);
        producer
            .set_compute_root_buffer_view(compute_view(0, arguments.allocation_id))
            .expect("argument buffer as UAV");
        producer.dispatch(1, 1, 1).expect("compute culling");
        producer
            .resource_barrier(&[BarrierEntry {
                allocation_id: arguments.allocation_id,
                generation: 1,
                offset: 0,
                length: 64,
                before_state: 0x8,
                after_state: 0x40,
            }])
            .expect("UAV barrier");
        let (colors, depth) = targets();
        producer.set_render_targets(&colors, depth).expect("targets");
        producer.set_pipeline(0x900).expect("graphics pipeline");
        producer
            .execute_indirect_draw(arguments)
            .expect("indirect draw");

        let submission = flush_for_wire(&mut producer);
        let opcodes: Vec<u16> = submission.commands.iter().map(CommandV3::opcode).collect();
        let dispatch_at = opcodes
            .iter()
            .position(|opcode| *opcode == OP_DISPATCH_3D)
            .expect("compute dispatch");
        let barrier_at = opcodes
            .iter()
            .position(|opcode| *opcode == OP_RESOURCE_BARRIER)
            .expect("barrier");
        let indirect_at = opcodes
            .iter()
            .position(|opcode| *opcode == OP_EXECUTE_INDIRECT_DRAW)
            .expect("indirect draw");
        assert!(
            dispatch_at < barrier_at && barrier_at < indirect_at,
            "dispatch -> barrier -> indirect draw, got {opcodes:?}"
        );
        match &submission.commands[dispatch_at] {
            CommandV3::Dispatch3d(dispatch) => assert_eq!(
                dispatch.bindings,
                vec![compute_binding(arguments.allocation_id)],
                "the dispatch writes the buffer the indirect draw reads"
            ),
            other => panic!("command {dispatch_at} is {other:?}"),
        }
        match &submission.commands[indirect_at] {
            CommandV3::ExecuteIndirectDraw(draw) => {
                assert_eq!(draw.arguments.allocation_id, arguments.allocation_id)
            }
            other => panic!("command {indirect_at} is {other:?}"),
        }
        let bytes = MetalProducer::encode(&submission).expect("encode");
        assert_eq!(decode_submission_v3(&bytes).expect("decode"), submission);
    }

    /// An unregistered or unresolved compute binding is `E_FAIL` with a named
    /// reason, and a list that recorded it submits nothing.
    #[test]
    fn an_unregistered_or_unresolved_compute_binding_is_e_fail() {
        const E_FAIL: i32 = 0x8000_4005_u32 as i32;
        for error in [
            ProducerError::unregistered(
                "a compute root view's allocation is not registered with the host",
            ),
            ProducerError::unresolved("a compute root view could not be resolved"),
        ] {
            assert_eq!(error.hresult(), E_FAIL, "the title sees E_FAIL");
            assert_eq!(error.opcode(), None, "no opcode is invented");

            let mut list = MetalList::new(38).expect("list");
            list.note_error(error);
            let closed = list.close().expect_err("close reports the refusal");
            assert_eq!(closed.hresult(), E_FAIL);
            assert!(
                list.take().is_empty(),
                "a refused list submits nothing (exactly-once)"
            );
        }
    }

    /// A view or table that names no live allocation is refused before it can
    /// reach a dispatch.
    #[test]
    fn invalid_compute_bindings_are_refused() {
        let mut producer = compute_producer(39);
        let error = producer
            .set_compute_root_buffer_view(RootBufferView {
                slot: 0,
                allocation_id: 0,
                generation: 1,
                offset: 0,
                length: 64,
            })
            .expect_err("zero allocation");
        assert_eq!(
            error,
            ProducerError::InvalidResource {
                operation: "resource identity"
            }
        );
        let error = producer
            .set_compute_descriptor_table(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Cbv,
                bindings: Vec::new(),
            })
            .expect_err("empty table");
        assert_eq!(error.opcode(), Some(OP_DESCRIPTOR_TABLE));
        assert!(
            producer.commands().is_empty(),
            "a refusal leaves no command"
        );
    }

    // ------------------------------------------ draw-list hint state (Boot #13)

    /// The four Boot #13 draw-list calls are state changes, not refusals: a
    /// null index buffer unbinds, the two hints record their value, and both
    /// the undefined topology and an adjacency topology are accepted without a
    /// command. No named refusal is counted and no command is emitted.
    #[test]
    fn the_four_boot_refusal_cases_are_named_counters_not_refusals() {
        let before = hint_census::counts();
        let refusals_before = crate::native_metal::telemetry::named_refusal_histogram();
        let mut producer = MetalProducer::new(40, 1).expect("producer");

        producer.unbind_index_buffer().expect("a null index is a legal unbind");
        producer
            .set_pixel_shader_depth_order(2)
            .expect("a depth-order hint");
        producer.set_shading_rate(3).expect("a shading-rate hint");
        producer
            .set_topology_value(0)
            .expect("UNDEFINED is no change");
        producer
            .set_topology_value(10)
            .expect("adjacency is a named hint");

        assert_eq!(
            producer.pixel_shader_depth_order(),
            Some(2),
            "the depth-order value is recorded"
        );
        assert_eq!(
            producer.shading_rate(),
            Some(3),
            "the shading-rate value is recorded"
        );
        assert!(
            producer.graphics.topology.is_none(),
            "neither UNDEFINED nor an adjacency hint changes the topology"
        );
        assert!(
            producer.commands().is_empty(),
            "none of the four emits a wire command"
        );

        let after = hint_census::counts();
        assert_eq!(
            [
                after[hint_census::DRAW_LIST_INDEX_UNBIND]
                    - before[hint_census::DRAW_LIST_INDEX_UNBIND],
                after[hint_census::DEPTH_ORDER_HINTS] - before[hint_census::DEPTH_ORDER_HINTS],
                after[hint_census::SHADING_RATE_HINTS] - before[hint_census::SHADING_RATE_HINTS],
                after[hint_census::TOPOLOGY_HINTS] - before[hint_census::TOPOLOGY_HINTS],
                after[hint_census::TOPOLOGY_UNDEFINED] - before[hint_census::TOPOLOGY_UNDEFINED],
            ],
            [1, 1, 1, 1, 1],
            "every accepted call is counted under its own name"
        );
        assert_eq!(
            crate::native_metal::telemetry::named_refusal_histogram(),
            refusals_before,
            "none of the four is a refusal"
        );
    }

    /// Every value of the D3D12 primitive-topology enum (`0..=64`) is either
    /// mapped to a Wire v3 topology or accepted as a named hint: zero refusals.
    #[test]
    fn every_d3d12_topology_value_is_mapped_or_an_accepted_hint() {
        let refusals_before = crate::native_metal::telemetry::named_refusal_histogram();
        let mut mapped = 0;
        let mut hints = 0;
        let mut undefined = 0;
        for value in 0..=64u32 {
            let before = hint_census::counts();
            let mut producer = MetalProducer::new(41, 1).expect("producer");
            producer
                .set_topology_value(value)
                .unwrap_or_else(|error| panic!("topology {value} was refused: {error}"));

            let expected = match value {
                0 => {
                    undefined += 1;
                    None
                }
                1 => {
                    mapped += 1;
                    Some(PrimitiveTopology::PointList)
                }
                2 => {
                    mapped += 1;
                    Some(PrimitiveTopology::LineList)
                }
                3 => {
                    mapped += 1;
                    Some(PrimitiveTopology::LineStrip)
                }
                4 => {
                    mapped += 1;
                    Some(PrimitiveTopology::TriangleList)
                }
                5 => {
                    mapped += 1;
                    Some(PrimitiveTopology::TriangleStrip)
                }
                _ => {
                    hints += 1;
                    None
                }
            };
            assert_eq!(
                producer.graphics.topology, expected,
                "topology {value} recorded the wrong mapping"
            );
            let after = hint_census::counts();
            assert_eq!(
                after[hint_census::TOPOLOGY_HINTS] > before[hint_census::TOPOLOGY_HINTS],
                value > 5,
                "topology {value} hint count"
            );
            assert_eq!(
                after[hint_census::TOPOLOGY_UNDEFINED] > before[hint_census::TOPOLOGY_UNDEFINED],
                value == 0,
                "topology {value} undefined count"
            );
            assert!(
                producer.commands().is_empty(),
                "topology {value} emits no command before a draw"
            );
        }
        assert_eq!((mapped, hints, undefined), (5, 59, 1));
        assert_eq!(
            crate::native_metal::telemetry::named_refusal_histogram(),
            refusals_before,
            "the whole enum is accepted without a refusal"
        );
    }

    /// A valid topology still lowers to the same `SetPrimitiveTopology` command
    /// as before: accepting the rest of the enum changes nothing for `1..=5`.
    #[test]
    fn a_mapped_topology_still_emits_the_same_wire_command() {
        for (value, expected) in [
            (1, PrimitiveTopology::PointList),
            (2, PrimitiveTopology::LineList),
            (3, PrimitiveTopology::LineStrip),
            (4, PrimitiveTopology::TriangleList),
            (5, PrimitiveTopology::TriangleStrip),
        ] {
            let mut producer = MetalProducer::new(42, 1).expect("producer");
            let (colors, depth) = targets();
            producer
                .set_render_targets(&colors, depth)
                .expect("render targets");
            producer.set_pipeline(0x900).expect("pipeline");
            producer
                .set_topology_value(value)
                .expect("a mapped topology");
            producer.draw(3, 1, 0, 0).expect("draw");

            let submission = flush_for_wire(&mut producer);
            assert!(
                submission.commands.iter().any(|command| {
                    matches!(
                        command,
                        CommandV3::SetPrimitiveTopology(topology) if *topology == expected
                    )
                }),
                "topology {value} did not emit {expected:?}"
            );
        }
    }

    /// `IASetIndexBuffer(null)` after a valid binding keeps the last buffer:
    /// D3D12 clears the view, but the title's state block draws with the
    /// retained binding. The unbind emits no command, is counted by name, and
    /// the following indexed draw succeeds against the remembered buffer.
    #[test]
    fn a_null_index_unbind_keeps_the_last_binding_and_a_later_indexed_draw_uses_it() {
        let mut producer = MetalProducer::new(43, 1).expect("producer");
        producer
            .set_index_buffer_at(0x20_0000, 256, 57)
            .expect("a valid index buffer");
        let before = hint_census::counts();
        producer
            .unbind_index_buffer()
            .expect("a null index is a legal unbind");
        assert_eq!(
            producer.graphics.index,
            Some(IndexBufferBinding {
                allocation_id: 0x20_0000,
                generation: 1,
                offset: 0,
                length: 256,
                format: IndexFormat::Uint16,
            }),
            "the last non-null index binding is retained"
        );
        assert!(
            producer.commands().is_empty(),
            "an unbind emits no wire command"
        );
        let after = hint_census::counts();
        assert_eq!(
            after[hint_census::DRAW_LIST_INDEX_UNBIND]
                - before[hint_census::DRAW_LIST_INDEX_UNBIND],
            1,
            "the unbind is counted by name"
        );

        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        producer
            .draw_indexed(3, 1, 0, 0, 0)
            .expect("a draw after a null view reads the retained index buffer");
        let submission = flush_for_wire(&mut producer);
        assert!(
            submission
                .commands
                .iter()
                .any(|command| matches!(command, CommandV3::DrawIndexedInstanced(_))),
            "the indexed draw reached the wire"
        );
        assert!(
            submission.commands.iter().any(|command| matches!(
                command,
                CommandV3::SetIndexBuffer(binding)
                    if binding.allocation_id == 0x20_0000 && binding.length == 256
            )),
            "the retained binding is re-emitted when the pass opens"
        );
    }

    /// A draw with no index buffer *ever* bound is the one genuine absence and
    /// keeps its named refusal (the refusal leaves no command behind).
    #[test]
    fn an_indexed_draw_with_no_index_buffer_ever_bound_is_refused_by_name() {
        let mut producer = MetalProducer::new(44, 1).expect("producer");
        producer
            .unbind_index_buffer()
            .expect("a null index is a legal unbind");

        let error = producer
            .draw_indexed(3, 1, 0, 0, 0)
            .expect_err("an indexed draw with no index buffer is a real error");
        assert_eq!(
            error.operation(),
            Some("DrawIndexed: no index buffer was ever bound")
        );
        assert_eq!(error.opcode(), Some(OP_DRAW_INDEXED_INSTANCED));
        assert!(
            producer.commands().is_empty(),
            "the refused draw leaves no command"
        );

        let (colors, depth) = targets();
        producer
            .set_render_targets(&colors, depth)
            .expect("render targets");
        producer.set_pipeline(0x900).expect("pipeline");
        producer
            .set_index_buffer_at(0x20_0000, 256, 57)
            .expect("a first index buffer");
        producer
            .draw_indexed(3, 1, 0, 0, 0)
            .expect("an index buffer now exists");
    }

    /// Two root constants with the same title slot but different translated
    /// spaces stay two records in the encoded plan, in set order.
    #[test]
    fn root_constants_with_the_same_slot_but_different_spaces_survive_encoding() {
        let mut producer = MetalProducer::new(51, 1).expect("producer");
        producer
            .set_graphics_root_constants_at(0, 0, &[0x11], 0)
            .expect("space 0 constants");
        producer
            .set_graphics_root_constants_at(0, 1, &[0x22], 0)
            .expect("space 1 constants");

        let plan = producer.root_arguments();
        assert_eq!(plan.len(), 2, "neither argument is collapsed");
        assert_eq!((plan[0].root_parameter, plan[0].space), (0, 0));
        assert_eq!((plan[1].root_parameter, plan[1].space), (0, 1));

        let encoded = producer.encode_root_arguments().expect("the plan encodes");
        assert_eq!(
            crate::native_metal::bindings::decode_root_arguments(&encoded)
                .expect("the plan decodes"),
            plan,
            "the plan round-trips with both spaces"
        );

        // The wire frame itself is untouched: the canonical codec still
        // round-trips the submission the producer flushes.
        let submission = flush_for_wire(&mut producer);
        let frame = MetalProducer::encode(&submission).expect("the frame encodes");
        let decoded = decode_submission_v3(&frame).expect("the frame decodes");
        assert_eq!(decoded, submission);
    }

    /// A constants write past the block a translated pipeline declared is a
    /// named refusal, counted under its own class.
    #[test]
    fn a_constants_write_past_the_declared_block_is_refused_by_name() {
        let mut producer = MetalProducer::new(52, 1).expect("producer");
        producer.declare_root_constant_block(RootSignatureStage::Graphics, 0, 0, 2);
        producer
            .set_graphics_root_constants(0, &[1, 2], 0)
            .expect("two dwords fit the declared block");
        let before = root_argument_census::counts()[root_argument_census::CONSTANTS_BEYOND_BLOCK];
        let error = producer
            .set_graphics_root_constants(0, &[1, 2, 3], 0)
            .expect_err("a third dword is past the block");
        assert_eq!(
            error.operation(),
            Some("Set*Root32BitConstants: the write is past the declared root-constant block")
        );
        assert_eq!(
            root_argument_census::counts()[root_argument_census::CONSTANTS_BEYOND_BLOCK],
            before + 1
        );
    }

    /// The submission seam hands a sink the plan beside the unchanged frame.
    #[test]
    fn a_submission_carries_the_root_argument_plan_to_its_sink() {
        let mut producer = MetalProducer::new(53, 1).expect("producer");
        producer
            .set_graphics_root_constants_at(4, 0, &[7], 0)
            .expect("space 0 constants");
        producer
            .set_graphics_root_constants_at(4, 1, &[9], 0)
            .expect("space 1 constants");

        let mut sink = RecordingSink::new();
        producer.submit(&mut sink).expect("submit");
        let plan = sink.root_arguments(0).expect("the sink kept the plan");
        assert_eq!(plan.len(), 2);
        assert_eq!((plan[0].root_parameter, plan[0].space), (4, 0));
        assert_eq!((plan[1].root_parameter, plan[1].space), (4, 1));
        assert_eq!(
            sink.submissions().len(),
            1,
            "the frame is still carried exactly once"
        );
    }
}
