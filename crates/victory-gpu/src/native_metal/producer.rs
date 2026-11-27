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

use victory_metal::wire::{
    BufferBinding, Dispatch3d, DispatchMesh, ListIdentity, Reply, ReplyStatus, ResourceBarrier,
    WireError, MAX_BINDINGS_PER_DISPATCH,
};
pub use victory_metal::wire::BarrierEntry;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBuffer, CopyTexture,
    CopyTextureRegion as WireCopyTextureRegion, DepthStencilAttachment, DescriptorTableEntryV3,
    DescriptorTableV3,
    DrawIndexedInstanced, DrawInstanced, ExecuteIndirectDispatch, ExecuteIndirectDispatchMesh,
    ExecuteIndirectDraw, ExecuteIndirectDrawIndexed, IndirectArguments,
    IndexBufferBinding, IndexFormat, LoadOp, PipelineRef, Present, PrimitiveTopology,
    ResolveSubresource as WireResolveSubresource, RootBufferView, RootConstantsV3, ScissorRect,
    StoreOp, TextureSubresource, VertexBuffer, VertexBufferSet, Viewport, MAX_BARRIERS_V3,
    MAX_COLOR_ATTACHMENTS, MAX_COMMANDS_V3, MAX_INDIRECT_COMMANDS_V3, MAX_ROOT_CONSTANT_BYTES_V3,
    MAX_VERTEX_BUFFERS, OP_BEGIN_RENDER_PASS, OP_COPY_BUFFER, OP_COPY_TEXTURE,
    OP_COPY_TEXTURE_REGION, OP_DESCRIPTOR_TABLE, OP_DISPATCH_3D, OP_DISPATCH_MESH,
    OP_EXECUTE_INDIRECT_DISPATCH, OP_EXECUTE_INDIRECT_DISPATCH_MESH, OP_EXECUTE_INDIRECT_DRAW,
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

/// `IASetPrimitiveTopology`'s number as a version 3 topology.
///
/// Console-only topologies (rectangles, line loops, quads, quad strips) have
/// no native equivalent: the D3D12 path draws them as a triangle list, but
/// the native path refuses them instead of changing the geometry the title
/// asked for.
fn primitive_topology(value: u32) -> Result<PrimitiveTopology, ProducerError> {
    Ok(match value {
        1 => PrimitiveTopology::PointList,
        2 => PrimitiveTopology::LineList,
        3 => PrimitiveTopology::LineStrip,
        4 => PrimitiveTopology::TriangleList,
        5 => PrimitiveTopology::TriangleStrip,
        _ => {
            return Err(ProducerError::unsupported_opcode(
                "IASetPrimitiveTopology: no native equivalent",
                OP_SET_PRIMITIVE_TOPOLOGY,
            ));
        }
    })
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
}

/// Sink that keeps every submission, for tests and diagnostics.
#[derive(Debug, Default)]
pub struct RecordingSink {
    submissions: Vec<SubmissionV3>,
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
            status: Some(status),
        }
    }

    /// The submissions received so far.
    pub fn submissions(&self) -> &[SubmissionV3] {
        &self.submissions
    }

    /// Takes the submissions received so far.
    pub fn take(&mut self) -> Vec<SubmissionV3> {
        std::mem::take(&mut self.submissions)
    }
}

impl WireV3Sink for RecordingSink {
    fn submit(&mut self, submission: &SubmissionV3) -> Result<Reply, ProducerError> {
        self.submissions.push(submission.clone());
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
    graphics: GraphicsState,
    dirty: GraphicsDirty,
    compute_pipeline: Option<u64>,
    mesh_pipeline: Option<u64>,
    compute_constants: BTreeMap<u16, Vec<u8>>,
    compute_views: BTreeMap<u16, RootBufferView>,
    compute_tables: BTreeMap<(u16, u16), DescriptorTableV3>,
    compute_local_size: Option<[u32; 3]>,
    emitted_compute_pipeline: Option<u64>,
    emitted_mesh_pipeline: Option<u64>,
    commands: Vec<CommandV3>,
}

impl MetalProducer {
    /// Starts a producer for one recorded list generation.
    pub fn new(list_id: u64, list_generation: u32) -> Result<Self, ProducerError> {
        Ok(Self {
            identity: checked_identity(list_id, list_generation)?,
            pass: PassState::default(),
            pass_open: false,
            graphics: GraphicsState::default(),
            dirty: GraphicsDirty::default(),
            compute_pipeline: None,
            mesh_pipeline: None,
            compute_constants: BTreeMap::new(),
            compute_views: BTreeMap::new(),
            compute_tables: BTreeMap::new(),
            compute_local_size: None,
            emitted_compute_pipeline: None,
            emitted_mesh_pipeline: None,
            commands: Vec::new(),
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
        self.emitted_compute_pipeline = None;
        self.emitted_mesh_pipeline = None;
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

        let next: Vec<ColorAttachment> = colors.iter().map(RenderTarget::attachment).collect();
        let next_depth = depth.map(|depth| depth.attachment());

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
        }

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
        Ok(())
    }

    /// `ClearRenderTargetView`: makes the bound target clear when its pass
    /// opens. A clear after the pass opened is refused: Metal expresses it as
    /// the pass's load action, not a separate command.
    pub fn clear_render_target(
        &mut self,
        target: ResourceId,
        clear_color: [f32; 4],
    ) -> Result<(), ProducerError> {
        ResourceId::new(target.allocation_id, target.generation)?;
        if self.pass_open {
            return Err(ProducerError::unsupported_opcode(
                "ClearRenderTargetView inside a recorded pass",
                OP_BEGIN_RENDER_PASS,
            ));
        }
        for color in &mut self.pass.colors {
            if color.allocation_id == target.allocation_id && color.generation == target.generation
            {
                color.load = LoadOp::Clear;
                color.store = StoreOp::Store;
                color.clear_color = clear_color.map(f64::from);
                return Ok(());
            }
        }
        Err(ProducerError::unsupported_opcode(
            "ClearRenderTargetView before OMSetRenderTargets",
            OP_BEGIN_RENDER_PASS,
        ))
    }

    /// `ClearDepthStencilView`: makes the bound depth target clear when its
    /// pass opens. `clear_depth` and `clear_stencil` follow the D3D12 clear
    /// flags.
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
        if self.pass_open {
            return Err(ProducerError::unsupported_opcode(
                "ClearDepthStencilView inside a recorded pass",
                OP_BEGIN_RENDER_PASS,
            ));
        }
        let Some(attachment) = self.pass.depth_stencil.as_mut() else {
            return Err(ProducerError::unsupported_opcode(
                "ClearDepthStencilView with no depth attachment bound",
                OP_BEGIN_RENDER_PASS,
            ));
        };
        if attachment.allocation_id != target.allocation_id
            || attachment.generation != target.generation
        {
            return Err(ProducerError::unsupported_opcode(
                "ClearDepthStencilView of an unbound depth target",
                OP_BEGIN_RENDER_PASS,
            ));
        }
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
    pub fn set_topology_value(&mut self, topology: u32) -> Result<(), ProducerError> {
        self.set_topology(primitive_topology(topology)?)
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
    pub fn set_graphics_root_constants(
        &mut self,
        slot: u32,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        let merge = merge_constants(&self.graphics.constants, slot, values, offset)?;
        self.graphics.constants.insert(slot as u16, merge);
        self.dirty.constants.insert(slot as u16);
        Ok(())
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
        let merge = merge_constants(&self.compute_constants, slot, values, offset)?;
        self.compute_constants.insert(slot as u16, merge);
        Ok(())
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
        ResourceId::new(view.allocation_id, view.generation)?;
        if view.length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "SetComputeRoot*View: empty range",
                OP_SET_ROOT_BUFFER_VIEW,
            ));
        }
        self.compute_views.insert(view.slot, view);
        Ok(())
    }

    /// A compute descriptor table (already resolved to buffer ranges).
    pub fn set_compute_descriptor_table(
        &mut self,
        table: DescriptorTableV3,
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
        self.compute_tables
            .insert((table.set, table.base_register), table);
        Ok(())
    }

    /// A graphics root buffer view (already resolved to an allocation).
    pub fn set_graphics_root_buffer_view(
        &mut self,
        view: RootBufferView,
    ) -> Result<(), ProducerError> {
        ResourceId::new(view.allocation_id, view.generation)?;
        if view.length == 0 {
            return Err(ProducerError::unsupported_opcode(
                "SetGraphicsRootShaderResourceView: empty range",
                OP_SET_ROOT_BUFFER_VIEW,
            ));
        }
        self.graphics.views.insert(view.slot, view);
        self.dirty.views.insert(view.slot);
        Ok(())
    }

    /// A graphics descriptor table (already resolved to buffer ranges).
    pub fn set_graphics_descriptor_table(
        &mut self,
        table: DescriptorTableV3,
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

    /// `CopyTextureRegion` of an explicit texel box between two textures.
    ///
    /// Buffer footprints (uploads and readbacks) are not expressible here:
    /// the caller refuses those with [`OP_COPY_TEXTURE_REGION`].
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
    pub fn flush(&mut self) -> Result<SubmissionV3, ProducerError> {
        self.close_pass()?;
        self.ensure_room(0)?;
        let commands = std::mem::take(&mut self.commands);
        // A clear latched for this submission happened; a later submission
        // that reuses the same attachments loads them.
        for color in &mut self.pass.colors {
            color.load = LoadOp::Load;
        }
        if let Some(depth) = &mut self.pass.depth_stencil {
            depth.depth_load = LoadOp::Load;
            depth.stencil_load = LoadOp::Load;
        }
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
        let reply = sink.submit(&submission)?;
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

    fn open_pass(&mut self, operation: &'static str) -> Result<(), ProducerError> {
        if self.pass_open {
            return Ok(());
        }
        if self.pass.colors.is_empty() {
            return Err(ProducerError::NoRenderTargets { operation });
        }
        self.ensure_room(1 + graphics_state_count(&self.graphics))?;
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
    pending: Vec<SubmissionV3>,
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
            self.pending.push(submission);
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
                    .all(|submission| submission.sequence == UNASSIGNED_SEQUENCE),
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
    pub fn take(&mut self) -> Vec<SubmissionV3> {
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
        match &submission.commands[5] {
            CommandV3::DrawIndexedInstanced(draw) => {
                assert_eq!(draw.index_count_per_instance, 36);
                assert_eq!(draw.instance_count, 1);
            }
            other => panic!("sixth command is {other:?}"),
        }
        match &submission.commands[8] {
            CommandV3::CopyBuffer(copy) => {
                assert_eq!(copy.src_allocation_id, 0x4000);
                assert_eq!(copy.dst_allocation_id, 0x5000);
                assert_eq!(copy.length, 256);
            }
            other => panic!("ninth command is {other:?}"),
        }
        match &submission.commands[9] {
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
                OP_DRAW_INDEXED_INSTANCED,
                OP_END_RENDER_PASS,
                OP_BEGIN_RENDER_PASS,
                OP_SET_GRAPHICS_PIPELINE,
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
            .set_topology_value(17)
            .expect_err("console rectangles have no native topology");
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
}
