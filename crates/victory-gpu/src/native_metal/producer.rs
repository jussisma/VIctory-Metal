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
//! - The first draw opens the pass. A draw never opens a second pass for the
//!   same attachments, so a pipeline, viewport or scissor change does not
//!   split it; changing the render targets or depth attachment does.
//! - [`MetalProducer::resource_barrier`], copies, compute dispatches and
//!   present close the pass first: the wire forbids those inside a pass.
//! - `flush` closes the pass and returns one submission with the next
//!   session-wide sequence; a pass never spans two submissions.
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
    Dispatch3d, DispatchMesh, ListIdentity, Reply, ReplyStatus, ResourceBarrier, WireError,
};
pub use victory_metal::wire::BarrierEntry;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBuffer, CopyTexture,
    DepthStencilAttachment, DescriptorTableV3, DrawIndexedInstanced, DrawInstanced,
    IndexBufferBinding, IndexFormat, LoadOp, PipelineRef, Present, PrimitiveTopology,
    RootBufferView, RootConstantsV3, ScissorRect, StoreOp, VertexBuffer, VertexBufferSet, Viewport,
    MAX_BARRIERS_V3, MAX_COLOR_ATTACHMENTS, MAX_COMMANDS_V3, MAX_ROOT_CONSTANT_BYTES_V3,
    MAX_VERTEX_BUFFERS, OP_BEGIN_RENDER_PASS, OP_COPY_BUFFER, OP_COPY_TEXTURE,
    OP_DESCRIPTOR_TABLE, OP_DISPATCH_3D, OP_DISPATCH_MESH, OP_PRESENT, OP_RESOURCE_BARRIER,
    OP_ROOT_CONSTANTS, OP_SET_INDEX_BUFFER, OP_SET_PRIMITIVE_TOPOLOGY, OP_SET_ROOT_BUFFER_VIEW,
    OP_SET_SCISSOR, OP_SET_VIEWPORT, OP_SET_VERTEX_BUFFERS,
};
pub use victory_metal::wire_v3::SubmissionV3;

/// Input-assembler vertex-buffer slots a D3D12 list tracks.
pub const MAX_VERTEX_SLOTS: usize = 32;

/// Sequence numbers are session-wide so two lists never collide.
static NEXT_SEQUENCE: AtomicU32 = AtomicU32::new(1);

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
fn index_format(format: u32) -> Result<IndexFormat, ProducerError> {
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

    /// An unsupported operation that would need `opcode`.
    pub fn unsupported_opcode(operation: &'static str, opcode: u16) -> Self {
        Self::Unsupported {
            operation,
            opcode: Some(opcode),
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
    compute_dirty: BTreeSet<u16>,
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
            compute_dirty: BTreeSet::new(),
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
    pub fn set_compute_root_constants(
        &mut self,
        slot: u32,
        values: &[u32],
        offset: u32,
    ) -> Result<(), ProducerError> {
        let merge = merge_constants(&self.compute_constants, slot, values, offset)?;
        self.compute_constants.insert(slot as u16, merge);
        self.compute_dirty.insert(slot as u16);
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
        for binding in &table.bindings {
            ResourceId::new(binding.allocation_id, binding.generation)?;
            if binding.length == 0 {
                return Err(ProducerError::unsupported_opcode(
                    "SetGraphicsRootDescriptorTable: empty range",
                    OP_DESCRIPTOR_TABLE,
                ));
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
        self.open_pass()?;
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
        self.open_pass()?;
        self.flush_graphics_state()?;
        self.push(CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
            index_count_per_instance: index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
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
        self.push(CommandV3::Dispatch3d(Dispatch3d {
            pipeline_id,
            group_counts: [x, y, z],
            threads_per_group,
            bindings: Vec::new(),
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
    /// Mesh dispatches are recorded outside a render pass: the producer
    /// refuses one while a pass is open instead of closing it under the
    /// title.
    pub fn dispatch_mesh(&mut self, x: u32, y: u32, z: u32) -> Result<(), ProducerError> {
        if x == 0 || y == 0 || z == 0 {
            return Ok(());
        }
        if self.pass_open {
            return Err(ProducerError::unsupported_opcode(
                "DispatchMesh inside a recorded pass",
                OP_DISPATCH_MESH,
            ));
        }
        let Some(pipeline_id) = self.mesh_pipeline else {
            return Err(ProducerError::MissingPipeline { stage: "mesh" });
        };
        if self.emitted_mesh_pipeline != Some(pipeline_id) {
            self.push(CommandV3::SetMeshPipeline(PipelineRef { pipeline_id }))?;
            self.emitted_mesh_pipeline = Some(pipeline_id);
        }
        self.push(CommandV3::DispatchMesh(DispatchMesh {
            pipeline_id,
            group_counts: [x, y, z],
            bindings: Vec::new(),
        }))
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
    /// The submission carries the next session-wide sequence, so every
    /// flush is strictly increasing across lists. An empty stream still
    /// consumes its sequence.
    pub fn flush(&mut self) -> Result<SubmissionV3, ProducerError> {
        self.close_pass()?;
        self.ensure_room(0)?;
        let sequence = next_sequence()?;
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
            sequence,
            list: Some(self.identity),
            commands,
        })
    }

    /// Flushes and carries the submission to `sink`, checking its reply.
    pub fn submit<S: WireV3Sink + ?Sized>(&mut self, sink: &mut S) -> Result<Reply, ProducerError> {
        let submission = self.flush()?;
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

    fn open_pass(&mut self) -> Result<(), ProducerError> {
        if self.pass_open {
            return Ok(());
        }
        if self.pass.colors.is_empty() {
            return Err(ProducerError::NoRenderTargets { operation: "draw" });
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
        let slots = std::mem::take(&mut self.compute_dirty);
        for slot in slots {
            if let Some(bytes) = self.compute_constants.get(&slot) {
                self.push(CommandV3::SetRootConstants(RootConstantsV3 {
                    slot,
                    bytes: bytes.clone(),
                }))?;
            }
        }
        Ok(())
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
    pub fn close(&mut self) -> Result<(), ProducerError> {
        if let Some(error) = self.failed.take() {
            self.producer.abandon();
            return Err(error);
        }
        let submission = self.producer.flush()?;
        if !submission.commands.is_empty() {
            self.pending.push(submission);
        }
        Ok(())
    }

    /// `Reset`: starts the next list generation and drops submissions the
    /// bridge did not take (the title reset before executing this list).
    pub fn reset(&mut self) -> Result<(), ProducerError> {
        if !self.pending.is_empty() {
            tracing::debug!(
                count = self.pending.len(),
                "native submissions dropped by a list reset before the bridge took them"
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
        OP_DRAW_INDEXED_INSTANCED, OP_END_RENDER_PASS, OP_SET_COMPUTE_PIPELINE,
        OP_SET_GRAPHICS_PIPELINE, OP_SET_MESH_PIPELINE,
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

        let submission = producer.flush().expect("flush");
        assert_eq!(
            submission.list,
            Some(ListIdentity {
                list_id: 7,
                list_generation: 1
            })
        );
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

    #[test]
    fn flushes_assign_increasing_sequences() {
        let mut producer = MetalProducer::new(12, 1).expect("producer");
        producer.present(1, 0).expect("present");
        let first = producer.flush().expect("first flush").sequence;
        producer.present(1, 0).expect("present");
        let second = producer.flush().expect("second flush").sequence;
        assert_eq!(second, first + 1);
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
    fn the_backend_default_remains_d3d12() {
        assert_eq!(GpuBackend::default(), GpuBackend::D3d12);
        assert_eq!(GpuBackend::parse(None), GpuBackend::D3d12);
    }
}
