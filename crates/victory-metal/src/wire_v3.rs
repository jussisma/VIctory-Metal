//! Version 3 wire protocol: a stateful command-buffer stream.
//!
//! The version 2 frame (`crate::wire`) carries one fixed preamble of
//! dispatch/draw commands, so a single D3D12 command list can only describe a
//! restricted shape. A recorded title list instead interleaves render-pass
//! boundaries (shadow cascades, G-buffer MRTs, lighting, UI), dynamic pipeline
//! and geometry state, compute dispatches and copies. Version 3 therefore
//! models the list as an ordered stream of stateful opcodes, and the host
//! replays it into contiguous encoder segments on one `MTLCommandBuffer`
//! (ADR-003 / ADR-006 in `.swarm/decisions.json`).
//!
//! # Ownership
//!
//! This module owns the *data model and frozen layout constants*. Encoding,
//! decoding, validation and the zero-allocation cursor decoder belong to the
//! wire v3 codec task and are added in this file without changing the public
//! types below.
//!
//! # Frame layout
//!
//! Same 20-byte header as version 2 with `version = 3`, followed by the
//! 12-byte list identity block, then `command_count` commands in execution
//! order. Every command is a `(opcode: u16, flags: u16)` prefix with `flags`
//! zero. All integers are little-endian and reserved words must be zero.
//!
//! Opcodes 1..=9 keep the version 2 meaning so a v2 batch is representable in
//! a v3 frame; opcodes 16..=40 are the stateful additions:
//!
//! | opcode | command |
//! | ---: | --- |
//! | 1 | `Dispatch` (1D thread count) |
//! | 2 | `Dispatch` (3D threadgroup grid) |
//! | 3 | `DrawIndexed` |
//! | 4 | `Draw` |
//! | 5 | `DispatchMesh` |
//! | 6 | `ResourceBarrier` |
//! | 7 | `SetPipeline` (deprecated in v3; see 20/21/22) |
//! | 8 | `SetRootConstants` |
//! | 9 | `SetDescriptorTable` |
//! | 16 | `BeginRenderPass` |
//! | 17 | `EndRenderPass` |
//! | 18 | `SetViewport` |
//! | 19 | `SetScissor` |
//! | 20 | `SetGraphicsPipeline` |
//! | 21 | `SetMeshPipeline` |
//! | 22 | `SetComputePipeline` |
//! | 23 | `SetPrimitiveTopology` |
//! | 24 | `SetVertexBuffers` |
//! | 25 | `SetIndexBuffer` |
//! | 26 | `SetRootBufferView` |
//! | 27 | `DrawInstanced` |
//! | 28 | `DrawIndexedInstanced` |
//! | 29 | `CopyBuffer` |
//! | 30 | `CopyTexture` |
//! | 31 | `Present` |
//!
//! # Invariants the codec and the host must both enforce
//!
//! - A render pass has 1..=8 color attachments and at most one depth/stencil
//!   attachment; attachments in one pass have equal extent.
//! - Commands between `BeginRenderPass` and `EndRenderPass` are graphics
//!   commands only; compute, copy and present commands outside a pass.
//! - Every resource handle names `(allocation_id, generation)` and is validated
//!   against the live registry *before* any command buffer is created.
//! - A list that has executed is refused with `AlreadyExecuted`; a validation
//!   refusal releases the claim (ADR-007).
//! - If a v3 opcode is unimplemented on the host, the submission fails loudly
//!   with an explicit error. There is no silent fallback to D3D12/D3DMetal
//!   after native submission starts (ADR-008).

use crate::wire::{BarrierEntry, BufferBinding, ListIdentity};

/// Protocol version carried in a version 3 frame.
pub const PROTOCOL_VERSION_V3: u16 = 3;

/// Maximum number of commands in one version 3 submission.
///
/// A title list can record thousands of calls; the ring buffer carries one
/// submission per flush, and 1024 leaves headroom over a typical frame while
/// keeping the frame under a few tens of kilobytes.
pub const MAX_COMMANDS_V3: usize = 1024;

/// Maximum color attachments in one render pass (D3D12 allows 8).
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// Maximum vertex-buffer slots bound by one `SetVertexBuffers` command.
pub const MAX_VERTEX_BUFFERS: usize = 16;

/// Maximum descriptor bindings in one table.
pub const MAX_DESCRIPTOR_BINDINGS_V3: usize = 32;

/// Maximum inline root-constant bytes in one command.
pub const MAX_ROOT_CONSTANT_BYTES_V3: usize = 256;

/// Maximum barrier entries in one `ResourceBarrier` command.
pub const MAX_BARRIERS_V3: usize = 32;

// Frozen v3 opcodes (see the module table).
pub const OP_DISPATCH_1D: u16 = 1;
pub const OP_DISPATCH_3D: u16 = 2;
pub const OP_DRAW_INDEXED: u16 = 3;
pub const OP_DRAW: u16 = 4;
pub const OP_DISPATCH_MESH: u16 = 5;
pub const OP_RESOURCE_BARRIER: u16 = 6;
pub const OP_SET_PIPELINE: u16 = 7;
pub const OP_ROOT_CONSTANTS: u16 = 8;
pub const OP_DESCRIPTOR_TABLE: u16 = 9;
pub const OP_BEGIN_RENDER_PASS: u16 = 16;
pub const OP_END_RENDER_PASS: u16 = 17;
pub const OP_SET_VIEWPORT: u16 = 18;
pub const OP_SET_SCISSOR: u16 = 19;
pub const OP_SET_GRAPHICS_PIPELINE: u16 = 20;
pub const OP_SET_MESH_PIPELINE: u16 = 21;
pub const OP_SET_COMPUTE_PIPELINE: u16 = 22;
pub const OP_SET_PRIMITIVE_TOPOLOGY: u16 = 23;
pub const OP_SET_VERTEX_BUFFERS: u16 = 24;
pub const OP_SET_INDEX_BUFFER: u16 = 25;
pub const OP_SET_ROOT_BUFFER_VIEW: u16 = 26;
pub const OP_DRAW_INSTANCED: u16 = 27;
pub const OP_DRAW_INDEXED_INSTANCED: u16 = 28;
pub const OP_COPY_BUFFER: u16 = 29;
pub const OP_COPY_TEXTURE: u16 = 30;
pub const OP_PRESENT: u16 = 31;

/// One version 3 submission: an ordered stateful command stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmissionV3 {
    /// Submission sequence; nonzero, strictly increasing per session.
    pub sequence: u32,
    /// Identity of the recorded command list, when the batch names one.
    pub list: Option<ListIdentity>,
    /// Commands in execution order.
    pub commands: Vec<CommandV3>,
}

/// One stateful command in a version 3 stream.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandV3 {
    /// 1D compute dispatch with an explicit total thread count.
    Dispatch1d(crate::wire::Dispatch1d),
    /// 3D compute dispatch with exact threadgroup counts.
    Dispatch3d(crate::wire::Dispatch3d),
    /// Indexed draw using the currently bound pipeline and geometry state.
    DrawIndexed(DrawIndexedState),
    /// Non-indexed draw using the current pipeline and vertex buffers.
    Draw(DrawState),
    /// Mesh-shader dispatch with exact threadgroup counts.
    DispatchMesh(crate::wire::DispatchMesh),
    /// Ordered resource transitions.
    ResourceBarrier(crate::wire::ResourceBarrier),
    /// Legacy pipeline selector kept for opcode 7 compatibility.
    SetPipeline(crate::wire::SetPipeline),
    /// Inline root constants for one root parameter slot.
    SetRootConstants(RootConstantsV3),
    /// Descriptor table at a base register.
    SetDescriptorTable(DescriptorTableV3),
    /// Opens a render pass, ending any active encoder segment first.
    BeginRenderPass(BeginRenderPass),
    /// Closes the active render pass.
    EndRenderPass,
    /// Sets the viewport of the active pass.
    SetViewport(Viewport),
    /// Sets the scissor rectangle of the active pass.
    SetScissor(ScissorRect),
    /// Selects a graphics pipeline.
    SetGraphicsPipeline(PipelineRef),
    /// Selects a mesh pipeline.
    SetMeshPipeline(PipelineRef),
    /// Selects a compute pipeline.
    SetComputePipeline(PipelineRef),
    /// Selects the primitive topology for later draws.
    SetPrimitiveTopology(PrimitiveTopology),
    /// Binds a contiguous run of vertex buffers.
    SetVertexBuffers(VertexBufferSet),
    /// Binds the index buffer and its element width.
    SetIndexBuffer(IndexBufferBinding),
    /// Binds a root 64-bit GPU virtual address view (root SRV/UAV/CBV).
    SetRootBufferView(RootBufferView),
    /// Instanced non-indexed draw.
    DrawInstanced(DrawInstanced),
    /// Instanced indexed draw.
    DrawIndexedInstanced(DrawIndexedInstanced),
    /// Buffer-to-buffer copy.
    CopyBuffer(CopyBuffer),
    /// Texture-to-texture copy.
    CopyTexture(CopyTexture),
    /// Presents a swapchain surface.
    Present(Present),
}

/// One color attachment of a render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorAttachment {
    /// Texture allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Mip level rendered into.
    pub mip_level: u16,
    /// Array slice rendered into.
    pub slice: u16,
    /// Load action applied at pass start.
    pub load: LoadOp,
    /// Store action applied at pass end.
    pub store: StoreOp,
    /// Clear color, used when `load == LoadOp::Clear`.
    pub clear_color: [f64; 4],
}

/// Depth/stencil attachment of a render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilAttachment {
    /// Texture allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Mip level rendered into.
    pub mip_level: u16,
    /// Array slice rendered into.
    pub slice: u16,
    /// Load action for depth.
    pub depth_load: LoadOp,
    /// Store action for depth.
    pub depth_store: StoreOp,
    /// Clear depth, used when `depth_load == LoadOp::Clear`.
    pub clear_depth: f32,
    /// Load action for stencil.
    pub stencil_load: LoadOp,
    /// Store action for stencil.
    pub stencil_store: StoreOp,
    /// Clear stencil value.
    pub clear_stencil: u32,
}

/// Opens a render pass with 1..=8 color attachments and optional depth.
#[derive(Debug, Clone, PartialEq)]
pub struct BeginRenderPass {
    /// Color attachments in D3D12 RTV slot order.
    pub colors: Vec<ColorAttachment>,
    /// Optional depth/stencil attachment.
    pub depth_stencil: Option<DepthStencilAttachment>,
}

/// Attachment load action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    /// Preserve previous contents.
    Load,
    /// Clear to the supplied value.
    Clear,
    /// Do not read previous contents.
    DontCare,
}

/// Attachment store action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    /// Write the pass result back.
    Store,
    /// Discard the pass result.
    DontCare,
}

/// Viewport in drawable pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// Top-left x.
    pub x: f64,
    /// Top-left y.
    pub y: f64,
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
    /// Near depth plane.
    pub near: f64,
    /// Far depth plane.
    pub far: f64,
}

/// Scissor rectangle in drawable pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScissorRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width in pixels; nonzero.
    pub width: u32,
    /// Height in pixels; nonzero.
    pub height: u32,
}

/// Host pipeline handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineRef {
    /// Pipeline ID created before submission; never zero.
    pub pipeline_id: u64,
}

/// D3D primitive topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    /// One vertex per point.
    PointList,
    /// Two vertices per line.
    LineList,
    /// Connected line strip.
    LineStrip,
    /// Three vertices per triangle.
    TriangleList,
    /// Connected triangle strip.
    TriangleStrip,
}

/// One vertex buffer binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexBuffer {
    /// Allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Byte offset of the first element.
    pub offset: u64,
    /// Byte stride between elements; nonzero.
    pub stride: u32,
    /// Byte length visible to the shader; nonzero.
    pub length: u64,
}

/// A contiguous run of vertex-buffer bindings starting at `start_slot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBufferSet {
    /// First input slot (D3D12 `IASetVertexBuffers(StartSlot, ...)`).
    pub start_slot: u16,
    /// Buffers in slot order; 1..=MAX_VERTEX_BUFFERS.
    pub buffers: Vec<VertexBuffer>,
}

/// Index-buffer element width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    /// 16-bit indices.
    Uint16,
    /// 32-bit indices.
    Uint32,
}

/// The bound index buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexBufferBinding {
    /// Allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Byte offset of the first index.
    pub offset: u64,
    /// Byte length of the index range; nonzero.
    pub length: u64,
    /// Element width.
    pub format: IndexFormat,
}

/// Inline root constants for one slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootConstantsV3 {
    /// Root parameter slot.
    pub slot: u16,
    /// Constant bytes; 1..=MAX_ROOT_CONSTANT_BYTES_V3.
    pub bytes: Vec<u8>,
}

/// Descriptor-table kind at the base register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    /// Constant buffer views.
    Cbv,
    /// Shader resource views.
    Srv,
    /// Unordered access views.
    Uav,
    /// Samplers.
    Sampler,
}

/// A descriptor table binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorTableV3 {
    /// Descriptor set index.
    pub set: u16,
    /// Base register of the table.
    pub base_register: u16,
    /// Register kind.
    pub kind: DescriptorKind,
    /// Buffers/textures in register order; 1..=MAX_DESCRIPTOR_BINDINGS_V3.
    pub bindings: Vec<BufferBinding>,
}

/// A root buffer view: a 64-bit GPU address for root SRV/UAV/CBV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootBufferView {
    /// Root parameter slot.
    pub slot: u16,
    /// Allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Byte offset within the allocation.
    pub offset: u64,
    /// Byte length; nonzero.
    pub length: u64,
}

/// Non-indexed draw with the current pipeline and vertex buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawState {
    /// Vertex count; nonzero.
    pub vertex_count: u32,
    /// Instance count; nonzero.
    pub instance_count: u32,
    /// First vertex.
    pub start_vertex: u32,
    /// First instance.
    pub start_instance: u32,
}

/// Indexed draw with the current pipeline and geometry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawIndexedState {
    /// Index count; nonzero.
    pub index_count: u32,
    /// Instance count; nonzero.
    pub instance_count: u32,
    /// First index value.
    pub start_index: u32,
    /// Signed value added to every index.
    pub base_vertex: i32,
    /// First instance.
    pub start_instance: u32,
}

/// Instanced non-indexed draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawInstanced {
    /// Vertices per instance; nonzero.
    pub vertex_count_per_instance: u32,
    /// Instances; nonzero.
    pub instance_count: u32,
    /// First vertex.
    pub start_vertex: u32,
    /// First instance.
    pub start_instance: u32,
}

/// Instanced indexed draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawIndexedInstanced {
    /// Indices per instance; nonzero.
    pub index_count_per_instance: u32,
    /// Instances; nonzero.
    pub instance_count: u32,
    /// First index value.
    pub start_index: u32,
    /// Signed value added to every index.
    pub base_vertex: i32,
    /// First instance.
    pub start_instance: u32,
}

/// Buffer-to-buffer copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyBuffer {
    /// Source allocation ID.
    pub src_allocation_id: u64,
    /// Source generation.
    pub src_generation: u32,
    /// Source byte offset.
    pub src_offset: u64,
    /// Destination allocation ID.
    pub dst_allocation_id: u64,
    /// Destination generation.
    pub dst_generation: u32,
    /// Destination byte offset.
    pub dst_offset: u64,
    /// Bytes copied; nonzero.
    pub length: u64,
}

/// Texture-to-texture copy of one mip/slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyTexture {
    /// Source allocation ID.
    pub src_allocation_id: u64,
    /// Source generation.
    pub src_generation: u32,
    /// Source mip level.
    pub src_mip: u16,
    /// Source array slice.
    pub src_slice: u16,
    /// Destination allocation ID.
    pub dst_allocation_id: u64,
    /// Destination generation.
    pub dst_generation: u32,
    /// Destination mip level.
    pub dst_mip: u16,
    /// Destination array slice.
    pub dst_slice: u16,
    /// Copy width in texels; nonzero.
    pub width: u32,
    /// Copy height in texels; nonzero.
    pub height: u32,
    /// Copy depth in texels/slices; nonzero.
    pub depth: u32,
}

/// Presents one swapchain surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Present {
    /// Swapchain/IOSurface handle.
    pub surface_id: u64,
    /// D3D12 sync interval (`0` = immediate, `1` = vsync).
    pub sync_interval: u32,
}

/// Convenience accessor: the frozen opcode of a command.
impl CommandV3 {
    /// Returns the wire opcode for this command.
    pub const fn opcode(&self) -> u16 {
        match self {
            Self::Dispatch1d(_) => OP_DISPATCH_1D,
            Self::Dispatch3d(_) => OP_DISPATCH_3D,
            Self::DrawIndexed(_) => OP_DRAW_INDEXED,
            Self::Draw(_) => OP_DRAW,
            Self::DispatchMesh(_) => OP_DISPATCH_MESH,
            Self::ResourceBarrier(_) => OP_RESOURCE_BARRIER,
            Self::SetPipeline(_) => OP_SET_PIPELINE,
            Self::SetRootConstants(_) => OP_ROOT_CONSTANTS,
            Self::SetDescriptorTable(_) => OP_DESCRIPTOR_TABLE,
            Self::BeginRenderPass(_) => OP_BEGIN_RENDER_PASS,
            Self::EndRenderPass => OP_END_RENDER_PASS,
            Self::SetViewport(_) => OP_SET_VIEWPORT,
            Self::SetScissor(_) => OP_SET_SCISSOR,
            Self::SetGraphicsPipeline(_) => OP_SET_GRAPHICS_PIPELINE,
            Self::SetMeshPipeline(_) => OP_SET_MESH_PIPELINE,
            Self::SetComputePipeline(_) => OP_SET_COMPUTE_PIPELINE,
            Self::SetPrimitiveTopology(_) => OP_SET_PRIMITIVE_TOPOLOGY,
            Self::SetVertexBuffers(_) => OP_SET_VERTEX_BUFFERS,
            Self::SetIndexBuffer(_) => OP_SET_INDEX_BUFFER,
            Self::SetRootBufferView(_) => OP_SET_ROOT_BUFFER_VIEW,
            Self::DrawInstanced(_) => OP_DRAW_INSTANCED,
            Self::DrawIndexedInstanced(_) => OP_DRAW_INDEXED_INSTANCED,
            Self::CopyBuffer(_) => OP_COPY_BUFFER,
            Self::CopyTexture(_) => OP_COPY_TEXTURE,
            Self::Present(_) => OP_PRESENT,
        }
    }
}

/// Re-exported barrier entry so producers do not import two modules.
pub type Barrier = BarrierEntry;
