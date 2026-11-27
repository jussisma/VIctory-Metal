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
//! This module owns the data model, the frozen layout constants and the codec
//! ([`encode_submission_v3`], [`decode_submission_v3`],
//! [`decode_submission_v3_into`] and [`ReaderV3`]). The public types below are
//! shared with the Win64 producer and the ARM64 host and are not restructured
//! by the codec.
//!
//! # Frame layout
//!
//! Same 20-byte header as version 2 with `version = 3`, followed by the
//! 12-byte list identity block, then `command_count` commands in execution
//! order. Every command is a `(opcode: u16, flags: u16)` prefix with `flags`
//! zero. All integers are little-endian and reserved words must be zero.
//!
//! Opcodes 1..=9 keep the version 2 meaning so a v2 batch is representable in
//! a v3 frame; opcodes 16..=31 are the stateful additions:
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
//! # Version 3 command payloads
//!
//! Every command is a 4-byte `(opcode: u16, flags: u16)` prefix with `flags`
//! zero, followed by the payload below. Offsets are relative to the first byte
//! of the command, matching the version 2 tables in [`crate::wire`]. `N` is a
//! count and `n` the matching array length; every `reserved` word must be zero.
//! All integer fields are little-endian. `f32`/`f64` fields are their IEEE-754
//! bit patterns as little-endian words, never a native float cast.
//!
//! Opcode 1 is a 1D compute dispatch. Its payload is byte-identical to the
//! version 2 opcode 1:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (1 = dispatch 1D) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//! | 12 | 4 | thread count (never zero) |
//! | 16 | 4 | threads per threadgroup (never zero) |
//! | 20 | 2 | binding count (0 ..= [`crate::wire::MAX_BINDINGS_PER_DISPATCH`]) |
//! | 22 | 2 | reserved, must be zero |
//! | 24 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 2 is a 3D compute dispatch and keeps the version 2 layout:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (2 = dispatch 3D groups) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//! | 12 | 4 x 3 | threadgroup counts X, Y, Z (each never zero) |
//! | 24 | 4 x 3 | threads per threadgroup X, Y, Z (each never zero) |
//! | 36 | 2 | binding count (0 ..= [`crate::wire::MAX_BINDINGS_PER_DISPATCH`]) |
//! | 38 | 2 | reserved, must be zero |
//! | 40 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 3 is an indexed draw. Unlike version 2 it names no pipeline and no
//! list: the active graphics pipeline (20) and the frame list identity apply.
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (3 = draw indexed) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 4 | index count (never zero) |
//! | 8 | 4 | instance count (never zero) |
//! | 12 | 4 | start index |
//! | 16 | 4 | base vertex (`i32`) |
//! | 20 | 4 | start instance |
//!
//! Opcode 4 is a non-indexed draw against the current pipeline and vertex
//! buffers:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (4 = draw) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 4 | vertex count (never zero) |
//! | 8 | 4 | instance count (never zero) |
//! | 12 | 4 | start vertex |
//! | 16 | 4 | start instance |
//!
//! Opcode 5 is a mesh-shader dispatch and keeps the version 2 layout:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (5 = dispatch mesh) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//! | 12 | 4 x 3 | threadgroup counts X, Y, Z (each never zero) |
//! | 24 | 2 | binding count (0 ..= [`crate::wire::MAX_BINDINGS_PER_DISPATCH`]) |
//! | 26 | 2 | reserved, must be zero |
//! | 28 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 6 is an ordered resource-barrier batch. Version 3 raises the entry
//! cap to [`MAX_BARRIERS_V3`]:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (6 = resource barrier) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | entry count (1 ..= [`MAX_BARRIERS_V3`]) |
//! | 6 | 2 | reserved, must be zero |
//! | 8 | 40 x N | barrier entries |
//!
//! Each barrier entry is `allocation_id: u64`, `generation: u32`, reserved
//! `u32`, `offset: u64`, `length: u64`, `before_state: u32`, `after_state:
//! u32`. States are carried verbatim as D3D12 transition words.
//!
//! Opcode 7 is the legacy pipeline selector:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (7 = set pipeline) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//!
//! Opcode 8 carries inline root constants (1 ..= [`MAX_ROOT_CONSTANT_BYTES_V3`]):
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (8 = root constants) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | root parameter slot |
//! | 6 | 2 | byte length (1 ..= [`MAX_ROOT_CONSTANT_BYTES_V3`]) |
//! | 8 | N | constant bytes |
//!
//! Opcode 9 binds a descriptor table. Version 3 adds the register kind word
//! and raises the binding cap to [`MAX_DESCRIPTOR_BINDINGS_V3`]:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (9 = descriptor table) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | descriptor set |
//! | 6 | 2 | base register |
//! | 8 | 2 | kind (0 = CBV, 1 = SRV, 2 = UAV, 3 = sampler) |
//! | 10 | 2 | binding count (1 ..= [`MAX_DESCRIPTOR_BINDINGS_V3`]) |
//! | 12 | 2 | entry tag (0 = 32-byte buffer ranges, 1 = 8-byte resource handles, 2 = per-entry tagged) |
//! | 14 | 32 x N | bound buffer ranges, when the tag is 0 |
//! | 14 | 8 x N | registered texture/sampler handles, when the tag is 1 |
//!
//! A tag of 2 carries one per-entry kind byte before every entry, so a table
//! may mix buffer ranges and registered resources. The per-entry kinds are `0`
//! (a 32-byte buffer range follows), `1` (an 8-byte registered texture/surface
//! handle follows) and `2` (an 8-byte registered sampler handle follows). A
//! homogeneous table keeps tag 0 or 1 and its old byte layout unchanged.
//!
//! Opcode 16 opens a render pass with 1 ..= [`MAX_COLOR_ATTACHMENTS`] color
//! attachments and one optional depth/stencil attachment:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (16 = begin render pass) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | color attachment count (1 ..= 8) |
//! | 6 | 2 | depth attachment count (0 or 1) |
//! | 8 | 4 | reserved, must be zero |
//! | 12 | 56 x C | color attachments |
//! | 12 + 56 x C | 36 x D | depth/stencil attachment, when D = 1 |
//!
//! Each 56-byte color attachment:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | allocation ID (never zero) |
//! | 8 | 4 | generation (never zero) |
//! | 12 | 2 | mip level |
//! | 14 | 2 | array slice |
//! | 16 | 2 | load op (0 = load, 1 = clear, 2 = don't care) |
//! | 18 | 2 | store op (0 = store, 1 = don't care) |
//! | 20 | 4 | reserved, must be zero |
//! | 24 | 32 | clear color `[f64; 4]` |
//!
//! Each 36-byte depth/stencil attachment:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | allocation ID (never zero) |
//! | 8 | 4 | generation (never zero) |
//! | 12 | 2 | mip level |
//! | 14 | 2 | array slice |
//! | 16 | 2 | depth load op (0 = load, 1 = clear, 2 = don't care) |
//! | 18 | 2 | depth store op (0 = store, 1 = don't care) |
//! | 20 | 4 | clear depth (`f32`) |
//! | 24 | 2 | stencil load op (0 = load, 1 = clear, 2 = don't care) |
//! | 26 | 2 | stencil store op (0 = store, 1 = don't care) |
//! | 28 | 4 | clear stencil |
//! | 32 | 4 | reserved, must be zero |
//!
//! Opcode 17 closes the active pass and has no payload:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (17 = end render pass) |
//! | 2 | 2 | flags, must be zero |
//!
//! Opcode 18 sets the viewport:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (18 = set viewport) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 x 6 | x, y, width, height, near, far (`f64` each) |
//!
//! Opcode 19 sets the scissor rectangle:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (19 = set scissor) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 4 | x |
//! | 8 | 4 | y |
//! | 12 | 4 | width (never zero) |
//! | 16 | 4 | height (never zero) |
//!
//! Opcodes 20, 21 and 22 select a graphics, mesh or compute pipeline:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (20, 21 or 22) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//!
//! Opcode 23 selects the primitive topology:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (23 = set primitive topology) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | topology (0 = point list, 1 = line list, 2 = line strip, 3 = triangle list, 4 = triangle strip) |
//! | 6 | 2 | reserved, must be zero |
//!
//! Opcode 24 binds a contiguous run of vertex buffers:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (24 = set vertex buffers) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | first input slot |
//! | 6 | 2 | buffer count (1 ..= [`MAX_VERTEX_BUFFERS`]) |
//! | 8 | 4 | reserved, must be zero |
//! | 12 | 32 x N | vertex buffers, in slot order |
//!
//! Each 32-byte vertex buffer:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | allocation ID (never zero) |
//! | 8 | 4 | generation (never zero) |
//! | 12 | 8 | byte offset of the first element |
//! | 20 | 4 | byte stride (never zero) |
//! | 24 | 8 | byte length (nonzero, at most [`crate::wire::MAX_RESOURCE_LEN`]) |
//!
//! Opcode 25 binds the index buffer:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (25 = set index buffer) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | allocation ID (never zero) |
//! | 12 | 4 | generation (never zero) |
//! | 16 | 8 | byte offset of the first index |
//! | 24 | 8 | byte length (nonzero, at most [`crate::wire::MAX_RESOURCE_LEN`]) |
//! | 32 | 2 | format (0 = `u16`, 1 = `u32`) |
//! | 34 | 2 | reserved, must be zero |
//!
//! Opcode 26 binds a root buffer view:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (26 = set root buffer view) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | root parameter slot |
//! | 6 | 2 | reserved, must be zero |
//! | 8 | 8 | allocation ID (never zero) |
//! | 16 | 4 | generation (never zero) |
//! | 20 | 4 | reserved, must be zero |
//! | 24 | 8 | byte offset within the allocation |
//! | 32 | 8 | byte length (nonzero, at most [`crate::wire::MAX_RESOURCE_LEN`]) |
//!
//! Opcode 27 is an instanced non-indexed draw:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (27 = draw instanced) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 4 | vertices per instance (never zero) |
//! | 8 | 4 | instance count (never zero) |
//! | 12 | 4 | start vertex |
//! | 16 | 4 | start instance |
//!
//! Opcode 28 is an instanced indexed draw:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (28 = draw indexed instanced) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 4 | indices per instance (never zero) |
//! | 8 | 4 | instance count (never zero) |
//! | 12 | 4 | start index |
//! | 16 | 4 | base vertex (`i32`) |
//! | 20 | 4 | start instance |
//!
//! Opcode 29 copies a byte range between buffers:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (29 = copy buffer) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | source allocation ID (never zero) |
//! | 12 | 4 | source generation (never zero) |
//! | 16 | 4 | reserved, must be zero |
//! | 20 | 8 | source byte offset |
//! | 28 | 8 | destination allocation ID (never zero) |
//! | 36 | 4 | destination generation (never zero) |
//! | 40 | 4 | reserved, must be zero |
//! | 44 | 8 | destination byte offset |
//! | 52 | 8 | byte length (nonzero, at most [`crate::wire::MAX_RESOURCE_LEN`]) |
//!
//! Opcode 30 copies one mip/slice region between textures:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (30 = copy texture) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | source allocation ID (never zero) |
//! | 12 | 4 | source generation (never zero) |
//! | 16 | 2 | source mip level |
//! | 18 | 2 | source array slice |
//! | 20 | 8 | destination allocation ID (never zero) |
//! | 28 | 4 | destination generation (never zero) |
//! | 32 | 2 | destination mip level |
//! | 34 | 2 | destination array slice |
//! | 36 | 4 | width in texels (never zero) |
//! | 40 | 4 | height in texels (never zero) |
//! | 44 | 4 | depth in texels/slices (never zero) |
//! | 48 | 4 | reserved, must be zero |
//!
//! Opcode 31 presents a swapchain surface:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (31 = present) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | swapchain/IOSurface handle |
//! | 12 | 4 | sync interval (0 = immediate, 1 = vsync) |
//! | 16 | 4 | reserved, must be zero |
//!
//! # Reply extension
//!
//! A kind 5 request is answered with one Wire reply frame (see
//! [`crate::wire::decode_reply_with_reason`]). An error reply may additionally
//! carry a bounded UTF-8 reason trailer guarded by the `VMR3` magic, so the
//! guest prints the exact host/Metal diagnostic instead of a bare
//! `ExecutionFailed`. A success reply has the frozen four-byte payload and is
//! byte-identical to the legacy frame; a legacy error reply still decodes with
//! no reason.
//!
//! # Validation
//!
//! Encoding and decoding enforce the same rules, so the writer cannot produce a
//! frame the reader rejects. A frame is refused, without exposing a partial
//! submission, when the sequence is zero, the list identity is half-populated,
//! the command count exceeds [`MAX_COMMANDS_V3`], an opcode is unknown, the
//! protocol version is not [`PROTOCOL_VERSION_V3`], the payload is truncated or
//! followed by trailing bytes, or a reserved or flags word is nonzero. Each
//! command additionally rejects zero pipeline IDs, zero thread/group
//! dimensions, zero index/vertex/instance counts, zero or oversized counts,
//! zero allocation IDs or generations, zero or oversized range lengths,
//! `offset + length` overflow, undefined enumeration codes and a `Present`
//! sync interval other than 0 or 1. Counts are bounded before the decoder
//! reserves storage.
//!
//! [`decode_submission_v3_into`] reuses the destination command vector, and the
//! nested payload vectors of a command whose opcode and variant match, so a
//! repeated decode of the same or a smaller submission does not reallocate.
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

use crate::wire::{
    BarrierEntry, BufferBinding, Dispatch1d, Dispatch3d, DispatchMesh, ListIdentity, MAGIC,
    MAX_BINDINGS_PER_DISPATCH, MAX_RESOURCE_LEN, ResourceBarrier, SetPipeline, WireError,
};

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

// ADR-015 opcode band (40+). Bytes 1..=31 stay frozen.
pub const OP_EXECUTE_INDIRECT_DRAW: u16 = 40;
pub const OP_EXECUTE_INDIRECT_DRAW_INDEXED: u16 = 41;
pub const OP_EXECUTE_INDIRECT_DISPATCH_MESH: u16 = 42;
pub const OP_EXECUTE_INDIRECT_DISPATCH: u16 = 43;
pub const OP_COPY_TEXTURE_REGION: u16 = 44;
pub const OP_RESOLVE_SUBRESOURCE: u16 = 45;

// ADR-026a core render-state band (46+): blend, depth-stencil, rasterizer,
// stencil reference, blend factor, explicit clears and pitched upload.
pub const OP_SET_BLEND_STATE: u16 = 46;
pub const OP_SET_DEPTH_STENCIL_STATE: u16 = 47;
pub const OP_SET_RASTERIZER_STATE: u16 = 48;
pub const OP_SET_STENCIL_REF: u16 = 49;
pub const OP_SET_BLEND_FACTOR: u16 = 50;
pub const OP_CLEAR_RENDER_TARGET_VIEW: u16 = 51;
pub const OP_CLEAR_DEPTH_STENCIL_VIEW: u16 = 52;
pub const OP_COPY_BUFFER_TO_TEXTURE: u16 = 53;

/// Maximum per-render-target blend entries in one `SetBlendState`.
pub const MAX_BLEND_ATTACHMENTS_V3: usize = 8;

/// Maximum number of indirect commands one v3 command may execute.
///
/// Without a count buffer D3D12 executes exactly `MaxCommandCount` commands;
/// the Metal executor issues one GPU-read indirect call per command, so the cap
/// bounds both the wire payload and the number of encoder calls.
pub const MAX_INDIRECT_COMMANDS_V3: u32 = 256;

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
    /// GPU-driven non-indexed instanced draw.
    ExecuteIndirectDraw(ExecuteIndirectDraw),
    /// GPU-driven indexed instanced draw.
    ExecuteIndirectDrawIndexed(ExecuteIndirectDrawIndexed),
    /// GPU-driven mesh-shader dispatch.
    ExecuteIndirectDispatchMesh(ExecuteIndirectDispatchMesh),
    /// GPU-driven compute dispatch.
    ExecuteIndirectDispatch(ExecuteIndirectDispatch),
    /// Texture-to-texture copy of an explicit texel box.
    CopyTextureRegion(CopyTextureRegion),
    /// Multisample resolve into a single-sample subresource.
    ResolveSubresource(ResolveSubresource),
    /// Per-render-target blend state (ADR-026a).
    SetBlendState(BlendState),
    /// Depth/stencil comparison, write and stencil-face state (ADR-026a).
    SetDepthStencilState(DepthStencilState),
    /// Rasterizer fill, cull, winding, bias and depth-clip state (ADR-026a).
    SetRasterizerState(RasterizerState),
    /// Stencil reference value (`OMSetStencilRef`).
    SetStencilRef(StencilRef),
    /// Blend factor constant (`OMSetBlendFactor`).
    SetBlendFactor(BlendFactorValue),
    /// Clears one colour attachment of the open render pass.
    ClearRenderTargetView(ClearRenderTargetView),
    /// Clears the depth and/or stencil plane of the open render pass.
    ClearDepthStencilView(ClearDepthStencilView),
    /// Pitched buffer-to-texture upload (`CopyTextureRegion` from an upload heap).
    CopyBufferToTexture(CopyBufferToTexture),
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

/// One entry of a version 3 descriptor table.
///
/// A buffer entry keeps the resolved `(allocation_id, generation, offset,
/// length)` range. A texture or sampler entry names a host registration
/// directly: the handle is the transport-kind-8 texture handle for
/// [`DescriptorKind::Srv`]/[`DescriptorKind::Uav`] tables, or an
/// `MTLSamplerState` registration handle for [`DescriptorKind::Sampler`]
/// tables. Every variant is `Copy`, so a decoded table allocates only its
/// entry vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorTableEntryV3 {
    /// A buffer range in the guest allocation registry.
    Buffer(BufferBinding),
    /// A registered texture handle (`gpuResourceID` is resolved on the host).
    Texture(u64),
    /// A registered sampler-state handle.
    Sampler(u64),
}

impl DescriptorTableEntryV3 {
    /// The buffer range, when this is a buffer entry.
    pub fn as_buffer(&self) -> Option<&BufferBinding> {
        match self {
            Self::Buffer(binding) => Some(binding),
            Self::Texture(_) | Self::Sampler(_) => None,
        }
    }

    /// The registered resource handle, when this is a texture or sampler entry.
    pub fn resource_handle(&self) -> Option<u64> {
        match self {
            Self::Texture(handle) | Self::Sampler(handle) => Some(*handle),
            Self::Buffer(_) => None,
        }
    }
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
    /// Table entries in register order; 1..=MAX_DESCRIPTOR_BINDINGS_V3.
    pub bindings: Vec<DescriptorTableEntryV3>,
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

/// Optional GPU-side command count for an `ExecuteIndirect` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectCountBuffer {
    /// Count-buffer allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Byte offset of the 32-bit count; 4-byte aligned.
    pub offset: u64,
}

/// The GPU-resident argument range of one `ExecuteIndirect` command.
///
/// `offset` and every argument record are 4-byte aligned. `max_count` is the
/// D3D12 `MaxCommandCount`; the argument records are read by the GPU only and
/// never copied to the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectArguments {
    /// Argument-buffer allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Byte offset of the first argument record; 4-byte aligned.
    pub offset: u64,
    /// Maximum commands to execute; 1..=MAX_INDIRECT_COMMANDS_V3.
    pub max_count: u32,
    /// Optional count buffer; `None` is encoded as allocation ID zero.
    pub count: Option<IndirectCountBuffer>,
}

impl IndirectArguments {
    /// Bytes covered by `max_count` argument records of `stride` bytes.
    pub fn byte_len(&self, stride: u64) -> Option<u64> {
        u64::from(self.max_count).checked_mul(stride)
    }
}

/// Non-indexed `ExecuteIndirect` draw arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteIndirectDraw {
    /// GPU-resident argument records.
    pub arguments: IndirectArguments,
}

/// Indexed `ExecuteIndirect` draw arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteIndirectDrawIndexed {
    /// GPU-resident argument records.
    pub arguments: IndirectArguments,
}

/// Mesh-shader `ExecuteIndirect` dispatch arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteIndirectDispatchMesh {
    /// Mesh pipeline handle; never zero.
    pub pipeline_id: u64,
    /// GPU-resident threadgroup-count records.
    pub arguments: IndirectArguments,
}

/// Compute `ExecuteIndirect` dispatch arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteIndirectDispatch {
    /// Compute pipeline handle; never zero.
    pub pipeline_id: u64,
    /// GPU-resident threadgroup-count records.
    pub arguments: IndirectArguments,
    /// Threadgroup size; Metal requires it alongside the indirect grid.
    pub threads_per_group: [u32; 3],
}

/// One texture subresource named by a copy or resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureSubresource {
    /// Texture allocation ID; never zero.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// Mip level.
    pub mip_level: u16,
    /// Array slice.
    pub slice: u16,
}

/// Texture-to-texture copy of an explicit texel box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyTextureRegion {
    /// Source subresource.
    pub src: TextureSubresource,
    /// Destination subresource.
    pub dst: TextureSubresource,
    /// Source box origin in texels.
    pub src_origin: [u32; 3],
    /// Destination box origin in texels.
    pub dst_origin: [u32; 3],
    /// Box size in texels; every component nonzero.
    pub size: [u32; 3],
}

/// Multisample resolve of one subresource into a single-sample target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolveSubresource {
    /// Multisample source subresource.
    pub src: TextureSubresource,
    /// Single-sample destination subresource.
    pub dst: TextureSubresource,
}

/// D3D12 blend factor, encoded with its D3D12 enum value.
///
/// Codes 12 and 13 are unused by D3D12 and are refused. The host maps each
/// variant onto the `MTLBlendFactor` that has the same arithmetic meaning; the
/// mapping lives in `macos::state` so the wire stays Metal-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// `D3D12_BLEND_ZERO`.
    Zero = 1,
    /// `D3D12_BLEND_ONE`.
    One = 2,
    /// `D3D12_BLEND_SRC_COLOR`.
    SrcColor = 3,
    /// `D3D12_BLEND_INV_SRC_COLOR`.
    InvSrcColor = 4,
    /// `D3D12_BLEND_SRC_ALPHA`.
    SrcAlpha = 5,
    /// `D3D12_BLEND_INV_SRC_ALPHA`.
    InvSrcAlpha = 6,
    /// `D3D12_BLEND_DEST_ALPHA`.
    DestAlpha = 7,
    /// `D3D12_BLEND_INV_DEST_ALPHA`.
    InvDestAlpha = 8,
    /// `D3D12_BLEND_DEST_COLOR`.
    DestColor = 9,
    /// `D3D12_BLEND_INV_DEST_COLOR`.
    InvDestColor = 10,
    /// `D3D12_BLEND_SRC_ALPHA_SAT`.
    SrcAlphaSat = 11,
    /// `D3D12_BLEND_BLEND_FACTOR`.
    BlendFactor = 14,
    /// `D3D12_BLEND_INV_BLEND_FACTOR`.
    InvBlendFactor = 15,
    /// `D3D12_BLEND_SRC1_COLOR`.
    Src1Color = 16,
    /// `D3D12_BLEND_INV_SRC1_COLOR`.
    InvSrc1Color = 17,
    /// `D3D12_BLEND_SRC1_ALPHA`.
    Src1Alpha = 18,
    /// `D3D12_BLEND_INV_SRC1_ALPHA`.
    InvSrc1Alpha = 19,
}

impl BlendFactor {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 blend-factor code, refusing the two unused codes.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            1 => Self::Zero,
            2 => Self::One,
            3 => Self::SrcColor,
            4 => Self::InvSrcColor,
            5 => Self::SrcAlpha,
            6 => Self::InvSrcAlpha,
            7 => Self::DestAlpha,
            8 => Self::InvDestAlpha,
            9 => Self::DestColor,
            10 => Self::InvDestColor,
            11 => Self::SrcAlphaSat,
            14 => Self::BlendFactor,
            15 => Self::InvBlendFactor,
            16 => Self::Src1Color,
            17 => Self::InvSrc1Color,
            18 => Self::Src1Alpha,
            19 => Self::InvSrc1Alpha,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "blend factor",
                    code: other,
                });
            }
        })
    }
}

/// D3D12 blend operation, encoded with its D3D12 enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    /// `D3D12_BLEND_OP_ADD`.
    Add = 1,
    /// `D3D12_BLEND_OP_SUBTRACT`.
    Subtract = 2,
    /// `D3D12_BLEND_OP_REV_SUBTRACT`.
    RevSubtract = 3,
    /// `D3D12_BLEND_OP_MIN`.
    Min = 4,
    /// `D3D12_BLEND_OP_MAX`.
    Max = 5,
}

impl BlendOperation {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 blend-operation code.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            1 => Self::Add,
            2 => Self::Subtract,
            3 => Self::RevSubtract,
            4 => Self::Min,
            5 => Self::Max,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "blend operation",
                    code: other,
                });
            }
        })
    }
}

/// D3D12 comparison function, encoded with its D3D12 enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// `D3D12_COMPARISON_FUNC_NEVER`.
    Never = 1,
    /// `D3D12_COMPARISON_FUNC_LESS`.
    Less = 2,
    /// `D3D12_COMPARISON_FUNC_EQUAL`.
    Equal = 3,
    /// `D3D12_COMPARISON_FUNC_LESS_EQUAL`.
    LessEqual = 4,
    /// `D3D12_COMPARISON_FUNC_GREATER`.
    Greater = 5,
    /// `D3D12_COMPARISON_FUNC_NOT_EQUAL`.
    NotEqual = 6,
    /// `D3D12_COMPARISON_FUNC_GREATER_EQUAL`.
    GreaterEqual = 7,
    /// `D3D12_COMPARISON_FUNC_ALWAYS`.
    Always = 8,
}

impl CompareFunction {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 comparison-function code.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            1 => Self::Never,
            2 => Self::Less,
            3 => Self::Equal,
            4 => Self::LessEqual,
            5 => Self::Greater,
            6 => Self::NotEqual,
            7 => Self::GreaterEqual,
            8 => Self::Always,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "comparison function",
                    code: other,
                });
            }
        })
    }
}

/// D3D12 stencil operation, encoded with its D3D12 enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    /// `D3D12_STENCIL_OP_KEEP`.
    Keep = 1,
    /// `D3D12_STENCIL_OP_ZERO`.
    Zero = 2,
    /// `D3D12_STENCIL_OP_REPLACE`.
    Replace = 3,
    /// `D3D12_STENCIL_OP_INCR_SAT`.
    IncrSat = 4,
    /// `D3D12_STENCIL_OP_DECR_SAT`.
    DecrSat = 5,
    /// `D3D12_STENCIL_OP_INVERT`.
    Invert = 6,
    /// `D3D12_STENCIL_OP_INCR`.
    Incr = 7,
    /// `D3D12_STENCIL_OP_DECR`.
    Decr = 8,
}

impl StencilOperation {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 stencil-operation code.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            1 => Self::Keep,
            2 => Self::Zero,
            3 => Self::Replace,
            4 => Self::IncrSat,
            5 => Self::DecrSat,
            6 => Self::Invert,
            7 => Self::Incr,
            8 => Self::Decr,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "stencil operation",
                    code: other,
                });
            }
        })
    }
}

/// D3D12 fill mode, encoded with its D3D12 enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillMode {
    /// `D3D12_FILL_MODE_WIREFRAME`.
    Wireframe = 2,
    /// `D3D12_FILL_MODE_SOLID`.
    Solid = 3,
}

impl FillMode {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 fill-mode code.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            2 => Self::Wireframe,
            3 => Self::Solid,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "fill mode",
                    code: other,
                });
            }
        })
    }
}

/// D3D12 cull mode, encoded with its D3D12 enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// `D3D12_CULL_MODE_NONE`.
    None = 1,
    /// `D3D12_CULL_MODE_FRONT`.
    Front = 2,
    /// `D3D12_CULL_MODE_BACK`.
    Back = 3,
}

impl CullMode {
    /// D3D12 enum value carried on the wire.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes a D3D12 cull-mode code.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            1 => Self::None,
            2 => Self::Front,
            3 => Self::Back,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "cull mode",
                    code: other,
                });
            }
        })
    }
}

/// Which triangle winding is front-facing.
///
/// D3D12 carries this as the `FrontCounterClockwise` boolean; the wire keeps
/// the boolean meaning in a two-valued enum so an undefined code is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFaceWinding {
    /// Clockwise triangles are front-facing (`FrontCounterClockwise = FALSE`).
    Clockwise = 0,
    /// Counter-clockwise triangles are front-facing (`TRUE`).
    CounterClockwise = 1,
}

impl FrontFaceWinding {
    /// Boolean wire code.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Decodes the `FrontCounterClockwise` boolean.
    pub fn from_code(code: u16) -> Result<Self, WireError> {
        Ok(match code {
            0 => Self::Clockwise,
            1 => Self::CounterClockwise,
            other => {
                return Err(WireError::InvalidEnumCode {
                    field: "front-face winding",
                    code: other,
                });
            }
        })
    }
}

/// One render target's blend state (`D3D12_RENDER_TARGET_BLEND_DESC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendAttachment {
    /// Whether blending is enabled for this target.
    pub enable: bool,
    /// Source colour factor.
    pub src_color: BlendFactor,
    /// Destination colour factor.
    pub dst_color: BlendFactor,
    /// Colour blend operation.
    pub color_op: BlendOperation,
    /// Source alpha factor.
    pub src_alpha: BlendFactor,
    /// Destination alpha factor.
    pub dst_alpha: BlendFactor,
    /// Alpha blend operation.
    pub alpha_op: BlendOperation,
    /// Colour write mask bits (`R=1, G=2, B=4, A=8`).
    pub write_mask: u8,
}

/// Per-render-target blend state (`D3D12_BLEND_DESC`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// `IndependentBlendEnable`: every target uses its own entry, otherwise
    /// entry 0 applies to all targets.
    pub independent_blend: bool,
    /// `AlphaToCoverageEnable`.
    pub alpha_to_coverage: bool,
    /// One entry per target (1..=`MAX_BLEND_ATTACHMENTS_V3`).
    pub attachments: Vec<BlendAttachment>,
}

/// One stencil face (`D3D12_DEPTH_STENCILOP_DESC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// Operation on stencil test failure.
    pub fail: StencilOperation,
    /// Operation on depth test failure.
    pub depth_fail: StencilOperation,
    /// Operation on both tests passing.
    pub pass: StencilOperation,
    /// Stencil comparison function.
    pub func: CompareFunction,
    /// Stencil read mask.
    pub read_mask: u8,
    /// Stencil write mask.
    pub write_mask: u8,
}

/// Depth/stencil state (`D3D12_DEPTH_STENCIL_DESC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilState {
    /// Depth testing enabled.
    pub depth_enable: bool,
    /// Depth writes enabled.
    pub depth_write: bool,
    /// Depth comparison function.
    pub depth_func: CompareFunction,
    /// Stencil testing enabled.
    pub stencil_enable: bool,
    /// Front-face stencil operations.
    pub front: StencilFaceState,
    /// Back-face stencil operations.
    pub back: StencilFaceState,
}

/// Rasterizer state (`D3D12_RASTERIZER_DESC`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterizerState {
    /// Fill mode.
    pub fill: FillMode,
    /// Cull mode.
    pub cull: CullMode,
    /// Front-face winding.
    pub front_face: FrontFaceWinding,
    /// Constant depth bias (`DepthBias`, an `INT`).
    pub depth_bias: i32,
    /// Slope-scaled depth bias (`SlopeScaledDepthBias`).
    pub slope_scaled_depth_bias: f32,
    /// Maximum clamped depth bias (`DepthBiasClamp`).
    pub depth_bias_clamp: f32,
    /// `DepthClipEnable`.
    pub depth_clip: bool,
}

/// Stencil reference value (`OMSetStencilRef`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilRef {
    /// Reference value written by `REPLACE` and compared by the stencil func.
    pub value: u8,
}

/// Blend factor constant (`OMSetBlendFactor`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendFactorValue {
    /// RGBA blend factors.
    pub value: [f32; 4],
}

/// Clears one colour attachment of the open render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearRenderTargetView {
    /// Colour attachment slot in the open pass (0 ..= 7).
    pub slot: u16,
    /// Optional scissor-style rectangle; `None` clears the whole attachment.
    pub rect: Option<ScissorRect>,
    /// Clear colour.
    pub color: [f32; 4],
}

/// Clears the depth and/or stencil plane of the open render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearDepthStencilView {
    /// Depth clear value, when the depth plane is cleared.
    pub depth: Option<f32>,
    /// Stencil clear value, when the stencil plane is cleared.
    pub stencil: Option<u32>,
    /// Optional rectangle; `None` clears the whole attachment.
    pub rect: Option<ScissorRect>,
}

/// Pitched upload-heap to private-texture copy.
///
/// Mirrors `D3D12_BOX` plus `D3D12_PLACED_SUBRESOURCE_FOOTPRINT` for one mip,
/// slice and origin. `bytes_per_image` covers one array slice of a 3D/array
/// copy; it is zero for a single-slice copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyBufferToTexture {
    /// Source buffer allocation ID; never zero.
    pub allocation_id: u64,
    /// Source generation; never zero.
    pub generation: u32,
    /// Destination texture allocation ID; never zero.
    pub destination_allocation_id: u64,
    /// Destination generation; never zero.
    pub destination_generation: u32,
    /// Byte offset of the first texel; 4-byte aligned.
    pub offset: u64,
    /// Source row pitch in bytes; nonzero.
    pub bytes_per_row: u32,
    /// Source slice pitch in bytes; zero or at least one full image.
    pub bytes_per_image: u32,
    /// Copy box size in texels; every component nonzero.
    pub extent: [u32; 3],
    /// Destination mip level.
    pub mip_level: u16,
    /// Destination array slice.
    pub slice: u16,
    /// Destination box origin in texels.
    pub origin: [u32; 3],
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
            Self::ExecuteIndirectDraw(_) => OP_EXECUTE_INDIRECT_DRAW,
            Self::ExecuteIndirectDrawIndexed(_) => OP_EXECUTE_INDIRECT_DRAW_INDEXED,
            Self::ExecuteIndirectDispatchMesh(_) => OP_EXECUTE_INDIRECT_DISPATCH_MESH,
            Self::ExecuteIndirectDispatch(_) => OP_EXECUTE_INDIRECT_DISPATCH,
            Self::CopyTextureRegion(_) => OP_COPY_TEXTURE_REGION,
            Self::ResolveSubresource(_) => OP_RESOLVE_SUBRESOURCE,
            Self::SetBlendState(_) => OP_SET_BLEND_STATE,
            Self::SetDepthStencilState(_) => OP_SET_DEPTH_STENCIL_STATE,
            Self::SetRasterizerState(_) => OP_SET_RASTERIZER_STATE,
            Self::SetStencilRef(_) => OP_SET_STENCIL_REF,
            Self::SetBlendFactor(_) => OP_SET_BLEND_FACTOR,
            Self::ClearRenderTargetView(_) => OP_CLEAR_RENDER_TARGET_VIEW,
            Self::ClearDepthStencilView(_) => OP_CLEAR_DEPTH_STENCIL_VIEW,
            Self::CopyBufferToTexture(_) => OP_COPY_BUFFER_TO_TEXTURE,
            Self::Present(_) => OP_PRESENT,
        }
    }
}

/// Re-exported barrier entry so producers do not import two modules.
pub type Barrier = BarrierEntry;

const FRAME_HEADER_LEN: usize = 20;
const LIST_BLOCK_LEN: usize = 12;
const KIND_SUBMISSION: u16 = 1;
const BINDING_LEN: usize = 32;
const RESOURCE_HANDLE_LEN: usize = 8;
/// Table-entry encoding tag in the descriptor-table header word at offset 12.
///
/// `0` is the legacy buffer-range encoding (32 bytes per entry) and decodes old
/// frames byte-compatibly. `1` is a resource-handle encoding (8 bytes per
/// entry) whose meaning is fixed by the table kind: `Srv`/`Uav` entries are
/// registered texture handles and `Sampler` entries are registered sampler
/// handles. `2` is the per-entry tagged encoding, used only when a table mixes
/// buffer and registered-resource entries; every entry then starts with a
/// one-byte kind tag.
const DESCRIPTOR_ENTRY_BUFFER: u16 = 0;
const DESCRIPTOR_ENTRY_RESOURCE: u16 = 1;
const DESCRIPTOR_ENTRY_TAGGED: u16 = 2;
/// Width of one per-entry kind tag inside a [`DESCRIPTOR_ENTRY_TAGGED`] table.
const DESCRIPTOR_ENTRY_TAG_LEN: usize = 1;
/// Per-entry kind tag for a 32-byte buffer range.
const DESCRIPTOR_ENTRY_KIND_BUFFER: u8 = 0;
/// Per-entry kind tag for a registered texture/surface handle.
const DESCRIPTOR_ENTRY_KIND_TEXTURE: u8 = 1;
/// Per-entry kind tag for a registered sampler-state handle.
const DESCRIPTOR_ENTRY_KIND_SAMPLER: u8 = 2;
const BARRIER_ENTRY_LEN: usize = 40;
const COLOR_ATTACHMENT_LEN: usize = 56;
const DEPTH_ATTACHMENT_LEN: usize = 36;
const VERTEX_BUFFER_LEN: usize = 32;

const DISPATCH_1D_LEN: usize = 24;
const DISPATCH_3D_LEN: usize = 40;
const DRAW_INDEXED_LEN: usize = 24;
const DRAW_LEN: usize = 20;
const DISPATCH_MESH_LEN: usize = 28;
const BARRIER_HEADER_LEN: usize = 8;
const SET_PIPELINE_LEN: usize = 12;
const ROOT_CONSTANTS_HEADER_LEN: usize = 8;
const DESCRIPTOR_TABLE_HEADER_LEN: usize = 14;
const BEGIN_RENDER_PASS_HEADER_LEN: usize = 12;
const END_RENDER_PASS_LEN: usize = 4;
const SET_VIEWPORT_LEN: usize = 52;
const SET_SCISSOR_LEN: usize = 20;
const PIPELINE_REF_LEN: usize = 12;
const SET_TOPOLOGY_LEN: usize = 8;
const VERTEX_BUFFERS_HEADER_LEN: usize = 12;
const SET_INDEX_BUFFER_LEN: usize = 36;
const SET_ROOT_BUFFER_VIEW_LEN: usize = 40;
const DRAW_INSTANCED_LEN: usize = 20;
const DRAW_INDEXED_INSTANCED_LEN: usize = 24;
const COPY_BUFFER_LEN: usize = 60;
const COPY_TEXTURE_LEN: usize = 52;
const INDIRECT_ARGUMENTS_LEN: usize = 56;
const EXECUTE_INDIRECT_DRAW_LEN: usize = 4 + INDIRECT_ARGUMENTS_LEN;
const EXECUTE_INDIRECT_DISPATCH_MESH_LEN: usize = 4 + 8 + INDIRECT_ARGUMENTS_LEN;
const EXECUTE_INDIRECT_DISPATCH_LEN: usize = 4 + 8 + INDIRECT_ARGUMENTS_LEN + 16;
const COPY_TEXTURE_REGION_LEN: usize = 4 + 72;
const RESOLVE_SUBRESOURCE_LEN: usize = 4 + 32;
const SET_BLEND_STATE_HEADER_LEN: usize = 12;
const BLEND_ATTACHMENT_LEN: usize = 8;
const SET_DEPTH_STENCIL_STATE_LEN: usize = 24;
const SET_RASTERIZER_STATE_LEN: usize = 24;
const SET_STENCIL_REF_LEN: usize = 8;
const SET_BLEND_FACTOR_LEN: usize = 24;
const CLEAR_RENDER_TARGET_VIEW_LEN: usize = 40;
const CLEAR_DEPTH_STENCIL_VIEW_LEN: usize = 32;
const COPY_BUFFER_TO_TEXTURE_LEN: usize = 76;
const PRESENT_LEN: usize = 20;

/// Checked little-endian cursor over a version 3 frame or payload.
///
/// Every read is bounds-checked and returns [`WireError::Truncated`] instead
/// of panicking. The cursor never borrows the destination storage, so a reader
/// can fill a caller-owned [`SubmissionV3`] while reading from a separate
/// frame slice.
#[derive(Debug, Clone)]
pub struct ReaderV3<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ReaderV3<'a> {
    /// Creates a cursor at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Number of bytes already consumed.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of bytes left to read.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    /// Returns true when every byte has been consumed.
    pub fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    /// Copies exactly `N` bytes into an array.
    pub fn take<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let mut value = [0u8; N];
        value.copy_from_slice(self.bytes(N)?);
        Ok(value)
    }

    /// Borrows the next `len` bytes.
    pub fn bytes(&mut self, len: usize) -> Result<&'a [u8], WireError> {
        let end = self.position.checked_add(len).ok_or(WireError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(WireError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    /// Reads one byte.
    pub fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take::<1>()?[0])
    }

    /// Reads a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    /// Reads a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    /// Reads a little-endian `i32`.
    pub fn i32(&mut self) -> Result<i32, WireError> {
        Ok(i32::from_le_bytes(self.take::<4>()?))
    }

    /// Reads a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    /// Reads an IEEE-754 `f32` from its little-endian bit pattern.
    pub fn f32(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// Reads an IEEE-754 `f64` from its little-endian bit pattern.
    pub fn f64(&mut self) -> Result<f64, WireError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn reserved_u16(&mut self) -> Result<(), WireError> {
        if self.u16()? == 0 {
            Ok(())
        } else {
            Err(WireError::NonZeroReserved)
        }
    }

    fn reserved_u32(&mut self) -> Result<(), WireError> {
        if self.u32()? == 0 {
            Ok(())
        } else {
            Err(WireError::NonZeroReserved)
        }
    }
}

/// Encodes one version 3 submission as a complete frame.
///
/// The write side enforces the same rules as [`decode_submission_v3`], so a
/// frame this function returns always decodes. Sequence monotonicity across
/// frames remains the session layer's responsibility.
pub fn encode_submission_v3(submission: &SubmissionV3) -> Result<Vec<u8>, WireError> {
    validate_sequence(submission.sequence)?;
    if let Some(list) = submission.list {
        validate_list_identity(list)?;
    }
    if submission.commands.len() > MAX_COMMANDS_V3 {
        return Err(WireError::TooManyCommandsV3(submission.commands.len()));
    }

    let mut payload_len = LIST_BLOCK_LEN;
    for command in &submission.commands {
        payload_len = payload_len
            .checked_add(command_encoded_len(command)?)
            .ok_or(WireError::FrameTooLarge)?;
    }
    let payload_len_field = u32::try_from(payload_len).map_err(|_| WireError::FrameTooLarge)?;
    let command_count = u16::try_from(submission.commands.len())
        .map_err(|_| WireError::TooManyCommandsV3(submission.commands.len()))?;
    let frame_len = FRAME_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(WireError::FrameTooLarge)?;

    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&MAGIC);
    write_u16(&mut out, PROTOCOL_VERSION_V3);
    write_u16(&mut out, KIND_SUBMISSION);
    write_u32(&mut out, submission.sequence);
    write_u16(&mut out, command_count);
    write_u16(&mut out, 0);
    write_u32(&mut out, payload_len_field);
    match submission.list {
        Some(list) => {
            write_u64(&mut out, list.list_id);
            write_u32(&mut out, list.list_generation);
        }
        None => {
            write_u64(&mut out, 0);
            write_u32(&mut out, 0);
        }
    }
    for command in &submission.commands {
        write_command(&mut out, command);
    }
    Ok(out)
}

/// Decodes one complete version 3 submission frame.
///
/// Returns no submission on any error. Counts are checked before the decoder
/// reserves storage. A version 2 frame is refused with
/// [`WireError::UnsupportedVersion`].
pub fn decode_submission_v3(bytes: &[u8]) -> Result<SubmissionV3, WireError> {
    let mut submission = SubmissionV3 {
        sequence: 0,
        list: None,
        commands: Vec::new(),
    };
    decode_submission_v3_into(&mut submission, bytes)?;
    Ok(submission)
}

/// Decodes one version 3 submission frame into caller-owned storage.
///
/// The destination command vector is reused in place, and a command whose
/// opcode and variant already match keeps its nested payload vectors, so
/// decoding the same or a smaller submission after a warm decode does not
/// reallocate. On error the destination contents are unspecified and must not
/// be used; the next successful decode fully overwrites them.
pub fn decode_submission_v3_into(out: &mut SubmissionV3, bytes: &[u8]) -> Result<(), WireError> {
    let (header, payload) = FrameHeaderV3::parse(bytes)?;
    if header.kind != KIND_SUBMISSION {
        return Err(WireError::UnexpectedKind(header.kind));
    }
    if header.command_count > MAX_COMMANDS_V3 {
        return Err(WireError::TooManyCommandsV3(header.command_count));
    }

    let mut reader = ReaderV3::new(payload);
    let list_id = reader.u64()?;
    let list_generation = reader.u32()?;
    let list = decode_list_identity(list_id, list_generation)?;

    let count = header.command_count;
    while out.commands.len() < count {
        // A payload-free placeholder keeps growth confined to the first decode;
        // `decode_command_into` replaces it with the recorded variant.
        out.commands.push(CommandV3::EndRenderPass);
    }
    for index in 0..count {
        decode_command_into(&mut reader, &mut out.commands[index])?;
    }
    out.commands.truncate(count);

    if !reader.is_empty() {
        return Err(WireError::TrailingBytes);
    }
    out.sequence = header.sequence;
    out.list = list;
    Ok(())
}

struct FrameHeaderV3 {
    kind: u16,
    sequence: u32,
    command_count: usize,
}

impl FrameHeaderV3 {
    fn parse(bytes: &[u8]) -> Result<(Self, &[u8]), WireError> {
        if bytes.len() < FRAME_HEADER_LEN {
            return Err(WireError::Truncated);
        }

        let mut reader = ReaderV3::new(bytes);
        let magic = reader.take::<4>()?;
        if magic != MAGIC {
            return Err(WireError::BadMagic);
        }

        let version = reader.u16()?;
        if version != PROTOCOL_VERSION_V3 {
            return Err(WireError::UnsupportedVersion(version));
        }

        let kind = reader.u16()?;
        let sequence = reader.u32()?;
        validate_sequence(sequence)?;
        let command_count = usize::from(reader.u16()?);
        reader.reserved_u16()?;
        let payload_len = usize::try_from(reader.u32()?).map_err(|_| WireError::FrameTooLarge)?;

        let remaining = &bytes[FRAME_HEADER_LEN..];
        match remaining.len().cmp(&payload_len) {
            std::cmp::Ordering::Less => Err(WireError::Truncated),
            std::cmp::Ordering::Greater => Err(WireError::TrailingBytes),
            std::cmp::Ordering::Equal => Ok((
                Self {
                    kind,
                    sequence,
                    command_count,
                },
                remaining,
            )),
        }
    }
}

fn decode_command_into(reader: &mut ReaderV3<'_>, slot: &mut CommandV3) -> Result<(), WireError> {
    let opcode = reader.u16()?;
    let flags = reader.u16()?;
    if flags != 0 {
        return Err(WireError::UnsupportedFlags(flags));
    }
    if slot.opcode() != opcode {
        *slot = empty_command(opcode)?;
    }
    read_command_payload(reader, slot)
}

fn empty_command(opcode: u16) -> Result<CommandV3, WireError> {
    Ok(match opcode {
        OP_DISPATCH_1D => CommandV3::Dispatch1d(Dispatch1d {
            pipeline_id: 0,
            threads: 0,
            threads_per_group: 0,
            bindings: Vec::new(),
        }),
        OP_DISPATCH_3D => CommandV3::Dispatch3d(Dispatch3d {
            pipeline_id: 0,
            group_counts: [0; 3],
            threads_per_group: [0; 3],
            bindings: Vec::new(),
        }),
        OP_DRAW_INDEXED => CommandV3::DrawIndexed(DrawIndexedState {
            index_count: 0,
            instance_count: 0,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }),
        OP_DRAW => CommandV3::Draw(DrawState {
            vertex_count: 0,
            instance_count: 0,
            start_vertex: 0,
            start_instance: 0,
        }),
        OP_DISPATCH_MESH => CommandV3::DispatchMesh(DispatchMesh {
            pipeline_id: 0,
            group_counts: [0; 3],
            bindings: Vec::new(),
        }),
        OP_RESOURCE_BARRIER => CommandV3::ResourceBarrier(ResourceBarrier {
            entries: Vec::new(),
        }),
        OP_SET_PIPELINE => CommandV3::SetPipeline(SetPipeline { pipeline_id: 0 }),
        OP_ROOT_CONSTANTS => CommandV3::SetRootConstants(RootConstantsV3 {
            slot: 0,
            bytes: Vec::new(),
        }),
        OP_DESCRIPTOR_TABLE => CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Cbv,
            bindings: Vec::new(),
        }),
        OP_BEGIN_RENDER_PASS => CommandV3::BeginRenderPass(BeginRenderPass {
            colors: Vec::new(),
            depth_stencil: None,
        }),
        OP_END_RENDER_PASS => CommandV3::EndRenderPass,
        OP_SET_VIEWPORT => CommandV3::SetViewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            near: 0.0,
            far: 0.0,
        }),
        OP_SET_SCISSOR => CommandV3::SetScissor(ScissorRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        }),
        OP_SET_GRAPHICS_PIPELINE => CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id: 0 }),
        OP_SET_MESH_PIPELINE => CommandV3::SetMeshPipeline(PipelineRef { pipeline_id: 0 }),
        OP_SET_COMPUTE_PIPELINE => CommandV3::SetComputePipeline(PipelineRef { pipeline_id: 0 }),
        OP_SET_PRIMITIVE_TOPOLOGY => CommandV3::SetPrimitiveTopology(PrimitiveTopology::PointList),
        OP_SET_VERTEX_BUFFERS => CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers: Vec::new(),
        }),
        OP_SET_INDEX_BUFFER => CommandV3::SetIndexBuffer(IndexBufferBinding {
            allocation_id: 0,
            generation: 0,
            offset: 0,
            length: 0,
            format: IndexFormat::Uint16,
        }),
        OP_SET_ROOT_BUFFER_VIEW => CommandV3::SetRootBufferView(RootBufferView {
            slot: 0,
            allocation_id: 0,
            generation: 0,
            offset: 0,
            length: 0,
        }),
        OP_DRAW_INSTANCED => CommandV3::DrawInstanced(DrawInstanced {
            vertex_count_per_instance: 0,
            instance_count: 0,
            start_vertex: 0,
            start_instance: 0,
        }),
        OP_DRAW_INDEXED_INSTANCED => CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
            index_count_per_instance: 0,
            instance_count: 0,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }),
        OP_COPY_BUFFER => CommandV3::CopyBuffer(CopyBuffer {
            src_allocation_id: 0,
            src_generation: 0,
            src_offset: 0,
            dst_allocation_id: 0,
            dst_generation: 0,
            dst_offset: 0,
            length: 0,
        }),
        OP_COPY_TEXTURE => CommandV3::CopyTexture(CopyTexture {
            src_allocation_id: 0,
            src_generation: 0,
            src_mip: 0,
            src_slice: 0,
            dst_allocation_id: 0,
            dst_generation: 0,
            dst_mip: 0,
            dst_slice: 0,
            width: 0,
            height: 0,
            depth: 0,
        }),
        OP_PRESENT => CommandV3::Present(Present {
            surface_id: 0,
            sync_interval: 0,
        }),
        OP_EXECUTE_INDIRECT_DRAW => CommandV3::ExecuteIndirectDraw(ExecuteIndirectDraw {
            arguments: empty_indirect_arguments(),
        }),
        OP_EXECUTE_INDIRECT_DRAW_INDEXED => {
            CommandV3::ExecuteIndirectDrawIndexed(ExecuteIndirectDrawIndexed {
                arguments: empty_indirect_arguments(),
            })
        }
        OP_EXECUTE_INDIRECT_DISPATCH_MESH => {
            CommandV3::ExecuteIndirectDispatchMesh(ExecuteIndirectDispatchMesh {
                pipeline_id: 0,
                arguments: empty_indirect_arguments(),
            })
        }
        OP_EXECUTE_INDIRECT_DISPATCH => {
            CommandV3::ExecuteIndirectDispatch(ExecuteIndirectDispatch {
                pipeline_id: 0,
                arguments: empty_indirect_arguments(),
                threads_per_group: [0; 3],
            })
        }
        OP_COPY_TEXTURE_REGION => CommandV3::CopyTextureRegion(CopyTextureRegion {
            src: empty_subresource(),
            dst: empty_subresource(),
            src_origin: [0; 3],
            dst_origin: [0; 3],
            size: [0; 3],
        }),
        OP_RESOLVE_SUBRESOURCE => CommandV3::ResolveSubresource(ResolveSubresource {
            src: empty_subresource(),
            dst: empty_subresource(),
        }),
        OP_SET_BLEND_STATE => CommandV3::SetBlendState(BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: Vec::new(),
        }),
        OP_SET_DEPTH_STENCIL_STATE => CommandV3::SetDepthStencilState(DepthStencilState {
            depth_enable: false,
            depth_write: false,
            depth_func: CompareFunction::Always,
            stencil_enable: false,
            front: empty_stencil_face(),
            back: empty_stencil_face(),
        }),
        OP_SET_RASTERIZER_STATE => CommandV3::SetRasterizerState(RasterizerState {
            fill: FillMode::Solid,
            cull: CullMode::None,
            front_face: FrontFaceWinding::Clockwise,
            depth_bias: 0,
            slope_scaled_depth_bias: 0.0,
            depth_bias_clamp: 0.0,
            depth_clip: true,
        }),
        OP_SET_STENCIL_REF => CommandV3::SetStencilRef(StencilRef { value: 0 }),
        OP_SET_BLEND_FACTOR => CommandV3::SetBlendFactor(BlendFactorValue { value: [0.0; 4] }),
        OP_CLEAR_RENDER_TARGET_VIEW => CommandV3::ClearRenderTargetView(ClearRenderTargetView {
            slot: 0,
            rect: None,
            color: [0.0; 4],
        }),
        OP_CLEAR_DEPTH_STENCIL_VIEW => CommandV3::ClearDepthStencilView(ClearDepthStencilView {
            depth: None,
            stencil: None,
            rect: None,
        }),
        OP_COPY_BUFFER_TO_TEXTURE => CommandV3::CopyBufferToTexture(CopyBufferToTexture {
            allocation_id: 0,
            generation: 0,
            destination_allocation_id: 0,
            destination_generation: 0,
            offset: 0,
            bytes_per_row: 0,
            bytes_per_image: 0,
            extent: [0; 3],
            mip_level: 0,
            slice: 0,
            origin: [0; 3],
        }),
        other => return Err(WireError::UnknownOpcode(other)),
    })
}

fn empty_indirect_arguments() -> IndirectArguments {
    IndirectArguments {
        allocation_id: 0,
        generation: 0,
        offset: 0,
        max_count: 0,
        count: None,
    }
}

fn empty_subresource() -> TextureSubresource {
    TextureSubresource {
        allocation_id: 0,
        generation: 0,
        mip_level: 0,
        slice: 0,
    }
}

fn empty_stencil_face() -> StencilFaceState {
    StencilFaceState {
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
        func: CompareFunction::Always,
        read_mask: 0,
        write_mask: 0,
    }
}

fn read_command_payload(
    reader: &mut ReaderV3<'_>,
    command: &mut CommandV3,
) -> Result<(), WireError> {
    match command {
        CommandV3::Dispatch1d(dispatch) => read_dispatch_1d(reader, dispatch),
        CommandV3::Dispatch3d(dispatch) => read_dispatch_3d(reader, dispatch),
        CommandV3::DrawIndexed(draw) => read_draw_indexed(reader, draw),
        CommandV3::Draw(draw) => read_draw(reader, draw),
        CommandV3::DispatchMesh(dispatch) => read_dispatch_mesh(reader, dispatch),
        CommandV3::ResourceBarrier(barrier) => read_barrier(reader, barrier),
        CommandV3::SetPipeline(set_pipeline) => read_set_pipeline(reader, set_pipeline),
        CommandV3::SetRootConstants(constants) => read_root_constants(reader, constants),
        CommandV3::SetDescriptorTable(table) => read_descriptor_table(reader, table),
        CommandV3::BeginRenderPass(pass) => read_begin_render_pass(reader, pass),
        CommandV3::EndRenderPass => Ok(()),
        CommandV3::SetViewport(viewport) => read_viewport(reader, viewport),
        CommandV3::SetScissor(scissor) => read_scissor(reader, scissor),
        CommandV3::SetGraphicsPipeline(pipeline)
        | CommandV3::SetMeshPipeline(pipeline)
        | CommandV3::SetComputePipeline(pipeline) => read_pipeline_ref(reader, pipeline),
        CommandV3::SetPrimitiveTopology(topology) => read_topology(reader, topology),
        CommandV3::SetVertexBuffers(set) => read_vertex_buffers(reader, set),
        CommandV3::SetIndexBuffer(binding) => read_index_buffer(reader, binding),
        CommandV3::SetRootBufferView(view) => read_root_buffer_view(reader, view),
        CommandV3::DrawInstanced(draw) => read_draw_instanced(reader, draw),
        CommandV3::DrawIndexedInstanced(draw) => read_draw_indexed_instanced(reader, draw),
        CommandV3::CopyBuffer(copy) => read_copy_buffer(reader, copy),
        CommandV3::CopyTexture(copy) => read_copy_texture(reader, copy),
        CommandV3::ExecuteIndirectDraw(draw) => read_execute_indirect_draw(reader, draw),
        CommandV3::ExecuteIndirectDrawIndexed(draw) => {
            read_execute_indirect_draw_indexed(reader, draw)
        }
        CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
            read_execute_indirect_dispatch_mesh(reader, dispatch)
        }
        CommandV3::ExecuteIndirectDispatch(dispatch) => {
            read_execute_indirect_dispatch(reader, dispatch)
        }
        CommandV3::CopyTextureRegion(copy) => read_copy_texture_region(reader, copy),
        CommandV3::ResolveSubresource(resolve) => read_resolve_subresource(reader, resolve),
        CommandV3::SetBlendState(state) => read_blend_state(reader, state),
        CommandV3::SetDepthStencilState(state) => read_depth_stencil_state(reader, state),
        CommandV3::SetRasterizerState(state) => read_rasterizer_state(reader, state),
        CommandV3::SetStencilRef(reference) => {
            let value = reader.u16()?;
            reader.reserved_u16()?;
            if value > u16::from(u8::MAX) {
                return Err(WireError::InvalidEnumCode {
                    field: "stencil reference",
                    code: value,
                });
            }
            reference.value = value as u8;
            Ok(())
        }
        CommandV3::SetBlendFactor(factor) => {
            for component in &mut factor.value {
                *component = reader.f32()?;
            }
            reader.reserved_u32()?;
            Ok(())
        }
        CommandV3::ClearRenderTargetView(clear) => read_clear_render_target_view(reader, clear),
        CommandV3::ClearDepthStencilView(clear) => read_clear_depth_stencil_view(reader, clear),
        CommandV3::CopyBufferToTexture(copy) => read_copy_buffer_to_texture(reader, copy),
        CommandV3::Present(present) => read_present(reader, present),
    }
}

fn read_dispatch_1d(reader: &mut ReaderV3<'_>, dispatch: &mut Dispatch1d) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let threads = reader.u32()?;
    let threads_per_group = reader.u32()?;
    validate_threads(threads, threads_per_group)?;
    read_bindings_into(reader, &mut dispatch.bindings)?;
    dispatch.pipeline_id = pipeline_id;
    dispatch.threads = threads;
    dispatch.threads_per_group = threads_per_group;
    Ok(())
}

fn read_dispatch_3d(reader: &mut ReaderV3<'_>, dispatch: &mut Dispatch3d) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let group_counts = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(group_counts)?;
    let threads_per_group = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(threads_per_group)?;
    let group_count = checked_dimension_product(group_counts);
    let local_count = checked_dimension_product(threads_per_group);
    if group_count
        .and_then(|groups| groups.checked_mul(local_count?))
        .is_none()
    {
        return Err(WireError::ThreadCountOverflow);
    }
    read_bindings_into(reader, &mut dispatch.bindings)?;
    dispatch.pipeline_id = pipeline_id;
    dispatch.group_counts = group_counts;
    dispatch.threads_per_group = threads_per_group;
    Ok(())
}

fn read_draw_indexed(
    reader: &mut ReaderV3<'_>,
    draw: &mut DrawIndexedState,
) -> Result<(), WireError> {
    let index_count = reader.u32()?;
    if index_count == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    draw.index_count = index_count;
    draw.instance_count = instance_count;
    draw.start_index = reader.u32()?;
    draw.base_vertex = reader.i32()?;
    draw.start_instance = reader.u32()?;
    Ok(())
}

fn read_draw(reader: &mut ReaderV3<'_>, draw: &mut DrawState) -> Result<(), WireError> {
    let vertex_count = reader.u32()?;
    if vertex_count == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    draw.vertex_count = vertex_count;
    draw.instance_count = instance_count;
    draw.start_vertex = reader.u32()?;
    draw.start_instance = reader.u32()?;
    Ok(())
}

fn read_dispatch_mesh(
    reader: &mut ReaderV3<'_>,
    dispatch: &mut DispatchMesh,
) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let group_counts = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(group_counts)?;
    if checked_dimension_product(group_counts).is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    read_bindings_into(reader, &mut dispatch.bindings)?;
    dispatch.pipeline_id = pipeline_id;
    dispatch.group_counts = group_counts;
    Ok(())
}

fn read_barrier(reader: &mut ReaderV3<'_>, barrier: &mut ResourceBarrier) -> Result<(), WireError> {
    let count = usize::from(reader.u16()?);
    reader.reserved_u16()?;
    if count == 0 {
        return Err(WireError::ZeroBarrierCount);
    }
    if count > MAX_BARRIERS_V3 {
        return Err(WireError::TooManyBarriersV3(count));
    }
    barrier.entries.clear();
    for _ in 0..count {
        let allocation_id = reader.u64()?;
        let generation = reader.u32()?;
        reader.reserved_u32()?;
        let offset = reader.u64()?;
        let length = reader.u64()?;
        let before_state = reader.u32()?;
        let after_state = reader.u32()?;
        validate_range(allocation_id, generation, offset, length)?;
        barrier.entries.push(BarrierEntry {
            allocation_id,
            generation,
            offset,
            length,
            before_state,
            after_state,
        });
    }
    Ok(())
}

fn read_set_pipeline(
    reader: &mut ReaderV3<'_>,
    set_pipeline: &mut SetPipeline,
) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    set_pipeline.pipeline_id = pipeline_id;
    Ok(())
}

fn read_root_constants(
    reader: &mut ReaderV3<'_>,
    constants: &mut RootConstantsV3,
) -> Result<(), WireError> {
    let slot = reader.u16()?;
    let byte_len = usize::from(reader.u16()?);
    if byte_len == 0 {
        return Err(WireError::ZeroRootConstants);
    }
    if byte_len > MAX_ROOT_CONSTANT_BYTES_V3 {
        return Err(WireError::RootConstantsTooLarge(byte_len));
    }
    let bytes = reader.bytes(byte_len)?;
    constants.slot = slot;
    constants.bytes.clear();
    constants.bytes.extend_from_slice(bytes);
    Ok(())
}

fn read_descriptor_table(
    reader: &mut ReaderV3<'_>,
    table: &mut DescriptorTableV3,
) -> Result<(), WireError> {
    let set = reader.u16()?;
    let base_register = reader.u16()?;
    let kind = descriptor_kind_from_code(reader.u16()?)?;
    let count = usize::from(reader.u16()?);
    let entry_tag = reader.u16()?;
    if count == 0 {
        return Err(WireError::ZeroDescriptorCount);
    }
    if count > MAX_DESCRIPTOR_BINDINGS_V3 {
        return Err(WireError::TooManyDescriptorBindingsV3(count));
    }
    table.bindings.clear();
    match entry_tag {
        DESCRIPTOR_ENTRY_BUFFER => {
            for _ in 0..count {
                table
                    .bindings
                    .push(DescriptorTableEntryV3::Buffer(read_binding(reader)?));
            }
        }
        DESCRIPTOR_ENTRY_RESOURCE => {
            let sampler = match kind {
                DescriptorKind::Srv | DescriptorKind::Uav => false,
                DescriptorKind::Sampler => true,
                DescriptorKind::Cbv => return Err(WireError::DescriptorEntryMismatch),
            };
            for _ in 0..count {
                let handle = reader.u64()?;
                if handle == 0 {
                    return Err(WireError::ZeroDescriptorHandle);
                }
                table.bindings.push(if sampler {
                    DescriptorTableEntryV3::Sampler(handle)
                } else {
                    DescriptorTableEntryV3::Texture(handle)
                });
            }
        }
        DESCRIPTOR_ENTRY_TAGGED => {
            for index in 0..count {
                let entry_kind = reader.u8()?;
                match entry_kind {
                    DESCRIPTOR_ENTRY_KIND_BUFFER => {
                        if matches!(kind, DescriptorKind::Sampler) {
                            return Err(WireError::DescriptorEntryMismatch);
                        }
                        table
                            .bindings
                            .push(DescriptorTableEntryV3::Buffer(read_binding(reader)?));
                    }
                    DESCRIPTOR_ENTRY_KIND_TEXTURE => {
                        if !matches!(kind, DescriptorKind::Srv | DescriptorKind::Uav) {
                            return Err(WireError::DescriptorEntryMismatch);
                        }
                        let handle = reader.u64()?;
                        if handle == 0 {
                            return Err(WireError::ZeroDescriptorHandle);
                        }
                        table.bindings.push(DescriptorTableEntryV3::Texture(handle));
                    }
                    DESCRIPTOR_ENTRY_KIND_SAMPLER => {
                        if kind != DescriptorKind::Sampler {
                            return Err(WireError::DescriptorEntryMismatch);
                        }
                        let handle = reader.u64()?;
                        if handle == 0 {
                            return Err(WireError::ZeroDescriptorHandle);
                        }
                        table.bindings.push(DescriptorTableEntryV3::Sampler(handle));
                    }
                    other => {
                        return Err(WireError::InvalidEnumCode {
                            field: descriptor_entry_tag_field(index),
                            code: u16::from(other),
                        });
                    }
                }
            }
        }
        other => {
            return Err(WireError::InvalidEnumCode {
                field: "descriptor table entry tag",
                code: other,
            });
        }
    }
    table.set = set;
    table.base_register = base_register;
    table.kind = kind;
    Ok(())
}

fn read_begin_render_pass(
    reader: &mut ReaderV3<'_>,
    pass: &mut BeginRenderPass,
) -> Result<(), WireError> {
    let color_count = usize::from(reader.u16()?);
    let depth_count = usize::from(reader.u16()?);
    reader.reserved_u32()?;
    if color_count == 0 {
        return Err(WireError::ZeroColorAttachments);
    }
    if color_count > MAX_COLOR_ATTACHMENTS {
        return Err(WireError::TooManyColorAttachments(color_count));
    }
    if depth_count > 1 {
        return Err(WireError::TooManyDepthAttachments(depth_count));
    }
    pass.colors.clear();
    for _ in 0..color_count {
        pass.colors.push(read_color_attachment(reader)?);
    }
    pass.depth_stencil = if depth_count == 1 {
        Some(read_depth_attachment(reader)?)
    } else {
        None
    };
    Ok(())
}

fn read_color_attachment(reader: &mut ReaderV3<'_>) -> Result<ColorAttachment, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let mip_level = reader.u16()?;
    let slice = reader.u16()?;
    let load = load_op_from_code(reader.u16()?)?;
    let store = store_op_from_code(reader.u16()?)?;
    reader.reserved_u32()?;
    let clear_color = [reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?];
    Ok(ColorAttachment {
        allocation_id,
        generation,
        mip_level,
        slice,
        load,
        store,
        clear_color,
    })
}

fn read_depth_attachment(reader: &mut ReaderV3<'_>) -> Result<DepthStencilAttachment, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let mip_level = reader.u16()?;
    let slice = reader.u16()?;
    let depth_load = load_op_from_code(reader.u16()?)?;
    let depth_store = store_op_from_code(reader.u16()?)?;
    let clear_depth = reader.f32()?;
    let stencil_load = load_op_from_code(reader.u16()?)?;
    let stencil_store = store_op_from_code(reader.u16()?)?;
    let clear_stencil = reader.u32()?;
    reader.reserved_u32()?;
    Ok(DepthStencilAttachment {
        allocation_id,
        generation,
        mip_level,
        slice,
        depth_load,
        depth_store,
        clear_depth,
        stencil_load,
        stencil_store,
        clear_stencil,
    })
}

fn read_viewport(reader: &mut ReaderV3<'_>, viewport: &mut Viewport) -> Result<(), WireError> {
    viewport.x = reader.f64()?;
    viewport.y = reader.f64()?;
    viewport.width = reader.f64()?;
    viewport.height = reader.f64()?;
    viewport.near = reader.f64()?;
    viewport.far = reader.f64()?;
    Ok(())
}

fn read_scissor(reader: &mut ReaderV3<'_>, scissor: &mut ScissorRect) -> Result<(), WireError> {
    let x = reader.u32()?;
    let y = reader.u32()?;
    let width = reader.u32()?;
    let height = reader.u32()?;
    validate_extent("scissor width", width)?;
    validate_extent("scissor height", height)?;
    scissor.x = x;
    scissor.y = y;
    scissor.width = width;
    scissor.height = height;
    Ok(())
}

fn read_pipeline_ref(
    reader: &mut ReaderV3<'_>,
    pipeline: &mut PipelineRef,
) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    pipeline.pipeline_id = pipeline_id;
    Ok(())
}

fn read_topology(
    reader: &mut ReaderV3<'_>,
    topology: &mut PrimitiveTopology,
) -> Result<(), WireError> {
    let code = reader.u16()?;
    reader.reserved_u16()?;
    *topology = topology_from_code(code)?;
    Ok(())
}

fn read_vertex_buffers(
    reader: &mut ReaderV3<'_>,
    set: &mut VertexBufferSet,
) -> Result<(), WireError> {
    let start_slot = reader.u16()?;
    let count = usize::from(reader.u16()?);
    reader.reserved_u32()?;
    if count == 0 {
        return Err(WireError::ZeroVertexBuffers);
    }
    if count > MAX_VERTEX_BUFFERS {
        return Err(WireError::TooManyVertexBuffers(count));
    }
    set.buffers.clear();
    for _ in 0..count {
        set.buffers.push(read_vertex_buffer(reader)?);
    }
    set.start_slot = start_slot;
    Ok(())
}

fn read_vertex_buffer(reader: &mut ReaderV3<'_>) -> Result<VertexBuffer, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let offset = reader.u64()?;
    let stride = reader.u32()?;
    validate_extent("vertex stride", stride)?;
    let length = reader.u64()?;
    validate_length(length)?;
    if offset.checked_add(length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    Ok(VertexBuffer {
        allocation_id,
        generation,
        offset,
        stride,
        length,
    })
}

fn read_index_buffer(
    reader: &mut ReaderV3<'_>,
    binding: &mut IndexBufferBinding,
) -> Result<(), WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let offset = reader.u64()?;
    let length = reader.u64()?;
    validate_length(length)?;
    if offset.checked_add(length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    let format = index_format_from_code(reader.u16()?)?;
    reader.reserved_u16()?;
    binding.allocation_id = allocation_id;
    binding.generation = generation;
    binding.offset = offset;
    binding.length = length;
    binding.format = format;
    Ok(())
}

fn read_root_buffer_view(
    reader: &mut ReaderV3<'_>,
    view: &mut RootBufferView,
) -> Result<(), WireError> {
    let slot = reader.u16()?;
    reader.reserved_u16()?;
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    reader.reserved_u32()?;
    let offset = reader.u64()?;
    let length = reader.u64()?;
    validate_length(length)?;
    if offset.checked_add(length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    view.slot = slot;
    view.allocation_id = allocation_id;
    view.generation = generation;
    view.offset = offset;
    view.length = length;
    Ok(())
}

fn read_draw_instanced(
    reader: &mut ReaderV3<'_>,
    draw: &mut DrawInstanced,
) -> Result<(), WireError> {
    let vertex_count_per_instance = reader.u32()?;
    if vertex_count_per_instance == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    draw.vertex_count_per_instance = vertex_count_per_instance;
    draw.instance_count = instance_count;
    draw.start_vertex = reader.u32()?;
    draw.start_instance = reader.u32()?;
    Ok(())
}

fn read_draw_indexed_instanced(
    reader: &mut ReaderV3<'_>,
    draw: &mut DrawIndexedInstanced,
) -> Result<(), WireError> {
    let index_count_per_instance = reader.u32()?;
    if index_count_per_instance == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    draw.index_count_per_instance = index_count_per_instance;
    draw.instance_count = instance_count;
    draw.start_index = reader.u32()?;
    draw.base_vertex = reader.i32()?;
    draw.start_instance = reader.u32()?;
    Ok(())
}

fn read_copy_buffer(reader: &mut ReaderV3<'_>, copy: &mut CopyBuffer) -> Result<(), WireError> {
    let src_allocation_id = reader.u64()?;
    let src_generation = reader.u32()?;
    validate_handle(src_allocation_id, src_generation)?;
    reader.reserved_u32()?;
    let src_offset = reader.u64()?;
    let dst_allocation_id = reader.u64()?;
    let dst_generation = reader.u32()?;
    validate_handle(dst_allocation_id, dst_generation)?;
    reader.reserved_u32()?;
    let dst_offset = reader.u64()?;
    let length = reader.u64()?;
    validate_length(length)?;
    if src_offset.checked_add(length).is_none() || dst_offset.checked_add(length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    copy.src_allocation_id = src_allocation_id;
    copy.src_generation = src_generation;
    copy.src_offset = src_offset;
    copy.dst_allocation_id = dst_allocation_id;
    copy.dst_generation = dst_generation;
    copy.dst_offset = dst_offset;
    copy.length = length;
    Ok(())
}

fn read_copy_texture(reader: &mut ReaderV3<'_>, copy: &mut CopyTexture) -> Result<(), WireError> {
    let src_allocation_id = reader.u64()?;
    let src_generation = reader.u32()?;
    validate_handle(src_allocation_id, src_generation)?;
    let src_mip = reader.u16()?;
    let src_slice = reader.u16()?;
    let dst_allocation_id = reader.u64()?;
    let dst_generation = reader.u32()?;
    validate_handle(dst_allocation_id, dst_generation)?;
    let dst_mip = reader.u16()?;
    let dst_slice = reader.u16()?;
    let width = reader.u32()?;
    validate_extent("copy width", width)?;
    let height = reader.u32()?;
    validate_extent("copy height", height)?;
    let depth = reader.u32()?;
    validate_extent("copy depth", depth)?;
    reader.reserved_u32()?;
    copy.src_allocation_id = src_allocation_id;
    copy.src_generation = src_generation;
    copy.src_mip = src_mip;
    copy.src_slice = src_slice;
    copy.dst_allocation_id = dst_allocation_id;
    copy.dst_generation = dst_generation;
    copy.dst_mip = dst_mip;
    copy.dst_slice = dst_slice;
    copy.width = width;
    copy.height = height;
    copy.depth = depth;
    Ok(())
}

fn read_present(reader: &mut ReaderV3<'_>, present: &mut Present) -> Result<(), WireError> {
    let surface_id = reader.u64()?;
    let sync_interval = reader.u32()?;
    if sync_interval > 1 {
        return Err(WireError::UnsupportedSyncInterval(sync_interval));
    }
    reader.reserved_u32()?;
    present.surface_id = surface_id;
    present.sync_interval = sync_interval;
    Ok(())
}

fn read_execute_indirect_draw(
    reader: &mut ReaderV3<'_>,
    draw: &mut ExecuteIndirectDraw,
) -> Result<(), WireError> {
    read_indirect_arguments(reader, &mut draw.arguments)
}

fn read_execute_indirect_draw_indexed(
    reader: &mut ReaderV3<'_>,
    draw: &mut ExecuteIndirectDrawIndexed,
) -> Result<(), WireError> {
    read_indirect_arguments(reader, &mut draw.arguments)
}

fn read_execute_indirect_dispatch_mesh(
    reader: &mut ReaderV3<'_>,
    dispatch: &mut ExecuteIndirectDispatchMesh,
) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    read_indirect_arguments(reader, &mut dispatch.arguments)?;
    dispatch.pipeline_id = pipeline_id;
    Ok(())
}

fn read_execute_indirect_dispatch(
    reader: &mut ReaderV3<'_>,
    dispatch: &mut ExecuteIndirectDispatch,
) -> Result<(), WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    read_indirect_arguments(reader, &mut dispatch.arguments)?;
    let threads_per_group = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(threads_per_group)?;
    reader.reserved_u32()?;
    dispatch.pipeline_id = pipeline_id;
    dispatch.threads_per_group = threads_per_group;
    Ok(())
}

/// Reads one indirect argument range and its optional count buffer.
///
/// An absent count buffer is the all-zero triple. A partially populated triple
/// is rejected so a producer cannot encode a count buffer whose generation or
/// offset is silently dropped.
fn read_indirect_arguments(
    reader: &mut ReaderV3<'_>,
    arguments: &mut IndirectArguments,
) -> Result<(), WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    reader.reserved_u32()?;
    let offset = reader.u64()?;
    let max_count = reader.u32()?;
    reader.reserved_u32()?;
    let count_allocation_id = reader.u64()?;
    let count_generation = reader.u32()?;
    reader.reserved_u32()?;
    let count_offset = reader.u64()?;

    validate_handle(allocation_id, generation)?;
    validate_alignment(offset, "indirect argument offset alignment")?;
    validate_indirect_max_count(max_count)?;
    let count = match count_allocation_id {
        0 => {
            if count_generation != 0 || count_offset != 0 {
                return Err(WireError::ZeroAllocationId);
            }
            None
        }
        allocation_id => {
            validate_handle(allocation_id, count_generation)?;
            validate_alignment(count_offset, "indirect count offset alignment")?;
            Some(IndirectCountBuffer {
                allocation_id,
                generation: count_generation,
                offset: count_offset,
            })
        }
    };

    arguments.allocation_id = allocation_id;
    arguments.generation = generation;
    arguments.offset = offset;
    arguments.max_count = max_count;
    arguments.count = count;
    Ok(())
}

fn read_copy_texture_region(
    reader: &mut ReaderV3<'_>,
    copy: &mut CopyTextureRegion,
) -> Result<(), WireError> {
    let src = read_subresource(reader)?;
    let dst = read_subresource(reader)?;
    let src_origin = [reader.u32()?, reader.u32()?, reader.u32()?];
    let dst_origin = [reader.u32()?, reader.u32()?, reader.u32()?];
    let size = [reader.u32()?, reader.u32()?, reader.u32()?];
    reader.reserved_u32()?;
    validate_box("copy region", src_origin, size)?;
    validate_box("copy region", dst_origin, size)?;
    *copy = CopyTextureRegion {
        src,
        dst,
        src_origin,
        dst_origin,
        size,
    };
    Ok(())
}

fn read_resolve_subresource(
    reader: &mut ReaderV3<'_>,
    resolve: &mut ResolveSubresource,
) -> Result<(), WireError> {
    resolve.src = read_subresource(reader)?;
    resolve.dst = read_subresource(reader)?;
    Ok(())
}

fn read_subresource(reader: &mut ReaderV3<'_>) -> Result<TextureSubresource, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let mip_level = reader.u16()?;
    let slice = reader.u16()?;
    Ok(TextureSubresource {
        allocation_id,
        generation,
        mip_level,
        slice,
    })
}

/// Reads the fixed flags word that carries two independent booleans.
fn read_state_flag(code: u16, field: &'static str) -> Result<bool, WireError> {
    match code {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(WireError::InvalidEnumCode {
            field,
            code: other,
        }),
    }
}

fn read_blend_factor_code(reader: &mut ReaderV3<'_>) -> Result<BlendFactor, WireError> {
    BlendFactor::from_code(u16::from(reader.u8()?))
}

fn read_blend_operation_code(reader: &mut ReaderV3<'_>) -> Result<BlendOperation, WireError> {
    BlendOperation::from_code(u16::from(reader.u8()?))
}

fn read_stencil_operation_code(reader: &mut ReaderV3<'_>) -> Result<StencilOperation, WireError> {
    StencilOperation::from_code(u16::from(reader.u8()?))
}

fn read_compare_function_code(reader: &mut ReaderV3<'_>) -> Result<CompareFunction, WireError> {
    CompareFunction::from_code(u16::from(reader.u8()?))
}

fn read_stencil_face(reader: &mut ReaderV3<'_>) -> Result<StencilFaceState, WireError> {
    let fail = read_stencil_operation_code(reader)?;
    let depth_fail = read_stencil_operation_code(reader)?;
    let pass = read_stencil_operation_code(reader)?;
    let func = read_compare_function_code(reader)?;
    let read_mask = reader.u8()?;
    let write_mask = reader.u8()?;
    Ok(StencilFaceState {
        fail,
        depth_fail,
        pass,
        func,
        read_mask,
        write_mask,
    })
}

fn read_blend_attachment(reader: &mut ReaderV3<'_>) -> Result<BlendAttachment, WireError> {
    let enable = read_state_flag(u16::from(reader.u8()?), "blend enable")?;
    let src_color = read_blend_factor_code(reader)?;
    let dst_color = read_blend_factor_code(reader)?;
    let color_op = read_blend_operation_code(reader)?;
    let src_alpha = read_blend_factor_code(reader)?;
    let dst_alpha = read_blend_factor_code(reader)?;
    let alpha_op = read_blend_operation_code(reader)?;
    let write_mask = reader.u8()?;
    Ok(BlendAttachment {
        enable,
        src_color,
        dst_color,
        color_op,
        src_alpha,
        dst_alpha,
        alpha_op,
        write_mask,
    })
}

fn read_blend_state(reader: &mut ReaderV3<'_>, state: &mut BlendState) -> Result<(), WireError> {
    let count = usize::from(reader.u16()?);
    let independent_blend = read_state_flag(reader.u16()?, "independent blend")?;
    let alpha_to_coverage = read_state_flag(reader.u16()?, "alpha to coverage")?;
    reader.reserved_u16()?;
    if count == 0 {
        return Err(WireError::ZeroBlendAttachments);
    }
    if count > MAX_BLEND_ATTACHMENTS_V3 {
        return Err(WireError::TooManyBlendAttachments(count));
    }
    state.attachments.clear();
    for _ in 0..count {
        state.attachments.push(read_blend_attachment(reader)?);
    }
    state.independent_blend = independent_blend;
    state.alpha_to_coverage = alpha_to_coverage;
    Ok(())
}

fn read_depth_stencil_state(
    reader: &mut ReaderV3<'_>,
    state: &mut DepthStencilState,
) -> Result<(), WireError> {
    let depth_enable = read_state_flag(u16::from(reader.u8()?), "depth enable")?;
    let depth_write = read_state_flag(u16::from(reader.u8()?), "depth write")?;
    let depth_func = read_compare_function_code(reader)?;
    let stencil_enable = read_state_flag(u16::from(reader.u8()?), "stencil enable")?;
    let front = read_stencil_face(reader)?;
    let back = read_stencil_face(reader)?;
    reader.reserved_u32()?;
    state.depth_enable = depth_enable;
    state.depth_write = depth_write;
    state.depth_func = depth_func;
    state.stencil_enable = stencil_enable;
    state.front = front;
    state.back = back;
    Ok(())
}

fn read_rasterizer_state(
    reader: &mut ReaderV3<'_>,
    state: &mut RasterizerState,
) -> Result<(), WireError> {
    let fill = FillMode::from_code(u16::from(reader.u8()?))?;
    let cull = CullMode::from_code(u16::from(reader.u8()?))?;
    let front_face = FrontFaceWinding::from_code(u16::from(reader.u8()?))?;
    let depth_clip = read_state_flag(u16::from(reader.u8()?), "depth clip enable")?;
    let depth_bias = reader.i32()?;
    let slope_scaled_depth_bias = reader.f32()?;
    let depth_bias_clamp = reader.f32()?;
    reader.reserved_u32()?;
    state.fill = fill;
    state.cull = cull;
    state.front_face = front_face;
    state.depth_bias = depth_bias;
    state.slope_scaled_depth_bias = slope_scaled_depth_bias;
    state.depth_bias_clamp = depth_bias_clamp;
    state.depth_clip = depth_clip;
    Ok(())
}

fn read_clear_rect(
    reader: &mut ReaderV3<'_>,
    has_rect: bool,
) -> Result<Option<ScissorRect>, WireError> {
    let x = reader.u32()?;
    let y = reader.u32()?;
    let width = reader.u32()?;
    let height = reader.u32()?;
    if !has_rect {
        if x != 0 || y != 0 || width != 0 || height != 0 {
            return Err(WireError::NonZeroReserved);
        }
        return Ok(None);
    }
    validate_extent("clear rect width", width)?;
    validate_extent("clear rect height", height)?;
    Ok(Some(ScissorRect {
        x,
        y,
        width,
        height,
    }))
}

fn read_clear_render_target_view(
    reader: &mut ReaderV3<'_>,
    clear: &mut ClearRenderTargetView,
) -> Result<(), WireError> {
    let slot = reader.u16()?;
    let flags = reader.u16()?;
    if flags & !1 != 0 {
        return Err(WireError::UnsupportedFlags(flags));
    }
    if usize::from(slot) >= MAX_COLOR_ATTACHMENTS {
        return Err(WireError::TooManyColorAttachments(usize::from(slot) + 1));
    }
    let color = [reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?];
    let rect = read_clear_rect(reader, flags & 1 != 0)?;
    clear.slot = slot;
    clear.rect = rect;
    clear.color = color;
    Ok(())
}

fn read_clear_depth_stencil_view(
    reader: &mut ReaderV3<'_>,
    clear: &mut ClearDepthStencilView,
) -> Result<(), WireError> {
    let flags = reader.u16()?;
    if flags & !7 != 0 {
        return Err(WireError::UnsupportedFlags(flags));
    }
    reader.reserved_u16()?;
    let clear_depth = flags & 1 != 0;
    let clear_stencil = flags & 2 != 0;
    if !clear_depth && !clear_stencil {
        return Err(WireError::EmptyClear);
    }
    let depth_bits = reader.f32()?;
    let stencil = reader.u32()?;
    let rect = read_clear_rect(reader, flags & 4 != 0)?;
    clear.depth = clear_depth.then_some(depth_bits);
    clear.stencil = clear_stencil.then_some(stencil);
    if clear.stencil.is_none() && stencil != 0 {
        return Err(WireError::NonZeroReserved);
    }
    clear.rect = rect;
    Ok(())
}

fn read_copy_buffer_to_texture(
    reader: &mut ReaderV3<'_>,
    copy: &mut CopyBufferToTexture,
) -> Result<(), WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    validate_handle(allocation_id, generation)?;
    let destination_allocation_id = reader.u64()?;
    let destination_generation = reader.u32()?;
    validate_handle(destination_allocation_id, destination_generation)?;
    let offset = reader.u64()?;
    validate_alignment(offset, "copy source offset alignment")?;
    let bytes_per_row = reader.u32()?;
    let bytes_per_image = reader.u32()?;
    let extent = [reader.u32()?, reader.u32()?, reader.u32()?];
    let mip_level = reader.u16()?;
    let slice = reader.u16()?;
    let origin = [reader.u32()?, reader.u32()?, reader.u32()?];
    reader.reserved_u32()?;
    validate_pitched_copy(
        offset,
        bytes_per_row,
        bytes_per_image,
        extent,
        origin,
    )?;
    copy.allocation_id = allocation_id;
    copy.generation = generation;
    copy.destination_allocation_id = destination_allocation_id;
    copy.destination_generation = destination_generation;
    copy.offset = offset;
    copy.bytes_per_row = bytes_per_row;
    copy.bytes_per_image = bytes_per_image;
    copy.extent = extent;
    copy.mip_level = mip_level;
    copy.slice = slice;
    copy.origin = origin;
    Ok(())
}

/// Validates the pitched-copy fields the wire can check without the texture.
fn validate_pitched_copy(
    offset: u64,
    bytes_per_row: u32,
    bytes_per_image: u32,
    extent: [u32; 3],
    origin: [u32; 3],
) -> Result<(), WireError> {
    if bytes_per_row == 0 {
        return Err(WireError::ZeroExtent {
            field: "copy source row pitch",
        });
    }
    validate_box("copy buffer to texture", origin, extent)?;
    let row_bytes = u64::from(bytes_per_row)
        .checked_mul(u64::from(extent[1]))
        .ok_or(WireError::OffsetOverflow)?;
    let image_bytes = u64::from(bytes_per_image);
    if extent[2] > 1 {
        if image_bytes == 0 {
            return Err(WireError::ZeroExtent {
                field: "copy source image pitch",
            });
        }
        if image_bytes < row_bytes {
            return Err(WireError::OffsetOverflow);
        }
    } else if image_bytes != 0 && image_bytes < row_bytes {
        return Err(WireError::OffsetOverflow);
    }
    let total = if image_bytes == 0 {
        row_bytes
    } else {
        image_bytes
            .checked_mul(u64::from(extent[2]))
            .ok_or(WireError::OffsetOverflow)?
    };
    offset.checked_add(total).ok_or(WireError::OffsetOverflow)?;
    Ok(())
}


fn read_bindings_into(
    reader: &mut ReaderV3<'_>,
    bindings: &mut Vec<BufferBinding>,
) -> Result<(), WireError> {
    let count = usize::from(reader.u16()?);
    reader.reserved_u16()?;
    if count > MAX_BINDINGS_PER_DISPATCH {
        return Err(WireError::TooManyBindings(count));
    }
    bindings.clear();
    for _ in 0..count {
        bindings.push(read_binding(reader)?);
    }
    Ok(())
}

fn read_binding(reader: &mut ReaderV3<'_>) -> Result<BufferBinding, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    reader.reserved_u32()?;
    let offset = reader.u64()?;
    let length = reader.u64()?;
    validate_range(allocation_id, generation, offset, length)?;
    Ok(BufferBinding {
        allocation_id,
        generation,
        offset,
        length,
    })
}

fn command_encoded_len(command: &CommandV3) -> Result<usize, WireError> {
    match command {
        CommandV3::Dispatch1d(dispatch) => dispatch_1d_encoded_len(dispatch),
        CommandV3::Dispatch3d(dispatch) => dispatch_3d_encoded_len(dispatch),
        CommandV3::DrawIndexed(draw) => draw_indexed_encoded_len(draw),
        CommandV3::Draw(draw) => draw_encoded_len(draw),
        CommandV3::DispatchMesh(dispatch) => dispatch_mesh_encoded_len(dispatch),
        CommandV3::ResourceBarrier(barrier) => barrier_encoded_len(barrier),
        CommandV3::SetPipeline(set_pipeline) => set_pipeline_encoded_len(set_pipeline),
        CommandV3::SetRootConstants(constants) => root_constants_encoded_len(constants),
        CommandV3::SetDescriptorTable(table) => descriptor_table_encoded_len(table),
        CommandV3::BeginRenderPass(pass) => begin_render_pass_encoded_len(pass),
        CommandV3::EndRenderPass => Ok(END_RENDER_PASS_LEN),
        CommandV3::SetViewport(_) => Ok(SET_VIEWPORT_LEN),
        CommandV3::SetScissor(scissor) => scissor_encoded_len(scissor),
        CommandV3::SetGraphicsPipeline(pipeline)
        | CommandV3::SetMeshPipeline(pipeline)
        | CommandV3::SetComputePipeline(pipeline) => pipeline_ref_encoded_len(pipeline),
        CommandV3::SetPrimitiveTopology(_) => Ok(SET_TOPOLOGY_LEN),
        CommandV3::SetVertexBuffers(set) => vertex_buffers_encoded_len(set),
        CommandV3::SetIndexBuffer(binding) => index_buffer_encoded_len(binding),
        CommandV3::SetRootBufferView(view) => root_buffer_view_encoded_len(view),
        CommandV3::DrawInstanced(draw) => draw_instanced_encoded_len(draw),
        CommandV3::DrawIndexedInstanced(draw) => draw_indexed_instanced_encoded_len(draw),
        CommandV3::CopyBuffer(copy) => copy_buffer_encoded_len(copy),
        CommandV3::CopyTexture(copy) => copy_texture_encoded_len(copy),
        CommandV3::ExecuteIndirectDraw(draw) => {
            execute_indirect_draw_encoded_len(&draw.arguments)
        }
        CommandV3::ExecuteIndirectDrawIndexed(draw) => {
            execute_indirect_draw_indexed_encoded_len(&draw.arguments)
        }
        CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
            execute_indirect_dispatch_mesh_encoded_len(dispatch)
        }
        CommandV3::ExecuteIndirectDispatch(dispatch) => {
            execute_indirect_dispatch_encoded_len(dispatch)
        }
        CommandV3::CopyTextureRegion(copy) => copy_texture_region_encoded_len(copy),
        CommandV3::ResolveSubresource(resolve) => resolve_subresource_encoded_len(resolve),
        CommandV3::SetBlendState(state) => blend_state_encoded_len(state),
        CommandV3::SetDepthStencilState(_) => Ok(SET_DEPTH_STENCIL_STATE_LEN),
        CommandV3::SetRasterizerState(_) => Ok(SET_RASTERIZER_STATE_LEN),
        CommandV3::SetStencilRef(_) => Ok(SET_STENCIL_REF_LEN),
        CommandV3::SetBlendFactor(_) => Ok(SET_BLEND_FACTOR_LEN),
        CommandV3::ClearRenderTargetView(clear) => clear_render_target_view_encoded_len(clear),
        CommandV3::ClearDepthStencilView(clear) => clear_depth_stencil_view_encoded_len(clear),
        CommandV3::CopyBufferToTexture(copy) => copy_buffer_to_texture_encoded_len(copy),
        CommandV3::Present(present) => present_encoded_len(present),
    }
}

fn dispatch_1d_encoded_len(dispatch: &Dispatch1d) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_threads(dispatch.threads, dispatch.threads_per_group)?;
    sized_with_bindings(DISPATCH_1D_LEN, &dispatch.bindings)
}

fn dispatch_3d_encoded_len(dispatch: &Dispatch3d) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_dimensions(dispatch.group_counts)?;
    validate_dimensions(dispatch.threads_per_group)?;
    let group_count = checked_dimension_product(dispatch.group_counts);
    let local_count = checked_dimension_product(dispatch.threads_per_group);
    if group_count
        .and_then(|groups| groups.checked_mul(local_count?))
        .is_none()
    {
        return Err(WireError::ThreadCountOverflow);
    }
    sized_with_bindings(DISPATCH_3D_LEN, &dispatch.bindings)
}

fn draw_indexed_encoded_len(draw: &DrawIndexedState) -> Result<usize, WireError> {
    if draw.index_count == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    Ok(DRAW_INDEXED_LEN)
}

fn draw_encoded_len(draw: &DrawState) -> Result<usize, WireError> {
    if draw.vertex_count == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    Ok(DRAW_LEN)
}

fn dispatch_mesh_encoded_len(dispatch: &DispatchMesh) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_dimensions(dispatch.group_counts)?;
    if checked_dimension_product(dispatch.group_counts).is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    sized_with_bindings(DISPATCH_MESH_LEN, &dispatch.bindings)
}

fn barrier_encoded_len(barrier: &ResourceBarrier) -> Result<usize, WireError> {
    if barrier.entries.is_empty() {
        return Err(WireError::ZeroBarrierCount);
    }
    if barrier.entries.len() > MAX_BARRIERS_V3 {
        return Err(WireError::TooManyBarriersV3(barrier.entries.len()));
    }
    for entry in &barrier.entries {
        validate_range(
            entry.allocation_id,
            entry.generation,
            entry.offset,
            entry.length,
        )?;
    }
    barrier
        .entries
        .len()
        .checked_mul(BARRIER_ENTRY_LEN)
        .and_then(|entries_len| entries_len.checked_add(BARRIER_HEADER_LEN))
        .ok_or(WireError::FrameTooLarge)
}

fn set_pipeline_encoded_len(set_pipeline: &SetPipeline) -> Result<usize, WireError> {
    validate_pipeline(set_pipeline.pipeline_id)?;
    Ok(SET_PIPELINE_LEN)
}

fn root_constants_encoded_len(constants: &RootConstantsV3) -> Result<usize, WireError> {
    if constants.bytes.is_empty() {
        return Err(WireError::ZeroRootConstants);
    }
    if constants.bytes.len() > MAX_ROOT_CONSTANT_BYTES_V3 {
        return Err(WireError::RootConstantsTooLarge(constants.bytes.len()));
    }
    constants
        .bytes
        .len()
        .checked_add(ROOT_CONSTANTS_HEADER_LEN)
        .ok_or(WireError::FrameTooLarge)
}

/// Wire names of each slot's per-entry kind tag.
///
/// [`WireError::InvalidEnumCode`] carries a `&'static str`, so the slot index
/// is part of the static name; a refusal still names both the offending slot
/// and the undefined tag.
const DESCRIPTOR_ENTRY_TAG_FIELDS: [&str; MAX_DESCRIPTOR_BINDINGS_V3] = [
    "descriptor table entry 0 tag",
    "descriptor table entry 1 tag",
    "descriptor table entry 2 tag",
    "descriptor table entry 3 tag",
    "descriptor table entry 4 tag",
    "descriptor table entry 5 tag",
    "descriptor table entry 6 tag",
    "descriptor table entry 7 tag",
    "descriptor table entry 8 tag",
    "descriptor table entry 9 tag",
    "descriptor table entry 10 tag",
    "descriptor table entry 11 tag",
    "descriptor table entry 12 tag",
    "descriptor table entry 13 tag",
    "descriptor table entry 14 tag",
    "descriptor table entry 15 tag",
    "descriptor table entry 16 tag",
    "descriptor table entry 17 tag",
    "descriptor table entry 18 tag",
    "descriptor table entry 19 tag",
    "descriptor table entry 20 tag",
    "descriptor table entry 21 tag",
    "descriptor table entry 22 tag",
    "descriptor table entry 23 tag",
    "descriptor table entry 24 tag",
    "descriptor table entry 25 tag",
    "descriptor table entry 26 tag",
    "descriptor table entry 27 tag",
    "descriptor table entry 28 tag",
    "descriptor table entry 29 tag",
    "descriptor table entry 30 tag",
    "descriptor table entry 31 tag",
];

fn descriptor_entry_tag_field(index: usize) -> &'static str {
    DESCRIPTOR_ENTRY_TAG_FIELDS
        .get(index)
        .copied()
        .unwrap_or("descriptor table entry tag")
}

fn descriptor_entry_payload_len(entry: &DescriptorTableEntryV3) -> usize {
    match entry {
        DescriptorTableEntryV3::Buffer(_) => BINDING_LEN,
        DescriptorTableEntryV3::Texture(_) | DescriptorTableEntryV3::Sampler(_) => {
            RESOURCE_HANDLE_LEN
        }
    }
}

fn descriptor_table_encoded_len(table: &DescriptorTableV3) -> Result<usize, WireError> {
    if table.bindings.is_empty() {
        return Err(WireError::ZeroDescriptorCount);
    }
    if table.bindings.len() > MAX_DESCRIPTOR_BINDINGS_V3 {
        return Err(WireError::TooManyDescriptorBindingsV3(
            table.bindings.len(),
        ));
    }
    let tag = descriptor_table_entry_tag(table)?;
    let bindings_len = match tag {
        DESCRIPTOR_ENTRY_TAGGED => {
            let mut total = 0usize;
            for entry in &table.bindings {
                let entry_len = DESCRIPTOR_ENTRY_TAG_LEN
                    .checked_add(descriptor_entry_payload_len(entry))
                    .ok_or(WireError::FrameTooLarge)?;
                total = total
                    .checked_add(entry_len)
                    .ok_or(WireError::FrameTooLarge)?;
            }
            total
        }
        DESCRIPTOR_ENTRY_BUFFER => table
            .bindings
            .len()
            .checked_mul(BINDING_LEN)
            .ok_or(WireError::FrameTooLarge)?,
        _ => table
            .bindings
            .len()
            .checked_mul(RESOURCE_HANDLE_LEN)
            .ok_or(WireError::FrameTooLarge)?,
    };
    bindings_len
        .checked_add(DESCRIPTOR_TABLE_HEADER_LEN)
        .ok_or(WireError::FrameTooLarge)
}

/// The legacy single-kind tag one entry would use on its own.
///
/// Buffer entries use tag 0; texture and sampler handles both use tag 1 with
/// the table kind discriminating between them.
fn descriptor_entry_tag(
    entry: &DescriptorTableEntryV3,
    kind: DescriptorKind,
) -> Result<u16, WireError> {
    match entry {
        DescriptorTableEntryV3::Buffer(binding) => {
            validate_binding(binding)?;
            if matches!(kind, DescriptorKind::Sampler) {
                return Err(WireError::DescriptorEntryMismatch);
            }
            Ok(DESCRIPTOR_ENTRY_BUFFER)
        }
        DescriptorTableEntryV3::Texture(handle) => {
            if *handle == 0 {
                return Err(WireError::ZeroDescriptorHandle);
            }
            if !matches!(kind, DescriptorKind::Srv | DescriptorKind::Uav) {
                return Err(WireError::DescriptorEntryMismatch);
            }
            Ok(DESCRIPTOR_ENTRY_RESOURCE)
        }
        DescriptorTableEntryV3::Sampler(handle) => {
            if *handle == 0 {
                return Err(WireError::ZeroDescriptorHandle);
            }
            if kind != DescriptorKind::Sampler {
                return Err(WireError::DescriptorEntryMismatch);
            }
            Ok(DESCRIPTOR_ENTRY_RESOURCE)
        }
    }
}

fn descriptor_entry_kind(entry: &DescriptorTableEntryV3) -> u8 {
    match entry {
        DescriptorTableEntryV3::Buffer(_) => DESCRIPTOR_ENTRY_KIND_BUFFER,
        DescriptorTableEntryV3::Texture(_) => DESCRIPTOR_ENTRY_KIND_TEXTURE,
        DescriptorTableEntryV3::Sampler(_) => DESCRIPTOR_ENTRY_KIND_SAMPLER,
    }
}

/// Validates every entry and returns the table's wire encoding tag.
///
/// A table whose entries all share a legacy kind keeps tag 0 or 1 and its old
/// byte layout. A table that mixes buffer and registered-resource entries uses
/// [`DESCRIPTOR_ENTRY_TAGGED`] and carries one kind byte per entry, so no entry
/// is dropped or guessed.
fn descriptor_table_entry_tag(table: &DescriptorTableV3) -> Result<u16, WireError> {
    let mut tag = None;
    let mut mixed = false;
    for entry in &table.bindings {
        let entry_tag = descriptor_entry_tag(entry, table.kind)?;
        match tag {
            Some(existing) if existing != entry_tag => mixed = true,
            None => tag = Some(entry_tag),
            _ => {}
        }
    }
    let tag = tag.ok_or(WireError::ZeroDescriptorCount)?;
    Ok(if mixed {
        DESCRIPTOR_ENTRY_TAGGED
    } else {
        tag
    })
}

fn begin_render_pass_encoded_len(pass: &BeginRenderPass) -> Result<usize, WireError> {
    if pass.colors.is_empty() {
        return Err(WireError::ZeroColorAttachments);
    }
    if pass.colors.len() > MAX_COLOR_ATTACHMENTS {
        return Err(WireError::TooManyColorAttachments(pass.colors.len()));
    }
    for color in &pass.colors {
        validate_handle(color.allocation_id, color.generation)?;
    }
    if let Some(depth) = &pass.depth_stencil {
        validate_handle(depth.allocation_id, depth.generation)?;
    }
    let colors_len = pass
        .colors
        .len()
        .checked_mul(COLOR_ATTACHMENT_LEN)
        .ok_or(WireError::FrameTooLarge)?;
    let depth_len = if pass.depth_stencil.is_some() {
        DEPTH_ATTACHMENT_LEN
    } else {
        0
    };
    BEGIN_RENDER_PASS_HEADER_LEN
        .checked_add(colors_len)
        .and_then(|len| len.checked_add(depth_len))
        .ok_or(WireError::FrameTooLarge)
}

fn scissor_encoded_len(scissor: &ScissorRect) -> Result<usize, WireError> {
    validate_extent("scissor width", scissor.width)?;
    validate_extent("scissor height", scissor.height)?;
    Ok(SET_SCISSOR_LEN)
}

fn pipeline_ref_encoded_len(pipeline: &PipelineRef) -> Result<usize, WireError> {
    validate_pipeline(pipeline.pipeline_id)?;
    Ok(PIPELINE_REF_LEN)
}

fn vertex_buffers_encoded_len(set: &VertexBufferSet) -> Result<usize, WireError> {
    if set.buffers.is_empty() {
        return Err(WireError::ZeroVertexBuffers);
    }
    if set.buffers.len() > MAX_VERTEX_BUFFERS {
        return Err(WireError::TooManyVertexBuffers(set.buffers.len()));
    }
    for buffer in &set.buffers {
        validate_handle(buffer.allocation_id, buffer.generation)?;
        validate_extent("vertex stride", buffer.stride)?;
        validate_length(buffer.length)?;
        if buffer.offset.checked_add(buffer.length).is_none() {
            return Err(WireError::OffsetOverflow);
        }
    }
    set.buffers
        .len()
        .checked_mul(VERTEX_BUFFER_LEN)
        .and_then(|buffers_len| buffers_len.checked_add(VERTEX_BUFFERS_HEADER_LEN))
        .ok_or(WireError::FrameTooLarge)
}

fn index_buffer_encoded_len(binding: &IndexBufferBinding) -> Result<usize, WireError> {
    validate_handle(binding.allocation_id, binding.generation)?;
    validate_length(binding.length)?;
    if binding.offset.checked_add(binding.length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    Ok(SET_INDEX_BUFFER_LEN)
}

fn root_buffer_view_encoded_len(view: &RootBufferView) -> Result<usize, WireError> {
    validate_handle(view.allocation_id, view.generation)?;
    validate_length(view.length)?;
    if view.offset.checked_add(view.length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    Ok(SET_ROOT_BUFFER_VIEW_LEN)
}

fn draw_instanced_encoded_len(draw: &DrawInstanced) -> Result<usize, WireError> {
    if draw.vertex_count_per_instance == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    Ok(DRAW_INSTANCED_LEN)
}

fn draw_indexed_instanced_encoded_len(draw: &DrawIndexedInstanced) -> Result<usize, WireError> {
    if draw.index_count_per_instance == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    Ok(DRAW_INDEXED_INSTANCED_LEN)
}

fn copy_buffer_encoded_len(copy: &CopyBuffer) -> Result<usize, WireError> {
    validate_handle(copy.src_allocation_id, copy.src_generation)?;
    validate_handle(copy.dst_allocation_id, copy.dst_generation)?;
    validate_length(copy.length)?;
    if copy.src_offset.checked_add(copy.length).is_none()
        || copy.dst_offset.checked_add(copy.length).is_none()
    {
        return Err(WireError::OffsetOverflow);
    }
    Ok(COPY_BUFFER_LEN)
}

fn copy_texture_encoded_len(copy: &CopyTexture) -> Result<usize, WireError> {
    validate_handle(copy.src_allocation_id, copy.src_generation)?;
    validate_handle(copy.dst_allocation_id, copy.dst_generation)?;
    validate_extent("copy width", copy.width)?;
    validate_extent("copy height", copy.height)?;
    validate_extent("copy depth", copy.depth)?;
    Ok(COPY_TEXTURE_LEN)
}

fn present_encoded_len(present: &Present) -> Result<usize, WireError> {
    if present.sync_interval > 1 {
        return Err(WireError::UnsupportedSyncInterval(present.sync_interval));
    }
    Ok(PRESENT_LEN)
}

fn validate_indirect_arguments(arguments: &IndirectArguments) -> Result<(), WireError> {
    validate_handle(arguments.allocation_id, arguments.generation)?;
    validate_alignment(
        arguments.offset,
        "indirect argument offset alignment",
    )?;
    validate_indirect_max_count(arguments.max_count)?;
    if let Some(count) = &arguments.count {
        validate_handle(count.allocation_id, count.generation)?;
        validate_alignment(count.offset, "indirect count offset alignment")?;
    }
    Ok(())
}

fn execute_indirect_draw_encoded_len(arguments: &IndirectArguments) -> Result<usize, WireError> {
    validate_indirect_arguments(arguments)?;
    Ok(EXECUTE_INDIRECT_DRAW_LEN)
}

fn execute_indirect_draw_indexed_encoded_len(
    arguments: &IndirectArguments,
) -> Result<usize, WireError> {
    validate_indirect_arguments(arguments)?;
    Ok(EXECUTE_INDIRECT_DRAW_LEN)
}

fn execute_indirect_dispatch_mesh_encoded_len(
    dispatch: &ExecuteIndirectDispatchMesh,
) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_indirect_arguments(&dispatch.arguments)?;
    Ok(EXECUTE_INDIRECT_DISPATCH_MESH_LEN)
}

fn execute_indirect_dispatch_encoded_len(
    dispatch: &ExecuteIndirectDispatch,
) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_indirect_arguments(&dispatch.arguments)?;
    validate_dimensions(dispatch.threads_per_group)?;
    Ok(EXECUTE_INDIRECT_DISPATCH_LEN)
}

fn copy_texture_region_encoded_len(copy: &CopyTextureRegion) -> Result<usize, WireError> {
    validate_subresource(&copy.src)?;
    validate_subresource(&copy.dst)?;
    validate_box("copy region", copy.src_origin, copy.size)?;
    validate_box("copy region", copy.dst_origin, copy.size)?;
    Ok(COPY_TEXTURE_REGION_LEN)
}

fn resolve_subresource_encoded_len(resolve: &ResolveSubresource) -> Result<usize, WireError> {
    validate_subresource(&resolve.src)?;
    validate_subresource(&resolve.dst)?;
    Ok(RESOLVE_SUBRESOURCE_LEN)
}

fn blend_state_encoded_len(state: &BlendState) -> Result<usize, WireError> {
    if state.attachments.is_empty() {
        return Err(WireError::ZeroBlendAttachments);
    }
    if state.attachments.len() > MAX_BLEND_ATTACHMENTS_V3 {
        return Err(WireError::TooManyBlendAttachments(state.attachments.len()));
    }
    state
        .attachments
        .len()
        .checked_mul(BLEND_ATTACHMENT_LEN)
        .and_then(|entries| entries.checked_add(SET_BLEND_STATE_HEADER_LEN))
        .ok_or(WireError::FrameTooLarge)
}

fn clear_render_target_view_encoded_len(
    clear: &ClearRenderTargetView,
) -> Result<usize, WireError> {
    if usize::from(clear.slot) >= MAX_COLOR_ATTACHMENTS {
        return Err(WireError::TooManyColorAttachments(usize::from(clear.slot) + 1));
    }
    if let Some(rect) = clear.rect {
        validate_extent("clear rect width", rect.width)?;
        validate_extent("clear rect height", rect.height)?;
    }
    Ok(CLEAR_RENDER_TARGET_VIEW_LEN)
}

fn clear_depth_stencil_view_encoded_len(
    clear: &ClearDepthStencilView,
) -> Result<usize, WireError> {
    if clear.depth.is_none() && clear.stencil.is_none() {
        return Err(WireError::EmptyClear);
    }
    if let Some(rect) = clear.rect {
        validate_extent("clear rect width", rect.width)?;
        validate_extent("clear rect height", rect.height)?;
    }
    Ok(CLEAR_DEPTH_STENCIL_VIEW_LEN)
}

fn copy_buffer_to_texture_encoded_len(copy: &CopyBufferToTexture) -> Result<usize, WireError> {
    validate_handle(copy.allocation_id, copy.generation)?;
    validate_handle(
        copy.destination_allocation_id,
        copy.destination_generation,
    )?;
    validate_alignment(copy.offset, "copy source offset alignment")?;
    validate_pitched_copy(
        copy.offset,
        copy.bytes_per_row,
        copy.bytes_per_image,
        copy.extent,
        copy.origin,
    )?;
    Ok(COPY_BUFFER_TO_TEXTURE_LEN)
}


fn validate_subresource(subresource: &TextureSubresource) -> Result<(), WireError> {
    validate_handle(subresource.allocation_id, subresource.generation)
}

/// Rejects an empty box and an origin/size pair that overflows `u32`.
///
/// The live texture extent is not on the wire; the host re-checks the box
/// against the resolved mip dimensions before it creates an encoder.
fn validate_box(_field: &'static str, origin: [u32; 3], size: [u32; 3]) -> Result<(), WireError> {
    if size[0] == 0 {
        return Err(WireError::ZeroExtent {
            field: "copy region width",
        });
    }
    if size[1] == 0 {
        return Err(WireError::ZeroExtent {
            field: "copy region height",
        });
    }
    if size[2] == 0 {
        return Err(WireError::ZeroExtent {
            field: "copy region depth",
        });
    }
    for (start, extent) in origin.into_iter().zip(size) {
        if start.checked_add(extent).is_none() {
            return Err(WireError::OffsetOverflow);
        }
    }
    Ok(())
}

fn validate_alignment(offset: u64, field: &'static str) -> Result<(), WireError> {
    if offset.is_multiple_of(4) {
        Ok(())
    } else {
        // `WireError` is frozen with wire.rs outside this task's scope and has
        // no alignment variant; the field names the rule and the code carries
        // the misaligned low bits.
        Err(WireError::InvalidEnumCode {
            field,
            code: (offset & 3) as u16,
        })
    }
}

fn validate_indirect_max_count(max_count: u32) -> Result<(), WireError> {
    if max_count == 0 {
        return Err(WireError::ZeroExtent {
            field: "indirect max command count",
        });
    }
    if max_count > MAX_INDIRECT_COMMANDS_V3 {
        return Err(WireError::TooManyCommandsV3(max_count as usize));
    }
    Ok(())
}

fn sized_with_bindings(header_len: usize, bindings: &[BufferBinding]) -> Result<usize, WireError> {
    if bindings.len() > MAX_BINDINGS_PER_DISPATCH {
        return Err(WireError::TooManyBindings(bindings.len()));
    }
    for binding in bindings {
        validate_binding(binding)?;
    }
    bindings
        .len()
        .checked_mul(BINDING_LEN)
        .and_then(|bindings_len| bindings_len.checked_add(header_len))
        .ok_or(WireError::FrameTooLarge)
}

fn write_command(out: &mut Vec<u8>, command: &CommandV3) {
    write_u16(out, command.opcode());
    write_u16(out, 0);
    match command {
        CommandV3::Dispatch1d(dispatch) => {
            write_u64(out, dispatch.pipeline_id);
            write_u32(out, dispatch.threads);
            write_u32(out, dispatch.threads_per_group);
            write_u16(out, dispatch.bindings.len() as u16);
            write_u16(out, 0);
            write_buffer_bindings(out, &dispatch.bindings);
        }
        CommandV3::Dispatch3d(dispatch) => {
            write_u64(out, dispatch.pipeline_id);
            for dimension in dispatch.group_counts {
                write_u32(out, dimension);
            }
            for dimension in dispatch.threads_per_group {
                write_u32(out, dimension);
            }
            write_u16(out, dispatch.bindings.len() as u16);
            write_u16(out, 0);
            write_buffer_bindings(out, &dispatch.bindings);
        }
        CommandV3::DrawIndexed(draw) => {
            write_u32(out, draw.index_count);
            write_u32(out, draw.instance_count);
            write_u32(out, draw.start_index);
            write_i32(out, draw.base_vertex);
            write_u32(out, draw.start_instance);
        }
        CommandV3::Draw(draw) => {
            write_u32(out, draw.vertex_count);
            write_u32(out, draw.instance_count);
            write_u32(out, draw.start_vertex);
            write_u32(out, draw.start_instance);
        }
        CommandV3::DispatchMesh(dispatch) => {
            write_u64(out, dispatch.pipeline_id);
            for dimension in dispatch.group_counts {
                write_u32(out, dimension);
            }
            write_u16(out, dispatch.bindings.len() as u16);
            write_u16(out, 0);
            write_buffer_bindings(out, &dispatch.bindings);
        }
        CommandV3::ResourceBarrier(barrier) => {
            write_u16(out, barrier.entries.len() as u16);
            write_u16(out, 0);
            for entry in &barrier.entries {
                write_u64(out, entry.allocation_id);
                write_u32(out, entry.generation);
                write_u32(out, 0);
                write_u64(out, entry.offset);
                write_u64(out, entry.length);
                write_u32(out, entry.before_state);
                write_u32(out, entry.after_state);
            }
        }
        CommandV3::SetPipeline(set_pipeline) => {
            write_u64(out, set_pipeline.pipeline_id);
        }
        CommandV3::SetRootConstants(constants) => {
            write_u16(out, constants.slot);
            write_u16(out, constants.bytes.len() as u16);
            out.extend_from_slice(&constants.bytes);
        }
        CommandV3::SetDescriptorTable(table) => {
            write_u16(out, table.set);
            write_u16(out, table.base_register);
            write_u16(out, descriptor_kind_code(table.kind));
            write_u16(out, table.bindings.len() as u16);
            let tag = descriptor_table_entry_tag(table).expect("validated table entry tag");
            write_u16(out, tag);
            match tag {
                DESCRIPTOR_ENTRY_BUFFER => write_table_buffer_bindings(out, &table.bindings),
                DESCRIPTOR_ENTRY_RESOURCE => {
                    for entry in &table.bindings {
                        write_u64(
                            out,
                            entry
                                .resource_handle()
                                .expect("resource entry tag matches the entries"),
                        );
                    }
                }
                DESCRIPTOR_ENTRY_TAGGED => write_table_tagged_entries(out, &table.bindings),
                other => unreachable!("unexpected descriptor table tag {other}"),
            }
        }
        CommandV3::BeginRenderPass(pass) => {
            write_u16(out, pass.colors.len() as u16);
            write_u16(out, u16::from(pass.depth_stencil.is_some()));
            write_u32(out, 0);
            for color in &pass.colors {
                write_u64(out, color.allocation_id);
                write_u32(out, color.generation);
                write_u16(out, color.mip_level);
                write_u16(out, color.slice);
                write_u16(out, load_op_code(color.load));
                write_u16(out, store_op_code(color.store));
                write_u32(out, 0);
                for component in color.clear_color {
                    write_f64(out, component);
                }
            }
            if let Some(depth) = &pass.depth_stencil {
                write_u64(out, depth.allocation_id);
                write_u32(out, depth.generation);
                write_u16(out, depth.mip_level);
                write_u16(out, depth.slice);
                write_u16(out, load_op_code(depth.depth_load));
                write_u16(out, store_op_code(depth.depth_store));
                write_f32(out, depth.clear_depth);
                write_u16(out, load_op_code(depth.stencil_load));
                write_u16(out, store_op_code(depth.stencil_store));
                write_u32(out, depth.clear_stencil);
                write_u32(out, 0);
            }
        }
        CommandV3::EndRenderPass => {}
        CommandV3::SetViewport(viewport) => {
            write_f64(out, viewport.x);
            write_f64(out, viewport.y);
            write_f64(out, viewport.width);
            write_f64(out, viewport.height);
            write_f64(out, viewport.near);
            write_f64(out, viewport.far);
        }
        CommandV3::SetScissor(scissor) => {
            write_u32(out, scissor.x);
            write_u32(out, scissor.y);
            write_u32(out, scissor.width);
            write_u32(out, scissor.height);
        }
        CommandV3::SetGraphicsPipeline(pipeline)
        | CommandV3::SetMeshPipeline(pipeline)
        | CommandV3::SetComputePipeline(pipeline) => {
            write_u64(out, pipeline.pipeline_id);
        }
        CommandV3::SetPrimitiveTopology(topology) => {
            write_u16(out, topology_code(*topology));
            write_u16(out, 0);
        }
        CommandV3::SetVertexBuffers(set) => {
            write_u16(out, set.start_slot);
            write_u16(out, set.buffers.len() as u16);
            write_u32(out, 0);
            for buffer in &set.buffers {
                write_u64(out, buffer.allocation_id);
                write_u32(out, buffer.generation);
                write_u64(out, buffer.offset);
                write_u32(out, buffer.stride);
                write_u64(out, buffer.length);
            }
        }
        CommandV3::SetIndexBuffer(binding) => {
            write_u64(out, binding.allocation_id);
            write_u32(out, binding.generation);
            write_u64(out, binding.offset);
            write_u64(out, binding.length);
            write_u16(out, index_format_code(binding.format));
            write_u16(out, 0);
        }
        CommandV3::SetRootBufferView(view) => {
            write_u16(out, view.slot);
            write_u16(out, 0);
            write_u64(out, view.allocation_id);
            write_u32(out, view.generation);
            write_u32(out, 0);
            write_u64(out, view.offset);
            write_u64(out, view.length);
        }
        CommandV3::DrawInstanced(draw) => {
            write_u32(out, draw.vertex_count_per_instance);
            write_u32(out, draw.instance_count);
            write_u32(out, draw.start_vertex);
            write_u32(out, draw.start_instance);
        }
        CommandV3::DrawIndexedInstanced(draw) => {
            write_u32(out, draw.index_count_per_instance);
            write_u32(out, draw.instance_count);
            write_u32(out, draw.start_index);
            write_i32(out, draw.base_vertex);
            write_u32(out, draw.start_instance);
        }
        CommandV3::CopyBuffer(copy) => {
            write_u64(out, copy.src_allocation_id);
            write_u32(out, copy.src_generation);
            write_u32(out, 0);
            write_u64(out, copy.src_offset);
            write_u64(out, copy.dst_allocation_id);
            write_u32(out, copy.dst_generation);
            write_u32(out, 0);
            write_u64(out, copy.dst_offset);
            write_u64(out, copy.length);
        }
        CommandV3::CopyTexture(copy) => {
            write_u64(out, copy.src_allocation_id);
            write_u32(out, copy.src_generation);
            write_u16(out, copy.src_mip);
            write_u16(out, copy.src_slice);
            write_u64(out, copy.dst_allocation_id);
            write_u32(out, copy.dst_generation);
            write_u16(out, copy.dst_mip);
            write_u16(out, copy.dst_slice);
            write_u32(out, copy.width);
            write_u32(out, copy.height);
            write_u32(out, copy.depth);
            write_u32(out, 0);
        }
        CommandV3::Present(present) => {
            write_u64(out, present.surface_id);
            write_u32(out, present.sync_interval);
            write_u32(out, 0);
        }
        CommandV3::ExecuteIndirectDraw(draw) => {
            write_indirect_arguments(out, &draw.arguments);
        }
        CommandV3::ExecuteIndirectDrawIndexed(draw) => {
            write_indirect_arguments(out, &draw.arguments);
        }
        CommandV3::ExecuteIndirectDispatchMesh(dispatch) => {
            write_u64(out, dispatch.pipeline_id);
            write_indirect_arguments(out, &dispatch.arguments);
        }
        CommandV3::ExecuteIndirectDispatch(dispatch) => {
            write_u64(out, dispatch.pipeline_id);
            write_indirect_arguments(out, &dispatch.arguments);
            for dimension in dispatch.threads_per_group {
                write_u32(out, dimension);
            }
            write_u32(out, 0);
        }
        CommandV3::CopyTextureRegion(copy) => {
            write_subresource(out, &copy.src);
            write_subresource(out, &copy.dst);
            for value in copy.src_origin {
                write_u32(out, value);
            }
            for value in copy.dst_origin {
                write_u32(out, value);
            }
            for value in copy.size {
                write_u32(out, value);
            }
            write_u32(out, 0);
        }
        CommandV3::ResolveSubresource(resolve) => {
            write_subresource(out, &resolve.src);
            write_subresource(out, &resolve.dst);
        }
        CommandV3::SetBlendState(state) => {
            write_u16(out, state.attachments.len() as u16);
            write_u16(out, u16::from(state.independent_blend));
            write_u16(out, u16::from(state.alpha_to_coverage));
            write_u16(out, 0);
            for attachment in &state.attachments {
                out.push(u8::from(attachment.enable));
                out.push(attachment.src_color.code() as u8);
                out.push(attachment.dst_color.code() as u8);
                out.push(attachment.color_op.code() as u8);
                out.push(attachment.src_alpha.code() as u8);
                out.push(attachment.dst_alpha.code() as u8);
                out.push(attachment.alpha_op.code() as u8);
                out.push(attachment.write_mask);
            }
        }
        CommandV3::SetDepthStencilState(state) => {
            out.push(u8::from(state.depth_enable));
            out.push(u8::from(state.depth_write));
            out.push(state.depth_func.code() as u8);
            out.push(u8::from(state.stencil_enable));
            write_stencil_face(out, &state.front);
            write_stencil_face(out, &state.back);
            write_u32(out, 0);
        }
        CommandV3::SetRasterizerState(state) => {
            out.push(state.fill.code() as u8);
            out.push(state.cull.code() as u8);
            out.push(state.front_face.code() as u8);
            out.push(u8::from(state.depth_clip));
            write_i32(out, state.depth_bias);
            write_f32(out, state.slope_scaled_depth_bias);
            write_f32(out, state.depth_bias_clamp);
            write_u32(out, 0);
        }
        CommandV3::SetStencilRef(reference) => {
            write_u16(out, u16::from(reference.value));
            write_u16(out, 0);
        }
        CommandV3::SetBlendFactor(factor) => {
            for component in factor.value {
                write_f32(out, component);
            }
            write_u32(out, 0);
        }
        CommandV3::ClearRenderTargetView(clear) => {
            write_u16(out, clear.slot);
            write_u16(out, u16::from(clear.rect.is_some()));
            for component in clear.color {
                write_f32(out, component);
            }
            write_clear_rect(out, clear.rect.as_ref());
        }
        CommandV3::ClearDepthStencilView(clear) => {
            let mut flags = 0u16;
            if clear.depth.is_some() {
                flags |= 1;
            }
            if clear.stencil.is_some() {
                flags |= 2;
            }
            if clear.rect.is_some() {
                flags |= 4;
            }
            write_u16(out, flags);
            write_u16(out, 0);
            write_f32(out, clear.depth.unwrap_or(0.0));
            write_u32(out, clear.stencil.unwrap_or(0));
            write_clear_rect(out, clear.rect.as_ref());
        }
        CommandV3::CopyBufferToTexture(copy) => {
            write_u64(out, copy.allocation_id);
            write_u32(out, copy.generation);
            write_u64(out, copy.destination_allocation_id);
            write_u32(out, copy.destination_generation);
            write_u64(out, copy.offset);
            write_u32(out, copy.bytes_per_row);
            write_u32(out, copy.bytes_per_image);
            for value in copy.extent {
                write_u32(out, value);
            }
            write_u16(out, copy.mip_level);
            write_u16(out, copy.slice);
            for value in copy.origin {
                write_u32(out, value);
            }
            write_u32(out, 0);
        }
    }
}

fn write_stencil_face(out: &mut Vec<u8>, face: &StencilFaceState) {
    out.push(face.fail.code() as u8);
    out.push(face.depth_fail.code() as u8);
    out.push(face.pass.code() as u8);
    out.push(face.func.code() as u8);
    out.push(face.read_mask);
    out.push(face.write_mask);
}

fn write_clear_rect(out: &mut Vec<u8>, rect: Option<&ScissorRect>) {
    match rect {
        Some(rect) => {
            write_u32(out, rect.x);
            write_u32(out, rect.y);
            write_u32(out, rect.width);
            write_u32(out, rect.height);
        }
        None => {
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
        }
    }
}

fn write_indirect_arguments(out: &mut Vec<u8>, arguments: &IndirectArguments) {
    write_u64(out, arguments.allocation_id);
    write_u32(out, arguments.generation);
    write_u32(out, 0);
    write_u64(out, arguments.offset);
    write_u32(out, arguments.max_count);
    write_u32(out, 0);
    match &arguments.count {
        Some(count) => {
            write_u64(out, count.allocation_id);
            write_u32(out, count.generation);
            write_u32(out, 0);
            write_u64(out, count.offset);
        }
        None => {
            write_u64(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
            write_u64(out, 0);
        }
    }
}

fn write_subresource(out: &mut Vec<u8>, subresource: &TextureSubresource) {
    write_u64(out, subresource.allocation_id);
    write_u32(out, subresource.generation);
    write_u16(out, subresource.mip_level);
    write_u16(out, subresource.slice);
}

fn write_buffer_bindings(out: &mut Vec<u8>, bindings: &[BufferBinding]) {
    for binding in bindings {
        write_u64(out, binding.allocation_id);
        write_u32(out, binding.generation);
        write_u32(out, 0);
        write_u64(out, binding.offset);
        write_u64(out, binding.length);
    }
}

fn write_table_buffer_bindings(out: &mut Vec<u8>, entries: &[DescriptorTableEntryV3]) {
    for entry in entries {
        let binding = entry
            .as_buffer()
            .expect("buffer-tagged table holds only buffer entries");
        write_u64(out, binding.allocation_id);
        write_u32(out, binding.generation);
        write_u32(out, 0);
        write_u64(out, binding.offset);
        write_u64(out, binding.length);
    }
}

fn write_table_tagged_entries(out: &mut Vec<u8>, entries: &[DescriptorTableEntryV3]) {
    for entry in entries {
        out.push(descriptor_entry_kind(entry));
        match entry {
            DescriptorTableEntryV3::Buffer(binding) => {
                write_u64(out, binding.allocation_id);
                write_u32(out, binding.generation);
                write_u32(out, 0);
                write_u64(out, binding.offset);
                write_u64(out, binding.length);
            }
            DescriptorTableEntryV3::Texture(handle) | DescriptorTableEntryV3::Sampler(handle) => {
                write_u64(out, *handle);
            }
        }
    }
}

fn write_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(out: &mut Vec<u8>, value: f32) {
    write_u32(out, value.to_bits());
}

fn write_f64(out: &mut Vec<u8>, value: f64) {
    write_u64(out, value.to_bits());
}

fn validate_sequence(sequence: u32) -> Result<(), WireError> {
    if sequence == 0 {
        Err(WireError::ZeroSequence)
    } else {
        Ok(())
    }
}

fn validate_list_identity(list: ListIdentity) -> Result<(), WireError> {
    if list.list_id == 0 {
        return Err(WireError::ZeroListId);
    }
    if list.list_generation == 0 {
        return Err(WireError::ZeroListGeneration);
    }
    Ok(())
}

fn decode_list_identity(
    list_id: u64,
    list_generation: u32,
) -> Result<Option<ListIdentity>, WireError> {
    match (list_id, list_generation) {
        (0, 0) => Ok(None),
        (0, _) => Err(WireError::ZeroListId),
        (_, 0) => Err(WireError::ZeroListGeneration),
        _ => Ok(Some(ListIdentity {
            list_id,
            list_generation,
        })),
    }
}

fn validate_pipeline(pipeline_id: u64) -> Result<(), WireError> {
    if pipeline_id == 0 {
        Err(WireError::ZeroPipelineId)
    } else {
        Ok(())
    }
}

fn validate_threads(threads: u32, threads_per_group: u32) -> Result<(), WireError> {
    if threads == 0 || threads_per_group == 0 {
        Err(WireError::ZeroThreads)
    } else {
        Ok(())
    }
}

fn validate_dimensions(dimensions: [u32; 3]) -> Result<(), WireError> {
    if dimensions.contains(&0) {
        Err(WireError::ZeroThreads)
    } else {
        Ok(())
    }
}

fn validate_handle(allocation_id: u64, generation: u32) -> Result<(), WireError> {
    if allocation_id == 0 {
        return Err(WireError::ZeroAllocationId);
    }
    if generation == 0 {
        return Err(WireError::ZeroGeneration);
    }
    Ok(())
}

fn validate_length(length: u64) -> Result<(), WireError> {
    if length == 0 {
        return Err(WireError::ZeroLength);
    }
    if length > MAX_RESOURCE_LEN {
        return Err(WireError::ResourceTooLarge(length));
    }
    Ok(())
}

fn validate_range(
    allocation_id: u64,
    generation: u32,
    offset: u64,
    length: u64,
) -> Result<(), WireError> {
    validate_handle(allocation_id, generation)?;
    validate_length(length)?;
    if offset.checked_add(length).is_none() {
        return Err(WireError::OffsetOverflow);
    }
    Ok(())
}

fn validate_binding(binding: &BufferBinding) -> Result<(), WireError> {
    validate_range(
        binding.allocation_id,
        binding.generation,
        binding.offset,
        binding.length,
    )
}

fn validate_extent(field: &'static str, value: u32) -> Result<(), WireError> {
    if value == 0 {
        Err(WireError::ZeroExtent { field })
    } else {
        Ok(())
    }
}

fn checked_dimension_product(dimensions: [u32; 3]) -> Option<u64> {
    dimensions.into_iter().try_fold(1u64, |product, dimension| {
        product.checked_mul(u64::from(dimension))
    })
}

fn topology_code(topology: PrimitiveTopology) -> u16 {
    match topology {
        PrimitiveTopology::PointList => 0,
        PrimitiveTopology::LineList => 1,
        PrimitiveTopology::LineStrip => 2,
        PrimitiveTopology::TriangleList => 3,
        PrimitiveTopology::TriangleStrip => 4,
    }
}

fn topology_from_code(code: u16) -> Result<PrimitiveTopology, WireError> {
    match code {
        0 => Ok(PrimitiveTopology::PointList),
        1 => Ok(PrimitiveTopology::LineList),
        2 => Ok(PrimitiveTopology::LineStrip),
        3 => Ok(PrimitiveTopology::TriangleList),
        4 => Ok(PrimitiveTopology::TriangleStrip),
        other => Err(WireError::InvalidEnumCode {
            field: "primitive topology",
            code: other,
        }),
    }
}

fn index_format_code(format: IndexFormat) -> u16 {
    match format {
        IndexFormat::Uint16 => 0,
        IndexFormat::Uint32 => 1,
    }
}

fn index_format_from_code(code: u16) -> Result<IndexFormat, WireError> {
    match code {
        0 => Ok(IndexFormat::Uint16),
        1 => Ok(IndexFormat::Uint32),
        other => Err(WireError::InvalidEnumCode {
            field: "index format",
            code: other,
        }),
    }
}

fn load_op_code(load: LoadOp) -> u16 {
    match load {
        LoadOp::Load => 0,
        LoadOp::Clear => 1,
        LoadOp::DontCare => 2,
    }
}

fn load_op_from_code(code: u16) -> Result<LoadOp, WireError> {
    match code {
        0 => Ok(LoadOp::Load),
        1 => Ok(LoadOp::Clear),
        2 => Ok(LoadOp::DontCare),
        other => Err(WireError::InvalidEnumCode {
            field: "load op",
            code: other,
        }),
    }
}

fn store_op_code(store: StoreOp) -> u16 {
    match store {
        StoreOp::Store => 0,
        StoreOp::DontCare => 1,
    }
}

fn store_op_from_code(code: u16) -> Result<StoreOp, WireError> {
    match code {
        0 => Ok(StoreOp::Store),
        1 => Ok(StoreOp::DontCare),
        other => Err(WireError::InvalidEnumCode {
            field: "store op",
            code: other,
        }),
    }
}

fn descriptor_kind_code(kind: DescriptorKind) -> u16 {
    match kind {
        DescriptorKind::Cbv => 0,
        DescriptorKind::Srv => 1,
        DescriptorKind::Uav => 2,
        DescriptorKind::Sampler => 3,
    }
}

fn descriptor_kind_from_code(code: u16) -> Result<DescriptorKind, WireError> {
    match code {
        0 => Ok(DescriptorKind::Cbv),
        1 => Ok(DescriptorKind::Srv),
        2 => Ok(DescriptorKind::Uav),
        3 => Ok(DescriptorKind::Sampler),
        other => Err(WireError::InvalidEnumCode {
            field: "descriptor kind",
            code: other,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire;

    const COMMAND_START: usize = FRAME_HEADER_LEN + LIST_BLOCK_LEN;

    fn binding(offset: u64) -> BufferBinding {
        BufferBinding {
            allocation_id: 1,
            generation: 1,
            offset,
            length: 65_536,
        }
    }

    fn entry(offset: u64) -> DescriptorTableEntryV3 {
        DescriptorTableEntryV3::Buffer(binding(offset))
    }

    fn vertex_buffer(offset: u64) -> VertexBuffer {
        VertexBuffer {
            allocation_id: 2,
            generation: 3,
            offset,
            stride: 32,
            length: 4096,
        }
    }

    fn barrier_entry() -> BarrierEntry {
        BarrierEntry {
            allocation_id: 3,
            generation: 2,
            offset: 0,
            length: 65_536,
            before_state: 0,
            after_state: 4,
        }
    }

    fn identity() -> ListIdentity {
        ListIdentity {
            list_id: 9,
            list_generation: 2,
        }
    }

    fn color(index: u64) -> ColorAttachment {
        ColorAttachment {
            allocation_id: 100 + index,
            generation: 2,
            mip_level: 0,
            slice: (index % 4) as u16,
            load: if index == 0 {
                LoadOp::Clear
            } else {
                LoadOp::Load
            },
            store: StoreOp::Store,
            clear_color: [index as f64 * 0.25, 0.5, 0.75, 1.0],
        }
    }

    fn depth_attachment() -> DepthStencilAttachment {
        DepthStencilAttachment {
            allocation_id: 200,
            generation: 3,
            mip_level: 1,
            slice: 2,
            depth_load: LoadOp::Clear,
            depth_store: StoreOp::Store,
            clear_depth: 1.0,
            stencil_load: LoadOp::DontCare,
            stencil_store: StoreOp::DontCare,
            clear_stencil: 0,
        }
    }

    fn index_binding(format: IndexFormat) -> IndexBufferBinding {
        IndexBufferBinding {
            allocation_id: 3,
            generation: 1,
            offset: 0,
            length: 4096,
            format,
        }
    }

    fn indirect_arguments(count: Option<IndirectCountBuffer>) -> IndirectArguments {
        IndirectArguments {
            allocation_id: 11,
            generation: 2,
            offset: 32,
            max_count: 3,
            count,
        }
    }

    fn subresource(allocation_id: u64, mip_level: u16, slice: u16) -> TextureSubresource {
        TextureSubresource {
            allocation_id,
            generation: 1,
            mip_level,
            slice,
        }
    }

    fn indirect_commands() -> Vec<CommandV3> {
        vec![
            CommandV3::ExecuteIndirectDraw(ExecuteIndirectDraw {
                arguments: indirect_arguments(None),
            }),
            CommandV3::ExecuteIndirectDrawIndexed(ExecuteIndirectDrawIndexed {
                arguments: indirect_arguments(Some(IndirectCountBuffer {
                    allocation_id: 12,
                    generation: 2,
                    offset: 64,
                })),
            }),
            CommandV3::ExecuteIndirectDispatchMesh(ExecuteIndirectDispatchMesh {
                pipeline_id: 10,
                arguments: indirect_arguments(None),
            }),
            CommandV3::ExecuteIndirectDispatch(ExecuteIndirectDispatch {
                pipeline_id: 11,
                arguments: indirect_arguments(None),
                threads_per_group: [8, 4, 2],
            }),
            CommandV3::CopyTextureRegion(CopyTextureRegion {
                src: subresource(9, 0, 1),
                dst: subresource(10, 1, 2),
                src_origin: [8, 16, 0],
                dst_origin: [24, 32, 0],
                size: [32, 16, 1],
            }),
            CommandV3::ResolveSubresource(ResolveSubresource {
                src: subresource(11, 0, 0),
                dst: subresource(12, 0, 0),
            }),
        ]
    }

    fn all_commands() -> Vec<CommandV3> {
        vec![
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 256,
                threads_per_group: 64,
                bindings: vec![binding(0), binding(65_536)],
            }),
            CommandV3::Dispatch3d(Dispatch3d {
                pipeline_id: 2,
                group_counts: [2, 3, 4],
                threads_per_group: [4, 2, 1],
                bindings: vec![binding(0)],
            }),
            CommandV3::DrawIndexed(DrawIndexedState {
                index_count: 36,
                instance_count: 2,
                start_index: 3,
                base_vertex: -7,
                start_instance: 5,
            }),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 12,
                start_instance: 4,
            }),
            CommandV3::DispatchMesh(DispatchMesh {
                pipeline_id: 5,
                group_counts: [8, 4, 2],
                bindings: vec![binding(0)],
            }),
            CommandV3::ResourceBarrier(ResourceBarrier {
                entries: vec![barrier_entry(), barrier_entry()],
            }),
            CommandV3::SetPipeline(SetPipeline { pipeline_id: 6 }),
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 2,
                bytes: vec![1, 0, 0, 0, 2, 0, 0, 0],
            }),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 2,
                base_register: 1,
                kind: DescriptorKind::Srv,
                bindings: vec![entry(0), entry(65_536)],
            }),
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: (0..8).map(color).collect(),
                depth_stencil: Some(depth_attachment()),
            }),
            CommandV3::EndRenderPass,
            CommandV3::SetViewport(Viewport {
                x: 0.0,
                y: 1.5,
                width: 1920.0,
                height: 1080.0,
                near: 0.0,
                far: 1.0,
            }),
            CommandV3::SetScissor(ScissorRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id: 7 }),
            CommandV3::SetMeshPipeline(PipelineRef { pipeline_id: 8 }),
            CommandV3::SetComputePipeline(PipelineRef { pipeline_id: 9 }),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleStrip),
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 2,
                buffers: vec![vertex_buffer(0), vertex_buffer(1024)],
            }),
            CommandV3::SetIndexBuffer(index_binding(IndexFormat::Uint16)),
            CommandV3::SetIndexBuffer(index_binding(IndexFormat::Uint32)),
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 1,
                allocation_id: 4,
                generation: 2,
                offset: 256,
                length: 1024,
            }),
            CommandV3::DrawInstanced(DrawInstanced {
                vertex_count_per_instance: 3,
                instance_count: 2,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
                index_count_per_instance: 6,
                instance_count: 2,
                start_index: 0,
                base_vertex: -1,
                start_instance: 0,
            }),
            CommandV3::CopyBuffer(CopyBuffer {
                src_allocation_id: 5,
                src_generation: 1,
                src_offset: 0,
                dst_allocation_id: 6,
                dst_generation: 1,
                dst_offset: 4096,
                length: 4096,
            }),
            CommandV3::CopyTexture(CopyTexture {
                src_allocation_id: 7,
                src_generation: 1,
                src_mip: 0,
                src_slice: 0,
                dst_allocation_id: 8,
                dst_generation: 1,
                dst_mip: 1,
                dst_slice: 2,
                width: 64,
                height: 64,
                depth: 1,
            }),
            CommandV3::ExecuteIndirectDraw(ExecuteIndirectDraw {
                arguments: indirect_arguments(None),
            }),
            CommandV3::ExecuteIndirectDrawIndexed(ExecuteIndirectDrawIndexed {
                arguments: indirect_arguments(Some(IndirectCountBuffer {
                    allocation_id: 12,
                    generation: 2,
                    offset: 64,
                })),
            }),
            CommandV3::ExecuteIndirectDispatchMesh(ExecuteIndirectDispatchMesh {
                pipeline_id: 10,
                arguments: indirect_arguments(None),
            }),
            CommandV3::ExecuteIndirectDispatch(ExecuteIndirectDispatch {
                pipeline_id: 11,
                arguments: indirect_arguments(None),
                threads_per_group: [8, 4, 2],
            }),
            CommandV3::CopyTextureRegion(CopyTextureRegion {
                src: subresource(9, 0, 1),
                dst: subresource(10, 1, 2),
                src_origin: [8, 16, 0],
                dst_origin: [24, 32, 0],
                size: [32, 16, 1],
            }),
            CommandV3::ResolveSubresource(ResolveSubresource {
                src: subresource(11, 0, 0),
                dst: subresource(12, 0, 0),
            }),
            CommandV3::SetBlendState(BlendState {
                independent_blend: true,
                alpha_to_coverage: false,
                attachments: vec![
                    BlendAttachment {
                        enable: true,
                        src_color: BlendFactor::SrcAlpha,
                        dst_color: BlendFactor::InvSrcAlpha,
                        color_op: BlendOperation::Add,
                        src_alpha: BlendFactor::One,
                        dst_alpha: BlendFactor::InvSrcAlpha,
                        alpha_op: BlendOperation::Max,
                        write_mask: 0xf,
                    },
                    BlendAttachment {
                        enable: false,
                        src_color: BlendFactor::One,
                        dst_color: BlendFactor::Zero,
                        color_op: BlendOperation::RevSubtract,
                        src_alpha: BlendFactor::DestColor,
                        dst_alpha: BlendFactor::SrcAlphaSat,
                        alpha_op: BlendOperation::Min,
                        write_mask: 0b0101,
                    },
                ],
            }),
            CommandV3::SetDepthStencilState(DepthStencilState {
                depth_enable: true,
                depth_write: true,
                depth_func: CompareFunction::LessEqual,
                stencil_enable: true,
                front: StencilFaceState {
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::IncrSat,
                    pass: StencilOperation::Replace,
                    func: CompareFunction::Equal,
                    read_mask: 0xff,
                    write_mask: 0x0f,
                },
                back: StencilFaceState {
                    fail: StencilOperation::Decr,
                    depth_fail: StencilOperation::Invert,
                    pass: StencilOperation::Zero,
                    func: CompareFunction::Always,
                    read_mask: 0x7f,
                    write_mask: 0xf0,
                },
            }),
            CommandV3::SetRasterizerState(RasterizerState {
                fill: FillMode::Wireframe,
                cull: CullMode::Back,
                front_face: FrontFaceWinding::CounterClockwise,
                depth_bias: -16,
                slope_scaled_depth_bias: 1.5,
                depth_bias_clamp: 0.25,
                depth_clip: false,
            }),
            CommandV3::SetStencilRef(StencilRef { value: 7 }),
            CommandV3::SetBlendFactor(BlendFactorValue {
                value: [0.25, 0.5, 0.75, 1.0],
            }),
            CommandV3::ClearRenderTargetView(ClearRenderTargetView {
                slot: 1,
                rect: Some(ScissorRect {
                    x: 4,
                    y: 8,
                    width: 16,
                    height: 32,
                }),
                color: [0.0, 0.25, 0.5, 1.0],
            }),
            CommandV3::ClearDepthStencilView(ClearDepthStencilView {
                depth: Some(0.5),
                stencil: Some(3),
                rect: None,
            }),
            CommandV3::CopyBufferToTexture(CopyBufferToTexture {
                allocation_id: 13,
                generation: 2,
                destination_allocation_id: 14,
                destination_generation: 3,
                offset: 1024,
                bytes_per_row: 256,
                bytes_per_image: 0,
                extent: [4, 4, 1],
                mip_level: 1,
                slice: 2,
                origin: [1, 2, 0],
            }),
            CommandV3::Present(Present {
                surface_id: 1,
                sync_interval: 1,
            }),
        ]
    }

    fn rich_submission() -> SubmissionV3 {
        SubmissionV3 {
            sequence: 5,
            list: Some(identity()),
            commands: all_commands(),
        }
    }

    fn submission_with(commands: Vec<CommandV3>) -> SubmissionV3 {
        SubmissionV3 {
            sequence: 1,
            list: None,
            commands,
        }
    }

    fn encoded_rich() -> Vec<u8> {
        encode_submission_v3(&rich_submission()).expect("rich fixture should encode")
    }

    fn single_frame(command: CommandV3) -> Vec<u8> {
        encode_submission_v3(&submission_with(vec![command])).expect("single fixture should encode")
    }

    fn assert_encode_error(command: CommandV3, expected: WireError) {
        assert_eq!(
            encode_submission_v3(&submission_with(vec![command])),
            Err(expected)
        );
    }

    fn assert_roundtrip(submission: &SubmissionV3) {
        let frame = encode_submission_v3(submission).expect("fixture should encode");
        assert_eq!(
            &decode_submission_v3(&frame).expect("fixture should decode"),
            submission
        );
    }

    fn assert_decode_error(frame: &[u8], offset: usize, bytes: &[u8], expected: WireError) {
        let mut frame = frame.to_vec();
        frame[offset..offset + bytes.len()].copy_from_slice(bytes);
        assert_eq!(decode_submission_v3(&frame), Err(expected));
    }

    #[test]
    fn wire_v3_roundtrips_every_command_variant() {
        let submission = rich_submission();
        let decoded = decode_submission_v3(&encoded_rich()).expect("rich frame decodes");
        assert_eq!(decoded, submission);
        assert_eq!(decoded.commands.len(), 40);

        let opcodes: std::collections::BTreeSet<u16> =
            decoded.commands.iter().map(CommandV3::opcode).collect();
        for opcode in [
            OP_DISPATCH_1D,
            OP_DISPATCH_3D,
            OP_DRAW_INDEXED,
            OP_DRAW,
            OP_DISPATCH_MESH,
            OP_RESOURCE_BARRIER,
            OP_SET_PIPELINE,
            OP_ROOT_CONSTANTS,
            OP_DESCRIPTOR_TABLE,
            OP_BEGIN_RENDER_PASS,
            OP_END_RENDER_PASS,
            OP_SET_VIEWPORT,
            OP_SET_SCISSOR,
            OP_SET_GRAPHICS_PIPELINE,
            OP_SET_MESH_PIPELINE,
            OP_SET_COMPUTE_PIPELINE,
            OP_SET_PRIMITIVE_TOPOLOGY,
            OP_SET_VERTEX_BUFFERS,
            OP_SET_INDEX_BUFFER,
            OP_SET_ROOT_BUFFER_VIEW,
            OP_DRAW_INSTANCED,
            OP_DRAW_INDEXED_INSTANCED,
            OP_COPY_BUFFER,
            OP_COPY_TEXTURE,
            OP_EXECUTE_INDIRECT_DRAW,
            OP_EXECUTE_INDIRECT_DRAW_INDEXED,
            OP_EXECUTE_INDIRECT_DISPATCH_MESH,
            OP_EXECUTE_INDIRECT_DISPATCH,
            OP_COPY_TEXTURE_REGION,
            OP_RESOLVE_SUBRESOURCE,
            OP_SET_BLEND_STATE,
            OP_SET_DEPTH_STENCIL_STATE,
            OP_SET_RASTERIZER_STATE,
            OP_SET_STENCIL_REF,
            OP_SET_BLEND_FACTOR,
            OP_CLEAR_RENDER_TARGET_VIEW,
            OP_CLEAR_DEPTH_STENCIL_VIEW,
            OP_COPY_BUFFER_TO_TEXTURE,
            OP_PRESENT,
        ] {
            assert!(opcodes.contains(&opcode), "opcode {opcode} missing");
        }
    }

    #[test]
    fn wire_v3_mrt_eight_colors_depth_and_both_index_widths_roundtrip() {
        let decoded = decode_submission_v3(&encoded_rich()).expect("rich frame decodes");

        let CommandV3::BeginRenderPass(pass) = &decoded.commands[9] else {
            panic!("expected a render pass at index 9");
        };
        assert_eq!(pass.colors.len(), 8);
        assert_eq!(pass.colors[0].load, LoadOp::Clear);
        assert_eq!(pass.colors[0].clear_color, [0.0, 0.5, 0.75, 1.0]);
        assert_eq!(pass.colors[1].load, LoadOp::Load);
        assert_eq!(pass.colors[7].allocation_id, 107);
        assert_eq!(pass.depth_stencil, Some(depth_attachment()));

        let formats: Vec<IndexFormat> = decoded
            .commands
            .iter()
            .filter_map(|command| match command {
                CommandV3::SetIndexBuffer(binding) => Some(binding.format),
                _ => None,
            })
            .collect();
        assert_eq!(formats, vec![IndexFormat::Uint16, IndexFormat::Uint32]);
    }

    #[test]
    fn wire_v3_truncation_sweep_rejects_every_prefix() {
        let frame = encoded_rich();
        for length in 0..frame.len() {
            assert!(
                decode_submission_v3(&frame[..length]).is_err(),
                "prefix {length} must not decode"
            );
        }
        assert!(decode_submission_v3(&frame).is_ok());
    }

    #[test]
    fn wire_v3_rejects_mutated_tags_reserved_words_and_trailing_bytes() {
        let frame = encoded_rich();

        assert_decode_error(
            &frame,
            4,
            &2u16.to_le_bytes(),
            WireError::UnsupportedVersion(2),
        );
        assert_decode_error(&frame, 6, &2u16.to_le_bytes(), WireError::UnexpectedKind(2));
        assert_decode_error(&frame, 14, &1u16.to_le_bytes(), WireError::NonZeroReserved);
        assert_decode_error(
            &frame,
            COMMAND_START,
            &99u16.to_le_bytes(),
            WireError::UnknownOpcode(99),
        );
        assert_decode_error(
            &frame,
            COMMAND_START + 2,
            &1u16.to_le_bytes(),
            WireError::UnsupportedFlags(1),
        );

        let mut trailing = frame.clone();
        trailing.push(0);
        assert_eq!(
            decode_submission_v3(&trailing),
            Err(WireError::TrailingBytes)
        );

        let mut short = frame;
        short.pop();
        assert_eq!(decode_submission_v3(&short), Err(WireError::Truncated));
    }

    #[test]
    fn wire_v3_rejects_zero_sequence_and_half_list_identity() {
        let frame = encoded_rich();
        assert_decode_error(&frame, 8, &[0; 4], WireError::ZeroSequence);
        assert_decode_error(&frame, 20, &[0; 8], WireError::ZeroListId);
        assert_decode_error(&frame, 28, &[0; 4], WireError::ZeroListGeneration);

        assert_eq!(
            encode_submission_v3(&SubmissionV3 {
                sequence: 0,
                list: None,
                commands: Vec::new(),
            }),
            Err(WireError::ZeroSequence)
        );
        assert_eq!(
            encode_submission_v3(&SubmissionV3 {
                sequence: 1,
                list: Some(ListIdentity {
                    list_id: 0,
                    list_generation: 1,
                }),
                commands: Vec::new(),
            }),
            Err(WireError::ZeroListId)
        );
        assert_eq!(
            encode_submission_v3(&SubmissionV3 {
                sequence: 1,
                list: Some(ListIdentity {
                    list_id: 1,
                    list_generation: 0,
                }),
                commands: Vec::new(),
            }),
            Err(WireError::ZeroListGeneration)
        );
    }

    #[test]
    fn wire_v3_rejects_zero_pipeline_and_thread_dimensions() {
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 0,
                threads: 1,
                threads_per_group: 1,
                bindings: Vec::new(),
            }),
            WireError::ZeroPipelineId,
        );
        assert_encode_error(
            CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id: 0 }),
            WireError::ZeroPipelineId,
        );
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 0,
                threads_per_group: 1,
                bindings: Vec::new(),
            }),
            WireError::ZeroThreads,
        );
        assert_encode_error(
            CommandV3::Dispatch3d(Dispatch3d {
                pipeline_id: 1,
                group_counts: [1, 0, 1],
                threads_per_group: [1, 1, 1],
                bindings: Vec::new(),
            }),
            WireError::ZeroThreads,
        );
        assert_encode_error(
            CommandV3::DispatchMesh(DispatchMesh {
                pipeline_id: 1,
                group_counts: [1, 1, 0],
                bindings: Vec::new(),
            }),
            WireError::ZeroThreads,
        );

        let overflow = [u32::MAX; 3];
        assert_encode_error(
            CommandV3::Dispatch3d(Dispatch3d {
                pipeline_id: 1,
                group_counts: overflow,
                threads_per_group: [1, 1, 1],
                bindings: Vec::new(),
            }),
            WireError::ThreadCountOverflow,
        );
        assert_encode_error(
            CommandV3::DispatchMesh(DispatchMesh {
                pipeline_id: 1,
                group_counts: overflow,
                bindings: Vec::new(),
            }),
            WireError::ThreadCountOverflow,
        );
    }

    #[test]
    fn wire_v3_rejects_zero_draw_counts() {
        assert_encode_error(
            CommandV3::DrawIndexed(DrawIndexedState {
                index_count: 0,
                instance_count: 1,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroIndexCount,
        );
        assert_encode_error(
            CommandV3::DrawIndexed(DrawIndexedState {
                index_count: 1,
                instance_count: 0,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroInstanceCount,
        );
        assert_encode_error(
            CommandV3::Draw(DrawState {
                vertex_count: 0,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroVertexCount,
        );
        assert_encode_error(
            CommandV3::Draw(DrawState {
                vertex_count: 1,
                instance_count: 0,
                start_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroInstanceCount,
        );
        assert_encode_error(
            CommandV3::DrawInstanced(DrawInstanced {
                vertex_count_per_instance: 0,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroVertexCount,
        );
        assert_encode_error(
            CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
                index_count_per_instance: 0,
                instance_count: 1,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroIndexCount,
        );
        assert_encode_error(
            CommandV3::DrawIndexedInstanced(DrawIndexedInstanced {
                index_count_per_instance: 1,
                instance_count: 0,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            WireError::ZeroInstanceCount,
        );
    }

    #[test]
    fn wire_v3_rejects_zero_handles_lengths_and_offset_overflow() {
        let mut zero_binding = binding(0);
        zero_binding.allocation_id = 0;
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![zero_binding],
            }),
            WireError::ZeroAllocationId,
        );

        let mut zero_generation = binding(0);
        zero_generation.generation = 0;
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![zero_generation],
            }),
            WireError::ZeroGeneration,
        );

        let mut zero_length = binding(0);
        zero_length.length = 0;
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![zero_length],
            }),
            WireError::ZeroLength,
        );

        let mut oversized = binding(0);
        oversized.length = MAX_RESOURCE_LEN + 1;
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![oversized],
            }),
            WireError::ResourceTooLarge(MAX_RESOURCE_LEN + 1),
        );

        let mut overflow = binding(0);
        overflow.offset = u64::MAX;
        overflow.length = 2;
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![overflow],
            }),
            WireError::OffsetOverflow,
        );

        let mut zero_color = color(0);
        zero_color.allocation_id = 0;
        assert_encode_error(
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![zero_color],
                depth_stencil: None,
            }),
            WireError::ZeroAllocationId,
        );

        let mut zero_stride = vertex_buffer(0);
        zero_stride.stride = 0;
        assert_encode_error(
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 0,
                buffers: vec![zero_stride],
            }),
            WireError::ZeroExtent {
                field: "vertex stride",
            },
        );

        let mut zero_index = index_binding(IndexFormat::Uint16);
        zero_index.length = 0;
        assert_encode_error(CommandV3::SetIndexBuffer(zero_index), WireError::ZeroLength);

        let mut copy = CopyBuffer {
            src_allocation_id: 1,
            src_generation: 1,
            src_offset: 0,
            dst_allocation_id: 2,
            dst_generation: 1,
            dst_offset: 0,
            length: 0,
        };
        assert_encode_error(CommandV3::CopyBuffer(copy), WireError::ZeroLength);
        copy.length = 4;
        copy.src_offset = u64::MAX;
        assert_encode_error(CommandV3::CopyBuffer(copy), WireError::OffsetOverflow);

        let mut barrier = ResourceBarrier {
            entries: vec![barrier_entry()],
        };
        barrier.entries[0].length = 0;
        assert_encode_error(CommandV3::ResourceBarrier(barrier), WireError::ZeroLength);

        let mut scissor = ScissorRect {
            x: 0,
            y: 0,
            width: 0,
            height: 1,
        };
        assert_encode_error(
            CommandV3::SetScissor(scissor),
            WireError::ZeroExtent {
                field: "scissor width",
            },
        );
        scissor.width = 1;
        scissor.height = 0;
        assert_encode_error(
            CommandV3::SetScissor(scissor),
            WireError::ZeroExtent {
                field: "scissor height",
            },
        );

        let mut copy_texture = CopyTexture {
            src_allocation_id: 1,
            src_generation: 1,
            src_mip: 0,
            src_slice: 0,
            dst_allocation_id: 2,
            dst_generation: 1,
            dst_mip: 0,
            dst_slice: 0,
            width: 0,
            height: 1,
            depth: 1,
        };
        assert_encode_error(
            CommandV3::CopyTexture(copy_texture),
            WireError::ZeroExtent {
                field: "copy width",
            },
        );
        copy_texture.width = 1;
        copy_texture.height = 0;
        assert_encode_error(
            CommandV3::CopyTexture(copy_texture),
            WireError::ZeroExtent {
                field: "copy height",
            },
        );
        copy_texture.height = 1;
        copy_texture.depth = 0;
        assert_encode_error(
            CommandV3::CopyTexture(copy_texture),
            WireError::ZeroExtent {
                field: "copy depth",
            },
        );
        copy_texture.depth = 1;
        assert_roundtrip(&submission_with(vec![CommandV3::CopyTexture(copy_texture)]));
    }

    #[test]
    fn wire_v3_rejects_undefined_enum_codes_and_present_interval() {
        let topology = single_frame(CommandV3::SetPrimitiveTopology(
            PrimitiveTopology::TriangleList,
        ));
        assert_decode_error(
            &topology,
            COMMAND_START + 4,
            &9u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "primitive topology",
                code: 9,
            },
        );

        let index = single_frame(CommandV3::SetIndexBuffer(index_binding(
            IndexFormat::Uint16,
        )));
        assert_decode_error(
            &index,
            COMMAND_START + 32,
            &9u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "index format",
                code: 9,
            },
        );

        let table = single_frame(CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Cbv,
            bindings: vec![entry(0)],
        }));
        assert_decode_error(
            &table,
            COMMAND_START + 8,
            &9u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "descriptor kind",
                code: 9,
            },
        );

        let pass = single_frame(CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color(0)],
            depth_stencil: Some(depth_attachment()),
        }));
        assert_decode_error(
            &pass,
            COMMAND_START + 12 + 16,
            &9u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "load op",
                code: 9,
            },
        );
        assert_decode_error(
            &pass,
            COMMAND_START + 12 + 18,
            &9u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "store op",
                code: 9,
            },
        );
        assert_decode_error(
            &pass,
            COMMAND_START + 6,
            &2u16.to_le_bytes(),
            WireError::TooManyDepthAttachments(2),
        );

        let present = single_frame(CommandV3::Present(Present {
            surface_id: 1,
            sync_interval: 0,
        }));
        assert_decode_error(
            &present,
            COMMAND_START + 12,
            &2u32.to_le_bytes(),
            WireError::UnsupportedSyncInterval(2),
        );
        assert_encode_error(
            CommandV3::Present(Present {
                surface_id: 1,
                sync_interval: 2,
            }),
            WireError::UnsupportedSyncInterval(2),
        );
    }

    #[test]
    fn wire_v3_encodes_table_resource_handles_and_keeps_buffer_frames() {
        // A buffer table keeps the legacy tag 0 and 32-byte entries, so a
        // frame produced before resource entries existed still decodes.
        let buffer_frame = single_frame(CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 1,
            base_register: 2,
            kind: DescriptorKind::Srv,
            bindings: vec![entry(0), entry(64)],
        }));
        let command = &buffer_frame[COMMAND_START..];
        assert_eq!(&command[12..14], &DESCRIPTOR_ENTRY_BUFFER.to_le_bytes());
        assert_eq!(
            buffer_frame.len(),
            COMMAND_START + DESCRIPTOR_TABLE_HEADER_LEN + 2 * BINDING_LEN
        );
        let decoded = decode_submission_v3(&buffer_frame).expect("buffer table decodes");
        assert_eq!(
            decoded.commands[0],
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 1,
                base_register: 2,
                kind: DescriptorKind::Srv,
                bindings: vec![entry(0), entry(64)],
            })
        );

        // Texture entries are 8-byte handles behind tag 1.
        let texture_table = DescriptorTableV3 {
            set: 3,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![
                DescriptorTableEntryV3::Texture(0x2100),
                DescriptorTableEntryV3::Texture(0x2101),
            ],
        };
        let texture_frame = single_frame(CommandV3::SetDescriptorTable(texture_table.clone()));
        let command = &texture_frame[COMMAND_START..];
        assert_eq!(&command[12..14], &DESCRIPTOR_ENTRY_RESOURCE.to_le_bytes());
        assert_eq!(
            texture_frame.len(),
            COMMAND_START + DESCRIPTOR_TABLE_HEADER_LEN + 2 * RESOURCE_HANDLE_LEN
        );
        assert_eq!(
            &command[14..22],
            &0x2100u64.to_le_bytes(),
            "first texture handle"
        );
        assert_eq!(
            &command[22..30],
            &0x2101u64.to_le_bytes(),
            "second texture handle"
        );
        assert_eq!(
            decode_submission_v3(&texture_frame).expect("texture table decodes"),
            submission_with(vec![CommandV3::SetDescriptorTable(texture_table)])
        );

        // Sampler entries use the same tag and decode to sampler handles.
        let sampler_table = DescriptorTableV3 {
            set: 4,
            base_register: 1,
            kind: DescriptorKind::Sampler,
            bindings: vec![DescriptorTableEntryV3::Sampler(0x3100)],
        };
        assert_roundtrip(&submission_with(vec![CommandV3::SetDescriptorTable(
            sampler_table,
        )]));

        // A truncated resource frame is refused, never partially decoded.
        let truncated = &texture_frame[..texture_frame.len() - 1];
        assert_eq!(
            decode_submission_v3(truncated),
            Err(WireError::Truncated),
            "a short resource entry must be refused"
        );

        // Zero handles and kind mismatches are encoding errors.
        assert_encode_error(
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![DescriptorTableEntryV3::Texture(0)],
            }),
            WireError::ZeroDescriptorHandle,
        );
        assert_encode_error(
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Sampler,
                bindings: vec![DescriptorTableEntryV3::Texture(0x2100)],
            }),
            WireError::DescriptorEntryMismatch,
        );
        assert_encode_error(
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Cbv,
                bindings: vec![entry(0), DescriptorTableEntryV3::Texture(0x2100)],
            }),
            WireError::DescriptorEntryMismatch,
        );
    }

    #[test]
    fn wire_v3_carries_a_mixed_descriptor_table_with_per_entry_tags() {
        // A buffer and a texture in one SRV table use the tagged form: each
        // entry starts with its own kind byte, and no entry is dropped.
        let mixed_table = DescriptorTableV3 {
            set: 1,
            base_register: 2,
            kind: DescriptorKind::Srv,
            bindings: vec![entry(0), DescriptorTableEntryV3::Texture(0x2100)],
        };
        let frame = single_frame(CommandV3::SetDescriptorTable(mixed_table.clone()));
        let command = &frame[COMMAND_START..];
        assert_eq!(&command[12..14], &DESCRIPTOR_ENTRY_TAGGED.to_le_bytes());
        assert_eq!(
            frame.len(),
            COMMAND_START
                + DESCRIPTOR_TABLE_HEADER_LEN
                + DESCRIPTOR_ENTRY_TAG_LEN
                + BINDING_LEN
                + DESCRIPTOR_ENTRY_TAG_LEN
                + RESOURCE_HANDLE_LEN
        );
        let texture_tag = DESCRIPTOR_TABLE_HEADER_LEN + DESCRIPTOR_ENTRY_TAG_LEN + BINDING_LEN;
        assert_eq!(
            command[DESCRIPTOR_TABLE_HEADER_LEN],
            DESCRIPTOR_ENTRY_KIND_BUFFER
        );
        assert_eq!(command[texture_tag], DESCRIPTOR_ENTRY_KIND_TEXTURE);
        assert_eq!(
            &command[texture_tag + DESCRIPTOR_ENTRY_TAG_LEN
                ..texture_tag + DESCRIPTOR_ENTRY_TAG_LEN + RESOURCE_HANDLE_LEN],
            &0x2100u64.to_le_bytes()
        );
        assert_eq!(
            decode_submission_v3(&frame).expect("mixed table decodes"),
            submission_with(vec![CommandV3::SetDescriptorTable(mixed_table)])
        );

        // A homogeneous table keeps its old tag and exact byte layout: no
        // tagged header, no per-entry bytes.
        let buffer_frame = single_frame(CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![entry(0), entry(64)],
        }));
        assert_eq!(
            &buffer_frame[COMMAND_START + 12..COMMAND_START + 14],
            &DESCRIPTOR_ENTRY_BUFFER.to_le_bytes()
        );
        assert_eq!(
            buffer_frame.len(),
            COMMAND_START + DESCRIPTOR_TABLE_HEADER_LEN + 2 * BINDING_LEN
        );

        let texture_frame = single_frame(CommandV3::SetDescriptorTable(DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![
                DescriptorTableEntryV3::Texture(0x2100),
                DescriptorTableEntryV3::Texture(0x2101),
            ],
        }));
        assert_eq!(
            &texture_frame[COMMAND_START + 12..COMMAND_START + 14],
            &DESCRIPTOR_ENTRY_RESOURCE.to_le_bytes()
        );
        assert_eq!(
            texture_frame.len(),
            COMMAND_START + DESCRIPTOR_TABLE_HEADER_LEN + 2 * RESOURCE_HANDLE_LEN
        );
    }

    #[test]
    fn wire_v3_keeps_every_mixed_table_entry_in_guest_order() {
        // Four entries that alternate shape: every entry keeps its own kind
        // byte and the guest's position, and nothing is merged or dropped.
        let mixed = DescriptorTableV3 {
            set: 5,
            base_register: 3,
            kind: DescriptorKind::Srv,
            bindings: vec![
                entry(0),
                DescriptorTableEntryV3::Texture(0x2100),
                entry(64),
                DescriptorTableEntryV3::Texture(0x2101),
            ],
        };
        let frame = single_frame(CommandV3::SetDescriptorTable(mixed.clone()));
        let command = &frame[COMMAND_START..];
        assert_eq!(&command[12..14], &DESCRIPTOR_ENTRY_TAGGED.to_le_bytes());

        let tag0 = DESCRIPTOR_TABLE_HEADER_LEN;
        let payload0 = tag0 + DESCRIPTOR_ENTRY_TAG_LEN;
        let tag1 = payload0 + BINDING_LEN;
        let payload1 = tag1 + DESCRIPTOR_ENTRY_TAG_LEN;
        let tag2 = payload1 + RESOURCE_HANDLE_LEN;
        let payload2 = tag2 + DESCRIPTOR_ENTRY_TAG_LEN;
        let tag3 = payload2 + BINDING_LEN;
        let payload3 = tag3 + DESCRIPTOR_ENTRY_TAG_LEN;
        assert_eq!(
            frame.len(),
            COMMAND_START + payload3 + RESOURCE_HANDLE_LEN,
            "two buffer payloads and two handle payloads, one tag byte each"
        );
        assert_eq!(command[tag0], DESCRIPTOR_ENTRY_KIND_BUFFER);
        assert_eq!(command[tag1], DESCRIPTOR_ENTRY_KIND_TEXTURE);
        assert_eq!(command[tag2], DESCRIPTOR_ENTRY_KIND_BUFFER);
        assert_eq!(command[tag3], DESCRIPTOR_ENTRY_KIND_TEXTURE);
        // A buffer payload is allocation(8) generation(4) reserved(4)
        // offset(8) length(8); the offset field proves the entry positions.
        assert_eq!(&command[payload0 + 16..payload0 + 24], &0u64.to_le_bytes());
        assert_eq!(&command[payload2 + 16..payload2 + 24], &64u64.to_le_bytes());
        assert_eq!(&command[payload1..payload1 + 8], &0x2100u64.to_le_bytes());
        assert_eq!(&command[payload3..payload3 + 8], &0x2101u64.to_le_bytes());

        assert_eq!(
            decode_submission_v3(&frame).expect("the mixed table decodes"),
            submission_with(vec![CommandV3::SetDescriptorTable(mixed)])
        );
    }

    #[test]
    fn wire_v3_keeps_a_mixed_view_table_and_a_sampler_table_in_one_set() {
        // One descriptor set holds a view table that mixes a buffer range with
        // a registered texture, plus its sampler table at its own base
        // register. Each root table keeps its own kind and entry order.
        let views = DescriptorTableV3 {
            set: 7,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![entry(0), DescriptorTableEntryV3::Texture(0x2100)],
        };
        let samplers = DescriptorTableV3 {
            set: 7,
            base_register: 4,
            kind: DescriptorKind::Sampler,
            bindings: vec![DescriptorTableEntryV3::Sampler(0x3100)],
        };
        let submission = submission_with(vec![
            CommandV3::SetDescriptorTable(views.clone()),
            CommandV3::SetDescriptorTable(samplers.clone()),
        ]);
        assert_roundtrip(&submission);

        let frame = encode_submission_v3(&submission).expect("the set encodes");
        let decoded = decode_submission_v3(&frame).expect("the set decodes");
        let CommandV3::SetDescriptorTable(first) = &decoded.commands[0] else {
            panic!("the first command is the mixed view table");
        };
        let CommandV3::SetDescriptorTable(second) = &decoded.commands[1] else {
            panic!("the second command is the sampler table");
        };
        assert_eq!(first, &views);
        assert_eq!(first.kind, DescriptorKind::Srv);
        assert_eq!(
            first.bindings,
            vec![entry(0), DescriptorTableEntryV3::Texture(0x2100)],
            "the buffer and the texture stay in guest order"
        );
        assert_eq!(second, &samplers);
        assert_eq!(second.kind, DescriptorKind::Sampler);
        assert_eq!(second.base_register, 4);

        // The uniform sampler table still uses the legacy single tag and
        // 8-byte handles, so the common case keeps its old byte layout.
        let sampler_frame = single_frame(CommandV3::SetDescriptorTable(samplers));
        assert_eq!(
            &sampler_frame[COMMAND_START + 12..COMMAND_START + 14],
            &DESCRIPTOR_ENTRY_RESOURCE.to_le_bytes()
        );
        assert_eq!(
            sampler_frame.len(),
            COMMAND_START + DESCRIPTOR_TABLE_HEADER_LEN + RESOURCE_HANDLE_LEN
        );
    }

    #[test]
    fn wire_v3_names_the_slot_of_an_unknown_per_entry_tag() {
        let table = DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![entry(0), DescriptorTableEntryV3::Texture(0x2100)],
        };
        let mut frame = single_frame(CommandV3::SetDescriptorTable(table));
        // The second entry's kind byte follows the header, the first kind
        // byte and the 32-byte buffer range.
        let second_tag = COMMAND_START
            + DESCRIPTOR_TABLE_HEADER_LEN
            + DESCRIPTOR_ENTRY_TAG_LEN
            + BINDING_LEN;
        frame[second_tag] = 9;
        assert_eq!(
            decode_submission_v3(&frame),
            Err(WireError::InvalidEnumCode {
                field: "descriptor table entry 1 tag",
                code: 9,
            })
        );
    }

    #[test]
    fn wire_v3_accepts_boundary_counts_and_rejects_one_more() {
        // 8 color attachments are accepted; 8 is the maximum.
        assert_roundtrip(&submission_with(vec![CommandV3::BeginRenderPass(
            BeginRenderPass {
                colors: (0..8).map(color).collect(),
                depth_stencil: Some(depth_attachment()),
            },
        )]));
        assert_encode_error(
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: (0..9).map(color).collect(),
                depth_stencil: None,
            }),
            WireError::TooManyColorAttachments(9),
        );
        assert_encode_error(
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: Vec::new(),
                depth_stencil: None,
            }),
            WireError::ZeroColorAttachments,
        );

        // 16 vertex buffers are accepted; 16 is the maximum.
        assert_roundtrip(&submission_with(vec![CommandV3::SetVertexBuffers(
            VertexBufferSet {
                start_slot: 0,
                buffers: (0..16).map(|index| vertex_buffer(index * 32)).collect(),
            },
        )]));
        assert_encode_error(
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 0,
                buffers: (0..17).map(|index| vertex_buffer(index * 32)).collect(),
            }),
            WireError::TooManyVertexBuffers(17),
        );
        assert_encode_error(
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 0,
                buffers: Vec::new(),
            }),
            WireError::ZeroVertexBuffers,
        );

        // 32 descriptor bindings are accepted; 32 is the maximum.
        assert_roundtrip(&submission_with(vec![CommandV3::SetDescriptorTable(
            DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Cbv,
                bindings: (0..32).map(entry).collect(),
            },
        )]));
        assert_encode_error(
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Cbv,
                bindings: vec![entry(0); 33],
            }),
            WireError::TooManyDescriptorBindingsV3(33),
        );
        assert_encode_error(
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Sampler,
                bindings: Vec::new(),
            }),
            WireError::ZeroDescriptorCount,
        );

        // 256 root-constant bytes are accepted; 256 is the maximum.
        assert_roundtrip(&submission_with(vec![CommandV3::SetRootConstants(
            RootConstantsV3 {
                slot: 0,
                bytes: vec![0xa5; 256],
            },
        )]));
        assert_encode_error(
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: vec![0xa5; 257],
            }),
            WireError::RootConstantsTooLarge(257),
        );
        assert_encode_error(
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: Vec::new(),
            }),
            WireError::ZeroRootConstants,
        );

        // 32 barrier entries are accepted; 32 is the maximum.
        assert_roundtrip(&submission_with(vec![CommandV3::ResourceBarrier(
            ResourceBarrier {
                entries: vec![barrier_entry(); 32],
            },
        )]));
        assert_encode_error(
            CommandV3::ResourceBarrier(ResourceBarrier {
                entries: vec![barrier_entry(); 33],
            }),
            WireError::TooManyBarriersV3(33),
        );
        assert_encode_error(
            CommandV3::ResourceBarrier(ResourceBarrier {
                entries: Vec::new(),
            }),
            WireError::ZeroBarrierCount,
        );

        // Dispatch bindings keep the version 2 per-command limit.
        assert_encode_error(
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 1,
                threads: 1,
                threads_per_group: 1,
                bindings: vec![binding(0); 17],
            }),
            WireError::TooManyBindings(17),
        );

        // 1024 commands are accepted; 1025 are rejected before allocation.
        let maximum = submission_with(vec![CommandV3::EndRenderPass; MAX_COMMANDS_V3]);
        let frame = encode_submission_v3(&maximum).expect("maximum command count encodes");
        assert_eq!(
            decode_submission_v3(&frame)
                .expect("maximum command count decodes")
                .commands
                .len(),
            MAX_COMMANDS_V3
        );
        assert_eq!(
            encode_submission_v3(&submission_with(vec![CommandV3::EndRenderPass; 1025])),
            Err(WireError::TooManyCommandsV3(1025))
        );
    }

    #[test]
    fn wire_v3_into_decode_reuses_command_and_nested_storage() {
        let frame = encoded_rich();
        let mut out = SubmissionV3 {
            sequence: 0,
            list: None,
            commands: Vec::new(),
        };
        decode_submission_v3_into(&mut out, &frame).expect("warm decode");
        assert_eq!(out, rich_submission());

        let command_pointer = out.commands.as_ptr();
        let command_capacity = out.commands.capacity();
        let CommandV3::Dispatch1d(dispatch) = &out.commands[0] else {
            panic!("expected a 1D dispatch first");
        };
        let binding_pointer = dispatch.bindings.as_ptr();
        let binding_capacity = dispatch.bindings.capacity();

        decode_submission_v3_into(&mut out, &frame).expect("second decode");
        assert_eq!(out, rich_submission());
        assert_eq!(out.commands.as_ptr(), command_pointer);
        assert_eq!(out.commands.capacity(), command_capacity);
        let CommandV3::Dispatch1d(dispatch) = &out.commands[0] else {
            panic!("expected a 1D dispatch first");
        };
        assert_eq!(dispatch.bindings.as_ptr(), binding_pointer);
        assert_eq!(dispatch.bindings.capacity(), binding_capacity);

        let smaller = submission_with(vec![CommandV3::Present(Present {
            surface_id: 3,
            sync_interval: 0,
        })]);
        let smaller_frame = encode_submission_v3(&smaller).expect("smaller frame encodes");
        decode_submission_v3_into(&mut out, &smaller_frame).expect("smaller decode");
        assert_eq!(out, smaller);
        assert_eq!(out.commands.as_ptr(), command_pointer);
        assert_eq!(out.commands.capacity(), command_capacity);
    }

    #[test]
    fn wire_v3_indirect_and_region_commands_roundtrip_and_reject_bad_fields() {
        for command in indirect_commands() {
            assert_roundtrip(&submission_with(vec![command]));
        }

        let draw = single_frame(CommandV3::ExecuteIndirectDraw(ExecuteIndirectDraw {
            arguments: indirect_arguments(None),
        }));
        let base = COMMAND_START + 4;
        assert_decode_error(&draw, base, &[0; 8], WireError::ZeroAllocationId);
        assert_decode_error(&draw, base + 8, &[0; 4], WireError::ZeroGeneration);
        assert_decode_error(
            &draw,
            base + 16,
            &2u64.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "indirect argument offset alignment",
                code: 2,
            },
        );
        assert_decode_error(
            &draw,
            base + 24,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "indirect max command count",
            },
        );
        assert_decode_error(
            &draw,
            base + 24,
            &(MAX_INDIRECT_COMMANDS_V3 + 1).to_le_bytes(),
            WireError::TooManyCommandsV3(MAX_INDIRECT_COMMANDS_V3 as usize + 1),
        );
        // A count buffer is the all-zero triple or fully populated.
        assert_decode_error(
            &draw,
            base + 40,
            &2u32.to_le_bytes(),
            WireError::ZeroAllocationId,
        );

        let indexed = single_frame(CommandV3::ExecuteIndirectDrawIndexed(
            ExecuteIndirectDrawIndexed {
                arguments: indirect_arguments(Some(IndirectCountBuffer {
                    allocation_id: 12,
                    generation: 2,
                    offset: 64,
                })),
            },
        ));
        assert_decode_error(&indexed, base + 40, &[0; 4], WireError::ZeroGeneration);
        assert_decode_error(&indexed, base + 32, &[0; 8], WireError::ZeroAllocationId);
        assert_decode_error(
            &indexed,
            base + 48,
            &2u64.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "indirect count offset alignment",
                code: 2,
            },
        );

        let mesh = single_frame(CommandV3::ExecuteIndirectDispatchMesh(
            ExecuteIndirectDispatchMesh {
                pipeline_id: 10,
                arguments: indirect_arguments(None),
            },
        ));
        assert_decode_error(&mesh, base, &[0; 8], WireError::ZeroPipelineId);
        assert_decode_error(&mesh, base + 8, &[0; 8], WireError::ZeroAllocationId);

        let dispatch = single_frame(CommandV3::ExecuteIndirectDispatch(
            ExecuteIndirectDispatch {
                pipeline_id: 11,
                arguments: indirect_arguments(None),
                threads_per_group: [8, 4, 2],
            },
        ));
        assert_decode_error(&dispatch, base + 64, &0u32.to_le_bytes(), WireError::ZeroThreads);
        assert_decode_error(&dispatch, base, &[0; 8], WireError::ZeroPipelineId);

        let region = single_frame(CommandV3::CopyTextureRegion(CopyTextureRegion {
            src: subresource(9, 0, 1),
            dst: subresource(10, 1, 2),
            src_origin: [8, 16, 0],
            dst_origin: [24, 32, 0],
            size: [32, 16, 1],
        }));
        assert_decode_error(
            &region,
            base + 56,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "copy region width",
            },
        );
        assert_decode_error(
            &region,
            base + 60,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "copy region height",
            },
        );
        assert_decode_error(
            &region,
            base + 64,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "copy region depth",
            },
        );
        assert_decode_error(&region, base, &[0; 8], WireError::ZeroAllocationId);
        assert_decode_error(
            &region,
            base + 32,
            &u32::MAX.to_le_bytes(),
            WireError::OffsetOverflow,
        );

        let resolve = single_frame(CommandV3::ResolveSubresource(ResolveSubresource {
            src: subresource(11, 0, 0),
            dst: subresource(12, 0, 0),
        }));
        assert_decode_error(&resolve, base, &[0; 8], WireError::ZeroAllocationId);
        assert_decode_error(&resolve, base + 16, &[0; 8], WireError::ZeroAllocationId);

        // The writer enforces the same rules before producing a frame.
        assert_encode_error(
            CommandV3::ExecuteIndirectDrawIndexed(ExecuteIndirectDrawIndexed {
                arguments: IndirectArguments {
                    offset: 6,
                    ..indirect_arguments(None)
                },
            }),
            WireError::InvalidEnumCode {
                field: "indirect argument offset alignment",
                code: 2,
            },
        );
        assert_encode_error(
            CommandV3::CopyTextureRegion(CopyTextureRegion {
                src: subresource(9, 0, 1),
                dst: subresource(10, 1, 2),
                src_origin: [8, 16, 0],
                dst_origin: [24, 32, 0],
                size: [0, 16, 1],
            }),
            WireError::ZeroExtent {
                field: "copy region width",
            },
        );
        assert_encode_error(
            CommandV3::ExecuteIndirectDispatch(ExecuteIndirectDispatch {
                pipeline_id: 11,
                arguments: indirect_arguments(None),
                threads_per_group: [8, 0, 2],
            }),
            WireError::ZeroThreads,
        );
    }

    #[test]
    fn wire_v3_new_opcodes_reject_every_truncated_prefix() {
        for command in indirect_commands() {
            let frame = single_frame(command);
            for length in 0..frame.len() {
                assert!(
                    decode_submission_v3(&frame[..length]).is_err(),
                    "prefix {length} of a new opcode must not decode"
                );
            }
            assert!(decode_submission_v3(&frame).is_ok());
        }
    }

    #[test]
    fn wire_v3_and_v2_versions_are_isolated() {
        let v2_submission = wire::Submission {
            sequence: 1,
            commands: vec![wire::Command::SetPipeline(wire::SetPipeline {
                pipeline_id: 1,
            })],
        };
        let v2_frame = wire::encode_submission(&v2_submission).expect("version 2 frame encodes");
        assert_eq!(
            wire::decode_submission(&v2_frame).expect("version 2 frame decodes"),
            v2_submission
        );
        assert_eq!(
            decode_submission_v3(&v2_frame),
            Err(WireError::UnsupportedVersion(2))
        );

        let v3_frame = encoded_rich();
        assert_eq!(
            decode_submission_v3(&v3_frame).expect("version 3 frame decodes"),
            rich_submission()
        );
        assert_eq!(
            wire::decode_submission(&v3_frame),
            Err(WireError::UnsupportedVersion(3))
        );
    }

    #[test]
    fn wire_v3_empty_submission_roundtrips() {
        let empty = SubmissionV3 {
            sequence: 1,
            list: Some(identity()),
            commands: Vec::new(),
        };
        assert_roundtrip(&empty);
        let frame = encode_submission_v3(&empty).expect("empty frame encodes");
        assert_eq!(frame.len(), FRAME_HEADER_LEN + LIST_BLOCK_LEN);
    }

    fn blend_attachment(enable: bool) -> BlendAttachment {
        BlendAttachment {
            enable,
            src_color: BlendFactor::SrcAlpha,
            dst_color: BlendFactor::InvSrcAlpha,
            color_op: BlendOperation::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
            alpha_op: BlendOperation::Add,
            write_mask: 0xf,
        }
    }

    fn blend_state(count: usize) -> BlendState {
        BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: (0..count).map(|_| blend_attachment(true)).collect(),
        }
    }

    fn depth_stencil_state() -> DepthStencilState {
        DepthStencilState {
            depth_enable: true,
            depth_write: true,
            depth_func: CompareFunction::Less,
            stencil_enable: false,
            front: empty_stencil_face(),
            back: empty_stencil_face(),
        }
    }

    fn rasterizer_state() -> RasterizerState {
        RasterizerState {
            fill: FillMode::Solid,
            cull: CullMode::Back,
            front_face: FrontFaceWinding::CounterClockwise,
            depth_bias: 0,
            slope_scaled_depth_bias: 0.0,
            depth_bias_clamp: 0.0,
            depth_clip: true,
        }
    }

    fn clear_rtv() -> ClearRenderTargetView {
        ClearRenderTargetView {
            slot: 0,
            rect: None,
            color: [0.0, 0.0, 1.0, 1.0],
        }
    }

    fn clear_dsv() -> ClearDepthStencilView {
        ClearDepthStencilView {
            depth: Some(1.0),
            stencil: None,
            rect: None,
        }
    }

    fn pitched_copy() -> CopyBufferToTexture {
        CopyBufferToTexture {
            allocation_id: 21,
            generation: 3,
            destination_allocation_id: 22,
            destination_generation: 4,
            offset: 0,
            bytes_per_row: 256,
            bytes_per_image: 0,
            extent: [4, 4, 1],
            mip_level: 0,
            slice: 0,
            origin: [0, 0, 0],
        }
    }

    fn new_state_commands() -> Vec<CommandV3> {
        vec![
            CommandV3::SetBlendState(blend_state(2)),
            CommandV3::SetDepthStencilState(depth_stencil_state()),
            CommandV3::SetRasterizerState(rasterizer_state()),
            CommandV3::SetStencilRef(StencilRef { value: 9 }),
            CommandV3::SetBlendFactor(BlendFactorValue {
                value: [0.5, 0.25, 0.125, 1.0],
            }),
            CommandV3::ClearRenderTargetView(clear_rtv()),
            CommandV3::ClearDepthStencilView(clear_dsv()),
            CommandV3::CopyBufferToTexture(pitched_copy()),
        ]
    }

    #[test]
    fn wire_v3_render_state_commands_roundtrip_and_reject_bad_fields() {
        for command in new_state_commands() {
            assert_roundtrip(&submission_with(vec![command]));
        }

        assert_encode_error(
            CommandV3::SetBlendState(blend_state(0)),
            WireError::ZeroBlendAttachments,
        );
        assert_encode_error(
            CommandV3::SetBlendState(blend_state(9)),
            WireError::TooManyBlendAttachments(9),
        );
        // 8 entries are the maximum and must round-trip.
        assert_roundtrip(&submission_with(vec![CommandV3::SetBlendState(blend_state(
            8,
        ))]));

        let blend = single_frame(CommandV3::SetBlendState(blend_state(1)));
        assert_decode_error(
            &blend,
            COMMAND_START + 13,
            &12u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "blend factor",
                code: 12,
            },
        );
        assert_decode_error(
            &blend,
            COMMAND_START + 15,
            &9u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "blend operation",
                code: 9,
            },
        );
        assert_decode_error(
            &blend,
            COMMAND_START + 12,
            &2u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "blend enable",
                code: 2,
            },
        );
        assert_decode_error(
            &blend,
            COMMAND_START + 6,
            &2u16.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "independent blend",
                code: 2,
            },
        );
        assert_decode_error(&blend, COMMAND_START + 10, &[1, 0], WireError::NonZeroReserved);

        let depth = single_frame(CommandV3::SetDepthStencilState(depth_stencil_state()));
        assert_decode_error(
            &depth,
            COMMAND_START + 6,
            &9u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "comparison function",
                code: 9,
            },
        );
        assert_decode_error(
            &depth,
            COMMAND_START + 8,
            &9u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "stencil operation",
                code: 9,
            },
        );
        assert_decode_error(
            &depth,
            COMMAND_START + 4,
            &2u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "depth enable",
                code: 2,
            },
        );

        let raster = single_frame(CommandV3::SetRasterizerState(rasterizer_state()));
        assert_decode_error(
            &raster,
            COMMAND_START + 4,
            &1u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "fill mode",
                code: 1,
            },
        );
        assert_decode_error(
            &raster,
            COMMAND_START + 5,
            &4u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "cull mode",
                code: 4,
            },
        );
        assert_decode_error(
            &raster,
            COMMAND_START + 6,
            &2u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "front-face winding",
                code: 2,
            },
        );
        assert_decode_error(
            &raster,
            COMMAND_START + 7,
            &2u8.to_le_bytes(),
            WireError::InvalidEnumCode {
                field: "depth clip enable",
                code: 2,
            },
        );

        let clear = single_frame(CommandV3::ClearRenderTargetView(clear_rtv()));
        assert_decode_error(
            &clear,
            COMMAND_START + 6,
            &2u16.to_le_bytes(),
            WireError::UnsupportedFlags(2),
        );
        assert_encode_error(
            CommandV3::ClearRenderTargetView(ClearRenderTargetView {
                slot: 8,
                ..clear_rtv()
            }),
            WireError::TooManyColorAttachments(9),
        );
        let mut rect_clear = clear_rtv();
        rect_clear.rect = Some(ScissorRect {
            x: 0,
            y: 0,
            width: 0,
            height: 1,
        });
        assert_encode_error(
            CommandV3::ClearRenderTargetView(rect_clear),
            WireError::ZeroExtent {
                field: "clear rect width",
            },
        );
        assert_decode_error(
            &clear,
            COMMAND_START + 24,
            &4u32.to_le_bytes(),
            WireError::NonZeroReserved,
        );

        let dsv = single_frame(CommandV3::ClearDepthStencilView(clear_dsv()));
        assert_encode_error(
            CommandV3::ClearDepthStencilView(ClearDepthStencilView {
                depth: None,
                stencil: None,
                rect: None,
            }),
            WireError::EmptyClear,
        );
        assert_decode_error(
            &dsv,
            COMMAND_START + 4,
            &8u16.to_le_bytes(),
            WireError::UnsupportedFlags(8),
        );
        // A stencil value without the stencil plane flag is a reserved word.
        assert_decode_error(
            &dsv,
            COMMAND_START + 12,
            &5u32.to_le_bytes(),
            WireError::NonZeroReserved,
        );

        let copy = single_frame(CommandV3::CopyBufferToTexture(pitched_copy()));
        let mut zero_row = pitched_copy();
        zero_row.bytes_per_row = 0;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(zero_row),
            WireError::ZeroExtent {
                field: "copy source row pitch",
            },
        );
        let mut misaligned = pitched_copy();
        misaligned.offset = 2;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(misaligned),
            WireError::InvalidEnumCode {
                field: "copy source offset alignment",
                code: 2,
            },
        );
        let mut volume = pitched_copy();
        volume.extent = [4, 4, 2];
        volume.bytes_per_image = 0;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(volume),
            WireError::ZeroExtent {
                field: "copy source image pitch",
            },
        );
        volume.bytes_per_image = 512;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(volume),
            WireError::OffsetOverflow,
        );
        volume.bytes_per_image = 1024;
        assert_roundtrip(&submission_with(vec![CommandV3::CopyBufferToTexture(
            volume,
        )]));
        let mut zero_alloc = pitched_copy();
        zero_alloc.allocation_id = 0;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(zero_alloc),
            WireError::ZeroAllocationId,
        );
        let mut zero_destination = pitched_copy();
        zero_destination.destination_allocation_id = 0;
        assert_encode_error(
            CommandV3::CopyBufferToTexture(zero_destination),
            WireError::ZeroAllocationId,
        );
        assert_decode_error(
            &copy,
            COMMAND_START + 4,
            &[0; 8],
            WireError::ZeroAllocationId,
        );
        assert_decode_error(
            &copy,
            COMMAND_START + 16,
            &[0; 8],
            WireError::ZeroAllocationId,
        );
        assert_decode_error(
            &copy,
            COMMAND_START + 36,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "copy source row pitch",
            },
        );
        assert_decode_error(
            &copy,
            COMMAND_START + 44,
            &0u32.to_le_bytes(),
            WireError::ZeroExtent {
                field: "copy region width",
            },
        );
    }

    #[test]
    fn wire_v3_render_state_opcodes_reject_every_truncated_prefix() {
        for command in new_state_commands() {
            let frame = single_frame(command);
            for length in 0..frame.len() {
                assert!(
                    decode_submission_v3(&frame[..length]).is_err(),
                    "prefix {length} of a render-state opcode must not decode"
                );
            }
            assert!(decode_submission_v3(&frame).is_ok());
        }
    }
}
