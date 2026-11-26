//! Versioned cross-process submission wire format for ordered command lists.
//!
//! This is the ABI between the Win64 Wine client and the ARM64 renderer host.
//! It frames command metadata only. File registration, socket or shared-memory
//! transport, allocation ownership and GPU lifetime live in separate layers.
//!
//! # Version 2 frame layout
//!
//! All integers are fixed-width little-endian. A frame is a 20-byte header
//! followed by exactly `payload_len` payload bytes. Short payloads are
//! rejected as truncated and trailing bytes are rejected, so a frame has a
//! single unambiguous encoding. The codec never casts structs or native
//! integers to bytes.
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 4 | magic `VMWF` |
//! | 4 | 2 | protocol version (2) |
//! | 6 | 2 | frame kind (1 = submission, 2 = reply) |
//! | 8 | 4 | submission sequence (never zero) |
//! | 12 | 2 | command count (submissions only, at most [`MAX_COMMANDS`]) |
//! | 14 | 2 | reserved, must be zero |
//! | 16 | 4 | payload length in bytes |
//!
//! A submission payload starts with a 12-byte list identity block:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | list ID (0 = no list identity) |
//! | 8 | 4 | list generation (0 = no list identity) |
//!
//! Both words are zero for a submission that does not name a recorded command
//! list, or both are nonzero for one that does. A half-populated identity is
//! rejected. The identity is exposed through [`encode_submission_with_list`]
//! and [`decode_submission_with_list`]. [`Submission`] itself deliberately
//! keeps its original two public fields so existing callers that build
//! `Submission` literals continue to compile unchanged; the identity is frame
//! metadata, not command metadata.
//!
//! The rest of the payload is an ordered list of commands. Every command is a
//! 4-byte `(opcode: u16, flags: u16)` prefix with `flags` zero, followed by the
//! opcode payload. Opcodes 1 and 2 keep the exact version 1 command payload;
//! version 2 only prepends the list identity block to the frame.
//!
//! The opcode 1 command layout is frozen:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (1 = dispatch 1D) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//! | 12 | 4 | thread count (never zero) |
//! | 16 | 4 | threads per threadgroup (never zero) |
//! | 20 | 2 | binding count (at most [`MAX_BINDINGS_PER_DISPATCH`]) |
//! | 22 | 2 | reserved, must be zero |
//! | 24 | 32 x N | bound buffer ranges, in slot order |
//!
//! The opcode 2 command carries the same handles and bindings but an explicit
//! D3D-style threadgroup grid:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (2 = dispatch 3D groups) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pre-created pipeline ID (never zero) |
//! | 12 | 4 | threadgroup count X (never zero) |
//! | 16 | 4 | threadgroup count Y (never zero) |
//! | 20 | 4 | threadgroup count Z (never zero) |
//! | 24 | 4 | threads per threadgroup X (never zero) |
//! | 28 | 4 | threads per threadgroup Y (never zero) |
//! | 32 | 4 | threads per threadgroup Z (never zero) |
//! | 36 | 2 | binding count (at most [`MAX_BINDINGS_PER_DISPATCH`]) |
//! | 38 | 2 | reserved, must be zero |
//! | 40 | 32 x N | bound buffer ranges, in slot order |
//!
//! The group count and local dimension products must each fit in `u64`, and
//! their combined total thread count must also fit; overflow is rejected by
//! [`WireError::ThreadCountOverflow`] before execution. The native host still
//! rejects a local threadgroup whose product exceeds the pipeline's hardware
//! limit before it creates a command buffer.
//!
//! ## Graphics and state opcodes
//!
//! Opcode 3 is an indexed draw. `base_vertex` is a signed D3D base vertex:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (3 = draw indexed) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pipeline ID (never zero) |
//! | 12 | 4 | index count (never zero) |
//! | 16 | 4 | instance count (never zero) |
//! | 20 | 4 | start index |
//! | 24 | 4 | base vertex (`i32`) |
//! | 28 | 4 | start instance |
//! | 32 | 8 | list ID (never zero) |
//! | 40 | 4 | list generation (never zero) |
//! | 44 | 2 | binding count (at most [`MAX_BINDINGS_PER_DISPATCH`]) |
//! | 46 | 2 | reserved, must be zero |
//! | 48 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 4 is a non-indexed draw:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (4 = draw) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pipeline ID (never zero) |
//! | 12 | 4 | vertex count (never zero) |
//! | 16 | 4 | instance count (never zero) |
//! | 20 | 4 | start vertex |
//! | 24 | 4 | start instance |
//! | 28 | 8 | list ID (never zero) |
//! | 36 | 4 | list generation (never zero) |
//! | 40 | 2 | binding count (at most [`MAX_BINDINGS_PER_DISPATCH`]) |
//! | 42 | 2 | reserved, must be zero |
//! | 44 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 5 is a mesh dispatch (`DispatchMesh`):
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (5 = dispatch mesh) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pipeline ID (never zero) |
//! | 12 | 4 x 3 | threadgroup counts X, Y, Z (each never zero) |
//! | 24 | 2 | binding count (at most [`MAX_BINDINGS_PER_DISPATCH`]) |
//! | 26 | 2 | reserved, must be zero |
//! | 28 | 32 x N | bound buffer ranges, in slot order |
//!
//! Opcode 6 is an ordered resource-barrier batch. The entry count is capped by
//! [`MAX_BARRIERS_PER_COMMAND`] and must be nonzero:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (6 = resource barrier) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | entry count (1 ..= [`MAX_BARRIERS_PER_COMMAND`]) |
//! | 6 | 2 | reserved, must be zero |
//! | 8 | 40 x N | barrier entries |
//!
//! Each 40-byte barrier entry is `allocation_id: u64`, `generation: u32`,
//! reserved `u32`, `offset: u64`, `length: u64`, `before_state: u32`,
//! `after_state: u32`. States are carried verbatim as D3D12 transition state
//! words; the codec does not interpret them. Allocation, generation and length
//! follow the same nonzero and overflow rules as bound ranges.
//!
//! Opcode 7 sets the active pipeline:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (7 = set pipeline) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 8 | pipeline ID (never zero) |
//!
//! Opcode 8 carries inline root constants (at most
//! [`MAX_ROOT_CONSTANT_BYTES`] and never empty):
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (8 = root constants) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | root parameter slot |
//! | 6 | 2 | byte length (1 ..= [`MAX_ROOT_CONSTANT_BYTES`]) |
//! | 8 | N | constant bytes |
//!
//! Opcode 9 binds a descriptor table (at most [`MAX_DESCRIPTOR_BINDINGS`] and
//! never empty):
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 2 | opcode (9 = descriptor table) |
//! | 2 | 2 | flags, must be zero |
//! | 4 | 2 | descriptor set |
//! | 6 | 2 | base register |
//! | 8 | 2 | binding count (1 ..= [`MAX_DESCRIPTOR_BINDINGS`]) |
//! | 10 | 2 | reserved, must be zero |
//! | 12 | 32 x N | bound buffer ranges, in register order |
//!
//! Each bound range is an allocation ID, generation, reserved word, byte
//! offset and byte length. IDs, generations and lengths must be nonzero;
//! `offset + length` is checked against overflow and `length` is capped by
//! [`MAX_RESOURCE_LEN`]. Ranges contain no raw guest virtual addresses.
//!
//! A decoder that predates an opcode rejects it as
//! [`WireError::UnknownOpcode`]; it never guesses at a newer payload layout.
//!
//! A reply payload is a status word plus a reserved word: 0 = ok,
//! 1 = unknown handle (pipeline or allocation/generation), 2 = execution
//! failed. The reply sequence echoes the submission sequence. Resolver and
//! GPU execution are outside this codec.
//!
//! # Ordering and atomicity
//!
//! Submissions carry an explicit `sequence`; the codec rejects zero. A future
//! session layer owns enforcing strictly increasing sequences starting at 1
//! across frames; a writer cannot be validated in isolation.
//!
//! Decoding is atomic: commands accumulate in a fresh vector and any error
//! discards the whole result. Count bounds are checked before reserving
//! storage, so an oversized count is rejected before the allocator runs.
//!
//! # Handles and lifetime
//!
//! The host resolver maps allocation IDs and generations to registered
//! file-backed ranges. It must validate the generation and the requested
//! bounds against the allocation's current length. It must also keep every
//! borrowed buffer range and every pipeline alive until all GPU work from the
//! submission has completed, including after the guest frees pages that
//! remain referenced. Registration, transport and completion signaling are
//! intentionally not part of this module.

use std::fmt;

/// Wire protocol version carried in every frame.
pub const PROTOCOL_VERSION: u16 = 2;

/// Frame magic. The version field, not the magic, selects the protocol.
pub const MAGIC: [u8; 4] = *b"VMWF";

/// Maximum number of commands in one submission.
pub const MAX_COMMANDS: usize = 64;

/// Maximum number of buffer bindings in one dispatch, draw or table.
pub const MAX_BINDINGS_PER_DISPATCH: usize = 16;

/// Maximum byte length of one bound buffer range.
pub const MAX_RESOURCE_LEN: u64 = 1 << 32;

/// Maximum number of entries in one resource-barrier command.
pub const MAX_BARRIERS_PER_COMMAND: usize = 16;

/// Maximum number of bindings in one descriptor table.
pub const MAX_DESCRIPTOR_BINDINGS: usize = 16;

/// Maximum number of inline root-constant bytes in one command.
pub const MAX_ROOT_CONSTANT_BYTES: usize = 256;

const KIND_SUBMISSION: u16 = 1;
const KIND_REPLY: u16 = 2;

const OPCODE_DISPATCH_1D: u16 = 1;
const OPCODE_DISPATCH_3D: u16 = 2;
const OPCODE_DRAW_INDEXED: u16 = 3;
const OPCODE_DRAW: u16 = 4;
const OPCODE_DISPATCH_MESH: u16 = 5;
const OPCODE_RESOURCE_BARRIER: u16 = 6;
const OPCODE_SET_PIPELINE: u16 = 7;
const OPCODE_ROOT_CONSTANTS: u16 = 8;
const OPCODE_DESCRIPTOR_TABLE: u16 = 9;

const FRAME_HEADER_LEN: usize = 20;
const LIST_BLOCK_LEN: usize = 12;
const COMMAND_HEADER_LEN: usize = 24;
const DISPATCH_3D_HEADER_LEN: usize = 40;
const DRAW_INDEXED_HEADER_LEN: usize = 48;
const DRAW_HEADER_LEN: usize = 44;
const DISPATCH_MESH_HEADER_LEN: usize = 28;
const BARRIER_HEADER_LEN: usize = 8;
const BARRIER_ENTRY_LEN: usize = 40;
const SET_PIPELINE_HEADER_LEN: usize = 12;
const ROOT_CONSTANTS_HEADER_LEN: usize = 8;
const DESCRIPTOR_TABLE_HEADER_LEN: usize = 12;
const BINDING_LEN: usize = 32;
const REPLY_PAYLOAD_LEN: usize = 4;

/// An ordered, atomic batch of commands in one frame.
///
/// The optional list identity is frame metadata carried by
/// [`encode_submission_with_list`] and returned by
/// [`decode_submission_with_list`]. It is not a field of this struct so that
/// callers which build `Submission { sequence, commands }` literals keep
/// compiling with the version 2 codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    /// Submission sequence. Must be nonzero; the session layer enforces that
    /// successive values increase from 1.
    pub sequence: u32,
    /// Commands in execution order.
    pub commands: Vec<Command>,
}

/// Identity of one recorded command list carried alongside a submission.
///
/// A list is identified by `(list_id, list_generation)`; every submitted list
/// records both so a replay or retry is distinguishable from new work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListIdentity {
    /// Host-visible command-list identifier, never zero.
    pub list_id: u64,
    /// Generation of that list, never zero.
    pub list_generation: u32,
}

impl ListIdentity {
    /// Builds a checked list identity.
    pub fn new(list_id: u64, list_generation: u32) -> Result<Self, WireError> {
        let identity = Self {
            list_id,
            list_generation,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), WireError> {
        if self.list_id == 0 {
            return Err(WireError::ZeroListId);
        }
        if self.list_generation == 0 {
            return Err(WireError::ZeroListGeneration);
        }
        Ok(())
    }
}

/// A command decoded from a submission payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// A pre-created compute pipeline dispatched over a 1D grid.
    Dispatch1d(Dispatch1d),
    /// A pre-created compute pipeline dispatched over an explicit 3D
    /// threadgroup grid.
    Dispatch3d(Dispatch3d),
    /// An indexed draw against a pre-created graphics pipeline.
    DrawIndexed(DrawIndexed),
    /// A non-indexed draw against a pre-created graphics pipeline.
    Draw(Draw),
    /// A mesh-shader dispatch over an explicit threadgroup grid.
    DispatchMesh(DispatchMesh),
    /// An ordered batch of resource transitions.
    ResourceBarrier(ResourceBarrier),
    /// Selects the pipeline used by later draw and dispatch commands.
    SetPipeline(SetPipeline),
    /// Writes inline root constants.
    RootConstants(RootConstants),
    /// Binds a descriptor table.
    DescriptorTable(DescriptorTable),
}

/// One ordered 1D compute dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch1d {
    /// Host pipeline handle, created before the submission.
    pub pipeline_id: u64,
    /// Total threads in the 1D grid.
    pub threads: u32,
    /// Threads per threadgroup.
    pub threads_per_group: u32,
    /// Bound buffer ranges in `[[buffer(0)]]`, `[[buffer(1)]]`, ... order.
    pub bindings: Vec<BufferBinding>,
}

/// One ordered 3D compute dispatch with an exact D3D threadgroup grid.
///
/// The native host maps `group_counts` to
/// `dispatchThreadgroups:threadsPerThreadgroup:` and `threads_per_group` to
/// the local `MTLSize`, preserving non-divisible and non-cubic grids exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch3d {
    /// Host pipeline handle, created before the submission.
    pub pipeline_id: u64,
    /// D3D `Dispatch(x, y, z)` threadgroup counts; every dimension is nonzero.
    pub group_counts: [u32; 3],
    /// Local threads per threadgroup; every dimension is nonzero.
    pub threads_per_group: [u32; 3],
    /// Bound buffer ranges in `[[buffer(0)]]`, `[[buffer(1)]]`, ... order.
    pub bindings: Vec<BufferBinding>,
}

impl Dispatch3d {
    /// Returns the checked product of the three group counts.
    pub fn checked_group_count(&self) -> Option<u64> {
        checked_dimension_product(self.group_counts)
    }

    /// Returns the checked product of the three local thread dimensions.
    pub fn checked_threads_per_group(&self) -> Option<u64> {
        checked_dimension_product(self.threads_per_group)
    }

    /// Returns the checked total thread count for the whole grid.
    pub fn checked_total_threads(&self) -> Option<u64> {
        self.checked_group_count()?
            .checked_mul(self.checked_threads_per_group()?)
    }
}

/// One indexed draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawIndexed {
    /// Host graphics pipeline handle, created before the submission.
    pub pipeline_id: u64,
    /// Index count; must be nonzero.
    pub index_count: u32,
    /// Instance count; must be nonzero.
    pub instance_count: u32,
    /// First index value used by the draw.
    pub start_index: u32,
    /// Signed value added to every index, as in D3D12 `DrawIndexedInstanced`.
    pub base_vertex: i32,
    /// First instance value used by the draw.
    pub start_instance: u32,
    /// Identity of the command list that recorded this draw.
    pub list_id: u64,
    /// Generation of the recording list.
    pub list_generation: u32,
    /// Bound buffer ranges in slot order.
    pub bindings: Vec<BufferBinding>,
}

impl DrawIndexed {
    /// Returns the typed identity of the recording list.
    pub fn list_identity(&self) -> ListIdentity {
        ListIdentity {
            list_id: self.list_id,
            list_generation: self.list_generation,
        }
    }
}

/// One non-indexed draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draw {
    /// Host graphics pipeline handle, created before the submission.
    pub pipeline_id: u64,
    /// Vertex count; must be nonzero.
    pub vertex_count: u32,
    /// Instance count; must be nonzero.
    pub instance_count: u32,
    /// First vertex value used by the draw.
    pub start_vertex: u32,
    /// First instance value used by the draw.
    pub start_instance: u32,
    /// Identity of the command list that recorded this draw.
    pub list_id: u64,
    /// Generation of the recording list.
    pub list_generation: u32,
    /// Bound buffer ranges in slot order.
    pub bindings: Vec<BufferBinding>,
}

impl Draw {
    /// Returns the typed identity of the recording list.
    pub fn list_identity(&self) -> ListIdentity {
        ListIdentity {
            list_id: self.list_id,
            list_generation: self.list_generation,
        }
    }
}

/// One mesh-shader dispatch with an exact D3D threadgroup grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchMesh {
    /// Host mesh pipeline handle, created before the submission.
    pub pipeline_id: u64,
    /// D3D `DispatchMesh(x, y, z)` threadgroup counts; every dimension is
    /// nonzero.
    pub group_counts: [u32; 3],
    /// Bound buffer ranges in slot order.
    pub bindings: Vec<BufferBinding>,
}

impl DispatchMesh {
    /// Returns the checked product of the three group counts.
    pub fn checked_group_count(&self) -> Option<u64> {
        checked_dimension_product(self.group_counts)
    }
}

/// An ordered batch of resource transitions executed as one command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBarrier {
    /// Transitions in execution order; never empty.
    pub entries: Vec<BarrierEntry>,
}

/// One transition of a borrowed allocation range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarrierEntry {
    /// Allocation registry key, not a guest virtual address.
    pub allocation_id: u64,
    /// Allocation generation, never zero.
    pub generation: u32,
    /// Byte offset within the allocation.
    pub offset: u64,
    /// Byte length of the range, never zero.
    pub length: u64,
    /// D3D12 state before the transition, carried verbatim.
    pub before_state: u32,
    /// D3D12 state after the transition, carried verbatim.
    pub after_state: u32,
}

impl BarrierEntry {
    /// Returns `offset + length`, or `None` when it overflows `u64`.
    pub fn checked_end(&self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

/// Selects the pipeline used by subsequent draw and dispatch commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetPipeline {
    /// Host pipeline handle, created before the submission.
    pub pipeline_id: u64,
}

/// Inline root constants for one root parameter slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootConstants {
    /// Root parameter slot (`0 ..= 63` in D3D12).
    pub slot: u16,
    /// Constant bytes, 1 ..= [`MAX_ROOT_CONSTANT_BYTES`].
    pub bytes: Vec<u8>,
}

/// A descriptor table binding at a base register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorTable {
    /// Descriptor set index.
    pub set: u16,
    /// Base register (`t0`, `b0`, `u0`, `s0`, ...) of the table.
    pub base_register: u16,
    /// Bound buffer ranges in register order; never empty.
    pub bindings: Vec<BufferBinding>,
}

fn checked_dimension_product(dimensions: [u32; 3]) -> Option<u64> {
    dimensions.into_iter().try_fold(1u64, |product, dimension| {
        product.checked_mul(u64::from(dimension))
    })
}

/// One borrowed allocation range bound to a buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferBinding {
    /// Allocation registry key, not a guest virtual address.
    pub allocation_id: u64,
    /// Allocation generation, never zero.
    pub generation: u32,
    /// Byte offset within the allocation.
    pub offset: u64,
    /// Byte length of the range, never zero.
    pub length: u64,
}

impl BufferBinding {
    /// Returns `offset + length`, or `None` when it overflows `u64`.
    pub fn checked_end(&self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

/// Reply to one submission, echoing its sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reply {
    /// Sequence of the submission this reply completes.
    pub sequence: u32,
    /// Execution outcome.
    pub status: ReplyStatus,
}

/// Distinguishable reply statuses on the version 2 wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyStatus {
    /// All commands were submitted and completed successfully.
    Ok,
    /// A referenced pipeline, allocation or generation was not registered.
    UnknownHandle,
    /// The host accepted the handles but GPU execution failed.
    ExecutionFailed,
}

impl ReplyStatus {
    const fn code(self) -> u16 {
        match self {
            Self::Ok => 0,
            Self::UnknownHandle => 1,
            Self::ExecutionFailed => 2,
        }
    }

    fn from_code(code: u16) -> Result<Self, WireError> {
        match code {
            0 => Ok(Self::Ok),
            1 => Ok(Self::UnknownHandle),
            2 => Ok(Self::ExecutionFailed),
            other => Err(WireError::UnsupportedStatus(other)),
        }
    }
}

/// Reasons a frame cannot be encoded or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    /// The byte slice ends before the fixed or declared payload does.
    Truncated,
    /// The frame does not start with [`MAGIC`].
    BadMagic,
    /// The frame version is not [`PROTOCOL_VERSION`].
    UnsupportedVersion(u16),
    /// A submission was decoded as a reply or vice versa.
    UnexpectedKind(u16),
    /// Submission sequence zero is reserved and forbidden.
    ZeroSequence,
    /// A reserved header, command or binding word was nonzero.
    NonZeroReserved,
    /// A command flags word was nonzero.
    UnsupportedFlags(u16),
    /// The submission contains more than [`MAX_COMMANDS`] commands.
    TooManyCommands(usize),
    /// A dispatch or draw contains more than [`MAX_BINDINGS_PER_DISPATCH`]
    /// bindings.
    TooManyBindings(usize),
    /// A resource-barrier command contains more than
    /// [`MAX_BARRIERS_PER_COMMAND`] entries.
    TooManyBarriers(usize),
    /// A descriptor table contains more than [`MAX_DESCRIPTOR_BINDINGS`]
    /// bindings.
    TooManyDescriptorBindings(usize),
    /// A reply unexpectedly carries commands.
    UnexpectedCommandCount(u16),
    /// The command opcode is not defined by this protocol version.
    UnknownOpcode(u16),
    /// Pipeline ID zero is not a valid pre-created handle.
    ZeroPipelineId,
    /// Thread or group counts were zero where the command requires work.
    ZeroThreads,
    /// A 3D group count, local dimension or total thread product overflowed
    /// `u64`.
    ThreadCountOverflow,
    /// An indexed draw carried a zero index count.
    ZeroIndexCount,
    /// A non-indexed draw carried a zero vertex count.
    ZeroVertexCount,
    /// A draw carried a zero instance count.
    ZeroInstanceCount,
    /// A resource-barrier command carried no entries.
    ZeroBarrierCount,
    /// A descriptor table carried no bindings.
    ZeroDescriptorCount,
    /// A root-constant command carried no bytes.
    ZeroRootConstants,
    /// A root-constant command exceeded [`MAX_ROOT_CONSTANT_BYTES`].
    RootConstantsTooLarge(usize),
    /// A command or frame list ID was zero.
    ZeroListId,
    /// A command or frame list generation was zero.
    ZeroListGeneration,
    /// Allocation ID zero is not a valid registry key.
    ZeroAllocationId,
    /// Generation zero does not identify a live allocation.
    ZeroGeneration,
    /// A bound range length of zero is not meaningful.
    ZeroLength,
    /// A bound range exceeds [`MAX_RESOURCE_LEN`].
    ResourceTooLarge(u64),
    /// `offset + length` overflows `u64`.
    OffsetOverflow,
    /// A reply status code is not defined by this protocol version.
    UnsupportedStatus(u16),
    /// A fixed-size payload did not have its exact declared size.
    LengthMismatch,
    /// Bytes follow the declared payload.
    TrailingBytes,
    /// The frame does not fit the 32-bit payload length field.
    FrameTooLarge,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated frame"),
            Self::BadMagic => write!(f, "bad frame magic"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported wire version {version}"),
            Self::UnexpectedKind(kind) => write!(f, "unexpected frame kind {kind}"),
            Self::ZeroSequence => write!(f, "submission sequence must be nonzero"),
            Self::NonZeroReserved => write!(f, "reserved field must be zero"),
            Self::UnsupportedFlags(flags) => write!(f, "unsupported command flags 0x{flags:04x}"),
            Self::TooManyCommands(count) => {
                write!(f, "{count} commands exceeds limit {MAX_COMMANDS}")
            }
            Self::TooManyBindings(count) => {
                write!(
                    f,
                    "{count} bindings exceeds limit {MAX_BINDINGS_PER_DISPATCH}"
                )
            }
            Self::TooManyBarriers(count) => {
                write!(
                    f,
                    "{count} barrier entries exceeds limit {MAX_BARRIERS_PER_COMMAND}"
                )
            }
            Self::TooManyDescriptorBindings(count) => {
                write!(
                    f,
                    "{count} descriptor bindings exceeds limit {MAX_DESCRIPTOR_BINDINGS}"
                )
            }
            Self::UnexpectedCommandCount(count) => {
                write!(f, "reply carries {count} commands")
            }
            Self::UnknownOpcode(opcode) => write!(f, "unknown command opcode {opcode}"),
            Self::ZeroPipelineId => write!(f, "pipeline ID must be nonzero"),
            Self::ZeroThreads => write!(f, "thread counts must be nonzero"),
            Self::ThreadCountOverflow => write!(f, "3D thread count product overflows u64"),
            Self::ZeroIndexCount => write!(f, "index count must be nonzero"),
            Self::ZeroVertexCount => write!(f, "vertex count must be nonzero"),
            Self::ZeroInstanceCount => write!(f, "instance count must be nonzero"),
            Self::ZeroBarrierCount => write!(f, "barrier entry count must be nonzero"),
            Self::ZeroDescriptorCount => write!(f, "descriptor binding count must be nonzero"),
            Self::ZeroRootConstants => write!(f, "root constants must not be empty"),
            Self::RootConstantsTooLarge(len) => write!(
                f,
                "root constant length {len} exceeds limit {MAX_ROOT_CONSTANT_BYTES}"
            ),
            Self::ZeroListId => write!(f, "list ID must be nonzero"),
            Self::ZeroListGeneration => write!(f, "list generation must be nonzero"),
            Self::ZeroAllocationId => write!(f, "allocation ID must be nonzero"),
            Self::ZeroGeneration => write!(f, "allocation generation must be nonzero"),
            Self::ZeroLength => write!(f, "bound range length must be nonzero"),
            Self::ResourceTooLarge(length) => {
                write!(
                    f,
                    "bound range length {length} exceeds limit {MAX_RESOURCE_LEN}"
                )
            }
            Self::OffsetOverflow => write!(f, "binding offset plus length overflows u64"),
            Self::UnsupportedStatus(status) => write!(f, "unsupported reply status {status}"),
            Self::LengthMismatch => write!(f, "payload length does not match field widths"),
            Self::TrailingBytes => write!(f, "trailing bytes after frame payload"),
            Self::FrameTooLarge => write!(f, "frame payload exceeds 32-bit length field"),
        }
    }
}

impl std::error::Error for WireError {}

/// Encodes one submission as a complete version 2 frame without a list
/// identity.
///
/// The same field rules as decoding are enforced, so a writer cannot produce
/// a frame this codec would reject. Sequence monotonicity across frames is not
/// checked here.
pub fn encode_submission(submission: &Submission) -> Result<Vec<u8>, WireError> {
    encode_submission_with_list(submission, None)
}

/// Encodes one submission as a complete version 2 frame with an optional list
/// identity.
///
/// The identity names the recorded command list the batch belongs to; it is
/// checked exactly like a per-draw list identity.
pub fn encode_submission_with_list(
    submission: &Submission,
    list: Option<ListIdentity>,
) -> Result<Vec<u8>, WireError> {
    validate_sequence(submission.sequence)?;
    if let Some(list) = list {
        list.validate()?;
    }
    if submission.commands.len() > MAX_COMMANDS {
        return Err(WireError::TooManyCommands(submission.commands.len()));
    }

    let mut payload_len = LIST_BLOCK_LEN;
    for command in &submission.commands {
        payload_len = payload_len
            .checked_add(command_encoded_len(command)?)
            .ok_or(WireError::FrameTooLarge)?;
    }
    let payload_len_field = u32::try_from(payload_len).map_err(|_| WireError::FrameTooLarge)?;
    let command_count = u16::try_from(submission.commands.len())
        .map_err(|_| WireError::TooManyCommands(submission.commands.len()))?;

    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + payload_len);
    write_frame_header(
        &mut out,
        KIND_SUBMISSION,
        submission.sequence,
        command_count,
        payload_len_field,
    );
    match list {
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

/// Decodes one complete submission frame, ignoring any list identity.
///
/// Use [`decode_submission_with_list`] when the identity matters. Returns no
/// commands on any error. Payload length, command count and binding count are
/// validated before the corresponding storage is reserved.
pub fn decode_submission(bytes: &[u8]) -> Result<Submission, WireError> {
    decode_submission_with_list(bytes).map(|(submission, _)| submission)
}

/// Decodes one complete submission frame and its optional list identity.
pub fn decode_submission_with_list(
    bytes: &[u8],
) -> Result<(Submission, Option<ListIdentity>), WireError> {
    let (header, payload) = FrameHeader::parse(bytes)?;
    if header.kind != KIND_SUBMISSION {
        return Err(WireError::UnexpectedKind(header.kind));
    }

    let command_count = usize::from(header.command_count);
    if command_count > MAX_COMMANDS {
        return Err(WireError::TooManyCommands(command_count));
    }

    let mut reader = Reader::new(payload);
    let list_id = reader.u64()?;
    let list_generation = reader.u32()?;
    let list = decode_list_identity(list_id, list_generation)?;

    let mut commands = Vec::with_capacity(command_count);
    for _ in 0..command_count {
        commands.push(read_command(&mut reader)?);
    }
    if !reader.is_empty() {
        return Err(WireError::TrailingBytes);
    }

    Ok((
        Submission {
            sequence: header.sequence,
            commands,
        },
        list,
    ))
}

/// Encodes one reply frame.
pub fn encode_reply(reply: &Reply) -> Result<Vec<u8>, WireError> {
    validate_sequence(reply.sequence)?;

    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + REPLY_PAYLOAD_LEN);
    write_frame_header(
        &mut out,
        KIND_REPLY,
        reply.sequence,
        0,
        REPLY_PAYLOAD_LEN as u32,
    );
    write_u16(&mut out, reply.status.code());
    write_u16(&mut out, 0);
    Ok(out)
}

/// Decodes one complete reply frame.
pub fn decode_reply(bytes: &[u8]) -> Result<Reply, WireError> {
    let (header, payload) = FrameHeader::parse(bytes)?;
    if header.kind != KIND_REPLY {
        return Err(WireError::UnexpectedKind(header.kind));
    }
    if header.command_count != 0 {
        return Err(WireError::UnexpectedCommandCount(header.command_count));
    }
    if payload.len() != REPLY_PAYLOAD_LEN {
        return Err(WireError::LengthMismatch);
    }

    let mut reader = Reader::new(payload);
    let status = ReplyStatus::from_code(reader.u16()?)?;
    if reader.u16()? != 0 {
        return Err(WireError::NonZeroReserved);
    }

    Ok(Reply {
        sequence: header.sequence,
        status,
    })
}

fn validate_sequence(sequence: u32) -> Result<(), WireError> {
    if sequence == 0 {
        Err(WireError::ZeroSequence)
    } else {
        Ok(())
    }
}

fn validate_pipeline(pipeline_id: u64) -> Result<(), WireError> {
    if pipeline_id == 0 {
        Err(WireError::ZeroPipelineId)
    } else {
        Ok(())
    }
}

fn validate_list(list_id: u64, list_generation: u32) -> Result<(), WireError> {
    if list_id == 0 {
        return Err(WireError::ZeroListId);
    }
    if list_generation == 0 {
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

fn validate_range(
    allocation_id: u64,
    generation: u32,
    offset: u64,
    length: u64,
) -> Result<(), WireError> {
    if allocation_id == 0 {
        return Err(WireError::ZeroAllocationId);
    }
    if generation == 0 {
        return Err(WireError::ZeroGeneration);
    }
    if length == 0 {
        return Err(WireError::ZeroLength);
    }
    if length > MAX_RESOURCE_LEN {
        return Err(WireError::ResourceTooLarge(length));
    }
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

fn dispatch_encoded_len(dispatch: &Dispatch1d) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_threads(dispatch.threads, dispatch.threads_per_group)?;
    encoded_len_with_bindings(COMMAND_HEADER_LEN, &dispatch.bindings)
}

fn dispatch3d_encoded_len(dispatch: &Dispatch3d) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_dimensions(dispatch.group_counts)?;
    validate_dimensions(dispatch.threads_per_group)?;
    if dispatch.checked_total_threads().is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    encoded_len_with_bindings(DISPATCH_3D_HEADER_LEN, &dispatch.bindings)
}

fn draw_indexed_encoded_len(draw: &DrawIndexed) -> Result<usize, WireError> {
    validate_pipeline(draw.pipeline_id)?;
    if draw.index_count == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    validate_list(draw.list_id, draw.list_generation)?;
    encoded_len_with_bindings(DRAW_INDEXED_HEADER_LEN, &draw.bindings)
}

fn draw_encoded_len(draw: &Draw) -> Result<usize, WireError> {
    validate_pipeline(draw.pipeline_id)?;
    if draw.vertex_count == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    if draw.instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    validate_list(draw.list_id, draw.list_generation)?;
    encoded_len_with_bindings(DRAW_HEADER_LEN, &draw.bindings)
}

fn dispatch_mesh_encoded_len(dispatch: &DispatchMesh) -> Result<usize, WireError> {
    validate_pipeline(dispatch.pipeline_id)?;
    validate_dimensions(dispatch.group_counts)?;
    if dispatch.checked_group_count().is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    encoded_len_with_bindings(DISPATCH_MESH_HEADER_LEN, &dispatch.bindings)
}

fn barrier_encoded_len(barrier: &ResourceBarrier) -> Result<usize, WireError> {
    if barrier.entries.is_empty() {
        return Err(WireError::ZeroBarrierCount);
    }
    if barrier.entries.len() > MAX_BARRIERS_PER_COMMAND {
        return Err(WireError::TooManyBarriers(barrier.entries.len()));
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
    Ok(SET_PIPELINE_HEADER_LEN)
}

fn root_constants_encoded_len(constants: &RootConstants) -> Result<usize, WireError> {
    if constants.bytes.is_empty() {
        return Err(WireError::ZeroRootConstants);
    }
    if constants.bytes.len() > MAX_ROOT_CONSTANT_BYTES {
        return Err(WireError::RootConstantsTooLarge(constants.bytes.len()));
    }
    constants
        .bytes
        .len()
        .checked_add(ROOT_CONSTANTS_HEADER_LEN)
        .ok_or(WireError::FrameTooLarge)
}

fn descriptor_table_encoded_len(table: &DescriptorTable) -> Result<usize, WireError> {
    if table.bindings.is_empty() {
        return Err(WireError::ZeroDescriptorCount);
    }
    if table.bindings.len() > MAX_DESCRIPTOR_BINDINGS {
        return Err(WireError::TooManyDescriptorBindings(
            table.bindings.len(),
        ));
    }
    for binding in &table.bindings {
        validate_binding(binding)?;
    }
    table
        .bindings
        .len()
        .checked_mul(BINDING_LEN)
        .and_then(|bindings_len| bindings_len.checked_add(DESCRIPTOR_TABLE_HEADER_LEN))
        .ok_or(WireError::FrameTooLarge)
}

fn encoded_len_with_bindings(
    header_len: usize,
    bindings: &[BufferBinding],
) -> Result<usize, WireError> {
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

fn command_encoded_len(command: &Command) -> Result<usize, WireError> {
    match command {
        Command::Dispatch1d(dispatch) => dispatch_encoded_len(dispatch),
        Command::Dispatch3d(dispatch) => dispatch3d_encoded_len(dispatch),
        Command::DrawIndexed(draw) => draw_indexed_encoded_len(draw),
        Command::Draw(draw) => draw_encoded_len(draw),
        Command::DispatchMesh(dispatch) => dispatch_mesh_encoded_len(dispatch),
        Command::ResourceBarrier(barrier) => barrier_encoded_len(barrier),
        Command::SetPipeline(set_pipeline) => set_pipeline_encoded_len(set_pipeline),
        Command::RootConstants(constants) => root_constants_encoded_len(constants),
        Command::DescriptorTable(table) => descriptor_table_encoded_len(table),
    }
}

fn write_command(out: &mut Vec<u8>, command: &Command) {
    match command {
        Command::Dispatch1d(dispatch) => write_dispatch(out, dispatch),
        Command::Dispatch3d(dispatch) => write_dispatch_3d(out, dispatch),
        Command::DrawIndexed(draw) => write_draw_indexed(out, draw),
        Command::Draw(draw) => write_draw(out, draw),
        Command::DispatchMesh(dispatch) => write_dispatch_mesh(out, dispatch),
        Command::ResourceBarrier(barrier) => write_barrier(out, barrier),
        Command::SetPipeline(set_pipeline) => write_set_pipeline(out, set_pipeline),
        Command::RootConstants(constants) => write_root_constants(out, constants),
        Command::DescriptorTable(table) => write_descriptor_table(out, table),
    }
}

fn read_command(reader: &mut Reader<'_>) -> Result<Command, WireError> {
    let opcode = reader.u16()?;
    let flags = reader.u16()?;
    if flags != 0 {
        return Err(WireError::UnsupportedFlags(flags));
    }
    match opcode {
        OPCODE_DISPATCH_1D => Ok(Command::Dispatch1d(read_dispatch(reader)?)),
        OPCODE_DISPATCH_3D => Ok(Command::Dispatch3d(read_dispatch_3d(reader)?)),
        OPCODE_DRAW_INDEXED => Ok(Command::DrawIndexed(read_draw_indexed(reader)?)),
        OPCODE_DRAW => Ok(Command::Draw(read_draw(reader)?)),
        OPCODE_DISPATCH_MESH => Ok(Command::DispatchMesh(read_dispatch_mesh(reader)?)),
        OPCODE_RESOURCE_BARRIER => Ok(Command::ResourceBarrier(read_barrier(reader)?)),
        OPCODE_SET_PIPELINE => Ok(Command::SetPipeline(read_set_pipeline(reader)?)),
        OPCODE_ROOT_CONSTANTS => Ok(Command::RootConstants(read_root_constants(reader)?)),
        OPCODE_DESCRIPTOR_TABLE => Ok(Command::DescriptorTable(read_descriptor_table(reader)?)),
        other => Err(WireError::UnknownOpcode(other)),
    }
}

fn read_dispatch(reader: &mut Reader<'_>) -> Result<Dispatch1d, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;

    let threads = reader.u32()?;
    let threads_per_group = reader.u32()?;
    validate_threads(threads, threads_per_group)?;

    let bindings = read_bindings(reader)?;

    Ok(Dispatch1d {
        pipeline_id,
        threads,
        threads_per_group,
        bindings,
    })
}

fn read_dispatch_3d(reader: &mut Reader<'_>) -> Result<Dispatch3d, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;

    let group_counts = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(group_counts)?;
    let threads_per_group = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(threads_per_group)?;

    let dispatch = Dispatch3d {
        pipeline_id,
        group_counts,
        threads_per_group,
        bindings: Vec::new(),
    };
    if dispatch.checked_total_threads().is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    let bindings = read_bindings(reader)?;
    Ok(Dispatch3d {
        bindings,
        ..dispatch
    })
}

fn read_draw_indexed(reader: &mut Reader<'_>) -> Result<DrawIndexed, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let index_count = reader.u32()?;
    if index_count == 0 {
        return Err(WireError::ZeroIndexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    let start_index = reader.u32()?;
    let base_vertex = reader.i32()?;
    let start_instance = reader.u32()?;
    let list_id = reader.u64()?;
    let list_generation = reader.u32()?;
    validate_list(list_id, list_generation)?;
    let bindings = read_bindings(reader)?;

    Ok(DrawIndexed {
        pipeline_id,
        index_count,
        instance_count,
        start_index,
        base_vertex,
        start_instance,
        list_id,
        list_generation,
        bindings,
    })
}

fn read_draw(reader: &mut Reader<'_>) -> Result<Draw, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let vertex_count = reader.u32()?;
    if vertex_count == 0 {
        return Err(WireError::ZeroVertexCount);
    }
    let instance_count = reader.u32()?;
    if instance_count == 0 {
        return Err(WireError::ZeroInstanceCount);
    }
    let start_vertex = reader.u32()?;
    let start_instance = reader.u32()?;
    let list_id = reader.u64()?;
    let list_generation = reader.u32()?;
    validate_list(list_id, list_generation)?;
    let bindings = read_bindings(reader)?;

    Ok(Draw {
        pipeline_id,
        vertex_count,
        instance_count,
        start_vertex,
        start_instance,
        list_id,
        list_generation,
        bindings,
    })
}

fn read_dispatch_mesh(reader: &mut Reader<'_>) -> Result<DispatchMesh, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    let group_counts = [reader.u32()?, reader.u32()?, reader.u32()?];
    validate_dimensions(group_counts)?;
    let dispatch = DispatchMesh {
        pipeline_id,
        group_counts,
        bindings: Vec::new(),
    };
    if dispatch.checked_group_count().is_none() {
        return Err(WireError::ThreadCountOverflow);
    }
    let bindings = read_bindings(reader)?;
    Ok(DispatchMesh {
        bindings,
        ..dispatch
    })
}

fn read_barrier(reader: &mut Reader<'_>) -> Result<ResourceBarrier, WireError> {
    let count = usize::from(reader.u16()?);
    if reader.u16()? != 0 {
        return Err(WireError::NonZeroReserved);
    }
    if count == 0 {
        return Err(WireError::ZeroBarrierCount);
    }
    if count > MAX_BARRIERS_PER_COMMAND {
        return Err(WireError::TooManyBarriers(count));
    }

    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let allocation_id = reader.u64()?;
        let generation = reader.u32()?;
        if reader.u32()? != 0 {
            return Err(WireError::NonZeroReserved);
        }
        let offset = reader.u64()?;
        let length = reader.u64()?;
        let before_state = reader.u32()?;
        let after_state = reader.u32()?;
        validate_range(allocation_id, generation, offset, length)?;
        entries.push(BarrierEntry {
            allocation_id,
            generation,
            offset,
            length,
            before_state,
            after_state,
        });
    }
    Ok(ResourceBarrier { entries })
}

fn read_set_pipeline(reader: &mut Reader<'_>) -> Result<SetPipeline, WireError> {
    let pipeline_id = reader.u64()?;
    validate_pipeline(pipeline_id)?;
    Ok(SetPipeline { pipeline_id })
}

fn read_root_constants(reader: &mut Reader<'_>) -> Result<RootConstants, WireError> {
    let slot = reader.u16()?;
    let byte_len = usize::from(reader.u16()?);
    if byte_len == 0 {
        return Err(WireError::ZeroRootConstants);
    }
    if byte_len > MAX_ROOT_CONSTANT_BYTES {
        return Err(WireError::RootConstantsTooLarge(byte_len));
    }
    let bytes = reader.bytes(byte_len)?.to_vec();
    Ok(RootConstants { slot, bytes })
}

fn read_descriptor_table(reader: &mut Reader<'_>) -> Result<DescriptorTable, WireError> {
    let set = reader.u16()?;
    let base_register = reader.u16()?;
    let binding_count = usize::from(reader.u16()?);
    if reader.u16()? != 0 {
        return Err(WireError::NonZeroReserved);
    }
    if binding_count == 0 {
        return Err(WireError::ZeroDescriptorCount);
    }
    if binding_count > MAX_DESCRIPTOR_BINDINGS {
        return Err(WireError::TooManyDescriptorBindings(binding_count));
    }

    let mut bindings = Vec::with_capacity(binding_count);
    for _ in 0..binding_count {
        bindings.push(read_binding(reader)?);
    }
    Ok(DescriptorTable {
        set,
        base_register,
        bindings,
    })
}

fn read_bindings(reader: &mut Reader<'_>) -> Result<Vec<BufferBinding>, WireError> {
    let binding_count = usize::from(reader.u16()?);
    if reader.u16()? != 0 {
        return Err(WireError::NonZeroReserved);
    }
    if binding_count > MAX_BINDINGS_PER_DISPATCH {
        return Err(WireError::TooManyBindings(binding_count));
    }

    let mut bindings = Vec::with_capacity(binding_count);
    for _ in 0..binding_count {
        bindings.push(read_binding(reader)?);
    }
    Ok(bindings)
}

fn read_binding(reader: &mut Reader<'_>) -> Result<BufferBinding, WireError> {
    let allocation_id = reader.u64()?;
    let generation = reader.u32()?;
    if reader.u32()? != 0 {
        return Err(WireError::NonZeroReserved);
    }
    let offset = reader.u64()?;
    let length = reader.u64()?;

    let binding = BufferBinding {
        allocation_id,
        generation,
        offset,
        length,
    };
    validate_binding(&binding)?;
    Ok(binding)
}

struct FrameHeader {
    kind: u16,
    sequence: u32,
    command_count: u16,
}

impl FrameHeader {
    fn parse(bytes: &[u8]) -> Result<(Self, &[u8]), WireError> {
        if bytes.len() < FRAME_HEADER_LEN {
            return Err(WireError::Truncated);
        }

        let mut reader = Reader::new(bytes);
        let magic = reader.take::<4>()?;
        if magic != MAGIC {
            return Err(WireError::BadMagic);
        }

        let version = reader.u16()?;
        if version != PROTOCOL_VERSION {
            return Err(WireError::UnsupportedVersion(version));
        }

        let kind = reader.u16()?;
        let sequence = reader.u32()?;
        validate_sequence(sequence)?;

        let command_count = reader.u16()?;
        if reader.u16()? != 0 {
            return Err(WireError::NonZeroReserved);
        }

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

fn write_frame_header(
    out: &mut Vec<u8>,
    kind: u16,
    sequence: u32,
    command_count: u16,
    payload_len: u32,
) {
    out.extend_from_slice(&MAGIC);
    write_u16(out, PROTOCOL_VERSION);
    write_u16(out, kind);
    write_u32(out, sequence);
    write_u16(out, command_count);
    write_u16(out, 0);
    write_u32(out, payload_len);
}

fn write_dispatch(out: &mut Vec<u8>, dispatch: &Dispatch1d) {
    write_u16(out, OPCODE_DISPATCH_1D);
    write_u16(out, 0);
    write_u64(out, dispatch.pipeline_id);
    write_u32(out, dispatch.threads);
    write_u32(out, dispatch.threads_per_group);
    write_u16(out, dispatch.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &dispatch.bindings);
}

fn write_dispatch_3d(out: &mut Vec<u8>, dispatch: &Dispatch3d) {
    write_u16(out, OPCODE_DISPATCH_3D);
    write_u16(out, 0);
    write_u64(out, dispatch.pipeline_id);
    for dimension in dispatch.group_counts {
        write_u32(out, dimension);
    }
    for dimension in dispatch.threads_per_group {
        write_u32(out, dimension);
    }
    write_u16(out, dispatch.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &dispatch.bindings);
}

fn write_draw_indexed(out: &mut Vec<u8>, draw: &DrawIndexed) {
    write_u16(out, OPCODE_DRAW_INDEXED);
    write_u16(out, 0);
    write_u64(out, draw.pipeline_id);
    write_u32(out, draw.index_count);
    write_u32(out, draw.instance_count);
    write_u32(out, draw.start_index);
    write_i32(out, draw.base_vertex);
    write_u32(out, draw.start_instance);
    write_u64(out, draw.list_id);
    write_u32(out, draw.list_generation);
    write_u16(out, draw.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &draw.bindings);
}

fn write_draw(out: &mut Vec<u8>, draw: &Draw) {
    write_u16(out, OPCODE_DRAW);
    write_u16(out, 0);
    write_u64(out, draw.pipeline_id);
    write_u32(out, draw.vertex_count);
    write_u32(out, draw.instance_count);
    write_u32(out, draw.start_vertex);
    write_u32(out, draw.start_instance);
    write_u64(out, draw.list_id);
    write_u32(out, draw.list_generation);
    write_u16(out, draw.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &draw.bindings);
}

fn write_dispatch_mesh(out: &mut Vec<u8>, dispatch: &DispatchMesh) {
    write_u16(out, OPCODE_DISPATCH_MESH);
    write_u16(out, 0);
    write_u64(out, dispatch.pipeline_id);
    for dimension in dispatch.group_counts {
        write_u32(out, dimension);
    }
    write_u16(out, dispatch.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &dispatch.bindings);
}

fn write_barrier(out: &mut Vec<u8>, barrier: &ResourceBarrier) {
    write_u16(out, OPCODE_RESOURCE_BARRIER);
    write_u16(out, 0);
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

fn write_set_pipeline(out: &mut Vec<u8>, set_pipeline: &SetPipeline) {
    write_u16(out, OPCODE_SET_PIPELINE);
    write_u16(out, 0);
    write_u64(out, set_pipeline.pipeline_id);
}

fn write_root_constants(out: &mut Vec<u8>, constants: &RootConstants) {
    write_u16(out, OPCODE_ROOT_CONSTANTS);
    write_u16(out, 0);
    write_u16(out, constants.slot);
    write_u16(out, constants.bytes.len() as u16);
    out.extend_from_slice(&constants.bytes);
}

fn write_descriptor_table(out: &mut Vec<u8>, table: &DescriptorTable) {
    write_u16(out, OPCODE_DESCRIPTOR_TABLE);
    write_u16(out, 0);
    write_u16(out, table.set);
    write_u16(out, table.base_register);
    write_u16(out, table.bindings.len() as u16);
    write_u16(out, 0);
    write_bindings(out, &table.bindings);
}

fn write_bindings(out: &mut Vec<u8>, bindings: &[BufferBinding]) {
    for binding in bindings {
        write_u64(out, binding.allocation_id);
        write_u32(out, binding.generation);
        write_u32(out, 0);
        write_u64(out, binding.offset);
        write_u64(out, binding.length);
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

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let mut value = [0u8; N];
        value.copy_from_slice(self.bytes(N)?);
        Ok(value)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], WireError> {
        let end = self.position.checked_add(len).ok_or(WireError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(WireError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    fn i32(&mut self) -> Result<i32, WireError> {
        Ok(i32::from_le_bytes(self.take::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Field offsets in the pinned version 2 layout, used to corrupt valid
    // frames and prove each validation rejects before execution.
    const KIND_AT: usize = 6;
    const SEQUENCE_AT: usize = 8;
    const COMMAND_COUNT_AT: usize = 12;
    const LIST_BLOCK_START: usize = FRAME_HEADER_LEN;
    const LIST_ID_AT: usize = LIST_BLOCK_START;
    const LIST_GENERATION_AT: usize = LIST_BLOCK_START + 8;
    const COMMAND_START: usize = FRAME_HEADER_LEN + LIST_BLOCK_LEN;
    const OPCODE_AT: usize = COMMAND_START;
    const FLAGS_AT: usize = COMMAND_START + 2;
    const PIPELINE_ID_AT: usize = COMMAND_START + 4;
    const THREADS_AT: usize = COMMAND_START + 12;
    const BINDING_COUNT_AT: usize = COMMAND_START + 20;
    const BINDING_START: usize = COMMAND_START + COMMAND_HEADER_LEN;
    const ALLOCATION_ID_AT: usize = BINDING_START;
    const GENERATION_AT: usize = BINDING_START + 8;
    const OFFSET_AT: usize = BINDING_START + 16;
    const LENGTH_AT: usize = BINDING_START + 24;
    const DISPATCH_3D_GROUP_X_AT: usize = COMMAND_START + 12;
    const DISPATCH_3D_LOCAL_Z_AT: usize = COMMAND_START + 32;
    const DISPATCH_3D_BINDING_START: usize = COMMAND_START + DISPATCH_3D_HEADER_LEN;
    const REPLY_STATUS_AT: usize = FRAME_HEADER_LEN;

    const GOLDEN_SUBMISSION: &str = "
        56 4d 57 46 02 00 01 00 01 00 00 00 02 00 00 00
        7c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        01 00 00 00 01 00 00 00 00 00 00 00 00 00 01 00
        40 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 01 00 00 00 02 00 00 00
        00 00 00 00 00 00 01 00 40 00 00 00 01 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00
    ";

    const GOLDEN_3D_SUBMISSION: &str = "
        56 4d 57 46 02 00 01 00 03 00 00 00 01 00 00 00
        54 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        02 00 00 00 03 00 00 00 00 00 00 00 02 00 00 00
        03 00 00 00 04 00 00 00 04 00 00 00 02 00 00 00
        01 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00
    ";

    const GOLDEN_REPLY: &str =
        "56 4d 57 46 02 00 02 00 01 00 00 00 00 00 00 00 04 00 00 00 01 00 00 00";

    const GOLDEN_GRAPHICS: &str = "
        56 4d 57 46 02 00 01 00 05 00 00 00 09 00 00 00
        44 02 00 00 09 00 00 00 00 00 00 00 02 00 00 00
        01 00 00 00 01 00 00 00 00 00 00 00 00 00 01 00
        40 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 02 00 00 00 03 00 00 00
        00 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00
        04 00 00 00 02 00 00 00 01 00 00 00 01 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
        03 00 00 00 04 00 00 00 00 00 00 00 24 00 00 00
        02 00 00 00 03 00 00 00 f9 ff ff ff 05 00 00 00
        09 00 00 00 00 00 00 00 02 00 00 00 02 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00
        04 00 00 00 05 00 00 00 00 00 00 00 03 00 00 00
        01 00 00 00 0c 00 00 00 04 00 00 00 09 00 00 00
        00 00 00 00 02 00 00 00 01 00 00 00 01 00 00 00
        00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00
        00 00 00 00 00 00 01 00 00 00 00 00 05 00 00 00
        06 00 00 00 00 00 00 00 08 00 00 00 04 00 00 00
        02 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 06 00 00 00 02 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
        00 00 00 00 04 00 00 00 02 00 00 00 00 00 00 00
        03 00 00 00 00 00 00 00 00 10 00 00 00 00 00 00
        00 20 00 00 00 00 00 00 04 00 00 00 08 00 00 00
        07 00 00 00 07 00 00 00 00 00 00 00 08 00 00 00
        02 00 08 00 01 00 00 00 02 00 00 00 09 00 00 00
        02 00 01 00 02 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 01 00 00 00 00 00 00 00
        01 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
        00 00 01 00 00 00 00 00
    ";

    fn unhex(text: &str) -> Vec<u8> {
        let digits: Vec<u8> = text
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect();
        assert!(
            digits.len().is_multiple_of(2),
            "hex fixture has an odd digit count"
        );
        (0..digits.len() / 2)
            .map(|index| {
                let pair = [digits[index * 2], digits[index * 2 + 1]];
                let pair = std::str::from_utf8(&pair).expect("fixture is ASCII");
                u8::from_str_radix(pair, 16).expect("fixture is valid hex")
            })
            .collect()
    }

    fn binding(offset: u64) -> BufferBinding {
        BufferBinding {
            allocation_id: 1,
            generation: 1,
            offset,
            length: 65_536,
        }
    }

    fn dispatch(pipeline_id: u64, offset: u64) -> Dispatch1d {
        Dispatch1d {
            pipeline_id,
            threads: 65_536,
            threads_per_group: 64,
            bindings: vec![binding(offset)],
        }
    }

    fn dispatch_3d(pipeline_id: u64, offset: u64) -> Dispatch3d {
        Dispatch3d {
            pipeline_id,
            group_counts: [2, 3, 4],
            threads_per_group: [4, 2, 1],
            bindings: vec![binding(offset)],
        }
    }

    fn list_identity() -> ListIdentity {
        ListIdentity {
            list_id: 9,
            list_generation: 2,
        }
    }

    fn draw_indexed() -> DrawIndexed {
        DrawIndexed {
            pipeline_id: 4,
            index_count: 36,
            instance_count: 2,
            start_index: 3,
            base_vertex: -7,
            start_instance: 5,
            list_id: 9,
            list_generation: 2,
            bindings: vec![binding(0), binding(65_536)],
        }
    }

    fn draw() -> Draw {
        Draw {
            pipeline_id: 5,
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 12,
            start_instance: 4,
            list_id: 9,
            list_generation: 2,
            bindings: vec![binding(0)],
        }
    }

    fn dispatch_mesh() -> DispatchMesh {
        DispatchMesh {
            pipeline_id: 6,
            group_counts: [8, 4, 2],
            bindings: vec![binding(0)],
        }
    }

    fn barrier() -> ResourceBarrier {
        ResourceBarrier {
            entries: vec![
                BarrierEntry {
                    allocation_id: 1,
                    generation: 1,
                    offset: 0,
                    length: 65_536,
                    before_state: 0,
                    after_state: 4,
                },
                BarrierEntry {
                    allocation_id: 2,
                    generation: 3,
                    offset: 4096,
                    length: 8192,
                    before_state: 4,
                    after_state: 8,
                },
            ],
        }
    }

    fn set_pipeline() -> SetPipeline {
        SetPipeline { pipeline_id: 7 }
    }

    fn root_constants() -> RootConstants {
        RootConstants {
            slot: 2,
            bytes: vec![1, 0, 0, 0, 2, 0, 0, 0],
        }
    }

    fn descriptor_table() -> DescriptorTable {
        DescriptorTable {
            set: 2,
            base_register: 1,
            bindings: vec![binding(0), binding(65_536)],
        }
    }

    fn rich_submission() -> Submission {
        Submission {
            sequence: 5,
            commands: vec![
                Command::Dispatch1d(dispatch(1, 0)),
                Command::Dispatch3d(dispatch_3d(3, 0)),
                Command::DrawIndexed(draw_indexed()),
                Command::Draw(draw()),
                Command::DispatchMesh(dispatch_mesh()),
                Command::ResourceBarrier(barrier()),
                Command::SetPipeline(set_pipeline()),
                Command::RootConstants(root_constants()),
                Command::DescriptorTable(descriptor_table()),
            ],
        }
    }

    fn two_dispatch_submission() -> Submission {
        Submission {
            sequence: 1,
            commands: vec![
                Command::Dispatch1d(dispatch(1, 0)),
                Command::Dispatch1d(dispatch(2, 65_536)),
            ],
        }
    }

    fn dispatch_3d_submission() -> Submission {
        Submission {
            sequence: 3,
            commands: vec![Command::Dispatch3d(dispatch_3d(3, 0))],
        }
    }

    fn dispatch_of(command: &Command) -> &Dispatch1d {
        match command {
            Command::Dispatch1d(dispatch) => dispatch,
            other => panic!("expected 1D dispatch, got {other:?}"),
        }
    }

    fn dispatch_3d_of(command: &Command) -> &Dispatch3d {
        match command {
            Command::Dispatch3d(dispatch) => dispatch,
            other => panic!("expected 3D dispatch, got {other:?}"),
        }
    }

    fn dispatch_1d_mut(command: &mut Command) -> &mut Dispatch1d {
        match command {
            Command::Dispatch1d(dispatch) => dispatch,
            other => panic!("expected 1D dispatch, got {other:?}"),
        }
    }

    fn encoded_fixture() -> Vec<u8> {
        encode_submission(&two_dispatch_submission()).expect("fixture encodes")
    }

    fn encoded_3d_fixture() -> Vec<u8> {
        encode_submission(&dispatch_3d_submission()).expect("3D fixture encodes")
    }

    #[test]
    fn golden_fixture_pins_cross_host_layout() {
        let submission = two_dispatch_submission();
        assert_eq!(encoded_fixture(), unhex(GOLDEN_SUBMISSION));
        assert_eq!(
            decode_submission(&unhex(GOLDEN_SUBMISSION)).expect("golden decodes"),
            submission
        );

        let reply = Reply {
            sequence: 1,
            status: ReplyStatus::UnknownHandle,
        };
        assert_eq!(
            encode_reply(&reply).expect("reply encodes"),
            unhex(GOLDEN_REPLY)
        );
        assert_eq!(
            decode_reply(&unhex(GOLDEN_REPLY)).expect("golden reply decodes"),
            reply
        );
    }

    #[test]
    fn golden_3d_fixture_pins_opcode_2_without_changing_opcode_1() {
        let submission = dispatch_3d_submission();
        assert_eq!(encoded_3d_fixture(), unhex(GOLDEN_3D_SUBMISSION));
        assert_eq!(
            decode_submission(&unhex(GOLDEN_3D_SUBMISSION)).expect("3D golden decodes"),
            submission
        );

        // The frozen opcode 1 command payload and reply remain byte-for-byte
        // identical apart from the version and the list block.
        assert_eq!(encoded_fixture(), unhex(GOLDEN_SUBMISSION));
        assert_eq!(
            decode_submission(&unhex(GOLDEN_SUBMISSION)).expect("opcode 1 golden decodes"),
            two_dispatch_submission()
        );
        assert_eq!(
            encode_reply(&Reply {
                sequence: 1,
                status: ReplyStatus::UnknownHandle,
            })
            .expect("reply encodes"),
            unhex(GOLDEN_REPLY)
        );
    }

    #[test]
    fn graphics_golden_pins_version_two_opcodes_three_through_nine() {
        let submission = rich_submission();
        let frame =
            encode_submission_with_list(&submission, Some(list_identity())).expect("encode");
        assert_eq!(frame, unhex(GOLDEN_GRAPHICS));

        let (decoded, list) = decode_submission_with_list(&unhex(GOLDEN_GRAPHICS))
            .expect("graphics golden decodes");
        assert_eq!(decoded, submission);
        assert_eq!(list, Some(list_identity()));
    }

    #[test]
    fn graphics_vocabulary_roundtrips_as_one_ordered_batch() {
        let submission = rich_submission();
        let frame =
            encode_submission_with_list(&submission, Some(list_identity())).expect("encode");
        let (decoded, list) = decode_submission_with_list(&frame).expect("decode");

        assert_eq!(decoded, submission);
        assert_eq!(list, Some(list_identity()));
        assert_eq!(decoded.commands.len(), 9);
    }

    #[test]
    fn list_identity_roundtrip_and_partial_identity_rejected() {
        let submission = two_dispatch_submission();
        let frame =
            encode_submission_with_list(&submission, Some(list_identity())).expect("encode");
        assert_eq!(
            decode_submission_with_list(&frame).expect("decode").1,
            Some(list_identity())
        );
        // The compatibility decoder still reads the same frame.
        assert_eq!(decode_submission(&frame).expect("decode"), submission);

        for (error, list_id, generation) in [
            (WireError::ZeroListId, 0u64, 1u32),
            (WireError::ZeroListGeneration, 1, 0),
        ] {
            let mut frame = frame.clone();
            frame[LIST_ID_AT..LIST_ID_AT + 8].copy_from_slice(&list_id.to_le_bytes());
            frame[LIST_GENERATION_AT..LIST_GENERATION_AT + 4]
                .copy_from_slice(&generation.to_le_bytes());
            assert_eq!(
                decode_submission_with_list(&frame),
                Err(error),
                "identity ({list_id}, {generation})"
            );
        }

        assert_eq!(
            encode_submission_with_list(
                &submission,
                Some(ListIdentity {
                    list_id: 0,
                    list_generation: 1,
                }),
            ),
            Err(WireError::ZeroListId)
        );
        assert_eq!(
            encode_submission_with_list(
                &submission,
                Some(ListIdentity {
                    list_id: 1,
                    list_generation: 0,
                }),
            ),
            Err(WireError::ZeroListGeneration)
        );
        assert_eq!(ListIdentity::new(0, 1), Err(WireError::ZeroListId));
        assert_eq!(ListIdentity::new(1, 0), Err(WireError::ZeroListGeneration));
        assert_eq!(
            ListIdentity::new(1, 1).expect("valid identity"),
            ListIdentity {
                list_id: 1,
                list_generation: 1,
            }
        );
    }

    #[test]
    fn draw_indexed_roundtrip_preserves_negative_base_vertex() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::DrawIndexed(draw_indexed())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::DrawIndexed(draw) = &decoded.commands[0] else {
            panic!("expected indexed draw, got {:?}", decoded.commands[0]);
        };
        assert_eq!(draw.pipeline_id, 4);
        assert_eq!(draw.index_count, 36);
        assert_eq!(draw.instance_count, 2);
        assert_eq!(draw.start_index, 3);
        assert_eq!(draw.base_vertex, -7);
        assert_eq!(draw.start_instance, 5);
        assert_eq!(draw.list_identity(), list_identity());
        assert_eq!(draw.bindings, vec![binding(0), binding(65_536)]);
    }

    #[test]
    fn draw_roundtrip_preserves_vertices_and_instances() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::Draw(draw())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::Draw(draw) = &decoded.commands[0] else {
            panic!("expected draw, got {:?}", decoded.commands[0]);
        };
        assert_eq!(draw.pipeline_id, 5);
        assert_eq!(draw.vertex_count, 3);
        assert_eq!(draw.instance_count, 1);
        assert_eq!(draw.start_vertex, 12);
        assert_eq!(draw.start_instance, 4);
        assert_eq!(draw.list_identity(), list_identity());
        assert_eq!(draw.bindings, vec![binding(0)]);
    }

    #[test]
    fn dispatch_mesh_roundtrip_rejects_zero_grids_and_overflow() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::DispatchMesh(dispatch_mesh())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::DispatchMesh(dispatch) = &decoded.commands[0] else {
            panic!("expected mesh dispatch, got {:?}", decoded.commands[0]);
        };
        assert_eq!(dispatch.pipeline_id, 6);
        assert_eq!(dispatch.group_counts, [8, 4, 2]);
        assert_eq!(dispatch.checked_group_count(), Some(64));
        assert_eq!(dispatch.bindings, vec![binding(0)]);

        let mut zero = dispatch_mesh();
        zero.group_counts[2] = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DispatchMesh(zero)],
            }),
            Err(WireError::ZeroThreads)
        );

        let mut overflow = dispatch_mesh();
        overflow.group_counts = [u32::MAX; 3];
        assert_eq!(overflow.checked_group_count(), None);
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DispatchMesh(overflow)],
            }),
            Err(WireError::ThreadCountOverflow)
        );
    }

    #[test]
    fn resource_barrier_roundtrip_and_limits() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::ResourceBarrier(barrier())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::ResourceBarrier(decoded_barrier) = &decoded.commands[0] else {
            panic!("expected barrier, got {:?}", decoded.commands[0]);
        };
        assert_eq!(decoded_barrier.entries.len(), 2);
        assert_eq!(decoded_barrier.entries[0].before_state, 0);
        assert_eq!(decoded_barrier.entries[0].after_state, 4);
        assert_eq!(decoded_barrier.entries[1].allocation_id, 2);
        assert_eq!(decoded_barrier.entries[1].checked_end(), Some(4096 + 8192));

        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::ResourceBarrier(ResourceBarrier {
                    entries: Vec::new(),
                })],
            }),
            Err(WireError::ZeroBarrierCount)
        );

        let too_many = ResourceBarrier {
            entries: vec![barrier().entries[0]; MAX_BARRIERS_PER_COMMAND + 1],
        };
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::ResourceBarrier(too_many)],
            }),
            Err(WireError::TooManyBarriers(MAX_BARRIERS_PER_COMMAND + 1))
        );
    }

    #[test]
    fn set_pipeline_roundtrip() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::SetPipeline(set_pipeline())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        assert_eq!(
            decoded.commands[0],
            Command::SetPipeline(SetPipeline { pipeline_id: 7 })
        );

        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::SetPipeline(SetPipeline { pipeline_id: 0 })],
            }),
            Err(WireError::ZeroPipelineId)
        );
    }

    #[test]
    fn root_constants_roundtrip_and_bound() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::RootConstants(root_constants())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::RootConstants(constants) = &decoded.commands[0] else {
            panic!("expected root constants, got {:?}", decoded.commands[0]);
        };
        assert_eq!(constants.slot, 2);
        assert_eq!(constants.bytes, vec![1, 0, 0, 0, 2, 0, 0, 0]);

        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::RootConstants(RootConstants {
                    slot: 0,
                    bytes: Vec::new(),
                })],
            }),
            Err(WireError::ZeroRootConstants)
        );

        let oversized = RootConstants {
            slot: 0,
            bytes: vec![0u8; MAX_ROOT_CONSTANT_BYTES + 1],
        };
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::RootConstants(oversized)],
            }),
            Err(WireError::RootConstantsTooLarge(
                MAX_ROOT_CONSTANT_BYTES + 1
            ))
        );

        // The cap itself is accepted and roundtrips.
        let maximum = RootConstants {
            slot: u16::MAX,
            bytes: vec![0xa5; MAX_ROOT_CONSTANT_BYTES],
        };
        let decoded = decode_submission(
            &encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::RootConstants(maximum.clone())],
            })
            .expect("maximum root constants encode"),
        )
        .expect("maximum root constants decode");
        assert_eq!(decoded.commands[0], Command::RootConstants(maximum));
    }

    #[test]
    fn descriptor_table_roundtrip_and_limits() {
        let submission = Submission {
            sequence: 2,
            commands: vec![Command::DescriptorTable(descriptor_table())],
        };
        let decoded = decode_submission(&encode_submission(&submission).expect("encode"))
            .expect("decode");
        let Command::DescriptorTable(table) = &decoded.commands[0] else {
            panic!("expected descriptor table, got {:?}", decoded.commands[0]);
        };
        assert_eq!(table.set, 2);
        assert_eq!(table.base_register, 1);
        assert_eq!(table.bindings, vec![binding(0), binding(65_536)]);

        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DescriptorTable(DescriptorTable {
                    set: 0,
                    base_register: 0,
                    bindings: Vec::new(),
                })],
            }),
            Err(WireError::ZeroDescriptorCount)
        );

        let too_many = DescriptorTable {
            set: 0,
            base_register: 0,
            bindings: vec![binding(0); MAX_DESCRIPTOR_BINDINGS + 1],
        };
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DescriptorTable(too_many)],
            }),
            Err(WireError::TooManyDescriptorBindings(
                MAX_DESCRIPTOR_BINDINGS + 1
            ))
        );
    }

    #[test]
    fn zero_draw_counts_are_rejected_before_storage() {
        let mut zero_index = draw_indexed();
        zero_index.index_count = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DrawIndexed(zero_index)],
            }),
            Err(WireError::ZeroIndexCount)
        );

        let mut zero_instances = draw_indexed();
        zero_instances.instance_count = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DrawIndexed(zero_instances)],
            }),
            Err(WireError::ZeroInstanceCount)
        );

        let mut zero_vertices = draw();
        zero_vertices.vertex_count = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::Draw(zero_vertices)],
            }),
            Err(WireError::ZeroVertexCount)
        );

        let mut zero_list = draw();
        zero_list.list_id = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::Draw(zero_list)],
            }),
            Err(WireError::ZeroListId)
        );

        let mut zero_generation = draw_indexed();
        zero_generation.list_generation = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::DrawIndexed(zero_generation)],
            }),
            Err(WireError::ZeroListGeneration)
        );
    }

    #[test]
    fn barrier_range_rules_match_bound_ranges() {
        let mut entry = barrier();
        entry.entries[0].allocation_id = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::ResourceBarrier(entry)],
            }),
            Err(WireError::ZeroAllocationId)
        );

        let mut entry = barrier();
        entry.entries[0].length = 0;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::ResourceBarrier(entry)],
            }),
            Err(WireError::ZeroLength)
        );

        let mut entry = barrier();
        entry.entries[0].offset = u64::MAX;
        entry.entries[0].length = 2;
        assert_eq!(
            encode_submission(&Submission {
                sequence: 1,
                commands: vec![Command::ResourceBarrier(entry)],
            }),
            Err(WireError::OffsetOverflow)
        );
    }

    #[test]
    fn dispatch3d_roundtrip_preserves_exact_grid_and_shared_bindings() {
        let submission = dispatch_3d_submission();
        let decoded =
            decode_submission(&encode_submission(&submission).expect("encode")).expect("decode");

        assert_eq!(decoded.sequence, 3);
        assert_eq!(decoded.commands.len(), 1);
        let dispatch = dispatch_3d_of(&decoded.commands[0]);
        assert_eq!(dispatch.pipeline_id, 3);
        assert_eq!(dispatch.group_counts, [2, 3, 4]);
        assert_eq!(dispatch.threads_per_group, [4, 2, 1]);
        assert_eq!(dispatch.checked_group_count(), Some(24));
        assert_eq!(dispatch.checked_threads_per_group(), Some(8));
        assert_eq!(dispatch.checked_total_threads(), Some(192));
        assert_eq!(dispatch.bindings.len(), 1);
        assert_eq!(dispatch.bindings[0], binding(0));
    }

    #[test]
    fn dispatch3d_rejects_zero_dimensions_and_overflow_before_storage() {
        for offset in [
            DISPATCH_3D_GROUP_X_AT,
            DISPATCH_3D_GROUP_X_AT + 4,
            DISPATCH_3D_GROUP_X_AT + 8,
            DISPATCH_3D_LOCAL_Z_AT - 8,
            DISPATCH_3D_LOCAL_Z_AT - 4,
            DISPATCH_3D_LOCAL_Z_AT,
        ] {
            let mut frame = encoded_3d_fixture();
            frame[offset..offset + 4].fill(0);
            assert_eq!(
                decode_submission(&frame),
                Err(WireError::ZeroThreads),
                "zero dimension at offset {offset}"
            );
        }

        let mut zero_group = dispatch_3d_submission();
        let Command::Dispatch3d(dispatch) = &mut zero_group.commands[0] else {
            unreachable!()
        };
        dispatch.group_counts[1] = 0;
        assert_eq!(encode_submission(&zero_group), Err(WireError::ZeroThreads));

        let mut overflow = dispatch_3d_submission();
        let Command::Dispatch3d(dispatch) = &mut overflow.commands[0] else {
            unreachable!()
        };
        dispatch.group_counts = [u32::MAX; 3];
        assert_eq!(dispatch.checked_group_count(), None);
        assert_eq!(dispatch.checked_total_threads(), None);
        assert_eq!(
            encode_submission(&overflow),
            Err(WireError::ThreadCountOverflow)
        );

        let mut frame = encoded_3d_fixture();
        for offset in [
            DISPATCH_3D_GROUP_X_AT,
            DISPATCH_3D_GROUP_X_AT + 4,
            DISPATCH_3D_GROUP_X_AT + 8,
        ] {
            frame[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        assert_eq!(
            decode_submission(&frame),
            Err(WireError::ThreadCountOverflow)
        );

        let mut oversized = dispatch_3d_submission();
        let Command::Dispatch3d(dispatch) = &mut oversized.commands[0] else {
            unreachable!()
        };
        dispatch.bindings = vec![binding(0); MAX_BINDINGS_PER_DISPATCH + 1];
        assert_eq!(
            encode_submission(&oversized),
            Err(WireError::TooManyBindings(MAX_BINDINGS_PER_DISPATCH + 1))
        );

        assert_eq!(
            DISPATCH_3D_BINDING_START,
            COMMAND_START + DISPATCH_3D_HEADER_LEN
        );
    }

    #[test]
    fn two_ordered_dispatches_roundtrip_with_offsets_zero_and_65536() {
        let submission = two_dispatch_submission();
        let decoded =
            decode_submission(&encode_submission(&submission).expect("encode")).expect("decode");

        assert_eq!(decoded.sequence, 1);
        assert_eq!(decoded.commands.len(), 2);

        let first = dispatch_of(&decoded.commands[0]);
        let second = dispatch_of(&decoded.commands[1]);
        assert_eq!(first.pipeline_id, 1);
        assert_eq!(second.pipeline_id, 2);
        assert_eq!(first.bindings[0].offset, 0);
        assert_eq!(second.bindings[0].offset, 65_536);
        assert_eq!(first.bindings[0].length, 65_536);
        assert_eq!(second.bindings[0].length, 65_536);
        assert_eq!(first.bindings[0].generation, 1);
        assert_eq!(second.bindings[0].generation, 1);
    }

    #[test]
    fn reply_roundtrip_preserves_sequence_and_status() {
        for status in [
            ReplyStatus::Ok,
            ReplyStatus::UnknownHandle,
            ReplyStatus::ExecutionFailed,
        ] {
            let reply = Reply {
                sequence: 42,
                status,
            };
            let decoded =
                decode_reply(&encode_reply(&reply).expect("encode reply")).expect("decode reply");
            assert_eq!(decoded.sequence, 42);
            assert_eq!(decoded.status, status);
        }
    }

    #[test]
    fn wrong_version_is_rejected() {
        let mut frame = encoded_fixture();
        frame[4..6].copy_from_slice(&(PROTOCOL_VERSION + 1).to_le_bytes());
        assert_eq!(
            decode_submission(&frame),
            Err(WireError::UnsupportedVersion(PROTOCOL_VERSION + 1))
        );

        // Version 1 frames are not silently reinterpreted as version 2.
        let mut version_one = encoded_fixture();
        version_one[4..6].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            decode_submission(&version_one),
            Err(WireError::UnsupportedVersion(1))
        );

        let mut reply = encode_reply(&Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        })
        .expect("encode reply");
        reply[4..6].copy_from_slice(&(PROTOCOL_VERSION + 1).to_le_bytes());
        assert_eq!(
            decode_reply(&reply),
            Err(WireError::UnsupportedVersion(PROTOCOL_VERSION + 1))
        );
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        for frame in [
            encoded_fixture(),
            encode_submission_with_list(&rich_submission(), Some(list_identity()))
                .expect("rich fixture encodes"),
        ] {
            for length in 0..frame.len() {
                assert!(
                    decode_submission(&frame[..length]).is_err(),
                    "submission prefix {length} must not decode"
                );
            }
        }

        let reply = encode_reply(&Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        })
        .expect("encode reply");
        for length in 0..reply.len() {
            assert!(
                decode_reply(&reply[..length]).is_err(),
                "reply prefix {length} must not decode"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut frame = encoded_fixture();
        frame.push(0);
        assert_eq!(decode_submission(&frame), Err(WireError::TrailingBytes));

        let mut reply = encode_reply(&Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        })
        .expect("encode reply");
        reply.push(0);
        assert_eq!(decode_reply(&reply), Err(WireError::TrailingBytes));
    }

    #[test]
    fn zero_sequence_is_rejected() {
        let mut frame = encoded_fixture();
        frame[SEQUENCE_AT..SEQUENCE_AT + 4].fill(0);
        assert_eq!(decode_submission(&frame), Err(WireError::ZeroSequence));

        assert_eq!(
            encode_submission(&Submission {
                sequence: 0,
                commands: Vec::new(),
            }),
            Err(WireError::ZeroSequence)
        );
        assert_eq!(
            encode_submission_with_list(
                &Submission {
                    sequence: 0,
                    commands: Vec::new(),
                },
                Some(list_identity()),
            ),
            Err(WireError::ZeroSequence)
        );
        assert_eq!(
            encode_reply(&Reply {
                sequence: 0,
                status: ReplyStatus::Ok,
            }),
            Err(WireError::ZeroSequence)
        );
    }

    #[test]
    fn zero_pipeline_allocation_generation_and_length_are_rejected() {
        let mutations: [(usize, usize, WireError); 4] = [
            (PIPELINE_ID_AT, 8, WireError::ZeroPipelineId),
            (ALLOCATION_ID_AT, 8, WireError::ZeroAllocationId),
            (GENERATION_AT, 4, WireError::ZeroGeneration),
            (LENGTH_AT, 8, WireError::ZeroLength),
        ];
        for (offset, width, error) in mutations {
            let mut frame = encoded_fixture();
            frame[offset..offset + width].fill(0);
            assert_eq!(
                decode_submission(&frame),
                Err(error),
                "corruption at offset {offset}"
            );
        }

        let mut zero_threads_frame = encoded_fixture();
        zero_threads_frame[THREADS_AT..THREADS_AT + 4].fill(0);
        assert_eq!(
            decode_submission(&zero_threads_frame),
            Err(WireError::ZeroThreads)
        );

        let mut zero_pipeline = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut zero_pipeline.commands[0]);
        dispatch.pipeline_id = 0;
        assert_eq!(
            encode_submission(&zero_pipeline),
            Err(WireError::ZeroPipelineId)
        );

        let mut zero_allocation = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut zero_allocation.commands[0]);
        dispatch.bindings[0].allocation_id = 0;
        assert_eq!(
            encode_submission(&zero_allocation),
            Err(WireError::ZeroAllocationId)
        );

        let mut zero_generation = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut zero_generation.commands[0]);
        dispatch.bindings[0].generation = 0;
        assert_eq!(
            encode_submission(&zero_generation),
            Err(WireError::ZeroGeneration)
        );

        let mut zero_length = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut zero_length.commands[0]);
        dispatch.bindings[0].length = 0;
        assert_eq!(encode_submission(&zero_length), Err(WireError::ZeroLength));

        let mut zero_threads = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut zero_threads.commands[0]);
        dispatch.threads = 0;
        assert_eq!(
            encode_submission(&zero_threads),
            Err(WireError::ZeroThreads)
        );
    }

    #[test]
    fn unsupported_reply_status_is_rejected() {
        let mut reply = encode_reply(&Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        })
        .expect("encode reply");
        reply[REPLY_STATUS_AT..REPLY_STATUS_AT + 2].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(decode_reply(&reply), Err(WireError::UnsupportedStatus(99)));
    }

    #[test]
    fn offset_overflow_is_rejected() {
        let mut frame = encoded_fixture();
        frame[OFFSET_AT..OFFSET_AT + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        frame[LENGTH_AT..LENGTH_AT + 8].copy_from_slice(&2u64.to_le_bytes());
        assert_eq!(decode_submission(&frame), Err(WireError::OffsetOverflow));

        let mut overflow = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut overflow.commands[0]);
        dispatch.bindings[0].offset = u64::MAX;
        dispatch.bindings[0].length = 2;
        assert_eq!(encode_submission(&overflow), Err(WireError::OffsetOverflow));

        let mut oversized = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut oversized.commands[0]);
        dispatch.bindings[0].length = MAX_RESOURCE_LEN + 1;
        assert_eq!(
            encode_submission(&oversized),
            Err(WireError::ResourceTooLarge(MAX_RESOURCE_LEN + 1))
        );

        let mut frame = encoded_fixture();
        frame[LENGTH_AT..LENGTH_AT + 8].copy_from_slice(&(MAX_RESOURCE_LEN + 1).to_le_bytes());
        assert_eq!(
            decode_submission(&frame),
            Err(WireError::ResourceTooLarge(MAX_RESOURCE_LEN + 1))
        );
    }

    #[test]
    fn oversized_counts_are_rejected_before_storage() {
        let mut frame = encoded_fixture();
        frame[COMMAND_COUNT_AT..COMMAND_COUNT_AT + 2]
            .copy_from_slice(&(MAX_COMMANDS as u16 + 1).to_le_bytes());
        assert_eq!(
            decode_submission(&frame),
            Err(WireError::TooManyCommands(MAX_COMMANDS + 1))
        );

        let mut frame = encoded_fixture();
        frame[BINDING_COUNT_AT..BINDING_COUNT_AT + 2]
            .copy_from_slice(&(MAX_BINDINGS_PER_DISPATCH as u16 + 1).to_le_bytes());
        assert_eq!(
            decode_submission(&frame),
            Err(WireError::TooManyBindings(MAX_BINDINGS_PER_DISPATCH + 1))
        );

        let too_many_commands = Submission {
            sequence: 1,
            commands: (0..=MAX_COMMANDS)
                .map(|index| Command::Dispatch1d(dispatch(index as u64 + 1, 0)))
                .collect(),
        };
        assert_eq!(
            encode_submission(&too_many_commands),
            Err(WireError::TooManyCommands(MAX_COMMANDS + 1))
        );

        let mut too_many_bindings = two_dispatch_submission();
        let dispatch = dispatch_1d_mut(&mut too_many_bindings.commands[0]);
        dispatch.bindings = vec![binding(0); MAX_BINDINGS_PER_DISPATCH + 1];
        assert_eq!(
            encode_submission(&too_many_bindings),
            Err(WireError::TooManyBindings(MAX_BINDINGS_PER_DISPATCH + 1))
        );
    }

    #[test]
    fn malformed_tags_and_reserved_words_are_rejected() {
        let mut wrong_kind = encoded_fixture();
        wrong_kind[KIND_AT..KIND_AT + 2].copy_from_slice(&KIND_REPLY.to_le_bytes());
        assert_eq!(
            decode_submission(&wrong_kind),
            Err(WireError::UnexpectedKind(KIND_REPLY))
        );

        // Opcodes 1 through 9 are defined in this version; anything else is
        // rejected explicitly instead of being interpreted by guesswork.
        let mut wrong_opcode = encoded_fixture();
        wrong_opcode[OPCODE_AT..OPCODE_AT + 2].copy_from_slice(&10u16.to_le_bytes());
        assert_eq!(
            decode_submission(&wrong_opcode),
            Err(WireError::UnknownOpcode(10))
        );

        let mut flags = encoded_fixture();
        flags[FLAGS_AT..FLAGS_AT + 2].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            decode_submission(&flags),
            Err(WireError::UnsupportedFlags(1))
        );

        let mut header_reserved = encoded_fixture();
        header_reserved[14..16].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            decode_submission(&header_reserved),
            Err(WireError::NonZeroReserved)
        );

        let mut binding_reserved = encoded_fixture();
        binding_reserved[GENERATION_AT + 4..GENERATION_AT + 8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            decode_submission(&binding_reserved),
            Err(WireError::NonZeroReserved)
        );

        let mut bad_magic = encoded_fixture();
        bad_magic[0] = b'X';
        assert_eq!(decode_submission(&bad_magic), Err(WireError::BadMagic));

        let mut reply_with_commands = encode_reply(&Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        })
        .expect("encode reply");
        reply_with_commands[COMMAND_COUNT_AT..COMMAND_COUNT_AT + 2]
            .copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            decode_reply(&reply_with_commands),
            Err(WireError::UnexpectedCommandCount(1))
        );
    }

    #[test]
    fn binding_end_checks_overflow() {
        assert_eq!(binding(0).checked_end(), Some(65_536));
        assert_eq!(
            BufferBinding {
                offset: u64::MAX,
                length: 2,
                ..binding(0)
            }
            .checked_end(),
            None
        );
    }
}
