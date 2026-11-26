//! Versioned cross-process submission wire format for ordered compute work.
//!
//! This is the first real ABI between the Win64 Wine client and the ARM64
//! renderer host. It frames command metadata only. File registration, socket
//! or shared-memory transport, allocation ownership and GPU lifetime live in
//! separate layers.
//!
//! # Version 1 frame layout
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
//! | 4 | 2 | protocol version (1) |
//! | 6 | 2 | frame kind (1 = submission, 2 = reply) |
//! | 8 | 4 | submission sequence (never zero) |
//! | 12 | 2 | command count (submissions only, at most [`MAX_COMMANDS`]) |
//! | 14 | 2 | reserved, must be zero |
//! | 16 | 4 | payload length in bytes |
//!
//! A submission payload is an ordered list of commands. Version 1 defines two
//! command opcodes: a 1D grid (opcode 1) and an explicit 3D threadgroup grid
//! (opcode 2). A decoder that predates opcode 2 rejects it as
//! [`WireError::UnknownOpcode`]; it never guesses at a newer payload layout.
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
//! Each bound range is an allocation ID, generation, reserved word, byte
//! offset and byte length. IDs, generations and lengths must be nonzero;
//! `offset + length` is checked against overflow and `length` is capped by
//! [`MAX_RESOURCE_LEN`]. Ranges contain no raw guest virtual addresses.
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
pub const PROTOCOL_VERSION: u16 = 1;

/// Frame magic. The version field, not the magic, selects the protocol.
pub const MAGIC: [u8; 4] = *b"VMWF";

/// Maximum number of commands in one submission.
pub const MAX_COMMANDS: usize = 64;

/// Maximum number of buffer bindings in one dispatch.
pub const MAX_BINDINGS_PER_DISPATCH: usize = 16;

/// Maximum byte length of one bound buffer range.
pub const MAX_RESOURCE_LEN: u64 = 1 << 32;

const KIND_SUBMISSION: u16 = 1;
const KIND_REPLY: u16 = 2;

const OPCODE_DISPATCH_1D: u16 = 1;
const OPCODE_DISPATCH_3D: u16 = 2;

const FRAME_HEADER_LEN: usize = 20;
const COMMAND_HEADER_LEN: usize = 24;
const DISPATCH_3D_HEADER_LEN: usize = 40;
const BINDING_LEN: usize = 32;
const REPLY_PAYLOAD_LEN: usize = 4;

/// An ordered, atomic batch of commands in one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    /// Submission sequence. Must be nonzero; the session layer enforces that
    /// successive values increase from 1.
    pub sequence: u32,
    /// Commands in execution order.
    pub commands: Vec<Command>,
}

/// A command decoded from a submission payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// A pre-created compute pipeline dispatched over a 1D grid.
    Dispatch1d(Dispatch1d),
    /// A pre-created compute pipeline dispatched over an explicit 3D
    /// threadgroup grid.
    Dispatch3d(Dispatch3d),
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

/// Distinguishable reply statuses on the version 1 wire.
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
    /// A dispatch contains more than [`MAX_BINDINGS_PER_DISPATCH`] bindings.
    TooManyBindings(usize),
    /// A reply unexpectedly carries commands.
    UnexpectedCommandCount(u16),
    /// The command opcode is not defined by this protocol version.
    UnknownOpcode(u16),
    /// Pipeline ID zero is not a valid pre-created handle.
    ZeroPipelineId,
    /// Thread count or threads per threadgroup was zero.
    ZeroThreads,
    /// A 3D group count, local dimension or total thread product overflowed
    /// `u64`.
    ThreadCountOverflow,
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
            Self::UnexpectedCommandCount(count) => {
                write!(f, "reply carries {count} commands")
            }
            Self::UnknownOpcode(opcode) => write!(f, "unknown command opcode {opcode}"),
            Self::ZeroPipelineId => write!(f, "pipeline ID must be nonzero"),
            Self::ZeroThreads => write!(f, "thread counts must be nonzero"),
            Self::ThreadCountOverflow => write!(f, "3D thread count product overflows u64"),
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

/// Encodes one submission as a complete version 1 frame.
///
/// The same field rules as decoding are enforced, so a writer cannot produce
/// a frame this codec would reject. Sequence monotonicity across frames is not
/// checked here.
pub fn encode_submission(submission: &Submission) -> Result<Vec<u8>, WireError> {
    validate_sequence(submission.sequence)?;
    if submission.commands.len() > MAX_COMMANDS {
        return Err(WireError::TooManyCommands(submission.commands.len()));
    }

    let mut payload_len = 0usize;
    for command in &submission.commands {
        match command {
            Command::Dispatch1d(dispatch) => {
                payload_len = payload_len
                    .checked_add(dispatch_encoded_len(dispatch)?)
                    .ok_or(WireError::FrameTooLarge)?;
            }
            Command::Dispatch3d(dispatch) => {
                payload_len = payload_len
                    .checked_add(dispatch3d_encoded_len(dispatch)?)
                    .ok_or(WireError::FrameTooLarge)?;
            }
        }
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
    for command in &submission.commands {
        match command {
            Command::Dispatch1d(dispatch) => write_dispatch(&mut out, dispatch),
            Command::Dispatch3d(dispatch) => write_dispatch_3d(&mut out, dispatch),
        }
    }
    Ok(out)
}

/// Decodes one complete submission frame.
///
/// Returns no commands on any error. Payload length, command count and binding
/// count are validated before the corresponding storage is reserved.
pub fn decode_submission(bytes: &[u8]) -> Result<Submission, WireError> {
    let (header, payload) = FrameHeader::parse(bytes)?;
    if header.kind != KIND_SUBMISSION {
        return Err(WireError::UnexpectedKind(header.kind));
    }

    let command_count = usize::from(header.command_count);
    if command_count > MAX_COMMANDS {
        return Err(WireError::TooManyCommands(command_count));
    }

    let mut reader = Reader::new(payload);
    let mut commands = Vec::with_capacity(command_count);
    for _ in 0..command_count {
        commands.push(read_command(&mut reader)?);
    }
    if !reader.is_empty() {
        return Err(WireError::TrailingBytes);
    }

    Ok(Submission {
        sequence: header.sequence,
        commands,
    })
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

fn validate_threads(threads: u32, threads_per_group: u32) -> Result<(), WireError> {
    if threads == 0 || threads_per_group == 0 {
        Err(WireError::ZeroThreads)
    } else {
        Ok(())
    }
}

fn validate_binding(binding: &BufferBinding) -> Result<(), WireError> {
    if binding.allocation_id == 0 {
        return Err(WireError::ZeroAllocationId);
    }
    if binding.generation == 0 {
        return Err(WireError::ZeroGeneration);
    }
    if binding.length == 0 {
        return Err(WireError::ZeroLength);
    }
    if binding.length > MAX_RESOURCE_LEN {
        return Err(WireError::ResourceTooLarge(binding.length));
    }
    if binding.checked_end().is_none() {
        return Err(WireError::OffsetOverflow);
    }
    Ok(())
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

fn validate_dimensions(dimensions: [u32; 3]) -> Result<(), WireError> {
    if dimensions.contains(&0) {
        Err(WireError::ZeroThreads)
    } else {
        Ok(())
    }
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

fn read_command(reader: &mut Reader<'_>) -> Result<Command, WireError> {
    let opcode = reader.u16()?;
    let flags = reader.u16()?;
    if flags != 0 {
        return Err(WireError::UnsupportedFlags(flags));
    }
    match opcode {
        OPCODE_DISPATCH_1D => Ok(Command::Dispatch1d(read_dispatch(reader)?)),
        OPCODE_DISPATCH_3D => Ok(Command::Dispatch3d(read_dispatch_3d(reader)?)),
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
        let end = self.position.checked_add(N).ok_or(WireError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(WireError::Truncated)?;
        self.position = end;

        let mut value = [0u8; N];
        value.copy_from_slice(slice);
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
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

    // Field offsets in the pinned version 1 layout, used to corrupt valid
    // frames and prove each validation rejects before execution.
    const KIND_AT: usize = 6;
    const SEQUENCE_AT: usize = 8;
    const COMMAND_COUNT_AT: usize = 12;
    const COMMAND_START: usize = FRAME_HEADER_LEN;
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
        56 4d 57 46 01 00 01 00 01 00 00 00 02 00 00 00 70 00 00 00
        01 00 00 00 01 00 00 00 00 00 00 00 00 00 01 00 40 00 00 00 01 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
        01 00 00 00 02 00 00 00 00 00 00 00 00 00 01 00 40 00 00 00 01 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00
    ";

    const GOLDEN_3D_SUBMISSION: &str = "
        56 4d 57 46 01 00 01 00 03 00 00 00 01 00 00 00 48 00 00 00
        02 00 00 00 03 00 00 00 00 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00
        04 00 00 00 02 00 00 00 01 00 00 00 01 00 00 00
        01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
        00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00
    ";

    const GOLDEN_REPLY: &str =
        "56 4d 57 46 01 00 02 00 01 00 00 00 00 00 00 00 04 00 00 00 01 00 00 00";

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

        // The frozen opcode 1 frame and reply remain byte-for-byte identical.
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
        let frame = encoded_fixture();
        for length in 0..frame.len() {
            assert!(
                decode_submission(&frame[..length]).is_err(),
                "submission prefix {length} must not decode"
            );
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

        // Opcode 2 is defined in this version; an opcode this decoder does not
        // know is rejected explicitly instead of being interpreted by guesswork.
        let mut wrong_opcode = encoded_fixture();
        wrong_opcode[OPCODE_AT..OPCODE_AT + 2].copy_from_slice(&3u16.to_le_bytes());
        assert_eq!(
            decode_submission(&wrong_opcode),
            Err(WireError::UnknownOpcode(3))
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
