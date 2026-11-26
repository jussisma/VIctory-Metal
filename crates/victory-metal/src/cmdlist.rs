//! Batched D3D12 command-list ring: per-list identity, ordering and
//! exactly-once delivery.
//!
//! The guest records a whole command list (draws, dispatches, resource
//! barriers, pipeline and root-argument state) and publishes it as contiguous
//! bytes in a shared-memory ring rather than one IPC round-trip per draw. The
//! host drains the ring and executes each list exactly once.
//!
//! Invariants (see `docs/ARM64-RENDERER.md` and `.swarm/decisions.json`):
//!
//! - A list is identified by `(list_id, generation)`. Execution is keyed by
//!   that pair plus the submission sequence, so a replay or retry of a list
//!   that already executed is detected and refused instead of re-executed.
//! - Publication is atomic: a record is visible to the host only after its
//!   length and checksum are committed, and the host never executes a record
//!   whose bytes fail validation.
//! - Wrapping never overwrites a record the host has not drained; the
//!   producer stops with an explicit backpressure error instead of corrupting
//!   an in-flight list.
//! - An uncertain native execution is reported as an error; the path never
//!   falls back to a second execution of the same list.
//!
//! This module is transport- and Metal-independent so its tests run without a
//! GPU. `wire` owns the per-command payload encoding; the ring owns framing,
//! ordering and the exactly-once ledger.
//!
//! # Record layout
//!
//! Every record starts on a [`RECORD_ALIGN`] byte boundary and has a fixed
//! 64-byte little-endian header followed by the payload:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 4 | magic `VMCL` (or `VMPD` for wrap padding) |
//! | 4 | 2 | record format version (1) |
//! | 6 | 2 | flags, must be zero |
//! | 8 | 4 | total record length, rounded up to 64 bytes |
//! | 12 | 4 | payload length |
//! | 16 | 4 | CRC-32/ISO-HDLC of the payload |
//! | 20 | 4 | reserved, must be zero |
//! | 24 | 8 | list ID |
//! | 32 | 4 | list generation |
//! | 36 | 4 | submission sequence |
//! | 40 | 8 | commit marker, written last |
//! | 48 | 16 | reserved, must be zero |
//!
//! The producer writes every other field, the payload and its padding before
//! it writes the commit marker. A record without the marker is never handed to
//! the consumer. A payload byte that changes fails the local CRC-32 check and
//! the corrupt record is discarded without being executed.
//!
//! Records never straddle the end of the ring. When the next record does not
//! fit in the tail, the producer commits a padding record up to the end and
//! starts the real record at offset zero. The padding record is also called
//! out by the commit marker so a torn wrap is never followed.
//!
//! # Capacity
//!
//! [`CmdRing::new`] rounds the requested capacity up to a multiple of
//! [`RECORD_ALIGN`] (64 bytes). The ring must hold one maximum record, so the
//! raw capacity must be at least [`MIN_CAPACITY_BYTES`]; a smaller value is
//! rejected by panicking in the constructor. [`MAX_RECORD_BYTES`] is larger
//! than the largest possible version-2 wire submission (64 barrier commands
//! of 16 entries produce a 41,504-byte frame), so any frame the codec can
//! produce fits.
//!
//! # Threading model
//!
//! The ring is single-producer/single-consumer. The producer calls
//! [`CmdRing::publish`] from one thread; the consumer calls
//! [`CmdRing::take_next`] and then exactly one of [`Claim::commit`] or
//! [`Claim::fail`] from one other thread. All shared state lives behind a
//! `std::sync::Mutex`, so the type is `Send + Sync` and needs no `unsafe`.
//! While a [`Claim`] is outstanding, `take_next` returns `Ok(None)`; the
//! consumer must resolve the claim before taking the next record. Dropping a
//! claim without resolving it is equivalent to [`Claim::fail`]: the slot is
//! released but the record is never re-delivered.
//!
//! [`ListLedger`] is independent of the ring and may be called from the same
//! consumer thread. It bounds its executed history and refuses new in-flight
//! lists once that in-flight cap is reached; in-flight entries are never
//! evicted.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::{Mutex, MutexGuard};

pub use crate::wire::ListIdentity;

/// Alignment of every record and of the ring capacity.
pub const RECORD_ALIGN: usize = 64;

/// Fixed width of a record header in bytes.
const HEADER_LEN: usize = 64;

/// Largest payload a single record may carry.
///
/// This comfortably exceeds the maximum version-2 wire submission (41,504
/// bytes), so the ring never rejects a frame the codec accepted.
pub const MAX_RECORD_BYTES: usize = 1 << 16;

/// Smallest raw capacity accepted by [`CmdRing::new`].
///
/// One maximum record needs [`MAX_RECORD_BYTES`] payload bytes plus the header,
/// rounded up to [`RECORD_ALIGN`].
pub const MIN_CAPACITY_BYTES: usize = MAX_RECORD_BYTES + RECORD_ALIGN;

/// Maximum number of completed list keys retained by [`ListLedger`].
pub const LEDGER_EXECUTED_CAPACITY: usize = 4096;

/// Maximum number of concurrently in-flight list keys in [`ListLedger`].
pub const LEDGER_INFLIGHT_CAPACITY: usize = 1024;

const RECORD_MAGIC: u32 = u32::from_le_bytes(*b"VMCL");
const PADDING_MAGIC: u32 = u32::from_le_bytes(*b"VMPD");
const HEADER_VERSION: u16 = 1;
const COMMIT_MARKER: u64 = u64::from_le_bytes(*b"COMMITM1");

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 4;
const OFF_FLAGS: usize = 6;
const OFF_TOTAL: usize = 8;
const OFF_PAYLOAD: usize = 12;
const OFF_CRC: usize = 16;
const OFF_HEADER_RESERVED: usize = 20;
const OFF_LIST_ID: usize = 24;
const OFF_LIST_GENERATION: usize = 32;
const OFF_SEQUENCE: usize = 36;
const OFF_COMMIT: usize = 40;
const OFF_HEADER_TAIL: usize = 48;

/// A published record's position inside the ring.
///
/// The token is informational: it lets a caller observe where the title was
/// written and how many bytes it occupies, and it lets tests corrupt a known
/// payload byte. It is not a capability and does not affect backpressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRef {
    offset: usize,
    record_len: usize,
    payload_len: usize,
}

impl SlotRef {
    /// Byte offset of the record header within the ring.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Total record length including the header and alignment padding.
    pub fn record_len(&self) -> usize {
        self.record_len
    }

    /// Payload length in bytes.
    pub fn payload_len(&self) -> usize {
        self.payload_len
    }
}

/// Reasons a ring or ledger operation was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingError {
    /// The ring cannot take the record without overwriting undrained bytes,
    /// or the ledger's in-flight cap is reached.
    Full,
    /// A record failed structural or checksum validation.
    Corrupt,
    /// The list identity (or exact identity plus sequence) already ran.
    AlreadyExecuted,
    /// The caller supplied a zero identity, zero sequence, empty payload or
    /// oversized payload.
    InvalidRecord,
}

impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "command-list ring or ledger is full"),
            Self::Corrupt => write!(f, "command-list ring record failed validation"),
            Self::AlreadyExecuted => write!(f, "command list already executed"),
            Self::InvalidRecord => write!(f, "invalid command-list ring record"),
        }
    }
}

impl std::error::Error for RingError {}

#[derive(Debug)]
struct RingState {
    bytes: Vec<u8>,
    /// Absolute (not modulo capacity) byte position of the next record.
    write_pos: u64,
    /// Absolute byte position of the next undrained record.
    read_pos: u64,
    /// Total length of the claimed record, held until commit/fail/drop.
    pending: Option<usize>,
    /// Set when an un-skippable malformed header is seen; the ring then
    /// refuses further reads instead of guessing.
    poisoned: bool,
}

/// A single-producer/single-consumer shared-memory command-list ring.
#[derive(Debug)]
pub struct CmdRing {
    state: Mutex<RingState>,
}

impl CmdRing {
    /// Creates a ring with at least `capacity_bytes` bytes, rounded up to
    /// [`RECORD_ALIGN`].
    ///
    /// # Panics
    ///
    /// Panics when `capacity_bytes` is smaller than [`MIN_CAPACITY_BYTES`],
    /// because such a ring cannot hold even one maximum record.
    pub fn new(capacity_bytes: usize) -> Self {
        assert!(
            capacity_bytes >= MIN_CAPACITY_BYTES,
            "command-list ring capacity {capacity_bytes} is smaller than one maximum record ({MIN_CAPACITY_BYTES} bytes)"
        );
        let capacity = align_up(capacity_bytes, RECORD_ALIGN);
        Self {
            state: Mutex::new(RingState {
                bytes: vec![0; capacity],
                write_pos: 0,
                read_pos: 0,
                pending: None,
                poisoned: false,
            }),
        }
    }

    /// Effective capacity in bytes, always a multiple of [`RECORD_ALIGN`].
    pub fn capacity(&self) -> usize {
        self.lock_state().bytes.len()
    }

    /// Publishes `frame` as one command list.
    ///
    /// The caller supplies the identity and submission sequence that the
    /// frame itself carries; the ring repeats them in the record header so the
    /// consumer can validate and de-duplicate without decoding first.
    ///
    /// Returns [`RingError::Full`] without touching the ring when the record
    /// would overwrite a record the consumer has not resolved. A failed
    /// publish leaves every byte of the ring unchanged.
    pub fn publish(
        &self,
        identity: ListIdentity,
        sequence: u32,
        frame: &[u8],
    ) -> Result<SlotRef, RingError> {
        self.publish_inner(identity, sequence, frame, true)
    }

    fn publish_inner(
        &self,
        identity: ListIdentity,
        sequence: u32,
        frame: &[u8],
        commit: bool,
    ) -> Result<SlotRef, RingError> {
        validate_identity(identity)?;
        if sequence == 0 || frame.is_empty() || frame.len() > MAX_RECORD_BYTES {
            return Err(RingError::InvalidRecord);
        }
        let total = align_up(
            HEADER_LEN
                .checked_add(frame.len())
                .ok_or(RingError::InvalidRecord)?,
            RECORD_ALIGN,
        );

        let mut state = self.lock_state();
        if state.poisoned {
            return Err(RingError::Corrupt);
        }

        let capacity = state.bytes.len();
        if total > capacity {
            return Err(RingError::InvalidRecord);
        }

        // An empty ring may start at the origin so one maximum record always
        // fits. Only reset when wrapping would need more than the whole ring.
        if state.write_pos == state.read_pos {
            let offset = (state.write_pos % capacity as u64) as usize;
            if offset != 0 {
                let padding = capacity - offset;
                if padding + total > capacity {
                    state.write_pos = 0;
                    state.read_pos = 0;
                }
            }
        }

        let used = state.write_pos - state.read_pos;
        let free = capacity as u64 - used;
        let offset = (state.write_pos % capacity as u64) as usize;
        let needs_padding = offset + total > capacity;
        let needed = if needs_padding {
            (capacity - offset) + total
        } else {
            total
        };
        if needed as u64 > free {
            return Err(RingError::Full);
        }

        if needs_padding {
            let padding = capacity - offset;
            write_padding(&mut state.bytes, offset, padding);
            state.write_pos += padding as u64;
        }

        let write_offset = (state.write_pos % capacity as u64) as usize;
        write_record(
            &mut state.bytes,
            write_offset,
            identity,
            sequence,
            frame,
            total,
            commit,
        );
        state.write_pos += total as u64;

        Ok(SlotRef {
            offset: write_offset,
            record_len: total,
            payload_len: frame.len(),
        })
    }

    /// Takes the next committed record, if any.
    ///
    /// Returns `Ok(None)` when the ring is empty, when the next record has not
    /// been committed yet (a torn publish), or when a previous [`Claim`] has
    /// not been resolved. A record whose payload fails CRC is dropped and
    /// reported as [`RingError::Corrupt`]; it is never returned and never
    /// returned again.
    pub fn take_next(&self) -> Result<Option<Claim<'_>>, RingError> {
        let mut state = self.lock_state();
        if state.poisoned {
            return Err(RingError::Corrupt);
        }
        if state.pending.is_some() {
            return Ok(None);
        }

        loop {
            if state.write_pos == state.read_pos {
                return Ok(None);
            }

            let capacity = state.bytes.len();
            let offset = (state.read_pos % capacity as u64) as usize;
            if offset + HEADER_LEN > capacity {
                state.poisoned = true;
                return Err(RingError::Corrupt);
            }

            let magic = read_u32(&state.bytes, offset + OFF_MAGIC);
            if magic == PADDING_MAGIC {
                let total = read_u32(&state.bytes, offset + OFF_TOTAL) as usize;
                if read_u64(&state.bytes, offset + OFF_COMMIT) != COMMIT_MARKER {
                    return Ok(None);
                }
                if total < HEADER_LEN || total % RECORD_ALIGN != 0 || total != capacity - offset {
                    state.poisoned = true;
                    return Err(RingError::Corrupt);
                }
                state.read_pos += total as u64;
                continue;
            }

            if magic != RECORD_MAGIC {
                // The producer writes the magic before the commit marker; a
                // missing magic is an incomplete publish, not a record.
                return Ok(None);
            }
            if read_u64(&state.bytes, offset + OFF_COMMIT) != COMMIT_MARKER {
                return Ok(None);
            }

            let version = read_u16(&state.bytes, offset + OFF_VERSION);
            let flags = read_u16(&state.bytes, offset + OFF_FLAGS);
            let total = read_u32(&state.bytes, offset + OFF_TOTAL) as usize;
            let payload_len = read_u32(&state.bytes, offset + OFF_PAYLOAD) as usize;
            let crc = read_u32(&state.bytes, offset + OFF_CRC);
            let reserved = read_u32(&state.bytes, offset + OFF_HEADER_RESERVED);
            let tail_zero = state.bytes[offset + OFF_HEADER_TAIL..offset + HEADER_LEN]
                .iter()
                .all(|&byte| byte == 0);

            if version != HEADER_VERSION
                || flags != 0
                || reserved != 0
                || !tail_zero
                || total < HEADER_LEN
                || total % RECORD_ALIGN != 0
                || total > capacity
                || offset + total > capacity
                || payload_len == 0
                || payload_len > MAX_RECORD_BYTES
                || total != align_up(HEADER_LEN + payload_len, RECORD_ALIGN)
            {
                state.poisoned = true;
                return Err(RingError::Corrupt);
            }

            let identity = ListIdentity {
                list_id: read_u64(&state.bytes, offset + OFF_LIST_ID),
                list_generation: read_u32(&state.bytes, offset + OFF_LIST_GENERATION),
            };
            let sequence = read_u32(&state.bytes, offset + OFF_SEQUENCE);
            let payload = &state.bytes[offset + HEADER_LEN..offset + HEADER_LEN + payload_len];

            if crc32_iso_hdlc(payload) != crc {
                state.read_pos += total as u64;
                return Err(RingError::Corrupt);
            }
            if identity.list_id == 0 || identity.list_generation == 0 || sequence == 0 {
                state.read_pos += total as u64;
                return Err(RingError::Corrupt);
            }

            let payload = payload.to_vec();
            state.pending = Some(total);
            return Ok(Some(Claim {
                ring: self,
                bytes: payload,
                identity,
                sequence,
                record_len: total,
                resolved: false,
            }));
        }
    }

    fn release_claim(&self, record_len: usize) {
        let mut state = self.lock_state();
        if state.pending == Some(record_len) {
            state.read_pos += record_len as u64;
            state.pending = None;
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, RingState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One record handed to the consumer, owned until it is resolved.
#[derive(Debug)]
pub struct Claim<'a> {
    ring: &'a CmdRing,
    bytes: Vec<u8>,
    identity: ListIdentity,
    sequence: u32,
    record_len: usize,
    resolved: bool,
}

impl Claim<'_> {
    /// The exact frame bytes that were published.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Identity carried in the record header.
    pub fn identity(&self) -> ListIdentity {
        self.identity
    }

    /// Submission sequence carried in the record header.
    pub fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Releases the slot after successful execution.
    ///
    /// The record is never delivered a second time whether it commits or
    /// fails; exactly-once is enforced by the record framing and
    /// [`ListLedger`], not by retrying the ring.
    pub fn commit(mut self) {
        self.resolved = true;
        self.ring.release_claim(self.record_len);
    }

    /// Releases the slot after an execution failure.
    ///
    /// Callers must report the failure instead of falling back to a second
    /// execution of the same list.
    pub fn fail(mut self) {
        self.resolved = true;
        self.ring.release_claim(self.record_len);
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if !self.resolved {
            self.ring.release_claim(self.record_len);
        }
    }
}

type LedgerKey = (u64, u32, u32);
type LedgerIdentity = (u64, u32);

#[derive(Debug, Default)]
struct LedgerState {
    inflight: HashSet<LedgerKey>,
    inflight_identities: HashSet<LedgerIdentity>,
    executed: HashSet<LedgerKey>,
    executed_identities: HashSet<LedgerIdentity>,
    executed_order: VecDeque<LedgerKey>,
}

/// Bounded exactly-once ledger for executed command lists.
///
/// A list is keyed by `(list_id, list_generation, sequence)`. Because the
/// generation is the recording-replay discriminator, `begin` also refuses a
/// list identity that is already in flight or executed with any sequence; a
/// retry of the same recording can therefore never execute twice.
///
/// At most [`LEDGER_EXECUTED_CAPACITY`] completed keys are retained, oldest
/// first out. At most [`LEDGER_INFLIGHT_CAPACITY`] lists may be in flight;
/// beyond that `begin` returns [`RingError::Full`]. In-flight entries are
/// never evicted, so a slow or uncertain execution cannot be silently
/// forgotten.
#[derive(Debug, Default)]
pub struct ListLedger {
    state: Mutex<LedgerState>,
}

impl ListLedger {
    /// Creates an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Claims a list for its single execution.
    ///
    /// Returns [`RingError::AlreadyExecuted`] when the identity is already in
    /// flight or completed, and leaves the ledger unchanged in that case.
    pub fn begin(&self, identity: ListIdentity, sequence: u32) -> Result<(), RingError> {
        validate_identity(identity)?;
        if sequence == 0 {
            return Err(RingError::InvalidRecord);
        }

        let identity_key = (identity.list_id, identity.list_generation);
        let key = (identity.list_id, identity.list_generation, sequence);
        let mut state = self.lock_state();

        if state.inflight.contains(&key)
            || state.executed.contains(&key)
            || state.inflight_identities.contains(&identity_key)
            || state.executed_identities.contains(&identity_key)
        {
            return Err(RingError::AlreadyExecuted);
        }
        if state.inflight.len() >= LEDGER_INFLIGHT_CAPACITY {
            return Err(RingError::Full);
        }

        state.inflight.insert(key);
        state.inflight_identities.insert(identity_key);
        Ok(())
    }

    /// Marks a previously begun list as completed.
    ///
    /// A pair that was not begun is ignored. Completed history is bounded by
    /// [`LEDGER_EXECUTED_CAPACITY`]; eviction only forgets the oldest completed
    /// key, never an in-flight one.
    pub fn complete(&self, identity: ListIdentity, sequence: u32) {
        let identity_key = (identity.list_id, identity.list_generation);
        let key = (identity.list_id, identity.list_generation, sequence);
        let mut state = self.lock_state();

        if !state.inflight.remove(&key) {
            return;
        }
        state.inflight_identities.remove(&identity_key);
        state.executed.insert(key);
        state.executed_identities.insert(identity_key);
        state.executed_order.push_back(key);

        while state.executed_order.len() > LEDGER_EXECUTED_CAPACITY {
            let Some(old_key) = state.executed_order.pop_front() else {
                break;
            };
            state.executed.remove(&old_key);
            let old_identity = (old_key.0, old_key.1);
            let still_executed = state
                .executed
                .iter()
                .any(|key| (key.0, key.1) == old_identity);
            if !still_executed {
                state.executed_identities.remove(&old_identity);
            }
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, LedgerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn validate_identity(identity: ListIdentity) -> Result<(), RingError> {
    if identity.list_id == 0 || identity.list_generation == 0 {
        return Err(RingError::InvalidRecord);
    }
    Ok(())
}

fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

fn write_record(
    bytes: &mut [u8],
    offset: usize,
    identity: ListIdentity,
    sequence: u32,
    payload: &[u8],
    total: usize,
    commit: bool,
) {
    write_u32(bytes, offset + OFF_MAGIC, RECORD_MAGIC);
    write_u16(bytes, offset + OFF_VERSION, HEADER_VERSION);
    write_u16(bytes, offset + OFF_FLAGS, 0);
    write_u32(bytes, offset + OFF_TOTAL, total as u32);
    write_u32(bytes, offset + OFF_PAYLOAD, payload.len() as u32);
    write_u32(bytes, offset + OFF_CRC, crc32_iso_hdlc(payload));
    write_u32(bytes, offset + OFF_HEADER_RESERVED, 0);
    write_u64(bytes, offset + OFF_LIST_ID, identity.list_id);
    write_u32(
        bytes,
        offset + OFF_LIST_GENERATION,
        identity.list_generation,
    );
    write_u32(bytes, offset + OFF_SEQUENCE, sequence);
    write_u64(bytes, offset + OFF_COMMIT, 0);
    bytes[offset + OFF_HEADER_TAIL..offset + HEADER_LEN].fill(0);

    let payload_start = offset + HEADER_LEN;
    let payload_end = payload_start + payload.len();
    bytes[payload_start..payload_end].copy_from_slice(payload);
    bytes[payload_end..offset + total].fill(0);

    // Commit marker last: a reader that misses it treats the record as torn.
    if commit {
        write_u64(bytes, offset + OFF_COMMIT, COMMIT_MARKER);
    }
}

fn write_padding(bytes: &mut [u8], offset: usize, total: usize) {
    write_u32(bytes, offset + OFF_MAGIC, PADDING_MAGIC);
    write_u16(bytes, offset + OFF_VERSION, HEADER_VERSION);
    write_u16(bytes, offset + OFF_FLAGS, 0);
    write_u32(bytes, offset + OFF_TOTAL, total as u32);
    write_u32(bytes, offset + OFF_PAYLOAD, 0);
    write_u32(bytes, offset + OFF_CRC, 0);
    write_u32(bytes, offset + OFF_HEADER_RESERVED, 0);
    write_u64(bytes, offset + OFF_LIST_ID, 0);
    write_u32(bytes, offset + OFF_LIST_GENERATION, 0);
    write_u32(bytes, offset + OFF_SEQUENCE, 0);
    write_u64(bytes, offset + OFF_COMMIT, 0);
    bytes[offset + OFF_HEADER_TAIL..offset + HEADER_LEN].fill(0);
    write_u64(bytes, offset + OFF_COMMIT, COMMIT_MARKER);
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    let mut raw = [0u8; 2];
    raw.copy_from_slice(&bytes[at..at + 2]);
    u16::from_le_bytes(raw)
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(raw)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(raw)
}

fn write_u16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// CRC-32/ISO-HDLC (the standard reflected CRC-32, polynomial `0xEDB88320`).
fn crc32_iso_hdlc(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{
        self, BarrierEntry, BufferBinding, Command, Dispatch1d, MAX_BARRIERS_PER_COMMAND,
        MAX_COMMANDS, ResourceBarrier, Submission,
    };

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

    fn two_command_submission(sequence: u32) -> Submission {
        Submission {
            sequence,
            commands: vec![
                Command::Dispatch1d(dispatch(1, 0)),
                Command::Dispatch1d(dispatch(2, 65_536)),
            ],
        }
    }

    fn frame_for(sequence: u32, identity: ListIdentity) -> Vec<u8> {
        wire::encode_submission_with_list(&two_command_submission(sequence), Some(identity))
            .expect("two-command frame encodes")
    }

    /// A maximum-size version-2 frame: 64 barrier commands of 16 entries.
    fn max_barrier_frame(sequence: u32, identity: ListIdentity) -> Vec<u8> {
        let entry = BarrierEntry {
            allocation_id: 1,
            generation: 1,
            offset: 0,
            length: 4096,
            before_state: 0,
            after_state: 4,
        };
        let barrier = ResourceBarrier {
            entries: vec![entry; MAX_BARRIERS_PER_COMMAND],
        };
        let submission = Submission {
            sequence,
            commands: (0..MAX_COMMANDS)
                .map(|_| Command::ResourceBarrier(barrier.clone()))
                .collect(),
        };
        wire::encode_submission_with_list(&submission, Some(identity)).expect("max frame encodes")
    }

    fn identity(list_id: u64, generation: u32) -> ListIdentity {
        ListIdentity {
            list_id,
            list_generation: generation,
        }
    }

    #[test]
    fn roundtrip_real_wire_frame_preserves_submission_and_identity() {
        let list = identity(9, 2);
        let submission = two_command_submission(1);
        let frame = frame_for(submission.sequence, list);
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);

        let slot = ring
            .publish(list, submission.sequence, &frame)
            .expect("publish");
        assert_eq!(slot.payload_len(), frame.len());

        let claim = ring.take_next().expect("take").expect("record present");
        assert_eq!(claim.identity(), list);
        assert_eq!(claim.sequence(), submission.sequence);
        let (decoded, decoded_list) =
            wire::decode_submission_with_list(claim.bytes()).expect("frame decodes");
        assert_eq!(decoded, submission);
        assert_eq!(decoded_list, Some(list));
        claim.commit();

        assert!(ring.take_next().expect("empty take").is_none());
    }

    #[test]
    fn wrapped_record_larger_than_tail_still_decodes() {
        let first_list = identity(21, 1);
        let second_list = identity(22, 1);
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);
        let capacity = ring.capacity();

        let first = max_barrier_frame(1, first_list);
        assert_eq!(first.len(), 41_504);
        assert!(first.len() <= MAX_RECORD_BYTES);
        let first_slot = ring.publish(first_list, 1, &first).expect("first publish");
        assert_eq!(first_slot.offset(), 0);
        let tail = capacity - first_slot.record_len();
        assert!(tail > 0);

        let first_claim = ring.take_next().expect("take").expect("first record");
        first_claim.commit();

        let second = max_barrier_frame(2, second_list);
        assert!(
            second.len() as usize > tail,
            "second record must be larger than the remaining tail"
        );
        let second_slot = ring
            .publish(second_list, 2, &second)
            .expect("wrapped publish");
        assert_eq!(
            second_slot.offset(),
            0,
            "record should wrap to the start of the ring"
        );

        let claim = ring.take_next().expect("take").expect("wrapped record");
        assert_eq!(claim.identity(), second_list);
        assert_eq!(claim.sequence(), 2);
        let (decoded, decoded_list) =
            wire::decode_submission_with_list(claim.bytes()).expect("wrapped frame decodes");
        assert_eq!(decoded.sequence, 2);
        assert_eq!(decoded.commands.len(), MAX_COMMANDS);
        assert_eq!(decoded_list, Some(second_list));
        claim.commit();

        assert!(ring.take_next().expect("empty take").is_none());
    }

    #[test]
    fn backpressure_is_full_and_leaves_ring_bytes_unchanged() {
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);
        let frame = vec![0xA5u8; 40_000];
        ring.publish(identity(1, 1), 1, &frame)
            .expect("first publish");

        let before = ring.lock_state().bytes.clone();
        let error = ring
            .publish(identity(2, 1), 2, &frame)
            .expect_err("second publish must backpressure");
        assert_eq!(error, RingError::Full);
        let after = ring.lock_state().bytes.clone();
        assert_eq!(before, after, "failed publish mutated the ring");
    }

    #[test]
    fn payload_corruption_is_reported_and_never_delivered_twice() {
        let corrupt_list = identity(31, 1);
        let good_list = identity(32, 1);
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);

        let corrupt = frame_for(1, corrupt_list);
        let slot = ring.publish(corrupt_list, 1, &corrupt).expect("publish");
        {
            let mut state = ring.lock_state();
            state.bytes[slot.offset() + HEADER_LEN] ^= 0xFF;
        }

        let error = ring.take_next().expect_err("corrupt record");
        assert_eq!(error, RingError::Corrupt);

        let good = frame_for(2, good_list);
        ring.publish(good_list, 2, &good).expect("good publish");
        let claim = ring.take_next().expect("take").expect("good record");
        assert_eq!(claim.identity(), good_list);
        assert_eq!(claim.sequence(), 2);
        claim.commit();

        assert!(ring.take_next().expect("empty take").is_none());
    }

    #[test]
    fn replay_of_an_executed_or_inflight_list_is_refused() {
        let ledger = ListLedger::new();
        let list = identity(41, 3);

        assert_eq!(ledger.begin(list, 7), Ok(()));
        assert_eq!(ledger.begin(list, 7), Err(RingError::AlreadyExecuted));
        assert_eq!(
            ledger.begin(list, 8),
            Err(RingError::AlreadyExecuted),
            "a new sequence does not re-execute a known recording"
        );

        ledger.complete(list, 7);
        assert_eq!(ledger.begin(list, 7), Err(RingError::AlreadyExecuted));

        let fresh = identity(42, 1);
        assert_eq!(ledger.begin(fresh, 9), Ok(()));
        ledger.complete(fresh, 9);
        assert_eq!(ledger.begin(fresh, 9), Err(RingError::AlreadyExecuted));
    }

    #[test]
    fn torn_publish_without_commit_marker_is_never_delivered() {
        let list = identity(51, 1);
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);
        let frame = frame_for(1, list);

        let slot = ring
            .publish_inner(list, 1, &frame, false)
            .expect("uncommitted write");
        assert_eq!(
            read_u64(&ring.lock_state().bytes, slot.offset() + OFF_COMMIT),
            0
        );
        assert!(ring.take_next().expect("torn take").is_none());
        assert!(ring.take_next().expect("torn take again").is_none());

        // Completing the marker is what makes the record visible, proving the
        // commit word (not just ring occupancy) gates delivery.
        {
            let mut state = ring.lock_state();
            write_u64(&mut state.bytes, slot.offset() + OFF_COMMIT, COMMIT_MARKER);
        }
        let claim = ring.take_next().expect("take").expect("committed record");
        assert_eq!(claim.identity(), list);
        claim.commit();
    }

    #[test]
    fn two_threads_preserve_order_without_loss_or_duplication() {
        const RECORDS: u32 = 10_000;
        let ring = CmdRing::new(MIN_CAPACITY_BYTES);
        let ring = &ring;

        std::thread::scope(|scope| {
            let producer = scope.spawn(move || {
                for sequence in 1..=RECORDS {
                    let payload = sequence.to_le_bytes();
                    let list = identity(u64::from(sequence), 1);
                    loop {
                        match ring.publish(list, sequence, &payload) {
                            Ok(_) => break,
                            Err(RingError::Full) => std::thread::yield_now(),
                            Err(error) => panic!("publish failed at {sequence}: {error}"),
                        }
                    }
                }
            });

            let consumer = scope.spawn(move || {
                let mut expected = 1u32;
                while expected <= RECORDS {
                    match ring.take_next() {
                        Ok(Some(claim)) => {
                            assert_eq!(claim.sequence(), expected);
                            assert_eq!(claim.identity().list_id, u64::from(expected));
                            let payload = expected.to_le_bytes();
                            assert_eq!(claim.bytes(), &payload[..]);
                            claim.commit();
                            expected += 1;
                        }
                        Ok(None) => std::thread::yield_now(),
                        Err(error) => panic!("take failed at {expected}: {error}"),
                    }
                }
            });

            producer.join().expect("producer finished");
            consumer.join().expect("consumer finished");
        });
    }

    #[test]
    fn capacity_rounds_up_to_record_alignment() {
        let ring = CmdRing::new(MIN_CAPACITY_BYTES + 1);
        assert_eq!(ring.capacity() % RECORD_ALIGN, 0);
        assert!(ring.capacity() >= MIN_CAPACITY_BYTES + 1);
    }

    #[test]
    #[should_panic(expected = "smaller than one maximum record")]
    fn capacity_below_one_maximum_record_is_rejected() {
        let _ = CmdRing::new(MIN_CAPACITY_BYTES - 1);
    }
}
