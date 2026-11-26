//! Headless local IPC host for native ARM64 Metal submissions.
//!
//! This module is the host half of the local transport v1 used by the native
//! renderer boundary. It is deliberately small: authentication, allocation
//! registration and ordered batch submission over loopback TCP. The wire
//! encoding of submissions and replies is [`crate::wire`]; this module only
//! adds the fixed transport envelope, file registration and session state.
//!
//! # Transport v1
//!
//! The listener is bound by the caller to `127.0.0.1` and serves exactly one
//! peer, synchronously. Every frame is a five-byte envelope followed by exactly
//! `payload_len` bytes:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 1 | frame kind |
//! | 1 | 4 | payload length, little-endian `u32`, at most [`MAX_PAYLOAD_LEN`] |
//!
//! Frame kinds:
//!
//! | kind | direction | payload |
//! | ---: | --- | --- |
//! | 1 | guest to host | exactly 32 secret bytes, first frame only |
//! | 2 | guest to host | `allocation_id: u64 LE`, `byte_len: u64 LE` |
//! | 3 | guest to host | one complete encoded [`crate::wire::Submission`] |
//! | 3 | host to guest | one complete encoded [`crate::wire::Reply`] |
//! | 4 | guest to host | `ticket: u64 LE`, `entry_len: u16 LE`, entry bytes |
//! | 128 | host to guest | one byte: 0 = accepted, 1 = rejected |
//! | 129 | host to guest | one byte status, then `pipeline_id: u64 LE` on success |
//!
//! Kind 128 acknowledges authentication and registration only. A rejected
//! authentication or registration frame is followed by connection close. GPU
//! execution outcomes stay in the kind 3 [`crate::wire::Reply`] and do not close
//! the session. Submission sequences start at 1 and increase by exactly one.
//! Any other kind, malformed envelope, wire decode failure or out-of-order
//! sequence is a protocol failure and closes the connection.
//!
//! # Pipeline files
//!
//! Kind 4 is opt-in: it loads the launcher-provided file
//! `<canonical share dir>/<ticket:016x>.metallib` and asks the executor to
//! register `entry` as a compute pipeline. Pipeline files deliberately share
//! the canonical launcher directory with allocation `.bin` files; the fixed
//! suffix keeps the two namespaces disjoint. The ticket must be nonzero and
//! strictly greater than every previously accepted ticket, and the entry name
//! must be a bounded ASCII Metal identifier. Neither a path nor arbitrary bytes
//! cross the wire, so the guest cannot select another file. The file must be a
//! regular file opened without following a final symlink, nonzero and at most
//! [`MAX_METALLIB_BYTES`].
//!
//! The host answers kind 4 with kind 129. Status 0 carries the native pipeline
//! id; status 1 is an explicit rejection (bad ticket or entry, missing or
//! invalid file, or a failed Metal load) and leaves the session open. The
//! loaded pipeline is registered in the same native table used by kind 3
//! submissions.
//!
//! # Allocation files
//!
//! Registration carries no path. For a registration of `id`, the host opens
//! `<canonical share dir>/<id:016x>.bin`, requires a regular file opened without
//! following a final symlink, and requires its size to equal `byte_len`. The
//! share directory is canonicalized once by [`IpcHost::new`]; the guest can
//! never name a path. The launcher owns the directory's privacy and must keep
//! the file stable for the whole session.
//!
//! Every registered file is mapped `MAP_SHARED` read/write and wrapped with
//! `newBufferWithBytesNoCopy`. The mapping length is rounded up to the Metal
//! device page size because Metal requires page-aligned length; the logical
//! `byte_len` remains the exact size from registration and every wire binding
//! is checked against it before GPU submission. Buffers are retained for the
//! whole session: this milestone has no per-allocation release frame.
//!
//! # Ownership and drop order
//!
//! [`FileMapping`] owns the address range and the file descriptor. Its `Drop`
//! calls `munmap` before the descriptor field is closed. [`IpcHost::release_all`]
//! first unregisters every Metal wrapper, and only then drops the mappings, so
//! Metal can never retain a pointer into an unmapped range. If unregistering a
//! wrapper fails, the corresponding mapping is intentionally leaked rather than
//! unmapped: a leaked mapping is safe until process exit, whereas unmapping
//! memory that Metal might still reference is not. `IpcHost` also calls
//! `release_all` from `Drop` as a last-resort teardown.
//!
//! The secret is compared in constant time, is never written to a log or an
//! error message, and is redacted from `Debug`.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;

use crate::abi::{VmBufferId, VmPipelineId};
use crate::cmdlist::{ListLedger, RingError};
use crate::guest_ipc::{MAX_PIPELINE_ENTRY_LEN, valid_pipeline_entry};
use crate::macos::device::DirectMetalTransport;
use crate::transport::MetalError;
use crate::wire::{
    self, BufferBinding, Command, ListIdentity, Reply, ReplyStatus, Submission, WireError,
};

/// Maximum payload length accepted by the transport envelope.
pub const MAX_PAYLOAD_LEN: u32 = 65_536;
/// Frame kind of the first, authentication frame.
pub const KIND_AUTH: u8 = 1;
/// Frame kind for allocation registration.
pub const KIND_REGISTER: u8 = 2;
/// Frame kind for a submission request and its reply.
pub const KIND_SUBMISSION: u8 = 3;
/// Frame kind for an opt-in precompiled pipeline registration.
pub const KIND_PIPELINE: u8 = 4;
/// Frame kind for an authentication or registration acknowledgement.
pub const KIND_ACK: u8 = 128;
/// Frame kind for a pipeline registration reply.
pub const KIND_PIPELINE_REPLY: u8 = 129;
/// Exact length of the shared session secret.
pub const SECRET_LEN: usize = 32;
/// Exact length of a registration payload.
pub const REGISTER_PAYLOAD_LEN: usize = 16;
/// Generation assigned to every registered allocation in transport v1.
pub const BASE_GENERATION: u32 = 1;
/// Fixed ticket and entry-length prefix of a pipeline registration payload.
pub const PIPELINE_HEADER_LEN: usize = 10;
/// Largest `.metallib` accepted from the launcher-owned directory.
pub const MAX_METALLIB_BYTES: u64 = 16 * 1024 * 1024;
/// Status byte of a successful pipeline registration reply.
pub const PIPELINE_REPLY_OK: u8 = 0;
/// Status byte of an explicitly rejected pipeline registration.
pub const PIPELINE_REPLY_REJECTED: u8 = 1;
/// Exact length of a successful pipeline reply: status plus `u64` id.
pub const PIPELINE_REPLY_OK_LEN: usize = 1 + 8;

const ENVELOPE_LEN: usize = 5;
const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Darwin `O_NOFOLLOW`: reject a final path component that is a symlink.
const O_NOFOLLOW: i32 = 0x0100;
/// Darwin `O_NONBLOCK`: a path swapped to a FIFO after the metadata check
/// cannot block the open before the descriptor is revalidated.
const O_NONBLOCK: i32 = 0x0004;
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
/// Darwin `MAP_SHARED`. GPU writes must reach the guest-visible file, so this
/// must not be `MAP_PRIVATE`.
const MAP_SHARED: i32 = 0x0001;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// Errors that end a host session.
///
/// No variant contains the session secret. Authentication failure is reported
/// without the rejected bytes.
#[derive(Debug, Error)]
pub enum HostError {
    /// Socket read, write or timeout failure.
    #[error("socket I/O failed: {0}")]
    Io(#[from] io::Error),
    /// The peer violated the transport protocol; the connection is closed.
    #[error("protocol violation: {0}")]
    Protocol(&'static str),
    /// The first frame did not carry the session secret.
    #[error("authentication failed")]
    Unauthenticated,
    /// A submission frame was not a valid version 1 wire frame.
    #[error("submission wire error: {0}")]
    Wire(#[from] WireError),
    /// A Metal or native file-mapping operation failed.
    #[error("Metal operation failed: {0}")]
    Metal(#[from] MetalError),
    /// The host was configured with an unusable value.
    #[error("invalid host configuration: {0}")]
    Config(&'static str),
}

/// A 32-byte session secret with constant-time comparison and redacted `Debug`.
#[derive(Clone, Copy)]
pub struct HostSecret([u8; SECRET_LEN]);

impl HostSecret {
    /// Wraps exactly 32 secret bytes.
    pub const fn new(bytes: [u8; SECRET_LEN]) -> Self {
        Self(bytes)
    }

    /// Parses exactly 64 hexadecimal characters into a secret.
    pub fn from_hex(text: &str) -> Result<Self, HostError> {
        let text = text.trim();
        if text.len() != SECRET_LEN * 2 {
            return Err(HostError::Config(
                "secret must be exactly 64 hexadecimal characters",
            ));
        }
        let mut bytes = [0u8; SECRET_LEN];
        for (index, slot) in bytes.iter_mut().enumerate() {
            let pair = text
                .as_bytes()
                .get(index * 2..index * 2 + 2)
                .ok_or(HostError::Config("secret hex is truncated"))?;
            let pair = std::str::from_utf8(pair)
                .map_err(|_| HostError::Config("secret hex is not ASCII"))?;
            *slot = u8::from_str_radix(pair, 16)
                .map_err(|_| HostError::Config("secret hex contains a non-hex digit"))?;
        }
        Ok(Self(bytes))
    }

    /// Compares without an early exit so a timing side channel cannot walk the
    /// secret one byte at a time.
    fn matches(&self, candidate: &[u8]) -> bool {
        if candidate.len() != SECRET_LEN {
            return false;
        }
        let mut difference = 0u8;
        for (expected, actual) in self.0.iter().zip(candidate) {
            difference |= expected ^ actual;
        }
        difference == 0
    }
}

impl std::fmt::Debug for HostSecret {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HostSecret(<redacted>)")
    }
}

/// Native operations the host needs from its batch executor.
///
/// [`DirectMetalTransport`] implements this over the existing
/// [`crate::transport::MetalTransport`] methods and
/// [`DirectMetalTransport::submit_batch`]. Tests substitute a CPU-only
/// implementation.
pub trait BatchExecutor {
    /// Required page alignment for zero-copy buffer registration.
    fn page_size(&self) -> Result<u64, MetalError>;
    /// Registers a live mapping as a zero-copy Metal buffer.
    fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError>;
    /// Releases the Metal wrapper, not the backing mapping.
    fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError>;
    /// Executes one decoded submission over the registered allocation snapshot.
    fn submit_batch(
        &self,
        submission: &Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError>;
    /// Loads a precompiled `.metallib` and registers `entry` as a compute
    /// pipeline, returning the native id used by later submissions.
    ///
    /// The default deliberately refuses the request, so pipeline registration
    /// is opt-in for an executor that can validate and load off-line libraries.
    fn load_pipeline(&self, _metallib: &[u8], _entry: &str) -> Result<VmPipelineId, MetalError> {
        Err(MetalError::NotSupported(
            "pipeline registration is not enabled by this executor",
        ))
    }
}

impl BatchExecutor for DirectMetalTransport {
    fn page_size(&self) -> Result<u64, MetalError> {
        Ok(crate::transport::MetalTransport::device_info(self)?.page_size)
    }

    // The pointer is produced only from `FileMapping`, which owns the mapping
    // until the corresponding unregister. The safe wrapper is therefore safe.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        // SAFETY: the caller (`IpcHost`) owns the mapping and retains it until
        // `unregister_window` returns for this handle.
        unsafe { crate::transport::MetalTransport::register_zerocopy(self, ptr, len) }
    }

    fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError> {
        crate::transport::MetalTransport::unregister_buffer(self, id)
    }

    fn submit_batch(
        &self,
        submission: &Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        DirectMetalTransport::submit_batch(self, submission, allocations)
    }

    fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        DirectMetalTransport::load_metallib(self, metallib, entry)
    }
}

/// A shared, read/write file mapping owned until the session ends.
struct FileMapping {
    /// The mapping must be unmapped before this handle is closed; `Drop`
    /// performs that ordering explicitly. The descriptor is intentionally not
    /// read after `open`, but it keeps the mapping's file alive.
    _file: File,
    ptr: usize,
    map_len: usize,
}

impl FileMapping {
    fn open(share_dir: &Path, id: u64, byte_len: u64, page_size: usize) -> Result<Self, HostError> {
        let byte_len = usize::try_from(byte_len)
            .map_err(|_| HostError::Protocol("allocation length does not fit the host"))?;
        if byte_len == 0 {
            return Err(HostError::Protocol("allocation length must be nonzero"));
        }

        let path = share_dir.join(format!("{id:016x}.bin"));
        // Reject a symlink, directory or special file before opening. The
        // single fixed file name cannot escape the canonical share directory.
        let before = std::fs::symlink_metadata(&path)?;
        if !before.file_type().is_file() {
            return Err(HostError::Protocol(
                "allocation backing must be a regular file",
            ));
        }
        if before.len() != byte_len as u64 {
            return Err(HostError::Protocol(
                "allocation backing size does not match registration",
            ));
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(O_NOFOLLOW)
            .open(&path)?;
        // `File::metadata` is an fstat on the opened object, so this closes the
        // check/open race for the final path component.
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(HostError::Protocol(
                "allocation backing is not a regular file",
            ));
        }
        if metadata.len() != byte_len as u64 {
            return Err(HostError::Protocol(
                "allocation backing size does not match registration",
            ));
        }

        let map_len = byte_len
            .checked_add(page_size - 1)
            .ok_or(HostError::Config("allocation mapping length overflows"))?
            / page_size
            * page_size;
        // SAFETY: `file` is open read/write, `map_len` is nonzero and rounded to
        // the page size, and the returned mapping is owned by `FileMapping`
        // until `Drop` unmaps it.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                map_len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error().into());
        }
        if !(ptr as usize).is_multiple_of(page_size) {
            // SAFETY: `ptr` is the live mapping returned by `mmap` above.
            unsafe { munmap(ptr, map_len) };
            return Err(HostError::Config(
                "file mapping is not device-page aligned",
            ));
        }

        Ok(Self {
            _file: file,
            ptr: ptr as usize,
            map_len,
        })
    }

    fn ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    fn map_len(&self) -> usize {
        self.map_len
    }
}

impl Drop for FileMapping {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`map_len` describe the mapping created in `open`; Drop
        // runs once and the file descriptor is still open until after this.
        unsafe { munmap(self.ptr as *mut c_void, self.map_len) };
    }
}

/// Reads `{ticket:016x}.metallib` from the canonical share directory.
///
/// The path is derived only from the validated numeric ticket. The final
/// component must be a regular file and is opened `O_NOFOLLOW`; size and type
/// are rechecked on the open descriptor before reading, so replacing the path
/// with a symlink or growing the file cannot smuggle other bytes in.
fn read_pipeline_file(share_dir: &Path, ticket: u64) -> Result<Vec<u8>, HostError> {
    let path = share_dir.join(format!("{ticket:016x}.metallib"));
    let before = std::fs::symlink_metadata(&path)?;
    if !before.file_type().is_file() {
        return Err(HostError::Protocol(
            "pipeline backing must be a regular file",
        ));
    }
    let announced = before.len();
    if announced == 0 || announced > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol(
            "pipeline backing is empty or exceeds the 16 MiB limit",
        ));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(&path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(HostError::Protocol(
            "pipeline backing is not a regular file",
        ));
    }
    if metadata.len() != announced || metadata.len() > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol(
            "pipeline backing changed size while opening",
        ));
    }

    let mut bytes = Vec::with_capacity(announced as usize);
    let read = (&mut file)
        .take(MAX_METALLIB_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if read as u64 != announced || bytes.is_empty() || bytes.len() as u64 > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol(
            "pipeline backing changed size while reading",
        ));
    }
    Ok(bytes)
}

struct RegisteredAllocation {
    /// Logical length from the registration frame; bindings are checked
    /// against this, not the page-rounded mapping length.
    byte_len: u64,
    buffer_id: VmBufferId,
    /// Kept alive until the Metal wrapper has been unregistered.
    _mapping: FileMapping,
}

/// Summary of one completed session, for launcher diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSummary {
    /// Submission frames answered with a wire reply.
    pub submissions: u64,
    /// Allocations registered during the session.
    pub allocations: usize,
    /// Sum of logical allocation lengths in bytes.
    pub bytes_mapped: u64,
}

/// Synchronous one-peer host session.
///
/// The caller accepts one loopback stream and passes it to
/// [`IpcHost::serve_once`]. The host holds all mappings and Metal wrappers until
/// [`IpcHost::release_all`] or drop.
pub struct IpcHost<E: BatchExecutor> {
    secret: HostSecret,
    share_dir: PathBuf,
    io_timeout: Duration,
    page_size: usize,
    executor: E,
    allocations: BTreeMap<u64, RegisteredAllocation>,
    last_allocation_id: u64,
    last_pipeline_ticket: u64,
    next_sequence: u32,
    submissions: u64,
    ledger: ListLedger,
}

impl<E: BatchExecutor> IpcHost<E> {
    /// Canonicalizes the launcher-controlled share directory and opens the
    /// executor's device page-size requirement.
    pub fn new(
        secret: HostSecret,
        share_dir: impl AsRef<Path>,
        executor: E,
    ) -> Result<Self, HostError> {
        let share_dir = std::fs::canonicalize(share_dir.as_ref())?;
        if !std::fs::metadata(&share_dir)?.is_dir() {
            return Err(HostError::Config("share dir is not a directory"));
        }
        let page_size = usize::try_from(executor.page_size()?)
            .map_err(|_| HostError::Config("device page size does not fit the host"))?;
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(HostError::Config(
                "device page size is zero or not a power of two",
            ));
        }

        Ok(Self {
            secret,
            share_dir,
            io_timeout: DEFAULT_IO_TIMEOUT,
            page_size,
            executor,
            allocations: BTreeMap::new(),
            last_allocation_id: 0,
            last_pipeline_ticket: 0,
            next_sequence: 1,
            submissions: 0,
            ledger: ListLedger::new(),
        })
    }

    /// Overrides the per-read/per-write socket timeout.
    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// Serves exactly one authenticated peer until clean disconnect or a
    /// protocol/execution failure.
    ///
    /// A clean disconnect at an envelope boundary returns
    /// [`SessionSummary`]. Authentication failure sends a rejection
    /// acknowledgement and returns [`HostError::Unauthenticated`]; any other
    /// error closes the stream when it is dropped.
    pub fn serve_once(&mut self, mut stream: TcpStream) -> Result<SessionSummary, HostError> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(self.io_timeout))?;
        stream.set_write_timeout(Some(self.io_timeout))?;

        let Some((kind, payload)) = read_frame(&mut stream)? else {
            return Err(HostError::Protocol(
                "connection closed before authentication",
            ));
        };
        if kind != KIND_AUTH || payload.len() != SECRET_LEN {
            reject_ack(&mut stream);
            return Err(HostError::Protocol(
                "first frame must authenticate with exactly 32 secret bytes",
            ));
        }
        if !self.secret.matches(&payload) {
            reject_ack(&mut stream);
            return Err(HostError::Unauthenticated);
        }
        write_frame(&mut stream, KIND_ACK, &[0])?;

        while let Some((kind, payload)) = read_frame(&mut stream)? {
            match kind {
                KIND_REGISTER => {
                    if let Err(error) = self.register(&payload) {
                        reject_ack(&mut stream);
                        return Err(error);
                    }
                    write_frame(&mut stream, KIND_ACK, &[0])?;
                }
                KIND_SUBMISSION => self.submit(&mut stream, &payload)?,
                KIND_PIPELINE => {
                    let reply = self.register_pipeline(&payload);
                    write_frame(&mut stream, KIND_PIPELINE_REPLY, &reply)?;
                }
                _ => {
                    return Err(HostError::Protocol(
                        "unexpected frame kind after authentication",
                    ));
                }
            }
        }

        Ok(SessionSummary {
            submissions: self.submissions,
            allocations: self.allocations.len(),
            bytes_mapped: self
                .allocations
                .values()
                .map(|allocation| allocation.byte_len)
                .sum(),
        })
    }

    /// Unregisters every Metal wrapper, then drops every mapping.
    ///
    /// Called automatically by `Drop`; exposed so the launcher can release at a
    /// deterministic point before process teardown.
    pub fn release_all(&mut self) {
        for (_, allocation) in std::mem::take(&mut self.allocations) {
            if self.executor.unregister_window(allocation.buffer_id).is_ok() {
                drop(allocation);
            } else {
                // Never unmap a range whose Metal wrapper may still exist. The
                // intentionally leaked mapping is reclaimed by process exit.
                std::mem::forget(allocation);
            }
        }
    }

    fn register(&mut self, payload: &[u8]) -> Result<(), HostError> {
        if payload.len() != REGISTER_PAYLOAD_LEN {
            return Err(HostError::Protocol(
                "registration payload must be exactly 16 bytes",
            ));
        }
        let id = read_u64(payload, 0);
        let byte_len = read_u64(payload, 8);
        if id == 0 {
            return Err(HostError::Protocol("allocation id must be nonzero"));
        }
        if byte_len == 0 {
            return Err(HostError::Protocol("allocation length must be nonzero"));
        }
        if id <= self.last_allocation_id || self.allocations.contains_key(&id) {
            return Err(HostError::Protocol(
                "allocation ids must increase and are never reused",
            ));
        }

        let mapping = FileMapping::open(&self.share_dir, id, byte_len, self.page_size)?;
        let buffer_id = self.executor.register_window(mapping.ptr(), mapping.map_len())?;
        self.allocations.insert(
            id,
            RegisteredAllocation {
                byte_len,
                buffer_id,
                _mapping: mapping,
            },
        );
        self.last_allocation_id = id;
        Ok(())
    }

    fn submit(&mut self, stream: &mut TcpStream, payload: &[u8]) -> Result<(), HostError> {
        let (submission, list) = wire::decode_submission_with_list(payload)?;
        if submission.sequence != self.next_sequence {
            return Err(HostError::Protocol(
                "submission sequence is not the next expected value",
            ));
        }

        let status = self.execute_or_refuse(&submission, list);

        let encoded = wire::encode_reply(&status)?;
        write_frame(stream, KIND_SUBMISSION, &encoded)?;
        self.next_sequence = submission
            .sequence
            .checked_add(1)
            .ok_or(HostError::Protocol("submission sequence exhausted"))?;
        self.submissions += 1;
        Ok(())
    }

    /// Executes one submission, enforcing exactly-once for list-bound batches.
    ///
    /// Host-side binding validation (`UnknownHandle`) runs before the ledger
    /// claim is taken, so a refused batch never consumes a claim. That is the
    /// observable equivalent of claiming and then releasing: no GPU work runs
    /// and the same list may be retried with corrected bindings once it passes
    /// validation. [`ListLedger`] has no separate abort, so only a claim that
    /// reached execution is retained; it is retired with `complete` on success
    /// and deliberately left in flight after an uncertain execution failure so
    /// the list is never executed twice.
    ///
    /// A replay of an executed list is refused with
    /// [`ReplyStatus::AlreadyExecuted`] and performs no GPU work. A submission
    /// without a list identity keeps the legacy execute-every-time behaviour.
    fn execute_or_refuse(&self, submission: &Submission, list: Option<ListIdentity>) -> Reply {
        let response = |status| Reply {
            sequence: submission.sequence,
            status,
        };

        if !self.submission_bindings_valid(submission) {
            return response(ReplyStatus::UnknownHandle);
        }

        let Some(identity) = list else {
            return self.execute_submission(submission);
        };

        match self.ledger.begin(identity, submission.sequence) {
            Ok(()) => {
                let reply = self.execute_submission(submission);
                if reply.status == ReplyStatus::Ok {
                    self.ledger.complete(identity, submission.sequence);
                }
                reply
            }
            Err(RingError::AlreadyExecuted) => response(ReplyStatus::AlreadyExecuted),
            // The ledger refused a claim for another reason (for example its
            // in-flight cap). No work ran and no claim is held.
            Err(_) => response(ReplyStatus::ExecutionFailed),
        }
    }

    fn submission_bindings_valid(&self, submission: &Submission) -> bool {
        submission
            .commands
            .iter()
            .all(|command| self.bindings_within_logical_lengths(command))
    }

    fn execute_submission(&self, submission: &Submission) -> Reply {
        let allocations: HashMap<(u64, u32), VmBufferId> = self
            .allocations
            .iter()
            .map(|(&id, allocation)| ((id, BASE_GENERATION), allocation.buffer_id))
            .collect();

        match self.executor.submit_batch(submission, &allocations) {
            Ok(reply) => reply,
            Err(_) => Reply {
                sequence: submission.sequence,
                status: ReplyStatus::ExecutionFailed,
            },
        }
    }

    fn bindings_within_logical_lengths(&self, command: &Command) -> bool {
        let bindings = match command {
            Command::Dispatch1d(dispatch) => &dispatch.bindings,
            Command::Dispatch3d(dispatch) => &dispatch.bindings,
            // Graphics and state commands are not executable by the compute
            // host yet; refuse them before any range reaches the GPU.
            _ => return false,
        };
        bindings
            .iter()
            .all(|binding| binding_within_allocation(binding, &self.allocations))
    }

    /// Handles one kind 4 request and returns the kind 129 reply payload.
    ///
    /// Every request-level failure is an explicit status-1 reply, not a
    /// connection close: the file contents never reach the error path and the
    /// peer may retry the failed ticket or a later one.
    fn register_pipeline(&mut self, payload: &[u8]) -> Vec<u8> {
        let Some((ticket, entry)) = decode_pipeline_request(payload) else {
            return vec![PIPELINE_REPLY_REJECTED];
        };
        if ticket <= self.last_pipeline_ticket {
            return vec![PIPELINE_REPLY_REJECTED];
        }
        let metallib = match read_pipeline_file(&self.share_dir, ticket) {
            Ok(metallib) => metallib,
            Err(_) => return vec![PIPELINE_REPLY_REJECTED],
        };
        let pipeline = match self.executor.load_pipeline(&metallib, entry) {
            Ok(pipeline) if pipeline.0 != 0 => pipeline,
            _ => return vec![PIPELINE_REPLY_REJECTED],
        };

        self.last_pipeline_ticket = ticket;
        let mut reply = Vec::with_capacity(PIPELINE_REPLY_OK_LEN);
        reply.push(PIPELINE_REPLY_OK);
        reply.extend_from_slice(&pipeline.0.to_le_bytes());
        reply
    }
}

impl<E: BatchExecutor> Drop for IpcHost<E> {
    fn drop(&mut self) {
        // `release_all` only needs the executor and the allocation map.
        self.release_all();
    }
}

fn binding_within_allocation(
    binding: &BufferBinding,
    allocations: &BTreeMap<u64, RegisteredAllocation>,
) -> bool {
    if binding.generation != BASE_GENERATION {
        return false;
    }
    let Some(allocation) = allocations.get(&binding.allocation_id) else {
        return false;
    };
    binding
        .checked_end()
        .is_some_and(|end| end <= allocation.byte_len)
}

fn read_u64(payload: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&payload[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

/// Decodes a kind 4 payload into a ticket and a validated entry name.
fn decode_pipeline_request(payload: &[u8]) -> Option<(u64, &str)> {
    if payload.len() < PIPELINE_HEADER_LEN {
        return None;
    }
    let ticket = read_u64(payload, 0);
    if ticket == 0 {
        return None;
    }
    let entry_len = u16::from_le_bytes([payload[8], payload[9]]) as usize;
    if entry_len == 0
        || entry_len > MAX_PIPELINE_ENTRY_LEN
        || payload.len() != PIPELINE_HEADER_LEN + entry_len
    {
        return None;
    }
    let entry = std::str::from_utf8(&payload[PIPELINE_HEADER_LEN..]).ok()?;
    valid_pipeline_entry(entry).then_some((ticket, entry))
}

fn reject_ack(stream: &mut TcpStream) {
    let _ = write_frame(stream, KIND_ACK, &[1]);
}

/// Reads one envelope plus payload. A clean EOF before the header returns
/// `None`; a partial header or payload is an I/O error.
fn read_frame(stream: &mut TcpStream) -> Result<Option<(u8, Vec<u8>)>, HostError> {
    let Some(header) = read_header(stream)? else {
        return Ok(None);
    };
    let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]);
    if len > MAX_PAYLOAD_LEN {
        return Err(HostError::Protocol(
            "frame payload exceeds the 65536-byte maximum",
        ));
    }
    let mut payload = vec![0u8; len as usize];
    stream.read_exact(&mut payload)?;
    Ok(Some((header[0], payload)))
}

fn read_header(stream: &mut TcpStream) -> Result<Option<[u8; ENVELOPE_LEN]>, HostError> {
    let mut header = [0u8; ENVELOPE_LEN];
    let mut filled = 0usize;
    while filled < ENVELOPE_LEN {
        match stream.read(&mut header[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => {
                return Err(HostError::Protocol(
                    "connection closed in the middle of an envelope",
                ));
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(Some(header))
}

/// Writes one exact envelope and payload, then flushes.
fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) -> Result<(), HostError> {
    let len = u32::try_from(payload.len())
        .map_err(|_| HostError::Protocol("frame payload does not fit u32"))?;
    if len > MAX_PAYLOAD_LEN {
        return Err(HostError::Protocol(
            "frame payload exceeds the 65536-byte maximum",
        ));
    }
    let mut header = [0u8; ENVELOPE_LEN];
    header[0] = kind;
    header[1..].copy_from_slice(&len.to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::wire::{BufferBinding, Command, Dispatch1d, Submission};

    const SECRET: [u8; SECRET_LEN] = [0x5a; SECRET_LEN];
    const ALLOCATION_ID: u64 = 1;
    const BYTES: usize = 65_536;
    const PIPELINE_ID: u64 = 7;

    #[derive(Default)]
    struct StubState {
        next_buffer: u64,
        next_pipeline: u64,
        live: HashMap<u64, (usize, usize)>,
        unregistered: Vec<u64>,
        submissions: Vec<Submission>,
        pipelines: Vec<(Vec<u8>, String)>,
        write_marker: Option<u8>,
        /// When set, `submit_batch` writes the execution ordinal into the first
        /// byte of every bound allocation, making re-execution observable in
        /// the shared mapping.
        exec_writes: bool,
        executions: u64,
    }

    struct StubExecutor {
        page_size: u64,
        state: Mutex<StubState>,
    }

    impl StubExecutor {
        fn new(page_size: u64) -> Self {
            Self {
                page_size,
                state: Mutex::new(StubState {
                    next_buffer: 1,
                    next_pipeline: 100,
                    ..StubState::default()
                }),
            }
        }

        fn state(&self) -> MutexGuard<'_, StubState> {
            self.state.lock().expect("stub state mutex")
        }
    }

    impl BatchExecutor for StubExecutor {
        fn page_size(&self) -> Result<u64, MetalError> {
            Ok(self.page_size)
        }

        fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
            let mut state = self.state();
            let id = state.next_buffer;
            state.next_buffer += 1;
            state.live.insert(id, (ptr as usize, len));
            if let Some(marker) = state.write_marker {
                // SAFETY: the host owns a mapping of `len >= 1` bytes at `ptr`.
                unsafe { ptr.write(marker) };
            }
            Ok(VmBufferId(id))
        }

        fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError> {
            let mut state = self.state();
            let Some((ptr, _len)) = state.live.remove(&id.0) else {
                return Err(MetalError::NotFound(format!("buffer {}", id.0)));
            };
            // If the host had already unmapped this range, this read would fault.
            // SAFETY: the host retains the mapping until unregister returns.
            unsafe {
                std::ptr::read_volatile(ptr as *const u8);
            }
            state.unregistered.push(id.0);
            Ok(())
        }

        fn submit_batch(
            &self,
            submission: &Submission,
            allocations: &HashMap<(u64, u32), VmBufferId>,
        ) -> Result<Reply, MetalError> {
            let mut state = self.state();
            state.submissions.push(submission.clone());
            if state.exec_writes {
                state.executions += 1;
                let ordinal = u8::try_from(state.executions).unwrap_or(u8::MAX);
                for command in &submission.commands {
                    let bindings = match command {
                        Command::Dispatch1d(dispatch) => &dispatch.bindings,
                        Command::Dispatch3d(dispatch) => &dispatch.bindings,
                        _ => continue,
                    };
                    for binding in bindings {
                        let Some(buffer) = allocations
                            .get(&(binding.allocation_id, binding.generation))
                        else {
                            continue;
                        };
                        let Some((ptr, len)) = state.live.get(&buffer.0) else {
                            continue;
                        };
                        let Ok(offset) = usize::try_from(binding.offset) else {
                            continue;
                        };
                        if offset < *len {
                            // SAFETY: the host owns a live mapping of `len`
                            // bytes at `ptr` and retains it until unregister.
                            unsafe { (*ptr as *mut u8).add(offset).write(ordinal) };
                        }
                    }
                }
            }
            Ok(Reply {
                sequence: submission.sequence,
                status: ReplyStatus::Ok,
            })
        }

        fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
            let mut state = self.state();
            if entry == "vm_fail" {
                return Err(MetalError::CompileFailed(
                    "synthetic pipeline failure".into(),
                ));
            }
            let id = state.next_pipeline;
            state.next_pipeline += 1;
            state.pipelines.push((metallib.to_vec(), entry.to_string()));
            Ok(VmPipelineId(id))
        }
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "victory-ipc-host-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).expect("create test dir");
            Self(path)
        }

        fn write_allocation(&self, id: u64, bytes: &[u8]) {
            std::fs::write(self.0.join(format!("{id:016x}.bin")), bytes)
                .expect("write allocation file");
        }

        fn read_allocation(&self, id: u64) -> Vec<u8> {
            std::fs::read(self.0.join(format!("{id:016x}.bin"))).expect("read allocation file")
        }

        fn write_pipeline(&self, ticket: u64, bytes: &[u8]) {
            std::fs::write(self.0.join(format!("{ticket:016x}.metallib")), bytes)
                .expect("write pipeline file");
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct Server {
        addr: std::net::SocketAddr,
        handle: thread::JoinHandle<(Result<SessionSummary, HostError>, IpcHost<StubExecutor>)>,
    }

    fn start(mut host: IpcHost<StubExecutor>) -> Server {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
        let addr = listener.local_addr().expect("test listener address");
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept test peer");
            let result = host.serve_once(stream);
            (result, host)
        });
        Server { addr, handle }
    }

    fn connect(server: &Server) -> TcpStream {
        let stream = TcpStream::connect(server.addr).expect("connect test peer");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("client read timeout");
        stream
    }

    fn authenticate(client: &mut TcpStream) {
        write_frame(client, KIND_AUTH, &SECRET).expect("send auth");
        assert_eq!(
            read_frame(client).expect("read auth ack"),
            Some((KIND_ACK, vec![0]))
        );
    }

    fn expect_ack(client: &mut TcpStream, accepted: bool) {
        let expected = if accepted { 0 } else { 1 };
        assert_eq!(
            read_frame(client).expect("read ack"),
            Some((KIND_ACK, vec![expected]))
        );
    }

    fn registration(id: u64, len: u64) -> [u8; REGISTER_PAYLOAD_LEN] {
        let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&id.to_le_bytes());
        payload[8..].copy_from_slice(&len.to_le_bytes());
        payload
    }

    fn pipeline_request(ticket: u64, entry: &str) -> Vec<u8> {
        let mut payload = Vec::with_capacity(PIPELINE_HEADER_LEN + entry.len());
        payload.extend_from_slice(&ticket.to_le_bytes());
        payload.extend_from_slice(&(entry.len() as u16).to_le_bytes());
        payload.extend_from_slice(entry.as_bytes());
        payload
    }

    fn read_pipeline_reply(client: &mut TcpStream) -> (u8, Option<u64>) {
        let (kind, payload) = read_frame(client)
            .expect("read pipeline reply")
            .expect("pipeline reply frame");
        assert_eq!(kind, KIND_PIPELINE_REPLY);
        match payload.first().copied() {
            Some(PIPELINE_REPLY_OK) => {
                assert_eq!(payload.len(), PIPELINE_REPLY_OK_LEN);
                let mut id = [0u8; 8];
                id.copy_from_slice(&payload[1..]);
                (PIPELINE_REPLY_OK, Some(u64::from_le_bytes(id)))
            }
            Some(PIPELINE_REPLY_REJECTED) => {
                assert_eq!(payload.len(), 1);
                (PIPELINE_REPLY_REJECTED, None)
            }
            other => panic!("unexpected pipeline reply: {other:?} / {payload:?}"),
        }
    }

    fn submission(sequence: u32, binding: BufferBinding) -> Submission {
        Submission {
            sequence,
            commands: vec![Command::Dispatch1d(Dispatch1d {
                pipeline_id: PIPELINE_ID,
                threads: (BYTES / 4) as u32,
                threads_per_group: 64,
                bindings: vec![binding],
            })],
        }
    }

    fn full_binding(length: u64) -> BufferBinding {
        BufferBinding {
            allocation_id: ALLOCATION_ID,
            generation: BASE_GENERATION,
            offset: 0,
            length,
        }
    }

    fn list_identity(list_id: u64, list_generation: u32) -> ListIdentity {
        ListIdentity {
            list_id,
            list_generation,
        }
    }

    /// Sends one submission frame and returns the decoded kind-3 reply.
    fn roundtrip_submission(
        client: &mut TcpStream,
        submission: &Submission,
        list: Option<ListIdentity>,
    ) -> Reply {
        let encoded = wire::encode_submission_with_list(submission, list)
            .expect("encode list submission");
        write_frame(client, KIND_SUBMISSION, &encoded).expect("send submission");
        let (kind, payload) = read_frame(client)
            .expect("read submission reply")
            .expect("submission reply frame");
        assert_eq!(kind, KIND_SUBMISSION);
        wire::decode_reply(&payload).expect("decode submission reply")
    }

    fn host(dir: &TestDir, executor: StubExecutor) -> IpcHost<StubExecutor> {
        IpcHost::new(HostSecret::new(SECRET), &dir.0, executor).expect("create host")
    }

    #[test]
    fn secret_hex_parsing_is_exact_and_redacted() {
        let text = "00".repeat(SECRET_LEN);
        assert!(HostSecret::from_hex(&text).is_ok());
        assert!(HostSecret::from_hex(&text[..text.len() - 2]).is_err());
        assert!(HostSecret::from_hex(&format!("{text}00")).is_err());
        assert!(HostSecret::from_hex(&"zz".repeat(SECRET_LEN)).is_err());
        assert_eq!(
            format!("{:?}", HostSecret::new(SECRET)),
            "HostSecret(<redacted>)"
        );
    }

    #[test]
    fn secret_comparison_requires_all_32_bytes() {
        let secret = HostSecret::new(SECRET);
        assert!(secret.matches(&SECRET));
        for index in 0..SECRET_LEN {
            let mut wrong = SECRET;
            wrong[index] ^= 1;
            assert!(!secret.matches(&wrong), "byte {index} ignored");
        }
        assert!(!secret.matches(&SECRET[..SECRET_LEN - 1]));
        assert!(!secret.matches(&[0u8; SECRET_LEN]));
    }

    #[test]
    fn oversized_envelope_is_rejected_before_payload_read() {
        let dir = TestDir::new();
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);

        let mut header = [0u8; ENVELOPE_LEN];
        header[0] = KIND_REGISTER;
        header[1..].copy_from_slice(&(MAX_PAYLOAD_LEN + 1).to_le_bytes());
        client.write_all(&header).expect("send oversized header");
        // No payload follows: the host must reject from the header alone.

        let (result, _host) = server.handle.join().expect("server thread");
        assert!(matches!(result, Err(HostError::Protocol(_))));
    }

    #[test]
    fn wrong_secret_gets_failure_ack_then_close() {
        let dir = TestDir::new();
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);

        let mut wrong = SECRET;
        wrong[0] ^= 1;
        write_frame(&mut client, KIND_AUTH, &wrong).expect("send wrong auth");
        expect_ack(&mut client, false);
        assert_eq!(read_frame(&mut client).expect("read close"), None);

        let (result, _host) = server.handle.join().expect("server thread");
        assert!(matches!(result, Err(HostError::Unauthenticated)));
    }

    #[test]
    fn auth_register_and_submission_roundtrip_writes_shared_mapping() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().write_marker = Some(0xa5);
        let server = start(host(&dir, executor));
        let mut client = connect(&server);

        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let request = submission(1, full_binding(BYTES as u64));
        let wire_request = wire::encode_submission(&request).expect("encode submission");
        write_frame(&mut client, KIND_SUBMISSION, &wire_request).expect("send submission");

        let (kind, payload) = read_frame(&mut client)
            .expect("read reply")
            .expect("reply frame");
        assert_eq!(kind, KIND_SUBMISSION);
        let reply = wire::decode_reply(&payload).expect("decode reply");
        assert_eq!(
            reply,
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, mut host) = server.handle.join().expect("server thread");
        let summary = result.expect("session succeeds");
        assert_eq!(
            summary,
            SessionSummary {
                submissions: 1,
                allocations: 1,
                bytes_mapped: BYTES as u64,
            }
        );
        assert_eq!(host.executor.state().submissions, vec![request]);
        host.release_all();
        let state = host.executor.state();
        assert_eq!(state.unregistered, vec![1]);
        assert!(state.live.is_empty());
        drop(state);

        // MAP_SHARED writes made through the Metal-facing pointer reach the file.
        let backed = dir.read_allocation(ALLOCATION_ID);
        assert_eq!(backed[0], 0xa5);
        assert!(backed[1..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn register_rejects_zero_duplicate_and_non_monotonic_ids() {
        // Zero id.
        {
            let dir = TestDir::new();
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(0, 4096)).expect("send register");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }

        // Duplicate and decreasing ids in one session.
        {
            let dir = TestDir::new();
            dir.write_allocation(2, &[0u8; 4096]);
            dir.write_allocation(1, &[0u8; 4096]);
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(2, 4096)).expect("register 2");
            expect_ack(&mut client, true);
            write_frame(&mut client, KIND_REGISTER, &registration(2, 4096)).expect("duplicate 2");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }

        {
            let dir = TestDir::new();
            dir.write_allocation(2, &[0u8; 4096]);
            dir.write_allocation(1, &[0u8; 4096]);
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(2, 4096)).expect("register 2");
            expect_ack(&mut client, true);
            write_frame(&mut client, KIND_REGISTER, &registration(1, 4096)).expect("register 1");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }
    }

    #[test]
    fn register_requires_exact_regular_file_size() {
        // Missing file.
        {
            let dir = TestDir::new();
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(1, 4096)).expect("register");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Io(_))));
        }

        // Wrong size.
        {
            let dir = TestDir::new();
            dir.write_allocation(1, &[0u8; 4095]);
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(1, 4096)).expect("register");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }

        // Directory with the expected name.
        {
            let dir = TestDir::new();
            std::fs::create_dir(dir.0.join("0000000000000001.bin")).expect("create dir");
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(1, 4096)).expect("register");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }

        // Zero length payload field.
        {
            let dir = TestDir::new();
            let server = start(host(&dir, StubExecutor::new(4096)));
            let mut client = connect(&server);
            authenticate(&mut client);
            write_frame(&mut client, KIND_REGISTER, &registration(1, 0)).expect("register");
            expect_ack(&mut client, false);
            let (result, _host) = server.handle.join().expect("server thread");
            assert!(matches!(result, Err(HostError::Protocol(_))));
        }
    }

    #[test]
    fn sequence_and_logical_binding_length_are_enforced() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("register");
        expect_ack(&mut client, true);

        // One byte past the exact file length is rejected without GPU work even
        // though the page-rounded Metal registration is larger.
        let too_long = submission(1, full_binding(BYTES as u64 + 1));
        write_frame(
            &mut client,
            KIND_SUBMISSION,
            &wire::encode_submission(&too_long).expect("encode too-long submission"),
        )
        .expect("send too-long submission");
        let (kind, payload) = read_frame(&mut client)
            .expect("read too-long reply")
            .expect("too-long reply frame");
        assert_eq!(kind, KIND_SUBMISSION);
        assert_eq!(
            wire::decode_reply(&payload).expect("decode too-long reply"),
            Reply {
                sequence: 1,
                status: ReplyStatus::UnknownHandle,
            }
        );

        let valid = submission(2, full_binding(BYTES as u64));
        write_frame(
            &mut client,
            KIND_SUBMISSION,
            &wire::encode_submission(&valid).expect("encode valid submission"),
        )
        .expect("send valid submission");
        let (kind, payload) = read_frame(&mut client)
            .expect("read valid reply")
            .expect("valid reply frame");
        assert_eq!(kind, KIND_SUBMISSION);
        assert_eq!(
            wire::decode_reply(&payload).expect("decode valid reply"),
            Reply {
                sequence: 2,
                status: ReplyStatus::Ok,
            }
        );

        // Skipping sequence 3 is a protocol failure: close, no reply.
        let skipped = submission(4, full_binding(BYTES as u64));
        write_frame(
            &mut client,
            KIND_SUBMISSION,
            &wire::encode_submission(&skipped).expect("encode skipped submission"),
        )
        .expect("send skipped submission");
        assert_eq!(read_frame(&mut client).expect("read after skip"), None);

        let (result, host) = server.handle.join().expect("server thread");
        assert!(matches!(result, Err(HostError::Protocol(_))));
        let state = host.executor.state();
        assert_eq!(state.submissions.len(), 1, "rejected batch reached GPU");
        assert_eq!(state.submissions[0].sequence, 2);
    }

    #[test]
    fn replayed_list_is_refused_and_leaves_buffer_bytes_unchanged() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().exec_writes = true;
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let identity = list_identity(10, 1);
        let first = submission(1, full_binding(BYTES as u64));
        let reply = roundtrip_submission(&mut client, &first, Some(identity));
        assert_eq!(
            reply,
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );
        let after_first = dir.read_allocation(ALLOCATION_ID);
        assert_eq!(after_first[0], 1, "first execution wrote ordinal 1");
        assert!(after_first[1..].iter().all(|&byte| byte == 0));

        // Same list identity, later sequence: the host must refuse the replay
        // and perform no GPU work at all.
        let replay = submission(2, full_binding(BYTES as u64));
        let reply = roundtrip_submission(&mut client, &replay, Some(identity));
        assert_eq!(
            reply,
            Reply {
                sequence: 2,
                status: ReplyStatus::AlreadyExecuted,
            }
        );
        let after_replay = dir.read_allocation(ALLOCATION_ID);
        assert_eq!(
            after_replay, after_first,
            "replay rewrote the GPU-visible buffer"
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        let summary = result.expect("session succeeds");
        assert_eq!(
            summary,
            SessionSummary {
                submissions: 2,
                allocations: 1,
                bytes_mapped: BYTES as u64,
            }
        );
        assert_eq!(host.next_sequence, 3, "one sequence per submission");
        let state = host.executor.state();
        assert_eq!(state.submissions, vec![first], "replay reached the executor");
        assert_eq!(state.executions, 1, "replay executed GPU work");
    }

    #[test]
    fn validation_refusal_leaves_the_list_retryable() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().exec_writes = true;
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let identity = list_identity(11, 1);
        // A stale allocation generation is refused before any claim is taken.
        let stale = submission(
            1,
            BufferBinding {
                allocation_id: ALLOCATION_ID,
                generation: BASE_GENERATION + 1,
                offset: 0,
                length: BYTES as u64,
            },
        );
        let reply = roundtrip_submission(&mut client, &stale, Some(identity));
        assert_eq!(
            reply,
            Reply {
                sequence: 1,
                status: ReplyStatus::UnknownHandle,
            }
        );

        // The same list identity with the corrected generation is not blocked
        // by a leftover in-flight claim.
        let corrected = submission(2, full_binding(BYTES as u64));
        let reply = roundtrip_submission(&mut client, &corrected, Some(identity));
        assert_eq!(
            reply,
            Reply {
                sequence: 2,
                status: ReplyStatus::Ok,
            }
        );
        let backed = dir.read_allocation(ALLOCATION_ID);
        assert_eq!(backed[0], 1, "corrected retry never executed");

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(host.next_sequence, 3, "one sequence per submission");
        let state = host.executor.state();
        assert_eq!(state.submissions.len(), 1, "refused list reached the executor");
        assert_eq!(state.submissions[0].sequence, 2);
        assert_eq!(state.executions, 1);
    }

    #[test]
    fn identity_free_submissions_keep_legacy_execute_every_time() {
        // Documented legacy behaviour: a submission without a list identity is
        // executed on every frame. Exactly-once applies only to lists.
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().exec_writes = true;
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        for sequence in 1..=2 {
            let request = submission(sequence, full_binding(BYTES as u64));
            let reply = roundtrip_submission(&mut client, &request, None);
            assert_eq!(
                reply,
                Reply {
                    sequence,
                    status: ReplyStatus::Ok,
                }
            );
        }
        let backed = dir.read_allocation(ALLOCATION_ID);
        assert_eq!(backed[0], 2, "legacy submission must execute twice");

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(host.next_sequence, 3, "one sequence per submission");
        assert_eq!(host.executor.state().submissions.len(), 2);
        assert_eq!(host.executor.state().executions, 2);
    }

    #[test]
    fn session_sequence_advances_once_per_submission_in_every_case() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let list_a = list_identity(21, 1);
        let list_b = list_identity(22, 1);
        let cases = [
            (1, Some(list_a), full_binding(BYTES as u64), ReplyStatus::Ok),
            (
                2,
                Some(list_a),
                full_binding(BYTES as u64),
                ReplyStatus::AlreadyExecuted,
            ),
            (
                3,
                Some(list_b),
                BufferBinding {
                    allocation_id: ALLOCATION_ID,
                    generation: BASE_GENERATION + 1,
                    offset: 0,
                    length: BYTES as u64,
                },
                ReplyStatus::UnknownHandle,
            ),
            (4, Some(list_b), full_binding(BYTES as u64), ReplyStatus::Ok),
            (5, None, full_binding(BYTES as u64), ReplyStatus::Ok),
        ];
        for (sequence, identity, binding, status) in cases {
            let reply = roundtrip_submission(&mut client, &submission(sequence, binding), identity);
            assert_eq!(reply, Reply { sequence, status }, "sequence {sequence}");
        }

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(host.next_sequence, 6, "exactly one sequence per submission");
    }

    #[test]
    fn pipeline_registration_returns_native_id_and_preserves_old_frames() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        dir.write_pipeline(5, b"synthetic metallib bytes");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        write_frame(
            &mut client,
            KIND_PIPELINE,
            &pipeline_request(5, "vm_test_affine"),
        )
        .expect("send pipeline request");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(100))
        );

        // Existing registration and submission frames still work in the same
        // session after the new opcode.
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);
        let request = submission(1, full_binding(BYTES as u64));
        write_frame(
            &mut client,
            KIND_SUBMISSION,
            &wire::encode_submission(&request).expect("encode submission"),
        )
        .expect("send submission");
        let (kind, payload) = read_frame(&mut client)
            .expect("read submission reply")
            .expect("submission reply frame");
        assert_eq!(kind, KIND_SUBMISSION);
        assert_eq!(
            wire::decode_reply(&payload).expect("decode reply"),
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert_eq!(
            result.expect("session succeeds"),
            SessionSummary {
                submissions: 1,
                allocations: 1,
                bytes_mapped: BYTES as u64,
            }
        );
        assert_eq!(
            host.executor.state().pipelines,
            vec![(
                b"synthetic metallib bytes".to_vec(),
                "vm_test_affine".to_string()
            )]
        );
    }

    #[test]
    fn pipeline_registration_rejects_bad_inputs_without_silent_loads() {
        let dir = TestDir::new();
        dir.write_pipeline(7, b"valid synthetic metallib");
        dir.write_pipeline(8, b"valid synthetic metallib");
        dir.write_pipeline(99, b"symlink target");
        std::os::unix::fs::symlink(
            dir.0.join("0000000000000063.metallib"),
            dir.0.join("0000000000000006.metallib"),
        )
        .expect("create pipeline symlink");
        let oversized = std::fs::File::create(dir.0.join("0000000000000004.metallib"))
            .expect("create oversized pipeline file");
        oversized
            .set_len(MAX_METALLIB_BYTES + 1)
            .expect("extend oversized pipeline file");

        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        for (label, request) in [
            ("zero ticket", pipeline_request(0, "vm_entry")),
            ("invalid entry", pipeline_request(3, "../vm_entry")),
            ("missing file", pipeline_request(5, "vm_entry")),
            ("oversized file", pipeline_request(4, "vm_entry")),
            ("symlinked file", pipeline_request(6, "vm_entry")),
            ("native load failure", pipeline_request(7, "vm_fail")),
        ] {
            write_frame(&mut client, KIND_PIPELINE, &request).expect("send rejected request");
            assert_eq!(
                read_pipeline_reply(&mut client),
                (PIPELINE_REPLY_REJECTED, None),
                "{label}"
            );
        }

        // A rejected load does not consume the ticket, and the session stays
        // usable when the same file is asked for a valid entry.
        write_frame(&mut client, KIND_PIPELINE, &pipeline_request(7, "vm_entry"))
            .expect("retry pipeline request");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(100))
        );

        for request in [
            pipeline_request(7, "vm_entry"),
            pipeline_request(6, "vm_entry"),
        ] {
            write_frame(&mut client, KIND_PIPELINE, &request).expect("send nonmonotonic request");
            assert_eq!(
                read_pipeline_reply(&mut client),
                (PIPELINE_REPLY_REJECTED, None),
                "ticket must be strictly increasing"
            );
        }

        write_frame(
            &mut client,
            KIND_PIPELINE,
            &pipeline_request(8, "vm_entry_b"),
        )
        .expect("send later pipeline request");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(101))
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(
            host.executor.state().pipelines,
            vec![
                (b"valid synthetic metallib".to_vec(), "vm_entry".to_string()),
                (b"valid synthetic metallib".to_vec(), "vm_entry_b".to_string()),
            ],
            "only tickets 7 and 8 reached the executor"
        );
    }

    #[test]
    fn clean_disconnect_after_auth_is_success() {
        let dir = TestDir::new();
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        drop(client);
        let (result, _host) = server.handle.join().expect("server thread");
        assert_eq!(
            result.expect("clean session"),
            SessionSummary {
                submissions: 0,
                allocations: 0,
                bytes_mapped: 0,
            }
        );
    }

    #[test]
    fn idle_peer_hits_bounded_read_timeout() {
        let dir = TestDir::new();
        let host = host(&dir, StubExecutor::new(4096))
            .with_io_timeout(Duration::from_millis(50));
        let server = start(host);
        let client = connect(&server);
        let (result, _host) = server.handle.join().expect("server thread");
        drop(client);
        match result {
            Err(HostError::Io(error)) => assert!(
                matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ),
                "unexpected timeout error: {error:?}"
            ),
            other => panic!("expected a bounded timeout, got {other:?}"),
        }
    }
}
