//! Synchronous Win64 guest IPC client for the ARM64 Metal renderer.
//!
//! This module owns the guest half of the local transport version 1. It is
//! deliberately transport-only: submission execution, allocation backing and
//! GPU lifetime remain in the host process, while the existing
//! [`crate::wire`] codec supplies the submission and reply payloads.
//!
//! # Transport
//!
//! The host listens on `127.0.0.1` at a port chosen by the launcher. The guest
//! connects only to the IPv4 loopback address and authenticates with a
//! 32-byte secret supplied out of band by the harness. The secret is never
//! logged and is not retained after the hello frame.
//!
//! Every message is a fixed 5-byte envelope followed by exactly the declared
//! payload:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 1 | frame kind |
//! | 1 | 4 | payload length, little-endian |
//!
//! Payloads are capped at [`MAX_FRAME_PAYLOAD`] (65,536 bytes). The reader
//! rejects an oversized length before reserving storage, so a hostile or
//! broken peer cannot make the guest allocate an unbounded buffer. Both
//! directions use exact reads and writes; a short read is a disconnect.
//!
//! # Frame kinds
//!
//! | kind | direction | payload |
//! | ---: | --- | --- |
//! | 1 | guest to host | exactly [`SECRET_LEN`] secret bytes |
//! | 2 | guest to host | `allocation_id: u64 LE`, `byte_len: u64 LE` |
//! | 3 | both | [`crate::wire::encode_submission`] / [`crate::wire::encode_reply`] |
//! | 4 | guest to host | `ticket: u64 LE`, `entry_len: u16 LE`, entry bytes |
//! | 128 | host to guest | exactly one byte: 0 accepted, 1 rejected |
//! | 129 | host to guest | one byte status, then `pipeline_id: u64 LE` on success |
//!
//! Kinds 1 and 2 are acknowledged with kind 128. A rejection, a malformed
//! acknowledgement or any transport failure closes the connection: there is
//! no partial-success state and no retry. Kind 3 execution outcomes stay in
//! the encoded [`crate::wire::Reply`]; `UnknownHandle` and `ExecutionFailed`
//! are valid replies and leave the session open.
//!
//! Kind 4 is the opt-in precompiled pipeline request. The ticket is nonzero and
//! strictly increasing, and the entry must be a bounded ASCII Metal identifier.
//! The host maps the ticket to its own private `{ticket:016x}.metallib`; no path
//! crosses the wire. A status-1 reply is an explicit rejection and leaves the
//! session open, while a malformed kind 129 reply poisons it.
//!
//! # Registration and ordering
//!
//! Registration IDs must be nonzero and strictly increasing, never reused.
//! The version 1 registration is generation 1. This milestone has no release
//! frame: the client retains every [`RegisteredAllocation`] until the session
//! closes, and the host keeps the backing files alive for the same interval.
//!
//! Submission sequences start at 1 and increment by one per frame actually
//! sent. [`GuestIpcClient::submit`] rejects a mismatched sequence locally
//! without touching the wire and without consuming a sequence number. A reply
//! must echo the submission sequence exactly; any other sequence poisons the
//! session.
//!
//! # Timeouts
//!
//! Connecting, reads and writes all carry a caller-supplied bound. A timeout
//! is reported as [`GuestIpcError::Timeout`] and closes the connection, which
//! keeps the synchronous contract from silently waiting forever on a stalled
//! host.

use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4, TcpStream};
use std::time::Duration;

use thiserror::Error;

use crate::abi::VmPipelineId;
use crate::wire::{self, Command, ListIdentity, Reply, Submission, WireError};

/// Frame kind for the first guest frame carrying the authentication secret.
pub const KIND_HELLO: u8 = 1;
/// Frame kind carrying one allocation registration.
pub const KIND_REGISTER: u8 = 2;
/// Frame kind carrying an encoded submission or reply.
pub const KIND_SUBMIT: u8 = 3;
/// Frame kind carrying an opt-in precompiled pipeline registration.
pub const KIND_PIPELINE: u8 = 4;
/// Frame kind for the one-byte authentication or registration acknowledgement.
pub const KIND_ACK: u8 = 128;
/// Frame kind for a pipeline registration reply.
pub const KIND_PIPELINE_REPLY: u8 = 129;

/// Fixed boundary of the version 1 transport envelope.
pub const ENVELOPE_LEN: usize = 5;
/// Largest payload either peer may declare or send.
pub const MAX_FRAME_PAYLOAD: usize = 65_536;
/// Exact size of the hello secret.
pub const SECRET_LEN: usize = 32;
/// Exact encoded size of a registration payload.
pub const REGISTER_PAYLOAD_LEN: usize = 16;
/// Fixed ticket and entry-length prefix of a pipeline registration payload.
pub const PIPELINE_HEADER_LEN: usize = 10;
/// Largest Metal entry-point identifier accepted for pipeline registration.
pub const MAX_PIPELINE_ENTRY_LEN: usize = 256;
/// Status byte of a successful pipeline registration reply.
pub const PIPELINE_REPLY_OK: u8 = 0;
/// Status byte of an explicitly rejected pipeline registration.
pub const PIPELINE_REPLY_REJECTED: u8 = 1;
/// Exact length of a successful pipeline reply: status plus `u64` id.
pub const PIPELINE_REPLY_OK_LEN: usize = 1 + 8;
/// The only allocation generation in transport version 1.
pub const REGISTRATION_GENERATION: u32 = 1;
/// Default bound applied to connecting, each read and each write.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Returns whether `entry` is a bounded, path-free Metal function identifier.
///
/// The accepted grammar is an ASCII letter or underscore followed by ASCII
/// letters, digits or underscores. It excludes path separators, dots, NUL,
/// non-ASCII bytes and over-long names, so it can only name a function inside
/// an already-selected library.
pub fn valid_pipeline_entry(entry: &str) -> bool {
    if entry.is_empty() || entry.len() > MAX_PIPELINE_ENTRY_LEN {
        return false;
    }
    let mut bytes = entry.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// One allocation registered for the lifetime of the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredAllocation {
    /// Nonzero, never reused allocation registry key.
    pub id: u64,
    /// Exact backing file length requested from the host.
    pub byte_len: u64,
    /// Allocation generation, always [`REGISTRATION_GENERATION`] in version 1.
    pub generation: u32,
}

/// Failures of the guest transport.
#[derive(Debug, Error)]
pub enum GuestIpcError {
    /// The caller passed port zero, which cannot identify a listening host.
    #[error("port zero does not identify a host")]
    InvalidPort,
    /// The caller passed a zero I/O timeout, which `std` rejects.
    #[error("I/O timeout must be nonzero")]
    InvalidTimeout,
    /// A bounded connect, read or write did not complete in time.
    #[error("{operation} timed out after {timeout:?}")]
    Timeout {
        /// Operation that timed out.
        operation: &'static str,
        /// Bound that expired.
        timeout: Duration,
    },
    /// The peer closed the connection or a pipe was already broken.
    #[error("peer disconnected during {operation}")]
    Disconnected {
        /// Operation that observed the disconnect.
        operation: &'static str,
    },
    /// Any other socket failure, including a refused connection.
    #[error("I/O failure during {operation}: {source}")]
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Underlying socket error.
        #[source]
        source: std::io::Error,
    },
    /// A declared or supplied payload exceeded [`MAX_FRAME_PAYLOAD`].
    #[error("frame payload length {length} exceeds limit {limit}")]
    FrameTooLarge {
        /// Declared or supplied length.
        length: usize,
        /// Inclusive maximum.
        limit: usize,
    },
    /// A frame did not have the kind the protocol step requires.
    #[error("frame kind {actual} does not match expected {expected}")]
    UnexpectedKind {
        /// Kind required by the protocol step.
        expected: u8,
        /// Kind actually received.
        actual: u8,
    },
    /// A kind 128 acknowledgement did not carry exactly one byte.
    #[error("{operation} acknowledgement carries {length} bytes, expected 1")]
    BadAckLength {
        /// Operation being acknowledged.
        operation: &'static str,
        /// Actual payload length.
        length: usize,
    },
    /// The host answered a kind 128 acknowledgement with failure.
    #[error("host rejected {operation}")]
    Rejected {
        /// Operation rejected by the host.
        operation: &'static str,
    },
    /// A submission did not carry the next expected sequence.
    #[error("submission sequence {actual} does not match next sequence {expected}")]
    SequenceMismatch {
        /// Next sequence the session requires.
        expected: u32,
        /// Sequence supplied by the caller.
        actual: u32,
    },
    /// The full `u32` submission sequence space has been consumed.
    #[error("submission sequence space exhausted")]
    SequenceExhausted,
    /// A kind 3 reply echoed a different sequence.
    #[error("reply sequence {actual} does not match submission sequence {expected}")]
    ReplySequenceMismatch {
        /// Sequence that was sent.
        expected: u32,
        /// Sequence echoed by the host.
        actual: u32,
    },
    /// Registration ID zero is reserved.
    #[error("allocation ID must be nonzero")]
    ZeroAllocationId,
    /// A zero-byte allocation cannot back a wire binding.
    #[error("allocation {0} length must be nonzero")]
    ZeroAllocationLength(u64),
    /// A registration reused or went backwards in the allocation ID space.
    #[error("allocation ID {id} must be greater than previous ID {previous}")]
    NonMonotonicAllocationId {
        /// Most recently registered ID.
        previous: u64,
        /// Rejected ID.
        id: u64,
    },
    /// Pipeline ticket zero is reserved.
    #[error("pipeline ticket must be nonzero")]
    ZeroPipelineTicket,
    /// A pipeline registration reused or went backwards in ticket space.
    #[error("pipeline ticket {ticket} must be greater than previous ticket {previous}")]
    NonMonotonicPipelineTicket {
        /// Most recently accepted ticket.
        previous: u64,
        /// Rejected ticket.
        ticket: u64,
    },
    /// The entry name was empty, too long or not a bounded Metal identifier.
    #[error("pipeline entry must be a bounded ASCII Metal identifier")]
    InvalidPipelineEntry,
    /// A kind 129 reply had a status and length combination the protocol forbids.
    #[error("malformed pipeline reply with {length} payload byte(s)")]
    MalformedPipelineReply {
        /// Actual reply payload length.
        length: usize,
    },
    /// The connection was already poisoned or explicitly closed.
    #[error("IPC session is closed")]
    Closed,
    /// The submission or reply did not satisfy the version 1 wire codec.
    #[error(transparent)]
    Wire(#[from] WireError),
}

/// Whether a `std::io::Error` means the peer went away or a bound expired.
fn classify_io(
    operation: &'static str,
    timeout: Duration,
    source: std::io::Error,
) -> GuestIpcError {
    match source.kind() {
        ErrorKind::UnexpectedEof
        | ErrorKind::ConnectionReset
        | ErrorKind::ConnectionAborted
        | ErrorKind::BrokenPipe
        | ErrorKind::NotConnected => GuestIpcError::Disconnected { operation },
        ErrorKind::WouldBlock | ErrorKind::TimedOut => {
            GuestIpcError::Timeout { operation, timeout }
        }
        _ => GuestIpcError::Io { operation, source },
    }
}

/// One synchronous, authenticated guest session.
///
/// Construction performs the hello exchange, so a client that exists has
/// already been accepted by the host. The client owns the socket and
/// registration records and closes the connection on drop.
pub struct GuestIpcClient {
    stream: TcpStream,
    timeout: Duration,
    next_sequence: u32,
    allocations: Vec<RegisteredAllocation>,
    last_pipeline_ticket: u64,
    closed: bool,
}

impl GuestIpcClient {
    /// Connects to `127.0.0.1:port` and authenticates with `secret`.
    ///
    /// Uses [`DEFAULT_IO_TIMEOUT`] for connecting and every later read/write.
    pub fn connect(port: u16, secret: &[u8; SECRET_LEN]) -> Result<Self, GuestIpcError> {
        Self::connect_with_timeout(port, secret, DEFAULT_IO_TIMEOUT)
    }

    /// Connects to `127.0.0.1:port` with explicit bounded timeouts.
    ///
    /// The hello frame is the first bytes on the wire; the returned client is
    /// only constructed after the host acknowledges it. A rejection or
    /// malformed acknowledgement closes the socket and returns an error.
    pub fn connect_with_timeout(
        port: u16,
        secret: &[u8; SECRET_LEN],
        timeout: Duration,
    ) -> Result<Self, GuestIpcError> {
        if port == 0 {
            return Err(GuestIpcError::InvalidPort);
        }
        if timeout.is_zero() {
            return Err(GuestIpcError::InvalidTimeout);
        }

        let address = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
        let stream = TcpStream::connect_timeout(&address, timeout)
            .map_err(|source| classify_io("connect", timeout, source))?;
        stream
            .set_nodelay(true)
            .map_err(|source| classify_io("configure socket", timeout, source))?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|source| classify_io("configure socket", timeout, source))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|source| classify_io("configure socket", timeout, source))?;

        let mut client = Self {
            stream,
            timeout,
            next_sequence: 1,
            allocations: Vec::new(),
            last_pipeline_ticket: 0,
            closed: false,
        };
        client.hello(secret)?;
        Ok(client)
    }

    /// Sends the version 1 hello frame and requires a successful acknowledgement.
    fn hello(&mut self, secret: &[u8; SECRET_LEN]) -> Result<(), GuestIpcError> {
        // Copy into the frame buffer, send, then clear the local copy. The
        // secret is not retained by the client.
        let mut payload = *secret;
        let sent = self.write_frame(KIND_HELLO, &payload, "hello");
        payload.fill(0);
        sent?;
        self.read_ack("hello")
    }

    /// Registers one file-backed allocation with the host.
    ///
    /// The host derives the backing path from this ID and `byte_len`; the
    /// guest never sends a path. IDs must be nonzero and strictly increasing.
    /// The record is retained until the session closes.
    pub fn register(
        &mut self,
        allocation_id: u64,
        byte_len: u64,
    ) -> Result<RegisteredAllocation, GuestIpcError> {
        self.ensure_open()?;
        if allocation_id == 0 {
            return Err(GuestIpcError::ZeroAllocationId);
        }
        if byte_len == 0 {
            return Err(GuestIpcError::ZeroAllocationLength(allocation_id));
        }
        if let Some(previous) = self.allocations.last()
            && allocation_id <= previous.id
        {
            return Err(GuestIpcError::NonMonotonicAllocationId {
                previous: previous.id,
                id: allocation_id,
            });
        }

        let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&allocation_id.to_le_bytes());
        payload[8..16].copy_from_slice(&byte_len.to_le_bytes());
        self.write_frame(KIND_REGISTER, &payload, "register")?;
        self.read_ack("register")?;

        let allocation = RegisteredAllocation {
            id: allocation_id,
            byte_len,
            generation: REGISTRATION_GENERATION,
        };
        self.allocations.push(allocation);
        Ok(allocation)
    }

    /// Registers a precompiled library in the host's private directory.
    ///
    /// The host maps `ticket` to `{ticket:016x}.metallib`; no path or library
    /// bytes cross the wire. Tickets must be nonzero and strictly increasing,
    /// and `entry` must satisfy [`valid_pipeline_entry`]. On success the host's
    /// native pipeline id is returned and can be used in later submissions.
    /// An explicit host rejection leaves the session open; a malformed reply
    /// poisons it.
    pub fn register_pipeline(
        &mut self,
        ticket: u64,
        entry: &str,
    ) -> Result<VmPipelineId, GuestIpcError> {
        self.ensure_open()?;
        if ticket == 0 {
            return Err(GuestIpcError::ZeroPipelineTicket);
        }
        if ticket <= self.last_pipeline_ticket {
            return Err(GuestIpcError::NonMonotonicPipelineTicket {
                previous: self.last_pipeline_ticket,
                ticket,
            });
        }
        if !valid_pipeline_entry(entry) {
            return Err(GuestIpcError::InvalidPipelineEntry);
        }
        let entry_len =
            u16::try_from(entry.len()).expect("entry length is bounded by MAX_PIPELINE_ENTRY_LEN");
        let mut payload = Vec::with_capacity(PIPELINE_HEADER_LEN + entry.len());
        payload.extend_from_slice(&ticket.to_le_bytes());
        payload.extend_from_slice(&entry_len.to_le_bytes());
        payload.extend_from_slice(entry.as_bytes());

        self.write_frame(KIND_PIPELINE, &payload, "pipeline registration")?;
        let (kind, reply) = self.read_frame("pipeline registration reply")?;
        if kind != KIND_PIPELINE_REPLY {
            let error = GuestIpcError::UnexpectedKind {
                expected: KIND_PIPELINE_REPLY,
                actual: kind,
            };
            return Err(self.poison(error));
        }

        match (reply.first().copied(), reply.len()) {
            (Some(PIPELINE_REPLY_OK), PIPELINE_REPLY_OK_LEN) => {
                let mut id = [0u8; 8];
                id.copy_from_slice(&reply[1..]);
                let pipeline = VmPipelineId(u64::from_le_bytes(id));
                if pipeline.0 == 0 {
                    let error = GuestIpcError::MalformedPipelineReply {
                        length: reply.len(),
                    };
                    return Err(self.poison(error));
                }
                self.last_pipeline_ticket = ticket;
                Ok(pipeline)
            }
            (Some(PIPELINE_REPLY_REJECTED), 1) => Err(GuestIpcError::Rejected {
                operation: "pipeline registration",
            }),
            _ => {
                let error = GuestIpcError::MalformedPipelineReply {
                    length: reply.len(),
                };
                Err(self.poison(error))
            }
        }
    }

    /// The sequence the next submission must carry.
    pub fn next_sequence(&self) -> u32 {
        self.next_sequence
    }

    /// Allocations registered in this session, in registration order.
    pub fn allocations(&self) -> &[RegisteredAllocation] {
        &self.allocations
    }

    /// Whether the session has been closed or poisoned by an error.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Encodes and submits `submission`, then waits for and validates its reply.
    ///
    /// `submission.sequence` must equal [`GuestIpcClient::next_sequence`].
    /// The sequence is consumed only after the whole frame is written. Any
    /// transport, framing or decoding failure closes the session.
    pub fn submit(&mut self, submission: &Submission) -> Result<Reply, GuestIpcError> {
        self.submit_with_identity(submission, None)
    }

    /// Builds a submission with the next sequence, tags it with `identity`, and
    /// submits it.
    ///
    /// The list identity lets the host enforce exactly-once execution for a
    /// recorded command list: a replay of an already executed list is refused
    /// with [`crate::wire::ReplyStatus::AlreadyExecuted`] instead of running a
    /// second time.
    pub fn submit_with_list(
        &mut self,
        commands: Vec<Command>,
        identity: ListIdentity,
    ) -> Result<Reply, GuestIpcError> {
        let submission = Submission {
            sequence: self.next_sequence,
            commands,
        };
        self.submit_with_identity(&submission, Some(identity))
    }

    fn submit_with_identity(
        &mut self,
        submission: &Submission,
        list: Option<ListIdentity>,
    ) -> Result<Reply, GuestIpcError> {
        self.ensure_open()?;
        if submission.sequence != self.next_sequence {
            return Err(GuestIpcError::SequenceMismatch {
                expected: self.next_sequence,
                actual: submission.sequence,
            });
        }
        let sequence = self.next_sequence;
        let next = sequence
            .checked_add(1)
            .ok_or(GuestIpcError::SequenceExhausted)?;
        let payload = wire::encode_submission_with_list(submission, list)?;

        self.write_frame(KIND_SUBMIT, &payload, "submit")?;
        self.next_sequence = next;

        let (kind, payload) = self.read_frame("submit reply")?;
        if kind != KIND_SUBMIT {
            let error = GuestIpcError::UnexpectedKind {
                expected: KIND_SUBMIT,
                actual: kind,
            };
            return Err(self.poison(error));
        }
        let reply = match wire::decode_reply(&payload) {
            Ok(reply) => reply,
            Err(error) => return Err(self.poison(GuestIpcError::Wire(error))),
        };
        if reply.sequence != sequence {
            let error = GuestIpcError::ReplySequenceMismatch {
                expected: sequence,
                actual: reply.sequence,
            };
            return Err(self.poison(error));
        }
        Ok(reply)
    }

    /// Builds a submission with the next sequence and submits it.
    pub fn submit_commands(&mut self, commands: Vec<Command>) -> Result<Reply, GuestIpcError> {
        let submission = Submission {
            sequence: self.next_sequence,
            commands,
        };
        self.submit(&submission)
    }

    /// Shuts the session down. Later calls return [`GuestIpcError::Closed`].
    pub fn close(&mut self) {
        self.poison(GuestIpcError::Closed);
    }

    fn ensure_open(&self) -> Result<(), GuestIpcError> {
        if self.closed {
            Err(GuestIpcError::Closed)
        } else {
            Ok(())
        }
    }

    /// Marks the session unusable, closes the socket and returns `error`.
    fn poison(&mut self, error: GuestIpcError) -> GuestIpcError {
        if !self.closed {
            self.closed = true;
            let _ = self.stream.shutdown(Shutdown::Both);
        }
        error
    }

    fn io_failure(&mut self, operation: &'static str, source: std::io::Error) -> GuestIpcError {
        let error = classify_io(operation, self.timeout, source);
        self.poison(error)
    }

    fn read_exact(
        &mut self,
        buffer: &mut [u8],
        operation: &'static str,
    ) -> Result<(), GuestIpcError> {
        let result = self.stream.read_exact(buffer);
        match result {
            Ok(()) => Ok(()),
            Err(source) => Err(self.io_failure(operation, source)),
        }
    }

    fn write_frame(
        &mut self,
        kind: u8,
        payload: &[u8],
        operation: &'static str,
    ) -> Result<(), GuestIpcError> {
        self.ensure_open()?;
        if payload.len() > MAX_FRAME_PAYLOAD {
            return Err(GuestIpcError::FrameTooLarge {
                length: payload.len(),
                limit: MAX_FRAME_PAYLOAD,
            });
        }
        let length = u32::try_from(payload.len()).expect("payload is bounded by MAX_FRAME_PAYLOAD");
        let mut envelope = [0u8; ENVELOPE_LEN];
        envelope[0] = kind;
        envelope[1..].copy_from_slice(&length.to_le_bytes());

        let header = self.stream.write_all(&envelope);
        if let Err(source) = header {
            return Err(self.io_failure(operation, source));
        }
        let body = self.stream.write_all(payload);
        if let Err(source) = body {
            return Err(self.io_failure(operation, source));
        }
        Ok(())
    }

    fn read_frame(&mut self, operation: &'static str) -> Result<(u8, Vec<u8>), GuestIpcError> {
        self.ensure_open()?;
        let mut envelope = [0u8; ENVELOPE_LEN];
        self.read_exact(&mut envelope, operation)?;

        let kind = envelope[0];
        let length = u32::from_le_bytes(
            envelope[1..]
                .try_into()
                .expect("envelope length field is four bytes"),
        ) as usize;
        if length > MAX_FRAME_PAYLOAD {
            let error = GuestIpcError::FrameTooLarge {
                length,
                limit: MAX_FRAME_PAYLOAD,
            };
            return Err(self.poison(error));
        }

        let mut payload = vec![0u8; length];
        self.read_exact(&mut payload, operation)?;
        Ok((kind, payload))
    }

    fn read_ack(&mut self, operation: &'static str) -> Result<(), GuestIpcError> {
        let (kind, payload) = self.read_frame(operation)?;
        if kind != KIND_ACK {
            let error = GuestIpcError::UnexpectedKind {
                expected: KIND_ACK,
                actual: kind,
            };
            return Err(self.poison(error));
        }
        if payload.len() != 1 {
            let error = GuestIpcError::BadAckLength {
                operation,
                length: payload.len(),
            };
            return Err(self.poison(error));
        }
        if payload[0] != 0 {
            return Err(self.poison(GuestIpcError::Rejected { operation }));
        }
        Ok(())
    }
}

impl Drop for GuestIpcClient {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddrV4, TcpListener};
    use std::thread::{self, JoinHandle};

    use super::*;
    use crate::wire::{BufferBinding, Dispatch1d, ListIdentity, ReplyStatus};

    const TEST_TIMEOUT: Duration = Duration::from_secs(2);
    const SECRET: [u8; SECRET_LEN] = [0x5a; SECRET_LEN];
    const ALLOCATION: u64 = 7;
    const ALLOCATION_BYTES: u64 = 4096;

    fn spawn_server<F>(handler: F) -> (u16, JoinHandle<()>)
    where
        F: FnOnce(TcpStream) + Send + 'static,
    {
        let listener =
            TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).expect("bind loopback");
        let port = listener.local_addr().expect("listener address").port();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept guest");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("server read timeout");
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .expect("server write timeout");
            handler(stream);
        });
        (port, handle)
    }

    fn read_frame(stream: &mut TcpStream) -> (u8, Vec<u8>) {
        let mut envelope = [0u8; ENVELOPE_LEN];
        stream
            .read_exact(&mut envelope)
            .expect("server frame header");
        let kind = envelope[0];
        let length = u32::from_le_bytes(envelope[1..].try_into().unwrap()) as usize;
        assert!(
            length <= MAX_FRAME_PAYLOAD,
            "server received oversized frame"
        );
        let mut payload = vec![0u8; length];
        stream
            .read_exact(&mut payload)
            .expect("server frame payload");
        (kind, payload)
    }

    fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) {
        assert!(payload.len() <= MAX_FRAME_PAYLOAD);
        let length = u32::try_from(payload.len()).unwrap().to_le_bytes();
        stream.write_all(&[kind]).expect("server kind");
        stream.write_all(&length).expect("server length");
        stream.write_all(payload).expect("server payload");
        stream.flush().expect("server flush");
    }

    fn send_ack(stream: &mut TcpStream, accepted: bool) {
        // Zero means accepted; one means rejected.
        write_frame(stream, KIND_ACK, &[u8::from(!accepted)]);
    }

    fn expect_hello(stream: &mut TcpStream) {
        let (kind, payload) = read_frame(stream);
        assert_eq!(kind, KIND_HELLO, "first frame must be hello");
        assert_eq!(payload, SECRET, "hello must carry the exact secret");
    }

    fn expect_eof(stream: &mut TcpStream) {
        let mut byte = [0u8; 1];
        match stream.read(&mut byte) {
            Ok(0) => {}
            Ok(count) => panic!("expected EOF, read {count} byte(s)"),
            Err(error) if error.kind() == ErrorKind::ConnectionReset => {}
            Err(error) => panic!("expected EOF, got {error}"),
        }
    }

    fn dispatch(sequence: u32) -> Submission {
        Submission {
            sequence,
            commands: vec![Command::Dispatch1d(Dispatch1d {
                pipeline_id: 3,
                threads: 64,
                threads_per_group: 32,
                bindings: vec![BufferBinding {
                    allocation_id: ALLOCATION,
                    generation: REGISTRATION_GENERATION,
                    offset: 0,
                    length: 256,
                }],
            })],
        }
    }

    fn send_reply(stream: &mut TcpStream, sequence: u32, status: ReplyStatus) {
        let reply = Reply { sequence, status };
        write_frame(
            stream,
            KIND_SUBMIT,
            &wire::encode_reply(&reply).expect("test reply encodes"),
        );
    }

    fn send_pipeline_reply(stream: &mut TcpStream, status: u8, pipeline_id: Option<u64>) {
        let mut payload = vec![status];
        if let Some(id) = pipeline_id {
            payload.extend_from_slice(&id.to_le_bytes());
        }
        write_frame(stream, KIND_PIPELINE_REPLY, &payload);
    }

    fn expect_pipeline_request(stream: &mut TcpStream, ticket: u64, entry: &str) {
        let (kind, payload) = read_frame(stream);
        assert_eq!(kind, KIND_PIPELINE, "expected a pipeline request");
        assert_eq!(payload.len(), PIPELINE_HEADER_LEN + entry.len());
        assert_eq!(
            u64::from_le_bytes(payload[0..8].try_into().unwrap()),
            ticket
        );
        assert_eq!(
            u16::from_le_bytes(payload[8..10].try_into().unwrap()) as usize,
            entry.len()
        );
        assert_eq!(&payload[PIPELINE_HEADER_LEN..], entry.as_bytes());
    }

    #[test]
    fn hello_register_and_two_submissions_roundtrip() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);

            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, KIND_REGISTER);
            assert_eq!(payload.len(), REGISTER_PAYLOAD_LEN);
            assert_eq!(
                u64::from_le_bytes(payload[0..8].try_into().unwrap()),
                ALLOCATION
            );
            assert_eq!(
                u64::from_le_bytes(payload[8..16].try_into().unwrap()),
                ALLOCATION_BYTES
            );
            send_ack(&mut stream, true);

            for expected in 1..=2 {
                let (kind, payload) = read_frame(&mut stream);
                assert_eq!(kind, KIND_SUBMIT);
                let submission = wire::decode_submission(&payload).expect("submission decodes");
                assert_eq!(submission, dispatch(expected));
                send_reply(&mut stream, expected, ReplyStatus::Ok);
            }
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        assert_eq!(client.next_sequence(), 1);

        let registered = client
            .register(ALLOCATION, ALLOCATION_BYTES)
            .expect("register");
        assert_eq!(
            registered,
            RegisteredAllocation {
                id: ALLOCATION,
                byte_len: ALLOCATION_BYTES,
                generation: 1,
            }
        );
        assert_eq!(client.allocations(), &[registered]);

        // A locally rejected sequence must not be sent or consumed.
        let error = client
            .submit(&dispatch(9))
            .expect_err("wrong sequence must fail locally");
        assert!(
            matches!(
                error,
                GuestIpcError::SequenceMismatch {
                    expected: 1,
                    actual: 9
                }
            ),
            "unexpected sequence error: {error}"
        );
        assert_eq!(client.next_sequence(), 1);

        // A codec rejection happens before the frame is written, so it must
        // leave both the session and the sequence untouched.
        let mut invalid = dispatch(1);
        let Command::Dispatch1d(dispatch_cmd) = &mut invalid.commands[0] else {
            panic!("expected a 1D dispatch command");
        };
        dispatch_cmd.pipeline_id = 0;
        let error = client.submit(&invalid).expect_err("invalid submission");
        assert!(
            matches!(error, GuestIpcError::Wire(WireError::ZeroPipelineId)),
            "unexpected wire error: {error}"
        );
        assert_eq!(client.next_sequence(), 1);
        assert!(!client.is_closed());

        let reply = client.submit(&dispatch(1)).expect("first submission");
        assert_eq!(
            reply,
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );
        assert_eq!(client.next_sequence(), 2);

        let reply = client
            .submit_commands(dispatch(2).commands)
            .expect("second submission");
        assert_eq!(reply.sequence, 2);
        assert_eq!(reply.status, ReplyStatus::Ok);
        assert_eq!(client.next_sequence(), 3);

        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn submit_with_list_carries_identity_and_reports_replay() {
        let identity = ListIdentity {
            list_id: 77,
            list_generation: 2,
        };
        let (port, server) = spawn_server(move |mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            for (sequence, status) in [
                (1u32, ReplyStatus::Ok),
                (2u32, ReplyStatus::AlreadyExecuted),
            ] {
                let (kind, payload) = read_frame(&mut stream);
                assert_eq!(kind, KIND_SUBMIT);
                let (submission, list) =
                    wire::decode_submission_with_list(&payload).expect("submission decodes");
                assert_eq!(submission.sequence, sequence);
                assert_eq!(list, Some(identity));
                send_reply(&mut stream, sequence, status);
            }
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let first = client
            .submit_with_list(dispatch(1).commands, identity)
            .expect("first list submission");
        assert_eq!(
            first,
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );
        assert_eq!(client.next_sequence(), 2);

        let replay = client
            .submit_with_list(dispatch(2).commands, identity)
            .expect("replay reply");
        assert_eq!(
            replay,
            Reply {
                sequence: 2,
                status: ReplyStatus::AlreadyExecuted,
            }
        );
        assert_eq!(client.next_sequence(), 3);
        assert!(!client.is_closed());
        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn rejected_hello_returns_no_session() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, false);
            expect_eof(&mut stream);
        });

        let error = GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT)
            .err()
            .expect("rejected hello must fail");
        assert!(
            matches!(error, GuestIpcError::Rejected { operation: "hello" }),
            "unexpected hello error: {error}"
        );
        server.join().expect("server thread");
    }

    #[test]
    fn rejected_register_poisons_the_session() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let (kind, _) = read_frame(&mut stream);
            assert_eq!(kind, KIND_REGISTER);
            send_ack(&mut stream, false);
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .register(ALLOCATION, ALLOCATION_BYTES)
            .expect_err("rejected register must fail");
        assert!(
            matches!(
                error,
                GuestIpcError::Rejected {
                    operation: "register"
                }
            ),
            "unexpected register error: {error}"
        );
        assert!(client.is_closed());

        // Nothing further may reach the wire; the server only sees EOF.
        let error = client
            .register(ALLOCATION + 1, 1)
            .expect_err("closed session must fail");
        assert!(
            matches!(error, GuestIpcError::Closed),
            "unexpected: {error}"
        );
        server.join().expect("server thread");
    }

    #[test]
    fn bad_reply_sequence_is_rejected_and_closes() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMIT);
            assert_eq!(wire::decode_submission(&payload).unwrap().sequence, 1);
            send_reply(&mut stream, 2, ReplyStatus::Ok);
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .submit(&dispatch(1))
            .expect_err("mismatched reply sequence must fail");
        assert!(
            matches!(
                error,
                GuestIpcError::ReplySequenceMismatch {
                    expected: 1,
                    actual: 2
                }
            ),
            "unexpected reply error: {error}"
        );
        assert!(client.is_closed());
        assert!(matches!(
            client.submit(&dispatch(2)),
            Err(GuestIpcError::Closed)
        ));
        server.join().expect("server thread");
    }

    #[test]
    fn unexpected_reply_kind_is_rejected_and_closes() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let _ = read_frame(&mut stream);
            send_ack(&mut stream, true);
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .submit(&dispatch(1))
            .expect_err("ack is not a submit reply");
        assert!(
            matches!(
                error,
                GuestIpcError::UnexpectedKind {
                    expected: KIND_SUBMIT,
                    actual: KIND_ACK
                }
            ),
            "unexpected kind error: {error}"
        );
        assert!(client.is_closed());
        server.join().expect("server thread");
    }

    #[test]
    fn oversized_declared_length_is_rejected_before_payload() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let _ = read_frame(&mut stream);
            let oversized = u32::try_from(MAX_FRAME_PAYLOAD + 1).unwrap().to_le_bytes();
            stream.write_all(&[KIND_ACK]).expect("kind");
            stream.write_all(&oversized).expect("oversized length");
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .register(ALLOCATION, ALLOCATION_BYTES)
            .expect_err("oversized frame must fail");
        assert!(
            matches!(
                error,
                GuestIpcError::FrameTooLarge {
                    length,
                    limit: MAX_FRAME_PAYLOAD
                } if length == MAX_FRAME_PAYLOAD + 1
            ),
            "unexpected frame error: {error}"
        );
        assert!(client.is_closed());
        server.join().expect("server thread");
    }

    #[test]
    fn disconnect_mid_reply_is_reported() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let (kind, _) = read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMIT);
            // Two bytes of a five-byte envelope, then close.
            stream.write_all(&[KIND_SUBMIT, 0x08]).expect("partial");
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .submit(&dispatch(1))
            .expect_err("partial reply must fail");
        assert!(
            matches!(
                error,
                GuestIpcError::Disconnected {
                    operation: "submit reply"
                }
            ),
            "unexpected disconnect error: {error}"
        );
        assert!(client.is_closed());
        server.join().expect("server thread");
    }

    #[test]
    fn stalled_reply_times_out_and_poisons_the_session() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let _ = read_frame(&mut stream);
            // Outlive the client's timeout; the test only needs the bound to
            // expire, not the server to stay silent forever.
            thread::sleep(Duration::from_millis(300));
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, Duration::from_millis(100))
                .expect("connect");
        let error = client
            .register(ALLOCATION, ALLOCATION_BYTES)
            .expect_err("stalled register must time out");
        assert!(
            matches!(
                error,
                GuestIpcError::Timeout {
                    operation: "register",
                    ..
                }
            ),
            "unexpected timeout error: {error}"
        );
        assert!(client.is_closed());
        assert!(matches!(
            client.register(ALLOCATION + 1, 1),
            Err(GuestIpcError::Closed)
        ));
        server.join().expect("server thread");
    }

    #[test]
    fn registration_ids_must_be_nonzero_and_strictly_increasing() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            // Exactly one registration reaches the host.
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, KIND_REGISTER);
            assert_eq!(u64::from_le_bytes(payload[0..8].try_into().unwrap()), 5);
            send_ack(&mut stream, true);
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        assert!(matches!(
            client.register(0, 100),
            Err(GuestIpcError::ZeroAllocationId)
        ));
        assert!(matches!(
            client.register(5, 0),
            Err(GuestIpcError::ZeroAllocationLength(5))
        ));

        client.register(5, 100).expect("first valid registration");
        assert!(matches!(
            client.register(5, 100),
            Err(GuestIpcError::NonMonotonicAllocationId { previous: 5, id: 5 })
        ));
        assert!(matches!(
            client.register(4, 100),
            Err(GuestIpcError::NonMonotonicAllocationId { previous: 5, id: 4 })
        ));
        assert_eq!(client.allocations().len(), 1);

        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn execution_status_is_returned_without_closing_the_session() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            let _ = read_frame(&mut stream);
            send_reply(&mut stream, 1, ReplyStatus::ExecutionFailed);
            let _ = read_frame(&mut stream);
            send_reply(&mut stream, 2, ReplyStatus::UnknownHandle);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let first = client.submit(&dispatch(1)).expect("first reply");
        assert_eq!(first.status, ReplyStatus::ExecutionFailed);
        assert!(!client.is_closed());
        let second = client.submit(&dispatch(2)).expect("second reply");
        assert_eq!(second.status, ReplyStatus::UnknownHandle);
        assert!(!client.is_closed());
        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn malformed_acknowledgement_length_is_rejected() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            write_frame(&mut stream, KIND_ACK, &[]);
            expect_eof(&mut stream);
        });

        let error = GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT)
            .err()
            .expect("empty ack must fail");
        assert!(
            matches!(
                error,
                GuestIpcError::BadAckLength {
                    operation: "hello",
                    length: 0
                }
            ),
            "unexpected ack error: {error}"
        );
        server.join().expect("server thread");
    }

    #[test]
    fn pipeline_registration_returns_native_id_and_enforces_local_contract() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            expect_pipeline_request(&mut stream, 3, "vm_entry");
            send_pipeline_reply(&mut stream, PIPELINE_REPLY_OK, Some(0x55));
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        assert_eq!(
            client
                .register_pipeline(3, "vm_entry")
                .expect("pipeline registration"),
            VmPipelineId(0x55)
        );

        // Locally rejected requests must not reach the wire.
        assert!(matches!(
            client.register_pipeline(0, "vm_entry"),
            Err(GuestIpcError::ZeroPipelineTicket)
        ));
        assert!(matches!(
            client.register_pipeline(3, "vm_entry"),
            Err(GuestIpcError::NonMonotonicPipelineTicket {
                previous: 3,
                ticket: 3
            })
        ));
        assert!(matches!(
            client.register_pipeline(2, "vm_entry"),
            Err(GuestIpcError::NonMonotonicPipelineTicket {
                previous: 3,
                ticket: 2
            })
        ));
        for entry in ["", "../vm_entry", "9lives", "vm entry", "vm_entry.1"] {
            assert!(
                matches!(
                    client.register_pipeline(4, entry),
                    Err(GuestIpcError::InvalidPipelineEntry)
                ),
                "entry {entry:?} should be rejected"
            );
        }
        let too_long = "a".repeat(MAX_PIPELINE_ENTRY_LEN + 1);
        assert!(matches!(
            client.register_pipeline(4, &too_long),
            Err(GuestIpcError::InvalidPipelineEntry)
        ));
        assert!(!client.is_closed());

        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn explicit_pipeline_rejection_keeps_the_session_open() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            expect_pipeline_request(&mut stream, 4, "vm_fail");
            send_pipeline_reply(&mut stream, PIPELINE_REPLY_REJECTED, None);
            expect_pipeline_request(&mut stream, 5, "vm_ok");
            send_pipeline_reply(&mut stream, PIPELINE_REPLY_OK, Some(9));
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .register_pipeline(4, "vm_fail")
            .expect_err("host rejection must be explicit");
        assert!(
            matches!(
                error,
                GuestIpcError::Rejected {
                    operation: "pipeline registration"
                }
            ),
            "unexpected rejection: {error}"
        );
        assert!(!client.is_closed());
        assert_eq!(
            client.register_pipeline(5, "vm_ok").expect("retry success"),
            VmPipelineId(9)
        );
        client.close();
        server.join().expect("server thread");
    }

    #[test]
    fn malformed_pipeline_reply_poisons_the_session() {
        let (port, server) = spawn_server(|mut stream| {
            expect_hello(&mut stream);
            send_ack(&mut stream, true);
            expect_pipeline_request(&mut stream, 6, "vm_entry");
            send_pipeline_reply(&mut stream, PIPELINE_REPLY_OK, None);
            expect_eof(&mut stream);
        });

        let mut client =
            GuestIpcClient::connect_with_timeout(port, &SECRET, TEST_TIMEOUT).expect("connect");
        let error = client
            .register_pipeline(6, "vm_entry")
            .expect_err("short success reply must fail");
        assert!(
            matches!(error, GuestIpcError::MalformedPipelineReply { length: 1 }),
            "unexpected malformed reply: {error}"
        );
        assert!(client.is_closed());
        server.join().expect("server thread");
    }

    #[test]
    fn pipeline_entry_validation_is_bounded_and_path_free() {
        for entry in ["a", "_", "vm_entry", "_9", "A0_z"] {
            assert!(valid_pipeline_entry(entry), "{entry:?} should be valid");
        }
        for entry in [
            "",
            "9lives",
            "vm-entry",
            "vm.entry",
            "vm/entry",
            "vm\\entry",
            "vm\0",
        ] {
            assert!(!valid_pipeline_entry(entry), "{entry:?} should be invalid");
        }
        assert!(!valid_pipeline_entry(
            &"a".repeat(MAX_PIPELINE_ENTRY_LEN + 1)
        ));
    }

    #[test]
    fn zero_port_and_zero_timeout_are_rejected_locally() {
        assert!(matches!(
            GuestIpcClient::connect_with_timeout(0, &SECRET, TEST_TIMEOUT),
            Err(GuestIpcError::InvalidPort)
        ));
        assert!(matches!(
            GuestIpcClient::connect_with_timeout(1, &SECRET, Duration::ZERO),
            Err(GuestIpcError::InvalidTimeout)
        ));
    }
}
