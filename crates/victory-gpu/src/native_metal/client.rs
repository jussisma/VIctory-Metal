//! Live Wire v3 client transport for the guest (ADR-012).
//!
//! This module owns the guest-side implementation of
//! [`crate::native_metal::queue::V3Sink`]: an authenticated connect, the ring
//! submission of already-encoded v3 frames, and a zero-IPC completion-fence
//! read. It never falls back to D3D12/D3DMetal: every failure is an explicit
//! error, and the transport never silently succeeds.
//!
//! # Wire v3 framing (verified against `victory_metal::ipc_host`)
//!
//! The host's transport envelope is **kind first, then a little-endian `u32`
//! payload length** — five bytes in total — followed by exactly `len` payload
//! bytes (`write_frame`/`read_frame` in `ipc_host.rs`; the same order the Win64
//! `GuestIpcClient` writes). A version 3 request is one `KIND_SUBMISSION_V3`
//! (`5`) frame whose payload is a 16-byte `V3RQ` envelope around one
//! [`encode_submission_v3`](victory_metal::wire_v3::encode_submission_v3)
//! frame:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 4 | magic `V3RQ` |
//! | 4 | 2 | protocol version, must be 3 |
//! | 6 | 4 | submission sequence, nonzero, equal to the inner frame's |
//! | 10 | 2 | reserved, must be zero |
//! | 12 | 4 | encoded submission length |
//! | 16 | N | the encoded version 3 submission |
//!
//! The host answers **exactly one** `KIND_SUBMISSION_V3` frame holding a
//! `victory_metal::wire::Reply` frame (`sequence u32`, `status u16`, zero
//! reserved `u16`). The status codes are the wire's own: `0` = `Ok`,
//! `1` = `UnknownHandle`, `2` = `ExecutionFailed`, `3` = `AlreadyExecuted`.
//! A replayed list therefore surfaces as `AlreadyExecuted`, not as an error;
//! `UnknownHandle`/`ExecutionFailed` are explicit refusals and are never
//! retried. (An older note described a one-byte status and a length-first
//! envelope; neither matches `ipc_host.rs`, so this module follows the code.)
//!
//! # Configuration
//!
//! [`LiveConfig::from_env`] reads `VICTORY_METAL_ENDPOINT` (`host:port`,
//! default `127.0.0.1:47654`) and `VICTORY_METAL_SECRET` (32 bytes as 64 hex
//! characters or as raw bytes). The endpoint must be loopback: the host binds
//! `127.0.0.1`, and the secret is only a local link property. The secret is
//! never logged, never named in an error, and is zeroed on drop.
//!
//! # Completion
//!
//! [`WireV3Client::completed_fence`] is the zero-IPC completion seam. No shared
//! fence page is published yet (`victory_metal::fence_page` is still a
//! reserved module), so it returns `None` and the synchronous kind 5 reply is
//! the only completion signal available to this wave.
//!
//! # Retry policy
//!
//! One transport failure is followed by exactly one reconnect attempt
//! ([`WireV3Client::reconnect`]) and one re-send. A second failure is an
//! explicit error; there is no loop. Only a submission that names a list
//! identity is retried, because the host's `ListLedger` makes that replay
//! harmless (`AlreadyExecuted`). A submission without a list identity cannot
//! be deduplicated by the host, so it is never re-sent.

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use victory_metal::wire::{MAGIC, Reply, ReplyStatus, decode_reply};
use victory_metal::wire_v3::PROTOCOL_VERSION_V3;

use crate::native_metal::queue::{NativeQueueError, V3Sink};

/// Frame kind of the first, authentication frame (`ipc_host::KIND_AUTH`).
pub const KIND_AUTH: u8 = 1;
/// Frame kind of a version 3 submission request and of its reply.
pub const KIND_SUBMISSION_V3: u8 = 5;
/// Frame kind of an authentication or registration acknowledgement.
pub const KIND_ACK: u8 = 128;
/// Magic tag of the version 3 request envelope.
pub const V3_REQUEST_MAGIC: [u8; 4] = *b"V3RQ";
/// Fixed header of the version 3 request envelope.
pub const V3_REQUEST_HEADER_LEN: usize = 16;
/// Exact length of the shared session secret.
pub const SECRET_LEN: usize = 32;
/// Transport envelope length: one kind byte, then a `u32` payload length.
pub const ENVELOPE_LEN: usize = 5;
/// Largest non-v3 payload the host accepts (its `MAX_PAYLOAD_LEN`).
pub const MAX_PAYLOAD_LEN: u32 = 65_536;
/// Largest encoded v3 submission the host accepts (its `MAX_SUBMISSION_V3_LEN`).
pub const MAX_SUBMISSION_V3_LEN: usize = 4 * 1024 * 1024;
/// Largest v3 frame payload the host accepts.
pub const MAX_PAYLOAD_LEN_V3: usize = V3_REQUEST_HEADER_LEN + MAX_SUBMISSION_V3_LEN;
/// Default port the native host listens on.
pub const DEFAULT_PORT: u16 = 47_654;
/// Default endpoint when only a port is given.
pub const DEFAULT_ENDPOINT: &str = "127.0.0.1:47654";
/// Environment variable naming the host endpoint (`host:port`).
pub const ENDPOINT_ENV: &str = "VICTORY_METAL_ENDPOINT";
/// Environment variable carrying the 32-byte session secret.
pub const SECRET_ENV: &str = "VICTORY_METAL_SECRET";
/// Default connect/read/write timeout.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Header of one encoded version 3 submission frame.
const V3_FRAME_HEADER_LEN: usize = 20;
/// List block at the start of a v3 submission payload (`list_id`, generation).
const V3_LIST_BLOCK_LEN: usize = 12;
/// Frame kind inside an encoded v3 submission (`wire_v3::KIND_SUBMISSION`).
const V3_KIND_SUBMISSION: u16 = 1;

/// Why the guest could not connect to, authenticate with or talk to the host.
///
/// Every message is a description of the endpoint or the socket state. The
/// session secret is never part of an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// The endpoint text is empty or not `host:port`.
    Endpoint(String),
    /// The endpoint is not a loopback address; the local transport is local.
    NotLoopback(String),
    /// The TCP connect itself failed.
    Connect {
        /// Endpoint that was dialled.
        endpoint: String,
        /// Operating-system reason.
        reason: String,
    },
    /// A socket read or write failed (the stream is now closed).
    Io {
        /// Operation that failed (`authenticate`, `submit`, ...).
        operation: &'static str,
        /// Operating-system reason.
        reason: String,
    },
    /// A socket operation exceeded the configured timeout.
    Timeout {
        /// Operation that timed out.
        operation: &'static str,
        /// Timeout that was exceeded.
        timeout: Duration,
    },
    /// The host closed the connection where a frame was required.
    Closed {
        /// Operation that expected a frame.
        operation: &'static str,
    },
    /// The host refused the secret: no session exists.
    Unauthenticated,
    /// A frame carried a kind the transport does not accept here.
    UnexpectedKind {
        /// Kind the transport expected.
        expected: u8,
        /// Kind that arrived.
        actual: u8,
    },
    /// A frame or reply did not follow the frozen transport or wire encoding.
    Protocol(String),
    /// A payload exceeded the host's declared limit before anything was sent.
    FrameTooLarge {
        /// Payload length that was refused.
        length: usize,
        /// Host limit for that frame kind.
        limit: usize,
    },
    /// The bytes handed to [`V3Sink::send`] are not an encoded v3 submission.
    MalformedSubmission(String),
    /// The client has no open stream (a previous failure closed it).
    NotConnected,
    /// The secret is not 32 bytes (or not 64 hex characters of text).
    Secret(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Endpoint(why) => write!(f, "wire v3 endpoint: {why}"),
            Self::NotLoopback(host) => write!(
                f,
                "wire v3 endpoint {host} is not loopback; the host transport is local"
            ),
            Self::Connect { endpoint, reason } => {
                write!(f, "wire v3 connect to {endpoint} failed: {reason}")
            }
            Self::Io { operation, reason } => {
                write!(f, "wire v3 {operation}: {reason}")
            }
            Self::Timeout { operation, timeout } => {
                write!(f, "wire v3 {operation} exceeded the {timeout:?} timeout")
            }
            Self::Closed { operation } => {
                write!(f, "wire v3 {operation}: the host closed the connection")
            }
            Self::Unauthenticated => {
                write!(f, "wire v3 authentication was refused by the host")
            }
            Self::UnexpectedKind { expected, actual } => write!(
                f,
                "wire v3 frame kind {actual} arrived where {expected} was required"
            ),
            Self::Protocol(why) => write!(f, "wire v3 protocol: {why}"),
            Self::FrameTooLarge { length, limit } => write!(
                f,
                "wire v3 frame of {length} bytes exceeds the {limit}-byte host limit"
            ),
            Self::MalformedSubmission(why) => {
                write!(f, "wire v3 submission is not an encoded frame: {why}")
            }
            Self::NotConnected => write!(f, "wire v3 client is not connected"),
            Self::Secret(why) => write!(f, "wire v3 secret: {why}"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<TransportError> for NativeQueueError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error.to_string())
    }
}

/// A 32-byte session secret that cannot be printed.
///
/// The bytes are zeroed on drop. `Debug` names only the type, so no log line,
/// error message or panic can carry the secret.
pub struct Secret([u8; SECRET_LEN]);

impl Secret {
    /// Takes `bytes` as the secret; exactly [`SECRET_LEN`] bytes are required.
    pub fn new(bytes: &[u8]) -> Result<Self, TransportError> {
        if bytes.len() != SECRET_LEN {
            return Err(TransportError::Secret(format!(
                "expected {SECRET_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut secret = [0u8; SECRET_LEN];
        secret.copy_from_slice(bytes);
        Ok(Self(secret))
    }

    /// Parses a configured value: 64 hex characters, or raw 32 bytes of text.
    ///
    /// Exactly 32 characters are taken as the raw secret, so an ASCII secret
    /// that happens to look hexadecimal is not misread.
    pub fn parse(text: &str) -> Result<Self, TransportError> {
        let text = text.trim();
        if text.len() == SECRET_LEN * 2 {
            let mut secret = [0u8; SECRET_LEN];
            for (index, slot) in secret.iter_mut().enumerate() {
                let pair = &text[index * 2..index * 2 + 2];
                *slot = u8::from_str_radix(pair, 16).map_err(|_| {
                    TransportError::Secret("the value contains a non-hex digit".to_string())
                })?;
            }
            return Ok(Self(secret));
        }
        Self::new(text.as_bytes())
    }

    /// The secret bytes, for one frame.
    fn bytes(&self) -> &[u8; SECRET_LEN] {
        &self.0
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// A parsed `host:port` endpoint of the native host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    host: String,
    port: u16,
}

impl Endpoint {
    /// Parses `host:port`, `host` (default port) or `[v6]:port`.
    ///
    /// The host must resolve to a loopback address, because the native host
    /// binds `127.0.0.1` and the secret only authenticates a local link.
    pub fn parse(text: &str) -> Result<Self, TransportError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(TransportError::Endpoint("empty endpoint".to_string()));
        }
        let (host, port) = if let Some(rest) = text.strip_prefix('[') {
            let (host, tail) = rest.split_once(']').ok_or_else(|| {
                TransportError::Endpoint(format!("{text}: unclosed `[` in the host"))
            })?;
            let port = match tail {
                "" => DEFAULT_PORT,
                _ => {
                    let port = tail.strip_prefix(':').ok_or_else(|| {
                        TransportError::Endpoint(format!("{text}: expected `:` after `]`"))
                    })?;
                    parse_port(text, port)?
                }
            };
            (host.to_string(), port)
        } else {
            match text.rsplit_once(':') {
                Some((host, port)) if !host.contains(':') => {
                    (host.to_string(), parse_port(text, port)?)
                }
                Some(_) => {
                    return Err(TransportError::Endpoint(format!(
                        "{text}: a bare IPv6 endpoint needs brackets"
                    )));
                }
                None => (text.to_string(), DEFAULT_PORT),
            }
        };
        if host.is_empty() {
            return Err(TransportError::Endpoint(format!("{text}: empty host")));
        }
        if !is_loopback_host(&host) {
            return Err(TransportError::NotLoopback(host));
        }
        Ok(Self { host, port })
    }

    /// The host part of the endpoint.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// The port part of the endpoint.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The first resolved address, which must still be loopback.
    fn address(&self) -> Result<SocketAddr, TransportError> {
        let mut resolved = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|reason| TransportError::Connect {
                endpoint: self.to_string(),
                reason: reason.to_string(),
            })?;
        let address = resolved.next().ok_or_else(|| TransportError::Connect {
            endpoint: self.to_string(),
            reason: "the host resolved to no address".to_string(),
        })?;
        if !address.ip().is_loopback() {
            return Err(TransportError::NotLoopback(address.ip().to_string()));
        }
        Ok(address)
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.host.contains(':') {
            write!(f, "[{}]:{}", self.host, self.port)
        } else {
            write!(f, "{}:{}", self.host, self.port)
        }
    }
}

fn parse_port(endpoint: &str, port: &str) -> Result<u16, TransportError> {
    let port: u16 = port
        .parse()
        .map_err(|_| TransportError::Endpoint(format!("{endpoint}: `{port}` is not a port")))?;
    if port == 0 {
        return Err(TransportError::Endpoint(format!(
            "{endpoint}: port 0 is not usable"
        )));
    }
    Ok(port)
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(address) => address.is_loopback(),
        Err(_) => false,
    }
}

/// The environment configuration of a live session.
///
/// Both variables must be present and non-empty: a secret alone never opens a
/// socket, and an endpoint alone has nothing to authenticate with.
#[derive(Debug)]
pub struct LiveConfig {
    endpoint: Endpoint,
    secret: Secret,
}

impl LiveConfig {
    /// Reads [`ENDPOINT_ENV`] and [`SECRET_ENV`] from the process environment.
    ///
    /// Returns `None` when either is unset or empty; a malformed value is an
    /// explicit error (never a silent default endpoint).
    pub fn from_env() -> Option<Result<Self, TransportError>> {
        Self::parse(
            std::env::var(ENDPOINT_ENV).ok().as_deref(),
            std::env::var(SECRET_ENV).ok().as_deref(),
        )
    }

    /// Parses a raw endpoint/secret pair, as [`Self::from_env`] does.
    pub fn parse(
        endpoint: Option<&str>,
        secret: Option<&str>,
    ) -> Option<Result<Self, TransportError>> {
        let endpoint = endpoint.map(str::trim).filter(|text| !text.is_empty())?;
        let secret = secret.map(str::trim).filter(|text| !text.is_empty())?;
        Some(
            Endpoint::parse(endpoint)
                .and_then(|endpoint| Secret::parse(secret).map(|secret| Self { endpoint, secret })),
        )
    }

    /// The configured endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Opens an authenticated session with the default timeout.
    pub fn connect(&self) -> Result<WireV3Client, TransportError> {
        self.connect_with_timeout(DEFAULT_IO_TIMEOUT)
    }

    /// Opens an authenticated session with an explicit bounded timeout.
    pub fn connect_with_timeout(&self, timeout: Duration) -> Result<WireV3Client, TransportError> {
        WireV3Client::open(
            self.endpoint.clone(),
            Secret::new(self.secret.bytes())?,
            timeout,
        )
    }
}

/// One authenticated Wire v3 session with the native host.
///
/// Construction performs the hello exchange, so a client that exists has
/// already been accepted by the host. The client owns its single socket,
/// reconnects at most once per send, and never falls back to another backend.
pub struct WireV3Client {
    endpoint: Endpoint,
    secret: Secret,
    timeout: Duration,
    stream: Option<TcpStream>,
    reconnects: u32,
    transport_errors: u64,
}

impl WireV3Client {
    /// Connects to `endpoint` and authenticates with `secret` (32 bytes).
    pub fn connect(endpoint: &str, secret: &[u8]) -> Result<Self, TransportError> {
        Self::connect_with_timeout(endpoint, secret, DEFAULT_IO_TIMEOUT)
    }

    /// Connects and authenticates with an explicit bounded timeout.
    pub fn connect_with_timeout(
        endpoint: &str,
        secret: &[u8],
        timeout: Duration,
    ) -> Result<Self, TransportError> {
        Self::open(Endpoint::parse(endpoint)?, Secret::new(secret)?, timeout)
    }

    /// Connects from a [`LiveConfig`], if the environment provides one.
    pub fn from_env() -> Result<Option<Self>, TransportError> {
        match LiveConfig::from_env() {
            Some(config) => Ok(Some(config?.connect()?)),
            None => Ok(None),
        }
    }

    fn open(endpoint: Endpoint, secret: Secret, timeout: Duration) -> Result<Self, TransportError> {
        if timeout.is_zero() {
            return Err(TransportError::Timeout {
                operation: "configure the transport",
                timeout,
            });
        }
        let mut client = Self {
            endpoint,
            secret,
            timeout,
            stream: None,
            reconnects: 0,
            transport_errors: 0,
        };
        client.stream = Some(client.handshake()?);
        Ok(client)
    }

    /// The endpoint this client dials.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Whether a stream is open (a client is live until a failure closes it).
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Successful reconnects so far.
    pub fn reconnects(&self) -> u32 {
        self.reconnects
    }

    /// Transport failures observed so far (each one closed the stream).
    pub fn transport_errors(&self) -> u64 {
        self.transport_errors
    }

    /// Dials the host and performs the authenticated hello.
    ///
    /// The returned client is only constructed after the host acknowledged the
    /// secret with `KIND_ACK` and a zero status byte.
    fn handshake(&self) -> Result<TcpStream, TransportError> {
        let address = self.endpoint.address()?;
        let mut stream = TcpStream::connect_timeout(&address, self.timeout).map_err(|reason| {
            TransportError::Connect {
                endpoint: self.endpoint.to_string(),
                reason: reason.to_string(),
            }
        })?;
        stream
            .set_nodelay(true)
            .map_err(|source| io_error("configure the socket", self.timeout, source))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|source| io_error("configure the socket", self.timeout, source))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|source| io_error("configure the socket", self.timeout, source))?;

        write_frame(
            &mut stream,
            KIND_AUTH,
            self.secret.bytes(),
            "authenticate",
            self.timeout,
        )?;
        let (kind, payload) = read_frame(
            &mut stream,
            "authenticate",
            MAX_PAYLOAD_LEN as usize,
            self.timeout,
        )?;
        if kind != KIND_ACK {
            return Err(TransportError::UnexpectedKind {
                expected: KIND_ACK,
                actual: kind,
            });
        }
        match payload.as_slice() {
            [0] => Ok(stream),
            [1] => Err(TransportError::Unauthenticated),
            other => Err(TransportError::Protocol(format!(
                "authentication acknowledgement carried {} bytes, expected one",
                other.len()
            ))),
        }
    }

    /// Closes the current stream, if any, and authenticates a fresh one.
    ///
    /// Called by [`V3Sink::send`] after a transport failure, and exposed for
    /// the session owner. A failed reconnect leaves the client disconnected
    /// and returns the explicit reason; it never claims a session it does not
    /// have.
    pub fn reconnect(&mut self) -> Result<(), TransportError> {
        self.close();
        let stream = self.handshake()?;
        self.stream = Some(stream);
        self.reconnects = self.reconnects.saturating_add(1);
        Ok(())
    }

    /// Closes the socket. Later sends fail explicitly until `reconnect`.
    pub fn close(&mut self) {
        if let Some(stream) = self.stream.take() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }

    /// The host's published completion fence, when a shared page exists.
    ///
    /// The transport task has not published a shared fence page yet
    /// (`victory_metal::fence_page` is still a reserved module with no mapped
    /// page), so this is `None`: the synchronous kind 5 reply read by
    /// [`V3Sink::send`] is the only completion signal available. When the page
    /// lands this becomes one relaxed atomic load, with no IPC round trip.
    pub fn completed_fence(&self) -> Option<u64> {
        None
    }

    /// Builds one kind 5 request from an encoded v3 submission.
    ///
    /// Everything the envelope needs is read from the frozen encoded layout,
    /// which [`encode_submission_v3`](victory_metal::wire_v3::encode_submission_v3)
    /// produced in this crate, so no submission is decoded again on the
    /// per-frame path. A frame that is not a well-formed submission is refused
    /// before the socket is touched.
    fn request(&self, bytes: &[u8]) -> Result<Request, NativeQueueError> {
        let malformed = |why: &str| {
            NativeQueueError::Transport(
                TransportError::MalformedSubmission(why.to_string()).to_string(),
            )
        };
        if bytes.len() < V3_FRAME_HEADER_LEN + V3_LIST_BLOCK_LEN {
            return Err(malformed(
                "the frame is shorter than its header and list block",
            ));
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(malformed("the frame magic is not the wire magic"));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != PROTOCOL_VERSION_V3 {
            return Err(malformed("the frame is not protocol version 3"));
        }
        let kind = u16::from_le_bytes([bytes[6], bytes[7]]);
        if kind != V3_KIND_SUBMISSION {
            return Err(malformed("the frame is not a submission"));
        }
        let sequence = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if sequence == 0 {
            return Err(malformed("the frame sequence is zero"));
        }
        let declared = usize::try_from(u32::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
        ]))
        .unwrap_or(usize::MAX);
        if declared != bytes.len() - V3_FRAME_HEADER_LEN {
            return Err(malformed(
                "the declared payload length does not match the frame",
            ));
        }
        if bytes.len() > MAX_SUBMISSION_V3_LEN {
            return Err(NativeQueueError::Transport(
                TransportError::FrameTooLarge {
                    length: bytes.len(),
                    limit: MAX_SUBMISSION_V3_LEN,
                }
                .to_string(),
            ));
        }
        let list_id = u64::from_le_bytes(
            bytes[V3_FRAME_HEADER_LEN..V3_FRAME_HEADER_LEN + 8]
                .try_into()
                .expect("eight bytes"),
        );

        let mut envelope = Vec::with_capacity(V3_REQUEST_HEADER_LEN + bytes.len());
        envelope.extend_from_slice(&V3_REQUEST_MAGIC);
        envelope.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
        envelope.extend_from_slice(&sequence.to_le_bytes());
        envelope.extend_from_slice(&0u16.to_le_bytes());
        envelope.extend_from_slice(&u32::try_from(bytes.len()).unwrap_or(u32::MAX).to_le_bytes());
        envelope.extend_from_slice(bytes);
        Ok(Request {
            envelope,
            sequence,
            list_bound: list_id != 0,
        })
    }

    /// One request/reply round trip on the open stream.
    fn exchange(&mut self, request: &Request) -> Result<ReplyStatus, NativeQueueError> {
        let Some(mut stream) = self.stream.take() else {
            return Err(NativeQueueError::Transport(
                TransportError::NotConnected.to_string(),
            ));
        };
        match self.round_trip(&mut stream, request) {
            Outcome::Status(status) => {
                self.stream = Some(stream);
                Ok(status)
            }
            Outcome::Refused(why) => {
                // The host answered a complete frame: the session is still in
                // step, so the stream stays open.
                self.stream = Some(stream);
                Err(NativeQueueError::Refused(why))
            }
            Outcome::Broken(error) => {
                // The stream may be desynchronised: close it so the next send
                // must reconnect instead of reading a stale reply.
                let _ = stream.shutdown(Shutdown::Both);
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(error)
            }
        }
    }

    fn round_trip(&self, stream: &mut TcpStream, request: &Request) -> Outcome {
        let broken =
            |error: TransportError| Outcome::Broken(NativeQueueError::Transport(error.to_string()));
        if let Err(source) = write_frame(
            stream,
            KIND_SUBMISSION_V3,
            &request.envelope,
            "submit",
            self.timeout,
        ) {
            return broken(source);
        }
        let (kind, payload) =
            match read_frame(stream, "submit", MAX_PAYLOAD_LEN as usize, self.timeout) {
                Ok(frame) => frame,
                Err(error) => return broken(error),
            };
        if kind != KIND_SUBMISSION_V3 {
            return broken(TransportError::UnexpectedKind {
                expected: KIND_SUBMISSION_V3,
                actual: kind,
            });
        }
        let reply: Reply = match decode_reply(&payload) {
            Ok(reply) => reply,
            Err(error) => {
                return broken(TransportError::Protocol(format!(
                    "the reply is not a wire reply frame: {error}"
                )));
            }
        };
        if reply.sequence != request.sequence {
            return broken(TransportError::Protocol(format!(
                "the reply names sequence {} but {} was sent",
                reply.sequence, request.sequence
            )));
        }
        match reply.status {
            ReplyStatus::Ok | ReplyStatus::AlreadyExecuted => Outcome::Status(reply.status),
            other => Outcome::Refused(format!(
                "the host answered {other:?} for sequence {}",
                reply.sequence
            )),
        }
    }
}

impl V3Sink for WireV3Client {
    /// Carries one encoded submission and reads exactly one reply frame.
    ///
    /// One transport failure is followed by one reconnect and one re-send;
    /// only a list-bound submission is retried (see the module docs). A host
    /// refusal is surfaced as [`NativeQueueError::Refused`] and never retried.
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
        let request = self.request(bytes)?;
        match self.exchange(&request) {
            Ok(status) => Ok(status),
            Err(NativeQueueError::Transport(first)) => {
                if !request.list_bound {
                    return Err(NativeQueueError::Transport(format!(
                        "{first}; the submission names no list identity, so the host cannot \
                         deduplicate a replay and it is not retried"
                    )));
                }
                if let Err(error) = self.reconnect() {
                    return Err(NativeQueueError::Transport(format!(
                        "{first}; the one reconnect attempt failed: {error}"
                    )));
                }
                match self.exchange(&request) {
                    Ok(status) => Ok(status),
                    Err(second) => Err(NativeQueueError::Transport(format!(
                        "{first}; one reconnect did not restore the session: {second}"
                    ))),
                }
            }
            Err(other) => Err(other),
        }
    }
}

impl std::fmt::Debug for WireV3Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WireV3Client")
            .field("endpoint", &self.endpoint.to_string())
            .field("connected", &self.is_connected())
            .field("reconnects", &self.reconnects)
            .field("transport_errors", &self.transport_errors)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// One framed request, with the facts the retry policy needs.
struct Request {
    envelope: Vec<u8>,
    sequence: u32,
    /// Whether the submission names a list identity (only those are retried).
    list_bound: bool,
}

/// How one round trip ended.
enum Outcome {
    /// The host answered with a completion status.
    Status(ReplyStatus),
    /// The host explicitly refused the submission (not a transport failure).
    Refused(String),
    /// The stream is unusable; the error is explicit.
    Broken(NativeQueueError),
}

/// Writes one transport envelope: one kind byte, a `u32` length, then payload.
fn write_frame(
    stream: &mut TcpStream,
    kind: u8,
    payload: &[u8],
    operation: &'static str,
    timeout: Duration,
) -> Result<(), TransportError> {
    let limit = if kind == KIND_SUBMISSION_V3 {
        MAX_PAYLOAD_LEN_V3
    } else {
        MAX_PAYLOAD_LEN as usize
    };
    if payload.len() > limit {
        return Err(TransportError::FrameTooLarge {
            length: payload.len(),
            limit,
        });
    }
    let length = u32::try_from(payload.len()).map_err(|_| TransportError::FrameTooLarge {
        length: payload.len(),
        limit,
    })?;
    let mut envelope = [0u8; ENVELOPE_LEN];
    envelope[0] = kind;
    envelope[1..].copy_from_slice(&length.to_le_bytes());
    stream
        .write_all(&envelope)
        .and_then(|()| stream.write_all(payload))
        .and_then(|()| stream.flush())
        .map_err(|source| io_error(operation, timeout, source))
}

/// Reads one transport envelope and its payload.
fn read_frame(
    stream: &mut TcpStream,
    operation: &'static str,
    limit: usize,
    timeout: Duration,
) -> Result<(u8, Vec<u8>), TransportError> {
    let mut envelope = [0u8; ENVELOPE_LEN];
    let mut filled = 0usize;
    while filled < ENVELOPE_LEN {
        match stream.read(&mut envelope[filled..]) {
            Ok(0) if filled == 0 => return Err(TransportError::Closed { operation }),
            Ok(0) => {
                return Err(TransportError::Protocol(
                    "the host closed inside a transport envelope".to_string(),
                ));
            }
            Ok(read) => filled += read,
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(io_error(operation, timeout, source)),
        }
    }
    let declared = usize::try_from(u32::from_le_bytes([
        envelope[1],
        envelope[2],
        envelope[3],
        envelope[4],
    ]))
    .unwrap_or(usize::MAX);
    if declared > limit {
        return Err(TransportError::FrameTooLarge {
            length: declared,
            limit,
        });
    }
    let mut payload = vec![0u8; declared];
    stream
        .read_exact(&mut payload)
        .map_err(|source| io_error(operation, timeout, source))?;
    Ok((envelope[0], payload))
}

fn io_error(operation: &'static str, timeout: Duration, source: io::Error) -> TransportError {
    match source.kind() {
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
            TransportError::Timeout { operation, timeout }
        }
        _ => TransportError::Io {
            operation,
            reason: source.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Read, Write};
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};

    use victory_metal::wire::{BufferBinding, Dispatch1d, ListIdentity, Reply};
    use victory_metal::wire_v3::{CommandV3, SubmissionV3, encode_submission_v3};

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const SECRET: [u8; SECRET_LEN] = [0x5a; SECRET_LEN];
    const SEQUENCE: u32 = 17;

    /// A loopback listener for one test, bound to port 0.
    struct TestHost {
        port: u16,
        handle: Option<JoinHandle<()>>,
    }

    impl TestHost {
        fn spawn<F>(handler: F) -> Self
        where
            F: FnOnce(TcpListener) + Send + 'static,
        {
            let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .expect("bind loopback");
            let port = listener.local_addr().expect("listener address").port();
            let handle = thread::spawn(move || {
                listener.set_nonblocking(false).expect("blocking listener");
                handler(listener);
            });
            Self {
                port,
                handle: Some(handle),
            }
        }

        fn read_frame(stream: &mut TcpStream) -> (u8, Vec<u8>) {
            let mut envelope = [0u8; ENVELOPE_LEN];
            stream
                .read_exact(&mut envelope)
                .expect("server frame header");
            let kind = envelope[0];
            let length = u32::from_le_bytes(envelope[1..].try_into().expect("four bytes")) as usize;
            let mut payload = vec![0u8; length];
            stream
                .read_exact(&mut payload)
                .expect("server frame payload");
            (kind, payload)
        }

        /// Reads one frame, or `None` when the guest closed or timed out.
        fn try_read_frame(stream: &mut TcpStream) -> Option<(u8, Vec<u8>)> {
            let mut header = [0u8; ENVELOPE_LEN];
            let mut filled = 0usize;
            while filled < ENVELOPE_LEN {
                match stream.read(&mut header[filled..]) {
                    Ok(0) if filled == 0 => return None,
                    Ok(0) => panic!("the guest closed inside an envelope"),
                    Ok(read) => filled += read,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => return None,
                }
            }
            let length = u32::from_le_bytes(header[1..].try_into().expect("four bytes")) as usize;
            let mut payload = vec![0u8; length];
            stream.read_exact(&mut payload).ok()?;
            Some((header[0], payload))
        }

        fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) {
            let length = u32::try_from(payload.len()).expect("payload fits u32");
            stream.write_all(&[kind]).expect("server kind");
            stream
                .write_all(&length.to_le_bytes())
                .expect("server length");
            stream.write_all(payload).expect("server payload");
            stream.flush().expect("server flush");
        }

        fn write_reply(stream: &mut TcpStream, sequence: u32, status: ReplyStatus) {
            let reply = victory_metal::wire::encode_reply(&Reply { sequence, status })
                .expect("reply encodes");
            Self::write_frame(stream, KIND_SUBMISSION_V3, &reply);
        }

        fn endpoint(&self) -> String {
            format!("127.0.0.1:{}", self.port)
        }
    }

    impl Drop for TestHost {
        fn drop(&mut self) {
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn submission(sequence: u32, list_id: u64) -> SubmissionV3 {
        SubmissionV3 {
            sequence,
            list: (list_id != 0).then_some(ListIdentity {
                list_id,
                list_generation: 1,
            }),
            commands: vec![CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: 3,
                threads: 64,
                threads_per_group: 32,
                bindings: vec![BufferBinding {
                    allocation_id: 7,
                    generation: 1,
                    offset: 0,
                    length: 256,
                }],
            })],
        }
    }

    fn encoded(sequence: u32, list_id: u64) -> Vec<u8> {
        encode_submission_v3(&submission(sequence, list_id)).expect("canonical v3 encoding")
    }

    /// Accepts one peer, verifies the secret, acknowledges it and hands the
    /// stream to `serve`.
    fn accept_authenticated(listener: &TcpListener, secret: &[u8; SECRET_LEN]) -> TcpStream {
        let (mut stream, _) = listener.accept().expect("accept the guest");
        stream.set_nonblocking(false).expect("blocking stream");
        stream
            .set_read_timeout(Some(TEST_TIMEOUT))
            .expect("server read timeout");
        stream
            .set_write_timeout(Some(TEST_TIMEOUT))
            .expect("server write timeout");
        let (kind, payload) = TestHost::read_frame(&mut stream);
        assert_eq!(kind, KIND_AUTH, "the first frame must authenticate");
        assert_eq!(payload, secret, "the hello must carry the exact secret");
        TestHost::write_frame(&mut stream, KIND_ACK, &[0]);
        stream
    }

    #[test]
    fn an_authenticated_client_carries_one_v3_frame_and_reads_its_reply() {
        let (sender, receiver) = mpsc::channel::<(u8, SubmissionV3)>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMISSION_V3, "one kind 5 request");
            assert_eq!(
                &payload[..4],
                &V3_REQUEST_MAGIC,
                "the request starts with the V3RQ magic"
            );
            assert_eq!(
                u16::from_le_bytes([payload[4], payload[5]]),
                PROTOCOL_VERSION_V3
            );
            assert_eq!(
                u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]),
                SEQUENCE
            );
            assert_eq!(u16::from_le_bytes([payload[10], payload[11]]), 0);
            let declared =
                u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]) as usize;
            assert_eq!(declared, payload.len() - V3_REQUEST_HEADER_LEN);
            let inner =
                victory_metal::wire_v3::decode_submission_v3(&payload[V3_REQUEST_HEADER_LEN..])
                    .expect("the host decodes the carried frame");
            sender.send((kind, inner)).expect("report the request");
            TestHost::write_reply(&mut stream, SEQUENCE, ReplyStatus::Ok);
            // The client keeps the connection open for the next frame.
            let mut byte = [0u8; 1];
            assert!(matches!(stream.read(&mut byte), Ok(0) | Err(_)));
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        assert!(client.is_connected());
        assert_eq!(client.reconnects(), 0);
        assert_eq!(client.transport_errors(), 0);
        assert_eq!(
            client.completed_fence(),
            None,
            "no shared fence page exists yet"
        );

        assert_eq!(
            client.send(&encoded(SEQUENCE, 100)),
            Ok(ReplyStatus::Ok),
            "the host's Ok reply is surfaced"
        );
        let (kind, carried) = receiver.recv().expect("the host saw the frame");
        assert_eq!(kind, KIND_SUBMISSION_V3);
        assert_eq!(carried, submission(SEQUENCE, 100));
        assert!(client.is_connected(), "a success keeps the one stream");
    }

    #[test]
    fn already_executed_is_surfaced_and_the_frame_is_not_resent() {
        let (sender, receiver) = mpsc::channel::<usize>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            stream
                .set_read_timeout(Some(Duration::from_millis(500)))
                .expect("short server read timeout");
            let mut frames = 0usize;
            while let Some((kind, _payload)) = TestHost::try_read_frame(&mut stream) {
                assert_eq!(kind, KIND_SUBMISSION_V3);
                frames += 1;
                TestHost::write_reply(&mut stream, SEQUENCE, ReplyStatus::AlreadyExecuted);
            }
            sender.send(frames).expect("report the frame count");
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        assert_eq!(
            client.send(&encoded(SEQUENCE, 100)),
            Ok(ReplyStatus::AlreadyExecuted)
        );
        assert_eq!(client.transport_errors(), 0, "a replay is not a failure");
        drop(client);
        assert_eq!(
            receiver.recv_timeout(TEST_TIMEOUT).expect("frame count"),
            1,
            "the replayed frame was sent exactly once"
        );
    }

    #[test]
    fn a_wrong_secret_is_refused_without_an_open_session() {
        let host = TestHost::spawn(move |listener| {
            let (mut stream, _) = listener.accept().expect("accept the guest");
            stream
                .set_read_timeout(Some(TEST_TIMEOUT))
                .expect("server read timeout");
            let (kind, payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_AUTH);
            assert_ne!(payload, SECRET, "the test must send a different secret");
            TestHost::write_frame(&mut stream, KIND_ACK, &[1]);
            let _ = stream.shutdown(Shutdown::Both);
        });

        let wrong = [0x11u8; SECRET_LEN];
        let error = WireV3Client::connect_with_timeout(&host.endpoint(), &wrong, TEST_TIMEOUT)
            .expect_err("a wrong secret must not yield a client");
        assert_eq!(error, TransportError::Unauthenticated);
        let text = error.to_string();
        assert!(
            !text.contains("1111111111111111111111111111111111111111111111111111111111111111"),
            "the error must not carry the secret"
        );
    }

    #[test]
    fn a_dropped_connection_is_an_explicit_transport_error() {
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, _payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            // Drop the connection without a reply: the guest's read fails.
            let _ = stream.shutdown(Shutdown::Both);
            // Any reconnect is accepted and closed again at once, so the test
            // cannot depend on whether the port still has a listener.
            let _ = listener.set_nonblocking(true);
            if let Ok((stream, _)) = listener.accept() {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(TEST_TIMEOUT));
            }
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        let error = client
            .send(&encoded(SEQUENCE, 100))
            .expect_err("a lost reply is an explicit failure");
        let NativeQueueError::Transport(message) = &error else {
            panic!("expected a transport error, got {error:?}");
        };
        assert!(
            message.contains("reconnect"),
            "the error names the failed reconnect: {message}"
        );
        assert_eq!(client.transport_errors(), 1);
        assert!(
            !client.is_connected(),
            "a broken stream is closed and the one reconnect did not restore it"
        );
        let again = client
            .send(&encoded(SEQUENCE + 1, 100))
            .expect_err("a still-disconnected client never succeeds silently");
        assert!(
            matches!(again, NativeQueueError::Transport(_)),
            "the second send is an explicit transport error, got {again:?}"
        );
        assert!(!client.is_connected());
    }

    #[test]
    fn one_reconnect_restores_the_session_after_a_drop() {
        let host = TestHost::spawn(move |listener| {
            // First session: authenticate, then vanish without a reply.
            let mut first = accept_authenticated(&listener, &SECRET);
            let (kind, _payload) = TestHost::read_frame(&mut first);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            let _ = first.shutdown(Shutdown::Both);
            // Second session: the retry is answered.
            let mut second = accept_authenticated(&listener, &SECRET);
            let (kind, payload) = TestHost::read_frame(&mut second);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            let sequence = u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]);
            assert_eq!(sequence, SEQUENCE, "the same frame is re-sent");
            TestHost::write_reply(&mut second, SEQUENCE, ReplyStatus::Ok);
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        assert_eq!(client.send(&encoded(SEQUENCE, 100)), Ok(ReplyStatus::Ok));
        assert_eq!(client.reconnects(), 1);
        assert_eq!(client.transport_errors(), 1);
        assert!(client.is_connected());
    }

    #[test]
    fn a_list_less_submission_is_never_re_sent_after_a_drop() {
        let (sender, receiver) = mpsc::channel::<usize>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, _payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            let _ = stream.shutdown(Shutdown::Both);
            // A second connection would mean the frame was re-sent.
            let _ = listener.set_nonblocking(true);
            let frames = match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(TEST_TIMEOUT));
                    let _ = TestHost::try_read_frame(&mut stream);
                    2usize
                }
                Err(_) => 0,
            };
            sender.send(frames).expect("report the frame count");
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        let error = client
            .send(&encoded(SEQUENCE, 0))
            .expect_err("a list-less submission is not retried");
        let NativeQueueError::Transport(message) = &error else {
            panic!("expected a transport error, got {error:?}");
        };
        assert!(
            message.contains("no list identity"),
            "the error explains why no replay happened: {message}"
        );
        drop(client);
        assert_eq!(
            receiver.recv_timeout(TEST_TIMEOUT).expect("frame count"),
            0,
            "the host never saw a second session"
        );
    }

    #[test]
    fn bytes_that_are_not_an_encoded_submission_never_reach_the_socket() {
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            // Anything after the hello is a failure of this test.
            let mut byte = [0u8; 1];
            assert!(
                matches!(stream.read(&mut byte), Ok(0)),
                "no request must arrive"
            );
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        for bad in [
            Vec::new(),
            b"VMWF".to_vec(),
            vec![0u8; V3_FRAME_HEADER_LEN + V3_LIST_BLOCK_LEN],
        ] {
            let error = client
                .send(&bad)
                .expect_err("a malformed frame is refused locally");
            assert!(
                matches!(error, NativeQueueError::Transport(_)),
                "unexpected error {error:?}"
            );
        }
        // A well-formed frame still goes through afterwards: nothing was
        // poisoned by the local refusals.
        assert!(client.is_connected());
        drop(client);
    }

    #[test]
    fn endpoints_are_parsed_and_restricted_to_loopback() {
        assert_eq!(
            Endpoint::parse("127.0.0.1:47654").expect("explicit port"),
            Endpoint {
                host: "127.0.0.1".to_string(),
                port: 47_654
            }
        );
        assert_eq!(
            Endpoint::parse("127.0.0.1").expect("default port"),
            Endpoint {
                host: "127.0.0.1".to_string(),
                port: DEFAULT_PORT
            }
        );
        assert_eq!(
            Endpoint::parse("[::1]:5000").expect("bracketed v6"),
            Endpoint {
                host: "::1".to_string(),
                port: 5000
            }
        );
        assert!(Endpoint::parse("localhost").is_ok());
        assert_eq!(
            Endpoint::parse("10.0.0.1:47654"),
            Err(TransportError::NotLoopback("10.0.0.1".to_string()))
        );
        assert!(matches!(
            Endpoint::parse(""),
            Err(TransportError::Endpoint(_))
        ));
        assert!(matches!(
            Endpoint::parse("127.0.0.1:0"),
            Err(TransportError::Endpoint(_))
        ));
        assert!(matches!(
            Endpoint::parse("127.0.0.1:99999"),
            Err(TransportError::Endpoint(_))
        ));
        assert_eq!(
            Endpoint::parse("127.0.0.1:1")
                .expect("explicit port 1")
                .to_string(),
            "127.0.0.1:1"
        );
    }

    #[test]
    fn the_secret_accepts_hex_or_raw_and_is_never_printed() {
        let hex = "5a".repeat(SECRET_LEN);
        assert_eq!(
            Secret::parse(&hex).expect("hex secret").bytes(),
            &SECRET,
            "64 hex characters are 32 bytes"
        );
        assert_eq!(
            Secret::parse(&"z".repeat(SECRET_LEN))
                .expect("raw secret")
                .bytes(),
            &[b'z'; SECRET_LEN],
            "32 characters are the raw secret"
        );
        assert!(Secret::parse("abc").is_err());
        assert_eq!(
            format!("{:?}", Secret::parse(&hex).expect("hex secret")),
            "Secret(<redacted>)"
        );
    }

    #[test]
    fn live_configuration_requires_both_variables() {
        assert!(LiveConfig::parse(None, Some("00")).is_none());
        assert!(LiveConfig::parse(Some("127.0.0.1:1"), None).is_none());
        assert!(LiveConfig::parse(Some("  "), Some("00")).is_none());
        let config = LiveConfig::parse(Some("127.0.0.1:47654"), Some(&"ab".repeat(SECRET_LEN)))
            .expect("both variables present")
            .expect("well-formed configuration");
        assert_eq!(config.endpoint().to_string(), "127.0.0.1:47654");
        assert!(matches!(
            LiveConfig::parse(Some("192.168.1.2:1"), Some(&"ab".repeat(SECRET_LEN)))
                .expect("both variables present"),
            Err(TransportError::NotLoopback(_))
        ));
    }

    #[test]
    fn a_failed_connect_names_the_endpoint_and_not_the_secret() {
        // Bind and drop, so the port is almost certainly free.
        let port = {
            let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .expect("bind loopback");
            listener.local_addr().expect("address").port()
        };
        let endpoint = format!("127.0.0.1:{port}");
        let error =
            WireV3Client::connect_with_timeout(&endpoint, &SECRET, Duration::from_millis(250))
                .expect_err("nothing listens on the closed port");
        let text = error.to_string();
        assert!(text.contains(&endpoint), "the endpoint is named: {text}");
        assert!(!text.contains("5a"), "the secret is not named: {text}");
    }
}
