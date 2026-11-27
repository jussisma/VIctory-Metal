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
//! retried. An error reply may append an optional `VMR3` reason trailer
//! (bounded to 256 UTF-8 bytes) carrying the host's own diagnostic; it is
//! decoded into the refusal, counted in telemetry, and logged once as
//! `METAL_HOST_ERROR sequence=<n> reason=<text>`. A reply without the trailer
//! keeps the pre-ADR-035 text exactly. (An older note described a one-byte
//! status and a length-first envelope; neither matches `ipc_host.rs`, so this
//! module follows the code.)
//!
//! # Registration frames
//!
//! | kind | direction | payload | reply |
//! | ---: | --- | --- | --- |
//! | 2 | guest to host | `id: u64 LE`, `byte_len: u64 LE` | kind 128, `[0]` |
//! | 4 | guest to host | `ticket: u64 LE`, `entry_len: u16 LE`, entry | kind 129 |
//! | 6 | guest to host | `byte_len: u64 LE` (fence page) | kind 128, `[0]` |
//! | 8 | guest to host | 32-byte texture descriptor | kind 128, `[0]`/`[1]` |
//! | 9 | guest to host | `surface: u64 LE`, `texture: u64 LE` | kind 128 |
//!
//! A kind 2 `byte_len` is the whole share file the id names (the host refuses a
//! mapping whose length differs, which also closes the session), and ids must
//! increase: the guest pre-checks that locally so a title can never brick its
//! session with an id the host cannot accept. Kind 129 is one status byte
//! (`0` ok, `1` rejected), then the host's `u64` native pipeline id.
//!
//! # Configuration
//!
//! [`LiveConfig::from_env`] reads `VICTORY_METAL_ENDPOINT` (`host:port`),
//! `VICTORY_METAL_SECRET` (32 bytes as 64 hex characters or as raw bytes), and
//! `VICTORY_BACKEND`/`VICTORY_METAL_PORT` for the ADR-016 guest auto-connect:
//!
//! 1. `VICTORY_METAL_ENDPOINT`, when set and non-empty, always wins.
//! 2. Otherwise, with `VICTORY_BACKEND=metal`, the client dials
//!    `127.0.0.1:${VICTORY_METAL_PORT:-47654}` with the short
//!    [`AUTO_CONNECT_TIMEOUT`] instead of the ordinary I/O timeout.
//! 3. A failed auto-connect is an explicit [`TransportError::Connect`] naming
//!    that endpoint; there is no silent fallback and no second endpoint.
//!
//! The secret is required in every case: an endpoint alone has nothing to
//! authenticate with. The endpoint must be loopback, because the host binds
//! `127.0.0.1` and the secret is only a local link property. The secret is
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
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use std::collections::BTreeSet;

use victory_metal::wire::{decode_reply_with_reason, ReplyStatus, MAGIC};
use victory_metal::wire_v3::PROTOCOL_VERSION_V3;

use crate::native_metal::fence_page_guest::GuestFencePage;
use crate::native_metal::sampler::SamplerStateDesc;
use crate::native_metal::queue::{NativeQueueError, V3Sink};

/// Frame kind of the first, authentication frame (`ipc_host::KIND_AUTH`).
pub const KIND_AUTH: u8 = 1;
/// Frame kind of a version 3 submission request and of its reply.
pub const KIND_SUBMISSION_V3: u8 = 5;
/// Frame kind of a host allocation registration (`ipc_host::KIND_REGISTER`).
pub const KIND_REGISTER: u8 = 2;
/// Frame kind of an opt-in precompiled pipeline registration
/// (`ipc_host::KIND_PIPELINE`).
pub const KIND_PIPELINE: u8 = 4;
/// Frame kind of the host's pipeline registration reply
/// (`ipc_host::KIND_PIPELINE_REPLY`).
pub const KIND_PIPELINE_REPLY: u8 = 129;
/// Frame kind of an authentication or registration acknowledgement.
pub const KIND_ACK: u8 = 128;
/// Exact length of a kind 2 registration payload: `id: u64 LE`, `byte_len: u64 LE`.
pub const REGISTER_PAYLOAD_LEN: usize = 16;
/// Fixed ticket and entry-length prefix of a kind 4 payload.
pub const PIPELINE_HEADER_LEN: usize = 10;
/// Largest pipeline entry name the host accepts.
pub const MAX_PIPELINE_ENTRY_LEN: usize = 256;
/// Status byte of a successful kind 129 pipeline reply.
pub const PIPELINE_REPLY_OK: u8 = 0;
/// Status byte of an explicitly rejected pipeline registration.
pub const PIPELINE_REPLY_REJECTED: u8 = 1;
/// Exact length of a successful kind 129 reply: status plus `u64` id.
pub const PIPELINE_REPLY_OK_LEN: usize = 1 + 8;
/// First four bytes of a version 2 kind 4 request (`ipc_host`).
pub const PIPELINE_FRAME_MAGIC: [u8; 4] = *b"VP2\0";
/// Version byte of the version 2 pipeline frame.
pub const PIPELINE_FRAME_VERSION_V2: u8 = 1;
/// Fixed header length of a version 2 pipeline frame.
pub const PIPELINE_V2_HEADER_LEN: usize = 24;
/// Largest number of stage entries one version 2 frame may carry.
pub const PIPELINE_V2_MAX_STAGES: usize = 4;
/// Version 2 pipeline kinds.
pub const PIPELINE_KIND_COMPUTE: u8 = 0;
/// A vertex/fragment render pipeline.
pub const PIPELINE_KIND_GRAPHICS: u8 = 1;
/// A mesh/object/fragment render pipeline.
pub const PIPELINE_KIND_MESH: u8 = 2;
/// Version 2 per-stage codes.
pub const PIPELINE_STAGE_VERTEX: u8 = 0;
/// Pixel/fragment stage.
pub const PIPELINE_STAGE_PIXEL: u8 = 1;
/// Mesh stage.
pub const PIPELINE_STAGE_MESH: u8 = 2;
/// Object stage.
pub const PIPELINE_STAGE_OBJECT: u8 = 3;
/// Compute stage.
pub const PIPELINE_STAGE_COMPUTE: u8 = 4;
/// Largest encoded state/ABI/root-signature blob the guest will send.
pub const MAX_PIPELINE_STATE_BYTES: usize = 4 * 1024;
/// Largest encoded pipeline ABI bytes.
pub const MAX_PIPELINE_ABI_BYTES: usize = 16 * 1024;
/// Largest raw root-signature bytes.
pub const MAX_ROOT_SIGNATURE_BYTES: usize = 16 * 1024;
/// First four bytes of the version 3 kind 4 request (`ipc_host`).
///
/// The version 3 frame names content-hashed DXIL stage files inside the share
/// directory instead of carrying metallibs or DXIL bytes on the wire, so a
/// 16 MiB stage never has to fit the 64 KiB transport frame.
pub const PIPELINE_FRAME_MAGIC_V3: [u8; 4] = *b"VP3\0";
/// Version byte of the version 3 pipeline frame.
pub const PIPELINE_FRAME_VERSION_V3: u8 = 1;
/// Fixed header length of a version 3 pipeline frame.
pub const PIPELINE_V3_HEADER_LEN: usize = 24;
/// Exact length of a share-directory stage name: `stages/<64 hex>.bin`.
pub const MAX_STAGE_FILE_NAME_LEN: usize = "stages/".len() + 64 + ".bin".len();
/// Largest converter rejection reason the host may put in a kind 129 reply.
pub const MAX_PIPELINE_REJECTION_REASON: usize = 256;

/// One `(stage, file, entry)` of a version 3 pipeline request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp3Stage<'a> {
    /// Wire stage code ([`PIPELINE_STAGE_VERTEX`] to [`PIPELINE_STAGE_COMPUTE`]).
    pub stage: u8,
    /// Share-relative DXIL file name (`stages/<blake3 hex>.bin`).
    pub file: &'a str,
    /// Entry point the host's converter and Metal library both name.
    pub entry: &'a str,
}

/// Whether `name` is the one shape a version 3 stage reference may take.
///
/// Only `stages/<64 lowercase hex digits>.bin` is accepted: no absolute path,
/// `..`, backslash, separator beyond the fixed one or symlink name can be
/// expressed, and the host revalidates the same grammar before opening.
pub fn valid_stage_file_name(name: &str) -> bool {
    let Some(hash) = name
        .strip_prefix("stages/")
        .and_then(|rest| rest.strip_suffix(".bin"))
    else {
        return false;
    };
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Builds one version 3 kind 4 payload.
///
/// | offset | size | field |
/// | ---: | ---: | --- |
/// | 0 | 4 | magic `VP3\0` |
/// | 4 | 1 | frame version, `1` |
/// | 5 | 1 | pipeline kind: 0 compute, 1 graphics, 2 mesh |
/// | 6 | 1 | stage count, 1..=4 |
/// | 7 | 1 | reserved, zero |
/// | 8 | 4 | encoded `PipelineState` length |
/// | 12 | 2 | root-signature file-name length, 0 or 75 |
/// | 14 | 2 | reserved, zero |
/// | 16 | 4 | reserved, zero |
/// | 20 | 4 | reserved, zero |
/// | 24 | N | stage entries: `stage u8, name_len u16, name, entry_len u16, entry` |
/// | ... | S | encoded `PipelineState` |
/// | ... | R | root-signature file name |
pub fn encode_pipeline_v3_request(
    kind: u8,
    stages: &[Vp3Stage<'_>],
    state: &[u8],
    root_signature_file: Option<&str>,
) -> Result<Vec<u8>, NativeQueueError> {
    if !matches!(
        kind,
        PIPELINE_KIND_COMPUTE | PIPELINE_KIND_GRAPHICS | PIPELINE_KIND_MESH
    ) {
        return Err(NativeQueueError::Refused(format!(
            "pipeline kind {kind} is not compute, graphics or mesh"
        )));
    }
    if stages.is_empty() || stages.len() > PIPELINE_V2_MAX_STAGES {
        return Err(NativeQueueError::Refused(format!(
            "pipeline registration needs 1..={PIPELINE_V2_MAX_STAGES} stages, got {}",
            stages.len()
        )));
    }
    if state.len() > MAX_PIPELINE_STATE_BYTES {
        return Err(NativeQueueError::Refused(
            "pipeline state exceeds its limit".to_string(),
        ));
    }
    let root_len = root_signature_file.map_or(0, str::len);
    if root_len != 0
        && (root_len > MAX_STAGE_FILE_NAME_LEN
            || !root_signature_file.is_some_and(valid_stage_file_name))
    {
        return Err(NativeQueueError::Refused(
            "the root-signature file name is not a share-relative stage name".to_string(),
        ));
    }
    let mut seen = Vec::with_capacity(stages.len());
    let mut payload = Vec::with_capacity(
        PIPELINE_V3_HEADER_LEN
            + stages
                .iter()
                .map(|stage| 1 + 2 + stage.file.len() + 2 + stage.entry.len())
                .sum::<usize>()
            + state.len()
            + root_len,
    );
    payload.extend_from_slice(&PIPELINE_FRAME_MAGIC_V3);
    payload.push(PIPELINE_FRAME_VERSION_V3);
    payload.push(kind);
    payload.push(u8::try_from(stages.len()).expect("stage count is bounded"));
    payload.push(0);
    payload.extend_from_slice(
        &u32::try_from(state.len())
            .expect("state length is bounded")
            .to_le_bytes(),
    );
    payload.extend_from_slice(
        &u16::try_from(root_len)
            .expect("root file name is bounded")
            .to_le_bytes(),
    );
    payload.extend_from_slice(&0u16.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    for stage in stages {
        if !matches!(
            stage.stage,
            PIPELINE_STAGE_VERTEX
                | PIPELINE_STAGE_PIXEL
                | PIPELINE_STAGE_MESH
                | PIPELINE_STAGE_OBJECT
                | PIPELINE_STAGE_COMPUTE
        ) || seen.contains(&stage.stage)
        {
            return Err(NativeQueueError::Refused(format!(
                "pipeline stage {} is unknown or repeated",
                stage.stage
            )));
        }
        if !valid_stage_file_name(stage.file) {
            return Err(NativeQueueError::Refused(format!(
                "pipeline stage {} does not name a share-relative stage file",
                stage.stage
            )));
        }
        if !valid_pipeline_entry(stage.entry) {
            return Err(NativeQueueError::Refused(format!(
                "pipeline entry `{}` is not a bounded ASCII Metal identifier",
                stage.entry
            )));
        }
        payload.push(stage.stage);
        payload.extend_from_slice(
            &u16::try_from(stage.file.len())
                .expect("stage name length is bounded")
                .to_le_bytes(),
        );
        payload.extend_from_slice(stage.file.as_bytes());
        payload.extend_from_slice(
            &u16::try_from(stage.entry.len())
                .expect("entry length is bounded")
                .to_le_bytes(),
        );
        payload.extend_from_slice(stage.entry.as_bytes());
        seen.push(stage.stage);
    }
    payload.extend_from_slice(state);
    if let Some(name) = root_signature_file {
        payload.extend_from_slice(name.as_bytes());
    }
    if payload.len() > MAX_PAYLOAD_LEN as usize {
        return Err(NativeQueueError::Refused(format!(
            "pipeline registration is {} bytes, over the {MAX_PAYLOAD_LEN}-byte transport limit",
            payload.len()
        )));
    }
    Ok(payload)
}

/// A decoded version 3 kind 4 payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPipelineV3 {
    /// Pipeline kind (0 compute, 1 graphics, 2 mesh).
    pub kind: u8,
    /// Stage entries in frame order.
    pub stages: Vec<(u8, String, String)>,
    /// Encoded `PipelineState`.
    pub state: Vec<u8>,
    /// Root-signature file name, empty when the frame carried none.
    pub root_signature_file: String,
}

/// Decodes a version 3 kind 4 payload, rejecting every malformed field.
pub fn decode_pipeline_v3_request(payload: &[u8]) -> Option<DecodedPipelineV3> {
    if payload.len() < PIPELINE_V3_HEADER_LEN || payload[..4] != PIPELINE_FRAME_MAGIC_V3 {
        return None;
    }
    let version = payload[4];
    let kind = payload[5];
    let stage_count = payload[6];
    if version != PIPELINE_FRAME_VERSION_V3
        || kind > PIPELINE_KIND_MESH
        || stage_count == 0
        || usize::from(stage_count) > PIPELINE_V2_MAX_STAGES
        || payload[7] != 0
        || payload[14] != 0
        || payload[15] != 0
        || payload[16..20] != [0u8; 4]
        || payload[20..24] != [0u8; 4]
    {
        return None;
    }
    let state_len = usize::try_from(u32::from_le_bytes(payload[8..12].try_into().ok()?)).ok()?;
    let root_len = usize::from(u16::from_le_bytes(payload[12..14].try_into().ok()?));
    if state_len > MAX_PIPELINE_STATE_BYTES || root_len > MAX_STAGE_FILE_NAME_LEN {
        return None;
    }
    let mut offset = PIPELINE_V3_HEADER_LEN;
    let mut stages = Vec::with_capacity(usize::from(stage_count));
    for _ in 0..stage_count {
        let stage = *payload.get(offset)?;
        offset = offset.checked_add(1)?;
        let name_len = usize::from(u16::from_le_bytes(
            payload.get(offset..offset + 2)?.try_into().ok()?,
        ));
        offset = offset.checked_add(2)?;
        let name = std::str::from_utf8(payload.get(offset..offset.checked_add(name_len)?)?).ok()?;
        offset = offset.checked_add(name_len)?;
        let entry_len = usize::from(u16::from_le_bytes(
            payload.get(offset..offset + 2)?.try_into().ok()?,
        ));
        offset = offset.checked_add(2)?;
        let entry =
            std::str::from_utf8(payload.get(offset..offset.checked_add(entry_len)?)?).ok()?;
        offset = offset.checked_add(entry_len)?;
        if !matches!(
            stage,
            PIPELINE_STAGE_VERTEX
                | PIPELINE_STAGE_PIXEL
                | PIPELINE_STAGE_MESH
                | PIPELINE_STAGE_OBJECT
                | PIPELINE_STAGE_COMPUTE
        ) || !valid_stage_file_name(name)
            || !valid_pipeline_entry(entry)
        {
            return None;
        }
        stages.push((stage, name.to_string(), entry.to_string()));
    }
    let state_end = offset.checked_add(state_len)?;
    let state = payload.get(offset..state_end)?.to_vec();
    let root_end = state_end.checked_add(root_len)?;
    let root = std::str::from_utf8(payload.get(state_end..root_end)?).ok()?;
    if root_end != payload.len() || (root_len != 0 && !valid_stage_file_name(root)) {
        return None;
    }
    Some(DecodedPipelineV3 {
        kind,
        stages,
        state,
        root_signature_file: root.to_string(),
    })
}

/// Frame kind registering the shared completion fence page (`ipc_host`).
pub const KIND_FENCE_PAGE: u8 = 6;
/// Frame kind registering a host texture (`ipc_host`).
pub const KIND_TEXTURE_REG: u8 = 8;
/// Frame kind registering a texture as a present surface (`ipc_host`).
pub const KIND_SURFACE_REG: u8 = 9;
/// Frame kind registering a reduced `MTLSamplerState` (`ipc_host`).
pub const KIND_SAMPLER_REG: u8 = 10;
/// Exact length of a fence page registration payload (`byte_len: u64 LE`).
pub const FENCE_PAGE_REGISTER_PAYLOAD_LEN: usize = 8;
/// Exact length of a texture registration payload.
pub const TEXTURE_REG_PAYLOAD_LEN: usize = 32;
/// Exact length of a present-surface registration payload.
pub const SURFACE_REG_PAYLOAD_LEN: usize = 16;
/// Exact length of a sampler registration payload: handle then five `u32`s.
pub const SAMPLER_REG_PAYLOAD_LEN: usize = 8 + 5 * 4;
/// Status byte of an accepted texture or surface registration.
pub const REGISTRATION_ACCEPTED: u8 = 0;
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
/// Environment variable selecting the backend (`metal` enables auto-connect).
pub const BACKEND_ENV: &str = "VICTORY_BACKEND";
/// Environment variable naming the loopback port for guest auto-connect.
pub const PORT_ENV: &str = "VICTORY_METAL_PORT";
/// Backend value that opts into the native Metal host auto-connect.
pub const METAL_BACKEND: &str = "metal";
/// Default connect/read/write timeout.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(10);
/// Short connect timeout used only by the implicit (`VICTORY_BACKEND=metal`)
/// endpoint, so a missing host fails fast instead of stalling the first frame.
pub const AUTO_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);

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

/// Whether `entry` is a bounded, path-free Metal function identifier.
///
/// The grammar is the host's (`ipc_host` uses `guest_ipc::valid_pipeline_entry`
/// for the same frame): an ASCII letter or underscore followed by ASCII
/// letters, digits or underscores, at most [`MAX_PIPELINE_ENTRY_LEN`] bytes.
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

/// The environment configuration of a live session.
///
/// The secret must be present and non-empty; an endpoint alone has nothing to
/// authenticate with. The endpoint is either configured explicitly or resolved
/// from `VICTORY_BACKEND=metal`/`VICTORY_METAL_PORT` by [`Self::from_env`].
#[derive(Debug)]
pub struct LiveConfig {
    endpoint: Endpoint,
    secret: Secret,
    /// Timeout used by [`Self::connect`]. It is the ordinary I/O timeout for an
    /// explicit endpoint and the short auto-connect timeout for an implicit
    /// one.
    connect_timeout: Duration,
}

impl LiveConfig {
    /// Reads the transport variables from the process environment.
    ///
    /// Returns `None` when no secret is configured (nothing can authenticate),
    /// even if an endpoint is present. A malformed endpoint, secret or port is
    /// an explicit error, never a silent fallback.
    pub fn from_env() -> Option<Result<Self, TransportError>> {
        Self::resolve(
            std::env::var(ENDPOINT_ENV).ok().as_deref(),
            std::env::var(SECRET_ENV).ok().as_deref(),
            std::env::var(BACKEND_ENV).ok().as_deref(),
            std::env::var(PORT_ENV).ok().as_deref(),
        )
    }

    /// Parses an explicit endpoint/secret pair.
    ///
    /// This is the strict form: with no endpoint there is no configuration, so
    /// it keeps [`Self::from_env`]'s explicit-endpoint behaviour.
    pub fn parse(
        endpoint: Option<&str>,
        secret: Option<&str>,
    ) -> Option<Result<Self, TransportError>> {
        let endpoint = endpoint.map(str::trim).filter(|text| !text.is_empty())?;
        let secret = secret.map(str::trim).filter(|text| !text.is_empty())?;
        Some(Endpoint::parse(endpoint).and_then(|endpoint| {
            Secret::parse(secret).map(|secret| Self {
                endpoint,
                secret,
                connect_timeout: DEFAULT_IO_TIMEOUT,
            })
        }))
    }

    /// Resolves the endpoint/secret pair, backend and port without touching
    /// the process environment.
    ///
    /// `endpoint` (when non-empty) takes precedence. Otherwise
    /// `backend == "metal"` selects the ADR-016 implicit endpoint
    /// `127.0.0.1:${port:-47654}` with [`AUTO_CONNECT_TIMEOUT`]. Any other
    /// backend with no endpoint yields `None`.
    pub fn resolve(
        endpoint: Option<&str>,
        secret: Option<&str>,
        backend: Option<&str>,
        port: Option<&str>,
    ) -> Option<Result<Self, TransportError>> {
        let secret = secret.map(str::trim).filter(|text| !text.is_empty())?;
        let secret = || Secret::parse(secret);

        if let Some(endpoint) = endpoint.map(str::trim).filter(|text| !text.is_empty()) {
            return Some(Endpoint::parse(endpoint).and_then(|endpoint| {
                Ok(Self {
                    endpoint,
                    secret: secret()?,
                    connect_timeout: DEFAULT_IO_TIMEOUT,
                })
            }));
        }

        let metal = backend
            .map(str::trim)
            .is_some_and(|value| value.eq_ignore_ascii_case(METAL_BACKEND));
        if !metal {
            return None;
        }

        let port = match port.map(str::trim).filter(|text| !text.is_empty()) {
            Some(text) => match text.parse::<u16>() {
                Ok(0) => {
                    return Some(Err(TransportError::Endpoint(
                        "VICTORY_METAL_PORT: port 0 is not usable".to_string(),
                    )));
                }
                Ok(port) => port,
                Err(_) => {
                    return Some(Err(TransportError::Endpoint(format!(
                        "VICTORY_METAL_PORT: `{text}` is not a port"
                    ))));
                }
            },
            None => DEFAULT_PORT,
        };
        let auto_endpoint = format!("127.0.0.1:{port}");
        Some(Endpoint::parse(&auto_endpoint).and_then(|endpoint| {
            Ok(Self {
                endpoint,
                secret: secret()?,
                connect_timeout: AUTO_CONNECT_TIMEOUT,
            })
        }))
    }

    /// The configured endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// The timeout [`Self::connect`] uses.
    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    /// Opens an authenticated session with the configured timeout.
    pub fn connect(&self) -> Result<WireV3Client, TransportError> {
        self.connect_with_timeout(self.connect_timeout)
    }

    /// Opens an authenticated session with an explicit bounded timeout.
    pub fn connect_with_timeout(&self, timeout: Duration) -> Result<WireV3Client, TransportError> {
        let client = WireV3Client::open(
            self.endpoint.clone(),
            Secret::new(self.secret.bytes())?,
            timeout,
        )?;
        // The queue owns the client; a pipeline worker borrows this weak view so
        // host-side DXIL translation sends on the same authenticated session.
        publish_live_pipeline_session(client.pipeline_session());
        Ok(client)
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
    /// The single session socket, shared with the pipeline-translation seam.
    ///
    /// The guest may create a PSO on a worker thread while the render thread
    /// submits; both use the same authenticated host session, so they share
    /// this mutex and every frame exchange holds it across write and read. A
    /// second socket would be a second host session, which the host does not
    /// serve.
    stream: Arc<Mutex<Option<TcpStream>>>,
    reconnects: u32,
    transport_errors: u64,
    /// The shared completion page, when the guest mapped one and the host
    /// accepted its registration (transport kind 6).
    fence_page: Option<GuestFencePage>,
    /// The page path the session was configured with, re-registered after a
    /// reconnect (a new host session starts from a cleared page).
    fence_page_path: Option<std::path::PathBuf>,
    /// Texture handles registered with the host in this session.
    textures: BTreeSet<u64>,
    /// Sampler-state handles registered with the host in this session.
    samplers: BTreeSet<u64>,
    /// Present surfaces registered with the host in this session.
    surfaces: BTreeSet<u64>,
    /// Guest allocations registered with the host through kind 2.
    ///
    /// The host requires strictly increasing, never reused allocation ids, so
    /// the set is also the guest's record of what may still be named in a
    /// submission. Aliased and placed resources inside one allocation resolve
    /// to the same id and share the first registration.
    allocations: BTreeSet<u64>,
    /// The highest allocation id this session registered (0: none).
    last_allocation: u64,
    /// The highest pipeline ticket this session registered (0: none).
    last_pipeline_ticket: u64,
    /// Native pipeline ids returned by kind 129, by ticket.
    pipelines: std::collections::BTreeMap<u64, u64>,
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

    /// Connects with an explicit completion page path (tests and the
    /// launcher-free host), instead of the one the environment names.
    pub fn connect_with_fence_page(
        endpoint: &str,
        secret: &[u8],
        timeout: Duration,
        page: Option<std::path::PathBuf>,
    ) -> Result<Self, TransportError> {
        Self::open_with(
            Endpoint::parse(endpoint)?,
            Secret::new(secret)?,
            timeout,
            page,
        )
    }

    fn open(endpoint: Endpoint, secret: Secret, timeout: Duration) -> Result<Self, TransportError> {
        let page = crate::native_metal::fence_page_guest::page_path_from_env();
        Self::open_with(endpoint, secret, timeout, page)
    }

    fn open_with(
        endpoint: Endpoint,
        secret: Secret,
        timeout: Duration,
        page: Option<std::path::PathBuf>,
    ) -> Result<Self, TransportError> {
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
            stream: Arc::new(Mutex::new(None)),
            reconnects: 0,
            transport_errors: 0,
            fence_page: None,
            fence_page_path: page,
            textures: BTreeSet::new(),
            samplers: BTreeSet::new(),
            surfaces: BTreeSet::new(),
            allocations: BTreeSet::new(),
            last_allocation: 0,
            last_pipeline_ticket: 0,
            pipelines: std::collections::BTreeMap::new(),
        };
        let mut stream = client.handshake()?;
        client.register_fence_page(&mut stream, client.fence_page_path.clone());
        *client
            .stream
            .lock()
            .unwrap_or_else(|failure| failure.into_inner()) = Some(stream);
        Ok(client)
    }

    /// The socket lock, with a poisoned mutex recovered.
    fn stream_slot(&self) -> std::sync::MutexGuard<'_, Option<TcpStream>> {
        self.stream
            .lock()
            .unwrap_or_else(|failure| failure.into_inner())
    }

    /// Runs one frame exchange on the open socket while holding the shared
    /// stream lock.
    ///
    /// Holding the lock across the write and the read keeps the session in
    /// step when a pipeline registration and a submission are issued from
    /// different guest threads.
    fn with_stream<T>(
        &self,
        exchange: impl FnOnce(&mut TcpStream) -> Result<T, TransportError>,
    ) -> Result<T, NativeQueueError> {
        let mut slot = self.stream_slot();
        let Some(stream) = slot.as_mut() else {
            return Err(NativeQueueError::Transport(
                TransportError::NotConnected.to_string(),
            ));
        };
        exchange(stream).map_err(|error| NativeQueueError::Transport(error.to_string()))
    }

    /// A handle that can send a version 3 pipeline request on this session.
    ///
    /// The handle shares the authenticated socket but holds no registration
    /// state of its own, so it is safe to keep on a pipeline worker thread.
    pub fn pipeline_session(&self) -> PipelineTranslateSession {
        PipelineTranslateSession {
            stream: Arc::downgrade(&self.stream),
            timeout: self.timeout,
        }
    }

    /// Maps the given shared fence page and registers it (kind 6).
    ///
    /// A page that is absent, cannot be mapped, or is refused by the host
    /// stays `None`: the synchronous kind 5 reply remains the only completion
    /// signal, and `GetCompletedValue` never fabricates a value from an
    /// unregistered page.
    fn register_fence_page(
        &mut self,
        stream: &mut TcpStream,
        path: Option<std::path::PathBuf>,
    ) {
        let page = match path {
            Some(path) => GuestFencePage::open(path),
            None => GuestFencePage::from_env(),
        };
        let Some(page) = page else {
            return;
        };
        let mut payload = [0u8; FENCE_PAGE_REGISTER_PAYLOAD_LEN];
        payload.copy_from_slice(&crate::native_metal::fence_page_guest::FENCE_PAGE_LEN.to_le_bytes());
        let status = write_frame(stream, KIND_FENCE_PAGE, &payload, "register fence page", self.timeout)
            .and_then(|()| read_frame(stream, "register fence page", MAX_PAYLOAD_LEN as usize, self.timeout))
            .map(|(kind, payload)| (kind, payload));
        match status {
            Ok((KIND_ACK, payload)) if payload == [0] => {
                tracing::info!(path = %page.path().display(), "shared GPU completion page registered");
                self.fence_page = Some(page);
            }
            other => {
                tracing::warn!(
                    ?other,
                    path = %page.path().display(),
                    "the shared GPU completion page was not registered; the kind 5 reply is the \
                     only completion signal"
                );
            }
        }
    }

    /// The endpoint this client dials.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Whether a stream is open (a client is live until a failure closes it).
    pub fn is_connected(&self) -> bool {
        self.stream_slot().is_some()
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
        let mut stream = self.handshake()?;
        // A new socket is a new host session: the fence page and every
        // registration belong to the old one and must be made again.
        self.fence_page = None;
        self.textures.clear();
        self.samplers.clear();
        self.surfaces.clear();
        self.allocations.clear();
        self.last_allocation = 0;
        self.last_pipeline_ticket = 0;
        self.pipelines.clear();
        let page = self.fence_page_path.clone();
        self.register_fence_page(&mut stream, page);
        *self.stream_slot() = Some(stream);
        self.reconnects = self.reconnects.saturating_add(1);
        Ok(())
    }

    /// Closes the socket. Later sends fail explicitly until `reconnect`.
    pub fn close(&mut self) {
        if let Some(stream) = self.stream_slot().take() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }

    /// The host's published completion fence, when a shared page is mapped
    /// and registered.
    ///
    /// This is one `Ordering::Acquire` load of shared memory: no syscall, no
    /// IPC round trip and no blocking wait. `None` means the page is genuinely
    /// absent (no `VICTORY_METAL_FENCE_PAGE`/share directory, an unusable
    /// file, or a registration the host refused), not that completion is
    /// unknown.
    pub fn completed_fence(&self) -> Option<u64> {
        self.fence_page.as_ref().map(GuestFencePage::completed)
    }

    /// The shared completion page's submitted word, one atomic load.
    pub fn submitted_fence(&self) -> Option<u64> {
        self.fence_page.as_ref().map(GuestFencePage::submitted)
    }

    /// The mapped page, when one is registered.
    pub fn fence_page(&self) -> Option<&GuestFencePage> {
        self.fence_page.as_ref()
    }

    /// Registers one host texture (transport kind 8) and returns whether the
    /// host accepted it.
    ///
    /// The codes are the stable kind 8 codes documented in `ipc_host`: a
    /// dimension (0 D1, 1 D2, 2 D2Array, 3 cube, 4 cube array, 5 D3), a
    /// format (0 Rgba8Unorm .. 8 Depth32FloatStencil8) and a usage (0
    /// sampled, 1 color attachment, 2 depth attachment, 3 UAV). A value
    /// outside them is refused by the host, which is surfaced here as
    /// `false`, never retried and never replaced by a fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn register_texture(
        &mut self,
        id: u64,
        generation: u32,
        dimension: u8,
        format: u8,
        usage: u8,
        sample_count: u8,
        width: u32,
        height: u32,
        depth: u32,
        mip_levels: u16,
        array_layers: u16,
    ) -> Result<bool, NativeQueueError> {
        if id == 0 || generation == 0 {
            return Err(NativeQueueError::Refused(
                "texture registration needs a nonzero id and generation".to_string(),
            ));
        }
        let mut payload = [0u8; TEXTURE_REG_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&id.to_le_bytes());
        payload[8..12].copy_from_slice(&generation.to_le_bytes());
        payload[12] = dimension;
        payload[13] = format;
        payload[14] = usage;
        payload[15] = sample_count;
        payload[16..20].copy_from_slice(&width.to_le_bytes());
        payload[20..24].copy_from_slice(&height.to_le_bytes());
        payload[24..28].copy_from_slice(&depth.to_le_bytes());
        payload[28..30].copy_from_slice(&mip_levels.to_le_bytes());
        payload[30..32].copy_from_slice(&array_layers.to_le_bytes());
        let status = self.registration(KIND_TEXTURE_REG, &payload, "register texture")?;
        let accepted = status == REGISTRATION_ACCEPTED;
        if accepted {
            self.textures.insert(id);
        }
        Ok(accepted)
    }

    /// Registers a texture handle as a present surface (transport kind 9).
    pub fn register_surface(
        &mut self,
        surface_id: u64,
        texture_id: u64,
    ) -> Result<bool, NativeQueueError> {
        let mut payload = [0u8; SURFACE_REG_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&surface_id.to_le_bytes());
        payload[8..16].copy_from_slice(&texture_id.to_le_bytes());
        let status = self.registration(KIND_SURFACE_REG, &payload, "register surface")?;
        let accepted = status == REGISTRATION_ACCEPTED;
        if accepted {
            self.surfaces.insert(surface_id);
        }
        Ok(accepted)
    }

    /// Whether a texture handle was accepted by this session.
    pub fn texture_registered(&self, id: u64) -> bool {
        self.textures.contains(&id)
    }

    /// Registers one reduced `MTLSamplerState` content (transport kind 10).
    ///
    /// The payload is `handle: u64 LE` followed by the five `u32` Metal codes
    /// of [`SamplerStateDesc`]. The caller derives the handle from the content
    /// (`native_metal::sampler::sampler_handle`), so the same descriptor is
    /// registered once and a second frame reuses the handle without a frame.
    /// A rejection is `Ok(false)`; the session stays open.
    pub fn register_sampler(
        &mut self,
        handle: u64,
        desc: &SamplerStateDesc,
    ) -> Result<bool, NativeQueueError> {
        if handle == 0 {
            return Err(NativeQueueError::Refused(
                "sampler registration needs a nonzero handle".to_string(),
            ));
        }
        if self.samplers.contains(&handle) {
            return Ok(true);
        }
        let mut payload = [0u8; SAMPLER_REG_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&handle.to_le_bytes());
        for (index, value) in [
            desc.min_filter,
            desc.mag_filter,
            desc.s_address_mode,
            desc.t_address_mode,
            desc.r_address_mode,
        ]
        .into_iter()
        .enumerate()
        {
            let at = 8 + index * 4;
            payload[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        let status = self.registration(KIND_SAMPLER_REG, &payload, "register sampler")?;
        let accepted = status == REGISTRATION_ACCEPTED;
        if accepted {
            self.samplers.insert(handle);
        }
        Ok(accepted)
    }

    /// Whether a sampler handle was accepted by this session.
    pub fn sampler_registered(&self, handle: u64) -> bool {
        self.samplers.contains(&handle)
    }

    /// Sampler registrations this session made.
    pub fn sampler_registrations(&self) -> usize {
        self.samplers.len()
    }

    /// Whether a present surface was accepted by this session.
    pub fn surface_registered(&self, id: u64) -> bool {
        self.surfaces.contains(&id)
    }

    /// Registers one guest host-visible allocation (transport kind 2).
    ///
    /// The payload is exactly `allocation_id: u64 LE, byte_len: u64 LE`, the
    /// layout of `ipc_host::REGISTER_PAYLOAD_LEN`. `byte_len` must be the
    /// *whole* share file the id names: the host reopens
    /// `<share dir>/{id:016x}.bin` and refuses a mapping whose length differs,
    /// which also closes the session, so the caller derives it from the file
    /// and never guesses (see `native_metal::pso::share_allocation_len`).
    ///
    /// Registration is idempotent: an id this session already registered
    /// returns `Ok(true)` without touching the wire, which is what aliased and
    /// placed resources inside one allocation rely on.
    ///
    /// The host accepts only ids greater than every id it has seen, and a
    /// registration it refuses closes the session. An id at or below the last
    /// accepted one is therefore refused *locally* as an explicit error (the
    /// host has no kind 2 form that carries an arbitrary id), never sent.
    pub fn register_allocation(
        &mut self,
        allocation_id: u64,
        byte_len: u64,
    ) -> Result<bool, NativeQueueError> {
        if allocation_id == 0 {
            return Err(NativeQueueError::Refused(
                "buffer registration needs a nonzero allocation id".to_string(),
            ));
        }
        if byte_len == 0 {
            return Err(NativeQueueError::Refused(format!(
                "buffer registration for allocation {allocation_id} names no bytes"
            )));
        }
        if self.allocations.contains(&allocation_id) {
            return Ok(true);
        }
        if allocation_id <= self.last_allocation {
            return Err(NativeQueueError::Refused(format!(
                "allocation {allocation_id} is below the last host registration {}; the kind 2 \
                 form carries no arbitrary id, so the host cannot accept it",
                self.last_allocation
            )));
        }
        let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&allocation_id.to_le_bytes());
        payload[8..16].copy_from_slice(&byte_len.to_le_bytes());
        let status = self.registration(KIND_REGISTER, &payload, "register allocation")?;
        if status != REGISTRATION_ACCEPTED {
            return Err(NativeQueueError::Refused(format!(
                "the host refused allocation {allocation_id} with status {status}"
            )));
        }
        self.allocations.insert(allocation_id);
        self.last_allocation = allocation_id;
        Ok(true)
    }

    /// Whether this session registered `allocation_id` with the host.
    pub fn allocation_registered(&self, allocation_id: u64) -> bool {
        self.allocations.contains(&allocation_id)
    }

    /// The highest allocation id registered in this session (0: none).
    pub fn last_allocation(&self) -> u64 {
        self.last_allocation
    }

    /// Registers one precompiled library as a pipeline (transport kind 4).
    ///
    /// The payload is `ticket: u64 LE, entry_len: u16 LE, entry bytes`; the
    /// host maps the ticket to `<share dir>/{ticket:016x}.metallib` itself, so
    /// no path or library bytes cross the wire. The reply is kind 129: one
    /// status byte, then the native pipeline id.
    ///
    /// An explicit host rejection is [`NativeQueueError::Refused`] and leaves
    /// the session open. A malformed reply is a protocol error and closes it.
    /// Tickets must be nonzero and strictly increasing, exactly like the
    /// host's own ledger.
    pub fn register_pipeline(
        &mut self,
        ticket: u64,
        entry: &str,
    ) -> Result<u64, NativeQueueError> {
        if ticket == 0 {
            return Err(NativeQueueError::Refused(
                "pipeline registration needs a nonzero ticket".to_string(),
            ));
        }
        if ticket <= self.last_pipeline_ticket {
            return Err(NativeQueueError::Refused(format!(
                "pipeline ticket {ticket} is not greater than the last accepted ticket {}",
                self.last_pipeline_ticket
            )));
        }
        if !valid_pipeline_entry(entry) {
            return Err(NativeQueueError::Refused(format!(
                "pipeline entry `{entry}` is not a bounded ASCII Metal identifier"
            )));
        }
        let entry_len = u16::try_from(entry.len()).expect("entry length is bounded");
        let mut payload = Vec::with_capacity(PIPELINE_HEADER_LEN + entry.len());
        payload.extend_from_slice(&ticket.to_le_bytes());
        payload.extend_from_slice(&entry_len.to_le_bytes());
        payload.extend_from_slice(entry.as_bytes());

        let result = self.with_stream(|stream| {
            write_frame(stream, KIND_PIPELINE, &payload, "register pipeline", self.timeout)
                .and_then(|()| {
                    read_frame(
                        stream,
                        "register pipeline",
                        MAX_PAYLOAD_LEN as usize,
                        self.timeout,
                    )
                })
        });
        match result {
            Ok((KIND_PIPELINE_REPLY, reply)) => {
                match (reply.first().copied(), reply.len()) {
                    (Some(PIPELINE_REPLY_OK), PIPELINE_REPLY_OK_LEN) => {
                        let id = u64::from_le_bytes(
                            reply[1..].try_into().expect("eight bytes"),
                        );
                        if id == 0 {
                            return Err(NativeQueueError::Transport(
                                "the host answered pipeline registration with id zero".to_string(),
                            ));
                        }
                        self.last_pipeline_ticket = ticket;
                        self.pipelines.insert(ticket, id);
                        Ok(id)
                    }
                    (Some(PIPELINE_REPLY_REJECTED), 1) => Err(NativeQueueError::Refused(
                        format!("the host rejected pipeline ticket {ticket}"),
                    )),
                    _ => Err(NativeQueueError::Transport(format!(
                        "malformed kind 129 reply of {} byte(s)",
                        reply.len()
                    ))),
                }
            }
            Ok((actual, _)) => {
                self.close();
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(NativeQueueError::Transport(
                    TransportError::UnexpectedKind {
                        expected: KIND_PIPELINE_REPLY,
                        actual,
                    }
                    .to_string(),
                ))
            }
            Err(error) => {
                self.close();
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(error)
            }
        }
    }

    /// Registers a multi-stage graphics, mesh or compute pipeline (kind 4 v2).
    ///
    /// `stages` is one `(wire stage, ticket, entry)` per shader; each ticket
    /// names the `{ticket:016x}.metallib` the caller already wrote. `state`,
    /// `abi` and `root_signature` are the encoded `PipelineState`, the `VAB1`
    /// per-stage ABI container and the raw root-signature bytes. Compute also
    /// accepts the v1 form; this method is the only way to express graphics and
    /// mesh.
    ///
    /// Tickets must be nonzero and strictly increasing, exactly like the host's
    /// ledger. The host validates every field and either registers the whole
    /// pipeline or answers one rejection byte; there is no partial pipeline.
    pub fn register_pipeline_v2(
        &mut self,
        kind: u8,
        stages: &[(u8, u64, &str)],
        state: &[u8],
        abi: &[u8],
        root_signature: &[u8],
    ) -> Result<u64, NativeQueueError> {
        if !matches!(
            kind,
            PIPELINE_KIND_COMPUTE | PIPELINE_KIND_GRAPHICS | PIPELINE_KIND_MESH
        ) {
            return Err(NativeQueueError::Refused(format!(
                "pipeline kind {kind} is not compute, graphics or mesh"
            )));
        }
        if stages.is_empty() || stages.len() > PIPELINE_V2_MAX_STAGES {
            return Err(NativeQueueError::Refused(format!(
                "pipeline registration needs 1..={PIPELINE_V2_MAX_STAGES} stages, got {}",
                stages.len()
            )));
        }
        if state.len() > MAX_PIPELINE_STATE_BYTES
            || abi.len() > MAX_PIPELINE_ABI_BYTES
            || root_signature.len() > MAX_ROOT_SIGNATURE_BYTES
        {
            return Err(NativeQueueError::Refused(
                "pipeline state, ABI or root signature exceeds its limit".to_string(),
            ));
        }
        let mut previous_ticket = self.last_pipeline_ticket;
        let mut seen = Vec::with_capacity(stages.len());
        for (stage, ticket, entry) in stages {
            if *ticket == 0 || *ticket <= previous_ticket {
                return Err(NativeQueueError::Refused(format!(
                    "pipeline ticket {ticket} is not greater than the last accepted ticket \
                     {previous_ticket}"
                )));
            }
            if !matches!(
                *stage,
                PIPELINE_STAGE_VERTEX
                    | PIPELINE_STAGE_PIXEL
                    | PIPELINE_STAGE_MESH
                    | PIPELINE_STAGE_OBJECT
                    | PIPELINE_STAGE_COMPUTE
            ) || seen.contains(stage)
            {
                return Err(NativeQueueError::Refused(format!(
                    "pipeline stage {stage} is unknown or repeated"
                )));
            }
            if !valid_pipeline_entry(entry) {
                return Err(NativeQueueError::Refused(format!(
                    "pipeline entry `{entry}` is not a bounded ASCII Metal identifier"
                )));
            }
            previous_ticket = *ticket;
            seen.push(*stage);
        }

        let mut payload = Vec::with_capacity(
            PIPELINE_V2_HEADER_LEN
                + stages.iter().map(|(_, _, entry)| 11 + entry.len()).sum::<usize>()
                + state.len()
                + abi.len()
                + root_signature.len(),
        );
        payload.extend_from_slice(&PIPELINE_FRAME_MAGIC);
        payload.push(PIPELINE_FRAME_VERSION_V2);
        payload.push(kind);
        payload.push(u8::try_from(stages.len()).expect("stage count is bounded"));
        payload.push(0);
        payload.extend_from_slice(
            &u32::try_from(state.len())
                .expect("state length is bounded")
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &u32::try_from(abi.len())
                .expect("ABI length is bounded")
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &u32::try_from(root_signature.len())
                .expect("root signature length is bounded")
                .to_le_bytes(),
        );
        payload.extend_from_slice(&0u32.to_le_bytes());
        for (stage, ticket, entry) in stages {
            payload.push(*stage);
            payload.extend_from_slice(&ticket.to_le_bytes());
            payload.extend_from_slice(
                &u16::try_from(entry.len())
                    .expect("entry length is bounded")
                    .to_le_bytes(),
            );
            payload.extend_from_slice(entry.as_bytes());
        }
        payload.extend_from_slice(state);
        payload.extend_from_slice(abi);
        payload.extend_from_slice(root_signature);
        if payload.len() > MAX_PAYLOAD_LEN as usize {
            return Err(NativeQueueError::Refused(format!(
                "pipeline registration is {} bytes, over the {MAX_PAYLOAD_LEN}-byte transport limit",
                payload.len()
            )));
        }

        let result = self.with_stream(|stream| {
            write_frame(
                stream,
                KIND_PIPELINE,
                &payload,
                "register pipeline v2",
                self.timeout,
            )
            .and_then(|()| {
                read_frame(
                    stream,
                    "register pipeline v2",
                    MAX_PAYLOAD_LEN as usize,
                    self.timeout,
                )
            })
        });
        match result {
            Ok((KIND_PIPELINE_REPLY, reply)) => {
                match (reply.first().copied(), reply.len()) {
                    (Some(PIPELINE_REPLY_OK), PIPELINE_REPLY_OK_LEN) => {
                        let id =
                            u64::from_le_bytes(reply[1..].try_into().expect("eight bytes"));
                        if id == 0 {
                            return Err(NativeQueueError::Transport(
                                "the host answered pipeline registration with id zero".to_string(),
                            ));
                        }
                        self.last_pipeline_ticket = previous_ticket;
                        for (_, ticket, _) in stages {
                            self.pipelines.insert(*ticket, id);
                        }
                        Ok(id)
                    }
                    (Some(PIPELINE_REPLY_REJECTED), 1) => Err(NativeQueueError::Refused(
                        "the host rejected the version 2 pipeline registration".to_string(),
                    )),
                    _ => Err(NativeQueueError::Transport(format!(
                        "malformed kind 129 reply of {} byte(s)",
                        reply.len()
                    ))),
                }
            }
            Ok((actual, _)) => {
                self.close();
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(NativeQueueError::Transport(
                    TransportError::UnexpectedKind {
                        expected: KIND_PIPELINE_REPLY,
                        actual,
                    }
                    .to_string(),
                ))
            }
            Err(error) => {
                self.close();
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(error)
            }
        }
    }

    /// Registers a pipeline whose DXIL stages the host translates itself
    /// (kind 4 version 3).
    ///
    /// The frame carries only share-relative `stages/<blake3>.bin` names, the
    /// wire stage codes, the entry points, the encoded `PipelineState` and an
    /// optional root-signature file name. The host owns the Rosetta
    /// `victory-irconverter --stdio` pool, derives the per-stage ABI from the
    /// converter reflection it receives and answers with the native id. A
    /// rejected frame answers status 1 and, unlike the older frames, an
    /// optional bounded UTF-8 reason.
    pub fn register_pipeline_v3(
        &self,
        kind: u8,
        stages: &[Vp3Stage<'_>],
        state: &[u8],
        root_signature_file: Option<&str>,
    ) -> Result<u64, NativeQueueError> {
        let payload = encode_pipeline_v3_request(kind, stages, state, root_signature_file)?;
        self.round_trip_pipeline_v3(&payload)
    }

    fn round_trip_pipeline_v3(&self, payload: &[u8]) -> Result<u64, NativeQueueError> {
        let result = self.with_stream(|stream| {
            write_frame(
                stream,
                KIND_PIPELINE,
                payload,
                "register pipeline v3",
                self.timeout,
            )
            .and_then(|()| {
                read_frame(
                    stream,
                    "register pipeline v3",
                    MAX_PAYLOAD_LEN as usize,
                    self.timeout,
                )
            })
        });
        decode_pipeline_v3_reply(result)
    }

    /// The native pipeline id a ticket registered, if any.
    pub fn pipeline_registered(&self, ticket: u64) -> Option<u64> {
        self.pipelines.get(&ticket).copied()
    }

    /// The next ticket this session may use, so tickets never repeat.
    pub fn next_pipeline_ticket(&self) -> u64 {
        self.last_pipeline_ticket.saturating_add(1).max(1)
    }

    /// Native pipeline registrations this session made.
    pub fn pipeline_registrations(&self) -> usize {
        self.pipelines.len()
    }

    /// One registration request/acknowledgement pair on the open stream.
    fn registration(
        &mut self,
        kind: u8,
        payload: &[u8],
        operation: &'static str,
    ) -> Result<u8, NativeQueueError> {
        let result = self.with_stream(|stream| {
            write_frame(stream, kind, payload, operation, self.timeout).and_then(|()| {
                read_frame(stream, operation, MAX_PAYLOAD_LEN as usize, self.timeout)
            })
        });
        match result {
            Ok((KIND_ACK, status)) if status.len() == 1 => Ok(status[0]),
            Ok((actual, _)) => {
                self.close();
                Err(NativeQueueError::Transport(
                    TransportError::UnexpectedKind {
                        expected: KIND_ACK,
                        actual,
                    }
                    .to_string(),
                ))
            }
            Err(error) => {
                self.close();
                self.transport_errors = self.transport_errors.saturating_add(1);
                Err(error)
            }
        }
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
        let outcome = self.with_stream(|stream| Ok(self.round_trip(stream, request)))?;
        match outcome {
            Outcome::Status(status) => Ok(status),
            // The host answered a complete frame: the session is still in step,
            // so the stream stays open.
            Outcome::Refused(why) => Err(NativeQueueError::Refused(why)),
            Outcome::Broken(error) => {
                // The stream may be desynchronised: close it so the next send
                // must reconnect instead of reading a stale reply.
                self.close();
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
        let reply = match decode_reply_with_reason(&payload) {
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
            other => {
                let mut message = format!(
                    "the host answered {other:?} for sequence {}",
                    reply.sequence
                );
                if let Some(reason) = reply.reason {
                    // The reason is the host's own diagnostic. It is bounded and
                    // sanitised once here, recorded in telemetry, and logged as
                    // the one stable line a boot log can be grepped for.
                    let record = crate::native_metal::telemetry::note_host_error(
                        reply.sequence,
                        &reason,
                    );
                    tracing::error!(
                        "{}",
                        crate::native_metal::telemetry::host_error_line(
                            record.sequence,
                            &record.reason
                        )
                    );
                    message.push_str(": ");
                    message.push_str(&record.reason);
                }
                Outcome::Refused(message)
            }
        }
    }
}

/// Decodes one kind 129 reply to a version 3 pipeline request.
///
/// Status 0 carries the `u64` native id. Status 1 carries either the legacy
/// bare rejection byte or `reason_len u16` + at most
/// [`MAX_PIPELINE_REJECTION_REASON`] UTF-8 bytes, so a failed host translation
/// names its category without ever carrying a title path or shader bytes.
fn decode_pipeline_v3_reply(
    result: Result<(u8, Vec<u8>), NativeQueueError>,
) -> Result<u64, NativeQueueError> {
    match result {
        Ok((KIND_PIPELINE_REPLY, reply)) => match (reply.first().copied(), reply.len()) {
            (Some(PIPELINE_REPLY_OK), PIPELINE_REPLY_OK_LEN) => {
                let id = u64::from_le_bytes(reply[1..].try_into().expect("eight bytes"));
                if id == 0 {
                    return Err(NativeQueueError::Transport(
                        "the host answered pipeline registration with id zero".to_string(),
                    ));
                }
                Ok(id)
            }
            (Some(PIPELINE_REPLY_REJECTED), 1) => Err(NativeQueueError::Refused(
                "the host rejected the version 3 pipeline registration".to_string(),
            )),
            (Some(PIPELINE_REPLY_REJECTED), len) if len >= 3 => {
                let reason_len = usize::from(u16::from_le_bytes(
                    reply[1..3].try_into().expect("two bytes"),
                ));
                if reason_len == 0
                    || reason_len > MAX_PIPELINE_REJECTION_REASON
                    || reply.len() != 3 + reason_len
                {
                    return Err(NativeQueueError::Transport(format!(
                        "malformed version 3 pipeline rejection of {} byte(s)",
                        reply.len()
                    )));
                }
                let reason = std::str::from_utf8(&reply[3..]).map_err(|_| {
                    NativeQueueError::Transport(
                        "the version 3 pipeline rejection reason is not UTF-8".to_string(),
                    )
                })?;
                Err(NativeQueueError::Refused(reason.to_string()))
            }
            _ => Err(NativeQueueError::Transport(format!(
                "malformed kind 129 reply of {} byte(s)",
                reply.len()
            ))),
        },
        Ok((actual, _)) => Err(NativeQueueError::Transport(
            TransportError::UnexpectedKind {
                expected: KIND_PIPELINE_REPLY,
                actual,
            }
            .to_string(),
        )),
        Err(error) => Err(error),
    }
}

/// A borrowed view of the live session's socket for host-side pipeline
/// translation.
///
/// It holds a weak reference: when the session client is dropped or closed the
/// handle still upgrades, but the socket slot is `None`, so a registration
/// fails explicitly instead of keeping a dead connection alive.
#[derive(Clone)]
pub struct PipelineTranslateSession {
    stream: Weak<Mutex<Option<TcpStream>>>,
    timeout: Duration,
}

impl PipelineTranslateSession {
    /// Registers a pipeline whose stages the host translates (kind 4 version 3).
    pub fn register_pipeline_v3(
        &self,
        kind: u8,
        stages: &[Vp3Stage<'_>],
        state: &[u8],
        root_signature_file: Option<&str>,
    ) -> Result<u64, NativeQueueError> {
        let payload = encode_pipeline_v3_request(kind, stages, state, root_signature_file)?;
        let Some(stream) = self.stream.upgrade() else {
            return Err(NativeQueueError::Transport(
                TransportError::NotConnected.to_string(),
            ));
        };
        let result = {
            let mut slot = stream.lock().unwrap_or_else(|failure| failure.into_inner());
            let Some(stream) = slot.as_mut() else {
                return Err(NativeQueueError::Transport(
                    TransportError::NotConnected.to_string(),
                ));
            };
            write_frame(
                stream,
                KIND_PIPELINE,
                &payload,
                "register pipeline v3",
                self.timeout,
            )
            .and_then(|()| {
                read_frame(
                    stream,
                    "register pipeline v3",
                    MAX_PAYLOAD_LEN as usize,
                    self.timeout,
                )
            })
            .map_err(|error| NativeQueueError::Transport(error.to_string()))
        };
        decode_pipeline_v3_reply(result)
    }
}

/// The most recent session opened through [`LiveConfig`].
///
/// The queue owns the `WireV3Client`; a pipeline worker borrows this weak
/// session view so both send on the one authenticated socket instead of opening
/// a second connection the host would never serve.
static LIVE_PIPELINE_SESSION: Mutex<Option<PipelineTranslateSession>> = Mutex::new(None);

/// The live session a pipeline worker may use, when a client has connected.
pub fn live_pipeline_session() -> Option<PipelineTranslateSession> {
    LIVE_PIPELINE_SESSION
        .lock()
        .unwrap_or_else(|failure| failure.into_inner())
        .clone()
}

fn publish_live_pipeline_session(session: PipelineTranslateSession) {
    *LIVE_PIPELINE_SESSION
        .lock()
        .unwrap_or_else(|failure| failure.into_inner()) = Some(session);
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
    use victory_metal::wire_v3::{encode_submission_v3, CommandV3, SubmissionV3};

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

        fn write_reply_with_reason(
            stream: &mut TcpStream,
            sequence: u32,
            status: ReplyStatus,
            reason: &str,
        ) {
            let reply = victory_metal::wire::encode_reply_with_reason(
                &Reply { sequence, status },
                Some(reason),
            )
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

    /// A Wire v3 error reply that carries a host reason is surfaced verbatim in
    /// the refusal error, recorded once in telemetry and formatted as the stable
    /// `METAL_HOST_ERROR` line.
    #[test]
    fn a_host_error_reason_reaches_the_guest_refusal_and_telemetry() {
        let reason = "Metal execution failed: target texture 640x360 was refused";
        let before = crate::native_metal::telemetry::host_error_count();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, _payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            TestHost::write_reply_with_reason(
                &mut stream,
                SEQUENCE,
                ReplyStatus::ExecutionFailed,
                reason,
            );
            let mut byte = [0u8; 1];
            assert!(matches!(stream.read(&mut byte), Ok(0) | Err(_)));
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        let error = client
            .send(&encoded(SEQUENCE, 100))
            .expect_err("an ExecutionFailed reply is a refusal");
        let NativeQueueError::Refused(message) = &error else {
            panic!("expected a host refusal, got {error:?}");
        };
        assert!(message.contains(reason), "{message}");
        assert_eq!(
            crate::native_metal::telemetry::host_error_count(),
            before + 1,
            "the reason is counted once"
        );
        let record = crate::native_metal::telemetry::last_host_error().expect("last host error");
        assert_eq!(record.sequence, SEQUENCE);
        assert_eq!(record.reason, reason);
        assert_eq!(
            crate::native_metal::telemetry::host_error_line(record.sequence, &record.reason),
            format!("METAL_HOST_ERROR sequence={SEQUENCE} reason={reason}")
        );
        assert!(client.is_connected(), "a refusal keeps the one stream open");
    }

    /// A legacy reply with no trailer keeps the exact pre-ADR-035 refusal text
    /// and does not move the host-error telemetry.
    #[test]
    fn an_absent_reason_trailer_keeps_the_legacy_refusal_text() {
        let before = crate::native_metal::telemetry::host_error_count();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, _payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            TestHost::write_reply(&mut stream, SEQUENCE, ReplyStatus::ExecutionFailed);
            let mut byte = [0u8; 1];
            assert!(matches!(stream.read(&mut byte), Ok(0) | Err(_)));
        });

        let mut client =
            WireV3Client::connect_with_timeout(&host.endpoint(), &SECRET, TEST_TIMEOUT)
                .expect("authenticated connect");
        let error = client
            .send(&encoded(SEQUENCE, 100))
            .expect_err("an ExecutionFailed reply is a refusal");
        assert_eq!(
            error,
            NativeQueueError::Refused(format!(
                "the host answered ExecutionFailed for sequence {SEQUENCE}"
            ))
        );
        assert_eq!(
            crate::native_metal::telemetry::host_error_count(),
            before,
            "an absent trailer must not be counted"
        );
    }

    /// The guest decoder accepts present and absent trailers and refuses an
    /// overlong one instead of guessing at it.
    #[test]
    fn a_v3_reply_reason_trailer_decodes_present_absent_and_oversized() {
        use victory_metal::wire::{
            decode_reply_with_reason, encode_reply, encode_reply_with_reason,
            MAX_REPLY_REASON_BYTES, REPLY_REASON_MAGIC,
        };

        let reply = Reply {
            sequence: SEQUENCE,
            status: ReplyStatus::ExecutionFailed,
        };

        let present = encode_reply_with_reason(&reply, Some("exact host text"))
            .expect("encode present trailer");
        let detail = decode_reply_with_reason(&present).expect("decode present trailer");
        assert_eq!(detail.reason.as_deref(), Some("exact host text"));

        let absent = encode_reply(&reply).expect("encode legacy reply");
        let detail = decode_reply_with_reason(&absent).expect("decode legacy reply");
        assert_eq!(detail.reason, None);

        let mut payload = Vec::new();
        payload.extend_from_slice(&2u16.to_le_bytes()); // ExecutionFailed
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&REPLY_REASON_MAGIC);
        payload.extend_from_slice(&((MAX_REPLY_REASON_BYTES as u16) + 1).to_le_bytes());
        payload.extend(std::iter::repeat(b'x').take(MAX_REPLY_REASON_BYTES + 1));
        let mut oversized = Vec::new();
        oversized.extend_from_slice(&MAGIC);
        oversized.extend_from_slice(&2u16.to_le_bytes()); // protocol version
        oversized.extend_from_slice(&2u16.to_le_bytes()); // reply kind
        oversized.extend_from_slice(&SEQUENCE.to_le_bytes());
        oversized.extend_from_slice(&0u16.to_le_bytes());
        oversized.extend_from_slice(&0u16.to_le_bytes());
        oversized.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        oversized.extend_from_slice(&payload);
        assert!(
            decode_reply_with_reason(&oversized).is_err(),
            "an overlong trailer must be refused"
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

    fn secret_text() -> String {
        "ab".repeat(SECRET_LEN)
    }

    #[test]
    fn metal_backend_auto_connects_to_the_default_loopback_endpoint() {
        let secret = secret_text();
        let config = LiveConfig::resolve(None, Some(&secret), Some("metal"), None)
            .expect("a secret is configured")
            .expect("auto-connect resolves");
        assert_eq!(config.endpoint().to_string(), "127.0.0.1:47654");
        assert_eq!(config.connect_timeout(), AUTO_CONNECT_TIMEOUT);

        let config = LiveConfig::resolve(None, Some(&secret), Some("METAL"), Some("50123"))
            .expect("a secret is configured")
            .expect("auto-connect resolves");
        assert_eq!(config.endpoint().to_string(), "127.0.0.1:50123");
        assert_eq!(config.endpoint().port(), 50123);
    }

    #[test]
    fn an_explicit_endpoint_beats_backend_auto_connect() {
        let secret = secret_text();
        let config = LiveConfig::resolve(
            Some("127.0.0.1:1234"),
            Some(&secret),
            Some("metal"),
            Some("50123"),
        )
        .expect("a secret is configured")
        .expect("explicit endpoint resolves");
        assert_eq!(config.endpoint().to_string(), "127.0.0.1:1234");
        assert_eq!(config.connect_timeout(), DEFAULT_IO_TIMEOUT);
    }

    #[test]
    fn auto_connect_requires_the_metal_backend_and_a_secret() {
        let secret = secret_text();
        assert!(LiveConfig::resolve(None, Some(&secret), Some("d3d12"), None).is_none());
        assert!(LiveConfig::resolve(None, Some(&secret), None, None).is_none());
        assert!(LiveConfig::resolve(None, None, Some("metal"), None).is_none());
        assert!(LiveConfig::resolve(None, Some("  "), Some("metal"), None).is_none());
    }

    #[test]
    fn an_invalid_auto_connect_port_is_an_explicit_endpoint_error() {
        let secret = secret_text();
        for port in ["0", "65536", "abc"] {
            let error = LiveConfig::resolve(None, Some(&secret), Some("metal"), Some(port))
                .expect("a secret is configured")
                .expect_err("a bad port is not silently defaulted");
            let text = error.to_string();
            assert!(
                text.contains("VICTORY_METAL_PORT") || text.contains("not a port"),
                "the error names the port setting: {text}"
            );
            assert!(!text.contains("abab"), "the secret is not named: {text}");
        }
    }

    #[test]
    fn a_failed_auto_connect_names_the_implicit_endpoint() {
        // Bind and drop so the port is almost certainly free.
        let port = {
            let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .expect("bind loopback");
            listener.local_addr().expect("address").port()
        };
        let secret = secret_text();
        let config =
            LiveConfig::resolve(None, Some(&secret), Some("metal"), Some(&port.to_string()))
                .expect("a secret is configured")
                .expect("auto-connect resolves");
        assert_eq!(config.connect_timeout(), AUTO_CONNECT_TIMEOUT);
        let error = config
            .connect()
            .expect_err("nothing listens on the closed port");
        let text = error.to_string();
        assert!(
            text.contains(&format!("127.0.0.1:{port}")),
            "the implicit endpoint is named: {text}"
        );
        assert!(!text.contains("abab"), "the secret is not named: {text}");
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

    /// A kind 2 registration is `id: u64 LE, byte_len: u64 LE`; a repeat of an
    /// id already accepted sends nothing, and an id that would not increase is
    /// refused locally instead of closing the host's session.
    #[test]
    fn buffer_registration_is_idempotent_and_ascending() {
        let (sender, receiver) = mpsc::channel::<Vec<(u64, u64)>>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let mut seen = Vec::new();
            while let Some((kind, payload)) = TestHost::try_read_frame(&mut stream) {
                assert_eq!(kind, KIND_REGISTER, "only kind 2 frames are expected");
                assert_eq!(payload.len(), REGISTER_PAYLOAD_LEN);
                let id = u64::from_le_bytes(payload[0..8].try_into().expect("eight bytes"));
                let len = u64::from_le_bytes(payload[8..16].try_into().expect("eight bytes"));
                seen.push((id, len));
                TestHost::write_frame(&mut stream, KIND_ACK, &[0]);
            }
            sender.send(seen).expect("report the frames");
        });

        let mut client =
            WireV3Client::connect_with_fence_page(&host.endpoint(), &SECRET, TEST_TIMEOUT, None)
                .expect("authenticated connect");
        assert_eq!(client.register_allocation(3, 4096), Ok(true));
        assert_eq!(
            client.register_allocation(3, 4096),
            Ok(true),
            "a repeated range is idempotent"
        );
        assert_eq!(client.register_allocation(5, 8192), Ok(true));
        assert_eq!(client.register_allocation(5, 8192), Ok(true));
        let refused = client
            .register_allocation(4, 1024)
            .expect_err("an id below the last accepted one cannot be sent");
        assert!(matches!(refused, NativeQueueError::Refused(_)));
        assert!(client.allocation_registered(5));
        assert!(!client.allocation_registered(4));
        assert_eq!(client.last_allocation(), 5);
        drop(client);
        assert_eq!(
            receiver.recv_timeout(TEST_TIMEOUT).expect("frame list"),
            vec![(3, 4096), (5, 8192)],
            "two frames, ascending, and nothing for the repeats"
        );
    }

    /// A kind 4 frame is `ticket: u64 LE, entry_len: u16 LE, entry`; the host
    /// answers kind 129 with one status byte and the native pipeline id.
    #[test]
    fn pipeline_registration_round_trips_kind_4_and_129() {
        let (sender, receiver) = mpsc::channel::<Vec<(u64, String)>>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let mut seen = Vec::new();
            for id in [0x51u64, 0x52] {
                let (kind, payload) = TestHost::read_frame(&mut stream);
                assert_eq!(kind, KIND_PIPELINE, "one kind 4 request per PSO");
                assert_eq!(payload.len(), PIPELINE_HEADER_LEN + 4);
                let ticket = u64::from_le_bytes(payload[0..8].try_into().expect("eight bytes"));
                let entry_len =
                    u16::from_le_bytes(payload[8..10].try_into().expect("two bytes"));
                assert_eq!(usize::from(entry_len), payload.len() - PIPELINE_HEADER_LEN);
                let entry = String::from_utf8(payload[PIPELINE_HEADER_LEN..].to_vec())
                    .expect("an ASCII entry");
                seen.push((ticket, entry));
                let mut reply = vec![PIPELINE_REPLY_OK];
                reply.extend_from_slice(&id.to_le_bytes());
                TestHost::write_frame(&mut stream, KIND_PIPELINE_REPLY, &reply);
            }
            sender.send(seen).expect("report the frames");
        });

        let mut client =
            WireV3Client::connect_with_fence_page(&host.endpoint(), &SECRET, TEST_TIMEOUT, None)
                .expect("authenticated connect");
        assert_eq!(client.next_pipeline_ticket(), 1);
        assert_eq!(client.register_pipeline(1, "main"), Ok(0x51));
        assert_eq!(client.pipeline_registered(1), Some(0x51));
        assert_eq!(client.next_pipeline_ticket(), 2);
        assert_eq!(client.register_pipeline(2, "main"), Ok(0x52));
        assert_eq!(client.pipeline_registrations(), 2);
        assert!(matches!(
            client.register_pipeline(1, "main"),
            Err(NativeQueueError::Refused(_))
        ));
        assert!(
            matches!(
                client.register_pipeline(3, "../escape"),
                Err(NativeQueueError::Refused(_))
            ),
            "an entry that is not a Metal identifier never reaches the wire"
        );
        drop(client);
        assert_eq!(
            receiver.recv_timeout(TEST_TIMEOUT).expect("frame list"),
            vec![(1, "main".to_string()), (2, "main".to_string())]
        );
    }

    /// A version 2 frame carries the kind, per-stage tickets/entries and the
    /// state/ABI/root-signature blobs; the reply is the same kind 129 id.
    #[test]
    fn pipeline_registration_v2_carries_stages_state_and_abi() {
        let (sender, receiver) = mpsc::channel::<Vec<u8>>();
        let host = TestHost::spawn(move |listener| {
            let mut stream = accept_authenticated(&listener, &SECRET);
            let (kind, payload) = TestHost::read_frame(&mut stream);
            assert_eq!(kind, KIND_PIPELINE);
            sender.send(payload).expect("report the frame");
            let mut reply = vec![PIPELINE_REPLY_OK];
            reply.extend_from_slice(&0x77u64.to_le_bytes());
            TestHost::write_frame(&mut stream, KIND_PIPELINE_REPLY, &reply);
        });

        let mut client =
            WireV3Client::connect_with_fence_page(&host.endpoint(), &SECRET, TEST_TIMEOUT, None)
                .expect("authenticated connect");
        assert_eq!(
            client.register_pipeline_v2(
                PIPELINE_KIND_GRAPHICS,
                &[
                    (PIPELINE_STAGE_VERTEX, 1, "main"),
                    (PIPELINE_STAGE_PIXEL, 2, "main"),
                ],
                b"VPS1state",
                b"VAB1abi",
                b"root-signature",
            ),
            Ok(0x77)
        );
        assert_eq!(client.pipeline_registered(1), Some(0x77));
        assert_eq!(client.pipeline_registered(2), Some(0x77));
        assert_eq!(client.next_pipeline_ticket(), 3);
        drop(client);

        let payload = receiver.recv_timeout(TEST_TIMEOUT).expect("one frame");
        assert_eq!(&payload[..4], &PIPELINE_FRAME_MAGIC);
        assert_eq!(payload[4], PIPELINE_FRAME_VERSION_V2);
        assert_eq!(payload[5], PIPELINE_KIND_GRAPHICS);
        assert_eq!(payload[6], 2, "two stage entries");
        assert_eq!(payload[7], 0);
        let state_len = u32::from_le_bytes(payload[8..12].try_into().expect("four bytes")) as usize;
        let abi_len = u32::from_le_bytes(payload[12..16].try_into().expect("four bytes")) as usize;
        let root_len = u32::from_le_bytes(payload[16..20].try_into().expect("four bytes")) as usize;
        assert_eq!((state_len, abi_len, root_len), (9, 7, 14));
        let entries_len = 2 * (1 + 8 + 2 + "main".len());
        assert_eq!(
            &payload[PIPELINE_V2_HEADER_LEN + entries_len..][..state_len],
            b"VPS1state"
        );
    }

    /// The version 3 control frame names share-relative DXIL files only, so it
    /// stays far under the 64 KiB transport frame and round-trips exactly.
    #[test]
    fn a_version_3_pipeline_frame_round_trips_and_rejects_paths() {
        let vertex = "stages/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bin";
        let pixel = "stages/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.bin";
        let root = "stages/cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc.bin";
        let stages = [
            Vp3Stage {
                stage: PIPELINE_STAGE_VERTEX,
                file: vertex,
                entry: "vs_main",
            },
            Vp3Stage {
                stage: PIPELINE_STAGE_PIXEL,
                file: pixel,
                entry: "ps_main",
            },
        ];
        let payload = encode_pipeline_v3_request(
            PIPELINE_KIND_GRAPHICS,
            &stages,
            b"VPS1state",
            Some(root),
        )
        .expect("encode version 3 frame");
        assert!(payload.len() < MAX_PAYLOAD_LEN as usize);
        let decoded = decode_pipeline_v3_request(&payload).expect("decode version 3 frame");
        assert_eq!(decoded.kind, PIPELINE_KIND_GRAPHICS);
        assert_eq!(decoded.state, b"VPS1state");
        assert_eq!(decoded.root_signature_file, root);
        assert_eq!(
            decoded.stages,
            vec![
                (PIPELINE_STAGE_VERTEX, vertex.to_string(), "vs_main".to_string()),
                (PIPELINE_STAGE_PIXEL, pixel.to_string(), "ps_main".to_string()),
            ]
        );

        for bad in [
            "/stages/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bin",
            "stages/../aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bin",
            "stages\\aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bin",
            "stages/AAAA.bin",
        ] {
            let error = encode_pipeline_v3_request(
                PIPELINE_KIND_COMPUTE,
                &[Vp3Stage {
                    stage: PIPELINE_STAGE_COMPUTE,
                    file: bad,
                    entry: "main",
                }],
                &[],
                None,
            )
            .expect_err("a path is not a stage reference");
            assert!(matches!(error, NativeQueueError::Refused(_)), "{bad}");
        }
    }
}
