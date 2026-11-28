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
//! | 4 | guest to host | v1 `ticket, entry_len, entry`, or the v2 pipeline frame |
//! | 5 | guest to host | one version 3 request envelope (below) |
//! | 5 | host to guest | one complete encoded [`crate::wire::Reply`] |
//! | 6 | guest to host | `byte_len: u64 LE`, must be 4096; shared fence page |
//! | 7 | guest to host | one version 3 request envelope, no reply frame |
//! | 8 | guest to host | `texture registration` (below) |
//! | 9 | guest to host | `surface registration` (below) |
//! | 10 | guest to host | `sampler registration` (below) |
//! | 128 | host to guest | one byte: 0 = accepted, 1 = rejected |
//! | 129 | host to guest | one byte status, then `pipeline_id: u64 LE` on success |
//!
//! # Sampler registration
//!
//! Kind 10 carries a fixed 28-byte reduced sampler registration: `handle: u64
//! LE`, then five `u32 LE` Metal codes (`min_filter`, `mag_filter`,
//! `s_address_mode`, `t_address_mode`, `r_address_mode`). The host creates the
//! `MTLSamplerState` through
//! [`DirectMetalTransport::register_sampler`] and remembers its content, so a
//! repeated handle with the same content is idempotent and a conflicting
//! content is rejected. A v3 sampler descriptor-table entry must name a handle
//! registered by kind 10 in this session; an unknown handle is `UnknownHandle`
//! before any encoder exists.
//!
//! # Texture and present-surface registration
//!
//! Kind 8 carries a fixed 32-byte texture registration:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | texture id, nonzero |
//! | 8 | 4 | generation, nonzero |
//! | 12 | 1 | dimension: 0=1D, 1=2D, 2=2D array, 3=cube, 4=cube array, 5=3D |
//! | 13 | 1 | format: see [`texture_format_from_code`] |
//! | 14 | 1 | usage: 0=sampled, 1=colour attachment, 2=depth attachment, 3=storage |
//! | 15 | 1 | sample count: 1, 2, 4 or 8 |
//! | 16 | 4 | width |
//! | 20 | 4 | height |
//! | 24 | 4 | depth |
//! | 28 | 2 | mip levels, at least 1 |
//! | 30 | 2 | array layers, at least 1 |
//!
//! The host creates the texture through
//! [`DirectMetalTransport::register_registry_texture`] and remembers its
//! generation. A later v3 `BeginRenderPass` colour or depth attachment names
//! `(id, generation)`, exactly like a registered buffer allocation; the session
//! allocation snapshot resolves that pair to the texture handle. Duplicate ids
//! (including an id already used by a buffer allocation), zero extents, zero
//! mip/layer counts, an unsupported dimension/format/usage and an out-of-range
//! sample count are refused with status 1 and create no GPU work.
//!
//! Kind 9 carries `surface_id: u64 LE` and `texture_id: u64 LE`. The texture
//! must already have been registered by kind 8; the host registers it as a
//! present surface through
//! [`DirectMetalTransport::register_present_surface`]. A duplicate surface id or
//! an unregistered texture is refused with status 1.
//!
//! Unlike kinds 2 and 6, kinds 8, 9 and 10 answer every request with kind 128
//! and leave the session open on a rejection, so a guest can correct a
//! descriptor and retry. No path crosses the wire: the texture or sampler is
//! device-created from the validated descriptor.
//!
//! # Shared completion fence page (ADR-012)
//!
//! Kind 6 is opt-in and registers the launcher-created fence page file
//! `<canonical share dir>/gpu-fence.v1.page` with the session. The payload is a
//! single `byte_len: u64 LE` that must equal 4096; the path itself never
//! crosses the wire, exactly like allocation registration. The host maps the
//! file `MAP_SHARED` and answers kind 128. A rejection closes the session.
//!
//! Once registered, every version 3 submission (kind 5 and kind 7) publishes
//! its sequence as `submitted_fence` before execution. `completed_fence` is
//! published only for work that is actually finished: an asynchronous
//! transport (ADR-017) publishes the contiguous completion watermark from its
//! `addCompletedHandler:` block straight into the guest-visible page, while a
//! synchronous executor — one that reports `false` from
//! [`BatchExecutor::attach_fence_page`] — keeps the older behaviour and the host
//! publishes on `ReplyStatus::Ok`. A failed, refused or replayed submission
//! never advances `completed_fence`, and publishing uses `fetch_max`, so a stale
//! replay can never lower it. Registration starts the page with
//! [`crate::fence_page::FencePage::begin_session`], which clears both fences and
//! advances the epoch; the page file must therefore be created for the session
//! and must not be shared with a different host.
//!
//! Kind 7 is the fire-and-forget path: the same version 3 envelope as kind 5,
//! executed with the same validation, ledger and fence publication, but with no
//! reply frame at all. The guest learns success from `completed_fence` and
//! failure from the fence not advancing. A malformed kind 7 frame closes the
//! session because there is no reply in which to report the error.
//!
//! The host submit path no longer waits: an accepted submission is committed and
//! returns, and the guest observes completion through the shared page. Detaching
//! the page at session end drains the GPU pipeline first, so a completion
//! handler can never publish into an unmapped page.
//!
//! # Version 3 request envelope
//!
//! Kind 5 carries a 16-byte envelope followed by exactly `submission_len`
//! bytes, which are an encoded [`crate::wire_v3::SubmissionV3`] decoded by
//! [`crate::wire_v3::decode_submission_v3`]:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 4 | magic `V3RQ` |
//! | 4 | 2 | protocol version, must be 3 |
//! | 6 | 4 | submission sequence, nonzero |
//! | 10 | 2 | reserved, must be zero |
//! | 12 | 4 | encoded submission length, at most [`MAX_SUBMISSION_V3_LEN`] |
//! | 16 | N | encoded version 3 submission |
//!
//! The envelope sequence must equal the encoded submission's sequence. A bad
//! version, reserved word, declared length, inner frame or trailing byte is
//! answered with [`crate::wire::ReplyStatus::ExecutionFailed`] and leaves the
//! session open, so the guest can retry the same sequence. Only a frame with no
//! recoverable sequence (short envelope or wrong magic) is answered and then
//! closes the connection.
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
//! A request whose first four bytes are [`PIPELINE_FRAME_MAGIC`] is the
//! version 2 form. It names the pipeline kind, one `(stage, ticket, entry)` per
//! stage, the encoded [`crate::macos::registry::PipelineState`], a `VAB1`
//! per-stage ABI container and the raw root-signature bytes. Every family is
//! created from the stage libraries the frame names: the vertex function comes
//! from the vertex stage's own metallib and the fragment function from the pixel
//! stage's own metallib (and likewise for mesh/object/pixel, while the staged
//! compute form resolves the kernel in the compute stage's own library), so the
//! persistent `--stdio` pool's one-library-per-stage output registers directly.
//! Every field is validated before the executor runs, so a malformed frame
//! creates no GPU work and consumes no ticket.
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
use std::net::{SocketAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use thiserror::Error;

use crate::abi::{VmBufferId, VmPipelineId};
use crate::cmdlist::{ListLedger, RingError};
use crate::compiler_pool::{CompileJob, CompilerPool, PoolError, PoolStats};
use crate::fence_page::{FencePage, FencePageMapping, Wake};
use crate::guest_ipc::{MAX_PIPELINE_ENTRY_LEN, valid_pipeline_entry};
use crate::macos::bindings::RootSignatureLayout;
use crate::macos::device::{DirectMetalTransport, PipelineAbi, SamplerDesc, bounded_gpu_reason};
use crate::macos::objc::{self, ObjcObject};
use crate::macos::registry::{
    FunctionRole, PipelineStage as RegistryPipelineStage, PipelineState, StageLibrary, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage,
};
use crate::macos::vertex_layout::VertexLayout;
use crate::transport::{GuestTransport, MetalError};
use crate::wire::{
    self, BufferBinding, Command, ListIdentity, Reply, ReplyStatus, Submission, WireError,
};
use crate::wire_v3::{
    decode_submission_v3, CommandV3, DescriptorTableEntryV3, SubmissionV3, PROTOCOL_VERSION_V3,
};

/// Maximum payload length accepted by the transport envelope.
pub const MAX_PAYLOAD_LEN: u32 = 65_536;
/// Transport frame kind for a version 3 submission request and its reply.
pub const KIND_SUBMISSION_V3: u8 = 5;
/// Magic tag of the version 3 request envelope.
pub const V3_REQUEST_MAGIC: [u8; 4] = *b"V3RQ";
/// Fixed header of a version 3 request envelope.
pub const V3_REQUEST_HEADER_LEN: usize = 16;
/// Largest encoded [`SubmissionV3`] accepted in one request.
///
/// A 1024-command frame whose commands are all maximum-size descriptor tables
/// is slightly over 1 MiB, so 4 MiB leaves bounded headroom while still
/// refusing a hostile declared length before any allocation.
pub const MAX_SUBMISSION_V3_LEN: usize = 4 * 1024 * 1024;
/// Maximum transport payload length accepted for a version 3 frame.
pub const MAX_PAYLOAD_LEN_V3: u32 = V3_REQUEST_HEADER_LEN as u32 + MAX_SUBMISSION_V3_LEN as u32;
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
/// Frame kind for an opt-in shared fence page registration.
pub const KIND_FENCE_PAGE: u8 = 6;
/// Frame kind for a fire-and-forget version 3 submission with no reply frame.
pub const KIND_SUBMISSION_ASYNC: u8 = 7;
/// Frame kind for host texture registration.
pub const KIND_TEXTURE_REG: u8 = 8;
/// Frame kind for present-surface registration.
pub const KIND_SURFACE_REG: u8 = 9;
/// Frame kind for reduced `MTLSamplerState` registration.
pub const KIND_SAMPLER_REG: u8 = 10;
/// Exact length of a texture registration payload.
pub const TEXTURE_REG_PAYLOAD_LEN: usize = 32;
/// Exact length of a present-surface registration payload.
pub const SURFACE_REG_PAYLOAD_LEN: usize = 16;
/// Exact length of a sampler registration payload (`handle` plus five `u32`s).
pub const SAMPLER_REG_PAYLOAD_LEN: usize = 8 + 5 * 4;
/// Status byte of an accepted texture or surface registration.
pub const REGISTRATION_ACCEPTED: u8 = 0;
/// Status byte of a rejected texture or surface registration.
pub const REGISTRATION_REJECTED: u8 = 1;
/// Fixed file name of the shared fence page inside the canonical share dir.
pub const FENCE_PAGE_FILE_NAME: &str = "gpu-fence.v1.page";
/// Exact length of a fence page registration payload (`byte_len: u64 LE`).
pub const FENCE_PAGE_REGISTER_PAYLOAD_LEN: usize = 8;
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
/// First four bytes of a version 2 kind 4 request. A version 1 request starts
/// with its nonzero `u64` ticket, so the two forms are selected by this magic
/// word; a v1 ticket whose low four bytes happen to equal it is refused rather
/// than guessed at.
pub const PIPELINE_FRAME_MAGIC: [u8; 4] = *b"VP2\0";
/// Version byte of the version 2 pipeline frame.
pub const PIPELINE_FRAME_VERSION_V2: u8 = 1;
/// Fixed header length of a version 2 pipeline frame.
pub const PIPELINE_V2_HEADER_LEN: usize = 24;
/// Largest number of stage entries in one version 2 frame (vertex, pixel,
/// mesh, object).
pub const PIPELINE_V2_MAX_STAGES: usize = 4;
/// Version 2 pipeline kinds.
pub const PIPELINE_KIND_COMPUTE: u8 = 0;
/// A vertex/fragment render pipeline.
pub const PIPELINE_KIND_GRAPHICS: u8 = 1;
/// A mesh/object/fragment render pipeline.
pub const PIPELINE_KIND_MESH: u8 = 2;
/// Version 2 per-stage codes (independent of the registry's stage enum).
pub const PIPELINE_STAGE_VERTEX: u8 = 0;
/// Pixel/fragment stage.
pub const PIPELINE_STAGE_PIXEL: u8 = 1;
/// Mesh stage.
pub const PIPELINE_STAGE_MESH: u8 = 2;
/// Object stage.
pub const PIPELINE_STAGE_OBJECT: u8 = 3;
/// Compute stage.
pub const PIPELINE_STAGE_COMPUTE: u8 = 4;
/// Largest encoded `PipelineState` blob accepted in a version 2 frame.
pub const MAX_PIPELINE_STATE_BYTES: usize = 4 * 1024;
/// Largest encoded pipeline ABI accepted in a version 2 frame.
pub const MAX_PIPELINE_ABI_BYTES: usize = 16 * 1024;
/// Largest raw root-signature blob accepted in a version 2 frame.
pub const MAX_ROOT_SIGNATURE_BYTES: usize = 16 * 1024;
/// Magic of the per-stage ABI container carried in a version 2 frame.
pub const PIPELINE_ABI_MAGIC: [u8; 4] = *b"VAB1";
/// First four bytes of a version 3 kind 4 request. The frame names
/// content-hashed DXIL stage files instead of carrying libraries or shader
/// bytes, so a 16 MiB stage travels through the share directory, never the
/// 64 KiB transport frame.
pub const PIPELINE_FRAME_MAGIC_V3: [u8; 4] = *b"VP3\0";
/// Version byte of the version 3 pipeline frame.
pub const PIPELINE_FRAME_VERSION_V3: u8 = 1;
/// Fixed header length of a version 3 pipeline frame.
pub const PIPELINE_V3_HEADER_LEN: usize = 24;
/// Exact length of a share-relative stage name: `stages/<64 lowercase hex>.bin`.
pub const STAGE_FILE_NAME_LEN: usize = "stages/".len() + 64 + ".bin".len();
/// Largest converter rejection reason a version 3 reply may carry.
pub const MAX_PIPELINE_REJECTION_REASON: usize = 256;

/// One stage of a version 2 registration: the stage's own library and the entry
/// point resolved inside it.
///
/// A version 2 frame names one ticket per stage, so a graphics or mesh pipeline
/// can be built from distinct metallibs (the persistent `--stdio` pool emits one
/// per stage). The slice is per stage on purpose: nothing downstream may assume
/// a single library holds every function.
#[derive(Debug, Clone, Copy)]
pub struct PipelineStageLibrary<'a> {
    /// Wire stage code, one of [`PIPELINE_STAGE_VERTEX`] to
    /// [`PIPELINE_STAGE_COMPUTE`].
    pub stage: u8,
    /// The stage's own metallib bytes.
    pub metallib: &'a [u8],
    /// Entry point the executor resolves inside that library.
    pub entry: &'a str,
}

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
    /// Executes one decoded version 3 submission over the registered allocation
    /// snapshot.
    ///
    /// The default deliberately refuses the request, so an executor that cannot
    /// replay a stateful stream fails loudly with [`MetalError::NotSupported`]
    /// instead of silently dropping opcodes.
    fn submit_v3(
        &self,
        _submission: &SubmissionV3,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        Err(MetalError::NotSupported(
            "version 3 submission is not enabled by this executor",
        ))
    }
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

    /// Registers a content-hashed graphics pipeline under `handle`.
    ///
    /// The default refuses the request; an executor that owns an ADR-013
    /// registry implements it through
    /// [`DirectMetalTransport::register_content_hashed_graphics_pipeline`].
    /// `state` is the encoded [`PipelineState`] the registry decodes, and `abi`
    /// is the per-stage root layout stored with the handle.
    fn register_graphics_pipeline(
        &self,
        _handle: u64,
        _metallib: &[u8],
        _vertex_entry: &str,
        _state: &[u8],
        _abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "graphics pipeline registration is not enabled by this executor",
        ))
    }

    /// Registers a content-hashed mesh pipeline under `handle`.
    ///
    /// The default refuses the request; see [`Self::register_graphics_pipeline`].
    fn register_mesh_pipeline(
        &self,
        _handle: u64,
        _metallib: &[u8],
        _mesh_entry: &str,
        _state: &[u8],
        _abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "mesh pipeline registration is not enabled by this executor",
        ))
    }

    /// Registers a content-hashed graphics pipeline from one library per stage.
    ///
    /// `stages` carries the vertex and pixel functions with the library each one
    /// lives in; the executor must resolve every function in its own library and
    /// create one `MTLRenderPipelineState`. A missing function is an explicit
    /// error naming the stage, never a fallback to another library.
    ///
    /// The default refuses the request: an executor that only implements the
    /// single-library entry point must not silently build both functions from
    /// the vertex library.
    fn register_graphics_pipeline_stages(
        &self,
        _handle: u64,
        _stages: &[PipelineStageLibrary<'_>],
        _state: &[u8],
        _abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "per-stage graphics pipeline registration is not enabled by this executor",
        ))
    }

    /// Registers a content-hashed mesh pipeline from one library per stage.
    ///
    /// The default refuses the request; see
    /// [`Self::register_graphics_pipeline_stages`].
    fn register_mesh_pipeline_stages(
        &self,
        _handle: u64,
        _stages: &[PipelineStageLibrary<'_>],
        _state: &[u8],
        _abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "per-stage mesh pipeline registration is not enabled by this executor",
        ))
    }

    /// Registers a content-hashed compute pipeline from one library per stage.
    ///
    /// A compute pipeline has one kernel function, but the staged form keeps the
    /// kernel in the library its own stage names rather than an assumed first
    /// library.
    fn register_compute_pipeline_stages(
        &self,
        _handle: u64,
        _stages: &[PipelineStageLibrary<'_>],
        _state: &[u8],
        _abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "per-stage compute pipeline registration is not enabled by this executor",
        ))
    }

    /// Creates a registry texture under `id`.
    ///
    /// The default deliberately refuses the request, so texture registration is
    /// opt-in for an executor that owns an ADR-013 registry. A rejection must
    /// leave the existing handle space untouched.
    fn register_texture(&self, _id: u64, _desc: TextureDesc) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "texture registration is not enabled by this executor",
        ))
    }

    /// Registers an already-created registry texture as a present surface.
    ///
    /// The default deliberately refuses the request. The surface maps a wire
    /// `surface_id` to a live texture handle so a v3 `Present` can validate it.
    fn register_surface(&self, _surface_id: u64, _texture_id: u64) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "surface registration is not enabled by this executor",
        ))
    }

    /// Creates (or reuses) a reduced `MTLSamplerState` under `id`.
    ///
    /// The default deliberately refuses the request. An executor that owns a
    /// sampler registry implements it through
    /// [`DirectMetalTransport::register_sampler`]; a second registration for the
    /// same id and content is idempotent, so a reconverted table cannot churn
    /// sampler objects.
    fn register_sampler(&self, _id: u64, _desc: SamplerDesc) -> Result<(), MetalError> {
        Err(MetalError::NotSupported(
            "sampler registration is not enabled by this executor",
        ))
    }

    /// Attaches the host-mapped completion page this executor must publish into.
    ///
    /// Returns `true` when the executor took ownership of the page and publishes
    /// `completed_fence` itself from its GPU completion handler. `false` (the
    /// default) means the executor finishes a submission synchronously, so the
    /// host publishes the completion when the executor returns `Ok` — the
    /// pre-ADR-017 contract for an executor without a GPU completion handler.
    ///
    /// The returned page is reference-counted, so an executor that keeps it
    /// cannot be left with an unmapped page; one that cannot publish
    /// asynchronously must not retain it.
    fn attach_fence_page(&self, _page: Arc<FencePageMapping>) -> Result<bool, MetalError> {
        Ok(false)
    }

    /// Releases a page attached by [`Self::attach_fence_page`].
    ///
    /// Called when the session ends. An executor that publishes asynchronously
    /// must drain its GPU pipeline here, so no completion handler can publish
    /// into a page the caller is about to unmap.
    fn detach_fence_page(&self) {}
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

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        DirectMetalTransport::submit_v3(self, submission, allocations)
    }

    fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        DirectMetalTransport::load_metallib(self, metallib, entry)
    }

    fn register_graphics_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        vertex_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        DirectMetalTransport::register_content_hashed_graphics_pipeline(
            self, handle, metallib, vertex_entry, state, abi,
        )
    }

    fn register_mesh_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        mesh_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        DirectMetalTransport::register_content_hashed_mesh_pipeline(
            self, handle, metallib, mesh_entry, state, abi,
        )
    }

    fn register_graphics_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        let stages = registry_stages(stages)?;
        DirectMetalTransport::register_content_hashed_graphics_pipeline_stages(
            self, handle, &stages, state, abi,
        )
    }

    fn register_mesh_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        let stages = registry_stages(stages)?;
        DirectMetalTransport::register_content_hashed_mesh_pipeline_stages(
            self, handle, &stages, state, abi,
        )
    }

    fn register_compute_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        let stages = registry_stages(stages)?;
        DirectMetalTransport::register_content_hashed_compute_pipeline_stages(
            self, handle, &stages, state, abi,
        )
    }

    fn register_texture(&self, id: u64, desc: TextureDesc) -> Result<(), MetalError> {
        DirectMetalTransport::register_registry_texture(self, id, desc)
    }

    fn register_sampler(&self, id: u64, desc: SamplerDesc) -> Result<(), MetalError> {
        DirectMetalTransport::register_sampler(self, id, desc)
    }

    fn register_surface(&self, surface_id: u64, texture_id: u64) -> Result<(), MetalError> {
        // The registry texture already lives in `state.textures` (see
        // `register_registry_texture`); retain that exact object and wrap it in
        // a `RenderTarget` so the existing `register_present_surface` path can
        // insert it into the surface table. No device or render module change is
        // needed, and the surface always aliases the already-registered texture.
        let retained = {
            let state = self.lock_state()?;
            let texture = state.textures.get(&texture_id).ok_or_else(|| {
                MetalError::NotFound(format!("texture handle {texture_id} is not registered"))
            })?;
            // SAFETY: `texture` is a live retained MTLTexture owned by the
            // locked device state; `retain` returns it with one extra +1.
            unsafe { objc::msg_send_id_0(texture.as_raw(), objc::sel(c"retain")) }
        };
        let retained = ObjcObject::from_retained(retained).ok_or_else(|| {
            MetalError::InternalError("registered texture retain returned null".to_string())
        })?;
        let target = crate::macos::render::RenderTarget::from_objc(retained)?;
        DirectMetalTransport::register_present_surface(self, surface_id, &target)
    }

    fn attach_fence_page(&self, page: Arc<FencePageMapping>) -> Result<bool, MetalError> {
        DirectMetalTransport::attach_shared_fence_page(self, page)?;
        Ok(true)
    }

    fn detach_fence_page(&self) {
        // Drains the GPU pipeline before the host unmaps the page, so no
        // completion handler can publish into released memory.
        let _ = DirectMetalTransport::detach_shared_fence_page(self);
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

/// Whether `name` is the only shape a version 3 stage reference may take:
/// `stages/<64 lowercase hex digits>.bin`.
///
/// An absolute path, `..`, backslash, extra separator or symlink target cannot
/// be expressed, and the host re-joins it under the canonical share directory.
fn valid_stage_file_name(name: &str) -> bool {
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

/// Reads one `stages/<blake3>.bin` DXIL stage from the canonical share dir.
///
/// The name is validated before the join and the final component is opened
/// `O_NOFOLLOW`; the descriptor's own metadata is rechecked so a path swapped
/// to a symlink or FIFO after the first stat cannot smuggle bytes in. A stage
/// may be up to 16 MiB, which is exactly why it never travels in the control
/// frame.
fn read_stage_file(share_dir: &Path, name: &str) -> Result<Vec<u8>, HostError> {
    if !valid_stage_file_name(name) {
        return Err(HostError::Protocol(
            "stage file name is not a share-relative content hash",
        ));
    }
    let path = share_dir.join(name);
    let before = std::fs::symlink_metadata(&path)?;
    if !before.file_type().is_file() {
        return Err(HostError::Protocol("stage backing must be a regular file"));
    }
    let announced = before.len();
    if announced == 0 || announced > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol(
            "stage backing is empty or exceeds the 16 MiB limit",
        ));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(&path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != announced || metadata.len() > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol("stage backing changed while opening"));
    }
    let mut bytes = Vec::with_capacity(announced as usize);
    let read = (&mut file)
        .take(MAX_METALLIB_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if read as u64 != announced || bytes.is_empty() || bytes.len() as u64 > MAX_METALLIB_BYTES {
        return Err(HostError::Protocol("stage backing changed while reading"));
    }
    Ok(bytes)
}

/// One process-wide translator pool, created on the first host-translated
/// pipeline (ADR-038).
///
/// The Win64 guest cannot run the x86_64 macOS converter, so the ARM64 daemon
/// owns the bounded Rosetta `--stdio` pool. Creating it lazily keeps a boot
/// that never needs a host translation from starting any worker.
static SHARED_COMPILER_POOL: OnceLock<Arc<CompilerPool>> = OnceLock::new();

fn shared_compiler_pool() -> Arc<CompilerPool> {
    Arc::clone(
        SHARED_COMPILER_POOL.get_or_init(|| Arc::new(CompilerPool::from_env())),
    )
}

/// The shared pool's statistics, without creating it.
pub fn shared_pool_stats() -> Option<PoolStats> {
    SHARED_COMPILER_POOL.get().map(|pool| pool.stats())
}

/// Stable `pool_status` string for host telemetry.
pub fn shared_pool_status() -> &'static str {
    match SHARED_COMPILER_POOL.get() {
        Some(pool) if pool.worker_program().is_some() => "ready",
        Some(_) => "cache-only",
        None => "skipped",
    }
}

struct RegisteredAllocation {
    /// Logical length from the registration frame; bindings are checked
    /// against this, not the page-rounded mapping length.
    byte_len: u64,
    buffer_id: VmBufferId,
    /// Kept alive until the Metal wrapper has been unregistered.
    _mapping: FileMapping,
}

/// Metadata of a texture registered through kind 8.
#[derive(Debug, Clone, Copy)]
struct RegisteredTexture {
    /// Generation the guest named in the registration; v3 attachments must
    /// repeat it so a stale handle can never resolve.
    generation: u32,
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

/// Live counters of one session, readable after `serve_once` returns in either
/// direction.
///
/// [`SessionSummary`] is only returned on a clean disconnect. A session that
/// fails on a transport or protocol error still answered frames, and this
/// snapshot lets the daemon publish them instead of reporting zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStats {
    /// Submission frames answered with a wire reply.
    pub submissions: u64,
    /// Answered frames whose status was not `Ok`.
    pub refused_submissions: u64,
    /// Allocations registered during the session.
    pub allocations: usize,
    /// Sum of logical allocation lengths in bytes.
    pub bytes_mapped: u64,
    /// Bounded reason for the last refusal or GPU failure.
    pub last_error: Option<String>,
    /// Successfully host-translated stages per wire stage kind, in the order
    /// vertex, pixel, mesh, object, compute.
    pub host_translated_stages: [u64; 5],
    /// Host-translated pipelines registered on the native device.
    pub host_pipeline_translations_ok: u64,
    /// Host-side translation or registration refusals.
    pub host_pipeline_translations_refused: u64,
    /// Stages answered from the host pool's content-hashed disk cache.
    pub host_pipeline_cache_hits: u64,
    /// Last bounded host-translation refusal reason.
    pub host_translation_reason: Option<String>,
    /// Pool status: `ready`, `cache-only` or `skipped`.
    pub pool_status: &'static str,
    /// Shared pool counters (all zero when the pool was never created).
    pub pool: PoolStats,
}

/// Synchronous single-peer host session.
///
/// The caller accepts one loopback stream and passes it to
/// [`IpcHost::serve_once`]. The host holds all mappings and Metal wrappers until
/// [`IpcHost::release_all`] or drop. A daemon serving several connections keeps
/// one `IpcHost` per connection and may share the executor's device and the
/// content-pipeline handle space across them.
pub struct IpcHost<E: BatchExecutor> {
    secret: HostSecret,
    share_dir: PathBuf,
    io_timeout: Duration,
    page_size: usize,
    executor: E,
    allocations: BTreeMap<u64, RegisteredAllocation>,
    /// Textures registered by kind 8, keyed by the handle a v3 attachment names.
    textures: BTreeMap<u64, RegisteredTexture>,
    /// Sampler states registered by kind 10, keyed by their handle.
    ///
    /// The descriptor is kept so a re-registration of the same handle is
    /// idempotent only when the content matches; a conflicting content is an
    /// explicit rejection rather than a silent replacement.
    samplers: BTreeMap<u64, SamplerDesc>,
    /// Present surfaces registered by kind 9, mapped to their texture handle.
    surfaces: BTreeMap<u64, u64>,
    last_allocation_id: u64,
    last_pipeline_ticket: u64,
    /// Next wire handle for a content-hashed version 2 or version 3 pipeline
    /// (graphics, mesh and compute). The v1 compute request keeps using the
    /// executor's own `load_pipeline` id space.
    ///
    /// A daemon that serves several connections shares one device registry, so
    /// two sessions must never name the same handle;
    /// [`Self::with_content_pipeline_handles`] injects one process-wide
    /// counter. A standalone host (tests) gets a private counter starting at 1.
    next_content_pipeline: Arc<AtomicU64>,
    /// Why the last kind 4 request was refused, with the executor's stage-named
    /// diagnostic when there is one. The wire answer stays one status byte.
    last_pipeline_rejection: Option<String>,
    /// Opt-in shared completion page registered by kind 6.
    ///
    /// The mapping is reference-counted because the executor keeps a clone while
    /// it publishes completions from its GPU completion handler.
    fence_page: Option<Arc<FencePageMapping>>,
    /// True when the executor publishes `completed_fence` itself.
    ///
    /// An asynchronous transport (ADR-017) publishes from the GPU completion
    /// handler, so the host must not publish after the executor merely accepted
    /// the submission. A synchronous executor keeps the old contract and the
    /// host publishes on `ReplyStatus::Ok`.
    fence_page_attached: bool,
    next_sequence: u32,
    submissions: u64,
    /// Answered frames whose status was not `Ok`.
    refused_submissions: u64,
    /// Bounded reason for the last refusal or GPU failure; sticky across a
    /// later success so telemetry keeps the first diagnosable fault.
    last_error: Option<String>,
    /// The host-owned Rosetta translator pool (ADR-038).
    ///
    /// `None` until the first version 3 pipeline request; `with_compiler_pool`
    /// lets a test inject a pool with a fake worker.
    compiler_pool: Option<Arc<CompilerPool>>,
    /// Host-translated stages per wire stage kind (vertex, pixel, mesh, object,
    /// compute).
    host_translated_stages: [u64; 5],
    /// Host-translated pipelines that registered on the native device.
    host_pipeline_translations_ok: u64,
    /// Host-side translation or registration refusals.
    host_pipeline_translations_refused: u64,
    /// Stages answered by the host pool's content-hashed disk cache.
    host_pipeline_cache_hits: u64,
    /// Last bounded host-translation refusal reason.
    host_translation_reason: Option<String>,
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
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            surfaces: BTreeMap::new(),
            last_allocation_id: 0,
            last_pipeline_ticket: 0,
            next_content_pipeline: Arc::new(AtomicU64::new(1)),
            last_pipeline_rejection: None,
            fence_page: None,
            fence_page_attached: false,
            next_sequence: 1,
            submissions: 0,
            refused_submissions: 0,
            last_error: None,
            compiler_pool: None,
            host_translated_stages: [0; 5],
            host_pipeline_translations_ok: 0,
            host_pipeline_translations_refused: 0,
            host_pipeline_cache_hits: 0,
            host_translation_reason: None,
            ledger: ListLedger::new(),
        })
    }

    /// Injects the translator pool a host-translated pipeline uses.
    ///
    /// Production leaves this unset and takes the process-wide lazy pool; tests
    /// inject a pool with a fake worker so translation needs no hardware.
    pub fn with_compiler_pool(mut self, pool: Arc<CompilerPool>) -> Self {
        self.compiler_pool = Some(pool);
        self
    }

    /// The pool this host is using, if a host translation asked for one.
    pub fn compiler_pool(&self) -> Option<&Arc<CompilerPool>> {
        self.compiler_pool.as_ref()
    }

    /// Overrides the per-read/per-write socket timeout.
    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// Shares one content-pipeline handle space across host sessions.
    ///
    /// The daemon serves each connection with its own [`IpcHost`] but one
    /// native device and registry, so a handle minted by a translation session
    /// must be unique with respect to the render session's handles. Passing the
    /// same `Arc<AtomicU64>` to every session gives that guarantee; without it
    /// the host keeps a private counter (the historical single-session
    /// behaviour).
    pub fn with_content_pipeline_handles(mut self, handles: Arc<AtomicU64>) -> Self {
        self.next_content_pipeline = handles;
        self
    }

    /// Mints the next content-pipeline handle in the shared id space.
    ///
    /// Zero is never a valid wire handle, so a wrapped counter skips it.
    fn allocate_content_pipeline_handle(&self) -> u64 {
        loop {
            let handle = self.next_content_pipeline.fetch_add(1, Ordering::Relaxed);
            if handle != 0 {
                return handle;
            }
        }
    }

    /// Returns a handle whose registration failed, when no later allocation
    /// raced past it.
    ///
    /// A refused frame must consume no pipeline id (the historical single
    /// session contract). The compare-exchange gives the handle back only while
    /// it is still the newest one; under concurrency a lost race leaves a gap,
    /// which is harmless because handles only have to be unique and nonzero.
    fn release_content_pipeline_handle(&self, handle: u64) {
        let _ = self.next_content_pipeline.compare_exchange(
            handle.wrapping_add(1),
            handle,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Why the last kind 4 request was refused, if it was.
    ///
    /// The wire reply is one status byte, so this host-side record carries the
    /// executor's stage-named diagnostic (a pixel library that does not contain
    /// the pixel entry point, for example) without changing the protocol.
    /// A successful registration clears it.
    pub fn last_pipeline_rejection(&self) -> Option<&str> {
        self.last_pipeline_rejection.as_deref()
    }

    /// Session counters and the last bounded refusal reason.
    ///
    /// Safe to call after [`Self::serve_once`] returned an error: the counters
    /// are updated as each frame is answered, not only on a clean summary.
    pub fn session_stats(&self) -> SessionStats {
        SessionStats {
            submissions: self.submissions,
            refused_submissions: self.refused_submissions,
            allocations: self.allocations.len(),
            bytes_mapped: self
                .allocations
                .values()
                .map(|allocation| allocation.byte_len)
                .sum(),
            last_error: self.last_error.clone(),
            host_translated_stages: self.host_translated_stages,
            host_pipeline_translations_ok: self.host_pipeline_translations_ok,
            host_pipeline_translations_refused: self.host_pipeline_translations_refused,
            host_pipeline_cache_hits: self.host_pipeline_cache_hits,
            host_translation_reason: self.host_translation_reason.clone(),
            // A host that never used its own pool reports the process-wide
            // state: a later session in the same daemon sees the live pool.
            pool_status: match self.compiler_pool.as_ref() {
                Some(pool) if pool.worker_program().is_some() => "ready",
                Some(_) => "cache-only",
                None => shared_pool_status(),
            },
            pool: self
                .compiler_pool
                .as_ref()
                .map(|pool| pool.stats())
                .or_else(shared_pool_stats)
                .unwrap_or_default(),
        }
    }

    /// The last bounded refusal or GPU failure reason, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Records one answered frame whose status was not `Ok`.
    ///
    /// The wire reason is never raw guest memory: callers pass either Metal's
    /// own diagnostic or a host-built protocol description, and it is bounded
    /// and flattened here before it can reach telemetry.
    fn note_refusal(&mut self, status: ReplyStatus, reason: impl AsRef<str>) {
        debug_assert_ne!(status, ReplyStatus::Ok, "a refusal cannot be Ok");
        self.refused_submissions = self.refused_submissions.saturating_add(1);
        self.last_error = Some(bounded_gpu_reason(reason.as_ref()));
    }

    /// Serves exactly one authenticated peer until clean disconnect or a
    /// protocol/execution failure.
    ///
    /// A clean disconnect at an envelope boundary returns
    /// [`SessionSummary`]. Authentication failure sends a rejection
    /// acknowledgement and returns [`HostError::Unauthenticated`]; any other
    /// error closes the stream when it is dropped.
    ///
    /// # Atomic transactions
    ///
    /// One call owns exactly one stream and one session state, and it reads a
    /// request and writes its reply before reading the next frame. A daemon
    /// that serves concurrent peers therefore gives every connection its own
    /// thread and its own host; two peers' frames can never interleave on one
    /// stream and a slow pipeline translation cannot delay a submission.
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
                KIND_SUBMISSION_V3 => self.serve_v3_submission(&mut stream, &payload)?,
                KIND_FENCE_PAGE => {
                    if let Err(error) = self.register_fence_page(&payload) {
                        reject_ack(&mut stream);
                        return Err(error);
                    }
                    write_frame(&mut stream, KIND_ACK, &[0])?;
                }
                KIND_SUBMISSION_ASYNC => self.serve_async_submission(&payload)?,
                KIND_TEXTURE_REG => {
                    let status = self.register_texture(&payload);
                    write_frame(&mut stream, KIND_ACK, &[status])?;
                }
                KIND_SAMPLER_REG => {
                    let status = self.register_sampler(&payload);
                    write_frame(&mut stream, KIND_ACK, &[status])?;
                }
                KIND_SURFACE_REG => {
                    let status = self.register_surface(&payload);
                    write_frame(&mut stream, KIND_ACK, &[status])?;
                }
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

    /// Registers the launcher-created fence page file for this session.
    ///
    /// The payload carries only `byte_len: u64 LE`; the path is the fixed
    /// [`FENCE_PAGE_FILE_NAME`] inside the canonical share directory, so a
    /// hostile peer cannot select another file. The mapping is revalidated as a
    /// regular, non-symlink, exactly-one-page file by
    /// [`FencePageMapping::open`]. The page is cleared and its epoch advanced
    /// before the acknowledgement, so the guest starts from a session-local
    /// fence state.
    ///
    /// The mapping is then handed to the executor (ADR-017): an asynchronous
    /// transport publishes the completion watermark from its GPU completion
    /// handler straight into the guest-visible page, so the host never reports a
    /// sequence complete before the GPU finished it. An executor that cannot
    /// publish asynchronously reports `false` and keeps the synchronous
    /// publish-on-`Ok` behaviour.
    fn register_fence_page(&mut self, payload: &[u8]) -> Result<(), HostError> {
        if payload.len() != FENCE_PAGE_REGISTER_PAYLOAD_LEN {
            return Err(HostError::Protocol(
                "fence page registration payload must be exactly 8 bytes",
            ));
        }
        if read_u64(payload, 0) != crate::fence_page::FENCE_PAGE_LEN as u64 {
            return Err(HostError::Protocol(
                "fence page registration length must be exactly 4096",
            ));
        }
        if self.fence_page.is_some() {
            return Err(HostError::Protocol(
                "fence page is already registered for this session",
            ));
        }

        let mapping = Arc::new(FencePageMapping::open(
            self.share_dir.join(FENCE_PAGE_FILE_NAME),
        )?);
        mapping.page().begin_session();
        self.fence_page_attached = self.executor.attach_fence_page(Arc::clone(&mapping))?;
        self.fence_page = Some(mapping);
        Ok(())
    }

    /// Registers a texture through the executor's ADR-013 registry.
    ///
    /// Returns the kind-128 status byte. Every validation failure is an
    /// explicit rejection rather than a session close, and no executor call is
    /// made unless the descriptor is fully validated, so a refused registration
    /// can never create GPU work or replace an existing handle.
    fn register_texture(&mut self, payload: &[u8]) -> u8 {
        let Some(request) = decode_texture_request(payload) else {
            return REGISTRATION_REJECTED;
        };
        if request.id == 0 || request.generation == 0 {
            return REGISTRATION_REJECTED;
        }
        // One handle space: a texture may not shadow a buffer allocation or an
        // earlier texture, and the snapshot key `(id, generation)` must be
        // unambiguous.
        if self.allocations.contains_key(&request.id)
            || self.textures.contains_key(&request.id)
        {
            return REGISTRATION_REJECTED;
        }
        if self.executor.register_texture(request.id, request.desc).is_err() {
            return REGISTRATION_REJECTED;
        }
        self.textures.insert(
            request.id,
            RegisteredTexture {
                generation: request.generation,
            },
        );
        REGISTRATION_ACCEPTED
    }

    /// Registers a reduced `MTLSamplerState` (kind 10).
    ///
    /// Returns the kind-128 status byte. A malformed payload or an out-of-range
    /// filter/address mode is an explicit rejection and leaves the session
    /// open. Re-registering a handle with the same content is accepted without
    /// touching the device; a conflicting content is rejected, so a handle can
    /// never silently change meaning inside one session.
    fn register_sampler(&mut self, payload: &[u8]) -> u8 {
        let Some(request) = decode_sampler_request(payload) else {
            return REGISTRATION_REJECTED;
        };
        if let Some(existing) = self.samplers.get(&request.id) {
            return if *existing == request.desc {
                REGISTRATION_ACCEPTED
            } else {
                REGISTRATION_REJECTED
            };
        }
        if self.executor.register_sampler(request.id, request.desc).is_err() {
            return REGISTRATION_REJECTED;
        }
        self.samplers.insert(request.id, request.desc);
        REGISTRATION_ACCEPTED
    }

    /// Registers an already-registered texture as a present surface.
    ///
    /// Returns the kind-128 status byte. The texture id must have been accepted
    /// by [`Self::register_texture`]; a surface id may only be used once.
    fn register_surface(&mut self, payload: &[u8]) -> u8 {
        if payload.len() != SURFACE_REG_PAYLOAD_LEN {
            return REGISTRATION_REJECTED;
        }
        let surface_id = read_u64(payload, 0);
        let texture_id = read_u64(payload, 8);
        if surface_id == 0 || texture_id == 0 {
            return REGISTRATION_REJECTED;
        }
        if self.surfaces.contains_key(&surface_id) || !self.textures.contains_key(&texture_id) {
            return REGISTRATION_REJECTED;
        }
        if self.executor.register_surface(surface_id, texture_id).is_err() {
            return REGISTRATION_REJECTED;
        }
        self.surfaces.insert(surface_id, texture_id);
        REGISTRATION_ACCEPTED
    }

    /// Publishes a version 3 submission's sequence before device execution.
    fn publish_submitted(&self, sequence: u32) {
        if let Some(fence) = &self.fence_page {
            fence.page().publish_submitted(u64::from(sequence));
        }
    }

    /// Publishes a version 3 submission's sequence as completed.
    ///
    /// Called only after the executor reported success. The page accumulates
    /// with `fetch_max`, so a stale or replayed sequence can never lower the
    /// value a guest already observed.
    fn publish_completed(&self, sequence: u32) {
        if let Some(fence) = &self.fence_page {
            fence.page().publish(u64::from(sequence));
        }
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

    /// Handles one kind 5 request: a versioned v3 envelope around one encoded
    /// [`SubmissionV3`].
    ///
    /// A framing failure is answered with [`ReplyStatus::ExecutionFailed`]
    /// instead of panicking. When the envelope carries a usable sequence the
    /// reply echoes it and the session stays open, so the guest can retry the
    /// same sequence. A frame that is too short or starts with the wrong magic
    /// has no trustworthy sequence; the host echoes the next expected sequence
    /// and then closes the session because the stream may be desynchronized.
    fn serve_v3_submission(
        &mut self,
        stream: &mut TcpStream,
        payload: &[u8],
    ) -> Result<(), HostError> {
        let submission = match decode_v3_request(payload) {
            Ok(submission) => submission,
            Err(V3RequestError::Rejected { sequence, reason }) => {
                self.note_refusal(ReplyStatus::ExecutionFailed, &reason);
                write_v3_reply(
                    stream,
                    Reply {
                        sequence,
                        status: ReplyStatus::ExecutionFailed,
                    },
                    Some(&reason),
                )?;
                self.submissions += 1;
                return Ok(());
            }
            Err(V3RequestError::Unsequenced) => {
                let reason = "version 3 request has no recoverable envelope sequence";
                self.note_refusal(ReplyStatus::ExecutionFailed, reason);
                write_v3_reply(
                    stream,
                    Reply {
                        sequence: self.next_sequence,
                        status: ReplyStatus::ExecutionFailed,
                    },
                    Some(reason),
                )?;
                self.submissions += 1;
                return Err(HostError::Protocol(reason));
            }
        };

        if submission.sequence != self.next_sequence {
            let reason = format!(
                "submission sequence mismatch: expected {}, got {}",
                self.next_sequence, submission.sequence
            );
            self.note_refusal(ReplyStatus::ExecutionFailed, &reason);
            write_v3_reply(
                stream,
                Reply {
                    sequence: submission.sequence,
                    status: ReplyStatus::ExecutionFailed,
                },
                Some(&reason),
            )?;
            self.submissions += 1;
            return Err(HostError::Protocol(
                "submission sequence is not the next expected value",
            ));
        }

        let reply = self.execute_or_refuse_v3(&submission);
        let reason = (reply.status != ReplyStatus::Ok).then(|| {
            self.last_error.clone().unwrap_or_else(|| {
                format!(
                    "the host refused the submission with {:?} but recorded no detail",
                    reply.status
                )
            })
        });
        write_v3_reply(stream, reply, reason.as_deref())?;
        self.next_sequence = submission
            .sequence
            .checked_add(1)
            .ok_or(HostError::Protocol("submission sequence exhausted"))?;
        self.submissions += 1;
        Ok(())
    }

    /// Handles one kind 7 request: a version 3 envelope with no reply frame.
    ///
    /// The frame is decoded, sequenced and executed exactly like kind 5, so the
    /// ledger and fence semantics are identical. There is no reply to carry a
    /// decode error, so any framing or sequencing failure closes the session;
    /// execution outcomes are reported only through the shared fence. Accepted
    /// work advances `completed_fence` when it actually finishes (see
    /// [`Self::execute_v3_with_fence`]); a refusal or executor failure leaves it
    /// behind `submitted_fence`, which the guest observes as pending.
    fn serve_async_submission(&mut self, payload: &[u8]) -> Result<(), HostError> {
        let submission = decode_v3_request(payload).map_err(|_| {
            self.note_refusal(
                ReplyStatus::ExecutionFailed,
                "asynchronous version 3 request has no decodable envelope",
            );
            HostError::Protocol("asynchronous version 3 request has no decodable envelope")
        })?;
        if submission.sequence != self.next_sequence {
            self.note_refusal(
                ReplyStatus::ExecutionFailed,
                format!(
                    "submission sequence mismatch: expected {}, got {}",
                    self.next_sequence, submission.sequence
                ),
            );
            return Err(HostError::Protocol(
                "submission sequence is not the next expected value",
            ));
        }

        // The reply carries only a status; there is no frame to send it in.
        let _reply = self.execute_or_refuse_v3(&submission);
        self.next_sequence = submission
            .sequence
            .checked_add(1)
            .ok_or(HostError::Protocol("submission sequence exhausted"))?;
        self.submissions += 1;
        Ok(())
    }

    /// Executes one version 3 submission, enforcing exactly-once for list-bound
    /// batches with the same [`ListLedger`] used by version 2.
    ///
    /// Unknown allocation or generation pairs are rejected before the ledger
    /// claim is taken, so a validation refusal leaves the identity retryable
    /// and performs no GPU work. A replay of an executed list returns
    /// [`ReplyStatus::AlreadyExecuted`] with zero executions. An executor
    /// failure (including [`MetalError::NotSupported`]) is reported as
    /// [`ReplyStatus::ExecutionFailed`]; an identity whose execution outcome is
    /// uncertain stays in flight, mirroring version 2, so it is never executed
    /// twice.
    ///
    /// When a fence page is registered, a submission that actually reaches the
    /// executor publishes its sequence as `submitted_fence`; only work that is
    /// actually finished advances `completed_fence` (from the GPU completion
    /// handler for an asynchronous transport, from this call for a synchronous
    /// executor). Refused and replayed submissions publish nothing.
    fn execute_or_refuse_v3(&mut self, submission: &SubmissionV3) -> Reply {
        let response = |status| Reply {
            sequence: submission.sequence,
            status,
        };

        if !self.v3_references_valid(submission) {
            self.note_refusal(
                ReplyStatus::UnknownHandle,
                "version 3 submission references an unregistered allocation, texture or sampler",
            );
            return response(ReplyStatus::UnknownHandle);
        }

        let allocations = self.allocation_snapshot();

        let Some(identity) = submission.list else {
            return self.execute_v3_with_fence(submission, &allocations);
        };

        match self.ledger.begin(identity, submission.sequence) {
            Ok(()) => {
                let reply = self.execute_v3_with_fence(submission, &allocations);
                if reply.status == ReplyStatus::Ok {
                    self.ledger.complete(identity, submission.sequence);
                }
                reply
            }
            Err(RingError::AlreadyExecuted) => {
                self.note_refusal(
                    ReplyStatus::AlreadyExecuted,
                    format!(
                        "list {} generation {} was already executed",
                        identity.list_id, identity.list_generation
                    ),
                );
                response(ReplyStatus::AlreadyExecuted)
            }
            Err(_) => {
                self.note_refusal(
                    ReplyStatus::ExecutionFailed,
                    "the list ledger refused a claim (capacity)",
                );
                response(ReplyStatus::ExecutionFailed)
            }
        }
    }

    /// Runs one executor submission and publishes its fence transitions.
    ///
    /// `submitted_fence` is published before the executor runs. `completed_fence`
    /// is only published here for a **synchronous** executor, which has already
    /// finished the GPU work when it returns: an asynchronous transport (ADR-017)
    /// publishes the real completion from its GPU completion handler, so an
    /// early publish here would tell the guest that work the GPU has not started
    /// is done.
    fn execute_v3_with_fence(
        &mut self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Reply {
        self.publish_submitted(submission.sequence);
        let reply = self.execute_v3(submission, allocations);
        if reply.status == ReplyStatus::Ok && !self.fence_page_attached {
            self.publish_completed(submission.sequence);
        }
        reply
    }

    fn allocation_snapshot(&self) -> HashMap<(u64, u32), VmBufferId> {
        let mut snapshot: HashMap<(u64, u32), VmBufferId> = self
            .allocations
            .iter()
            .map(|(&id, allocation)| ((id, BASE_GENERATION), allocation.buffer_id))
            .collect();
        // Textures join the same snapshot with their own generation; the v3
        // attachment resolver then finds the registry texture by handle.
        for (&id, texture) in &self.textures {
            snapshot.insert((id, texture.generation), VmBufferId(id));
        }
        snapshot
    }

    fn execute_v3(
        &mut self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Reply {
        match self.executor.submit_v3(submission, allocations) {
            Ok(reply) => {
                if reply.status != ReplyStatus::Ok {
                    self.note_refusal(
                        reply.status,
                        format!("the executor answered {:?}", reply.status),
                    );
                }
                reply
            }
            Err(error) => {
                // The Metal diagnostic (for example `not supported: ...`, a
                // command-buffer fault, or a pipeline error) is bounded and
                // kept for telemetry; the guest status stays ExecutionFailed.
                let reason = format!("Metal execution failed: {error}");
                tracing::error!(sequence = submission.sequence, reason = %reason, "Metal submission failed");
                self.note_refusal(ReplyStatus::ExecutionFailed, reason);
                Reply {
                    sequence: submission.sequence,
                    status: ReplyStatus::ExecutionFailed,
                }
            }
        }
    }

    fn v3_references_valid(&self, submission: &SubmissionV3) -> bool {
        submission
            .commands
            .iter()
            .all(|command| self.v3_command_references_valid(command))
    }

    fn v3_command_references_valid(&self, command: &CommandV3) -> bool {
        match command {
            CommandV3::Dispatch1d(dispatch) => self.bindings_valid(&dispatch.bindings),
            CommandV3::Dispatch3d(dispatch) => self.bindings_valid(&dispatch.bindings),
            CommandV3::DispatchMesh(dispatch) => self.bindings_valid(&dispatch.bindings),
            CommandV3::SetDescriptorTable(table) => self.table_bindings_valid(&table.bindings),
            CommandV3::BeginRenderPass(pass) => {
                pass.colors
                    .iter()
                    .all(|color| self.reference_valid(color.allocation_id, color.generation))
                    && pass.depth_stencil.as_ref().is_none_or(|depth| {
                        self.reference_valid(depth.allocation_id, depth.generation)
                    })
            }
            CommandV3::SetVertexBuffers(set) => set.buffers.iter().all(|buffer| {
                self.range_valid(
                    buffer.allocation_id,
                    buffer.generation,
                    buffer.offset,
                    buffer.length,
                )
            }),
            CommandV3::SetIndexBuffer(binding) => self.range_valid(
                binding.allocation_id,
                binding.generation,
                binding.offset,
                binding.length,
            ),
            CommandV3::SetRootBufferView(view) => self.range_valid(
                view.allocation_id,
                view.generation,
                view.offset,
                view.length,
            ),
            CommandV3::ResourceBarrier(barrier) => barrier.entries.iter().all(|entry| {
                self.range_valid(
                    entry.allocation_id,
                    entry.generation,
                    entry.offset,
                    entry.length,
                )
            }),
            CommandV3::CopyBuffer(copy) => {
                self.range_valid(
                    copy.src_allocation_id,
                    copy.src_generation,
                    copy.src_offset,
                    copy.length,
                ) && self.range_valid(
                    copy.dst_allocation_id,
                    copy.dst_generation,
                    copy.dst_offset,
                    copy.length,
                )
            }
            CommandV3::CopyTexture(copy) => {
                self.reference_valid(copy.src_allocation_id, copy.src_generation)
                    && self.reference_valid(copy.dst_allocation_id, copy.dst_generation)
            }
            // Draw, state, constant, topology, pipeline and present commands
            // name no allocation from the registered window map. The device
            // validates pipelines, surfaces and pass ordering before it creates
            // a command buffer.
            _ => true,
        }
    }

    fn bindings_valid(&self, bindings: &[BufferBinding]) -> bool {
        bindings.iter().all(|binding| {
            self.range_valid(
                binding.allocation_id,
                binding.generation,
                binding.offset,
                binding.length,
            )
        })
    }

    /// A v3 table entry is either a registered allocation range or a resource
    /// handle the host registered out of band.
    ///
    /// Texture handles are the kind 8 registrations and sampler handles are the
    /// kind 10 registrations, so an unknown handle is `UnknownHandle` before any
    /// encoder exists.
    fn table_bindings_valid(&self, entries: &[DescriptorTableEntryV3]) -> bool {
        entries.iter().all(|entry| match entry {
            DescriptorTableEntryV3::Buffer(binding) => self.range_valid(
                binding.allocation_id,
                binding.generation,
                binding.offset,
                binding.length,
            ),
            DescriptorTableEntryV3::Texture(handle) => self.textures.contains_key(handle),
            DescriptorTableEntryV3::Sampler(handle) => self.samplers.contains_key(handle),
        })
    }

    fn reference_valid(&self, allocation_id: u64, generation: u32) -> bool {
        if self.allocations.contains_key(&allocation_id) {
            return generation == BASE_GENERATION;
        }
        self.textures
            .get(&allocation_id)
            .is_some_and(|texture| generation == texture.generation)
    }

    fn range_valid(&self, allocation_id: u64, generation: u32, offset: u64, length: u64) -> bool {
        if generation != BASE_GENERATION {
            return false;
        }
        let Some(allocation) = self.allocations.get(&allocation_id) else {
            return false;
        };
        offset
            .checked_add(length)
            .is_some_and(|end| end <= allocation.byte_len)
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
    fn execute_or_refuse(&mut self, submission: &Submission, list: Option<ListIdentity>) -> Reply {
        let response = |status| Reply {
            sequence: submission.sequence,
            status,
        };

        if !self.submission_bindings_valid(submission) {
            self.note_refusal(
                ReplyStatus::UnknownHandle,
                "submission references an unregistered allocation or range",
            );
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
            Err(RingError::AlreadyExecuted) => {
                self.note_refusal(
                    ReplyStatus::AlreadyExecuted,
                    format!(
                        "list {} generation {} was already executed",
                        identity.list_id, identity.list_generation
                    ),
                );
                response(ReplyStatus::AlreadyExecuted)
            }
            // The ledger refused a claim for another reason (for example its
            // in-flight cap). No work ran and no claim is held.
            Err(_) => {
                self.note_refusal(
                    ReplyStatus::ExecutionFailed,
                    "the list ledger refused a claim (capacity)",
                );
                response(ReplyStatus::ExecutionFailed)
            }
        }
    }

    fn submission_bindings_valid(&self, submission: &Submission) -> bool {
        submission
            .commands
            .iter()
            .all(|command| self.bindings_within_logical_lengths(command))
    }

    fn execute_submission(&mut self, submission: &Submission) -> Reply {
        let allocations: HashMap<(u64, u32), VmBufferId> = self
            .allocations
            .iter()
            .map(|(&id, allocation)| ((id, BASE_GENERATION), allocation.buffer_id))
            .collect();

        match self.executor.submit_batch(submission, &allocations) {
            Ok(reply) => {
                if reply.status != ReplyStatus::Ok {
                    self.note_refusal(
                        reply.status,
                        format!("the executor answered {:?}", reply.status),
                    );
                }
                reply
            }
            Err(error) => {
                let reason = format!("Metal execution failed: {error}");
                // The wire carries a status only; this tracing error and
                // telemetry `last_error` carry the bounded Metal text.
                tracing::error!(sequence = submission.sequence, reason = %reason, "Metal batch failed");
                self.note_refusal(ReplyStatus::ExecutionFailed, reason);
                Reply {
                    sequence: submission.sequence,
                    status: ReplyStatus::ExecutionFailed,
                }
            }
        }
    }

    fn bindings_within_logical_lengths(&self, command: &Command) -> bool {
        // Every command that names buffers is checked here so the executor
        // never resolves a range past its allocation's logical length. State
        // and barrier commands carry no draw bindings; the device validates
        // their own fields (handles, generations, entry counts) before it
        // creates a command buffer.
        let bindings = match command {
            Command::Dispatch1d(dispatch) => &dispatch.bindings,
            Command::Dispatch3d(dispatch) => &dispatch.bindings,
            Command::DrawIndexed(draw) => &draw.bindings,
            Command::Draw(draw) => &draw.bindings,
            Command::DispatchMesh(dispatch) => &dispatch.bindings,
            Command::DescriptorTable(table) => &table.bindings,
            Command::SetPipeline(_) | Command::RootConstants(_) | Command::ResourceBarrier(_) => {
                return true;
            }
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
        match decode_pipeline_request(payload) {
            Some(PipelineRequest::V1 { ticket, entry }) => {
                self.register_compute_pipeline_v1(ticket, entry)
            }
            Some(PipelineRequest::V2(request)) => self.register_pipeline_v2(&request),
            Some(PipelineRequest::V3(request)) => self.register_pipeline_v3(&request),
            None => {
                if payload.len() >= PIPELINE_FRAME_MAGIC_V3.len()
                    && payload[..PIPELINE_FRAME_MAGIC_V3.len()] == PIPELINE_FRAME_MAGIC_V3
                {
                    self.reject_host_translation("version 3 pipeline frame failed validation")
                } else {
                    vec![PIPELINE_REPLY_REJECTED]
                }
            }
        }
    }

    /// The original kind 4 form: one ticket, one entry, a compute pipeline.
    ///
    /// The bytes on the wire are unchanged, so an old guest keeps working.
    fn register_compute_pipeline_v1(&mut self, ticket: u64, entry: &str) -> Vec<u8> {
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
        pipeline_reply_ok(pipeline.0)
    }

    /// Version 2: one frame carries the pipeline kind, one library per stage,
    /// the encoded [`PipelineState`], the encoded [`PipelineAbi`] and the raw
    /// root-signature bytes.
    ///
    /// Every field is validated before the executor is called, so a malformed
    /// frame creates no GPU work and consumes no ticket. All three families use
    /// the registry's content-hashed per-stage entry points and the host's own
    /// handle space: each stage's function is resolved in the library its own
    /// ticket names, so a `--stdio` pool that emits one metallib per stage is
    /// registrable. A rejection is one status byte and leaves a diagnostic that
    /// names the failing family (and, from the executor, the failing stage).
    fn register_pipeline_v2(&mut self, request: &PipelineRequestV2) -> Vec<u8> {
        self.last_pipeline_rejection = None;
        let abi = match validate_pipeline_v2(request) {
            Ok(abi) => abi,
            Err(()) => {
                self.last_pipeline_rejection =
                    Some("version 2 pipeline frame failed validation".to_string());
                return vec![PIPELINE_REPLY_REJECTED];
            }
        };
        if request
            .stages
            .iter()
            .any(|stage| stage.ticket <= self.last_pipeline_ticket)
        {
            self.last_pipeline_rejection = Some(format!(
                "a stage ticket is not greater than the last accepted ticket {}",
                self.last_pipeline_ticket
            ));
            return vec![PIPELINE_REPLY_REJECTED];
        }

        // Read every per-stage library before creating anything: a missing
        // file rejects the whole frame, never a partially registered pipeline.
        let mut metallibs = Vec::with_capacity(request.stages.len());
        for stage in &request.stages {
            match read_pipeline_file(&self.share_dir, stage.ticket) {
                Ok(metallib) => metallibs.push(metallib),
                Err(_) => {
                    self.last_pipeline_rejection = Some(format!(
                        "the {} stage file is missing or unreadable",
                        pipeline_stage_label(stage.stage)
                    ));
                    return vec![PIPELINE_REPLY_REJECTED];
                }
            }
        }

        let stages: Vec<PipelineStageLibrary<'_>> = request
            .stages
            .iter()
            .zip(&metallibs)
            .map(|(stage, metallib)| PipelineStageLibrary {
                stage: stage.stage,
                metallib,
                entry: &stage.entry,
            })
            .collect();
        let handle = self.allocate_content_pipeline_handle();
        let outcome = match request.kind {
            PIPELINE_KIND_COMPUTE => self.executor.register_compute_pipeline_stages(
                handle,
                &stages,
                &request.state,
                abi,
            ),
            PIPELINE_KIND_GRAPHICS => self.executor.register_graphics_pipeline_stages(
                handle,
                &stages,
                &request.state,
                abi,
            ),
            PIPELINE_KIND_MESH => self.executor.register_mesh_pipeline_stages(
                handle,
                &stages,
                &request.state,
                abi,
            ),
            _ => {
                self.release_content_pipeline_handle(handle);
                self.last_pipeline_rejection = Some("unknown pipeline kind".to_string());
                return vec![PIPELINE_REPLY_REJECTED];
            }
        };
        let id = match outcome {
            Ok(()) => handle,
            Err(error) => {
                self.release_content_pipeline_handle(handle);
                self.last_pipeline_rejection = Some(format!(
                    "{} pipeline registration failed: {error}",
                    pipeline_kind_label(request.kind)
                ));
                return vec![PIPELINE_REPLY_REJECTED];
            }
        };

        self.last_pipeline_ticket = request
            .stages
            .iter()
            .map(|stage| stage.ticket)
            .max()
            .unwrap_or(self.last_pipeline_ticket);
        pipeline_reply_ok(id)
    }

    /// Version 3: the guest uploaded real DXIL stages to the share directory and
    /// the host translates them with its own Rosetta pool (ADR-038).
    ///
    /// Every field and file is validated before any GPU work. Translation runs
    /// concurrently across the stage set through the owned [`CompilerPool`]; a
    /// failed stage is a bounded refusal naming only its stage kind and the
    /// converter failure category. The per-stage ABI is derived from the
    /// `VMRES2` reflection, never from a guest claim, and the resulting in-memory
    /// metallibs reuse the existing per-stage registry path.
    fn register_pipeline_v3(&mut self, request: &PipelineRequestV3) -> Vec<u8> {
        self.last_pipeline_rejection = None;
        if let Err(reason) = validate_pipeline_v3(request) {
            return self.reject_host_translation(reason);
        }
        if let Some(name) = request.root_signature_file.as_deref()
            && read_stage_file(&self.share_dir, name).is_err()
        {
            return self
                .reject_host_translation("the root-signature file is missing or unreadable");
        }
        let mut dxil = Vec::with_capacity(request.stages.len());
        for stage in &request.stages {
            match read_stage_file(&self.share_dir, &stage.file) {
                Ok(bytes) => dxil.push(bytes),
                Err(_) => {
                    return self.reject_host_translation(format!(
                        "the {} stage file is missing or unreadable",
                        pipeline_stage_label(stage.stage)
                    ));
                }
            }
        }

        let pool = self
            .compiler_pool
            .get_or_insert_with(shared_compiler_pool)
            .clone();
        let before = pool.stats();
        let results: Vec<Result<crate::compiler_pool::CompiledShader, PoolError>> =
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(request.stages.len());
                for (stage, bytes) in request.stages.iter().zip(&dxil) {
                    let pool = pool.as_ref();
                    handles.push(scope.spawn(move || {
                        let job = CompileJob {
                            id: stage_job_id(stage.stage, &stage.entry, bytes),
                            stage: converter_stage_code(stage.stage),
                            entry: &stage.entry,
                            dxil: bytes,
                        };
                        pool.compile_with_reflection(&job)
                    }));
                }
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap_or(Err(PoolError::Closed)))
                    .collect()
            });
        let cache_hits = pool.stats().cache_hits.saturating_sub(before.cache_hits);

        if let Some((stage, error)) = request
            .stages
            .iter()
            .zip(&results)
            .find_map(|(stage, result)| result.as_ref().err().map(|error| (stage, error)))
        {
            // The worker's own bounded reason, when it is short enough for
            // telemetry, is the only way a boot can say WHY a converter died.
            let detail = pool
                .worker_stderr_for_telemetry()
                .map(|stderr| format!(": {stderr}"))
                .unwrap_or_default();
            return self.reject_host_translation(format!(
                "{} stage translation failed: {}{detail}",
                pipeline_stage_label(stage.stage),
                pool_error_category(error)
            ));
        }
        let reflections: Vec<&[u8]> = results
            .iter()
            .map(|result| {
                result
                    .as_ref()
                    .expect("every stage translation succeeded")
                    .reflection
                    .as_slice()
            })
            .collect();
        let abi = match abi_from_converter_reflections(request, &reflections) {
            Ok(abi) => abi,
            Err(reason) => return self.reject_host_translation(reason),
        };

        // A graphics vertex function with `[[attribute(n)]]` inputs needs a
        // Metal vertex descriptor that the guest's state blob does not carry.
        // The vertex stage's own DXIL container is the only signature source;
        // a layout is appended after the state so the BatchExecutor trait and
        // every older guest frame keep their exact back-compatible shape.
        let mut state = request.state.clone();
        if request.kind == PIPELINE_KIND_GRAPHICS {
            let Some(vertex_index) = request
                .stages
                .iter()
                .position(|stage| stage.stage == PIPELINE_STAGE_VERTEX)
            else {
                return self.reject_host_translation("a graphics pipeline needs a vertex stage");
            };
            match VertexLayout::from_dxil_container(&dxil[vertex_index]) {
                Ok(layout) if !layout.is_empty() => layout.append_to(&mut state),
                Ok(_) => {}
                Err(error) => {
                    return self.reject_host_translation(format!(
                        "the vertex input signature cannot become a Metal vertex layout: {error}"
                    ));
                }
            }
        }

        let stages: Vec<PipelineStageLibrary<'_>> = request
            .stages
            .iter()
            .zip(&results)
            .map(|(stage, result)| PipelineStageLibrary {
                stage: stage.stage,
                metallib: result
                    .as_ref()
                    .expect("every stage translation succeeded")
                    .metallib
                    .as_slice(),
                entry: &stage.entry,
            })
            .collect();
        let handle = self.allocate_content_pipeline_handle();
        let outcome = match request.kind {
            PIPELINE_KIND_COMPUTE => self.executor.register_compute_pipeline_stages(
                handle,
                &stages,
                &state,
                abi,
            ),
            PIPELINE_KIND_GRAPHICS => self.executor.register_graphics_pipeline_stages(
                handle,
                &stages,
                &state,
                abi,
            ),
            PIPELINE_KIND_MESH => self.executor.register_mesh_pipeline_stages(
                handle,
                &stages,
                &state,
                abi,
            ),
            _ => {
                self.release_content_pipeline_handle(handle);
                return self.reject_host_translation("unknown pipeline kind");
            }
        };
        if let Err(error) = outcome {
            self.release_content_pipeline_handle(handle);
            return self.reject_host_translation(format!(
                "{} pipeline registration failed: {error}",
                pipeline_kind_label(request.kind)
            ));
        }

        self.host_pipeline_translations_ok = self.host_pipeline_translations_ok.saturating_add(1);
        self.host_pipeline_cache_hits = self.host_pipeline_cache_hits.saturating_add(cache_hits);
        for stage in &request.stages {
            if let Some(index) = stage_kind_index(stage.stage) {
                self.host_translated_stages[index] =
                    self.host_translated_stages[index].saturating_add(1);
            }
        }
        pipeline_reply_ok(handle)
    }

    /// Records and answers one version 3 refusal with a bounded category.
    fn reject_host_translation(&mut self, reason: impl AsRef<str>) -> Vec<u8> {
        let reason = bounded_gpu_reason(reason.as_ref());
        self.host_pipeline_translations_refused =
            self.host_pipeline_translations_refused.saturating_add(1);
        self.host_translation_reason = Some(reason.clone());
        self.last_pipeline_rejection = Some(reason.clone());
        pipeline_reply_rejected(&reason)
    }
}

fn pipeline_reply_ok(id: u64) -> Vec<u8> {
    let mut reply = Vec::with_capacity(PIPELINE_REPLY_OK_LEN);
    reply.push(PIPELINE_REPLY_OK);
    reply.extend_from_slice(&id.to_le_bytes());
    reply
}

/// One version 3 rejection: status byte, bounded reason length and reason.
///
/// The older v1/v2 frames keep their one-byte rejection; only the new frame
/// carries a reason, so no existing guest decodes a different shape.
fn pipeline_reply_rejected(reason: &str) -> Vec<u8> {
    let reason = bounded_gpu_reason(reason);
    let bytes = reason.as_bytes();
    let len = bytes.len().min(MAX_PIPELINE_REJECTION_REASON);
    let mut reply = Vec::with_capacity(1 + 2 + len);
    reply.push(PIPELINE_REPLY_REJECTED);
    reply.extend_from_slice(&u16::try_from(len).expect("reason is bounded").to_le_bytes());
    reply.extend_from_slice(&bytes[..len]);
    reply
}

/// The converter's own stage code for a wire stage code.
fn converter_stage_code(stage: u8) -> u32 {
    match stage {
        PIPELINE_STAGE_VERTEX => 1,
        PIPELINE_STAGE_PIXEL => 2,
        PIPELINE_STAGE_MESH => 5,
        PIPELINE_STAGE_OBJECT => 6,
        _ => 8,
    }
}

/// The telemetry bucket of a wire stage code.
fn stage_kind_index(stage: u8) -> Option<usize> {
    Some(match stage {
        PIPELINE_STAGE_VERTEX => 0,
        PIPELINE_STAGE_PIXEL => 1,
        PIPELINE_STAGE_MESH => 2,
        PIPELINE_STAGE_OBJECT => 3,
        PIPELINE_STAGE_COMPUTE => 4,
        _ => return None,
    })
}

/// A stable, non-proprietary category for a pool failure.
///
/// It never contains a worker path, a file name, reflection text or shader
/// bytes: the boot telemetry only needs to say which converter stage failed.
fn pool_error_category(error: &PoolError) -> &'static str {
    match error {
        PoolError::NoWorkerProgram => "no converter worker",
        PoolError::InvalidJob(_) => "invalid conversion job",
        PoolError::Spawn { .. } => "converter worker unavailable",
        PoolError::WorkerWrite { .. } | PoolError::WorkerRead { .. } => {
            "converter worker I/O failed"
        }
        PoolError::WorkerExited { .. } => "converter worker exited",
        PoolError::SequenceMismatch { .. } | PoolError::IdMismatch { .. } => {
            "converter protocol mismatch"
        }
        PoolError::WorkerFailed { .. } => "converter rejected the shader",
        PoolError::Protocol { .. } => "converter protocol error",
        PoolError::Timeout { .. } => "converter timed out",
        PoolError::Closed => "converter pool closed",
    }
}

/// A stable job id for one stage, matching the guest's own derivation.
fn stage_job_id(stage: u8, entry: &str, dxil: &[u8]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&converter_stage_code(stage).to_le_bytes());
    hasher.update(entry.as_bytes());
    hasher.update(dxil);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("eight bytes"))
}

/// Builds the per-stage [`PipelineAbi`] from the converter reflection JSON.
///
/// This is the host-side equivalent of the guest's `VAB1` decoder: the host
/// just received the reflection itself, so it derives the layout directly
/// instead of trusting a guest blob. An empty or unparsable reflection, or one
/// whose entry point disagrees with the frame, is an explicit bounded refusal.
fn abi_from_converter_reflections(
    request: &PipelineRequestV3,
    reflections: &[&[u8]],
) -> Result<PipelineAbi, String> {
    let mut abi = PipelineAbi::default();
    for (stage, reflection) in request.stages.iter().zip(reflections) {
        let label = pipeline_stage_label(stage.stage);
        let text = std::str::from_utf8(reflection)
            .map_err(|_| format!("the {label} stage reflection is not UTF-8"))?;
        if text.trim().is_empty() {
            return Err(format!("the {label} stage reflection is empty"));
        }
        let layout = RootSignatureLayout::from_converter_reflection_json_detailed(text)
            .map_err(|error| {
                format!("the {label} stage reflection is not a usable root layout: {error}")
            })?;
        if !layout.entry_point().is_empty() && layout.entry_point() != stage.entry {
            return Err(format!(
                "the {label} stage reflection names a different entry point"
            ));
        }
        match stage.stage {
            PIPELINE_STAGE_VERTEX if request.kind == PIPELINE_KIND_GRAPHICS => {
                abi.vertex = Some(layout)
            }
            PIPELINE_STAGE_PIXEL if request.kind != PIPELINE_KIND_COMPUTE => {
                abi.fragment = Some(layout)
            }
            PIPELINE_STAGE_MESH if request.kind == PIPELINE_KIND_MESH => abi.mesh = Some(layout),
            PIPELINE_STAGE_OBJECT if request.kind == PIPELINE_KIND_MESH => abi.object = Some(layout),
            PIPELINE_STAGE_COMPUTE if request.kind == PIPELINE_KIND_COMPUTE => {
                abi.compute = Some(layout)
            }
            _ => {
                return Err(format!(
                    "the {label} stage cannot appear in a {} pipeline",
                    pipeline_kind_label(request.kind)
                ));
            }
        }
    }
    Ok(abi)
}

/// Human-readable name of a version 2 pipeline kind.
fn pipeline_kind_label(kind: u8) -> &'static str {
    match kind {
        PIPELINE_KIND_COMPUTE => "compute",
        PIPELINE_KIND_GRAPHICS => "graphics",
        PIPELINE_KIND_MESH => "mesh",
        _ => "unknown",
    }
}

/// Human-readable name of a version 2 stage code.
fn pipeline_stage_label(stage: u8) -> &'static str {
    match stage {
        PIPELINE_STAGE_VERTEX => "vertex",
        PIPELINE_STAGE_PIXEL => "pixel",
        PIPELINE_STAGE_MESH => "mesh",
        PIPELINE_STAGE_OBJECT => "object",
        PIPELINE_STAGE_COMPUTE => "compute",
        _ => "unknown",
    }
}

/// Maps one wire stage code to the registry role of the function it names.
fn registry_stage_role(stage: u8) -> Result<FunctionRole, MetalError> {
    Ok(match stage {
        PIPELINE_STAGE_VERTEX => FunctionRole::Vertex,
        PIPELINE_STAGE_PIXEL => FunctionRole::Fragment,
        PIPELINE_STAGE_MESH => FunctionRole::Mesh,
        PIPELINE_STAGE_OBJECT => FunctionRole::Object,
        PIPELINE_STAGE_COMPUTE => FunctionRole::Kernel,
        _ => {
            return Err(MetalError::InvalidArg(
                "pipeline stage code is not one an executor registers",
            ));
        }
    })
}

/// Rewrites the frame's per-stage libraries as registry stage libraries.
fn registry_stages<'a>(
    stages: &'a [PipelineStageLibrary<'a>],
) -> Result<Vec<StageLibrary<'a>>, MetalError> {
    stages
        .iter()
        .map(|stage| {
            Ok(StageLibrary {
                role: registry_stage_role(stage.stage)?,
                metallib: stage.metallib,
                entry: stage.entry,
            })
        })
        .collect()
}

impl<E: BatchExecutor> Drop for IpcHost<E> {
    fn drop(&mut self) {
        // The guest-visible page is unmapped when this struct drops its mapping.
        // Detach it first: an asynchronous executor drains its GPU pipeline there,
        // so no completion handler can publish into released memory.
        if self.fence_page.take().is_some() {
            self.executor.detach_fence_page();
        }
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

/// Reason a version 3 request envelope could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
enum V3RequestError {
    /// The envelope carried no usable sequence, so no reply can be correlated.
    Unsequenced,
    /// The envelope sequence is known, but the request was refused.
    Rejected {
        /// Sequence the reply must echo.
        sequence: u32,
        /// Bounded host reason carried by the error reply.
        reason: String,
    },
}

/// Decodes one version 3 request envelope.
///
/// Layout: `magic[4]`, `version u16`, `sequence u32`, reserved `u16`,
/// `submission_len u32`, then exactly `submission_len` bytes decoded by
/// [`decode_submission_v3`]. The envelope sequence must match the encoded
/// submission. Every length and version check happens before the codec sees the
/// frame, and the codec itself rejects truncation, trailing bytes and oversized
/// payloads, so hostile input never panics.
fn decode_v3_request(payload: &[u8]) -> Result<SubmissionV3, V3RequestError> {
    if payload.len() < V3_REQUEST_HEADER_LEN {
        return Err(V3RequestError::Unsequenced);
    }
    if payload[..V3_REQUEST_MAGIC.len()] != V3_REQUEST_MAGIC {
        return Err(V3RequestError::Unsequenced);
    }

    let version = read_u16(payload, 4);
    let sequence = read_u32(payload, 6);
    if sequence == 0 {
        // A zero sequence is not a reply-correlatable request.
        return Err(V3RequestError::Unsequenced);
    }
    if version != PROTOCOL_VERSION_V3 {
        return Err(V3RequestError::Rejected {
            sequence,
            reason: format!(
                "version 3 request envelope has version {version}, expected {PROTOCOL_VERSION_V3}"
            ),
        });
    }
    if read_u16(payload, 10) != 0 {
        return Err(V3RequestError::Rejected {
            sequence,
            reason: "version 3 request reserved word is nonzero".to_string(),
        });
    }

    let declared = usize::try_from(read_u32(payload, 12)).map_err(|_| V3RequestError::Rejected {
        sequence,
        reason: "version 3 request length does not fit the host address space".to_string(),
    })?;
    if declared > MAX_SUBMISSION_V3_LEN {
        return Err(V3RequestError::Rejected {
            sequence,
            reason: format!(
                "version 3 request declares {declared} bytes, over the {MAX_SUBMISSION_V3_LEN}-byte limit"
            ),
        });
    }
    let frame = &payload[V3_REQUEST_HEADER_LEN..];
    if frame.len() != declared {
        return Err(V3RequestError::Rejected {
            sequence,
            reason: format!(
                "version 3 request declares {declared} bytes but carries {}",
                frame.len()
            ),
        });
    }

    let submission = decode_submission_v3(frame).map_err(|error| V3RequestError::Rejected {
        sequence,
        reason: format!("version 3 submission frame was refused: {error}"),
    })?;
    if submission.sequence != sequence {
        return Err(V3RequestError::Rejected {
            sequence,
            reason: format!(
                "version 3 request sequence {sequence} does not match submission sequence {}",
                submission.sequence
            ),
        });
    }
    Ok(submission)
}

fn write_v3_reply(
    stream: &mut TcpStream,
    reply: Reply,
    reason: Option<&str>,
) -> Result<(), HostError> {
    let encoded = wire::encode_reply_with_reason(&reply, reason)?;
    write_frame(stream, KIND_SUBMISSION_V3, &encoded)
}

fn read_u16(payload: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([payload[offset], payload[offset + 1]])
}

fn read_u32(payload: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&payload[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn read_u64(payload: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&payload[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

/// One kind 4 request decoded from the wire.
enum PipelineRequest<'a> {
    /// The original `ticket + entry` compute request.
    V1 { ticket: u64, entry: &'a str },
    /// The versioned multi-stage request.
    V2(PipelineRequestV2),
    /// The versioned share-dir DXIL request the host translates itself.
    V3(PipelineRequestV3),
}

/// One `(stage, ticket, entry)` of a version 2 request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PipelineStageEntry {
    stage: u8,
    ticket: u64,
    entry: String,
}

/// A fully decoded version 2 kind 4 request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PipelineRequestV2 {
    kind: u8,
    stages: Vec<PipelineStageEntry>,
    state: Vec<u8>,
    abi: Vec<u8>,
    root_signature: Vec<u8>,
}

/// One `(stage, share-relative DXIL file, entry)` of a version 3 request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PipelineStageFile {
    stage: u8,
    file: String,
    entry: String,
}

/// A fully decoded version 3 kind 4 request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PipelineRequestV3 {
    kind: u8,
    stages: Vec<PipelineStageFile>,
    state: Vec<u8>,
    root_signature_file: Option<String>,
}

/// Decodes a kind 4 payload into a ticket and a validated entry name.
fn decode_pipeline_request(payload: &[u8]) -> Option<PipelineRequest<'_>> {
    if payload.len() >= PIPELINE_FRAME_MAGIC_V3.len()
        && payload[..PIPELINE_FRAME_MAGIC_V3.len()] == PIPELINE_FRAME_MAGIC_V3
    {
        return decode_pipeline_request_v3(payload).map(PipelineRequest::V3);
    }
    if payload.len() >= PIPELINE_FRAME_MAGIC.len()
        && payload[..PIPELINE_FRAME_MAGIC.len()] == PIPELINE_FRAME_MAGIC
    {
        return decode_pipeline_request_v2(payload).map(PipelineRequest::V2);
    }
    decode_pipeline_request_v1(payload).map(|(ticket, entry)| PipelineRequest::V1 { ticket, entry })
}

/// Decodes the original kind 4 payload: `ticket: u64, entry_len: u16, entry`.
fn decode_pipeline_request_v1(payload: &[u8]) -> Option<(u64, &str)> {
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

/// Decodes the version 2 frame.
///
/// | offset | size | field |
/// | ---: | ---: | --- |
/// | 0 | 4 | magic `VP2\0` |
/// | 4 | 1 | frame version, `1` |
/// | 5 | 1 | pipeline kind: 0 compute, 1 graphics, 2 mesh |
/// | 6 | 1 | stage count, 1..=4 |
/// | 7 | 1 | reserved, zero |
/// | 8 | 4 | encoded `PipelineState` length |
/// | 12 | 4 | encoded pipeline ABI length |
/// | 16 | 4 | raw root-signature length |
/// | 20 | 4 | reserved, zero |
/// | 24 | N | stage entries: `stage u8, ticket u64, entry_len u16, entry` |
/// | ... | S | encoded `PipelineState` |
/// | ... | A | encoded pipeline ABI (`VAB1` container) |
/// | ... | R | raw root-signature bytes |
fn decode_pipeline_request_v2(payload: &[u8]) -> Option<PipelineRequestV2> {
    if payload.len() < PIPELINE_V2_HEADER_LEN {
        return None;
    }
    if payload[..4] != PIPELINE_FRAME_MAGIC {
        return None;
    }
    let version = payload[4];
    let kind = payload[5];
    let stage_count = payload[6];
    if version != PIPELINE_FRAME_VERSION_V2
        || kind > PIPELINE_KIND_MESH
        || stage_count == 0
        || usize::from(stage_count) > PIPELINE_V2_MAX_STAGES
        || payload[7] != 0
    {
        return None;
    }
    let state_len = usize::try_from(read_u32_checked(payload, 8)?).ok()?;
    let abi_len = usize::try_from(read_u32_checked(payload, 12)?).ok()?;
    let root_len = usize::try_from(read_u32_checked(payload, 16)?).ok()?;
    if read_u32_checked(payload, 20)? != 0
        || state_len > MAX_PIPELINE_STATE_BYTES
        || abi_len > MAX_PIPELINE_ABI_BYTES
        || root_len > MAX_ROOT_SIGNATURE_BYTES
    {
        return None;
    }

    let mut offset = PIPELINE_V2_HEADER_LEN;
    let mut stages = Vec::with_capacity(usize::from(stage_count));
    for _ in 0..stage_count {
        let stage = *payload.get(offset)?;
        offset = offset.checked_add(1)?;
        let ticket = read_u64_checked(payload, offset)?;
        offset = offset.checked_add(8)?;
        let entry_len = usize::from(read_u16_checked(payload, offset)?);
        offset = offset.checked_add(2)?;
        if entry_len == 0 || entry_len > MAX_PIPELINE_ENTRY_LEN {
            return None;
        }
        let end = offset.checked_add(entry_len)?;
        let entry = std::str::from_utf8(payload.get(offset..end)?).ok()?;
        if !valid_pipeline_entry(entry) {
            return None;
        }
        stages.push(PipelineStageEntry {
            stage,
            ticket,
            entry: entry.to_string(),
        });
        offset = end;
    }

    let state_end = offset.checked_add(state_len)?;
    let state = payload.get(offset..state_end)?.to_vec();
    let abi_end = state_end.checked_add(abi_len)?;
    let abi = payload.get(state_end..abi_end)?.to_vec();
    let root_end = abi_end.checked_add(root_len)?;
    let root_signature = payload.get(abi_end..root_end)?.to_vec();
    if root_end != payload.len() {
        return None;
    }

    Some(PipelineRequestV2 {
        kind,
        stages,
        state,
        abi,
        root_signature,
    })
}

/// Decodes the version 3 frame.
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
/// | ... | R | root-signature file name (optional) |
fn decode_pipeline_request_v3(payload: &[u8]) -> Option<PipelineRequestV3> {
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
        || read_u16_checked(payload, 14)? != 0
        || read_u32_checked(payload, 16)? != 0
        || read_u32_checked(payload, 20)? != 0
    {
        return None;
    }
    let state_len = usize::try_from(read_u32_checked(payload, 8)?).ok()?;
    let root_len = usize::from(read_u16_checked(payload, 12)?);
    if state_len > MAX_PIPELINE_STATE_BYTES || root_len > STAGE_FILE_NAME_LEN {
        return None;
    }

    let mut offset = PIPELINE_V3_HEADER_LEN;
    let mut stages = Vec::with_capacity(usize::from(stage_count));
    for _ in 0..stage_count {
        let stage = *payload.get(offset)?;
        offset = offset.checked_add(1)?;
        let name_len = usize::from(read_u16_checked(payload, offset)?);
        offset = offset.checked_add(2)?;
        if name_len != STAGE_FILE_NAME_LEN {
            return None;
        }
        let name_end = offset.checked_add(name_len)?;
        let file = std::str::from_utf8(payload.get(offset..name_end)?).ok()?;
        if !valid_stage_file_name(file) {
            return None;
        }
        offset = name_end;
        let entry_len = usize::from(read_u16_checked(payload, offset)?);
        offset = offset.checked_add(2)?;
        if entry_len == 0 || entry_len > MAX_PIPELINE_ENTRY_LEN {
            return None;
        }
        let entry_end = offset.checked_add(entry_len)?;
        let entry = std::str::from_utf8(payload.get(offset..entry_end)?).ok()?;
        if !valid_pipeline_entry(entry) {
            return None;
        }
        offset = entry_end;
        stages.push(PipelineStageFile {
            stage,
            file: file.to_string(),
            entry: entry.to_string(),
        });
    }

    let state_end = offset.checked_add(state_len)?;
    let state = payload.get(offset..state_end)?.to_vec();
    let root_end = state_end.checked_add(root_len)?;
    let root = std::str::from_utf8(payload.get(state_end..root_end)?).ok()?;
    if root_end != payload.len() || (root_len != 0 && !valid_stage_file_name(root)) {
        return None;
    }

    Some(PipelineRequestV3 {
        kind,
        stages,
        state,
        root_signature_file: (root_len != 0).then(|| root.to_string()),
    })
}

/// Validates a version 3 request before any file is opened or GPU work exists.
fn validate_pipeline_v3(request: &PipelineRequestV3) -> Result<(), &'static str> {
    let mut seen = Vec::with_capacity(request.stages.len());
    for stage in &request.stages {
        if !matches!(
            stage.stage,
            PIPELINE_STAGE_VERTEX
                | PIPELINE_STAGE_PIXEL
                | PIPELINE_STAGE_MESH
                | PIPELINE_STAGE_OBJECT
                | PIPELINE_STAGE_COMPUTE
        ) || seen.contains(&stage.stage)
            || !valid_stage_file_name(&stage.file)
        {
            return Err("version 3 pipeline frame failed validation");
        }
        seen.push(stage.stage);
    }

    let index_of = |stage_code: u8| {
        request
            .stages
            .iter()
            .position(|stage| stage.stage == stage_code)
    };
    let stage_count = request.stages.len();
    match request.kind {
        PIPELINE_KIND_COMPUTE => {
            if stage_count != 1 {
                return Err("a compute pipeline needs exactly one stage");
            }
            index_of(PIPELINE_STAGE_COMPUTE).ok_or("a compute pipeline needs a compute stage")?;
            if !request.state.is_empty()
                && PipelineState::decode_for(&request.state, RegistryPipelineStage::Compute)
                    .is_err()
            {
                return Err("the compute pipeline state is not a valid VPS1 blob");
            }
        }
        PIPELINE_KIND_GRAPHICS => {
            if stage_count != 2
                || !request.stages.iter().all(|stage| {
                    matches!(stage.stage, PIPELINE_STAGE_VERTEX | PIPELINE_STAGE_PIXEL)
                })
            {
                return Err("a graphics pipeline needs vertex and pixel stages");
            }
            index_of(PIPELINE_STAGE_VERTEX).ok_or("a graphics pipeline needs a vertex stage")?;
            let pixel = index_of(PIPELINE_STAGE_PIXEL)
                .ok_or("a graphics pipeline needs a pixel stage")?;
            let state =
                PipelineState::decode_for(&request.state, RegistryPipelineStage::Graphics)
                    .map_err(|_| "the graphics pipeline state is not a valid VPS1 blob")?;
            if state.fragment_entry.as_deref() != Some(request.stages[pixel].entry.as_str())
                || state.object_entry.is_some()
                || state.color_formats.is_empty()
                || state.color_formats.len() > 8
            {
                return Err("the graphics pipeline state does not match the pixel stage");
            }
        }
        PIPELINE_KIND_MESH => {
            if !(2..=3).contains(&stage_count)
                || !request.stages.iter().all(|stage| {
                    matches!(
                        stage.stage,
                        PIPELINE_STAGE_MESH | PIPELINE_STAGE_PIXEL | PIPELINE_STAGE_OBJECT
                    )
                })
            {
                return Err("a mesh pipeline needs mesh and pixel stages, optionally object");
            }
            index_of(PIPELINE_STAGE_MESH).ok_or("a mesh pipeline needs a mesh stage")?;
            let pixel = index_of(PIPELINE_STAGE_PIXEL)
                .ok_or("a mesh pipeline needs a pixel stage")?;
            let object = index_of(PIPELINE_STAGE_OBJECT);
            let state = PipelineState::decode_for(&request.state, RegistryPipelineStage::Mesh)
                .map_err(|_| "the mesh pipeline state is not a valid VPS1 blob")?;
            if state.fragment_entry.as_deref() != Some(request.stages[pixel].entry.as_str()) {
                return Err("the mesh pipeline state does not match the pixel stage");
            }
            match (object, state.object_entry.as_deref()) {
                (None, None) | (Some(_), Some(_)) => {}
                _ => return Err("the mesh pipeline state and object stage disagree"),
            }
            if state
                .mesh_threadgroup
                .iter()
                .chain(state.object_threadgroup.iter())
                .any(|dimension| *dimension == 0)
            {
                return Err("the mesh pipeline threadgroup has a zero dimension");
            }
        }
        _ => return Err("unknown pipeline kind"),
    }
    Ok(())
}

/// Validates a version 2 request against its kind, stage set, [`PipelineState`]
/// and [`PipelineAbi`], returning the decoded ABI.
///
/// Every failure is an explicit rejection before the executor is called; no
/// partial pipeline is ever created.
fn validate_pipeline_v2(request: &PipelineRequestV2) -> Result<PipelineAbi, ()> {
    let mut previous_ticket = 0u64;
    let mut seen = Vec::with_capacity(request.stages.len());
    for stage in &request.stages {
        if stage.ticket == 0 || stage.ticket <= previous_ticket {
            return Err(());
        }
        if !matches!(
            stage.stage,
            PIPELINE_STAGE_VERTEX
                | PIPELINE_STAGE_PIXEL
                | PIPELINE_STAGE_MESH
                | PIPELINE_STAGE_OBJECT
                | PIPELINE_STAGE_COMPUTE
        ) || seen.contains(&stage.stage)
        {
            return Err(());
        }
        previous_ticket = stage.ticket;
        seen.push(stage.stage);
    }

    let index_of = |stage_code: u8| {
        request
            .stages
            .iter()
            .position(|stage| stage.stage == stage_code)
    };

    let stage_count = request.stages.len();
    match request.kind {
        PIPELINE_KIND_COMPUTE => {
            if stage_count != 1 {
                return Err(());
            }
            index_of(PIPELINE_STAGE_COMPUTE).ok_or(())?;
            if !request.state.is_empty()
                && PipelineState::decode_for(&request.state, RegistryPipelineStage::Compute)
                    .is_err()
            {
                return Err(());
            }
        }
        PIPELINE_KIND_GRAPHICS => {
            if stage_count != 2
                || !request.stages.iter().all(|stage| {
                    matches!(stage.stage, PIPELINE_STAGE_VERTEX | PIPELINE_STAGE_PIXEL)
                })
            {
                return Err(());
            }
            index_of(PIPELINE_STAGE_VERTEX).ok_or(())?;
            let pixel = index_of(PIPELINE_STAGE_PIXEL).ok_or(())?;
            let state = PipelineState::decode_for(
                &request.state,
                RegistryPipelineStage::Graphics,
            )
            .map_err(|_| ())?;
            if state.fragment_entry.as_deref() != Some(request.stages[pixel].entry.as_str())
                || state.object_entry.is_some()
                || state.color_formats.is_empty()
                || state.color_formats.len() > 8
            {
                return Err(());
            }
        }
        PIPELINE_KIND_MESH => {
            if !(2..=3).contains(&stage_count)
                || !request.stages.iter().all(|stage| {
                    matches!(
                        stage.stage,
                        PIPELINE_STAGE_MESH | PIPELINE_STAGE_PIXEL | PIPELINE_STAGE_OBJECT
                    )
                })
            {
                return Err(());
            }
            index_of(PIPELINE_STAGE_MESH).ok_or(())?;
            let pixel = index_of(PIPELINE_STAGE_PIXEL).ok_or(())?;
            let object = index_of(PIPELINE_STAGE_OBJECT);
            let state =
                PipelineState::decode_for(&request.state, RegistryPipelineStage::Mesh)
                    .map_err(|_| ())?;
            if state.fragment_entry.as_deref() != Some(request.stages[pixel].entry.as_str()) {
                return Err(());
            }
            match (object, state.object_entry.as_deref()) {
                (None, None) | (Some(_), Some(_)) => {}
                _ => return Err(()),
            }
            if state
                .mesh_threadgroup
                .iter()
                .chain(state.object_threadgroup.iter())
                .any(|dimension| *dimension == 0)
            {
                return Err(());
            }
        }
        _ => return Err(()),
    }

    let abi = decode_pipeline_abi(&request.abi, request).ok_or(())?;
    if !abi.is_empty() && request.root_signature.is_empty() {
        return Err(());
    }
    Ok(abi)
}

/// Decodes the `VAB1` per-stage ABI container into a [`PipelineAbi`].
///
/// Layout: `magic[4]`, `version u8 = 1`, `count u8`, then per entry
/// `stage u8`, `json_len u32`, converter reflection JSON. An empty blob means
/// "no automatic root binding" and decodes to the default ABI. Every stage
/// must be one the frame already carries; a layout for a stage the frame does
/// not name is a mismatch, not a partial ABI.
fn decode_pipeline_abi(bytes: &[u8], request: &PipelineRequestV2) -> Option<PipelineAbi> {
    if bytes.is_empty() {
        return Some(PipelineAbi::default());
    }
    if bytes.len() < 6 || bytes[..4] != PIPELINE_ABI_MAGIC || bytes[4] != 1 {
        return None;
    }
    let count = usize::from(bytes[5]);
    if count == 0 || count > PIPELINE_V2_MAX_STAGES {
        return None;
    }
    let mut offset = 6usize;
    let mut abi = PipelineAbi::default();
    let mut seen = Vec::with_capacity(count);
    for _ in 0..count {
        let stage = *bytes.get(offset)?;
        offset = offset.checked_add(1)?;
        let len = usize::try_from(read_u32_checked(bytes, offset)?).ok()?;
        offset = offset.checked_add(4)?;
        let end = offset.checked_add(len)?;
        let json = std::str::from_utf8(bytes.get(offset..end)?).ok()?;
        offset = end;
        if !request.stages.iter().any(|entry| entry.stage == stage) || seen.contains(&stage) {
            return None;
        }
        let layout = RootSignatureLayout::from_converter_reflection_json(json).ok()?;
        let frame_entry = request
            .stages
            .iter()
            .find(|entry| entry.stage == stage)?
            .entry
            .as_str();
        if !layout.entry_point().is_empty() && layout.entry_point() != frame_entry {
            return None;
        }
        match stage {
            PIPELINE_STAGE_VERTEX if request.kind == PIPELINE_KIND_GRAPHICS => {
                abi.vertex = Some(layout)
            }
            PIPELINE_STAGE_PIXEL if request.kind != PIPELINE_KIND_COMPUTE => {
                abi.fragment = Some(layout)
            }
            PIPELINE_STAGE_MESH if request.kind == PIPELINE_KIND_MESH => abi.mesh = Some(layout),
            PIPELINE_STAGE_OBJECT if request.kind == PIPELINE_KIND_MESH => abi.object = Some(layout),
            PIPELINE_STAGE_COMPUTE if request.kind == PIPELINE_KIND_COMPUTE => {
                abi.compute = Some(layout)
            }
            _ => return None,
        }
        seen.push(stage);
    }
    if offset != bytes.len() {
        return None;
    }
    Some(abi)
}

fn read_u16_checked(payload: &[u8], offset: usize) -> Option<u16> {
    let end = offset.checked_add(2)?;
    let bytes = payload.get(offset..end)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_checked(payload: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let bytes = payload.get(offset..end)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64_checked(payload: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let bytes = payload.get(offset..end)?;
    let mut value = [0u8; 8];
    value.copy_from_slice(bytes);
    Some(u64::from_le_bytes(value))
}

/// One validated kind 8 texture registration.
struct TextureRequest {
    id: u64,
    generation: u32,
    desc: TextureDesc,
}

/// Decodes a kind 8 payload into a validated [`TextureDesc`].
///
/// The length is exact, every enum code is range-checked and zero extents,
/// zero mip/layer counts and unsupported sample counts are refused before the
/// executor is called.
fn decode_texture_request(payload: &[u8]) -> Option<TextureRequest> {
    if payload.len() != TEXTURE_REG_PAYLOAD_LEN {
        return None;
    }
    let id = read_u64(payload, 0);
    let generation = read_u32(payload, 8);
    let dimension = texture_dimension_from_code(payload[12])?;
    let format = texture_format_from_code(payload[13])?;
    let usage = texture_usage_for_format(payload[14], format)?;
    let sample_count = u32::from(payload[15]);
    let width = read_u32(payload, 16);
    let height = read_u32(payload, 20);
    let depth = read_u32(payload, 24);
    let mip_levels = u32::from(read_u16(payload, 28));
    let array_layers = u32::from(read_u16(payload, 30));
    if width == 0 || height == 0 || depth == 0 {
        return None;
    }
    if mip_levels == 0 || array_layers == 0 {
        return None;
    }
    if !matches!(sample_count, 1 | 2 | 4 | 8) {
        return None;
    }
    Some(TextureRequest {
        id,
        generation,
        desc: TextureDesc {
            width,
            height,
            depth,
            mip_levels,
            sample_count,
            array_layers,
            dimension,
            format,
            usage,
            label: None,
        },
    })
}

/// One decoded kind 10 sampler registration.
struct SamplerRequest {
    id: u64,
    desc: SamplerDesc,
}

fn decode_sampler_request(payload: &[u8]) -> Option<SamplerRequest> {
    if payload.len() != SAMPLER_REG_PAYLOAD_LEN {
        return None;
    }
    let id = read_u64(payload, 0);
    if id == 0 {
        return None;
    }
    let min_filter = read_u32(payload, 8);
    let mag_filter = read_u32(payload, 12);
    let s_address_mode = read_u32(payload, 16);
    let t_address_mode = read_u32(payload, 20);
    let r_address_mode = read_u32(payload, 24);
    // The host sampler shape is `MTLSamplerMinMagFilter` (0/1) and
    // `MTLSamplerAddressMode` (0..=4). Anything else is refused before the
    // device sees a descriptor.
    if min_filter > 1 || mag_filter > 1 {
        return None;
    }
    if [s_address_mode, t_address_mode, r_address_mode]
        .into_iter()
        .any(|mode| mode > 4)
    {
        return None;
    }
    Some(SamplerRequest {
        id,
        desc: SamplerDesc {
            min_filter: min_filter as usize,
            mag_filter: mag_filter as usize,
            s_address_mode: s_address_mode as usize,
            t_address_mode: t_address_mode as usize,
            r_address_mode: r_address_mode as usize,
        },
    })
}

/// Maps a kind 8 dimension code to a registry dimension.
fn texture_dimension_from_code(code: u8) -> Option<TextureDimension> {
    Some(match code {
        0 => TextureDimension::D1,
        1 => TextureDimension::D2,
        2 => TextureDimension::D2Array,
        3 => TextureDimension::Cube,
        4 => TextureDimension::CubeArray,
        5 => TextureDimension::D3,
        _ => return None,
    })
}

/// Maps a kind 8 format code to a registry format.
///
/// The codes are a stable, documented subset of the registry formats; a code
/// outside this table is an explicit "unsupported format" rejection. This
/// kind-8 numbering is separate from [`TextureFormat::code`] (the `VPS1`/`VPS2`
/// pipeline-state numbering), where `R32Uint`/`R32Sint` are codes 18 and 19;
/// the two spaces never decode each other's bytes.
fn texture_format_from_code(code: u8) -> Option<TextureFormat> {
    // ADR-033: the canonical table owns the kind 8 code space. Codes 0-11 keep
    // their historical meaning (RGBA8, BGRA8, RGBA8 sRGB, R32F, R16F,
    // R11G11B10F, RGB10A2, D32F, D32S8, RGBA16F, R32Uint, R32Sint); every
    // later code is the canonical table's `wire` value, so a guest format the
    // table maps is accepted here without a second partial match.
    TextureFormat::from_wire_code(code)
}

/// Maps a kind 8 usage code to a registry usage set.
fn texture_usage_from_code(code: u8) -> Option<TextureUsage> {
    Some(match code {
        0 => TextureUsage::sampled(),
        1 => TextureUsage::color_attachment(),
        2 => TextureUsage::depth_attachment(),
        3 => TextureUsage {
            shader_read: true,
            shader_write: true,
            render_target: false,
            depth_stencil: false,
        },
        _ => return None,
    })
}

/// The usage a kind 8 usage code names, widened for the integer R32 formats.
///
/// R32_UINT and R32_SINT are storage (UAV) targets as well as colour
/// attachments. The guest names a UAV target (resource flag 0x4) with usage 1
/// (colour attachment), so the host adds the shader-write bit those formats
/// need for a compute write and keeps the render-target bit for an RTV. A
/// sampled or depth use is left exactly as sent.
fn texture_usage_for_format(code: u8, format: TextureFormat) -> Option<TextureUsage> {
    let usage = texture_usage_from_code(code)?;
    if (!usage.render_target && !usage.shader_write) || format.is_depth_stencil() {
        return Some(usage);
    }
    // ADR-033: an integer storage target (R8/R16/R32 UINT or SINT, including the
    // title's 640x360 R8_UINT pass) is readable, writable and attachable. The
    // historical R32-only widening could not express any other integer format.
    let name = format.name();
    if !name.contains("UINT") && !name.contains("SINT") {
        return Some(usage);
    }
    Some(TextureUsage {
        shader_read: true,
        shader_write: true,
        render_target: true,
        depth_stencil: false,
    })
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
    let limit = if header[0] == KIND_SUBMISSION_V3 || header[0] == KIND_SUBMISSION_ASYNC {
        MAX_PAYLOAD_LEN_V3
    } else {
        MAX_PAYLOAD_LEN
    };
    if len > limit {
        return Err(HostError::Protocol(
            "frame payload exceeds the maximum for its kind",
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
fn write_frame(stream: &mut impl Write, kind: u8, payload: &[u8]) -> Result<(), HostError> {
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

/// Guest-side fire-and-forget transport for the loopback host session.
///
/// This is the concrete [`GuestTransport`] implementation for the native
/// renderer boundary: it speaks the documented local wire protocol, publishes
/// the submitted fence locally, and reads GPU completion from the shared page
/// with a single atomic load. It references no Wine or D3D12 symbol.
///
/// The host still performs the GPU work synchronously inside its session
/// thread; only this caller is non-blocking. See the module note on the
/// remaining limitation.
///
/// One transport owns the session's sequence space: every frame it sends is a
/// kind 7 submission whose sequence becomes the fence value, and mixing kind 5
/// frames on the same session is not supported.
pub struct LoopbackAsyncTransport {
    stream: TcpStream,
    page: FencePageMapping,
    waker: Box<dyn Wake>,
}

impl LoopbackAsyncTransport {
    /// Connects, authenticates, and registers `page` as the session fence page.
    ///
    /// `page` must be a mapping of `<share dir>/gpu-fence.v1.page`. The
    /// handshake consumes exactly the authentication and registration
    /// acknowledgements; after it returns, the host sends no frame this
    /// transport has to read.
    pub fn connect(
        addr: SocketAddr,
        secret: [u8; SECRET_LEN],
        page: FencePageMapping,
        waker: Box<dyn Wake>,
    ) -> Result<Self, HostError> {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(DEFAULT_IO_TIMEOUT))?;
        stream.set_write_timeout(Some(DEFAULT_IO_TIMEOUT))?;

        write_frame(&mut stream, KIND_AUTH, &secret)?;
        match read_frame(&mut stream)? {
            Some((KIND_ACK, payload)) if payload == [0] => {}
            Some(_) => return Err(HostError::Unauthenticated),
            None => return Err(HostError::Protocol("host closed before authenticating")),
        }

        let mut registration = [0u8; FENCE_PAGE_REGISTER_PAYLOAD_LEN];
        registration.copy_from_slice(&(crate::fence_page::FENCE_PAGE_LEN as u64).to_le_bytes());
        write_frame(&mut stream, KIND_FENCE_PAGE, &registration)?;
        match read_frame(&mut stream)? {
            Some((KIND_ACK, payload)) if payload == [0] => {}
            Some(_) => {
                return Err(HostError::Protocol(
                    "host rejected the shared fence page registration",
                ));
            }
            None => {
                return Err(HostError::Protocol(
                    "host closed before registering the fence page",
                ));
            }
        }

        Ok(Self {
            stream,
            page,
            waker,
        })
    }

    /// The TCP stream this transport owns, for diagnostics and tests.
    ///
    /// No reply frame corresponds to an asynchronous submission, so reading
    /// from this stream must not be used to learn about completion.
    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }
}

impl GuestTransport for LoopbackAsyncTransport {
    fn submit_async(&self, frame: &[u8]) -> Result<u64, MetalError> {
        let submission = decode_submission_v3(frame).map_err(|_| {
            MetalError::InvalidArg("asynchronous submission is not a valid version 3 frame")
        })?;
        let sequence = submission.sequence;
        if sequence == 0 {
            return Err(MetalError::InvalidArg(
                "asynchronous submission sequence must be nonzero",
            ));
        }
        if frame.len() > MAX_SUBMISSION_V3_LEN {
            return Err(MetalError::InvalidArg(
                "asynchronous submission exceeds the version 3 frame limit",
            ));
        }
        if u64::from(sequence) <= self.page.page().submitted() {
            // Reject locally: the host would treat this as a sequencing
            // violation and close the session.
            return Err(MetalError::InvalidArg(
                "asynchronous submission sequence must increase",
            ));
        }
        let declared = u32::try_from(frame.len()).map_err(|_| {
            MetalError::InvalidArg("asynchronous submission length does not fit u32")
        })?;

        let mut envelope = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
        envelope.extend_from_slice(&V3_REQUEST_MAGIC);
        envelope.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
        envelope.extend_from_slice(&sequence.to_le_bytes());
        envelope.extend_from_slice(&0u16.to_le_bytes());
        envelope.extend_from_slice(&declared.to_le_bytes());
        envelope.extend_from_slice(frame);

        let mut writer = &self.stream;
        write_frame(&mut writer, KIND_SUBMISSION_ASYNC, &envelope).map_err(|error| {
            MetalError::InternalError(format!("asynchronous submission write failed: {error}"))
        })?;

        let submitted = self.page.page().publish_submitted(u64::from(sequence));
        self.waker.wake_best_effort();
        Ok(submitted)
    }

    fn completed_fence(&self) -> u64 {
        // One `Ordering::Acquire` load; no syscall and no IPC round-trip.
        self.page.page().completed()
    }

    fn fence_page(&self) -> &FencePage {
        self.page.page()
    }

    fn wake(&self) {
        self.waker.wake_best_effort();
    }
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
        v3_submissions: Vec<SubmissionV3>,
        v3_executions: u64,
        v3_not_supported: bool,
        /// When set, the next `submit_v3` fails with a synthetic Metal error
        /// and clears the flag, so a later submission in the same session can
        /// succeed and prove the loop stayed open.
        v3_fail_once: bool,
        /// Synthetic v3 executor failures, consumed one per `submit_v3`, used
        /// to exercise the reason trailer with short, maximal and overlong
        /// host diagnostics.
        v3_failure_reasons: Vec<String>,
        /// Content-hashed per-stage graphics registrations: `(handle, stages,
        /// state, abi)` where each stage is `(wire code, metallib, entry)`.
        graphics_pipelines: Vec<(u64, Vec<(u8, Vec<u8>, String)>, Vec<u8>, PipelineAbi)>,
        /// Content-hashed per-stage mesh registrations, same shape.
        mesh_pipelines: Vec<(u64, Vec<(u8, Vec<u8>, String)>, Vec<u8>, PipelineAbi)>,
        /// Content-hashed per-stage compute registrations, same shape.
        compute_pipelines: Vec<(u64, Vec<(u8, Vec<u8>, String)>, Vec<u8>, PipelineAbi)>,
        /// Refusals the executor returned to the host, in arrival order.
        refusals: Vec<String>,
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

        /// Records one per-stage registration, refusing an entry point the stub
        /// pretends is absent from its own library.
        ///
        /// The real executor resolves every function in the library the stage
        /// names; `ps_missing` (and `vm_fail` for any stage) stands in for a
        /// library that does not contain the requested function, so its error
        /// message names the failing stage exactly like a real Metal failure.
        fn record_stages(
            &self,
            family: &'static str,
            handle: u64,
            stages: &[PipelineStageLibrary<'_>],
            state: &[u8],
            abi: PipelineAbi,
        ) -> Result<(), MetalError> {
            let recorded: Vec<(u8, Vec<u8>, String)> = stages
                .iter()
                .map(|stage| {
                    (
                        stage.stage,
                        stage.metallib.to_vec(),
                        stage.entry.to_string(),
                    )
                })
                .collect();
            if let Some(stage) = recorded
                .iter()
                .find(|stage| stage.2 == "ps_missing" || stage.2 == "vm_fail")
            {
                let message = format!(
                    "the {} stage entry point `{}` was not found in its library",
                    pipeline_stage_label(stage.0),
                    stage.2
                );
                self.state().refusals.push(message.clone());
                return Err(MetalError::InternalError(message));
            }
            let mut stub = self.state();
            let record = (handle, recorded, state.to_vec(), abi);
            match family {
                "graphics" => stub.graphics_pipelines.push(record),
                "mesh" => stub.mesh_pipelines.push(record),
                _ => stub.compute_pipelines.push(record),
            }
            Ok(())
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

        fn submit_v3(
            &self,
            submission: &SubmissionV3,
            _allocations: &HashMap<(u64, u32), VmBufferId>,
        ) -> Result<Reply, MetalError> {
            let mut state = self.state();
            if state.v3_not_supported {
                return Err(MetalError::NotSupported("synthetic v3 refusal"));
            }
            if !state.v3_failure_reasons.is_empty() {
                return Err(MetalError::InternalError(
                    state.v3_failure_reasons.remove(0),
                ));
            }
            if state.v3_fail_once {
                state.v3_fail_once = false;
                return Err(MetalError::InternalError(
                    "draw #12 (MTLCommandBuffer.status=4): synthetic out-of-bounds write"
                        .to_string(),
                ));
            }
            state.v3_submissions.push(submission.clone());
            state.v3_executions += 1;
            Ok(Reply {
                sequence: submission.sequence,
                status: ReplyStatus::Ok,
            })
        }

        /// Accepts kind 8 registrations so a session test can submit a
        /// descriptor table that names a registered texture handle.
        fn register_texture(&self, _id: u64, _desc: TextureDesc) -> Result<(), MetalError> {
            Ok(())
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

        fn register_graphics_pipeline_stages(
            &self,
            handle: u64,
            stages: &[PipelineStageLibrary<'_>],
            state: &[u8],
            abi: PipelineAbi,
        ) -> Result<(), MetalError> {
            self.record_stages("graphics", handle, stages, state, abi)
        }

        fn register_mesh_pipeline_stages(
            &self,
            handle: u64,
            stages: &[PipelineStageLibrary<'_>],
            state: &[u8],
            abi: PipelineAbi,
        ) -> Result<(), MetalError> {
            self.record_stages("mesh", handle, stages, state, abi)
        }

        fn register_compute_pipeline_stages(
            &self,
            handle: u64,
            stages: &[PipelineStageLibrary<'_>],
            state: &[u8],
            abi: PipelineAbi,
        ) -> Result<(), MetalError> {
            self.record_stages("compute", handle, stages, state, abi)
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

    /// The exact version 2 frame the guest builds.
    fn pipeline_request_v2(
        kind: u8,
        stages: &[(u8, u64, &str)],
        state: &[u8],
        abi: &[u8],
        root_signature: &[u8],
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&PIPELINE_FRAME_MAGIC);
        payload.push(PIPELINE_FRAME_VERSION_V2);
        payload.push(kind);
        payload.push(stages.len() as u8);
        payload.push(0);
        payload.extend_from_slice(&(state.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(abi.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(root_signature.len() as u32).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        for (stage, ticket, entry) in stages {
            payload.push(*stage);
            payload.extend_from_slice(&ticket.to_le_bytes());
            payload.extend_from_slice(&(entry.len() as u16).to_le_bytes());
            payload.extend_from_slice(entry.as_bytes());
        }
        payload.extend_from_slice(state);
        payload.extend_from_slice(abi);
        payload.extend_from_slice(root_signature);
        payload
    }

    fn abi_container(stages: &[(u8, &str)]) -> Vec<u8> {
        let mut abi = Vec::new();
        abi.extend_from_slice(&PIPELINE_ABI_MAGIC);
        abi.push(1);
        abi.push(stages.len() as u8);
        for (stage, json) in stages {
            abi.push(*stage);
            abi.extend_from_slice(&(json.len() as u32).to_le_bytes());
            abi.extend_from_slice(json.as_bytes());
        }
        abi
    }

    const VERTEX_REFLECTION: &str =
        r#"{"EntryPoint":"vs_main","ShaderType":"Vertex","TopLevelArgumentBuffer":[]}"#;
    const PIXEL_REFLECTION: &str =
        r#"{"EntryPoint":"ps_main","ShaderType":"Fragment","TopLevelArgumentBuffer":[]}"#;
    const PIXEL_MISSING_REFLECTION: &str =
        r#"{"EntryPoint":"ps_missing","ShaderType":"Fragment","TopLevelArgumentBuffer":[]}"#;
    const MESH_REFLECTION: &str =
        r#"{"EntryPoint":"ms_main","ShaderType":"Mesh","TopLevelArgumentBuffer":[]}"#;

    fn graphics_state() -> Vec<u8> {
        PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            depth_format: Some(TextureFormat::Depth32Float),
            fragment_entry: Some("ps_main".to_string()),
            ..PipelineState::default()
        }
        .encode(RegistryPipelineStage::Graphics)
    }

    fn mesh_state() -> Vec<u8> {
        PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            depth_format: None,
            fragment_entry: Some("ps_main".to_string()),
            mesh_threadgroup: [32, 1, 1],
            object_threadgroup: [1, 1, 1],
            ..PipelineState::default()
        }
        .encode(RegistryPipelineStage::Mesh)
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

    fn v3_envelope(submission: &SubmissionV3) -> Vec<u8> {
        let frame = crate::wire_v3::encode_submission_v3(submission).expect("encode v3 submission");
        v3_envelope_from_frame(submission.sequence, &frame)
    }

    /// Wraps one already-encoded v3 frame in the `V3RQ` request envelope.
    fn v3_envelope_from_frame(sequence: u32, frame: &[u8]) -> Vec<u8> {
        let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
        payload.extend_from_slice(&V3_REQUEST_MAGIC);
        payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
        payload.extend_from_slice(&sequence.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        payload.extend_from_slice(frame);
        payload
    }

    fn v3_submission(
        sequence: u32,
        list: Option<ListIdentity>,
        binding: BufferBinding,
    ) -> SubmissionV3 {
        SubmissionV3 {
            sequence,
            list,
            commands: vec![CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: PIPELINE_ID,
                threads: (BYTES / 4) as u32,
                threads_per_group: 64,
                bindings: vec![binding],
            })],
        }
    }

    fn roundtrip_v3(
        client: &mut TcpStream,
        submission: &SubmissionV3,
    ) -> Reply {
        roundtrip_v3_detail(client, submission).without_reason()
    }

    /// Sends one v3 submission and decodes the reply together with its optional
    /// reason trailer.
    fn roundtrip_v3_detail(
        client: &mut TcpStream,
        submission: &SubmissionV3,
    ) -> wire::ReplyWithReason {
        write_frame(client, KIND_SUBMISSION_V3, &v3_envelope(submission))
            .expect("send v3 submission");
        let (kind, payload) = read_frame(client)
            .expect("read v3 reply")
            .expect("v3 reply frame");
        assert_eq!(kind, KIND_SUBMISSION_V3);
        wire::decode_reply_with_reason(&payload).expect("decode v3 reply")
    }

    /// Asserts one v3 envelope was refused for `sequence` with a non-empty
    /// reason containing `needle`.
    fn assert_rejected_with_reason(
        result: &Result<SubmissionV3, V3RequestError>,
        sequence: u32,
        needle: &str,
    ) {
        match result {
            Err(V3RequestError::Rejected {
                sequence: actual,
                reason,
            }) => {
                assert_eq!(*actual, sequence);
                assert!(
                    reason.contains(needle),
                    "reason `{reason}` does not name `{needle}`"
                );
                assert!(!reason.is_empty(), "the refusal must carry a reason");
            }
            other => panic!("expected a rejected envelope, got {other:?}"),
        }
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

    /// A version 2 graphics frame builds each function from the library its own
    /// stage names: a distinct vertex and pixel metallib both reach the
    /// executor, in stage order, with the exact state and ABI the guest sent.
    #[test]
    fn pipeline_registration_v2_graphics_uses_each_stage_library() {
        let dir = TestDir::new();
        dir.write_pipeline(1, b"vertex metallib");
        dir.write_pipeline(2, b"pixel metallib");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let state = graphics_state();
        let abi = abi_container(&[
            (PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_REFLECTION),
        ]);
        let request = pipeline_request_v2(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, 1, "vs_main"),
                (PIPELINE_STAGE_PIXEL, 2, "ps_main"),
            ],
            &state,
            &abi,
            b"raw root signature",
        );
        write_frame(&mut client, KIND_PIPELINE, &request).expect("send v2 graphics");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(1))
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert!(host.last_pipeline_rejection().is_none());
        let state = host.executor.state();
        assert!(state.pipelines.is_empty(), "no compute load for a graphics PSO");
        assert_eq!(state.graphics_pipelines.len(), 1);
        let (handle, stages, captured_state, abi) = &state.graphics_pipelines[0];
        assert_eq!(*handle, 1);
        assert_eq!(
            stages,
            &vec![
                (PIPELINE_STAGE_VERTEX, b"vertex metallib".to_vec(), "vs_main".to_string()),
                (PIPELINE_STAGE_PIXEL, b"pixel metallib".to_vec(), "ps_main".to_string()),
            ],
            "the vertex and pixel functions come from their own libraries"
        );
        assert_eq!(captured_state, &graphics_state());
        assert!(abi.vertex.is_some(), "vertex ABI captured");
        assert!(abi.fragment.is_some(), "fragment ABI captured");
        assert!(abi.mesh.is_none() && abi.object.is_none() && abi.compute.is_none());
    }

    /// A graphics frame whose pixel library exists but does not contain the
    /// pixel entry point is refused: status 1, a diagnostic naming the pixel
    /// stage, and no content pipeline handle consumed.
    #[test]
    fn pipeline_registration_v2_refuses_a_missing_pixel_stage_function() {
        let dir = TestDir::new();
        dir.write_pipeline(1, b"vertex metallib");
        dir.write_pipeline(2, b"pixel metallib");
        dir.write_pipeline(3, b"vertex metallib");
        dir.write_pipeline(4, b"pixel metallib");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let missing_state = PipelineState {
            color_formats: vec![TextureFormat::Rgba8Unorm],
            fragment_entry: Some("ps_missing".to_string()),
            ..PipelineState::default()
        }
        .encode(RegistryPipelineStage::Graphics);
        let missing_abi = abi_container(&[
            (PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_MISSING_REFLECTION),
        ]);
        let refused = pipeline_request_v2(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, 1, "vs_main"),
                (PIPELINE_STAGE_PIXEL, 2, "ps_missing"),
            ],
            &missing_state,
            &missing_abi,
            b"raw root signature",
        );
        write_frame(&mut client, KIND_PIPELINE, &refused).expect("send refused v2 graphics");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_REJECTED, None),
            "a missing pixel function is an explicit rejection"
        );

        // The refused frame consumed no handle: the next valid registration is
        // the first content pipeline.
        let abi = abi_container(&[
            (PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_REFLECTION),
        ]);
        let accepted = pipeline_request_v2(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, 3, "vs_main"),
                (PIPELINE_STAGE_PIXEL, 4, "ps_main"),
            ],
            &graphics_state(),
            &abi,
            b"raw root signature",
        );
        write_frame(&mut client, KIND_PIPELINE, &accepted).expect("send valid v2 graphics");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(1)),
            "the refusal allocated no pipeline id"
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert_eq!(state.graphics_pipelines.len(), 1);
        assert_eq!(state.graphics_pipelines[0].0, 1);
        assert_eq!(state.refusals.len(), 1);
        let refusal = &state.refusals[0];
        assert!(refusal.contains("pixel"), "{refusal}");
        assert!(refusal.contains("ps_missing"), "{refusal}");
        assert_eq!(
            host.last_pipeline_rejection(),
            None,
            "the later successful registration cleared the record"
        );
    }

    /// A version 2 mesh frame registers the mesh and pixel functions from their
    /// own libraries.
    #[test]
    fn pipeline_registration_v2_round_trips_mesh() {
        let dir = TestDir::new();
        dir.write_pipeline(1, b"mesh metallib");
        dir.write_pipeline(2, b"pixel metallib");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let state = mesh_state();
        let abi = abi_container(&[
            (PIPELINE_STAGE_MESH, MESH_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_REFLECTION),
        ]);
        let request = pipeline_request_v2(
            PIPELINE_KIND_MESH,
            &[
                (PIPELINE_STAGE_MESH, 1, "ms_main"),
                (PIPELINE_STAGE_PIXEL, 2, "ps_main"),
            ],
            &state,
            &abi,
            b"mesh root signature",
        );
        write_frame(&mut client, KIND_PIPELINE, &request).expect("send v2 mesh");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(1))
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert_eq!(state.mesh_pipelines.len(), 1);
        let (_, stages, captured_state, abi) = &state.mesh_pipelines[0];
        assert_eq!(
            stages,
            &vec![
                (PIPELINE_STAGE_MESH, b"mesh metallib".to_vec(), "ms_main".to_string()),
                (PIPELINE_STAGE_PIXEL, b"pixel metallib".to_vec(), "ps_main".to_string()),
            ]
        );
        assert_eq!(captured_state, &mesh_state());
        assert!(abi.mesh.is_some(), "mesh ABI captured");
        assert!(abi.fragment.is_some(), "fragment ABI captured");
    }

    /// A version 2 compute frame resolves its kernel in the compute stage's own
    /// library through the content-hashed family, not an assumed first library.
    #[test]
    fn pipeline_registration_v2_compute_uses_its_stage_library() {
        let dir = TestDir::new();
        dir.write_pipeline(1, b"compute metallib");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let request = pipeline_request_v2(
            PIPELINE_KIND_COMPUTE,
            &[(PIPELINE_STAGE_COMPUTE, 1, "cs_main")],
            b"",
            b"",
            b"",
        );
        write_frame(&mut client, KIND_PIPELINE, &request).expect("send v2 compute");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(1))
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert!(state.pipelines.is_empty(), "v2 compute keeps a content handle");
        assert_eq!(state.compute_pipelines.len(), 1);
        let (_, stages, _, _) = &state.compute_pipelines[0];
        assert_eq!(
            stages,
            &vec![(PIPELINE_STAGE_COMPUTE, b"compute metallib".to_vec(), "cs_main".to_string())]
        );
    }

    /// Malformed version 2 frames are one rejection byte before any executor
    /// call: a bad stage count, a mismatched ABI and a missing root signature.
    #[test]
    fn pipeline_registration_v2_rejects_malformed_frames() {
        let dir = TestDir::new();
        dir.write_pipeline(1, b"vertex metallib");
        dir.write_pipeline(2, b"pixel metallib");
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let state = graphics_state();
        let good_abi = abi_container(&[
            (PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_REFLECTION),
        ]);
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "one stage is not a graphics pipeline",
                pipeline_request_v2(
                    PIPELINE_KIND_GRAPHICS,
                    &[(PIPELINE_STAGE_VERTEX, 1, "vs_main")],
                    &state,
                    &good_abi,
                    b"root",
                ),
            ),
            (
                "graphics ABI with a mesh layout",
                pipeline_request_v2(
                    PIPELINE_KIND_GRAPHICS,
                    &[
                        (PIPELINE_STAGE_VERTEX, 1, "vs_main"),
                        (PIPELINE_STAGE_PIXEL, 2, "ps_main"),
                    ],
                    &state,
                    &abi_container(&[(PIPELINE_STAGE_MESH, MESH_REFLECTION)]),
                    b"root",
                ),
            ),
            (
                "ABI with no root signature",
                pipeline_request_v2(
                    PIPELINE_KIND_GRAPHICS,
                    &[
                        (PIPELINE_STAGE_VERTEX, 1, "vs_main"),
                        (PIPELINE_STAGE_PIXEL, 2, "ps_main"),
                    ],
                    &state,
                    &good_abi,
                    b"",
                ),
            ),
            (
                "mesh state on a graphics frame",
                pipeline_request_v2(
                    PIPELINE_KIND_GRAPHICS,
                    &[
                        (PIPELINE_STAGE_VERTEX, 1, "vs_main"),
                        (PIPELINE_STAGE_PIXEL, 2, "ps_main"),
                    ],
                    &mesh_state(),
                    &good_abi,
                    b"root",
                ),
            ),
        ];
        for (label, request) in cases {
            write_frame(&mut client, KIND_PIPELINE, &request).expect("send malformed v2");
            assert_eq!(
                read_pipeline_reply(&mut client),
                (PIPELINE_REPLY_REJECTED, None),
                "{label}"
            );
        }

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert!(state.graphics_pipelines.is_empty());
        assert!(state.mesh_pipelines.is_empty());
        assert!(state.compute_pipelines.is_empty());
        assert!(state.pipelines.is_empty());
    }

    /// The monotonic ticket ledger covers every stage ticket and is only
    /// advanced by a successful version 2 registration.
    #[test]
    fn pipeline_registration_v2_keeps_the_monotonic_ticket_ledger() {
        let dir = TestDir::new();
        for ticket in 1..=4 {
            dir.write_pipeline(ticket, b"a stage metallib");
        }
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);

        let state = graphics_state();
        let abi = abi_container(&[
            (PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION),
            (PIPELINE_STAGE_PIXEL, PIXEL_REFLECTION),
        ]);
        let request = |first: u64, second: u64| {
            pipeline_request_v2(
                PIPELINE_KIND_GRAPHICS,
                &[
                    (PIPELINE_STAGE_VERTEX, first, "vs_main"),
                    (PIPELINE_STAGE_PIXEL, second, "ps_main"),
                ],
                &state,
                &abi,
                b"root",
            )
        };

        write_frame(&mut client, KIND_PIPELINE, &request(1, 2)).expect("first v2");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(1))
        );
        // The last ledger entry is 2, so a frame reusing ticket 2 is rejected
        // without consuming 3 or 4.
        write_frame(&mut client, KIND_PIPELINE, &request(2, 3)).expect("nonmonotonic v2");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_REJECTED, None)
        );
        write_frame(&mut client, KIND_PIPELINE, &request(3, 4)).expect("later v2");
        assert_eq!(
            read_pipeline_reply(&mut client),
            (PIPELINE_REPLY_OK, Some(2)),
            "the next content handle is used"
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(host.executor.state().graphics_pipelines.len(), 2);
        assert_eq!(host.last_pipeline_ticket, 4);
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

    /// Two peers accepted by one listener are served as two independent
    /// sessions. Peer A's `VP3\0` pipeline registration and peer B's v3
    /// submission each read their own reply frame, exactly the ADR-041 layout
    /// where a translation connection runs beside the render connection.
    #[test]
    fn concurrent_peers_keep_their_own_reply_frames() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind the concurrent listener");
        let addr = listener.local_addr().expect("concurrent listener address");
        let server = thread::spawn(move || {
            let (pipeline_stream, _) = listener.accept().expect("accept the pipeline peer");
            let (submit_stream, _) = listener.accept().expect("accept the submission peer");
            let mut pipeline_host = host(&dir, StubExecutor::new(4096));
            let mut submission_host = host(&dir, StubExecutor::new(4096));
            let pipeline = thread::spawn(move || pipeline_host.serve_once(pipeline_stream));
            let submission = submission_host.serve_once(submit_stream);
            (
                pipeline.join().expect("pipeline session thread"),
                submission,
            )
        });

        let mut pipeline_peer = TcpStream::connect(addr).expect("connect the pipeline peer");
        let mut submit_peer = TcpStream::connect(addr).expect("connect the submission peer");
        for peer in [&mut pipeline_peer, &mut submit_peer] {
            peer.set_read_timeout(Some(Duration::from_secs(5)))
                .expect("peer read timeout");
            authenticate(peer);
        }

        // The pipeline frame names stage files that do not exist, so the host
        // answers a bounded kind 129 refusal without needing a converter or GPU;
        // the important part is that the frame is routed as a pipeline request.
        let vertex_name = stage_name(b"missing-vertex-dxil");
        let pixel_name = stage_name(b"missing-pixel-dxil");
        let pipeline = pipeline_request_v3(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, vertex_name.as_str(), "vs_main"),
                (PIPELINE_STAGE_PIXEL, pixel_name.as_str(), "ps_main"),
            ],
            &graphics_state(),
            None,
        );
        write_frame(&mut pipeline_peer, KIND_PIPELINE, &pipeline).expect("send the pipeline frame");
        // The submission session resolves its own allocation snapshot; register
        // the file on that connection, not the pipeline one.
        write_frame(
            &mut submit_peer,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("register the submission allocation");
        expect_ack(&mut submit_peer, true);
        write_frame(
            &mut submit_peer,
            KIND_SUBMISSION_V3,
            &v3_envelope(&v3_submission(1, None, full_binding(BYTES as u64))),
        )
        .expect("send the v3 submission");

        let (kind, reply) = read_frame(&mut pipeline_peer)
            .expect("read the pipeline reply")
            .expect("pipeline reply frame");
        assert_eq!(
            kind, KIND_PIPELINE_REPLY,
            "the pipeline peer reads its own kind 129 reply"
        );
        assert_eq!(reply.first().copied(), Some(PIPELINE_REPLY_REJECTED));
        assert!(
            reply.len() > 1,
            "a version 3 rejection carries a bounded reason, not one byte"
        );

        let (kind, reply) = read_frame(&mut submit_peer)
            .expect("read the submission reply")
            .expect("submission reply frame");
        assert_eq!(
            kind, KIND_SUBMISSION_V3,
            "the render peer reads its own kind 5 reply"
        );
        let reply = wire::decode_reply(&reply).expect("decode the submission reply");
        assert_eq!(reply.sequence, 1);
        assert_eq!(reply.status, ReplyStatus::Ok);

        drop(pipeline_peer);
        drop(submit_peer);
        let (pipeline, submission) = server.join().expect("concurrent server thread");
        pipeline.expect("the pipeline session ends cleanly");
        submission.expect("the submission session ends cleanly");
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

    #[test]
    fn v3_envelope_rejects_hostile_frames_without_panicking() {
        let valid = v3_submission(1, None, full_binding(BYTES as u64));
        let payload = v3_envelope(&valid);
        assert!(decode_v3_request(&payload).is_ok());

        // Unknown envelope version.
        let mut wrong_version = payload.clone();
        wrong_version[4..6].copy_from_slice(&(PROTOCOL_VERSION_V3 + 1).to_le_bytes());
        assert_rejected_with_reason(&decode_v3_request(&wrong_version), 1, "version");

        // Nonzero reserved word.
        let mut reserved = payload.clone();
        reserved[10..12].copy_from_slice(&1u16.to_le_bytes());
        assert_rejected_with_reason(&decode_v3_request(&reserved), 1, "reserved");

        // Truncated body, declared length unchanged.
        let truncated = &payload[..payload.len() - 1];
        assert_rejected_with_reason(&decode_v3_request(truncated), 1, "carries");

        // Trailing byte after the declared submission.
        let mut trailing = payload.clone();
        trailing.push(0);
        assert_rejected_with_reason(&decode_v3_request(&trailing), 1, "carries");

        // Oversized declared submission length.
        let mut oversized = payload.clone();
        oversized[12..16].copy_from_slice(&((MAX_SUBMISSION_V3_LEN as u32) + 1).to_le_bytes());
        assert_rejected_with_reason(&decode_v3_request(&oversized), 1, "limit");

        // Unknown inner codec version.
        let mut inner_version = payload.clone();
        inner_version[V3_REQUEST_HEADER_LEN + 4..V3_REQUEST_HEADER_LEN + 6]
            .copy_from_slice(&(PROTOCOL_VERSION_V3 + 1).to_le_bytes());
        assert_rejected_with_reason(&decode_v3_request(&inner_version), 1, "unsupported");

        // Short envelope: no recoverable sequence.
        assert_eq!(
            decode_v3_request(&payload[..V3_REQUEST_HEADER_LEN - 1]),
            Err(V3RequestError::Unsequenced)
        );
        // Wrong magic: no recoverable sequence.
        let mut wrong_magic = payload.clone();
        wrong_magic[0] ^= 0xff;
        assert_eq!(
            decode_v3_request(&wrong_magic),
            Err(V3RequestError::Unsequenced)
        );
        // Zero sequence: not reply-correlatable.
        let mut zero_sequence = payload.clone();
        zero_sequence[6..10].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            decode_v3_request(&zero_sequence),
            Err(V3RequestError::Unsequenced)
        );
    }

    #[test]
    fn v3_roundtrip_executes_and_refuses_replay() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let identity = list_identity(31, 1);
        let first = v3_submission(1, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &first),
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );
        let replay = v3_submission(2, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &replay),
            Reply {
                sequence: 2,
                status: ReplyStatus::AlreadyExecuted,
            }
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert_eq!(
            result.expect("session succeeds"),
            SessionSummary {
                submissions: 2,
                allocations: 1,
                bytes_mapped: BYTES as u64,
            }
        );
        let state = host.executor.state();
        assert_eq!(
            state.v3_submissions,
            vec![first],
            "replay reached the executor"
        );
        assert_eq!(state.v3_executions, 1, "replay executed GPU work");
    }

    #[test]
    fn v3_mixed_descriptor_table_round_trips_through_the_host() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(
            &mut client,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("send register");
        expect_ack(&mut client, true);
        // id 0x21, generation 1, RGBA16Float 2560x1440: the fixture's texture.
        write_frame(
            &mut client,
            KIND_TEXTURE_REG,
            &texture_registration_payload(9),
        )
        .expect("send texture register");
        expect_ack(&mut client, true);

        let table = crate::wire_v3::DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: crate::wire_v3::DescriptorKind::Srv,
            bindings: vec![
                DescriptorTableEntryV3::Buffer(full_binding(BYTES as u64)),
                DescriptorTableEntryV3::Texture(0x21),
            ],
        };
        let submission = SubmissionV3 {
            sequence: 1,
            list: None,
            commands: vec![CommandV3::SetDescriptorTable(table.clone())],
        };
        // The guest encodes the mixed table; the host decodes the same bytes
        // and validates every entry against its live registries.
        assert_eq!(
            roundtrip_v3(&mut client, &submission),
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert_eq!(
            state.v3_submissions,
            vec![submission],
            "the mixed table reached the executor with both entries intact"
        );
        match &state.v3_submissions[0].commands[0] {
            CommandV3::SetDescriptorTable(recorded) => {
                assert_eq!(recorded.bindings, table.bindings);
                assert_eq!(recorded.kind, table.kind);
            }
            other => panic!("expected the descriptor table, got {other:?}"),
        }
    }

    #[test]
    fn v3_submission_publishes_shared_completion_fence() {
        use crate::fence_page::FENCE_PAGE_LEN;

        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let fence_path = dir.0.join(FENCE_PAGE_FILE_NAME);
        // The launcher owns the page file; the host maps it on registration.
        let launcher_page = FencePageMapping::create(&fence_path).expect("create fence page");
        assert_eq!(launcher_page.page().completed(), 0);

        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);
        write_frame(
            &mut client,
            KIND_FENCE_PAGE,
            &(FENCE_PAGE_LEN as u64).to_le_bytes(),
        )
        .expect("send fence page registration");
        expect_ack(&mut client, true);

        // A second, independent mapping of the same file is the guest's view.
        let guest_page = FencePageMapping::open(&fence_path).expect("open shared fence page");
        assert_eq!(
            guest_page.page().epoch(),
            1,
            "registration must start a new session epoch"
        );

        let identity = list_identity(41, 1);
        let first = v3_submission(1, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &first),
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );
        assert_eq!(guest_page.page().completed(), 1);
        assert_eq!(guest_page.page().submitted(), 1);
        assert!(!guest_page.page().now_pending());

        // A replay is refused and must not advance or lower the fence.
        let replay = v3_submission(2, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &replay),
            Reply {
                sequence: 2,
                status: ReplyStatus::AlreadyExecuted,
            }
        );
        assert_eq!(
            guest_page.page().completed(),
            1,
            "replay advanced the completion fence"
        );
        assert_eq!(guest_page.page().submitted(), 1);

        // A validation refusal never reaches the executor and is not published.
        let unknown = v3_submission(
            3,
            None,
            BufferBinding {
                allocation_id: ALLOCATION_ID + 1,
                generation: BASE_GENERATION,
                offset: 0,
                length: BYTES as u64,
            },
        );
        assert_eq!(
            roundtrip_v3(&mut client, &unknown),
            Reply {
                sequence: 3,
                status: ReplyStatus::UnknownHandle,
            }
        );
        assert_eq!(guest_page.page().completed(), 1);
        assert_eq!(guest_page.page().submitted(), 1);

        // A later success publishes its own sequence.
        let later = v3_submission(4, None, full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &later),
            Reply {
                sequence: 4,
                status: ReplyStatus::Ok,
            }
        );
        assert_eq!(guest_page.page().completed(), 4);

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        assert_eq!(host.executor.state().v3_executions, 2);
    }

    #[test]
    fn v3_unknown_allocation_is_refused_and_leaves_identity_retryable() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let identity = list_identity(32, 1);
        let unknown = v3_submission(
            1,
            Some(identity),
            BufferBinding {
                allocation_id: ALLOCATION_ID + 1,
                generation: BASE_GENERATION,
                offset: 0,
                length: BYTES as u64,
            },
        );
        assert_eq!(
            roundtrip_v3(&mut client, &unknown),
            Reply {
                sequence: 1,
                status: ReplyStatus::UnknownHandle,
            }
        );

        // The refused identity was never claimed, so the corrected retry runs.
        let corrected = v3_submission(2, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &corrected),
            Reply {
                sequence: 2,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert_eq!(state.v3_submissions, vec![corrected]);
        assert_eq!(state.v3_executions, 1);
    }

    #[test]
    fn v3_stream_error_keeps_session_open_for_an_exact_retry() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let submission = v3_submission(1, None, full_binding(BYTES as u64));
        let mut frame = v3_envelope(&submission);
        frame[4..6].copy_from_slice(&(PROTOCOL_VERSION_V3 + 1).to_le_bytes());
        write_frame(&mut client, KIND_SUBMISSION_V3, &frame).expect("send bad version");
        let (kind, payload) = read_frame(&mut client)
            .expect("read rejection")
            .expect("rejection frame");
        assert_eq!(kind, KIND_SUBMISSION_V3);
        assert_eq!(
            wire::decode_reply(&payload).expect("decode rejection"),
            Reply {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            }
        );

        // The rejected frame did not advance the session sequence.
        assert_eq!(
            roundtrip_v3(&mut client, &submission),
            Reply {
                sequence: 1,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, _host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
    }

    #[test]
    fn v3_executor_refusal_is_an_explicit_reply_status() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().v3_not_supported = true;
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let submission = v3_submission(1, None, full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &submission),
            Reply {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            },
            "an unsupported v3 command must be a status, never a silent success"
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let state = host.executor.state();
        assert!(state.v3_submissions.is_empty());
        assert_eq!(state.v3_executions, 0);
    }

    /// Every v3 error reply carries the executor's bounded text, including the
    /// 256-byte ceiling and a UTF-8-truncated multi-byte reason.
    #[test]
    fn v3_error_reply_reason_roundtrips_bounded_text() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        let exact = "a".repeat(256);
        let long = "b".repeat(300);
        let unicode = "é".repeat(200);
        executor.state().v3_failure_reasons = vec![
            "target texture 640x360 was refused".to_string(),
            exact,
            long,
            unicode,
        ];
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(
            &mut client,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("send register");
        expect_ack(&mut client, true);

        for (sequence, expected_prefix) in [
            (1u32, "Metal execution failed: internal Metal error: target texture"),
            (2, "Metal execution failed: internal Metal error: aaa"),
            (3, "Metal execution failed: internal Metal error: bbb"),
            (4, "Metal execution failed: internal Metal error: ééé"),
        ] {
            let detail = roundtrip_v3_detail(
                &mut client,
                &v3_submission(sequence, None, full_binding(BYTES as u64)),
            );
            assert_eq!(detail.sequence, sequence, "the reply echoes the sequence");
            assert_eq!(detail.status, ReplyStatus::ExecutionFailed);
            let reason = detail.reason.expect("every error reply carries a reason");
            assert!(
                reason.starts_with(expected_prefix),
                "unexpected reason: {reason}"
            );
            assert!(
                reason.len() <= crate::wire::MAX_REPLY_REASON_BYTES,
                "reason is {} bytes",
                reason.len()
            );
            assert!(reason.is_char_boundary(reason.len()));
            assert!(!reason.contains('\n'), "the reason must stay one line");
        }

        drop(client);
        let (result, _host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
    }

    /// A reason-carrying validation refusal leaves the sequence and ledger
    /// exactly as the status-only reply did: no claim is taken and the same
    /// list identity executes once under the next sequence.
    #[test]
    fn v3_reason_reply_leaves_session_state_unchanged() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(
            &mut client,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("send register");
        expect_ack(&mut client, true);

        let identity = list_identity(77, 1);
        let failed = v3_submission(
            1,
            Some(identity),
            BufferBinding {
                allocation_id: ALLOCATION_ID + 1,
                generation: BASE_GENERATION,
                offset: 0,
                length: BYTES as u64,
            },
        );
        let detail = roundtrip_v3_detail(&mut client, &failed);
        assert_eq!(detail.sequence, 1, "sequence echo is unchanged");
        assert_eq!(detail.status, ReplyStatus::UnknownHandle);
        assert!(
            detail.reason.as_deref().is_some_and(|reason| reason
                .contains("unregistered allocation, texture or sampler")),
            "the host's validation text is carried"
        );

        // The refused submission took no ledger claim; the same list identity
        // executes once under sequence 2.
        let retried = v3_submission(2, Some(identity), full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &retried),
            Reply {
                sequence: 2,
                status: ReplyStatus::Ok,
            }
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("session succeeds");
        let stats = host.session_stats();
        assert_eq!(stats.submissions, 2);
        assert_eq!(stats.refused_submissions, 1);
        assert_eq!(host.executor.state().v3_executions, 1);
        assert_eq!(host.executor.state().v3_submissions, vec![retried]);
    }

    /// A framing rejection names the offending envelope/frame field, and a
    /// sequence mismatch names both the expected and the received sequence.
    #[test]
    fn v3_framing_rejection_reason_names_the_failure() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(
            &mut client,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("send register");
        expect_ack(&mut client, true);

        let submission = v3_submission(1, None, full_binding(BYTES as u64));
        let mut frame = crate::wire_v3::encode_submission_v3(&submission)
            .expect("encode the frame to corrupt");
        // Frame header (20) + list block (12) is the first command's opcode.
        assert!(frame.len() >= 34, "encoded frame is too short");
        frame[32..34].copy_from_slice(&0x00FFu16.to_le_bytes());
        write_frame(
            &mut client,
            KIND_SUBMISSION_V3,
            &v3_envelope_from_frame(1, &frame),
        )
        .expect("send corrupt frame");

        let (kind, payload) = read_frame(&mut client)
            .expect("read rejection")
            .expect("rejection frame");
        assert_eq!(kind, KIND_SUBMISSION_V3);
        let detail = wire::decode_reply_with_reason(&payload).expect("decode rejection");
        assert_eq!(detail.sequence, 1);
        assert_eq!(detail.status, ReplyStatus::ExecutionFailed);
        assert!(
            detail
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("unknown command opcode 255")),
            "unexpected framing reason: {:?}",
            detail.reason
        );

        drop(client);
        let (result, _host) = server.handle.join().expect("server thread");
        result.expect("the rejected frame leaves the session usable");
    }

    #[test]
    fn v3_sequence_mismatch_reply_carries_a_reason() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(
            &mut client,
            KIND_REGISTER,
            &registration(ALLOCATION_ID, BYTES as u64),
        )
        .expect("send register");
        expect_ack(&mut client, true);

        let out_of_order = v3_submission(2, None, full_binding(BYTES as u64));
        let detail = roundtrip_v3_detail(&mut client, &out_of_order);
        assert_eq!(detail.sequence, 2, "the reply echoes the received sequence");
        assert_eq!(detail.status, ReplyStatus::ExecutionFailed);
        assert!(
            detail
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("expected 1, got 2")),
            "unexpected mismatch reason: {:?}",
            detail.reason
        );

        drop(client);
        let (result, _host) = server.handle.join().expect("server thread");
        assert!(matches!(result, Err(HostError::Protocol(_))));
    }

    /// Builds one kind 8 payload for the format-code tests.
    fn texture_registration_payload(format: u8) -> [u8; TEXTURE_REG_PAYLOAD_LEN] {
        let mut payload = [0u8; TEXTURE_REG_PAYLOAD_LEN];
        payload[0..8].copy_from_slice(&0x21u64.to_le_bytes()); // id
        payload[8..12].copy_from_slice(&1u32.to_le_bytes()); // generation
        payload[12] = 1; // 2D
        payload[13] = format;
        payload[14] = 1; // colour attachment
        payload[15] = 1; // sample count
        payload[16..20].copy_from_slice(&2560u32.to_le_bytes());
        payload[20..24].copy_from_slice(&1440u32.to_le_bytes());
        payload[24..28].copy_from_slice(&1u32.to_le_bytes()); // depth
        payload[28..30].copy_from_slice(&1u16.to_le_bytes()); // mip levels
        payload[30..32].copy_from_slice(&1u16.to_le_bytes()); // array layers
        payload
    }

    /// Guest kind 8 code 9 is the title's FP16 present target: the host
    /// decodes it to `TextureFormat::Rgba16Float`, whose Metal pixel format is
    /// `MTLPixelFormatRGBA16Float` (115), and accepts the 2560x1440 descriptor.
    #[test]
    fn host_code_9_is_an_rgba16_float_present_target() {
        assert_eq!(
            texture_format_from_code(9),
            Some(TextureFormat::Rgba16Float)
        );
        assert_eq!(TextureFormat::Rgba16Float.mtl_pixel_format(), 115);
        assert!(!TextureFormat::Rgba16Float.is_depth_stencil());
        assert!(!TextureFormat::Rgba16Float.is_block_compressed());
        assert!(TextureFormat::Rgba16Float.is_color_render_target());

        let request = decode_texture_request(&texture_registration_payload(9))
            .expect("a 2560x1440 RGBA16Float colour target is a valid registration");
        assert_eq!(request.desc.format, TextureFormat::Rgba16Float);
        assert_eq!((request.desc.width, request.desc.height), (2560, 1440));
    }

    /// Guest kind 8 codes 10 and 11 are the integer storage/colour targets:
    /// R32_UINT -> `TextureFormat::R32Uint` (`MTLPixelFormatR32Uint`, 53) and
    /// R32_SINT -> `TextureFormat::R32Sint` (`MTLPixelFormatR32Sint`, 54).
    /// A 1280x720 registration (the Boot #2b target) decodes to a descriptor
    /// the registry accepts, with the storage/colour usage widened so a UAV
    /// write and an RTV are both legal.
    #[test]
    fn host_codes_10_and_11_are_r32_storage_colour_targets() {
        for (host_code, format, mtl) in [
            (10u8, TextureFormat::R32Uint, 53usize),
            (11u8, TextureFormat::R32Sint, 54usize),
        ] {
            assert_eq!(texture_format_from_code(host_code), Some(format));
            assert_eq!(format.mtl_pixel_format(), mtl);
            assert!(!format.is_depth_stencil());
            assert!(!format.is_block_compressed());
            assert!(format.is_color_render_target());

            // Usage 1 is what the guest computes for a UAV target (flags 0x4).
            let mut payload = texture_registration_payload(host_code);
            payload[14] = 1; // colour attachment
            payload[16..20].copy_from_slice(&1280u32.to_le_bytes());
            payload[20..24].copy_from_slice(&720u32.to_le_bytes());
            let request = decode_texture_request(&payload)
                .expect("a 1280x720 R32 target is a valid registration");
            assert_eq!(request.desc.format, format);
            assert_eq!((request.desc.width, request.desc.height), (1280, 720));
            assert!(request.desc.usage.shader_read);
            assert!(request.desc.usage.shader_write);
            assert!(request.desc.usage.render_target);
            assert!(request.desc.validate().is_ok(), "the descriptor is accepted");
        }
    }

    /// A storage-usage registration for R32_UINT is accepted too and also
    /// carries the colour-attachment bit, so the same host texture can back a
    /// UAV and an RTV.
    #[test]
    fn an_r32_uint_storage_usage_is_accepted() {
        let mut payload = texture_registration_payload(10);
        payload[14] = 3; // storage
        payload[16..20].copy_from_slice(&1280u32.to_le_bytes());
        payload[20..24].copy_from_slice(&720u32.to_le_bytes());
        let request =
            decode_texture_request(&payload).expect("a 1280x720 R32_UINT storage target decodes");
        assert_eq!(request.desc.format, TextureFormat::R32Uint);
        assert!(request.desc.usage.shader_write);
        assert!(request.desc.usage.render_target);
        assert!(request.desc.validate().is_ok());
    }

    /// ADR-033: every canonical table row is decodable by its kind 8 wire code,
    /// so the host no longer keeps a second partial format map. Codes the table
    /// does not own stay refused with no default substitution.
    #[test]
    fn host_wire_codes_follow_the_canonical_table() {
        for format in TextureFormat::ALL {
            let code = format.wire_code();
            assert_eq!(
                texture_format_from_code(code),
                Some(format),
                "wire code {code} ({})",
                format.name()
            );
        }
        for code in 0..=u8::MAX {
            let mapped = texture_format_from_code(code);
            assert_eq!(
                mapped.is_some(),
                TextureFormat::from_wire_code(code).is_some(),
                "code {code}"
            );
            if mapped.is_none() {
                assert!(decode_texture_request(&texture_registration_payload(code)).is_none());
            }
        }
    }

    /// Boot #3: the 640x360 UAV+RT target the title stopped on is DXGI 62
    /// (`R8_UINT`, Metal `R8Uint` 13). Its kind 8 wire code decodes, and the
    /// integer widening makes one host texture legal as a UAV write and an RTV.
    #[test]
    fn the_640x360_r8_uint_target_registers() {
        let format = TextureFormat::from_dxgi(62).expect("DXGI 62 is mapped");
        assert_eq!(format, TextureFormat::R8Uint);
        assert_eq!(format.mtl_pixel_format(), 13);
        let code = format.wire_code();
        assert_eq!(texture_format_from_code(code), Some(format));

        let mut payload = texture_registration_payload(code);
        payload[14] = 1; // colour attachment
        payload[16..20].copy_from_slice(&640u32.to_le_bytes());
        payload[20..24].copy_from_slice(&360u32.to_le_bytes());
        let request = decode_texture_request(&payload)
            .expect("a 640x360 R8_UINT UAV+RT target is a valid registration");
        assert_eq!(request.desc.format, format);
        assert_eq!((request.desc.width, request.desc.height), (640, 360));
        assert!(request.desc.usage.shader_read);
        assert!(request.desc.usage.shader_write);
        assert!(request.desc.usage.render_target);
        assert!(request.desc.validate().is_ok(), "the descriptor is accepted");
    }

    /// A GPU `ExecutionFailed` is a per-submission refusal, not a protocol
    /// error: the reason is recorded, the refusal counter advances, and the
    /// same session accepts the next submission.
    #[test]
    fn v3_execution_failure_is_recorded_and_the_session_continues() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let executor = StubExecutor::new(4096);
        executor.state().v3_fail_once = true;
        let server = start(host(&dir, executor));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        let first = v3_submission(1, None, full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &first),
            Reply {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            },
            "a GPU fault is still an ExecutionFailed reply"
        );

        // The session loop must still be open: submission 2 is accepted.
        let second = v3_submission(2, None, full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &second),
            Reply {
                sequence: 2,
                status: ReplyStatus::Ok,
            },
            "a GPU refusal must not end the session"
        );

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        result.expect("the session ends cleanly, not fatally");
        let stats = host.session_stats();
        assert_eq!(stats.submissions, 2);
        assert_eq!(stats.refused_submissions, 1);
        let last_error = stats.last_error.expect("the GPU reason is recorded");
        assert!(
            last_error.contains("synthetic out-of-bounds write"),
            "unexpected reason: {last_error}"
        );
        assert!(last_error.len() <= crate::macos::device::MAX_GPU_REASON_BYTES);
    }

    /// A sequence mismatch is a protocol error: the guest gets a status, the
    /// refusal is recorded, and the session closes non-zero.
    #[test]
    fn v3_sequence_mismatch_ends_the_session_as_a_protocol_error() {
        let dir = TestDir::new();
        dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
        let server = start(host(&dir, StubExecutor::new(4096)));
        let mut client = connect(&server);
        authenticate(&mut client);
        write_frame(&mut client, KIND_REGISTER, &registration(ALLOCATION_ID, BYTES as u64))
            .expect("send register");
        expect_ack(&mut client, true);

        // The host expects sequence 1.
        let out_of_order = v3_submission(2, None, full_binding(BYTES as u64));
        assert_eq!(
            roundtrip_v3(&mut client, &out_of_order),
            Reply {
                sequence: 2,
                status: ReplyStatus::ExecutionFailed,
            }
        );
        assert_eq!(read_frame(&mut client).expect("read close"), None);

        let (result, host) = server.handle.join().expect("server thread");
        assert!(
            matches!(result, Err(HostError::Protocol(_))),
            "a protocol error must end the session"
        );
        let stats = host.session_stats();
        assert_eq!(stats.submissions, 1);
        assert_eq!(stats.refused_submissions, 1);
        let last_error = stats.last_error.expect("the protocol reason is recorded");
        assert!(
            last_error.contains("sequence mismatch"),
            "unexpected reason: {last_error}"
        );
    }

    // ---- ADR-038: host-owned DXIL translation (version 3 frame) ----

    use crate::compiler_pool::PoolConfig;

    /// One fake `victory-irconverter --stdio` worker: it answers `VMRES2` with a
    /// deterministic metallib and a reflection naming the request's entry point.
    fn write_host_worker(dir: &TestDir) -> PathBuf {
        let path = dir.0.join("host-worker.sh");
        let script = r#"#!/bin/sh
set -u
while IFS=' ' read -r magic seq id stage entry_len dxil_len; do
    [ "$magic" = "VMJOB" ] || exit 3
    scratch="${TMPDIR:-/tmp}/victory-host-stage-$$"
    dd bs=1 count="$entry_len" of="$scratch.entry" 2>/dev/null
    dd bs=1 count="$dxil_len" of="$scratch.dxil" 2>/dev/null
    entry=$(cat "$scratch.entry")
    metallib="MTLB:$entry"
    if [ "$entry" = "empty_reflection" ]; then
        reflection=""
    else
        reflection="{\"EntryPoint\":\"$entry\",\"ShaderType\":\"Unknown\",\"TopLevelArgumentBuffer\":[]}"
    fi
    printf 'VMRES2 %s %s 0 %s %s\n' "$seq" "$id" "${#metallib}" "${#reflection}"
    printf '%s' "$metallib"
    printf '%s' "$reflection"
    rm -f "$scratch.entry" "$scratch.dxil"
done
exit 0
"#;
        std::fs::write(&path, script).expect("write host worker");
        let mut permissions = std::fs::metadata(&path).expect("stat worker").permissions();
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).expect("chmod worker");
        path
    }

    fn host_translation_pool(dir: &TestDir) -> Arc<CompilerPool> {
        let program = write_host_worker(dir);
        Arc::new(CompilerPool::new(
            PoolConfig {
                workers: 2,
                queue_depth: 8,
                job_timeout: Duration::from_secs(10),
                cache_dir: None,
                converter_version: "victory-metal-test/host-translate/1".to_string(),
            },
            Some(program),
        ))
    }

    /// The exact share-relative name a stage's blake3 hash produces.
    fn stage_name(dxil: &[u8]) -> String {
        format!("stages/{}.bin", blake3::hash(dxil).to_hex())
    }

    fn write_stage(dir: &TestDir, dxil: &[u8]) -> String {
        let name = stage_name(dxil);
        let path = dir.0.join(&name);
        std::fs::create_dir_all(path.parent().expect("stage parent"))
            .expect("create stages dir");
        std::fs::write(&path, dxil).expect("write stage");
        name
    }

    /// The exact version 3 frame the guest builds.
    fn pipeline_request_v3(
        kind: u8,
        stages: &[(u8, &str, &str)],
        state: &[u8],
        root: Option<&str>,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&PIPELINE_FRAME_MAGIC_V3);
        payload.push(PIPELINE_FRAME_VERSION_V3);
        payload.push(kind);
        payload.push(stages.len() as u8);
        payload.push(0);
        payload.extend_from_slice(&(state.len() as u32).to_le_bytes());
        let root = root.unwrap_or_default();
        payload.extend_from_slice(&(root.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        for (stage, file, entry) in stages {
            payload.push(*stage);
            payload.extend_from_slice(&(file.len() as u16).to_le_bytes());
            payload.extend_from_slice(file.as_bytes());
            payload.extend_from_slice(&(entry.len() as u16).to_le_bytes());
            payload.extend_from_slice(entry.as_bytes());
        }
        payload.extend_from_slice(state);
        payload.extend_from_slice(root.as_bytes());
        payload
    }

    /// Sends one kind 4 frame and decodes a version 3 reply (status, id, reason).
    fn send_v3_pipeline(
        client: &mut TcpStream,
        payload: &[u8],
    ) -> (u8, Option<u64>, Option<String>) {
        write_frame(client, KIND_PIPELINE, payload).expect("send v3 pipeline");
        let (kind, reply) = read_frame(client)
            .expect("read v3 pipeline reply")
            .expect("v3 pipeline reply frame");
        assert_eq!(kind, KIND_PIPELINE_REPLY);
        match reply.first().copied() {
            Some(PIPELINE_REPLY_OK) => {
                assert_eq!(reply.len(), PIPELINE_REPLY_OK_LEN);
                let id = u64::from_le_bytes(reply[1..].try_into().expect("eight bytes"));
                (PIPELINE_REPLY_OK, Some(id), None)
            }
            Some(PIPELINE_REPLY_REJECTED) => {
                if reply.len() == 1 {
                    return (PIPELINE_REPLY_REJECTED, None, None);
                }
                let reason_len =
                    usize::from(u16::from_le_bytes([reply[1], reply[2]]));
                assert_eq!(reply.len(), 3 + reason_len);
                let reason = String::from_utf8(reply[3..].to_vec()).expect("utf8 reason");
                assert!(reason.len() <= MAX_PIPELINE_REJECTION_REASON);
                (PIPELINE_REPLY_REJECTED, None, Some(reason))
            }
            other => panic!("unexpected v3 pipeline reply: {other:?} / {reply:?}"),
        }
    }

    #[test]
    fn version_1_and_version_2_pipeline_frames_still_decode() {
        let v1 = pipeline_request(7, "vm_entry");
        assert!(matches!(
            decode_pipeline_request(&v1),
            Some(PipelineRequest::V1 { ticket: 7, .. })
        ));
        let v2 = pipeline_request_v2(
            PIPELINE_KIND_GRAPHICS,
            &[(PIPELINE_STAGE_VERTEX, 1, "vs_main"), (PIPELINE_STAGE_PIXEL, 2, "ps_main")],
            &graphics_state(),
            &abi_container(&[(PIPELINE_STAGE_VERTEX, VERTEX_REFLECTION)]),
            b"{}",
        );
        assert!(matches!(
            decode_pipeline_request(&v2),
            Some(PipelineRequest::V2(_))
        ));
    }

    #[test]
    fn a_version_3_frame_host_translates_and_registers_a_pipeline() {
        let dir = TestDir::new();
        let vertex = b"vertex-dxil-for-host-translation";
        let pixel = b"pixel-dxil-for-host-translation";
        let vertex_name = write_stage(&dir, vertex);
        let pixel_name = write_stage(&dir, pixel);
        let pool = host_translation_pool(&dir);
        let server = start(
            host(&dir, StubExecutor::new(4096)).with_compiler_pool(pool),
        );
        let mut client = connect(&server);
        authenticate(&mut client);

        let request = pipeline_request_v3(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, vertex_name.as_str(), "vs_main"),
                (PIPELINE_STAGE_PIXEL, pixel_name.as_str(), "ps_main"),
            ],
            &graphics_state(),
            None,
        );
        let (status, id, reason) = send_v3_pipeline(&mut client, &request);
        assert_eq!(status, PIPELINE_REPLY_OK, "reason: {reason:?}");
        let id = id.expect("registered pipeline id");
        assert_ne!(id, 0, "a successful version 3 registration returns a real id");
        assert!(reason.is_none());

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert!(result.is_ok(), "the session must stay healthy: {result:?}");
        let state = host.executor.state();
        assert_eq!(state.graphics_pipelines.len(), 1);
        let (_, stages, _, abi) = &state.graphics_pipelines[0];
        assert_eq!(stages[0].0, PIPELINE_STAGE_VERTEX);
        assert_eq!(stages[0].1, b"MTLB:vs_main");
        assert_eq!(stages[1].0, PIPELINE_STAGE_PIXEL);
        assert_eq!(stages[1].1, b"MTLB:ps_main");
        assert!(abi.vertex.is_some() && abi.fragment.is_some());
        drop(state);

        let stats = host.session_stats();
        assert_eq!(stats.host_pipeline_translations_ok, 1);
        assert_eq!(stats.host_pipeline_translations_refused, 0);
        assert_eq!(stats.host_translated_stages[0], 1, "vertex stage counted");
        assert_eq!(stats.host_translated_stages[1], 1, "pixel stage counted");
        assert_eq!(stats.pool_status, "ready");
    }

    #[test]
    fn a_malformed_version_3_frame_is_rejected_with_no_gpu_work() {
        let dir = TestDir::new();
        let pool = host_translation_pool(&dir);
        let server = start(host(&dir, StubExecutor::new(4096)).with_compiler_pool(pool));
        let mut client = connect(&server);
        authenticate(&mut client);

        // Stage name is not the 75-byte share-relative grammar.
        let mut request = pipeline_request_v3(
            PIPELINE_KIND_GRAPHICS,
            &[
                (PIPELINE_STAGE_VERTEX, "stages/not-a-hash.bin", "vs_main"),
                (PIPELINE_STAGE_PIXEL, "stages/also-not-a-hash.bin", "ps_main"),
            ],
            &graphics_state(),
            None,
        );
        // A trailing byte makes even a decodable header malformed.
        request.push(0x7f);
        let (status, id, reason) = send_v3_pipeline(&mut client, &request);
        assert_eq!(status, PIPELINE_REPLY_REJECTED);
        assert!(id.is_none());
        assert!(reason.is_some());

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert!(result.is_ok());
        assert!(host.executor.state().graphics_pipelines.is_empty());
        assert_eq!(host.session_stats().host_pipeline_translations_ok, 0);
        assert_eq!(host.session_stats().host_pipeline_translations_refused, 1);
    }

    #[test]
    fn version_3_stage_names_that_escape_the_share_dir_are_rejected() {
        let dir = TestDir::new();
        let pool = host_translation_pool(&dir);
        let server = start(host(&dir, StubExecutor::new(4096)).with_compiler_pool(pool));
        let mut client = connect(&server);
        authenticate(&mut client);

        let hash = "a".repeat(64);
        // Absolute path, parent traversal and backslash all fail the grammar
        // before any file is opened.
        for bad in [
            format!("/stages/{hash}.bin"),
            format!("stages/../{hash}.bin"),
            format!("stages\\{hash}.bin"),
        ] {
            let request = pipeline_request_v3(
                PIPELINE_KIND_COMPUTE,
                &[(PIPELINE_STAGE_COMPUTE, bad.as_str(), "main")],
                &[],
                None,
            );
            let (status, _, _) = send_v3_pipeline(&mut client, &request);
            assert_eq!(status, PIPELINE_REPLY_REJECTED, "accepted {bad}");
        }

        // A syntactically valid name whose final component is a symlink is
        // rejected by the O_NOFOLLOW open even though the target is a file.
        let outside = dir.0.join("outside.dxil");
        std::fs::write(&outside, b"dxil-outside").expect("write outside");
        let name = format!("stages/{hash}.bin");
        std::fs::create_dir_all(dir.0.join("stages")).expect("stages dir");
        let symlink = dir.0.join(&name);
        std::os::unix::fs::symlink(&outside, &symlink).expect("create symlink");
        let request = pipeline_request_v3(
            PIPELINE_KIND_COMPUTE,
            &[(PIPELINE_STAGE_COMPUTE, name.as_str(), "main")],
            &[],
            None,
        );
        let (status, _, reason) = send_v3_pipeline(&mut client, &request);
        assert_eq!(status, PIPELINE_REPLY_REJECTED);
        assert!(reason.is_some());

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert!(result.is_ok());
        assert!(host.executor.state().compute_pipelines.is_empty());
        assert_eq!(host.session_stats().host_pipeline_translations_ok, 0);
    }

    #[test]
    fn an_empty_converter_reflection_is_a_bounded_host_refusal() {
        let dir = TestDir::new();
        // The fake worker answers with an empty reflection for this entry, which
        // `compile_with_reflection` refuses as a protocol error.
        let name = write_stage(&dir, b"compute-dxil-empty-reflection");
        let pool = host_translation_pool(&dir);
        let server = start(host(&dir, StubExecutor::new(4096)).with_compiler_pool(pool));
        let mut client = connect(&server);
        authenticate(&mut client);

        let request = pipeline_request_v3(
            PIPELINE_KIND_COMPUTE,
            &[(PIPELINE_STAGE_COMPUTE, name.as_str(), "empty_reflection")],
            &[],
            None,
        );
        let (status, id, reason) = send_v3_pipeline(&mut client, &request);
        // The pool refuses an empty reflection before the host can derive an
        // ABI; the host answers a bounded non-proprietary reason either way.
        assert_eq!(status, PIPELINE_REPLY_REJECTED);
        assert!(id.is_none());
        let reason = reason.expect("bounded refusal reason");
        assert!(reason.len() <= MAX_PIPELINE_REJECTION_REASON);
        assert!(!reason.contains('/'), "reason must not carry a path: {reason}");
        assert!(!reason.contains("stages"), "reason must not name a file: {reason}");

        drop(client);
        let (result, host) = server.handle.join().expect("server thread");
        assert!(result.is_ok());
        assert!(host.executor.state().compute_pipelines.is_empty());
    }
}
