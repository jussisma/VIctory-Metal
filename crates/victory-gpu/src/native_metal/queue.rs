//! Native queue submission and present under `VICTORY_BACKEND=metal`.
//!
//! Reserved by the orchestrator so the queue/present task owns this file
//! without touching `native_metal/mod.rs`. The contract (ADR-010):
//!
//! * `ExecuteCommandLists` under `VICTORY_BACKEND=metal` takes the Wire v3
//!   submissions a closed title list produced and carries them over IPC in
//!   recorded order;
//! * `Present` hands the produced surface to the host `NativeSwapchain`
//!   (IOSurface) instead of the D3D12 swapchain;
//! * neither path falls back to D3D12/D3DMetal after native submission starts:
//!   an unsupported opcode is an explicit error.
//!
//! # Shape
//!
//! [`NativeQueue`] owns an ordered outbound queue of encoded v3 submissions.
//! It never merges submissions ([`enqueue`](NativeQueue::enqueue) encodes each
//! one with the canonical [`encode_submission_v3`] codec and appends it), and
//! it refuses an empty batch with [`NativeQueueError::Empty`]. It is also the
//! single place a submission sequence is assigned: a frame that carries
//! commands gets the next session-wide sequence exactly once, here, so an
//! empty or dropped stream never leaves a hole in the host's ledger.
//! [`NativeQueue::submit`] sends every queued frame exactly once, in order,
//! and returns the host's reply status; a replayed list's
//! [`ReplyStatus::AlreadyExecuted`] is surfaced, never re-sent.
//!
//! [`V3Sink`] is the transport seam. [`NativeSink`] starts
//! [`DisconnectedSink`] and becomes [`Live`](NativeSink::Live) on the first
//! submission once `VICTORY_METAL_ENDPOINT` and `VICTORY_METAL_SECRET` are
//! configured ([`WireV3Client`]). With no endpoint the missing transport stays
//! an explicit `E_FAIL` instead of a silent D3D12/D3DMetal fallback.
//!
//! The present half is [`NativePresenter`], which wraps a [`NativeSwapchain`]
//! and the host [`SurfacePresenter`]. The queue entry points look it up with
//! [`present_registered`]; nothing here opens a window or a device (the live
//! sink is the one socket, and only when it is configured).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use victory_metal::wire::{ListIdentity, ReplyStatus, WireError};
use victory_metal::wire_v3::{SubmissionV3, encode_submission_v3};

use crate::backend::GpuBackend;
use crate::native_metal::client::{LiveConfig, WireV3Client};
#[cfg(feature = "diagnostics")]
use crate::native_metal::present::guard_native_only;
use crate::native_metal::present::{
    InFlightFrame, NativeSwapchain, PresentError, PresentedSurface, SurfaceFormat, SurfacePresenter,
    present_surface,
};
use crate::native_metal::producer::{self, ProducerError};

/// `E_FAIL`: the refusal code for everything that is not an unsupported
/// operation or an invalid argument.
pub const E_FAIL: i32 = 0x8000_4005_u32 as i32;

/// Why a native queue submission or present was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeQueueError {
    /// No submission carried the list identity the queue was asked to run.
    Empty(&'static str),
    /// The IPC channel is not connected or refused the batch.
    Transport(String),
    /// The host rejected the batch; the message is the host's own.
    Refused(String),
    /// A present was asked for a surface the native swapchain does not own.
    NoSurface(u64),
    /// No native swapchain is installed for the present path.
    NoSwapchain,
    /// The canonical v3 codec refused a submission before it was carried.
    Encode(WireError),
    /// The producer refused a call; the queue surfaces its failure HRESULT.
    Producer(ProducerError),
    /// The native swapchain or its host presenter refused a frame.
    Present(PresentError),
}

impl std::fmt::Display for NativeQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty(why) => write!(f, "native queue submission empty: {why}"),
            Self::Transport(why) => write!(f, "native queue transport: {why}"),
            Self::Refused(why) => write!(f, "native queue refused: {why}"),
            Self::NoSurface(id) => write!(f, "native swapchain does not own surface {id}"),
            Self::NoSwapchain => write!(f, "no native swapchain is installed for present"),
            Self::Encode(why) => write!(f, "native queue encode: {why}"),
            Self::Producer(why) => write!(f, "native producer refused: {why}"),
            Self::Present(why) => write!(f, "native present refused: {why}"),
        }
    }
}

impl std::error::Error for NativeQueueError {}

impl From<ProducerError> for NativeQueueError {
    fn from(error: ProducerError) -> Self {
        Self::Producer(error)
    }
}

impl From<PresentError> for NativeQueueError {
    fn from(error: PresentError) -> Self {
        Self::Present(error)
    }
}

impl NativeQueueError {
    /// The HRESULT the guest entry point returns to the title.
    ///
    /// An unsupported producer operation keeps its `E_NOTIMPL`; an empty or
    /// malformed submission is `E_INVALIDARG`; everything else (transport,
    /// host refusal, present) is `E_FAIL`. There is no success fallback: the
    /// title sees the failure instead of a D3D12/D3DMetal path.
    pub const fn hresult(&self) -> crate::object::HRESULT {
        match self {
            Self::Empty(_) => crate::object::E_INVALIDARG,
            Self::Producer(error) => error.hresult(),
            Self::Encode(WireError::ZeroSequence | WireError::UnsupportedVersion(_)) => {
                crate::object::E_INVALIDARG
            }
            _ => E_FAIL,
        }
    }
}

/// One encoded submission waiting for the transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutboundFrame {
    /// Session sequence assigned when the frame was enqueued.
    pub sequence: u32,
    /// The list that produced it, when the submission names one.
    pub list: Option<ListIdentity>,
    /// Canonical v3 encoded bytes, handed to the sink unchanged.
    pub bytes: Vec<u8>,
}

/// Where the outbound submissions go.
///
/// This crate opens no socket (the `WIRE-V3-HOST-BRIDGE` task owns the host
/// side and its transport): a sink carries one already-encoded v3 frame and
/// answers with the host reply status.
pub trait V3Sink {
    /// Carries one encoded version 3 submission.
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError>;
}

/// The sink in place until the IPC bridge lands: every send is an explicit
/// transport error, never a silent fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct DisconnectedSink;

impl V3Sink for DisconnectedSink {
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                bytes = bytes.len(),
                "VICTORY_BACKEND=metal: no Wire v3 host bridge is connected; submissions fail \
                 explicitly (no D3D12/D3DMetal fallback)"
            );
        });
        Err(NativeQueueError::Transport(
            "the Wire v3 host bridge is not connected".to_string(),
        ))
    }
}

/// The session's transport: the explicit stub, or a live Wire v3 client.
///
/// The queue starts [`Disconnected`](Self::Disconnected) so a process without
/// a configured host never opens a socket; [`native_queue_install_client`]
/// (and the lazy connect on the first submission) replaces it with a live
/// [`WireV3Client`]. There is no third state and no other backend.
#[derive(Debug)]
pub enum NativeSink {
    /// No host bridge is connected: every send is an explicit `E_FAIL`.
    Disconnected(DisconnectedSink),
    /// One authenticated Wire v3 session with the ARM64 host.
    Live(WireV3Client),
}

impl NativeSink {
    /// Whether a live host session is installed.
    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live(_))
    }

    /// The live client, when one is installed.
    pub fn client_mut(&mut self) -> Option<&mut WireV3Client> {
        match self {
            Self::Live(client) => Some(client),
            Self::Disconnected(_) => None,
        }
    }

    /// The live client, when one is installed.
    pub fn client(&self) -> Option<&WireV3Client> {
        match self {
            Self::Live(client) => Some(client),
            Self::Disconnected(_) => None,
        }
    }

    /// The host's completed fence, one atomic load when a page is mapped.
    pub fn client_completed_fence(&self) -> Option<u64> {
        self.client().and_then(WireV3Client::completed_fence)
    }
}

impl V3Sink for NativeSink {
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
        match self {
            Self::Disconnected(sink) => sink.send(bytes),
            Self::Live(client) => client.send(bytes),
        }
    }
}

/// The ordered outbound queue for one guest session.
///
/// Submissions keep the order a closed list produced them in; nothing is
/// merged, reordered or dropped on success. A transport failure leaves the
/// unsent frames at the front of the queue (the failure is reported, not
/// hidden), while a host reply status is always consumed exactly once.
#[derive(Debug)]
pub struct NativeQueue<S: V3Sink> {
    sink: S,
    outbound: VecDeque<OutboundFrame>,
    sent: u64,
    highest_sequence: u32,
}

impl<S: V3Sink> NativeQueue<S> {
    /// A queue with an empty outbound batch, usable in a `static`.
    pub const fn new(sink: S) -> Self {
        Self {
            sink,
            outbound: VecDeque::new(),
            sent: 0,
            highest_sequence: 0,
        }
    }

    /// The transport.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// The transport, mutably.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Encoded submissions waiting for the transport.
    pub fn pending(&self) -> usize {
        self.outbound.len()
    }

    /// Whether nothing is waiting for the transport.
    pub fn is_empty(&self) -> bool {
        self.outbound.is_empty()
    }

    /// Frames handed to the sink and answered so far.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    /// The highest sequence assigned to a frame of this session.
    ///
    /// Sequences are handed out in enqueue order and never reused, so after a
    /// successful [`flush`](Self::flush) this is the sequence of the last
    /// frame handed to the host; the fence path records it.
    pub fn highest_sequence(&self) -> u32 {
        self.highest_sequence
    }

    /// Assigns a sequence and encodes `submissions`, in recorded order.
    ///
    /// This is the transport choke point and the only place a frame takes a
    /// session sequence: a submission that carries no commands is skipped (it
    /// is never handed to the host and consumes nothing), and every other one
    /// gets the next sequence exactly once, through
    /// [`producer::ensure_sequence`]. A submission that already carries a
    /// sequence (a frame built by a caller, or a retried one) keeps it, so a
    /// retry never renumbers. Nothing is coalesced: one [`SubmissionV3`]
    /// becomes one outbound frame, encoded with the canonical v3 codec. An
    /// empty batch is refused with [`NativeQueueError::Empty`]. Nothing is
    /// queued if any frame fails to encode, so a refused batch never leaves a
    /// partial prefix behind.
    pub fn enqueue(&mut self, submissions: &[SubmissionV3]) -> Result<usize, NativeQueueError> {
        if submissions.is_empty() {
            return Err(NativeQueueError::Empty(
                "ExecuteCommandLists carried no submission",
            ));
        }
        let mut encoded = Vec::with_capacity(submissions.len());
        for submission in submissions {
            if submission.commands.is_empty() {
                // An empty stream is not a frame: it is never handed to the
                // host and must not consume a sequence the host will never
                // see, or every later frame is a protocol violation.
                continue;
            }
            // A submitted stream leaves the producer unassigned; assign it
            // here, exactly once, in recorded order. A frame that already
            // carries a sequence keeps it.
            let assigned = if submission.sequence == producer::UNASSIGNED_SEQUENCE {
                let mut copy = submission.clone();
                producer::ensure_sequence(&mut copy)?;
                Some(copy)
            } else {
                None
            };
            let submission = assigned.as_ref().unwrap_or(submission);
            let bytes = encode_submission_v3(submission).map_err(NativeQueueError::Encode)?;
            self.highest_sequence = self.highest_sequence.max(submission.sequence);
            encoded.push(OutboundFrame {
                sequence: submission.sequence,
                list: submission.list,
                bytes,
            });
        }
        let count = encoded.len();
        self.outbound.extend(encoded);
        Ok(count)
    }

    /// Sends every queued frame in order, exactly once.
    ///
    /// Returns one host status per frame, in send order. A transport error
    /// stops the pump and leaves the failed frame (and everything behind it)
    /// queued; a host status, including [`ReplyStatus::AlreadyExecuted`], is
    /// consumed and surfaced, never retried.
    pub fn flush(&mut self) -> Result<Vec<ReplyStatus>, NativeQueueError> {
        let mut statuses = Vec::with_capacity(self.outbound.len());
        while let Some(frame) = self.outbound.pop_front() {
            match self.sink.send(&frame.bytes) {
                Ok(status) => {
                    self.sent += 1;
                    statuses.push(status);
                }
                Err(error) => {
                    self.outbound.push_front(frame);
                    return Err(error);
                }
            }
        }
        Ok(statuses)
    }

    /// Queues one closed list's submissions and sends them in recorded order.
    ///
    /// Returns the batch's host reply: the first status that is not
    /// [`ReplyStatus::Ok`], or `Ok` when every frame succeeded. A replayed
    /// list's `AlreadyExecuted` is therefore surfaced; the frame is not sent
    /// a second time.
    pub fn submit(
        &mut self,
        submissions: &[SubmissionV3],
    ) -> Result<ReplyStatus, NativeQueueError> {
        self.submit_measured(submissions)
            .map(|(status, _timing)| status)
    }

    /// [`Self::submit`], also reporting where the batch spent its time.
    ///
    /// The reply rule is identical (the first status that is not `Ok`, else
    /// `Ok`): encoding and the transport pump are timed separately so the
    /// frame telemetry can attribute them, and nothing about ordering,
    /// coalescing or the retry rule changes.
    pub fn submit_measured(
        &mut self,
        submissions: &[SubmissionV3],
    ) -> Result<(ReplyStatus, SubmitTiming), NativeQueueError> {
        let started = Instant::now();
        self.enqueue(submissions)?;
        let encode_us = micros(started.elapsed());

        let started = Instant::now();
        let statuses = self.flush()?;
        let submit_us = micros(started.elapsed());

        let status = statuses
            .into_iter()
            .find(|status| *status != ReplyStatus::Ok)
            .unwrap_or(ReplyStatus::Ok);
        Ok((
            status,
            SubmitTiming {
                encode_us,
                submit_us,
            },
        ))
    }

    /// Installs `sink`, dropping every unsent frame and the sent counter.
    ///
    /// This is session teardown and test isolation, never the submission path:
    /// the live path reports a refused batch and leaves its frames queued
    /// ([`Self::flush`]).
    pub fn reset(&mut self, sink: S) {
        self.sink = sink;
        self.outbound.clear();
        self.sent = 0;
    }
}

/// Where one submission batch spent its guest-side time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubmitTiming {
    /// Canonical v3 encoding of the batch, in microseconds.
    pub encode_us: u64,
    /// Handing every frame to the sink, in microseconds.
    pub submit_us: u64,
}

fn micros(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
}

/// What one `ExecuteCommandLists` batch did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDispatch {
    /// The backend is D3D12: the host path runs and this queue is untouched.
    Host,
    /// The batch went over the native queue in recorded order.
    Native {
        /// Host reply status for the batch.
        status: ReplyStatus,
        /// Submissions carried by the batch.
        submissions: usize,
    },
}

/// Routes one `ExecuteCommandLists` batch by backend.
///
/// D3D12 leaves the native queue completely untouched (not even a sink call);
/// Metal encodes and carries the batch in recorded order. This is the only
/// place the batch decides between the two paths, and it never falls back.
pub fn dispatch<S: V3Sink>(
    backend: GpuBackend,
    queue: &mut NativeQueue<S>,
    submissions: &[SubmissionV3],
) -> Result<NativeDispatch, NativeQueueError> {
    dispatch_measured(backend, queue, submissions).map(|(dispatch, _timing)| dispatch)
}

/// [`dispatch`], also reporting the batch's timing when it went native.
///
/// The D3D12 branch returns `None` and touches neither the queue nor the
/// telemetry; only the Metal branch reports a [`SubmitTiming`], which is what
/// the frame telemetry records.
pub fn dispatch_measured<S: V3Sink>(
    backend: GpuBackend,
    queue: &mut NativeQueue<S>,
    submissions: &[SubmissionV3],
) -> Result<(NativeDispatch, Option<SubmitTiming>), NativeQueueError> {
    match backend {
        GpuBackend::D3d12 => Ok((NativeDispatch::Host, None)),
        GpuBackend::Metal => {
            let (status, timing) = queue.submit_measured(submissions)?;
            Ok((
                NativeDispatch::Native {
                    status,
                    submissions: submissions.len(),
                },
                Some(timing),
            ))
        }
    }
}

/// The last submission sequence handed to the host in this session.
static LAST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A fence signal waiting for the host to complete `sequence`.
#[derive(Debug, Clone, Copy)]
struct PendingSignal {
    fence: usize,
    value: u64,
    sequence: u64,
}

/// Fence signals whose host submission has not completed yet.
#[cfg(any(windows, test, target_os = "macos"))]
fn pending_signals() -> std::sync::MutexGuard<'static, Vec<PendingSignal>> {
    static PENDING: std::sync::Mutex<Vec<PendingSignal>> = std::sync::Mutex::new(Vec::new());
    PENDING.lock().unwrap_or_else(|error| error.into_inner())
}

/// Records a queue `Signal` and completes it as soon as the host has run the
/// submission it follows.
///
/// With no shared page the kind 5 transport is synchronous (the reply is
/// written after the host executed the batch), so the signal completes now.
/// With a page the value moves only once `completed_fence` has reached the
/// sequence submitted before the signal: this is the zero-IPC path ADR-019
/// asks for, and `GetCompletedValue` never waits.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn signal_native(fence: crate::object::This, value: u64) {
    let sequence = LAST_SEQUENCE.load(Ordering::Acquire);
    pending_signals().push(PendingSignal {
        fence: fence as usize,
        value,
        sequence,
    });
    poll_native_fence();
}

/// Advances every fence whose host submission has completed.
///
/// One atomic load of the shared page per call; no blocking wait and no IPC.
/// Without a registered page every pending signal is completed immediately
/// because the transport reply already followed host execution.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn poll_native_fence() {
    let completion = {
        let queue = native_queue();
        queue.sink().client_completed_fence()
    };
    let mut pending = pending_signals();
    match completion {
        Some(completed) => {
            pending.retain(|signal| {
                if signal.sequence <= completed {
                    crate::fence::set(signal.fence as crate::object::This, signal.value);
                    false
                } else {
                    true
                }
            });
        }
        None => {
            for signal in pending.drain(..) {
                crate::fence::set(signal.fence as crate::object::This, signal.value);
            }
        }
    }
}

/// The session's native queue. It starts disconnected and connects lazily.
#[cfg(any(windows, test, target_os = "macos"))]
fn native_queue() -> std::sync::MutexGuard<'static, NativeQueue<NativeSink>> {
    static QUEUE: std::sync::Mutex<NativeQueue<NativeSink>> =
        std::sync::Mutex::new(NativeQueue::new(NativeSink::Disconnected(DisconnectedSink)));
    QUEUE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the session queue holds a live Wire v3 host session.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn native_queue_is_live() -> bool {
    native_queue().sink().is_live()
}

/// Installs an authenticated client as the session queue's transport.
///
/// Replaces whatever the queue held (the lazy connect and the daemon task both
/// use it); frames already queued keep their order and go out over `client`.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn native_queue_install_client(client: WireV3Client) {
    *native_queue().sink_mut() = NativeSink::Live(client);
}

/// Drops the session queue's sink and every unsent frame (teardown, tests).
#[cfg(any(windows, test, target_os = "macos"))]
pub fn native_queue_reset() {
    native_queue().reset(NativeSink::Disconnected(DisconnectedSink));
    TEXTURE_FORMATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// Connects the session queue on first use, if the environment names a host.
///
/// Returns whether the queue is live. A configuration that does not parse is
/// logged without the secret and leaves the queue disconnected, so submissions
/// keep failing explicitly instead of silently succeeding.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn native_queue_connect_from_env() -> bool {
    match ensure_live(LiveConfig::from_env) {
        Ok(live) => live,
        Err(error) => {
            tracing::warn!(
                %error,
                "VICTORY_BACKEND=metal: the Wire v3 host endpoint is configured but unusable"
            );
            false
        }
    }
}

/// Registers one host texture for a title resource identity.
///
/// The codes are the stable kinds documented in `ipc_host::register_texture`
/// (dimension, format, usage). A configured endpoint connects on first use; a
/// refusal is an explicit error, never a fallback.
#[cfg(any(windows, test, target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn register_texture(
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
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    if queue
        .sink()
        .client()
        .is_some_and(|client| client.texture_registered(id))
    {
        // Several title resources share one memory (placed views and
        // aliases); the host registers the handle once.
        note_texture_format(id, format);
        return Ok(true);
    }
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    let accepted = client.register_texture(
        id,
        generation,
        dimension,
        format,
        usage,
        sample_count,
        width,
        height,
        depth,
        mip_levels,
        array_layers,
    )?;
    if accepted {
        // Remember the host format so a headless surface installed from the
        // presented target keeps its own format (FP16 stays FP16).
        note_texture_format(id, format);
    }
    Ok(accepted)
}

/// Host format codes of registered textures, keyed by handle.
///
/// A title that names no surface in the environment presents a target texture
/// instead (see the guest `Present` entry point); the surface installed for it
/// must carry that texture's format, never a BGRA8 replacement.
#[cfg(any(windows, test, target_os = "macos"))]
static TEXTURE_FORMATS: std::sync::Mutex<std::collections::BTreeMap<u64, u8>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(any(windows, test, target_os = "macos"))]
fn note_texture_format(id: u64, format: u8) {
    TEXTURE_FORMATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, format);
}

#[cfg(any(windows, test, target_os = "macos"))]
fn texture_format_code(id: u64) -> Option<u8> {
    TEXTURE_FORMATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&id)
        .copied()
}

/// Registers a texture handle as a present surface with the host.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_surface(surface_id: u64, texture_id: u64) -> Result<bool, NativeQueueError> {
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    client.register_surface(surface_id, texture_id)
}

/// Registers one sampler content with the host and returns its stable handle.
///
/// The handle is derived from the descriptor content
/// ([`crate::native_metal::sampler::sampler_handle`]), so the same descriptor
/// returns the same handle on every frame and after a reconnect; only the first
/// registration sends a frame. A content map that is full is an explicit error,
/// never an eviction. Registration failures are explicit: there is no default
/// sampler fallback.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_sampler_state(
    desc: &crate::native_metal::sampler::SamplerStateDesc,
) -> Result<u64, NativeQueueError> {
    let handle = crate::native_metal::sampler::sampler_handle(desc)
        .map_err(|error| NativeQueueError::Refused(error.to_string()))?;
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    if client.sampler_registered(handle) {
        return Ok(handle);
    }
    if client.register_sampler(handle, desc)? {
        Ok(handle)
    } else {
        Err(NativeQueueError::Refused(format!(
            "the host refused sampler handle 0x{handle:016x}"
        )))
    }
}

/// Whether a live session registered this sampler handle.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn sampler_registered(handle: u64) -> bool {
    match LiveConfig::from_env() {
        Some(Ok(_)) => native_queue()
            .sink()
            .client()
            .is_some_and(|client| client.sampler_registered(handle)),
        _ => false,
    }
}

/// Whether a live session registered this texture handle.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn texture_registered(handle: u64) -> bool {
    match LiveConfig::from_env() {
        Some(Ok(_)) => native_queue()
            .sink()
            .client()
            .is_some_and(|client| client.texture_registered(handle)),
        _ => false,
    }
}

/// Registers one guest host-visible allocation with the host (kind 2).
///
/// `byte_len` is the whole share file the allocation id names (the guest reads
/// it from `<share dir>/{id:016x}.bin`), never a page window or a resource
/// length. A configured endpoint connects on first use; an id the session
/// already registered is `Ok(true)` without a frame (aliases and placed
/// resources share the first descriptor), and a refused registration is an
/// explicit error the caller turns into a structured HRESULT.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_buffer(allocation_id: u64, byte_len: u64) -> Result<bool, NativeQueueError> {
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    client.register_allocation(allocation_id, byte_len)
}

/// Whether a live session registered `allocation_id` with the host.
///
/// `false` means the range may not be named in a submission yet: the host
/// validates every binding against its kind 2 registry. No session is `false`.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn allocation_registered(allocation_id: u64) -> bool {
    match LiveConfig::from_env() {
        Some(Ok(_)) => native_queue()
            .sink()
            .client()
            .is_some_and(|client| client.allocation_registered(allocation_id)),
        _ => false,
    }
}

/// Makes sure the host has `allocation_id` registered, registering it when it
/// is new (idempotent).
///
/// The id comes from the `GuestRangeResolver` seam and the length from the
/// share file that id names; nothing is fabricated. Allocations that already
/// exist at or below the requested id are registered first, ascending, because
/// the host refuses (and closes the session on) a kind 2 id that does not
/// increase.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn ensure_allocation(allocation_id: u64) -> Result<bool, NativeQueueError> {
    if allocation_id == 0 {
        return Err(NativeQueueError::Refused(
            "a buffer registration needs a nonzero allocation id".to_string(),
        ));
    }
    if allocation_registered(allocation_id) {
        return Ok(true);
    }
    let pending = crate::native_metal::pso::shares_up_to(allocation_id)
        .map_err(NativeQueueError::Refused)?;
    if !pending.iter().any(|(id, _)| *id == allocation_id) {
        return Err(NativeQueueError::Refused(format!(
            "allocation {allocation_id} has no share file in the session's share directory"
        )));
    }
    for (id, byte_len) in pending {
        register_buffer(id, byte_len)?;
    }
    Ok(true)
}

/// Registers a translated metallib as a native pipeline (kind 4/129).
///
/// The bytes are written to `<share dir>/{ticket:016x}.metallib` (temporary
/// name, then rename, so the host never reads a half-written library) and the
/// ticket plus entry name go over the wire; the host answers with its native
/// pipeline id. A configured endpoint connects on first use. Nothing here
/// reads a resource directory or guesses a ticket: the ticket is the live
/// session's own next one, and the write path is the launcher's share
/// directory.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_pipeline(metallib: &[u8], entry: &str) -> Result<u64, NativeQueueError> {
    if metallib.is_empty() {
        return Err(NativeQueueError::Refused(
            "pipeline registration needs a nonempty metallib".to_string(),
        ));
    }
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    let ticket = client.next_pipeline_ticket();
    let path = crate::native_metal::pso::metallib_path(ticket).ok_or_else(|| {
        NativeQueueError::Refused(
            "pipeline registration needs VICTORY_METAL_SHARE_DIR for the host's metallib file"
                .to_string(),
        )
    })?;
    crate::native_metal::pso::write_metallib(&path, metallib)
        .map_err(NativeQueueError::Transport)?;
    client.register_pipeline(ticket, entry)
}

/// One pre-translated stage of a version 2 pipeline registration.
#[cfg(any(windows, test, target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub struct PipelineStageLibrary<'a> {
    /// Wire stage code (`client::PIPELINE_STAGE_*`).
    pub stage: u8,
    /// Serialized metallib for this stage, written to the next ticket's file.
    pub metallib: &'a [u8],
    /// Entry point the host resolves inside that library.
    pub entry: &'a str,
}

/// Registers a multi-stage graphics, mesh or compute pipeline (kind 4 v2).
///
/// Each stage gets its own consecutive ticket and `{ticket:016x}.metallib`
/// file, then one v2 frame carries the stage list, the encoded `PipelineState`,
/// the encoded ABI and the raw root-signature bytes. Nothing is written until
/// the whole request is validated by the caller, and a host rejection leaves
/// the session open.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_pipeline_v2(
    kind: u8,
    stages: &[PipelineStageLibrary<'_>],
    state: &[u8],
    abi: &[u8],
    root_signature: &[u8],
) -> Result<u64, NativeQueueError> {
    if stages.is_empty() {
        return Err(NativeQueueError::Refused(
            "pipeline registration needs at least one stage".to_string(),
        ));
    }
    for stage in stages {
        if stage.metallib.is_empty() {
            return Err(NativeQueueError::Refused(format!(
                "pipeline stage {} has no metallib",
                stage.stage
            )));
        }
    }
    ensure_live(LiveConfig::from_env)
        .map_err(|error| NativeQueueError::Transport(format!("{error}")))?;
    let mut queue = native_queue();
    let client = queue.sink_mut().client_mut().ok_or_else(|| {
        NativeQueueError::Transport("no live Wire v3 host session".to_string())
    })?;
    let first = client.next_pipeline_ticket();
    let mut wire: Vec<(u8, u64, &str)> = Vec::with_capacity(stages.len());
    for (index, stage) in stages.iter().enumerate() {
        let ticket = first
            .checked_add(u64::try_from(index).expect("stage index fits u64"))
            .ok_or_else(|| {
                NativeQueueError::Refused("pipeline ticket space exhausted".to_string())
            })?;
        let path = crate::native_metal::pso::metallib_path(ticket).ok_or_else(|| {
            NativeQueueError::Refused(
                "pipeline registration needs VICTORY_METAL_SHARE_DIR for the host's metallib \
                 file"
                    .to_string(),
            )
        })?;
        crate::native_metal::pso::write_metallib(&path, stage.metallib)
            .map_err(NativeQueueError::Transport)?;
        wire.push((stage.stage, ticket, stage.entry));
    }
    client.register_pipeline_v2(kind, &wire, state, abi, root_signature)
}

/// Connects the session queue from the environment exactly once per process.
///
/// Device creation calls this under `VICTORY_BACKEND=metal`. The result is
/// remembered (including a failure), so a title cannot get a second attempt
/// later in the run: a configured endpoint that refused the session is an
/// explicit error the device creation surfaces, and no endpoint leaves the
/// queue disconnected with an explicit transport failure on first submission.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn connect_once_from_env() -> Result<bool, NativeQueueError> {
    static ONCE: std::sync::Mutex<Option<Result<bool, NativeQueueError>>> =
        std::sync::Mutex::new(None);
    let mut slot = ONCE.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(result) = slot.as_ref() {
        return result.clone();
    }
    let result = native_queue_connect_from_env_try();
    *slot = Some(result.clone());
    result
}

/// [`native_queue_connect_from_env`] without collapsing a configuration
/// failure to `false`.
#[cfg(any(windows, test, target_os = "macos"))]
fn native_queue_connect_from_env_try() -> Result<bool, NativeQueueError> {
    ensure_live(LiveConfig::from_env)
}

/// Connects the session queue through `config`, once.
///
/// `Ok(true)` means a live sink is installed, `Ok(false)` that no endpoint is
/// configured (the queue stays disconnected and fails explicitly), and `Err`
/// that a configured endpoint refused the session.
#[cfg(any(windows, test, target_os = "macos"))]
fn ensure_live<F>(config: F) -> Result<bool, NativeQueueError>
where
    F: FnOnce() -> Option<Result<LiveConfig, crate::native_metal::client::TransportError>>,
{
    if native_queue_is_live() {
        return Ok(true);
    }
    let Some(config) = config() else {
        return Ok(false);
    };
    let config = config.map_err(|error| {
        NativeQueueError::Transport(format!("the Wire v3 configuration is invalid: {error}"))
    })?;
    // The connect happens outside the queue lock: it may block for the whole
    // connect timeout, and no submission may wait behind it.
    let client = config.connect().map_err(|error| {
        NativeQueueError::Transport(format!("the Wire v3 host bridge did not connect: {error}"))
    })?;
    native_queue_install_client(client);
    Ok(true)
}

/// Carries one `ExecuteCommandLists` batch over the native queue.
///
/// `record_us` is the guest time spent draining the finished lists' recorded
/// submissions; it is reported with the batch's own encode and transport time.
/// The Metal path records one frame sample, including a frame the producer
/// refused for an unsupported opcode (that is what the histogram counts). A
/// transport failure records nothing: no frame reached the host.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn submit_recorded_batch(
    submissions: &[SubmissionV3],
    record_us: u64,
) -> Result<NativeDispatch, NativeQueueError> {
    // The backend decision was made by `crate::queue::execute` before it
    // called the native path, so the environment is the only gate left.
    if let Err(error) = ensure_live(LiveConfig::from_env) {
        tracing::warn!(%error, "VICTORY_BACKEND=metal: the native queue has no live sink");
    }

    let mut queue = native_queue();
    let outcome = dispatch_measured(GpuBackend::Metal, &mut queue, submissions);
    // The sequences were assigned at enqueue, so the callers' submissions may
    // still be unassigned here; the queue knows the last one it handed out.
    let highest_sequence = queue.highest_sequence();
    drop(queue);

    match outcome {
        Ok((dispatch, timing)) => {
            if highest_sequence != 0 {
                LAST_SEQUENCE.fetch_max(u64::from(highest_sequence), Ordering::AcqRel);
            }
            if let Some(timing) = timing {
                // `gpu_us` is 0: the kind 5 reply carries a status only. The
                // host's completion still advances the shared fence page
                // (`WireV3Client::completed_fence`), which the guest reads with
                // one atomic load.
                crate::native_metal::telemetry::record_frame(
                    record_us,
                    timing.submit_us,
                    timing.encode_us,
                    0,
                    None,
                );
            }
            // A host status other than `Ok` is a refusal, including the
            // replayed `AlreadyExecuted` this queue surfaces rather than
            // retries. The opcode of a producer refusal was already counted
            // where the call was refused (`list::with_metal`), so the frame
            // sample here never counts it twice.
            if matches!(
                dispatch,
                NativeDispatch::Native {
                    status: ReplyStatus::Ok,
                    ..
                }
            ) {
                crate::native_metal::telemetry::note_submit_ok();
            } else if matches!(dispatch, NativeDispatch::Native { .. }) {
                crate::native_metal::telemetry::note_submit_refused();
                crate::native_metal::telemetry::record_frame(record_us, 0, 0, 0, None);
            }
            Ok(dispatch)
        }
        Err(error) => {
            crate::native_metal::telemetry::note_submit_refused();
            if let NativeQueueError::Producer(producer) = &error
                && producer.opcode().is_some()
            {
                crate::native_metal::telemetry::record_frame(record_us, 0, 0, 0, None);
            }
            Err(error)
        }
    }
}

/// `ExecuteCommandLists` under `VICTORY_BACKEND=metal`.
///
/// Drains each title list's finished Wire v3 submissions (in list order, and
/// in each list's recorded order) into the native queue. Host D3D12 work is
/// never enqueued on this path: an empty list or a refused batch is logged and
/// surfaced through the queue, with no fallback. The first batch also connects
/// the session when `VICTORY_METAL_ENDPOINT`/`VICTORY_METAL_SECRET` are set.
#[cfg(windows)]
pub fn execute_native(count: u32, lists: *const crate::object::This) {
    if count == 0 || lists.is_null() {
        return;
    }
    // SAFETY: the title passes `count` lists, as `queue::execute` documents.
    let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
    for &list in lists {
        let started = Instant::now();
        let submissions = crate::list::take_submissions(list);
        // Guest time spent draining the finished list, reported as the frame's
        // record phase: the title's own recording happens in the list calls.
        let record_us = micros(started.elapsed());
        if submissions.is_empty() {
            let failures = crate::list::metal_failures(list);
            if failures > 0 {
                tracing::debug!(
                    failures,
                    "native list refused calls and carried no submission; the queue reports the \
                     refusal the title will see at Close"
                );
            }
        }
        match submit_recorded_batch(&submissions, record_us) {
            Ok(NativeDispatch::Native {
                status,
                submissions: carried,
            }) => {
                let list = submissions.first().and_then(|submission| submission.list);
                if status == ReplyStatus::Ok {
                    tracing::trace!(carried, ?list, "native queue carried a list");
                } else {
                    tracing::warn!(
                        carried,
                        ?list,
                        ?status,
                        "VICTORY_BACKEND=metal: the host did not execute a submitted list"
                    );
                }
            }
            Ok(NativeDispatch::Host) => {}
            Err(NativeQueueError::Empty(_)) => {
                tracing::debug!(
                    "VICTORY_BACKEND=metal: a submitted list carried no native submission"
                );
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "VICTORY_BACKEND=metal: the native queue refused a submission"
                );
            }
        }
    }
}

/// A registered native swapchain and the host presenter that consumes it.
pub struct NativePresenter {
    swapchain: NativeSwapchain,
    presenter: Box<dyn SurfacePresenter + Send>,
}

impl NativePresenter {
    /// Registers `swapchain` with `presenter`.
    pub fn new(swapchain: NativeSwapchain, presenter: Box<dyn SurfacePresenter + Send>) -> Self {
        Self {
            swapchain,
            presenter,
        }
    }

    /// The swapchain state this presenter owns.
    pub fn swapchain(&self) -> &NativeSwapchain {
        &self.swapchain
    }

    /// Presents the swapchain's own surface at its current pacing.
    ///
    /// shortcut: the host completion channel is not connected yet, so the
    /// accepted frame is finished at once; otherwise the second present would
    /// be refused as [`PresentError::FrameInFlight`].
    ///
    /// With the `diagnostics` feature the end of every present runs the strict
    /// zero-D3DMetal guard ([`guard_native_only`]) and writes the frame
    /// telemetry window as one line. A tripped guard refuses the frame
    /// explicitly; it is never logged and shown.
    pub fn present(&mut self) -> Result<PresentedSurface, NativeQueueError> {
        let step = self
            .swapchain
            .present_step(self.swapchain.sync_interval().as_wire())?;
        let frame = present_surface(&mut self.swapchain, step, &mut *self.presenter)?;
        self.swapchain.finish_present()?;
        #[cfg(feature = "diagnostics")]
        {
            guard_native_only(crate::native_metal::telemetry::global())?;
            crate::native_metal::telemetry::log_frame();
        }
        Ok(frame)
    }

    /// Marks the in-flight frame as finished.
    pub fn finish(&mut self) -> Result<InFlightFrame, NativeQueueError> {
        Ok(self.swapchain.finish_present()?)
    }
}

impl std::fmt::Debug for NativePresenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePresenter")
            .field("surface_id", &self.swapchain.surface_id())
            .field("frame_index", &self.swapchain.next_frame_index())
            .finish_non_exhaustive()
    }
}

/// The session's installed native swapchain.
static PRESENT: std::sync::Mutex<Option<NativePresenter>> = std::sync::Mutex::new(None);

/// Installs the native swapchain the present entry point hands frames to.
///
/// Only one may be installed: a second registration is refused with
/// [`NativeQueueError::Refused`] rather than replacing a live presenter.
pub fn install_present(presenter: NativePresenter) -> Result<(), NativeQueueError> {
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_some() {
        return Err(NativeQueueError::Refused(
            "a native swapchain is already installed".to_string(),
        ));
    }
    *slot = Some(presenter);
    Ok(())
}

/// Takes the installed presenter, if any (swapchain teardown and tests).
pub fn take_present() -> Option<NativePresenter> {
    PRESENT.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Whether a native presenter is installed.
pub fn native_queue_has_presenter() -> bool {
    PRESENT
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_some()
}

/// Presents one frame into the installed native swapchain.
///
/// With no swapchain installed this is [`NativeQueueError::NoSwapchain`]; the
/// queue entry point turns that into a failure HRESULT and never reaches the
/// D3D12 swapchain.
pub fn present_registered() -> Result<PresentedSurface, NativeQueueError> {
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    let result = match slot.as_mut() {
        Some(presenter) => {
            let format = presenter.swapchain().format();
            let result = presenter.present();
            if result.is_ok() {
                tracing::debug!(
                    surface = presenter.swapchain().surface_id(),
                    format = format.as_str(),
                    "native frame presented in its target format"
                );
            }
            result
        }
        None => {
            tracing::warn!(
                surface_installed = "none",
                "VICTORY_BACKEND=metal: present refused: no native surface is installed"
            );
            Err(NativeQueueError::NoSwapchain)
        }
    };
    match &result {
        Ok(_) => crate::native_metal::telemetry::note_present_ok(),
        Err(_) => crate::native_metal::telemetry::note_present_refused(),
    }
    result
}

/// The native presenter used by the real title path: it carries one
/// `CommandV3::Present` to the host through the session queue.
///
/// No pixels cross the boundary: the host owns the surface (registered with
/// kind 9) and presents it. A refused registration or submission is returned
/// unchanged, so the guest `Present` entry point returns a real failure
/// HRESULT instead of falling back.
pub struct WirePresenter {
    surface_id: u64,
    texture_id: u64,
    registered: bool,
}

impl WirePresenter {
    /// A presenter for `surface_id`, backed by the already-registered
    /// `texture_id`.
    pub fn new(surface_id: u64, texture_id: u64) -> Self {
        Self {
            surface_id,
            texture_id,
            registered: false,
        }
    }
}

impl SurfacePresenter for WirePresenter {
    fn present(&mut self, frame: PresentedSurface) -> Result<(), PresentError> {
        if !self.registered {
            match crate::native_metal::queue::register_surface(self.surface_id, self.texture_id) {
                Ok(true) => self.registered = true,
                Ok(false) => {
                    return Err(PresentError::SinkRefused {
                        frame_index: frame.frame_index,
                        reason: format!(
                            "the host refused surface {} backed by texture {}",
                            self.surface_id, self.texture_id
                        ),
                    });
                }
                Err(error) => {
                    return Err(PresentError::SinkRefused {
                        frame_index: frame.frame_index,
                        reason: error.to_string(),
                    });
                }
            }
        }
        use std::sync::atomic::AtomicU64;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let list_id = NEXT.fetch_add(1, Ordering::Relaxed).max(1);
        let mut producer =
            crate::native_metal::producer::MetalProducer::new(list_id, 1).map_err(|error| {
                PresentError::SinkRefused {
                    frame_index: frame.frame_index,
                    reason: error.to_string(),
                }
            })?;
        producer
            .present(self.surface_id, frame.sync_interval.as_wire())
            .map_err(|error| PresentError::SinkRefused {
                frame_index: frame.frame_index,
                reason: error.to_string(),
            })?;
        let submission = producer.flush().map_err(|error| PresentError::SinkRefused {
            frame_index: frame.frame_index,
            reason: error.to_string(),
        })?;
        let submissions = [submission];
        match submit_recorded_batch(&submissions, 0) {
            Ok(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                ..
            }) => Ok(()),
            Ok(NativeDispatch::Native { status, .. }) => Err(PresentError::SinkRefused {
                frame_index: frame.frame_index,
                reason: format!("the host answered {status:?} for the present"),
            }),
            Ok(NativeDispatch::Host) => Err(PresentError::SinkRefused {
                frame_index: frame.frame_index,
                reason: "the present reached the D3D12 path".to_string(),
            }),
            Err(error) => Err(PresentError::SinkRefused {
                frame_index: frame.frame_index,
                reason: error.to_string(),
            }),
        }
    }
}

/// Installs the wire presenter the launcher's environment names.
///
/// `VICTORY_METAL_SURFACE` (a nonzero u64) selects the surface;
/// `VICTORY_METAL_PRESENT_TEXTURE` names the registered texture backing it
/// (default: the surface id). Width and height come from
/// `VICTORY_METAL_SURFACE_WIDTH`/`_HEIGHT` (default 3840x2160). With no
/// surface named nothing is installed, and `Present` fails explicitly with
/// `NoSwapchain` instead of using the D3D12 swapchain.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn install_present_from_env() -> Result<bool, NativeQueueError> {
    fn number(name: &str) -> Option<u64> {
        std::env::var(name)
            .ok()
            .and_then(|text| text.trim().parse().ok())
    }
    let Some(surface_id) = number("VICTORY_METAL_SURFACE").filter(|&id| id != 0) else {
        return Ok(false);
    };
    let texture_id = number("VICTORY_METAL_PRESENT_TEXTURE")
        .filter(|&id| id != 0)
        .unwrap_or(surface_id);
    let width = number("VICTORY_METAL_SURFACE_WIDTH").unwrap_or(3840) as u32;
    let height = number("VICTORY_METAL_SURFACE_HEIGHT").unwrap_or(2160) as u32;
    install_wire_present_as(
        surface_id,
        texture_id,
        width,
        height,
        SurfaceOrigin::Env,
    )?;
    Ok(true)
}

/// Where an installed surface's identity and extent came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceOrigin {
    /// The launcher's environment (`VICTORY_METAL_SURFACE*`).
    Env,
    /// The target texture the title presented (the environment named none).
    Target,
}

impl SurfaceOrigin {
    /// The label reported in present telemetry.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Target => "target",
        }
    }
}

/// Installs a [`WirePresenter`] over a headless swapchain (tests and the
/// launcher path).
///
/// The surface format follows the target texture's registered host format when
/// it is known (code 9 = RGBA16Float); otherwise the BGRA8 default is kept.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn install_wire_present(
    surface_id: u64,
    texture_id: u64,
    width: u32,
    height: u32,
) -> Result<(), NativeQueueError> {
    install_wire_present_as(
        surface_id,
        texture_id,
        width,
        height,
        SurfaceOrigin::Target,
    )
}

#[cfg(any(windows, test, target_os = "macos"))]
fn install_wire_present_as(
    surface_id: u64,
    texture_id: u64,
    width: u32,
    height: u32,
    origin: SurfaceOrigin,
) -> Result<(), NativeQueueError> {
    let format = texture_format_code(texture_id)
        .and_then(SurfaceFormat::from_host_format_code)
        .unwrap_or(SurfaceFormat::Bgra8Unorm);
    let swapchain = NativeSwapchain::new(surface_id, width, height, format, GpuBackend::Metal)?;
    tracing::info!(
        surface = surface_id,
        texture = texture_id,
        width,
        height,
        format = format.as_str(),
        surface_installed = origin.as_str(),
        "native present surface installed"
    );
    install_present(NativePresenter::new(
        swapchain,
        Box::new(WirePresenter::new(surface_id, texture_id)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
    use std::sync::mpsc::{self, Receiver};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    use victory_metal::wire_v3::{CommandV3, Present, decode_submission_v3};

    use crate::object::S_OK;
    use crate::native_metal::client::{ENVELOPE_LEN, KIND_ACK, KIND_AUTH, KIND_SUBMISSION_V3};
    use crate::native_metal::producer::{MetalList, UNASSIGNED_SEQUENCE, reset_sequences};
    use crate::native_metal::telemetry;

    const SECRET: [u8; 32] = [0x5a; 32];
    const TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    /// Serialises the tests that drive the process-global queue, presenter,
    /// fence sequence and telemetry. Those are one instance per process by
    /// design (`native_queue`, `PRESENT`, `LAST_SEQUENCE`, `pending_signals`),
    /// and the tests exercise the real entry points rather than injected
    /// instances, so a scoped lock is the only way to stop one test's reset or
    /// installed client from changing another's expectations. Tests that use a
    /// private `NativeQueue` or `RecordingSink` do not take it.
    fn serialized() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// A sink that keeps every frame and answers from a script.
    #[derive(Debug, Default)]
    struct RecordingSink {
        frames: Vec<Vec<u8>>,
        statuses: VecDeque<ReplyStatus>,
    }

    impl V3Sink for RecordingSink {
        fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
            self.frames.push(bytes.to_vec());
            Ok(self.statuses.pop_front().unwrap_or(ReplyStatus::Ok))
        }
    }

    fn submission(sequence: u32, list_id: u64, surface_id: u64) -> SubmissionV3 {
        SubmissionV3 {
            sequence,
            list: Some(ListIdentity {
                list_id,
                list_generation: 1,
            }),
            commands: vec![CommandV3::Present(Present {
                surface_id,
                sync_interval: 0,
            })],
        }
    }

    /// Two closed title lists submitted one after the other reach the sink in
    /// recorded order, one send each, with their identities intact.
    #[test]
    fn lists_are_sent_in_recorded_order_without_coalescing() {
        let mut queue = NativeQueue::new(RecordingSink::default());
        let first = submission(1, 100, 1);
        let second = submission(2, 200, 2);

        assert_eq!(
            dispatch(GpuBackend::Metal, &mut queue, std::slice::from_ref(&first)),
            Ok(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                submissions: 1,
            })
        );
        assert_eq!(
            dispatch(GpuBackend::Metal, &mut queue, std::slice::from_ref(&second)),
            Ok(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                submissions: 1,
            })
        );

        assert_eq!(queue.sent(), 2);
        assert!(queue.is_empty());
        let frames = &queue.sink().frames;
        assert_eq!(frames.len(), 2, "one send per submission");
        let decoded: Vec<SubmissionV3> = frames
            .iter()
            .map(|bytes| decode_submission_v3(bytes).expect("canonical v3 encoding"))
            .collect();
        assert_eq!(decoded, vec![first, second]);
        assert_eq!(
            decoded[0].list.map(|list| list.list_id),
            Some(100),
            "the first list's identity is preserved"
        );
        assert_eq!(decoded[1].sequence, 2, "the second list keeps its sequence");
    }

    /// A replayed list's `AlreadyExecuted` is returned to the caller and the
    /// frame is not re-sent: the queue holds nothing to retry.
    #[test]
    fn a_replay_status_is_surfaced_and_not_retried() {
        let mut sink = RecordingSink::default();
        sink.statuses.push_back(ReplyStatus::AlreadyExecuted);
        let mut queue = NativeQueue::new(sink);
        let replayed = submission(7, 70, 1);

        assert_eq!(
            queue.submit(std::slice::from_ref(&replayed)),
            Ok(ReplyStatus::AlreadyExecuted)
        );
        assert_eq!(queue.sent(), 1);
        assert_eq!(queue.sink().frames.len(), 1);

        assert_eq!(queue.flush(), Ok(Vec::new()), "nothing is retried");
        assert_eq!(queue.sink().frames.len(), 1, "the replayed frame is gone");
    }

    /// A batch with no submission is refused before anything reaches the
    /// transport.
    #[test]
    fn an_empty_batch_is_refused() {
        let mut queue = NativeQueue::new(RecordingSink::default());
        assert!(matches!(queue.submit(&[]), Err(NativeQueueError::Empty(_))));
        assert_eq!(queue.sink().frames.len(), 0);
        assert_eq!(queue.sent(), 0);
        assert!(queue.is_empty());
    }

    /// The D3D12 backend leaves the native queue untouched: no encode, no
    /// sink call, no pending frame.
    #[test]
    fn the_d3d12_backend_does_not_touch_the_native_queue() {
        let mut queue = NativeQueue::new(RecordingSink::default());
        let batch = [submission(1, 1, 1)];

        assert_eq!(
            dispatch(GpuBackend::D3d12, &mut queue, &batch),
            Ok(NativeDispatch::Host)
        );
        assert_eq!(queue.sink().frames.len(), 0);
        assert_eq!(queue.sent(), 0);
        assert!(queue.is_empty());
    }

    /// An unsupported producer operation keeps `E_NOTIMPL`; the other
    /// refusals are explicit failure HRESULTs.
    #[test]
    fn an_unsupported_producer_error_becomes_a_failure_hresult() {
        assert_eq!(
            NativeQueueError::from(ProducerError::unsupported_opcode(
                "ExecuteIndirect",
                victory_metal::wire_v3::OP_DISPATCH_3D
            ))
            .hresult(),
            0x8000_4001_u32 as i32
        );
        assert_eq!(
            NativeQueueError::from(ProducerError::InvalidListIdentity).hresult(),
            E_FAIL
        );
        assert_eq!(
            NativeQueueError::Empty("test").hresult(),
            0x8007_0057_u32 as i32
        );
        assert_eq!(
            NativeQueueError::Transport("test".to_string()).hresult(),
            E_FAIL
        );
        assert_eq!(NativeQueueError::NoSwapchain.hresult(), E_FAIL);
    }

    /// A presenter that records what the swapchain handed it.
    #[derive(Debug, Default, Clone)]
    struct SharedPresenter {
        frames: Arc<Mutex<Vec<PresentedSurface>>>,
    }

    impl SurfacePresenter for SharedPresenter {
        fn present(&mut self, frame: PresentedSurface) -> Result<(), PresentError> {
            self.frames
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(frame);
            Ok(())
        }
    }

    /// The queue presents into the installed `NativeSwapchain`, advancing its
    /// frame index and finishing each frame; with none installed it is an
    /// explicit error.
    #[test]
    fn the_queue_presents_into_the_installed_native_swapchain() {
        let _serial = serialized();
        let _ = take_present();
        assert_eq!(present_registered(), Err(NativeQueueError::NoSwapchain));

        let presenter = SharedPresenter::default();
        let frames = Arc::clone(&presenter.frames);
        let swapchain = NativeSwapchain::new_headless(9, 64, 64).expect("headless swapchain");
        install_present(NativePresenter::new(swapchain, Box::new(presenter)))
            .expect("first install");

        let first = present_registered().expect("first frame");
        assert_eq!((first.surface_id, first.frame_index), (9, 0));
        let second = present_registered().expect("second frame");
        assert_eq!((second.surface_id, second.frame_index), (9, 1));

        let recorded = frames.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(recorded, vec![first, second]);

        let swapchain = NativeSwapchain::new_headless(10, 32, 32).expect("headless swapchain");
        assert_eq!(
            install_present(NativePresenter::new(
                swapchain,
                Box::new(SharedPresenter::default())
            )),
            Err(NativeQueueError::Refused(
                "a native swapchain is already installed".to_string()
            ))
        );
        let _ = take_present();
        assert_eq!(present_registered(), Err(NativeQueueError::NoSwapchain));
    }

    /// A registered 16F target installs a headless RGBA16Float surface of the
    /// target's extent: the present path does not coerce it to BGRA8.
    #[test]
    fn a_registered_fp16_target_installs_an_rgba16_float_surface() {
        let _serial = serialized();
        let _ = take_present();
        note_texture_format(0x16f, 9);
        install_wire_present(0x16f, 0x16f, 2560, 1440).expect("install the wire presenter");
        let installed = take_present().expect("the presenter is installed");
        assert_eq!(installed.swapchain().format(), SurfaceFormat::Rgba16Float);
        assert_eq!(
            (installed.swapchain().width(), installed.swapchain().height()),
            (2560, 1440)
        );
        native_queue_reset();
    }

    // ------------------------------------------------------------ live sink

    /// A loopback host that authenticates one peer, answers `statuses` in
    /// order and reports every carried submission.
    struct TestHost {
        port: u16,
        handle: Option<JoinHandle<()>>,
    }

    impl TestHost {
        fn spawn(statuses: Vec<ReplyStatus>) -> (Self, Receiver<SubmissionV3>) {
            let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .expect("bind loopback");
            let port = listener.local_addr().expect("listener address").port();
            let (sender, receiver) = mpsc::channel();
            let handle = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().expect("accept the guest");
                stream
                    .set_read_timeout(Some(TEST_TIMEOUT))
                    .expect("server read timeout");
                let (kind, payload) = read_frame(&mut stream);
                assert_eq!(kind, KIND_AUTH, "the first frame must authenticate");
                assert_eq!(payload, SECRET, "the hello must carry the exact secret");
                write_frame(&mut stream, KIND_ACK, &[0]);

                for status in statuses {
                    let (kind, payload) = read_frame(&mut stream);
                    assert_eq!(kind, KIND_SUBMISSION_V3);
                    let inner = decode_submission_v3(&payload[16..]).expect("carried v3 frame");
                    sender.send(inner).expect("report the submission");
                    let reply = victory_metal::wire::encode_reply(&victory_metal::wire::Reply {
                        sequence: u32::from_le_bytes([
                            payload[6], payload[7], payload[8], payload[9],
                        ]),
                        status,
                    })
                    .expect("reply encodes");
                    write_frame(&mut stream, KIND_SUBMISSION_V3, &reply);
                }
            });
            (
                Self {
                    port,
                    handle: Some(handle),
                },
                receiver,
            )
        }

        fn endpoint(&self) -> String {
            format!("127.0.0.1:{}", self.port)
        }

        fn config(&self) -> LiveConfig {
            LiveConfig::parse(Some(&self.endpoint()), Some(&"5a".repeat(32)))
                .expect("both variables present")
                .expect("well-formed configuration")
        }
    }

    impl Drop for TestHost {
        fn drop(&mut self) {
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn read_frame(stream: &mut std::net::TcpStream) -> (u8, Vec<u8>) {
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

    fn write_frame(stream: &mut std::net::TcpStream, kind: u8, payload: &[u8]) {
        let length = u32::try_from(payload.len()).expect("payload fits u32");
        stream.write_all(&[kind]).expect("server kind");
        stream
            .write_all(&length.to_le_bytes())
            .expect("server length");
        stream.write_all(payload).expect("server payload");
        stream.flush().expect("server flush");
    }

    /// A sink that refuses the first `failures` sends and then records frames.
    #[derive(Debug, Default)]
    struct FlakySink {
        failures: usize,
        frames: Vec<Vec<u8>>,
    }

    impl V3Sink for FlakySink {
        fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
            if self.failures > 0 {
                self.failures -= 1;
                return Err(NativeQueueError::Transport("test refusal".to_string()));
            }
            self.frames.push(bytes.to_vec());
            Ok(ReplyStatus::Ok)
        }
    }

    /// A configured endpoint connects on the first submission, carries the
    /// batch over one authenticated socket, and records one native frame with
    /// no D3DMetal fallback.
    #[test]
    fn a_configured_endpoint_connects_lazily_and_carries_the_batch() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        let (host, carried) = TestHost::spawn(vec![ReplyStatus::Ok]);
        assert!(!native_queue_is_live(), "a reset queue starts disconnected");

        let config = host.config();
        assert_eq!(ensure_live(|| Some(Ok(config))), Ok(true));
        assert!(native_queue_is_live(), "the first submission connected");

        let batch = submission(31, 300, 4);
        assert_eq!(
            submit_recorded_batch(std::slice::from_ref(&batch), 9),
            Ok(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                submissions: 1,
            })
        );
        assert_eq!(
            carried
                .recv_timeout(TEST_TIMEOUT)
                .expect("host saw the frame"),
            batch,
            "the submission reached the real socket unchanged"
        );

        assert_eq!(telemetry::global().recorded_frames(), 1);
        assert_eq!(
            telemetry::global().d3dmetal_fallback_count(),
            0,
            "the native path never records a D3DMetal fallback"
        );
        assert_eq!(telemetry::global().assert_no_fallback(), Ok(()));

        native_queue_reset();
        assert!(!native_queue_is_live());
    }

    /// With no endpoint configured the queue keeps today's explicit transport
    /// failure: nothing connects, nothing is recorded, the frame stays queued.
    #[test]
    fn without_an_endpoint_the_queue_fails_explicitly() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        assert_eq!(ensure_live(|| None), Ok(false));
        assert!(!native_queue_is_live());

        let error = submit_recorded_batch(&[submission(32, 301, 5)], 7)
            .expect_err("no host means no success");
        assert!(
            matches!(&error, NativeQueueError::Transport(why) if why.contains("not connected")),
            "unexpected error {error:?}"
        );
        assert_eq!(
            telemetry::global().recorded_frames(),
            0,
            "a frame that never reached the host is not a native frame"
        );
        assert_eq!(
            native_queue().pending(),
            1,
            "the refused frame is still queued, not dropped"
        );
        native_queue_reset();
        assert_eq!(native_queue().pending(), 0);
    }

    /// A refused send leaves the batch at the front of the queue, intact and
    /// unsent; the next flush carries exactly those bytes once.
    #[test]
    fn a_refused_send_leaves_the_outbound_frame_intact() {
        let mut queue = NativeQueue::new(FlakySink {
            failures: 1,
            frames: Vec::new(),
        });
        let batch = submission(33, 302, 6);
        assert!(matches!(
            queue.submit(std::slice::from_ref(&batch)),
            Err(NativeQueueError::Transport(_))
        ));
        assert_eq!(queue.pending(), 1, "the failed frame stays queued");
        assert_eq!(queue.sent(), 0, "nothing was counted as sent");
        assert!(queue.sink().frames.is_empty());

        assert_eq!(queue.flush(), Ok(vec![ReplyStatus::Ok]));
        assert_eq!(queue.pending(), 0);
        assert_eq!(queue.sent(), 1);
        assert_eq!(
            decode_submission_v3(&queue.sink().frames[0]).expect("canonical v3"),
            batch,
            "the bytes that finally went out are the bytes that were refused"
        );
    }

    /// A live sink's `AlreadyExecuted` still reaches the caller and is not
    /// retried: the queue holds nothing to send again.
    #[test]
    fn a_live_already_executed_reply_is_surfaced_once() {
        let _serial = serialized();
        native_queue_reset();
        let (host, carried) = TestHost::spawn(vec![ReplyStatus::AlreadyExecuted]);
        let config = host.config();
        assert_eq!(ensure_live(|| Some(Ok(config))), Ok(true));

        assert_eq!(
            submit_recorded_batch(&[submission(34, 303, 7)], 1),
            Ok(NativeDispatch::Native {
                status: ReplyStatus::AlreadyExecuted,
                submissions: 1,
            })
        );
        let _ = carried.recv_timeout(TEST_TIMEOUT).expect("one frame");
        assert_eq!(native_queue().pending(), 0, "nothing is retried");
        native_queue_reset();
    }

    /// The D3D12 backend touches neither the native queue nor the telemetry
    /// counters: no encode, no sink call, no frame sample.
    #[test]
    fn the_d3d12_backend_leaves_the_queue_and_telemetry_untouched() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        let mut queue = NativeQueue::new(RecordingSink::default());
        let before = telemetry::global().snapshot();

        let (dispatch, timing) =
            dispatch_measured(GpuBackend::D3d12, &mut queue, &[submission(35, 304, 8)])
                .expect("the host path never fails on the native queue");
        assert_eq!(dispatch, NativeDispatch::Host);
        assert_eq!(timing, None, "the host path reports no native timing");

        assert_eq!(queue.sink().frames.len(), 0);
        assert_eq!(queue.sent(), 0);
        assert!(queue.is_empty());
        assert_eq!(telemetry::global().snapshot(), before);
        assert!(!native_queue_is_live());
    }

    /// A unique path for a test's shared fence page.
    fn test_page(label: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-gpu-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// A loopback host that accepts one session, acknowledges the fence page
    /// (kind 6) and then hands the stream to `serve`.
    fn page_host<F>(ready: mpsc::Sender<()>, serve: F) -> (String, JoinHandle<()>)
    where
        F: FnOnce(&mut std::net::TcpStream) + Send + 'static,
    {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
            .expect("bind loopback");
        let port = listener.local_addr().expect("listener address").port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept the guest");
            stream
                .set_read_timeout(Some(TEST_TIMEOUT))
                .expect("server read timeout");
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, KIND_AUTH, "the first frame authenticates");
            assert_eq!(payload, SECRET);
            write_frame(&mut stream, KIND_ACK, &[0]);
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, crate::native_metal::client::KIND_FENCE_PAGE);
            assert_eq!(payload.len(), 8);
            write_frame(&mut stream, KIND_ACK, &[0]);
            ready.send(()).expect("report ready");
            serve(&mut stream);
        });
        (format!("127.0.0.1:{port}"), handle)
    }

    /// `GetCompletedValue` reads the shared page once: a signal stays pending
    /// until the host publishes the submission's sequence, then moves with no
    /// IPC round trip.
    #[test]
    fn the_shared_page_gates_signals_with_one_atomic_read() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        super::LAST_SEQUENCE.store(7, Ordering::Release);
        let path = test_page("fence");
        let (ready, ready_rx) = mpsc::channel();
        let (endpoint, handle) = page_host(ready, |stream| {
            // Hold the session open until the client is dropped.
            let mut byte = [0u8; 1];
            let _ = stream.read(&mut byte);
        });
        let client = WireV3Client::connect_with_fence_page(
            &endpoint,
            &SECRET,
            TEST_TIMEOUT,
            Some(path.clone()),
        )
        .expect("an authenticated client with a page");
        assert_eq!(client.completed_fence(), Some(0), "one atomic load");
        native_queue_install_client(client);
        ready_rx.recv_timeout(TEST_TIMEOUT).expect("page registered");

        let mut fence = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut fence
            ),
            S_OK
        );
        signal_native(fence, 9);
        assert_eq!(
            crate::fence::value(fence),
            0,
            "the signal waits for the host's completed sequence"
        );

        // The host publishes sequence 7 (the submission before the signal).
        let host_page = crate::native_metal::fence_page_guest::GuestFencePage::open(&path)
            .expect("the host's own mapping");
        host_page.publish(7);
        poll_native_fence();
        assert_eq!(crate::fence::value(fence), 9, "completed with one load");

        native_queue_reset();
        drop(handle);
        let _ = std::fs::remove_file(&path);
        // SAFETY: the fence is one of ours; drop the test reference.
        assert_eq!(
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(fence, 0x10)(fence),
            0
        );
    }

    /// With no shared page the synchronous kind 5 reply means the host has
    /// already run the batch, so a signal completes immediately.
    #[test]
    fn without_a_page_a_signal_completes_after_the_synchronous_reply() {
        let _serial = serialized();
        native_queue_reset();
        let mut fence = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut fence
            ),
            S_OK
        );
        signal_native(fence, 3);
        assert_eq!(crate::fence::value(fence), 3);
        // SAFETY: the fence is one of ours.
        assert_eq!(
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(fence, 0x10)(fence),
            0
        );
    }

    /// The native presenter registers its surface and carries one `Present`
    /// command over the live queue; a refused registration is an explicit
    /// presenter error.
    #[test]
    fn the_wire_presenter_registers_the_surface_and_carries_a_present() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        let _ = take_present();
        let path = test_page("present");
        let (ready, ready_rx) = mpsc::channel();
        let (carried, carried_rx) = mpsc::channel();
        let (endpoint, handle) = page_host(ready, move |stream| {
            let (kind, payload) = read_frame(stream);
            assert_eq!(kind, crate::native_metal::client::KIND_SURFACE_REG);
            assert_eq!(payload.len(), 16);
            write_frame(stream, KIND_ACK, &[0]);
            let (kind, payload) = read_frame(stream);
            assert_eq!(kind, KIND_SUBMISSION_V3);
            let inner = decode_submission_v3(&payload[16..]).expect("a v3 submission");
            carried.send(inner).expect("report the present");
            let reply = victory_metal::wire::encode_reply(&victory_metal::wire::Reply {
                sequence: u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]),
                status: ReplyStatus::Ok,
            })
            .expect("reply encodes");
            write_frame(stream, KIND_SUBMISSION_V3, &reply);
            let mut byte = [0u8; 1];
            let _ = stream.read(&mut byte);
        });
        let client = WireV3Client::connect_with_fence_page(
            &endpoint,
            &SECRET,
            TEST_TIMEOUT,
            Some(path.clone()),
        )
        .expect("an authenticated client");
        native_queue_install_client(client);
        ready_rx.recv_timeout(TEST_TIMEOUT).expect("registered");
        install_wire_present(42, 42, 64, 64).expect("install the wire presenter");
        let before = telemetry::global().present_counts();

        let frame = present_registered().expect("the host accepted the frame");
        assert_eq!((frame.surface_id, frame.frame_index), (42, 0));
        let delivered = carried_rx.recv_timeout(TEST_TIMEOUT).expect("the host saw it");
        assert_eq!(delivered.list.map(|list| list.list_id), Some(1));
        assert!(
            matches!(
                delivered.commands.as_slice(),
                [victory_metal::wire_v3::CommandV3::Present(present)]
                    if present.surface_id == 42 && present.sync_interval == 0
            ),
            "unexpected commands {:?}",
            delivered.commands
        );
        assert_eq!(
            telemetry::global().present_counts(),
            (before.0 + 1, before.1)
        );
        assert_eq!(telemetry::global().d3dmetal_fallback_count(), 0);

        let _ = take_present();
        native_queue_reset();
        drop(handle);
        let _ = std::fs::remove_file(&path);
    }

    /// A surface the host refuses makes `Present` fail explicitly and is
    /// counted as a refused present.
    #[test]
    fn a_refused_surface_registration_refuses_the_present() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        let _ = take_present();
        let path = test_page("present-refused");
        let (ready, ready_rx) = mpsc::channel();
        let (endpoint, handle) = page_host(ready, move |stream| {
            let (kind, _payload) = read_frame(stream);
            assert_eq!(kind, crate::native_metal::client::KIND_SURFACE_REG);
            write_frame(stream, KIND_ACK, &[1]);
            let mut byte = [0u8; 1];
            let _ = stream.read(&mut byte);
        });
        let client = WireV3Client::connect_with_fence_page(
            &endpoint,
            &SECRET,
            TEST_TIMEOUT,
            Some(path.clone()),
        )
        .expect("an authenticated client");
        native_queue_install_client(client);
        ready_rx.recv_timeout(TEST_TIMEOUT).expect("registered");
        install_wire_present(43, 43, 64, 64).expect("install");
        let before = telemetry::global().present_counts();

        let error = present_registered().expect_err("a refused surface cannot present");
        assert!(
            matches!(error, NativeQueueError::Present(_)),
            "unexpected error {error:?}"
        );
        assert_eq!(
            telemetry::global().present_counts(),
            (before.0, before.1 + 1)
        );
        assert_eq!(telemetry::global().d3dmetal_fallback_count(), 0);
        let _ = take_present();
        native_queue_reset();
        drop(handle);
        let _ = std::fs::remove_file(&path);
    }

    // --------------------------------------------------- dense sequences

    /// The Boot #2 failure, reproduced: a hundred `Close` calls on an empty
    /// list must consume no sequence, so the first real frame is sequence 1
    /// instead of 586 and the host's ledger stays dense.
    #[test]
    fn a_hundred_empty_closes_do_not_consume_a_sequence() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(RecordingSink::default());
        let mut list = MetalList::new(101).expect("list");

        for _ in 0..100 {
            list.close().expect("an empty close is not a submission");
            assert!(list.take().is_empty(), "an empty stream is never queued");
        }
        assert!(queue.is_empty(), "nothing reached the transport");

        list.producer_mut().present(1, 0).expect("present");
        list.close().expect("the first real frame closes");
        let batch = list.take();
        assert_eq!(batch.len(), 1);
        assert_eq!(
            batch[0].sequence, UNASSIGNED_SEQUENCE,
            "the list closes before the transport assigns the sequence"
        );

        assert_eq!(queue.submit(&batch), Ok(ReplyStatus::Ok));
        assert_eq!(queue.sink().frames.len(), 1, "one frame, not 101");
        let carried = decode_submission_v3(&queue.sink().frames[0]).expect("canonical v3");
        assert_eq!(carried.sequence, 1, "the first real frame is sequence 1");
        assert_eq!(carried.list.map(|list| list.list_id), Some(101));
    }

    /// Two lists closed, one empty and one with a frame: the empty one yields
    /// nothing, so exactly one outbound frame goes out, at sequence 1.
    #[test]
    fn an_empty_list_beside_a_real_list_yields_one_frame() {
        let _serial = serialized();
        reset_sequences(1);
        let mut empty = MetalList::new(201).expect("empty list");
        empty.close().expect("close");
        let mut real = MetalList::new(202).expect("real list");
        real.producer_mut().present(1, 0).expect("present");
        real.close().expect("close");

        let mut batch = empty.take();
        assert!(batch.is_empty(), "the empty close carried no submission");
        batch.extend(real.take());
        assert_eq!(batch.len(), 1);

        let mut queue = NativeQueue::new(RecordingSink::default());
        assert_eq!(queue.submit(&batch), Ok(ReplyStatus::Ok));
        assert_eq!(queue.sink().frames.len(), 1);
        let carried = decode_submission_v3(&queue.sink().frames[0]).expect("canonical v3");
        assert_eq!(carried.sequence, 1);
        assert_eq!(carried.list.map(|list| list.list_id), Some(202));
    }

    /// A list reset before execution drops an unassigned submission, so the
    /// frames it would have preceded still take sequences 1, 2, ... in order.
    #[test]
    fn a_list_reset_before_execution_keeps_the_ledger_dense() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(RecordingSink::default());
        let mut list = MetalList::new(301).expect("list");

        list.producer_mut().present(1, 0).expect("present");
        list.close().expect("close");
        // The title resets the list before `ExecuteCommandLists` takes it:
        // `reset` drops the pending submission, which never held a sequence
        // (its debug assertion checks that ordering).
        list.reset().expect("reset drops the pending frame");
        assert!(list.take().is_empty());

        for _ in 0..2 {
            list.producer_mut().present(1, 0).expect("present");
            list.close().expect("close");
            assert_eq!(queue.submit(&list.take()), Ok(ReplyStatus::Ok));
        }

        assert_eq!(queue.sink().frames.len(), 2);
        let sequences: Vec<u32> = queue
            .sink()
            .frames
            .iter()
            .map(|bytes| decode_submission_v3(bytes).expect("canonical v3").sequence)
            .collect();
        assert_eq!(
            sequences,
            vec![1, 2],
            "the dropped frame took no sequence from the ledger"
        );
    }

    /// A refused send leaves the frames queued with the sequences they were
    /// assigned at enqueue; the retry sends those same sequences once, in
    /// order, and never renumbers them.
    #[test]
    fn a_refused_send_retries_the_same_sequences_without_renumbering() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(FlakySink {
            failures: 1,
            frames: Vec::new(),
        });
        let batch = [submission(0, 401, 1), submission(0, 402, 2)];

        assert!(matches!(
            queue.submit(&batch),
            Err(NativeQueueError::Transport(_))
        ));
        assert_eq!(queue.pending(), 2, "the refused frames stay queued");
        assert_eq!(queue.sent(), 0);
        assert!(queue.sink().frames.is_empty());

        assert_eq!(queue.flush(), Ok(vec![ReplyStatus::Ok, ReplyStatus::Ok]));
        assert_eq!(queue.sent(), 2);
        let decoded: Vec<SubmissionV3> = queue
            .sink()
            .frames
            .iter()
            .map(|bytes| decode_submission_v3(bytes).expect("canonical v3"))
            .collect();
        assert_eq!(
            decoded.iter().map(|frame| frame.sequence).collect::<Vec<_>>(),
            vec![1, 2],
            "the retry keeps the sequences assigned at enqueue"
        );
        assert_eq!(
            decoded
                .iter()
                .map(|frame| frame.list.map(|list| list.list_id))
                .collect::<Vec<_>>(),
            vec![Some(401), Some(402)],
            "the retry keeps the recorded order"
        );

        assert_eq!(queue.flush(), Ok(Vec::new()), "nothing is sent twice");
        assert_eq!(queue.sink().frames.len(), 2);
    }
}
