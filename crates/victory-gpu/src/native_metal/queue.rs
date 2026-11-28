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
//! [`ReplyStatus::AlreadyExecuted`] is surfaced, never re-sent. A host refusal
//! (`ExecutionFailed`/`UnknownHandle`) is a spent sequence too: the host
//! answered it and moved on, so the frame is consumed and the refusal is
//! returned as an ordinary reply — a refused pipeline never re-sends its
//! sequence and never desynchronises the next real submission.
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
use victory_metal::wire_v3::{CommandV3, SubmissionV3, encode_submission_v3};

use crate::backend::GpuBackend;
#[cfg(any(windows, test, target_os = "macos"))]
use crate::native_metal::bindings::{self, GuestRange, GuestRangeResolver, ResolveError};
use crate::native_metal::client::{LiveConfig, WireV3Client};
#[cfg(feature = "diagnostics")]
use crate::native_metal::present::guard_native_only;
use crate::native_metal::present::{
    InFlightFrame, NativeSwapchain, PresentError, PresentedSurface, SurfaceFormat, SurfacePresenter,
    present_surface,
};
use crate::native_metal::producer::{self, ProducerError};
use crate::native_metal::telemetry::submit_refusals;

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
    /// The versioned root-argument plan could not be encoded or framed before
    /// anything was sent. Nothing reached the host, so the frame stays queued.
    Plan(String),
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
            Self::Plan(why) => write!(f, "native queue root-argument plan: {why}"),
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
            Self::Encode(WireError::ZeroSequence | WireError::UnsupportedVersion(_))
            | Self::Plan(_) => crate::object::E_INVALIDARG,
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
    /// Encoded `VRAB` root-argument plan of the recorded arguments.
    ///
    /// Every frame this queue enqueues carries a plan section: an empty plan is
    /// the versioned header alone, never `None`. `None` remains only for a
    /// frame built by an older path that predates the section.
    pub plan: Option<Vec<u8>>,
    /// The first work/handle-bearing command, for a refusal diagnostic.
    pub(crate) probe: Option<RefusalProbe>,
    /// The registration epoch this frame must not overtake, when it selects a
    /// pipeline (see [`crate::native_metal::pso::wait_for_registration_epoch`]).
    pub(crate) required_epoch: Option<u64>,
}

/// The first command of a submission that names what the list was doing.
///
/// This is guest-side attribution only: the Wire v3 `UnknownHandle` reply
/// carries no per-handle site, so the queue records the first draw, dispatch,
/// pipeline selection, binding, copy or present command and its exact index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RefusalProbe {
    /// Zero-based index of the command in the submission.
    pub index: u32,
    /// Its wire opcode.
    pub opcode: u16,
    /// Its command class, one of `telemetry::refusal_sites`.
    pub class: &'static str,
}

/// The command class of one command, if it is a work/handle-bearing command.
fn command_class(command: &CommandV3) -> Option<&'static str> {
    use crate::native_metal::telemetry::refusal_sites as sites;
    match command {
        CommandV3::Draw(_)
        | CommandV3::DrawIndexed(_)
        | CommandV3::DrawInstanced(_)
        | CommandV3::DrawIndexedInstanced(_) => Some(sites::DRAW),
        CommandV3::Dispatch1d(_) | CommandV3::Dispatch3d(_) => Some(sites::DISPATCH),
        CommandV3::ExecuteIndirectDraw(_)
        | CommandV3::ExecuteIndirectDrawIndexed(_)
        | CommandV3::ExecuteIndirectDispatch(_)
        | CommandV3::ExecuteIndirectDispatchMesh(_) => Some(sites::INDIRECT),
        CommandV3::SetGraphicsPipeline(_) | CommandV3::SetPipeline(_) => {
            Some(sites::PIPELINE_GRAPHICS)
        }
        CommandV3::SetMeshPipeline(_) => Some(sites::PIPELINE_MESH),
        CommandV3::SetComputePipeline(_) => Some(sites::PIPELINE_COMPUTE),
        CommandV3::SetVertexBuffers(_)
        | CommandV3::SetIndexBuffer(_)
        | CommandV3::SetRootBufferView(_)
        | CommandV3::SetRootConstants(_)
        | CommandV3::SetDescriptorTable(_)
        | CommandV3::BeginRenderPass(_) => Some(sites::BINDING),
        CommandV3::CopyBuffer(_)
        | CommandV3::CopyTexture(_)
        | CommandV3::CopyTextureRegion(_)
        | CommandV3::CopyBufferToTexture(_)
        | CommandV3::ResolveSubresource(_) => Some(sites::COPY),
        CommandV3::Present(_) => Some(sites::PRESENT),
        _ => None,
    }
}

/// The first work/handle-bearing command of `commands`, in stream order.
fn first_refusal_probe(commands: &[CommandV3]) -> Option<RefusalProbe> {
    commands
        .iter()
        .enumerate()
        .find_map(|(index, command)| {
            command_class(command).map(|class| RefusalProbe {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                opcode: command.opcode(),
                class,
            })
        })
}

/// Whether any command selects a host pipeline.
fn selects_a_pipeline(commands: &[CommandV3]) -> bool {
    commands.iter().any(|command| {
        matches!(
            command,
            CommandV3::SetGraphicsPipeline(_)
                | CommandV3::SetMeshPipeline(_)
                | CommandV3::SetComputePipeline(_)
                | CommandV3::SetPipeline(_)
        )
    })
}

/// Where the outbound submissions go.
///
/// This crate opens no socket (the `WIRE-V3-HOST-BRIDGE` task owns the host
/// side and its transport): a sink carries one already-encoded v3 frame and
/// answers with the host reply status.
pub trait V3Sink {
    /// Carries one encoded version 3 submission.
    fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError>;

    /// Carries one encoded version 3 submission and its optional encoded
    /// root-argument plan (`VRAB` bytes).
    ///
    /// The plan is a versioned trailing section of the same request; the
    /// submission frame itself is carried unchanged. The default forwards to
    /// [`Self::send`], which is exactly what a transport that predates the plan
    /// does: the frame goes out as before and the host keeps its legacy wire-slot
    /// lookup. The live [`NativeSink`] and [`WireV3Client`] carry the section.
    fn send_with_plan(
        &mut self,
        bytes: &[u8],
        _plan: Option<&[u8]>,
    ) -> Result<ReplyStatus, NativeQueueError> {
        self.send(bytes)
    }
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

    fn send_with_plan(
        &mut self,
        bytes: &[u8],
        plan: Option<&[u8]>,
    ) -> Result<ReplyStatus, NativeQueueError> {
        match self {
            Self::Disconnected(sink) => sink.send_with_plan(bytes, plan),
            Self::Live(client) => client.send_with_plan(bytes, plan),
        }
    }
}

/// The ordered outbound queue for one guest session.
///
/// Submissions keep the order a closed list produced them in; nothing is
/// merged, reordered or dropped on success. A transport failure leaves the
/// unsent frames at the front of the queue (the failure is reported, not
/// hidden), while a host reply (a status, including a refusal) is always
/// consumed exactly once and never re-sent.
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
    /// sequence keeps it (a frame a caller built, or one the queue kept after
    /// a transport failure), so a resend after a transport error never
    /// renumbers. Nothing is coalesced: one [`SubmissionV3`]
    /// becomes one outbound frame, encoded with the canonical v3 codec. An
    /// empty batch is refused with [`NativeQueueError::Empty`]. Nothing is
    /// queued if any frame fails to encode, so a refused batch never leaves a
    /// partial prefix behind.
    pub fn enqueue(&mut self, submissions: &[SubmissionV3]) -> Result<usize, NativeQueueError> {
        self.enqueue_with_root_arguments(submissions, &[])
    }

    /// [`Self::enqueue`] with the root-argument plan each submission recorded.
    ///
    /// `plans` is index-aligned with `submissions`: the plan at index `i`
    /// belongs to the submission at index `i`. Every frame carries a plan
    /// section: a missing or empty plan is encoded as the empty `VRAB` header,
    /// which the host decodes as "no entries", so a batch that binds no root
    /// constant still carries the pipeline's declared plan. A plan the encoder
    /// refuses fails the whole batch with [`NativeQueueError::Plan`] before
    /// anything is queued, so a refused batch never leaves a partial prefix
    /// behind.
    pub fn enqueue_with_root_arguments(
        &mut self,
        submissions: &[SubmissionV3],
        plans: &[&[bindings::RootArgument]],
    ) -> Result<usize, NativeQueueError> {
        if submissions.is_empty() {
            return Err(NativeQueueError::Empty(
                "ExecuteCommandLists carried no submission",
            ));
        }
        let mut encoded = Vec::with_capacity(submissions.len());
        for (index, submission) in submissions.iter().enumerate() {
            if submission.commands.is_empty() {
                // An empty stream is not a frame: it is never handed to the
                // host and must not consume a sequence the host will never
                // see, or every later frame is a protocol violation.
                continue;
            }
            // Every frame carries a plan section, even when the list recorded no
            // root argument: an empty plan is the seven-byte `VRAB` header, which
            // the host decodes as "plan present, no entries". A frame without the
            // section remains legal for older guests, but this guest never sends
            // one, so a batch that binds no root constant still describes its
            // pipeline's declared root arguments.
            let plan = Some(
                bindings::encode_root_arguments(plans.get(index).copied().unwrap_or(&[]))
                    .map_err(|error| NativeQueueError::Plan(error.to_string()))?,
            );
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
            // A frame that selects a pipeline may not overtake a registration
            // that was already in flight when it was enqueued. The requirement
            // is the epoch every such registration settles at.
            let required_epoch = selects_a_pipeline(&submission.commands).then(|| {
                let registration = crate::native_metal::pso::registration_snapshot();
                registration
                    .epoch
                    .saturating_add(u64::try_from(registration.in_flight).unwrap_or(u64::MAX))
            });
            self.highest_sequence = self.highest_sequence.max(submission.sequence);
            encoded.push(OutboundFrame {
                sequence: submission.sequence,
                list: submission.list,
                bytes,
                plan,
                probe: first_refusal_probe(&submission.commands),
                required_epoch,
            });
        }
        let count = encoded.len();
        self.outbound.extend(encoded);
        Ok(count)
    }

    /// Sends every queued frame in order, exactly once.
    ///
    /// Returns one host status per frame, in send order. A **transport error**
    /// stops the pump and leaves the failed frame (and everything behind it)
    /// queued, because the host never saw it: the next flush carries the same
    /// sequences again, once. A **host refusal**
    /// ([`NativeQueueError::Refused`]: `ExecutionFailed` or `UnknownHandle`)
    /// is not a transport failure: the host answered the frame and advanced
    /// its ledger past that sequence, so the frame is consumed here and is
    /// never re-sent. The remaining frames are still in sequence and keep
    /// going out; the first refusal is returned to the caller after the pump
    /// drains, so a refused frame can never leave the host expecting a
    /// sequence the guest will not send.
    pub fn flush(&mut self) -> Result<Vec<ReplyStatus>, NativeQueueError> {
        let mut statuses = Vec::with_capacity(self.outbound.len());
        let mut refusal = None;
        while let Some(frame) = self.outbound.pop_front() {
            // A frame that selects a pipeline waits on the registration epoch
            // it observed at enqueue: it never reaches the render socket ahead
            // of a registration that was already in flight. The guest is
            // synchronous today (no list can name an id before the kind 129
            // reply), so this is the invariant guard, not a reordering.
            if let Some(required) = frame.required_epoch {
                #[cfg(test)]
                flush_reached_pipeline_wait();
                crate::native_metal::pso::wait_for_registration_epoch(required);
            }
            match self.sink.send_with_plan(&frame.bytes, frame.plan.as_deref()) {
                Ok(ReplyStatus::UnknownHandle) => {
                    self.sent += 1;
                    note_refused_site(&frame, None);
                    statuses.push(ReplyStatus::UnknownHandle);
                }
                Ok(status) => {
                    self.sent += 1;
                    statuses.push(status);
                }
                Err(error) => match &error {
                    NativeQueueError::Refused(reason) => {
                        // The host consumed the sequence and wrote a reply. The
                        // frame is spent: keeping it would re-send an already
                        // consumed sequence and poison the session.
                        self.sent += 1;
                        if is_unknown_handle_refusal(reason) {
                            note_refused_site(&frame, Some(reason.as_str()));
                        }
                        refusal.get_or_insert(error);
                    }
                    _ => {
                        self.outbound.push_front(frame);
                        return Err(error);
                    }
                },
            }
        }
        match refusal {
            // The statuses of the frames behind a refusal are still consumed;
            // the refusal is what the caller must see.
            Some(error) => Err(error),
            None => Ok(statuses),
        }
    }

    /// Queues one closed list's submissions and sends them in recorded order.
    ///
    /// Returns the batch's host reply: the first status that is not
    /// [`ReplyStatus::Ok`] when the host answered with a status, or the host's
    /// [`NativeQueueError::Refused`] when it answered `ExecutionFailed` /
    /// `UnknownHandle`. A replayed list's `AlreadyExecuted` is therefore
    /// surfaced; the frame is not sent a second time. A refused frame is
    /// consumed the same way: its refusal is returned to the caller and its
    /// sequence is spent, so the next frame takes the next sequence with no
    /// hole.
    pub fn submit(
        &mut self,
        submissions: &[SubmissionV3],
    ) -> Result<ReplyStatus, NativeQueueError> {
        self.submit_measured(submissions)
            .map(|(status, _timing)| status)
    }

    /// [`Self::submit`] with each submission's root-argument plan.
    ///
    /// The reply rule is identical; the plan is encoded at enqueue and carried
    /// beside the unchanged frame.
    pub fn submit_with_root_arguments(
        &mut self,
        submissions: &[SubmissionV3],
        plans: &[&[bindings::RootArgument]],
    ) -> Result<ReplyStatus, NativeQueueError> {
        self.submit_measured_with_root_arguments(submissions, plans)
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
        self.submit_measured_with_root_arguments(submissions, &[])
    }

    /// [`Self::submit_measured`] with each submission's root-argument plan.
    pub fn submit_measured_with_root_arguments(
        &mut self,
        submissions: &[SubmissionV3],
        plans: &[&[bindings::RootArgument]],
    ) -> Result<(ReplyStatus, SubmitTiming), NativeQueueError> {
        let started = Instant::now();
        self.enqueue_with_root_arguments(submissions, plans)?;
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

/// A test hook: a sender notified when a flush reaches a pipeline-epoch wait.
///
/// The barrier is observed, not guessed at: a test registers a channel and
/// waits for the notification before completing the in-flight registration, so
/// the ordering proof does not depend on thread scheduling.
#[cfg(test)]
static PIPELINE_WAIT_NOTIFY: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>> =
    std::sync::Mutex::new(None);

/// Installs (or clears) the pipeline-wait notification channel.
#[cfg(test)]
pub(crate) fn notify_on_pipeline_wait(sender: Option<std::sync::mpsc::Sender<()>>) {
    *PIPELINE_WAIT_NOTIFY
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = sender;
}

#[cfg(test)]
fn flush_reached_pipeline_wait() {
    if let Some(sender) = PIPELINE_WAIT_NOTIFY
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()
    {
        let _ = sender.send(());
    }
}

/// Whether a host refusal text names `UnknownHandle`.
///
/// The Wire v3 reply status is not carried on [`NativeQueueError::Refused`]
/// (only the formatted reason is), so the client's stable `Debug` spelling is
/// the signal. `ExecutionFailed` refusals are not recorded as site details.
fn is_unknown_handle_refusal(reason: &str) -> bool {
    reason.contains("answered UnknownHandle")
}

/// Records one refused frame's guest-side site and logs its stable line.
///
/// `reason` is the host text when the transport carried it; an `UnknownHandle`
/// reply has no trailer, so the record says that by keeping the reason empty.
fn note_refused_site(frame: &OutboundFrame, reason: Option<&str>) {
    use crate::native_metal::telemetry::{self, SubmitRefusalSite, refusal_sites};

    let probe = frame.probe;
    let site = SubmitRefusalSite {
        sequence: frame.sequence,
        list_id: frame.list.map_or(0, |list| list.list_id),
        list_generation: frame.list.map_or(0, |list| list.list_generation),
        command_index: probe.map(|probe| probe.index),
        opcode: probe.map(|probe| probe.opcode),
        class: probe.map_or(refusal_sites::UNCLASSIFIED, |probe| probe.class),
        reason: reason.unwrap_or_default().to_string(),
        registration_epoch: crate::native_metal::pso::registration_epoch(),
    };
    let record = telemetry::note_submit_refusal_site(site);
    tracing::warn!("{}", telemetry::refusal_site_line(&record));
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

/// The host-supplied refusal detail, when the transport carried one.
///
/// The live Wire v3 client puts the host's own text into
/// [`NativeQueueError::Refused`]. The reply frame itself is a fixed status-only
/// payload, so this is where a host reason enters the guest; a transport error
/// has its own text but no host reason.
pub(crate) fn refusal_detail(error: &NativeQueueError) -> Option<&str> {
    match error {
        NativeQueueError::Refused(reason) if !reason.is_empty() => Some(reason.as_str()),
        _ => None,
    }
}

/// The single line the guest logs when a native submission is refused.
///
/// This is deliberately built from [`refusal_detail`] so a host reason is never
/// dropped on the floor between the client and the `native queue refused:`
/// warning.
pub(crate) fn refusal_warning(error: &NativeQueueError) -> String {
    match refusal_detail(error) {
        Some(reason) => format!("native queue refused: {reason}"),
        None => error.to_string(),
    }
}

/// Whether a failed send is a host/GPU refusal, which the refusal counter
/// records. Every failed submission is counted; this predicate is the explicit
/// classification the warning uses to distinguish a host reason from a
/// transport failure.
pub(crate) fn counts_as_refused_submission(error: &NativeQueueError) -> bool {
    matches!(error, NativeQueueError::Refused(_))
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

/// [`dispatch`] with each submission's root-argument plan.
pub fn dispatch_with_root_arguments<S: V3Sink>(
    backend: GpuBackend,
    queue: &mut NativeQueue<S>,
    submissions: &[SubmissionV3],
    plans: &[&[bindings::RootArgument]],
) -> Result<NativeDispatch, NativeQueueError> {
    dispatch_measured_with_root_arguments(backend, queue, submissions, plans)
        .map(|(dispatch, _timing)| dispatch)
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
    dispatch_measured_with_root_arguments(backend, queue, submissions, &[])
}

/// [`dispatch_measured`] with each submission's root-argument plan.
pub fn dispatch_measured_with_root_arguments<S: V3Sink>(
    backend: GpuBackend,
    queue: &mut NativeQueue<S>,
    submissions: &[SubmissionV3],
    plans: &[&[bindings::RootArgument]],
) -> Result<(NativeDispatch, Option<SubmitTiming>), NativeQueueError> {
    match backend {
        GpuBackend::D3d12 => Ok((NativeDispatch::Host, None)),
        GpuBackend::Metal => {
            let (status, timing) = queue.submit_measured_with_root_arguments(submissions, plans)?;
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
    deferred_batches_reset();
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
    register_texture_with_reason(
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
    )
    .map(|(accepted, _)| accepted)
}

/// [`register_texture`] with the host's bounded refusal reason.
///
/// `Ok((true, None))` registered; `Ok((false, Some(reason)))` is the host's
/// named refusal; `Ok((false, None))` is an older host that answers with only
/// the classic status byte.
#[cfg(any(windows, test, target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn register_texture_with_reason(
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
) -> Result<(bool, Option<String>), NativeQueueError> {
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
        return Ok((true, None));
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
    let reason = if accepted {
        // Remember the host format so a headless surface installed from the
        // presented target keeps its own format (FP16 stays FP16).
        note_texture_format(id, format);
        None
    } else {
        client.last_texture_refusal_reason().map(str::to_string)
    };
    Ok((accepted, reason))
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

/// Why a guest buffer's share allocation could not be registered on first
/// bind.
///
/// Every variant is a named refusal: a missing resolver, an unresolvable guest
/// range, a missing or empty share file, or a kind 2 id the host's strictly
/// increasing ledger cannot accept. No variant stands for a silent skip, and
/// none is turned into a D3DMetal fallback.
#[cfg(any(windows, test, target_os = "macos"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferRegistrationError {
    /// No [`GuestRangeResolver`] is installed in this process.
    NoResolver,
    /// The guest address or length cannot name a share range at all.
    InvalidBinding {
        /// Guest GPU virtual address the binding named.
        address: u64,
        /// Byte length the binding named.
        len: u64,
        /// Why the pair cannot be resolved.
        reason: &'static str,
    },
    /// The resolver could not map the guest range to a share allocation.
    Unresolved {
        /// Guest GPU virtual address the binding named.
        address: u64,
        /// Byte length the binding named.
        len: u64,
        /// The resolver's own reason.
        reason: ResolveError,
    },
    /// The resolved range is not a share allocation a kind 2 frame can carry.
    InvalidRange {
        /// Allocation id the range named.
        allocation_id: u64,
        /// Why the range cannot be registered.
        reason: &'static str,
    },
    /// The session's share directory could not be scanned.
    ShareDirectory(String),
    /// The allocation has no `<share dir>/{id:016x}.bin` file.
    MissingShareBacking {
        /// Allocation id the resolver named.
        allocation_id: u64,
        /// The file the host would open.
        path: String,
    },
    /// The allocation's share file is empty, so no length can be named.
    EmptyShareBacking {
        /// Allocation id the resolver named.
        allocation_id: u64,
        /// The file the host would open.
        path: String,
    },
    /// The host's kind 2 ledger has already accepted a higher id.
    ///
    /// The host refuses (and closes the session on) an id that does not
    /// increase, so the guest refuses locally instead of sending it.
    Decreasing {
        /// Allocation id the registration named.
        allocation_id: u64,
        /// The highest id the host accepted so far.
        last_registered: u64,
    },
    /// The registration frame was refused by the host or the transport.
    Queue(NativeQueueError),
}

#[cfg(any(windows, test, target_os = "macos"))]
impl std::fmt::Display for BufferRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoResolver => write!(
                f,
                "no guest range resolver: the buffer's share allocation cannot be named"
            ),
            Self::InvalidBinding {
                address,
                len,
                reason,
            } => write!(
                f,
                "guest binding 0x{address:x}+{len} cannot name a share allocation: {reason}"
            ),
            Self::Unresolved {
                address,
                len,
                reason,
            } => write!(
                f,
                "guest binding 0x{address:x}+{len} does not resolve to a share allocation: {reason}"
            ),
            Self::InvalidRange {
                allocation_id,
                reason,
            } => write!(f, "allocation {allocation_id} cannot be registered: {reason}"),
            Self::ShareDirectory(why) => write!(
                f,
                "the session's share directory is unusable: {why}"
            ),
            Self::MissingShareBacking {
                allocation_id,
                path,
            } => write!(
                f,
                "allocation {allocation_id} has no share file ({path}); the guest never creates \
                 the host's backing"
            ),
            Self::EmptyShareBacking {
                allocation_id,
                path,
            } => write!(f, "allocation {allocation_id}'s share file ({path}) is empty"),
            Self::Decreasing {
                allocation_id,
                last_registered,
            } => write!(
                f,
                "allocation {allocation_id} is below the last host registration \
                 {last_registered}; the host accepts only increasing kind 2 ids"
            ),
            Self::Queue(error) => write!(f, "{error}"),
        }
    }
}

#[cfg(any(windows, test, target_os = "macos"))]
impl std::error::Error for BufferRegistrationError {}

#[cfg(any(windows, test, target_os = "macos"))]
impl From<BufferRegistrationError> for NativeQueueError {
    fn from(error: BufferRegistrationError) -> Self {
        match error {
            // A host or transport refusal keeps its own shape (`Refused` vs
            // `Transport`) so the queue's refusal classification still holds.
            BufferRegistrationError::Queue(error) => error,
            other => NativeQueueError::Refused(other.to_string()),
        }
    }
}

/// Makes sure the host has `allocation_id` registered, registering it and
/// every lower id the share directory holds, ascending (idempotent).
///
/// This is the entry form of the lazy first-bind path: a descriptor table, root
/// view, vertex/index buffer or indirect binding that already resolved its
/// guest range names the share allocation id here. The length always comes from
/// the share file that id names; nothing is fabricated and there is no size
/// cap, so a 182 MB heap allocation registers exactly like a 352 B one.
///
/// The share directory is scanned on every call, so an allocation whose
/// `<share dir>/{id:016x}.bin` only appears after an earlier bind is discovered
/// on the next one; the guest never assumes or creates the host's backing. A
/// missing file, an empty file, a missing resolver or a non-increasing id is a
/// named [`BufferRegistrationError`].
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_allocation_entry(allocation_id: u64) -> Result<(), BufferRegistrationError> {
    if allocation_id == 0 {
        return Err(BufferRegistrationError::InvalidRange {
            allocation_id,
            reason: "a buffer registration needs a nonzero allocation id",
        });
    }
    if session_allocation_registered(allocation_id) {
        // Idempotent: a second bind of the same allocation issues no frame.
        return Ok(());
    }
    let last_registered = last_registered_allocation();
    if allocation_id < last_registered {
        return Err(BufferRegistrationError::Decreasing {
            allocation_id,
            last_registered,
        });
    }
    // Discover the session's share files now: an allocation created after an
    // earlier bind (its file appears later) is registered on this call, and no
    // cached listing can hide one.
    let pending = crate::native_metal::pso::shares_up_to(allocation_id)
        .map_err(BufferRegistrationError::ShareDirectory)?;
    if !pending.iter().any(|(id, _)| *id == allocation_id) {
        return Err(share_backing_error(allocation_id));
    }
    for (id, byte_len) in pending {
        if session_allocation_registered(id) {
            continue;
        }
        if id < last_registered {
            // A file for an older id appeared after the host's ledger passed
            // it. The kind 2 form cannot carry it and sending a lower id would
            // close the session, so this is refused by name instead.
            return Err(BufferRegistrationError::Decreasing {
                allocation_id: id,
                last_registered,
            });
        }
        register_buffer(id, byte_len).map_err(BufferRegistrationError::Queue)?;
    }
    // A new allocation may complete a title-memory map whose chunks arrived
    // before its kind 2 frame; send it now that every id exists.
    let _ = sync_title_memory_map();
    Ok(())
}

/// Makes sure the host has `allocation_id` registered (idempotent).
///
/// The id comes from the `GuestRangeResolver` seam and the length from the
/// share file that id names; nothing is fabricated. Allocations that already
/// exist at or below the requested id are registered first, ascending, because
/// the host refuses (and closes the session on) a kind 2 id that does not
/// increase. This is [`register_allocation_entry`] with the refusal flattened
/// into the queue's error type, kept for the create-time registration path.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn ensure_allocation(allocation_id: u64) -> Result<bool, NativeQueueError> {
    register_allocation_entry(allocation_id)
        .map(|()| true)
        .map_err(NativeQueueError::from)
}

/// The highest allocation id this session's host has accepted (0: none).
#[cfg(any(windows, test, target_os = "macos"))]
fn last_registered_allocation() -> u64 {
    let queue = native_queue();
    queue
        .sink()
        .client()
        .map(WireV3Client::last_allocation)
        .unwrap_or(0)
}

/// Whether the live session registered `allocation_id` with the host.
///
/// Unlike [`allocation_registered`] this asks the installed client directly,
/// so it does not need the endpoint environment once the session is live.
#[cfg(any(windows, test, target_os = "macos"))]
fn session_allocation_registered(allocation_id: u64) -> bool {
    let queue = native_queue();
    queue
        .sink()
        .client()
        .is_some_and(|client| client.allocation_registered(allocation_id))
}

/// The named error for an id whose share file the scan did not find.
#[cfg(any(windows, test, target_os = "macos"))]
fn share_backing_error(allocation_id: u64) -> BufferRegistrationError {
    let Some(path) = crate::native_metal::pso::allocation_path(allocation_id) else {
        return BufferRegistrationError::ShareDirectory(
            "no VICTORY_METAL_SHARE_DIR: the guest cannot name the allocation's share file"
                .to_string(),
        );
    };
    let shown = path.display().to_string();
    match std::fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
            // A present, nonempty file the scan skipped is not a case the 16
            // hex digit name allows; keep the refusal named either way.
            BufferRegistrationError::MissingShareBacking {
                allocation_id,
                path: shown,
            }
        }
        Ok(_) => BufferRegistrationError::EmptyShareBacking {
            allocation_id,
            path: shown,
        },
        Err(_) => BufferRegistrationError::MissingShareBacking {
            allocation_id,
            path: shown,
        },
    }
}

/// Registers the share allocation one guest buffer address and length name.
///
/// A convenience form of [`register_guest_buffers`] for the single-buffer
/// paths.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_guest_buffer(
    address: u64,
    len: u64,
) -> Result<GuestRange, BufferRegistrationError> {
    let mut ranges = register_guest_buffers(&[(address, len)])?;
    Ok(ranges.remove(0))
}

/// Registers both buffers one buffer copy names, ascending by allocation id.
///
/// The returned ranges keep the caller's `(source, destination)` order, so a
/// copy can build both wire identities from the ranges that were actually
/// registered.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_copy_buffers(
    source: (u64, u64),
    destination: (u64, u64),
) -> Result<(GuestRange, GuestRange), BufferRegistrationError> {
    let mut ranges = register_guest_buffers(&[source, destination])?;
    let destination = ranges.pop().expect("two ranges were resolved");
    let source = ranges.pop().expect("two ranges were resolved");
    Ok((source, destination))
}

/// Registers every share allocation the guest buffers in `ranges` name.
///
/// Each `(address, len)` is resolved exactly once through the installed
/// [`GuestRangeResolver`] (`NoResolver` when none is installed), then the
/// distinct allocations are registered ascending by id because the host accepts
/// only increasing kind 2 ids. The returned ranges keep the caller's order, and
/// a second call for an already registered set issues no frame.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_guest_buffers(
    ranges: &[(u64, u64)],
) -> Result<Vec<GuestRange>, BufferRegistrationError> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    let resolver = bindings::resolver().ok_or(BufferRegistrationError::NoResolver)?;
    let mut resolved = Vec::with_capacity(ranges.len());
    for &(address, len) in ranges {
        resolved.push(resolve_guest_buffer(&*resolver, address, len)?);
    }
    // Register in ascending allocation order even when the caller named the
    // source and destination the other way round.
    let mut order: Vec<usize> = (0..resolved.len()).collect();
    order.sort_by_key(|&index| resolved[index].allocation_id);
    for index in order {
        register_resolved_range(&resolved[index])?;
    }
    Ok(resolved)
}

/// Registers the share allocation a range another helper already resolved
/// names, without resolving (and therefore without pinning) it a second time.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn register_resolved_range(range: &GuestRange) -> Result<(), BufferRegistrationError> {
    checked_guest_range(range)?;
    register_allocation_entry(range.allocation_id)
}

/// Generation of the last title-memory map the guest sent (ADR-055).
static TITLE_MAP_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Sends the guest's VA → share-allocation map when it changed (ADR-055).
///
/// The ranges come from the installed [`GuestRangeResolver::mappings`]: the
/// views the guest actually mapped, never a guess. A process without a
/// resolver, or with no host-visible mapping, sends nothing. The send is
/// opt-out with `VICTORY_METAL_TITLE_MEMORY_MAP=0`, which keeps a new guest
/// on an older host (that host refuses the kind explicitly, it does not
/// misread it).
///
/// Returns whether a frame was sent. A transport problem is returned, but
/// callers on the submission path may treat it as non-fatal: the page table is
/// an optimization for shaders that use title memory, and dropping the frame
/// never changes an existing submission's bytes.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn sync_title_memory_map() -> Result<bool, NativeQueueError> {
    if std::env::var_os("VICTORY_METAL_TITLE_MEMORY_MAP").is_some_and(|value| value == "0") {
        return Ok(false);
    }
    let Some(resolver) = bindings::resolver() else {
        return Ok(false);
    };
    let mappings = resolver.mappings();
    if mappings.is_empty() {
        return Ok(false);
    }
    let ranges: Vec<victory_metal::title_map::Range> = mappings
        .into_iter()
        .map(|mapping| victory_metal::title_map::Range {
            guest_address: mapping.guest_address,
            allocation_id: mapping.allocation_id,
            byte_offset: mapping.byte_offset,
            length: mapping.length,
        })
        .collect();
    let mut queue = native_queue();
    let Some(client) = queue.sink_mut().client_mut() else {
        return Ok(false);
    };
    let generation = u32::try_from(TITLE_MAP_GENERATION.fetch_add(1, Ordering::Relaxed) % u64::from(u32::MAX - 1) + 1)
        .unwrap_or(1);
    client.send_title_memory_map(generation, &ranges)
}

/// Resolves one guest buffer range through `resolver` without pinning it.
#[cfg(any(windows, test, target_os = "macos"))]
fn resolve_guest_buffer(
    resolver: &dyn GuestRangeResolver,
    address: u64,
    len: u64,
) -> Result<GuestRange, BufferRegistrationError> {
    if address == 0 {
        return Err(BufferRegistrationError::InvalidBinding {
            address,
            len,
            reason: "the guest address is zero",
        });
    }
    if len == 0 {
        return Err(BufferRegistrationError::InvalidBinding {
            address,
            len,
            reason: "the guest range is empty",
        });
    }
    let range = resolver
        .resolve(address, len)
        .map_err(|reason| BufferRegistrationError::Unresolved {
            address,
            len,
            reason,
        })?;
    checked_guest_range(&range)?;
    Ok(range)
}

/// Rejects a range no kind 2 registration can name.
#[cfg(any(windows, test, target_os = "macos"))]
fn checked_guest_range(range: &GuestRange) -> Result<(), BufferRegistrationError> {
    if range.allocation_id == 0 {
        return Err(BufferRegistrationError::InvalidRange {
            allocation_id: 0,
            reason: "the resolved range names no share allocation",
        });
    }
    if range.generation != victory_metal::guest_ipc::REGISTRATION_GENERATION {
        return Err(BufferRegistrationError::InvalidRange {
            allocation_id: range.allocation_id,
            reason: "the range's generation is not the host's kind 2 generation",
        });
    }
    if range.size == 0 {
        return Err(BufferRegistrationError::InvalidRange {
            allocation_id: range.allocation_id,
            reason: "the resolved range has no backing bytes",
        });
    }
    Ok(())
}

// ------------------------------------------------- named range refusals

/// The most distinct guest ranges a boot names. A title that binds the same
/// backing for every draw names it once and then counts; a boot that touches
/// more distinct unbacked ranges than this folds the rest into one aggregate
/// so the log cannot grow with the frame count.
pub(crate) const MAX_NAMED_RANGE_REFUSALS: usize = 64;

/// One refused guest range, as the census counts it.
#[derive(Default)]
struct RangeRefusal {
    count: u64,
    branch: &'static str,
}

/// The bounded census over refused guest ranges.
#[derive(Default)]
struct RangeRefusalCensus {
    /// Keyed by the exact `(address, len)` the resolver was asked for.
    named: std::collections::HashMap<(u64, u64), RangeRefusal>,
    /// Distinct ranges seen after the naming cap; only their total is kept.
    overflow: u64,
    total: u64,
}

impl RangeRefusalCensus {
    /// Counts one refusal and logs the first occurrence of a named range.
    ///
    /// The log line carries the branch (`heap_table`, `title_address_table`
    /// or `xmem_share_table`), the count of refusals for that exact range,
    /// the table's range when the branch owns one, and the resolver's own
    /// reason. Repeats only move the count: one line per distinct range, at
    /// most [`MAX_NAMED_RANGE_REFUSALS`] of them, plus one aggregate line.
    fn note(
        &mut self,
        address: u64,
        len: u64,
        branch: crate::resource::RangeRefusalBranch,
        reason: &str,
    ) {
        self.total = self.total.saturating_add(1);
        let key = (address, len);
        if let Some(entry) = self.named.get_mut(&key) {
            entry.count = entry.count.saturating_add(1);
            // Logarithmic reminders: bounded by the count's bit width, never
            // one line per draw.
            if entry.count.is_power_of_two() {
                tracing::warn!(
                    address = format_args!("{address:#x}"),
                    bytes = len,
                    branch = entry.branch,
                    count = entry.count,
                    reason,
                    "VICTORY_BACKEND=metal: a guest range is still refused"
                );
            }
            return;
        }
        if self.named.len() >= MAX_NAMED_RANGE_REFUSALS {
            let first = self.overflow == 0;
            self.overflow = self.overflow.saturating_add(1);
            if first {
                tracing::warn!(
                    named = MAX_NAMED_RANGE_REFUSALS,
                    "VICTORY_BACKEND=metal: further disturbed guest ranges are counted, not named"
                );
            }
            return;
        }
        self.named.insert(
            key,
            RangeRefusal {
                count: 1,
                branch: branch.name(),
            },
        );
        let range = branch.range();
        tracing::warn!(
            address = format_args!("{address:#x}"),
            bytes = len,
            branch = branch.name(),
            heap = ?range,
            reason,
            "VICTORY_BACKEND=metal: a guest range is refused"
        );
    }

    /// The aggregate numbers: `(named, distinct refusals, overflow, total)`.
    #[cfg(test)]
    fn counts(&self) -> (usize, u64, u64, u64) {
        (
            self.named.len(),
            self.named.values().map(|entry| entry.count).sum(),
            self.overflow,
            self.total,
        )
    }
}

static RANGE_REFUSALS: std::sync::LazyLock<std::sync::Mutex<RangeRefusalCensus>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(RangeRefusalCensus::default()));

/// Counts one refused guest range against the bounded census, naming the
/// branch that rejected it on the first occurrence.
///
/// This is the single place a refusal is counted, so a range refused for
/// every draw costs one line and a running count, never a line per draw.
#[cfg(any(windows, test, target_os = "macos"))]
pub(crate) fn note_refused_guest_range(
    address: u64,
    len: u64,
    branch: crate::resource::RangeRefusalBranch,
    reason: impl std::fmt::Display,
) {
    let reason = reason.to_string();
    RANGE_REFUSALS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .note(address, len, branch, &reason);
}

/// The census numbers, for the tests: `(named, refusals they count, overflow,
/// total)`.
#[cfg(test)]
pub(crate) fn guest_range_refusal_counts() -> (usize, u64, u64, u64) {
    RANGE_REFUSALS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .counts()
}

/// Empties the range refusal census: tests start each case from a known
/// state.
#[cfg(test)]
pub(crate) fn reset_guest_range_refusals() {
    *RANGE_REFUSALS
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = RangeRefusalCensus::default();
}

/// The installed resolver, wrapped so a range inside memory this backend
/// created is refused by name.
///
/// The share resolver only knows the `xmem` tables: a D3D12 heap allocated
/// from host memory, or a resource at a title-mapped address, has no share
/// file and would otherwise be refused as `no shared xmem range`, which reads
/// like an `xmem` mapping bug. The wrapper leaves every range the share
/// table resolves untouched and only renames the refusal of a range this
/// backend's tables cover (see [`crate::resource::guest_range_branch`]).
#[cfg(any(windows, test, target_os = "macos"))]
struct CreatedRangeResolver {
    inner: std::sync::Arc<dyn GuestRangeResolver>,
}

#[cfg(any(windows, test, target_os = "macos"))]
impl GuestRangeResolver for CreatedRangeResolver {
    fn resolve(&self, guest_address: u64, len: u64) -> Result<GuestRange, ResolveError> {
        match self.inner.resolve(guest_address, len) {
            Ok(range) => Ok(range),
            Err(error) => {
                let branch = crate::resource::guest_range_branch(guest_address, len);
                let Some(refusal) = branch.refusal() else {
                    // The share table's own refusal is already named; still
                    // count it once per distinct range.
                    note_refused_guest_range(guest_address, len, branch, &error);
                    return Err(error);
                };
                note_refused_guest_range(guest_address, len, branch, &error);
                Err(ResolveError::Unsupported(refusal))
            }
        }
    }

    fn retain(&self, range: &GuestRange) -> Result<u64, ResolveError> {
        self.inner.retain(range)
    }

    fn retire(&self, token: u64) -> Result<(), ResolveError> {
        self.inner.retire(token)
    }
}

/// Installs [`CreatedRangeResolver`] over the process-wide resolver, once.
///
/// Called when this backend creates its first heap or title-mapped range,
/// which the console device creation has already followed by installing the
/// share resolver. With no resolver installed (the D3D12 reference backend)
/// this is a no-op and a later call can still wrap the resolver that appears.
#[cfg(any(windows, test, target_os = "macos"))]
pub(crate) fn install_created_range_resolver() {
    static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }
    match bindings::resolver() {
        Some(inner) => {
            bindings::set_resolver_arc(std::sync::Arc::new(CreatedRangeResolver { inner }));
        }
        None => INSTALLED.store(false, Ordering::Release),
    }
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
    let registration = crate::native_metal::pso::begin_pipeline_registration();
    let result = client.register_pipeline(ticket, entry);
    if result.is_ok() {
        registration.succeeded();
    }
    result
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
    let registration = crate::native_metal::pso::begin_pipeline_registration();
    let result = client.register_pipeline_v2(kind, &wire, state, abi, root_signature);
    if result.is_ok() {
        registration.succeeded();
    }
    result
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
///
/// A batch that carries no frame at all is not a submission: it is refused with
/// [`NativeQueueError::Empty`] before anything is encoded, sent or counted, so
/// an empty list executed after a `Close` that carried no command buffer can
/// never inflate the refusal total. Every counted refusal passes a static
/// reason from [`crate::native_metal::telemetry::SUBMIT_REFUSAL_NAMES`].
#[cfg(any(windows, test, target_os = "macos"))]
pub fn submit_recorded_batch(
    submissions: &[SubmissionV3],
    record_us: u64,
) -> Result<NativeDispatch, NativeQueueError> {
    submit_recorded_batch_with_root_arguments(submissions, &[], record_us)
}

/// [`submit_recorded_batch`] with the root-argument plan each submission
/// recorded.
///
/// The plan is encoded at enqueue and carried beside the unchanged frame, so a
/// title root-parameter index reaches the host together with the register space,
/// kind and register the translated root signature actually declares. Every
/// counter, refusal name and empty-batch rule is the same.
#[cfg(any(windows, test, target_os = "macos"))]
pub fn submit_recorded_batch_with_root_arguments(
    submissions: &[SubmissionV3],
    plans: &[&[bindings::RootArgument]],
    record_us: u64,
) -> Result<NativeDispatch, NativeQueueError> {
    // An empty stream is not a frame: nothing reaches the host, nothing is
    // counted and nothing consumes a sequence. Counting it as a refused
    // submission is exactly the unnamed per-boot refusal this guards against.
    if !carries_a_frame(submissions) {
        tracing::debug!(
            submissions = submissions.len(),
            "VICTORY_BACKEND=metal: the batch carried no command buffer; it is not a submission"
        );
        return Err(NativeQueueError::Empty(
            "the batch carried no command buffer",
        ));
    }

    // The backend decision was made by `crate::queue::execute` before it
    // called the native path, so the environment is the only gate left.
    let attached = match ensure_live(LiveConfig::from_env) {
        Ok(attached) => attached,
        Err(error) => {
            tracing::warn!(%error, "VICTORY_BACKEND=metal: the native queue has no live sink");
            false
        }
    };

    // ADR-055: keep the host's title-memory page table current before the
    // batch names it. The map is metadata; a failure here never changes the
    // submission's bytes and is reported only in the host's own counters.
    let _ = sync_title_memory_map();

    let mut queue = native_queue();
    let outcome = dispatch_measured_with_root_arguments(
        GpuBackend::Metal,
        &mut queue,
        submissions,
        plans,
    );
    // The sequences were assigned at enqueue, so the callers' submissions may
    // still be unassigned here; the queue knows the last one it handed out.
    let highest_sequence = queue.highest_sequence();
    drop(queue);

    match outcome {
        Ok((dispatch, timing)) => {
            // A carried batch ends any no-client deferral streak.
            deferred_batches_reset();
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
                crate::native_metal::telemetry::note_submit_refused(submit_refusals::HOST_REFUSED);
                crate::native_metal::telemetry::record_frame(record_us, 0, 0, 0, None);
            }
            Ok(dispatch)
        }
        Err(error) => {
            let reason = submit_refusal_reason(&error, attached);
            // A submission that arrives before the client is attached is
            // deferred, not refused: `flush` keeps its frame queued, so the
            // first batch after the client attaches carries it. The deferral
            // is bounded, and at the bound the refusal is counted exactly once
            // under `no_client` (never once per attempt).
            let suppressed = reason == submit_refusals::NO_CLIENT && defer_no_client_refusal();
            if !suppressed {
                crate::native_metal::telemetry::note_submit_refused(reason);
            }
            if let NativeQueueError::Producer(producer) = &error
                && producer.opcode().is_some()
            {
                crate::native_metal::telemetry::record_frame(record_us, 0, 0, 0, None);
            }
            Err(error)
        }
    }
}

/// Whether `submissions` carries at least one command buffer.
///
/// A batch with no commands (or no submission at all) is an empty stream: the
/// queue would carry nothing, so it is never a submission and never a refusal.
#[cfg(any(windows, test, target_os = "macos"))]
fn carries_a_frame(submissions: &[SubmissionV3]) -> bool {
    submissions
        .iter()
        .any(|submission| !submission.commands.is_empty())
}

/// The static reason a refused submission batch is counted under.
///
/// `attached` is whether a live Wire v3 client was installed when the batch was
/// offered, so a transport error before the client existed is named `no_client`
/// instead of a generic transport fault. The `Empty` arm is defensive: an empty
/// batch is refused before it can reach the counter, and if a future caller
/// bypasses that check the refusal is still named.
fn submit_refusal_reason(error: &NativeQueueError, attached: bool) -> &'static str {
    use crate::native_metal::telemetry::submit_refusals as names;
    match error {
        NativeQueueError::Empty(_) => names::NO_COMMAND_BUFFER,
        NativeQueueError::Encode(_) | NativeQueueError::Plan(_) => names::ENCODE,
        NativeQueueError::Refused(_) => names::HOST_REFUSED,
        NativeQueueError::Transport(_) if !attached => names::NO_CLIENT,
        NativeQueueError::Transport(_) => names::TRANSPORT,
        NativeQueueError::Producer(producer) => producer.submit_refusal_name(),
        NativeQueueError::NoSurface(_) | NativeQueueError::NoSwapchain => names::PRESENT_REFUSED,
        NativeQueueError::Present(_) => names::PRESENT_REFUSED,
    }
}

/// How many batches of one streak may be deferred while no client is attached.
///
/// A guest that submits before the transport is attached defers its frame (the
/// queue keeps it and the next submission carries it). Past this bound the
/// deferral is abandoned and the refusal is counted once under `no_client`; the
/// count is not repeated for the rest of the streak, and any carried batch
/// starts a new one.
pub const NO_CLIENT_DEFER_BOUND: u32 = 8;

/// Batches deferred in the current no-client streak.
#[cfg(any(windows, test, target_os = "macos"))]
static DEFERRED_WITHOUT_CLIENT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Counts one deferred batch; true while the refusal must stay uncounted.
///
/// The first [`NO_CLIENT_DEFER_BOUND`] batches are silent, the next one is
/// counted (this returns false) and every later batch of the same streak is
/// silent again, so the refusal is counted exactly once.
#[cfg(any(windows, test, target_os = "macos"))]
fn defer_no_client_refusal() -> bool {
    let deferred = DEFERRED_WITHOUT_CLIENT.fetch_add(1, Ordering::AcqRel);
    let suppressed = deferred != NO_CLIENT_DEFER_BOUND;
    if suppressed {
        tracing::debug!(
            deferred = deferred.saturating_add(1),
            budget = NO_CLIENT_DEFER_BOUND,
            "VICTORY_BACKEND=metal: the batch is deferred until the Wire v3 client attaches"
        );
    }
    suppressed
}

/// Ends the no-client deferral streak (a batch was carried, or a new session).
#[cfg(any(windows, test, target_os = "macos"))]
fn deferred_batches_reset() {
    DEFERRED_WITHOUT_CLIENT.store(0, Ordering::Release);
}

/// How many batches of the current streak were deferred without a client.
#[cfg(all(test, any(windows, target_os = "macos")))]
fn deferred_batches() -> u32 {
    DEFERRED_WITHOUT_CLIENT.load(Ordering::Acquire)
}

/// Carries one list's recorded batch over the native queue.
///
/// A list that recorded nothing is not a submission: it returns `Ok(None)`, the
/// sink is never touched and no counter moves. This is the per-list decision
/// the live entry point and the tests share, so an empty stream is skipped in
/// exactly one place instead of becoming a refused frame.
#[cfg(any(windows, test, target_os = "macos"))]
#[allow(dead_code)] // the live entry point is Windows-only; the tests use it everywhere
pub(crate) fn submit_list_batch(
    submissions: &[SubmissionV3],
    record_us: u64,
) -> Result<Option<NativeDispatch>, NativeQueueError> {
    submit_list_batch_with_root_arguments(submissions, &[], record_us)
}

/// [`submit_list_batch`] with each submission's root-argument plan.
#[cfg(any(windows, test, target_os = "macos"))]
#[allow(dead_code)] // the live entry point is Windows-only; the tests use it everywhere
pub(crate) fn submit_list_batch_with_root_arguments(
    submissions: &[SubmissionV3],
    plans: &[&[bindings::RootArgument]],
    record_us: u64,
) -> Result<Option<NativeDispatch>, NativeQueueError> {
    if !carries_a_frame(submissions) {
        tracing::debug!(
            submissions = submissions.len(),
            "VICTORY_BACKEND=metal: a list carried no native submission; nothing is submitted"
        );
        return Ok(None);
    }
    submit_recorded_batch_with_root_arguments(submissions, plans, record_us).map(Some)
}

/// `ExecuteCommandLists` under `VICTORY_BACKEND=metal`.
///
/// Drains each title list's finished Wire v3 submissions (in list order, and
/// in each list's recorded order) into the native queue. Host D3D12 work is
/// never enqueued on this path: an empty list or a refused batch is logged and
/// surfaced through the queue, with no fallback. A list that recorded nothing
/// is skipped: it carries no frame, so it is neither submitted nor counted (the
/// one refusal per boot this path used to report). The first batch also
/// connects the session when `VICTORY_METAL_ENDPOINT`/`VICTORY_METAL_SECRET`
/// are set.
#[cfg(windows)]
pub fn execute_native(count: u32, lists: *const crate::object::This) {
    if count == 0 || lists.is_null() {
        return;
    }
    // SAFETY: the title passes `count` lists, as `queue::execute` documents.
    let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
    for &list in lists {
        let started = Instant::now();
        let batch = crate::list::take_submissions(list);
        // Guest time spent draining the finished list, reported as the frame's
        // record phase: the title's own recording happens in the list calls.
        let record_us = micros(started.elapsed());
        if batch.is_empty() {
            let failures = crate::list::metal_failures(list);
            if failures > 0 {
                tracing::debug!(
                    failures,
                    "native list refused calls and carried no submission; the queue reports the \
                     refusal the title will see at Close"
                );
            }
        }
        // The root-argument plan travels with its submission: a title
        // root-parameter index is not a translated slot, so the host needs the
        // register identity the plan recorded.
        let (submissions, plans): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .map(|recorded| (recorded.submission, recorded.root_arguments))
            .unzip();
        let plans: Vec<&[crate::native_metal::bindings::RootArgument]> =
            plans.iter().map(Vec::as_slice).collect();
        match submit_list_batch_with_root_arguments(&submissions, &plans, record_us) {
            Ok(None) => {}
            Ok(Some(NativeDispatch::Native {
                status,
                submissions: carried,
            })) => {
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
            Ok(Some(NativeDispatch::Host)) => {}
            Err(NativeQueueError::Empty(_)) => {
                tracing::debug!(
                    "VICTORY_BACKEND=metal: a submitted list carried no native submission"
                );
            }
            Err(error) => {
                let warning = refusal_warning(&error);
                let host_refused = counts_as_refused_submission(&error);
                tracing::warn!(
                    reason = %warning,
                    host_refused,
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

    use victory_metal::wire::Dispatch1d;
    use victory_metal::wire_v3::{CommandV3, PipelineRef, Present, decode_submission_v3};

    use crate::object::S_OK;
    use crate::native_metal::bindings::{self, GuestRange, GuestRangeResolver, ResolveError};
    use crate::native_metal::client::{
        ENVELOPE_LEN, KIND_ACK, KIND_AUTH, KIND_REGISTER, KIND_SUBMISSION_V3,
    };
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

    /// One compute pipeline selection followed by its dispatch.
    fn compute_pipeline_submission(
        sequence: u32,
        list_id: u64,
        pipeline_id: u64,
    ) -> SubmissionV3 {
        SubmissionV3 {
            sequence,
            list: Some(ListIdentity {
                list_id,
                list_generation: 1,
            }),
            commands: vec![
                CommandV3::SetComputePipeline(PipelineRef { pipeline_id }),
                CommandV3::Dispatch1d(Dispatch1d {
                    pipeline_id,
                    threads: 1,
                    threads_per_group: 1,
                    bindings: Vec::new(),
                }),
            ],
        }
    }

    /// A sink that records the order in which sends arrive.
    #[derive(Debug, Clone)]
    struct OrderedSink {
        order: Arc<(Mutex<Vec<&'static str>>, std::sync::Condvar)>,
    }

    impl V3Sink for OrderedSink {
        fn send(&mut self, _bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
            let (order, wake) = &*self.order;
            order
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push("submission");
            wake.notify_all();
            Ok(ReplyStatus::Ok)
        }
    }

    /// A submission that selects a pipeline cannot reach the render socket
    /// before an in-flight registration settles: the flush waits on the epoch
    /// it observed at enqueue. This is the ordering invariant behind the
    /// pipeline-class `UnknownHandle` guard; the guest is synchronous today, so
    /// the test drives the barrier directly.
    #[test]
    fn a_pipeline_submission_cannot_overtake_an_inflight_registration() {
        use crate::native_metal::pso;

        let _serial = pso::registration_test_lock();
        pso::reset_registration_barrier();
        let (ready, reached) = mpsc::channel();
        notify_on_pipeline_wait(Some(ready));

        let order = Arc::new((Mutex::new(Vec::<&'static str>::new()), std::sync::Condvar::new()));
        let mut queue = NativeQueue::new(OrderedSink {
            order: Arc::clone(&order),
        });
        let registration = pso::begin_pipeline_registration();
        let handle = std::thread::spawn(move || {
            let result = queue.submit(&[compute_pipeline_submission(1, 10, 7)]);
            (queue, result)
        });

        reached
            .recv_timeout(TEST_TIMEOUT)
            .expect("the flush reached the pipeline epoch wait");
        assert!(
            order
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty(),
            "the submission reached the host before the registration settled"
        );

        // The registration is ordered before the submission by completion, not
        // by the test's thread scheduling.
        order
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push("registration");
        registration.succeeded();
        let (_queue, result) = handle.join().expect("submission thread");
        assert_eq!(result, Ok(ReplyStatus::Ok));
        assert_eq!(
            *order
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec!["registration", "submission"],
            "registration completes before the submission is carried"
        );
        notify_on_pipeline_wait(None);
    }

    /// A submission that names no pipeline does not wait on a registration:
    /// the barrier is scoped to the frames that can actually name a pipeline.
    #[test]
    fn a_submission_without_a_pipeline_does_not_wait_for_registration() {
        use crate::native_metal::pso;

        let _serial = pso::registration_test_lock();
        pso::reset_registration_barrier();
        let (ready, reached) = mpsc::channel();
        notify_on_pipeline_wait(Some(ready));
        let registration = pso::begin_pipeline_registration();
        let mut queue = NativeQueue::new(RecordingSink::default());

        assert_eq!(
            queue.submit(&[submission(1, 10, 1)]),
            Ok(ReplyStatus::Ok),
            "a present-only frame is carried while a registration is in flight"
        );
        assert!(
            reached.try_recv().is_err(),
            "the present path never waits on a pipeline epoch"
        );
        drop(registration);
        notify_on_pipeline_wait(None);
    }

    /// A pipeline-class `UnknownHandle` names the first work command, the list
    /// and the sequence, and the histogram counts the class exactly.
    #[test]
    fn an_unknown_handle_refusal_names_the_first_work_command() {
        let _serial = serialized();
        reset_sequences(1);
        telemetry::global().reset_submit_refusal_sites_for_test();
        let mut queue = NativeQueue::new(HostStatusSink::default());

        let error = queue
            .submit(&[compute_pipeline_submission(0, 21, 5)])
            .expect_err("an odd list id is refused UnknownHandle");
        assert!(matches!(error, NativeQueueError::Refused(_)), "{error:?}");

        assert_eq!(
            telemetry::global().refusal_site_histogram(),
            std::collections::BTreeMap::from([(
                telemetry::refusal_sites::PIPELINE_COMPUTE,
                1
            )])
        );
        assert_eq!(telemetry::global().unknown_handle_sites(), 1);
        let sites = telemetry::global().submit_refusal_sites();
        assert_eq!(sites.len(), 1, "the detail is retained and bounded");
        let site = &sites[0];
        assert_eq!(site.sequence, 1);
        assert_eq!((site.list_id, site.list_generation), (21, 1));
        assert_eq!(site.command_index, Some(0));
        assert_eq!(
            site.opcode,
            Some(victory_metal::wire_v3::OP_SET_COMPUTE_PIPELINE)
        );
        assert_eq!(site.class, telemetry::refusal_sites::PIPELINE_COMPUTE);
        assert!(
            site.reason.contains("UnknownHandle"),
            "the host reason is retained: {site:?}"
        );
        let line = telemetry::global().snapshot().to_string();
        assert!(line.contains("refusal_sites=[pipeline:compute=1]"), "{line}");
        assert!(line.contains("sequence=1 list=21 generation=1 index=0"), "{line}");
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
                    let inner = carried_submission(&payload);
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

    /// The submission inside a version 3 request envelope.
    ///
    /// The envelope declares the submission length at bytes 12..16; everything
    /// after it is the optional root-argument plan section, which is not part of
    /// the submission frame.
    fn carried_submission(payload: &[u8]) -> SubmissionV3 {
        let declared = u32::from_le_bytes(payload[12..16].try_into().expect("four bytes")) as usize;
        decode_submission_v3(&payload[16..16 + declared]).expect("carried v3 frame")
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

    /// A sink that answers like a live host: it records every sequence it was
    /// handed and refuses a frame bound to an odd list id with the same
    /// `UnknownHandle` a refused pipeline registration produces. The refusal is
    /// a reply, not a transport fault, so it exercises the spent-sequence rule.
    #[derive(Debug, Default)]
    struct HostStatusSink {
        sequences: Vec<u32>,
        frames: Vec<Vec<u8>>,
    }

    impl V3Sink for HostStatusSink {
        fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
            let decoded = decode_submission_v3(bytes).expect("canonical v3");
            self.sequences.push(decoded.sequence);
            self.frames.push(bytes.to_vec());
            if decoded.list.is_some_and(|list| list.list_id % 2 == 1) {
                return Err(NativeQueueError::Refused(format!(
                    "the host answered UnknownHandle for sequence {}",
                    decoded.sequence
                )));
            }
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

    // ------------------------------------------- named submission refusals

    /// The per-boot refusal this task fixes: a list that recorded nothing
    /// reaches the queue as an empty batch. It is not a submission, so the sink
    /// is never touched and no refusal is counted. Before the fix it moved
    /// `submits_refused` by one, unnamed.
    #[test]
    fn an_empty_list_batch_is_not_a_refused_submission() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        assert_eq!(ensure_live(|| None), Ok(false));
        // The submit counters are session totals, so the test compares the
        // delta: another test in this process may already have counted one.
        let before = telemetry::global().submit_counts();
        let before_named = telemetry::global().named_submit_refusal_histogram();

        let error = submit_recorded_batch(&[], 0).expect_err("an empty batch carries no frame");
        assert!(
            matches!(&error, NativeQueueError::Empty(why) if why.contains("no command buffer")),
            "unexpected error {error:?}"
        );
        assert_eq!(
            telemetry::global().submit_counts(),
            before,
            "an empty stream is not a refused submission"
        );
        assert_eq!(
            telemetry::global().named_submit_refusal_histogram(),
            before_named,
            "nothing was refused, so nothing is named"
        );
        assert_eq!(native_queue().pending(), 0, "nothing was queued either");
        native_queue_reset();
    }

    /// The other empty-stream shape: submissions present, none carrying a
    /// command. Still not a submission, still not a refusal.
    #[test]
    fn a_batch_with_no_command_buffer_is_not_a_submission() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        assert_eq!(ensure_live(|| None), Ok(false));
        let before = telemetry::global().submit_counts();
        let before_named = telemetry::global().named_submit_refusal_histogram();

        let mut empty = submission(41, 411, 1);
        empty.commands.clear();
        assert!(matches!(
            submit_recorded_batch(std::slice::from_ref(&empty), 0),
            Err(NativeQueueError::Empty(_))
        ));
        assert_eq!(telemetry::global().submit_counts(), before);
        assert_eq!(
            telemetry::global().named_submit_refusal_histogram(),
            before_named
        );
        assert_eq!(native_queue().pending(), 0);
        native_queue_reset();
    }

    /// The per-list seam skips a list that recorded nothing and returns
    /// `Ok(None)`; a list that did record is carried exactly as before.
    #[test]
    fn a_list_that_recorded_nothing_is_skipped_not_refused() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        assert_eq!(ensure_live(|| None), Ok(false));
        let before = telemetry::global().submit_counts();
        let before_named = telemetry::global().named_submit_refusal_histogram();

        assert_eq!(
            submit_list_batch(&[], 0),
            Ok(None),
            "no frame: nothing to submit"
        );
        assert_eq!(telemetry::global().submit_counts(), before);
        assert_eq!(
            telemetry::global().named_submit_refusal_histogram(),
            before_named
        );
        assert_eq!(native_queue().pending(), 0);
        native_queue_reset();
    }

    /// The no-client deferral is bounded: the first
    /// [`NO_CLIENT_DEFER_BOUND`] batches are deferred (their frames stay
    /// queued), the next one counts the refusal exactly once under
    /// `no_client`, and the rest of the streak stays silent.
    #[test]
    fn a_no_client_streak_refuses_exactly_once_at_the_bound() {
        let _serial = serialized();
        native_queue_reset();
        telemetry::global().reset();
        assert_eq!(ensure_live(|| None), Ok(false));
        let (ok_before, refused_before) = telemetry::global().submit_counts();
        let named_before = telemetry::global()
            .named_submit_refusal_histogram()
            .get(submit_refusals::NO_CLIENT)
            .copied()
            .unwrap_or(0);

        let batch = submission(0, 512, 1);
        for deferred in 0..NO_CLIENT_DEFER_BOUND {
            assert_eq!(deferred_batches(), deferred);
            let error = submit_recorded_batch(std::slice::from_ref(&batch), 0)
                .expect_err("no client means no success");
            assert!(
                matches!(error, NativeQueueError::Transport(_)),
                "unexpected error {error:?}"
            );
            assert_eq!(
                telemetry::global().submit_counts().1,
                refused_before,
                "batch {deferred} is deferred, not refused"
            );
        }
        assert_eq!(deferred_batches(), NO_CLIENT_DEFER_BOUND);

        // The bound is hit: one refusal, named.
        let _ = submit_recorded_batch(std::slice::from_ref(&batch), 0).expect_err("still no host");
        assert_eq!(telemetry::global().submit_counts().1, refused_before + 1);
        assert_eq!(telemetry::global().submit_counts().0, ok_before);
        assert_eq!(
            telemetry::global()
                .named_submit_refusal_histogram()
                .get(submit_refusals::NO_CLIENT)
                .copied(),
            Some(named_before + 1),
            "the one refusal carries the no_client name"
        );

        // Later attempts of the same streak do not inflate the count.
        for _ in 0..3 {
            let _ = submit_recorded_batch(std::slice::from_ref(&batch), 0);
            assert_eq!(telemetry::global().submit_counts().1, refused_before + 1);
        }

        // Teardown starts a new streak; a carried batch does the same (the
        // reconstructed boot below carries one and leaves the streak at zero).
        native_queue_reset();
        assert_eq!(deferred_batches(), 0);
        assert_eq!(native_queue().pending(), 0);
    }

    /// Every refusal the queue counts passes a static reason from the fixed
    /// table, and each class maps to its own name.
    #[test]
    fn every_refused_batch_is_counted_under_a_static_reason() {
        use crate::native_metal::telemetry::{SUBMIT_REFUSAL_NAMES, submit_refusals};

        let producer = ProducerError::unsupported_opcode("test", 7);
        let cases = [
            (
                NativeQueueError::Empty("test"),
                submit_refusals::NO_COMMAND_BUFFER,
            ),
            (
                NativeQueueError::Encode(WireError::ZeroSequence),
                submit_refusals::ENCODE,
            ),
            (
                NativeQueueError::Refused("host".to_string()),
                submit_refusals::HOST_REFUSED,
            ),
            (
                NativeQueueError::Transport("reset".to_string()),
                submit_refusals::NO_CLIENT,
            ),
            (
                NativeQueueError::Producer(producer),
                submit_refusals::PRODUCER_UNSUPPORTED,
            ),
            (NativeQueueError::NoSwapchain, submit_refusals::PRESENT_REFUSED),
        ];
        for (error, expected) in cases {
            for attached in [false, true] {
                let reason = submit_refusal_reason(&error, attached);
                assert!(
                    SUBMIT_REFUSAL_NAMES.contains(&reason),
                    "{reason} is not in the fixed table"
                );
                if !(attached && matches!(&error, NativeQueueError::Transport(_))) {
                    assert_eq!(reason, expected, "{error:?} attached={attached}");
                }
            }
        }
        // A transport fault on a live session is a transport refusal, not a
        // missing client.
        assert_eq!(
            submit_refusal_reason(&NativeQueueError::Transport("reset".to_string()), true),
            submit_refusals::TRANSPORT
        );
    }

    /// The boot sequence reconstructed: two lists with commands are carried,
    /// the empty one is skipped, and the session reports two successes and no
    /// refused submission.
    #[test]
    fn a_reconstructed_boot_sequence_reports_no_refused_submission() {
        let _serial = serialized();
        native_queue_reset();
        reset_sequences(1);
        telemetry::global().reset();
        let (ok_before, refused_before) = telemetry::global().submit_counts();
        let named_before = telemetry::global().named_submit_refusal_histogram();
        let (host, carried) = TestHost::spawn(vec![ReplyStatus::Ok, ReplyStatus::Ok]);
        let config = host.config();
        assert_eq!(ensure_live(|| Some(Ok(config))), Ok(true));

        let first = submission(0, 601, 1);
        let second = submission(0, 602, 1);
        assert!(matches!(
            submit_list_batch(std::slice::from_ref(&first), 3),
            Ok(Some(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                ..
            }))
        ));
        // The title executes a list that recorded nothing: skipped, not refused.
        assert_eq!(submit_list_batch(&[], 1), Ok(None));
        assert!(matches!(
            submit_list_batch(std::slice::from_ref(&second), 2),
            Ok(Some(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                ..
            }))
        ));

        assert_eq!(
            telemetry::global().submit_counts(),
            (ok_before + 2, refused_before),
            "the reconstructed boot reports two submissions and no refusal"
        );
        assert_eq!(
            telemetry::global().named_submit_refusal_histogram(),
            named_before,
            "no submission-refusal reason was recorded"
        );
        assert_eq!(telemetry::global().recorded_frames(), 2);
        assert_eq!(deferred_batches(), 0, "a carried batch ends the streak");
        assert_eq!(
            telemetry::global().d3dmetal_fallback_count(),
            0,
            "no host D3D12 and no D3DMetal involvement"
        );
        let mut expected_first = first.clone();
        expected_first.sequence = 1;
        let mut expected_second = second.clone();
        expected_second.sequence = 2;
        assert_eq!(
            carried.recv_timeout(TEST_TIMEOUT).expect("first"),
            expected_first,
            "the first list's identity and sequence are preserved"
        );
        assert_eq!(
            carried.recv_timeout(TEST_TIMEOUT).expect("second"),
            expected_second,
            "the skipped empty list consumed no sequence"
        );
        native_queue_reset();    }

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

    /// End-to-end over a loopback host: a submission the host refuses with
    /// `ExecutionFailed` is answered, consumed and surfaced, and the next
    /// submission keeps the same session alive at the next sequence. This is
    /// Boot #21's shape without a GPU: no resend, no protocol violation.
    #[test]
    fn a_live_host_refusal_does_not_desynchronise_the_next_submission() {
        let _serial = serialized();
        native_queue_reset();
        reset_sequences(1);
        let (host, carried) = TestHost::spawn(vec![ReplyStatus::ExecutionFailed, ReplyStatus::Ok]);
        let config = host.config();
        assert_eq!(ensure_live(|| Some(Ok(config))), Ok(true));

        let error = submit_recorded_batch(&[submission(0, 41, 1)], 1)
            .expect_err("the host refused the first submission");
        assert!(
            counts_as_refused_submission(&error),
            "the refusal reaches the caller: {error}"
        );

        assert_eq!(
            submit_recorded_batch(&[submission(0, 42, 1)], 1),
            Ok(NativeDispatch::Native {
                status: ReplyStatus::Ok,
                submissions: 1,
            }),
            "the session keeps running after an ordinary refusal"
        );

        let first = carried.recv_timeout(TEST_TIMEOUT).expect("first frame");
        let second = carried.recv_timeout(TEST_TIMEOUT).expect("second frame");
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2, "the refused sequence is not re-sent");
        assert!(
            native_queue_is_live(),
            "the refusal did not end the session"
        );
        assert_eq!(native_queue().pending(), 0, "nothing is left to resend");
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
            let inner = carried_submission(&payload);
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

    /// A host-supplied refusal reason reaches the exact `native queue refused:`
    /// warning and is classified as a refused submission. The live client puts
    /// the host text in `Refused`; this test exercises that seam without a
    /// socket.
    #[test]
    fn a_host_refusal_reason_is_forwarded_to_the_warning() {
        let host_reason = "draw #12 (MTLCommandBuffer.status=4): out-of-bounds write";
        let error = NativeQueueError::Refused(format!(
            "the host answered ExecutionFailed for sequence 586: {host_reason}"
        ));

        let warning = refusal_warning(&error);
        assert!(
            warning.starts_with("native queue refused: "),
            "{warning}"
        );
        assert!(warning.contains(host_reason), "{warning}");
        assert!(counts_as_refused_submission(&error));

        // A transport fault is not a host reason: it keeps its own text.
        let transport = NativeQueueError::Transport("connection reset".to_string());
        assert!(refusal_detail(&transport).is_none());
        assert!(!counts_as_refused_submission(&transport));
        assert_eq!(
            refusal_warning(&transport),
            "native queue transport: connection reset"
        );
    }

    /// A host refusal is a reply, not a transport failure: the host consumed
    /// the sequence and wrote an answer, so the frame is spent and is never
    /// re-sent. This is Boot #21: a refused pipeline answered `UnknownHandle`
    /// for sequence 3, the frame stayed queued, and the next present re-sent
    /// sequence 3 into a host that already expected 4, which ended the session.
    #[test]
    fn a_host_refused_send_is_consumed_and_never_resent() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(HostStatusSink::default());

        let error = queue
            .submit(&[submission(0, 1, 1)])
            .expect_err("a host refusal is returned to the caller");
        assert!(
            counts_as_refused_submission(&error),
            "the ledger still reports the refusal: {error}"
        );
        assert_eq!(
            error,
            NativeQueueError::Refused("the host answered UnknownHandle for sequence 1".to_string()),
            "the host's reply reaches the caller unchanged"
        );
        assert!(
            refusal_warning(&error).starts_with("native queue refused: "),
            "the live log still names the host reason"
        );
        assert_eq!(queue.sent(), 1, "the host answered the refused frame");
        assert_eq!(queue.pending(), 0, "a refused frame is never retried");
        assert_eq!(queue.highest_sequence(), 1);

        // The render channel keeps running: the next real submission takes the
        // next sequence, with no hole and no resend of the refused one.
        assert_eq!(queue.submit(&[submission(0, 2, 1)]), Ok(ReplyStatus::Ok));
        assert_eq!(
            queue.sink().sequences,
            vec![1, 2],
            "the refused sequence is consumed exactly once"
        );
    }

    /// A hundred refused submissions interleaved with a hundred real ones keep
    /// the ledger dense: every frame is sent exactly once, in order, and the
    /// sequences are 1..=200 with no reuse and no hole.
    #[test]
    fn a_hundred_refusals_interleaved_with_real_submissions_keep_the_ledger_dense() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(HostStatusSink::default());
        let sites_before = telemetry::global().unknown_handle_sites();
        let present_before = telemetry::global()
            .refusal_site_histogram()
            .get(telemetry::refusal_sites::PRESENT)
            .copied()
            .unwrap_or(0);

        for index in 0..100u64 {
            let refused = queue.submit(&[submission(0, index * 2 + 1, 1)]);
            assert!(
                matches!(refused, Err(NativeQueueError::Refused(_))),
                "submission {index} must be refused: {refused:?}"
            );
            assert_eq!(
                queue.submit(&[submission(0, index * 2 + 2, 1)]),
                Ok(ReplyStatus::Ok),
                "the real submission after refusal {index} must be carried"
            );
        }

        assert_eq!(queue.sent(), 200);
        assert_eq!(queue.pending(), 0);
        assert!(queue.is_empty());
        assert_eq!(queue.highest_sequence(), 200);
        assert_eq!(
            queue.sink().sequences,
            (1..=200).collect::<Vec<u32>>(),
            "each frame is sent once, in order, with no reused or skipped sequence"
        );
        let mut distinct = queue.sink().sequences.clone();
        distinct.dedup();
        assert_eq!(distinct.len(), 200, "no sequence is sent twice");

        // Every refusal is named exactly, and the retained detail stays at the
        // fixed cap: a hundred interleaved refusals never grow the snapshot.
        assert_eq!(
            telemetry::global().unknown_handle_sites() - sites_before,
            100,
            "one site per refused submission, no more and no less"
        );
        assert_eq!(
            telemetry::global()
                .refusal_site_histogram()
                .get(telemetry::refusal_sites::PRESENT)
                .copied()
                .unwrap_or(0)
                - present_before,
            100
        );
        assert!(
            telemetry::global().submit_refusal_sites().len()
                <= telemetry::MAX_SUBMIT_REFUSAL_SITES,
            "the retained detail is bounded"
        );
    }

    /// A retry after a refusal is a new submission with a new sequence: it
    /// reuses nothing and skips nothing.
    #[test]
    fn a_retry_after_a_refusal_takes_the_next_sequence() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(HostStatusSink::default());
        let refused_batch = [submission(0, 21, 1)];

        assert!(matches!(
            queue.submit(&refused_batch),
            Err(NativeQueueError::Refused(_))
        ));
        // The same batch is offered again: it is a new frame with a new
        // sequence, never a resend of the consumed one.
        assert!(matches!(
            queue.submit(&refused_batch),
            Err(NativeQueueError::Refused(_))
        ));
        assert_eq!(queue.sink().sequences, vec![1, 2]);
        assert_eq!(queue.highest_sequence(), 2);
        // Nothing was skipped either: the next accepted frame is 3.
        assert_eq!(queue.submit(&[submission(0, 22, 1)]), Ok(ReplyStatus::Ok));
        assert_eq!(queue.sink().sequences, vec![1, 2, 3]);
    }

    /// A submission whose commands were all refused client-side carries no
    /// frame: it is dropped before the wire and consumes no sequence, so the
    /// next real submission is still sequence 1.
    #[test]
    fn an_all_refused_submission_consumes_no_sequence() {
        let _serial = serialized();
        reset_sequences(1);
        let mut queue = NativeQueue::new(RecordingSink::default());

        for list_id in 0..100u64 {
            let mut dropped = submission(0, 31 + list_id, 1);
            // Every command was refused by the producer before enqueue.
            dropped.commands.clear();
            assert_eq!(
                queue.submit(std::slice::from_ref(&dropped)),
                Ok(ReplyStatus::Ok)
            );
        }
        assert_eq!(queue.sink().frames.len(), 0, "nothing reached the host");
        assert_eq!(queue.highest_sequence(), 0, "no sequence was spent");
        assert_eq!(queue.sent(), 0);

        assert_eq!(queue.submit(&[submission(0, 77, 1)]), Ok(ReplyStatus::Ok));
        assert_eq!(
            decode_submission_v3(&queue.sink().frames[0])
                .expect("canonical v3")
                .sequence,
            1,
            "the first real frame is sequence 1"
        );
    }

    // ------------------------------------------- lazy first-bind registration

    /// A loopback host that answers authentication and records every kind 2
    /// allocation registration, acknowledging each one like the real host.
    struct RegistrationHost {
        port: u16,
        handle: Option<JoinHandle<()>>,
        seen: Receiver<(u64, u64)>,
    }

    impl RegistrationHost {
        fn spawn() -> Self {
            let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .expect("bind loopback");
            let port = listener.local_addr().expect("listener address").port();
            let (sender, seen) = mpsc::channel();
            let handle = std::thread::spawn(move || {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                stream
                    .set_read_timeout(Some(TEST_TIMEOUT))
                    .expect("server read timeout");
                let Some((kind, payload)) = frame_or_eof(&mut stream) else {
                    return;
                };
                assert_eq!(kind, KIND_AUTH, "the first frame must authenticate");
                assert_eq!(payload, SECRET, "the hello must carry the exact secret");
                write_frame(&mut stream, KIND_ACK, &[0]);
                while let Some((kind, payload)) = frame_or_eof(&mut stream) {
                    match kind {
                        // The session registered the share directory's fence
                        // page while connecting.
                        crate::native_metal::client::KIND_FENCE_PAGE => {
                            assert_eq!(payload.len(), 8);
                            write_frame(&mut stream, KIND_ACK, &[0]);
                        }
                        KIND_REGISTER => {
                            assert_eq!(payload.len(), 16, "a kind 2 payload");
                            let id = u64::from_le_bytes(
                                payload[0..8].try_into().expect("eight bytes"),
                            );
                            let len = u64::from_le_bytes(
                                payload[8..16].try_into().expect("eight bytes"),
                            );
                            sender.send((id, len)).expect("report the registration");
                            write_frame(&mut stream, KIND_ACK, &[0]);
                        }
                        other => panic!("unexpected frame kind {other}"),
                    }
                }
            });
            Self {
                port,
                handle: Some(handle),
                seen,
            }
        }

        fn config(&self) -> LiveConfig {
            LiveConfig::parse(
                Some(&format!("127.0.0.1:{}", self.port)),
                Some(&"5a".repeat(32)),
            )
            .expect("both variables present")
            .expect("well-formed configuration")
        }

        /// One recorded registration, in arrival order.
        fn next(&self) -> (u64, u64) {
            self.seen
                .recv_timeout(TEST_TIMEOUT)
                .expect("the host saw the registration")
        }

        /// Asserts no further registration reaches this session.
        fn assert_quiet(&self) {
            assert!(
                self.seen.recv_timeout(std::time::Duration::from_millis(100)).is_err(),
                "an idempotent bind must issue no registration frame"
            );
        }
    }

    impl Drop for RegistrationHost {
        fn drop(&mut self) {
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    /// [`read_frame`] that answers `None` at the end of the session instead of
    /// panicking, so the fixture thread can end when the client is dropped.
    fn frame_or_eof(stream: &mut std::net::TcpStream) -> Option<(u8, Vec<u8>)> {
        let mut envelope = [0u8; ENVELOPE_LEN];
        stream.read_exact(&mut envelope).ok()?;
        let kind = envelope[0];
        let length = usize::try_from(u32::from_le_bytes(
            envelope[1..].try_into().ok()?,
        ))
        .ok()?;
        let mut payload = vec![0u8; length];
        stream.read_exact(&mut payload).ok()?;
        Some((kind, payload))
    }

    /// A resolver over synthetic guest ranges, `(base, size, allocation id)`.
    struct RangeMapResolver {
        ranges: Mutex<Vec<(u64, u64, u64)>>,
    }

    impl RangeMapResolver {
        fn new(ranges: &[(u64, u64, u64)]) -> Self {
            Self {
                ranges: Mutex::new(ranges.to_vec()),
            }
        }
    }

    impl GuestRangeResolver for RangeMapResolver {
        fn resolve(&self, address: u64, len: u64) -> Result<GuestRange, ResolveError> {
            let ranges = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
            for &(base, size, allocation_id) in ranges.iter() {
                let end = base.saturating_add(size);
                if address >= base && address.saturating_add(len) <= end {
                    let pages = size.div_ceil(crate::native_metal::bindings::PAGE)
                        * crate::native_metal::bindings::PAGE;
                    return Ok(GuestRange {
                        allocation_id,
                        generation: victory_metal::guest_ipc::REGISTRATION_GENERATION,
                        byte_offset: 0,
                        size: pages,
                    });
                }
            }
            Err(ResolveError::Unmapped { address, len })
        }

        fn retain(&self, _range: &GuestRange) -> Result<u64, ResolveError> {
            Ok(1)
        }

        fn retire(&self, _token: u64) -> Result<(), ResolveError> {
            Ok(())
        }
    }

    // -------------------------------------- named range refusals (heap table)

    /// The Boot #15b range shape: a ~1 MiB vertex buffer at a non-zero offset
    /// in a heap whose share table has nothing is refused by name, never as
    /// an `xmem` miss.
    #[test]
    fn a_heap_range_with_no_share_backing_is_refused_by_name() {
        let _serial = serialized();
        reset_guest_range_refusals();
        crate::resource::clear_created_ranges_for_test();
        crate::resource::note_created_range_for_test(
            0x147a0_0000,
            0x20_0000,
            crate::resource::CreatedRangeOrigin::Heap,
        );
        let resolver = CreatedRangeResolver {
            inner: Arc::new(RangeMapResolver::new(&[])),
        };
        let error = resolver
            .resolve(0x147a2_b0f0, 1_010_960)
            .expect_err("the heap has no share backing");
        assert_eq!(
            error,
            ResolveError::Unsupported("heap has no host-visible backing")
        );
        let shown = error.to_string();
        assert!(
            shown.contains("heap has no host-visible backing"),
            "{shown}"
        );
        assert!(
            !shown.contains("no shared xmem range"),
            "the heap refusal is distinguishable: {shown}"
        );
        let (named, counted, overflow, total) = guest_range_refusal_counts();
        assert_eq!((named, counted, overflow, total), (1, 1, 0, 1));
        crate::resource::clear_created_ranges_for_test();
        reset_guest_range_refusals();
    }

    /// The same shape inside a heap the share table resolves is a normal
    /// binding: the wrapper only renames refusals.
    #[test]
    fn a_one_mib_buffer_inside_a_resolvable_heap_resolves() {
        let _serial = serialized();
        reset_guest_range_refusals();
        crate::resource::clear_created_ranges_for_test();
        let base = 0x147a0_0000;
        let len = 0x20_0000;
        crate::resource::note_created_range_for_test(
            base,
            len,
            crate::resource::CreatedRangeOrigin::Heap,
        );
        let resolver = CreatedRangeResolver {
            inner: Arc::new(RangeMapResolver::new(&[(base, len, 41)])),
        };
        let range = resolver
            .resolve(base + 0x2b0f0, 1_010_960)
            .expect("the heap resolves to its share allocation");
        assert_eq!(range.allocation_id, 41);
        assert_eq!(
            range.generation,
            victory_metal::guest_ipc::REGISTRATION_GENERATION
        );
        assert_eq!(guest_range_refusal_counts().3, 0, "nothing was refused");
        crate::resource::clear_created_ranges_for_test();
    }

    /// A range no table here covers keeps the share resolver's own name.
    #[test]
    fn a_range_outside_every_created_table_keeps_the_share_table_name() {
        let _serial = serialized();
        reset_guest_range_refusals();
        crate::resource::clear_created_ranges_for_test();
        let resolver = CreatedRangeResolver {
            inner: Arc::new(RangeMapResolver::new(&[])),
        };
        let error = resolver
            .resolve(0x10_0000, 4096)
            .expect_err("no range covers it");
        assert!(matches!(error, ResolveError::Unmapped { .. }));
        assert!(error.to_string().contains("no shared xmem range"));
        assert_eq!(
            crate::resource::guest_range_branch(0x10_0000, 4096),
            crate::resource::RangeRefusalBranch::ShareTable
        );
        let (named, counted, overflow, total) = guest_range_refusal_counts();
        assert_eq!((named, counted, overflow, total), (1, 1, 0, 1));
        reset_guest_range_refusals();
    }

    /// The census names each distinct range once and counts repeats: a range
    /// refused for every draw cannot produce a line per draw, and the number
    /// of named ranges is bounded.
    #[test]
    fn range_refusals_name_each_distinct_range_once_and_are_bounded() {
        let _serial = serialized();
        reset_guest_range_refusals();
        let branch = crate::resource::RangeRefusalBranch::ShareTable;
        for index in 0..100u64 {
            note_refused_guest_range(
                0x2000_0000 + index * 0x1000,
                0x1000,
                branch,
                "no shared xmem range",
            );
        }
        for _ in 0..50 {
            note_refused_guest_range(0x2000_0000, 0x1000, branch, "no shared xmem range");
        }
        let (named, counted, overflow, total) = guest_range_refusal_counts();
        assert_eq!(
            named, MAX_NAMED_RANGE_REFUSALS,
            "the distinct names are capped"
        );
        assert_eq!(overflow, 36, "the rest are counted, not named");
        assert_eq!(counted, 64 + 50, "repeats only move the count");
        assert_eq!(total, 150);
        reset_guest_range_refusals();
    }

    /// A session-private share directory with one sparse file per allocation.
    struct ShareDir {
        directory: tempfile::TempDir,
        previous: Option<std::ffi::OsString>,
    }

    impl ShareDir {
        fn install(files: &[(u64, u64)]) -> Self {
            let directory = tempfile::tempdir().expect("a share directory");
            for &(id, len) in files {
                create_share_file(directory.path(), id, len);
            }
            let previous = std::env::var_os(crate::native_metal::fence_page_guest::SHARE_DIR_ENV);
            // SAFETY: the serialized lock keeps every other test in this
            // module out of the process environment.
            unsafe {
                std::env::set_var(
                    crate::native_metal::fence_page_guest::SHARE_DIR_ENV,
                    directory.path(),
                );
            }
            Self {
                directory,
                previous,
            }
        }

        fn path(&self) -> &std::path::Path {
            self.directory.path()
        }
    }

    impl Drop for ShareDir {
        fn drop(&mut self) {
            // SAFETY: as in `install`.
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var(
                        crate::native_metal::fence_page_guest::SHARE_DIR_ENV,
                        value,
                    ),
                    None => std::env::remove_var(
                        crate::native_metal::fence_page_guest::SHARE_DIR_ENV,
                    ),
                }
            }
        }
    }

    fn create_share_file(directory: &std::path::Path, id: u64, len: u64) {
        let path = directory.join(format!("{id:016x}.bin"));
        std::fs::File::create(&path)
            .and_then(|file| file.set_len(len))
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }

    const BIG_ALLOCATION: u64 = 182_452_224;

    /// Connects the process-global queue to a fresh registration host.
    fn live_registration_host() -> RegistrationHost {
        native_queue_reset();
        let host = RegistrationHost::spawn();
        let config = host.config();
        assert_eq!(ensure_live(|| Some(Ok(config))), Ok(true));
        assert!(native_queue_is_live());
        host
    }

    /// A buffer created before the resolver existed registers on its first
    /// bind: creation-time failure is not final, and the bind registers the
    /// allocation once, from the file the resolver names.
    #[test]
    fn a_buffer_created_before_the_resolver_registers_on_its_first_bind() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let address = 0x1_0000_0000;
        let _share = ShareDir::install(&[(1, 352)]);
        let host = live_registration_host();

        // Creation ran before the HLE installed the resolver: the buffer's
        // allocation could not be named, so nothing was registered.
        assert_eq!(
            register_guest_buffer(address, 352),
            Err(BufferRegistrationError::NoResolver)
        );
        host.assert_quiet();

        // The resolver appears; the first bind is the safety net.
        bindings::set_resolver(RangeMapResolver::new(&[(address, 352, 1)]));
        let range = register_guest_buffer(address, 352).expect("the first bind registers");
        assert_eq!(range.allocation_id, 1);
        assert_eq!(range.generation, 1);
        assert_eq!(host.next(), (1, 352), "the whole share file length");

        // Idempotent: a second bind of the same buffer sends no frame.
        assert_eq!(
            register_guest_buffer(address, 352).expect("the second bind is a no-op").allocation_id,
            1
        );
        host.assert_quiet();
        assert!(session_allocation_registered(1));

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// The 182 MB startup heap allocation registers exactly like a small one:
    /// one kind 2 frame with the whole file length, then none, with no size cap.
    #[test]
    fn a_182_mib_allocation_registers_in_one_frame_and_idempotently() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let address = 0x2_0000_0000;
        let _share = ShareDir::install(&[(1, BIG_ALLOCATION)]);
        let host = live_registration_host();
        bindings::set_resolver(RangeMapResolver::new(&[(address, BIG_ALLOCATION, 1)]));

        let range = register_guest_buffer(address, BIG_ALLOCATION).expect("a large allocation");
        assert_eq!(range.allocation_id, 1);
        assert_eq!(range.size, BIG_ALLOCATION, "no small-buffer cap");
        assert_eq!(host.next(), (1, BIG_ALLOCATION));

        assert_eq!(
            register_guest_buffer(address, BIG_ALLOCATION)
                .expect("the repeat is a no-op")
                .allocation_id,
            1
        );
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// The six startup buffers created before the resolver register on their
    /// first bind, ascending by allocation id with their real share lengths;
    /// the second bind of the whole set issues no frame.
    #[test]
    fn six_pre_resolver_buffers_register_ascending_on_one_bind() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let base = 0x3_0000_0000;
        let stride = 0x1000_0000;
        let sizes = [BIG_ALLOCATION, 19_680, 19_680, 19_680, 19_680, 352];
        let files: Vec<(u64, u64)> = sizes
            .iter()
            .enumerate()
            .map(|(index, size)| (index as u64 + 1, *size))
            .collect();
        let _share = ShareDir::install(&files);
        let host = live_registration_host();
        let ranges: Vec<(u64, u64, u64)> = sizes
            .iter()
            .enumerate()
            .map(|(index, size)| (base + index as u64 * stride, *size, index as u64 + 1))
            .collect();
        bindings::set_resolver(RangeMapResolver::new(&ranges));
        let bindings_named: Vec<(u64, u64)> = ranges
            .iter()
            .map(|(address, size, _)| (*address, *size))
            .collect();

        // The title's first bind names all six, in its own order.
        let resolved = register_guest_buffers(&bindings_named).expect("six buffers register");
        assert_eq!(
            resolved
                .iter()
                .map(|range| range.allocation_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6],
            "the caller's order is preserved"
        );
        for &(id, len) in &files {
            assert_eq!(host.next(), (id, len), "ascending, exact file length");
        }
        host.assert_quiet();
        for (id, _) in &files {
            assert!(session_allocation_registered(*id));
        }

        // The same set again: every id is known, so nothing goes on the wire.
        register_guest_buffers(&bindings_named).expect("the repeat is a no-op");
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// A share file that appears after an earlier bind is discovered on the
    /// next bind and registered ascending, once, without re-registering the
    /// ids the host already accepted.
    #[test]
    fn an_allocation_appearing_after_bind_time_registers_once_ascending() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let first = 0x4_0000_0000;
        let third = 0x4_1000_0000;
        let fifth = 0x4_2000_0000;
        let share = ShareDir::install(&[(1, 4096)]);
        let host = live_registration_host();
        bindings::set_resolver(RangeMapResolver::new(&[
            (first, 4096, 1),
            (third, 4096, 3),
            (fifth, 4096, 5),
        ]));

        register_guest_buffer(first, 4096).expect("the first bind registers id 1");
        assert_eq!(host.next(), (1, 4096));

        // The launcher materialises two later allocations after that bind.
        create_share_file(share.path(), 3, 8192);
        create_share_file(share.path(), 5, 8192);

        register_guest_buffer(fifth, 4096).expect("the later bind sweeps up");
        assert_eq!(host.next(), (3, 8192), "the lower pending id goes first");
        assert_eq!(host.next(), (5, 8192), "then the requested id");
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// A missing share backing, a missing resolver and an unresolvable range
    /// are named refusals; none registers an id or sends a frame.
    #[test]
    fn missing_backing_and_unresolved_ranges_are_named_refusals() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let address = 0x5_0000_0000;
        // A share directory with a different allocation only.
        let share = ShareDir::install(&[(1, 4096)]);
        let host = live_registration_host();

        assert_eq!(
            register_guest_buffer(address, 4096),
            Err(BufferRegistrationError::NoResolver)
        );

        bindings::set_resolver(RangeMapResolver::new(&[(address, 4096, 9)]));
        let error = register_guest_buffer(address, 4096).expect_err("no share file for id 9");
        assert!(
            matches!(
                &error,
                BufferRegistrationError::MissingShareBacking { allocation_id: 9, .. }
            ),
            "unexpected error {error:?}"
        );
        host.assert_quiet();
        assert!(!session_allocation_registered(9));

        // An address the resolver does not map.
        let error = register_guest_buffer(address + 0x10_0000, 4096)
            .expect_err("an unmapped guest range");
        assert!(
            matches!(&error, BufferRegistrationError::Unresolved { .. }),
            "unexpected error {error:?}"
        );
        host.assert_quiet();

        // An empty file cannot name a length either.
        create_share_file(share.path(), 9, 0);
        let error = register_guest_buffer(address, 4096).expect_err("an empty share file");
        assert!(
            matches!(
                &error,
                BufferRegistrationError::EmptyShareBacking { allocation_id: 9, .. }
            ),
            "unexpected error {error:?}"
        );
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// A copy registers both its source and destination, ascending even when
    /// the source allocation id is the higher one, and the second copy of the
    /// same buffers sends nothing.
    #[test]
    fn a_copy_path_registers_its_source_and_destination() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let source = 0x6_0000_0000;
        let destination = 0x6_1000_0000;
        let _share = ShareDir::install(&[(2, 4096), (3, 8192)]);
        let host = live_registration_host();
        // The source names the higher id: registration must still ascend.
        bindings::set_resolver(RangeMapResolver::new(&[
            (source, 4096, 3),
            (destination, 4096, 2),
        ]));

        let (from, to) = register_copy_buffers((source, 4096), (destination, 4096))
            .expect("both copy buffers register");
        assert_eq!(from.allocation_id, 3, "the source keeps its own identity");
        assert_eq!(to.allocation_id, 2, "the destination keeps its own identity");
        assert_eq!(host.next(), (2, 4096), "the lower id is registered first");
        assert_eq!(host.next(), (3, 8192), "then the source's id");
        host.assert_quiet();

        register_copy_buffers((source, 4096), (destination, 4096))
            .expect("the second copy is a no-op");
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// A copy that names an id below the host's ledger is refused locally with
    /// a named error: the host never sees a decreasing kind 2 id (it closes the
    /// session on one).
    #[test]
    fn a_decreasing_registration_is_refused_before_the_wire() {
        let _serial = serialized();
        let _ = bindings::clear_resolver();
        let high = 0x7_0000_0000;
        let low = 0x7_1000_0000;
        // Only the higher id exists at first: the initial bind cannot see a
        // lower file at all.
        let share = ShareDir::install(&[(7, 8192)]);
        let host = live_registration_host();
        bindings::set_resolver(RangeMapResolver::new(&[
            (high, 4096, 7),
            (low, 4096, 5),
        ]));

        register_guest_buffer(high, 4096).expect("the higher id registers");
        assert_eq!(host.next(), (7, 8192));

        // An older allocation's file appears after the ledger passed its id.
        create_share_file(share.path(), 5, 4096);
        let error = register_guest_buffer(low, 4096).expect_err("a lower id cannot register");
        assert!(
            matches!(
                &error,
                BufferRegistrationError::Decreasing {
                    allocation_id: 5,
                    last_registered: 7
                }
            ),
            "unexpected error {error:?}"
        );
        host.assert_quiet();

        let _ = bindings::clear_resolver();
        native_queue_reset();
    }

    /// A sink that keeps the frame *and* the plan section that rode with it.
    #[derive(Debug, Default)]
    struct PlanRecordingSink {
        frames: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    }

    impl V3Sink for PlanRecordingSink {
        fn send(&mut self, bytes: &[u8]) -> Result<ReplyStatus, NativeQueueError> {
            self.frames.push((bytes.to_vec(), None));
            Ok(ReplyStatus::Ok)
        }

        fn send_with_plan(
            &mut self,
            bytes: &[u8],
            plan: Option<&[u8]>,
        ) -> Result<ReplyStatus, NativeQueueError> {
            self.frames.push((bytes.to_vec(), plan.map(<[u8]>::to_vec)));
            Ok(ReplyStatus::Ok)
        }
    }

    /// A producer with one compute dispatch and the root arguments it recorded.
    fn compute_producer(
        list_id: u64,
        record: impl FnOnce(&mut producer::MetalProducer),
    ) -> (SubmissionV3, Vec<bindings::RootArgument>) {
        let mut producer = producer::MetalProducer::new(list_id, 1).expect("a list generation");
        record(&mut producer);
        producer.set_compute_pipeline(0x700).expect("compute pipeline");
        producer
            .set_compute_threadgroup_size([64, 1, 1])
            .expect("threadgroup size");
        producer.dispatch(1, 1, 1).expect("dispatch");
        let submission = producer.flush().expect("flush");
        let plan = producer.root_arguments().to_vec();
        (submission, plan)
    }

    /// The list's recorded plan reaches the sink beside the unchanged frame, and
    /// the frame still decodes as the canonical v3 submission it always was.
    #[test]
    fn a_submission_carries_its_root_argument_plan_to_the_sink() {
        let (submission, plan) = compute_producer(401, |producer| {
            producer
                .set_compute_root_constants_at(0, 0, &[7], 0)
                .expect("root constants");
            producer
                .set_compute_root_buffer_view_at(
                    victory_metal::wire_v3::RootBufferView {
                        slot: 0,
                        allocation_id: 0x9000,
                        generation: 1,
                        offset: 0,
                        length: 16,
                    },
                    victory_shader::emit::ROOT_CBV_SPACE as u16,
                )
                .expect("root view");
        });
        assert_eq!(plan.len(), 2);

        let mut queue = NativeQueue::new(PlanRecordingSink::default());
        let mut frame = submission;
        producer::ensure_sequence(&mut frame).expect("sequence");
        queue
            .submit_with_root_arguments(std::slice::from_ref(&frame), &[plan.as_slice()])
            .expect("the batch carries");

        let sink = queue.sink();
        assert_eq!(sink.frames.len(), 1);
        let (bytes, carried) = &sink.frames[0];
        assert_eq!(
            decode_submission_v3(bytes).expect("the frame is still a v3 submission"),
            frame,
            "the plan never enters the frame"
        );
        let carried = carried.as_ref().expect("the plan rode with the frame");
        assert_eq!(
            bindings::decode_root_arguments(carried).expect("the plan decodes"),
            plan,
            "the sink sees the plan the list recorded"
        );
    }

    /// A plan at the entry cap is carried unchanged, and an empty plan is
    /// carried as the versioned header alone (the host decodes it as present
    /// with no entries) instead of being dropped.
    #[test]
    fn a_capped_plan_and_an_empty_plan_are_both_carried() {
        let (recorded, plan) = compute_producer(402, |producer| {
            for slot in 0..bindings::MAX_ROOT_ARGUMENTS as u16 {
                producer
                    .set_compute_root_constants_at(u32::from(slot), 0, &[1], 0)
                    .expect("root constants at the cap");
            }
        });
        assert_eq!(plan.len(), bindings::MAX_ROOT_ARGUMENTS);

        let mut queue = NativeQueue::new(PlanRecordingSink::default());
        let mut frame = recorded;
        producer::ensure_sequence(&mut frame).expect("sequence");
        queue
            .submit_with_root_arguments(std::slice::from_ref(&frame), &[plan.as_slice()])
            .expect("a plan at the cap carries");
        let carried = queue.sink().frames[0]
            .1
            .as_ref()
            .expect("the capped plan rode with the frame");
        // The documented version 2 shape: a 7-byte header and 15-byte records.
        assert_eq!(
            carried.len(),
            7 + 15 * bindings::MAX_ROOT_ARGUMENTS,
            "the capped plan is the documented record layout"
        );
        assert_eq!(
            bindings::decode_root_arguments(carried)
                .expect("the capped plan decodes")
                .len(),
            bindings::MAX_ROOT_ARGUMENTS
        );

        // Every submission carries a plan section. An empty plan is the
        // versioned header alone, which the host decodes as "present, no
        // entries"; the submission frame itself is unchanged.
        let unplanned = submission(7, 402, 1);
        queue
            .submit_with_root_arguments(std::slice::from_ref(&unplanned), &[&[]])
            .expect("an empty plan is still a plan section");
        let (bytes, carried) = &queue.sink().frames[1];
        let carried = carried.as_ref().expect("the empty plan section is carried");
        assert_eq!(
            carried.len(),
            7,
            "an empty plan is the seven-byte versioned header"
        );
        assert!(
            bindings::decode_root_arguments(carried)
                .expect("an empty plan decodes")
                .is_empty()
        );
        assert_eq!(
            decode_submission_v3(bytes).expect("the frame decodes"),
            unplanned
        );
    }

    /// A sink that predates the plan (the trait's default) still receives the
    /// frame: the plan is dropped, never the submission, and the host's legacy
    /// lookup stays in charge.
    #[test]
    fn a_sink_without_plan_support_still_carries_the_frame() {
        let mut queue = NativeQueue::new(RecordingSink::default());
        let submission = submission(8, 403, 1);
        let plan = vec![bindings::RootArgument::root_view(
            0,
            victory_shader::emit::ROOT_CBV_SPACE as u16,
            16,
        )];
        queue
            .submit_with_root_arguments(std::slice::from_ref(&submission), &[plan.as_slice()])
            .expect("the frame carries");
        assert_eq!(queue.sink().frames.len(), 1);
        assert_eq!(
            decode_submission_v3(&queue.sink().frames[0]).expect("canonical v3"),
            submission
        );
    }

    /// A plan the encoder refuses fails the whole batch before anything is
    /// queued: no partial prefix, no consumed sequence.
    #[test]
    fn a_refused_plan_queues_nothing() {
        let mut queue = NativeQueue::new(RecordingSink::default());
        let submission = submission(9, 404, 1);
        // A sampler table outside the sampler space is not a plan this backend
        // can have translated.
        let plan = vec![bindings::RootArgument::descriptor_table(
            0,
            1,
            0,
            bindings::RootArgumentKind::SamplerTable,
            4,
        )];
        let error = queue
            .submit_with_root_arguments(std::slice::from_ref(&submission), &[plan.as_slice()])
            .expect_err("the plan is refused");
        assert!(
            matches!(&error, NativeQueueError::Plan(why) if why.contains("sampler")),
            "unexpected error {error:?}"
        );
        assert_eq!(queue.pending(), 0, "a refused plan leaves no frame queued");
        assert!(queue.sink().frames.is_empty(), "nothing reached the sink");
    }
}
