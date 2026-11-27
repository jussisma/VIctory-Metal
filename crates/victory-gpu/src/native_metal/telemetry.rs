//! Per-frame telemetry for the native Metal backend (ADR-016).
//!
//! This module owns guest-side frame accounting: record microseconds, ring
//! submit microseconds, frame time percentiles (p50/p95/p99), FPS, the
//! unsupported-opcode histogram and the strict `d3dmetal_fallback_count == 0`
//! guard.
//!
//! # What one sample is
//!
//! [`FrameTelemetry::record_frame`] records one `ExecuteCommandLists` batch:
//! the guest time to record it, the canonical v3 encode time, the IPC
//! round-trip time and the host-reported GPU time. The sample's frame time is
//! their sum, so percentiles describe the instrumented cost of one submitted
//! frame, not a wall-clock interval. [`FrameTelemetry::fps`] is the rate of
//! those samples over the window: it is derived from the recorded timestamps,
//! so it is exact on a fixture and independent of the wall clock in tests
//! ([`FrameTelemetry::record_frame_at`] takes the timestamp explicitly).
//!
//! The window keeps the last [`WINDOW_FRAMES`] samples, so memory is bounded
//! and p99 always has at least 100 samples once a run is that long.
//!
//! # The zero-D3DMetal guard
//!
//! `d3dmetal_fallback_count` is a hard counter: it only moves through
//! [`FrameTelemetry::note_d3dmetal_fallback`], it is never reset by
//! [`FrameTelemetry::snapshot_and_reset`], and [`assert_no_fallback`] refuses a
//! frame while it is nonzero. Nothing on the native path calls it; it exists so
//! that any future fallback hook has to declare itself instead of silently
//! producing a D3DMetal frame (ADR-016). The D3D12 backend never touches this
//! module at all.
//!
//! # Percentiles
//!
//! Exact nearest-rank, no sampling: for `n` ascending samples and percentile
//! `p`, the result is `sorted[ceil(p / 100 * n) - 1]`. A 100-sample fixture
//! `1..=100` therefore answers exactly `50`, `95` and `99`; a single sample
//! answers itself for every percentile; an empty window answers zero.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// How many recent frames the window keeps (all of them, in order).
pub const WINDOW_FRAMES: usize = 4096;

/// Histogram key for a refused call that has no wire opcode.
pub const NO_OPCODE: u16 = u16::MAX;

/// Maximum bytes of a host error reason kept in telemetry and logged.
///
/// Matches [`victory_metal::wire::MAX_REPLY_REASON_BYTES`] so the guest never
/// retains more text than the Wire v3 reply can carry.
pub const MAX_HOST_ERROR_BYTES: usize = 256;

/// One bounded host error carried by a Wire v3 error reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    /// Sequence of the refused submission.
    pub sequence: u32,
    /// Bounded, single-line host diagnostic.
    pub reason: String,
}

/// One recorded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSample {
    /// Monotonic timestamp of the sample, in microseconds.
    pub at_us: u64,
    /// Guest time to record the frame's commands, in microseconds.
    pub record_us: u64,
    /// Ring submit (IPC round trip) time, in microseconds.
    pub ring_submit_us: u64,
    /// Canonical v3 encode time, in microseconds.
    pub encode_us: u64,
    /// Host-reported GPU time, in microseconds (0 when the reply carries none).
    pub gpu_us: u64,
    /// The opcode a refused frame needed, when the producer refused it.
    pub unsupported_opcode: Option<u16>,
}

impl FrameSample {
    /// The frame cost this sample contributes: every instrumented phase.
    pub const fn frame_us(&self) -> u64 {
        self.record_us
            .saturating_add(self.ring_submit_us)
            .saturating_add(self.encode_us)
            .saturating_add(self.gpu_us)
    }
}

/// Exact nearest-rank percentiles of the window, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Percentiles {
    /// Median frame cost.
    pub p50: u64,
    /// 95th percentile frame cost.
    pub p95: u64,
    /// 99th percentile frame cost.
    pub p99: u64,
}

impl Percentiles {
    /// The nearest-rank percentile of ascending `sorted`.
    fn nearest_rank(sorted: &[u64], percent: u64) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let n = sorted.len() as u64;
        // ceil(percent * n / 100), computed in integers so it is exact.
        let rank = (percent * n).div_ceil(100).clamp(1, n);
        sorted[(rank - 1) as usize]
    }
}

/// The strict zero-D3DMetal guard tripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryError {
    /// A frame reached D3DMetal; the native path is no longer exclusive.
    D3dMetalFallback {
        /// Fallbacks recorded so far.
        count: u64,
    },
}

impl TelemetryError {
    /// Fallbacks recorded when this error was produced.
    pub fn fallback_count(&self) -> u64 {
        match self {
            Self::D3dMetalFallback { count } => *count,
        }
    }
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D3dMetalFallback { count } => write!(
                f,
                "{count} frame(s) fell back to D3DMetal; the native path is not exclusive"
            ),
        }
    }
}

impl std::error::Error for TelemetryError {}

/// The window and histogram one [`FrameTelemetry`] keeps.
#[derive(Debug, Default)]
struct Window {
    frames: VecDeque<FrameSample>,
    opcodes: BTreeMap<u16, u64>,
    recorded: u64,
}

/// Per-frame telemetry for one process.
#[derive(Debug, Default)]
pub struct FrameTelemetry {
    window: Mutex<Window>,
    /// Hard counter: only [`Self::note_d3dmetal_fallback`] moves it.
    d3dmetal_fallback_count: AtomicU64,
    /// Native submissions the host answered `Ok` (or a replay) for.
    submits_ok: AtomicU64,
    /// Native submissions the transport or the host refused.
    submits_refused: AtomicU64,
    /// Frames the installed native presenter accepted.
    presents_ok: AtomicU64,
    /// Frames the native presenter (or its absence) refused.
    presents_refused: AtomicU64,
    /// Unsupported producer calls, by the wire opcode they needed.
    refusals: Mutex<BTreeMap<u16, u64>>,
    /// Unsupported producer calls by operation name, with or without a wire
    /// opcode.
    ///
    /// The refusal histogram keys unnamed calls under [`NO_OPCODE`], which
    /// would otherwise merge every one into a single bucket; this map keeps
    /// each refused operation countable by name (ADR-030).
    named_refusals: Mutex<BTreeMap<&'static str, u64>>,
    /// Host allocation registrations the host accepted (kind 2).
    buffers_registered: AtomicU64,
    /// Pipeline registrations the host accepted (kind 4/129).
    psos_registered: AtomicU64,
    /// PSO translations a `--stdio` pool worker ran (cold).
    pool_cold: AtomicU64,
    /// PSO translations the pool's disk cache served (warm).
    pool_warm: AtomicU64,
    /// Buffer or pipeline registrations that failed explicitly.
    registration_failures: AtomicU64,
    /// Wire v3 error replies that carried a host reason trailer.
    host_errors: AtomicU64,
    /// The most recent bounded host error, retained for the telemetry snapshot.
    last_host_error: Mutex<Option<HostError>>,
    /// Compute dispatches recorded with no root view or descriptor table.
    ///
    /// The producer cannot see a pipeline's declared root layout, so it reports
    /// the one fact it has: the recording produced no compute buffer binding
    /// (inline root constants are not buffer bindings). A nonzero count makes an
    /// empty-binding dispatch visible instead of silent.
    compute_dispatches_without_bindings: AtomicU64,
}
impl FrameTelemetry {
    /// Empty telemetry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one frame with the current monotonic timestamp.
    ///
    /// `opcode` names the wire opcode a refused frame needed
    /// ([`crate::native_metal::producer::ProducerError::Unsupported`]); it is
    /// `None` for a frame the producer fully expressed.
    pub fn record_frame(
        &self,
        record_us: u64,
        ring_submit_us: u64,
        encode_us: u64,
        gpu_us: u64,
        opcode: Option<u16>,
    ) {
        self.record_frame_at(
            monotonic_us(),
            record_us,
            ring_submit_us,
            encode_us,
            gpu_us,
            opcode,
        );
    }

    /// Records one frame at an explicit timestamp (tests and replays).
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame_at(
        &self,
        at_us: u64,
        record_us: u64,
        ring_submit_us: u64,
        encode_us: u64,
        gpu_us: u64,
        opcode: Option<u16>,
    ) {
        let mut window = self.lock();
        if let Some(opcode) = opcode {
            *window.opcodes.entry(opcode).or_insert(0) += 1;
        }
        window.frames.push_back(FrameSample {
            at_us,
            record_us,
            ring_submit_us,
            encode_us,
            gpu_us,
            unsupported_opcode: opcode,
        });
        window.recorded = window.recorded.saturating_add(1);
        while window.frames.len() > WINDOW_FRAMES {
            window.frames.pop_front();
        }
    }

    /// The samples in the window, oldest first.
    pub fn samples(&self) -> Vec<FrameSample> {
        self.lock().frames.iter().copied().collect()
    }

    /// The window's frame costs (`record + ring submit + encode + gpu`), oldest
    /// first.
    pub fn frame_times(&self) -> Vec<u64> {
        self.lock()
            .frames
            .iter()
            .map(FrameSample::frame_us)
            .collect()
    }

    /// Exact nearest-rank p50/p95/p99 of the window.
    pub fn percentiles(&self) -> Percentiles {
        let mut times = self.frame_times();
        times.sort_unstable();
        Percentiles {
            p50: Percentiles::nearest_rank(&times, 50),
            p95: Percentiles::nearest_rank(&times, 95),
            p99: Percentiles::nearest_rank(&times, 99),
        }
    }

    /// The window's sample rate: `(n - 1) / (last - first)` timestamps.
    ///
    /// Zero for fewer than two samples or a zero-length window, so a single
    /// frame never divides by zero.
    pub fn fps(&self) -> f64 {
        let window = self.lock();
        if window.frames.len() < 2 {
            return 0.0;
        }
        let first = window.frames.front().map(|frame| frame.at_us).unwrap_or(0);
        let last = window.frames.back().map(|frame| frame.at_us).unwrap_or(0);
        let span = last.saturating_sub(first);
        if span == 0 {
            return 0.0;
        }
        let intervals = (window.frames.len() - 1) as f64;
        intervals * 1_000_000.0 / span as f64
    }

    /// Frames whose producer needed an unsupported wire opcode, by opcode.
    pub fn unsupported_opcode_histogram(&self) -> BTreeMap<u16, u64> {
        self.lock().opcodes.clone()
    }

    /// Frames recorded since construction (not limited to the window).
    pub fn recorded_frames(&self) -> u64 {
        self.lock().recorded
    }

    /// Frames in the window.
    pub fn window_len(&self) -> usize {
        self.lock().frames.len()
    }

    /// The hard D3DMetal fallback count.
    pub fn d3dmetal_fallback_count(&self) -> u64 {
        self.d3dmetal_fallback_count.load(Ordering::Relaxed)
    }

    /// Declares one D3DMetal fallback. The only way the counter moves.
    pub fn note_d3dmetal_fallback(&self) {
        self.d3dmetal_fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one producer refusal by the wire opcode it needed.
    ///
    /// Called the moment a recorded call is refused (not at submission time),
    /// so a list whose `Close` fails still contributes its histogram entry.
    /// An operation with no wire opcode is logged by the caller and counted
    /// under [`NO_OPCODE`].
    pub fn note_unsupported_opcode(&self, opcode: Option<u16>) {
        let opcode = opcode.unwrap_or(NO_OPCODE);
        let mut refusals = self
            .refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *refusals.entry(opcode).or_insert(0) += 1;
    }

    /// Refusals recorded by opcode ([`NO_OPCODE`] for an unnamed call).
    pub fn refusal_histogram(&self) -> BTreeMap<u16, u64> {
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Records one refusal by the operation name it carries.
    ///
    /// Unlike the opcode histogram this keeps calls that share no wire opcode
    /// distinct: `DiscardResource` and a shading-rate hint never share a
    /// bucket. It is recorded alongside the opcode histogram, not instead of
    /// it.
    pub fn note_named_refusal(&self, operation: &'static str) {
        let mut refusals = self
            .named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *refusals.entry(operation).or_insert(0) += 1;
    }

    /// Refusals recorded by operation name.
    pub fn named_refusal_histogram(&self) -> BTreeMap<&'static str, u64> {
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// A native submission batch the host answered successfully.
    pub fn note_submit_ok(&self) {
        self.submits_ok.fetch_add(1, Ordering::Relaxed);
    }

    /// A native submission batch the transport or host refused.
    pub fn note_submit_refused(&self) {
        self.submits_refused.fetch_add(1, Ordering::Relaxed);
    }

    /// A frame the installed native presenter accepted.
    pub fn note_present_ok(&self) {
        self.presents_ok.fetch_add(1, Ordering::Relaxed);
    }

    /// A frame the native present path refused (including no swapchain).
    pub fn note_present_refused(&self) {
        self.presents_refused.fetch_add(1, Ordering::Relaxed);
    }

    /// One buffer registration the host accepted (kind 2).
    pub fn note_buffer_registered(&self) {
        self.buffers_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO registration the host accepted (kind 4/129).
    pub fn note_pso_registered(&self) {
        self.psos_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO translation a pool worker ran (a cold translation).
    pub fn note_pool_cold(&self) {
        self.pool_cold.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO translation the pool's disk cache served (a warm translation).
    pub fn note_pool_warm(&self) {
        self.pool_warm.fetch_add(1, Ordering::Relaxed);
    }

    /// One buffer or pipeline registration that failed explicitly.
    pub fn note_registration_failure(&self) {
        self.registration_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one Wire v3 error reply that carried a host reason.
    ///
    /// The reason is sanitised to one line and bounded to
    /// [`MAX_HOST_ERROR_BYTES`], so neither the counter nor the retained last
    /// reason grows without limit. Returns the bounded record for the one log
    /// line the caller prints.
    pub fn note_host_error(&self, sequence: u32, reason: &str) -> HostError {
        let record = HostError {
            sequence,
            reason: sanitize_host_error(reason),
        };
        self.host_errors.fetch_add(1, Ordering::Relaxed);
        *self
            .last_host_error
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(record.clone());
        record
    }

    /// Wire v3 error replies that carried a host reason.
    pub fn host_error_count(&self) -> u64 {
        self.host_errors.load(Ordering::Relaxed)
    }

    /// The most recent bounded host error, if any.
    pub fn last_host_error(&self) -> Option<HostError> {
        self.last_host_error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// One compute dispatch recorded with no root view or descriptor table.
    pub fn note_compute_dispatch_without_bindings(&self) {
        self.compute_dispatches_without_bindings
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Compute dispatches recorded with no root view or descriptor table.
    pub fn compute_dispatches_without_bindings(&self) -> u64 {
        self.compute_dispatches_without_bindings
            .load(Ordering::Relaxed)
    }

    /// Accepted buffer and PSO registrations.
    pub fn registration_counts(&self) -> (u64, u64) {
        (
            self.buffers_registered.load(Ordering::Relaxed),
            self.psos_registered.load(Ordering::Relaxed),
        )
    }

    /// Cold (worker) and warm (cache) pool translations.
    pub fn pool_translations(&self) -> (u64, u64) {
        (
            self.pool_cold.load(Ordering::Relaxed),
            self.pool_warm.load(Ordering::Relaxed),
        )
    }

    /// Registration failures counted so far.
    pub fn registration_failures(&self) -> u64 {
        self.registration_failures.load(Ordering::Relaxed)
    }

    /// Successful and refused native submission batches.
    pub fn submit_counts(&self) -> (u64, u64) {
        (
            self.submits_ok.load(Ordering::Relaxed),
            self.submits_refused.load(Ordering::Relaxed),
        )
    }

    /// Accepted and refused native presents.
    pub fn present_counts(&self) -> (u64, u64) {
        (
            self.presents_ok.load(Ordering::Relaxed),
            self.presents_refused.load(Ordering::Relaxed),
        )
    }

    /// The strict guard: `Ok` only while no frame fell back to D3DMetal.
    pub fn assert_no_fallback(&self) -> Result<(), TelemetryError> {
        let count = self.d3dmetal_fallback_count();
        if count == 0 {
            Ok(())
        } else {
            Err(TelemetryError::D3dMetalFallback { count })
        }
    }

    /// The window, percentiles and histogram, without resetting anything.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let (submits_ok, submits_refused) = self.submit_counts();
        let (presents_ok, presents_refused) = self.present_counts();
        TelemetrySnapshot {
            frames: self.window_len(),
            recorded: self.recorded_frames(),
            percentiles: self.percentiles(),
            fps: self.fps(),
            opcodes: self.unsupported_opcode_histogram(),
            refusals: self.refusal_histogram(),
            named_refusals: self.named_refusal_histogram(),
            d3dmetal_fallback_count: self.d3dmetal_fallback_count(),
            submits_ok,
            submits_refused,
            presents_ok,
            presents_refused,
            buffers_registered: self.buffers_registered.load(Ordering::Relaxed),
            psos_registered: self.psos_registered.load(Ordering::Relaxed),
            pool_cold: self.pool_cold.load(Ordering::Relaxed),
            pool_warm: self.pool_warm.load(Ordering::Relaxed),
            registration_failures: self.registration_failures.load(Ordering::Relaxed),
            compute_dispatches_without_bindings: self.compute_dispatches_without_bindings(),
            pipeline_census: crate::native_metal::pso::census(),
            host_errors: self.host_error_count(),
            last_host_error: self.last_host_error(),
        }
    }

    /// Takes the window and histogram for one log line and clears them.
    ///
    /// The hard fallback counter is deliberately not cleared: it is an
    /// invariant for the whole session, not window state.
    pub fn snapshot_and_reset(&self) -> TelemetrySnapshot {
        let snapshot = self.snapshot();
        {
            let mut window = self.lock();
            window.frames.clear();
            window.opcodes.clear();
        }
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        snapshot
    }

    /// Clears the window and histograms (never the fallback counter).
    ///
    /// Submit and present counters are session totals and survive a reset too.
    pub fn reset(&self) {
        let mut window = self.lock();
        window.frames.clear();
        window.opcodes.clear();
        window.recorded = 0;
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Window> {
        self.window
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// One frame window, ready for the one-line log.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySnapshot {
    /// Frames in the window when it was taken.
    pub frames: usize,
    /// Frames recorded before it was taken.
    pub recorded: u64,
    /// p50/p95/p99 frame cost of the window, in microseconds.
    pub percentiles: Percentiles,
    /// Sample rate of the window.
    pub fps: f64,
    /// Unsupported-opcode histogram of the window.
    pub opcodes: BTreeMap<u16, u64>,
    /// Refused producer calls by opcode ([`NO_OPCODE`] when none applies).
    pub refusals: BTreeMap<u16, u64>,
    /// Refused producer calls by operation name, including those without a
    /// wire opcode.
    pub named_refusals: BTreeMap<&'static str, u64>,
    /// The session's hard D3DMetal fallback count.
    pub d3dmetal_fallback_count: u64,
    /// Native submission batches the host answered successfully.
    pub submits_ok: u64,
    /// Native submission batches the transport or host refused.
    pub submits_refused: u64,
    /// Frames the native presenter accepted.
    pub presents_ok: u64,
    /// Frames the native present path refused.
    pub presents_refused: u64,
    /// Host buffer registrations the host accepted (kind 2).
    pub buffers_registered: u64,
    /// PSO registrations the host accepted (kind 4/129).
    pub psos_registered: u64,
    /// PSO translations a pool worker ran (cold).
    pub pool_cold: u64,
    /// PSO translations the pool's disk cache served (warm).
    pub pool_warm: u64,
    /// Buffer or pipeline registrations that failed explicitly.
    pub registration_failures: u64,
    /// Compute dispatches recorded with no root view or descriptor table.
    pub compute_dispatches_without_bindings: u64,
    /// Native pipeline attempts, creations, reuses and refusals by kind
    /// (session totals: a pipeline made in an early frame stays counted).
    pub pipeline_census: crate::native_metal::pso::PipelineCensusSnapshot,
    /// Wire v3 error replies that carried a host reason.
    pub host_errors: u64,
    /// The most recent bounded host error, if any.
    pub last_host_error: Option<HostError>,
}

impl std::fmt::Display for TelemetrySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "native frame: {} in window ({}) {:.3}/{:.3}/{:.3} ms p50/p95/p99, {:.1} fps, \
             fallback={}",
            self.frames,
            self.recorded,
            self.percentiles.p50 as f64 / 1000.0,
            self.percentiles.p95 as f64 / 1000.0,
            self.percentiles.p99 as f64 / 1000.0,
            self.fps,
            self.d3dmetal_fallback_count
        )?;
        write!(
            f,
            ", submits={}/{} ok/refused, presents={}/{} ok/refused",
            self.submits_ok, self.submits_refused, self.presents_ok, self.presents_refused
        )?;
        write!(
            f,
            ", registered={} buffers/{} psos, pool={} cold/{} warm, registration_failures={}, \
             compute_unbound={}",
            self.buffers_registered,
            self.psos_registered,
            self.pool_cold,
            self.pool_warm,
            self.registration_failures,
            self.compute_dispatches_without_bindings
        )?;
        write!(f, ", {}", self.pipeline_census.fields())?;
        write!(f, "{}", opcode_list(&self.opcodes, "opcodes"))?;
        write!(f, "{}", opcode_list(&self.refusals, "refusals"))?;
        if self.named_refusals.is_empty() {
            write!(f, ", named_refusals=[]")?;
        } else {
            write!(f, ", named_refusals=[")?;
            for (index, (operation, count)) in self.named_refusals.iter().enumerate() {
                if index > 0 {
                    write!(f, "; ")?;
                }
                write!(f, "{operation}={count}")?;
            }
            write!(f, "]")?;
        }
        write!(f, ", host_errors={}", self.host_errors)?;
        if let Some(error) = &self.last_host_error {
            write!(
                f,
                ", last_host_error=sequence {}: {}",
                error.sequence, error.reason
            )?;
        }
        Ok(())
    }
}

/// One `, name=[opcode:count ...]` field for the frame line.
fn opcode_list(opcodes: &BTreeMap<u16, u64>, name: &str) -> String {
    if opcodes.is_empty() {
        return format!(", {name}=[]");
    }
    let mut text = format!(", {name}=[");
    for (index, (opcode, count)) in opcodes.iter().enumerate() {
        if index > 0 {
            text.push(' ');
        }
        if *opcode == NO_OPCODE {
            text.push_str("none");
        } else {
            text.push_str(&opcode.to_string());
        }
        text.push(':');
        text.push_str(&count.to_string());
    }
    text.push(']');
    text
}

/// Sanitises a host diagnostic into one bounded UTF-8 line.
///
/// Control characters become single spaces, whitespace is trimmed and the text
/// is truncated on a character boundary at [`MAX_HOST_ERROR_BYTES`]. Empty text
/// becomes a stable placeholder so the log line never ends in `reason=`.
pub fn sanitize_host_error(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len().min(MAX_HOST_ERROR_BYTES));
    for character in text.chars() {
        let character = if character.is_control() {
            ' '
        } else {
            character
        };
        if cleaned.len() + character.len_utf8() > MAX_HOST_ERROR_BYTES {
            break;
        }
        if character == ' ' && (cleaned.is_empty() || cleaned.ends_with(' ')) {
            continue;
        }
        cleaned.push(character);
    }
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        "unknown host error".to_string()
    } else {
        cleaned.to_string()
    }
}

/// The stable single guest log line for one Wire v3 host error.
///
/// The format is fixed so an operator can grep a boot log for
/// `METAL_HOST_ERROR sequence=` and read the exact host/Metal text.
pub fn host_error_line(sequence: u32, reason: &str) -> String {
    format!("METAL_HOST_ERROR sequence={sequence} reason={reason}")
}

/// The process-wide telemetry the entry points record into.
pub fn global() -> &'static FrameTelemetry {
    static TELEMETRY: OnceLock<FrameTelemetry> = OnceLock::new();
    TELEMETRY.get_or_init(FrameTelemetry::new)
}

/// Records one frame into the process-wide telemetry.
pub fn record_frame(
    record_us: u64,
    ring_submit_us: u64,
    encode_us: u64,
    gpu_us: u64,
    opcode: Option<u16>,
) {
    global().record_frame(record_us, ring_submit_us, encode_us, gpu_us, opcode);
}

/// Records one frame at an explicit timestamp into the process-wide telemetry.
pub fn record_frame_at(
    at_us: u64,
    record_us: u64,
    ring_submit_us: u64,
    encode_us: u64,
    gpu_us: u64,
    opcode: Option<u16>,
) {
    global().record_frame_at(at_us, record_us, ring_submit_us, encode_us, gpu_us, opcode);
}

/// The process-wide window's frame costs, oldest first.
pub fn frame_times() -> Vec<u64> {
    global().frame_times()
}

/// The process-wide exact p50/p95/p99.
pub fn percentiles() -> Percentiles {
    global().percentiles()
}

/// The process-wide sample rate.
pub fn fps() -> f64 {
    global().fps()
}

/// The process-wide unsupported-opcode histogram.
pub fn unsupported_opcode_histogram() -> BTreeMap<u16, u64> {
    global().unsupported_opcode_histogram()
}

/// The process-wide hard D3DMetal fallback count.
pub fn d3dmetal_fallback_count() -> u64 {
    global().d3dmetal_fallback_count()
}

/// One buffer registration the host accepted (kind 2).
pub fn note_buffer_registered() {
    global().note_buffer_registered();
}

/// One PSO registration the host accepted (kind 4/129).
pub fn note_pso_registered() {
    global().note_pso_registered();
}

/// One cold (worker) PSO translation.
pub fn note_pool_cold() {
    global().note_pool_cold();
}

/// One warm (disk-cache) PSO translation.
pub fn note_pool_warm() {
    global().note_pool_warm();
}

/// One buffer or pipeline registration that failed explicitly.
pub fn note_registration_failure() {
    global().note_registration_failure();
}

/// Records one Wire v3 host error reason in the process-wide telemetry.
pub fn note_host_error(sequence: u32, reason: &str) -> HostError {
    global().note_host_error(sequence, reason)
}

/// The process-wide count of Wire v3 error replies that carried a reason.
pub fn host_error_count() -> u64 {
    global().host_error_count()
}

/// The process-wide most recent bounded host error.
pub fn last_host_error() -> Option<HostError> {
    global().last_host_error()
}

/// One compute dispatch recorded with no root view or descriptor table.
pub fn note_compute_dispatch_without_bindings() {
    global().note_compute_dispatch_without_bindings();
}

/// The process-wide accepted buffer and PSO registration counts.
pub fn registration_counts() -> (u64, u64) {
    global().registration_counts()
}

/// The process-wide cold/warm pool translation counts.
pub fn pool_translations() -> (u64, u64) {
    global().pool_translations()
}

/// The process-wide registration failure count.
pub fn registration_failures() -> u64 {
    global().registration_failures()
}

/// The process-wide count of compute dispatches without a root binding.
pub fn compute_dispatches_without_bindings() -> u64 {
    global().compute_dispatches_without_bindings()
}

/// Declares one D3DMetal fallback in the process-wide telemetry.
pub fn note_d3dmetal_fallback() {
    global().note_d3dmetal_fallback();
}

/// Records one refused producer call by the wire opcode it needed.
pub fn note_unsupported_opcode(opcode: Option<u16>) {
    global().note_unsupported_opcode(opcode);
}

/// The process-wide refusal histogram.
pub fn refusal_histogram() -> BTreeMap<u16, u64> {
    global().refusal_histogram()
}

/// Records one refused producer call by its operation name.
pub fn note_named_refusal(operation: &'static str) {
    global().note_named_refusal(operation);
}

/// The process-wide refusal histogram by operation name.
pub fn named_refusal_histogram() -> BTreeMap<&'static str, u64> {
    global().named_refusal_histogram()
}

/// Records one successful native submission batch.
pub fn note_submit_ok() {
    global().note_submit_ok();
}

/// Records one refused native submission batch.
pub fn note_submit_refused() {
    global().note_submit_refused();
}

/// Records one accepted native present.
pub fn note_present_ok() {
    global().note_present_ok();
}

/// Records one refused native present.
pub fn note_present_refused() {
    global().note_present_refused();
}

/// The process-wide strict zero-D3DMetal guard.
pub fn assert_no_fallback() -> Result<(), TelemetryError> {
    global().assert_no_fallback()
}

/// The process-wide snapshot, without resetting.
pub fn snapshot() -> TelemetrySnapshot {
    global().snapshot()
}

/// Takes the process-wide window for one line and clears it.
pub fn snapshot_and_reset() -> TelemetrySnapshot {
    global().snapshot_and_reset()
}

/// Logs the process-wide window as one line and starts a new window.
pub fn log_frame() {
    let snapshot = snapshot_and_reset();
    tracing::info!(%snapshot, "native frame telemetry");
}

/// Microseconds since the first call, for sample timestamps.
fn monotonic_us() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn telemetry() -> FrameTelemetry {
        FrameTelemetry::new()
    }

    /// A frame whose only cost is `record_us`.
    fn frame(record_us: u64) {
        global().record_frame_at(0, record_us, 0, 0, 0, None);
    }

    #[test]
    fn percentiles_are_exact_on_a_hundred_sample_fixture() {
        let telemetry = telemetry();
        for cost in 1..=100u64 {
            telemetry.record_frame_at(cost, cost, 0, 0, 0, None);
        }
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 50,
                p95: 95,
                p99: 99
            }
        );
        assert_eq!(telemetry.recorded_frames(), 100);
        assert_eq!(telemetry.frame_times().len(), 100);
        assert_eq!(telemetry.frame_times()[99], 100, "oldest first");
    }

    #[test]
    fn one_sample_answers_every_percentile() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 7, 0, 0, 0, None);
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 7,
                p95: 7,
                p99: 7
            }
        );
        assert_eq!(telemetry.fps(), 0.0, "one sample has no interval");
    }

    #[test]
    fn two_samples_use_nearest_rank() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 20, 0, 0, 0, None);
        telemetry.record_frame_at(1, 10, 0, 0, 0, None);
        // Sorted: [10, 20]. ceil(0.5 * 2) = 1 -> 10; ceil(0.95 * 2) = 2 -> 20.
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 10,
                p95: 20,
                p99: 20
            }
        );
    }

    #[test]
    fn an_empty_or_zero_length_window_is_zero_instead_of_panicking() {
        let telemetry = telemetry();
        assert_eq!(telemetry.percentiles(), Percentiles::default());
        assert_eq!(telemetry.fps(), 0.0);
        assert_eq!(telemetry.frame_times(), Vec::<u64>::new());
        assert_eq!(telemetry.recorded_frames(), 0);

        // Two samples at the same instant: no interval, no division by zero.
        telemetry.record_frame_at(50, 1, 0, 0, 0, None);
        telemetry.record_frame_at(50, 2, 0, 0, 0, None);
        assert_eq!(telemetry.fps(), 0.0);
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 1,
                p95: 2,
                p99: 2
            }
        );
    }

    #[test]
    fn fps_is_the_window_sample_rate() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 1, 0, 0, 0, None);
        telemetry.record_frame_at(1_000_000, 1, 0, 0, 0, None);
        telemetry.record_frame_at(2_000_000, 1, 0, 0, 0, None);
        assert_eq!(telemetry.fps(), 1.0, "two intervals over two seconds");
    }

    #[test]
    fn the_frame_time_is_the_sum_of_every_instrumented_phase() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 100, 200, 300, 400, None);
        assert_eq!(telemetry.frame_times(), vec![1000]);
        let sample = telemetry.samples()[0];
        assert_eq!(sample.record_us, 100);
        assert_eq!(sample.ring_submit_us, 200);
        assert_eq!(sample.encode_us, 300);
        assert_eq!(sample.gpu_us, 400);
        assert_eq!(sample.unsupported_opcode, None);
    }

    #[test]
    fn the_opcode_histogram_counts_only_named_opcodes() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 1, 0, 0, 0, Some(7));
        telemetry.record_frame_at(1, 1, 0, 0, 0, Some(7));
        telemetry.record_frame_at(2, 1, 0, 0, 0, Some(9));
        telemetry.record_frame_at(3, 1, 0, 0, 0, None);
        assert_eq!(
            telemetry.unsupported_opcode_histogram(),
            BTreeMap::from([(7, 2), (9, 1)])
        );
        assert_eq!(telemetry.window_len(), 4);
    }

    #[test]
    fn the_fallback_counter_only_moves_through_its_note() {
        let telemetry = telemetry();
        assert_eq!(telemetry.d3dmetal_fallback_count(), 0);
        assert_eq!(telemetry.assert_no_fallback(), Ok(()));

        telemetry.record_frame_at(0, 1, 0, 0, 0, None);
        telemetry.snapshot_and_reset();
        telemetry.reset();
        assert_eq!(
            telemetry.d3dmetal_fallback_count(),
            0,
            "recording and resetting never move the hard counter"
        );

        telemetry.note_d3dmetal_fallback();
        assert_eq!(telemetry.d3dmetal_fallback_count(), 1);
        assert_eq!(
            telemetry.assert_no_fallback(),
            Err(TelemetryError::D3dMetalFallback { count: 1 })
        );
        telemetry.snapshot_and_reset();
        assert_eq!(
            telemetry.d3dmetal_fallback_count(),
            1,
            "the snapshot never clears the hard counter"
        );
    }

    #[test]
    fn snapshot_and_reset_returns_the_window_and_starts_a_new_one() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 10, 0, 0, 0, Some(3));
        telemetry.record_frame_at(1, 30, 0, 0, 0, None);
        let snapshot = telemetry.snapshot_and_reset();
        assert_eq!(snapshot.frames, 2);
        assert_eq!(snapshot.recorded, 2);
        assert_eq!(
            snapshot.percentiles,
            Percentiles {
                p50: 10,
                p95: 30,
                p99: 30
            }
        );
        assert_eq!(snapshot.opcodes, BTreeMap::from([(3, 1)]));
        assert_eq!(snapshot.d3dmetal_fallback_count, 0);
        let line = snapshot.to_string();
        assert!(
            !line.contains('\n'),
            "the log line must be one line: {line}"
        );
        assert!(line.contains("fallback=0"), "{line}");
        assert!(line.contains("opcodes=[3:1]"), "{line}");

        assert_eq!(telemetry.window_len(), 0);
        assert_eq!(telemetry.unsupported_opcode_histogram(), BTreeMap::new());
        assert_eq!(telemetry.recorded_frames(), 2, "totals survive a reset");
    }

    #[test]
    fn the_window_is_bounded() {
        let telemetry = telemetry();
        for cost in 0..(WINDOW_FRAMES as u64 + 10) {
            telemetry.record_frame_at(cost, cost, 0, 0, 0, None);
        }
        assert_eq!(telemetry.window_len(), WINDOW_FRAMES);
        assert_eq!(telemetry.recorded_frames() as usize, WINDOW_FRAMES + 10);
        assert_eq!(telemetry.frame_times()[0], 10, "the oldest samples dropped");
    }

    /// The snapshot names the unsupported opcodes and the submit/present
    /// statuses the native path observed, with the hard fallback count.
    /// An empty-binding compute dispatch is a named count, not silence.
    #[test]
    fn unbound_compute_dispatches_are_counted_by_name() {
        let telemetry = telemetry();
        assert_eq!(telemetry.compute_dispatches_without_bindings(), 0);
        telemetry.note_compute_dispatch_without_bindings();
        telemetry.note_compute_dispatch_without_bindings();
        assert_eq!(telemetry.compute_dispatches_without_bindings(), 2);

        let line = telemetry.snapshot().to_string();
        assert!(line.contains("compute_unbound=2"), "{line}");
        telemetry.reset();
        assert_eq!(
            telemetry.compute_dispatches_without_bindings(),
            2,
            "the count is a session total, like the submit counters"
        );
    }

    /// The snapshot names the pipeline-attempt census: how many attempts the
    /// title made, how many became host pipelines and, when one was refused,
    /// the static name and bounded reason of the first refusal. A boot can then
    /// tell "no pipeline asked for yet" from "every request refused".
    #[test]
    fn the_snapshot_carries_the_pipeline_attempt_census() {
        use crate::native_metal::pso;

        pso::reset_census();
        pso::note_created(pso::AttemptKind::Graphics);
        pso::note_translator_unavailable(
            pso::AttemptKind::Compute,
            pso::RefusalClass::ConverterUnavailable,
            "shader pool has no worker program configured",
        );

        let snapshot = telemetry().snapshot();
        assert_eq!(snapshot.pipeline_census.attempts, 2);
        assert_eq!(snapshot.pipeline_census.created, 1);
        assert_eq!(snapshot.pipeline_census.refused, 0);
        assert_eq!(snapshot.pipeline_census.translator_unavailable, 1);
        let line = snapshot.to_string();
        for expected in [
            "pipeline_attempts=2",
            "pipeline_created=1",
            "pipeline_refused=0",
            "pipeline_translator_unavailable=1",
            "pipeline_first_refusal=compute/converter-unavailable",
        ] {
            assert!(line.contains(expected), "{expected} missing from {line}");
        }
        assert!(!line.contains('\n'), "the log line must stay one line: {line}");
    }

    #[test]
    fn the_snapshot_carries_refusals_and_submit_present_statuses() {
        let telemetry = telemetry();
        telemetry.note_unsupported_opcode(Some(44));
        telemetry.note_unsupported_opcode(Some(44));
        telemetry.note_unsupported_opcode(None);
        telemetry.note_submit_ok();
        telemetry.note_submit_refused();
        telemetry.note_present_ok();
        telemetry.note_present_refused();
        telemetry.note_present_refused();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.refusals, BTreeMap::from([(44, 2), (NO_OPCODE, 1)]));
        assert_eq!((snapshot.submits_ok, snapshot.submits_refused), (1, 1));
        assert_eq!((snapshot.presents_ok, snapshot.presents_refused), (1, 2));
        assert_eq!(snapshot.d3dmetal_fallback_count, 0);
        let line = snapshot.to_string();
        assert!(line.contains("refusals=[44:2 none:1]"), "{line}");
        assert!(line.contains("submits=1/1"), "{line}");
        assert!(line.contains("presents=1/2"), "{line}");
        assert!(line.contains("fallback=0"), "{line}");
        // The session totals survive a window reset.
        telemetry.reset();
        let after = telemetry.snapshot();
        assert_eq!(after.refusals, BTreeMap::new());
        assert_eq!((after.submits_ok, after.submits_refused), (1, 1));
        assert_eq!((after.presents_ok, after.presents_refused), (1, 2));
    }

    /// ADR-030: a refusal without a wire opcode is still counted by the
    /// operation name, so two different console states never share a bucket.
    #[test]
    fn unnamed_refusals_are_counted_by_operation_name() {
        let telemetry = telemetry();
        telemetry.note_unsupported_opcode(None);
        telemetry.note_named_refusal("DiscardResource: a partial discard");
        telemetry.note_named_refusal("DiscardResource: a partial discard");
        telemetry.note_named_refusal("SetShadingRate: no Wire v3 command");

        assert_eq!(
            telemetry.named_refusal_histogram(),
            BTreeMap::from([
                ("DiscardResource: a partial discard", 2),
                ("SetShadingRate: no Wire v3 command", 1),
            ])
        );
        // The opcode histogram keeps its own unchanged view.
        assert_eq!(telemetry.refusal_histogram(), BTreeMap::from([(NO_OPCODE, 1)]));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.named_refusals.len(), 2);
        assert!(
            snapshot.to_string().contains("named_refusals=["),
            "{snapshot}"
        );

        telemetry.snapshot_and_reset();
        assert_eq!(telemetry.named_refusal_histogram(), BTreeMap::new());
    }

    /// A host error is bounded, flattened to one line, counted and retained as
    /// the last reason; the snapshot names both without unbounded growth.
    #[test]
    fn host_errors_are_bounded_counted_and_named_once() {
        let telemetry = telemetry();
        assert_eq!(telemetry.host_error_count(), 0);
        assert_eq!(telemetry.last_host_error(), None);

        let short = "Metal execution failed: target texture 640x360 was refused";
        let record = telemetry.note_host_error(7, short);
        assert_eq!(record.sequence, 7);
        assert_eq!(record.reason, short);
        assert_eq!(telemetry.host_error_count(), 1);
        assert_eq!(
            telemetry.last_host_error(),
            Some(HostError {
                sequence: 7,
                reason: short.to_string()
            })
        );
        assert_eq!(
            host_error_line(record.sequence, &record.reason),
            format!("METAL_HOST_ERROR sequence=7 reason={short}")
        );

        let long = "x".repeat(MAX_HOST_ERROR_BYTES * 2);
        let record = telemetry.note_host_error(8, &long);
        assert_eq!(record.reason.len(), MAX_HOST_ERROR_BYTES);
        assert!(record.reason.is_char_boundary(record.reason.len()));

        let unicode = "é".repeat(MAX_HOST_ERROR_BYTES);
        let record = telemetry.note_host_error(9, &unicode);
        assert!(record.reason.len() <= MAX_HOST_ERROR_BYTES);
        assert!(record.reason.is_char_boundary(record.reason.len()));

        let multiline = telemetry.note_host_error(10, "line one\n\tline two");
        assert_eq!(multiline.reason, "line one line two");

        assert_eq!(telemetry.host_error_count(), 4);
        assert_eq!(
            telemetry.last_host_error().map(|error| error.sequence),
            Some(10)
        );
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.host_errors, 4);
        assert_eq!(snapshot.last_host_error, Some(multiline));
        let line = snapshot.to_string();
        assert!(line.contains("host_errors=4"), "{line}");
        assert!(line.contains("last_host_error=sequence 10"), "{line}");
        assert!(!line.contains('\n'), "the log line must stay one line: {line}");
    }

    #[test]
    fn the_global_recorders_reach_the_process_telemetry() {
        global().reset();
        frame(5);
        frame(9);
        assert_eq!(frame_times(), vec![5, 9]);
        assert_eq!(
            percentiles(),
            Percentiles {
                p50: 5,
                p95: 9,
                p99: 9
            }
        );
        assert_eq!(d3dmetal_fallback_count(), 0);
        assert_eq!(assert_no_fallback(), Ok(()));
        let snapshot = snapshot_and_reset();
        assert_eq!(snapshot.frames, 2);
        assert_eq!(global().window_len(), 0);
    }
}
