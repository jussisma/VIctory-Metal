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
        TelemetrySnapshot {
            frames: self.window_len(),
            recorded: self.recorded_frames(),
            percentiles: self.percentiles(),
            fps: self.fps(),
            opcodes: self.unsupported_opcode_histogram(),
            d3dmetal_fallback_count: self.d3dmetal_fallback_count(),
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
        snapshot
    }

    /// Clears the window and histogram (never the fallback counter).
    pub fn reset(&self) {
        let mut window = self.lock();
        window.frames.clear();
        window.opcodes.clear();
        window.recorded = 0;
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
    /// The session's hard D3DMetal fallback count.
    pub d3dmetal_fallback_count: u64,
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
        if self.opcodes.is_empty() {
            return write!(f, ", opcodes=[]");
        }
        write!(f, ", opcodes=[")?;
        for (index, (opcode, count)) in self.opcodes.iter().enumerate() {
            if index > 0 {
                write!(f, " ")?;
            }
            write!(f, "{opcode}:{count}")?;
        }
        write!(f, "]")
    }
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

/// Declares one D3DMetal fallback in the process-wide telemetry.
pub fn note_d3dmetal_fallback() {
    global().note_d3dmetal_fallback();
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
