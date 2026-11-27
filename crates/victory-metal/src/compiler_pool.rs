//! Bounded multi-process Rosetta IRConverter worker pool (ADR-014).
//!
//! Apple's `libmetalirconverter.dylib` is `x86_64`-only, so DXIL translation
//! runs in `victory-irconverter` child processes under Rosetta 2. That call is
//! serial and dominates first-run shader stalls; this module keeps a bounded
//! pool of those processes and dispatches independent jobs to them, while a
//! content-hashed disk cache (see [`crate::macos::cache`]) absorbs repeat
//! translations on warm runs.
//!
//! # Sizing and configuration
//!
//! [`PoolConfig::from_env`] reads:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `VICTORY_SHADER_WORKERS` | child processes, clamped to `1..=8` | `4` |
//! | `VICTORY_SHADER_QUEUE_DEPTH` | jobs accepted but not completed | `64` |
//! | `VICTORY_SHADER_JOB_TIMEOUT_MS` | per-job wall-clock timeout | `30000` |
//!
//! Invalid values (non-numeric, empty, out of range) fall back to the default
//! instead of panicking. `VICTORY_METAL_SHADER_CACHE` selects the disk cache
//! directory, matching [`crate::macos::irconverter`].
//!
//! # Worker protocol (version 1)
//!
//! Each worker is a child process started as `victory-irconverter --stdio`
//! (see [`pool_stdio_enabled`]) and driven over stdin/stdout with
//! length-prefixed frames. When `VICTORY_METAL_IRCONVERTER` or `D3DMETAL_RES`
//! names a converter library, the pool also passes `--converter <path>`; a
//! worker that rejects `--stdio` exits before answering and is reported as
//! [`PoolError::WorkerExited`] on the affected job, never silently replaced by
//! the one-shot path. The protocol version is implicit in the header tokens.
//!
//! Request, pool -> worker stdin:
//!
//! ```text
//! "VMJOB" SP <seq> SP <job-id> SP <stage> SP <entry-len> SP <dxil-len> LF
//! <entry: entry-len raw bytes><dxil: dxil-len raw bytes>
//! ```
//!
//! Response, worker -> pool stdout:
//!
//! ```text
//! "VMRES" SP <seq> SP <job-id> SP <status> SP <payload-len> LF
//! <payload: payload-len raw bytes>
//! ```
//!
//! `<seq>` and `<job-id>` are decimal `u64`, `<stage>` is a decimal `u32`,
//! and `<entry-len>`, `<dxil-len>` and `<payload-len>` are decimal byte
//! counts. `<seq>` is the pool's internal monotonic sequence number and is
//! what the pool uses for routing and cancellation; `<job-id>` is the caller's
//! id. `status == 0` means `payload` is the serialized metallib; any other
//! status means the payload is a UTF-8 diagnostic. The worker must echo both
//! numbers: the pool rejects a mismatched sequence or caller id as
//! [`PoolError::SequenceMismatch`] or [`PoolError::IdMismatch`] instead of
//! accepting another caller's bytes.
//!
//! Requests and responses are strictly alternating on one pipe pair, so the
//! pool never needs a worker-side scheduler. A worker that exits, closes its
//! stdout or violates the frame layout is reported as an explicit error, its
//! process is reaped, and the slot is respawned for the next job; a caller
//! that exceeds `job_timeout` cancels its job, kills the worker and receives
//! [`PoolError::Timeout`], and the pool remains usable.
//!
//! # Cache
//!
//! [`CompilerPool::compile`] consults the content-hashed disk cache first. The
//! key is [`crate::macos::cache::TranslationKey`] built from the job's DXIL,
//! stage and entry point plus the pool converter version; a hit returns the
//! stored metallib without starting or waking any worker, a miss fills the
//! cache after the worker replies.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Largest worker count the pool will ever start.
pub const MAX_WORKERS: usize = 8;

/// Default worker count when `VICTORY_SHADER_WORKERS` is unset or invalid.
pub const DEFAULT_WORKERS: usize = 4;

/// Default number of accepted-but-incomplete jobs.
pub const DEFAULT_QUEUE_DEPTH: usize = 64;

/// Default per-job timeout.
pub const DEFAULT_JOB_TIMEOUT_MS: u64 = 30_000;

/// Largest accepted DXIL input, matching the converter's own limit.
pub const MAX_JOB_DXIL_BYTES: usize = 16 * 1024 * 1024;

/// Largest accepted entry-point name, matching the converter's own limit.
pub const MAX_JOB_ENTRY_BYTES: usize = 256;

/// Largest accepted worker payload, matching the serialized metallib limit.
pub const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;

/// Version mixed into the pool's cache key.
pub const POOL_CONVERTER_VERSION: &str = concat!(
    "victory-metal/",
    env!("CARGO_PKG_VERSION"),
    "/shader-pool-1"
);

#[cfg(unix)]
unsafe extern "C" {
    /// POSIX `kill(2)`; a negative pid targets a process group.
    fn kill(pid: i32, signal: i32) -> i32;
}

#[cfg(unix)]
const SIGKILL: i32 = 9;

const PROTOCOL_MAGIC_JOB: &str = "VMJOB";
const PROTOCOL_MAGIC_RESULT: &str = "VMRES";
const MAX_FRAME_HEADER_BYTES: usize = 4096;
/// Argument that selects the worker's persistent framed protocol.
const STDIO_ARG: &str = "--stdio";
/// Argument that names the x86_64 IRConverter library to the worker.
const CONVERTER_ARG: &str = "--converter";

/// Whether this build starts workers in persistent `--stdio` mode.
///
/// It is always true: the pool no longer has a one-shot fallback, so a worker
/// that does not understand `--stdio` fails explicitly instead of being driven
/// once per job. Tests use this to assert the transport.
pub const fn pool_stdio_enabled() -> bool {
    true
}

/// Converter library passed to a `--stdio` worker, taken from the same
/// environment the renderer uses. `None` lets the worker report the missing
/// converter itself, which surfaces as [`PoolError::WorkerExited`].
fn converter_from_env() -> Option<PathBuf> {
    std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                .filter(|path| path.is_file())
        })
}

/// Pool sizing, timeout and cache configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    /// Child processes to keep, clamped to `1..=8` by [`CompilerPool::new`].
    pub workers: usize,
    /// Jobs accepted but not yet completed before [`CompilerPool::compile`]
    /// blocks. Must be at least 1.
    pub queue_depth: usize,
    /// Per-job wall-clock timeout.
    pub job_timeout: Duration,
    /// Disk cache directory; `None` disables the cache.
    pub cache_dir: Option<PathBuf>,
    /// Converter/protocol version mixed into the cache key.
    pub converter_version: String,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            workers: DEFAULT_WORKERS,
            queue_depth: DEFAULT_QUEUE_DEPTH,
            job_timeout: Duration::from_millis(DEFAULT_JOB_TIMEOUT_MS),
            cache_dir: default_cache_dir(),
            converter_version: POOL_CONVERTER_VERSION.to_string(),
        }
    }
}

impl PoolConfig {
    /// Reads the pool configuration from the documented environment
    /// variables, falling back to defaults for missing or invalid values.
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(raw) = std::env::var("VICTORY_SHADER_WORKERS") {
            config.workers = clamp_workers(Some(&raw));
        }
        if let Ok(raw) = std::env::var("VICTORY_SHADER_QUEUE_DEPTH") {
            config.queue_depth = parse_bounded(Some(&raw), DEFAULT_QUEUE_DEPTH, 1, 4096);
        }
        if let Ok(raw) = std::env::var("VICTORY_SHADER_JOB_TIMEOUT_MS") {
            let millis = parse_bounded(
                Some(&raw),
                DEFAULT_JOB_TIMEOUT_MS as usize,
                1,
                60 * 60 * 1000,
            );
            config.job_timeout =
                Duration::from_millis(u64::try_from(millis).unwrap_or(DEFAULT_JOB_TIMEOUT_MS));
        }
        if let Some(dir) = std::env::var_os("VICTORY_METAL_SHADER_CACHE") {
            config.cache_dir = Some(PathBuf::from(dir));
        }
        config
    }
}

/// Clamps `VICTORY_SHADER_WORKERS` to `1..=8`, defaulting on invalid input.
pub fn clamp_workers(raw: Option<&str>) -> usize {
    parse_bounded(raw, DEFAULT_WORKERS, 1, MAX_WORKERS)
}

fn parse_bounded(raw: Option<&str>, default: usize, min: usize, max: usize) -> usize {
    raw.and_then(|text| text.trim().parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

/// Default disk cache directory shared with [`crate::macos::irconverter`].
pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("VICTORY_METAL_SHADER_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join("Library/Caches/victory-metal/shader-cache"))
        })
        .or_else(|| Some(std::env::temp_dir().join("victory-metal-shader-cache")))
}

/// One translation request handed to [`CompilerPool::compile`].
///
/// `id` is the caller's job id. It is echoed in the wire frame and validated on
/// the response, so a result is never attributed to another caller's id.
#[derive(Debug, Clone, Copy)]
pub struct CompileJob<'a> {
    /// Caller-chosen job id, echoed and validated by the worker protocol.
    pub id: u64,
    /// Shader stage reported by the converter (`8` is compute).
    pub stage: u32,
    /// Entry point named in the DXIL module.
    pub entry: &'a str,
    /// Raw DXIL bytes.
    pub dxil: &'a [u8],
}

/// Pool statistics snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total [`CompilerPool::compile`] calls.
    pub jobs: u64,
    /// Jobs answered from the disk cache.
    pub cache_hits: u64,
    /// Jobs that had to run in a worker.
    pub cache_misses: u64,
    /// Worker processes started (a restarted worker counts again).
    pub worker_starts: u64,
    /// Workers reaped after a crash, timeout or protocol error.
    pub worker_restarts: u64,
    /// Jobs that completed successfully, including cache hits.
    pub completed: u64,
    /// Jobs that returned an error.
    pub errors: u64,
    /// Callers currently dispatched to or waiting for a worker.
    pub in_flight: usize,
    /// Jobs actually run inside a worker (cache hits excluded).
    pub worker_jobs: u64,
    /// Sum of worker job wall times, including the first job's process start.
    pub worker_job_micros_total: u64,
    /// Longest single worker job wall time.
    pub worker_job_micros_max: u64,
}

/// Per-worker lifetime counters and wall-time instrumentation.
///
/// `first_job_micros` includes process start and the Rosetta/dylib warm-up;
/// `subsequent_job_micros_total / subsequent_jobs` is the steady-state cost.
/// A persistent `--stdio` worker should show a first job far slower than the
/// average of the rest, which is the amortisation the pool exists for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerTiming {
    /// Jobs answered by this worker slot (across respawns).
    pub jobs: u64,
    /// Wall time of the first job after the slot's first spawn.
    pub first_job_micros: u64,
    /// Jobs after the first one.
    pub subsequent_jobs: u64,
    /// Sum of wall times for the jobs after the first one.
    pub subsequent_job_micros_total: u64,
}

impl WorkerTiming {
    /// Average wall time of jobs after the first, when there is one.
    pub fn subsequent_avg_micros(&self) -> Option<u64> {
        (self.subsequent_jobs > 0).then(|| self.subsequent_job_micros_total / self.subsequent_jobs)
    }
}

/// Explicit pool failure. Every variant is terminal for the affected job only.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// No worker program was configured and the cache missed.
    #[error("shader pool has no worker program configured")]
    NoWorkerProgram,
    /// The job itself is outside the accepted bounds.
    #[error("invalid shader job: {0}")]
    InvalidJob(&'static str),
    /// The worker process could not be started.
    #[error("failed to spawn shader worker {program}: {source}")]
    Spawn {
        /// Program that failed to spawn.
        program: PathBuf,
        /// Underlying spawn error.
        source: std::io::Error,
    },
    /// Writing the request frame failed.
    #[error("failed to send job {job_id} to the shader worker: {source}")]
    WorkerWrite {
        /// Job whose frame failed.
        job_id: u64,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// Reading the response frame failed.
    #[error("failed to read the shader worker response for job {job_id}: {source}")]
    WorkerRead {
        /// Job whose response failed.
        job_id: u64,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// The worker closed stdout before answering, which includes a crash.
    #[error("shader worker exited before answering job {job_id}")]
    WorkerExited {
        /// Job that lost its worker.
        job_id: u64,
    },
    /// The response sequence number did not match the dispatched request.
    #[error("shader worker answered sequence {expected} as sequence {got}")]
    SequenceMismatch {
        /// Sequence sent with the request.
        expected: u64,
        /// Sequence echoed by the worker.
        got: u64,
    },
    /// The response caller id did not match the request id.
    #[error("shader worker answered job {expected} as job {got}")]
    IdMismatch {
        /// Id sent with the request.
        expected: u64,
        /// Id echoed by the worker.
        got: u64,
    },
    /// The worker reported a failed translation.
    #[error("shader worker failed job {job_id} with status {status}: {message}")]
    WorkerFailed {
        /// Job that failed.
        job_id: u64,
        /// Worker status code.
        status: u32,
        /// Worker diagnostic.
        message: String,
    },
    /// The response frame did not follow protocol version 1.
    #[error("shader worker protocol error for job {job_id}: {detail}")]
    Protocol {
        /// Job whose response was malformed.
        job_id: u64,
        /// What was wrong.
        detail: String,
    },
    /// The caller's job exceeded the configured timeout; its worker was killed.
    #[error("shader job {job_id} timed out after {timeout_ms} ms")]
    Timeout {
        /// Job that timed out.
        job_id: u64,
        /// Configured timeout.
        timeout_ms: u64,
    },
    /// The pool has shut down (its worker queues are closed).
    #[error("shader pool is closed")]
    Closed,
}

struct JobFrame {
    seq: u64,
    id: u64,
    stage: u32,
    entry: String,
    dxil: Vec<u8>,
    reply: SyncSender<Result<Vec<u8>, PoolError>>,
}

#[derive(Default)]
struct AtomicStats {
    jobs: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    worker_starts: AtomicU64,
    worker_restarts: AtomicU64,
    completed: AtomicU64,
    errors: AtomicU64,
    in_flight: AtomicUsize,
    worker_jobs: AtomicU64,
    worker_job_micros_total: AtomicU64,
    worker_job_micros_max: AtomicU64,
}

/// Bounded, lazily spawned worker pool with a content-hashed disk cache.
///
/// Clone-free and shareable across threads (`&self`); one dispatcher thread per
/// configured worker receives jobs through its own bounded queue, and the
/// worker process is spawned on that thread's first job. Dropping the pool
/// closes the queues, kills live children and joins the threads.
pub struct CompilerPool {
    config: PoolConfig,
    program: Option<PathBuf>,
    cache: Option<cache_bridge::Slot>,
    slots: Vec<Arc<WorkerSlot>>,
    senders: Vec<SyncSender<JobFrame>>,
    handles: Vec<JoinHandle<()>>,
    gate: Gate,
    stats: Arc<AtomicStats>,
    next_worker: AtomicUsize,
    next_seq: AtomicU64,
}

impl CompilerPool {
    /// Builds a pool. `worker_program` is the child executable that speaks the
    /// documented frame protocol (normally the `victory-irconverter` binary);
    /// `None` leaves the pool cache-only.
    pub fn new(config: PoolConfig, worker_program: Option<PathBuf>) -> Self {
        let workers = config.workers.clamp(1, MAX_WORKERS);
        let queue_depth = config.queue_depth.max(1);
        let stats = Arc::new(AtomicStats::default());
        let cache = config
            .cache_dir
            .as_deref()
            .and_then(cache_bridge::Slot::open);

        let mut slots = Vec::with_capacity(workers);
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let slot = Arc::new(WorkerSlot::default());
            let (tx, rx) = mpsc::sync_channel(queue_depth);
            match worker_program.clone() {
                Some(program) => {
                    let thread_stats = stats.clone();
                    let thread_slot = Arc::clone(&slot);
                    let name = format!("victory-shader-pool-{index}");
                    match thread::Builder::new().name(name).spawn(move || {
                        worker_loop(program, thread_slot, rx, thread_stats);
                    }) {
                        Ok(handle) => {
                            slots.push(slot);
                            senders.push(tx);
                            handles.push(handle);
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                worker = index,
                                "shader pool could not start a dispatcher thread"
                            );
                        }
                    }
                }
                None => {
                    // Cache-only pool: no dispatcher is needed.
                }
            }
        }

        Self {
            config: PoolConfig {
                workers,
                queue_depth,
                ..config
            },
            program: worker_program,
            cache,
            slots,
            senders,
            handles,
            gate: Gate::new(queue_depth),
            stats,
            next_worker: AtomicUsize::new(0),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Configured worker count after clamping.
    pub fn workers(&self) -> usize {
        self.config.workers
    }

    /// Worker program, when one is configured.
    pub fn worker_program(&self) -> Option<&Path> {
        self.program.as_deref()
    }

    /// Builds a pool from [`PoolConfig::from_env`], resolving the worker from
    /// `VICTORY_IRCONVERTER_WORKER` or the `victory-irconverter` sibling of
    /// the current executable. A pool with neither is cache-only.
    pub fn from_env() -> Self {
        let program = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                let exe = std::env::current_exe().ok()?;
                let sibling = exe.parent()?.join("victory-irconverter");
                sibling.is_file().then_some(sibling)
            });
        Self::new(PoolConfig::from_env(), program)
    }

    /// Cache directory actually in use, when caching is enabled and openable.
    pub fn cache_dir(&self) -> Option<&Path> {
        self.cache.as_ref().map(|cache| cache.dir())
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> PoolStats {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        PoolStats {
            jobs: load(&self.stats.jobs),
            cache_hits: load(&self.stats.cache_hits),
            cache_misses: load(&self.stats.cache_misses),
            worker_starts: load(&self.stats.worker_starts),
            worker_restarts: load(&self.stats.worker_restarts),
            completed: load(&self.stats.completed),
            errors: load(&self.stats.errors),
            in_flight: self.stats.in_flight.load(Ordering::Relaxed),
            worker_jobs: load(&self.stats.worker_jobs),
            worker_job_micros_total: load(&self.stats.worker_job_micros_total),
            worker_job_micros_max: load(&self.stats.worker_job_micros_max),
        }
    }

    /// Jobs answered by each worker slot, in slot order.
    ///
    /// A cache hit does not touch a worker, so the counts do not move on a warm
    /// run. The counts span respawns in the same slot.
    pub fn worker_job_counts(&self) -> Vec<u64> {
        self.slots
            .iter()
            .map(|slot| slot.jobs_served.load(Ordering::Relaxed))
            .collect()
    }

    /// Per-slot job counts and wall-time instrumentation.
    pub fn worker_timings(&self) -> Vec<WorkerTiming> {
        self.slots
            .iter()
            .map(|slot| WorkerTiming {
                jobs: slot.jobs_served.load(Ordering::Relaxed),
                first_job_micros: slot.first_job_micros.load(Ordering::Relaxed),
                subsequent_jobs: slot.subsequent_jobs.load(Ordering::Relaxed),
                subsequent_job_micros_total: slot
                    .subsequent_job_micros_total
                    .load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Compiles one job, consulting the disk cache before any worker runs.
    ///
    /// The returned bytes are either the cached metallib or the worker's
    /// payload for exactly `job.id`. Concurrency is bounded by
    /// [`PoolConfig::queue_depth`]; callers beyond the bound block until a
    /// queued job completes.
    pub fn compile(&self, job: &CompileJob<'_>) -> Result<Vec<u8>, PoolError> {
        self.validate(job)?;
        self.stats.jobs.fetch_add(1, Ordering::Relaxed);

        let cache_key = self
            .cache
            .as_ref()
            .map(|_| cache_bridge::Slot::key(job, &self.config.converter_version));
        if let (Some(cache), Some(key)) = (&self.cache, &cache_key) {
            if let Some(metallib) = cache.get(key) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.stats.completed.fetch_add(1, Ordering::Relaxed);
                return Ok(metallib);
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        if self.program.is_none() || self.senders.is_empty() {
            self.stats.errors.fetch_add(1, Ordering::Relaxed);
            return Err(PoolError::NoWorkerProgram);
        }
        let Some(_gate) = self.gate.acquire() else {
            return Err(PoolError::Closed);
        };

        let index = self.pick_worker();
        let slot = self.slots[index].clone();
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let frame = JobFrame {
            seq,
            id: job.id,
            stage: job.stage,
            entry: job.entry.to_string(),
            dxil: job.dxil.to_vec(),
            reply: reply_tx,
        };

        slot.in_flight.fetch_add(1, Ordering::Relaxed);
        self.stats.in_flight.fetch_add(1, Ordering::Relaxed);
        let sent = self.senders[index].send(frame);
        if sent.is_err() {
            slot.in_flight.fetch_sub(1, Ordering::Relaxed);
            self.stats.in_flight.fetch_sub(1, Ordering::Relaxed);
            self.stats.errors.fetch_add(1, Ordering::Relaxed);
            return Err(PoolError::Closed);
        }

        let waited = reply_rx.recv_timeout(self.config.job_timeout);
        slot.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.stats.in_flight.fetch_sub(1, Ordering::Relaxed);

        match waited {
            Ok(Ok(metallib)) => {
                if let (Some(cache), Some(key)) = (&self.cache, &cache_key) {
                    cache.put(key, job, &metallib, &self.config.converter_version);
                }
                self.stats.completed.fetch_add(1, Ordering::Relaxed);
                Ok(metallib)
            }
            Ok(Err(err)) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                Err(err)
            }
            Err(RecvTimeoutError::Timeout) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                slot.cancel(seq);
                Err(PoolError::Timeout {
                    job_id: job.id,
                    timeout_ms: self.config.job_timeout.as_millis() as u64,
                })
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                Err(PoolError::Closed)
            }
        }
    }

    fn validate(&self, job: &CompileJob<'_>) -> Result<(), PoolError> {
        if job.entry.is_empty() {
            return Err(PoolError::InvalidJob("entry point is empty"));
        }
        if job.entry.len() > MAX_JOB_ENTRY_BYTES {
            return Err(PoolError::InvalidJob("entry point exceeds 256 bytes"));
        }
        if job.dxil.is_empty() {
            return Err(PoolError::InvalidJob("DXIL input is empty"));
        }
        if job.dxil.len() > MAX_JOB_DXIL_BYTES {
            return Err(PoolError::InvalidJob("DXIL input exceeds 16 MiB"));
        }
        Ok(())
    }

    fn pick_worker(&self) -> usize {
        let count = self.slots.len();
        let start = self.next_worker.fetch_add(1, Ordering::Relaxed) % count;
        let mut best = start;
        let mut best_load = usize::MAX;
        for offset in 0..count {
            let index = (start + offset) % count;
            let load = self.slots[index].in_flight.load(Ordering::Relaxed);
            if load < best_load {
                best = index;
                best_load = load;
            }
        }
        best
    }
}

impl Drop for CompilerPool {
    fn drop(&mut self) {
        // Closing the queues ends every dispatcher loop.
        self.senders.clear();
        // Kill any live child so a blocked read cannot outlive the pool.
        for slot in &self.slots {
            slot.reap();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[derive(Default)]
struct WorkerSlot {
    in_flight: AtomicUsize,
    jobs_served: AtomicU64,
    first_job_micros: AtomicU64,
    subsequent_jobs: AtomicU64,
    subsequent_job_micros_total: AtomicU64,
    child: Mutex<Option<Child>>,
    poisoned: Mutex<HashSet<u64>>,
}

impl WorkerSlot {
    /// Records one completed worker job. Only the slot's dispatcher thread
    /// calls this, so the first/rest split needs no compare-and-swap.
    fn record_job(&self, micros: u64) {
        if self.jobs_served.fetch_add(1, Ordering::Relaxed) == 0 {
            self.first_job_micros.store(micros, Ordering::Relaxed);
        } else {
            self.subsequent_jobs.fetch_add(1, Ordering::Relaxed);
            self.subsequent_job_micros_total
                .fetch_add(micros, Ordering::Relaxed);
        }
    }

    /// Marks `seq` as timed out and kills the worker so a blocked read
    /// returns and the dispatcher can respawn it.
    fn cancel(&self, seq: u64) {
        if let Ok(mut poisoned) = self.poisoned.lock() {
            poisoned.insert(seq);
        }
        self.reap();
    }

    /// Kills and reaps the worker, including descendants that inherited the
    /// worker's stdout pipe.
    fn reap(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                kill_worker(&mut child);
            }
        }
    }
}

/// Kills a worker and its process group, then reaps it.
///
/// Workers are started with `process_group(0)`, so a wrapper script's child
/// processes cannot keep the stdout pipe open after the direct child dies;
/// without the group kill a hung grandchild would make the pool's read block
/// past the job timeout.
fn kill_worker(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(child.id()) {
            // SAFETY: `pid` is the live child's process id, and the child was
            // spawned into its own process group (`process_group(0)`), so the
            // negative pid names that group and SIGKILL has no Rust-side
            // preconditions. A failure (already-exited group) is ignored and
            // followed by the direct `kill`/`wait` below.
            unsafe {
                kill(-pid, SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

struct ChildIo {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ChildIo {
    fn run(&mut self, job: &JobFrame) -> Result<Vec<u8>, PoolError> {
        let header = format!(
            "{PROTOCOL_MAGIC_JOB} {} {} {} {} {}\n",
            job.seq,
            job.id,
            job.stage,
            job.entry.len(),
            job.dxil.len()
        );
        self.stdin
            .write_all(header.as_bytes())
            .and_then(|()| self.stdin.write_all(job.entry.as_bytes()))
            .and_then(|()| self.stdin.write_all(&job.dxil))
            .and_then(|()| self.stdin.flush())
            .map_err(|source| PoolError::WorkerWrite {
                job_id: job.id,
                source,
            })?;

        let mut line = Vec::new();
        let read =
            self.stdout
                .read_until(b'\n', &mut line)
                .map_err(|source| PoolError::WorkerRead {
                    job_id: job.id,
                    source,
                })?;
        if read == 0 {
            return Err(PoolError::WorkerExited { job_id: job.id });
        }
        if line.len() > MAX_FRAME_HEADER_BYTES {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: "response header exceeds 4096 bytes".to_string(),
            });
        }

        let text = std::str::from_utf8(&line).map_err(|_| PoolError::Protocol {
            job_id: job.id,
            detail: "response header is not UTF-8".to_string(),
        })?;
        let mut fields = text.split_ascii_whitespace();
        let magic = fields.next();
        let seq = fields.next();
        let id = fields.next();
        let status = fields.next();
        let len = fields.next();
        if magic != Some(PROTOCOL_MAGIC_RESULT) || fields.next().is_some() {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: format!("expected `{PROTOCOL_MAGIC_RESULT} <seq> <id> <status> <len>`"),
            });
        }
        let got_seq: u64 =
            seq.and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response sequence is not a decimal u64".to_string(),
                })?;
        if got_seq != job.seq {
            return Err(PoolError::SequenceMismatch {
                expected: job.seq,
                got: got_seq,
            });
        }
        let got_id: u64 =
            id.and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response id is not a decimal u64".to_string(),
                })?;
        if got_id != job.id {
            return Err(PoolError::IdMismatch {
                expected: job.id,
                got: got_id,
            });
        }
        let status: u32 =
            status
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response status is not a decimal u32".to_string(),
                })?;
        let len: usize =
            len.and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response length is not a decimal usize".to_string(),
                })?;
        if len > MAX_RESULT_BYTES {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: format!("response payload of {len} bytes exceeds 64 MiB"),
            });
        }

        let mut payload = vec![0u8; len];
        self.stdout
            .read_exact(&mut payload)
            .map_err(|source| PoolError::WorkerRead {
                job_id: job.id,
                source,
            })?;
        if status == 0 {
            Ok(payload)
        } else {
            Err(PoolError::WorkerFailed {
                job_id: job.id,
                status,
                message: String::from_utf8_lossy(&payload).into_owned(),
            })
        }
    }
}

fn spawn_child(program: &Path, slot: &WorkerSlot) -> Result<ChildIo, PoolError> {
    let mut command = Command::new(program);
    if pool_stdio_enabled() {
        command.arg(STDIO_ARG);
    }
    if let Some(converter) = converter_from_env() {
        command.arg(CONVERTER_ARG).arg(converter);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command.spawn().map_err(|source| PoolError::Spawn {
        program: program.to_path_buf(),
        source,
    })?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let (Some(stdin), Some(stdout)) = (stdin, stdout) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(PoolError::Spawn {
            program: program.to_path_buf(),
            source: std::io::Error::other("worker stdio pipes were not created"),
        });
    };
    if let Ok(mut guard) = slot.child.lock() {
        *guard = Some(child);
    }
    Ok(ChildIo {
        stdin,
        stdout: BufReader::new(stdout),
    })
}

fn worker_loop(
    program: PathBuf,
    slot: Arc<WorkerSlot>,
    jobs: Receiver<JobFrame>,
    stats: Arc<AtomicStats>,
) {
    let mut io: Option<ChildIo> = None;
    while let Ok(job) = jobs.recv() {
        let cancelled = slot
            .poisoned
            .lock()
            .map(|mut poisoned| poisoned.remove(&job.seq))
            .unwrap_or(false);
        if cancelled {
            // The caller already returned a timeout; dropping the reply closes
            // its channel and no work is wasted on this job.
            slot.reap();
            io = None;
            continue;
        }
        // The first job of a slot includes process start and Rosetta/dylib
        // warm-up; later jobs reuse the same `--stdio` child.
        let started = Instant::now();
        if io.is_none() {
            match spawn_child(&program, &slot) {
                Ok(child) => {
                    stats.worker_starts.fetch_add(1, Ordering::Relaxed);
                    io = Some(child);
                }
                Err(err) => {
                    let _ = job.reply.send(Err(err));
                }
            }
        }
        let Some(child) = io.as_mut() else {
            continue;
        };
        let outcome = child.run(&job);
        let micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        slot.record_job(micros);
        stats.worker_jobs.fetch_add(1, Ordering::Relaxed);
        stats
            .worker_job_micros_total
            .fetch_add(micros, Ordering::Relaxed);
        stats
            .worker_job_micros_max
            .fetch_max(micros, Ordering::Relaxed);
        // A late cancellation may have raced with the reply; drop it so the
        // caller's id/sequence can be reused by a later job.
        if let Ok(mut poisoned) = slot.poisoned.lock() {
            poisoned.remove(&job.seq);
        }
        match outcome {
            Ok(metallib) => {
                let _ = job.reply.send(Ok(metallib));
            }
            Err(err) => {
                // Any IO or protocol failure leaves the pipeline unusable:
                // reap the child and respawn on the next job.
                io = None;
                slot.reap();
                stats.worker_restarts.fetch_add(1, Ordering::Relaxed);
                let _ = job.reply.send(Err(err));
            }
        }
    }
    slot.reap();
}

struct Gate {
    state: Mutex<usize>,
    available: Condvar,
    capacity: usize,
}

impl Gate {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(0),
            available: Condvar::new(),
            capacity,
        }
    }

    fn acquire(&self) -> Option<GateGuard<'_>> {
        let mut pending = self.state.lock().ok()?;
        while *pending >= self.capacity {
            pending = self.available.wait(pending).ok()?;
        }
        *pending += 1;
        Some(GateGuard { gate: self })
    }
}

struct GateGuard<'a> {
    gate: &'a Gate,
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.gate.state.lock() {
            *pending = pending.saturating_sub(1);
        }
        self.gate.available.notify_one();
    }
}

#[cfg(target_os = "macos")]
mod cache_bridge {
    use std::path::Path;

    use super::CompileJob;
    use crate::macos::cache::{
        Cached, DiskCache, ReflectionPayload, TranslationKey, TranslationOptions,
    };

    /// Content-hashed disk cache handle.
    pub(super) struct Slot(DiskCache);

    /// Cache key for one job.
    pub(super) struct Key(TranslationKey);

    impl Slot {
        pub(super) fn open(dir: &Path) -> Option<Self> {
            match DiskCache::open(dir) {
                Ok(cache) => Some(Self(cache)),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        dir = %dir.display(),
                        "shader pool disk cache disabled: cannot create directory"
                    );
                    None
                }
            }
        }

        pub(super) fn dir(&self) -> &Path {
            self.0.dir()
        }

        /// Key over the job's DXIL, stage, entry point and pool version. The
        /// absent root layout uses the cache module's fixed domain value.
        pub(super) fn key(job: &CompileJob<'_>, version: &str) -> Key {
            Key(TranslationKey::new(
                job.dxil,
                None,
                job.entry,
                job.stage,
                version,
                &TranslationOptions::default(),
            ))
        }

        pub(super) fn get(&self, key: &Key) -> Option<Vec<u8>> {
            self.0.get(&key.0).map(|cached| cached.metallib)
        }

        pub(super) fn put(&self, key: &Key, job: &CompileJob<'_>, metallib: &[u8], version: &str) {
            let reflection = ReflectionPayload::from_translation(
                metallib,
                job.stage,
                job.entry,
                job.dxil,
                None,
                version,
                &TranslationOptions::default(),
                &[],
            );
            let cached = Cached {
                metallib: metallib.to_vec(),
                reflection,
            };
            if let Err(err) = self.0.put(&key.0, &cached) {
                tracing::warn!(
                    error = %err,
                    entry = job.entry,
                    "shader pool could not store a cache entry"
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod cache_bridge {
    use std::path::Path;

    use super::CompileJob;

    /// Non-macOS builds have no DXIL disk cache; the pool stays worker-only.
    pub(super) struct Slot;

    /// Placeholder key for the disabled cache.
    pub(super) struct Key;

    impl Slot {
        pub(super) fn open(_dir: &Path) -> Option<Self> {
            None
        }

        pub(super) fn dir(&self) -> &Path {
            Path::new("")
        }

        pub(super) fn key(_job: &CompileJob<'_>, _version: &str) -> Key {
            Key
        }

        pub(super) fn get(&self, _key: &Key) -> Option<Vec<u8>> {
            None
        }

        pub(super) fn put(
            &self,
            _key: &Key,
            _job: &CompileJob<'_>,
            _metallib: &[u8],
            _version: &str,
        ) {
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_are_driven_in_persistent_stdio_mode() {
        assert!(pool_stdio_enabled());
    }

    #[test]
    fn fresh_pool_reports_zero_worker_jobs_and_no_timings() {
        // A program is configured so the slots exist; no job is submitted, so
        // no child is ever spawned.
        let pool = CompilerPool::new(PoolConfig::default(), Some(PathBuf::from("/bin/false")));
        assert_eq!(pool.worker_job_counts().len(), pool.workers());
        assert!(pool.worker_job_counts().iter().all(|&jobs| jobs == 0));
        assert!(
            pool.worker_timings()
                .iter()
                .all(|timing| *timing == WorkerTiming::default())
        );
        let stats = pool.stats();
        assert_eq!(stats.worker_jobs, 0);
        assert_eq!(stats.worker_job_micros_total, 0);
        assert_eq!(stats.worker_job_micros_max, 0);
    }

    #[test]
    fn worker_env_is_clamped_and_invalid_values_default() {
        assert_eq!(clamp_workers(Some("99")), 8);
        assert_eq!(clamp_workers(Some("0")), 1);
        assert_eq!(clamp_workers(Some("8")), 8);
        assert_eq!(clamp_workers(Some("3")), 3);
        assert_eq!(clamp_workers(Some("not-a-number")), DEFAULT_WORKERS);
        assert_eq!(clamp_workers(Some("")), DEFAULT_WORKERS);
        assert_eq!(clamp_workers(None), DEFAULT_WORKERS);
    }

    #[test]
    fn queue_depth_and_timeout_are_bounded() {
        assert_eq!(parse_bounded(Some("0"), DEFAULT_QUEUE_DEPTH, 1, 4096), 1);
        assert_eq!(
            parse_bounded(Some("999999"), DEFAULT_QUEUE_DEPTH, 1, 4096),
            4096
        );
        assert_eq!(
            parse_bounded(Some("nope"), DEFAULT_QUEUE_DEPTH, 1, 4096),
            DEFAULT_QUEUE_DEPTH
        );
    }

    #[test]
    fn empty_pool_without_workers_or_cache_is_an_explicit_error() {
        let pool = CompilerPool::new(
            PoolConfig {
                workers: 2,
                cache_dir: None,
                ..PoolConfig::default()
            },
            None,
        );
        let err = pool
            .compile(&CompileJob {
                id: 1,
                stage: 8,
                entry: "main",
                dxil: b"dxil",
            })
            .expect_err("no worker and no cache must fail");
        assert!(matches!(err, PoolError::NoWorkerProgram));
        assert_eq!(pool.stats().errors, 1);
        assert_eq!(pool.stats().jobs, 1);
    }

    #[test]
    fn job_validation_rejects_empty_and_oversized_inputs() {
        let pool = CompilerPool::new(PoolConfig::default(), None);
        let oversized = vec![0u8; MAX_JOB_DXIL_BYTES + 1];
        for (entry, dxil) in [
            ("", &b"dxil"[..]),
            ("main", &b""[..]),
            ("main", &oversized[..]),
        ] {
            let err = pool
                .compile(&CompileJob {
                    id: 7,
                    stage: 8,
                    entry,
                    dxil,
                })
                .expect_err("invalid job must be rejected");
            assert!(matches!(err, PoolError::InvalidJob(_)));
        }
        assert_eq!(pool.stats().jobs, 0, "rejected jobs are not counted");
    }
}
