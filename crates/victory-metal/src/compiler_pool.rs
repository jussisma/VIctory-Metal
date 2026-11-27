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
//! # Worker protocol
//!
//! Each worker is a child process started as
//! `victory-irconverter --stdio --converter <libmetalirconverter.dylib>` (see
//! [`pool_stdio_enabled`]) and driven over stdin/stdout with length-prefixed
//! frames. The converter is resolved once, when the pool is built, from
//! `VICTORY_METAL_IRCONVERTER` and then `D3DMETAL_RES`; every child is started
//! with that explicit path, because `--stdio` without `--converter` is a usage
//! error and the child would exit before reading a job. A pool built by
//! [`CompilerPool::from_env`] (the host) refuses a job with a named
//! "no converter library" error instead of spawning such a worker; a worker
//! that really dies is reported as [`PoolError::WorkerExited`] with its bounded
//! stderr captured for the diagnostic, never silently replaced by the one-shot
//! path. The protocol version is implicit in the header tokens.
//!
//! Request, pool -> worker stdin:
//!
//! ```text
//! "VMJOB" SP <seq> SP <job-id> SP <stage> SP <entry-len> SP <dxil-len> LF
//! <entry: entry-len raw bytes><dxil: dxil-len raw bytes>
//! ```
//!
//! Response, worker -> pool stdout. Version 2 carries the converter reflection
//! JSON next to the metallib; version 1 (the metallib only) is still accepted
//! by [`CompilerPool::compile`] so an older worker keeps working:
//!
//! ```text
//! "VMRES"  SP <seq> SP <job-id> SP <status> SP <payload-len> LF
//! <payload: payload-len raw bytes>
//!
//! "VMRES2" SP <seq> SP <job-id> SP <status> SP <metallib-len> SP <reflection-len> LF
//! <metallib: metallib-len raw bytes><reflection: reflection-len raw bytes>
//! ```
//!
//! `<seq>` and `<job-id>` are decimal `u64`, `<stage>` is a decimal `u32`,
//! and the length fields are decimal byte counts. `<seq>` is the pool's
//! internal monotonic sequence number and is what the pool uses for routing
//! and cancellation; `<job-id>` is the caller's id. `status == 0` means the
//! payload is the serialized metallib (and, for `VMRES2`, its reflection JSON);
//! any other status means the payload is a UTF-8 diagnostic. The worker must
//! echo both numbers: the pool rejects a mismatched sequence or caller id as
//! [`PoolError::SequenceMismatch`] or [`PoolError::IdMismatch`] instead of
//! accepting another caller's bytes.
//!
//! The pool asks for the version 2 answer by setting
//! `VICTORY_IRCONVERTER_STDIO_REFLECTION=1` on each child. A worker that
//! predates it ignores the variable and answers `VMRES`; the request frame is
//! unchanged, so such a worker keeps serving [`CompilerPool::compile`].
//!
//! [`CompilerPool::compile_with_reflection`] needs the reflection, so a
//! version 1 answer is an explicit [`PoolError::Protocol`] for that call
//! rather than a silently empty ABI. It is never upgraded by guessing: the
//! pool reaps that answer path and the next job gets a fresh worker, which may
//! be a newer binary.
//!
//! Requests and responses are strictly alternating on one pipe pair, so the
//! pool never needs a worker-side scheduler. A worker that exits, closes its
//! stdout or violates the frame layout is reported as an explicit error, its
//! process is reaped with its exit code or signal and stderr tail, and the slot
//! is respawned for the next job. A `--stdio` supervisor that answers a
//! complete nonzero frame stays alive and keeps serving: a failed conversion is
//! not a restart. A caller that exceeds `job_timeout` cancels its job, kills
//! the worker and receives [`PoolError::Timeout`], and the pool remains usable.
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
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
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

/// Largest accepted reflection JSON from a worker.
///
/// Matches the cache module's 4 MiB reflection bound; the literal keeps the
/// pool compiling for the Win64 guest, where the `macos` module is not built.
/// A larger document is refused instead of being truncated into an unusable
/// ABI.
pub const MAX_REFLECTION_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

/// Version of the reflection channel a `--stdio` worker produces.
///
/// Mixed into [`POOL_CONVERTER_VERSION`] so a metallib cached before the pool
/// asked for reflection is a different key and can never be served to
/// [`CompilerPool::compile_with_reflection`] as if it carried a layout.
pub const POOL_REFLECTION_PRODUCER_VERSION: &str = "reflection-json-1";

/// Largest worker stderr text kept for a diagnostic.
///
/// A worker's usage text or converter error fits comfortably; the bound keeps
/// one host log line and one error message finite. The tail is kept (the
/// reason a worker died is the end of its stderr), not the first bytes.
pub const MAX_WORKER_STDERR_BYTES: usize = 4096;

/// Raw bytes the stderr reader retains before escaping.
///
/// The reported text is still capped at [`MAX_WORKER_STDERR_BYTES`]; this
/// larger window keeps a crash line that follows a long startup banner, and
/// [`escape_worker_stderr`] keeps the *tail* of it, because a worker's last
/// words are the reason it died.
const MAX_WORKER_STDERR_CAPTURE_BYTES: usize = MAX_WORKER_STDERR_BYTES * 4;

/// Longest worker stderr text that may enter telemetry.
///
/// Longer text stays behind [`CompilerPool::last_worker_stderr`] and the host
/// log; [`CompilerPool::worker_stderr_for_telemetry`] is `None` for it so a
/// telemetry document never carries an unbounded or path-bearing line.
pub const MAX_WORKER_STDERR_TELEMETRY_CHARS: usize = 200;

/// Stable identity of the refusal returned when no converter library resolves.
///
/// It is carried as the source of [`PoolError::Spawn`] so the pool's variant
/// set, and therefore every existing consumer of [`PoolError`], stays
/// unchanged. A pool in this state never spawns a worker: `--stdio` without
/// `--converter` is a usage error, and that child would exit before reading a
/// single job.
pub const NO_CONVERTER_LIBRARY: &str =
    "no converter library (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)";

/// Placeholder that replaces an absolute path in reported worker stderr.
const REDACTED_PATH: &str = "<path>";

/// Version mixed into the pool's cache key.
pub const POOL_CONVERTER_VERSION: &str = concat!(
    "victory-metal/",
    env!("CARGO_PKG_VERSION"),
    "/shader-pool-2/reflection-json-1"
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
const PROTOCOL_MAGIC_RESULT_V2: &str = "VMRES2";
const MAX_FRAME_HEADER_BYTES: usize = 4096;
/// Argument that selects the worker's persistent framed protocol.
const STDIO_ARG: &str = "--stdio";
/// Argument that names the x86_64 IRConverter library to the worker.
const CONVERTER_ARG: &str = "--converter";
/// Environment switch that makes a worker answer `VMRES2` (metallib plus
/// reflection) instead of the original metallib-only `VMRES`.
///
/// It is an environment variable rather than a request header or CLI flag so
/// an unchanged worker binary keeps working (it ignores the variable and
/// answers `VMRES`, which [`CompilerPool::compile`] still accepts), while the
/// request frame itself stays byte-for-byte identical.
const REFLECTION_FRAMES_ENV: &str = "VICTORY_IRCONVERTER_STDIO_REFLECTION";

/// Whether this build starts workers in persistent `--stdio` mode.
///
/// It is always true: the pool no longer has a one-shot fallback, so a worker
/// that does not understand `--stdio` fails explicitly instead of being driven
/// once per job. Tests use this to assert the transport.
pub const fn pool_stdio_enabled() -> bool {
    true
}

/// Converter library passed to every `--stdio` worker.
///
/// Resolved the same way the launcher does, in this order: the
/// `VICTORY_METAL_IRCONVERTER` file itself, then
/// `D3DMETAL_RES/libmetalirconverter.dylib`. The pool resolves it once, at
/// construction, so every child of this pool is started with the same
/// explicit `--converter` and a later environment change cannot split the
/// worker set. `None` means nothing resolves; a pool that requires a
/// converter refuses the job instead of spawning a worker that can only exit
/// on a usage error.
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

/// One translated job with the converter's reflection JSON.
///
/// The metallib is what the host opens; the reflection is the
/// `IRShaderReflectionCopyJSONString` document the guest turns into the
/// per-stage `PipelineAbi` (`TopLevelArgumentBuffer`, `UsedResources`, stage
/// and entry point). It is never fabricated from the metallib.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledShader {
    /// Serialized metallib bytes.
    pub metallib: Vec<u8>,
    /// Converter reflection JSON; non-empty for a successful translation.
    pub reflection: Vec<u8>,
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
    /// Workers reaped after a crash, timeout or protocol desync.
    ///
    /// A conversion the supervisor reports as failed (`VMRES2` with a nonzero
    /// status) is not a restart: the supervisor stayed in step and serves the
    /// next job.
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

/// How a worker process ended, as observed by reaping it.
///
/// A `--stdio` supervisor that survives a failed conversion answers a complete
/// nonzero frame and is *not* reaped; this type describes the cases where the
/// pool lost the process (crash, killed on timeout, protocol desync). On Unix
/// a signal is distinguished from an exit code so the diagnostic can say
/// `signal 11 (SIGSEGV)` rather than inventing a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerExit {
    /// Normal exit with this code.
    Code(i32),
    /// Killed by this Unix signal number.
    Signal(i32),
    /// The pool could not collect a status (the child was already reaped).
    Unknown,
}

impl WorkerExit {
    fn from_status(status: ExitStatus) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if let Some(signal) = status.signal() {
                return Self::Signal(signal);
            }
        }
        match status.code() {
            Some(code) => Self::Code(code),
            None => Self::Unknown,
        }
    }
}

impl std::fmt::Display for WorkerExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Code(code) => write!(f, "exit code {code}"),
            Self::Signal(signal) => match signal_name(*signal) {
                Some(name) => write!(f, "signal {signal} ({name})"),
                None => write!(f, "signal {signal}"),
            },
            Self::Unknown => f.write_str("unknown exit status"),
        }
    }
}

/// Names the signals a converter process most often dies from.
fn signal_name(signal: i32) -> Option<&'static str> {
    match signal {
        1 => Some("SIGHUP"),
        2 => Some("SIGINT"),
        3 => Some("SIGQUIT"),
        4 => Some("SIGILL"),
        5 => Some("SIGTRAP"),
        6 => Some("SIGABRT"),
        8 => Some("SIGFPE"),
        9 => Some("SIGKILL"),
        10 => Some("SIGBUS"),
        11 => Some("SIGSEGV"),
        12 => Some("SIGSYS"),
        13 => Some("SIGPIPE"),
        14 => Some("SIGALRM"),
        15 => Some("SIGTERM"),
        _ => None,
    }
}

/// Everything the pool knows about a worker it lost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerDeath {
    /// Exit code, signal or unknown status.
    pub exit: WorkerExit,
    /// Bounded, escaped stderr tail; empty when nothing was captured.
    pub stderr: String,
}

impl std::fmt::Display for WorkerDeath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.exit)?;
        if !self.stderr.is_empty() {
            write!(f, "; worker stderr: {}", self.stderr)?;
        }
        Ok(())
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
    /// The worker closed stdout before answering, including a crash.
    #[error("shader worker exited before answering job {job_id}: {death}")]
    WorkerExited {
        /// Job that lost its worker.
        job_id: u64,
        /// Exit code or signal plus the captured stderr tail.
        death: WorkerDeath,
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

impl PoolError {
    /// Whether this error is the pool's "no converter library" refusal.
    ///
    /// The refusal is a [`PoolError::Spawn`] whose source is
    /// [`NO_CONVERTER_LIBRARY`], so the error message names the missing input
    /// while the variant set every existing consumer matches stays unchanged.
    pub fn is_no_converter_library(&self) -> bool {
        matches!(
            self,
            PoolError::Spawn { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound
                    && source.to_string().contains(NO_CONVERTER_LIBRARY)
        )
    }
}

/// One answer from a worker: the metallib and, for protocol version 2, the
/// reflection JSON.
struct WorkerReply {
    metallib: Vec<u8>,
    reflection: Option<Vec<u8>>,
}

struct JobFrame {
    seq: u64,
    id: u64,
    stage: u32,
    entry: String,
    dxil: Vec<u8>,
    reply: SyncSender<Result<WorkerReply, PoolError>>,
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
    /// Converter library every child of this pool is started with, resolved
    /// once from the environment.
    converter: Option<PathBuf>,
    cache: Option<cache_bridge::Slot>,
    slots: Vec<Arc<WorkerSlot>>,
    senders: Vec<SyncSender<JobFrame>>,
    handles: Vec<JoinHandle<()>>,
    gate: Gate,
    stats: Arc<AtomicStats>,
    /// Bounded stderr of the last worker that failed a job.
    last_stderr: Arc<Mutex<Option<String>>>,
    next_worker: AtomicUsize,
    next_seq: AtomicU64,
}

impl CompilerPool {
    /// Builds a pool. `worker_program` is the child executable that speaks the
    /// documented frame protocol (normally the `victory-irconverter` binary);
    /// `None` leaves the pool cache-only.
    pub fn new(config: PoolConfig, worker_program: Option<PathBuf>) -> Self {
        Self::build(config, worker_program, converter_from_env(), false)
    }

    /// Builds the pool, resolving the converter once and passing it on.
    fn build(
        config: PoolConfig,
        worker_program: Option<PathBuf>,
        converter: Option<PathBuf>,
        require_converter: bool,
    ) -> Self {
        let workers = config.workers.clamp(1, MAX_WORKERS);
        let queue_depth = config.queue_depth.max(1);
        let stats = Arc::new(AtomicStats::default());
        let last_stderr = Arc::new(Mutex::new(None));
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
                    let thread_converter = converter.clone();
                    let thread_stderr = Arc::clone(&last_stderr);
                    let name = format!("victory-shader-pool-{index}");
                    match thread::Builder::new().name(name).spawn(move || {
                        worker_loop(
                            program,
                            thread_converter,
                            require_converter,
                            thread_slot,
                            rx,
                            thread_stats,
                            thread_stderr,
                        );
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
            converter,
            cache,
            slots,
            senders,
            handles,
            gate: Gate::new(queue_depth),
            stats,
            last_stderr,
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

    /// Converter library every child of this pool is started with, when one
    /// resolved at construction.
    pub fn converter(&self) -> Option<&Path> {
        self.converter.as_deref()
    }

    /// The bounded, escaped stderr of the last worker that failed a job.
    ///
    /// The text is at most [`MAX_WORKER_STDERR_BYTES`], has control characters
    /// replaced, and has absolute paths redacted to `<path>`: it is safe to put
    /// on one diagnostic line. A worker that exited before answering carries
    /// the same tail in [`PoolError::WorkerExited`]; a `--stdio` supervisor
    /// that answered a nonzero frame contributes its bounded reason (which
    /// already includes the conversion child's exit status and stderr tail).
    /// This accessor keeps the last one for telemetry and the host log.
    pub fn last_worker_stderr(&self) -> Option<String> {
        self.last_stderr.lock().ok().and_then(|text| text.clone())
    }

    /// [`CompilerPool::last_worker_stderr`] only when it is short enough for
    /// telemetry (at most [`MAX_WORKER_STDERR_TELEMETRY_CHARS`] characters).
    pub fn worker_stderr_for_telemetry(&self) -> Option<String> {
        telemetry_stderr(self.last_worker_stderr().as_deref())
    }

    /// Builds a pool from [`PoolConfig::from_env`], resolving the worker from
    /// `VICTORY_IRCONVERTER_WORKER` or the `victory-irconverter` sibling of
    /// the current executable. A pool with neither is cache-only.
    ///
    /// This is the host's pool: it requires the converter library, because
    /// every `--stdio` job needs one and a child spawned without it exits on a
    /// usage error before reading the job.
    pub fn from_env() -> Self {
        let program = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                let exe = std::env::current_exe().ok()?;
                let sibling = exe.parent()?.join("victory-irconverter");
                sibling.is_file().then_some(sibling)
            });
        Self::build(PoolConfig::from_env(), program, converter_from_env(), true)
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
    ///
    /// This is the metallib-only API. It still accepts a protocol version 1
    /// worker (`VMRES`), so a caller that needs no reflection keeps working
    /// against an older binary.
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

        let reply = match self.dispatch(job) {
            Ok(reply) => reply,
            Err(err) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        };
        if let (Some(cache), Some(key)) = (&self.cache, &cache_key) {
            cache.put(key, job, &reply.metallib, &self.config.converter_version);
        }
        self.stats.completed.fetch_add(1, Ordering::Relaxed);
        Ok(reply.metallib)
    }

    /// Compiles one job and returns the converter reflection JSON with it.
    ///
    /// The reflection is required, not optional: a warm cache entry that has
    /// no reflection sidecar is a miss (the worker re-runs and fills it), and a
    /// worker that answers the metallib-only `VMRES` frame is an explicit
    /// [`PoolError::Protocol`] instead of a pipeline registered with an empty
    /// ABI. The cache key mixes [`POOL_REFLECTION_PRODUCER_VERSION`], so a
    /// metallib cached before the pool asked for reflection is a different
    /// entry and cannot be served here.
    pub fn compile_with_reflection(
        &self,
        job: &CompileJob<'_>,
    ) -> Result<CompiledShader, PoolError> {
        self.validate(job)?;
        self.stats.jobs.fetch_add(1, Ordering::Relaxed);

        let cache_key = self
            .cache
            .as_ref()
            .map(|_| cache_bridge::Slot::key(job, &self.config.converter_version));
        if let (Some(cache), Some(key)) = (&self.cache, &cache_key) {
            if let Some((metallib, reflection)) = cache.get_with_reflection(key) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.stats.completed.fetch_add(1, Ordering::Relaxed);
                return Ok(CompiledShader {
                    metallib,
                    reflection,
                });
            }
            // A metallib without a reflection sidecar is stale for this call;
            // count a miss so the warm ratio never claims a served ABI.
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        let reply = match self.dispatch(job) {
            Ok(reply) => reply,
            Err(err) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        };
        let Some(reflection) = reply.reflection else {
            self.stats.errors.fetch_add(1, Ordering::Relaxed);
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: format!(
                    "worker answered a metallib-only VMRES frame; reflection is required for a \
                     pipeline ABI (worker protocol {POOL_REFLECTION_PRODUCER_VERSION} expected)"
                ),
            });
        };
        if reflection.is_empty() {
            self.stats.errors.fetch_add(1, Ordering::Relaxed);
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: "worker answered an empty reflection".to_string(),
            });
        }
        if let (Some(cache), Some(key)) = (&self.cache, &cache_key) {
            cache.put(key, job, &reply.metallib, &self.config.converter_version);
            cache.put_reflection(key, &reflection);
        }
        self.stats.completed.fetch_add(1, Ordering::Relaxed);
        Ok(CompiledShader {
            metallib: reply.metallib,
            reflection,
        })
    }

    /// Sends one validated job to a worker and waits for its reply.
    ///
    /// Cache policy and stats live in the public entry points; this function
    /// only owns dispatch, the queue gate, cancellation and timeout.
    fn dispatch(&self, job: &CompileJob<'_>) -> Result<WorkerReply, PoolError> {
        if self.program.is_none() || self.senders.is_empty() {
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
            return Err(PoolError::Closed);
        }

        let waited = reply_rx.recv_timeout(self.config.job_timeout);
        slot.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.stats.in_flight.fetch_sub(1, Ordering::Relaxed);

        match waited {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => {
                slot.cancel(seq);
                Err(PoolError::Timeout {
                    job_id: job.id,
                    timeout_ms: self.config.job_timeout.as_millis() as u64,
                })
            }
            Err(RecvTimeoutError::Disconnected) => Err(PoolError::Closed),
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
                let _ = kill_worker(&mut child);
            }
        }
    }

    /// Kills and reaps the worker, returning how it ended.
    ///
    /// An already exited child is reaped as-is, so a worker that closed its
    /// stdout and then died reports its real exit code or signal. A child the
    /// pool has to kill (protocol desync, timeout) reports the kill.
    fn reap_status(&self) -> Option<ExitStatus> {
        let mut guard = self.child.lock().ok()?;
        let mut child = guard.take()?;
        kill_worker(&mut child)
    }
}

/// Kills a worker and its process group, then reaps it and reports its status.
///
/// Workers are started with `process_group(0)`, so a wrapper script's child
/// processes cannot keep the stdout pipe open after the direct child dies;
/// without the group kill a hung grandchild would make the pool's read block
/// past the job timeout. A worker that already exited keeps its own status
/// instead of being reported as `SIGKILL`.
fn kill_worker(child: &mut Child) -> Option<ExitStatus> {
    if let Ok(Some(status)) = child.try_wait() {
        return Some(status);
    }
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
    child.wait().ok()
}

/// Reads a worker's stderr into a bounded tail ring and sends the escaped
/// text back when the pipe closes.
///
/// The reader never blocks the worker: it drains the pipe for the whole child
/// lifetime and only keeps the last [`MAX_WORKER_STDERR_CAPTURE_BYTES`] bytes,
/// because the reason a worker died is at the end of its stderr. The receiver
/// carries the escaped tail once, at EOF; a caller that already reaped the
/// child gets it immediately, so the pool reaps before reading when the child
/// closed stdout.
fn spawn_stderr_reader(mut pipe: ChildStderr) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    let spawned = thread::Builder::new()
        .name("victory-shader-stderr".to_string())
        .spawn(move || {
            let mut ring: Vec<u8> = Vec::new();
            let mut truncated = false;
            let mut chunk = [0u8; 1024];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        ring.extend_from_slice(&chunk[..read]);
                        if ring.len() > MAX_WORKER_STDERR_CAPTURE_BYTES {
                            let overflow = ring.len() - MAX_WORKER_STDERR_CAPTURE_BYTES;
                            ring.drain(..overflow);
                            truncated = true;
                        }
                    }
                }
            }
            let _ = tx.send(escape_worker_stderr(&ring, truncated));
        });
    if let Err(err) = spawned {
        // Without a reader the job still fails explicitly; it just has no
        // captured reason. Dropping the closure also disconnects `rx`.
        tracing::warn!(error = %err, "shader pool could not start a worker stderr reader");
    }
    rx
}

/// Escapes a bounded worker-stderr tail into one line of diagnostics.
///
/// Newlines and tabs become spaces, other control characters `?`, runs of
/// spaces collapse, and absolute paths are replaced with [`REDACTED_PATH`], so
/// the text can be recorded without a host path breaking a `key=value` line or
/// leaking into telemetry.
fn escape_worker_stderr(raw: &[u8], truncated: bool) -> String {
    let lossy = String::from_utf8_lossy(raw);
    let mut cleaned = String::with_capacity(lossy.len());
    let mut last_space = false;
    for ch in lossy.chars() {
        let mapped = match ch {
            '\n' | '\r' | '\t' => ' ',
            ch if ch.is_control() => '?',
            ch => ch,
        };
        if mapped == ' ' {
            if !last_space && !cleaned.is_empty() {
                cleaned.push(' ');
            }
            last_space = true;
        } else {
            cleaned.push(mapped);
            last_space = false;
        }
    }
    let escaped = redact_absolute_paths(cleaned.trim_end());
    let suffix = " [truncated]";
    let room = MAX_WORKER_STDERR_BYTES.saturating_sub(suffix.len());
    let mut out = truncate_chars_tail(&escaped, room);
    if truncated || out.len() < escaped.len() {
        out.push_str(suffix);
    }
    out
}

/// Replaces every absolute path token with [`REDACTED_PATH`].
fn redact_absolute_paths(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (index, token) in text.split(' ').enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&redact_token(token));
    }
    out
}

/// Redacts an absolute path at the start of a token or after a `key=` prefix.
fn redact_token(token: &str) -> String {
    let (prefix, value) = match token.split_once('=') {
        Some((prefix, value)) if !prefix.is_empty() => (Some(prefix), value),
        _ => (None, token),
    };
    let core = value.trim_matches(|ch: char| {
        matches!(
            ch,
            '\'' | '"' | '`' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '<' | '>' | '.'
        )
    });
    if is_absolute_path(core) {
        match prefix {
            Some(prefix) => format!("{prefix}={REDACTED_PATH}"),
            None => REDACTED_PATH.to_string(),
        }
    } else {
        token.to_string()
    }
}

/// Whether `token` names an absolute POSIX or Windows path.
fn is_absolute_path(token: &str) -> bool {
    if token.starts_with('/') || token.starts_with("~/") {
        return true;
    }
    let bytes = token.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
}

/// Keeps the last `max` bytes of `text` without splitting a character.
///
/// Stderr is reported as a *tail*: the worker's last line (a panic, a converter
/// abort, a signal handler's note) is the diagnostic, while the first line is
/// often a startup banner. Truncating from the front would discard exactly the
/// bytes that explain the failure.
fn truncate_chars_tail(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len().saturating_sub(max);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

/// [`MAX_WORKER_STDERR_TELEMETRY_CHARS`]-bounded view of captured stderr.
fn telemetry_stderr(text: Option<&str>) -> Option<String> {
    text.filter(|text| {
        !text.is_empty() && text.chars().count() <= MAX_WORKER_STDERR_TELEMETRY_CHARS
    })
    .map(str::to_string)
}

/// Records the most recent worker stderr for [`CompilerPool::last_worker_stderr`].
fn record_worker_stderr(sink: &Mutex<Option<String>>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Ok(mut slot) = sink.lock() {
        *slot = Some(text.to_string());
    }
}

/// Appends captured stderr to a worker-reported failure, when it fits.
///
/// A status-failure payload is already a diagnostic; the stderr tail is added
/// only when the payload is short, so one error message stays bounded.
fn attach_stderr(error: PoolError, stderr: String) -> PoolError {
    match error {
        PoolError::WorkerFailed {
            job_id,
            status,
            message,
        } if message.len() <= MAX_WORKER_STDERR_BYTES => {
            let message = if message.is_empty() {
                stderr
            } else {
                format!("{message}; worker stderr: {stderr}")
            };
            PoolError::WorkerFailed {
                job_id,
                status,
                message,
            }
        }
        other => other,
    }
}

struct ChildIo {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Escaped stderr, delivered once when the worker's pipe closes.
    stderr: Option<Receiver<String>>,
}

impl ChildIo {
    /// Waits briefly for the stderr reader to reach EOF.
    ///
    /// A worker that exited has already closed the pipe, so this returns at
    /// once; the timeout only covers a descendant that escaped the
    /// process-group kill and kept the pipe open.
    fn take_stderr(&mut self) -> Option<String> {
        let reader = self.stderr.take()?;
        match reader.recv_timeout(Duration::from_millis(250)) {
            Ok(text) if !text.is_empty() => Some(text),
            _ => None,
        }
    }

    fn run(&mut self, job: &JobFrame) -> Result<WorkerReply, PoolError> {
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
            // The supervisor is gone or closed its stdout. The worker loop
            // reaps it and fills in the exit code/signal and stderr tail, so
            // the reason is not lost behind a bare "exited" message.
            return Err(PoolError::WorkerExited {
                job_id: job.id,
                death: WorkerDeath {
                    exit: WorkerExit::Unknown,
                    stderr: String::new(),
                },
            });
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
        let with_reflection = match fields.next() {
            Some(PROTOCOL_MAGIC_RESULT) => false,
            Some(PROTOCOL_MAGIC_RESULT_V2) => true,
            _ => {
                return Err(PoolError::Protocol {
                    job_id: job.id,
                    detail: format!(
                        "expected `{PROTOCOL_MAGIC_RESULT} <seq> <id> <status> <len>` or \
                         `{PROTOCOL_MAGIC_RESULT_V2} <seq> <id> <status> <metallib-len> \
                         <reflection-len>`"
                    ),
                });
            }
        };
        let seq = fields.next();
        let id = fields.next();
        let status = fields.next();
        let metallib_len = fields.next();
        let reflection_len = if with_reflection { fields.next() } else { None };
        if fields.next().is_some() {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: "response header has trailing fields".to_string(),
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
        let metallib_len: usize =
            metallib_len
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response metallib length is not a decimal usize".to_string(),
                })?;
        if metallib_len > MAX_RESULT_BYTES {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: format!("response payload of {metallib_len} bytes exceeds 64 MiB"),
            });
        }
        let reflection_len: usize = if with_reflection {
            reflection_len
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| PoolError::Protocol {
                    job_id: job.id,
                    detail: "response reflection length is not a decimal usize".to_string(),
                })?
        } else {
            0
        };
        if reflection_len > MAX_REFLECTION_OUTPUT_BYTES {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: format!(
                    "response reflection of {reflection_len} bytes exceeds \
                     {MAX_REFLECTION_OUTPUT_BYTES}"
                ),
            });
        }
        if status != 0 && reflection_len != 0 {
            return Err(PoolError::Protocol {
                job_id: job.id,
                detail: "failed response carries a reflection payload".to_string(),
            });
        }

        let mut payload = vec![0u8; metallib_len];
        self.stdout
            .read_exact(&mut payload)
            .map_err(|source| PoolError::WorkerRead {
                job_id: job.id,
                source,
            })?;
        let mut reflection = if reflection_len == 0 {
            None
        } else {
            let mut bytes = vec![0u8; reflection_len];
            self.stdout
                .read_exact(&mut bytes)
                .map_err(|source| PoolError::WorkerRead {
                    job_id: job.id,
                    source,
                })?;
            Some(bytes)
        };
        if status == 0 {
            Ok(WorkerReply {
                metallib: payload,
                reflection: reflection.take(),
            })
        } else {
            Err(PoolError::WorkerFailed {
                job_id: job.id,
                status,
                message: String::from_utf8_lossy(&payload).into_owned(),
            })
        }
    }
}

/// Builds the exact command one worker child is started with.
///
/// The pool always asks for the persistent `--stdio` transport and names the
/// converter explicitly: `--stdio` without `--converter` is a usage error, and
/// such a child exits before reading a frame. `--converter` is omitted only for
/// a program the pool was built to drive without one (the protocol fixtures).
fn worker_command(program: &Path, converter: Option<&Path>) -> Command {
    let mut command = Command::new(program);
    if pool_stdio_enabled() {
        command.arg(STDIO_ARG);
    }
    if let Some(converter) = converter {
        command.arg(CONVERTER_ARG).arg(converter);
    }
    // Ask for the reflection frame; a worker that predates it ignores the
    // variable and answers `VMRES`, which `compile` still accepts.
    command.env(REFLECTION_FRAMES_ENV, "1");
    command
}

/// The named refusal for a pool that cannot give its worker a converter.
fn no_converter_library(program: &Path, converter: Option<&Path>) -> PoolError {
    let detail = match converter {
        Some(path) => format!(
            "{NO_CONVERTER_LIBRARY}: {} is not a regular file",
            path.display()
        ),
        None => NO_CONVERTER_LIBRARY.to_string(),
    };
    PoolError::Spawn {
        program: program.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, detail),
    }
}

/// Cheap startup self-check: the worker program must be a runnable file.
///
/// This is a filesystem check only. It sends no probe frame and starts no GPU
/// work, so a broken installation is a named [`PoolError::Spawn`] before the
/// pool pays for a child process.
fn validate_worker_program(program: &Path) -> Result<(), PoolError> {
    let metadata = std::fs::metadata(program).map_err(|source| PoolError::Spawn {
        program: program.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(PoolError::Spawn {
            program: program.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "worker program is not a regular file",
            ),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(PoolError::Spawn {
                program: program.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "worker program is not executable",
                ),
            });
        }
    }
    Ok(())
}

fn spawn_child(
    program: &Path,
    converter: Option<&Path>,
    require_converter: bool,
    slot: &WorkerSlot,
) -> Result<ChildIo, PoolError> {
    validate_worker_program(program)?;
    let converter = match converter {
        Some(path) if path.is_file() => Some(path),
        Some(path) => return Err(no_converter_library(program, Some(path))),
        None if require_converter => return Err(no_converter_library(program, None)),
        None => None,
    };
    let mut command = worker_command(program, converter);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command.spawn().map_err(|source| PoolError::Spawn {
        program: program.to_path_buf(),
        source,
    })?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take().map(spawn_stderr_reader);
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
        stderr,
    })
}

fn worker_loop(
    program: PathBuf,
    converter: Option<PathBuf>,
    require_converter: bool,
    slot: Arc<WorkerSlot>,
    jobs: Receiver<JobFrame>,
    stats: Arc<AtomicStats>,
    last_stderr: Arc<Mutex<Option<String>>>,
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
            match spawn_child(&program, converter.as_deref(), require_converter, &slot) {
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
            Ok(reply) => {
                let _ = job.reply.send(Ok(reply));
            }
            // A complete nonzero answer is a *conversion* failure, not a dead
            // worker: the supervisor stayed in step on stdin and serves the
            // next job. Reaping it here is what made Boot #17 show one process
            // start per job.
            Err(err @ PoolError::WorkerFailed { .. }) => {
                // Carry the supervisor's bounded reason (which includes the
                // conversion child's signal/exit and stderr tail) into the
                // same diagnostic channel as captured worker stderr, so the
                // host's `host_translation_reason` can say why the stage
                // failed instead of only "converter rejected the shader".
                if let PoolError::WorkerFailed { message, .. } = &err {
                    let detail = escape_worker_stderr(message.as_bytes(), false);
                    if !detail.is_empty() {
                        record_worker_stderr(&last_stderr, &detail);
                        tracing::warn!(
                            job_id = job.id,
                            worker_reason = %detail,
                            "shader worker rejected a job with a bounded reason"
                        );
                    }
                }
                let _ = job.reply.send(Err(err));
            }
            Err(err) => {
                // The pipeline is unusable (EOF, IO error, protocol desync).
                // Reap first so the stderr reader sees EOF and the child's real
                // exit code or signal is collected instead of being reported as
                // SIGKILL; then take the bounded tail.
                let exit = slot.reap_status().map(WorkerExit::from_status);
                let stderr = child.take_stderr();
                io = None;
                stats.worker_restarts.fetch_add(1, Ordering::Relaxed);
                if let Some(text) = &stderr {
                    record_worker_stderr(&last_stderr, text);
                    tracing::warn!(
                        job_id = job.id,
                        exit = %exit.unwrap_or(WorkerExit::Unknown),
                        worker_stderr = %text,
                        "shader worker failed a job and wrote to stderr"
                    );
                }
                let err = match err {
                    PoolError::WorkerExited { job_id, .. } => PoolError::WorkerExited {
                        job_id,
                        death: WorkerDeath {
                            exit: exit.unwrap_or(WorkerExit::Unknown),
                            stderr: stderr.unwrap_or_default(),
                        },
                    },
                    other => attach_stderr(other, stderr.unwrap_or_default()),
                };
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
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::CompileJob;
    use crate::macos::cache::{
        Cached, DiskCache, ReflectionPayload, TranslationKey, TranslationOptions,
    };

    /// Suffix of the reflection JSON stored next to the `.vce` entry.
    const REFLECTION_SIDECAR_SUFFIX: &str = ".reflection.json";

    static REFLECTION_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

        /// Reads a warm entry that has both the metallib and its reflection.
        ///
        /// The converter reflection JSON is stored next to the `.vce` entry as
        /// `<digest>.reflection.json` (the cache entry format itself is owned
        /// by [`crate::macos::cache`], whose eval only carries the binary
        /// [`ReflectionPayload`]). A metallib whose sidecar is missing or
        /// unreadable is *not* a hit: the caller must re-run the worker rather
        /// than register a pipeline with no layout.
        pub(super) fn get_with_reflection(&self, key: &Key) -> Option<(Vec<u8>, Vec<u8>)> {
            let cached = self.0.get(&key.0)?;
            let reflection = self.get_reflection(key)?;
            Some((cached.metallib, reflection))
        }

        fn reflection_path(&self, key: &Key) -> PathBuf {
            self.0.dir().join(format!(
                "{}{}",
                key.0.digest_hex(),
                REFLECTION_SIDECAR_SUFFIX
            ))
        }

        fn get_reflection(&self, key: &Key) -> Option<Vec<u8>> {
            let path = self.reflection_path(key);
            let metadata = std::fs::metadata(&path).ok()?;
            if !metadata.is_file() || metadata.len() == 0 {
                return None;
            }
            if metadata.len() > super::MAX_REFLECTION_OUTPUT_BYTES as u64 {
                return None;
            }
            let bytes = std::fs::read(&path).ok()?;
            if bytes.is_empty() || bytes.len() > super::MAX_REFLECTION_OUTPUT_BYTES {
                return None;
            }
            Some(bytes)
        }

        /// Stores the reflection JSON through a same-directory temporary file
        /// and an atomic rename, so a concurrent reader never sees half a
        /// document.
        pub(super) fn put_reflection(&self, key: &Key, reflection: &[u8]) {
            if reflection.is_empty() || reflection.len() > super::MAX_REFLECTION_OUTPUT_BYTES {
                tracing::warn!(
                    entry = key.0.digest_hex(),
                    bytes = reflection.len(),
                    "shader pool did not store an out-of-bounds reflection"
                );
                return;
            }
            let path = self.reflection_path(key);
            let sequence = REFLECTION_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary = self.0.dir().join(format!(
                ".{}.{}.{}.reflection.tmp",
                key.0.digest_hex(),
                std::process::id(),
                sequence
            ));
            let written = std::fs::write(&temporary, reflection)
                .and_then(|()| std::fs::rename(&temporary, &path));
            if let Err(err) = written {
                let _ = std::fs::remove_file(&temporary);
                tracing::warn!(
                    error = %err,
                    entry = key.0.digest_hex(),
                    "shader pool could not store the converter reflection"
                );
            }
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

        pub(super) fn get_with_reflection(&self, _key: &Key) -> Option<(Vec<u8>, Vec<u8>)> {
            None
        }

        pub(super) fn put_reflection(&self, _key: &Key, _reflection: &[u8]) {}

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

    /// The exact command a worker child is started with.
    #[test]
    fn worker_command_names_the_stdio_transport_and_the_converter() {
        let command = worker_command(
            Path::new("/opt/victory/victory-irconverter"),
            Some(Path::new("/share/libmetalirconverter.dylib")),
        );
        assert_eq!(
            command.get_program(),
            std::ffi::OsStr::new("/opt/victory/victory-irconverter")
        );
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            ["--stdio", "--converter", "/share/libmetalirconverter.dylib"],
            "every job needs an explicit converter; --stdio alone is a usage error"
        );
        assert_eq!(
            command
                .get_envs()
                .find(|(key, _)| *key == REFLECTION_FRAMES_ENV),
            Some((
                std::ffi::OsStr::new(REFLECTION_FRAMES_ENV),
                Some(std::ffi::OsStr::new("1"))
            ))
        );

        let command = worker_command(Path::new("/opt/victory/pool-fixture.sh"), None);
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["--stdio"]);
    }

    /// A pool that must convert never spawns a worker without a converter.
    #[test]
    fn a_pool_without_a_converter_refuses_the_job_without_spawning() {
        let program = std::env::current_exe().expect("the test executable exists");
        let pool = CompilerPool::build(
            PoolConfig {
                workers: 1,
                queue_depth: 1,
                cache_dir: None,
                ..PoolConfig::default()
            },
            Some(program),
            None,
            true,
        );
        let err = pool
            .compile(&CompileJob {
                id: 9,
                stage: 1,
                entry: "main",
                dxil: b"dxil",
            })
            .expect_err("a pool with no converter library must refuse");
        assert!(err.is_no_converter_library(), "unexpected error: {err:?}");
        assert!(
            err.to_string().contains(NO_CONVERTER_LIBRARY),
            "the refusal must name the missing library: {err}"
        );
        let stats = pool.stats();
        assert_eq!(stats.worker_starts, 0, "no worker may be started");
        assert_eq!(stats.worker_jobs, 0, "no job may reach a worker");
        assert_eq!(stats.errors, 1);
        assert!(pool.last_worker_stderr().is_none());
    }

    /// A worker that exits before answering leaves its own reason behind.
    #[cfg(unix)]
    #[test]
    fn a_worker_that_exits_before_answering_reports_its_stderr() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "victory-pool-stderr-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        let program = dir.join("immediate-exit.sh");
        std::fs::write(
            &program,
            "#!/bin/sh\n\
             printf 'victory-irconverter: --converter /Users/someone/lib/libmetalirconverter.dylib is not a regular file\\n' >&2\n\
             exit 2\n",
        )
        .expect("write the failing worker");
        let mut permissions = std::fs::metadata(&program)
            .expect("stat the failing worker")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&program, permissions).expect("chmod the failing worker");

        let pool = CompilerPool::build(
            PoolConfig {
                workers: 1,
                queue_depth: 1,
                cache_dir: None,
                ..PoolConfig::default()
            },
            Some(program),
            None,
            false,
        );
        let err = pool
            .compile(&CompileJob {
                id: 1,
                stage: 8,
                entry: "main",
                dxil: b"dxil",
            })
            .expect_err("a worker that exits before answering must fail the job");
        assert!(
            matches!(err, PoolError::WorkerExited { job_id: 1, .. }),
            "unexpected error: {err:?}"
        );
        let stderr = pool
            .last_worker_stderr()
            .expect("the exited worker's stderr must be captured");
        assert!(
            stderr.contains("--converter <path> is not a regular file"),
            "the captured stderr must keep the reason: {stderr}"
        );
        assert!(
            !stderr.contains("/Users"),
            "an absolute path must be redacted: {stderr}"
        );
        assert!(stderr.len() <= MAX_WORKER_STDERR_BYTES);
        let telemetry = pool.worker_stderr_for_telemetry();
        if stderr.chars().count() <= MAX_WORKER_STDERR_TELEMETRY_CHARS {
            assert_eq!(telemetry.as_deref(), Some(stderr.as_str()));
        } else {
            assert!(
                telemetry.is_none(),
                "long worker stderr must stay out of telemetry"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn worker_stderr_is_escaped_and_paths_are_redacted() {
        let text = escape_worker_stderr(
            b"victory-irconverter: --converter /Users/x/lib.dylib\nis not\ta file\0",
            false,
        );
        assert_eq!(
            text,
            "victory-irconverter: --converter <path> is not a file?"
        );
        assert_eq!(
            escape_worker_stderr(b"converter=C:\\tools\\lib.dylib failed", false),
            "converter=<path> failed"
        );
        assert_eq!(escape_worker_stderr(b"", false), "");
    }

    #[test]
    fn worker_stderr_is_bounded_and_marks_the_tail() {
        let raw = vec![b'a'; MAX_WORKER_STDERR_BYTES + 512];
        let text = escape_worker_stderr(&raw, true);
        assert!(
            text.len() <= MAX_WORKER_STDERR_BYTES,
            "{} bytes",
            text.len()
        );
        assert!(text.ends_with(" [truncated]"), "{text}");
    }

    #[test]
    fn worker_stderr_keeps_the_crash_line_after_a_long_banner() {
        let banner = "victory-irconverter: serving --stdio with converter <path>\n";
        let mut raw = banner.as_bytes().to_vec();
        raw.extend(std::iter::repeat_n(b'.', MAX_WORKER_STDERR_BYTES * 3));
        raw.extend_from_slice(b"\nconverter child died: signal 11\n");
        let text = escape_worker_stderr(&raw, true);
        assert!(
            text.ends_with("converter child died: signal 11 [truncated]"),
            "the tail, not the banner, must survive: {text}"
        );
        assert!(
            text.len() <= MAX_WORKER_STDERR_BYTES,
            "{} bytes",
            text.len()
        );
    }

    #[test]
    fn worker_exit_names_codes_and_signals() {
        assert_eq!(WorkerExit::Code(9).to_string(), "exit code 9");
        assert_eq!(WorkerExit::Signal(11).to_string(), "signal 11 (SIGSEGV)");
        assert_eq!(WorkerExit::Signal(6).to_string(), "signal 6 (SIGABRT)");
        assert_eq!(WorkerExit::Signal(77).to_string(), "signal 77");
        assert_eq!(WorkerExit::Unknown.to_string(), "unknown exit status");
        let death = WorkerDeath {
            exit: WorkerExit::Signal(11),
            stderr: String::new(),
        };
        assert_eq!(death.to_string(), "signal 11 (SIGSEGV)");
    }

    #[test]
    fn worker_stderr_reaches_telemetry_only_when_short() {
        assert_eq!(telemetry_stderr(None), None);
        assert_eq!(telemetry_stderr(Some("")), None);
        assert_eq!(
            telemetry_stderr(Some("converter worker exited")),
            Some("converter worker exited".to_string())
        );
        let long = "x".repeat(MAX_WORKER_STDERR_TELEMETRY_CHARS + 1);
        assert_eq!(telemetry_stderr(Some(&long)), None);
    }
}
