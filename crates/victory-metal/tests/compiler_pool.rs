//! Multi-process shader pool behaviour without XPC or a Metal device.
//!
//! Every test drives the documented `VMJOB`/`VMRES` frame protocol against a
//! tiny POSIX `sh` fixture, so pool concurrency, id matching, cache
//! accounting, crash containment, timeouts and environment clamping are
//! covered on any aarch64 macOS host. The real `victory-irconverter` binary is
//! only used to prove that a worker which does not speak the protocol fails
//! explicitly instead of hanging.

#![cfg(target_os = "macos")]

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use victory_metal::compiler_pool::{
    CompileJob, CompilerPool, DEFAULT_WORKERS, MAX_WORKERS, PoolConfig, PoolError,
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "victory-metal-pool-{label}-{}-{sequence}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Writes the `sh` fixture that speaks protocol version 1.
///
/// Modes selected by the job's entry point:
/// * `crash`  — exits before answering (mid-job worker death);
/// * `hang`   — sleeps 30 s before answering (caller timeout);
/// * `badseq` — echoes a different internal sequence number;
/// * `badid`  — echoes a different caller id;
/// * `fail`   — answers with status 7 and a diagnostic;
/// * `empty`  — answers with a zero-length payload;
/// * anything else is echoed back as `entry ++ dxil`.
fn write_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("pool-fixture.sh");
    let script = r#"#!/bin/sh
set -u
while IFS=' ' read -r magic seq id stage entry_len dxil_len; do
    [ "$magic" = "VMJOB" ] || exit 3
    scratch="${TMPDIR:-/tmp}/victory-pool-fixture-$$"
    dd bs=1 count="$entry_len" of="$scratch.entry" 2>/dev/null
    dd bs=1 count="$dxil_len" of="$scratch.dxil" 2>/dev/null
    entry=$(cat "$scratch.entry")
    case "$entry" in
        crash)
            rm -f "$scratch.entry" "$scratch.dxil"
            exit 9
            ;;
        hang)
            sleep 30
            rm -f "$scratch.entry" "$scratch.dxil"
            exit 9
            ;;
        badseq)
            seq=$((seq + 1))
            ;;
        badid)
            id=$((id + 1))
            ;;
        fail)
            printf 'VMRES %s %s 7 9\nboom-fail' "$seq" "$id"
            rm -f "$scratch.entry" "$scratch.dxil"
            continue
            ;;
        empty)
            printf 'VMRES %s %s 0 0\n' "$seq" "$id"
            rm -f "$scratch.entry" "$scratch.dxil"
            continue
            ;;
    esac
    printf 'VMRES %s %s 0 %s\n' "$seq" "$id" "$((entry_len + dxil_len))"
    cat "$scratch.entry"
    cat "$scratch.dxil"
    rm -f "$scratch.entry" "$scratch.dxil"
done
exit 0
"#;
    let mut file = std::fs::File::create(&path).expect("create fixture script");
    file.write_all(script.as_bytes())
        .expect("write fixture script");
    file.sync_all().expect("sync fixture script");
    let mut permissions = std::fs::metadata(&path)
        .expect("stat fixture")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&path, permissions).expect("chmod fixture");
    path
}

fn test_config(dir: &Path) -> PoolConfig {
    PoolConfig {
        workers: 2,
        queue_depth: 16,
        job_timeout: Duration::from_secs(10),
        cache_dir: Some(dir.join("cache")),
        converter_version: "victory-metal-test/pool/1".to_string(),
    }
}

fn job<'a>(id: u64, entry: &'a str, dxil: &'a [u8]) -> CompileJob<'a> {
    CompileJob {
        id,
        stage: 8,
        entry,
        dxil,
    }
}

/// Test (a): eight concurrent callers on two workers each receive exactly their
/// own entry and DXIL bytes, proving dispatch and id-matched routing.
#[test]
fn eight_concurrent_jobs_across_two_workers_return_matched_results() {
    let dir = TempDir::new("concurrent");
    let program = write_fixture(dir.path());
    let pool = Arc::new(CompilerPool::new(
        test_config(dir.path()),
        Some(program.clone()),
    ));

    let callers = 8usize;
    let barrier = Arc::new(Barrier::new(callers));
    let mut handles = Vec::with_capacity(callers);
    for index in 0..callers {
        let pool = pool.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            let entry = format!("job{index}");
            // Non-trivial, mostly non-UTF-8 payload so a byte swap would show.
            let dxil: Vec<u8> = (0..4096u32)
                .map(|byte| (byte as u8).wrapping_mul(37).wrapping_add(index as u8))
                .collect();
            barrier.wait();
            let result = pool.compile(&job(100 + index as u64, &entry, &dxil));
            (entry, dxil, result)
        }));
    }
    for handle in handles {
        let (entry, dxil, result) = handle.join().expect("caller thread");
        let metallib = result.expect("job should succeed");
        let mut expected = entry.clone().into_bytes();
        expected.extend_from_slice(&dxil);
        assert_eq!(
            metallib, expected,
            "caller {entry} received another job's bytes"
        );
    }

    let stats = pool.stats();
    assert_eq!(stats.jobs, callers as u64);
    assert_eq!(stats.completed, callers as u64);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.cache_misses, callers as u64);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.in_flight, 0);
    assert_eq!(
        stats.worker_starts, 2,
        "both configured workers must have been started"
    );
    assert_eq!(pool.workers(), 2);
}

/// Test (b): a repeated job is served from the disk cache without touching a
/// worker again.
#[test]
fn cache_hit_avoids_a_second_worker_start() {
    let dir = TempDir::new("cache-hit");
    let program = write_fixture(dir.path());
    let pool = CompilerPool::new(test_config(dir.path()), Some(program));
    let dxil = b"dxil-content-for-cache".to_vec();

    let first = pool
        .compile(&job(1, "main", &dxil))
        .expect("cold compile should succeed");
    let first_stats = pool.stats();
    assert_eq!(first_stats.worker_starts, 1);
    assert_eq!(first_stats.cache_misses, 1);
    assert_eq!(first_stats.cache_hits, 0);

    let second = pool
        .compile(&job(2, "main", &dxil))
        .expect("warm compile should hit the cache");
    assert_eq!(second, first);
    let stats = pool.stats();
    assert_eq!(stats.cache_hits, 1, "second compile must be a cache hit");
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(
        stats.worker_starts, 1,
        "a cache hit must not start or wake another worker"
    );

    let entries: Vec<_> = std::fs::read_dir(pool.cache_dir().expect("cache enabled"))
        .expect("read cache dir")
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("vce"))
        .collect();
    assert_eq!(entries.len(), 1, "one content-addressed entry on disk");
}

/// Test (c): a worker that exits mid-job is an explicit error, and the pool
/// respawns for the next job.
#[test]
fn worker_crash_is_explicit_and_pool_stays_usable() {
    let dir = TempDir::new("crash");
    let program = write_fixture(dir.path());
    let pool = CompilerPool::new(test_config(dir.path()), Some(program));

    let err = pool
        .compile(&job(1, "crash", b"dxil"))
        .expect_err("a dead worker must not produce a result");
    assert!(
        matches!(err, PoolError::WorkerExited { job_id: 1 }),
        "unexpected crash error: {err:?}"
    );

    let recovered = pool
        .compile(&job(2, "main", b"dxil-after-crash"))
        .expect("pool must remain usable after a worker crash");
    let mut expected = b"main".to_vec();
    expected.extend_from_slice(b"dxil-after-crash");
    assert_eq!(recovered, expected);

    let stats = pool.stats();
    assert!(
        stats.worker_restarts >= 1,
        "crash must be counted: {stats:?}"
    );
    assert!(
        stats.worker_starts >= 2,
        "worker must be respawned: {stats:?}"
    );
    assert_eq!(stats.errors, 1);
    assert_eq!(stats.completed, 1);
}

/// A caller that exceeds the job timeout gets an explicit error, its worker is
/// killed, and later jobs still run.
#[test]
fn worker_timeout_is_explicit_and_pool_stays_usable() {
    let dir = TempDir::new("timeout");
    let program = write_fixture(dir.path());
    let config = PoolConfig {
        job_timeout: Duration::from_millis(400),
        ..test_config(dir.path())
    };
    let pool = CompilerPool::new(config, Some(program));

    let err = pool
        .compile(&job(1, "hang", b"dxil"))
        .expect_err("a hung worker must time out");
    match err {
        PoolError::Timeout { job_id, timeout_ms } => {
            assert_eq!(job_id, 1);
            assert_eq!(timeout_ms, 400);
        }
        other => panic!("unexpected timeout error: {other:?}"),
    }

    let recovered = pool
        .compile(&job(2, "main", b"dxil-after-timeout"))
        .expect("pool must remain usable after a timeout");
    let mut expected = b"main".to_vec();
    expected.extend_from_slice(b"dxil-after-timeout");
    assert_eq!(recovered, expected);
    assert!(pool.stats().errors >= 1);
}

/// A worker that answers with the wrong sequence or caller id is rejected.
#[test]
fn mismatched_worker_frames_are_rejected() {
    let dir = TempDir::new("mismatch");
    let program = write_fixture(dir.path());
    let pool = CompilerPool::new(test_config(dir.path()), Some(program));

    let err = pool
        .compile(&job(1, "badseq", b"dxil"))
        .expect_err("a wrong sequence must be rejected");
    assert!(
        matches!(
            err,
            PoolError::SequenceMismatch {
                expected: 0,
                got: 1
            }
        ),
        "unexpected sequence error: {err:?}"
    );

    let err = pool
        .compile(&job(2, "badid", b"dxil"))
        .expect_err("a wrong caller id must be rejected");
    assert!(
        matches!(
            err,
            PoolError::IdMismatch {
                expected: 2,
                got: 3
            }
        ),
        "unexpected id error: {err:?}"
    );

    // Worker status and empty payloads are explicit too.
    let err = pool
        .compile(&job(3, "fail", b"dxil"))
        .expect_err("worker status 7 must be an error");
    assert!(
        matches!(
            err,
            PoolError::WorkerFailed {
                job_id: 3,
                status: 7,
                ref message
            } if message == "boom-fail"
        ),
        "unexpected status error: {err:?}"
    );

    // A zero-length payload is a valid (empty) result at the frame level.
    assert_eq!(
        pool.compile(&job(4, "empty", b"dxil"))
            .expect("empty payload is well framed"),
        Vec::<u8>::new()
    );
}

/// Test (d): `VICTORY_SHADER_WORKERS=99` clamps to 8 and `=0` clamps to 1.
#[test]
fn env_worker_counts_are_clamped() {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: This test file is run with --test-threads=1 and no other
            // test reads `VICTORY_SHADER_WORKERS`.
            unsafe { std::env::remove_var("VICTORY_SHADER_WORKERS") };
        }
    }
    let _restore = Restore;

    // SAFETY: See `Restore`; the variable is process-global state.
    unsafe { std::env::set_var("VICTORY_SHADER_WORKERS", "99") };
    assert_eq!(PoolConfig::from_env().workers, MAX_WORKERS);
    assert_eq!(PoolConfig::from_env().workers.clamp(1, MAX_WORKERS), 8);

    // SAFETY: Same single-threaded test contract as above.
    unsafe { std::env::set_var("VICTORY_SHADER_WORKERS", "0") };
    assert_eq!(PoolConfig::from_env().workers, 1);

    // SAFETY: Same single-threaded test contract as above.
    unsafe { std::env::set_var("VICTORY_SHADER_WORKERS", "not-a-number") };
    assert_eq!(PoolConfig::from_env().workers, DEFAULT_WORKERS);

    // SAFETY: Same single-threaded test contract as above.
    unsafe { std::env::remove_var("VICTORY_SHADER_WORKERS") };
    assert_eq!(PoolConfig::from_env().workers, DEFAULT_WORKERS);
}

/// The real `victory-irconverter` binary has no stdio mode yet, so a pool
/// pointed at it must fail explicitly and quickly instead of hanging.
#[test]
fn real_irconverter_binary_without_stdio_protocol_fails_explicitly() {
    let _dir = TempDir::new("real-binary");
    let config = PoolConfig {
        workers: 1,
        queue_depth: 1,
        job_timeout: Duration::from_secs(10),
        cache_dir: None,
        converter_version: "victory-metal-test/pool/1".to_string(),
    };
    let pool = CompilerPool::new(
        config,
        Some(PathBuf::from(env!("CARGO_BIN_EXE_victory-irconverter"))),
    );
    let started = std::time::Instant::now();
    let err = pool
        .compile(&job(1, "main", b"dxil"))
        .expect_err("the path-in/path-out binary does not speak the frame protocol");
    assert!(
        matches!(
            err,
            PoolError::WorkerExited { .. }
                | PoolError::WorkerRead { .. }
                | PoolError::Protocol { .. }
        ),
        "unexpected error from the real binary: {err:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "a non-conforming worker must not hang the pool"
    );
}
