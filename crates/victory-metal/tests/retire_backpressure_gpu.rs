//! Hardware proof that the retire queue is a **hard** bound under load.
//!
//! The retire queue pins staging objects and caller ids behind the fence of the
//! command buffer that may still read them. It used to accept pins over capacity
//! and merely count the refusal, so a stalled completion grew host memory without
//! limit. The transport now applies backpressure on the oldest live fence and
//! fails a pin with a typed [`MetalError::Backpressure`] when that fence never
//! completes.
//!
//! 1. [`pins_never_exceed_the_hard_retire_capacity`] submits several dependent
//!    dispatches, then pins far more caller ids than the configured capacity
//!    while the GPU is still busy. It asserts the observed depth never exceeds
//!    the capacity, that the soft path counted no refusal, that every pin is
//!    released exactly once in FIFO order, and that the guest fence page reaches
//!    the last submitted sequence.
//! 2. [`lost_completion_is_a_typed_error_not_a_hang`] suppresses completion
//!    publication (a lost completion handler) and proves the next pin fails with
//!    `MetalError::Backpressure` inside the bounded wait instead of blocking the
//!    host. Both tests run under a watchdog that aborts the process after
//!    [`WATCHDOG_TIMEOUT`], so a deadlock is a failure rather than a hung suite.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test retire_backpressure_gpu -- --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use victory_metal::MetalError;
use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::wire::{BufferBinding, Dispatch1d, ReplyStatus};
use victory_metal::wire_v3::{CommandV3, SubmissionV3};

/// Metal page size used for the zero-copy work buffer.
const PAGE: usize = 16 * 1024;
/// Grid size: large enough that the GPU is still busy while the host pins, small
/// enough to stay well under a second per submission.
const WORK_THREADS: u32 = 1 << 22;
/// Bytes the work kernel writes.
const WORK_BYTES: usize = WORK_THREADS as usize * 4;
/// Registered wire allocation id of the work buffer.
const WORK_ALLOCATION: u64 = 1;
const GENERATION: u32 = 1;
/// Hard retire bound used by the bounded-depth test.
const RETIRE_CAPACITY: usize = 4;
/// How many caller pins the test forces through that bound.
const PINS: u64 = 24;
/// Abort the process if a test has not finished in this long. A regression that
/// deadlocks the pipeline must fail the suite, not hang it.
const WATCHDOG_TIMEOUT: Duration = Duration::from_secs(30);

/// Writes a distinct pattern into the work buffer so the GPU is genuinely busy
/// while the host pins: a few hundred dependent integer steps per thread keeps
/// one submission well above the microsecond scale the pin loop runs at.
const WORK_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void work(device uint *work [[buffer(0)]],
                 uint tid [[thread_position_in_grid]]) {
    uint x = tid * 2654435761u;
    for (uint step = 0u; step < 256u; step++) {
        x = x * 1664525u + 1013904223u;
    }
    work[tid] = x ^ 0x5A5A5A5Au;
}
"#;

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn free(ptr: *mut c_void);
}

/// Page-aligned host memory registered zero-copy with Metal.
struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn zeroed(len: usize) -> Self {
        assert!(
            len.is_multiple_of(PAGE),
            "a zero-copy page must be a multiple of {PAGE} bytes"
        );
        let mut ptr: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a page-aligned allocation of `len`
        // bytes on success; the pointer is owned by `HostPage`.
        let rc = unsafe { posix_memalign(&mut ptr, PAGE, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!ptr.is_null(), "posix_memalign returned null");
        // SAFETY: `ptr` is a live `len`-byte allocation.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, len) };
        Self { ptr, len }
    }

    fn register(&self, transport: &DirectMetalTransport) -> VmBufferId {
        // SAFETY: the page outlives every command buffer and the transport, which
        // is the documented zero-copy registration contract.
        unsafe { transport.register_zerocopy(self.ptr.cast::<u8>(), self.len) }
            .expect("zero-copy registration should succeed")
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `posix_memalign` and is freed exactly once.
        unsafe { free(self.ptr) };
    }
}

/// Aborts the process when a test overruns [`WATCHDOG_TIMEOUT`].
///
/// A stuck `waitUntilCompleted` or a lock held across it cannot be interrupted
/// from Rust, so the only way to turn a deadlock into a test failure is a
/// separate thread that aborts. Dropping the guard disarms it.
struct Watchdog {
    done: Arc<AtomicBool>,
}

impl Watchdog {
    fn arm(test_name: &'static str) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        std::thread::spawn(move || {
            let deadline = Instant::now() + WATCHDOG_TIMEOUT;
            while Instant::now() < deadline {
                if flag.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            eprintln!(
                "watchdog: {test_name} did not finish within {WATCHDOG_TIMEOUT:?}; \
                 the retire backpressure path deadlocked"
            );
            std::process::abort();
        });
        Self { done }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
    }
}

fn dispatch(pipeline: u64, threads: u32) -> CommandV3 {
    CommandV3::Dispatch1d(Dispatch1d {
        pipeline_id: pipeline,
        threads,
        threads_per_group: 256,
        bindings: vec![BufferBinding {
            allocation_id: WORK_ALLOCATION,
            generation: GENERATION,
            offset: 0,
            length: WORK_BYTES as u64,
        }],
    })
}

/// Compiles the work kernel and returns a transport plus the allocation table
/// for the registered work page.
fn setup(
    retire_capacity: usize,
) -> (
    DirectMetalTransport,
    HostPage,
    HashMap<(u64, u32), VmBufferId>,
    u64,
) {
    let transport = DirectMetalTransport::with_retire_capacity(retire_capacity)
        .expect("Metal device should open");
    let pipeline = transport
        .compile_msl(WORK_MSL, "work")
        .expect("work kernel should compile");
    let work = HostPage::zeroed(WORK_BYTES);
    let work_id = work.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((WORK_ALLOCATION, GENERATION), work_id);
    (transport, work, allocations, pipeline.0)
}

/// Submits one real dispatch with the pipeline id found by [`setup`].
fn submit_work(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    pipeline: u64,
    sequence: u32,
) -> ReplyStatus {
    let submission = SubmissionV3 {
        sequence,
        list: None,
        commands: vec![dispatch(pipeline, WORK_THREADS)],
    };
    transport
        .submit_v3(&submission, allocations)
        .expect("a valid submission should be accepted")
        .status
}

/// More pins than the hard retire capacity, applied while the GPU is still
/// reading: depth stays bounded, every pin is released once in FIFO order, and
/// the guest fence reaches the last sequence.
#[test]
fn pins_never_exceed_the_hard_retire_capacity() {
    let _watchdog = Watchdog::arm("pins_never_exceed_the_hard_retire_capacity");
    let (transport, _work, allocations, pipeline) = setup(RETIRE_CAPACITY);
    let fence_page = transport.fence_page();

    let mut last_sequence = 0u32;
    let mut released: Vec<u64> = Vec::new();

    for pin in 0..PINS {
        // Keep one submission in flight: the pin is only meaningful behind live
        // GPU work, and a drained watermark makes `pin_until_complete` a no-op.
        if transport.async_submit_stats().in_flight == 0 {
            last_sequence += 1;
            assert_eq!(
                submit_work(&transport, &allocations, pipeline, last_sequence),
                ReplyStatus::Ok,
                "submission {last_sequence} was refused"
            );
        }
        let fence = transport
            .pin_until_complete(pin)
            .expect("a pin behind in-flight work must be accepted")
            .expect("a submission is in flight");
        assert!(fence >= 1, "pin {pin} behind fence {fence}");

        let stats = transport.async_submit_stats();
        assert!(
            stats.retire_depth <= RETIRE_CAPACITY,
            "retire depth {} exceeded the hard capacity {RETIRE_CAPACITY} at pin {pin}",
            stats.retire_depth
        );
        assert!(
            stats.retire_high_watermark <= RETIRE_CAPACITY,
            "retire high watermark {} exceeded the hard capacity {RETIRE_CAPACITY}",
            stats.retire_high_watermark
        );
        assert_eq!(
            stats.retire_refusals, 0,
            "the device must never take the soft over-capacity path"
        );
        assert_eq!(stats.retire_capacity, RETIRE_CAPACITY);

        released.extend(transport.drain_retired());
    }

    // Complete everything, then drain what is left.
    transport
        .wait_for_gpu()
        .expect("the pipeline should drain without a deadlock");
    released.extend(transport.drain_retired());
    if transport.async_submit_stats().retire_depth > 0 {
        transport.wait_for_gpu().expect("a second drain");
        released.extend(transport.drain_retired());
    }

    let stats = transport.async_submit_stats();
    assert_eq!(stats.retire_depth, 0, "every pin must be retired");
    assert!(
        stats.retire_high_watermark <= RETIRE_CAPACITY,
        "peak depth {} exceeded the hard capacity {RETIRE_CAPACITY}",
        stats.retire_high_watermark
    );
    assert_eq!(stats.retire_refusals, 0);
    assert!(
        stats.backpressure_waits >= 1,
        "a pin above the capacity must have waited on the oldest fence"
    );
    assert_eq!(
        released,
        (0..PINS).collect::<Vec<_>>(),
        "every pin must be released exactly once, in FIFO order"
    );
    assert!(
        stats.submitted >= 2,
        "the test must have kept the pipeline fed (submitted {})",
        stats.submitted
    );
    assert_eq!(stats.completed, u64::from(last_sequence));
    assert_eq!(
        fence_page.completed(),
        u64::from(last_sequence),
        "the guest fence must reach the last submitted sequence"
    );

    eprintln!(
        "retire backpressure bounded: pins={PINS} capacity={RETIRE_CAPACITY} \
         peak={} submitted={} waits={} max_wait_us={} fence={} released={}",
        stats.retire_high_watermark,
        stats.submitted,
        stats.backpressure_waits,
        stats.retire_max_wait_micros,
        fence_page.completed(),
        released.len()
    );
}

/// A completion that is never published must surface as a typed error inside the
/// bounded wait, not as a host hang.
#[test]
fn lost_completion_is_a_typed_error_not_a_hang() {
    let _watchdog = Watchdog::arm("lost_completion_is_a_typed_error_not_a_hang");
    const STUCK_CAPACITY: usize = 2;
    let (transport, _work, allocations, pipeline) = setup(STUCK_CAPACITY);
    let fence_page = transport.fence_page();

    // Suppress publication before the first submission so not even the very
    // first completion handler can make room: this is the "lost completion"
    // failure the bounded wait must survive.
    transport
        .set_stall_completion_publication(true)
        .expect("the diagnostics hook must be settable");

    assert_eq!(
        submit_work(&transport, &allocations, pipeline, 1),
        ReplyStatus::Ok
    );

    // Fill the hard capacity; both pins are behind fence 1.
    assert_eq!(
        transport
            .pin_until_complete(0xdead)
            .expect("pin 1")
            .expect("a submission is in flight"),
        1
    );
    assert_eq!(
        transport
            .pin_until_complete(0xbeef)
            .expect("pin 2")
            .expect("a submission is in flight"),
        1
    );
    assert_eq!(transport.async_submit_stats().retire_depth, STUCK_CAPACITY);

    // The GPU work finishes, but its publication is suppressed: the queue stays
    // full and the next pin must give up after the bounded number of waits.
    let started = Instant::now();
    let error = transport
        .pin_until_complete(0xfeed)
        .expect_err("a lost completion must refuse the pin");
    let elapsed = started.elapsed();
    match error {
        MetalError::Backpressure {
            capacity,
            oldest,
            completed,
            waits,
        } => {
            assert_eq!(capacity, STUCK_CAPACITY);
            assert_eq!(oldest, 1, "the oldest live fence gates the queue");
            assert_eq!(completed, 0, "no completion was ever published");
            assert_eq!(
                waits,
                victory_metal::macos::retire::RetireQueue::MAX_BACKPRESSURE_WAITS,
                "the pin must use exactly the documented budget"
            );
        }
        other => panic!("expected a typed backpressure error, got {other:?}"),
    }
    assert!(
        elapsed < WATCHDOG_TIMEOUT,
        "the bounded wait took {elapsed:?}; it must fail well inside the watchdog"
    );
    assert_eq!(
        fence_page.completed(),
        0,
        "the suppressed publication must not advance the guest fence"
    );
    let stats = transport.async_submit_stats();
    assert_eq!(stats.retire_depth, STUCK_CAPACITY, "nothing was dropped");
    assert_eq!(stats.retire_refusals, 0);

    // Recover: publication is restored, the barrier records the completion it
    // waits on, and the two live pins are released.
    transport
        .set_stall_completion_publication(false)
        .expect("the diagnostics hook must be clearable");
    transport
        .wait_for_gpu()
        .expect("the pipeline should recover");
    let released = transport.drain_retired();
    assert!(released.contains(&0xdead), "pin 1 was never released");
    assert!(released.contains(&0xbeef), "pin 2 was never released");
    assert_eq!(fence_page.completed(), 1);
    assert_eq!(transport.async_submit_stats().retire_depth, 0);

    eprintln!(
        "retire backpressure stuck fence: capacity={STUCK_CAPACITY} oldest=1 completed=0 \
         elapsed_us={} waits={} released={}",
        elapsed.as_micros(),
        victory_metal::macos::retire::RetireQueue::MAX_BACKPRESSURE_WAITS,
        released.len()
    );
}
