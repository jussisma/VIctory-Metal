//! Hardware proof for the asynchronous submit pipeline (ADR-017).
//!
//! `submit_v3` used to end every command buffer with `commit` +
//! `waitUntilCompleted`, so one frame at a time reached the GPU and the host
//! blocked on the CPU-side stall. These tests drive the new contract:
//!
//! 1. [`dependent_dispatches_pipeline_three_deep`] submits six dependent
//!    compute dispatches as six separate submissions and never waits inside a
//!    submit. It asserts that at least three command buffers were in flight at
//!    once, that the published fence does not reach the final value before the
//!    GPU work is done, that every value is exact (no lost or replayed work),
//!    and that a `Present` frame boundary leaves an empty retire queue.
//! 2. [`staging_reuse_waits_for_retirement`] recycles one staging page 200 times
//!    and only rewrites it after the retire queue reported the previous pin
//!    released, with a checksum that is wrong if any overlap happened.
//! 3. [`refused_submissions_never_publish`] proves that a submission refused
//!    before `commit` (illegal ordering, unknown handle, unregistered present
//!    surface) publishes no fence value and counts no submission.
//!
//! The corresponding CPU-only state-machine proof — a completion for a fence
//! that was never committed publishes nothing — lives in
//! `macos::device::async_pipeline_tests`.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test async_pipeline_gpu -- --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::{Duration, Instant};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::render::PixelFormat;
use victory_metal::wire::{BufferBinding, Dispatch1d, ReplyStatus};
use victory_metal::wire_v3::{CommandV3, Present, SubmissionV3};

/// Metal page size used for every zero-copy registration in this file.
const PAGE: usize = 16 * 1024;
/// Grid size of the pipelining kernel: large enough that one submission is
/// still executing while the next two commit, small enough to stay fast.
const WORK_THREADS: u32 = 1 << 20;
/// Bytes of the work buffer the pipelining kernel writes.
const WORK_BYTES: usize = WORK_THREADS as usize * 4;
/// Constant the work kernel XORs into every element.
const WORK_XOR: u32 = 0xA5A5_0000;
/// Wire handles used by the submissions below.
const COUNTER_ALLOCATION: u64 = 1;
const WORK_ALLOCATION: u64 = 2;
const STAGING_ALLOCATION: u64 = 3;
const GENERATION: u32 = 1;
/// Registered present surface used for the frame-boundary check.
const PRESENT_SURFACE: u64 = 9_001;
/// Staging reuse iterations; the checksum must equal `1 + .. + 200`.
const STAGING_ITERATIONS: u32 = 200;

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn free(ptr: *mut c_void);
}

/// Page-aligned host memory, registered zero-copy with Metal.
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
        // bytes on success; the returned pointer is owned by `HostPage`.
        let rc = unsafe { posix_memalign(&mut ptr, PAGE, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!ptr.is_null(), "posix_memalign returned null");
        // SAFETY: `ptr` is a live `len`-byte allocation.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, len) };
        Self { ptr, len }
    }

    fn words(&self) -> &[u32] {
        // SAFETY: `ptr` is a live allocation of `len` bytes, a multiple of 4.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u32>(), self.len / 4) }
    }

    /// Writes one dword from the host while the GPU is not reading the page.
    fn set_word(&self, index: usize, value: u32) {
        assert!(index < self.len / 4, "word index out of range");
        // SAFETY: `index` is in range of the live allocation; the caller
        // guarantees no command buffer still reads this page.
        unsafe { self.ptr.cast::<u32>().add(index).write(value) };
    }

    fn register(&self, transport: &DirectMetalTransport) -> VmBufferId {
        // SAFETY: the page stays live until after `wait_for_gpu` and the
        // transport is dropped, which is the documented registration contract.
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

/// One submission that runs the work kernel over a fixed grid and increments a
/// counter exactly once.
const CHAIN_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void chain(device uint *counter [[buffer(0)]],
                  device uint *work [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    work[tid] = tid ^ 0xA5A50000u;
    if (tid == 0u) {
        counter[0] = counter[0] + 1u;
    }
}
"#;

/// Adds one staging dword into an accumulator.
const ACCUMULATE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void accumulate(device uint *acc [[buffer(0)]],
                       device const uint *input [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) {
        acc[0] = acc[0] + input[0];
    }
}
"#;

fn dispatch(
    pipeline: u64,
    threads: u32,
    threads_per_group: u32,
    bindings: Vec<BufferBinding>,
) -> CommandV3 {
    CommandV3::Dispatch1d(Dispatch1d {
        pipeline_id: pipeline,
        threads,
        threads_per_group,
        bindings,
    })
}

fn binding(allocation_id: u64, length: u64) -> BufferBinding {
    BufferBinding {
        allocation_id,
        generation: GENERATION,
        offset: 0,
        length,
    }
}

fn submit(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    sequence: u32,
    commands: Vec<CommandV3>,
) -> ReplyStatus {
    let submission = SubmissionV3 {
        sequence,
        list: None,
        commands,
    };
    transport
        .submit_v3(&submission, allocations)
        .expect("a valid submission should be accepted")
        .status
}

/// Polls `predicate` until it holds or the deadline expires.
fn wait_until(mut predicate: impl FnMut() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !predicate() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::yield_now();
    }
}

/// Six dependent submissions, three deep, exact values, empty retire queue.
///
/// Real GPU work: the run must hold the shared `/tmp/victory-gpu.lock` and use
/// `--test-threads=1`.
#[test]
fn dependent_dispatches_pipeline_three_deep() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let pipeline = transport
        .compile_msl(CHAIN_MSL, "chain")
        .expect("chain kernel should compile");

    let counter = HostPage::zeroed(PAGE);
    let work = HostPage::zeroed(WORK_BYTES);
    let counter_id = counter.register(&transport);
    let work_id = work.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((COUNTER_ALLOCATION, GENERATION), counter_id);
    allocations.insert((WORK_ALLOCATION, GENERATION), work_id);

    // A registered colour target gives the last submission a real `Present`
    // frame boundary; presentation itself is not part of this test.
    let target = transport
        .create_render_target_2d(4, 4, PixelFormat::Rgba8Unorm)
        .expect("present target should be created");
    transport
        .register_present_surface(PRESENT_SURFACE, &target)
        .expect("present surface should register");

    let stats_before = transport.async_submit_stats();
    assert_eq!(stats_before.frames_in_flight_cap, 3);
    assert_eq!(stats_before.submitted, 0);
    assert_eq!(transport.fence_page().completed(), 0);

    let fence_page = transport.fence_page();
    let bindings = vec![
        binding(COUNTER_ALLOCATION, 4),
        binding(WORK_ALLOCATION, WORK_BYTES as u64),
    ];
    for sequence in 1..=6u32 {
        let mut commands = vec![dispatch(pipeline.0, WORK_THREADS, 256, bindings.clone())];
        if sequence == 6 {
            commands.push(CommandV3::Present(Present {
                surface_id: PRESENT_SURFACE,
                sync_interval: 1,
            }));
        }
        let status = submit(&transport, &allocations, sequence, commands);
        assert_eq!(status, ReplyStatus::Ok, "submission {sequence} was refused");
    }

    // No submission waited for its own command buffer, and the cap forced the
    // pipeline to overlap: three buffers were resident at the peak.
    let stats = transport.async_submit_stats();
    assert_eq!(stats.submitted, 6, "every accepted submission must commit");
    assert_eq!(stats.in_flight_peak, 3, "expected a three-deep pipeline");
    assert!(
        stats.backpressure_waits >= 1,
        "the fourth submission must have waited on the oldest buffer"
    );
    assert!(
        stats.retire_max_wait_micros > 0 || stats.backpressure_waits == 0,
        "a frame-boundary wait should be measured"
    );

    // The final fence cannot have been published yet: submission 6 was committed
    // microseconds ago and is still executing.
    assert!(
        fence_page.completed() < 6,
        "fence {} published before the work finished",
        fence_page.completed()
    );
    assert!(
        counter.words()[0] < 6,
        "the GPU ran all six dependent dispatches before any fence was polled"
    );

    // Pin the staging page behind the newest commit; the retire entry must stay
    // until the completion handler observes the fence.
    let pin = transport
        .pin_until_complete(0x57a6e)
        .expect("pinning an in-flight resource should succeed");
    assert_eq!(pin, Some(6), "the pin follows the newest committed fence");
    assert!(
        !transport.drain_retired().contains(&0x57a6e),
        "retire entry released before the fence was observed"
    );

    wait_until(|| fence_page.completed() >= 6, "the final fence value");
    transport
        .wait_for_gpu()
        .expect("the pipeline should drain");

    // Every dispatch incremented the counter exactly once and the last one wrote
    // the whole work buffer; nothing was lost, replayed or reordered.
    assert_eq!(
        counter.words()[0],
        6,
        "expected exactly one increment per submission"
    );
    let last = WORK_THREADS as usize - 1;
    assert_eq!(
        work.words()[last],
        (last as u32) ^ WORK_XOR,
        "the final work-buffer element is wrong"
    );
    assert_eq!(
        work.words()[0],
        WORK_XOR,
        "the first work-buffer element is wrong"
    );

    // The Present frame boundary and the completion handler both drained the
    // retire queue.
    let released = transport.drain_retired();
    assert!(
        released.contains(&0x57a6e),
        "the pinned resource was never released"
    );
    let after = transport.async_submit_stats();
    assert_eq!(after.in_flight, 0);
    assert_eq!(after.retire_depth, 0, "retire queue must be empty after Present");
    assert_eq!(after.completed, 6);
    assert_eq!(after.failed, 0);

    eprintln!(
        "async pipeline: submitted={} completed={} peak={} cap={} backpressure_waits={} \
         max_wait_us={} counter={} fence={}",
        after.submitted,
        after.completed,
        after.in_flight_peak,
        after.frames_in_flight_cap,
        after.backpressure_waits,
        after.retire_max_wait_micros,
        counter.words()[0],
        after.completed_fence,
    );
}

/// A staging page is rewritten only after its retire entry drained; 200
/// submissions accumulate a checksum that is wrong if any overlap happened.
#[test]
fn staging_reuse_waits_for_retirement() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let pipeline = transport
        .compile_msl(ACCUMULATE_MSL, "accumulate")
        .expect("accumulate kernel should compile");

    let accumulator = HostPage::zeroed(PAGE);
    let staging = HostPage::zeroed(PAGE);
    let acc_id = accumulator.register(&transport);
    let staging_id = staging.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((COUNTER_ALLOCATION, GENERATION), acc_id);
    allocations.insert((STAGING_ALLOCATION, GENERATION), staging_id);

    let fence_page = transport.fence_page();
    let mut released_total: Vec<u64> = Vec::new();
    let mut pending: Option<(u64, u64)> = None;

    for iteration in 0..STAGING_ITERATIONS {
        // Reuse the staging page only after the transport reported the previous
        // pin released, i.e. only after the fence it was pinned behind.
        if let Some((caller_id, fence)) = pending.take() {
            wait_until(
                || {
                    released_total.extend(transport.drain_retired());
                    released_total.contains(&caller_id)
                },
                "the previous staging pin to be released",
            );
            assert!(
                fence_page.completed() >= fence,
                "staging released while fence {fence} was still pending"
            );
        }

        staging.set_word(0, iteration + 1);
        let status = submit(
            &transport,
            &allocations,
            iteration + 1,
            vec![dispatch(
                pipeline.0,
                1,
                1,
                vec![
                    binding(COUNTER_ALLOCATION, 4),
                    binding(STAGING_ALLOCATION, 4),
                ],
            )],
        );
        assert_eq!(status, ReplyStatus::Ok, "iteration {iteration} was refused");

        let fence = transport
            .pin_until_complete(u64::from(iteration))
            .expect("pinning the staging page should succeed")
            .expect("a submission is in flight");
        assert_eq!(fence, u64::from(iteration) + 1);
        // Safety invariant: a release may only be reported after the published
        // fence covers it.
        let released = transport.drain_retired();
        if released.contains(&u64::from(iteration)) {
            assert!(fence_page.completed() >= fence);
        }
        released_total.extend(released);
        pending = Some((u64::from(iteration), fence));
    }

    if let Some((caller_id, _)) = pending.take() {
        wait_until(
            || {
                released_total.extend(transport.drain_retired());
                released_total.contains(&caller_id)
            },
            "the last staging pin to be released",
        );
    }
    transport.wait_for_gpu().expect("the pipeline should drain");

    let expected: u32 = (1..=STAGING_ITERATIONS).sum();
    assert_eq!(
        accumulator.words()[0],
        expected,
        "the staging checksum is wrong: a page was rewritten while the GPU read it"
    );
    let stats = transport.async_submit_stats();
    assert_eq!(stats.submitted, u64::from(STAGING_ITERATIONS));
    assert_eq!(stats.in_flight, 0);
    assert_eq!(stats.retire_depth, 0, "every pin must be retired");
    assert_eq!(stats.rejected, 0);

    eprintln!(
        "staging reuse: iterations={} checksum={} submitted={} peak={} max_wait_us={}",
        STAGING_ITERATIONS,
        accumulator.words()[0],
        stats.submitted,
        stats.in_flight_peak,
        stats.retire_max_wait_micros,
    );
}

/// Refusals are decided before `commit`, so they never publish a fence value.
#[test]
fn refused_submissions_never_publish() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let pipeline = transport
        .compile_msl(ACCUMULATE_MSL, "accumulate")
        .expect("accumulate kernel should compile");
    let counter = HostPage::zeroed(PAGE);
    let staging = HostPage::zeroed(PAGE);
    let counter_id = counter.register(&transport);
    let staging_id = staging.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((COUNTER_ALLOCATION, GENERATION), counter_id);
    allocations.insert((STAGING_ALLOCATION, GENERATION), staging_id);
    let fence_page = transport.fence_page();
    staging.set_word(0, 1);

    // 1. Illegal ordering: compute inside an open render pass is an explicit
    // error before any command buffer exists.
    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            dispatch(
                pipeline.0,
                1,
                1,
                vec![binding(STAGING_ALLOCATION, 4)],
            ),
            CommandV3::BeginRenderPass(victory_metal::wire_v3::BeginRenderPass {
                colors: vec![victory_metal::wire_v3::ColorAttachment {
                    allocation_id: COUNTER_ALLOCATION,
                    generation: GENERATION,
                    mip_level: 0,
                    slice: 0,
                    load: victory_metal::wire_v3::LoadOp::Clear,
                    store: victory_metal::wire_v3::StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                }],
                depth_stencil: None,
            }),
        ],
    };
    let error = transport
        .submit_v3(&submission, &allocations)
        .expect_err("an illegal ordering must be refused before any commit");
    assert!(
        matches!(error, victory_metal::transport::MetalError::InvalidArg(_)),
        "unexpected error: {error:?}"
    );
    assert_eq!(transport.fence_page().completed(), 0);

    // 2. Unknown pipeline: an explicit reply, still no commit and no publish.
    let status = submit(
        &transport,
        &allocations,
        2,
        vec![dispatch(
            999_999,
            1,
            1,
            vec![binding(STAGING_ALLOCATION, 4)],
        )],
    );
    assert_eq!(status, ReplyStatus::UnknownHandle);
    assert_eq!(fence_page.completed(), 0);

    // 3. A `Present` naming an unregistered surface is refused the same way.
    let status = submit(
        &transport,
        &allocations,
        3,
        vec![CommandV3::Present(Present {
            surface_id: 424_242,
            sync_interval: 1,
        })],
    );
    assert_eq!(status, ReplyStatus::UnknownHandle);
    assert_eq!(fence_page.completed(), 0);

    let refused = transport.async_submit_stats();
    assert_eq!(refused.submitted, 0, "no refusal may reach commit");
    assert_eq!(refused.in_flight, 0);
    assert_eq!(refused.rejected, 3);

    // 4. A real submission publishes exactly one fence value, after it runs.
    let status = submit(
        &transport,
        &allocations,
        4,
        vec![dispatch(
            pipeline.0,
            1,
            1,
            vec![
                binding(COUNTER_ALLOCATION, 4),
                binding(STAGING_ALLOCATION, 4),
            ],
        )],
    );
    assert_eq!(status, ReplyStatus::Ok);
    transport.wait_for_gpu().expect("the pipeline should drain");
    assert_eq!(fence_page.completed(), 4);
    assert_eq!(counter.words()[0], 1);

    let stats = transport.async_submit_stats();
    assert_eq!(stats.submitted, 1);
    assert_eq!(stats.completed, 1);
    assert_eq!(stats.in_flight, 0);
    assert_eq!(stats.retire_depth, 0);
    assert_eq!(stats.ignored_completions, 0);

    eprintln!(
        "refusals: rejected={} submitted={} completed={} fence={}",
        stats.rejected, stats.submitted, stats.completed, stats.completed_fence,
    );
}
