#![cfg(target_os = "macos")]
//! End-to-end proof of the transport-agnostic shared completion fence (ADR-012).
//!
//! The tests cover three layers:
//!
//! * [`FencePage`] has a fixed one-page ABI and left-to-right monotonic
//!   publication.
//! * [`FencePageMapping`] makes the page visible through two independent
//!   `MAP_SHARED` handles, which is the same mechanism a guest process uses.
//! * [`LoopbackAsyncTransport`] submits a version 3 frame over the real
//!   loopback host session and observes completion through the shared page with
//!   no reply frame consumed, for success, replay and executor-failure cases.
//!
//! The host keeps using the synchronous `submit_v3` device entry point, so these
//! tests are CPU-only: they prove the wire, ledger and fence-publication path,
//! not GPU timing.

use std::io::Read;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use victory_metal::VmBufferId;
use victory_metal::fence_page::{
    FENCE_PAGE_ALIGN, FENCE_PAGE_LEN, FencePage, FencePageMapping, LoopbackTcpWake, UnixSocketWake,
    Wake,
};
use victory_metal::ipc_host::{
    BatchExecutor, FENCE_PAGE_FILE_NAME, HostError, HostSecret, IpcHost, LoopbackAsyncTransport,
    SessionSummary,
};
use victory_metal::transport::{GuestTransport, MetalError};
use victory_metal::wire::{ListIdentity, Reply, ReplyStatus, Submission};
use victory_metal::wire_v3::{CommandV3, SubmissionV3, encode_submission_v3};

const SECRET: [u8; 32] = [0x5a; 32];

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-transport-fence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create test dir");
        Self(path)
    }

    fn fence_page_path(&self) -> PathBuf {
        self.0.join(FENCE_PAGE_FILE_NAME)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct StubState {
    /// Sequences the executor was asked to run, including the failing one.
    attempts: Vec<u32>,
    /// Sequences that completed successfully.
    completed: Vec<u32>,
    fail_sequence: Option<u32>,
}

#[derive(Clone)]
struct StubExecutor {
    state: Arc<Mutex<StubState>>,
}

impl StubExecutor {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(StubState::default())),
        }
    }

    fn set_fail_sequence(&self, sequence: u32) {
        self.state.lock().expect("stub state").fail_sequence = Some(sequence);
    }

    fn attempted(&self) -> Vec<u32> {
        self.state.lock().expect("stub state").attempts.clone()
    }

    fn completed(&self) -> Vec<u32> {
        self.state.lock().expect("stub state").completed.clone()
    }
}

impl BatchExecutor for StubExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        Ok(4096)
    }

    fn register_window(&self, _ptr: *mut u8, _len: usize) -> Result<VmBufferId, MetalError> {
        Ok(VmBufferId(1))
    }

    fn unregister_window(&self, _id: VmBufferId) -> Result<(), MetalError> {
        Ok(())
    }

    fn submit_batch(
        &self,
        submission: &Submission,
        _allocations: &std::collections::HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        _allocations: &std::collections::HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let mut state = self.state.lock().expect("stub state");
        state.attempts.push(submission.sequence);
        if state.fail_sequence == Some(submission.sequence) {
            return Err(MetalError::InternalError(
                "synthetic device failure".to_string(),
            ));
        }
        state.completed.push(submission.sequence);
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }
}

struct Host {
    addr: SocketAddr,
    handle: thread::JoinHandle<(Result<SessionSummary, HostError>, IpcHost<StubExecutor>)>,
}

fn start_host(dir: &TestDir, executor: StubExecutor) -> Host {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind host listener");
    let addr = listener.local_addr().expect("host listener address");
    let mut host = IpcHost::new(HostSecret::new(SECRET), &dir.0, executor)
        .expect("create host")
        .with_io_timeout(Duration::from_secs(10));
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept guest");
        let result = host.serve_once(stream);
        (result, host)
    });
    Host { addr, handle }
}

/// One valid version 3 dispatch with no buffer bindings.
///
/// Empty bindings pass host reference validation, which is all these CPU-only
/// wire tests need; the executor is a stub, so no GPU work runs.
fn submission(sequence: u32, list: Option<ListIdentity>) -> SubmissionV3 {
    SubmissionV3 {
        sequence,
        list,
        commands: vec![CommandV3::Dispatch1d(victory_metal::wire::Dispatch1d {
            pipeline_id: 1,
            threads: 1,
            threads_per_group: 1,
            bindings: Vec::new(),
        })],
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !predicate() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------
// Page ABI and wake channels
// ---------------------------------------------------------------------------

#[test]
fn fence_page_abi_is_one_aligned_page() {
    assert_eq!(std::mem::size_of::<FencePage>(), 4096);
    assert_eq!(FENCE_PAGE_LEN, 4096);
    assert_eq!(std::mem::align_of::<FencePage>(), 4096);
    assert_eq!(FENCE_PAGE_ALIGN, 4096);

    let page = FencePage::default();
    assert_eq!(page.completed(), 0);
    assert_eq!(page.submitted(), 0);
    assert!(!page.now_pending());
}

#[test]
fn loopback_tcp_wake_delivers_one_byte_to_a_real_listener() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind wake listener");
    let port = listener.local_addr().expect("wake address").port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept wake");
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).expect("read wake byte");
        byte[0]
    });

    LoopbackTcpWake::new(port).wake().expect("wake connects");
    assert_eq!(handle.join().expect("wake thread"), 1);
}

#[cfg(unix)]
#[test]
fn unix_socket_wake_delivers_one_byte_to_a_real_listener() {
    use std::os::unix::net::UnixListener;

    let dir = TestDir::new();
    let path = dir.0.join("wake.sock");
    let listener = UnixListener::bind(&path).expect("bind unix wake listener");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept unix wake");
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).expect("read unix wake byte");
        byte[0]
    });

    UnixSocketWake::new(&path)
        .wake()
        .expect("unix wake connects");
    assert_eq!(handle.join().expect("unix wake thread"), 1);
}

#[cfg(not(unix))]
#[test]
fn unix_socket_wake_reports_unsupported() {
    let error = UnixSocketWake::new("/tmp/victory-wake.sock")
        .wake()
        .expect_err("non-unix wake must fail explicitly");
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn guest_transport_is_implemented_without_wine_symbols() {
    fn assert_transport<T: GuestTransport>() {}
    assert_transport::<LoopbackAsyncTransport>();
}

// ---------------------------------------------------------------------------
// Real IPC: async submit, single-load completion, no reply frame
// ---------------------------------------------------------------------------

#[test]
fn async_submit_publishes_completion_and_failures_stay_pending() {
    let dir = TestDir::new();
    let executor = StubExecutor::new();
    // Sequence 3 models a device-side execution failure.
    executor.set_fail_sequence(3);
    let host = start_host(&dir, executor.clone());

    // The launcher owns the page file; the guest maps it before the handshake.
    let page = FencePageMapping::create(dir.fence_page_path()).expect("create fence page");

    // A separate listener proves the wake channel is really rung.
    let wake_listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind wake listener");
    let wake_addr = wake_listener.local_addr().expect("wake address");
    let wake_thread = thread::spawn(move || {
        let (mut stream, _) = wake_listener.accept().expect("accept wake");
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).expect("read wake byte");
        byte[0]
    });

    let guest = LoopbackAsyncTransport::connect(
        host.addr,
        SECRET,
        page,
        Box::new(LoopbackTcpWake::with_timeout(
            wake_addr,
            Duration::from_secs(2),
        )),
    )
    .expect("guest handshake");

    // A second descriptor on the same socket detects any stray reply frame.
    let mut probe = guest.stream().try_clone().expect("clone guest socket");
    probe
        .set_read_timeout(Some(Duration::from_millis(200)))
        .expect("probe read timeout");

    let identity = ListIdentity {
        list_id: 7,
        list_generation: 1,
    };
    let frame = encode_submission_v3(&submission(1, Some(identity))).expect("encode first frame");
    let submitted = guest.submit_async(&frame).expect("async submit");
    assert_eq!(submitted, 1, "fence value is the submission sequence");
    assert_eq!(wake_thread.join().expect("wake thread"), 1);

    wait_until(|| guest.completed_fence() == 1, "first completion");
    assert_eq!(guest.fence_page().completed(), 1);
    assert_eq!(guest.fence_page().submitted(), 1);
    assert!(!guest.fence_page().now_pending());

    // A replay of the same list is refused with no GPU work and no publish.
    let replay = encode_submission_v3(&submission(2, Some(identity))).expect("encode replay");
    assert_eq!(guest.submit_async(&replay).expect("async replay"), 2);

    // Sequence 3 reaches the executor and fails there. Once the stub reports
    // the attempt, the host has processed both the replay and the failure.
    let failing = encode_submission_v3(&submission(3, None)).expect("encode failing frame");
    assert_eq!(guest.submit_async(&failing).expect("async failing"), 3);
    wait_until(|| executor.attempted().contains(&3), "sequence 3 attempt");
    assert_eq!(
        guest.fence_page().completed(),
        1,
        "a replay or executor failure advanced the completion fence"
    );

    // The next successful submission publishes its own sequence.
    let later = encode_submission_v3(&submission(4, None)).expect("encode later frame");
    assert_eq!(guest.submit_async(&later).expect("async later"), 4);
    wait_until(|| guest.completed_fence() == 4, "sequence 4 completion");
    assert_eq!(guest.fence_page().submitted(), 4);
    assert!(!guest.fence_page().now_pending());

    // No reply frame was ever produced for a kind 7 submission: the fence value
    // came from the shared page, not from a response.
    let mut byte = [0u8; 1];
    match probe.read(&mut byte) {
        Err(error) => assert!(
            matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
            "unexpected probe error: {error}"
        ),
        Ok(0) => panic!("host closed the session before teardown"),
        Ok(read) => panic!("host sent {read} unexpected reply byte(s)"),
    }

    assert_eq!(executor.completed(), vec![1, 4]);
    assert_eq!(executor.attempted(), vec![1, 3, 4]);

    drop(probe);
    drop(guest);
    let (result, _host) = host.handle.join().expect("host thread");
    let summary = result.expect("session succeeds");
    assert_eq!(
        summary,
        SessionSummary {
            submissions: 4,
            allocations: 0,
            bytes_mapped: 0,
        }
    );
}

#[test]
fn async_submit_rejects_a_non_increasing_sequence_locally() {
    let dir = TestDir::new();
    let host = start_host(&dir, StubExecutor::new());
    let page = FencePageMapping::create(dir.fence_page_path()).expect("create fence page");
    let guest = LoopbackAsyncTransport::connect(
        host.addr,
        SECRET,
        page,
        Box::new(UnixSocketWake::new(dir.0.join("unused-wake.sock"))),
    )
    .expect("guest handshake");

    let first = encode_submission_v3(&submission(1, None)).expect("encode first frame");
    guest.submit_async(&first).expect("first submit");
    wait_until(|| guest.completed_fence() == 1, "first completion");

    // A repeated sequence would desynchronize the host; reject it before the
    // wire so the session stays usable.
    let duplicate = encode_submission_v3(&submission(1, None)).expect("encode duplicate frame");
    let error = guest
        .submit_async(&duplicate)
        .expect_err("duplicate sequence must be refused");
    assert!(matches!(error, MetalError::InvalidArg(_)));

    let next = encode_submission_v3(&submission(2, None)).expect("encode next frame");
    assert_eq!(guest.submit_async(&next).expect("next submit"), 2);
    wait_until(|| guest.completed_fence() == 2, "second completion");

    drop(guest);
    host.handle
        .join()
        .expect("host thread")
        .0
        .expect("session succeeds");
}
