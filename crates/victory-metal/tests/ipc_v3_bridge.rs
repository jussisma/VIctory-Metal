//! Integration tests for the version 3 host bridge in [`victory_metal::ipc_host`].
//!
//! The tests drive the real loopback session with a CPU-only executor, so the
//! envelope, allocation resolution, list ledger and reply framing are all
//! exercised without a Metal device.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use victory_metal::abi::VmBufferId;
use victory_metal::ipc_host::{
    BatchExecutor, HostError, HostSecret, IpcHost, SessionSummary, KIND_ACK, KIND_AUTH,
    KIND_REGISTER, KIND_SUBMISSION, KIND_SUBMISSION_V3, MAX_SUBMISSION_V3_LEN,
    REGISTER_PAYLOAD_LEN, V3_REQUEST_HEADER_LEN, V3_REQUEST_MAGIC,
};
use victory_metal::transport::MetalError;
use victory_metal::wire::{
    self, BufferBinding, Command, Dispatch1d, ListIdentity, Reply, ReplyStatus, Submission,
};
use victory_metal::wire_v3::{self, CommandV3, SubmissionV3, PROTOCOL_VERSION_V3};

const SECRET: [u8; 32] = [0x5a; 32];
const ALLOCATION_ID: u64 = 1;
const BYTES: usize = 4096;
const PIPELINE_ID: u64 = 7;

#[derive(Default)]
struct FakeState {
    next_buffer: u64,
    v2_submissions: Vec<Submission>,
    v3_submissions: Vec<SubmissionV3>,
    v3_executions: u64,
}

struct FakeExecutor {
    state: Arc<Mutex<FakeState>>,
}

impl FakeExecutor {
    fn state(&self) -> MutexGuard<'_, FakeState> {
        self.state.lock().expect("fake state mutex")
    }
}

impl BatchExecutor for FakeExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        Ok(4096)
    }

    fn register_window(&self, _ptr: *mut u8, _len: usize) -> Result<VmBufferId, MetalError> {
        let mut state = self.state();
        state.next_buffer += 1;
        Ok(VmBufferId(state.next_buffer))
    }

    fn unregister_window(&self, _id: VmBufferId) -> Result<(), MetalError> {
        Ok(())
    }

    fn submit_batch(
        &self,
        submission: &Submission,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let mut state = self.state();
        state.v2_submissions.push(submission.clone());
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let mut state = self.state();
        state.v3_submissions.push(submission.clone());
        state.v3_executions += 1;
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-ipc-v3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create test dir");
        Self(path)
    }

    fn write_allocation(&self, id: u64, bytes: &[u8]) {
        std::fs::write(self.0.join(format!("{id:016x}.bin")), bytes)
            .expect("write allocation file");
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Server {
    addr: SocketAddr,
    handle: thread::JoinHandle<(Result<SessionSummary, HostError>, IpcHost<FakeExecutor>)>,
}

fn host(dir: &Path, state: Arc<Mutex<FakeState>>) -> IpcHost<FakeExecutor> {
    IpcHost::new(HostSecret::new(SECRET), dir, FakeExecutor { state }).expect("create host")
}

fn start(mut host: IpcHost<FakeExecutor>) -> Server {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept test peer");
        let result = host.serve_once(stream);
        (result, host)
    });
    Server { addr, handle }
}

fn connect(server: &Server) -> TcpStream {
    let stream = TcpStream::connect(server.addr).expect("connect test peer");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("client read timeout");
    stream
}

fn read_frame(stream: &mut TcpStream) -> Result<Option<(u8, Vec<u8>)>, HostError> {
    let mut header = [0u8; 5];
    let mut filled = 0usize;
    while filled < header.len() {
        match stream.read(&mut header[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err(HostError::Protocol("closed inside envelope")),
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok(Some((header[0], payload)))
}

fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) -> Result<(), HostError> {
    let mut header = [0u8; 5];
    header[0] = kind;
    header[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

fn authenticate(client: &mut TcpStream) {
    write_frame(client, KIND_AUTH, &SECRET).expect("send auth");
    assert_eq!(
        read_frame(client).expect("read auth ack"),
        Some((KIND_ACK, vec![0]))
    );
}

fn register(client: &mut TcpStream, id: u64, len: u64) {
    let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
    payload[..8].copy_from_slice(&id.to_le_bytes());
    payload[8..].copy_from_slice(&len.to_le_bytes());
    write_frame(client, KIND_REGISTER, &payload).expect("send register");
    assert_eq!(
        read_frame(client).expect("read register ack"),
        Some((KIND_ACK, vec![0]))
    );
}

fn v3_envelope(submission: &SubmissionV3) -> Vec<u8> {
    let frame = wire_v3::encode_submission_v3(submission).expect("encode v3 submission");
    let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
    payload.extend_from_slice(&V3_REQUEST_MAGIC);
    payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
    payload.extend_from_slice(&submission.sequence.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes());
    payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    payload.extend_from_slice(&frame);
    payload
}

fn v3_submission(
    sequence: u32,
    list: Option<ListIdentity>,
    binding: BufferBinding,
) -> SubmissionV3 {
    SubmissionV3 {
        sequence,
        list,
        commands: vec![CommandV3::Dispatch1d(Dispatch1d {
            pipeline_id: PIPELINE_ID,
            threads: (BYTES / 4) as u32,
            threads_per_group: 64,
            bindings: vec![binding],
        })],
    }
}

fn v2_submission(sequence: u32, binding: BufferBinding) -> Submission {
    Submission {
        sequence,
        commands: vec![Command::Dispatch1d(Dispatch1d {
            pipeline_id: PIPELINE_ID,
            threads: (BYTES / 4) as u32,
            threads_per_group: 64,
            bindings: vec![binding],
        })],
    }
}

fn full_binding() -> BufferBinding {
    BufferBinding {
        allocation_id: ALLOCATION_ID,
        generation: 1,
        offset: 0,
        length: BYTES as u64,
    }
}

fn read_v3_reply(client: &mut TcpStream) -> Reply {
    let (kind, payload) = read_frame(client)
        .expect("read v3 reply")
        .expect("v3 reply frame");
    assert_eq!(kind, KIND_SUBMISSION_V3);
    wire::decode_reply(&payload).expect("decode v3 reply")
}

fn roundtrip_v3(client: &mut TcpStream, submission: &SubmissionV3) -> Reply {
    write_frame(client, KIND_SUBMISSION_V3, &v3_envelope(submission)).expect("send v3 submission");
    read_v3_reply(client)
}

#[test]
fn v3_executes_once_and_refuses_replay() {
    let dir = TestDir::new();
    dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
    let state = Arc::new(Mutex::new(FakeState::default()));
    let server = start(host(&dir.0, Arc::clone(&state)));
    let mut client = connect(&server);
    authenticate(&mut client);
    register(&mut client, ALLOCATION_ID, BYTES as u64);

    let identity = ListIdentity {
        list_id: 31,
        list_generation: 1,
    };
    let first = v3_submission(1, Some(identity), full_binding());
    assert_eq!(
        roundtrip_v3(&mut client, &first),
        Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        }
    );

    let replay = v3_submission(2, Some(identity), full_binding());
    assert_eq!(
        roundtrip_v3(&mut client, &replay),
        Reply {
            sequence: 2,
            status: ReplyStatus::AlreadyExecuted,
        }
    );

    drop(client);
    let (result, _host) = server.handle.join().expect("server thread");
    assert_eq!(
        result.expect("session succeeds"),
        SessionSummary {
            submissions: 2,
            allocations: 1,
            bytes_mapped: BYTES as u64,
        }
    );
    let state = state.lock().expect("fake state mutex");
    assert_eq!(
        state.v3_submissions,
        vec![first],
        "replay reached the executor"
    );
    assert_eq!(state.v3_executions, 1, "replay executed GPU work");
}

#[test]
fn v3_unknown_allocation_consumes_no_identity() {
    let dir = TestDir::new();
    dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
    let state = Arc::new(Mutex::new(FakeState::default()));
    let server = start(host(&dir.0, Arc::clone(&state)));
    let mut client = connect(&server);
    authenticate(&mut client);
    register(&mut client, ALLOCATION_ID, BYTES as u64);

    let identity = ListIdentity {
        list_id: 32,
        list_generation: 1,
    };
    let unknown = v3_submission(
        1,
        Some(identity),
        BufferBinding {
            allocation_id: ALLOCATION_ID + 1,
            generation: 1,
            offset: 0,
            length: BYTES as u64,
        },
    );
    assert_eq!(
        roundtrip_v3(&mut client, &unknown),
        Reply {
            sequence: 1,
            status: ReplyStatus::UnknownHandle,
        }
    );
    assert!(state.lock().expect("state").v3_submissions.is_empty());

    let corrected = v3_submission(2, Some(identity), full_binding());
    assert_eq!(
        roundtrip_v3(&mut client, &corrected),
        Reply {
            sequence: 2,
            status: ReplyStatus::Ok,
        }
    );

    drop(client);
    let (result, _host) = server.handle.join().expect("server thread");
    result.expect("session succeeds");
    let state = state.lock().expect("state");
    assert_eq!(state.v3_submissions, vec![corrected]);
    assert_eq!(state.v3_executions, 1);
}

#[test]
fn v3_malformed_frames_are_rejected_without_closing_a_recoverable_session() {
    let dir = TestDir::new();
    dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
    let state = Arc::new(Mutex::new(FakeState::default()));
    let server = start(host(&dir.0, Arc::clone(&state)));
    let mut client = connect(&server);
    authenticate(&mut client);
    register(&mut client, ALLOCATION_ID, BYTES as u64);

    let submission = v3_submission(1, None, full_binding());
    let good = v3_envelope(&submission);

    let mut wrong_version = good.clone();
    wrong_version[4..6].copy_from_slice(&(PROTOCOL_VERSION_V3 + 1).to_le_bytes());

    let truncated = good[..good.len() - 1].to_vec();

    let mut trailing = good.clone();
    trailing.push(0);

    let mut oversized = good.clone();
    oversized[12..16].copy_from_slice(&((MAX_SUBMISSION_V3_LEN as u32) + 1).to_le_bytes());

    for (label, frame) in [
        ("unknown version", wrong_version),
        ("truncated", truncated),
        ("trailing bytes", trailing),
        ("oversized", oversized),
    ] {
        write_frame(&mut client, KIND_SUBMISSION_V3, &frame).expect("send malformed frame");
        assert_eq!(
            read_v3_reply(&mut client),
            Reply {
                sequence: 1,
                status: ReplyStatus::ExecutionFailed,
            },
            "{label}"
        );
        assert!(
            state.lock().expect("state").v3_submissions.is_empty(),
            "{label} reached the executor"
        );
    }

    // No refusal advanced the session sequence, so an exact retry is accepted.
    assert_eq!(
        roundtrip_v3(&mut client, &submission),
        Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        }
    );

    drop(client);
    let (result, _host) = server.handle.join().expect("server thread");
    result.expect("session succeeds");
    assert_eq!(state.lock().expect("state").v3_executions, 1);
}

#[test]
fn v3_unsequenced_frame_is_answered_then_closed() {
    let dir = TestDir::new();
    let state = Arc::new(Mutex::new(FakeState::default()));
    let server = start(host(&dir.0, Arc::clone(&state)));
    let mut client = connect(&server);
    authenticate(&mut client);

    write_frame(&mut client, KIND_SUBMISSION_V3, &[0u8; 8]).expect("send short frame");
    assert_eq!(
        read_v3_reply(&mut client),
        Reply {
            sequence: 1,
            status: ReplyStatus::ExecutionFailed,
        }
    );
    assert_eq!(read_frame(&mut client).expect("read close"), None);

    let (result, _host) = server.handle.join().expect("server thread");
    assert!(matches!(result, Err(HostError::Protocol(_))));
}

#[test]
fn v2_and_v3_frames_share_one_session_and_sequence() {
    let dir = TestDir::new();
    dir.write_allocation(ALLOCATION_ID, &vec![0u8; BYTES]);
    let state = Arc::new(Mutex::new(FakeState::default()));
    let server = start(host(&dir.0, Arc::clone(&state)));
    let mut client = connect(&server);
    authenticate(&mut client);
    register(&mut client, ALLOCATION_ID, BYTES as u64);

    let v2 = v2_submission(1, full_binding());
    write_frame(
        &mut client,
        KIND_SUBMISSION,
        &wire::encode_submission(&v2).expect("encode v2 submission"),
    )
    .expect("send v2 submission");
    let (kind, payload) = read_frame(&mut client)
        .expect("read v2 reply")
        .expect("v2 reply frame");
    assert_eq!(kind, KIND_SUBMISSION);
    assert_eq!(
        wire::decode_reply(&payload).expect("decode v2 reply"),
        Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        }
    );

    let identity = ListIdentity {
        list_id: 33,
        list_generation: 1,
    };
    let v3 = v3_submission(2, Some(identity), full_binding());
    assert_eq!(
        roundtrip_v3(&mut client, &v3),
        Reply {
            sequence: 2,
            status: ReplyStatus::Ok,
        }
    );

    drop(client);
    let (result, _host) = server.handle.join().expect("server thread");
    result.expect("session succeeds");
    let state = state.lock().expect("state");
    assert_eq!(state.v2_submissions, vec![v2]);
    assert_eq!(state.v3_submissions, vec![v3]);
}
