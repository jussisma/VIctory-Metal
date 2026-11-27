//! Live title command-list seam (ADR-019), driven through the production
//! `victory-gpu` D3D12 entry points under `VICTORY_BACKEND=metal`.
//!
//! The test creates the real device (which auto-connects the Wire v3 session
//! and installs the native presenter), a queue, an allocator, a command list
//! and a fence through the device's own vtable, then:
//!
//! 1. creates a committed buffer whose range the installed resolver maps to the
//!    one share allocation; the creation registers it with the host (kind 2,
//!    `id + whole share-file length`) and the host sees exactly one such frame;
//! 2. closes a list with a barrier on that buffer and runs
//!    `ExecuteCommandLists`; the host must receive exactly one ordered
//!    submission for that list generation, and a replay sends nothing;
//! 3. binds the same buffer twice (`IASetVertexBuffers`); the second bind sends
//!    no registration (idempotent by allocation);
//! 4. binds a buffer whose range resolves to an allocation with no share file:
//!    the list refuses it explicitly, `Close` returns a failure HRESULT and
//!    nothing is submitted (no silent skip, no D3DMetal fallback);
//! 5. registers a PSO: kind 4 carries the session ticket and entry name, kind
//!    129 returns the host's native pipeline id, and a producer submission
//!    names that id in `SetComputePipeline` and the registered allocation in
//!    `SetVertexBuffers`; a graphics PSO then sends the version 2 frame with a
//!    distinct ticket (its own metallib file) per stage, and the returned id is
//!    the one `SetPipelineState` records in `SetGraphicsPipeline`;
//! 6. prints which path served the pool translation (a worker, the disk cache,
//!    or the explicit "converter unavailable" refusal of a Win64 guest);
//! 7. `Signal`s the fence, reads `GetCompletedValue`, and presents one
//!    `CommandV3::Present` through the installed `WirePresenter`.
//!
//! The endpoint is an in-process loopback V3 host fixture, labelled here as
//! such: the 5B live gate repeats the same chain over the real host socket.
//! This test is Windows-only because the D3D12 entry points and their
//! descriptors exist only in the Win64 guest build; `cargo check --target
//! x86_64-pc-windows-gnullvm --all-targets` compiles it, and the gate runs
//! it. The transport halves of kinds 2 and 4/129 run on every target in
//! `native_metal::client`'s unit tests.

#![cfg(windows)]

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::ptr::null;
use std::sync::{Arc, Mutex, mpsc::{self, Receiver}};
use std::thread::JoinHandle;
use std::time::Duration;

use victory_gpu::device;
use victory_gpu::native_metal::bindings::{self, GuestRange, GuestRangeResolver, ResolveError};
use victory_gpu::native_metal::client::{
    PIPELINE_FRAME_MAGIC, PIPELINE_KIND_GRAPHICS, PIPELINE_KIND_MESH,
};
use victory_gpu::native_metal::{pso, queue, telemetry};
use victory_gpu::object::{Object, S_OK};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{decode_submission_v3, CommandV3, SubmissionV3};

const SECRET_HEX: &str = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";
/// Minimal converter reflection JSON per stage (only TopLevelArgumentBuffer is
/// required by the host's `RootSignatureLayout` parser).
const VERTEX_REFLECTION: &str =
    r#"{"EntryPoint":"main","ShaderType":"Vertex","TopLevelArgumentBuffer":[]}"#;
const PIXEL_REFLECTION: &str =
    r#"{"EntryPoint":"main","ShaderType":"Fragment","TopLevelArgumentBuffer":[]}"#;
const MESH_REFLECTION: &str =
    r#"{"EntryPoint":"main","ShaderType":"Mesh","TopLevelArgumentBuffer":[]}"#;
const TIMEOUT: Duration = Duration::from_secs(10);
/// Allocation id the fixture's share file backs.
const SHARE_ID: u64 = 1;
/// Its file length, which is exactly what a kind 2 frame must carry.
const SHARE_LEN: u64 = 64 * 1024;
/// Bytes one share page holds.
const PAGE_LEN: u64 = 4096;
/// `ID3D12Device` creation parameter version the layouts were checked against.
const CHECKED_VERSION: u32 = 0x0002_0004;
/// The device-creation interface id.
const DEVICE_IID: [u8; 16] = guid(
    0x189819f1,
    0x1db6,
    0x4b57,
    [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
);
const QUEUE_IID: [u8; 16] = guid(
    0x0ec870a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);
const ALLOCATOR_IID: [u8; 16] = guid(
    0x6102dee4,
    0xaf59,
    0x4b09,
    [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
);
const LIST_IID: [u8; 16] = guid(
    0x7116d91c,
    0xe7e4,
    0x47ce,
    [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
);
const FENCE_IID: [u8; 16] = guid(
    0x0a753dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);

const fn guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> [u8; 16] {
    let (a, b, c) = (a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes());
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6],
        d[7],
    ]
}

type This = *const Object;
type Hresult = i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

#[repr(C)]
struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

const RESOURCE_IID: [u8; 16] = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

/// `crate::draw::VertexView` as the title passes it (private there).
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalVertexView {
    address: u64,
    size: u32,
    stride: u32,
}

/// A D3D12 buffer barrier on the buffer at `resource`.
#[repr(C)]
struct Barrier {
    kind: u32,
    flags: u32,
    resource: This,
    second: u64,
    last: u64,
}

/// Calls one slot of an object's own function table.
///
/// SAFETY: `object` must be an interface the guest handed out, and `F` the
/// exact signature of that slot (the same shape the title calls).
unsafe fn slot<F: Copy>(object: This, offset: usize) -> F {
    unsafe {
        let table = *(object.cast::<*const *const ()>());
        let at = table.add(offset / 8);
        std::mem::transmute_copy(&*at)
    }
}

type CreateQueue =
    unsafe extern "system" fn(This, *const QueueDesc, *const [u8; 16], *mut This) -> Hresult;
type CreateAllocator =
    unsafe extern "system" fn(This, u32, *const [u8; 16], *mut This) -> Hresult;
type CreateList = unsafe extern "system" fn(
    This,
    u32,
    u32,
    This,
    This,
    *const [u8; 16],
    *mut This,
) -> Hresult;
type CreateFence =
    unsafe extern "system" fn(This, u64, u32, *const [u8; 16], *mut This) -> Hresult;
type CreateCommitted = unsafe extern "system" fn(
    This,
    *const HeapProperties,
    u32,
    *const ResourceDesc,
    u32,
    *const std::ffi::c_void,
    *const [u8; 16],
    *mut This,
) -> Hresult;
type BarrierCall = unsafe extern "system" fn(This, u32, *const Barrier) -> u32;
type Close = unsafe extern "system" fn(This) -> Hresult;
type SetVertexBuffers =
    unsafe extern "system" fn(This, u32, u32, *const MetalVertexView) -> ();
type Execute = unsafe extern "system" fn(This, u32, *const This);
type Signal = unsafe extern "system" fn(This, This, u64) -> Hresult;
type Completed = unsafe extern "system" fn(This) -> u64;
type Present = unsafe extern "system" fn(This, u32, *const u8, *const u8) -> Hresult;

/// What the loopback fixture saw, in arrival order.
#[derive(Debug)]
enum Seen {
    /// One version 3 submission.
    Submission(SubmissionV3),
    /// One kind 2 allocation registration: `(id, byte_len)`.
    Buffer(u64, u64),
    /// One kind 4 v1 pipeline registration: `(ticket, entry)`.
    Pipeline(u64, String),
    /// One kind 4 v2 registration: kind, `(stage, ticket, entry)` list, and
    /// the exact state/ABI/root-signature bytes.
    PipelineV2 {
        kind: u8,
        stages: Vec<(u8, u64, String)>,
        state: Vec<u8>,
        abi: Vec<u8>,
        root_signature: Vec<u8>,
    },
}

/// A loopback V3 host fixture: authenticates one peer, acknowledges surface,
/// fence and allocation registrations, answers pipeline registrations with a
/// fresh native id, and answers every submission with `Ok`, reporting each
/// frame it saw.
struct TestHost {
    port: u16,
    handle: Option<JoinHandle<()>>,
}

impl TestHost {
    fn spawn() -> (Self, Receiver<Seen>) {
        let listener =
            TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).expect("bind loopback");
        let port = listener.local_addr().expect("address").port();
        let (sender, receiver) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept the guest");
            stream.set_read_timeout(Some(TIMEOUT)).expect("read timeout");
            stream.set_write_timeout(Some(TIMEOUT)).expect("write timeout");
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, 1, "the first frame must authenticate");
            assert_eq!(payload.len(), 32);
            write_frame(&mut stream, 128, &[0]);

            let mut next_pipeline = 0x51u64;
            while let Some((kind, payload)) = try_frame(&mut stream) {
                match kind {
                    // KIND_FENCE_PAGE
                    6 => write_frame(&mut stream, 128, &[0]),
                    // KIND_TEXTURE_REG / KIND_SURFACE_REG
                    8 | 9 => write_frame(&mut stream, 128, &[0]),
                    // KIND_REGISTER: authentication closes on a refusal, so
                    // the length is validated here before the ack.
                    2 => {
                        assert_eq!(payload.len(), 16, "a kind 2 payload");
                        let id =
                            u64::from_le_bytes(payload[0..8].try_into().expect("eight bytes"));
                        let len =
                            u64::from_le_bytes(payload[8..16].try_into().expect("eight bytes"));
                        sender.send(Seen::Buffer(id, len)).expect("report the registration");
                        write_frame(&mut stream, 128, &[0]);
                    }
                    // KIND_PIPELINE: v1 `ticket, entry_len, entry`, or the
                    // v2 frame selected by its magic word.
                    4 => {
                        if payload.len() >= 4 && payload[..4] == PIPELINE_FRAME_MAGIC {
                            let kind = payload[5];
                            let stage_count = usize::from(payload[6]);
                            let state_len = u32::from_le_bytes(
                                payload[8..12].try_into().expect("four bytes"),
                            ) as usize;
                            let abi_len = u32::from_le_bytes(
                                payload[12..16].try_into().expect("four bytes"),
                            ) as usize;
                            let root_len = u32::from_le_bytes(
                                payload[16..20].try_into().expect("four bytes"),
                            ) as usize;
                            let mut offset = 24;
                            let mut stages = Vec::with_capacity(stage_count);
                            for _ in 0..stage_count {
                                let stage = payload[offset];
                                offset += 1;
                                let ticket = u64::from_le_bytes(
                                    payload[offset..offset + 8]
                                        .try_into()
                                        .expect("eight bytes"),
                                );
                                offset += 8;
                                let entry_len = u16::from_le_bytes(
                                    payload[offset..offset + 2]
                                        .try_into()
                                        .expect("two bytes"),
                                ) as usize;
                                offset += 2;
                                let entry = String::from_utf8(
                                    payload[offset..offset + entry_len].to_vec(),
                                )
                                .expect("an ASCII entry name");
                                offset += entry_len;
                                stages.push((stage, ticket, entry));
                            }
                            let state = payload[offset..offset + state_len].to_vec();
                            offset += state_len;
                            let abi = payload[offset..offset + abi_len].to_vec();
                            offset += abi_len;
                            let root_signature = payload[offset..offset + root_len].to_vec();
                            offset += root_len;
                            assert_eq!(offset, payload.len(), "no trailing bytes");
                            sender
                                .send(Seen::PipelineV2 {
                                    kind,
                                    stages,
                                    state,
                                    abi,
                                    root_signature,
                                })
                                .expect("report the registration");
                        } else {
                            let ticket = u64::from_le_bytes(
                                payload[0..8].try_into().expect("eight bytes"),
                            );
                            let entry_len =
                                u16::from_le_bytes(payload[8..10].try_into().expect("two bytes"))
                                    as usize;
                            assert_eq!(payload.len(), 10 + entry_len);
                            let entry = String::from_utf8(payload[10..].to_vec())
                                .expect("an ASCII entry name");
                            sender
                                .send(Seen::Pipeline(ticket, entry))
                                .expect("report the registration");
                        }
                        let mut reply = vec![0u8];
                        reply.extend_from_slice(&next_pipeline.to_le_bytes());
                        next_pipeline += 1;
                        write_frame(&mut stream, 129, &reply);
                    }
                    // KIND_SUBMISSION_V3
                    5 => {
                        let submission =
                            decode_submission_v3(&payload[16..]).expect("a canonical v3 frame");
                        let sequence = u32::from_le_bytes([
                            payload[6], payload[7], payload[8], payload[9],
                        ]);
                        sender
                            .send(Seen::Submission(submission))
                            .expect("report the submission");
                        let reply =
                            victory_metal::wire::encode_reply(&victory_metal::wire::Reply {
                                sequence,
                                status: ReplyStatus::Ok,
                            })
                            .expect("reply encodes");
                        write_frame(&mut stream, 5, &reply);
                    }
                    other => panic!("unexpected frame kind {other}"),
                }
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
}

impl Drop for TestHost {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_frame(stream: &mut TcpStream) -> (u8, Vec<u8>) {
    let mut header = [0u8; 5];
    stream.read_exact(&mut header).expect("frame header");
    let kind = header[0];
    let length = u32::from_le_bytes(header[1..].try_into().expect("four bytes")) as usize;
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload).expect("frame payload");
    (kind, payload)
}

fn try_frame(stream: &mut TcpStream) -> Option<(u8, Vec<u8>)> {
    let mut header = [0u8; 5];
    let mut filled = 0;
    while filled < header.len() {
        match stream.read(&mut header[filled..]) {
            Ok(0) if filled == 0 => return None,
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    let length = u32::from_le_bytes(header[1..].try_into().expect("four bytes")) as usize;
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload).ok()?;
    Some((header[0], payload))
}

fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) {
    stream.write_all(&[kind]).expect("kind");
    stream
        .write_all(&(payload.len() as u32).to_le_bytes())
        .expect("length");
    stream.write_all(payload).expect("payload");
    stream.flush().expect("flush");
}

/// A resolver the test drives: a list of `(base, size, allocation)` rules,
/// matched first to last, over pinned ranges it counts.
struct TestResolver {
    rules: Arc<Mutex<Vec<(u64, u64, u64)>>>,
    retained: Arc<Mutex<Vec<u64>>>,
}

impl TestResolver {
    fn new(rules: &[(u64, u64, u64)]) -> (Self, Arc<Mutex<Vec<(u64, u64, u64)>>>) {
        let rules = Arc::new(Mutex::new(rules.to_vec()));
        (
            Self {
                rules: rules.clone(),
                retained: Arc::new(Mutex::new(Vec::new())),
            },
            rules,
        )
    }
}

impl GuestRangeResolver for TestResolver {
    fn resolve(&self, address: u64, len: u64) -> Result<GuestRange, ResolveError> {
        if address == 0 || len == 0 {
            return Err(ResolveError::Invalid("zero address or length"));
        }
        let rules = self.rules.lock().unwrap_or_else(|e| e.into_inner());
        for (base, size, allocation) in rules.iter() {
            let end = base.saturating_add(*size);
            if address >= *base && address.saturating_add(len) <= end {
                return Ok(GuestRange {
                    allocation_id: *allocation,
                    generation: 1,
                    byte_offset: 0,
                    size: SHARE_LEN,
                });
            }
        }
        Err(ResolveError::Unmapped { address, len })
    }

    fn retain(&self, range: &GuestRange) -> Result<u64, ResolveError> {
        let mut retained = self.retained.lock().unwrap_or_else(|e| e.into_inner());
        let token = retained.len() as u64 + 1;
        retained.push(range.allocation_id);
        Ok(token)
    }

    fn retire(&self, _token: u64) -> Result<(), ResolveError> {
        Ok(())
    }
}

/// The real device/queue/list/fence/present chain against the fixture host.
#[test]
fn a_recorded_title_list_reaches_the_host_exactly_once() {
    let (host, seen) = TestHost::spawn();
    // The launcher's share directory: one host-visible allocation file with the
    // whole length a kind 2 registration must carry.
    let share = tempfile::tempdir().expect("a share directory");
    let share_file = share.path().join(format!("{SHARE_ID:016x}.bin"));
    std::fs::File::create(&share_file)
        .and_then(|file| file.set_len(SHARE_LEN))
        .expect("a share file of the allocation's exact length");

    // SAFETY: this test owns the process environment (one test per binary).
    unsafe {
        std::env::set_var("VICTORY_BACKEND", "metal");
        std::env::set_var("VICTORY_METAL_ENDPOINT", host.endpoint());
        std::env::set_var("VICTORY_METAL_SECRET", SECRET_HEX);
        // The installed presenter's surface (the host fixture accepts it).
        std::env::set_var("VICTORY_METAL_SURFACE", "7");
        std::env::set_var("VICTORY_METAL_SHARE_DIR", share.path());
    }
    // Every address resolves to the one shared allocation to begin with; the
    // refusal case rewrites the rules to an allocation with no share file.
    let (resolver, rules) = TestResolver::new(&[(0, u64::MAX, SHARE_ID)]);
    bindings::set_resolver(resolver);

    // Device creation auto-connects the session and installs the presenter.
    let version = CHECKED_VERSION;
    let mut device_ptr: This = null();
    let hr = device::create(null(), &version, &DEVICE_IID, &mut device_ptr);
    assert_eq!(hr, S_OK, "device creation connects the native session");
    assert!(!device_ptr.is_null());

    // Queue, allocator, list and fence through the device's own table.
    let queue: CreateQueue = unsafe { slot(device_ptr, 0x40) };
    let allocator_create: CreateAllocator = unsafe { slot(device_ptr, 0x48) };
    let list_create: CreateList = unsafe { slot(device_ptr, 0x60) };
    let fence_create: CreateFence = unsafe { slot(device_ptr, 0x120) };
    let desc = QueueDesc {
        kind: 0,
        priority: 0,
        flags: 0,
        nodes: 1,
    };
    let mut queue_ptr: This = null();
    assert_eq!(
        unsafe { queue(device_ptr, &desc, &QUEUE_IID, &mut queue_ptr) },
        S_OK
    );
    let mut allocator: This = null();
    assert_eq!(
        unsafe { allocator_create(device_ptr, 0, &ALLOCATOR_IID, &mut allocator) },
        S_OK
    );
    let mut list: This = null();
    let list_desc = ListDesc {
        kind: 0,
        flags: 0,
        nodes: 1,
        barrier_batch: 0,
    };
    assert_eq!(
        unsafe {
            list_create(
                device_ptr,
                0,
                list_desc.kind,
                allocator,
                null(),
                &LIST_IID,
                &mut list,
            )
        },
        S_OK
    );
    let mut fence: This = null();
    assert_eq!(
        unsafe { fence_create(device_ptr, 0, 0, &FENCE_IID, &mut fence) },
        S_OK
    );

    // A committed buffer. Creating it registers its share allocation with the
    // host (kind 2) because the resolver maps its range to the one share file
    // this test created.
    let create_resource: CreateCommitted = unsafe { slot(device_ptr, 0xd8) };
    let heap = HeapProperties {
        kind: 0,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    let desc = ResourceDesc {
        dimension: 1, // D3D12_RESOURCE_DIMENSION_BUFFER
        alignment: 0,
        width: 4096,
        height: 1,
        depth_or_array: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 0,
        flags: 0,
    };
    let mut buffer: This = null();
    assert_eq!(
        unsafe {
            create_resource(
                device_ptr,
                &heap,
                0,
                &desc,
                0,
                null(),
                &RESOURCE_IID,
                &mut buffer,
            )
        },
        S_OK,
        "a committed buffer is created through the real device"
    );
    match seen.recv_timeout(TIMEOUT).expect("the kind 2 registration") {
        Seen::Buffer(id, len) => assert_eq!(
            (id, len),
            (SHARE_ID, SHARE_LEN),
            "the registration carries the allocation id and the whole share file"
        ),
        other => panic!("expected a buffer registration, saw {other:?}"),
    }
    // The buffer's GPU virtual address is what the resolver and the wire
    // bindings name; read it through the resource's own table.
    let gpu_address: unsafe extern "system" fn(This) -> u64 = unsafe { slot(buffer, 0x58) };
    let address = unsafe { gpu_address(buffer) };
    assert_ne!(address, 0, "a committed buffer has a GPU virtual address");

    // A UAV barrier on the registered buffer: a non-empty list generation whose
    // single ordered submission the host must see once.
    let barrier_call: BarrierCall = unsafe { slot(list, 0xd0) };
    let barrier = Barrier {
        kind: 2, // UAV
        flags: 0,
        resource: buffer,
        second: 0,
        last: 0,
    };
    unsafe { barrier_call(list, 1, &barrier) };
    let close: Close = unsafe { slot(list, 0x48) };
    assert_eq!(unsafe { close(list) }, S_OK);
    let execute: Execute = unsafe { slot(queue_ptr, 0x50) };
    unsafe { execute(queue_ptr, 1, &list) };
    match seen.recv_timeout(TIMEOUT).expect("one ordered submission") {
        Seen::Submission(first) => {
            assert_eq!(first.list.map(|identity| identity.list_id), Some(1));
            assert!(
                matches!(
                    first.commands.as_slice(),
                    [CommandV3::ResourceBarrier(barrier)] if !barrier.entries.is_empty()
                ),
                "unexpected commands {:?}",
                first.commands
            );
        }
        other => panic!("expected a submission, saw {other:?}"),
    }

    // Draining the same list generation again sends nothing: the submission is
    // consumed exactly once.
    unsafe { execute(queue_ptr, 1, &list) };
    assert!(
        seen.recv_timeout(Duration::from_millis(100)).is_err(),
        "a replayed list generation must not be sent twice"
    );

    // The same buffer bound twice: two `IASetVertexBuffers` calls, one host
    // registration (the fixture saw exactly one kind 2 frame above, and the
    // binds send none).
    let close_list: Close = unsafe { slot(list, 0x48) };
    let mut list3: This = null();
    assert_eq!(
        unsafe {
            list_create(
                device_ptr,
                0,
                list_desc.kind,
                allocator,
                null(),
                &LIST_IID,
                &mut list3,
            )
        },
        S_OK
    );
    let set_vertex_buffers: SetVertexBuffers = unsafe { slot(list3, 0x180) };
    let view = MetalVertexView {
        address,
        size: PAGE_LEN as u32,
        stride: 16,
    };
    unsafe { set_vertex_buffers(list3, 0, 1, &view) };
    unsafe { set_vertex_buffers(list3, 0, 1, &view) };
    assert_eq!(
        unsafe { close_list(list3) },
        S_OK,
        "binding the same range twice is idempotent"
    );
    assert!(
        seen.recv_timeout(Duration::from_millis(100)).is_err(),
        "the second bind of a registered range sends no registration"
    );

    // A buffer whose range resolves to an allocation with no share file is
    // refused explicitly: the list records the refusal and `Close` returns it,
    // and nothing is submitted (no silent skip, no D3DMetal fallback).
    *rules.lock().unwrap_or_else(|e| e.into_inner()) = vec![(0, u64::MAX, SHARE_ID + 1)];
    let mut refused: This = null();
    assert_eq!(
        unsafe {
            create_resource(
                device_ptr,
                &heap,
                0,
                &desc,
                0,
                null(),
                &RESOURCE_IID,
                &mut refused,
            )
        },
        S_OK,
        "creation itself does not depend on host visibility"
    );
    let mut list4: This = null();
    assert_eq!(
        unsafe {
            list_create(
                device_ptr,
                0,
                list_desc.kind,
                allocator,
                null(),
                &LIST_IID,
                &mut list4,
            )
        },
        S_OK
    );
    let set_refused: SetVertexBuffers = unsafe { slot(list4, 0x180) };
    let refused_view = MetalVertexView {
        address,
        size: PAGE_LEN as u32,
        stride: 16,
    };
    unsafe { set_refused(list4, 0, 1, &refused_view) };
    let close_refused: Close = unsafe { slot(list4, 0x48) };
    assert_ne!(
        unsafe { close_refused(list4) },
        S_OK,
        "an unregistered range is an explicit failure HRESULT"
    );
    assert!(
        seen.recv_timeout(Duration::from_millis(100)).is_err(),
        "a refused list submits nothing"
    );
    *rules.lock().unwrap_or_else(|e| e.into_inner()) = vec![(0, u64::MAX, SHARE_ID)];

    // PSO registration: kind 4 carries the session's ticket and the entry name,
    // kind 129 answers with the host's native pipeline id.
    let pipeline_id = queue::register_pipeline(b"MTLB fixture bytes", "main")
        .expect("the host accepts a pipeline registration");
    assert_eq!(pipeline_id, 0x51, "the host's own id is what the guest keeps");
    match seen.recv_timeout(TIMEOUT).expect("the kind 4 registration") {
        Seen::Pipeline(ticket, entry) => {
            assert_eq!(ticket, 1, "the session's first ticket");
            assert_eq!(entry, "main");
        }
        other => panic!("expected a pipeline registration, saw {other:?}"),
    }
    let (buffers_registered, psos_registered) = telemetry::registration_counts();
    assert!(buffers_registered >= 1, "the buffer was registered");
    assert!(psos_registered >= 1, "the PSO was registered");

    // The registered pipeline id and a resolved vertex binding are what the
    // wire carries: a producer submission names the host's id, never a title
    // pointer or a guest address.
    let mut producer =
        victory_gpu::native_metal::producer::MetalProducer::new(9, 1).expect("a list identity");
    let resolved = bindings::vertex_buffer(
        &*bindings::resolver().expect("the test installed a resolver"),
        address,
        PAGE_LEN,
        16,
    )
    .expect("the range resolves");
    let (bound, _tokens) = resolved.into_parts();
    producer
        .set_vertex_buffers_bound(0, &[bound])
        .expect("the resolved binding records");
    producer
        .set_render_targets(
            &[victory_gpu::native_metal::producer::RenderTarget::cleared(
                victory_gpu::native_metal::producer::ResourceId::new(SHARE_ID, 1)
                    .expect("a target identity"),
                [0.0, 0.0, 0.0, 1.0],
            )],
            None,
        )
        .expect("one colour target");
    // A graphics PSO cannot be registered over kind 4, so the producer-level
    // draw that forces the vertex state out uses a literal id: what is under
    // test here is the resolved binding the wire carries.
    producer.set_pipeline(0xA1).expect("a graphics pipeline id");
    producer
        .draw(3, 1, 0, 0)
        .expect("a draw flushes the bound vertex state");
    producer
        .set_compute_pipeline(pipeline_id)
        .expect("the host's pipeline id is the compute pipeline");
    producer
        .set_compute_threadgroup_size([64, 1, 1])
        .expect("a threadgroup size");
    producer.dispatch(1, 1, 1).expect("a dispatch records");
    let submission = producer.flush().expect("one submission");
    queue::submit_recorded_batch(&[submission], 0).expect("the host accepts the batch");
    match seen.recv_timeout(TIMEOUT).expect("the submitted batch") {
        Seen::Submission(carried) => {
            assert!(
                carried.commands.iter().any(|command| matches!(
                    command,
                    CommandV3::SetVertexBuffers(set)
                        if set.buffers.iter().all(|buffer| buffer.allocation_id == SHARE_ID)
                )),
                "the vertex binding names the registered allocation: {:?}",
                carried.commands
            );
            assert!(
                carried.commands.iter().any(|command| matches!(
                    command,
                    CommandV3::SetComputePipeline(reference) if reference.pipeline_id == pipeline_id
                )),
                "the submission names the host's pipeline id: {:?}",
                carried.commands
            );
        }
        other => panic!("expected a submission, saw {other:?}"),
    }

    // The pool path is explicit about what served a translation. A macOS
    // process with a converter uses the persistent workers (never a new
    // process per shader); this Win64 guest cannot load the x86_64 macOS
    // converter, so a cold shader is an explicit refusal that names it.
    let program = pso::StageProgram {
        stage: pso::PipelineStage::Compute,
        dxil: b"DXIL fixture bytes the converter would read".to_vec(),
        entry: "main".to_string(),
    };
    match pso::translate(&program) {
        Ok((metallib, serving)) => {
            eprintln!(
                "pool served one translation: {} metallib bytes, warm = {}",
                metallib.len(),
                serving.warm
            );
        }
        Err(error) => eprintln!(
            "pool translation skipped (converter available: {}): {error}",
            pso::converter_available()
        ),
    }
    eprintln!(
        "pool worker job counts: {:?}, worker program: {:?}, converter: {:?}",
        pso::pool_worker_jobs(),
        pso::pool_worker_program(),
        pso::converter_library()
    );
    // A graphics PSO registers with the version 2 frame: one library per stage,
    // the encoded PipelineState, a VAB1 ABI and the raw root-signature bytes.
    let graphics_state = pso::encode_graphics_state(&[pso::DEFAULT_COLOR_FORMAT], "main");
    let graphics_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_VERTEX, VERTEX_REFLECTION),
        (pso::WIRE_STAGE_PIXEL, PIXEL_REFLECTION),
    ]);
    let graphics_details = pso::PipelineDetails {
        state: &graphics_state,
        abi: &graphics_abi,
        root_signature: b"graphics root signature",
    };
    let graphics_id = pso::register_translated(
        &[
            (pso::PipelineStage::Vertex, b"MTLB vertex bytes", "main"),
            (pso::PipelineStage::Pixel, b"MTLB pixel bytes", "main"),
        ],
        &graphics_details,
    )
    .expect("the fixture host accepts a v2 graphics registration");
    match seen.recv_timeout(TIMEOUT).expect("the second kind 4 registration") {
        Seen::PipelineV2 {
            kind,
            stages,
            state,
            abi,
            root_signature,
        } => {
            assert_eq!(kind, PIPELINE_KIND_GRAPHICS);
            assert_eq!(stages.len(), 2);
            assert_eq!(stages[0].0, pso::WIRE_STAGE_VERTEX);
            assert_eq!(stages[1].0, pso::WIRE_STAGE_PIXEL);
            // One library per stage: the persistent pool writes a distinct
            // metallib file for the vertex and the pixel stage, and the host
            // resolves each function in its own stage's library.
            let (vertex_ticket, pixel_ticket) = (stages[0].1, stages[1].1);
            assert_ne!(
                vertex_ticket, pixel_ticket,
                "each stage names its own ticket"
            );
            assert_eq!(
                pixel_ticket,
                vertex_ticket + 1,
                "stage tickets are consecutive"
            );
            let vertex_library = share
                .path()
                .join(pso::metallib_name(vertex_ticket));
            let pixel_library = share.path().join(pso::metallib_name(pixel_ticket));
            assert_eq!(
                std::fs::read(&vertex_library).expect("the vertex stage's library"),
                b"MTLB vertex bytes",
                "ticket {vertex_ticket} names the vertex library"
            );
            assert_eq!(
                std::fs::read(&pixel_library).expect("the pixel stage's library"),
                b"MTLB pixel bytes",
                "ticket {pixel_ticket} names the pixel library"
            );
            assert_eq!(state, graphics_state);
            assert_eq!(abi, graphics_abi);
            assert_eq!(root_signature, b"graphics root signature");
        }
        other => panic!("expected a v2 graphics registration, saw {other:?}"),
    }

    // The registered graphics id is what `SetPipelineState` names, and the
    // submission carries it in `SetGraphicsPipeline`.
    let mut graphics_producer =
        victory_gpu::native_metal::producer::MetalProducer::new(10, 1).expect("a list identity");
    graphics_producer
        .set_pipeline(graphics_id)
        .expect("the graphics pipeline id");
    graphics_producer
        .set_render_targets(
            &[victory_gpu::native_metal::producer::RenderTarget::cleared(
                victory_gpu::native_metal::producer::ResourceId::new(SHARE_ID, 1)
                    .expect("a target identity"),
                [0.0, 0.0, 0.0, 1.0],
            )],
            None,
        )
        .expect("one colour target");
    graphics_producer
        .draw(3, 1, 0, 0)
        .expect("a draw records the graphics pipeline");
    let graphics_submission = graphics_producer.flush().expect("one submission");
    queue::submit_recorded_batch(&[graphics_submission], 0).expect("the host accepts the batch");
    match seen.recv_timeout(TIMEOUT).expect("the graphics submission") {
        Seen::Submission(carried) => assert!(
            carried.commands.iter().any(|command| matches!(
                command,
                CommandV3::SetGraphicsPipeline(reference) if reference.pipeline_id == graphics_id
            )),
            "the submission names the host's graphics id: {:?}",
            carried.commands
        ),
        other => panic!("expected a graphics submission, saw {other:?}"),
    }

    // A mesh PSO round-trips the same way and its id reaches `SetMeshPipeline`.
    let mesh_state = pso::encode_mesh_state(&[pso::DEFAULT_COLOR_FORMAT], "main");
    let mesh_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_MESH, MESH_REFLECTION),
        (pso::WIRE_STAGE_PIXEL, PIXEL_REFLECTION),
    ]);
    let mesh_details = pso::PipelineDetails {
        state: &mesh_state,
        abi: &mesh_abi,
        root_signature: b"mesh root signature",
    };
    let mesh_id = pso::register_translated(
        &[
            (pso::PipelineStage::Mesh, b"MTLB mesh bytes", "main"),
            (pso::PipelineStage::Pixel, b"MTLB pixel bytes", "main"),
        ],
        &mesh_details,
    )
    .expect("the fixture host accepts a v2 mesh registration");
    match seen.recv_timeout(TIMEOUT).expect("the mesh registration") {
        Seen::PipelineV2 { kind, stages, .. } => {
            assert_eq!(kind, PIPELINE_KIND_MESH);
            assert_eq!(stages.len(), 2);
            assert_eq!(stages[0].0, pso::WIRE_STAGE_MESH);
        }
        other => panic!("expected a v2 mesh registration, saw {other:?}"),
    }
    let mut mesh_producer =
        victory_gpu::native_metal::producer::MetalProducer::new(11, 1).expect("a list identity");
    mesh_producer
        .set_mesh_pipeline(mesh_id)
        .expect("the mesh pipeline id");
    mesh_producer
        .dispatch_mesh(1, 1, 1)
        .expect("a mesh dispatch records");
    let mesh_submission = mesh_producer.flush().expect("one submission");
    queue::submit_recorded_batch(&[mesh_submission], 0).expect("the host accepts the batch");
    match seen.recv_timeout(TIMEOUT).expect("the mesh submission") {
        Seen::Submission(carried) => assert!(
            carried.commands.iter().any(|command| matches!(
                command,
                CommandV3::SetMeshPipeline(reference) if reference.pipeline_id == mesh_id
            )),
            "the submission names the host's mesh id: {:?}",
            carried.commands
        ),
        other => panic!("expected a mesh submission, saw {other:?}"),
    }

    // A stage set that cannot form one pipeline is an explicit refusal naming
    // the failing stage; it never produces a partial registration.
    match pso::register_translated(
        &[(pso::PipelineStage::Vertex, b"MTLB vertex bytes", "main")],
        &graphics_details,
    ) {
        Err(pso::PipelineError::HostCannotRegister { stage, .. }) => {
            assert_eq!(stage, pso::PipelineStage::Vertex);
        }
        other => panic!("a vertex-only PSO must be refused explicitly, saw {other:?}"),
    }

    // Signal advances the fence; GetCompletedValue reads it.
    let signal: Signal = unsafe { slot(queue_ptr, 0x70) };
    assert_eq!(unsafe { signal(queue_ptr, fence, 7) }, S_OK);
    let completed: Completed = unsafe { slot(fence, 0x40) };
    assert_eq!(unsafe { completed(fence) }, 7);

    // Present: the installed WirePresenter registers surface 7 and carries one
    // Present command to the host.
    let present: Present = unsafe { slot(queue_ptr, 0xf0) };
    let hr = unsafe { present(queue_ptr, 0, null(), null()) };
    assert_eq!(hr, S_OK, "the host accepted the native present");
    match seen.recv_timeout(TIMEOUT).expect("the present submission") {
        Seen::Submission(presented) => assert!(
            matches!(
                presented.commands.as_slice(),
                [CommandV3::Present(present)] if present.surface_id == 7
            ),
            "unexpected commands {:?}",
            presented.commands
        ),
        other => panic!("expected a submission, saw {other:?}"),
    }

    // Nothing on this path may fall back to D3DMetal.
    assert_eq!(telemetry::d3dmetal_fallback_count(), 0);
    assert!(
        telemetry::registration_failures() >= 1,
        "the refused range was counted, not swallowed"
    );
    bindings::clear_resolver();
}
