//! Live title command-list seam (ADR-019), driven through the production
//! `victory-gpu` D3D12 entry points under `VICTORY_BACKEND=metal`.
//!
//! The test creates the real device (which auto-connects the Wire v3 session
//! and installs the native presenter), a queue, an allocator, a command list
//! and a fence through the device's own vtable, then:
//!
//! 1. closes the list and runs `ExecuteCommandLists`; the host must receive
//!    exactly one ordered submission for that list generation;
//! 2. replays the same list; nothing is sent again (`AlreadyExecuted` is the
//!    host's answer for a replayed identity, and an empty drain is not a
//!    batch);
//! 3. `Signal`s the fence and reads `GetCompletedValue`;
//! 4. presents with no swapchain-less path: the installed `WirePresenter`
//!    registers the surface and carries one `CommandV3::Present`, so the
//!    guest entry point returns `S_OK` only when the host accepted it.
//!
//! The endpoint is an in-process loopback V3 host fixture, labelled here as
//! such: the 5B live gate repeats the same chain over the real host socket.
//! This test is Windows-only because the D3D12 entry points and their
//! descriptors exist only in the Win64 guest build; `cargo check --target
//! x86_64-pc-windows-gnullvm --all-targets` compiles it, and the gate runs
//! it.

#![cfg(windows)]

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::ptr::null;
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::Duration;

use victory_gpu::device;
use victory_gpu::object::{Object, S_OK};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{decode_submission_v3, CommandV3};

const SECRET_HEX: &str = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";
const TIMEOUT: Duration = Duration::from_secs(10);
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
type Execute = unsafe extern "system" fn(This, u32, *const This);
type Signal = unsafe extern "system" fn(This, This, u64) -> Hresult;
type Completed = unsafe extern "system" fn(This) -> u64;
type Present = unsafe extern "system" fn(This, u32, *const u8, *const u8) -> Hresult;

/// A loopback V3 host fixture: authenticates one peer, acknowledges surface
/// registrations and answers every submission with `Ok`, reporting each
/// decoded submission it saw.
struct TestHost {
    port: u16,
    handle: Option<JoinHandle<()>>,
}

impl TestHost {
    fn spawn() -> (Self, Receiver<victory_metal::wire_v3::SubmissionV3>) {
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

            while let Some((kind, payload)) = try_frame(&mut stream) {
                match kind {
                    // KIND_FENCE_PAGE
                    6 => write_frame(&mut stream, 128, &[0]),
                    // KIND_TEXTURE_REG / KIND_SURFACE_REG
                    8 | 9 => write_frame(&mut stream, 128, &[0]),
                    // KIND_SUBMISSION_V3
                    5 => {
                        let submission =
                            decode_submission_v3(&payload[16..]).expect("a canonical v3 frame");
                        let sequence = u32::from_le_bytes([
                            payload[6], payload[7], payload[8], payload[9],
                        ]);
                        sender.send(submission).expect("report the submission");
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

/// The real device/queue/list/fence/present chain against the fixture host.
#[test]
fn a_recorded_title_list_reaches_the_host_exactly_once() {
    let (host, submissions) = TestHost::spawn();
    // SAFETY: this test owns the process environment (one test per binary).
    unsafe {
        std::env::set_var("VICTORY_BACKEND", "metal");
        std::env::set_var("VICTORY_METAL_ENDPOINT", host.endpoint());
        std::env::set_var("VICTORY_METAL_SECRET", SECRET_HEX);
        // The installed presenter's surface (the host fixture accepts it).
        std::env::set_var("VICTORY_METAL_SURFACE", "7");
    }

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

    // A committed buffer, then one UAV barrier on it: a non-empty list
    // generation whose single ordered submission the host must see once.
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
    let barrier_call: BarrierCall = unsafe { slot(list, 0xd0) };
    let barrier = Barrier {
        kind: 2, // UAV
        flags: 0,
        resource: buffer,
        second: 0,
        last: 0,
    };
    unsafe { barrier_call(list, 1, &barrier) };

    // Close the list (flushes its one wire submission) and run it.
    let close: Close = unsafe { slot(list, 0x48) };
    assert_eq!(unsafe { close(list) }, S_OK);
    let execute: Execute = unsafe { slot(queue_ptr, 0x50) };
    unsafe { execute(queue_ptr, 1, &list) };
    let first = submissions
        .recv_timeout(TIMEOUT)
        .expect("one ordered submission for the list generation");
    assert_eq!(first.list.map(|identity| identity.list_id), Some(1));
    assert!(
        matches!(
            first.commands.as_slice(),
            [CommandV3::ResourceBarrier(barrier)] if !barrier.entries.is_empty()
        ),
        "unexpected commands {:?}",
        first.commands
    );

    // Draining the same list generation again sends nothing: the submission
    // is consumed exactly once.
    unsafe { execute(queue_ptr, 1, &list) };
    assert!(
        submissions.recv_timeout(Duration::from_millis(100)).is_err(),
        "a replayed list generation must not be sent twice"
    );

    // Signal advances the fence; GetCompletedValue reads it.
    let signal: Signal = unsafe { slot(queue_ptr, 0x70) };
    assert_eq!(unsafe { signal(queue_ptr, fence, 7) }, S_OK);
    let completed: Completed = unsafe { slot(fence, 0x40) };
    assert_eq!(unsafe { completed(fence) }, 7);

    // Present: the installed WirePresenter registers surface 7 and carries
    // one Present command to the host.
    let present: Present = unsafe { slot(queue_ptr, 0xf0) };
    let hr = unsafe { present(queue_ptr, 0, null(), null()) };
    assert_eq!(hr, S_OK, "the host accepted the native present");
    let presented = submissions
        .recv_timeout(TIMEOUT)
        .expect("the host saw the present submission");
    assert!(
        matches!(
            presented.commands.as_slice(),
            [CommandV3::Present(present)] if present.surface_id == 7
        ),
        "unexpected commands {:?}",
        presented.commands
    );

    // A refused frame is a real failure HRESULT (the no-swapchain shape is
    // covered by the queue unit tests; a host that refuses the surface is
    // what the gate injects).
}
