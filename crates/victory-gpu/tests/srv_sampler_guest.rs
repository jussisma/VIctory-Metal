//! Guest-side proof for typed v3 descriptor tables and sampler registration.
//!
//! The first half is CPU-only: it resolves a two-entry texture table and a
//! one-entry sampler table, encodes a real v3 submission, decodes it again and
//! asserts the entry shapes and the wire lengths. It also proves the typed
//! refusals: a texture entry is not a buffer range (and vice versa), an entry
//! shape the reflected kind cannot express is refused, and a handle the
//! session did not register is a typed error before any submission.
//!
//! The second half drives the real loopback IPC session: an
//! `IpcHost<FakeExecutor>` serves one `WireV3Client`, the guest registers a
//! sampler (kind 10) and a texture (kind 8), and a v3 submission with the
//! texture and sampler tables is accepted. Re-registering the same sampler
//! content is idempotent, sends no second frame and the handle still works in a
//! later frame; an unknown sampler handle is refused by the host with
//! `UnknownHandle`.
//!
//! No D3DMetal path exists in this test; the executor is CPU-only and records
//! what it was asked to do.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use victory_gpu::native_metal::bindings::{
    DescriptorEntry, DescriptorSlot, GuestBinding, GuestRange, GuestRangeResolver, HandleError,
    ResolveError, TypedDescriptorTableRun, descriptor_table_typed,
    enumerate_descriptor_table_typed, require_registered_handles,
};
use victory_gpu::native_metal::client::WireV3Client;
use victory_gpu::native_metal::queue::V3Sink;
use victory_gpu::native_metal::sampler::{
    MAX_SAMPLER_STATES, SamplerStateDesc, reset_sampler_states, sampler_handle, sampler_state,
    sampler_state_count,
};
use victory_metal::abi::VmBufferId;
use victory_metal::ipc_host::{BatchExecutor, HostError, HostSecret, IpcHost, SessionSummary};
use victory_metal::macos::device::SamplerDesc;
use victory_metal::macos::registry::TextureDesc;
use victory_metal::transport::MetalError;
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    CommandV3, DescriptorKind, DescriptorTableEntryV3, DescriptorTableV3, SubmissionV3,
    decode_submission_v3, encode_submission_v3,
};

const SECRET: [u8; 32] = [0x77; 32];
const TEXTURE0: u64 = 0x3000;
const TEXTURE1: u64 = 0x3001;
const SAMPLER_HANDLE: u64 = 0x5300_0000_0000_0042;

/// A resolver that refuses every buffer lookup: the typed table test names only
/// registered texture/sampler handles.
struct NoBufferResolver;

impl GuestRangeResolver for NoBufferResolver {
    fn resolve(&self, _guest_address: u64, _len: u64) -> Result<GuestRange, ResolveError> {
        Err(ResolveError::Unmapped { address: 0, len: 0 })
    }

    fn retain(&self, _range: &GuestRange) -> Result<u64, ResolveError> {
        Err(ResolveError::Retain("no buffer is expected"))
    }

    fn retire(&self, _token: u64) -> Result<(), ResolveError> {
        Err(ResolveError::Retire("no buffer is expected"))
    }
}

fn texture_table() -> DescriptorTableV3 {
    DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Srv,
        bindings: vec![
            DescriptorTableEntryV3::Texture(TEXTURE0),
            DescriptorTableEntryV3::Texture(TEXTURE1),
        ],
    }
}

fn sampler_table(handle: u64) -> DescriptorTableV3 {
    DescriptorTableV3 {
        set: 1,
        base_register: 0,
        kind: DescriptorKind::Sampler,
        bindings: vec![DescriptorTableEntryV3::Sampler(handle)],
    }
}

#[test]
fn typed_texture_and_sampler_tables_round_trip_the_wire() {
    let resolver = NoBufferResolver;
    let texture = descriptor_table_typed(
        &resolver,
        0x100,
        0,
        0,
        DescriptorKind::Srv,
        &[
            DescriptorEntry::Texture(TEXTURE0),
            DescriptorEntry::Texture(TEXTURE1),
        ],
    )
    .expect("a two-entry texture table");
    let (resolved_texture, tokens) = texture.into_parts();
    assert!(tokens.is_empty(), "texture entries pin no buffer");
    assert_eq!(resolved_texture, texture_table());

    let sampler = descriptor_table_typed(
        &resolver,
        0x200,
        1,
        0,
        DescriptorKind::Sampler,
        &[DescriptorEntry::Sampler(SAMPLER_HANDLE)],
    )
    .expect("a one-entry sampler table");
    let (resolved_sampler, tokens) = sampler.into_parts();
    assert!(tokens.is_empty(), "sampler entries pin no buffer");
    assert_eq!(resolved_sampler, sampler_table(SAMPLER_HANDLE));

    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::SetDescriptorTable(resolved_texture.clone()),
            CommandV3::SetDescriptorTable(resolved_sampler.clone()),
        ],
    };
    let encoded = encode_submission_v3(&submission).expect("encode the submission");
    let decoded = decode_submission_v3(&encoded).expect("decode the submission");
    assert_eq!(decoded, submission);

    // The texture table is the 8-byte resource-handle encoding and the sampler
    // table is too; the legacy buffer tag stays 0. The decoded values are the
    // observable contract, so assert the entry shapes rather than an internal
    // offset constant.
    let CommandV3::SetDescriptorTable(decoded_texture) = &decoded.commands[0] else {
        panic!("the first command is the texture table");
    };
    assert_eq!(decoded_texture.bindings.len(), 2);
    assert_eq!(
        decoded_texture.bindings[0].resource_handle(),
        Some(TEXTURE0)
    );
    assert!(
        decoded_texture.bindings[0].as_buffer().is_none(),
        "a texture entry is not a buffer range"
    );
    let CommandV3::SetDescriptorTable(decoded_sampler) = &decoded.commands[1] else {
        panic!("the second command is the sampler table");
    };
    assert_eq!(
        decoded_sampler.bindings[0].resource_handle(),
        Some(SAMPLER_HANDLE)
    );
}

#[test]
fn a_texture_entry_is_not_a_buffer_range_and_incompatible_shapes_are_refused() {
    let resolver = NoBufferResolver;
    // A buffer entry in a sampler table.
    assert!(matches!(
        descriptor_table_typed(
            &resolver,
            0x100,
            0,
            0,
            DescriptorKind::Sampler,
            &[DescriptorEntry::Buffer(GuestBinding {
                address: 0x1_0000,
                len: 16,
            })]
        ),
        Err(ResolveError::Unsupported(_))
    ));
    // A texture entry in a CBV table.
    assert!(matches!(
        descriptor_table_typed(
            &resolver,
            0x101,
            0,
            0,
            DescriptorKind::Cbv,
            &[DescriptorEntry::Texture(TEXTURE0)]
        ),
        Err(ResolveError::Unsupported(_))
    ));
    // A buffer and a texture in one SRV table are legal: the encoder keeps
    // both shapes in guest order, so what fails here is the buffer lookup, not
    // an entry-shape refusal.
    assert_eq!(
        descriptor_table_typed(
            &resolver,
            0x102,
            0,
            0,
            DescriptorKind::Srv,
            &[
                DescriptorEntry::Buffer(GuestBinding {
                    address: 0x1_0000,
                    len: 16,
                }),
                DescriptorEntry::Texture(TEXTURE0),
            ]
        ),
        Err(ResolveError::Unmapped { address: 0, len: 0 })
    );
    // A zero registered handle is invalid before any submission.
    assert!(matches!(
        descriptor_table_typed(
            &resolver,
            0x103,
            0,
            0,
            DescriptorKind::Srv,
            &[DescriptorEntry::Texture(0)]
        ),
        Err(ResolveError::Invalid(_))
    ));
}

#[test]
fn unregistered_handles_are_a_typed_error_before_submission() {
    let table = texture_table();
    assert_eq!(
        require_registered_handles(&table, |_| false, |_| false),
        Err(HandleError::UnregisteredTexture(TEXTURE0))
    );
    require_registered_handles(
        &table,
        |handle| handle == TEXTURE0 || handle == TEXTURE1,
        |_| false,
    )
    .expect("both texture handles are registered");

    let sampler = sampler_table(SAMPLER_HANDLE);
    assert_eq!(
        require_registered_handles(&sampler, |_| false, |_| false),
        Err(HandleError::UnregisteredSampler(SAMPLER_HANDLE))
    );
    require_registered_handles(&sampler, |_| false, |handle| handle == SAMPLER_HANDLE)
        .expect("the sampler handle is registered");
}

#[test]
fn sampler_content_is_idempotent_and_bounded() {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
    reset_sampler_states();
    let desc = SamplerStateDesc::NEAREST_CLAMP;
    let first = sampler_handle(&desc).expect("first registration");
    let second = sampler_handle(&desc).expect("same content");
    assert_eq!(first, second, "same descriptor -> same handle");
    assert_eq!(sampler_state(first), Some(desc));
    assert_eq!(sampler_state_count(), 1);
    assert_eq!(
        first,
        desc.handle(),
        "the handle is derived from the content, not an allocation order"
    );
    assert_ne!(
        first,
        SamplerStateDesc {
            min_filter: 1,
            ..desc
        }
        .handle(),
        "different content -> different handle"
    );
    assert_eq!(MAX_SAMPLER_STATES, 256);
    reset_sampler_states();
}

#[test]
fn descriptor_runs_enumerate_a_texture_srv_range_and_a_sampler_range() {
    const TABLE_BASE: u64 = 0x2000_0000;
    let texture_run: TypedDescriptorTableRun =
        enumerate_descriptor_table_typed(TABLE_BASE, |handle| match (handle - TABLE_BASE) / 32 {
            0 => DescriptorSlot::Texture(DescriptorKind::Srv, TEXTURE0),
            1 => DescriptorSlot::Texture(DescriptorKind::Srv, TEXTURE1),
            _ => DescriptorSlot::Empty,
        })
        .expect("a two-entry texture range");
    assert_eq!(texture_run.kind, DescriptorKind::Srv);
    assert_eq!(
        texture_run.entries,
        vec![
            DescriptorEntry::Texture(TEXTURE0),
            DescriptorEntry::Texture(TEXTURE1),
        ]
    );

    let sampler_run = enumerate_descriptor_table_typed(TABLE_BASE, |handle| {
        if handle == TABLE_BASE {
            DescriptorSlot::Sampler(SAMPLER_HANDLE)
        } else {
            DescriptorSlot::Empty
        }
    })
    .expect("a one-entry sampler range");
    assert_eq!(sampler_run.kind, DescriptorKind::Sampler);
    assert_eq!(
        sampler_run.entries,
        vec![DescriptorEntry::Sampler(SAMPLER_HANDLE)]
    );
}

// ---------------------------------------------------------------- IPC session

#[derive(Default)]
struct FakeState {
    next_buffer: u64,
    samplers: Vec<(u64, SamplerDesc)>,
    textures: Vec<u64>,
    submissions: Vec<SubmissionV3>,
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
        _submission: &victory_metal::wire::Submission,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<victory_metal::wire::Reply, MetalError> {
        Err(MetalError::NotSupported("this proof drives only version 3"))
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<victory_metal::wire::Reply, MetalError> {
        self.state().submissions.push(submission.clone());
        Ok(victory_metal::wire::Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }

    fn register_texture(&self, id: u64, _desc: TextureDesc) -> Result<(), MetalError> {
        self.state().textures.push(id);
        Ok(())
    }

    fn register_sampler(&self, id: u64, desc: SamplerDesc) -> Result<(), MetalError> {
        self.state().samplers.push((id, desc));
        Ok(())
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-srv-sampler-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create test dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
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

#[test]
fn sampler_registration_and_typed_tables_survive_a_second_frame() {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let dir = TestDir::new();
    let state = Arc::new(Mutex::new(FakeState::default()));
    let host = IpcHost::new(
        HostSecret::new(SECRET),
        dir.path(),
        FakeExecutor {
            state: Arc::clone(&state),
        },
    )
    .expect("create the host");
    let server = start(host);

    let mut client = WireV3Client::connect_with_timeout(
        &server.addr.to_string(),
        &SECRET,
        Duration::from_secs(5),
    )
    .expect("connect the guest client");

    // Kind 10: one sampler registration, idempotent by content.
    let desc = SamplerStateDesc::NEAREST_CLAMP;
    let handle = sampler_handle(&desc).expect("content handle");
    assert!(
        client
            .register_sampler(handle, &desc)
            .expect("register sampler")
    );
    assert_eq!(client.sampler_registrations(), 1);
    // The same content is a no-op: no second frame, same handle.
    assert!(
        client
            .register_sampler(handle, &desc)
            .expect("re-register sampler")
    );
    assert_eq!(client.sampler_registrations(), 1);
    assert_eq!(state.lock().unwrap().samplers.len(), 1);

    // Kind 8: the texture table's handles must be session-registered.
    for (texture, label) in [(TEXTURE0, "texture0"), (TEXTURE1, "texture1")] {
        assert!(
            client
                .register_texture(texture, 1, 1, 0, 0, 1, 4, 4, 1, 1, 1)
                .expect("register texture"),
            "texture {label} is accepted"
        );
    }

    // Frame 1: a two-entry SRV texture table and the one-entry sampler table.
    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::SetDescriptorTable(texture_table()),
            CommandV3::SetDescriptorTable(sampler_table(handle)),
        ],
    };
    let encoded = encode_submission_v3(&submission).expect("encode frame 1");
    assert_eq!(
        client.send(&encoded).expect("frame 1 is accepted"),
        ReplyStatus::Ok
    );

    // The handle survives a second frame: a new submission reuses it and the
    // host still resolves it, with no extra kind 10 frame.
    let second = SubmissionV3 {
        sequence: 2,
        list: None,
        commands: vec![CommandV3::SetDescriptorTable(sampler_table(handle))],
    };
    let encoded = encode_submission_v3(&second).expect("encode frame 2");
    assert_eq!(
        client.send(&encoded).expect("frame 2 is accepted"),
        ReplyStatus::Ok
    );
    assert!(client.sampler_registered(handle));
    assert_eq!(state.lock().unwrap().samplers.len(), 1);

    // An unknown sampler handle is refused by the host before execution, and
    // the guest's typed check catches it even earlier.
    let unknown = SAMPLER_HANDLE ^ 0x0100;
    assert_eq!(
        require_registered_handles(
            &sampler_table(unknown),
            |_| false,
            |id| { client.sampler_registered(id) }
        ),
        Err(HandleError::UnregisteredSampler(unknown))
    );
    let bad = SubmissionV3 {
        sequence: 3,
        list: None,
        commands: vec![CommandV3::SetDescriptorTable(sampler_table(unknown))],
    };
    let encoded = encode_submission_v3(&bad).expect("encode unknown frame");
    match client.send(&encoded) {
        Err(victory_gpu::native_metal::queue::NativeQueueError::Refused(message)) => {
            assert!(message.contains("UnknownHandle"), "{message}");
        }
        other => panic!("an unregistered sampler must be refused, got {other:?}"),
    }
    assert_eq!(
        state.lock().unwrap().submissions.len(),
        2,
        "the unknown-handle frame never reached the executor"
    );

    client.close();
    let (result, _host) = server.handle.join().expect("join the host thread");
    let summary = result.expect("the session ended cleanly");
    assert_eq!(
        summary.submissions, 3,
        "two executed frames and one refusal"
    );
}
