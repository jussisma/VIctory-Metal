//! CPU/loopback tests for the kind 8 texture and kind 9 surface registration
//! frames.
//!
//! These tests never touch the GPU: a CPU `BatchExecutor` records the
//! registrations and the v3 allocation snapshot. They prove, over the real
//! authenticated loopback transport that:
//!
//! 1. a valid texture and surface registration is acknowledged and a later v3
//!    `BeginRenderPass` resolves the texture handle through the snapshot;
//! 2. a duplicate texture id, a zero extent, an unsupported format and a
//!    surface for an unregistered texture are refused with status 1 without
//!    creating anything;
//! 3. kind 8 cannot precede authentication.
//!
//! Run with:
//! ```text
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test ipc_texture_reg -- --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use victory_metal::ipc_host::{
    BatchExecutor, HostError, HostSecret, IpcHost, KIND_ACK, KIND_SURFACE_REG, KIND_TEXTURE_REG,
    SURFACE_REG_PAYLOAD_LEN, TEXTURE_REG_PAYLOAD_LEN, V3_REQUEST_HEADER_LEN, V3_REQUEST_MAGIC,
};
use victory_metal::macos::registry::TextureDesc;
use victory_metal::transport::MetalError;
use victory_metal::wire::{decode_reply, Reply, ReplyStatus, Submission};
use victory_metal::wire_v3::{
    encode_submission_v3, BeginRenderPass, ColorAttachment, CommandV3, LoadOp, StoreOp,
    SubmissionV3, PROTOCOL_VERSION_V3,
};
use victory_metal::VmBufferId;

const SECRET: [u8; 32] = [0x5a; 32];
const TEXTURE_ID: u64 = 0x1000;
const SURFACE_ID: u64 = 1;
const ALLOC_ID: u64 = 7;

#[derive(Default)]
struct StubState {
    textures: Vec<(u64, TextureDesc)>,
    surfaces: Vec<(u64, u64)>,
    v3: Vec<(SubmissionV3, HashMap<(u64, u32), VmBufferId>)>,
}

struct StubExecutor {
    state: Arc<Mutex<StubState>>,
}

impl StubExecutor {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(StubState::default())),
        }
    }

    fn state(&self) -> MutexGuard<'_, StubState> {
        self.state.lock().expect("stub state mutex")
    }
}

impl BatchExecutor for StubExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        Ok(4096)
    }

    fn register_window(&self, _ptr: *mut u8, _len: usize) -> Result<VmBufferId, MetalError> {
        Ok(VmBufferId(ALLOC_ID))
    }

    fn unregister_window(&self, _id: VmBufferId) -> Result<(), MetalError> {
        Ok(())
    }

    fn submit_batch(
        &self,
        submission: &Submission,
        _allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        self.state()
            .v3
            .push((submission.clone(), allocations.clone()));
        Ok(Reply {
            sequence: submission.sequence,
            status: ReplyStatus::Ok,
        })
    }

    fn register_texture(&self, id: u64, desc: TextureDesc) -> Result<(), MetalError> {
        self.state().textures.push((id, desc));
        Ok(())
    }

    fn register_surface(&self, surface_id: u64, texture_id: u64) -> Result<(), MetalError> {
        self.state().surfaces.push((surface_id, texture_id));
        Ok(())
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-ipc-texture-reg-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create test share dir");
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

struct Server {
    addr: SocketAddr,
    handle: thread::JoinHandle<Result<(), HostError>>,
    state: Arc<Mutex<StubState>>,
}

fn start(share: &TempDir) -> Server {
    let executor = StubExecutor::new();
    let state = Arc::clone(&executor.state);
    let mut host =
        IpcHost::new(HostSecret::new(SECRET), share.path(), executor).expect("create ipc host");
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept test peer");
        host.serve_once(stream).map(|_| ())
    });
    Server {
        addr,
        handle,
        state,
    }
}

fn connect(server: &Server) -> TcpStream {
    let stream = TcpStream::connect(server.addr).expect("connect test peer");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("client read timeout");
    stream
}

fn authenticate(client: &mut TcpStream) {
    write_frame(client, 1, &SECRET).expect("send auth");
    expect_ack(client, true);
}

fn write_frame(client: &mut TcpStream, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 5];
    header[0] = kind;
    header[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    client.write_all(&header)?;
    client.write_all(payload)?;
    client.flush()
}

fn read_frame(client: &mut TcpStream) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; 5];
    let mut filled = 0usize;
    while filled < header.len() {
        match client.read(&mut header[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed inside an envelope",
                ));
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let mut payload = vec![0u8; len];
    client.read_exact(&mut payload)?;
    Ok(Some((header[0], payload)))
}

fn expect_ack(client: &mut TcpStream, accepted: bool) {
    let expected = if accepted { 0 } else { 1 };
    assert_eq!(
        read_frame(client).expect("read ack"),
        Some((KIND_ACK, vec![expected]))
    );
}

/// Encodes a kind 8 texture registration payload.
#[allow(clippy::too_many_arguments)]
fn texture_payload(
    id: u64,
    generation: u32,
    dimension: u8,
    format: u8,
    usage: u8,
    sample_count: u8,
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u16,
    array_layers: u16,
) -> [u8; TEXTURE_REG_PAYLOAD_LEN] {
    let mut payload = [0u8; TEXTURE_REG_PAYLOAD_LEN];
    payload[0..8].copy_from_slice(&id.to_le_bytes());
    payload[8..12].copy_from_slice(&generation.to_le_bytes());
    payload[12] = dimension;
    payload[13] = format;
    payload[14] = usage;
    payload[15] = sample_count;
    payload[16..20].copy_from_slice(&width.to_le_bytes());
    payload[20..24].copy_from_slice(&height.to_le_bytes());
    payload[24..28].copy_from_slice(&depth.to_le_bytes());
    payload[28..30].copy_from_slice(&mip_levels.to_le_bytes());
    payload[30..32].copy_from_slice(&array_layers.to_le_bytes());
    payload
}

/// A valid 2D RGBA8 colour-target registration.
fn colour_target(id: u64, width: u32, height: u32) -> [u8; TEXTURE_REG_PAYLOAD_LEN] {
    texture_payload(id, 1, 1, 0, 1, 1, width, height, 1, 1, 1)
}

fn surface_payload(surface_id: u64, texture_id: u64) -> [u8; SURFACE_REG_PAYLOAD_LEN] {
    let mut payload = [0u8; SURFACE_REG_PAYLOAD_LEN];
    payload[0..8].copy_from_slice(&surface_id.to_le_bytes());
    payload[8..16].copy_from_slice(&texture_id.to_le_bytes());
    payload
}

fn register(client: &mut TcpStream, kind: u8, payload: &[u8]) {
    write_frame(client, kind, payload).expect("send registration");
}

fn v3_envelope(submission: &SubmissionV3) -> Vec<u8> {
    let frame = encode_submission_v3(submission).expect("encode v3 submission");
    let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
    payload.extend_from_slice(&V3_REQUEST_MAGIC);
    payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
    payload.extend_from_slice(&submission.sequence.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes());
    payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    payload.extend_from_slice(&frame);
    payload
}

fn roundtrip_v3(client: &mut TcpStream, submission: &SubmissionV3) -> Reply {
    write_frame(client, 5, &v3_envelope(submission)).expect("send v3 submission");
    let (kind, payload) = read_frame(client)
        .expect("read v3 reply")
        .expect("v3 reply frame");
    assert_eq!(kind, 5);
    decode_reply(&payload).expect("decode v3 reply")
}

fn render_pass_submission(sequence: u32, allocation_id: u64) -> SubmissionV3 {
    SubmissionV3 {
        sequence,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![ColorAttachment {
                    allocation_id,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                }],
                depth_stencil: None,
            }),
            CommandV3::EndRenderPass,
        ],
    }
}

#[test]
fn texture_and_surface_registration_resolve_a_render_pass() {
    let dir = TempDir::new();
    let server = start(&dir);
    let mut client = connect(&server);
    authenticate(&mut client);

    register(
        &mut client,
        KIND_TEXTURE_REG,
        &colour_target(TEXTURE_ID, 1280, 720),
    );
    expect_ack(&mut client, true);
    register(
        &mut client,
        KIND_SURFACE_REG,
        &surface_payload(SURFACE_ID, TEXTURE_ID),
    );
    expect_ack(&mut client, true);

    let submission = render_pass_submission(1, TEXTURE_ID);
    let reply = roundtrip_v3(&mut client, &submission);
    assert_eq!(reply.status, ReplyStatus::Ok);

    drop(client);
    server
        .handle
        .join()
        .expect("server thread")
        .expect("session");

    let state = server.state.lock().expect("stub state mutex");
    assert_eq!(state.textures.len(), 1);
    assert_eq!(state.textures[0].0, TEXTURE_ID);
    assert_eq!(state.textures[0].1.width, 1280);
    assert_eq!(state.textures[0].1.height, 720);
    assert_eq!(state.surfaces, vec![(SURFACE_ID, TEXTURE_ID)]);
    assert_eq!(state.v3.len(), 1);
    assert_eq!(
        state.v3[0].1.get(&(TEXTURE_ID, 1)),
        Some(&VmBufferId(TEXTURE_ID)),
        "the render pass must resolve the registered texture through the snapshot"
    );
}

#[test]
fn duplicate_texture_id_is_refused_without_replacement() {
    let dir = TempDir::new();
    let server = start(&dir);
    let mut client = connect(&server);
    authenticate(&mut client);

    register(
        &mut client,
        KIND_TEXTURE_REG,
        &colour_target(TEXTURE_ID, 64, 64),
    );
    expect_ack(&mut client, true);
    register(
        &mut client,
        KIND_TEXTURE_REG,
        &colour_target(TEXTURE_ID, 32, 32),
    );
    expect_ack(&mut client, false);

    drop(client);
    server
        .handle
        .join()
        .expect("server thread")
        .expect("session");

    let state = server.state.lock().expect("stub state mutex");
    assert_eq!(
        state.textures.len(),
        1,
        "the duplicate must not replace the texture"
    );
    assert_eq!(state.textures[0].1.width, 64);
    assert_eq!(state.textures[0].1.height, 64);
}

#[test]
fn zero_extent_and_unsupported_format_are_refused() {
    let dir = TempDir::new();
    let server = start(&dir);
    let mut client = connect(&server);
    authenticate(&mut client);

    register(
        &mut client,
        KIND_TEXTURE_REG,
        &colour_target(TEXTURE_ID, 0, 64),
    );
    expect_ack(&mut client, false);

    // Format code 200 is outside the documented table.
    let unsupported = texture_payload(TEXTURE_ID, 1, 1, 200, 1, 1, 64, 64, 1, 1, 1);
    register(&mut client, KIND_TEXTURE_REG, &unsupported);
    expect_ack(&mut client, false);

    // Wrong payload length is also a rejection, not a session close.
    register(&mut client, KIND_TEXTURE_REG, &[0u8; 8]);
    expect_ack(&mut client, false);

    drop(client);
    server
        .handle
        .join()
        .expect("server thread")
        .expect("session");

    let state = server.state.lock().expect("stub state mutex");
    assert!(
        state.textures.is_empty(),
        "a refused registration creates no GPU work"
    );
}

#[test]
fn surface_for_unregistered_texture_is_refused() {
    let dir = TempDir::new();
    let server = start(&dir);
    let mut client = connect(&server);
    authenticate(&mut client);

    register(
        &mut client,
        KIND_SURFACE_REG,
        &surface_payload(SURFACE_ID, TEXTURE_ID),
    );
    expect_ack(&mut client, false);

    drop(client);
    server
        .handle
        .join()
        .expect("server thread")
        .expect("session");

    let state = server.state.lock().expect("stub state mutex");
    assert!(state.textures.is_empty());
    assert!(state.surfaces.is_empty());
}

#[test]
fn kind_8_before_authentication_is_refused() {
    let dir = TempDir::new();
    let server = start(&dir);
    let mut client = connect(&server);

    register(
        &mut client,
        KIND_TEXTURE_REG,
        &colour_target(TEXTURE_ID, 64, 64),
    );
    // The first frame must authenticate; the host answers with the rejection
    // ack and closes the session.
    expect_ack(&mut client, false);
    assert_eq!(read_frame(&mut client).expect("read close"), None);

    let result = server.handle.join().expect("server thread");
    assert!(matches!(result, Err(HostError::Protocol(_))));

    let state = server.state.lock().expect("stub state mutex");
    assert!(state.textures.is_empty());
}
