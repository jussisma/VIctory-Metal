//! Hardware proof for ADR-054: a `flags == 0` render/depth attachment is
//! registered with the host before a render pass needs it.
//!
//! Boot #26 refused 52 `BeginRenderPass` submissions with `UnknownHandle`
//! because the guest bound an RTV/DSV whose texture had no creation-time kind 8
//! registration (`flags == 0`: an at-address, placed or lazily made texture).
//! This test drives the failure the guest fixed, end to end and on the real
//! GPU:
//!
//! 1. A real [`DirectMetalTransport`] serves one [`IpcHost`] session over a
//!    loopback socket, exactly what `victory-metal-host` serves the guest.
//! 2. The guest registers an offscreen RGBA8 colour target and a D16 depth
//!    target (both `flags == 0` in D3D12 terms: the only use named is the
//!    attachment) through the same kind 8 queue the bind path uses.
//! 3. The real [`MetalProducer`] records `OMSetRenderTargets`, a colour clear
//!    and a depth clear; `flush` encodes the `BeginRenderPass`/`EndRenderPass`
//!    pair those clears fold into.
//! 4. `queue::submit_recorded_batch` carries the frame over the socket. The
//!    host must answer `Ok`, encode at least one render pass, and the
//!    process-wide `UnknownHandle` id must stay zero.
//!
//! Ignored by default: run it under the GPU lock with `--ignored`.
//!
//! ```text
//! .swarm/bin/gpu-lock sh -c 'export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!   cargo test --locked -p victory-gpu --target aarch64-apple-darwin \
//!     --test attachment_registration_gpu -- --ignored --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use victory_gpu::native_metal::client::WireV3Client;
use victory_gpu::native_metal::producer::{
    DepthTarget, MetalProducer, RenderTarget, ResourceId,
};
use victory_gpu::native_metal::queue::{
    NativeDispatch, native_queue_install_client, native_queue_reset, register_texture_with_reason,
    submit_recorded_batch,
};
use victory_metal::DirectMetalTransport;
use victory_metal::ipc_host::{HostError, HostSecret, IpcHost, SessionSummary};
use victory_metal::macos::device::last_unknown_id;
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::CommandV3;

const SECRET: [u8; 32] = [0x5a; 32];
/// Edge of the offscreen colour and depth targets.
const EDGE: u32 = 64;
/// Guest-address-shaped handles (the kind 8 identity a flags-0 texture has).
const COLOUR_HANDLE: u64 = 0x0000_0002_E2A6_A000;
const DEPTH_HANDLE: u64 = 0x0000_0002_E2A6_B000;
/// Kind 8 wire codes: DXGI 28 `R8G8B8A8_UNORM` is 0, DXGI 55 `D16_UNORM`
/// is 54.
const RGBA8_WIRE: u8 = 0;
const D16_WIRE: u8 = 54;
/// Kind 8 usage codes.
const USAGE_COLOR: u8 = 1;
const USAGE_DEPTH: u8 = 2;

struct Server {
    addr: String,
    handle: thread::JoinHandle<(Result<SessionSummary, HostError>, IpcHost<DirectMetalTransport>)>,
}

fn start(mut host: IpcHost<DirectMetalTransport>) -> Server {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept test peer");
        let result = host.serve_once(stream);
        (result, host)
    });
    Server {
        addr: addr.to_string(),
        handle,
    }
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-attachment-gpu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create test dir");
        Self(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Serializes the hardware proof with any other GPU test in the process.
fn serialized() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

#[test]
#[ignore = "GPU proof; run under .swarm/bin/gpu-lock with --ignored"]
fn a_flags_zero_colour_and_depth_attachment_encode_a_render_pass() {
    let _guard = serialized();
    native_queue_reset();
    let dir = TestDir::new();
    let host = IpcHost::new(
        HostSecret::new(SECRET),
        dir.path(),
        DirectMetalTransport::new().expect("the Metal device opens"),
    )
    .expect("create the host");
    let server = start(host);

    let client = WireV3Client::connect_with_timeout(
        &server.addr,
        &SECRET,
        Duration::from_secs(5),
    )
    .expect("connect the guest client");
    native_queue_install_client(client);

    // Kind 8: both textures are named only by an attachment, the class that
    // has no creation-time registration (`flags == 0`).
    for (handle, format, usage, label) in [
        (COLOUR_HANDLE, RGBA8_WIRE, USAGE_COLOR, "colour"),
        (DEPTH_HANDLE, D16_WIRE, USAGE_DEPTH, "depth"),
    ] {
        let (accepted, reason) = register_texture_with_reason(
            handle, 1, 1, format, usage, 1, EDGE, EDGE, 1, 1, 1,
        )
        .expect("the kind 8 registration reaches the host");
        assert!(
            accepted,
            "the host accepts the {label} attachment registration: {reason:?}"
        );
    }

    // OMSetRenderTargets + both clears, through the production producer.
    let colour = ResourceId::from_guest_address(COLOUR_HANDLE).expect("colour identity");
    let depth = ResourceId::from_guest_address(DEPTH_HANDLE).expect("depth identity");
    let mut producer = MetalProducer::new(1, 1).expect("a producer for list 1 generation 1");
    producer
        .set_render_targets(
            &[RenderTarget::cleared(colour, [0.0; 4])],
            Some(DepthTarget::new(depth)),
        )
        .expect("OMSetRenderTargets records the pass");
    producer
        .clear_render_target(colour, [0.25, 0.5, 0.75, 1.0])
        .expect("ClearRenderTargetView folds into the pass");
    producer
        .clear_depth_stencil(depth, true, 1.0, false, 0)
        .expect("ClearDepthStencilView folds into the pass");
    let submission = producer.flush().expect("the producer flushes");
    assert!(
        matches!(submission.commands.first(), Some(CommandV3::BeginRenderPass(_))),
        "the clears encode as one pass: {:?}",
        submission.commands.first()
    );

    let dispatch = submit_recorded_batch(std::slice::from_ref(&submission), 0)
        .expect("the host session carries the frame");
    match dispatch {
        NativeDispatch::Native {
            status,
            submissions,
        } => {
            assert_eq!(submissions, 1);
            assert_eq!(
                status,
                ReplyStatus::Ok,
                "the pass with a flags-0 colour and depth attachment is accepted"
            );
        }
        NativeDispatch::Host => panic!("the native queue carried the frame"),
    }

    // End the session so the host hands the transport back.
    native_queue_reset();
    let (result, host) = server.handle.join().expect("join the host thread");
    let summary = result.expect("the session ended cleanly");
    assert!(
        summary.submissions >= 1,
        "the host executed the frame: {summary:?}"
    );

    let counters = host.executor().encode_counters();
    assert!(
        counters.render_passes >= 1,
        "render_passes_encoded advanced: {counters:?}"
    );
    assert_eq!(
        last_unknown_id(),
        0,
        "no UnknownHandle was recorded for an attachment"
    );
}
