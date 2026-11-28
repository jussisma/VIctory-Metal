//! ADR-020 native `NSWindow` + `CAMetalLayer` presenter gate.
//!
//! The test runs the real `victory-metal-host` twice over loopback:
//!
//! 1. headless (`--aaa-frame`) with two cleared frames and a `Present` each;
//! 2. windowed (`--window`, `VICTORY_METAL_WINDOW=64x64`) with the exact same
//!    two frames.
//!
//! It then asserts that the window acquired and presented two drawables, that
//! the presented frame checksum equals the headless checksum, and that no
//! D3DMetal image is mapped on either run. The frames are ordinary v3
//! submissions registered through IPC kinds 8 and 9, so the test exercises the
//! production host path, not a synthetic shortcut.
//!
//! A window server is required. When `CGSessionCopyCurrentDictionary()` is
//! null (headless CI, no Aqua session) the test prints an explicit `SKIP`
//! reason and passes without pretending it ran.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test native_window_gpu -- --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use victory_metal::ipc_host::{
    KIND_ACK, KIND_AUTH, KIND_SUBMISSION_V3, KIND_SURFACE_REG, KIND_TEXTURE_REG,
    SURFACE_REG_PAYLOAD_LEN, TEXTURE_REG_PAYLOAD_LEN, V3_REQUEST_HEADER_LEN, V3_REQUEST_MAGIC,
};
use victory_metal::wire::{Reply, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DepthStencilAttachment, DrawState, LoadOp,
    PROTOCOL_VERSION_V3, PipelineRef, Present, PrimitiveTopology, RootConstantsV3, ScissorRect,
    StoreOp, SubmissionV3, Viewport, encode_submission_v3,
};

const SECRET: [u8; 32] = [0x5a; 32];
const WAIT: Duration = Duration::from_secs(30);
const TEXTURE_ID: u64 = 7;
const SURFACE_ID: u64 = 11;
const GENERATION: u32 = 1;
const EDGE: u32 = 64;

/// The AAA frame-gate pipeline ids are published here by the host.
const GRAPHICS_PIPELINE_FILE: &str = "victory-metal-host.graphics.id";
/// A back-buffer deliberately larger than the window: the title draws
/// 3840x2160 into a 1280x720 window, so the drawer must size the drawable to
/// the source rather than clear it.
const DRAWN_WIDTH: u32 = 128;
const DRAWN_HEIGHT: u32 = 96;
/// Second MRT colour and the depth target the built-in pipeline expects.
const COLOR1_ID: u64 = 8;
const DEPTH_ID: u64 = 9;
/// A descriptor-identical colour allocation that is registered as its own
/// present surface and used as the pass target, while the frame presents
/// `SURFACE_ID` (another object of the same shape). This is the composition
/// shape the boot shows: the drawn object and the presented object are
/// different `MTLTexture`s with one descriptor.
const PASS_TARGET_ID: u64 = 12;
const PASS_TARGET_SURFACE_ID: u64 = 13;

unsafe extern "C" {
    fn CGSessionCopyCurrentDictionary() -> *const c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

fn window_server_available() -> bool {
    // SAFETY: The copy function returns a +1 CFDictionary or null; the non-null
    // result is released exactly once.
    unsafe {
        let session = CGSessionCopyCurrentDictionary();
        if session.is_null() {
            return false;
        }
        CFRelease(session);
        true
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-native-window-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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

struct Host {
    child: Child,
    stdout: BufReader<ChildStdout>,
    port: u16,
}

impl Host {
    fn spawn(share: &TempDir, extra: &[&str], window_env: Option<&str>) -> Self {
        let mut envs = Vec::new();
        if let Some(value) = window_env {
            envs.push(("VICTORY_METAL_WINDOW", value));
        }
        Self::spawn_with_env(share, extra, &envs)
    }

    fn spawn_with_env(share: &TempDir, extra: &[&str], envs: &[(&str, &str)]) -> Self {
        let stderr_path = share.path().join("host.stderr");
        let stderr = std::fs::File::create(&stderr_path).expect("create stderr log");
        let mut command = Command::new(env!("CARGO_BIN_EXE_victory_metal_host"));
        command
            .arg("--secret-hex")
            .arg(hex(&SECRET))
            .arg("--port")
            .arg("0")
            .arg("--once")
            .arg("--share-dir")
            .arg(share.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr));
        for (name, value) in envs {
            command.env(name, value);
        }
        command.args(extra);
        let mut child = command.spawn().expect("spawn victory-metal-host");
        let stdout = child.stdout.take().expect("host stdout");
        let mut stdout = BufReader::new(stdout);
        let mut banner = String::new();
        stdout
            .read_line(&mut banner)
            .expect("read the ready banner");
        let port = parse_port(&banner).unwrap_or_else(|| {
            let status = child.wait().ok();
            let stderr = std::fs::read_to_string(&stderr_path).unwrap_or_default();
            panic!("no ready banner (exit {status:?}): {banner:?} {stderr}");
        });
        Self {
            child,
            stdout,
            port,
        }
    }

    fn wait(mut self) -> std::process::Output {
        drop(self.stdout);
        let deadline = Instant::now() + WAIT;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    panic!("the host did not exit within {WAIT:?}");
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(error) => panic!("try_wait failed: {error}"),
            }
        }
        let mut stdout = Vec::new();
        if let Some(mut pipe) = self.child.stdout.take() {
            let _ = pipe.read_to_end(&mut stdout);
        }
        let mut stderr = Vec::new();
        if let Some(mut pipe) = self.child.stderr.take() {
            let _ = pipe.read_to_end(&mut stderr);
        }
        let status = self.child.wait().expect("wait for the host");
        std::process::Output {
            status,
            stdout,
            stderr,
        }
    }
}

fn hex(secret: &[u8]) -> String {
    secret.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_port(banner: &str) -> Option<u16> {
    let rest = banner.strip_prefix("victory-metal-host: ready port=")?;
    rest.split_whitespace().next()?.parse().ok()
}

fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 5];
    header[0] = kind;
    header[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_frame(stream: &mut TcpStream) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; 5];
    let mut filled = 0usize;
    while filled < header.len() {
        match stream.read(&mut header[filled..]) {
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
    stream.read_exact(&mut payload)?;
    Ok(Some((header[0], payload)))
}

struct Client {
    stream: TcpStream,
}

impl Client {
    fn connect(port: u16) -> Self {
        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to the host");
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .expect("read timeout");
        let mut client = Self { stream };
        client
            .write_frame(KIND_AUTH, &SECRET)
            .expect("send authentication");
        assert_eq!(
            client.read_frame().expect("read authentication ack"),
            Some((KIND_ACK, vec![0])),
            "host did not acknowledge authentication"
        );
        client
    }

    fn register_texture(&mut self) {
        self.register_texture_as(TEXTURE_ID, 1, 1, EDGE, EDGE);
    }

    /// Registers one kind 8 texture with an explicit wire format/usage code.
    fn register_texture_as(&mut self, id: u64, format: u8, usage: u8, width: u32, height: u32) {
        let mut payload = [0u8; TEXTURE_REG_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&id.to_le_bytes());
        payload[8..12].copy_from_slice(&GENERATION.to_le_bytes());
        payload[12] = 1; // TextureDimension::D2
        payload[13] = format;
        payload[14] = usage;
        payload[15] = 1; // sample count
        payload[16..20].copy_from_slice(&width.to_le_bytes());
        payload[20..24].copy_from_slice(&height.to_le_bytes());
        payload[24..28].copy_from_slice(&1u32.to_le_bytes());
        payload[28..30].copy_from_slice(&1u16.to_le_bytes());
        payload[30..32].copy_from_slice(&1u16.to_le_bytes());
        self.write_frame(KIND_TEXTURE_REG, &payload)
            .expect("send texture registration");
        assert_eq!(
            self.read_frame().expect("read texture ack"),
            Some((KIND_ACK, vec![0])),
            "host rejected texture {id} registration"
        );
    }

    fn register_surface(&mut self) {
        self.register_surface_as(SURFACE_ID, TEXTURE_ID);
    }

    fn register_surface_as(&mut self, surface_id: u64, texture_id: u64) {
        let mut payload = [0u8; SURFACE_REG_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&surface_id.to_le_bytes());
        payload[8..].copy_from_slice(&texture_id.to_le_bytes());
        self.write_frame(KIND_SURFACE_REG, &payload)
            .expect("send surface registration");
        assert_eq!(
            self.read_frame().expect("read surface ack"),
            Some((KIND_ACK, vec![0])),
            "host rejected surface registration"
        );
    }

    fn submit(&mut self, sequence: u32, clear: [f64; 4]) -> Reply {
        self.submit_commands(
            sequence,
            vec![
                CommandV3::BeginRenderPass(BeginRenderPass {
                    colors: vec![ColorAttachment {
                        allocation_id: TEXTURE_ID,
                        generation: GENERATION,
                        mip_level: 0,
                        slice: 0,
                        load: LoadOp::Clear,
                        store: StoreOp::Store,
                        clear_color: clear,
                    }],
                    depth_stencil: None,
                }),
                CommandV3::EndRenderPass,
                CommandV3::Present(Present {
                    surface_id: SURFACE_ID,
                    sync_interval: 1,
                }),
            ],
        )
    }

    fn submit_commands(&mut self, sequence: u32, commands: Vec<CommandV3>) -> Reply {
        let submission = SubmissionV3 {
            sequence,
            list: None,
            commands,
        };
        let frame = encode_submission_v3(&submission).expect("encode v3 submission");
        let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
        payload.extend_from_slice(&V3_REQUEST_MAGIC);
        payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
        payload.extend_from_slice(&sequence.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        payload.extend_from_slice(&frame);
        self.write_frame(KIND_SUBMISSION_V3, &payload)
            .expect("send v3 submission");
        let (kind, reply) = self
            .read_frame()
            .expect("read v3 reply")
            .expect("v3 reply frame");
        assert_eq!(kind, KIND_SUBMISSION_V3, "unexpected reply kind");
        victory_metal::wire::decode_reply(&reply).expect("decode v3 reply")
    }

    fn write_frame(&mut self, kind: u8, payload: &[u8]) -> std::io::Result<()> {
        write_frame(&mut self.stream, kind, payload)
    }

    fn read_frame(&mut self) -> std::io::Result<Option<(u8, Vec<u8>)>> {
        read_frame(&mut self.stream)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

/// Runs two identical cleared frames through a host and returns its telemetry.
fn run_two_frames(share: &TempDir, extra: &[&str], window_env: Option<&str>) -> String {
    let telemetry = share.path().join("telemetry.json");
    let telemetry_text = telemetry.to_string_lossy().into_owned();
    let mut args = vec!["--aaa-frame", "--telemetry-json", telemetry_text.as_str()];
    args.extend_from_slice(extra);
    let host = Host::spawn(share, &args, window_env);

    let mut client = Client::connect(host.port);
    client.register_texture();
    client.register_surface();

    for (sequence, clear) in [(1u32, [1.0, 0.0, 0.0, 1.0]), (2u32, [0.0, 1.0, 0.0, 1.0])] {
        let reply = client.submit(sequence, clear);
        assert_eq!(
            reply.status,
            ReplyStatus::Ok,
            "frame {sequence} was refused: {reply:?}"
        );
    }
    drop(client);

    let output = host.wait();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // A missing window server is the documented skip path.
        if stderr.contains("no window server session") {
            eprintln!("SKIP: the host reported no window server session:\n{stderr}");
            return String::new();
        }
        panic!(
            "host failed: {:?}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            stderr
        );
    }
    std::fs::read_to_string(&telemetry).expect("telemetry JSON was written")
}

/// Extracts a numeric JSON field emitted by the host telemetry writer.
fn number_field(json: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|character: char| !(character.is_ascii_digit() || character == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Reads the numeric pipeline id the host published under `name`.
fn read_pipeline_id(path: &Path) -> u64 {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    text.trim()
        .strip_prefix("id=")
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or_else(|| panic!("{} does not hold an id: {text:?}", path.display()))
}

/// Builds the AAA frame-gate draw: two RGBA8 targets plus D32F, a full-target
/// triangle from the host's built-in pipeline, and a `Present` of colour 0.
fn drawn_frame(pipeline: u64, width: u32, height: u32) -> Vec<CommandV3> {
    drawn_frame_into(pipeline, width, height, TEXTURE_ID, SURFACE_ID)
}

/// [`drawn_frame`] with an explicit pass target and presented surface.
///
/// The pass's first colour attachment is `target`, and the frame presents
/// `surface`. A test can therefore draw into one descriptor-identical object
/// and present another, which is exactly the identity split the present path
/// must resolve to the drawn object.
fn drawn_frame_into(
    pipeline: u64,
    width: u32,
    height: u32,
    target: u64,
    surface: u64,
) -> Vec<CommandV3> {
    // The built-in `vm_frame_vertex` reads two RGBA tints from root constants.
    let tints: [f32; 8] = [1.0, 128.0 / 255.0, 0.0, 1.0, 0.0, 128.0 / 255.0, 1.0, 1.0];
    let mut constants = Vec::with_capacity(32);
    for tint in tints {
        constants.extend_from_slice(&tint.to_le_bytes());
    }
    vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                ColorAttachment {
                    allocation_id: target,
                    generation: GENERATION,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                },
                ColorAttachment {
                    allocation_id: COLOR1_ID,
                    generation: GENERATION,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                },
            ],
            depth_stencil: Some(DepthStencilAttachment {
                allocation_id: DEPTH_ID,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
                depth_load: LoadOp::Clear,
                depth_store: StoreOp::Store,
                clear_depth: 1.0,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                clear_stencil: 0,
            }),
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: pipeline,
        }),
        CommandV3::SetViewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(width),
            height: f64::from(height),
            near: 0.0,
            far: 1.0,
        }),
        CommandV3::SetScissor(ScissorRect {
            x: 0,
            y: 0,
            width,
            height,
        }),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::SetRootConstants(RootConstantsV3 {
            slot: 0,
            bytes: constants,
        }),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
        CommandV3::Present(Present {
            surface_id: surface,
            sync_interval: 1,
        }),
    ]
}

#[test]
fn native_window_presents_a_drawn_backbuffer_larger_than_the_window() {
    if !window_server_available() {
        eprintln!(
            "SKIP: no window server session (CGSessionCopyCurrentDictionary is null); \
             drawn native window frame not exercised"
        );
        return;
    }

    let share = TempDir::new("drawn");
    let telemetry = share.path().join("telemetry.json");
    let telemetry_text = telemetry.to_string_lossy().into_owned();
    let args = vec!["--aaa-frame", "--telemetry-json", telemetry_text.as_str()];
    // The window is smaller than the back-buffer, exactly the real boot shape
    // (3840x2160 title surface in a 1280x720 window). The probe samples the
    // presented drawable after the blit.
    let host = Host::spawn_with_env(
        &share,
        &args,
        &[
            ("VICTORY_METAL_WINDOW", "64x64"),
            ("VICTORY_METAL_FRAME_PROBE", "1"),
        ],
    );
    let pipeline = read_pipeline_id(&share.path().join(GRAPHICS_PIPELINE_FILE));

    let mut client = Client::connect(host.port);
    // Wire format 0 is RGBA8Unorm, the built-in frame-gate colour format;
    // wire format 7 is D32Float and usage 2 is a depth attachment.
    client.register_texture_as(TEXTURE_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
    client.register_texture_as(COLOR1_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
    client.register_texture_as(DEPTH_ID, 7, 2, DRAWN_WIDTH, DRAWN_HEIGHT);
    client.register_surface_as(SURFACE_ID, TEXTURE_ID);

    let reply = client.submit_commands(1, drawn_frame(pipeline, DRAWN_WIDTH, DRAWN_HEIGHT));
    assert_eq!(
        reply.status,
        ReplyStatus::Ok,
        "the drawn frame was refused: {reply:?}"
    );
    drop(client);

    let output = host.wait();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("no window server session") {
            eprintln!("SKIP: the host reported no window server session:\n{stderr}");
            return;
        }
        panic!(
            "host failed: {:?}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            stderr
        );
    }
    let json = std::fs::read_to_string(&telemetry).expect("telemetry JSON was written");

    assert_eq!(
        number_field(&json, "window_drawables"),
        Some(1),
        "the window must acquire one drawable: {json}"
    );
    assert_eq!(
        number_field(&json, "window_presented"),
        Some(1),
        "the window must present one drawable: {json}"
    );
    assert!(
        json.contains("\"window_error\": null"),
        "the window path reported an error: {json}"
    );
    assert!(
        json.contains("\"fallback_clear\": false"),
        "the presented frame is still the diagnostic clear: {json}"
    );
    assert_eq!(
        number_field(&json, "sampled_frames"),
        Some(1),
        "the frame probe must sample the presented frame: {json}"
    );
    let non_black = number_field(&json, "non_black_pixels").expect("probe non_black_pixels");
    assert!(
        non_black > 0,
        "the drawn back-buffer did not reach the presented surface: {json}"
    );
    assert_ne!(
        number_field(&json, "present_checksum"),
        Some(-1),
        "the present texture readback failed: {json}"
    );

    eprintln!(
        "native window drawn gate: surface={DRAWN_WIDTH}x{DRAWN_HEIGHT} window=64x64 \
         fallback_clear=false non_black_pixels={non_black} pipeline={pipeline}"
    );
}

#[test]
fn native_window_acquires_and_presents_two_frames_matching_headless() {
    if !window_server_available() {
        eprintln!(
            "SKIP: no window server session (CGSessionCopyCurrentDictionary is null); \
             native window presenter not exercised"
        );
        return;
    }

    let headless_share = TempDir::new("headless");
    let headless = run_two_frames(&headless_share, &[], None);
    if headless.is_empty() {
        eprintln!("SKIP: headless host lost the window server before the test");
        return;
    }
    assert!(
        headless.contains("\"d3dmetal_loaded\": false"),
        "headless run reported D3DMetal loaded: {headless}"
    );
    let headless_checksum = number_field(&headless, "present_checksum")
        .filter(|value| *value != -1)
        .expect("headless telemetry must contain a present checksum");
    assert!(
        headless.contains("\"d3dmetal_images\": []"),
        "headless run mapped a D3DMetal image: {headless}"
    );

    let window_share = TempDir::new("window");
    let windowed = run_two_frames(&window_share, &["--window"], Some("64x64"));
    if windowed.is_empty() {
        eprintln!("SKIP: no window server session after the headless run");
        return;
    }

    assert_eq!(
        number_field(&windowed, "window_drawables"),
        Some(2),
        "the window must acquire exactly two drawables: {windowed}"
    );
    assert_eq!(
        number_field(&windowed, "window_presented"),
        Some(2),
        "the window must present exactly two drawables: {windowed}"
    );
    assert!(
        windowed.contains("\"window_error\": null"),
        "the window path reported an error: {windowed}"
    );
    assert_eq!(
        number_field(&windowed, "present_checksum"),
        Some(headless_checksum),
        "the presented frame must be the same frame as headless: {windowed}"
    );
    assert!(
        windowed.contains("\"d3dmetal_loaded\": false"),
        "windowed run reported D3DMetal loaded: {windowed}"
    );
    assert!(
        windowed.contains("\"d3dmetal_images\": []"),
        "windowed run mapped a D3DMetal image: {windowed}"
    );

    eprintln!(
        "native window gate: drawables=2 presented=2 checksum={headless_checksum} \
         (headless == windowed), D3DMetal images=0"
    );
}

/// Runs one drawn frame through the windowed host with the frame probe on and
/// returns its telemetry JSON.
///
/// `register` declares the textures/surfaces the frame uses and `commands`
/// builds the submission from the host's published pipeline id. `draw_trace`
/// arms the bounded draw tracer, which is the only source of "which object did
/// a pass write" the present path accepts. An empty string means the documented
/// window-server skip.
fn run_drawn_window_frame(
    tag: &str,
    draw_trace: bool,
    register: impl FnOnce(&mut Client),
    commands: impl FnOnce(u64) -> Vec<CommandV3>,
) -> String {
    if !window_server_available() {
        eprintln!(
            "SKIP: no window server session (CGSessionCopyCurrentDictionary is null); \
             {tag} frame not exercised"
        );
        return String::new();
    }

    let share = TempDir::new(tag);
    let telemetry = share.path().join("telemetry.json");
    let telemetry_text = telemetry.to_string_lossy().into_owned();
    let args = vec!["--aaa-frame", "--telemetry-json", telemetry_text.as_str()];
    let mut envs = vec![
        ("VICTORY_METAL_WINDOW", "64x64"),
        ("VICTORY_METAL_FRAME_PROBE", "1"),
    ];
    if draw_trace {
        // The tracer must be armed in the host process: the pass target is the
        // evidence the present path resolves the presented object with.
        envs.push(("VICTORY_METAL_DRAW_TRACE", "1"));
    }
    let host = Host::spawn_with_env(&share, &args, &envs);
    let pipeline = read_pipeline_id(&share.path().join(GRAPHICS_PIPELINE_FILE));

    let mut client = Client::connect(host.port);
    register(&mut client);
    let reply = client.submit_commands(1, commands(pipeline));
    assert_eq!(reply.status, ReplyStatus::Ok, "frame was refused: {reply:?}");
    drop(client);

    let output = host.wait();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("no window server session") {
            eprintln!("SKIP: the host reported no window server session:\n{stderr}");
            return String::new();
        }
        panic!(
            "host failed: {:?}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            stderr
        );
    }
    std::fs::read_to_string(&telemetry).expect("telemetry JSON was written")
}

/// The presented surface is a second, descriptor-identical object that no pass
/// wrote; the pass wrote another registered surface of the same shape.
///
/// The window must sample the object the pass actually drew (the exact uniform
/// colour the frame-gate fragment writes), and the trace must name that
/// resolution with `surface_path_pass_target`. Before the resolution this frame
/// was counted as `surface_path_unrelated` and the window showed the undrawn
/// surface's bytes.
#[test]
fn native_window_samples_the_pass_target_object_not_an_undrawn_surface() {
    let json = run_drawn_window_frame(
        "pass-target",
        true,
        |client| {
            // The presented surface, the drawn object, the second MRT colour and
            // the depth target all share one 128x96 RGBA8 shape.
            client.register_texture_as(TEXTURE_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_texture_as(PASS_TARGET_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_texture_as(COLOR1_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_texture_as(DEPTH_ID, 7, 2, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_surface_as(SURFACE_ID, TEXTURE_ID);
            // The drawn object is one of the host's registered surfaces, so the
            // present path can adopt it without fabricating an alias.
            client.register_surface_as(PASS_TARGET_SURFACE_ID, PASS_TARGET_ID);
        },
        |pipeline| {
            drawn_frame_into(
                pipeline,
                DRAWN_WIDTH,
                DRAWN_HEIGHT,
                PASS_TARGET_ID,
                SURFACE_ID,
            )
        },
    );
    if json.is_empty() {
        return;
    }

    assert_eq!(
        number_field(&json, "window_drawables"),
        Some(1),
        "the window must acquire one drawable: {json}"
    );
    assert_eq!(
        number_field(&json, "window_presented"),
        Some(1),
        "the window must present one drawable: {json}"
    );
    assert!(
        json.contains("\"window_error\": null"),
        "the window path reported an error: {json}"
    );
    // The resolution: the pass target was adopted, so the frame is a pass-target
    // frame and not an unrelated-allocation frame.
    assert_eq!(
        number_field(&json, "surface_path_pass_target"),
        Some(1),
        "the presented frame must be the pass target: {json}"
    );
    assert_eq!(
        number_field(&json, "surface_path_unrelated"),
        Some(0),
        "a pass wrote a same-shape surface this frame, so no frame may be \
         unrelated: {json}"
    );
    assert_eq!(
        number_field(&json, "surface_path_alias_missing"),
        Some(0),
        "the alias was resolved, not missing: {json}"
    );
    assert_eq!(
        number_field(&json, "composition_copies"),
        Some(0),
        "the resolution is an alias, not a copy: {json}"
    );

    // Exact pixel proof: the drawable holds the colour the fragment wrote into
    // the pass target. The undrawn surface was cleared to transparent black, so
    // any pixel read is the adopted object's content.
    assert_eq!(
        number_field(&json, "sampled_frames"),
        Some(1),
        "the frame probe must sample the presented drawable: {json}"
    );
    let non_black = number_field(&json, "non_black_pixels").expect("probe non_black_pixels");
    let probe_pixels = number_field(&json, "probe_pixels").expect("probe probe_pixels");
    assert_eq!(
        non_black, probe_pixels,
        "every sampled pixel must be the drawn colour: {json}"
    );
    assert!(
        json.contains("\"uniform_color\": \"swatch_ff8000\""),
        "the presented drawable must hold the exact drawn colour (255,128,0): {json}"
    );
    assert_ne!(
        number_field(&json, "nonzero_bytes"),
        Some(0),
        "the sampled drawable must not be an all-zero surface: {json}"
    );
    assert!(
        json.contains("\"fallback_clear\": false"),
        "the drawn object reached the drawable, not the diagnostic clear: {json}"
    );

    eprintln!(
        "native window pass-target gate: surface={DRAWN_WIDTH}x{DRAWN_HEIGHT} \
         presented=1 surface_path_pass_target=1 surface_path_unrelated=0 \
         non_black_pixels={non_black} uniform_color=swatch_ff8000"
    );
}

/// Negative control for the resolution: the pass wrote a descriptor-identical
/// object that is *not* one of the host's registered surfaces.
///
/// Such an object cannot be shown without fabricating an alias, so the frame
/// presents nothing (the diagnostic clear) and stays counted as
/// `surface_path_unrelated`. The probe must never read the foreign object's
/// drawn colour.
#[test]
fn native_window_never_samples_an_unrelated_pass_target() {
    let json = run_drawn_window_frame(
        "unrelated",
        true,
        |client| {
            client.register_texture_as(TEXTURE_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            // The pass target is registered as a texture only: it is not a
            // present surface, so it has no alias the present path may use.
            client.register_texture_as(PASS_TARGET_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_texture_as(COLOR1_ID, 0, 1, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_texture_as(DEPTH_ID, 7, 2, DRAWN_WIDTH, DRAWN_HEIGHT);
            client.register_surface_as(SURFACE_ID, TEXTURE_ID);
        },
        |pipeline| {
            drawn_frame_into(
                pipeline,
                DRAWN_WIDTH,
                DRAWN_HEIGHT,
                PASS_TARGET_ID,
                SURFACE_ID,
            )
        },
    );
    if json.is_empty() {
        return;
    }

    assert_eq!(
        number_field(&json, "window_presented"),
        Some(1),
        "the window must present one (cleared) drawable: {json}"
    );
    assert!(
        json.contains("\"window_error\": null"),
        "the window path reported an error: {json}"
    );
    assert_eq!(
        number_field(&json, "surface_path_unrelated"),
        Some(1),
        "the unrelated-allocation path stays counted: {json}"
    );
    assert_eq!(
        number_field(&json, "surface_path_pass_target"),
        Some(0),
        "a foreign object must never be presented as the pass target: {json}"
    );
    assert!(
        json.contains("\"surface_object_mismatch_reason\": \"surface-is-unrelated-allocation\""),
        "the split is named, not folded into success: {json}"
    );

    // Nothing was sampled: the drawable is the diagnostic clear, so no pixel of
    // the foreign object is read.
    assert_eq!(
        number_field(&json, "sampled_frames"),
        Some(1),
        "the frame probe must sample the presented drawable: {json}"
    );
    assert_eq!(
        number_field(&json, "non_black_pixels"),
        Some(0),
        "a foreign allocation's pixels must not reach the drawable: {json}"
    );
    assert!(
        json.contains("\"uniform_color\": \"swatch_0d1a33\""),
        "the drawable must hold the diagnostic clear, not the drawn colour: {json}"
    );
    assert!(
        json.contains("\"fallback_clear\": true"),
        "the frame presented nothing: {json}"
    );

    eprintln!(
        "native window unrelated gate: surface={DRAWN_WIDTH}x{DRAWN_HEIGHT} presented=1 \
         surface_path_unrelated=1 surface_path_pass_target=0 non_black_pixels=0"
    );
}
