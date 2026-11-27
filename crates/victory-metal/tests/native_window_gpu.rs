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
    BeginRenderPass, ColorAttachment, CommandV3, LoadOp, PROTOCOL_VERSION_V3, Present, StoreOp,
    SubmissionV3, encode_submission_v3,
};

const SECRET: [u8; 32] = [0x5a; 32];
const WAIT: Duration = Duration::from_secs(30);
const TEXTURE_ID: u64 = 7;
const SURFACE_ID: u64 = 11;
const GENERATION: u32 = 1;
const EDGE: u32 = 64;

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
        if let Some(value) = window_env {
            command.env("VICTORY_METAL_WINDOW", value);
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
        let mut payload = [0u8; TEXTURE_REG_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&TEXTURE_ID.to_le_bytes());
        payload[8..12].copy_from_slice(&GENERATION.to_le_bytes());
        payload[12] = 1; // TextureDimension::D2
        payload[13] = 1; // TextureFormat::Bgra8Unorm
        payload[14] = 1; // TextureUsage::color_attachment
        payload[15] = 1; // sample count
        payload[16..20].copy_from_slice(&EDGE.to_le_bytes());
        payload[20..24].copy_from_slice(&EDGE.to_le_bytes());
        payload[24..28].copy_from_slice(&1u32.to_le_bytes());
        payload[28..30].copy_from_slice(&1u16.to_le_bytes());
        payload[30..32].copy_from_slice(&1u16.to_le_bytes());
        self.write_frame(KIND_TEXTURE_REG, &payload)
            .expect("send texture registration");
        assert_eq!(
            self.read_frame().expect("read texture ack"),
            Some((KIND_ACK, vec![0])),
            "host rejected the texture registration"
        );
    }

    fn register_surface(&mut self) {
        let mut payload = [0u8; SURFACE_REG_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&SURFACE_ID.to_le_bytes());
        payload[8..].copy_from_slice(&TEXTURE_ID.to_le_bytes());
        self.write_frame(KIND_SURFACE_REG, &payload)
            .expect("send surface registration");
        assert_eq!(
            self.read_frame().expect("read surface ack"),
            Some((KIND_ACK, vec![0])),
            "host rejected the surface registration"
        );
    }

    fn submit(&mut self, sequence: u32, clear: [f64; 4]) -> Reply {
        let submission = SubmissionV3 {
            sequence,
            list: None,
            commands: vec![
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
