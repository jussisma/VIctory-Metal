//! CPU/loopback tests for the standalone ARM64 `victory-metal-host` daemon.
//!
//! These tests spawn the real binary, connect over loopback and exercise the
//! authentication, `--once` teardown and telemetry contracts. They submit no
//! GPU work: a clean authenticated session with no submissions only queries the
//! Metal device page size, so the suite is safe to run without an exclusive GPU.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

const SECRET: [u8; 32] = [0x5a; 32];
const KIND_AUTH: u8 = 1;
const KIND_ACK: u8 = 128;
const WAIT: Duration = Duration::from_secs(20);

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "victory-host-daemon-{tag}-{}-{}",
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
    fn spawn(share: &TempDir, extra: &[&str]) -> Self {
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
            .stderr(Stdio::piped());
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
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            panic!("no ready banner (exit {status:?}): {banner:?} {stderr}");
        });
        assert_eq!(parse_pid(&banner), Some(child.id()), "banner pid");
        Self {
            child,
            stdout,
            port,
        }
    }

    fn connect(&self) -> TcpStream {
        let stream = TcpStream::connect(("127.0.0.1", self.port)).expect("connect to the host");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("read timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        stream
    }

    fn authenticate(&self, secret: &[u8; 32]) -> TcpStream {
        let mut stream = self.connect();
        write_frame(&mut stream, KIND_AUTH, secret).expect("send authentication");
        assert_eq!(
            read_frame(&mut stream).expect("read the ack"),
            Some((KIND_ACK, vec![0]))
        );
        stream
    }

    fn wait(self) -> std::process::Output {
        let Host {
            mut child, stdout, ..
        } = self;
        drop(stdout);
        wait_bounded(&mut child);
        let mut stdout = Vec::new();
        if let Some(mut pipe) = child.stdout.take() {
            let _ = pipe.read_to_end(&mut stdout);
        }
        let mut stderr = Vec::new();
        if let Some(mut pipe) = child.stderr.take() {
            let _ = pipe.read_to_end(&mut stderr);
        }
        let status = child.wait().expect("wait for the host");
        std::process::Output {
            status,
            stdout,
            stderr,
        }
    }
}

fn wait_bounded(child: &mut Child) {
    let deadline = Instant::now() + WAIT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("the host did not exit within {WAIT:?}");
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(error) => panic!("try_wait failed: {error}"),
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

fn parse_pid(banner: &str) -> Option<u32> {
    banner
        .split_whitespace()
        .find_map(|word| word.strip_prefix("pid="))
        .and_then(|pid| pid.parse().ok())
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

#[test]
fn ephemeral_port_banner_and_once_accepts_one_session() {
    let share = TempDir::new("banner");
    let host = Host::spawn(&share, &[]);
    assert_ne!(host.port, 0, "port 0 must resolve to a real ephemeral port");

    // Correct secret: acknowledged, then a clean disconnect ends the session.
    let stream = host.authenticate(&SECRET);
    drop(stream);

    let output = host.wait();
    assert!(
        output.status.success(),
        "--once must exit 0 after one clean session: {:?} {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn wrong_secret_exits_nonzero_with_one_error_line() {
    let share = TempDir::new("wrong-secret");
    let host = Host::spawn(&share, &[]);

    let mut stream = host.connect();
    let mut wrong = SECRET;
    wrong[0] ^= 1;
    write_frame(&mut stream, KIND_AUTH, &wrong).expect("send the wrong secret");
    assert_eq!(
        read_frame(&mut stream).expect("read the rejection"),
        Some((KIND_ACK, vec![1])),
        "the host must reject the wrong secret explicitly"
    );
    drop(stream);

    let output = host.wait();
    assert!(
        !output.status.success(),
        "a refused session must exit non-zero"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf-8 stderr");
    let lines: Vec<&str> = stderr.lines().collect();
    assert_eq!(lines.len(), 1, "exactly one error line: {stderr:?}");
    assert!(
        lines[0].starts_with("victory-metal-host: error "),
        "unexpected error line: {stderr:?}"
    );
}

#[test]
fn telemetry_json_is_valid_after_a_session() {
    let share = TempDir::new("telemetry");
    let telemetry = share.path().join("telemetry.json");
    let telemetry_text = telemetry.to_string_lossy().into_owned();
    let host = Host::spawn(&share, &["--telemetry-json", &telemetry_text]);

    let stream = host.authenticate(&SECRET);
    drop(stream);

    let output = host.wait();
    assert!(
        output.status.success(),
        "the telemetry session must succeed: {:?} {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let json = std::fs::read_to_string(&telemetry).expect("telemetry JSON was written");
    assert!(
        json.contains("\"sessions\": 1"),
        "telemetry should count the session: {json}"
    );
    assert!(
        json.contains("\"submissions\": 0"),
        "no submissions were sent: {json}"
    );
    assert!(
        json.contains("\"d3dmetal_fallback_count\": 0"),
        "the native path has no D3DMetal fallback: {json}"
    );
    assert!(
        json.contains("\"d3dmetal_loaded\": false"),
        "the daemon must not load D3DMetal: {json}"
    );

    // Parse with the platform Python so this is a real JSON validity check,
    // not a substring approximation.
    let status = Command::new("python3")
        .arg("-c")
        .arg("import json,sys; json.load(open(sys.argv[1]))")
        .arg(&telemetry)
        .status()
        .expect("run python3");
    assert!(status.success(), "telemetry is not valid JSON: {json}");
}
