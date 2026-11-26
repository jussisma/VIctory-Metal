//! Native ARM64 headless Metal host worker for the local IPC end-to-end test.
//!
//! This binary is the launcher-facing entry point for [`victory_metal::ipc_host`].
//! It binds only `127.0.0.1`, serves exactly one authenticated peer and exits.
//! The 32-byte secret comes from the harness through `--secret-hex` or a
//! private `--secret-file`; it is never printed.
//!
//! With `--bootstrap-compute` the worker compiles one deterministic MSL kernel
//! and publishes its pipeline ID on stdout as `VICTORY_PIPELINE_ID=<id>`.
//! That pipeline exists only for synthetic end-to-end validation; it is not
//! part of the renderer's production shader design.
//!
//! Startup markers on stdout (flushed before accepting):
//!
//! ```text
//! VICTORY_PIPELINE_ID=<id>        # only with --bootstrap-compute
//! VICTORY_IPC_PORT=<port>
//! VICTORY_IPC_LISTEN=127.0.0.1:<port>
//! VICTORY_IPC_READY=1
//! ```
//!
//! The native host exists only on the ARM64 macOS host. Windows is still built
//! for `--all-targets`, so this target compiles there only to keep that build
//! honest: the macOS-only helpers are then intentionally unused, and the
//! explicit `main` below reports that the worker is unsupported.

#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::env;
use std::fs;
use std::io::{self, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use victory_metal::ipc_host::{HostSecret, IpcHost};
#[cfg(target_os = "macos")]
use victory_metal::{DirectMetalTransport, MetalTransport};

/// Deterministic, synthetic compute kernel used by the e2e harness.
const BOOTSTRAP_ENTRY: &str = "vm_ipc_bootstrap_mul_add";
const BOOTSTRAP_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vm_ipc_bootstrap_mul_add(device uint *data [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 3u + 7u;
}
"#;

const USAGE: &str = "\
usage: metal-worker --secret-hex <64 hex chars> --share-dir <private dir> [options]

  --secret-hex <hex>         exactly 32 secret bytes as 64 hex characters
  --secret-file <path>       raw 32 bytes, or 64 hex characters with whitespace
  --share-dir <dir>          launcher-controlled private directory
  --port <port>              loopback port, 0 selects an ephemeral port (default 0)
  --bootstrap-compute        compile and publish the synthetic e2e pipeline
  --io-timeout-secs <n>      per-read/per-write socket timeout (default 30)
  --accept-timeout-secs <n>  wait for the single peer (default 60)
  -h, --help

The worker binds 127.0.0.1 only and serves one peer synchronously.
";

#[cfg(target_os = "macos")]
struct Args {
    port: u16,
    secret: HostSecret,
    share_dir: PathBuf,
    bootstrap_compute: bool,
    io_timeout: Duration,
    accept_timeout: Duration,
}

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    match parse_args() {
        Ok(None) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("metal-worker: {error}");
                ExitCode::from(1)
            }
        },
        Err(error) => {
            eprintln!("metal-worker: {error}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// The native host worker has no non-macOS implementation; fail explicitly
/// instead of pretending this target could serve a session.
#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("metal-worker: the native Metal host is only available on ARM64 macOS");
    ExitCode::from(2)
}

#[cfg(target_os = "macos")]
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let transport = DirectMetalTransport::new()?;
    if args.bootstrap_compute {
        let pipeline = transport.compile_msl(BOOTSTRAP_MSL, BOOTSTRAP_ENTRY)?;
        println!("VICTORY_PIPELINE_ID={}", pipeline.0);
        io::stdout().flush()?;
    }

    let mut host =
        IpcHost::new(args.secret, &args.share_dir, transport)?.with_io_timeout(args.io_timeout);

    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, args.port))?;
    let local = listener.local_addr()?;
    if !local.ip().is_loopback() {
        return Err("listener did not bind to loopback".into());
    }

    println!("VICTORY_IPC_PORT={}", local.port());
    println!("VICTORY_IPC_LISTEN={local}");
    println!("VICTORY_IPC_READY=1");
    io::stdout().flush()?;

    listener.set_nonblocking(true)?;
    let stream = accept_bounded(&listener, args.accept_timeout)?;
    stream.set_nonblocking(false)?;

    let summary = host.serve_once(stream)?;
    host.release_all();
    eprintln!(
        "metal-worker: session complete submissions={} allocations={} bytes_mapped={}",
        summary.submissions, summary.allocations, summary.bytes_mapped
    );
    Ok(())
}

/// Accepts the single peer within a bounded window. The listener is
/// non-blocking so a harness that fails to connect cannot hang the worker.
fn accept_bounded(listener: &TcpListener, timeout: Duration) -> io::Result<TcpStream> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, peer)) => {
                if !peer.ip().is_loopback() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "peer is not loopback",
                    ));
                }
                return Ok(stream);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "timed out waiting for the single peer",
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

#[cfg(target_os = "macos")]
fn parse_args() -> Result<Option<Args>, String> {
    let mut port = 0u16;
    let mut secret = None;
    let mut secret_source = None;
    let mut share_dir = None;
    let mut bootstrap_compute = false;
    let mut io_timeout_secs = 30u64;
    let mut accept_timeout_secs = 60u64;

    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "--port" => {
                let value = next_value(&mut args, "--port")?;
                port = value
                    .parse()
                    .map_err(|_| format!("invalid --port value `{value}`"))?;
            }
            "--secret-hex" => {
                let value = next_value(&mut args, "--secret-hex")?;
                secret = Some(
                    HostSecret::from_hex(&value).map_err(|error| error.to_string())?,
                );
                secret_source = Some("--secret-hex");
            }
            "--secret-file" => {
                let value = next_value(&mut args, "--secret-file")?;
                secret = Some(read_secret_file(&value)?);
                secret_source = Some("--secret-file");
            }
            "--share-dir" => {
                share_dir = Some(PathBuf::from(next_value(&mut args, "--share-dir")?));
            }
            "--bootstrap-compute" => bootstrap_compute = true,
            "--io-timeout-secs" => {
                let value = next_value(&mut args, "--io-timeout-secs")?;
                io_timeout_secs = parse_timeout(&value, "--io-timeout-secs")?;
            }
            "--accept-timeout-secs" => {
                let value = next_value(&mut args, "--accept-timeout-secs")?;
                accept_timeout_secs = parse_timeout(&value, "--accept-timeout-secs")?;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    if secret.is_some() && secret_source.is_none() {
        return Err("secret was set without a source".to_string());
    }
    let secret = secret.ok_or("missing --secret-hex or --secret-file")?;
    let share_dir = share_dir.ok_or("missing --share-dir")?;
    if !share_dir.is_dir() {
        return Err(format!(
            "--share-dir `{}` is not an existing directory",
            share_dir.display()
        ));
    }

    Ok(Some(Args {
        port,
        secret,
        share_dir,
        bootstrap_compute,
        io_timeout: Duration::from_secs(io_timeout_secs),
        accept_timeout: Duration::from_secs(accept_timeout_secs),
    }))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_timeout(value: &str, flag: &str) -> Result<u64, String> {
    let seconds = value
        .parse::<u64>()
        .map_err(|_| format!("invalid {flag} value `{value}`"))?;
    if !(1..=3600).contains(&seconds) {
        return Err(format!("{flag} must be between 1 and 3600 seconds"));
    }
    Ok(seconds)
}

#[cfg(target_os = "macos")]
fn read_secret_file(path: &str) -> Result<HostSecret, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read secret file: {error}"))?;
    if bytes.len() == 32 {
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&bytes);
        return Ok(HostSecret::new(secret));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| "secret file is neither 32 raw bytes nor hex text".to_string())?;
    HostSecret::from_hex(text).map_err(|error| error.to_string())
}
