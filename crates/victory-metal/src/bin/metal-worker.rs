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
use std::collections::HashMap;

#[cfg(target_os = "macos")]
use victory_metal::ipc_host::{BatchExecutor, HostSecret, IpcHost};
#[cfg(target_os = "macos")]
use victory_metal::macos::render::PixelFormat;
#[cfg(target_os = "macos")]
use victory_metal::wire::{Command, Reply, ReplyStatus, Submission};
#[cfg(target_os = "macos")]
use victory_metal::{DirectMetalTransport, MetalError, MetalTransport, VmBufferId, VmPipelineId};

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

/// Synthetic graphics pipeline for the guest-to-native frame gate.
///
/// The guest writes the triangle into its own shared allocation and sends
/// `SetPipeline`, `RootConstants` (the flat colour at vertex buffer 1) and
/// `DrawIndexed` (indices at binding 0, positions at binding 1). The vertex
/// shader reads positions from `[[buffer(0)]]`, exactly the device's documented
/// graphics convention.
#[cfg(target_os = "macos")]
const FRAME_GFX_VERTEX: &str = "vm_frame_vertex";
#[cfg(target_os = "macos")]
const FRAME_GFX_FRAGMENT: &str = "vm_frame_fragment";
#[cfg(target_os = "macos")]
const FRAME_GFX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float4 color;
};

vertex VOut vm_frame_vertex(uint vid [[vertex_id]],
                            constant float2 *positions [[buffer(0)]],
                            constant float4 &color [[buffer(1)]]) {
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.color = color;
    return out;
}

fragment float4 vm_frame_fragment(VOut in [[stage_in]]) {
    return in.color;
}
"#;

/// Edge of the executor's shared host render target, in pixels.
#[cfg(target_os = "macos")]
const FRAME_TARGET_EDGE: usize = 64;
/// Row-major index of the target's centre pixel.
#[cfg(target_os = "macos")]
const FRAME_CENTRE_PIXEL: usize = (FRAME_TARGET_EDGE / 2) * FRAME_TARGET_EDGE + FRAME_TARGET_EDGE / 2;

const USAGE: &str = "\
usage: metal-worker --secret-hex <64 hex chars> --share-dir <private dir> [options]

  --secret-hex <hex>         exactly 32 secret bytes as 64 hex characters
  --secret-file <path>       raw 32 bytes, or 64 hex characters with whitespace
  --share-dir <dir>          launcher-controlled private directory
  --port <port>              loopback port, 0 selects an ephemeral port (default 0)
  --bootstrap-compute        compile and publish the synthetic e2e pipeline
  --frame-gate               publish the compute and graphics pipelines and
                             report each observed submission (implies the
                             bootstrap compute pipeline)
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
    frame_gate: bool,
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
    if args.frame_gate {
        // The frame gate publishes both pipelines the guest may name: the
        // existing bootstrap compute kernel and the synthetic graphics pair.
        let compute = transport.compile_msl(BOOTSTRAP_MSL, BOOTSTRAP_ENTRY)?;
        println!("VICTORY_PIPELINE_ID={}", compute.0);
        let graphics = transport.register_graphics_pipeline(
            FRAME_GFX_MSL,
            FRAME_GFX_VERTEX,
            FRAME_GFX_FRAGMENT,
            PixelFormat::Rgba8Unorm,
        )?;
        println!("VICTORY_GRAPHICS_PIPELINE_ID={}", graphics.0);
    } else if args.bootstrap_compute {
        let pipeline = transport.compile_msl(BOOTSTRAP_MSL, BOOTSTRAP_ENTRY)?;
        println!("VICTORY_PIPELINE_ID={}", pipeline.0);
    }
    io::stdout().flush()?;

    if args.frame_gate {
        let executor = FrameGateExecutor::new(transport, args.share_dir.clone());
        let host = IpcHost::new(args.secret, &args.share_dir, executor)?
            .with_io_timeout(args.io_timeout);
        serve(host, args.port, args.accept_timeout)
    } else {
        let host = IpcHost::new(args.secret, &args.share_dir, transport)?
            .with_io_timeout(args.io_timeout);
        serve(host, args.port, args.accept_timeout)
    }
}

/// Binds the loopback listener, accepts the single peer and reports the
/// session summary. Shared by the existing modes and `--frame-gate`.
#[cfg(target_os = "macos")]
fn serve<E: BatchExecutor>(
    mut host: IpcHost<E>,
    port: u16,
    accept_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
    let local = listener.local_addr()?;
    if !local.ip().is_loopback() {
        return Err("listener did not bind to loopback".into());
    }

    println!("VICTORY_IPC_PORT={}", local.port());
    println!("VICTORY_IPC_LISTEN={local}");
    println!("VICTORY_IPC_READY=1");
    io::stdout().flush()?;

    listener.set_nonblocking(true)?;
    let stream = accept_bounded(&listener, accept_timeout)?;
    stream.set_nonblocking(false)?;

    let summary = host.serve_once(stream)?;
    host.release_all();
    eprintln!(
        "metal-worker: session complete submissions={} allocations={} bytes_mapped={}",
        summary.submissions, summary.allocations, summary.bytes_mapped
    );
    Ok(())
}

/// Executor wrapper used by `--frame-gate`.
///
/// It forwards every native operation to the real [`DirectMetalTransport`] and,
/// after an accepted submission, reports one machine-readable line:
///
/// ```text
/// kind=compute value=<dword>                 # first dword of the bound allocation
/// kind=draw centre=<r,g,b,a> pixels=<n>      # n = pixels matching the centre
/// ```
///
/// The compute value is read from the guest's own backing file (the host maps
/// it `MAP_SHARED`), so it is what the GPU actually wrote, not a prediction. The
/// draw centre comes from the transport's shared 64x64 RGBA8 host target.
#[cfg(target_os = "macos")]
struct FrameGateExecutor {
    transport: DirectMetalTransport,
    share_dir: PathBuf,
}

#[cfg(target_os = "macos")]
impl FrameGateExecutor {
    fn new(transport: DirectMetalTransport, share_dir: PathBuf) -> Self {
        Self {
            transport,
            share_dir,
        }
    }

    /// Reads the first bound dword of a compute submission from its backing
    /// file after the GPU batch completed.
    fn compute_value(&self, submission: &Submission) -> Result<u32, MetalError> {
        let binding = submission
            .commands
            .iter()
            .find_map(|command| match command {
                Command::Dispatch1d(dispatch) => dispatch.bindings.first(),
                Command::Dispatch3d(dispatch) => dispatch.bindings.first(),
                _ => None,
            })
            .ok_or_else(|| {
                MetalError::InternalError(
                    "compute submission has no bound allocation to observe".to_string(),
                )
            })?;
        let path = self
            .share_dir
            .join(format!("{:016x}.bin", binding.allocation_id));
        let bytes = fs::read(&path).map_err(|error| {
            MetalError::InternalError(format!(
                "cannot read compute backing {}: {error}",
                path.display()
            ))
        })?;
        let start = usize::try_from(binding.offset)
            .map_err(|_| MetalError::InvalidArg("compute binding offset overflows usize"))?;
        let word = bytes.get(start..start + 4).ok_or_else(|| {
            MetalError::InternalError(
                "compute binding is outside its backing allocation".to_string(),
            )
        })?;
        Ok(u32::from_le_bytes(
            word.try_into().expect("a four-byte slice"),
        ))
    }

    fn report_draw(&self) -> Result<(), MetalError> {
        let pixels = self.transport.host_target_pixels()?;
        let centre = pixels
            .get(FRAME_CENTRE_PIXEL)
            .copied()
            .ok_or_else(|| {
                MetalError::InternalError(
                    "host render target is smaller than the frame gate expects".to_string(),
                )
            })?;
        let matching = pixels.iter().filter(|pixel| **pixel == centre).count();
        println!(
            "kind=draw centre={},{},{},{} pixels={matching}",
            centre[0], centre[1], centre[2], centre[3]
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl BatchExecutor for FrameGateExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        <DirectMetalTransport as BatchExecutor>::page_size(&self.transport)
    }

    fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        BatchExecutor::register_window(&self.transport, ptr, len)
    }

    fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError> {
        BatchExecutor::unregister_window(&self.transport, id)
    }

    fn submit_batch(
        &self,
        submission: &Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let reply = BatchExecutor::submit_batch(&self.transport, submission, allocations)?;
        if reply.status == ReplyStatus::Ok {
            let graphics = submission.commands.iter().any(is_graphics_command);
            if graphics {
                self.report_draw()?;
            } else {
                let value = self.compute_value(submission)?;
                println!("kind=compute value={value}");
            }
            io::stdout()
                .flush()
                .map_err(|error| MetalError::InternalError(format!("stdout flush failed: {error}")))?;
        }
        Ok(reply)
    }

    fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        BatchExecutor::load_pipeline(&self.transport, metallib, entry)
    }
}

#[cfg(target_os = "macos")]
fn is_graphics_command(command: &Command) -> bool {
    matches!(
        command,
        Command::SetPipeline(_)
            | Command::RootConstants(_)
            | Command::DescriptorTable(_)
            | Command::DrawIndexed(_)
            | Command::Draw(_)
            | Command::ResourceBarrier(_)
            | Command::DispatchMesh(_)
    )
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
    let mut frame_gate = false;
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
            "--frame-gate" => frame_gate = true,
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
        frame_gate,
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
