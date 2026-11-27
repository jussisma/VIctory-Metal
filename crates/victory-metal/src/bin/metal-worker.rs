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
//! The native host exists only on the ARM64 macOS host. The Win64 build is the
//! counterpart guest probe: `metal-worker --guest-v3` connects to a running
//! host, registers one real file-backed allocation, and drives Wire v3 frames
//! (`KIND_SUBMISSION_V3`) over the same loopback envelope the production queue
//! will carry. It proves the guest end of the ring, the explicit failure
//! statuses and that the guest process has no D3DMetal module loaded. Without
//! `--guest-v3` the Win64 build still reports that the host is unsupported.

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
use std::sync::Arc;

#[cfg(target_os = "macos")]
use victory_metal::fence_page::FencePageMapping;
use victory_metal::ipc_host::{BatchExecutor, HostSecret, IpcHost};
#[cfg(target_os = "macos")]
use victory_metal::macos::render::PixelFormat;
#[cfg(target_os = "macos")]
use victory_metal::wire::{BufferBinding, Command, Reply, ReplyStatus, Submission};
#[cfg(target_os = "macos")]
use victory_metal::wire_v3::{CommandV3, SubmissionV3};
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
const FRAME_CENTRE_PIXEL: usize =
    (FRAME_TARGET_EDGE / 2) * FRAME_TARGET_EDGE + FRAME_TARGET_EDGE / 2;

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

/// On Win64 the worker is the guest probe (`--guest-v3`); it never serves the
/// host. Any other invocation fails explicitly instead of pretending this
/// target could run a native session.
#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    if env::args().nth(1).as_deref() == Some("--guest-v3") {
        return match guest_v3::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("metal-worker guest: {error}");
                ExitCode::from(1)
            }
        };
    }
    eprintln!(
        "metal-worker: the native Metal host is only available on ARM64 macOS; \
         the Win64 build is the guest probe (--guest-v3)"
    );
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
        let host =
            IpcHost::new(args.secret, &args.share_dir, executor)?.with_io_timeout(args.io_timeout);
        serve(host, args.port, args.accept_timeout)
    } else {
        let host =
            IpcHost::new(args.secret, &args.share_dir, transport)?.with_io_timeout(args.io_timeout);
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
        self.binding_value(binding)
    }

    /// Version 3 counterpart of [`Self::compute_value`].
    fn compute_value_v3(&self, submission: &SubmissionV3) -> Result<u32, MetalError> {
        let binding = submission
            .commands
            .iter()
            .find_map(|command| match command {
                CommandV3::Dispatch1d(dispatch) => dispatch.bindings.first(),
                CommandV3::Dispatch3d(dispatch) => dispatch.bindings.first(),
                _ => None,
            })
            .ok_or_else(|| {
                MetalError::InternalError(
                    "v3 compute submission has no bound allocation to observe".to_string(),
                )
            })?;
        self.binding_value(binding)
    }

    /// Reads the first dword of `binding` from the guest-visible backing file.
    fn binding_value(&self, binding: &BufferBinding) -> Result<u32, MetalError> {
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
        let centre = pixels.get(FRAME_CENTRE_PIXEL).copied().ok_or_else(|| {
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
    fn attach_fence_page(&self, page: Arc<FencePageMapping>) -> Result<bool, MetalError> {
        BatchExecutor::attach_fence_page(&self.transport, page)
    }

    fn detach_fence_page(&self) {
        BatchExecutor::detach_fence_page(&self.transport);
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
            io::stdout().flush().map_err(|error| {
                MetalError::InternalError(format!("stdout flush failed: {error}"))
            })?;
        }
        Ok(reply)
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let reply = BatchExecutor::submit_v3(&self.transport, submission, allocations)?;
        if reply.status == ReplyStatus::Ok {
            let graphics = submission.commands.iter().any(is_graphics_v3_command);
            if graphics {
                self.report_draw()?;
            } else {
                let value = self.compute_value_v3(submission)?;
                println!("kind=compute value={value}");
            }
            io::stdout().flush().map_err(|error| {
                MetalError::InternalError(format!("stdout flush failed: {error}"))
            })?;
        }
        Ok(reply)
    }

    fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        BatchExecutor::load_pipeline(&self.transport, metallib, entry)
    }
}

#[cfg(target_os = "macos")]
fn is_graphics_v3_command(command: &CommandV3) -> bool {
    matches!(
        command,
        CommandV3::BeginRenderPass(_)
            | CommandV3::Draw(_)
            | CommandV3::DrawIndexed(_)
            | CommandV3::DrawInstanced(_)
            | CommandV3::DrawIndexedInstanced(_)
            | CommandV3::DispatchMesh(_)
    )
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
                secret = Some(HostSecret::from_hex(&value).map_err(|error| error.to_string())?);
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

/// Win64 guest probe for the Wire v3 ring.
///
/// This is the largest guest-side slice the current tree can carry: the
/// production `victory-gpu` producer still installs `DisconnectedSink` and
/// `GuestIpcClient` has no v3 encoder, so this binary frames the same
/// `SubmissionV3` bytes with the canonical codec and the frozen kind-5
/// envelope. It registers one real file-backed allocation in the session share
/// directory, executes a compute list on the published pipeline, and asserts
/// the alias-visible result, replay refusal, unknown-allocation refusal and
/// unsupported-opcode refusal. It also enumerates its own PE modules and
/// records that no D3DMetal module is loaded.
#[cfg(not(target_os = "macos"))]
mod guest_v3 {
    use std::ffi::c_void;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use victory_metal::wire::{
        BufferBinding, Dispatch1d, ListIdentity, Reply, ReplyStatus, decode_reply,
    };
    use victory_metal::wire_v3::{
        CommandV3, PROTOCOL_VERSION_V3, SubmissionV3, encode_submission_v3,
    };

    /// Frozen envelope constants shared with `victory_metal::ipc_host`
    /// (`ipc_host` is macOS-only, so the Win64 guest names them itself).
    const KIND_AUTH: u8 = 1;
    const KIND_REGISTER: u8 = 2;
    const KIND_SUBMISSION_V3: u8 = 5;
    const KIND_ACK: u8 = 128;
    const V3_REQUEST_MAGIC: [u8; 4] = *b"V3RQ";
    const V3_REQUEST_HEADER_LEN: usize = 16;
    const REGISTER_PAYLOAD_LEN: usize = 16;
    const SECRET_LEN: usize = 32;
    const ALLOCATION_ID: u64 = 1;
    const GENERATION: u32 = 1;
    const PAGE: usize = 64 * 1024;
    const THREADS: u32 = 64;
    const COMPUTE_BYTES: u64 = THREADS as u64 * 4;

    fn env(name: &str) -> Result<String, String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{name} is not set"))
    }

    fn secret() -> Result<[u8; SECRET_LEN], String> {
        let path = PathBuf::from(env("VICTORY_IPC_SECRET_FILE")?);
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let text = text.trim();
        if text.len() != SECRET_LEN * 2 {
            return Err(format!(
                "the secret file must be {} hex characters",
                SECRET_LEN * 2
            ));
        }
        let mut secret = [0u8; SECRET_LEN];
        for (index, slot) in secret.iter_mut().enumerate() {
            let pair = &text[index * 2..index * 2 + 2];
            *slot = u8::from_str_radix(pair, 16)
                .map_err(|_| "the secret file contains a non-hex digit".to_string())?;
        }
        Ok(secret)
    }

    struct Client {
        stream: TcpStream,
    }

    impl Client {
        fn connect(port: u16, secret: &[u8; SECRET_LEN]) -> Result<Self, String> {
            let stream = TcpStream::connect(("127.0.0.1", port))
                .map_err(|error| format!("connect 127.0.0.1:{port}: {error}"))?;
            stream
                .set_read_timeout(Some(Duration::from_secs(60)))
                .map_err(|error| format!("set read timeout: {error}"))?;
            let mut client = Self { stream };
            client.write_frame(KIND_AUTH, secret)?;
            match client.read_frame()? {
                Some((KIND_ACK, payload)) if payload == [0] => Ok(client),
                other => Err(format!("authentication was not acknowledged: {other:?}")),
            }
        }

        fn register(&mut self, id: u64, len: u64) -> Result<(), String> {
            let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
            payload[..8].copy_from_slice(&id.to_le_bytes());
            payload[8..].copy_from_slice(&len.to_le_bytes());
            self.write_frame(KIND_REGISTER, &payload)?;
            match self.read_frame()? {
                Some((KIND_ACK, payload)) if payload == [0] => Ok(()),
                other => Err(format!("registration was not acknowledged: {other:?}")),
            }
        }

        fn write_frame(&mut self, kind: u8, payload: &[u8]) -> Result<(), String> {
            let mut header = [0u8; 5];
            header[0] = kind;
            header[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
            self.stream
                .write_all(&header)
                .and_then(|()| self.stream.write_all(payload))
                .and_then(|()| self.stream.flush())
                .map_err(|error| format!("write frame: {error}"))
        }

        fn read_frame(&mut self) -> Result<Option<(u8, Vec<u8>)>, String> {
            let mut header = [0u8; 5];
            let mut filled = 0usize;
            while filled < header.len() {
                match self.stream.read(&mut header[filled..]) {
                    Ok(0) if filled == 0 => return Ok(None),
                    Ok(0) => return Err("the host closed inside an envelope".to_string()),
                    Ok(read) => filled += read,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => return Err(format!("read frame: {error}")),
                }
            }
            let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
            let mut payload = vec![0u8; len];
            self.stream
                .read_exact(&mut payload)
                .map_err(|error| format!("read payload: {error}"))?;
            Ok(Some((header[0], payload)))
        }

        fn submit(&mut self, sequence: u32, frame: &[u8]) -> Result<Reply, String> {
            let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
            payload.extend_from_slice(&V3_REQUEST_MAGIC);
            payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
            payload.extend_from_slice(&sequence.to_le_bytes());
            payload.extend_from_slice(&0u16.to_le_bytes());
            payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
            payload.extend_from_slice(frame);
            self.write_frame(KIND_SUBMISSION_V3, &payload)?;
            let (kind, reply) = self
                .read_frame()?
                .ok_or_else(|| "the host closed before the v3 reply".to_string())?;
            if kind != KIND_SUBMISSION_V3 {
                return Err(format!("unexpected reply kind {kind}"));
            }
            decode_reply(&reply).map_err(|error| format!("decode v3 reply: {error}"))
        }
    }

    fn dispatch(pipeline: u64, allocation: u64) -> SubmissionV3 {
        SubmissionV3 {
            sequence: 0,
            list: None,
            commands: vec![CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: pipeline,
                threads: THREADS,
                threads_per_group: THREADS,
                bindings: vec![BufferBinding {
                    allocation_id: allocation,
                    generation: GENERATION,
                    offset: 0,
                    length: COMPUTE_BYTES,
                }],
            })],
        }
    }

    fn alias_words(path: &Path, count: usize) -> Result<Vec<u32>, String> {
        let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        if bytes.len() < count * 4 {
            return Err(format!(
                "{} is {} bytes, expected at least {}",
                path.display(),
                bytes.len(),
                count * 4
            ));
        }
        Ok(bytes[..count * 4]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect())
    }

    type Handle = *mut c_void;

    #[link(name = "psapi")]
    unsafe extern "system" {
        fn EnumProcessModules(
            process: Handle,
            modules: *mut Handle,
            bytes: u32,
            needed: *mut u32,
        ) -> i32;
        fn GetModuleFileNameExW(process: Handle, module: Handle, name: *mut u16, size: u32) -> u32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn GetModuleHandleW(name: *const u16) -> Handle;
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Enumerates the guest process's PE modules through psapi.
    fn loaded_modules() -> Result<Vec<String>, String> {
        // SAFETY: The pseudo-handle names this process and the buffer is sized
        // for 2048 module handles; psapi writes at most `bytes`.
        unsafe {
            let process = GetCurrentProcess();
            let mut handles = vec![std::ptr::null_mut(); 2048];
            let mut needed = 0u32;
            let ok = EnumProcessModules(
                process,
                handles.as_mut_ptr(),
                (handles.len() * std::mem::size_of::<Handle>()) as u32,
                &mut needed,
            );
            if ok == 0 {
                return Err(format!(
                    "EnumProcessModules failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let count = ((needed as usize) / std::mem::size_of::<Handle>()).min(handles.len());
            let mut names = Vec::with_capacity(count);
            for module in &handles[..count] {
                let mut buffer = [0u16; 1024];
                let len = GetModuleFileNameExW(
                    process,
                    *module,
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                );
                if len > 0 {
                    names.push(String::from_utf16_lossy(&buffer[..len as usize]));
                }
            }
            Ok(names)
        }
    }

    fn d3dmetal_loaded() -> Result<(usize, bool), String> {
        let modules = loaded_modules()?;
        let found = modules
            .iter()
            .any(|name| name.to_ascii_lowercase().contains("d3dmetal"));
        // Belt and braces: a named module lookup catches a lazily loaded module
        // whose path psapi did not report.
        // SAFETY: `wide` is NUL-terminated and GetModuleHandleW only reads it.
        let named = unsafe { GetModuleHandleW(wide("d3dmetal.dll").as_ptr()) };
        Ok((modules.len(), found || !named.is_null()))
    }

    pub(super) fn run() -> Result<(), String> {
        let share = PathBuf::from(env("VICTORY_METAL_SHARE_DIR")?);
        let port: u16 = env("VICTORY_IPC_PORT")?
            .parse()
            .map_err(|_| "VICTORY_IPC_PORT is not a number".to_string())?;
        let pipeline: u64 = env("VICTORY_IPC_PIPELINE_ID")?
            .parse()
            .map_err(|_| "VICTORY_IPC_PIPELINE_ID is not a number".to_string())?;
        let secret = secret()?;
        let allocation = share.join(format!("{ALLOCATION_ID:016x}.bin"));

        // One real page-backed allocation the host maps through the session
        // share directory. This is the guest-visible alias the GPU writes.
        fs::write(&allocation, vec![0u8; PAGE])
            .map_err(|error| format!("create {}: {error}", allocation.display()))?;
        {
            let mut bytes = fs::read(&allocation).expect("read the new allocation");
            for index in 0..THREADS as usize {
                let at = index * 4;
                bytes[at..at + 4].copy_from_slice(&(index as u32 + 1).to_le_bytes());
            }
            fs::write(&allocation, &bytes)
                .map_err(|error| format!("write inputs {}: {error}", allocation.display()))?;
        }

        let mut client = Client::connect(port, &secret)?;
        client.register(ALLOCATION_ID, PAGE as u64)?;

        let identity = ListIdentity {
            list_id: 7,
            list_generation: 1,
        };
        let mut compute = dispatch(pipeline, ALLOCATION_ID);
        compute.sequence = 1;
        compute.list = Some(identity);
        let frame = encode_submission_v3(&compute).map_err(|error| error.to_string())?;
        let reply = client.submit(1, &frame)?;
        if reply.status != ReplyStatus::Ok {
            return Err(format!("the compute list was refused: {reply:?}"));
        }
        let words = alias_words(&allocation, THREADS as usize)?;
        for (index, value) in words.iter().enumerate() {
            let expected = (index as u32 + 1) * 3 + 7;
            if *value != expected {
                return Err(format!(
                    "alias dword {index} is {value}, expected {expected}"
                ));
            }
        }

        // Replay of the same list identity: refused, alias unchanged.
        let before = fs::read(&allocation).expect("snapshot the alias");
        let mut replay = compute.clone();
        replay.sequence = 2;
        let frame = encode_submission_v3(&replay).map_err(|error| error.to_string())?;
        let reply = client.submit(2, &frame)?;
        if reply.status != ReplyStatus::AlreadyExecuted {
            return Err(format!("the replay was not refused: {reply:?}"));
        }
        let after = fs::read(&allocation).expect("re-read the alias");
        if after != before {
            return Err("the refused replay changed the guest alias".to_string());
        }

        // Unknown allocation: explicit UnknownHandle, no GPU work.
        let mut unknown = dispatch(pipeline, ALLOCATION_ID + 9_999);
        unknown.sequence = 3;
        let frame = encode_submission_v3(&unknown).map_err(|error| error.to_string())?;
        let reply = client.submit(3, &frame)?;
        if reply.status != ReplyStatus::UnknownHandle {
            return Err(format!(
                "an unknown allocation was not refused explicitly: {reply:?}"
            ));
        }
        if fs::read(&allocation).expect("re-read after refusal") != before {
            return Err("an unknown allocation changed the guest alias".to_string());
        }

        // Unsupported opcode: the decoder refuses the frame and the host answers.
        let mut frame = encode_submission_v3(&replay).map_err(|error| error.to_string())?;
        if frame.len() < 34 {
            return Err("encoded frame is too short to carry an opcode".to_string());
        }
        frame[32..34].copy_from_slice(&0x00FFu16.to_le_bytes());
        let reply = client.submit(4, &frame)?;
        if reply.status != ReplyStatus::ExecutionFailed {
            return Err(format!(
                "an unsupported opcode was not an explicit failure: {reply:?}"
            ));
        }

        let (modules, d3dmetal) = d3dmetal_loaded()?;
        if d3dmetal {
            return Err("the guest process loaded a D3DMetal module".to_string());
        }

        println!(
            "GUEST_V3_PASS compute_threads={THREADS} alias_first={} alias_last={} \
             replay=AlreadyExecuted alias_unchanged=1 \
             unknown_allocation=UnknownHandle unsupported_opcode=ExecutionFailed \
             d3dmetal_modules=0 guest_modules={modules} method=psapi_EnumProcessModules",
            words[0],
            words[THREADS as usize - 1]
        );
        Ok(())
    }
}
