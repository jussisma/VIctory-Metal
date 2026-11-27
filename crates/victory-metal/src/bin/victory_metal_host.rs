//! Standalone ARM64 `victory-metal-host` daemon (ADR-016).
//!
//! The daemon owns one [`victory_metal::ipc_host::IpcHost`] session at a time
//! around a real [`victory_metal::DirectMetalTransport`]. A guest started with
//! `VICTORY_BACKEND=metal` auto-connects to the loopback listener, authenticates
//! with the shared secret, registers its file-backed allocations and submits
//! Wire v3 work directly to the native Metal device. No Wine, D3D12 or D3DMetal
//! component is loaded or required: the only native libraries touched are Metal
//! and the C runtime, and startup refuses to continue if a D3DMetal image is
//! already mapped into the process.
//!
//! # Startup contract
//!
//! On success the daemon prints exactly one machine-readable banner line,
//! flushed before it accepts:
//!
//! ```text
//! victory-metal-host: ready port=<n> pid=<pid> compute_pipeline=<id> graphics_pipeline=<id>
//! ```
//!
//! The pipeline ids are `0` unless the built-in pipelines were registered
//! (`--frame-gate` or `--aaa-frame`).
//!
//! A failure prints exactly one `victory-metal-host: error <reason>` line and
//! exits non-zero. Usage errors exit `2`; runtime failures exit `1`.
//!
//! # CLI
//!
//! ```text
//! victory-metal-host --secret-hex <64 hex> [--port <0|n>] [--once]
//!                    [--frame-gate] [--aaa-frame] [--telemetry-json <path>]
//!                    [--share-dir <path>] [--help]
//! ```
//!
//! `--port 0` selects an ephemeral loopback port. `--once` serves one session
//! and exits, which is what the launcher and the tests use. Without `--once`
//! the daemon keeps accepting sessions, each with a fresh sequence space and
//! allocation registry.
//!
//! The share directory is launcher-owned and holds the allocation `.bin` files,
//! the optional `gpu-fence.v1.page`, and the `--frame-gate` pipeline ids. It is
//! taken from `--share-dir` or `VICTORY_METAL_SHARE_DIR`; with neither the
//! daemon creates a private `0700` directory under the system temporary
//! directory and removes it on exit.
//!
//! # `--frame-gate` and `--aaa-frame`
//!
//! Both modes compile the synthetic compute kernel and the MRT+depth graphics
//! pair before the banner is printed, register them on the native transport and
//! publish their ids in the banner. They are also written to
//! `<share>/victory-metal-host.compute.id` and
//! `<share>/victory-metal-host.graphics.id` (`id=<n>\n`) for launchers that
//! poll files. The shaders are validation scaffolding, not the production
//! shader design.
//!
//! `--aaa-frame` additionally lets the daemon report the AAA frame outcome in
//! telemetry: the reply status of the last submission containing a render pass
//! (`graphics_status`), of the last submission containing a `Present`
//! (`present_status`), and the FNV-1a 64 checksum of the registered present
//! texture read back as RGBA8 (`present_checksum`). The texture and surface are
//! registered by the authenticated guest through IPC kinds 8 and 9; the daemon
//! never guesses a texture handle.
//!
//! # Telemetry
//!
//! With `--telemetry-json <path>` the daemon writes one JSON object on
//! shutdown: submission and frame counts, frame-time `p50_ms`/`p95_ms`/
//! `p99_ms`, sustained `fps`, the unsupported-opcode histogram,
//! `graphics_status`, `present_status`, `present_checksum`,
//! `d3dmetal_fallback_count` (always `0` on this path) and the observed D3DMetal
//! image list. A write failure is reported, never silently ignored.

#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

const USAGE: &str = "\
usage: victory-metal-host (--secret-hex <hex> | --secret-file <path>) [options]

  --port <port>              loopback port, 0 selects an ephemeral port (default 0)
  --secret-hex <hex>         exactly 32 secret bytes as 64 hex characters
  --secret-file <path>       raw 32 bytes, or 64 hex characters with whitespace
  --share-dir <path>         launcher-owned private directory
                             (otherwise VICTORY_METAL_SHARE_DIR, otherwise a
                             private temporary directory)
  --once                     serve exactly one session, then exit
  --frame-gate               compile and publish the built-in compute and
                             MRT+depth graphics pipeline ids before the banner
  --aaa-frame                like --frame-gate, and report graphics/present
                             status plus the present texture checksum
  --window                   open an NSWindow and present the registered
                             surface with a CAMetalLayer (also enabled by
                             VICTORY_METAL_WINDOW=1 or <width>x<height>)
  --telemetry-json <path>    write session telemetry JSON on shutdown
  -h, --help

The daemon binds 127.0.0.1 only and never requires Wine, D3D12 or D3DMetal.
";

/// Synthetic compute kernel shared with `metal-worker --frame-gate`.
const BOOTSTRAP_ENTRY: &str = "vm_ipc_bootstrap_mul_add";
const BOOTSTRAP_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vm_ipc_bootstrap_mul_add(device uint *data [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 3u + 7u;
}
"#;

/// Synthetic MRT+depth graphics pair used by the AAA frame gate.
///
/// Two RGBA8 colour attachments and one D32F depth attachment. The vertex
/// stage reads two RGBA tints from the root-constants slot bound by
/// `SetRootConstants(slot 0)` and forwards them as varyings, because the
/// non-ABI graphics path binds root constants to the vertex stage only.
const FRAME_GFX_VERTEX: &str = "vm_frame_vertex";
const FRAME_GFX_FRAGMENT: &str = "vm_frame_fragment";
const FRAME_GFX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float4 color0;
    float4 color1;
};

struct MrtOut {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

vertex VOut vm_frame_vertex(uint vid [[vertex_id]],
                            constant float4 *colors [[buffer(0)]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    out.color0 = colors[0];
    out.color1 = colors[1];
    return out;
}

fragment MrtOut vm_frame_fragment(VOut in [[stage_in]]) {
    MrtOut out;
    out.a = in.color0;
    out.b = in.color1;
    return out;
}
"#;

/// Exact colour formats of the built-in AAA graphics pipeline.
///
/// Only the macOS daemon compiles Metal pipelines; the Windows target links the
/// binary but cannot construct these formats.
#[cfg(target_os = "macos")]
const FRAME_COLOR_FORMATS: [PixelFormat; 2] = [PixelFormat::Rgba8Unorm, PixelFormat::Rgba8Unorm];
/// Exact depth format of the built-in AAA graphics pipeline.
#[cfg(target_os = "macos")]
const FRAME_DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// File names the `--frame-gate` pipeline ids are published under.
const COMPUTE_PIPELINE_FILE: &str = "victory-metal-host.compute.id";
const GRAPHICS_PIPELINE_FILE: &str = "victory-metal-host.graphics.id";

/// Per-read/per-write timeout of one authenticated session. A guest that stops
/// talking cannot pin the daemon forever.
const SESSION_IO_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("victory-metal-host: error the native host is only available on ARM64 macOS");
    ExitCode::from(2)
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
            Err(reason) => {
                // One line only: the launcher parses this as the failure.
                eprintln!("victory-metal-host: error {}", one_line(&reason));
                ExitCode::from(1)
            }
        },
        Err(reason) => {
            eprintln!("victory-metal-host: error {}", one_line(&reason));
            print!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(target_os = "macos")]
fn parse_args() -> Result<Option<Args>, String> {
    let mut port = 0u16;
    let mut secret = None;
    let mut secret_source: Option<&'static str> = None;
    let mut share_dir = None;
    let mut once = false;
    let mut frame_gate = false;
    let mut aaa_frame = false;
    let mut window = false;
    let mut telemetry_json = None;

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
                if secret_source.is_some() {
                    return Err("give exactly one of --secret-hex or --secret-file".to_string());
                }
                let value = next_value(&mut args, "--secret-hex")?;
                secret = Some(HostSecret::from_hex(&value).map_err(|error| error.to_string())?);
                secret_source = Some("--secret-hex");
            }
            "--secret-file" => {
                if secret_source.is_some() {
                    return Err("give exactly one of --secret-hex or --secret-file".to_string());
                }
                let value = next_value(&mut args, "--secret-file")?;
                secret = Some(read_secret_file(&value)?);
                secret_source = Some("--secret-file");
            }
            "--share-dir" => {
                share_dir = Some(PathBuf::from(next_value(&mut args, "--share-dir")?));
            }
            "--once" => once = true,
            "--frame-gate" => frame_gate = true,
            "--aaa-frame" => {
                aaa_frame = true;
                frame_gate = true;
            }
            "--window" => window = true,
            "--telemetry-json" => {
                telemetry_json = Some(PathBuf::from(next_value(&mut args, "--telemetry-json")?));
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    let secret = secret.ok_or("missing --secret-hex or --secret-file")?;
    let share_dir = share_dir.or_else(|| {
        env::var_os("VICTORY_METAL_SHARE_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });

    // `--window` opens the default geometry; `VICTORY_METAL_WINDOW` may still
    // select a size. Without `--window`, a window is opened only when the
    // environment asks for one, so the headless default is unchanged.
    let window = match (window, WindowConfig::from_env()) {
        (_, Some(Err(()))) => {
            return Err(format!(
                "invalid {} value",
                victory_metal::macos::window::WINDOW_ENV
            ));
        }
        (true, Some(Ok(config))) => Some(config),
        (true, None) => Some(WindowConfig::default()),
        (false, Some(Ok(config))) => Some(config),
        (false, None) => None,
    };

    Ok(Some(Args {
        port,
        secret,
        share_dir,
        once,
        frame_gate,
        aaa_frame,
        window,
        telemetry_json,
    }))
}

#[cfg(target_os = "macos")]
fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[cfg(target_os = "macos")]
fn read_secret_file(path: &str) -> Result<HostSecret, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("cannot read secret file: {error}"))?;
    if bytes.len() == 32 {
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&bytes);
        return Ok(HostSecret::new(secret));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| "secret file is neither 32 raw bytes nor hex text".to_string())?;
    HostSecret::from_hex(text).map_err(|error| error.to_string())
}

/// Collapses any embedded newline so a failure stays one line.
fn one_line(reason: &str) -> String {
    reason
        .replace(['\n', '\r'], " ")
        .trim()
        .trim_end_matches('.')
        .to_string()
}

#[cfg(target_os = "macos")]
use std::collections::{BTreeMap, HashMap};
#[cfg(target_os = "macos")]
use std::ffi::{c_char, CStr};
#[cfg(target_os = "macos")]
use std::fs::{self, OpenOptions};
#[cfg(target_os = "macos")]
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
#[cfg(target_os = "macos")]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
#[cfg(target_os = "macos")]
use std::process;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "macos")]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "macos")]
use victory_metal::fence_page::FencePageMapping;
use victory_metal::ipc_host::{BatchExecutor, HostSecret, IpcHost, PipelineStageLibrary};
#[cfg(target_os = "macos")]
use victory_metal::macos::render::{GraphicsPipeline, PixelFormat};
#[cfg(target_os = "macos")]
use victory_metal::macos::window::{NativeWindow, WindowConfig};
#[cfg(target_os = "macos")]
use victory_metal::wire::{Reply, ReplyStatus};
#[cfg(target_os = "macos")]
use victory_metal::macos::device::PipelineAbi;
use victory_metal::wire_v3::{CommandV3, SubmissionV3};
#[cfg(target_os = "macos")]
use victory_metal::{DirectMetalTransport, MetalError, MetalTransport, VmBufferId, VmPipelineId};

#[cfg(target_os = "macos")]
struct Args {
    port: u16,
    secret: HostSecret,
    share_dir: Option<PathBuf>,
    once: bool,
    frame_gate: bool,
    aaa_frame: bool,
    window: Option<WindowConfig>,
    telemetry_json: Option<PathBuf>,
}

/// A launcher-owned or daemon-created share directory.
#[cfg(target_os = "macos")]
struct ShareDir {
    path: PathBuf,
    owned: bool,
}

#[cfg(target_os = "macos")]
impl Drop for ShareDir {
    fn drop(&mut self) {
        if self.owned {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(target_os = "macos")]
fn prepare_share(explicit: Option<PathBuf>) -> Result<ShareDir, String> {
    if let Some(path) = explicit {
        fs::create_dir_all(&path)
            .map_err(|error| format!("cannot create share dir {}: {error}", path.display()))?;
        let path = fs::canonicalize(&path)
            .map_err(|error| format!("cannot canonicalize share dir: {error}"))?;
        if !fs::metadata(&path)
            .map(|meta| meta.is_dir())
            .unwrap_or(false)
        {
            return Err(format!("share dir {} is not a directory", path.display()));
        }
        return Ok(ShareDir { path, owned: false });
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let path = env::temp_dir().join(format!("victory-metal-host-{}-{nanos}", process::id()));
    fs::DirBuilder::new()
        .recursive(false)
        .mode(0o700)
        .create(&path)
        .map_err(|error| format!("cannot create a private share dir: {error}"))?;
    let path = fs::canonicalize(&path)
        .map_err(|error| format!("cannot canonicalize the share dir: {error}"))?;
    Ok(ShareDir { path, owned: true })
}

// Images mapped into this process whose name mentions D3DMetal.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(index: u32) -> *const c_char;
}

#[cfg(target_os = "macos")]
fn d3dmetal_images() -> Vec<String> {
    let mut found = Vec::new();
    // SAFETY: `_dyld_image_count` and `_dyld_get_image_name` are documented
    // dyld introspection calls; the returned name pointer is NUL-terminated and
    // owned by dyld for the life of the process.
    unsafe {
        let count = _dyld_image_count();
        for index in 0..count {
            let name = _dyld_get_image_name(index);
            if name.is_null() {
                continue;
            }
            let name = CStr::from_ptr(name).to_string_lossy();
            let lower = name.to_ascii_lowercase();
            if lower.contains("d3dmetal") || lower.contains("metalirconverter") {
                found.push(name.into_owned());
            }
        }
    }
    found
}

#[cfg(target_os = "macos")]
struct Stats {
    sessions: u64,
    submissions: u64,
    frames: u64,
    refused: u64,
    frame_ms: Vec<f64>,
    unsupported: BTreeMap<String, u64>,
    allocations: usize,
    bytes_mapped: u64,
    pipelines: Vec<(String, u64)>,
    /// Reply status of the last submission that contained a render pass; -1
    /// until one is observed.
    graphics_status: i32,
    /// Reply status of the last submission that contained a `Present`; -1
    /// until one is observed.
    present_status: i32,
    /// Texture handle of the last registered present surface.
    present_texture_id: Option<u64>,
    /// FNV-1a 64 of the present texture read back as RGBA8.
    present_checksum: Option<u64>,
    /// Drawables presented by the opt-in native window.
    window_presented: u64,
    /// Drawables acquired by the opt-in native window.
    window_drawables: u64,
    /// First native-window error, if any.
    window_error: Option<String>,
}

#[cfg(target_os = "macos")]
impl Default for Stats {
    fn default() -> Self {
        Self {
            sessions: 0,
            submissions: 0,
            frames: 0,
            refused: 0,
            frame_ms: Vec::new(),
            unsupported: BTreeMap::new(),
            allocations: 0,
            bytes_mapped: 0,
            pipelines: Vec::new(),
            graphics_status: -1,
            present_status: -1,
            present_texture_id: None,
            present_checksum: None,
            window_presented: 0,
            window_drawables: 0,
            window_error: None,
        }
    }
}

#[cfg(target_os = "macos")]
struct DaemonExecutor {
    transport: Arc<DirectMetalTransport>,
    stats: Arc<Mutex<Stats>>,
    window: Option<Arc<Mutex<NativeWindow>>>,
}

#[cfg(target_os = "macos")]
impl DaemonExecutor {
    fn new(
        transport: Arc<DirectMetalTransport>,
        stats: Arc<Mutex<Stats>>,
        window: Option<Arc<Mutex<NativeWindow>>>,
    ) -> Self {
        Self {
            transport,
            stats,
            window,
        }
    }

    /// Presents one frame on the opt-in native window.
    ///
    /// Only a successful submission whose command stream contains a `Present`
    /// reaches the window, and the drawable is acquired after the transport's
    /// submission has completed, so the registered surface is final. All
    /// AppKit/Metal calls stay on the main thread the daemon runs on.
    fn present_window_frame(&self, submission: &SubmissionV3, result: &Result<Reply, MetalError>) {
        let Some(window) = &self.window else {
            return;
        };
        let Some(surface_id) = submission
            .commands
            .iter()
            .find_map(|command| match command {
                CommandV3::Present(present) => Some(present.surface_id),
                _ => None,
            })
        else {
            return;
        };

        let mut window = window.lock().unwrap_or_else(|poison| poison.into_inner());
        window.pump_events();
        if !matches!(result, Ok(reply) if reply.status == ReplyStatus::Ok) {
            return;
        }

        let outcome = window
            .set_present_surface(&self.transport, surface_id)
            .and_then(|()| match window.next_drawable() {
                Some(_) => Ok(()),
                None => Err(victory_metal::macos::window::WindowError::Metal(
                    "CAMetalLayer.nextDrawable returned nil".to_string(),
                )),
            })
            .and_then(|()| window.present());
        let presented = window.presented();
        let drawables = window.drawable_serial();
        drop(window);

        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        stats.window_presented = presented;
        stats.window_drawables = drawables;
        if stats.window_error.is_none() {
            if let Err(error) = outcome {
                stats.window_error = Some(error.to_string());
            }
        }
    }

    fn record(&self, started: Instant, result: &Result<Reply, MetalError>) {
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        stats.submissions += 1;
        match result {
            Ok(reply) if reply.status == ReplyStatus::Ok => {
                stats.frames += 1;
                stats
                    .frame_ms
                    .push(started.elapsed().as_secs_f64() * 1_000.0);
            }
            Ok(_) => stats.refused += 1,
            Err(MetalError::NotSupported(message)) => {
                stats.refused += 1;
                let key: String = message.chars().take(120).collect();
                *stats.unsupported.entry(key).or_default() += 1;
            }
            Err(_) => stats.refused += 1,
        }
    }

    /// Records the AAA stage outcome of one v3 submission.
    ///
    /// `graphics_status` follows the reply of a submission containing a render
    /// pass and `present_status` of one containing a `Present`. The worst
    /// observed status is retained so a later success cannot hide an earlier
    /// failure.
    fn record_v3(
        &self,
        started: Instant,
        submission: &SubmissionV3,
        result: &Result<Reply, MetalError>,
    ) {
        let code = reply_status_code(result);
        let has_pass = submission
            .commands
            .iter()
            .any(|command| matches!(command, CommandV3::BeginRenderPass(_)));
        let has_present = submission
            .commands
            .iter()
            .any(|command| matches!(command, CommandV3::Present(_)));
        self.record(started, result);
        if has_pass || has_present {
            let mut stats = self
                .stats
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if has_pass {
                stats.graphics_status = stats.graphics_status.max(code);
            }
            if has_present {
                stats.present_status = stats.present_status.max(code);
            }
        }
    }
}

/// Maps an executor reply to the wire status code the guest sees.
#[cfg(target_os = "macos")]
fn reply_status_code(result: &Result<Reply, MetalError>) -> i32 {
    match result {
        Ok(reply) => match reply.status {
            ReplyStatus::Ok => 0,
            ReplyStatus::UnknownHandle => 1,
            ReplyStatus::ExecutionFailed => 2,
            ReplyStatus::AlreadyExecuted => 3,
        },
        Err(_) => 2,
    }
}

/// FNV-1a 64 over RGBA8 pixels, byte order row-major.
#[cfg(target_os = "macos")]
fn checksum_rgba8(pixels: &[[u8; 4]]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for pixel in pixels {
        for &byte in pixel {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

#[cfg(target_os = "macos")]
impl BatchExecutor for DaemonExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        <DirectMetalTransport as BatchExecutor>::page_size(self.transport.as_ref())
    }

    fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        BatchExecutor::register_window(self.transport.as_ref(), ptr, len)
    }

    fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError> {
        BatchExecutor::unregister_window(self.transport.as_ref(), id)
    }
    fn attach_fence_page(&self, page: Arc<FencePageMapping>) -> Result<bool, MetalError> {
        // Delegate to the transport so the daemon's sessions publish completion
        // from the GPU completion handler instead of the synchronous fallback.
        BatchExecutor::attach_fence_page(self.transport.as_ref(), page)
    }

    fn register_graphics_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        vertex_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        BatchExecutor::register_graphics_pipeline(
            self.transport.as_ref(), handle, metallib, vertex_entry, state, abi,
        )
    }

    fn register_mesh_pipeline(
        &self,
        handle: u64,
        metallib: &[u8],
        mesh_entry: &str,
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        BatchExecutor::register_mesh_pipeline(self.transport.as_ref(), handle, metallib, mesh_entry, state, abi)
    }

    fn register_graphics_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        BatchExecutor::register_graphics_pipeline_stages(self.transport.as_ref(), handle, stages, state, abi)
    }

    fn register_mesh_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        BatchExecutor::register_mesh_pipeline_stages(self.transport.as_ref(), handle, stages, state, abi)
    }

    fn register_compute_pipeline_stages(
        &self,
        handle: u64,
        stages: &[PipelineStageLibrary<'_>],
        state: &[u8],
        abi: PipelineAbi,
    ) -> Result<(), MetalError> {
        BatchExecutor::register_compute_pipeline_stages(self.transport.as_ref(), handle, stages, state, abi)
    }

    fn detach_fence_page(&self) {
        BatchExecutor::detach_fence_page(self.transport.as_ref());
    }

    fn submit_batch(
        &self,
        submission: &victory_metal::wire::Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let started = Instant::now();
        let result = BatchExecutor::submit_batch(self.transport.as_ref(), submission, allocations);
        self.record(started, &result);
        result
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        let started = Instant::now();
        let result =
            DirectMetalTransport::submit_v3(self.transport.as_ref(), submission, allocations);
        self.record_v3(started, submission, &result);
        self.present_window_frame(submission, &result);
        result
    }

    fn load_pipeline(&self, metallib: &[u8], entry: &str) -> Result<VmPipelineId, MetalError> {
        BatchExecutor::load_pipeline(self.transport.as_ref(), metallib, entry)
    }

    fn register_texture(
        &self,
        id: u64,
        desc: victory_metal::macos::registry::TextureDesc,
    ) -> Result<(), MetalError> {
        self.transport.register_registry_texture(id, desc)
    }

    fn register_surface(&self, surface_id: u64, texture_id: u64) -> Result<(), MetalError> {
        BatchExecutor::register_surface(self.transport.as_ref(), surface_id, texture_id)?;
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        stats.present_texture_id = Some(texture_id);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_pipeline_id(share: &Path, name: &str, id: u64) -> Result<(), String> {
    let path = share.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    writeln!(file, "id={id}")
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))
}

#[cfg(target_os = "macos")]
fn build_executor(
    share: &Path,
    frame_gate: bool,
    stats: &Arc<Mutex<Stats>>,
    window: Option<Arc<Mutex<NativeWindow>>>,
) -> Result<(DaemonExecutor, Arc<DirectMetalTransport>), String> {
    let transport = Arc::new(
        DirectMetalTransport::new()
            .map_err(|error| format!("cannot open the Metal device: {error}"))?,
    );
    if frame_gate {
        let compute =
            MetalTransport::compile_msl(transport.as_ref(), BOOTSTRAP_MSL, BOOTSTRAP_ENTRY)
                .map_err(|error| {
                    format!("cannot compile the frame-gate compute pipeline: {error}")
                })?;
        write_pipeline_id(share, COMPUTE_PIPELINE_FILE, compute.0)?;
        let graphics_state = GraphicsPipeline::compile_mrt(
            transport.as_ref(),
            FRAME_GFX_MSL,
            FRAME_GFX_VERTEX,
            FRAME_GFX_FRAGMENT,
            &FRAME_COLOR_FORMATS,
            Some(FRAME_DEPTH_FORMAT),
        )
        .map_err(|error| format!("cannot compile the frame-gate graphics pipeline: {error}"))?;
        let graphics = transport
            .register_graphics_pipeline_state(graphics_state)
            .map_err(|error| {
                format!("cannot register the frame-gate graphics pipeline: {error}")
            })?;
        write_pipeline_id(share, GRAPHICS_PIPELINE_FILE, graphics.0)?;
        let mut stats = stats.lock().unwrap_or_else(|poison| poison.into_inner());
        stats.pipelines = vec![
            ("compute".to_string(), compute.0),
            ("graphics".to_string(), graphics.0),
        ];
    }
    let executor = DaemonExecutor::new(Arc::clone(&transport), Arc::clone(stats), window);
    Ok((executor, transport))
}

#[cfg(target_os = "macos")]
fn pipeline_id(stats: &Stats, name: &str) -> u64 {
    stats
        .pipelines
        .iter()
        .find(|(label, _)| label == name)
        .map_or(0, |(_, id)| *id)
}

/// Reads the last registered present texture back as RGBA8 and records its
/// FNV-1a 64 checksum. A readback failure leaves the checksum unset, so the
/// launcher reports an honest gap instead of a fabricated value.
#[cfg(target_os = "macos")]
fn read_back_present(transport: &DirectMetalTransport, stats: &Arc<Mutex<Stats>>) {
    let texture_id = {
        let stats = stats.lock().unwrap_or_else(|poison| poison.into_inner());
        stats.present_texture_id
    };
    let Some(texture_id) = texture_id else {
        return;
    };
    let checksum = transport
        .read_registered_texture_rgba8(texture_id)
        .map(|pixels| checksum_rgba8(&pixels))
        .ok();
    let mut stats = stats.lock().unwrap_or_else(|poison| poison.into_inner());
    stats.present_checksum = checksum;
}

#[cfg(target_os = "macos")]
fn percentile(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((percent / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(target_os = "macos")]
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control < ' ' => out.push_str(&format!("\\u{:04x}", control as u32)),
            other => out.push(other),
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn write_telemetry(
    path: &Path,
    port: u16,
    stats: &Stats,
    d3dmetal: &[String],
) -> Result<(), String> {
    let mut frame_ms = stats.frame_ms.clone();
    frame_ms.sort_by(|left, right| left.total_cmp(right));
    let p50 = percentile(&frame_ms, 50.0);
    let p95 = percentile(&frame_ms, 95.0);
    let p99 = percentile(&frame_ms, 99.0);
    let total_ms: f64 = frame_ms.iter().sum();
    let fps = if total_ms > 0.0 {
        stats.frames as f64 / (total_ms / 1_000.0)
    } else {
        0.0
    };

    let unsupported = stats
        .unsupported
        .iter()
        .map(|(message, count)| format!("\"{}\": {count}", json_escape(message)))
        .collect::<Vec<_>>()
        .join(", ");
    let pipelines = stats
        .pipelines
        .iter()
        .map(|(name, id)| format!("\"{}\": {id}", json_escape(name)))
        .collect::<Vec<_>>()
        .join(", ");
    let images = d3dmetal
        .iter()
        .map(|name| format!("\"{}\"", json_escape(name)))
        .collect::<Vec<_>>()
        .join(", ");
    let window_error = stats.window_error.as_deref().map_or_else(
        || "null".to_string(),
        |message| format!("\"{}\"", json_escape(message)),
    );

    let json = format!(
        concat!(
            "{{\n",
            "  \"host\": \"victory-metal-host\",\n",
            "  \"pid\": {pid},\n",
            "  \"port\": {port},\n",
            "  \"sessions\": {sessions},\n",
            "  \"submissions\": {submissions},\n",
            "  \"frames\": {frames},\n",
            "  \"refused_submissions\": {refused},\n",
            "  \"p50_ms\": {p50:.3},\n",
            "  \"p95_ms\": {p95:.3},\n",
            "  \"p99_ms\": {p99:.3},\n",
            "  \"fps\": {fps:.3},\n",
            "  \"allocations\": {allocations},\n",
            "  \"bytes_mapped\": {bytes_mapped},\n",
            "  \"unsupported_opcodes\": {{{unsupported}}},\n",
            "  \"pipelines\": {{{pipelines}}},\n",
            "  \"graphics_status\": {graphics_status},\n",
            "  \"present_status\": {present_status},\n",
            "  \"present_checksum\": {present_checksum},\n",
            "  \"window_presented\": {window_presented},\n",
            "  \"window_drawables\": {window_drawables},\n",
            "  \"window_error\": {window_error},\n",
            "  \"d3dmetal_loaded\": {loaded},\n",
            "  \"d3dmetal_images\": [{images}],\n",
            "  \"d3dmetal_fallback_count\": 0\n",
            "}}\n"
        ),
        pid = process::id(),
        port = port,
        sessions = stats.sessions,
        submissions = stats.submissions,
        frames = stats.frames,
        refused = stats.refused,
        p50 = p50,
        p95 = p95,
        p99 = p99,
        fps = fps,
        allocations = stats.allocations,
        bytes_mapped = stats.bytes_mapped,
        unsupported = unsupported,
        pipelines = pipelines,
        graphics_status = stats.graphics_status,
        present_status = stats.present_status,
        present_checksum = stats
            .present_checksum
            .map_or_else(|| "-1".to_string(), |value| value.to_string()),
        window_presented = stats.window_presented,
        window_drawables = stats.window_drawables,
        window_error = window_error,
        images = images,
        loaded = !d3dmetal.is_empty(),
    );

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!("cannot create telemetry dir {}: {error}", parent.display())
        })?;
    }
    let temporary = path.with_extension(format!("tmp-{}", process::id()));
    fs::write(&temporary, json)
        .map_err(|error| format!("cannot write telemetry {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("cannot publish telemetry {}: {error}", path.display()))
}

#[cfg(target_os = "macos")]
fn run(args: Args) -> Result<(), String> {
    // ADR-016: the daemon must never require D3DMetal. Refuse to start if the
    // launcher's environment already mapped it into this process.
    let loaded = d3dmetal_images();
    if !loaded.is_empty() {
        return Err(format!(
            "a D3DMetal image is already loaded: {}",
            loaded.join(", ")
        ));
    }

    let share = prepare_share(args.share_dir.clone())?;
    let stats = Arc::new(Mutex::new(Stats::default()));

    // ADR-020: on-screen presentation is opt-in. Opening the window before the
    // banner means a requested window that cannot open is a startup failure,
    // never a silent fallback to headless.
    let window = match args.window.clone() {
        Some(config) => {
            let open = NativeWindow::open(config)
                .map_err(|error| format!("cannot open the native window: {error}"))?;
            Some(Arc::new(Mutex::new(open)))
        }
        None => None,
    };

    // Build the first session's executor (and its frame-gate pipeline ids)
    // before the banner, so a launcher that waits for `ready` can read them.
    let mut prepared = Some(build_executor(
        &share.path,
        args.frame_gate,
        &stats,
        window.clone(),
    )?);

    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, args.port)))
        .map_err(|error| format!("cannot bind 127.0.0.1:{}: {error}", args.port))?;
    let local = listener
        .local_addr()
        .map_err(|error| format!("cannot read the listener address: {error}"))?;
    if !local.ip().is_loopback() {
        return Err("the listener did not bind to loopback".to_string());
    }
    let port = local.port();

    let (compute_pipeline, graphics_pipeline) = {
        let stats = stats.lock().unwrap_or_else(|poison| poison.into_inner());
        (
            pipeline_id(&stats, "compute"),
            pipeline_id(&stats, "graphics"),
        )
    };
    let window_field = args.window.as_ref().map_or_else(String::new, |config| {
        format!(" window={}x{}", config.width, config.height)
    });
    println!(
        "victory-metal-host: ready port={port} pid={} compute_pipeline={compute_pipeline} graphics_pipeline={graphics_pipeline}{window_field}",
        process::id()
    );
    io::stdout()
        .flush()
        .map_err(|error| format!("cannot flush the ready banner: {error}"))?;

    let mut outcome = Ok(());
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                outcome = Err(format!("accept failed: {error}"));
                break;
            }
        };
        if !peer.ip().is_loopback() {
            outcome = Err(format!("refused a non-loopback peer {peer}"));
            break;
        }

        let (executor, transport) = match prepared.take() {
            Some(pair) => pair,
            None => match build_executor(&share.path, args.frame_gate, &stats, window.clone()) {
                Ok(pair) => pair,
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            },
        };

        let mut host = match IpcHost::new(args.secret, &share.path, executor) {
            Ok(host) => host.with_io_timeout(SESSION_IO_TIMEOUT),
            Err(error) => {
                outcome = Err(format!("cannot start the session: {error}"));
                break;
            }
        };

        let session = host.serve_once(stream);
        if let Ok(summary) = &session {
            let mut stats = stats.lock().unwrap_or_else(|poison| poison.into_inner());
            stats.sessions += 1;
            stats.allocations = summary.allocations;
            stats.bytes_mapped = summary.bytes_mapped;
        }
        if args.aaa_frame {
            read_back_present(transport.as_ref(), &stats);
        }
        host.release_all();
        drop(host);

        match session {
            Ok(_) => {}
            Err(error) => {
                outcome = Err(format!("session failed: {error}"));
                break;
            }
        }

        if args.once {
            break;
        }
    }

    // `--once` teardown must not hang: closing the window is a synchronous
    // orderOut/close with no nested run loop.
    if let Some(window) = &window {
        window
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .close();
    }

    let loaded = d3dmetal_images();
    if !loaded.is_empty() && outcome.is_ok() {
        outcome = Err(format!(
            "a D3DMetal image was loaded during the session: {}",
            loaded.join(", ")
        ));
    }

    if let Some(path) = args.telemetry_json.as_deref() {
        let snapshot = stats
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        if let Err(error) = write_telemetry(path, port, &snapshot, &loaded) {
            if outcome.is_ok() {
                outcome = Err(error);
            }
        }
    }

    outcome
}

#[cfg(target_os = "macos")]
impl Clone for Stats {
    fn clone(&self) -> Self {
        Self {
            sessions: self.sessions,
            submissions: self.submissions,
            frames: self.frames,
            refused: self.refused,
            frame_ms: self.frame_ms.clone(),
            unsupported: self.unsupported.clone(),
            allocations: self.allocations,
            bytes_mapped: self.bytes_mapped,
            pipelines: self.pipelines.clone(),
            graphics_status: self.graphics_status,
            present_status: self.present_status,
            present_texture_id: self.present_texture_id,
            present_checksum: self.present_checksum,
            window_presented: self.window_presented,
            window_drawables: self.window_drawables,
            window_error: self.window_error.clone(),
        }
    }
}
