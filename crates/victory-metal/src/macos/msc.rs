//! Typed binder for the verified tiny Metal Shader Converter compute root
//! signature.
//!
//! Metal Shader Converter (`libmetalirconverter.dylib`) lowers a DXIL compute
//! shader with the synthetic root signature below into a Metal function whose
//! only resource is reached through a *top-level global argument buffer*:
//!
//! ```text
//! HLSL: RootFlags(0), UAV(u0)
//! Metal: buffer(2) -> 8-byte device address of the UAV at argument offset 0
//! ```
//!
//! The constants were observed end-to-end by `tools/macos/metal-poc/arm64-dxil.sh`
//! (x86_64 converter process, separate arm64 renderer process). This module does
//! not guess layouts for arbitrary titles: it is an opt-in typed binding for
//! exactly that verified shape. Callers that need another layout must derive and
//! verify it first, then extend this module explicitly.
//!
//! The binder writes a checked [`MscUavDispatch`] against the existing pipeline
//! and registered-buffer tables; see
//! [`crate::macos::device::DirectMetalTransport::dispatch_msc_uav_1d`].
//!
//! This module also owns the host's *second translation path*: [`MscTool`]
//! resolves and drives Apple's `metal-shaderconverter` command-line tool when
//! the Rosetta `libmetalirconverter.dylib` worker dies on a shader. See the
//! "Command-line fallback" section below for the exact invocation.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use crate::macos::irconverter::{
    MAX_DXIL_INPUT_BYTES, MAX_ENTRY_POINT_BYTES, MAX_METALLIB_OUTPUT_BYTES,
};
use crate::transport::MetalError;
use crate::wire;

/// Byte length of the synthetic top-level argument buffer.
///
/// Matches the verified harness: a 32-byte buffer with the UAV device address at
/// offset 0. Metal only reads the declared fields, so the slack keeps the shape
/// explicit without changing converted-shader semantics.
pub const MSC_ARGUMENT_BUFFER_LENGTH: usize = 32;

/// Metal buffer argument index of the converted top-level argument buffer.
pub const MSC_DEFAULT_ARGUMENT_BUFFER_INDEX: usize = 2;

/// Byte offset of the UAV device address inside the top-level argument buffer.
pub const MSC_DEFAULT_UAV_ADDRESS_OFFSET: usize = 0;

/// Size of one encoded GPU device address (`MTLGPUAddress`).
pub const MSC_DEVICE_ADDRESS_SIZE: usize = 8;

/// Exclusive upper bound for a Metal buffer argument index.
///
/// Metal exposes buffer argument indices `0..=30` for compute encoders; index 31
/// and above is never a valid `setBuffer:offset:atIndex:` target.
pub const MSC_MAX_BUFFER_ARGUMENT_INDEX: usize = 31;

/// Validated layout of the converted top-level argument buffer.
///
/// Fields are private so a caller cannot construct a layout that violates the
/// device-address alignment or the argument-buffer length; use
/// [`MscUavLayout::new`] or [`MscUavLayout::DEFAULT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MscUavLayout {
    argument_buffer_index: usize,
    address_offset: usize,
}

impl MscUavLayout {
    /// The verified layout from `arm64-dxil.sh`: `buffer(2)`, address at offset 0.
    pub const DEFAULT: Self = Self {
        argument_buffer_index: MSC_DEFAULT_ARGUMENT_BUFFER_INDEX,
        address_offset: MSC_DEFAULT_UAV_ADDRESS_OFFSET,
    };

    /// Builds an explicit layout for a verified converted root signature.
    ///
    /// * `argument_buffer_index` must be a valid Metal buffer argument index
    ///   (`0..31`).
    /// * `address_offset` must be 8-byte aligned and leave room for one device
    ///   address inside [`MSC_ARGUMENT_BUFFER_LENGTH`].
    ///
    /// Returns [`MetalError::InvalidArg`] for any other combination. This binder
    /// intentionally has no fallback and no probing: an invalid layout is
    /// rejected before any Metal object is created.
    pub fn new(argument_buffer_index: usize, address_offset: usize) -> Result<Self, MetalError> {
        if argument_buffer_index >= MSC_MAX_BUFFER_ARGUMENT_INDEX {
            return Err(MetalError::InvalidArg(
                "argument buffer index must be a valid Metal buffer index",
            ));
        }
        if !address_offset.is_multiple_of(MSC_DEVICE_ADDRESS_SIZE) {
            return Err(MetalError::InvalidArg(
                "UAV device address offset must be 8-byte aligned",
            ));
        }
        let Some(end) = address_offset.checked_add(MSC_DEVICE_ADDRESS_SIZE) else {
            return Err(MetalError::InvalidArg(
                "UAV device address offset overflows the argument buffer",
            ));
        };
        if end > MSC_ARGUMENT_BUFFER_LENGTH {
            return Err(MetalError::InvalidArg(
                "UAV device address does not fit in the argument buffer",
            ));
        }
        Ok(Self {
            argument_buffer_index,
            address_offset,
        })
    }

    /// Metal buffer argument index where the argument buffer is bound.
    pub fn argument_buffer_index(&self) -> usize {
        self.argument_buffer_index
    }

    /// Byte offset of the UAV device address inside the argument buffer.
    pub fn address_offset(&self) -> usize {
        self.address_offset
    }

    /// Byte length of the argument buffer this layout describes.
    pub fn argument_buffer_len(&self) -> usize {
        MSC_ARGUMENT_BUFFER_LENGTH
    }
}

impl Default for MscUavLayout {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One opt-in 1D compute dispatch through the converted MSC argument buffer.
///
/// `offset`/`length` are an explicit range inside the *registered backing* of
/// `uav`, resolved by the caller from the win64 allocation/generation mapping.
/// They are validated again against the registry at dispatch time; the binder
/// never trusts a caller-computed end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MscUavDispatch {
    /// Pipeline registered by `compile_dxil`, `compile_msl`, or `load_metallib`.
    pub pipeline: crate::abi::VmPipelineId,
    /// Registered buffer that backs the converted UAV.
    pub uav: crate::abi::VmBufferId,
    /// Byte offset of the UAV range inside the registered backing.
    pub offset: u64,
    /// Byte length of the UAV range.
    pub length: u64,
    /// Total threads in the 1D grid (nonzero).
    pub threads: u32,
    /// Threads per threadgroup (nonzero, within the pipeline maximum).
    pub threads_per_group: u32,
    /// Layout of the converted top-level argument buffer.
    pub layout: MscUavLayout,
}

/// Validates a UAV byte range against the registered backing length.
///
/// Returns `(offset, end)` as host `usize` values. The rules mirror the legacy
/// wire batch contract: nonzero length capped by [`wire::MAX_RESOURCE_LEN`],
/// 4-byte offset alignment, overflow-checked end, and `end <= registered_len`.
pub(crate) fn checked_binding_range(
    offset: u64,
    length: u64,
    registered_len: usize,
) -> Result<(usize, usize), MetalError> {
    if length == 0 {
        return Err(MetalError::InvalidArg("UAV range length must be nonzero"));
    }
    if length > wire::MAX_RESOURCE_LEN {
        return Err(MetalError::InvalidArg(
            "UAV range exceeds the maximum resource length",
        ));
    }
    if !offset.is_multiple_of(crate::macos::device::MTL_BUFFER_OFFSET_ALIGNMENT) {
        return Err(MetalError::InvalidArg(
            "UAV range offset must be 4-byte aligned",
        ));
    }
    let Some(end) = offset.checked_add(length) else {
        return Err(MetalError::InvalidArg("UAV range end overflows"));
    };
    let (Ok(offset), Ok(end)) = (usize::try_from(offset), usize::try_from(end)) else {
        return Err(MetalError::InvalidArg(
            "UAV range does not fit host address space",
        ));
    };
    if end > registered_len {
        return Err(MetalError::InvalidArg(
            "UAV range exceeds the registered backing length",
        ));
    }
    Ok((offset, end))
}

/// Encodes `address` into `arguments` according to `layout`.
///
/// The 8-byte field is written little-endian, matching the Metal ABI on both
/// supported Mac architectures. The rest of the buffer is left untouched so the
/// caller can zero and reuse it deterministically.
pub(crate) fn write_uav_address(
    arguments: &mut [u8],
    layout: MscUavLayout,
    address: u64,
) -> Result<(), MetalError> {
    let end = layout
        .address_offset()
        .checked_add(MSC_DEVICE_ADDRESS_SIZE)
        .ok_or(MetalError::InvalidArg(
            "UAV device address offset overflows",
        ))?;
    if end > arguments.len() {
        return Err(MetalError::InvalidArg(
            "argument buffer slice is shorter than its layout",
        ));
    }
    arguments[layout.address_offset()..end].copy_from_slice(&address.to_le_bytes());
    Ok(())
}

// ---------------------------------------------------------------------------
// Command-line fallback: `metal-shaderconverter`
// ---------------------------------------------------------------------------
//
// The typed binder above describes the *output* ABI. This section is the
// second *translation* path: when the Rosetta `libmetalirconverter.dylib`
// process dies on a shader, the host retries that shader through Apple's
// supported command-line successor, `metal-shaderconverter` (MSC), which ships
// next to the same framework resources.
//
// The invocation is intentionally explicit and version-checked at runtime
// rather than guessed from the input:
//
// ```text
// metal-shaderconverter <input.dxil> -o <output.metallib> \
//     -e <entry-point> --reflection <reflection.json>
// ```
//
// `<input.dxil>` is the whole container the legacy path receives (DXBC/DXIL),
// byte for byte; the shader stage travels inside that container and MSC reports
// it back in the reflection. `<reflection.json>` is Apple's
// `IRShaderReflectionCopyJSONString` document. A tool build that rejects these
// arguments, writes no metallib or writes an empty reflection is a named,
// bounded failure: the caller keeps the legacy failure reason and reports both
// paths. No fallback ever fabricates a layout.

/// Executable name of the Apple Metal Shader Converter command-line tool.
pub const MSC_TOOL_NAME: &str = "metal-shaderconverter";

/// Environment variable naming an explicit `metal-shaderconverter` executable.
///
/// When set, it is authoritative: a value that does not name an executable file
/// is an unavailable-tool failure, never a reason to silently use a different
/// installation.
pub const MSC_TOOL_ENV: &str = "VICTORY_MSC_SHADER_CONVERTER";

/// Upper bound for the reflection JSON an MSC run may produce.
///
/// Matches the pool's `MAX_REFLECTION_OUTPUT_BYTES`; a larger document is
/// refused instead of truncated into an unusable ABI.
pub const MAX_MSC_REFLECTION_BYTES: usize = 4 * 1024 * 1024;

/// Upper bound for one MSC process's stderr retained for diagnostics.
pub const MAX_MSC_DIAGNOSTIC_BYTES: usize = 4096;

/// Raw bytes retained per pipe before escaping and tail-truncating.
const MSC_DIAGNOSTIC_CAPTURE_BYTES: usize = MAX_MSC_DIAGNOSTIC_BYTES * 4;

/// Longest one-line reason this module reports.
const MSC_REASON_BYTES: usize = 512;

/// How often a running MSC process is checked for exit.
const MSC_POLL_INTERVAL: Duration = Duration::from_millis(10);

static MSC_SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
unsafe extern "C" {
    /// POSIX `kill(2)`; a negative pid targets a process group.
    fn kill(pid: i32, signal: i32) -> i32;
}

#[cfg(unix)]
const SIGKILL: i32 = 9;

/// Coarse, named class of an MSC failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MscErrorKind {
    /// No executable resolved (or an explicit path is not executable).
    Unavailable,
    /// The executable exists but could not be started.
    Spawn,
    /// The tool ran and refused the shader / arguments.
    Rejected,
    /// The tool did not finish before the deadline and was killed.
    TimedOut,
    /// A metallib or reflection file was missing, empty or oversized.
    InvalidOutput,
    /// Local scratch I/O failed.
    Io,
}

/// One bounded, single-line MSC failure.
///
/// Every constructor runs [`sanitize_msc_reason`], so `Display` is always safe
/// for a telemetry line: no newlines, no absolute host paths, bounded length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MscError {
    kind: MscErrorKind,
    message: String,
}

impl MscError {
    fn new(kind: MscErrorKind, message: impl AsRef<str>) -> Self {
        Self {
            kind,
            message: sanitize_msc_reason(message.as_ref(), MSC_REASON_BYTES),
        }
    }

    /// Named failure class.
    pub fn kind(&self) -> MscErrorKind {
        self.kind
    }

    /// Bounded, path-scrubbed one-line reason.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for MscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MscError {}

/// One MSC translation request.
///
/// The DXIL is the complete container, exactly what the legacy converter
/// receives; `stage` is the caller's declared stage and is used to make a
/// mismatch visible in the failure text. `timeout` bounds the child so a hung
/// tool can never outlive the job.
#[derive(Debug, Clone, Copy)]
pub struct MscRequest<'a> {
    /// Complete DXIL/DXBC container bytes.
    pub dxil: &'a [u8],
    /// Entry point named in the container.
    pub entry: &'a str,
    /// Declared shader stage (`1` vertex, `2` pixel, `5` mesh, `6` object,
    /// `8` compute).
    pub stage: u32,
    /// Wall-clock bound for one MSC invocation.
    pub timeout: Duration,
}

/// One successful MSC translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MscOutput {
    /// Serialized metallib bytes.
    pub metallib: Vec<u8>,
    /// Apple's reflection JSON document (never empty).
    pub reflection: Vec<u8>,
}

/// A resolved `metal-shaderconverter` executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MscTool {
    path: PathBuf,
}

impl MscTool {
    /// Resolves the tool the same way the launcher resolves the converter.
    ///
    /// Order: an explicit [`MSC_TOOL_ENV`] path (authoritative even when
    /// missing), then `metal-shaderconverter` next to the converter library
    /// (`D3DMETAL_RES`, the Homebrew/CrossOver framework cache, or the
    /// `VICTORY_METAL_IRCONVERTER` sibling), then the same name on `PATH`.
    ///
    /// The returned failure names the missing tool in one line and never
    /// panics; an absent MSC is a normal deployment state.
    pub fn resolve() -> Result<Self, MscError> {
        if let Some(raw) = std::env::var_os(MSC_TOOL_ENV) {
            if raw.is_empty() {
                return Err(MscError::new(
                    MscErrorKind::Unavailable,
                    format!("{MSC_TOOL_ENV} is empty"),
                ));
            }
            let path = PathBuf::from(raw);
            return Self::from_path(path);
        }
        for candidate in Self::candidates() {
            if is_executable_file(&candidate) {
                return Ok(Self { path: candidate });
            }
        }
        Err(MscError::new(
            MscErrorKind::Unavailable,
            format!(
                "MSC unavailable: no {MSC_TOOL_NAME} next to libmetalirconverter.dylib \
                 (set {MSC_TOOL_ENV})"
            ),
        ))
    }

    /// Validates one explicit tool path.
    ///
    /// Unlike [`MscTool::resolve`] this never searches other locations, so a
    /// test or operator can point at a stub or a private installation without
    /// picking up an unrelated tool.
    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, MscError> {
        let path = path.into();
        if is_executable_file(&path) {
            Ok(Self { path })
        } else {
            Err(MscError::new(
                MscErrorKind::Unavailable,
                format!(
                    "MSC unavailable: {MSC_TOOL_NAME} is not an executable file \
                     (set {MSC_TOOL_ENV})"
                ),
            ))
        }
    }

    /// Resolved executable path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        let mut push_dir = |dir: &Path| {
            candidates.push(dir.join(MSC_TOOL_NAME));
            candidates.push(dir.join("bin").join(MSC_TOOL_NAME));
        };
        if let Some(res) = std::env::var_os("D3DMETAL_RES")
            && !res.is_empty()
        {
            push_dir(Path::new(&res));
        }
        if let Some(lib) = std::env::var_os("VICTORY_METAL_IRCONVERTER")
            && !lib.is_empty()
            && let Some(parent) = Path::new(&lib).parent()
        {
            push_dir(parent);
        }
        if let Some(home) = std::env::var_os("HOME") {
            let resources = PathBuf::from(home).join(
                "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                 D3DMetal.framework/Versions/A/Resources",
            );
            push_dir(&resources);
        }
        push_dir(Path::new(
            "/Library/Application Support/CrossOver/D3DMetal.framework/Versions/A/Resources",
        ));
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                candidates.push(dir.join(MSC_TOOL_NAME));
            }
        }
        candidates
    }

    /// Runs `--version` under a short bound and returns its first line.
    ///
    /// `None` means the tool printed nothing usable or did not exit
    /// successfully; a version string is diagnostic only and never required.
    pub fn version(&self) -> Option<String> {
        let mut command = Command::new(&self.path);
        command.arg("--version");
        let output = run_bounded(command, Duration::from_secs(2)).ok()?;
        if !output.status.success() {
            return None;
        }
        let text = if output.stdout.iter().any(|byte| !byte.is_ascii_whitespace()) {
            String::from_utf8_lossy(&output.stdout)
        } else {
            String::from_utf8_lossy(&output.stderr)
        };
        text.lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| sanitize_msc_reason(line, 160))
    }

    /// Converts one complete DXIL container to a metallib plus reflection JSON.
    ///
    /// The tool runs in its own process group, its stderr is drained on a
    /// thread (so a chatty tool cannot block on a full pipe), and the direct
    /// child is always reaped. A deadline kills the whole group, so no helper
    /// process is left behind. Scratch files live under the OS temporary
    /// directory and are deleted by [`MscScratch`] on every exit path.
    pub fn convert(&self, request: &MscRequest<'_>) -> Result<MscOutput, MscError> {
        if request.dxil.is_empty() {
            return Err(MscError::new(
                MscErrorKind::Rejected,
                "msc input is empty",
            ));
        }
        if request.dxil.len() > MAX_DXIL_INPUT_BYTES {
            return Err(MscError::new(
                MscErrorKind::Rejected,
                format!(
                    "msc input is {} bytes; limit is {MAX_DXIL_INPUT_BYTES}",
                    request.dxil.len()
                ),
            ));
        }
        if request.entry.is_empty() || request.entry.len() > MAX_ENTRY_POINT_BYTES {
            return Err(MscError::new(
                MscErrorKind::Rejected,
                "msc entry point is empty or over the 256-byte limit",
            ));
        }

        let scratch = MscScratch::new()?;
        let input = scratch.path().join("input.dxil");
        let metallib_path = scratch.path().join("output.metallib");
        let reflection_path = scratch.path().join("reflection.json");
        std::fs::write(&input, request.dxil).map_err(|err| {
            MscError::new(
                MscErrorKind::Io,
                format!("msc could not write its DXIL scratch input: {err}"),
            )
        })?;

        let mut command = Command::new(&self.path);
        command
            .arg(&input)
            .arg("-o")
            .arg(&metallib_path)
            .arg("-e")
            .arg(request.entry)
            .arg("--reflection")
            .arg(&reflection_path);

        let output = run_bounded(command, request.timeout)?;
        if !output.status.success() {
            return Err(MscError::new(
                MscErrorKind::Rejected,
                format!(
                    "msc rejected the shader for stage {} ({}): {}",
                    request.stage,
                    describe_status(&output.status),
                    output.reason()
                ),
            ));
        }

        let metallib = read_bounded(
            &metallib_path,
            MAX_METALLIB_OUTPUT_BYTES,
            "msc metallib output",
        )?;
        let reflection = read_bounded(
            &reflection_path,
            MAX_MSC_REFLECTION_BYTES,
            "msc reflection output",
        )?;
        if reflection.iter().all(u8::is_ascii_whitespace) {
            return Err(MscError::new(
                MscErrorKind::InvalidOutput,
                format!(
                    "msc produced no reflection for stage {}; the host requires a non-empty \
                     reflection JSON",
                    request.stage
                ),
            ));
        }
        Ok(MscOutput {
            metallib,
            reflection,
        })
    }
}

/// One bounded process result.
struct BoundedOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl BoundedOutput {
    fn reason(&self) -> String {
        let raw = if self.stderr.iter().any(|byte| !byte.is_ascii_whitespace()) {
            &self.stderr
        } else {
            &self.stdout
        };
        if raw.is_empty() {
            return "no diagnostic".to_string();
        }
        sanitize_msc_reason(&String::from_utf8_lossy(raw), MSC_REASON_BYTES)
    }
}

/// Runs `command` with a deadline, bounded pipes and a reaped process group.
fn run_bounded(mut command: Command, timeout: Duration) -> Result<BoundedOutput, MscError> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|err| {
        MscError::new(
            MscErrorKind::Spawn,
            format!("msc could not start {MSC_TOOL_NAME}: {err}"),
        )
    })?;
    let stdout = child.stdout.take().map(spawn_diagnostic_reader);
    let stderr = child.stderr.take().map(spawn_diagnostic_reader);
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_process_group(&mut child);
                    // The readers close once the whole group is gone; their
                    // retained tails are intentionally dropped here.
                    return Err(MscError::new(
                        MscErrorKind::TimedOut,
                        format!(
                            "msc did not finish within {} ms and was killed",
                            timeout.as_millis()
                        ),
                    ));
                }
                thread::sleep(MSC_POLL_INTERVAL);
            }
            Err(err) => {
                kill_process_group(&mut child);
                return Err(MscError::new(
                    MscErrorKind::Io,
                    format!("msc could not be waited for: {err}"),
                ));
            }
        }
    };
    Ok(BoundedOutput {
        status,
        stdout: drain_reader(stdout),
        stderr: drain_reader(stderr),
    })
}

/// Kills a child's process group, then the child itself, and reaps it.
fn kill_process_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(child.id()) {
            // SAFETY: `pid` is the live child's process id and the child was
            // spawned into its own process group (`process_group(0)`), so the
            // negative pid names exactly that group. An already-gone group is
            // ignored and followed by the direct kill/wait below.
            unsafe {
                kill(-pid, SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn_diagnostic_reader(mut pipe: impl Read + Send + 'static) -> Receiver<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    let spawned = thread::Builder::new()
        .name("victory-msc-stderr".to_string())
        .spawn(move || {
            let mut ring: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        ring.extend_from_slice(&chunk[..read]);
                        if ring.len() > MSC_DIAGNOSTIC_CAPTURE_BYTES {
                            let overflow = ring.len() - MSC_DIAGNOSTIC_CAPTURE_BYTES;
                            ring.drain(..overflow);
                        }
                    }
                }
            }
            let _ = tx.send(ring);
        });
    if let Err(err) = spawned {
        tracing::warn!(error = %err, "victory-metal could not read MSC stderr");
    }
    rx
}

fn drain_reader(reader: Option<Receiver<Vec<u8>>>) -> Vec<u8> {
    reader
        .and_then(|rx| rx.recv_timeout(Duration::from_millis(500)).ok())
        .unwrap_or_default()
}

fn describe_status(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("signal {signal}");
        }
    }
    match status.code() {
        Some(code) => format!("exit code {code}"),
        None => "unknown exit status".to_string(),
    }
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return false;
        }
    }
    true
}

/// Reads one bounded regular file, refusing empty or oversized output.
fn read_bounded(path: &Path, limit: usize, label: &str) -> Result<Vec<u8>, MscError> {
    let metadata = std::fs::metadata(path).map_err(|err| {
        MscError::new(
            MscErrorKind::InvalidOutput,
            format!("msc did not write its {label}: {err}"),
        )
    })?;
    if !metadata.is_file() {
        return Err(MscError::new(
            MscErrorKind::InvalidOutput,
            format!("msc {label} is not a regular file"),
        ));
    }
    if metadata.len() == 0 {
        return Err(MscError::new(
            MscErrorKind::InvalidOutput,
            format!("msc {label} is empty"),
        ));
    }
    if metadata.len() > limit as u64 {
        return Err(MscError::new(
            MscErrorKind::InvalidOutput,
            format!("msc {label} exceeds the {limit}-byte limit"),
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    std::fs::File::open(path)
        .and_then(|file| file.take(limit as u64 + 1).read_to_end(&mut bytes))
        .map_err(|err| {
            MscError::new(
                MscErrorKind::InvalidOutput,
                format!("msc {label} could not be read: {err}"),
            )
        })?;
    if bytes.is_empty() || bytes.len() > limit {
        return Err(MscError::new(
            MscErrorKind::InvalidOutput,
            format!("msc {label} has an invalid size"),
        ));
    }
    Ok(bytes)
}

/// Bounded scratch directory under the OS temporary directory.
struct MscScratch(PathBuf);

impl MscScratch {
    fn new() -> Result<Self, MscError> {
        let sequence = MSC_SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "victory-msc-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).map_err(|err| {
            MscError::new(
                MscErrorKind::Io,
                format!("msc could not create its scratch directory: {err}"),
            )
        })?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for MscScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Flattens one MSC diagnostic into a bounded, path-scrubbed single line.
pub(crate) fn sanitize_msc_reason(text: &str, max: usize) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
        let mapped = match ch {
            '\n' | '\r' | '\t' => ' ',
            ch if ch.is_control() => '?',
            ch => ch,
        };
        if mapped == ' ' {
            if !last_space && !cleaned.is_empty() {
                cleaned.push(' ');
            }
            last_space = true;
        } else {
            cleaned.push(mapped);
            last_space = false;
        }
    }
    let redacted: Vec<String> = cleaned
        .trim()
        .split(' ')
        .map(|token| {
            let (prefix, value) = match token.split_once('=') {
                Some((prefix, value)) if !prefix.is_empty() => (Some(prefix), value),
                _ => (None, token),
            };
            let core = value.trim_matches(|ch: char| {
                matches!(
                    ch,
                    '\'' | '"' | '`' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '<' | '>' | '.'
                )
            });
            if core.starts_with('/') || core.starts_with("~/") {
                match prefix {
                    Some(prefix) => format!("{prefix}=<path>"),
                    None => "<path>".to_string(),
                }
            } else {
                token.to_string()
            }
        })
        .collect();
    let joined = redacted.join(" ");
    if joined.len() <= max {
        return joined;
    }
    let mut start = joined.len() - max;
    while start < joined.len() && !joined.is_char_boundary(start) {
        start += 1;
    }
    format!("{} [truncated]", &joined[start..])
}

#[cfg(test)]
mod tests {
    use super::{
        MSC_ARGUMENT_BUFFER_LENGTH, MSC_DEVICE_ADDRESS_SIZE, MscUavLayout, checked_binding_range,
        write_uav_address,
    };
    use crate::transport::MetalError;
    use crate::wire;

    #[test]
    fn default_layout_matches_verified_harness() {
        let layout = MscUavLayout::DEFAULT;
        assert_eq!(layout.argument_buffer_index(), 2);
        assert_eq!(layout.address_offset(), 0);
        assert_eq!(layout.argument_buffer_len(), 32);
        assert_eq!(MscUavLayout::default(), layout);
    }

    #[test]
    fn layout_rejects_out_of_range_index() {
        assert!(matches!(
            MscUavLayout::new(31, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(usize::MAX, 0),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(MscUavLayout::new(30, 0).is_ok());
    }

    #[test]
    fn layout_rejects_misaligned_or_oversized_address_offset() {
        assert!(matches!(
            MscUavLayout::new(2, 1),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(2, MSC_ARGUMENT_BUFFER_LENGTH - MSC_DEVICE_ADDRESS_SIZE + 1),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            MscUavLayout::new(2, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(MscUavLayout::new(2, MSC_ARGUMENT_BUFFER_LENGTH - MSC_DEVICE_ADDRESS_SIZE).is_ok());
    }

    #[test]
    fn range_validation_matches_registry_contract() {
        assert_eq!(
            checked_binding_range(0, 8, 8).expect("in-bounds range"),
            (0, 8)
        );
        assert_eq!(
            checked_binding_range(4, 8, 12).expect("in-bounds offset range"),
            (4, 12)
        );
        assert!(matches!(
            checked_binding_range(0, 0, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(2, 4, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(8, 4, 8),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(0, wire::MAX_RESOURCE_LEN + 1, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
        assert!(matches!(
            checked_binding_range(u64::MAX - 3, 8, usize::MAX),
            Err(MetalError::InvalidArg(_))
        ));
    }

    #[test]
    fn uav_address_is_encoded_little_endian_at_the_layout_offset() {
        let mut arguments = [0u8; MSC_ARGUMENT_BUFFER_LENGTH];
        let address = 0x1122_3344_5566_7788u64;
        write_uav_address(&mut arguments, MscUavLayout::DEFAULT, address)
            .expect("default layout must fit the verified argument buffer");
        assert_eq!(
            &arguments[..MSC_DEVICE_ADDRESS_SIZE],
            &address.to_le_bytes()
        );
        assert!(arguments[MSC_DEVICE_ADDRESS_SIZE..].iter().all(|&b| b == 0));

        let shifted = MscUavLayout::new(2, 8).expect("offset 8 is aligned and fits");
        let mut arguments = [0u8; MSC_ARGUMENT_BUFFER_LENGTH];
        write_uav_address(&mut arguments, shifted, address).expect("shifted layout must fit");
        assert_eq!(&arguments[..8], &[0u8; 8]);
        assert_eq!(&arguments[8..16], &address.to_le_bytes());
    }

    #[test]
    fn address_encoding_rejects_a_short_slice() {
        let mut arguments = [0u8; MSC_DEVICE_ADDRESS_SIZE - 1];
        assert!(matches!(
            write_uav_address(&mut arguments, MscUavLayout::DEFAULT, 1),
            Err(MetalError::InvalidArg(_))
        ));
    }
}
