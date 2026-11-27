//! Guest PSO registration and IRConverter pool wiring (ADR-018/ADR-019).
//!
//! Two things happen here and nowhere else on the guest:
//!
//! 1. a title shader container's DXIL is translated to a metallib through the
//!    persistent `--stdio` [`CompilerPool`], so repeated shader creation reuses
//!    the same Rosetta worker instead of starting one process per shader;
//! 2. the metallib is handed to the host as transport kind 4
//!    (`{ticket:016x}.metallib` in the launcher's share directory) and the
//!    native pipeline id the host answers with (kind 129) is what
//!    `SetComputePipeline`, `SetPipeline` or `SetMeshPipeline` names.
//!
//! # Pipeline families
//!
//! A compute PSO with no converter reflection keeps the original version 1
//! frame: one `ticket + entry` and the host's `load_metallib` compute path. A
//! compute PSO that has a `VAB1` ABI (the normal case now that the `--stdio`
//! worker returns reflection) sends the version 2 frame instead, so the host
//! stores a `PipelineAbi` and the executor uploads the argument buffer that
//! makes the kernel's root UAV reach the converted MSL.
//!
//! A graphics or mesh PSO sends the version 2 frame: one `(stage, ticket,
//! entry)` per stage, the encoded `PipelineState` (`VPS1`), the encoded
//! per-stage ABI (`VAB1`) and the root-layout witness bytes. The host reads one
//! library per stage and creates the pipeline through
//! `register_content_hashed_graphics_pipeline` / `..._mesh_pipeline`, so the
//! version 2 path is the same registry the live v3 executor resolves.
//!
//! A stage set that cannot form one valid pipeline (a vertex without a pixel,
//! a hull/domain/geometry stage, ...) is [`PipelineError::HostCannotRegister`],
//! naming the failing stage. It is never a silent skip and never a D3DMetal
//! fallback.
//!
//! # Entry point
//!
//! The kind 4 entry name is the DXIL entry the converter was told to convert
//! and the name the host looks up in the metallib. The container does not
//! spell it out, and every container this repository has converted uses
//! `main`, so that is the default; `VICTORY_METAL_PSO_ENTRY` overrides it for
//! a title that names its entry differently. (Hypothesis, not verified against
//! a real title's container yet.)
//!
//! # Where the pool can run
//!
//! [`CompilerPool`] starts `victory-irconverter --stdio` children. That works
//! on a macOS process (the aarch64 host and the in-process tests). The Win64
//! guest is a Windows PE and cannot load the x86_64 macOS converter, so there
//! the pool is cache-only: a shader the shared disk cache already holds is
//! served (warm), and a cold shader is [`PipelineError::ConverterUnavailable`]
//! — an explicit failure that names the missing converter. The guest never
//! fabricates a metallib and never falls back to the MSL `shader_cache` path
//! for a *native* pipeline: that path keeps feeding the existing draw path it
//! always fed.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use victory_metal::compiler_pool::{
    CompileJob, CompiledShader, CompilerPool, PoolError, PoolStats,
};
#[cfg(target_os = "macos")]
use victory_metal::macos::bindings::RootSignatureLayout;

/// The shader stage a wire pipeline registration names.
///
/// The numbers are the converter's own stage codes, the ones the pool forwards
/// to `victory-irconverter --stdio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Vertex shader (converter stage 1).
    Vertex,
    /// Pixel/fragment shader (converter stage 2).
    Pixel,
    /// Mesh shader (converter stage 5).
    Mesh,
    /// Compute shader (converter stage 8).
    Compute,
}

impl PipelineStage {
    /// The converter's stage code for this stage.
    pub const fn converter_stage(self) -> u32 {
        match self {
            Self::Vertex => 1,
            Self::Pixel => 2,
            Self::Mesh => 5,
            Self::Compute => 8,
        }
    }

    /// The version 2 wire stage code for this stage.
    pub const fn wire_code(self) -> u8 {
        match self {
            Self::Vertex => crate::native_metal::client::PIPELINE_STAGE_VERTEX,
            Self::Pixel => crate::native_metal::client::PIPELINE_STAGE_PIXEL,
            Self::Mesh => crate::native_metal::client::PIPELINE_STAGE_MESH,
            Self::Compute => crate::native_metal::client::PIPELINE_STAGE_COMPUTE,
        }
    }

    /// The stage name the title pipeline uses (`pipeline::Shader::stage`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "vertex" => Some(Self::Vertex),
            "pixel" => Some(Self::Pixel),
            "mesh" => Some(Self::Mesh),
            "compute" => Some(Self::Compute),
            _ => None,
        }
    }
}

/// One stage of a title PSO: its DXIL and the entry point the converter and
/// the host both name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageProgram {
    /// The stage this program belongs to.
    pub stage: PipelineStage,
    /// Raw DXIL bytes (the container's `DXIL` part).
    pub dxil: Vec<u8>,
    /// Entry point name inside the DXIL module.
    pub entry: String,
}

/// Why a PSO could not be registered with the native host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// A stage has no DXIL part (a console-only container).
    NoDxil(&'static str),
    /// The container could not be parsed.
    Container(String),
    /// No worker program and nothing in the disk cache: the converter this
    /// build needs is not available to this process.
    ConverterUnavailable(String),
    /// The pool failed for some other reason.
    Translation(String),
    /// The share directory the host reads is not configured.
    NoShareDir,
    /// Writing the metallib the host will open failed.
    Write(String),
    /// The host refused the registration.
    Refused(String),
    /// The transport failed.
    Transport(String),
    /// The stage set cannot be expressed as one native pipeline.
    HostCannotRegister {
        /// The stage that cannot be registered.
        stage: PipelineStage,
        /// What the pipeline is missing.
        missing: &'static str,
    },
    /// The converter reflection is missing, malformed or cannot be turned into
    /// a `VAB1` ABI. A PSO is never registered with a silently empty layout.
    Reflection(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDxil(stage) => write!(f, "the {stage} shader has no DXIL part"),
            Self::Container(why) => write!(f, "shader container: {why}"),
            Self::ConverterUnavailable(why) => {
                write!(f, "no IRConverter is available to this process: {why}")
            }
            Self::Translation(why) => write!(f, "DXIL translation failed: {why}"),
            Self::NoShareDir => write!(
                f,
                "pipeline registration needs VICTORY_METAL_SHARE_DIR (the host's metallib directory)"
            ),
            Self::Write(why) => write!(f, "metallib could not be written: {why}"),
            Self::Refused(why) => write!(f, "the host refused the pipeline: {why}"),
            Self::Transport(why) => write!(f, "pipeline transport: {why}"),
            Self::HostCannotRegister { stage, missing } => write!(
                f,
                "a {} PSO cannot be registered: {missing} (failing stage: {stage:?})",
                match stage {
                    PipelineStage::Vertex | PipelineStage::Pixel => "graphics",
                    PipelineStage::Mesh => "mesh",
                    PipelineStage::Compute => "compute",
                }
            ),
            Self::Reflection(why) => write!(f, "converter reflection: {why}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// One pipeline's encoded state, ABI and raw root-signature bytes.
///
/// A caller with converter reflection can pass the exact `PipelineState` and
/// `VAB1` ABI it wants; an empty `state` means "use the stage set's default"
/// (RGBA8 colour, no depth, the pixel entry as the fragment function).
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineDetails<'a> {
    /// Encoded `PipelineState` bytes (`VPS1`); empty selects the default.
    pub state: &'a [u8],
    /// Encoded `VAB1` per-stage ABI container; empty means no auto-binding.
    pub abi: &'a [u8],
    /// Raw root-signature bytes; required when `abi` is non-empty.
    pub root_signature: &'a [u8],
}

/// Stable `TextureFormat::Rgba8Unorm` code used by the default state.
pub const DEFAULT_COLOR_FORMAT: u8 = 7;
/// Wire stage code for a vertex shader.
pub const WIRE_STAGE_VERTEX: u8 = 0;
/// Wire stage code for a pixel/fragment shader.
pub const WIRE_STAGE_PIXEL: u8 = 1;
/// Wire stage code for a mesh shader.
pub const WIRE_STAGE_MESH: u8 = 2;
/// Wire stage code for a compute shader.
pub const WIRE_STAGE_COMPUTE: u8 = crate::native_metal::client::PIPELINE_STAGE_COMPUTE;

/// Encodes a graphics `PipelineState` (`VPS1`) for the stage set.
///
/// Mirrors `victory_metal::macos::registry::PipelineState::encode(Graphics)`:
/// magic, stage code `0`, colour formats, no depth, a fragment entry, no
/// object entry and zero mesh budgets.
pub fn encode_graphics_state(color_formats: &[u8], fragment_entry: &str) -> Vec<u8> {
    encode_state(0, color_formats, fragment_entry, None, [0; 3], [0; 3])
}

/// Encodes a mesh `PipelineState` (`VPS1`) for the stage set.
///
/// Mirrors `PipelineState::encode(Mesh)`: the fragment entry is required and
/// the mesh/object threadgroups are set; the object entry is supplied by a
/// caller that has an object stage.
pub fn encode_mesh_state(color_formats: &[u8], fragment_entry: &str) -> Vec<u8> {
    encode_state(
        1,
        color_formats,
        fragment_entry,
        None,
        [32, 1, 1],
        [1, 1, 1],
    )
}

fn encode_state(
    stage: u8,
    color_formats: &[u8],
    fragment_entry: &str,
    object_entry: Option<&str>,
    mesh_threadgroup: [u16; 3],
    object_threadgroup: [u16; 3],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + fragment_entry.len());
    out.extend_from_slice(b"VPS1");
    out.push(stage);
    out.push(u8::try_from(color_formats.len()).expect("colour format count exceeds u8"));
    out.extend_from_slice(color_formats);
    out.push(0); // no depth attachment
    let object = object_entry.unwrap_or("");
    out.push(u8::from(!fragment_entry.is_empty()) | (u8::from(!object.is_empty()) << 1));
    push_u16_string(&mut out, fragment_entry);
    push_u16_string(&mut out, object);
    for value in mesh_threadgroup.iter().chain(object_threadgroup.iter()) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&0u16.to_le_bytes()); // payload memory length
    out.extend_from_slice(&0u32.to_le_bytes()); // max threadgroups per mesh grid
    out
}

fn push_u16_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(
        &u16::try_from(text.len())
            .expect("pipeline state string exceeds u16")
            .to_le_bytes(),
    );
    out.extend_from_slice(text.as_bytes());
}

/// Encodes the `VAB1` per-stage ABI container from converter reflection JSON.
///
/// `stages` is `(wire stage, reflection JSON)`; the host parses each JSON with
/// `RootSignatureLayout::from_converter_reflection_json`. An empty slice is an
/// empty container, not a blob: the host distinguishes "no ABI" from a
/// malformed one by the frame's length.
pub fn encode_abi(stages: &[(u8, &str)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + stages.len() * 8);
    out.extend_from_slice(b"VAB1");
    out.push(1);
    out.push(u8::try_from(stages.len()).expect("ABI stage count exceeds u8"));
    for (stage, json) in stages {
        out.push(*stage);
        out.extend_from_slice(
            &u32::try_from(json.len())
                .expect("reflection JSON length exceeds u32")
                .to_le_bytes(),
        );
        out.extend_from_slice(json.as_bytes());
    }
    out
}

/// The process-wide pool. One pool means one persistent worker set for the
/// whole run (ADR-018); it is created from the environment on first use.
fn pool() -> &'static CompilerPool {
    static POOL: OnceLock<CompilerPool> = OnceLock::new();
    POOL.get_or_init(CompilerPool::from_env)
}

/// The pool's statistics before and after one translation, so a caller can say
/// whether the disk cache or a worker served it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolServing {
    /// `true`: the disk cache answered; `false`: a `--stdio` worker ran.
    pub warm: bool,
}

/// The share directory the launcher created for this session, if any.
pub fn share_dir() -> Option<PathBuf> {
    std::env::var_os(crate::native_metal::fence_page_guest::SHARE_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The file the host opens for allocation `id` (`ipc_host::FileMapping::open`).
pub fn allocation_path(id: u64) -> Option<PathBuf> {
    share_dir().map(|dir| dir.join(format!("{id:016x}.bin")))
}

/// The whole length of the share file allocation `id` names, in bytes.
///
/// The host revalidates `metadata.len() == byte_len` before mapping, so this is
/// the only `byte_len` a kind 2 registration may carry. The id comes from the
/// `GuestRangeResolver` seam; the length comes from the file that id names.
pub fn allocation_len(id: u64) -> Result<u64, String> {
    let path = allocation_path(id).ok_or_else(|| {
        "no VICTORY_METAL_SHARE_DIR: the guest cannot name the allocation's share file".to_string()
    })?;
    let metadata = std::fs::metadata(&path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    Ok(metadata.len())
}

/// Every share allocation with an id at or below `up_to`, ascending by id.
///
/// The host accepts only strictly increasing allocation ids and closes the
/// session on a refused kind 2 frame, so the guest registers the allocations a
/// title already made in ascending order before the one it just asked for. A
/// section created later always gets a higher id (the share provider's counter
/// only moves forward), so sweeping up to the requested id keeps every later
/// registration ascending too.
///
/// The directory must be the session's own, as the launcher creates it: a
/// reused directory holding files from an earlier process would carry ids this
/// process never created (and its own ids would restart below them).
pub fn shares_up_to(up_to: u64) -> Result<Vec<(u64, u64)>, String> {
    let dir = share_dir().ok_or_else(|| {
        "no VICTORY_METAL_SHARE_DIR: the guest cannot name the session's allocations".to_string()
    })?;
    let entries = std::fs::read_dir(&dir)
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", dir.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(stem) = name.strip_suffix(".bin") else { continue };
        if stem.len() != 16 || !stem.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        let Ok(id) = u64::from_str_radix(stem, 16) else { continue };
        if id == 0 || id > up_to {
            continue;
        }
        let len = allocation_len(id)?;
        if len != 0 {
            found.push((id, len));
        }
    }
    found.sort_unstable_by_key(|(id, _)| *id);
    Ok(found)
}

/// The metallib file name the host derives from `ticket`.
pub fn metallib_name(ticket: u64) -> String {
    format!("{ticket:016x}.metallib")
}

/// The metallib path the host reads for `ticket`.
pub fn metallib_path(ticket: u64) -> Option<PathBuf> {
    share_dir().map(|dir| dir.join(metallib_name(ticket)))
}

/// Writes `bytes` to `path` atomically: temporary name, then rename, so the
/// host never maps a half-written library.
pub fn write_metallib(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file = path.to_string_lossy();
    let temporary = PathBuf::from(format!("{file}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("{}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("{}: {error}", path.display())
    })
}

/// The DXIL part and entry point of one shader container.
///
/// A container with no `DXIL` part (console GCN only) is
/// [`PipelineError::NoDxil`]: there is nothing the converter could translate.
pub fn stage_program(
    stage: PipelineStage,
    container: &[u8],
) -> Result<StageProgram, PipelineError> {
    let parsed = victory_shader::Container::parse(container)
        .map_err(|error| PipelineError::Container(error.to_string()))?;
    let part = parsed
        .part(victory_shader::FourCC::DXIL)
        .ok_or(PipelineError::NoDxil(match stage {
            PipelineStage::Vertex => "vertex",
            PipelineStage::Pixel => "pixel",
            PipelineStage::Mesh => "mesh",
            PipelineStage::Compute => "compute",
        }))?;
    Ok(StageProgram {
        stage,
        dxil: part.data.to_vec(),
        entry: entry_point(),
    })
}

/// The entry point named in every converted DXIL module this repository has
/// seen (`main`), overridable with `VICTORY_METAL_PSO_ENTRY`.
pub fn entry_point() -> String {
    std::env::var("VICTORY_METAL_PSO_ENTRY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "main".to_string())
}

/// The pool statistics, for telemetry and tests.
pub fn pool_stats() -> PoolStats {
    pool().stats()
}

/// The worker program the pool will start, when one is configured.
///
/// The pool resolves it from `VICTORY_IRCONVERTER_WORKER` or the
/// `victory-irconverter` sibling of this executable. The Win64 guest has
/// neither (the converter and its worker are macOS builds), so this is `None`
/// there and every cold translation is explicit.
pub fn pool_worker_program() -> Option<PathBuf> {
    pool().worker_program().map(Path::to_path_buf)
}

/// The x86_64 converter library a worker would load.
///
/// Resolved the same way the pool does: `VICTORY_METAL_IRCONVERTER` names the
/// library itself, `D3DMETAL_RES` names a resource directory holding
/// `libmetalirconverter.dylib`. `None` means the environment names nothing (the
/// worker reports the missing library itself).
pub fn converter_library() -> Option<PathBuf> {
    std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                .filter(|path| path.is_file())
        })
}

/// Whether this process can translate DXIL at all: a worker program to run and
/// a converter library to load, or nothing (cache-only, explicit refusal).
pub fn converter_available() -> bool {
    pool_worker_program().is_some() && converter_library().is_some()
}

/// The per-worker job counts, for tests that prove worker reuse.
pub fn pool_worker_jobs() -> Vec<u64> {
    pool().worker_job_counts()
}

/// Maps a pool failure onto the PSO path's explicit error vocabulary.
fn translate_error(error: PoolError) -> PipelineError {
    match error {
        PoolError::NoWorkerProgram => PipelineError::ConverterUnavailable(error.to_string()),
        other => PipelineError::Translation(other.to_string()),
    }
}

/// Translates one stage's DXIL to a metallib, reporting whether the cache or a
/// worker served it.
///
/// This is the metallib-only API; a pipeline that needs root bindings uses
/// [`translate_with_reflection`].
pub fn translate(program: &StageProgram) -> Result<(Vec<u8>, PoolServing), PipelineError> {
    let before = pool().stats();
    let job = compile_job(program);
    let metallib = pool().compile(&job).map_err(translate_error)?;
    let after = pool().stats();
    Ok((
        metallib,
        PoolServing {
            warm: after.cache_hits > before.cache_hits,
        },
    ))
}

/// Translates one stage and returns the converter reflection JSON with the
/// metallib.
///
/// The reflection is what [`encode_abi`] turns into the per-stage `VAB1`
/// container the host parses with
/// [`victory_metal::macos::bindings::RootSignatureLayout`]. A worker that
/// cannot provide it is an explicit [`PipelineError::Translation`], never a
/// pipeline registered with an empty ABI.
pub fn translate_with_reflection(
    program: &StageProgram,
) -> Result<(CompiledShader, PoolServing), PipelineError> {
    let before = pool().stats();
    let job = compile_job(program);
    let compiled = pool()
        .compile_with_reflection(&job)
        .map_err(translate_error)?;
    let after = pool().stats();
    Ok((
        compiled,
        PoolServing {
            warm: after.cache_hits > before.cache_hits,
        },
    ))
}

fn compile_job(program: &StageProgram) -> CompileJob<'_> {
    CompileJob {
        id: job_id(program),
        stage: program.stage.converter_stage(),
        entry: &program.entry,
        dxil: &program.dxil,
    }
}

/// The pool job id: the DXIL length and stage, stable for one program and
/// enough for the worker to echo back what it answered.
fn job_id(program: &StageProgram) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&program.stage.converter_stage().to_le_bytes());
    hasher.update(program.entry.as_bytes());
    hasher.update(&program.dxil);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("eight bytes"))
}

/// Registers one PSO's programs with the host and returns the native id.
///
/// A compute PSO keeps the original v1 `ticket + entry` frame. A graphics or
/// mesh PSO translates each stage, writes one library per stage and sends the
/// version 2 frame with the encoded state, ABI and root-signature bytes.
#[cfg(any(windows, test))]
pub fn register(programs: &[StageProgram]) -> Result<(u64, PoolServing), PipelineError> {
    register_with_details(programs, &PipelineDetails::default())
}

/// [`register`] with explicit state/ABI/root-signature bytes.
///
/// Each stage is translated with reflection. When the caller supplies no ABI,
/// the reflections are parsed with
/// [`RootSignatureLayout::from_converter_reflection_json`] and encoded into the
/// `VAB1` container, so a compute kernel's UAV/CBV/SRV root entries and a
/// graphics/mesh pipeline's root entries reach the host automatically. A
/// reflection that does not parse fails the whole registration; it is never
/// dropped into an empty ABI.
#[cfg(any(windows, test))]
pub fn register_with_details(
    programs: &[StageProgram],
    details: &PipelineDetails<'_>,
) -> Result<(u64, PoolServing), PipelineError> {
    let mut warm = true;
    let mut translated: Vec<(PipelineStage, CompiledShader, &str)> =
        Vec::with_capacity(programs.len());
    for program in programs {
        let (compiled, serving) = translate_with_reflection(program)?;
        warm &= serving.warm;
        translated.push((program.stage, compiled, &program.entry));
    }

    let reflections: Vec<(u8, &str)> = translated
        .iter()
        .map(|(stage, compiled, _)| {
            std::str::from_utf8(&compiled.reflection)
                .map(|json| (stage.wire_code(), json))
                .map_err(|_| {
                    PipelineError::Reflection(format!(
                        "{stage:?}: converter reflection is not UTF-8"
                    ))
                })
        })
        .collect::<Result<_, PipelineError>>()?;
    // Parse every reflection before anything is written: an unusable layout is
    // an explicit failure, not a pipeline with auto-binding silently disabled.
    //
    // The Win64 guest build has no `macos` module (and cannot reach a
    // converter), so it cannot parse here; the host's `VAB1` decoder is then
    // the validator and refuses the registration explicitly.
    #[cfg(target_os = "macos")]
    for (stage, json) in &reflections {
        RootSignatureLayout::from_converter_reflection_json(json)
            .map_err(|error| PipelineError::Reflection(format!("stage {stage}: {error}")))?;
    }
    #[cfg(not(target_os = "macos"))]
    for (stage, json) in &reflections {
        if json.trim().is_empty() {
            return Err(PipelineError::Reflection(format!(
                "stage {stage}: empty reflection JSON"
            )));
        }
    }

    let derived_abi;
    let derived_layout_witness;
    let (abi, root_signature) = if details.abi.is_empty()
        && details.root_signature.is_empty()
        && !reflections.is_empty()
    {
        derived_abi = encode_abi(&reflections);
        derived_layout_witness = reflection_layout_witness(&reflections)?;
        (derived_abi.as_slice(), derived_layout_witness.as_slice())
    } else {
        (details.abi, details.root_signature)
    };

    let stages: Vec<(PipelineStage, &[u8], &str)> = translated
        .iter()
        .map(|(stage, compiled, entry)| (*stage, compiled.metallib.as_slice(), *entry))
        .collect();
    let translated_details = PipelineDetails {
        state: details.state,
        abi,
        root_signature,
    };
    let id = register_translated(&stages, &translated_details)?;
    Ok((id, PoolServing { warm }))
}

/// The non-empty bytes the version 2 frame carries next to a `VAB1` ABI.
///
/// The host requires a non-empty `root_signature` field whenever the ABI is
/// non-empty, but it only validates the field's length: the registry binds
/// from the parsed reflection, never from these bytes. The converter consumed
/// the root signature embedded in the DXIL, and the reflection JSON is its
/// flattened description, so that document is the honest witness when the
/// caller supplied no raw root-signature blob of its own.
fn reflection_layout_witness(reflections: &[(u8, &str)]) -> Result<Vec<u8>, PipelineError> {
    let (stage, json) = reflections.first().ok_or_else(|| {
        PipelineError::Reflection("no stage reflection to witness a root layout".to_string())
    })?;
    if json.is_empty() {
        return Err(PipelineError::Reflection(format!(
            "stage {stage}: empty reflection JSON"
        )));
    }
    if json.len() > crate::native_metal::client::MAX_ROOT_SIGNATURE_BYTES {
        return Err(PipelineError::Reflection(format!(
            "stage {stage}: reflection JSON is {} bytes, over the host's {}-byte root-signature \
             field",
            json.len(),
            crate::native_metal::client::MAX_ROOT_SIGNATURE_BYTES
        )));
    }
    Ok(json.as_bytes().to_vec())
}

/// Registers already-translated stages (one metallib per stage).
///
/// This is the seam tests and callers with a prebuilt library use. A compute
/// stage with no ABI keeps the v1 `ticket + entry` frame; a compute stage with
/// a `VAB1` ABI, and every graphics/mesh stage, sends the version 2 frame that
/// carries the state, ABI and root-layout witness. The stage set and the
/// ABI/root pair are validated before anything is written, so an incomplete
/// pipeline or a layout that cannot bind is an explicit error naming the
/// failing part.
#[cfg(any(windows, test))]
pub fn register_translated(
    stages: &[(PipelineStage, &[u8], &str)],
    details: &PipelineDetails<'_>,
) -> Result<u64, PipelineError> {
    let stage_names: Vec<PipelineStage> = stages.iter().map(|(stage, _, _)| *stage).collect();
    let kind = pipeline_kind(&stage_names)?;
    if !details.abi.is_empty() && details.root_signature.is_empty() {
        return Err(PipelineError::Reflection(
            "a non-empty VAB1 ABI needs the root-layout witness bytes the version 2 frame \
             carries"
                .to_string(),
        ));
    }
    if kind == PipelineKind::Compute && details.abi.is_empty() {
        // No reflection was available: keep the original v1 frame. The host
        // binds nothing for this pipeline, which is exactly the old behaviour.
        let (_, metallib, entry) = stages[0];
        return crate::native_metal::queue::register_pipeline(metallib, entry).map_err(|error| {
            match error {
                crate::native_metal::queue::NativeQueueError::Refused(why) => {
                    PipelineError::Refused(why)
                }
                other => PipelineError::Transport(other.to_string()),
            }
        });
    }
    for (stage, metallib, _) in stages {
        if metallib.is_empty() {
            return Err(PipelineError::Refused(format!(
                "the {stage:?} stage has no metallib"
            )));
        }
    }
    let wire: Vec<crate::native_metal::queue::PipelineStageLibrary<'_>> = stages
        .iter()
        .map(|(stage, metallib, entry)| crate::native_metal::queue::PipelineStageLibrary {
            stage: stage.wire_code(),
            metallib,
            entry,
        })
        .collect();
    let pixel_entry = stages
        .iter()
        .find(|(stage, _, _)| *stage == PipelineStage::Pixel)
        .map(|(_, _, entry)| *entry)
        .unwrap_or_default();
    let kind_code = match kind {
        PipelineKind::Graphics => crate::native_metal::client::PIPELINE_KIND_GRAPHICS,
        PipelineKind::Mesh => crate::native_metal::client::PIPELINE_KIND_MESH,
        PipelineKind::Compute => crate::native_metal::client::PIPELINE_KIND_COMPUTE,
    };
    let state_storage: Vec<u8>;
    let state: &[u8] = if !details.state.is_empty() {
        details.state
    } else if kind == PipelineKind::Compute {
        // A compute `PipelineState` carries no colour/depth shape; the host
        // accepts an empty state for a compute pipeline.
        &[]
    } else {
        state_storage = if kind == PipelineKind::Mesh {
            encode_mesh_state(&[DEFAULT_COLOR_FORMAT], pixel_entry)
        } else {
            encode_graphics_state(&[DEFAULT_COLOR_FORMAT], pixel_entry)
        };
        &state_storage
    };
    crate::native_metal::queue::register_pipeline_v2(
        kind_code,
        &wire,
        state,
        details.abi,
        details.root_signature,
    )
    .map_err(|error| match error {
        crate::native_metal::queue::NativeQueueError::Refused(why) => {
            PipelineError::Refused(why)
        }
        other => PipelineError::Transport(other.to_string()),
    })
}

/// The native pipeline family a stage set describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
    Graphics,
    Mesh,
    Compute,
}

/// Validates the stage set and names the family it describes.
///
/// Only the stages the native registration understands are accepted; a set
/// that cannot form one valid graphics/mesh/compute pipeline is refused with
/// the stage that is missing or unsupported.
fn pipeline_kind(stages: &[PipelineStage]) -> Result<PipelineKind, PipelineError> {
    match stages {
        [PipelineStage::Compute] => Ok(PipelineKind::Compute),
        [PipelineStage::Vertex, PipelineStage::Pixel]
        | [PipelineStage::Pixel, PipelineStage::Vertex] => Ok(PipelineKind::Graphics),
        [PipelineStage::Mesh, PipelineStage::Pixel]
        | [PipelineStage::Pixel, PipelineStage::Mesh] => Ok(PipelineKind::Mesh),
        [] => Err(PipelineError::Translation(
            "no shader stage to register".to_string(),
        )),
        [stage] => Err(PipelineError::HostCannotRegister {
            stage: *stage,
            missing: match stage {
                PipelineStage::Vertex => "a graphics pipeline also needs a pixel stage",
                PipelineStage::Pixel => "a render pipeline also needs a vertex or mesh stage",
                PipelineStage::Mesh => "a mesh pipeline also needs a pixel stage",
                PipelineStage::Compute => "a compute pipeline needs exactly one stage",
            },
        }),
        _ => {
            let stage = *stages
                .iter()
                .find(|stage| {
                    !matches!(
                        stage,
                        PipelineStage::Vertex
                            | PipelineStage::Pixel
                            | PipelineStage::Mesh
                            | PipelineStage::Compute
                    )
                })
                .unwrap_or(&stages[stages.len() - 1]);
            Err(PipelineError::HostCannotRegister {
                stage,
                missing: "the native path registers vertex+pixel, mesh+pixel or one compute \
                          stage only",
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_codes_match_the_converter() {
        assert_eq!(PipelineStage::Vertex.converter_stage(), 1);
        assert_eq!(PipelineStage::Pixel.converter_stage(), 2);
        assert_eq!(PipelineStage::Mesh.converter_stage(), 5);
        assert_eq!(PipelineStage::Compute.converter_stage(), 8);
        assert_eq!(
            PipelineStage::from_name("compute"),
            Some(PipelineStage::Compute)
        );
        assert_eq!(PipelineStage::from_name("hull"), None);
    }

    #[test]
    fn a_container_without_dxil_is_refused() {
        let error = stage_program(PipelineStage::Compute, b"not a container")
            .expect_err("a non-container is a container error");
        assert!(matches!(error, PipelineError::Container(_)));
    }

    #[test]
    fn an_incomplete_stage_set_names_the_failing_stage() {
        let error = pipeline_kind(&[PipelineStage::Vertex])
            .expect_err("a vertex-only graphics pipeline is refused");
        let text = error.to_string();
        assert!(text.contains("graphics"), "{text}");
        assert!(text.contains("Vertex"), "{text}");
    }

    #[test]
    fn graphics_and_mesh_state_blobs_are_vps1() {
        let graphics = encode_graphics_state(&[DEFAULT_COLOR_FORMAT], "main");
        assert_eq!(&graphics[..4], b"VPS1");
        assert_eq!(graphics[4], 0, "graphics stage code");
        let mesh = encode_mesh_state(&[DEFAULT_COLOR_FORMAT], "main");
        assert_eq!(&mesh[..4], b"VPS1");
        assert_eq!(mesh[4], 1, "mesh stage code");
    }

    #[test]
    fn abi_container_carries_stage_json() {
        let abi = encode_abi(&[(WIRE_STAGE_VERTEX, r#"{"TopLevelArgumentBuffer":[]}"#)]);
        assert_eq!(&abi[..4], b"VAB1");
        assert_eq!(abi[4], 1);
        assert_eq!(abi[5], 1);
        assert_eq!(abi[6], WIRE_STAGE_VERTEX);
    }

    const COMPUTE_REFLECTION_WITH_UAV: &str = r#"{"EntryPoint":"main","ShaderType":"Compute","TopLevelArgumentBuffer":[{"EltOffset":0,"Name":"","Size":8,"Slot":0,"Space":0,"Type":"UAV"}]}"#;

    /// The compute ABI entry the guest sends is the converter's own
    /// `TopLevelArgumentBuffer` field: offset 0, size 8, slot 0, kind UAV.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_compute_reflection_parses_into_the_uav_the_executor_binds() {
        let layout =
            RootSignatureLayout::from_converter_reflection_json(COMPUTE_REFLECTION_WITH_UAV)
                .expect("the converter reflection parses");
        assert_eq!(layout.parameters().len(), 1);
        let parameter = &layout.parameters()[0];
        assert_eq!(parameter.argument_offset(), 0);
        assert_eq!(parameter.size(), 8);
        assert_eq!(parameter.register(), 0);
        let abi = encode_abi(&[(
            crate::native_metal::client::PIPELINE_STAGE_COMPUTE,
            COMPUTE_REFLECTION_WITH_UAV,
        )]);
        assert_eq!(&abi[..4], b"VAB1");
        assert_eq!(abi[5], 1);
        assert!(
            abi.windows(COMPUTE_REFLECTION_WITH_UAV.len())
                .any(|window| window == COMPUTE_REFLECTION_WITH_UAV.as_bytes()),
            "the container carries the reflection verbatim"
        );
    }

    /// A compute ABI with no root-layout witness would be refused by the host;
    /// the guest rejects it before writing a metallib or sending a frame.
    #[test]
    fn a_compute_abi_without_a_root_witness_is_refused() {
        let abi = encode_abi(&[(
            crate::native_metal::client::PIPELINE_STAGE_COMPUTE,
            COMPUTE_REFLECTION_WITH_UAV,
        )]);
        let details = PipelineDetails {
            state: b"",
            abi: &abi,
            root_signature: b"",
        };
        match register_translated(&[(PipelineStage::Compute, b"MTLB compute", "main")], &details) {
            Err(PipelineError::Reflection(why)) => {
                assert!(why.contains("root-layout witness"), "{why}");
            }
            other => panic!("an unwitnessed ABI must be refused, saw {other:?}"),
        }
    }

    #[test]
    fn a_ticket_names_one_metallib_file() {
        assert_eq!(metallib_name(7), "0000000000000007.metallib");
        assert_eq!(metallib_name(0xdead_beef), "00000000deadbeef.metallib");
    }

    #[test]
    fn a_metallib_is_written_whole_or_not_at_all() {
        let dir = std::env::temp_dir().join(format!("victory-pso-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temporary directory");
        let path = dir.join(metallib_name(3));
        write_metallib(&path, b"MTLB test bytes").expect("the library is written");
        assert_eq!(std::fs::read(&path).expect("read it back"), b"MTLB test bytes");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
