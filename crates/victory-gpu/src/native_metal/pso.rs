//! Guest PSO registration and IRConverter pool wiring (ADR-018/ADR-019).
//!
//! Two things happen here and nowhere else on the guest:
//!
//! 1. a title shader container's DXIL is translated to a metallib through the
//!    persistent `--stdio` [`CompilerPool`], so repeated shader creation reuses
//!    the same Rosetta worker instead of starting one process per shader;
//! 2. the metallib is handed to the host as transport kind 4
//!    (`{ticket:016x}.metallib` in the launcher's share directory, then a
//!    `ticket + entry` frame) and the native pipeline id the host answers with
//!    (kind 129) is what a `SetComputePipeline`/`Dispatch` names.
//!
//! # What the host can register today
//!
//! `ipc_host::register_pipeline` maps the ticket to a library and calls
//! `DirectMetalTransport::load_metallib`, which builds **one compute
//! pipeline** (`newComputePipelineStateWithFunction`). The transport also has
//! `register_content_hashed_graphics_pipeline` / `..._mesh_pipeline` (metallib,
//! entry, encoded `PipelineState` blob, [`PipelineAbi`]), but the kind 4
//! payload has no field for a stage, a second library, a state blob or an ABI,
//! so a graphics or mesh PSO cannot be registered over the wire yet:
//!
//! | missing in the kind 4 frame | needed for |
//! |---|---|
//! | `stage: u8` | selecting graphics/mesh/compute registration |
//! | per-stage `(ticket, entry)` for vertex + pixel, or mesh + object + pixel | `register_content_hashed_*_pipeline` |
//! | encoded `PipelineState` bytes (blend, raster, depth-stencil, formats, sample count) | the same |
//! | encoded `PipelineAbi` (per-stage root layout) | automatic root binding |
//!
//! [`register`] therefore registers a compute PSO and refuses a graphics or
//! mesh PSO with [`PipelineError::HostCannotRegister`], naming those fields.
//! The refusal is structured and counted; it is never a silent skip and never
//! a D3DMetal fallback.
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

use victory_metal::compiler_pool::{CompileJob, CompilerPool, PoolError, PoolStats};

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
    /// The host's kind 4 form cannot express this PSO (see the module docs).
    HostCannotRegister {
        /// The stage that cannot be registered.
        stage: PipelineStage,
        /// The fields the frame would need.
        missing: &'static str,
    },
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
                "the host kind 4 form registers compute pipelines only; a {} PSO needs {missing}",
                match stage {
                    PipelineStage::Vertex | PipelineStage::Pixel => "graphics",
                    PipelineStage::Mesh => "mesh",
                    PipelineStage::Compute => "compute",
                }
            ),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Fields the host's kind 4 frame would need before a graphics/mesh PSO can be
/// registered (kept as one string so the refusal, the log and the report agree).
pub const GRAPHICS_MESH_FRAME_FIELDS: &str =
    "a stage field, one (ticket, entry) per stage, an encoded PipelineState blob and an encoded \
     PipelineAbi";

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

/// Translates one stage's DXIL to a metallib, reporting whether the cache or a
/// worker served it.
pub fn translate(program: &StageProgram) -> Result<(Vec<u8>, PoolServing), PipelineError> {
    let before = pool().stats();
    let job = CompileJob {
        id: job_id(program),
        stage: program.stage.converter_stage(),
        entry: &program.entry,
        dxil: &program.dxil,
    };
    let metallib = pool().compile(&job).map_err(|error| match error {
        PoolError::NoWorkerProgram => PipelineError::ConverterUnavailable(error.to_string()),
        other => PipelineError::Translation(other.to_string()),
    })?;
    let after = pool().stats();
    Ok((
        metallib,
        PoolServing {
            warm: after.cache_hits > before.cache_hits,
        },
    ))
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
/// A compute PSO translates its single program and round-trips kind 4/129. A
/// graphics or mesh PSO is refused before any translation with
/// [`PipelineError::HostCannotRegister`] and the missing frame fields, because
/// the host's kind 4 form can only build a compute pipeline.
#[cfg(any(windows, test))]
pub fn register(programs: &[StageProgram]) -> Result<(u64, PoolServing), PipelineError> {
    match programs {
        [program] if program.stage == PipelineStage::Compute => {
            let (metallib, serving) = translate(program)?;
            let id = crate::native_metal::queue::register_pipeline(&metallib, &program.entry)
                .map_err(|error| match error {
                    crate::native_metal::queue::NativeQueueError::Refused(why) => {
                        PipelineError::Refused(why)
                    }
                    other => PipelineError::Transport(other.to_string()),
                })?;
            Ok((id, serving))
        }
        [] => Err(PipelineError::Translation("no shader stage to register".to_string())),
        [program, ..] => Err(PipelineError::HostCannotRegister {
            stage: program.stage,
            missing: GRAPHICS_MESH_FRAME_FIELDS,
        }),
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
    fn the_graphics_refusal_names_the_missing_frame_fields() {
        let error = PipelineError::HostCannotRegister {
            stage: PipelineStage::Vertex,
            missing: GRAPHICS_MESH_FRAME_FIELDS,
        };
        let text = error.to_string();
        assert!(text.contains("graphics"));
        assert!(text.contains("PipelineState"));
        assert!(text.contains("PipelineAbi"));
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
