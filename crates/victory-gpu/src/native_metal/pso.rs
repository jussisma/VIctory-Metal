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

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use victory_metal::compiler_pool::{
    CompileJob, CompiledShader, CompilerPool, PoolError, PoolStats,
};
#[cfg(target_os = "macos")]
use victory_metal::macos::bindings::RootSignatureLayout;
use victory_metal::wire_v3::{
    BlendAttachment, BlendState, DepthStencilState, RasterizerState, StencilFaceState,
};

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
    /// Object/amplification shader (converter stage 6).
    Object,
    /// Compute shader (converter stage 8).
    Compute,
}

impl PipelineStage {
    /// The converter's stage code for this stage.
    ///
    /// These are the values `IRObjectGetMetalIRShaderStage` reports: vertex 1,
    /// fragment 2, mesh 5, object/amplification 6, kernel 8. The object code is
    /// the ADR-024a ordering; it is reported back by the converter and checked
    /// against `CompileJob::stage`.
    pub const fn converter_stage(self) -> u32 {
        match self {
            Self::Vertex => 1,
            Self::Pixel => 2,
            Self::Mesh => 5,
            Self::Object => 6,
            Self::Compute => 8,
        }
    }

    /// The version 2 wire stage code for this stage.
    pub const fn wire_code(self) -> u8 {
        match self {
            Self::Vertex => crate::native_metal::client::PIPELINE_STAGE_VERTEX,
            Self::Pixel => crate::native_metal::client::PIPELINE_STAGE_PIXEL,
            Self::Mesh => crate::native_metal::client::PIPELINE_STAGE_MESH,
            Self::Object => crate::native_metal::client::PIPELINE_STAGE_OBJECT,
            Self::Compute => crate::native_metal::client::PIPELINE_STAGE_COMPUTE,
        }
    }

    /// The stage name the title pipeline uses (`pipeline::Shader::stage`).
    ///
    /// The console names the object stage an amplification shader, and that is
    /// the name `pipeline.rs` maps console subobject type 23 to.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "vertex" => Some(Self::Vertex),
            "pixel" => Some(Self::Pixel),
            "mesh" => Some(Self::Mesh),
            "amplification" => Some(Self::Object),
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
                    PipelineStage::Mesh | PipelineStage::Object => "mesh",
                    PipelineStage::Compute => "compute",
                }
            ),
            Self::Reflection(why) => write!(f, "converter reflection: {why}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Most bytes of one refusal reason retained in telemetry and logged.
///
/// The reason is flattened to one line and truncated on a character boundary,
/// so a translator or host diagnostic can never flood a boot log.
pub const MAX_REFUSAL_REASON_BYTES: usize = 200;

/// The pipeline family one creation attempt is counted under.
///
/// `Library` is the host pipeline library's own load/create of a PSO (the
/// D3D12 backing path), kept beside the native registrations so a boot states
/// whether anything in the pipeline path ran at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttemptKind {
    /// Vertex+pixel (and any other non-mesh graphics set).
    Graphics,
    /// One compute stage.
    Compute,
    /// Mesh+pixel, optionally with an object/amplification stage.
    MeshObject,
    /// A host pipeline library load or create.
    Library,
}

impl AttemptKind {
    /// The stable name the census buckets and the snapshot line use.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Graphics => "graphics",
            Self::Compute => "compute",
            Self::MeshObject => "mesh",
            Self::Library => "library",
        }
    }
}

/// The outcome of one pipeline attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The host made a new pipeline.
    Created,
    /// An identical registration (same stages, metallib and entry) was already
    /// hosted; the same id is reused and the host is not asked again.
    Reused,
    /// The attempt was refused with an explicit reason; the list refuses the
    /// bind instead of naming a pointer the host cannot resolve.
    Refused,
    /// The converter this process needs is not available, so no translation
    /// could be attempted. Kept apart from [`Self::Refused`]: the failure is
    /// environmental, not a shader the host rejected.
    TranslatorUnavailable,
}

/// The static part of a refusal's bucket name (the kind is the other part).
///
/// A bucket is `(kind, class)`, both static strings: two distinct failure
/// classes never share a bucket, and no title text, path or identifier ever
/// becomes a key. The dynamic text is kept only as the first bounded reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RefusalClass {
    /// The container could not be parsed.
    Container,
    /// The stage has no DXIL part (a console-only container).
    NoDxil,
    /// No converter worker and nothing in the disk cache.
    ConverterUnavailable,
    /// The pool failed to translate.
    Translation,
    /// No share directory for the metallib the host reads.
    NoShareDir,
    /// Writing (or renaming) the metallib failed.
    Write,
    /// The host rejected the registration.
    HostRefused,
    /// The transport to the host failed.
    Transport,
    /// The stage set cannot form one native pipeline.
    HostCannotRegister,
    /// The converter reflection is missing or malformed.
    Reflection,
    /// The guest render state is not representable as a pipeline state.
    RenderState,
    /// The title pipeline carries a stage the native host cannot register.
    UnsupportedStage,
    /// The host pipeline library refused to create the PSO.
    LibraryCreate,
}

impl RefusalClass {
    /// The stable name the census buckets use.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Container => "container",
            Self::NoDxil => "no-dxil",
            Self::ConverterUnavailable => "converter-unavailable",
            Self::Translation => "translation",
            Self::NoShareDir => "no-share-dir",
            Self::Write => "write",
            Self::HostRefused => "host-refused",
            Self::Transport => "transport",
            Self::HostCannotRegister => "host-cannot-register",
            Self::Reflection => "reflection",
            Self::RenderState => "render-state",
            Self::UnsupportedStage => "unsupported-stage",
            Self::LibraryCreate => "library-create",
        }
    }

    /// The class a [`PipelineError`] belongs to.
    pub fn of(error: &PipelineError) -> Self {
        match error {
            PipelineError::NoDxil(_) => Self::NoDxil,
            PipelineError::Container(_) => Self::Container,
            PipelineError::ConverterUnavailable(_) => Self::ConverterUnavailable,
            PipelineError::Translation(_) => Self::Translation,
            PipelineError::NoShareDir => Self::NoShareDir,
            PipelineError::Write(_) => Self::Write,
            PipelineError::Refused(_) => Self::HostRefused,
            PipelineError::Transport(_) => Self::Transport,
            PipelineError::HostCannotRegister { .. } => Self::HostCannotRegister,
            PipelineError::Reflection(_) => Self::Reflection,
        }
    }
}

/// Attempts, creations and refusals of one pipeline kind (session totals).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindCounts {
    /// Every attempt counted under the kind.
    pub attempts: u64,
    /// Attempts the host made a new pipeline for.
    pub created: u64,
    /// Identical registrations answered with the id already hosted.
    pub reused: u64,
    /// Attempts refused with an explicit reason.
    pub refused: u64,
    /// Attempts that could not reach a translator.
    pub translator_unavailable: u64,
}

/// The first refusal of the session, bounded and flattened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstRefusal {
    /// The kind of the first refused attempt.
    pub kind: AttemptKind,
    /// Its static failure class.
    pub class: RefusalClass,
    /// Its bounded, single-line reason (at most
    /// [`MAX_REFUSAL_REASON_BYTES`] bytes, no paths).
    pub reason: String,
}

impl FirstRefusal {
    /// `kind/class`, the stable name of the first refusal.
    pub fn name(&self) -> String {
        format!("{}/{}", self.kind.name(), self.class.name())
    }
}

/// One point-in-time view of the pipeline-attempt census.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineCensusSnapshot {
    /// Every pipeline attempt counted since the process started.
    pub attempts: u64,
    /// Attempts that made a new host pipeline.
    pub created: u64,
    /// Attempts answered by an already-hosted identical pipeline.
    pub reused: u64,
    /// Attempts refused with an explicit reason.
    pub refused: u64,
    /// Attempts that could not reach a translator.
    pub translator_unavailable: u64,
    /// The per-kind counts, in `AttemptKind` order.
    pub by_kind: BTreeMap<AttemptKind, KindCounts>,
    /// Refusals by `(kind name, class name)`, in name order.
    pub refusals: BTreeMap<(&'static str, &'static str), u64>,
    /// The first refusal of the session, if any.
    pub first_refusal: Option<FirstRefusal>,
}

impl PipelineCensusSnapshot {
    /// The census fields of the `native frame telemetry` snapshot line.
    ///
    /// The first three fields are the ones a boot census reads:
    /// `pipeline_attempts`, `pipeline_created` and `pipeline_refused`. The
    /// first refusal's name (and its bounded reason) follows, so a boot that
    /// refused every pipeline names the exact failure instead of showing an
    /// empty `registered=0 psos` line.
    pub fn fields(&self) -> String {
        let mut text = format!(
            "pipeline_attempts={} pipeline_created={} pipeline_refused={} \
             pipeline_reused={} pipeline_translator_unavailable={}",
            self.attempts,
            self.created,
            self.refused,
            self.reused,
            self.translator_unavailable
        );
        text.push_str(" pipeline_by_kind=[");
        for (index, (kind, counts)) in self.by_kind.iter().enumerate() {
            if index > 0 {
                text.push_str("; ");
            }
            // attempts/created/reused/refused/unavailable, always in that order.
            text.push_str(&format!(
                "{}:{}/{}/{}/{}/{}",
                kind.name(),
                counts.attempts,
                counts.created,
                counts.reused,
                counts.refused,
                counts.translator_unavailable
            ));
        }
        text.push(']');
        if let Some(first) = &self.first_refusal {
            text.push_str(&format!(
                " pipeline_first_refusal={} pipeline_first_refusal_reason=\"{}\"",
                first.name(),
                first.reason
            ));
        }
        text
    }
}

/// Whether a whitespace-separated token names a file system path.
///
/// Host and translator diagnostics sometimes carry a path (`failed to spawn
/// shader worker /Users/...: ...`); the census must not retain it, so any
/// token with a path separator is replaced wholesale.
fn looks_like_path(word: &str) -> bool {
    word.contains('/') || word.contains('\\')
}

/// Flattens `text` to one bounded line: controls become spaces, whitespace
/// runs collapse, path tokens become `<path>`, and the result is truncated on
/// a character boundary at [`MAX_REFUSAL_REASON_BYTES`].
///
/// Empty text becomes a stable placeholder, so a snapshot field never ends in
/// `reason=""`.
pub fn bounded_refusal_reason(text: &str) -> String {
    let flattened: String = text
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else if character == '"' {
                // The snapshot prints the reason inside double quotes; a quote
                // in the text would end the field early.
                '\''
            } else {
                character
            }
        })
        .collect();
    let mut bounded = String::with_capacity(flattened.len().min(MAX_REFUSAL_REASON_BYTES));
    'words: for word in flattened.split_whitespace() {
        let word = if looks_like_path(word) { "<path>" } else { word };
        if !bounded.is_empty() {
            if bounded.len() + 1 > MAX_REFUSAL_REASON_BYTES {
                break;
            }
            bounded.push(' ');
        }
        for character in word.chars() {
            if bounded.len() + character.len_utf8() > MAX_REFUSAL_REASON_BYTES {
                break 'words;
            }
            bounded.push(character);
        }
    }
    let bounded = bounded.trim_end();
    if bounded.is_empty() {
        "unknown refusal".to_string()
    } else {
        bounded.to_string()
    }
}

/// The process-wide pipeline-attempt census.
#[derive(Debug, Default)]
struct Census {
    kinds: BTreeMap<AttemptKind, KindCounts>,
    refusals: BTreeMap<(&'static str, &'static str), u64>,
    first: Option<FirstRefusal>,
}

static CENSUS: Mutex<Census> = Mutex::new(Census {
    kinds: BTreeMap::new(),
    refusals: BTreeMap::new(),
    first: None,
});

impl Census {
    /// Counts one attempt and, for a refusal, its static bucket and (once) its
    /// bounded reason.
    fn record(
        &mut self,
        kind: AttemptKind,
        outcome: AttemptOutcome,
        class: Option<RefusalClass>,
        reason: Option<&str>,
    ) {
        let counts = self.kinds.entry(kind).or_default();
        counts.attempts = counts.attempts.saturating_add(1);
        match outcome {
            AttemptOutcome::Created => counts.created = counts.created.saturating_add(1),
            AttemptOutcome::Reused => counts.reused = counts.reused.saturating_add(1),
            AttemptOutcome::Refused => counts.refused = counts.refused.saturating_add(1),
            AttemptOutcome::TranslatorUnavailable => {
                counts.translator_unavailable = counts.translator_unavailable.saturating_add(1)
            }
        }
        if let Some(class) = class {
            *self
                .refusals
                .entry((kind.name(), class.name()))
                .or_insert(0) += 1;
            if self.first.is_none() {
                self.first = Some(FirstRefusal {
                    kind,
                    class,
                    reason: bounded_refusal_reason(reason.unwrap_or_default()),
                });
            }
        }
    }

    fn snapshot(&self) -> PipelineCensusSnapshot {
        let mut snapshot = PipelineCensusSnapshot {
            by_kind: self.kinds.clone(),
            refusals: self.refusals.clone(),
            first_refusal: self.first.clone(),
            ..PipelineCensusSnapshot::default()
        };
        for counts in snapshot.by_kind.values() {
            snapshot.attempts = snapshot.attempts.saturating_add(counts.attempts);
            snapshot.created = snapshot.created.saturating_add(counts.created);
            snapshot.reused = snapshot.reused.saturating_add(counts.reused);
            snapshot.refused = snapshot.refused.saturating_add(counts.refused);
            snapshot.translator_unavailable = snapshot
                .translator_unavailable
                .saturating_add(counts.translator_unavailable);
        }
        snapshot
    }
}

/// One native pipeline the host made a new object for.
pub fn note_created(kind: AttemptKind) {
    CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .record(kind, AttemptOutcome::Created, None, None);
}

/// One native pipeline answered by an already-hosted identical registration.
pub fn note_reused(kind: AttemptKind) {
    CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .record(kind, AttemptOutcome::Reused, None, None);
}

/// One pipeline attempt refused with an explicit, static failure class.
pub fn note_refused(kind: AttemptKind, class: RefusalClass, reason: &str) {
    CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .record(kind, AttemptOutcome::Refused, Some(class), Some(reason));
}

/// One pipeline attempt that could not reach a translator.
pub fn note_translator_unavailable(kind: AttemptKind, class: RefusalClass, reason: &str) {
    CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .record(
            kind,
            AttemptOutcome::TranslatorUnavailable,
            Some(class),
            Some(reason),
        );
}

/// The process-wide pipeline-attempt census (session totals; never reset by a
/// frame snapshot, so a pipeline made in an early frame stays counted).
pub fn census() -> PipelineCensusSnapshot {
    CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .snapshot()
}

/// Clears the census (tests only: the counts are session totals in a run).
#[cfg(test)]
pub(crate) fn reset_census() {
    *CENSUS.lock().unwrap_or_else(|error| error.into_inner()) = Census::default();
}

/// The kind a title pipeline's stage names are counted under.
pub fn attempt_kind_of_names(names: &[&str]) -> AttemptKind {
    if names
        .iter()
        .any(|name| matches!(*name, "mesh" | "amplification"))
    {
        AttemptKind::MeshObject
    } else if !names.is_empty() && names.iter().all(|name| *name == "compute") {
        AttemptKind::Compute
    } else {
        AttemptKind::Graphics
    }
}

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
/// Wire stage code for an object/amplification shader.
///
/// Must equal the host's `PIPELINE_STAGE_OBJECT` (`ipc_host.rs`); the version 2
/// mesh frame carries the object stage between the mesh and pixel stages.
pub const WIRE_STAGE_OBJECT: u8 = 3;
/// Wire stage code for a compute shader.
pub const WIRE_STAGE_COMPUTE: u8 = crate::native_metal::client::PIPELINE_STAGE_COMPUTE;

const _: () = assert!(WIRE_STAGE_OBJECT == crate::native_metal::client::PIPELINE_STAGE_OBJECT);

/// The mesh threadgroup encoded when a caller supplies no explicit `VPS1`
/// state. This is the historical default (one 32-thread mesh threadgroup).
pub const DEFAULT_MESH_THREADGROUP: [u16; 3] = [32, 1, 1];
/// The object threadgroup encoded when an object stage is present and the
/// caller supplies no explicit `VPS1` state.
pub const DEFAULT_OBJECT_THREADGROUP: [u16; 3] = [1, 1, 1];
/// The payload budget encoded when an object stage is present and the caller
/// supplies no `VPS1` state.
///
/// A caller with converter reflection (the object stage's
/// `state.max_payload_size_in_bytes`) should pass an explicit state; this is a
/// conservative budget for the pool path, which does not decode the object
/// reflection into threadgroup limits. It is deliberately non-zero: an object
/// function that writes any payload needs a non-empty payload allocation.
pub const DEFAULT_PAYLOAD_MEMORY_LENGTH: u16 = 1024;

/// Encodes a graphics `PipelineState` (`VPS1`) for the stage set.
///
/// Mirrors `victory_metal::macos::registry::PipelineState::encode(Graphics)`
/// with no render state: magic, stage code `0`, colour formats, no depth, a
/// fragment entry, no object entry and zero mesh budgets.
pub fn encode_graphics_state(color_formats: &[u8], fragment_entry: &str) -> Vec<u8> {
    encode_graphics_state_with_render(
        color_formats,
        fragment_entry,
        None,
        None,
        None,
        None,
    )
}

/// Encodes a graphics `PipelineState` carrying the guest render state.
///
/// The optional values are exactly the state Metal bakes at pipeline creation
/// (blend and alpha-to-coverage) or hashes into the content key (depth/stencil
/// and rasterizer). Passing them all `None` is the historical `VPS1` encoding,
/// so the wire stays byte-compatible; any value selects `VPS2`.
#[allow(clippy::too_many_arguments)]
pub fn encode_graphics_state_with_render(
    color_formats: &[u8],
    fragment_entry: &str,
    depth_format: Option<u8>,
    blend: Option<&BlendState>,
    depth_stencil: Option<&DepthStencilState>,
    rasterizer: Option<&RasterizerState>,
) -> Vec<u8> {
    encode_state(
        0,
        color_formats,
        fragment_entry,
        None,
        depth_format,
        [0; 3],
        [0; 3],
        0,
        0,
        RenderStateRefs {
            blend,
            depth_stencil,
            rasterizer,
        },
    )
}

/// Encodes a mesh `PipelineState` (`VPS1`) without an object stage.
///
/// Mirrors `PipelineState::encode(Mesh)` for a mesh+pixel pair: the fragment
/// entry is required and the mesh threadgroup is the historical default; no
/// object entry and no payload budget are recorded.
pub fn encode_mesh_state(color_formats: &[u8], fragment_entry: &str) -> Vec<u8> {
    encode_mesh_state_for(
        color_formats,
        fragment_entry,
        None,
        DEFAULT_MESH_THREADGROUP,
        [1, 1, 1],
        0,
    )
}

/// Encodes a mesh `PipelineState` (`VPS1`) for mesh (+ optional object) +
/// pixel.
///
/// `object_entry` must be the entry point of the object/amplification stage the
/// version 2 frame carries; the host cross-checks the two and refuses a frame
/// whose state names an object entry the stage set does not carry (or the
/// reverse). `payload_memory_length` is the object stage's payload budget.
#[allow(clippy::too_many_arguments)]
pub fn encode_mesh_state_for(
    color_formats: &[u8],
    fragment_entry: &str,
    object_entry: Option<&str>,
    mesh_threadgroup: [u16; 3],
    object_threadgroup: [u16; 3],
    payload_memory_length: u16,
) -> Vec<u8> {
    encode_mesh_state_for_with_render(
        color_formats,
        fragment_entry,
        object_entry,
        mesh_threadgroup,
        object_threadgroup,
        payload_memory_length,
        None,
        None,
        None,
        None,
    )
}

/// [`encode_mesh_state_for`] plus the guest render state (`VPS2` when any is
/// present).
#[allow(clippy::too_many_arguments)]
pub fn encode_mesh_state_for_with_render(
    color_formats: &[u8],
    fragment_entry: &str,
    object_entry: Option<&str>,
    mesh_threadgroup: [u16; 3],
    object_threadgroup: [u16; 3],
    payload_memory_length: u16,
    depth_format: Option<u8>,
    blend: Option<&BlendState>,
    depth_stencil: Option<&DepthStencilState>,
    rasterizer: Option<&RasterizerState>,
) -> Vec<u8> {
    encode_state(
        1,
        color_formats,
        fragment_entry,
        object_entry,
        depth_format,
        mesh_threadgroup,
        object_threadgroup,
        payload_memory_length,
        0,
        RenderStateRefs {
            blend,
            depth_stencil,
            rasterizer,
        },
    )
}

/// Borrowed render state one `VPS2` encoding carries.
#[derive(Clone, Copy, Default)]
struct RenderStateRefs<'a> {
    blend: Option<&'a BlendState>,
    depth_stencil: Option<&'a DepthStencilState>,
    rasterizer: Option<&'a RasterizerState>,
}

impl RenderStateRefs<'_> {
    const fn is_empty(self) -> bool {
        self.blend.is_none() && self.depth_stencil.is_none() && self.rasterizer.is_none()
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_state(
    stage: u8,
    color_formats: &[u8],
    fragment_entry: &str,
    object_entry: Option<&str>,
    depth_format: Option<u8>,
    mesh_threadgroup: [u16; 3],
    object_threadgroup: [u16; 3],
    payload_memory_length: u16,
    max_total_threadgroups_per_mesh_grid: u32,
    render: RenderStateRefs<'_>,
) -> Vec<u8> {
    let has_render_state = !render.is_empty();
    let mut out = Vec::with_capacity(if has_render_state {
        192 + fragment_entry.len()
    } else {
        64 + fragment_entry.len()
    });
    out.extend_from_slice(if has_render_state { b"VPS2" } else { b"VPS1" });
    out.push(stage);
    out.push(u8::try_from(color_formats.len()).expect("colour format count exceeds u8"));
    out.extend_from_slice(color_formats);
    match depth_format {
        Some(format) => {
            out.push(1);
            out.push(format);
        }
        None => out.push(0),
    }
    let object = object_entry.unwrap_or("");
    out.push(u8::from(!fragment_entry.is_empty()) | (u8::from(!object.is_empty()) << 1));
    push_u16_string(&mut out, fragment_entry);
    push_u16_string(&mut out, object);
    for value in mesh_threadgroup.iter().chain(object_threadgroup.iter()) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&payload_memory_length.to_le_bytes());
    out.extend_from_slice(&max_total_threadgroups_per_mesh_grid.to_le_bytes());
    if has_render_state {
        push_render_state(&mut out, render);
    }
    out
}

/// Appends the `VPS2` render-state section (fixed order: blend, depth/stencil,
/// rasterizer).
fn push_render_state(out: &mut Vec<u8>, render: RenderStateRefs<'_>) {
    let mut flags = 0u8;
    if render.blend.is_some() {
        flags |= 1;
    }
    if render.depth_stencil.is_some() {
        flags |= 2;
    }
    if render.rasterizer.is_some() {
        flags |= 4;
    }
    out.push(flags);
    if let Some(blend) = render.blend {
        out.push(u8::from(blend.independent_blend));
        out.push(u8::from(blend.alpha_to_coverage));
        out.push(u8::try_from(blend.attachments.len()).expect("blend attachment count exceeds u8"));
        for attachment in &blend.attachments {
            push_blend_attachment(out, attachment);
        }
    }
    if let Some(depth) = render.depth_stencil {
        out.push(u8::from(depth.depth_enable));
        out.push(u8::from(depth.depth_write));
        push_u16(out, depth.depth_func.code());
        out.push(u8::from(depth.stencil_enable));
        push_stencil_face(out, &depth.front);
        push_stencil_face(out, &depth.back);
    }
    if let Some(raster) = render.rasterizer {
        push_u16(out, raster.fill.code());
        push_u16(out, raster.cull.code());
        push_u16(out, raster.front_face.code());
        out.extend_from_slice(&raster.depth_bias.to_le_bytes());
        out.extend_from_slice(&raster.slope_scaled_depth_bias.to_bits().to_le_bytes());
        out.extend_from_slice(&raster.depth_bias_clamp.to_bits().to_le_bytes());
        out.push(u8::from(raster.depth_clip));
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_blend_attachment(out: &mut Vec<u8>, attachment: &BlendAttachment) {
    out.push(u8::from(attachment.enable));
    push_u16(out, attachment.src_color.code());
    push_u16(out, attachment.dst_color.code());
    push_u16(out, attachment.color_op.code());
    push_u16(out, attachment.src_alpha.code());
    push_u16(out, attachment.dst_alpha.code());
    push_u16(out, attachment.alpha_op.code());
    out.push(attachment.write_mask);
}

fn push_stencil_face(out: &mut Vec<u8>, face: &StencilFaceState) {
    push_u16(out, face.fail.code());
    push_u16(out, face.depth_fail.code());
    push_u16(out, face.pass.code());
    push_u16(out, face.func.code());
    out.push(face.read_mask);
    out.push(face.write_mask);
}

/// Canonical DXGI -> (PSO code, depth/stencil) rows for the guest (Windows)
/// build, where `victory-metal`'s macOS-only registry module is not compiled.
///
/// The canonical table in
/// `victory_metal::macos::registry::FORMAT_TABLE` is the source of truth; the
/// macOS test [`pso_format_codes_match_the_canonical_table`] asserts every row
/// here agrees with it. BC formats and stencil-only read views are not
/// pipeline attachment formats and are omitted.
#[allow(dead_code)]
const DXGI_FORMAT_CODES: &[(u32, u8, bool)] = &[
    (1, 23, false),
    (2, 20, false),
    (3, 21, false),
    (4, 22, false),
    (9, 28, false),
    (10, 17, false),
    (11, 24, false),
    (12, 25, false),
    (13, 26, false),
    (14, 27, false),
    (15, 32, false),
    (16, 29, false),
    (17, 30, false),
    (18, 31, false),
    (19, 35, true),
    (20, 16, true),
    (21, 33, true),
    (23, 37, false),
    (24, 11, false),
    (25, 36, false),
    (26, 10, false),
    (27, 41, false),
    (28, 7, false),
    (29, 8, false),
    (30, 38, false),
    (31, 39, false),
    (32, 40, false),
    (33, 47, false),
    (34, 42, false),
    (35, 43, false),
    (36, 44, false),
    (37, 45, false),
    (38, 46, false),
    (39, 48, false),
    (40, 14, true),
    (41, 12, false),
    (42, 18, false),
    (43, 19, false),
    (44, 51, true),
    (45, 15, true),
    (46, 49, true),
    (48, 56, false),
    (49, 52, false),
    (50, 53, false),
    (51, 54, false),
    (52, 55, false),
    (53, 62, false),
    (54, 13, false),
    (55, 61, true),
    (56, 57, false),
    (57, 58, false),
    (58, 59, false),
    (59, 60, false),
    (60, 67, false),
    (61, 63, false),
    (62, 64, false),
    (63, 65, false),
    (64, 66, false),
    (65, 68, false),
    (67, 69, false),
    (85, 70, false),
    (86, 71, false),
    (87, 9, false),
    (88, 74, false),
    (90, 73, false),
    (91, 72, false),
    (92, 76, false),
    (93, 75, false),
];

/// The canonical table's answer for `dxgi`, or the cross-target mirror.
#[cfg(target_os = "macos")]
fn canonical_format_code(dxgi: u32) -> Option<(u8, bool)> {
    let format = victory_metal::macos::registry::TextureFormat::from_dxgi(dxgi).ok()?;
    Some((format.code(), format.is_depth_stencil()))
}

#[cfg(not(target_os = "macos"))]
fn canonical_format_code(dxgi: u32) -> Option<(u8, bool)> {
    DXGI_FORMAT_CODES
        .iter()
        .find(|(value, _, _)| *value == dxgi)
        .map(|(_, code, depth)| (*code, *depth))
}

#[cfg(target_os = "macos")]
fn format_is_block_compressed(dxgi: u32) -> bool {
    victory_metal::macos::registry::TextureFormat::from_dxgi(dxgi)
        .is_ok_and(|format| format.is_block_compressed())
}

#[cfg(not(target_os = "macos"))]
const fn format_is_block_compressed(_dxgi: u32) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn format_is_colour_attachment(dxgi: u32) -> bool {
    use victory_metal::macos::registry::FormatKind;
    victory_metal::macos::registry::TextureFormat::from_dxgi(dxgi)
        .is_ok_and(|format| matches!(format.kind(), FormatKind::Colour))
}

#[cfg(not(target_os = "macos"))]
const fn format_is_colour_attachment(_dxgi: u32) -> bool {
    true
}

/// Host `TextureFormat` code for a PC `DXGI_FORMAT` render-target format.
///
/// Every colour row of the canonical table is accepted (including typeless
/// rows, which the host creates with their concrete base); block-compressed,
/// depth and stencil-only formats are never colour attachments. An unmapped
/// format leaves the caller's historical `Rgba8Unorm` default, exactly as the
/// pre-`VPS2` default state did.
pub fn color_format_code(dxgi: u32) -> Option<u8> {
    canonical_format_code(dxgi).and_then(|(code, depth)| {
        (!depth && !format_is_block_compressed(dxgi) && format_is_colour_attachment(dxgi))
            .then_some(code)
    })
}

/// Host `TextureFormat` code for a PC `DXGI_FORMAT` depth format.
pub fn depth_format_code(dxgi: u32) -> Option<u8> {
    canonical_format_code(dxgi).and_then(|(code, depth)| depth.then_some(code))
}

fn push_u16_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(
        &u16::try_from(text.len())
            .expect("pipeline state string exceeds u16")
            .to_le_bytes(),
    );
    out.extend_from_slice(text.as_bytes());
}

/// The default `VPS1` mesh state derived from the registered stage set.
///
/// The fragment entry always comes from the pixel stage. When the set carries
/// an object/amplification stage (`[Mesh, Object, Pixel]`, the only order
/// [`pipeline_kind`] accepts for it), its entry becomes the encoded object
/// entry and the default object threadgroup and payload budget are recorded;
/// a mesh+pixel set keeps the historical no-object state exactly.
fn default_mesh_state(
    stages: &[(PipelineStage, &[u8], &str)],
    color_formats: &[u8],
) -> Vec<u8> {
    let fragment_entry = stages
        .iter()
        .find(|(stage, _, _)| *stage == PipelineStage::Pixel)
        .map(|(_, _, entry)| *entry)
        .unwrap_or_default();
    let object_entry = stages
        .iter()
        .find(|(stage, _, _)| *stage == PipelineStage::Object)
        .map(|(_, _, entry)| *entry);
    encode_mesh_state_for(
        color_formats,
        fragment_entry,
        object_entry,
        DEFAULT_MESH_THREADGROUP,
        if object_entry.is_some() {
            DEFAULT_OBJECT_THREADGROUP
        } else {
            [1, 1, 1]
        },
        if object_entry.is_some() {
            DEFAULT_PAYLOAD_MEMORY_LENGTH
        } else {
            0
        },
    )
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
            PipelineStage::Object => "object",
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
    let result = register_with_details_inner(programs, details);
    record_outcome(
        attempt_kind_of_programs(programs),
        result.as_ref().ok().map(|(_, _, outcome)| *outcome),
        result.as_ref().err(),
    );
    result.map(|(id, serving, _)| (id, serving))
}

/// Records one finished registration in the census: a creation, a reuse, or
/// the explicit class and bounded reason of a refusal.
#[cfg(any(windows, test))]
fn record_outcome(
    kind: AttemptKind,
    outcome: Option<AttemptOutcome>,
    error: Option<&PipelineError>,
) {
    let mut census = CENSUS.lock().unwrap_or_else(|failure| failure.into_inner());
    match (outcome, error) {
        (Some(outcome), _) => census.record(kind, outcome, None, None),
        (None, Some(error)) => {
            let class = RefusalClass::of(error);
            let outcome = if class == RefusalClass::ConverterUnavailable {
                AttemptOutcome::TranslatorUnavailable
            } else {
                AttemptOutcome::Refused
            };
            census.record(kind, outcome, Some(class), Some(&error.to_string()));
        }
        (None, None) => {}
    }
}

/// The kind a translated stage set is counted under.
#[cfg(any(windows, test))]
fn attempt_kind_of_stages(stages: &[(PipelineStage, &[u8], &str)]) -> AttemptKind {
    if stages
        .iter()
        .any(|(stage, _, _)| matches!(stage, PipelineStage::Mesh | PipelineStage::Object))
    {
        AttemptKind::MeshObject
    } else if !stages.is_empty()
        && stages
            .iter()
            .all(|(stage, _, _)| *stage == PipelineStage::Compute)
    {
        AttemptKind::Compute
    } else {
        AttemptKind::Graphics
    }
}

/// The kind a program set is counted under.
pub fn attempt_kind_of_programs(programs: &[StageProgram]) -> AttemptKind {
    if programs
        .iter()
        .any(|program| matches!(program.stage, PipelineStage::Mesh | PipelineStage::Object))
    {
        AttemptKind::MeshObject
    } else if !programs.is_empty()
        && programs
            .iter()
            .all(|program| program.stage == PipelineStage::Compute)
    {
        AttemptKind::Compute
    } else {
        AttemptKind::Graphics
    }
}

/// The host id one content key already named, if its registration succeeded.
///
/// A repeated registration with the same stages, metallib bytes, entries and
/// state is idempotent: the host registry answers with the object it already
/// has, so the attempt is counted as a reuse. The registration still goes to
/// the host (this map is only the census's memory, never an answer the guest
/// invents): after a reconnect the new host session must be told again.
#[cfg(any(windows, test))]
static REGISTERED: Mutex<Option<HashMap<[u8; 32], u64>>> = Mutex::new(None);

/// Remembers `key` as hosted under `id`; the previous id for the same key, if
/// any (a second registration of one content key is a reuse, not a creation).
#[cfg(any(windows, test))]
fn remember(key: [u8; 32], id: u64) -> Option<u64> {
    REGISTERED
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, id)
}

/// The outcome of one successful host registration: [`AttemptOutcome::Reused`]
/// when this content key was registered before, [`AttemptOutcome::Created`]
/// otherwise.
#[cfg(any(windows, test))]
fn remember_outcome(key: [u8; 32], id: u64) -> AttemptOutcome {
    match remember(key, id) {
        Some(_) => AttemptOutcome::Reused,
        None => AttemptOutcome::Created,
    }
}

/// The content key of one registration: every input the host hashes into a
/// pipeline, so two registrations share it exactly when the host would.
///
/// `details` is `None` for the compute v1 frame, which carries only the
/// metallib and the entry (`register_pipeline`), so a state blob the v1 path
/// ignores must not make a different key.
#[cfg(any(windows, test))]
fn registration_key(
    stages: &[PipelineStage],
    libraries: &[(PipelineStage, &[u8], &str)],
    details: Option<&PipelineDetails<'_>>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"victory-native-pipeline-registration-v1");
    hasher.update(
        &u8::try_from(stages.len())
            .unwrap_or(u8::MAX)
            .to_le_bytes(),
    );
    for stage in stages {
        hasher.update(&stage.converter_stage().to_le_bytes());
    }
    for (stage, metallib, entry) in libraries {
        hasher.update(&stage.converter_stage().to_le_bytes());
        hasher.update(&(metallib.len() as u64).to_le_bytes());
        hasher.update(metallib);
        hasher.update(&(entry.len() as u64).to_le_bytes());
        hasher.update(entry.as_bytes());
    }
    if let Some(details) = details {
        for part in [details.state, details.abi, details.root_signature] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
    }
    *hasher.finalize().as_bytes()
}

#[cfg(any(windows, test))]
fn register_with_details_inner(
    programs: &[StageProgram],
    details: &PipelineDetails<'_>,
) -> Result<(u64, PoolServing, AttemptOutcome), PipelineError> {
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
    let (id, outcome) = register_translated_inner(&stages, &translated_details)?;
    Ok((id, PoolServing { warm }, outcome))
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
    let result = register_translated_inner(stages, details);
    record_outcome(
        attempt_kind_of_stages(stages),
        result.as_ref().ok().map(|(_, outcome)| *outcome),
        result.as_ref().err(),
    );
    result.map(|(id, _)| id)
}

/// [`register_translated`] without the census: the outcome (created or reused)
/// is returned to the one caller that records it.
#[cfg(any(windows, test))]
fn register_translated_inner(
    stages: &[(PipelineStage, &[u8], &str)],
    details: &PipelineDetails<'_>,
) -> Result<(u64, AttemptOutcome), PipelineError> {
    let stage_names: Vec<PipelineStage> = stages.iter().map(|(stage, _, _)| *stage).collect();
    let kind = pipeline_kind(&stage_names)?;
    if !details.abi.is_empty() && details.root_signature.is_empty() {
        return Err(PipelineError::Reflection(
            "a non-empty VAB1 ABI needs the root-layout witness bytes the version 2 frame \
             carries"
                .to_string(),
        ));
    }
    let v1_compute = kind == PipelineKind::Compute && details.abi.is_empty();
    let key = registration_key(
        &stage_names,
        stages,
        if v1_compute { None } else { Some(details) },
    );
    if v1_compute {
        // No reflection was available: keep the original v1 frame. The host
        // binds nothing for this pipeline, which is exactly the old behaviour.
        let (_, metallib, entry) = stages[0];
        let id = crate::native_metal::queue::register_pipeline(metallib, entry)
            .map_err(map_queue_error)?;
        return Ok((id, remember_outcome(key, id)));
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
            default_mesh_state(stages, &[DEFAULT_COLOR_FORMAT])
        } else {
            encode_graphics_state(&[DEFAULT_COLOR_FORMAT], pixel_entry)
        };
        &state_storage
    };
    let id = crate::native_metal::queue::register_pipeline_v2(
        kind_code,
        &wire,
        state,
        details.abi,
        details.root_signature,
    )
    .map_err(map_queue_error)?;
    Ok((id, remember_outcome(key, id)))
}

/// Maps a queue failure onto the PSO path's explicit error vocabulary.
#[cfg(any(windows, test))]
fn map_queue_error(error: crate::native_metal::queue::NativeQueueError) -> PipelineError {
    match error {
        crate::native_metal::queue::NativeQueueError::Refused(why) => PipelineError::Refused(why),
        other => PipelineError::Transport(other.to_string()),
    }
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
        // ADR-024a: the version 2 frame carries the object/amplification stage
        // between the mesh and pixel stages, the order the host binds.
        [PipelineStage::Mesh, PipelineStage::Object, PipelineStage::Pixel] => {
            Ok(PipelineKind::Mesh)
        }
        [PipelineStage::Object, PipelineStage::Mesh, PipelineStage::Pixel] => {
            Err(PipelineError::HostCannotRegister {
                stage: PipelineStage::Object,
                missing: "the object/amplification stage must follow the mesh stage and precede \
                          the pixel stage (mesh, object, pixel)",
            })
        }
        [] => Err(PipelineError::Translation(
            "no shader stage to register".to_string(),
        )),
        [stage] => Err(PipelineError::HostCannotRegister {
            stage: *stage,
            missing: match stage {
                PipelineStage::Vertex => "a graphics pipeline also needs a pixel stage",
                PipelineStage::Pixel => "a render pipeline also needs a vertex or mesh stage",
                PipelineStage::Mesh => "a mesh pipeline also needs a pixel stage",
                PipelineStage::Object => {
                    "an object/amplification pipeline also needs mesh and pixel stages"
                }
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
                            | PipelineStage::Object
                            | PipelineStage::Compute
                    )
                })
                .unwrap_or(&stages[stages.len() - 1]);
            Err(PipelineError::HostCannotRegister {
                stage,
                missing: "the native path registers vertex+pixel, mesh+pixel, \
                          mesh+object+pixel or one compute stage only",
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_metal::wire_v3::{BlendFactor, BlendOperation};

    #[test]
    fn stage_codes_match_the_converter() {
        assert_eq!(PipelineStage::Vertex.converter_stage(), 1);
        assert_eq!(PipelineStage::Pixel.converter_stage(), 2);
        assert_eq!(PipelineStage::Mesh.converter_stage(), 5);
        assert_eq!(PipelineStage::Object.converter_stage(), 6);
        assert_eq!(PipelineStage::Compute.converter_stage(), 8);
        assert_eq!(
            PipelineStage::from_name("compute"),
            Some(PipelineStage::Compute)
        );
        assert_eq!(
            PipelineStage::from_name("amplification"),
            Some(PipelineStage::Object)
        );
        assert_eq!(PipelineStage::Object.wire_code(), WIRE_STAGE_OBJECT);
        assert_eq!(WIRE_STAGE_OBJECT, 3);
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

    /// R32_UINT/R32_SINT colour attachments encode the registry codes the host
    /// decodes (`TextureFormat::R32Uint`/`R32Sint`), not the Rgba8Unorm
    /// default an unmapped format would silently take.
    #[test]
    fn r32_integer_targets_map_to_the_r32_host_codes() {
        assert_eq!(color_format_code(42), Some(18));
        assert_eq!(color_format_code(43), Some(19));
        assert_eq!(color_format_code(11), Some(24), "R16G16B16A16_UNORM");
        assert_eq!(color_format_code(5), None, "R32G32B32_TYPELESS is refused");
        #[cfg(target_os = "macos")]
        {
            use victory_metal::macos::registry::TextureFormat;
            assert_eq!(TextureFormat::from_code(18), Some(TextureFormat::R32Uint));
            assert_eq!(TextureFormat::from_code(19), Some(TextureFormat::R32Sint));
            assert_eq!(TextureFormat::R32Uint.mtl_pixel_format(), 53);
            assert_eq!(TextureFormat::R32Sint.mtl_pixel_format(), 54);
        }
    }

    /// The full canonical DXGI set maps to the shared PSO codes; the values the
    /// cycle directive listed are corrected by name (62 is R8_UINT, 57 is
    /// R16_UINT, 64 is R8_SINT, 88 is B8G8R8X8_UNORM).
    #[test]
    fn canonical_pso_codes_cover_the_standard_set() {
        assert_eq!(color_format_code(62), Some(64), "R8_UINT");
        assert_eq!(color_format_code(57), Some(58), "R16_UINT");
        assert_eq!(color_format_code(63), Some(65), "R8_SNORM");
        assert_eq!(color_format_code(64), Some(66), "R8_SINT");
        assert_eq!(color_format_code(65), Some(68), "A8_UNORM");
        assert_eq!(color_format_code(24), Some(11), "R10G10B10A2_UNORM");
        assert_eq!(color_format_code(26), Some(10), "R11G11B10_FLOAT");
        assert_eq!(color_format_code(87), Some(9), "B8G8R8A8_UNORM");
        assert_eq!(color_format_code(88), Some(74), "B8G8R8X8_UNORM");
        assert_eq!(color_format_code(6), None, "three-channel refused");
        assert_eq!(color_format_code(71), None, "BC is not an attachment");
        assert_eq!(depth_format_code(40), Some(14), "D32_FLOAT");
        assert_eq!(depth_format_code(45), Some(15), "D24_UNORM_S8_UINT");
        assert_eq!(depth_format_code(20), Some(16), "D32_FLOAT_S8X24_UINT");
        assert_eq!(depth_format_code(55), Some(61), "D16_UNORM");
        assert_eq!(depth_format_code(28), None, "RGBA8 is not depth");
    }

    /// The guest (Windows) mirror must agree with the canonical table for every
    /// row it carries.
    #[cfg(target_os = "macos")]
    #[test]
    fn pso_format_codes_match_the_canonical_table() {
        use victory_metal::macos::registry::TextureFormat;
        for &(dxgi, code, depth) in DXGI_FORMAT_CODES {
            let format = TextureFormat::from_dxgi(dxgi)
                .unwrap_or_else(|error| panic!("DXGI {dxgi}: {error}"));
            assert_eq!(format.code(), code, "{} code", format.name());
            assert_eq!(
                format.is_depth_stencil(),
                depth,
                "{} depth-ness",
                format.name()
            );
        }
    }

    fn sample_blend(enable: bool) -> BlendState {
        BlendState {
            independent_blend: false,
            alpha_to_coverage: false,
            attachments: vec![BlendAttachment {
                enable,
                src_color: BlendFactor::SrcAlpha,
                dst_color: BlendFactor::InvSrcAlpha,
                color_op: BlendOperation::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::Zero,
                alpha_op: BlendOperation::Add,
                write_mask: 0x0f,
            }],
        }
    }

    /// The guest serialises the D3D12 render state in a fixed order: identical
    /// state is identical bytes, a changed blend is different bytes, and the
    /// host decodes the same values (including the depth format).
    #[test]
    fn guest_render_state_is_deterministic_vps2() {
        let enabled = sample_blend(true);
        let disabled = sample_blend(false);
        let encoded = encode_graphics_state_with_render(
            &[DEFAULT_COLOR_FORMAT],
            "main",
            Some(14),
            Some(&enabled),
            None,
            None,
        );
        assert_eq!(&encoded[..4], b"VPS2");
        assert_eq!(
            encoded,
            encode_graphics_state_with_render(
                &[DEFAULT_COLOR_FORMAT],
                "main",
                Some(14),
                Some(&enabled),
                None,
                None,
            ),
            "the same state encodes to the same bytes"
        );
        let other = encode_graphics_state_with_render(
            &[DEFAULT_COLOR_FORMAT],
            "main",
            Some(14),
            Some(&disabled),
            None,
            None,
        );
        assert_ne!(encoded, other, "a changed blend changes the encoded state");

        #[cfg(target_os = "macos")]
        {
            use victory_metal::macos::registry::{
                PipelineStage as RegistryStage, PipelineState, TextureFormat,
            };
            let decoded = PipelineState::decode_for(&encoded, RegistryStage::Graphics)
                .expect("the guest VPS2 blob decodes on the host");
            assert_eq!(decoded.blend, Some(enabled.clone()));
            assert_eq!(decoded.depth_format, Some(TextureFormat::Depth32Float));
            assert!(decoded.depth_stencil.is_none() && decoded.rasterizer.is_none());
            // The guest and the host encoders are byte-identical for the same
            // state, so the BLAKE3 key covers exactly what the host decodes.
            let host = PipelineState {
                color_formats: vec![TextureFormat::Rgba8Unorm],
                depth_format: Some(TextureFormat::Depth32Float),
                fragment_entry: Some("main".to_string()),
                blend: Some(enabled),
                ..PipelineState::default()
            }
            .encode(RegistryStage::Graphics);
            assert_eq!(host, encoded, "guest and host VPS2 encodings agree");
        }
    }

    /// The fields of a `VPS1` blob the guest encoder can produce.
    #[derive(Debug)]
    struct DecodedState {
        stage: u8,
        colors: Vec<u8>,
        fragment: String,
        object: Option<String>,
        mesh_threadgroup: [u16; 3],
        object_threadgroup: [u16; 3],
        payload_memory_length: u16,
        max_total_threadgroups_per_mesh_grid: u32,
    }

    /// Decodes the exact layout
    /// `victory_metal::macos::registry::PipelineState` writes.
    fn decode_state(bytes: &[u8]) -> DecodedState {
        let mut at = 0usize;
        macro_rules! take {
            ($count:expr) => {{
                let count = $count;
                let slice = &bytes[at..at + count];
                at += count;
                slice
            }};
        }
        macro_rules! u16_at {
            () => {
                u16::from_le_bytes(take!(2).try_into().unwrap())
            };
        }
        assert_eq!(take!(4), b"VPS1");
        let stage = take!(1)[0];
        let color_count = take!(1)[0] as usize;
        let colors = take!(color_count).to_vec();
        assert_eq!(take!(1)[0], 0, "no depth attachment");
        let flags = take!(1)[0];
        let length = usize::from(u16_at!());
        let fragment = std::str::from_utf8(take!(length)).unwrap().to_string();
        let length = usize::from(u16_at!());
        let object = std::str::from_utf8(take!(length)).unwrap().to_string();
        let mut threadgroups = || {
            let mut out = [0u16; 3];
            for value in &mut out {
                *value = u16_at!();
            }
            out
        };
        let mesh_threadgroup = threadgroups();
        let object_threadgroup = threadgroups();
        let payload_memory_length = u16_at!();
        let max_total_threadgroups_per_mesh_grid =
            u32::from_le_bytes(take!(4).try_into().unwrap());
        assert_eq!(at, bytes.len(), "no trailing bytes");
        DecodedState {
            stage,
            colors,
            fragment: if flags & 1 != 0 { fragment } else { String::new() },
            object: (flags & 2 != 0).then_some(object),
            mesh_threadgroup,
            object_threadgroup,
            payload_memory_length,
            max_total_threadgroups_per_mesh_grid,
        }
    }

    /// ADR-024a: `[Mesh, Object, Pixel]` is a valid mesh family and its
    /// default state encodes the object entry, wire stage 3 and the object
    /// threadgroup/payload budgets the host binds.
    #[test]
    fn a_mesh_object_pixel_order_encodes_the_object_stage() {
        assert_eq!(
            pipeline_kind(&[
                PipelineStage::Mesh,
                PipelineStage::Object,
                PipelineStage::Pixel
            ]),
            Ok(PipelineKind::Mesh)
        );
        let stages: [(PipelineStage, &[u8], &str); 3] = [
            (PipelineStage::Mesh, b"mesh metallib", "ms_main"),
            (PipelineStage::Object, b"object metallib", "as_main"),
            (PipelineStage::Pixel, b"pixel metallib", "ps_main"),
        ];
        let state = default_mesh_state(&stages, &[DEFAULT_COLOR_FORMAT]);
        let decoded = decode_state(&state);
        assert_eq!(decoded.stage, 1, "mesh VPS1 stage code");
        assert_eq!(decoded.colors, vec![DEFAULT_COLOR_FORMAT]);
        assert_eq!(decoded.fragment, "ps_main");
        assert_eq!(decoded.object.as_deref(), Some("as_main"));
        assert_eq!(decoded.mesh_threadgroup, DEFAULT_MESH_THREADGROUP);
        assert_eq!(decoded.object_threadgroup, DEFAULT_OBJECT_THREADGROUP);
        assert_eq!(
            decoded.payload_memory_length,
            DEFAULT_PAYLOAD_MEMORY_LENGTH
        );
        assert!(
            decoded.payload_memory_length > 0,
            "an object stage needs a payload budget"
        );
        assert_eq!(decoded.max_total_threadgroups_per_mesh_grid, 0);
        assert_eq!(PipelineStage::Object.wire_code(), 3);
    }

    /// A mesh+pixel set keeps the historical no-object state exactly.
    #[test]
    fn a_mesh_pixel_order_still_encodes_without_an_object_stage() {
        assert_eq!(
            pipeline_kind(&[PipelineStage::Mesh, PipelineStage::Pixel]),
            Ok(PipelineKind::Mesh)
        );
        let stages: [(PipelineStage, &[u8], &str); 2] = [
            (PipelineStage::Mesh, b"mesh metallib", "ms_main"),
            (PipelineStage::Pixel, b"pixel metallib", "ps_main"),
        ];
        let decoded = decode_state(&default_mesh_state(&stages, &[DEFAULT_COLOR_FORMAT]));
        assert_eq!(decoded.object, None);
        assert_eq!(decoded.payload_memory_length, 0);
        assert_eq!(decoded.object_threadgroup, [1, 1, 1]);
    }

    /// `[Object, Mesh, Pixel]` is the wrong order and is refused explicitly;
    /// the guest never reorders or drops the stage.
    #[test]
    fn a_wrong_object_stage_order_is_refused_explicitly() {
        let error = pipeline_kind(&[
            PipelineStage::Object,
            PipelineStage::Mesh,
            PipelineStage::Pixel,
        ])
        .expect_err("object before mesh is refused");
        match error {
            PipelineError::HostCannotRegister {
                stage: PipelineStage::Object,
                missing,
            } => assert!(missing.contains("mesh, object, pixel"), "{missing}"),
            other => panic!("expected an explicit order refusal, saw {other:?}"),
        }
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

    /// Every attempt is counted under its kind and outcome, each (kind, class)
    /// refuses into its own bucket, and the kinds report in a fixed order.
    #[test]
    fn pipeline_attempts_are_counted_by_kind_and_outcome() {
        reset_census();
        note_created(AttemptKind::Graphics);
        note_reused(AttemptKind::MeshObject);
        note_refused(
            AttemptKind::Compute,
            RefusalClass::Container,
            "shader container: bad part",
        );
        note_translator_unavailable(
            AttemptKind::Library,
            RefusalClass::ConverterUnavailable,
            "shader pool has no worker program configured",
        );

        let census = census();
        assert_eq!(
            (
                census.attempts,
                census.created,
                census.reused,
                census.refused,
                census.translator_unavailable
            ),
            (4, 1, 1, 1, 1)
        );
        assert_eq!(
            census.by_kind.keys().copied().collect::<Vec<_>>(),
            [
                AttemptKind::Graphics,
                AttemptKind::Compute,
                AttemptKind::MeshObject,
                AttemptKind::Library,
            ],
            "the kinds always report in the same order"
        );
        assert_eq!(census.by_kind[&AttemptKind::Graphics].created, 1);
        assert_eq!(census.by_kind[&AttemptKind::Compute].refused, 1);
        assert_eq!(census.by_kind[&AttemptKind::MeshObject].reused, 1);
        assert_eq!(
            census.by_kind[&AttemptKind::Library].translator_unavailable,
            1
        );
        // Distinct classes never share a bucket.
        assert_eq!(
            census.refusals,
            BTreeMap::from([
                (("compute", "container"), 1),
                (("library", "converter-unavailable"), 1),
            ])
        );
        let fields = census.fields();
        for expected in [
            "pipeline_attempts=4",
            "pipeline_created=1",
            "pipeline_refused=1",
            "pipeline_reused=1",
            "pipeline_translator_unavailable=1",
            "pipeline_by_kind=[graphics:1/1/0/0/0; compute:1/0/0/1/0; mesh:1/0/1/0/0; \
             library:1/0/0/0/1]",
            "pipeline_first_refusal=compute/container",
        ] {
            assert!(fields.contains(expected), "{expected} missing from {fields}");
        }
    }

    /// The first refusal is kept once, flattened to one line, and every later
    /// refusal keeps its own bucket without replacing it.
    #[test]
    fn the_first_refusal_is_kept_and_each_class_has_its_own_bucket() {
        reset_census();
        note_refused(
            AttemptKind::Graphics,
            RefusalClass::UnsupportedStage,
            "the native host registers vertex, pixel\n\tmesh and compute stages only; \
             this PSO also carries hull",
        );
        note_refused(
            AttemptKind::Graphics,
            RefusalClass::NoDxil,
            "the vertex shader has no DXIL part",
        );

        let census = census();
        let first = census.first_refusal.as_ref().expect("the first refusal is kept");
        assert_eq!(first.kind, AttemptKind::Graphics);
        assert_eq!(first.class, RefusalClass::UnsupportedStage);
        assert_eq!(
            first.reason,
            "the native host registers vertex, pixel mesh and compute stages only; \
             this PSO also carries hull"
        );
        assert!(!first.reason.contains('\n'), "{}", first.reason);
        assert!(first.reason.len() <= MAX_REFUSAL_REASON_BYTES);
        assert_eq!(census.refusals[&("graphics", "unsupported-stage")], 1);
        assert_eq!(census.refusals[&("graphics", "no-dxil")], 1);
        assert_eq!(
            census.first_refusal.as_ref().map(FirstRefusal::name),
            Some("graphics/unsupported-stage".to_string()),
            "a later refusal never replaces the first"
        );
    }

    /// A refusal reason is one bounded line with no path or title text, and a
    /// multi-byte character is never cut in half.
    #[test]
    fn a_refusal_reason_is_one_bounded_pathless_line() {
        let reason = bounded_refusal_reason(
            "failed to spawn shader worker /Users/someone/bin/victory-irconverter: not found",
        );
        assert_eq!(
            reason,
            "failed to spawn shader worker <path> not found",
            "a path token is redacted"
        );
        assert!(!reason.contains("Users"));

        let multiline = bounded_refusal_reason("line one\n\tline two \"quoted\"");
        assert_eq!(multiline, "line one line two 'quoted'");
        assert!(!multiline.contains('\n'));

        let long = bounded_refusal_reason(&"x".repeat(MAX_REFUSAL_REASON_BYTES * 3));
        assert_eq!(long.len(), MAX_REFUSAL_REASON_BYTES);

        // Two bytes per character: 200 bytes is 100 of them, on a boundary.
        let multibyte = bounded_refusal_reason(&"é".repeat(MAX_REFUSAL_REASON_BYTES));
        assert_eq!(multibyte.len(), MAX_REFUSAL_REASON_BYTES);
        assert!(multibyte.is_char_boundary(multibyte.len()));
        assert_eq!(multibyte.chars().count(), MAX_REFUSAL_REASON_BYTES / 2);

        assert_eq!(bounded_refusal_reason(" \n\t "), "unknown refusal");
    }

    /// An identical registration is counted once as a creation and once as a
    /// reuse, never as a refusal; it is still sent to the host, which answers
    /// with the object it already has.
    #[test]
    fn a_duplicated_registration_is_reused_not_refused() {
        reset_census();
        let key = registration_key(&[PipelineStage::Compute], &[], None);

        // The first registration made the pipeline on the host.
        let first = remember_outcome(key, 77);
        assert_eq!(first, AttemptOutcome::Created);
        record_outcome(AttemptKind::Compute, Some(first), None);

        // The identical registration reuses the object the host already has.
        let second = remember_outcome(key, 77);
        assert_eq!(second, AttemptOutcome::Reused);
        record_outcome(AttemptKind::Compute, Some(second), None);

        let census = census();
        assert_eq!(census.attempts, 2);
        assert_eq!(census.created, 1);
        assert_eq!(census.reused, 1);
        assert_eq!(census.refused, 0);
        assert_eq!(census.translator_unavailable, 0);
        assert_eq!(census.by_kind[&AttemptKind::Compute].reused, 1);
        assert!(census.refusals.is_empty(), "{:?}", census.refusals);

        // Different content is a different key, so it is not taken for a reuse.
        let other = registration_key(
            &[PipelineStage::Compute],
            &[(PipelineStage::Compute, b"MTLB other".as_slice(), "main")],
            None,
        );
        assert_ne!(key, other);
        assert_eq!(
            registration_key(
                &[PipelineStage::Compute],
                &[(PipelineStage::Compute, b"MTLB compute".as_slice(), "main")],
                None
            ),
            registration_key(
                &[PipelineStage::Compute],
                &[(PipelineStage::Compute, b"MTLB compute".as_slice(), "main")],
                None
            ),
            "the same content always makes the same key"
        );
    }

    /// Stage names map onto the four census kinds, including the stages the
    /// native host cannot register (they are still a graphics attempt).
    #[test]
    fn stage_names_map_to_the_census_kinds() {
        assert_eq!(
            attempt_kind_of_names(&["vertex", "pixel"]),
            AttemptKind::Graphics
        );
        assert_eq!(
            attempt_kind_of_names(&["vertex", "pixel", "hull"]),
            AttemptKind::Graphics
        );
        assert_eq!(
            attempt_kind_of_names(&["compute"]),
            AttemptKind::Compute
        );
        assert_eq!(
            attempt_kind_of_names(&["mesh", "pixel"]),
            AttemptKind::MeshObject
        );
        assert_eq!(
            attempt_kind_of_names(&["amplification", "mesh", "pixel"]),
            AttemptKind::MeshObject
        );
        assert_eq!(attempt_kind_of_names(&[]), AttemptKind::Graphics);
    }
}
