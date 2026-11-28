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
//! The entry name is the DXIL entry the converter was told to convert and the
//! name the host looks up in the metallib. It is read from the container's own
//! `PSV0` table (the runtime-info entry-point string offset and the string
//! table it points into), never guessed: `VICTORY_METAL_PSO_ENTRY` is only an
//! explicit override, and `main` is only the fallback for a container that
//! predates that table.
//!
//! # What the converter is handed
//!
//! `IRObjectCreateFromDXIL` takes a *complete* `DXBC` container, the bytes a
//! D3D12 title passes as `D3D12_SHADER_BYTECODE`. Handing it a bare `DXIL`
//! part (the program header and bitcode without the container) crashes the
//! converter child, which surfaces as a worker exit and refuses every title
//! PSO. [`stage_program`] therefore keeps the whole container and validates it
//! before anything is translated or uploaded: `DXBC` magic, a bounded size, a
//! `DXIL` part whose `Program` kind agrees with the requested stage, and a
//! `PSV0` stage byte that agrees with it too. A mismatch is a named
//! [`PipelineError::Container`] refusal, never a worker exit.
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
#[cfg(not(test))]
use std::sync::OnceLock;
use std::sync::{Condvar, LazyLock, Mutex};

#[cfg(windows)]
use victory_metal::compiler_pool::PoolConfig;
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

    /// The DXIL `Program` kind a container of this stage must declare.
    ///
    /// These are the container's own program kinds (pixel 0, vertex 1, compute
    /// 5, mesh 13, amplification 14), not the converter's stage codes:
    /// IRConverter numbers mesh 5 and compute 8. A container whose `DXIL` part
    /// declares another kind is refused instead of being sent.
    pub const fn dxil_kind(self) -> victory_shader::ShaderKind {
        match self {
            Self::Vertex => victory_shader::ShaderKind::Vertex,
            Self::Pixel => victory_shader::ShaderKind::Pixel,
            Self::Mesh => victory_shader::ShaderKind::Mesh,
            Self::Object => victory_shader::ShaderKind::Amplification,
            Self::Compute => victory_shader::ShaderKind::Compute,
        }
    }

    /// The stage name used in refusal reasons and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Pixel => "pixel",
            Self::Mesh => "mesh",
            Self::Object => "object/amplification",
            Self::Compute => "compute",
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

/// One stage of a title PSO: its complete shader container and the entry point
/// the converter and the host both name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageProgram {
    /// The stage this program belongs to.
    pub stage: PipelineStage,
    /// The complete `DXBC` container the converter consumes (the container's
    /// `DXIL` part is its program; the container also carries `PSV0` and the
    /// signatures). A bare `DXIL` part is never valid here: IRConverter's
    /// `IRObjectCreateFromDXIL` takes the whole container.
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
    /// The host owns the translator pool but refused to translate or register
    /// the pipeline; the reason is the host's bounded category.
    HostTranslation(String),
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
            Self::HostTranslation(why) => {
                write!(f, "the host could not translate the pipeline: {why}")
            }
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

/// The monotonic registration epoch and the in-flight counter behind it.
///
/// A native pipeline registration and a render submission travel on
/// independent sockets (ADR-041), so the render socket has no ordering
/// guarantee of its own: a submission that selects a pipeline must not be
/// handed to the host while that pipeline's registration is still in flight.
/// Every registration settles the epoch exactly once, success or failure, and
/// a submission that names a pipeline waits on the epoch it observed after
/// enqueue. A successful registration also advances the separate
/// [`successful_registrations`] counter, so the boot can tell a settled failure
/// (which still has to release a waiter) from a hosted pipeline.
///
/// The guest is synchronous today: `pipeline::native_pipeline` holds the
/// pipeline's cell across the registration exchange and only installs a host id
/// after the reply, so no list can name a pipeline before its registration
/// returns. The barrier is therefore an invariant guard and a regression test,
/// not a change to the observed host ordering.
#[derive(Debug)]
struct RegistrationBarrier {
    inner: Mutex<RegistrationState>,
    settled: Condvar,
}

#[derive(Debug, Default)]
struct RegistrationState {
    epoch: u64,
    successful: u64,
    in_flight: u64,
    #[cfg(any(windows, test))]
    poisoned: BTreeMap<[u8; 32], String>,
}

impl RegistrationBarrier {
    fn new() -> Self {
        Self {
            inner: Mutex::new(RegistrationState::default()),
            settled: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RegistrationState> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn snapshot(&self) -> RegistrationSnapshot {
        let state = self.lock();
        RegistrationSnapshot {
            epoch: state.epoch,
            successful: state.successful,
            in_flight: usize::try_from(state.in_flight).unwrap_or(usize::MAX),
        }
    }

    fn begin(&self) -> PipelineRegistration {
        let mut state = self.lock();
        state.in_flight = state.in_flight.saturating_add(1);
        drop(state);
        PipelineRegistration { successful: false }
    }

    fn settle(&self, successful: bool) {
        let mut state = self.lock();
        state.epoch = state.epoch.saturating_add(1);
        if successful {
            state.successful = state.successful.saturating_add(1);
        }
        state.in_flight = state.in_flight.saturating_sub(1);
        drop(state);
        self.settled.notify_all();
    }

    fn wait_for_epoch(&self, required: u64) {
        let mut state = self.lock();
        while state.epoch < required {
            state = self
                .settled
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    #[cfg(any(windows, test))]
    fn poisoned(&self, key: &[u8; 32]) -> Option<String> {
        self.lock().poisoned.get(key).cloned()
    }

    #[cfg(any(windows, test))]
    fn poison(&self, key: [u8; 32], reason: &str) {
        self.lock().poisoned.insert(key, reason.to_string());
    }

    #[cfg(test)]
    fn reset(&self) {
        let mut state = self.lock();
        *state = RegistrationState::default();
    }
}

static REGISTRATION_BARRIER: LazyLock<RegistrationBarrier> =
    LazyLock::new(RegistrationBarrier::new);

/// The registration state a submission observes at enqueue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegistrationSnapshot {
    /// Settled registrations (successes and failures) so far.
    pub epoch: u64,
    /// Registrations that completed successfully.
    pub successful: u64,
    /// Registrations in flight when the snapshot was taken.
    pub in_flight: usize,
}

/// A guard for one in-flight native pipeline registration.
///
/// Dropping the guard settles the registration. [`Self::succeeded`] marks it a
/// success before the drop; every other path settles it as a failure, so a
/// refused registration releases a waiting submission instead of leaving it
/// blocked on an epoch that will never arrive.
#[derive(Debug)]
#[must_use = "a registration guard settles the epoch when it is dropped"]
pub struct PipelineRegistration {
    successful: bool,
}

impl PipelineRegistration {
    /// Marks this registration as successful (settled on drop).
    pub fn succeeded(mut self) {
        self.successful = true;
    }
}

impl Drop for PipelineRegistration {
    fn drop(&mut self) {
        REGISTRATION_BARRIER.settle(self.successful);
    }
}

/// The current registration snapshot.
pub fn registration_snapshot() -> RegistrationSnapshot {
    REGISTRATION_BARRIER.snapshot()
}

/// The monotonic registration epoch: settled registrations so far.
pub fn registration_epoch() -> u64 {
    REGISTRATION_BARRIER.snapshot().epoch
}

/// Native pipeline registrations that completed successfully.
pub fn successful_registrations() -> u64 {
    REGISTRATION_BARRIER.snapshot().successful
}

/// Native pipeline registrations in flight right now.
pub fn registrations_in_flight() -> usize {
    REGISTRATION_BARRIER.snapshot().in_flight
}

/// Announces one native pipeline registration that is about to reach the host.
///
/// The returned guard must be marked [`PipelineRegistration::succeeded`] on a
/// successful registration; its drop settles the epoch either way.
pub fn begin_pipeline_registration() -> PipelineRegistration {
    REGISTRATION_BARRIER.begin()
}

/// Waits until `required` registrations have settled.
///
/// A submission that observed `epoch + in_flight` at enqueue calls this before
/// its frame reaches the render socket: it cannot overtake a registration that
/// was already in flight. The wait is a condition variable, never a spin or a
/// sleep loop; a failed registration settles the epoch, so the wait cannot
/// outlive the registration's own socket timeout.
pub fn wait_for_registration_epoch(required: u64) {
    REGISTRATION_BARRIER.wait_for_epoch(required);
}

/// The cached refusal of a pipeline registration content key, if any.
#[cfg(any(windows, test))]
fn poisoned_pipeline(key: &[u8; 32]) -> Option<String> {
    REGISTRATION_BARRIER.poisoned(key)
}

/// Marks a pipeline registration content key as permanently refused.
#[cfg(any(windows, test))]
fn poison_pipeline(key: [u8; 32], reason: &str) {
    REGISTRATION_BARRIER.poison(key, reason);
}

/// Empties the registration barrier: tests start each case from a known state.
#[cfg(test)]
pub(crate) fn reset_registration_barrier() {
    REGISTRATION_BARRIER.reset();
}

/// Serialises the tests that drive the process-global registration barrier.
///
/// The barrier is one instance per process by design; a test that holds a
/// registration open must not race another test's reset or registration.
#[cfg(test)]
pub(crate) fn registration_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

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
    /// The ARM64 host translated the DXIL with its own converter pool and
    /// registered the pipeline (ADR-038).
    HostTranslated,
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
    /// The host owns the translator but could not translate or register it.
    HostTranslation,
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
            Self::HostTranslation => "host-translation",
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
            PipelineError::HostTranslation(_) => Self::HostTranslation,
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
    /// Attempts the host translated with its own converter pool (ADR-038).
    pub host_translated: u64,
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
    /// Attempts the host translated with its own converter pool.
    pub host_translated: u64,
    /// Stages whose complete DXIL container was taken from the raw console
    /// bytes because it already carried a real module (pass-through).
    pub stages_passed_through: u64,
    /// Stages whose DXIL container was emitted locally by `victory_shader`
    /// before upload because the console container carried only a stub.
    pub stages_translated_locally: u64,
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
             pipeline_reused={} pipeline_translator_unavailable={} \
             pipeline_host_translated={} pipeline_stages_translated_locally={} \
             pipeline_stages_passed_through={}",
            self.attempts,
            self.created,
            self.refused,
            self.reused,
            self.translator_unavailable,
            self.host_translated,
            self.stages_translated_locally,
            self.stages_passed_through
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
    stages_passed_through: u64,
    stages_translated_locally: u64,
}

static CENSUS: Mutex<Census> = Mutex::new(Census {
    kinds: BTreeMap::new(),
    refusals: BTreeMap::new(),
    first: None,
    stages_passed_through: 0,
    stages_translated_locally: 0,
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
            AttemptOutcome::HostTranslated => {
                counts.host_translated = counts.host_translated.saturating_add(1)
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
            stages_passed_through: self.stages_passed_through,
            stages_translated_locally: self.stages_translated_locally,
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
            snapshot.host_translated = snapshot
                .host_translated
                .saturating_add(counts.host_translated);
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

/// Counts one stage whose DXIL container was taken from the raw console bytes
/// unchanged (it already carried a real module).
pub fn note_stage_passed_through() {
    let mut census = CENSUS.lock().unwrap_or_else(|error| error.into_inner());
    census.stages_passed_through = census.stages_passed_through.saturating_add(1);
}

/// Counts one stage whose DXIL container was emitted locally by
/// `victory_shader` before it was uploaded for translation.
pub fn note_stage_translated_locally() {
    let mut census = CENSUS.lock().unwrap_or_else(|error| error.into_inner());
    census.stages_translated_locally = census.stages_translated_locally.saturating_add(1);
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

/// The environment variable that names the converter's `--stdio` worker.
///
/// ADR-038 has the macOS launcher export it for `victory_metal_host`; it is
/// host-only and must not reach the Win64 guest.
#[cfg(any(windows, test))]
const WORKER_ENV: &str = "VICTORY_IRCONVERTER_WORKER";

/// Whether the Win64 guest could ever run `path` as its converter worker.
///
/// ADR-038 keeps the converter worker and `libmetalirconverter.dylib` on the
/// macOS host. A guest that inherits `VICTORY_IRCONVERTER_WORKER` must ignore
/// the macOS binary (a Mach-O path under `/` or `Z:\`): the child would die
/// immediately, which surfaces as `Translation` and hides the fact that no
/// converter can run in this process. Only a Windows `.exe` outside the macOS
/// root qualifies; ignoring everything else keeps the pool cache-only so a cold
/// stage is `ConverterUnavailable` and selects the VP3 host-translation path.
#[cfg(any(windows, test))]
fn guest_can_run_worker(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let macos_root = text.starts_with('/')
        || text
            .as_bytes()
            .get(..2)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"Z:"));
    let windows_exe = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"));
    windows_exe && !macos_root
}

/// Resolves the pool's worker program.
///
/// `guest` is `true` in the Win64 guest (`cfg(windows)`). There an inherited
/// macOS worker path is ignored with an explicit warning, so the pool stays
/// cache-only and the caller sees `ConverterUnavailable`, not `Translation`.
/// The macOS host keeps accepting the same path; the split lives in the
/// launcher and this is its defence in depth (ADR-038).
#[cfg(any(windows, test))]
fn resolve_worker_program(guest: bool) -> Option<PathBuf> {
    let program = std::env::var_os(WORKER_ENV)
        .map(PathBuf::from)
        .filter(|path| path.is_file())?;
    if guest && !guest_can_run_worker(&program) {
        tracing::warn!(
            worker = %program.display(),
            "VICTORY_IRCONVERTER_WORKER names a macOS converter (ADR-038); the Win64 guest ignores it and stays cache-only"
        );
        return None;
    }
    Some(program)
}

/// Builds the process-wide pool.
///
/// The macOS host keeps [`CompilerPool::from_env`]'s resolution, including the
/// `victory-irconverter` sibling fallback. The Win64 guest resolves through
/// [`resolve_worker_program`] instead: it can only run a Windows worker and
/// must not start the host-only macOS binary the launcher exports (ADR-038).
fn build_pool() -> CompilerPool {
    #[cfg(windows)]
    {
        CompilerPool::new(PoolConfig::from_env(), resolve_worker_program(true))
    }
    #[cfg(not(windows))]
    {
        CompilerPool::from_env()
    }
}

/// The process-wide pool. One pool means one persistent worker set for the
/// whole run (ADR-018); it is created from the environment on first use.
#[cfg(not(test))]
fn pool() -> &'static CompilerPool {
    static POOL: OnceLock<CompilerPool> = OnceLock::new();
    POOL.get_or_init(build_pool)
}

/// The test binary's pool, rebuildable between tests.
///
/// The production pool is a `OnceLock` because one worker set must serve the
/// whole run; a unit test that must drive the real converter needs the pool to
/// read the environment it just set, so the test build keeps the pool behind a
/// mutex and [`reset_pool`] discards it. A discarded pool is leaked on
/// purpose: its workers see EOF when the test process closes their stdin and
/// exit, and no test may hold a reference to it across a reset.
#[cfg(test)]
static TEST_POOL: Mutex<Option<&'static CompilerPool>> = Mutex::new(None);

#[cfg(test)]
fn pool() -> &'static CompilerPool {
    let mut slot = TEST_POOL.lock().unwrap_or_else(|error| error.into_inner());
    *slot.get_or_insert_with(|| Box::leak(Box::new(build_pool())))
}

/// Discards the test binary's pool so the next use rebuilds it from the
/// current environment (tests only).
#[cfg(test)]
fn reset_pool() {
    let mut slot = TEST_POOL.lock().unwrap_or_else(|error| error.into_inner());
    *slot = None;
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

/// Largest shader container the converter is handed.
///
/// It matches the pool's `MAX_JOB_DXIL_BYTES` and the host translator's
/// upload bound, so a container the guest accepts is one the host accepts.
pub const MAX_STAGE_CONTAINER_BYTES: usize = 16 * 1024 * 1024;

/// Largest accepted entry-point name, matching the converter's own limit.
pub const MAX_ENTRY_POINT_NAME_BYTES: usize = 256;

const _: () = assert!(MAX_STAGE_CONTAINER_BYTES as u64 == MAX_HOST_STAGE_BYTES);

/// The entry point used when neither an override nor the container names one.
pub const DEFAULT_ENTRY_POINT: &str = "main";

/// A `PSV0` runtime-info size that carries the entry-point string offset.
///
/// DXC writes 52: the size field itself plus the 48-byte runtime info, then the
/// entry-point string offset. The repository's own emitter writes 48 (no entry
/// name), which is why [`container_entry_point`] returns `None` there.
const PSV0_RUNTIME_INFO_WITH_ENTRY: u32 = 52;

/// Offset of the entry-point string offset inside a `PSV0` part.
const PSV0_ENTRY_OFFSET: usize = 52;

/// The raw `Program` kind code of a DXIL program header.
const fn raw_shader_kind(kind: victory_shader::ShaderKind) -> u16 {
    use victory_shader::ShaderKind;
    match kind {
        ShaderKind::Pixel => 0,
        ShaderKind::Vertex => 1,
        ShaderKind::Geometry => 2,
        ShaderKind::Hull => 3,
        ShaderKind::Domain => 4,
        ShaderKind::Compute => 5,
        ShaderKind::Library => 6,
        ShaderKind::RayGeneration => 7,
        ShaderKind::Intersection => 8,
        ShaderKind::AnyHit => 9,
        ShaderKind::ClosestHit => 10,
        ShaderKind::Miss => 11,
        ShaderKind::Callable => 12,
        ShaderKind::Mesh => 13,
        ShaderKind::Amplification => 14,
        ShaderKind::Node => 15,
        ShaderKind::Unknown(raw) => raw,
    }
}

/// The stage byte of the container's own `PSV0` part, when it carries one.
///
/// The part starts with the runtime-info size, then the info; its stage byte
/// is `info[24]` (offset 28 of the part). `PSV0`'s stage numbering is the DXIL
/// program kind (pixel 0, vertex 1, compute 5, mesh 13, amplification 14).
fn psv0_stage(container: &[u8]) -> Option<u16> {
    let parsed = victory_shader::Container::parse(container).ok()?;
    let psv = parsed.part(victory_shader::FourCC::PSV0)?.data;
    let size = u32::from_le_bytes(psv.get(..4)?.try_into().ok()?);
    if size < 28 || usize::try_from(size).ok()? > psv.len() {
        return None;
    }
    psv.get(28).map(|byte| u16::from(*byte))
}

/// The entry point the container's own `PSV0` table names, if it carries one.
///
/// `PSV0` holds the entry-point string offset at [`PSV0_ENTRY_OFFSET`] and a
/// NUL-separated string table, preceded by its `u32` size, after the runtime
/// info and the resource/signature records. The name is resolved by that
/// offset against a table whose declared size, leading empty string and string
/// bounds all agree; a container without that table (the repository's own
/// emitter) yields `None` instead of a guess.
fn container_entry_point(container: &[u8]) -> Option<String> {
    let parsed = victory_shader::Container::parse(container).ok()?;
    let psv = parsed.part(victory_shader::FourCC::PSV0)?.data;
    let runtime_info = u32::from_le_bytes(psv.get(..4)?.try_into().ok()?);
    if runtime_info != PSV0_RUNTIME_INFO_WITH_ENTRY {
        return None;
    }
    let entry = usize::try_from(u32::from_le_bytes(
        psv.get(PSV0_ENTRY_OFFSET..PSV0_ENTRY_OFFSET + 4)?
            .try_into()
            .ok()?,
    ))
    .ok()?;
    // The table's size field is the `u32` immediately before it.
    for start in (PSV0_ENTRY_OFFSET + 4..psv.len()).step_by(4) {
        let Some(size) = psv
            .get(start - 4..start)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_le_bytes)
            .and_then(|size| usize::try_from(size).ok())
        else {
            continue;
        };
        let Some(end) = start.checked_add(size) else {
            continue;
        };
        if end > psv.len() {
            continue;
        }
        let table = &psv[start..end];
        if table.first() != Some(&0) || entry >= table.len() {
            continue;
        }
        // The name must start a whole string in the table.
        if entry != 0 && table[entry - 1] != 0 {
            continue;
        }
        let name = &table[entry..];
        let Some(length) = name.iter().position(|byte| *byte == 0) else {
            continue;
        };
        let name = &name[..length];
        if name.is_empty() || name.len() > MAX_ENTRY_POINT_NAME_BYTES {
            continue;
        }
        if !name.iter().all(|byte| (0x20..=0x7e).contains(byte)) {
            continue;
        }
        return std::str::from_utf8(name).ok().map(str::to_string);
    }
    None
}

/// The readable numbers of a container's `DXIL` program header.
///
/// A real container's `DXIL` part is a [`victory_shader::DxilProgram`] header
/// followed by the bitcode, but producers differ in what sits between the
/// header and the bitcode (a 20-byte digest in newer containers, none in older
/// ones, Xbox `XDAT` tables in console containers). Only the header is needed
/// to learn the stage and the shader model: kind/SM in the version dword, the
/// program size in dwords at offset 4, and the bitcode offset/size at 16/20
/// (both relative to the `DXIL` magic at offset 8). Bitcode well-formedness is
/// deliberately not checked here — the converter is the authority and answers
/// a named error now that each job is isolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DxilHeader {
    /// The container's own program kind (pixel 0, vertex 1, compute 5,
    /// mesh 13, amplification 14).
    kind: u16,
    /// Shader model major (6 for every DXIL container).
    major: u8,
    /// Shader model minor (0, 5, 6, 7, 8, ...).
    minor: u8,
    /// The header's bitcode offset, relative to the `DXIL` magic at byte 8.
    bitcode_offset: u32,
    /// The header's bitcode size.
    bitcode_size: u32,
}

impl DxilHeader {
    /// Reads the header fields whose bounds are the 24-byte header itself.
    ///
    /// This is the census view: it reports the numbers even when the program
    /// size dword disagrees with the part, because that disagreement is itself
    /// worth seeing. Validation uses [`Self::checked`].
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 24 || data.get(8..12) != Some(b"DXIL") {
            return None;
        }
        let version = u32::from_le_bytes(data.get(..4)?.try_into().ok()?);
        Some(Self {
            kind: (version >> 16) as u16,
            major: ((version >> 4) & 0xF) as u8,
            minor: (version & 0xF) as u8,
            bitcode_offset: u32::from_le_bytes(data.get(16..20)?.try_into().ok()?),
            bitcode_size: u32::from_le_bytes(data.get(20..24)?.try_into().ok()?),
        })
    }

    /// Reads the header only when its program-size dword agrees with the part.
    ///
    /// The stage is derived from a header the converter would also accept as
    /// bounded; a size that is below the header or past the part means the
    /// header itself cannot be trusted.
    fn checked(data: &[u8]) -> Option<Self> {
        let header = Self::parse(data)?;
        let size = usize::try_from(u32::from_le_bytes(data.get(4..8)?.try_into().ok()?))
            .ok()?
            .checked_mul(4)?;
        if size < 24 || size > data.len() {
            return None;
        }
        Some(header)
    }

    /// `kind (Name), SM major.minor`, the bounded form used in refusals.
    fn summary(self) -> String {
        format!(
            "kind {} ({}, SM {}.{})",
            self.kind,
            shader_kind_name(self.kind),
            self.major,
            self.minor
        )
    }
}

/// The checked `DXIL` program header of a container, if it can be read.
fn dxil_header(container: &[u8]) -> Option<DxilHeader> {
    let parsed = victory_shader::Container::parse(container).ok()?;
    DxilHeader::checked(parsed.part(victory_shader::FourCC::DXIL)?.data)
}

/// The fewest bitcode bytes a container must declare to count as carrying a
/// real DXIL module.
///
/// An Xbox `DXIL` part may be a 4-byte placeholder beside the console ISA
/// chunk, and older synthetic/malformed parts declare at most a handful of
/// bytes. A module this repository's own emitter writes is at least several
/// hundred bytes even for a trivial stage (the fixtures are 1,116-1,292
/// bytes), so 512 bytes cleanly separates a stub from a module without
/// rejecting a small real one.
pub const REAL_DXIL_BITCODE_FLOOR: u32 = 512;

/// The bitcode size a container's `DXIL` program header declares, when the
/// header is structurally sound and the bitcode lies inside the part.
fn declared_bitcode_size(container: &[u8]) -> Option<u32> {
    let parsed = victory_shader::Container::parse(container).ok()?;
    let part = parsed.part(victory_shader::FourCC::DXIL)?.data;
    let header = DxilHeader::checked(part)?;
    let end = 8usize
        .checked_add(header.bitcode_offset as usize)?
        .checked_add(header.bitcode_size as usize)?;
    (header.bitcode_size > 0 && end <= part.len()).then_some(header.bitcode_size)
}

/// Whether `container` carries a real DXIL module rather than only a stub.
///
/// This is the pass-through test: the raw console container's `DXIL` part must
/// parse as a program header, declare a real bitcode size (at least
/// [`REAL_DXIL_BITCODE_FLOOR`] bytes) and keep that bitcode inside the part.
/// A 4-byte placeholder, a headerless part or a truncated one is not a module
/// and must never be handed to the converter as if it were.
pub fn carries_real_dxil(container: &[u8]) -> bool {
    declared_bitcode_size(container).is_some_and(|bytes| bytes >= REAL_DXIL_BITCODE_FLOOR)
}

/// Whether a locally translated container carries a usable (non-stub) DXIL
/// module: the same structural test without the pass-through size floor.
///
/// The emitter's output is trusted to be a module, but a container whose
/// `DXIL` part is empty or truncated is still refused instead of uploaded.
pub fn carries_usable_dxil(container: &[u8]) -> bool {
    declared_bitcode_size(container).is_some()
}

/// Whether a translated mesh container's declared outputs exceed Metal's
/// budget ([`victory_shader::emit::METAL_MESH_OUTPUT_BYTES`]).
///
/// Counted as the runtime counts them: the larger of the packed component
/// footprint (`MeshOutputs::bytes`: 4 bytes per declared component of a vertex
/// and of a primitive) and Metal's aligned layout, which pads a row to 16
/// bytes and can overrun the limit while the packed count fits (#172). This is
/// the threshold `victory_shader`'s linker itself uses. A mesh stage over the
/// limit is re-translated together with its pixel shader so the outputs the
/// pixel shader never reads are pruned.
pub fn mesh_outputs_over_budget(mesh: &[u8]) -> bool {
    let packed = victory_shader::emit::mesh_outputs(mesh).map(|outputs| outputs.bytes());
    let aligned = victory_shader::emit::mesh_output_layout(mesh)
        .ok()
        .flatten()
        .map(|layout| layout.aligned_before);
    packed
        .max(aligned)
        .is_some_and(|bytes| bytes > victory_shader::emit::METAL_MESH_OUTPUT_BYTES)
}

/// Whether a `[mesh, pixel]` pair must use the linked mesh/pixel translation
/// instead of the standalone mesh translation.
///
/// Linking needs the pixel shader to know which outputs are read, and is only
/// worth doing when the standalone mesh outputs are over Metal's budget.
pub fn linked_mesh_pair_needed(has_pixel: bool, mesh: &[u8]) -> bool {
    has_pixel && mesh_outputs_over_budget(mesh)
}

/// The `Program` kind name of a raw kind code, for refusal reasons.
fn shader_kind_name(raw: u16) -> String {
    use victory_shader::ShaderKind;
    let kind = match raw {
        0 => ShaderKind::Pixel,
        1 => ShaderKind::Vertex,
        2 => ShaderKind::Geometry,
        3 => ShaderKind::Hull,
        4 => ShaderKind::Domain,
        5 => ShaderKind::Compute,
        6 => ShaderKind::Library,
        7 => ShaderKind::RayGeneration,
        8 => ShaderKind::Intersection,
        9 => ShaderKind::AnyHit,
        10 => ShaderKind::ClosestHit,
        11 => ShaderKind::Miss,
        12 => ShaderKind::Callable,
        13 => ShaderKind::Mesh,
        14 => ShaderKind::Amplification,
        15 => ShaderKind::Node,
        other => ShaderKind::Unknown(other),
    };
    format!("{kind:?}")
}

/// Opt-in structural census of the shader containers the guest validates.
///
/// Set `VICTORY_METAL_PART_CENSUS=1` to log one line per distinct container:
/// its size bucket, its part fourccs with part sizes (at most
/// [`MAX_CENSUS_PARTS`]) and, for a `DXIL` part, the program kind, shader-model
/// major/minor and bitcode offset/size from the program header. The line never
/// carries shader bytes, entry-point names or paths, so it is safe in a devlog.
/// Off by default: a normal boot logs nothing here.
pub const PART_CENSUS_ENV: &str = "VICTORY_METAL_PART_CENSUS";

/// Most parts named in one census line; the rest are reported as a count.
pub const MAX_CENSUS_PARTS: usize = 12;

/// Most distinct containers one process censuses before it stops emitting.
///
/// The bound keeps a title with thousands of shaders from flooding a boot log
/// and keeps the dedup memory fixed; a single truncation line states the cut.
const MAX_CENSUS_CONTAINERS: usize = 256;

/// The content hashes already censused, and whether the bound was reported.
struct PartCensusState {
    seen: Vec<[u8; 32]>,
    truncated: bool,
}

static PART_CENSUS: Mutex<PartCensusState> = Mutex::new(PartCensusState {
    seen: Vec::new(),
    truncated: false,
});

/// Whether the opt-in part census is enabled (`VICTORY_METAL_PART_CENSUS=1`).
fn part_census_enabled() -> bool {
    std::env::var_os(PART_CENSUS_ENV).is_some_and(|value| value.to_str() == Some("1"))
}

/// The power-of-two upper bound of a container size, as `<=N KiB`.
///
/// A bucket, not the exact size: the census answers "which scale of container"
/// without printing a byte length that could fingerprint an asset.
fn container_size_bucket(bytes: usize) -> String {
    let mut bound = 1024usize;
    while bytes > bound && bound < (1 << 40) {
        bound <<= 1;
    }
    format!("<={}KiB", bound / 1024)
}

/// The census line of one container, or `None` if it is not a `DXBC` container.
///
/// The returned hash is the full BLAKE3 content hash (the caller dedups on it);
/// the line prints only its first 16 hex characters. Fields are bounded: at
/// most [`MAX_CENSUS_PARTS`] parts, each fourcc (4 printable bytes) and a
/// decimal size, then one `DXIL` header (five small integers). No part data,
/// entry-point text or path is ever included.
fn part_census_line(container: &[u8]) -> Option<([u8; 32], String)> {
    let parsed = victory_shader::Container::parse(container).ok()?;
    let digest = blake3::hash(container);
    let hash = *digest.as_bytes();
    let hex = digest.to_hex();
    let short = &hex.as_str()[..16];
    let mut line = format!(
        "shader container census hash={short} size_bucket={} parts={} [",
        container_size_bucket(container.len()),
        parsed.parts.len()
    );
    for (index, part) in parsed.parts.iter().take(MAX_CENSUS_PARTS).enumerate() {
        if index > 0 {
            line.push(' ');
        }
        line.push_str(&format!("{}:{}", part.fourcc, part.data.len()));
    }
    if parsed.parts.len() > MAX_CENSUS_PARTS {
        line.push_str(&format!(" +{} more", parsed.parts.len() - MAX_CENSUS_PARTS));
    }
    line.push(']');
    match parsed
        .part(victory_shader::FourCC::DXIL)
        .and_then(|part| DxilHeader::parse(part.data))
    {
        Some(header) => line.push_str(&format!(
            " dxil=kind:{}({}) sm:{}.{} bitcode_offset:{} bitcode_size:{}",
            header.kind,
            shader_kind_name(header.kind),
            header.major,
            header.minor,
            header.bitcode_offset,
            header.bitcode_size
        )),
        None => line.push_str(" dxil=none"),
    }
    Some((hash, line))
}

/// Logs the census of `container` once per distinct content, when enabled.
///
/// Returns the emitted line (for tests); `None` when the census is off, the
/// bytes are not a `DXBC` container, this content was already censused, or the
/// [`MAX_CENSUS_CONTAINERS`] bound was reached. The bound reports itself once.
fn note_part_census(container: &[u8]) -> Option<String> {
    if !part_census_enabled() {
        return None;
    }
    let (hash, line) = part_census_line(container)?;
    {
        let mut state = PART_CENSUS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.seen.contains(&hash) {
            return None;
        }
        if state.seen.len() >= MAX_CENSUS_CONTAINERS {
            if state.truncated {
                return None;
            }
            state.truncated = true;
            drop(state);
            tracing::info!(
                target: "victory_pso_census",
                "shader container census truncated at {MAX_CENSUS_CONTAINERS} distinct containers"
            );
            return None;
        }
        state.seen.push(hash);
    }
    tracing::info!(target: "victory_pso_census", "{}", line);
    Some(line)
}

/// Forgets the censused containers (tests only: a process censuses once).
#[cfg(test)]
pub(crate) fn reset_part_census() {
    *PART_CENSUS
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = PartCensusState {
        seen: Vec::new(),
        truncated: false,
    };
}

/// Validates `container` as the complete shader container of `stage`.
///
/// IRConverter's `IRObjectCreateFromDXIL` takes the whole `DXBC` container: a
/// bare `DXIL` part (program header plus bitcode) crashes the converter child
/// instead of answering. The container's own `Program` part and `PSV0` stage
/// byte must both name `stage` when they can be read; a disagreement is a named
/// refusal here, never a worker exit or a wrong library. The bitcode itself is
/// left to the converter, which now reports a named error instead of crashing
/// because every job runs in its own child.
fn checked_stage_container(
    stage: PipelineStage,
    container: &[u8],
) -> Result<(), PipelineError> {
    if container.is_empty() {
        return Err(PipelineError::NoDxil(stage.label()));
    }
    if container.len() > MAX_STAGE_CONTAINER_BYTES {
        return Err(PipelineError::Container(format!(
            "the {} container is {} bytes, over the {MAX_STAGE_CONTAINER_BYTES}-byte converter \
             limit",
            stage.label(),
            container.len()
        )));
    }
    if container.get(..4) != Some(&b"DXBC"[..]) {
        return Err(PipelineError::Container(format!(
            "the {} stage is not a complete DXBC container: the converter takes the whole \
             container (`DXBC` magic), never a bare DXIL part",
            stage.label()
        )));
    }
    // The part table is what must hold: magic, bounds and a `DXIL` part. A
    // truncated table, an out-of-bounds part or a non-container is refused
    // here; the bitcode that follows is the converter's to judge.
    let parsed = victory_shader::Container::parse(container)
        .map_err(|error| PipelineError::Container(format!("{}: {error}", stage.label())))?;
    // Opt-in telemetry: the structural facts of every container this path
    // validates, so a converter refusal can be read as "kind X, SM Y".
    note_part_census(container);
    parsed
        .part(victory_shader::FourCC::DXIL)
        .ok_or(PipelineError::NoDxil(stage.label()))?;

    let header = dxil_header(container);
    let declared = header.map(|header| header.kind);
    let psv = psv0_stage(container);
    let expected = raw_shader_kind(stage.dxil_kind());
    if let Some(header) = header
        && header.kind != expected
    {
        return Err(PipelineError::Container(format!(
            "the {} stage's container declares {}, not a {} program",
            stage.label(),
            header.summary(),
            shader_kind_name(expected)
        )));
    }
    if let (Some(header), Some(psv)) = (header, psv)
        && header.kind != psv
    {
        return Err(PipelineError::Container(format!(
            "the {} stage's container Program part ({}) disagrees with its PSV0 stage byte \
             ({psv})",
            stage.label(),
            header.summary()
        )));
    }
    if let Some(psv) = psv
        && psv != expected
    {
        return Err(PipelineError::Container(format!(
            "the {} stage's container PSV0 stage byte ({psv}) names another stage than the \
             requested one (kind {expected})",
            stage.label()
        )));
    }
    if declared.is_none() && psv.is_none() {
        return Err(PipelineError::Container(format!(
            "the {} stage's DXIL part is malformed: no readable DXIL program header and no \
             PSV0 stage to derive the stage from",
            stage.label()
        )));
    }
    Ok(())
}

/// The explicit `VICTORY_METAL_PSO_ENTRY` override, when one is set.
fn explicit_entry_point() -> Option<String> {
    std::env::var("VICTORY_METAL_PSO_ENTRY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Validates the entry point a [`StageProgram`] will name to the converter.
fn checked_entry(program: &StageProgram) -> Result<(), PipelineError> {
    if program.entry.is_empty() {
        return Err(PipelineError::Container(format!(
            "the {} stage has no entry point",
            program.stage.label()
        )));
    }
    if program.entry.len() > MAX_ENTRY_POINT_NAME_BYTES {
        return Err(PipelineError::Container(format!(
            "the {} stage's entry point is {} bytes, over the {MAX_ENTRY_POINT_NAME_BYTES}-byte \
             converter limit",
            program.stage.label(),
            program.entry.len()
        )));
    }
    if program.entry.as_bytes().contains(&0) {
        return Err(PipelineError::Container(format!(
            "the {} stage's entry point contains a NUL",
            program.stage.label()
        )));
    }
    Ok(())
}

/// Validates a [`StageProgram`] before any bytes reach a converter.
///
/// Every path that hands a program to `victory-irconverter` (the in-process
/// pool and the host-translation upload) calls this first, so a container that
/// is not a complete `DXBC` container, names another stage, or carries no
/// usable entry point is a bounded [`PipelineError`] instead of a converter
/// crash the pool can only report as `WorkerExited`.
pub fn validate_stage_program(program: &StageProgram) -> Result<(), PipelineError> {
    checked_stage_container(program.stage, &program.dxil)?;
    checked_entry(program)
}

/// The complete shader container and entry point of one title stage.
///
/// A container with no `DXIL` part (console GCN only) is
/// [`PipelineError::NoDxil`]: there is nothing the converter could translate.
/// The returned [`StageProgram`] keeps the whole container, because that is
/// what the converter consumes, and names the entry point the container's own
/// `PSV0` table exports; `main` is only the fallback for a container without
/// that table (see [`container_entry_point`]).
pub fn stage_program(
    stage: PipelineStage,
    container: &[u8],
) -> Result<StageProgram, PipelineError> {
    checked_stage_container(stage, container)?;
    let entry = explicit_entry_point()
        .or_else(|| container_entry_point(container))
        .unwrap_or_else(|| DEFAULT_ENTRY_POINT.to_string());
    let program = StageProgram {
        stage,
        dxil: container.to_vec(),
        entry,
    };
    checked_entry(&program)?;
    Ok(program)
}

/// The program one stage must register, choosing between the raw console
/// container and the container translated locally.
///
/// The rule (Cycle 14): a raw container that already carries a real DXIL
/// module ([`carries_real_dxil`]) is passed through unchanged, so the Win64
/// D3D12 reference path and the synthetic converter tests keep byte parity.
/// Otherwise `translated` (the complete `DXBC` container
/// `victory_shader::emit::translate` produced, normally
/// [`crate::pipeline::Shader::translated`]'s `host.dxil`) is validated and
/// used, and the stage is counted as translated locally. A stage with neither
/// a real raw module nor a translation is a named
/// [`PipelineError::Translation`] refusal: a 4-byte placeholder is never
/// uploaded to the converter under any circumstances.
///
/// The chosen container is validated by [`stage_program`], so the converted
/// entry point is read from the container that is actually sent (the
/// emitter's modules export `main` and carry no `PSV0` entry table), never
/// from the raw console container's `PSV0`.
pub fn select_stage_program(
    stage: PipelineStage,
    raw: &[u8],
    translated: Option<&[u8]>,
) -> Result<StageProgram, PipelineError> {
    if carries_real_dxil(raw) {
        let program = stage_program(stage, raw)?;
        note_stage_passed_through();
        return Ok(program);
    }
    let Some(translated) = translated else {
        return Err(PipelineError::Translation(format!(
            "the {} stage's console container carries no real DXIL module (a bitcode size below \
             {REAL_DXIL_BITCODE_FLOOR} bytes is a stub) and it did not translate",
            stage.label()
        )));
    };
    if !carries_usable_dxil(translated) {
        return Err(PipelineError::Translation(format!(
            "the {} stage's locally translated container carries no usable DXIL module",
            stage.label()
        )));
    }
    let program = stage_program(stage, translated)?;
    note_stage_translated_locally();
    Ok(program)
}

/// The entry point named in every converted DXIL module this repository has
/// seen (`main`), overridable with `VICTORY_METAL_PSO_ENTRY`.
///
/// [`stage_program`] prefers the container's own `PSV0` name over this; the
/// helper is kept for callers that build a [`StageProgram`] themselves.
pub fn entry_point() -> String {
    explicit_entry_point().unwrap_or_else(|| DEFAULT_ENTRY_POINT.to_string())
}

/// The pool statistics, for telemetry and tests.
pub fn pool_stats() -> PoolStats {
    pool().stats()
}

/// The worker program the pool will start, when one is configured.
///
/// The pool resolves it from `VICTORY_IRCONVERTER_WORKER` or the
/// `victory-irconverter` sibling of this executable. The Win64 guest can run
/// neither (the converter and its worker are macOS builds), so it ignores an
/// inherited macOS `VICTORY_IRCONVERTER_WORKER` (ADR-038) and this is `None`
/// there: every cold translation is then explicit.
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
///
/// A pool with no worker program is the environmental case: nothing in this
/// process can translate, so the refusal is [`PipelineError::ConverterUnavailable`]
/// (which selects the VP3 host-translation path) and not a translation failure.
/// A worker that did start and failed keeps [`PipelineError::Translation`].
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
    validate_stage_program(program)?;
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
    validate_stage_program(program)?;
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

/// The share-relative directory host-translated DXIL stages are uploaded to.
pub const HOST_STAGE_DIR: &str = "stages";
/// Largest DXIL stage the host translator accepts, matching the pool's limit.
pub const MAX_HOST_STAGE_BYTES: u64 = 16 * 1024 * 1024;

/// Uploads one stage's DXIL to `stages/<blake3>.bin`, idempotently.
///
/// The name is derived only from the bytes, so a repeat upload of the same
/// module is a metadata check and no write. A temporary name is renamed into
/// place, so the host can never open a half-written stage.
#[cfg(any(windows, test))]
fn upload_host_stage(share_dir: &Path, bytes: &[u8]) -> Result<String, PipelineError> {
    if bytes.is_empty() {
        return Err(PipelineError::HostTranslation(
            "a shader stage has no DXIL bytes to upload".to_string(),
        ));
    }
    if bytes.len() as u64 > MAX_HOST_STAGE_BYTES {
        return Err(PipelineError::HostTranslation(format!(
            "a shader stage is {} bytes, over the {MAX_HOST_STAGE_BYTES}-byte host limit",
            bytes.len()
        )));
    }
    let hash = blake3::hash(bytes).to_hex().to_string();
    let name = format!("{HOST_STAGE_DIR}/{hash}.bin");
    let path = share_dir.join(&name);
    if let Ok(metadata) = std::fs::symlink_metadata(&path)
        && metadata.is_file()
        && metadata.len() == bytes.len() as u64
    {
        return Ok(name);
    }
    let directory = share_dir.join(HOST_STAGE_DIR);
    std::fs::create_dir_all(&directory)
        .map_err(|error| PipelineError::Write(format!("{}: {error}", directory.display())))?;
    let temporary = directory.join(format!(".{hash}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)
        .map_err(|error| PipelineError::Write(format!("{}: {error}", temporary.display())))?;
    std::fs::rename(&temporary, &path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        PipelineError::Write(format!("{}: {error}", path.display()))
    })?;
    Ok(name)
}

/// The `PipelineState` blob a host-translated registration carries.
#[cfg(any(windows, test))]
fn host_pipeline_state(
    programs: &[StageProgram],
    details: &PipelineDetails<'_>,
    kind: PipelineKind,
) -> Result<Vec<u8>, PipelineError> {
    if !details.state.is_empty() {
        return Ok(details.state.to_vec());
    }
    Ok(match kind {
        // A compute `PipelineState` carries no colour/depth shape.
        PipelineKind::Compute => Vec::new(),
        PipelineKind::Graphics => {
            let pixel_entry = programs
                .iter()
                .find(|program| program.stage == PipelineStage::Pixel)
                .map(|program| program.entry.as_str())
                .unwrap_or_default();
            encode_graphics_state(&[DEFAULT_COLOR_FORMAT], pixel_entry)
        }
        PipelineKind::Mesh => {
            let stages: Vec<(PipelineStage, &[u8], &str)> = programs
                .iter()
                .map(|program| (program.stage, &[][..], program.entry.as_str()))
                .collect();
            default_mesh_state(&stages, &[DEFAULT_COLOR_FORMAT])
        }
    })
}

/// Hands the real DXIL stages to the ARM64 host when this process has no usable
/// converter (the Win64 Wine guest).
///
/// Every stage and the optional root-signature witness are uploaded to
/// `VICTORY_METAL_SHARE_DIR/stages/<blake3>.bin`; only their share-relative
/// names, stage codes, entries, encoded state and the root name travel over the
/// 64 KiB control frame. The host translates with its own bounded Rosetta pool,
/// derives the `VAB1`-equivalent ABI from the reflection and answers the native
/// pipeline id. A refusal becomes [`PipelineError::HostTranslation`] with the
/// host's bounded category, never a fabricated library or a D3DMetal fallback.
#[cfg(any(windows, test))]
fn register_with_host_translation(
    programs: &[StageProgram],
    details: &PipelineDetails<'_>,
) -> Result<(u64, PoolServing, AttemptOutcome), PipelineError> {
    let share = share_dir().ok_or(PipelineError::NoShareDir)?;
    let stage_codes: Vec<PipelineStage> = programs.iter().map(|program| program.stage).collect();
    let kind = pipeline_kind(&stage_codes)?;
    // Validate every stage before uploading any file: the host hands these
    // bytes to its own converter, which crashes on anything but a complete
    // container, so a malformed stage must be a named refusal here.
    for program in programs {
        validate_stage_program(program)?;
    }
    let mut uploaded: Vec<(u8, String, &str)> = Vec::with_capacity(programs.len());
    for program in programs {
        let file = upload_host_stage(&share, &program.dxil)?;
        uploaded.push((program.stage.wire_code(), file, program.entry.as_str()));
    }
    let root_signature_file = if details.root_signature.is_empty() {
        None
    } else {
        Some(upload_host_stage(&share, details.root_signature)?)
    };
    let state = host_pipeline_state(programs, details, kind)?;
    // The content key is every input the host hashes into a pipeline; it is
    // computed before the host call so a refused registration can be cached and
    // a repeat of the same content refused locally.
    let libraries: Vec<(PipelineStage, &[u8], &str)> = programs
        .iter()
        .map(|program| (program.stage, program.dxil.as_slice(), program.entry.as_str()))
        .collect();
    let details_key = PipelineDetails {
        state: &state,
        abi: &[],
        root_signature: &[],
    };
    let key = registration_key(&stage_codes, &libraries, Some(&details_key));
    if let Some(reason) = poisoned_pipeline(&key) {
        return Err(PipelineError::Refused(reason));
    }
    let kind_code = match kind {
        PipelineKind::Graphics => crate::native_metal::client::PIPELINE_KIND_GRAPHICS,
        PipelineKind::Mesh => crate::native_metal::client::PIPELINE_KIND_MESH,
        PipelineKind::Compute => crate::native_metal::client::PIPELINE_KIND_COMPUTE,
    };
    let session = crate::native_metal::client::live_pipeline_session().ok_or_else(|| {
        PipelineError::Transport(
            "no live Wire v3 host session to host-translate the pipeline".to_string(),
        )
    })?;
    let wire: Vec<crate::native_metal::client::Vp3Stage<'_>> = uploaded
        .iter()
        .map(|(stage, file, entry)| crate::native_metal::client::Vp3Stage {
            stage: *stage,
            file,
            entry,
        })
        .collect();
    let id = match session.register_pipeline_v3(
        kind_code,
        &wire,
        &state,
        root_signature_file.as_deref(),
    ) {
        Ok(id) => id,
        Err(crate::native_metal::queue::NativeQueueError::Refused(reason)) => {
            let error = PipelineError::HostTranslation(bounded_refusal_reason(&reason));
            poison_pipeline(key, &error.to_string());
            return Err(error);
        }
        Err(error) => return Err(PipelineError::Transport(error.to_string())),
    };

    // The census memory only records that this content is hosted; a later
    // reconnect must still ask the (new) host session again.
    let _ = remember(key, id);
    Ok((id, PoolServing { warm: false }, AttemptOutcome::HostTranslated))
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
        let (compiled, serving) = match translate_with_reflection(program) {
            Ok(translated_stage) => translated_stage,
            // The Win64 Wine guest cannot run the x86_64 macOS converter. The
            // ARM64 host can, so hand it the DXIL through the share directory
            // instead of refusing the PSO (ADR-038). A guest with no share
            // directory keeps the explicit refusal below.
            Err(PipelineError::ConverterUnavailable(_)) if share_dir().is_some() => {
                return register_with_host_translation(programs, details);
            }
            Err(error) => return Err(error),
        };
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
    if let Some(reason) = poisoned_pipeline(&key) {
        // A content key the host already refused is never sent again: the
        // registration error is cached, so a repeated list bind is refused
        // locally instead of re-translating and re-sending it forever.
        return Err(PipelineError::Refused(reason));
    }
    if v1_compute {
        // No reflection was available: keep the original v1 frame. The host
        // binds nothing for this pipeline, which is exactly the old behaviour.
        let (_, metallib, entry) = stages[0];
        let id = register_and_cache_refusal(
            key,
            crate::native_metal::queue::register_pipeline(metallib, entry).map_err(map_queue_error),
        )?;
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
    let id = register_and_cache_refusal(
        key,
        crate::native_metal::queue::register_pipeline_v2(
            kind_code,
            &wire,
            state,
            details.abi,
            details.root_signature,
        )
        .map_err(map_queue_error),
    )?;
    Ok((id, remember_outcome(key, id)))
}

/// Runs one pipeline registration and caches a permanent refusal by content key.
///
/// A transport fault and a missing converter are environmental and deliberately
/// not cached: the next attempt may reach a live host or a working pool. Every
/// other failure is the host's or the shader's answer to these exact bytes, so a
/// repeat registration of the same content is refused locally with the cached
/// text instead of a second host round trip.
#[cfg(any(windows, test))]
fn register_and_cache_refusal(
    key: [u8; 32],
    result: Result<u64, PipelineError>,
) -> Result<u64, PipelineError> {
    match result {
        Ok(id) => Ok(id),
        Err(error) => {
            if !matches!(
                error,
                PipelineError::Transport(_) | PipelineError::ConverterUnavailable(_)
            ) {
                poison_pipeline(key, &error.to_string());
            }
            Err(error)
        }
    }
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

    /// A compute stage set and details whose key a poison test can name exactly.
    fn poisoned_key_fixture() -> (
        [(PipelineStage, &'static [u8], &'static str); 1],
        PipelineDetails<'static>,
        [u8; 32],
    ) {
        let stages = [(PipelineStage::Compute, b"MTLB compute".as_slice(), "main")];
        let details = PipelineDetails {
            state: &[],
            abi: b"VAB1",
            root_signature: b"root",
        };
        let key = registration_key(&[PipelineStage::Compute], &stages, Some(&details));
        (stages, details, key)
    }

    /// The registration epoch advances once per settled registration; the
    /// success counter only moves for a success.
    #[test]
    fn the_registration_epoch_advances_on_every_settled_registration() {
        let _serial = registration_test_lock();
        reset_registration_barrier();
        let before = registration_snapshot();
        assert_eq!(before.epoch, 0);
        assert_eq!(before.successful, 0);

        let registration = begin_pipeline_registration();
        assert_eq!(registrations_in_flight(), 1);
        registration.succeeded();

        let after_success = registration_snapshot();
        assert_eq!(after_success.epoch, before.epoch + 1);
        assert_eq!(after_success.successful, before.successful + 1);
        assert_eq!(after_success.in_flight, 0);

        // A failed registration must also settle the epoch, or a submission
        // waiting on it would block forever even though the pipeline will
        // never exist.
        drop(begin_pipeline_registration());
        let after_failure = registration_snapshot();
        assert_eq!(after_failure.epoch, after_success.epoch + 1);
        assert_eq!(
            after_failure.successful, after_success.successful,
            "a failure is not a successful registration"
        );
        assert_eq!(after_failure.in_flight, 0);
    }

    /// A waiter on the observed epoch does not return until the in-flight
    /// registration settles, and a refused registration releases it too.
    #[test]
    fn a_waiter_on_the_registration_epoch_blocks_until_it_settles() {
        let _serial = registration_test_lock();
        reset_registration_barrier();
        let registration = begin_pipeline_registration();
        let snapshot = registration_snapshot();
        let required = snapshot.epoch + snapshot.in_flight as u64;

        let (sender, receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            wait_for_registration_epoch(required);
            sender.send(()).expect("report the wake");
        });
        assert!(
            receiver
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "the waiter must not pass an unsettled registration"
        );
        drop(registration); // settles as a failure, never a success
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("a settled registration releases the waiter");
        waiter.join().expect("waiter thread");
        assert_eq!(successful_registrations(), 0);
    }

    /// A refused registration content key is refused locally with the cached
    /// text: the same bytes are never translated or sent to the host again.
    #[test]
    fn a_poisoned_pipeline_is_refused_locally_with_the_registration_error() {
        let _serial = registration_test_lock();
        reset_registration_barrier();
        let (stages, details, key) = poisoned_key_fixture();
        let cached = "the host refused the pipeline: pixel stage could not be translated";
        poison_pipeline(key, cached);

        let error = register_translated(&stages, &details)
            .expect_err("a poisoned content key is refused before any host call");
        assert_eq!(error, PipelineError::Refused(cached.to_string()));
        assert!(
            error.to_string().contains("pixel stage could not be translated"),
            "{error}"
        );

        // The cache is keyed by content: different entry points do not inherit
        // the refusal.
        let other = [(PipelineStage::Compute, b"MTLB compute".as_slice(), "other")];
        let other_error = register_translated(&other, &details)
            .expect_err("no live host is configured in this test");
        assert_ne!(other_error, PipelineError::Refused(cached.to_string()));
    }

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

    // ---- ADR-038: guest uploads DXIL stages and asks the host to translate ----

    use std::io::{Read, Write};
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::sync::Mutex as StdMutex;
    use std::thread;
    use std::time::Duration;

    use crate::native_metal::client::{self, LiveConfig};

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    struct TempShare(PathBuf);

    impl TempShare {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "victory-pso-host-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).expect("create share dir");
            Self(path)
        }
    }

    impl Drop for TempShare {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read_frame(stream: &mut TcpStream) -> (u8, Vec<u8>) {
        let mut envelope = [0u8; 5];
        stream.read_exact(&mut envelope).expect("read envelope");
        let len = u32::from_le_bytes(envelope[1..].try_into().expect("four bytes")) as usize;
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).expect("read payload");
        (envelope[0], payload)
    }

    fn write_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) {
        let mut envelope = [0u8; 5];
        envelope[0] = kind;
        envelope[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        stream.write_all(&envelope).expect("write envelope");
        stream.write_all(payload).expect("write payload");
        stream.flush().expect("flush frame");
    }

    /// One authenticated fake host: it authenticates the render peer, then
    /// authenticates the DEDICATED pipeline peer (ADR-041) and answers the VP3
    /// payload on that second connection with a native id, returning the exact
    /// control frame it received.
    fn fake_host(secret: [u8; 32], id: u64) -> (u16, mpsc::Receiver<Vec<u8>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
            .expect("bind fake host");
        let port = listener.local_addr().expect("listener address").port();
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            let (mut render, _) = listener.accept().expect("accept guest");
            let (kind, payload) = read_frame(&mut render);
            assert_eq!(kind, client::KIND_AUTH);
            assert_eq!(payload, secret);
            write_frame(&mut render, client::KIND_ACK, &[0]);

            // The pipeline registration opens its own connection; the render
            // session above stays open and untouched.
            let (mut stream, _) = listener.accept().expect("accept pipeline guest");
            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, client::KIND_AUTH);
            assert_eq!(payload, secret);
            write_frame(&mut stream, client::KIND_ACK, &[0]);

            let (kind, payload) = read_frame(&mut stream);
            assert_eq!(kind, client::KIND_PIPELINE);
            assert_eq!(&payload[..4], &client::PIPELINE_FRAME_MAGIC_V3);
            let mut reply = vec![client::PIPELINE_REPLY_OK];
            reply.extend_from_slice(&id.to_le_bytes());
            write_frame(&mut stream, client::KIND_PIPELINE_REPLY, &reply);
            sender.send(payload).expect("send captured frame");
        });
        (port, receiver, handle)
    }

    /// Windows-edition behaviour: the guest cannot run the macOS converter, so
    /// it uploads the DXIL stages to the share directory and sends one VP3
    /// frame whose exact bytes name those uploads.
    #[test]
    fn a_missing_converter_uploads_stages_and_asks_the_host_to_translate() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let share = TempShare::new();
        let secret = [0x5a; 32];
        let (port, receiver, host) = fake_host(secret, 0x5151);

        // Hermetic fence page: an unmappable path keeps this test from reading a
        // `VICTORY_METAL_SHARE_DIR` another test leaked into the environment.
        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::set_var(
                crate::native_metal::fence_page_guest::FENCE_PAGE_ENV,
                "/nonexistent/victory-test/gpu-fence.v1.page",
            );
        }

        // Production guest configuration: a live authenticated session.
        let config = LiveConfig::resolve(
            Some(&format!("127.0.0.1:{port}")),
            Some(&"5a".repeat(32)),
            None,
            None,
        )
        .expect("parse endpoint")
        .expect("live config");
        let client = config.connect().expect("connect fake host");

        // SAFETY: the test runs with `--test-threads=1`; the lock serialises
        // the environment across this module's tests.
        unsafe {
            std::env::set_var(
                crate::native_metal::fence_page_guest::SHARE_DIR_ENV,
                &share.0,
            );
            std::env::remove_var("VICTORY_IRCONVERTER_WORKER");
            std::env::remove_var("VICTORY_METAL_IRCONVERTER");
        }
        reset_pool();
        reset_census();

        let vertex_program = stage_program(PipelineStage::Vertex, &fixture(PipelineStage::Vertex))
            .expect("the vertex fixture is a complete container");
        let pixel_program = stage_program(PipelineStage::Pixel, &fixture(PipelineStage::Pixel))
            .expect("the pixel fixture is a complete container");
        let vertex = vertex_program.dxil.clone();
        let pixel = pixel_program.dxil.clone();
        let programs = [vertex_program, pixel_program];
        let (id, serving) =
            register_with_details(&programs, &PipelineDetails::default()).expect("host registration");
        assert_eq!(id, 0x5151);
        assert!(!serving.warm);
        assert_eq!(census().host_translated, 1, "recorded as a host translation");

        let payload = receiver.recv_timeout(Duration::from_secs(5)).expect("captured frame");
        drop(client);
        host.join().expect("fake host thread");

        let decoded = client::decode_pipeline_v3_request(&payload).expect("decode VP3 bytes");
        assert_eq!(decoded.kind, client::PIPELINE_KIND_GRAPHICS);
        let expected = [
            (
                client::PIPELINE_STAGE_VERTEX,
                format!("stages/{}.bin", blake3::hash(&vertex).to_hex()),
                "main",
            ),
            (
                client::PIPELINE_STAGE_PIXEL,
                format!("stages/{}.bin", blake3::hash(&pixel).to_hex()),
                "main",
            ),
        ];
        assert_eq!(decoded.stages.len(), 2);
        for (index, (stage, name, entry)) in expected.iter().enumerate() {
            assert_eq!(decoded.stages[index].0, *stage);
            assert_eq!(&decoded.stages[index].1, name);
            assert_eq!(decoded.stages[index].2, *entry);
            let path = share.0.join(name);
            let bytes = std::fs::read(&path).expect("uploaded stage");
            assert_eq!(bytes, if index == 0 { vertex.as_slice() } else { pixel.as_slice() });
        }
        assert!(!decoded.state.is_empty(), "graphics state travels in the frame");

        // Idempotent upload: a second call reuses the same files.
        let before: Vec<_> = std::fs::read_dir(share.0.join("stages"))
            .expect("stages dir")
            .flatten()
            .map(|entry| entry.file_name())
            .collect();
        upload_host_stage(&share.0, &vertex).expect("repeat upload");
        let after: Vec<_> = std::fs::read_dir(share.0.join("stages"))
            .expect("stages dir")
            .flatten()
            .map(|entry| entry.file_name())
            .collect();
        assert_eq!(before.len(), after.len(), "an existing stage is not rewritten");

        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::remove_var(crate::native_metal::fence_page_guest::SHARE_DIR_ENV);
            std::env::remove_var(crate::native_metal::fence_page_guest::FENCE_PAGE_ENV);
        }
    }

    /// With no share directory the guest keeps today's explicit refusal: it can
    /// neither upload DXIL nor pretend a metallib exists.
    #[test]
    fn without_a_share_dir_a_missing_converter_is_still_an_explicit_refusal() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::remove_var(crate::native_metal::fence_page_guest::SHARE_DIR_ENV);
            std::env::remove_var("VICTORY_IRCONVERTER_WORKER");
            std::env::remove_var("VICTORY_METAL_IRCONVERTER");
        }
        reset_pool();
        let programs = [stage_program(PipelineStage::Compute, &fixture(PipelineStage::Compute))
            .expect("the compute fixture is a complete container")];
        let error = register_with_details(&programs, &PipelineDetails::default())
            .expect_err("no converter and no share dir must refuse");
        assert!(
            matches!(error, PipelineError::ConverterUnavailable(_)),
            "unexpected error: {error:?}"
        );
    }

    /// ADR-038 defence in depth: if the Win64 guest still inherits the host's
    /// macOS converter worker, it ignores it. The pool built that way has no
    /// worker, so a cold stage is `ConverterUnavailable` (which selects the VP3
    /// host-translation path) instead of `Translation` from a macOS child that
    /// cannot start. The same path stays acceptable to the macOS host.
    #[test]
    fn a_guest_ignores_an_inherited_macos_converter_worker() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let macos_worker = std::env::current_exe().expect("the test executable exists");
        assert!(
            !macos_worker
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("exe")),
            "the fixture must look like the host's macOS worker: {}",
            macos_worker.display()
        );
        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::set_var(WORKER_ENV, &macos_worker);
        }

        assert_eq!(
            resolve_worker_program(false),
            Some(macos_worker.clone()),
            "the macOS host keeps the launcher's worker"
        );
        assert_eq!(
            resolve_worker_program(true),
            None,
            "the Win64 guest must ignore a macOS worker path"
        );

        let pool = CompilerPool::new(
            victory_metal::compiler_pool::PoolConfig {
                cache_dir: None,
                ..victory_metal::compiler_pool::PoolConfig::default()
            },
            resolve_worker_program(true),
        );
        assert!(
            pool.worker_program().is_none(),
            "the guest started no worker"
        );
        let dxil = b"guest-ignores-a-macos-worker";
        let job = CompileJob {
            id: 0x5eed_c0de,
            stage: PipelineStage::Compute.converter_stage(),
            entry: "main",
            dxil: &dxil[..],
        };
        let error = pool
            .compile_with_reflection(&job)
            .expect_err("a cache-only pool cannot answer a cold job");
        assert!(
            matches!(
                translate_error(error),
                PipelineError::ConverterUnavailable(_)
            ),
            "the environmental case must select host translation"
        );

        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::remove_var(WORKER_ENV);
        }
    }

    #[test]
    fn a_windows_executable_path_is_accepted_by_name() {
        assert!(!guest_can_run_worker(Path::new("/Users/x/victory-irconverter")));
        assert!(!guest_can_run_worker(Path::new(
            "Z:\\Users\\x\\victory-irconverter"
        )));
        assert!(!guest_can_run_worker(Path::new(
            "Z:\\Users\\x\\victory-irconverter.exe"
        )));
        assert!(guest_can_run_worker(Path::new(
            "C:\\tools\\victory-irconverter.exe"
        )));
    }

    // ---- ADR-045 item 3: the converter's stage contract ---------------------

    /// Decodes a whitespace-tolerant hex fixture.
    fn hex_bytes(text: &str) -> Vec<u8> {
        let digits: Vec<u8> = text
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect();
        assert_eq!(digits.len() % 2, 0, "odd hex fixture length");
        digits
            .chunks_exact(2)
            .map(|pair| {
                let hi = (pair[0] as char).to_digit(16).expect("hex digit") as u8;
                let lo = (pair[1] as char).to_digit(16).expect("hex digit") as u8;
                (hi << 4) | lo
            })
            .collect()
    }

    /// The complete synthetic containers the converter is proven against: a
    /// vertex, a pixel and a compute stage compiled from this repository's own
    /// HLSL with the checked-in `dxc`. Nothing here comes from a title.
    /// Vertex: a complete `DXBC` container the converter accepts (stage 1).
    const VS_CONTAINER_HEX: &str = "445842432b75a5e9ffdd7cf6479eedd423d0139901000000c40b0000070000003c0000004c00000088000000f0000000\
        940100007c06000098060000534649300800000000000000000000004953473134000000010000000800000000000000\
        2800000000000000060000000100000000000000010100000000000053565f5665727465784944004f53473160000000\
        02000000080000000000000048000000000000000100000003000000000000000f000000000000000000000054000000\
        00000000000000000300000001000000030c00000000000053565f506f736974696f6e00544558434f4f524400000000\
        505356309c000000340000000100000000000000000000000000000000000000ffffffff010000000102000102000000\
        0000000000000000000000000a000000000000001000000000544558434f4f5244006d61696e00000100000000000000\
        100000000000000000000000010041010100000000000000000000000100440303040000010000000000000001014200\
        030200003300000000000000000000000000000053544154e004000060000100380100004458494c0001000010000000\
        c80400004243c0de210c00002f0100000b82200002000000130000000781239141c80449061032399201840c25050819\
        1e048b628010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818c\
        e183e58a0421460651180000060000001b8ce0ffffffff074002a80d84f0ffffffff0320010000004918000002000000\
        1382604220000000892000000f00000032220809206485041322a484041322e384a19014124c888c0b84844c10302300\
        25008a3902309823408a313343433503500c98193ac281801c180000131472c08774608736688779680372c0870daf50\
        0e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d0\
        06e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0\
        077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c0510000100000000000000\
        0c791020000400000000000000c802010b000000321e981419114c908c092647c604439a121801288632288f12212a85\
        121801288932288422a01d6b088c390079180000650000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7\
        a50604052646cc46268c8ce6266543104c1086618230101b8481d820100405bbb909c2506c180e84982008c00660c340\
        2ccb8680d9300c4a334158a00dc143a22d2ccd8dcb94d59755991c5d195e12d1048148260884b221202608c43241188c\
        0dc2356c580869a22a6ba0080bdb10645ca6acbea0dee6d2e8d2dedc2608043341209a0902e14c10866383707d1b1662\
        e3baca1b3cc2020322544558434f4f52441304e2d9205cd786651003ceaac6601883c122830d421894c1864133036043\
        a14467000034ccd8dec2e8e658a4b9cdd1cd4d100684c65cdad9171b198db9b4b3af393a1675696e74731b943440d460\
        0dd8606803c40dae2a6c6c766d2e6964656e745382a00a199e8b5d99dc5cda9bdb9480684286e76217c6665726372528\
        ea90e1b9cca1859195c935bd9195b14d09904a64782e74797065416e6e6f746174696f6e735382a60e199e8b5d5ad95d\
        12d9145d185dd994e0a94386e752e646279707f596e64637372538832e64782e636f756e746572735302370079180000\
        4c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dce\
        a11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc11\
        0eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033b\
        ccf0147660077b680737688772680737808770908770600776280776f8057678877780875f0887711887729887799881\
        2ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339\
        c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e420\
        0ee7e006f6100ef2c00ee1900fef500ff4000000712000000b00000016300d97ef3cfee20083d83cd4e423b76d02d570\
        f9cee34b93131128353dd4e417b76d000403200d00000000000000004841534814000000000000006c2e0d6a5faa63fa\
        be82e6a999fe1bd94458494c2405000060000100490100004458494c00010000100000000c0500004243c0de210c0000\
        400100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42\
        841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000\
        060000001b8ce0ffffffff074002a80d84f0ffffffff0320010000004918000002000000138260422000000089200000\
        0f00000032220809206485041322a484041322e384a19014124c888c0b84844c1030230025008a3902309823408a3133\
        43433503500c98193ac281801c180000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a30\
        0772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d90\
        0e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160\
        077a300772a007764007439e0000000000000000000000863c05100001000000000000000c7910200004000000000000\
        00c802010b000000321e981419114c908c092647c604439a1218012889622883f2202a85121801288932288422a01d6b\
        088c3900791800004c0000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6\
        266543104c1086618230101b848198200cc506613028d8cd4d100663c38024c40461813604cb04410048b485a5b97199\
        b2fab22a93a32bc34b229a2010c90481503604c40481582608c3b141a0860d0bf14091340d1131551b028bcb94d517d4\
        db5c1a5dda9bdb0481602608443341209c09c2806c10286ec3426099266dc3464c1d11aa22aca1a72729a20902f16c10\
        286ac3327cd92481c10006c314061b044f0c360cd718001b8ac6210300a8c2c666d7e6924656e646372508aa90e1b9d8\
        95c9cda5bdb94d09882664782e76616c7665725302a30e199ecb1c5a1859995cd31b5919db9420a94386e76297567697\
        443645174657362558ea90e1b994b9d1c9e541bda5b9d1cd4d09c800791800004c0000003308801cc4e11c6614013d88\
        433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8\
        433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e\
        3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737\
        808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a1\
        1ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc\
        833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500f\
        f4000000712000000b00000016300d97ef3cfee20083d83cd4e423b76d02d570f9cee34b93131128353dd4e417b76d00\
        0403200d0000000061200000280000001304412c100000000800000034a550048530034037460082200882c118010882\
        20fe0b23006304200882f80723060900826050604655394505d75590e80557452c7ac1950d8a7c4c58e463c2021f1318\
        f88c1824000882011206d2f761c28841028020182061207d5f168c1824000882011206d2f751cc884102802018206120\
        7d9fb58c182400088201120612187cd83162900020080648184860f06505020000000000";

    /// Pixel: a complete `DXBC` container the converter accepts (stage 2).
    const PS_CONTAINER_HEX: &str = "44584243b080005e3971ca36961a47a6a2b4ce16010000005c0a0000070000003c0000004c00000088000000c4000000\
        50010000c4050000e0050000534649300800000000000000000000004953473134000000010000000800000000000000\
        28000000000000000100000003000000000000000f0000000000000053565f506f736974696f6e004f53473134000000\
        01000000080000000000000028000000000000004000000003000000000000000f0000000000000053565f5461726765\
        740000005053563084000000340000000000000000000000000000000000000000000000ffffffff0000000001010001\
        01000000000000000000000000000000010000000000000008000000006d61696e000000010000000000000010000000\
        000000000000000001004403030400000000000000000000010044100300000000000000000000000000000000000000\
        535441546c040000600000001b0100004458494c0001000010000000540400004243c0de210c0000120100000b822000\
        02000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42841032143808184b\
        0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000030000001b88e0ff\
        ffffff0740020000491800000100000013820000892000000e00000032220809206485041322a484041322e384a19014\
        124c888c0b84844c1028230025008a39023098234066008a013343453610900603000000131472c08774608736688779\
        680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e\
        7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a3007\
        72d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c061000\
        0100000000000000648100000b000000321e981019114c908c092647c604439a121801288632288f522804a292288342\
        18012810cab10c820000000079180000550000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a5060405\
        2646cc46268c8ce6266543104c10066182300c1b8481d820100405b8b909c2406c180e84982008c00660c3402ccb8680\
        d9300c4a3341489c0dc143a22d2ccd8dcb94d517d4db5c1a5dda9bdb048148260884b221202608c4324120980d0b214d\
        54650d1671011b028cc994d5175598dc5919dd0481682608433141188c0d02d76d58086ddaaa6bb088cbdb107c1b860c\
        0c800d8512850100b048739ba39b9b200c078db9b4b3af39ba09c2806c20c6800ccac00caab0b1d9b5b9a49195b9d14d\
        09822a64782e76657273696f6e5302a209199e8b5d189b5d99dc94a0a84386e7328716465626d7f44656c63625402a91\
        e1b9d0e5c19505b9b9bdd185d1a5bdb9cd4d099a3a64782e76696577496453746174655382a70e199e4b991b9d5c1ed4\
        5b9a1bdddc94200cba90e1b98cbdd5b9d195c9cd4d09cc00791800004c0000003308801cc4e11c6614013d88433884c3\
        8c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c03\
        3dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41d\
        de211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b68073768877268073780877090\
        8770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11c\
        cca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc82\
        3bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000\
        712000000600000016500d97ef3cbe3439118152d3434d7e71db06403000d20000000000484153481400000000000000\
        a305c4dc465b505c9b96d84be325e5b74458494c74040000600000001d0100004458494c00010000100000005c040000\
        4243c0de210c0000140100000b82200002000000130000000781239141c80449061032399201840c250508191e048b62\
        8010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a\
        0421460651180000030000001b88e0ffffffff0740020000491800000100000013820000892000000e00000032220809\
        206485041322a484041322e384a19014124c888c0b84844c1028230025008a39023098234066008a0133434536109006\
        03000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0\
        077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e610\
        0776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e\
        0000000000000000000000863c0610000100000000000000648100000b000000321e981019114c908c092647c604439a\
        1218012889622883f2202a89322884118002a11ccb2008000000000079180000420000001a034c90460213c48e0c6fec\
        ed4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c10066182300c1b848198200cc406613028c0cd\
        4d108662c38024c40421693604cb04410048b485a5b97199b2fa827a9b4ba34b7b739b2010c80481483604c404815026\
        08c4b261211e2892a66122286043503199b2faa20a933b2ba39b2010cc0461302608c3b141c8b40d0b714198440d1341\
        6d1b026ec36075c086a2713c00a8c2c666d7e6924656e646372508aa90e1b9d895c9cda5bdb94d09882664782e76616c\
        7665725302a30e199ecb1c5a1859995cd31b5919db9420a94386e76297567697443645174657362558ea90e1b994b9d1\
        c9e541bda5b9d1cd4d093c00791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6\
        000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948\
        877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e663089\
        3bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678\
        877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211c\
        c4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70\
        037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000600000016500d97\
        ef3cbe3439118152d3434d7e71db06403000d20061200000130000001304412c100000000400000034a5405402454035\
        46008220887f00002306090082606044c4f31cc18841028020181811f13c433062900020080646443c8f108c18240008\
        82811111cf93040800000000";

    /// Compute: a complete `DXBC` container the converter accepts (stage 8).
    const CS_CONTAINER_HEX: &str = "44584243131f46e559fc8bb24fb844b0e487365c01000000400b0000070000003c0000004c0000005c0000006c000000\
        dc000000200600003c06000053464930080000000000000000000000495347310800000000000000080000004f534731\
        0800000000000000080000005053563068000000340000000000000000000000000000000000000000000000ffffffff\
        050000000000000000000000400000000100000001000000010000000100000018000000080000000000000000000000\
        000000000c0000000000000008000000006d61696e00000000000000535441543c050000600005004f0100004458494c\
        0001000010000000240500004243c0de210c0000460100000b82200002000000130000000781239141c8044906103239\
        9201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e90\
        9122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662\
        f8ffffffff019000491800000300000013826042204c080600000000892000002500000032224809206485049322a484\
        049322e384a19014124c8a8c0b84a44c10482300250014e608c0a00c630c22730408997b86cb9fb08790fc1068868540\
        c19923088a81861923912a0a18688c31c630c4060266328371608770988779700359b88559a00779a88771a0877a9087\
        72200751a8077330877290073ea8077798877438077728077200837470077af003140c7a7304a000131472c087746087\
        36688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320\
        076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320\
        077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e000800000000000000000086\
        3c04100001000000000000000c79142000040000000000000018f23040000c000000000000009005020000000c000000\
        321e981419114c908c092647c604431a253002500c85510865500ea5300350800145406904806281109c01a037030000\
        791800005b0000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce626654310\
        4c1006628230141b8481d820100485b1b909c2606c180e849820401189b3a73aba09c2704c1092678230201b84c1d9b0\
        10ca421003d334cdb32180361040040013844762803641189209c2a04c10866583815015615d138481d92010d98640db\
        2010db86c1c1b80d033175130401d8006c18083000830d41186c18864f0c48b485a5b94d1022688230341b063318860d\
        045106ce196c283e320024346091e63647374722cd8d6e6e8230384ce8caf0bee6e8dee4ca36186a80ac011bb4c15085\
        8dcdaecd258daccc8d6e4a105421c373b12b939b4b7b739b12104dc8f05cecc2d8eccae4a604451d323c9739b430b232\
        b9a637b232b6290152860ccf45ae6ceead4e6eac6c6e4a105522c373a1cb832b0b72737ba30ba34b7b739b9b22746250\
        870ccfa5cc8d4e2e0fea2dcd8d6e6e4a80065dc8f05cc6deeadce8cae4e6a6046d000000791800004c0000003308801c\
        c4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d8843\
        3884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e\
        300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b68\
        0737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5\
        c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c3389443\
        3888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2\
        c00ee1900fef500ff4000000712000000e00000036500d97ef3c7e40150511b19313113e72db26200d97ef3cbe1011c0\
        448440332c840548c3e53b8f3f1d11010ce2e023b76d000403200d0000000000484153481400000000000000d58deaf6\
        1d33caaf2e2a8053a1d91dd64458494cfc040000600005003f0100004458494c0001000010000000e40400004243c0de\
        210c0000360100000b82200002000000130000000781239141c80449061032399201840c250508191e048b6280144502\
        42920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a04294606\
        51180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffffff0190004918000003000000\
        13826042204c080600000000892000002500000032224809206485049322a484049322e384a19014124c8a8c0b84a44c\
        10482300250014e608c0a00c630c22730408997b86cb9fb08790fc1068868540c19923088a81861923912a0a18688c31\
        c630c4060266328371608770988779700359b88559a00779a88771a0877a908772200751a8077330877290073ea80777\
        98877438077728077200837470077af003140c7a7304a000131472c08774608736688779680372c0870daf500e6dd00e\
        7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e93007\
        72a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a007764007\
        6de00e78a0077160077a300772a007764007439e0000000000000000000000863c04100001000000000000000c791420\
        00040000000000000018f23040000c000000000000009005020000000a000000321e981019114c908c092647c604431a\
        2530025012c5501885508001944600281608c1190000000079180000390000001a034c90460213c48e0c6feced4d0c24\
        c625c745c60626c7a50604052646cc46268c8ce6266543104c1006628230141b848198200cc6066130288ccd4d108663\
        c38024c404016a084c10066482902c138421d9200ccf8685581882181ac771a00d41b4810024009820080089b6b034b7\
        0942c44c100665c3700dc30682b01e6c434155c094556163b36b7349232b73a39b120455c8f05ceccae4e6d2dedca604\
        4413323c17bb3036bb32b9298151870ccf650e2d8cac4caee98dac8c6d4a909421c373912b9b7bab931b2b9b9b124875\
        c8f05ccadce8e4f2a0ded2dce8e6a60419000000791800004c0000003308801cc4e11c6614013d88433884c38c428007\
        7978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c\
        7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8\
        211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b6807376887726807378087709087706007\
        76280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4\
        a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033b\
        b0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff400000071200000\
        0e00000036500d97ef3c7e40150511b19313113e72db26200d97ef3cbe1011c0448440332c840548c3e53b8f3f1d1101\
        0ce2e023b76d000403200d0061200000120000001304412c10000000040000003466004aae7403caa01c4a31804c0900\
        230609008260a04c460041ca88810180201810951155604805068c1834000882016319c44005cbb21408000000000000";

    /// One fixture container, by the stage it declares.
    fn fixture(stage: PipelineStage) -> Vec<u8> {
        hex_bytes(match stage {
            PipelineStage::Vertex => VS_CONTAINER_HEX,
            PipelineStage::Pixel => PS_CONTAINER_HEX,
            PipelineStage::Compute => CS_CONTAINER_HEX,
            other => panic!("no synthetic fixture for {other:?}"),
        })
    }

    /// Builds a `DXBC` container from parts (the shader crate's builder is
    /// test-only).
    fn dxbc_container(parts: &[(victory_shader::FourCC, &[u8])]) -> Vec<u8> {
        const HEADER: usize = 0x20;
        let table = HEADER + 4 * parts.len();
        let total = table + parts.iter().map(|(_, data)| 8 + data.len()).sum::<usize>();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        let mut offset = table;
        for (_, data) in parts {
            out.extend_from_slice(&(offset as u32).to_le_bytes());
            offset += 8 + data.len();
        }
        for (fourcc, data) in parts {
            out.extend_from_slice(&fourcc.0);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(data);
        }
        out
    }

    /// A `PSV0` part in the DXC runtime-info shape: a 52-byte size (itself and
    /// the 48-byte info), a stage byte at info[24], the entry-point string
    /// offset, then a sized NUL-separated string table.
    fn psv0_part(stage: u8, entry_offset: u32, table: &[u8]) -> Vec<u8> {
        let mut psv = Vec::with_capacity(64 + table.len());
        psv.extend_from_slice(&52u32.to_le_bytes());
        let mut info = [0u8; 48];
        info[24] = stage;
        psv.extend_from_slice(&info);
        psv.extend_from_slice(&entry_offset.to_le_bytes());
        psv.extend_from_slice(&0u32.to_le_bytes()); // no bound resources
        psv.extend_from_slice(&(table.len() as u32).to_le_bytes());
        psv.extend_from_slice(table);
        psv
    }

    /// The whole container is what the converter consumes, and the entry name
    /// is the one the container's own `PSV0` table exports.
    #[test]
    fn the_whole_container_is_kept_and_its_entry_table_is_used() {
        for stage in [
            PipelineStage::Vertex,
            PipelineStage::Pixel,
            PipelineStage::Compute,
        ] {
            let container = fixture(stage);
            let program = stage_program(stage, &container)
                .unwrap_or_else(|error| panic!("{stage:?}: {error}"));
            assert_eq!(program.entry, "main", "{stage:?}: PSV0 entry");
            assert_eq!(
                program.dxil, container,
                "{stage:?}: the converter gets the complete container, not its DXIL part"
            );
            assert_eq!(
                container_entry_point(&container).as_deref(),
                Some("main"),
                "{stage:?}: the name comes from the container table"
            );
        }

        // A table that names another export is reported as it is, not guessed.
        let table = b"\0vs_main\0";
        let psv = psv0_part(1, 1, table);
        let container = dxbc_container(&[(victory_shader::FourCC::PSV0, &psv)]);
        assert_eq!(
            container_entry_point(&container).as_deref(),
            Some("vs_main")
        );

        // The repository's own emitter writes a 48-byte runtime info with no
        // entry offset: no name is invented out of the binding table.
        let mut emitter = psv.clone();
        emitter[..4].copy_from_slice(&48u32.to_le_bytes());
        let container = dxbc_container(&[(victory_shader::FourCC::PSV0, &emitter)]);
        assert_eq!(container_entry_point(&container), None);
    }

    /// A bare `DXIL` part is the shape that crashes the converter child. It is
    /// a bounded refusal before the pool is asked, naming the container form.
    #[test]
    fn a_bare_dxil_part_is_a_named_refusal_before_any_converter() {
        let container = fixture(PipelineStage::Compute);
        let parsed = victory_shader::Container::parse(&container).expect("the fixture parses");
        let part = parsed
            .part(victory_shader::FourCC::DXIL)
            .expect("the fixture has a DXIL part");
        assert_ne!(
            part.data.get(..4),
            Some(&b"DXBC"[..]),
            "the bare part has no container magic"
        );

        match stage_program(PipelineStage::Compute, part.data) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("complete DXBC container"), "{reason}");
            }
            other => panic!("a bare DXIL part must be refused, saw {other:?}"),
        }

        // The translation path refuses the same bytes before dispatching a job,
        // so the pool can never report it as a worker exit.
        let program = StageProgram {
            stage: PipelineStage::Compute,
            dxil: part.data.to_vec(),
            entry: "main".to_string(),
        };
        let error = validate_stage_program(&program).expect_err("a bare part is invalid");
        assert!(matches!(error, PipelineError::Container(_)), "{error:?}");
        assert_eq!(RefusalClass::of(&error), RefusalClass::Container);
        match translate(&program) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("complete DXBC container"), "{reason}");
            }
            other => panic!("translate must refuse a bare part, saw {other:?}"),
        }
    }

    /// A container whose `Program` part declares another stage is refused with
    /// both names; the stage is never relabelled or reordered.
    #[test]
    fn a_container_of_another_stage_is_refused_by_name() {
        let container = fixture(PipelineStage::Compute);
        match stage_program(PipelineStage::Vertex, &container) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("vertex"), "{reason}");
                assert!(reason.contains("Compute"), "{reason}");
            }
            other => panic!("a compute container as a vertex stage must be refused, saw {other:?}"),
        }
    }

    /// The opt-in census names the size bucket, every part (capped) and the
    /// `DXIL` header's kind, shader model and bitcode offset/size. The exact
    /// line is asserted, so the Boot #20 format is pinned here.
    #[test]
    fn a_synthetic_container_yields_the_expected_census_line() {
        use victory_shader::FourCC;

        let container = dxbc_container(&[
            (FourCC::ISG1, &[0u8; 40]),
            (FourCC::DXIL, &dxil_part_sm(13, 6, 8)),
        ]);
        let (hash, line) = part_census_line(&container).expect("a DXBC container is censused");
        assert_eq!(hash, *blake3::hash(&container).as_bytes());
        let expected = format!(
            "shader container census hash={} size_bucket=<=1KiB parts=2 [ISG1:40 DXIL:32] \
             dxil=kind:13(Mesh) sm:6.8 bitcode_offset:16 bitcode_size:8",
            &blake3::hash(&container).to_hex().as_str()[..16]
        );
        assert_eq!(line, expected);
        assert!(line.is_ascii(), "one printable line: {line}");
        assert!(!line.contains('\n'), "a census line never wraps: {line}");
    }

    /// Thirteen parts produce twelve named parts plus a remainder count; the
    /// line stays bounded instead of listing every part of a large container.
    #[test]
    fn the_census_names_at_most_twelve_parts() {
        use victory_shader::FourCC;

        let bodies: Vec<[u8; 4]> = (0..13).map(|index| [index as u8; 4]).collect();
        let parts: Vec<(FourCC, &[u8])> = bodies
            .iter()
            .map(|body| (FourCC::ISG1, body.as_slice()))
            .collect();
        let container = dxbc_container(&parts);
        let (_, line) = part_census_line(&container).expect("a DXBC container is censused");
        assert!(line.contains("parts=13"), "{line}");
        assert_eq!(
            line.matches("ISG1:").count(),
            MAX_CENSUS_PARTS,
            "the list is capped at {MAX_CENSUS_PARTS} parts: {line}"
        );
        assert!(line.contains("+1 more"), "{line}");
    }

    /// The census is off by default, enabled only by the literal `1`, and a
    /// container's content is censused once even when it is validated again.
    #[test]
    fn the_part_census_is_silent_until_enabled_and_censuses_each_container_once() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let previous = std::env::var_os(PART_CENSUS_ENV);
        reset_part_census();
        let first = dxbc_container(&[(victory_shader::FourCC::DXIL, &dxil_part(5))]);
        let second = dxbc_container(&[(victory_shader::FourCC::DXIL, &dxil_part(1))]);

        // SAFETY: the test holds `ENV_LOCK` and runs with `--test-threads=1`.
        unsafe {
            std::env::remove_var(PART_CENSUS_ENV);
        }
        assert_eq!(note_part_census(&first), None, "the census defaults to off");

        // SAFETY: as above.
        unsafe {
            std::env::set_var(PART_CENSUS_ENV, "0");
        }
        assert_eq!(
            note_part_census(&first),
            None,
            "only the literal `1` enables it"
        );

        // SAFETY: as above.
        unsafe {
            std::env::set_var(PART_CENSUS_ENV, "1");
        }
        assert!(
            note_part_census(&first).is_some(),
            "an enabled census names a new container"
        );
        assert!(
            note_part_census(&second).is_some(),
            "each distinct content is censused"
        );
        assert_eq!(
            note_part_census(&first),
            None,
            "the same content is never censused twice"
        );

        // The validation path itself censuses: a container censused by
        // `stage_program` is already known here.
        reset_part_census();
        let via_validation = dxbc_container(&[(victory_shader::FourCC::DXIL, &dxil_part(1))]);
        stage_program(PipelineStage::Vertex, &via_validation).expect("a valid vertex container");
        assert_eq!(
            note_part_census(&via_validation),
            None,
            "checked_stage_container emits the census"
        );

        // SAFETY: as above; restore whatever the harness started with.
        unsafe {
            match previous {
                Some(value) => std::env::set_var(PART_CENSUS_ENV, value),
                None => std::env::remove_var(PART_CENSUS_ENV),
            }
        }
    }

    /// A stage mismatch explains itself with the container's own numbers: the
    /// program kind and shader model reach `pipeline_first_refusal_reason`,
    /// bounded to [`MAX_REFUSAL_REASON_BYTES`].
    #[test]
    fn a_stage_mismatch_names_the_program_kind_and_shader_model() {
        use victory_shader::FourCC;

        let container = dxbc_container(&[(FourCC::DXIL, &dxil_part_sm(13, 6, 8))]);
        match stage_program(PipelineStage::Vertex, &container) {
            Err(error @ PipelineError::Container(_)) => {
                let reason = error.to_string();
                assert!(reason.contains("vertex"), "{reason}");
                assert!(reason.contains("kind 13"), "{reason}");
                assert!(reason.contains("SM 6.8"), "{reason}");
                assert!(reason.contains("Vertex"), "{reason}");
                assert!(bounded_refusal_reason(&reason).len() <= MAX_REFUSAL_REASON_BYTES);
            }
            other => panic!("a mesh container as a vertex stage must be refused, saw {other:?}"),
        }

        // The Program/PSV0 disagreement keeps the SM too.
        let psv = psv0_part(5, 0, b"\0");
        let container = dxbc_container(&[
            (FourCC::DXIL, &dxil_part_sm(13, 6, 8)),
            (FourCC::PSV0, &psv),
        ]);
        match stage_program(PipelineStage::Mesh, &container) {
            Err(error @ PipelineError::Container(_)) => {
                let reason = error.to_string();
                assert!(reason.contains("SM 6.8"), "{reason}");
                assert!(reason.contains("disagrees"), "{reason}");
                assert!(bounded_refusal_reason(&reason).len() <= MAX_REFUSAL_REASON_BYTES);
            }
            other => panic!("a disagreed PSV0 stage must be refused, saw {other:?}"),
        }
    }

    /// The pass-through test: a 4-byte placeholder and a header-only part are
    /// stubs, while a module at or over the documented floor is real. The
    /// shipped DXC fixtures are real modules.
    #[test]
    fn only_a_real_dxil_module_is_passed_through() {
        use victory_shader::FourCC;

        let stub = dxbc_container(&[(FourCC::DXIL, &[0x42, 0x43, 0xC0, 0xDE])]);
        assert_eq!(declared_bitcode_size(&stub), None, "4 bytes is no program header");
        assert!(!carries_real_dxil(&stub));

        let tiny = dxbc_container(&[(FourCC::DXIL, &dxil_part_sm(1, 6, 0))]);
        assert_eq!(declared_bitcode_size(&tiny), Some(8));
        assert!(!carries_real_dxil(&tiny), "8 bytes is below the floor");
        assert!(carries_usable_dxil(&tiny), "but it is a readable module");

        let floor = vec![0x42; REAL_DXIL_BITCODE_FLOOR as usize];
        let at_floor = dxbc_container(&[(FourCC::DXIL, &dxil_part_with_bitcode(1, 6, 0, &floor))]);
        assert!(carries_real_dxil(&at_floor), "the floor is inclusive");

        for stage in [
            PipelineStage::Vertex,
            PipelineStage::Pixel,
            PipelineStage::Compute,
        ] {
            assert!(
                carries_real_dxil(&fixture(stage)),
                "{stage:?}: the checked-in DXC fixture is a real module"
            );
        }

        // A header that promises more bitcode than the part holds is not one.
        let mut short = dxil_part_with_bitcode(1, 6, 0, &vec![0x42; 600]);
        short.truncate(100);
        let short = dxbc_container(&[(FourCC::DXIL, &short)]);
        assert!(!carries_real_dxil(&short));
        assert!(!carries_usable_dxil(&short));
    }

    /// A console container whose `DXIL` part is a stub selects the locally
    /// translated container, never the stub. The converted entry point is the
    /// translated module's own (`main`, the name `victory_shader`'s emitter
    /// gives its DXIL function), not the raw container's `PSV0` name.
    #[test]
    fn a_stub_console_container_selects_the_translated_module() {
        use victory_shader::FourCC;

        reset_census();
        let stub = dxbc_container(&[
            (FourCC::DXIL, &[0u8; 4]),
            (FourCC::PSV0, &psv0_part(1, 1, b"\0vs_console\0")),
        ]);
        assert_eq!(
            container_entry_point(&stub).as_deref(),
            Some("vs_console"),
            "the raw console container names its own entry"
        );
        let translated = dxbc_container(&[(
            FourCC::DXIL,
            &dxil_part_with_bitcode(1, 6, 6, &vec![0x42; 600]),
        )]);
        // The emitter writes a 48-byte runtime info with no entry table.
        assert_eq!(container_entry_point(&translated), None);

        let program = select_stage_program(PipelineStage::Vertex, &stub, Some(&translated))
            .expect("the translated container is selected");
        assert_eq!(program.dxil, translated, "the 4-byte stub is never uploaded");
        assert_eq!(program.entry, "main", "the translated module's entry, not PSV0's");
        let census = census();
        assert_eq!(census.stages_translated_locally, 1);
        assert_eq!(census.stages_passed_through, 0);
    }

    /// A raw container that already carries a real DXIL module is passed
    /// through byte for byte, even when a translation is available: the Win64
    /// reference path keeps parity.
    #[test]
    fn a_raw_container_with_real_dxil_is_passed_through_unchanged() {
        use victory_shader::FourCC;

        reset_census();
        let raw = fixture(PipelineStage::Vertex);
        let translated = dxbc_container(&[(
            FourCC::DXIL,
            &dxil_part_with_bitcode(1, 6, 6, &vec![0x7e; 600]),
        )]);
        let program = select_stage_program(PipelineStage::Vertex, &raw, Some(&translated))
            .expect("the raw module is selected");
        assert_eq!(program.dxil, raw, "byte parity for the reference path");
        assert_eq!(program.entry, "main", "the fixture's PSV0 entry");
        let census = census();
        assert_eq!(census.stages_passed_through, 1);
        assert_eq!(census.stages_translated_locally, 0);
    }

    /// A stub with no translation is a named `Translation` refusal: no bytes
    /// are invented and the stub never reaches a converter.
    #[test]
    fn a_stub_without_a_translation_is_a_named_refusal() {
        use victory_shader::FourCC;

        reset_census();
        let stub = dxbc_container(&[(FourCC::DXIL, &[0u8; 4])]);
        let error = select_stage_program(PipelineStage::Vertex, &stub, None)
            .expect_err("a stub with no translation must be refused");
        match &error {
            PipelineError::Translation(reason) => {
                assert!(reason.contains("vertex"), "{reason}");
                assert!(reason.contains("did not translate"), "{reason}");
                assert!(reason.contains(&REAL_DXIL_BITCODE_FLOOR.to_string()), "{reason}");
            }
            other => panic!("expected a named translation refusal, saw {other:?}"),
        }
        assert_eq!(RefusalClass::of(&error), RefusalClass::Translation);
        let census = census();
        assert_eq!(
            (census.stages_translated_locally, census.stages_passed_through),
            (0, 0),
            "a refused stage is not counted as selected"
        );
    }

    /// A mesh `OSG1` part with `rows` register-bearing 4-component rows.
    fn mesh_signature(rows: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(rows as u32).to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        for index in 0..rows {
            let mut row = [0u8; 32];
            row[20..24].copy_from_slice(&(index as u32).to_le_bytes());
            row[24] = 0xF;
            out.extend_from_slice(&row);
        }
        out
    }

    /// A mesh `PSV0` part: stage 13, declared vertices and primitives.
    fn mesh_psv0(vertices: u16, primitives: u16) -> Vec<u8> {
        let mut psv = Vec::new();
        psv.extend_from_slice(&48u32.to_le_bytes());
        psv.resize(4 + 48, 0);
        psv[4 + 12..4 + 14].copy_from_slice(&vertices.to_le_bytes());
        psv[4 + 14..4 + 16].copy_from_slice(&primitives.to_le_bytes());
        psv[4 + 24] = 13;
        psv
    }

    /// A synthetic mesh container with `rows` vertex outputs.
    fn mesh_outputs_container(vertices: u16, rows: usize) -> Vec<u8> {
        use victory_shader::FourCC;
        dxbc_container(&[
            (FourCC::OSG1, &mesh_signature(rows)),
            (FourCC::PSG1, &mesh_signature(0)),
            (FourCC::PSV0, &mesh_psv0(vertices, 0)),
        ])
    }

    /// A `[mesh, pixel]` pair whose mesh outputs exceed Metal's 32 KiB budget
    /// selects the linked translation for both stages; an under-budget pair and
    /// a mesh with no pixel stage do not.
    #[test]
    fn an_over_budget_mesh_pair_selects_the_linked_translation() {
        use victory_shader::FourCC;

        // 17 rows × 4 components × 128 vertices × 4 bytes = 34,816 > 32,768.
        let over = mesh_outputs_container(128, 17);
        let under = mesh_outputs_container(8, 1);
        assert!(mesh_outputs_over_budget(&over));
        assert!(!mesh_outputs_over_budget(&under));
        assert!(linked_mesh_pair_needed(true, &over));
        assert!(
            !linked_mesh_pair_needed(false, &over),
            "there is no pixel shader to link with"
        );
        assert!(!linked_mesh_pair_needed(true, &under));

        // Both linked containers become the stage programs.
        let stub = |kind: u8| {
            dxbc_container(&[
                (FourCC::DXIL, &[0u8; 4]),
                (FourCC::PSV0, &psv0_part(kind, 0, b"\0")),
            ])
        };
        let linked_mesh = dxbc_container(&[(
            FourCC::DXIL,
            &dxil_part_with_bitcode(13, 6, 6, &vec![0x42; 600]),
        )]);
        let linked_pixel = dxbc_container(&[(
            FourCC::DXIL,
            &dxil_part_with_bitcode(0, 6, 6, &vec![0x7e; 600]),
        )]);
        let mesh = select_stage_program(PipelineStage::Mesh, &stub(13), Some(&linked_mesh))
            .expect("the linked mesh translation is selected");
        let pixel = select_stage_program(PipelineStage::Pixel, &stub(0), Some(&linked_pixel))
            .expect("the linked pixel translation is selected");
        assert_eq!(mesh.dxil, linked_mesh);
        assert_eq!(pixel.dxil, linked_pixel);
        assert_eq!((mesh.entry.as_str(), pixel.entry.as_str()), ("main", "main"));
    }

    /// A minimal standard-layout `DXIL` part for one container `Program` kind.
    fn dxil_part(kind: u16) -> Vec<u8> {
        dxil_part_sm(kind, 6, 0)
    }

    /// A minimal standard-layout `DXIL` part for one `Program` kind and shader
    /// model.
    fn dxil_part_sm(kind: u16, major: u8, minor: u8) -> Vec<u8> {
        dxil_part_with_bitcode(kind, major, minor, &[0x42, 0x43, 0xC0, 0xDE, 0, 0, 0, 0])
    }

    /// A standard-layout `DXIL` part carrying `bitcode` bytes.
    fn dxil_part_with_bitcode(kind: u16, major: u8, minor: u8, bitcode: &[u8]) -> Vec<u8> {
        let size_dwords = (24 + bitcode.len()).div_ceil(4);
        let mut part = Vec::with_capacity(24 + bitcode.len());
        let version = (u32::from(kind) << 16) | (u32::from(major) << 4) | u32::from(minor);
        part.extend_from_slice(&version.to_le_bytes());
        part.extend_from_slice(&(size_dwords as u32).to_le_bytes());
        part.extend_from_slice(b"DXIL");
        part.extend_from_slice(&0x100u32.to_le_bytes());
        part.extend_from_slice(&16u32.to_le_bytes());
        part.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
        part.extend_from_slice(bitcode);
        part
    }

    /// Rebuilds `container` with a 20-byte digest between the `DXIL` part's
    /// program header and its bitcode.
    ///
    /// `advance_offset` moves the header's bitcode offset past the digest (the
    /// layout the converter translates); leaving it false models a producer
    /// that keeps the older offset, whose bitcode pointer lands on the digest.
    fn container_with_dxil_digest(
        container: &[u8],
        digest: [u8; 20],
        advance_offset: bool,
    ) -> Vec<u8> {
        let parsed = victory_shader::Container::parse(container).expect("the fixture parses");
        let mut parts: Vec<(victory_shader::FourCC, Vec<u8>)> = parsed
            .parts
            .iter()
            .map(|part| (part.fourcc, part.data.to_vec()))
            .collect();
        for (fourcc, body) in &mut parts {
            if *fourcc != victory_shader::FourCC::DXIL {
                continue;
            }
            assert!(body.len() >= 24, "the fixture DXIL part has a program header");
            let offset = u32::from_le_bytes(body[16..20].try_into().expect("4 bytes")) as usize;
            let start = 8 + offset;
            assert!(start <= body.len(), "the fixture bitcode offset is in bounds");
            body.splice(start..start, digest);
            if advance_offset {
                body[16..20].copy_from_slice(&((offset + digest.len()) as u32).to_le_bytes());
            }
            let size_dwords = body.len().div_ceil(4);
            body[4..8].copy_from_slice(&(size_dwords as u32).to_le_bytes());
        }
        let refs: Vec<_> = parts
            .iter()
            .map(|(fourcc, data)| (*fourcc, data.as_slice()))
            .collect();
        dxbc_container(&refs)
    }

    /// Each of the converter's five stages maps to the container `Program`
    /// kind it declares; the container kind is never confused with the
    /// converter stage code (converter 5 is mesh, container kind 5 is compute).
    #[test]
    fn each_stage_maps_to_its_own_program_kind() {
        for (stage, kind) in [
            (PipelineStage::Vertex, 1u16),
            (PipelineStage::Pixel, 0),
            (PipelineStage::Mesh, 13),
            (PipelineStage::Object, 14),
            (PipelineStage::Compute, 5),
        ] {
            let container = dxbc_container(&[(victory_shader::FourCC::DXIL, &dxil_part(kind))]);
            let program = stage_program(stage, &container)
                .unwrap_or_else(|error| panic!("{stage:?} kind {kind}: {error}"));
            assert_eq!(
                program.entry, "main",
                "{stage:?}: a container with no PSV0 table falls back to `main`"
            );
        }

        let container = dxbc_container(&[(victory_shader::FourCC::DXIL, &dxil_part(5))]);
        match stage_program(PipelineStage::Mesh, &container) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("mesh"), "{reason}");
                assert!(reason.contains("Compute"), "{reason}");
            }
            other => panic!("a compute container must not pass as mesh, saw {other:?}"),
        }
    }

    /// A 20-byte digest between the `DXIL` program header and the bitcode is
    /// not a container refusal: the header and `PSV0` still name the stage and
    /// the entry, and the bitcode is the converter's to judge. This is the
    /// layout Boot #18b refused with `bitcode lacks LLVM magic`.
    #[test]
    fn a_digest_before_the_bitcode_is_not_a_container_refusal() {
        for stage in [
            PipelineStage::Vertex,
            PipelineStage::Pixel,
            PipelineStage::Compute,
        ] {
            let with_digest = container_with_dxil_digest(&fixture(stage), [0x5a; 20], true);
            let program = stage_program(stage, &with_digest)
                .unwrap_or_else(|error| panic!("{stage:?} with digest: {error}"));
            assert_eq!(program.entry, "main", "{stage:?}: PSV0 still names the entry");
            assert_eq!(
                program.dxil, with_digest,
                "{stage:?}: the converter still gets the whole container"
            );
            validate_stage_program(&program).unwrap_or_else(|error| panic!("{stage:?}: {error}"));
        }

        // A producer that leaves the header's bitcode offset pointing at the
        // digest used to stop at `bitcode lacks LLVM magic`. The container is
        // structurally sound, so it now reaches the converter, which is the
        // authority on the bitcode and answers a named error in its own child.
        let not_advanced =
            container_with_dxil_digest(&fixture(PipelineStage::Compute), [0xa5; 20], false);
        let program = stage_program(PipelineStage::Compute, &not_advanced)
            .expect("the program header and PSV0 still name the stage");
        validate_stage_program(&program)
            .expect("container validation must not parse the bitcode");
        assert!(!program.dxil.is_empty(), "the container is forwarded intact");

        // The literal "`DXIL` fourcc + part size + 20-byte digest + LLVM
        // bitcode" layout carries no program header at all. The stage comes
        // from `PSV0`, the entry from its table, and the converter judges the
        // bytes; nothing here rejects it for the digest.
        let standard = fixture(PipelineStage::Compute);
        let parsed = victory_shader::Container::parse(&standard).expect("the fixture parses");
        let dxil = parsed
            .part(victory_shader::FourCC::DXIL)
            .expect("the fixture has a DXIL part");
        let offset = u32::from_le_bytes(dxil.data[16..20].try_into().expect("4 bytes")) as usize;
        let size = u32::from_le_bytes(dxil.data[20..24].try_into().expect("4 bytes")) as usize;
        let mut digest_and_bitcode = vec![0x7e; 20];
        digest_and_bitcode.extend_from_slice(&dxil.data[8 + offset..8 + offset + size]);
        let parts: Vec<_> = parsed
            .parts
            .iter()
            .map(|part| {
                let data: &[u8] = if part.fourcc == victory_shader::FourCC::DXIL {
                    &digest_and_bitcode
                } else {
                    part.data
                };
                (part.fourcc, data)
            })
            .collect();
        let headerless = dxbc_container(&parts);
        let program = stage_program(PipelineStage::Compute, &headerless)
            .expect("PSV0 names the compute stage when the DXIL part has no header");
        assert_eq!(program.entry, "main", "the entry still comes from PSV0");
        validate_stage_program(&program)
            .expect("a headerless DXIL part with a PSV0 stage is the converter's to judge");
    }

    /// A truncated part table is a named container refusal, never a forwarded
    /// byte string: the converter only ever sees a table whose parts fit.
    #[test]
    fn a_truncated_part_table_is_a_named_refusal() {
        let mut broken = fixture(PipelineStage::Compute);
        broken[0x1c..0x20].copy_from_slice(&200u32.to_le_bytes());
        broken.truncate(0x40);
        let truncated_len = broken.len() as u32;
        broken[0x18..0x1c].copy_from_slice(&truncated_len.to_le_bytes());
        let error = stage_program(PipelineStage::Compute, &broken)
            .expect_err("a truncated part table must be refused");
        assert!(matches!(error, PipelineError::Container(_)), "{error:?}");
        assert!(!error.to_string().is_empty(), "the reason is named");
        assert_eq!(RefusalClass::of(&error), RefusalClass::Container);
    }

    /// Every container refusal is a named, bounded `Container`/`NoDxil`
    /// reason, so a boot's `pipeline_first_refusal_reason` names the malformed
    /// form instead of a converter worker exit.
    #[test]
    fn every_container_refusal_names_its_reason() {
        use victory_shader::FourCC;

        // No bytes at all is "no DXIL", not a converter job.
        assert!(matches!(
            stage_program(PipelineStage::Compute, b""),
            Err(PipelineError::NoDxil(_))
        ));

        // A container over the converter's limit is refused by size.
        let oversized = vec![0u8; MAX_STAGE_CONTAINER_BYTES + 1];
        match stage_program(PipelineStage::Compute, &oversized) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("limit"), "{reason}");
            }
            other => panic!("an oversized container must be refused, saw {other:?}"),
        }

        // Neither a non-container nor a corrupt container reaches the pool.
        for bytes in [
            b"not a container".as_slice(),
            b"DXBC and then nothing at all".as_slice(),
        ] {
            match stage_program(PipelineStage::Compute, bytes) {
                Err(PipelineError::Container(reason)) => {
                    assert!(!reason.is_empty(), "the reason is named");
                }
                other => panic!("{bytes:?} must be refused, saw {other:?}"),
            }
        }

        // A `DXIL` part the program-header parser rejects is malformed.
        let container = dxbc_container(&[(FourCC::DXIL, &[0u8; 32])]);
        match stage_program(PipelineStage::Compute, &container) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("malformed"), "{reason}");
            }
            other => panic!("a malformed DXIL part must be refused, saw {other:?}"),
        }

        // The entry point a caller supplies is bounded and NUL-free.
        let mut program =
            stage_program(PipelineStage::Compute, &fixture(PipelineStage::Compute))
                .expect("the compute fixture is a complete container");
        for (entry, needle) in [
            (String::new(), "no entry point"),
            ("a".repeat(MAX_ENTRY_POINT_NAME_BYTES + 1), "limit"),
            ("a\0b".to_string(), "NUL"),
        ] {
            program.entry = entry;
            match validate_stage_program(&program) {
                Err(error @ PipelineError::Container(_)) => {
                    assert!(error.to_string().contains(needle), "{error}");
                    assert_eq!(RefusalClass::of(&error), RefusalClass::Container);
                    assert!(
                        bounded_refusal_reason(&error.to_string()).len() <= MAX_REFUSAL_REASON_BYTES
                    );
                }
                other => panic!("entry {needle:?} must be refused, saw {other:?}"),
            }
        }
    }

    /// The `Program` part and the container's `PSV0` stage byte must agree;
    /// a container whose two tables disagree is refused, not trusted.
    #[test]
    fn a_psv0_stage_byte_that_disagrees_with_the_program_is_refused() {
        let mut container = fixture(PipelineStage::Compute);
        let at = container
            .windows(4)
            .position(|window| window == b"PSV0")
            .expect("the fixture has a PSV0 part")
            + 8;
        assert_eq!(container[at + 28], 5, "the fixture is a compute PSV0");
        container[at + 28] = 1;
        match stage_program(PipelineStage::Compute, &container) {
            Err(PipelineError::Container(reason)) => {
                assert!(reason.contains("PSV0"), "{reason}");
            }
            other => panic!("a disagreed PSV0 stage must be refused, saw {other:?}"),
        }
    }

    /// The hermetic contract proof: a vertex+pixel set and a compute stage,
    /// reachable exactly as `pipeline.rs` builds them, translate through the
    /// real `--stdio` converter pool and answer a non-empty metallib and the
    /// converter's own reflection.
    #[test]
    fn complete_synthetic_containers_translate_through_the_pool() {
        let Some(worker) = converter_worker_for_test() else {
            eprintln!(
                "skipped: no x86_64 victory-irconverter (build it with `cargo build -p \
                 victory-metal --bin victory-irconverter --target x86_64-apple-darwin`)"
            );
            return;
        };
        let Some(converter) = converter_library_for_test() else {
            eprintln!("skipped: no libmetalirconverter.dylib (VICTORY_METAL_IRCONVERTER)");
            return;
        };
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let cache = TempShare::new();
        // SAFETY: the test holds `ENV_LOCK`; every other test that reads these
        // variables takes the same lock.
        unsafe {
            std::env::set_var("VICTORY_IRCONVERTER_WORKER", &worker);
            std::env::set_var("VICTORY_METAL_IRCONVERTER", &converter);
            std::env::set_var("VICTORY_METAL_SHADER_CACHE", &cache.0);
        }
        reset_pool();

        // The two stage sets `pipeline.rs` builds: a graphics PSO (vertex +
        // pixel) and a compute PSO. Each stage is a separate converter job, as
        // `register_with_details` dispatches them.
        for (set, shader_types) in [
            (
                &[PipelineStage::Vertex, PipelineStage::Pixel][..],
                &["Vertex", "Fragment"][..],
            ),
            (&[PipelineStage::Compute][..], &["Compute"][..]),
        ] {
            for (stage, shader_type) in set.iter().copied().zip(shader_types) {
                let program = stage_program(stage, &fixture(stage))
                    .unwrap_or_else(|error| panic!("{stage:?}: {error}"));
                let (compiled, serving) = translate_with_reflection(&program)
                    .unwrap_or_else(|error| panic!("{stage:?} through the pool: {error}"));
                assert!(
                    compiled.metallib.len() > 4 && compiled.metallib.starts_with(b"MTLB"),
                    "{stage:?}: the converter answered a serialized metallib"
                );
                assert!(
                    !serving.warm,
                    "{stage:?}: a cold temp cache never serves a translation warm"
                );
                let reflection =
                    std::str::from_utf8(&compiled.reflection).expect("reflection is UTF-8 JSON");
                assert!(
                    reflection.contains(&format!("\"ShaderType\":\"{shader_type}\"")),
                    "{stage:?}: reflection reports {shader_type}: {reflection}"
                );
                assert!(
                    reflection.contains("\"EntryPoint\":\"main\""),
                    "{stage:?}: reflection names the container's entry: {reflection}"
                );
            }
        }

        // The work really ran in workers of the pool, not in the cache.
        let stats = pool_stats();
        assert_eq!(stats.worker_jobs, 3, "three cold stages ran in workers");
        assert_eq!(stats.errors, 0, "no worker failed");

        // SAFETY: the test holds `ENV_LOCK`.
        unsafe {
            std::env::remove_var("VICTORY_IRCONVERTER_WORKER");
            std::env::remove_var("VICTORY_METAL_IRCONVERTER");
            std::env::remove_var("VICTORY_METAL_SHADER_CACHE");
        }
        reset_pool();
    }

    /// Both accepted `DXIL` layouts translate through the real `--stdio` pool:
    /// a standard container and one with a 20-byte digest before the bitcode.
    /// The second is the layout Boot #18b refused before any job was dispatched.
    #[test]
    fn digest_and_standard_containers_translate_through_the_pool() {
        let Some(worker) = converter_worker_for_test() else {
            eprintln!(
                "skipped: no x86_64 victory-irconverter (build it with `cargo build -p \
                 victory-metal --bin victory-irconverter --target x86_64-apple-darwin`)"
            );
            return;
        };
        let Some(converter) = converter_library_for_test() else {
            eprintln!("skipped: no libmetalirconverter.dylib (VICTORY_METAL_IRCONVERTER)");
            return;
        };
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let cache = TempShare::new();
        // SAFETY: the test holds `ENV_LOCK`; every other test that reads these
        // variables takes the same lock.
        unsafe {
            std::env::set_var("VICTORY_IRCONVERTER_WORKER", &worker);
            std::env::set_var("VICTORY_METAL_IRCONVERTER", &converter);
            std::env::set_var("VICTORY_METAL_SHADER_CACHE", &cache.0);
        }
        reset_pool();

        let mut jobs = 0;
        for (stage, shader_type) in [
            (PipelineStage::Vertex, "Vertex"),
            (PipelineStage::Compute, "Compute"),
        ] {
            for (layout, container) in [
                ("standard", fixture(stage)),
                (
                    "digest",
                    container_with_dxil_digest(&fixture(stage), [0x33; 20], true),
                ),
            ] {
                let program = stage_program(stage, &container)
                    .unwrap_or_else(|error| panic!("{stage:?}/{layout}: {error}"));
                let (compiled, serving) = translate_with_reflection(&program)
                    .unwrap_or_else(|error| panic!("{stage:?}/{layout} through the pool: {error}"));
                assert!(
                    compiled.metallib.len() > 4 && compiled.metallib.starts_with(b"MTLB"),
                    "{stage:?}/{layout}: the converter answered a serialized metallib"
                );
                assert!(!serving.warm, "{stage:?}/{layout}: a cold cache never serves warm");
                let reflection =
                    std::str::from_utf8(&compiled.reflection).expect("reflection is UTF-8 JSON");
                assert!(
                    reflection.contains(&format!("\"ShaderType\":\"{shader_type}\"")),
                    "{stage:?}/{layout}: reflection reports {shader_type}: {reflection}"
                );
                assert!(
                    reflection.contains("\"EntryPoint\":\"main\""),
                    "{stage:?}/{layout}: reflection names the entry: {reflection}"
                );
                jobs += 1;
            }
        }

        let stats = pool_stats();
        assert_eq!(stats.worker_jobs, jobs, "each cold layout ran in a worker");
        assert_eq!(stats.errors, 0, "no worker failed");

        // SAFETY: the test holds `ENV_LOCK`.
        unsafe {
            std::env::remove_var("VICTORY_IRCONVERTER_WORKER");
            std::env::remove_var("VICTORY_METAL_IRCONVERTER");
            std::env::remove_var("VICTORY_METAL_SHADER_CACHE");
        }
        reset_pool();
    }

    /// The x86_64 Rosetta worker built from this worktree, when it exists.
    fn converter_worker_for_test() -> Option<PathBuf> {
        std::env::var_os("VICTORY_IRCONVERTER_WORKER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(Path::parent)?
                    .join("target/x86_64-apple-darwin/debug/victory-irconverter");
                path.is_file().then_some(path)
            })
    }

    /// Apple's x86_64 converter library, from the documented search order.
    fn converter_library_for_test() -> Option<PathBuf> {
        std::env::var_os("VICTORY_METAL_IRCONVERTER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                std::env::var_os("D3DMETAL_RES")
                    .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                    .filter(|path| path.is_file())
            })
            .or_else(|| {
                std::env::var_os("HOME").map(|home| {
                    PathBuf::from(home).join(
                        "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                         D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
                    )
                })
            })
            .filter(|path| path.is_file())
    }

}
