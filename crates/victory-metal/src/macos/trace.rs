//! ADR-053: exact host encode counters and the opt-in draw / offscreen-RT
//! X-ray tracer.
//!
//! The counters are always on and are cheap (one relaxed atomic per encoded
//! command); they answer "did a real draw or dispatch reach Metal at all?".
//! The tracer is off unless `VICTORY_METAL_DRAW_TRACE` is a truthy value, and
//! it is bounded: at most [`MAX_TRACED_WORK`] draws/dispatches and
//! [`MAX_TRACED_PASSES`] render passes are described per host process, so a
//! long boot never grows its log without bound.
//!
//! The trace never records shader bytes, resource contents or title paths: it
//! describes the encoder call (kind, pipeline id, counts, viewport, scissor,
//! argument-buffer shape), the attachment geometry and format, whether that
//! attachment is the registered swapchain surface, and a bounded pixel summary
//! of the first traced offscreen render target. A pass is sampled only when its
//! render segment encoded at least one draw/dispatch, so a clear-only pass is
//! never reported as drawn output. [`report`] folds the bounded counts into the
//! `draw_trace` object `telemetry.json` carries.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Environment flag that arms the tracer. Off when unset.
pub const DRAW_TRACE_ENV: &str = "VICTORY_METAL_DRAW_TRACE";

/// Draws/dispatches the tracer describes per process.
pub const MAX_TRACED_WORK: usize = 10;

/// Render passes the tracer samples per process.
pub const MAX_TRACED_PASSES: u32 = 10;

/// Edge, in texels, of the bounded offscreen-RT pixel sample.
pub const RT_SAMPLE_EDGE: u32 = 64;

/// Environment flag that arms the tracer, read once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| enabled_from(std::env::var(DRAW_TRACE_ENV).ok().as_deref()))
}

/// Pure form of [`enabled`]: `1`, `true`, `on` or `yes` arm the tracer.
pub fn enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

/// The exact encoded-work counters of one host process (ADR-053).
///
/// They are atomics so a read-only planner path can count without taking the
/// device state mutex again; ordering is relaxed because the counters are
/// monotonic diagnostics, never synchronization.
#[derive(Default)]
pub struct EncodeCounters {
    /// Non-indexed draw calls encoded (including instanced ones).
    pub draws: AtomicU64,
    /// Indexed draw calls encoded (including instanced ones).
    pub draws_indexed: AtomicU64,
    /// Direct compute dispatches encoded.
    pub dispatches: AtomicU64,
    /// Mesh dispatches encoded (`dispatchThreadgroups` on the mesh stage).
    pub mesh_dispatches: AtomicU64,
    /// Indirect draw/dispatch calls encoded (one per GPU-driven command).
    pub indirect: AtomicU64,
    /// Render passes opened (a clear inside a pass opens a continuation).
    pub render_passes: AtomicU64,
    /// Explicit clear commands folded into pass load actions.
    pub clear_ops: AtomicU64,
}

impl EncodeCounters {
    /// Records one non-indexed draw.
    pub fn bump_draw(&self) {
        self.draws.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one indexed draw.
    pub fn bump_draw_indexed(&self) {
        self.draws_indexed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one direct compute dispatch.
    pub fn bump_dispatch(&self) {
        self.dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one mesh dispatch.
    pub fn bump_mesh_dispatch(&self) {
        self.mesh_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one indirect draw/dispatch call.
    pub fn bump_indirect(&self) {
        self.indirect.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one opened render pass.
    pub fn bump_render_pass(&self) {
        self.render_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one explicit clear folded into a load action.
    pub fn bump_clear(&self) {
        self.clear_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// A consistent-enough copy of every counter.
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            draws: self.draws.load(Ordering::Relaxed),
            draws_indexed: self.draws_indexed.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            mesh_dispatches: self.mesh_dispatches.load(Ordering::Relaxed),
            indirect: self.indirect.load(Ordering::Relaxed),
            render_passes: self.render_passes.load(Ordering::Relaxed),
            clear_ops: self.clear_ops.load(Ordering::Relaxed),
        }
    }
}

/// Plain copy of [`EncodeCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Non-indexed draws encoded.
    pub draws: u64,
    /// Indexed draws encoded.
    pub draws_indexed: u64,
    /// Direct compute dispatches encoded.
    pub dispatches: u64,
    /// Mesh dispatches encoded.
    pub mesh_dispatches: u64,
    /// Indirect draw/dispatch calls encoded.
    pub indirect: u64,
    /// Render passes opened.
    pub render_passes: u64,
    /// Clears folded into load actions.
    pub clear_ops: u64,
}

impl CounterSnapshot {
    /// The seven counters as `telemetry.json` members (no trailing comma).
    pub fn json_fields(&self) -> String {
        format!(
            "\"draws_encoded\": {},\n  \"draw_indexed_encoded\": {},\n  \
             \"dispatches_encoded\": {},\n  \"mesh_dispatches_encoded\": {},\n  \
             \"indirect_encoded\": {},\n  \"render_passes_encoded\": {},\n  \
             \"clear_ops_encoded\": {}",
            self.draws,
            self.draws_indexed,
            self.dispatches,
            self.mesh_dispatches,
            self.indirect,
            self.render_passes,
            self.clear_ops
        )
    }

    /// Total draw-ish calls (draws, indexed draws, mesh dispatches).
    pub fn total_draws(&self) -> u64 {
        self.draws
            .saturating_add(self.draws_indexed)
            .saturating_add(self.mesh_dispatches)
    }
}

/// Maximum distinct keys the descriptor-table refusal census retains.
///
/// A boot refuses one class at a time; sixteen distinct `(pipeline, set,
/// declared tables, plan)` shapes are enough to attribute every real class
/// without letting a hostile or pathological title grow host memory. Keys past
/// the cap are counted in [`DescriptorTableRefusalCensus::omitted`].
pub const MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS: usize = 16;

/// One bounded key of the descriptor-table census.
///
/// It names only counts and fixed plan-kind/outcome words: no shader bytes,
/// resource contents, register values or title identifiers appear here. The
/// reflected/stage/map fields are what makes the cause of a miss visible in a
/// single boot: the translated root signature never declares a descriptor
/// table (tables are lowered to directly indexed heaps), so `map_tables` and
/// `reflected_tables` both reading 0 identifies the plan-less heap-lowered
/// class without a second run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTableRefusalKey {
    /// Host pipeline id the refused table was selected with.
    pub pipeline_id: u64,
    /// Wire `DescriptorTableV3.set` the submission carried.
    pub wire_set: u16,
    /// Descriptor-table parameters the selected pipeline declares (the larger
    /// of its authoritative map and its widest stage; the pre-existing field).
    pub declared_tables: u16,
    /// Descriptor-table parameters converter reflection reports, summed over
    /// every stage layout of the selected pipeline.
    pub reflected_tables: u16,
    /// Bit mask of the selected pipeline's stage layouts: bit 0 vertex,
    /// bit 1 fragment, bit 2 mesh, bit 3 object, bit 4 compute.
    pub stage_mask: u8,
    /// Descriptor tables the authoritative translated root-parameter map
    /// declares (`0` when the pipeline was registered without one).
    pub map_tables: u16,
    /// Whether the submission carried a root-argument plan at all.
    pub had_plan: bool,
    /// The plan's kind for this set (`none` without an entry).
    pub plan_kind: &'static str,
    /// Fixed outcome word: [`OUTCOME_REFUSED`], [`OUTCOME_STALE`] or
    /// [`OUTCOME_INERT`].
    pub outcome: &'static str,
}

/// Outcome word: the table was refused and the batch aborted.
pub const OUTCOME_REFUSED: &str = "refused";

/// Outcome word: a plan named a table the bound pipeline does not declare.
pub const OUTCOME_STALE: &str = "stale";

/// Outcome word: a plan-less set on a zero-table signature was accepted as
/// inert because the bound pipeline cannot address a table at any root
/// parameter.
pub const OUTCOME_INERT: &str = "inert";

/// Outcome word: the table resolved to a declared parameter, but its wire kind
/// cannot fill the descriptor shape that parameter's reflected range kind
/// requires.
pub const OUTCOME_KIND_MISMATCH: &str = "kind-mismatch";

/// One bounded key of the descriptor-table *kind mismatch* census.
///
/// It names only fixed kind words and counts: the reflected table kind from the
/// resolved root parameter, the wire `DescriptorTableV3.kind`, and the entry
/// variant shape (`buffer`, `texture`, `sampler`, `mixed` or `empty`). No
/// handles, register values, shader bytes or resource contents appear here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTableKindMismatchKey {
    /// Host pipeline id the refused table was selected with.
    pub pipeline_id: u64,
    /// Wire `DescriptorTableV3.set` the submission carried.
    pub wire_set: u16,
    /// Reflected range kind of the resolved root parameter
    /// (`ConstantBufferView`/`ShaderResourceView`/`UnorderedAccessView`/
    /// `Texture`/`Sampler`/`Unspecified`).
    pub reflected_kind: &'static str,
    /// Wire table kind (`Cbv`/`Srv`/`Uav`/`Sampler`).
    pub wire_kind: &'static str,
    /// Entry variant shape (`buffer`/`texture`/`sampler`/`mixed`/`empty`).
    pub entry_shape: &'static str,
}

/// One counted kind-mismatch key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTableKindMismatchCount {
    /// The bounded key.
    pub key: DescriptorTableKindMismatchKey,
    /// How many refusals the key describes in this process.
    pub count: u64,
}

/// One counted key of the descriptor-table refusal census.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTableRefusalCount {
    /// The bounded key.
    pub key: DescriptorTableRefusalKey,
    /// How many refusals the key describes in this process.
    pub count: u64,
}

/// Bounded per-process descriptor-table census.
///
/// It is the one host-side answer to "which descriptor-table set did not
/// resolve, did the guest send a plan for it, and what does the selected
/// pipeline's translated/reflected signature declare?". Reading it never arms
/// the draw tracer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescriptorTableRefusalCensus {
    /// Distinct keys retained across both classes
    /// (`<= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`).
    pub distinct_keys: u64,
    /// Events whose key did not fit the cap.
    pub omitted: u64,
    /// Retained resolution keys in first-observed order.
    pub entries: Vec<DescriptorTableRefusalCount>,
    /// Retained kind-mismatch keys in first-observed order.
    pub kind_mismatches: Vec<DescriptorTableKindMismatchCount>,
}

impl DescriptorTableRefusalCensus {
    /// Whether no descriptor table was refused or accepted as inert.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.kind_mismatches.is_empty()
    }

    /// The census as the `descriptor_table_refusal_census` telemetry member.
    ///
    /// The caller appends this after a comma; it is exactly
    /// `"descriptor_table_refusal_census": <json_value>`.
    pub fn json_fields(&self) -> String {
        format!(
            "\"descriptor_table_refusal_census\": {}",
            self.json_value()
        )
    }

    /// The census as the `descriptor_table_refusal_census` telemetry object.
    ///
    /// Field names, in emitted order: `distinct_keys`, `omitted`, `keys`,
    /// `kind_mismatch_keys`; each resolution key carries `pipeline_id`,
    /// `wire_set`, `declared_tables`, `reflected_tables`, `stage_mask`,
    /// `map_tables`, `had_plan`, `plan_kind`, `outcome` and `count`. Each
    /// kind-mismatch key carries `pipeline_id`, `wire_set`, `reflected_kind`,
    /// `wire_kind`, `entry_shape`, `outcome` (`"kind-mismatch"`) and `count`.
    pub fn json_value(&self) -> String {
        let keys = self
            .entries
            .iter()
            .map(|entry| {
                format!(
                    "{{\"pipeline_id\": {}, \"wire_set\": {}, \"declared_tables\": {}, \
                     \"reflected_tables\": {}, \"stage_mask\": {}, \"map_tables\": {}, \
                     \"had_plan\": {}, \"plan_kind\": \"{}\", \"outcome\": \"{}\", \
                     \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.declared_tables,
                    entry.key.reflected_tables,
                    entry.key.stage_mask,
                    entry.key.map_tables,
                    entry.key.had_plan,
                    entry.key.plan_kind,
                    entry.key.outcome,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let kind_mismatches = self
            .kind_mismatches
            .iter()
            .map(|entry| {
                format!(
                    "{{\"pipeline_id\": {}, \"wire_set\": {}, \"reflected_kind\": \"{}\", \
                     \"wire_kind\": \"{}\", \"entry_shape\": \"{}\", \"outcome\": \"{}\", \
                     \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.reflected_kind,
                    entry.key.wire_kind,
                    entry.key.entry_shape,
                    OUTCOME_KIND_MISMATCH,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\"distinct_keys\": {}, \"omitted\": {}, \"keys\": [{}], \
             \"kind_mismatch_keys\": [{}]}}",
            self.distinct_keys, self.omitted, keys, kind_mismatches
        )
    }
}

#[derive(Default)]
struct DescriptorTableRefusalLog {
    entries: Vec<(DescriptorTableRefusalKey, u64)>,
    kind_mismatches: Vec<(DescriptorTableKindMismatchKey, u64)>,
    omitted: u64,
}

impl DescriptorTableRefusalLog {
    fn retained(&self) -> usize {
        self.entries.len() + self.kind_mismatches.len()
    }
}

static DESCRIPTOR_TABLE_REFUSALS: OnceLock<Mutex<DescriptorTableRefusalLog>> = OnceLock::new();

fn descriptor_table_refusals() -> &'static Mutex<DescriptorTableRefusalLog> {
    DESCRIPTOR_TABLE_REFUSALS.get_or_init(|| Mutex::new(DescriptorTableRefusalLog::default()))
}

/// Counts one descriptor-table census event under its bounded key.
///
/// Refusals, stale-set refusals and accepted inert sets all land here so a
/// single boot can attribute every class; `key.outcome` says which. The hot
/// encode path allocates nothing on an existing key. The map is capped at
/// [`MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`]; keys past the cap only bump
/// `omitted`.
pub fn note_descriptor_table_refusal(key: DescriptorTableRefusalKey) {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, count)) = log.entries.iter_mut().find(|(known, _)| *known == key) {
        *count = count.saturating_add(1);
        return;
    }
    if log.retained() >= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS {
        log.omitted = log.omitted.saturating_add(1);
        return;
    }
    log.entries.push((key, 1));
}

/// Counts one descriptor-table kind-mismatch event under its bounded key.
///
/// The class is distinct from a resolution refusal: the table already named a
/// declared parameter, so `abi_refusals` must not move. It shares the one
/// [`MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`] cap with the resolution keys; keys past
/// the cap only bump `omitted`.
pub fn note_descriptor_table_kind_mismatch(key: DescriptorTableKindMismatchKey) {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, count)) = log
        .kind_mismatches
        .iter_mut()
        .find(|(known, _)| *known == key)
    {
        *count = count.saturating_add(1);
        return;
    }
    if log.retained() >= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS {
        log.omitted = log.omitted.saturating_add(1);
        return;
    }
    log.kind_mismatches.push((key, 1));
}

/// A copy of the process-wide descriptor-table refusal census.
pub fn descriptor_table_refusal_census() -> DescriptorTableRefusalCensus {
    let log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    DescriptorTableRefusalCensus {
        distinct_keys: log.retained() as u64,
        omitted: log.omitted,
        entries: log
            .entries
            .iter()
            .map(|(key, count)| DescriptorTableRefusalCount {
                key: *key,
                count: *count,
            })
            .collect(),
        kind_mismatches: log
            .kind_mismatches
            .iter()
            .map(|(key, count)| DescriptorTableKindMismatchCount {
                key: *key,
                count: *count,
            })
            .collect(),
    }
}

/// Clears the process-wide descriptor-table refusal census.
///
/// Test-only: a proof that runs several refusal shapes in one process starts
/// each case from a clean census. Never called by the host.
#[doc(hidden)]
pub fn reset_descriptor_table_refusal_census() {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *log = DescriptorTableRefusalLog::default();
}

/// Maximum bytes kept in the bounded [`TraceReport::failure`] string.
pub const MAX_FAILURE_BYTES: usize = 96;

/// The tracer's answer for `telemetry.json` and the Boot #23 gate.
///
/// It is always present as an object, even when the trace is off: `enabled`
/// distinguishes "never armed" from "armed and saw nothing". Every field is a
/// count or a bounded string; no resource contents or title paths appear here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceReport {
    /// Whether `VICTORY_METAL_DRAW_TRACE` armed the tracer in this process.
    pub enabled: bool,
    /// Draws/dispatches described by the tracer (bounded by
    /// [`MAX_TRACED_WORK`]).
    pub traced_work: u64,
    /// Render passes whose pixels were sampled (bounded by
    /// [`MAX_TRACED_PASSES`]).
    pub traced_passes: u64,
    /// `nonzero_bytes` of the last sampled offscreen attachment.
    pub last_offscreen_nonzero_bytes: u64,
    /// `distinct_values` of the last sampled offscreen attachment.
    pub last_offscreen_distinct_values: u32,
    /// First named tracer failure, truncated to [`MAX_FAILURE_BYTES`].
    pub failure: Option<String>,
}

/// Attachment and pass context shared between [`TraceState::next_pass`] and
/// [`sample_grid`], which the host calls through separate device functions.
#[derive(Debug, Clone, Copy)]
struct SampleContext {
    pass: u32,
    draws: u32,
    target: Option<TraceTarget>,
}

/// Process-wide tracer state read by the host for telemetry and by the GPU
/// proof. Only touched while [`enabled`] is true.
#[derive(Default)]
struct TraceLog {
    records_described: u64,
    passes_sampled: u64,
    samples_recorded: u64,
    last_offscreen_nonzero_bytes: u64,
    last_offscreen_distinct_values: u32,
    failure: Option<String>,
    last_record: Option<DrawRecord>,
    sample_context: Option<SampleContext>,
    last_sample: Option<PixelStats>,
}

static TRACE_LOG: OnceLock<Mutex<TraceLog>> = OnceLock::new();

fn trace_log() -> &'static Mutex<TraceLog> {
    TRACE_LOG.get_or_init(|| Mutex::new(TraceLog::default()))
}

fn with_log<R>(f: impl FnOnce(&mut TraceLog) -> R) -> R {
    let mut guard = trace_log()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    f(&mut guard)
}

/// Process-wide ABI binding totals: parameters a title wrote vs. the zero
/// default, summed over every recorded flush. Always on (cheap relaxed
/// atomics), independent of the opt-in draw tracer, so `telemetry.json` can
/// answer G6 on a boot that did not arm the trace.
static ABI_BOUND_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static ABI_UNBOUND_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static ABI_BOUND_BYTES: AtomicU64 = AtomicU64::new(0);
static ABI_UNBOUND_BYTES: AtomicU64 = AtomicU64::new(0);
static ABI_LAST_BOUND_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static ABI_LAST_UNBOUND_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static ABI_LAST_BOUND_BYTES: AtomicU64 = AtomicU64::new(0);
static ABI_LAST_KIND_BITS: AtomicU64 = AtomicU64::new(0);

/// Exact process-wide ABI binding totals and the last draw's shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbiBindingTelemetry {
    /// Title-written parameters across every recorded flush.
    pub bound_parameters: u64,
    /// Parameters filled with the zero default across every recorded flush.
    pub unbound_parameters: u64,
    /// Bytes the title-written arguments occupy.
    pub bound_bytes: u64,
    /// Bytes the zero default covers.
    pub unbound_bytes: u64,
    /// Last recorded flush's title-written parameter count.
    pub last_bound_parameters: u64,
    /// Last recorded flush's zero-default parameter count.
    pub last_unbound_parameters: u64,
    /// Last recorded flush's title-written bytes.
    pub last_bound_bytes: u64,
    /// Last recorded flush's `none|constants|view|table` shape.
    pub last_bound_kind: &'static str,
}

/// Records one ABI flush's binding summary.
pub fn note_abi_binding(summary: AbiBindingSummary) {
    ABI_BOUND_PARAMETERS.fetch_add(u64::from(summary.bound_parameters), Ordering::Relaxed);
    ABI_UNBOUND_PARAMETERS.fetch_add(u64::from(summary.unbound_parameters), Ordering::Relaxed);
    ABI_BOUND_BYTES.fetch_add(u64::from(summary.bound_bytes), Ordering::Relaxed);
    ABI_UNBOUND_BYTES.fetch_add(u64::from(summary.unbound_bytes), Ordering::Relaxed);
    ABI_LAST_BOUND_PARAMETERS.store(u64::from(summary.bound_parameters), Ordering::Relaxed);
    ABI_LAST_UNBOUND_PARAMETERS.store(u64::from(summary.unbound_parameters), Ordering::Relaxed);
    ABI_LAST_BOUND_BYTES.store(u64::from(summary.bound_bytes), Ordering::Relaxed);
    ABI_LAST_KIND_BITS.store(u64::from(summary.kind_bits()), Ordering::Relaxed);
}

/// The last recorded ABI binding summary.
///
/// Unlike [`last_record`] this is available without arming the draw tracer, so
/// the GPU proof can assert the per-draw counts directly.
pub fn last_abi_binding() -> AbiBindingSummary {
    let bits = ABI_LAST_KIND_BITS.load(Ordering::Relaxed) as u8;
    AbiBindingSummary {
        bound_parameters: ABI_LAST_BOUND_PARAMETERS.load(Ordering::Relaxed) as u32,
        unbound_parameters: ABI_LAST_UNBOUND_PARAMETERS.load(Ordering::Relaxed) as u32,
        bound_bytes: ABI_LAST_BOUND_BYTES.load(Ordering::Relaxed) as u32,
        unbound_bytes: 0,
        constants: bits & 0b001 != 0,
        view: bits & 0b010 != 0,
        table: bits & 0b100 != 0,
    }
}

/// A snapshot of the process-wide ABI binding totals.
pub fn abi_binding_telemetry() -> AbiBindingTelemetry {
    AbiBindingTelemetry {
        bound_parameters: ABI_BOUND_PARAMETERS.load(Ordering::Relaxed),
        unbound_parameters: ABI_UNBOUND_PARAMETERS.load(Ordering::Relaxed),
        bound_bytes: ABI_BOUND_BYTES.load(Ordering::Relaxed),
        unbound_bytes: ABI_UNBOUND_BYTES.load(Ordering::Relaxed),
        last_bound_parameters: ABI_LAST_BOUND_PARAMETERS.load(Ordering::Relaxed),
        last_unbound_parameters: ABI_LAST_UNBOUND_PARAMETERS.load(Ordering::Relaxed),
        last_bound_bytes: ABI_LAST_BOUND_BYTES.load(Ordering::Relaxed),
        last_bound_kind: AbiBindingSummary {
            bound_parameters: ABI_LAST_BOUND_PARAMETERS.load(Ordering::Relaxed) as u32,
            unbound_parameters: ABI_LAST_UNBOUND_PARAMETERS.load(Ordering::Relaxed) as u32,
            bound_bytes: ABI_LAST_BOUND_BYTES.load(Ordering::Relaxed) as u32,
            unbound_bytes: 0,
            constants: ABI_LAST_KIND_BITS.load(Ordering::Relaxed) & 0b001 != 0,
            view: ABI_LAST_KIND_BITS.load(Ordering::Relaxed) & 0b010 != 0,
            table: ABI_LAST_KIND_BITS.load(Ordering::Relaxed) & 0b100 != 0,
        }
        .bound_kind(),
    }
}

/// Clears the process-wide ABI binding totals and last-draw summary.
///
/// Test-only: a proof that runs several ABI shapes in one process starts each
/// case from a clean count. Never called by the host.
#[doc(hidden)]
pub fn reset_abi_binding_telemetry() {
    ABI_BOUND_PARAMETERS.store(0, Ordering::Relaxed);
    ABI_UNBOUND_PARAMETERS.store(0, Ordering::Relaxed);
    ABI_BOUND_BYTES.store(0, Ordering::Relaxed);
    ABI_UNBOUND_BYTES.store(0, Ordering::Relaxed);
    ABI_LAST_BOUND_PARAMETERS.store(0, Ordering::Relaxed);
    ABI_LAST_UNBOUND_PARAMETERS.store(0, Ordering::Relaxed);
    ABI_LAST_BOUND_BYTES.store(0, Ordering::Relaxed);
    ABI_LAST_KIND_BITS.store(0, Ordering::Relaxed);
}

/// Bounded picture of what the tracer saw, for `telemetry.json`.
///
/// When the trace is off this returns an all-zero report with
/// `enabled: false` without allocating or locking the log.
pub fn report() -> TraceReport {
    let enabled = enabled();
    if !enabled {
        return TraceReport::default();
    }
    let log = trace_log()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // A pass that was selected but whose readback never reached `sample_grid`
    // (commit error, dead texture, host shutdown) is visible as a mismatch
    // instead of silently keeping stale zeroes.
    let failure = log.failure.clone().or_else(|| {
        (log.passes_sampled > log.samples_recorded).then(|| "sample-not-read".to_string())
    });
    TraceReport {
        enabled,
        traced_work: log.records_described,
        traced_passes: log.passes_sampled,
        last_offscreen_nonzero_bytes: log.last_offscreen_nonzero_bytes,
        last_offscreen_distinct_values: log.last_offscreen_distinct_values,
        failure,
    }
}

/// The most recent described draw/dispatch, if the trace is on and one was
/// encoded. Diagnostic accessor for the GPU proof; the host reads the counts.
pub fn last_record() -> Option<DrawRecord> {
    if !enabled() {
        return None;
    }
    trace_log()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .last_record
}

/// The most recent pixel summary, if the trace is on and a pass was sampled.
pub fn last_sample() -> Option<PixelStats> {
    if !enabled() {
        return None;
    }
    trace_log()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .last_sample
}

/// Records the first bounded tracer failure.
///
/// The device readback path currently prints `draw_trace_rt failure=...`
/// directly; wiring it here as well is a device.rs follow-up.
pub fn note_failure(message: &str) {
    with_log(|log| {
        if log.failure.is_none() {
            let mut bounded = message.to_string();
            bounded.truncate(MAX_FAILURE_BYTES);
            log.failure = Some(bounded);
        }
    });
}

/// Clears the process-wide report.
///
/// Test-only: the GPU proof runs two attachments in one process, so each case
/// starts from a clean report. Never called by the host.
#[doc(hidden)]
pub fn reset() {
    with_log(|log| *log = TraceLog::default());
}

/// One attachment the tracer may sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceTarget {
    /// Level-0 texture of the pass's first colour attachment.
    ///
    /// Stored as a plain address so the tracer state stays `Send`; the host
    /// casts it back while the registry still owns the texture.
    pub texture: usize,
    /// Attachment width in texels.
    pub width: u32,
    /// Attachment height in texels.
    pub height: u32,
    /// Attachment pixel format name.
    pub format: &'static str,
    /// Whether the texture is the registered swapchain surface.
    pub swapchain: bool,
}

/// What one ABI flush wrote into the top-level argument buffer(s).
///
/// A parameter is **bound** only when a title `SetRootConstants`,
/// `SetRootBufferView` or `SetDescriptorTable` wrote it. Every declared
/// parameter without such a write is **unbound** and is filled with the zero
/// default the builder starts from (`MEMORY_SPACE` host-owned parameters are
/// also counted here: they are not title writes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbiBindingSummary {
    /// Declared parameters that received a title-written argument.
    pub bound_parameters: u32,
    /// Declared parameters with no title write (zero default).
    pub unbound_parameters: u32,
    /// Bytes a title-written argument occupies (constants bytes, 8-byte
    /// device addresses / table pointers).
    pub bound_bytes: u32,
    /// Bytes the zero default covers for the unbound parameters.
    pub unbound_bytes: u32,
    /// At least one bound parameter is a root-constants block.
    pub constants: bool,
    /// At least one bound parameter is a root CBV/SRV/UAV.
    pub view: bool,
    /// At least one bound parameter is a descriptor table.
    pub table: bool,
}

impl AbiBindingSummary {
    /// `none`, `constants`, `view`, `table`, or the `+`-joined combination.
    pub fn bound_kind(&self) -> &'static str {
        let bits = self.kind_bits();
        match bits {
            0 => "none",
            0b001 => "constants",
            0b010 => "view",
            0b100 => "table",
            0b011 => "constants+view",
            0b101 => "constants+table",
            0b110 => "view+table",
            _ => "constants+view+table",
        }
    }

    /// Bit mask: bit 0 constants, bit 1 view, bit 2 table.
    pub fn kind_bits(&self) -> u8 {
        u8::from(self.constants) | (u8::from(self.view) << 1) | (u8::from(self.table) << 2)
    }
}

/// One traced draw or dispatch.
#[derive(Debug, Clone, Copy)]
pub struct DrawRecord {
    /// One-based index among traced work (1..=10).
    pub ordinal: usize,
    /// `draw`, `draw_indexed`, `draw_indirect`, `dispatch`, `dispatch_indirect`
    /// or `mesh_dispatch`.
    pub kind: &'static str,
    /// Selected pipeline id.
    pub pipeline_id: u64,
    /// Vertices, indices or threadgroups the call names.
    pub count: u32,
    /// Instances (draws only; `1` otherwise).
    pub instances: u32,
    /// Viewport rectangle `(x, y, width, height)` in pixels, when set.
    pub viewport: Option<[u32; 4]>,
    /// Scissor rectangle `(x, y, width, height)` in pixels, when set.
    pub scissor: Option<[u32; 4]>,
    /// Reflected argument-buffer parameters of the active pipeline.
    pub abi_parameters: u32,
    /// Bytes of the stage argument buffers the call binds (the declared layout
    /// size; see [`AbiBindingSummary`] for what was title-written).
    pub abi_bytes: u32,
    /// Parameters the title wrote for this draw.
    pub abi_bound_parameters: u32,
    /// Declared parameters this draw left at the zero default.
    pub abi_unbound_parameters: u32,
    /// Bytes the title-written arguments occupy.
    pub abi_bound_bytes: u32,
    /// `none|constants|view|table` (or a `+`-joined combination).
    pub abi_bound: &'static str,
    /// Attachment the pass renders into, when known.
    pub target: Option<TraceTarget>,
}

impl DrawRecord {
    /// One log line, counts and shapes only.
    pub fn line(&self) -> String {
        let mut out = format!(
            "draw_trace ordinal={} kind={} pipeline={} count={} instances={}",
            self.ordinal, self.kind, self.pipeline_id, self.count, self.instances
        );
        if let Some([x, y, w, h]) = self.viewport {
            out.push_str(&format!(" viewport={x},{y},{w},{h}"));
        } else {
            out.push_str(" viewport=none");
        }
        if let Some([x, y, w, h]) = self.scissor {
            out.push_str(&format!(" scissor={x},{y},{w},{h}"));
        } else {
            out.push_str(" scissor=none");
        }
        out.push_str(&format!(
            " abi_parameters={} abi_bytes={}",
            self.abi_parameters, self.abi_bytes
        ));
        out.push_str(&format!(
            " abi_bound_parameters={} abi_unbound_parameters={} abi_bound_bytes={} bound={}",
            self.abi_bound_parameters, self.abi_unbound_parameters, self.abi_bound_bytes,
            self.abi_bound
        ));
        match self.target {
            Some(target) => out.push_str(&format!(" rt={}", target_name(&target))),
            None => out.push_str(" rt=none"),
        }
        out
    }
}

/// `offscreen|swapchain WIDTHxHEIGHT format`, the attachment shape both log
/// lines name.
fn target_name(target: &TraceTarget) -> String {
    format!(
        "{} {}x{} {}",
        if target.swapchain {
            "swapchain"
        } else {
            "offscreen"
        },
        target.width,
        target.height,
        target.format
    )
}

/// Bounded per-process tracer state.
#[derive(Default)]
pub struct TraceState {
    /// Traced calls, in encode order.
    pub records: Vec<DrawRecord>,
    /// Passes offered to the sampler so far.
    pub passes: u32,
    /// Passes whose pixels were sampled.
    pub sampled_passes: u32,
    /// Draws/dispatches recorded since the last sampling decision. A pass with
    /// zero pending draws is never sampled.
    pub pending_draws: u32,
    /// Pending draws the last sampled pass carried.
    pub last_pass_draws: u32,
    /// Attachment of the most recent record, published with the pass line.
    pub last_target: Option<TraceTarget>,
}

impl TraceState {
    /// Whether [`MAX_TRACED_WORK`] calls were already described.
    pub fn saturated(&self) -> bool {
        self.records.len() >= MAX_TRACED_WORK
    }

    /// Appends and formats one record when the process still traces.
    pub fn record(&mut self, record: DrawRecord) -> Option<String> {
        if self.saturated() {
            return None;
        }
        let line = record.line();
        self.records.push(record);
        self.pending_draws = self.pending_draws.saturating_add(1);
        self.last_target = record.target;
        if enabled() {
            with_log(|log| {
                log.records_described = log.records_described.saturating_add(1);
                log.last_record = Some(record);
            });
        }
        Some(line)
    }

    /// Counts one sampling opportunity and reports whether it should be
    /// sampled.
    ///
    /// A pass whose render segment encoded no draw/dispatch returns `false`:
    /// a clear-only pass must never be mistaken for drawn, non-zero output.
    /// At most [`MAX_TRACED_PASSES`] passes are sampled.
    pub fn next_pass(&mut self) -> bool {
        self.passes = self.passes.saturating_add(1);
        if self.pending_draws == 0 {
            return false;
        }
        if self.sampled_passes >= MAX_TRACED_PASSES {
            return false;
        }
        self.sampled_passes = self.sampled_passes.saturating_add(1);
        self.last_pass_draws = self.pending_draws;
        self.pending_draws = 0;
        if enabled() {
            let context = SampleContext {
                pass: self.passes,
                draws: self.last_pass_draws,
                target: self.last_target,
            };
            with_log(|log| {
                log.passes_sampled = log.passes_sampled.saturating_add(1);
                log.sample_context = Some(context);
            });
        }
        true
    }
}

/// Bounded summary of a sampled `RGBA8`/`BGRA8` attachment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelStats {
    /// Bytes that are not zero.
    pub nonzero_bytes: u64,
    /// Bytes sampled.
    pub sampled_bytes: u64,
    /// Distinct byte values, saturated at 256.
    pub distinct_values: u32,
    /// Sampling-opportunity counter of the pass this summary belongs to; `0`
    /// for a summary produced outside the device trace path.
    pub pass: u32,
    /// Draws/dispatches the sampled pass encoded; `0` when uncounted.
    pub pass_draws: u32,
    /// Attachment that was sampled, when the trace named one.
    pub target: Option<TraceTarget>,
}

impl PixelStats {
    /// The pass line (when known) followed by the pixel summary line.
    ///
    /// The pass line always precedes its own summary so a reader can attribute
    /// `nonzero_bytes` to a pass without correlating unrelated log lines.
    pub fn line(&self) -> String {
        let summary = format!(
            "draw_trace_rt nonzero_bytes={} sampled_bytes={} distinct_values={}",
            self.nonzero_bytes, self.sampled_bytes, self.distinct_values
        );
        if self.pass == 0 {
            return summary;
        }
        format!(
            "draw_trace_pass pass={} draws={} rt={}\n{}",
            self.pass,
            self.pass_draws,
            self.target
                .as_ref()
                .map_or_else(|| "none".to_string(), target_name),
            summary
        )
    }
}

/// Summarizes a `RT_SAMPLE_EDGE`-square grid of evenly spaced texels from a
/// whole `RGBA8`/`BGRA8` attachment.
///
/// The input is the level-0 image (`width * height * 4` bytes, tightly packed).
/// At most [`RT_SAMPLE_EDGE`]² texels are read, so a 4K attachment costs the
/// same bounded summary as a 64×64 one.
pub fn sample_grid(bytes: &[u8], width: u32, height: u32) -> PixelStats {
    let enabled = enabled();
    if width == 0 || height == 0 {
        if enabled {
            note_failure("empty-attachment");
        }
        return PixelStats::default();
    }
    let row_bytes = width as usize * 4;
    if bytes.len() < row_bytes * height as usize {
        if enabled {
            note_failure("short-readback");
        }
        return PixelStats::default();
    }
    let edge = RT_SAMPLE_EDGE.min(width).min(height);
    let step_x = (width / edge).max(1) as usize;
    let step_y = (height / edge).max(1) as usize;
    let mut sample = Vec::with_capacity((edge * edge * 4) as usize);
    for row in 0..edge as usize {
        let y = (row * step_y).min(height as usize - 1);
        for column in 0..edge as usize {
            let x = (column * step_x).min(width as usize - 1);
            let at = y * row_bytes + x * 4;
            sample.extend_from_slice(&bytes[at..at + 4]);
        }
    }
    let mut stats = pixel_stats(&sample);
    if enabled {
        with_log(|log| {
            if let Some(context) = log.sample_context.take() {
                stats.pass = context.pass;
                stats.pass_draws = context.draws;
                stats.target = context.target;
            }
            if let Some(target) = stats.target
                && !target.swapchain
            {
                log.last_offscreen_nonzero_bytes = stats.nonzero_bytes;
                log.last_offscreen_distinct_values = stats.distinct_values;
            }
            log.samples_recorded = log.samples_recorded.saturating_add(1);
            log.last_sample = Some(stats);
        });
    }
    stats
}

/// Summarizes a sampled attachment without copying it.
pub fn pixel_stats(bytes: &[u8]) -> PixelStats {
    let mut nonzero = 0u64;
    let mut seen = [false; 256];
    let mut distinct = 0u32;
    for &byte in bytes {
        if byte != 0 {
            nonzero += 1;
        }
        let slot = usize::from(byte);
        if !seen[slot] {
            seen[slot] = true;
            distinct += 1;
        }
    }
    PixelStats {
        nonzero_bytes: nonzero,
        sampled_bytes: bytes.len() as u64,
        distinct_values: distinct,
        ..PixelStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tracer_is_off_without_a_truthy_flag() {
        assert!(!enabled_from(None));
        assert!(!enabled_from(Some("")));
        assert!(!enabled_from(Some("0")));
        assert!(!enabled_from(Some("false")));
        for value in ["1", "true", " on ", "yes", "ON"] {
            assert!(enabled_from(Some(value)), "{value:?} arms the tracer");
        }
    }

    #[test]
    fn counters_bump_independently_and_snapshot() {
        let counters = EncodeCounters::default();
        counters.bump_draw();
        counters.bump_draw_indexed();
        counters.bump_dispatch();
        counters.bump_mesh_dispatch();
        counters.bump_indirect();
        counters.bump_render_pass();
        counters.bump_clear();
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.draws, 1);
        assert_eq!(snapshot.draws_indexed, 1);
        assert_eq!(snapshot.total_draws(), 3);
        assert_eq!(snapshot.dispatches, 1);
        assert_eq!(snapshot.mesh_dispatches, 1);
        assert_eq!(snapshot.indirect, 1);
        assert_eq!(snapshot.render_passes, 1);
        assert_eq!(snapshot.clear_ops, 1);
        assert!(snapshot.json_fields().contains("\"draws_encoded\": 1"));
    }

    #[test]
    fn trace_state_bounds_records_and_passes() {
        let mut state = TraceState::default();
        for ordinal in 1..=MAX_TRACED_WORK {
            let record = DrawRecord {
                ordinal,
                kind: "draw",
                pipeline_id: u64::from(ordinal as u32),
                count: 3,
                instances: 1,
                viewport: None,
                scissor: None,
                abi_parameters: 0,
                abi_bytes: 0,
                abi_bound_parameters: 0,
                abi_unbound_parameters: 0,
                abi_bound_bytes: 0,
                abi_bound: "none",
                target: None,
            };
            assert!(state.record(record).is_some());
        }
        assert!(state.saturated());
        assert!(
            state
                .record(DrawRecord {
                    ordinal: 11,
                    kind: "draw",
                    pipeline_id: 0,
                    count: 1,
                    instances: 1,
                    viewport: None,
                    scissor: None,
                    abi_parameters: 0,
                    abi_bytes: 0,
                    abi_bound_parameters: 0,
                    abi_unbound_parameters: 0,
                    abi_bound_bytes: 0,
                    abi_bound: "none",
                    target: None,
                })
                .is_none()
        );
        for _ in 0..MAX_TRACED_PASSES {
            state.pending_draws = 1;
            assert!(state.next_pass());
        }
        state.pending_draws = 1;
        assert!(!state.next_pass());
    }

    #[test]
    fn a_pass_without_draws_is_never_sampled() {
        let mut state = TraceState::default();
        assert!(!state.next_pass());
        assert_eq!(state.sampled_passes, 0);
        state.pending_draws = 2;
        assert!(state.next_pass());
        assert_eq!(state.last_pass_draws, 2);
        assert_eq!(state.pending_draws, 0);
    }

    #[test]
    fn the_pass_line_precedes_its_pixel_summary() {
        let stats = PixelStats {
            nonzero_bytes: 12,
            sampled_bytes: 16384,
            distinct_values: 3,
            pass: 4,
            pass_draws: 2,
            target: Some(TraceTarget {
                texture: 0,
                width: 64,
                height: 64,
                format: "rgba8unorm",
                swapchain: false,
            }),
        };
        let line = stats.line();
        let mut lines = line.lines();
        assert_eq!(
            lines.next(),
            Some("draw_trace_pass pass=4 draws=2 rt=offscreen 64x64 rgba8unorm")
        );
        assert_eq!(
            lines.next(),
            Some("draw_trace_rt nonzero_bytes=12 sampled_bytes=16384 distinct_values=3")
        );
    }

    #[test]
    fn an_uncounted_summary_stays_a_single_line() {
        let stats = PixelStats {
            nonzero_bytes: 0,
            sampled_bytes: 4096,
            distinct_values: 1,
            ..PixelStats::default()
        };
        assert_eq!(
            stats.line(),
            "draw_trace_rt nonzero_bytes=0 sampled_bytes=4096 distinct_values=1"
        );
    }

    #[test]
    fn a_record_names_counts_shapes_and_attachment() {
        let record = DrawRecord {
            ordinal: 2,
            kind: "draw_indexed",
            pipeline_id: 9,
            count: 36,
            instances: 4,
            viewport: Some([0, 0, 1280, 720]),
            scissor: Some([0, 0, 640, 360]),
            abi_parameters: 3,
            abi_bytes: 40,
            abi_bound_parameters: 2,
            abi_unbound_parameters: 1,
            abi_bound_bytes: 24,
            abi_bound: "constants+view",
            target: Some(TraceTarget {
                texture: 0,
                width: 1920,
                height: 1080,
                format: "rgba8unorm",
                swapchain: false,
            }),
        };
        let line = record.line();
        assert!(line.contains("kind=draw_indexed"));
        assert!(line.contains("count=36"));
        assert!(line.contains("viewport=0,0,1280,720"));
        assert!(line.contains("abi_bound_parameters=2"));
        assert!(line.contains("abi_unbound_parameters=1"));
        assert!(line.contains("abi_bound_bytes=24"));
        assert!(line.contains("bound=constants+view"));
        assert!(line.contains("rt=offscreen 1920x1080 rgba8unorm"));
    }

    fn key(set: u16) -> DescriptorTableRefusalKey {
        DescriptorTableRefusalKey {
            pipeline_id: 7,
            wire_set: set,
            declared_tables: 2,
            reflected_tables: 2,
            stage_mask: 0b10,
            map_tables: 2,
            had_plan: false,
            plan_kind: "none",
            outcome: OUTCOME_REFUSED,
        }
    }

    #[test]
    fn the_descriptor_table_census_counts_repeated_keys_once() {
        reset_descriptor_table_refusal_census();
        note_descriptor_table_refusal(key(1));
        note_descriptor_table_refusal(key(1));
        note_descriptor_table_refusal(key(0));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, 2);
        assert_eq!(census.omitted, 0);
        assert_eq!(census.entries[0].key.wire_set, 1);
        assert_eq!(census.entries[0].count, 2);
        assert_eq!(census.entries[1].key.wire_set, 0);
        assert_eq!(census.entries[1].count, 1);
        let json = census.json_value();
        assert!(json.contains("\"distinct_keys\": 2"));
        assert!(json.contains("\"plan_kind\": \"none\""));
        assert!(json.contains("\"outcome\": \"refused\""));
        assert!(json.contains("\"reflected_tables\": 2"));
        assert!(json.contains("\"map_tables\": 2"));
        assert!(json.contains("\"stage_mask\": 2"));
        reset_descriptor_table_refusal_census();
    }

    #[test]
    fn the_descriptor_table_census_is_capped() {
        reset_descriptor_table_refusal_census();
        for set in 0..MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u16 {
            note_descriptor_table_refusal(key(set));
        }
        note_descriptor_table_refusal(key(99));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u64);
        assert_eq!(census.entries.len(), MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS);
        assert_eq!(census.omitted, 1);
        reset_descriptor_table_refusal_census();
        assert!(descriptor_table_refusal_census().is_empty());
    }

    fn kind_key(wire_set: u16) -> DescriptorTableKindMismatchKey {
        DescriptorTableKindMismatchKey {
            pipeline_id: 2,
            wire_set,
            reflected_kind: "ShaderResourceView",
            wire_kind: "Cbv",
            entry_shape: "buffer",
        }
    }

    #[test]
    fn the_kind_mismatch_census_counts_and_names_the_class() {
        reset_descriptor_table_refusal_census();
        note_descriptor_table_kind_mismatch(kind_key(1));
        note_descriptor_table_kind_mismatch(kind_key(1));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, 1);
        assert_eq!(census.omitted, 0);
        assert!(census.entries.is_empty(), "a kind mismatch is not a refusal key");
        assert_eq!(census.kind_mismatches.len(), 1);
        let entry = &census.kind_mismatches[0];
        assert_eq!(entry.count, 2);
        assert_eq!(entry.key.pipeline_id, 2);
        assert_eq!(entry.key.wire_set, 1);
        assert_eq!(entry.key.reflected_kind, "ShaderResourceView");
        assert_eq!(entry.key.wire_kind, "Cbv");
        assert_eq!(entry.key.entry_shape, "buffer");
        let json = census.json_value();
        assert!(json.contains("\"kind_mismatch_keys\""));
        assert!(json.contains("\"reflected_kind\": \"ShaderResourceView\""));
        assert!(json.contains("\"wire_kind\": \"Cbv\""));
        assert!(json.contains("\"entry_shape\": \"buffer\""));
        assert!(json.contains("\"outcome\": \"kind-mismatch\""));
        assert!(json.contains("\"count\": 2"));
        reset_descriptor_table_refusal_census();
        assert!(descriptor_table_refusal_census().is_empty());
    }

    #[test]
    fn kind_mismatch_keys_share_the_resolution_cap() {
        reset_descriptor_table_refusal_census();
        for set in 0..MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u16 {
            note_descriptor_table_refusal(key(set));
        }
        note_descriptor_table_kind_mismatch(kind_key(1));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u64);
        assert!(census.kind_mismatches.is_empty(), "the cap is shared");
        assert_eq!(census.omitted, 1);
        reset_descriptor_table_refusal_census();
    }

    #[test]
    fn pixel_stats_are_exact_and_bounded() {
        let clear = vec![0u8; 4096];
        assert_eq!(pixel_stats(&clear).nonzero_bytes, 0);
        assert_eq!(pixel_stats(&clear).distinct_values, 1);
        let mut drawn = vec![0u8; 4096];
        drawn[100] = 0xff;
        drawn[101] = 0x7f;
        let stats = pixel_stats(&drawn);
        assert_eq!(stats.nonzero_bytes, 2);
        assert_eq!(stats.distinct_values, 3);
        assert_eq!(stats.sampled_bytes, 4096);
    }
}
