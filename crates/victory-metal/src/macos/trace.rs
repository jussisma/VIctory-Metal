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
//!
//! ADR-068 adds the per-render-target "written after last clear" evidence. Each
//! sampled pass already carries its clear state, so the same bounded readback
//! now answers three questions a black boot could not: did the shader draw
//! zeros (`drew-zeros`), did a later clear erase non-zero pixels
//! (`erased-by-later-clear`), or was the RT never sampled at all
//! (`never-sampled`)? The evidence lives in a [`MAX_RT_HISTORY`]-bounded
//! history keyed by texture handle; no readback, allocation or state grows with
//! the pass count.

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

/// Distinct RGBA texels [`pixel_stats`] reports at most.
///
/// The grid holds at most [`RT_SAMPLE_EDGE`]² texels; the counter stops there
/// like `distinct_values` stops at 256 so the summary stays a bounded integer.
pub const MAX_DISTINCT_PIXELS: u32 = 256;

/// Distinct render targets the per-RT written-after-clear history retains.
///
/// A boot encodes thousands of passes but the tracer samples at most
/// [`MAX_TRACED_PASSES`] of them and describes at most [`MAX_TRACED_WORK`]
/// draws, so this bound is never reached by traced work alone; it exists so a
/// hostile or degenerate stream cannot grow host memory. Keys past the cap are
/// counted as dropped.
pub const MAX_RT_HISTORY: usize = 32;

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

/// Outcome word: a mixed resource table was split by entry shape and bound to
/// its declared consecutive table parameters.
pub const OUTCOME_SPLIT: &str = "split";

/// Outcome word: a mixed resource table was bound to the declared slots whose
/// ranges could carry its entries, and the entries no slot could address were
/// counted as residue ([`DescriptorTablePartitionResidueKey`]).
pub const OUTCOME_PARTIAL: &str = "partial";

/// Outcome word: a mixed resource table was bound whole, in wire order, as one
/// converted descriptor array at a declared `Unspecified` table slot.
///
/// Converter reflection reports every range as `"Table"` without a range kind,
/// so such a slot is the converter's own 24-byte descriptor array and every
/// entry keeps the index the title gave it. No partition is needed, and none is
/// dropped: the wire entries are the authoritative shape of each element.
pub const OUTCOME_MERGED: &str = "merged";

/// Drop reason: the bound pipeline declares no descriptor table at all, so a
/// wire table has no root parameter it could address.
pub const DROP_NO_DECLARED_TABLE: &str = "no-declared-table";

/// Drop reason: the bound pipeline declares table parameters, but none names
/// the resolved slot for a mixed table's partitions.
pub const DROP_NO_DECLARED_SLOT: &str = "no-declared-slot";

/// Drop reason: the declared range kind of the resolved slot cannot carry any
/// partition of the mix (a resource view mix under a `Sampler` range, or every
/// entry shape failing the range kind's descriptor shape).
pub const DROP_DECLARED_KIND_REJECTS_PARTITIONS: &str = "declared-kind-rejects-partitions";

/// Drop reason: a declared fragment texture/sampler source was dropped as the
/// residue of a partially bound table.
pub const DROP_FRAGMENT_SOURCE_RESIDUE: &str = "fragment-source-residue";

/// Drop reason: the fragment declares a texture/sampler source and this draw
/// bound no texture or sampler entry to it — either no wire table was written
/// for the pipeline, or the table that was written carries only address
/// entries. Either way the declared source has no bound entry, and the class is
/// named instead of inferred.
pub const DROP_DECLARED_SOURCE_UNFILLED: &str = "declared-source-unfilled";

/// One bounded key of the descriptor-table *kind mismatch* census.
///
/// It names only fixed kind words and counts: the reflected table kind from the
/// resolved root parameter, the wire `DescriptorTableV3.kind`, the entry
/// variant shape (`buffer`, `texture`, `sampler`, `mixed` or `empty`), and the
/// exact entry combination class (`single`, `buffer+texture`,
/// `buffer+sampler`, `texture+sampler`, `all`, `empty`). No handles, register
/// values, shader bytes or resource contents appear here.
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
    /// Entry combination class, finer than `entry_shape`: `single`,
    /// `buffer+texture`, `buffer+sampler`, `texture+sampler`, `all` or `empty`.
    pub mix_class: &'static str,
}

/// One bounded key of the descriptor-table *mix* census.
///
/// A D3D12 table may name several ranges in one resource heap, and the
/// converter flattens each range as its own table parameter. `mix_class` names
/// the entry-variant combination and `outcome` says how the host encoded it:
/// [`OUTCOME_SPLIT`] (bound as per-shape sub-tables), [`OUTCOME_INERT`] (the
/// declared layout has no consecutive slots for it; the submission still
/// encodes) or [`OUTCOME_REFUSED`] (an illegal sampler/resource mix). Counts
/// only; no handles or payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTableMixKey {
    /// Host pipeline id the table was selected with.
    pub pipeline_id: u64,
    /// Wire `DescriptorTableV3.set` the submission carried.
    pub wire_set: u16,
    /// Entry combination class (`single`, `buffer+texture`, `buffer+sampler`,
    /// `texture+sampler`, `all` or `empty`).
    pub mix_class: &'static str,
    /// Outcome word: [`OUTCOME_SPLIT`], [`OUTCOME_PARTIAL`],
    /// [`OUTCOME_INERT`] or [`OUTCOME_REFUSED`].
    pub outcome: &'static str,
}

/// One bounded key of the descriptor-table *partition residue* census.
///
/// A mixed resource table whose declared layout has fewer matching table slots
/// than entry shapes binds the entries the slots can address and drops the
/// rest. Every dropped partition is named here with the declared slot index it
/// could not fill and its entry count, so the drop is observable in telemetry
/// instead of vanishing behind an inert acceptance. Counts and fixed shape
/// words only: no handles, payloads or title identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTablePartitionResidueKey {
    /// Host pipeline id the table was selected with.
    pub pipeline_id: u64,
    /// Wire `DescriptorTableV3.set` the submission carried.
    pub wire_set: u16,
    /// Flattened parameter index of the declared slot that could not carry the
    /// partition.
    pub slot_index: u16,
    /// Reflected range kind of that declared slot.
    pub declared_kind: &'static str,
    /// Entry variant shape of the dropped partition (`buffer`/`texture`/
    /// `sampler`).
    pub entry_shape: &'static str,
    /// Entries the partition carried and the host dropped.
    pub dropped_entries: u16,
}

/// One counted descriptor-table partition-residue key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTablePartitionResidueCount {
    /// The bounded key.
    pub key: DescriptorTablePartitionResidueKey,
    /// How many partitions the key describes in this process.
    pub count: u64,
}

/// One bounded key naming *why* a declared table was not bound.
///
/// The `mix`/`inert` keys say a table combination was accepted inertly, but not
/// which early return dropped it. This key answers that in one boot: the
/// resolved declared range kind(s), how many table slots the pipeline declares,
/// the entry combination class, and the fixed reason word of the return that
/// dropped the table. Counts and fixed words only: no handles, register values,
/// shader bytes or title identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTableBindingDropKey {
    /// Host pipeline id the dropped table was selected with.
    pub pipeline_id: u64,
    /// Wire `DescriptorTableV3.set` the submission carried.
    pub wire_set: u16,
    /// Reflected range kind of the resolved slot
    /// (`ConstantBufferView`/`ShaderResourceView`/`UnorderedAccessView`/
    /// `Texture`/`Sampler`/`Unspecified`, or `none` when no slot resolved).
    pub declared_kind: &'static str,
    /// Descriptor-table parameters the bound pipeline declares.
    pub declared_tables: u16,
    /// Descriptor-table parameters the widest single stage layout declares.
    pub declared_slots: u16,
    /// Entry combination class (`single`, `buffer+texture`, `buffer+sampler`,
    /// `texture+sampler`, `all`, `empty`).
    pub mix_class: &'static str,
    /// Fixed reason word: one of the `DROP_*` constants.
    pub reason: &'static str,
}

/// One counted descriptor-table binding-drop key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTableBindingDropCount {
    /// The bounded key.
    pub key: DescriptorTableBindingDropKey,
    /// How many dropped tables the key describes in this process.
    pub count: u64,
}

/// One counted descriptor-table mix key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorTableMixCount {
    /// The bounded key.
    pub key: DescriptorTableMixKey,
    /// How many tables the key describes in this process.
    pub count: u64,
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
    /// Retained mixed-table keys in first-observed order.
    pub mix_keys: Vec<DescriptorTableMixCount>,
    /// Retained mixed-table partition residues in first-observed order.
    pub partition_residues: Vec<DescriptorTablePartitionResidueCount>,
    /// Retained declared-table binding drops in first-observed order.
    pub binding_drops: Vec<DescriptorTableBindingDropCount>,
    /// Per-stage fragment ABI observability of the last recorded flush.
    ///
    /// `telemetry.json` embeds this census, so these counters reach the boot
    /// log without the draw tracer being armed.
    pub stage_abi: StageAbiTelemetry,
}

impl DescriptorTableRefusalCensus {
    /// Whether no descriptor table was refused or accepted as inert.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.kind_mismatches.is_empty()
            && self.mix_keys.is_empty()
            && self.partition_residues.is_empty()
            && self.binding_drops.is_empty()
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
    /// `kind_mismatch_keys`, `mix_keys`, `partition_residues`, `binding_drops`;
    /// each resolution key carries `pipeline_id`, `wire_set`, `declared_tables`,
    /// `reflected_tables`, `stage_mask`, `map_tables`, `had_plan`, `plan_kind`,
    /// `outcome` and `count`. Each kind-mismatch key carries `pipeline_id`,
    /// `wire_set`, `reflected_kind`, `wire_kind`, `entry_shape`, `mix_class`,
    /// `outcome` (`"kind-mismatch"`) and `count`. Each mix key carries
    /// `pipeline_id`, `wire_set`, `mix_class`, `outcome` and `count`. Each
    /// partition-residue key carries `pipeline_id`, `wire_set`, `slot_index`,
    /// `declared_kind`, `entry_shape`, `dropped_entries` and `count`. Each
    /// binding-drop key carries `pipeline_id`, `wire_set`, `declared_kind`,
    /// `declared_tables`, `declared_slots`, `mix_class`, `reason` and `count`.
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
                     \"wire_kind\": \"{}\", \"entry_shape\": \"{}\", \"mix_class\": \"{}\", \
                     \"outcome\": \"{}\", \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.reflected_kind,
                    entry.key.wire_kind,
                    entry.key.entry_shape,
                    entry.key.mix_class,
                    OUTCOME_KIND_MISMATCH,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mix_keys = self
            .mix_keys
            .iter()
            .map(|entry| {
                format!(
                    "{{\"pipeline_id\": {}, \"wire_set\": {}, \"mix_class\": \"{}\", \
                     \"outcome\": \"{}\", \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.mix_class,
                    entry.key.outcome,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let partition_residues = self
            .partition_residues
            .iter()
            .map(|entry| {
                format!(
                    "{{\"pipeline_id\": {}, \"wire_set\": {}, \"slot_index\": {}, \
                     \"declared_kind\": \"{}\", \"entry_shape\": \"{}\", \
                     \"dropped_entries\": {}, \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.slot_index,
                    entry.key.declared_kind,
                    entry.key.entry_shape,
                    entry.key.dropped_entries,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let stage_abi = format!(
            "{{\"stage_mask\": {}, \"ps_abi_parameters\": {}, \
             \"ps_bound_parameters\": {}, \"ps_bound_constants\": {}, \
             \"ps_bound_views\": {}, \"ps_bound_tables\": {}, \
             \"fragment_texture_sources_declared\": {}, \
             \"fragment_texture_sources_bound\": {}, \
             \"fragment_texture_sources_dropped\": {}, \
             \"fragment_source_drop_reason\": \"{}\", \"tables_dropped_inert\": {}, \
             \"ps_tables_dropped_inert\": {}, \"fragment_source_unreachable\": {}, \
             \"fragment_source_unreachable_reason\": \"{}\"}}",
            self.stage_abi.stage_mask,
            self.stage_abi.ps_abi_parameters,
            self.stage_abi.ps_bound_parameters,
            self.stage_abi.ps_bound_constants,
            self.stage_abi.ps_bound_views,
            self.stage_abi.ps_bound_tables,
            self.stage_abi.fragment_texture_sources_declared,
            self.stage_abi.fragment_texture_sources_bound,
            self.stage_abi.fragment_texture_sources_dropped,
            self.stage_abi.fragment_source_drop_reason,
            self.stage_abi.tables_dropped_inert,
            self.stage_abi.ps_tables_dropped_inert,
            self.stage_abi.fragment_source_unreachable,
            self.stage_abi.fragment_source_unreachable_reason
        );
        let binding_drops = self
            .binding_drops
            .iter()
            .map(|entry| {
                format!(
                    "{{\"pipeline_id\": {}, \"wire_set\": {}, \"declared_kind\": \"{}\", \
                     \"declared_tables\": {}, \"declared_slots\": {}, \
                     \"mix_class\": \"{}\", \"reason\": \"{}\", \"count\": {}}}",
                    entry.key.pipeline_id,
                    entry.key.wire_set,
                    entry.key.declared_kind,
                    entry.key.declared_tables,
                    entry.key.declared_slots,
                    entry.key.mix_class,
                    entry.key.reason,
                    entry.count
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\"distinct_keys\": {}, \"omitted\": {}, \"keys\": [{}], \
             \"kind_mismatch_keys\": [{}], \"mix_keys\": [{}], \
             \"partition_residues\": [{}], \"binding_drops\": [{}], \
             \"stage_abi\": {}}}",
            self.distinct_keys,
            self.omitted,
            keys,
            kind_mismatches,
            mix_keys,
            partition_residues,
            binding_drops,
            stage_abi
        )
    }
}

#[derive(Default)]
struct DescriptorTableRefusalLog {
    entries: Vec<(DescriptorTableRefusalKey, u64)>,
    kind_mismatches: Vec<(DescriptorTableKindMismatchKey, u64)>,
    mix: Vec<(DescriptorTableMixKey, u64)>,
    residues: Vec<(DescriptorTablePartitionResidueKey, u64)>,
    drops: Vec<(DescriptorTableBindingDropKey, u64)>,
    omitted: u64,
}

impl DescriptorTableRefusalLog {
    fn retained(&self) -> usize {
        self.entries.len()
            + self.kind_mismatches.len()
            + self.mix.len()
            + self.residues.len()
            + self.drops.len()
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

/// Counts one mixed descriptor-table event under its bounded key.
///
/// It records every mixed table the host sees, whatever its outcome
/// ([`OUTCOME_SPLIT`], [`OUTCOME_INERT`] or [`OUTCOME_REFUSED`]), so one boot
/// names the exact entry combination without any payload. It shares the one
/// [`MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`] cap; keys past the cap only bump
/// `omitted`.
pub fn note_descriptor_table_mix(key: DescriptorTableMixKey) {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, count)) = log.mix.iter_mut().find(|(known, _)| *known == key) {
        *count = count.saturating_add(1);
        return;
    }
    if log.retained() >= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS {
        log.omitted = log.omitted.saturating_add(1);
        return;
    }
    log.mix.push((key, 1));
}

/// Counts one mixed-table partition the declared slots could not address.
///
/// The table is still bound (the entries a slot *can* carry are uploaded), so
/// this is not an inert acceptance: the residue is its own named class so a
/// boot can see exactly how many entries the composition path dropped, at
/// which declared slot index and of which entry shape. It shares the one
/// [`MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`] cap; keys past the cap only bump
/// `omitted`.
pub fn note_descriptor_table_partition_residue(key: DescriptorTablePartitionResidueKey) {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, count)) = log.residues.iter_mut().find(|(known, _)| *known == key) {
        *count = count.saturating_add(1);
        return;
    }
    if log.retained() >= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS {
        log.omitted = log.omitted.saturating_add(1);
        return;
    }
    log.residues.push((key, 1));
}

/// Counts one dropped declared table under its bounded reason key.
///
/// `key.reason` is the fixed word of the early return that dropped it, so a boot
/// never has to infer the cause from a coarse `inert` outcome. It shares the one
/// [`MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS`] cap; keys past the cap only bump
/// `omitted`.
pub fn note_descriptor_table_binding_drop(key: DescriptorTableBindingDropKey) {
    let mut log = descriptor_table_refusals()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, count)) = log.drops.iter_mut().find(|(known, _)| *known == key) {
        *count = count.saturating_add(1);
        return;
    }
    if log.retained() >= MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS {
        log.omitted = log.omitted.saturating_add(1);
        return;
    }
    log.drops.push((key, 1));
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
        mix_keys: log
            .mix
            .iter()
            .map(|(key, count)| DescriptorTableMixCount {
                key: *key,
                count: *count,
            })
            .collect(),
        partition_residues: log
            .residues
            .iter()
            .map(|(key, count)| DescriptorTablePartitionResidueCount {
                key: *key,
                count: *count,
            })
            .collect(),
        binding_drops: log
            .drops
            .iter()
            .map(|(key, count)| DescriptorTableBindingDropCount {
                key: *key,
                count: *count,
            })
            .collect(),
        stage_abi: stage_abi_telemetry(),
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
    reset_stage_abi_telemetry();
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

impl TraceReport {
    /// ADR-068: the bounded per-RT written-after-clear evidence.
    ///
    /// It is read from the same process-wide log as the counters, so the host
    /// can fold it beside the existing keys without a second trace path. It is
    /// empty while the tracer is off.
    pub fn rt_evidence(&self) -> RtWriteEvidence {
        rt_write_evidence()
    }
}

/// Attachment and pass context shared between [`TraceState::next_pass`] and
/// [`sample_grid`], which the host calls through separate device functions.
#[derive(Debug, Clone, Copy)]
struct SampleContext {
    pass: u32,
    draws: u32,
    target: Option<TraceTarget>,
    clear: Option<TraceClear>,
}

/// ADR-068: the clear state of the colour attachment a sampled pass renders to.
///
/// A pass either clears its first colour attachment at pass start or loads the
/// previous contents. A sub-pass that follows an explicit
/// `ClearRenderTargetView` is a clear too, so the sampled bytes can be
/// attributed to "after the clear" or "after the draws".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TraceClear {
    /// The pass begins with a clear of the traced attachment.
    pub is_clear: bool,
    /// Colour that clear writes.
    pub color: [f64; 4],
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
    /// ADR-068: bounded per-RT written-after-clear history.
    rts: RtHistory,
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

/// The exact missing wire concept when a texture/sampler source reaches the
/// fragment stage through no root parameter.
///
/// D3D12 exposes those sources through a descriptor heap. This wire has no
/// `SetDescriptorHeaps` command, so a pipeline whose converter reflection
/// lowers tables to directly indexed heap access (`buffer(0)`/`buffer(1)`)
/// can only be served by a heap command the guest never sends. The counter is
/// [`StageAbiTelemetry::fragment_source_unreachable`].
pub const FRAGMENT_SOURCE_UNREACHABLE_REASON: &str =
    "no wire SetDescriptorHeaps command and the fragment declares no table/texture root parameter";

static LAST_STAGE_MASK: AtomicU64 = AtomicU64::new(0);
static LAST_PS_ABI_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_BOUND_PARAMETERS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_BOUND_CONSTANTS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_BOUND_VIEWS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_BOUND_TABLES: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_TEXTURE_SOURCES_DECLARED: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_TEXTURE_SOURCES_BOUND: AtomicU64 = AtomicU64::new(0);
static LAST_TABLES_DROPPED_INERT: AtomicU64 = AtomicU64::new(0);
static LAST_PS_TABLES_DROPPED_INERT: AtomicU64 = AtomicU64::new(0);
static LAST_FRAGMENT_TEXTURE_SOURCES_DROPPED: AtomicU64 = AtomicU64::new(0);
static FRAGMENT_SOURCE_UNREACHABLE: AtomicU64 = AtomicU64::new(0);

/// Fixed reason word of the last fragment texture-source drop.
///
/// A `&'static str` cannot live in an atomic, and a flush already takes the
/// device mutex; this one is written only when the word changes.
static LAST_FRAGMENT_SOURCE_DROP_REASON: Mutex<&'static str> = Mutex::new("none");

/// Per-stage ABI observability of the last recorded draw.
///
/// Always on (relaxed atomics), independent of the opt-in draw tracer, so
/// `telemetry.json` can answer gate G5/G11 on a boot that did not arm the
/// trace. Every field is a count or a fixed reason word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageAbiTelemetry {
    /// Stage layouts the last flush carried (bit 0 vertex … bit 4 compute).
    pub stage_mask: u8,
    /// Declared fragment parameters.
    pub ps_abi_parameters: u64,
    /// Fragment parameters a title write bound.
    pub ps_bound_parameters: u64,
    /// Bound fragment parameters of kind `Constant`.
    pub ps_bound_constants: u64,
    /// Bound fragment parameters of kind `CBV`/`SRV`/`UAV`.
    pub ps_bound_views: u64,
    /// Bound fragment parameters of kind `Table`/`Sampler`.
    pub ps_bound_tables: u64,
    /// Fragment texture/sampler sources the layout declares.
    pub fragment_texture_sources_declared: u64,
    /// Declared fragment sources a wire table bound.
    pub fragment_texture_sources_bound: u64,
    /// Declared fragment sources this draw could not bind.
    ///
    /// `declared > 0 && bound == 0` always pairs with a non-zero value here, so
    /// no draw leaves a declared source unbound without a counted reason; the
    /// word is [`Self::fragment_source_drop_reason`].
    pub fragment_texture_sources_dropped: u64,
    /// Fixed reason word for the drop: one of the `DROP_*` constants, or
    /// `none` when nothing was dropped.
    pub fragment_source_drop_reason: &'static str,
    /// Inert wire tables in the last flush.
    pub tables_dropped_inert: u64,
    /// Inert wire tables the fragment would have had to consume.
    pub ps_tables_dropped_inert: u64,
    /// Wire tables the fragment stage cannot address because it declares no
    /// table parameter (process-wide; see
    /// [`FRAGMENT_SOURCE_UNREACHABLE_REASON`]).
    pub fragment_source_unreachable: u64,
    /// Fixed reason word: the missing wire concept, or `none` when the count
    /// is zero.
    pub fragment_source_unreachable_reason: &'static str,
}

/// Records one wire table whose fragment texture/sampler source no declared
/// parameter can address.
pub fn note_fragment_source_unreachable() {
    FRAGMENT_SOURCE_UNREACHABLE.fetch_add(1, Ordering::Relaxed);
}

/// The per-stage ABI telemetry of the last recorded flush.
pub fn stage_abi_telemetry() -> StageAbiTelemetry {
    let unreachable = FRAGMENT_SOURCE_UNREACHABLE.load(Ordering::Relaxed);
    StageAbiTelemetry {
        stage_mask: LAST_STAGE_MASK.load(Ordering::Relaxed) as u8,
        ps_abi_parameters: LAST_PS_ABI_PARAMETERS.load(Ordering::Relaxed),
        ps_bound_parameters: LAST_FRAGMENT_BOUND_PARAMETERS.load(Ordering::Relaxed),
        ps_bound_constants: LAST_FRAGMENT_BOUND_CONSTANTS.load(Ordering::Relaxed),
        ps_bound_views: LAST_FRAGMENT_BOUND_VIEWS.load(Ordering::Relaxed),
        ps_bound_tables: LAST_FRAGMENT_BOUND_TABLES.load(Ordering::Relaxed),
        fragment_texture_sources_declared: LAST_FRAGMENT_TEXTURE_SOURCES_DECLARED
            .load(Ordering::Relaxed),
        fragment_texture_sources_bound: LAST_FRAGMENT_TEXTURE_SOURCES_BOUND.load(Ordering::Relaxed),
        fragment_texture_sources_dropped: LAST_FRAGMENT_TEXTURE_SOURCES_DROPPED
            .load(Ordering::Relaxed),
        fragment_source_drop_reason: *LAST_FRAGMENT_SOURCE_DROP_REASON
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()),
        tables_dropped_inert: LAST_TABLES_DROPPED_INERT.load(Ordering::Relaxed),
        ps_tables_dropped_inert: LAST_PS_TABLES_DROPPED_INERT.load(Ordering::Relaxed),
        fragment_source_unreachable: unreachable,
        fragment_source_unreachable_reason: if unreachable == 0 {
            "none"
        } else {
            FRAGMENT_SOURCE_UNREACHABLE_REASON
        },
    }
}

/// Clears the per-stage ABI telemetry.
///
/// Test-only: a proof that runs several ABI shapes in one process starts each
/// case from a clean count. Never called by the host.
#[doc(hidden)]
pub fn reset_stage_abi_telemetry() {
    LAST_STAGE_MASK.store(0, Ordering::Relaxed);
    LAST_PS_ABI_PARAMETERS.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_BOUND_PARAMETERS.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_BOUND_CONSTANTS.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_BOUND_VIEWS.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_BOUND_TABLES.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_TEXTURE_SOURCES_DECLARED.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_TEXTURE_SOURCES_BOUND.store(0, Ordering::Relaxed);
    LAST_FRAGMENT_TEXTURE_SOURCES_DROPPED.store(0, Ordering::Relaxed);
    *LAST_FRAGMENT_SOURCE_DROP_REASON
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = "none";
    LAST_TABLES_DROPPED_INERT.store(0, Ordering::Relaxed);
    LAST_PS_TABLES_DROPPED_INERT.store(0, Ordering::Relaxed);
    FRAGMENT_SOURCE_UNREACHABLE.store(0, Ordering::Relaxed);
}

/// Records one ABI flush's binding summary.
pub fn note_abi_binding(summary: AbiBindingSummary) {
    // Fragment counters describe the last *fragment* flush: a following
    // compute dispatch must not overwrite them with zeros.
    if summary.stage_mask & 0b10 != 0 {
        LAST_STAGE_MASK.store(u64::from(summary.stage_mask), Ordering::Relaxed);
        LAST_PS_ABI_PARAMETERS.store(u64::from(summary.ps_abi_parameters), Ordering::Relaxed);
        LAST_FRAGMENT_BOUND_PARAMETERS
            .store(u64::from(summary.fragment_bound_parameters), Ordering::Relaxed);
        LAST_FRAGMENT_BOUND_CONSTANTS
            .store(u64::from(summary.fragment_bound_constants), Ordering::Relaxed);
        LAST_FRAGMENT_BOUND_VIEWS.store(u64::from(summary.fragment_bound_views), Ordering::Relaxed);
        LAST_FRAGMENT_BOUND_TABLES
            .store(u64::from(summary.fragment_bound_tables), Ordering::Relaxed);
        LAST_FRAGMENT_TEXTURE_SOURCES_DECLARED.store(
            u64::from(summary.fragment_texture_sources_declared),
            Ordering::Relaxed,
        );
        LAST_FRAGMENT_TEXTURE_SOURCES_BOUND.store(
            u64::from(summary.fragment_texture_sources_bound),
            Ordering::Relaxed,
        );
        LAST_FRAGMENT_TEXTURE_SOURCES_DROPPED.store(
            u64::from(summary.fragment_texture_sources_dropped),
            Ordering::Relaxed,
        );
        let reason = if summary.fragment_texture_sources_dropped == 0 {
            "none"
        } else {
            summary.fragment_source_drop_reason
        };
        *LAST_FRAGMENT_SOURCE_DROP_REASON
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = reason;
        LAST_TABLES_DROPPED_INERT.store(u64::from(summary.tables_dropped_inert), Ordering::Relaxed);
        LAST_PS_TABLES_DROPPED_INERT
            .store(u64::from(summary.ps_tables_dropped_inert), Ordering::Relaxed);
    }
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
        ..AbiBindingSummary::default()
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
            ..AbiBindingSummary::default()
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
    reset_stage_abi_telemetry();
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

/// Converter ABI stage slots a draw trace may describe, in the same bit order
/// as the pipeline's stage-presence mask: vertex, fragment, mesh, object,
/// compute.
pub const ABI_STAGE_SLOTS: usize = 5;

/// Stable stage label for slot `index` of [`StageAbiShape`].
pub fn abi_stage_name(index: usize) -> &'static str {
    match index {
        0 => "vertex",
        1 => "fragment",
        2 => "mesh",
        3 => "object",
        4 => "compute",
        _ => "unknown",
    }
}

/// Bounded ABI shape of one stage layout for one encoded draw.
///
/// Counts and argument-buffer byte offsets only: no register values, handles
/// or payload bytes. `encoded_*` describes what the builder actually wrote, so
/// a declared parameter left at the zero default is visible as a declared /
/// encoded mismatch instead of an inferred one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageAbiShape {
    /// The selected pipeline carries this stage layout.
    pub present: bool,
    /// Declared parameters in the stage's top-level argument buffer.
    pub declared_parameters: u32,
    /// Declared parameters of kind `Constant`.
    pub declared_constants: u32,
    /// Declared parameters of kind `CBV`/`SRV`/`UAV`.
    pub declared_views: u32,
    /// Declared parameters of kind `Table`/`Sampler`.
    pub declared_tables: u32,
    /// Parameters the builder wrote for this draw (title writes and the
    /// host's MEMORY_SPACE zero defaults).
    pub encoded_parameters: u32,
    /// Bytes those writes covered.
    pub encoded_bytes: u32,
    /// First byte offset any write touched.
    pub encoded_start: u32,
    /// One past the last byte offset any write touched.
    pub encoded_end: u32,
    /// Offset of the parameter a bound root view filled (`0` with no bound
    /// view).
    pub bound_root_view_offset: u32,
    /// Reflected size of that parameter (`0` with no bound view).
    pub bound_root_view_size: u32,
}

/// What one ABI flush wrote into the top-level argument buffer(s).
///
/// A parameter is **bound** only when a title `SetRootConstants`,
/// `SetRootBufferView` or `SetDescriptorTable` wrote it. Every declared
/// parameter without such a write is **unbound** and is filled with the zero
/// default the builder starts from (`MEMORY_SPACE` host-owned parameters are
/// also counted here: they are not title writes).
///
/// The `fragment_*` fields answer gate G5/G11 of the Boot #32 audit per stage:
/// what the fragment stage declares, what it actually bound, and how many wire
/// tables that carry a texture/sampler source could not be bound at all.
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
    /// Bit mask of the stage layouts the flushed pipeline carries (bit 0
    /// vertex, bit 1 fragment, bit 2 mesh, bit 3 object, bit 4 compute).
    pub stage_mask: u8,
    /// Declared parameters of the fragment stage. `0` when the pipeline has no
    /// fragment layout.
    pub ps_abi_parameters: u32,
    /// Fragment parameters a title root write bound.
    pub fragment_bound_parameters: u32,
    /// Bound fragment parameters of kind `Constant`.
    pub fragment_bound_constants: u32,
    /// Bound fragment parameters of kind `CBV`/`SRV`/`UAV`.
    pub fragment_bound_views: u32,
    /// Bound fragment parameters of kind `Table`/`Sampler`.
    pub fragment_bound_tables: u32,
    /// Fragment table parameters whose range kind can carry a texture or
    /// sampler source (a reflected `Unspecified` table is included: converter
    /// reflection never names the range kind).
    pub fragment_texture_sources_declared: u32,
    /// Those declared fragment sources a wire table with a texture/sampler
    /// entry actually bound.
    pub fragment_texture_sources_bound: u32,
    /// Declared fragment sources this draw could not bind.
    ///
    /// `declared > 0 && bound == 0` always pairs with a non-zero value here so a
    /// declared source is never unbound without a counted reason; the word is
    /// [`Self::fragment_source_drop_reason`].
    pub fragment_texture_sources_dropped: u32,
    /// Fixed reason word of the drop (one of the `DROP_*` constants, or `none`).
    pub fragment_source_drop_reason: &'static str,
    /// Wire tables this draw left inert (the bound pipeline declares no root
    /// parameter that can address them).
    pub tables_dropped_inert: u32,
    /// Inert wire tables the fragment stage would have had to consume.
    pub ps_tables_dropped_inert: u32,
    /// Inert wire tables while the fragment stage declares no table parameter
    /// at all: the fragment has no root-parameter path to them, and the wire
    /// carries no descriptor heap. The missing concept is named by
    /// [`FRAGMENT_SOURCE_UNREACHABLE_REASON`].
    pub fragment_source_unreachable: u32,
    /// Per-stage declared/encoded shape, indexed by [`abi_stage_name`].
    pub stages: [StageAbiShape; ABI_STAGE_SLOTS],
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
    /// Bit mask of the stage layouts the pipeline carries (bit 0 vertex, bit 1
    /// fragment, bit 2 mesh, bit 3 object, bit 4 compute).
    pub stage_mask: u8,
    /// Declared fragment parameters.
    pub ps_abi_parameters: u32,
    /// Fragment parameters a title root write bound.
    pub ps_bound_parameters: u32,
    /// Bound fragment parameters of kind `Constant`.
    pub ps_bound_constants: u32,
    /// Bound fragment parameters of kind `CBV`/`SRV`/`UAV`.
    pub ps_bound_views: u32,
    /// Bound fragment parameters of kind `Table`/`Sampler`.
    pub ps_bound_tables: u32,
    /// Fragment table parameters that can carry a texture/sampler source.
    pub fragment_texture_sources_declared: u32,
    /// Those sources a wire table actually bound.
    pub fragment_texture_sources_bound: u32,
    /// Declared fragment sources this draw could not bind.
    pub fragment_texture_sources_dropped: u32,
    /// Fixed reason word of the drop, or `none`.
    pub fragment_source_drop_reason: &'static str,
    /// Wire tables this draw left inert.
    pub tables_dropped_inert: u32,
    /// Inert wire tables the fragment would have had to consume.
    pub ps_tables_dropped_inert: u32,
    /// Wire tables carrying a fragment texture/sampler source with no declared
    /// fragment parameter.
    pub fragment_source_unreachable: u32,
    /// Per-stage declared/encoded ABI shape, indexed by [`abi_stage_name`].
    pub stages: [StageAbiShape; ABI_STAGE_SLOTS],
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
        // Per-stage ABI observability (gate G5/G11): what each active stage
        // declares, what the argument-buffer builder actually wrote, and the
        // parameter a bound root view filled. Counts and offsets only.
        out.push_str(&format!(
            " stage_mask={} ps_abi_parameters={} ps_bound_parameters={} \
             ps_bound_kinds=constants:{},view:{},table:{} \
             fragment_texture_sources_declared={} fragment_texture_sources_bound={} \
             fragment_texture_sources_dropped={} fragment_source_drop_reason={} \
             tables_dropped_inert={} ps_tables_dropped_inert={} \
             fragment_source_unreachable={}",
            self.stage_mask,
            self.ps_abi_parameters,
            self.ps_bound_parameters,
            self.ps_bound_constants,
            self.ps_bound_views,
            self.ps_bound_tables,
            self.fragment_texture_sources_declared,
            self.fragment_texture_sources_bound,
            self.fragment_texture_sources_dropped,
            self.fragment_source_drop_reason,
            self.tables_dropped_inert,
            self.ps_tables_dropped_inert,
            self.fragment_source_unreachable
        ));
        for (index, shape) in self.stages.iter().enumerate() {
            if !shape.present {
                continue;
            }
            let span = if shape.encoded_parameters == 0 {
                "none".to_string()
            } else {
                format!("{}..{}", shape.encoded_start, shape.encoded_end)
            };
            out.push_str(&format!(
                " abi_stage={} params=constants:{},view:{},table:{} wrote={} bytes={} span={}",
                abi_stage_name(index),
                shape.declared_constants,
                shape.declared_views,
                shape.declared_tables,
                shape.encoded_parameters,
                shape.encoded_bytes,
                span
            ));
            if shape.bound_root_view_size > 0 {
                out.push_str(&format!(
                    " bound_view={}:{}",
                    shape.bound_root_view_offset, shape.bound_root_view_size
                ));
            } else {
                out.push_str(" bound_view=none");
            }
        }
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
                // ADR-068: a described draw is a draw target even when its pass
                // is never sampled (the record bound saturated first).
                if let Some(target) = record.target {
                    log.rts.note_draw_target(target);
                }
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
        self.next_pass_with_clear(None)
    }

    /// [`Self::next_pass`] carrying the sampled attachment's clear state.
    ///
    /// ADR-068: the clear state is what separates "the shader drew zeros" from
    /// "the shader drew non-zero pixels that the pass load/clear never
    /// preserved".
    pub fn next_pass_with_clear(&mut self, clear: Option<TraceClear>) -> bool {
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
                clear,
            };
            with_log(|log| {
                log.passes_sampled = log.passes_sampled.saturating_add(1);
                log.sample_context = Some(context);
                // ADR-068: the pass is a per-RT event even before its pixels
                // are read; a cleared pass of an already-written target is the
                // "erased by a later clear" class.
                if let Some(target) = context.target {
                    log.rts.note_pass(target, clear);
                }
            });
        }
        true
    }
}

/// Bounded summary of a sampled `RGBA8`/`BGRA8` attachment.
///
/// ADR-068: the last five fields are the "written after last clear" evidence.
/// They separate (a) a shader that drew zeros from (b) a shader that drew
/// non-zero pixels the window's clear never preserved, without a second boot.
/// `Eq` is not derived because a clear colour is `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PixelStats {
    /// Bytes that are not zero (the real readback count).
    pub nonzero_bytes: u64,
    /// Bytes sampled.
    pub sampled_bytes: u64,
    /// Distinct byte values, saturated at 256.
    pub distinct_values: u32,
    /// Distinct RGBA texels in the sampled grid, saturated at
    /// [`MAX_DISTINCT_PIXELS`].
    ///
    /// The pixel-level companion of [`Self::distinct_values`]: a uniform red
    /// grid is one distinct pixel but two distinct byte values, so the two
    /// counters are never conflated.
    pub distinct_pixels_after_draws: u32,
    /// Draws/dispatches of the sampling window that follow the window's clear;
    /// `0` when the pass loads instead of clearing.
    pub draws_since_clear: u32,
    /// Colour the window's clear wrote; `None` when the pass loaded.
    pub clear_color: Option<[f64; 4]>,
    /// Non-zero bytes the clear alone accounts for: four bytes per sampled
    /// texel when any clear channel is non-zero, zero when every channel is
    /// zero, `None` on a load (the previous contents are unknown).
    pub nonzero_bytes_after_clear: Option<u64>,
    /// Sampling-opportunity counter of the pass this summary belongs to; `0`
    /// for a summary produced outside the device trace path.
    pub pass: u32,
    /// Draws/dispatches the sampled pass encoded; `0` when uncounted.
    pub pass_draws: u32,
    /// Attachment that was sampled, when the trace named one.
    pub target: Option<TraceTarget>,
}

impl PixelStats {
    /// Non-zero bytes the draws left: the real readback count.
    ///
    /// This is [`Self::nonzero_bytes`] under the ADR-068 name, so a reader of
    /// `telemetry.json` never has to infer which count is the drawn one.
    pub fn nonzero_bytes_after_draws(&self) -> u64 {
        self.nonzero_bytes
    }

    /// The pass line (when known) followed by the pixel summary line.
    ///
    /// The pass line always precedes its own summary so a reader can attribute
    /// `nonzero_bytes` to a pass without correlating unrelated log lines. Every
    /// pre-ADR-068 `key=value` token keeps its spelling and position; the new
    /// evidence is appended.
    pub fn line(&self) -> String {
        let clear_color = self.clear_color.map_or_else(
            || "none".to_string(),
            |[r, g, b, a]| format!("{r:.3},{g:.3},{b:.3},{a:.3}"),
        );
        let nonzero_after_clear = self
            .nonzero_bytes_after_clear
            .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let summary = format!(
            "draw_trace_rt nonzero_bytes={} sampled_bytes={} distinct_values={} \
             draws_since_clear={} clear_color={} nonzero_bytes_after_clear={} \
             nonzero_bytes_after_draws={} distinct_pixels_after_draws={}",
            self.nonzero_bytes,
            self.sampled_bytes,
            self.distinct_values,
            self.draws_since_clear,
            clear_color,
            nonzero_after_clear,
            self.nonzero_bytes_after_draws(),
            self.distinct_pixels_after_draws,
        );
        if self.pass == 0 {
            return summary;
        }
        format!(
            "draw_trace_pass pass={} draws={} rt={} draws_since_clear={} clear={}\n{}",
            self.pass,
            self.pass_draws,
            self.target
                .as_ref()
                .map_or_else(|| "none".to_string(), target_name),
            self.draws_since_clear,
            if self.clear_color.is_some() {
                "clear"
            } else {
                "load"
            },
            summary
        )
    }
}

/// ADR-068: derives the clear evidence of one sample window from its context.
///
/// A window that clears attributes every sampled non-zero texel byte to the
/// clear colour: four bytes per texel when any channel is non-zero, zero when
/// the clear is fully zero. A window that loads cannot be attributed, so the
/// analytic count is `None` and `draws_since_clear` is `0` (no clear in the
/// window to count draws after).
fn apply_clear(stats: &mut PixelStats, clear: Option<TraceClear>) {
    let Some(clear) = clear.filter(|clear| clear.is_clear) else {
        stats.draws_since_clear = 0;
        stats.clear_color = None;
        stats.nonzero_bytes_after_clear = None;
        return;
    };
    stats.draws_since_clear = stats.pass_draws;
    stats.clear_color = Some(clear.color);
    stats.nonzero_bytes_after_clear = Some(if clear.color.iter().any(|channel| *channel != 0.0) {
        stats.sampled_bytes
    } else {
        0
    });
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
                // ADR-068: attribute the sampled bytes to the window's clear.
                apply_clear(&mut stats, context.clear);
            }
            if let Some(target) = stats.target {
                log.rts.note_sample(target, &stats);
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
    // Distinct RGBA texels, saturated at MAX_DISTINCT_PIXELS. A uniform grid is
    // one distinct pixel even when its bytes are several distinct values.
    let mut pixels: Vec<[u8; 4]> = Vec::new();
    for texel in bytes.chunks_exact(4) {
        let pixel = [texel[0], texel[1], texel[2], texel[3]];
        if pixels.contains(&pixel) {
            continue;
        }
        pixels.push(pixel);
        if pixels.len() as u32 >= MAX_DISTINCT_PIXELS {
            break;
        }
    }
    PixelStats {
        nonzero_bytes: nonzero,
        sampled_bytes: bytes.len() as u64,
        distinct_values: distinct,
        distinct_pixels_after_draws: pixels.len() as u32,
        ..PixelStats::default()
    }
}

// ---------------------------------------------------------------------------
// ADR-068: bounded per-RT "written after last clear" history.
//
// Boot #36 sampled 4 of 156 passes and reported `non_black_pixels = 0` without
// distinguishing "the shader drew zeros" from "the shader drew non-zero pixels
// a later clear erased" from "the shader drew into an RT nothing ever showed".
// The history below answers that per render-target, from the passes the tracer
// already samples, with no extra readback and no unbounded state.
// ---------------------------------------------------------------------------

/// Fixed class word: the RT's sample showed non-zero drawn pixels after a clear.
pub const RT_CLASS_WRITTEN: &str = "written-after-clear";

/// Fixed class word: a later pass cleared an RT that had already been written.
pub const RT_CLASS_ERASED: &str = "erased-by-later-clear";

/// Fixed class word: the RT was named as a draw target but never sampled.
pub const RT_CLASS_NEVER_SAMPLED: &str = "never-sampled";

/// Fixed class word: sampled after a clear and read back all zero.
pub const RT_CLASS_ZERO: &str = "drew-zeros";

/// Fixed class word: only ever sampled on a load, so attribution is unknown.
pub const RT_CLASS_LOADED: &str = "loaded-unknown";

/// Fixed class word: no draw-trace evidence exists for this texture.
pub const RT_CLASS_UNTRACED: &str = "not-traced";

/// One render target's bounded written-after-clear record.
///
/// Every field is a count, a fixed word or a clear colour: no texture address,
/// title path or pixel payload is ever published.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RtWriteRecord {
    /// One-based first-seen ordinal published instead of the raw handle.
    rt: u32,
    /// Attachment geometry the first time this target was seen.
    target: Option<TraceTarget>,
    /// Traced draw/dispatch records that named this target.
    draw_notes: u32,
    /// Sampling opportunities (passes) this target was offered with.
    passes: u32,
    /// Passes of this target whose pixels were read back.
    sampled_passes: u32,
    /// Offered passes that began with a clear of this target.
    clears: u32,
    /// The most recent clear colour this target was written with.
    last_clear: Option<TraceClear>,
    /// Draws the last sample counted after its clear.
    last_draws_since_clear: u32,
    /// Analytic clear-only non-zero bytes of the last sample.
    last_nonzero_bytes_after_clear: Option<u64>,
    /// Real readback non-zero bytes of the last sample.
    last_nonzero_bytes_after_draws: u64,
    /// Distinct RGBA texels of the last sample.
    last_distinct_pixels_after_draws: u32,
    /// A sample read back non-zero bytes drawn after a clear.
    written_after_clear: bool,
    /// Samples that read back all zero after a clear.
    zero_after_clear: u32,
    /// A later cleared pass of this target followed a written sample.
    erased_by_later_clear: bool,
}

impl RtWriteRecord {
    /// The one-word class the record currently supports.
    fn class(&self) -> &'static str {
        if self.erased_by_later_clear {
            RT_CLASS_ERASED
        } else if self.sampled_passes == 0 && (self.passes > 0 || self.draw_notes > 0) {
            RT_CLASS_NEVER_SAMPLED
        } else if self.written_after_clear {
            RT_CLASS_WRITTEN
        } else if self.zero_after_clear > 0 {
            RT_CLASS_ZERO
        } else if self.sampled_passes > 0 {
            RT_CLASS_LOADED
        } else {
            RT_CLASS_UNTRACED
        }
    }
}

/// The bounded per-RT history, keyed by texture handle.
///
/// It is a plain value so the pure classification can be proven without a GPU
/// or the process-wide tracer. [`MAX_RT_HISTORY`] caps distinct targets; an
/// existing key is always updated, so a cap hit never hides later evidence for
/// a target already retained.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtHistory {
    entries: Vec<RtWriteRecord>,
    dropped: u64,
}

impl RtHistory {
    /// Retained distinct targets (`<= MAX_RT_HISTORY`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no target was retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Distinct targets that did not fit [`MAX_RT_HISTORY`].
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Retained record for `texture`, if any.
    fn record(&self, texture: usize) -> Option<&RtWriteRecord> {
        self.entries
            .iter()
            .find(|entry| entry.target.is_some_and(|target| target.texture == texture))
    }

    /// Finds or creates the record for `target`.
    ///
    /// Returns `None` when the history is full and the target is new; the
    /// caller's event is then only counted as dropped.
    fn upsert(&mut self, target: TraceTarget) -> Option<usize> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.target.is_some_and(|seen| seen.texture == target.texture))
        {
            return Some(index);
        }
        if self.entries.len() >= MAX_RT_HISTORY {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        let rt = self.entries.len() as u32 + 1;
        self.entries.push(RtWriteRecord {
            rt,
            target: Some(target),
            ..RtWriteRecord::default()
        });
        Some(self.entries.len() - 1)
    }

    /// Records that one traced draw/dispatch named `target`.
    pub fn note_draw_target(&mut self, target: TraceTarget) -> Option<u32> {
        let index = self.upsert(target)?;
        let entry = &mut self.entries[index];
        entry.draw_notes = entry.draw_notes.saturating_add(1);
        Some(entry.rt)
    }

    /// Records one sampling opportunity of `target` with its clear state.
    ///
    /// A cleared pass of a target that was already written is exactly the
    /// "erased by a later clear" class, and it is recorded here even when the
    /// later pass's own pixels are zero.
    pub fn note_pass(&mut self, target: TraceTarget, clear: Option<TraceClear>) -> Option<u32> {
        let index = self.upsert(target)?;
        let entry = &mut self.entries[index];
        entry.passes = entry.passes.saturating_add(1);
        if let Some(clear) = clear.filter(|clear| clear.is_clear) {
            entry.clears = entry.clears.saturating_add(1);
            entry.last_clear = Some(clear);
            if entry.written_after_clear {
                entry.erased_by_later_clear = true;
            }
        }
        Some(entry.rt)
    }

    /// Records the readback summary of one sampled pass of `target`.
    pub fn note_sample(&mut self, target: TraceTarget, stats: &PixelStats) -> Option<u32> {
        let index = self.upsert(target)?;
        let entry = &mut self.entries[index];
        entry.sampled_passes = entry.sampled_passes.saturating_add(1);
        entry.last_draws_since_clear = stats.draws_since_clear;
        if let Some(color) = stats.clear_color {
            entry.last_clear = Some(TraceClear {
                is_clear: true,
                color,
            });
        }
        entry.last_nonzero_bytes_after_clear = stats.nonzero_bytes_after_clear;
        entry.last_nonzero_bytes_after_draws = stats.nonzero_bytes;
        entry.last_distinct_pixels_after_draws = stats.distinct_pixels_after_draws;
        if stats.draws_since_clear > 0 {
            if stats.nonzero_bytes > 0 {
                entry.written_after_clear = true;
            } else {
                entry.zero_after_clear = entry.zero_after_clear.saturating_add(1);
            }
        }
        Some(entry.rt)
    }

    /// One target's published evidence, by texture handle.
    pub fn entry(&self, texture: usize) -> Option<RtWriteEntry> {
        self.record(texture).map(RtWriteEntry::from_record)
    }

    /// One target's class word, by texture handle.
    pub fn class_of(&self, texture: usize) -> Option<&'static str> {
        self.record(texture).map(RtWriteRecord::class)
    }

    /// The whole bounded history as the telemetry value.
    pub fn evidence(&self) -> RtWriteEvidence {
        let rts: Vec<RtWriteEntry> = self.entries.iter().map(RtWriteEntry::from_record).collect();
        RtWriteEvidence {
            rt_traced: rts.len() as u64,
            rt_sampled: rts.iter().filter(|entry| entry.sampled_passes > 0).count() as u64,
            rt_written_after_clear: rts
                .iter()
                .filter(|entry| entry.written_after_clear)
                .count() as u64,
            rt_erased_by_later_clear: rts
                .iter()
                .filter(|entry| entry.erased_by_later_clear)
                .count() as u64,
            rt_never_sampled: rts
                .iter()
                .filter(|entry| entry.class == RT_CLASS_NEVER_SAMPLED)
                .count() as u64,
            rt_dropped: self.dropped,
            rts,
        }
    }
}

/// One render target's published written-after-clear evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RtWriteEntry {
    /// One-based first-seen ordinal (never the raw texture address).
    pub rt: u32,
    /// Attachment width in texels.
    pub width: u32,
    /// Attachment height in texels.
    pub height: u32,
    /// Attachment pixel-format name.
    pub format: &'static str,
    /// The target is a registered swapchain surface.
    pub swapchain: bool,
    /// Traced draw/dispatch records that named this target.
    pub draw_notes: u32,
    /// Sampling opportunities this target was offered with.
    pub passes: u32,
    /// Passes of this target whose pixels were read back.
    pub sampled_passes: u32,
    /// Offered passes that began with a clear of this target.
    pub clears: u32,
    /// Last clear colour, `None` when every pass loaded.
    pub clear_color: Option<[f64; 4]>,
    /// Draws the last sample counted after its clear.
    pub draws_since_clear: u32,
    /// Analytic clear-only non-zero bytes of the last sample.
    pub nonzero_bytes_after_clear: Option<u64>,
    /// Real readback non-zero bytes of the last sample.
    pub nonzero_bytes_after_draws: u64,
    /// Distinct RGBA texels of the last sample.
    pub distinct_pixels_after_draws: u32,
    /// A sample read back non-zero bytes drawn after a clear.
    pub written_after_clear: bool,
    /// A later cleared pass followed a written sample.
    pub erased_by_later_clear: bool,
    /// Fixed class word derived from the flags above.
    pub class: &'static str,
}

impl RtWriteEntry {
    fn from_record(record: &RtWriteRecord) -> Self {
        let target = record.target.unwrap_or_else(default_target);
        Self {
            rt: record.rt,
            width: target.width,
            height: target.height,
            format: target.format,
            swapchain: target.swapchain,
            draw_notes: record.draw_notes,
            passes: record.passes,
            sampled_passes: record.sampled_passes,
            clears: record.clears,
            clear_color: record.last_clear.map(|clear| clear.color),
            draws_since_clear: record.last_draws_since_clear,
            nonzero_bytes_after_clear: record.last_nonzero_bytes_after_clear,
            nonzero_bytes_after_draws: record.last_nonzero_bytes_after_draws,
            distinct_pixels_after_draws: record.last_distinct_pixels_after_draws,
            written_after_clear: record.written_after_clear,
            erased_by_later_clear: record.erased_by_later_clear,
            class: record.class(),
        }
    }
}

/// A zeroed target placeholder: a record always has one, this only satisfies
/// the type without inventing geometry.
fn default_target() -> TraceTarget {
    TraceTarget {
        texture: 0,
        width: 0,
        height: 0,
        format: "none",
        swapchain: false,
    }
}

/// Bounded written-after-clear evidence for every retained render target.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtWriteEvidence {
    /// Distinct targets retained (`<= MAX_RT_HISTORY`).
    pub rt_traced: u64,
    /// Retained targets that produced at least one pixel sample.
    pub rt_sampled: u64,
    /// Targets whose sample showed non-zero drawn bytes after a clear.
    pub rt_written_after_clear: u64,
    /// Targets a later cleared pass followed after they had been written.
    pub rt_erased_by_later_clear: u64,
    /// Targets named as draw targets but never sampled.
    pub rt_never_sampled: u64,
    /// Targets that did not fit [`MAX_RT_HISTORY`].
    pub rt_dropped: u64,
    /// Per-target rows in first-seen order.
    pub rts: Vec<RtWriteEntry>,
}

impl RtWriteEvidence {
    /// The evidence as the `draw_trace` members (no surrounding braces).
    ///
    /// Field names, in emitted order: `rt_traced`, `rt_sampled`,
    /// `rt_written_after_clear`, `rt_erased_by_later_clear`,
    /// `rt_never_sampled`, `rt_dropped`, `rts`. Each row carries `rt`, `width`,
    /// `height`, `format`, `swapchain`, `draw_notes`, `passes`, `sampled`,
    /// `clears`, `clear_color`, `draws_since_clear`,
    /// `nonzero_bytes_after_clear`, `nonzero_bytes_after_draws`,
    /// `distinct_pixels_after_draws` and `class`.
    pub fn json_members(&self) -> String {
        let rts = self
            .rts
            .iter()
            .map(|entry| {
                let clear_color = entry.clear_color.map_or_else(
                    || "null".to_string(),
                    |[r, g, b, a]| format!("[{r:.3}, {g:.3}, {b:.3}, {a:.3}]"),
                );
                let nonzero_after_clear = entry
                    .nonzero_bytes_after_clear
                    .map_or_else(|| "null".to_string(), |bytes| bytes.to_string());
                format!(
                    "{{\"rt\": {}, \"width\": {}, \"height\": {}, \"format\": \"{}\", \
                     \"swapchain\": {}, \"draw_notes\": {}, \"passes\": {}, \
                     \"sampled\": {}, \"clears\": {}, \"clear_color\": {}, \
                     \"draws_since_clear\": {}, \"nonzero_bytes_after_clear\": {}, \
                     \"nonzero_bytes_after_draws\": {}, \
                     \"distinct_pixels_after_draws\": {}, \"class\": \"{}\"}}",
                    entry.rt,
                    entry.width,
                    entry.height,
                    entry.format,
                    entry.swapchain,
                    entry.draw_notes,
                    entry.passes,
                    entry.sampled_passes,
                    entry.clears,
                    clear_color,
                    entry.draws_since_clear,
                    nonzero_after_clear,
                    entry.nonzero_bytes_after_draws,
                    entry.distinct_pixels_after_draws,
                    entry.class,
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "\"rt_traced\": {}, \"rt_sampled\": {}, \"rt_written_after_clear\": {}, \
             \"rt_erased_by_later_clear\": {}, \"rt_never_sampled\": {}, \
             \"rt_dropped\": {}, \"rts\": [{}]",
            self.rt_traced,
            self.rt_sampled,
            self.rt_written_after_clear,
            self.rt_erased_by_later_clear,
            self.rt_never_sampled,
            self.rt_dropped,
            rts
        )
    }

    /// The evidence as one JSON object.
    pub fn json_value(&self) -> String {
        format!("{{{}}}", self.json_members())
    }
}

/// The process-wide per-RT written-after-clear evidence.
///
/// It is empty while the tracer is off: reading it never arms the tracer.
pub fn rt_write_evidence() -> RtWriteEvidence {
    if !enabled() {
        return RtWriteEvidence::default();
    }
    with_log(|log| log.rts.evidence())
}

/// One render target's evidence, looked up by texture handle.
///
/// The handle is used only as a key here; the returned value carries the
/// one-based ordinal, never the address.
pub fn rt_evidence_for(texture: usize) -> Option<RtWriteEntry> {
    if !enabled() || texture == 0 {
        return None;
    }
    with_log(|log| log.rts.entry(texture))
}

/// One render target's class word, looked up by texture handle.
pub fn rt_class_for(texture: usize) -> Option<&'static str> {
    if !enabled() || texture == 0 {
        return None;
    }
    with_log(|log| log.rts.class_of(texture))
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
                stage_mask: 0,
                ps_abi_parameters: 0,
                ps_bound_parameters: 0,
                ps_bound_constants: 0,
                ps_bound_views: 0,
                ps_bound_tables: 0,
                fragment_texture_sources_declared: 0,
                fragment_texture_sources_bound: 0,
                fragment_texture_sources_dropped: 0,
                fragment_source_drop_reason: "none",
                tables_dropped_inert: 0,
                ps_tables_dropped_inert: 0,
                fragment_source_unreachable: 0,
                stages: [StageAbiShape::default(); ABI_STAGE_SLOTS],
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
                    stage_mask: 0,
                    ps_abi_parameters: 0,
                    ps_bound_parameters: 0,
                    ps_bound_constants: 0,
                    ps_bound_views: 0,
                    ps_bound_tables: 0,
                    fragment_texture_sources_declared: 0,
                    fragment_texture_sources_bound: 0,
                    fragment_texture_sources_dropped: 0,
                    fragment_source_drop_reason: "none",
                    tables_dropped_inert: 0,
                    ps_tables_dropped_inert: 0,
                    fragment_source_unreachable: 0,
                    stages: [StageAbiShape::default(); ABI_STAGE_SLOTS],
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
            distinct_pixels_after_draws: 2,
            draws_since_clear: 2,
            clear_color: Some([0.0, 0.0, 0.0, 0.0]),
            nonzero_bytes_after_clear: Some(0),
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
            Some(
                "draw_trace_pass pass=4 draws=2 rt=offscreen 64x64 rgba8unorm \
                 draws_since_clear=2 clear=clear"
            )
        );
        assert_eq!(
            lines.next(),
            Some(
                "draw_trace_rt nonzero_bytes=12 sampled_bytes=16384 distinct_values=3 \
                 draws_since_clear=2 clear_color=0.000,0.000,0.000,0.000 \
                 nonzero_bytes_after_clear=0 nonzero_bytes_after_draws=12 \
                 distinct_pixels_after_draws=2"
            )
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
            "draw_trace_rt nonzero_bytes=0 sampled_bytes=4096 distinct_values=1 \
             draws_since_clear=0 clear_color=none nonzero_bytes_after_clear=unknown \
             nonzero_bytes_after_draws=0 distinct_pixels_after_draws=0"
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
            stage_mask: 0b11,
            ps_abi_parameters: 2,
            ps_bound_parameters: 1,
            ps_bound_constants: 0,
            ps_bound_views: 0,
            ps_bound_tables: 1,
            fragment_texture_sources_declared: 1,
            fragment_texture_sources_bound: 1,
            fragment_texture_sources_dropped: 0,
            fragment_source_drop_reason: "none",
            tables_dropped_inert: 0,
            ps_tables_dropped_inert: 0,
            fragment_source_unreachable: 0,
            stages: {
                let mut stages = [StageAbiShape::default(); ABI_STAGE_SLOTS];
                stages[0] = StageAbiShape {
                    present: true,
                    declared_parameters: 1,
                    declared_constants: 1,
                    encoded_parameters: 1,
                    encoded_bytes: 16,
                    encoded_start: 0,
                    encoded_end: 16,
                    ..StageAbiShape::default()
                };
                stages[1] = StageAbiShape {
                    present: true,
                    declared_parameters: 2,
                    declared_views: 1,
                    declared_tables: 1,
                    encoded_parameters: 2,
                    encoded_bytes: 32,
                    encoded_start: 0,
                    encoded_end: 32,
                    bound_root_view_offset: 8,
                    bound_root_view_size: 24,
                    ..StageAbiShape::default()
                };
                stages
            },
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
        assert!(line.contains("stage_mask=3"));
        assert!(line.contains("ps_abi_parameters=2"));
        assert!(line.contains("ps_bound_kinds=constants:0,view:0,table:1"));
        assert!(line.contains("fragment_texture_sources_declared=1"));
        assert!(line.contains("fragment_texture_sources_bound=1"));
        assert!(line.contains("abi_stage=vertex params=constants:1,view:0,table:0"));
        assert!(line.contains("abi_stage=fragment params=constants:0,view:1,table:1"));
        assert!(line.contains("bound_view=8:24"));
        assert!(line.contains("rt=offscreen 1920x1080 rgba8unorm"));
    }

    #[test]
    fn the_stage_abi_telemetry_names_the_unreachable_fragment_source() {
        reset_stage_abi_telemetry();
        note_abi_binding(AbiBindingSummary {
            stage_mask: 0b10,
            ps_abi_parameters: 0,
            fragment_texture_sources_declared: 0,
            tables_dropped_inert: 2,
            ps_tables_dropped_inert: 2,
            fragment_source_unreachable: 2,
            ..AbiBindingSummary::default()
        });
        note_fragment_source_unreachable();
        let telemetry = stage_abi_telemetry();
        assert_eq!(telemetry.stage_mask, 0b10);
        assert_eq!(telemetry.ps_abi_parameters, 0);
        assert_eq!(telemetry.tables_dropped_inert, 2);
        assert_eq!(telemetry.fragment_source_unreachable, 1);
        assert_eq!(
            telemetry.fragment_source_unreachable_reason,
            FRAGMENT_SOURCE_UNREACHABLE_REASON
        );
        let json = descriptor_table_refusal_census().json_value();
        assert!(json.contains("\"stage_abi\""));
        assert!(json.contains("\"ps_abi_parameters\": 0"));
        assert!(json.contains("\"fragment_texture_sources_bound\": 0"));
        assert!(json.contains("\"fragment_source_unreachable\": 1"));
        reset_stage_abi_telemetry();
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
            mix_class: "single",
        }
    }

    fn mix_key(
        wire_set: u16,
        mix_class: &'static str,
        outcome: &'static str,
    ) -> DescriptorTableMixKey {
        DescriptorTableMixKey {
            pipeline_id: 3,
            wire_set,
            mix_class,
            outcome,
        }
    }

    /// The mix census names the exact entry combination and how the host
    /// encoded it, counting repeats once per key.
    #[test]
    fn the_binding_drop_census_names_the_declared_shape_and_reason() {
        reset_descriptor_table_refusal_census();
        let key = DescriptorTableBindingDropKey {
            pipeline_id: 3,
            wire_set: 0,
            declared_kind: "Unspecified",
            declared_tables: 1,
            declared_slots: 1,
            mix_class: "buffer+texture",
            reason: DROP_DECLARED_KIND_REJECTS_PARTITIONS,
        };
        note_descriptor_table_binding_drop(key);
        note_descriptor_table_binding_drop(key);
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, 1);
        assert!(census.mix_keys.is_empty(), "a drop key is its own class");
        assert_eq!(census.binding_drops.len(), 1);
        assert_eq!(census.binding_drops[0].count, 2);
        assert_eq!(
            census.binding_drops[0].key,
            DescriptorTableBindingDropKey { ..key }
        );
        let json = census.json_value();
        assert!(json.contains("\"binding_drops\""));
        assert!(json.contains("\"declared_kind\": \"Unspecified\""));
        assert!(json.contains("\"declared_slots\": 1"));
        assert!(json.contains("\"reason\": \"declared-kind-rejects-partitions\""));
        reset_descriptor_table_refusal_census();
        assert!(descriptor_table_refusal_census().is_empty());
    }

    #[test]
    fn the_stage_abi_telemetry_names_a_declared_source_drop() {
        reset_stage_abi_telemetry();
        note_abi_binding(AbiBindingSummary {
            stage_mask: 0b10,
            ps_abi_parameters: 1,
            fragment_texture_sources_declared: 1,
            fragment_texture_sources_bound: 0,
            fragment_texture_sources_dropped: 1,
            fragment_source_drop_reason: DROP_FRAGMENT_SOURCE_RESIDUE,
            ..AbiBindingSummary::default()
        });
        let telemetry = stage_abi_telemetry();
        assert_eq!(telemetry.fragment_texture_sources_declared, 1);
        assert_eq!(telemetry.fragment_texture_sources_bound, 0);
        assert_eq!(telemetry.fragment_texture_sources_dropped, 1);
        assert_eq!(
            telemetry.fragment_source_drop_reason,
            DROP_FRAGMENT_SOURCE_RESIDUE
        );
        let json = descriptor_table_refusal_census().json_value();
        assert!(json.contains("\"fragment_texture_sources_dropped\": 1"));
        assert!(json.contains("\"fragment_source_drop_reason\": \"fragment-source-residue\""));
        reset_stage_abi_telemetry();
    }

    #[test]
    fn the_mix_census_counts_by_class_and_outcome() {
        reset_descriptor_table_refusal_census();
        note_descriptor_table_mix(mix_key(0, "buffer+texture", OUTCOME_SPLIT));
        note_descriptor_table_mix(mix_key(0, "buffer+texture", OUTCOME_SPLIT));
        note_descriptor_table_mix(mix_key(0, "buffer+texture", OUTCOME_INERT));
        note_descriptor_table_mix(mix_key(0, "texture+sampler", OUTCOME_REFUSED));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, 3);
        assert!(census.entries.is_empty());
        assert!(census.kind_mismatches.is_empty());
        assert_eq!(census.mix_keys.len(), 3);
        assert_eq!(census.mix_keys[0].count, 2);
        assert_eq!(census.mix_keys[0].key.mix_class, "buffer+texture");
        assert_eq!(census.mix_keys[0].key.outcome, OUTCOME_SPLIT);
        assert_eq!(census.mix_keys[1].key.outcome, OUTCOME_INERT);
        assert_eq!(census.mix_keys[2].key.mix_class, "texture+sampler");
        assert_eq!(census.mix_keys[2].key.outcome, OUTCOME_REFUSED);
        let json = census.json_value();
        assert!(json.contains("\"mix_keys\""));
        assert!(json.contains("\"mix_class\": \"buffer+texture\""));
        assert!(json.contains("\"outcome\": \"split\""));
        assert!(json.contains("\"outcome\": \"inert\""));
        reset_descriptor_table_refusal_census();
        assert!(descriptor_table_refusal_census().is_empty());
    }

    #[test]
    fn mix_keys_share_the_resolution_cap() {
        reset_descriptor_table_refusal_census();
        for set in 0..MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u16 {
            note_descriptor_table_refusal(key(set));
        }
        note_descriptor_table_mix(mix_key(0, "buffer+texture", OUTCOME_SPLIT));
        let census = descriptor_table_refusal_census();
        assert_eq!(census.distinct_keys, MAX_DESCRIPTOR_TABLE_REFUSAL_KEYS as u64);
        assert!(census.mix_keys.is_empty(), "the cap is shared");
        assert_eq!(census.omitted, 1);
        reset_descriptor_table_refusal_census();
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

    // -----------------------------------------------------------------------
    // ADR-068: per-RT written-after-last-clear evidence. Every case is pure:
    // no GPU, no environment flag, no process-wide state.
    // -----------------------------------------------------------------------

    fn target(texture: usize) -> TraceTarget {
        TraceTarget {
            texture,
            width: 64,
            height: 64,
            format: "rgba8unorm",
            swapchain: false,
        }
    }

    fn window(draws: u32, clear: Option<TraceClear>, nonzero: u64) -> PixelStats {
        let sampled_bytes = 64 * 64 * 4;
        let mut stats = PixelStats {
            nonzero_bytes: nonzero,
            sampled_bytes,
            distinct_values: 2,
            distinct_pixels_after_draws: 2,
            pass: 1,
            pass_draws: draws,
            target: Some(target(7)),
            ..PixelStats::default()
        };
        apply_clear(&mut stats, clear);
        stats
    }

    /// A zero clear attributes zero bytes; a non-zero clear attributes four
    /// bytes per sampled texel; a load leaves the answer unknown.
    #[test]
    fn clear_derivation_names_zero_nonzero_and_unknown() {
        let zero = window(
            3,
            Some(TraceClear {
                is_clear: true,
                color: [0.0, 0.0, 0.0, 0.0],
            }),
            0,
        );
        assert_eq!(zero.draws_since_clear, 3);
        assert_eq!(zero.clear_color, Some([0.0, 0.0, 0.0, 0.0]));
        assert_eq!(zero.nonzero_bytes_after_clear, Some(0));
        assert_eq!(zero.nonzero_bytes_after_draws(), 0);

        let pink = window(
            2,
            Some(TraceClear {
                is_clear: true,
                color: [0.25, 0.5, 0.75, 1.0],
            }),
            0,
        );
        assert_eq!(pink.draws_since_clear, 2);
        assert_eq!(pink.nonzero_bytes_after_clear, Some(64 * 64 * 4));
        assert_eq!(
            pink.nonzero_bytes_after_draws(),
            0,
            "the shader drew zeros over a non-zero clear"
        );

        let loaded = window(
            4,
            Some(TraceClear {
                is_clear: false,
                color: [1.0, 0.0, 0.0, 1.0],
            }),
            4096,
        );
        assert_eq!(loaded.draws_since_clear, 0, "no clear in the window");
        assert_eq!(loaded.clear_color, None);
        assert_eq!(loaded.nonzero_bytes_after_clear, None);
        assert_eq!(loaded.nonzero_bytes_after_draws(), 4096);
        assert_eq!(loaded.distinct_pixels_after_draws, 2);
    }

    /// `next_pass_with_clear` is the only source of `draws_since_clear`, so a
    /// drawless pass never claims draws after a clear.
    #[test]
    fn draws_since_clear_is_the_sampled_window_only() {
        let mut state = TraceState::default();
        assert!(!state.next_pass_with_clear(Some(TraceClear {
            is_clear: true,
            color: [0.0; 4],
        })));
        state.last_pass_draws = 0;
        let mut stats = PixelStats {
            pass_draws: 0,
            ..PixelStats::default()
        };
        apply_clear(
            &mut stats,
            Some(TraceClear {
                is_clear: true,
                color: [0.0; 4],
            }),
        );
        assert_eq!(stats.draws_since_clear, 0);
    }

    /// A draw sample read back non-zero is `written-after-clear`; a later
    /// cleared pass of the same target is `erased-by-later-clear`.
    #[test]
    fn a_later_clear_erases_a_written_target() {
        let mut history = RtHistory::default();
        let rt = target(0x1000);
        let clear = TraceClear {
            is_clear: true,
            color: [0.0; 4],
        };
        history.note_draw_target(rt);
        assert_eq!(history.note_pass(rt, Some(clear)), Some(1));
        assert_eq!(history.note_sample(rt, &window(1, Some(clear), 16_384)), Some(1));
        assert_eq!(history.class_of(rt.texture), Some(RT_CLASS_WRITTEN));

        // The second pass clears the same target before encoding its draw.
        assert_eq!(history.note_pass(rt, Some(clear)), Some(1));
        assert_eq!(history.class_of(rt.texture), Some(RT_CLASS_ERASED));
        assert_eq!(
            history.note_sample(rt, &window(1, Some(clear), 0)),
            Some(1)
        );
        let evidence = history.evidence();
        assert_eq!(evidence.rt_traced, 1);
        assert_eq!(evidence.rt_sampled, 1);
        assert_eq!(evidence.rt_written_after_clear, 1);
        assert_eq!(evidence.rt_erased_by_later_clear, 1);
        assert_eq!(evidence.rt_never_sampled, 0);
        let entry = evidence.rts.first().expect("one retained RT");
        assert_eq!(entry.rt, 1, "the ordinal, not the texture address");
        assert_eq!(entry.clear_color, Some([0.0; 4]));
        assert_eq!(entry.draws_since_clear, 1);
        assert_eq!(entry.nonzero_bytes_after_clear, Some(0));
        assert_eq!(entry.nonzero_bytes_after_draws, 0);
        assert!(entry.written_after_clear);
        assert!(entry.erased_by_later_clear);
    }

    /// A target named by a draw but never sampled is `never-sampled`.
    #[test]
    fn a_draw_target_that_is_never_sampled_is_named() {
        let mut history = RtHistory::default();
        let rt = target(0x2000);
        assert_eq!(history.note_draw_target(rt), Some(1));
        let evidence = history.evidence();
        assert_eq!(evidence.rt_traced, 1);
        assert_eq!(evidence.rt_sampled, 0);
        assert_eq!(evidence.rt_never_sampled, 1);
        assert_eq!(evidence.rts[0].class, RT_CLASS_NEVER_SAMPLED);
        assert_eq!(history.class_of(rt.texture), Some(RT_CLASS_NEVER_SAMPLED));
    }

    /// A cleared sample that reads back all zero is `drew-zeros`, not erased.
    #[test]
    fn a_zero_draw_after_a_clear_is_drew_zeros() {
        let mut history = RtHistory::default();
        let rt = target(0x3000);
        let clear = TraceClear {
            is_clear: true,
            color: [0.0, 0.0, 0.0, 0.0],
        };
        history.note_pass(rt, Some(clear));
        history.note_sample(rt, &window(2, Some(clear), 0));
        assert_eq!(history.class_of(rt.texture), Some(RT_CLASS_ZERO));
        let evidence = history.evidence();
        assert_eq!(evidence.rt_written_after_clear, 0);
        assert_eq!(evidence.rt_never_sampled, 0);
        assert_eq!(evidence.rts[0].class, RT_CLASS_ZERO);
    }

    /// The history never grows past `MAX_RT_HISTORY`; a retained key keeps
    /// updating after the cap.
    #[test]
    fn the_rt_history_is_bounded() {
        let mut history = RtHistory::default();
        for texture in 0..MAX_RT_HISTORY {
            assert_eq!(
                history.note_draw_target(target(0x4000 + texture)),
                Some(texture as u32 + 1)
            );
        }
        assert_eq!(history.len(), MAX_RT_HISTORY);
        assert!(!history.is_empty());
        assert_eq!(history.dropped(), 0);
        for texture in MAX_RT_HISTORY..MAX_RT_HISTORY + 8 {
            assert_eq!(history.note_draw_target(target(0x4000 + texture)), None);
        }
        assert_eq!(history.len(), MAX_RT_HISTORY, "the cap holds");
        assert_eq!(history.dropped(), 8);
        // An already-retained target still records after the cap.
        assert_eq!(history.note_draw_target(target(0x4000)), Some(1));
        assert_eq!(history.evidence().rts[0].draw_notes, 2);
        let evidence = history.evidence();
        assert_eq!(evidence.rt_traced, MAX_RT_HISTORY as u64);
        assert_eq!(evidence.rt_dropped, 8);
    }

    /// The telemetry JSON carries the required class counters and per-RT rows.
    #[test]
    fn the_rt_evidence_json_names_the_classes() {
        let mut history = RtHistory::default();
        let clear = TraceClear {
            is_clear: true,
            color: [0.0, 0.0, 0.0, 0.0],
        };
        let written = target(0x5000);
        history.note_pass(written, Some(clear));
        history.note_sample(written, &window(1, Some(clear), 16_384));
        let never = target(0x5001);
        history.note_draw_target(never);
        let erased = target(0x5002);
        history.note_pass(erased, Some(clear));
        history.note_sample(erased, &window(1, Some(clear), 16_384));
        history.note_pass(erased, Some(clear));

        let evidence = history.evidence();
        assert_eq!(evidence.rt_traced, 3);
        assert_eq!(evidence.rt_sampled, 2);
        assert_eq!(evidence.rt_written_after_clear, 2);
        assert_eq!(evidence.rt_erased_by_later_clear, 1);
        assert_eq!(evidence.rt_never_sampled, 1);
        let json = evidence.json_value();
        for expected in [
            "\"rt_traced\": 3",
            "\"rt_sampled\": 2",
            "\"rt_written_after_clear\": 2",
            "\"rt_erased_by_later_clear\": 1",
            "\"rt_never_sampled\": 1",
            "\"rt_dropped\": 0",
            "\"rts\": [",
            "\"rt\": 1",
            "\"format\": \"rgba8unorm\"",
            "\"swapchain\": false",
            "\"clear_color\": [0.000, 0.000, 0.000, 0.000]",
            "\"nonzero_bytes_after_clear\": 0",
            "\"nonzero_bytes_after_draws\": 16384",
            "\"distinct_pixels_after_draws\": 2",
            "\"class\": \"written-after-clear\"",
            "\"class\": \"erased-by-later-clear\"",
            "\"class\": \"never-sampled\"",
        ] {
            assert!(json.contains(expected), "{expected} missing from {json}");
        }
        assert_eq!(
            json.matches('{').count(),
            json.matches('}').count(),
            "balanced braces: {json}"
        );
        // A loaded window serializes the unknown counts as null.
        let mut loaded = RtHistory::default();
        let load = target(0x5003);
        loaded.note_pass(
            load,
            Some(TraceClear {
                is_clear: false,
                color: [0.0; 4],
            }),
        );
        loaded.note_sample(
            load,
            &window(
                4,
                Some(TraceClear {
                    is_clear: false,
                    color: [0.0; 4],
                }),
                4096,
            ),
        );
        let json = loaded.evidence().json_value();
        assert!(json.contains("\"clear_color\": null"), "{json}");
        assert!(json.contains("\"nonzero_bytes_after_clear\": null"), "{json}");
        assert!(json.contains("\"class\": \"loaded-unknown\""), "{json}");
    }

    /// Distinct pixels and distinct bytes are different counters.
    #[test]
    fn distinct_pixels_are_not_distinct_bytes() {
        let uniform = vec![0xff_u8, 0x00, 0x00, 0xff].repeat(64 * 64);
        let stats = pixel_stats(&uniform);
        assert_eq!(stats.distinct_pixels_after_draws, 1);
        assert_eq!(stats.distinct_values, 2);

        let mixed = [
            [0_u8, 0, 0, 0].repeat(8),
            [0xff_u8, 0, 0, 0xff].repeat(8),
        ]
        .concat();
        let stats = pixel_stats(&mixed);
        assert_eq!(stats.distinct_pixels_after_draws, 2);
        assert_eq!(stats.distinct_values, 2);
    }
}
