//! ADR-058: live `MEMORY_SPACE` telemetry and the flush gate the native
//! renderer needs before `flush_abi` binds the title page table.
//!
//! `flush_abi` is the only place a submission's stages receive the live
//! page table `t0`/`u0` and the heap argument buffer. Its callers,
//! `flush_render_abi` and `flush_compute_abi`, used to return as soon as
//! `AbiPending` was empty. A translated stage whose `MEMORY_SPACE` bindings are
//! host-owned (the guest never sends a root argument for `t0`/`u0`) therefore
//! drew with no page table at all: Boot #28 encoded 52 draws with
//! `memory_space_bound` 0 while the offscreen attachment held exactly one
//! value.
//!
//! The gate below is the one-line integration the renderer needs:
//!
//! ```text
//! // in `flush_render_abi` and `flush_compute_abi`, before flush_abi:
//! let stages = /* the stage list for this pipeline */;
//! if pending.is_empty()
//!     && !crate::macos::memory_space_bind::declares_memory_space(&stages)
//! {
//!     return Ok(());
//! }
//! ```
//!
//! `flush_abi` already returns immediately for an empty `pending` when no
//! stage declares [`victory_shader::emit::MEMORY_SPACE`], so the gate only
//! changes behaviour when a stage really needs the table. Removing the two
//! early returns is equivalent and equally correct.
//!
//! # Telemetry
//!
//! The counters live here rather than on `DirectMetalTransport` so the host
//! binary can publish them without a device-state accessor. They are
//! process-wide totals (one native host serves one device):
//!
//! | field | meaning |
//! | --- | --- |
//! | `gate_checks` | ABI flushes whose stage list the gate examined |
//! | `declared_stages` | those in which at least one stage declared `MEMORY_SPACE` |
//! | `requests` | flushes that asked for the live page table |
//! | `no_map` | requests with no installed map (nothing to build from) |
//! | `updates` | map applications that changed the range set |
//! | `pages` | pages in the table most recently built (0 when none) |
//! | `invalidations` | built tables dropped by a changed map or allocation |
//! | `bindings` | flushes that bound a live table |
//! | `misses` | requests that had a map and still could not build a table |
//! | `ranges` | ranges the installed map names |
//! | `heap_pairs` | heap pairs the built table names |
//!
//! `requests` counts every flush that needed the table, whatever its pending
//! root arguments; `gate_checks` counts only the flushes whose pending set was
//! empty, because `flush_render_abi`/`flush_compute_abi` short-circuit the
//! scan when `pending` is non-empty (`pending.is_empty() && !declares(..)`).
//! `no_map` therefore separates "a stage wanted a table and the guest had sent
//! no map" from "the map was there and the table was refused": `requests > 0`
//! and `no_map == 0` with `bindings == 0` names the map application path, not
//! the emit decision.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::macos::bindings::RootSignatureLayout;
use crate::macos::device::ShaderStage;

/// Map applications that changed the range set (ADR-058).
static UPDATES: AtomicU64 = AtomicU64::new(0);
/// Pages in the table most recently built; a gauge.
static PAGES: AtomicU64 = AtomicU64::new(0);
/// Built tables dropped by a changed map or allocation.
static INVALIDATIONS: AtomicU64 = AtomicU64::new(0);
/// Flushes that bound a live table.
static BINDINGS: AtomicU64 = AtomicU64::new(0);
/// Flushes that needed a table and could not build one.
static MISSES: AtomicU64 = AtomicU64::new(0);
/// Ranges the installed map names; a gauge.
static RANGES: AtomicU64 = AtomicU64::new(0);
/// Heap pairs the built table names; a gauge.
static HEAP_PAIRS: AtomicU64 = AtomicU64::new(0);
/// Stage lists (per ABI flush) in which at least one stage declared
/// `MEMORY_SPACE`; separates "no shader needs title memory" from "the table
/// never bound".
static DECLARED_STAGES: AtomicU64 = AtomicU64::new(0);
/// ABI flush stage lists the gate examined (an empty pending set).
static GATE_CHECKS: AtomicU64 = AtomicU64::new(0);
/// Flushes that asked for the live title page table.
static REQUESTS: AtomicU64 = AtomicU64::new(0);
/// Requests that found no installed map to build the table from.
static NO_MAP: AtomicU64 = AtomicU64::new(0);

/// Exact process-wide live-`MEMORY_SPACE` counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemorySpaceTelemetry {
    /// ABI flushes whose stage list the flush gate examined.
    pub gate_checks: u64,
    /// ABI flushes in which a stage declared `MEMORY_SPACE`.
    pub declared_stages: u64,
    /// Flushes that asked for the live title page table.
    pub requests: u64,
    /// Requests with no installed map (nothing to build the table from).
    pub no_map: u64,
    /// Map applications that changed the range set.
    pub updates: u64,
    /// Pages in the table most recently built (a gauge; 0 when none).
    pub pages: u64,
    /// Built tables dropped by a changed map or allocation.
    pub invalidations: u64,
    /// Flushes that bound a live table.
    pub bindings: u64,
    /// Flushes that needed a table and could not build one.
    pub misses: u64,
    /// Ranges the installed map names.
    pub ranges: u64,
    /// Heap pairs the built table names.
    pub heap_pairs: u64,
}

/// A snapshot of the live counters.
pub fn telemetry() -> MemorySpaceTelemetry {
    MemorySpaceTelemetry {
        gate_checks: GATE_CHECKS.load(Ordering::Relaxed),
        declared_stages: DECLARED_STAGES.load(Ordering::Relaxed),
        requests: REQUESTS.load(Ordering::Relaxed),
        no_map: NO_MAP.load(Ordering::Relaxed),
        updates: UPDATES.load(Ordering::Relaxed),
        pages: PAGES.load(Ordering::Relaxed),
        invalidations: INVALIDATIONS.load(Ordering::Relaxed),
        bindings: BINDINGS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        ranges: RANGES.load(Ordering::Relaxed),
        heap_pairs: HEAP_PAIRS.load(Ordering::Relaxed),
    }
}

/// Whether any stage of `stages` declares a `MEMORY_SPACE` root parameter.
///
/// This is the gate `flush_render_abi`/`flush_compute_abi` need: their early
/// return is only valid when no stage needs the page table. Stages are exactly
/// the `(stage, reflected layout)` list those functions already build for
/// `flush_abi`.
///
/// Every call is one ABI flush the gate examined; the module notes explain why
/// a flush with a non-empty pending set does not reach this function.
pub fn declares_memory_space(stages: &[(ShaderStage, RootSignatureLayout)]) -> bool {
    GATE_CHECKS.fetch_add(1, Ordering::Relaxed);
    let declared = stages.iter().any(|(_, layout)| {
        layout
            .parameters()
            .iter()
            .any(|parameter| parameter.space() == victory_shader::emit::MEMORY_SPACE)
    });
    if declared {
        DECLARED_STAGES.fetch_add(1, Ordering::Relaxed);
    }
    declared
}

/// Records one flush that asked the title-memory map for the live table.
///
/// `has_map` is whether an installed map existed at all; a request without one
/// cannot build a table, so it is counted apart from a refused build.
pub(crate) fn note_table_request(has_map: bool) {
    REQUESTS.fetch_add(1, Ordering::Relaxed);
    if !has_map {
        NO_MAP.fetch_add(1, Ordering::Relaxed);
    }
}

/// Records a map application that changed the range set; `ranges` is the new
/// gauge.
pub(crate) fn note_map_application(ranges: u64) {
    UPDATES.fetch_add(1, Ordering::Relaxed);
    RANGES.store(ranges, Ordering::Relaxed);
}

/// Records one built table dropped.
pub(crate) fn note_invalidation() {
    INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Records one flush that asked for the live table; `bound` says whether a
/// table was produced, and `pages`/`heap_pairs` publish the built gauges.
///
/// Only a request that had a map to build from reaches here; a request with no
/// installed map is counted by [`note_table_request`] instead, so `misses`
/// never mixes "the map was empty" with "the map would not build".
pub(crate) fn note_binding(pages: u64, heap_pairs: u64, bound: bool) {
    if bound {
        BINDINGS.fetch_add(1, Ordering::Relaxed);
        PAGES.store(pages, Ordering::Relaxed);
        HEAP_PAIRS.store(heap_pairs, Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, Ordering::Relaxed);
        PAGES.store(0, Ordering::Relaxed);
        HEAP_PAIRS.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macos::bindings::{ReflectedStage, RootParameterKind, RootParameterSpec};

    fn layout(space: u32) -> RootSignatureLayout {
        RootSignatureLayout::from_parameters(
            ReflectedStage::Fragment,
            &[RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: 16,
                register: 0,
                space,
            }],
        )
        .expect("layout")
    }

    /// The gate counter moves for every stage list it looks at, and the
    /// declaration counter only when that list names `MEMORY_SPACE`; this is
    /// what separates "no shader needs title memory" from "the flush never
    /// ran". The counters are process-wide, so only their deltas are asserted
    /// (the suite runs with `--test-threads=1`).
    #[test]
    fn the_gate_and_declaration_counters_separate_the_two_no_table_cases() {
        let before = telemetry();
        let plain = layout(0);
        let memory = layout(victory_shader::emit::MEMORY_SPACE);

        assert!(!declares_memory_space(&[(
            ShaderStage::Fragment,
            plain.clone()
        )]));
        let after = telemetry();
        assert_eq!(after.gate_checks, before.gate_checks + 1);
        assert_eq!(after.declared_stages, before.declared_stages);

        assert!(declares_memory_space(&[(ShaderStage::Fragment, memory)]));
        let after = telemetry();
        assert_eq!(after.gate_checks, before.gate_checks + 2);
        assert_eq!(after.declared_stages, before.declared_stages + 1);
        // A flush with pending root arguments skips the gate itself; only the
        // request counter sees it.
        assert_eq!(after.requests, before.requests);
    }

    /// A request counts once, and a request with no map is also counted as
    /// `no_map`, so `requests > 0, no_map == 0, bindings == 0` names the map
    /// application path rather than the emit decision.
    #[test]
    fn a_request_without_a_map_is_counted_apart_from_one_with_a_map() {
        let before = telemetry();
        note_table_request(false);
        let after = telemetry();
        assert_eq!(after.requests, before.requests + 1);
        assert_eq!(after.no_map, before.no_map + 1);

        note_table_request(true);
        let after = telemetry();
        assert_eq!(after.requests, before.requests + 2);
        assert_eq!(after.no_map, before.no_map + 1);
    }
}
