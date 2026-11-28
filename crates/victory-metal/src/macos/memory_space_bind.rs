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
//! | `updates` | map applications that changed the range set |
//! | `pages` | pages in the table most recently built (0 when none) |
//! | `invalidations` | built tables dropped by a changed map or allocation |
//! | `bindings` | flushes that bound a live table |
//! | `misses` | flushes that needed a table and could not build one |
//! | `ranges` | ranges the installed map names |
//! | `heap_pairs` | heap pairs the built table names |
//!
//! `bindings` and `misses` separate "the flush ran and the map was empty" from
//! "the flush never ran", which the old `memory_space_bound`/`bytes` pair could
//! not.

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

/// Exact process-wide live-`MEMORY_SPACE` counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemorySpaceTelemetry {
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
pub fn declares_memory_space(stages: &[(ShaderStage, RootSignatureLayout)]) -> bool {
    stages.iter().any(|(_, layout)| {
        layout
            .parameters()
            .iter()
            .any(|parameter| parameter.space() == victory_shader::emit::MEMORY_SPACE)
    })
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
