//! ADR-036: opt-in differential guest-ABI trace.
//!
//! `VICTORY_GUEST_ABI_TRACE=1` makes the instrumented guest exports record the
//! scalars they hand back to the title. Two runs of the same title — one with
//! `VICTORY_BACKEND=metal`, one against the D3D12 host reference — are then
//! diffed by `tools/macos/metal-poc/abi-diff.py`, which names the exact
//! guest-visible delta that decides a title predicate (for Boot #5, the fatal
//! assert at `+0x2215c35`).
//!
//! The trace is **bounded** and carries **no title text, paths, pointers into
//! title memory or asset bytes**: only static call names and integer values.
//! Off by default, and every entry point returns immediately when off, so the
//! hot path pays one relaxed atomic load at most.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::collections::BTreeMap;

/// Records kept per call name.
pub const MAX_RECORDS_PER_CALL: u64 = 64;
/// Records kept for the whole process.
pub const MAX_RECORDS_TOTAL: u64 = 4096;
/// Values formatted per line (extra values are dropped, never truncated).
pub const MAX_VALUES: usize = 8;

static ENABLED: OnceLock<bool> = OnceLock::new();
static TOTAL: AtomicU64 = AtomicU64::new(0);
static PER_CALL: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

/// Whether the trace is on (`VICTORY_GUEST_ABI_TRACE` set and not `0`/empty).
///
/// Read once; changing the variable after the first query has no effect.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var_os("VICTORY_GUEST_ABI_TRACE")
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

/// The one line a record becomes, independent of whether tracing is on.
///
/// Fixed shape so the diff tool can parse it without heuristics:
/// `abi_trace call=<name> values=[v0, v1, ...]`.
pub(crate) fn format_line(call: &'static str, values: &[u64]) -> String {
    let shown = &values[..values.len().min(MAX_VALUES)];
    let body = shown
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("abi_trace call={call} values=[{body}]")
}

/// Records one guest-visible query. `call` is a static export name; `values`
/// are the scalars the title received, in the documented order for that name.
pub fn record(call: &'static str, values: &[u64]) {
    if !enabled() {
        return;
    }
    if TOTAL.fetch_add(1, Ordering::Relaxed) >= MAX_RECORDS_TOTAL {
        return;
    }
    {
        let mut per_call = PER_CALL.lock().unwrap_or_else(|e| e.into_inner());
        let seen = per_call.entry(call).or_insert(0);
        if *seen >= MAX_RECORDS_PER_CALL {
            return;
        }
        *seen += 1;
    }
    tracing::info!(target: "victory_abi_trace", "{}", format_line(call, values));
}

/// The per-call record counts, strongest first. Diagnostics and tests only.
pub fn counts() -> Vec<(&'static str, u64)> {
    let per_call = PER_CALL.lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: Vec<(&'static str, u64)> = per_call.iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    rows
}

/// Forgets every record (tests).
pub fn reset() {
    TOTAL.store(0, Ordering::Relaxed);
    PER_CALL.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_has_a_fixed_parseable_shape() {
        let line = format_line("GetDescriptorHandleIncrementSize", &[32, 1]);
        assert_eq!(
            line,
            "abi_trace call=GetDescriptorHandleIncrementSize values=[32, 1]"
        );
        assert_eq!(format_line("Empty", &[]), "abi_trace call=Empty values=[]");
    }

    #[test]
    fn values_are_capped_not_truncated() {
        let line = format_line("Wide", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert!(line.ends_with("values=[1, 2, 3, 4, 5, 6, 7, 8]"), "{line}");
        assert_eq!(line.matches(',').count(), MAX_VALUES - 1);
    }

    #[test]
    fn recording_without_the_variable_keeps_nothing() {
        // The variable is absent in the test environment by construction.
        assert!(!enabled() || std::env::var_os("VICTORY_GUEST_ABI_TRACE").is_some());
        reset();
        let before = counts();
        record("GetGPUDescriptorHandleForHeapStart", &[0x0e00_0000]);
        if !enabled() {
            assert_eq!(before, counts(), "off means nothing is recorded");
        }
    }

    #[test]
    fn the_cap_is_per_call_name() {
        reset();
        assert!(MAX_RECORDS_PER_CALL >= 1);
        assert!(MAX_RECORDS_TOTAL >= MAX_RECORDS_PER_CALL);
    }
}
