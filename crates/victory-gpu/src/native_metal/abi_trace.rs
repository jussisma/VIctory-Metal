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
//!
//! # Value order
//!
//! One documented order per call name; `record` is called exactly once per
//! guest query, after the answer is computed:
//!
//! - [`calls::CREATE_DESCRIPTOR_HEAP`]:
//!   `[kind, count, flags, start_lo, start_hi, source]`, where `start_lo` and
//!   `start_hi` are the low and high 32-bit halves of the heap's first
//!   descriptor address and `source` is `0` host memory, `1` the title
//!   allocator or `2` a title placement.
//! - [`calls::GET_CPU_DESCRIPTOR_HANDLE_FOR_HEAP_START`] and
//!   [`calls::GET_GPU_DESCRIPTOR_HANDLE_FOR_HEAP_START`]:
//!   `[start_lo, start_hi, kind]`, the halves of the handle handed back and
//!   the heap's descriptor kind.
//! - [`calls::GET_DESCRIPTOR_HANDLE_INCREMENT_SIZE`]: `[kind, increment]`.
//! - [`calls::GET_GPU_VIRTUAL_ADDRESS`]: `[address, buffer_or_texture]`
//!   (`0` buffer, `1` texture).
//! - [`calls::CHECK_FEATURE_SUPPORT`]: `[feature, size, hr, ..words]`, the
//!   first returned output words (at most [`MAX_VALUES`] values in total; a
//!   `FEATURE_LEVELS` answer records its count and highest level instead of
//!   the raw words, because one of those words is a title pointer).
//! - [`calls::CREATE_COMMITTED_RESOURCE`], [`calls::CREATE_PLACED_RESOURCE`],
//!   [`calls::CREATE_RESERVED_RESOURCE`] and [`calls::CREATE_AT_ADDRESS`]:
//!   `[dimension, width, height, depth_or_array, format, hr, gpu_va]`.
//!
//! `hr` is always the 32-bit HRESULT bit pattern. Only the guest vtable's call
//! reaches `record`: the engine's own creation helpers pass a null device and
//! are not guest queries, so they add no rows.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// The static call names one trace line can carry, one per guest export.
pub mod calls {
    /// `ID3D12Device::CreateDescriptorHeap`.
    pub const CREATE_DESCRIPTOR_HEAP: &str = "CreateDescriptorHeap";
    /// `ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart`.
    pub const GET_CPU_DESCRIPTOR_HANDLE_FOR_HEAP_START: &str =
        "GetCPUDescriptorHandleForHeapStart";
    /// `ID3D12DescriptorHeap::GetGPUDescriptorHandleForHeapStart`.
    pub const GET_GPU_DESCRIPTOR_HANDLE_FOR_HEAP_START: &str =
        "GetGPUDescriptorHandleForHeapStart";
    /// `ID3D12Device::GetDescriptorHandleIncrementSize`.
    pub const GET_DESCRIPTOR_HANDLE_INCREMENT_SIZE: &str = "GetDescriptorHandleIncrementSize";
    /// `ID3D12Resource::GetGPUVirtualAddress`.
    pub const GET_GPU_VIRTUAL_ADDRESS: &str = "GetGPUVirtualAddress";
    /// `ID3D12Device::CheckFeatureSupport`.
    pub const CHECK_FEATURE_SUPPORT: &str = "CheckFeatureSupport";
    /// `ID3D12Device::CreateCommittedResource` (and its console variants).
    pub const CREATE_COMMITTED_RESOURCE: &str = "CreateCommittedResource";
    /// `ID3D12Device::CreatePlacedResource`.
    pub const CREATE_PLACED_RESOURCE: &str = "CreatePlacedResource";
    /// `ID3D12Device::CreateReservedResource`.
    pub const CREATE_RESERVED_RESOURCE: &str = "CreateReservedResource";
    /// `ID3D12Device9::CreatePlacedResourceX`/`CreateAtAddress`.
    pub const CREATE_AT_ADDRESS: &str = "CreateAtAddress";
}

/// Records kept per call name.
pub const MAX_RECORDS_PER_CALL: u64 = 64;
/// Records kept for the whole process.
pub const MAX_RECORDS_TOTAL: u64 = 4096;
/// Values formatted per line (extra values are dropped, never truncated).
pub const MAX_VALUES: usize = 8;

static ENABLED: OnceLock<bool> = OnceLock::new();
static TOTAL: AtomicU64 = AtomicU64::new(0);
static PER_CALL: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

/// Lines captured by a test trace, if one is running.
#[cfg(test)]
static TEST_TRACE: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Whether the trace is on (`VICTORY_GUEST_ABI_TRACE` set and not `0`/empty).
///
/// Read once; changing the variable after the first query has no effect. Tests
/// may additionally turn it on for their own capture window.
pub fn enabled() -> bool {
    if *ENABLED.get_or_init(|| {
        std::env::var_os("VICTORY_GUEST_ABI_TRACE")
            .is_some_and(|value| !value.is_empty() && value != "0")
    }) {
        return true;
    }
    #[cfg(test)]
    {
        test_tracing()
    }
    #[cfg(not(test))]
    {
        false
    }
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
    let line = format_line(call, values);
    #[cfg(test)]
    if let Some(trace) = TEST_TRACE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        trace.push(line.clone());
    }
    tracing::info!(target: "victory_abi_trace", "{}", line);
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

/// Serialises the tests that capture a trace window.
#[cfg(test)]
pub(crate) fn test_serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts a capture window: tracing is on for tests until [`end_test_trace`].
///
/// The counters are reset with the window, so a focused test sees exactly the
/// lines its own calls produce.
#[cfg(test)]
pub(crate) fn begin_test_trace() {
    reset();
    *TEST_TRACE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Vec::new());
}

/// Ends the capture window and returns its lines.
#[cfg(test)]
pub(crate) fn end_test_trace() -> Vec<String> {
    TEST_TRACE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default()
}

#[cfg(test)]
fn test_tracing() -> bool {
    TEST_TRACE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
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
    fn a_capture_window_records_the_formatted_lines_and_stops() {
        let _serial = test_serial();
        begin_test_trace();
        record(calls::GET_DESCRIPTOR_HANDLE_INCREMENT_SIZE, &[0, 32]);
        record(calls::GET_GPU_VIRTUAL_ADDRESS, &[0x1234, 1]);
        let lines = end_test_trace();
        assert_eq!(
            lines,
            [
                "abi_trace call=GetDescriptorHandleIncrementSize values=[0, 32]",
                "abi_trace call=GetGPUVirtualAddress values=[4660, 1]",
            ]
        );
        // The window is over: nothing more is kept.
        record(calls::GET_GPU_VIRTUAL_ADDRESS, &[0x99, 0]);
        assert!(!test_tracing());
    }

    #[test]
    fn the_cap_is_per_call_name() {
        let _serial = test_serial();
        begin_test_trace();
        for value in 0..(MAX_RECORDS_PER_CALL + 3) {
            record(calls::CHECK_FEATURE_SUPPORT, &[value]);
        }
        record(calls::GET_GPU_VIRTUAL_ADDRESS, &[1]);
        let lines = end_test_trace();
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains("call=CheckFeatureSupport"))
                .count() as u64,
            MAX_RECORDS_PER_CALL
        );
        assert!(
            lines
                .iter()
                .filter(|line| line.contains("call=GetGPUVirtualAddress"))
                .count()
                >= 1,
            "one name's cap does not spend another's"
        );
        assert!(MAX_RECORDS_TOTAL >= MAX_RECORDS_PER_CALL);
    }
}
