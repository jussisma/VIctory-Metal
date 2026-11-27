//! Per-frame telemetry for the native Metal backend (ADR-016).
//!
//! This module owns guest-side frame accounting: record microseconds, ring
//! submit microseconds, frame time percentiles (p50/p95/p99), FPS, the
//! unsupported-opcode histogram and the strict `d3dmetal_fallback_count == 0`
//! guard. It also keeps the session's descriptor-heap binding audit: every
//! heap the title binds is recorded by identity (start and kind), a refused
//! slot is counted under a static name, and the last bound set is reported on
//! the frame line, so a heap the native path cannot serve is never a silent
//! skip.
//!
//! # Named submission refusals
//!
//! [`FrameTelemetry::note_submit_refused`] takes a static reason from the fixed
//! [`SUBMIT_REFUSAL_NAMES`] table and moves the refusal total and its named
//! histogram in one call, so `submits=n/m ok/refused` can never appear beside
//! an empty `named_submit_refusals`. The reasons are session totals: a boot's
//! one refusal keeps its name after any window reset. An empty batch is not a
//! submission and is never counted as a refused one.
//!
//! # What one sample is
//!
//! [`FrameTelemetry::record_frame`] records one `ExecuteCommandLists` batch:
//! the guest time to record it, the canonical v3 encode time, the IPC
//! round-trip time and the host-reported GPU time. The sample's frame time is
//! their sum, so percentiles describe the instrumented cost of one submitted
//! frame, not a wall-clock interval. [`FrameTelemetry::fps`] is the rate of
//! those samples over the window: it is derived from the recorded timestamps,
//! so it is exact on a fixture and independent of the wall clock in tests
//! ([`FrameTelemetry::record_frame_at`] takes the timestamp explicitly).
//!
//! The window keeps the last [`WINDOW_FRAMES`] samples, so memory is bounded
//! and p99 always has at least 100 samples once a run is that long.
//!
//! # The zero-D3DMetal guard
//!
//! `d3dmetal_fallback_count` is a hard counter: it only moves through
//! [`FrameTelemetry::note_d3dmetal_fallback`], it is never reset by
//! [`FrameTelemetry::snapshot_and_reset`], and [`assert_no_fallback`] refuses a
//! frame while it is nonzero. Nothing on the native path calls it; it exists so
//! that any future fallback hook has to declare itself instead of silently
//! producing a D3DMetal frame (ADR-016). The D3D12 backend never touches this
//! module at all.
//!
//! # Percentiles
//!
//! Exact nearest-rank, no sampling: for `n` ascending samples and percentile
//! `p`, the result is `sorted[ceil(p / 100 * n) - 1]`. A 100-sample fixture
//! `1..=100` therefore answers exactly `50`, `95` and `99`; a single sample
//! answers itself for every percentile; an empty window answers zero.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// How many recent frames the window keeps (all of them, in order).
pub const WINDOW_FRAMES: usize = 4096;

/// Histogram key for a refused call that has no wire opcode.
pub const NO_OPCODE: u16 = u16::MAX;

/// Maximum bytes of a host error reason kept in telemetry and logged.
///
/// Matches [`victory_metal::wire::MAX_REPLY_REASON_BYTES`] so the guest never
/// retains more text than the Wire v3 reply can carry.
pub const MAX_HOST_ERROR_BYTES: usize = 256;

/// One bounded host error carried by a Wire v3 error reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    /// Sequence of the refused submission.
    pub sequence: u32,
    /// Bounded, single-line host diagnostic.
    pub reason: String,
}

/// One recorded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSample {
    /// Monotonic timestamp of the sample, in microseconds.
    pub at_us: u64,
    /// Guest time to record the frame's commands, in microseconds.
    pub record_us: u64,
    /// Ring submit (IPC round trip) time, in microseconds.
    pub ring_submit_us: u64,
    /// Canonical v3 encode time, in microseconds.
    pub encode_us: u64,
    /// Host-reported GPU time, in microseconds (0 when the reply carries none).
    pub gpu_us: u64,
    /// The opcode a refused frame needed, when the producer refused it.
    pub unsupported_opcode: Option<u16>,
}

impl FrameSample {
    /// The frame cost this sample contributes: every instrumented phase.
    pub const fn frame_us(&self) -> u64 {
        self.record_us
            .saturating_add(self.ring_submit_us)
            .saturating_add(self.encode_us)
            .saturating_add(self.gpu_us)
    }
}

/// Exact nearest-rank percentiles of the window, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Percentiles {
    /// Median frame cost.
    pub p50: u64,
    /// 95th percentile frame cost.
    pub p95: u64,
    /// 99th percentile frame cost.
    pub p99: u64,
}

impl Percentiles {
    /// The nearest-rank percentile of ascending `sorted`.
    fn nearest_rank(sorted: &[u64], percent: u64) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let n = sorted.len() as u64;
        // ceil(percent * n / 100), computed in integers so it is exact.
        let rank = (percent * n).div_ceil(100).clamp(1, n);
        sorted[(rank - 1) as usize]
    }
}

/// The strict zero-D3DMetal guard tripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryError {
    /// A frame reached D3DMetal; the native path is no longer exclusive.
    D3dMetalFallback {
        /// Fallbacks recorded so far.
        count: u64,
    },
}

impl TelemetryError {
    /// Fallbacks recorded when this error was produced.
    pub fn fallback_count(&self) -> u64 {
        match self {
            Self::D3dMetalFallback { count } => *count,
        }
    }
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D3dMetalFallback { count } => write!(
                f,
                "{count} frame(s) fell back to D3DMetal; the native path is not exclusive"
            ),
        }
    }
}

impl std::error::Error for TelemetryError {}

/// Maximum bytes of one first-failure line (name, HRESULT, arguments).
///
/// The record itself is a fixed-size struct; this bounds the readable form,
/// so a pathological name is truncated on a character boundary instead of
/// growing the log line.
pub const MAX_CALL_FAILURE_BYTES: usize = 160;

/// Scalar arguments a failure record keeps.
pub const MAX_CALL_FAILURE_ARGS: usize = 3;

/// The names the instrumented GPU exports pass to the failure latch.
///
/// All names are static: the latch keeps no title text, paths or memory.
pub mod call_names {
    /// `ID3D12Device::CreateDescriptorHeap`.
    pub const CREATE_DESCRIPTOR_HEAP: &str = "CreateDescriptorHeap";
    /// `ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart`.
    pub const GET_CPU_DESCRIPTOR_HANDLE_FOR_HEAP_START: &str =
        "GetCPUDescriptorHandleForHeapStart";
    /// `ID3D12DescriptorHeap::GetGPUDescriptorHandleForHeapStart`.
    pub const GET_GPU_DESCRIPTOR_HANDLE_FOR_HEAP_START: &str =
        "GetGPUDescriptorHandleForHeapStart";
    /// `ID3D12Device::CreateConstantBufferView`.
    pub const CREATE_CONSTANT_BUFFER_VIEW: &str = "CreateConstantBufferView";
    /// `ID3D12Device::CreateShaderResourceView`.
    pub const CREATE_SHADER_RESOURCE_VIEW: &str = "CreateShaderResourceView";
    /// `ID3D12Device::CreateUnorderedAccessView`.
    pub const CREATE_UNORDERED_ACCESS_VIEW: &str = "CreateUnorderedAccessView";
    /// `ID3D12Device::CreateRenderTargetView`.
    pub const CREATE_RENDER_TARGET_VIEW: &str = "CreateRenderTargetView";
    /// `ID3D12Device::CreateDepthStencilView`.
    pub const CREATE_DEPTH_STENCIL_VIEW: &str = "CreateDepthStencilView";
    /// `ID3D12Device::CopyDescriptorsSimple`.
    pub const COPY_DESCRIPTORS_SIMPLE: &str = "CopyDescriptorsSimple";
    /// `ID3D12Resource::Map`.
    pub const MAP: &str = "Map";
    /// `ID3D12Resource::Unmap`.
    pub const UNMAP: &str = "Unmap";
    /// `ID3D12Device::CreateCommittedResource`.
    pub const CREATE_COMMITTED_RESOURCE: &str = "CreateCommittedResource";
    /// `ID3D12Device::CreatePlacedResource`.
    pub const CREATE_PLACED_RESOURCE: &str = "CreatePlacedResource";
    /// `ID3D12Device::CreateReservedResource`.
    pub const CREATE_RESERVED_RESOURCE: &str = "CreateReservedResource";
    /// `ID3D12Device9::CreatePlacedResourceX`/`CreateAtAddress`.
    pub const CREATE_AT_ADDRESS: &str = "CreateAtAddress";
    /// `ID3D12Resource::GetGPUVirtualAddress`.
    pub const GET_GPU_VIRTUAL_ADDRESS: &str = "GetGPUVirtualAddress";
}

/// The fixed set of names the histogram counts, in a stable order.
///
/// A failure can only be counted under a known name, so the histogram never
/// grows and its contents do not depend on the order calls failed in.
pub const CALL_FAILURE_NAMES: &[&str] = &[
    call_names::CREATE_DESCRIPTOR_HEAP,
    call_names::GET_CPU_DESCRIPTOR_HANDLE_FOR_HEAP_START,
    call_names::GET_GPU_DESCRIPTOR_HANDLE_FOR_HEAP_START,
    call_names::CREATE_CONSTANT_BUFFER_VIEW,
    call_names::CREATE_SHADER_RESOURCE_VIEW,
    call_names::CREATE_UNORDERED_ACCESS_VIEW,
    call_names::CREATE_RENDER_TARGET_VIEW,
    call_names::CREATE_DEPTH_STENCIL_VIEW,
    call_names::COPY_DESCRIPTORS_SIMPLE,
    call_names::MAP,
    call_names::UNMAP,
    call_names::CREATE_COMMITTED_RESOURCE,
    call_names::CREATE_PLACED_RESOURCE,
    call_names::CREATE_RESERVED_RESOURCE,
    call_names::CREATE_AT_ADDRESS,
    call_names::GET_GPU_VIRTUAL_ADDRESS,
];

/// The first failing GPU call, kept once for the telemetry snapshot.
///
/// Fixed-size and copied out by value, so reading it never allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallFailure {
    /// The export's static name.
    pub name: &'static str,
    /// The HRESULT the export returned (0 when it returned `S_OK` and left a
    /// required output zero or null).
    pub hr: i32,
    /// Up to [`MAX_CALL_FAILURE_ARGS`] compact scalar arguments.
    pub args: [u64; MAX_CALL_FAILURE_ARGS],
}

impl CallFailure {
    /// The record as one line, at most [`MAX_CALL_FAILURE_BYTES`] bytes.
    pub fn line(&self) -> String {
        let mut text = format!(
            "{} hr={:#x} args=[{:#x},{:#x},{:#x}]",
            self.name, self.hr as u32, self.args[0], self.args[1], self.args[2]
        );
        if text.len() > MAX_CALL_FAILURE_BYTES {
            let mut end = MAX_CALL_FAILURE_BYTES;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
        text
    }
}

const FIRST_EMPTY: u64 = 0;
const FIRST_WRITING: u64 = 1;
const FIRST_READY: u64 = 2;

/// The single bounded first-failure record, published with one atomic claim.
///
/// The first thread to move `state` from empty to writing owns every field;
/// later failures only count. A reader sees the record only once `state` is
/// `ready`, published with release/acquire.
#[derive(Debug, Default)]
struct FirstCallFailure {
    state: AtomicU64,
    name: AtomicUsize,
    name_len: AtomicUsize,
    hr: AtomicI32,
    args: [AtomicU64; MAX_CALL_FAILURE_ARGS],
}

impl FirstCallFailure {
    /// Claims the record for `failure`; false if another failure owns it.
    fn claim(&self, failure: CallFailure) -> bool {
        if self
            .state
            .compare_exchange(
                FIRST_EMPTY,
                FIRST_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.name.store(failure.name.as_ptr() as usize, Ordering::Relaxed);
        self.name_len.store(failure.name.len(), Ordering::Relaxed);
        self.hr.store(failure.hr, Ordering::Relaxed);
        for (slot, value) in self.args.iter().zip(failure.args) {
            slot.store(value, Ordering::Relaxed);
        }
        self.state.store(FIRST_READY, Ordering::Release);
        true
    }

    /// The recorded failure, if the record is complete.
    fn get(&self) -> Option<CallFailure> {
        if self.state.load(Ordering::Acquire) != FIRST_READY {
            return None;
        }
        let pointer = self.name.load(Ordering::Relaxed) as *const u8;
        let len = self.name_len.load(Ordering::Relaxed);
        // SAFETY: the only names the latch accepts are `&'static str`, so the
        // bytes stay valid for the process.
        let name = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(pointer, len)) };
        let mut args = [0u64; MAX_CALL_FAILURE_ARGS];
        for (slot, value) in self.args.iter().zip(args.iter_mut()) {
            *value = slot.load(Ordering::Relaxed);
        }
        Some(CallFailure {
            name,
            hr: self.hr.load(Ordering::Relaxed),
            args,
        })
    }

    #[cfg(test)]
    fn reset(&self) {
        self.state.store(FIRST_EMPTY, Ordering::Relaxed);
    }
}

/// A bounded, lock-free latch over failing GPU calls.
///
/// Every failure moves [`Self::count`], its name's slot in a fixed
/// [`CALL_FAILURE_NAMES`] histogram and, for the first failure only, a
/// fixed-size [`CallFailure`]. Repeated failures of one name only count.
#[derive(Debug, Default)]
pub struct CallFailureLatch {
    total: AtomicU64,
    by_name: [AtomicU64; CALL_FAILURE_NAMES.len()],
    first: FirstCallFailure,
}

/// Saturating increment returning the previous value.
fn bump(counter: &AtomicU64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_add(1))
        })
        .unwrap_or(u64::MAX)
}

/// The histogram slot for `name`, if it is one of [`CALL_FAILURE_NAMES`].
fn call_failure_index(name: &str) -> Option<usize> {
    CALL_FAILURE_NAMES.iter().position(|&known| known == name)
}

impl CallFailureLatch {
    /// Records one failing call and logs the first failure of its name.
    pub fn note(&self, name: &'static str, hr: i32, args: [u64; MAX_CALL_FAILURE_ARGS]) {
        bump(&self.total);
        let first_for_name = match call_failure_index(name) {
            Some(index) => bump(&self.by_name[index]) == 0,
            None => false,
        };
        if first_for_name {
            tracing::warn!(
                call = name,
                hr = format_args!("{:#010x}", hr as u32),
                arg0 = args[0],
                arg1 = args[1],
                arg2 = args[2],
                "GPU call failed"
            );
        }
        self.first.claim(CallFailure { name, hr, args });
    }

    /// Records `hr` only when it is a failure, so a successful export adds
    /// nothing.
    pub fn note_result(
        &self,
        name: &'static str,
        hr: i32,
        args: [u64; MAX_CALL_FAILURE_ARGS],
    ) {
        if hr < 0 {
            self.note(name, hr, args);
        }
    }

    /// Failing calls recorded so far.
    pub fn count(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Failures by export name, in the fixed names' sorted order.
    pub fn histogram(&self) -> BTreeMap<&'static str, u64> {
        CALL_FAILURE_NAMES
            .iter()
            .zip(self.by_name.iter())
            .filter_map(|(&name, count)| {
                let count = count.load(Ordering::Relaxed);
                (count > 0).then_some((name, count))
            })
            .collect()
    }

    /// The one bounded first failure, if any has happened.
    pub fn first(&self) -> Option<CallFailure> {
        self.first.get()
    }

    /// Clears the latch (tests only).
    #[cfg(test)]
    pub(crate) fn reset(&self) {
        self.total.store(0, Ordering::Relaxed);
        for count in &self.by_name {
            count.store(0, Ordering::Relaxed);
        }
        self.first.reset();
    }
}

/// Maximum heap identities the last-bound set keeps.
///
/// A `SetDescriptorHeaps` call names at most two heaps in D3D12, so eight
/// leaves room for a title that over-fills the set; the line never grows past
/// [`MAX_HEAP_BINDINGS`] pairs.
pub const MAX_HEAP_BINDINGS: usize = 8;

/// Maximum distinct heap identities the session census tracks.
///
/// Heap addresses are not reused while a heap lives, so a boot binds a few
/// dozen at most. The cap keeps the census bounded over a long session;
/// `heaps_distinct` saturates there instead of growing without limit.
pub const MAX_DISTINCT_HEAPS: usize = 64;

/// The static refusal classes of the descriptor-heap binding audit.
///
/// Every refusal is counted under exactly one of these names, so a boot log
/// can tell a null slot from a heap the wire cannot serve.
pub mod heap_refusals {
    /// A null slot in the bound set.
    pub const NULL_SLOT: &str = "null_slot";
    /// A non-null object with no descriptor start: not a heap this path made.
    pub const UNKNOWN_OBJECT: &str = "unknown_object";
    /// A heap of a kind the wire cannot express.
    pub const KIND_NOT_EXPRESSIBLE: &str = "kind_not_expressible";
}

/// The heap refusal classes, in a stable order.
pub const HEAP_REFUSAL_NAMES: &[&str] = &[
    heap_refusals::NULL_SLOT,
    heap_refusals::UNKNOWN_OBJECT,
    heap_refusals::KIND_NOT_EXPRESSIBLE,
];

/// The descriptor-heap kinds the native wire can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HeapKind {
    /// A `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` heap (the view heap).
    Views,
    /// A `D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER` heap.
    Samplers,
}

impl HeapKind {
    /// The static name the snapshot prints alongside the heap's start.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Views => "views",
            Self::Samplers => "samplers",
        }
    }
}

/// A bound heap's identity: where its first descriptor is and what it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BoundHeap {
    /// The address of the heap's first descriptor (CPU and GPU start are the
    /// same on the console).
    pub start: u64,
    /// The descriptor kind the heap holds.
    pub kind: HeapKind,
}

impl BoundHeap {
    /// `start:kind` in the snapshot's fixed form (16 hex digits, then kind).
    fn field(&self) -> String {
        format!("{:016x}:{}", self.start, self.kind.name())
    }
}

/// One slot of a `SetDescriptorHeaps` call, as the binding path classified it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapBind {
    /// A heap whose identity the native path can serve.
    Bound(BoundHeap),
    /// A null slot in the bound set.
    NullSlot,
    /// A non-null object with no descriptor start (not a heap we made).
    UnknownObject,
    /// A heap of a kind the native wire cannot express.
    KindNotExpressible,
}

impl HeapBind {
    /// Classifies one `SetDescriptorHeaps` slot from the facts the binding
    /// path has: the heap's first descriptor, whether it is a sampler heap,
    /// and whether that first descriptor is placeable in it.
    ///
    /// Shader-visibility flags are deliberately not an input: a heap whose
    /// flags do not mark it shader-visible is a heap the title still bound, so
    /// it is recorded like any other one (the observed flags-0 boot case).
    pub const fn classify(start: u64, samplers: bool, placeable: bool) -> Self {
        if start == 0 {
            return Self::UnknownObject;
        }
        if !placeable {
            return Self::KindNotExpressible;
        }
        Self::Bound(BoundHeap {
            start,
            kind: if samplers {
                HeapKind::Samplers
            } else {
                HeapKind::Views
            },
        })
    }
}

/// One point-in-time view of the descriptor-heap binding audit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeapBindingSnapshot {
    /// Heap objects recorded as bound across every `SetDescriptorHeaps` call.
    pub bindings: u64,
    /// Slots refused with a static reason.
    pub refusals: u64,
    /// Distinct `(start, kind)` identities bound, saturating at
    /// [`MAX_DISTINCT_HEAPS`].
    pub distinct: u64,
    /// The last bound set's heaps, in bind order, at most
    /// [`MAX_HEAP_BINDINGS`].
    pub last: Vec<BoundHeap>,
    /// Refusals by static class name (only the classes seen).
    pub refusal_classes: BTreeMap<&'static str, u64>,
}

impl HeapBindingSnapshot {
    /// The heap fields of the `native frame telemetry` snapshot line:
    /// `heaps_bound`, `heaps_distinct`, `heaps_last` and
    /// `heap_bind_refusals`.
    pub fn fields(&self) -> String {
        let mut text = format!(
            "heaps_bound={} heaps_distinct={} heaps_last=[",
            self.bindings, self.distinct
        );
        for (index, heap) in self.last.iter().enumerate() {
            if index > 0 {
                text.push_str("; ");
            }
            text.push_str(&heap.field());
        }
        text.push_str("] heap_bind_refusals=");
        text.push_str(&self.refusals.to_string());
        text
    }
}

/// The session totals of the descriptor-heap binding audit.
#[derive(Debug, Default)]
struct HeapBindCensus {
    bindings: u64,
    refusals: u64,
    refusal_classes: BTreeMap<&'static str, u64>,
    distinct: BTreeSet<BoundHeap>,
    last: Vec<BoundHeap>,
}

impl HeapBindCensus {
    /// Counts one `SetDescriptorHeaps` call: bound heaps by identity, refused
    /// slots by their static class, and the call's bound set as `last`.
    fn record(&mut self, set: &[HeapBind]) {
        let mut last = Vec::with_capacity(set.len().min(MAX_HEAP_BINDINGS));
        for bind in set {
            match *bind {
                HeapBind::Bound(heap) => {
                    self.bindings = self.bindings.saturating_add(1);
                    if self.distinct.len() < MAX_DISTINCT_HEAPS {
                        self.distinct.insert(heap);
                    }
                    if last.len() < MAX_HEAP_BINDINGS {
                        last.push(heap);
                    }
                }
                HeapBind::NullSlot => self.refuse(heap_refusals::NULL_SLOT),
                HeapBind::UnknownObject => self.refuse(heap_refusals::UNKNOWN_OBJECT),
                HeapBind::KindNotExpressible => {
                    self.refuse(heap_refusals::KIND_NOT_EXPRESSIBLE)
                }
            }
        }
        // A set that bound nothing leaves an empty last set: the line never
        // keeps a heap the title has since unbound.
        self.last = last;
    }

    fn refuse(&mut self, class: &'static str) {
        self.refusals = self.refusals.saturating_add(1);
        let count = self.refusal_classes.entry(class).or_insert(0);
        *count = count.saturating_add(1);
        if *count == 1 {
            tracing::warn!(refusal = class, "descriptor heap binding refused");
        }
    }

    fn snapshot(&self) -> HeapBindingSnapshot {
        HeapBindingSnapshot {
            bindings: self.bindings,
            refusals: self.refusals,
            distinct: self.distinct.len() as u64,
            last: self.last.clone(),
            refusal_classes: self.refusal_classes.clone(),
        }
    }
}

/// The static reasons a native submission is refused under.
///
/// A submission refusal is counted under exactly one of these names in the same
/// call that moves the refusal total, so a boot can never report a bare
/// `submits=n/m ok/refused` beside an empty reason list. The table is fixed:
/// [`SUBMIT_REFUSAL_NAMES`] is the only set the histogram holds, and an
/// unregistered name is counted under [`UNKNOWN`] instead of growing it.
pub mod submit_refusals {
    /// No Wire v3 client is attached: no endpoint is configured, or the
    /// configured endpoint refused the session.
    pub const NO_CLIENT: &str = "no_client";
    /// The batch carried no submission at all (the list closed with nothing).
    pub const NO_COMMAND_BUFFER: &str = "no_command_buffer";
    /// The batch's submissions carried no commands.
    pub const EMPTY_STREAM: &str = "empty_stream";
    /// The canonical v3 codec refused the submission before it was carried.
    pub const ENCODE: &str = "encode";
    /// The transport failed while carrying the batch on a live session.
    pub const TRANSPORT: &str = "transport";
    /// The host answered a status other than `Ok`.
    pub const HOST_REFUSED: &str = "host_refused";
    /// The host answered a different sequence than the one sent.
    pub const SEQUENCE_MISMATCH: &str = "sequence_mismatch";
    /// The present half reached the submission counter.
    pub const PRESENT_REFUSED: &str = "present_refused";
    /// `ProducerError::InvalidResource`.
    pub const PRODUCER_INVALID_RESOURCE: &str = "producer_invalid_resource";
    /// `ProducerError::InvalidListIdentity`.
    pub const PRODUCER_INVALID_LIST_IDENTITY: &str = "producer_invalid_list_identity";
    /// `ProducerError::Unsupported`.
    pub const PRODUCER_UNSUPPORTED: &str = "producer_unsupported";
    /// `ProducerError::Unregistered`.
    pub const PRODUCER_UNREGISTERED: &str = "producer_unregistered";
    /// `ProducerError::Unresolved`.
    pub const PRODUCER_UNRESOLVED: &str = "producer_unresolved";
    /// `ProducerError::MissingPipeline`.
    pub const PRODUCER_MISSING_PIPELINE: &str = "producer_missing_pipeline";
    /// `ProducerError::NoRenderTargets`.
    pub const PRODUCER_NO_RENDER_TARGETS: &str = "producer_no_render_targets";
    /// `ProducerError::TooManyColorAttachments`.
    pub const PRODUCER_TOO_MANY_COLOR_ATTACHMENTS: &str = "producer_too_many_color_attachments";
    /// `ProducerError::TooManyBarriers`.
    pub const PRODUCER_TOO_MANY_BARRIERS: &str = "producer_too_many_barriers";
    /// `ProducerError::TooManyCommands`.
    pub const PRODUCER_TOO_MANY_COMMANDS: &str = "producer_too_many_commands";
    /// `ProducerError::SequenceExhausted`.
    pub const PRODUCER_SEQUENCE_EXHAUSTED: &str = "producer_sequence_exhausted";
    /// A refusal whose reason is not in [`SUBMIT_REFUSAL_NAMES`].
    pub const UNKNOWN: &str = "unknown_reason";
}

/// The fixed set of submission-refusal names, in a stable order.
///
/// A refusal can only be counted under one of these, so the histogram never
/// grows and its contents do not depend on the order submissions failed in.
pub const SUBMIT_REFUSAL_NAMES: &[&str] = &[
    submit_refusals::NO_CLIENT,
    submit_refusals::NO_COMMAND_BUFFER,
    submit_refusals::EMPTY_STREAM,
    submit_refusals::ENCODE,
    submit_refusals::TRANSPORT,
    submit_refusals::HOST_REFUSED,
    submit_refusals::SEQUENCE_MISMATCH,
    submit_refusals::PRESENT_REFUSED,
    submit_refusals::PRODUCER_INVALID_RESOURCE,
    submit_refusals::PRODUCER_INVALID_LIST_IDENTITY,
    submit_refusals::PRODUCER_UNSUPPORTED,
    submit_refusals::PRODUCER_UNREGISTERED,
    submit_refusals::PRODUCER_UNRESOLVED,
    submit_refusals::PRODUCER_MISSING_PIPELINE,
    submit_refusals::PRODUCER_NO_RENDER_TARGETS,
    submit_refusals::PRODUCER_TOO_MANY_COLOR_ATTACHMENTS,
    submit_refusals::PRODUCER_TOO_MANY_BARRIERS,
    submit_refusals::PRODUCER_TOO_MANY_COMMANDS,
    submit_refusals::PRODUCER_SEQUENCE_EXHAUSTED,
    submit_refusals::UNKNOWN,
];

/// The histogram slot for `reason`, if it is one of [`SUBMIT_REFUSAL_NAMES`].
pub(crate) fn submit_refusal_index(reason: &str) -> Option<usize> {
    SUBMIT_REFUSAL_NAMES.iter().position(|&known| known == reason)
}

/// The table name a submission refusal is counted under.
///
/// An unregistered reason is a caller bug, not a new class: it is logged once
/// and counted under [`submit_refusals::UNKNOWN`], so the refusal total always
/// equals the sum of the named histogram and no count is ever bare.
fn submit_refusal_name(reason: &'static str) -> &'static str {
    match submit_refusal_index(reason) {
        Some(index) => SUBMIT_REFUSAL_NAMES[index],
        None => {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    reason,
                    "an unregistered submission refusal reason was counted under its fallback name"
                );
            });
            submit_refusals::UNKNOWN
        }
    }
}

/// The window and histogram one [`FrameTelemetry`] keeps.
#[derive(Debug, Default)]
struct Window {
    frames: VecDeque<FrameSample>,
    opcodes: BTreeMap<u16, u64>,
    recorded: u64,
}

/// Per-frame telemetry for one process.
#[derive(Debug, Default)]
pub struct FrameTelemetry {
    window: Mutex<Window>,
    /// Hard counter: only [`Self::note_d3dmetal_fallback`] moves it.
    d3dmetal_fallback_count: AtomicU64,
    /// Native submissions the host answered `Ok` (or a replay) for.
    submits_ok: AtomicU64,
    /// Native submissions the transport or the host refused.
    submits_refused: AtomicU64,
    /// Native submission refusals by the static reason they happened for.
    ///
    /// Every increment of [`Self::submits_refused`] writes one entry here in
    /// the same call, so a boot can never report a bare refusal count. Session
    /// totals: like the submit counters, they survive a window reset.
    named_submit_refusals: Mutex<BTreeMap<&'static str, u64>>,
    /// Frames the installed native presenter accepted.
    presents_ok: AtomicU64,
    /// Frames the native presenter (or its absence) refused.
    presents_refused: AtomicU64,
    /// Unsupported producer calls, by the wire opcode they needed.
    refusals: Mutex<BTreeMap<u16, u64>>,
    /// Unsupported producer calls by operation name, with or without a wire
    /// opcode.
    ///
    /// The refusal histogram keys unnamed calls under [`NO_OPCODE`], which
    /// would otherwise merge every one into a single bucket; this map keeps
    /// each refused operation countable by name (ADR-030).
    named_refusals: Mutex<BTreeMap<&'static str, u64>>,
    /// Host allocation registrations the host accepted (kind 2).
    buffers_registered: AtomicU64,
    /// Pipeline registrations the host accepted (kind 4/129).
    psos_registered: AtomicU64,
    /// PSO translations a `--stdio` pool worker ran (cold).
    pool_cold: AtomicU64,
    /// PSO translations the pool's disk cache served (warm).
    pool_warm: AtomicU64,
    /// Buffer or pipeline registrations that failed explicitly.
    registration_failures: AtomicU64,
    /// Wire v3 error replies that carried a host reason trailer.
    host_errors: AtomicU64,
    /// The most recent bounded host error, retained for the telemetry snapshot.
    last_host_error: Mutex<Option<HostError>>,
    /// Compute dispatches recorded with no root view or descriptor table.
    ///
    /// The producer cannot see a pipeline's declared root layout, so it reports
    /// the one fact it has: the recording produced no compute buffer binding
    /// (inline root constants are not buffer bindings). A nonzero count makes an
    /// empty-binding dispatch visible instead of silent.
    compute_dispatches_without_bindings: AtomicU64,
    /// The bounded first-failure latch over the create and map boundary.
    call_failures: CallFailureLatch,
    /// The descriptor-heap binding audit (session totals).
    heap_census: Mutex<HeapBindCensus>,
}
impl FrameTelemetry {
    /// Empty telemetry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one frame with the current monotonic timestamp.
    ///
    /// `opcode` names the wire opcode a refused frame needed
    /// ([`crate::native_metal::producer::ProducerError::Unsupported`]); it is
    /// `None` for a frame the producer fully expressed.
    pub fn record_frame(
        &self,
        record_us: u64,
        ring_submit_us: u64,
        encode_us: u64,
        gpu_us: u64,
        opcode: Option<u16>,
    ) {
        self.record_frame_at(
            monotonic_us(),
            record_us,
            ring_submit_us,
            encode_us,
            gpu_us,
            opcode,
        );
    }

    /// Records one frame at an explicit timestamp (tests and replays).
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame_at(
        &self,
        at_us: u64,
        record_us: u64,
        ring_submit_us: u64,
        encode_us: u64,
        gpu_us: u64,
        opcode: Option<u16>,
    ) {
        let mut window = self.lock();
        if let Some(opcode) = opcode {
            *window.opcodes.entry(opcode).or_insert(0) += 1;
        }
        window.frames.push_back(FrameSample {
            at_us,
            record_us,
            ring_submit_us,
            encode_us,
            gpu_us,
            unsupported_opcode: opcode,
        });
        window.recorded = window.recorded.saturating_add(1);
        while window.frames.len() > WINDOW_FRAMES {
            window.frames.pop_front();
        }
    }

    /// The samples in the window, oldest first.
    pub fn samples(&self) -> Vec<FrameSample> {
        self.lock().frames.iter().copied().collect()
    }

    /// The window's frame costs (`record + ring submit + encode + gpu`), oldest
    /// first.
    pub fn frame_times(&self) -> Vec<u64> {
        self.lock()
            .frames
            .iter()
            .map(FrameSample::frame_us)
            .collect()
    }

    /// Exact nearest-rank p50/p95/p99 of the window.
    pub fn percentiles(&self) -> Percentiles {
        let mut times = self.frame_times();
        times.sort_unstable();
        Percentiles {
            p50: Percentiles::nearest_rank(&times, 50),
            p95: Percentiles::nearest_rank(&times, 95),
            p99: Percentiles::nearest_rank(&times, 99),
        }
    }

    /// The window's sample rate: `(n - 1) / (last - first)` timestamps.
    ///
    /// Zero for fewer than two samples or a zero-length window, so a single
    /// frame never divides by zero.
    pub fn fps(&self) -> f64 {
        let window = self.lock();
        if window.frames.len() < 2 {
            return 0.0;
        }
        let first = window.frames.front().map(|frame| frame.at_us).unwrap_or(0);
        let last = window.frames.back().map(|frame| frame.at_us).unwrap_or(0);
        let span = last.saturating_sub(first);
        if span == 0 {
            return 0.0;
        }
        let intervals = (window.frames.len() - 1) as f64;
        intervals * 1_000_000.0 / span as f64
    }

    /// Frames whose producer needed an unsupported wire opcode, by opcode.
    pub fn unsupported_opcode_histogram(&self) -> BTreeMap<u16, u64> {
        self.lock().opcodes.clone()
    }

    /// Frames recorded since construction (not limited to the window).
    pub fn recorded_frames(&self) -> u64 {
        self.lock().recorded
    }

    /// Frames in the window.
    pub fn window_len(&self) -> usize {
        self.lock().frames.len()
    }

    /// The hard D3DMetal fallback count.
    pub fn d3dmetal_fallback_count(&self) -> u64 {
        self.d3dmetal_fallback_count.load(Ordering::Relaxed)
    }

    /// Declares one D3DMetal fallback. The only way the counter moves.
    pub fn note_d3dmetal_fallback(&self) {
        self.d3dmetal_fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one producer refusal by the wire opcode it needed.
    ///
    /// Called the moment a recorded call is refused (not at submission time),
    /// so a list whose `Close` fails still contributes its histogram entry.
    /// An operation with no wire opcode is logged by the caller and counted
    /// under [`NO_OPCODE`].
    pub fn note_unsupported_opcode(&self, opcode: Option<u16>) {
        let opcode = opcode.unwrap_or(NO_OPCODE);
        let mut refusals = self
            .refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *refusals.entry(opcode).or_insert(0) += 1;
    }

    /// Refusals recorded by opcode ([`NO_OPCODE`] for an unnamed call).
    pub fn refusal_histogram(&self) -> BTreeMap<u16, u64> {
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Records one refusal by the operation name it carries.
    ///
    /// Unlike the opcode histogram this keeps calls that share no wire opcode
    /// distinct: `DiscardResource` and a shading-rate hint never share a
    /// bucket. It is recorded alongside the opcode histogram, not instead of
    /// it.
    pub fn note_named_refusal(&self, operation: &'static str) {
        let mut refusals = self
            .named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *refusals.entry(operation).or_insert(0) += 1;
    }

    /// Refusals recorded by operation name.
    pub fn named_refusal_histogram(&self) -> BTreeMap<&'static str, u64> {
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// A native submission batch the host answered successfully.
    pub fn note_submit_ok(&self) {
        self.submits_ok.fetch_add(1, Ordering::Relaxed);
    }

    /// A native submission batch the transport or host refused.
    ///
    /// `reason` is one of [`SUBMIT_REFUSAL_NAMES`]: the count and its name move
    /// together, so a refusal can never be counted without one. An empty batch
    /// is not a submission at all and must not be counted here.
    pub fn note_submit_refused(&self, reason: &'static str) {
        let reason = submit_refusal_name(reason);
        self.submits_refused.fetch_add(1, Ordering::Relaxed);
        let mut refusals = self
            .named_submit_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *refusals.entry(reason).or_insert(0) += 1;
    }

    /// Native submission refusals by static reason (only the reasons seen).
    pub fn named_submit_refusal_histogram(&self) -> BTreeMap<&'static str, u64> {
        self.named_submit_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// A frame the installed native presenter accepted.
    pub fn note_present_ok(&self) {
        self.presents_ok.fetch_add(1, Ordering::Relaxed);
    }

    /// A frame the native present path refused (including no swapchain).
    pub fn note_present_refused(&self) {
        self.presents_refused.fetch_add(1, Ordering::Relaxed);
    }

    /// One buffer registration the host accepted (kind 2).
    pub fn note_buffer_registered(&self) {
        self.buffers_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO registration the host accepted (kind 4/129).
    pub fn note_pso_registered(&self) {
        self.psos_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO translation a pool worker ran (a cold translation).
    pub fn note_pool_cold(&self) {
        self.pool_cold.fetch_add(1, Ordering::Relaxed);
    }

    /// One PSO translation the pool's disk cache served (a warm translation).
    pub fn note_pool_warm(&self) {
        self.pool_warm.fetch_add(1, Ordering::Relaxed);
    }

    /// One buffer or pipeline registration that failed explicitly.
    pub fn note_registration_failure(&self) {
        self.registration_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one Wire v3 error reply that carried a host reason.
    ///
    /// The reason is sanitised to one line and bounded to
    /// [`MAX_HOST_ERROR_BYTES`], so neither the counter nor the retained last
    /// reason grows without limit. Returns the bounded record for the one log
    /// line the caller prints.
    pub fn note_host_error(&self, sequence: u32, reason: &str) -> HostError {
        let record = HostError {
            sequence,
            reason: sanitize_host_error(reason),
        };
        self.host_errors.fetch_add(1, Ordering::Relaxed);
        *self
            .last_host_error
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(record.clone());
        record
    }

    /// Wire v3 error replies that carried a host reason.
    pub fn host_error_count(&self) -> u64 {
        self.host_errors.load(Ordering::Relaxed)
    }

    /// The most recent bounded host error, if any.
    pub fn last_host_error(&self) -> Option<HostError> {
        self.last_host_error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// One compute dispatch recorded with no root view or descriptor table.
    pub fn note_compute_dispatch_without_bindings(&self) {
        self.compute_dispatches_without_bindings
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Compute dispatches recorded with no root view or descriptor table.
    pub fn compute_dispatches_without_bindings(&self) -> u64 {
        self.compute_dispatches_without_bindings
            .load(Ordering::Relaxed)
    }

    /// Records one failing GPU call in the bounded latch.
    ///
    /// The first failure of each name is logged once at warn; repeated
    /// failures of that name only count.
    pub fn note_call_failure(
        &self,
        name: &'static str,
        hr: i32,
        args: [u64; MAX_CALL_FAILURE_ARGS],
    ) {
        self.call_failures.note(name, hr, args);
    }

    /// Records `hr` only if it is a failure.
    pub fn note_call_result(&self, name: &'static str, hr: i32, args: [u64; MAX_CALL_FAILURE_ARGS]) {
        self.call_failures.note_result(name, hr, args);
    }

    /// Failing GPU calls recorded so far.
    pub fn call_failure_count(&self) -> u64 {
        self.call_failures.count()
    }

    /// Failing GPU calls by export name.
    pub fn call_failure_histogram(&self) -> BTreeMap<&'static str, u64> {
        self.call_failures.histogram()
    }

    /// The one bounded first failure, if any.
    pub fn first_call_failure(&self) -> Option<CallFailure> {
        self.call_failures.first()
    }

    /// Clears the failure latch (tests only).
    #[cfg(test)]
    pub(crate) fn reset_call_failures(&self) {
        self.call_failures.reset();
    }

    /// Records one `SetDescriptorHeaps` call: every slot as classified by the
    /// binding path, bound heaps by identity and refused slots by class.
    pub fn note_heap_bindings(&self, bindings: &[HeapBind]) {
        self.heap_census
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .record(bindings);
    }

    /// The session's descriptor-heap binding audit.
    pub fn heap_bindings(&self) -> HeapBindingSnapshot {
        self.heap_census
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .snapshot()
    }

    /// Accepted buffer and PSO registrations.
    pub fn registration_counts(&self) -> (u64, u64) {
        (
            self.buffers_registered.load(Ordering::Relaxed),
            self.psos_registered.load(Ordering::Relaxed),
        )
    }

    /// Cold (worker) and warm (cache) pool translations.
    pub fn pool_translations(&self) -> (u64, u64) {
        (
            self.pool_cold.load(Ordering::Relaxed),
            self.pool_warm.load(Ordering::Relaxed),
        )
    }

    /// Registration failures counted so far.
    pub fn registration_failures(&self) -> u64 {
        self.registration_failures.load(Ordering::Relaxed)
    }

    /// Successful and refused native submission batches.
    pub fn submit_counts(&self) -> (u64, u64) {
        (
            self.submits_ok.load(Ordering::Relaxed),
            self.submits_refused.load(Ordering::Relaxed),
        )
    }

    /// Accepted and refused native presents.
    pub fn present_counts(&self) -> (u64, u64) {
        (
            self.presents_ok.load(Ordering::Relaxed),
            self.presents_refused.load(Ordering::Relaxed),
        )
    }

    /// The strict guard: `Ok` only while no frame fell back to D3DMetal.
    pub fn assert_no_fallback(&self) -> Result<(), TelemetryError> {
        let count = self.d3dmetal_fallback_count();
        if count == 0 {
            Ok(())
        } else {
            Err(TelemetryError::D3dMetalFallback { count })
        }
    }

    /// The window, percentiles and histogram, without resetting anything.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let (submits_ok, submits_refused) = self.submit_counts();
        let (presents_ok, presents_refused) = self.present_counts();
        TelemetrySnapshot {
            frames: self.window_len(),
            recorded: self.recorded_frames(),
            percentiles: self.percentiles(),
            fps: self.fps(),
            opcodes: self.unsupported_opcode_histogram(),
            refusals: self.refusal_histogram(),
            named_refusals: self.named_refusal_histogram(),
            d3dmetal_fallback_count: self.d3dmetal_fallback_count(),
            submits_ok,
            submits_refused,
            presents_ok,
            presents_refused,
            buffers_registered: self.buffers_registered.load(Ordering::Relaxed),
            psos_registered: self.psos_registered.load(Ordering::Relaxed),
            pool_cold: self.pool_cold.load(Ordering::Relaxed),
            pool_warm: self.pool_warm.load(Ordering::Relaxed),
            registration_failures: self.registration_failures.load(Ordering::Relaxed),
            compute_dispatches_without_bindings: self.compute_dispatches_without_bindings(),
            pipeline_census: crate::native_metal::pso::census(),
            host_errors: self.host_error_count(),
            last_host_error: self.last_host_error(),
            call_failures: self.call_failures.count(),
            first_call_failure: self.call_failures.first(),
            heaps: self.heap_bindings(),
            named_submit_refusals: self.named_submit_refusal_histogram(),
        }
    }

    /// Takes the window and histogram for one log line and clears them.
    ///
    /// The hard fallback counter is deliberately not cleared: it is an
    /// invariant for the whole session, not window state. The heap-binding
    /// audit is a session total too, so the last bound set survives a frame
    /// line that had no `SetDescriptorHeaps` call. The named submission
    /// refusals are session totals for the same reason: the reason a boot's
    /// refusal happened for must not be lost to the next window.
    pub fn snapshot_and_reset(&self) -> TelemetrySnapshot {
        let snapshot = self.snapshot();
        {
            let mut window = self.lock();
            window.frames.clear();
            window.opcodes.clear();
        }
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        snapshot
    }

    /// Clears the window and histograms (never the fallback counter).
    ///
    /// Submit and present counters are session totals and survive a reset too,
    /// as do their named submission-refusal reasons, the pipeline and
    /// heap-binding censuses. A refusal counted in an early window therefore
    /// keeps its name for the rest of the boot.
    pub fn reset(&self) {
        let mut window = self.lock();
        window.frames.clear();
        window.opcodes.clear();
        window.recorded = 0;
        self.refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.named_refusals
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Window> {
        self.window
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// One frame window, ready for the one-line log.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySnapshot {
    /// Frames in the window when it was taken.
    pub frames: usize,
    /// Frames recorded before it was taken.
    pub recorded: u64,
    /// p50/p95/p99 frame cost of the window, in microseconds.
    pub percentiles: Percentiles,
    /// Sample rate of the window.
    pub fps: f64,
    /// Unsupported-opcode histogram of the window.
    pub opcodes: BTreeMap<u16, u64>,
    /// Refused producer calls by opcode ([`NO_OPCODE`] when none applies).
    pub refusals: BTreeMap<u16, u64>,
    /// Refused producer calls by operation name, including those without a
    /// wire opcode.
    pub named_refusals: BTreeMap<&'static str, u64>,
    /// The session's hard D3DMetal fallback count.
    pub d3dmetal_fallback_count: u64,
    /// Native submission batches the host answered successfully.
    pub submits_ok: u64,
    /// Native submission batches the transport or host refused.
    pub submits_refused: u64,
    /// Frames the native presenter accepted.
    pub presents_ok: u64,
    /// Frames the native present path refused.
    pub presents_refused: u64,
    /// Host buffer registrations the host accepted (kind 2).
    pub buffers_registered: u64,
    /// PSO registrations the host accepted (kind 4/129).
    pub psos_registered: u64,
    /// PSO translations a pool worker ran (cold).
    pub pool_cold: u64,
    /// PSO translations the pool's disk cache served (warm).
    pub pool_warm: u64,
    /// Buffer or pipeline registrations that failed explicitly.
    pub registration_failures: u64,
    /// Compute dispatches recorded with no root view or descriptor table.
    pub compute_dispatches_without_bindings: u64,
    /// Native pipeline attempts, creations, reuses and refusals by kind
    /// (session totals: a pipeline made in an early frame stays counted).
    pub pipeline_census: crate::native_metal::pso::PipelineCensusSnapshot,
    /// Wire v3 error replies that carried a host reason.
    pub host_errors: u64,
    /// The most recent bounded host error, if any.
    pub last_host_error: Option<HostError>,
    /// Failing GPU calls recorded so far (session total).
    pub call_failures: u64,
    /// The one bounded first failure, if any.
    pub first_call_failure: Option<CallFailure>,
    /// The descriptor-heap binding audit (session totals: a heap bound in an
    /// early frame stays counted after a window reset).
    pub heaps: HeapBindingSnapshot,
    /// Native submission refusals by static reason (session totals: a refusal
    /// in an early window stays counted, so the reason of the boot's one
    /// refusal is never lost to a later reset).
    pub named_submit_refusals: BTreeMap<&'static str, u64>,
}

impl std::fmt::Display for TelemetrySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "native frame: {} in window ({}) {:.3}/{:.3}/{:.3} ms p50/p95/p99, {:.1} fps, \
             fallback={}",
            self.frames,
            self.recorded,
            self.percentiles.p50 as f64 / 1000.0,
            self.percentiles.p95 as f64 / 1000.0,
            self.percentiles.p99 as f64 / 1000.0,
            self.fps,
            self.d3dmetal_fallback_count
        )?;
        write!(
            f,
            ", submits={}/{} ok/refused, presents={}/{} ok/refused",
            self.submits_ok, self.submits_refused, self.presents_ok, self.presents_refused
        )?;
        write!(
            f,
            ", registered={} buffers/{} psos, pool={} cold/{} warm, registration_failures={}, \
             compute_unbound={}",
            self.buffers_registered,
            self.psos_registered,
            self.pool_cold,
            self.pool_warm,
            self.registration_failures,
            self.compute_dispatches_without_bindings
        )?;
        write!(f, ", {}", self.pipeline_census.fields())?;
        write!(f, ", {}", self.heaps.fields())?;
        write!(f, "{}", opcode_list(&self.opcodes, "opcodes"))?;
        write!(f, "{}", opcode_list(&self.refusals, "refusals"))?;
        if self.named_refusals.is_empty() {
            write!(f, ", named_refusals=[]")?;
        } else {
            write!(f, ", named_refusals=[")?;
            for (index, (operation, count)) in self.named_refusals.iter().enumerate() {
                if index > 0 {
                    write!(f, "; ")?;
                }
                write!(f, "{operation}={count}")?;
            }
            write!(f, "]")?;
        }
        write!(f, ", host_errors={}", self.host_errors)?;
        if let Some(error) = &self.last_host_error {
            write!(
                f,
                ", last_host_error=sequence {}: {}",
                error.sequence, error.reason
            )?;
        }
        write!(f, ", call_failures={}", self.call_failures)?;
        match &self.first_call_failure {
            Some(failure) => write!(
                f,
                " first_call_failure={} hr={:#x}",
                failure.name, failure.hr as u32
            )?,
            None => write!(f, " first_call_failure=none hr=0x0")?,
        }
        write!(
            f,
            "{}",
            name_count_list(&self.named_submit_refusals, "named_submit_refusals")
        )?;
        Ok(())
    }
}

/// One `, name=[key=count; ...]` field for the frame line, empty list included.
fn name_count_list(counts: &BTreeMap<&'static str, u64>, name: &str) -> String {
    if counts.is_empty() {
        return format!(", {name}=[]");
    }
    let mut text = format!(", {name}=[");
    for (index, (key, count)) in counts.iter().enumerate() {
        if index > 0 {
            text.push_str("; ");
        }
        text.push_str(key);
        text.push('=');
        text.push_str(&count.to_string());
    }
    text.push(']');
    text
}

/// One `, name=[opcode:count ...]` field for the frame line.
fn opcode_list(opcodes: &BTreeMap<u16, u64>, name: &str) -> String {
    if opcodes.is_empty() {
        return format!(", {name}=[]");
    }
    let mut text = format!(", {name}=[");
    for (index, (opcode, count)) in opcodes.iter().enumerate() {
        if index > 0 {
            text.push(' ');
        }
        if *opcode == NO_OPCODE {
            text.push_str("none");
        } else {
            text.push_str(&opcode.to_string());
        }
        text.push(':');
        text.push_str(&count.to_string());
    }
    text.push(']');
    text
}

/// Sanitises a host diagnostic into one bounded UTF-8 line.
///
/// Control characters become single spaces, whitespace is trimmed and the text
/// is truncated on a character boundary at [`MAX_HOST_ERROR_BYTES`]. Empty text
/// becomes a stable placeholder so the log line never ends in `reason=`.
pub fn sanitize_host_error(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len().min(MAX_HOST_ERROR_BYTES));
    for character in text.chars() {
        let character = if character.is_control() {
            ' '
        } else {
            character
        };
        if cleaned.len() + character.len_utf8() > MAX_HOST_ERROR_BYTES {
            break;
        }
        if character == ' ' && (cleaned.is_empty() || cleaned.ends_with(' ')) {
            continue;
        }
        cleaned.push(character);
    }
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        "unknown host error".to_string()
    } else {
        cleaned.to_string()
    }
}

/// The stable single guest log line for one Wire v3 host error.
///
/// The format is fixed so an operator can grep a boot log for
/// `METAL_HOST_ERROR sequence=` and read the exact host/Metal text.
pub fn host_error_line(sequence: u32, reason: &str) -> String {
    format!("METAL_HOST_ERROR sequence={sequence} reason={reason}")
}

/// The process-wide telemetry the entry points record into.
pub fn global() -> &'static FrameTelemetry {
    static TELEMETRY: OnceLock<FrameTelemetry> = OnceLock::new();
    TELEMETRY.get_or_init(FrameTelemetry::new)
}

/// Records one frame into the process-wide telemetry.
pub fn record_frame(
    record_us: u64,
    ring_submit_us: u64,
    encode_us: u64,
    gpu_us: u64,
    opcode: Option<u16>,
) {
    global().record_frame(record_us, ring_submit_us, encode_us, gpu_us, opcode);
}

/// Records one frame at an explicit timestamp into the process-wide telemetry.
pub fn record_frame_at(
    at_us: u64,
    record_us: u64,
    ring_submit_us: u64,
    encode_us: u64,
    gpu_us: u64,
    opcode: Option<u16>,
) {
    global().record_frame_at(at_us, record_us, ring_submit_us, encode_us, gpu_us, opcode);
}

/// The process-wide window's frame costs, oldest first.
pub fn frame_times() -> Vec<u64> {
    global().frame_times()
}

/// The process-wide exact p50/p95/p99.
pub fn percentiles() -> Percentiles {
    global().percentiles()
}

/// The process-wide sample rate.
pub fn fps() -> f64 {
    global().fps()
}

/// The process-wide unsupported-opcode histogram.
pub fn unsupported_opcode_histogram() -> BTreeMap<u16, u64> {
    global().unsupported_opcode_histogram()
}

/// The process-wide hard D3DMetal fallback count.
pub fn d3dmetal_fallback_count() -> u64 {
    global().d3dmetal_fallback_count()
}

/// One buffer registration the host accepted (kind 2).
pub fn note_buffer_registered() {
    global().note_buffer_registered();
}

/// One PSO registration the host accepted (kind 4/129).
pub fn note_pso_registered() {
    global().note_pso_registered();
}

/// One cold (worker) PSO translation.
pub fn note_pool_cold() {
    global().note_pool_cold();
}

/// One warm (disk-cache) PSO translation.
pub fn note_pool_warm() {
    global().note_pool_warm();
}

/// One buffer or pipeline registration that failed explicitly.
pub fn note_registration_failure() {
    global().note_registration_failure();
}

/// Records one Wire v3 host error reason in the process-wide telemetry.
pub fn note_host_error(sequence: u32, reason: &str) -> HostError {
    global().note_host_error(sequence, reason)
}

/// The process-wide count of Wire v3 error replies that carried a reason.
pub fn host_error_count() -> u64 {
    global().host_error_count()
}

/// The process-wide most recent bounded host error.
pub fn last_host_error() -> Option<HostError> {
    global().last_host_error()
}

/// One compute dispatch recorded with no root view or descriptor table.
pub fn note_compute_dispatch_without_bindings() {
    global().note_compute_dispatch_without_bindings();
}

/// The process-wide accepted buffer and PSO registration counts.
pub fn registration_counts() -> (u64, u64) {
    global().registration_counts()
}

/// The process-wide cold/warm pool translation counts.
pub fn pool_translations() -> (u64, u64) {
    global().pool_translations()
}

/// The process-wide registration failure count.
pub fn registration_failures() -> u64 {
    global().registration_failures()
}

/// The process-wide count of compute dispatches without a root binding.
pub fn compute_dispatches_without_bindings() -> u64 {
    global().compute_dispatches_without_bindings()
}

/// Records one failing GPU call in the process-wide latch.
pub fn note_call_failure(name: &'static str, hr: i32, args: [u64; MAX_CALL_FAILURE_ARGS]) {
    global().note_call_failure(name, hr, args);
}

/// Records a GPU call's `hr` in the process-wide latch only if it failed.
pub fn note_call_result(name: &'static str, hr: i32, args: [u64; MAX_CALL_FAILURE_ARGS]) {
    global().note_call_result(name, hr, args);
}

/// The process-wide count of failing GPU calls.
pub fn call_failure_count() -> u64 {
    global().call_failure_count()
}

/// The process-wide histogram of failing GPU calls by export name.
pub fn call_failure_histogram() -> BTreeMap<&'static str, u64> {
    global().call_failure_histogram()
}

/// The process-wide bounded first failure, if any.
pub fn first_call_failure() -> Option<CallFailure> {
    global().first_call_failure()
}

/// Records one `SetDescriptorHeaps` call in the process-wide telemetry.
///
/// Every slot is recorded: a heap the title bound is never silently dropped,
/// and a null or foreign slot is counted under its static refusal name.
pub fn note_heap_bindings(bindings: &[HeapBind]) {
    global().note_heap_bindings(bindings);
}

/// The process-wide descriptor-heap binding audit.
pub fn heap_bindings() -> HeapBindingSnapshot {
    global().heap_bindings()
}

/// Declares one D3DMetal fallback in the process-wide telemetry.
pub fn note_d3dmetal_fallback() {
    global().note_d3dmetal_fallback();
}

/// Records one refused producer call by the wire opcode it needed.
pub fn note_unsupported_opcode(opcode: Option<u16>) {
    global().note_unsupported_opcode(opcode);
}

/// The process-wide refusal histogram.
pub fn refusal_histogram() -> BTreeMap<u16, u64> {
    global().refusal_histogram()
}

/// Records one refused producer call by its operation name.
pub fn note_named_refusal(operation: &'static str) {
    global().note_named_refusal(operation);
}

/// The process-wide refusal histogram by operation name.
pub fn named_refusal_histogram() -> BTreeMap<&'static str, u64> {
    global().named_refusal_histogram()
}

/// Records one successful native submission batch.
pub fn note_submit_ok() {
    global().note_submit_ok();
}

/// Records one refused native submission batch under its static reason.
pub fn note_submit_refused(reason: &'static str) {
    global().note_submit_refused(reason);
}

/// Records one accepted native present.
pub fn note_present_ok() {
    global().note_present_ok();
}

/// Records one refused native present.
pub fn note_present_refused() {
    global().note_present_refused();
}

/// The process-wide strict zero-D3DMetal guard.
pub fn assert_no_fallback() -> Result<(), TelemetryError> {
    global().assert_no_fallback()
}

/// The process-wide snapshot, without resetting.
pub fn snapshot() -> TelemetrySnapshot {
    global().snapshot()
}

/// Takes the process-wide window for one line and clears it.
pub fn snapshot_and_reset() -> TelemetrySnapshot {
    global().snapshot_and_reset()
}

/// Logs the process-wide window as one line and starts a new window.
pub fn log_frame() {
    let snapshot = snapshot_and_reset();
    tracing::info!(%snapshot, "native frame telemetry");
}

/// Microseconds since the first call, for sample timestamps.
fn monotonic_us() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn telemetry() -> FrameTelemetry {
        FrameTelemetry::new()
    }

    /// A frame whose only cost is `record_us`.
    fn frame(record_us: u64) {
        global().record_frame_at(0, record_us, 0, 0, 0, None);
    }

    #[test]
    fn percentiles_are_exact_on_a_hundred_sample_fixture() {
        let telemetry = telemetry();
        for cost in 1..=100u64 {
            telemetry.record_frame_at(cost, cost, 0, 0, 0, None);
        }
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 50,
                p95: 95,
                p99: 99
            }
        );
        assert_eq!(telemetry.recorded_frames(), 100);
        assert_eq!(telemetry.frame_times().len(), 100);
        assert_eq!(telemetry.frame_times()[99], 100, "oldest first");
    }

    #[test]
    fn one_sample_answers_every_percentile() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 7, 0, 0, 0, None);
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 7,
                p95: 7,
                p99: 7
            }
        );
        assert_eq!(telemetry.fps(), 0.0, "one sample has no interval");
    }

    #[test]
    fn two_samples_use_nearest_rank() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 20, 0, 0, 0, None);
        telemetry.record_frame_at(1, 10, 0, 0, 0, None);
        // Sorted: [10, 20]. ceil(0.5 * 2) = 1 -> 10; ceil(0.95 * 2) = 2 -> 20.
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 10,
                p95: 20,
                p99: 20
            }
        );
    }

    #[test]
    fn an_empty_or_zero_length_window_is_zero_instead_of_panicking() {
        let telemetry = telemetry();
        assert_eq!(telemetry.percentiles(), Percentiles::default());
        assert_eq!(telemetry.fps(), 0.0);
        assert_eq!(telemetry.frame_times(), Vec::<u64>::new());
        assert_eq!(telemetry.recorded_frames(), 0);

        // Two samples at the same instant: no interval, no division by zero.
        telemetry.record_frame_at(50, 1, 0, 0, 0, None);
        telemetry.record_frame_at(50, 2, 0, 0, 0, None);
        assert_eq!(telemetry.fps(), 0.0);
        assert_eq!(
            telemetry.percentiles(),
            Percentiles {
                p50: 1,
                p95: 2,
                p99: 2
            }
        );
    }

    #[test]
    fn fps_is_the_window_sample_rate() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 1, 0, 0, 0, None);
        telemetry.record_frame_at(1_000_000, 1, 0, 0, 0, None);
        telemetry.record_frame_at(2_000_000, 1, 0, 0, 0, None);
        assert_eq!(telemetry.fps(), 1.0, "two intervals over two seconds");
    }

    #[test]
    fn the_frame_time_is_the_sum_of_every_instrumented_phase() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 100, 200, 300, 400, None);
        assert_eq!(telemetry.frame_times(), vec![1000]);
        let sample = telemetry.samples()[0];
        assert_eq!(sample.record_us, 100);
        assert_eq!(sample.ring_submit_us, 200);
        assert_eq!(sample.encode_us, 300);
        assert_eq!(sample.gpu_us, 400);
        assert_eq!(sample.unsupported_opcode, None);
    }

    #[test]
    fn the_opcode_histogram_counts_only_named_opcodes() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 1, 0, 0, 0, Some(7));
        telemetry.record_frame_at(1, 1, 0, 0, 0, Some(7));
        telemetry.record_frame_at(2, 1, 0, 0, 0, Some(9));
        telemetry.record_frame_at(3, 1, 0, 0, 0, None);
        assert_eq!(
            telemetry.unsupported_opcode_histogram(),
            BTreeMap::from([(7, 2), (9, 1)])
        );
        assert_eq!(telemetry.window_len(), 4);
    }

    #[test]
    fn the_fallback_counter_only_moves_through_its_note() {
        let telemetry = telemetry();
        assert_eq!(telemetry.d3dmetal_fallback_count(), 0);
        assert_eq!(telemetry.assert_no_fallback(), Ok(()));

        telemetry.record_frame_at(0, 1, 0, 0, 0, None);
        telemetry.snapshot_and_reset();
        telemetry.reset();
        assert_eq!(
            telemetry.d3dmetal_fallback_count(),
            0,
            "recording and resetting never move the hard counter"
        );

        telemetry.note_d3dmetal_fallback();
        assert_eq!(telemetry.d3dmetal_fallback_count(), 1);
        assert_eq!(
            telemetry.assert_no_fallback(),
            Err(TelemetryError::D3dMetalFallback { count: 1 })
        );
        telemetry.snapshot_and_reset();
        assert_eq!(
            telemetry.d3dmetal_fallback_count(),
            1,
            "the snapshot never clears the hard counter"
        );
    }

    #[test]
    fn snapshot_and_reset_returns_the_window_and_starts_a_new_one() {
        let telemetry = telemetry();
        telemetry.record_frame_at(0, 10, 0, 0, 0, Some(3));
        telemetry.record_frame_at(1, 30, 0, 0, 0, None);
        let snapshot = telemetry.snapshot_and_reset();
        assert_eq!(snapshot.frames, 2);
        assert_eq!(snapshot.recorded, 2);
        assert_eq!(
            snapshot.percentiles,
            Percentiles {
                p50: 10,
                p95: 30,
                p99: 30
            }
        );
        assert_eq!(snapshot.opcodes, BTreeMap::from([(3, 1)]));
        assert_eq!(snapshot.d3dmetal_fallback_count, 0);
        let line = snapshot.to_string();
        assert!(
            !line.contains('\n'),
            "the log line must be one line: {line}"
        );
        assert!(line.contains("fallback=0"), "{line}");
        assert!(line.contains("opcodes=[3:1]"), "{line}");

        assert_eq!(telemetry.window_len(), 0);
        assert_eq!(telemetry.unsupported_opcode_histogram(), BTreeMap::new());
        assert_eq!(telemetry.recorded_frames(), 2, "totals survive a reset");
    }

    #[test]
    fn the_window_is_bounded() {
        let telemetry = telemetry();
        for cost in 0..(WINDOW_FRAMES as u64 + 10) {
            telemetry.record_frame_at(cost, cost, 0, 0, 0, None);
        }
        assert_eq!(telemetry.window_len(), WINDOW_FRAMES);
        assert_eq!(telemetry.recorded_frames() as usize, WINDOW_FRAMES + 10);
        assert_eq!(telemetry.frame_times()[0], 10, "the oldest samples dropped");
    }

    /// The snapshot names the unsupported opcodes and the submit/present
    /// statuses the native path observed, with the hard fallback count.
    /// An empty-binding compute dispatch is a named count, not silence.
    #[test]
    fn unbound_compute_dispatches_are_counted_by_name() {
        let telemetry = telemetry();
        assert_eq!(telemetry.compute_dispatches_without_bindings(), 0);
        telemetry.note_compute_dispatch_without_bindings();
        telemetry.note_compute_dispatch_without_bindings();
        assert_eq!(telemetry.compute_dispatches_without_bindings(), 2);

        let line = telemetry.snapshot().to_string();
        assert!(line.contains("compute_unbound=2"), "{line}");
        telemetry.reset();
        assert_eq!(
            telemetry.compute_dispatches_without_bindings(),
            2,
            "the count is a session total, like the submit counters"
        );
    }

    /// The snapshot names the pipeline-attempt census: how many attempts the
    /// title made, how many became host pipelines and, when one was refused,
    /// the static name and bounded reason of the first refusal. A boot can then
    /// tell "no pipeline asked for yet" from "every request refused".
    #[test]
    fn the_snapshot_carries_the_pipeline_attempt_census() {
        use crate::native_metal::pso;

        pso::reset_census();
        pso::note_created(pso::AttemptKind::Graphics);
        pso::note_translator_unavailable(
            pso::AttemptKind::Compute,
            pso::RefusalClass::ConverterUnavailable,
            "shader pool has no worker program configured",
        );

        let snapshot = telemetry().snapshot();
        assert_eq!(snapshot.pipeline_census.attempts, 2);
        assert_eq!(snapshot.pipeline_census.created, 1);
        assert_eq!(snapshot.pipeline_census.refused, 0);
        assert_eq!(snapshot.pipeline_census.translator_unavailable, 1);
        let line = snapshot.to_string();
        for expected in [
            "pipeline_attempts=2",
            "pipeline_created=1",
            "pipeline_refused=0",
            "pipeline_translator_unavailable=1",
            "pipeline_first_refusal=compute/converter-unavailable",
        ] {
            assert!(line.contains(expected), "{expected} missing from {line}");
        }
        assert!(!line.contains('\n'), "the log line must stay one line: {line}");
    }

    /// Binding 1, then 2, then 8 heaps records the counts, the distinct
    /// identities and the last set in bind order. Repeating an identity in a
    /// later set counts the binding but not a second distinct heap.
    #[test]
    fn heap_bindings_count_distinct_and_keep_the_last_set() {
        let telemetry = telemetry();
        let view = |start: u64| {
            HeapBind::Bound(BoundHeap {
                start,
                kind: HeapKind::Views,
            })
        };

        telemetry.note_heap_bindings(&[view(0x1000)]);
        let snapshot = telemetry.heap_bindings();
        assert_eq!((snapshot.bindings, snapshot.distinct), (1, 1));
        assert_eq!(
            snapshot.last,
            vec![BoundHeap {
                start: 0x1000,
                kind: HeapKind::Views
            }]
        );

        telemetry.note_heap_bindings(&[view(0x1000), view(0x2000)]);
        let snapshot = telemetry.heap_bindings();
        assert_eq!(
            (snapshot.bindings, snapshot.distinct),
            (3, 2),
            "a repeated heap counts once as a binding and not again as distinct"
        );
        assert_eq!(snapshot.last.len(), 2);
        assert_eq!(
            snapshot.last[0],
            BoundHeap {
                start: 0x1000,
                kind: HeapKind::Views
            }
        );
        assert_eq!(
            snapshot.last[1],
            BoundHeap {
                start: 0x2000,
                kind: HeapKind::Views
            }
        );

        let eight: Vec<HeapBind> = (0..8).map(|i| view(0x3000 + i * 0x20)).collect();
        telemetry.note_heap_bindings(&eight);
        let snapshot = telemetry.heap_bindings();
        assert_eq!((snapshot.bindings, snapshot.distinct), (11, 10));
        let last: Vec<BoundHeap> = eight
            .iter()
            .map(|bind| match bind {
                HeapBind::Bound(heap) => *heap,
                other => panic!("a bound slot: {other:?}"),
            })
            .collect();
        assert_eq!(snapshot.last, last, "the last set keeps bind order");
        assert_eq!(snapshot.refusals, 0);
    }

    /// A null slot, an object with no heap identity and an unexpressible kind
    /// are refusals with distinct static names, never silent skips.
    #[test]
    fn null_and_unknown_heaps_are_named_refusals() {
        let telemetry = telemetry();
        telemetry.note_heap_bindings(&[
            HeapBind::NullSlot,
            HeapBind::UnknownObject,
            HeapBind::KindNotExpressible,
            HeapBind::classify(0, false, true),
        ]);

        let snapshot = telemetry.heap_bindings();
        assert_eq!(snapshot.bindings, 0);
        assert_eq!(snapshot.distinct, 0);
        assert!(snapshot.last.is_empty());
        assert_eq!(snapshot.refusals, 4);
        assert_eq!(
            snapshot.refusal_classes,
            BTreeMap::from([
                (heap_refusals::NULL_SLOT, 1),
                (heap_refusals::UNKNOWN_OBJECT, 2),
                (heap_refusals::KIND_NOT_EXPRESSIBLE, 1),
            ])
        );

        let line = telemetry.snapshot().to_string();
        assert!(line.contains("heap_bind_refusals=4"), "{line}");
        assert!(line.contains("heaps_last=[]"), "{line}");
    }

    /// The boot's observed case: a heap whose flags do not mark it
    /// shader-visible (`flags 0`) is still bound and recorded. Flags are not
    /// an input to the classifier at all.
    #[test]
    fn a_heap_with_flags_zero_is_still_bound_and_recorded() {
        let start = 0x10_0000_0000;
        let bind = HeapBind::classify(start, false, true);
        assert_eq!(
            bind,
            HeapBind::Bound(BoundHeap {
                start,
                kind: HeapKind::Views,
            })
        );

        let telemetry = telemetry();
        telemetry.note_heap_bindings(&[bind]);
        let snapshot = telemetry.heap_bindings();
        assert_eq!(
            (snapshot.bindings, snapshot.distinct, snapshot.refusals),
            (1, 1, 0)
        );
        let line = telemetry.snapshot().to_string();
        assert!(line.contains("heaps_bound=1 heaps_distinct=1"), "{line}");
        assert!(
            line.contains("heaps_last=[0000001000000000:views]"),
            "{line}"
        );
    }

    /// The snapshot's heap fields are deterministic, bounded to eight pairs
    /// and formatted as 16 hex digits plus the kind.
    #[test]
    fn the_heap_snapshot_line_is_deterministic_and_bounded() {
        let telemetry = telemetry();
        let twelve: Vec<HeapBind> = (0..12u64)
            .map(|i| HeapBind::classify(0x2000 + i * 0x20, i == 0, true))
            .collect();
        telemetry.note_heap_bindings(&twelve);

        let first = telemetry.snapshot().to_string();
        let second = telemetry.snapshot().to_string();
        assert_eq!(first, second, "the same census prints the same line");
        assert!(
            first.contains("heaps_bound=12 heaps_distinct=12"),
            "{first}"
        );
        assert!(first.contains("heap_bind_refusals=0"), "{first}");
        assert!(!first.contains('\n'), "one line: {first}");

        let start = first.find("heaps_last=[").expect("the field") + "heaps_last=[".len();
        let end = start + first[start..].find(']').expect("the field's end");
        let pairs = &first[start..end];
        assert_eq!(pairs.matches(':').count(), 8, "bounded to eight pairs: {pairs}");
        assert!(
            pairs.starts_with("0000000000002000:samplers; 0000000000002020:views"),
            "{pairs}"
        );
    }

    #[test]
    fn the_snapshot_carries_refusals_and_submit_present_statuses() {
        let telemetry = telemetry();
        telemetry.note_unsupported_opcode(Some(44));
        telemetry.note_unsupported_opcode(Some(44));
        telemetry.note_unsupported_opcode(None);
        telemetry.note_submit_ok();
        telemetry.note_submit_refused(submit_refusals::HOST_REFUSED);
        telemetry.note_present_ok();
        telemetry.note_present_refused();
        telemetry.note_present_refused();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.refusals, BTreeMap::from([(44, 2), (NO_OPCODE, 1)]));
        assert_eq!((snapshot.submits_ok, snapshot.submits_refused), (1, 1));
        assert_eq!((snapshot.presents_ok, snapshot.presents_refused), (1, 2));
        assert_eq!(snapshot.d3dmetal_fallback_count, 0);
        let line = snapshot.to_string();
        assert!(line.contains("refusals=[44:2 none:1]"), "{line}");
        assert!(line.contains("submits=1/1"), "{line}");
        assert!(line.contains("presents=1/2"), "{line}");
        assert!(line.contains("fallback=0"), "{line}");
        // The session totals survive a window reset.
        telemetry.reset();
        let after = telemetry.snapshot();
        assert_eq!(after.refusals, BTreeMap::new());
        assert_eq!((after.submits_ok, after.submits_refused), (1, 1));
        assert_eq!((after.presents_ok, after.presents_refused), (1, 2));
    }

    /// ADR-030: a refusal without a wire opcode is still counted by the
    /// operation name, so two different console states never share a bucket.
    #[test]
    fn unnamed_refusals_are_counted_by_operation_name() {
        let telemetry = telemetry();
        telemetry.note_unsupported_opcode(None);
        telemetry.note_named_refusal("DiscardResource: a partial discard");
        telemetry.note_named_refusal("DiscardResource: a partial discard");
        telemetry.note_named_refusal("SetShadingRate: no Wire v3 command");

        assert_eq!(
            telemetry.named_refusal_histogram(),
            BTreeMap::from([
                ("DiscardResource: a partial discard", 2),
                ("SetShadingRate: no Wire v3 command", 1),
            ])
        );
        // The opcode histogram keeps its own unchanged view.
        assert_eq!(telemetry.refusal_histogram(), BTreeMap::from([(NO_OPCODE, 1)]));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.named_refusals.len(), 2);
        assert!(
            snapshot.to_string().contains("named_refusals=["),
            "{snapshot}"
        );

        telemetry.snapshot_and_reset();
        assert_eq!(telemetry.named_refusal_histogram(), BTreeMap::new());
    }

    /// A host error is bounded, flattened to one line, counted and retained as
    /// the last reason; the snapshot names both without unbounded growth.
    #[test]
    fn host_errors_are_bounded_counted_and_named_once() {
        let telemetry = telemetry();
        assert_eq!(telemetry.host_error_count(), 0);
        assert_eq!(telemetry.last_host_error(), None);

        let short = "Metal execution failed: target texture 640x360 was refused";
        let record = telemetry.note_host_error(7, short);
        assert_eq!(record.sequence, 7);
        assert_eq!(record.reason, short);
        assert_eq!(telemetry.host_error_count(), 1);
        assert_eq!(
            telemetry.last_host_error(),
            Some(HostError {
                sequence: 7,
                reason: short.to_string()
            })
        );
        assert_eq!(
            host_error_line(record.sequence, &record.reason),
            format!("METAL_HOST_ERROR sequence=7 reason={short}")
        );

        let long = "x".repeat(MAX_HOST_ERROR_BYTES * 2);
        let record = telemetry.note_host_error(8, &long);
        assert_eq!(record.reason.len(), MAX_HOST_ERROR_BYTES);
        assert!(record.reason.is_char_boundary(record.reason.len()));

        let unicode = "é".repeat(MAX_HOST_ERROR_BYTES);
        let record = telemetry.note_host_error(9, &unicode);
        assert!(record.reason.len() <= MAX_HOST_ERROR_BYTES);
        assert!(record.reason.is_char_boundary(record.reason.len()));

        let multiline = telemetry.note_host_error(10, "line one\n\tline two");
        assert_eq!(multiline.reason, "line one line two");

        assert_eq!(telemetry.host_error_count(), 4);
        assert_eq!(
            telemetry.last_host_error().map(|error| error.sequence),
            Some(10)
        );
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.host_errors, 4);
        assert_eq!(snapshot.last_host_error, Some(multiline));
        let line = snapshot.to_string();
        assert!(line.contains("host_errors=4"), "{line}");
        assert!(line.contains("last_host_error=sequence 10"), "{line}");
        assert!(!line.contains('\n'), "the log line must stay one line: {line}");
    }

    /// A failure is counted once per call, once per name, and the first one
    /// keeps its name, HRESULT and scalar arguments.
    #[test]
    fn call_failures_are_counted_once_per_name_and_keep_one_record() {
        use crate::object::S_OK;

        let telemetry = telemetry();
        telemetry.note_call_failure(call_names::MAP, S_OK, [7, 8, 9]);
        telemetry.note_call_failure(call_names::MAP, S_OK, [7, 8, 9]);
        telemetry.note_call_failure(call_names::CREATE_DESCRIPTOR_HEAP, -1, [4, 5, 6]);

        assert_eq!(telemetry.call_failure_count(), 3);
        assert_eq!(
            telemetry.call_failure_histogram(),
            BTreeMap::from([
                (call_names::CREATE_DESCRIPTOR_HEAP, 1),
                (call_names::MAP, 2),
            ])
        );
        assert_eq!(
            telemetry.first_call_failure(),
            Some(CallFailure {
                name: call_names::MAP,
                hr: S_OK,
                args: [7, 8, 9],
            }),
            "only the first failure is kept"
        );

        let line = telemetry.snapshot().to_string();
        assert!(line.contains("call_failures=3"), "{line}");
        assert!(
            line.contains("call_failures=3 first_call_failure=Map hr=0x0"),
            "{line}"
        );
        assert!(!line.contains('\n'), "one line: {line}");
    }

    /// A successful call records nothing.
    #[test]
    fn a_successful_call_records_nothing() {
        use crate::object::S_OK;

        let telemetry = telemetry();
        telemetry.note_call_result(call_names::MAP, S_OK, [1, 2, 3]);
        telemetry.note_call_result(call_names::CREATE_DESCRIPTOR_HEAP, 1, [0, 0, 0]);

        assert_eq!(telemetry.call_failure_count(), 0);
        assert_eq!(telemetry.call_failure_histogram(), BTreeMap::new());
        assert_eq!(telemetry.first_call_failure(), None);

        let line = telemetry.snapshot().to_string();
        assert!(line.contains("call_failures=0"), "{line}");
        assert!(line.contains("first_call_failure=none hr=0x0"), "{line}");
    }

    /// The first-failure record is bounded: a long static name is truncated
    /// on a character boundary, never growing the snapshot line.
    #[test]
    fn the_first_failure_record_is_bounded() {
        const LONG_NAME: &str = "CreateSomethingWithAnExtremelyLongExportNameThatKeepsGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoingAndGoing";

        let telemetry = telemetry();
        telemetry.note_call_failure(LONG_NAME, -1, [u64::MAX, 0, 1]);

        let first = telemetry.first_call_failure().expect("a failure");
        assert_eq!(first.name, LONG_NAME, "the static name itself is kept");
        let line = first.line();
        assert!(line.len() <= MAX_CALL_FAILURE_BYTES, "{} bytes", line.len());
        assert!(line.is_char_boundary(line.len()));
        assert!(!line.contains('\n'));
        assert!(
            LONG_NAME.len() > MAX_CALL_FAILURE_BYTES,
            "the fixture is long enough to truncate"
        );
    }

    /// The histogram is the same whatever order the failures arrived in.
    #[test]
    fn the_call_failure_histogram_is_deterministic() {
        let first = telemetry();
        for (name, count) in [
            (call_names::MAP, 3),
            (call_names::CREATE_DESCRIPTOR_HEAP, 1),
            (call_names::COPY_DESCRIPTORS_SIMPLE, 2),
        ] {
            for _ in 0..count {
                first.note_call_failure(name, -1, [0; MAX_CALL_FAILURE_ARGS]);
            }
        }

        let second = telemetry();
        for (name, count) in [
            (call_names::COPY_DESCRIPTORS_SIMPLE, 2),
            (call_names::CREATE_DESCRIPTOR_HEAP, 1),
            (call_names::MAP, 3),
        ] {
            for _ in 0..count {
                second.note_call_failure(name, -1, [0; MAX_CALL_FAILURE_ARGS]);
            }
        }

        assert_eq!(first.call_failure_histogram(), second.call_failure_histogram());
        assert_eq!(
            first.call_failure_histogram().keys().copied().collect::<Vec<_>>(),
            vec![
                call_names::COPY_DESCRIPTORS_SIMPLE,
                call_names::CREATE_DESCRIPTOR_HEAP,
                call_names::MAP,
            ],
            "sorted, not insertion order"
        );
    }


    #[test]
    fn the_global_recorders_reach_the_process_telemetry() {
        global().reset();
        frame(5);
        frame(9);
        assert_eq!(frame_times(), vec![5, 9]);
        assert_eq!(
            percentiles(),
            Percentiles {
                p50: 5,
                p95: 9,
                p99: 9
            }
        );
        assert_eq!(d3dmetal_fallback_count(), 0);
        assert_eq!(assert_no_fallback(), Ok(()));
        let snapshot = snapshot_and_reset();
        assert_eq!(snapshot.frames, 2);
        assert_eq!(global().window_len(), 0);
    }

    /// Every reason in the fixed table is countable under its own name: no
    /// reason shares a bucket and none is a bare count.
    #[test]
    fn every_submit_refusal_reason_is_counted_under_its_own_name() {
        for &reason in SUBMIT_REFUSAL_NAMES {
            let telemetry = telemetry();
            telemetry.note_submit_refused(reason);
            telemetry.note_submit_refused(reason);

            let (ok, refused) = telemetry.submit_counts();
            assert_eq!((ok, refused), (0, 2), "{reason}");
            assert_eq!(
                telemetry.named_submit_refusal_histogram(),
                BTreeMap::from([(reason, 2)]),
                "{reason} must keep its own bucket"
            );
            let line = telemetry.snapshot().to_string();
            assert!(line.contains("submits=0/2"), "{reason}: {line}");
            assert!(
                line.contains(&format!("named_submit_refusals=[{reason}=2]")),
                "{reason}: {line}"
            );
            assert!(!line.contains('\n'), "one line: {line}");
        }
    }

    /// The histogram is bounded: a name outside the table is counted under the
    /// fixed fallback slot, so the table never grows and the refusal total
    /// always has a name.
    #[test]
    fn an_unregistered_submit_refusal_reason_uses_the_fixed_fallback() {
        const NOT_IN_THE_TABLE: &str = "not_a_submit_refusal_reason";

        let telemetry = telemetry();
        telemetry.note_submit_refused(NOT_IN_THE_TABLE);

        let histogram = telemetry.named_submit_refusal_histogram();
        assert_eq!(
            histogram,
            BTreeMap::from([(submit_refusals::UNKNOWN, 1)]),
            "the unknown name is not a new bucket"
        );
        assert!(
            histogram.keys().all(|reason| submit_refusal_index(reason).is_some()),
            "every key is a table name: {histogram:?}"
        );
        assert_eq!(telemetry.submit_counts().1, 1);
        assert!(
            telemetry
                .snapshot()
                .to_string()
                .contains("named_submit_refusals=[unknown_reason=1]"),
            "{}",
            telemetry.snapshot()
        );
    }

    /// The named refusal histogram is a session total: a window reset keeps it,
    /// so an early refusal's reason is still on the boot's last frame line.
    #[test]
    fn named_submit_refusals_survive_a_window_reset() {
        let telemetry = telemetry();
        telemetry.note_submit_refused(submit_refusals::TRANSPORT);
        telemetry.record_frame_at(0, 10, 0, 0, 0, None);

        let first = telemetry.snapshot_and_reset();
        assert_eq!(
            first.named_submit_refusals,
            BTreeMap::from([(submit_refusals::TRANSPORT, 1)])
        );

        telemetry.reset();
        let after = telemetry.snapshot();
        assert_eq!((after.submits_ok, after.submits_refused), (0, 1));
        assert_eq!(
            after.named_submit_refusals,
            BTreeMap::from([(submit_refusals::TRANSPORT, 1)]),
            "the reason outlives the window it was counted in"
        );
        assert!(after.to_string().contains("named_submit_refusals=[transport=1]"));
    }

    /// The new field is appended: every existing field keeps its position on
    /// the frame line.
    #[test]
    fn the_frame_line_keeps_its_existing_field_order() {
        let line = telemetry().snapshot().to_string();
        let order = [
            "native frame:",
            "fallback=",
            "submits=",
            "presents=",
            "registered=",
            "compute_unbound=",
            "pipeline_attempts=",
            "heaps_bound=",
            "opcodes=",
            "refusals=",
            "named_refusals=",
            "host_errors=",
            "call_failures=",
            "first_call_failure=",
            "named_submit_refusals=",
        ];
        let mut from = 0;
        for field in order {
            let found = line[from..]
                .find(field)
                .unwrap_or_else(|| panic!("{field} missing or out of order in {line}"));
            from += found;
        }
        assert!(line.ends_with("named_submit_refusals=[]"), "{line}");
    }
}
