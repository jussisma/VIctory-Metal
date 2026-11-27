//! Opt-in host-visible backing for XMem physical pages.
//!
//! By default XMem physical pages live in the pagefile, which another process
//! cannot name. A native host process (the Metal renderer) needs a file it can
//! map: set `VICTORY_METAL_SHARE_DIR` to a session-private, Win-visible
//! directory and every physical-page allocation is backed by a file there
//! instead. This holds in every build, the diagnostics guest the launcher runs
//! included. The guest still maps those pages through the existing
//! placeholder views; only the backing changes.
//!
//! The directory is the whole contract: it must belong to this session and
//! be cleaned up by the harness. Files are named `{id:016x}.bin`, where the
//! id is monotonic and never reused. [`Backing::generation`] is fixed at 1
//! for this first protocol, so a stale id can never name a new allocation.
//! The directory is resolved once per process, at the first section creation,
//! so a session that sets it before D3D12 device creation needs no later
//! lookup. Tests (and only tests) re-read it so one can set or clear the
//! variable between sections.
//!
//! Unset (or empty), the provider is off and the pagefile path is unchanged.
//! A configured provider that cannot create or map its file falls back to the
//! pagefile too, but never pretends otherwise: [`resolve_guest_range`] only
//! succeeds for a tracked view of a section with a [`Backing`], and reports
//! [`ResolveError::Pagefile`] otherwise.
//!
//! The resolver itself is installed once, at console device creation (and
//! again, idempotently, from the first shared section or range resolution), so
//! a Metal session that never creates a shared section still has one. The
//! install never depends on the directory: an unconfigured session resolves
//! from the page registry and refuses what it cannot prove.
//!
//! Sparse files matter here: the console allocates multi-GiB heaps, so the
//! file is sized with `set_len` and its pages are never written ahead of the
//! guest. An allocation whose size is not a whole number of 64 KiB pages is
//! rounded up for the file only; its section keeps the real size, so the
//! padding tail is never guest-visible. Only small synthetic sizes are
//! exercised by the tests below.
//!
//! A resolved [`Lease`] may be retained for a submission with
//! [`Lease::retain`]. The registry keeps the section, and with it the share
//! file's handle and mapping, alive until that submission's fence retires
//! ([`retire`]), even if the guest has already freed the page numbers or
//! unmapped every view showing them. Each range is registered by allocation
//! id, generation, byte offset and length; a lookup for a missing or stale
//! range is an explicit error, never an empty or zero-filled range.
//!
//! Retirement is deliberately not idempotent: a duplicate fence callback or
//! a second release is reported ([`RegistryError::Retired`] or
//! [`RegistryError::NotRetained`]), because it means the submission
//! bookkeeping has gone wrong.

// The resolution and registry APIs' consumer is the host-IPC wave; until
// then the tests below are their only callers.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
#[cfg(not(test))]
use std::sync::OnceLock;

use super::{PAGE, Section};

use victory_gpu::backend::GpuBackend;
use victory_gpu::native_metal::bindings::{
    self as bindings, GuestRange, GuestRangeResolver, ResolveError as SeamError,
};

/// Names the session's directory for host-visible XMem backing.
pub(crate) const DIRECTORY: &str = "VICTORY_METAL_SHARE_DIR";

/// The only backing generation understood by this first protocol.
pub(crate) const GENERATION: u64 = 1;

/// Attempts at a fresh file name before giving up (a reused directory can
/// already hold names from an earlier process).
const ATTEMPTS: usize = 8;

/// Default cap on the registry's submission rows.
///
/// Retired rows are evicted, lowest id first, once the map would exceed this,
/// so sustained streaming cannot grow it without limit. Open submissions are
/// live and are never evicted; their count is bounded by the host's
/// frames-in-flight cap, which makes the map O(in-flight) rather than
/// O(allocations over the session).
pub(crate) const SUBMISSIONS_CAPACITY: usize = 4096;

/// Overrides [`SUBMISSIONS_CAPACITY`] for the process with a positive integer.
/// Read once, when the registry is first created.
pub(crate) const SUBMISSIONS_CAPACITY_ENV: &str = "VICTORY_XMEM_RETIRED_CAPACITY";

/// The next share id; ids are monotonic and never reused.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Share backings alive in this process. Every one owns a file in the share
/// directory, so the count makes a leaked section observable in the
/// diagnostics trace.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// The number of share files currently backing a live section.
pub(crate) fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// A host-visible file behind one physical-page section.
///
/// Dropping it closes the file handle but never unlinks the file: a host
/// process may still hold a lease on it, and this first session leaves
/// cleanup to the harness. The directory owner creates it; nothing here
/// deletes it.
pub(crate) struct Backing {
    pub(super) id: u64,
    pub(super) generation: u64,
    path: PathBuf,
    /// Kept open while the section lives, so the mapping has a file handle.
    _file: File,
}

impl Backing {
    /// Takes ownership of a fresh share file and counts it as live until the
    /// last lease and section drop it.
    fn new(id: u64, generation: u64, path: PathBuf, file: File) -> Arc<Self> {
        LIVE.fetch_add(1, Ordering::Relaxed);
        Arc::new(Backing {
            id,
            generation,
            path,
            _file: file,
        })
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Why a configured share provider could not back an allocation.
#[derive(Debug)]
pub(crate) enum Failure {
    /// `VICTORY_METAL_SHARE_DIR` is unset or empty.
    Unconfigured,
    /// Size is zero or rounding it up to whole 64 KiB pages overflows.
    Size(usize),
    /// The directory could not be created.
    Directory { path: PathBuf, error: io::Error },
    /// A file name could not be claimed.
    Create { path: PathBuf, error: io::Error },
    /// The file could not be sized.
    Length { path: PathBuf, error: io::Error },
    /// The file could not be mapped as a section.
    Mapping(io::Error),
    /// No fresh id or file name was available.
    Exhausted,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Unconfigured => write!(f, "{DIRECTORY} is not set"),
            Failure::Size(size) => {
                write!(
                    f,
                    "{size} is zero or cannot be rounded up to whole {PAGE}-byte pages"
                )
            }
            Failure::Directory { path, error } => {
                write!(f, "creating {} failed: {error}", path.display())
            }
            Failure::Create { path, error } => {
                write!(f, "creating {} failed: {error}", path.display())
            }
            Failure::Length { path, error } => {
                write!(f, "sizing {} failed: {error}", path.display())
            }
            Failure::Mapping(error) => write!(f, "mapping the share file failed: {error}"),
            Failure::Exhausted => write!(f, "no fresh share id or file name is available"),
        }
    }
}

/// A guest range resolved to a share file, section offset and length.
///
/// The caller's range is a byte range and need not be page aligned; the
/// share mapping is: `offset` is the page containing the range's start and
/// `[offset, offset + length)` covers every page the range touches. The
/// caller still knows the bytes it asked for, so the exact subrange is
/// `[at - offset, at - offset + len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub(crate) id: u64,
    pub(crate) generation: u64,
    /// Byte offset in the share file of the first covered page; always
    /// 64 KiB aligned.
    pub(crate) offset: u64,
    /// Length in bytes covering the requested range; always a non-zero
    /// multiple of 64 KiB.
    pub(crate) length: u64,
    /// The share file's total length, for guest registration.
    pub(crate) file_size: u64,
}

/// A resolved range that keeps the section, and with it the backing file,
/// alive. It deliberately says nothing about the guest address: the caller
/// must not treat the original pointer as a host-shareable claim.
#[derive(Clone)]
pub(crate) struct Lease {
    pub(crate) range: Range,
    path: PathBuf,
    /// Keeps the section handle and file open until the host lease ends.
    section: Arc<Section>,
}

impl Lease {
    /// The Win-visible share file the range lives in.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The share file's total length, for guest registration.
    pub(crate) fn file_size(&self) -> u64 {
        self.range.file_size
    }

    /// Registers this range and retains it for `submission` until that
    /// submission's fence retires.
    ///
    /// The registry keeps its own [`Arc`] to the section, so the share file
    /// stays open and mapped past every guest page free and unmapped view.
    /// Retaining the same range for one submission again adds a reference;
    /// [`release`] drops one and [`retire`] drops all of them. The same
    /// range may be retained by any number of submissions, and it is
    /// released only when the last reference retires.
    ///
    /// The returned key names the registered range: allocation id,
    /// generation, byte offset and length.
    pub(crate) fn retain(&self, submission: u64) -> Result<RangeKey, RegistryError> {
        registry().retain(self, submission)
    }

    /// Drops one reference `submission` holds on this lease's range.
    ///
    /// See the free [`release`] function.
    pub(crate) fn release(&self, submission: u64) -> Result<(), RegistryError> {
        release(RangeKey::of(&self.range), submission)
    }
}

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease")
            .field("range", &self.range)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// The identity of a retained range: allocation id, generation, byte offset
/// and length in its share file. The file size is not part of the identity;
/// it is fixed by the allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RangeKey {
    pub(crate) id: u64,
    pub(crate) generation: u64,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

impl RangeKey {
    fn of(range: &Range) -> Self {
        RangeKey {
            id: range.id,
            generation: range.generation,
            offset: range.offset,
            length: range.length,
        }
    }
}

/// Why a range could not be retained, released or looked up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegistryError {
    /// The range names no share allocation (id zero).
    ZeroId,
    /// The generation is not [`GENERATION`], or the caller named a backing
    /// the registry no longer retains.
    StaleGeneration { id: u64, generation: u64 },
    /// The window is empty, unaligned, or leaves its share file.
    InvalidRange { offset: u64, length: u64 },
    /// The submission never retained a range.
    UnknownSubmission(u64),
    /// The submission's fence already retired.
    Retired(u64),
    /// The submission does not hold this range.
    NotRetained { submission: u64, key: RangeKey },
    /// No retained range names this allocation.
    Missing { id: u64 },
    /// The allocation is live at that generation, but not this window.
    MissingRange { id: u64, offset: u64, length: u64 },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::ZeroId => f.write_str("the range names no share allocation"),
            RegistryError::StaleGeneration { id, generation } => write!(
                f,
                "allocation {id} is not retained at generation {generation}"
            ),
            RegistryError::InvalidRange { offset, length } => write!(
                f,
                "the range at {offset} for {length} bytes leaves its share file"
            ),
            RegistryError::UnknownSubmission(submission) => {
                write!(f, "submission {submission} never retained a range")
            }
            RegistryError::Retired(submission) => {
                write!(f, "submission {submission} already retired")
            }
            RegistryError::NotRetained { submission, key } => write!(
                f,
                "submission {submission} does not retain allocation {} at {} for {} bytes",
                key.id, key.offset, key.length
            ),
            RegistryError::Missing { id } => {
                write!(f, "no retained range names allocation {id}")
            }
            RegistryError::MissingRange { id, offset, length } => write!(
                f,
                "allocation {id} is not retained at {offset} for {length} bytes"
            ),
        }
    }
}

/// A range the registry keeps alive, and the submissions that reference it.
struct Retained {
    /// A clone of the lease keeps the section, and with it the share file's
    /// handle and mapping, alive past every guest view and page free.
    lease: Lease,
    /// How many references each live submission holds.
    submissions: BTreeMap<u64, usize>,
}

/// A submission's references, or its retirement.
enum Submission {
    Open(BTreeMap<RangeKey, usize>),
    Retired,
}

struct Registry {
    /// Retained ranges, by identity.
    retained: HashMap<RangeKey, Retained>,
    /// Every submission that retained something, open or retired. Retired
    /// rows are evicted once the map would exceed [`Registry::capacity`];
    /// [`Registry::retired_watermark`] remembers the highest evicted id, so a
    /// reused number still cannot resurrect a stale range and a duplicate
    /// retirement is visible.
    submissions: HashMap<u64, Submission>,
    /// Highest retired submission id ever evicted, or `None` before the first
    /// eviction. Monotonic: eviction is lowest-id-first and never lowers it.
    retired_watermark: Option<u64>,
    /// Soft cap on `submissions`; open rows are never evicted, so a pathological
    /// caller with more in-flight submissions than this can exceed it.
    capacity: usize,
}

impl Registry {
    fn new(capacity: usize) -> Self {
        Self {
            retained: HashMap::new(),
            submissions: HashMap::new(),
            retired_watermark: None,
            capacity,
        }
    }

    /// The exact "is this submission retired?" predicate used by every
    /// lookup.
    ///
    /// * A present `Submission::Retired` row is retired.
    /// * A present `Submission::Open` row is live, even when its id is at or
    ///   below the watermark (an older submission can still be in flight).
    /// * An **absent** number at or below [`Self::retired_watermark`] was
    ///   retired and then evicted: eviction only drops `Submission::Retired`
    ///   rows, so "absent and <= watermark" can only mean that. It must be
    ///   refused exactly as if its retired row were still here.
    /// * An absent number above the watermark was never seen: unknown.
    fn is_retired(&self, submission: u64) -> bool {
        match self.submissions.get(&submission) {
            Some(Submission::Retired) => true,
            Some(Submission::Open(_)) => false,
            None => self
                .retired_watermark
                .is_some_and(|watermark| submission <= watermark),
        }
    }

    /// Drops retired rows, lowest id first, until at most `target` rows remain.
    ///
    /// Only `Submission::Retired` rows are evictable. `retire` drops every
    /// reference a submission held before it marks the row retired, so a
    /// retired row can never be named by a retained range or a live pin; the
    /// debug assertion catches a future change that breaks that.
    ///
    /// The watermark advances to the highest evicted id and never moves
    /// backwards, so numbers evicted by an earlier call stay retired forever.
    fn evict_to(&mut self, target: usize) {
        if self.submissions.len() <= target {
            return;
        }
        let mut retired: Vec<u64> = self
            .submissions
            .iter()
            .filter(|(_, state)| matches!(state, Submission::Retired))
            .map(|(submission, _)| *submission)
            .collect();
        retired.sort_unstable();
        let mut excess = self.submissions.len() - target;
        for submission in retired {
            if excess == 0 {
                break;
            }
            debug_assert!(
                !self
                    .retained
                    .values()
                    .any(|entry| entry.submissions.contains_key(&submission)),
                "retired submission {submission} is still referenced by a retained range"
            );
            self.submissions.remove(&submission);
            self.retired_watermark = Some(match self.retired_watermark {
                Some(watermark) => watermark.max(submission),
                None => submission,
            });
            excess -= 1;
        }
    }

    fn retain(&mut self, lease: &Lease, submission: u64) -> Result<RangeKey, RegistryError> {
        let key = RangeKey::of(&lease.range);
        if key.id == 0 {
            return Err(RegistryError::ZeroId);
        }
        if key.generation != GENERATION {
            return Err(RegistryError::StaleGeneration {
                id: key.id,
                generation: key.generation,
            });
        }
        let end = key
            .offset
            .checked_add(key.length)
            .ok_or(RegistryError::InvalidRange {
                offset: key.offset,
                length: key.length,
            })?;
        if key.length == 0
            || !key.offset.is_multiple_of(PAGE as u64)
            || !key.length.is_multiple_of(PAGE as u64)
            || end > lease.range.file_size
        {
            return Err(RegistryError::InvalidRange {
                offset: key.offset,
                length: key.length,
            });
        }
        if self.is_retired(submission) {
            return Err(RegistryError::Retired(submission));
        }
        if matches!(self.submissions.get(&submission), Some(Submission::Open(_))) {
            let Some(Submission::Open(references)) = self.submissions.get_mut(&submission) else {
                unreachable!("an open submission was checked just above");
            };
            *references.entry(key).or_insert(0) += 1;
        } else {
            // Make room before adding a row, so a new submission cannot push
            // the map past capacity while retired rows are evictable.
            self.evict_to(self.capacity.saturating_sub(1));
            let mut references = BTreeMap::new();
            references.insert(key, 1);
            self.submissions
                .insert(submission, Submission::Open(references));
        }
        self.retained
            .entry(key)
            .or_insert_with(|| Retained {
                lease: lease.clone(),
                submissions: BTreeMap::new(),
            })
            .submissions
            .entry(submission)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        Ok(key)
    }

    fn release(&mut self, key: RangeKey, submission: u64) -> Result<(), RegistryError> {
        if self.is_retired(submission) {
            return Err(RegistryError::Retired(submission));
        }
        let Some(Submission::Open(references)) = self.submissions.get_mut(&submission) else {
            return Err(RegistryError::UnknownSubmission(submission));
        };
        let Some(count) = references.get_mut(&key) else {
            return Err(RegistryError::NotRetained { submission, key });
        };
        if *count > 1 {
            *count -= 1;
            if let Some(entry) = self.retained.get_mut(&key)
                && let Some(count) = entry.submissions.get_mut(&submission)
            {
                *count -= 1;
            }
            return Ok(());
        }
        references.remove(&key);
        drop_one(self, &key, submission);
        Ok(())
    }

    fn retire(&mut self, submission: u64) -> Result<usize, RegistryError> {
        if self.is_retired(submission) {
            return Err(RegistryError::Retired(submission));
        }
        let Some(Submission::Open(references)) = self.submissions.remove(&submission) else {
            return Err(RegistryError::UnknownSubmission(submission));
        };
        let mut released = 0;
        for (key, count) in &references {
            released += count;
            drop_one(self, key, submission);
        }
        self.submissions.insert(submission, Submission::Retired);
        // Deterministic eviction point: the fence callback that retires this
        // submission is where a retired row becomes evictable. No timers and
        // no best-effort heuristics: only rows at or below `capacity` survive,
        // lowest id first.
        self.evict_to(self.capacity);
        Ok(released)
    }

    fn retained(
        &self,
        submission: u64,
        id: u64,
        generation: u64,
        offset: u64,
        length: u64,
    ) -> Result<Lease, RegistryError> {
        if self.is_retired(submission) {
            return Err(RegistryError::Retired(submission));
        }
        if !self.submissions.contains_key(&submission) {
            return Err(RegistryError::UnknownSubmission(submission));
        }
        let (mut named, mut live_generation) = (false, false);
        for (key, entry) in &self.retained {
            if key.id != id {
                continue;
            }
            named = true;
            if key.generation != generation {
                continue;
            }
            live_generation = true;
            if key.offset == offset && key.length == length {
                if entry.submissions.contains_key(&submission) {
                    return Ok(entry.lease.clone());
                }
                return Err(RegistryError::NotRetained {
                    submission,
                    key: *key,
                });
            }
        }
        if !named {
            return Err(RegistryError::Missing { id });
        }
        if !live_generation {
            return Err(RegistryError::StaleGeneration { id, generation });
        }
        Err(RegistryError::MissingRange { id, offset, length })
    }

    fn is_retained(&self, key: &RangeKey) -> bool {
        self.retained.contains_key(key)
    }
}

/// The configured capacity: [`SUBMISSIONS_CAPACITY`] unless
/// [`SUBMISSIONS_CAPACITY_ENV`] names a positive integer.
fn configured_capacity() -> usize {
    parse_capacity(std::env::var(SUBMISSIONS_CAPACITY_ENV).ok().as_deref())
}

fn parse_capacity(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(SUBMISSIONS_CAPACITY)
}

static REGISTRY: LazyLock<Mutex<Registry>> =
    LazyLock::new(|| Mutex::new(Registry::new(configured_capacity())));

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    crate::modules::lock(&REGISTRY)
}

/// Drops one reference `submission` holds on `key`.
///
/// A second release, a release after retirement, and a release by a
/// submission that never retained the range are explicit errors, not silent
/// no-ops: they mean the fence bookkeeping is wrong. The range (and its
/// backing) is dropped only when its last reference goes.
pub(crate) fn release(key: RangeKey, submission: u64) -> Result<(), RegistryError> {
    registry().release(key, submission)
}

/// Retires `submission`'s fence: drops every reference it holds and releases
/// the ranges no other submission references any more.
///
/// Returns the number of references released (a submission that retained one
/// range twice releases two). A duplicate retirement is
/// [`RegistryError::Retired`], not a silent success, and a retired
/// submission may not retain again.
///
/// Retirement is also the deterministic eviction point: once this fence is
/// retired, [`Registry::evict_to`] may drop retired rows, lowest id first, so
/// the map stays O(in-flight) over a long session.
pub(crate) fn retire(submission: u64) -> Result<usize, RegistryError> {
    registry().retire(submission)
}

/// The lease of a range that `submission` retains, checked against the
/// caller's `generation` as well.
///
/// This succeeds after the guest frees the page numbers or unmaps every view
/// showing the range: the registry, not the guest page table, keeps the
/// backing alive until retirement. A missing allocation, a stale generation,
/// a window nobody retained, and a reference from another (or a retired or
/// evicted) submission are all explicit errors.
pub(crate) fn retained(
    submission: u64,
    id: u64,
    generation: u64,
    offset: u64,
    length: u64,
) -> Result<Lease, RegistryError> {
    registry().retained(submission, id, generation, offset, length)
}

/// Whether the registry still keeps `key`'s backing alive (tests and the
/// host's audit path).
pub(crate) fn is_retained(key: RangeKey) -> bool {
    registry().is_retained(&key)
}

/// A snapshot of the registry's bounds, for tests: how many submission rows
/// and retained ranges it holds, its capacity, and the eviction watermark.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegistryBounds {
    pub(crate) submissions: usize,
    pub(crate) retained: usize,
    pub(crate) capacity: usize,
    pub(crate) watermark: Option<u64>,
}

/// The registry's current bounds. Test-only: production code must not depend
/// on a particular eviction moment.
#[cfg(test)]
pub(crate) fn bounds() -> RegistryBounds {
    let registry = registry();
    RegistryBounds {
        submissions: registry.submissions.len(),
        retained: registry.retained.len(),
        capacity: registry.capacity,
        watermark: registry.retired_watermark,
    }
}

/// Removes `submission`'s reference to `key`, dropping the retained range
/// (and the backing with it) when no submission references it any more.
fn drop_one(registry: &mut Registry, key: &RangeKey, submission: u64) {
    let empty = match registry.retained.get_mut(key) {
        Some(entry) => {
            entry.submissions.remove(&submission);
            entry.submissions.is_empty()
        }
        None => return,
    };
    if empty {
        registry.retained.remove(key);
    }
}

/// Why an existing guest range could not be resolved to a share file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolveError {
    /// Zero length.
    Empty,
    /// `at + len` or the rounded page boundary does not fit.
    OutOfBounds,
    /// Not every page of the range is shown by a tracked view.
    NotMapped,
    /// A page of the range is reserved by a virtual region but not
    /// committed: its address space exists, its bytes do not.
    Uncommitted,
    /// A page of the range is committed, but its protection forbids reading
    /// (or it is a guard page): the bytes exist, the host may not map them.
    Inaccessible,
    /// The views do not show one contiguous run of one section.
    NotContiguous,
    /// The backing is not this session's: a foreign id or generation.
    Stale,
    /// The section is pagefile-backed (provider unset or unavailable).
    Pagefile,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            ResolveError::Empty => "the range is empty",
            ResolveError::OutOfBounds => "the range leaves its section",
            ResolveError::NotMapped => "the range is not mapped by tracked views",
            ResolveError::Uncommitted => "the range is reserved but not committed",
            ResolveError::Inaccessible => "the range is committed but not readable",
            ResolveError::NotContiguous => "the range is not one contiguous section run",
            ResolveError::Stale => "the range's backing is stale or not this session's",
            ResolveError::Pagefile => "the range's section has no host-visible backing",
        };
        f.write_str(message)
    }
}

// A test's per-thread share directory override: `Some(Some(path))` configures
// the provider and `Some(None)` forces it unconfigured, whatever the process
// environment says. Absent, the tests fall back to the variable itself.
#[cfg(test)]
thread_local! {
    static DIRECTORY_OVERRIDE: std::cell::RefCell<Option<Option<PathBuf>>> =
        const { std::cell::RefCell::new(None) };
}

/// The directory variable as this process currently sees it.
fn read_directory() -> Option<PathBuf> {
    std::env::var_os(DIRECTORY)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The configured directory, if any.
///
/// A non-test build resolves it once, at the first section creation, so every
/// later section sees the same session directory even if something changes
/// the environment (the launcher sets it before the process starts). Tests
/// re-read it so a test can set or clear the variable between sections, and
/// they may install a thread-local override instead.
fn directory() -> Option<PathBuf> {
    #[cfg(test)]
    {
        if let Some(directory) = DIRECTORY_OVERRIDE.with(|slot| slot.borrow().clone()) {
            return directory;
        }
        return read_directory();
    }
    #[cfg(not(test))]
    {
        static RESOLVED: OnceLock<Option<PathBuf>> = OnceLock::new();
        RESOLVED.get_or_init(read_directory).clone()
    }
}

/// Whether a share directory is configured for this thread/process: the
/// virtual region path (`XMemVirtualAlloc`, ADR-043) only takes over from
/// the host allocator when the guest can name host-visible backing.
pub(crate) fn configured() -> bool {
    directory().is_some()
}

/// A fresh, monotonic, non-zero id. `None` only if the counter would wrap.
fn next_id() -> Option<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .ok()
}

/// Backs a new physical-page section of `size` bytes with a file in the
/// configured directory and maps it.
///
/// `size` is the allocation's real size. The file and its section are sized to
/// the next whole 64 KiB page instead, but the returned [`Section`] keeps
/// `size`, so the padding tail is never guest-visible. The caller falls back
/// to the pagefile on any error; the error stays precise so the fallback is
/// never mistaken for a shared section. A file created before a later failure
/// is removed again, so a failed allocation leaves no file for a section that
/// never existed.
pub(crate) fn section(size: usize) -> Result<Arc<Section>, Failure> {
    install_if_metal();
    let Some(directory) = directory() else {
        return Err(Failure::Unconfigured);
    };
    if size == 0 {
        return Err(Failure::Size(size));
    }
    let map_size = size
        .checked_next_multiple_of(PAGE)
        .ok_or(Failure::Size(size))?;
    std::fs::create_dir_all(&directory).map_err(|error| Failure::Directory {
        path: directory.clone(),
        error,
    })?;
    let (id, file, path) = create_file(&directory, map_size)?;
    let backing = Backing::new(id, GENERATION, path, file);
    match super::file_section_sized(&backing._file, size, map_size, Some(backing.clone())) {
        Some(section) => Ok(section),
        None => {
            let error = io::Error::last_os_error();
            let path = backing.path.clone();
            drop(backing);
            // Nothing can have leased a section that was never returned.
            let _ = std::fs::remove_file(&path);
            Err(Failure::Mapping(error))
        }
    }
}

/// Creates and sizes the share file, never writing its contents. A name that
/// already exists is skipped (and therefore never reused) rather than
/// truncated.
fn create_file(directory: &Path, size: usize) -> Result<(u64, File, PathBuf), Failure> {
    for _ in 0..ATTEMPTS {
        let Some(id) = next_id() else {
            return Err(Failure::Exhausted);
        };
        let path = directory.join(format!("{id:016x}.bin"));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => {
                // Sizing is not writing: a multi-GiB heap must stay sparse.
                if let Err(error) = file.set_len(size as u64) {
                    drop(file);
                    // No section exists yet, so the file must not outlive the
                    // failed allocation.
                    let _ = std::fs::remove_file(&path);
                    return Err(Failure::Length { path, error });
                }
                return Ok((id, file, path));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Failure::Create { path, error }),
        }
    }
    Err(Failure::Exhausted)
}

/// The protections that let the host read a committed page.
const READ: u32 = 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80;
/// `PAGE_NOACCESS` and `PAGE_GUARD`: committed, but the host must not read it.
const NOT_READ: u32 = 0x01 | 0x100;

/// Whether a commit's protection lets the host read the page's bytes.
fn readable(protect: u32) -> bool {
    protect & READ != 0 && protect & NOT_READ == 0
}

/// Checks that every page of `[start, stop)` is a committed, readable page of
/// `region`. This is the one rule that keeps a bind from mapping memory the
/// guest only reserved, or committed without read access: the answer is
/// always either a whole committed window or a named refusal.
fn committed_span(
    base: usize,
    region: &super::Reservation,
    start: usize,
    stop: usize,
) -> Result<(), ResolveError> {
    let region_end = base
        .checked_add(region.size)
        .ok_or(ResolveError::OutOfBounds)?;
    if stop > region_end {
        return Err(ResolveError::OutOfBounds);
    }
    let mut cursor = start;
    while cursor < stop {
        let offset = cursor.checked_sub(base).ok_or(ResolveError::OutOfBounds)?;
        let Some((run_start, commit)) = region.committed_at(offset) else {
            return Err(ResolveError::Uncommitted);
        };
        if !readable(commit.protect) {
            return Err(ResolveError::Inaccessible);
        }
        let run_end = base
            .checked_add(run_start)
            .and_then(|at| at.checked_add(commit.len))
            .ok_or(ResolveError::OutOfBounds)?;
        // `committed_at` proved this run reaches past `cursor`: the loop
        // always advances, so a committed run can never stall a bind.
        cursor = run_end.min(stop);
    }
    Ok(())
}

/// Resolves the guest byte range `[at, at + len)` to a share file, offset and
/// length.
///
/// `at` and `len` need not be 64 KiB aligned: the result starts at the page
/// containing `at` and covers every page the range touches, so the host can
/// map it. The whole rounded span must be shown by tracked physical-page
/// views or be committed pages of one virtual region (ADR-043), and be one
/// contiguous run of pages of one section, all within it. In particular, a
/// range whose last byte is mapped is still refused if rounding it would
/// reach an unmapped or foreign page: never grant a page the guest did not
/// map.
///
/// This takes the physical-page lock only to inspect and clone what the
/// views already hold; the returned lease keeps the section (and its file)
/// alive with an `Arc`.
pub(crate) fn resolve_guest_range(at: usize, len: usize) -> Result<Lease, ResolveError> {
    install_if_metal();
    if len == 0 {
        return Err(ResolveError::Empty);
    }
    let end = at.checked_add(len).ok_or(ResolveError::OutOfBounds)?;
    // The page containing the first byte and the page boundary after the
    // last: a byte range may share pages with bytes the caller does not own,
    // so rounding must not wrap either.
    let start = at & !(PAGE - 1);
    let stop = end.checked_add(PAGE - 1).ok_or(ResolveError::OutOfBounds)? & !(PAGE - 1);
    let physical = super::physical();
    // A physical-page view's section and page run, or the one virtual region
    // the whole range is committed in; a range spanning both is refused.
    let mut section: Option<Arc<Section>> = None;
    let mut first_index = None;
    let mut next_index = None;
    let mut direct: Option<(usize, usize, Arc<Section>)> = None;
    let mut cursor = start;
    while cursor < stop {
        if let Some((base, size, slice)) = view_at(&physical, cursor) {
            if direct.is_some() {
                return Err(ResolveError::NotContiguous);
            }
            let index = slice
                .index
                .checked_add((cursor - base) / PAGE)
                .ok_or(ResolveError::OutOfBounds)?;
            match &section {
                None => {
                    section = Some(slice.section.clone());
                    first_index = Some(index);
                }
                Some(section) if !Arc::ptr_eq(section, &slice.section) => {
                    return Err(ResolveError::NotContiguous);
                }
                Some(_) => {}
            }
            if next_index.is_some_and(|next| next != index) {
                return Err(ResolveError::NotContiguous);
            }
            next_index = index.checked_add(1);
            let view_end = base.checked_add(size).ok_or(ResolveError::OutOfBounds)?;
            cursor = view_end.min(stop);
            continue;
        }
        // A virtual region: only its committed pages have bytes, and the
        // host can map them from the region's one section at their offset.
        let Some((base, region)) = physical.reservation_at(cursor) else {
            return Err(ResolveError::NotMapped);
        };
        if section.is_some() {
            return Err(ResolveError::NotContiguous);
        }
        let region_end = base
            .checked_add(region.size)
            .ok_or(ResolveError::OutOfBounds)?;
        if stop > region_end {
            return Err(ResolveError::OutOfBounds);
        }
        // A reservation with no backing was never committed: its address
        // space exists but no page is readable. A 64 KiB physical-page
        // reservation keeps the older, coarser answer for its unmapped
        // pages.
        let Some(reservation_section) = region.section.clone() else {
            return Err(if region.physical_pages {
                ResolveError::NotMapped
            } else {
                ResolveError::Uncommitted
            });
        };
        match &direct {
            None => direct = Some((base, region_end, reservation_section)),
            Some((first_base, _, first_section)) => {
                if *first_base != base || !Arc::ptr_eq(first_section, &reservation_section) {
                    return Err(ResolveError::NotContiguous);
                }
            }
        }
        let offset = cursor.checked_sub(base).ok_or(ResolveError::OutOfBounds)?;
        let Some((run_start, commit)) = region.committed_at(offset) else {
            return Err(ResolveError::Uncommitted);
        };
        let run_end = base
            .checked_add(run_start)
            .and_then(|start| start.checked_add(commit.len))
            .ok_or(ResolveError::OutOfBounds)?;
        cursor = run_end.min(stop);
    }
    if let Some((base, region_end, section)) = direct {
        let offset = start.checked_sub(base).ok_or(ResolveError::OutOfBounds)?;
        let length = stop.checked_sub(start).ok_or(ResolveError::OutOfBounds)?;
        if stop > region_end {
            return Err(ResolveError::OutOfBounds);
        }
        // Every page of the rounded span, not merely the ones the window walk
        // happened to land on, must be committed and readable in this one
        // region. A reservation with any uncommitted or inaccessible page is
        // a named refusal, never a bindable range.
        let Some((start_base, region)) = physical.reservation_at(start) else {
            return Err(ResolveError::NotMapped);
        };
        if start_base != base {
            return Err(ResolveError::NotContiguous);
        }
        committed_span(base, region, start, stop)?;
        let backing = section.share.as_ref().ok_or(ResolveError::Pagefile)?;
        if backing.generation != GENERATION || backing.id == 0 {
            return Err(ResolveError::Stale);
        }
        return Ok(Lease {
            range: Range {
                id: backing.id,
                generation: backing.generation,
                offset: offset as u64,
                length: length as u64,
                file_size: section.map_size as u64,
            },
            path: backing.path.clone(),
            section,
        });
    }
    let section = section.ok_or(ResolveError::NotMapped)?;
    let offset = first_index
        .ok_or(ResolveError::NotMapped)?
        .checked_mul(PAGE)
        .ok_or(ResolveError::OutOfBounds)?;
    let length = stop.checked_sub(start).ok_or(ResolveError::OutOfBounds)?;
    // Even though every page is shown by a view, check the whole rounded
    // span lies in the section: no view may grant a page outside it.
    let end_offset = offset
        .checked_add(length)
        .ok_or(ResolveError::OutOfBounds)?;
    if end_offset > section.size {
        return Err(ResolveError::OutOfBounds);
    }
    let backing = section.share.as_ref().ok_or(ResolveError::Pagefile)?;
    // Ids are never reused and this first protocol understands one
    // generation, so any other id or generation names a backing this process
    // did not create now. Refuse it rather than hand a host a file it must
    // not map.
    if backing.generation != GENERATION || backing.id == 0 {
        return Err(ResolveError::Stale);
    }
    Ok(Lease {
        range: Range {
            id: backing.id,
            generation: backing.generation,
            offset: offset as u64,
            length: length as u64,
            file_size: section.map_size as u64,
        },
        path: backing.path.clone(),
        section,
    })
}

/// The tracked view holding `at`, as `(base, length, first slice)`.
fn view_at(
    physical: &super::Physical,
    at: usize,
) -> Option<(usize, usize, &super::Slice)> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < base.wrapping_add(*size))
        .map(|(base, (size, slice))| (*base, *size, slice))
}

// ---------------------------------------------------------------- the seam

/// The lease behind a range this resolver returned, kept until the retain
/// that pins it (a [`Lease`] keeps the section, and with it the share file's
/// handle and mapping, alive on its own).
///
/// Keyed by the range exactly as [`ShareResolver::resolve`] returned it: the
/// wire binding needs the page window, the registry needs its identity, and
/// the caller passes back what it was given.
static PENDING: LazyLock<Mutex<HashMap<GuestRange, Vec<Lease>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolved-but-not-yet-retained ranges are transient; this bounds a caller
/// that resolves and never pins (a rejected descriptor, a refused table).
const PENDING_LIMIT: usize = 4096;

/// Pins taken through the seam; a token is a registry submission id.
///
/// High enough that it cannot collide with the small submission numbers the
/// queue uses for its own fences, and monotonic for the whole process.
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1 << 32);

fn next_token() -> Result<u64, SeamError> {
    NEXT_TOKEN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |token| {
            token.checked_add(1)
        })
        .map_err(|_| SeamError::Retain("the pin token counter is exhausted"))
}

fn remember(range: GuestRange, lease: Lease) {
    let mut pending = crate::modules::lock(&PENDING);
    if pending.len() >= PENDING_LIMIT && !pending.contains_key(&range) {
        // A caller that resolved without retaining cannot pin anything with
        // these; dropping the leases only closes already-unused sections.
        pending.clear();
    }
    pending.entry(range).or_default().push(lease);
}

/// The share provider as the native backend's [`GuestRangeResolver`].
///
/// A range that is not shown by tracked views, leaves its section, spans two
/// sections, or lives in a pagefile-backed section is an explicit error. The
/// pagefile path itself is untouched: nothing here runs unless the native
/// backend asks, and the provider stays off when
/// `VICTORY_METAL_SHARE_DIR` is unset.
pub(crate) struct ShareResolver;

impl GuestRangeResolver for ShareResolver {
    fn resolve(&self, guest_address: u64, len: u64) -> Result<GuestRange, SeamError> {
        if guest_address == 0 {
            return Err(SeamError::Invalid("the guest address is zero"));
        }
        let at = usize::try_from(guest_address).map_err(|_| SeamError::Unmapped {
            address: guest_address,
            len,
        })?;
        let size = usize::try_from(len).map_err(|_| SeamError::Unmapped {
            address: guest_address,
            len,
        })?;
        let lease = resolve_guest_range(at, size)
            .map_err(|error| seam_resolve_error(error, guest_address, len))?;
        let generation = u32::try_from(lease.range.generation).map_err(|_| {
            SeamError::Invalid("the allocation generation does not fit the wire field")
        })?;
        let range = GuestRange {
            allocation_id: lease.range.id,
            generation,
            byte_offset: lease.range.offset,
            size: lease.range.length,
        };
        remember(range, lease);
        Ok(range)
    }

    fn retain(&self, range: &GuestRange) -> Result<u64, SeamError> {
        let lease = {
            let mut pending = crate::modules::lock(&PENDING);
            let leases = pending.get_mut(range).ok_or(SeamError::Retain(
                "the range was not resolved by this process, or was already pinned",
            ))?;
            let lease = leases.pop().ok_or(SeamError::Retain(
                "the range was not resolved by this process, or was already pinned",
            ))?;
            if leases.is_empty() {
                pending.remove(range);
            }
            lease
        };
        let token = next_token()?;
        lease.retain(token).map_err(seam_retain_error)?;
        Ok(token)
    }

    fn retire(&self, token: u64) -> Result<(), SeamError> {
        retire(token).map(|_| ()).map_err(seam_retire_error)
    }
}

fn seam_resolve_error(error: ResolveError, address: u64, len: u64) -> SeamError {
    match error {
        ResolveError::Empty => SeamError::Invalid("the guest range is empty"),
        ResolveError::OutOfBounds | ResolveError::NotMapped => SeamError::Unmapped { address, len },
        ResolveError::Uncommitted => {
            SeamError::Invalid("the range is reserved but not committed")
        }
        ResolveError::Inaccessible => {
            SeamError::Invalid("the range is committed but not readable")
        }
        ResolveError::NotContiguous => {
            SeamError::Invalid("the range is not one contiguous shared allocation")
        }
        ResolveError::Stale => {
            SeamError::Invalid("the range's backing is stale or not this session's")
        }
        ResolveError::Pagefile => {
            SeamError::Unsupported("the range's section has no host-visible backing")
        }
    }
}

fn seam_retain_error(error: RegistryError) -> SeamError {
    match error {
        RegistryError::StaleGeneration { .. } => {
            SeamError::Invalid("the range's backing generation is stale")
        }
        RegistryError::ZeroId => SeamError::Invalid("the range names no share allocation"),
        RegistryError::InvalidRange { .. } => {
            SeamError::Invalid("the range is unaligned or leaves its share file")
        }
        RegistryError::Retired(_) => SeamError::Retain("the pin's submission already retired"),
        _ => SeamError::Retain("the share registry refused the range"),
    }
}

fn seam_retire_error(error: RegistryError) -> SeamError {
    match error {
        RegistryError::Retired(_) => SeamError::Retire("the pin was already retired"),
        RegistryError::UnknownSubmission(_) => SeamError::Retire("the pin was never retained"),
        _ => SeamError::Retire("the share registry refused the retirement"),
    }
}

/// Whether [`ShareResolver`] has been installed. Every section allocation and
/// every bind-time resolution checks this, so it stays an atomic fast path.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Serializes the one installation. The caller re-checks [`INSTALLED`] under
/// this lock, so two racing installs cannot both name the resolver.
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

/// Installs [`ShareResolver`] as the process-wide guest range resolver when
/// `VICTORY_BACKEND=metal`.
///
/// Idempotent: the console device creation installs it before the device can
/// create a resource, and the first shared section or range resolution may
/// call this again. With the default backend nothing is installed and the
/// pagefile path is exactly as before.
pub(crate) fn install_if_metal() {
    install_for_backend(GpuBackend::from_env());
}

/// Installs the resolver for `backend`, at most once.
///
/// Only the native Metal backend resolves guest ranges through this seam; the
/// D3D12 reference backend must never install it. Whether the share directory
/// is configured does not matter: the resolver reads the page registry and
/// refuses a range it cannot prove, rather than fabricating an allocation id.
fn install_for_backend(backend: GpuBackend) {
    if backend != GpuBackend::Metal || INSTALLED.load(Ordering::Acquire) {
        return;
    }
    let _guard = crate::modules::lock(&INSTALL_LOCK);
    if INSTALLED.load(Ordering::Acquire) {
        return;
    }
    bindings::set_resolver(ShareResolver);
    INSTALLED.store(true, Ordering::Release);
}

/// Drops the one install and its resolver, so an install test can start from
/// a known state. Tests run with `--test-threads=1`.
#[cfg(test)]
fn uninstall_for_test() {
    let _guard = crate::modules::lock(&INSTALL_LOCK);
    INSTALLED.store(false, Ordering::Release);
    let _ = bindings::clear_resolver();
}

#[cfg(test)]
mod tests {
    use super::*;

    use windows_sys::Win32::Foundation::{ERROR_INVALID_ADDRESS, GetLastError, SetLastError};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEM_RESET, PAGE_NOACCESS,
        PAGE_READWRITE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use crate::modules::xmem::{
        BasicInformation, PAGES_64K, VirtualFree, XMemAllocatePhysicalPages, XMemFreePhysicalPages,
        XMemMapPhysicalPages, XMemVirtualAlloc, XMemVirtualQuery, last_refusal, virtual_counters,
    };

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// A 64 KiB-page placeholder the title would map physical pages into.
    fn reserve(len: usize) -> *mut std::ffi::c_void {
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            len,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null());
        at
    }

    /// Sets this thread's share directory for section-creating calls, restoring
    /// the previous override on drop. `unconfigured` forces the provider off
    /// even when the harness set the variable.
    struct ShareDirectory(Option<Option<PathBuf>>);

    impl ShareDirectory {
        fn configured(path: &Path) -> Self {
            Self::install(Some(Some(path.to_path_buf())))
        }

        fn unconfigured() -> Self {
            Self::install(Some(None))
        }

        fn install(directory: Option<Option<PathBuf>>) -> Self {
            let previous = DIRECTORY_OVERRIDE.with(|slot| slot.replace(directory));
            ShareDirectory(previous)
        }
    }

    impl Drop for ShareDirectory {
        fn drop(&mut self) {
            DIRECTORY_OVERRIDE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }

    /// Every build, the diagnostics guest included, takes the share path when
    /// the directory is configured: the section owns a backing and a tracked
    /// view resolves to it.
    #[test]
    fn configured_directory_backs_sections_in_every_build() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let before = live();
        let pages = allocate(2);
        assert_eq!(live(), before + 1, "the new section counts as live");
        let at = reserve(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);

        {
            let physical = super::super::physical();
            let section = &physical
                .pages
                .get(&pages[0])
                .expect("a tracked page")
                .section;
            assert!(
                section.share.is_some(),
                "the section must have a share backing"
            );
            assert_eq!(section.size, 2 * PAGE);
            assert_eq!(section.map_size, 2 * PAGE);
        }

        let lease = resolve_guest_range(at as usize, PAGE).expect("a tracked view resolves");
        assert_eq!(lease.range.offset, 0);
        assert_eq!(lease.range.length, PAGE as u64);
        assert_eq!(lease.file_size(), (2 * PAGE) as u64);
        assert_eq!(lease.path().parent(), Some(directory.path()));
        drop(lease);

        assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        assert_eq!(live(), before, "freeing the pages releases the share backing");
    }

    /// A non-page-multiple allocation must still take the provider: the file
    /// is rounded up to whole pages, while `Section::size` keeps the guest's
    /// real size so the padding tail never resolves.
    #[test]
    fn non_page_multiple_sizes_round_up_instead_of_falling_back() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let size = PAGE + 0x321;
        let section = super::super::new_section(size).expect("a share-backed section");
        assert!(
            section.share.is_some(),
            "a rounded size must not silently downgrade to the pagefile"
        );
        assert_eq!(section.size, size, "the guest size is unchanged");
        assert_eq!(section.map_size, 2 * PAGE, "the backing is whole pages");
        let path = section.share.as_ref().expect("a backing").path.clone();
        assert_eq!(
            std::fs::metadata(&path).expect("the share file").len(),
            (2 * PAGE) as u64
        );

        let view = FakeView::install(
            FAKE,
            2 * PAGE,
            super::super::Slice {
                section: section.clone(),
                index: 0,
            },
        );
        let lease = resolve_guest_range(FAKE, PAGE).expect("the real page resolves");
        assert_eq!((lease.range.offset, lease.range.length), (0, PAGE as u64));
        assert_eq!(lease.file_size(), (2 * PAGE) as u64);
        assert_eq!(
            resolve_guest_range(FAKE + PAGE, PAGE).unwrap_err(),
            ResolveError::OutOfBounds,
            "the padded page must not be guest-visible"
        );
        drop((lease, view));
    }

    /// With no directory, sections stay pagefile-backed and the resolver says
    /// so explicitly.
    #[test]
    fn an_unconfigured_directory_leaves_sections_on_the_pagefile() {
        let _share = ShareDirectory::unconfigured();
        let pages = allocate(1);
        let at = reserve(PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 1, pages.as_ptr()), at);
        {
            let physical = super::super::physical();
            let section = &physical
                .pages
                .get(&pages[0])
                .expect("a tracked page")
                .section;
            assert!(section.share.is_none(), "no provider, no share backing");
        }
        assert_eq!(
            resolve_guest_range(at as usize, PAGE).unwrap_err(),
            ResolveError::Pagefile
        );
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    /// A provider that fails leaves an explicit pagefile section, never a
    /// fabricated share range.
    #[test]
    fn a_failed_provider_is_a_pagefile_error_not_a_fabricated_range() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let not_a_directory = directory.path().join("regular-file");
        std::fs::write(&not_a_directory, b"x").expect("a regular file");
        let _share = ShareDirectory::configured(&not_a_directory);

        let pages = allocate(1);
        let at = reserve(PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 1, pages.as_ptr()), at);
        {
            let physical = super::super::physical();
            let section = &physical
                .pages
                .get(&pages[0])
                .expect("a tracked page")
                .section;
            assert!(section.share.is_none(), "a failed provider has no backing");
        }
        assert_eq!(
            resolve_guest_range(at as usize, PAGE).unwrap_err(),
            ResolveError::Pagefile
        );
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    /// A provider-backed section keeps a lease alive through page-free and
    /// unmap, and retirement still drops it.
    #[test]
    fn provider_backed_sections_keep_leases_until_retirement() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let pages = allocate(1);
        let at = reserve(PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 1, pages.as_ptr()), at);

        let lease = resolve_guest_range(at as usize, PAGE).expect("the view resolves");
        let id = lease.range.id;
        let submission = fresh_submission();
        let key = lease.retain(submission).expect("the lease retains");
        drop(lease);

        // No page numbers and no view remain: only the registry's reference
        // keeps the section and its share file alive.
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        assert!(is_retained(key), "the retained range keeps the backing");
        assert!(retained(submission, id, GENERATION, 0, PAGE as u64).is_ok());
        assert_eq!(retire(submission).expect("retire the submission"), 1);
        assert!(!is_retained(key), "retirement drops the retained range");
    }

    /// The production wiring: a configured share directory backs physical
    /// pages with a named file, and a tracked guest range resolves to that
    /// file's id, offset and length. No GPU work.
    #[test]
    fn configured_directory_backs_pages_and_resolves_guest_ranges() {
        let Ok(directory) = std::env::var(DIRECTORY) else {
            eprintln!("skipped: {DIRECTORY} is not set");
            return;
        };
        if directory.is_empty() {
            eprintln!("skipped: {DIRECTORY} is empty");
            return;
        }
        let directory = PathBuf::from(directory);
        let pages = allocate(2);
        let at = reserve(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);

        // Page 0 at section offset 0, in the named file, still untouched:
        // sizing a sparse file must not write it.
        let first = resolve_guest_range(at as usize, PAGE).expect("page 0 resolves");
        assert_ne!(first.range.id, 0, "share ids are non-zero");
        assert_eq!(first.range.generation, GENERATION);
        assert_eq!(first.range.offset, 0);
        assert_eq!(first.range.length, PAGE as u64);
        assert_eq!(first.file_size(), (2 * PAGE) as u64);
        let name = format!("{:016x}.bin", first.range.id);
        assert_eq!(first.path().parent(), Some(directory.as_path()));
        assert_eq!(
            first.path().file_name().and_then(|name| name.to_str()),
            Some(name.as_str())
        );
        let first_path = first.path().to_path_buf();
        let first_id = first.range.id;
        assert_eq!(
            std::fs::metadata(first.path())
                .expect("the share file")
                .len(),
            (2 * PAGE) as u64
        );
        let bytes = std::fs::read(first.path()).expect("the share file's bytes");
        assert_eq!(bytes.len(), 2 * PAGE);
        assert!(
            bytes.iter().all(|byte| *byte == 0),
            "the file was pre-written"
        );

        // A guest write through the existing view reaches the file, which is
        // what a native host process would map.
        let payload: Vec<u8> = (0..256).map(|i| (i * 13 + 7) as u8).collect();
        // SAFETY: the placeholder view is mapped read-write for two pages.
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), at.cast::<u8>(), payload.len());
        }
        let bytes = std::fs::read(first.path()).expect("the written share file");
        assert_eq!(&bytes[..payload.len()], &payload[..]);

        // Page 1 has a nonzero section offset but the same file id.
        let second = resolve_guest_range(at as usize + PAGE, PAGE).expect("page 1 resolves");
        assert_eq!(second.range.id, first.range.id);
        assert_eq!(second.range.offset, PAGE as u64);
        assert_eq!(second.range.length, PAGE as u64);
        // Adjacent tracked pages resolve as one contiguous section range.
        let both = resolve_guest_range(at as usize, 2 * PAGE).expect("both pages resolve");
        assert_eq!(
            (both.range.id, both.range.offset, both.range.length),
            (first.range.id, 0, (2 * PAGE) as u64)
        );
        assert_eq!(both.path(), first_path.as_path());
        let second_path = second.path().to_path_buf();
        drop((first, second, both));

        // Precise failures, never a fabricated id.
        assert_eq!(
            resolve_guest_range(at as usize, 0).unwrap_err(),
            ResolveError::Empty
        );
        // Byte ranges are accepted: this one stays inside page 0 and this
        // one reaches the start of page 1, which is mapped.
        let inside = resolve_guest_range(at as usize + 1, PAGE - 2).expect("a byte subrange");
        assert_eq!((inside.range.offset, inside.range.length), (0, PAGE as u64));
        let reached = resolve_guest_range(at as usize, PAGE + 1).expect("a range ending in page 1");
        assert_eq!(
            (reached.range.offset, reached.range.length),
            (0, (2 * PAGE) as u64)
        );
        assert_eq!(
            resolve_guest_range(at as usize + 2 * PAGE, PAGE).unwrap_err(),
            ResolveError::NotMapped
        );
        assert_eq!(second_path, first_path);
        assert!(first_path.exists(), "the registered file is still there");

        // A later allocation gets a larger id; the earlier file is never
        // reused and remains for harness cleanup.
        assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        let later = allocate(1);
        let elsewhere = reserve(PAGE);
        assert_eq!(
            XMemMapPhysicalPages(elsewhere, 1, later.as_ptr()),
            elsewhere
        );
        let next = resolve_guest_range(elsewhere as usize, PAGE).expect("a new backing");
        assert!(
            next.range.id > first_id,
            "{} must be after {first_id}",
            next.range.id
        );
        assert_eq!(next.range.generation, GENERATION);
        assert_ne!(next.path(), first_path.as_path());
        let next_path = next.path().to_path_buf();
        drop(next);
        assert_eq!(
            XMemMapPhysicalPages(elsewhere, 1, std::ptr::null()),
            elsewhere
        );
        assert_eq!(XMemFreePhysicalPages(1, later.as_ptr()), 1);

        // Dropping the leases and sections leaves every registered file.
        assert!(first_path.exists(), "a backing file must not be unlinked");
        assert!(next_path.exists(), "a backing file must not be unlinked");
    }

    /// A compute view hands the resolver a byte range, not pages. A range
    /// may cross adjacent tracked views of one section, but rounding it to
    /// pages must never grant an unmapped page, a page gap or another
    /// section.
    #[test]
    fn byte_ranges_cross_adjacent_views_and_refuse_rounding_grants() {
        let Ok(directory) = std::env::var(DIRECTORY) else {
            eprintln!("skipped: {DIRECTORY} is not set");
            return;
        };
        if directory.is_empty() {
            eprintln!("skipped: {DIRECTORY} is empty");
            return;
        }
        let directory = PathBuf::from(directory);

        // Four pages of one section, shown as two adjacent two-page views.
        let pages = allocate(4);
        let at = reserve(4 * PAGE);
        let second_view = at
            .cast::<u8>()
            .wrapping_add(2 * PAGE)
            .cast::<std::ffi::c_void>();
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);
        assert_eq!(
            XMemMapPhysicalPages(second_view, 2, pages[2..].as_ptr()),
            second_view
        );

        // A byte range starting in the first view and ending in the second:
        // both pages the rounded span touches are mapped, adjacent and of
        // one section.
        let crossing = resolve_guest_range(at as usize + 2 * PAGE - 0x80, 0x200)
            .expect("a range across adjacent views");
        assert_eq!(
            (crossing.range.offset, crossing.range.length),
            (PAGE as u64, (2 * PAGE) as u64)
        );
        assert_eq!(crossing.file_size(), (4 * PAGE) as u64);
        assert_eq!(crossing.path().parent(), Some(directory.as_path()));

        // Unaligned one-page subranges: the requested bytes fit in one
        // mapped page, so the mapping is that whole page.
        let subrange = resolve_guest_range(at as usize + 0x123, 0x456)
            .expect("an unaligned one-page subrange");
        assert_eq!(
            (subrange.range.offset, subrange.range.length),
            (0, PAGE as u64)
        );
        let tail = resolve_guest_range(at as usize + 3 * PAGE + 0x10, 0x20)
            .expect("an unaligned subrange of the last page");
        assert_eq!(
            (tail.range.offset, tail.range.length),
            ((3 * PAGE) as u64, PAGE as u64)
        );

        // Page index 0 then 2 is a gap in the section, even though both
        // pages are mapped at adjacent addresses.
        let gap = reserve(2 * PAGE);
        let gap_next = gap
            .cast::<u8>()
            .wrapping_add(PAGE)
            .cast::<std::ffi::c_void>();
        assert_eq!(XMemMapPhysicalPages(gap, 1, pages.as_ptr()), gap);
        assert_eq!(
            XMemMapPhysicalPages(gap_next, 1, pages[2..].as_ptr()),
            gap_next
        );
        assert_eq!(
            resolve_guest_range(gap as usize, 2 * PAGE).unwrap_err(),
            ResolveError::NotContiguous,
            "a page gap must not resolve"
        );

        // Adjacent pages of two sections are not one run either.
        let other = allocate(1);
        let mixed = reserve(2 * PAGE);
        let mixed_next = mixed
            .cast::<u8>()
            .wrapping_add(PAGE)
            .cast::<std::ffi::c_void>();
        assert_eq!(XMemMapPhysicalPages(mixed, 1, pages.as_ptr()), mixed);
        assert_eq!(
            XMemMapPhysicalPages(mixed_next, 1, other.as_ptr()),
            mixed_next
        );
        assert_eq!(
            resolve_guest_range(mixed as usize, 2 * PAGE).unwrap_err(),
            ResolveError::NotContiguous,
            "pages of two sections must not resolve"
        );

        // Arithmetic and rounding are checked, never wrapped.
        assert_eq!(
            resolve_guest_range(usize::MAX - 0x10, 0x20).unwrap_err(),
            ResolveError::OutOfBounds
        );
        assert_eq!(
            resolve_guest_range(usize::MAX, 1).unwrap_err(),
            ResolveError::OutOfBounds,
            "rounding the last byte up would overflow"
        );
        // The byte past the last mapped page is not granted by rounding. An
        // adjacent view of another section may make the refusal
        // `NotContiguous` rather than `NotMapped`; either way it must refuse.
        assert!(
            resolve_guest_range(at as usize + 4 * PAGE - 1, 2).is_err(),
            "the byte past the last mapped page must not be granted by rounding"
        );

        drop((crossing, subrange, tail));
        assert_eq!(XMemMapPhysicalPages(at, 4, std::ptr::null()), at);
        assert_eq!(XMemMapPhysicalPages(gap, 2, std::ptr::null()), gap);
        assert_eq!(XMemMapPhysicalPages(mixed, 2, std::ptr::null()), mixed);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, other.as_ptr()), 1);
    }

    /// A section without host-visible backing (the pagefile fallback, or a
    /// test file) must never resolve as if it had an id.
    #[test]
    fn pagefile_backed_sections_are_reported_not_shared() {
        let file = tempfile::tempfile().expect("a temporary file");
        file.set_len(PAGE as u64).expect("one page");
        crate::modules::xmem::tests::SECTION_FILE
            .with(|slot| *slot.borrow_mut() = Some(file.try_clone().expect("a copy")));
        let pages = allocate(1);
        crate::modules::xmem::tests::SECTION_FILE.with(|slot| *slot.borrow_mut() = None);
        let at = reserve(PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 1, pages.as_ptr()), at);
        assert_eq!(
            resolve_guest_range(at as usize, PAGE).unwrap_err(),
            ResolveError::Pagefile
        );
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    // The registry tests below need a section with real host-visible
    // backing, but not the session's shared directory: they build one over
    // their own temporary file, so they run without a game or a Wine
    // launcher.

    /// A submission number no other test has used.
    ///
    /// The registry deliberately remembers a retirement for the life of the
    /// process (a reused number must not resurrect a stale range), and the
    /// test harness runs these tests in one process in *name* order, which is
    /// not source order. A number shared between two tests is therefore
    /// retired by the first and refused by the second, wherever they run:
    /// the sharing was the defect, not the registry's permanent retirement.
    fn fresh_submission() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// An address no real view occupies; the resolver reads only the page
    /// table, so a range here is never dereferenced.
    const FAKE: usize = 1 << 47;

    /// A share file of `pages` pages and the section over it, with the
    /// backing identity the test wants. The file must outlive the section.
    fn share_section(
        directory: &tempfile::TempDir,
        name: &str,
        pages: usize,
        id: u64,
        generation: u64,
    ) -> (Arc<Section>, PathBuf) {
        let path = directory.path().join(name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("a share file");
        let size = pages * PAGE;
        file.set_len(size as u64).expect("the share file's length");
        let backing = Backing::new(
            id,
            generation,
            path.clone(),
            file.try_clone().expect("a copy"),
        );
        let section =
            super::super::file_section(&file, size, Some(backing)).expect("a share section");
        (section, path)
    }

    /// A share file, its section and the directory owning both.
    struct SyntheticShare {
        section: Arc<Section>,
        path: PathBuf,
        size: usize,
        _directory: tempfile::TempDir,
    }

    impl SyntheticShare {
        fn new(pages: usize) -> Self {
            let directory = tempfile::tempdir().expect("a temporary directory");
            let id = next_id().expect("a share id");
            let (section, path) = share_section(&directory, "share.bin", pages, id, GENERATION);
            SyntheticShare {
                section,
                path,
                size: pages * PAGE,
                _directory: directory,
            }
        }

        /// A lease on `[offset, offset + length)` of the share file, as
        /// [`resolve_guest_range`] would return it.
        fn lease(&self, offset: usize, length: usize) -> Lease {
            let backing = self.section.share.as_ref().expect("a share backing");
            Lease {
                range: Range {
                    id: backing.id,
                    generation: backing.generation,
                    offset: offset as u64,
                    length: length as u64,
                    file_size: self.size as u64,
                },
                path: self.path.clone(),
                section: self.section.clone(),
            }
        }

        /// Moves the section out, so the caller owns every reference and can
        /// prove the registry keeps the backing alive on its own.
        fn into_section(self) -> (Arc<Section>, PathBuf, tempfile::TempDir, usize) {
            (self.section, self.path, self._directory, self.size)
        }
    }

    /// A page-table view entry installed directly and removed when dropped.
    /// The resolver reads only this bookkeeping, so tests can forge bounds a
    /// title could never map.
    struct FakeView(usize);

    impl FakeView {
        fn install(at: usize, size: usize, slice: super::super::Slice) -> Self {
            super::super::physical().views.insert(at, (size, slice));
            FakeView(at)
        }
    }

    impl Drop for FakeView {
        fn drop(&mut self) {
            super::super::physical().views.remove(&self.0);
        }
    }

    /// A range stays retained, and its backing mapped, after the guest frees
    /// the page numbers and unmaps the view. Missing windows and allocations
    /// are explicit errors, never a zero-filled range.
    #[test]
    fn retained_ranges_outlive_freed_pages_and_unmapped_views() {
        let (section, path, _directory, size) = SyntheticShare::new(2).into_section();
        let id = section.share.as_ref().expect("a share backing").id;
        let pages = allocate(2);
        {
            let mut physical = super::super::physical();
            for (index, page) in pages.iter().enumerate() {
                physical.pages.insert(
                    *page,
                    super::super::Slice {
                        section: section.clone(),
                        index,
                    },
                );
            }
        }
        // From here on the page table, then the mapped view, are the only
        // owners of the section; later the registry must be.
        drop(section);
        let at = reserve(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);

        let lease = resolve_guest_range(at as usize, PAGE).expect("page 0 resolves");
        assert_eq!(lease.range.id, id);
        let submission = fresh_submission();
        let key = lease
            .retain(submission)
            .expect("the submission retains the range");
        drop(lease);

        // The guest frees the page numbers while its view is still mapped:
        // the view keeps resolving and the range stays retained.
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        let through_view =
            resolve_guest_range(at as usize, PAGE).expect("a live view still resolves");
        assert_eq!(through_view.range.id, id);
        drop(through_view);
        assert!(retained(submission, id, GENERATION, 0, PAGE as u64).is_ok());

        // Unmap the view: neither page numbers nor views remain, so only the
        // retained range can keep the backing alive now.
        assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        assert_eq!(
            resolve_guest_range(at as usize, PAGE).unwrap_err(),
            ResolveError::NotMapped,
            "a range with no page numbers or views must not resolve"
        );
        let held = retained(submission, id, GENERATION, 0, PAGE as u64)
            .expect("the retained range keeps the backing alive");
        assert_eq!(held.path(), path.as_path());
        assert_eq!(held.file_size(), size as u64);
        assert_eq!(held.range.length, PAGE as u64);
        assert!(path.exists(), "the share file must not be unlinked");

        // A window and an allocation nobody retained are explicit errors.
        assert_eq!(
            retained(submission, id, GENERATION, PAGE as u64, PAGE as u64).unwrap_err(),
            RegistryError::MissingRange {
                id,
                offset: PAGE as u64,
                length: PAGE as u64,
            }
        );
        assert_eq!(
            retained(submission, id + 1, GENERATION, 0, PAGE as u64).unwrap_err(),
            RegistryError::Missing { id: id + 1 }
        );

        drop(held);
        assert_eq!(retire(submission).expect("retire the submission"), 1);
        assert!(
            !is_retained(key),
            "the last reference must drop the retained range"
        );
    }

    /// Two submissions retain one range: it lives until both fences retire,
    /// in either order.
    #[test]
    fn two_submissions_share_a_range_until_the_last_retires() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let first = fresh_submission();
        let second = fresh_submission();
        let key = lease.retain(first).expect("the first submission retains");
        assert_eq!(
            lease.retain(second).expect("the second submission retains"),
            key
        );
        assert!(is_retained(key), "the range is registered");

        // The first fence retires: the second submission still holds it.
        assert_eq!(retire(first).expect("retire the first"), 1);
        assert!(
            is_retained(key),
            "a second submission still references the range"
        );
        assert!(retained(second, key.id, GENERATION, key.offset, key.length).is_ok());

        // The last fence retires: the registry drops the range and backing.
        assert_eq!(retire(second).expect("retire the second"), 1);
        assert!(!is_retained(key), "the last reference must drop the range");
        assert_eq!(
            retained(second, key.id, GENERATION, key.offset, key.length).unwrap_err(),
            RegistryError::Retired(second),
        );
    }

    /// Retiring the later submission first must leave the earlier one's
    /// range intact.
    #[test]
    fn release_order_does_not_matter() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let earlier = fresh_submission();
        let later = fresh_submission();
        let key = lease.retain(earlier).expect("retain for the earlier fence");
        lease.retain(later).expect("retain for the later fence");

        assert_eq!(retire(later).expect("retire the later fence"), 1);
        assert!(is_retained(key), "the earlier fence still references it");
        assert!(retained(earlier, key.id, GENERATION, key.offset, key.length).is_ok());

        assert_eq!(retire(earlier).expect("retire the earlier fence"), 1);
        assert!(!is_retained(key), "the last reference is gone");
    }

    /// A generation the caller still believes in, or a backing the page
    /// table forged, is an explicit error.
    #[test]
    fn stale_generations_are_explicit_errors() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let submission = fresh_submission();
        let key = lease.retain(submission).expect("retain");
        assert_eq!(
            retained(submission, key.id, GENERATION + 1, 0, PAGE as u64).unwrap_err(),
            RegistryError::StaleGeneration {
                id: key.id,
                generation: GENERATION + 1,
            }
        );
        assert_eq!(retire(submission).expect("retire"), 1);

        // A backing with a foreign generation or a zero id can only come
        // from a corrupted page table; the resolver refuses it rather than
        // naming a file the host must not map.
        let directory = tempfile::tempdir().expect("a temporary directory");
        for (index, (id, generation)) in [(7, GENERATION + 1), (0, GENERATION)]
            .into_iter()
            .enumerate()
        {
            let at = FAKE + index * PAGE;
            let name = format!("stale-{index}.bin");
            let (section, _path) = share_section(&directory, &name, 1, id, generation);
            let view = FakeView::install(at, PAGE, super::super::Slice { section, index: 0 });
            assert_eq!(
                resolve_guest_range(at, PAGE).unwrap_err(),
                ResolveError::Stale,
                "id {id} generation {generation} must be refused"
            );
            drop(view);
        }
    }

    /// A rounded span that leaves its section is refused, even when a view
    /// claims to show it; a span that fits still resolves.
    #[test]
    fn ranges_crossing_a_section_end_are_refused() {
        let share = SyntheticShare::new(1);
        let view = FakeView::install(
            FAKE,
            2 * PAGE,
            super::super::Slice {
                section: share.section.clone(),
                index: 0,
            },
        );
        assert_eq!(
            resolve_guest_range(FAKE, 2 * PAGE).unwrap_err(),
            ResolveError::OutOfBounds,
            "the second page is outside the one-page section"
        );
        let inside = resolve_guest_range(FAKE, PAGE).expect("the first page is inside");
        assert_eq!((inside.range.offset, inside.range.length), (0, PAGE as u64));
        drop(view);
    }

    /// Releasing twice, retiring twice, and retaining after retirement are
    /// explicit errors (the documented policy), not silent no-ops.
    #[test]
    fn double_release_is_an_explicit_error() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let submission = fresh_submission();
        let key = lease.retain(submission).expect("retain");
        assert_eq!(release(key, submission), Ok(()));
        assert_eq!(
            release(key, submission).unwrap_err(),
            RegistryError::NotRetained {
                submission,
                key,
            }
        );
        assert!(!is_retained(key), "the released range is gone");

        // A submission with no references left may retire, but only once.
        assert_eq!(retire(submission).expect("retire after release"), 0);
        assert_eq!(
            retire(submission).unwrap_err(),
            RegistryError::Retired(submission)
        );
        assert_eq!(
            lease.retain(submission).unwrap_err(),
            RegistryError::Retired(submission)
        );
        assert_eq!(
            release(key, submission).unwrap_err(),
            RegistryError::Retired(submission)
        );
    }

    /// A retirement is remembered for the life of the process: the retired
    /// number is refused from then on, and only a fresh one may retain. The
    /// lease tests once shared fixed numbers and depended on their execution
    /// order because of this; the numbers are now per test.
    #[test]
    fn a_retired_submission_number_never_names_a_new_lease() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let retired = fresh_submission();
        let fresh = fresh_submission();
        assert_ne!(retired, fresh, "the allocator must not repeat a number");

        let key = lease.retain(retired).expect("retain");
        assert_eq!(retire(retired).expect("retire"), 1);
        assert!(!is_retained(key), "the retirement drops the range");
        assert_eq!(
            lease.retain(retired).unwrap_err(),
            RegistryError::Retired(retired),
            "a retired number must not retain again"
        );

        // A fresh number for the very same range is independent of it.
        let key = lease.retain(fresh).expect("a fresh submission retains");
        assert!(is_retained(key), "the fresh reference keeps the range");
        assert!(retained(fresh, key.id, GENERATION, 0, PAGE as u64).is_ok());
        assert_eq!(retire(fresh).expect("retire the fresh submission"), 1);
        assert!(!is_retained(key), "the last reference is gone");
    }

    /// The seam over the real share provider: an unaligned guest address
    /// resolves to the page window that holds it, retaining pins the range in
    /// the registry until the token retires, and a second retirement is
    /// refused. No GPU work.
    #[test]
    fn share_resolver_pins_unaligned_ranges_until_retire() {
        let share = SyntheticShare::new(2);
        let view = FakeView::install(
            FAKE,
            2 * PAGE,
            super::super::Slice {
                section: share.section.clone(),
                index: 0,
            },
        );
        let resolver = ShareResolver;

        let range = resolver
            .resolve(FAKE as u64 + 0x123, 0x456)
            .expect("an unaligned guest address");
        assert_eq!(range.allocation_id, share.section.share.as_ref().unwrap().id);
        assert_eq!(range.generation, 1);
        assert_eq!(range.byte_offset, 0, "the containing page is the window");
        assert_eq!(range.size, PAGE as u64);
        let key = RangeKey {
            id: range.allocation_id,
            generation: GENERATION,
            offset: range.byte_offset,
            length: range.size,
        };

        let token = resolver.retain(&range).expect("retain the range");
        assert!(is_retained(key), "the pin must register the range");
        // Retaining a range that was not resolved here is refused, not
        // silently accepted.
        assert!(matches!(
            resolver.retain(&range),
            Err(SeamError::Retain(_)),
        ));

        resolver.retire(token).expect("retire the pin");
        assert!(!is_retained(key), "the pin must release the range");
        assert!(
            matches!(resolver.retire(token), Err(SeamError::Retire(_))),
            "a second retirement is refused"
        );

        // A range reaching into the second page resolves to both pages.
        let spanning = resolver
            .resolve(FAKE as u64 + PAGE as u64 - 0x10, 0x20)
            .expect("a page-spanning range");
        assert_eq!((spanning.byte_offset, spanning.size), (0, 2 * PAGE as u64));
        let token = resolver.retain(&spanning).expect("retain the span");
        resolver.retire(token).expect("retire the span");

        // The byte past the mapped view is not granted.
        assert!(matches!(
            resolver.resolve(FAKE as u64 + 2 * PAGE as u64, 1),
            Err(SeamError::Unmapped { .. }),
        ));
        assert!(matches!(
            resolver.resolve(0, 1),
            Err(SeamError::Invalid(_)),
        ));
        drop(view);
    }

    /// A pagefile-backed section has no host-visible backing: the seam
    /// reports it instead of naming a file that does not exist.
    #[test]
    fn share_resolver_reports_pagefile_and_stale_backing() {
        let file = tempfile::tempfile().expect("a temporary file");
        file.set_len(PAGE as u64).expect("one page");
        let section = super::super::file_section(&file, PAGE, None).expect("a pagefile section");
        let view = FakeView::install(
            FAKE,
            PAGE,
            super::super::Slice {
                section: section.clone(),
                index: 0,
            },
        );
        assert!(
            matches!(
                ShareResolver.resolve(FAKE as u64, 0x10),
                Err(SeamError::Unsupported(_))
            ),
            "a pagefile-backed range must not resolve"
        );
        drop(view);

        // A backing with another generation is stale, not reused.
        let directory = tempfile::tempdir().expect("a temporary directory");
        let (section, _path) = share_section(&directory, "stale.bin", 1, 7, GENERATION + 1);
        let view = FakeView::install(
            FAKE + PAGE,
            PAGE,
            super::super::Slice { section, index: 0 },
        );
        assert!(
            matches!(
                ShareResolver.resolve((FAKE + PAGE) as u64, 0x10),
                Err(SeamError::Invalid(_))
            ),
            "a stale generation must be refused"
        );
        drop(view);
    }

    /// Console device creation installs the share resolver for a Metal
    /// session, so every later bind has one, and never installs it for the
    /// D3D12 reference backend. A second install is a no-op: the same
    /// resolver stays installed, exactly once.
    ///
    /// This is the device-creation path's install entry (the HLE calls
    /// `install_share_resolver` from `d3d12x::create_device`); the share
    /// directory is deliberately not consulted, so an unconfigured session
    /// still resolves ranges from the page registry.
    #[test]
    fn device_creation_installs_the_resolver_only_for_metal_and_once() {
        uninstall_for_test();
        install_for_backend(GpuBackend::D3d12);
        assert!(
            bindings::resolver().is_none(),
            "the D3D12 reference backend must not install the share resolver"
        );

        install_for_backend(GpuBackend::Metal);
        let first = bindings::resolver().expect("the Metal backend installs the resolver");
        install_for_backend(GpuBackend::Metal);
        let second = bindings::resolver().expect("the resolver stays installed");
        assert!(
            Arc::ptr_eq(&first, &second),
            "a second install must not replace the resolver"
        );

        uninstall_for_test();
        assert!(bindings::resolver().is_none());
    }

    /// The literal device-creation entry (`d3d12x::create_device` calls
    /// `xmem::install_share_resolver`) installs the resolver in a Metal
    /// session. Run with `VICTORY_BACKEND=metal`; skipped otherwise, because
    /// the backend switch is read once per process.
    #[test]
    fn device_creation_entry_installs_under_a_metal_session() {
        let metal = std::env::var("VICTORY_BACKEND")
            .map(|value| value.trim().eq_ignore_ascii_case("metal"))
            .unwrap_or(false);
        if !metal {
            eprintln!("skipped: VICTORY_BACKEND is not metal");
            return;
        }
        uninstall_for_test();
        super::super::install_share_resolver();
        assert!(
            bindings::resolver().is_some(),
            "device creation must leave the resolver installed"
        );
        uninstall_for_test();
    }

    /// The resolver installed at device creation is the real share provider:
    /// it names the page registry's allocation id and refuses an unprovable
    /// range by name instead of fabricating one.
    #[test]
    fn the_installed_resolver_uses_the_real_id_and_refuses_unknown_ranges() {
        let share = SyntheticShare::new(1);
        let at = FAKE + 16 * PAGE;
        let view = FakeView::install(
            at,
            PAGE,
            super::super::Slice {
                section: share.section.clone(),
                index: 0,
            },
        );
        uninstall_for_test();
        install_for_backend(GpuBackend::Metal);
        let resolver = bindings::resolver().expect("installed at device creation");

        let range = resolver
            .resolve(at as u64 + 0x40, 0x80)
            .expect("a tracked range resolves");
        let id = share.section.share.as_ref().expect("a share backing").id;
        assert_eq!(range.allocation_id, id, "the registry's real allocation id");
        assert_eq!(range.generation, 1);
        assert_eq!((range.byte_offset, range.size), (0, PAGE as u64));

        let token = resolver.retain(&range).expect("pin the resolved range");
        resolver.retire(token).expect("retire the pin");

        // A page no tracked view shows is refused by name: no allocation id
        // is invented for it.
        let missing = at + PAGE;
        assert_eq!(
            resolver.resolve(missing as u64, 0x10).unwrap_err(),
            SeamError::Unmapped {
                address: missing as u64,
                len: 0x10,
            }
        );

        drop(view);
        uninstall_for_test();
    }

    /// The retired-row cap is a positive default and the env knob only accepts
    /// a positive integer; the registry seeds itself with the result.
    #[test]
    fn retired_capacity_defaults_and_parses() {
        assert_eq!(parse_capacity(None), SUBMISSIONS_CAPACITY);
        assert_eq!(parse_capacity(Some("")), SUBMISSIONS_CAPACITY);
        assert_eq!(parse_capacity(Some("0")), SUBMISSIONS_CAPACITY);
        assert_eq!(parse_capacity(Some("-4")), SUBMISSIONS_CAPACITY);
        assert_eq!(parse_capacity(Some("many")), SUBMISSIONS_CAPACITY);
        assert_eq!(parse_capacity(Some("17")), 17);
        assert_eq!(parse_capacity(Some(" 17 ")), 17);

        let registry = Registry::new(SUBMISSIONS_CAPACITY);
        assert_eq!(registry.capacity, SUBMISSIONS_CAPACITY);
        assert_eq!(registry.retired_watermark, None);
        assert!(registry.submissions.is_empty());
        // The process registry has a real, positive bound.
        assert!(bounds().capacity >= 1);
    }

    /// A sustained streaming loop must keep the submission map O(in-flight)
    /// rather than growing one row per retired submission forever.
    ///
    /// 10,000 cycles of retain/use/retire (two pins each) over one retained
    /// range, with fences retiring out of order and one live pin spanning the
    /// whole run. A local registry with a small capacity keeps the test
    /// deterministic and independent of the process registry's contents.
    #[test]
    fn sustained_streaming_bounds_the_retired_submission_map() {
        const CYCLES: u64 = 10_000;
        const CAPACITY: usize = 64;

        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let mut registry = Registry::new(CAPACITY);

        // The live pin spans the whole run: the retained range and its backing
        // must survive every eviction until this fence retires.
        let live = 1;
        let victim = 2;
        let key = registry.retain(&lease, live).expect("the live pin retains");
        registry
            .retain(&lease, victim)
            .expect("the reuse victim retains");
        assert_eq!(registry.retire(victim).expect("retire the victim"), 1);

        let mut watermark = registry.retired_watermark;
        let mut peak = registry.submissions.len();
        for cycle in 0..CYCLES {
            let older = 1000 + cycle * 2;
            let newer = older + 1;
            assert_eq!(
                registry
                    .retain(&lease, older)
                    .expect("retain the older fence"),
                key
            );
            assert_eq!(
                registry
                    .retain(&lease, newer)
                    .expect("retain the newer fence"),
                key
            );

            // A completion handler may run for the newer fence first: the
            // registry must not let that release the older fence's reference.
            assert_eq!(registry.retire(newer).expect("retire the newer fence"), 1);
            assert_eq!(registry.retire(older).expect("retire the older fence"), 1);

            // The live pin's range is never dropped, at any point in the loop.
            assert!(
                registry.is_retained(&key),
                "cycle {cycle} dropped a live pin's retained range"
            );
            assert!(
                registry
                    .retained(live, key.id, GENERATION, key.offset, key.length)
                    .is_ok(),
                "cycle {cycle} lost the live pin's lease"
            );

            let len = registry.submissions.len();
            assert!(
                len <= registry.capacity,
                "cycle {cycle}: {len} submission rows exceed capacity {}",
                registry.capacity
            );
            assert!(
                registry.retained.len() == 1,
                "cycle {cycle}: one range must stay retained"
            );
            peak = peak.max(len);

            // The watermark only exists once the map first exceeds capacity,
            // then advances monotonically as more retired rows are evicted.
            if let Some(advanced) = registry.retired_watermark {
                assert!(
                    watermark.is_none_or(|watermark| advanced >= watermark),
                    "cycle {cycle}: the watermark went backwards"
                );
                watermark = Some(advanced);
            }
        }

        let watermark = watermark.expect("the map was bounded by eviction");
        assert!(
            victim <= watermark,
            "the victim {victim} must be at or below the watermark {watermark}"
        );
        assert!(!registry.submissions.contains_key(&victim));

        // Reusing the evicted number must not resurrect a stale range: the
        // absent row is retired by the watermark, exactly like a present
        // `Submission::Retired` row.
        assert_eq!(
            registry.retain(&lease, victim).unwrap_err(),
            RegistryError::Retired(victim),
            "an evicted number must not retain again"
        );
        assert_eq!(
            registry.release(key, victim).unwrap_err(),
            RegistryError::Retired(victim)
        );
        assert_eq!(
            registry
                .retained(victim, key.id, GENERATION, key.offset, key.length)
                .unwrap_err(),
            RegistryError::Retired(victim)
        );
        assert_eq!(
            registry.retire(victim).unwrap_err(),
            RegistryError::Retired(victim)
        );
        // Above the watermark an absent row is unknown, not retired.
        let never = 1 << 40;
        assert_eq!(
            registry.release(key, never).unwrap_err(),
            RegistryError::UnknownSubmission(never)
        );

        // The live pin still owns the range; only its own fence drops it.
        assert!(registry.is_retained(&key));
        assert_eq!(registry.retire(live).expect("retire the live pin"), 1);
        assert!(!registry.is_retained(&key));

        eprintln!(
            "bounded share registry: cycles={CYCLES} peak={peak} capacity={} watermark={watermark}",
            registry.capacity
        );
    }

    /// `XMemVirtualQuery` as the guest sees it.
    fn guest_query(at: *const std::ffi::c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, at, &mut info), 1);
        info
    }

    /// ADR-043: a reservation owns no backing, each commit maps exactly the
    /// pages it names over the reservation's placeholders, all commits share
    /// one section at their own offsets, a decommit returns pages to the
    /// reservation, a re-commit is zeroed (Win32 leaves the contents
    /// undefined; this is the documented deterministic result), and the
    /// host reads the guest's bytes from the named file.
    #[test]
    fn virtual_regions_track_commits_decommits_and_resolve() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let live_before = live();
        let size = 8 * PAGE;
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null(), "a virtual reservation failed");
        let base = at as usize;

        // A reserve alone owns address space, not bytes: no share file, no
        // resolvable page, and the query says `MEM_RESERVE`.
        assert_eq!(live(), live_before, "a reserve must not create a backing");
        assert_eq!(
            resolve_guest_range(base, PAGE).unwrap_err(),
            ResolveError::Uncommitted
        );
        let reserved = guest_query(at);
        assert_eq!(reserved.host.State, MEM_RESERVE);
        assert_eq!(reserved.host.Protect, PAGE_NOACCESS);
        assert_eq!(reserved.host.RegionSize, size);
        assert_eq!(reserved.host.AllocationBase as usize, base);

        // Commit the second page only: exactly that page resolves, and the
        // host can map its bytes from the one reservation section.
        let second = base + 2 * PAGE;
        let committed = XMemVirtualAlloc(
            second as *mut std::ffi::c_void,
            PAGE,
            MEM_COMMIT,
            0,
            PAGE_READWRITE,
        );
        assert_eq!(committed as usize, second);
        assert_eq!(live(), live_before + 1, "the commit creates the backing");
        let state = guest_query(committed);
        assert_eq!(state.host.State, MEM_COMMIT);
        assert_eq!(state.host.Protect, PAGE_READWRITE);
        assert_eq!(state.host.BaseAddress as usize, second);
        assert_eq!(state.host.RegionSize, PAGE);
        assert_eq!(state.host.AllocationBase as usize, base);
        let gap = guest_query(at);
        assert_eq!(gap.host.State, MEM_RESERVE);
        assert_eq!(gap.host.RegionSize, second - base, "up to the commit");

        let payload: Vec<u8> = (0..96).map(|i| 0xA0u8 ^ i as u8).collect();
        // SAFETY: the committed page is read-write and the payload fits.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                (second + 0x80) as *mut u8,
                payload.len(),
            );
            let mut read = vec![0u8; payload.len()];
            std::ptr::copy_nonoverlapping(
                (second + 0x80) as *const u8,
                read.as_mut_ptr(),
                payload.len(),
            );
            assert_eq!(read, payload, "the guest reads what it wrote");
        }

        // The host maps the same file the guest wrote: the resolve names the
        // committed page at section offset 2 pages, and the bytes agree.
        let lease = resolve_guest_range(second, PAGE).expect("the committed page resolves");
        assert_eq!(lease.range.offset, (2 * PAGE) as u64);
        assert_eq!(lease.range.length, PAGE as u64);
        assert_eq!(lease.file_size(), size as u64);
        assert_eq!(lease.path().parent(), Some(directory.path()));
        let bytes = std::fs::read(lease.path()).expect("the share file");
        assert_eq!(
            &bytes[2 * PAGE + 0x80..2 * PAGE + 0x80 + payload.len()],
            &payload[..]
        );
        drop(lease);

        // A second commit extends the same backing; adjacent committed runs
        // resolve as one contiguous file range, not two.
        let third = base + 3 * PAGE;
        assert_eq!(
            XMemVirtualAlloc(
                third as *mut std::ffi::c_void,
                PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            third
        );
        let both = resolve_guest_range(second, 2 * PAGE).expect("two commits, one run");
        assert_eq!(both.range.offset, (2 * PAGE) as u64);
        assert_eq!(both.range.length, (2 * PAGE) as u64);
        assert_eq!(live(), live_before + 1, "both commits share one file");
        drop(both);

        // Decommit the second page: the reservation keeps the range, the
        // page becomes inaccessible, its neighbour is untouched.
        assert_eq!(
            XMemVirtualAlloc(
                second as *mut std::ffi::c_void,
                PAGE,
                MEM_DECOMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            second
        );
        assert_eq!(guest_query(second as *const std::ffi::c_void).host.State, MEM_RESERVE);
        assert_eq!(
            resolve_guest_range(second, PAGE).unwrap_err(),
            ResolveError::Uncommitted
        );
        assert_eq!(
            resolve_guest_range(third, PAGE)
                .expect("the neighbour page survives")
                .range
                .offset,
            (3 * PAGE) as u64
        );

        // Re-commit the same range: readable, and zeroed rather than stale.
        assert_eq!(
            XMemVirtualAlloc(
                second as *mut std::ffi::c_void,
                PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            second
        );
        assert_eq!(guest_query(second as *const std::ffi::c_void).host.State, MEM_COMMIT);
        // SAFETY: the page is committed read-write again.
        assert_eq!(
            unsafe { ((second + 0x80) as *const u8).read() },
            0,
            "a re-commit is deterministic (zeroed), not stale bytes"
        );
        assert!(resolve_guest_range(second, 2 * PAGE).is_ok());

        // `MEM_RESET` keeps the page committed and discards its contents.
        // SAFETY: the page is committed read-write.
        unsafe { *(second as *mut u8) = 0x33 };
        assert_eq!(
            XMemVirtualAlloc(
                second as *mut std::ffi::c_void,
                PAGE,
                MEM_RESET,
                0,
                PAGE_READWRITE
            ) as usize,
            second
        );
        assert_eq!(
            guest_query(second as *const std::ffi::c_void).host.State,
            MEM_COMMIT
        );
        // SAFETY: still committed and readable.
        assert_eq!(unsafe { (second as *const u8).read() }, 0);

        // The counters are named, and every refusal reason is bounded.
        let counters = virtual_counters();
        assert!(counters.reserves >= 1, "{counters:?}");
        assert!(counters.commits >= 3, "{counters:?}");
        assert!(counters.decommits >= 1, "{counters:?}");
        assert_eq!(counters.backings, 1, "one file per reservation");
        assert!(last_refusal().len() <= 120);

        // Release frees the whole reservation and its backing exactly once.
        assert_eq!(VirtualFree(at, 0, MEM_RELEASE), 1);
        assert_eq!(live(), live_before, "the share backing is released");
        assert_eq!(
            resolve_guest_range(second, PAGE).unwrap_err(),
            ResolveError::NotMapped
        );
        assert_eq!(
            VirtualFree(at, 0, MEM_RELEASE),
            0,
            "a double free is rejected"
        );
        assert!(virtual_counters().releases >= 1);
    }

    /// A commit past its reservation is `ERROR_INVALID_ADDRESS`, never a
    /// page that looks committed and faults on first touch. An unaligned
    /// commit size rounds up to whole 64 KiB pages.
    #[test]
    fn virtual_commits_are_bounded_and_page_rounded() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null());
        let base = at as usize;

        let before = virtual_counters().refusals;
        // SAFETY: reads and writes this thread's last-error value.
        unsafe { SetLastError(0) };
        let past = XMemVirtualAlloc(
            (base + PAGE) as *mut std::ffi::c_void,
            4 * PAGE,
            MEM_COMMIT,
            0,
            PAGE_READWRITE,
        );
        assert!(past.is_null(), "a commit past the reservation must fail");
        // SAFETY: reads this thread's last-error value.
        assert_eq!(unsafe { GetLastError() }, ERROR_INVALID_ADDRESS);
        assert_eq!(virtual_counters().refusals, before + 1, "counted once");

        // A page-and-a-bit commit covers two whole pages.
        let size = PAGE + 0x321;
        let committed = XMemVirtualAlloc(
            base as *mut std::ffi::c_void,
            size,
            MEM_COMMIT,
            0,
            PAGE_READWRITE,
        );
        assert_eq!(committed as usize, base);
        assert_eq!(guest_query(committed).host.RegionSize, 2 * PAGE);
        let lease = resolve_guest_range(base, size).expect("the rounded commit resolves");
        assert_eq!((lease.range.offset, lease.range.length), (0, (2 * PAGE) as u64));
        drop(lease);
        // The last byte of the rounded range is a real page, not padding.
        // SAFETY: both pages were committed by the call above.
        unsafe {
            *((base + 2 * PAGE - 1) as *mut u8) = 0x5a;
            assert_eq!(((base + 2 * PAGE - 1) as *const u8).read(), 0x5a);
        }
        assert!(last_refusal().len() <= 120);
        // The console has no separate virtual-free export: releasing through
        // `XMemVirtualAlloc` itself works once; a second release is refused.
        assert_eq!(
            XMemVirtualAlloc(
                at as *mut std::ffi::c_void,
                0,
                MEM_RELEASE,
                0,
                PAGE_NOACCESS
            ) as usize,
            at as usize
        );
        assert!(
            XMemVirtualAlloc(
                at as *mut std::ffi::c_void,
                0,
                MEM_RELEASE,
                0,
                PAGE_NOACCESS
            )
            .is_null()
        );
    }

    /// A decommit inside one multi-page commit unmaps its view, keeps both
    /// neighbours committed with their bytes, and re-commits the hole.
    #[test]
    fn decommitting_inside_one_commit_keeps_its_neighbours() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null());
        let base = at as usize;
        // One commit call maps one three-page view.
        assert_eq!(
            XMemVirtualAlloc(
                at,
                3 * PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            base
        );
        // SAFETY: pages 0 and 2 are committed read-write.
        unsafe {
            *(base as *mut u8) = 0x11;
            *((base + 2 * PAGE) as *mut u8) = 0x33;
        }
        assert_eq!(resolve_guest_range(base, 3 * PAGE).unwrap().range.length, (3 * PAGE) as u64);

        // Decommit the middle page of that one view.
        assert_eq!(
            XMemVirtualAlloc(
                (base + PAGE) as *mut std::ffi::c_void,
                PAGE,
                MEM_DECOMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            base + PAGE
        );
        assert_eq!(
            resolve_guest_range(base + PAGE, PAGE).unwrap_err(),
            ResolveError::Uncommitted
        );
        let left = resolve_guest_range(base, PAGE).expect("the left page survives");
        assert_eq!((left.range.offset, left.range.length), (0, PAGE as u64));
        drop(left);
        let right = resolve_guest_range(base + 2 * PAGE, PAGE).expect("the right page survives");
        assert_eq!(
            (right.range.offset, right.range.length),
            ((2 * PAGE) as u64, PAGE as u64)
        );
        drop(right);
        // SAFETY: both neighbours are committed read-write.
        unsafe {
            assert_eq!((base as *const u8).read(), 0x11);
            assert_eq!(((base + 2 * PAGE) as *const u8).read(), 0x33);
        }

        // Re-commit the hole: zeroed, and the three pages are one run again.
        assert_eq!(
            XMemVirtualAlloc(
                (base + PAGE) as *mut std::ffi::c_void,
                PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            base + PAGE
        );
        // SAFETY: the hole is committed read-write again.
        assert_eq!(unsafe { ((base + PAGE) as *const u8).read() }, 0);
        let all = resolve_guest_range(base, 3 * PAGE).expect("all three pages resolve");
        assert_eq!((all.range.offset, all.range.length), (0, (3 * PAGE) as u64));
        drop(all);
        assert_eq!(VirtualFree(at, 0, MEM_RELEASE), 1);
    }

    /// A 64 KiB placeholder region keeps the physical-page flow (`map`
    /// splits the placeholder, `/mem` views it), and releasing it now goes
    /// through the guarded `VirtualFree`: every view becomes a placeholder
    /// and the split reservation is freed as one.
    #[test]
    fn releasing_a_region_with_physical_page_views_frees_it() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let pages = allocate(2);
        let at = reserve(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);
        assert_eq!(guest_query(at).host.State, MEM_COMMIT);
        assert_eq!(VirtualFree(at, 0, MEM_RELEASE), 1);
        assert_eq!(guest_query(at).host.State, MEM_FREE, "the range is free");
        assert_eq!(VirtualFree(at, 0, MEM_RELEASE), 0, "a double free is rejected");
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The shape of the title's first at-address buffer (guest VA
    /// 0xE730000, +174 MiB): a reservation whose head is committed and whose
    /// tail is only reserved. The resolver must never answer a bindable range
    /// for it — only the whole committed head or the named refusal — so
    /// resource creation keeps its resilient path and no bind can map a page
    /// the guest never committed.
    #[test]
    fn partially_committed_reservation_never_resolves_as_a_bindable_range() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let _share = ShareDirectory::configured(directory.path());
        let size = 8 * PAGE;
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null(), "the reservation failed");
        let base = at as usize;
        // Two committed pages, six only reserved: the title's pattern.
        assert_eq!(
            XMemVirtualAlloc(
                base as *mut std::ffi::c_void,
                2 * PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            base
        );

        // The committed head alone is a readable, bindable window.
        let head = resolve_guest_range(base, 2 * PAGE).expect("the committed head resolves");
        assert_eq!((head.range.offset, head.range.length), (0, (2 * PAGE) as u64));
        drop(head);

        // A range that starts in the committed head and reaches into the
        // reserved tail is refused by name, not silently truncated to its
        // committed prefix.
        let error = resolve_guest_range(base, 6 * PAGE).unwrap_err();
        assert_eq!(error, ResolveError::Uncommitted);
        assert!(error.to_string().len() <= 120, "the reason is bounded");
        // The seam the host binds through answers the same named refusal: a
        // resource at this range is created and never bound, never guessed.
        match ShareResolver.resolve(base as u64, (6 * PAGE) as u64) {
            Err(SeamError::Invalid(reason)) => {
                assert!(reason.len() <= 120, "the reason is bounded");
                assert!(reason.contains("not committed"), "{reason}");
            }
            other => panic!("a partially committed range resolved: {other:?}"),
        }
        // The tail alone is refused too.
        assert_eq!(
            resolve_guest_range(base + 4 * PAGE, PAGE).unwrap_err(),
            ResolveError::Uncommitted
        );

        // Committing the tail makes the whole span one contiguous window.
        assert_eq!(
            XMemVirtualAlloc(
                (base + 2 * PAGE) as *mut std::ffi::c_void,
                6 * PAGE,
                MEM_COMMIT,
                0,
                PAGE_READWRITE
            ) as usize,
            base + 2 * PAGE
        );
        let all = resolve_guest_range(base, size).expect("the whole span resolves");
        assert_eq!((all.range.offset, all.range.length), (0, size as u64));
        drop(all);

        // A committed page without read access is still not bindable: the
        // bytes exist, but the host must not map them.
        assert_eq!(
            XMemVirtualAlloc(
                base as *mut std::ffi::c_void,
                PAGE,
                MEM_COMMIT,
                0,
                PAGE_NOACCESS
            ) as usize,
            base
        );
        assert_eq!(
            resolve_guest_range(base, PAGE).unwrap_err(),
            ResolveError::Inaccessible
        );
        assert!(VirtualFree(at, 0, MEM_RELEASE) == 1);
    }
}
