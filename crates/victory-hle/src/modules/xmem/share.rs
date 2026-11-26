//! Opt-in host-visible backing for XMem physical pages.
//!
//! Production XMem physical pages live in the pagefile, which another
//! process cannot name. A native host process (the Metal renderer) needs a
//! file it can map: set `VICTORY_METAL_SHARE_DIR` to a session-private,
//! Win-visible directory and every physical-page allocation is backed by a
//! file there instead. The guest still maps those pages through the existing
//! placeholder views; only the backing changes.
//!
//! The directory is the whole contract: it must belong to this session and
//! be cleaned up by the harness. Files are named `{id:016x}.bin`, where the
//! id is monotonic and never reused. [`Backing::generation`] is fixed at 1
//! for this first protocol, so a stale id can never name a new allocation.
//!
//! Unset (or empty), the provider is off and the pagefile path is unchanged.
//! A configured provider that cannot create or map its file falls back to the
//! pagefile too, but never pretends otherwise: [`resolve_guest_range`] only
//! succeeds for a tracked view of a section with a [`Backing`], and reports
//! [`ResolveError::Pagefile`] otherwise.
//!
//! Sparse files matter here: the console allocates multi-GiB heaps, so the
//! file is sized with `set_len` and its pages are never written ahead of the
//! guest. Only small synthetic sizes are exercised by the tests below.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use super::{PAGE, Section};

/// Names the session's directory for host-visible XMem backing.
pub(crate) const DIRECTORY: &str = "VICTORY_METAL_SHARE_DIR";

/// The only backing generation understood by this first protocol.
pub(crate) const GENERATION: u64 = 1;

/// Attempts at a fresh file name before giving up (a reused directory can
/// already hold names from an earlier process).
const ATTEMPTS: usize = 8;

/// The next share id; ids are monotonic and never reused.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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

/// Why a configured share provider could not back an allocation.
#[derive(Debug)]
pub(crate) enum Failure {
    /// `VICTORY_METAL_SHARE_DIR` is unset or empty.
    Unconfigured,
    /// Size is zero or not whole 64 KiB pages.
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
            Failure::Size(size) => write!(f, "{size} is not a non-zero multiple of {PAGE}"),
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
        let key = RangeKey::of(&self.range);
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
            || end > self.range.file_size
        {
            return Err(RegistryError::InvalidRange {
                offset: key.offset,
                length: key.length,
            });
        }
        let mut registry = registry();
        match registry.submissions.get_mut(&submission) {
            Some(Submission::Retired) => return Err(RegistryError::Retired(submission)),
            Some(Submission::Open(references)) => {
                *references.entry(key).or_insert(0) += 1;
            }
            None => {
                let mut references = BTreeMap::new();
                references.insert(key, 1);
                registry
                    .submissions
                    .insert(submission, Submission::Open(references));
            }
        }
        registry
            .retained
            .entry(key)
            .or_insert_with(|| Retained {
                lease: self.clone(),
                submissions: BTreeMap::new(),
            })
            .submissions
            .entry(submission)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        Ok(key)
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

#[derive(Default)]
struct Registry {
    /// Retained ranges, by identity.
    retained: HashMap<RangeKey, Retained>,
    /// Every submission that retained something, open or retired. Retired
    /// ones stay so a reused number cannot resurrect a stale range and a
    /// duplicate retirement is visible.
    submissions: HashMap<u64, Submission>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));

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
    let mut registry = registry();
    match registry.submissions.get_mut(&submission) {
        None => return Err(RegistryError::UnknownSubmission(submission)),
        Some(Submission::Retired) => return Err(RegistryError::Retired(submission)),
        Some(Submission::Open(references)) => {
            let Some(count) = references.get_mut(&key) else {
                return Err(RegistryError::NotRetained { submission, key });
            };
            if *count > 1 {
                *count -= 1;
                if let Some(entry) = registry.retained.get_mut(&key)
                    && let Some(count) = entry.submissions.get_mut(&submission)
                {
                    *count -= 1;
                }
                return Ok(());
            }
            references.remove(&key);
        }
    }
    drop_one(&mut registry, &key, submission);
    Ok(())
}

/// Retires `submission`'s fence: drops every reference it holds and releases
/// the ranges no other submission references any more.
///
/// Returns the number of references released (a submission that retained one
/// range twice releases two). A duplicate retirement is
/// [`RegistryError::Retired`], not a silent success, and a retired
/// submission may not retain again.
pub(crate) fn retire(submission: u64) -> Result<usize, RegistryError> {
    let mut registry = registry();
    match registry.submissions.get(&submission) {
        None => return Err(RegistryError::UnknownSubmission(submission)),
        Some(Submission::Retired) => return Err(RegistryError::Retired(submission)),
        Some(Submission::Open(_)) => {}
    }
    let Some(Submission::Open(references)) = registry.submissions.remove(&submission) else {
        unreachable!("an open submission was checked just above");
    };
    let mut released = 0;
    for (key, count) in &references {
        released += count;
        drop_one(&mut registry, key, submission);
    }
    registry.submissions.insert(submission, Submission::Retired);
    Ok(released)
}

/// The lease of a range that `submission` retains, checked against the
/// caller's `generation` as well.
///
/// This succeeds after the guest frees the page numbers or unmaps every view
/// showing the range: the registry, not the guest page table, keeps the
/// backing alive until retirement. A missing allocation, a stale generation,
/// a window nobody retained, and a reference from another (or a retired)
/// submission are all explicit errors.
pub(crate) fn retained(
    submission: u64,
    id: u64,
    generation: u64,
    offset: u64,
    length: u64,
) -> Result<Lease, RegistryError> {
    let registry = registry();
    match registry.submissions.get(&submission) {
        None => return Err(RegistryError::UnknownSubmission(submission)),
        Some(Submission::Retired) => return Err(RegistryError::Retired(submission)),
        Some(Submission::Open(_)) => {}
    }
    let (mut named, mut live_generation) = (false, false);
    for (key, entry) in &registry.retained {
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

/// Whether the registry still keeps `key`'s backing alive (tests and the
/// host's audit path).
pub(crate) fn is_retained(key: RangeKey) -> bool {
    registry().retained.contains_key(&key)
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
            ResolveError::NotContiguous => "the range is not one contiguous section run",
            ResolveError::Stale => "the range's backing is stale or not this session's",
            ResolveError::Pagefile => "the range's section has no host-visible backing",
        };
        f.write_str(message)
    }
}

/// The configured directory, if any. Read per allocation rather than cached:
/// the variable is the session's switch, and tests and tools set it around a
/// run.
fn directory() -> Option<PathBuf> {
    std::env::var_os(DIRECTORY)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// A fresh, monotonic, non-zero id. `None` only if the counter would wrap.
fn next_id() -> Option<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .ok()
}

/// Backs a new physical-page section of `size` with a file in the configured
/// directory and maps it.
///
/// The caller falls back to the pagefile on any error; the error stays
/// precise so the fallback is never mistaken for a shared section. A file
/// created before a later failure remains on disk for harness cleanup (it is
/// never reused, because ids are not).
pub(crate) fn section(size: usize) -> Result<Arc<Section>, Failure> {
    let Some(directory) = directory() else {
        return Err(Failure::Unconfigured);
    };
    if size == 0 || !size.is_multiple_of(PAGE) {
        return Err(Failure::Size(size));
    }
    std::fs::create_dir_all(&directory).map_err(|error| Failure::Directory {
        path: directory.clone(),
        error,
    })?;
    let (id, file, path) = create_file(&directory, size)?;
    let backing = Arc::new(Backing {
        id,
        generation: GENERATION,
        path,
        _file: file,
    });
    super::file_section(&backing._file, size, Some(backing.clone()))
        .ok_or_else(|| Failure::Mapping(io::Error::last_os_error()))
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
                file.set_len(size as u64).map_err(|error| Failure::Length {
                    path: path.clone(),
                    error,
                })?;
                return Ok((id, file, path));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Failure::Create { path, error }),
        }
    }
    Err(Failure::Exhausted)
}

/// Resolves the guest byte range `[at, at + len)` to a share file, offset and
/// length.
///
/// `at` and `len` need not be 64 KiB aligned: the result starts at the page
/// containing `at` and covers every page the range touches, so the host can
/// map it. The whole rounded span must be shown by the existing tracked
/// views and be one contiguous run of pages of one section, all within the
/// section. In particular, a range whose last byte is mapped is still
/// refused if rounding it would reach an unmapped or foreign page: never
/// grant a page the guest did not map.
///
/// This takes the physical-page lock only to inspect and clone what the
/// views already hold; the returned lease keeps the section (and its file)
/// alive with an `Arc`.
pub(crate) fn resolve_guest_range(at: usize, len: usize) -> Result<Lease, ResolveError> {
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
    let mut section: Option<Arc<Section>> = None;
    let mut first_index = None;
    let mut next_index = None;
    let mut cursor = start;
    while cursor < stop {
        let Some((base, size, slice)) = view_at(&physical, cursor) else {
            return Err(ResolveError::NotMapped);
        };
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
            file_size: section.size as u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_READWRITE};

    use crate::modules::xmem::{
        PAGES_64K, XMemAllocatePhysicalPages, XMemFreePhysicalPages, XMemMapPhysicalPages,
        XMemVirtualAlloc,
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

    /// The production wiring: a configured share directory backs physical
    /// pages with a named file, and a tracked guest range resolves to that
    /// file's id, offset and length. No GPU work.
    #[test]
    fn configured_directory_backs_pages_and_resolves_guest_ranges() {
        if cfg!(feature = "diagnostics") {
            eprintln!("skipped: diagnostic heaps keep the pagefile path");
            return;
        }
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
        if cfg!(feature = "diagnostics") {
            eprintln!("skipped: diagnostic heaps keep the pagefile path");
            return;
        }
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
        // The byte past the last mapped page is not granted by rounding.
        assert_eq!(
            resolve_guest_range(at as usize + 4 * PAGE - 1, 2).unwrap_err(),
            ResolveError::NotMapped
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

    /// The submissions the registry tests use. Each test retires what it
    /// retains, and the numbers are distinct so a test that fails mid-way
    /// cannot collide with another.
    const SUB_A: u64 = 0xA;
    const SUB_B: u64 = 0xB;
    const SUB_C: u64 = 0xC;

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
        let backing = Arc::new(Backing {
            id,
            generation,
            path: path.clone(),
            _file: file.try_clone().expect("a copy"),
        });
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
        let key = lease
            .retain(SUB_A)
            .expect("the submission retains the range");
        drop(lease);

        // The guest frees the page numbers while its view is still mapped:
        // the view keeps resolving and the range stays retained.
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        let through_view =
            resolve_guest_range(at as usize, PAGE).expect("a live view still resolves");
        assert_eq!(through_view.range.id, id);
        drop(through_view);
        assert!(retained(SUB_A, id, GENERATION, 0, PAGE as u64).is_ok());

        // Unmap the view: neither page numbers nor views remain, so only the
        // retained range can keep the backing alive now.
        assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        assert_eq!(
            resolve_guest_range(at as usize, PAGE).unwrap_err(),
            ResolveError::NotMapped,
            "a range with no page numbers or views must not resolve"
        );
        let held = retained(SUB_A, id, GENERATION, 0, PAGE as u64)
            .expect("the retained range keeps the backing alive");
        assert_eq!(held.path(), path.as_path());
        assert_eq!(held.file_size(), size as u64);
        assert_eq!(held.range.length, PAGE as u64);
        assert!(path.exists(), "the share file must not be unlinked");

        // A window and an allocation nobody retained are explicit errors.
        assert_eq!(
            retained(SUB_A, id, GENERATION, PAGE as u64, PAGE as u64).unwrap_err(),
            RegistryError::MissingRange {
                id,
                offset: PAGE as u64,
                length: PAGE as u64,
            }
        );
        assert_eq!(
            retained(SUB_A, id + 1, GENERATION, 0, PAGE as u64).unwrap_err(),
            RegistryError::Missing { id: id + 1 }
        );

        drop(held);
        assert_eq!(retire(SUB_A).expect("retire the submission"), 1);
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
        let key = lease.retain(SUB_A).expect("the first submission retains");
        assert_eq!(
            lease.retain(SUB_B).expect("the second submission retains"),
            key
        );
        assert!(is_retained(key), "the range is registered");

        // The first fence retires: the second submission still holds it.
        assert_eq!(retire(SUB_A).expect("retire the first"), 1);
        assert!(
            is_retained(key),
            "a second submission still references the range"
        );
        assert!(retained(SUB_B, key.id, GENERATION, key.offset, key.length).is_ok());

        // The last fence retires: the registry drops the range and backing.
        assert_eq!(retire(SUB_B).expect("retire the second"), 1);
        assert!(!is_retained(key), "the last reference must drop the range");
        assert_eq!(
            retained(SUB_B, key.id, GENERATION, key.offset, key.length).unwrap_err(),
            RegistryError::Retired(SUB_B),
        );
    }

    /// Retiring the later submission first must leave the earlier one's
    /// range intact.
    #[test]
    fn release_order_does_not_matter() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let key = lease.retain(SUB_A).expect("retain for the earlier fence");
        lease.retain(SUB_C).expect("retain for the later fence");

        assert_eq!(retire(SUB_C).expect("retire the later fence"), 1);
        assert!(is_retained(key), "the earlier fence still references it");
        assert!(retained(SUB_A, key.id, GENERATION, key.offset, key.length).is_ok());

        assert_eq!(retire(SUB_A).expect("retire the earlier fence"), 1);
        assert!(!is_retained(key), "the last reference is gone");
    }

    /// A generation the caller still believes in, or a backing the page
    /// table forged, is an explicit error.
    #[test]
    fn stale_generations_are_explicit_errors() {
        let share = SyntheticShare::new(1);
        let lease = share.lease(0, PAGE);
        let key = lease.retain(SUB_A).expect("retain");
        assert_eq!(
            retained(SUB_A, key.id, GENERATION + 1, 0, PAGE as u64).unwrap_err(),
            RegistryError::StaleGeneration {
                id: key.id,
                generation: GENERATION + 1,
            }
        );
        assert_eq!(retire(SUB_A).expect("retire"), 1);

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
        let key = lease.retain(SUB_A).expect("retain");
        assert_eq!(release(key, SUB_A), Ok(()));
        assert_eq!(
            release(key, SUB_A).unwrap_err(),
            RegistryError::NotRetained {
                submission: SUB_A,
                key,
            }
        );
        assert!(!is_retained(key), "the released range is gone");

        // A submission with no references left may retire, but only once.
        assert_eq!(retire(SUB_A).expect("retire after release"), 0);
        assert_eq!(retire(SUB_A).unwrap_err(), RegistryError::Retired(SUB_A));
        assert_eq!(
            lease.retain(SUB_A).unwrap_err(),
            RegistryError::Retired(SUB_A)
        );
        assert_eq!(
            release(key, SUB_A).unwrap_err(),
            RegistryError::Retired(SUB_A)
        );
    }
}
