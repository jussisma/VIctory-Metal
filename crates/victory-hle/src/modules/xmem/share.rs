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

// The resolution API's consumer is the host-IPC wave; until then the tests
// below are its only callers.
#![allow(dead_code)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub(crate) id: u64,
    pub(crate) generation: u64,
    /// Byte offset in the share file; always 64 KiB aligned.
    pub(crate) offset: u64,
    /// Length in bytes; always a non-zero multiple of 64 KiB.
    pub(crate) length: u64,
}

/// A resolved range that keeps the section, and with it the backing file,
/// alive. It deliberately says nothing about the guest address: the caller
/// must not treat the original pointer as a host-shareable claim.
pub(crate) struct Lease {
    pub(crate) range: Range,
    path: PathBuf,
    /// Keeps the section handle and file open until the host lease ends.
    _section: Arc<Section>,
}

impl Lease {
    /// The Win-visible share file the range lives in.
    pub(crate) fn path(&self) -> &Path {
        &self.path
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

/// Why an existing guest range could not be resolved to a share file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolveError {
    /// Zero length.
    Empty,
    /// Not aligned to 64 KiB pages.
    Unaligned,
    /// `at + len` or the section boundary does not fit.
    OutOfBounds,
    /// Not every page of the range is shown by a tracked view.
    NotMapped,
    /// The views do not show one contiguous run of one section.
    NotContiguous,
    /// The section is pagefile-backed (provider unset or unavailable).
    Pagefile,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            ResolveError::Empty => "the range is empty",
            ResolveError::Unaligned => "the range is not 64 KiB aligned",
            ResolveError::OutOfBounds => "the range leaves its section",
            ResolveError::NotMapped => "the range is not mapped by tracked views",
            ResolveError::NotContiguous => "the range is not one contiguous section run",
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

/// Resolves the guest range `[at, at + len)` to a share file, offset and
/// length.
///
/// The range must be covered by the existing tracked views and show one
/// contiguous run of pages of one section, all within the section. This takes
/// the physical-page lock only to inspect and clone what the views already
/// hold; the returned lease keeps the section (and its file) alive with an
/// `Arc`.
pub(crate) fn resolve_guest_range(at: usize, len: usize) -> Result<Lease, ResolveError> {
    if len == 0 {
        return Err(ResolveError::Empty);
    }
    if !at.is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
        return Err(ResolveError::Unaligned);
    }
    let end = at.checked_add(len).ok_or(ResolveError::OutOfBounds)?;
    let physical = super::physical();
    let mut section: Option<Arc<Section>> = None;
    let mut first_index = None;
    let mut next_index = None;
    let mut cursor = at;
    while cursor < end {
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
        cursor = view_end.min(end);
    }
    let section = section.ok_or(ResolveError::NotMapped)?;
    let offset = first_index
        .ok_or(ResolveError::NotMapped)?
        .checked_mul(PAGE)
        .ok_or(ResolveError::OutOfBounds)?;
    let end_offset = offset.checked_add(len).ok_or(ResolveError::OutOfBounds)?;
    if end_offset > section.size {
        return Err(ResolveError::OutOfBounds);
    }
    let backing = section.share.as_ref().ok_or(ResolveError::Pagefile)?;
    debug_assert_eq!(backing.generation, GENERATION);
    Ok(Lease {
        range: Range {
            id: backing.id,
            generation: backing.generation,
            offset: offset as u64,
            length: len as u64,
        },
        path: backing.path.clone(),
        _section: section,
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
        assert_eq!(
            resolve_guest_range(at as usize + 1, PAGE).unwrap_err(),
            ResolveError::Unaligned
        );
        assert_eq!(
            resolve_guest_range(at as usize, PAGE + 1).unwrap_err(),
            ResolveError::Unaligned
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
}
