//! Shared-memory GPU completion fence page (ADR-012).
//!
//! Reserved by the orchestrator. This module owns the transport-agnostic
//! `AtomicU64` completion fence that an ARM64 Metal host publishes from a
//! command-buffer completion handler and a guest reads with one atomic load
//! (`GetCompletedValue`), with no IPC round-trip and no Wine dependency.
//!
//! The page is one 4 KiB, page-aligned, `#[repr(C)]` record. Two processes map
//! the same file `MAP_SHARED` and communicate only through the atomics:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 8 | `completed_fence`, monotonic, published after GPU completion |
//! | 8 | 8 | `submitted_fence`, monotonic, published when work is enqueued |
//! | 16 | 4 | `epoch`, advanced once per host session |
//! | 20 | 4076 | reserved padding, always zero |
//!
//! Every read is a single `Ordering::Acquire` load, so a guest completion check
//! is a plain load from mapped memory: no syscall, no doorbell, no IPC
//! round-trip. Writes use `fetch_max`, so the published values never move
//! backwards even if a stale or replayed submission is observed.
//!
//! # Wake channels
//!
//! The [`Wake`] trait is the only transport-specific part of a notification.
//! [`LoopbackTcpWake`] and [`UnixSocketWake`] are constructive without any OS
//! asset (a listener does not have to exist); a wake is best-effort, so a host
//! that is already polling loses nothing when it fails.
//!
//! # File mapping
//!
//! [`FencePageMapping`] owns a `MAP_SHARED` mapping of a 4 KiB file. The
//! launcher creates the file in the canonical share directory, the host opens
//! it, and both sides publish/observe through the same page. The mapping is
//! `Drop`-unmapped; no reference to [`FencePageMapping::page`] may outlive the
//! mapping.

use std::io::{self, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

#[cfg(unix)]
use std::path::PathBuf;

/// Exact size in bytes of the shared fence page.
pub const FENCE_PAGE_LEN: usize = 4096;
/// Required alignment in bytes of the shared fence page.
pub const FENCE_PAGE_ALIGN: usize = 4096;
/// Default connect and write timeout of [`LoopbackTcpWake`].
pub const DEFAULT_WAKE_TIMEOUT: Duration = Duration::from_millis(50);

/// Bytes reserved after the three published words.
const RESERVED_LEN: usize = FENCE_PAGE_LEN - 8 - 8 - 4;

/// One page of shared GPU completion state.
///
/// The layout is fixed and size/alignment checked at compile time, so the same
/// binary view can be produced by a guest, a C shim or another process. All
/// fields are atomics, so sharing `&FencePage` across threads and processes is
/// sound.
#[repr(C, align(4096))]
pub struct FencePage {
    completed_fence: AtomicU64,
    submitted_fence: AtomicU64,
    epoch: AtomicU32,
    reserved: [u8; RESERVED_LEN],
}

// Compile-time `static_assert` equivalents: one exact, aligned page and fixed
// field offsets. A layout change is a wire-format change and must fail here.
const _: () = assert!(std::mem::size_of::<FencePage>() == FENCE_PAGE_LEN);
const _: () = assert!(std::mem::align_of::<FencePage>() == FENCE_PAGE_ALIGN);
const _: () = assert!(std::mem::offset_of!(FencePage, completed_fence) == 0);
const _: () = assert!(std::mem::offset_of!(FencePage, submitted_fence) == 8);
const _: () = assert!(std::mem::offset_of!(FencePage, epoch) == 16);

impl FencePage {
    /// Creates a zeroed, offline page. This is a plain allocation, not a
    /// mapping; use [`FencePageMapping`] for a page shared with another
    /// process.
    pub const fn new() -> Self {
        Self {
            completed_fence: AtomicU64::new(0),
            submitted_fence: AtomicU64::new(0),
            epoch: AtomicU32::new(0),
            reserved: [0u8; RESERVED_LEN],
        }
    }

    /// Starts a new host session: clears both fences and advances the epoch.
    ///
    /// The caller must guarantee that no peer is submitting or observing the
    /// page while this runs. Session registration is the only intended call
    /// site, before the registration acknowledgement is written.
    pub fn begin_session(&self) {
        self.completed_fence.store(0, Ordering::Release);
        self.submitted_fence.store(0, Ordering::Release);
        self.advance_epoch();
    }

    /// Publishes a completed fence value monotonically.
    ///
    /// Returns the value now visible in [`FencePage::completed`], which is the
    /// larger of `value` and the previously published value.
    pub fn publish(&self, value: u64) -> u64 {
        let previous = self.completed_fence.fetch_max(value, Ordering::AcqRel);
        previous.max(value)
    }

    /// Publishes a submitted fence value monotonically.
    ///
    /// Returns the value now visible in [`FencePage::submitted`].
    pub fn publish_submitted(&self, value: u64) -> u64 {
        let previous = self.submitted_fence.fetch_max(value, Ordering::AcqRel);
        previous.max(value)
    }

    /// Reads the completed fence with one `Ordering::Acquire` load.
    ///
    /// This is the guest-side fast path: no syscall, no IPC round-trip.
    pub fn completed(&self) -> u64 {
        self.completed_fence.load(Ordering::Acquire)
    }

    /// Reads the submitted fence with one `Ordering::Acquire` load.
    pub fn submitted(&self) -> u64 {
        self.submitted_fence.load(Ordering::Acquire)
    }

    /// True when submitted work has not completed yet.
    ///
    /// Both loads are `Ordering::Acquire`; the two loads are not a single
    /// snapshot, so a concurrent publish may make this transiently stale in
    /// either direction. It is a hint, never a correctness barrier.
    pub fn now_pending(&self) -> bool {
        self.submitted_fence.load(Ordering::Acquire) > self.completed_fence.load(Ordering::Acquire)
    }

    /// Advances the session epoch and returns the new value.
    pub fn advance_epoch(&self) -> u32 {
        self.epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    /// Reads the session epoch with one `Ordering::Acquire` load.
    pub fn epoch(&self) -> u32 {
        self.epoch.load(Ordering::Acquire)
    }
}

impl Default for FencePage {
    fn default() -> Self {
        Self::new()
    }
}

/// Host-visible wake mechanism, independent of the IPC transport.
///
/// A renderer holds the wake object but never needs to know whether the host
/// waits on a TCP port, a Unix socket, or (in a future Wine-free host) a Mach
/// port. Wake is a hint: failure must not fail the submission.
pub trait Wake: Send + Sync {
    /// Delivers one wake byte. Best-effort; a missing listener is an error.
    fn wake(&self) -> io::Result<()>;

    /// Delivers one wake byte, ignoring any failure.
    fn wake_best_effort(&self) {
        let _ = self.wake();
    }
}

/// Wake hint over a loopback TCP listener.
///
/// `new(port)` is constructive without a listener; the connect and write both
/// use [`DEFAULT_WAKE_TIMEOUT`] so a dead host cannot stall a submitter.
#[derive(Debug, Clone)]
pub struct LoopbackTcpWake {
    addr: SocketAddr,
    timeout: Duration,
}

impl LoopbackTcpWake {
    /// Targets `127.0.0.1:port` with the default short timeout.
    pub fn new(port: u16) -> Self {
        Self::with_timeout(
            SocketAddr::from(([127, 0, 0, 1], port)),
            DEFAULT_WAKE_TIMEOUT,
        )
    }

    /// Targets an explicit address and timeout.
    pub fn with_timeout(addr: SocketAddr, timeout: Duration) -> Self {
        Self { addr, timeout }
    }

    /// The address this wake connects to.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Wake for LoopbackTcpWake {
    fn wake(&self) -> io::Result<()> {
        let mut stream = TcpStream::connect_timeout(&self.addr, self.timeout)?;
        stream.set_write_timeout(Some(self.timeout))?;
        stream.write_all(&[1])
    }
}

/// Wake hint over a Unix domain socket.
///
/// On non-Unix targets the type still exists and constructs, but [`Wake::wake`]
/// returns an explicit [`io::ErrorKind::Unsupported`] error instead of silently
/// pretending the notification was delivered.
#[cfg(unix)]
#[derive(Debug, Clone)]
pub struct UnixSocketWake {
    path: PathBuf,
    timeout: Duration,
}

#[cfg(unix)]
impl UnixSocketWake {
    /// Targets a socket path. Construction does not touch the filesystem.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::with_timeout(path, DEFAULT_WAKE_TIMEOUT)
    }

    /// Targets a socket path with an explicit timeout.
    pub fn with_timeout(path: impl AsRef<Path>, timeout: Duration) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            timeout,
        }
    }

    /// The socket path this wake connects to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(unix)]
impl Wake for UnixSocketWake {
    fn wake(&self) -> io::Result<()> {
        use std::os::unix::net::UnixStream;
        let mut stream = UnixStream::connect(&self.path)?;
        stream.set_write_timeout(Some(self.timeout))?;
        stream.write_all(&[1])
    }
}

/// Non-Unix `UnixSocketWake`: constructible, but every wake is `Unsupported`.
#[cfg(not(unix))]
#[derive(Debug, Clone, Copy, Default)]
pub struct UnixSocketWake {
    _private: (),
}

#[cfg(not(unix))]
impl UnixSocketWake {
    /// Retains the cross-platform constructor signature; wakes always fail.
    pub fn new<T: AsRef<Path>>(_path: T) -> Self {
        Self { _private: () }
    }

    /// Retains the cross-platform constructor signature; wakes always fail.
    pub fn with_timeout<T: AsRef<Path>>(_path: T, _timeout: Duration) -> Self {
        Self { _private: () }
    }
}

#[cfg(not(unix))]
impl Wake for UnixSocketWake {
    fn wake(&self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "unix domain sockets are not supported on this platform",
        ))
    }
}

#[cfg(target_os = "macos")]
mod mapping {
    use std::ffi::c_void;
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::{Path, PathBuf};

    use super::{FENCE_PAGE_ALIGN, FENCE_PAGE_LEN, FencePage};

    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    /// Darwin `MAP_SHARED`: both mappings and the file share one page cache.
    const MAP_SHARED: i32 = 0x0001;
    /// Darwin `O_NOFOLLOW`: reject a final path component that is a symlink,
    /// matching the allocation registration hardening in `ipc_host`.
    const O_NOFOLLOW: i32 = 0x0100;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    /// A `MAP_SHARED` mapping of one 4 KiB fence page file.
    ///
    /// Both the host and the guest open the same path and map it read/write.
    /// The mapping keeps its file descriptor open so the page stays alive; the
    /// `Drop` implementation unmaps exactly [`FENCE_PAGE_LEN`] bytes.
    pub struct FencePageMapping {
        _file: File,
        ptr: *mut u8,
        path: PathBuf,
    }

    // SAFETY: the mapped region contains only atomics; sharing the handle
    // across threads is equivalent to sharing `&FencePage`.
    unsafe impl Send for FencePageMapping {}
    // SAFETY: see above; every access to the page is an atomic operation.
    unsafe impl Sync for FencePageMapping {}

    impl FencePageMapping {
        /// Creates a new fence page file, failing if the path already exists.
        ///
        /// A fresh file is zero-filled by the filesystem, so the new page reads
        /// `completed = submitted = epoch = 0`.
        pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
            let path = path.as_ref();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .custom_flags(O_NOFOLLOW)
                .open(path)?;
            file.set_len(FENCE_PAGE_LEN as u64)?;
            Self::map(file, path.to_path_buf())
        }

        /// Opens an existing fence page file and maps it `MAP_SHARED`.
        ///
        /// The final path component must be a regular, non-symlink file of
        /// exactly one page.
        pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
            let path = path.as_ref();
            let before = std::fs::symlink_metadata(path)?;
            if !before.file_type().is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "fence page backing must be a regular file",
                ));
            }
            if before.len() != FENCE_PAGE_LEN as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "fence page backing must be exactly 4096 bytes",
                ));
            }
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(O_NOFOLLOW)
                .open(path)?;
            // Recheck on the opened object so a path swap cannot race the
            // metadata check above.
            let metadata = file.metadata()?;
            if !metadata.is_file() || metadata.len() != FENCE_PAGE_LEN as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "fence page backing changed while opening",
                ));
            }
            Self::map(file, path.to_path_buf())
        }

        /// Opens an existing page, or creates it when the path is absent.
        ///
        /// Intended for a launcher that owns the share directory. An existing
        /// file is never truncated: its current fence values stay visible.
        pub fn open_or_create(path: impl AsRef<Path>) -> io::Result<Self> {
            match Self::open(path.as_ref()) {
                Ok(mapping) => Ok(mapping),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Self::create(path),
                Err(error) => Err(error),
            }
        }

        /// Borrows the shared page.
        pub fn page(&self) -> &FencePage {
            // SAFETY: `ptr` is the live, page-aligned, 4096-byte `MAP_SHARED`
            // mapping created in `map`; `FencePage` is exactly one page of
            // atomics and padding, so every byte is initialized and no other
            // reference aliases it in this process.
            unsafe { &*(self.ptr as *const FencePage) }
        }

        /// The path this mapping was created from.
        pub fn path(&self) -> &Path {
            &self.path
        }

        fn map(file: File, path: PathBuf) -> io::Result<Self> {
            // SAFETY: `file` is open read/write and at least `FENCE_PAGE_LEN`
            // bytes long; the mapping is owned by the returned value and
            // unmapped in `Drop`.
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    FENCE_PAGE_LEN,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            if !(ptr as usize).is_multiple_of(FENCE_PAGE_ALIGN) {
                // SAFETY: `ptr` is the live mapping returned by `mmap` above.
                unsafe { munmap(ptr, FENCE_PAGE_LEN) };
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "fence page mapping is not page aligned",
                ));
            }
            Ok(Self {
                _file: file,
                ptr: ptr as *mut u8,
                path,
            })
        }
    }

    impl Drop for FencePageMapping {
        fn drop(&mut self) {
            // SAFETY: `ptr` and `FENCE_PAGE_LEN` describe the mapping from
            // `map`; Drop runs once and the file descriptor is still open.
            unsafe { munmap(self.ptr as *mut c_void, FENCE_PAGE_LEN) };
        }
    }
}

#[cfg(target_os = "macos")]
pub use mapping::FencePageMapping;

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[test]
    fn layout_is_one_exact_page_with_fixed_word_offsets() {
        assert_eq!(std::mem::size_of::<FencePage>(), FENCE_PAGE_LEN);
        assert_eq!(std::mem::align_of::<FencePage>(), FENCE_PAGE_ALIGN);
        assert_eq!(std::mem::offset_of!(FencePage, completed_fence), 0);
        assert_eq!(std::mem::offset_of!(FencePage, submitted_fence), 8);
        assert_eq!(std::mem::offset_of!(FencePage, epoch), 16);
    }

    #[test]
    fn default_page_starts_idle() {
        let page = FencePage::default();
        assert_eq!(page.completed(), 0);
        assert_eq!(page.submitted(), 0);
        assert_eq!(page.epoch(), 0);
        assert!(!page.now_pending());
    }

    #[test]
    fn publish_is_monotonic_and_now_pending_tracks_progress() {
        let page = FencePage::new();
        assert_eq!(page.publish_submitted(3), 3);
        assert!(page.now_pending(), "submitted work must read as pending");

        assert_eq!(page.publish(2), 2);
        assert!(page.now_pending(), "2 < 3 still leaves work pending");

        assert_eq!(page.publish(5), 5);
        assert!(!page.now_pending());

        // A stale or replayed value never lowers the published fence.
        assert_eq!(page.publish(3), 5);
        assert_eq!(page.publish_submitted(1), 3);
        assert_eq!(page.completed(), 5);
        assert_eq!(page.submitted(), 3);
    }

    #[test]
    fn epoch_advances_and_session_start_clears_fences() {
        let page = FencePage::new();
        assert_eq!(page.advance_epoch(), 1);
        assert_eq!(page.advance_epoch(), 2);
        assert_eq!(page.epoch(), 2);

        page.publish(9);
        page.publish_submitted(10);
        assert!(page.now_pending());
        page.begin_session();
        assert_eq!(page.completed(), 0);
        assert_eq!(page.submitted(), 0);
        assert_eq!(page.epoch(), 3);
        assert!(!page.now_pending());
    }

    #[cfg(target_os = "macos")]
    struct TempPage(std::path::PathBuf);

    #[cfg(target_os = "macos")]
    impl TempPage {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "victory-fence-page-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_file(&path);
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for TempPage {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn file_mapping_is_shared_between_two_handles() {
        let temp = TempPage::new();
        let first = FencePageMapping::create(temp.path()).expect("create fence page");
        assert_eq!(first.path(), temp.path());
        assert_eq!(first.page().completed(), 0);

        // A second mapping is exactly what the guest process performs.
        let second = FencePageMapping::open(temp.path()).expect("open shared fence page");
        assert_ne!(
            first.page() as *const FencePage,
            second.page() as *const FencePage,
            "the two handles must be independent virtual mappings"
        );

        first.page().begin_session();
        assert_eq!(second.page().epoch(), 1);

        assert_eq!(first.page().publish(7), 7);
        assert_eq!(
            second.page().completed(),
            7,
            "second mapping missed publish"
        );

        assert_eq!(second.page().publish_submitted(8), 8);
        assert!(first.page().now_pending());

        assert_eq!(second.page().publish(8), 8);
        assert!(!first.page().now_pending());
        assert_eq!(first.page().completed(), 8);

        // The backing file is the same one both handles mapped.
        assert_eq!(
            std::fs::metadata(temp.path())
                .expect("fence page metadata")
                .len(),
            FENCE_PAGE_LEN as u64
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn file_mapping_rejects_wrong_size_and_duplicate_create() {
        let temp = TempPage::new();
        let mapping = FencePageMapping::create(temp.path()).expect("create fence page");
        assert!(
            FencePageMapping::create(temp.path()).is_err(),
            "creating over an existing page must fail"
        );
        drop(mapping);

        std::fs::write(temp.path(), [0u8; 32]).expect("write short page file");
        assert!(
            FencePageMapping::open(temp.path()).is_err(),
            "opening a short page file must fail"
        );
    }

    #[test]
    fn loopback_tcp_wake_is_constructive_without_a_listener() {
        // Construction never touches the network; the wake itself fails fast.
        let wake = LoopbackTcpWake::new(9);
        assert_eq!(wake.addr().port(), 9);
        wake.wake_best_effort();
    }

    #[test]
    fn unix_socket_wake_is_constructive_without_a_socket() {
        let wake = UnixSocketWake::new("/nonexistent/victory-wake.sock");
        wake.wake_best_effort();
    }
}
