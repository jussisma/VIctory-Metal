//! Guest-side mapping of the shared `gpu-fence.v1.page` (ADR-019).
//!
//! The host maps `<share dir>/gpu-fence.v1.page` when the guest registers it
//! (transport kind 6). The guest maps the same 4 KiB file and reads the
//! completed fence word with one `Ordering::Acquire` load: no syscall and no
//! IPC round trip per `GetCompletedValue`.
//!
//! The page path comes from `VICTORY_METAL_FENCE_PAGE`, or from
//! `VICTORY_METAL_SHARE_DIR` joined with the canonical file name. With
//! neither variable set, or with a mapping that cannot be made, the page is
//! genuinely absent and [`GuestFencePage`] is `None`; the transport then
//! keeps its synchronous kind 5 reply as the only completion signal.
//!
//! The Windows guest maps the file with `CreateFileMappingW`/`MapViewOfFile`
//! (the process runs under Wine, whose file APIs serve the shared path the
//! launcher made). macOS builds use [`victory_metal::fence_page`] directly;
//! that path exists for the in-process tests and the Wine-free host.

use std::path::{Path, PathBuf};

/// Exact size of the shared page.
pub const FENCE_PAGE_LEN: u64 = 4096;

/// Environment variable naming the page file explicitly.
pub const FENCE_PAGE_ENV: &str = "VICTORY_METAL_FENCE_PAGE";
/// Environment variable naming the launcher-owned share directory.
pub const SHARE_DIR_ENV: &str = "VICTORY_METAL_SHARE_DIR";
/// Canonical page file name inside the share directory
/// (`victory_metal::ipc_host::FENCE_PAGE_FILE_NAME`).
pub const FENCE_PAGE_FILE_NAME: &str = "gpu-fence.v1.page";

/// The page path the environment names, if any.
pub fn page_path_from_env() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(FENCE_PAGE_ENV).filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    std::env::var_os(SHARE_DIR_ENV)
        .filter(|path| !path.is_empty())
        .map(|dir| PathBuf::from(dir).join(FENCE_PAGE_FILE_NAME))
}

/// A mapped shared completion page.
pub struct GuestFencePage {
    path: PathBuf,
    #[cfg(target_os = "macos")]
    mapping: victory_metal::fence_page::FencePageMapping,
    #[cfg(windows)]
    mapping: windows_mapping::WindowsMapping,
}

impl GuestFencePage {
    /// Opens (or creates) the page the environment names.
    ///
    /// `None` means the page is absent or unusable; a caller must not treat
    /// that as completion and must not invent a value.
    pub fn from_env() -> Option<Self> {
        Self::open(page_path_from_env()?)
    }

    /// Opens (or creates) the page at `path`.
    pub fn open(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        #[cfg(target_os = "macos")]
        {
            let mapping = victory_metal::fence_page::FencePageMapping::open_or_create(path).ok()?;
            Some(Self {
                path: path.to_path_buf(),
                mapping,
            })
        }
        #[cfg(windows)]
        {
            let mapping = windows_mapping::WindowsMapping::open_or_create(path)?;
            Some(Self {
                path: path.to_path_buf(),
                mapping,
            })
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            let _ = path;
            None
        }
    }

    /// Reads the completed fence with one `Ordering::Acquire` load.
    pub fn completed(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            self.mapping.page().completed()
        }
        #[cfg(windows)]
        {
            self.mapping.completed()
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            0
        }
    }

    /// Publishes a completed value (tests and a Wine-free in-process host).
    ///
    /// Uses the same monotonic `fetch_max` the host's `FencePage::publish`
    /// uses, so a stale value can never lower the word.
    pub fn publish(&self, value: u64) -> u64 {
        #[cfg(target_os = "macos")]
        {
            self.mapping.page().publish(value)
        }
        #[cfg(windows)]
        {
            self.mapping.publish(value)
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            let _ = value;
            0
        }
    }

    /// Reads the submitted fence with one `Ordering::Acquire` load.
    pub fn submitted(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            self.mapping.page().submitted()
        }
        #[cfg(windows)]
        {
            self.mapping.submitted()
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            0
        }
    }

    /// The backing path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(windows)]
mod windows_mapping {
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS,
        MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
    };

    use super::FENCE_PAGE_LEN;

    /// One `MAP_SHARED`-equivalent view of the page file.
    ///
    /// The file handle stays open for the lifetime of the view, as on the
    /// host side, so a rename or delete by the launcher does not detach the
    /// mapping from its pages.
    pub struct WindowsMapping {
        at: MEMORY_MAPPED_VIEW_ADDRESS,
        _file: File,
    }

    // SAFETY: the page holds only atomics; sharing the mapping across threads
    // is equivalent to sharing `&AtomicU64`.
    unsafe impl Send for WindowsMapping {}
    // SAFETY: as above.
    unsafe impl Sync for WindowsMapping {}

    impl WindowsMapping {
        /// Opens the page, creating a zero-filled 4 KiB file when absent.
        pub fn open_or_create(path: &Path) -> Option<Self> {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .ok()?;
            let len = file.metadata().ok()?.len();
            if len == 0 {
                file.set_len(FENCE_PAGE_LEN).ok()?;
            } else if len != FENCE_PAGE_LEN {
                return None;
            }
            // SAFETY: the file is open read/write for at least one page.
            let at = unsafe {
                let mapping = CreateFileMappingW(
                    file.as_raw_handle(),
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    0,
                    std::ptr::null(),
                );
                if mapping.is_null() {
                    return None;
                }
                let at = MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
                CloseHandle(mapping);
                at
            };
            if at.Value.is_null() {
                return None;
            }
            Some(Self { at, _file: file })
        }

        /// One `Ordering::Acquire` load of the completed word at offset 0.
        pub fn completed(&self) -> u64 {
            // SAFETY: the view is live, page aligned and 4 KiB long; the
            // word is an `AtomicU64` written by the host with `fetch_max`.
            unsafe { (&*(self.at.Value.cast::<AtomicU64>())).load(Ordering::Acquire) }
        }

        /// One `Ordering::Acquire` load of the submitted word at offset 8.
        pub fn submitted(&self) -> u64 {
            // SAFETY: as above; the word is 8 bytes into the same page.
            unsafe {
                (&*(self.at.Value.cast::<u8>().add(8).cast::<AtomicU64>())).load(Ordering::Acquire)
            }
        }

        /// Publishes a completed value with `fetch_max`.
        pub fn publish(&self, value: u64) -> u64 {
            // SAFETY: as above; the word is the page's `AtomicU64` at 0.
            let word = unsafe { &*(self.at.Value.cast::<AtomicU64>()) };
            word.fetch_max(value, Ordering::AcqRel).max(value)
        }

        /// The session epoch word at offset 16 (diagnostics/tests).
        #[allow(dead_code)]
        pub fn epoch(&self) -> u32 {
            // SAFETY: as above; 16 bytes into the same page.
            unsafe { (&*(self.at.Value.cast::<u8>().add(16).cast::<AtomicU32>())).load(Ordering::Acquire) }
        }
    }

    impl Drop for WindowsMapping {
        fn drop(&mut self) {
            // SAFETY: the view came from `MapViewOfFile` and is dropped once.
            unsafe { UnmapViewOfFile(self.at) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page file created by the guest is exactly one page, and two handles
    /// observe the same published word.
    #[test]
    fn the_guest_maps_the_shared_page_the_host_publishes() {
        let path = std::env::temp_dir().join(format!(
            "victory-gpu-fence-page-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos())
        ));
        let _ = std::fs::remove_file(&path);
        let page = GuestFencePage::open(&path).expect("the guest page");
        assert_eq!(page.completed(), 0);
        assert_eq!(page.submitted(), 0);

        #[cfg(target_os = "macos")]
        {
            // A second mapping is exactly what the host does.
            let host =
                victory_metal::fence_page::FencePageMapping::open(&path).expect("the host page");
            host.page().publish(7);
            assert_eq!(page.completed(), 7, "one atomic load sees the host");
            host.page().publish_submitted(9);
            assert_eq!(page.submitted(), 9);
            // Monotonic: an older value never moves the word back.
            host.page().publish(3);
            assert_eq!(page.completed(), 7);
        }
        let _ = std::fs::remove_file(&path);
    }
}
