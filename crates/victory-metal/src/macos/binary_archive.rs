//! Persistent `MTLBinaryArchive` pipeline cache (ADR-014).
//!
//! `MTLBinaryArchive` keeps compiled pipeline state (including the driver's
//! shader compilation) in a file that can be reopened by a later process. A
//! warm start that finds the archive can therefore create the pipeline without
//! invoking the Metal compiler again.
//!
//! This module owns:
//!
//! - creation/loading through `-[MTLDevice newBinaryArchiveWithDescriptor:error:]`,
//!   guarded by `respondsToSelector:` so an old runtime reports
//!   [`ArchiveError::Unsupported`] instead of crashing;
//! - `addRenderPipelineFunctionsWithDescriptor:error:` and
//!   `addComputePipelineFunctionsWithDescriptor:error:`;
//! - atomic serialization (`serializeToURL:error:` into a temporary file, then
//!   `rename(2)`), so a crash never leaves a half-written archive;
//! - content-addressed identity through [`archive_content_key`]
//!   (`BLAKE3(metallib ‖ entry ‖ state)`), so a corrupt, unreadable or
//!   mismatched archive is replaced by an empty one rather than failing the
//!   device.
//!
//! The archive is *not* a translation cache: it stores Metal-side pipeline
//! state. [`crate::compiler_pool`] and [`crate::macos::cache`] store the
//! DXIL-to-metallib translation. A caller normally uses both: the disk cache
//! yields the metallib, the archive key names the pipeline built from it.
//!
//! ```ignore
//! let key = BinaryArchive::path_for_key(dir, &content_key(&metallib, entry, state));
//! let archive = BinaryArchive::open_or_create(&device, key)?;
//! // build the descriptor with binaryArchives = @[archive], add it, serialize
//! ```
//!
//! # Reuse contract
//!
//! A bundled archive only takes effect when the pipeline descriptor's
//! `binaryArchives` property includes it. `open_or_create` does not mutate any
//! descriptor; callers own that wiring and should pass the same archive to
//! `-[MTLDevice newComputePipelineStateWithDescriptor:error:]` or
//! `newRenderPipelineStateWithDescriptor:error:`.

use std::ffi::c_void;
use std::fs;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::macos::cache::{archive_content_key, hash_hex};
use crate::macos::device::DirectMetalTransport;
use crate::macos::objc::{self, AutoreleasePool, ObjcObject};

/// File extension of a content-keyed binary archive.
pub const ARCHIVE_EXTENSION: &str = "mtlarchive";

/// Largest archive this module will load or write (1 GiB).
///
/// A bound keeps a corrupt or hostile file from being handed to Metal without
/// limit. Real archives are tens of MiB for large games.
pub const MAX_ARCHIVE_BYTES: u64 = 1024 * 1024 * 1024;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Explicit binary archive failure. Every variant is recoverable: a caller can
/// delete the archive file and retry [`BinaryArchive::open_or_create`].
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The runtime or device does not implement `newBinaryArchiveWithDescriptor:error:`.
    #[error("MTLBinaryArchive is not supported on this device/runtime")]
    Unsupported,
    /// The archive could not be created or loaded.
    #[error("failed to create MTLBinaryArchive: {0}")]
    Create(String),
    /// The archive could not be serialized.
    #[error("failed to serialize MTLBinaryArchive: {0}")]
    Serialize(String),
    /// Adding a pipeline to the archive failed.
    #[error("failed to add a {stage} pipeline to MTLBinaryArchive: {message}")]
    AddPipeline {
        /// Pipeline family (`compute` or `render`).
        stage: &'static str,
        /// Metal diagnostic.
        message: String,
    },
    /// A path was not valid UTF-8 for `NSURL`.
    #[error("archive path is not valid UTF-8: {0}")]
    InvalidPath(PathBuf),
    /// The archive file exceeds [`MAX_ARCHIVE_BYTES`].
    #[error("archive {path} is {bytes} bytes; limit is {limit}")]
    TooLarge {
        /// Archive path.
        path: PathBuf,
        /// Observed size.
        bytes: u64,
        /// Configured limit.
        limit: u64,
    },
    /// Filesystem failure.
    #[error("MTLBinaryArchive filesystem error: {0}")]
    Io(#[from] std::io::Error),
}

/// A loaded or newly created `MTLBinaryArchive`.
///
/// `BinaryArchive` owns one retained archive object. Dropping it releases the
/// Objective-C object but never deletes the file.
pub struct BinaryArchive {
    archive: ObjcObject,
    path: PathBuf,
    key: Option<[u8; 32]>,
}

impl std::fmt::Debug for BinaryArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryArchive")
            .field("path", &self.path)
            .field("key", &self.key.as_ref().map(hash_hex))
            .finish()
    }
}

impl BinaryArchive {
    /// Whether `device` implements the binary archive entry point.
    ///
    /// This is the `respondsToSelector:` probe the rest of the module is
    /// guarded by; it never creates an archive or touches the GPU.
    pub fn is_supported(device: &DirectMetalTransport) -> bool {
        // The concrete `MTLBinaryArchive` class is not always registered under
        // its public name (macOS may vend the instances privately), so the
        // device's `respondsToSelector:` answer, not `objc_getClass`, is the
        // authoritative probe. The descriptor class is required to build one.
        if objc::class(c"MTLBinaryArchiveDescriptor").is_null() {
            return false;
        }
        // SAFETY: `device_raw` is a live `MTLDevice` and the selector is a
        // valid registered selector for `respondsToSelector:`.
        unsafe {
            objc::msg_send_responds_to(
                device.device_raw(),
                objc::sel(c"newBinaryArchiveWithDescriptor:error:"),
            )
        }
    }

    /// Opens the archive at `path`, or creates an empty archive when the file
    /// is absent, empty, corrupt or unreadable.
    ///
    /// A file that Metal refuses to load is deleted and replaced by a fresh
    /// archive, so stale state after a driver or GPU change degrades to a cold
    /// compile instead of a device failure.
    pub fn open_or_create(
        device: &DirectMetalTransport,
        path: impl AsRef<Path>,
    ) -> Result<Self, ArchiveError> {
        let path = path.as_ref().to_path_buf();
        if !Self::is_supported(device) {
            return Err(ArchiveError::Unsupported);
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let _pool = AutoreleasePool::new();
        let device_raw = device.device_raw();

        if let Ok(metadata) = fs::metadata(&path) {
            if metadata.is_file() && metadata.len() > MAX_ARCHIVE_BYTES {
                return Err(ArchiveError::TooLarge {
                    path,
                    bytes: metadata.len(),
                    limit: MAX_ARCHIVE_BYTES,
                });
            }
            match open_at(device_raw, &path) {
                Ok(archive) => {
                    return Ok(Self {
                        archive,
                        path,
                        key: None,
                    });
                }
                Err(message) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %message,
                        "discarding unreadable MTLBinaryArchive and starting a new one"
                    );
                    match fs::remove_file(&path) {
                        Ok(()) => {}
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                        Err(err) => return Err(ArchiveError::Io(err)),
                    }
                }
            }
        }

        let archive = new_empty(device_raw).map_err(ArchiveError::Create)?;
        Ok(Self {
            archive,
            path,
            key: None,
        })
    }

    /// Opens or creates the archive whose path is derived from
    /// `BLAKE3(metallib ‖ entry ‖ state)` under `dir`.
    pub fn open_or_create_keyed(
        device: &DirectMetalTransport,
        dir: impl AsRef<Path>,
        metallib: &[u8],
        entry: &str,
        state: &[u8],
    ) -> Result<Self, ArchiveError> {
        let key = archive_content_key(metallib, entry, state);
        let path = path_for_key(dir.as_ref(), &key);
        let mut archive = Self::open_or_create(device, path)?;
        archive.key = Some(key);
        Ok(archive)
    }

    /// Content-keyed archive path `dir/<hex key>.mtlarchive`.
    pub fn path_for_key(dir: &Path, key: &[u8; 32]) -> PathBuf {
        path_for_key(dir, key)
    }

    /// On-disk archive path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Content key the archive was opened with, when it was keyed.
    pub fn key(&self) -> Option<[u8; 32]> {
        self.key
    }

    /// Adds a render pipeline descriptor to the archive.
    ///
    /// # Safety
    ///
    /// `descriptor` must be a live `MTLRenderPipelineDescriptor *`. The caller
    /// must keep it alive for the duration of the call and must not pass a
    /// descriptor from another process.
    pub unsafe fn add_pipeline(&self, descriptor: *mut c_void) -> Result<(), ArchiveError> {
        if descriptor.is_null() {
            return Err(ArchiveError::AddPipeline {
                stage: "render",
                message: "descriptor is null".to_string(),
            });
        }
        let _pool = AutoreleasePool::new();
        let mut ns_err = objc::empty_id();
        // SAFETY: Caller guarantees a live render pipeline descriptor; the
        // selector is `addRenderPipelineFunctionsWithDescriptor:error:`.
        let ok = unsafe {
            msg_send_bool_err(
                self.archive.as_raw(),
                objc::sel(c"addRenderPipelineFunctionsWithDescriptor:error:"),
                descriptor.cast(),
                &mut ns_err,
            )
        };
        if ok == 0 {
            // SAFETY: Metal populated `ns_err` on failure.
            let message = unsafe { objc::nserror_message(ns_err) };
            return Err(ArchiveError::AddPipeline {
                stage: "render",
                message,
            });
        }
        Ok(())
    }

    /// Adds a compute pipeline descriptor to the archive.
    ///
    /// The descriptor should carry `binaryArchives = @[self]` so Metal records
    /// the functions against this archive; that wiring is the caller's (it is
    /// the same archive object that is later serialized).
    ///
    /// # Safety
    ///
    /// `descriptor` must be a live `MTLComputePipelineDescriptor *`.
    pub unsafe fn add_compute_pipeline(&self, descriptor: *mut c_void) -> Result<(), ArchiveError> {
        if descriptor.is_null() {
            return Err(ArchiveError::AddPipeline {
                stage: "compute",
                message: "descriptor is null".to_string(),
            });
        }
        let _pool = AutoreleasePool::new();
        let mut ns_err = objc::empty_id();
        // SAFETY: Caller guarantees a live compute pipeline descriptor; the
        // selector is `addComputePipelineFunctionsWithDescriptor:error:`.
        let ok = unsafe {
            msg_send_bool_err(
                self.archive.as_raw(),
                objc::sel(c"addComputePipelineFunctionsWithDescriptor:error:"),
                descriptor.cast(),
                &mut ns_err,
            )
        };
        if ok == 0 {
            // SAFETY: Metal populated `ns_err` on failure.
            let message = unsafe { objc::nserror_message(ns_err) };
            return Err(ArchiveError::AddPipeline {
                stage: "compute",
                message,
            });
        }
        Ok(())
    }

    /// Serializes the archive to `self.path` atomically.
    ///
    /// Metal writes a same-directory temporary file, which is then renamed
    /// over the destination, so an interrupted run leaves either the old
    /// archive or none, never a truncated one.
    pub fn serialize(&self) -> Result<(), ArchiveError> {
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let _pool = AutoreleasePool::new();
        let tmp = temp_path(&self.path);
        let _ = fs::remove_file(&tmp);
        let url = file_url(&tmp)?;
        let mut ns_err = objc::empty_id();
        // SAFETY: `self.archive` is a live MTLBinaryArchive and `url` is an
        // autoreleased NSURL naming a writable same-directory path.
        let ok = unsafe {
            msg_send_bool_err(
                self.archive.as_raw(),
                objc::sel(c"serializeToURL:error:"),
                url,
                &mut ns_err,
            )
        };
        if ok == 0 {
            let _ = fs::remove_file(&tmp);
            // SAFETY: Metal populated `ns_err` on failure.
            let message = unsafe { objc::nserror_message(ns_err) };
            return Err(ArchiveError::Serialize(message));
        }
        if let Err(err) = fs::rename(&tmp, &self.path) {
            let _ = fs::remove_file(&tmp);
            return Err(ArchiveError::Io(err));
        }
        // Best effort: make the rename durable across a crash. A read-only
        // filesystem still leaves a valid archive in place.
        if let Ok(file) = fs::File::open(&self.path) {
            let _ = file.sync_all();
        }
        Ok(())
    }
}

/// Content-addressed archive path under `dir`.
pub fn path_for_key(dir: &Path, key: &[u8; 32]) -> PathBuf {
    dir.join(format!("{}.{ARCHIVE_EXTENSION}", hash_hex(key)))
}

fn temp_path(path: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("archive");
    path.with_file_name(format!(".{stem}.{}.{}.tmp", std::process::id(), sequence))
}

fn file_url(path: &Path) -> Result<objc::Id, ArchiveError> {
    let text = path
        .to_str()
        .ok_or_else(|| ArchiveError::InvalidPath(path.to_path_buf()))?;
    let ns_path =
        objc::nsstring(text).ok_or_else(|| ArchiveError::InvalidPath(path.to_path_buf()))?;
    let ns_url_cls = objc::class(c"NSURL");
    if ns_url_cls.is_null() {
        return Err(ArchiveError::Unsupported);
    }
    // SAFETY: `ns_path` is a live NSString and `fileURLWithPath:` is the
    // documented factory; the returned NSURL is autoreleased and outlives the
    // call because every caller holds an `AutoreleasePool`.
    let url = unsafe {
        objc::msg_send_id_1(ns_url_cls, objc::sel(c"fileURLWithPath:"), ns_path.as_raw())
    };
    if url.is_null() {
        return Err(ArchiveError::InvalidPath(path.to_path_buf()));
    }
    Ok(url)
}

fn descriptor_with_url(url: Option<objc::Id>) -> Result<ObjcObject, String> {
    let cls = objc::class(c"MTLBinaryArchiveDescriptor");
    if cls.is_null() {
        return Err("MTLBinaryArchiveDescriptor is unavailable".to_string());
    }
    // SAFETY: `+new` on the descriptor class returns a retained instance.
    let raw = unsafe { objc::msg_send_id_0(cls, objc::sel(c"new")) };
    let descriptor =
        ObjcObject::from_retained(raw).ok_or_else(|| "descriptor allocation failed".to_string())?;
    if let Some(url) = url {
        // SAFETY: `descriptor` is a live MTLBinaryArchiveDescriptor and `url`
        // is a live NSURL.
        unsafe {
            objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setUrl:"), url);
        }
    }
    Ok(descriptor)
}

fn new_archive_with_descriptor(
    device: objc::Id,
    descriptor: &ObjcObject,
) -> Result<ObjcObject, String> {
    let mut ns_err = objc::empty_id();
    // SAFETY: `device` is a live MTLDevice, `descriptor` a live archive
    // descriptor and `ns_err` receives an autoreleased NSError.
    let raw = unsafe {
        objc::msg_send_id_err(
            device,
            objc::sel(c"newBinaryArchiveWithDescriptor:error:"),
            descriptor.as_raw(),
            &mut ns_err,
        )
    };
    ObjcObject::from_retained(raw).ok_or_else(|| {
        // SAFETY: Metal populated `ns_err` when the archive could not be
        // created or loaded.
        unsafe { objc::nserror_message(ns_err) }
    })
}

fn new_empty(device: objc::Id) -> Result<ObjcObject, String> {
    let descriptor = descriptor_with_url(None)?;
    new_archive_with_descriptor(device, &descriptor)
}

fn open_at(device: objc::Id, path: &Path) -> Result<ObjcObject, String> {
    let url = file_url(path).map_err(|err| err.to_string())?;
    let descriptor = descriptor_with_url(Some(url))?;
    new_archive_with_descriptor(device, &descriptor)
}

/// `BOOL`-returning selector with one object argument and `NSError **`.
///
/// # Safety
///
/// `target`, `argument` and `error` must match the selector's Objective-C
/// signature; the selector must take exactly one object plus `NSError **` and
/// return `BOOL`.
unsafe fn msg_send_bool_err(
    target: objc::Id,
    selector: objc::Sel,
    argument: objc::Id,
    error: *mut objc::Id,
) -> u8 {
    type Fn = unsafe extern "C" fn(objc::Id, objc::Sel, objc::Id, *mut objc::Id) -> u8;
    // SAFETY: Caller guarantees the signature; `msg_send_entry` is the shared
    // `objc_msgSend` dispatch entry point.
    unsafe {
        let f: Fn = mem::transmute(objc::msg_send_entry());
        f(target, selector, argument, error)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::macos::cache::{
        Cached, DiskCache, ReflectionPayload, TranslationKey, TranslationOptions,
    };

    const KERNEL_ENTRY: &str = "vm_archive_scale";
    const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vm_archive_scale(device uint *data [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 3u + 7u;
}
"#;

    /// `/tmp/victory-gpu.lock` mutual exclusion, matching the repository rule
    /// that GPU and archive steps are serialized across sessions.
    struct GpuLock {
        path: PathBuf,
        held: bool,
    }

    impl GpuLock {
        fn acquire() -> Self {
            let path = PathBuf::from("/tmp/victory-gpu.lock");
            let deadline = Instant::now() + Duration::from_secs(300);
            loop {
                match std::fs::create_dir(&path) {
                    Ok(()) => {
                        return Self { path, held: true };
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                        assert!(
                            Instant::now() < deadline,
                            "timed out waiting for /tmp/victory-gpu.lock"
                        );
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    Err(err) => panic!("cannot create /tmp/victory-gpu.lock: {err}"),
                }
            }
        }
    }

    impl Drop for GpuLock {
        fn drop(&mut self) {
            if self.held {
                let _ = std::fs::remove_dir(&self.path);
            }
        }
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "victory-metal-archive-{label}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create archive test dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Generated, asset-free metallib via the installed Metal toolchain.
    fn build_metallib(dir: &Path) -> Option<Vec<u8>> {
        let source = dir.join("archive_test.metal");
        let air = dir.join("archive_test.air");
        let metallib = dir.join("archive_test.metallib");
        std::fs::write(&source, KERNEL_SOURCE).ok()?;
        let compile = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .output()
            .ok()?;
        if !compile.status.success() {
            println!(
                "SKIP: xcrun metal failed: {}",
                String::from_utf8_lossy(&compile.stderr).trim()
            );
            return None;
        }
        let link = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&air)
            .arg("-o")
            .arg(&metallib)
            .output()
            .ok()?;
        if !link.status.success() {
            println!(
                "SKIP: xcrun metallib failed: {}",
                String::from_utf8_lossy(&link.stderr).trim()
            );
            return None;
        }
        std::fs::read(&metallib).ok()
    }

    /// Prints the exact selector/class probes behind [`BinaryArchive::is_supported`].
    fn print_selector_probe(device: &DirectMetalTransport) {
        let responds = |selector: &std::ffi::CStr| -> bool {
            // SAFETY: `device` is a live MTLDevice and `respondsToSelector:`
            // accepts any registered selector.
            unsafe { objc::msg_send_responds_to(device.device_raw(), objc::sel(selector)) }
        };
        println!(
            "probe: MTLBinaryArchive class={} MTLBinaryArchiveDescriptor class={} \
             respondsToSelector(newBinaryArchiveWithDescriptor:error:)={} \
             respondsToSelector(newLibraryWithData:error:)={} \
             respondsToSelector(newComputePipelineStateWithDescriptor:error:)={}",
            !objc::class(c"MTLBinaryArchive").is_null(),
            !objc::class(c"MTLBinaryArchiveDescriptor").is_null(),
            responds(c"newBinaryArchiveWithDescriptor:error:"),
            responds(c"newLibraryWithData:error:"),
            responds(c"newComputePipelineStateWithDescriptor:error:"),
        );
    }

    fn open_device() -> Option<DirectMetalTransport> {
        match DirectMetalTransport::new() {
            Ok(device) => Some(device),
            Err(err) => {
                println!("SKIP: no Metal device: {err}");
                None
            }
        }
    }

    struct Library {
        library: ObjcObject,
    }

    impl Library {
        fn load(device: &DirectMetalTransport, metallib: &[u8]) -> Result<Self, String> {
            let _pool = AutoreleasePool::new();
            let data = objc::dispatch_data_copy(metallib)
                .ok_or_else(|| "dispatch_data_create returned null".to_string())?;
            let mut ns_err = objc::empty_id();
            // SAFETY: Device and dispatch_data_t are live; ns_err is writable.
            let raw = unsafe {
                objc::msg_send_id_err(
                    device.device_raw(),
                    objc::sel(c"newLibraryWithData:error:"),
                    data.as_raw(),
                    &mut ns_err,
                )
            };
            ObjcObject::from_retained(raw)
                .map(|library| Self { library })
                .ok_or_else(|| unsafe { objc::nserror_message(ns_err) })
        }

        fn function(&self, entry: &str) -> Result<ObjcObject, String> {
            let _pool = AutoreleasePool::new();
            let name =
                objc::nsstring(entry).ok_or_else(|| "NSString allocation failed".to_string())?;
            // SAFETY: `library` is a live MTLLibrary and `name` a live NSString.
            let raw = unsafe {
                objc::msg_send_id_1(
                    self.library.as_raw(),
                    objc::sel(c"newFunctionWithName:"),
                    name.as_raw(),
                )
            };
            ObjcObject::from_retained(raw)
                .ok_or_else(|| format!("function `{entry}` not found in MTLLibrary"))
        }
    }

    /// Builds `MTLComputePipelineDescriptor` with `binaryArchives = @[archive]`.
    fn compute_descriptor(
        function: &ObjcObject,
        archive: Option<&BinaryArchive>,
    ) -> Result<ObjcObject, String> {
        let class = objc::class(c"MTLComputePipelineDescriptor");
        if class.is_null() {
            return Err("MTLComputePipelineDescriptor is unavailable".to_string());
        }
        // SAFETY: `+new` returns a retained descriptor instance.
        let raw = unsafe { objc::msg_send_id_0(class, objc::sel(c"new")) };
        let descriptor =
            ObjcObject::from_retained(raw).ok_or_else(|| "descriptor allocation".to_string())?;
        // SAFETY: `descriptor` is live and `function` is a live MTLFunction.
        unsafe {
            objc::msg_send_void_1(
                descriptor.as_raw(),
                objc::sel(c"setComputeFunction:"),
                function.as_raw(),
            );
        }

        let Some(archive) = archive else {
            return Ok(descriptor);
        };
        let array_class = objc::class(c"NSArray");
        if array_class.is_null() {
            return Err("NSArray is unavailable".to_string());
        }
        // SAFETY: `archive` is live; `arrayWithObject:` returns an autoreleased
        // NSArray that the descriptor retains.
        let array = unsafe {
            objc::msg_send_id_1(
                array_class,
                objc::sel(c"arrayWithObject:"),
                archive.archive.as_raw(),
            )
        };
        if array.is_null() {
            return Err("NSArray allocation failed".to_string());
        }
        // SAFETY: `descriptor` is live and `array` is an autoreleased NSArray.
        unsafe {
            objc::msg_send_void_1(descriptor.as_raw(), objc::sel(c"setBinaryArchives:"), array);
        }
        Ok(descriptor)
    }

    fn new_compute_pipeline(
        device: &DirectMetalTransport,
        descriptor: &ObjcObject,
    ) -> Result<ObjcObject, String> {
        let _pool = AutoreleasePool::new();
        let mut ns_err = objc::empty_id();
        // SAFETY: `device` and `descriptor` are live Metal objects.
        let raw = unsafe {
            objc::msg_send_id_err(
                device.device_raw(),
                objc::sel(c"newComputePipelineStateWithDescriptor:error:"),
                descriptor.as_raw(),
                &mut ns_err,
            )
        };
        ObjcObject::from_retained(raw).ok_or_else(|| unsafe { objc::nserror_message(ns_err) })
    }

    fn temp_archive_dir(label: &str) -> TempDir {
        TempDir::new(label)
    }

    /// Create -> add compute pipeline -> serialize -> reopen.
    #[test]
    fn archive_create_add_serialize_and_reopen() {
        let _lock = GpuLock::acquire();
        let Some(device) = open_device() else {
            return;
        };
        println!(
            "respondsToSelector:newBinaryArchiveWithDescriptor:error: = {}",
            BinaryArchive::is_supported(&device)
        );
        if !BinaryArchive::is_supported(&device) {
            println!("PARTIAL: MTLBinaryArchive is unsupported on this runtime");
            return;
        }
        let dir = temp_archive_dir("roundtrip");
        let Some(metallib) = build_metallib(dir.path()) else {
            return;
        };
        assert!(!metallib.is_empty());

        let key = archive_content_key(&metallib, KERNEL_ENTRY, b"compute-pso-v1");
        let path = BinaryArchive::path_for_key(dir.path(), &key);
        let library = Library::load(&device, &metallib).expect("metallib should load");
        let function = library.function(KERNEL_ENTRY).expect("kernel function");

        {
            let archive = BinaryArchive::open_or_create(&device, &path).expect("create archive");
            let descriptor = compute_descriptor(&function, Some(&archive)).expect("descriptor");
            let pso = new_compute_pipeline(&device, &descriptor)
                .expect("cold compute pipeline should compile");
            drop(pso);
            // SAFETY: `descriptor` is a live MTLComputePipelineDescriptor that
            // names the archive in `binaryArchives`.
            unsafe { archive.add_compute_pipeline(descriptor.as_raw()) }
                .expect("compute pipeline should be added");
            archive.serialize().expect("archive should serialize");
        }

        let written = std::fs::metadata(&path).expect("archive file exists").len();
        println!("archive bytes: {written}");
        assert!(written > 0, "serialized archive must not be empty");

        let reopened = BinaryArchive::open_or_create(&device, &path).expect("reopen archive");
        let descriptor = compute_descriptor(&function, Some(&reopened)).expect("warm descriptor");
        let warm = new_compute_pipeline(&device, &descriptor)
            .expect("warm compute pipeline should load from the archive");
        drop(warm);
        assert_eq!(reopened.path(), path.as_path());
    }

    /// A corrupt archive is deleted and replaced, never fatal.
    #[test]
    fn corrupt_archive_is_replaced_without_failing() {
        let _lock = GpuLock::acquire();
        let Some(device) = open_device() else {
            return;
        };
        if !BinaryArchive::is_supported(&device) {
            println!("PARTIAL: MTLBinaryArchive is unsupported on this runtime");
            return;
        }
        let dir = temp_archive_dir("corrupt");
        let Some(metallib) = build_metallib(dir.path()) else {
            return;
        };
        let key = archive_content_key(&metallib, KERNEL_ENTRY, b"compute-pso-v1");
        let path = BinaryArchive::path_for_key(dir.path(), &key);

        let library = Library::load(&device, &metallib).expect("metallib should load");
        let function = library.function(KERNEL_ENTRY).expect("kernel function");

        // Write one valid archive, then destroy the file.
        {
            let archive = BinaryArchive::open_or_create(&device, &path).expect("create archive");
            let descriptor = compute_descriptor(&function, Some(&archive)).expect("descriptor");
            let pso = new_compute_pipeline(&device, &descriptor).expect("cold PSO");
            drop(pso);
            // SAFETY: Live compute descriptor naming this archive.
            unsafe { archive.add_compute_pipeline(descriptor.as_raw()) }
                .expect("add compute pipeline");
            archive.serialize().expect("serialize valid archive");
        }
        assert!(std::fs::metadata(&path).expect("archive stat").len() > 0);
        std::fs::write(&path, b"this is not a Metal binary archive").expect("write junk");

        // Recovery: the junk file is deleted and a fresh archive is handed back.
        let archive = BinaryArchive::open_or_create(&device, &path)
            .expect("a corrupt archive must be replaced, not fatal");
        assert!(
            !path.exists(),
            "the corrupt archive file must be deleted before a fresh archive is created"
        );
        let descriptor =
            compute_descriptor(&function, Some(&archive)).expect("recovery descriptor");
        let pso = new_compute_pipeline(&device, &descriptor)
            .expect("pipeline should compile into the replacement archive");
        drop(pso);
        // SAFETY: Live compute descriptor naming the replacement archive.
        unsafe { archive.add_compute_pipeline(descriptor.as_raw()) }
            .expect("add compute pipeline to replacement");
        archive
            .serialize()
            .expect("replacement archive should serialize");

        let reopened = BinaryArchive::open_or_create(&device, &path)
            .expect("the freshly written archive should reopen");
        assert!(
            std::fs::metadata(reopened.path())
                .expect("archive stat")
                .len()
                > 0
        );
        let descriptor = compute_descriptor(&function, Some(&reopened)).expect("reopen descriptor");
        let pso = new_compute_pipeline(&device, &descriptor)
            .expect("pipeline should load from the reopened archive");
        drop(pso);
    }

    /// Cold vs warm compute pipeline creation with the same archive path
    /// cleared before the cold phase.
    #[test]
    fn cold_and_warm_compute_pipeline_creation_are_measured() {
        let _lock = GpuLock::acquire();
        let Some(device) = open_device() else {
            return;
        };
        print_selector_probe(&device);
        let supported = BinaryArchive::is_supported(&device);
        println!(
            "MTLBinaryArchive supported = {supported} (respondsToSelector:newBinaryArchiveWithDescriptor:error:)"
        );
        let dir = temp_archive_dir("bench");
        let Some(metallib) = build_metallib(dir.path()) else {
            return;
        };
        if !supported {
            let elapsed = fallback_disk_cache_timing(dir.path(), &metallib);
            println!(
                "PARTIAL: MTLBinaryArchive unsupported; disk cache {elapsed:?} for one put+get"
            );
            return;
        }

        let key = archive_content_key(&metallib, KERNEL_ENTRY, b"compute-pso-v1");
        let path = BinaryArchive::path_for_key(dir.path(), &key);
        let _ = std::fs::remove_file(&path);

        // Cold: empty archive -> load library -> compile PSO -> add -> serialize.
        let cold_start = Instant::now();
        let cold_archive = BinaryArchive::open_or_create(&device, &path).expect("cold archive");
        let cold_open = cold_start.elapsed();
        let library = Library::load(&device, &metallib).expect("metallib should load");
        let function = library.function(KERNEL_ENTRY).expect("kernel function");
        let descriptor = compute_descriptor(&function, Some(&cold_archive)).expect("descriptor");
        let pso_start = Instant::now();
        let pso = new_compute_pipeline(&device, &descriptor).expect("cold PSO should compile");
        let cold_pso = pso_start.elapsed();
        drop(pso);
        // SAFETY: Live compute descriptor naming this archive.
        unsafe { cold_archive.add_compute_pipeline(descriptor.as_raw()) }
            .expect("cold add_compute_pipeline");
        let add_and_serialize_start = Instant::now();
        cold_archive.serialize().expect("cold serialize");
        let cold_store = add_and_serialize_start.elapsed();
        let cold = cold_start.elapsed();
        let archive_bytes = std::fs::metadata(&path).expect("archive stat").len();
        drop(descriptor);
        drop(cold_archive);

        // Warm: reopen the serialized archive and build the same PSO again.
        let warm_start = Instant::now();
        let warm_archive = BinaryArchive::open_or_create(&device, &path).expect("warm archive");
        let warm_open = warm_start.elapsed();
        let warm_library = Library::load(&device, &metallib).expect("metallib should load");
        let warm_function = warm_library
            .function(KERNEL_ENTRY)
            .expect("kernel function");
        let warm_descriptor =
            compute_descriptor(&warm_function, Some(&warm_archive)).expect("warm descriptor");
        let warm_pso_start = Instant::now();
        let warm_pso =
            new_compute_pipeline(&device, &warm_descriptor).expect("warm PSO should load");
        let warm_pso_time = warm_pso_start.elapsed();
        drop(warm_pso);
        let warm = warm_start.elapsed();

        // Control: the same PSO built from the same binary metallib without any
        // archive. It separates archive reuse from the OS shader cache.
        let control_descriptor =
            compute_descriptor(&warm_function, None).expect("control descriptor");
        let control_start = Instant::now();
        let control_pso =
            new_compute_pipeline(&device, &control_descriptor).expect("control PSO should compile");
        let control = control_start.elapsed();
        drop(control_pso);

        let ratio = cold.as_secs_f64() / warm.as_secs_f64().max(f64::MIN_POSITIVE);
        println!(
            "MTLBinaryArchive cold {:.3} ms (open {:.3}, pso {:.3}, store {:.3}), warm {:.3} ms \
             (open {:.3}, pso {:.3}), no-archive control pso {:.3} ms, ratio {:.2}x, archive {} bytes",
            cold.as_secs_f64() * 1e3,
            cold_open.as_secs_f64() * 1e3,
            cold_pso.as_secs_f64() * 1e3,
            cold_store.as_secs_f64() * 1e3,
            warm.as_secs_f64() * 1e3,
            warm_open.as_secs_f64() * 1e3,
            warm_pso_time.as_secs_f64() * 1e3,
            control.as_secs_f64() * 1e3,
            ratio,
            archive_bytes
        );
        if ratio <= 1.0 {
            println!(
                "NOTE: warm start was not faster in this run (cold {:.3} ms, warm {:.3} ms)",
                cold.as_secs_f64() * 1e3,
                warm.as_secs_f64() * 1e3
            );
        }
    }

    /// Disk-cache fallback timing when `MTLBinaryArchive` is unavailable.
    fn fallback_disk_cache_timing(dir: &Path, metallib: &[u8]) -> Duration {
        let cache = DiskCache::open(dir.join("shader-cache")).expect("open disk cache");
        let options = TranslationOptions::default();
        let key = TranslationKey::new(
            metallib,
            None,
            KERNEL_ENTRY,
            8,
            "victory-metal-test/archive-bench",
            &options,
        );
        let reflection = ReflectionPayload::from_translation(
            metallib,
            8,
            KERNEL_ENTRY,
            metallib,
            None,
            "victory-metal-test/archive-bench",
            &options,
            &[],
        );
        let cached = Cached {
            metallib: metallib.to_vec(),
            reflection,
        };
        let start = Instant::now();
        cache.put(&key, &cached).expect("disk cache put");
        cache.get(&key).expect("disk cache get");
        start.elapsed()
    }
}
