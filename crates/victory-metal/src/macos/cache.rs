//! Persistent disk cache for DXIL to Metal library translation.
//!
//! A translation result is keyed by content and layout, never by file name or
//! pipeline order:
//!
//! - the DXIL byte hash,
//! - the root signature / root layout bytes or hash,
//! - the entry-point name and stage,
//! - the IRConverter version and the translation options that can change
//!   output.
//!
//! The cache stores the serialized metallib plus the reflection payload the
//! binder needs, writes through a temporary file and a rename so a crash
//! never leaves a partial entry, and treats an unreadable or mismatched entry
//! as a miss rather than a failure.
//!
//! # On-disk key layout (format version 1)
//!
//! [`TranslationKey::canonical_bytes`] is the byte string hashed with BLAKE3 to
//! obtain [`TranslationKey::digest`]. All integers are little-endian and every
//! variable-length field is length-prefixed so another process can reproduce
//! the key byte for byte:
//!
//! ```text
//! offset  size  field
//! 0       4     magic b"VTCK"
//! 4       2     format version u16 = 1
//! 6       2     reserved u16 = 0
//! 8       4     dxil_hash length u32 = 32
//! 12      32    dxil_hash: BLAKE3 of the raw DXIL bytes
//! 44      4     root_layout_hash length u32 = 32
//! 48      32    root_layout_hash (see below)
//! 80      4     stage u32 (0 = unspecified, the DXIL content names the stage)
//! 84      4     entry length u32
//! 88      N     entry UTF-8 bytes
//! +0      4     converter_version length u32
//! +4      M     converter_version UTF-8 bytes
//! +0      4     options.target length u32
//! +4      T     options.target UTF-8 bytes
//! +0      4     options.min_os length u32
//! +4      O     options.min_os UTF-8 bytes
//! +0      4     options.flags count u32
//! ...           per flag: length u32 followed by UTF-8 bytes
//! ```
//!
//! `root_layout_hash` is `BLAKE3(raw root signature bytes)` when the caller
//! supplied the raw layout. When it did not, it is
//! `BLAKE3(b"victory-metal/root-layout/absent/v1")`, a fixed domain-separated
//! value that can never collide with a real signature of the same short text.
//!
//! Binding metadata (`ReflectionPayload::bindings`) is *output* metadata tied to
//! the root layout; it is stored in the reflection payload but deliberately
//! does not add a key dimension, because the root-layout hash already selects
//! the translated resource layout.
//!
//! # Disk entry layout (format version 1)
//!
//! One file per key, named `<digest-hex>.vce`:
//!
//! ```text
//! 0   4   magic b"VTCE"
//! 4   2   format version u16 = 1
//! 6   2   reserved u16 = 0
//! 8   4   stage u32
//! 12  4   metallib length u32
//! 16  4   reflection length u32
//! 20  4   reserved u32 = 0
//! 24  32  metallib BLAKE3 hash
//! 56  32  key digest
//! 88  ... metallib bytes
//! ... ... encoded reflection payload
//! end 32  BLAKE3 of every preceding byte
//! ```
//!
//! Any read that disagrees with the header, the sizes, the key digest, the
//! metallib hash, the reflection payload or the trailing hash is a miss.
//!
//! # Retention policy
//!
//! [`DiskCache::open`] bounds the cache at [`DEFAULT_MAX_TOTAL_BYTES`] (256 MiB).
//! `put` writes the new entry first and then evicts least-recently-modified
//! `.vce` entries (oldest `mtime` first) until the total is within the bound.
//! The entry just written is never evicted, even if it alone exceeds the bound;
//! in that case the directory stays over budget until another entry can be
//! removed. Temporary files are not counted and are never evicted by this
//! policy.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Current on-disk key and entry format version.
pub const CACHE_FORMAT_VERSION: u16 = 1;

/// Default total-size bound for a [`DiskCache`] directory (256 MiB).
pub const DEFAULT_MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

/// Largest metallib accepted into or returned from the cache (64 MiB).
///
/// Mirrors the conversion limit without importing the converter module.
pub const MAX_CACHED_METALLIB_BYTES: usize = 64 * 1024 * 1024;

/// Largest raw root signature carried in a key/reflection payload.
pub const MAX_CACHED_ROOT_SIGNATURE_BYTES: usize = 1024 * 1024;

/// Largest encoded reflection payload accepted from disk or a worker.
pub const MAX_REFLECTION_BYTES: usize = 4 * 1024 * 1024;

const KEY_MAGIC: &[u8; 4] = b"VTCK";
const ENTRY_MAGIC: &[u8; 4] = b"VTCE";
const REFLECTION_MAGIC: &[u8; 4] = b"VTRF";
const ABSENT_ROOT_LAYOUT_DOMAIN: &[u8] = b"victory-metal/root-layout/absent/v1";
/// Domain separation for [`archive_content_key`].
const ARCHIVE_KEY_DOMAIN: &[u8] = b"victory-metal/binary-archive-key/v1";
const ENTRY_HEADER_BYTES: usize = 88;
const ENTRY_TRAILER_BYTES: usize = 32;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// BLAKE3 hash of `bytes` as a fixed-size array.
pub fn blake3_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Hash of the caller's root layout, or the documented absent-layout value.
pub fn root_layout_hash(root_signature: Option<&[u8]>) -> [u8; 32] {
    match root_signature {
        Some(bytes) => blake3_bytes(bytes),
        None => blake3_bytes(ABSENT_ROOT_LAYOUT_DOMAIN),
    }
}

/// Content key for a persistent `MTLBinaryArchive` entry.
///
/// The key is `BLAKE3` over a fixed domain separator followed by
/// length-prefixed `metallib` bytes, the entry-point name and the caller's
/// pipeline `state` bytes. Length prefixes keep `metallib = b"ab", entry = "c"`
/// distinct from `metallib = b"a", entry = "bc"`.
///
/// This is the single key builder for binary archives: `macos::binary_archive`
/// calls it instead of re-implementing the hash layout, so the on-disk archive
/// identity can be reproduced by any process.
pub fn archive_content_key(metallib: &[u8], entry: &str, state: &[u8]) -> [u8; 32] {
    let mut out = Vec::with_capacity(ARCHIVE_KEY_DOMAIN.len() + 32 + metallib.len() + state.len());
    out.extend_from_slice(ARCHIVE_KEY_DOMAIN);
    push_u64(
        &mut out,
        u64::try_from(metallib.len()).expect("metallib length exceeds u64"),
    );
    out.extend_from_slice(metallib);
    push_string(&mut out, entry);
    push_u64(
        &mut out,
        u64::try_from(state.len()).expect("state length exceeds u64"),
    );
    out.extend_from_slice(state);
    blake3_bytes(&out)
}

/// Lowercase hexadecimal rendering of a 32-byte digest.
///
/// Public so path construction for content-addressed artifacts is shared with
/// [`TranslationKey::digest_hex`] instead of copied.
pub fn hash_hex(hash: &[u8; 32]) -> String {
    hex(hash)
}

/// Translation options that can change converter output.
///
/// `target` and `min_os` describe the deployment target handed to the converter
/// process; `flags` is an ordered list of converter flags. All three participate
/// in [`TranslationKey`], so a future flag can never silently reuse output
/// produced by a different translation configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationOptions {
    /// Deployment target, for example `apple-macos`.
    pub target: String,
    /// Minimum OS version, for example `14.0`. Empty means the converter
    /// default.
    pub min_os: String,
    /// Ordered converter flags. Empty for the verified default pipeline.
    pub flags: Vec<String>,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            target: "apple-macos".to_string(),
            min_os: String::new(),
            flags: Vec::new(),
        }
    }
}

impl TranslationOptions {
    fn encode_into(&self, out: &mut Vec<u8>) {
        push_string(out, &self.target);
        push_string(out, &self.min_os);
        push_u32(
            out,
            u32::try_from(self.flags.len()).expect("flag count exceeds u32"),
        );
        for flag in &self.flags {
            push_string(out, flag);
        }
    }

    fn decode(reader: &mut Reader<'_>) -> Option<Self> {
        let target = reader.string(4096)?;
        let min_os = reader.string(4096)?;
        let count = reader.u32()? as usize;
        if count > 1024 {
            return None;
        }
        let mut flags = Vec::with_capacity(count);
        for _ in 0..count {
            flags.push(reader.string(4096)?);
        }
        Some(Self {
            target,
            min_os,
            flags,
        })
    }
}

/// Kind of resource binding recorded in the reflection payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    /// Unordered access view (`uN` in HLSL).
    Uav,
}

impl BindingKind {
    /// Stable one-byte encoding used in the reflection payload.
    pub fn code(self) -> u8 {
        match self {
            Self::Uav => 1,
        }
    }

    /// Decodes a stable one-byte binding kind.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Uav),
            _ => None,
        }
    }

    /// Worker command-line spelling (`uav`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uav => "uav",
        }
    }

    /// Parses the worker command-line spelling.
    pub fn from_str(text: &str) -> Option<Self> {
        match text {
            "uav" => Some(Self::Uav),
            _ => None,
        }
    }
}

/// Binding metadata carried alongside a translation result.
///
/// The converter cannot derive arbitrary layouts on its own; a caller that has
/// a verified layout supplies these entries and they are stored in the
/// reflection payload so the binder can consume the same layout without
/// re-translating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingMetadata {
    /// Resource kind.
    pub kind: BindingKind,
    /// HLSL register index.
    pub register: u32,
    /// Metal buffer argument index.
    pub metal_index: u32,
    /// Byte offset inside the Metal argument buffer.
    pub byte_offset: u32,
}

/// Content-addressed translation key.
///
/// The fields are hashed by [`Self::digest`]; equality and the canonical byte
/// encoding cover every field. See the module documentation for the exact byte
/// layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationKey {
    /// BLAKE3 of the raw DXIL bytes.
    pub dxil_hash: [u8; 32],
    /// Root-layout hash (raw signature bytes or the absent-layout domain).
    pub root_layout_hash: [u8; 32],
    /// Entry-point name.
    pub entry: String,
    /// Declared shader stage; `0` means "unspecified, named by the DXIL".
    pub stage: u32,
    /// Converter/protocol version string.
    pub converter_version: String,
    /// Translation options that can change output.
    pub options: TranslationOptions,
}

impl TranslationKey {
    /// Builds a key from the translation inputs.
    pub fn new(
        dxil: &[u8],
        root_signature: Option<&[u8]>,
        entry: &str,
        stage: u32,
        converter_version: &str,
        options: &TranslationOptions,
    ) -> Self {
        Self {
            dxil_hash: blake3_bytes(dxil),
            root_layout_hash: root_layout_hash(root_signature),
            entry: entry.to_string(),
            stage,
            converter_version: converter_version.to_string(),
            options: options.clone(),
        }
    }

    /// Canonical byte encoding hashed by [`Self::digest`].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128 + self.entry.len() + self.converter_version.len());
        out.extend_from_slice(KEY_MAGIC);
        push_u16(&mut out, CACHE_FORMAT_VERSION);
        push_u16(&mut out, 0);
        push_hash(&mut out, &self.dxil_hash);
        push_hash(&mut out, &self.root_layout_hash);
        push_u32(&mut out, self.stage);
        push_string(&mut out, &self.entry);
        push_string(&mut out, &self.converter_version);
        self.options.encode_into(&mut out);
        out
    }

    /// BLAKE3 digest of [`Self::canonical_bytes`].
    pub fn digest(&self) -> [u8; 32] {
        blake3_bytes(&self.canonical_bytes())
    }

    /// Lowercase hexadecimal digest, used as the on-disk file stem.
    pub fn digest_hex(&self) -> String {
        hex(&self.digest())
    }
}

/// Metadata produced alongside a metallib and stored in the cache.
///
/// Encoding is a self-describing, length-prefixed binary payload (magic
/// `VTRF`, format version 1) so the worker process and the renderer can pass it
/// without a serialization dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectionPayload {
    /// Metal shader stage reported by the converter.
    pub stage: u32,
    /// Entry point the metallib was compiled for.
    pub entry: String,
    /// BLAKE3 of the DXIL bytes this payload belongs to.
    pub dxil_hash: [u8; 32],
    /// Root-layout hash the key was built with.
    pub root_layout_hash: [u8; 32],
    /// Raw root signature bytes when the caller supplied them.
    pub root_signature: Option<Vec<u8>>,
    /// Converter/protocol version string.
    pub converter_version: String,
    /// Translation options used for this result.
    pub options: TranslationOptions,
    /// Binding metadata carried for the binder.
    pub bindings: Vec<BindingMetadata>,
    /// Length of the stored metallib.
    pub metallib_bytes: u64,
    /// BLAKE3 of the stored metallib.
    pub metallib_hash: [u8; 32],
}

impl ReflectionPayload {
    /// Builds the reflection payload for a completed translation.
    pub fn from_translation(
        metallib: &[u8],
        stage: u32,
        entry: &str,
        dxil: &[u8],
        root_signature: Option<&[u8]>,
        converter_version: &str,
        options: &TranslationOptions,
        bindings: &[BindingMetadata],
    ) -> Self {
        Self {
            stage,
            entry: entry.to_string(),
            dxil_hash: blake3_bytes(dxil),
            root_layout_hash: root_layout_hash(root_signature),
            root_signature: root_signature.map(<[u8]>::to_vec),
            converter_version: converter_version.to_string(),
            options: options.clone(),
            bindings: bindings.to_vec(),
            metallib_bytes: u64::try_from(metallib.len()).expect("metallib length exceeds u64"),
            metallib_hash: blake3_bytes(metallib),
        }
    }

    /// Encodes the payload into its stable binary representation.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + self.entry.len());
        out.extend_from_slice(REFLECTION_MAGIC);
        push_u16(&mut out, CACHE_FORMAT_VERSION);
        push_u16(&mut out, 0);
        push_u32(&mut out, self.stage);
        out.extend_from_slice(&self.metallib_bytes.to_le_bytes());
        push_raw_hash(&mut out, &self.metallib_hash);
        push_raw_hash(&mut out, &self.dxil_hash);
        push_raw_hash(&mut out, &self.root_layout_hash);
        push_string(&mut out, &self.entry);
        push_string(&mut out, &self.converter_version);
        self.options.encode_into(&mut out);
        match &self.root_signature {
            Some(bytes) => {
                out.push(1);
                push_u32(
                    &mut out,
                    u32::try_from(bytes.len()).expect("root signature length exceeds u32"),
                );
                out.extend_from_slice(bytes);
            }
            None => out.push(0),
        }
        push_u32(
            &mut out,
            u32::try_from(self.bindings.len()).expect("binding count exceeds u32"),
        );
        for binding in &self.bindings {
            out.push(binding.kind.code());
            push_u32(&mut out, binding.register);
            push_u32(&mut out, binding.metal_index);
            push_u32(&mut out, binding.byte_offset);
        }
        out
    }

    /// Decodes a payload, returning `None` for any malformed or out-of-bounds
    /// input. Never panics.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_REFLECTION_BYTES {
            return None;
        }
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != REFLECTION_MAGIC {
            return None;
        }
        if reader.u16()? != CACHE_FORMAT_VERSION || reader.u16()? != 0 {
            return None;
        }
        let stage = reader.u32()?;
        let metallib_bytes = reader.u64()?;
        let metallib_hash = reader.hash()?;
        let dxil_hash = reader.hash()?;
        let root_layout_hash = reader.hash()?;
        let entry = reader.string(256)?;
        let converter_version = reader.string(4096)?;
        let options = TranslationOptions::decode(&mut reader)?;
        let root_signature = match reader.u8()? {
            0 => None,
            1 => {
                let len = reader.u32()? as usize;
                if len > MAX_CACHED_ROOT_SIGNATURE_BYTES {
                    return None;
                }
                Some(reader.take(len)?.to_vec())
            }
            _ => return None,
        };
        let binding_count = reader.u32()? as usize;
        if binding_count > 1024 {
            return None;
        }
        let mut bindings = Vec::with_capacity(binding_count);
        for _ in 0..binding_count {
            let kind = BindingKind::from_code(reader.u8()?)?;
            bindings.push(BindingMetadata {
                kind,
                register: reader.u32()?,
                metal_index: reader.u32()?,
                byte_offset: reader.u32()?,
            });
        }
        if !reader.is_empty() {
            return None;
        }
        Some(Self {
            stage,
            entry,
            dxil_hash,
            root_layout_hash,
            root_signature,
            converter_version,
            options,
            bindings,
            metallib_bytes,
            metallib_hash,
        })
    }
}

/// One cached translation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cached {
    /// Serialized metallib bytes.
    pub metallib: Vec<u8>,
    /// Reflection metadata the binder consumes.
    pub reflection: ReflectionPayload,
}

/// Directory-scoped persistent translation cache.
///
/// `DiskCache` is immutable and safe to share across threads; `get` and `put`
/// rely only on the filesystem and the atomic rename.
#[derive(Debug, Clone)]
pub struct DiskCache {
    dir: PathBuf,
    max_total_bytes: u64,
}

impl DiskCache {
    /// Opens (creating) a cache directory with the default size bound.
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::with_max_total_bytes(dir, DEFAULT_MAX_TOTAL_BYTES)
    }

    /// Opens (creating) a cache directory with an explicit size bound.
    pub fn with_max_total_bytes(
        dir: impl Into<PathBuf>,
        max_total_bytes: u64,
    ) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            max_total_bytes,
        })
    }

    /// Cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Least-recently-modified eviction bound in bytes.
    pub fn max_total_bytes(&self) -> u64 {
        self.max_total_bytes
    }

    /// Path of the entry file for `key`.
    pub fn path_for(&self, key: &TranslationKey) -> PathBuf {
        self.dir.join(format!("{}.vce", key.digest_hex()))
    }

    /// Reads a valid entry, or `None` for a missing, truncated, unreadable,
    /// size-mismatched or hash-mismatched entry. Never fails.
    pub fn get(&self, key: &TranslationKey) -> Option<Cached> {
        let bytes = fs::read(self.path_for(key)).ok()?;
        decode_entry(&bytes, key)
    }

    /// Stores an entry through a same-directory temporary file and an atomic
    /// rename, then enforces the size bound.
    pub fn put(&self, key: &TranslationKey, cached: &Cached) -> std::io::Result<()> {
        let encoded = encode_entry(key, cached).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "translation result exceeds the cache entry limits",
            )
        })?;
        let path = self.path_for(key);
        let tmp = self.unique_temp_path(key);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let write_result = file.write_all(&encoded).and_then(|()| file.sync_all());
        drop(file);
        if let Err(err) = write_result {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
        if let Err(err) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
        self.enforce_budget(&path);
        Ok(())
    }

    fn unique_temp_path(&self, key: &TranslationKey) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        self.dir.join(format!(
            ".{}.{}.{}.tmp",
            key.digest_hex(),
            std::process::id(),
            sequence
        ))
    }

    /// Evicts the oldest entries until the directory is within budget.
    ///
    /// `keep` is the entry that was just written and is never removed.
    fn enforce_budget(&self, keep: &Path) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut files: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        let mut total = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("vce") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            total = total.saturating_add(metadata.len());
            files.push((path, metadata.len(), modified));
        }
        if total <= self.max_total_bytes {
            return;
        }
        files.sort_by_key(|(_, _, modified)| *modified);
        for (path, len, _) in files {
            if total <= self.max_total_bytes {
                break;
            }
            if path == keep {
                continue;
            }
            if fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }
}

fn encode_entry(key: &TranslationKey, cached: &Cached) -> Option<Vec<u8>> {
    if cached.metallib.is_empty() || cached.metallib.len() > MAX_CACHED_METALLIB_BYTES {
        return None;
    }
    let reflection = cached.reflection.encode();
    if reflection.len() > MAX_REFLECTION_BYTES {
        return None;
    }
    let metallib_len = u32::try_from(cached.metallib.len()).ok()?;
    let reflection_len = u32::try_from(reflection.len()).ok()?;
    let metallib_hash = blake3_bytes(&cached.metallib);
    let mut out = Vec::with_capacity(ENTRY_HEADER_BYTES + cached.metallib.len() + reflection.len());
    out.extend_from_slice(ENTRY_MAGIC);
    push_u16(&mut out, CACHE_FORMAT_VERSION);
    push_u16(&mut out, 0);
    push_u32(&mut out, cached.reflection.stage);
    push_u32(&mut out, metallib_len);
    push_u32(&mut out, reflection_len);
    push_u32(&mut out, 0);
    push_raw_hash(&mut out, &metallib_hash);
    push_raw_hash(&mut out, &key.digest());
    out.extend_from_slice(&cached.metallib);
    out.extend_from_slice(&reflection);
    let trailer = blake3_bytes(&out);
    out.extend_from_slice(&trailer);
    Some(out)
}

fn decode_entry(bytes: &[u8], key: &TranslationKey) -> Option<Cached> {
    if bytes.len() < ENTRY_HEADER_BYTES + ENTRY_TRAILER_BYTES {
        return None;
    }
    let (body, trailer) = bytes.split_at(bytes.len() - ENTRY_TRAILER_BYTES);
    if blake3_bytes(body) != trailer {
        return None;
    }
    let mut reader = Reader::new(body);
    if reader.take(4)? != ENTRY_MAGIC {
        return None;
    }
    if reader.u16()? != CACHE_FORMAT_VERSION || reader.u16()? != 0 {
        return None;
    }
    let stage = reader.u32()?;
    let metallib_len = reader.u32()? as usize;
    let reflection_len = reader.u32()? as usize;
    if reader.u32()? != 0 {
        return None;
    }
    let metallib_hash = reader.hash()?;
    let key_digest = reader.hash()?;
    if key_digest != key.digest() {
        return None;
    }
    if metallib_len == 0 || metallib_len > MAX_CACHED_METALLIB_BYTES {
        return None;
    }
    if reflection_len > MAX_REFLECTION_BYTES {
        return None;
    }
    let metallib = reader.take(metallib_len)?.to_vec();
    let reflection_bytes = reader.take(reflection_len)?;
    if !reader.is_empty() {
        return None;
    }
    if blake3_bytes(&metallib) != metallib_hash {
        return None;
    }
    let reflection = ReflectionPayload::decode(reflection_bytes)?;
    if reflection.metallib_bytes != metallib_len as u64
        || reflection.metallib_hash != metallib_hash
        || reflection.stage != stage
    {
        return None;
    }
    Some(Cached {
        metallib,
        reflection,
    })
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_hash(out: &mut Vec<u8>, hash: &[u8; 32]) {
    push_u32(out, 32);
    out.extend_from_slice(hash);
}

fn push_raw_hash(out: &mut Vec<u8>, hash: &[u8; 32]) {
    out.extend_from_slice(hash);
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u32(
        out,
        u32::try_from(value.len()).expect("string length exceeds u32"),
    );
    out.extend_from_slice(value.as_bytes());
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(len)?;
        let slice = self.data.get(self.position..end)?;
        self.position = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        self.take(1)?.first().copied()
    }

    fn u16(&mut self) -> Option<u16> {
        let bytes = self.take(2)?;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let bytes = self.take(4)?;
        Some(u32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))
    }

    fn u64(&mut self) -> Option<u64> {
        let bytes = self.take(8)?;
        Some(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn hash(&mut self) -> Option<[u8; 32]> {
        self.take(32)?.try_into().ok()
    }

    fn string(&mut self, max: usize) -> Option<String> {
        let len = self.u32()? as usize;
        if len > max {
            return None;
        }
        String::from_utf8(self.take(len)?.to_vec()).ok()
    }

    fn is_empty(&self) -> bool {
        self.position == self.data.len()
    }
}

/// Test-only scratch directory.
#[cfg(test)]
pub(crate) struct TempDir(PathBuf);

#[cfg(test)]
impl TempDir {
    pub(crate) fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "victory-metal-{label}-{}-{sequence}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create test temp dir");
        Self(path)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key() -> TranslationKey {
        TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "main",
            8,
            "victory-metal-test/1",
            &TranslationOptions {
                target: "apple-macos".to_string(),
                min_os: "14.0".to_string(),
                flags: vec!["-O2".to_string()],
            },
        )
    }

    fn sample_cached() -> Cached {
        let metallib = b"METALLIB-PAYLOAD".to_vec();
        let reflection = ReflectionPayload::from_translation(
            &metallib,
            8,
            "main",
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "victory-metal-test/1",
            &TranslationOptions {
                target: "apple-macos".to_string(),
                min_os: "14.0".to_string(),
                flags: vec!["-O2".to_string()],
            },
            &[BindingMetadata {
                kind: BindingKind::Uav,
                register: 0,
                metal_index: 2,
                byte_offset: 0,
            }],
        );
        Cached {
            metallib,
            reflection,
        }
    }

    #[test]
    fn key_byte_layout_is_stable_against_a_golden_fixture() {
        let key = sample_key();
        let canonical = key.canonical_bytes();
        assert_eq!(
            hex(&canonical),
            "5654434b0100000020000000712691c8e004b60d4f1c9d86858329a6f4330c08a1b25554317a3045f3508adb2000000002e7c14f43736f1723e789eeedbe2d1c505bd14b3dc091c829782f19324ee36c08000000040000006d61696e14000000766963746f72792d6d6574616c2d746573742f310b0000006170706c652d6d61636f730400000031342e3001000000030000002d4f32"
        );
        assert_eq!(
            key.digest_hex(),
            "1b533dd5f89d7c671cf54f4b909a962429874187f1f704a472979fe730d4ea6d"
        );
        assert_eq!(canonical.len(), 150);
    }

    #[test]
    fn key_changes_with_every_input_dimension() {
        let base = sample_key();
        let digest = base.digest();

        let dxil = TranslationKey::new(
            b"dxil-golden-input-2",
            Some(b"root-signature-golden"),
            "main",
            8,
            "victory-metal-test/1",
            &base.options,
        );
        let root = TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-other"),
            "main",
            8,
            "victory-metal-test/1",
            &base.options,
        );
        let absent_root = TranslationKey::new(
            b"dxil-golden-input",
            None,
            "main",
            8,
            "victory-metal-test/1",
            &base.options,
        );
        let entry = TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "other",
            8,
            "victory-metal-test/1",
            &base.options,
        );
        let stage = TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "main",
            7,
            "victory-metal-test/1",
            &base.options,
        );
        let version = TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "main",
            8,
            "victory-metal-test/2",
            &base.options,
        );
        let options = TranslationKey::new(
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "main",
            8,
            "victory-metal-test/1",
            &TranslationOptions {
                target: "apple-macos".to_string(),
                min_os: "15.0".to_string(),
                flags: vec!["-O2".to_string()],
            },
        );

        for (name, key) in [
            ("dxil", dxil),
            ("root", root),
            ("absent_root", absent_root),
            ("entry", entry),
            ("stage", stage),
            ("version", version),
            ("options", options),
        ] {
            assert_ne!(key.digest(), digest, "dimension {name} did not change key");
        }
    }

    #[test]
    fn absent_root_layout_uses_a_stable_domain_value() {
        assert_eq!(
            root_layout_hash(None),
            blake3_bytes(b"victory-metal/root-layout/absent/v1")
        );
        assert_ne!(root_layout_hash(None), root_layout_hash(Some(b"")));
    }

    #[test]
    fn archive_content_key_binds_metallib_entry_and_state() {
        let base = archive_content_key(b"METALLIB", "main", b"state-v1");
        assert_eq!(base, archive_content_key(b"METALLIB", "main", b"state-v1"));
        assert_eq!(hash_hex(&base), hex(&base));
        assert_eq!(hash_hex(&base).len(), 64);

        assert_ne!(base, archive_content_key(b"METALLIB2", "main", b"state-v1"));
        assert_ne!(base, archive_content_key(b"METALLIB", "main2", b"state-v1"));
        assert_ne!(base, archive_content_key(b"METALLIB", "main", b"state-v2"));
        // Length prefixes keep the field boundary unambiguous.
        assert_ne!(
            archive_content_key(b"ab", "c", b""),
            archive_content_key(b"a", "bc", b"")
        );
    }

    #[test]
    fn disk_cache_round_trips_and_reports_entry_size() {
        let dir = TempDir::new("cache-roundtrip");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let key = sample_key();
        let cached = sample_cached();
        assert!(cache.get(&key).is_none());

        cache.put(&key, &cached).expect("store entry");
        let read = cache.get(&key).expect("cache hit");
        assert_eq!(read, cached);

        let entry_bytes = fs::metadata(cache.path_for(&key)).expect("stat entry").len();
        println!(
            "stored entry bytes: {entry_bytes} (metallib {} + reflection {})",
            cached.metallib.len(),
            cached.reflection.encode().len()
        );
        assert!(entry_bytes > cached.metallib.len() as u64);
    }

    #[test]
    fn truncated_corrupt_or_mismatched_entry_is_a_miss() {
        let dir = TempDir::new("cache-corrupt");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let key = sample_key();
        let cached = sample_cached();
        cache.put(&key, &cached).expect("store entry");
        let path = cache.path_for(&key);

        let original = fs::read(&path).expect("read entry");
        let truncated = &original[..original.len() - 8];
        fs::write(&path, truncated).expect("write truncated");
        assert!(cache.get(&key).is_none());

        let mut corrupt = original.clone();
        let middle = corrupt.len() / 2;
        corrupt[middle] ^= 0xff;
        fs::write(&path, &corrupt).expect("write corrupt");
        assert!(cache.get(&key).is_none());

        fs::write(&path, b"not a cache entry").expect("write junk");
        assert!(cache.get(&key).is_none());

        // A different key must not accept an entry stored for another key.
        fs::write(&path, &original).expect("restore entry");
        let other = TranslationKey::new(
            b"other-dxil",
            Some(b"root-signature-golden"),
            "main",
            8,
            "victory-metal-test/1",
            &sample_key().options,
        );
        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn concurrent_put_from_two_threads_leaves_one_valid_entry() {
        let dir = TempDir::new("cache-concurrent");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let key = sample_key();
        let first = sample_cached();
        let mut second = sample_cached();
        second.metallib = b"METALLIB-PAYLOAD-THREAD-2".to_vec();
        second.reflection = ReflectionPayload::from_translation(
            &second.metallib,
            8,
            "main",
            b"dxil-golden-input",
            Some(b"root-signature-golden"),
            "victory-metal-test/1",
            &sample_key().options,
            &[],
        );

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|scope| {
            for cached in [&first, &second] {
                let cache = cache.clone();
                let key = key.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    cache.put(&key, cached).expect("concurrent store");
                });
            }
        });

        let entries: Vec<_> = fs::read_dir(dir.path())
            .expect("read cache dir")
            .flatten()
            .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("vce"))
            .collect();
        assert_eq!(entries.len(), 1, "exactly one entry file must remain");
        let read = cache.get(&key).expect("one valid entry");
        assert!(read == first || read == second);
    }

    #[test]
    fn budget_eviction_never_removes_the_entry_just_written() {
        let dir = TempDir::new("cache-budget");
        let cache = DiskCache::with_max_total_bytes(dir.path(), 1).expect("open cache");
        let key = sample_key();
        let cached = sample_cached();
        cache.put(&key, &cached).expect("store oversized entry");
        // The bound is smaller than the entry, so it must survive.
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn cache_hit_latency_is_measured() {
        let dir = TempDir::new("cache-latency");
        let cache = DiskCache::open(dir.path()).expect("open cache");
        let key = sample_key();
        cache.put(&key, &sample_cached()).expect("store entry");

        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            assert!(cache.get(&key).is_some());
        }
        let elapsed = start.elapsed();
        println!(
            "cache hit latency: {} reads in {:?} ({:.2} us/read)",
            iterations,
            elapsed,
            elapsed.as_secs_f64() * 1e6 / iterations as f64
        );
    }
}
