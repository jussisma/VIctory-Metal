//! Guest D3D12 descriptor and root-binding resolution for the native backend.
//!
//! This module is the seam between the Win64 guest and the native host's
//! allocation registry. `victory-gpu` cannot depend on `victory-hle` (the
//! dependency runs the other way), so the guest memory provider is installed
//! at run time as a [`GuestRangeResolver`] and the producer-side helpers below
//! turn guest GPU virtual addresses into the Wire v3 root/descriptor bindings.
//!
//! The contract (ADR-010):
//!
//! * a guest descriptor-table handle resolves to
//!   `(allocation_id, generation, byte_offset, size)` for every descriptor the
//!   table covers, through `xmem::share`;
//! * a root CBV/SRV/UAV GPU virtual address resolves the same way;
//! * resolved ranges populate the Wire v3 root/descriptor bindings;
//! * every referenced backing range stays registered (fence-pinned) until the
//!   GPU completion that retires the submission.
//!
//! [`GuestRange`] is the *pinned backing window*: `byte_offset` is the first
//! covered page's offset in the share file and `size` the page-rounded span
//! (both multiples of [`PAGE`]). A wire binding offsets into that window by
//! the guest address's in-page byte, so an unaligned guest address keeps its
//! exact offset instead of being rounded down.
//!
//! A zero address or handle, an unmapped range, a stale generation and an
//! unsupported descriptor kind are explicit [`ResolveError`]s, never a
//! zero-filled binding.

use std::sync::{Arc, LazyLock, RwLock};

use victory_metal::wire::BufferBinding;
use victory_metal::wire_v3::{
    DescriptorKind, DescriptorTableEntryV3, DescriptorTableV3, IndexBufferBinding, IndexFormat,
    MAX_DESCRIPTOR_BINDINGS_V3,
    MAX_ROOT_CONSTANT_BYTES_V3, RootBufferView, RootConstantsV3, VertexBuffer,
};
use victory_metal::wire::MAX_RESOURCE_LEN;

/// The share provider's page: descriptor windows are rounded to it.
pub const PAGE: u64 = 64 * 1024;

/// Why a guest binding could not be resolved into a native range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The guest address or descriptor handle is zero or malformed.
    Invalid(&'static str),
    /// No shared `xmem` range covers the requested bytes.
    Unmapped { address: u64, len: u64 },
    /// A descriptor entry names a handle this backend cannot express.
    Unsupported(&'static str),
    /// The provider could not pin the resolved backing range.
    Retain(&'static str),
    /// The provider could not retire a pin (a second retirement included).
    Retire(&'static str),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid guest binding: {why}"),
            Self::Unmapped { address, len } => {
                write!(f, "no shared xmem range for 0x{address:x}+{len}")
            }
            Self::Unsupported(why) => write!(f, "unsupported guest binding: {why}"),
            Self::Retain(why) => write!(f, "guest backing not pinned: {why}"),
            Self::Retire(why) => write!(f, "guest backing not retired: {why}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// A guest range resolved to one share allocation.
///
/// `byte_offset` and `size` are the page-aligned backing window the provider
/// keeps alive, not necessarily the caller's exact byte range. The wire
/// binding's offset is computed from the original guest address
/// ([`binding_offset`]) so unaligned addresses stay exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuestRange {
    /// Share allocation id; never zero, never reused.
    pub allocation_id: u64,
    /// Allocation generation; never zero.
    pub generation: u32,
    /// First covered page's byte offset in the share file; a multiple of
    /// [`PAGE`].
    pub byte_offset: u64,
    /// Page-rounded byte length covering the guest range; a non-zero multiple
    /// of [`PAGE`].
    pub size: u64,
}

/// The guest memory registry the native backend resolves and pins through.
///
/// Implemented by the HLE's `xmem::share` provider. `retain` returns a token
/// that names the pin; `retire` releases it after the GPU completion and must
/// refuse a second retirement of the same token (submission bookkeeping that
/// goes wrong is visible, not swallowed).
pub trait GuestRangeResolver: Send + Sync {
    /// Resolves `[guest_address, guest_address + len)` to the share allocation
    /// and backing window that covers it.
    fn resolve(&self, guest_address: u64, len: u64) -> Result<GuestRange, ResolveError>;

    /// Pins the backing window named by `range` until [`retire`](Self::retire).
    fn retain(&self, range: &GuestRange) -> Result<u64, ResolveError>;

    /// Releases the pin `token` names. A second retirement is an error.
    fn retire(&self, token: u64) -> Result<(), ResolveError>;
}

static RESOLVER: LazyLock<RwLock<Option<Arc<dyn GuestRangeResolver>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Installs the process-wide resolver (the HLE installs it under
/// `VICTORY_BACKEND=metal`).
///
/// Returns the resolver it replaced, if any.
pub fn set_resolver<R>(resolver: R) -> Option<Arc<dyn GuestRangeResolver>>
where
    R: GuestRangeResolver + 'static,
{
    set_resolver_arc(Arc::new(resolver))
}

/// Installs an already shared resolver.
pub fn set_resolver_arc(resolver: Arc<dyn GuestRangeResolver>) -> Option<Arc<dyn GuestRangeResolver>> {
    RESOLVER
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .replace(resolver)
}

/// Removes the process-wide resolver, if one was installed.
pub fn clear_resolver() -> Option<Arc<dyn GuestRangeResolver>> {
    RESOLVER
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .take()
}

/// The installed resolver, if any.
pub fn resolver() -> Option<Arc<dyn GuestRangeResolver>> {
    RESOLVER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// One guest descriptor's or root view's GPU virtual address and byte length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestBinding {
    /// Guest GPU virtual address; never zero.
    pub address: u64,
    /// Referenced byte length; never zero.
    pub len: u64,
}

/// Bytes between consecutive live view descriptors (CBV/SRV/UAV) in a guest
/// descriptor heap.
///
/// This is the view entry of `descriptor.rs`'s per-heap increments
/// `[32, 16, 128, 128]`; the unit tests below compare it with
/// `crate::descriptor::increment_size` so the two cannot drift.
pub const VIEW_DESCRIPTOR_INCREMENT: u64 = 32;

/// Bytes between consecutive live sampler descriptors in a guest descriptor
/// heap (the sampler entry of the same per-heap increments).
pub const SAMPLER_DESCRIPTOR_INCREMENT: u64 = 16;

/// The guest stride of one descriptor of `kind`.
pub fn descriptor_increment(kind: DescriptorKind) -> u64 {
    match kind {
        DescriptorKind::Sampler => SAMPLER_DESCRIPTOR_INCREMENT,
        DescriptorKind::Cbv | DescriptorKind::Srv | DescriptorKind::Uav => {
            VIEW_DESCRIPTOR_INCREMENT
        }
    }
}

/// One expressible descriptor-table entry before it is resolved or encoded.
///
/// A buffer entry still has to resolve and pin its guest range; a texture or
/// sampler entry already names a handle the host registered (kind 8 or the
/// sampler-registration frame). Wire v3 encodes one entry shape per table, so a
/// run may not mix the three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorEntry {
    /// A guest buffer range, resolved through [`GuestRangeResolver`].
    Buffer(GuestBinding),
    /// A registered texture handle (`gpuResourceID` resolved on the host).
    Texture(u64),
    /// A registered sampler-state handle.
    Sampler(u64),
}

impl DescriptorEntry {
    /// Whether this entry shape can be carried by a table of `kind`.
    pub fn is_compatible_with(&self, kind: DescriptorKind) -> bool {
        match self {
            Self::Buffer(_) => kind != DescriptorKind::Sampler,
            Self::Texture(_) => matches!(kind, DescriptorKind::Srv | DescriptorKind::Uav),
            Self::Sampler(_) => kind == DescriptorKind::Sampler,
        }
    }

    /// The guest buffer range, when this is a buffer entry.
    pub fn as_buffer(&self) -> Option<&GuestBinding> {
        match self {
            Self::Buffer(binding) => Some(binding),
            Self::Texture(_) | Self::Sampler(_) => None,
        }
    }

    /// The registered resource handle, when this is a texture or sampler entry.
    pub fn resource_handle(&self) -> Option<u64> {
        match self {
            Self::Texture(handle) | Self::Sampler(handle) => Some(*handle),
            Self::Buffer(_) => None,
        }
    }

    /// A short name for error messages and diagnostics.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Buffer(_) => "buffer",
            Self::Texture(_) => "texture",
            Self::Sampler(_) => "sampler",
        }
    }
}

/// One guest descriptor slot as a table run reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorSlot {
    /// A live buffer descriptor and the kind its entry declares.
    Live(DescriptorKind, GuestBinding),
    /// A live texture descriptor and its registered texture handle.
    Texture(DescriptorKind, u64),
    /// A live sampler descriptor and its registered sampler handle.
    Sampler(u64),
    /// No live descriptor names this slot (freed, stale or never written).
    Empty,
    /// A live descriptor this backend cannot put in a table entry.
    Unresolvable(&'static str),
}

/// Why a descriptor-table run could not be enumerated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableRunError {
    /// The table handle is zero (`Set*RootDescriptorTable(null)`).
    HandleZero,
    /// The table's first slot holds no live descriptor.
    Empty,
    /// The first slot names a descriptor this backend cannot express.
    Unsupported(&'static str),
    /// A run of more than [`MAX_DESCRIPTOR_BINDINGS_V3`] same-kind descriptors
    /// was seen. It is refused instead of silently truncated.
    OverCap {
        /// Entries the run would need (the cap plus one at the time it is
        /// detected).
        count: usize,
    },
    /// A run changes descriptor kind inside one table. Wire v3 carries one
    /// kind per table, so the boundary must be a separate root parameter.
    MixedKind {
        /// Kind of the entries already enumerated.
        run_kind: DescriptorKind,
        /// Kind of the slot that ended the run.
        next_kind: DescriptorKind,
    },
}

impl std::fmt::Display for TableRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HandleZero => write!(f, "the descriptor-table handle is zero"),
            Self::Empty => write!(f, "the descriptor table's first slot holds no live descriptor"),
            Self::Unsupported(why) => write!(f, "the descriptor table entry is not expressible: {why}"),
            Self::OverCap { count } => write!(
                f,
                "the descriptor table run needs {count} entries, more than the {} a v3 table carries",
                MAX_DESCRIPTOR_BINDINGS_V3
            ),
            Self::MixedKind { run_kind, next_kind } => write!(
                f,
                "the descriptor table mixes {run_kind:?} and {next_kind:?} entries; \
                 bind each range as its own root descriptor table"
            ),
        }
    }
}

impl std::error::Error for TableRunError {}

/// Where a homogeneous descriptor-table run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableRunBoundary {
    /// The next slot held no live descriptor: the normal table end.
    Empty,
    /// The next slot held a descriptor this backend cannot express.
    Unresolvable(&'static str),
    /// The next slot held another `kind`: the boundary of the next root
    /// parameter's table (or a genuinely mixed range, which wire v3 cannot
    /// carry).
    KindChange {
        /// Kind of the slot that ended the run.
        kind: DescriptorKind,
    },
}

/// One homogeneous run of consecutive buffer descriptors in register order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorTableRun {
    /// The kind every entry carries.
    pub kind: DescriptorKind,
    /// The buffer entries, in register order.
    pub entries: Vec<GuestBinding>,
    /// Why the run stopped.
    pub boundary: TableRunBoundary,
}

impl DescriptorTableRun {
    /// Number of entries enumerated.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the run has no entries (never: a run always has at least one).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Refuses a run that ended on a different descriptor kind.
    ///
    /// The binding paths in `list.rs` stop at the kind change and bind the
    /// homogeneous prefix: a descriptor heap commonly holds tables of
    /// different kinds back to back, and each is named by its own
    /// `Set*RootDescriptorTable`, so a kind change is the normal boundary.
    /// A caller that knows one root parameter is a single mixed range can use
    /// this to refuse instead of binding its first range.
    pub fn require_homogeneous(&self) -> Result<(), TableRunError> {
        self.boundary.require_homogeneous(self.kind)
    }
}

impl TableRunBoundary {
    /// Refuses a boundary that is a descriptor-kind change.
    pub fn require_homogeneous(&self, run_kind: DescriptorKind) -> Result<(), TableRunError> {
        match self {
            Self::KindChange { kind } => Err(TableRunError::MixedKind {
                run_kind,
                next_kind: *kind,
            }),
            _ => Ok(()),
        }
    }
}

/// Enumerates the consecutive same-kind buffer descriptors of one table.
///
/// `read` returns the slot at a guest descriptor handle. The run starts at
/// `base_handle`, always includes that slot, and walks forward by the kind's
/// real per-entry increment ([`descriptor_increment`]) until it meets an empty
/// slot, an unresolvable slot or a slot of another kind. It never reads past
/// [`MAX_DESCRIPTOR_BINDINGS_V3`]: a 33rd same-kind live entry is
/// [`TableRunError::OverCap`], never a silent truncation.
///
/// This buffer-only form is kept for callers that predate registered-resource
/// entries. A sampler or texture slot is [`TableRunError::Unsupported`];
/// [`enumerate_descriptor_table_typed`] enumerates every entry shape.
pub fn enumerate_descriptor_table<F>(
    base_handle: u64,
    mut read: F,
) -> Result<DescriptorTableRun, TableRunError>
where
    F: FnMut(u64) -> DescriptorSlot,
{
    if base_handle == 0 {
        return Err(TableRunError::HandleZero);
    }
    let (kind, first) = match read(base_handle) {
        DescriptorSlot::Live(kind, binding) => (kind, binding),
        DescriptorSlot::Empty => return Err(TableRunError::Empty),
        DescriptorSlot::Unresolvable(why) => return Err(TableRunError::Unsupported(why)),
        DescriptorSlot::Texture(..) | DescriptorSlot::Sampler(_) => {
            return Err(TableRunError::Unsupported(
                "the buffer-only enumerator cannot carry a registered resource entry",
            ));
        }
    };
    if kind == DescriptorKind::Sampler {
        return Err(TableRunError::Unsupported(
            "sampler descriptor tables have no buffer binding",
        ));
    }
    let increment = descriptor_increment(kind);
    let mut entries = Vec::with_capacity(4);
    entries.push(first);
    let mut distance = increment;
    let boundary = loop {
        let Some(next_handle) = base_handle.checked_add(distance) else {
            break TableRunBoundary::Empty;
        };
        match read(next_handle) {
            DescriptorSlot::Empty => break TableRunBoundary::Empty,
            DescriptorSlot::Unresolvable(why) => break TableRunBoundary::Unresolvable(why),
            DescriptorSlot::Texture(..) | DescriptorSlot::Sampler(_) => {
                break TableRunBoundary::Unresolvable(
                    "the buffer-only enumerator cannot carry a registered resource entry",
                );
            }
            DescriptorSlot::Live(next_kind, binding) => {
                if next_kind != kind {
                    break TableRunBoundary::KindChange { kind: next_kind };
                }
                if entries.len() >= MAX_DESCRIPTOR_BINDINGS_V3 {
                    return Err(TableRunError::OverCap {
                        count: MAX_DESCRIPTOR_BINDINGS_V3 + 1,
                    });
                }
                entries.push(binding);
                distance += increment;
            }
        }
    };
    Ok(DescriptorTableRun {
        kind,
        entries,
        boundary,
    })
}

/// One homogeneous run of consecutive typed descriptors in register order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedDescriptorTableRun {
    /// The kind every entry carries.
    pub kind: DescriptorKind,
    /// The entries (buffer, texture or sampler), in register order.
    pub entries: Vec<DescriptorEntry>,
    /// Why the run stopped.
    pub boundary: TableRunBoundary,
}

impl TypedDescriptorTableRun {
    /// Number of entries enumerated.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the run has no entries (never: a run always has at least one).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Refuses a run that ended on a different descriptor kind.
    pub fn require_homogeneous(&self) -> Result<(), TableRunError> {
        self.boundary.require_homogeneous(self.kind)
    }
}

/// Enumerates the consecutive same-kind descriptors of one table, keeping each
/// entry's shape (buffer range, registered texture handle or sampler handle).
///
/// Sampler tables advance at the 16-byte sampler increment and view tables at
/// the 32-byte view increment. A first slot that is empty, unresolvable or
/// whose entry shape the kind cannot express is an error. A later slot of the
/// same kind is included even when its entry shape differs, so
/// [`descriptor_table_typed`] can refuse the mixed table explicitly instead of
/// silently binding a prefix.
pub fn enumerate_descriptor_table_typed<F>(
    base_handle: u64,
    mut read: F,
) -> Result<TypedDescriptorTableRun, TableRunError>
where
    F: FnMut(u64) -> DescriptorSlot,
{
    if base_handle == 0 {
        return Err(TableRunError::HandleZero);
    }
    let (kind, first) = match read(base_handle) {
        DescriptorSlot::Live(kind, binding) => (kind, DescriptorEntry::Buffer(binding)),
        DescriptorSlot::Texture(kind, handle) => (kind, DescriptorEntry::Texture(handle)),
        DescriptorSlot::Sampler(handle) => {
            (DescriptorKind::Sampler, DescriptorEntry::Sampler(handle))
        }
        DescriptorSlot::Empty => return Err(TableRunError::Empty),
        DescriptorSlot::Unresolvable(why) => return Err(TableRunError::Unsupported(why)),
    };
    if !first.is_compatible_with(kind) {
        return Err(TableRunError::Unsupported(
            "the descriptor table's first entry shape does not match its kind",
        ));
    }
    let increment = descriptor_increment(kind);
    let mut entries = Vec::with_capacity(4);
    entries.push(first);
    let mut distance = increment;
    let boundary = loop {
        let Some(next_handle) = base_handle.checked_add(distance) else {
            break TableRunBoundary::Empty;
        };
        match read(next_handle) {
            DescriptorSlot::Empty => break TableRunBoundary::Empty,
            DescriptorSlot::Unresolvable(why) => break TableRunBoundary::Unresolvable(why),
            DescriptorSlot::Live(next_kind, binding) => {
                if next_kind != kind {
                    break TableRunBoundary::KindChange { kind: next_kind };
                }
                if entries.len() >= MAX_DESCRIPTOR_BINDINGS_V3 {
                    return Err(TableRunError::OverCap {
                        count: MAX_DESCRIPTOR_BINDINGS_V3 + 1,
                    });
                }
                entries.push(DescriptorEntry::Buffer(binding));
                distance += increment;
            }
            DescriptorSlot::Texture(next_kind, handle) => {
                if next_kind != kind {
                    break TableRunBoundary::KindChange { kind: next_kind };
                }
                if entries.len() >= MAX_DESCRIPTOR_BINDINGS_V3 {
                    return Err(TableRunError::OverCap {
                        count: MAX_DESCRIPTOR_BINDINGS_V3 + 1,
                    });
                }
                entries.push(DescriptorEntry::Texture(handle));
                distance += increment;
            }
            DescriptorSlot::Sampler(handle) => {
                if kind != DescriptorKind::Sampler {
                    break TableRunBoundary::KindChange {
                        kind: DescriptorKind::Sampler,
                    };
                }
                if entries.len() >= MAX_DESCRIPTOR_BINDINGS_V3 {
                    return Err(TableRunError::OverCap {
                        count: MAX_DESCRIPTOR_BINDINGS_V3 + 1,
                    });
                }
                entries.push(DescriptorEntry::Sampler(handle));
                distance += increment;
            }
        }
    };
    Ok(TypedDescriptorTableRun {
        kind,
        entries,
        boundary,
    })
}

/// A Wire v3 entry together with the pins its ranges hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved<T> {
    /// The entry to record into the producer.
    pub entry: T,
    /// Tokens returned by [`GuestRangeResolver::retain`], one per range.
    pub tokens: Vec<u64>,
}

impl<T> Resolved<T> {
    /// Splits the entry from its pins.
    pub fn into_parts(self) -> (T, Vec<u64>) {
        (self.entry, self.tokens)
    }
}

/// Rejects ranges that cannot name a live share allocation.
fn checked_range(range: &GuestRange) -> Result<(), ResolveError> {
    if range.allocation_id == 0 {
        return Err(ResolveError::Invalid("the range names no share allocation"));
    }
    if range.generation == 0 {
        return Err(ResolveError::Invalid("the range's generation is zero"));
    }
    if range.size == 0 {
        return Err(ResolveError::Invalid("the range's backing window is empty"));
    }
    if !range.byte_offset.is_multiple_of(PAGE) {
        return Err(ResolveError::Invalid(
            "the backing window does not start on a share page",
        ));
    }
    if !range.size.is_multiple_of(PAGE) {
        return Err(ResolveError::Invalid(
            "the backing window is not a whole number of share pages",
        ));
    }
    Ok(())
}

/// Rejects a generation the host's kind 2 registry does not carry.
///
/// The host registers every allocation as generation 1
/// (`victory_metal::guest_ipc::REGISTRATION_GENERATION`) and validates every
/// binding against it, so a range the provider reports with another generation
/// is refused here instead of producing a submission the host must reject.
fn check_generation(range: &GuestRange) -> Result<(), ResolveError> {
    if range.generation != victory_metal::guest_ipc::REGISTRATION_GENERATION {
        return Err(ResolveError::Unsupported(
            "the range's allocation generation is not the host's kind 2 generation",
        ));
    }
    Ok(())
}

/// The wire byte offset of `[address, address + len)` inside `range`.
fn binding_offset(range: &GuestRange, address: u64, len: u64) -> Result<u64, ResolveError> {
    let within = address & (PAGE - 1);
    let end = within
        .checked_add(len)
        .ok_or(ResolveError::Invalid("the guest range overflows"))?;
    if end > range.size {
        return Err(ResolveError::Invalid(
            "the guest range leaves the backing window it resolved to",
        ));
    }
    range
        .byte_offset
        .checked_add(within)
        .ok_or(ResolveError::Invalid("the guest range overflows"))
}

/// Resolves and pins the range a root CBV/SRV/UAV names.
///
/// The guest call carries only the GPU virtual address; `len` is the caller's
/// knowledge of what the shader may read. The returned entry's offset is the
/// exact in-allocation byte offset of `address`, so an unaligned address is
/// preserved rather than rounded down to its page.
pub fn root_buffer_view(
    resolver: &dyn GuestRangeResolver,
    slot: u16,
    address: u64,
    len: u64,
) -> Result<Resolved<RootBufferView>, ResolveError> {
    if address == 0 {
        return Err(ResolveError::Invalid("the root view address is zero"));
    }
    if len == 0 {
        return Err(ResolveError::Invalid("the root view length is zero"));
    }
    if len > MAX_RESOURCE_LEN {
        return Err(ResolveError::Unsupported("the root view is too large"));
    }
    let range = resolver.resolve(address, len)?;
    checked_range(&range)?;
    let offset = binding_offset(&range, address, len)?;
    let token = resolver.retain(&range)?;
    Ok(Resolved {
        entry: RootBufferView {
            slot,
            allocation_id: range.allocation_id,
            generation: range.generation,
            offset,
            length: len,
        },
        tokens: vec![token],
    })
}

/// Resolves and pins a root view whose guest call carries no byte length
/// (D3D12's `SetGraphicsRoot*View` passes only the GPU virtual address).
///
/// The view covers the rest of the page window the address resolves into, and
/// its offset is the exact in-allocation byte offset of the address.
pub fn root_buffer_view_page(
    resolver: &dyn GuestRangeResolver,
    slot: u16,
    address: u64,
) -> Result<Resolved<RootBufferView>, ResolveError> {
    if address == 0 {
        return Err(ResolveError::Invalid("the root view address is zero"));
    }
    let range = resolver.resolve(address, 1)?;
    checked_range(&range)?;
    let within = address & (PAGE - 1);
    let len = range
        .size
        .checked_sub(within)
        .ok_or(ResolveError::Invalid(
            "the root view address leaves the backing window",
        ))?;
    if len == 0 {
        return Err(ResolveError::Invalid("the root view has no backing bytes"));
    }
    if len > MAX_RESOURCE_LEN {
        return Err(ResolveError::Unsupported("the root view is too large"));
    }
    let offset = binding_offset(&range, address, len)?;
    let token = resolver.retain(&range)?;
    Ok(Resolved {
        entry: RootBufferView {
            slot,
            allocation_id: range.allocation_id,
            generation: range.generation,
            offset,
            length: len,
        },
        tokens: vec![token],
    })
}

/// Resolves and pins one homogeneous descriptor table of buffer ranges.
///
/// This is the buffer-only form kept for callers that already enumerated
/// [`GuestBinding`]s; [`descriptor_table_typed`] is the full encoder for
/// buffer, registered-texture and registered-sampler entries.
pub fn descriptor_table(
    resolver: &dyn GuestRangeResolver,
    handle: u64,
    set: u16,
    base_register: u16,
    kind: DescriptorKind,
    descriptors: &[GuestBinding],
) -> Result<Resolved<DescriptorTableV3>, ResolveError> {
    let entries: Vec<DescriptorEntry> = descriptors
        .iter()
        .copied()
        .map(DescriptorEntry::Buffer)
        .collect();
    descriptor_table_typed(resolver, handle, set, base_register, kind, &entries)
}

/// Resolves and pins one homogeneous descriptor table.
///
/// `handle` is the D3D12 GPU descriptor handle of the table's first entry; it
/// is validated (non-zero) but the Wire frame carries the resolved entries, not
/// the handle. `entries` are the table entries in register order, as the caller
/// read them from guest descriptor memory.
///
/// Wire v3 encodes one entry tag per table: all buffers (legacy 32-byte
/// encoding), all registered texture handles, or all registered sampler
/// handles. A table that mixes entry shapes, or whose entry shape the reflected
/// range kind cannot express (a texture in a CBV table, a sampler in a view
/// table, a buffer in a sampler table), is refused instead of being split,
/// truncated or written with a guessed tag.
pub fn descriptor_table_typed(
    resolver: &dyn GuestRangeResolver,
    handle: u64,
    set: u16,
    base_register: u16,
    kind: DescriptorKind,
    entries: &[DescriptorEntry],
) -> Result<Resolved<DescriptorTableV3>, ResolveError> {
    if handle == 0 {
        return Err(ResolveError::Invalid("the descriptor-table handle is zero"));
    }
    if entries.is_empty() {
        return Err(ResolveError::Invalid("the descriptor table is empty"));
    }
    if entries.len() > MAX_DESCRIPTOR_BINDINGS_V3 {
        return Err(ResolveError::Unsupported(
            "the descriptor table has more bindings than a v3 table carries",
        ));
    }

    let mut shape: Option<&'static str> = None;
    for entry in entries {
        if !entry.is_compatible_with(kind) {
            return Err(ResolveError::Unsupported(
                "the descriptor table entry shape does not match the reflected range kind",
            ));
        }
        let name = entry.kind_name();
        match shape {
            Some(existing) if existing != name => {
                return Err(ResolveError::Unsupported(
                    "the descriptor table mixes buffer and registered-resource entries, \
                     which one v3 entry tag cannot carry",
                ));
            }
            _ => shape = Some(name),
        }
    }

    let mut bindings = Vec::with_capacity(entries.len());
    let mut tokens = Vec::with_capacity(entries.len());
    for entry in entries {
        let encoded = match entry {
            DescriptorEntry::Buffer(descriptor) => {
                let resolved = match resolve_descriptor(resolver, descriptor) {
                    Ok(resolved) => resolved,
                    Err(error) => {
                        retire_all(resolver, &tokens);
                        return Err(error);
                    }
                };
                tokens.push(resolved.1);
                DescriptorTableEntryV3::Buffer(resolved.0)
            }
            DescriptorEntry::Texture(handle) => {
                if *handle == 0 {
                    retire_all(resolver, &tokens);
                    return Err(ResolveError::Invalid(
                        "a texture table entry's registered handle is zero",
                    ));
                }
                DescriptorTableEntryV3::Texture(*handle)
            }
            DescriptorEntry::Sampler(handle) => {
                if *handle == 0 {
                    retire_all(resolver, &tokens);
                    return Err(ResolveError::Invalid(
                        "a sampler table entry's registered handle is zero",
                    ));
                }
                DescriptorTableEntryV3::Sampler(*handle)
            }
        };
        bindings.push(encoded);
    }
    Ok(Resolved {
        entry: DescriptorTableV3 {
            set,
            base_register,
            kind,
            bindings,
        },
        tokens,
    })
}

/// Why a table names a handle this session did not register.
///
/// This is refused before the producer records the table, so an unregistered
/// handle can never reach a native submission (and never becomes a default
/// resource or sampler).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleError {
    /// No texture registration accepted this handle in the session.
    UnregisteredTexture(u64),
    /// No sampler registration accepted this handle in the session.
    UnregisteredSampler(u64),
}

impl std::fmt::Display for HandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnregisteredTexture(handle) => write!(
                f,
                "descriptor table names unregistered texture handle 0x{handle:016x}"
            ),
            Self::UnregisteredSampler(handle) => write!(
                f,
                "descriptor table names unregistered sampler handle 0x{handle:016x}"
            ),
        }
    }
}

impl std::error::Error for HandleError {}

/// Checks every registered-resource entry against the session's registrations.
///
/// Buffer entries are validated by the host's allocation registry and are not
/// checked here. The two predicates are the live client's per-session sets, so
/// a handle from another session (or a hand-built one) is an explicit
/// [`HandleError`] rather than a silent default.
pub fn require_registered_handles(
    table: &DescriptorTableV3,
    texture_registered: impl Fn(u64) -> bool,
    sampler_registered: impl Fn(u64) -> bool,
) -> Result<(), HandleError> {
    for entry in &table.bindings {
        match entry {
            DescriptorTableEntryV3::Buffer(_) => {}
            DescriptorTableEntryV3::Texture(handle) => {
                if !texture_registered(*handle) {
                    return Err(HandleError::UnregisteredTexture(*handle));
                }
            }
            DescriptorTableEntryV3::Sampler(handle) => {
                if !sampler_registered(*handle) {
                    return Err(HandleError::UnregisteredSampler(*handle));
                }
            }
        }
    }
    Ok(())
}

/// Resolves, validates and pins one table entry.
fn resolve_descriptor(
    resolver: &dyn GuestRangeResolver,
    descriptor: &GuestBinding,
) -> Result<(BufferBinding, u64), ResolveError> {
    if descriptor.address == 0 {
        return Err(ResolveError::Invalid("a descriptor's address is zero"));
    }
    if descriptor.len == 0 {
        return Err(ResolveError::Invalid("a descriptor's length is zero"));
    }
    if descriptor.len > MAX_RESOURCE_LEN {
        return Err(ResolveError::Unsupported("a descriptor range is too large"));
    }
    let range = resolver.resolve(descriptor.address, descriptor.len)?;
    checked_range(&range)?;
    let offset = binding_offset(&range, descriptor.address, descriptor.len)?;
    let token = resolver.retain(&range)?;
    Ok((
        BufferBinding {
            allocation_id: range.allocation_id,
            generation: range.generation,
            offset,
            length: descriptor.len,
        },
        token,
    ))
}

/// Best-effort rollback of pins taken for a binding that was then refused.
///
/// Retirement failures are ignored here: the primary error is the one the
/// caller reports, and the provider's registry makes a duplicate or unknown
/// retirement visible to the queue that owns the established pins.
pub fn retire_all(resolver: &dyn GuestRangeResolver, tokens: &[u64]) {
    for &token in tokens {
        let _ = resolver.retire(token);
    }
}

/// Builds the inline root-constant entry the Wire v3 stream carries.
///
/// Root constants are guest values, not guest addresses: an all-zero value is
/// valid, but an empty or oversized constant block is not.
pub fn root_constants(slot: u16, bytes: &[u8]) -> Result<RootConstantsV3, ResolveError> {
    if bytes.is_empty() {
        return Err(ResolveError::Invalid("the root constant block is empty"));
    }
    if bytes.len() > MAX_ROOT_CONSTANT_BYTES_V3 {
        return Err(ResolveError::Unsupported(
            "the root constant block is larger than a v3 command carries",
        ));
    }
    Ok(RootConstantsV3 {
        slot,
        bytes: bytes.to_vec(),
    })
}

/// Resolves and pins one D3D12 vertex-buffer view (`IASetVertexBuffers`).
///
/// The title passes the GPU virtual address of the first element and the byte
/// length visible to the shader; the Wire v3 binding carries the share
/// allocation, the exact in-allocation offset and that length, so the host can
/// validate the range against its kind 2 registration. The allocation id and
/// generation come from the resolver, never from the address.
pub fn vertex_buffer(
    resolver: &dyn GuestRangeResolver,
    address: u64,
    len: u64,
    stride: u32,
) -> Result<Resolved<VertexBuffer>, ResolveError> {
    if address == 0 {
        return Err(ResolveError::Invalid("the vertex buffer address is zero"));
    }
    if len == 0 {
        return Err(ResolveError::Invalid("the vertex buffer length is zero"));
    }
    if stride == 0 {
        return Err(ResolveError::Invalid("the vertex buffer stride is zero"));
    }
    if len > MAX_RESOURCE_LEN {
        return Err(ResolveError::Unsupported("the vertex buffer is too large"));
    }
    let range = resolver.resolve(address, len)?;
    checked_range(&range)?;
    check_generation(&range)?;
    let offset = binding_offset(&range, address, len)?;
    let token = resolver.retain(&range)?;
    Ok(Resolved {
        entry: VertexBuffer {
            allocation_id: range.allocation_id,
            generation: range.generation,
            offset,
            stride,
            length: len,
        },
        tokens: vec![token],
    })
}

/// Resolves and pins one D3D12 index-buffer view (`IASetIndexBuffer`).
///
/// `format` is the wire's element width, converted by the caller from the DXGI
/// format number; a range the title names but the backing window does not cover
/// is an explicit error, never a clamped binding.
pub fn index_buffer(
    resolver: &dyn GuestRangeResolver,
    address: u64,
    len: u64,
    format: IndexFormat,
) -> Result<Resolved<IndexBufferBinding>, ResolveError> {
    if address == 0 {
        return Err(ResolveError::Invalid("the index buffer address is zero"));
    }
    if len == 0 {
        return Err(ResolveError::Invalid("the index buffer length is zero"));
    }
    if len > MAX_RESOURCE_LEN {
        return Err(ResolveError::Unsupported("the index buffer is too large"));
    }
    let range = resolver.resolve(address, len)?;
    checked_range(&range)?;
    check_generation(&range)?;
    let offset = binding_offset(&range, address, len)?;
    let token = resolver.retain(&range)?;
    Ok(Resolved {
        entry: IndexBufferBinding {
            allocation_id: range.allocation_id,
            generation: range.generation,
            offset,
            length: len,
            format,
        },
        tokens: vec![token],
    })
}

/// [`root_constants`] for a title's 32-bit words, little-endian as the wire
/// requires.
pub fn root_constants_words(slot: u16, words: &[u32]) -> Result<RootConstantsV3, ResolveError> {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    root_constants(slot, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use victory_metal::wire_v3::DescriptorKind;

    /// A resolver over one synthetic share allocation of `pages` pages.
    struct FakeResolver {
        base: u64,
        allocation_id: u64,
        generation: u32,
        pages: u64,
        retained: Mutex<HashMap<u64, GuestRange>>,
        pins: AtomicU64,
        retires: AtomicU64,
    }

    impl FakeResolver {
        fn new() -> Self {
            Self {
                base: 0x1_0000_0000,
                allocation_id: 7,
                generation: 1,
                pages: 4,
                retained: Mutex::new(HashMap::new()),
                pins: AtomicU64::new(1),
                retires: AtomicU64::new(0),
            }
        }

        fn page_address(&self, page: u64) -> u64 {
            self.base + page * PAGE
        }

        fn live(&self) -> usize {
            self.retained.lock().unwrap_or_else(|e| e.into_inner()).len()
        }
    }

    impl GuestRangeResolver for FakeResolver {
        fn resolve(&self, guest_address: u64, len: u64) -> Result<GuestRange, ResolveError> {
            if guest_address == 0 || len == 0 {
                return Err(ResolveError::Invalid("zero address or length"));
            }
            if guest_address < self.base {
                return Err(ResolveError::Unmapped {
                    address: guest_address,
                    len,
                });
            }
            let first = (guest_address - self.base) / PAGE;
            let last = (guest_address - self.base + len - 1) / PAGE;
            if last >= self.pages {
                return Err(ResolveError::Unmapped {
                    address: guest_address,
                    len,
                });
            }
            Ok(GuestRange {
                allocation_id: self.allocation_id,
                generation: self.generation,
                byte_offset: first * PAGE,
                size: (last - first + 1) * PAGE,
            })
        }

        fn retain(&self, range: &GuestRange) -> Result<u64, ResolveError> {
            let token = self.pins.fetch_add(1, Ordering::Relaxed);
            self.retained
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(token, *range);
            Ok(token)
        }

        fn retire(&self, token: u64) -> Result<(), ResolveError> {
            self.retires.fetch_add(1, Ordering::Relaxed);
            if self
                .retained
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&token)
                .is_some()
            {
                Ok(())
            } else {
                Err(ResolveError::Retire("the token is unknown or already retired"))
            }
        }
    }

    #[test]
    fn unaligned_addresses_resolve_to_a_page_window_and_exact_offsets() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);
        let address = page + 0x123;

        let resolved = root_buffer_view(&resolver, 3, address, 0x456).expect("an unaligned view");
        let (view, tokens) = resolved.into_parts();
        assert_eq!(view.allocation_id, 7);
        assert_eq!(view.generation, 1);
        assert_eq!(view.slot, 3);
        assert_eq!(view.offset, 0x123, "the in-page byte is preserved");
        assert_eq!(view.length, 0x456);
        assert_eq!(tokens.len(), 1);
        assert_eq!(resolver.live(), 1);

        // The same range resolved directly is the page-aligned window.
        let range = resolver.resolve(address, 0x456).expect("the resolved range");
        assert_eq!(
            range,
            GuestRange {
                allocation_id: 7,
                generation: 1,
                byte_offset: 0,
                size: PAGE,
            }
        );

        // A root view whose guest call carries no length covers the rest of
        // its page window, still at the exact offset.
        let window = root_buffer_view_page(&resolver, 1, address)
            .expect("a root view with no guest length");
        assert_eq!(window.entry.offset, 0x123);
        assert_eq!(window.entry.length, PAGE - 0x123);
        assert_eq!(window.tokens.len(), 1);
        assert_eq!(resolver.live(), 2);
        retire_all(&resolver, &window.tokens);

        // A range that starts in page 0 and reaches page 1 resolves to both.
        let spanning = resolver
            .resolve(page + PAGE - 0x10, 0x20)
            .expect("a spanning range");
        assert_eq!(
            (spanning.byte_offset, spanning.size),
            (0, 2 * PAGE),
            "the covered window is page rounded"
        );

        resolver.retire(tokens[0]).expect("retire the pin");
        assert_eq!(resolver.live(), 0);
    }

    #[test]
    fn descriptor_tables_bind_every_entry_at_its_exact_offset() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);
        let descriptors = [
            GuestBinding {
                address: page + 0x20,
                len: 0x100,
            },
            GuestBinding {
                address: page + PAGE + 0x10,
                len: 0x40,
            },
        ];
        let resolved = descriptor_table(
            &resolver,
            0xdead_0000,
            2,
            5,
            DescriptorKind::Cbv,
            &descriptors,
        )
        .expect("a two-entry table");
        let (table, tokens) = resolved.into_parts();
        assert_eq!((table.set, table.base_register), (2, 5));
        assert_eq!(table.kind, DescriptorKind::Cbv);
        assert_eq!(table.bindings.len(), 2);
        let first = table.bindings[0].as_buffer().expect("first buffer entry");
        let second = table.bindings[1].as_buffer().expect("second buffer entry");
        assert_eq!((first.offset, first.length), (0x20, 0x100));
        assert_eq!((second.offset, second.length), (PAGE + 0x10, 0x40));
        assert_eq!(tokens.len(), 2);
        assert_eq!(resolver.live(), 2);

        retire_all(&resolver, &tokens);
        assert_eq!(resolver.live(), 0);
        assert_eq!(resolver.retires.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn zero_addresses_handles_and_lengths_are_rejected_before_resolving() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);

        assert!(matches!(
            root_buffer_view(&resolver, 0, 0, 16),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            root_buffer_view(&resolver, 0, page, 0),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            descriptor_table(&resolver, 0, 0, 0, DescriptorKind::Srv, &[
                GuestBinding {
                    address: page,
                    len: 16
                }
            ]),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            descriptor_table(&resolver, 0x100, 0, 0, DescriptorKind::Srv, &[
                GuestBinding {
                    address: 0,
                    len: 16
                }
            ]),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            descriptor_table(&resolver, 0x100, 0, 0, DescriptorKind::Srv, &[
                GuestBinding {
                    address: page,
                    len: 0
                }
            ]),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            descriptor_table(&resolver, 0x100, 0, 0, DescriptorKind::Srv, &[]),
            Err(ResolveError::Invalid(_))
        ));
        assert_eq!(resolver.live(), 0, "a refusal pins nothing");
    }

    #[test]
    fn unmapped_ranges_are_explicit_errors() {
        let resolver = FakeResolver::new();
        // One page past the allocation.
        let outside = resolver.page_address(resolver.pages);
        assert_eq!(
            root_buffer_view(&resolver, 0, outside, 1).unwrap_err(),
            ResolveError::Unmapped {
                address: outside,
                len: 1
            }
        );
        // A range that starts inside but ends outside the last page.
        let late = resolver.page_address(resolver.pages - 1) + PAGE - 8;
        assert_eq!(
            root_buffer_view(&resolver, 0, late, 16).unwrap_err(),
            ResolveError::Unmapped {
                address: late,
                len: 16
            }
        );
        assert_eq!(resolver.live(), 0);
    }

    #[test]
    fn unsupported_kinds_and_oversized_blocks_are_explicit_errors() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);

        assert!(matches!(
            descriptor_table(
                &resolver,
                0x100,
                0,
                0,
                DescriptorKind::Sampler,
                &[GuestBinding {
                    address: page,
                    len: 16
                }]
            ),
            Err(ResolveError::Unsupported(_))
        ));

        let too_many: Vec<GuestBinding> = (0..=MAX_DESCRIPTOR_BINDINGS_V3)
            .map(|_| GuestBinding {
                address: page,
                len: 16,
            })
            .collect();
        assert!(matches!(
            descriptor_table(&resolver, 0x100, 0, 0, DescriptorKind::Srv, &too_many),
            Err(ResolveError::Unsupported(_))
        ));

        assert!(matches!(
            root_constants(0, &[]),
            Err(ResolveError::Invalid(_))
        ));
        assert!(matches!(
            root_constants(0, &vec![0u8; MAX_ROOT_CONSTANT_BYTES_V3 + 1]),
            Err(ResolveError::Unsupported(_))
        ));
        let constants = root_constants_words(4, &[0x1122_3344, 0]).expect("root constants");
        assert_eq!(constants.slot, 4);
        assert_eq!(constants.bytes, [0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0]);
        assert_eq!(resolver.live(), 0);
    }

    #[test]
    fn a_failed_later_descriptor_rolls_back_the_earlier_pins() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);
        let outside = resolver.page_address(resolver.pages);
        let descriptors = [
            GuestBinding {
                address: page,
                len: 16,
            },
            GuestBinding {
                address: outside,
                len: 16,
            },
        ];
        assert!(matches!(
            descriptor_table(&resolver, 0x100, 0, 0, DescriptorKind::Uav, &descriptors),
            Err(ResolveError::Unmapped { .. })
        ));
        assert_eq!(resolver.live(), 0, "the first entry's pin was released");
        assert_eq!(resolver.retires.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stale_generations_are_explicit_errors() {
        let mut resolver = FakeResolver::new();
        resolver.generation = 0;
        let page = resolver.page_address(0);
        assert_eq!(
            root_buffer_view(&resolver, 0, page, 16).unwrap_err(),
            ResolveError::Invalid("the range's generation is zero")
        );
        assert_eq!(resolver.live(), 0);
    }

    #[test]
    fn retire_is_single_use_and_double_retire_is_refused() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);
        let token = resolver
            .retain(&resolver.resolve(page, 16).expect("a range"))
            .expect("retain");
        resolver.retire(token).expect("the first retirement");
        assert!(matches!(
            resolver.retire(token),
            Err(ResolveError::Retire(_))
        ));
        assert_eq!(resolver.live(), 0);
        assert_eq!(resolver.retires.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_installed_resolver_is_process_wide() {
        // Tests run with `--test-threads=1`, and this guard keeps a parallel
        // run from replacing another test's resolver.
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver();
        assert!(resolver().is_none());
        set_resolver(FakeResolver::new());
        let installed = resolver().expect("the installed resolver");
        let page = 0x1_0000_0000;
        assert!(installed.resolve(page, 16).is_ok());
        let replaced = clear_resolver().expect("the installed resolver");
        assert!(Arc::ptr_eq(&replaced, &installed));
        assert!(resolver().is_none());
    }

    /// A synthetic guest descriptor heap: entry `i` lives at
    /// `TABLE_BASE + i * VIEW_DESCRIPTOR_INCREMENT`.
    const TABLE_BASE: u64 = 0x2000_0000;

    fn binding_for(index: usize) -> GuestBinding {
        GuestBinding {
            address: 0x1_0000 + index as u64 * 0x100,
            len: 16,
        }
    }

    fn entry_for(index: usize) -> DescriptorEntry {
        DescriptorEntry::Buffer(binding_for(index))
    }

    /// Enumerates a heap whose first `kinds.len()` slots are `kinds[i]` and
    /// whose following slot is empty unless `unresolvable` names it.
    fn enumerate_heap(
        kinds: &[Option<DescriptorKind>],
        unresolvable: Option<(usize, &'static str)>,
    ) -> Result<DescriptorTableRun, TableRunError> {
        let slots: Vec<DescriptorSlot> = kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| match kind {
                Some(kind) => DescriptorSlot::Live(*kind, binding_for(index)),
                None => DescriptorSlot::Empty,
            })
            .collect();
        enumerate_descriptor_table(TABLE_BASE, |handle| {
            let index = ((handle - TABLE_BASE) / VIEW_DESCRIPTOR_INCREMENT) as usize;
            if let Some((at, why)) = unresolvable
                && at == index
            {
                return DescriptorSlot::Unresolvable(why);
            }
            slots.get(index).copied().unwrap_or(DescriptorSlot::Empty)
        })
    }

    #[test]
    fn descriptor_runs_enumerate_one_two_four_and_thirty_two_entries() {
        for count in [1usize, 2, 4, MAX_DESCRIPTOR_BINDINGS_V3] {
            let mut kinds = vec![Some(DescriptorKind::Cbv); count];
            kinds.push(None);
            let run = enumerate_heap(&kinds, None).expect("a homogeneous table");
            assert_eq!(run.kind, DescriptorKind::Cbv, "run kind");
            assert_eq!(run.len(), count, "entry count");
            assert_eq!(run.boundary, TableRunBoundary::Empty, "table end");
            assert_eq!(run.entries[0], binding_for(0), "first entry");
            assert_eq!(
                run.entries[count - 1],
                binding_for(count - 1),
                "last entry keeps register order"
            );
            run.require_homogeneous().expect("no kind change");
        }
    }

    #[test]
    fn a_thirty_third_same_kind_entry_is_refused_not_truncated() {
        let kinds = vec![Some(DescriptorKind::Srv); MAX_DESCRIPTOR_BINDINGS_V3 + 1];
        assert_eq!(
            enumerate_heap(&kinds, None),
            Err(TableRunError::OverCap {
                count: MAX_DESCRIPTOR_BINDINGS_V3 + 1
            })
        );
        // The message names the cap instead of silently dropping entry 33.
        let message = TableRunError::OverCap { count: 33 }.to_string();
        assert!(message.contains("33"), "{message}");
    }

    #[test]
    fn an_empty_second_slot_leaves_a_one_entry_run() {
        let run = enumerate_heap(&[Some(DescriptorKind::Srv), None], None)
            .expect("a one-entry table followed by an empty slot");
        assert_eq!(run.len(), 1);
        assert_eq!(run.entries[0], binding_for(0));
        assert_eq!(run.boundary, TableRunBoundary::Empty);
    }

    #[test]
    fn a_kind_change_stops_the_run_and_a_mixed_run_is_refused() {
        let run = enumerate_heap(
            &[
                Some(DescriptorKind::Cbv),
                Some(DescriptorKind::Cbv),
                Some(DescriptorKind::Srv),
                Some(DescriptorKind::Srv),
            ],
            None,
        )
        .expect("the homogeneous prefix");
        assert_eq!(run.kind, DescriptorKind::Cbv);
        assert_eq!(run.len(), 2, "the next root parameter's table is not consumed");
        assert_eq!(
            run.boundary,
            TableRunBoundary::KindChange {
                kind: DescriptorKind::Srv
            }
        );
        let error = run
            .require_homogeneous()
            .expect_err("a single mixed range cannot be carried by one v3 table");
        assert_eq!(
            error,
            TableRunError::MixedKind {
                run_kind: DescriptorKind::Cbv,
                next_kind: DescriptorKind::Srv,
            }
        );
        let message = error.to_string();
        assert!(message.contains("own root descriptor table"), "{message}");
    }

    #[test]
    fn an_unresolvable_slot_stops_the_run() {
        let run = enumerate_heap(
            &[
                Some(DescriptorKind::Uav),
                Some(DescriptorKind::Uav),
                Some(DescriptorKind::Uav),
            ],
            Some((2, "a texture descriptor cannot be carried by a buffer table")),
        )
        .expect("the resolvable prefix");
        assert_eq!(run.len(), 2);
        assert_eq!(
            run.boundary,
            TableRunBoundary::Unresolvable(
                "a texture descriptor cannot be carried by a buffer table"
            )
        );
        run.require_homogeneous()
            .expect("an unresolvable end is not a kind change");
    }

    #[test]
    fn a_zero_handle_or_empty_first_slot_is_refused() {
        assert_eq!(
            enumerate_descriptor_table(0, |_| DescriptorSlot::Empty),
            Err(TableRunError::HandleZero)
        );
        assert_eq!(
            enumerate_descriptor_table(TABLE_BASE, |_| DescriptorSlot::Empty),
            Err(TableRunError::Empty)
        );
        assert_eq!(
            enumerate_descriptor_table(TABLE_BASE, |_| {
                DescriptorSlot::Unresolvable("a sampler descriptor has no host equivalent")
            }),
            Err(TableRunError::Unsupported(
                "a sampler descriptor has no host equivalent"
            ))
        );
    }

    #[test]
    fn a_sampler_table_enumerates_registered_sampler_handles() {
        const HANDLE: u64 = 0x5300_0000_0000_0042;
        let run = enumerate_descriptor_table_typed(TABLE_BASE, |handle| {
            if handle == TABLE_BASE {
                DescriptorSlot::Sampler(HANDLE)
            } else {
                DescriptorSlot::Empty
            }
        })
        .expect("a one-entry sampler table");
        assert_eq!(run.kind, DescriptorKind::Sampler);
        assert_eq!(run.len(), 1);
        assert_eq!(run.entries[0], DescriptorEntry::Sampler(HANDLE));
        assert_eq!(run.boundary, TableRunBoundary::Empty);
        assert_eq!(
            descriptor_increment(DescriptorKind::Sampler),
            SAMPLER_DESCRIPTOR_INCREMENT,
            "the sampler run advances at the sampler increment"
        );
        // The buffer-only enumerator still refuses a sampler slot.
        assert!(matches!(
            enumerate_descriptor_table(TABLE_BASE, |_| DescriptorSlot::Sampler(HANDLE)),
            Err(TableRunError::Unsupported(_))
        ));
    }

    #[test]
    fn an_entry_shape_the_kind_cannot_represent_is_refused() {
        // A buffer entry in a sampler table.
        assert!(matches!(
            enumerate_descriptor_table_typed(TABLE_BASE, |_| {
                DescriptorSlot::Live(DescriptorKind::Sampler, binding_for(0))
            }),
            Err(TableRunError::Unsupported(_))
        ));
        // A texture entry in a sampler table.
        assert!(matches!(
            enumerate_descriptor_table_typed(TABLE_BASE, |_| {
                DescriptorSlot::Texture(DescriptorKind::Sampler, 0x3000)
            }),
            Err(TableRunError::Unsupported(_))
        ));
        // A texture entry in a constant-buffer table.
        assert!(matches!(
            enumerate_descriptor_table_typed(TABLE_BASE, |_| {
                DescriptorSlot::Texture(DescriptorKind::Cbv, 0x3000)
            }),
            Err(TableRunError::Unsupported(_))
        ));
        // A buffer after a texture in one SRV range is kept, so the typed
        // encoder can refuse the mixed table instead of binding a prefix.
        let run = enumerate_descriptor_table_typed(TABLE_BASE, |handle| {
            match ((handle - TABLE_BASE) / VIEW_DESCRIPTOR_INCREMENT) as usize {
                0 => DescriptorSlot::Texture(DescriptorKind::Srv, 0x3000),
                1 => DescriptorSlot::Live(DescriptorKind::Srv, binding_for(1)),
                _ => DescriptorSlot::Empty,
            }
        })
        .expect("the mixed run enumerates");
        assert_eq!(run.entries.len(), 2);
        assert_eq!(run.entries[0], DescriptorEntry::Texture(0x3000));
        assert_eq!(run.entries[1], entry_for(1));
    }

    #[test]
    fn typed_runs_enumerate_two_texture_entries_in_register_order() {
        let run = enumerate_descriptor_table_typed(TABLE_BASE, |handle| {
            match ((handle - TABLE_BASE) / VIEW_DESCRIPTOR_INCREMENT) as usize {
                0 => DescriptorSlot::Texture(DescriptorKind::Srv, 0x3000),
                1 => DescriptorSlot::Texture(DescriptorKind::Srv, 0x3001),
                _ => DescriptorSlot::Empty,
            }
        })
        .expect("a two-entry texture range");
        assert_eq!(run.kind, DescriptorKind::Srv);
        assert_eq!(
            run.entries,
            vec![
                DescriptorEntry::Texture(0x3000),
                DescriptorEntry::Texture(0x3001),
            ]
        );
        assert_eq!(run.boundary, TableRunBoundary::Empty);
    }

    #[test]
    fn typed_texture_and_sampler_tables_keep_their_entry_shape() {
        let resolver = FakeResolver::new();
        let page = resolver.page_address(0);

        let textures = descriptor_table_typed(
            &resolver,
            0x100,
            1,
            0,
            DescriptorKind::Srv,
            &[
                DescriptorEntry::Texture(0x3000),
                DescriptorEntry::Texture(0x3001),
            ],
        )
        .expect("a two-entry texture table");
        let (texture_table, texture_tokens) = textures.into_parts();
        assert_eq!(texture_table.kind, DescriptorKind::Srv);
        assert_eq!(
            texture_table.bindings,
            vec![
                DescriptorTableEntryV3::Texture(0x3000),
                DescriptorTableEntryV3::Texture(0x3001),
            ]
        );
        assert!(
            texture_table.bindings[0].as_buffer().is_none(),
            "a texture handle is not a buffer range"
        );
        assert_eq!(texture_table.bindings[0].resource_handle(), Some(0x3000));
        assert!(texture_tokens.is_empty(), "a texture entry pins no range");

        let samplers = descriptor_table_typed(
            &resolver,
            0x200,
            2,
            1,
            DescriptorKind::Sampler,
            &[DescriptorEntry::Sampler(0x5300_0000_0000_0042)],
        )
        .expect("a one-entry sampler table");
        let (sampler_table, sampler_tokens) = samplers.into_parts();
        assert_eq!(sampler_table.kind, DescriptorKind::Sampler);
        assert_eq!(
            sampler_table.bindings,
            vec![DescriptorTableEntryV3::Sampler(0x5300_0000_0000_0042)]
        );
        assert_eq!(
            sampler_table.bindings[0].resource_handle(),
            Some(0x5300_0000_0000_0042)
        );
        assert!(sampler_tokens.is_empty());

        // A buffer entry in a sampler table and vice versa are refused.
        let buffer = DescriptorEntry::Buffer(GuestBinding {
            address: page,
            len: 16,
        });
        assert!(matches!(
            descriptor_table_typed(
                &resolver,
                0x300,
                0,
                0,
                DescriptorKind::Sampler,
                &[buffer]
            ),
            Err(ResolveError::Unsupported(_))
        ));
        assert!(matches!(
            descriptor_table_typed(
                &resolver,
                0x301,
                0,
                0,
                DescriptorKind::Cbv,
                &[DescriptorEntry::Texture(0x3000)]
            ),
            Err(ResolveError::Unsupported(_))
        ));
        // A mixed buffer/texture table has no single wire entry tag.
        assert!(matches!(
            descriptor_table_typed(
                &resolver,
                0x302,
                0,
                0,
                DescriptorKind::Srv,
                &[buffer, DescriptorEntry::Texture(0x3000)]
            ),
            Err(ResolveError::Unsupported(_))
        ));
        // A zero registered handle is refused before any submission.
        assert!(matches!(
            descriptor_table_typed(
                &resolver,
                0x303,
                0,
                0,
                DescriptorKind::Srv,
                &[DescriptorEntry::Texture(0)]
            ),
            Err(ResolveError::Invalid(_))
        ));
        assert_eq!(resolver.live(), 0, "no refusal pinned a range");
    }

    #[test]
    fn unregistered_resource_handles_are_a_typed_error() {
        let table = DescriptorTableV3 {
            set: 0,
            base_register: 0,
            kind: DescriptorKind::Srv,
            bindings: vec![
                DescriptorTableEntryV3::Texture(0x3000),
                DescriptorTableEntryV3::Texture(0x3001),
            ],
        };
        assert_eq!(
            require_registered_handles(&table, |_| false, |_| false),
            Err(HandleError::UnregisteredTexture(0x3000))
        );
        assert!(require_registered_handles(&table, |handle| handle == 0x3000, |_| false)
            .is_err());

        let sampler = DescriptorTableV3 {
            set: 1,
            base_register: 0,
            kind: DescriptorKind::Sampler,
            bindings: vec![DescriptorTableEntryV3::Sampler(0x5300_0000_0000_0042)],
        };
        assert_eq!(
            require_registered_handles(&sampler, |_| false, |_| false),
            Err(HandleError::UnregisteredSampler(0x5300_0000_0000_0042))
        );
        require_registered_handles(&sampler, |_| false, |handle| {
            handle == 0x5300_0000_0000_0042
        })
        .expect("a registered sampler");
        let message = HandleError::UnregisteredSampler(1).to_string();
        assert!(message.contains("sampler"), "{message}");
    }

    #[test]
    fn the_enumeration_increments_match_the_guest_descriptor_heap() {
        // `descriptor.rs` owns the heap increments `[32, 16, 128, 128]`; the
        // enumeration must walk views at 32 and samplers at 16.
        assert_eq!(
            VIEW_DESCRIPTOR_INCREMENT,
            u64::from(crate::descriptor::increment_size(std::ptr::null(), 0))
        );
        assert_eq!(
            SAMPLER_DESCRIPTOR_INCREMENT,
            u64::from(crate::descriptor::increment_size(std::ptr::null(), 1))
        );
        assert_eq!(descriptor_increment(DescriptorKind::Cbv), 32);
        assert_eq!(descriptor_increment(DescriptorKind::Srv), 32);
        assert_eq!(descriptor_increment(DescriptorKind::Uav), 32);
        assert_eq!(descriptor_increment(DescriptorKind::Sampler), 16);
    }
}
