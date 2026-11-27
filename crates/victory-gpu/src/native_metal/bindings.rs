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
    DescriptorKind, DescriptorTableV3, IndexBufferBinding, IndexFormat, MAX_DESCRIPTOR_BINDINGS_V3,
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

/// Resolves and pins one homogeneous descriptor table.
///
/// `handle` is the D3D12 GPU descriptor handle of the table's first entry; it
/// is validated (non-zero) but the Wire frame carries the resolved ranges, not
/// the handle. `descriptors` are the table entries in register order, as the
/// caller read them from guest descriptor memory. Sampler tables have no Wire
/// v3 buffer binding and are refused.
pub fn descriptor_table(
    resolver: &dyn GuestRangeResolver,
    handle: u64,
    set: u16,
    base_register: u16,
    kind: DescriptorKind,
    descriptors: &[GuestBinding],
) -> Result<Resolved<DescriptorTableV3>, ResolveError> {
    if handle == 0 {
        return Err(ResolveError::Invalid("the descriptor-table handle is zero"));
    }
    if kind == DescriptorKind::Sampler {
        return Err(ResolveError::Unsupported(
            "sampler descriptor tables have no buffer binding",
        ));
    }
    if descriptors.is_empty() {
        return Err(ResolveError::Invalid("the descriptor table is empty"));
    }
    if descriptors.len() > MAX_DESCRIPTOR_BINDINGS_V3 {
        return Err(ResolveError::Unsupported(
            "the descriptor table has more bindings than a v3 table carries",
        ));
    }

    let mut bindings = Vec::with_capacity(descriptors.len());
    let mut tokens = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let resolved = match resolve_descriptor(resolver, descriptor) {
            Ok(resolved) => resolved,
            Err(error) => {
                retire_all(resolver, &tokens);
                return Err(error);
            }
        };
        bindings.push(resolved.0);
        tokens.push(resolved.1);
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
        assert_eq!((table.bindings[0].offset, table.bindings[0].length), (0x20, 0x100));
        assert_eq!(
            (table.bindings[1].offset, table.bindings[1].length),
            (PAGE + 0x10, 0x40)
        );
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
}
