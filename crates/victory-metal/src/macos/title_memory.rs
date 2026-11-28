//! Live title-memory page table on the native Metal host (ADR-055).
//!
//! The guest sends its VA → (allocation, offset) map as
//! [`crate::title_map`] chunks; [`crate::ipc_host`] resolves each allocation id
//! to the live [`crate::abi::VmBufferId`] and hands the transport
//! [`TitleRange`]s. This module keeps that map and builds, from the already
//! registered share-file buffers, the dword table [`crate::macos::pagetable`]
//! documents:
//!
//! * one MTLBuffer per share allocation is named by a heap pair (raw SRV, raw
//!   UAV) in the converter's resource descriptor heap argument buffer at
//!   Metal `buffer(0)`;
//! * the page table's `t0` names each 64 KiB page's heap pair and the title
//!   address its allocation starts at;
//! * index 0/1 of the heap is the null pair, pointing at a zero-filled buffer,
//!   so a page no range names reads 0 and a write there is discarded.
//!
//! A new mapping generation, or a changed range set, drops the built table so
//! the next submission rebuilds it. Nothing here copies resource data: the
//! table and heap argument buffer hold device addresses only.
//!
//! The write bitmap `u0` is allocated zeroed and bound beside the table. Its
//! set bits name pages shaders wrote; the copy-back that consumes them is a
//! later step, so the bitmap is not yet cleared or read back here.

use std::collections::HashMap;

use crate::abi::VmBufferId;
use crate::macos::device::RegisteredBuffer;
use crate::macos::objc::{self, Id, ObjcObject};
use crate::macos::pagetable::{self, Target};
use crate::transport::MetalError;

/// Guest range the IPC host resolved to a live Metal buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TitleRange {
    /// First guest address of the range; 64 KiB aligned.
    pub guest_address: u64,
    /// Registered share allocation holding the page.
    pub buffer: VmBufferId,
    /// First byte of the range in the share file; 64 KiB aligned.
    pub byte_offset: u64,
    /// Range length; a non-zero multiple of 64 KiB.
    pub length: u64,
}

/// The zero buffer a miss reads. A miss's in-buffer offset is the address's
/// low dword (block 0's start is 0), so the buffer covers the full 32-bit
/// range when the device allows it; a smaller allocation leaves high miss
/// addresses uncovered, never a guessed page.
const NULL_BYTES: usize = 1 << 32;

/// Smallest null buffer the host accepts before it treats title memory as
/// unsupported (a miss at a higher address would fault).
const MIN_NULL_BYTES: usize = 1 << 20;

/// Built table, heap and null buffers, plus the map they were built from.
pub(crate) struct TitleMemory {
    generation: u32,
    ranges: Vec<TitleRange>,
    built: Option<Built>,
    /// Accepted map applications that changed the range set (ADR-058).
    updates: u64,
    /// Built tables dropped because the map or an allocation changed.
    invalidations: u64,
    /// Flushes that bound a live table.
    bindings: u64,
    /// Flushes that needed a table and could not build one (an allocation
    /// missing or a range outside its buffer), so the zero fallback stayed.
    misses: u64,
    /// Pages in the table most recently built; 0 when nothing is built.
    pages: u64,
    /// Heap pairs the built table names (one per share allocation).
    heap_pairs: u64,
}

/// Exact live-title-memory counters for one device (ADR-058).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Map applications that changed the range set.
    pub updates: u64,
    /// Pages in the table most recently built (a gauge).
    pub pages: u64,
    /// Built tables invalidated by a changed map or registration.
    pub invalidations: u64,
    /// Flushes that bound a live table.
    pub bindings: u64,
    /// Flushes that needed a table and could not build one.
    pub misses: u64,
    /// Ranges the installed map names.
    pub ranges: u64,
    /// Heap pairs the built table names.
    pub heap_pairs: u64,
}

struct Built {
    table: ObjcObject,
    bitmap: ObjcObject,
    heap: ObjcObject,
    null: ObjcObject,
    /// Buffers the table and heap name, retained until the next rebuild.
    allocations: Vec<ObjcObject>,
    table_bytes: u64,
    bitmap_bytes: u64,
    heap_bytes: u64,
    /// Pages the built table names explicitly (ADR-058 gauge).
    pages: u64,
    /// Heap pairs the built table names.
    heap_pairs: u64,
}

/// One submission's title-memory bindings: raw handles plus byte sizes.
pub(crate) struct Binding {
    pub table: Id,
    pub bitmap: Id,
    pub heap: Id,
    pub table_bytes: u64,
    pub bitmap_bytes: u64,
    pub heap_bytes: u64,
    /// Every resource the argument buffers reference (table, bitmap, heap,
    /// null and the allocations), for `useResource:`/residency.
    pub resources: Vec<Id>,
}

impl TitleMemory {
    /// A map with nothing bound.
    pub fn new() -> Self {
        Self {
            generation: 0,
            ranges: Vec::new(),
            built: None,
            updates: 0,
            invalidations: 0,
            bindings: 0,
            misses: 0,
            pages: 0,
            heap_pairs: 0,
        }
    }

    /// Replaces the map when `generation` is newer; older generations are
    /// ignored and an identical one is idempotent.
    ///
    /// A map that arrives before its share allocations are registered is kept
    /// whole, never discarded: [`Self::binding`] resolves it lazily against the
    /// current registered buffers. A later identical application changes
    /// nothing and rebuilds nothing (ADR-058).
    pub fn set_map(&mut self, generation: u32, ranges: Vec<TitleRange>) {
        if generation < self.generation {
            return;
        }
        if generation == self.generation && ranges == self.ranges {
            return;
        }
        if self.built.is_some() {
            self.invalidations = self.invalidations.saturating_add(1);
            crate::macos::memory_space_bind::note_invalidation();
        }
        self.generation = generation;
        self.ranges = ranges;
        self.built = None;
        self.pages = 0;
        self.heap_pairs = 0;
        self.updates = self.updates.saturating_add(1);
        crate::macos::memory_space_bind::note_map_application(self.ranges.len() as u64);
    }

    /// Exact counters of this device's live map and built table.
    ///
    /// Used by the ADR-058 tests; a future device-state accessor publishes the
    /// same numbers without touching `DirectMetalTransport`.
    #[allow(dead_code)]
    pub fn stats(&self) -> Stats {
        Stats {
            updates: self.updates,
            pages: self.pages,
            invalidations: self.invalidations,
            bindings: self.bindings,
            misses: self.misses,
            ranges: self.ranges.len() as u64,
            heap_pairs: self.heap_pairs,
        }
    }

    /// Builds the table and heap if the map changed, or returns the built
    /// binding. `Ok(None)` when no map is installed or a named allocation is
    /// not registered yet; the caller then leaves the zero fallback in place.
    pub fn binding(
        &mut self,
        device: Id,
        buffers: &HashMap<u64, RegisteredBuffer>,
    ) -> Result<Option<Binding>, MetalError> {
        if self.ranges.is_empty() {
            return Ok(None);
        }
        if self.built.is_none() {
            match build(device, &self.ranges, buffers)? {
                Some(built) => {
                    self.pages = built.pages;
                    self.heap_pairs = built.heap_pairs;
                    self.built = Some(built);
                }
                None => {
                    // Every registered allocation was present last time, so a
                    // refused build means the map or a length changed under a
                    // live table: drop it and try again on the next flush.
                    self.misses = self.misses.saturating_add(1);
                    self.pages = 0;
                    self.heap_pairs = 0;
                    crate::macos::memory_space_bind::note_binding(
                        self.pages,
                        self.heap_pairs,
                        false,
                    );
                    return Ok(None);
                }
            }
        }
        self.bindings = self.bindings.saturating_add(1);
        crate::macos::memory_space_bind::note_binding(self.pages, self.heap_pairs, true);
        let built = self.built.as_ref().expect("built above");
        let mut resources = vec![
            built.table.as_raw(),
            built.bitmap.as_raw(),
            built.heap.as_raw(),
            built.null.as_raw(),
        ];
        resources.extend(built.allocations.iter().map(ObjcObject::as_raw));
        Ok(Some(Binding {
            table: built.table.as_raw(),
            bitmap: built.bitmap.as_raw(),
            heap: built.heap.as_raw(),
            table_bytes: built.table_bytes,
            bitmap_bytes: built.bitmap_bytes,
            heap_bytes: built.heap_bytes,
            resources,
        }))
    }
}

impl Default for TitleMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// A map resolved against the currently registered share allocations.
struct Resolved {
    /// `(allocation id, raw SRV heap index, buffer)` per allocation, in
    /// first-use order.
    pairs: Vec<(u64, u32, Id)>,
    /// One page-table target per range, naming its allocation's heap pair.
    targets: Vec<Target>,
}

/// Resolves every range to a registered allocation and heap pair, or `None`
/// when any range names an allocation that is not registered yet (or is
/// shorter than the range). `Ok(None)` keeps the whole map pending instead of
/// building it from a partial address relation.
///
/// Pure: `registered` returns the allocation's `(byte length, raw buffer)`, so
/// the liveness ordering (map first, allocation later) is testable without a
/// Metal device.
fn resolve(
    ranges: &[TitleRange],
    mut registered: impl FnMut(u64) -> Option<(usize, Id)>,
) -> Result<Option<Resolved>, MetalError> {
    let mut pairs: Vec<(u64, u32, Id)> = Vec::new();
    for range in ranges {
        let Some((len, raw)) = registered(range.buffer.0) else {
            return Ok(None);
        };
        let end = range
            .byte_offset
            .checked_add(range.length)
            .ok_or(MetalError::InvalidArg("title-memory range overflows"))?;
        if end > len as u64 {
            return Ok(None);
        }
        if !pairs.iter().any(|(buffer, _, _)| *buffer == range.buffer.0) {
            let index = pagetable::FIRST_ALLOCATION_HEAP_INDEX + 2 * pairs.len() as u32;
            pairs.push((range.buffer.0, index, raw));
        }
    }

    let mut targets = Vec::with_capacity(ranges.len());
    for range in ranges {
        let pair = pairs
            .iter()
            .find(|(buffer, _, _)| *buffer == range.buffer.0)
            .map(|(_, pair, _)| *pair)
            .expect("every range resolved to a pair above");
        targets.push(Target {
            guest_address: range.guest_address,
            length: range.length,
            heap_pair: pair,
            start_low: range.guest_address.wrapping_sub(range.byte_offset) as u32,
        });
    }
    Ok(Some(Resolved { pairs, targets }))
}

/// Builds the buffers for `ranges`; `None` when an allocation is missing or a
/// range is out of its buffer.
fn build(
    device: Id,
    ranges: &[TitleRange],
    buffers: &HashMap<u64, RegisteredBuffer>,
) -> Result<Option<Built>, MetalError> {
    let Some(Resolved { pairs, targets }) =
        resolve(ranges, |id| {
            buffers
                .get(&id)
                .map(|registered| (registered.len, registered.buffer.as_raw()))
        })?
    else {
        return Ok(None);
    };

    let mut null = create_zero_buffer(device, NULL_BYTES)?;
    if null.is_none() {
        null = create_zero_buffer(device, MIN_NULL_BYTES)?;
    }
    let Some(null) = null else {
        return Ok(None);
    };
    let null_address = gpu_address(null.as_raw());

    // The heap argument buffer: null pair, then one pair per allocation.
    let heap_entries = pagetable::FIRST_ALLOCATION_HEAP_INDEX + 2 * pairs.len() as u32;
    let mut heap = vec![0u8; pagetable::heap_bytes(heap_entries)];
    pagetable::write_heap_entry(&mut heap, 0, null_address);
    pagetable::write_heap_entry(&mut heap, 1, null_address);
    for (buffer, index, raw) in &pairs {
        // The entry names the allocation's first byte, not the request's.
        let address = gpu_address(*raw);
        pagetable::write_heap_entry(&mut heap, *index, address);
        pagetable::write_heap_entry(&mut heap, *index + 1, address);
        let _ = buffer;
    }

    let table = match pagetable::build(&targets) {
        Ok(table) => table,
        // A map the page table cannot express is refused whole, never
        // truncated into guessed pages.
        Err(_) => return Ok(None),
    };
    let bitmap_bytes = write_bitmap_bytes();
    let bitmap = vec![0u8; bitmap_bytes];

    let table_buffer = create_buffer_with(device, &table.bytes)?;
    let bitmap_buffer = create_buffer_with(device, &bitmap)?;
    let heap_buffer = create_buffer_with(device, &heap)?;
    let mut allocations = Vec::with_capacity(pairs.len());
    for (_, _, raw) in &pairs {
        let Some(retained) = retain(*raw) else {
            return Ok(None);
        };
        allocations.push(retained);
    }
    Ok(Some(Built {
        table: table_buffer,
        bitmap: bitmap_buffer,
        heap: heap_buffer,
        null,
        table_bytes: table.bytes.len() as u64,
        bitmap_bytes: bitmap_bytes as u64,
        heap_bytes: heap.len() as u64,
        pages: table.pages,
        heap_pairs: pairs.len() as u64,
        allocations,
    }))
}

/// Byte size of the write bitmap: one bit per level-2 entry across all blocks.
pub fn write_bitmap_bytes() -> usize {
    // (MAX_BLOCKS + 1 null block) * 4096 pages, 8 bits a byte, 256-aligned.
    (((pagetable::MAX_BLOCKS + 1) * pagetable::BLOCK_ENTRIES).div_ceil(8)).next_multiple_of(256)
}

/// A shared `MTLBuffer` holding `bytes`, zero-filling the tail.
fn create_buffer_with(device: Id, bytes: &[u8]) -> Result<ObjcObject, MetalError> {
    let raw = create_zero_buffer(device, bytes.len())?.ok_or_else(|| {
        MetalError::InternalError("title-memory buffer allocation returned null".to_string())
    })?;
    let contents = unsafe { objc::msg_send_ptr_0(raw.as_raw(), objc::sel(c"contents")) };
    if contents.is_null() && !bytes.is_empty() {
        return Err(MetalError::InternalError(
            "title-memory buffer has no CPU-visible contents".to_string(),
        ));
    }
    if !bytes.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.cast::<u8>(),
                bytes.len(),
            );
        }
    }
    Ok(raw)
}

/// A zero-filled shared `MTLBuffer` of `len` bytes, or `None` if Metal
/// refuses the length (a 4 GiB null buffer is allowed on Apple Silicon but not
/// on every device).
fn create_zero_buffer(device: Id, len: usize) -> Result<Option<ObjcObject>, MetalError> {
    if len == 0 {
        return Err(MetalError::InvalidArg("a title-memory buffer names no bytes"));
    }
    let raw = unsafe { objc::msg_send_new_buffer(device, len, 0) };
    let Some(buffer) = ObjcObject::from_retained(raw) else {
        return Ok(None);
    };
    let contents = unsafe { objc::msg_send_ptr_0(buffer.as_raw(), objc::sel(c"contents")) };
    if !contents.is_null() {
        unsafe { std::ptr::write_bytes(contents.cast::<u8>(), 0, len) };
    }
    Ok(Some(buffer))
}

fn gpu_address(buffer: Id) -> u64 {
    unsafe { objc::msg_send_u64_0(buffer, objc::sel(c"gpuAddress")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = crate::title_map::PAGE_BYTES;

    fn range(address: u64, allocation: u64, offset: u64, length: u64) -> TitleRange {
        TitleRange {
            guest_address: address,
            buffer: VmBufferId(allocation),
            byte_offset: offset,
            length,
        }
    }

    /// ADR-058: the map may arrive before the share allocations it names. The
    /// unresolved ranges keep the whole map pending; once the allocation
    /// registers, the same stored map resolves to its heap pair and target.
    #[test]
    fn a_map_applied_before_its_allocation_resolves_after_registration() {
        let ranges = [range(0x1_0000, 7, 0, PAGE)];
        let mut lengths: HashMap<u64, usize> = HashMap::new();
        let raw = 0x1000usize as Id;
        assert!(
            resolve(&ranges, |id| lengths.get(&id).map(|len| (*len, raw)))
                .expect("resolution")
                .is_none(),
            "an unregistered allocation keeps the map pending"
        );
        lengths.insert(7, 2 * PAGE as usize);
        let resolved = resolve(&ranges, |id| lengths.get(&id).map(|len| (*len, raw)))
            .expect("resolution")
            .expect("the map now resolves");
        assert_eq!(
            resolved.pairs,
            vec![(7, pagetable::FIRST_ALLOCATION_HEAP_INDEX, raw)],
            "the registered allocation takes the first real heap pair"
        );
        assert_eq!(resolved.targets.len(), 1);
        assert_eq!(resolved.targets[0].guest_address, 0x1_0000);
        assert_eq!(
            resolved.targets[0].heap_pair,
            pagetable::FIRST_ALLOCATION_HEAP_INDEX
        );
        assert_eq!(resolved.targets[0].start_low, 0x1_0000);
    }

    #[test]
    fn one_unregistered_allocation_refuses_the_whole_map() {
        let ranges = [
            range(0x1_0000, 1, 0, PAGE),
            range(0x2_0000, 2, 0, PAGE),
        ];
        let raw = 0x1000usize as Id;
        let resolved = resolve(&ranges, |id| {
            (id == 1).then_some((PAGE as usize, raw))
        })
        .expect("resolution");
        assert!(resolved.is_none(), "no partial table is ever built");
    }

    #[test]
    fn an_out_of_range_offset_keeps_the_map_pending() {
        let ranges = [range(0x1_0000, 1, PAGE, 2 * PAGE)];
        let raw = 0x1000usize as Id;
        let resolved = resolve(&ranges, |_| Some((PAGE as usize, raw))).expect("resolution");
        assert!(resolved.is_none(), "a range past its buffer is not built");
    }

    #[test]
    fn identical_map_applications_are_idempotent_and_counted() {
        let mut memory = TitleMemory::new();
        let ranges = vec![range(0x1_0000, 1, 0, PAGE)];
        memory.set_map(1, ranges.clone());
        assert_eq!(memory.stats().updates, 1);
        assert_eq!(memory.stats().ranges, 1);
        memory.set_map(1, ranges.clone());
        assert_eq!(memory.stats().updates, 1, "an identical application is a no-op");
        memory.set_map(2, ranges.clone());
        assert_eq!(memory.stats().updates, 2, "a changed generation is an update");
        memory.set_map(1, Vec::new());
        assert_eq!(memory.stats().updates, 2, "an older generation is ignored");
        assert_eq!(memory.stats().ranges, 1);
    }
}

fn retain(raw: Id) -> Option<ObjcObject> {
    if raw.is_null() {
        return None;
    }
    unsafe { ObjcObject::from_retained(objc::msg_send_id_0(raw, objc::sel(c"retain"))) }
}
