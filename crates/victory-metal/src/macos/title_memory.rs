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
        }
    }

    /// Replaces the map when `generation` is newer; older generations are
    /// ignored and an identical one is idempotent.
    pub fn set_map(&mut self, generation: u32, ranges: Vec<TitleRange>) {
        if generation < self.generation {
            return;
        }
        if generation == self.generation && ranges == self.ranges {
            return;
        }
        self.generation = generation;
        self.ranges = ranges;
        self.built = None;
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
                Some(built) => self.built = Some(built),
                None => return Ok(None),
            }
        }
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

/// Builds the buffers for `ranges`; `None` when an allocation is missing or a
/// range is out of its buffer.
fn build(
    device: Id,
    ranges: &[TitleRange],
    buffers: &HashMap<u64, RegisteredBuffer>,
) -> Result<Option<Built>, MetalError> {
    // Resolve every allocation to exactly one heap pair, in first-use order.
    let mut pairs: Vec<(u64, u32, Id)> = Vec::new();
    for range in ranges {
        let Some(registered) = buffers.get(&range.buffer.0) else {
            return Ok(None);
        };
        let end = range
            .byte_offset
            .checked_add(range.length)
            .ok_or(MetalError::InvalidArg("title-memory range overflows"))?;
        if end > registered.len as u64 {
            return Ok(None);
        }
        if !pairs.iter().any(|(buffer, _, _)| *buffer == range.buffer.0) {
            let index = pagetable::FIRST_ALLOCATION_HEAP_INDEX + 2 * pairs.len() as u32;
            pairs.push((range.buffer.0, index, registered.buffer.as_raw()));
        }
    }

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

    let mut targets = Vec::with_capacity(ranges.len());
    for range in ranges {
        let pair = pairs
            .iter()
            .find(|(buffer, _, _)| *buffer == range.buffer.0)
            .map(|(_, pair, _)| *pair)
            .expect("every allocation has a pair");
        targets.push(Target {
            guest_address: range.guest_address,
            length: range.length,
            heap_pair: pair,
            start_low: range.guest_address.wrapping_sub(range.byte_offset) as u32,
        });
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

fn retain(raw: Id) -> Option<ObjcObject> {
    if raw.is_null() {
        return None;
    }
    unsafe { ObjcObject::from_retained(objc::msg_send_id_0(raw, objc::sel(c"retain"))) }
}
