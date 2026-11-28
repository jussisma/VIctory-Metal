//! The native host's title-memory page table (ADR-055).
//!
//! A translated shader that reads or writes memory at an address it computes
//! looks the address up in the two-level table documented by
//! `victory_shader::emit` ("Title memory"). This module builds that table from
//! the guest's VA → share-allocation map and lays out the resource descriptor
//! heap argument buffer its entries index.
//!
//! # Page table dwords
//!
//! The table is a dword array in the root SRV `t0` of `MEMORY_SPACE`:
//!
//! ```text
//! [0, 4096)                 level-1 entries, address bits 28-39:
//!                           level-2 block number, 0 for "no block"
//! [4096 + b*8192, ...)      block b: 4096 two-dword entries, page bits 16-27
//!                           entry = (raw SRV heap index, title address low dword)
//!                           the raw UAV of the same copy is the next heap index
//! ```
//!
//! Block 0 is the null block: every one of its entries is `[0, 0]`, and
//! `[0, 0]` is what the shader reads as "no copy holds this page". A guest
//! address with no range in the map stays at its level-1 entry's 0, which the
//! shader turns into a block-0 lookup, so it reads the null entry and marks the
//! miss in the write bitmap. The host never guesses a page.
//!
//! The second dword is the *low dword of the title address the allocation's
//! first byte corresponds to*: `start = guest_address - byte_offset`, so the
//! shader's `address - start` is the byte offset in the allocation's buffer.
//! A non-zero allocation offset therefore shifts `start` down, exactly as the
//! D3D12 builder's copy starts do (`victory_gpu::memory`).
//!
//! # Resource descriptor heap argument buffer
//!
//! The shader turns a table entry's heap index into a raw handle with DXIL
//! `createHandleFromHeap`. Metal Shader Converter lowers that to a load from
//! the implicit resource descriptor heap argument buffer at Metal `buffer(0)`.
//! Its entry layout was read from the converter's own AIR:
//!
//! ```text
//! %entry = getelementptr i8, i8* <buffer(0)>, i64 index * 24
//! %address = load <T> addrspace(1)*, <T> addrspace(1)** %entry, align 8
//! ```
//!
//! So entry `i` starts at byte `i * 24` and its first 8 bytes are the raw
//! buffer's 64-bit device address (the rest is unused, as for the converted
//! descriptor tables). [`write_heap_entry`] writes exactly that.
//!
//! Index 0 is the null pair's raw SRV and index 1 its raw UAV; both name a
//! zero-filled buffer so an unmapped page reads 0 and a write is discarded
//! (`null_readable_bytes` covers the address's in-buffer offset, which for a
//! miss is the full low dword because block 0's start is 0). Real allocations
//! take consecutive pairs from index 2.
//!
//! # Bounds
//!
//! [`MAX_BLOCKS`] blocks besides the null one, [`MAX_RANGES`] ranges and
//! [`MAX_MAPPED_PAGES`] pages keep the table (and the work to build it) bounded
//! well below the D3D12 path's 512-block table. A map past a cap is refused
//! whole, never truncated.

use crate::title_map::{PAGE_BYTES, Range};

/// Dwords in the level-1 index: 2^40 bytes in 256 MiB blocks.
pub const LEVEL1_ENTRIES: usize = 4096;

/// Entries per level-2 block: 4096 pages of 64 KiB, 256 MiB.
pub const BLOCK_ENTRIES: usize = 4096;

/// Level-2 blocks a table has room for besides the null one.
pub const MAX_BLOCKS: usize = 512;

/// Ranges one table may be built from.
pub const MAX_RANGES: usize = 1 << 16;

/// Pages one table may name (64 GiB).
pub const MAX_MAPPED_PAGES: u64 = 1 << 20;

/// Bytes of one resource-descriptor-heap argument-buffer entry.
///
/// Observed in the converter's AIR for `createHandleFromHeap`: `index * 24`
/// then an 8-byte address load.
pub const HEAP_ENTRY_BYTES: usize = 24;

/// First heap index a real allocation may use (0 and 1 are the null pair).
pub const FIRST_ALLOCATION_HEAP_INDEX: u32 = 2;

/// Bytes needed for heap indices `0..entries`.
pub fn heap_bytes(entries: u32) -> usize {
    entries as usize * HEAP_ENTRY_BYTES
}

/// Writes one resource-descriptor-heap entry: a raw buffer device address at
/// the entry's first 8 bytes, zero elsewhere.
pub fn write_heap_entry(out: &mut [u8], index: u32, address: u64) {
    let at = index as usize * HEAP_ENTRY_BYTES;
    out[at..at + 8].copy_from_slice(&address.to_le_bytes());
    out[at + 8..at + HEAP_ENTRY_BYTES].fill(0);
}

/// Bytes of a page table with `blocks` level-2 blocks plus the null block.
pub fn table_bytes(blocks: usize) -> usize {
    (LEVEL1_ENTRIES + 2 * (blocks + 1) * BLOCK_ENTRIES) * 4
}

/// A range with the heap pair its pages name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// First guest address of the range.
    pub guest_address: u64,
    /// Length in bytes; a multiple of [`PAGE_BYTES`].
    pub length: u64,
    /// Heap index of the range's raw SRV; the raw UAV is the next index.
    pub heap_pair: u32,
    /// Low dword of the title address the allocation's first byte holds.
    pub start_low: u32,
}

impl Target {
    /// The target for one wire range resolved to heap index `heap_pair`.
    ///
    /// `start = guest_address - byte_offset` wraps in the low dword, exactly as
    /// the shader's `address - start` arithmetic does.
    pub fn for_range(range: &Range, heap_pair: u32) -> Self {
        Self {
            guest_address: range.guest_address,
            length: range.length,
            heap_pair,
            start_low: (range.guest_address.wrapping_sub(range.byte_offset)) as u32,
        }
    }
}

/// Why a table could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// Two ranges name the same guest byte.
    Overlap { first: u64, second: u64 },
    /// A range is empty, unaligned or past 2^40.
    InvalidRange { guest_address: u64 },
    /// More blocks than [`MAX_BLOCKS`], ranges than [`MAX_RANGES`] or pages
    /// than [`MAX_MAPPED_PAGES`].
    TooLarge(&'static str),
    /// A heap pair names the null pair or the index space is exhausted.
    InvalidHeap(u32),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overlap { first, second } => write!(
                f,
                "title-memory ranges at 0x{first:x} and 0x{second:x} overlap"
            ),
            Self::InvalidRange { guest_address } => {
                write!(f, "title-memory range 0x{guest_address:x} is invalid")
            }
            Self::TooLarge(what) => write!(f, "title-memory table is too large: {what}"),
            Self::InvalidHeap(index) => write!(f, "title-memory heap pair {index} is invalid"),
        }
    }
}

impl std::error::Error for BuildError {}

/// A built page table.
pub struct Built {
    /// The table's little-endian dwords, ready for an `MTLBuffer`.
    pub bytes: Vec<u8>,
    /// Level-1 index of each block; `[usize::MAX]` is the null block 0.
    pub blocks: Vec<usize>,
    /// Pages the table names explicitly.
    pub pages: u64,
}

/// Builds the two-level page table for `targets`, in any order.
///
/// Ranges must not overlap. Blocks are numbered from 1 in order of the
/// first (sorted) address they hold; block 0 stays null. Every unmapped page
/// keeps the null entry `[0, 0]`.
pub fn build(targets: &[Target]) -> Result<Built, BuildError> {
    if targets.len() > MAX_RANGES {
        return Err(BuildError::TooLarge("ranges"));
    }
    let mut sorted: Vec<Target> = targets.to_vec();
    sorted.sort_by_key(|t| t.guest_address);
    let mut pages = 0u64;
    let mut previous_end: Option<u64> = None;
    for target in &sorted {
        let end = target
            .guest_address
            .checked_add(target.length)
            .ok_or(BuildError::InvalidRange {
                guest_address: target.guest_address,
            })?;
        if target.guest_address == 0
            || target.length == 0
            || !target.guest_address.is_multiple_of(PAGE_BYTES)
            || !target.length.is_multiple_of(PAGE_BYTES)
            || end > crate::title_map::ADDRESS_LIMIT
        {
            return Err(BuildError::InvalidRange {
                guest_address: target.guest_address,
            });
        }
        if target.heap_pair < FIRST_ALLOCATION_HEAP_INDEX || target.heap_pair % 2 != 0 {
            return Err(BuildError::InvalidHeap(target.heap_pair));
        }
        if let Some(end) = previous_end
            && target.guest_address < end
        {
            return Err(BuildError::Overlap {
                first: end - PAGE_BYTES,
                second: target.guest_address,
            });
        }
        pages = pages
            .checked_add(target.length / PAGE_BYTES)
            .filter(|pages| *pages <= MAX_MAPPED_PAGES)
            .ok_or(BuildError::TooLarge("pages"))?;
        previous_end = Some(target.guest_address + target.length);
    }

    // Assign one block number per 256 MiB region that has a target.
    let mut level1 = vec![0u32; LEVEL1_ENTRIES];
    let mut blocks = vec![usize::MAX];
    let mut next_block = 1usize;
    for target in &sorted {
        let first = target.guest_address >> 28;
        let last = (target.guest_address + target.length - 1) >> 28;
        for region in first..=last {
            let at = (region & 0xFFF) as usize;
            if level1[at] == 0 {
                if next_block > MAX_BLOCKS {
                    return Err(BuildError::TooLarge("blocks"));
                }
                level1[at] = next_block as u32;
                blocks.push(at);
                next_block += 1;
            }
        }
    }

    let len = table_bytes(next_block - 1);
    let mut bytes = vec![0u8; len];
    let mut dwords = vec![0u32; len / 4];
    dwords[..LEVEL1_ENTRIES].copy_from_slice(&level1);
    for target in &sorted {
        let first = target.guest_address >> 16;
        let last = (target.guest_address + target.length - 1) >> 16;
        for page in first..=last {
            let block = level1[(page >> 12) as usize & 0xFFF] as usize;
            let entry = block * BLOCK_ENTRIES + (page & 0xFFF) as usize;
            let at = LEVEL1_ENTRIES + 2 * entry;
            dwords[at] = target.heap_pair;
            dwords[at + 1] = target.start_low;
        }
    }
    for (dword, slot) in dwords.iter().zip(bytes.chunks_exact_mut(4)) {
        slot.copy_from_slice(&dword.to_le_bytes());
    }
    Ok(Built {
        bytes,
        blocks,
        pages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(address: u64, length: u64, heap_pair: u32, start_low: u32) -> Target {
        Target {
            guest_address: address,
            length,
            heap_pair,
            start_low,
        }
    }

    fn dword(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().expect("four bytes"))
    }

    fn entry(bytes: &[u8], block: usize, page: usize) -> (u32, u32) {
        let at = LEVEL1_ENTRIES + 2 * (block * BLOCK_ENTRIES + page);
        (dword(bytes, at), dword(bytes, at + 1))
    }

    #[test]
    fn an_empty_map_is_the_null_block() {
        let built = build(&[]).expect("build");
        assert_eq!(built.blocks, vec![usize::MAX]);
        assert_eq!(built.pages, 0);
        assert_eq!(built.bytes.len(), table_bytes(0));
        for l1 in 0..LEVEL1_ENTRIES {
            assert_eq!(dword(&built.bytes, l1), 0, "level-1 {l1} is null");
        }
        assert_eq!(entry(&built.bytes, 0, 0), (0, 0), "block 0 entry 0 is null");
        assert_eq!(entry(&built.bytes, 0, 0xFFF), (0, 0), "every null entry");
    }

    #[test]
    fn a_page_at_a_non_zero_address_offset_lands_in_its_block() {
        let address = 0x12_3450_0000u64;
        let built = build(&[target(address, PAGE_BYTES, 2, address as u32)]).expect("build");
        let region = ((address >> 28) & 0xFFF) as usize;
        assert_eq!(built.blocks, vec![usize::MAX, region]);
        assert_eq!(dword(&built.bytes, region), 1, "level-1 names block 1");
        assert_eq!(
            entry(&built.bytes, 1, ((address >> 16) & 0xFFF) as usize),
            (2, address as u32)
        );
        // The neighbouring page stays null.
        assert_eq!(
            entry(&built.bytes, 1, (((address >> 16) & 0xFFF) as usize + 1) & 0xFFF),
            (0, 0)
        );
    }

    #[test]
    fn an_unmapped_va_stays_null() {
        let mapped = 0x2_0000u64;
        let unmapped = 0x10_0000u64;
        let built = build(&[target(mapped, PAGE_BYTES, 2, mapped as u32)]).expect("build");
        // Both addresses share 256 MiB region 0, so the region has a block;
        // the unmapped page itself is still the null entry.
        assert_eq!(dword(&built.bytes, (unmapped >> 28) as usize), 1);
        assert_eq!(entry(&built.bytes, 1, ((unmapped >> 16) & 0xFFF) as usize), (0, 0));
        // A region with no target at all keeps its level-1 entry 0, which the
        // shader turns into block 0's null entry.
        assert_eq!(dword(&built.bytes, 0x100), 0);
        assert_eq!(entry(&built.bytes, 0, 0), (0, 0));
    }

    #[test]
    fn a_non_zero_allocation_offset_shifts_the_start_dword() {
        let range = Range {
            guest_address: 0x10_0000,
            allocation_id: 7,
            byte_offset: 0x2_0000,
            length: PAGE_BYTES,
        };
        let built = build(&[Target::for_range(&range, 2)]).expect("build");
        let (heap, start) = entry(&built.bytes, 1, ((range.guest_address >> 16) & 0xFFF) as usize);
        assert_eq!(heap, 2);
        assert_eq!(
            start,
            range.guest_address as u32 - 0x2_0000,
            "start is the title address of the allocation's first byte"
        );
        assert_eq!(
            range.guest_address as u32 - start,
            0x2_0000,
            "the shader's offset into the heap view is the allocation offset"
        );
    }

    #[test]
    fn ranges_spanning_regions_get_one_block_each() {
        let a = 0x0FFF_0000u64; // last page of region 0
        let b = 0x1000_0000u64; // first page of region 1
        let built = build(&[
            target(a, PAGE_BYTES, 2, a as u32),
            target(b, PAGE_BYTES, 4, b as u32),
        ])
        .expect("build");
        assert_eq!(built.blocks, vec![usize::MAX, 0, 1]);
        assert_eq!(dword(&built.bytes, 0), 1);
        assert_eq!(dword(&built.bytes, 1), 2);
        assert_eq!(entry(&built.bytes, 1, 0xFFF), (2, a as u32));
        assert_eq!(entry(&built.bytes, 2, 0), (4, b as u32));
    }

    #[test]
    fn overlapping_or_invalid_ranges_are_refused() {
        let first = target(0x10_0000, 2 * PAGE_BYTES, 2, 0x10_0000);
        let second = target(0x11_0000, PAGE_BYTES, 4, 0x11_0000);
        assert!(matches!(
            build(&[first, second]),
            Err(BuildError::Overlap { .. })
        ));
        assert!(matches!(
            build(&[target(0x10_0001, PAGE_BYTES, 2, 0)]),
            Err(BuildError::InvalidRange { .. })
        ));
        assert!(matches!(
            build(&[target(0x10_0000, PAGE_BYTES, 1, 0)]),
            Err(BuildError::InvalidHeap(1))
        ));
    }

    #[test]
    fn the_heap_argument_buffer_entry_is_a_24_byte_address() {
        let mut heap = vec![0u8; heap_bytes(3)];
        write_heap_entry(&mut heap, 0, 0xAAAA);
        write_heap_entry(&mut heap, 1, 0xBBBB);
        write_heap_entry(&mut heap, 2, 0xCCCC);
        assert_eq!(heap.len(), 72);
        for (index, address) in [(0usize, 0xAAAAu64), (1, 0xBBBB), (2, 0xCCCC)] {
            let at = index * HEAP_ENTRY_BYTES;
            assert_eq!(
                u64::from_le_bytes(heap[at..at + 8].try_into().expect("eight bytes")),
                address
            );
            assert!(heap[at + 8..at + HEAP_ENTRY_BYTES].iter().all(|&b| b == 0));
        }
        assert_eq!(HEAP_ENTRY_BYTES, 24, "stride observed in the converter AIR");
    }
}
