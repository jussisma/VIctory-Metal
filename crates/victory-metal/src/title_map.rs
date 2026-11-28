//! Guest title-memory map: the VA → share-allocation relation a translated
//! title-memory shader needs (ADR-055).
//!
//! A translated shader that reads or writes memory at an address it computes
//! looks the address up in a two-level page table (`victory_shader::emit`,
//! "Title memory"). The console shader addresses memory by guest GPU virtual
//! address; the host holds the bytes as one MTLBuffer per share file. The page
//! table can only name a page if it knows which share allocation and byte
//! offset that guest address maps to, so the guest transmits the relation once
//! per mapping generation.
//!
//! # Frame
//!
//! The frame is a new post-authentication transport kind
//! ([`KIND_TITLE_MAP`](crate::ipc_host::KIND_TITLE_MAP)), never a changed
//! existing payload. Every existing frame keeps its exact bytes, so a host or
//! guest that does not know this kind still speaks the protocol it did; a host
//! that receives the kind without support refuses it by name
//! (`HostError::Protocol`), before any submission, rather than misreading it.
//! The producer only sends the frame when a title-memory map exists, and
//! `VICTORY_METAL_TITLE_MEMORY_MAP=0` turns it off for an older host.
//!
//! One payload is one chunk of a generation:
//!
//! | offset | size | field |
//! | ---: | ---: | --- |
//! | 0 | 1 | version, [`VERSION`] |
//! | 1 | 1 | reserved, 0 |
//! | 2 | 4 | mapping generation, little-endian `u32`, never 0 |
//! | 6 | 2 | chunk index, 0-based |
//! | 8 | 2 | chunk count |
//! | 10 | 2 | entries in this chunk |
//! | 12 | `n * 40` | entries |
//!
//! One entry is `guest_address: u64`, `allocation_id: u64`,
//! `byte_offset: u64`, `length: u64`, `reserved: u64`, all little-endian. The
//! guest address and byte offset are 64 KiB aligned, the length is a non-zero
//! multiple of 64 KiB, the allocation id is never 0, and
//! `guest_address + length` stays below 2^40 (the page table's address width).
//!
//! # Chunking and idempotency
//!
//! A frame payload is bounded by the transport's 64 KiB payload
//! ([`MAX_ENTRIES_PER_FRAME`] entries), so a larger map is sent as several
//! chunks of one generation. Chunks may be re-sent: applying the same entry
//! twice is the same mapping. A newer generation replaces the previous one
//! atomically at the host; an older generation is ignored. The host also caps
//! the total range count ([`MAX_RANGES`]) and mapped page count
//! ([`MAX_MAPPED_PAGES`]) so a hostile or broken peer cannot grow the map
//! without bound.
//!
//! No asset bytes, ids or paths travel here: only the address relation.

/// Transport frame kind carrying one title-memory map chunk.
pub const KIND_TITLE_MAP: u8 = 11;

/// Version of the payload below.
pub const VERSION: u8 = 1;

/// Fixed chunk header length.
pub const HEADER_LEN: usize = 12;

/// Fixed encoded length of one entry.
pub const ENTRY_LEN: usize = 40;

/// Largest payload the transport envelope accepts (`ipc_host::MAX_PAYLOAD_LEN`).
pub const MAX_PAYLOAD_LEN: usize = 65_536;

/// Largest number of entries one chunk carries.
pub const MAX_ENTRIES_PER_FRAME: usize = (MAX_PAYLOAD_LEN - HEADER_LEN) / ENTRY_LEN;

/// The share provider's page: every address, offset and length is a multiple.
pub const PAGE_BYTES: u64 = 64 * 1024;

/// Largest guest address the two-level page table can name (2^40).
pub const ADDRESS_LIMIT: u64 = 1 << 40;

/// Bound on the ranges one host keeps from one peer.
pub const MAX_RANGES: usize = 1 << 16;

/// Bound on the pages one host maps through the page table (64 GiB).
pub const MAX_MAPPED_PAGES: u64 = 1 << 20;

/// Why a map payload was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitleMapError {
    /// Payload shorter than the fixed header.
    TruncatedHeader,
    /// The payload declares a version this codec does not implement.
    UnsupportedVersion(u8),
    /// The chunk header is internally inconsistent.
    Malformed(&'static str),
    /// The payload length does not match its declared entry count.
    Length,
    /// An entry names a zero or unaligned field, or overruns 2^40.
    InvalidEntry(&'static str),
    /// More entries than one frame or the host's map may hold.
    TooMany(&'static str),
}

impl std::fmt::Display for TitleMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "title map payload is shorter than its header"),
            Self::UnsupportedVersion(version) => {
                write!(f, "title map version {version} is not supported")
            }
            Self::Malformed(why) => write!(f, "title map payload is malformed: {why}"),
            Self::Length => write!(f, "title map payload length does not match its entry count"),
            Self::InvalidEntry(why) => write!(f, "title map entry is invalid: {why}"),
            Self::TooMany(what) => write!(f, "title map names too many {what}"),
        }
    }
}

impl std::error::Error for TitleMapError {}

/// One guest virtual address range and the share allocation holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    /// First guest GPU virtual address of the range; 64 KiB aligned.
    pub guest_address: u64,
    /// Share allocation id the host registered (kind 2); never 0.
    pub allocation_id: u64,
    /// First byte of the range in the share file; 64 KiB aligned.
    pub byte_offset: u64,
    /// Range length; a non-zero multiple of 64 KiB.
    pub length: u64,
}

impl Range {
    /// Validates the entry against the wire contract.
    pub fn validate(&self) -> Result<(), TitleMapError> {
        if self.allocation_id == 0 {
            return Err(TitleMapError::InvalidEntry("allocation id is zero"));
        }
        if self.guest_address == 0 {
            return Err(TitleMapError::InvalidEntry("guest address is zero"));
        }
        if !self.guest_address.is_multiple_of(PAGE_BYTES) {
            return Err(TitleMapError::InvalidEntry(
                "guest address is not 64 KiB aligned",
            ));
        }
        if !self.byte_offset.is_multiple_of(PAGE_BYTES) {
            return Err(TitleMapError::InvalidEntry(
                "allocation offset is not 64 KiB aligned",
            ));
        }
        if self.length == 0 || !self.length.is_multiple_of(PAGE_BYTES) {
            return Err(TitleMapError::InvalidEntry(
                "length is not a non-zero multiple of 64 KiB",
            ));
        }
        let end = self
            .guest_address
            .checked_add(self.length)
            .ok_or(TitleMapError::InvalidEntry("guest range overflows"))?;
        if end > ADDRESS_LIMIT {
            return Err(TitleMapError::InvalidEntry(
                "guest range reaches past the page table's 2^40 address limit",
            ));
        }
        Ok(())
    }

    /// Pages the range names.
    pub fn pages(&self) -> u64 {
        self.length / PAGE_BYTES
    }
}

/// One decoded chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Mapping generation this chunk belongs to; never 0.
    pub generation: u32,
    /// Zero-based chunk index.
    pub index: u16,
    /// Number of chunks in this generation.
    pub count: u16,
    /// Ranges this chunk carries.
    pub ranges: Vec<Range>,
}

/// Encodes one chunk, refusing an oversized or malformed entry set.
pub fn encode(
    generation: u32,
    index: u16,
    count: u16,
    ranges: &[Range],
) -> Result<Vec<u8>, TitleMapError> {
    if generation == 0 {
        return Err(TitleMapError::Malformed("generation is zero"));
    }
    if count == 0 || index >= count {
        return Err(TitleMapError::Malformed(
            "chunk index or count is inconsistent",
        ));
    }
    if ranges.len() > MAX_ENTRIES_PER_FRAME {
        return Err(TitleMapError::TooMany("entries in one chunk"));
    }
    for range in ranges {
        range.validate()?;
    }
    let mut out = Vec::with_capacity(HEADER_LEN + ranges.len() * ENTRY_LEN);
    out.push(VERSION);
    out.push(0);
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&index.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(ranges.len() as u16).to_le_bytes());
    for range in ranges {
        out.extend_from_slice(&range.guest_address.to_le_bytes());
        out.extend_from_slice(&range.allocation_id.to_le_bytes());
        out.extend_from_slice(&range.byte_offset.to_le_bytes());
        out.extend_from_slice(&range.length.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
    }
    Ok(out)
}

/// Chunks a whole map, one payload per chunk.
pub fn chunks(generation: u32, ranges: &[Range]) -> Result<Vec<Vec<u8>>, TitleMapError> {
    if generation == 0 {
        return Err(TitleMapError::Malformed("generation is zero"));
    }
    if ranges.len() > MAX_RANGES {
        return Err(TitleMapError::TooMany("ranges in one map"));
    }
    let mut pages = 0u64;
    for range in ranges {
        range.validate()?;
        pages = pages
            .checked_add(range.pages())
            .ok_or(TitleMapError::TooMany("mapped pages"))?;
    }
    if pages > MAX_MAPPED_PAGES {
        return Err(TitleMapError::TooMany("mapped pages"));
    }
    let count = ranges.len().div_ceil(MAX_ENTRIES_PER_FRAME).max(1) as u16;
    let mut out = Vec::with_capacity(count as usize);
    for (index, part) in ranges.chunks(MAX_ENTRIES_PER_FRAME).enumerate() {
        out.push(encode(generation, index as u16, count, part)?);
    }
    if out.is_empty() {
        // An empty generation is still one (empty) chunk, so the host can
        // replace an old map with "nothing is mapped" instead of keeping it.
        out.push(encode(generation, 0, 1, &[])?);
    }
    Ok(out)
}

/// Decodes one chunk, refusing malformed, truncated or unsupported payloads.
pub fn decode(payload: &[u8]) -> Result<Chunk, TitleMapError> {
    if payload.len() < HEADER_LEN {
        return Err(TitleMapError::TruncatedHeader);
    }
    let version = payload[0];
    if version != VERSION {
        return Err(TitleMapError::UnsupportedVersion(version));
    }
    if payload[1] != 0 {
        return Err(TitleMapError::Malformed("reserved header byte is set"));
    }
    let generation = u32::from_le_bytes(
        payload[2..6]
            .try_into()
            .map_err(|_| TitleMapError::TruncatedHeader)?,
    );
    if generation == 0 {
        return Err(TitleMapError::Malformed("generation is zero"));
    }
    let index = u16::from_le_bytes(
        payload[6..8]
            .try_into()
            .map_err(|_| TitleMapError::TruncatedHeader)?,
    );
    let count = u16::from_le_bytes(
        payload[8..10]
            .try_into()
            .map_err(|_| TitleMapError::TruncatedHeader)?,
    );
    let entries = usize::from(u16::from_le_bytes(
        payload[10..12]
            .try_into()
            .map_err(|_| TitleMapError::TruncatedHeader)?,
    ));
    if count == 0 || index >= count {
        return Err(TitleMapError::Malformed(
            "chunk index or count is inconsistent",
        ));
    }
    if entries > MAX_ENTRIES_PER_FRAME {
        return Err(TitleMapError::TooMany("entries in one chunk"));
    }
    let expected = HEADER_LEN
        .checked_add(entries.checked_mul(ENTRY_LEN).ok_or(TitleMapError::Length)?)
        .ok_or(TitleMapError::Length)?;
    if payload.len() != expected {
        return Err(TitleMapError::Length);
    }
    let mut ranges = Vec::with_capacity(entries);
    for at in (HEADER_LEN..expected).step_by(ENTRY_LEN) {
        let field = |offset: usize| -> Result<u64, TitleMapError> {
            payload
                .get(at + offset..at + offset + 8)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u64::from_le_bytes)
                .ok_or(TitleMapError::Length)
        };
        let range = Range {
            guest_address: field(0)?,
            allocation_id: field(8)?,
            byte_offset: field(16)?,
            length: field(24)?,
        };
        let reserved = field(32)?;
        if reserved != 0 {
            return Err(TitleMapError::InvalidEntry("reserved entry field is set"));
        }
        range.validate()?;
        ranges.push(range);
    }
    Ok(Chunk {
        generation,
        index,
        count,
        ranges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(address: u64, allocation: u64, offset: u64, length: u64) -> Range {
        Range {
            guest_address: address,
            allocation_id: allocation,
            byte_offset: offset,
            length,
        }
    }

    #[test]
    fn one_chunk_round_trips_exactly() {
        let ranges = [
            range(0x1_0000, 3, 0, 0x2_0000),
            range(0x1_0000_0000, 5, 0x20_0000, 0x1_0000),
        ];
        let bytes = encode(7, 0, 1, &ranges).expect("encode");
        let chunk = decode(&bytes).expect("decode");
        assert_eq!(chunk.generation, 7);
        assert_eq!((chunk.index, chunk.count), (0, 1));
        assert_eq!(chunk.ranges, ranges);
        assert_eq!(bytes.len(), HEADER_LEN + 2 * ENTRY_LEN);
        // Encoding is stable, so a re-send is byte-identical (idempotent).
        assert_eq!(encode(7, 0, 1, &ranges).expect("encode"), bytes);
    }

    #[test]
    fn a_large_map_is_chunked_and_every_chunk_round_trips() {
        let ranges: Vec<Range> = (0..(MAX_ENTRIES_PER_FRAME * 2 + 3))
            .map(|k| range(0x1_0000 + (k as u64) * PAGE_BYTES, 1 + k as u64, 0, PAGE_BYTES))
            .collect();
        let chunks = chunks(9, &ranges).expect("chunk");
        assert_eq!(chunks.len(), 3);
        let mut rebuilt = Vec::new();
        for (index, bytes) in chunks.iter().enumerate() {
            let chunk = decode(bytes).expect("decode chunk");
            assert_eq!(usize::from(chunk.index), index);
            assert_eq!(usize::from(chunk.count), chunks.len());
            rebuilt.extend(chunk.ranges);
        }
        assert_eq!(rebuilt, ranges, "chunking preserves every range in order");
        for bytes in &chunks {
            assert!(bytes.len() <= MAX_PAYLOAD_LEN, "every chunk fits the envelope");
        }
    }

    #[test]
    fn an_empty_generation_is_one_empty_chunk() {
        let chunks = chunks(4, &[]).expect("chunk");
        assert_eq!(chunks.len(), 1);
        let chunk = decode(&chunks[0]).expect("decode");
        assert!(chunk.ranges.is_empty());
        assert_eq!(chunk.generation, 4);
    }

    #[test]
    fn malformed_payloads_are_refused() {
        let good = encode(1, 0, 1, &[range(0x1_0000, 9, 0, PAGE_BYTES)]).expect("encode");
        assert_eq!(decode(&good[..HEADER_LEN - 1]), Err(TitleMapError::TruncatedHeader));

        let mut version = good.clone();
        version[0] = 2;
        assert_eq!(decode(&version), Err(TitleMapError::UnsupportedVersion(2)));

        let mut reserved = good.clone();
        reserved[1] = 1;
        assert!(matches!(decode(&reserved), Err(TitleMapError::Malformed(_))));

        let mut count = good.clone();
        count[8..10].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(decode(&count), Err(TitleMapError::Malformed(_))));

        let mut length = good.clone();
        length.push(0);
        assert_eq!(decode(&length), Err(TitleMapError::Length));

        let mut unaligned = good.clone();
        unaligned[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&0x1_0001u64.to_le_bytes());
        assert!(matches!(decode(&unaligned), Err(TitleMapError::InvalidEntry(_))));

        let mut zero_length = encode(1, 0, 1, &[range(0x1_0000, 9, 0, PAGE_BYTES)]).expect("encode");
        zero_length[HEADER_LEN + 24..HEADER_LEN + 32].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(decode(&zero_length), Err(TitleMapError::InvalidEntry(_))));
    }

    #[test]
    fn an_overlapping_range_is_not_silently_clamped() {
        let ranges = [range(0x1_0000, 1, 0, 0x4_0000)];
        let bytes = encode(1, 0, 1, &ranges).expect("encode");
        assert_eq!(decode(&bytes).expect("decode").ranges, ranges);
        // The codec carries what the guest named; overlap policy belongs to
        // the host map, which refuses a conflicting conflict explicitly.
    }

    #[test]
    fn entries_past_the_address_limit_are_refused() {
        let at_limit = ADDRESS_LIMIT - PAGE_BYTES;
        assert_eq!(range(at_limit, 1, 0, PAGE_BYTES).validate(), Ok(()));
        assert!(matches!(
            range(ADDRESS_LIMIT, 1, 0, PAGE_BYTES).validate(),
            Err(TitleMapError::InvalidEntry(_))
        ));
        let mut bytes = encode(1, 0, 1, &[range(at_limit, 1, 0, PAGE_BYTES)]).expect("encode");
        bytes[HEADER_LEN + 24..HEADER_LEN + 32]
            .copy_from_slice(&(2 * PAGE_BYTES).to_le_bytes());
        assert!(matches!(decode(&bytes), Err(TitleMapError::InvalidEntry(_))));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn old_frames_keep_their_bytes_and_kind() {
        // The new kind shares its number with no existing frame kind, and a
        // wire submission payload is never a valid title-map chunk: an old
        // host that receives kind 11 refuses it by name instead of decoding
        // it as a submission.
        let existing = [
            crate::ipc_host::KIND_AUTH,
            crate::ipc_host::KIND_REGISTER,
            crate::ipc_host::KIND_SUBMISSION,
            crate::ipc_host::KIND_PIPELINE,
            crate::ipc_host::KIND_SUBMISSION_V3,
            crate::ipc_host::KIND_FENCE_PAGE,
            crate::ipc_host::KIND_SUBMISSION_ASYNC,
            crate::ipc_host::KIND_TEXTURE_REG,
            crate::ipc_host::KIND_SURFACE_REG,
            crate::ipc_host::KIND_SAMPLER_REG,
        ];
        assert!(!existing.contains(&KIND_TITLE_MAP));
        let submission = crate::wire::Submission {
            commands: Vec::new(),
            sequence: 1,
        };
        let frame = crate::wire::encode_submission(&submission).expect("encode");
        assert_ne!(decode(&frame).map(|_| ()), Ok(()), "a submission is not a chunk");
        assert_eq!(
            crate::wire::decode_submission(&frame).expect("submission").sequence,
            1,
            "the existing frame still decodes exactly as before"
        );
    }

    #[test]
    fn the_range_and_page_caps_are_enforced() {
        let too_many = vec![range(0x1_0000, 1, 0, PAGE_BYTES); MAX_RANGES + 1];
        assert!(matches!(chunks(1, &too_many), Err(TitleMapError::TooMany(_))));
        let too_many_pages = vec![range(0x1_0000, 1, 0, (MAX_MAPPED_PAGES + 1) * PAGE_BYTES)];
        assert!(matches!(chunks(1, &too_many_pages), Err(TitleMapError::TooMany(_))));
    }
}
