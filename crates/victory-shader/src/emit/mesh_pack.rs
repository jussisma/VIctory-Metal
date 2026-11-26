//! Whole-signature mesh output packing for pairs above Metal's 32 KiB limit.
//!
//! Metal counts a translated mesh shader's outputs not by their packed
//! component count but by its aligned layout: the clip/cull distances as
//! 4-byte scalars, the position (16 bytes), then every other signature
//! element in emitted order, each rounded up to its own size (4, 8, 12 or 16
//! bytes). A signature whose packed count fits can therefore still overrun the
//! limit, which is what the title's refused mesh/pixel pairs show (#172,
//! `linked_mesh_budget_uses_metal_aligned_layout`).
//!
//! [`mesh_output_bytes`] mirrors the runtime's own check
//! (`victory_gpu::pipeline::host::metal_mesh_bytes` and `mesh_output_bytes`) so
//! the linked path refuses a pair it cannot fit instead of emitting an
//! over-budget signature the host then drops.
//!
//! The packers below are deliberately expressed over signature elements, the
//! same shape [`crate::emit::lower::mesh`] assigns rows to, so the estimator
//! and the packing share one definition of Metal's cost.

use std::collections::HashMap;

use crate::dxbc::{FourCC, u32_at};
use crate::{Container, ShaderError};

use super::container::SigElement;

/// Metal's limit on a mesh shader's outputs per thread group, in bytes (#132).
///
/// Re-exported from [`crate::emit::METAL_MESH_OUTPUT_BYTES`]; kept here so the
/// estimator's callers do not need both modules.
pub(crate) const METAL_MESH_OUTPUT_BYTES: u32 = crate::emit::METAL_MESH_OUTPUT_BYTES;

/// Components of one element as Metal declares it: its mask's population.
fn component_count(mask: u8) -> u32 {
    u32::from(mask.count_ones())
}

/// Metal's aligned end for `sizes` laid out in order from `end`: every element
/// is rounded up to its own size (4, 8, 12 or 16 bytes).
fn fold(sizes: impl Iterator<Item = u32>, mut end: u32) -> u32 {
    for size in sizes {
        if size == 0 {
            continue;
        }
        end = end.next_multiple_of(size) + size;
    }
    end
}

/// One signature part's `(kind, mask)` rows in the order the container stores
/// them, which is the order Metal lays them out in.
fn part_elements(part: Option<&[u8]>) -> Result<Vec<(u32, u8)>, ShaderError> {
    let Some(data) = part else {
        return Ok(Vec::new());
    };
    let count = u32_at(data, 0)? as usize;
    let base = u32_at(data, 4)? as usize;
    let mut elements = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let kind = u32_at(data, at + 12)?;
        let mask = *data
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated signature row"))?;
        elements.push((kind, mask));
    }
    Ok(elements)
}

/// The position system value's DXIL kind: it always occupies 16 bytes in the
/// aligned layout, whatever its mask.
const POSITION: u32 = 1;
/// Clip and cull distances: counted once as 4-byte scalars ahead of the
/// position, and again as ordinary elements in emitted order.
const DISTANCES: [u32; 2] = [2, 3];

/// Metal's aligned bytes for one vertex's outputs: distances as scalars, the
/// position, then the remaining elements in emitted order.
fn vertex_bytes(elements: &[(u32, u8)]) -> u32 {
    let distances: u32 = elements
        .iter()
        .filter(|(kind, _)| DISTANCES.contains(kind))
        .map(|&(_, mask)| component_count(mask))
        .sum();
    fold(
        std::iter::repeat_n(4u32, distances as usize)
            .chain([16])
            .chain(
                elements
                    .iter()
                    .filter(|(kind, _)| *kind != POSITION)
                    .map(|&(_, mask)| 4 * component_count(mask)),
            ),
        0,
    )
}

/// Metal's aligned bytes for one primitive's outputs: every element in emitted
/// order, its size rounded to its own alignment.
fn primitive_bytes(elements: &[(u32, u8)]) -> u32 {
    fold(
        elements.iter().map(|&(_, mask)| 4 * component_count(mask)),
        0,
    )
}

/// Metal's aligned bytes for one vertex's outputs, from typed elements.
pub(crate) fn vertex_aligned(elements: &[SigElement]) -> u32 {
    let distances: u32 = elements
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7))
        .map(|e| u32::from(e.cols))
        .sum();
    fold(
        std::iter::repeat_n(4u32, distances as usize)
            .chain([16])
            .chain(
                elements
                    .iter()
                    .filter(|e| e.kind != 3)
                    .map(element_bytes),
            ),
        0,
    )
}

/// Metal's aligned bytes for one primitive's outputs, from typed elements.
pub(crate) fn primitive_aligned(elements: &[SigElement]) -> u32 {
    fold(elements.iter().map(element_bytes), 0)
}

/// The declared `(vertices, primitives)` of a mesh container, from `PSV0`.
pub(crate) fn declared_mesh_counts(dxil: &[u8]) -> Result<Option<(u32, u32)>, ShaderError> {
    mesh_counts(dxil)
}

/// The declared maxima and whether the container is a mesh shader, from
/// `PSV0`'s runtime info.
fn mesh_counts(dxil: &[u8]) -> Result<Option<(u32, u32)>, ShaderError> {
    let Ok(container) = Container::parse(dxil) else {
        return Ok(None);
    };
    let Some(psv) = container.part(FourCC::PSV0) else {
        return Ok(None);
    };
    let size = u32_at(psv.data, 0)? as usize;
    let Some(info) = psv.data.get(4..4 + size).filter(|i| i.len() >= 33) else {
        return Ok(None);
    };
    // Shader kind 13: mesh.
    if info[24] != 13 {
        return Ok(None);
    }
    let half = |at: usize| u32::from(u16::from_le_bytes([info[at], info[at + 1]]));
    Ok(Some((half(12), half(14))))
}

/// Output bytes as Apple's Metal counts them for a translated mesh shader
/// (measured on D3DMetal): per vertex, the distances, position and remaining
/// outputs in emitted order aligned to their own size; likewise per primitive;
/// multiplied by the declared maxima. `None` for a non-mesh container.
pub(crate) fn aligned_mesh_bytes(dxil: &[u8]) -> Result<Option<u32>, ShaderError> {
    let Some((vertices, primitives)) = mesh_counts(dxil)? else {
        return Ok(None);
    };
    let Ok(container) = Container::parse(dxil) else {
        return Ok(None);
    };
    let vertex = part_elements(container.part(FourCC::OSG1).map(|p| p.data))?;
    let primitive = part_elements(container.part(FourCC::PSG1).map(|p| p.data))?;
    Ok(Some(
        vertices
            .saturating_mul(vertex_bytes(&vertex))
            .saturating_add(primitives.saturating_mul(primitive_bytes(&primitive))),
    ))
}

/// The packed output bytes [`crate::emit::MeshOutputs::bytes`] counts: 4 per
/// declared component of every register-bearing vertex and primitive output.
pub(crate) fn packed_mesh_bytes(dxil: &[u8]) -> Option<u32> {
    crate::emit::mesh_outputs(dxil).map(|o| o.bytes())
}

/// The larger of the packed footprint and Metal's aligned layout: the runtime's
/// linking and refusal threshold.
pub(crate) fn mesh_output_bytes(dxil: &[u8]) -> Result<Option<u32>, ShaderError> {
    let packed = packed_mesh_bytes(dxil);
    let aligned = aligned_mesh_bytes(dxil)?;
    Ok(match (packed, aligned) {
        (Some(p), Some(a)) => Some(p.max(a)),
        (p, a) => p.or(a),
    })
}

/// Whether a translated mesh shader fits Metal's output budget by the runtime's
/// own count. Malformed containers are reported, not silently fitted.
pub(crate) fn fits_mesh_output_budget(dxil: &[u8]) -> Result<bool, ShaderError> {
    Ok(
        mesh_output_bytes(dxil)?
            .is_none_or(|bytes| bytes <= METAL_MESH_OUTPUT_BYTES),
    )
}

// ---------------------------------------------------------------------------
// Linked-pair proof.
//
// The linked path (`crate::emit::translate_linked`) already drops arbitrary
// mesh outputs no pixel shader input reads and then packs the survivors. The
// proof below performs the same decision on two emitted `DXIL` containers and
// reports what would be carried, so a caller can prove a pair fits without a
// device, and see exactly which rows and channels were given up.
//
// It is deliberately more conservative than the runtime path: a row is only
// pruned when the paired pixel container proves it reads none of its channels.
// If the pixel stage cannot be identified, the signature is left whole and the
// pair is refused (`fits == false`) rather than truncated. System values are
// never pruned, and a retained arbitrary row keeps its semantic, index, guest
// register row and every channel the pixel shader reads.

/// Why [`mesh_pack_proof`] does not carry a mesh output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshPruneReason {
    /// No pixel shader input (`ISG1`) reads any channel of this semantic.
    UnreadByPixel,
    /// An arbitrary row with no guest register: no export can name it.
    Unallocated,
}

/// One `OSG1` row's outcome under [`mesh_pack_proof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshRowProof {
    pub name: String,
    pub index: u32,
    /// DXIL semantic kind (mapped the same way the emitter's signature reader
    /// maps it).
    pub kind: u8,
    /// The guest register row, before packing.
    pub row: Option<u32>,
    /// The channel span carried (trimmed for a retained arbitrary row).
    pub col: u8,
    pub cols: u8,
    /// The channel span the signature declared, before compaction.
    pub declared_col: u8,
    pub declared_cols: u8,
    /// The paired pixel shader's read mask, absolute within the four channels.
    pub read: u8,
    /// Why the row was dropped; `None` when it is retained.
    pub reason: Option<MeshPruneReason>,
}

impl MeshRowProof {
    /// Whether the row was retained with a narrower channel span than declared.
    pub fn compacted(&self) -> bool {
        self.reason.is_none() && (self.col != self.declared_col || self.cols != self.declared_cols)
    }
}

/// The linked mesh/pixel footprint before and after dead-export pruning and
/// packing, as [`mesh_pack_proof`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshPackProof {
    pub vertices: u32,
    pub primitives: u32,
    /// Metal's limit, [`crate::emit::METAL_MESH_OUTPUT_BYTES`].
    pub limit: u32,
    /// 4 bytes per declared register-bearing component, before pruning.
    pub declared_packed: u32,
    /// Metal's aligned layout before pruning (the runtime's larger count).
    pub declared_aligned: u32,
    /// The same two counts after pruning, channel compaction and packing.
    pub packed_after: u32,
    pub aligned_after: u32,
    /// Whether the paired pixel stage was identified well enough to prove
    /// which rows it does not read. `false` keeps every row and refuses.
    pub linkage_proven: bool,
    /// `max(packed_after, aligned_after) <= limit`, and the linkage proven.
    pub fits: bool,
    /// Rows carried into the packed signature, before wide-row splitting.
    pub retained: Vec<MeshRowProof>,
    /// Rows not carried, with their reason.
    pub pruned: Vec<MeshRowProof>,
    /// The packed vertex outputs: `(register row, col, cols, kind)`.
    pub rows_after: Vec<(u32, u8, u8, u8)>,
}

/// A parsed `OSG1`/`PSG1` row and the channel mask its record declares.
struct SpanRow {
    element: SigElement,
    mask: u8,
}

/// The semantic-kind mapping the emitter's signature reader uses; a row the
/// proof cannot classify is reported rather than guessed at.
fn mapped_kind(raw: u32) -> Result<u8, ShaderError> {
    Ok(match raw {
        0 => 0,
        1 => 3,
        2 => 6,
        3 => 7,
        4 => 4,
        5 => 5,
        6 => 1,
        7 => 10,
        8 => 2,
        9 => 13,
        10 => 12,
        23 => 28,
        25 => 30,
        64 => 16,
        65 => 17,
        66 => 14,
        67 => 19,
        68 => 18,
        69 => 20,
        70 => 15,
        _ => {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a system value in the proof signature",
            });
        }
    })
}

/// The rows of an `OSG1`/`PSG1`/`ISG1` part, with their declared channel mask.
fn read_rows(part: &[u8]) -> Result<Vec<SpanRow>, ShaderError> {
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated signature row"))?;
        let register = u32_at(part, at + 20)?;
        out.push(SpanRow {
            element: SigElement {
                name: String::from_utf8_lossy(name).into_owned(),
                index: u32_at(part, at + 8)?,
                kind: mapped_kind(u32_at(part, at + 12)?)?,
                comp: 9,
                interp: 0,
                row: (register != u32::MAX).then_some(register),
                col: mask.trailing_zeros().min(3) as u8,
                cols: mask.count_ones().max(1) as u8,
                usage: 0,
                half: false,
            },
            mask,
        });
    }
    Ok(out)
}

/// The packed count the runtime uses: 4 bytes per declared, register-bearing
/// component of each declared vertex and primitive.
fn packed_count(
    outputs: &[SigElement],
    prims: &[SigElement],
    vertices: u32,
    primitives: u32,
) -> u32 {
    let components = |els: &[SigElement]| -> u32 {
        els.iter()
            .filter(|e| e.row.is_some())
            .map(|e| u32::from(e.cols))
            .sum()
    };
    4u32.saturating_mul(
        components(outputs)
            .saturating_mul(vertices)
            .saturating_add(components(prims).saturating_mul(primitives)),
    )
}

/// Metal's aligned layout for the declared counts.
fn aligned_count(
    outputs: &[SigElement],
    prims: &[SigElement],
    vertices: u32,
    primitives: u32,
) -> u32 {
    vertices
        .saturating_mul(vertex_aligned(outputs))
        .saturating_add(primitives.saturating_mul(primitive_aligned(prims)))
}

/// Whether a container is a pixel shader: `PSV0`'s shader kind is 0. `None`
/// when the container, part or runtime info cannot be read.
fn pixel_stage(dxil: &[u8]) -> Option<bool> {
    let c = Container::parse(dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?;
    let size = u32_at(psv.data, 0).ok()? as usize;
    let info = psv.data.get(4..4 + size).filter(|i| i.len() >= 33)?;
    Some(info[24] == 0)
}

/// The pixel inputs' read masks, or `None` when the linkage cannot be proven.
fn proven_pixel_reads(pixel: &[u8]) -> Option<super::lower::KeptOutputs> {
    if pixel_stage(pixel) != Some(true) {
        return None;
    }
    super::pixel_input_reads(pixel).ok()
}

/// The rows a signature declared, as proof records (all retained).
fn whole_rows(rows: &[SpanRow], reads: Option<&super::lower::KeptOutputs>) -> Vec<MeshRowProof> {
    rows.iter()
        .map(|span| {
            let e = &span.element;
            let read = reads
                .map(|r| {
                    r.get(&(e.name.to_ascii_uppercase(), e.index))
                        .copied()
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            MeshRowProof {
                name: e.name.clone(),
                index: e.index,
                kind: e.kind,
                row: e.row,
                col: e.col,
                cols: e.cols,
                declared_col: e.col,
                declared_cols: e.cols,
                read,
                reason: None,
            }
        })
        .collect()
}

/// Proves what a mesh/pixel `DXIL` pair would carry under the linked packer.
///
/// `mesh` is an emitted mesh container and `pixel` its emitted pixel consumer
/// (`None` for a depth-only draw). Returns `None` for a non-mesh container.
/// The proof prunes an arbitrary mesh output row only when the pixel stage is
/// positively identified and reads none of its channels; a retained row keeps
/// its semantic, index and guest register row, and its declared channel span is
/// narrowed only to the read channels. When the pixel linkage cannot be
/// proven, every row is kept and [`MeshPackProof::fits`] is `false`.
pub fn mesh_pack_proof(
    mesh: &[u8],
    pixel: Option<&[u8]>,
) -> Result<Option<MeshPackProof>, ShaderError> {
    let Some((vertices, primitives)) = declared_mesh_counts(mesh)? else {
        return Ok(None);
    };
    let c = Container::parse(mesh)?;
    let outputs = match c.part(FourCC::OSG1) {
        Some(part) => read_rows(part.data)?,
        None => Vec::new(),
    };
    let prims = match c.part(FourCC::PSG1) {
        Some(part) => read_rows(part.data)?,
        None => Vec::new(),
    };
    let output_elements: Vec<SigElement> = outputs.iter().map(|r| r.element.clone()).collect();
    let prim_elements: Vec<SigElement> = prims.iter().map(|r| r.element.clone()).collect();
    let declared_packed = packed_count(&output_elements, &prim_elements, vertices, primitives);
    let declared_aligned = aligned_count(&output_elements, &prim_elements, vertices, primitives);

    // A depth-only draw has no consumer, so no arbitrary output is observable.
    // An absent or non-pixel consumer proves nothing: keep every row.
    let (reads, linkage_proven) = match pixel {
        None => (Some(super::lower::KeptOutputs::new()), true),
        Some(bytes) => match proven_pixel_reads(bytes) {
            Some(reads) => (Some(reads), true),
            None => (None, false),
        },
    };
    if !linkage_proven {
        let retained = whole_rows(&outputs, None);
        return Ok(Some(MeshPackProof {
            vertices,
            primitives,
            limit: METAL_MESH_OUTPUT_BYTES,
            declared_packed,
            declared_aligned,
            packed_after: declared_packed,
            aligned_after: declared_aligned,
            linkage_proven: false,
            fits: false,
            retained,
            pruned: Vec::new(),
            rows_after: output_elements
                .iter()
                .filter_map(|e| e.row.map(|row| (row, e.col, e.cols, e.kind)))
                .collect(),
        }));
    }
    let reads = reads.expect("proven linkage has reads");

    let mut retained = Vec::new();
    let mut retained_proofs = Vec::new();
    let mut pruned = Vec::new();
    let mut channel_mismatch = false;
    for span in &outputs {
        let e = &span.element;
        let read = reads
            .get(&(e.name.to_ascii_uppercase(), e.index))
            .copied()
            .unwrap_or(0);
        let base = MeshRowProof {
            name: e.name.clone(),
            index: e.index,
            kind: e.kind,
            row: e.row,
            col: e.col,
            cols: e.cols,
            declared_col: e.col,
            declared_cols: e.cols,
            read,
            reason: None,
        };
        if e.kind != 0 {
            // System values are observed by the pipeline itself (position,
            // clip/cull, targets): never prune.
            retained.push(e.clone());
            retained_proofs.push(base);
            continue;
        }
        if e.row.is_none() {
            // No register means no export, but a pixel that still reads the
            // semantic cannot be linked: report the pair as unproven rather
            // than silently dropping a row it depends on.
            if read != 0 {
                channel_mismatch = true;
            }
            pruned.push(MeshRowProof {
                reason: Some(MeshPruneReason::Unallocated),
                ..base
            });
            continue;
        }
        if read == 0 {
            pruned.push(MeshRowProof {
                reason: Some(MeshPruneReason::UnreadByPixel),
                ..base
            });
            continue;
        }
        if read & !span.mask != 0 {
            // The pixel declares a channel this row does not: not provable.
            // Keep the row whole; the whole pair is refused below.
            channel_mismatch = true;
            retained.push(e.clone());
            retained_proofs.push(base);
            continue;
        }
        // Channel compaction within the declared span: keep the contiguous
        // range from the first to the last channel the pixel reads. Unread
        // edge channels are not observable; interior gaps stay because the
        // declared semantic width does not let us renumber channels.
        let rel = ((read >> e.col) as u16) & ((1u16 << u32::from(e.cols)) - 1);
        let first = rel.trailing_zeros() as u8;
        let last = 15 - rel.leading_zeros() as u8;
        let mut carried = e.clone();
        carried.col = e.col + first;
        carried.cols = last - first + 1;
        let proof = MeshRowProof {
            col: carried.col,
            cols: carried.cols,
            ..base
        };
        retained.push(carried);
        retained_proofs.push(proof);
    }
    if channel_mismatch {
        // A read outside the declared span means the pair cannot be linked as
        // claimed: report the declared signature and refuse it, untrimmed.
        return Ok(Some(MeshPackProof {
            vertices,
            primitives,
            limit: METAL_MESH_OUTPUT_BYTES,
            declared_packed,
            declared_aligned,
            packed_after: declared_packed,
            aligned_after: declared_aligned,
            linkage_proven: false,
            fits: false,
            retained: whole_rows(&outputs, None),
            pruned: Vec::new(),
            rows_after: output_elements
                .iter()
                .filter_map(|e| e.row.map(|row| (row, e.col, e.cols, e.kind)))
                .collect(),
        }));
    }

    let mut after = retained.clone();
    split_wide_rows(&mut after);
    pack_within_rows(&mut after);
    if let Some(order) = row_order(&after) {
        for e in &mut after {
            if let Some(emitted) = e.row.and_then(|row| order.get(&row).copied()) {
                e.row = Some(emitted);
            }
        }
        after.sort_by_key(|e| e.row.unwrap_or(u32::MAX));
    }
    let packed_after = packed_count(&after, &prim_elements, vertices, primitives);
    let aligned_after = aligned_count(&after, &prim_elements, vertices, primitives);
    Ok(Some(MeshPackProof {
        vertices,
        primitives,
        limit: METAL_MESH_OUTPUT_BYTES,
        declared_packed,
        declared_aligned,
        packed_after,
        aligned_after,
        linkage_proven: true,
        fits: packed_after.max(aligned_after) <= METAL_MESH_OUTPUT_BYTES,
        retained: retained_proofs,
        pruned,
        rows_after: after
            .iter()
            .filter_map(|e| e.row.map(|row| (row, e.col, e.cols, e.kind)))
            .collect(),
    }))
}

// ---------------------------------------------------------------------------
// Whole-signature packing.
//
// A mesh output signature is written to `OSG1` sorted by register row (stable
// within a row), and Metal lays that emitted order out element by element.
// The packer has two levers: the order of the elements inside each register
// row, and the row numbers themselves (which select the order the rows appear
// in). Splitting a single signature element that declares more channels than
// one 4-channel Metal row can carry into one element per row is the third.
//
// Rows are only ever renumbered among the register numbers the signature
// already uses, so the guest -> emitted map stays a bijection on rows and the
// export lookup (`Lower::emitted_row`) keeps resolving every channel.

/// One element's size as Metal lays a mesh output out: 4 bytes per declared
/// component (its mask's components).
pub(crate) fn element_bytes(e: &SigElement) -> u32 {
    4 * u32::from(e.cols)
}

/// Whether `elements` come in the order the container writes them (by
/// register, unallocated last). The packers only touch such vectors.
fn register_sorted(elements: &[SigElement]) -> bool {
    elements
        .windows(2)
        .all(|w| w[0].row.unwrap_or(u32::MAX) <= w[1].row.unwrap_or(u32::MAX))
}

/// The fixed start of a vertex's output layout: every clip/cull distance
/// component as a 4-byte scalar, then the 16-byte position. Distances are
/// counted again in emitted order, as the host counts them.
pub(crate) fn head_bytes(elements: &[SigElement]) -> u32 {
    let distances: u32 = elements
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7))
        .map(|e| u32::from(e.cols))
        .sum();
    4 * distances + 16
}

/// The order of `sizes` that leaves the smallest aligned end after `start`.
/// A register row's elements usually number at most four, which is tried
/// exhaustively (ties keep the declared order); wider rows try a bounded set
/// of size orders.
fn least_padded(sizes: &[u32], start: u32) -> Vec<usize> {
    let identity: Vec<usize> = (0..sizes.len()).collect();
    if sizes.len() <= 7 {
        fn walk(
            sizes: &[u32],
            start: u32,
            order: &mut Vec<usize>,
            used: &mut [bool],
            best: &mut (u32, Vec<usize>),
        ) {
            if order.len() == sizes.len() {
                let seq: Vec<u32> = order.iter().map(|&k| sizes[k]).collect();
                let end = fold(seq.iter().copied(), start);
                if end < best.0 {
                    *best = (end, order.clone());
                }
                return;
            }
            for k in 0..sizes.len() {
                if used[k] {
                    continue;
                }
                used[k] = true;
                order.push(k);
                walk(sizes, start, order, used, best);
                order.pop();
                used[k] = false;
            }
        }
        let mut best = (u32::MAX, Vec::new());
        walk(
            sizes,
            start,
            &mut Vec::with_capacity(sizes.len()),
            &mut vec![false; sizes.len()],
            &mut best,
        );
        return best.1;
    }
    let mut widest = identity.clone();
    widest.sort_by_key(|&k| std::cmp::Reverse(sizes[k]));
    let mut narrowest = identity.clone();
    narrowest.sort_by_key(|&k| sizes[k]);
    let mut best = (u32::MAX, identity.clone());
    for order in [identity, widest, narrowest] {
        let seq: Vec<u32> = order.iter().map(|&k| sizes[k]).collect();
        let end = fold(seq.iter().copied(), start);
        if end < best.0 {
            best = (end, order);
        }
    }
    best.1
}

/// Splits a signature element that declares more channels than one Metal row
/// can carry (a console mask of more than four components, or components that
/// run past the row's fourth channel) into one element per 4-channel row. The
/// first chunk keeps the guest register row; the rest take fresh rows above
/// every used register, which [`row_order`] then places. Semantics, indexes,
/// component types and interpolation are unchanged, and the total declared
/// component count is preserved.
pub(crate) fn split_wide_rows(elements: &mut Vec<SigElement>) {
    if !register_sorted(elements) {
        return;
    }
    let mut next_row = elements
        .iter()
        .filter_map(|e| e.row)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    // Register rows are `u8` in `OSG1`/`PSV0`; do not split past what they hold.
    if next_row > u32::from(u8::MAX) {
        return;
    }
    let mut out = Vec::with_capacity(elements.len());
    for e in elements.drain(..) {
        let end = u32::from(e.col) + u32::from(e.cols);
        if e.row.is_none() || e.kind != 0 || end <= 4 {
            out.push(e);
            continue;
        }
        let mut col = e.col;
        let mut remaining = e.cols;
        let mut first = true;
        while remaining > 0 {
            let room = 4u32.saturating_sub(u32::from(col)).max(1) as u8;
            let take = remaining.min(room);
            let mut chunk = e.clone();
            chunk.col = col;
            chunk.cols = take;
            if !first {
                chunk.row = Some(next_row);
                next_row = next_row.saturating_add(1);
            }
            out.push(chunk);
            remaining -= take;
            first = false;
            col = 0;
        }
    }
    *elements = out;
}

/// Reorders the arbitrary (`kind == 0`) elements **within each register row**
/// to the order Metal's aligned layout counts smallest. The host lays mesh
/// outputs out as the clip/cull distances, the position, then every other
/// element in emitted order, each rounded up to its own size. Trimming cannot
/// remove the padding between elements; their order can.
///
/// Elements never move across a row here, so a guest row still names the same
/// elements for a mesh export, the vector stays register-sorted, and each
/// element's index therefore stays the signature id its emitted store names.
pub(crate) fn pack_within_rows(elements: &mut [SigElement]) {
    if !register_sorted(elements) {
        return;
    }
    let mut end = head_bytes(elements);
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == row {
            j += 1;
        }
        if row.is_some() && j - i >= 2 && elements[i..j].iter().all(|e| e.kind == 0) {
            let sizes: Vec<u32> = elements[i..j].iter().map(element_bytes).collect();
            let order = least_padded(&sizes, end);
            if order.iter().enumerate().any(|(at, &k)| at != k) {
                let before: Vec<SigElement> = elements[i..j].to_vec();
                for (at, &k) in order.iter().enumerate() {
                    elements[i + at] = before[k].clone();
                }
            }
        }
        let sizes: Vec<u32> = elements[i..j]
            .iter()
            .filter(|e| e.kind != 3)
            .map(element_bytes)
            .collect();
        end = fold(sizes.iter().copied(), end);
        i = j;
    }
}

/// One register row's slot in the emitted signature.
struct Block {
    row: u32,
    sizes: Vec<u32>,
    fixed: bool,
}

/// Whole-signature row ordering: guest register row -> emitted row, chosen to
/// shrink Metal's aligned output layout. `None` when nothing moves or the
/// signature is outside what is proven safe.
///
/// Rows move as whole registers (`pack_within_rows` has already ordered each
/// row's elements), so every element keeps its semantic, index, channel mask,
/// component type and interpolation mode, and a guest row still names the same
/// elements. The position row and colour-target rows stay where they are: the
/// position is laid out specially and an `SV_Target` export names its register
/// directly. Every other row (arbitrary outputs, clip/cull distances and the
/// other system values) moves among the register numbers the signature already
/// uses, which is exactly what the export lookup resolves through
/// [`crate::emit::lower::mesh::row_order`]'s map.
pub(crate) fn row_order(elements: &[SigElement]) -> Option<HashMap<u32, u32>> {
    if !register_sorted(elements) {
        return None;
    }
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == row {
            j += 1;
        }
        blocks.push(Block {
            row: row.unwrap_or(u32::MAX),
            sizes: elements[i..j]
                .iter()
                .filter(|e| e.kind != 3)
                .map(element_bytes)
                .collect(),
            fixed: row.is_none()
                || elements[i..j]
                    .iter()
                    .any(|e| matches!(e.kind, 3 | 16)),
        });
        i = j;
    }
    let movable: Vec<usize> = (0..blocks.len()).filter(|&b| !blocks[b].fixed).collect();
    if movable.len() < 2 {
        return None;
    }
    let mut slots: Vec<(u32, Option<usize>)> = blocks
        .iter()
        .enumerate()
        .map(|(b, block)| (block.row, (!block.fixed).then_some(b)))
        .collect();
    slots.sort_by_key(|(row, _)| *row);
    let fixed: HashMap<u32, &[u32]> = blocks
        .iter()
        .filter(|b| b.fixed)
        .map(|b| (b.row, b.sizes.as_slice()))
        .collect();
    let head = head_bytes(elements);
    let end = |order: &[usize]| -> u32 {
        let mut end = head;
        let mut at = 0;
        for (row, movable_slot) in &slots {
            let sizes: &[u32] = match movable_slot {
                Some(_) => blocks[movable[*order.get(at).unwrap_or(&at)]].sizes.as_slice(),
                None => fixed.get(row).copied().unwrap_or(&[]),
            };
            if movable_slot.is_some() {
                at += 1;
            }
            end = fold(sizes.iter().copied(), end);
        }
        end
    };
    let n = movable.len();
    let identity: Vec<usize> = (0..n).collect();
    let start = end(&identity);
    let mut best = start;
    let mut best_order = identity.clone();

    if n <= 18 {
        // Exact Held-Karp over the movable rows. Fixed slots are folded as the
        // sequence reaches them; a state's minimal end dominates because the
        // fold is monotone in its start.
        let positions: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| m.is_some())
            .map(|(i, _)| i)
            .collect();
        let full = (1usize << n) - 1;
        let mut best_end = vec![u32::MAX; full + 1];
        let mut choice = vec![usize::MAX; full + 1];
        // Fold the fixed slots before the first movable slot.
        let mut base = head;
        for (_, (row, m)) in slots.iter().enumerate().take(positions[0]) {
            if m.is_none() {
                base = fold(fixed.get(row).copied().unwrap_or(&[]).iter().copied(), base);
            }
        }
        best_end[0] = base;
        for mask in 0..=full {
            let cost = best_end[mask];
            if cost == u32::MAX {
                continue;
            }
            let k = mask.count_ones() as usize;
            if k == n {
                continue;
            }
            // The chosen block takes the k-th movable slot; then fold the
            // fixed slots up to the next movable slot.
            let stop = if k + 1 < n { positions[k + 1] } else { slots.len() };
            for j in 0..n {
                let bit = 1usize << j;
                if mask & bit != 0 {
                    continue;
                }
                let mut candidate = fold(blocks[movable[j]].sizes.iter().copied(), cost);
                for (_, (row, m)) in slots.iter().enumerate().take(stop).skip(positions[k] + 1) {
                    if m.is_none() {
                        candidate =
                            fold(fixed.get(row).copied().unwrap_or(&[]).iter().copied(), candidate);
                    }
                }
                if candidate < best_end[mask | bit] {
                    best_end[mask | bit] = candidate;
                    choice[mask | bit] = j;
                }
            }
        }
        let cost = best_end[full];
        if cost < best {
            best = cost;
            let mut order = vec![usize::MAX; n];
            let mut mask = full;
            for k in (0..n).rev() {
                let j = choice[mask];
                order[k] = j;
                mask &= !(1usize << j);
            }
            best_order = order;
        }
    } else {
        // Too many rows for the exact search: several seeds (largest first,
        // smallest first, declared) plus bounded best-improvement swaps and
        // moves, as the per-row packer does.
        let mut seeds = vec![identity.clone()];
        let mut by_size = identity.clone();
        by_size.sort_by_key(|&b| {
            std::cmp::Reverse(blocks[movable[b]].sizes.iter().sum::<u32>())
        });
        seeds.push(by_size);
        let mut by_lead = identity.clone();
        by_lead.sort_by_key(|&b| {
            std::cmp::Reverse(blocks[movable[b]].sizes.first().copied().unwrap_or(0))
        });
        seeds.push(by_lead);
        for mut order in seeds {
            let mut cost = end(&order);
            for _ in 0..n * n {
                let mut moved = None;
                for a in 0..n {
                    for b in (a + 1)..n {
                        let mut trial = order.clone();
                        trial.swap(a, b);
                        let c = end(&trial);
                        if c < cost {
                            cost = c;
                            moved = Some(trial);
                        }
                        let mut trial = order.clone();
                        let x = trial.remove(a);
                        trial.insert(b, x);
                        let c = end(&trial);
                        if c < cost {
                            cost = c;
                            moved = Some(trial);
                        }
                    }
                }
                match moved {
                    Some(trial) => order = trial,
                    None => break,
                }
            }
            if cost < best {
                best = cost;
                best_order = order;
            }
        }
    }
    if best >= start {
        return None;
    }
    let movable_slots: Vec<u32> = slots
        .iter()
        .filter_map(|(row, m)| m.map(|_| *row))
        .collect();
    Some(
        best_order
            .iter()
            .enumerate()
            .map(|(at, &b)| (blocks[movable[b]].row, movable_slots[at]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxbc::build;

    /// One `OSG1`/`PSG1` row: kind, register row, mask.
    fn row(kind: u32, register: u32, mask: u8) -> [u8; 32] {
        let mut rec = [0u8; 32];
        rec[12..16].copy_from_slice(&kind.to_le_bytes());
        rec[20..24].copy_from_slice(&register.to_le_bytes());
        rec[24] = mask;
        rec
    }

    /// A `DXBC` container with `OSG1`/`PSG1` and a mesh `PSV0`.
    fn container(osg: &[[u8; 32]], psg: &[[u8; 32]], vertices: u16, primitives: u16) -> Vec<u8> {
        let part = |rows: &[[u8; 32]]| {
            let mut out = Vec::new();
            out.extend((rows.len() as u32).to_le_bytes());
            out.extend(8u32.to_le_bytes());
            for r in rows {
                out.extend_from_slice(r);
            }
            out
        };
        let mut psv = Vec::new();
        psv.extend(48u32.to_le_bytes());
        psv.resize(4 + 48, 0);
        psv[4 + 12..4 + 14].copy_from_slice(&vertices.to_le_bytes());
        psv[4 + 14..4 + 16].copy_from_slice(&primitives.to_le_bytes());
        psv[4 + 24] = 13;
        build(&[
            (FourCC::OSG1, &part(osg)),
            (FourCC::PSG1, &part(psg)),
            (FourCC::PSV0, &psv),
        ])
    }

    /// The measured shape: position, 16 full rows and a scalar. Packed it is
    /// 69 components (31,568 B at 118 vertices), but Metal pads the scalar up
    /// to the next 16-byte row (288 B per vertex, 33,984 B).
    fn packed_fits_aligned_does_not() -> Vec<u8> {
        let mut osg = vec![row(POSITION, 0, 0xF)];
        osg.push(row(0, 1, 0x1)); // scalar first
        osg.extend((2..18).map(|r| row(0, r, 0xF)));
        container(&osg, &[], 118, 0)
    }

    #[test]
    fn aligned_layout_counts_padding_the_packed_count_misses() {
        let dxil = packed_fits_aligned_does_not();
        let packed = packed_mesh_bytes(&dxil).unwrap();
        let aligned = aligned_mesh_bytes(&dxil).unwrap().unwrap();
        assert_eq!(packed, 69 * 4 * 118);
        assert!(packed <= METAL_MESH_OUTPUT_BYTES);
        assert_eq!(aligned, 288 * 118);
        assert!(aligned > METAL_MESH_OUTPUT_BYTES);
        assert_eq!(mesh_output_bytes(&dxil).unwrap(), Some(aligned));
        assert!(!fits_mesh_output_budget(&dxil).unwrap());
    }

    fn el(name: &str, kind: u8, row: u32, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: if kind == 3 { 4 } else { 2 },
            row: Some(row),
            col,
            cols,
            usage: if cols >= 8 { u8::MAX } else { (1u8 << cols) - 1 },
            half: false,
        }
    }

    /// A console mask can name more than one Metal row's four channels; the
    /// split gives each chunk its own register row, keeps the first at the
    /// guest row, preserves the declared components and semantics, and leaves
    /// every chunk within one row's four channels.
    #[test]
    fn wide_elements_split_across_four_channel_rows() {
        let mut els = vec![el("SV_Position", 3, 0, 0, 4), el("TEXCOORD", 0, 3, 0, 8)];
        split_wide_rows(&mut els);
        assert_eq!(els.len(), 3);
        assert_eq!((els[1].row, els[1].col, els[1].cols), (Some(3), 0, 4));
        assert_eq!((els[2].row, els[2].col, els[2].cols), (Some(4), 0, 4));
        assert_eq!(
            els[1..].iter().map(|e| u32::from(e.cols)).sum::<u32>(),
            8
        );
        assert!(els.iter().all(|e| u32::from(e.col) + u32::from(e.cols) <= 4));

        // A row that starts mid-row splits at the row boundary first.
        let mut els = vec![el("SV_Position", 3, 0, 0, 4), el("TEXCOORD", 0, 5, 2, 4)];
        split_wide_rows(&mut els);
        assert_eq!((els[1].row, els[1].col, els[1].cols), (Some(5), 2, 2));
        assert_eq!((els[2].row, els[2].col, els[2].cols), (Some(6), 0, 2));
    }

    /// Rows move among the register numbers the signature already uses: a
    /// 12-byte row crossing before two scalar rows removes its alignment
    /// padding, and a non-position system value (here the render-target array
    /// index) travels with it. The position and a colour target stay put:
    /// the position is laid out specially and an `SV_Target` export names its
    /// register directly.
    #[test]
    fn whole_row_order_moves_system_rows() {
        let mut els = vec![el("SV_Position", 3, 0, 0, 4)];
        els.extend((1..=8).map(|r| el("TEXCOORD", 0, r, 0, 4)));
        els.push(el("SV_RenderTargetArrayIndex", 4, 9, 0, 1));
        els.push(el("TEXCOORD", 0, 10, 0, 1));
        els.push(el("TEXCOORD", 0, 11, 0, 3));
        let order = row_order(&els).expect("a better order exists");
        assert_eq!(order.get(&0), None, "the position row is fixed");
        assert_eq!(order.get(&11), Some(&9), "the 12-byte row leads the movables");
        assert_eq!(order.get(&9), Some(&10), "the system value travels with it");
        assert_eq!(order.get(&10), Some(&11));
    }

    /// A colour target's `SV_Target` export names its register directly, so
    /// its row must not move even while the other rows are packed.
    #[test]
    fn whole_row_order_keeps_colour_target_rows() {
        let mut els = vec![
            el("SV_Position", 3, 0, 0, 4),
            el("SV_Target", 16, 1, 0, 4),
        ];
        els.extend((2..=5).map(|r| el("TEXCOORD", 0, r, 0, 4)));
        els.push(el("SV_RenderTargetArrayIndex", 4, 6, 0, 1));
        els.push(el("TEXCOORD", 0, 7, 0, 1));
        els.push(el("TEXCOORD", 0, 8, 0, 3));
        let order = row_order(&els).expect("a better order exists");
        assert_eq!(order.get(&0), None);
        assert_eq!(order.get(&1), None, "a colour target names its register");
        assert_eq!(order.get(&8), Some(&6));
        assert_eq!(order.get(&6), Some(&7));
        assert_eq!(order.get(&7), Some(&8));
    }

    /// When every row has the same width there is nothing to gain, and the
    /// packer reports no change rather than a tie-equivalent permutation.
    #[test]
    fn whole_row_order_is_none_without_an_improvement() {
        let mut els = vec![el("SV_Position", 3, 0, 0, 4)];
        els.extend((1..=8).map(|r| el("TEXCOORD", 0, r, 0, 4)));
        assert_eq!(row_order(&els), None);
    }

    #[test]
    fn none_for_a_non_mesh_or_non_container() {
        assert_eq!(aligned_mesh_bytes(&[]).unwrap(), None);
        let psv = {
            let mut psv = Vec::new();
            psv.extend(48u32.to_le_bytes());
            psv.resize(4 + 48, 0);
            psv[4 + 24] = 0; // pixel, not mesh
            psv
        };
        let dxil = build(&[(FourCC::PSV0, &psv)]);
        assert_eq!(aligned_mesh_bytes(&dxil).unwrap(), None);
        assert_eq!(mesh_output_bytes(&dxil).unwrap(), None);
        assert!(fits_mesh_output_budget(&dxil).unwrap());
    }

    // -----------------------------------------------------------------
    // Linked-pair proof: the three refused size classes, and the cases it
    // must keep refusing.

    /// One console signature row with its name and channel spans.
    #[derive(Clone)]
    struct RowSpec {
        name: &'static str,
        index: u32,
        /// Raw D3D system value (`0` arbitrary, `1` position).
        kind: u32,
        register: u32,
        /// The declared channel mask.
        declared: u8,
        /// For a pixel `ISG1` row, the channels read.
        read: u8,
    }

    /// `OSG1`/`PSG1`/`ISG1` bytes: count, 32-byte rows from offset 8, names.
    fn sig_part(rows: &[RowSpec]) -> Vec<u8> {
        let mut names = Vec::new();
        let mut offsets = Vec::with_capacity(rows.len());
        for r in rows {
            offsets.push((8 + 32 * rows.len() + names.len()) as u32);
            names.extend_from_slice(r.name.as_bytes());
            names.push(0);
        }
        let mut out = Vec::new();
        out.extend((rows.len() as u32).to_le_bytes());
        out.extend(8u32.to_le_bytes());
        for (r, &offset) in rows.iter().zip(&offsets) {
            out.extend(0u32.to_le_bytes()); // stream
            out.extend(offset.to_le_bytes());
            out.extend(r.index.to_le_bytes());
            out.extend(r.kind.to_le_bytes());
            out.extend(3u32.to_le_bytes()); // float32
            out.extend(r.register.to_le_bytes());
            out.extend([r.declared, r.read, 0, 0]);
            out.extend(0u32.to_le_bytes());
        }
        out.extend(names);
        out
    }

    /// A mesh container: `OSG1`/`PSG1` plus a mesh `PSV0`.
    fn mesh_container(osg: &[RowSpec], psg: &[RowSpec], vertices: u16, primitives: u16) -> Vec<u8> {
        let mut psv = Vec::new();
        psv.extend(48u32.to_le_bytes());
        psv.resize(4 + 48, 0);
        psv[4 + 12..4 + 14].copy_from_slice(&vertices.to_le_bytes());
        psv[4 + 14..4 + 16].copy_from_slice(&primitives.to_le_bytes());
        psv[4 + 24] = 13; // mesh
        build(&[
            (FourCC::OSG1, &sig_part(osg)),
            (FourCC::PSG1, &sig_part(psg)),
            (FourCC::PSV0, &psv),
        ])
    }

    /// A pixel container: `ISG1` plus a pixel `PSV0`.
    fn pixel_container(isg: &[RowSpec]) -> Vec<u8> {
        let mut psv = Vec::new();
        psv.extend(48u32.to_le_bytes());
        psv.resize(4 + 48, 0);
        psv[4 + 24] = 0; // pixel
        build(&[(FourCC::ISG1, &sig_part(isg)), (FourCC::PSV0, &psv)])
    }

    /// Position plus `widths`, each width list one register row, `TEXCOORD`
    /// indices assigned in emitted order.
    fn mesh_shape(widths: &[&[u8]], vertices: u16) -> Vec<RowSpec> {
        let mut rows = vec![RowSpec {
            name: "SV_Position",
            index: 0,
            kind: 1,
            register: 0,
            declared: 0xF,
            read: 0,
        }];
        let mut index = 0u32;
        for (register, width) in (1..).zip(widths) {
            let mut col = 0u8;
            for &cols in *width {
                let mask = (((1u32 << cols) - 1) << col) as u8;
                rows.push(RowSpec {
                    name: "TEXCOORD",
                    index,
                    kind: 0,
                    register,
                    declared: mask,
                    read: 0,
                });
                index += 1;
                col += cols;
            }
        }
        rows
    }

    /// The pixel inputs reading `read(row)` non-zero, one `ISG1` row each.
    fn pixel_inputs(mesh: &[RowSpec], read: impl Fn(&RowSpec) -> u8) -> Vec<RowSpec> {
        mesh.iter()
            .filter(|r| r.kind == 0)
            .filter_map(|r| {
                let mask = read(r);
                (mask != 0).then(|| RowSpec {
                    name: "TEXCOORD",
                    index: r.index,
                    kind: 0,
                    register: 0,
                    declared: mask,
                    read: mask,
                })
            })
            .collect()
    }

    /// Every declared row appears exactly once as retained or pruned.
    fn assert_partition(mesh: &[RowSpec], proof: &MeshPackProof) {
        let mut got: Vec<(String, u32)> = proof
            .retained
            .iter()
            .chain(&proof.pruned)
            .map(|r| (r.name.clone(), r.index))
            .collect();
        got.sort();
        let mut want: Vec<(String, u32)> =
            mesh.iter().map(|r| (r.name.to_string(), r.index)).collect();
        want.sort();
        assert_eq!(got, want, "every declared row is accounted for once");
    }

    /// Class 1: 33,280 B (520 B per vertex at 64 vertices). One dead full row
    /// and one over-declared 3-component element. Dead-export pruning and
    /// channel compaction bring it under the limit.
    #[test]
    fn proof_fits_the_33280_class_after_pruning_and_channel_compaction() {
        let mut widths: Vec<&[u8]> = vec![&[4]; 29];
        widths.push(&[1, 3]);
        widths.extend([&[2][..]; 2]);
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let inputs = pixel_inputs(&mesh, |r| match r.index {
            5 => 0,          // a whole full row nobody observes
            30 => 0b0110,    // the 3-component element: read channels 1 and 2
            _ => r.declared, // every other row is read in full
        });
        let pixel = pixel_container(&inputs);
        let proof = mesh_pack_proof(&container, Some(&pixel))
            .expect("the mesh container parses")
            .expect("a mesh output declaration");

        assert_eq!(proof.declared_aligned, 33_280);
        assert_eq!(proof.declared_packed, 128 * 4 * 64);
        assert!(proof.declared_aligned > METAL_MESH_OUTPUT_BYTES);
        assert!(proof.linkage_proven);
        assert!(proof.fits, "{proof:?}");
        assert!(proof.aligned_after <= METAL_MESH_OUTPUT_BYTES);
        assert!(proof.packed_after <= METAL_MESH_OUTPUT_BYTES);
        assert_eq!(proof.pruned.len(), 1);
        assert_eq!(proof.pruned[0].index, 5);
        assert_eq!(proof.pruned[0].reason, Some(MeshPruneReason::UnreadByPixel));
        let compacted = proof
            .retained
            .iter()
            .find(|r| r.index == 30)
            .expect("the 3-component element is retained");
        assert_eq!((compacted.col, compacted.cols), (1, 2));
        assert!(compacted.compacted());
        assert_eq!(compacted.read, 0b0110);
        // The position and every retained semantic keep their identity.
        assert!(proof.retained.iter().any(|r| r.kind == 3 && r.index == 0));
        assert_partition(&mesh, &proof);
        for row in &proof.rows_after {
            assert!(u32::from(row.1) + u32::from(row.2) <= 4, "{row:?}");
        }
    }

    /// Class 2: 33,792 B (528 B per vertex: position plus 32 full rows at 64
    /// vertices). Dropping the one arbitrary row the pixel does not read lands
    /// on exactly 32,768 B.
    #[test]
    fn proof_fits_the_33792_class_by_dropping_one_dead_row() {
        let widths: Vec<&[u8]> = vec![&[4]; 32];
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let inputs = pixel_inputs(&mesh, |r| if r.index == 7 { 0 } else { r.declared });
        let pixel = pixel_container(&inputs);
        let proof = mesh_pack_proof(&container, Some(&pixel))
            .unwrap()
            .expect("a mesh output declaration");

        assert_eq!(proof.declared_aligned, 33_792);
        assert_eq!(proof.declared_packed, 132 * 4 * 64);
        assert!(proof.linkage_proven);
        assert!(proof.fits, "{proof:?}");
        assert_eq!(proof.aligned_after, 32_768);
        assert_eq!(proof.packed_after, 32_768);
        assert_eq!(proof.pruned.len(), 1);
        assert_eq!(proof.pruned[0].index, 7);
        assert_partition(&mesh, &proof);
    }

    /// Class 3: 35,328 B (552 B per vertex: position, 33 full rows and a
    /// 2-component row at 64 vertices). Three dead rows go; the result is
    /// exactly the budget, not a truncation.
    #[test]
    fn proof_fits_the_35328_class_by_dropping_its_dead_rows() {
        let mut widths: Vec<&[u8]> = vec![&[4]; 33];
        widths.push(&[2]);
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let inputs = pixel_inputs(&mesh, |r| match r.index {
            3 | 10 | 33 => 0,
            _ => r.declared,
        });
        let pixel = pixel_container(&inputs);
        let proof = mesh_pack_proof(&container, Some(&pixel))
            .unwrap()
            .expect("a mesh output declaration");

        assert_eq!(proof.declared_aligned, 35_328);
        assert_eq!(proof.declared_packed, 138 * 4 * 64);
        assert!(proof.linkage_proven);
        assert!(proof.fits, "{proof:?}");
        assert!(proof.aligned_after <= METAL_MESH_OUTPUT_BYTES);
        assert_eq!(proof.pruned.len(), 3);
        assert_eq!(proof.aligned_after, 32_768);
        assert_partition(&mesh, &proof);
    }

    /// A row the pixel reads is never pruned, even if it is the only reader of
    /// one channel: it is retained and narrowed to that channel.
    #[test]
    fn a_row_the_pixel_reads_is_never_pruned() {
        let widths: Vec<&[u8]> = vec![&[4]; 16];
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let inputs = pixel_inputs(&mesh, |r| if r.index == 3 { 0b0100 } else { 0 });
        let pixel = pixel_container(&inputs);
        let proof = mesh_pack_proof(&container, Some(&pixel))
            .unwrap()
            .expect("a mesh output declaration");

        assert!(proof.pruned.iter().all(|r| r.index != 3));
        let kept = proof
            .retained
            .iter()
            .find(|r| r.index == 3)
            .expect("a read row is retained");
        assert_eq!((kept.col, kept.cols), (2, 1));
        assert_eq!(kept.read, 0b0100);
        assert!(proof.fits);
    }

    /// A pair that still does not fit is reported as such: every row stays
    /// (nothing is truncated) and `fits` is false.
    #[test]
    fn a_pair_that_cannot_fit_is_reported_not_truncated() {
        let mut widths: Vec<&[u8]> = vec![&[4]; 33];
        widths.push(&[2]);
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let inputs = pixel_inputs(&mesh, |r| r.declared);
        let pixel = pixel_container(&inputs);
        let proof = mesh_pack_proof(&container, Some(&pixel))
            .unwrap()
            .expect("a mesh output declaration");

        assert!(proof.pruned.is_empty());
        assert_eq!(proof.retained.len(), mesh.len());
        assert_eq!(proof.aligned_after, proof.declared_aligned);
        assert_eq!(proof.aligned_after, 35_328);
        assert!(!proof.fits);
        assert_partition(&mesh, &proof);
    }

    /// An unidentifiable pixel stage proves no linkage: every row is kept and
    /// the pair is refused rather than pruned on a guess.
    #[test]
    fn unprovable_pixel_linkage_keeps_every_row_and_refuses() {
        let widths: Vec<&[u8]> = vec![&[4]; 32];
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let proof = mesh_pack_proof(&container, Some(b"not a container"))
            .unwrap()
            .expect("a mesh output declaration");

        assert!(!proof.linkage_proven);
        assert!(!proof.fits);
        assert!(proof.pruned.is_empty());
        assert_eq!(proof.retained.len(), mesh.len());
    }

    /// With no pixel stage (a depth-only draw), no arbitrary output is
    /// observable: all arbitrary rows are pruned, only the system values stay.
    #[test]
    fn a_depth_only_pair_prunes_arbitrary_outputs_and_keeps_system_values() {
        let widths: Vec<&[u8]> = vec![&[4]; 32];
        let mesh = mesh_shape(&widths, 64);
        let container = mesh_container(&mesh, &[], 64, 0);
        let proof = mesh_pack_proof(&container, None)
            .unwrap()
            .expect("a mesh output declaration");

        assert!(proof.linkage_proven);
        assert!(proof.fits);
        assert_eq!(proof.retained.len(), 1);
        assert_eq!(proof.retained[0].kind, 3);
        assert_eq!(proof.pruned.len(), mesh.len() - 1);
        assert!(
            proof
                .pruned
                .iter()
                .all(|r| r.reason == Some(MeshPruneReason::UnreadByPixel))
        );
    }
}
