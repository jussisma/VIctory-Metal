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

use crate::dxbc::{FourCC, u32_at};
use crate::{Container, ShaderError};

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
}
