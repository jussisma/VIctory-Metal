//! Mesh shader output signatures, counts and lane indices.

use super::{Attr, HashMap, Lower, SigElement, V, Why, pred};
use crate::emit::shader::Mesh;

/// Semantic key `(UPPERCASE_NAME, index)` -> 4-bit absolute channel mask of
/// arbitrary (`kind == 0`) inputs read by the paired pixel shader.
pub(crate) type KeptOutputs = HashMap<(String, u32), u8>;

/// Keeps only arbitrary (`kind == 0`) signature elements whose semantic has at
/// least one channel read in `keep`, trimming unread leading and trailing
/// channels. System values (`kind != 0`) stay untouched. Element order is
/// preserved: callers that need Metal's aligned layout call [`pack_rows`]
/// explicitly (mesh outputs only).
pub(super) fn prune_signature(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    elements.retain_mut(|e| {
        if e.kind != 0 {
            return true;
        }
        if e.row.is_none() {
            return false;
        }
        let key = (e.name.to_ascii_uppercase(), e.index);
        let used = keep.get(&key).copied().unwrap_or(0);
        let rel = (used >> e.col) & ((1u8 << e.cols) - 1);
        if rel == 0 {
            return false;
        }
        let first = rel.trailing_zeros() as u8;
        let last = 7 - rel.leading_zeros() as u8;
        e.col += first;
        e.cols = last - first + 1;
        true
    });
}

/// Reorders the elements **within each register row** (and splits an element
/// that declares more channels than one 4-channel Metal row can carry) to the
/// order Metal's aligned output layout counts smallest. See
/// [`crate::emit::mesh_pack::pack_within_rows`].
pub(super) fn pack_rows(elements: &mut Vec<SigElement>) {
    crate::emit::mesh_pack::split_wide_rows(elements);
    crate::emit::mesh_pack::pack_within_rows(elements);
}

/// Whole-signature row ordering, guest register row -> emitted row, chosen to
/// shrink Metal's aligned output layout. See
/// [`crate::emit::mesh_pack::row_order`].
pub(super) fn row_order(elements: &[SigElement]) -> Option<HashMap<u32, u32>> {
    crate::emit::mesh_pack::row_order(elements)
}

impl Lower<'_> {
    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    pub(super) fn mesh_prologue(&mut self, mesh: Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    pub(super) fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }
}

#[cfg(test)]
mod probe {
    use super::*;

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
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    fn aligned(els: &[SigElement], vertices: u32) -> u32 {
        let distances: u32 = els
            .iter()
            .filter(|e| matches!(e.kind, 6 | 7))
            .map(|e| 4 * u32::from(e.cols))
            .sum();
        let mut end = distances + 16;
        for e in els.iter().filter(|e| e.kind != 3) {
            let size = 4 * u32::from(e.cols);
            end = end.next_multiple_of(size) + size;
        }
        end * vertices
    }

    fn packed(els: &[SigElement], vertices: u32) -> u32 {
        4 * els.iter().map(|e| u32::from(e.cols)).sum::<u32>() * vertices
    }

    fn signature(rows: &[&[u8]]) -> Vec<SigElement> {
        let mut out = vec![el("SV_Position", 3, 0, 0, 4)];
        for (r, widths) in (1..).zip(rows) {
            let mut col = 0;
            for &cols in *widths {
                out.push(el("TEXCOORD", 0, r, col, cols));
                col += cols;
            }
        }
        out
    }

    fn show(label: &str, mut elements: Vec<SigElement>, vertices: u32) {
        let before = aligned(&elements, vertices);
        let packed_before = packed(&elements, vertices);
        pack_rows(&mut elements);
        let after_rows = aligned(&elements, vertices);
        let map = row_order(&elements);
        let mut emitted = elements.clone();
        if let Some(order) = &map {
            for e in &mut emitted {
                if let Some(m) = e.row.and_then(|r| order.get(&r).copied()) {
                    e.row = Some(m);
                }
            }
            emitted.sort_by_key(|e| e.row.unwrap_or(u32::MAX));
        }
        eprintln!(
            "{label}: packed {packed_before} aligned {before} | rows {after_rows} | ordered {} aligned {} packed {}",
            map.is_some(),
            aligned(&emitted, vertices),
            packed(&emitted, vertices)
        );
    }

    #[test]
    fn probe_shapes() {
        let mut rows: Vec<&[u8]> = vec![&[4]; 29];
        rows.push(&[1, 3]);
        rows.extend([&[2][..]; 2]);
        show("A 128c/64v", signature(&rows), 64);

        let rows: Vec<&[u8]> = std::iter::once(&[1u8][..])
            .chain(std::iter::repeat_n(&[4u8][..], 16))
            .collect();
        show("B 69c/118v", signature(&rows), 118);

        let rows: Vec<&[u8]> = std::iter::repeat_n(&[4u8][..], 15)
            .chain(std::iter::once(&[1u8][..]))
            .collect();
        show("C 65c/128v", signature(&rows), 128);

        // Distances: position, a 4-wide clip distance, then a 3-wide row, two
        // scalars and 14 full rows (112 vertices: packed under the budget,
        // aligned over it until the rows are ordered).
        let mut d = signature(&vec![&[4u8][..]; 0]);
        let mut clip = el("SV_ClipDistance0", 6, 1, 0, 4);
        clip.interp = 0;
        d.push(clip);
        for (r, widths) in (2..).zip([
            &[3u8][..],
            &[1][..],
            &[1][..],
        ]
        .into_iter()
        .chain(std::iter::repeat_n(&[4u8][..], 14)))
        {
            let mut col = 0;
            for &cols in widths {
                d.push(el("TEXCOORD", 0, r, col, cols));
                col += cols;
            }
        }
        d.sort_by_key(|e| e.row.unwrap());
        show("D distances 69c/112v", d, 112);

        // Mixed interpolation inside one row (float + int 1-component).
        let mut m = vec![el("SV_Position", 3, 0, 0, 4)];
        for r in 1..=16u32 {
            if r == 9 {
                m.push(el("TEXCOORD", 0, r, 0, 2));
                let mut u = el("TEXCOORD", 0, r, 2, 1);
                u.comp = 5;
                u.interp = 1;
                m.push(u);
            } else {
                m.push(el("TEXCOORD", 0, r, 0, 4));
            }
        }
        show("E mixed interp 65c/64v", m, 64);

        // Five elements in one register row (overlapping masks).
        let mut f = vec![el("SV_Position", 3, 0, 0, 4)];
        for r in 1..=15u32 {
            f.push(el("TEXCOORD", 0, r, 0, 4));
        }
        f.push(el("TEXCOORD", 0, 16, 0, 1));
        f.push(el("TEXCOORD", 0, 16, 0, 1));
        f.push(el("TEXCOORD", 0, 16, 1, 1));
        f.push(el("TEXCOORD", 0, 16, 2, 1));
        f.push(el("TEXCOORD", 0, 16, 3, 1));
        show("F five-in-row 65c/64v", f, 64);
    }
}
