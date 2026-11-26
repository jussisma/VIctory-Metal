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

/// One element's size as Metal lays a mesh output out: 4 bytes per declared
/// component (its mask's components).
fn element_bytes(e: &SigElement) -> u32 {
    4 * u32::from(e.cols)
}

/// Whether `elements` come in the order the container writes them (by
/// register, unallocated last). Both packers only touch such vectors.
fn register_sorted(elements: &[SigElement]) -> bool {
    elements
        .windows(2)
        .all(|w| w[0].row.unwrap_or(u32::MAX) <= w[1].row.unwrap_or(u32::MAX))
}

/// Where the host starts a mesh vertex's output layout: every clip/cull
/// distance component as a 4-byte scalar, then the 16-byte position. The
/// distances are counted again in signature order by [`fold`], as the host
/// counts them.
fn head_bytes(elements: &[SigElement]) -> u32 {
    let distances: u32 = elements
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7))
        .map(|e| u32::from(e.cols))
        .sum();
    4 * distances + 16
}

/// Metal's aligned end for `sizes` laid out in order from `end`: every
/// element is rounded up to its own size (4, 8, 12 or 16 bytes). This is the
/// fold the host applies to a mesh output signature.
fn fold(sizes: &[u32], mut end: u32) -> u32 {
    for &size in sizes {
        end = end.next_multiple_of(size) + size;
    }
    end
}

/// Reorders the arbitrary (`kind == 0`) elements **within each register row**
/// to the order Metal's aligned output layout counts smallest. The host lays
/// mesh outputs out as the clip/cull distances, the position (16 bytes), then
/// every other element in signature order, each rounded up to its own size
/// (4, 8, 12 or 16 bytes). Trimming cannot remove the padding between
/// elements; their order can.
///
/// Elements never move across a row, so a guest row still names the same
/// elements for a mesh export, the vector stays register-sorted, and each
/// element's index therefore stays the signature id its emitted store names.
/// Only the order inside one register changes: semantics, indexes, masks,
/// component types, interpolation and system values are untouched.
pub(super) fn pack_rows(elements: &mut [SigElement]) {
    // Only act on the register-sorted vectors the container writes; anything
    // else keeps its declared order.
    if !register_sorted(elements) {
        return;
    }
    // Where the host starts: clip/cull distances, then the position.
    let mut end = head_bytes(elements);
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == row {
            j += 1;
        }
        // A row's own elements are ordered when that ends smaller. Rows of
        // one element have nothing to order, and rows holding a system value
        // are left as declared. Rows may hold more elements than four
        // components when masks overlap; `least_padded` bounds its search.
        if row.is_some() && j - i >= 2 && elements[i..j].iter().all(|e| e.kind == 0) {
            let order = least_padded(&elements[i..j], end);
            if order.iter().enumerate().any(|(at, &k)| at != k) {
                let before: Vec<SigElement> = elements[i..j].to_vec();
                for (at, &k) in order.iter().enumerate() {
                    elements[i + at] = before[k].clone();
                }
            }
        }
        // Mirror the host: the position is part of the fixed start, distances
        // are counted again in signature order as it counts them.
        let sizes: Vec<u32> = elements[i..j]
            .iter()
            .filter(|e| e.kind != 3)
            .map(element_bytes)
            .collect();
        end = fold(&sizes, end);
        i = j;
    }
}

/// Whole-register row permutation: guest row -> row emitted in the linked
/// signature, chosen to shrink Metal's aligned output layout. `None` when
/// nothing moves or the signature is outside what is proven safe.
///
/// Rows move as whole registers (`pack_rows` has already ordered each row's
/// elements), so every element keeps its semantic, index, channel mask,
/// component type and interpolation mode, and a guest row still names the
/// same elements. Only the register numbers the host lays the signature out
/// by change, among the arbitrary row numbers the signature already has: the
/// position row and every clip/cull distance row stay where they are.
pub(super) fn row_order(elements: &[SigElement]) -> Option<HashMap<u32, u32>> {
    // Only the register-sorted vectors the container writes.
    if !register_sorted(elements) {
        return None;
    }
    // Proven safe only beside the position: no 16-bit values (their halves
    // pair by guest row), and no system values besides the position and the
    // clip/cull distances, which stay fixed.
    if elements
        .iter()
        .any(|e| e.half || !matches!(e.kind, 0 | 3 | 6 | 7))
    {
        return None;
    }
    if elements.iter().filter(|e| e.kind == 3).count() > 1 {
        return None;
    }
    // The row groups, by the size of each element: the fixed ones (a
    // distance element's row) and the ones free to move.
    let mut fixed: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut movable: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row?;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == Some(row) {
            j += 1;
        }
        let group = &elements[i..j];
        let sizes: Vec<u32> = group.iter().map(element_bytes).collect();
        if group.iter().any(|e| e.kind == 3) {
            // The position is fixed and one element; a mixed row is unproven.
            if group.len() != 1 || group[0].kind != 3 {
                return None;
            }
        } else if group.iter().any(|e| matches!(e.kind, 6 | 7)) {
            fixed.push((row, sizes));
        } else {
            movable.push((row, sizes));
        }
        i = j;
    }
    if movable.len() < 2 {
        return None;
    }
    // The rows a movable block may take, and every slot either kind fills,
    // in the signature order the host lays them out in.
    let mut movable_slots: Vec<u32> = movable.iter().map(|b| b.0).collect();
    movable_slots.sort_unstable();
    let mut slots: Vec<u32> = movable_slots.iter().copied().chain(fixed.iter().map(|f| f.0)).collect();
    slots.sort_unstable();
    let fixed_at: HashMap<u32, &[u32]> =
        fixed.iter().map(|(row, sizes)| (*row, sizes.as_slice())).collect();
    let head = head_bytes(elements);
    // The host's fold: the fixed start, then every remaining element in
    // signature order, each rounded up to its own size.
    let end = |order: &[usize]| -> u32 {
        let mut assigned: HashMap<u32, &[u32]> = HashMap::new();
        for (at, &block) in order.iter().enumerate() {
            assigned.insert(movable_slots[at], movable[block].1.as_slice());
        }
        let mut end = head;
        for slot in &slots {
            // Every slot is a fixed row or one of the movable ones.
            if let Some(&sizes) = assigned.get(slot).or_else(|| fixed_at.get(slot)) {
                end = fold(sizes, end);
            }
        }
        end
    };
    let n = movable.len();
    let identity: Vec<usize> = (0..n).collect();
    let start = end(&identity);
    // Largest registers first: one that ends on a 16-byte boundary leaves
    // every narrower row after it unpadded.
    let mut seeds = vec![identity.clone()];
    let mut by_size: Vec<usize> = (0..n).collect();
    by_size.sort_by_key(|&b| {
        let sizes = &movable[b].1;
        (
            std::cmp::Reverse(sizes.iter().sum::<u32>()),
            std::cmp::Reverse(sizes.iter().copied().max().unwrap_or(0)),
            b,
        )
    });
    seeds.push(by_size);
    let mut by_lead: Vec<usize> = (0..n).collect();
    by_lead.sort_by_key(|&b| {
        let sizes = &movable[b].1;
        (
            std::cmp::Reverse(sizes.first().copied().unwrap_or(0)),
            std::cmp::Reverse(sizes.iter().sum::<u32>()),
            b,
        )
    });
    seeds.push(by_lead);
    let mut best = start;
    let mut best_order = identity;
    for mut order in seeds {
        let mut cost = end(&order);
        // Bounded best-improvement local search over swaps and single-row
        // moves; ties keep the earlier order, so the result is deterministic.
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
                Some(order_) => order = order_,
                None => break,
            }
        }
        if cost < best {
            best = cost;
            best_order = order;
        }
    }
    if best >= start {
        return None;
    }
    Some(
        best_order
            .iter()
            .enumerate()
            .map(|(at, &b)| (movable[b].0, movable_slots[at]))
            .collect(),
    )
}

/// The order of `elements` that leaves the smallest aligned end after
/// `start`. A register row's elements usually number at most four, which is
/// tried exhaustively (so ties keep the declared order); rows whose masks
/// overlap can declare more, and those try a bounded set of size orders
/// instead.
fn least_padded(elements: &[SigElement], start: u32) -> Vec<usize> {
    fn walk(
        sizes: &[u32],
        start: u32,
        order: &mut Vec<usize>,
        used: &mut [bool],
        best: &mut (u32, Vec<usize>),
    ) {
        if order.len() == sizes.len() {
            let end = fold(
                &order.iter().map(|&k| sizes[k]).collect::<Vec<u32>>(),
                start,
            );
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
    let sizes: Vec<u32> = elements.iter().map(element_bytes).collect();
    let identity: Vec<usize> = (0..sizes.len()).collect();
    let mut best = (u32::MAX, Vec::new());
    if sizes.len() <= 6 {
        // 6! is 720 orders, and a row cannot declare more one-component
        // elements than it has dwords unless its masks overlap.
        walk(
            &sizes,
            start,
            &mut Vec::with_capacity(sizes.len()),
            &mut vec![false; sizes.len()],
            &mut best,
        );
        return best.1;
    }
    // More elements than a row can hold distinct components: try the
    // declared order, widest first and narrowest first, keeping the first of
    // equal ends (the declared order).
    let mut widest = identity.clone();
    widest.sort_by_key(|&k| std::cmp::Reverse(sizes[k]));
    let mut narrowest = identity.clone();
    narrowest.sort_by_key(|&k| sizes[k]);
    for order in [identity, widest, narrowest] {
        let end = fold(&order.iter().map(|&k| sizes[k]).collect::<Vec<u32>>(), start);
        if end < best.0 {
            best = (end, order);
        }
    }
    best.1
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
