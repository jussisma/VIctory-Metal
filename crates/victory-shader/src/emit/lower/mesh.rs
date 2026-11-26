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
    if !elements.windows(2).all(|w| {
        w[0].row.unwrap_or(u32::MAX) <= w[1].row.unwrap_or(u32::MAX)
    }) {
        return;
    }
    // Where the host starts: clip/cull distances, then the position.
    let distances: u32 = elements
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7))
        .map(element_bytes)
        .sum();
    let has_position = elements.iter().any(|e| e.kind == 3);
    let mut end = distances + if has_position { 16 } else { 0 };
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == row {
            j += 1;
        }
        // A register row holds at most four one-component elements.
        if row.is_some()
            && (2..=4).contains(&(j - i))
            && elements[i..j].iter().all(|e| e.kind == 0)
        {
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
        for e in &elements[i..j] {
            if e.kind == 3 {
                continue;
            }
            let size = element_bytes(e);
            end = end.next_multiple_of(size) + size;
        }
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
/// fixed position row and every other system value stay where they are.
pub(super) fn row_order(elements: &[SigElement]) -> Option<HashMap<u32, u32>> {
    // Only the register-sorted vectors the container writes.
    if !elements.windows(2).all(|w| {
        w[0].row.unwrap_or(u32::MAX) <= w[1].row.unwrap_or(u32::MAX)
    }) {
        return None;
    }
    // Proven safe only beside the position: no 16-bit values, no other
    // system values (so no distance rows), and one mode per moved row.
    if elements.iter().any(|e| e.half || !matches!(e.kind, 0 | 3)) {
        return None;
    }
    let mut blocks: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut slots: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].row?;
        let mut j = i + 1;
        while j < elements.len() && elements[j].row == Some(row) {
            j += 1;
        }
        let group = &elements[i..j];
        if group.iter().any(|e| e.kind == 3) {
            // The position is fixed and one element; a mixed row is unproven.
            if group.len() != 1 || group[0].kind != 3 {
                return None;
            }
        } else {
            let interp = group[0].interp;
            if group.iter().any(|e| e.interp != interp) {
                return None;
            }
            slots.push(row);
            blocks.push((row, group.iter().map(|e| 4 * u32::from(e.cols)).collect()));
        }
        i = j;
    }
    if blocks.len() < 2 {
        return None;
    }
    // The host's fold: the fixed 16-byte position, then every remaining
    // element in signature order, each rounded up to its own size.
    let end = |order: &[usize]| -> u32 {
        order.iter().fold(16, |end, &b| {
            blocks[b]
                .1
                .iter()
                .fold(end, |end, &size| end.next_multiple_of(size) + size)
        })
    };
    let n = blocks.len();
    let identity: Vec<usize> = (0..n).collect();
    let start = end(&identity);
    // Largest registers first: one that ends on a 16-byte boundary leaves
    // every narrower row after it unpadded.
    let mut seeds = vec![identity.clone()];
    let mut by_size: Vec<usize> = (0..n).collect();
    by_size.sort_by_key(|&b| {
        let sizes = &blocks[b].1;
        (
            std::cmp::Reverse(sizes.iter().sum::<u32>()),
            std::cmp::Reverse(sizes.iter().copied().max().unwrap_or(0)),
            b,
        )
    });
    seeds.push(by_size);
    let mut by_lead: Vec<usize> = (0..n).collect();
    by_lead.sort_by_key(|&b| {
        let sizes = &blocks[b].1;
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
            .map(|(at, &b)| (blocks[b].0, slots[at]))
            .collect(),
    )
}

/// The order of `elements` that leaves the smallest aligned end after
/// `start`. Rows carry at most four elements, so trying every order is
/// bounded; on ties the declared order is kept.
fn least_padded(elements: &[SigElement], start: u32) -> Vec<usize> {
    fn walk(
        sizes: &[u32],
        start: u32,
        order: &mut Vec<usize>,
        used: &mut [bool],
        best: &mut (u32, Vec<usize>),
    ) {
        if order.len() == sizes.len() {
            let mut end = start;
            for &k in order.iter() {
                let size = sizes[k];
                end = end.next_multiple_of(size) + size;
            }
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
    let mut best = (u32::MAX, (0..sizes.len()).collect());
    walk(
        &sizes,
        start,
        &mut Vec::with_capacity(sizes.len()),
        &mut vec![false; sizes.len()],
        &mut best,
    );
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
