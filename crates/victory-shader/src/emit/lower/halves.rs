//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact. Reads into the other half pass
//! between the two host waves through group-shared memory, between group
//! barriers ([`Lower::exchange_halves`]); barriers need the whole group, so
//! only in a block every lane of the group runs together, the same number
//! of times ([`in_step`]: once, or in a loop on a count every host wave
//! computes alike). Elsewhere they are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), and `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code),
//!   `v_writelane_b32` (the lane named takes the scalar) and the row a DPP
//!   row mask names (rows 2 and 3 are the upper half's) are decided by the
//!   console lane ([`Lower::console_lane`]);
//! * lane masks read as data (ballots) hold the other half's lanes: unless
//!   the whole group fits one host wave, the halves pass each other their
//!   ballots ([`Lower::ballot_across_halves`]), for a mask written in such a
//!   block; elsewhere refused ([`MASK_ACROSS`]);
//! * `v_readlane_b32` of a lane that may be in the other half (SGPR spills
//!   read back are not lane reads): each half passes the value of the lane
//!   with that index in it, and the console lane's half picks
//!   ([`Lower::read_lane_across`]);
//! * whole-wave reductions and prefix sums: each half reduces its own lanes,
//!   and they pass each other the result; a prefix in the upper half adds
//!   the lower half's total ([`Lower::reduce_across`]);
//! * `v_readfirstlane_b32` reads the host wave's first lane, which in the
//!   upper half is console lane 32 or later: kept as it is for a value the
//!   same in every lane of the console wave ([`lane_uniform`], what the
//!   compiler reads first lanes for) and in a group of 32 threads or fewer
//!   (one host wave holds the whole console wave); otherwise the lower
//!   half's first lane ([`Lower::first_lane_across`]); refused where no
//!   exchange can go ([`FIRST_ACROSS`]).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! LDS: the console compiler leaves out barriers between the lanes of one
//! wave, and so between all of a group's threads only when the group is one
//! wave. A group of 33 to 64 threads with LDS (two host waves sharing what
//! the console's one wave ordered by itself) is refused; groups of several
//! console waves order their LDS with barriers, and those of 32 threads or
//! fewer are one host wave.
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name. Which values are the same in every lane of the console
//! wave ([`lane_uniform`]): justrealyme's analysis (chippy01/VIctory#1),
//! ported from fxceless/victory-fork (#497 there).

use super::*;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Why a program in halves cannot run: LDS a group of one console wave
    /// shares without barriers, split across two host waves.
    pub(super) fn halves_lds(group: u32, lds_dwords: u32) -> Option<Why> {
        (lds_dwords > 0 && group > HALF && group <= 64)
            .then_some("LDS one console wave64 shares without barriers, on a host of 32-lane waves")
    }

    /// The flat thread id, relabelled so that each host wave holds a run of
    /// [`Shader::wave_size`]-aligned ids from its first lane (see
    /// [`Lower::thread_id`]); made once, in the entry block.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Whether the group fits one host wave: a console wave64 of 32 threads
    /// or fewer has no lane in the upper half.
    pub(super) fn one_host_wave(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() <= HALF)
    }

    /// Whether every console wave of the group is a full 64 lanes (the group
    /// is a positive multiple of 64). Only then is the live mask of every
    /// wave all 64 bits, which a wave-uniform `true` mask read as data is.
    pub(super) fn full_console_waves(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() % (2 * HALF) == 0)
    }

    /// Where the halves pass each other a lane mask written in IR block
    /// `def` (none: a program input, in the entry): there, if every lane of
    /// the group runs it together ([`in_step`]); in the block being
    /// lowered, where the reading instruction starts ([`Lower::level`],
    /// before any guard of its own).
    pub(super) fn across_at(&self, def: Option<usize>) -> Option<usize> {
        if !self.step.get(def.unwrap_or(0)).copied().unwrap_or(false) {
            return None;
        }
        match def {
            None => Some(self.s.entry),
            Some(b) if b == self.cur_ir => Some(self.level),
            Some(b) => self.tail[b],
        }
    }

    /// Whether the halves can pass each other values where the instruction
    /// being lowered is: in halves, in a group of more than one host wave, in
    /// a block every lane of the group runs together ([`in_step`]), before
    /// anything of the instruction's own split it ([`Lower::level`]).
    pub(super) fn across_here(&self) -> bool {
        self.halves
            && !self.one_host_wave()
            && self.cur == self.level
            && self.step.get(self.cur_ir).copied().unwrap_or(false)
    }

    /// The half of the console wave this lane is in: 0 or 1.
    fn half(&mut self, at: usize) -> V {
        let f = self.relabelled_flat();
        let (five, one) = (self.s.m.i32(5), self.s.m.i32(1));
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        self.s.m.bin(at, Bin::And, slot, one)
    }

    /// Both dwords of the console wave's ballot of `bit`, in halves: each
    /// host wave's own (its half's lanes), passed between them
    /// ([`Lower::exchange_halves`]). A console wave with no upper half has a
    /// high dword of 0. Every lane of the group must run `at` together
    /// ([`Lower::across_at`]).
    pub(super) fn ballot_across_halves(&mut self, bit: V, at: usize) -> Result<(V, V), Why> {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.s.m.extract(at, r, 0);
        let zero = self.s.m.i32(0);
        self.exchange_halves(own, zero, at)
    }

    /// What each half of this lane's console wave gave as `own` (a dword
    /// the same in every lane of a host wave): (lower half's, upper half's).
    /// Each host wave stores its own in a group-shared dword of its own, and
    /// reads its console wave's two after a group barrier; a barrier before
    /// the store keeps it from overwriting what the other half has not read
    /// of an earlier exchange. A console wave with no upper half (a group not
    /// a multiple of 64) reads `missing` for it. Every lane of the group must
    /// run `at` together ([`in_step`]), in halves.
    pub(super) fn exchange_halves(&mut self, own: V, missing: V, at: usize) -> Result<(V, V), Why> {
        #[cfg(test)]
        {
            self.half_exchanges += 1;
        }
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        // Two dwords per console wave, so the high dword of the last one is
        // inside even where the group has no upper half.
        let dwords = group.div_ceil(2 * HALF) * 2;
        if self.s.lds_dwords + dwords > GROUP_SHARED_DWORDS {
            return Err("no group-shared memory left to pass values between 32-lane halves");
        }
        let i32t = self.s.t.i32;
        let f = self.relabelled_flat();
        let words = match self.half_words {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(dwords, i32t);
                let g = self.s.m.global("half_words", arr, 3);
                self.half_words = Some(g);
                g
            }
        };
        // Host wave `2w + h` (its flat ids `32 (2w + h)` on) holds console
        // wave `w`'s half `h`.
        let five = self.s.m.i32(5);
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        let not_one = self.s.m.i32(!1);
        let low = self.s.m.bin(at, Bin::And, slot, not_one);
        let one = self.s.m.i32(1);
        let high = self.s.m.bin(at, Bin::Or, slot, one);
        let sync = |l: &mut Self| {
            // Group sync with a group-shared memory fence.
            let mode = l.s.m.i32(1 | 8);
            let void = l.s.t.void;
            l.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        };
        sync(self);
        let mine = self.s.m.element(at, words, slot);
        self.s.m.store(at, mine, own);
        sync(self);
        let lo = self.s.m.element(at, words, low);
        let lo = self.s.m.load(at, lo);
        let hi = self.s.m.element(at, words, high);
        let mut hi = self.s.m.load(at, hi);
        if !group.is_multiple_of(2 * HALF) {
            let first = self.s.m.bin(at, Bin::Shl, high, five);
            let n = self.s.m.i32(group);
            let there = self.s.m.cmp(at, pred::IULT, first, n);
            hi = self.s.m.select(at, there, hi, missing);
        }
        Ok((lo, hi))
    }

    /// `v_readlane_b32` of `x` at console lane `lane` (the low 6 bits), in
    /// halves ([`Lower::across_here`]): each half reads the lane with that
    /// index in it, and the lane's own half gives the value.
    pub(super) fn read_lane_across(&mut self, x: V, lane: V) -> Result<V, Why> {
        let c = self.cur;
        let (k31, k32, zero) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        let index = self.s.m.bin(c, Bin::And, lane, k31);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.read_lane(x, index);
        let (lo, hi) = self.exchange_halves(own, zero, c)?;
        let upper = self.s.m.bin(c, Bin::And, lane, k32);
        let upper = self.s.m.cmp(c, pred::NE, upper, zero);
        Ok(self.s.m.select(c, upper, hi, lo))
    }

    /// `v_readfirstlane_b32` of `x`, in halves ([`Lower::across_here`]): the
    /// lower half's first lane. Every lane of the group runs the block, so
    /// the console's first lane is in the lower half.
    pub(super) fn first_lane_across(&mut self, x: V) -> Result<V, Why> {
        let c = self.cur;
        let (i32t, zero) = (self.s.t.i32, self.s.m.i32(0));
        self.s.flags |= flags::WAVE_OPS;
        let own = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        Ok(self.exchange_halves(own, zero, c)?.0)
    }

    /// A whole-wave reduction or prefix of `op`, in halves
    /// ([`Lower::across_here`]): `own` is this host wave's (its half's
    /// lanes), `total` its half's total (a prefix's; the reduction itself
    /// otherwise), both as the operation's type. A reduction takes `op` of
    /// both halves' totals; a prefix in the upper half adds the lower half's
    /// total. A console wave with no upper half reads `ident` for it.
    pub(super) fn reduce_across(
        &mut self,
        op: super::super::wave::Op,
        own: V,
        total: V,
        prefix: bool,
        ident: u32,
    ) -> Result<V, Why> {
        use super::super::wave::Op;
        let c = self.cur;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let bits = if float { self.as_int(total) } else { total };
        let missing = self.s.m.i32(ident);
        let (lo, hi) = self.exchange_halves(bits, missing, c)?;
        let (lo, hi) = if float {
            (self.as_float(lo), self.as_float(hi))
        } else {
            (lo, hi)
        };
        if prefix {
            let h = self.half(c);
            let zero = self.s.m.i32(0);
            let upper = self.s.m.cmp(c, pred::NE, h, zero);
            let added = self.s.m.bin(c, Bin::Add, lo, own);
            return Ok(self.s.m.select(c, upper, added, own));
        }
        let unsigned_or_signed = |l: &mut Self, p: u8| {
            let less = l.s.m.cmp(c, p, lo, hi);
            l.s.m.select(c, less, lo, hi)
        };
        Ok(match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinU => unsigned_or_signed(self, pred::IULT),
            Op::MaxU => unsigned_or_signed(self, pred::IUGT),
            Op::MinI => unsigned_or_signed(self, pred::SLT),
            Op::MaxI => unsigned_or_signed(self, pred::SGT),
            // FMin, FMax: as the host's own wave minimum and maximum.
            Op::MinF => self.binary(36, lo, hi),
            Op::MaxF => self.binary(35, lo, hi),
            Op::AddH => return Err("a 16-bit wave reduction over both 32-lane halves"),
        })
    }
}

/// Why a lane mask read as data stops in halves: it is written where the
/// lanes of the group may not all be, as often, so no group barrier can pass
/// it between the halves.
pub(super) const MASK_ACROSS: Why = "a lane mask read as data (wave64 on a 32-lane host)";

/// The group-shared dwords a D3D12 compute program may declare, at most
/// (32 KiB).
const GROUP_SHARED_DWORDS: u32 = 8192;

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, and the value is not known to be the same there.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)";

/// Every branch (by its block) with its region: the blocks reachable from
/// it that do not post-dominate it, which only the lanes taking one side may
/// run (the branch's own block among them where a cycle leads back to it).
/// None with an indirect exit.
pub(super) fn branch_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // Post-dominators, as bitsets (a virtual exit after every end).
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for (w, p) in v.iter_mut().zip(&pdom[t]) {
                    *w &= p;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        if !matches!(e, LaneExit::Branch { .. }) {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            // The branch's own block is in its region if a cycle leads back
            // to it (a loop's latch): only the lanes going round run it.
            if seen[b] || b != d && pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The thread id's x (`v0` on entry) of a wave64 compute program whose
/// console waves are runs of 64 ids along x (a group 64-aligned along x,
/// or one row): `id >> 6` and up is one value per console wave.
fn wave_tid(x: &Parts) -> Option<Value> {
    let StageIo::Cs(cs) = &x.native.io else {
        return None;
    };
    let [tx, ty, tz] = cs.threads;
    if !x.native.wave64 || !(tx % 64 == 0 || ty * tz == 1) {
        return None;
    }
    x.p.inputs
        .iter()
        .find(|(r, _)| *r == Reg::V(0))
        .map(|(_, v)| *v)
}

/// VGPR values that are the same in every lane of a console wave holding
/// them, so that a first-lane read in either half gives the console's
/// value: written lane by lane from scalars, constants and such values, by
/// an instruction every arriving lane runs or that runs under a condition
/// itself the same in every lane (then only over an old value that is such
/// a value too); loaded from LDS, or from a buffer through a scalar
/// descriptor, at such an address; or the thread id's x shifted right past
/// the console wave ([`wave_tid`]). The console compiler keeps wave-uniform
/// data in VGPRs and fetches it with `v_readfirstlane` (a typed buffer's
/// element, say, which only a vector load reads). A scalar is one value,
/// except the dwords a waterfall loop reads, which are each lane's own.
///
/// A phi counts where every predecessor is a block every lane of the wave
/// reaches together: no branch on a lane-dependent condition can have split
/// them (a branch on alike values does not). The analysis runs to a fixed
/// point from nothing, so a loop-carried value that needs itself to be
/// alike never is.
pub(super) fn lane_uniform(x: &Parts) -> HashSet<Value> {
    let (p, st) = (x.p, x.st);
    let wave_tid = wave_tid(x);
    let regions = branch_regions(st);
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let per_lane = |v: Value| x.res.waterfall.reads.contains(&v);
    let mut same: HashSet<Value> = HashSet::new();
    loop {
        let alike_atom = |same: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => same.contains(&v),
            _ => false,
        };
        let alike_bool = |same: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| alike_atom(same, a))
        };
        // Blocks every lane of a console wave reaches together.
        let wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for (d, region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[*d] else {
                        continue;
                    };
                    if alike_bool(&same, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => vec![false; st.exits.len()],
        };
        let mut next: HashSet<Value> = HashSet::new();
        for b in crate::ir::reverse_postorder(&p.blocks) {
            let block = &p.blocks[b];
            let whole = block
                .preds
                .iter()
                .all(|&q| wave_all.get(q).copied().unwrap_or(false));
            for phi in &block.phis {
                if matches!(phi.reg, Reg::V(_))
                    && whole
                    && phi
                        .args
                        .iter()
                        .all(|a| next.contains(a) || same.contains(a))
                {
                    next.insert(phi.value);
                }
            }
            let known = |next: &HashSet<Value>, v: &Value| next.contains(v) || same.contains(v);
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Under a condition: only one every lane holds alike, and
                // over an old value that is alike too.
                if let Some(g) = st.guards[b][i] {
                    if !alike_bool(&same, g) {
                        continue;
                    }
                    let old_alike = inst
                        .writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .all(|w| inst.read(w.0).is_none_or(|old| known(&next, &old)));
                    if !old_alike {
                        continue;
                    }
                }
                // A buffer load at an alike index and offset, through a scalar
                // descriptor, reads one value in every lane.
                if let Fields::Buffer {
                    addr,
                    rsrc,
                    soffset,
                    offen,
                    idxen,
                    ..
                } = &inst.fields
                    && (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                {
                    let vector = |op: &Operand| match *op {
                        Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                            inst.read(Reg::V(reg + k))
                                .is_some_and(|v| known(&next, &v))
                        }),
                        _ => true,
                    };
                    let scalar = |op: &Operand| match *op {
                        Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                            .all(|k| inst.read(Reg::S(code + k)).is_none_or(|v| !per_lane(v))),
                        _ => true,
                    };
                    if (!(*offen || *idxen) || vector(addr)) && scalar(rsrc) && scalar(soffset) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // An LDS read at an alike address reads one value in every lane.
                if let Fields::Ds {
                    addr: Some(Operand::Vector { reg, .. }),
                    dst: Some(_),
                    data,
                    ..
                } = &inst.fields
                    && name.starts_with("ds_read")
                    && data.is_empty()
                {
                    if inst.read(Reg::V(*reg)).is_some_and(|a| known(&next, &a)) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                let ops: Vec<Operand> = match &inst.fields {
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane") =>
                    {
                        v.src.clone()
                    }
                    _ => continue,
                };
                // The thread id's x shifted past the console wave: one value
                // per console wave (the b32 shift uses five bits of its count).
                if let (
                    Some(tid),
                    "v_lshrrev_b32",
                    [
                        Operand::Const(k @ 128..=192),
                        Operand::Vector { reg, .. },
                        ..,
                    ],
                ) = (wave_tid, name, ops.as_slice())
                    && ((k - 128) & 31) >= 6
                    && inst.read(Reg::V(*reg)) == Some(tid)
                {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                    continue;
                }
                let alike = ops.iter().all(|op| match *op {
                    Operand::Vector { reg, .. } => {
                        // every dword the instruction reads from that register on
                        let read: Vec<Value> = inst
                            .reads
                            .iter()
                            .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                            .map(|r| r.1)
                            .collect();
                        !read.is_empty() && read.iter().all(|x| known(&next, x))
                    }
                    // A scalar is one value; a lane mask is not, unless a
                    // compare of alike values made it.
                    Operand::Scalar { code, .. } if code != NULL => inst
                        .read(Reg::S(code))
                        .is_none_or(|x| !is_mask(x) && !per_lane(x) || known(&next, &x)),
                    _ => true,
                });
                if alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) {
            return same;
        }
        same.extend(next);
    }
}

/// Lane masks read as data across the two 32-lane halves
/// ([`Lower::mask_dword`], [`Lower::exchange_halves`]): which shapes are
/// admitted and where, and which still refuse. The hardware proof of the
/// admitted shapes is `victory-metal`'s `tests/wave_lower_gpu.rs`.
#[cfg(test)]
mod lane_mask_tests {
    use super::{Lowered, MASK_ACROSS, Parts, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    pub(super) const S_ENDPGM: u32 = 0xBF81_0000;
    pub(super) const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

    /// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
    pub(super) const PROLOGUE: [u32; 5] = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        S_WAITCNT_LGKM,
    ];

    /// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
    pub(super) const fn store(vdata: u8, offset: u32) -> [u32; 2] {
        [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
    }

    pub(super) const V_LSHLREV_V16_5_V0: u32 = 0x3420_0085; // v_lshlrev_b32 v16, 5, v0
    pub(super) const V_MOV_V1_S12: u32 = 0x7E02_020C; // v_mov_b32 v1, s12
    const V_MOV_V1_VCC_HI: u32 = 0x7E02_026B; // v_mov_b32 v1, vcc_hi
    pub(super) const S_BCNT1_S12_VCC: u32 = 0xBE8C_106A; // s_bcnt1_i32_b64 s12, vcc
    const S_MOV_S20_0: u32 = 0xBE94_0380; // s_mov_b32 s20, 0
    const S_ADD_S20_1: u32 = 0x8014_8114; // s_add_u32 s20, s20, 1
    const S_CMP_LT_S20_S21: u32 = 0xBF0A_1514; // s_cmp_lt_u32 s20, s21
    const V_MOV_V2_1: u32 = 0x7E04_0281; // v_mov_b32 v2, 1
    const READFIRSTLANE_S21_V2: u32 = 0x7E2A_0502; // v_readfirstlane_b32 s21, v2
    const V_LSHRREV_V2_6_V0: u32 = 0x2C04_0086; // v_lshrrev_b32 v2, 6, v0
    pub(super) const SAVE_EXEC: u32 = 0xBE80_047E; // s_mov_b64 s[0:1], exec
    pub(super) const CMPX_V0_EQ_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec, 0, v0
    pub(super) const RESTORE_EXEC: u32 = 0xBEFE_0400; // s_mov_b64 exec, s[0:1]

    /// `v_cmp_gt_u32 vcc, k, v0` (the lanes below `k`), an inline constant.
    pub(super) fn cmp_below(k: u32) -> u32 {
        0x7D88_0000 | (128 + k)
    }

    pub(super) fn native(words: &[u32], threads: u32) -> NativeProgram {
        let range = |kind, table_offset_dwords, dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords,
            user_data: 0,
            slot: 0,
        };
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 32,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                range(ResourceType::Srv, 0, 8),
                range(ResourceType::Srv, 8, 4),
                range(ResourceType::Uav, 16, 4),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// The program lowered for a host of 32-lane waves, or its refusal.
    fn lower_halves(words: &[u32], threads: u32) -> Result<Lowered, crate::ShaderError> {
        let n = native(words, threads);
        let p = ir::build(&n.code, n.wave64, n.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &n).expect("io::raise");
        lower(&Parts {
            native: &n,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: false,
            host_wave32: true,
            bvh_no_hit: false,
        })
    }

    /// `lower_halves`'s outcome as the refusal string (`translated` when it
    /// lowers), the form the admit/refuse table asserts.
    fn outcome(words: &[u32], threads: u32) -> String {
        match lower_halves(words, threads) {
            Ok(_) => "translated".into(),
            Err(e) => e.to_string(),
        }
    }

    /// A `before`/`body`/`after` loop on `s20 < s21` (`s21` set by `before`),
    /// with the mask `vcc = lanes < 40` compared in `body`, each pass ended by
    /// `S_ADD_S20_1`, `S_CMP_LT_S20_S21` and a branch back to `body`.
    fn looped(before: &[u32], body: &[u32], after: &[u32]) -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(before);
        w.push(S_MOV_S20_0);
        w.extend(body);
        let back = -(body.len() as i32 + 3);
        w.extend([
            S_ADD_S20_1,
            S_CMP_LT_S20_S21,
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(after);
        w.push(S_ENDPGM);
        w
    }

    /// `body` of a loop: the mask, then its high dword stored as data.
    fn mask_body() -> Vec<u32> {
        vec![cmp_below(40), V_MOV_V1_VCC_HI]
    }

    /// A one-wave group (64 threads) whose loop count is a uniform first lane
    /// (1): the loop body is `in_step`, so the ballot goes there.
    pub(super) fn uniform_count_loop_reading_inside() -> Vec<u32> {
        let mut after = Vec::new();
        after.extend(store(1, 0));
        looped(&[V_MOV_V2_1, READFIRSTLANE_S21_V2], &mask_body(), &after)
    }

    /// The same, with the first-lane read under a lane-dependent mask: each
    /// half's first lane differs, so the loop is not `in_step`.
    pub(super) fn guarded_count_loop_reading_inside() -> Vec<u32> {
        let mut after = Vec::new();
        after.extend(store(1, 0));
        looped(
            &[
                V_MOV_V2_1,
                SAVE_EXEC,
                CMPX_V0_EQ_0,
                READFIRSTLANE_S21_V2,
                RESTORE_EXEC,
            ],
            &mask_body(),
            &after,
        )
    }

    /// A group of two console waves (128 threads) whose loop count is one
    /// value per console wave: the waves run different numbers of passes, so
    /// no group barrier can meet in the loop.
    pub(super) fn per_wave_count_loop_reading_inside() -> Vec<u32> {
        let mut after = Vec::new();
        after.extend(store(1, 0));
        looped(
            &[V_LSHRREV_V2_6_V0, READFIRSTLANE_S21_V2],
            &mask_body(),
            &after,
        )
    }

    /// The same per-wave-count loop, but the mask is read only after it: every
    /// lane runs the block the read starts in once, so the ballot goes there.
    pub(super) fn per_wave_count_loop_reading_after() -> Vec<u32> {
        let mut after = vec![S_BCNT1_S12_VCC, V_MOV_V1_S12];
        after.extend(store(1, 0));
        looped(
            &[V_LSHRREV_V2_6_V0, READFIRSTLANE_S21_V2],
            &[cmp_below(40)],
            &after,
        )
    }

    /// A mask computed in a per-lane region and read after reconvergence.
    ///
    /// ```text
    /// v_cmp_gt_u32 vcc, 32, v0      ; vcc = lanes < 32
    /// s_and_saveexec_b64 s[0:1], vcc ; EXEC = lanes < 32
    /// s_cbranch_execz skip          ; no lane to run: skip the region
    ///   v_cmp_gt_u32 vcc, 16, v0    ; region: vcc = lanes < 16 in EXEC
    /// skip:
    /// s_mov_b64 exec, s[0:1]        ; every lane again
    /// s_bcnt1_i32_b64 s12, vcc      ; the joined mask's number: 16
    /// ```
    ///
    /// The region block is a per-lane branch's (`in_step` is false there), so
    /// the ballot cannot be taken there; the join every lane reaches is.
    pub(super) fn region_mask_read_after_the_join() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            cmp_below(32),
            0xBE80_246A, // s_and_saveexec_b64 s[0:1], vcc
            0xBF88_0001, // s_cbranch_execz +1 (skip the region)
            cmp_below(16),
            0xBEFE_0400, // s_mov_b64 exec, s[0:1]
            S_BCNT1_S12_VCC,
            V_MOV_V1_S12,
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// `s_bcnt1_i32_b64 s12, exec`: a full console wave's live mask, 64.
    pub(super) fn live_mask_read() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xBE8C_107E, // s_bcnt1_i32_b64 s12, exec
            V_MOV_V1_S12,
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A literal lane mask kept in `other` (used as a `v_cndmask` predicate)
    /// and read as data: 0xFFFF_FFFF_FFFF_FFFC has 62 bits set.
    pub(super) fn literal_mask_read() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xBE94_04C4, // s_mov_b64 s[20:21], -4
            0xD501_0006,
            0x0051_0280, // v_cndmask_b32 v6, 0, 1, s[20:21]
            0xBE8C_1014, // s_bcnt1_i32_b64 s12, s[20:21]
            V_MOV_V1_S12,
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    #[test]
    fn a_per_wave_count_loop_reads_its_mask_after_it() {
        // The mask's block is the loop (not `in_step`: the waves run
        // different numbers of passes), but the block the read starts in is
        // (`once_per_lane`), so the ballot of both halves' bits goes there.
        let w = per_wave_count_loop_reading_after();
        let l = lower_halves(&w, 128).expect("the mask is balloted where read");
        assert_eq!(l.wave_size, Some(32), "a console wave in halves");
        assert!(l.wave_ops);
        assert!(l.half_exchanges > 0, "the halves passed their ballots");
    }

    #[test]
    fn a_mask_read_inside_a_per_wave_count_loop_still_refuses() {
        // The read starts in the loop body itself, which the two host halves
        // need not reach the same number of times: no barrier there.
        let why = outcome(&per_wave_count_loop_reading_inside(), 128);
        assert!(why.contains(MASK_ACROSS), "{why}");
    }

    #[test]
    fn a_guarded_first_lane_count_still_refuses_the_loop_ballot() {
        // The first-lane read is under a lane-dependent mask, so the loop is
        // not `in_step`: the same refusal as the per-wave count.
        let why = outcome(&guarded_count_loop_reading_inside(), 64);
        assert!(why.contains(MASK_ACROSS), "{why}");
    }

    #[test]
    fn a_uniform_count_loop_ballots_in_the_loop() {
        // A one-wave group's loop on a uniform first lane is `in_step`: the
        // ballot goes in the loop, where both halves meet.
        let l = lower_halves(&uniform_count_loop_reading_inside(), 64).expect("the loop is step");
        assert_eq!(l.wave_size, Some(32));
        assert!(l.wave_ops);
        assert!(l.half_exchanges > 0, "the halves passed their ballots");
    }

    #[test]
    fn a_region_mask_read_after_the_join_ballots_at_the_join() {
        let w = region_mask_read_after_the_join();
        let p = ir::build(
            &w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            true,
            32,
        )
        .unwrap();
        let st = structure::structure(&p).unwrap();
        let step = super::super::barriers::in_step(&p, &st);
        let region = (1..p.blocks.len())
            .find(|&b| p.blocks[b].insts.iter().any(|i| i.name() == "v_cmp_gt_u32"))
            .expect("region");
        assert!(!step[region], "the region is per-lane");
        assert!(step[0], "the entry runs in every lane");
        assert!(
            (0..p.blocks.len()).any(|b| b != region && step[b] && b != 0),
            "a later block every lane runs"
        );
        let l = lower_halves(&w, 64).expect("the joined mask is balloted at the join");
        assert_eq!(l.wave_size, Some(32));
        assert!(l.wave_ops);
        assert!(l.half_exchanges > 0, "the halves passed their ballots");
    }

    #[test]
    fn a_full_wave_live_mask_reads_without_an_exchange() {
        // A group of 64 is one full console wave: EXEC is all 64 bits in
        // every lane, so its number needs no ballot and no exchange.
        let l = lower_halves(&live_mask_read(), 64).expect("the live mask");
        assert_eq!(l.wave_size, Some(32));
        assert_eq!(l.half_exchanges, 0, "no exchange for a wave-uniform mask");
    }

    #[test]
    fn a_literal_mask_register_reads_as_its_number() {
        // A literal mask does not depend on the lanes, so the program needs
        // no console wave size (no halves): it reads the register's own
        // number (62 bits set in -4).
        let l = lower_halves(&literal_mask_read(), 64).expect("a literal mask");
        assert_eq!(l.wave_size, None, "a literal mask is wave-size independent");
        assert!(!l.wave_ops);
        assert_eq!(l.half_exchanges, 0);
    }

    /// The admit/refuse table for lane masks read as data, and the reason
    /// each refusal keeps. The checkpoint's other four cases are the
    /// pre-existing `lower::tests`: a cross-half ballot in a one-wave loop on
    /// a uniform first-lane read (admitted), a guarded first-lane count
    /// (refused), a mask joined with data at a per-lane loop (refused) and a
    /// per-console-wave first-lane count (refused).
    #[test]
    fn lane_mask_read_admit_and_refuse_table() {
        let table: Vec<(&str, Vec<u32>, u32, Option<&str>)> = vec![
            (
                "region mask read after the join",
                region_mask_read_after_the_join(),
                64,
                None,
            ),
            (
                "uniform count loop read inside",
                uniform_count_loop_reading_inside(),
                64,
                None,
            ),
            (
                "per-wave count loop read after",
                per_wave_count_loop_reading_after(),
                128,
                None,
            ),
            ("full wave live mask", live_mask_read(), 64, None),
            ("literal mask register", literal_mask_read(), 64, None),
            (
                "per-wave count loop read inside",
                per_wave_count_loop_reading_inside(),
                128,
                Some(MASK_ACROSS),
            ),
            (
                "guarded first-lane count loop",
                guarded_count_loop_reading_inside(),
                64,
                Some(MASK_ACROSS),
            ),
        ];
        for (name, words, threads, want) in table {
            let got = outcome(&words, threads);
            match want {
                None => assert_eq!(got, "translated", "{name}: {got}"),
                Some(reason) => {
                    assert!(got.contains(reason), "{name}: expected {reason:?}, got {got}")
                }
            }
        }
    }
}

/// Buckets 2-5 of the census (`wave_audit`): LDS a one-wave group shares
/// without barriers, whole-wave reductions and prefixes, first-lane reads and
/// lane reads that may cross the two 32-lane halves. Each shape the lowering
/// admits, and each adversarial shape it must still refuse, on the CPU.
#[cfg(test)]
mod cross_half_tests {
    use super::lane_mask_tests::{
        CMPX_V0_EQ_0, PROLOGUE, RESTORE_EXEC, SAVE_EXEC, S_ENDPGM, S_WAITCNT_LGKM,
        V_LSHLREV_V16_5_V0, cmp_below, native, store,
    };
    use super::{FIRST_ACROSS, Lowered, Parts, lower};
    use crate::native::StageIo;
    use crate::{io, ir, resource, structure};

    // ----- instructions the fixtures are made of ---------------------------

    /// `v2` = this lane's LDS dword, `v1` its `v_mbcnt`, `v4` the dword of
    /// the lane 32 away (the other half).
    const LDS_ADDRESSES: [u32; 5] = [
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
    ];

    /// The LDS access a one-wave group must get barriers around: a write of
    /// this lane's dword and a read of the other half's.
    fn lds_body() -> Vec<u32> {
        vec![
            0xD834_0000,
            0x0000_0102, // ds_write_b32 v2, v1
            S_WAITCNT_LGKM,
            0xD8D8_0000,
            0x0300_0004, // ds_read_b32 v3, v4
            S_WAITCNT_LGKM,
        ]
    }

    const V_MOV_V1_V0: u32 = 0x7E02_0300; // v_mov_b32 v1, v0
    const V_MOV_V5_1: u32 = 0x7E0A_0281; // v_mov_b32 v5, 1
    const V_READFIRSTLANE_S22_V5: u32 = 0x7E2C_0505; // v_readfirstlane_b32 s22, v5
    const V_MOV_V1_S22: u32 = 0x7E02_0216; // v_mov_b32 v1, s22
    const S_MOV_S20_0: u32 = 0xBE94_0380; // s_mov_b32 s20, 0
    const S_ADD_S20_1: u32 = 0x8014_8114; // s_add_u32 s20, s20, 1
    const S_CMP_LT_S20_22: u32 = 0xBF0A_1614; // s_cmp_lt_u32 s20, s22
    const S_CMP_LT_S20_2: u32 = 0xBF0A_8214; // s_cmp_lt_u32 s20, 2
    const S_MOV_B64_EXEC_ALL: u32 = 0xBEFE_04C1; // s_mov_b64 exec, -1

    /// A uniform compare of two scalars the table load wrote (`s40`, `s41`)
    /// into a lane mask: `s_cselect_b64 s[12:13], -1, 0`.
    const UNIFORM_COMPARE: [u32; 3] = [
        0xBF0A_2928, // s_cmp_lt_u32 s40, s41
        0x858C_80C1, // s_cselect_b64 s[12:13], -1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];

    /// VOP2 opcodes of the operations the reduction fixture takes.
    const V_ADD_U32: u32 = 0x25;
    const V_ADD_F32: u32 = 0x03;
    const V_MIN_U32: u32 = 0x13;
    const V_MAX_U32: u32 = 0x14;
    const V_MIN_I32: u32 = 0x11;
    const V_MAX_I32: u32 = 0x12;
    const V_AND_B32: u32 = 0x1b;
    const V_OR_B32: u32 = 0x1c;
    const V_XOR_B32: u32 = 0x1d;
    /// `v_mov_b32 v3, v0` (the input the reductions fold).
    const V_MOV_V3_V0: u32 = 0x7E06_0300;
    /// `v_xor_b32 v3, 42, v0`: values that differ per lane without a prefix
    /// of them summing to themselves.
    const V_XOR_V3_42_V0: u32 = 0x3A06_00AA;

    /// The console compiler's whole-wave reduction of `v3` by the VOP2
    /// operation `op`, the inline constant `ident` in inactive lanes (the
    /// same words as `emit::halves_tests`).
    fn reduction(op: u32, ident: u32, input: u32) -> Vec<u32> {
        let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            input,       // v3 from v0
            0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
            0xD501_0001,
            0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
        ]);
        for control in [0x161, 0x162, 0x164, 0x168] {
            w.extend([
                vop2(0x0602_02FA),
                0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
            ]);
        }
        w.extend([
            0xD778_0002,
            0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
            vop2(0x0602_0302), // op v1, v2, v1
            0xD760_000E,
            0x0001_4101,       // v_readlane_b32 s14, v1, 32
            vop2(0x0602_020E), // op v1, s14, v1
            0xD760_000F,
            0x0001_0101, // v_readlane_b32 s15, v1, 0
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
            0x7E02_020F, // v_mov_b32 v1, s15
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// Everything `reduction` adds after the prologue and address word, up to
    /// its store and end: the whole-wave section.
    fn reduction_section(op: u32, ident: u32, input: u32) -> Vec<u32> {
        let r = reduction(op, ident, input);
        r[PROLOGUE.len() + 1..r.len() - 3].to_vec()
    }

    /// The console compiler's exclusive prefix sum of the thread id
    /// (the same words as `emit::halves_tests`).
    fn exclusive_prefix() -> Vec<u32> {
        const EXEC: u32 = 126;
        let sop1 = |sdst: u32, op: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
        let vop1 = |vdst: u32, src0: u32| 0x7E00_0200 | vdst << 17 | src0;
        let add = |src0: u32| 0x4A02_0200 | src0; // v_add_nc_u32 v1, src0, v1
        let dpp = |ctrl: u32| 0xFF00_0001 | ctrl << 8; // of v1, row and bank masks 0xf
        let readlane = |lane: u32| [0xD760_0002, 0x0000_0101 | (128 + lane) << 9]; // s2, v1
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
            0xD501_0001,
            0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        ]);
        for shift in [1, 2, 4, 8] {
            w.extend([add(0xFA), dpp(0x110 + shift)]); // v_add_nc_u32 v1, v1, v1 row_shr
        }
        for (last, exec_lo, exec_hi) in [
            (15, 0xFFFF_0000, 0),
            (31, 0, 0x0000_FFFF),
            (47, 0, 0xFFFF_0000),
        ] {
            w.extend(readlane(last));
            w.extend([
                sop1(EXEC, 0x03, 255),
                exec_lo,
                sop1(EXEC + 1, 0x03, 255),
                exec_hi,
            ]);
            w.push(add(2)); // v_add_nc_u32 v1, s2, v1
        }
        w.extend([
            sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
            vop1(3, 257),          // v_mov_b32 v3, v1
            vop1(3, 0xFA),
            dpp(0x111), // v_mov_b32 v3, v1 row_shr:1
        ]);
        for (last, exec_lo, exec_hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
            w.extend(readlane(last));
            w.extend([
                sop1(EXEC, 0x03, 255),
                exec_lo,
                sop1(EXEC + 1, 0x03, 255),
                exec_hi,
            ]);
            w.push(vop1(3, 2)); // v_mov_b32 v3, s2
        }
        w.extend([
            sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
            vop1(2, 128),          // v_mov_b32 v2, 0
            sop1(EXEC, 0x04, 194), // s_mov_b64 exec, -2
            vop1(2, 259),          // v_mov_b32 v2, v3
            sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]
        ]);
        w.extend(store(2, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A `body` in a scalar loop counted to two, then the scalar `s<result>`
    /// stored: a count every host wave computes alike.
    fn loop_twice(body: &[u32], result: u32) -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([V_LSHLREV_V16_5_V0, S_MOV_S20_0]);
        let back = -(body.len() as i32 + 3);
        w.extend_from_slice(body);
        w.extend([
            S_ADD_S20_1,
            S_CMP_LT_S20_2,
            0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the body)
            0x7E02_0200 | result,                 // v_mov_b32 v1, s<result>
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A `body` in a scalar loop counted to a first-lane read of `v0` at
    /// least one and up to 63: one the host halves need not share.
    fn loop_to_first_lane(body: &[u32], result: u32) -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0x7E2A_0500, // v_readfirstlane_b32 s21, v0
            S_MOV_S20_0,
        ]);
        let back = -(body.len() as i32 + 3);
        w.extend_from_slice(body);
        w.extend([
            S_ADD_S20_1,
            0xBF0A_1514, // s_cmp_lt_u32 s20, s21
            0xBF85_0000 | (back as u32 & 0xFFFF),
            0x7E02_0200 | result, // v_mov_b32 v1, s<result>
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    // ----- LDS -------------------------------------------------------------

    /// A one-wave group of 64 writing its dword and reading the other half's,
    /// in one block.
    fn lds_straight() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend(lds_body());
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The same LDS access in a loop counted to four: a count every host
    /// wave computes alike, so the barriers go inside the loop.
    fn lds_loop() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend([V_MOV_V1_V0, S_MOV_S20_0]);
        let body = lds_body();
        let back = -(body.len() as i32 + 3);
        w.extend(body);
        w.extend([
            S_ADD_S20_1,
            0xBF0A_8414, // s_cmp_lt_u32 s20, 4
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The LDS loop counted to a first-lane read under an EXEC narrowing
    /// whose condition is an alike scalar (`s_cmp_lt_u32 s40, s41` of the
    /// loaded table dword): both halves enter the read, so it is alike in
    /// both and the loop is a block every lane runs together.
    fn lds_loop_guarded_by_alike_scalar() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend([V_MOV_V5_1]);
        w.extend(UNIFORM_COMPARE);
        w.extend([V_READFIRSTLANE_S22_V5, S_MOV_B64_EXEC_ALL]);
        w.extend([S_MOV_S20_0]);
        let body = lds_body();
        let back = -(body.len() as i32 + 3);
        w.extend(body);
        w.extend([
            S_ADD_S20_1,
            S_CMP_LT_S20_22,
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The same loop, but the first-lane read is under a lane mask
    /// (`v_cmpx_eq_u32 exec, 0, v0` of the thread id): each half's first
    /// active lane is its own, so the count may differ between them and the
    /// loop keeps refusing.
    fn lds_loop_guarded_by_lane_mask() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend([SAVE_EXEC, CMPX_V0_EQ_0]);
        w.extend([0x7E2C_0500, RESTORE_EXEC]); // v_readfirstlane_b32 s22, v0
        w.extend([S_MOV_S20_0]);
        let body = lds_body();
        let back = -(body.len() as i32 + 3);
        w.extend(body);
        w.extend([
            S_ADD_S20_1,
            S_CMP_LT_S20_22,
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// `region` run only by the lanes below `k`: a real lane branch
    /// (`v_cmp_gt_u32 vcc, k, v0`, `s_and_saveexec_b64`, a skip over the
    /// region, EXEC restored), whose blocks are then no blocks every lane of
    /// both halves runs together. An EXEC narrowing with no branch would not
    /// do: the block itself stays one all lanes reach.
    fn lane_region(k: u32, region: &[u32]) -> Vec<u32> {
        let skip = region.len() as u32;
        let mut w = vec![
            cmp_below(k),
            0xBE80_246A, // s_and_saveexec_b64 s[0:1], vcc
            0xBF88_0000 | skip, // s_cbranch_execz over the region
        ];
        w.extend_from_slice(region);
        w.push(RESTORE_EXEC); // s_mov_b64 exec, s[0:1]
        w
    }

    /// The instructions `words` decodes to (their number, not their words).
    fn instruction_count(words: &[u32]) -> usize {
        let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        code.extend(S_ENDPGM.to_le_bytes());
        crate::gfx10::decode(&code).expect("decodes").len() - 1
    }

    /// A write and a read of the other half's dword inside a lane region:
    /// the barrier they need can go only where some lanes do not reach.
    fn lds_accesses_in_a_lane_region() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend(lane_region(32, &lds_body()));
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A write every lane made, the read of the other half's dword in a lane
    /// region: the barrier goes at the end of the block every lane runs.
    fn lds_region_read_after_a_step_write() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend([0xD834_0000, 0x0000_0102, S_WAITCNT_LGKM]); // ds_write_b32 v2, v1
        w.extend(lane_region(
            32,
            &[0xD8D8_0000, 0x0300_0004, S_WAITCNT_LGKM], // ds_read_b32 v3, v4
        ));
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The write inside a lane region, the read of it after reconvergence:
    /// the barrier goes in the block every lane runs.
    fn lds_write_in_a_lane_region_read_after() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(LDS_ADDRESSES);
        w.extend(lane_region(32, &[0xD834_0000, 0x0000_0102, S_WAITCNT_LGKM]));
        w.extend([0xD8D8_0000, 0x0300_0004, S_WAITCNT_LGKM]);
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    // ----- reductions and prefixes ----------------------------------------

    /// A sum of the thread id in the entry block: a reduction every lane of
    /// the group runs together once.
    fn reduction_entry() -> Vec<u32> {
        reduction(V_ADD_U32, 0x80, V_MOV_V3_V0)
    }

    /// The same reduction inside a lane region: the block is not one every
    /// lane of both halves runs, so no group barrier can go there.
    fn reduction_in_a_lane_region() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend(lane_region(32, &reduction_section(V_ADD_U32, 0x80, V_MOV_V3_V0)));
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A sum in a loop counted to a first-lane read of `v0`: in a group of
    /// 128 the second console wave runs a pass and the first does not, so
    /// the loop body is no block both halves meet in.
    fn reduction_in_a_per_wave_loop() -> Vec<u32> {
        let section = reduction_section(V_ADD_U32, 0x80, V_MOV_V3_V0);
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0x2C04_0086, // v_lshrrev_b32 v2, 6, v0 (the console wave)
            0x7E2A_0502, // v_readfirstlane_b32 s21, v2
            S_MOV_S20_0,
        ]);
        let back = -(section.len() as i32 + 3);
        w.extend(section);
        w.extend([
            S_ADD_S20_1,
            0xBF0A_1514, // s_cmp_lt_u32 s20, s21
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// A sum in a loop counted to a first-lane read under an EXEC narrowing
    /// whose condition is an alike scalar: both halves enter the read, the
    /// loop is one block every lane runs the same number of times, and the
    /// reduction's exchange goes in it.
    fn reduction_in_a_guarded_uniform_loop() -> Vec<u32> {
        let section = reduction_section(V_ADD_U32, 0x80, V_MOV_V3_V0);
        let mut w = PROLOGUE.to_vec();
        w.push(V_LSHLREV_V16_5_V0);
        w.extend([V_MOV_V5_1]);
        w.extend(UNIFORM_COMPARE);
        w.extend([V_READFIRSTLANE_S22_V5, S_MOV_B64_EXEC_ALL, S_MOV_S20_0]);
        let back = -(section.len() as i32 + 3);
        w.extend(section);
        w.extend([
            S_ADD_S20_1,
            S_CMP_LT_S20_22,
            0xBF85_0000 | (back as u32 & 0xFFFF),
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    // ----- first-lane and lane reads --------------------------------------

    /// `v_readfirstlane_b32` of the thread id, stored: the console's first
    /// lane is in the lower half, so the two halves exchange.
    fn first_lane_of_the_thread_id() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0 (console lanes)
            0x7E18_0500, // v_readfirstlane_b32 s12, v0
            0x7E04_020C, // v_mov_b32 v2, s12
        ]);
        w.extend(store(2, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The first lane of `v0 >> 6` (one value per console wave, the same in
    /// both its halves): every lane already holds it.
    fn first_lane_alike() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
            0x7E18_0502, // v_readfirstlane_b32 s12, v2
            0x7E06_020C, // v_mov_b32 v3, s12
        ]);
        w.extend(store(3, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The first lane of the thread id inside a lane region: each half would
    /// read its own.
    fn first_lane_in_a_lane_region() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([V_LSHLREV_V16_5_V0, 0xD765_0001, 0x0001_00C1]);
        w.extend(lane_region(32, &[0x7E18_0500])); // v_readfirstlane_b32 s12, v0
        w.extend([0x7E04_020C]); // v_mov_b32 v2, s12
        w.extend(store(2, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The first lane of the thread id in a loop counted to it: a count the
    /// halves need not share.
    fn first_lane_in_a_per_wave_loop() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x7E2A_0500, // v_readfirstlane_b32 s21, v0
            S_MOV_S20_0,
            0x7E2C_0500, // v_readfirstlane_b32 s22, v0 (in the loop)
        ]);
        let back = -4;
        w.extend([
            S_ADD_S20_1,
            0xBF0A_1514, // s_cmp_lt_u32 s20, s21
            0xBF85_0000 | (back as u32 & 0xFFFF),
            0x7E04_0216, // v_mov_b32 v2, s22
        ]);
        w.extend(store(2, 0));
        w.push(S_ENDPGM);
        w
    }

    /// `v_readlane_b32 s12, v0, 40`: a constant console lane, in the other
    /// half for every lane of the lower one.
    fn lane_read_constant() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xD760_000C,
            0x0001_5100, // v_readlane_b32 s12, v0, 40
            0x7E02_020C, // v_mov_b32 v1, s12
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// `v_readlane_b32 s12, v3, s13` of a value every lane holds (`v3` = 1,
    /// the lane in `s13`): no exchange is needed.
    fn lane_read_uniform_source() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0x7E06_0281, // v_mov_b32 v3, 1
            0xBE8D_03FF,
            33, // s_mov_b32 s13, 33
            0xD760_000C,
            0x0001_1B03, // v_readlane_b32 s12, v3, s13
            0x7E02_020C, // v_mov_b32 v1, s12
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// `v_readlane_b32 s12, v0, v1` of a per-lane index (`v1` the lane's
    /// `v_mbcnt`): the exchange holds one dword per half, so a lane read
    /// whose lanes ask for different console lanes keeps refusing.
    fn lane_read_per_lane_index() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            V_LSHLREV_V16_5_V0,
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0xD760_000C,
            0x0002_0300, // v_readlane_b32 s12, v0, v1
            0x7E02_020C, // v_mov_b32 v1, s12
        ]);
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    /// The same read inside a lane region: no block both halves meet in.
    fn lane_read_in_a_lane_region() -> Vec<u32> {
        let mut w = PROLOGUE.to_vec();
        w.extend([V_LSHLREV_V16_5_V0, 0xD765_0001, 0x0001_00C1]);
        w.extend(lane_region(32, &[0xD760_000C, 0x0001_5100]));
        w.extend([0x7E02_020C]); // v_mov_b32 v1, s12
        w.extend(store(1, 0));
        w.push(S_ENDPGM);
        w
    }

    // ----- the harness -----------------------------------------------------

    /// The program lowered for a host of 32-lane waves, with `lds_bytes` of
    /// LDS declared and `narrow_waves` (D3DMetal's `split_wave_barriers`).
    fn lower_with(
        words: &[u32],
        threads: u32,
        lds_bytes: u32,
        narrow_waves: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_flags(words, threads, lds_bytes, narrow_waves, true)
    }

    /// [`lower_with`] with both host-width options given: `host_wave32` (the
    /// host refuses `WaveSize(64)`, so a wave64 program runs in halves) and
    /// `narrow_waves` (the host runs a wave64 program in 32-lane waves whatever
    /// `WaveSize` it declares).
    fn lower_flags(
        words: &[u32],
        threads: u32,
        lds_bytes: u32,
        narrow_waves: bool,
        host_wave32: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let mut n = native(words, threads);
        if let StageIo::Cs(cs) = &mut n.io {
            cs.rsrc2 |= lds_bytes.div_ceil(512) << 15;
        }
        let p = ir::build(&n.code, n.wave64, n.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &n).expect("io::raise");
        lower(&Parts {
            native: &n,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves,
            host_wave32,
            bvh_no_hit: false,
        })
    }

    /// One row of an admit/refuse table: the fixture, its group, LDS bytes,
    /// whether narrow host waves sync LDS, the refusal it keeps (or none) and
    /// the barriers a lowered program must have placed.
    struct Row {
        name: &'static str,
        words: Vec<u32>,
        threads: u32,
        lds_bytes: u32,
        narrow_waves: bool,
        refuse: Option<&'static str>,
        barriers: usize,
        exchanges: Option<u32>,
    }

    fn row(
        name: &'static str,
        words: Vec<u32>,
        threads: u32,
        lds_bytes: u32,
        narrow_waves: bool,
        refuse: Option<&'static str>,
        barriers: usize,
        exchanges: Option<u32>,
    ) -> Row {
        Row {
            name,
            words,
            threads,
            lds_bytes,
            narrow_waves,
            refuse,
            barriers,
            exchanges,
        }
    }

    /// Asserts every row: the admitted ones lower in halves with the
    /// barriers and exchanges they need, the refused ones keep their reason.
    fn check(rows: Vec<Row>, want_wave: Option<u32>) {
        for Row {
            name,
            words,
            threads,
            lds_bytes,
            narrow_waves,
            refuse,
            barriers,
            exchanges,
        } in rows
        {
            match lower_with(&words, threads, lds_bytes, narrow_waves) {
                Ok(l) => {
                    assert!(
                        refuse.is_none(),
                        "{name}: expected {:?}, lowered",
                        refuse.unwrap()
                    );
                    assert_eq!(l.wave_size, want_wave, "{name}: wave size");
                    assert_eq!(l.split_barriers, barriers, "{name}: barriers");
                    if let Some(want) = exchanges {
                        assert_eq!(l.half_exchanges, want, "{name}: exchanges");
                    }
                }
                Err(e) => {
                    let got = e.to_string();
                    match refuse {
                        Some(reason) => assert!(
                            got.contains(reason),
                            "{name}: expected {reason:?}, got {got}"
                        ),
                        None => panic!("{name}: refused ({got})"),
                    }
                }
            }
        }
    }

    const LDS_WHY: &str =
        "LDS one console wave64 shares without barriers, on a host of 32-lane waves";
    const REDUCTION_WHY: &str =
        "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)";
    const LANE_READ_WHY: &str =
        "a lane read that may cross 32-lane halves (wave64 on a 32-lane host)";

    #[test]
    fn lds_admit_and_refuse_table() {
        check(
            vec![
                // Straight-line LDS: one barrier between the write and the
                // read of the other half.
                row("straight", lds_straight(), 64, 512, true, None, 1, None),
                // A loop counted to four: a barrier before the read and one
                // before the next pass's write.
                row("loop", lds_loop(), 64, 512, true, None, 2, None),
                // A loop counted to a first-lane read under an alike-scalar
                // EXEC narrowing: the widening this task makes.
                row(
                    "loop on a guarded uniform count",
                    lds_loop_guarded_by_alike_scalar(),
                    64,
                    512,
                    true,
                    None,
                    2,
                    None,
                ),
                // The same read under a lane mask: the halves may run
                // different numbers of passes, so the loop keeps refusing.
                row(
                    "loop on a per-lane count",
                    lds_loop_guarded_by_lane_mask(),
                    64,
                    512,
                    true,
                    Some(LDS_WHY),
                    0,
                    None,
                ),
                // The write and the read in a lane region: no barrier can
                // go where only some lanes reach.
                row(
                    "accesses in a lane region",
                    lds_accesses_in_a_lane_region(),
                    64,
                    512,
                    true,
                    Some(LDS_WHY),
                    0,
                    None,
                ),
                // A write every lane made, read by some: the barrier sits at
                // the end of the block every lane runs, before the region.
                row(
                    "read in a region after a step write",
                    lds_region_read_after_a_step_write(),
                    64,
                    512,
                    true,
                    None,
                    1,
                    None,
                ),
                // The write in a per-lane region, the read after it: the
                // barrier goes in the block every lane runs.
                row(
                    "write in a region, read after",
                    lds_write_in_a_lane_region_read_after(),
                    64,
                    512,
                    true,
                    None,
                    1,
                    None,
                ),
                // Without D3DMetal's narrow host waves but on a host whose
                // waves are 32 lanes (halves), the same barrier is placed:
                // the split, not the switch, is what needs it.
                row("straight, no narrow waves", lds_straight(), 64, 512, false, None, 1, None),
                // Where no barrier can go (LDS in a per-lane region), the
                // group still refuses with that reason, switch or not.
                row(
                    "accesses in a lane region, no narrow waves",
                    lds_accesses_in_a_lane_region(),
                    64,
                    512,
                    false,
                    Some(LDS_WHY),
                    0,
                    None,
                ),
                row(
                    "loop on a per-lane count, no narrow waves",
                    lds_loop_guarded_by_lane_mask(),
                    64,
                    512,
                    false,
                    Some(LDS_WHY),
                    0,
                    None,
                ),
                // A group of 32 or fewer is one host wave: no split, no
                // barrier, no refusal.
                row("straight, one host wave", lds_straight(), 32, 512, false, None, 0, None),
            ],
            Some(32),
        );
    }

    #[test]
    fn reduction_admit_and_refuse_table() {
        // A sum, the bit operations, the unsigned and signed min and max, the
        // floats, and the exclusive prefix sum, each over the console wave.
        let mut rows = vec![
            row("sum", reduction_entry(), 64, 0, false, None, 0, None),
            row("or", reduction(V_OR_B32, 0x80, V_MOV_V3_V0), 64, 0, false, None, 0, None),
            row("and", reduction(V_AND_B32, 0xC1, V_XOR_V3_42_V0), 64, 0, false, None, 0, None),
            row("xor", reduction(V_XOR_B32, 0x80, V_XOR_V3_42_V0), 64, 0, false, None, 0, None),
            row("min_u32", reduction(V_MIN_U32, 0xC1, V_XOR_V3_42_V0), 64, 0, false, None, 0, None),
            row("max_u32", reduction(V_MAX_U32, 0x80, V_XOR_V3_42_V0), 64, 0, false, None, 0, None),
            row("prefix", exclusive_prefix(), 64, 0, false, None, 0, None),
            // Every admitted reduction in halves passed its halves' results:
            // at least one exchange, on a group of two host waves.
            row("sum, two host waves", reduction_entry(), 96, 0, false, None, 0, Some(1)),
        ];
        for r in &mut rows {
            if r.exchanges.is_none() {
                r.exchanges = Some(1);
            }
        }
        rows.push(row(
            "sum, one host wave",
            reduction_entry(),
            32,
            0,
            false,
            None,
            0,
            Some(0),
        ));
        rows.push(row(
            "sum in a lane region",
            reduction_in_a_lane_region(),
            64,
            0,
            false,
            Some(REDUCTION_WHY),
            0,
            None,
        ));
        rows.push(row(
            "sum in a per-wave loop",
            reduction_in_a_per_wave_loop(),
            128,
            0,
            false,
            Some(REDUCTION_WHY),
            0,
            None,
        ));
        // A loop on a first-lane read under an alike-scalar narrowing runs
        // alike in both halves, so the reduction's exchange goes inside it.
        rows.push(row(
            "sum in a guarded uniform loop",
            reduction_in_a_guarded_uniform_loop(),
            64,
            0,
            false,
            None,
            0,
            Some(1),
        ));
        check(rows, Some(32));
    }

    #[test]
    fn first_lane_and_lane_read_admit_and_refuse_table() {
        check(
            vec![
                // A value every lane holds: no exchange.
                row("first lane, alike", first_lane_alike(), 128, 0, false, None, 0, Some(0)),
                // A per-lane value: the lower half's first lane, exchanged.
                row(
                    "first lane, per lane",
                    first_lane_of_the_thread_id(),
                    64,
                    0,
                    false,
                    None,
                    0,
                    Some(1),
                ),
                row(
                    "first lane in a lane region",
                    first_lane_in_a_lane_region(),
                    64,
                    0,
                    false,
                    Some(FIRST_ACROSS),
                    0,
                    None,
                ),
                row(
                    "first lane in a per-wave loop",
                    first_lane_in_a_per_wave_loop(),
                    64,
                    0,
                    false,
                    Some(FIRST_ACROSS),
                    0,
                    None,
                ),
                // A constant console lane in the other half: exchanged.
                row("lane read, constant lane", lane_read_constant(), 64, 0, false, None, 0, Some(1)),
                // A value every lane holds: no exchange.
                row(
                    "lane read, alike source",
                    lane_read_uniform_source(),
                    64,
                    0,
                    false,
                    None,
                    0,
                    Some(0),
                ),
                // A per-lane lane index (`v_mbcnt`): the 2-dword exchange
                // holds one value per half, so the lanes would all read the
                // one lane that stored last. Refused, as before.
                row(
                    "lane read, per-lane index",
                    lane_read_per_lane_index(),
                    64,
                    0,
                    false,
                    Some(LANE_READ_WHY),
                    0,
                    None,
                ),
                row(
                    "lane read in a lane region",
                    lane_read_in_a_lane_region(),
                    64,
                    0,
                    false,
                    Some(LANE_READ_WHY),
                    0,
                    None,
                ),
            ],
            Some(32),
        );
    }

    /// The re-census of the historical 111 compute rejects on a synthetic
    /// corpus (`wave_audit`, "Census of the title run"). The title shaders and
    /// the diagnostic log are not in the repository, so the corpus here is
    /// synthetic: the shapes each bucket is made of, as the fixtures in this
    /// file and `lane_mask_tests` encode them. Every count this test prints is
    /// a count of these programs, never of the title's.
    ///
    /// Each program is lowered for the two option sets a 32-lane host can be
    /// translated with: `wine` (`narrow_waves` + `host_wave32`, D3DMetal's
    /// switch on a host whose device says 32 lanes) and `wave32` (the host
    /// refuses `WaveSize(64)`, so the program runs in halves; no D3DMetal
    /// switch). A program both sets translate must come out as the same DXIL:
    /// the halves' LDS barriers are the same barriers. A third set,
    /// `narrow_waves` without `host_wave32`, is refused for every program
    /// (ADR-025a, `ShaderError::Options`).
    mod wave_corpus_tests {
        use super::super::lane_mask_tests::{
            PROLOGUE, S_BCNT1_S12_VCC, S_ENDPGM, V_LSHLREV_V16_5_V0, V_MOV_V1_S12, cmp_below,
            guarded_count_loop_reading_inside, live_mask_read, literal_mask_read,
            per_wave_count_loop_reading_after, per_wave_count_loop_reading_inside,
            region_mask_read_after_the_join, store, uniform_count_loop_reading_inside,
        };
        use super::{
            V_ADD_U32, V_MOV_V3_V0, V_OR_B32,
            exclusive_prefix, first_lane_alike,
            first_lane_in_a_lane_region, first_lane_in_a_per_wave_loop,
            first_lane_of_the_thread_id, lane_read_constant, lane_read_in_a_lane_region,
            lane_read_per_lane_index, lane_read_uniform_source, lane_region,
            lds_accesses_in_a_lane_region, lds_loop, lds_loop_guarded_by_alike_scalar,
            lds_loop_guarded_by_lane_mask, lds_region_read_after_a_step_write, lds_straight,
            lds_write_in_a_lane_region_read_after, lower_flags, lower_with, reduction,
            reduction_entry, reduction_in_a_guarded_uniform_loop,
            reduction_in_a_lane_region, reduction_in_a_per_wave_loop, reduction_section,
        };
        use std::collections::BTreeMap;

        /// `s_mov_b64 s[20:21], vcc`: a mask's value kept in a scalar pair.
        const SAVE_VCC_S20: u32 = 0xBE94_036A;
        /// `s_bcnt1_i32_b64 s12, s[20:21]`: that pair's number.
        const BCNT_S12_S20: u32 = 0xBE8C_1014;

        /// One corpus program: its historical bucket, what it is, its words,
        /// its group and its LDS bytes.
        struct Program {
            bucket: &'static str,
            name: &'static str,
            words: Vec<u32>,
            threads: u32,
            lds: u32,
        }

        fn p(
            bucket: &'static str,
            name: &'static str,
            words: Vec<u32>,
            threads: u32,
            lds: u32,
        ) -> Program {
            Program {
                bucket,
                name,
                words,
                threads,
                lds,
            }
        }

        /// A mask computed and read in the entry block: the console's ballot
        /// at its most common place.
        fn ballot_at_entry() -> Vec<u32> {
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0, cmp_below(40), S_BCNT1_S12_VCC, V_MOV_V1_S12]);
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// The same mask kept in `s[20:21]` at the entry, read inside a lane
        /// region: the ballot can still go where the mask is written.
        fn ballot_at_entry_read_in_a_lane_region() -> Vec<u32> {
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0, cmp_below(40), SAVE_VCC_S20]);
            w.extend(lane_region(32, &[BCNT_S12_S20, V_MOV_V1_S12]));
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// A mask computed and read inside the same lane region: the two host
        /// halves do not meet there, so it keeps refusing.
        fn ballot_in_a_lane_region_read_there() -> Vec<u32> {
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0]);
            w.extend(lane_region(32, &[cmp_below(16), S_BCNT1_S12_VCC, V_MOV_V1_S12]));
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// A mask written at the entry, read inside a loop counted to two: a
        /// block every lane runs the same number of times.
        fn ballot_at_entry_read_in_a_counted_loop() -> Vec<u32> {
            const S_MOV_S22_0: u32 = 0xBE96_0380; // s_mov_b32 s22, 0
            const S_ADD_S22_1: u32 = 0x8016_8116; // s_add_u32 s22, s22, 1
            const S_CMP_LT_S22_2: u32 = 0xBF0A_8216; // s_cmp_lt_u32 s22, 2
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0, cmp_below(40), SAVE_VCC_S20, S_MOV_S22_0]);
            let body = [BCNT_S12_S20, V_MOV_V1_S12];
            let back = -(body.len() as i32 + 3);
            w.extend(body);
            w.extend([S_ADD_S22_1, S_CMP_LT_S22_2, 0xBF85_0000 | (back as u32 & 0xFFFF)]);
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// A literal mask read inside a lane region: its number does not
        /// depend on the lanes, so no ballot and no exchange is needed there.
        fn literal_mask_in_a_lane_region() -> Vec<u32> {
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0]);
            w.extend([0xBE94_04C4]); // s_mov_b64 s[20:21], -4
            w.extend(lane_region(32, &[BCNT_S12_S20, V_MOV_V1_S12]));
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// A sum in a loop counted to two: a count every host wave computes
        /// alike, the plain form of the guarded loop.
        fn reduction_in_a_counted_loop() -> Vec<u32> {
            const S_MOV_S22_0: u32 = 0xBE96_0380;
            const S_ADD_S22_1: u32 = 0x8016_8116;
            const S_CMP_LT_S22_2: u32 = 0xBF0A_8216;
            let section = reduction_section(V_ADD_U32, 0x80, V_MOV_V3_V0);
            let mut w = PROLOGUE.to_vec();
            w.extend([V_LSHLREV_V16_5_V0, S_MOV_S22_0]);
            let back = -(section.len() as i32 + 3);
            w.extend(section);
            w.extend([S_ADD_S22_1, S_CMP_LT_S22_2, 0xBF85_0000 | (back as u32 & 0xFFFF)]);
            w.extend(store(1, 0));
            w.push(S_ENDPGM);
            w
        }

        /// The corpus: the shapes the historical buckets are made of (54 lane
        /// masks, 30 LDS, 12 reductions, 8 first lanes, 3 lane reads, 2 masks
        /// outside the wave64 parenthetical) with the adversarial shapes each
        /// change must still refuse. The title run's two non-wave rejects
        /// (buckets 7-8) are not programs this crate lowers: one is a
        /// `victory-gpu` diagnostic switch, the other a descriptor shape.
        fn corpus() -> Vec<Program> {
            vec![
                p("mask", "ballot at entry", ballot_at_entry(), 64, 0),
                p(
                    "mask",
                    "region mask read after the join",
                    region_mask_read_after_the_join(),
                    64,
                    0,
                ),
                p(
                    "mask",
                    "uniform count loop read inside",
                    uniform_count_loop_reading_inside(),
                    64,
                    0,
                ),
                p(
                    "mask",
                    "per-wave count loop read after",
                    per_wave_count_loop_reading_after(),
                    128,
                    0,
                ),
                p(
                    "mask",
                    "per-wave count loop read inside",
                    per_wave_count_loop_reading_inside(),
                    128,
                    0,
                ),
                p(
                    "mask",
                    "guarded first-lane count loop",
                    guarded_count_loop_reading_inside(),
                    64,
                    0,
                ),
                p("mask", "full wave live mask", live_mask_read(), 64, 0),
                p("mask", "literal mask register", literal_mask_read(), 64, 0),
                p(
                    "mask",
                    "entry ballot read in a lane region",
                    ballot_at_entry_read_in_a_lane_region(),
                    64,
                    0,
                ),
                p(
                    "mask",
                    "ballot in a lane region read there",
                    ballot_in_a_lane_region_read_there(),
                    64,
                    0,
                ),
                p(
                    "mask",
                    "entry ballot read in a counted loop",
                    ballot_at_entry_read_in_a_counted_loop(),
                    64,
                    0,
                ),
                p("lds", "straight", lds_straight(), 64, 512),
                p("lds", "loop", lds_loop(), 64, 512),
                p(
                    "lds",
                    "loop on a guarded uniform count",
                    lds_loop_guarded_by_alike_scalar(),
                    64,
                    512,
                ),
                p(
                    "lds",
                    "loop on a per-lane count",
                    lds_loop_guarded_by_lane_mask(),
                    64,
                    512,
                ),
                p(
                    "lds",
                    "accesses in a lane region",
                    lds_accesses_in_a_lane_region(),
                    64,
                    512,
                ),
                p(
                    "lds",
                    "read in a region after a step write",
                    lds_region_read_after_a_step_write(),
                    64,
                    512,
                ),
                p(
                    "lds",
                    "write in a region, read after",
                    lds_write_in_a_lane_region_read_after(),
                    64,
                    512,
                ),
                p("reduction", "sum", reduction_entry(), 64, 0),
                p("reduction", "prefix", exclusive_prefix(), 64, 0),
                p("reduction", "or", reduction(V_OR_B32, 0x80, V_MOV_V3_V0), 64, 0),
                p(
                    "reduction",
                    "sum in a lane region",
                    reduction_in_a_lane_region(),
                    64,
                    0,
                ),
                p(
                    "reduction",
                    "sum in a per-wave loop",
                    reduction_in_a_per_wave_loop(),
                    128,
                    0,
                ),
                p(
                    "reduction",
                    "sum in a guarded uniform loop",
                    reduction_in_a_guarded_uniform_loop(),
                    64,
                    0,
                ),
                p(
                    "reduction",
                    "sum in a counted loop",
                    reduction_in_a_counted_loop(),
                    64,
                    0,
                ),
                p(
                    "first lane",
                    "per-lane value",
                    first_lane_of_the_thread_id(),
                    64,
                    0,
                ),
                p("first lane", "alike value", first_lane_alike(), 128, 0),
                p(
                    "first lane",
                    "in a lane region",
                    first_lane_in_a_lane_region(),
                    64,
                    0,
                ),
                p(
                    "first lane",
                    "in a per-wave loop",
                    first_lane_in_a_per_wave_loop(),
                    64,
                    0,
                ),
                p("lane read", "constant lane", lane_read_constant(), 64, 0),
                p(
                    "lane read",
                    "alike source",
                    lane_read_uniform_source(),
                    64,
                    0,
                ),
                p(
                    "lane read",
                    "per-lane index",
                    lane_read_per_lane_index(),
                    64,
                    0,
                ),
                p(
                    "lane read",
                    "in a lane region",
                    lane_read_in_a_lane_region(),
                    64,
                    0,
                ),
                p(
                    "mask",
                    "partial wave live mask",
                    live_mask_read(),
                    40,
                    0,
                ),
                p(
                    "mask",
                    "literal mask read in a lane region",
                    literal_mask_in_a_lane_region(),
                    64,
                    0,
                ),
            ]
        }

        fn outcome(program: &Program, narrow: bool, host_wave32: bool) -> String {
            match lower_flags(
                &program.words,
                program.threads,
                program.lds,
                narrow,
                host_wave32,
            ) {
                Ok(_) => "translated".to_string(),
                Err(e) => e.to_string(),
            }
        }

        /// The reason string without its `GFX10 code at 0x..:` prefix and
        /// `is not supported` suffix, the form the historical census counted.
        fn reason_of(outcome: &str) -> String {
            let start = outcome.find(": ").map_or(0, |i| i + 2);
            let text = outcome[start..].trim();
            text.strip_suffix(" is not supported")
                .unwrap_or(text)
                .to_string()
        }

        /// Per bucket, the refusals by reason, and the totals.
        fn census(
            programs: &[Program],
            narrow: bool,
            host_wave32: bool,
        ) -> (BTreeMap<String, usize>, BTreeMap<&'static str, usize>, usize) {
            let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
            let mut buckets: BTreeMap<&'static str, usize> = BTreeMap::new();
            let mut refused = 0;
            for program in programs {
                let got = outcome(program, narrow, host_wave32);
                if got != "translated" {
                    refused += 1;
                    *reasons.entry(reason_of(&got)).or_default() += 1;
                    *buckets.entry(program.bucket).or_default() += 1;
                }
            }
            (reasons, buckets, refused)
        }

        #[test]
        fn wave64_corpus_census() {
            let programs = corpus();
            let buckets = programs.len();
            for (label, narrow, host_wave32) in [
                ("narrow waves alone (refused)", true, false),
                ("wine, halves on", true, true),
                ("wave32 host, no switch", false, true),
            ] {
                let (reasons, by_bucket, refused) = census(&programs, narrow, host_wave32);
                eprintln!(
                    "census {label}: {buckets} programs, {refused} refused, {} translated",
                    buckets - refused
                );
                for (bucket, count) in &by_bucket {
                    eprintln!("census {label}:   {count:2} refused in {bucket}");
                }
                for (reason, count) in &reasons {
                    eprintln!("census {label}:   {count:2} {reason}");
                }
                for program in &programs {
                    let got = outcome(program, narrow, host_wave32);
                    let shown = reason_of(&got);
                    eprintln!(
                        "census {label}: {:<10} {:<40} {}",
                        program.bucket, program.name, shown
                    );
                }
            }
            // Pinned: the corpus' own counts, so a shape that starts or stops
            // refusing is noticed. 35 programs: 13 lane masks, 7 LDS, 7
            // reductions, 4 first lanes and 4 lane reads; the historical
            // census' other shapes and its two non-wave rejects are described
            // in `wave_audit`, not lowered here.
            let (wine_reasons, wine_buckets, wine_refused) =
                census(&programs, true, true);
            assert_eq!((programs.len(), wine_refused), (35, 11), "wine config");
            assert_eq!(
                wine_buckets,
                BTreeMap::from([
                    ("first lane", 2),
                    ("lane read", 2),
                    ("lds", 2),
                    ("mask", 3),
                    ("reduction", 2),
                ]),
                "wine config: refusals by bucket"
            );
            assert_eq!(
                wine_reasons,
                BTreeMap::from([
                    (
                        "a first-lane read of a value that may differ between 32-lane halves \
                         (wave64 on a 32-lane host)"
                            .to_string(),
                        2
                    ),
                    (
                        "a lane mask read as data (wave64 on a 32-lane host)".to_string(),
                        3
                    ),
                    (
                        "a lane read that may cross 32-lane halves (wave64 on a 32-lane host)"
                            .to_string(),
                        2
                    ),
                    (
                        "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)"
                            .to_string(),
                        2
                    ),
                    (
                        "LDS one console wave64 shares without barriers, on a host of 32-lane \
                         waves"
                            .to_string(),
                        2
                    ),
                ]),
                "wine config: refusals by reason"
            );
            // A host that refuses WaveSize(64) needs the same LDS lowering as
            // D3DMetal's narrow waves: before the barriers were keyed to the
            // split rather than to the switch, all seven LDS programs here
            // refused on it (16 refused) and only the divergent two do now.
            let (_, wave32_buckets, wave32_refused) = census(&programs, false, true);
            assert_eq!(wave32_refused, wine_refused, "wave32 config");
            assert_eq!(wave32_buckets, wine_buckets, "wave32 config: by bucket");
            // The LDS barriers are the same in both option sets: a program the
            // wave32 host config admits emits the DXIL the D3DMetal config
            // emits, which is the one the GPU fixtures ran.
            for program in &programs {
                if program.lds == 0 {
                    continue;
                }
                if let (Ok(a), Ok(b)) = (
                    lower_with(&program.words, program.threads, program.lds, true),
                    lower_with(&program.words, program.threads, program.lds, false),
                ) {
                    assert_eq!(a.dxil, b.dxil, "{}: both configs' DXIL", program.name);
                }
            }
            // With the halves lowering off (`host_wave32` false) and
            // `narrow_waves` on, every program is refused (ADR-025a): that
            // pair asserts the host runs 32-lane waves and then lowers as if
            // it ran 64, so the ballots' high dword would be read as 0. The
            // refusal is a single explicit options error before any code is
            // emitted, not a per-shape reason.
            let (halves_off_reasons, _, halves_off) = census(&programs, true, false);
            assert_eq!(
                halves_off, buckets,
                "narrow_waves without host_wave32 refuses every program"
            );
            assert_eq!(
                halves_off_reasons,
                BTreeMap::from([(
                    crate::emit::NARROW_WAVES_WITHOUT_HOST_WAVE32.to_string(),
                    buckets
                )]),
                "the unsound pair's one reason"
            );
        }

        /// ADR-025a, fail closed: `narrow_waves` without `host_wave32` is
        /// refused for every corpus program with the explicit options error,
        /// before any code is emitted — never lowered with a zero high dword.
        #[test]
        fn narrow_waves_without_host_wave32_refuses_the_whole_corpus() {
            let programs = corpus();
            assert_eq!(programs.len(), 35);
            for program in &programs {
                match lower_flags(
                    &program.words,
                    program.threads,
                    program.lds,
                    true,
                    false,
                ) {
                    Ok(_) => panic!(
                        "{}: lowered `narrow_waves` without `host_wave32`",
                        program.name
                    ),
                    Err(crate::ShaderError::Options(why)) => assert_eq!(
                        why,
                        crate::emit::NARROW_WAVES_WITHOUT_HOST_WAVE32,
                        "{}: reason",
                        program.name
                    ),
                    Err(e) => panic!("{}: expected the options refusal, got {e}", program.name),
                }
            }
        }

        /// The supported pair still lowers: every program that needs the
        /// console's lanes declares `WaveSize(32)` and takes the halves path
        /// (the `!31` half-wave mask), and at least one does.
        #[test]
        fn narrow_waves_with_host_wave32_lowers_in_halves() {
            let mut in_halves = 0;
            for program in &corpus() {
                let Ok(l) = lower_flags(&program.words, program.threads, program.lds, true, true)
                else {
                    continue;
                };
                if let Some(wave) = l.wave_size {
                    assert_eq!(wave, 32, "{}: wave size", program.name);
                    assert!(
                        l.has_half_wave_mask,
                        "{}: the halves' `!31` mask",
                        program.name
                    );
                    in_halves += 1;
                }
            }
            assert_eq!(in_halves, 22, "programs that need the console's lanes");
        }
    }
}
