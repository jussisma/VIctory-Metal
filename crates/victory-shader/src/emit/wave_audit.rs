//! Categorised audit and lowering plan for refused compute shaders.
//!
//! Reserved by the orchestrator so the compute-unblock task can add modules and
//! tests without touching `emit/mod.rs`. The wave task owns this file.
//!
//! # Census of the title run
//!
//! Source: the diagnostic log
//! `/private/var/folders/2t/mr_0f1ts0t3_g4jghhw9lnjh0000gn/T/victory-mesh-a5991572.ha_4rh9o.log`
//! (2,461,886 bytes, 12,300 lines), filtered to
//! `shader does not translate ... stage="compute"`. The 111 lines split into
//! the reasons below; the count is of title shaders (hashes), not of
//! instructions inside them. No shader bytes or log contents are committed:
//! only the reason strings and the counts they appeared with.
//!
//! | # | reason string (as emitted) | count | exact lowering site |
//! |---|----------------------------|------:|---------------------|
//! | 1 | `a lane mask read as data (wave64 on a 32-lane host)` | 54 | `super::lower::halves::MASK_ACROSS` (`halves.rs`), raised in `Lower::mask_dword` (`lower.rs`) when neither the block writing the mask nor the block the read starts in is `in_step` |
//! | 2 | `LDS one console wave64 shares without barriers, on a host of 32-lane waves` | 30 | `Lower::halves_lds` (`halves.rs`), raised through `halves_refused` in `Lower::run` (`lower.rs`) when `split_wave_barriers` (`barriers.rs`) returns no placement for a `step` block |
//! | 3 | `a wave reduction over both 32-lane halves (wave64 on a 32-lane host)` | 12 | `ACROSS` in `Lower::shape` (`lower.rs`), with `Lower::reduce_across` (`halves.rs`) as the lowering it guards |
//! | 4 | `a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)` | 8 | `halves::FIRST_ACROSS` (`halves.rs`), raised in the `v_readfirstlane_b32` arm of `Lower::integer` (`lower.rs`) when `across_here()` is false |
//! | 5 | `a lane read that may cross 32-lane halves (wave64 on a 32-lane host)` | 3 | literal in the `v_readlane_b32` arm of `Lower::integer` (`lower.rs`), guarding `Lower::read_lane_across` (`halves.rs`) |
//! | 6 | `a lane mask read as data` | 2 | `LANE_MASK` (`lower.rs`), raised by `Lower::mask_dword` and the scalar-mask write path in `Lower::scalar` (`lower.rs`) |
//! | 7 | `screen-probe ray activation skipped without rays (VICTORY_BVH_NO_HIT=1)` | 1 | `victory-gpu/src/pipeline.rs`, not a wave64 site: the `VICTORY_BVH_NO_HIT` diagnostic switch |
//! | 8 | `a descriptor read only as data` | 1 | `Lower::view_addresses` (`lower.rs`), not a wave64 site: a root signature/descriptor shape the emitter cannot express |
//!
//! Total: 54 + 30 + 12 + 8 + 3 + 2 + 1 + 1 = **111**, the whole compute
//! reject set of that run. Buckets 1-5 (107 shaders) are wave64-on-a-32-lane
//! host refusals; bucket 6 is the wave-size-independent form of bucket 1;
//! buckets 7-8 are unrelated to wave lowering and stay as they are.
//!
//! # What the wave lowering already proves
//!
//! Buckets 3-5 are admitted where every lane of the group runs the operation
//! together, the same number of times, so a group barrier can pass values
//! between the two host halves:
//!
//! * cross-half ballots and reductions go through `halves::exchange_halves`
//!   only in a block `barriers::in_step` marks as run by every lane;
//! * a first-lane or lane read of a console value goes through
//!   `Lower::first_lane_across` / `read_lane_across` in such a block;
//! * a one-wave group of 33-64 threads gets its LDS barriers from
//!   `barriers::split_wave_barriers` in those same blocks, whenever the host's
//!   waves are narrower than the console's: `host_wave32` (the host refuses
//!   `WaveSize(64)`, so the group is two host waves) or `narrow_waves`
//!   (D3DMetal runs a wave64 program in 32-lane waves whatever it declares).
//!
//! `barriers::in_step_one_wave` extends the `step` proof for a group that is
//! one console wave: a first-lane read of a value `halves::lane_uniform`
//! proved the same in every lane, or a lane read of such a value or of a
//! constant console lane, is one value for the whole wave. Such a scalar may
//! then be a loop condition, which is what makes the loop a `step` block and
//! admits the exchanges and LDS barriers inside it. The admission requires the
//! defining block to be `step` and the instruction to be unguarded or guarded
//! only by scalars already proven alike.
//!
//! # What still refuses
//!
//! * a ballot, reduction, prefix, first-lane or lane read whose block is not a
//!   `step` block (a per-lane branch, a loop on a lane-dependent count), and a
//!   lane mask read as data whose write **and** read blocks are not `step`:
//!   the read in a block every lane runs after reconvergence is admitted, the
//!   read inside the per-lane region is not;
//! * a first-lane or lane read of a per-lane value in a block every lane of
//!   both halves does not run together (a per-lane branch, a loop on a count
//!   the halves need not share): each half would read its own first lane, and
//!   no barrier can pass one between them;
//! * LDS of a one-wave group with no barrier that can go where a lane reads
//!   another's write (a divergent write/read, or a loop no `step` proof
//!   covers);
//! * a group of several console waves whose loop count is one value per
//!   console wave: `in_step` (not `in_step_one_wave`) must not admit it, or
//!   the waves would meet at different barrier counts.
//!
//! # Buckets in priority order
//!
//! 1. Lane-mask reads (56) and reductions/first-lanes/lane reads (23): the
//!    `in_step_one_wave` wave-uniform loop proof. Widening it to guards on
//!    alike scalars admits more `step` loops without weakening the
//!    lane-dependent guard refusal. (Done for the masks in `WAVE64-LANE-MASK`;
//!    the guard rule itself is `barriers::in_step_impl`'s `guard_alike`, and
//!    the reductions, first-lane and lane reads use it too — see "Buckets
//!    2-5" below.)
//! 2. LDS in one-wave 33-64 groups (30): the same `step` proof feeds
//!    `split_wave_barriers`; no separate lowering change.
//! 3. A lane read of a provably wave-uniform VGPR needs no exchange at all:
//!    every lane already holds the value it would read. (Done:
//!    `Lower::read_lane`'s `source_uniform` and the first-lane arm.)
//!
//! These are the bounded changes this task makes; the unrelated buckets 7-8
//! are left for their own tasks.
//!
//! # Lane masks read as data (buckets 1 and 6: 56 of 111)
//!
//! `Lower::mask_dword` (`lower.rs`) gives a mask's number. Where the host
//! halves a console wave (`halves`), the ballot of the lanes' bits is taken
//! into the two 32-lane host halves by `Lower::ballot_across_halves` ->
//! `Lower::exchange_halves`: one group-shared `uint32` slot per half of the
//! console wave, written and read between two `barrier` calls with
//! `mem_flags::mem_threadgroup` (`1 | 8`), both dwords recombined before the
//! read. Such a barrier needs every lane of the group, so it may go only in a
//! block `barriers::in_step` proves every lane runs the same number of times.
//!
//! Admitted without any exchange:
//!
//! | mask | why its number needs no ballot |
//! |------|-------------------------------|
//! | a literal lane mask (`Lower::literal_mask`: none, all lanes, or one `Atom::Constant`) | the register holds the same 64-bit number in every lane |
//! | the live mask of a full console wave (`Bool::TRUE`, `Lower::full_console_waves`) | every one of the 64 lanes of every wave is dispatched and active, so the number is all 64 bits |
//!
//! Admitted with the halves' exchange, at a block every lane runs the same
//! number of times:
//!
//! | mask | where the ballot goes |
//! |------|------------------------|
//! | a mask written in such a block (the console compiler's place) | the block that writes it (`Lower::across_at`) |
//! | a pure-mask phi at a join every lane reaches | the join |
//! | a mask whose write is a per-lane region or a join of one, read in a block every lane runs after reconvergence (`Lower::across_here`) | the block the read starts in (bucket 1's 54: the closing shape) |
//!
//! Still refused, with the reason unchanged:
//!
//! | mask | reason |
//! |------|--------|
//! | read inside the per-lane region that defines it, or in a loop whose count differs between console waves | `a lane mask read as data (wave64 on a 32-lane host)` (`MASK_ACROSS`): no block both halves meet in |
//! | in `other` and not a literal (an opaque value, a join, a mask structure resolved only at its use) | `a lane mask read as data` (`LANE_MASK`) |
//! | a register joining a lane mask and data, read where no ballot can be placed | `a register joining a lane mask and data, read as data` |
//! | a mask whose lanes depend on its own value | `a lane mask whose lanes depend on its own value` |
//!
//! The census cannot be re-counted here (no title shader bytes or log
//! contents are committed), so which of the 54 the closing shape covers is
//! not claimed: the CPU table (`lower/halves.rs`, `lane_mask_tests`) and the
//! Metal proof (`victory-metal/tests/wave_lower_gpu.rs`) cover each shape
//! above.
//!
//! # Buckets 2-5: LDS, reductions, first-lane and lane reads (53 of 111)
//!
//! The other four wave64-on-a-32-lane-host buckets (30 + 12 + 8 + 3) are
//! admitted the same way the masks are: only in a block `barriers::in_step`
//! proves every lane of the group runs, or — for a group that is one console
//! wave — `barriers::in_step_one_wave` proves every lane of both host halves
//! runs, the same number of times. A group barrier, or the two group-shared
//! dwords an exchange writes and reads, can go nowhere else. What each bucket
//! admits, and what keeps its reason:
//!
//! | # | census | admitted now | still refused, reason unchanged |
//! |---|-------:|--------------|--------------------------------|
//! | 2 | 30 | LDS of a one-wave 33-64 group: a group barrier (`dx.op.barrier` with `mem_flags::mem_threadgroup`, mode 9) before every read of what another lane may have written and every write over what another lane may yet read, in `step` blocks, plus one at the end of such a block where the code after it reads or overwrites LDS (`barriers::split_wave_barriers`) | a read of what another lane may have written in a block neither proof covers (the call returns `None`, `Lower::halves_lds` says `LDS one console wave64 shares without barriers, on a host of 32-lane waves`): a per-lane branch, or a loop whose count is per-lane or per-console-wave |
//! | 3 | 12 | `WaveActiveSum`/`Min`/`Max`/`BitOr`/`BitAnd`/`BitXor`/`WavePrefixSum` over both halves: each host half's own operation, the 2-dword exchange, a barrier, then the combine (`Lower::reduce_across`: `+`, `|`, `&`, `^`, `min`/`max` unsigned or signed, `min`/`max` f32; a prefix adds the lower half's total in the upper one) | a reduction in a per-lane block or a loop on a count the halves need not share, an f16 (`Shape::Half`) reduction, and a mesh shader's: `a wave reduction over both 32-lane halves (wave64 on a 32-lane host)` |
//! | 4 | 8 | a first-lane read of a wave-uniform value (`halves::lane_uniform`) with no exchange (every lane already holds it); of any other value at a `step` block as the lower half's first lane, exchanged (`Lower::first_lane_across`) | a first-lane read of a per-lane value where no such block exists: `halves::FIRST_ACROSS` |
//! | 5 | 3 | a lane read of a wave-uniform value with no exchange; of a constant console lane or of a lane index that is one value for the whole group (a scalar, or a VGPR `halves::lane_uniform` proved) through the per-half read and exchange (`Lower::read_lane_across`), at a `step` block | a lane read of a per-lane value or at a per-lane lane index (one dword per half cannot name 32 lanes) outside such a block: `a lane read that may cross 32-lane halves (wave64 on a 32-lane host)` |
//!
//! ## The one-wave loop guard
//!
//! The proof above turns on `Structured::guards`: an instruction run under an
//! EXEC narrowing is alike in both halves only if every lane enters or none
//! does. That used to mean no guard at all. It now means a guard whose
//! condition is over scalars already proved alike (`barriers::in_step_impl`'s
//! `guard_alike`): an `s_cselect_b64` of a uniform compare narrowing EXEC, say,
//! enters in both halves together, so a first-lane read of a wave-uniform
//! value, or a lane read into a scalar, under it is still one value for the
//! console wave — and a loop on that scalar still runs as many times in each
//! half, so its LDS and its reductions get their barriers. A guard over a lane
//! mask, a per-lane value or a scalar not shown alike leaves the block out of
//! the proof, and everything in it keeps refusing with the reason it had.
//! (`in_step_impl` keeps the guard out of the *mask-as-data* proof entirely:
//! a ballot is `Lower::mask_dword`'s business, not this one's.)
//!
//! ## What the census cannot say
//!
//! The counts are of title shaders, not of the shapes inside them, and the
//! shaders and the log are not committed: this table records which shapes each
//! bucket's change admits, not how many of the 30/12/8/3 fall in them. The 30
//! are exactly the shaders where `split_wave_barriers` returned `None` (a read
//! in a block the one-wave proof did not cover); which of those were loops on
//! a guard this task now proves is not claimable here. The CPU table
//! (`lower/halves.rs`, `cross_half_tests`) and the Metal proof
//! (`victory-metal/tests/wave_lower_gpu.rs`, the LDS, reduction and
//! first-lane tests) cover each shape above, admitted and refused.
//!
//! # Re-census on a synthetic corpus
//!
//! The `105/111` in `.swarm/reports/architect-sync-04.md` cannot be
//! reproduced from this tree: it is not a count any test or report here
//! derives, and the two wave64 tasks merged before it (`f71b0b3d`,
//! `d42b7694`) explicitly declined to re-count (their reports say the census
//! "cannot be re-attributed" without the title's bytes). The title's shaders
//! and the log are still not in the repository, so the title census cannot be
//! re-counted either. It is re-counted on a synthetic corpus of the shapes
//! each bucket is made of
//! (`lower::halves::cross_half_tests::wave_corpus_tests`): 35 programs, of
//! which 13 are lane masks, 7 LDS, 7 reductions, 4 first lanes and 4 lane
//! reads. These are counts of that corpus, **not** of the 111 title shaders.
//!
//! | configuration | refused | by bucket |
//! |---|---|---|
//! | `narrow_waves` + `host_wave32` (a Wine/D3DMetal run whose device query says 32) | 11/35 | masks 3, LDS 2, reductions 2, first lanes 2, lane reads 2 |
//! | `host_wave32` only (NVIDIA, Intel: the host refuses `WaveSize(64)`) | 11/35 | the same 11 |
//! | `narrow_waves` only (halves off) | 1/35 | the lowering translates 34, with no halves |
//!
//! So on the configuration the historical refusals came from, the shapes that
//! close the census buckets admit; the 11 that keep refusing are the divergent
//! ones listed above. The census reason strings themselves prove the halves
//! lowering ran in the title's run: `MASK_ACROSS`, `FIRST_ACROSS` and the lane
//! read's reason are only raised in halves.
//!
//! ## The wiring defect: the LDS barriers were keyed to a switch
//!
//! `Lower::new` (`lower.rs`) asked for `split_wave_barriers` only under
//! `narrow_waves`, and computed `fits` the same way. `narrow_waves` is the
//! D3DMetal workaround ("the host runs a wave64 program in 32-lane waves
//! whatever `WaveSize` says"); `host_wave32` is the device fact ("the host
//! refuses `WaveSize(64)`"). On a genuinely 32-lane host (NVIDIA, Intel) the
//! barrier pass was therefore never run, and every one-wave 33-64 group with
//! LDS was refused with bucket 2's reason although the same barriers are
//! sound and needed there: 7 of the corpus' 35 programs, the shape of the
//! 30 title shaders in that bucket. Keying the pass to
//! `native.wave64 && (narrow_waves || host_wave32)` admits them: the corpus'
//! wave32 configuration drops from 16 to 11 refusals and emits, for every LDS
//! program it now admits, byte-identical DXIL to the configuration the GPU
//! fixtures ran (`wave_corpus_tests` asserts that identity).
//!
//! ## The flag that is still allowed to disagree
//!
//! `narrow_waves` alone (`host_wave32` false) translates 34 of the 35 corpus
//! programs, ballots and lane reads included, taking only a 32-lane host
//! wave's mask bits (the high dword is 0) while asserting the host runs 32-lane
//! waves: that configuration lowers as if the host ran 64. It is reachable in
//! production because `victory-gpu`'s `host_wave32()` returns false when
//! `CheckFeatureSupport` fails (`host.rs` logs "host wave lanes not known" and
//! answers false) while `narrow_waves()` still answers true under Wine. Making
//! `narrow_waves` imply halves is the sound fix, but it changes every
//! D3DMetal-translated wave64 shader from "declare 64" to "declare 32 with
//! exchanges" and the `halves_tests` three-way table, so it is left as a
//! decision for the orchestrator/architect, not a side effect of this task.
//!
//! ## What still refuses, and why it must
//!
//! | # | shape | why no barrier can go |
//! |---|-------|-----------------------|
//! | 1 | a mask read, reduction, first-lane or lane read in a per-lane region or a loop on a per-lane/per-console-wave count | the two host halves need not run it, or run it a different number of times, so a group barrier there would hang or read the wrong pass's values |
//! | 2 | a lane mask written and read in the same per-lane region | the ballot covers both halves' bits, so it needs a block every lane reaches |
//! | 3 | a lane read with a per-lane lane index | the 2-dword exchange holds one value per half; N lanes asking for N different console lanes need a full 64-lane permutation, which one dword per half cannot carry |
//!
//! Each of these has a negative test in `lower/halves.rs`
//! (`cross_half_tests::wave_corpus_tests` for both option sets, and the two
//! admit/refuse tables) asserting the exact reason string it keeps.
