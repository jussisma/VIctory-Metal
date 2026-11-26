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
//! | 1 | `a lane mask read as data (wave64 on a 32-lane host)` | 54 | `super::lower::halves::MASK_ACROSS` (`halves.rs`), raised in `Lower::mask_dword` (`lower.rs`, `across_at(def).ok_or(MASK_ACROSS)`) when the ballot's defining block is not `in_step` |
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
//!   `barriers::split_wave_barriers` in those same blocks.
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
//!   `step` block (a per-lane branch, a loop on a lane-dependent count);
//! * a first-lane read of a value that differs between the halves (`v0`, a
//!   lane-dependent mask) even in a `step` block, because each half would read
//!   its own first lane;
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
//!    lane-dependent guard refusal.
//! 2. LDS in one-wave 33-64 groups (30): the same `step` proof feeds
//!    `split_wave_barriers`; no separate lowering change.
//! 3. A lane read of a provably wave-uniform VGPR needs no exchange at all:
//!    every lane already holds the value it would read.
//!
//! These are the bounded changes this task makes; the unrelated buckets 7-8
//! are left for their own tasks.
