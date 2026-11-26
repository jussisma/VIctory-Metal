//! Batched D3D12 command-list ring: per-list identity, ordering and
//! exactly-once delivery.
//!
//! The guest records a whole command list (draws, dispatches, resource
//! barriers, pipeline and root-argument state) and publishes it as contiguous
//! bytes in a shared-memory ring rather than one IPC round-trip per draw. The
//! host drains the ring and executes each list exactly once.
//!
//! Invariants (see `docs/ARM64-RENDERER.md` and `.swarm/decisions.json`):
//!
//! - A list is identified by `(list_id, generation)`. Execution is keyed by
//!   that pair plus the submission sequence, so a replay or retry of a list
//!   that already executed is detected and refused instead of re-executed.
//! - Publication is atomic: a record is visible to the host only after its
//!   length and checksum are committed, and the host never executes a record
//!   whose bytes fail validation.
//! - Wrapping never overwrites a record the host has not drained; the
//!   producer stops with an explicit backpressure error instead of corrupting
//!   an in-flight list.
//! - An uncertain native execution is reported as an error; the path never
//!   falls back to a second execution of the same list.
//!
//! This module is transport- and Metal-independent so its tests run without a
//! GPU. `wire` owns the per-command payload encoding; the ring owns framing,
//! ordering and the exactly-once ledger.
