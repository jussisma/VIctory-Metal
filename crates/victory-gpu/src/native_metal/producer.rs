//! D3D12 command-list to Wire v3 producer.
//!
//! Implements the `VICTORY_BACKEND=metal` translation of recorded command-list
//! state (render targets, viewports, PSOs, root arguments, vertex/index
//! buffers, draws, dispatches, barriers, copies, present) into
//! `victory_metal::wire_v3` submissions. Implemented by the GPU-PRODUCER-RING
//! task; this file is the only file that task owns besides `list.rs` hook code.
