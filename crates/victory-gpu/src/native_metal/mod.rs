//! Native ARM64 Metal backend producer (`VICTORY_BACKEND=metal`).
//!
//! Reserved by the orchestrator so the producer and the native presenter own
//! separate files. `producer` translates recorded D3D12 command-list state into
//! version 3 wire submissions; `present` connects the swapchain present path to
//! the host's IOSurface presenter without D3DMetal and without CPU readback.
//!
//! Neither module may fall back to D3D12/D3DMetal after native submission
//! starts: an unimplemented opcode returns an explicit error (ADR-008).

pub mod bindings;
pub mod client;
pub mod present;
pub mod producer;
pub mod queue;
pub mod telemetry;
