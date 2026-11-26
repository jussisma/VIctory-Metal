//! GPU-driven indirect command encoding (ADR-015).
//!
//! Reserved by the orchestrator. This module owns Metal encoding of indirect
//! draws (`drawIndirect`, `drawIndexedIndirect`) and indirect mesh/compute
//! dispatch (`dispatchMeshThreadgroupsWithIndirectBuffer`,
//! `dispatchThreadgroupsWithIndirectBuffer`) including the optional count
//! buffer. Argument buffers stay GPU-resident: no CPU readback.
