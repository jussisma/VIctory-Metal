//! Persistent `MTLResidencySet` bindless residency (ADR-013).
//!
//! Reserved by the orchestrator. This module owns attachment of live shared
//! Metal buffers and registered textures to one persistent residency set bound
//! to the command queue, with an explicit `useResources` fallback when the
//! runtime does not expose `MTLResidencySet`.
