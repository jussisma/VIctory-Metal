//! Guest-side D3D12 render-state tracking for the native Metal backend.
//!
//! Owner module reserved by the orchestrator for Cycle 06 (ADR-026): blend,
//! depth-stencil, rasterizer, stencil reference and blend factor state recorded
//! by the `victory-gpu` D3D12 API path and forwarded to the host over Wire v3.
//!
//! The producer must never fall back to D3D12 or D3DMetal after native
//! submission starts: an unsupported state returns an explicit error.
