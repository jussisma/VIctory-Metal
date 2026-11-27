//! Native Metal pipeline and encoder state for the D3D12 core render states.
//!
//! Owner module reserved by the orchestrator for Cycle 06 (ADR-026): blend
//! state, depth-stencil state (including `OMSetStencilRef`/`OMSetBlendFactor`),
//! rasterizer state (cull mode, winding, fill mode, depth bias), render-target
//! and depth clears, and pitched upload-to-private texture copies.
//!
//! Nothing here falls back to D3D12 or D3DMetal. An unrepresentable state is an
//! explicit error, never a silent default.
