//! Metal Shader Converter root-signature ABI bindings (ADR-006 scope).
//!
//! Owns the mapping from a translated DXIL root signature to the argument-buffer
//! layout the converter emits (top-level argument buffer, root constants, CBV /
//! SRV / UAV / Sampler tables) and the encoders that bind it on an
//! `MTLRenderCommandEncoder` / `MTLComputeCommandEncoder`.
//!
//! This module is reserved by the orchestrator so wire, encoder and shader
//! tasks never contend on `macos/mod.rs`. The root-bindings task implements it.
