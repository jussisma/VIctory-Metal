//! Texture region copies and subresource resolves (ADR-015).
//!
//! Reserved by the orchestrator. This module owns `copyFromTexture` with an
//! explicit source/destination subresource plus origin/size box, and
//! `resolveSubresource` (multisample resolve into a single-sample target).
