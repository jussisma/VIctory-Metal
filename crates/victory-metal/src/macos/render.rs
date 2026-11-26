//! ARM64 Metal 3D render encoding: render passes, graphics pipelines and
//! draw submission.
//!
//! Scope of this module:
//!
//! - `MTLRenderPassDescriptor` with colour and depth/stencil attachments,
//!   load/store actions and clear values.
//! - A render command encoder that binds a graphics pipeline state, vertex
//!   and index buffers, viewport and scissor, root constants and
//!   descriptor-table ranges, then issues indexed and non-indexed draws.
//! - One command buffer per recorded list with an explicit completion and
//!   error path; no implicit CPU readback and no double execution.
//!
//! The Metal FFI lives here (or in private helpers of this module) so the
//! module can grow without editing `objc.rs`, which other modules share.
