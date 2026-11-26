//! Native ARM64 presentation of rendered frames, without D3DMetal and
//! without a CPU readback in the present path.
//!
//! Scope of this module:
//!
//! - A `PresentSink` abstraction the host can drive from a headless test and
//!   from a window-backed host, so presentation has one code path.
//! - An IOSurface-backed swapchain image: the renderer draws into the
//!   surface's `MTLTexture`, and handing the surface to the sink is the only
//!   cross-process step.
//! - Frame identity and pacing: monotonic frame index, one in-flight present
//!   at a time, explicit error when the sink refuses a frame.
//!
//! Reading pixels back is a test-only verification step, never part of the
//! production present path.
