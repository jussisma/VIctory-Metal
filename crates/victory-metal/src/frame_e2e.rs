//! End-to-end frame path: a recorded command list produces a frame that is
//! presented through the IOSurface sink, verified from the surface itself.
//!
//! This module composes the pieces that each have their own unit tests
//! (`wire`, `cmdlist`, `macos::render`, `macos::present`, `macos::views`) into
//! one frame, so a regression at a seam fails here rather than in the title.
//!
//! Scope of the gate:
//!
//! - A `wire::Submission` built with `Command::SetPipeline`, `RootConstants`,
//!   `DrawIndexed` (and, where supported, `Draw`) is executed by the device
//!   against a render target.
//! - That target is an IOSurface-backed swapchain image, so the presented
//!   frame is the frame that was drawn, not a copy.
//! - The sink receives a `PresentedFrame` (surface identity, index, time) with
//!   no pixel bytes; the test alone reads the surface once to assert pixels.
//! - Presenting an already-drawn surface must not clear it.
//!
//! The Wine guest half of the gate lives in
//! `crates/victory-hle/src/modules/xmem/e2e.rs` and
//! `tools/macos/metal-poc/synthetic-frame.sh`; this module proves the host
//! half on the real GPU without Wine.
