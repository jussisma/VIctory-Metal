//! Swapchain present over the native ARM64 Metal presenter.
//!
//! Implements the `VICTORY_BACKEND=metal` present path: the swapchain's
//! back-buffer becomes a host-visible IOSurface presented by
//! `victory_metal::macos::present`, with no D3DMetal call and no CPU readback.
//! Implemented by the SWAPCHAIN-NATIVE-PRESENT task.
