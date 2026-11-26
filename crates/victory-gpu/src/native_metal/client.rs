//! Live Wire v3 client transport for the guest (ADR-012).
//!
//! Reserved by the orchestrator. This module owns the guest-side implementation
//! of [`crate::native_metal::queue::V3Sink`]: authenticated connect, the ring
//! submission of already-encoded v3 frames, and a zero-IPC completion-fence
//! read. It must never fall back to D3D12/D3DMetal.
