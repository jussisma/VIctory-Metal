//! Per-frame telemetry for the native Metal backend (ADR-016).
//!
//! Reserved by the orchestrator. This module owns guest-side frame accounting:
//! record microseconds, ring submit microseconds, frame time percentiles
//! (p50/p95/p99), FPS, the unsupported-opcode histogram and the strict
//! `d3dmetal_fallback_count == 0` guard.
