//! Bounded multi-process Rosetta IRConverter worker pool (ADR-014).
//!
//! Reserved by the orchestrator. This module owns the bounded pool of
//! `victory-irconverter` worker processes (size from `VICTORY_SHADER_WORKERS`),
//! its job/result protocol and its content-hashed disk cache lookups. It must
//! not own the Metal device or any transport.
