//! Shared-memory GPU completion fence page (ADR-012).
//!
//! Reserved by the orchestrator. This module owns the transport-agnostic
//! `AtomicU64` completion fence that an ARM64 Metal host publishes from a
//! command-buffer completion handler and a guest reads with one atomic load
//! (`GetCompletedValue`), with no IPC round-trip and no Wine dependency.
