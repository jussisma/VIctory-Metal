//! Persistent `MTLBinaryArchive` pipeline cache (ADR-014).
//!
//! Reserved by the orchestrator. This module owns serialization of the device
//! binary archive to a content-keyed on-disk location and its reuse on warm
//! runs so pipeline and shader compilation is skipped.
