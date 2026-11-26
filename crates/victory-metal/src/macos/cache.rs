//! Persistent disk cache for DXIL to Metal library translation.
//!
//! A translation result is keyed by content and layout, never by file name or
//! pipeline order:
//!
//! - the DXIL byte hash,
//! - the root signature / root layout bytes or hash,
//! - the entry-point name and stage,
//! - the IRConverter version and the translation options that can change
//!   output.
//!
//! The cache stores the serialized metallib plus the reflection payload the
//! binder needs, writes through a temporary file and a rename so a crash
//! never leaves a partial entry, and treats an unreadable or mismatched entry
//! as a miss rather than a failure.
