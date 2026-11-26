//! Guest D3D12 descriptor and root-binding resolution for the native backend.
//!
//! Reserved by the orchestrator so the descriptor-resolution task owns this
//! file without touching `native_metal/mod.rs`. The contract (ADR-010):
//!
//! * a guest descriptor-table handle resolves to
//!   `(allocation_id, generation, byte_offset, size)` for every descriptor the
//!   table covers, through `xmem::share`;
//! * a root CBV/SRV/UAV GPU virtual address resolves the same way;
//! * resolved ranges populate the Wire v3 root/descriptor bindings;
//! * every referenced backing range stays registered (fence-pinned) until the
//!   GPU completion that retires the submission.

/// Why a guest binding could not be resolved into a native range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The guest address or descriptor handle is zero or malformed.
    Invalid(&'static str),
    /// No shared `xmem` range covers the requested bytes.
    Unmapped { address: u64, len: u64 },
    /// A descriptor entry names a handle this backend cannot express.
    Unsupported(&'static str),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid guest binding: {why}"),
            Self::Unmapped { address, len } => {
                write!(f, "no shared xmem range for 0x{address:x}+{len}")
            }
            Self::Unsupported(why) => write!(f, "unsupported guest binding: {why}"),
        }
    }
}

impl std::error::Error for ResolveError {}
