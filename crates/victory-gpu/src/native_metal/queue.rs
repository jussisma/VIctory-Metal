//! Native queue submission and present under `VICTORY_BACKEND=metal`.
//!
//! Reserved by the orchestrator so the queue/present task owns this file
//! without touching `native_metal/mod.rs`. The contract (ADR-010):
//!
//! * `ExecuteCommandLists` under `VICTORY_BACKEND=metal` takes the Wire v3
//!   submissions a closed title list produced and carries them over IPC in
//!   recorded order;
//! * `Present` hands the produced surface to the host `NativeSwapchain`
//!   (IOSurface) instead of the D3D12 swapchain;
//! * neither path falls back to D3D12/D3DMetal after native submission starts:
//!   an unsupported opcode is an explicit error.

/// Why a native queue submission or present was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeQueueError {
    /// No submission carried the list identity the queue was asked to run.
    Empty(&'static str),
    /// The IPC channel is not connected or refused the batch.
    Transport(String),
    /// The host rejected the batch; the message is the host's own.
    Refused(String),
    /// A present was asked for a surface the native swapchain does not own.
    NoSurface(u64),
}

impl std::fmt::Display for NativeQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty(why) => write!(f, "native queue submission empty: {why}"),
            Self::Transport(why) => write!(f, "native queue transport: {why}"),
            Self::Refused(why) => write!(f, "native queue refused: {why}"),
            Self::NoSurface(id) => write!(f, "native swapchain does not own surface {id}"),
        }
    }
}

impl std::error::Error for NativeQueueError {}
