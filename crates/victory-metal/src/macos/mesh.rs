//! Native Metal mesh pipeline and mesh dispatch.
//!
//! Reserved by the orchestrator so the mesh-pipeline task owns this file
//! without touching `macos/mod.rs`. The public contract this module must
//! satisfy (ADR-009):
//!
//! * create an `MTLMeshRenderPipelineDescriptor` with `meshFunction`,
//!   `fragmentFunction`, an optional `objectFunction`,
//!   `maxTotalThreadsPerMeshThreadgroup`, `payloadMemoryLength` and the color /
//!   depth attachment pixel formats of the pass it draws into;
//! * create an `MTLRenderPipelineState` from it and give the device a native
//!   `SetMeshPipeline` command;
//! * encode `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
//!
//! Nothing here falls back to `MTLRenderPipelineDescriptor` or to D3DMetal:
//! an unsupported request returns an explicit error.

/// Why a mesh pipeline request could not be satisfied on this device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// The host OS or device does not offer mesh pipeline objects.
    Unsupported(&'static str),
    /// One of the pipeline's declared budgets is outside the device limits.
    Budget(&'static str),
    /// Metal refused to create the pipeline; the message is Metal's own.
    Create(String),
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(why) => write!(f, "mesh pipeline unsupported: {why}"),
            Self::Budget(why) => write!(f, "mesh pipeline budget: {why}"),
            Self::Create(why) => write!(f, "mesh pipeline creation failed: {why}"),
        }
    }
}

impl std::error::Error for MeshError {}
