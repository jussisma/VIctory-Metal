//! Host texture and pipeline registry (ADR-013).
//!
//! Reserved by the orchestrator. This module owns host-side registration of
//! Metal textures (2D/2DArray/Cube/3D, mipmaps, BC1-BC7, depth/stencil,
//! R11G11B10F, R10G10B10A2 and RTV/DSV/SRV/UAV views) and content-hashed
//! graphics/mesh/compute pipeline states created from converted metallibs.
//! It never owns a transport or a session.
