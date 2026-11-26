//! Native resource views: formats, depth/stencil surfaces and descriptor
//! binding for the ARM64 Metal renderer.
//!
//! Scope of this module:
//!
//! - The guest format to `MTLPixelFormat` mapping, including the depth and
//!   stencil formats, with typed, sRGB and compressed families covered and an
//!   explicit error for formats Metal cannot represent.
//! - Private 2D colour and depth/stencil texture creation, sampled and
//!   storage usage flags, mip and array levels.
//! - The host-side descriptor shape for SRV/UAV/RTV/DSV use: format, view
//!   dimension, first mip, mip count, first slice, slice count and the
//!   resource handle they resolve against.
//! - Upload paths: zero-copy when the guest range is already page-aligned and
//!   shareable, otherwise a GPU blit from a staging buffer.
//!
//! Metal FFI specific to textures and depth states lives here so `objc.rs`
//! stays a shared, stable helper module.
