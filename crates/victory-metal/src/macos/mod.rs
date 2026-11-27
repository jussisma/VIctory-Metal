//! macOS-native Metal, IRConverter, and C ABI implementations.

pub mod binary_archive;
pub mod bindings;
pub mod blit;
pub mod c_api;
pub mod cache;
pub mod device;
pub mod indirect;
pub mod irconverter;
pub mod msc;
pub mod mesh;
pub(crate) mod objc;
pub mod present;
pub mod registry;
pub mod render;
pub mod residency;
pub mod retire;
pub mod texture;
pub mod views;
pub mod window;
