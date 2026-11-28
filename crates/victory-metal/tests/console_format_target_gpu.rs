//! Boot #27's console depth target on the real Apple GPU.
//!
//! The guest refuses texel format 118 (the console's 16-bit depth with an
//! 8-bit stencil plane) as "format has no host code" unless the registration
//! path applies the `pc_format` console mapping first. That mapping names
//! `DXGI_FORMAT_D32_FLOAT_S8X24_UINT` (DXGI 20), whose kind 8 wire code is 8;
//! this test proves the host end of that mapping is creatable at the exact
//! Boot #27 shape and that the other target usage is refused by name.
//!
//! What this proves on hardware:
//!
//! 1. Wire code 8, [`TextureFormat::from_wire_code`], is
//!    [`TextureFormat::Depth32FloatStencil8`] and is DXGI 20 — the value
//!    `victory_gpu`'s `pc_format` maps console 118 to.
//! 2. A 1024x1024 depth target in that format registers: `TextureRegistry`
//!    re-reads the created `MTLTexture`'s type, pixel format, extents, mip
//!    count, sample count and array length from Metal, so success is a real
//!    object, not bookkeeping.
//! 3. A DSV view of it is created.
//! 4. The same format registered with colour-attachment usage is refused with
//!    the exact reason, never approximated to a colour format.
//!
//! The exact depth clear/read is a documented blocker, asserted below: the
//! public registry API has only `clear_color` (RTV views) and
//! `read_view_rgba8` (RGBA8/BGRA8 views), so a depth-stencil target cannot be
//! cleared or read back through it today. The test prints `PARTIAL` with that
//! gap instead of claiming an unproven pixel value.
//!
//! Run (serialised with the GPU lock):
//! ```text
//! .swarm/bin/gpu-lock -- sh -c 'export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!   cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!     --test console_format_target_gpu -- --ignored --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use victory_metal::MetalTransport;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::registry::{
    TextureDesc, TextureDimension, TextureFormat, TextureRegistry, TextureSubresource,
    TextureUsage, TextureViewKind,
};

/// The Boot #27 shapes: 1024x1024 depth (the shadow-map target) and the
/// 2560x1440 depth target that is also unordered.
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

fn target(format: TextureFormat, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        label: Some("console-118-target".to_string()),
    }
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn the_console_depth_stencil_target_registers_on_hardware() {
    // The guest's end of the mapping (victory-gpu unit test
    // `console_depth_stencil_targets_register_with_the_mapped_pc_format` sends
    // wire code 8 for console 118).
    let mapped = TextureFormat::from_wire_code(8).expect("wire code 8 is in the host table");
    assert_eq!(mapped, TextureFormat::Depth32FloatStencil8);
    assert_eq!(mapped.dxgi(), Some(20), "the PC format pc_format names");
    assert_eq!(
        TextureFormat::from_dxgi(20).expect("DXGI 20"),
        mapped,
        "wire code 8 and DXGI 20 are the same row"
    );

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let mut registry = TextureRegistry::new(&transport).expect("texture registry");

    registry
        .register(1, target(mapped, TextureUsage::depth_attachment()))
        .expect("the mapped format must create at the Boot #27 shape");
    let entry = registry.get(1).expect("the target is registered");
    assert_eq!(entry.desc().format, mapped);
    assert_eq!((entry.desc().width, entry.desc().height), (WIDTH, HEIGHT));
    assert_eq!(entry.desc().mip_levels, 1);
    assert_eq!(entry.desc().array_layers, 1);

    let dsv = registry
        .create_view(1, TextureViewKind::Dsv, TextureSubresource::single(0, 0))
        .expect("a depth/stencil view of the target");
    assert_eq!(dsv.format(), mapped);
    assert_eq!(dsv.kind(), TextureViewKind::Dsv);

    // The colour usage a console 118 render target could also name is the
    // host's explicit refusal, not a silent colour format.
    let error = registry
        .register(2, target(mapped, TextureUsage::color_attachment()))
        .expect_err("a depth format must never be registered as a colour target");
    assert_eq!(
        error.to_string(),
        "invalid texture descriptor: a depth format cannot be registered as a colour render target",
        "{error}"
    );

    // The clear/readback blocker, as observed from the public API: neither
    // clear nor readback accepts a depth view.
    let error = registry
        .clear_color(&dsv, [0.0, 0.0, 0.0, 1.0])
        .expect_err("clear_color is RTV-only");
    assert_eq!(
        error.to_string(),
        "unsupported operation: clear_color requires an RTV view",
        "{error}"
    );
    let error = registry
        .read_view_rgba8(&dsv)
        .expect_err("read_view_rgba8 is 8-bit colour only");
    assert_eq!(
        error.to_string(),
        "unsupported operation: readback requires an RGBA8 or BGRA8 view",
        "{error}"
    );

    registry.destroy(1).expect("destroy the target");
    assert!(registry.is_empty());

    eprintln!(
        "PARTIAL: {} wire 8 = {:?} (DXGI {}) registered at {WIDTH}x{HEIGHT} with a DSV view; \
         colour usage refused with its exact reason; the exact depth clear/read cannot be \
         encoded: TextureRegistry exposes only clear_color (RTV) and read_view_rgba8 \
         (RGBA8/BGRA8), both refused for a depth-stencil view (proved above) on {}",
        mapped.name(),
        mapped,
        mapped.dxgi().unwrap_or(0),
        info.name_str(),
    );
}
