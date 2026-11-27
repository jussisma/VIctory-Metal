//! Hardware proof for the host resource / PSO registry and residency set.
//!
//! Three `#[ignore]`-gated tests run on the real Apple GPU:
//!
//! 1. [`textures_and_views_are_real_on_hardware`] registers a BC1 sample, an
//!    RGBA8 colour target and a Depth32Float target, creates RTV/SRV/DSV views,
//!    clears the RTV through a render pass and reads the exact pixel back
//!    through the SRV with a blit. The pixel proves the registry created real
//!    `MTLTexture` objects rather than bookkeeping.
//! 2. [`compute_pso_is_content_hashed_and_runs_exact_value`] compiles MSL with
//!    Apple's offline toolchain, registers the metallib through
//!    [`PipelineRegistry::register_compute`], proves duplicate-key reuse (one
//!    Metal object), proves a one-byte delta changes the key, proves an
//!    unparsable metallib is rejected, and runs one dispatch whose exact
//!    values are read back.
//! 3. [`residency_set_binds_allocations_and_keeps_values_exact`] adds a shared
//!    buffer and a registered texture to an [`MTLResidencySet`] when the
//!    runtime exposes one, or asserts the documented `O(n)` fallback and prints
//!    `PARTIAL` with the exact failing selector and `file:line`.
//!
//! Run (serialised with the GPU lock):
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test resource_registry_gpu -- --ignored --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::registry::{
    PipelineKey, PipelineRegistry, PipelineStage, RegistryError, TextureDesc, TextureDimension,
    TextureFormat, TextureRegistry, TextureSubresource, TextureUsage, TextureViewKind,
};
use victory_metal::macos::residency::ResidencySet;

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn free(ptr: *mut c_void);
}

/// Synthetic MSL used by the compute and residency proofs.
const FILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void registry_fill(device uint *out [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {
    out[tid] = 0xA5A50000u + tid;
}
"#;

const FILL_ENTRY: &str = "registry_fill";
const FILL_BASE: u32 = 0xA5A5_0000;
const THREADS: u32 = 64;
const THREADS_PER_GROUP: u32 = 32;
const PAGE: usize = 16 * 1024;

struct PageGuard(*mut c_void);

impl Drop for PageGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
        unsafe { free(self.0) };
    }
}

/// Allocates a page-aligned, page-multiple buffer and registers it with Metal.
fn register_page(transport: &DirectMetalTransport, bytes: usize) -> (PageGuard, VmBufferId) {
    let mut raw: *mut c_void = null_mut();
    // SAFETY: `posix_memalign` writes a valid `PAGE`-aligned allocation.
    let rc = unsafe { posix_memalign(&mut raw, PAGE, bytes) };
    assert_eq!(rc, 0, "posix_memalign failed");
    assert!(!raw.is_null());
    let guard = PageGuard(raw);
    // SAFETY: The allocation is live for `bytes` and outlives the GPU work.
    let id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), bytes) }
        .expect("register_zerocopy should succeed");
    (guard, id)
}

fn fill_pattern(transport: &DirectMetalTransport) -> (PageGuard, VmBufferId, *mut u32) {
    let (guard, id) = register_page(transport, PAGE);
    let words = guard.0.cast::<u32>();
    // SAFETY: The page holds `PAGE / 4` writable u32 words.
    unsafe {
        std::ptr::write_bytes(words, 0, PAGE / 4);
    }
    (guard, id, words)
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path =
            std::env::temp_dir().join(format!("victory-{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tool_path(name: &str) -> Option<PathBuf> {
    let output = Command::new("xcrun").args(["-f", name]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

/// Compiles `FILL_MSL` into a real `.metallib`, or `None` without Xcode tools.
fn compile_fill_metallib(dir: &Path) -> Option<Vec<u8>> {
    let metal = tool_path("metal")?;
    let metallib = tool_path("metallib")?;
    let source = dir.join("registry_fill.metal");
    let air = dir.join("registry_fill.air");
    let library = dir.join("registry_fill.metallib");
    fs::write(&source, FILL_MSL).expect("write MSL source");

    let compiled = Command::new(metal)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&air)
        .output()
        .expect("spawn metal");
    assert!(
        compiled.status.success(),
        "metal -c failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let linked = Command::new(metallib)
        .arg(&air)
        .arg("-o")
        .arg(&library)
        .output()
        .expect("spawn metallib");
    assert!(
        linked.status.success(),
        "metallib failed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    Some(fs::read(&library).expect("read compiled metallib"))
}

fn rgba8_desc(width: u32, height: u32, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        width,
        height,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage,
        label: Some("registry-proof".to_string()),
    }
}

fn depth_desc(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        width,
        height,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::depth_attachment(),
        label: Some("registry-depth".to_string()),
    }
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn textures_and_views_are_real_on_hardware() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let mut registry = TextureRegistry::new(&transport).expect("texture registry");

    // BC1 sample: 8x8 is two full 4x4 blocks.
    let bc1 = TextureDesc {
        width: 8,
        height: 8,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Bc1Rgba,
        usage: TextureUsage::sampled(),
        label: Some("registry-bc1".to_string()),
    };
    registry.register(1, bc1).expect("BC1 texture must create");
    registry
        .register(2, rgba8_desc(8, 8, TextureUsage::color_attachment()))
        .expect("RGBA8 colour target must create");
    registry
        .register(3, depth_desc(8, 8))
        .expect("Depth32Float target must create");

    let rtv = registry
        .create_view(2, TextureViewKind::Rtv, TextureSubresource::single(0, 0))
        .expect("RTV view");
    let srv = registry
        .create_view(2, TextureViewKind::Srv, TextureSubresource::single(0, 0))
        .expect("SRV view");
    let bc1_srv = registry
        .create_view(1, TextureViewKind::Srv, TextureSubresource::single(0, 0))
        .expect("BC1 SRV view");
    let dsv = registry
        .create_view(3, TextureViewKind::Dsv, TextureSubresource::single(0, 0))
        .expect("DSV view");

    assert_eq!(rtv.kind(), TextureViewKind::Rtv);
    assert_eq!(srv.kind(), TextureViewKind::Srv);
    assert_eq!(bc1_srv.format(), TextureFormat::Bc1Rgba);
    assert_eq!(dsv.format(), TextureFormat::Depth32Float);

    // Explicit hardware-side rejections.
    assert!(matches!(
        registry.create_view(2, TextureViewKind::Uav, TextureSubresource::single(0, 0)),
        Err(RegistryError::Unsupported(_))
    ));
    assert!(matches!(
        registry.create_view(2, TextureViewKind::Srv, TextureSubresource::single(1, 0)),
        Err(RegistryError::InvalidSubresource(_))
    ));
    assert!(matches!(
        registry.register(2, rgba8_desc(8, 8, TextureUsage::sampled())),
        Err(RegistryError::DuplicateTexture(2))
    ));

    // Clear to exact 8-bit values, then read the SRV back through a blit.
    registry
        .clear_color(&rtv, [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0])
        .expect("RTV clear must complete");
    let pixels = registry
        .read_view_rgba8(&srv)
        .expect("SRV readback must complete");
    assert_eq!(pixels.len(), 64);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel,
            [64, 128, 192, 255],
            "pixel {index} after RTV clear through the SRV"
        );
    }

    registry.destroy(1).expect("BC1 destroy");
    registry.destroy(2).expect("RGBA8 destroy");
    registry.destroy(3).expect("depth destroy");
    assert!(registry.is_empty());

    eprintln!(
        "texture proof on {}: BC1+Rgba8+D32F registered, RTV/SRV/DSV views created, \
         cleared pixel = {:?} for all {} texels; duplicate/out-of-range/UAV rejects ok",
        info.name_str(),
        pixels[0],
        pixels.len()
    );
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn compute_pso_is_content_hashed_and_runs_exact_value() {
    let dir = TempDir::new("registry-pso");
    let Some(metallib) = compile_fill_metallib(dir.path()) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; PSO proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let mut registry = PipelineRegistry::new(&transport).expect("pipeline registry");

    let state = b"compute-state-v1";
    let key = PipelineKey::from_parts(&metallib, FILL_ENTRY, PipelineStage::Compute, state);
    let first = registry
        .register_compute(key.clone(), &metallib, FILL_ENTRY, state)
        .expect("compute PSO must create");
    let second = registry
        .register_compute(key.clone(), &metallib, FILL_ENTRY, state)
        .expect("duplicate registration must reuse");
    assert_eq!(
        first, second,
        "duplicate key must reuse one pipeline object"
    );
    assert_eq!(registry.created_count(), 1);
    assert_eq!(registry.reuse_count(), 1);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.get(first).unwrap().stage(), PipelineStage::Compute);

    // A one-byte delta in any part changes the key; an unparsable metallib is
    // an explicit error, never a silently substituted entry point.
    let mut corrupt = metallib.clone();
    corrupt[0] ^= 0xFF;
    let corrupt_key = PipelineKey::from_parts(&corrupt, FILL_ENTRY, PipelineStage::Compute, state);
    assert_ne!(key, corrupt_key);
    assert!(matches!(
        registry.register_compute(corrupt_key, &corrupt, FILL_ENTRY, state),
        Err(RegistryError::InvalidMetallib(_))
    ));
    let bad_key = PipelineKey::from_parts(&metallib, "not_a_kernel", PipelineStage::Compute, state);
    assert!(matches!(
        registry.register_compute(bad_key.clone(), &metallib, "not_a_kernel", state),
        Err(RegistryError::EntryPointNotFound(_))
    ));

    let (_guard, buffer, words) = fill_pattern(&transport);
    registry
        .dispatch_compute_1d(&transport, first, &[buffer], THREADS, THREADS_PER_GROUP)
        .expect("registry compute dispatch must complete");
    for index in 0..THREADS as usize {
        // SAFETY: The shared page holds at least `THREADS` u32 words.
        let actual = unsafe { *words.add(index) };
        assert_eq!(
            actual,
            FILL_BASE + index as u32,
            "compute output word {index}"
        );
    }

    registry.destroy(first).expect("pipeline destroy");
    assert!(registry.is_empty());
    transport
        .unregister_buffer(buffer)
        .expect("unregister buffer");

    eprintln!(
        "compute PSO proof on {}: key={} created={} reused={}, exact {}-{} values",
        info.name_str(),
        key.digest_hex(),
        registry.created_count(),
        registry.reuse_count(),
        FILL_BASE,
        FILL_BASE + THREADS - 1
    );
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn residency_set_binds_allocations_and_keeps_values_exact() {
    let dir = TempDir::new("registry-residency");
    let Some(metallib) = compile_fill_metallib(dir.path()) else {
        eprintln!(
            "PARTIAL: Apple Metal toolchain (metal/metallib) not found; residency proof skipped"
        );
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let mut textures = TextureRegistry::new(&transport).expect("texture registry");
    let mut pipelines = PipelineRegistry::new(&transport).expect("pipeline registry");
    let mut residency = ResidencySet::new(&transport).expect("residency set");

    let key = PipelineKey::from_parts(&metallib, FILL_ENTRY, PipelineStage::Compute, b"");
    let pipeline = pipelines
        .register_compute(key, &metallib, FILL_ENTRY, b"")
        .expect("compute PSO");

    let (_guard, buffer, words) = fill_pattern(&transport);
    residency
        .add_buffer(&transport, buffer, PAGE)
        .expect("buffer must join the residency set");
    textures
        .register(11, rgba8_desc(4, 4, TextureUsage::sampled()))
        .expect("residency texture must create");
    residency
        .add_texture(&textures, 11)
        .expect("texture must join the residency set");
    residency.commit();
    residency
        .attach_to_queue(&transport)
        .expect("residency set must attach to the queue");

    if residency.is_native() {
        assert!(residency.contains_buffer(&transport, buffer).unwrap());
        assert!(residency.contains_texture(&textures, 11).unwrap());
        residency.request_residency();
        eprintln!("residency: NATIVE MTLResidencySet accepted buffer + texture");
    } else {
        assert!(!residency.is_native());
        assert_eq!(residency.allocation_count(), 2);
        let reason = residency
            .fallback_reason()
            .expect("fallback must record the failing selector");
        assert!(
            reason.contains("newResidencySetWithDescriptor")
                || reason.contains("MTLResidencySetDescriptor"),
            "unexpected fallback reason: {reason}"
        );
        eprintln!("PARTIAL: residency fallback in use: {reason}");
    }

    pipelines
        .dispatch_compute_1d(&transport, pipeline, &[buffer], THREADS, THREADS_PER_GROUP)
        .expect("dispatch under residency must complete");
    for index in 0..THREADS as usize {
        // SAFETY: The shared page holds at least `THREADS` u32 words.
        let actual = unsafe { *words.add(index) };
        assert_eq!(
            actual,
            FILL_BASE + index as u32,
            "residency dispatch output word {index}"
        );
    }
    residency.end_residency();

    textures.destroy(11).expect("texture destroy");
    transport
        .unregister_buffer(buffer)
        .expect("unregister buffer");

    eprintln!(
        "residency proof on {}: native={}, allocations={}, reserved_bytes={}, exact {}-{} values",
        info.name_str(),
        residency.is_native(),
        residency.allocation_count(),
        residency.reserved_bytes(),
        FILL_BASE,
        FILL_BASE + THREADS - 1
    );
}
