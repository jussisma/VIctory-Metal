//! Hardware proof for the descriptor-table kind-mismatch class of Boot #30.
//!
//! The table has already resolved to a declared root parameter; only the
//! *kind* pair disagreed. The converter writes one 24-byte
//! `{device_address, 0, 0}` descriptor for every buffer range, so a template
//! that reflects a range as `ShaderResourceView` (or `ConstantBufferView`)
//! while the wire tags the same buffer entries `Cbv`/`Srv`/`Uav` is
//! shape-identical and must encode. A genuinely impossible pair (an address
//! table declared as a `Sampler` range) stays a named refusal, is counted in
//! the bounded kind-mismatch census, and never moves `abi_refusals`.
//!
//! The fragment shader writes a constant colour and reads no table, so the
//! asserted pixels prove the draw executed while the mixed-kind table was
//! encoded — not that the table was silently dropped.
//!
//! The test takes `/tmp/victory-gpu.lock` itself. Run directly:
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test descriptor_table_kind_gpu -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::bindings::{
    DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::macos::trace::{
    descriptor_table_refusal_census, reset_descriptor_table_refusal_census,
};
use victory_metal::wire::BufferBinding;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DescriptorKind, DescriptorTableEntryV3,
    DescriptorTableV3, DrawState, LoadOp, PipelineRef, PrimitiveTopology, ScissorRect, StoreOp,
    SubmissionV3, Viewport,
};

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const PAGE_BYTES: usize = 16 * 1024;

const TARGET: u64 = 0x2000;
const ALLOC_TARGET: u64 = 1;
const ALLOC_SIDE: u64 = 2;

/// The draw's colour; distinct from the pass clear so an un-encoded draw is
/// visible in the readback.
const DRAWN_RGBA: [u8; 4] = [7, 11, 13, 255];

/// A fragment that writes a constant colour and reads no root argument.
const CONSTANT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
};

vertex VOut constant_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 constant_fragment() {
    return float4(7.0 / 255.0, 11.0 / 255.0, 13.0 / 255.0, 1.0);
}
"#;

fn tool_path(name: &str) -> Option<PathBuf> {
    let output = Command::new("xcrun").args(["-f", name]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| PathBuf::from(text))
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

/// Compiles `source` into a real `.metallib`; a missing toolchain fails loudly.
fn compile_metallib(dir: &Path, stem: &str, source_text: &str) -> Vec<u8> {
    let metal = tool_path("metal").expect("the Metal toolchain (metal) must be installed");
    let metallib = tool_path("metallib").expect("the Metal toolchain (metallib) must be installed");
    let source = dir.join(format!("{stem}.metal"));
    let air = dir.join(format!("{stem}.air"));
    let library = dir.join(format!("{stem}.library.metallib"));
    fs::write(&source, source_text).expect("write MSL source");

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
    fs::read(&library).expect("read compiled metallib")
}

fn rgba8_desc(width: u32, height: u32, usage: TextureUsage, label: &str) -> TextureDesc {
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
        label: Some(label.to_string()),
    }
}

/// A translated-looking signature with exactly one table of `kind` at space 1,
/// register 0: the reflected range kind the wire table must fill.
fn one_table_abi(kind: DescriptorTableKind) -> PipelineAbi {
    let layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[RootParameterSpec {
            kind: RootParameterKind::DescriptorTable(kind),
            size: 8,
            register: 0,
            space: 1,
        }],
    )
    .expect("one-table fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

fn buffer_table(kind: DescriptorKind) -> DescriptorTableV3 {
    DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(BufferBinding {
                allocation_id: ALLOC_SIDE,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            DescriptorTableEntryV3::Buffer(BufferBinding {
                allocation_id: ALLOC_SIDE,
                generation: 1,
                offset: 16,
                length: 16,
            }),
        ],
    }
}

fn color_attachment(allocation_id: u64, clear: [f64; 4]) -> ColorAttachment {
    ColorAttachment {
        allocation_id,
        generation: 1,
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: clear,
    }
}

fn full_viewport() -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: f64::from(EDGE),
        height: f64::from(EDGE),
        near: 0.0,
        far: 1.0,
    }
}

fn full_scissor() -> ScissorRect {
    ScissorRect {
        x: 0,
        y: 0,
        width: EDGE,
        height: EDGE,
    }
}

/// One mixed-kind draw: a declared table of `reflected` kind fed by a wire
/// table of `wire` kind.
fn submission(sequence: u32, pipeline: u64, wire: DescriptorKind) -> SubmissionV3 {
    SubmissionV3 {
        sequence,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(ALLOC_TARGET, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: pipeline,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetDescriptorTable(buffer_table(wire)),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    }
}

/// A page-aligned zero-copy host page.
struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn new(len: usize) -> Self {
        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        }
        let mut raw: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation.
        let rc = unsafe { posix_memalign(&mut raw, 16 * 1024, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());
        // SAFETY: `raw` is a fresh allocation of `len` writable bytes.
        unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, len) };
        Self { ptr: raw, len }
    }

    fn raw(&self) -> *mut u8 {
        self.ptr.cast()
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut c_void);
        }
        // SAFETY: `self.ptr` came from `posix_memalign` and is no longer
        // registered with the transport when this guard drops.
        unsafe { free(self.ptr) };
        let _ = self.len;
    }
}

// ----- GPU lock -------------------------------------------------------------

struct GpuLock {
    path: PathBuf,
}

impl GpuLock {
    /// Takes `/tmp/victory-gpu.lock` (an atomic `mkdir`), reclaiming a lock
    /// whose recorded PID is gone. `None` after five minutes of a live holder.
    fn acquire() -> Option<Self> {
        let path = PathBuf::from("/tmp/victory-gpu.lock");
        let deadline = SystemTime::now() + Duration::from_secs(300);
        loop {
            match fs::create_dir(&path) {
                Ok(()) => {
                    let pid = std::process::id();
                    let _ = fs::write(path.join("owner"), format!("{pid} 0\n"));
                    return Some(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        let _ = fs::remove_dir_all(&path);
                        continue;
                    }
                    if SystemTime::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
                Err(_) => return None,
            }
        }
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn lock_is_stale(path: &Path) -> bool {
    let owner = fs::read_to_string(path.join("owner")).unwrap_or_default();
    let pid = owner
        .split_whitespace()
        .next()
        .and_then(|pid| pid.parse::<u32>().ok());
    let Some(pid) = pid else {
        // No parseable owner: reclaim only an old directory.
        return fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > Duration::from_secs(900));
    };
    !Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// A buffer-entry table reflected as SRV/CBV encodes with a `Cbv`, `Srv` or
/// `Uav` wire kind, and an address table under a `Sampler` range stays a
/// named, censused refusal that does not move `abi_refusals`.
#[test]
fn a_mixed_kind_buffer_table_encodes_and_an_impossible_pair_is_named() {
    let Some(_lock) = GpuLock::acquire() else {
        eprintln!("PARTIAL: another GPU run holds /tmp/victory-gpu.lock; proof skipped");
        return;
    };
    reset_descriptor_table_refusal_census();

    let dir = TempDir::new("descriptor-table-kind");
    let metallib = compile_metallib(dir.path(), "kind", CONSTANT_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    transport
        .register_registry_texture(
            TARGET,
            rgba8_desc(EDGE, EDGE, TextureUsage::color_attachment(), "target"),
        )
        .expect("registry texture must create");

    let side = HostPage::new(PAGE_BYTES);
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submissions and is unregistered before it drops.
    let side_id = unsafe { transport.register_zerocopy(side.raw(), PAGE_BYTES) }
        .expect("side page must register");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_SIDE, 1), side_id);

    let pipeline_state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("constant_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    // Reflected ShaderResourceView fed by wire Cbv: the Boot #30 class.
    const SRV_RANGE: u64 = 0x6001;
    // Reflected ConstantBufferView fed by wire Srv: the reverse direction.
    const CBV_RANGE: u64 = 0x6002;
    // Reflected Sampler fed by wire Cbv: genuinely impossible.
    const SAMPLER_RANGE: u64 = 0x6003;
    for (handle, kind) in [
        (SRV_RANGE, DescriptorTableKind::ShaderResourceView),
        (CBV_RANGE, DescriptorTableKind::ConstantBufferView),
        (SAMPLER_RANGE, DescriptorTableKind::Sampler),
    ] {
        transport
            .register_content_hashed_graphics_pipeline(
                handle,
                &metallib,
                "constant_vertex",
                &pipeline_state,
                one_table_abi(kind),
            )
            .expect("content-hashed graphics pipeline must register");
    }

    for (sequence, pipeline, wire) in [
        (501, SRV_RANGE, DescriptorKind::Cbv),
        (502, SRV_RANGE, DescriptorKind::Uav),
        (503, CBV_RANGE, DescriptorKind::Srv),
    ] {
        // `Ok` here is exactly "refused_submissions did not move for this
        // batch": the host answered ExecutionFailed only for refusals.
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission(sequence, pipeline, wire), &allocations)
            .expect("a mixed-kind buffer table must encode");
        assert_eq!(
            reply.status,
            victory_metal::wire::ReplyStatus::Ok,
            "sequence {sequence} status"
        );
        assert_eq!(stats.render_encoders, 1, "sequence {sequence} one encoder");
    }

    let counters = transport.encode_counters();
    assert_eq!(counters.draws, 3, "all three draws encoded");
    assert_eq!(counters.render_passes, 3, "all three passes encoded");
    assert_eq!(
        transport.abi_refusal_count(),
        0,
        "a kind mismatch is not an ABI refusal"
    );
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        0,
        "each pipeline declares its table"
    );
    let census = descriptor_table_refusal_census();
    assert!(
        census.kind_mismatches.is_empty(),
        "legal mixed-kind pairs record no mismatch: {}",
        census.json_value()
    );

    // A sampler range cannot be filled by a buffer address table. The refusal
    // names both kinds and lands in the same bounded census.
    let error = transport
        .submit_v3_with_stats(
            &submission(504, SAMPLER_RANGE, DescriptorKind::Cbv),
            &allocations,
        )
        .expect_err("an address table cannot fill a sampler range");
    let named = error.to_string();
    assert!(
        named.contains("Cbv") && named.contains("Sampler"),
        "the refusal names both kinds: {named}"
    );
    assert_eq!(
        transport.abi_refusal_count(),
        0,
        "the kind refusal is not a root-argument ABI refusal"
    );
    let census = descriptor_table_refusal_census();
    assert_eq!(census.distinct_keys, 1);
    assert_eq!(census.kind_mismatches.len(), 1);
    let entry = &census.kind_mismatches[0];
    assert_eq!(entry.count, 1);
    assert_eq!(entry.key.pipeline_id, SAMPLER_RANGE);
    assert_eq!(entry.key.wire_set, 0);
    assert_eq!(entry.key.reflected_kind, "Sampler");
    assert_eq!(entry.key.wire_kind, "Cbv");
    assert_eq!(entry.key.entry_shape, "buffer");

    transport
        .wait_for_gpu()
        .expect("the constant passes should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_eq!(pixels.len(), PIXELS, "target pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, DRAWN_RGBA, "target pixel {index}");
    }

    transport
        .unregister_buffer(side_id)
        .expect("unregister the side page");
    eprintln!(
        "mixed-kind table proof on {}: draws={}, abi_refusals={}, census={}",
        info.name_str(),
        counters.draws,
        transport.abi_refusal_count(),
        census.json_value()
    );
    reset_descriptor_table_refusal_census();
}
