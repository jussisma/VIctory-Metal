//! Hardware proof for the plan-less descriptor-table class of Boot #29.
//!
//! The bound pipeline's reflected/translated signature declares **no**
//! descriptor-table parameter (`declared_descriptor_table_count() == 0`): the
//! translated root signature lowers title tables to directly indexed heap
//! access. A plan-less wire `SetDescriptorTable` therefore names no root
//! parameter. The resolver must classify that as a *named, counted inert
//! acceptance*: the pass and draw encode, `abi_refusals` stays 0, and the census
//! attributes the class (`reflected_tables: 0`, `map_tables: 0`, `had_plan:
//! false`, `outcome: "inert"`).
//!
//! The fragment shader writes a constant colour and never reads a table, so the
//! asserted pixels prove the draw really executed while the table binding was
//! inert — not that the table was silently bound somewhere unexpected.
//!
//! Run directly; the shared GPU lock is acquired around `cargo test`:
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test descriptor_table_pipeline2_gpu -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::bindings::{
    ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::macos::trace::{
    OUTCOME_INERT, descriptor_table_refusal_census, reset_descriptor_table_refusal_census,
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

/// A translated-looking signature with **zero** descriptor tables: one root
/// constants block and one root SRV, exactly the shape a converted pipeline
/// has when the title's tables were lowered to directly indexed heaps.
fn zero_table_abi() -> PipelineAbi {
    let layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[
            RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: 16,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::ShaderResourceView,
                size: 8,
                register: 0,
                space: 1,
            },
        ],
    )
    .expect("zero-table fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
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

/// A plan-less table on a zero-declaration signature is accepted as inert: the
/// draw encodes, `abi_refusals` is 0, and the census names the class.
#[test]
fn a_plan_less_table_on_a_zero_table_signature_encodes_inert() {
    reset_descriptor_table_refusal_census();
    let dir = TempDir::new("descriptor-table-pipeline2");
    let metallib = compile_metallib(dir.path(), "pipeline2", CONSTANT_MSL);
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for (handle, label) in [(TARGET, "target")] {
        transport
            .register_registry_texture(
                handle,
                rgba8_desc(EDGE, EDGE, TextureUsage::color_attachment(), label),
            )
            .expect("registry texture must create");
    }

    let side = HostPage::new(PAGE_BYTES);
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
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

    const HANDLE: u64 = 0x5000;
    let abi = zero_table_abi();
    assert_eq!(
        abi.declared_descriptor_table_count(),
        0,
        "the fixture must declare no table"
    );
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "constant_vertex",
            &pipeline_state,
            abi,
        )
        .expect("content-hashed graphics pipeline must register");

    let submission = SubmissionV3 {
        sequence: 402,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(ALLOC_TARGET, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            // The Boot #29 shape: no plan section, wire set 0, on a pipeline
            // whose reflected signature declares no table at all.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![DescriptorTableEntryV3::Buffer(BufferBinding {
                    allocation_id: ALLOC_SIDE,
                    generation: 1,
                    offset: 0,
                    length: 16,
                })],
            }),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };

    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the plan-less table submission must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok, "status");
    assert_eq!(stats.render_encoders, 1, "one render encoder");
    assert_eq!(
        transport.abi_refusal_count(),
        0,
        "the inert acceptance is not an ABI refusal"
    );
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        1,
        "the class is counted, not dropped"
    );
    assert_eq!(
        transport.descriptor_table_stale_refusal_count(),
        0,
        "no plan was present, so nothing is stale"
    );

    let counters = transport.encode_counters();
    assert_eq!(counters.draws, 1, "draws_encoded increments");
    assert_eq!(counters.render_passes, 1, "render_passes_encoded increments");

    let census = descriptor_table_refusal_census();
    assert_eq!(census.distinct_keys, 1, "one bounded key");
    assert_eq!(census.omitted, 0);
    let entry = &census.entries[0];
    assert_eq!(entry.count, 1);
    assert_eq!(entry.key.pipeline_id, HANDLE);
    assert_eq!(entry.key.wire_set, 0);
    assert_eq!(entry.key.declared_tables, 0);
    assert_eq!(entry.key.reflected_tables, 0);
    assert_eq!(entry.key.map_tables, 0);
    assert_eq!(entry.key.stage_mask, 0b10, "fragment layout only");
    assert!(!entry.key.had_plan, "the submission carried no plan section");
    assert_eq!(entry.key.plan_kind, "none");
    assert_eq!(entry.key.outcome, OUTCOME_INERT);

    transport
        .wait_for_gpu()
        .expect("the constant pass should complete");
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
    reset_descriptor_table_refusal_census();

    eprintln!(
        "pipeline2 inert-table proof on {}: pixels={:?}, abi_refusals={}, inert={}, census={}",
        info.name_str(),
        pixels[0],
        transport.abi_refusal_count(),
        transport.descriptor_table_inert_acceptance_count(),
        census.json_value()
    );
}
