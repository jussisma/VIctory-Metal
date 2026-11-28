//! Hardware proof for ADR-066: a converted pipeline's **reflected** fragment
//! descriptor-table kind and descriptor slot count bind the wire set's resolved
//! `MTLTexture`/`MTLSamplerState` handles instead of the table being dropped as
//! inert.
//!
//! Boot #36 (head `87d1c7cd`) measured the composition pipeline (id 3) as
//! `declared_kind=Sampler, declared_tables=1, declared_slots=1,
//! mix_class=buffer+texture, reason=declared-kind-rejects-partitions`: the
//! converter typed the table field by the range kind it guessed, so the wire
//! resource set filled no declared partition and every draw was inert
//! (`fragment_texture_sources_declared=1, bound=0, dropped=1`,
//! `tables_dropped_inert=1`).
//!
//! This file reproduces that exact declared shape — one fragment table field
//! reflected as `Sampler` with the converter's 24-byte descriptor extent — and
//! proves:
//!
//! 1. the buffer+texture wire set binds **whole** at that field in wire order
//!    (adoption), so `declared=1, bound=1, dropped=0` and `inert=0`;
//! 2. a fragment that samples through a reflected texture table **and** a
//!    reflected sampler table yields exact non-black texels.
//!
//! Run directly (the GPU lock is taken by the caller when one is needed):
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test fragment_table_binding_gpu -- --test-threads=1 --nocapture
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
    DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi, SamplerDesc};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::macos::trace::{
    OUTCOME_MERGED, descriptor_table_refusal_census, reset_abi_binding_telemetry,
    reset_descriptor_table_refusal_census, stage_abi_telemetry,
};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire::BufferBinding;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyBufferToTexture, DescriptorKind,
    DescriptorTableEntryV3, DescriptorTableV3, DrawState, LoadOp, PipelineRef, PrimitiveTopology,
    ScissorRect, StoreOp, SubmissionV3, Viewport,
};

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const TEXTURE_EDGE: u32 = 4;
const ROW_PITCH: usize = 256;
const PAGE_BYTES: usize = 16 * 1024;

const TARGET: u64 = 0x2000;
const TEXTURE0: u64 = 0x3000;
const SAMPLER: u64 = 0x4000;

const ALLOC_TARGET: u64 = 1;
const ALLOC_TEXTURE0: u64 = 2;
const ALLOC_UPLOAD: u64 = 3;
const ALLOC_BUFFER: u64 = 4;

/// Exact RGBA8 upload colour; the whole 4x4 image is this texel.
const COLOR0: [u8; 4] = [10, 20, 30, 255];

/// The converter's per-table argument-buffer shape.
///
/// `TextureDescriptor` is the 24-byte `{texture, texture, i64}` element the host
/// writes `gpuResourceID`s into. The fragment samples entry **1** — the texture
/// entry of the measured `buffer+texture` wire set — so a host that renumbered,
/// dropped, or bound only the buffer entry cannot produce the asserted texel.
const COMPOSITION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct TableAbi {
    device TextureDescriptor *textures [[id(0)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut fragment_table_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 composition_fragment(constant TableAbi &abi [[buffer(2)]]) {
    constexpr sampler smp(filter::nearest, address::clamp_to_edge, coord::normalized);
    float4 texel = abi.textures[1].primary.sample(smp, float2(0.5, 0.5));
    return float4(texel.r, texel.g, texel.b, 1.0);
}
"#;

/// A reflected fragment texture table plus a reflected sampler table, both read
/// through the argument buffer exactly as the converter emits them.
const TABLE_AND_SAMPLER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextureDescriptor {
    texture2d<float> primary;
    texture2d<float> duplicate;
    long reserved;
};

struct SamplerDescriptor {
    sampler state;
    long reserved0;
    long reserved1;
};

struct TableAbi {
    device TextureDescriptor *textures [[id(0)]];
    device SamplerDescriptor *samplers [[id(1)]];
};

struct VOut {
    float4 position [[position]];
};

vertex VOut table_sampler_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment float4 table_sampler_fragment(constant TableAbi &abi [[buffer(2)]]) {
    float4 texel = abi.textures[0].primary.sample(abi.samplers[0].state, float2(0.5, 0.5));
    return float4(texel.r, texel.g, texel.b, 1.0);
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
    let library = dir.join(format!("{stem}.metallib"));
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

    fn write(&mut self, offset: usize, bytes: &[u8]) {
        assert!(
            offset + bytes.len() <= self.len,
            "host page write out of bounds"
        );
        // SAFETY: `offset + bytes.len() <= self.len` and `self.ptr` is live.
        unsafe {
            let destination = self.ptr.cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
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
    }
}

fn assert_uniform(pixels: &[[u8; 4]], expected: [u8; 4], label: &str) {
    assert_eq!(pixels.len(), PIXELS, "{label} pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, expected, "{label} pixel {index}");
    }
}

fn copy_to_texture(
    allocation_id: u64,
    destination_allocation_id: u64,
    offset: u64,
) -> CommandV3 {
    CommandV3::CopyBufferToTexture(CopyBufferToTexture {
        allocation_id,
        generation: 1,
        destination_allocation_id,
        destination_generation: 1,
        offset,
        bytes_per_row: ROW_PITCH as u32,
        bytes_per_image: 0,
        extent: [TEXTURE_EDGE, TEXTURE_EDGE, 1],
        mip_level: 0,
        slice: 0,
        origin: [0, 0, 0],
    })
}

/// Opens the transport, or returns `None` (with a loud eprintln) when the
/// machine has no Metal device: a skipped hardware proof is not evidence.
fn open_transport(label: &str) -> Option<DirectMetalTransport> {
    match DirectMetalTransport::new() {
        Ok(transport) => Some(transport),
        Err(error) => {
            eprintln!("PARTIAL: {label}: no Metal device ({error}); hardware proof skipped");
            None
        }
    }
}

/// Registers the 4x4 upload texture plus the 16x16 RGBA8 render target.
fn register_textures(transport: &DirectMetalTransport) {
    for (handle, label) in [(TEXTURE0, "texture0"), (TARGET, "target")] {
        let edge = if handle == TARGET { EDGE } else { TEXTURE_EDGE };
        transport
            .register_registry_texture(
                handle,
                rgba8_desc(edge, edge, TextureUsage::color_attachment(), label),
            )
            .expect("registry texture must create");
    }
}

/// A page whose first image is `COLOR0`; the pitch gap stays poison so a
/// tight-pitch copy is visible.
fn upload_page() -> HostPage {
    let mut upload = HostPage::new(PAGE_BYTES);
    upload.write(0, &[0xaau8; PAGE_BYTES]);
    for row in 0..TEXTURE_EDGE as usize {
        for column in 0..TEXTURE_EDGE as usize {
            upload.write(row * ROW_PITCH + column * 4, &COLOR0);
        }
    }
    upload
}

/// The composition shape (pipeline 3 of Boot #36): **one** fragment table field
/// reflected as `Sampler` with the converter's 24-byte descriptor extent, fed a
/// `buffer+texture` wire set. The adoption binds the whole array in wire order,
/// so entry 1 — the registered texture — is sampled and the counters read
/// `declared=1, bound=1, dropped=0, inert=0`.
#[test]
fn a_reflected_fragment_table_binds_a_buffer_texture_set_exactly() {
    let Some(transport) = open_transport("composition-table") else {
        return;
    };
    let info = transport.device_info().expect("device info");
    reset_descriptor_table_refusal_census();
    reset_abi_binding_telemetry();

    let dir = TempDir::new("fragment-table-binding");
    let metallib = compile_metallib(dir.path(), "fragment_table", COMPOSITION_MSL);
    register_textures(&transport);

    let mut upload = upload_page();
    upload.write(PAGE_BYTES / 2, &[0xaau8; PAGE_BYTES / 2]);
    let buffer_page = HostPage::new(PAGE_BYTES);
    // SAFETY: both pages are live, 16 KiB-aligned allocations that outlive the
    // synchronous submission and are unregistered before they drop.
    let upload_id = unsafe { transport.register_zerocopy(upload.raw(), PAGE_BYTES) }
        .expect("upload page must register");
    // SAFETY: same zero-copy contract as the upload page.
    let buffer_id = unsafe { transport.register_zerocopy(buffer_page.raw(), PAGE_BYTES) }
        .expect("buffer page must register");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_UPLOAD, 1), upload_id);
    allocations.insert((ALLOC_BUFFER, 1), buffer_id);

    // Exactly the reflected shape Boot #36's `binding_drops` key named:
    // `declared_kind=Sampler, declared_slots=1`, the converter's 24-byte
    // descriptor extent at argument-buffer offset 0.
    let fragment = RootSignatureLayout::from_converter_reflection_json(
        r#"{"EntryPoint":"composition_fragment","ShaderType":"Fragment","TopLevelArgumentBuffer":[{"EltOffset":0,"Name":"","Size":24,"Slot":0,"Space":0,"Type":"Sampler"}]}"#,
    )
    .expect("one reflected fragment table must build");
    assert_eq!(
        fragment.declared_tables()[0].field_bytes, 24,
        "the converter's descriptor extent is carried from registration"
    );

    const HANDLE: u64 = 0x6601;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "fragment_table_vertex",
            &PipelineState {
                color_formats: vec![TextureFormat::Rgba8Unorm],
                depth_format: None,
                fragment_entry: Some("composition_fragment".to_string()),
                ..Default::default()
            }
            .encode(PipelineStage::Graphics),
            PipelineAbi::graphics(None, Some(fragment)),
        )
        .expect("composition graphics pipeline must register");

    let submission = SubmissionV3 {
        sequence: 801,
        list: None,
        commands: vec![
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE0, 0),
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
            // The measured mix: a buffer entry first, the registered texture
            // second, one `Srv` set.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![
                    DescriptorTableEntryV3::Buffer(BufferBinding {
                        allocation_id: ALLOC_BUFFER,
                        generation: 1,
                        offset: 0,
                        length: 16,
                    }),
                    DescriptorTableEntryV3::Texture(TEXTURE0),
                ],
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
        .expect("the reflected-table submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "status");
    assert_eq!(stats.render_encoders, 1, "one render encoder draws");
    assert_eq!(transport.abi_refusal_count(), 0, "the set is not refused");
    assert_eq!(
        transport.descriptor_table_array_adoption_count(),
        1,
        "the declared field adopted the converter's descriptor array"
    );
    assert_eq!(
        transport.descriptor_table_inert_acceptance_count(),
        0,
        "the table is bound, not inert"
    );
    assert_eq!(
        transport.descriptor_table_mixed_unrepresentable_count(),
        0,
        "no mixed table is unrepresentable any more"
    );
    assert_eq!(
        transport.descriptor_table_partial_binding_count(),
        0,
        "nothing was dropped as residue"
    );

    let census = descriptor_table_refusal_census();
    assert!(
        census.binding_drops.is_empty(),
        "no drop is named: {}",
        census.json_value()
    );
    assert_eq!(census.mix_keys.len(), 1, "{}", census.json_value());
    assert_eq!(census.mix_keys[0].key.mix_class, "buffer+texture");
    assert_eq!(census.mix_keys[0].key.outcome, OUTCOME_MERGED);

    let abi = stage_abi_telemetry();
    assert_eq!(abi.stage_mask & 0b10, 0b10, "the fragment layout is bound");
    assert_eq!(abi.ps_abi_parameters, 1, "one reflected fragment parameter");
    assert_eq!(abi.ps_bound_parameters, 1, "the wire set binds it");
    assert_eq!(abi.ps_bound_tables, 1, "the bound parameter is a table");
    assert_eq!(
        abi.fragment_texture_sources_declared, 1,
        "one declared fragment table field"
    );
    assert_eq!(
        abi.fragment_texture_sources_bound, 1,
        "the texture entry is bound"
    );
    assert_eq!(abi.fragment_texture_sources_dropped, 0);
    assert_eq!(abi.fragment_source_drop_reason, "none");
    assert_eq!(abi.tables_dropped_inert, 0);
    assert_eq!(abi.ps_tables_dropped_inert, 0);

    transport
        .wait_for_gpu()
        .expect("the reflected-table pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, COLOR0, "reflected fragment table target");

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");
    transport
        .unregister_buffer(buffer_id)
        .expect("unregister the buffer page");
    eprintln!(
        "reflected fragment table proof on {}: pixel={:?}, declared={}, bound={}, dropped={}, \
         reason={}, inert={}, adoptions={}, outcome={}",
        info.name_str(),
        pixels[0],
        abi.fragment_texture_sources_declared,
        abi.fragment_texture_sources_bound,
        abi.fragment_texture_sources_dropped,
        abi.fragment_source_drop_reason,
        abi.tables_dropped_inert,
        transport.descriptor_table_array_adoption_count(),
        census.mix_keys[0].key.outcome
    );
}

/// A reflected fragment texture table **plus** a reflected sampler table bind
/// their resolved `MTLTexture`/`MTLSamplerState` handles, and the draw samples
/// the exact registered texel through the table sampler.
#[test]
fn a_reflected_fragment_table_and_sampler_table_sample_exactly() {
    let Some(transport) = open_transport("fragment-table-sampler") else {
        return;
    };
    let info = transport.device_info().expect("device info");
    reset_descriptor_table_refusal_census();
    reset_abi_binding_telemetry();

    let dir = TempDir::new("fragment-table-sampler");
    let metallib = compile_metallib(dir.path(), "table_sampler", TABLE_AND_SAMPLER_MSL);
    register_textures(&transport);
    transport
        .register_sampler(SAMPLER, SamplerDesc::nearest_clamp())
        .expect("sampler must register");

    let mut upload = upload_page();
    upload.write(PAGE_BYTES / 2, &[0xaau8; PAGE_BYTES / 2]);
    // SAFETY: the page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
    let upload_id = unsafe { transport.register_zerocopy(upload.raw(), PAGE_BYTES) }
        .expect("upload page must register");

    let mut allocations = HashMap::new();
    allocations.insert((ALLOC_TARGET, 1), VmBufferId(TARGET));
    allocations.insert((ALLOC_TEXTURE0, 1), VmBufferId(TEXTURE0));
    allocations.insert((ALLOC_UPLOAD, 1), upload_id);

    let fragment = RootSignatureLayout::from_parameters(
        ReflectedStage::Fragment,
        &[
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Unspecified),
                size: 8,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::Sampler),
                size: 8,
                register: 0,
                space: 0,
            },
        ],
    )
    .expect("texture+sampler fragment tables must build");

    const HANDLE: u64 = 0x6602;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "table_sampler_vertex",
            &PipelineState {
                color_formats: vec![TextureFormat::Rgba8Unorm],
                depth_format: None,
                fragment_entry: Some("table_sampler_fragment".to_string()),
                ..Default::default()
            }
            .encode(PipelineStage::Graphics),
            PipelineAbi::graphics(None, Some(fragment)),
        )
        .expect("table+sampler graphics pipeline must register");

    let submission = SubmissionV3 {
        sequence: 802,
        list: None,
        commands: vec![
            copy_to_texture(ALLOC_UPLOAD, ALLOC_TEXTURE0, 0),
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
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![DescriptorTableEntryV3::Texture(TEXTURE0)],
            }),
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 1,
                base_register: 0,
                kind: DescriptorKind::Sampler,
                bindings: vec![DescriptorTableEntryV3::Sampler(SAMPLER)],
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
        .expect("the table+sampler submission must execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "status");
    assert_eq!(stats.render_encoders, 1, "one render encoder draws");
    assert_eq!(transport.abi_refusal_count(), 0, "both sets resolve");

    let abi = stage_abi_telemetry();
    assert_eq!(abi.ps_abi_parameters, 2, "texture table then sampler table");
    assert_eq!(abi.ps_bound_parameters, 2);
    assert_eq!(abi.ps_bound_tables, 2);
    assert_eq!(
        abi.fragment_texture_sources_declared, 2,
        "one texture table and one sampler table are declared"
    );
    assert_eq!(
        abi.fragment_texture_sources_bound, 2,
        "both declared sources are bound"
    );
    assert_eq!(abi.fragment_texture_sources_dropped, 0);
    assert_eq!(abi.tables_dropped_inert, 0);

    transport
        .wait_for_gpu()
        .expect("the table+sampler pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("target readback");
    assert_uniform(&pixels, COLOR0, "reflected table+sampler target");

    transport
        .unregister_buffer(upload_id)
        .expect("unregister the upload page");
    eprintln!(
        "reflected table+sampler proof on {}: pixel={:?}, declared={}, bound={}, inert={}",
        info.name_str(),
        pixels[0],
        abi.fragment_texture_sources_declared,
        abi.fragment_texture_sources_bound,
        abi.tables_dropped_inert
    );
}
