//! Hardware proof for the v3 registry/ABI/residency submit path.
//!
//! These tests are `#[ignore]`-gated and run on the real Apple GPU. They drive
//! [`DirectMetalTransport::submit_v3_with_stats`] end to end:
//!
//! 1. [`graphics_mrt_depth_registry_pixels_are_exact`] registers two RGBA8
//!    colour targets and a Depth32Float target through the ADR-013 registry,
//!    creates a content-hashed graphics PSO from MSL compiled with Apple's
//!    offline toolchain, binds a root CBV and a root-constants block through the
//!    reflected ABI, draws a full-screen triangle and asserts the exact pixels
//!    on both colour targets plus the live residency path.
//! 2. [`compute_content_pso_abi_runs_exact_value`] is the compute counterpart;
//!    it is reported `PARTIAL` with the exact failing selector when the x86_64
//!    IRConverter is unavailable.
//! 3. [`unknown_handles_are_refused_without_encoders`] proves an unknown texture
//!    or pipeline handle returns [`ReplyStatus::UnknownHandle`] with zero
//!    encoders and an unchanged target.
//!
//! Run serialised with the GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test submit_v3_bindings_gpu -- --ignored --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

mod c14_fixtures;

use std::collections::HashMap;
use std::ffi::{OsString, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::compiler_pool::{CompileJob, CompilerPool, PoolConfig};
use victory_metal::macos::bindings::{
    ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::device::{DirectMetalTransport, EncoderStats, PipelineAbi};
use victory_metal::macos::registry::{
    FunctionRole, PipelineStage, StageLibrary, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage,
};
use victory_metal::wire::ListIdentity;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DepthStencilAttachment, DrawIndexedState,
    DrawState, IndexBufferBinding, IndexFormat, LoadOp, PipelineRef, PrimitiveTopology,
    RootBufferView, RootConstantsV3, ScissorRect, StoreOp, SubmissionV3, VertexBuffer,
    VertexBufferSet, Viewport,
};

const EDGE: u32 = 64;
const PIXELS: usize = (EDGE * EDGE) as usize;

/// Exact 8-bit results the MRT fragment stage must write.
const TARGET0_RGBA: [u8; 4] = [255, 128, 0, 255];
const TARGET1_RGBA: [u8; 4] = [0, 128, 255, 255];

/// Content-hashed graphics shader.
///
/// The fragment stage reads a reflected top-level argument buffer at
/// `buffer(2)`: parameter 0 is a 16-byte root-constants block (the tint) and
/// parameter 1 is a root CBV (a `device const float4*` read as the second
/// colour). The vertex stage is a full-screen triangle and touches no ABI.
const GRAPHICS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BindAbi {
    float4 tint;
    device const float4 *second [[id(1)]];
};

struct VOut {
    float4 position [[position]];
};

struct MrtOut {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

vertex VOut bind_vertex(uint vid [[vertex_id]]) {
    float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut out;
    out.position = float4(corners[vid], 0.0, 1.0);
    return out;
}

fragment MrtOut bind_fragment(constant BindAbi &abi [[buffer(2)]]) {
    MrtOut out;
    out.a = abi.tint;
    out.b = abi.second[0];
    return out;
}
"#;

/// Content-hashed compute shader used by the optional compute proof.
const COMPUTE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BindAbi {
    float4 scale;
    device uint *out [[id(1)]];
};

kernel void bind_kernel(constant BindAbi &abi [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    abi.out[tid] = uint(abi.scale.x) + tid;
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

/// Compiles `source` into a real `.metallib`, or `None` without Xcode tools.
fn compile_metallib(dir: &Path, stem: &str, source_text: &str) -> Option<Vec<u8>> {
    let metal = tool_path("metal")?;
    let metallib = tool_path("metallib")?;
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
    Some(fs::read(&library).expect("read compiled metallib"))
}

fn rgba8_desc(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        width,
        height,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::color_attachment(),
        label: Some("submit-v3-bindings-colour".to_string()),
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
        label: Some("submit-v3-bindings-depth".to_string()),
    }
}

/// Reflected graphics ABI: root constants at parameter 0, root CBV at 1.
fn graphics_abi() -> PipelineAbi {
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
                kind: RootParameterKind::ConstantBufferView,
                size: 8,
                register: 0,
                space: 0,
            },
        ],
    )
    .expect("fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

fn float4_bytes(values: [f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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
        assert!(offset + bytes.len() <= self.len, "host page write out of bounds");
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

/// Registers two colour targets and one depth target through the ADR-013
/// registry and returns the `(allocation, generation) -> handle` snapshot.
fn register_mrt_targets(transport: &DirectMetalTransport) -> HashMap<(u64, u32), VmBufferId> {
    const COLOUR0: u64 = 0x1000;
    const COLOUR1: u64 = 0x1001;
    const DEPTH: u64 = 0x1002;
    for (id, desc) in [
        (COLOUR0, rgba8_desc(EDGE, EDGE)),
        (COLOUR1, rgba8_desc(EDGE, EDGE)),
        (DEPTH, depth_desc(EDGE, EDGE)),
    ] {
        transport
            .register_registry_texture(id, desc)
            .expect("registry texture must create");
    }
    let mut allocations = HashMap::new();
    allocations.insert((1u64, 1u32), VmBufferId(COLOUR0));
    allocations.insert((2u64, 1u32), VmBufferId(COLOUR1));
    allocations.insert((3u64, 1u32), VmBufferId(DEPTH));
    allocations
}

fn assert_uniform(pixels: &[[u8; 4]], expected: [u8; 4], label: &str) {
    assert_eq!(pixels.len(), PIXELS, "{label} pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, expected, "{label} pixel {index}");
    }
}

/// Exact-value graphics proof: two RGBA8 render targets plus D32F depth, a
/// content-hashed MRT PSO, a root CBV and root constants through the reflected
/// ABI, and a full-screen triangle.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn graphics_mrt_depth_registry_pixels_are_exact() {
    let dir = TempDir::new("submit-v3-bindings-gfx");
    let Some(metallib) = compile_metallib(dir.path(), "graphics", GRAPHICS_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; graphics proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    let allocations = register_mrt_targets(&transport);

    // Root CBV backing the second colour target.
    let mut cbv = HostPage::new(16 * 1024);
    cbv.write(0, &float4_bytes([0.0, 128.0 / 255.0, 1.0, 1.0]));
    // SAFETY: `cbv` is a live 16 KiB page that outlives the submission.
    let cbv_id = unsafe { transport.register_zerocopy(cbv.raw(), 16 * 1024) }
        .expect("root CBV page must register");

    let pipeline_state = victory_metal::macos::registry::PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm],
        depth_format: Some(TextureFormat::Depth32Float),
        fragment_entry: Some("bind_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x4000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "bind_vertex",
            &pipeline_state,
            graphics_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    let mut snapshot = allocations.clone();
    snapshot.insert((4u64, 1u32), cbv_id);

    let commands = vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                color_attachment(1, [0.0, 0.0, 0.0, 1.0]),
                color_attachment(2, [0.0, 0.0, 0.0, 1.0]),
            ],
            depth_stencil: Some(DepthStencilAttachment {
                allocation_id: 3,
                generation: 1,
                mip_level: 0,
                slice: 0,
                depth_load: LoadOp::Clear,
                depth_store: StoreOp::Store,
                clear_depth: 1.0,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                clear_stencil: 0,
            }),
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: HANDLE,
        }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetScissor(full_scissor()),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        // Parameter 0: inline root constants (the tint).
        CommandV3::SetRootConstants(RootConstantsV3 {
            slot: 0,
            bytes: float4_bytes([1.0, 128.0 / 255.0, 0.0, 1.0]),
        }),
        // Parameter 1: root CBV, 16 bytes of the shared page.
        CommandV3::SetRootBufferView(RootBufferView {
            slot: 1,
            allocation_id: 4,
            generation: 1,
            offset: 0,
            length: 16,
        }),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ];
    let submission = SubmissionV3 {
        sequence: 101,
        list: Some(ListIdentity {
            list_id: 11,
            list_generation: 1,
        }),
        commands,
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("MRT submission must execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "MRT submission status"
    );
    assert_eq!(
        stats,
        EncoderStats {
            command_buffers: 1,
            compute_encoders: 0,
            render_encoders: 1,
            blit_encoders: 0,
            barriers: 0,
        },
        "one render encoder must cover the pass"
    );

    // `submit_v3` returns after `commit`: wait for the GPU before reading the
    // registry textures back, so the readback cannot race the render pass.
    transport
        .wait_for_gpu()
        .expect("the MRT pass should complete");

    let target0 = transport
        .read_registered_texture_rgba8(0x1000)
        .expect("target 0 readback");
    let target1 = transport
        .read_registered_texture_rgba8(0x1001)
        .expect("target 1 readback");
    assert_uniform(&target0, TARGET0_RGBA, "target 0");
    assert_uniform(&target1, TARGET1_RGBA, "target 1");

    // The live residency path must have observed the targets and the CBV.
    let native = transport
        .residency_is_native()
        .expect("residency state is readable");
    let allocations_in_set = transport
        .residency_allocation_count()
        .expect("residency allocation count");
    let fallback = transport
        .residency_fallback_reason()
        .expect("residency fallback reason is readable");
    assert!(
        native || fallback.is_some(),
        "a non-native residency path must record its fallback reason"
    );
    assert!(
        allocations_in_set >= 3,
        "residency set must hold at least the two targets and the CBV (got {allocations_in_set})"
    );

    transport
        .unregister_buffer(cbv_id)
        .expect("unregister the CBV page");

    eprintln!(
        "graphics MRT+depth proof on {}: target0={:?}, target1={:?}, \
         native_residency={}, allocations={}, fallback={:?}",
        info.name_str(),
        target0[0],
        target1[0],
        native,
        allocations_in_set,
        fallback
    );
}

/// Two-attribute MSL graphics pipeline. Attribute 0 is the position and
/// attribute 1 the colour; the colour is `[[flat]]` so the whole primitive
/// takes one provoking-vertex colour and the readback is exact, not
/// interpolation-dependent.
const VERTEX_DESCRIPTOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VIn {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VOut {
    float4 position [[position]];
    float4 color [[flat]];
};

struct MrtOut {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

vertex VOut vd_vertex(VIn in [[stage_in]]) {
    VOut out;
    out.position = in.position;
    out.color = in.color;
    return out;
}

fragment MrtOut vd_fragment(VOut in [[stage_in]]) {
    MrtOut out;
    out.a = in.color;
    out.b = in.color;
    return out;
}
"#;

/// The exact layout the host derives for the MSL above: float4 at offset 0,
/// float4 at offset 16, both in buffer 0, stride 32.
fn two_attribute_layout() -> (
    victory_metal::macos::vertex_layout::VertexLayout,
    u32,
) {
    use victory_metal::macos::vertex_layout::{
        VertexAttribute, VertexFormat, VertexLayout,
    };
    let position = VertexFormat::with_components(VertexFormat::FLOAT, 4).unwrap();
    let color = VertexFormat::with_components(VertexFormat::FLOAT, 4).unwrap();
    let layout = VertexLayout::from_attributes(vec![
        VertexAttribute {
            index: 0,
            format: position,
            offset: 0,
            buffer_index: 0,
            stride: 32,
        },
        VertexAttribute {
            index: 1,
            format: color,
            offset: 16,
            buffer_index: 0,
            stride: 32,
        },
    ])
    .expect("a two-attribute layout");
    (layout, 32)
}

/// Exact-value proof that a host-derived vertex descriptor is enough for
/// Metal to create the PSO and fetch two attributes: draw one indexed
/// triangle whose vertices carry distinct colours, and assert the flat
/// provoking-vertex colour in every pixel.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn graphics_vertex_descriptor_registers_and_draws_exact_pixels() {
    let dir = TempDir::new("submit-v3-vertex-descriptor");
    let Some(metallib) = compile_metallib(dir.path(), "vertex-descriptor", VERTEX_DESCRIPTOR_MSL)
    else {
        eprintln!(
            "PARTIAL: Apple Metal toolchain (metal/metallib) not found; vertex-descriptor proof skipped"
        );
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    const TARGET: u64 = 0x2000;
    transport
        .register_registry_texture(TARGET, rgba8_desc(EDGE, EDGE))
        .expect("registry colour target must create");

    let (layout, stride) = two_attribute_layout();
    let pipeline_state = victory_metal::macos::registry::PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("vd_fragment".to_string()),
        vertex_layout: Some(layout),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x6000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "vd_vertex",
            &pipeline_state,
            PipelineAbi::graphics(None, None),
        )
        .expect("the vertex descriptor must let Metal create the pipeline");

    // Three interleaved vertices: position (float4) then colour (float4).
    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let corners = [
        [-1.0f32, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let colours = [RED, GREEN, BLUE];
    let mut vertices = Vec::with_capacity(3 * stride as usize);
    for (corner, colour) in corners.iter().zip(colours.iter()) {
        for value in corner.iter().chain(colour.iter()) {
            vertices.extend_from_slice(&value.to_le_bytes());
        }
    }
    assert_eq!(vertices.len(), 3 * stride as usize);

    let mut vertex_page = HostPage::new(16 * 1024);
    vertex_page.write(0, &vertices);
    // SAFETY: `vertex_page` is a live 16 KiB page that outlives the submission.
    let vertex_id = unsafe { transport.register_zerocopy(vertex_page.raw(), 16 * 1024) }
        .expect("vertex page must register");

    let indices: [u16; 3] = [0, 1, 2];
    let mut index_page = HostPage::new(16 * 1024);
    let mut index_bytes = Vec::with_capacity(6);
    for index in indices {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }
    index_page.write(0, &index_bytes);
    // SAFETY: `index_page` is a live 16 KiB page that outlives the submission.
    let index_id = unsafe { transport.register_zerocopy(index_page.raw(), 16 * 1024) }
        .expect("index page must register");

    let mut snapshot = HashMap::new();
    snapshot.insert((1u64, 1u32), VmBufferId(TARGET));
    snapshot.insert((2u64, 1u32), vertex_id);
    snapshot.insert((3u64, 1u32), index_id);

    let submission = SubmissionV3 {
        sequence: 111,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 0,
                buffers: vec![VertexBuffer {
                    allocation_id: 2,
                    generation: 1,
                    offset: 0,
                    stride,
                    length: vertices.len() as u64,
                }],
            }),
            CommandV3::SetIndexBuffer(IndexBufferBinding {
                allocation_id: 3,
                generation: 1,
                offset: 0,
                length: index_bytes.len() as u64,
                format: IndexFormat::Uint16,
            }),
            CommandV3::DrawIndexed(DrawIndexedState {
                index_count: 3,
                instance_count: 1,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("the vertex-descriptor pass must execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "vertex-descriptor submission status"
    );
    assert_eq!(stats.render_encoders, 1, "one render encoder");

    transport
        .wait_for_gpu()
        .expect("the vertex-descriptor pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("vertex-descriptor target readback");
    // Metal's flat interpolation takes the first (provoking) vertex; RED must
    // cover the whole triangle, proving the layout's offset and buffer index.
    assert_uniform(&pixels, [255, 0, 0, 255], "vertex descriptor target");

    transport
        .unregister_buffer(vertex_id)
        .expect("unregister the vertex page");
    transport
        .unregister_buffer(index_id)
        .expect("unregister the index page");
    eprintln!(
        "vertex-descriptor proof on {}: {} pixels = {:?}, stride {}",
        info.name_str(),
        pixels.len(),
        pixels[0],
        stride
    );
}

/// Parses a whitespace-free lowercase hex string.
fn unhex(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0, "hex string has an odd length");
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("valid hex pair"))
        .collect()
}

/// Apple's x86_64 IRConverter library, if the environment names one.
fn converter_library() -> Option<PathBuf> {
    std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                .filter(|path| path.is_file())
        })
        .or_else(|| {
            let path = PathBuf::from(std::env::var_os("HOME")?).join(
                "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                 D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
            );
            path.is_file().then_some(path)
        })
}

/// The x86_64 `victory-irconverter --stdio` worker, if it has been built.
fn converter_worker() -> Option<PathBuf> {
    std::env::var_os("VICTORY_IRCONVERTER_WORKER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/x86_64-apple-darwin/debug/victory-irconverter");
            path.is_file().then_some(path)
        })
}

/// Sets a converter environment variable for the life of one test.
struct ConverterEnv {
    name: &'static str,
    old: Option<OsString>,
}

impl ConverterEnv {
    fn set(name: &'static str, value: &Path) -> Self {
        let old = std::env::var_os(name);
        // SAFETY: this binary runs with --test-threads=1 and no other test
        // reads these variables at the same time.
        unsafe { std::env::set_var(name, value) };
        Self { name, old }
    }
}

impl Drop for ConverterEnv {
    fn drop(&mut self) {
        // SAFETY: see `ConverterEnv::set`.
        unsafe {
            match &self.old {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

/// Converter-backed proof: translate this repository's own two-attribute DXIL
/// vertex and pixel stages with the real Metal Shader Converter, register the
/// graphics PSO from the derived layout, and draw the indexed triangle.
///
/// The translated vertex function reports Metal attribute indices 11 and 12
/// (`POSITION` float3, `COLOR0` float4), which is the index base
/// `VertexLayout::from_dxil_container` derives. The translated shaders contain
/// no title bytes; the DXIL fixture is this repository's own HLSL.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn converted_vertex_descriptor_pso_links_and_draws_exact_pixels() {
    use victory_metal::macos::vertex_layout::{VertexFormat, VertexLayout};

    let Some(converter) = converter_library() else {
        eprintln!(
            "PARTIAL: no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES); converter-backed vertex-descriptor proof skipped"
        );
        return;
    };
    let Some(worker) = converter_worker() else {
        eprintln!(
            "PARTIAL: no x86_64 victory-irconverter worker; build it with `cargo build -p victory-metal --bin victory-irconverter --target x86_64-apple-darwin`; converter-backed vertex-descriptor proof skipped"
        );
        return;
    };
    let _env = ConverterEnv::set("VICTORY_METAL_IRCONVERTER", &converter);

    let vertex_dxil = unhex(c14_fixtures::VERTEX_DXIL_HEX);
    let pixel_dxil = unhex(c14_fixtures::PIXEL_DXIL_HEX);
    let layout = VertexLayout::from_dxil_container(&vertex_dxil)
        .expect("the two-attribute vertex signature lays out");
    let float3 = VertexFormat::with_components(VertexFormat::FLOAT, 3).unwrap();
    let float4 = VertexFormat::with_components(VertexFormat::FLOAT, 4).unwrap();
    assert_eq!(layout.attributes().len(), 2);
    assert_eq!(layout.attributes()[0].index, 11);
    assert_eq!(layout.attributes()[0].format, float3);
    assert_eq!(layout.attributes()[0].offset, 0);
    assert_eq!(layout.attributes()[1].index, 12);
    assert_eq!(layout.attributes()[1].format, float4);
    assert_eq!(layout.attributes()[1].offset, 12);
    assert_eq!(layout.attributes()[0].stride, 28);

    let config = PoolConfig {
        workers: 1,
        queue_depth: 4,
        job_timeout: Duration::from_secs(120),
        cache_dir: None,
        converter_version: "victory-metal-test/c14-vertex-descriptor/1".to_string(),
    };
    let pool = CompilerPool::new(config, Some(worker));
    let vertex = pool
        .compile_with_reflection(&CompileJob {
            id: 1,
            stage: 1,
            entry: "main",
            dxil: &vertex_dxil,
        })
        .expect("the two-attribute vertex stage must translate");
    let pixel = pool
        .compile_with_reflection(&CompileJob {
            id: 2,
            stage: 2,
            entry: "main",
            dxil: &pixel_dxil,
        })
        .expect("the pixel stage must translate");
    assert!(vertex.metallib.starts_with(b"MTLB"), "vertex metallib");
    assert!(pixel.metallib.starts_with(b"MTLB"), "pixel metallib");
    assert!(!vertex.reflection.is_empty() && !pixel.reflection.is_empty());

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    const TARGET: u64 = 0x2100;
    transport
        .register_registry_texture(TARGET, rgba8_desc(EDGE, EDGE))
        .expect("registry colour target must create");

    let state = victory_metal::macos::registry::PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("main".to_string()),
        vertex_layout: Some(layout.clone()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);
    const HANDLE: u64 = 0x6100;
    transport
        .register_content_hashed_graphics_pipeline_stages(
            HANDLE,
            &[
                StageLibrary {
                    role: FunctionRole::Vertex,
                    metallib: &vertex.metallib,
                    entry: "main",
                },
                StageLibrary {
                    role: FunctionRole::Fragment,
                    metallib: &pixel.metallib,
                    entry: "main",
                },
            ],
            &state,
            PipelineAbi::graphics(None, None),
        )
        .expect("the converter attribute indices must match the derived layout");

    // POSITION float3 at offset 0, COLOR0 float4 at offset 12, stride 28.
    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let corners = [
        [-1.0f32, -1.0, 0.0],
        [3.0, -1.0, 0.0],
        [-1.0, 3.0, 0.0],
    ];
    let colours = [RED, GREEN, BLUE];
    let stride = layout.attributes()[0].stride;
    let mut vertices = vec![0u8; 3 * stride as usize];
    for (index, (corner, colour)) in corners.iter().zip(colours.iter()).enumerate() {
        let base = index * stride as usize;
        for (component, value) in corner.iter().enumerate() {
            vertices[base + component * 4..base + component * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        for (component, value) in colour.iter().enumerate() {
            let at = base + 12 + component * 4;
            vertices[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    let mut vertex_page = HostPage::new(16 * 1024);
    vertex_page.write(0, &vertices);
    // SAFETY: `vertex_page` is a live 16 KiB page that outlives the submission.
    let vertex_id = unsafe { transport.register_zerocopy(vertex_page.raw(), 16 * 1024) }
        .expect("vertex page must register");
    let mut index_page = HostPage::new(16 * 1024);
    let mut index_bytes = Vec::new();
    for index in [0u16, 1, 2] {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }
    index_page.write(0, &index_bytes);
    // SAFETY: `index_page` is a live 16 KiB page that outlives the submission.
    let index_id = unsafe { transport.register_zerocopy(index_page.raw(), 16 * 1024) }
        .expect("index page must register");

    let mut snapshot = HashMap::new();
    snapshot.insert((1u64, 1u32), VmBufferId(TARGET));
    snapshot.insert((2u64, 1u32), vertex_id);
    snapshot.insert((3u64, 1u32), index_id);
    let submission = SubmissionV3 {
        sequence: 112,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetVertexBuffers(VertexBufferSet {
                start_slot: 0,
                buffers: vec![VertexBuffer {
                    allocation_id: 2,
                    generation: 1,
                    offset: 0,
                    stride,
                    length: vertices.len() as u64,
                }],
            }),
            CommandV3::SetIndexBuffer(IndexBufferBinding {
                allocation_id: 3,
                generation: 1,
                offset: 0,
                length: index_bytes.len() as u64,
                format: IndexFormat::Uint16,
            }),
            CommandV3::DrawIndexed(DrawIndexedState {
                index_count: 3,
                instance_count: 1,
                start_index: 0,
                base_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("the converter-backed pass must execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "converter-backed submission status"
    );
    assert_eq!(stats.render_encoders, 1, "one render encoder");
    transport
        .wait_for_gpu()
        .expect("the converter-backed pass should complete");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("converter-backed target readback");
    // The shader's flat colour takes the first vertex; RED proves both the
    // float3 position fetch at index 11/offset 0 and the float4 colour fetch
    // at index 12/offset 12.
    assert_uniform(&pixels, [255, 0, 0, 255], "converter-backed target");

    transport
        .unregister_buffer(vertex_id)
        .expect("unregister the vertex page");
    transport
        .unregister_buffer(index_id)
        .expect("unregister the index page");
    eprintln!(
        "converter-backed vertex-descriptor proof on {}: vertex {} bytes, pixel {} bytes, {} pixels = {:?}",
        info.name_str(),
        vertex.metallib.len(),
        pixel.metallib.len(),
        pixels.len(),
        pixels[0]
    );
}

/// Optional compute counterpart: a content-hashed compute PSO with the same
/// reflected ABI shape (root constants + root CBV acting as the UAV) writes
/// exact dwords.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn compute_content_pso_abi_runs_exact_value() {
    const THREADS: u32 = 64;
    let dir = TempDir::new("submit-v3-bindings-compute");
    let Some(metallib) = compile_metallib(dir.path(), "compute", COMPUTE_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; compute proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    let layout = RootSignatureLayout::from_parameters(
        ReflectedStage::Compute,
        &[
            RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: 16,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::UnorderedAccessView,
                size: 8,
                register: 0,
                space: 1,
            },
        ],
    )
    .expect("compute root signature must build");

    let mut output = HostPage::new(16 * 1024);
    let zeros = vec![0u8; (THREADS as usize) * 4];
    output.write(0, &zeros);
    // SAFETY: `output` is a live 16 KiB page that outlives the submission.
    let out_id = unsafe { transport.register_zerocopy(output.raw(), 16 * 1024) }
        .expect("UAV page must register");

    let pipeline_state = victory_metal::macos::registry::PipelineState::default()
        .encode(PipelineStage::Compute);
    const HANDLE: u64 = 0x5000;
    transport
        .register_content_hashed_compute_pipeline(
            HANDLE,
            &metallib,
            "bind_kernel",
            &pipeline_state,
            PipelineAbi::compute(layout),
        )
        .expect("content-hashed compute pipeline must register");

    let mut snapshot = HashMap::new();
    snapshot.insert((1u64, 1u32), out_id);

    let submission = SubmissionV3 {
        sequence: 102,
        list: None,
        commands: vec![
            CommandV3::SetComputePipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: float4_bytes([7.0, 0.0, 0.0, 0.0]),
            }),
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 1,
                allocation_id: 1,
                generation: 1,
                offset: 0,
                length: (THREADS as u64) * 4,
            }),
            CommandV3::Dispatch1d(victory_metal::wire::Dispatch1d {
                pipeline_id: HANDLE,
                threads: THREADS,
                threads_per_group: 32,
                bindings: Vec::new(),
            }),
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("compute submission must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    assert_eq!(stats.compute_encoders, 1);

    // The submission is asynchronous: the shared page holds the kernel's output
    // only after the GPU completed it.
    transport
        .wait_for_gpu()
        .expect("the compute submission should complete");

    // SAFETY: the shared page holds `THREADS` readable u32 words that Metal wrote.
    for index in 0..THREADS as usize {
        let actual = unsafe { *output.ptr.cast::<u32>().add(index) };
        assert_eq!(actual, 7 + index as u32, "compute dword {index}");
    }

    transport
        .unregister_buffer(out_id)
        .expect("unregister the UAV page");
    eprintln!(
        "compute ABI proof on {}: exact dwords 7..{}",
        info.name_str(),
        7 + THREADS - 1
    );
}

/// Unknown texture and unknown pipeline handles are refused with
/// `UnknownHandle`, zero encoders and an unchanged target.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn unknown_handles_are_refused_without_encoders() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let allocations = register_mrt_targets(&transport);

    // Establish known content on target 0 through a valid clear-only pass.
    let clear_pass = SubmissionV3 {
        sequence: 103,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.1, 0.2, 0.3, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, _) = transport
        .submit_v3_with_stats(&clear_pass, &allocations)
        .expect("clear-only pass must execute");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    transport
        .wait_for_gpu()
        .expect("the clear-only pass should complete");
    let cleared = transport
        .read_registered_texture_rgba8(0x1000)
        .expect("target 0 readback before refusal");
    assert_eq!(cleared.len(), PIXELS);

    // Unknown pipeline handle inside a pass.
    let unknown_pipeline = SubmissionV3 {
        sequence: 104,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: 0xDEAD_BEEF,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&unknown_pipeline, &allocations)
        .expect("an unknown pipeline must still return a reply");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::UnknownHandle);
    assert_eq!(stats, EncoderStats::default(), "no encoder may be created");

    // Unknown texture handle: the allocation name resolves to a handle that was
    // never registered.
    let mut bad_allocations = allocations.clone();
    bad_allocations.insert((9u64, 1u32), VmBufferId(0xDEAD));
    let unknown_texture = SubmissionV3 {
        sequence: 105,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(9, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&unknown_texture, &bad_allocations)
        .expect("an unknown texture must still return a reply");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::UnknownHandle);
    assert_eq!(stats, EncoderStats::default(), "no encoder may be created");

    let after = transport
        .read_registered_texture_rgba8(0x1000)
        .expect("target 0 readback after refusal");
    assert_eq!(
        cleared, after,
        "a refused submission must not modify the target"
    );

    eprintln!(
        "unknown-handle proof on {}: pixels unchanged={}, refused pipeline + texture",
        info.name_str(),
        after.len()
    );
}
/// Reflected graphics ABI whose *flattened* parameter order is not the title's
/// root-parameter order: the translated user-data block is parameter 0 and the
/// root CBV is parameter 1, at `(space 1, register 0)` — the translated
/// root-CBV space.
///
/// This is the Boot #27 shape: a title's root parameter 0 is its root CBV, but
/// flattened slot 0 is a 32-bit-constants parameter, so a wire-slot lookup binds
/// the wrong parameter and refuses the view.
fn translated_order_abi() -> PipelineAbi {
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
                kind: RootParameterKind::ConstantBufferView,
                size: 8,
                register: 0,
                space: 1,
            },
        ],
    )
    .expect("fragment root signature must build");
    PipelineAbi::graphics(None, Some(layout))
}

/// One plan record built with the documented `VRAB` version 2 layout, which the
/// host and guest share without a crate dependency.
fn root_plan_record(root_parameter: u16, space: u16, kind: u8, register: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&root_parameter.to_le_bytes());
    bytes.extend_from_slice(&space.to_le_bytes());
    bytes.push(kind);
    bytes.extend_from_slice(&register.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes
}

fn encoded_root_plan(records: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"VRAB");
    bytes.push(2);
    bytes.extend_from_slice(&(records.len() as u16).to_le_bytes());
    for record in records {
        bytes.extend_from_slice(record);
    }
    bytes
}

/// A title root-parameter index is not a translated slot: title slot 0 (the
/// root CBV) binds the translated `(space 1, register 0)` parameter through the
/// guest's root-argument plan, and the pass plus draw encode with no ABI
/// refusal.
///
/// The same submission without a plan reproduces Boot #27: the legacy wire-slot
/// lookup resolves slot 0 to the flattened constants parameter and refuses the
/// view with `root buffer view names a non-buffer root parameter`.
#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn a_planned_root_view_binds_the_translated_parameter_and_draws() {
    let dir = TempDir::new("submit-v3-root-plan");
    let Some(metallib) = compile_metallib(dir.path(), "graphics", GRAPHICS_MSL) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; plan proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");
    let allocations = register_mrt_targets(&transport);

    let mut cbv = HostPage::new(16 * 1024);
    cbv.write(0, &float4_bytes([0.0, 128.0 / 255.0, 1.0, 1.0]));
    // SAFETY: `cbv` is a live 16 KiB page that outlives the submission.
    let cbv_id = unsafe { transport.register_zerocopy(cbv.raw(), 16 * 1024) }
        .expect("root CBV page must register");

    let pipeline_state = victory_metal::macos::registry::PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm],
        depth_format: Some(TextureFormat::Depth32Float),
        fragment_entry: Some("bind_fragment".to_string()),
        ..Default::default()
    }
    .encode(PipelineStage::Graphics);

    const HANDLE: u64 = 0x5000;
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE,
            &metallib,
            "bind_vertex",
            &pipeline_state,
            translated_order_abi(),
        )
        .expect("content-hashed graphics pipeline must register");

    let mut snapshot = allocations.clone();
    snapshot.insert((4u64, 1u32), cbv_id);

    // The title's root parameters: 5 is the constants block, 0 the root CBV.
    // (One title index never names two different arguments, so the constants
    // stay at 5.)
    let title_commands = || {
        vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![
                    color_attachment(1, [0.0, 0.0, 0.0, 1.0]),
                    color_attachment(2, [0.0, 0.0, 0.0, 1.0]),
                ],
                depth_stencil: Some(DepthStencilAttachment {
                    allocation_id: 3,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    depth_load: LoadOp::Clear,
                    depth_store: StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load: LoadOp::DontCare,
                    stencil_store: StoreOp::DontCare,
                    clear_stencil: 0,
                }),
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::SetViewport(full_viewport()),
            CommandV3::SetScissor(full_scissor()),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            // Title root parameter 5 is the constants block; the translated
            // signature puts it at flattened parameter 0.
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 5,
                bytes: float4_bytes([1.0, 128.0 / 255.0, 0.0, 1.0]),
            }),
            // Title root parameter 0 is the root CBV; the translated signature
            // declares it at flattened parameter 1, (space 1, register 0).
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 0,
                allocation_id: 4,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ]
    };

    // Control: no plan, so the legacy wire-slot lookup resolves the view's
    // slot 0 to the flattened constants parameter and refuses it. The control's
    // own constants are bound at slot 0 for the same reason a title's root
    // parameter 0 was: this is the Boot #27 refusal, reproduced without a
    // title.
    let mut legacy_commands = title_commands();
    legacy_commands[5] = CommandV3::SetRootConstants(RootConstantsV3 {
        slot: 0,
        bytes: float4_bytes([1.0, 128.0 / 255.0, 0.0, 1.0]),
    });
    let unplanned = SubmissionV3 {
        sequence: 111,
        list: Some(ListIdentity {
            list_id: 21,
            list_generation: 1,
        }),
        commands: legacy_commands,
    };
    let refusals_before = transport.abi_refusal_count();
    let error = transport
        .submit_v3_with_stats(&unplanned, &snapshot)
        .expect_err("a slot-only root view must not bind a constants parameter");
    assert!(
        error.to_string().contains("non-buffer root parameter"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        transport.abi_refusal_count(),
        refusals_before + 1,
        "the legacy mismatch is one counted ABI refusal"
    );

    // The plan names the translated identity of both title root parameters.
    let plan = victory_metal::root_arguments::decode_root_plan(&encoded_root_plan(&[
        root_plan_record(5, 0, 0, 0),
        root_plan_record(0, 1, 1, 0),
    ]))
    .expect("the plan decodes");

    let submission = SubmissionV3 {
        sequence: 112,
        list: Some(ListIdentity {
            list_id: 22,
            list_generation: 1,
        }),
        commands: title_commands(),
    };
    let refusals_before = transport.abi_refusal_count();
    let legacy_before = transport.legacy_root_binding_count();
    let draws_before = transport.encode_counters().draws;
    let (reply, stats) = transport
        .submit_v3_with_stats_and_plan(&submission, &snapshot, Some(&plan))
        .expect("the planned submission must execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "the planned submission status"
    );
    assert_eq!(
        transport.abi_refusal_count(),
        refusals_before,
        "a planned binding is not an ABI refusal"
    );
    assert_eq!(
        transport.legacy_root_binding_count(),
        legacy_before,
        "every planned root argument resolved by its translated identity"
    );
    assert_eq!(stats.render_encoders, 1, "the pass encoded");
    assert_eq!(
        transport.encode_counters().draws,
        draws_before + 1,
        "the draw encoded through the harness"
    );
    transport.wait_for_gpu().expect("the pass should complete");

    let target0 = transport
        .read_registered_texture_rgba8(0x1000)
        .expect("target 0 readback");
    let target1 = transport
        .read_registered_texture_rgba8(0x1001)
        .expect("target 1 readback");
    assert_uniform(&target0, TARGET0_RGBA, "target 0");
    assert_uniform(&target1, TARGET1_RGBA, "target 1");

    transport
        .unregister_buffer(cbv_id)
        .expect("unregister the CBV page");

    eprintln!(
        "root-plan proof on {}: title slot 0 -> translated (space 1, register 0); \
         target0={:?}, target1={:?}, draws_encoded=+1, abi_refusals=0, \
         legacy_bindings=0",
        info.name_str(),
        target0[0],
        target1[0]
    );
}
