//! Hardware proof for ADR-061: the guest `D3D12_INPUT_LAYOUT` carried in
//! `VMVD` v2 reaches `MTLVertexDescriptor` unchanged.
//!
//! The correct layout puts attribute 0 in vertex buffer 0 and attribute 1 in
//! vertex buffer 1, with a non-zero offset in buffer 1 and a padded stride in
//! both. The same shader with the old host-invented "one packed buffer 0"
//! descriptor reads the wrong stream and produces a different pixel, so the
//! wrong descriptor cannot pass this proof.
//!
//! The second proof carries a per-instance slot: two instances translate one
//! half-size quad left and right, so the exact pixels prove `setStepFunction:`
//! and `setStepRate:` reached Metal; the same data under a per-vertex slot is
//! the negative control.
//!
//! Run serialised; the test itself does not take the GPU lock (the harness
//! does):
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test guest_vertex_layout_gpu -- --test-threads=1 --nocapture
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
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::macos::vertex_layout::{
    VERTEX_LAYOUT_VERSION, VertexAttribute, VertexFormat, VertexLayout, VertexSlot,
    VertexStepFunction,
};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DrawIndexedState, IndexBufferBinding, IndexFormat,
    LoadOp, PipelineRef, PrimitiveTopology, ScissorRect, StoreOp, SubmissionV3, VertexBuffer,
    VertexBufferSet, Viewport,
};

const EDGE: u32 = 64;
const PIXELS: usize = (EDGE * EDGE) as usize;
const STRIDE: u32 = 32;
/// The old host-derived packed stride for float2 + float4.
const PACKED_STRIDE: u32 = 24;

const TARGET_CORRECT: u64 = 0x2300;
const TARGET_PACKED: u64 = 0x2301;
const TARGET_INSTANCE: u64 = 0x2302;
const TARGET_INSTANCE_VERTEX: u64 = 0x2303;

const GREEN: [u8; 4] = [0, 255, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
const CLEAR_BLUE: [u8; 4] = [0, 0, 255, 255];

const HANDLE_CORRECT: u64 = 0x6300;
const HANDLE_PACKED: u64 = 0x6301;
const HANDLE_INSTANCE: u64 = 0x6302;
const HANDLE_INSTANCE_VERTEX: u64 = 0x6303;

/// Allocation ids in the wire snapshots.
const TARGET_CORRECT_ALLOCATION: u64 = 1;
const SLOT0_ALLOCATION: u64 = 2;
const SLOT1_ALLOCATION: u64 = 3;
const TARGET_PACKED_ALLOCATION: u64 = 4;
const INDEX_ALLOCATION: u64 = 5;
const PACKED_ALLOCATION: u64 = 6;
const TARGET_INSTANCE_ALLOCATION: u64 = 7;
const TARGET_INSTANCE_VERTEX_ALLOCATION: u64 = 8;
const POSITIONS_ALLOCATION: u64 = 9;
const INSTANCES_ALLOCATION: u64 = 10;
const INSTANCE_INDEX_ALLOCATION: u64 = 11;

/// Per-vertex position stride for the instanced proof (one `float2`).
const POSITION_STRIDE: u32 = 8;
/// Per-instance data stride for the instanced proof (one `float4`:
/// translation x, y, tint, unused).
const INSTANCE_STRIDE: u32 = 16;

/// Attribute 0 is a `float2` in vertex buffer 0 at offset 0; attribute 1 is a
/// `float4` in vertex buffer 1 at offset 8. The fragment forces opaque alpha,
/// so every result is an exact byte pattern.
const GUEST_LAYOUT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VIn {
    float2 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VOut {
    float4 position [[position]];
    float4 color [[flat]];
};

vertex VOut gv_vertex(VIn in [[stage_in]]) {
    VOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 gv_fragment(VOut in [[stage_in]]) {
    return float4(in.color.rgb, 1.0);
}
"#;

/// The instanced proof: `attribute(0)` is a `float2` clip-space position
/// (per-vertex, buffer 0) and `attribute(1)` is one `float4` per instance
/// (per-instance, buffer 1) holding `(translate x, translate y, tint, unused)`.
/// The quad is half-size, so instance 0 covers the left half and instance 1 the
/// right half exactly; a per-vertex step would read one instance element per
/// vertex and could not produce those two halves.
const INSTANCE_LAYOUT_MSL: &str = r#"
struct IIn {
    float2 position [[attribute(0)]];
    float4 instance [[attribute(1)]];
};

struct IOut {
    float4 position [[position]];
    float4 color [[flat]];
};

vertex IOut gvi_vertex(IIn in [[stage_in]]) {
    IOut out;
    out.position = float4(in.position * 0.5 + in.instance.xy, 0.0, 1.0);
    out.color = float4(in.instance.z, 1.0 - in.instance.z, 0.0, 1.0);
    return out;
}

fragment float4 gvi_fragment(IOut in [[stage_in]]) {
    return in.color;
}
"#;

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

/// Compiles the MSL above into a real `.metallib`, or `None` without Xcode.
fn compile_metallib(dir: &Path) -> Option<Vec<u8>> {
    let metal = tool_path("metal")?;
    let metallib = tool_path("metallib")?;
    let source = dir.join("guest-layout.metal");
    let air = dir.join("guest-layout.air");
    let library = dir.join("guest-layout.metallib");
    fs::write(
        &source,
        format!("{GUEST_LAYOUT_MSL}\n{INSTANCE_LAYOUT_MSL}"),
    )
    .expect("write MSL source");
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

fn float_format(components: u8) -> VertexFormat {
    VertexFormat::with_components(VertexFormat::FLOAT, components).unwrap()
}

/// The guest layout the test carries: two vertex buffers, a non-zero offset in
/// the second, padded stride 32 in both.
fn correct_layout() -> VertexLayout {
    VertexLayout::from_attributes_and_slots(
        vec![
            VertexAttribute {
                index: 0,
                format: float_format(2),
                offset: 0,
                buffer_index: 0,
                stride: STRIDE,
            },
            VertexAttribute {
                index: 1,
                format: float_format(4),
                offset: 8,
                buffer_index: 1,
                stride: STRIDE,
            },
        ],
        vec![
            VertexSlot {
                buffer_index: 0,
                stride: STRIDE,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
            },
            VertexSlot {
                buffer_index: 1,
                stride: STRIDE,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
            },
        ],
    )
    .expect("two-slot guest layout")
}

/// The old host-invented shape: every attribute in packed vertex buffer 0 with
/// one packed stride.
fn packed_buffer_zero_layout() -> VertexLayout {
    VertexLayout::from_attributes(vec![
        VertexAttribute {
            index: 0,
            format: float_format(2),
            offset: 0,
            buffer_index: 0,
            stride: PACKED_STRIDE,
        },
        VertexAttribute {
            index: 1,
            format: float_format(4),
            offset: 8,
            buffer_index: 0,
            stride: PACKED_STRIDE,
        },
    ])
    .expect("packed buffer-0 layout")
}

/// Encodes a graphics state and appends the guest chunk exactly as
/// `victory-gpu` does, then decodes it back to prove the wire path.
fn encoded_state(layout: &VertexLayout) -> Vec<u8> {
    let mut state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        fragment_entry: Some("gv_fragment".to_string()),
        ..PipelineState::default()
    }
    .encode(PipelineStage::Graphics);
    let chunk = layout.encode();
    assert_eq!(chunk[4], VERTEX_LAYOUT_VERSION, "VMVD version");
    let decoded = VertexLayout::decode(&chunk).expect("the guest chunk decodes");
    assert_eq!(&decoded, layout);
    layout.append_to(&mut state);
    let decoded_state =
        PipelineState::decode_for(&state, PipelineStage::Graphics).expect("the state decodes");
    assert_eq!(
        decoded_state.vertex_layout.as_ref(),
        Some(layout),
        "the appended guest layout survives the state decode"
    );
    state
}

fn rgba8_desc() -> TextureDesc {
    TextureDesc {
        width: EDGE,
        height: EDGE,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::color_attachment(),
        label: Some("guest-vertex-layout-colour".to_string()),
    }
}

fn color_attachment(allocation_id: u64) -> ColorAttachment {
    ColorAttachment {
        allocation_id,
        generation: 1,
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: [0.0, 0.0, 1.0, 1.0],
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
        // SAFETY: `self.ptr` came from `posix_memalign` and is unregistered by
        // the time this guard drops.
        unsafe { free(self.ptr) };
    }
}

fn assert_uniform(pixels: &[[u8; 4]], expected: [u8; 4], label: &str) {
    assert_eq!(pixels.len(), PIXELS, "{label} pixel count");
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(*pixel, expected, "{label} pixel {index}");
    }
}

/// One full-viewport tri draw with the supplied vertex buffers bound from slot
/// 0.
fn draw_commands(
    pipeline_id: u64,
    target_allocation: u64,
    buffers: Vec<VertexBuffer>,
) -> Vec<CommandV3> {
    vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color_attachment(target_allocation)],
            depth_stencil: None,
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetScissor(full_scissor()),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers,
        }),
        CommandV3::SetIndexBuffer(IndexBufferBinding {
            allocation_id: INDEX_ALLOCATION,
            generation: 1,
            offset: 0,
            length: 6,
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
    ]
}

/// Interleaved stride-32 vertices whose float2 position sits at offset 0.
fn positions(stride: usize) -> Vec<u8> {
    let corners = [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let mut data = Vec::with_capacity(3 * stride);
    for corner in corners {
        let mut vertex = vec![0u8; stride];
        for (component, value) in corner.iter().enumerate() {
            vertex[component * 4..component * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        data.extend_from_slice(&vertex);
    }
    data
}

/// The guest layout the instanced proof carries: `attribute(0)` per-vertex in
/// buffer 0 and `attribute(1)` in buffer 1 with `step_function`/rate 1.
fn instance_layout(step_function: VertexStepFunction) -> VertexLayout {
    VertexLayout::from_attributes_and_slots(
        vec![
            VertexAttribute {
                index: 0,
                format: float_format(2),
                offset: 0,
                buffer_index: 0,
                stride: POSITION_STRIDE,
            },
            VertexAttribute {
                index: 1,
                format: float_format(4),
                offset: 0,
                buffer_index: 1,
                stride: INSTANCE_STRIDE,
            },
        ],
        vec![
            VertexSlot {
                buffer_index: 0,
                stride: POSITION_STRIDE,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
            },
            VertexSlot {
                buffer_index: 1,
                stride: INSTANCE_STRIDE,
                step_function,
                step_rate: 1,
            },
        ],
    )
    .expect("a two-slot layout with one stepped slot")
}

/// The exact pixel pattern two per-instance translations must produce: the
/// half-size quad is translated left by instance 0 and right by instance 1, so
/// columns 0..32 are GREEN, 32..64 RED and the rest keeps the CLEAR_BLUE clear.
fn instance_expected_pixels() -> Vec<[u8; 4]> {
    let mut expected = Vec::with_capacity(PIXELS);
    for row in 0..EDGE as usize {
        for column in 0..EDGE as usize {
            let color = if !(16..48).contains(&row) {
                CLEAR_BLUE
            } else if column < 32 {
                GREEN
            } else {
                RED
            };
            expected.push(color);
        }
    }
    expected
}

/// One full-viewport instanced draw of the half-size quad (six indices),
/// instance 0 translating left and instance 1 translating right.
fn instance_draw_commands(
    pipeline_id: u64,
    target_allocation: u64,
    buffers: Vec<VertexBuffer>,
) -> Vec<CommandV3> {
    vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color_attachment(target_allocation)],
            depth_stencil: None,
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
        CommandV3::SetViewport(full_viewport()),
        CommandV3::SetScissor(full_scissor()),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        CommandV3::SetVertexBuffers(VertexBufferSet {
            start_slot: 0,
            buffers,
        }),
        CommandV3::SetIndexBuffer(IndexBufferBinding {
            allocation_id: INSTANCE_INDEX_ALLOCATION,
            generation: 1,
            offset: 0,
            length: 12,
            format: IndexFormat::Uint16,
        }),
        CommandV3::DrawIndexed(DrawIndexedState {
            index_count: 6,
            instance_count: 2,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
    ]
}

#[test]
fn guest_multi_slot_layout_draws_exact_pixels_and_the_packed_zero_layout_cannot() {
    let dir = TempDir::new("guest-vertex-layout");
    let Some(metallib) = compile_metallib(dir.path()) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for target in [TARGET_CORRECT, TARGET_PACKED] {
        transport
            .register_registry_texture(target, rgba8_desc())
            .expect("registry colour target must create");
    }

    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE_CORRECT,
            &metallib,
            "gv_vertex",
            &encoded_state(&correct_layout()),
            PipelineAbi::graphics(None, None),
        )
        .expect("the guest two-slot descriptor must let Metal create the pipeline");
    transport
        .register_content_hashed_graphics_pipeline(
            HANDLE_PACKED,
            &metallib,
            "gv_vertex",
            &encoded_state(&packed_buffer_zero_layout()),
            PipelineAbi::graphics(None, None),
        )
        .expect("the packed buffer-0 descriptor is itself a valid Metal layout");

    // Correct path: slot 0 is the float2 position stream (stride 32); slot 1
    // is the float4 colour at offset 8 (8-byte prefix padding, 8-byte suffix).
    let mut slot0_page = HostPage::new(16 * 1024);
    slot0_page.write(0, &positions(STRIDE as usize));
    // SAFETY: `slot0_page` is a live 16 KiB page that outlives the submission.
    let slot0_id = unsafe { transport.register_zerocopy(slot0_page.raw(), 16 * 1024) }
        .expect("slot 0 page must register");

    let mut slot1 = Vec::with_capacity(3 * STRIDE as usize);
    for _ in 0..3 {
        let mut vertex = vec![0u8; STRIDE as usize];
        for (component, value) in [0.0f32, 1.0, 0.0, 1.0].iter().enumerate() {
            let at = 8 + component * 4;
            vertex[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        slot1.extend_from_slice(&vertex);
    }
    let mut slot1_page = HostPage::new(16 * 1024);
    slot1_page.write(0, &slot1);
    // SAFETY: `slot1_page` is a live 16 KiB page that outlives the submission.
    let slot1_id = unsafe { transport.register_zerocopy(slot1_page.raw(), 16 * 1024) }
        .expect("slot 1 page must register");

    // Negative control: the packed buffer-0 stream. Position float2 at 0 and
    // colour float4 at 8 within 24 bytes; the old descriptor reads RED, so it
    // can never produce the correct GREEN.
    let mut packed = Vec::with_capacity(3 * PACKED_STRIDE as usize);
    for corner in [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]] {
        let mut vertex = vec![0u8; PACKED_STRIDE as usize];
        for (component, value) in corner.iter().enumerate() {
            vertex[component * 4..component * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (component, value) in [1.0f32, 0.0, 0.0, 1.0].iter().enumerate() {
            let at = 8 + component * 4;
            vertex[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        packed.extend_from_slice(&vertex);
    }
    let mut packed_page = HostPage::new(16 * 1024);
    packed_page.write(0, &packed);
    // SAFETY: `packed_page` is a live 16 KiB page that outlives the submission.
    let packed_id = unsafe { transport.register_zerocopy(packed_page.raw(), 16 * 1024) }
        .expect("packed page must register");

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
    snapshot.insert(
        (TARGET_CORRECT_ALLOCATION, 1u32),
        VmBufferId(TARGET_CORRECT),
    );
    snapshot.insert((SLOT0_ALLOCATION, 1u32), slot0_id);
    snapshot.insert((SLOT1_ALLOCATION, 1u32), slot1_id);
    snapshot.insert((TARGET_PACKED_ALLOCATION, 1u32), VmBufferId(TARGET_PACKED));
    snapshot.insert((INDEX_ALLOCATION, 1u32), index_id);
    snapshot.insert((PACKED_ALLOCATION, 1u32), packed_id);

    let correct_buffers = vec![
        VertexBuffer {
            allocation_id: SLOT0_ALLOCATION,
            generation: 1,
            offset: 0,
            stride: STRIDE,
            length: 3 * STRIDE as u64,
        },
        VertexBuffer {
            allocation_id: SLOT1_ALLOCATION,
            generation: 1,
            offset: 0,
            stride: STRIDE,
            length: 3 * STRIDE as u64,
        },
    ];
    let packed_buffers = vec![VertexBuffer {
        allocation_id: PACKED_ALLOCATION,
        generation: 1,
        offset: 0,
        stride: PACKED_STRIDE,
        length: 3 * PACKED_STRIDE as u64,
    }];

    for (sequence, pipeline, target, buffers, label) in [
        (
            211u32,
            HANDLE_CORRECT,
            TARGET_CORRECT_ALLOCATION,
            correct_buffers,
            "guest two-slot",
        ),
        (
            212,
            HANDLE_PACKED,
            TARGET_PACKED_ALLOCATION,
            packed_buffers,
            "packed buffer 0",
        ),
    ] {
        let submission = SubmissionV3 {
            sequence,
            list: None,
            commands: draw_commands(pipeline, target, buffers),
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &snapshot)
            .expect("the vertex-layout pass must execute");
        assert_eq!(
            reply.status,
            victory_metal::wire::ReplyStatus::Ok,
            "{label} submission status"
        );
        assert_eq!(stats.render_encoders, 1, "{label}: one render encoder");
        transport
            .wait_for_gpu()
            .expect("the vertex-layout pass should complete");
    }

    let correct = transport
        .read_registered_texture_rgba8(TARGET_CORRECT)
        .expect("correct target readback");
    // GREEN proves the float4 came from slot 1 at offset 8, not from slot 0.
    assert_uniform(&correct, GREEN, "guest two-slot target");

    let packed = transport
        .read_registered_texture_rgba8(TARGET_PACKED)
        .expect("packed target readback");
    // The packed buffer-0 descriptor reads the RED stream from buffer 0.
    assert_uniform(&packed, RED, "packed buffer-0 target");
    assert_ne!(packed[0], correct[0], "the wrong descriptor cannot pass");

    for id in [slot0_id, slot1_id, packed_id, index_id] {
        transport
            .unregister_buffer(id)
            .expect("unregister a test page");
    }
    eprintln!(
        "guest vertex-layout proof on {}: {} pixels = {:?}, packed buffer 0 = {:?}, stride {}",
        info.name_str(),
        correct.len(),
        correct[0],
        packed[0],
        STRIDE
    );
}

/// ADR-061: a per-instance slot must reach `setStepFunction:`/`setStepRate:`.
///
/// Two instances translate one half-size quad by `(-0.5, 0)` and `(+0.5, 0)`,
/// so the correct descriptor covers the left and right halves exactly with two
/// uniform colours. The same data under a per-vertex slot reads one instance
/// element per vertex and cannot reproduce that pattern, so a host that lost
/// the step function cannot pass this proof.
#[test]
fn guest_per_instance_slot_steps_once_per_instance() {
    let dir = TempDir::new("guest-instance-layout");
    let Some(metallib) = compile_metallib(dir.path()) else {
        eprintln!("PARTIAL: Apple Metal toolchain (metal/metallib) not found; proof skipped");
        return;
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    for target in [TARGET_INSTANCE, TARGET_INSTANCE_VERTEX] {
        transport
            .register_registry_texture(target, rgba8_desc())
            .expect("registry colour target must create");
    }
    for (handle, layout) in [
        (
            HANDLE_INSTANCE,
            instance_layout(VertexStepFunction::PerInstance),
        ),
        (
            HANDLE_INSTANCE_VERTEX,
            instance_layout(VertexStepFunction::PerVertex),
        ),
    ] {
        transport
            .register_content_hashed_graphics_pipeline(
                handle,
                &metallib,
                "gvi_vertex",
                &encoded_state(&layout),
                PipelineAbi::graphics(None, None),
            )
            .expect("the stepped descriptor must let Metal create the pipeline");
    }

    // Vertex stream: a 2x2 quad in clip space, one float2 per vertex.
    let mut positions = Vec::new();
    for corner in [[-1.0f32, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]] {
        positions.extend_from_slice(&corner[0].to_le_bytes());
        positions.extend_from_slice(&corner[1].to_le_bytes());
    }
    let mut positions_page = HostPage::new(16 * 1024);
    positions_page.write(0, &positions);
    // SAFETY: `positions_page` is a live 16 KiB page outliving the submission.
    let positions_id = unsafe { transport.register_zerocopy(positions_page.raw(), 16 * 1024) }
        .expect("position page must register");

    // Instance stream: entry 0 translates left (GREEN), entry 1 right (RED).
    // Entries 2 and 3 exist so the per-vertex control reads defined data.
    let mut instances = Vec::new();
    for entry in [
        [-0.5f32, 0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5, 1.0],
    ] {
        for value in entry {
            instances.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut instances_page = HostPage::new(16 * 1024);
    instances_page.write(0, &instances);
    // SAFETY: `instances_page` is a live 16 KiB page outliving the submission.
    let instances_id = unsafe { transport.register_zerocopy(instances_page.raw(), 16 * 1024) }
        .expect("instance page must register");

    let mut index_page = HostPage::new(16 * 1024);
    let mut index_bytes = Vec::new();
    for index in [0u16, 1, 2, 2, 1, 3] {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }
    index_page.write(0, &index_bytes);
    // SAFETY: `index_page` is a live 16 KiB page outliving the submission.
    let index_id = unsafe { transport.register_zerocopy(index_page.raw(), 16 * 1024) }
        .expect("index page must register");

    let mut snapshot = HashMap::new();
    snapshot.insert(
        (TARGET_INSTANCE_ALLOCATION, 1u32),
        VmBufferId(TARGET_INSTANCE),
    );
    snapshot.insert(
        (TARGET_INSTANCE_VERTEX_ALLOCATION, 1u32),
        VmBufferId(TARGET_INSTANCE_VERTEX),
    );
    snapshot.insert((POSITIONS_ALLOCATION, 1u32), positions_id);
    snapshot.insert((INSTANCES_ALLOCATION, 1u32), instances_id);
    snapshot.insert((INSTANCE_INDEX_ALLOCATION, 1u32), index_id);

    let buffers = vec![
        VertexBuffer {
            allocation_id: POSITIONS_ALLOCATION,
            generation: 1,
            offset: 0,
            stride: POSITION_STRIDE,
            length: 4 * u64::from(POSITION_STRIDE),
        },
        VertexBuffer {
            allocation_id: INSTANCES_ALLOCATION,
            generation: 1,
            offset: 0,
            stride: INSTANCE_STRIDE,
            length: 4 * u64::from(INSTANCE_STRIDE),
        },
    ];

    for (sequence, pipeline, target, label) in [
        (
            213u32,
            HANDLE_INSTANCE,
            TARGET_INSTANCE_ALLOCATION,
            "per-instance",
        ),
        (
            214,
            HANDLE_INSTANCE_VERTEX,
            TARGET_INSTANCE_VERTEX_ALLOCATION,
            "per-vertex control",
        ),
    ] {
        let submission = SubmissionV3 {
            sequence,
            list: None,
            commands: instance_draw_commands(pipeline, target, buffers.clone()),
        };
        let (reply, stats) = transport
            .submit_v3_with_stats(&submission, &snapshot)
            .expect("the instanced layout pass must execute");
        assert_eq!(
            reply.status,
            victory_metal::wire::ReplyStatus::Ok,
            "{label} submission status"
        );
        assert_eq!(stats.render_encoders, 1, "{label}: one render encoder");
        transport
            .wait_for_gpu()
            .expect("the instanced pass should complete");
    }

    let expected = instance_expected_pixels();
    let stepped = transport
        .read_registered_texture_rgba8(TARGET_INSTANCE)
        .expect("per-instance target readback");
    assert_eq!(stepped.len(), PIXELS, "per-instance pixel count");
    for (index, pixel) in stepped.iter().enumerate() {
        assert_eq!(
            *pixel,
            expected[index],
            "per-instance pixel {index} (column {}, row {})",
            index % EDGE as usize,
            index / EDGE as usize
        );
    }

    let per_vertex = transport
        .read_registered_texture_rgba8(TARGET_INSTANCE_VERTEX)
        .expect("per-vertex control readback");
    let differing = per_vertex
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    assert!(
        differing > 0,
        "the per-vertex control must not reproduce the two-translation halves"
    );

    for id in [positions_id, instances_id, index_id] {
        transport
            .unregister_buffer(id)
            .expect("unregister a test page");
    }
    eprintln!(
        "per-instance proof on {}: {} pixels match the two translations exactly, per-vertex control differs in {differing} pixels",
        info.name_str(),
        stepped.len()
    );
}
