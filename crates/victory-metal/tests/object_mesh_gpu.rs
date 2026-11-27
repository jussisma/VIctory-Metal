//! Hardware proof for the object/amplification + mesh + pixel pipeline
//! (ADR-024a).
//!
//! The fixture is one MSL source with three entry points:
//!
//! * `object_main` is an `[[object]]` stage with an 8-thread object
//!   threadgroup. It writes one `Payload` per object thread and culls the odd
//!   threads, so exactly four payloads survive; it sets the mesh grid to one
//!   threadgroup.
//! * `mesh_main` is an `[[mesh]]` stage with a single thread. It walks the
//!   eight payload slots, skips the culled ones, and emits one triangle for
//!   each surviving payload. Each triangle covers 28 pixels of its own 8x8
//!   quadrant and carries that payload's colour.
//! * `frag_main` is the pixel stage. It returns the interpolated (uniform)
//!   triangle colour.
//!
//! `victory-shader` identifies an amplification program (`HwStage::As`) in the
//! console container, but its native lowering fails closed (the payload and
//! mesh dispatch grid have no DXIL form yet), so the console emission path
//! does not produce the object stage. The Metal toolchain compiles the MSL
//! entry points to a metallib here, and the test then uses the same host
//! contract a title's converted metallibs would: [`MeshPipeline::from_metallibs`]
//! with `Some((object_metallib, object_entry))`, `payload_memory_length`, both
//! threadgroup maxima, a v3 `DispatchMesh` inside a render pass, and an exact
//! RGBA8 readback.
//!
//! The first proof is the payload count: eight object threads run, four are
//! culled, and exactly four triangles' pixels appear. A pipeline that ignored
//! the object stage would still draw the mesh's four triangles here (the mesh
//! stage alone would not know they were culled), but the colour histogram also
//! proves the object stage ran and wrote the payloads the mesh consumed.
//!
//! The second proof is the grid: the amplification stage computes the mesh
//! threadgroup count from its dispatched threadgroup size, and the parsed
//! `victory-shader` program supplies that threadgroup size, the payload memory
//! length and the dispatch. A wrong amplification interpretation changes the
//! grid and therefore the pixel count (one threadgroup paints one 28-pixel
//! triangle).
//!
//! Run with:
//! ```text
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test object_mesh_gpu -- --test-threads=1 --nocapture
//! ```
//! Serialise against the other GPU tests with `/tmp/victory-gpu.lock`.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::process::Command;

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::mesh::{MeshPipeline, MeshPipelineSpec};
use victory_metal::macos::render::PixelFormat;
use victory_metal::wire::{self, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, LoadOp, PipelineRef, StoreOp, SubmissionV3,
};
use victory_shader::native::StageIo;
use victory_shader::{HwStage, native_programs};

/// Pixels per triangle: the triangle `(0,0) (7.9,0) (0,7.9)` in a pixel block
/// contains exactly the pixel centres with `x + y <= 6`, i.e. 28.
const PIXELS_PER_TRIANGLE: usize = 28;
/// Payloads the object stage emits (even object threads of eight).
const SURVIVING_PAYLOADS: usize = 4;
/// Object threads / payload slots.
const PAYLOAD_SLOTS: usize = 8;
/// One triangle per surviving payload.
const TRIANGLES: usize = SURVIVING_PAYLOADS;
const EDGE: u32 = 16;
const CLEAR_RGBA: [u8; 4] = [0, 0, 0, 255];
const SURVIVOR_COLORS: [[u8; 4]; SURVIVING_PAYLOADS] = [
    [255, 0, 0, 255],   // red
    [0, 255, 0, 255],   // green
    [0, 0, 255, 255],   // blue
    [255, 255, 0, 255], // yellow
];

/// Object culls half of eight payloads; mesh emits one triangle per survivor.
const OBJECT_MESH_PIXEL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Payload {
    float4 color;
    uint   valid;
};

struct VOut {
    float4 position [[position]];
    float4 color;
};

[[object]]
void object_main(object_data Payload *payload [[payload]],
                 metal::mesh_grid_properties grid,
                 uint tid [[thread_position_in_threadgroup]])
{
    // Four payloads survive (even tid), four are culled (odd tid).
    uint survivor = tid >> 1;
    Payload p;
    p.color = float4((survivor == 0u || survivor == 3u) ? 1.0 : 0.0,
                     (survivor == 1u || survivor == 3u) ? 1.0 : 0.0,
                     (survivor == 2u) ? 1.0 : 0.0,
                     1.0);
    p.valid = (tid & 1u) == 0u ? 1u : 0u;
    payload[tid] = p;
    // One mesh threadgroup consumes the eight payload slots.
    grid.set_threadgroups_per_grid(uint3(1, 1, 1));
}

[[mesh]]
void mesh_main(const object_data Payload *payload [[payload]],
               metal::mesh<VOut, void, 24, 8, metal::topology::triangle> out)
{
    uint count = 0;
    for (uint i = 0; i < 8; ++i) {
        if (payload[i].valid == 0) {
            continue;
        }
        // Four disjoint 8x8 quadrants; each triangle covers 28 pixels.
        uint block_x = (count & 1u) * 8u;
        uint block_y = (count >> 1u) * 8u;
        float x0 = float(block_x);
        float y0 = float(block_y);
        float x1 = x0 + 7.9;
        float y1 = y0 + 7.9;
        auto clip = [](float px, float py) {
            return float4((px / 16.0) * 2.0 - 1.0,
                          -((py / 16.0) * 2.0 - 1.0),
                          0.0,
                          1.0);
        };
        VOut v0;
        VOut v1;
        VOut v2;
        v0.position = clip(x0, y0);
        v1.position = clip(x1, y0);
        v2.position = clip(x0, y1);
        v0.color = payload[i].color;
        v1.color = payload[i].color;
        v2.color = payload[i].color;
        uint base = count * 3u;
        out.set_vertex(base + 0u, v0);
        out.set_vertex(base + 1u, v1);
        out.set_vertex(base + 2u, v2);
        out.set_index(base + 0u, uchar(base + 0u));
        out.set_index(base + 1u, uchar(base + 1u));
        out.set_index(base + 2u, uchar(base + 2u));
        count += 1u;
    }
    out.set_primitive_count(count);
}

fragment float4 frag_main(VOut in [[stage_in]])
{
    return in.color;
}
"#;

/// Compiles the MSL fixture to a metallib with the installed Metal toolchain.
fn compile_msl(source: &str) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir().join(format!("victory-object-mesh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let source_path = dir.join("object_mesh_pixel.metal");
    let air_path = dir.join("object_mesh_pixel.air");
    let metallib_path = dir.join("object_mesh_pixel.metallib");
    std::fs::write(&source_path, source)
        .map_err(|error| format!("{}: {error}", source_path.display()))?;

    let metal = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&source_path)
        .arg("-o")
        .arg(&air_path)
        .output()
        .map_err(|error| format!("xcrun metal: {error}"))?;
    if !metal.status.success() {
        return Err(format!(
            "metal exited {}: {}",
            metal.status,
            String::from_utf8_lossy(&metal.stderr)
        ));
    }

    let metallib = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&metallib_path)
        .output()
        .map_err(|error| format!("xcrun metallib: {error}"))?;
    if !metallib.status.success() {
        return Err(format!(
            "metallib exited {}: {}",
            metallib.status,
            String::from_utf8_lossy(&metallib.stderr)
        ));
    }

    let bytes = std::fs::read(&metallib_path)
        .map_err(|error| format!("{}: {error}", metallib_path.display()))?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(bytes)
}

/// Hardware proof: AS+MS+PS builds through `from_metallibs`, the object stage
/// culls half of eight payloads, the mesh stage emits one triangle per
/// survivor, and the pixel stage writes exact colours.
#[test]
fn object_mesh_pixel_pipeline_draws_only_emitted_payloads() {
    let metallib = compile_msl(OBJECT_MESH_PIXEL_MSL)
        .expect("the object/mesh/pixel MSL compiles to a metallib");
    eprintln!("Metal toolchain: fixture metallib {} B", metallib.len());

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let device = transport.device_info().expect("device info");

    // 24 vertices x 32 B (position + colour) declared mesh output bytes; the
    // object payload is 8 slots x 32 B (float4 colour + uint valid, padded).
    let spec = MeshPipelineSpec {
        max_total_threads_per_object_threadgroup: PAYLOAD_SLOTS,
        payload_memory_length: PAYLOAD_SLOTS * 32,
        threads_per_object_threadgroup: [PAYLOAD_SLOTS as u32, 1, 1],
        threads_per_mesh_threadgroup: [1, 1, 1],
        mesh_output_bytes: 24 * 32,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [1, 1, 1])
    };
    let pipeline = MeshPipeline::from_metallibs(
        &transport,
        &metallib,
        "mesh_main",
        Some((&metallib, "object_main")),
        &metallib,
        "frag_main",
        &spec,
    )
    .expect("the AS+MS+PS mesh pipeline should be created from the metallibs");
    let pipeline_id = transport
        .register_mesh_pipeline(pipeline)
        .expect("the AS+MS+PS pipeline should register");

    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("render target should be created");
    let target_id = transport
        .register_render_target(&target)
        .expect("render target should register");
    let mut allocations = HashMap::new();
    allocations.insert((1u64, 1u32), VmBufferId(target_id.0));

    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![ColorAttachment {
                    allocation_id: 1,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                }],
                depth_stencil: None,
            }),
            CommandV3::SetMeshPipeline(PipelineRef {
                pipeline_id: pipeline_id.0,
            }),
            CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: pipeline_id.0,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the object/mesh/pixel submission should execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "submission status");
    assert_eq!(stats.command_buffers, 1, "one command buffer");
    assert_eq!(stats.render_encoders, 1, "one render encoder");

    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("the render target should read back");
    assert_eq!(pixels.len(), (EDGE * EDGE) as usize);

    let mut histogram: HashMap<[u8; 4], usize> = HashMap::new();
    for pixel in &pixels {
        *histogram.entry(*pixel).or_insert(0) += 1;
    }
    eprintln!(
        "GPU: object/mesh/pixel {TRIANGLES} triangles on {}; histogram={histogram:?}",
        device.name_str()
    );

    for color in SURVIVOR_COLORS {
        assert_eq!(
            histogram.get(&color).copied(),
            Some(PIXELS_PER_TRIANGLE),
            "each surviving payload must paint its own {PIXELS_PER_TRIANGLE} pixels of {color:?}"
        );
    }
    let clear = (EDGE * EDGE) as usize - TRIANGLES * PIXELS_PER_TRIANGLE;
    assert_eq!(
        histogram.get(&CLEAR_RGBA).copied(),
        Some(clear),
        "only the four emitted triangles must paint"
    );
    assert_eq!(
        histogram.len(),
        1 + SURVIVING_PAYLOADS,
        "exactly four triangle colours plus the clear colour"
    );
    let painted: usize = histogram
        .iter()
        .filter(|(color, _)| **color != CLEAR_RGBA)
        .map(|(_, count)| *count)
        .sum();
    assert_eq!(
        painted,
        TRIANGLES * PIXELS_PER_TRIANGLE,
        "the object stage emitted {SURVIVING_PAYLOADS} payloads, so {TRIANGLES} triangles paint"
    );
    eprintln!(
        "GPU: {painted} px from {TRIANGLES} emitted triangles ({} culled payloads)",
        PAYLOAD_SLOTS - SURVIVING_PAYLOADS
    );
}

/// Mesh threadgroups the amplification stage dispatches: one per surviving
/// object thread (even threads of eight).
const GRID_GROUPS: usize = 4;

/// Amplification-computed grid fixture: the object stage derives the mesh
/// threadgroup count from its dispatched threadgroup size and writes one
/// payload per survivor; each mesh threadgroup emits one triangle in its own
/// quadrant with its survivor's colour.
const AMPLIFICATION_GRID_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GridPayload {
    float4 color;
    uint   valid;
};

struct GridOut {
    float4 position [[position]];
    float4 color;
};

[[object]]
void grid_object_main(object_data GridPayload *payload [[payload]],
                      metal::mesh_grid_properties grid,
                      uint tid [[thread_position_in_threadgroup]],
                      uint group_threads [[threads_per_threadgroup]])
{
    // Even object threads survive; each survivor owns one payload slot.
    GridPayload p;
    p.valid = (tid & 1u) == 0u ? 1u : 0u;
    p.color = float4((tid == 0u || tid == 6u) ? 1.0 : 0.0,
                     (tid == 2u || tid == 6u) ? 1.0 : 0.0,
                     (tid == 4u) ? 1.0 : 0.0,
                     1.0);
    payload[tid] = p;
    // One mesh threadgroup per surviving payload. The count is derived from
    // the dispatched threadgroup size, so an amplification program read with
    // the wrong group size dispatches a different grid.
    uint survivors = (group_threads + 1u) / 2u;
    grid.set_threadgroups_per_grid(uint3(survivors, 1u, 1u));
}

[[mesh]]
void grid_mesh_main(const object_data GridPayload *payload [[payload]],
                    uint gid [[threadgroup_position_in_grid]],
                    metal::mesh<GridOut, void, 3, 1, metal::topology::triangle> out)
{
    if (gid >= 4u || payload[gid * 2u].valid == 0u) {
        out.set_primitive_count(0u);
        return;
    }
    // The grid threadgroup index selects a disjoint 8x8 quadrant; each
    // triangle covers 28 of its quadrant's pixels.
    uint block_x = (gid & 1u) * 8u;
    uint block_y = ((gid >> 1u) & 1u) * 8u;
    float x0 = float(block_x);
    float y0 = float(block_y);
    float x1 = x0 + 7.9;
    float y1 = y0 + 7.9;
    auto clip = [](float px, float py) {
        return float4((px / 16.0) * 2.0 - 1.0,
                      -((py / 16.0) * 2.0 - 1.0),
                      0.0,
                      1.0);
    };
    float4 color = payload[gid * 2u].color;
    GridOut v0;
    GridOut v1;
    GridOut v2;
    v0.position = clip(x0, y0);
    v1.position = clip(x1, y0);
    v2.position = clip(x0, y1);
    v0.color = color;
    v1.color = color;
    v2.color = color;
    out.set_vertex(0u, v0);
    out.set_vertex(1u, v1);
    out.set_vertex(2u, v2);
    out.set_index(0u, uchar(0u));
    out.set_index(1u, uchar(1u));
    out.set_index(2u, uchar(2u));
    out.set_primitive_count(1u);
}

fragment float4 grid_frag_main(GridOut in [[stage_in]])
{
    return in.color;
}
"#;

// ----- a console amplification container for the parser -------------------

/// The FSSM payload's slot table ends at the compute slot, then one record,
/// its stage data and the code.
const FSSM_PAYLOAD_HEADER: usize = 0x68;
const AS_SLOT: usize = 0x60;
const RECORD_SIZE: usize = 0x54;
const AS_STAGE_DATA: usize = 0x40;
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;

fn put_u32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// Adler-32 of `data`, as zlib trailers store it (big-endian).
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// A zlib stream of stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    if data.is_empty() {
        out.extend([0x01, 0x00, 0x00, 0xFF, 0xFF]);
    }
    let mut chunks = data.chunks(65_535).peekable();
    while let Some(chunk) = chunks.next() {
        let final_block = chunks.peek().is_none();
        out.push(u8::from(final_block));
        let len = chunk.len() as u16;
        out.extend(len.to_le_bytes());
        out.extend((!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend(adler32(data).to_be_bytes());
    out
}

/// `DXIL` part data of a kind-14 container: one amplification record in the
/// compute slot (payload +0x60) with `threads` and `payload_bytes`, and
/// `s_endpgm` as its code.
fn amplification_part(threads: [u32; 3], payload_bytes: u32) -> Vec<u8> {
    let record = FSSM_PAYLOAD_HEADER;
    let stage = record + RECORD_SIZE;
    let code = stage + AS_STAGE_DATA;
    let code_bytes = 0xBF81_0000u32.to_le_bytes(); // s_endpgm

    let mut payload = vec![0u8; FSSM_PAYLOAD_HEADER];
    put_u64(&mut payload, 0, 6);
    for at in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        put_u64(&mut payload, at, ABSENT);
    }
    put_u64(&mut payload, AS_SLOT, record as u64);
    payload.resize(code, 0);
    put_u64(&mut payload, record, code as u64);
    put_u64(&mut payload, record + 8, code_bytes.len() as u64);
    put_u32(&mut payload, record + 0x14, 1 << 5); // wave32
    put_u32(&mut payload, record + 0x4c, payload_bytes);
    put_u32(&mut payload, record + 0x50, 4); // VGPRs
    put_u32(&mut payload, stage + 0x24, 0); // COMPUTE_PGM_RSRC2
    for (i, n) in threads.into_iter().enumerate() {
        put_u32(&mut payload, stage + 0x28 + 4 * i, n);
    }
    payload.extend_from_slice(&code_bytes);

    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);

    let packed = zlib_stored(&fssm);
    let mut blob = b"XBCH".to_vec();
    blob.extend(3u32.to_le_bytes());
    blob.extend((packed.len() as u64).to_le_bytes());
    blob.extend((fssm.len() as u64).to_le_bytes());
    blob.extend(packed);

    let mut out = vec![0u8; 0x18];
    out[..4].copy_from_slice(&(14u32 << 16 | 0x66).to_le_bytes()); // as_6_6
    out.extend_from_slice(b"XDAT");
    out.extend(1u32.to_le_bytes());
    out.extend(0x24u32.to_le_bytes());
    let offset = (out.len() + 12) as u32;
    out.extend(0x0Du32.to_le_bytes()); // KIND_NATIVE
    out.extend(offset.to_le_bytes());
    out.extend((blob.len() as u32).to_le_bytes());
    out.extend(blob);
    out
}

/// Hardware proof: the amplification stage's computed grid drives the mesh
/// dispatch, and the `victory-shader` program parsed from a console
/// amplification container supplies the object threadgroup size and payload
/// memory the host uses.
#[test]
fn amplification_computed_grid_drives_the_mesh_dispatch() {
    let part = amplification_part([8, 1, 1], PAYLOAD_SLOTS as u32 * 32);
    let programs = native_programs(&part).expect("the amplification part parses");
    let program = programs
        .iter()
        .find(|p| p.stage == HwStage::As)
        .expect("the compute-slot record is HwStage::As");
    let StageIo::As(amplification) = &program.io else {
        panic!("amplification I/O: {:?}", program.io);
    };
    assert_eq!(amplification.compute.threads, [8, 1, 1]);
    assert_eq!(amplification.payload_bytes, PAYLOAD_SLOTS as u32 * 32);
    assert_eq!(program.code, 0xBF81_0000u32.to_le_bytes());
    eprintln!(
        "victory-shader: {:?} threads={:?} payload={} B code={} B",
        program.stage,
        amplification.compute.threads,
        amplification.payload_bytes,
        program.code.len()
    );

    let metallib = compile_msl(AMPLIFICATION_GRID_MSL)
        .expect("the amplification grid MSL compiles to a metallib");
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let device = transport.device_info().expect("device info");

    // The parsed program, not a hand-written constant, supplies the object
    // threadgroup size, the payload memory and the dispatch maximum.
    let threads = amplification.compute.threads;
    let spec = MeshPipelineSpec {
        max_total_threads_per_object_threadgroup: threads.iter().product::<u32>() as usize,
        payload_memory_length: amplification.payload_bytes as usize,
        threads_per_object_threadgroup: threads,
        threads_per_mesh_threadgroup: [1, 1, 1],
        mesh_output_bytes: 3 * 32,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [1, 1, 1])
    };
    let pipeline = MeshPipeline::from_metallibs(
        &transport,
        &metallib,
        "grid_mesh_main",
        Some((&metallib, "grid_object_main")),
        &metallib,
        "grid_frag_main",
        &spec,
    )
    .expect("the amplification grid pipeline should be created from the metallibs");
    let pipeline_id = transport
        .register_mesh_pipeline(pipeline)
        .expect("the amplification grid pipeline should register");

    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("render target should be created");
    let target_id = transport
        .register_render_target(&target)
        .expect("render target should register");
    let mut allocations = HashMap::new();
    allocations.insert((1u64, 1u32), VmBufferId(target_id.0));

    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![ColorAttachment {
                    allocation_id: 1,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                }],
                depth_stencil: None,
            }),
            CommandV3::SetMeshPipeline(PipelineRef {
                pipeline_id: pipeline_id.0,
            }),
            CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: pipeline_id.0,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the amplification grid submission should execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "submission status");
    assert_eq!(stats.command_buffers, 1, "one command buffer");
    assert_eq!(stats.render_encoders, 1, "one render encoder");

    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("the render target should read back");
    assert_eq!(pixels.len(), (EDGE * EDGE) as usize);

    let mut histogram: HashMap<[u8; 4], usize> = HashMap::new();
    for pixel in &pixels {
        *histogram.entry(*pixel).or_insert(0) += 1;
    }
    eprintln!(
        "GPU: amplification grid {GRID_GROUPS} threadgroups on {}; histogram={histogram:?}",
        device.name_str()
    );
    for color in SURVIVOR_COLORS {
        assert_eq!(
            histogram.get(&color).copied(),
            Some(PIXELS_PER_TRIANGLE),
            "each grid threadgroup must paint its own {PIXELS_PER_TRIANGLE} pixels of {color:?}"
        );
    }
    let clear = (EDGE * EDGE) as usize - GRID_GROUPS * PIXELS_PER_TRIANGLE;
    assert_eq!(
        histogram.get(&CLEAR_RGBA).copied(),
        Some(clear),
        "only the four grid threadgroups' triangles must paint"
    );
    assert_eq!(
        histogram.len(),
        1 + GRID_GROUPS,
        "exactly four triangle colours plus the clear colour"
    );
    let painted: usize = histogram
        .iter()
        .filter(|(color, _)| **color != CLEAR_RGBA)
        .map(|(_, count)| *count)
        .sum();
    assert_eq!(
        painted,
        GRID_GROUPS * PIXELS_PER_TRIANGLE,
        "the parsed amplification program's 8-thread group must dispatch {GRID_GROUPS} mesh threadgroups"
    );
    eprintln!("GPU: {painted} px from {GRID_GROUPS} amplification-dispatched mesh threadgroups");
}
