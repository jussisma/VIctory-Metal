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
//! `victory-shader` has no amplification/object hardware stage (its
//! `HwStage` covers only LS/HS/ES/GS/VS/PS/MS/CS), so no object DXIL can be
//! produced through the console emission path. The Metal toolchain compiles
//! the three MSL entry points to a metallib here, and the test then uses the
//! same host contract a title's converted metallibs would:
//! [`MeshPipeline::from_metallibs`] with `Some((object_metallib,
//! object_entry))`, `payload_memory_length`, both threadgroup maxima, a v3
//! `DispatchMesh` inside a render pass, and an exact RGBA8 readback.
//!
//! The proof is the payload count: eight object threads run, four are culled,
//! and exactly four triangles' pixels appear. A pipeline that ignored the
//! object stage would still draw the mesh's four triangles here (the mesh
//! stage alone would not know they were culled), but the colour histogram also
//! proves the object stage ran and wrote the payloads the mesh consumed.
//!
//! Run with:
//! ```text
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test object_mesh_gpu -- --ignored --test-threads=1 --nocapture
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
#[ignore = "hardware: compiles MSL with the Metal toolchain and needs an exclusive Metal device"]
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
