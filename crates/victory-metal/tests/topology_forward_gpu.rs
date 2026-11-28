//! Hardware proof for ADR-065: every primitive topology reaches
//! `MTLRenderCommandEncoder` unchanged, and an Xbox rect list is expanded into
//! two real triangles instead of a triangle-list draw.
//!
//! Two tests run on a real Metal device and assert exact pixel sets, not fuzzy
//! ranges:
//!
//! 1. [`wire_topologies_reach_metal_unchanged`] submits the same geometry
//!    through the v3 wire path, non-indexed and indexed, for `PointList`,
//!    `LineList`, `LineStrip`, `TriangleList` and `TriangleStrip`. Each pass
//!    must paint exactly the pixels its topology implies and nothing else.
//! 2. [`rect_list_expands_to_two_triangles`] drives the render encoder
//!    directly: the five topologies again for the non-indexed half, then a
//!    three-vertex `RectList` whose fourth corner is synthesized. The rect must
//!    paint the exact rectangle, never a degenerate triangle.
//!
//! The tests fail rather than skip when Metal is unavailable: the task's gate
//! is a non-black pixel proof, and a skipped proof cannot establish it.
//!
//! Run serialised with the GPU lock:
//! ```text
//! .swarm/bin/gpu-lock --wait 1800 -- cargo test --locked -p victory-metal \
//!   --target aarch64-apple-darwin --test topology_forward_gpu \
//!   -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::null_mut;

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::render::{
    GraphicsPipeline, PixelFormat, PrimitiveType, RectListOutcome, RectListRefusal,
    RenderAttachment, RenderEncoder, RenderPass, rect_list_stats, reset_rect_list_stats,
};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DrawIndexedState, DrawState, IndexBufferBinding,
    IndexFormat, LoadOp, PipelineRef, PrimitiveTopology, ScissorRect as WireScissor, StoreOp,
    SubmissionV3, VertexBuffer, VertexBufferSet, Viewport as WireViewport,
};

const EDGE: u32 = 64;
const PIXELS: usize = (EDGE * EDGE) as usize;
const PAGE: usize = 16 * 1024;
const INDEX_OFFSET: usize = 4096;
const TARGET_ALLOCATION: u64 = 1;
const PAGE_ALLOCATION: u64 = 2;

const RED: [u8; 4] = [255, 0, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];

/// Positions come from the bound vertex buffer; the fragment paints opaque red.
const TOPOLOGY_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float size [[point_size]];
};

vertex VOut topology_vertex(uint vid [[vertex_id]],
                            constant float2 *positions [[buffer(0)]]) {
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.size = 1.0;
    return out;
}

fragment float4 topology_fragment() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#;

/// Page-aligned host memory registered zero-copy as the guest vertex/index
/// page. Writes happen only while the GPU is idle (`wait_for_gpu`).
struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn zeroed(len: usize) -> Self {
        assert!(len.is_multiple_of(PAGE), "page length must be a multiple of {PAGE}");
        let mut ptr: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes one page-aligned allocation on success.
        let rc = unsafe { posix_memalign(&mut ptr, PAGE, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!ptr.is_null(), "posix_memalign returned null");
        // SAFETY: `ptr` is a live `len`-byte allocation.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, len) };
        Self { ptr, len }
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len, "host page write out of range");
        // SAFETY: the range was checked against the live allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.ptr.cast::<u8>().add(offset),
                bytes.len(),
            );
        }
    }

    fn raw(&self) -> *mut u8 {
        self.ptr.cast()
    }

    fn register(&self, transport: &DirectMetalTransport) -> VmBufferId {
        // SAFETY: the page lives until after the last `wait_for_gpu`, and the
        // transport drops after this guard.
        unsafe { transport.register_zerocopy(self.raw(), self.len) }
            .expect("zero-copy page registration")
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut c_void);
        }
        // SAFETY: `ptr` came from `posix_memalign` and is freed once.
        unsafe { free(self.ptr) };
    }
}

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
}

/// Pixel-space centre to clip space. `x`/`y` are pixel coordinates where the
/// centre of pixel `(col, row)` is `(col + 0.5, row + 0.5)`.
fn clip(x: f32, y: f32) -> [f32; 2] {
    let edge = EDGE as f32;
    [x / edge * 2.0 - 1.0, 1.0 - y / edge * 2.0]
}

fn positions_bytes(vertices: &[[f32; 2]]) -> Vec<u8> {
    vertices
        .iter()
        .flat_map(|[x, y]| [x.to_le_bytes(), y.to_le_bytes()])
        .flatten()
        .collect()
}

fn row(y: u32, x0: u32, x1: u32) -> Vec<(u32, u32)> {
    (x0..=x1).map(|x| (x, y)).collect()
}

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<(u32, u32)> {
    let mut pixels = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            pixels.push((x, y));
        }
    }
    pixels
}

/// One geometry case: the positions, the wire topology and the exact set of
/// pixels the topology must cover on this 64x64 target.
struct Case {
    name: &'static str,
    topology: PrimitiveTopology,
    vertices: Vec<[f32; 2]>,
    expected: Vec<(u32, u32)>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "PointList",
            topology: PrimitiveTopology::PointList,
            // Four points exactly on pixel centres: one fragment each.
            vertices: vec![
                clip(8.5, 8.5),
                clip(24.5, 8.5),
                clip(40.5, 8.5),
                clip(56.5, 8.5),
            ],
            expected: vec![(8, 8), (24, 8), (40, 8), (56, 8)],
        },
        Case {
            name: "LineList",
            topology: PrimitiveTopology::LineList,
            // Two horizontal lines through pixel-centre rows. Metal's
            // diamond-exit rule covers the start pixel and excludes the final
            // endpoint pixel, so each line covers 15 pixels (30 total).
            vertices: vec![
                clip(8.5, 16.5),
                clip(23.5, 16.5),
                clip(40.5, 48.5),
                clip(55.5, 48.5),
            ],
            expected: {
                let mut pixels = row(16, 8, 22);
                pixels.extend(row(48, 40, 54));
                pixels
            },
        },
        Case {
            name: "LineStrip",
            topology: PrimitiveTopology::LineStrip,
            // Four collinear vertices: three connected segments. Diamond-exit
            // excludes each segment's final endpoint pixel; the next segment's
            // start covers it, so the union is 24 pixels (x = 8..=31), where a
            // LineList of these vertices would cover 16.
            vertices: vec![
                clip(8.5, 16.5),
                clip(16.5, 16.5),
                clip(24.5, 16.5),
                clip(32.5, 16.5),
            ],
            expected: row(16, 8, 31),
        },
        Case {
            name: "TriangleList",
            topology: PrimitiveTopology::TriangleList,
            // Two triangles filling the bottom half (rows 32..=63).
            vertices: vec![
                clip(0.0, 64.0),
                clip(64.0, 64.0),
                clip(0.0, 32.0),
                clip(64.0, 64.0),
                clip(64.0, 32.0),
                clip(0.0, 32.0),
            ],
            expected: rect(0, 32, 63, 63),
        },
        Case {
            name: "TriangleStrip",
            topology: PrimitiveTopology::TriangleStrip,
            // Four vertices: two strip triangles filling the bottom half.
            vertices: vec![
                clip(0.0, 64.0),
                clip(0.0, 32.0),
                clip(64.0, 64.0),
                clip(64.0, 32.0),
            ],
            expected: rect(0, 32, 63, 63),
        },
    ]
}

/// The three-vertex Xbox rect list: `v0`, `v1`, `v2` with the implicit fourth
/// corner `v3 = v0 + v2 - v1`.
fn rect_list_vertices() -> Vec<[f32; 2]> {
    vec![clip(64.0, 64.0), clip(0.0, 64.0), clip(0.0, 32.0)]
}

fn wire_rect_list_expected() -> Vec<(u32, u32)> {
    rect(0, 32, 63, 63)
}

fn assert_exact_red(pixels: &[[u8; 4]], expected: &[(u32, u32)], label: &str) -> usize {
    assert_eq!(pixels.len(), PIXELS, "{label}: readback length");
    let expected: HashSet<(u32, u32)> = expected.iter().copied().collect();
    let mut covered = HashSet::new();
    for y in 0..EDGE {
        for x in 0..EDGE {
            let pixel = pixels[(y * EDGE + x) as usize];
            match pixel {
                RED => {
                    covered.insert((x, y));
                }
                BLACK => {}
                other => panic!("{label}: unexpected pixel ({x},{y}) = {other:?}"),
            }
        }
    }
    assert_eq!(
        covered, expected,
        "{label}: covered pixels differ (covered={}, expected={})",
        covered.len(),
        expected.len()
    );
    eprintln!(
        "topology_forward_gpu: {label} covered_pixels={} exact_red",
        covered.len()
    );
    covered.len()
}

fn full_viewport() -> WireViewport {
    WireViewport {
        x: 0.0,
        y: 0.0,
        width: f64::from(EDGE),
        height: f64::from(EDGE),
        near: 0.0,
        far: 1.0,
    }
}

fn full_scissor() -> WireScissor {
    WireScissor {
        x: 0,
        y: 0,
        width: EDGE,
        height: EDGE,
    }
}

fn begin_pass() -> CommandV3 {
    CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![ColorAttachment {
            allocation_id: TARGET_ALLOCATION,
            generation: 1,
            mip_level: 0,
            slice: 0,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }],
        depth_stencil: None,
    })
}

fn wire_pipeline(transport: &DirectMetalTransport) -> u64 {
    let pipeline = GraphicsPipeline::compile(
        transport,
        TOPOLOGY_MSL,
        "topology_vertex",
        "topology_fragment",
        PixelFormat::Rgba8Unorm,
        None,
    )
    .expect("topology pipeline compiles");
    transport
        .register_graphics_pipeline_state(pipeline)
        .expect("topology pipeline registers")
        .0
}

fn submit(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    sequence: u32,
    commands: Vec<CommandV3>,
) {
    let (reply, _stats) = transport
        .submit_v3_with_stats(
            &SubmissionV3 {
                sequence,
                list: None,
                commands,
            },
            allocations,
        )
        .expect("submission executes");
    assert_eq!(reply.status, ReplyStatus::Ok, "submission status");
    transport.wait_for_gpu().expect("GPU completes");
}

/// Proves the wire -> `MTLRenderCommandEncoder` topology map for every wire
/// topology, non-indexed and indexed.
#[test]
fn wire_topologies_reach_metal_unchanged() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("offscreen RGBA8 target");
    let target_handle = transport
        .register_render_target(&target)
        .expect("target registers");
    let mut page = HostPage::zeroed(PAGE);
    let page_id = page.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((TARGET_ALLOCATION, 1), VmBufferId(target_handle.0));
    allocations.insert((PAGE_ALLOCATION, 1), page_id);
    let pipeline_id = wire_pipeline(&transport);

    let mut sequence = 1u32;
    for case in cases() {
        for indexed in [false, true] {
            let label = format!(
                "wire {} {}",
                case.name,
                if indexed { "indexed" } else { "non-indexed" }
            );
            page.write(0, &positions_bytes(&case.vertices));
            if indexed {
                let indices: Vec<u8> = (0..case.vertices.len() as u16)
                    .flat_map(u16::to_le_bytes)
                    .collect();
                page.write(INDEX_OFFSET, &indices);
            }

            let mut commands = vec![
                begin_pass(),
                CommandV3::SetGraphicsPipeline(PipelineRef { pipeline_id }),
                CommandV3::SetViewport(full_viewport()),
                CommandV3::SetScissor(full_scissor()),
                CommandV3::SetPrimitiveTopology(case.topology),
                CommandV3::SetVertexBuffers(VertexBufferSet {
                    start_slot: 0,
                    buffers: vec![VertexBuffer {
                        allocation_id: PAGE_ALLOCATION,
                        generation: 1,
                        offset: 0,
                        stride: 8,
                        length: (case.vertices.len() * 8) as u64,
                    }],
                }),
            ];
            if indexed {
                commands.push(CommandV3::SetIndexBuffer(IndexBufferBinding {
                    allocation_id: PAGE_ALLOCATION,
                    generation: 1,
                    offset: INDEX_OFFSET as u64,
                    length: (case.vertices.len() * 2) as u64,
                    format: IndexFormat::Uint16,
                }));
                commands.push(CommandV3::DrawIndexed(DrawIndexedState {
                    index_count: case.vertices.len() as u32,
                    instance_count: 1,
                    start_index: 0,
                    base_vertex: 0,
                    start_instance: 0,
                }));
            } else {
                commands.push(CommandV3::Draw(DrawState {
                    vertex_count: case.vertices.len() as u32,
                    instance_count: 1,
                    start_vertex: 0,
                    start_instance: 0,
                }));
            }
            commands.push(CommandV3::EndRenderPass);

            submit(&transport, &allocations, sequence, commands);
            sequence += 1;
            let pixels = transport
                .read_render_target_rgba8(&target)
                .expect("target readback");
            assert_exact_red(&pixels, &case.expected, &label);
        }
    }

    transport
        .unregister_buffer(page_id)
        .expect("zero-copy page unregisters");
}

/// Drives the encoder directly for the five topologies and then the real rect
/// list expansion. The rect list's 64x32 bottom half is exactly 2048 pixels;
/// a triangle-list fallback of its three inputs covers only half of that.
#[test]
fn rect_list_expands_to_two_triangles() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let pipeline = GraphicsPipeline::compile(
        &transport,
        TOPOLOGY_MSL,
        "topology_vertex",
        "topology_fragment",
        PixelFormat::Rgba8Unorm,
        None,
    )
    .expect("topology pipeline compiles");

    let draw_direct = |vertices: &[[f32; 2]],
                       topology: Option<PrimitiveTopology>,
                       rect_list: bool|
     -> Vec<[u8; 4]> {
        let target = transport
            .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
            .expect("offscreen RGBA8 target");
        let pass = RenderPass::new(RenderAttachment::color(target))
            .expect("colour-only render pass validates");
        let mut encoder =
            RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder should be created");
        if rect_list {
            let outcome = encoder
                .draw_rect_list(&positions_bytes(vertices), 8, 0, 0, vertices.len())
                .expect("rect list encode");
            match outcome {
                RectListOutcome::Drawn {
                    rects,
                    vertices: synthesized,
                    indices,
                    vertex_bytes,
                } => {
                    eprintln!(
                        "topology_forward_gpu: rect_list rects={rects} vertices={synthesized} \
                         indices={indices} vertex_bytes={vertex_bytes} triangles={}",
                        rects * 2
                    );
                }
                RectListOutcome::Refused { reason } => {
                    panic!("rect list must expand, not refuse: {}", reason.name())
                }
            }
        } else {
            encoder
                .set_vertex_bytes(&positions_bytes(vertices), 0)
                .expect("vertex root constants bind");
            encoder
                .draw_primitives(
                    match topology.expect("non-rect case names a topology") {
                        PrimitiveTopology::PointList => PrimitiveType::Point,
                        PrimitiveTopology::LineList => PrimitiveType::Line,
                        PrimitiveTopology::LineStrip => PrimitiveType::LineStrip,
                        PrimitiveTopology::TriangleList => PrimitiveType::Triangle,
                        PrimitiveTopology::TriangleStrip => PrimitiveType::TriangleStrip,
                    },
                    0,
                    vertices.len(),
                )
                .expect("topology draw encodes");
        }
        encoder.finish().expect("render pass completes");
        transport
            .read_render_target_rgba8(pass.color_target())
            .expect("target readback")
    };

    for case in cases() {
        let pixels = draw_direct(&case.vertices, Some(case.topology), false);
        assert_exact_red(
            &pixels,
            &case.expected,
            &format!("encoder {}", case.name),
        );
    }

    reset_rect_list_stats();
    let pixels = draw_direct(&rect_list_vertices(), None, true);
    let covered = assert_exact_red(&pixels, &wire_rect_list_expected(), "encoder RectList");
    assert_eq!(covered, 2048, "the rect list must paint the full 64x32 rectangle");

    let stats = rect_list_stats();
    assert_eq!(stats.drawn_calls, 1, "one expanded rect-list call");
    assert_eq!(stats.drawn_rects, 1, "one expanded rectangle");
    assert_eq!(stats.expanded_vertices, 4, "four synthesized vertices");
    assert_eq!(stats.encoded_indices, 6, "six encoded triangle indices");
    assert_eq!(stats.refusals, [0; 6], "no rect-list refusal");
    eprintln!(
        "topology_forward_gpu: rect_list_stats drawn_calls={} drawn_rects={} \
         expanded_vertices={} encoded_indices={} refusals={:?}",
        stats.drawn_calls,
        stats.drawn_rects,
        stats.expanded_vertices,
        stats.encoded_indices,
        stats.refusals
    );

    // The refusal path is named and counted; it never becomes a triangle draw.
    reset_rect_list_stats();
    let two_rects = vec![
        clip(16.0, 64.0),
        clip(0.0, 64.0),
        clip(0.0, 48.0),
        clip(64.0, 16.0),
        clip(48.0, 16.0),
        clip(48.0, 0.0),
    ];
    let pixels = draw_direct(&two_rects, None, true);
    let mut expected = rect(0, 48, 15, 63);
    expected.extend(rect(48, 0, 63, 15));
    let covered = assert_exact_red(&pixels, &expected, "encoder RectListPair");
    assert_eq!(covered, 512, "two 16x16 rects paint 512 pixels");
    let stats = rect_list_stats();
    assert_eq!(stats.drawn_calls, 1, "one two-rect call");
    assert_eq!(stats.drawn_rects, 2, "two rectangles expand");
    assert_eq!(stats.encoded_indices, 12, "two rects encode twelve indices");

    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("offscreen RGBA8 target");
    let pass = RenderPass::new(RenderAttachment::color(target)).expect("render pass validates");
    let mut encoder = RenderEncoder::new(&transport, &pass, &pipeline).expect("encoder");
    let refused = encoder
        .draw_rect_list(&positions_bytes(&rect_list_vertices()), 4, 0, 0, 3)
        .expect("refusal is a normal outcome");
    assert_eq!(
        refused,
        RectListOutcome::Refused {
            reason: RectListRefusal::StrideTooSmall
        }
    );
    encoder.finish().expect("empty pass completes");
    let pixels = transport
        .read_render_target_rgba8(pass.color_target())
        .expect("target readback");
    assert!(
        pixels.iter().all(|pixel| *pixel == BLACK),
        "a refused rect list must encode no geometry"
    );
    let stats = rect_list_stats();
    assert_eq!(
        stats.refusals[RectListRefusal::StrideTooSmall.index()],
        1,
        "the refusal is counted once"
    );
}
