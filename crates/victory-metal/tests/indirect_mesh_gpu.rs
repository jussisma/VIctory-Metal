//! Hardware proof for GPU-driven mesh dispatch inside an open render pass
//! (ADR-024b).
//!
//! One submission runs a compute kernel that writes the three-`uint32`
//! `MTLDispatchThreadgroupsIndirectArguments` record for
//! `drawMeshThreadgroupsWithIndirectBuffer:...` into a shared buffer, then
//! opens a render pass and records `SetMeshPipeline` +
//! `ExecuteIndirectDispatchMesh` inside it. The mesh shader paints one
//! 4-pixel-wide vertical strip per mesh threadgroup, so the GPU-written
//! `threadgroupsPerGrid.x` decides how many pixels are written:
//!
//! * count `1` paints strip 0 (4 columns, 64 pixels);
//! * count `2` paints strips 0 and 1 (8 columns, 128 pixels);
//! * a direct `DispatchMesh` with `group_counts = [2, 1, 1]` is the control
//!   and must match the count-2 indirect frame exactly.
//!
//! The count buffer is zero on the host and never read back: only the compute
//! kernel writes it, and the render encoder consumes it. The colour target is
//! the only CPU readback.
//!
//! The compute dispatch is its own `MTLComputeCommandEncoder` segment before
//! `BeginRenderPass`: the host rejects a compute dispatch inside a render pass,
//! and Metal orders the encoders of one command buffer. The *mesh* dispatch is
//! the part that executes inside the pass, which is what this test proves.
//!
//! Run one at a time on real Apple Silicon hardware:
//! ```text
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test indirect_mesh_gpu -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::mesh::MeshPipelineSpec;
use victory_metal::macos::render::PixelFormat;
use victory_metal::wire::{BarrierEntry, BufferBinding, Dispatch1d, ResourceBarrier};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, ExecuteIndirectDispatchMesh, IndirectArguments,
    LoadOp, PipelineRef, StoreOp, SubmissionV3,
};

const ALLOCATION_RECORD: u64 = 1;
const ALLOCATION_SEED: u64 = 2;
const GENERATION: u32 = 1;
const TARGET_EDGE: u32 = 16;
/// Pixels painted by one mesh threadgroup: 4 columns of the 16-pixel target.
const STRIP_PIXELS: usize = 4 * TARGET_EDGE as usize;

/// Mesh shader: one threadgroup emits the clip-space quad for its
/// `threadgroup_position_in_grid.x` slot. `max_vertices = 4`,
/// `max_primitives = 2`.
const MESH_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VMIndirectMeshOut {
    float4 position [[position]];
};

[[mesh]] void vm_indirect_mesh(
        mesh<VMIndirectMeshOut, void, 4, 2, topology::triangle> out,
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        float x0 = -1.0 + 0.5 * float(gid);
        float x1 = x0 + 0.5;
        VMIndirectMeshOut v;
        v.position = float4(x0, -1.0, 0.0, 1.0);
        out.set_vertex(0, v);
        v.position = float4(x1, -1.0, 0.0, 1.0);
        out.set_vertex(1, v);
        v.position = float4(x1, 1.0, 0.0, 1.0);
        out.set_vertex(2, v);
        v.position = float4(x0, 1.0, 0.0, 1.0);
        out.set_vertex(3, v);
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_index(3, 0);
        out.set_index(4, 2);
        out.set_index(5, 3);
        out.set_primitive_count(2);
    }
}

fragment float4 vm_indirect_mesh_fragment() {
    return float4(1.0, 1.0, 1.0, 1.0);
}
"#;

/// Compute kernel: copies the host-supplied seed into the first word of the
/// indirect record and sets `y`/`z` to one. The record buffer itself is never
/// written by the CPU.
const WRITE_COUNTS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vm_write_mesh_counts(device uint* record [[buffer(0)]],
                                 device const uint* seed [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        record[0] = seed[0];
        record[1] = 1u;
        record[2] = 1u;
    }
}
"#;

/// Page-aligned host memory backing one `register_zerocopy` allocation.
struct PageBuffer {
    ptr: *mut u8,
    len: usize,
}

impl PageBuffer {
    fn new(page_size: usize, len: usize) -> Self {
        assert!(len != 0 && len.is_multiple_of(page_size));
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the output pointer is a valid slot and the alignment/size are
        // the device page size and a multiple of it.
        let status = unsafe { posix_memalign(&mut ptr, page_size, len) };
        assert_eq!(status, 0, "posix_memalign failed with {status}");
        // SAFETY: `ptr` covers `len` writable bytes.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, len) };
        Self {
            ptr: ptr.cast::<u8>(),
            len,
        }
    }

    fn raw(&self) -> *mut u8 {
        self.ptr
    }

    /// Writes one 32-bit word. Used only for the compute kernel's seed input,
    /// never for the indirect record.
    fn write_u32(&self, index: usize, value: u32) {
        // SAFETY: the caller stays inside the allocation; the first page holds
        // the single seed word.
        unsafe {
            std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), self.ptr.add(index * 4), 4);
        }
    }
}

impl Drop for PageBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `posix_memalign` above and is freed once.
        unsafe { free(self.ptr.cast::<c_void>()) };
    }
}

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn free(ptr: *mut c_void);
}

fn register_buffer(transport: &DirectMetalTransport, buffer: &PageBuffer) -> VmBufferId {
    // SAFETY: `buffer` is page-aligned, page-multiple and stays alive until
    // every submission in this test has completed.
    unsafe { transport.register_zerocopy(buffer.raw(), buffer.len) }
        .expect("zero-copy allocation should register")
}

fn color_target(allocation_id: u64) -> ColorAttachment {
    ColorAttachment {
        allocation_id,
        generation: GENERATION,
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    }
}

fn submit(
    transport: &DirectMetalTransport,
    allocations: &HashMap<(u64, u32), VmBufferId>,
    sequence: u32,
    commands: Vec<CommandV3>,
) {
    let submission = SubmissionV3 {
        sequence,
        list: None,
        commands,
    };
    let reply = transport
        .submit_v3(&submission, allocations)
        .expect("v3 submission should execute");
    assert_eq!(
        reply.status,
        victory_metal::wire::ReplyStatus::Ok,
        "submission {sequence} failed with {:?}",
        reply.status
    );
}

fn checksum(pixels: &[[u8; 4]]) -> u64 {
    pixels
        .iter()
        .flat_map(|pixel| *pixel)
        .fold(0xcbf2_9ce4_8422_2325, |state, byte| {
            (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn pixels_equal(left: &[[u8; 4]], right: &[[u8; 4]]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
}

fn white_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|pixel| **pixel == [255, 255, 255, 255])
        .count()
}

/// Whether every pixel of column `x` is the mesh shader's white output.
fn column_is_white(pixels: &[[u8; 4]], x: usize) -> bool {
    (0..TARGET_EDGE as usize).all(|y| pixels[y * TARGET_EDGE as usize + x] == [255, 255, 255, 255])
}

/// One indirect mesh pass: compute writes the record, then the pass runs the
/// indirect mesh dispatch. The record buffer is zero on the host.
fn indirect_mesh_pass(compute: u64, mesh: u64, target: u64) -> Vec<CommandV3> {
    vec![
        CommandV3::Dispatch1d(Dispatch1d {
            pipeline_id: compute,
            threads: 1,
            threads_per_group: 1,
            bindings: vec![
                BufferBinding {
                    allocation_id: ALLOCATION_RECORD,
                    generation: GENERATION,
                    offset: 0,
                    length: 16,
                },
                BufferBinding {
                    allocation_id: ALLOCATION_SEED,
                    generation: GENERATION,
                    offset: 0,
                    length: 4,
                },
            ],
        }),
        CommandV3::ResourceBarrier(ResourceBarrier {
            entries: vec![BarrierEntry {
                allocation_id: ALLOCATION_RECORD,
                generation: GENERATION,
                offset: 0,
                length: 12,
                before_state: 0,
                after_state: 1,
            }],
        }),
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color_target(target)],
            depth_stencil: None,
        }),
        CommandV3::SetMeshPipeline(PipelineRef { pipeline_id: mesh }),
        CommandV3::ExecuteIndirectDispatchMesh(ExecuteIndirectDispatchMesh {
            pipeline_id: mesh,
            arguments: IndirectArguments {
                allocation_id: ALLOCATION_RECORD,
                generation: GENERATION,
                offset: 0,
                max_count: 1,
                count: None,
            },
        }),
        CommandV3::EndRenderPass,
    ]
}

#[test]
#[ignore = "requires the M4 Max Metal GPU; run with --ignored --test-threads=1"]
fn indirect_mesh_dispatch_in_an_open_pass_matches_a_direct_mesh_draw() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let device = transport.device_info().expect("device info");
    let page_size = usize::try_from(device.page_size).expect("page size fits usize");
    // The indirect record is zero on the host and only the compute kernel
    // writes it. The seeds are the compute kernel's input.
    let record = PageBuffer::new(page_size, page_size);
    let seed_one = PageBuffer::new(page_size, page_size);
    let seed_two = PageBuffer::new(page_size, page_size);
    seed_one.write_u32(0, 1);
    seed_two.write_u32(0, 2);

    let record_id = register_buffer(&transport, &record);
    let seed_one_id = register_buffer(&transport, &seed_one);
    let seed_two_id = register_buffer(&transport, &seed_two);

    let compute = transport
        .compile_msl(WRITE_COUNTS_MSL, "vm_write_mesh_counts")
        .expect("count-writing kernel should compile");
    let spec = MeshPipelineSpec {
        // One position-only vertex per lane: 4 vertices x 16 B.
        mesh_output_bytes: 4 * 16,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [4, 1, 1])
    };
    let mesh = transport
        .register_mesh_pipeline_msl(
            MESH_MSL,
            "vm_indirect_mesh",
            None,
            "vm_indirect_mesh_fragment",
            &spec,
        )
        .expect("mesh pipeline should compile from MSL");

    let count_one_target = transport
        .create_render_target_2d(TARGET_EDGE, TARGET_EDGE, PixelFormat::Rgba8Unorm)
        .expect("count-one target");
    let count_two_target = transport
        .create_render_target_2d(TARGET_EDGE, TARGET_EDGE, PixelFormat::Rgba8Unorm)
        .expect("count-two target");
    let direct_target = transport
        .create_render_target_2d(TARGET_EDGE, TARGET_EDGE, PixelFormat::Rgba8Unorm)
        .expect("direct target");
    let count_one_id = transport
        .register_render_target(&count_one_target)
        .expect("count-one target registers")
        .0;
    let count_two_id = transport
        .register_render_target(&count_two_target)
        .expect("count-two target registers")
        .0;
    let direct_id = transport
        .register_render_target(&direct_target)
        .expect("direct target registers")
        .0;

    let allocations = HashMap::from([((ALLOCATION_RECORD, GENERATION), record_id)]);
    // `ALLOCATION_SEED` names the host seed buffer; both seeds share it in
    // turn, so each submission binds the seed it should read.
    let mut seed_one_allocations = allocations.clone();
    seed_one_allocations.insert((ALLOCATION_SEED, GENERATION), seed_one_id);
    let mut seed_two_allocations = allocations.clone();
    seed_two_allocations.insert((ALLOCATION_SEED, GENERATION), seed_two_id);
    for id in [count_one_id, count_two_id, direct_id] {
        seed_one_allocations.insert((id, GENERATION), VmBufferId(id));
        seed_two_allocations.insert((id, GENERATION), VmBufferId(id));
    }

    // 1. Compute writes count 1; the indirect mesh draws one strip.
    submit(
        &transport,
        &seed_one_allocations,
        1,
        indirect_mesh_pass(compute.0, mesh.0, count_one_id),
    );

    // 2. Compute writes count 2; the same indirect mesh draws two strips.
    submit(
        &transport,
        &seed_two_allocations,
        2,
        indirect_mesh_pass(compute.0, mesh.0, count_two_id),
    );

    // 3. Control: the same mesh draw recorded directly with two threadgroups.
    let direct = vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![color_target(direct_id)],
            depth_stencil: None,
        }),
        CommandV3::SetMeshPipeline(PipelineRef {
            pipeline_id: mesh.0,
        }),
        CommandV3::DispatchMesh(victory_metal::wire::DispatchMesh {
            pipeline_id: mesh.0,
            group_counts: [2, 1, 1],
            bindings: Vec::new(),
        }),
        CommandV3::EndRenderPass,
    ];
    submit(&transport, &seed_two_allocations, 3, direct);

    // The only CPU readback is the colour target.
    let count_one_pixels = transport
        .read_render_target_rgba8(&count_one_target)
        .expect("count-one readback");
    let count_two_pixels = transport
        .read_render_target_rgba8(&count_two_target)
        .expect("count-two readback");
    let direct_pixels = transport
        .read_render_target_rgba8(&direct_target)
        .expect("direct readback");
    assert_eq!(count_one_pixels.len(), (TARGET_EDGE * TARGET_EDGE) as usize);

    let count_one_white = white_pixels(&count_one_pixels);
    let count_two_white = white_pixels(&count_two_pixels);
    let direct_white = white_pixels(&direct_pixels);
    let count_one_checksum = checksum(&count_one_pixels);
    let count_two_checksum = checksum(&count_two_pixels);
    let direct_checksum = checksum(&direct_pixels);
    let differing = pixels_equal(&count_two_pixels, &direct_pixels);
    eprintln!(
        "GPU {}: indirect mesh count=1 white={count_one_white} checksum=0x{count_one_checksum:016x}; \
         count=2 white={count_two_white} checksum=0x{count_two_checksum:016x}; \
         direct=[2,1,1] white={direct_white} checksum=0x{direct_checksum:016x}; \
         indirect_vs_direct_differing={differing}",
        device.name_str()
    );

    // The GPU-written count changed the number of written pixels: one strip
    // versus two.
    assert_eq!(
        count_one_white, STRIP_PIXELS,
        "count=1 must paint exactly one 4-column strip"
    );
    assert_eq!(
        count_two_white,
        2 * STRIP_PIXELS,
        "count=2 must paint exactly two 4-column strips"
    );
    assert!(
        count_two_white > count_one_white,
        "the compute-written count must drive the draw"
    );
    for x in 0..4 {
        assert!(column_is_white(&count_one_pixels, x), "count=1 column {x}");
    }
    for x in 4..8 {
        assert!(
            !column_is_white(&count_one_pixels, x),
            "count=1 must leave column {x} clear"
        );
    }
    for x in 0..8 {
        assert!(column_is_white(&count_two_pixels, x), "count=2 column {x}");
    }
    for x in 8..TARGET_EDGE as usize {
        assert!(
            !column_is_white(&count_two_pixels, x),
            "count=2 must leave column {x} clear"
        );
    }

    // Control: the indirect path equals the direct mesh draw exactly.
    assert_eq!(
        differing, 0,
        "the indirect mesh dispatch must match a direct DispatchMesh"
    );
    assert_eq!(count_two_checksum, direct_checksum);
    assert_eq!(count_two_white, direct_white);

    // Structural half of the no-readback proof: the record allocation is never
    // read through a Metal contents pointer, and the test never names a CPU
    // word extraction of it.
    let source = include_str!("indirect_mesh_gpu.rs");
    assert!(!source.contains(concat!("get", "Bytes")));
}
