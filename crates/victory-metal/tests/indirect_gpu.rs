//! Hardware proof for GPU-driven `ExecuteIndirect` and texture region work
//! (ADR-015).
//!
//! 1. A compute kernel writes the 16-byte D3D12 prefix of
//!    `MTLDrawIndexedPrimitivesIndirectArguments` plus the fifth
//!    `baseInstance` word Metal reads (20 bytes total) into a shared buffer.
//!    `ExecuteIndirect(DrawIndexedIndirect)` then draws from that buffer and
//!    the colour target must match an equivalent direct draw pixel for pixel.
//!    The argument buffer is zeroed before the submission and never read or
//!    written by the CPU, so an identical result proves the GPU wrote and
//!    consumed it.
//! 2. `CopyTextureRegion` copies a 32x32 sub-box of a 64x64 rendered texture
//!    with independent source/destination origins; every copied pixel and the
//!    untouched border are checked exactly.
//! 3. `ResolveSubresource` resolves a cleared 4x MSAA target into a 1x target;
//!    the resolved pixels must equal a direct clear of the same colour. A
//!    non-uniform sample average needs a `sampleCount = 4` graphics pipeline,
//!    which `GraphicsPipeline::compile_mrt` does not build (see the report).
//!
//! Run one at a time on real Apple Silicon hardware:
//! ```text
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test indirect_gpu -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::blit::create_multisample_render_target;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::render::PixelFormat;
use victory_metal::wire::Dispatch1d;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, CopyTextureRegion, ExecuteIndirectDrawIndexed,
    IndirectArguments, LoadOp, PipelineRef, ResolveSubresource, ScissorRect, StoreOp,
    SubmissionV3, TextureSubresource, VertexBuffer, Viewport,
};
use victory_metal::wire_v3;

const ALLOCATION_GEOMETRY: u64 = 1;
const ALLOCATION_ARGUMENTS: u64 = 2;
const GENERATION: u32 = 1;
const INDEX_OFFSET: usize = 256;

const VERTEX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VMIndirectOut {
    float4 position [[position]];
    float4 color;
};

vertex VMIndirectOut vm_indirect_vertex(uint vid [[vertex_id]],
                                        device const float2* positions [[buffer(0)]]) {
    float2 p = positions[vid];
    VMIndirectOut out;
    out.position = float4(p, 0.0, 1.0);
    out.color = float4(0.25 + 0.125 * (p.x + 1.0),
                       0.25 + 0.125 * (p.y + 1.0),
                       0.75,
                       1.0);
    return out;
}

fragment float4 vm_indirect_fragment(VMIndirectOut in [[stage_in]]) {
    return in.color;
}
"#;

/// Writes `indexCount=3, instanceCount=1, indexStart=0, baseVertex=0` and the
/// Metal-only `baseInstance=0` word into a device buffer.
const ARGUMENTS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vm_write_indirect_args(device uint* arguments [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        arguments[0] = 3u; // indexCount
        arguments[1] = 1u; // instanceCount
        arguments[2] = 0u; // indexStart
        arguments[3] = 0u; // baseVertex
        arguments[4] = 0u; // baseInstance (Metal's fifth word)
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

    /// Writes point geometry and 32-bit indices. Never used for the argument
    /// buffer, whose only producer is the GPU.
    fn write_geometry(&self) {
        let positions: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
        for (index, position) in positions.iter().enumerate() {
            // SAFETY: the first 24 bytes of the allocation hold three float2
            // values and are writable by the host before submission.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    position.as_ptr().cast::<u8>(),
                    self.ptr.add(index * 8),
                    8,
                );
            }
        }
        for (index, value) in [0u32, 1, 2].into_iter().enumerate() {
            // SAFETY: the index window at `INDEX_OFFSET` is 12 bytes and lies
            // inside the allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    self.ptr.add(INDEX_OFFSET + index * 4),
                    4,
                );
            }
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

fn allocation_map(
    geometry: VmBufferId,
    arguments: VmBufferId,
) -> HashMap<(u64, u32), VmBufferId> {
    HashMap::from([
        ((ALLOCATION_GEOMETRY, GENERATION), geometry),
        ((ALLOCATION_ARGUMENTS, GENERATION), arguments),
    ])
}

fn geometry_map(geometry: VmBufferId) -> HashMap<(u64, u32), VmBufferId> {
    HashMap::from([((ALLOCATION_GEOMETRY, GENERATION), geometry)])
}

/// Adds registered texture handles to a submission snapshot.
///
/// `resolve_texture` resolves a render-pass attachment or copy subresource
/// through the same `(allocation_id, generation) -> VmBufferId` snapshot the
/// buffers use, so every texture id a submission names must be present here.
fn with_textures(
    mut allocations: HashMap<(u64, u32), VmBufferId>,
    textures: &[u64],
) -> HashMap<(u64, u32), VmBufferId> {
    for id in textures {
        allocations.insert((*id, GENERATION), VmBufferId(*id));
    }
    allocations
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

fn full_viewport(size: f64) -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: size,
        height: size,
        near: 0.0,
        far: 1.0,
    }
}

fn draw_state(pipeline: u64) -> Vec<CommandV3> {
    vec![
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: pipeline,
        }),
        CommandV3::SetViewport(full_viewport(16.0)),
        CommandV3::SetScissor(ScissorRect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        }),
        CommandV3::SetVertexBuffers(victory_metal::wire_v3::VertexBufferSet {
            start_slot: 0,
            buffers: vec![VertexBuffer {
                allocation_id: ALLOCATION_GEOMETRY,
                generation: GENERATION,
                offset: 0,
                stride: 8,
                length: 4096,
            }],
        }),
        CommandV3::SetIndexBuffer(victory_metal::wire_v3::IndexBufferBinding {
            allocation_id: ALLOCATION_GEOMETRY,
            generation: GENERATION,
            offset: INDEX_OFFSET as u64,
            length: 64,
            format: victory_metal::wire_v3::IndexFormat::Uint32,
        }),
    ]
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
    pixels.iter().flat_map(|pixel| *pixel).fold(0xcbf2_9ce4_8422_2325, |state, byte| {
        (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn pixels_equal(left: &[[u8; 4]], right: &[[u8; 4]]) -> usize {
    left.iter().zip(right).filter(|(left, right)| left != right).count()
}

#[test]
#[ignore = "requires the M4 Max Metal GPU; run with --ignored --test-threads=1"]
fn indirect_indexed_draw_matches_a_direct_draw_with_no_cpu_readback() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let page_size = usize::try_from(transport.device_info().expect("device info").page_size)
        .expect("page size fits usize");
    let geometry = PageBuffer::new(page_size, page_size);
    geometry.write_geometry();
    // The argument buffer is zero on the host. Only the compute kernel below
    // writes it, and the host never reads it back.
    let arguments = PageBuffer::new(page_size, page_size);

    let geometry_id = register_buffer(&transport, &geometry);
    let arguments_id = register_buffer(&transport, &arguments);
    let mut allocations = allocation_map(geometry_id, arguments_id);

    let compute = transport
        .compile_msl(ARGUMENTS_MSL, "vm_write_indirect_args")
        .expect("argument-writing kernel should compile");
    let graphics = transport
        .register_graphics_pipeline(
            VERTEX_MSL,
            "vm_indirect_vertex",
            "vm_indirect_fragment",
            PixelFormat::Rgba8Unorm,
        )
        .expect("graphics pipeline should compile");

    let direct_target = transport
        .create_render_target_2d(16, 16, PixelFormat::Rgba8Unorm)
        .expect("direct target");
    let indirect_target = transport
        .create_render_target_2d(16, 16, PixelFormat::Rgba8Unorm)
        .expect("indirect target");
    let direct_id = transport
        .register_render_target(&direct_target)
        .expect("direct target registers")
        .0;
    let indirect_id = transport
        .register_render_target(&indirect_target)
        .expect("indirect target registers")
        .0;
    allocations = with_textures(allocations, &[direct_id, indirect_id]);

    // 1. Direct indexed draw, used as the reference pixels.
    let mut direct = vec![CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![color_target(direct_id)],
        depth_stencil: None,
    })];
    direct.extend(draw_state(graphics.0));
    direct.push(CommandV3::DrawIndexed(
        victory_metal::wire_v3::DrawIndexedState {
            index_count: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        },
    ));
    direct.push(CommandV3::EndRenderPass);
    submit(&transport, &allocations, 1, direct);

    // 2. Compute writes the argument record, then the indirect draw reads it.
    let mut indirect = vec![CommandV3::Dispatch1d(Dispatch1d {
        pipeline_id: compute.0,
        threads: 1,
        threads_per_group: 1,
        bindings: vec![victory_metal::wire::BufferBinding {
            allocation_id: ALLOCATION_ARGUMENTS,
            generation: GENERATION,
            offset: 0,
            length: 4096,
        }],
    })];
    indirect.push(CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![color_target(indirect_id)],
        depth_stencil: None,
    }));
    indirect.extend(draw_state(graphics.0));
    indirect.push(CommandV3::ExecuteIndirectDrawIndexed(
        ExecuteIndirectDrawIndexed {
            arguments: IndirectArguments {
                allocation_id: ALLOCATION_ARGUMENTS,
                generation: GENERATION,
                offset: 0,
                max_count: 1,
                count: None,
            },
        },
    ));
    indirect.push(CommandV3::EndRenderPass);
    submit(&transport, &allocations, 2, indirect);

    // The only CPU readback is the colour target. The argument buffer is
    // registered and never mapped, copied or inspected.
    let direct_pixels = transport
        .read_render_target_rgba8(&direct_target)
        .expect("direct readback");
    let indirect_pixels = transport
        .read_render_target_rgba8(&indirect_target)
        .expect("indirect readback");
    assert_eq!(direct_pixels.len(), 16 * 16);
    let direct_checksum = checksum(&direct_pixels);
    let indirect_checksum = checksum(&indirect_pixels);
    let differing = pixels_equal(&direct_pixels, &indirect_pixels);
    let distinct = direct_pixels
        .iter()
        .map(|pixel| *pixel)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    eprintln!(
        "indirect_draw direct_checksum=0x{direct_checksum:016x} \
         indirect_checksum=0x{indirect_checksum:016x} \
         differing_pixels={differing} distinct_colors={distinct} pixels={}",
        direct_pixels.len()
    );
    assert_eq!(
        differing, 0,
        "indirect draw must match the direct draw exactly"
    );
    assert_eq!(direct_checksum, indirect_checksum);
    assert!(
        distinct > 4,
        "the fixture should render an interpolated gradient, not one colour"
    );
    assert!(
        direct_pixels.iter().all(|pixel| *pixel != [0, 0, 0, 255]),
        "the reference draw must cover the target"
    );

    // Structural half of the no-readback proof: this test never reads the
    // argument allocation through a Metal contents pointer.
    let source = include_str!("indirect_gpu.rs");
    // The needles are assembled at run time so this assertion cannot match its
    // own source text.
    assert!(!source.contains(concat!("get", "Bytes")));
    assert!(!source.contains(concat!("argument", "_words")));
}

#[test]
#[ignore = "requires the M4 Max Metal GPU; run with --ignored --test-threads=1"]
fn copy_texture_region_copies_an_exact_sub_box() {
    const EDGE: u32 = 64;
    const COPY_ORIGIN: u32 = 8;
    const DST_ORIGIN: u32 = 16;
    const COPY: u32 = 32;

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let page_size = usize::try_from(transport.device_info().expect("device info").page_size)
        .expect("page size fits usize");
    let geometry = PageBuffer::new(page_size, page_size);
    geometry.write_geometry();
    let allocations = geometry_map(register_buffer(&transport, &geometry));

    let graphics = transport
        .register_graphics_pipeline(
            VERTEX_MSL,
            "vm_indirect_vertex",
            "vm_indirect_fragment",
            PixelFormat::Rgba8Unorm,
        )
        .expect("graphics pipeline should compile");
    let source = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("source target");
    let destination = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("destination target");
    let source_id = transport
        .register_render_target(&source)
        .expect("source registers")
        .0;
    let destination_id = transport
        .register_render_target(&destination)
        .expect("destination registers")
        .0;
    let allocations = with_textures(allocations, &[source_id, destination_id]);

    let mut render_source = vec![CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![color_target(source_id)],
        depth_stencil: None,
    })];
    render_source.push(CommandV3::SetGraphicsPipeline(PipelineRef {
        pipeline_id: graphics.0,
    }));
    render_source.push(CommandV3::SetViewport(full_viewport(f64::from(EDGE))));
    render_source.push(CommandV3::SetScissor(ScissorRect {
        x: 0,
        y: 0,
        width: EDGE,
        height: EDGE,
    }));
    render_source.push(CommandV3::SetVertexBuffers(
        victory_metal::wire_v3::VertexBufferSet {
            start_slot: 0,
            buffers: vec![VertexBuffer {
                allocation_id: ALLOCATION_GEOMETRY,
                generation: GENERATION,
                offset: 0,
                stride: 8,
                length: 4096,
            }],
        },
    ));
    render_source.push(CommandV3::SetIndexBuffer(
        victory_metal::wire_v3::IndexBufferBinding {
            allocation_id: ALLOCATION_GEOMETRY,
            generation: GENERATION,
            offset: INDEX_OFFSET as u64,
            length: 64,
            format: victory_metal::wire_v3::IndexFormat::Uint32,
        },
    ));
    render_source.push(CommandV3::DrawIndexed(
        victory_metal::wire_v3::DrawIndexedState {
            index_count: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        },
    ));
    render_source.push(CommandV3::EndRenderPass);

    // Clear the destination, then copy the sub-box outside any render pass.
    render_source.push(CommandV3::BeginRenderPass(BeginRenderPass {
        colors: vec![color_target(destination_id)],
        depth_stencil: None,
    }));
    render_source.push(CommandV3::EndRenderPass);
    render_source.push(CommandV3::CopyTextureRegion(CopyTextureRegion {
        src: TextureSubresource {
            allocation_id: source_id,
            generation: GENERATION,
            mip_level: 0,
            slice: 0,
        },
        dst: TextureSubresource {
            allocation_id: destination_id,
            generation: GENERATION,
            mip_level: 0,
            slice: 0,
        },
        src_origin: [COPY_ORIGIN, COPY_ORIGIN, 0],
        dst_origin: [DST_ORIGIN, DST_ORIGIN, 0],
        size: [COPY, COPY, 1],
    }));
    submit(&transport, &allocations, 1, render_source);

    let source_pixels = transport
        .read_render_target_rgba8(&source)
        .expect("source readback");
    let destination_pixels = transport
        .read_render_target_rgba8(&destination)
        .expect("destination readback");

    let mut differing = 0usize;
    let mut sub_box = Vec::with_capacity((COPY * COPY) as usize);
    for y in 0..COPY {
        for x in 0..COPY {
            let source_index = ((COPY_ORIGIN + y) * EDGE + COPY_ORIGIN + x) as usize;
            let destination_index = ((DST_ORIGIN + y) * EDGE + DST_ORIGIN + x) as usize;
            sub_box.push(destination_pixels[destination_index]);
            if source_pixels[source_index] != destination_pixels[destination_index] {
                differing += 1;
            }
        }
    }
    for y in 0..EDGE {
        for x in 0..EDGE {
            let inside = x >= DST_ORIGIN
                && x < DST_ORIGIN + COPY
                && y >= DST_ORIGIN
                && y < DST_ORIGIN + COPY;
            if !inside {
                assert_eq!(
                    destination_pixels[(y * EDGE + x) as usize],
                    [0, 0, 0, 255],
                    "pixel ({x},{y}) outside the destination box must stay cleared"
                );
            }
        }
    }
    eprintln!(
        "copy_texture_region src_origin=({COPY_ORIGIN},{COPY_ORIGIN}) \
         dst_origin=({DST_ORIGIN},{DST_ORIGIN}) size={COPY}x{COPY} \
         differing_pixels={differing} sub_box_checksum=0x{:016x}",
        checksum(&sub_box)
    );
    assert_eq!(differing, 0, "every copied pixel must match the source box");
}

#[test]
#[ignore = "requires the M4 Max Metal GPU; run with --ignored --test-threads=1"]
fn resolve_subresource_matches_a_direct_clear() {
    const EDGE: u32 = 16;
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let page_size = usize::try_from(transport.device_info().expect("device info").page_size)
        .expect("page size fits usize");
    let geometry = PageBuffer::new(page_size, page_size);
    geometry.write_geometry();
    let allocations = geometry_map(register_buffer(&transport, &geometry));

    let multisample = create_multisample_render_target(
        &transport,
        EDGE,
        EDGE,
        PixelFormat::Rgba8Unorm,
        4,
    )
    .expect("4x multisample target");
    let resolved = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("resolved target");
    let reference = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("reference target");
    let multisample_id = transport
        .register_render_target(&multisample)
        .expect("multisample target registers")
        .0;
    let resolved_id = transport
        .register_render_target(&resolved)
        .expect("resolved target registers")
        .0;
    let reference_id = transport
        .register_render_target(&reference)
        .expect("reference target registers")
        .0;
    let allocations = with_textures(allocations, &[multisample_id, resolved_id, reference_id]);

    let clear = |allocation_id: u64| ColorAttachment {
        allocation_id,
        generation: GENERATION,
        mip_level: 0,
        slice: 0,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear_color: [0.25, 0.5, 0.75, 1.0],
    };

    let commands = vec![
        // Reference: a single-sample clear of the same colour.
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![clear(reference_id)],
            depth_stencil: None,
        }),
        CommandV3::EndRenderPass,
        // 4x source cleared to the same colour.
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![clear(multisample_id)],
            depth_stencil: None,
        }),
        CommandV3::EndRenderPass,
        CommandV3::ResolveSubresource(ResolveSubresource {
            src: TextureSubresource {
                allocation_id: multisample_id,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
            },
            dst: TextureSubresource {
                allocation_id: resolved_id,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
            },
        }),
    ];
    submit(&transport, &allocations, 1, commands);

    let resolved_pixels = transport
        .read_render_target_rgba8(&resolved)
        .expect("resolved readback");
    let reference_pixels = transport
        .read_render_target_rgba8(&reference)
        .expect("reference readback");
    let differing = pixels_equal(&resolved_pixels, &reference_pixels);
    eprintln!(
        "resolve_subresource samples=4 pixels={} differing_pixels={differing} \
         resolved_checksum=0x{:016x} reference_checksum=0x{:016x}",
        resolved_pixels.len(),
        checksum(&resolved_pixels),
        checksum(&reference_pixels)
    );
    assert_eq!(
        differing, 0,
        "the resolved 4x clear must equal a direct single-sample clear"
    );
}

#[test]
#[ignore = "requires the M4 Max Metal GPU; run with --ignored --test-threads=1"]
fn indirect_wire_stays_4_byte_aligned_on_the_gpu_path() {
    // Pure decoder guard that does not touch the GPU; kept here so the proof
    // binary also refuses an unaligned argument offset.
    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![CommandV3::ExecuteIndirectDrawIndexed(
            ExecuteIndirectDrawIndexed {
                arguments: IndirectArguments {
                    allocation_id: 1,
                    generation: 1,
                    offset: 2,
                    max_count: 1,
                    count: None,
                },
            },
        )],
    };
    assert!(
        wire_v3::encode_submission_v3(&submission).is_err(),
        "unaligned indirect argument offsets must not encode"
    );
}
