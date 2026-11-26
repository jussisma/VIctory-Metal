//! Hardware proof for the native Metal mesh pipeline path.
//!
//! The fixture is a synthetic console mesh program (4 threads, 4 vertices,
//! 4 primitives) that exports one full-screen triangle with indices `(0, 1, 2)`
//! and a constant-colour pixel shader. Both stages go through the repository's
//! own `victory_shader::emit` pipeline, then through Apple's IRConverter
//! (`libmetalirconverter.dylib`, x86_64 under Rosetta 2) to metallibs. The
//! mesh pipeline is created on the real device with
//! `newRenderPipelineStateWithMeshDescriptor:error:`, a v3 submission draws
//! `drawMeshThreadgroups`, and every pixel of a 16x16 offscreen target is read
//! back and compared exactly.
//!
//! # The historical `XPC_ERROR_CONNECTION_INTERRUPTED`
//!
//! That failure is not caused by the mesh library's deployment target, toolchain
//! flags, entry name or output budget. The AIR is built as a mesh library
//! (`air.mesh`, `air.trianglemesh`, `IRObjectGetMetalIRShaderStage` reports
//! `5`) and `main` exists. Feeding the same library to
//! `DirectMetalTransport::load_metallib` fails because that entry point asks
//! `newComputePipelineStateWithFunction:error:` for a *compute* pipeline:
//! `MTLCompilerService` crashes compiling a mesh-stage function as a kernel
//! (`EXC_BAD_ACCESS` in `MTLCompilerObject::backendCompileModule`) and Metal
//! reports the interrupted XPC connection. The same bytes load fine as an
//! `MTLLibrary`, and a within-budget mesh function produces the same XPC on the
//! compute path, which isolates the stage, not the bytes.
//!
//! The separate 32 KiB budget is real and checked in
//! [`MeshPipelineSpec::mesh_output_bytes`]. If the guard is bypassed, the mesh
//! descriptor path returns Metal's own clean diagnostic
//! (`Total mesh size (33280) exceeds the maximum mesh size allowed (32768)`).
//!
//! Run with:
//! ```text
//! VICTORY_IRCONVERTER_WORKER=<x86_64 victory-irconverter> \
//! VICTORY_METAL_IRCONVERTER=<libmetalirconverter.dylib> \
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test mesh_pipeline_gpu -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::cache::TranslationOptions;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::irconverter::{ProcessWorker, translate_dxil_with};
use victory_metal::macos::mesh::{MeshPipeline, MeshPipelineSpec};
use victory_metal::macos::render::PixelFormat;
use victory_metal::wire::{self, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, LoadOp, PipelineRef, StoreOp, SubmissionV3,
};
use victory_shader::emit::emit;
use victory_shader::native::{MeshIo, NativeProgram, StageIo, Usage, VsOutput};
use victory_shader::{Container, FourCC, HwStage, ShaderError, io, ir, resource, structure};

// ----- console fixtures -----------------------------------------------------

/// A `DXBC` container from parts (the shader crate's test helper is private).
fn build_container(parts: &[(FourCC, Vec<u8>)]) -> Vec<u8> {
    let table = 0x20 + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, data) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + data.len();
    }
    for (fourcc, data) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// One console signature row.
struct Row<'a> {
    name: &'a str,
    index: u32,
    /// D3D system value (`0` arbitrary, `1` position).
    sv: u32,
    register: u32,
    /// Four-bit channel mask.
    mask: u8,
    /// Raw component type (`3`: float32).
    raw: u32,
}

/// `OSG1`/`PSG1`/`ISG1` bytes: count, rows from offset 8, then the names.
fn signature_part(rows: &[Row<'_>]) -> Vec<u8> {
    let mut names = Vec::new();
    let mut offsets = Vec::with_capacity(rows.len());
    for row in rows {
        offsets.push((8 + 32 * rows.len() + names.len()) as u32);
        names.extend_from_slice(row.name.as_bytes());
        names.push(0);
    }
    let mut out = Vec::new();
    out.extend((rows.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    for (row, &offset) in rows.iter().zip(&offsets) {
        out.extend(0u32.to_le_bytes()); // stream
        out.extend(offset.to_le_bytes());
        out.extend(row.index.to_le_bytes());
        out.extend(row.sv.to_le_bytes());
        out.extend(row.raw.to_le_bytes());
        out.extend(row.register.to_le_bytes());
        out.extend([row.mask, 0, 0, 0]);
        out.extend(0u32.to_le_bytes());
    }
    out.extend(names);
    out
}

/// Runs the repository's emission pipeline for one prepared console program.
fn translate(container: &[u8], native: &NativeProgram) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(container)?;
    let program = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&program)?;
    let res = resource::track(&program, &native.layout, native.stage, Some(&st))?;
    let raised = io::raise(&program, native)?;
    let host = emit(&c, native, &program, &st, &raised, &res)?;
    Ok(host.dxil)
}

/// One console mesh program exporting `packed_indices` per lane.
///
/// The fixture's positions are the lane-bit pattern the `MESH-19` analysis
/// used: lane 0 is `(-1, -1)`, lane 1 `(3, -1)`, lane 2 `(-1, 3)`, lane 3
/// `(3, 3)`. Packing indices `(0, 1, 2)` therefore covers the whole clip
/// square, and the constant-colour fragment shader paints every pixel.
fn mesh_shader(
    rows: &[&[u8]],
    vertices: u32,
    primitives: u32,
    threads: u32,
    packed_indices: u32,
) -> Result<Vec<u8>, ShaderError> {
    let mut words: Vec<u32> = vec![
        0xD765_0001, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001_00C1,
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E12_02FF, // v_mov_b32 v9, <literal>
        packed_indices,
    ];
    let mut osg = vec![Row {
        name: "SV_Position",
        index: 0,
        sv: 1,
        register: 0,
        mask: 0xF,
        raw: 3,
    }];
    let mut outputs = Vec::new();
    let mut param = 0u32;
    let mut index = 0u32;
    for (row, widths) in (1..).zip(rows) {
        let mut col = 0u8;
        for &cols in *widths {
            let mask = ((1u32 << cols) - 1) << col;
            osg.push(Row {
                name: "TEXCOORD",
                index,
                sv: 0,
                register: row,
                mask: mask as u8,
                raw: 3,
            });
            index += 1;
            col += cols;
        }
        // exp param<n> v4, v5, v6, v7 (the row's channels).
        let mask = ((1u32 << col) - 1) | ((32 + param) << 4);
        words.extend([0xF800_0000 | mask, 0x0706_0504]);
        outputs.push(VsOutput {
            usage: Usage::Generic,
            row: row as u8,
            param: param as u8,
            fp16: false,
            pack2: false,
        });
        param += 1;
    }
    words.extend([
        0xF800_0941, // exp prim v9, off, off, off done
        0x0000_0009,
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 10,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs,
            clip_cntl: 0,
            threads,
            max_vertices: vertices,
            max_primitives: primitives,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = build_container(&[(FourCC::OSG1, signature_part(&osg))]);
    translate(&container, &native)
}

/// A pixel shader with no inputs that writes `(1, 1, 1, 1)`.
fn constant_pixel() -> Result<Vec<u8>, ShaderError> {
    let words: [u32; 4] = [
        0x7E04_02F2, // v_mov_b32 v2, 1.0
        0xF800_180F, // exp mrt0, v2, v2, v2, v2 done vm
        0x0202_0202,
        0xBF81_0000,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg = signature_part(&[]);
    let osg = signature_part(&[Row {
        name: "SV_Target",
        index: 0,
        sv: 64,
        register: 0,
        mask: 0xF,
        raw: 3,
    }]);
    let container = build_container(&[(FourCC::ISG1, isg), (FourCC::OSG1, osg)]);
    translate(&container, &native)
}

/// The `MESH-19` fixture: 29 full rows, one 1+3 row, two 2-wide rows at 64
/// vertices (33,280 declared bytes).
fn wide_rows() -> Vec<&'static [u8]> {
    let mut rows: Vec<&'static [u8]> = vec![&[4]; 29];
    rows.push(&[1, 3]);
    rows.extend([&[2][..]; 2]);
    rows
}

// ----- IRConverter worker ---------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// Resolves a path-valued environment variable, interpreting relative values
/// against the test's cwd first and the workspace root second.
fn resolve_var(name: &str) -> Option<PathBuf> {
    let raw = PathBuf::from(std::env::var_os(name)?);
    if raw.is_absolute() {
        return raw.is_file().then_some(raw);
    }
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(&raw));
    }
    candidates.push(workspace_root().join(&raw));
    candidates.into_iter().find(|path| path.is_file())
}

/// The x86_64 Rosetta worker and the x86_64 IRConverter library.
fn converter_paths() -> Result<(PathBuf, PathBuf), String> {
    let worker = resolve_var("VICTORY_IRCONVERTER_WORKER")
        .or_else(|| {
            let path =
                workspace_root().join("target/x86_64-apple-darwin/debug/victory-irconverter");
            path.is_file().then_some(path)
        })
        .ok_or_else(|| {
            "no x86_64 victory-irconverter worker: build it with \
             `cargo build -p victory-metal --bin victory-irconverter \
             --target x86_64-apple-darwin` or set VICTORY_IRCONVERTER_WORKER"
                .to_string()
        })?;
    let library = resolve_var("VICTORY_METAL_IRCONVERTER")
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
        .ok_or_else(|| {
            "no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)"
                .to_string()
        })?;
    Ok((worker, library))
}

fn convert(worker: &Path, library: &Path, dxil: &[u8], label: &str) -> Result<Vec<u8>, String> {
    let worker = ProcessWorker::new(worker.to_path_buf(), library.to_path_buf());
    match translate_dxil_with(
        None,
        &worker,
        dxil,
        "main",
        0,
        None,
        &TranslationOptions::default(),
        &[],
    ) {
        Ok(outcome) if !outcome.metallib.is_empty() => Ok(outcome.metallib),
        Ok(_) => Err(format!("{label}: IRConverter returned an empty metallib")),
        Err(error) => Err(format!("{label}: {error}")),
    }
}

// ----- tests ----------------------------------------------------------------

/// Hardware proof: an IRConverter mesh + fragment pair creates a mesh pipeline,
/// draws with `drawMeshThreadgroups`, and paints every pixel exactly.
#[test]
#[ignore = "hardware: needs the x86_64 IRConverter worker and an exclusive Metal device"]
fn irconverter_mesh_pair_draws_exact_pixels() {
    let mesh_dxil = mesh_shader(&[], 4, 4, 4, 0x0020_0400).expect("mesh fixture translates");
    let pixel_dxil = constant_pixel().expect("pixel fixture translates");
    let (worker, library) = converter_paths().expect("IRConverter worker and library");
    let mesh_lib = convert(&worker, &library, &mesh_dxil, "mesh").expect("mesh conversion");
    let pixel_lib = convert(&worker, &library, &pixel_dxil, "pixel").expect("pixel conversion");
    eprintln!(
        "IRConverter: mesh {} B, pixel {} B; worker={}",
        mesh_lib.len(),
        pixel_lib.len(),
        worker.display()
    );

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let device = transport.device_info().expect("device info");
    let spec = MeshPipelineSpec {
        // Position only: 4 vertices x 16 B = 64 declared output bytes.
        mesh_output_bytes: 4 * 16,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [4, 1, 1])
    };
    let pipeline = MeshPipeline::from_metallibs(
        &transport, &mesh_lib, "main", None, &pixel_lib, "main", &spec,
    )
    .expect("mesh pipeline should be created from the IRConverter pair");
    let pipeline_id = transport
        .register_mesh_pipeline(pipeline)
        .expect("mesh pipeline should register");

    const EDGE: u32 = 16;
    let target = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Rgba8Unorm)
        .expect("mesh render target should be created");
    let target_id = transport
        .register_render_target(&target)
        .expect("mesh render target should register");
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
        .expect("mesh submission should execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "mesh submission status");
    assert_eq!(stats.command_buffers, 1, "one command buffer");
    assert_eq!(stats.render_encoders, 1, "one render encoder");

    let pixels = transport
        .read_render_target_rgba8(&target)
        .expect("mesh target should read back");
    assert_eq!(pixels.len(), (EDGE * EDGE) as usize);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel,
            [255, 255, 255, 255],
            "mesh draw must paint pixel {index} white"
        );
    }
    eprintln!(
        "GPU: mesh pipeline drew {} px on {}; stats={stats:?}",
        EDGE * EDGE,
        device.name_str()
    );
}

/// Diagnostic: the over-budget IRConverter mesh library is rejected cleanly by
/// the mesh descriptor path.
///
/// `mesh_output_bytes` is deliberately `0` so the spec guard does not fire and
/// Metal's own behavior is observed. Metal returns the exact 32 KiB diagnostic
/// instead of crashing, which is why the budget is not the XPC cause.
#[test]
#[ignore = "hardware: needs the x86_64 IRConverter worker and an exclusive Metal device"]
fn irconverter_over_budget_mesh_descriptor_is_rejected_cleanly() {
    let mesh_dxil =
        mesh_shader(&wide_rows(), 64, 32, 32, 0x0000_0009).expect("wide mesh fixture translates");
    let pixel_dxil = constant_pixel().expect("pixel fixture translates");
    let (worker, library) = converter_paths().expect("IRConverter worker and library");
    let mesh_lib = convert(&worker, &library, &mesh_dxil, "mesh").expect("mesh conversion");
    let pixel_lib = convert(&worker, &library, &pixel_dxil, "pixel").expect("pixel conversion");
    eprintln!(
        "IRConverter: over-budget mesh {} B, pixel {} B",
        mesh_lib.len(),
        pixel_lib.len()
    );

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let spec = MeshPipelineSpec {
        mesh_output_bytes: 0, // observe Metal, not the spec guard
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [32, 1, 1])
    };
    let error = match MeshPipeline::from_metallibs(
        &transport, &mesh_lib, "main", None, &pixel_lib, "main", &spec,
    ) {
        Ok(_) => panic!("an over-budget mesh descriptor must be rejected"),
        Err(error) => error,
    };
    let message = error.to_string();
    eprintln!("GPU: over-budget mesh descriptor refused: {message}");
    assert!(
        message.contains("Total mesh size (33280) exceeds the maximum mesh size allowed (32768)"),
        "expected Metal's 32 KiB diagnostic, got: {message}"
    );
}

/// Diagnostic: the historical `load_metallib(mesh)` XPC failure.
///
/// A *within-budget* mesh metallib (the one the draw test uses) is handed to
/// the compute path. The library load succeeds; the compute pipeline compile of
/// a mesh-stage function is what crashes `MTLCompilerService`.
#[test]
#[ignore = "hardware: reproduces the MTLCompilerService crash for a mesh function on the compute path"]
fn mesh_function_through_the_compute_path_fails_with_xpc() {
    let mesh_dxil = mesh_shader(&[], 4, 4, 4, 0x0020_0400).expect("mesh fixture translates");
    let pixel_dxil = constant_pixel().expect("pixel fixture translates");
    let (worker, library) = converter_paths().expect("IRConverter worker and library");
    let mesh_lib = convert(&worker, &library, &mesh_dxil, "mesh").expect("mesh conversion");
    let pixel_lib = convert(&worker, &library, &pixel_dxil, "pixel").expect("pixel conversion");
    eprintln!(
        "IRConverter: within-budget mesh {} B, pixel {} B",
        mesh_lib.len(),
        pixel_lib.len()
    );

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let error = transport
        .load_metallib(&mesh_lib, "main")
        .expect_err("a mesh function must not create a compute pipeline");
    let message = error.to_string();
    eprintln!("GPU: mesh via load_metallib refused: {message}");
    assert!(
        message.contains("XPC_ERROR_CONNECTION_INTERRUPTED"),
        "expected the recorded Metal compiler XPC failure, got: {message}"
    );
}

/// The spec guard refuses the same 33,280-byte declaration before Metal sees
/// it. This is the local defense for the failure the test above records.
#[test]
fn over_budget_mesh_output_is_refused_before_loading() {
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let spec = MeshPipelineSpec {
        mesh_output_bytes: 33_280,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [32, 1, 1])
    };
    // The guard runs before any library bytes are touched; garbage is fine.
    let error = match MeshPipeline::from_metallibs(
        &transport,
        b"not a metallib",
        "main",
        None,
        b"not a metallib",
        "main",
        &spec,
    ) {
        Ok(_) => panic!("the 32 KiB guard must fire first"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "mesh pipeline budget: mesh output declaration exceeds Metal's 32 KiB budget"
    );
}
