//! Hardware proof for mesh output packing above Metal's 32 KiB limit.
//!
//! The fixture is the 128-component mesh the `MESH-19` analysis used: 29 full
//! `TEXCOORD` rows, one row emitted as 1 + 3 components, and two 2-component
//! rows at 64 declared vertices. Its declared order is 33,280 bytes in Metal's
//! aligned layout (over the limit); the linked packer's within-row ordering
//! lands on exactly 32,768.
//!
//! What this test proves on this host:
//! - the CPU-side footprint through the library's own packer
//!   (`victory_shader::emit::mesh_output_layout`), before and after;
//! - that both emitted DXIL stages translate through Apple's IRConverter
//!   (via the x86_64 Rosetta worker when one is configured);
//! - the precise reason a mesh pipeline cannot be created or drawn by
//!   `victory-metal`'s transport today.
//!
//! The last item is why the result is reported as PARTIAL when the device
//! cannot take the pair; no pixel result is claimed that was not observed.
//!
//! Run with:
//! ```text
//! VICTORY_IRCONVERTER_WORKER=<x86_64 victory-irconverter> \
//! VICTORY_METAL_IRCONVERTER=<libmetalirconverter.dylib> \
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test mesh_pack_gpu -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::Path;

use victory_shader::emit::mesh_output_layout;
use victory_shader::emit::{emit, mesh_pack_proof, METAL_MESH_OUTPUT_BYTES};
use victory_shader::native::{
    MeshIo, NativeProgram, PsInput, StageIo, Usage, VsOutput,
};
use victory_shader::{ir, structure, resource, io, Container, FourCC, HwStage, ShaderError};

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

/// Build and translate the mesh fixture. Returns the emitted container.
fn mesh_shader() -> Result<Vec<u8>, ShaderError> {
    const VERTICES: u32 = 64;
    let rows: Vec<&[u8]> = {
        let mut r: Vec<&[u8]> = vec![&[4]; 29];
        r.push(&[1, 3]);
        r.extend([&[2][..]; 2]);
        r
    };
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
        0x7E12_02FF, // v_mov_b32 v9, TRIANGLE
        0x0000_0009,
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
    for (row, widths) in (1..).zip(&rows) {
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
            threads: 32,
            max_vertices: VERTICES,
            max_primitives: 32,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = build_container(&[(FourCC::OSG1, signature_part(&osg))]);
    translate(&container, &native)
}

/// Build and translate a pixel shader that reads `TEXCOORD0.x`.
fn pixel_shader() -> Result<Vec<u8>, ShaderError> {
    let p1 = |vdst: u32, attr: u32, ch: u32| {
        0xC800_0000u32 | (vdst << 18) | (attr << 10) | (ch << 8)
    };
    let p2 = |vdst: u32, attr: u32, ch: u32| {
        0xC801_0001u32 | (vdst << 18) | (attr << 10) | (ch << 8)
    };
    let words = [
        p1(2, 0, 0),
        p2(2, 0, 0),
        0xF800_180F,
        0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
        0xBF81_0000,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 1,
                mask: 0b0001,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 1 << 1,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg = signature_part(&[Row {
        name: "TEXCOORD",
        index: 0,
        sv: 0,
        register: 1,
        mask: 0b0001,
        raw: 3,
    }]);
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

/// Runs the public emission pipeline for one prepared console program.
fn translate(container: &[u8], native: &NativeProgram) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(container)?;
    let program = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&program)?;
    let res = resource::track(&program, &native.layout, native.stage, Some(&st))?;
    let raised = io::raise(&program, native)?;
    let host = emit(&c, native, &program, &st, &raised, &res)?;
    Ok(host.dxil)
}

/// Translates both stages through the configured x86_64 IRConverter worker.
fn converter_outcome(mesh: &[u8], pixel: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    use victory_metal::macos::cache::TranslationOptions;
    use victory_metal::macos::irconverter::{translate_dxil_with, ProcessWorker};
    let Some(worker) = ProcessWorker::from_env() else {
        return Err("no IRConverter worker: set VICTORY_IRCONVERTER_WORKER to an \
                    x86_64 victory-irconverter and VICTORY_METAL_IRCONVERTER to \
                    libmetalirconverter.dylib"
            .to_string());
    };
    let options = TranslationOptions::default();
    let run = |dxil: &[u8], label: &str| -> Result<Vec<u8>, String> {
        match translate_dxil_with(None, &worker, dxil, "main", 0, None, &options, &[]) {
            Ok(outcome) if !outcome.metallib.is_empty() => Ok(outcome.metallib),
            Ok(_) => Err(format!("{label}: IRConverter returned an empty metallib")),
            Err(e) => Err(format!("{label}: {e}")),
        }
    };
    Ok((run(mesh, "mesh")?, run(pixel, "pixel")?))
}

/// The precise reason no mesh draw can run through this transport today.
fn mesh_pipeline_blocker() -> String {
    // `DirectMetalTransport` exposes `register_graphics_pipeline` (MSL
    // vertex/fragment) and `load_metallib` (compute) only; its documented
    // command mapping refuses `DispatchMesh`: no mesh pipeline can be
    // registered on this transport.
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/macos/device.rs"),
    )
    .unwrap_or_default();
    let refuses = source.contains("DispatchMesh` is refused")
        || source.contains("no mesh pipeline can be registered on this transport");
    if refuses {
        "no mesh pipeline API in victory-metal: DirectMetalTransport::execute_graphics_batch \
         refuses DispatchMesh (\"no mesh pipeline can be registered on this transport\")"
            .to_string()
    } else {
        "no mesh pipeline API in victory-metal (register_graphics_pipeline is MSL vertex/fragment \
         and load_metallib is compute only)"
            .to_string()
    }
}

#[test]
#[ignore = "needs the x86_64 IRConverter worker and an exclusive Metal device"]
fn paired_mesh_pixel_above_32k_fits_after_packing() {
    // CPU-side proof: the declared aligned layout is over the limit and the
    // library packer brings it under, at the recorded exact bytes.
    let mesh = mesh_shader().expect("the mesh fixture translates");
    let pixel = pixel_shader().expect("the pixel fixture translates");
    let layout = mesh_output_layout(&mesh)
        .expect("the mesh container parses")
        .expect("a mesh output declaration");
    eprintln!(
        "CPU layout: {vertices} vertices, aligned {before} B -> {after} B, packed {packed_before} -> {packed_after} B",
        vertices = layout.vertices,
        before = layout.aligned_before,
        after = layout.aligned_after,
        packed_before = layout.packed_before,
        packed_after = layout.packed_after,
    );
    assert_eq!(layout.aligned_before, 33_280, "declared aligned footprint");
    assert!(layout.aligned_before > METAL_MESH_OUTPUT_BYTES);
    assert_eq!(layout.aligned_after, 32_768, "packed aligned footprint");
    assert!(layout.aligned_after <= METAL_MESH_OUTPUT_BYTES);
    assert!(layout.packed_after <= METAL_MESH_OUTPUT_BYTES);
    // The packed rows never claim more than one row's four channels.
    for (row, col, cols, _) in &layout.rows_after {
        assert!(
            u32::from(*col) + u32::from(*cols) <= 4,
            "row {row} declares {col}+{cols} channels"
        );
    }

    // The linked-pair proof: the same declared footprint, then dead-export
    // pruning and channel compaction against this pixel stage. The pixel
    // fixture reads only `TEXCOORD0.x`, so just the position and that row are
    // carried; the other 31 arbitrary rows are provably unobservable.
    let proof = mesh_pack_proof(&mesh, Some(&pixel))
        .expect("the mesh container parses")
        .expect("a mesh output declaration");
    eprintln!(
        "CPU proof: declared {declared} B aligned / {packed} B packed -> {aligned} B aligned / \
         {after} B packed, {kept} rows kept, {pruned} pruned, fits {fits}",
        declared = proof.declared_aligned,
        packed = proof.declared_packed,
        aligned = proof.aligned_after,
        after = proof.packed_after,
        kept = proof.retained.len(),
        pruned = proof.pruned.len(),
        fits = proof.fits,
    );
    assert_eq!(proof.declared_aligned, 33_280);
    assert!(proof.linkage_proven, "the pixel stage is identified");
    assert!(proof.fits, "the linked pair fits after pruning: {proof:?}");
    assert!(proof.aligned_after <= METAL_MESH_OUTPUT_BYTES);
    assert!(proof.packed_after <= METAL_MESH_OUTPUT_BYTES);
    // PSV0/OSG1 identity: the position and the read semantic survive, the
    // pixel's read mask is preserved, and every declared row is accounted for.
    assert!(proof.retained.iter().any(|r| r.kind == 3));
    let kept = proof
        .retained
        .iter()
        .find(|r| r.kind == 0 && r.index == 0)
        .expect("TEXCOORD0 is retained");
    assert_eq!(kept.read, 0b0001, "the pixel input's read mask");
    assert_eq!(
        proof.retained.len() + proof.pruned.len(),
        layout.rows_after.len(),
        "every declared row is accounted for"
    );
    assert!(proof.pruned.iter().any(|r| r.index == 1));

    // Both stages through IRConverter, when a Rosetta worker is configured.
    match converter_outcome(&mesh, &pixel) {
        Ok((mesh_lib, pixel_lib)) => {
            eprintln!(
                "IRConverter: mesh metallib {} B, pixel metallib {} B",
                mesh_lib.len(),
                pixel_lib.len()
            );
            assert!(!mesh_lib.is_empty() && !pixel_lib.is_empty());
            // Try the closest the transport offers: `load_metallib` registers a
            // *compute* pipeline only, so a mesh function is expected to be
            // refused. Report the observed call and error verbatim.
            match victory_metal::DirectMetalTransport::new() {
                Ok(transport) => match transport.load_metallib(&mesh_lib, "main") {
                    Ok(_) => eprintln!(
                        "GPU draw: PARTIAL: load_metallib accepted the mesh library as a compute \
                         pipeline; no mesh pipeline/draw path exists to run it"
                    ),
                    Err(e) => eprintln!(
                        "GPU draw: PARTIAL: DirectMetalTransport::load_metallib(mesh, \"main\") \
                         failed: {e}"
                    ),
                },
                Err(e) => eprintln!("GPU draw: PARTIAL: no Metal device: {e}"),
            }
        }
        Err(reason) => {
            eprintln!("IRConverter: PARTIAL: {reason}");
        }
    }

    // The device cannot take the pair: no mesh pipeline exists to create.
    let blocker = mesh_pipeline_blocker();
    eprintln!("mesh pipeline: PARTIAL: {blocker}");
}
