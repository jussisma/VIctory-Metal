//! Paired hardware proof: a mesh/pixel fixture whose declared outputs exceed
//! Metal's 32 KiB budget is packed under it by the repository's linked packer,
//! converted with Apple's IRConverter, and drawn through
//! `MTLMeshRenderPipelineDescriptor` on the real Metal device.
//!
//! The fixture is the 128-component shape the `MESH-19` analysis used: 29 full
//! `TEXCOORD` rows, one row emitted as 1 + 3 components and two 2-component
//! rows at 64 declared vertices. Standalone, its declared alignment is 33,280 B
//! (over the 32,768 B limit). The pixel stage reads only `TEXCOORD0.x`, so the
//! linked packer prunes the other 31 arbitrary rows and compacts the survivor
//! to 5 components (position + `TEXCOORD0.x`) = 1,280 B.
//!
//! The inputs are console containers in the format `victory_shader` reads: an
//! `XDAT` index holding one `XBCH`-packed `FSSM` blob with the GFX10 code and
//! stage data, plus the `OSG1`/`ISG1` signature parts. That lets the test call
//! the public linked packer (`victory_shader::emit::translate_linked`) on the
//! same fixture the CPU proof uses, rather than approximating it.
//!
//! What this asserts on hardware:
//! - the standalone declaration is 33,280 B and the packed pair is 1,280 B;
//! - both emitted stages translate through IRConverter (`x86_64` Rosetta
//!   worker when configured);
//! - the mesh pipeline is created and `drawMeshThreadgroups` paints every pixel
//!   of a 16x16 target with the interpolated `TEXCOORD0.x` value, compared
//!   pixel-for-pixel against a CPU-computed expectation;
//! - the same fixture with packing disabled is refused: the spec/proof path
//!   returns the explicit 32 KiB budget error, and Metal itself refuses the
//!   descriptor with `Total mesh size (33280) exceeds the maximum mesh size
//!   allowed (32768)` rather than silently drawing.
//!
//! Run with:
//! ```text
//! VICTORY_IRCONVERTER_WORKER=<x86_64 victory-irconverter> \
//! VICTORY_METAL_IRCONVERTER=<libmetalirconverter.dylib> \
//! cargo test -p victory-metal --target aarch64-apple-darwin \
//!     --test mesh_pair_gpu -- --ignored --test-threads=1 --nocapture
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
use victory_shader::emit::{
    METAL_MESH_OUTPUT_BYTES, Options, mesh_output_layout, mesh_outputs, mesh_pack_proof, translate,
    translate_linked,
};
use victory_shader::native::{MeshIo, NativeProgram, PsInput, StageIo, Usage, VsOutput};
use victory_shader::{FourCC, HwStage};

// ----- fixture --------------------------------------------------------------

const VERTICES: u32 = 64;
const PRIMITIVES: u32 = 32;
const THREADS: u32 = 32;
/// Mesh `exp prim` literal packing indices `(0, 1, 2)` in 10-bit fields.
const INDICES_0_1_2: u32 = 0x0020_0400;

/// A `DXBC` container from parts (the shader crate's builder is test-only).
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
    /// D3D system value (`0` arbitrary, `1` position, `64` target).
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

/// The `MESH-19` fixture rows: 29 full rows, one 1+3 row, two 2-wide rows.
fn wide_rows() -> Vec<&'static [u8]> {
    let mut rows: Vec<&'static [u8]> = vec![&[4]; 29];
    rows.push(&[1, 3]);
    rows.extend([&[2][..]; 2]);
    rows
}

/// `OSG1` declaration of the wide fixture, and the native export list that
/// writes it: the lane bit pattern `(4x - 1, 2y - 1, 0.5, 1)` per row.
fn wide_outputs() -> (Vec<Row<'static>>, Vec<VsOutput>) {
    let rows = wide_rows();
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
        outputs.push(VsOutput {
            usage: Usage::Generic,
            row: row as u8,
            param: param as u8,
            fp16: false,
            pack2: false,
        });
        param += 1;
    }
    (osg, outputs)
}

/// The wide mesh program's GFX10 code.
fn mesh_words() -> Vec<u32> {
    let rows = wide_rows();
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
        INDICES_0_1_2,
    ];
    for (row, widths) in (1..).zip(&rows) {
        let col: u8 = widths.iter().sum();
        let mask = ((1u32 << col) - 1) | ((32 + (row - 1)) << 4);
        words.extend([0xF800_0000 | mask, 0x0706_0504]);
    }
    words.extend([
        0xF800_0941, // exp prim v9, off, off, off done
        0x0000_0009,
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ]);
    words
}

/// Pixel program reading `TEXCOORD0.x` and writing it to all four channels.
fn pixel_words() -> Vec<u32> {
    let p1 =
        |vdst: u32, attr: u32, ch: u32| 0xC800_0000u32 | (vdst << 18) | (attr << 10) | (ch << 8);
    let p2 =
        |vdst: u32, attr: u32, ch: u32| 0xC801_0001u32 | (vdst << 18) | (attr << 10) | (ch << 8);
    vec![
        p1(2, 0, 0),
        p2(2, 0, 0),
        0xF800_180F,
        0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
        0xBF81_0000,
    ]
}

// ----- console container encoding ------------------------------------------
//
// `victory_shader::emit::translate*` reads a console container: the `DXIL`
// part holds an `XDAT` index, one entry of which is an `XBCH`-packed `FSSM`
// blob with the native records (see `victory_shader::native`). The synthetic
// fixtures above only have signature parts, so this test writes that part too;
// the stage data is what `native_programs` reads the I/O from.

fn put_u16(out: &mut [u8], at: usize, value: u16) {
    out[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn usage_raw(usage: Usage) -> u8 {
    match usage {
        Usage::Generic => 6,
        Usage::ClipDistance => 7,
        Usage::Other(value) => value,
    }
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
///
/// `native::unpack` inflates with `miniz_oxide`, which accepts stored blocks;
/// this keeps the test free of a compression dependency.
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

/// `FSSM` payload offsets: header (slot table), then one record + stage data +
/// code. The slot table is sized for the compute slot (`0x60..0x68`).
const FSSM_PAYLOAD_HEADER: usize = 0x68;
const RECORD_SIZE: usize = 0x54;
/// Stage data spans every field this test's stages use; code follows it.
const STAGE_DATA: usize = 0xFAC;
/// `PS_PATCHES + 8 * PATCH_BYTES` in `victory_shader::native`.
const PATCH_BYTES: usize = 4 + 10 * 10 * 4;

/// One `FSSM` blob with a single program at its stage's slot.
fn fssm_blob(native: &NativeProgram) -> Vec<u8> {
    let slot = match native.stage {
        HwStage::Ps => 0x30usize,
        HwStage::Ms => 0x48usize,
        other => panic!("unsupported fixture stage {other:?}"),
    };
    let record_at = FSSM_PAYLOAD_HEADER;
    let stage_at = record_at + RECORD_SIZE;
    let code_at = stage_at + STAGE_DATA;
    let mut payload = vec![0u8; FSSM_PAYLOAD_HEADER];
    put_u64(&mut payload, 0, 6); // graphics slot count
    for at in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        put_u64(&mut payload, at, 0xDEAD_BEEF_DEAD_BEEF); // absent stage
    }
    put_u64(&mut payload, slot, record_at as u64);
    payload.resize(code_at, 0);
    // Record: code range, flags (wave32), LDS, VGPRs.
    put_u64(&mut payload, record_at, code_at as u64);
    put_u64(&mut payload, record_at + 8, native.code.len() as u64);
    put_u32(&mut payload, record_at + 0x14, 1 << 5);
    put_u32(&mut payload, record_at + 0x4c, 0);
    put_u32(&mut payload, record_at + 0x50, u32::from(native.vgprs));
    match &native.io {
        StageIo::Ms(ms) => {
            assert!(ms.outputs.len() <= 32);
            for (i, output) in ms.outputs.iter().enumerate() {
                let at = stage_at + 0xc8 + 6 * i;
                payload[at] = usage_raw(output.usage);
                payload[at + 1] = output.row;
                payload[at + 2] = output.param;
                payload[at + 3] = output.row;
                payload[at + 4] = 0;
                payload[at + 5] = u8::from(output.fp16) | (u8::from(output.pack2) << 1);
            }
            put_u32(&mut payload, stage_at + 0x188, ms.outputs.len() as u32);
            put_u32(&mut payload, stage_at + 0x194, ms.clip_cntl);
            let info = stage_at + 0x448;
            put_u16(&mut payload, info, ms.threads as u16);
            put_u16(&mut payload, info + 2, ms.max_vertices as u16);
            put_u16(&mut payload, info + 4, ms.max_primitives as u16);
            payload[info + 6] = 0; // user SGPRs
            payload[info + 7] = ms.topology;
        }
        StageIo::Ps {
            inputs,
            input_addr,
            db_shader_control,
            ..
        } => {
            assert!(inputs.len() <= 32);
            for (i, input) in inputs.iter().enumerate() {
                let at = stage_at + 0x24 + 8 * i;
                payload[at] = usage_raw(input.usage);
                payload[at + 1] = input.row;
                payload[at + 2] = input.mask;
                payload[at + 3] = input.attr;
                payload[at + 4] = 0;
                payload[at + 5] = input.row2;
                payload[at + 6] = 0;
                payload[at + 7] = u8::from(input.flat)
                    | (u8::from(input.fp16) << 3)
                    | (u8::from(input.pack2) << 4);
            }
            put_u32(&mut payload, stage_at + 0x124, inputs.len() as u32);
            put_u32(&mut payload, stage_at + 0x134, *input_addr);
            put_u32(&mut payload, stage_at + 0x138, *db_shader_control);
            put_u32(&mut payload, stage_at + 0x148, 0);
            for mrt in 0..8 {
                put_u32(&mut payload, stage_at + 0x14c + mrt * PATCH_BYTES, u32::MAX);
            }
        }
        other => panic!("unsupported fixture I/O {other:?}"),
    }
    payload.truncate(code_at);
    payload.extend_from_slice(&native.code);

    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);
    fssm
}

/// `DXIL` part data: `XDAT` index with one XBCH-packed `FSSM` blob.
fn xdat_part(native: &NativeProgram) -> Vec<u8> {
    let fssm = fssm_blob(native);
    let packed = zlib_stored(&fssm);
    let mut blob = b"XBCH".to_vec();
    blob.extend(3u32.to_le_bytes());
    blob.extend((packed.len() as u64).to_le_bytes());
    blob.extend((fssm.len() as u64).to_le_bytes());
    blob.extend(packed);

    let mut out = vec![0u8; 0x18];
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

/// The mesh container and its paired pixel container.
struct Fixture {
    mesh: Vec<u8>,
    pixel: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let (osg, outputs) = wide_outputs();
        let mesh_native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 10,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs,
                clip_cntl: 0,
                threads: THREADS,
                max_vertices: VERTICES,
                max_primitives: PRIMITIVES,
                topology: 2,
                lds_bytes: 0,
            }),
            code: mesh_words().iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mesh = build_container(&[
            (FourCC::DXIL, xdat_part(&mesh_native)),
            (FourCC::OSG1, signature_part(&osg)),
        ]);

        let pixel_native = NativeProgram {
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
            code: pixel_words().iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let isg = signature_part(&[Row {
            name: "TEXCOORD",
            index: 0,
            sv: 0,
            register: 1,
            mask: 0b0001,
            raw: 3,
        }]);
        let psg = signature_part(&[Row {
            name: "SV_Target",
            index: 0,
            sv: 64,
            register: 0,
            mask: 0xF,
            raw: 3,
        }]);
        let pixel = build_container(&[
            (FourCC::DXIL, xdat_part(&pixel_native)),
            (FourCC::ISG1, isg),
            (FourCC::OSG1, psg),
        ]);
        Self { mesh, pixel }
    }
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
            std::env::var("HOME").ok().and_then(|home| {
                let path = PathBuf::from(home).join(
                    "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                     D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
                );
                path.is_file().then_some(path)
            })
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

/// The expected RGBA8 value of `TEXCOORD0.x` at a pixel centre.
///
/// The mesh fixture writes `clip.x` into `TEXCOORD0.x` at every vertex with
/// `w = 1`, so the perspective-correct interpolation equals the pixel's NDC x.
fn expected_column(column: u32) -> u8 {
    let ndc = (column as f32 + 0.5) / 16.0 * 2.0 - 1.0;
    (ndc.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Hardware proof: the packed linked pair creates a mesh pipeline, draws with
/// `drawMeshThreadgroups`, and paints the interpolated attribute exactly.
#[test]
#[ignore = "hardware: needs the x86_64 IRConverter worker and an exclusive Metal device"]
fn packed_pair_draws_exact_pixels() {
    let fixture = Fixture::new();

    // The packer's proof on the emitted stages: 33,280 B declared, 1,280 B
    // carried after pruning the 31 arbitrary rows the pixel never reads.
    let standalone = translate(&fixture.mesh).expect("standalone mesh translates");
    let pixel_host = translate(&fixture.pixel).expect("pixel translates");
    let declared = mesh_output_layout(&standalone.dxil)
        .expect("the emitted mesh container parses")
        .expect("a mesh output declaration");
    assert_eq!(
        declared.aligned_before, 33_280,
        "the wide fixture's declared aligned footprint"
    );
    assert!(declared.aligned_before > METAL_MESH_OUTPUT_BYTES);
    let proof = mesh_pack_proof(&standalone.dxil, Some(&pixel_host.dxil))
        .expect("the emitted mesh container parses")
        .expect("a mesh output declaration");
    eprintln!(
        "packer: declared {declared_al} B aligned / {declared_packed} B packed -> \
         {aligned} B aligned / {packed} B packed, kept {kept}, pruned {pruned}, fits {fits}",
        declared_al = proof.declared_aligned,
        declared_packed = proof.declared_packed,
        aligned = proof.aligned_after,
        packed = proof.packed_after,
        kept = proof.retained.len(),
        pruned = proof.pruned.len(),
        fits = proof.fits,
    );
    assert_eq!(proof.declared_aligned, 33_280);
    assert!(proof.linkage_proven, "the pixel stage is identified");
    assert!(proof.fits, "the linked pair fits: {proof:?}");
    assert_eq!(
        proof.aligned_after, 1_280,
        "position + TEXCOORD0.x at 64 vertices"
    );
    assert_eq!(proof.packed_after, 1_280);
    assert!(proof.pruned.iter().any(|row| row.index == 1));

    // The real linked path (the packer the runtime uses), on the same input.
    let (linked_mesh, linked_pixel) =
        translate_linked(&fixture.mesh, Some(&fixture.pixel), Options::default())
            .expect("the linked packer translates the pair");
    let linked_pixel = linked_pixel.expect("a paired pixel stage");
    let linked = mesh_outputs(&linked_mesh.dxil).expect("the linked mesh declares outputs");
    let linked_bytes = linked.bytes();
    eprintln!(
        "linked: {vertices} vertices x {components} components = {linked_bytes} B \
         (limit {METAL_MESH_OUTPUT_BYTES}), pixel DXIL {pixel_dxil} B",
        vertices = linked.vertices,
        components = linked.vertex_components,
        pixel_dxil = linked_pixel.dxil.len(),
    );
    assert_eq!(linked.vertices, VERTICES);
    assert_eq!(linked.vertex_components, 5);
    assert_eq!(linked_bytes, proof.packed_after);
    assert!(linked_bytes <= METAL_MESH_OUTPUT_BYTES);
    let linked_layout = mesh_output_layout(&linked_mesh.dxil)
        .expect("the linked container parses")
        .expect("a mesh output declaration");
    assert_eq!(linked_layout.aligned_before, linked_bytes);
    assert!(linked_layout.aligned_before <= METAL_MESH_OUTPUT_BYTES);

    // Both linked stages through Apple's IRConverter.
    let (worker, library) = converter_paths().expect("IRConverter worker and library");
    let mesh_lib = convert(&worker, &library, &linked_mesh.dxil, "mesh").expect("mesh conversion");
    let pixel_lib =
        convert(&worker, &library, &linked_pixel.dxil, "pixel").expect("pixel conversion");
    eprintln!(
        "IRConverter: mesh {} B, pixel {} B; x86_64 worker={}",
        mesh_lib.len(),
        pixel_lib.len(),
        worker.display()
    );

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let device = transport.device_info().expect("device info");
    let spec = MeshPipelineSpec {
        mesh_output_bytes: linked_bytes as usize,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [THREADS, 1, 1])
    };
    let pipeline = MeshPipeline::from_metallibs(
        &transport, &mesh_lib, "main", None, &pixel_lib, "main", &spec,
    )
    .expect("the packed mesh pipeline should be created");
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
    let mut checksum = 0u64;
    let mut columns = Vec::new();
    for column in 0..EDGE {
        let value = expected_column(column);
        columns.push(value);
        for row in 0..EDGE {
            let index = (row * EDGE + column) as usize;
            let pixel = pixels[index];
            checksum = checksum
                .wrapping_mul(31)
                .wrapping_add(u64::from(pixel[0]))
                .wrapping_add(u64::from(pixel[1]) << 8)
                .wrapping_add(u64::from(pixel[2]) << 16)
                .wrapping_add(u64::from(pixel[3]) << 24);
            assert_eq!(
                pixel,
                [value, value, value, value],
                "pixel ({column}, {row}) must carry TEXCOORD0.x = {value}"
            );
        }
    }
    eprintln!(
        "GPU: mesh pipeline drew {} px on {}; cols {:?}; checksum {checksum:#018x}; stats={stats:?}",
        EDGE * EDGE,
        device.name_str(),
        columns,
    );
}

/// The same fixture with packing disabled: the declared 33,280 B output is
/// refused with an explicit budget error, by the spec guard and by Metal.
#[test]
#[ignore = "hardware: needs the x86_64 IRConverter worker and an exclusive Metal device"]
fn unpacked_pair_above_32k_is_refused() {
    let fixture = Fixture::new();
    let standalone = translate(&fixture.mesh).expect("standalone mesh translates");
    let pixel_host = translate(&fixture.pixel).expect("pixel translates");
    let declared = mesh_output_layout(&standalone.dxil)
        .expect("the emitted mesh container parses")
        .expect("a mesh output declaration");
    eprintln!(
        "unpacked: {vertices} vertices, aligned {} B, packed {} B",
        declared.aligned_before,
        declared.packed_before,
        vertices = declared.vertices,
    );
    assert_eq!(declared.aligned_before, 33_280);
    assert!(declared.aligned_before > METAL_MESH_OUTPUT_BYTES);

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    // The pipeline spec guard refuses the declared footprint before any
    // metallib is loaded.
    let spec = MeshPipelineSpec {
        mesh_output_bytes: declared.aligned_before as usize,
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [THREADS, 1, 1])
    };
    let error = match MeshPipeline::from_metallibs(
        &transport,
        b"not a metallib",
        "main",
        None,
        b"not a metallib",
        "main",
        &spec,
    ) {
        Ok(_) => panic!("an over-budget mesh declaration must be refused"),
        Err(error) => error,
    };
    eprintln!("pipeline guard: {error}");
    assert_eq!(
        error.to_string(),
        "mesh pipeline budget: mesh output declaration exceeds Metal's 32 KiB budget"
    );

    // With the guard skipped, Metal itself refuses the converted library.
    let (worker, library) = converter_paths().expect("IRConverter worker and library");
    let mesh_lib = convert(&worker, &library, &standalone.dxil, "mesh").expect("mesh conversion");
    let pixel_lib =
        convert(&worker, &library, &pixel_host.dxil, "pixel").expect("pixel conversion");
    eprintln!(
        "IRConverter: unpacked mesh {} B, pixel {} B; x86_64 worker={}",
        mesh_lib.len(),
        pixel_lib.len(),
        worker.display()
    );
    let spec = MeshPipelineSpec {
        mesh_output_bytes: 0, // observe Metal, not the spec guard
        ..MeshPipelineSpec::new(vec![PixelFormat::Rgba8Unorm], None, [THREADS, 1, 1])
    };
    let error = match MeshPipeline::from_metallibs(
        &transport, &mesh_lib, "main", None, &pixel_lib, "main", &spec,
    ) {
        Ok(_) => panic!("Metal must refuse an over-budget mesh descriptor"),
        Err(error) => error,
    };
    let message = error.to_string();
    eprintln!("Metal: {message}");
    assert!(
        message.contains("Total mesh size (33280) exceeds the maximum mesh size allowed (32768)"),
        "expected Metal's 32 KiB diagnostic, got: {message}"
    );
}
