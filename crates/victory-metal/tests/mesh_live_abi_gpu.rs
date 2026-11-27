//! Live mesh ABI proof.
//!
//! The Cycle 04 mesh work proved that a packed mesh/pixel pair above Metal's
//! 32 KiB mesh-output budget can be created and drawn through a direct
//! `MeshPipeline` harness. It did not prove the *live* path: a wire v3
//! `SetMeshPipeline` + `DispatchMesh` stream whose converted root signature
//! binds root constants, CBV/SRV/UAV root views and descriptor-table argument
//! buffers, with every allocation resident in the transport's
//! `MTLResidencySet`.
//!
//! This file has three parts:
//!
//! * [`packed_pair_and_binding_plan`] is pure CPU/reflection accounting: the
//!   declared footprint is 33,280 B, the linked packer carries 1,280 B, and a
//!   converted root-signature layout maps root constants, root views and a
//!   descriptor table onto one argument-buffer plan.
//! * [`live_submit_v3_mesh_draw_is_exact`] runs the linked pair through the
//!   Rosetta IRConverter, registers it as a content-hashed mesh PSO under a wire
//!   handle with the reflected ABI, submits `BeginRenderPass` /
//!   `SetMeshPipeline` / descriptor table / bindings / `DispatchMesh` /
//!   `EndRenderPass` through `submit_v3_with_stats`, and checks the exact pixel
//!   checksum after the submission completes.
//! * [`mesh_dispatch_without_a_selected_pipeline_is_refused`] is the negative
//!   regression: an unknown mesh handle and a dispatch without a preceding
//!   `SetMeshPipeline` are refused before any command buffer exists.
//!
//! Run with the exclusive GPU lock:
//! ```text
//! sh -c 'while ! mkdir /tmp/victory-gpu.lock 2>/dev/null; do sleep 5; done;
//!        trap "rmdir /tmp/victory-gpu.lock" EXIT;
//!        export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//!        cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!          --test mesh_live_abi_gpu -- --ignored --test-threads=1 --nocapture'
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::bindings::{
    DescriptorTableKind, ReflectedStage, RootParameterKind, RootParameterSpec, RootSignatureLayout,
};
use victory_metal::macos::cache::TranslationOptions;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::irconverter::{ProcessWorker, translate_dxil_with};
use victory_metal::macos::mesh::{
    MAX_MESH_ROOT_CONSTANT_DWORDS, MeshAbiPlan, MeshTableExtent,
};
use victory_metal::macos::registry::{
    PipelineStage, PipelineState, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::wire::{self, BufferBinding, ReplyStatus};
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DescriptorKind, DescriptorTableV3, LoadOp,
    PipelineRef, RootBufferView, RootConstantsV3, StoreOp, SubmissionV3,
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

/// Real `IRShaderReflectionCopyJSONString` output for the packed fixture's
/// linked mesh stage. Captured on this host with the x86_64 worker; the stage
/// reads no root signature, so its top-level argument buffer is empty.
const MESH_REFLECTION: &str = r#"{"ConstantBufferIndices":[],"EntryPoint":"main","Features":0,"FunctionConstants":[],"HasYieldLoop":false,"NeedsFunctionConstants":false,"ResourceCount":0,"ResourceUsages":0,"SamplerIndices":[],"ShaderID":"1621887162674351779516453121658487405624","ShaderResourceViewIndices":[],"ShaderType":"Mesh","TopLevelArgumentBuffer":[],"UnorderedAccessViewIndices":[],"UsedResources":[],"UsesRayQuery":false,"state":{"max_payload_size_in_bytes":0,"max_primitive_output_count":32,"max_vertex_output_count":64,"num_threads":[32,1,1],"primitive_topology":"Triangle"}}"#;

/// Real reflection for the linked pixel stage (also no root signature).
const PIXEL_REFLECTION: &str = r#"{"ConstantBufferIndices":[],"EntryPoint":"main","Features":0,"FunctionConstants":[],"HasYieldLoop":false,"NeedsFunctionConstants":false,"ResourceCount":0,"ResourceUsages":0,"SamplerIndices":[],"ShaderID":"732193972597990833616453121658487405624","ShaderResourceViewIndices":[],"ShaderType":"Fragment","TopLevelArgumentBuffer":[],"UnorderedAccessViewIndices":[],"UsedResources":[],"UsesRayQuery":false,"state":{"discards":false,"num_render_targets":1,"rt_index_int":0}}"#;

/// CPU/reflection proof: the declared output is 33,280 B, the linked packer
/// carries 1,280 B, and a converted binding plan accounts for inline root
/// constants, root CBV/SRV/UAV views and a packed descriptor table.
#[test]
fn packed_pair_and_binding_plan() {
    let fixture = Fixture::new();
    let standalone = translate(&fixture.mesh).expect("standalone mesh translates");
    let pixel_host = translate(&fixture.pixel).expect("pixel translates");
    let declared = mesh_output_layout(&standalone.dxil)
        .expect("the emitted mesh container parses")
        .expect("a mesh output declaration");
    assert_eq!(declared.aligned_before, 33_280);
    assert!(declared.aligned_before > METAL_MESH_OUTPUT_BYTES);
    let proof = mesh_pack_proof(&standalone.dxil, Some(&pixel_host.dxil))
        .expect("the emitted mesh container parses")
        .expect("a mesh output declaration");
    assert_eq!(proof.declared_aligned, 33_280);
    assert!(proof.fits);
    assert_eq!(proof.aligned_after, 1_280);
    assert_eq!(proof.packed_after, 1_280);

    let (linked_mesh, linked_pixel) =
        translate_linked(&fixture.mesh, Some(&fixture.pixel), Options::default())
            .expect("the linked packer translates the pair");
    let linked_pixel = linked_pixel.expect("a paired pixel stage");
    let linked = mesh_outputs(&linked_mesh.dxil).expect("linked mesh declares outputs");
    assert_eq!(linked.vertices, VERTICES);
    assert_eq!(linked.vertex_components, 5);
    assert_eq!(linked.bytes(), 1_280);
    assert!(linked.bytes() <= METAL_MESH_OUTPUT_BYTES);
    assert!(!linked_pixel.dxil.is_empty());

    // The captured converter reflection parses. The fixture reads no root
    // signature, so the real layouts are empty; the plan still records the
    // argument-buffer and table budgets the live executor enforces.
    let tables = [MeshTableExtent {
        kind: DescriptorTableKind::ShaderResourceView,
        count: 4,
    }];
    let reflected =
        MeshAbiPlan::from_reflection_json(MESH_REFLECTION, None, Some(PIXEL_REFLECTION), &tables)
            .expect("the captured reflection builds");
    assert_eq!(reflected.mesh().argument_buffer_len(), 8);
    assert_eq!(
        reflected
            .fragment()
            .expect("fragment layout")
            .argument_buffer_len(),
        8
    );
    assert_eq!(reflected.argument_buffer_bytes(), 16);
    assert_eq!(reflected.descriptor_table_bytes(), 32);

    // The explicit plan the GPU test registers: a table plus inline constants
    // and root CBV/SRV/UAV views, in one consistent order on both stages so
    // the table ordinal and parameter index agree.
    let mesh = binding_layout(ReflectedStage::Other);
    let fragment = binding_layout(ReflectedStage::Fragment);
    let plan = MeshAbiPlan::new(mesh, None, Some(fragment), &tables)
        .expect("the explicit binding plan builds");
    let constants = plan
        .mesh()
        .parameters()
        .iter()
        .filter(|parameter| parameter.kind() == RootParameterKind::Constants)
        .map(|parameter| parameter.size())
        .sum::<usize>();
    assert_eq!(constants, 16, "four inline root constant dwords");
    assert!(constants <= MAX_MESH_ROOT_CONSTANT_DWORDS * 4);
    assert_eq!(plan.mesh().parameters().len(), 5);
    assert_eq!(plan.descriptor_table_bytes(), 32);
    assert!(plan.argument_buffer_bytes() > 0);
    let abi = plan.pipeline_abi();
    assert!(abi.mesh.is_some() && abi.object.is_none() && abi.fragment.is_some());
}

/// The layout the GPU test binds: table ordinal 0, then constants, CBV, SRV,
/// UAV. Root constants are parameter 1 and stay inline.
fn binding_layout(stage: ReflectedStage) -> RootSignatureLayout {
    RootSignatureLayout::from_parameters(
        stage,
        &[
            RootParameterSpec {
                kind: RootParameterKind::DescriptorTable(DescriptorTableKind::ShaderResourceView),
                size: 0,
                register: 0,
                space: 0,
            },
            RootParameterSpec {
                kind: RootParameterKind::Constants,
                size: 16,
                register: 0,
                space: 1,
            },
            RootParameterSpec {
                kind: RootParameterKind::ConstantBufferView,
                size: 8,
                register: 0,
                space: 2,
            },
            RootParameterSpec {
                kind: RootParameterKind::ShaderResourceView,
                size: 8,
                register: 0,
                space: 3,
            },
            RootParameterSpec {
                kind: RootParameterKind::UnorderedAccessView,
                size: 8,
                register: 0,
                space: 4,
            },
        ],
    )
    .expect("the mesh binding layout builds")
}

/// The fragment stage of the live submission declares no root parameter.
///
/// The fixture's pixel shader reads none, and an empty fragment layout makes
/// the descriptor-table ordinal resolvable *only* through the mesh stages — the
/// exact ABI gap this test guards.
fn fragment_layout() -> RootSignatureLayout {
    RootSignatureLayout::from_parameters(ReflectedStage::Fragment, &[])
        .expect("an empty fragment layout builds")
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
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut c_void);
        }
        // SAFETY: `self.ptr` came from `posix_memalign` and is unregistered
        // before this guard drops.
        unsafe { free(self.ptr) };
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path =
            std::env::temp_dir().join(format!("victory-{label}-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Links the two IRConverter metallibs into one library with Apple's
/// `metal-objcopy`.
///
/// `MeshPipeline` accepts separate mesh/fragment libraries, but the
/// content-hashed registration path loads one library for both functions.
/// `metal-objcopy --output-target=irobject` extracts the raw AIR module and
/// `--add-module` appends it; a duplicate `main` is renamed (`__main_1`), which
/// is returned as the fragment entry.
fn combine_metallibs(mesh: &[u8], pixel: &[u8]) -> Result<(Vec<u8>, String), String> {
    let dir = TempDir::new("mesh-abi-combine");
    let mesh_path = dir.path().join("mesh.metallib");
    let pixel_path = dir.path().join("pixel.metallib");
    let pixel_air = dir.path().join("pixel.air");
    let combined_path = dir.path().join("combined.metallib");
    std::fs::write(&mesh_path, mesh).map_err(|error| format!("write mesh: {error}"))?;
    std::fs::write(&pixel_path, pixel).map_err(|error| format!("write pixel: {error}"))?;
    let run = |args: &[String]| -> Result<String, String> {
        let output = Command::new("xcrun")
            .args(args)
            .output()
            .map_err(|error| format!("xcrun {args:?}: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "xcrun {args:?} failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    };
    run(&[
        "metal-objcopy".to_string(),
        "--output-target=irobject".to_string(),
        pixel_path.display().to_string(),
        pixel_air.display().to_string(),
    ])?;
    run(&[
        "metal-objcopy".to_string(),
        format!("--add-module={}", pixel_air.display()),
        mesh_path.display().to_string(),
        combined_path.display().to_string(),
    ])?;
    let symbols = run(&["metal-nm".to_string(), combined_path.display().to_string()])?;
    let fragment = symbols
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .find(|name| *name != "main")
        .ok_or_else(|| format!("combined metallib has no renamed fragment: {symbols}"))?
        .to_string();
    let bytes = std::fs::read(&combined_path)
        .map_err(|error| format!("read combined metallib: {error}"))?;
    Ok((bytes, fragment))
}

/// The expected RGBA8 value of `TEXCOORD0.x` at a pixel centre.
fn expected_column(column: u32) -> u8 {
    let ndc = (column as f32 + 0.5) / 16.0 * 2.0 - 1.0;
    (ndc.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rgba8_target_desc(width: u32, height: u32) -> TextureDesc {
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
        label: Some("mesh-live-abi-colour".to_string()),
    }
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

fn float4_bytes(values: [f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Hardware proof: the packed, Roasetta-converted pair draws through the live
/// `submit_v3` mesh path with a reflected argument-buffer ABI.
///
/// The fixture's real reflection is empty (it reads no resources), so the test
/// registers the explicit binding plan from [`binding_layout`]. The pipeline is
/// registered **only** through the content-hashed mesh path — no legacy
/// `MeshPipeline` shares its handle — so the `DispatchMesh` resolution and the
/// per-stage mesh threadgroup sizes come from the `PipelineState` blob alone.
///
/// The submission also binds an SRV `SetDescriptorTable`: its converted table
/// `MTLBuffer` used to be released when the pending bindings were cleared, before
/// the submission was encoded (SIGSEGV in `objc_retain`). The table is now owned
/// by the prepared argument buffer and pinned until the fence completes, and the
/// pixel checksum proves the draw itself is still exact.
#[test]
fn live_submit_v3_mesh_draw_is_exact() {
    let fixture = Fixture::new();
    let (linked_mesh, linked_pixel) =
        translate_linked(&fixture.mesh, Some(&fixture.pixel), Options::default())
            .expect("the linked packer translates the pair");
    let linked_pixel = linked_pixel.expect("a paired pixel stage");
    let linked = mesh_outputs(&linked_mesh.dxil).expect("linked mesh declares outputs");
    assert_eq!(linked.bytes(), 1_280);

    let (worker, library) = match converter_paths() {
        Ok(paths) => paths,
        Err(reason) => {
            eprintln!("PARTIAL: {reason}");
            return;
        }
    };
    let mesh_lib = convert(&worker, &library, &linked_mesh.dxil, "mesh").expect("mesh conversion");
    let pixel_lib =
        convert(&worker, &library, &linked_pixel.dxil, "pixel").expect("pixel conversion");
    let (combined, fragment_entry) = match combine_metallibs(&mesh_lib, &pixel_lib) {
        Ok(combined) => combined,
        Err(reason) => {
            eprintln!("PARTIAL: {reason}");
            return;
        }
    };
    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let info = transport.device_info().expect("device info");

    // Content-hashed registration is the only one: a mesh PSO created from the
    // combined metallib must be dispatchable without a legacy `MeshPipeline`
    // registered under the same handle.
    const MESH_HANDLE: u64 = 0x3_000;
    let tables = [MeshTableExtent {
        kind: DescriptorTableKind::ShaderResourceView,
        count: 1,
    }];
    let plan = MeshAbiPlan::new(
        binding_layout(ReflectedStage::Other),
        None,
        Some(fragment_layout()),
        &tables,
    )
    .expect("the ABI plan builds");
    let state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some(fragment_entry),
        object_entry: None,
        // The per-stage threadgroup sizes `DispatchMesh` uses; the wire only
        // carries group counts.
        mesh_threadgroup: [THREADS, 1, 1],
        object_threadgroup: [1, 1, 1],
        ..Default::default()
    }
    .encode(PipelineStage::Mesh);
    transport
        .register_content_hashed_mesh_pipeline(
            MESH_HANDLE,
            &combined,
            "main",
            &state,
            plan.pipeline_abi(),
        )
        .expect("the content-hashed mesh pipeline should register");

    const EDGE: u32 = 16;
    const TARGET: u64 = 0x2_000;
    transport
        .register_registry_texture(TARGET, rgba8_target_desc(EDGE, EDGE))
        .expect("mesh target should register");
    let mut allocations = HashMap::new();
    allocations.insert((1u64, 1u32), VmBufferId(TARGET));

    let mut cbv = HostPage::new(16 * 1024);
    cbv.write(0, &float4_bytes([1.0, 0.25, 0.5, 1.0]));
    let mut srv = HostPage::new(16 * 1024);
    srv.write(0, &float4_bytes([2.0, 3.0, 4.0, 5.0]));
    let mut uav = HostPage::new(16 * 1024);
    uav.write(0, &float4_bytes([6.0, 7.0, 8.0, 9.0]));
    // SAFETY: each page is a live, 16 KiB-aligned allocation that outlives the
    // synchronous submission and is unregistered before it drops.
    let cbv_id = unsafe { transport.register_zerocopy(cbv.ptr.cast::<u8>(), 16 * 1024) }
        .expect("CBV page must register");
    // SAFETY: same zero-copy contract as the CBV page.
    let srv_id = unsafe { transport.register_zerocopy(srv.ptr.cast::<u8>(), 16 * 1024) }
        .expect("SRV page must register");
    // SAFETY: same zero-copy contract as the CBV page.
    let uav_id = unsafe { transport.register_zerocopy(uav.ptr.cast::<u8>(), 16 * 1024) }
        .expect("UAV page must register");
    allocations.insert((2u64, 1u32), cbv_id);
    allocations.insert((3u64, 1u32), srv_id);
    allocations.insert((4u64, 1u32), uav_id);

    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetMeshPipeline(PipelineRef {
                pipeline_id: MESH_HANDLE,
            }),
            // Parameter 0: the SRV descriptor table. Binding it exercises the
            // converted table path whose `MTLBuffer` used to be released before
            // the submission was encoded.
            CommandV3::SetDescriptorTable(DescriptorTableV3 {
                set: 0,
                base_register: 0,
                kind: DescriptorKind::Srv,
                bindings: vec![BufferBinding {
                    allocation_id: 3,
                    generation: 1,
                    offset: 0,
                    length: 16,
                }],
            }),
            // Parameter 1: inline root constants.
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 1,
                bytes: float4_bytes([0.5, 0.25, 0.125, 1.0]),
            }),
            // Parameters 2..=4: root CBV, SRV and UAV views.
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 2,
                allocation_id: 2,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 3,
                allocation_id: 3,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 4,
                allocation_id: 4,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            // NOTE: the descriptor table and root views above and the mesh
            // dispatch all live in one pass; the table buffer must stay alive
            // from encoding through completion.
            CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: MESH_HANDLE,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &allocations)
        .expect("the live mesh submission should execute");
    assert_eq!(reply.status, ReplyStatus::Ok, "mesh submission status");
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.render_encoders, 1);

    // `submit_v3` returns after `commit`; the readback below reads host-visible
    // memory, so the submission must have completed first.
    transport
        .wait_for_gpu()
        .expect("the mesh submission should complete");

    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("mesh target should read back after completion");
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

    let native = transport
        .residency_is_native()
        .expect("residency state is readable");
    let resident = transport
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
        resident >= 4,
        "the residency set must hold the target and three bound pages (got {resident})"
    );

    transport.unregister_buffer(cbv_id).expect("unregister CBV");
    transport.unregister_buffer(srv_id).expect("unregister SRV");
    transport.unregister_buffer(uav_id).expect("unregister UAV");

    eprintln!(
        "live mesh ABI on {}: packed {} B, {} px, checksum {checksum:#018x}, \
         columns {columns:?}, abi argument-buffer {} B, bound table {} B, \
         native_residency={native}, resident={resident}, fallback={fallback:?}, stats={stats:?}",
        info.name_str(),
        linked.bytes(),
        EDGE * EDGE,
        plan.argument_buffer_bytes(),
        plan.descriptor_table_bytes(),
    );
}

/// Negative proof: a mesh dispatch without a selected recognisable pipeline is
/// refused before any command buffer or pixel write.
#[test]
fn mesh_dispatch_without_a_selected_pipeline_is_refused() {
    let transport = match DirectMetalTransport::new() {
        Ok(transport) => transport,
        Err(reason) => {
            eprintln!("PARTIAL: no Metal device: {reason}");
            return;
        }
    };
    const EDGE: u32 = 4;
    const TARGET: u64 = 0x2_001;
    transport
        .register_registry_texture(TARGET, rgba8_target_desc(EDGE, EDGE))
        .expect("negative target should register");
    let allocations: HashMap<(u64, u32), VmBufferId> =
        HashMap::from([((1u64, 1u32), VmBufferId(TARGET))]);

    // A dispatch with no preceding SetMeshPipeline is a planning error.
    let missing_pipeline = SubmissionV3 {
        sequence: 2,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: 0xDEAD,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let error = transport
        .submit_v3_with_stats(&missing_pipeline, &allocations)
        .expect_err("a mesh dispatch without SetMeshPipeline must be refused");
    assert!(
        matches!(
            error,
            victory_metal::MetalError::InvalidArg(message)
                if message.contains("preceding SetMeshPipeline")
        ),
        "unexpected error: {error:?}"
    );

    // An unknown mesh handle is an UnknownHandle reply with no encoders.
    let unknown = SubmissionV3 {
        sequence: 3,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![color_attachment(1, [0.0, 0.0, 0.0, 1.0])],
                depth_stencil: None,
            }),
            CommandV3::SetMeshPipeline(PipelineRef {
                pipeline_id: 0xDEAD,
            }),
            CommandV3::DispatchMesh(wire::DispatchMesh {
                pipeline_id: 0xDEAD,
                group_counts: [1, 1, 1],
                bindings: Vec::new(),
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&unknown, &allocations)
        .expect("an unknown mesh handle must still return a reply");
    assert_eq!(reply.status, ReplyStatus::UnknownHandle);
    assert_eq!(stats, victory_metal::macos::device::EncoderStats::default());

    eprintln!(
        "negative mesh dispatch on this device: missing pipeline refused, \
         unknown handle returned UnknownHandle with no encoders"
    );
}
