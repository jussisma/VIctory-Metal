//! Lane-tagged hardware proof for the wave64 lowering (`emit::lower::halves`
//! and `emit::lower::barriers`).
//!
//! Each test builds a synthetic console compute shader with the same
//! hand-encoded GFX10 words the `victory-shader` emit tests use (a root table
//! load, a UAV store, and the console's own ballot, lane-read, reduction,
//! prefix and LDS sequences), translates it through `victory_shader::emit` to
//! DXIL, converts the DXIL with Apple's IRConverter, dispatches one group of
//! 64 threads on the Metal GPU, and asserts every lane's tagged dwords equal
//! the console semantics computed in Rust.
//!
//! IRConverter's `libmetalirconverter.dylib` is x86_64-only, so the conversion
//! runs in the separate x86_64 `victory-irconverter` worker under Rosetta 2
//! (`ProcessWorker`). Build it once:
//!
//! ```sh
//! cargo build -p victory-metal --bin victory-irconverter --target x86_64-apple-darwin
//! ```
//!
//! The hardware tests are `#[ignore]`d because the GPU is shared; run them one
//! at a time with `--ignored --test-threads=1`.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use victory_metal::macos::cache::TranslationOptions;
use victory_metal::macos::device::DirectMetalTransport;
use victory_metal::macos::irconverter::{ProcessWorker, translate_dxil_with};
use victory_metal::transport::MetalTransport;
use victory_shader::emit::{self, Options};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{dxbc, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// An empty DXBC container: `emit` reads only signature parts, and a compute
/// shader has none. The bytes are the minimum `Container::parse` accepts.
fn empty_container() -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

/// A wave64 compute program writing `per_lane` dwords per thread at `l * 32`
/// through the root table's UAV, lowered for a host of 32-lane waves.
fn lowered_dxil(words: &[u32], threads: u32, lds_bytes: u32, narrow_waves: bool) -> Vec<u8> {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let raised = io::raise(&p, &native).expect("io::raise");
    let bytes = empty_container();
    let c = dxbc::Container::parse(&bytes).expect("empty container");
    let host = emit::emit_with(
        &c,
        &native,
        &p,
        &st,
        &raised,
        &res,
        Options {
            host_wave32: true,
            narrow_waves,
            ..Options::default()
        },
    )
    .expect("the wave64 program lowers for a 32-lane host");
    assert!(
        host.dxil.windows(4).any(|w| w == b"DXIL"),
        "emitted container has no DXIL part"
    );
    host.dxil
}

/// The x86_64 Rosetta converter worker and the x86_64 IRConverter library.
fn converter() -> Result<(PathBuf, PathBuf), String> {
    let worker = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = workspace_root()?
                .join("target/x86_64-apple-darwin/debug/victory-irconverter");
            p.is_file().then_some(p)
        })
        .ok_or_else(|| {
            "no x86_64 victory-irconverter worker: build it with \
             `cargo build -p victory-metal --bin victory-irconverter \
             --target x86_64-apple-darwin` or set VICTORY_IRCONVERTER_WORKER"
                .to_string()
        })?;
    let lib = std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|r| PathBuf::from(r).join("libmetalirconverter.dylib"))
                .filter(|p| p.is_file())
        })
        .or_else(|| {
            let p = PathBuf::from(std::env::var_os("HOME")?).join(
                "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                 D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
            );
            p.is_file().then_some(p)
        })
        .ok_or_else(|| {
            "no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)"
                .to_string()
        })?;
    Ok((worker, lib))
}

fn workspace_root() -> Option<PathBuf> {
    // `CARGO_MANIFEST_DIR` is `crates/victory-metal`; the workspace is two up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
}

/// Converts the emitted DXIL with the x86_64 IRConverter worker (Rosetta 2).
fn convert(dxil: &[u8]) -> Result<Vec<u8>, String> {
    let (worker, lib) = converter()?;
    let worker = ProcessWorker::new(worker, lib);
    let out = translate_dxil_with(
        None,
        &worker,
        dxil,
        "main",
        8, // compute
        None,
        &TranslationOptions::default(),
        &[],
    )
    .map_err(|e| format!("IRConverter: {e}"))?;
    Ok(out.metallib)
}

/// A 16 KiB-aligned allocation readable and writable from both the CPU and the
/// GPU (Apple Silicon unified memory).
struct Page {
    raw: *mut std::ffi::c_void,
    len: usize,
}

impl Page {
    fn new(len: usize) -> Self {
        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut std::ffi::c_void, alignment: usize, size: usize)
                -> i32;
        }
        const ALIGN: usize = 16384;
        // `register_zerocopy` maps whole 16 KiB pages, so round the request up.
        let len = len.next_multiple_of(ALIGN);
        let mut raw = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());
        // SAFETY: the allocation covers `len` bytes and is zeroed for a clean run.
        unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, len) };
        Self { raw, len }
    }

    fn words(&mut self) -> &mut [u32] {
        // SAFETY: `raw` is a live 16 KiB-aligned allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.raw.cast::<u32>(), self.len / 4) }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut std::ffi::c_void);
        }
        // SAFETY: `raw` came from `posix_memalign` and is no longer registered.
        unsafe { free(self.raw) };
    }
}

/// Lowers, converts, dispatches `threads` threads of one group, and returns the
/// UAV words.
fn run(words: &[u32], threads: u32, lds_bytes: u32, narrow_waves: bool) -> Vec<u32> {
    let dxil = lowered_dxil(words, threads, lds_bytes, narrow_waves);
    let metallib = convert(&dxil).expect("IRConverter conversion");
    let transport = DirectMetalTransport::new().expect("Metal device");
    let pso = transport
        .load_metallib(&metallib, "main")
        .expect("load converted metallib");
    let mut page = Page::new(threads as usize * 32);
    // SAFETY: `page.raw` is a live aligned allocation that outlives the dispatch.
    let buf = unsafe { transport.register_zerocopy(page.raw.cast::<u8>(), page.len) }
        .expect("register the UAV");
    transport
        .dispatch_compute_1d(pso, &[buf], threads, 64)
        .expect("compute dispatch");
    transport.unregister_buffer(buf).expect("unregister the UAV");
    page.words()[..threads as usize * 8].to_vec()
}

/// Per-lane console semantics for the reduction fixture.
fn console_reduction(threads: u32, l: u32, f: impl Fn(u32, u32) -> u32, input: impl Fn(u32) -> u32) -> u32 {
    let wave = l & !63;
    (wave + 1..(wave + 64).min(threads)).fold(input(wave), |a, t| f(a, input(t)))
}

// ----- fixtures (the same words as `halves_tests`) -------------------------

const V_ADD_F32: u32 = 0x03;

/// The console compiler's whole-wave reduction by the VOP2 operation `op`,
/// one dword per thread at `l * 32`.
fn reduction(op: u32, ident: u32, input: u32) -> Vec<u32> {
    let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        input,       // v3 from v0
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
    ]);
    for control in [0x161, 0x162, 0x164, 0x168] {
        w.extend([
            vop2(0x0602_02FA),
            0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
        ]);
    }
    w.extend([
        0xD778_0002,
        0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
        vop2(0x0602_0302), // op v1, v2, v1
        0xD760_000E,
        0x0001_4101,       // v_readlane_b32 s14, v1, 32
        vop2(0x0602_020E), // op v1, s14, v1
        0xD760_000F,
        0x0001_0101, // v_readlane_b32 s15, v1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_020F, // v_mov_b32 v1, s15
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readlane_b32` of the thread id at console lanes `k` and `s`, two dwords
/// per thread at `l * 32`.
fn lane_reads(k: u32, s: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD760_000C,
        0x0001_0100 | (128 + k) << 9, // v_readlane_b32 s12, v0, k
        0xBE8D_03FF,
        s, // s_mov_b32 s13, s
        0xD760_000E,
        0x0000_1B00, // v_readlane_b32 s14, v0, s13
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_020E, // v_mov_b32 v2, s14
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// The lanes below 40 as data: count, high ballot dword, compaction index, at
/// `l * 32`.
fn lanes_below() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
        0xD765_0003,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v3, vcc_lo, 0
        0xD766_0003,
        0x0002_066B, // v_mbcnt_hi_u32_b32 v3, vcc_hi, v3
    ]);
    for (vdata, slot) in [(1, 0), (2, 1), (3, 2)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// The console's exclusive prefix sum of the thread id, one dword per thread.
fn exclusive_prefix() -> Vec<u32> {
    const EXEC: u32 = 126;
    let sop1 = |sdst: u32, op: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
    let vop1 = |vdst: u32, src0: u32| 0x7E00_0200 | vdst << 17 | src0;
    let add = |src0: u32| 0x4A02_0200 | src0; // v_add_nc_u32 v1, src0, v1
    let dpp = |ctrl: u32| 0xFF00_0001 | ctrl << 8; // of v1, row and bank masks 0xf
    let readlane = |lane: u32| [0xD760_0002, 0x0000_0101 | (128 + lane) << 9]; // s2, v1
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,        // v_lshlrev_b32 v16, 5, v0
        sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
    ]);
    for shift in [1, 2, 4, 8] {
        w.extend([add(0xFA), dpp(0x110 + shift)]); // v_add_nc_u32 v1, v1, v1 row_shr
    }
    // Each row's sum so far into the rows after it.
    for (last, exec_lo, exec_hi) in [
        (15, 0xFFFF_0000, 0),
        (31, 0, 0x0000_FFFF),
        (47, 0, 0xFFFF_0000),
    ] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(add(2)); // v_add_nc_u32 v1, s2, v1
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(3, 257),          // v_mov_b32 v3, v1
        vop1(3, 0xFA),
        dpp(0x111), // v_mov_b32 v3, v1 row_shr:1
    ]);
    // The first lane of each row from the last of the row before.
    for (last, exec_lo, exec_hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(vop1(3, 2)); // v_mov_b32 v3, s2
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(2, 128),          // v_mov_b32 v2, 0
        sop1(EXEC, 0x04, 194), // s_mov_b64 exec, -2
        vop1(2, 259),          // v_mov_b32 v2, v3
        sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// A one-wave group of 64 writes its `v_mbcnt` to LDS and reads the dword of
/// the lane 32 away (the other half), synced by a group barrier.
fn lds_across_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readfirstlane_b32` of `v0 >> 6` (one value per console wave), at `l * 32`.
fn first_lane_alike() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E18_0502, // v_readfirstlane_b32 s12, v2
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(1, 4));
    w.push(S_ENDPGM);
    w
}

/// The CPU lowering must emit DXIL for every fixture (no translation refusal).
#[test]
fn lowered_wave_fixtures_emit_dxil() {
    let c = empty_container();
    assert_eq!(
        dxbc::Container::parse(&c).expect("empty container").parts.len(),
        0
    );
    for (name, words, threads, lds) in [
        ("reduction", reduction(V_ADD_F32, 0x80, 0x7E06_0D00), 64, 0),
        ("lane_reads", lane_reads(40, 33), 64, 0),
        ("ballot", lanes_below(), 64, 0),
        ("prefix", exclusive_prefix(), 64, 0),
        ("lds", lds_across_halves(), 64, 256),
        ("first_lane", first_lane_alike(), 128, 0),
    ] {
        let dxil = lowered_dxil(&words, threads, lds, lds > 0);
        assert!(dxil.len() > 4, "{name}: no DXIL");
    }
}

/// Hardware proof: every fixture, converted and run on the Metal GPU, gives
/// the console's per-lane dwords.
#[test]
#[ignore = "hardware: run once on an exclusive GPU with --ignored --test-threads=1"]
fn lowered_wave_patterns_run_like_the_console_on_metal() {
    let _ = converter().expect("IRConverter worker and library");

    // Whole-wave sum of the thread ids (f32 bits).
    let threads = 64;
    let want: Vec<u32> = (0..threads)
        .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t) as f32).to_bits())
        .collect();
    let got = run(&reduction(V_ADD_F32, 0x80, 0x7E06_0D00), threads, 0, false);
    if got.iter().all(|&w| w == 0) {
        panic!(
            "the converted shader wrote nothing: `dispatch_compute_1d` binds raw buffers, but \
             the emitted shader reads its UAV descriptor from the root table \
             (`s_load_dwordx4 s[40:43], s[4:5], 0x40`), which only the D3D12 root-signature \
             host (`victory-gpu`) supplies. The verified IRConverter shape is `UAV(u0)` at \
             buffer(2) (`IR-BRIDGE-CONTRACT.md`); the emitted root table is not that shape."
        );
    }
    assert_eq!(got, want, "reduction");

    // Two lane reads of the first console wave.
    let want: Vec<u32> = (0..threads).flat_map(|l| [(l & !63) + 40, (l & !63) + 33]).collect();
    assert_eq!(run(&lane_reads(40, 33), threads, 0, false), want, "lane reads");

    // Ballot masks: count, high dword, compaction index.
    let want: Vec<u32> = (0..threads)
        .flat_map(|l| {
            let wave = l & !63;
            let below = |end: u32| 40u32.min(threads).min(end).saturating_sub(wave);
            let high = below(wave + 64).saturating_sub(32);
            [
                below(wave + 64),
                if high == 32 { u32::MAX } else { (1 << high) - 1 },
                below(l),
            ]
        })
        .collect();
    assert_eq!(run(&lanes_below(), threads, 0, false), want, "ballot");

    // Exclusive prefix sum of the thread id.
    let want: Vec<u32> = (0..threads)
        .map(|l| {
            let (wave, k) = (l & !63, l & 63);
            wave * k + k.saturating_sub(1) * k / 2
        })
        .collect();
    assert_eq!(run(&exclusive_prefix(), threads, 0, false), want, "prefix");

    // LDS after the barrier: the lane 32 away's count.
    let want: Vec<u32> = (0..threads).map(|l| (l ^ 32).min(32)).collect();
    assert_eq!(run(&lds_across_halves(), threads, 256, true), want, "lds");

    // First lane of each console wave, two dwords.
    let threads = 128;
    let want: Vec<u32> = (0..threads).flat_map(|l| [l >> 6, l & 63]).collect();
    assert_eq!(run(&first_lane_alike(), threads, 0, false), want, "first lane");
}
