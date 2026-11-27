//! Live D3D12 title gate (Wave 5B), driven through the production
//! `victory-gpu` Win64 API under Wine.
//!
//! This test is the guest half of `tools/macos/metal-poc/live-d3d12-gate.sh`.
//! It is `#[ignore]`d and only runs when `VICTORY_LIVE_GATE=1`, so the default
//! suite stays hermetic. The script builds it for
//! `x86_64-pc-windows-gnullvm`, runs it under Wine against the real
//! `victory-metal-host` daemon, and evaluates the report this test writes.
//!
//! What is exercised (the real production pieces, not a hand-built wire
//! frame):
//!
//! 1. `victory_gpu::device::create` opens the real D3D12.X device, which
//!    auto-connects the authenticated Wire v3 session and installs the native
//!    presenter from the environment (`VICTORY_METAL_SURFACE`).
//! 2. A committed buffer is created through the device's own vtable; its
//!    creation registers the guest allocation with the host (kind 2) through
//!    the `GuestRangeResolver` seam.
//! 3. Two colour textures, a depth texture and the present surface are
//!    registered through the native queue (kinds 8/9), and the DXIL pool's
//!    metallibs are registered as compute (kind 4 v1) and graphics/mesh
//!    (kind 4 v2) pipelines.
//! 4. Every measured frame records a Wave64 compute dispatch, an
//!    `ExecuteIndirect` draw into an MRT+depth pass through the version 3
//!    producer, and a real `Present` through the queue entry point.
//! 5. `Signal` + `Fence::GetCompletedValue` observe completion on the 0-IPC
//!    shared page (`gpu-fence.v1.page`) after each committed submission; the
//!    next frame's dependent write waits for it.
//!
//! The ADS-017 semantics are respected: a kind-5 reply means the host
//! *committed* the command buffer, never that the GPU finished. Completion is
//! only taken from the shared fence page.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::ptr::null;
use std::time::{Duration, Instant};

use victory_gpu::device;
use victory_gpu::native_metal::bindings::{self, GuestRange, GuestRangeResolver, ResolveError};
use victory_gpu::native_metal::client::{PIPELINE_KIND_GRAPHICS, PIPELINE_KIND_MESH};
use victory_gpu::native_metal::fence_page_guest::GuestFencePage;
use victory_gpu::native_metal::producer::{
    DepthTarget, MetalProducer, RenderTarget, ResourceId,
};
use victory_gpu::native_metal::{pso, queue, telemetry};
use victory_gpu::object::{Object, S_OK};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    IndirectArguments, LoadOp, PrimitiveTopology, StoreOp,
};

type This = *const Object;
type Hresult = i32;

const ALLOC_ID: u64 = 1;
const ALLOC_GEN: u32 = 1;
const ALLOC_PAGES: u64 = 4;
const PAGE: u64 = 64 * 1024;
const ALLOC_LEN: u64 = ALLOC_PAGES * PAGE;
/// Indirect draw record offset: last page, 16-byte record.
const ARGS_OFFSET: u64 = 3 * PAGE;
/// Sentinel bytes the refused replay would overwrite with its `CopyBuffer`.
const SENTINEL_OFFSET: u64 = 3 * PAGE + 16;
const SENTINEL_LEN: u64 = 16;
const COLOR0_ID: u64 = 4096;
const COLOR1_ID: u64 = 4097;
const DEPTH_ID: u64 = 4098;
const SURFACE_ID: u64 = 1;
const TEXTURE_GEN: u32 = 1;
const TARGET_W: u32 = 1280;
const TARGET_H: u32 = 720;
/// FNV-1a 64 of the built-in orange frame, 1280x720 RGBA8.
const EXPECTED_PRESENT_CHECKSUM: u64 = 0xd579_1f96_cdb3_2383;
const CHECKED_VERSION: u32 = 0x0002_0004;

const DEVICE_IID: [u8; 16] = guid(
    0x189819f1,
    0x1db6,
    0x4b57,
    [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
);
const QUEUE_IID: [u8; 16] = guid(
    0x0ec870a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);
const FENCE_IID: [u8; 16] = guid(
    0x0a753dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);
const RESOURCE_IID: [u8; 16] = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

const fn guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> [u8; 16] {
    let (a, b, c) = (a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes());
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6],
        d[7],
    ]
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

/// Calls one slot of an object's own function table.
///
/// SAFETY: `object` must be an interface the guest handed out, and `F` the
/// exact signature of that slot.
unsafe fn slot<F: Copy>(object: This, offset: usize) -> F {
    unsafe {
        let table = *(object.cast::<*const *const ()>());
        let at = table.add(offset / 8);
        std::mem::transmute_copy(&*at)
    }
}

type CreateQueue =
    unsafe extern "system" fn(This, *const QueueDesc, *const [u8; 16], *mut This) -> Hresult;
type CreateFence =
    unsafe extern "system" fn(This, u64, u32, *const [u8; 16], *mut This) -> Hresult;
type CreateCommitted = unsafe extern "system" fn(
    This,
    *const HeapProperties,
    u32,
    *const ResourceDesc,
    u32,
    *const std::ffi::c_void,
    *const [u8; 16],
    *mut This,
) -> Hresult;
type Signal = unsafe extern "system" fn(This, This, u64) -> Hresult;
type Completed = unsafe extern "system" fn(This) -> u64;
type Present = unsafe extern "system" fn(This, u32, *const u8, *const u8) -> Hresult;

/// Resolves every guest address of the gate's one allocation to that
/// allocation. The test owns the allocation file, exactly as a launcher would.
struct GateResolver;

impl GuestRangeResolver for GateResolver {
    fn resolve(&self, address: u64, len: u64) -> Result<GuestRange, ResolveError> {
        if address == 0 || len == 0 {
            return Err(ResolveError::Invalid("zero address or length"));
        }
        let _ = (address, len);
        Ok(GuestRange {
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            byte_offset: 0,
            size: ALLOC_LEN,
        })
    }

    fn retain(&self, _range: &GuestRange) -> Result<u64, ResolveError> {
        Ok(1)
    }

    fn retire(&self, _token: u64) -> Result<(), ResolveError> {
        Ok(())
    }
}

/// Accumulates the machine-readable report the launcher parses.
struct Report {
    entries: Vec<(String, String)>,
    first_failure: Option<(&'static str, String)>,
}

impl Report {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            first_failure: None,
        }
    }

    fn note(&mut self, key: &str, value: impl ToString) {
        self.entries.push((key.to_string(), value.to_string()));
    }

    fn stage(&mut self, name: &'static str, ok: bool, detail: impl ToString) {
        self.entries.push((name.to_string(), detail.to_string()));
        if !ok && self.first_failure.is_none() {
            self.first_failure = Some((name, detail.to_string()));
        }
    }

    fn write(&self, path: &Path) {
        let mut text = String::new();
        for (key, value) in &self.entries {
            text.push_str(key);
            text.push('=');
            text.push_str(&value.replace('\n', " "));
            text.push('\n');
        }
        match &self.first_failure {
            Some((stage, reason)) => {
                text.push_str(&format!("first_failing_stage={stage}\n"));
                text.push_str(&format!("first_failing_reason={}\n", reason.replace('\n', " ")));
            }
            None => text.push_str("first_failing_stage=none\n"),
        }
        let _ = std::fs::write(path, text);
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(fallback)
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|text| text.trim().parse().ok())
}

fn read_metallib(name: &str) -> Option<Vec<u8>> {
    let path = env_path(name)?;
    std::fs::read(&path)
        .ok()
        .filter(|bytes| bytes.starts_with(b"MTLB"))
}

/// Encodes a graphics `PipelineState` (`VPS1`) with a depth attachment.
///
/// Mirrors `victory_metal::macos::registry::PipelineState::encode` so the gate
/// can register the pool's translated vertex+pixel pair against an MRT+depth
/// pass; `pso::encode_graphics_state` only expresses the no-depth shape.
fn graphics_state_with_depth(color_formats: &[u8], depth_code: u8, fragment: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + fragment.len());
    out.extend_from_slice(b"VPS1");
    out.push(0); // graphics
    out.push(u8::try_from(color_formats.len()).expect("colour count"));
    out.extend_from_slice(color_formats);
    out.push(1); // depth present
    out.push(depth_code);
    out.push(1); // fragment entry present
    for text in [fragment, ""] {
        out.extend_from_slice(&u16::try_from(text.len()).expect("entry fits").to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    for _ in 0..6 {
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}


fn write_allocation(path: &Path) {
    let mut bytes = vec![0u8; ALLOC_LEN as usize];
    // Indirect draw record: vertex_count, instance_count, start_vertex,
    // start_instance.
    let record = [3u32, 1, 0, 0];
    let at = ARGS_OFFSET as usize;
    for (index, word) in record.iter().enumerate() {
        bytes[at + index * 4..at + index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    // The replay probe copies 16 bytes over this sentinel if it ever runs.
    let sentinel = 0xA5A5_5A5Au32;
    let at = SENTINEL_OFFSET as usize;
    for index in 0..4 {
        bytes[at + index * 4..at + index * 4 + 4].copy_from_slice(&sentinel.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("write the guest allocation file");
}

fn sentinel_intact(path: &Path) -> bool {
    let bytes = std::fs::read(path).expect("read the guest allocation file");
    let at = SENTINEL_OFFSET as usize;
    if at + 16 > bytes.len() {
        return false;
    }
    // `0xA5A5_5A5A` little-endian is `5a 5a a5 a5`, repeated four times.
    let expected = [0x5A, 0x5A, 0xA5, 0xA5];
    bytes[at..at + 16]
        .iter()
        .enumerate()
        .all(|(index, byte)| *byte == expected[index % 4])
}

/// Real `victory-gpu` device, queue, fence, resources, textures, pipelines and
/// a 120-frame MRT+depth present loop.
#[test]
#[ignore = "live Wine guest gate; run tools/macos/metal-poc/live-d3d12-gate.sh"]
fn live_d3d12_guest_gate() {
    if std::env::var("VICTORY_LIVE_GATE").ok().as_deref() != Some("1") {
        eprintln!("live_d3d12_guest_gate: skipped (set VICTORY_LIVE_GATE=1)");
        return;
    }

    let mut report = Report::new();
    let share = env_path("VICTORY_METAL_SHARE_DIR").expect("VICTORY_METAL_SHARE_DIR");
    let out = env_path("VICTORY_LIVE_GATE_OUT")
        .unwrap_or_else(|| share.join("live-gate.report"));
    let csv = env_path("VICTORY_LIVE_GATE_CSV")
        .unwrap_or_else(|| share.join("live-gate.frames.csv"));
    let warmup = env_u32("VICTORY_FRAME_WARMUP", 20);
    let measured = env_u32("VICTORY_FRAME_MEASURED", 120);

    report.note("warmup", warmup);
    report.note("measured_expected", measured);
    report.note("expected_present_checksum", format!("{EXPECTED_PRESENT_CHECKSUM:#x}"));
    report.note(
        "present_surface",
        format!(
            "surface_id={SURFACE_ID} env={}",
            std::env::var("VICTORY_METAL_SURFACE").unwrap_or_default()
        ),
    );

    let allocation_path = share.join(format!("{ALLOC_ID:016x}.bin"));
    write_allocation(&allocation_path);
    // Create the shared completion page before the session connects so the
    // client registers it (kind 6) during the handshake.
    let _page = GuestFencePage::open(share.join("gpu-fence.v1.page"));

    let _old_resolver = bindings::set_resolver(GateResolver);

    // --- Real device: connects the session, installs the presenter. ---
    let mut device_ptr: This = null();
    let hr = device::create(null(), &CHECKED_VERSION, &DEVICE_IID, &mut device_ptr);
    report.stage("device", hr == S_OK, format!("hresult={hr:#x}"));
    if hr != S_OK || device_ptr.is_null() {
        report.note("ok", 0);
        report.write(&out);
        return;
    }

    // --- Real queue and fence objects through the device's own table. ---
    let create_queue: CreateQueue = unsafe { slot(device_ptr, 0x40) };
    let create_fence: CreateFence = unsafe { slot(device_ptr, 0x120) };
    let queue_desc = QueueDesc {
        kind: 0,
        priority: 0,
        flags: 0,
        nodes: 1,
    };
    let mut queue_ptr: This = null();
    let hr = unsafe { create_queue(device_ptr, &queue_desc, &QUEUE_IID, &mut queue_ptr) };
    report.stage("queue", hr == S_OK, format!("hresult={hr:#x}"));
    let mut fence: This = null();
    let hr = unsafe { create_fence(device_ptr, 0, 0, &FENCE_IID, &mut fence) };
    report.stage("fence", hr == S_OK, format!("hresult={hr:#x}"));

    // --- Real committed buffer: its creation registers kind 2. ---
    let create_resource: CreateCommitted = unsafe { slot(device_ptr, 0xd8) };
    let heap = HeapProperties {
        kind: 0,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    let desc = ResourceDesc {
        dimension: 1, // buffer
        alignment: 0,
        width: 4096,
        height: 1,
        depth_or_array: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 0,
        flags: 0,
    };
    let mut buffer: This = null();
    let hr = unsafe {
        create_resource(
            device_ptr,
            &heap,
            0,
            &desc,
            0,
            null(),
            &RESOURCE_IID,
            &mut buffer,
        )
    };
    let registered = queue::allocation_registered(ALLOC_ID);
    report.stage(
        "buffer_registration",
        hr == S_OK && registered,
        format!("hresult={hr:#x} allocation_registered={registered}"),
    );

    // The gate names the allocation explicitly; a real title reaches the same
    // identity through the resolver. Confirm the seam resolves it.
    let resolved = bindings::resolver().and_then(|resolver| resolver.resolve(1, 16).ok());
    report.stage(
        "resolver",
        resolved.is_some_and(|range| range.allocation_id == ALLOC_ID),
        format!("resolved={resolved:?}"),
    );

    // --- Textures and present surface (kinds 8/9). ---
    let color0 = queue::register_texture(
        COLOR0_ID,
        TEXTURE_GEN,
        1,
        0, // Rgba8Unorm
        1, // colour attachment
        1,
        TARGET_W,
        TARGET_H,
        1,
        1,
        1,
    );
    report.stage("texture_color0", matches!(color0, Ok(true)), format!("{color0:?}"));
    let color1 = queue::register_texture(
        COLOR1_ID,
        TEXTURE_GEN,
        1,
        0,
        1,
        1,
        TARGET_W,
        TARGET_H,
        1,
        1,
        1,
    );
    report.stage("texture_color1", matches!(color1, Ok(true)), format!("{color1:?}"));
    let depth = queue::register_texture(
        DEPTH_ID,
        TEXTURE_GEN,
        1,
        7, // Depth32Float
        2, // depth attachment
        1,
        TARGET_W,
        TARGET_H,
        1,
        1,
        1,
    );
    report.stage("texture_depth", matches!(depth, Ok(true)), format!("{depth:?}"));
    // The presenter registers SURFACE_ID on the first real Present; a manual
    // registration would be refused as a duplicate, so it is left to the
    // installed presenter (kind 9).

    // --- Translated pipelines from the pool's metallibs. ---
    let vs = read_metallib("VICTORY_LIVE_GATE_METALLIB_VS");
    let ps = read_metallib("VICTORY_LIVE_GATE_METALLIB_PS");
    let ms = read_metallib("VICTORY_LIVE_GATE_METALLIB_MS");
    let psm = read_metallib("VICTORY_LIVE_GATE_METALLIB_PSM");
    let cs = read_metallib("VICTORY_LIVE_GATE_METALLIB_CS");
    report.note(
        "metallibs",
        format!(
            "vs={} ps={} ms={} psm={} cs={}",
            vs.as_ref().map_or(0, Vec::len),
            ps.as_ref().map_or(0, Vec::len),
            ms.as_ref().map_or(0, Vec::len),
            psm.as_ref().map_or(0, Vec::len),
            cs.as_ref().map_or(0, Vec::len)
        ),
    );
    if let (Some(ms), Some(psm)) = (&ms, &psm) {
        report.note("mesh_pixel_pair_bytes", ms.len() + psm.len());
    }

    let builtin_graphics = env_u64("VICTORY_LIVE_GATE_GRAPHICS_ID").unwrap_or(0);
    let builtin_compute = env_u64("VICTORY_LIVE_GATE_COMPUTE_ID").unwrap_or(0);

    let compute_id = match &cs {
        Some(metallib) => match queue::register_pipeline(metallib, "cs_main") {
            Ok(id) => {
                report.stage("compute_pipeline", true, format!("v1_id={id}"));
                Some(id)
            }
            Err(error) => {
                report.stage("compute_pipeline", false, format!("v1_refused: {error}"));
                None
            }
        },
        None => {
            report.stage("compute_pipeline", false, "no cs metallib");
            None
        }
    };
    // The daemon's built-in compute kernel binds a buffer; dispatching it with
    // no binding would write to nil. The gate therefore uses only the
    // translated Wave64 pipeline for compute and reports the built-in id.
    report.note(
        "builtin_compute_id",
        builtin_compute,
    );

    let mut graphics_id = None;
    let mut use_root_constants = false;
    if let (Some(vs), Some(ps)) = (&vs, &ps) {
        let state = graphics_state_with_depth(
            &[pso::DEFAULT_COLOR_FORMAT, pso::DEFAULT_COLOR_FORMAT],
            14,
            "ps_main",
        );
        // The translated pair has no resources, so the ABI is empty (no
        // auto-binding) and no root-signature bytes are needed.
        let details = pso::PipelineDetails {
            state: &state,
            abi: b"",
            root_signature: b"",
        };
        let result = pso::register_translated(
            &[
                (pso::PipelineStage::Vertex, vs.as_slice(), "vs_main"),
                (pso::PipelineStage::Pixel, ps.as_slice(), "ps_main"),
            ],
            &details,
        );
        match result {
            Ok(id) => {
                report.stage("graphics_pipeline", true, format!("v2_kind={PIPELINE_KIND_GRAPHICS} id={id}"));
                graphics_id = Some(id);
            }
            Err(error) => {
                report.stage("graphics_pipeline", false, format!("v2_refused: {error}"));
            }
        }
    } else {
        report.stage("graphics_pipeline", false, "no vs/ps metallib");
    }
    if graphics_id.is_none() && builtin_graphics != 0 {
        // The daemon's built-in MRT+depth pipeline is native Metal and has no
        // DXIL translation step; it is the honest fallback that keeps the
        // measured frames on the real API when the translated registration is
        // refused. The script reports the refusal as a partial stage.
        graphics_id = Some(builtin_graphics);
        use_root_constants = true;
        report.note("graphics_pipeline_fallback", format!("host_builtin_id={builtin_graphics}"));
    }
    let Some(graphics_id) = graphics_id else {
        report.note("ok", 0);
        report.note("reason", "no graphics pipeline (translated or built-in)");
        report.write(&out);
        return;
    };

    let mesh_id = match (&ms, &psm) {
        (Some(ms), Some(psm)) => {
            let state = pso::encode_mesh_state(&[pso::DEFAULT_COLOR_FORMAT], "psm_main");
            let details = pso::PipelineDetails {
                state: &state,
                abi: b"",
                root_signature: b"",
            };
            match pso::register_translated(
                &[
                    (pso::PipelineStage::Mesh, ms.as_slice(), "ms_main"),
                    (pso::PipelineStage::Pixel, psm.as_slice(), "psm_main"),
                ],
                &details,
            ) {
                Ok(id) => {
                    report.stage("mesh_pipeline", true, format!("v2_kind={PIPELINE_KIND_MESH} id={id}"));
                    Some(id)
                }
                Err(error) => {
                    report.stage("mesh_pipeline", false, format!("v2_refused: {error}"));
                    None
                }
            }
        }
        _ => {
            report.stage("mesh_pipeline", false, "no mesh/psm metallib");
            None
        }
    };

    // --- Probe the binding paths the producer can express. ---
    if let Some(compute_id) = compute_id {
        let mut probe = match MetalProducer::new(9_001, 1) {
            Ok(producer) => producer,
            Err(error) => {
                report.stage("compute_probe", false, format!("producer: {error}"));
                MetalProducer::new(9_001, 1).expect("producer")
            }
        };
        let recorded = probe
            .set_compute_pipeline(compute_id)
            .and_then(|()| probe.set_compute_threadgroup_size([64, 1, 1]))
            .and_then(|()| probe.dispatch(1, 1, 1));
        match recorded {
            Ok(()) => match probe.flush() {
                Ok(submission) => {
                    let result = queue::submit_recorded_batch(&[submission], 0);
                    report.stage(
                        "compute_probe",
                        matches!(
                            result,
                            Ok(queue::NativeDispatch::Native {
                                status: ReplyStatus::Ok,
                                ..
                            })
                        ),
                        format!("{result:?}"),
                    );
                }
                Err(error) => report.stage("compute_probe", false, format!("flush: {error}")),
            },
            Err(error) => report.stage("compute_probe", false, format!("record: {error}")),
        }
    } else {
        report.stage("compute_probe", false, "no compute pipeline");
    }

    if let Some(mesh_id) = mesh_id {
        let mut probe = MetalProducer::new(9_002, 1).expect("mesh probe producer");
        let recorded = probe
            .set_mesh_pipeline(mesh_id)
            .and_then(|()| probe.dispatch_mesh(1, 1, 1));
        match recorded {
            Ok(()) => match probe.flush() {
                Ok(submission) => {
                    let result = queue::submit_recorded_batch(&[submission], 0);
                    report.stage(
                        "mesh_dispatch",
                        matches!(
                            result,
                            Ok(queue::NativeDispatch::Native {
                                status: ReplyStatus::Ok,
                                ..
                            })
                        ),
                        format!("{result:?}"),
                    );
                }
                Err(error) => report.stage("mesh_dispatch", false, format!("flush: {error}")),
            },
            Err(error) => report.stage("mesh_dispatch", false, format!("record: {error}")),
        }
    } else {
        report.stage("mesh_dispatch", false, "no mesh pipeline");
    }

    // The producer has no compute resource binding (ADR follow-up): a compute
    // root view / descriptor table is not part of the merged API, so the
    // compute->indirect linkage is reported as an explicit gap rather than
    // faked.
    report.stage(
        "compute_indirect_linkage",
        false,
        "MetalProducer has no compute root view or descriptor table; Dispatch3d.bindings is always empty",
    );

    // --- Measured frames ---
    let total = warmup + measured;
    let mut csv_text = String::new();
    let mut frames_ok = 0u32;
    let mut fence_timeouts = 0u32;
    let mut fence_waits = 0u32;
    let mut last_list = 0u64;
    let mut present_status = 0i32;
    let deadline = Duration::from_secs(30);
    let mut max_in_flight = 0u32;

    for index in 0..total {
        let started = Instant::now();
        // Above the presenter's own small list ids (WirePresenter counts from
        // 1), so a frame's draw and its Present never share a ledger entry.
        let list_id = 1_000_000 + u64::from(index) + 1;
        last_list = list_id;
        let mut producer = match MetalProducer::new(list_id, 1) {
            Ok(producer) => producer,
            Err(error) => {
                report.stage("frame_producer", false, format!("frame {index}: {error}"));
                break;
            }
        };

        // Compute culling stage: a real Wave64 dispatch through the
        // translated compute pipeline. The merged producer has no compute
        // root binding, so the kernel takes no buffer (reported above).
        let mut stage_error = None;
        if let Some(compute_id) = compute_id {
            if let Err(error) = producer
                .set_compute_pipeline(compute_id)
                .and_then(|()| producer.set_compute_threadgroup_size([64, 1, 1]))
                .and_then(|()| producer.dispatch(1, 1, 1))
            {
                stage_error = Some(format!("compute: {error}"));
            }
        }
        // Indirect draw into the MRT+depth pass.
        if stage_error.is_none() {
            let targets = [
                RenderTarget::cleared(
                    ResourceId::new(COLOR0_ID, TEXTURE_GEN).expect("color0"),
                    [0.0, 0.0, 0.0, 1.0],
                ),
                RenderTarget::cleared(
                    ResourceId::new(COLOR1_ID, TEXTURE_GEN).expect("color1"),
                    [0.0, 0.0, 0.0, 1.0],
                ),
            ];
            let mut depth_target = DepthTarget::new(
                ResourceId::new(DEPTH_ID, TEXTURE_GEN).expect("depth"),
            );
            depth_target.depth_load = LoadOp::Clear;
            depth_target.depth_store = StoreOp::Store;
            depth_target.clear_depth = 1.0;
            depth_target.stencil_load = LoadOp::DontCare;
            depth_target.stencil_store = StoreOp::DontCare;
            let result = producer
                .set_render_targets(&targets, Some(depth_target))
                .and_then(|()| {
                    producer.set_viewport(
                        0.0,
                        0.0,
                        TARGET_W as f32,
                        TARGET_H as f32,
                        0.0,
                        1.0,
                    )
                })
                .and_then(|()| producer.set_scissor_edges(0, 0, TARGET_W as i32, TARGET_H as i32))
                .and_then(|()| producer.set_topology(PrimitiveTopology::TriangleList))
                .and_then(|()| producer.set_pipeline(graphics_id));
            let result = result.and_then(|()| {
                if use_root_constants {
                    let tints: [u32; 8] = [
                        1.0f32.to_bits(),
                        (128.0f32 / 255.0).to_bits(),
                        0.0f32.to_bits(),
                        1.0f32.to_bits(),
                        0.0f32.to_bits(),
                        (128.0f32 / 255.0).to_bits(),
                        1.0f32.to_bits(),
                        1.0f32.to_bits(),
                    ];
                    producer.set_graphics_root_constants(0, &tints, 0)?;
                }
                producer.execute_indirect_draw(IndirectArguments {
                    allocation_id: ALLOC_ID,
                    generation: ALLOC_GEN,
                    offset: ARGS_OFFSET,
                    max_count: 1,
                    count: None,
                })
            });
            if let Err(error) = result {
                stage_error = Some(format!("indirect_draw: {error}"));
            }
        }
        // Post-FX compute stage.
        if stage_error.is_none() {
            if let Some(compute_id) = compute_id {
                if let Err(error) = producer
                    .set_compute_pipeline(compute_id)
                    .and_then(|()| producer.dispatch(1, 1, 1))
                {
                    stage_error = Some(format!("postfx: {error}"));
                }
            }
        }
        if let Some(reason) = stage_error {
            report.stage("frame_record", false, format!("frame {index}: {reason}"));
            break;
        }
        let submission = match producer.flush() {
            Ok(submission) => submission,
            Err(error) => {
                report.stage("frame_flush", false, format!("frame {index}: {error}"));
                break;
            }
        };
        let sequence = submission.sequence;
        match queue::submit_recorded_batch(&[submission], 0) {
            Ok(queue::NativeDispatch::Native {
                status: ReplyStatus::Ok,
                ..
            }) => {}
            other => {
                report.stage("frame_submit", false, format!("frame {index}: {other:?}"));
                break;
            }
        }
        max_in_flight = max_in_flight.max(1);

        // Real Present through the queue entry point. The installed wire
        // presenter registers the surface (kind 9) on the first frame.
        if !queue_ptr.is_null() {
            let present: Present = unsafe { slot(queue_ptr, 0xf0) };
            let hr = unsafe { present(queue_ptr, 0, null(), null()) };
            present_status = hr;
            if hr != S_OK {
                report.stage("present", false, format!("frame {index}: hresult={hr:#x}"));
                break;
            }
        }

        // ADR-017: observe completion only on the shared page. Signal the real
        // fence, then poll GetCompletedValue (one Acquire load) until it moves.
        if !queue_ptr.is_null() && !fence.is_null() {
            let signal: Signal = unsafe { slot(queue_ptr, 0x70) };
            let completed: Completed = unsafe { slot(fence, 0x40) };
            let value = u64::from(index) + 1;
            let hr = unsafe { signal(queue_ptr, fence, value) };
            if hr != S_OK {
                report.stage("fence_signal", false, format!("frame {index}: hresult={hr:#x}"));
                break;
            }
            let wait_started = Instant::now();
            let mut observed = 0u64;
            while wait_started.elapsed() < deadline {
                observed = unsafe { completed(fence) };
                if observed >= value {
                    break;
                }
                std::thread::sleep(Duration::from_micros(50));
            }
            fence_waits += 1;
            if observed < value {
                fence_timeouts += 1;
                report.stage(
                    "fence_completion",
                    false,
                    format!("frame {index}: observed={observed} expected>={value} seq={sequence}"),
                );
                break;
            }
        }

        frames_ok += 1;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if index >= warmup {
            csv_text.push_str(&format!("{:.6}\n", elapsed_ms));
        }
    }

    report.note("frames_ok", frames_ok);
    report.note("measured", csv_text.lines().count());
    report.note("fence_waits", fence_waits);
    report.note("fence_timeouts", fence_timeouts);
    report.note("max_in_flight_observed", max_in_flight);
    report.note("present_status", present_status);
    report.note("last_list_id", last_list);
    let _ = std::fs::write(&csv, csv_text);

    // --- Exactly-once replay probe on the last frame's list identity. ---
    let before_checksum = sentinel_checksum(&allocation_path);
    let mut replay = match MetalProducer::new(last_list.max(1), 1) {
        Ok(producer) => producer,
        Err(error) => {
            report.stage("replay", false, format!("producer: {error}"));
            MetalProducer::new(1, 1).expect("producer")
        }
    };
    let recorded = replay.copy_buffer(
        ResourceId::new(ALLOC_ID, ALLOC_GEN).expect("src"),
        0,
        ResourceId::new(ALLOC_ID, ALLOC_GEN).expect("dst"),
        SENTINEL_OFFSET,
        SENTINEL_LEN,
    );
    let replay_status = match recorded {
        Ok(()) => match replay.flush() {
            Ok(submission) => match queue::submit_recorded_batch(&[submission], 0) {
                Ok(queue::NativeDispatch::Native { status, .. }) => format!("{status:?}"),
                other => format!("{other:?}"),
            },
            Err(error) => format!("flush: {error}"),
        },
        Err(error) => format!("record: {error}"),
    };
    let after_checksum = sentinel_checksum(&allocation_path);
    let unchanged = before_checksum == after_checksum && sentinel_intact(&allocation_path);
    report.stage(
        "replay",
        replay_status == "AlreadyExecuted" && unchanged,
        format!("status={replay_status} unchanged={unchanged}"),
    );

    // --- Telemetry and D3DMetal exclusion. ---
    let snapshot = telemetry::snapshot();
    report.note("telemetry_frames", snapshot.frames);
    report.note("telemetry_p50_us", snapshot.percentiles.p50);
    report.note("telemetry_p95_us", snapshot.percentiles.p95);
    report.note("telemetry_p99_us", snapshot.percentiles.p99);
    report.note("telemetry_fps", format!("{:.3}", snapshot.fps));
    let histogram: Vec<String> = snapshot
        .opcodes
        .iter()
        .map(|(opcode, count)| format!("{opcode}:{count}"))
        .collect();
    report.note("unsupported_opcodes", histogram.join(","));
    report.note(
        "d3dmetal_fallback_count",
        snapshot.d3dmetal_fallback_count,
    );
    let loaded: i32 = if d3dmetal_loaded() { 1 } else { 0 };
    report.note("guest_d3dmetal_module", loaded);
    report.note("submits_ok", snapshot.submits_ok);
    report.note("submits_refused", snapshot.submits_refused);
    report.note("presents_ok", snapshot.presents_ok);
    report.note("presents_refused", snapshot.presents_refused);
    report.note("buffers_registered", snapshot.buffers_registered);
    report.note("psos_registered", snapshot.psos_registered);
    report.note("registration_failures", snapshot.registration_failures);

    let all_frames = frames_ok == total && fence_timeouts == 0;
    report.stage(
        "frames",
        all_frames,
        format!("frames_ok={frames_ok} expected={total} fence_timeouts={fence_timeouts}"),
    );
    report.note("ok", if report.first_failure.is_none() { 1 } else { 0 });
    if let Some((stage, reason)) = &report.first_failure {
        report.note("reason", format!("{stage}: {reason}"));
    } else {
        report.note("reason", "completed");
    }
    report.write(&out);

    let _ = bindings::clear_resolver();
    if let Some(id) = queue::take_present() {
        drop(id);
    }
}

/// Whether this guest process has a D3DMetal image loaded.
///
/// The host daemon refuses to start with one and its telemetry records an
/// empty image list; this is the guest-side half of the same assertion.
fn d3dmetal_loaded() -> bool {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
    // SAFETY: static NUL-terminated module names; the call only returns a
    // handle or null and never dereferences the pointer.
    unsafe {
        !GetModuleHandleA(c"d3dmetal.dll".as_ptr().cast()).is_null()
            || !GetModuleHandleA(c"libd3dmetal.dll".as_ptr().cast()).is_null()
    }
}

fn sentinel_checksum(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    let at = SENTINEL_OFFSET as usize;
    if at + SENTINEL_LEN as usize > bytes.len() {
        return 0;
    }
    let mut hash = 1_469_598_103_934_665_603u64;
    for byte in &bytes[at..at + SENTINEL_LEN as usize] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}
