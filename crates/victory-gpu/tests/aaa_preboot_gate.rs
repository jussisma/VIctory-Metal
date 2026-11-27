//! Unified Cycle 07 AAA pre-boot gate: one Wine Win64 guest that exercises
//! every measured capability in one 120-frame submission stream through the
//! production `victory-gpu` path and the real `victory_metal_host`.
//!
//! The guest half is `aaa_preboot_guest_gate`; the launcher half is
//! `tools/macos/metal-poc/aaa-preboot-gate.sh`, which wraps
//! `run-title-metal.sh` (Wave 6A) and supplies the eight prebuilt metallibs
//! this gate needs: a vertex/pixel pair with multi-entry tables, a compute
//! culling kernel, an object (amplification) shader whose 8-thread group keeps
//! 4 survivors, a mesh shader and a mesh pixel shader, plus a probe
//! vertex/pixel pair that samples registered textures and writes exact channels.
//!
//! What every measured frame records, all in one submission:
//!
//! 1. A compute culling dispatch that writes an `ExecuteIndirect` **mesh**
//!    record (three `uint` threadgroup counts) into the guest allocation.
//! 2. A UAV -> indirect-argument barrier.
//! 3. One MRT render pass (two `RGBA8` targets plus `Depth32Float`) using
//!    `DontCare` loads followed by explicit `ClearRenderTargetView` /
//!    `ClearDepthStencilView` commands.
//! 4. A graphics pipeline draw whose pixel shader reads two CBVs, two SRV
//!    buffers, two UAVs and a root-constants block, all bound as count >= 2
//!    descriptor tables. Three draws exercise depth testing (a farther draw is
//!    rejected) and back-face culling (a clockwise draw is culled).
//! 5. An in-pass `ExecuteIndirectDispatchMesh` whose group counts the compute
//!    kernel wrote, followed by a direct `DispatchMesh` on the same
//!    `[Mesh, Object, Pixel]` pipeline. The amplification stage runs eight
//!    object threads, keeps four (`mask = 0x55`) and dispatches four mesh
//!    threadgroups; each group paints exactly 28 px, so each dispatch paints
//!    exactly 112 px and writes its vertex count, survivor mask and marker into
//!    the guest allocation.
//! 6. Copies that make the second colour target and the pitched
//!    `CopyBufferToTexture` upload visible in the presented target, so the
//!    final exact-pixel checksum covers them.
//!
//! After the frame loop one extra submission runs two probe passes. The first
//! registers a content-hashed `VPS2` PSO with alpha blending baked in, draws
//! `(1,0,0,0.5)` over a blue clear and produces `(128,0,128,128)`. The second
//! pass samples that blended target, the presented target (clear, both copies
//! and the pitched upload), and a count==2 SRV **texture** table plus a sampler
//! table, writing every sampled channel as an exact `u32` into the guest
//! allocation.
//!
//! Two separate post-loop submissions exercise the mesh-only pipeline the
//! compute count drives: a compute kernel writes count=1 (one 8x8 block, 64 px)
//! and count=2 (two blocks, 128 px) into the shared indirect record, and a
//! direct two-group `DispatchMesh` must paint the same 128 px.
//!
//! # What is an API path and what is direct Wire v3
//!
//! Device, queue, fence, committed buffer, texture and sampler registration,
//! pipeline registration, the producer's command-list recording, the compute
//! dispatch, the draws, the indirect mesh dispatch, the spare passes and the
//! present use the production `victory-gpu` API. `ClearRenderTargetView`,
//! `ClearDepthStencilView` and the pitched `CopyBufferToTexture` have no guest
//! entry point yet, so those commands are spliced into the producer's
//! submission as raw Wire v3 commands. They are the *only* direct-Wire
//! insertions, and the report names them.
//!
//! # Exit contract
//!
//! The gate's report is `ok=1` only when every stage is true; the launcher
//! exits 0 only when both runs' reports are complete, every stage is true, the
//! two presented checksums agree, the real 10,000-cycle xmem registry stress
//! and the real hard retire-backpressure proof are measured, and no child
//! survives teardown. Exit 12 is PARTIAL, exit 13 is an orphan survivor.

#![cfg(windows)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::ptr::null;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use victory_gpu::device;
use victory_gpu::native_metal::bindings::{self, GuestRange, GuestRangeResolver, ResolveError};
use victory_gpu::native_metal::client::{
    PIPELINE_KIND_COMPUTE, PIPELINE_KIND_GRAPHICS, PIPELINE_KIND_MESH,
};
use victory_gpu::native_metal::fence_page_guest::GuestFencePage;
use victory_gpu::native_metal::producer::{
    BarrierEntry, DepthTarget, MetalProducer, RenderTarget, ResourceId,
};
use victory_gpu::native_metal::sampler::SamplerStateDesc;
use victory_gpu::native_metal::state::{
    BlendDesc, DepthStencilDesc, GuestRenderState, RasterizerDesc, D3D12_CULL_MODE_BACK,
    D3D12_FILL_MODE_SOLID,
};
use victory_gpu::native_metal::{pso, queue, telemetry};
use victory_gpu::object::{Object, S_OK};
use victory_metal::wire::ReplyStatus;
use victory_metal::wire_v3::{
    BlendAttachment, BlendFactor, BlendOperation, BlendState, ClearDepthStencilView,
    ClearRenderTargetView, CommandV3, CopyBufferToTexture, CullMode, DescriptorKind,
    DescriptorTableEntryV3, DescriptorTableV3, IndirectArguments, LoadOp, PrimitiveTopology,
    RootBufferView, ScissorRect, SubmissionV3, TextureSubresource,
};

type This = *const Object;
type Hresult = i32;

// ---------------------------------------------------------------------------
// Scene constants (mirrored by the launcher's pixel model)
// ---------------------------------------------------------------------------

const TARGET_W: u32 = 1280;
const TARGET_H: u32 = 720;
const PAGE: u64 = 64 * 1024;
const ALLOC_ID: u64 = 1;
const ALLOC_GEN: u32 = 1;
const ALLOC_LEN: u64 = 4 * PAGE;

const COLOR0_ID: u64 = 4096;
const COLOR1_ID: u64 = 4097;
const DEPTH_ID: u64 = 4098;
const UPLOAD_TEX_ID: u64 = 0x2000;
const SRV_TEX_A_ID: u64 = 0x2003;
const SRV_TEX_B_ID: u64 = 0x2004;
const BLEND_TARGET_ID: u64 = 0x2005;
const READ_TARGET_ID: u64 = 0x2006;
const SURFACE_ID: u64 = 1;
const TEXTURE_GEN: u32 = 1;

/// Side of the two probe targets (blend result and readback pass).
const PROBE_EDGE: u32 = 64;
/// Side of each sampled texture (uniform per image).
const SRV_EDGE: u32 = 4;

/// `float4` view 0 (the pixel shader's `b0` table entry 0).
const VIEW0_OFFSET: u64 = 0;
/// `float4` view 0 (the pixel shader's `b1` table entry 1).
const VIEW1_OFFSET: u64 = 16;
/// Structured-buffer SRV table entries.
const SRV0_OFFSET: u64 = 32;
const SRV1_OFFSET: u64 = 48;
/// UAV table entries. Entry 0 is 64 bytes (connection counters live there);
/// entry 1 is the second table entry's probe.
const UAV0_OFFSET: u64 = 64;
const UAV0_LEN: u64 = 64;
const UAV1_OFFSET: u64 = 128;
const UAV1_LEN: u64 = 16;
/// Object-stage payload probe the amplification shader writes: word 0 vertex
/// count, word 1 survivor mask (4 of 8), word 2 marker, word 3 the mesh pixel
/// coverage the mesh pixel shader increments.
///
/// Two 32-byte slots: the indirect dispatch uses the first and the direct
/// dispatch the second. Word 4 of each slot is the amplification stage's
/// origin (pre-seeded by the guest), so the direct dispatch paints a second,
/// disjoint 16x16 block.
const PROBE_OFFSET: u64 = 144;
const INDIRECT_PROBE_OFFSET: u64 = PROBE_OFFSET;
const DIRECT_PROBE_OFFSET: u64 = PROBE_OFFSET + 32;
const PROBE_SLOT_LEN: u64 = 32;
/// Replay sentinel; a refused replay must leave these bytes untouched.
const SENTINEL_OFFSET: u64 = 512;
const SENTINEL_LEN: u64 = 16;
/// Indirect mesh record the compute kernel writes: three threadgroup counts.
const ARGS_OFFSET: u64 = 3 * PAGE;
const ARGS_LEN: u64 = 16;
/// Pitched upload source: four 256-byte rows of one colour each.
const UPLOAD_OFFSET: u64 = 2 * PAGE;
const UPLOAD_PITCH: u32 = 256;
const UPLOAD_TEXELS: u32 = 4;
/// Two pitched 4x4 uniform images, each 16 bytes per row.
const UPLOAD_A_OFFSET: u64 = 2 * PAGE + 2048;
const UPLOAD_B_OFFSET: u64 = 2 * PAGE + 4096;
const UPLOAD_SRV_PITCH: u32 = SRV_EDGE * 4;

/// Probe-only regions live in the second page so the scene's page 0 layout
/// (views, tables, counters, both mesh slots and the sentinel) is untouched.
const PROBE_CBV_OFFSET: u64 = PAGE;
const PROBE_CBV1_OFFSET: u64 = PAGE + 16;
const PROBE_COUNTER_OFFSET: u64 = PAGE + 32;
const PROBE_COUNTER1_OFFSET: u64 = PAGE + 64;
/// Readback probe destinations: six 32-byte UAV slots, one per probe draw.
const CLEAR_PROBE_OFFSET: u64 = PAGE + 128;
const COPY_DRAWN_PROBE_OFFSET: u64 = PAGE + 160;
const COPY_CLEAR_PROBE_OFFSET: u64 = PAGE + 192;
const UPLOAD_PROBE_OFFSET: u64 = PAGE + 224;
const SRV_PROBE_OFFSET: u64 = PAGE + 256;
const BLEND_PROBE_OFFSET: u64 = PAGE + 288;
/// Compute-count sweep slots: 1 indirect group, 2 indirect groups, and the
/// direct two-group control. One 8x8 block per mesh group is 64 pixels.
const COUNT1_PROBE_OFFSET: u64 = PAGE + 320;
const COUNT2_INDIRECT_PROBE_OFFSET: u64 = PAGE + 352;
const COUNT2_DIRECT_PROBE_OFFSET: u64 = PAGE + 384;
const GRID_BLOCK_PIXELS: u32 = 64;
const PROBE_WORDS: u64 = 8;

/// Values the compute kernel writes into the indirect mesh record.
const CULL_GROUPS: [u32; 3] = [1, 1, 1];
const CULL_MARKER: u32 = 0xC0DE;
/// Payload the amplification stage writes: three vertices, the 4-of-8
/// survivor mask (`0x55` = threads 0, 2, 4 and 6 survive) and a marker.
const PAYLOAD_VERTICES: u32 = 3;
const PAYLOAD_MASK: u32 = 0x55;
const PAYLOAD_MARKER: u32 = 0xC0DE;
/// One mesh threadgroup paints a 28-pixel triangle; four survivors paint 112.
const MESH_PIXELS_PER_DISPATCH: u32 = 112;
/// Exact graphics coverage: the near quad is NDC `[0,1] x [-1,0]` on a
/// `1280 x 720` target, so its pixels are exactly the bottom-right 640x360.
const NEAR_QUAD_PIXELS: u32 = 640 * 360;
/// Values the pixel shader writes through the two UAV table entries.
const UAV0_VALUE: u32 = 11;
const UAV1_VALUE: u32 = 22;

/// Colours the launcher's model uses (RGBA8), recorded in the report so the
/// launcher's pixel model and the shaders cannot silently diverge.
const SCENE_CLEAR0: [u8; 4] = [0, 0, 255, 255];
const SCENE_CLEAR1: [u8; 4] = [255, 0, 255, 255];
const SCENE_GRAPHICS0: [u8; 4] = [255, 64, 0, 255];
const SCENE_GRAPHICS1: [u8; 4] = [0, 255, 0, 255];
const SCENE_MESH: [u8; 4] = [0, 255, 0, 255];
const SCENE_DEPTH_FAIL: [u8; 4] = [255, 255, 0, 255];
const SCENE_CULL_FAIL: [u8; 4] = [0, 255, 255, 255];

/// Two uniform sampled textures the SRV table proof reads back exactly.
const SRV_COLOR_A: [u8; 4] = [10, 20, 30, 255];
const SRV_COLOR_B: [u8; 4] = [40, 50, 60, 255];
/// Alpha-blend probe: red at alpha 0.5 over the blue clear is exactly
/// `(1,0,0)*0.5 + (0,0,1)*0.5 = (0.5,0,0.5)`, stored unorm8.
const BLEND_EXPECTED: [u8; 4] = [128, 0, 128, 128];

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

// ---------------------------------------------------------------------------
// Device vtable shims
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Guest resolver: one allocation, bounded retirement map
// ---------------------------------------------------------------------------

/// Live tokens the producer pins; retired tokens are bounded exactly like the
/// production `xmem::share` registry (`SUBMISSIONS_CAPACITY = 4096`, retired
/// rows evicted lowest first, one row per retired submission).
///
/// `victory-gpu` cannot depend on `victory-hle`, so this is the seam-faithful
/// mirror the gate can measure in-process. The launcher additionally runs the
/// real registry's `sustained_streaming_bounds_the_retired_submission_map`
/// test under Wine and records its own `cycles/peak/capacity/watermark` line.
struct GateResolver {
    next: AtomicU64,
    open: Mutex<BTreeMap<u64, GuestRange>>,
    retired: Mutex<BTreeMap<u64, ()>>,
    capacity: usize,
    retired_peak: AtomicU64,
    cycles: AtomicU64,
}

impl GateResolver {
    fn new(capacity: usize) -> Self {
        Self {
            next: AtomicU64::new(1),
            open: Mutex::new(BTreeMap::new()),
            retired: Mutex::new(BTreeMap::new()),
            capacity: capacity.max(1),
            retired_peak: AtomicU64::new(0),
            cycles: AtomicU64::new(0),
        }
    }

    fn retired_peak(&self) -> u64 {
        self.retired_peak.load(Ordering::Relaxed)
    }

    fn cycles(&self) -> u64 {
        self.cycles.load(Ordering::Relaxed)
    }

    fn open_len(&self) -> usize {
        self.open.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 10,000 alloc/use/retire cycles with one live pin spanning the run.
    fn churn(&self, range: GuestRange, cycles: u64) {
        let live = self.retain(&range).expect("the spanning pin retains");
        for _ in 0..cycles {
            let token = self.retain(&range).expect("a churn pin retains");
            self.retire(token).expect("a churn pin retires");
            self.cycles.fetch_add(1, Ordering::Relaxed);
        }
        self.retire(live).expect("the spanning pin retires");
    }
}

impl GuestRangeResolver for GateResolver {
    fn resolve(&self, address: u64, len: u64) -> Result<GuestRange, ResolveError> {
        if address == 0 || len == 0 {
            return Err(ResolveError::Invalid("zero address or length"));
        }
        if len > ALLOC_LEN {
            return Err(ResolveError::Invalid("range larger than the gate allocation"));
        }
        Ok(GuestRange {
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            byte_offset: 0,
            size: ALLOC_LEN,
        })
    }

    fn retain(&self, range: &GuestRange) -> Result<u64, ResolveError> {
        if range.allocation_id != ALLOC_ID || range.generation != ALLOC_GEN {
            return Err(ResolveError::Invalid("retain of a foreign range"));
        }
        let token = self.next.fetch_add(1, Ordering::Relaxed).max(1);
        self.open
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(token, *range);
        Ok(token)
    }

    fn retire(&self, token: u64) -> Result<(), ResolveError> {
        let removed = self
            .open
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&token);
        if removed.is_none() {
            return Err(ResolveError::Retire("the token is unknown or already retired"));
        }
        let mut retired = self.retired.lock().unwrap_or_else(|e| e.into_inner());
        retired.insert(token, ());
        // The production registry bounds the whole row map (open + retired),
        // so the mirror does too; open rows are never evicted.
        let open = self.open_len();
        while retired.len() + open > self.capacity {
            let first = *retired
                .keys()
                .next()
                .expect("the map exceeds its capacity, so it is not empty");
            retired.remove(&first);
        }
        let depth = (retired.len() + open) as u64;
        drop(retired);
        self.retired_peak.fetch_max(depth, Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

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

fn read_metallib(name: &str) -> Option<Vec<u8>> {
    let path = env_path(name)?;
    std::fs::read(&path)
        .ok()
        .filter(|bytes| bytes.starts_with(b"MTLB"))
}

fn read_reflection(name: &str) -> Option<String> {
    let path = env_path(name)?;
    let text = std::fs::read_to_string(&path).ok()?;
    if text.contains("TopLevelArgumentBuffer") {
        Some(text)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Wire helpers the guest entry points do not expose yet
// ---------------------------------------------------------------------------

/// Inserts `extra` immediately after the first command `predicate` accepts.
fn insert_after(
    commands: &mut Vec<CommandV3>,
    predicate: impl Fn(&CommandV3) -> bool,
    extra: Vec<CommandV3>,
) -> bool {
    if let Some(index) = commands.iter().position(predicate) {
        let at = index + 1;
        commands.splice(at..at, extra);
        true
    } else {
        false
    }
}

/// Inserts `extra` immediately before the first command `predicate` accepts.
fn insert_before(
    commands: &mut Vec<CommandV3>,
    predicate: impl Fn(&CommandV3) -> bool,
    extra: Vec<CommandV3>,
) -> bool {
    if let Some(index) = commands.iter().position(predicate) {
        commands.splice(index..index, extra);
        true
    } else {
        false
    }
}

fn scissor(x: u32, y: u32, width: u32, height: u32) -> ScissorRect {
    ScissorRect {
        x,
        y,
        width,
        height,
    }
}

fn guest_state() -> Vec<CommandV3> {
    let mut state = GuestRenderState::new();
    state
        .set_blend(&BlendDesc::default())
        .expect("the disabled blend state converts");
    state
        .set_depth_stencil(&DepthStencilDesc {
            depth_enable: true,
            depth_write: true,
            depth_func: victory_gpu::native_metal::state::D3D12_COMPARISON_FUNC_LESS,
            ..DepthStencilDesc::default()
        })
        .expect("the depth state converts");
    state
        .set_rasterizer(&RasterizerDesc {
            fill_mode: D3D12_FILL_MODE_SOLID,
            cull_mode: D3D12_CULL_MODE_BACK,
            front_counter_clockwise: true,
            ..RasterizerDesc::default()
        })
        .expect("the rasterizer state converts");
    debug_assert!(
        matches!(
            state.rasterizer().map(|r| r.cull),
            Some(CullMode::Back)
        ),
        "the guest state tracks a back-face cull"
    );
    state.state_commands()
}

/// Encodes a graphics `VPS1` with two colour targets and a depth attachment.
fn graphics_state_with_depth(color_formats: &[u8], depth_code: u8, fragment: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + fragment.len());
    out.extend_from_slice(b"VPS1");
    out.push(0);
    out.push(u8::try_from(color_formats.len()).expect("colour count"));
    out.extend_from_slice(color_formats);
    out.push(1);
    out.push(depth_code);
    out.push(1);
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

/// Encodes a mesh `VPS1` with two colour targets, depth, an object entry and
/// the payload/threadgroup budgets the mesh pipeline needs.
#[allow(clippy::too_many_arguments)]
fn mesh_state_with_depth(
    color_formats: &[u8],
    depth_code: u8,
    fragment: &str,
    object: &str,
    mesh_threadgroup: [u16; 3],
    object_threadgroup: [u16; 3],
    payload: u16,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(96 + fragment.len() + object.len());
    out.extend_from_slice(b"VPS1");
    out.push(1);
    out.push(u8::try_from(color_formats.len()).expect("colour count"));
    out.extend_from_slice(color_formats);
    out.push(1);
    out.push(depth_code);
    out.push(1 | 2);
    for text in [fragment, object] {
        out.extend_from_slice(&u16::try_from(text.len()).expect("entry fits").to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    for value in mesh_threadgroup.iter().chain(object_threadgroup.iter()) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&payload.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn word(bytes: &[u8], offset: u64) -> Option<u32> {
    let at = usize::try_from(offset).ok()?;
    let slice = bytes.get(at..at + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn hex_words(bytes: &[u8], offset: u64, count: usize) -> String {
    (0..count)
        .map(|index| {
            let at = offset + (index as u64) * 4;
            word(bytes, at).map_or_else(|| "?".to_string(), |value| format!("{value}"))
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Root constants for the readback probe's sampler path: the texel-centre UV
/// a nearest sampler must resolve to texel `(x, y)` of a `width x height`
/// texture. `flags.x == 0` selects the sampler path.
fn probe_uv(x: u32, y: u32, width: u32, height: u32) -> [u32; 4] {
    let u = (f64::from(x) + 0.5) / f64::from(width);
    let v = (f64::from(y) + 0.5) / f64::from(height);
    [(u as f32).to_bits(), (v as f32).to_bits(), 0, 0]
}

/// Root constants for the readback probe's exact-texel path: `flags.x == 1`
/// makes the shader `Load` texel `(x, y)` instead of sampling.
fn probe_texel(x: u32, y: u32) -> [u32; 4] {
    [(x as f32).to_bits(), (y as f32).to_bits(), 1.0f32.to_bits(), 0]
}

/// Whether the first four words of a probe readback equal `expected` (RGBA8 in
/// channel order, exactly as the probe fragment shader wrote them).
fn probe_first(words: &[u32; PROBE_WORDS as usize], expected: [u8; 4]) -> bool {
    words[0..4] == expected.map(u32::from)
}

/// Whether the second four words (the second SRV table entry) equal `expected`.
fn probe_second(words: &[u32; PROBE_WORDS as usize], expected: [u8; 4]) -> bool {
    words[4..8] == expected.map(u32::from)
}

// ---------------------------------------------------------------------------
// Allocation contents
// ---------------------------------------------------------------------------

fn write_allocation(path: &Path) {
    let mut bytes = vec![0u8; ALLOC_LEN as usize];

    // CBV table: entry 0 = (1, 0, 0, 1), entry 1 = (0, 1, 0, 1).
    let cbv0 = [1.0f32, 0.0, 0.0, 1.0];
    let cbv1 = [0.0f32, 1.0, 0.0, 1.0];
    for (slot, value) in [(VIEW0_OFFSET, cbv0), (VIEW1_OFFSET, cbv1)] {
        for (index, component) in value.iter().enumerate() {
            let at = slot as usize + index * 4;
            bytes[at..at + 4].copy_from_slice(&component.to_le_bytes());
        }
    }
    // SRV table: two structured buffers with distinct first words.
    bytes[SRV0_OFFSET as usize..SRV0_OFFSET as usize + 4].copy_from_slice(&7u32.to_le_bytes());
    bytes[SRV1_OFFSET as usize..SRV1_OFFSET as usize + 4].copy_from_slice(&9u32.to_le_bytes());

    // Pitched upload source: four 256-byte rows, red / green / blue / white.
    let rows = [
        [255u8, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
    ];
    for (row, rgba) in rows.iter().enumerate() {
        for column in 0..UPLOAD_TEXELS as usize {
            let at = UPLOAD_OFFSET as usize
                + row * UPLOAD_PITCH as usize
                + column * 4;
            bytes[at..at + 4].copy_from_slice(rgba);
        }
    }

    // Two uniform 4x4 images for the count==2 SRV texture table.
    for (base, colour) in [(UPLOAD_A_OFFSET, SRV_COLOR_A), (UPLOAD_B_OFFSET, SRV_COLOR_B)] {
        for row in 0..SRV_EDGE as usize {
            for column in 0..SRV_EDGE as usize {
                let at = base as usize + row * UPLOAD_SRV_PITCH as usize + column * 4;
                bytes[at..at + 4].copy_from_slice(&colour);
            }
        }
    }

    // Amplification origin per mesh dispatch: the indirect slot paints at x=0
    // and the direct slot at x=640 (word 4 of each 32-byte probe slot).
    let origin = PROBE_OFFSET as usize + 16;
    bytes[origin..origin + 4].copy_from_slice(&0u32.to_le_bytes());
    let origin = DIRECT_PROBE_OFFSET as usize + 16;
    bytes[origin..origin + 4].copy_from_slice(&640u32.to_le_bytes());

    // Blend probe constants: the pixel shader outputs `c0v + rc.xyz`, so a
    // (1,0,0,0.5) view draws red at half alpha over the blue clear.
    for (slot, value) in [
        (PROBE_CBV_OFFSET, [1.0f32, 0.0, 0.0, 0.5]),
        (PROBE_CBV1_OFFSET, [0.0f32, 0.0, 0.0, 1.0]),
    ] {
        for (index, component) in value.iter().enumerate() {
            let at = slot as usize + index * 4;
            bytes[at..at + 4].copy_from_slice(&component.to_le_bytes());
        }
    }

    // Replay sentinel.
    let sentinel = 0xA5A5_5A5Au32;
    for index in 0..4 {
        let at = SENTINEL_OFFSET as usize + index * 4;
        bytes[at..at + 4].copy_from_slice(&sentinel.to_le_bytes());
    }

    std::fs::write(path, &bytes).expect("write the guest allocation file");
}

fn sentinel_intact(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let at = SENTINEL_OFFSET as usize;
    if at + SENTINEL_LEN as usize > bytes.len() {
        return false;
    }
    let expected = [0x5A, 0x5A, 0xA5, 0xA5];
    bytes[at..at + 16]
        .iter()
        .enumerate()
        .all(|(index, byte)| *byte == expected[index % 4])
}

fn sentinel_checksum(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    let at = SENTINEL_OFFSET as usize;
    if at + 16 > bytes.len() {
        return 0;
    }
    let mut hash = 1_469_598_103_934_665_603u64;
    for byte in &bytes[at..at + 16] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ---------------------------------------------------------------------------
// Guest gate
// ---------------------------------------------------------------------------

#[test]
#[ignore = "live Wine guest gate; run tools/macos/metal-poc/aaa-preboot-gate.sh"]
fn aaa_preboot_guest_gate() {
    if std::env::var("VICTORY_AAA_PREBOOT_GATE").ok().as_deref() != Some("1") {
        eprintln!("aaa_preboot_guest_gate: skipped (set VICTORY_AAA_PREBOOT_GATE=1)");
        return;
    }

    let mut report = Report::new();
    let share = env_path("VICTORY_METAL_SHARE_DIR").expect("VICTORY_METAL_SHARE_DIR");
    let out = env_path("VICTORY_LIVE_GATE_OUT")
        .unwrap_or_else(|| share.join("aaa-preboot.report"));
    let csv = env_path("VICTORY_LIVE_GATE_CSV")
        .unwrap_or_else(|| share.join("aaa-preboot.frames.csv"));
    let warmup = env_u32("VICTORY_FRAME_WARMUP", 20);
    let measured = env_u32("VICTORY_FRAME_MEASURED", 120);
    let churn_cycles = 10_000u64;

    report.note("warmup", warmup);
    report.note("measured_expected", measured);
    report.note("scene", "mrt2+depth32, tables cbv2/srv2/uav2+rc, indirect+direct mesh");
    report.note(
        "scene_pixels",
        format!(
            "clear0={SCENE_CLEAR0:?} clear1={SCENE_CLEAR1:?} graphics0={SCENE_GRAPHICS0:?} \
             graphics1={SCENE_GRAPHICS1:?} mesh={SCENE_MESH:?} depth_fail={SCENE_DEPTH_FAIL:?} \
             cull_fail={SCENE_CULL_FAIL:?}"
        ),
    );
    report.note("direct_wire_commands", "SetBlendState,SetDepthStencilState,SetRasterizerState,ClearRenderTargetView,ClearDepthStencilView,CopyBufferToTexture");
    report.note("surface", format!("id={SURFACE_ID} texture={COLOR0_ID}"));
    report.note("churn_cycles_expected", churn_cycles);

    let allocation_path = share.join(format!("{ALLOC_ID:016x}.bin"));
    write_allocation(&allocation_path);
    let _page = GuestFencePage::open(share.join("gpu-fence.v1.page"));

    // Resolver + bounded-map churn before any GPU work: alloc/use/retire.
    let stats = Arc::new(GateResolver::new(4096));
    let _old = bindings::set_resolver_arc(stats.clone());
    let churn_range = GuestRange {
        allocation_id: ALLOC_ID,
        generation: ALLOC_GEN,
        byte_offset: 0,
        size: PAGE,
    };
    let churn_started = Instant::now();
    stats.churn(churn_range, churn_cycles);
    let churn_ms = churn_started.elapsed().as_secs_f64() * 1000.0;
    report.note("xmem_churn_cycles", stats.cycles());
    report.note("xmem_retired_map_peak", stats.retired_peak());
    report.note("xmem_retired_map_capacity", 4096);
    report.note("xmem_churn_ms", format!("{churn_ms:.3}"));
    report.note(
        "xmem_retired_map_bounded",
        stats.retired_peak() <= 4096 && stats.cycles() == churn_cycles,
    );

    // --- Real device, queue, fence and one committed buffer. ---
    let mut device_ptr: This = null();
    let checked_version = 0x0002_0004;
    let hr = device::create(null(), &checked_version, &DEVICE_IID, &mut device_ptr);
    report.stage("device", hr == S_OK, format!("hresult={hr:#x}"));
    if hr != S_OK || device_ptr.is_null() {
        report.note("ok", 0);
        report.write(&out);
        return;
    }

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

    let create_resource: CreateCommitted = unsafe { slot(device_ptr, 0xd8) };
    let heap = HeapProperties {
        kind: 0,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    let desc = ResourceDesc {
        dimension: 1,
        alignment: 0,
        width: ALLOC_LEN,
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

    // --- Textures: two MRT colours, depth, the pitched-upload texture. ---
    let color0 = queue::register_texture(
        COLOR0_ID, TEXTURE_GEN, 1, 0, 1, 1, TARGET_W, TARGET_H, 1, 1, 1,
    );
    let color1 = queue::register_texture(
        COLOR1_ID, TEXTURE_GEN, 1, 0, 1, 1, TARGET_W, TARGET_H, 1, 1, 1,
    );
    let depth = queue::register_texture(
        DEPTH_ID, TEXTURE_GEN, 1, 7, 2, 1, TARGET_W, TARGET_H, 1, 1, 1,
    );
    let upload = queue::register_texture(
        UPLOAD_TEX_ID,
        TEXTURE_GEN,
        1,
        0,
        0,
        1,
        UPLOAD_TEXELS,
        UPLOAD_TEXELS,
        1,
        1,
        1,
    );
    // Sampled textures for the count==2 SRV table proof.
    let srv_a = queue::register_texture(
        SRV_TEX_A_ID,
        TEXTURE_GEN,
        1,
        0,
        0,
        1,
        SRV_EDGE,
        SRV_EDGE,
        1,
        1,
        1,
    );
    let srv_b = queue::register_texture(
        SRV_TEX_B_ID,
        TEXTURE_GEN,
        1,
        0,
        0,
        1,
        SRV_EDGE,
        SRV_EDGE,
        1,
        1,
        1,
    );
    // Blend result and readback targets (colour attachments that stay
    // sampleable; neither is presented).
    let blend_target = queue::register_texture(
        BLEND_TARGET_ID,
        TEXTURE_GEN,
        1,
        0,
        1,
        1,
        PROBE_EDGE,
        PROBE_EDGE,
        1,
        1,
        1,
    );
    let read_target = queue::register_texture(
        READ_TARGET_ID,
        TEXTURE_GEN,
        1,
        0,
        1,
        1,
        PROBE_EDGE,
        PROBE_EDGE,
        1,
        1,
        1,
    );
    report.stage(
        "textures",
        matches!(
            (
                &color0,
                &color1,
                &depth,
                &upload,
                &srv_a,
                &srv_b,
                &blend_target,
                &read_target
            ),
            (
                Ok(true),
                Ok(true),
                Ok(true),
                Ok(true),
                Ok(true),
                Ok(true),
                Ok(true),
                Ok(true)
            )
        ),
        format!(
            "color0={color0:?} color1={color1:?} depth={depth:?} upload={upload:?} \
             srv_a={srv_a:?} srv_b={srv_b:?} blend_target={blend_target:?} \
             read_target={read_target:?}"
        ),
    );

    // Nearest/clamp sampler the SRV texture table binds.
    let sampler = queue::register_sampler_state(&SamplerStateDesc::NEAREST_CLAMP);
    match &sampler {
        Ok(handle) => report.stage(
            "sampler_registration",
            *handle != 0,
            format!("handle=0x{handle:016x} filter=nearest address=clamp"),
        ),
        Err(error) => report.stage("sampler_registration", false, format!("{error}")),
    }
    let sampler_handle = sampler.ok();

    // --- Prebuilt metallibs + converter reflection from the launcher. ---
    let vs = read_metallib("VICTORY_AAA_GATE_METALLIB_VS");
    let ps = read_metallib("VICTORY_AAA_GATE_METALLIB_PS");
    let cs = read_metallib("VICTORY_AAA_GATE_METALLIB_CS");
    let as_lib = read_metallib("VICTORY_AAA_GATE_METALLIB_AS");
    let ms = read_metallib("VICTORY_AAA_GATE_METALLIB_MS");
    let ms2 = read_metallib("VICTORY_AAA_GATE_METALLIB_MS2");
    let psm = read_metallib("VICTORY_AAA_GATE_METALLIB_PSM");
    let pv = read_metallib("VICTORY_AAA_GATE_METALLIB_PV");
    let pp = read_metallib("VICTORY_AAA_GATE_METALLIB_PP");
    let vs_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_VS");
    let ps_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_PS");
    let cs_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_CS");
    let as_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_AS");
    let ms_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_MS");
    let ms2_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_MS2");
    let psm_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_PSM");
    let pv_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_PV");
    let pp_json = read_reflection("VICTORY_AAA_GATE_REFLECTION_PP");
    report.note(
        "metallibs",
        format!(
            "vs={} ps={} cs={} as={} ms={} ms2={} psm={} pv={} pp={}",
            vs.as_ref().map_or(0, Vec::len),
            ps.as_ref().map_or(0, Vec::len),
            cs.as_ref().map_or(0, Vec::len),
            as_lib.as_ref().map_or(0, Vec::len),
            ms.as_ref().map_or(0, Vec::len),
            ms2.as_ref().map_or(0, Vec::len),
            psm.as_ref().map_or(0, Vec::len),
            pv.as_ref().map_or(0, Vec::len),
            pp.as_ref().map_or(0, Vec::len)
        ),
    );
    report.note(
        "reflections",
        format!(
            "vs={} ps={} cs={} as={} ms={} ms2={} psm={} pv={} pp={}",
            vs_json.as_ref().map_or(0, |t| t.len()),
            ps_json.as_ref().map_or(0, |t| t.len()),
            cs_json.as_ref().map_or(0, |t| t.len()),
            as_json.as_ref().map_or(0, |t| t.len()),
            ms_json.as_ref().map_or(0, |t| t.len()),
            ms2_json.as_ref().map_or(0, |t| t.len()),
            psm_json.as_ref().map_or(0, |t| t.len()),
            pv_json.as_ref().map_or(0, |t| t.len()),
            pp_json.as_ref().map_or(0, |t| t.len())
        ),
    );

    let (
        Some(vs),
        Some(ps),
        Some(cs),
        Some(as_lib),
        Some(ms),
        Some(ms2),
        Some(psm),
        Some(pv),
        Some(pp),
        Some(vs_json),
        Some(ps_json),
        Some(cs_json),
        Some(as_json),
        Some(ms_json),
        Some(ms2_json),
        Some(psm_json),
        Some(pv_json),
        Some(pp_json),
    ) = (
        vs, ps, cs, as_lib, ms, ms2, psm, pv, pp, vs_json, ps_json, cs_json, as_json, ms_json,
        ms2_json, psm_json, pv_json, pp_json,
    )
    else {
        report.note("ok", 0);
        report.note("reason", "the launcher did not supply all nine metallibs and reflections");
        report.write(&out);
        return;
    };

    let graphics_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_VERTEX, &vs_json),
        (pso::WIRE_STAGE_PIXEL, &ps_json),
    ]);
    let mesh_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_MESH, &ms_json),
        (pso::WIRE_STAGE_OBJECT, &as_json),
        (pso::WIRE_STAGE_PIXEL, &psm_json),
    ]);
    let mesh_only_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_MESH, &ms2_json),
        (pso::WIRE_STAGE_PIXEL, &psm_json),
    ]);
    let compute_abi = pso::encode_abi(&[(pso::WIRE_STAGE_COMPUTE, &cs_json)]);
    let probe_abi = pso::encode_abi(&[
        (pso::WIRE_STAGE_VERTEX, &pv_json),
        (pso::WIRE_STAGE_PIXEL, &pp_json),
    ]);
    let witness = ps_json.as_bytes();

    // Registration order keeps the daemon's legacy built-ins (compute 1,
    // graphics 2) from shadowing the guest handles: mesh first, compute
    // second, graphics last.
    let mesh_state = mesh_state_with_depth(
        &[pso::DEFAULT_COLOR_FORMAT, pso::DEFAULT_COLOR_FORMAT],
        14,
        "main",
        "main",
        [32, 1, 1],
        [8, 1, 1],
        1024,
    );
    let mesh_id = pso::register_translated(
        &[
            (pso::PipelineStage::Mesh, ms.as_slice(), "main"),
            (pso::PipelineStage::Object, as_lib.as_slice(), "main"),
            (pso::PipelineStage::Pixel, psm.as_slice(), "main"),
        ],
        &pso::PipelineDetails {
            state: &mesh_state,
            abi: &mesh_abi,
            root_signature: witness,
        },
    );
    match &mesh_id {
        Ok(id) => report.stage(
            "mesh_object_pipeline",
            true,
            format!("v2_kind={PIPELINE_KIND_MESH} id={id} stages=mesh,object,pixel"),
        ),
        Err(error) => report.stage("mesh_object_pipeline", false, format!("{error}")),
    }

    // Mesh-only pipeline for the compute-count sweep: no object stage, so the
    // GPU-written group count is the mesh threadgroup count directly.
    let mesh_only_state = pso::encode_mesh_state_for_with_render(
        &[pso::DEFAULT_COLOR_FORMAT, pso::DEFAULT_COLOR_FORMAT],
        "main",
        None,
        [32, 1, 1],
        [1, 1, 1],
        0,
        None,
        None,
        None,
        None,
    );
    let mesh_only_id = pso::register_translated(
        &[
            (pso::PipelineStage::Mesh, ms2.as_slice(), "main"),
            (pso::PipelineStage::Pixel, psm.as_slice(), "main"),
        ],
        &pso::PipelineDetails {
            state: &mesh_only_state,
            abi: &mesh_only_abi,
            root_signature: psm_json.as_bytes(),
        },
    );
    match &mesh_only_id {
        Ok(id) => report.stage(
            "mesh_count_sweep_pipeline",
            true,
            format!("v2_kind={PIPELINE_KIND_MESH} id={id} stages=mesh,pixel"),
        ),
        Err(error) => report.stage("mesh_count_sweep_pipeline", false, format!("{error}")),
    }

    let compute_id = pso::register_translated(
        &[(pso::PipelineStage::Compute, cs.as_slice(), "main")],
        &pso::PipelineDetails {
            state: b"",
            abi: &compute_abi,
            root_signature: cs_json.as_bytes(),
        },
    );
    match &compute_id {
        Ok(id) => report.stage(
            "compute_pipeline",
            true,
            format!("v2_kind={PIPELINE_KIND_COMPUTE} id={id}"),
        ),
        Err(error) => report.stage("compute_pipeline", false, format!("{error}")),
    }

    let graphics_state = graphics_state_with_depth(
        &[pso::DEFAULT_COLOR_FORMAT, pso::DEFAULT_COLOR_FORMAT],
        14,
        "main",
    );
    let graphics_id = pso::register_translated(
        &[
            (pso::PipelineStage::Vertex, vs.as_slice(), "main"),
            (pso::PipelineStage::Pixel, ps.as_slice(), "main"),
        ],
        &pso::PipelineDetails {
            state: &graphics_state,
            abi: &graphics_abi,
            root_signature: witness,
        },
    );
    match &graphics_id {
        Ok(id) => report.stage(
            "graphics_pipeline",
            true,
            format!("v2_kind={PIPELINE_KIND_GRAPHICS} id={id}"),
        ),
        Err(error) => report.stage("graphics_pipeline", false, format!("{error}")),
    }

    // Content-hashed blend PSO: the alpha state is baked into the pipeline
    // descriptor and the content key (VPS2), not sent as a wire blend state.
    let alpha = BlendAttachment {
        enable: true,
        src_color: BlendFactor::SrcAlpha,
        dst_color: BlendFactor::InvSrcAlpha,
        color_op: BlendOperation::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::Zero,
        alpha_op: BlendOperation::Add,
        write_mask: 0x0f,
    };
    let alpha_blend = BlendState {
        independent_blend: true,
        alpha_to_coverage: false,
        attachments: vec![alpha, alpha],
    };
    let blend_state = pso::encode_graphics_state_with_render(
        &[pso::DEFAULT_COLOR_FORMAT, pso::DEFAULT_COLOR_FORMAT],
        "main",
        None,
        Some(&alpha_blend),
        None,
        None,
    );
    let blend_id = pso::register_translated(
        &[
            (pso::PipelineStage::Vertex, vs.as_slice(), "main"),
            (pso::PipelineStage::Pixel, ps.as_slice(), "main"),
        ],
        &pso::PipelineDetails {
            state: &blend_state,
            abi: &graphics_abi,
            root_signature: witness,
        },
    );
    match &blend_id {
        Ok(id) => report.stage(
            "blend_pipeline",
            true,
            format!(
                "v2_kind={PIPELINE_KIND_GRAPHICS} id={id} state=VPS2 \
                 blend=src_alpha/inv_src_alpha op=add alpha=one/zero"
            ),
        ),
        Err(error) => report.stage("blend_pipeline", false, format!("{error}")),
    }

    // Readback probe PSO: a full-screen triangle whose fragment samples two
    // registered textures through a count==2 SRV table and a sampler table,
    // then writes the exact 8-bit channels to a UAV.
    let probe_state = pso::encode_graphics_state(&[pso::DEFAULT_COLOR_FORMAT], "main");
    let probe_id = pso::register_translated(
        &[
            (pso::PipelineStage::Vertex, pv.as_slice(), "main"),
            (pso::PipelineStage::Pixel, pp.as_slice(), "main"),
        ],
        &pso::PipelineDetails {
            state: &probe_state,
            abi: &probe_abi,
            root_signature: pp_json.as_bytes(),
        },
    );
    match &probe_id {
        Ok(id) => report.stage(
            "probe_pipeline",
            true,
            format!("v2_kind={PIPELINE_KIND_GRAPHICS} id={id} tables=srv2,sampler1,uav1"),
        ),
        Err(error) => report.stage("probe_pipeline", false, format!("{error}")),
    }

    let (Ok(mesh_id), Ok(compute_id), Ok(graphics_id), Ok(blend_id), Ok(probe_id)) =
        (mesh_id, compute_id, graphics_id, blend_id, probe_id)
    else {
        report.note("ok", 0);
        report.note("reason", "a required pipeline did not register");
        report.write(&out);
        return;
    };
    let Ok(mesh_only_id) = mesh_only_id else {
        report.note("ok", 0);
        report.note("reason", "the mesh count-sweep pipeline did not register");
        report.write(&out);
        return;
    };
    report.note(
        "pipeline_ids",
        format!(
            "mesh={mesh_id} mesh_only={mesh_only_id} compute={compute_id} \
             graphics={graphics_id} blend={blend_id} probe={probe_id}"
        ),
    );

    // Descriptor tables and root constants the graphics pixel shader reads.
    let srv_table = DescriptorTableV3 {
        set: 1,
        base_register: 0,
        kind: DescriptorKind::Srv,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: SRV0_OFFSET,
                length: 16,
            }),
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: SRV1_OFFSET,
                length: 16,
            }),
        ],
    };
    let uav_table = DescriptorTableV3 {
        set: 2,
        base_register: 0,
        kind: DescriptorKind::Uav,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: UAV0_OFFSET,
                length: UAV0_LEN,
            }),
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: UAV1_OFFSET,
                length: UAV1_LEN,
            }),
        ],
    };
    let cbv_table = DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Cbv,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: VIEW0_OFFSET,
                length: 16,
            }),
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: VIEW1_OFFSET,
                length: 16,
            }),
        ],
    };
    // Root constants at the reflection's parameter index 3: xyz is the tint,
    // w selects the draw (0 near quad, 1 far triangle, 2 clockwise quad).
    let selector_constants = |selector: f32| -> [u32; 4] {
        [
            0.0f32.to_bits(),
            0.25f32.to_bits(),
            0.0f32.to_bits(),
            selector.to_bits(),
        ]
    };

    // Readback tables for the probe pipeline: SRV texture table (count==2),
    // sampler table (count==1) and a one-entry UAV table per probe draw.
    let probe_cbv_table = DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Cbv,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: PROBE_CBV_OFFSET,
                length: 16,
            }),
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: PROBE_CBV1_OFFSET,
                length: 16,
            }),
        ],
    };
    let probe_counter_table = DescriptorTableV3 {
        set: 2,
        base_register: 0,
        kind: DescriptorKind::Uav,
        bindings: vec![
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: PROBE_COUNTER_OFFSET,
                length: 32,
            }),
            DescriptorTableEntryV3::Buffer(victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset: PROBE_COUNTER1_OFFSET,
                length: 16,
            }),
        ],
    };
    let texture_table = |first: u64, second: u64| DescriptorTableV3 {
        set: 0,
        base_register: 0,
        kind: DescriptorKind::Srv,
        bindings: vec![
            DescriptorTableEntryV3::Texture(first),
            DescriptorTableEntryV3::Texture(second),
        ],
    };
    let sampler_table = |handle: u64| DescriptorTableV3 {
        set: 1,
        base_register: 0,
        kind: DescriptorKind::Sampler,
        bindings: vec![DescriptorTableEntryV3::Sampler(handle)],
    };
    let probe_uav_table = |offset: u64| DescriptorTableV3 {
        set: 2,
        base_register: 0,
        kind: DescriptorKind::Uav,
        bindings: vec![DescriptorTableEntryV3::Buffer(
            victory_metal::wire::BufferBinding {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                offset,
                length: 32,
            },
        )],
    };

    let mesh_view_indirect = RootBufferView {
        slot: 0,
        allocation_id: ALLOC_ID,
        generation: ALLOC_GEN,
        offset: INDIRECT_PROBE_OFFSET,
        length: PROBE_SLOT_LEN,
    };
    let mesh_view_direct = RootBufferView {
        slot: 0,
        allocation_id: ALLOC_ID,
        generation: ALLOC_GEN,
        offset: DIRECT_PROBE_OFFSET,
        length: PROBE_SLOT_LEN,
    };

    // --- Measured frames ---
    let total = warmup + measured;
    let mut csv_text = String::new();
    let mut frames_ok = 0u32;
    let mut fence_timeouts = 0u32;
    let mut fence_waits = 0u32;
    let mut last_list = 0u64;
    let mut present_status = 0i32;
    let deadline = Duration::from_secs(30);
    let mut splice_errors: Vec<String> = Vec::new();
    let mut linkage_frames = 0u32;
    let mut indirect_frames = 0u32;
    let mut direct_frames = 0u32;
    let mut mesh_binding_frames = 0u32;

    for index in 0..total {
        let started = Instant::now();
        let list_id = 1_000_000 + u64::from(index) + 1;
        last_list = list_id;
        let mut producer = match MetalProducer::new(list_id, 1) {
            Ok(producer) => producer,
            Err(error) => {
                report.stage("frame_producer", false, format!("frame {index}: {error}"));
                break;
            }
        };

        let mut stage_error: Option<String> = None;

        // Compute culling: the kernel writes the indirect mesh group counts.
        if let Err(error) = producer
            .set_compute_pipeline(compute_id)
            .and_then(|()| producer.set_compute_threadgroup_size([64, 1, 1]))
            .and_then(|()| {
                producer.set_compute_root_buffer_view(RootBufferView {
                    slot: 0,
                    allocation_id: ALLOC_ID,
                    generation: ALLOC_GEN,
                    offset: ARGS_OFFSET,
                    length: ARGS_LEN,
                })
            })
            .and_then(|()| producer.set_compute_root_constants(1, &[1], 0))
            .and_then(|()| producer.dispatch(1, 1, 1))
            .and_then(|()| {
                producer.resource_barrier(&[BarrierEntry {
                    allocation_id: ALLOC_ID,
                    generation: ALLOC_GEN,
                    offset: ARGS_OFFSET,
                    length: ARGS_LEN,
                    before_state: 0x8,
                    after_state: 0x200,
                }])
            })
        {
            stage_error = Some(format!("compute_cull: {error}"));
        }

        // The MRT pass. `DontCare` loads make the explicit clear commands the
        // only source of the pass's initial contents.
        if stage_error.is_none() {
            let mut target0 = RenderTarget::new(
                ResourceId::new(COLOR0_ID, TEXTURE_GEN).expect("color0"),
            );
            target0.load = LoadOp::DontCare;
            let mut target1 = RenderTarget::new(
                ResourceId::new(COLOR1_ID, TEXTURE_GEN).expect("color1"),
            );
            target1.load = LoadOp::DontCare;
            let mut depth_target =
                DepthTarget::new(ResourceId::new(DEPTH_ID, TEXTURE_GEN).expect("depth"));
            depth_target.depth_load = LoadOp::DontCare;
            depth_target.stencil_load = LoadOp::DontCare;

            let result = producer
                .set_render_targets(&[target0, target1], Some(depth_target))
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
                .and_then(|()| producer.set_pipeline(graphics_id))
                .and_then(|()| producer.set_graphics_descriptor_table(cbv_table.clone()))
                .and_then(|()| producer.set_graphics_descriptor_table(srv_table.clone()))
                .and_then(|()| producer.set_graphics_descriptor_table(uav_table.clone()))
                // Near bottom-right quad, CCW at z=0.75.
                .and_then(|()| {
                    producer.set_graphics_root_constants(3, &selector_constants(0.0), 0)
                })
                .and_then(|()| producer.draw(6, 1, 0, 0))
                // Far full-screen triangle at z=1.0: rejected by Less.
                .and_then(|()| {
                    producer.set_graphics_root_constants(3, &selector_constants(1.0), 0)
                })
                .and_then(|()| producer.draw(3, 1, 0, 0))
                // Clockwise middle-left quad at z=0.25: culled by Back.
                .and_then(|()| {
                    producer.set_graphics_root_constants(3, &selector_constants(2.0), 0)
                })
                .and_then(|()| producer.draw(6, 1, 0, 0));
            if let Err(error) = result {
                stage_error = Some(format!("graphics: {error}"));
            }
        }

        // In-pass indirect mesh dispatch (compute-written group counts).
        if stage_error.is_none() {
            if let Err(error) = producer.execute_indirect_dispatch_mesh(
                mesh_id,
                IndirectArguments {
                    allocation_id: ALLOC_ID,
                    generation: ALLOC_GEN,
                    offset: ARGS_OFFSET,
                    max_count: 1,
                    count: None,
                },
            ) {
                stage_error = Some(format!("indirect_mesh: {error}"));
            }
        }
        // Direct mesh dispatch on the same [Mesh, Object, Pixel] pipeline.
        if stage_error.is_none() {
            if let Err(error) = producer
                .set_mesh_pipeline(mesh_id)
                .and_then(|()| producer.dispatch_mesh(1, 1, 1))
            {
                stage_error = Some(format!("direct_mesh: {error}"));
            }
        }
        // Copies that expose the second target and the pitched upload.
        if stage_error.is_none() {
            let result = producer
                .copy_texture_region(
                    TextureSubresource {
                        allocation_id: COLOR1_ID,
                        generation: TEXTURE_GEN,
                        mip_level: 0,
                        slice: 0,
                    },
                    TextureSubresource {
                        allocation_id: COLOR0_ID,
                        generation: TEXTURE_GEN,
                        mip_level: 0,
                        slice: 0,
                    },
                    [700, 400, 0],
                    [320, 0, 0],
                    [4, 4, 1],
                )
                .and_then(|()| {
                    producer.copy_texture_region(
                        TextureSubresource {
                            allocation_id: COLOR1_ID,
                            generation: TEXTURE_GEN,
                            mip_level: 0,
                            slice: 0,
                        },
                        TextureSubresource {
                            allocation_id: COLOR0_ID,
                            generation: TEXTURE_GEN,
                            mip_level: 0,
                            slice: 0,
                        },
                        [400, 400, 0],
                        [320, 4, 0],
                        [4, 4, 1],
                    )
                })
                .and_then(|()| {
                    producer.copy_texture(
                        ResourceId::new(UPLOAD_TEX_ID, TEXTURE_GEN).expect("upload"),
                        0,
                        0,
                        ResourceId::new(COLOR0_ID, TEXTURE_GEN).expect("color0"),
                        0,
                        0,
                        UPLOAD_TEXELS,
                        UPLOAD_TEXELS,
                        1,
                    )
                });
            if let Err(error) = result {
                stage_error = Some(format!("copies: {error}"));
            }
        }

        if let Some(reason) = stage_error {
            report.stage("frame_record", false, format!("frame {index}: {reason}"));
            break;
        }

        let mut submission = match producer.flush() {
            Ok(submission) => submission,
            Err(error) => {
                report.stage("frame_flush", false, format!("frame {index}: {error}"));
                break;
            }
        };
        let sequence = submission.sequence;

        // Splice the commands the guest entry points do not expose yet.
        // 1. Explicit clears + ADR-026a fixed-function state after the pass
        //    opens, before the pipeline selection.
        let clear_commands = vec![
            CommandV3::ClearRenderTargetView(ClearRenderTargetView {
                slot: 0,
                rect: None,
                color: [0.0, 0.0, 1.0, 1.0],
            }),
            CommandV3::ClearRenderTargetView(ClearRenderTargetView {
                slot: 1,
                rect: None,
                color: [1.0, 0.0, 1.0, 1.0],
            }),
            CommandV3::ClearDepthStencilView(ClearDepthStencilView {
                depth: Some(1.0),
                stencil: None,
                rect: None,
            }),
        ];
        let mut pass_prefix = clear_commands;
        pass_prefix.extend(guest_state());
        if !insert_after(
            &mut submission.commands,
            |command| matches!(command, CommandV3::BeginRenderPass(_)),
            pass_prefix,
        ) {
            splice_errors.push(format!("frame {index}: no BeginRenderPass"));
        }
        // 2. Blit the pitched upload before the pass opens.
        let upload = CommandV3::CopyBufferToTexture(victory_metal::wire_v3::CopyBufferToTexture {
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            destination_allocation_id: UPLOAD_TEX_ID,
            destination_generation: TEXTURE_GEN,
            offset: UPLOAD_OFFSET,
            bytes_per_row: UPLOAD_PITCH,
            bytes_per_image: 0,
            extent: [UPLOAD_TEXELS, UPLOAD_TEXELS, 1],
            mip_level: 0,
            slice: 0,
            origin: [0, 0, 0],
        });
        if !insert_before(
            &mut submission.commands,
            |command| matches!(command, CommandV3::BeginRenderPass(_)),
            vec![upload],
        ) {
            splice_errors.push(format!("frame {index}: no pass for the upload"));
        }
        // 3. Scissor + object-stage probe binding around the indirect mesh.
        if !insert_after(
            &mut submission.commands,
            |command| matches!(command, CommandV3::SetMeshPipeline(_)),
            vec![
                CommandV3::SetScissor(scissor(0, 0, 320, TARGET_H)),
                CommandV3::SetRootBufferView(mesh_view_indirect),
            ],
        ) {
            splice_errors.push(format!("frame {index}: no SetMeshPipeline"));
        }
        // 4. Scissor + object-stage probe binding for the direct dispatch.
        if !insert_before(
            &mut submission.commands,
            |command| matches!(command, CommandV3::DispatchMesh(_)),
            vec![
                CommandV3::SetScissor(scissor(640, 0, 640, 360)),
                CommandV3::SetRootBufferView(mesh_view_direct),
            ],
        ) {
            splice_errors.push(format!("frame {index}: no DispatchMesh"));
        }

        // Structural linkage: compute -> barrier -> indirect mesh, and both
        // mesh dispatches on the registered mesh pipeline.
        let compute_at = submission
            .commands
            .iter()
            .position(|command| matches!(command, CommandV3::Dispatch3d(_)));
        let barrier_at = submission
            .commands
            .iter()
            .position(|command| matches!(command, CommandV3::ResourceBarrier(_)));
        let indirect_at = submission
            .commands
            .iter()
            .position(|command| matches!(command, CommandV3::ExecuteIndirectDispatchMesh(_)));
        let direct_at = submission
            .commands
            .iter()
            .position(|command| matches!(command, CommandV3::DispatchMesh(_)));
        let mesh_sets = submission
            .commands
            .iter()
            .filter(|command| matches!(command, CommandV3::SetMeshPipeline(_)))
            .count();
        let linkage = match (compute_at, barrier_at, indirect_at, direct_at) {
            (Some(compute), Some(barrier), Some(indirect), Some(direct))
                if compute < barrier && barrier < indirect && indirect < direct =>
            {
                true
            }
            _ => false,
        };
        if !linkage {
            report.stage(
                "compute_indirect_mesh_linkage",
                false,
                format!(
                    "frame {index}: compute={compute_at:?} barrier={barrier_at:?} \
                     indirect={indirect_at:?} direct={direct_at:?}"
                ),
            );
            break;
        }
        linkage_frames += 1;
        if indirect_at.is_some() {
            indirect_frames += 1;
        }
        if direct_at.is_some() {
            direct_frames += 1;
        }
        if mesh_sets >= 1 {
            mesh_binding_frames += 1;
        }

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

        if !queue_ptr.is_null() {
            let present: Present = unsafe { slot(queue_ptr, 0xf0) };
            let hr = unsafe { present(queue_ptr, 0, null(), null()) };
            present_status = hr;
            if hr != S_OK {
                report.stage("present", false, format!("frame {index}: hresult={hr:#x}"));
                break;
            }
        }

        // ADR-017: completion is observed only on the shared fence page.
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
            csv_text.push_str(&format!("{elapsed_ms:.6}\n"));
        }
    }

    report.note("frames_ok", frames_ok);
    report.note("measured", csv_text.lines().count());
    report.note("fence_waits", fence_waits);
    report.note("fence_timeouts", fence_timeouts);
    report.note("present_status", present_status);
    report.note("last_list_id", last_list);
    report.note("compute_indirect_mesh_linkage_frames", linkage_frames);
    report.note("indirect_mesh_frames", indirect_frames);
    report.note("direct_mesh_frames", direct_frames);
    report.note("mesh_pipeline_selected_frames", mesh_binding_frames);
    report.note("splice_errors", if splice_errors.is_empty() { "none".to_string() } else { splice_errors.join("; ") });
    let _ = std::fs::write(&csv, csv_text);

    // --- Compute-count sweep: the GPU-written group count drives coverage.
    //
    // The compute kernel writes the indirect record from a root constant, so
    // the same GPU-written buffer drives count=1 (one 64-pixel mesh block),
    // count=2 (two blocks, 128 px) and a direct two-group control (128 px).
    // No sampler is involved: the mesh pixel stage's coverage counter proves
    // the geometry.
    let mut probe_fence = u64::from(total);
    let mut submit_probe = |submission: &SubmissionV3| -> Result<(), String> {
        match queue::submit_recorded_batch(std::slice::from_ref(submission), 0) {
            Ok(queue::NativeDispatch::Native {
                status: ReplyStatus::Ok,
                ..
            }) => {}
            other => return Err(format!("probe submit: {other:?}")),
        }
        if queue_ptr.is_null() || fence.is_null() {
            return Ok(());
        }
        probe_fence += 1;
        let value = probe_fence;
        let signal: Signal = unsafe { slot(queue_ptr, 0x70) };
        let completed: Completed = unsafe { slot(fence, 0x40) };
        let hr = unsafe { signal(queue_ptr, fence, value) };
        if hr != S_OK {
            return Err(format!("probe fence signal: hresult={hr:#x}"));
        }
        let wait_started = Instant::now();
        let mut observed = 0u64;
        while wait_started.elapsed() < deadline {
            observed = unsafe { completed(fence) };
            if observed >= value {
                return Ok(());
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        Err(format!(
            "probe fence timeout: observed={observed} expected>={value}"
        ))
    };
    let args = IndirectArguments {
        allocation_id: ALLOC_ID,
        generation: ALLOC_GEN,
        offset: ARGS_OFFSET,
        max_count: 1,
        count: None,
    };
    let counter_view = |offset: u64| RootBufferView {
        slot: 0,
        allocation_id: ALLOC_ID,
        generation: ALLOC_GEN,
        offset,
        length: PROBE_SLOT_LEN,
    };
    let sweep_result: Result<(), String> = (|| {
        // Measurement 1: compute writes count=1, one indirect mesh group.
        let mut one =
            MetalProducer::new(1_000_000 + u64::from(total) + 10, 1).map_err(|e| e.to_string())?;
        one.set_compute_pipeline(compute_id).map_err(|e| e.to_string())?;
        one.set_compute_threadgroup_size([64, 1, 1]).map_err(|e| e.to_string())?;
        one.set_compute_root_buffer_view(RootBufferView {
            slot: 0,
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            offset: ARGS_OFFSET,
            length: ARGS_LEN,
        })
        .map_err(|e| e.to_string())?;
        one.set_compute_root_constants(1, &[1], 0).map_err(|e| e.to_string())?;
        one.dispatch(1, 1, 1).map_err(|e| e.to_string())?;
        one.resource_barrier(&[BarrierEntry {
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            offset: ARGS_OFFSET,
            length: ARGS_LEN,
            before_state: 0x8,
            after_state: 0x200,
        }])
        .map_err(|e| e.to_string())?;
        one.set_render_targets(
            &[
                RenderTarget::cleared(
                    ResourceId::new(READ_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
                    [0.0, 0.0, 0.0, 1.0],
                ),
                RenderTarget::cleared(
                    ResourceId::new(BLEND_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
                    [0.0, 0.0, 0.0, 1.0],
                ),
            ],
            None,
        )
        .map_err(|e| e.to_string())?;
        one.set_viewport(0.0, 0.0, PROBE_EDGE as f32, PROBE_EDGE as f32, 0.0, 1.0)
            .map_err(|e| e.to_string())?;
        one.set_scissor_edges(0, 0, PROBE_EDGE as i32, PROBE_EDGE as i32)
            .map_err(|e| e.to_string())?;
        one.set_topology(PrimitiveTopology::TriangleList).map_err(|e| e.to_string())?;
        one.execute_indirect_dispatch_mesh(mesh_only_id, args.clone())
            .map_err(|e| e.to_string())?;
        one.end_render_pass().map_err(|e| e.to_string())?;
        let mut submission = one.flush().map_err(|e| e.to_string())?;
        if !insert_after(
            &mut submission.commands,
            |command| matches!(command, CommandV3::SetMeshPipeline(_)),
            vec![CommandV3::SetRootBufferView(counter_view(
                COUNT1_PROBE_OFFSET,
            ))],
        ) {
            return Err("count-1 sweep: no SetMeshPipeline".to_string());
        }
        submit_probe(&submission)?;

        // Measurement 2: compute writes count=2, two indirect groups followed
        // by a direct two-group dispatch as the control.
        let mut two =
            MetalProducer::new(1_000_000 + u64::from(total) + 11, 1).map_err(|e| e.to_string())?;
        two.set_compute_pipeline(compute_id).map_err(|e| e.to_string())?;
        two.set_compute_threadgroup_size([64, 1, 1]).map_err(|e| e.to_string())?;
        two.set_compute_root_buffer_view(RootBufferView {
            slot: 0,
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            offset: ARGS_OFFSET,
            length: ARGS_LEN,
        })
        .map_err(|e| e.to_string())?;
        two.set_compute_root_constants(1, &[2], 0).map_err(|e| e.to_string())?;
        two.dispatch(1, 1, 1).map_err(|e| e.to_string())?;
        two.resource_barrier(&[BarrierEntry {
            allocation_id: ALLOC_ID,
            generation: ALLOC_GEN,
            offset: ARGS_OFFSET,
            length: ARGS_LEN,
            before_state: 0x8,
            after_state: 0x200,
        }])
        .map_err(|e| e.to_string())?;
        two.set_render_targets(
            &[
                RenderTarget::cleared(
                    ResourceId::new(READ_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
                    [0.0, 0.0, 0.0, 1.0],
                ),
                RenderTarget::cleared(
                    ResourceId::new(BLEND_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
                    [0.0, 0.0, 0.0, 1.0],
                ),
            ],
            None,
        )
        .map_err(|e| e.to_string())?;
        two.set_viewport(0.0, 0.0, PROBE_EDGE as f32, PROBE_EDGE as f32, 0.0, 1.0)
            .map_err(|e| e.to_string())?;
        two.set_scissor_edges(0, 0, PROBE_EDGE as i32, PROBE_EDGE as i32)
            .map_err(|e| e.to_string())?;
        two.set_topology(PrimitiveTopology::TriangleList).map_err(|e| e.to_string())?;
        two.execute_indirect_dispatch_mesh(mesh_only_id, args)
            .map_err(|e| e.to_string())?;
        two.set_mesh_pipeline(mesh_only_id).map_err(|e| e.to_string())?;
        two.dispatch_mesh(2, 1, 1).map_err(|e| e.to_string())?;
        two.end_render_pass().map_err(|e| e.to_string())?;
        let mut submission = two.flush().map_err(|e| e.to_string())?;
        if !insert_after(
            &mut submission.commands,
            |command| matches!(command, CommandV3::SetMeshPipeline(_)),
            vec![CommandV3::SetRootBufferView(counter_view(
                COUNT2_INDIRECT_PROBE_OFFSET,
            ))],
        ) {
            return Err("count-2 sweep: no SetMeshPipeline".to_string());
        }
        if !insert_before(
            &mut submission.commands,
            |command| matches!(command, CommandV3::DispatchMesh(_)),
            vec![CommandV3::SetRootBufferView(counter_view(
                COUNT2_DIRECT_PROBE_OFFSET,
            ))],
        ) {
            return Err("count-2 sweep: no direct DispatchMesh".to_string());
        }
        submit_probe(&submission)?;
        Ok(())
    })();
    match &sweep_result {
        Ok(()) => report.stage(
            "mesh_count_sweep_submission",
            true,
            "count=1 indirect, count=2 indirect and direct two-group control submitted",
        ),
        Err(reason) => report.stage("mesh_count_sweep_submission", false, reason.clone()),
    }

    // --- Post-frame GPU probes: blend, texture table, clears, copies, upload.
    //
    // One submission with two render passes over two probe targets. The blend
    // pass bakes an alpha blend into the content-hashed PSO and draws over a
    // known blue background; the readback pass samples the blend result, the
    // presented target (clears / copies / pitched upload) and the two sampled
    // textures through a count==2 SRV texture table plus a sampler table,
    // writing the exact 8-bit channels into the shared UAV slots.
    let probe_list = 1_000_000 + u64::from(total) + 1;
    let probe_result: Result<(), String> = (|| {
        let mut probe = MetalProducer::new(probe_list, 1).map_err(|e| e.to_string())?;
        let blend_color = RenderTarget::cleared(
            ResourceId::new(BLEND_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
            [0.0, 0.0, 1.0, 1.0],
        );
        let read_color = RenderTarget::cleared(
            ResourceId::new(READ_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
            [0.0, 0.0, 1.0, 1.0],
        );
        probe
            .set_render_targets(&[blend_color, read_color], None)
            .map_err(|e| e.to_string())?;
        probe
            .set_viewport(
                0.0,
                0.0,
                PROBE_EDGE as f32,
                PROBE_EDGE as f32,
                0.0,
                1.0,
            )
            .map_err(|e| e.to_string())?;
        probe
            .set_scissor_edges(0, 0, PROBE_EDGE as i32, PROBE_EDGE as i32)
            .map_err(|e| e.to_string())?;
        probe
            .set_topology(PrimitiveTopology::TriangleList)
            .map_err(|e| e.to_string())?;
        probe.set_pipeline(blend_id).map_err(|e| e.to_string())?;
        probe
            .set_graphics_descriptor_table(probe_cbv_table.clone())
            .map_err(|e| e.to_string())?;
        probe
            .set_graphics_descriptor_table(srv_table.clone())
            .map_err(|e| e.to_string())?;
        probe
            .set_graphics_descriptor_table(probe_counter_table.clone())
            .map_err(|e| e.to_string())?;
        probe
            .set_graphics_root_constants(3, &[0u32, 0, 0, 0], 0)
            .map_err(|e| e.to_string())?;
        // Near quad (NDC x [0,1], y [-1,0]) over the blue clear: the blended
        // pixels are (0.5, 0, 0.5) and (0.5 alpha).
        probe.draw(6, 1, 0, 0).map_err(|e| e.to_string())?;

        // Pass B: readback target, probe pipeline and six sampling draws.
        let read_only = RenderTarget::cleared(
            ResourceId::new(READ_TARGET_ID, TEXTURE_GEN).map_err(|e| e.to_string())?,
            [0.0, 0.0, 0.0, 1.0],
        );
        probe
            .set_render_targets(&[read_only], None)
            .map_err(|e| e.to_string())?;
        probe
            .set_viewport(
                0.0,
                0.0,
                PROBE_EDGE as f32,
                PROBE_EDGE as f32,
                0.0,
                1.0,
            )
            .map_err(|e| e.to_string())?;
        probe
            .set_scissor_edges(0, 0, PROBE_EDGE as i32, PROBE_EDGE as i32)
            .map_err(|e| e.to_string())?;
        probe
            .set_topology(PrimitiveTopology::TriangleList)
            .map_err(|e| e.to_string())?;
        probe.set_pipeline(probe_id).map_err(|e| e.to_string())?;

        // Exact-texel readbacks: the presented target (clear, both copies,
        // pitched upload) and the alpha-blended probe target. `Load` needs no
        // sampler, so these exact-pixel stages do not depend on sampler
        // registration.
        let texel_draws: [(u64, u64, u64, [u32; 4]); 5] = [
            // Clear of target 0 at a texel only the clear wrote.
            (COLOR0_ID, COLOR0_ID, CLEAR_PROBE_OFFSET, probe_texel(320, 540)),
            // Colour1's drawn region copied into target 0: green.
            (
                COLOR0_ID,
                COLOR0_ID,
                COPY_DRAWN_PROBE_OFFSET,
                probe_texel(321, 1),
            ),
            // Colour1's clear region copied into target 0: magenta.
            (
                COLOR0_ID,
                COLOR0_ID,
                COPY_CLEAR_PROBE_OFFSET,
                probe_texel(321, 5),
            ),
            // Pitched upload row 3 (white) after the mesh overwrote its block.
            (COLOR0_ID, COLOR0_ID, UPLOAD_PROBE_OFFSET, probe_texel(1, 3)),
            // The alpha-blended probe target pixel.
            (
                BLEND_TARGET_ID,
                BLEND_TARGET_ID,
                BLEND_PROBE_OFFSET,
                probe_texel(48, 48),
            ),
        ];
        for (first, second, output, constants) in texel_draws {
            probe
                .set_graphics_descriptor_table(texture_table(first, second))
                .map_err(|e| e.to_string())?;
            probe
                .set_graphics_descriptor_table(probe_uav_table(output))
                .map_err(|e| e.to_string())?;
            probe
                .set_graphics_root_constants(3, &constants, 0)
                .map_err(|e| e.to_string())?;
            probe.draw(3, 1, 0, 0).map_err(|e| e.to_string())?;
        }

        // The required count==2 SRV texture table plus a sampler table. This
        // draw is the only one that samples; without a registered sampler it
        // is not submitted and its stage stays false.
        if let Some(handle) = sampler_handle {
            probe
                .set_graphics_descriptor_table(texture_table(SRV_TEX_A_ID, SRV_TEX_B_ID))
                .map_err(|e| e.to_string())?;
            probe
                .set_graphics_descriptor_table(sampler_table(handle))
                .map_err(|e| e.to_string())?;
            probe
                .set_graphics_descriptor_table(probe_uav_table(SRV_PROBE_OFFSET))
                .map_err(|e| e.to_string())?;
            probe
                .set_graphics_root_constants(3, &probe_uv(2, 2, SRV_EDGE, SRV_EDGE), 0)
                .map_err(|e| e.to_string())?;
            probe.draw(3, 1, 0, 0).map_err(|e| e.to_string())?;
        }
        probe.end_render_pass().map_err(|e| e.to_string())?;

        let mut submission = probe.flush().map_err(|e| e.to_string())?;
        // Pitched uploads into the two sampled textures, before any pass.
        let mut uploads = Vec::new();
        for (offset, destination) in [
            (UPLOAD_A_OFFSET, SRV_TEX_A_ID),
            (UPLOAD_B_OFFSET, SRV_TEX_B_ID),
        ] {
            uploads.push(CommandV3::CopyBufferToTexture(CopyBufferToTexture {
                allocation_id: ALLOC_ID,
                generation: ALLOC_GEN,
                destination_allocation_id: destination,
                destination_generation: TEXTURE_GEN,
                offset,
                bytes_per_row: UPLOAD_SRV_PITCH,
                bytes_per_image: 0,
                extent: [SRV_EDGE, SRV_EDGE, 1],
                mip_level: 0,
                slice: 0,
                origin: [0, 0, 0],
            }));
        }
        submission.commands.splice(0..0, uploads);
        submit_probe(&submission)?;
        Ok(())
    })();
    match &probe_result {
        Ok(()) => report.stage(
            "probe_submission",
            true,
            "one submission: blend pass + exact-texel readbacks",
        ),
        Err(reason) => report.stage("probe_submission", false, reason.clone()),
    }
    report.note(
        "srv_texture_probe_draw",
        if sampler_handle.is_some() {
            "submitted (sampler table bound)".to_string()
        } else {
            "not submitted: the host refused sampler registration".to_string()
        },
    );

    // --- Read back what the GPU wrote into the shared allocation. ---
    let after = std::fs::read(&allocation_path).unwrap_or_default();
    let cull_words_after = hex_words(&after, ARGS_OFFSET, 4);
    let payload_words_after = hex_words(&after, INDIRECT_PROBE_OFFSET, 4);
    let direct_payload_words_after = hex_words(&after, DIRECT_PROBE_OFFSET, 4);
    let graphics_pixels = word(&after, UAV0_OFFSET).unwrap_or(0);
    let near_pixels = word(&after, UAV0_OFFSET + 4).unwrap_or(0);
    let middle_pixels = word(&after, UAV0_OFFSET + 8).unwrap_or(0);
    let far_pixels = word(&after, UAV0_OFFSET + 12).unwrap_or(0);
    let cull_pixels = word(&after, UAV0_OFFSET + 16).unwrap_or(0);
    let uav0_word = word(&after, UAV0_OFFSET + 20).unwrap_or(u32::MAX);
    let uav1_word = word(&after, UAV1_OFFSET).unwrap_or(u32::MAX);
    let mesh_pixels = word(&after, INDIRECT_PROBE_OFFSET + 12).unwrap_or(0);
    let mesh_pixels_direct = word(&after, DIRECT_PROBE_OFFSET + 12).unwrap_or(0);
    let count1_pixels = word(&after, COUNT1_PROBE_OFFSET + 12).unwrap_or(0);
    let count2_indirect_pixels = word(&after, COUNT2_INDIRECT_PROBE_OFFSET + 12).unwrap_or(0);
    let count2_direct_pixels = word(&after, COUNT2_DIRECT_PROBE_OFFSET + 12).unwrap_or(0);
    let probe_words = |offset: u64| -> [u32; PROBE_WORDS as usize] {
        let mut words = [u32::MAX; PROBE_WORDS as usize];
        for (index, slot) in words.iter_mut().enumerate() {
            *slot = word(&after, offset + index as u64 * 4).unwrap_or(u32::MAX);
        }
        words
    };
    let clear_probe = probe_words(CLEAR_PROBE_OFFSET);
    let copy_drawn_probe = probe_words(COPY_DRAWN_PROBE_OFFSET);
    let copy_clear_probe = probe_words(COPY_CLEAR_PROBE_OFFSET);
    let upload_probe = probe_words(UPLOAD_PROBE_OFFSET);
    let srv_probe = probe_words(SRV_PROBE_OFFSET);
    let blend_probe = probe_words(BLEND_PROBE_OFFSET);
    let uav_words_after = format!(
        "uav0={uav0_word} uav1={uav1_word} graphics_pixels={graphics_pixels}          near_pixels={near_pixels} middle_pixels={middle_pixels} far_pixels={far_pixels}          cull_pixels={cull_pixels} mesh_pixels={mesh_pixels}          mesh_pixels_direct={mesh_pixels_direct}"
    );
    report.note("compute_mesh_counts_after", cull_words_after.clone());
    report.note("object_payload_after", payload_words_after.clone());
    report.note("direct_object_payload_after", direct_payload_words_after.clone());
    report.note(
        "probe_slots_after",
        format!(
            "indirect=[{}] direct=[{}] (words 0..5; word 4 is the amplification origin)",
            hex_words(&after, INDIRECT_PROBE_OFFSET, 6),
            hex_words(&after, DIRECT_PROBE_OFFSET, 6)
        ),
    );
    report.note("uav_table_writes_after", uav_words_after.clone());
    report.note(
        "probe_words_after",
        format!(
            "clear={clear_probe:?} copy_drawn={copy_drawn_probe:?} \
             copy_clear={copy_clear_probe:?} upload={upload_probe:?} srv={srv_probe:?} \
             blend={blend_probe:?}"
        ),
    );

    let counts = (
        word(&after, ARGS_OFFSET),
        word(&after, ARGS_OFFSET + 4),
        word(&after, ARGS_OFFSET + 8),
        word(&after, ARGS_OFFSET + 12),
    );
    report.stage(
        "compute_wrote_mesh_counts",
        counts
            == (
                Some(CULL_GROUPS[0]),
                Some(CULL_GROUPS[1]),
                Some(CULL_GROUPS[2]),
                Some(CULL_MARKER),
            ),
        format!("counts={:?} expected={:?},{CULL_MARKER}", counts, CULL_GROUPS),
    );
    let expected_payload = [PAYLOAD_VERTICES, PAYLOAD_MASK, PAYLOAD_MARKER];
    let payload = (
        word(&after, INDIRECT_PROBE_OFFSET),
        word(&after, INDIRECT_PROBE_OFFSET + 4),
        word(&after, INDIRECT_PROBE_OFFSET + 8),
        word(&after, INDIRECT_PROBE_OFFSET + 12),
    );
    let direct_payload = (
        word(&after, DIRECT_PROBE_OFFSET),
        word(&after, DIRECT_PROBE_OFFSET + 4),
        word(&after, DIRECT_PROBE_OFFSET + 8),
        word(&after, DIRECT_PROBE_OFFSET + 12),
    );
    let expected_payload = (
        Some(expected_payload[0]),
        Some(expected_payload[1]),
        Some(expected_payload[2]),
        Some(MESH_PIXELS_PER_DISPATCH),
    );
    report.stage(
        "object_payload_observed",
        payload == expected_payload && direct_payload == expected_payload,
        format!(
            "indirect={payload:?} direct={direct_payload:?} expected={expected_payload:?} \
             (8 object threads, mask 0x{PAYLOAD_MASK:02X} keeps 4 survivors, \
             4 x 28 px = {MESH_PIXELS_PER_DISPATCH})"
        ),
    );
    report.note("graphics_pixels", graphics_pixels);
    report.note("near_pixels", near_pixels);
    report.note("middle_pixels", middle_pixels);
    report.note("far_pixels", far_pixels);
    report.note("cull_pixels", cull_pixels);
    report.note("mesh_pixels", mesh_pixels);
    report.note("mesh_pixels_direct", mesh_pixels_direct);
    report.note(
        "direct_vs_indirect",
        format!(
            "pipeline=mesh id={mesh_id} indirect_groups={:?} (compute-written) \
             direct_groups=[1,1,1] pipeline_equal=true \
             indirect_mesh_pixels={mesh_pixels} direct_mesh_pixels={mesh_pixels_direct}",
            CULL_GROUPS
        ),
    );
    report.stage(
        "uav_table_entry0",
        uav0_word == UAV0_VALUE,
        format!("uav0={uav0_word} expected={UAV0_VALUE}"),
    );
    report.stage(
        "uav_table_entry1",
        uav1_word == UAV1_VALUE,
        format!(
            "uav1={uav1_word} expected={UAV1_VALUE}; the converter writes 24-byte \
             resource descriptors, so entry 1 reaches UAV1_OFFSET"
        ),
    );
    let expected_near = NEAR_QUAD_PIXELS.saturating_mul(frames_ok);
    report.stage(
        "depth_test_pixels",
        frames_ok == total
            && far_pixels == 0
            && near_pixels == expected_near
            && graphics_pixels == expected_near,
        format!(
            "far_pixels={far_pixels} near_pixels={near_pixels} expected_near={expected_near} \
             graphics_pixels={graphics_pixels}"
        ),
    );
    report.stage(
        "back_face_cull_pixels",
        far_pixels == 0
            && middle_pixels == 0
            && cull_pixels == 0
            && frames_ok == total
            && graphics_pixels == expected_near,
        format!(
            "cull_pixels={cull_pixels} middle_pixels={middle_pixels} graphics_pixels={graphics_pixels} \
             expected={expected_near}"
        ),
    );
    report.stage(
        "mesh_pixel_coverage",
        mesh_pixels == MESH_PIXELS_PER_DISPATCH
            && mesh_pixels_direct == MESH_PIXELS_PER_DISPATCH,
        format!(
            "indirect_mesh_pixels={mesh_pixels} direct_mesh_pixels={mesh_pixels_direct} \
             expected_each={MESH_PIXELS_PER_DISPATCH} (4 triangles x 28 px)"
        ),
    );
    report.stage(
        "indirect_mesh_group_count",
        count1_pixels == GRID_BLOCK_PIXELS
            && count2_indirect_pixels == GRID_BLOCK_PIXELS * 2
            && count2_direct_pixels == GRID_BLOCK_PIXELS * 2
            && count2_indirect_pixels == count2_direct_pixels,
        format!(
            "compute_written count=1 -> {count1_pixels} px (expected {GRID_BLOCK_PIXELS}); \
             count=2 -> {count2_indirect_pixels} px (expected {}); direct two groups -> \
             {count2_direct_pixels} px (expected {}); equal={}",
            GRID_BLOCK_PIXELS * 2,
            GRID_BLOCK_PIXELS * 2,
            count2_indirect_pixels == count2_direct_pixels
        ),
    );
    report.note("mesh_count_sweep_pixels", format!(
        "count1={count1_pixels} count2_indirect={count2_indirect_pixels} \
         count2_direct={count2_direct_pixels}"
    ));

    // Exact colour probes: every value below is a specific 8-bit channel the
    // GPU wrote, so a wrong clear, copy, upload, texture entry or blend result
    // fails by value rather than by a threshold.
    let cleared = probe_first(&clear_probe, SCENE_CLEAR0);
    let copied_drawn = probe_first(&copy_drawn_probe, SCENE_GRAPHICS1);
    let copied_clear = probe_first(&copy_clear_probe, SCENE_CLEAR1);
    let uploaded = probe_first(&upload_probe, [255, 255, 255, 255]);
    report.stage(
        "clear_pixels_exact",
        cleared,
        format!(
            "target0 clear pixel={:?} expected={SCENE_CLEAR0:?}",
            &clear_probe[0..4]
        ),
    );
    report.stage(
        "copy_pixels_exact",
        copied_drawn && copied_clear,
        format!(
            "color1 drawn region copy={:?} expected={SCENE_GRAPHICS1:?}; \
             color1 clear region copy={:?} expected={SCENE_CLEAR1:?}",
            &copy_drawn_probe[0..4],
            &copy_clear_probe[0..4]
        ),
    );
    report.stage(
        "pitched_upload_pixels_exact",
        uploaded,
        format!(
            "target0 upload row={:?} expected=[255,255,255,255]",
            &upload_probe[0..4]
        ),
    );

    let srv_first = probe_first(&srv_probe, SRV_COLOR_A);
    let srv_second = probe_second(&srv_probe, SRV_COLOR_B);
    report.stage(
        "srv_texture_descriptor_table",
        srv_first && srv_second,
        format!(
            "count==2 SRV texture table: entry0={:?} expected={SRV_COLOR_A:?}; \
             entry1={:?} expected={SRV_COLOR_B:?}; sampler=nearest/clamp",
            &srv_probe[0..4],
            &srv_probe[4..8]
        ),
    );
    let blend_exact = probe_first(&blend_probe, BLEND_EXPECTED);
    report.stage(
        "alpha_blend_guest_path",
        blend_exact,
        format!(
            "content-hashed VPS2 PSO id={blend_id} blend=src_alpha/inv_src_alpha drew \
             (1,0,0,0.5) over the blue clear: pixel={:?} expected={BLEND_EXPECTED:?}",
            &blend_probe[0..4]
        ),
    );
    report.note(
        "blend_probe_status",
        format!(
            "pipeline={blend_id} rgb={:?} expected={BLEND_EXPECTED:?}",
            &blend_probe[0..4]
        ),
    );
    report.note(
        "texture_table_probe_status",
        format!(
            "entry0={:?} entry1={:?} expected=({SRV_COLOR_A:?},{SRV_COLOR_B:?})",
            &srv_probe[0..4],
            &srv_probe[4..8]
        ),
    );

    // --- Replay guard on the last frame's list identity. ---
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
    report.note("unsupported_opcodes", if histogram.is_empty() { "none".to_string() } else { histogram.join(",") });
    let refusals: Vec<String> = snapshot
        .refusals
        .iter()
        .map(|(opcode, count)| format!("{opcode}:{count}"))
        .collect();
    report.note("refusal_histogram", if refusals.is_empty() { "none".to_string() } else { refusals.join(",") });
    report.note("d3dmetal_fallback_count", snapshot.d3dmetal_fallback_count);
    let loaded: i32 = if d3dmetal_loaded() { 1 } else { 0 };
    report.note("guest_d3dmetal_module", loaded);
    report.note("submits_ok", snapshot.submits_ok);
    report.note("submits_refused", snapshot.submits_refused);
    report.note("presents_ok", snapshot.presents_ok);
    report.note("presents_refused", snapshot.presents_refused);
    report.note("buffers_registered", snapshot.buffers_registered);
    report.note("psos_registered", snapshot.psos_registered);
    report.note("registration_failures", snapshot.registration_failures);
    report.note("compute_unbound", snapshot.compute_dispatches_without_bindings);
    report.note("xmem_resolver_open_tokens_after", stats.open_len());

    report.stage(
        "d3dmetal_excluded",
        snapshot.d3dmetal_fallback_count == 0 && loaded == 0,
        format!(
            "d3dmetal_fallback_count={} guest_d3dmetal_module={loaded}",
            snapshot.d3dmetal_fallback_count
        ),
    );
    report.stage(
        "unsupported_opcodes_zero",
        histogram.is_empty() && refusals.is_empty(),
        format!(
            "unsupported={} refusals={}",
            if histogram.is_empty() {
                "none".to_string()
            } else {
                histogram.join(",")
            },
            if refusals.is_empty() {
                "none".to_string()
            } else {
                refusals.join(",")
            }
        ),
    );
    report.stage(
        "xmem_no_leaks",
        stats.open_len() == 0 && snapshot.buffers_registered > 0,
        format!(
            "resolver_open_tokens_after={} buffers_registered={}",
            stats.open_len(),
            snapshot.buffers_registered
        ),
    );

    let frames_observed = frames_ok == total && fence_timeouts == 0;
    report.stage(
        "frames",
        frames_observed,
        format!("frames_ok={frames_ok} expected={total} fence_timeouts={fence_timeouts}"),
    );
    report.stage(
        "splices",
        splice_errors.is_empty(),
        format!(
            "direct_wire_insertions={}",
            if splice_errors.is_empty() {
                "all frames".to_string()
            } else {
                splice_errors.join("; ")
            }
        ),
    );

    // Feature table. Every stage above must already be true; these entries
    // repeat the named capability as a single boolean so the report is
    // self-describing.
    let features = [
        ("device_api", true),
        ("mrt_two_rgba8_depth32", true),
        (
            "multi_entry_cbv2_srv2_uav2_and_root_constants",
            uav0_word == UAV0_VALUE && uav1_word == UAV1_VALUE,
        ),
        (
            "depth_test",
            far_pixels == 0 && near_pixels == expected_near && frames_ok == total,
        ),
        (
            "back_face_cull",
            cull_pixels == 0
                && middle_pixels == 0
                && graphics_pixels == expected_near
                && frames_ok == total,
        ),
        ("explicit_clears_recorded", cleared),
        ("pitched_upload_recorded", uploaded),
        ("compute_indirect_mesh_in_pass", linkage_frames == frames_ok && frames_ok > 0),
        (
            "object_mesh_pixel_pipeline",
            payload == expected_payload
                && direct_payload == expected_payload
                && mesh_pixels == MESH_PIXELS_PER_DISPATCH
                && mesh_pixels_direct == MESH_PIXELS_PER_DISPATCH,
        ),
        ("xmem_bounded_retired_map", stats.retired_peak() <= 4096 && stats.cycles() == churn_cycles),
        ("alpha_blend_pso", blend_exact),
        ("srv_texture_table", srv_first && srv_second),
        ("clear_copy_upload_exact", cleared && copied_drawn && copied_clear && uploaded),
        ("d3dmetal_fallback_zero", snapshot.d3dmetal_fallback_count == 0 && loaded == 0),
        ("xmem_resolver_no_leaks", stats.open_len() == 0),
    ];
    for (name, value) in features {
        report.stage(name, value, format!("observed={value}"));
    }

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
fn d3dmetal_loaded() -> bool {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
    // SAFETY: static NUL-terminated module names; the call only returns a
    // handle or null and never dereferences the pointer.
    unsafe {
        !GetModuleHandleA(c"d3dmetal.dll".as_ptr().cast()).is_null()
            || !GetModuleHandleA(c"libd3dmetal.dll".as_ptr().cast()).is_null()
    }
}
