//! ADR-067 hardware proof: a compute PSO's `[numthreads]` reaches its dispatch
//! through the production registration shape.
//!
//! The test builds a repo-owned synthetic `DXBC` container whose `PSV0` runtime
//! info declares a compute stage with `[numthreads(64, 1, 1)]` (no title bytes,
//! paths or names), reads the size with the same production parser the host
//! uses (`registry::compute_threadgroup_from_dxil_container`), registers a real
//! `MTLComputePipelineState` from a compiled MSL kernel through
//! `DirectMetalTransport::register_content_hashed_compute_pipeline_stages`, and
//! dispatches one group of those threads with the *parsed* size — the test never
//! calls `set_compute_threadgroup_size`.
//!
//! The kernel writes `out[tid] = tid * 3 + 7` for 64 threads. The test asserts
//! the exact 64 words and that the host encode counter reports exactly one
//! dispatch. It also proves the registry stores the size on the pipeline entry
//! (`PipelineEntry::compute_threadgroup`).
//!
//! Run directly (the binary self-acquires `/tmp/victory-gpu.lock` through the
//! shared GPU harness when present):
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-metal --target aarch64-apple-darwin \
//!   --test compute_threadgroup_prod_gpu -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_metal::MetalTransport;
use victory_metal::VmBufferId;
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    compute_threadgroup_from_dxil_container, encode_compute_threadgroup_state, FunctionRole,
    PipelineKey, PipelineRegistry, StageLibrary,
};
use victory_metal::wire::{BufferBinding, Dispatch3d};
use victory_metal::wire_v3::{CommandV3, PipelineRef, SubmissionV3};

/// Threads the kernel runs: one group of the declared 64 threads.
const THREADS: u32 = 64;
/// Payload words the kernel writes.
const WORDS: usize = THREADS as usize;
/// Payload bytes the guest allocation binds.
const PAYLOAD_BYTES: u64 = (WORDS * 4) as u64;
/// Page size of the zero-copy allocation (Metal requires a page multiple).
const PAGE: usize = 16 * 1024;
/// Host pipeline handle the wire `Dispatch3d` names.
const HANDLE: u64 = 0xC20;
/// Allocation identity and generation used by the submission.
const ALLOCATION: u64 = 1;
const GENERATION: u32 = 1;

/// The kernel a real Metal compiler turns into the registered pipeline.
const COMPUTE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void write_words(device uint *out [[buffer(0)]],
                        uint tid [[thread_position_in_grid]]) {
    out[tid] = tid * 3u + 7u;
}
"#;

fn tool_path(name: &str) -> Option<PathBuf> {
    let output = Command::new("xcrun").args(["-f", name]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path =
            std::env::temp_dir().join(format!("victory-{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Compiles `source` into a real `.metallib`; `None` when the toolchain is
/// unavailable (the test then reports a skip instead of a failure).
fn compile_metallib(dir: &Path, stem: &str, source_text: &str) -> Option<Vec<u8>> {
    let metal = tool_path("metal")?;
    let metallib = tool_path("metallib")?;
    let source = dir.join(format!("{stem}.metal"));
    let air = dir.join(format!("{stem}.air"));
    let library = dir.join(format!("{stem}.metallib"));
    fs::write(&source, source_text).expect("write MSL source");
    let status = Command::new(metal)
        .args(["-c", "-o"])
        .arg(&air)
        .arg(&source)
        .status()
        .expect("run metal compiler");
    assert!(status.success(), "metal compiler failed");
    let status = Command::new(metallib)
        .args(["-o"])
        .arg(&library)
        .arg(&air)
        .status()
        .expect("run metallib linker");
    assert!(status.success(), "metallib linker failed");
    Some(fs::read(&library).expect("read metallib"))
}

/// A `DXBC` container with one `PSV0` part (the metadata shape a translated
/// compute container carries). The body is the runtime-info block both sides
/// parse: size, then `NumThreadsX/Y/Z` at offsets 36/40/44.
fn compute_container(threads: [u32; 3]) -> Vec<u8> {
    const HEADER: usize = 0x20;
    let mut psv = Vec::new();
    psv.extend_from_slice(&48u32.to_le_bytes());
    let mut info = [0u8; 48];
    info[24] = 5; // D3D12_SHADER_VERSION_TYPE_CS
    for (index, value) in threads.into_iter().enumerate() {
        info[36 + 4 * index..40 + 4 * index].copy_from_slice(&value.to_le_bytes());
    }
    psv.extend_from_slice(&info);

    let table = HEADER + 4;
    let total = table + 8 + psv.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(table as u32).to_le_bytes());
    out.extend_from_slice(b"PSV0");
    out.extend_from_slice(&(psv.len() as u32).to_le_bytes());
    out.extend_from_slice(&psv);
    out
}

unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn free(ptr: *mut c_void);
}

/// Page-aligned host memory registered zero-copy with Metal.
struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn zeroed(len: usize) -> Self {
        assert!(len.is_multiple_of(PAGE));
        let mut ptr: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes a page-aligned `len`-byte allocation.
        let rc = unsafe { posix_memalign(&mut ptr, PAGE, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!ptr.is_null(), "posix_memalign returned null");
        // SAFETY: `ptr` is a live `len`-byte allocation.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, len) };
        Self { ptr, len }
    }

    fn words(&self) -> &[u32] {
        // SAFETY: `ptr` is a live allocation of `len` bytes, a multiple of 4.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u32>(), self.len / 4) }
    }

    fn register(&self, transport: &DirectMetalTransport) -> VmBufferId {
        // SAFETY: the page stays live until after `wait_for_gpu` and the
        // transport is dropped, the documented zero-copy registration contract.
        unsafe { transport.register_zerocopy(self.ptr.cast::<u8>(), self.len) }
            .expect("zero-copy registration should succeed")
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `posix_memalign` and is freed exactly once.
        unsafe { free(self.ptr) };
    }
}

#[test]
fn a_registered_compute_pso_dispatches_with_its_parsed_threadgroup() {
    let Ok(transport) = DirectMetalTransport::new() else {
        eprintln!("skipping: no Metal device on this host");
        return;
    };
    let dir = TempDir::new("compute-threadgroup-prod");
    let Some(metallib) = compile_metallib(dir.path(), "write_words", COMPUTE_MSL) else {
        eprintln!("skipping: the Metal shader toolchain is unavailable");
        return;
    };

    // 1. Production parse: the container declares 64x1x1, not a default.
    let declared = [64u32, 1, 1];
    let container = compute_container(declared);
    let parsed = compute_threadgroup_from_dxil_container(&container)
        .expect("the compute container declares its threadgroup");
    assert_eq!(parsed, declared, "the parsed [numthreads] triple");
    let state = encode_compute_threadgroup_state(parsed);
    assert_eq!(state.len(), 11, "the compute state chunk is carried");

    // 2. Production registration: the host builds the real
    //    `MTLComputePipelineState` from the stage library and state.
    let stages = [StageLibrary {
        role: FunctionRole::Kernel,
        metallib: metallib.as_slice(),
        entry: "write_words",
    }];
    transport
        .register_content_hashed_compute_pipeline_stages(
            HANDLE,
            &stages,
            &state,
            PipelineAbi::default(),
        )
        .expect("the compute PSO registers");

    // 3. The registry stores the declared size on the pipeline entry.
    let mut registry = PipelineRegistry::new(&transport).expect("registry");
    let key = PipelineKey::from_stages(&stages, victory_metal::macos::registry::PipelineStage::Compute, &state);
    let id = registry
        .register_compute_stages(key, &stages, &state)
        .expect("staged compute registration");
    assert_eq!(
        registry.get(id).and_then(|entry| entry.compute_threadgroup()),
        Some(declared),
        "the registered pipeline carries its threadgroup size"
    );

    // 4. Dispatch with the parsed size (no manual threadgroup call anywhere).
    let page = HostPage::zeroed(PAGE);
    let buffer = page.register(&transport);
    let mut allocations = HashMap::new();
    allocations.insert((ALLOCATION, GENERATION), buffer);

    let counters_before = transport.encode_counters();
    assert_eq!(counters_before.dispatches, 0);
    let submission = SubmissionV3 {
        sequence: 1,
        list: None,
        commands: vec![
            CommandV3::SetComputePipeline(PipelineRef {
                pipeline_id: HANDLE,
            }),
            CommandV3::Dispatch3d(Dispatch3d {
                pipeline_id: HANDLE,
                group_counts: [1, 1, 1],
                threads_per_group: parsed,
                bindings: vec![BufferBinding {
                    allocation_id: ALLOCATION,
                    generation: GENERATION,
                    offset: 0,
                    length: PAYLOAD_BYTES,
                }],
            }),
        ],
    };
    let reply = transport
        .submit_v3(&submission, &allocations)
        .expect("a valid compute submission is accepted");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    transport.wait_for_gpu().expect("the GPU drains");

    // 5. Exact output: 64 words, each `tid * 3 + 7`.
    let words = page.words();
    for (index, word) in words.iter().take(WORDS).enumerate() {
        let expected = index as u32 * 3 + 7;
        assert_eq!(*word, expected, "word {index}");
    }
    assert!(
        words[WORDS..].iter().all(|word| *word == 0),
        "nothing past the 64 threads was written"
    );

    let counters_after = transport.encode_counters();
    assert_eq!(
        counters_after.dispatches,
        counters_before.dispatches + 1,
        "the host encoded exactly one dispatch"
    );
    eprintln!(
        "compute threadgroup proof: parsed={parsed:?} words={WORDS} \
         dispatches={} (before {})",
        counters_after.dispatches, counters_before.dispatches
    );
}
