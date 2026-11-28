//! ADR-067 regression proof on real Metal: a compute pipeline whose container
//! has no threadgroup metadata still registers, and the guest producer refuses
//! its `Dispatch` by name instead of guessing a size.
//!
//! The host half registers a real `MTLComputePipelineState` through the
//! production device path with an empty compute state (the exact state a
//! container without `PSV0`/`VCTG` metadata carries). The guest half binds that
//! host pipeline id in a `MetalProducer`; because no threadgroup size was
//! recorded, `dispatch` must refuse with `Dispatch: compute threadgroup size
//! not set`, and the guest dispatch census must show one attempt, zero encoded
//! and one refusal.
//!
//! The binary uses a repo-owned MSL kernel and a synthetic empty state: no
//! title bytes, paths, shader dumps or names.
//!
//! Run directly (the binary self-acquires `/tmp/victory-gpu.lock`):
//! ```text
//! export PATH="/opt/homebrew/opt/rustup/bin:$PATH";
//! cargo test --locked -p victory-gpu --target aarch64-apple-darwin \
//!   --test compute_threadgroup_missing_gpu -- --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use victory_gpu::native_metal::producer::MetalProducer;
use victory_gpu::native_metal::telemetry::compute_dispatch_counts;
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    FunctionRole, PipelineKey, PipelineRegistry, StageLibrary,
};

/// Host pipeline handle the producer binds.
const HANDLE: u64 = 0xC2B;

const COMPUTE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void write_words(device uint *out [[buffer(0)]],
                        uint tid [[thread_position_in_grid]]) {
    out[tid] = 1u;
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

#[test]
fn a_registered_compute_pso_without_metadata_refuses_dispatch_by_name() {
    let Ok(transport) = DirectMetalTransport::new() else {
        eprintln!("skipping: no Metal device on this host");
        return;
    };
    let dir = TempDir::new("compute-threadgroup-missing-prod");
    let Some(metallib) = compile_metallib(dir.path(), "write_words", COMPUTE_MSL) else {
        eprintln!("skipping: the Metal shader toolchain is unavailable");
        return;
    };
    let stages = [StageLibrary {
        role: FunctionRole::Kernel,
        metallib: metallib.as_slice(),
        entry: "write_words",
    }];
    let state: &[u8] = &[];

    // 1. Host: the production device path registers the PSO with no size.
    transport
        .register_content_hashed_compute_pipeline_stages(
            HANDLE,
            &stages,
            state,
            PipelineAbi::default(),
        )
        .expect("a compute PSO without threadgroup metadata registers");

    // 2. Host registry: the missing size is `None`, not a refusal.
    let mut registry = PipelineRegistry::new(&transport).expect("registry");
    let key = PipelineKey::from_stages(
        &stages,
        victory_metal::macos::registry::PipelineStage::Compute,
        state,
    );
    let id = registry
        .register_compute_stages(key, &stages, state)
        .expect("the registry accepts a pipeline with no threadgroup record");
    assert_eq!(
        registry.get(id).and_then(|entry| entry.compute_threadgroup()),
        None,
        "the registered pipeline reports a missing size"
    );

    // 3. Guest: binding the host id with no recorded size must refuse the
    //    dispatch by name, and the census must show attempt/refusal.
    let before = compute_dispatch_counts();
    let mut producer = MetalProducer::new(0xC2B, 1).expect("producer");
    producer
        .set_compute_pipeline(HANDLE)
        .expect("the pipeline id is valid");
    let error = producer
        .dispatch(1, 1, 1)
        .expect_err("a dispatch with no known size is refused");
    assert!(
        error.to_string().contains("compute threadgroup size not set"),
        "the refusal names the missing size: {error}"
    );
    let after = compute_dispatch_counts();
    assert_eq!(after.0 - before.0, 1, "one dispatch attempt");
    assert_eq!(after.1 - before.1, 0, "nothing encoded");
    assert_eq!(after.2 - before.2, 1, "one named refusal");

    eprintln!(
        "compute threadgroup missing proof: registry=None dispatch_refusal=`{error}`"
    );
}
