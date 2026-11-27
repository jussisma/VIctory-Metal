//! Production-path gate for the Wire v3 producer pipeline.
//!
//! This test is the host half of the `PRODUCER-E2E-GATE` task. It exercises the
//! real pieces that exist today, in the order a closed title list would use
//! them:
//!
//! 1. A version 3 submission is framed by [`victory_metal::wire_v3`] and sent
//!    over the loopback IPC envelope (`KIND_SUBMISSION_V3`) to a real
//!    [`victory_metal::ipc_host::IpcHost`] serving a real
//!    [`victory_metal::DirectMetalTransport`]. The allocation is a real
//!    file-backed guest allocation in the session share directory; the compute
//!    result is read back from that file, exactly the bytes the guest maps.
//! 2. A second submission replays the same list identity; the host refuses it
//!    with [`ReplyStatus::AlreadyExecuted`] and the allocation bytes are
//!    byte-for-byte unchanged.
//! 3. An unknown allocation and an unknown opcode each produce an explicit
//!    failure reply ([`ReplyStatus::UnknownHandle`] /
//!    [`ReplyStatus::ExecutionFailed`]); neither falls back to D3D12/D3DMetal.
//! 4. The graphics half of the gate runs the MRT + depth path through
//!    `DirectMetalTransport::submit_v3` on one command buffer, presents the
//!    drawn IOSurface through the host present path, and asserts exact pixels
//!    for two colour attachments including the depth-tested centre.
//! 5. The compute shader that writes the second allocation page is a real
//!    Rosetta-converted DXIL shader: an x86_64 `victory-irconverter` worker
//!    (`VICTORY_IRCONVERTER_WORKER`) translates synthetic DXIL through
//!    `libmetalirconverter.dylib`, and the ARM64 host dispatches the converted
//!    metallib through the verified MSC single-UAV argument buffer.
//!
//! # What this gate does not cover (reported, never claimed)
//!
//! The Win64 `victory-gpu` producer and its `V3Sink` are not part of this
//! binary: `victory-gpu` depends on `victory-metal`, so a `victory-metal`
//! integration test cannot link it, and the producer queue currently installs
//! `DisconnectedSink`. Likewise `GuestIpcClient` has no version 3 encoder, so no
//! Wine guest can carry a v3 frame through the production client yet. The test
//! therefore drives the same framed bytes a future sink must carry.
//!
//! The IPC host also cannot resolve render-target textures: `KIND_REGISTER`
//! only maps file-backed buffers, so a v3 `BeginRenderPass` whose attachments
//! name host textures is refused `UnknownHandle` before GPU work. The graphics
//! submission is therefore executed through `submit_v3` directly on the same
//! transport, not through the wire. Both gaps are reported by the task, not
//! papered over here.
//!
//! # D3DMetal
//!
//! The gate process scans its own Mach-O image list (`_dyld_image_count` /
//! `_dyld_get_image_name`) for any `D3DMetal` image, and `otool -L` is run on
//! the x86_64 converter worker. The Wine guest process is not launched by this
//! test; that image check is reported as skipped.
//!
//! Run with:
//!
//! ```sh
//! cargo build -p victory-metal --bin victory-irconverter --target x86_64-apple-darwin
//! VICTORY_IRCONVERTER_WORKER=target/x86_64-apple-darwin/debug/victory-irconverter \
//! VICTORY_METAL_IRCONVERTER=/path/libmetalirconverter.dylib \
//! cargo test -p victory-metal --target aarch64-apple-darwin --test producer_e2e \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use victory_metal::abi::VmBufferId;
use victory_metal::fence_page::{FENCE_PAGE_LEN, FencePageMapping};
use victory_metal::ipc_host::{
    BatchExecutor, FENCE_PAGE_FILE_NAME, FENCE_PAGE_REGISTER_PAYLOAD_LEN, HostSecret, IpcHost,
    KIND_ACK, KIND_AUTH, KIND_FENCE_PAGE, KIND_REGISTER, KIND_SUBMISSION_V3, REGISTER_PAYLOAD_LEN,
    V3_REQUEST_HEADER_LEN, V3_REQUEST_MAGIC,
};
use victory_metal::macos::cache::TranslationOptions;
use victory_metal::macos::irconverter::{ProcessWorker, translate_dxil_with};
use victory_metal::macos::msc::{MscUavDispatch, MscUavLayout};
use victory_metal::macos::present::{PresentPath, RecordingSink, SurfaceImage};
use victory_metal::macos::render::{GraphicsPipeline, PixelFormat, RenderTarget};
use victory_metal::transport::MetalError;
use victory_metal::wire::{self, BufferBinding, ListIdentity, Reply, ReplyStatus};
use victory_metal::wire_v3::{
    self, BeginRenderPass, ColorAttachment, CommandV3, DepthStencilAttachment, DrawIndexedState,
    DrawState, IndexBufferBinding, IndexFormat, LoadOp, PROTOCOL_VERSION_V3, PipelineRef, Present,
    PrimitiveTopology, RootConstantsV3, StoreOp, SubmissionV3, VertexBuffer, VertexBufferSet,
};
use victory_metal::{DirectMetalTransport, MetalTransport};

/// 64 KiB guest page, matching the xmem share provider.
const PAGE: usize = 64 * 1024;
/// Three guest pages: compute input/output, graphics geometry, converted-DXIL output.
const FILE_LEN: usize = 3 * PAGE;
/// The single file-backed allocation the guest registers.
const ALLOCATION_ID: u64 = 1;
/// Generation the IPC bridge publishes for a registered allocation.
const GENERATION: u32 = 1;
/// Edge of the two MRT colour targets and the depth target.
const EDGE: u32 = 32;
/// Threads the published MSL compute kernel processes.
const COMPUTE_THREADS: u32 = 64;
/// Payload bytes the MSL kernel reads and writes (64 dwords).
const COMPUTE_BYTES: u64 = COMPUTE_THREADS as u64 * 4;
/// Synthetic DXIL: `RootFlags(0), UAV(u0)` with `gOut[tid] = tid * 3 + 7`.
const DXIL_HEX: &str = concat!(
    "4458424361207e033dba9d6ccb9b5b1a9f973eed010000007c0b00000800000040000000500000006000000070000000",
    "e0000000180100005c060000780600005346493008000000000000000000000049534731080000000000000008000000",
    "4f5347310800000000000000080000005053563068000000340000000000000000000000000000000000000000000000",
    "ffffffff0500000000000000000000004000000001000000010000000100000001000000180000000800000000000000",
    "00000000000000000c0000000000000008000000006d61696e0000000000000052545330300000000200000001000000",
    "18000000000000003000000000000000040000000000000024000000000000000000000000000000535441543c050000",
    "600005004f0100004458494c0001000010000000240500004243c0de210c0000460100000b8220000200000013000000",
    "0781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420",
    "434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d",
    "86f0ffffffff032001d50662f8ffffffff019000491800000300000013826042204c0806000000008920000025000000",
    "32224809206485049322a484049322e384a19014124c8a8c0b84a44c10482300250014e608c0a00c630c22730408997b",
    "86cb9fb08790fc1068868540c19923088a81861923912a0a18688c31c630c4060266328371608770988779700359b885",
    "59a00779a88771a0877a908772200751a8077330877290073ea8077798877438077728077200837470077af003140c7a",
    "7304a000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0",
    "077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e610",
    "0776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e",
    "0008000000000000000000863c04100001000000000000000c79142000040000000000000018f23040000c0000000000",
    "00009005020000000c000000321e981419114c908c092647c604431a253002500c85510865500ea53003508001454069",
    "04806281109c01a037030000791800005b0000001a034c90460213c48e0c6feced4d0c24c625c745c60626c7a5060405",
    "2646cc46268c8ce6266543104c1006628230141b8481d820100485b1b909c2606c180e849820401189b3a73aba09c270",
    "4c1092678230201b84c1d9b010ca421003d334cdb32180361040040013844762803641189209c2a04c10866583815015",
    "615d138481d92010d98640db2010db86c1c1b80d033175130401d8006c18083000830d41186c18864f0c48b485a5b94d",
    "1022688230341b063318860d045106ce196c283e320024346091e63647374722cd8d6e6e8230384ce8caf0bee6e8dee4",
    "ca36186a80ac011bb4c150858dcdaecd258daccc8d6e4a105421c373b12b939b4b7b739b12104dc8f05cecc2d8eccae4",
    "a604451d323c9739b430b232b9a637b232b6290152860ccf45ae6ceead4e6eac6c6e4a105522c373a1cb832b0b72737b",
    "a30ba34b7b739b9b22746250870ccfa5cc8d4e2e0fea2dcd8d6e6e4a80065dc8f05cc6deeadce8cae4e6a6046d000000",
    "791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421e",
    "c2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a70037678877020",
    "8719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc83",
    "3c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298",
    "877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c843",
    "39984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8",
    "e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000e00000036500d97ef3c7e40150511b19313113e",
    "72db26200d97ef3cbe1011c0448440332c840548c3e53b8f3f1d11010ce2e023b76d000403200d000000000048415348",
    "1400000000000000d58deaf61d33caaf2e2a8053a1d91dd64458494cfc040000600005003f0100004458494c00010000",
    "10000000e40400004243c0de210c0000360100000b82200002000000130000000781239141c80449061032399201840c",
    "250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c450",
    "4151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffff",
    "ff019000491800000300000013826042204c080600000000892000002500000032224809206485049322a484049322e3",
    "84a19014124c8a8c0b84a44c10482300250014e608c0a00c630c22730408997b86cb9fb08790fc1068868540c1992308",
    "8a81861923912a0a18688c31c630c4060266328371608770988779700359b88559a00779a88771a0877a908772200751",
    "a8077330877290073ea8077798877438077728077200837470077af003140c7a7304a000131472c08774608736688779",
    "680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e",
    "7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a3007",
    "72d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c041000",
    "01000000000000000c79142000040000000000000018f23040000c000000000000009005020000000a000000321e9810",
    "19114c908c092647c604431a2530025012c5501885508001944600281608c1190000000079180000390000001a034c90",
    "460213c48e0c6feced4d0c24c625c745c60626c7a50604052646cc46268c8ce6266543104c1006628230141b84819820",
    "0cc6066130288ccd4d108663c38024c404016a084c10066482902c138421d9200ccf8685581882181ac771a00d41b481",
    "0024009820080089b6b034b70942c44c100665c3700dc30682b01e6c434155c094556163b36b7349232b73a39b120455",
    "c8f05ceccae4e6d2dedca6044413323c17bb3036bb32b9298151870ccf650e2d8cac4caee98dac8c6d4a909421c37391",
    "2b9b7bab931b2b9b9b124875c8f05ccadce8e4f2a0ded2dce8e6a60419000000791800004c0000003308801cc4e11c66",
    "14013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831b",
    "cc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f0",
    "0ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b6807376887",
    "72680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec30",
    "0362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b",
    "94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee190",
    "0fef500ff4000000712000000e00000036500d97ef3c7e40150511b19313113e72db26200d97ef3cbe1011c044844033",
    "2c840548c3e53b8f3f1d11010ce2e023b76d000403200d0061200000120000001304412c10000000040000003466004a",
    "ae7403caa01c4a31804c0900230609008260a04c460041ca88810180201810951155604805068c1834000882016319c4",
    "4005cbb21408000000000000",
);

/// MSL used for the graphics pass. `SetRootConstants` binds vertex bytes in the
/// host's v3 render path, so the fragment colours travel through `VOut`.
const GRAPHICS_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    struct VOut {
        float4 position [[position]];
        float4 color0 [[flat]];
        float4 color1 [[flat]];
    };

    vertex VOut pe_vertex(uint vid [[vertex_id]],
                          constant float4 *positions [[buffer(0)]],
                          constant float4 &color0 [[buffer(1)]],
                          constant float4 &color1 [[buffer(2)]]) {
        VOut out;
        out.position = positions[vid];
        out.color0 = color0;
        out.color1 = color1;
        return out;
    }

    struct MrtOut {
        float4 color0 [[color(0)]];
        float4 color1 [[color(1)]];
    };

    fragment MrtOut pe_fragment(VOut in [[stage_in]]) {
        MrtOut out;
        out.color0 = in.color0;
        out.color1 = in.color1;
        return out;
    }
"#;

/// MSL compute kernel published by the native host for the IPC half. It is the
/// ordinary argument-binding shape (`device uint *[[buffer(0)]]`), which is what
/// `Dispatch1d` binds; it is deliberately not the converted root-signature ABI.
const COMPUTE_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void pe_alias_mul_add(device uint *data [[buffer(0)]],
                                 uint tid [[thread_position_in_grid]]) {
        data[tid] = data[tid] * 3u + 7u;
    }
"#;

/// `IOSurfaceLock` read-only option.
const IOSURFACE_LOCK_READ_ONLY: u32 = 1;
/// `PROT_READ`.
const PROT_READ: i32 = 1;
/// `PROT_WRITE`.
const PROT_WRITE: i32 = 2;
/// `MAP_SHARED`.
const MAP_SHARED: i32 = 1;

/// Bgra8 bytes for an opaque colour given as RGBA floats.
fn bgra(rgba: [f32; 4]) -> [u8; 4] {
    [
        (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

const RED_RGBA: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const CYAN_RGBA: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
const GREEN_RGBA: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const MAGENTA_RGBA: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const BLUE_RGBA: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const WHITE_RGBA: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

fn float4_bytes(rgba: [f32; 4]) -> Vec<u8> {
    rgba.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    assert!(hex.len() % 2 == 0, "hex fixture has an odd length");
    (0..hex.len() / 2)
        .map(|index| {
            u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).expect("hex fixture digit")
        })
        .collect()
}

fn workspace_root() -> Option<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

/// The x86_64 Rosetta converter worker and the x86_64 IRConverter library.
fn converter_paths() -> Result<(PathBuf, PathBuf), String> {
    let worker = std::env::var_os("VICTORY_IRCONVERTER_WORKER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let path =
                workspace_root()?.join("target/x86_64-apple-darwin/debug/victory-irconverter");
            path.is_file().then_some(path)
        })
        .ok_or_else(|| {
            "no x86_64 victory-irconverter worker (set VICTORY_IRCONVERTER_WORKER or build \
             `cargo build -p victory-metal --bin victory-irconverter \
             --target x86_64-apple-darwin`)"
                .to_string()
        })?;
    let library = std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                .filter(|path| path.is_file())
        })
        .ok_or_else(|| {
            "no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES)"
                .to_string()
        })?;
    Ok((worker, library))
}

/// A private session directory removed on drop.
struct SessionDir(PathBuf);

impl SessionDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "victory-producer-e2e-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir(&path).expect("create session directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn allocation_path(&self) -> PathBuf {
        self.0.join(format!("{ALLOCATION_ID:016x}.bin"))
    }
}

impl Drop for SessionDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A page-aligned shared mapping of the guest allocation file.
struct SharedMapping {
    ptr: *mut c_void,
    len: usize,
}

impl SharedMapping {
    fn open(path: &Path, len: usize) -> Self {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("open guest allocation");
        use std::os::fd::AsRawFd;
        // SAFETY: `len` is the file's exact length and the fd stays open for the
        // duration of the call; MAP_SHARED makes GPU writes visible in the file.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert!(
            !ptr.is_null() && ptr as isize != -1,
            "mmap of the guest allocation failed"
        );
        Self { ptr, len }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` maps exactly `len` writable bytes for this guard's life.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for SharedMapping {
    fn drop(&mut self) {
        // SAFETY: Unmaps the exact region created by `SharedMapping::open`.
        unsafe {
            munmap(self.ptr, self.len);
        }
    }
}

/// Forwards the real transport to the IPC host while the test keeps its own
/// `Arc` for direct v3 submissions.
struct SharedExecutor(Arc<DirectMetalTransport>);

impl BatchExecutor for SharedExecutor {
    fn page_size(&self) -> Result<u64, MetalError> {
        BatchExecutor::page_size(&*self.0)
    }

    fn register_window(&self, ptr: *mut u8, len: usize) -> Result<VmBufferId, MetalError> {
        BatchExecutor::register_window(&*self.0, ptr, len)
    }

    fn unregister_window(&self, id: VmBufferId) -> Result<(), MetalError> {
        BatchExecutor::unregister_window(&*self.0, id)
    }

    fn submit_batch(
        &self,
        submission: &victory_metal::wire::Submission,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        BatchExecutor::submit_batch(&*self.0, submission, allocations)
    }

    fn submit_v3(
        &self,
        submission: &SubmissionV3,
        allocations: &HashMap<(u64, u32), VmBufferId>,
    ) -> Result<Reply, MetalError> {
        BatchExecutor::submit_v3(&*self.0, submission, allocations)
    }

    fn load_pipeline(
        &self,
        metallib: &[u8],
        entry: &str,
    ) -> Result<victory_metal::VmPipelineId, MetalError> {
        BatchExecutor::load_pipeline(&*self.0, metallib, entry)
    }

    fn attach_fence_page(&self, page: Arc<FencePageMapping>) -> Result<bool, MetalError> {
        // The wrapper must forward this: the default would report a synchronous
        // executor and make the host publish completion itself, which is exactly
        // the ADR-017 defect.
        BatchExecutor::attach_fence_page(&*self.0, page)
    }

    fn detach_fence_page(&self) {
        BatchExecutor::detach_fence_page(&*self.0)
    }
}

/// Raw TCP v3 client: the same framing the production sink must carry.
struct V3Client {
    stream: TcpStream,
}

impl V3Client {
    fn connect(addr: std::net::SocketAddr, secret: &[u8; 32]) -> Self {
        let stream = TcpStream::connect(addr).expect("connect to the native host");
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .expect("client read timeout");
        let mut client = Self { stream };
        client
            .write_frame(KIND_AUTH, secret)
            .expect("send authentication");
        assert_eq!(
            client.read_frame().expect("read authentication ack"),
            Some((KIND_ACK, vec![0])),
            "host did not acknowledge authentication"
        );
        client
    }

    fn register(&mut self, id: u64, len: u64) {
        let mut payload = [0u8; REGISTER_PAYLOAD_LEN];
        payload[..8].copy_from_slice(&id.to_le_bytes());
        payload[8..].copy_from_slice(&len.to_le_bytes());
        self.write_frame(KIND_REGISTER, &payload)
            .expect("send allocation registration");
        assert_eq!(
            self.read_frame().expect("read registration ack"),
            Some((KIND_ACK, vec![0])),
            "host did not acknowledge registration"
        );
    }

    /// Registers the shared completion page (kind 6): one `byte_len` word.
    fn register_fence_page(&mut self) {
        let mut payload = [0u8; FENCE_PAGE_REGISTER_PAYLOAD_LEN];
        payload.copy_from_slice(&(FENCE_PAGE_LEN as u64).to_le_bytes());
        self.write_frame(KIND_FENCE_PAGE, &payload)
            .expect("send fence page registration");
        assert_eq!(
            self.read_frame().expect("read fence page ack"),
            Some((KIND_ACK, vec![0])),
            "host did not acknowledge the fence page"
        );
    }

    fn write_frame(&mut self, kind: u8, payload: &[u8]) -> std::io::Result<()> {
        let mut header = [0u8; 5];
        header[0] = kind;
        header[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        self.stream.write_all(&header)?;
        self.stream.write_all(payload)?;
        self.stream.flush()
    }

    fn read_frame(&mut self) -> std::io::Result<Option<(u8, Vec<u8>)>> {
        let mut header = [0u8; 5];
        let mut filled = 0usize;
        while filled < header.len() {
            match self.stream.read(&mut header[filled..]) {
                Ok(0) if filled == 0 => return Ok(None),
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "closed inside the IPC envelope",
                    ));
                }
                Ok(read) => filled += read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload)?;
        Ok(Some((header[0], payload)))
    }

    /// Sends one encoded v3 frame inside the host's versioned envelope.
    fn submit_v3(&mut self, sequence: u32, frame: &[u8]) -> Reply {
        let mut payload = Vec::with_capacity(V3_REQUEST_HEADER_LEN + frame.len());
        payload.extend_from_slice(&V3_REQUEST_MAGIC);
        payload.extend_from_slice(&PROTOCOL_VERSION_V3.to_le_bytes());
        payload.extend_from_slice(&sequence.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        payload.extend_from_slice(frame);
        self.write_frame(KIND_SUBMISSION_V3, &payload)
            .expect("send v3 submission");
        let (kind, reply) = self
            .read_frame()
            .expect("read v3 reply")
            .expect("v3 reply frame");
        assert_eq!(kind, KIND_SUBMISSION_V3, "unexpected reply kind");
        wire::decode_reply(&reply).expect("decode v3 reply")
    }
}

impl Drop for V3Client {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

/// Reads the presented IOSurface's rows as raw BGRA bytes.
fn read_surface_bgra(image: &SurfaceImage) -> Vec<u8> {
    let surface = image.iosurface_raw();
    // SAFETY: `surface` is the image's live IOSurface; a read-only lock is the
    // documented way to obtain stable CPU access to its backing store.
    let rc = unsafe { IOSurfaceLock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "IOSurfaceLock(read-only) failed");
    // SAFETY: The surface is locked, so the base address covers its rows.
    let base = unsafe { IOSurfaceGetBaseAddress(surface) }.cast::<u8>();
    assert!(!base.is_null(), "locked IOSurface has no base address");
    let len = image.bytes_per_row() * image.height() as usize;
    let mut pixels = vec![0u8; len];
    // SAFETY: `base` points to `len` readable bytes and `pixels` is a distinct
    // allocation of exactly that size.
    unsafe { std::ptr::copy_nonoverlapping(base, pixels.as_mut_ptr(), len) };
    // SAFETY: Unlocks the same surface with the matching read-only option.
    let rc = unsafe { IOSurfaceUnlock(surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "IOSurfaceUnlock failed");
    pixels
}

/// Retains a borrowed `MTLTexture` so it can be wrapped as a `RenderTarget`.
fn retained_texture(raw: *mut c_void) -> *mut c_void {
    assert!(!raw.is_null(), "surface texture is nil");
    // SAFETY: `raw` is a live `MTLTexture`; `retain` returns the same object +1.
    unsafe {
        let selector = sel_registerName(c"retain".as_ptr());
        let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(raw, selector)
    }
}

/// Wraps a surface image's texture as a render target without taking the
/// image's own reference.
fn wrap_surface_texture(image: &SurfaceImage) -> RenderTarget {
    let retained = retained_texture(image.texture_raw());
    // SAFETY: `retained` is a +1 retained `MTLTexture` over a BGRA8 IOSurface,
    // exactly what `from_retained_raw` takes ownership of.
    unsafe { RenderTarget::from_retained_raw(retained) }
        .expect("surface texture should wrap as a render target")
}

/// Every Mach-O image loaded in this process.
fn loaded_images() -> Vec<String> {
    let mut names = Vec::new();
    // SAFETY: The dyld image list is process-global and stable for the call.
    unsafe {
        let count = _dyld_image_count();
        for index in 0..count {
            let name = _dyld_get_image_name(index);
            if !name.is_null() {
                names.push(CStr::from_ptr(name).to_string_lossy().into_owned());
            }
        }
    }
    names
}

fn assert_no_d3dmetal_image() {
    let images = loaded_images();
    let leaked: Vec<&String> = images
        .iter()
        .filter(|name| name.to_ascii_lowercase().contains("d3dmetal"))
        .collect();
    assert!(
        leaked.is_empty(),
        "the gate process loaded D3DMetal images: {leaked:?}"
    );
    eprintln!(
        "d3dmetal_gate_process_images={} d3dmetal_matches=0 method=dyld_image_scan",
        images.len()
    );
}

fn assert_worker_links_no_d3dmetal(worker: &Path) {
    let output = Command::new("/usr/bin/otool")
        .arg("-L")
        .arg(worker)
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let listing = String::from_utf8_lossy(&output.stdout);
            let leaked: Vec<&str> = listing
                .lines()
                .filter(|line| line.to_ascii_lowercase().contains("d3dmetal"))
                .collect();
            assert!(
                leaked.is_empty(),
                "the converter worker links D3DMetal: {leaked:?}"
            );
            eprintln!(
                "d3dmetal_converter_worker_links=0 method=otool_L worker={}",
                worker.display()
            );
        }
        _ => eprintln!("d3dmetal_converter_worker_check=unavailable method=otool_L"),
    }
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> *mut c_void;
}

#[link(name = "System")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(index: u32) -> *const c_char;
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// Builds the MRT + depth graphics submission.
///
/// Draw A is a full-target triangle at z=0.9 (red/cyan). Draw B repeats it at
/// z=0.95 (blue/white) and must lose the depth test everywhere. Draw C repeats
/// it at z=0.1 (green/magenta) under a 1x1 centre scissor and must win only the
/// centre pixel. The pass therefore proves MRT routing, depth rejection and
/// depth ordering on one command buffer.
fn graphics_commands(
    pipeline: u64,
    geometry: u64,
    color0: u64,
    color1: u64,
    depth: u64,
    surface_id: u64,
) -> Vec<CommandV3> {
    let vertex_buffer = |offset: u64| VertexBufferSet {
        start_slot: 0,
        buffers: vec![VertexBuffer {
            allocation_id: geometry,
            generation: GENERATION,
            offset,
            stride: 16,
            length: 48,
        }],
    };
    let constants = |slot: u16, rgba: [f32; 4]| {
        CommandV3::SetRootConstants(RootConstantsV3 {
            slot,
            bytes: float4_bytes(rgba),
        })
    };
    vec![
        CommandV3::BeginRenderPass(BeginRenderPass {
            colors: vec![
                ColorAttachment {
                    allocation_id: color0,
                    generation: GENERATION,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                },
                ColorAttachment {
                    allocation_id: color1,
                    generation: GENERATION,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                },
            ],
            depth_stencil: Some(DepthStencilAttachment {
                allocation_id: depth,
                generation: GENERATION,
                mip_level: 0,
                slice: 0,
                depth_load: LoadOp::Clear,
                depth_store: StoreOp::Store,
                clear_depth: 1.0,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                clear_stencil: 0,
            }),
        }),
        CommandV3::SetGraphicsPipeline(PipelineRef {
            pipeline_id: pipeline,
        }),
        CommandV3::SetViewport(wire_v3::Viewport {
            x: 0.0,
            y: 0.0,
            width: f64::from(EDGE),
            height: f64::from(EDGE),
            near: 0.0,
            far: 1.0,
        }),
        CommandV3::SetScissor(wire_v3::ScissorRect {
            x: 0,
            y: 0,
            width: EDGE,
            height: EDGE,
        }),
        CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
        // Draw A: full-target triangle at z=0.9, indices at PAGE+0.
        CommandV3::SetVertexBuffers(vertex_buffer((PAGE + 16) as u64)),
        CommandV3::SetIndexBuffer(IndexBufferBinding {
            allocation_id: geometry,
            generation: GENERATION,
            offset: PAGE as u64,
            length: 6,
            format: IndexFormat::Uint16,
        }),
        constants(1, RED_RGBA),
        constants(2, CYAN_RGBA),
        CommandV3::DrawIndexed(DrawIndexedState {
            index_count: 3,
            instance_count: 1,
            start_index: 0,
            base_vertex: 0,
            start_instance: 0,
        }),
        // Draw B: full-target at z=0.95, rejected by the depth test everywhere.
        CommandV3::SetVertexBuffers(vertex_buffer((PAGE + 64) as u64)),
        constants(1, BLUE_RGBA),
        constants(2, WHITE_RGBA),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        // Draw C: full-target at z=0.1 under a 1x1 centre scissor; nearer depth
        // wins only the centre pixel.
        CommandV3::SetVertexBuffers(vertex_buffer((PAGE + 112) as u64)),
        CommandV3::SetScissor(wire_v3::ScissorRect {
            x: EDGE / 2,
            y: EDGE / 2,
            width: 1,
            height: 1,
        }),
        constants(1, GREEN_RGBA),
        constants(2, MAGENTA_RGBA),
        CommandV3::Draw(DrawState {
            vertex_count: 3,
            instance_count: 1,
            start_vertex: 0,
            start_instance: 0,
        }),
        CommandV3::EndRenderPass,
        CommandV3::Present(Present {
            surface_id,
            sync_interval: 1,
        }),
    ]
}

/// Asserts one colour target's exact pixels: `background` everywhere except the
/// centre pixel, which the depth-tested draw overwrites with `centre`.
fn assert_target_pixels(pixels: &[[u8; 4]], background: [u8; 4], centre: [u8; 4], label: &str) {
    let edge = EDGE as usize;
    assert_eq!(pixels.len(), edge * edge, "{label}: unexpected pixel count");
    let centre_at = (edge / 2) * edge + edge / 2;
    for (index, pixel) in pixels.iter().enumerate() {
        let expected = if index == centre_at {
            centre
        } else {
            background
        };
        assert_eq!(
            *pixel,
            expected,
            "{label}: pixel ({}, {}) was {pixel:?}, expected {expected:?}",
            index % edge,
            index / edge
        );
    }
}

fn read_file_words(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).expect("read the guest allocation");
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("a four-byte word")))
        .collect()
}

#[test]
#[ignore = "GPU hardware gate; x86_64 IRConverter worker and a free M4 Max Metal device required"]
fn v3_production_gate_compute_alias_graphics_mrt_and_refusals() {
    let (worker_path, converter) = converter_paths().expect("IRConverter worker paths");
    assert_no_d3dmetal_image();
    assert_worker_links_no_d3dmetal(&worker_path);

    let directory = SessionDir::new();
    let allocation = directory.allocation_path();
    fs::write(&allocation, vec![0u8; FILE_LEN]).expect("create the guest allocation");

    let mut mapping = SharedMapping::open(&allocation, FILE_LEN);
    {
        let bytes = mapping.as_mut_slice();
        // Page 0: the IPC compute input, 64 dwords.
        for index in 0..COMPUTE_THREADS as usize {
            let at = index * 4;
            bytes[at..at + 4].copy_from_slice(&(index as u32 + 1).to_le_bytes());
        }
        // Page 1: index data at 0, draw A vertices at 16, draw B at 64.
        for (index, value) in [0u16, 1, 2].iter().enumerate() {
            let at = PAGE + index * 2;
            bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
        }
        // Three full-target triangles: far (0.9), farther (0.95) and near (0.1).
        let at_depth = |z: f32| [-1.0f32, -1.0, z, 1.0, 3.0, -1.0, z, 1.0, -1.0, 3.0, z, 1.0];
        for (offset, vertices) in [
            (PAGE + 16, at_depth(0.9)),
            (PAGE + 64, at_depth(0.95)),
            (PAGE + 112, at_depth(0.1)),
        ] {
            for (index, value) in vertices.iter().enumerate() {
                let at = offset + index * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }

    let transport = Arc::new(DirectMetalTransport::new().expect("Metal device should open"));
    let page_size = BatchExecutor::page_size(&*transport).expect("device page size");
    assert!(
        page_size.is_power_of_two() && page_size >= 4096,
        "unexpected device page size {page_size}"
    );
    assert_eq!(
        FILE_LEN % page_size as usize,
        0,
        "the allocation must cover whole device pages"
    );

    // Register the same file-backed window the guest owns, for the direct
    // converted-DXIL and graphics submissions.
    // SAFETY: `mapping` is a live, page-aligned, MAP_SHARED view of the file and
    // stays alive until every submission below has completed.
    let buffer_id = unsafe { transport.register_zerocopy(mapping.ptr.cast::<u8>(), FILE_LEN) }
        .expect("register allocation");

    // Host pipelines: the MSL kernel dispatched over IPC, the MRT graphics
    // pipeline, and the Rosetta-converted DXIL kernel.
    let compute_pipeline = transport
        .compile_msl(COMPUTE_MSL, "pe_alias_mul_add")
        .expect("MSL compute pipeline should compile");

    let dxil = hex_bytes(DXIL_HEX);
    assert!(
        dxil.windows(4).any(|window| window == b"DXIL"),
        "the embedded fixture is not DXIL"
    );
    let worker = ProcessWorker::new(&worker_path, &converter);
    let converted = match translate_dxil_with(
        None,
        &worker,
        &dxil,
        "main",
        8,
        None,
        &TranslationOptions::default(),
        &[],
    ) {
        Ok(outcome) => outcome,
        Err(error) => panic!(
            "the Rosetta IRConverter worker must translate the DXIL fixture: {error} \
             (worker {})",
            worker_path.display()
        ),
    };
    assert!(
        !converted.metallib.is_empty(),
        "converter produced no metallib"
    );
    assert_eq!(
        &converted.metallib[..4],
        b"MTLB",
        "converter output is not a metallib"
    );
    let dxil_pipeline = transport
        .load_metallib(&converted.metallib, "main")
        .expect("converted metallib should load on ARM64 Metal");

    let mrt_pipeline = GraphicsPipeline::compile_mrt(
        &transport,
        GRAPHICS_MSL,
        "pe_vertex",
        "pe_fragment",
        &[PixelFormat::Bgra8Unorm, PixelFormat::Bgra8Unorm],
        Some(PixelFormat::Depth32Float),
    )
    .expect("MRT graphics pipeline with depth should compile");
    let graphics_pipeline = transport
        .register_graphics_pipeline_state(mrt_pipeline)
        .expect("graphics pipeline should register");

    // Two IOSurface-backed colour targets and one depth target.
    let mut image0 = SurfaceImage::create(&transport, EDGE, EDGE).expect("first IOSurface");
    let image1 = SurfaceImage::create(&transport, EDGE, EDGE).expect("second IOSurface");
    let target0 = wrap_surface_texture(&image0);
    let target1 = wrap_surface_texture(&image1);
    let depth = transport
        .create_render_target_2d(EDGE, EDGE, PixelFormat::Depth32Float)
        .expect("depth target should be created");
    let color0 = transport
        .register_render_target(&target0)
        .expect("first colour target should register");
    let color1 = transport
        .register_render_target(&target1)
        .expect("second colour target should register");
    let depth_id = transport
        .register_render_target(&depth)
        .expect("depth target should register");
    transport
        .register_present_surface(u64::from(image0.surface_id()), &target0)
        .expect("present surface should register");

    // --- IPC session over the real v3 host bridge -------------------------
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind the native host");
    let address = listener.local_addr().expect("listener address");
    let secret = [0x5au8; 32];
    let host = IpcHost::new(
        HostSecret::new(secret),
        directory.path(),
        SharedExecutor(Arc::clone(&transport)),
    )
    .expect("create the IPC host");
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept the gate client");
        let mut host = host;
        let summary = host.serve_once(stream);
        host.release_all();
        summary
    });

    let mut client = V3Client::connect(address, &secret);
    client.register(ALLOCATION_ID, FILE_LEN as u64);

    // The launcher creates the page; the host opens it (kind 6) and publishes
    // completions from its GPU completion handler. This second mapping is the
    // guest's view: one atomic load, no IPC round-trip.
    let fence_path = directory.path().join(FENCE_PAGE_FILE_NAME);
    let launcher_page = FencePageMapping::create(&fence_path).expect("create the fence page");
    client.register_fence_page();
    let guest_page = FencePageMapping::open(&fence_path).expect("open the guest fence page");
    // `begin_session` ran before the acknowledgement: both fences start at zero
    // and the epoch advanced.
    assert_eq!(guest_page.page().completed(), 0, "session must start idle");
    assert_eq!(guest_page.page().submitted(), 0);
    assert_eq!(guest_page.page().epoch(), 1, "session epoch must advance");
    drop(launcher_page);

    let identity = ListIdentity {
        list_id: 41,
        list_generation: 1,
    };
    let compute_commands = vec![CommandV3::Dispatch1d(victory_metal::wire::Dispatch1d {
        pipeline_id: compute_pipeline.0,
        threads: COMPUTE_THREADS,
        threads_per_group: COMPUTE_THREADS,
        bindings: vec![BufferBinding {
            allocation_id: ALLOCATION_ID,
            generation: GENERATION,
            offset: 0,
            length: COMPUTE_BYTES,
        }],
    })];
    let compute = SubmissionV3 {
        sequence: 1,
        list: Some(identity),
        commands: compute_commands.clone(),
    };
    let frame = wire_v3::encode_submission_v3(&compute).expect("encode the compute list");
    assert_eq!(
        client.submit_v3(1, &frame),
        Reply {
            sequence: 1,
            status: ReplyStatus::Ok,
        }
    );
    // The host submits asynchronously (ADR-017): `Ok` means the command buffer
    // was accepted, so the guest-visible alias is only final after the GPU
    // completed the submission. `submitted_fence` is published before the
    // executor runs; `completed_fence` only from the completion handler.
    assert!(
        guest_page.page().submitted() >= 1,
        "the host must publish the submitted sequence before executing"
    );
    transport
        .wait_for_gpu()
        .expect("the IPC compute submission should complete");
    assert_eq!(
        guest_page.page().completed(),
        1,
        "the completion handler must publish sequence 1 exactly once"
    );
    assert_eq!(transport.completed_fence(), 1);
    let after_compute = fs::read(&allocation).expect("read the alias after compute");
    for index in 0..COMPUTE_THREADS as usize {
        let at = index * 4;
        let value = u32::from_le_bytes(after_compute[at..at + 4].try_into().unwrap());
        let expected = (index as u32 + 1) * 3 + 7;
        assert_eq!(
            value, expected,
            "alias dword {index} is not the kernel's output"
        );
    }
    eprintln!(
        "alias_compute_dwords={COMPUTE_THREADS} first={} last={} pipeline={}",
        u32::from_le_bytes(after_compute[..4].try_into().unwrap()),
        u32::from_le_bytes(
            after_compute[COMPUTE_BYTES as usize - 4..COMPUTE_BYTES as usize]
                .try_into()
                .unwrap()
        ),
        compute_pipeline.0
    );

    // --- Replaying the same list identity must be refused, bytes unchanged.
    let replay = SubmissionV3 {
        sequence: 2,
        list: Some(identity),
        commands: compute_commands,
    };
    let frame = wire_v3::encode_submission_v3(&replay).expect("encode the replay");
    assert_eq!(
        client.submit_v3(2, &frame),
        Reply {
            sequence: 2,
            status: ReplyStatus::AlreadyExecuted,
        },
        "a replay of the executed list must be refused"
    );
    let after_replay = fs::read(&allocation).expect("read the alias after replay");
    assert_eq!(
        after_replay, after_compute,
        "the refused replay changed the guest-visible alias"
    );
    eprintln!("replay=AlreadyExecuted alias_unchanged=1 list_id=41");

    // --- Unknown allocation: explicit failure, no GPU work.
    let unknown = SubmissionV3 {
        sequence: 3,
        list: None,
        commands: vec![CommandV3::Dispatch1d(victory_metal::wire::Dispatch1d {
            pipeline_id: compute_pipeline.0,
            threads: 1,
            threads_per_group: 1,
            bindings: vec![BufferBinding {
                allocation_id: ALLOCATION_ID + 9_999,
                generation: GENERATION,
                offset: 0,
                length: 4,
            }],
        })],
    };
    let frame = wire_v3::encode_submission_v3(&unknown).expect("encode the unknown allocation");
    assert_eq!(
        client.submit_v3(3, &frame),
        Reply {
            sequence: 3,
            status: ReplyStatus::UnknownHandle,
        },
        "an unknown allocation must be an explicit refusal"
    );
    assert_eq!(
        fs::read(&allocation).expect("read the alias after the refusal"),
        after_compute,
        "an unknown allocation must not run GPU work"
    );
    eprintln!("unknown_allocation=UnknownHandle no_fallback=1");

    // --- Unsupported opcode: the decoder refuses the frame, the host answers.
    let mut unsupported = wire_v3::encode_submission_v3(&SubmissionV3 {
        sequence: 4,
        list: None,
        commands: vec![CommandV3::Dispatch1d(victory_metal::wire::Dispatch1d {
            pipeline_id: compute_pipeline.0,
            threads: 1,
            threads_per_group: 1,
            bindings: vec![BufferBinding {
                allocation_id: ALLOCATION_ID,
                generation: GENERATION,
                offset: 0,
                length: 4,
            }],
        })],
    })
    .expect("encode the frame to corrupt");
    // Frame header (20) + list block (12) is the first command's opcode.
    assert!(unsupported.len() >= 34, "encoded frame is too short");
    unsupported[32..34].copy_from_slice(&0x00FFu16.to_le_bytes());
    assert_eq!(
        client.submit_v3(4, &unsupported),
        Reply {
            sequence: 4,
            status: ReplyStatus::ExecutionFailed,
        },
        "an unsupported opcode must be an explicit failure"
    );
    // The rejection did not advance the sequence; a valid frame 4 is accepted.
    let recovered = wire_v3::encode_submission_v3(&SubmissionV3 {
        sequence: 4,
        list: None,
        commands: Vec::new(),
    })
    .expect("encode the recovery frame");
    assert_eq!(
        client.submit_v3(4, &recovered),
        Reply {
            sequence: 4,
            status: ReplyStatus::Ok,
        },
        "a rejected frame must leave the session usable"
    );
    eprintln!("unsupported_opcode=ExecutionFailed session_recovered=1");

    drop(client);
    let summary = server.join().expect("host thread").expect("host session");
    assert_eq!(summary.allocations, 1, "unexpected allocation count");
    assert_eq!(summary.submissions, 5, "unexpected submission count");
    assert_eq!(
        summary.bytes_mapped, FILE_LEN as u64,
        "unexpected mapped bytes"
    );

    // --- Converted DXIL compute writes the guest-visible third page.
    let expected_dxil: Vec<u32> = (0..COMPUTE_THREADS).map(|tid| tid * 3 + 7).collect();
    let request = MscUavDispatch {
        pipeline: dxil_pipeline,
        uav: buffer_id,
        offset: (2 * PAGE) as u64,
        length: COMPUTE_BYTES,
        threads: COMPUTE_THREADS,
        threads_per_group: COMPUTE_THREADS,
        layout: MscUavLayout::DEFAULT,
    };
    transport
        .dispatch_msc_uav_1d(&request)
        .expect("the converted DXIL kernel must dispatch");
    let words = read_file_words(&allocation);
    let base = 2 * PAGE / 4;
    for (index, expected) in expected_dxil.iter().enumerate() {
        assert_eq!(
            words[base + index],
            *expected,
            "converted DXIL dword {index} mismatch"
        );
    }
    eprintln!(
        "dxil_compute=Rosetta worker={} metallib_bytes={} dwords={} first={} last={}",
        worker_path.display(),
        converted.metallib.len(),
        COMPUTE_THREADS,
        words[base],
        words[base + COMPUTE_THREADS as usize - 1]
    );

    // --- Graphics: MRT + depth through submit_v3, presented through IOSurface.
    let geometry = ALLOCATION_ID;
    let commands = graphics_commands(
        graphics_pipeline.0,
        geometry,
        color0.0,
        color1.0,
        depth_id.0,
        u64::from(image0.surface_id()),
    );
    let graphics = SubmissionV3 {
        sequence: 5,
        list: Some(ListIdentity {
            list_id: 42,
            list_generation: 1,
        }),
        commands,
    };
    let mut allocations = HashMap::new();
    allocations.insert((ALLOCATION_ID, GENERATION), buffer_id);
    allocations.insert((color0.0, GENERATION), VmBufferId(color0.0));
    allocations.insert((color1.0, GENERATION), VmBufferId(color1.0));
    allocations.insert((depth_id.0, GENERATION), VmBufferId(depth_id.0));
    let reply = transport
        .submit_v3(&graphics, &allocations)
        .expect("the MRT submission must execute");
    assert_eq!(
        reply,
        Reply {
            sequence: 5,
            status: ReplyStatus::Ok,
        }
    );
    // The draw is committed, not finished: the presented surface and the second
    // colour target are read on the CPU below, so observe GPU completion first.
    transport
        .wait_for_gpu()
        .expect("the MRT submission should complete");

    // The frame that was drawn is the frame the host presents.
    let mut path = PresentPath::new(&transport);
    let mut sink = RecordingSink::new();
    path.present_drawn(&mut image0, &mut sink)
        .expect("the drawn frame must present");
    assert_eq!(sink.frames().len(), 1, "the sink must observe one frame");
    assert_eq!(sink.frames()[0].surface_id, image0.surface_id());
    path.finish_present()
        .expect("the presented frame must finish");

    let presented = read_surface_bgra(&image0);
    let row_bytes = image0.bytes_per_row();
    let mut frame_pixels = Vec::with_capacity((EDGE * EDGE) as usize);
    for y in 0..EDGE as usize {
        for x in 0..EDGE as usize {
            let at = y * row_bytes + x * 4;
            frame_pixels.push([
                presented[at],
                presented[at + 1],
                presented[at + 2],
                presented[at + 3],
            ]);
        }
    }
    let red = bgra(RED_RGBA);
    let cyan = bgra(CYAN_RGBA);
    let green = bgra(GREEN_RGBA);
    let magenta = bgra(MAGENTA_RGBA);
    assert!(
        !frame_pixels
            .iter()
            .any(|pixel| *pixel == bgra(BLUE_RGBA) || *pixel == bgra(WHITE_RGBA)),
        "target 0 contains the depth-rejected far draw"
    );
    assert_target_pixels(&frame_pixels, red, green, "target 0 (presented)");
    let second = transport
        .read_render_target_rgba8(&target1)
        .expect("second target should read back");
    assert_target_pixels(&second, cyan, magenta, "target 1");
    eprintln!(
        "graphics=submit_v3 mrt=2 depth=1 pixels={} centre0={:?} centre1={:?}",
        EDGE * EDGE,
        frame_pixels[((EDGE / 2) * EDGE + EDGE / 2) as usize],
        second[((EDGE / 2) * EDGE + EDGE / 2) as usize]
    );

    // A replay of the graphics list must not touch the drawn frame.
    let snapshot = presented.clone();
    let reply = transport
        .submit_v3(&graphics, &allocations)
        .expect("a graphics replay must return a reply");
    assert_eq!(reply.status, ReplyStatus::AlreadyExecuted);
    assert_eq!(
        read_surface_bgra(&image0),
        snapshot,
        "the refused graphics replay modified the presented surface"
    );
    eprintln!("graphics_replay=AlreadyExecuted presented_surface_unchanged=1");

    transport
        .unregister_buffer(buffer_id)
        .expect("unregister the shared allocation");
    let device = transport.device_info().expect("device info");
    eprintln!(
        "producer_e2e PASS device={} colour_targets=2 depth=1 list_ids=41,42 \
         d3dmetal_images=0 wine_guest_image_check=host-only \
         wine_guest_probe=\"synthetic-frame.sh: psapi d3dmetal_modules=0\" \
         unsupported_link=\"GuestIpcClient has no v3 encoder; victory-gpu installs DisconnectedSink\"",
        device.name_str()
    );
}
