//! Wine-decoupled native Apple Metal backend for VIctory.
//!
//! Designed with strict separation between the Metal GPU engine and any OS/Wine
//! transport layer:
//! - [`abi`] defines stable `#[repr(C)]` status codes, handles, and device info.
//! - [`transport`] defines [`MetalTransport`], implemented in-process on macOS
//!   without any Wine, D3D12, or Windows dependencies.
//! - [`wire`] defines the versioned little-endian submission protocol shared
//!   between the Win64 Wine client and the ARM64 host process.
//! - [`guest_ipc`] is the synchronous Win64 guest client for the local
//!   loopback transport that carries hello, registration and wire frames.
//! - On macOS (`target_os = "macos"`), [`macos::device::DirectMetalTransport`]
//!   provides real `MTLDevice`, `newBufferWithBytesNoCopy` unified-memory buffers,
//!   compute dispatch, `libmetalirconverter.dylib` DXIL compilation, and [`macos::c_api`].
//! - On macOS, [`ipc_host`] adds the loopback transport, authentication,
//!   allocation file mapping and one-peer session around the same
//!   [`macos::device::DirectMetalTransport::submit_batch`] entry point.
//! - [`macos::msc`] is the opt-in typed binder for the verified Metal Shader
//!   Converter `UAV(u0)` argument-buffer shape; see
//!   `crates/victory-metal/docs/IR-BRIDGE-CONTRACT.md`.

pub mod abi;
pub mod cmdlist;
pub mod guest_ipc;
#[cfg(target_os = "macos")]
pub mod ipc_host;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod transport;
pub mod wire;

pub use abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmLibraryId, VmPipelineId, VmStatus};
pub use guest_ipc::{
    GuestIpcClient, GuestIpcError, MAX_FRAME_PAYLOAD as GUEST_IPC_MAX_FRAME_PAYLOAD,
    RegisteredAllocation,
};
#[cfg(target_os = "macos")]
pub use macos::device::DirectMetalTransport;
pub use transport::{MetalError, MetalTransport};

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use std::ffi::c_void;

    #[cfg(target_os = "macos")]
    use crate::{MetalTransport, VmBufferId};
    use crate::{MetalError, VmDeviceInfo, VmStatus};

    #[test]
    fn error_maps_to_expected_abi_status() {
        assert_eq!(MetalError::DeviceUnavailable.status(), VmStatus::DeviceUnavailable);
        assert_eq!(
            MetalError::NotSupported("x").status(),
            VmStatus::NotSupported
        );
        assert_eq!(
            MetalError::AlignmentError {
                ptr: 4096,
                len: 16384,
                page_size: 16384,
            }
            .status(),
            VmStatus::AlignmentError
        );
    }

    #[test]
    fn device_info_name_trims_nul_padding() {
        let mut info = VmDeviceInfo::default();
        let bytes = b"Apple M4 Max";
        info.name[..bytes.len()].copy_from_slice(bytes);
        assert_eq!(info.name_str(), "Apple M4 Max");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_device_is_unified_memory() {
        let transport = crate::DirectMetalTransport::new().expect("Metal device should open");
        let info = transport.device_info().expect("device_info should succeed");
        assert!(!info.name_str().is_empty(), "empty Metal device name");
        assert_eq!(info.has_unified_memory, 1, "expected Apple Silicon unified memory");
        assert!(
            info.max_threads_per_threadgroup >= 256,
            "unexpected threadgroup size: {}",
            info.max_threads_per_threadgroup
        );
        assert!(info.page_size >= 16384, "unexpected page_size: {}", info.page_size);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn zerocopy_compute_roundtrip_without_memcpy() {
        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const BYTES: usize = 65536;
        const COUNT: usize = 1024;

        let transport = crate::DirectMetalTransport::new().expect("Metal device should open");
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered in Metal.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // SAFETY: `[raw, raw + BYTES)` is freshly allocated and aligned for `COUNT` u32 elements.
        let host = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), COUNT) };
        for (i, slot) in host.iter_mut().enumerate() {
            *slot = (i as u32) + 1;
        }

        // Verify misaligned pointer cleanly fails before touching Metal.
        // SAFETY: Intentionally misaligned pointer + length tested only for validation failure.
        let misalign = unsafe { transport.register_zerocopy(raw.cast::<u8>().add(4), BYTES) };
        assert!(matches!(misalign, Err(MetalError::AlignmentError { .. })));

        // SAFETY: `raw` points to 64 KiB valid memory that remains live past `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void vm_test_mul_add(device uint *data [[buffer(0)]],
                                        uint tid [[thread_position_in_grid]]) {
                data[tid] = data[tid] * 3u + 7u;
            }
        "#;
        let pso = transport
            .compile_msl(msl, "vm_test_mul_add")
            .expect("MSL kernel compilation should succeed");

        transport
            .dispatch_compute_1d(pso, &[buf_id], COUNT as u32, 64)
            .expect("Metal compute dispatch should complete");

        // Zero memcpy: read back directly from the CPU pointer.
        assert_eq!(host[0], 10);
        assert_eq!(host[COUNT - 1], (COUNT as u32) * 3 + 7);
        for (i, &val) in host.iter().enumerate() {
            let expected = ((i as u32) + 1) * 3 + 7;
            assert_eq!(val, expected, "mismatch at dword {i}");
        }

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");
        assert!(matches!(
            transport.unregister_buffer(VmBufferId(buf_id.0)),
            Err(MetalError::NotFound(_))
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn wire_batch_two_dispatches_at_nonzero_offset_and_stale_generation_is_untouched() {
        use std::collections::HashMap;

        use crate::wire::{
            self, BufferBinding, Command, Dispatch1d, Reply, ReplyStatus, Submission,
        };

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const RANGE_BYTES: usize = 65536;
        const TOTAL_BYTES: usize = RANGE_BYTES * 2;
        const DWORDS: usize = RANGE_BYTES / 4;
        const OFFSET: u64 = RANGE_BYTES as u64;
        const ALLOCATION_ID: u64 = 1;
        const GENERATION: u32 = 1;
        const SEQUENCE: u32 = 7;
        const SENTINEL: u32 = 0xDEAD_BEEF;

        let transport = crate::DirectMetalTransport::new().expect("Metal device should open");
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, TOTAL_BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // `[0, 64 KiB)` is an untouched sentinel prefix; `[64 KiB, 128 KiB)`
        // is the known input that both dispatches operate on in order.
        // SAFETY: `[raw, raw + TOTAL_BYTES)` is freshly allocated and u32-aligned.
        let all = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), TOTAL_BYTES / 4) };
        let (prefix, work) = all.split_at_mut(DWORDS);
        prefix.fill(SENTINEL);
        for (index, slot) in work.iter_mut().enumerate() {
            *slot = index as u32 + 1;
        }

        // SAFETY: `raw` points to 128 KiB valid memory that outlives `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), TOTAL_BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void vm_batch_scale_add(device uint *data [[buffer(0)]],
                                           uint tid [[thread_position_in_grid]]) {
                data[tid] = data[tid] * 3u + 7u;
            }
            kernel void vm_batch_scale_sub(device uint *data [[buffer(0)]],
                                           uint tid [[thread_position_in_grid]]) {
                data[tid] = data[tid] * 5u - 2u;
            }
        "#;
        let first_pipeline = transport
            .compile_msl(msl, "vm_batch_scale_add")
            .expect("first batch kernel should compile");
        let second_pipeline = transport
            .compile_msl(msl, "vm_batch_scale_sub")
            .expect("second batch kernel should compile");

        let mut allocations = HashMap::new();
        allocations.insert((ALLOCATION_ID, GENERATION), buf_id);

        let dispatch = |pipeline_id: u64| Dispatch1d {
            pipeline_id,
            threads: DWORDS as u32,
            threads_per_group: 64,
            bindings: vec![BufferBinding {
                allocation_id: ALLOCATION_ID,
                generation: GENERATION,
                offset: OFFSET,
                length: RANGE_BYTES as u64,
            }],
        };
        let submission = Submission {
            sequence: SEQUENCE,
            commands: vec![
                Command::Dispatch1d(dispatch(first_pipeline.0)),
                Command::Dispatch1d(dispatch(second_pipeline.0)),
            ],
        };

        // Exercise the version-1 wire codec before the native entry point.
        let frame = wire::encode_submission(&submission).expect("submission should encode");
        let decoded = wire::decode_submission(&frame).expect("submission should decode");
        assert_eq!(decoded, submission);

        // A stale generation on the second command must reject the whole batch
        // before any command buffer is created: the first command's writes must
        // not appear and the prefix sentinel must stay intact.
        let mut stale = decoded.clone();
        let Command::Dispatch1d(second) = &mut stale.commands[1] else {
            panic!("expected a 1D dispatch")
        };
        second.bindings[0].generation = GENERATION + 1;
        let reply = transport
            .submit_batch(&stale, &allocations)
            .expect("stale batch should return a reply");
        assert_eq!(
            reply,
            Reply {
                sequence: SEQUENCE,
                status: ReplyStatus::UnknownHandle,
            }
        );
        for (index, &value) in work.iter().enumerate() {
            assert_eq!(
                value,
                index as u32 + 1,
                "stale batch partially wrote dword {index}"
            );
        }
        assert!(
            prefix.iter().all(|&value| value == SENTINEL),
            "stale batch modified the prefix sentinel"
        );

        // An unknown pipeline is likewise rejected before GPU work.
        let mut unknown_pipeline = decoded.clone();
        let Command::Dispatch1d(first) = &mut unknown_pipeline.commands[0] else {
            panic!("expected a 1D dispatch")
        };
        first.pipeline_id = u64::MAX;
        let reply = transport
            .submit_batch(&unknown_pipeline, &allocations)
            .expect("unknown pipeline should return a reply");
        assert_eq!(reply.status, ReplyStatus::UnknownHandle);
        assert!(
            work.iter()
                .enumerate()
                .all(|(index, &value)| value == index as u32 + 1),
            "unknown-pipeline batch modified the input range"
        );

        // The valid batch runs both ordered arithmetic ops at the nonzero offset.
        let reply = transport
            .submit_batch(&decoded, &allocations)
            .expect("valid batch should complete");
        assert_eq!(
            reply,
            Reply {
                sequence: SEQUENCE,
                status: ReplyStatus::Ok,
            }
        );
        for (index, &value) in work.iter().enumerate() {
            let expected = ((index as u32 + 1) * 3 + 7) * 5 - 2;
            assert_eq!(value, expected, "mismatch at dword {index}");
        }
        assert!(
            prefix.iter().all(|&value| value == SENTINEL),
            "batch modified memory before the bound range offset"
        );

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn wire_batch_rejects_zero_handle_fields_and_oversized_threadgroup_without_gpu_work() {
        use std::collections::HashMap;

        use crate::wire::{BufferBinding, Command, Dispatch1d, Reply, ReplyStatus, Submission};

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const BYTES: usize = 65536;
        const DWORDS: usize = BYTES / 4;
        const ALLOCATION_ID: u64 = 1;
        const GENERATION: u32 = 1;
        const SEQUENCE: u32 = 9;

        let transport = crate::DirectMetalTransport::new().expect("Metal device should open");
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // SAFETY: `[raw, raw + BYTES)` is freshly allocated and u32-aligned.
        let work = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), DWORDS) };
        for (index, slot) in work.iter_mut().enumerate() {
            *slot = index as u32 + 1;
        }

        // SAFETY: `raw` points to 64 KiB valid memory that outlives `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void vm_batch_guard(device uint *data [[buffer(0)]],
                                       uint tid [[thread_position_in_grid]]) {
                data[tid] = data[tid] + 1u;
            }
        "#;
        let pipeline = transport
            .compile_msl(msl, "vm_batch_guard")
            .expect("guard kernel should compile");

        // Deliberately seed the snapshot with the forbidden zero keys: the
        // programmatic entry point must reject them exactly like the codec.
        let mut allocations = HashMap::new();
        allocations.insert((0, GENERATION), buf_id);
        allocations.insert((ALLOCATION_ID, 0), buf_id);
        allocations.insert((ALLOCATION_ID, GENERATION), buf_id);

        let submission = |allocation_id: u64, generation: u32, threads_per_group: u32| Submission {
            sequence: SEQUENCE,
            commands: vec![Command::Dispatch1d(Dispatch1d {
                pipeline_id: pipeline.0,
                threads: DWORDS as u32,
                threads_per_group,
                bindings: vec![BufferBinding {
                    allocation_id,
                    generation,
                    offset: 0,
                    length: BYTES as u64,
                }],
            })],
        };
        let expect_unknown_handle = |submission: &Submission, label: &str| {
            let reply = transport
                .submit_batch(submission, &allocations)
                .expect("invalid handle fields should return a reply");
            assert_eq!(
                reply,
                Reply {
                    sequence: SEQUENCE,
                    status: ReplyStatus::UnknownHandle,
                },
                "{label}"
            );
        };

        expect_unknown_handle(&submission(0, GENERATION, 64), "zero allocation ID");
        expect_unknown_handle(&submission(ALLOCATION_ID, 0, 64), "zero generation");
        assert!(
            work.iter()
                .enumerate()
                .all(|(index, &value)| value == index as u32 + 1),
            "rejected zero-handle batch modified GPU-visible memory"
        );

        // Any explicit width above the pipeline maximum must fail instead of
        // being clamped to a different threadgroup shape.
        let device_max = transport
            .device_info()
            .expect("device_info should succeed")
            .max_threads_per_threadgroup;
        let oversized = device_max.saturating_add(1);
        let err = transport
            .submit_batch(&submission(ALLOCATION_ID, GENERATION, oversized), &allocations)
            .expect_err("oversized threadgroup width should be rejected");
        assert!(
            matches!(err, MetalError::InvalidArg(_)),
            "unexpected oversized-width error: {err:?}"
        );
        assert!(
            work.iter()
                .enumerate()
                .all(|(index, &value)| value == index as u32 + 1),
            "rejected oversized-threadgroup batch modified GPU-visible memory"
        );

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");
    }

    /// Synthetic 3D grid proof for `dispatchThreadgroups:threadsPerThreadgroup:`.
    ///
    /// Ignored by default because it owns the only Metal device on the host.
    /// Run explicitly with `--ignored` on ARM64 macOS after compile-only or
    /// CPU-wire validation.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires an exclusive Metal GPU; run explicitly"]
    fn wire_batch_3d_dispatches_exact_group_grid_and_rejects_oversized_local_grid() {
        use std::collections::HashMap;

        use crate::wire::{self, BufferBinding, Command, Dispatch3d, ReplyStatus, Submission};

        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
            fn free(ptr: *mut c_void);
        }

        const ALIGN: usize = 16384;
        const BYTES: usize = 16384;
        const DWORDS: usize = BYTES / 4;
        const ALLOCATION_ID: u64 = 11;
        const GENERATION: u32 = 4;
        const SEQUENCE: u32 = 13;
        // groups [5, 3, 2] x local [3, 2, 1] => exact grid [15, 6, 2].
        const GRID_WIDTH: usize = 15;
        const GRID_DEPTH_STRIDE: usize = 15 * 6;
        const GRID_DWORDS: usize = 15 * 6 * 2;
        const SENTINEL: u32 = 0xDEAD_BEEF;

        let transport = crate::DirectMetalTransport::new().expect("Metal device should open");
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes a valid 16 KiB-aligned allocation into `raw`.
        let rc = unsafe { posix_memalign(&mut raw, ALIGN, BYTES) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());

        struct PageGuard(*mut c_void);
        impl Drop for PageGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `posix_memalign` and is no longer registered.
                unsafe { free(self.0) };
            }
        }
        let _guard = PageGuard(raw);

        // SAFETY: `[raw, raw + BYTES)` is freshly allocated and u32-aligned.
        let work = unsafe { std::slice::from_raw_parts_mut(raw.cast::<u32>(), DWORDS) };
        work.fill(SENTINEL);

        // SAFETY: `raw` points to 16 KiB valid memory that outlives `unregister_buffer`.
        let buf_id = unsafe { transport.register_zerocopy(raw.cast::<u8>(), BYTES) }
            .expect("register_zerocopy should succeed on 16 KiB unified memory");

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void vm_dispatch3d_mark(device uint *data [[buffer(0)]],
                                           uint3 tid [[thread_position_in_grid]]) {
                uint index = tid.x + tid.y * 15u + tid.z * 90u;
                data[index] = index + 1u;
            }
        "#;
        let pipeline = transport
            .compile_msl(msl, "vm_dispatch3d_mark")
            .expect("3D guard kernel should compile");

        let mut allocations = HashMap::new();
        allocations.insert((ALLOCATION_ID, GENERATION), buf_id);

        let submission = |group_counts: [u32; 3], threads_per_group: [u32; 3]| Submission {
            sequence: SEQUENCE,
            commands: vec![Command::Dispatch3d(Dispatch3d {
                pipeline_id: pipeline.0,
                group_counts,
                threads_per_group,
                bindings: vec![BufferBinding {
                    allocation_id: ALLOCATION_ID,
                    generation: GENERATION,
                    offset: 0,
                    length: BYTES as u64,
                }],
            })],
        };

        // Exercise the version-1 wire codec before the native entry point.
        let valid = submission([5, 3, 2], [3, 2, 1]);
        let frame = wire::encode_submission(&valid).expect("3D submission should encode");
        assert_eq!(
            wire::decode_submission(&frame).expect("3D submission should decode"),
            valid
        );

        // A local threadgroup above the pipeline's hardware maximum must fail
        // before a command buffer exists, leaving every dword untouched.
        let device_max = transport
            .device_info()
            .expect("device_info should succeed")
            .max_threads_per_threadgroup;
        let oversized = submission([1, 1, 1], [device_max.saturating_add(1), 1, 1]);
        let err = transport
            .submit_batch(&oversized, &allocations)
            .expect_err("oversized 3D local dimensions should be rejected");
        assert!(
            matches!(err, MetalError::InvalidArg(_)),
            "unexpected oversized-3D error: {err:?}"
        );
        assert!(
            work.iter().all(|&value| value == SENTINEL),
            "rejected oversized 3D batch modified GPU-visible memory"
        );

        // The exact group grid executes. A total-thread reshape would map a
        // different (x, y, z), so every expected index must match.
        let reply = transport
            .submit_batch(&valid, &allocations)
            .expect("valid 3D batch should complete");
        assert_eq!(reply.status, ReplyStatus::Ok);
        for (index, &value) in work[..GRID_DWORDS].iter().enumerate() {
            assert_eq!(value, index as u32 + 1, "mismatch at grid dword {index}");
        }
        assert!(
            work[GRID_DWORDS..].iter().all(|&value| value == SENTINEL),
            "3D dispatch wrote outside the exact group grid"
        );
        assert_eq!(GRID_DEPTH_STRIDE, 90);
        assert_eq!(GRID_WIDTH, 15);

        transport
            .unregister_buffer(buf_id)
            .expect("unregister_buffer should succeed");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn c_api_roundtrip_opens_engine_and_queries_device() {
        let mut engine = std::ptr::null_mut();
        // SAFETY: `engine` is a valid stack pointer for the C API out parameter.
        let st = unsafe { crate::macos::c_api::vm_engine_create(&mut engine) };
        assert_eq!(st, VmStatus::Ok);
        assert!(!engine.is_null());

        let mut info = VmDeviceInfo::default();
        // SAFETY: `engine` was created above and `info` is valid/writable.
        let st_info = unsafe { crate::macos::c_api::vm_device_info(engine, &mut info) };
        assert_eq!(st_info, VmStatus::Ok);
        assert_eq!(info.has_unified_memory, 1);

        // SAFETY: `engine` was created by `vm_engine_create` and is destroyed once here.
        unsafe { crate::macos::c_api::vm_engine_destroy(engine) };
    }
}
