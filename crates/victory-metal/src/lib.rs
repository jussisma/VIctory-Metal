//! Wine-decoupled native Apple Metal backend for VIctory.
//!
//! Designed with strict separation between the Metal GPU engine and any OS/Wine
//! transport layer:
//! - [`abi`] defines stable `#[repr(C)]` status codes, handles, and device info.
//! - [`transport`] defines [`MetalTransport`], implemented in-process on macOS
//!   without any Wine, D3D12, or Windows dependencies.
//! - On macOS (`target_os = "macos"`), [`macos::device::DirectMetalTransport`]
//!   provides real `MTLDevice`, `newBufferWithBytesNoCopy` unified-memory buffers,
//!   compute dispatch, `libmetalirconverter.dylib` DXIL compilation, and [`macos::c_api`].

pub mod abi;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod transport;

pub use abi::{VmBufferId, VmDeviceInfo, VmDxilCompileOutput, VmLibraryId, VmPipelineId, VmStatus};
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
