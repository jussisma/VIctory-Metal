# Metal Shader Converter argument-buffer binder — integration contract

Status: implemented in `crates/victory-metal/src/macos/msc.rs` and
`DirectMetalTransport::dispatch_msc_uav_1d`. CPU validation only; the hardware
test is compiled but **not executed** because the native ARM64 GPU is reserved by
the IPC shader gate and the mesh gate.

## 1. Provenance and scope

The only supported converted shape is the one observed end-to-end by the
synthetic harness `tools/macos/metal-poc/arm64-dxil.sh` (DS-DXIL, commit
`fcb87ee1`):

```text
HLSL root signature : RootFlags(0), UAV(u0)
DXIL                : RWStructuredBuffer<uint> gOut : register(u0)
                      [numthreads(64,1,1)] gOut[tid] = tid*3 + 7
Metal binding       : argument buffer bound at buffer(2)
UAV payload         : 8-byte device address of gOut at argument offset 0
Residency           : encoder useResource(gOut, read|write)
```

Observed converter facts: `dxc -T cs_6_0`, `stage=8` (compute), metallib 4180
bytes, 1024 GPU-verified words on Apple M4 Max. This module does **not** probe or
guess root layouts for arbitrary titles. A different converted layout requires a
new verified observation and an explicit extension, not a heuristic.

## 2. Rust API

```rust
use victory_metal::{DirectMetalTransport, MetalTransport};
use victory_metal::macos::msc::{MscUavDispatch, MscUavLayout};

let pipeline = transport.load_metallib(&metallib_bytes, "main")?;
let request = MscUavDispatch {
    pipeline,
    uav: buffer_id,          // VmBufferId from register_zerocopy
    offset,                  // u64 byte offset inside the registered backing
    length,                  // u64 byte length of the UAV range
    threads,
    threads_per_group,
    layout: MscUavLayout::DEFAULT, // buffer(2), address offset 0
};
transport.dispatch_msc_uav_1d(&request)?;
```

Constants:

| Item | Value |
| --- | --- |
| argument buffer length | 32 bytes (`MSC_ARGUMENT_BUFFER_LENGTH`) |
| argument buffer Metal index | 2 (`MSC_DEFAULT_ARGUMENT_BUFFER_INDEX`) |
| UAV device-address offset | 0 (`MSC_DEFAULT_UAV_ADDRESS_OFFSET`) |
| encoded address size | 8 bytes little-endian (`MSC_DEVICE_ADDRESS_SIZE`) |
| valid Metal buffer indices | `0..31` exclusive (`MSC_MAX_BUFFER_ARGUMENT_INDEX`) |

`MscUavLayout::new(index, address_offset)` accepts only an index `< 31` and an
8-byte-aligned offset with `offset + 8 <= 32`; every other combination returns
`MetalError::InvalidArg` before any Metal object exists. Fields are private, so
an invalid layout cannot be constructed.

## 3. IR-BRIDGE / registration sequence

1. Resolve the guest `(allocation_id, generation)` to a live host backing and a
   `VmBufferId`. Generation validation and backing lifetime stay in the
   registration layer; the binder only re-validates the byte range against the
   registered length.
2. Do not mutate or duplicate the converted ABI in the bridge: the binder owns
   the argument buffer allocation, address encoding, `setBuffer` at the layout
   index, and the `useResource:usage:(Read|Write)` declaration. A caller must not
   also bind slot 2 or write the GPU address.
3. Pass the guest UAV byte range as `offset`/`length` inside the registered
   backing. Rules, all checked before any command buffer is created:
   - `length != 0` and `length <= wire::MAX_RESOURCE_LEN`;
   - `offset % 4 == 0` (Metal buffer-offset ABI minimum);
   - `offset + length` does not overflow and fits the host address space;
   - `offset + length <= registered_len` (the length reported by
     `register_zerocopy`).
4. Pass an explicit `threads_per_group`. Widths above the pipeline's
   `maxTotalThreadsPerThreadgroup` are rejected with `MetalError::InvalidArg`,
   never clamped, because clamping changes threadgroup indices.
5. The call is synchronous. It holds the internal registry lock through
   `waitUntilCompleted`, keeps the pipeline and registered backing alive, and
   owns the argument buffer locally until completion. Returning `Ok` means the
   GPU work completed; the guest backing may then be released by the caller.
6. Error mapping suitable for the wire reply layer:
   - `MetalError::NotFound` → unknown/stale pipeline or buffer handle
     (`ReplyStatus::UnknownHandle` equivalent);
   - `MetalError::InvalidArg` → reject the submission before GPU work;
   - `MetalError::InternalError` → GPU execution fault
     (`ReplyStatus::ExecutionFailed` equivalent).

Pipelines come from `DirectMetalTransport::load_metallib(bytes, entry)`. On
ARM64 the conversion itself must run in the separate x86_64 Rosetta converter
host (Apple's `libmetalirconverter.dylib` is x86_64-only); the ARM64 renderer
receives bytes only. `compile_dxil` remains available for the legacy x86_64
process path and is not used by this binder.

## 4. Explicit non-goals

This contract covers exactly one UAV, one top-level argument buffer, no root
constants, no CBV/SRV, no descriptor tables, no textures/samplers, no graphics
stages and no cross-process transfer. It adds no new `wire` opcode. The legacy
paths — `MetalTransport::dispatch_compute_1d`, `submit_batch`, the version-1 wire
codec, `compile_msl`, `compile_dxil` and the C ABI exports — are unchanged.

## 5. Validation

CPU-only, no GPU execution:

```sh
cargo test -p victory-metal --target aarch64-apple-darwin msc::tests -- --test-threads=1
cargo test -p victory-metal --target aarch64-apple-darwin --no-run
cargo clippy -p victory-metal --target aarch64-apple-darwin --all-targets -- -D warnings
git diff --check
```

The six `msc::tests` cases cover the default layout, index and address-offset
bounds, the registry range contract (zero length, misalignment, overflow,
out-of-range end, `MAX_RESOURCE_LEN`) and little-endian address encoding.

Hardware test (do not run while the GPU gate is held):

```sh
VICTORY_MSC_METALLIB=/path/to/arm64-dxil/main.metallib \
  cargo test -p victory-metal --target aarch64-apple-darwin \
  msc_uav_argument_buffer_hardware_dispatch -- --ignored --test-threads=1
```

With `VICTORY_MSC_METALLIB` set, the test expects the synthetic converted shader
(entry `main`, `gOut[tid] = tid*3+7`) and verifies all 1024 words. Unset, it
compiles an ABI-equivalent MSL fixture (`buffer(2)` argument buffer with a
`device uint *` at id 0) and verifies the same values, so the binder path is
still exercised without an x86_64 converter process. The test is `#[ignore]`d and
is not part of the default suite.
