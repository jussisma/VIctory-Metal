# Development log — native ARM64 Metal renderer

Chronological engineering record for the native ARM64 Metal path in
`VIctory-Metal`, Cycles 01–03. It records what was built, what ran on hardware,
and what did not work. It contains no game assets, shader bytes, cache contents,
logs, prompts or credentials, and it names no title beyond what the repository
already records.

## Scope

Proven (synthetic, hardware): a Win64 guest can reach an ARM64 Metal device
through a versioned command protocol, encode compute/graphics/mesh work on one
ordered `MTLCommandBuffer`, draw exact pixels, present an IOSurface frame, and
do all of it with zero D3DMetal modules loaded in the measured process.
Rendering is proven against synthetic lists, synthetic shaders and synthetic
sizes derived from the title's diagnostic census.

Not proven: any frame of the title itself. The title still runs on D3DMetal.
No geometry, material, lighting or streaming data from the title has been
rendered on this path, and no frame time on this path has been measured.

Status at `210df67c`: the synthetic vertical slice is complete and
GPU-verified; the production guest transport and resource binding that a title
needs are still missing. The title is not on this path.

Hardware for every hardware claim below: Apple M4 Max, macOS 27.0 (26A428),
Metal 4. All GPU suites were run serialized with `--test-threads=1` on an
exclusive device (../.swarm/reports/architect-sync-03.md,
docs/ARM64-RENDERER.md).

All file paths in this document are relative to the `VIctory-Metal` repository
root; `../.swarm/` is the sibling orchestration state directory, outside the
tracked tree.

## Chronology

| Cycle | Date (2026) | Base → head | Headline | Gate |
| --- | --- | --- | --- | --- |
| 01 | 2026-11-26 | `1da600cf` → `25fe0caf` | Native device, wire v2, ring, present, first Wine→ARM64 synthetic frame | PASS (synthetic only) |
| 02 | 2026-11-26 | `25fe0caf` → `5d4875c0` | Wire v3 stateful stream, mixed encoder, root-signature ABI, mesh census | PARTIAL (mesh draw refused) |
| 03 | 2026-11-26 → 2026-11-27 | `5d4875c0` → `210df67c` | Mesh pipeline, packed mesh pair on hardware, production `submit_v3` slice, wave64 bucket lowering | PARTIAL (production sink disconnected) |
| 04 | 2026-11-27 | `210df67c` → `79a5e039` | Async fence page, resource/PSO registries, `ExecuteIndirect`+blit, shader pool/archive, live client, AAA synthetic gate | PASS (synthetic AAA-shaped) |
| 05 | 2026-11-27 | `79a5e039` → `11b2fa66` | Live D3D12 device/lists/fences/present, guest-owned PSOs, per-stage libraries, native window | PASS (live gate, 120 frames) |
| 06 | 2026-11-27 | `11b2fa66` → cycle-06 head | Multi-entry tables, AS+MS+PS, in-pass indirect mesh, fail-closed wave64, bounded xmem GC, core render states, one-command launcher | PASS (pre-boot gate, see 6.3) |

Base and head values: `1da600cf` and `210df67c` from
`integration.base` / `integration.head` in ../.swarm/state.json. `25fe0caf` is
the Cycle 01 head and Cycle 02 base in
../.swarm/reports/architect-sync-01.md and -02.md; `5d4875c0` is the Cycle 02
head and Cycle 03 base in ../.swarm/reports/architect-sync-02.md and -03.md.
Dates are the commit dates of those heads.

## Cycle 01 — device path and the first synthetic frame

### Objective

Stand up the Wine-independent `victory-metal` crate and prove on hardware that a
Win64 guest can drive an ARM64 Metal device through a versioned protocol and
present a frame, with no D3DMetal in the guest.

### Decisions

- **ADR-001** — role-based, model-agnostic orchestration: role is assigned by
  the launcher, never inferred from the model, so planning, execution and review
  stay auditable and substitutable (../.swarm/decisions.json).
- **ADR-002** — close the preserved 3D-dispatch, xmem byte-range and
  compute-threadgroup branches before the synthetic entry: those three were the
  direct blockers for a synthetic draw list reaching ordered native submission,
  and the WIP branches already held bounded implementations
  (../.swarm/decisions.json).
- **ADR-003** (file ownership) — pre-declared owner modules keep ten concurrent
  writers off `lib.rs` and `macos/mod.rs` (../.swarm/decisions.json).
- **ADR-004** — exactly-once keyed by `(list_id, list_generation)`, not by
  session sequence: a retry after an uncertain execution must neither duplicate
  GPU work nor be refused outright (../.swarm/decisions.json).

### What changed

`crates/victory-metal/src/{cmdlist,wire}.rs`,
`crates/victory-metal/src/macos/{render,present,views,cache}.rs`,
`crates/victory-metal/src/frame_e2e.rs`; `crates/victory-gpu/src/backend.rs`
and `crates/victory-gpu/src/native_metal/producer.rs`; mesh packing and wave
lowering in
`crates/victory-shader/src/emit/{mesh_pack.rs,lower.rs,lower/barriers.rs,lower/halves.rs}`;
the xmem share provider in `crates/victory-hle/src/modules/xmem/share.rs`; and
`tools/macos/metal-poc/synthetic-frame.sh`. Module names per
../.swarm/reports/architect-sync-01.md and ../.swarm/decisions.json.

### Evidence

| Check | Result | Source |
| --- | --- | --- |
| Win64 diagnostic build + all-targets checks | exit 0 (4/4) | ../.swarm/reports/architect-sync-01.md |
| `victory-metal` aarch64, serialized | 126 passed / 0 failed / 3 ignored | ../.swarm/reports/architect-sync-01.md |
| `victory-shader` aarch64 | 215 passed (+3 integration) / 0 failed | ../.swarm/reports/architect-sync-01.md |
| `victory-hle` aarch64 | 81 passed / 0 failed | ../.swarm/reports/architect-sync-01.md |
| `victory-gpu` | 105 passed / 0 failed / 1 ignored | ../.swarm/state.json (`integration.final_gate`) |
| `git diff --check` | clean | ../.swarm/reports/architect-sync-01.md |

GPU proofs on the M4 Max: 22 new live-device tests — render 5, present 5, frame
2, views 6, device 4 — asserting pixels, formats, depth planes and in-flight
refusal (../.swarm/reports/architect-sync-01.md). Present latency on a 64×64
clear was samples `[5.96 ms, 0.216 ms, 0.185 ms]`, median `0.216 ms`, first
sample inflated by queue warmup (../.swarm/reports/NATIVE-PRESENT-1.md).

End-to-end: `tools/macos/metal-poc/synthetic-frame.sh` exited 0. The Wine guest
submitted an indexed draw plus a compute dispatch; the host drew
`centre=0,255,0,255 pixels=4096`, returned the computed value `25` through the
guest alias, and refused a replayed list identity as `AlreadyExecuted` with the
alias byte-identical. The guest loaded no D3DMetal module
(../.swarm/state.json `integration.e2e_frame_gate`;
../.swarm/reports/architect-sync-01.md).

### What stayed unproven

- It is one synthetic frame, not an FPS measurement, and not title geometry or
  streaming (../.swarm/state.json `integration.e2e_frame_gate.caveat`).
- The draw shape is host-fixed: binding 0 is the index buffer, `Uint32`,
  triangles, one 64×64 target. Wire v2 cannot carry attachments, topology,
  index type or input layout (../.swarm/reports/architect-sync-01.md).
- Mixed compute+graphics lists are refused outright
  (../.swarm/reports/architect-sync-01.md).
- No production guest producer: `victory-gpu` still records to host D3D12 and
  nothing publishes a title list into the ring
  (../.swarm/reports/architect-sync-01.md).
- Shader gates are CPU-only at this point: 19 of 21 observed mesh pairs refused,
  111 compute rejects unclassified (../.swarm/reports/architect-sync-01.md).

## Cycle 02 — Wire v3, mixed encoding, mesh census

### Objective

Replace the fixed v2 preamble with a stateful command stream, permit mixed
compute/graphics/copy work on one ordered command buffer, and classify the two
hardware-stage gaps that blocked mesh and compute coverage.

### Decisions

- **ADR-006** — permit mixed compute, graphics and copy in one list, segmented
  by encoder class on one `MTLCommandBuffer`, with barriers only where the list
  asks: one encoder per command cannot express the passes a title records
  (../.swarm/decisions.json).
- **ADR-007** — upgrade the protocol to Wire v3 (stateful command stream, up to
  8 colour attachments plus depth/stencil, full draw state, present ordering),
  keeping v2 decodable (../.swarm/decisions.json).
- **ADR-008** — run the native producer and the shader unblocking concurrently
  behind a hard `VICTORY_BACKEND=metal` switch: a partial native path must fail
  loudly, never silently fall back to D3DMetal (../.swarm/decisions.json).

The directive's proposed ADR-003/004/005 collided with existing IDs and were
recorded as ADR-006/007/008 instead
(../.swarm/reports/architect-sync-02.md).

### What changed

`crates/victory-metal/src/wire_v3.rs`, `crates/victory-metal/src/ipc_host.rs`,
`crates/victory-metal/src/macos/render.rs` (contiguous encoder segmentation),
`crates/victory-metal/src/macos/device.rs` (`submit_v3`),
`crates/victory-metal/src/macos/bindings.rs` (root-signature ABI),
`crates/victory-metal/src/macos/irconverter.rs` (reflection JSON),
`crates/victory-metal/src/macos/present.rs` (`NativeSwapchain`);
`crates/victory-shader/src/emit/{mesh_pack.rs,lower.rs,lower/barriers.rs,lower/halves.rs,wave_audit.rs}`;
`crates/victory-gpu/src/native_metal/producer.rs`.

### Evidence

| Check | Result | Source |
| --- | --- | --- |
| Win64 diagnostic build + all-targets checks | exit 0 | ../.swarm/reports/architect-sync-02.md |
| `victory-metal` aarch64, serialized | 177 passed / 0 failed / 5 ignored | ../.swarm/reports/architect-sync-02.md |
| `victory-shader` aarch64 | 227 passed / 0 failed / 2 ignored (+3 integration) | ../.swarm/reports/architect-sync-02.md |
| `victory-gpu` | 121 passed / 0 failed / 1 ignored | ../.swarm/reports/architect-sync-02.md |
| `git diff --check` | clean | ../.swarm/reports/architect-sync-02.md |

GPU proofs on the M4 Max:

- Mixed encoder, `macos::device::tests::v3_mixed_segments_one_command_buffer_mrt_depth_scissor_and_checksum`:
  `pixels=4096`, `EncoderStats { command_buffers: 1, compute_encoders: 2,
  render_encoders: 2, blit_encoders: 1, barriers: 1 }`, `checksum=0xf9f70000`;
  two-colour MRT plus depth, scissor on pass two, depth-rejected draw absent,
  replay `AlreadyExecuted` with zero encoders
  (../.swarm/reports/MIXED-ENCODER-3D-2.md).
- Root-signature ABI: `root_signature_abi_compute_hardware` asserted 64 dwords
  exactly `7.75` (root constants, root CBV/SRV/UAV, texture and sampler
  tables); `root_signature_abi_graphics_hardware` asserted 4 pixels exactly
  `[255, 0, 128, 255]` (../.swarm/reports/ROOT-BINDINGS-ABI-2.md).
- Present: `macos::present::tests::two_surfaces_present_distinct_frames_with_interval_and_refusal_discipline`,
  two frames exact BGRA, in-flight and refused-sink discipline
  (../.swarm/reports/SWAPCHAIN-NATIVE-PRESENT-1.md).
- Wave: six lane-tagged fixtures matched console semantics through the converted
  root-signature ABI, commit `c6843aec`
  (../.swarm/reports/architect-sync-02.md,
  ../.swarm/state.json `wave_5.tasks.WAVE-GPU-PROOF`).
- Wire v3: 25 opcodes, 13 `wire_v3` tests, the 45 v2 tests unchanged; producer
  ring 16 tests (../.swarm/reports/architect-sync-02.md).

Census work: all 19 mesh-pixel refusals classified as the same Metal-aligned
`>32 KiB` budget — 7×33,280 B, 9×33,792 B, 3×35,328 B — with zero linkage
failures; general packing implemented and one fixture proven to fit
33,280 → 32,768 B (../.swarm/reports/architect-sync-02.md,
../.swarm/reports/MESH-19-UNBLOCK-2.md). The 111 compute rejects were
classified into the buckets recorded in the shader scorecard below
(../.swarm/reports/WAVE64-111-AUDIT-FIX-2.md).

### What stayed unproven

- No mesh draw: `DirectMetalTransport` refused `SetMeshPipeline`/`DispatchMesh`
  and `load_metallib(mesh)` failed with `XPC_ERROR_CONNECTION_INTERRUPTED`
  (../.swarm/reports/architect-sync-02.md). The packing proof was CPU layout
  plus IRConverter translation only, no texels
  (../.swarm/reports/MESH-19-UNBLOCK-2.md).
- The wave audit's own GPU test wrote nothing because the emitted root table was
  not bound by `dispatch_compute_1d`; it was reported blocked, not passed
  (../.swarm/reports/WAVE64-111-AUDIT-FIX-2.md). The lane-tagged proof came
  from the separate `WAVE-GPU-PROOF` task through the root-signature ABI.
- No production guest path: `WIRE-V3-HOST-BRIDGE` was unwritten, and `list.rs`
  could not resolve descriptor-table or root-view heap addresses
  (../.swarm/reports/architect-sync-02.md).
- No title frame under `VICTORY_BACKEND=metal`; 105 of 111 compute rejects
  remained refused by design (../.swarm/reports/architect-sync-02.md).

## Cycle 03 — mesh pipeline, packed pair on hardware, production slice

### Objective

Finish the mesh stage, connect a real production-shaped transport slice to
`submit_v3`, and lower the highest-frequency wave64 buckets with hardware proof.

### Decisions

- **ADR-009** — implement the native mesh pipeline
  (`MTLMeshRenderPipelineDescriptor`, `SetMeshPipeline`, `DispatchMesh`) and
  unblock all 19 refused mesh+pixel size classes by pruning provably unread
  output rows while preserving export lookup, OSG1, PSV0 and pixel linkage
  (../.swarm/decisions.json).
- **ADR-010** — close the remaining guest→host gaps: v3 decode plus ledger in
  `ipc_host`, `xmem::share` descriptor resolution, and a `VICTORY_BACKEND=metal`
  queue/present path with no silent D3D12 fallback. Recorded in the directive
  but absent from `decisions.json` (see Known gaps);
  source: ../.swarm/prompts/architect-directive-03.md and
  ../.swarm/state.json `wave_6.adrs`.
- **ADR-011** — lower the three highest-frequency wave64 buckets (54 lane-mask,
  30 unbarriered one-wave LDS, 12 cross-half reduction, plus 8 first-lane) with
  sound wave-uniformity proofs and 64-lane lane-tagged hardware fixtures; CPU
  tests are not acceptance (../.swarm/decisions.json).

### What changed

`crates/victory-metal/src/ipc_host.rs` (v3 envelope, kind 5),
`crates/victory-metal/src/macos/{mesh.rs,objc.rs,render.rs}`,
`crates/victory-metal/src/macos/device.rs`;
`crates/victory-shader/src/emit/{mesh_pack.rs,lower.rs,lower/barriers.rs,lower/halves.rs,wave_audit.rs}`;
`crates/victory-gpu/src/native_metal/{queue.rs,present.rs}`;
`crates/victory-hle/src/modules/xmem/share.rs` (guest range resolver);
`crates/victory-metal/tests/{mesh_pipeline_gpu,mesh_pair_gpu,producer_e2e}.rs`.

### Evidence

| Check | Result | Source |
| --- | --- | --- |
| Win64 diagnostic build + all-targets checks | exit 0 | ../.swarm/reports/architect-sync-03.md |
| `victory-metal` aarch64, serialized | 188 passed / 0 failed / 5 ignored | ../.swarm/reports/architect-sync-03.md |
| `victory-shader` aarch64 | 245 passed / 0 failed / 2 ignored | ../.swarm/reports/architect-sync-03.md |
| `victory-gpu` | 136 passed / 0 failed / 1 ignored | ../.swarm/reports/architect-sync-03.md |
| `git diff --check` | clean | ../.swarm/reports/architect-sync-03.md |

GPU suites on the M4 Max (../.swarm/reports/architect-sync-03.md):

- `wave_lower_gpu` 4 passed: `lowered_wave_patterns_run_like_the_console_on_metal`,
  `lds_barriers_run_like_the_console_on_metal`,
  `reductions_across_halves_run_like_the_console_on_metal`,
  `first_lane_and_lane_reads_across_halves_run_like_the_console_on_metal`
  (../.swarm/reports/WAVE64-LDS-REDUCE-1.md; names from
  crates/victory-metal/tests/wave_lower_gpu.rs).
- `mesh_pack_gpu` 1 passed: paired fixture in the 33,280 B class, aligned
  33,280 → 32,768 B (1,280 B packed), both stages IRConverter-translated, no
  draw (../.swarm/reports/MESH-19-PACK-ALL-1.md).
- `mesh_pipeline_gpu` 3 passed: `irconverter_mesh_pair_draws_exact_pixels`
  (256 px exact), over-budget descriptor refused cleanly, and the XPC failure
  isolated to a mesh stage fed to `newComputePipelineState`
  (crates/victory-metal/tests/mesh_pipeline_gpu.rs;
  ../.swarm/reports/architect-sync-03.md).
- `mesh_pair_gpu` 2 passed: `packed_pair_draws_exact_pixels` —
  33,280 B declared, 1,280 B packed, 256 px on 16×16 RGBA8Unorm, columns
  `[0,0,0,0,0,0,0,0,16,48,80,112,143,175,207,239]`, checksum
  `0x98e4ed7c5ed77c00`; `unpacked_pair_above_32k_is_refused` never draws
  (../.swarm/reports/MESH-HW-PAIR-2.md).
- `producer_e2e` 1 passed: 2×32×32 MRT pixels exactly (`centre0
  [0,255,0,255]`, `centre1 [255,0,255,255]`), depth-rejected draw absent,
  64-dword compute alias, two replay refusals, unsupported opcode fails with
  recovery; `pixels=1024` total. The Wine guest probe printed
  `GUEST_V3_PASS ... d3dmetal_modules=0 method=psapi_EnumProcessModules`,
  `session complete submissions=4 allocations=1`
  (../.swarm/reports/PRODUCER-E2E-GATE-2.md).

Supporting CPU counts: Wire v3 host bridge 21 `ipc_host` tests plus 5 bridge
tests; `GPU-QUEUE-PRESENT-WIRE` added 6 tests; `MESH-19-PACK-ALL` proved the
33,280/33,792/35,328 B classes fit under 32,768 B
(../.swarm/reports/architect-sync-03.md).

### What stayed unproven

- The 19/19 mesh pairs remain inferred from size classes; no title shader bytes
  were available, and the hardware pixel proof is one 33,280 B pair
  (../.swarm/reports/architect-sync-03.md).
- `96/111` is shape coverage measured against the census, not a re-measured
  title count; adversarial per-lane and cross-half cases still refuse
  (../.swarm/reports/architect-sync-03.md).
- The production sink is still `DisconnectedSink` and `GuestIpcClient` has no v3
  encoder, so title lists never reach the host
  (../.swarm/state.json `global_blockers`).
- IPC cannot register render-target textures; `submit_v3` has no graphics PSO
  from converted metallibs and no IRConverter argument-buffer binder
  (../.swarm/state.json `global_blockers`).
- No frame capture or fallback counter on a real title run, and no measured
  frame time on this path (../.swarm/reports/architect-sync-03.md).

## Shader scorecard

Both tables come from the historical diagnostic census taken from one
diagnostic run of the title (`victory-mesh-a5991572`), as recorded in
../.swarm/reports/WAVE64-111-AUDIT-FIX-1.md and
../.swarm/reports/MESH-19-UNBLOCK-2.md. They are not from a fresh run, and
the repository contains no shader bytes that would let the counts be
re-attributed to specific pipelines
(../.swarm/reports/WAVE64-111-AUDIT-FIX-1.md,
../.swarm/reports/WAVE64-LDS-REDUCE-1.md).

### Compute rejects — 111 refusals

| Bucket | Count | Lowered in cycle | Hardware proof |
| --- | --- | --- | --- |
| Lane mask read as data (wave64 on a 32-lane host) | 54 | 03, `f71b0b3d` | 64-lane fixtures `live_mask_read`, `literal_mask_read`, `mask_after_a_per_wave_loop`; `wave_lower_gpu` 4/4 |
| LDS one console wave64 shares without barriers | 30 | 03, `d42b7694` | `lds_barriers_run_like_the_console_on_metal` |
| Wave reduction over both 32-lane halves | 12 | 03, `d42b7694` | `reductions_across_halves_run_like_the_console_on_metal` |
| First-lane read of a value that may differ between halves | 8 | 03, `d42b7694` (`guard_alike` widened; the CPU rule was tested in 02, `8a5429d3`) | `first_lane_and_lane_reads_across_halves_run_like_the_console_on_metal` |
| Lane read that may cross 32-lane halves | 3 | 02 rule `8a5429d3`, extended and proven in 03 `d42b7694` | same fixture |
| Lane mask read as data (non-wave variant) | 2 | 03, `f71b0b3d` | same mask fixtures; buckets 1+6 counted as 56 of 111 |
| Screen-probe ray activation skipped without rays | 1 | not a lowering (diagnostic flag, `VICTORY_BVH_NO_HIT=1`) | n/a |
| Descriptor read only as data | 1 | still refused | n/a |

Total 111 (../.swarm/reports/WAVE64-111-AUDIT-FIX-1.md). The `96/111` figure
quoted by Cycle 03 (ADR-011 and ../.swarm/reports/architect-sync-03.md) is
the three-bucket sum 54 + 30 + 12; Cycle 03 additionally quotes the mask bucket
as 56, which includes the two non-wave mask rejects. The GPU proofs are
lane-tagged 64-lane compute fixtures; they prove the lowering shapes, not the
title shaders.

### Mesh + pixel pairs — 19 refused of 21 observed

| Refused size class | Count | Classified | Packed proof | Hardware proof |
| --- | --- | --- | --- | --- |
| 33,280 B aligned | 7 | 02, `186b39dc` | 03, `b30fd36c` (33,280 → 32,768 B) | 03, `189bd9b4` (→ 1,280 B, 256 px exact, checksum `0x98e4ed7c5ed77c00`) |
| 33,792 B aligned | 9 | 02, `186b39dc` | 03, `b30fd36c`, fits per ../.swarm/reports/architect-sync-03.md | inferred only, no title bytes |
| 35,328 B aligned | 3 | 02, `186b39dc` | 03, `b30fd36c`, fits per ../.swarm/reports/architect-sync-03.md | inferred only, no title bytes |

Sources: ../.swarm/reports/architect-sync-02.md,
../.swarm/reports/MESH-19-PACK-ALL-1.md,
../.swarm/reports/MESH-HW-PAIR-2.md. Two further non-pair diagnostics were
recorded (one standalone mesh output at 34,816 B, one "no host pipeline") and
are not counted among the 19 (../.swarm/reports/MESH-19-UNBLOCK-2.md).

## Reproducible commands

Run from the repository root (`VIctory-Metal`). Use the Homebrew rustup first;
the keg is not on `PATH` by default.

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

# 1. Win64 diagnostic build of the full title binary.
cargo build --locked --profile diag --target x86_64-pc-windows-gnullvm --package victory --features diagnostics
# expected: "Finished ...", exit 0

# 2. Renderer crate, native ARM64, serialized (exclusive GPU for the hardware tests).
cargo test --locked -p victory-metal --target aarch64-apple-darwin -- --test-threads=1
# expected at 210df67c: "test result: ok. 188 passed; 0 failed; 5 ignored"

# 3. Shader compiler, native ARM64.
cargo test --locked -p victory-shader --target aarch64-apple-darwin
# expected at 210df67c: "test result: ok. 245 passed; 0 failed; 2 ignored" (plus 3 integration)

# 4. Guest-layer and GPU-layer counts recorded for the same head.
cargo test --locked -p victory-gpu
# expected at 210df67c: "test result: ok. 136 passed; 0 failed; 1 ignored"

# 5. Whitespace and conflict-marker check.
git diff --check
# expected: no output
```

Hardware suites that are `#[ignore]`d by default and must be run one at a time
on a free device:

```sh
cargo test --locked -p victory-metal --target aarch64-apple-darwin --test mesh_pair_gpu -- --ignored --test-threads=1
# expected: "test result: ok. 2 passed; 0 failed"

cargo test --locked -p victory-metal --target aarch64-apple-darwin --test wave_lower_gpu -- --ignored --test-threads=1
# expected: "test result: ok. 4 passed; 0 failed"

cargo test --locked -p victory-metal --target aarch64-apple-darwin --test producer_e2e -- --ignored --test-threads=1
# expected: "test result: ok. 1 passed; 0 failed"
```

Synthetic probes, each self-contained and asset-free:

| Script | What it proves | Expected output (one line) |
| --- | --- | --- |
| `tools/macos/metal-poc/build-and-run.sh` | zero-copy `newBufferWithBytesNoCopy` + DXIL→`MTLLibrary` under Rosetta | `PASS Test 1 (Zero-Copy Unified Memory): ... verified=1024 dwords` (Test 2 needs a cached or supplied `.dxil`) |
| `tools/macos/metal-poc/arm64-shared-buffer.sh` | Win64 client ↔ ARM64 GPU over a shared section, plus a GPU blit triangle | `PASS verified_words=2293760 app_copy_bytes=0`, `PASS triangle_verified_pixels=4096` |
| `tools/macos/metal-poc/xmem-arm64.sh` | real HLE xmem alias write read by an ARM64 Metal shader at offset 64 KiB | `PASS xmem-arm64 offset=65536 sum=<v> xor=<v> sentinel=<v> guard=<v>` |
| `tools/macos/metal-poc/xmem-ipc.sh` | real guest IPC client → native host, two ordered dispatches, ticket registration | `PASS xmem-ipc wine-guest=1 guest-alias=1 ordered-dispatches=2 ...` |
| `tools/macos/metal-poc/arm64-dxil.sh` | DXIL→metallib in an x86_64 Rosetta process, executed in an ARM64 process | `PASS stage=compute root_sig=embedded-dxil binding=argument-buffer(2)/uav0 gpu_verified=1024` |
| `tools/macos/metal-poc/synthetic-frame.sh` | Wire v3 guest probe plus `submit_v3` MRT/depth graphics and converted compute | `PASS synthetic-frame mode=host-v3 graphics=MRT+depth dxil=rosetta pixels=1024` |

The last script also prints
`PARTIAL synthetic-frame wine-guest-consumes-victory-gpu=0` and
`NOTE one frame; this is not an FPS measurement.` —
both are expected at Cycle 03 (tools/macos/metal-poc/synthetic-frame.sh).
The shared-buffer probe's 2,293,760 words are 140 dispatches of 16,384 words
(docs/ARM64-RENDERER.md). Scripts print their own pass lines; do not run two at
once on one GPU.

## Cycle 04 — transport, registries, indirect work and the AAA synthetic frame

Base `210df67c`, code head `6d14b9db`, scaffold `a43c8dee`. Decisions ADR-012
through ADR-016 (../.swarm/decisions.json). Hardware: Apple M4 Max, macOS 27.0
(26A428); every GPU run serialized with `/tmp/victory-gpu.lock` and
`--test-threads=1`.

Result: a Wine Win64 guest drives an ARM64 Metal host through 120 measured
frames of a synthetic AAA-shaped pipeline (compute culling -> indirect draw ->
MRT/depth graphics -> post-FX compute -> present) with exact pixels, a replay
guard, a shared fence page, zero D3DMetal and no leaked processes. The frame is
synthetic: no title list, no title assets, no title frame time.

### 4.1 Transport and fence (ADR-012)

`FencePage` is one 4 KiB `#[repr(C)]` page (`completed_fence`, `submitted_fence`,
epoch) mapped `MAP_SHARED`, published after a real v3 submission returns `Ok`
and never republished by a replay or a refusal; `GuestTransport` is Wine-free and
`Wake` has `LoopbackTcpWake` and `UnixSocketWake` (`Unsupported` off-Unix).

- Reply-less async submissions (kind 7): one reply frame removed per
  `ExecuteCommandLists`; the transport test observes completion with a cloned
  socket and zero reply frames (`tests/transport_fence.rs`, 6 passed).
- 0-IPC fence read: 0.67 ns per atomic load, measured as a loop average in the
  benchmark guest, not as a cache-miss latency.
- Host submit is still synchronous (`waitUntilCompleted` in
  `macos/device.rs`), so the 0-IPC property is proven; the host-side
  non-blocking submit is not.

### 4.2 Resource, PSO and residency (ADR-013)

`macos/registry.rs` registers 17 formats (BC1-BC7, D32F, D32F_S8, R11G11B10F,
R10G10B10A2, colour and depth), 2D/array/cube/3D dimensions, mips and MSAA, with
RTV/DSV/SRV/UAV views and blake3 content keys; `macos/residency.rs` creates a
native `MTLResidencySet` with a documented `useResources` fallback.

- Graphics proof (`tests/submit_v3_bindings_gpu.rs`, 3 passed): two registry
  RGBA8 targets plus D32F depth, content-hashed MRT PSO, root constants and root
  CBV through the reflected ABI -> target0 `[255,128,0,255]` for all 4096 texels,
  target1 `[0,128,255,255]` for all 4096 texels; residency native = true, 4
  allocations, fallback = none.
- Compute ABI proof: content-hashed compute PSO, root constants + CBV/UAV ->
  exact 64 dwords `7..70`.
- Content-key dedup: `created=1 reused=1` for a duplicate key; one pipeline
  object per key.
- Handle validation: unknown pipeline and unknown texture return `UnknownHandle`
  with `EncoderStats::default()` and the 4096-pixel target unchanged.
- Mesh PSO construction is implemented but not hardware-proven; the mesh draw
  proof remains the Cycle 03 `mesh_pair_gpu` fixture.

### 4.3 Indirect and blit (ADR-015)

Wire v3 opcodes 40-45: indirect draw, indexed draw, mesh dispatch and compute
dispatch (optional count buffer), `CopyTextureRegion` with independent origins
and a texel box, and `ResolveSubresource` as a render-pass store action.

- Direct vs indirect indexed draw: identical exact pixels, checksum
  `0xc25466880556e425`, 256/256 pixels, 256 distinct colours; the argument
  record is written by a compute kernel and never read back.
- `CopyTextureRegion` 32x32 sub-box: 0 differing pixels, checksum
  `0xabbeaf356e39f5a5`.
- 4x MSAA `ResolveSubresource` equals a direct clear: checksum
  `0x80af855529174d25`.
- `tests/indirect_gpu.rs`, 4 passed.

### 4.4 Shader pool and archive (ADR-014)

`compiler_pool.rs` runs a bounded pool (1-8 workers from
`VICTORY_SHADER_WORKERS`, default 4) over a length-prefixed job/result framing
with the content-hashed disk cache in front; `macos/binary_archive.rs` persists
an `MTLBinaryArchive` with atomic writes and corrupt-archive recovery.

- Pool: 7 tests pass (8 concurrent id-matched jobs, cache hit without a second
  worker, crash containment, env clamping, timeout).
- `MTLBinaryArchive`: supported (`respondsToSelector:` probe true). Cold
  9.707 ms (open 0.026, PSO 0.097, store 9.495) -> warm 0.182 ms (open 0.152,
  PSO 0.016), 53.27x, archive 12,992 B; 3 tests pass.
- The control PSO without the archive costs 0.002 ms, so on this trivial kernel
  the ratio is dominated by skipping archive serialization, not by PSO
  compilation. The pool's real converter path is untested because
  `victory-irconverter` has no stdin/stdout mode.

### 4.5 Host daemon, live client and telemetry (ADR-016)

`victory_metal_host` serves the authenticated loopback session with a
machine-readable ready banner (`compute_pipeline`, `graphics_pipeline` ids),
`--once`, `--frame-gate`, `--share-dir` and `--telemetry-json`.
`crates/victory-gpu/src/native_metal/client.rs` authenticates, sends one
`KIND_SUBMISSION_V3` frame per batch and maps the reply status; the native queue
connects lazily from `VICTORY_METAL_ENDPOINT`/`VICTORY_METAL_SECRET` and still
fails explicitly without one. `telemetry.rs` records record/ring/encode/GPU
microseconds, exact nearest-rank p50/p95/p99, FPS, an unsupported-opcode
histogram and a hard `d3dmetal_fallback_count`.

- Client 11 tests, telemetry 11 tests, `native_metal` 59 tests, Windows
  all-targets clean.
- Histogram is exact on fixtures only: the producer's refusal detail is private,
  so the live path records `opcode: None` (`list.rs:234`).
- `gpu_us` is 0 on the live path: the reply carries status, not GPU time.

### 4.6 120-frame benchmark (20 warmup frames)

`tools/macos/metal-poc/aaa-frame-bench.sh`, one measured run on the integrated
head (`6d14b9db`), exit 0:

| Metric | Measured |
| --- | --- |
| Frames measured | 120 / 120 (140 submitted, 20 warmup) |
| Frame time p50 | 0.624 ms |
| Frame time p95 | 0.857 ms |
| Frame time p99 | 1.672 ms |
| Sustained FPS | 1440.68 |
| Host-side telemetry p50 / p95 / p99 | 0.116 / 0.462 / 0.527 ms |
| 0-IPC fence read | 0.67 ns per atomic load (loop average) |
| Shader pool cold vs warm | 9.707 -> 0.182 ms, 53.27x (archive, not a game shader) |
| Present checksum | `0xd5791f96cdb32383` (matches expected) |
| `AlreadyExecuted` replay guard | status 3, alias unchanged |
| D3DMetal calls / modules | 0 / 0 (host fallback count 0) |
| Leaked processes | 0 |

Stage statuses were all 0: `texture0`, `texture1`, `depth`, `surface`,
`graphics`, `present`. Two submissions per frame (281 executions including the
replay probe). This is a synthetic pipeline; the ms values are not a prediction
of title frame time.

The target (p50 <= 33.3 ms, zero D3DMetal) is met on the synthetic pipeline only.

### 4.7 Readiness for Cycle 05

- Geometry/materials/lighting/streaming on a title frame: **not demonstrated**;
  no title list has reached the native path.
- Title frame time p50 / p95: **not measured**.
- Top three blockers: (1) title command-list recording into Wire v3 and the
  production queue still installing the disconnected sink at runtime, so no
  title submission exists (`crates/victory-gpu/src/native_metal/queue.rs`);
  (2) the shader pool cannot drive the real converter (no stdio mode,
  `crates/victory-metal/src/bin/victory-irconverter.rs:29`) and 105 of 111
  historical compute rejects remain refused; (3) mesh PSOs are not yet proven
  through the live ABI, and Wine-only `xmem::share` lease tests have four
  pre-existing failures.
- Go / no-go: **go for a diagnostic title boot under `VICTORY_BACKEND=metal`
  only after the title command-list producer is wired**; expect explicit native
  failures rather than D3DMetal output, and do not treat a frame count as visual
  acceptance.

## Known gaps

State at `6d14b9db`, traced to source:

1. **No title frame on this path.** The title still runs on D3DMetal; only
   synthetic frames exist (../.swarm/state.json `goal`;
   ../.swarm/reports/architect-sync-04.md).
2. **No title command list reaches the native path.** `victory-gpu` now installs
   a lazy live `WireV3Client`, but nothing records a recorded D3D12 title list
   into Wire v3, so no title submission has been made
   (`crates/victory-gpu/src/native_metal/queue.rs`;
   ../.swarm/state.json `global_blockers`).
3. **The new resource/PSO/IPC path is synthetic-only.** Textures and surfaces
   can be registered over IPC (kinds 8/9) and `submit_v3` binds content-hashed
   PSOs and argument buffers, but no title pipeline, texture or shader has gone
   through it (`crates/victory-metal/src/ipc_host.rs`,
   `tests/submit_v3_bindings_gpu.rs`).
4. **Telemetry covers the synthetic gate only.** `d3dmetal_fallback_count`
   exists and is asserted on the benchmark, but there is no title frame capture
   and no title frame time; the >=30 FPS acceptance is unmet and unmeasured
   (../.swarm/reports/architect-sync-04.md).
5. **Shader coverage is shape coverage.** 105 of 111 compute rejects are still
   refused by design and 19/19 mesh pairs remain inferred from size classes; the
   one hardware pixel proof is a single 33,280 B pair
   (../.swarm/reports/architect-sync-03.md).
6. **Wine-backed xmem range lease tests are partly red.**
   `victory-hle` is unchanged from `210df67c` and its aarch64 host suite is
   81/81 green, but on the Windows target four `xmem::share` lease tests fail
   (release order, retained ranges, stale generations, two-submission share);
   the guard test passes under Wine. Pre-existing, not introduced by Cycle 04
   (../.swarm/logs/; ../.swarm/reports/architect-sync-04.md).
7. **Census is historical.** The 111 and 19 counts come from one diagnostic run
   and cannot be re-attributed without the shader bytes
   (../.swarm/reports/WAVE64-111-AUDIT-FIX-1.md).
8. **The D3DMetal reference is still the measured baseline.** It measured
   544–585 ms/frame (1.7–1.8 FPS) in world gameplay on `mac-turbo` `v5`; that is
   a historical lead from another tree, not a measurement of the native path
   (docs/ROADMAP-METAL-NATIVE.md).
9. **Decision ledger has holes.** `decisions.json` contains two distinct
   ADR-003 entries and no ADR-005 or ADR-010; the directive-03 text for ADR-010
   is the only record of that decision (../.swarm/decisions.json versus
   ../.swarm/prompts/architect-directive-03.md).

## Cycle 05 — live D3D12 wiring, async GPU pipelining, persistent IRConverter, native window

Base `79a5e039` (`cycle-04`). Scaffold `246b48cb`. Integration head is the
commit this section lands on; local annotated tag `cycle-05` marks it. Every
number below comes from a command run in this workspace on Apple M4 Max; a
number that could not be reproduced is labelled as such.

### 5.1 Decisions (ADR-017…022)

- **ADR-017 — async host execution.** Each Wire v3 submission is committed
  without `waitUntilCompleted`; an `addCompletedHandler` block publishes the
  contiguous completion watermark into the shared `FencePage` and retires the
  submission's pinned staging slices and leases through `RetireQueue`. Frames in
  flight are capped (`VICTORY_METAL_FRAMES_IN_FLIGHT`, default 3); backpressure
  happens at the frame boundary, never per command list.
- **ADR-018 — persistent `--stdio` converter.** `victory-irconverter --stdio`
  stays alive and answers the pool's framed jobs, so Rosetta startup is paid once
  per worker. A crashed worker is reaped and respawned; the disk cache still
  absorbs repeats.
- **ADR-019 — live `victory-gpu` D3D12 wiring.** Under `VICTORY_BACKEND=metal`,
  device connect, buffer (kind 2), texture (kind 8), surface (kind 9), versioned
  pipeline (kind 4/129), command-list recording, exactly-once
  `ExecuteCommandLists`, `Fence::GetCompletedValue` from the shared page, and
  `Present` all go through `GuestTransport`. An unimplemented call is an explicit
  native failure with an opcode histogram entry; `d3dmetal_fallback_count` stays 0.
- **ADR-020 — opt-in native presenter.** `VICTORY_METAL_WINDOW` (or `--window`)
  opens an `NSWindow` + `CAMetalLayer` and presents the registered surface
  drawable with no CPU readback. Headless `IOSurface` remains the default.
- **ADR-021 — reply semantics.** A kind-5 reply means *committed*, not
  *complete*. Completion is only the shared fence. The Cycle 04 AAA gate polled
  the fence once right after the reply and failed under the new semantics until
  it used its bounded `wait_fence` helper.
- **ADR-022 — reflection-derived ABIs.** The stdio converter returns reflection
  (`VMRES2`) and the guest builds its `VAB1` ABI from the converter JSON. Without
  it the host records bindings but binds no root arguments: the gate saw
  `dispatch_bindings=1` with a zero argument buffer before this landed.

### 5.2 Chronology of the work

| Step | Commit | Evidence |
|---|---|---|
| scaffold`retire.rs`/`window.rs` | `246b48cb` | 9 unit tests on the two seams |
| W5-XMEM-WINE-LEASE-FIX | `c8aa3182` | the four "Wine-only" failures were test isolation (fixed submission ids shared through a process-global registry), not lease lifetime; Wine `xmem::share` 12/12, full Wine 178, aarch64 81 |
| W5-WAVE64-LIVE-GAP-CLOSE | `9e774b44` | barrier split keyed to `narrow_waves` instead of `narrow_waves \|\| host_wave32` (`emit/lower.rs:853/872`); reproducible 35-program corpus 16 → 11 refusals on a 32-lane host, LDS 7 → 2 |
| W5-IRCONVERTER-STDIO-POOL | `4de3c508` | one worker served 16 jobs; cold 6.569 ms/job → warm 0.026 ms/job (254.5×); V/P/M/C translated |
| W5-NATIVE-WINDOW-PRESENT | `ca80fa25` | real window, 2 drawables acquired and presented, windowed checksum == headless, 0 D3DMetal images |
| W5-ASYNC-SUBMIT-PIPELINE | `fb9161ad` | peak in-flight 3 of cap 3, fence below the sequence while busy, 200-submission checksum, retire depth 0 after Present |
| W5-MESH-LIVE-ABI-PROOF | `c8a3bb02` | live `submit_v3` mesh draw, 256 px exact `0x98e4ed7c5ed77c00`, resident 4 |
| W5-TITLE-CMDLIST-LIVE | `04eac489` | device once-connect, texture/surface registration, `ExecuteIndirect`/`CopyTextureRegion`/`Resolve`, fence page, `Present`; `victory-gpu` 178 |
| W5-HOST-FENCE-AND-MESH-FIX | `b146b545` | completion published only by the GPU handler; content-hashed mesh PSO, mesh-stage descriptor tables, table-buffer retention (removing the retain reproduced the SIGSEGV) |
| W5-GUEST-PSO-BUFFER-REG | `e05bb51d` | kind 2 buffers idempotent through `GuestRangeResolver`, compute PSO 4/129, `CompilerPool` wired; `victory-gpu` 185 |
| orchestrator fixes | `e46a0790`, `f514bd14`, `15a51f1b` | daemon forwards fence-page attach; AAA gate waits on the fence; daemon forwards the five `register_*_pipeline*` calls the gate proved were missing |
| W5-PIPELINE-REG-GRAPHICS | `fd5a1efc` | kind 4 v2: per-stage tickets, `PipelineState`, `PipelineAbi`, root signature; v1 compute byte-compatible |
| W5-LIVE-D3D12-TITLE-GATE (first) | `df0aec78` | real API path, 120/120, p50 0.374 ms; PARTIAL on the guest's own PSO |
| W5-PSO-MULTISTAGE-LIB | `d6bb7cca` | one `MTLLibrary` per stage file, BLAKE3 key over every stage, missing function names its stage |
| W5-COMPUTE-ROOT-BINDINGS | `3aa7ff88` | compute CBV/SRV/UAV/table/constants into Wire v3 bindings, `compute_unbound` counter, dispatch→indirect ordering test |
| W5-LIVE-D3D12-TITLE-GATE-2 | `22af74d4` | 120/120, p50 0.385 ms; PARTIAL with three named blockers |
| W5-MESH-PASS-OPEN | `5f274356` | `BeginRenderPass → SetMeshPipeline → DispatchMesh → EndRenderPass` for a mesh pass opened by targets only |
| W5-STDIO-REFLECTION | `b22866a5` | `VMRES2` reflection, `compile_with_reflection`, ABI from `TopLevelArgumentBuffer`; 576 B reflection, ~6 ms/job extra |
| W5-LIVE-D3D12-TITLE-GATE-3 | `94caf219` | PASS: guest-owned PSO ids (graphics 3, compute 2, mesh 1), compute-written indirect arguments `3,1,0,0`, exact pixels, 120/120 |

### 5.3 Measured evidence

**Synthetic AAA gate after the async landing** (`tools/macos/metal-poc/aaa-frame-bench.sh`,
exit 0, 120 measured frames after 20 warmup): p50 **0.577 ms**, p95 **1.285 ms**,
p99 **1.638 ms**, **1733 FPS** at p50 (script average 1597), 281 executions,
0-IPC fence read **0.77 ns**, present checksum `0xd5791f96cdb32383` unchanged,
replay `AlreadyExecuted`, D3DMetal modules 0, leaks 0. Cycle 04's synchronous gate
was p50 0.624 ms / 1440 FPS.

**Live D3D12 gate** (`tools/macos/metal-poc/live-d3d12-gate.sh`, exit 0,
1280×720, MRT RGBA8×2 + Depth32Float, 120 measured frames after 20 warmup):
two independent runs: p50 **0.405 ms** / p95 0.768 / p99 0.793 / **2106 FPS**
(orchestrator) and p50 0.443 / p95 0.798 / p99 0.820 / 1962 FPS (worker). The graphics
(id 3), compute (id 2) and mesh (id 1) pipelines are the guest's own v2
registrations (`owner=guest_v2`), not the daemon's built-ins, and the recorded
streams name exactly those ids. The compute dispatch carries one root binding and
a barrier, the argument buffer it writes is the one `ExecuteIndirectDraw` reads
(`args_after_frames=3,1,0,0`, `unbound_dispatches=0`, linkage 140/140), the
present checksum is exactly `0xd5791f96cdb32383`, replay is `AlreadyExecuted`
with the target unchanged, `d3dmetal_fallback_count` 0, host D3DMetal images `[]`,
the unsupported-opcode histogram is empty, and the fence was observed 140 times
with 0 timeouts. Pool: 5 jobs, 16.44–31.32 jobs/s across runs, first job 150.9–296.3 ms
(the converter start dominates), subsequent 1.1–1.3 ms, disk cache 14.7–27.2×. In-flight depth is 1 because this guest waits on the
fence per frame; the 3-deep overlap is proven separately by
`async_pipeline_gpu::completion_is_published_only_after_the_gpu_finishes`.

### 5.4 Counts (integration head)

- Compiled: Win64 `diag` `victory` build; `victory-metal`/`victory-gpu`
  Windows all-targets checks; x86_64 `victory-irconverter`.
- CPU-tested: `victory-metal` lib **261 passed / 5 ignored**, `victory-shader`
  **246 + 3** integration, `victory-gpu` **205 passed / 1 ignored**,
  `victory-hle` **81** aarch64 and **178** under Wine (focused `xmem::share`
  12/12 re-run by the orchestrator).
- GPU-verified on M4 Max, serialized with `/tmp/victory-gpu.lock` and
  `--test-threads=1`: async pipeline, mesh live ABI, window presenter,
  IRConverter pool (9 tests), AAA gate, live D3D12 gate.

### 5.5 Reproduce

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
cd VIctory-Metal
cargo test --locked -p victory-metal --target aarch64-apple-darwin --lib -- --test-threads=1
cargo test --locked -p victory-shader --target aarch64-apple-darwin
cargo test --locked -p victory-gpu -- --test-threads=1
tools/macos/test-wine.sh -p victory-hle -- --test-threads=1
tools/macos/metal-poc/aaa-frame-bench.sh
tools/macos/metal-poc/live-d3d12-gate.sh
```

### 5.6 Remaining gaps (honest)

- **No title binary has run.** The gate drives the real `victory-gpu` D3D12
  entry points and the real host, but not a retail title; no proprietary asset
  was used. Geometry/materials/streaming on a title frame remain unproven, and
  the >=30 FPS game gate is still unmet.
- **In-flight depth 1 in the gate** because the harness waits per frame. The
  3-deep pipeline is proven by a separate GPU test, not by the gate.
- **Indirect mesh dispatch** (`execute_indirect_dispatch_mesh`) still records
  outside a render pass and refuses an open pass; direct `DispatchMesh` is fixed.
- **`narrow_waves` without `host_wave32`** lowers 34/35 corpus programs with a
  zero high ballot dword — a silent miscompile on the D3DMetal reference path.
  Deferred with a fail-closed recommendation; not reachable on the native
  64-lane path.
- **Placed/aliased textures** share the first registered descriptor.
- **Guest object stages and multi-entry descriptor tables** are explicit
  refusals; the ABI witness is the reflection JSON (the host checks it is
  non-empty, not semantically).
- **The 105/111 historical compute rejects are not reproduced** from this tree;
  the census is the 35-program corpus above (11 refusals on a 32-lane host).
- **xmem registry growth**: one `Retired` entry per submission that ever
  retained a range; bounded-memory design is still open.

## Cycle 06 — pre-boot AAA hardening: multi-entry tables, object+mesh+pixel, in-pass indirect mesh, fail-closed wave64, bounded xmem GC, core render states, one-command launcher

Base `11b2fa66` (`cycle-05`). Scaffold `6c6e20b7` reserves `macos/state.rs` and
`victory-gpu/src/native_metal/state.rs` as owner modules and raises the v3
`SetDescriptorTable` cap from the frozen v2 16 to `MAX_DESCRIPTOR_BINDINGS_V3`
(32) in both the compute and graphics prepare paths. The integration head is the
commit this section lands on; local annotated tag `cycle-06` marks it. Every
number below comes from a command run in this workspace on Apple M4 Max; a number
or behaviour that could not be observed is labelled as such.

### 6.1 Decisions (ADR-023…026)

- **ADR-023 — multi-entry descriptor tables.** A D3D12 root descriptor table binds
  1..=32 homogeneous CBV, SRV or UAV descriptors (and sampler tables where the
  guest can carry them). The guest enumerates live consecutive slots from the
  table base at the kind's real increment (views 32 B, samplers 16 B) and stops at
  the first empty, unresolvable, mixed-kind or off-end slot; the host binds the
  enumerated prefix as one argument-buffer table under the pipeline's reflection
  offsets with bulk `MTLResidencySet` residency. More than 32 entries is
  `OverCap{count}` — never a silent truncation.
- **ADR-024 — Amplification/Object + Mesh + Pixel and in-pass indirect mesh.** The
  optional object stage joins the `VPS1`/`VPS2` registration (`converter_stage()`
  6, wire code 3, `from_name("amplification")`) and maps to
  `MTLMeshRenderPipelineDescriptor.objectFunction`, `payloadMemoryLength` and
  `maxTotalThreadsPerObjectThreadgroup`. `ExecuteIndirect` for `DispatchMesh`
  records inside the open `BeginRenderPass`..`EndRenderPass` instead of refusing
  an open pass; a following graphics draw re-emits its pipeline into the same
  pass.
- **ADR-025 — fail-closed `narrow_waves` and bounded xmem retirement.**
  `narrow_waves && !host_wave32` refuses at `lower::lower_pruned` (and again at
  the public `Options` boundary) with one explicit message, so the silent
  zero-high-ballot-dword lowering is unreachable; Apple Silicon Metal lowering
  always sets `host_wave32`. The `xmem::share` registry evicts retired submission
  rows deterministically (ascending id, `Retired` rows only) to a capacity
  (default 4096, `VICTORY_XMEM_RETIRED_CAPACITY`), keeps a monotonic retired
  watermark so an evicted id can never resurrect a range, and the retire queue
  exposes capacity/`high_watermark`/`try_push`/`refusals`.
- **ADR-026 — AAA core render state and the turnkey launcher.** Wire v3 opcodes
  46..=53 carry blend, depth-stencil, rasterizer, stencil ref, blend factor,
  `ClearRenderTargetView`, `ClearDepthStencilView` and `CopyBufferToTexture`. The
  host bakes blend into the pipeline and the depth state into a cached
  `MTLDepthStencilState`, applies cull/winding/fill/depth bias per encoder,
  folds an explicit clear into the next continuation pass load action (never a
  dropped clear), and performs the pitched upload as
  `copyFromBuffer:sourceBytesPerRow:sourceBytesPerImage:...`. `run-title-metal.sh`
  turns the diagnostics into one command.

### 6.2 Chronology of the work

| Step | Commit | Evidence |
|---|---|---|
| scaffold | `6c6e20b7` | owner modules reserved; v3 descriptor-table cap 16 → 32 |
| C6-WAVE64-FAIL-CLOSED | `06dab3d1` | refusal at `emit/lower.rs` `lower_pruned` + `emit/mod.rs`; 35/35 corpus programs refused with one reason each (13 mask, 7 LDS, 7 reduction, 4 first-lane, 4 lane-read); supported pair unchanged 24 translated / 11 refused; `victory-shader` 248 / 0 / 2; M4 Max `wave_lower_gpu` 5 passed incl. both-halves mask/ballot |
| C6-XMEM-BOUNDED-GC | `333461f3` | deterministic eviction + watermark; capacity 4096 (env-overridable); retire queue capacity/high-watermark/`try_push`; Wine `victory-hle` 180 passed / 1 ignored (pre-existing display test); stress `cycles=10000 peak=64 capacity=64 watermark=20936`; evicted id refused as `Retired`, id above watermark `UnknownSubmission` |
| C6-MULTI-ENTRY-TABLES | `29ac5fea` | guest enumeration 1/2/4/32 admit, 33rd `OverCap`; M4 Max one submission with 4 CBV + 4 SRV + 4 UAV + root constants: target `[100,200,255,255]`, UAV words per entry `[0,4,0,8]`…`[30,16,60,32]`, builder extents 4×8 B and 4×24 B, 7 residency allocations; fixed the host rejecting unaligned opaque `gpuResourceID`s |
| C6-OBJECT-MESH-PIPELINE | `5d802a58` | `[Mesh, Object, Pixel]` `VPS1` flags `0b11`, object threadgroup `[1,1,1]`, payload 1024 B; `[Object, Mesh, Pixel]` refused as `HostCannotRegister`; M4 Max: 8 object threads, odd culled, 4 payloads drawn = 112 exact px; `victory-gpu` serial 208 / 0 / 1 |
| C6-INDIRECT-MESH-IN-PASS | `713e4d1b` | producer opens the pass for indirect mesh; compute-written count 1 → 64 px (`0x0fcebc438a043e25`), count 2 → 128 px (`0x60094f9b132ff125`) equal to direct `DispatchMesh` (`diff = 0`); record buffer written only by the GPU; `indirect_gpu` 4 passed; producer tests 26 → 29 |
| C6-AAA-RENDER-STATES | `e34744b9`+`a1fd12b5` | wire opcodes 46..=53 and host state modules; see 6.3 for the two root causes of the vanishing draw and the restored proof; `render_states_gpu` 1/1, `submit_v3_bindings_gpu` 3/3, `indirect_gpu` 4/4 |
| C6-TITLE-LAUNCHER | `c91686de` | `tools/macos/metal-poc/run-title-metal.sh` (942 lines) + host `--headless`/ready-line fields; headless exit 0, 120 measured frames, p50 0.395 / p95 0.766 / p99 0.779 ms, 2153.76 FPS, guest `frames_ok=140`, `d3dmetal_fallback_count=0`, teardown with no leaked process; window mode vsync-limited 61.18 FPS; `--window --headless` exit 2 |
| orchestrator fix | `11f0c0d8` | the macOS-only `ipc_host` import in `bin/metal-worker.rs` is gated so Windows `--all-targets` checks compile (pre-existing failure surfaced by C6-TITLE-LAUNCHER) |
| C6-AAA-PREBOOT-GATE | `826a0f37` | `crates/victory-gpu/tests/aaa_preboot_gate.rs` (1710 lines) + `tools/macos/metal-poc/aaa-preboot-gate.sh` (719 lines); two serialized 120-frame runs, exit 12 / PARTIAL because three integrated host gaps are observed and named rather than skipped (see 6.3 and 6.6) |

### 6.3 Measured evidence

**Render-state root causes (C6-AAA-RENDER-STATES).** The blend submission
rendered no fragments whenever a `SetRasterizerState` command was present, even
with the rasterizer selector bodies neutered. Two real defects, both in
`macos/state.rs`:

1. `MTLTriangleFillMode` constants were inverted (`Lines = 0`, `Fill = 1`), so
   `FillMode::Solid` called `setTriangleFillMode:` with line fill; almost no
   fragments landed. Corrected to `Fill = 0`, `Lines = 1`, and the unit test that
   locked the wrong values was fixed with it.
2. D3D12 colour write masks were passed straight to `MTLColorWriteMask`, whose bit
   order is reversed (D3D12 `R=1,G=2,B=4,A=8`; Metal `R=8,G=4,B=2,A=1`), so mask
   `0b0101` blacked the second target. A translation helper now maps the mask
   (`0b0101 → 0b1010`) and is unit-tested.

The bisection had also disabled `setDepthBias:slopeScale:clamp:`; that emission is
restored. `CopyBufferToTexture` carried no destination handle, so the host could
never resolve the upload target; the opcode now carries
`destination_allocation_id`/`destination_generation`. Converted texture descriptor
tables are encoded from `gpuResourceID` directly because those ids are opaque and
not address-aligned.

**Render-state proof (M4 Max, debug, serialized).** Submission 1 (pipeline
compiled with blending, explicit `ClearRenderTargetView`/`ClearDepthStencilView`,
wire blend + depth state): target 0 all `[255,0,255,255]`, target 1 all
`[255,255,0,255]`. Submission 2 (depth test Less, back-face culling, depth bias):
`(8,8)` green, `(56,8)` cyan (biased quad wins at equal z), `(8,56)` yellow,
`(56,56)` blue (far quad survives; clockwise quad culled). Submission 3: target 0
all `[0,255,0,255]` and target 1 all `[0,0,255,255]`, i.e. the pitched-uploaded
4×4 texture sampled at `(1,1)` and `(2,2)` with a 0xAA-poisoned source and
`blit_encoders=1`.

**Wave64 fail-closed.** 35/35 corpus programs refuse the unsound pair with one
reason each; the supported pair is unchanged. On hardware the 64-lane fixture
reports live mask 64 and a high ballot dword of `0xff`, so both halves execute.

**Bounded xmem GC.** 10,000 alloc/use/unmap/retire cycles under Wine with the
registry capacity deliberately set to 64: peak map 64, watermark 20936, the live
pin's range and lease held every cycle, evicted id 2 refused as `Retired`.

**AAA pre-boot gate.** `tools/macos/metal-poc/aaa-preboot-gate.sh`, two
serialized runs, 1280×720, warmup 20, 120 measured frames, dev profile, exit 12
(`status=PARTIAL`). Run 1: p50 **0.829 ms**, p95 0.929, p99 0.956, **1193.56
FPS**; run 2: p50 0.839, p95 0.928, p99 0.971, 1194.57 FPS. Present checksum
`17274905532806365283` (`0xefbccb0e435a8063`) identical in both runs. 120/120
frames measured, 140/140 guest frames ok, 140 presents, 280 submissions ok and
one refused (the replay), replay `AlreadyExecuted` with the sentinel unchanged,
`d3dmetal_fallback_count=0`, guest D3DMetal module 0, unsupported-opcode
histogram empty, fence timeouts 0, registration failures 0, `compute_unbound` 0,
`PROCESS_CHECK host=0 guest=0 leaks=0`.

Observed in both runs: device/queue/fence, 4 textures, guest PSO ids (mesh 1,
compute 2, graphics 3); compute culling writes mesh counts
(`compute_mesh_counts_after=1,1,1,49374` with the `0xC0DE` marker, 140/140 linkage
frames); in-pass `ExecuteIndirectDispatchMesh` with the
compute → barrier → indirect → direct order checked; AS+MS+PS payload
(`object_payload_after=3,1,49374`, `mesh_pixels=64,512,000` =
140×(320×720+640×360)); depth test (`far_pixels=0`, `near_pixels=32,256,000`);
back-face culling (`cull_pixels=0`, `middle_pixels=0`); explicit MRT+depth clears
spliced every frame; pitched `CopyBufferToTexture` (row pitch 256, 4×4) every
frame; bounded xmem churn 10,000 cycles with peak retired rows **4096** at the
4096 capacity.

Not observed, and the reason the gate is PARTIAL rather than PASS:

1. **UAV table entry 1.** A count==2 UAV table binds entry 0 (`uav0=11`) but entry
   1 stays 0 (`uav1=0`, expected 22). The host binds one converted array; the
   second entry does not reach the shader. Candidate sites:
   `macos/bindings.rs:186` (8-byte `Unspecified` stride),
   `device.rs:3276-3311` (`abi_table_kind`/`with_descriptor_table_kinds`).
2. **SRV texture tables.** Reflection `Type=Table` maps to
   `DescriptorTableKind::Unspecified` (`macos/bindings.rs:365-376`), and
   `macos/device.rs:3199` only takes the texture path for kind `Texture`, so a
   guest texture handle resolves as a buffer range (`device.rs:2904`); the gate
   binds two SRV buffers instead. The pitched upload is proven by its copy into
   the presented target, not by sampling.
3. **Alpha blending with a guest PSO.** `check_blend_state`
   (`macos/device.rs:5007-5034`) refuses an enabled wire blend state for
   content-hashed v2 pipelines, because those compile without blending. Depth,
   cull and clears are real; an alpha-blended draw is not claimed.

The launcher's independent rectangle model does not reproduce the host checksum;
the measured checksum is exact and stable but is not an independently predicted
image. It is reported as informational, not as a pass criterion.

### 6.4 Counts (integration head)

- Compiled: Win64 `diag` `victory`; `victory-metal`/`victory-hle`/`victory-gpu`
  Windows all-targets; x86_64 `victory-irconverter`.
- Compiled: Win64 `diag` `victory` (exit 0); `victory-metal`/`victory-hle`/
  `victory-gpu` (diagnostics) Windows all-targets (exit 0); x86_64
  `victory-irconverter` binary (exit 0).
- CPU-tested at head `9a6477f1`, serialized: `victory-metal` **325 passed /
  0 failed / 33 ignored** across 26 test binaries (lib 278 / 5 ignored, the rest
  hardware-gated); `victory-shader` **251 / 0 / 3**; `victory-gpu` **229 / 0 / 1**;
  `victory-hle` under Wine **183 / 0 / 2** (lib 180 / 1 pre-existing display test).
- GPU-verified on M4 Max, serialized with `/tmp/victory-gpu.lock` and
  `--test-threads=1`: 16 hardware tests across 7 binaries — `render_states_gpu`
  1, `multi_entry_tables_gpu` 1, `object_mesh_gpu` 1, `indirect_mesh_gpu` 1,
  `wave_lower_gpu` 5, `submit_v3_bindings_gpu` 3, `indirect_gpu` 4 — plus the
  two-run pre-boot gate.

### 6.5 Reproduce

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
cd VIctory-Metal
cargo build --locked --profile diag --target x86_64-pc-windows-gnullvm --package victory --features diagnostics
cargo test --locked -p victory-metal --target aarch64-apple-darwin --lib -- --test-threads=1
cargo test --locked -p victory-shader --target aarch64-apple-darwin
cargo test --locked -p victory-gpu --target aarch64-apple-darwin -- --test-threads=1
tools/macos/test-wine.sh -p victory-hle -- --test-threads=1
cargo test --locked -p victory-metal --target aarch64-apple-darwin --test render_states_gpu -- --ignored --test-threads=1
tools/macos/metal-poc/run-title-metal.sh --headless --frames 120 -- <win64-exe> [args...]
```

### 6.6 Remaining gaps (honest)

- **No title binary has run.** The gate drives the real `victory-gpu` D3D12 entry
  points through the launcher, but not a retail title; no proprietary asset was
  used. Geometry, materials and streaming on a title frame remain unproven, and
  the ≥30 FPS game gate is still unmet.
- **The pre-boot gate is PARTIAL, not PASS.** Three integration gaps are observed
  in the merged code and named above: multi-entry UAV tables stop after entry 0,
  SRV tables never take the texture path, and an enabled blend state is refused
  for content-hashed guest pipelines. A title that uses multi-entry tables or
  alpha blending will hit them on frame 1, so a first live boot is a diagnostics
  run, not an acceptance run, until they are fixed.
- **The new render-state opcodes have no guest D3D12 entry point yet.** Blend,
  depth-stencil, rasterizer, clears and pitched upload are reachable through Wire
  v3 and proven on the host, but `victory-gpu`'s `OMSet*`/`Clear*`/upload
  entry points still record into `ignored_state` (owned by `list.rs`). A title
  will not exercise them until that wiring lands.
- **Sampler descriptor tables** are refused end to end: the host builder supports
  them, the guest has no sampler-registration route.
- **Guest texture-SRV tables** stop as unresolvable; textures do not yet travel as
  buffer ranges.
- **Object-stage conversion** is exercised through hand-authored MSL, not through
  console DXIL: `victory-shader` has no amplification `HwStage` yet.
- **Retire queue capacity is a soft bound** when the front fence never completes;
  `device.rs` still uses `push` (which retains and counts) rather than `try_push`
  backpressure.
- **`victory-gpu` parallel execution** has pre-existing global-state races in
  `native_metal::queue` tests; every suite here is run with `--test-threads=1`.

## Cycle 07 — pre-boot gate PASS: 24-byte converter tables, guest texture/sampler tables, blended content-hashed PSOs, amplification identification, hard retire bound

The Cycle 06 gate was PARTIAL with three named frame-1 gaps (multi-entry UAV
tables, SRV texture tables, blend on content-hashed guest PSOs). Cycle 07 closed
those three, added the two remaining hardenings, and turned the gate into a
zero-gap contract: `tools/macos/metal-poc/aaa-preboot-gate.sh --runs 2` exits 0
with `not_observed=none` on both 120-frame runs.

### 7.1 What changed

- **Converter descriptor tables (ADR-027).** `metal-objdump` of the converted
  gate shader shows `%struct__desc = { i8 addrspace(1)*,
  %struct._texture_buffer_1d_t addrspace(1)*, i64 }` and entry 1 at byte offset
  `+24`; the host wrote 8-byte addresses, so `uav1` read zeroes. Address tables
  (CBV/SRV/UAV/`Unspecified`) now emit 24-byte `{device_address,0,0}`,
  textures `{gpuResourceID,gpuResourceID,0}` and samplers `{gpuResourceID,0,0}`.
  The old 8-byte shape existed only in a hand-written fixture.
- **Typed wire tables and guest texture/sampler binding (ADR-028).**
  `DescriptorTableV3` entries are tagged buffer/texture/sampler; buffer entries
  keep the old encoding. `victory-gpu` enumerates a heap range by its real kind,
  registers textures on first bind and samplers by content (kind 10, bounded
  256, no eviction) with idempotent handles. Host-side residency attaches every
  referenced texture/sampler; an unregistered handle is a typed refusal before
  any encoder.
- **Render state inside content-hashed PSOs (ADR-029).** `PipelineState` emits
  VPS1 byte-for-byte when no render state is present and VPS2 when it is; blend,
  alpha-to-coverage, depth/stencil and rasterizer state are encoded
  deterministically into the blob, which is part of the BLAKE3 key. Two PSOs
  that differ only in blending are `PipelineId(1)` and `PipelineId(2)`; the
  identical registration is reused. VPS1 blobs still decode to defaults.
- **Amplification stage.** `HwStage::As` exists, AS+MS+PS containers parse
  without a stage refusal and AS is never read as GS/CS. The record slot is a
  documented fallback (compute slot +x60, selected by the kind-14 program
  header) because no shipped AS FSSM payload exists in the repository. Native
  AS lowering still fails closed by name; the converter path is what is proven.
- **Hard retire bound.** `RetireQueue` capacity is now a hard bound; the device
  pins only through `try_push`, waits on the oldest committed command buffer,
  publishes the completion, drains and retries within `MAX_BACKPRESSURE_WAITS`.
  A never-completing fence returns `MetalError::Backpressure` instead of growing
  the queue or hanging.
- **Process hygiene.** A shared `process-hygiene.sh` gives every metal-poc script
  tracked PIDs, TERM→KILL escalation, ownership-proven reaping and an orphan
  assertion (exit 13), with a hermetic self-test.
- **Host defect found by the gate.** `DaemonExecutor` and `FrameGateExecutor` did
  not forward `BatchExecutor::register_sampler` (trait default refuses), so the
  guest sampler table was refused by the daemon while the bare transport
  accepted it. Same defect class as Cycle 05's five missing delegations.

### 7.2 Gate evidence (Apple M4 Max, both runs PASS)

```text
run1 status=PASS frames=120 p50=0.825 p95=0.891 p99=0.947 ms fps=1205.78
run2 status=PASS frames=120 p50=0.822 p95=0.903 p99=0.960 ms fps=1201.21
checksum both runs = 11527919792119713699   not_observed=none
uav0=11 uav1=22
mesh counts = 1,1,1,49374        payload = 3,85,49374,112
AS+MS+PS = 4 of 8 payloads, 112 px exact; count sweep 1->64, 2->128, direct=128
clear=[0,0,255,255] copy=[0,255,0,255]/[255,0,255,255] upload=[255,255,255,255]
srv=[10,20,30,255],[40,50,60,255]   blend=[128,0,128,128] (VPS2 id 5)
xmem 10000 cycles peak=4096/4096; retire pins=24 cap=4 peak=4 waits=5 released=24
fence_waits=140 timeouts=0 d3dmetal_fallback=0 guest_module=0 unsupported=none
teardown: zero victory_metal_host / victory-irconverter / wine survivors
```

### 7.3 Counts (integration head)

- Win64 `diag` `victory` build exit 0 (35 s); `victory-metal`, `victory-hle`,
  `victory-gpu` (diagnostics) Windows all-targets exit 0; Win64
  `victory-irconverter` exit 0.
- `victory-metal` lib **284 passed / 0 failed / 5 ignored**;
  `victory-shader` **254 / 0 / 2** (+ integration 3 / 0 / 1);
  `victory-gpu` **237 / 0 / 1**; `victory-hle` under Wine **183 / 0 / 2**.
- GPU, serialized: `retire_backpressure_gpu` 2, `object_mesh_gpu` 2,
  `multi_entry_tables_gpu` 2, `srv_texture_table_gpu` 1, `render_states_gpu`
  1 (legacy `#[ignore]` proof re-run: 1), plus the two-run gate.

### 7.4 Reproduce

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
cd VIctory-Metal
cargo test --locked -p victory-metal --target aarch64-apple-darwin --lib -- --test-threads=1
cargo test --locked -p victory-shader --target aarch64-apple-darwin -- --test-threads=1
cargo test --locked -p victory-gpu --target aarch64-apple-darwin --lib -- --test-threads=1
tools/macos/test-wine.sh -p victory-hle -- --test-threads=1
tools/macos/metal-poc/aaa-preboot-gate.sh --runs 2   # exit 0 required
```

### 7.5 Remaining gaps (honest)

- **No title binary has run.** The gate drives the real `victory-gpu` D3D12
  entry points, not a retail title. Geometry, materials, streaming and ≥30 FPS
  on a world sequence remain unproven; no proprietary asset was used.
- **Native amplification lowering fails closed.** AS is identified, parsed and
  resource-tracked, and the converter path draws exact pixels, but
  `victory-shader` refuses to lower an AS program to DXIL because payload
  writes and the mesh grid have no DXIL form yet. The record-slot rule is a
  documented fallback, unverified against a shipped AS payload.
- **The measured checksum is synthetic.** `11527919792119713699` identifies this
  gate frame, not a game frame.
- **Logic-op blending and depth bounds** are not representable in Metal and are
  explicitly omitted rather than approximated.
- **Guest texture bytes** are uploaded by the gate through `CopyBufferToTexture`;
  a sampled guest texture with no upload samples its cleared contents.
- **Retire capacity** is a hard bound only where the device pins through
  `push_backpressure`; `RetireQueue::push` remains as documented legacy.

## Cycle 08 — first live title execution on the native path: two presented frames, zero D3DMetal, and the sequence ledger

Base `7d7bc67b` (cycle-07). This cycle crossed into real-title execution under
`VICTORY_BACKEND=metal` with a locally authorized title. No proprietary binary,
asset, key or path is recorded here: only counts, formats, opcodes and frame
times. Every number below comes from a run in this workspace on the Apple M4 Max.

### 8.1 What changed

- **ADR-030 — no silent state drops.** Every former `ignored_state` slot of the
  console list table now has a named handler: root-signature commits are recorded
  per stage, `DiscardResource` reuses the clear lowering, and the three console
  hints with no Wire v3 form (pixel-shader depth order, shading rate, state
  flush) are refused by name and counted by operation instead of vanishing. A
  table audit test fails the suite if a future slot becomes a silent stub.
- **ADR-030 — no host D3D12 device on the metal path.** `host::host()` returns
  `None` when `VICTORY_BACKEND=metal`, before any `D3D12CreateDevice` or factory,
  with a one-shot guard line (`host_d3d12_creation_attempts=0`). The d3d12
  backend is unchanged.
- **ADR-030 — clear discipline.** `ClearRenderTargetView`/`ClearDepthStencilView`
  before the bind, inside a pass, and with no following draw are all expressible:
  pending clears latch onto the attachment at pass open, and an unconsumed clear
  becomes its own single-attachment pass. Duplicate clear+bind pairs clear once
  per clear.
- **ADR-031 — turnkey launcher.** `tools/macos/victory-metal.sh` wraps
  `run-title-metal.sh`: builds the Win64 `diag` binary, the ARM64 host and the
  x86_64 converter, supports `--window/--headless/--timeout/--frames/--out/--dry-run`,
  exports `VICTORY_BACKEND=metal` plus the full IPC/share/fence environment,
  advertises AVX/AVX2 under Rosetta, bounds the run, and reports backend,
  endpoint, share dir and fence page without the secret.
- **ADR-032 — PID-verified GPU lock.** `tools/macos/metal-poc/gpu-lock` and the
  rewritten `process-hygiene.sh` lock share one protocol: atomic `mkdir`, an
  `owner` record, stale reclaim when `kill -0` fails, a bounded wait (exit 75),
  release only by the creator, and re-entrant acquisition by a wrapper's child.
- **Submission sequence ledger.** `MetalProducer::flush()` no longer consumes a
  sequence; `NativeQueue::enqueue` assigns exactly one per real frame. An empty
  `Close` and a reset that drops unassigned pending frames can no longer desync
  the host.
- **Formats.** Guest kind-8 code 9 is RGBA16Float (`MTLPixelFormatRGBA16Float`,
  115) for the title's FP16 present target; codes 10/11 are R32Uint/R32Sint
  (Metal 53/54, registry codes 18/19).
- **Host failure telemetry.** A failed Metal command buffer carries its
  `localizedDescription` coarsely attributed to a command family, host telemetry
  gains `last_error`, and session counters survive a `serve_once` error. A GPU
  `ExecutionFailed` is a reply; only transport/protocol errors end the session.

### 8.2 Boot telemetry (counts only)

| | Boot #1 (cycle-07 head) | Boot #2 (integrated head) |
|---|---|---|
| host handshake / fence page | ready / registered | ready / registered |
| `d3dmetal_fallback_count` | 0 | 0 |
| host D3D12 device created | **yes (defect)** | **no** (`creation_attempts=0`) |
| D3D12.X device | created (v0x20004) | created (v0x20004) |
| native submissions / frames presented | 0 / 0 | **2 / 2** (`graphics_status=0`, `present_status=0`) |
| host p50 / p95 | — | 0.008 / 2.873 ms |
| guest p50 / p95 / fps sample | — | 0.400 / 3.952 ms / 167.7 |
| producer refusals | 1 (`ClearRenderTargetView` before targets) | 0 |
| target texts with no host code | 1 (2560x1440, DXGI 10) | 1 (640x360, DXGI 62) after DXGI 10 and 42 registered |
| buffers created before the resolver | 6 (largest 182,452,224 B) | 6 (unchanged) |
| furthest milestone | first present attempt, then title assert | two frames presented, then title assert on the next resource |
| teardown | 0 survivors | 0 survivors |

Intermediate boots located the two defects that dominated the cycle: an empty
`Close` consumed a sequence and was then dropped (host answered `protocol
violation: submission sequence is not the next expected value` at frame 586), and
the guest still created a host D3D12 device. Both are fixed above.

### 8.3 Why the title still stops

The title probes resource classes in order. DXGI 10 (FP16 present) and DXGI 42
(R32_UINT storage, 1280x720) now register; DXGI 62 (640x360, UAV flag) does not,
device creation returns `E_FAIL`, and the title raises its own breakpoint
(`0x80000003`) and then faults inside Wine's crash logger. This is a format-table
completeness problem, not an architectural one: the guest maps nine formats and
the host decodes eleven.

### 8.4 Counts at the integration head

- `victory-gpu` lib **268 passed / 0 failed / 1 ignored**;
  `victory-metal` lib **294 / 0 / 5**; `victory-shader` **254 / 0 / 2**
  (+ integration 3 / 0 / 1); `victory-hle` under Wine **180 / 0 / 1** plus two
  targeted single-test runs; `victory` bin **16 / 0 / 0**.
- Harness: `test-gpu-lock.sh` and `test-process-hygiene.sh` exit 0.
- Win64 `diag` build exit 0; `victory-metal`/`victory-hle` Windows all-targets and
  `victory-gpu` (diagnostics) all-targets exit 0; Win64 `victory-irconverter`
  exit 0.
- `aaa-preboot-gate.sh --runs 1` exit 0 at the integration head, `status=PASS`,
  `not_observed=none`, `d3dmetal_fallback_count=0`; checksum
  `11527919792119713699` unchanged; p50 0.839 / p95 0.977 / p99 1.082 ms,
  1179.95 FPS. An earlier run of the same gate at the Wave 8A head measured
  p50 0.931 ms while three workers were compiling; both exited 0 with the same
  checksum.

### 8.5 Remaining gaps (honest)

- **Two frames, not a world.** No geometry, materials, streaming or controls have
  been validated; the two frames are the title's first presented submissions, and
  no frame checksum is read back in title mode (`present_checksum=-1`).
- **The title still aborts** on the third unmapped resource format (DXGI 62). The
  next cycle should replace the incremental format mapping with one complete
  DXGI-to-host table and prove it against the title's probe order.
- **Six buffers are created before the guest range resolver exists.** Their first
  bind is refused unless a later bind resolves again; the 182 MB allocation is
  the first thing to watch once frames advance.
- **DStorage creates duplicate queues** for the same source (eleven identical
  queue descriptions in one startup); not fatal, unexplained.
- **Wire replies carry no failure text.** The Metal reason reaches host telemetry
  and tracing only; the guest shows a bare `ExecutionFailed`. A bounded reason
  frame is the follow-up.
- **Retail shader translation is unexercised.** No title pipeline reached the
  converter in these boots; the 19/21 mesh-pair and 111 compute census remains
  historical.

## Cycle 09 — structural unblock: the complete format table, lazy buffer registration, wire failure text, and a title-side assert

Base `c78ef5f5` (cycle-08). This cycle removed the three structural blockers
Cycle 08 identified, then pushed the same locally authorized title past resource
probes into its full render-target set and descriptor-heap binding. No
proprietary binary, asset, key or path is recorded here: only counts, DXGI
values, categories and frame times. Every number comes from a run in this
workspace on the Apple M4 Max.

### 9.1 What changed

- **ADR-033 — one canonical DXGI→MTLPixelFormat table.** The three partial maps
  (`PixelFormat` in the host views module, the host `TextureFormat` code table,
  and the guest wire/PSO code mirrors) are replaced by a 93-row table: 91
  DXGI-backed rows plus two Metal-only sRGB mirrors. Every row answers dxgi
  value, Metal pixel format, one-byte PSO code, one-byte kind 8 wire code,
  view family, depth kind, element size, sRGB pair, typeless base and
  view-compatibility. `*_TYPELESS` resources are created with
  `MTLTextureUsagePixelFormatView` and a compatible base format; typed views
  are accepted inside a family only. Existing codes 0–19 (PSO) and 0–11
  (kind 8) keep their meaning. The SDK enumeration is authoritative: **62 is
  `R8_UINT` (Metal 13), 63 `R8_SNORM`, 64 `R8_SINT`, 65 `A8_UNORM`,
  57 `R16_UINT`, 87 `B8G8R8A8_UNORM`, 88 `B8G8R8X8_UNORM`**; the cycle
  directive's guessed names were off by one and are corrected in the table and
  tests. Formats Metal cannot express stay explicit named refusals.
- **ADR-034 — lazy first-bind buffer registration.** One idempotent helper
  family resolves a guest address or a resolved range, registers ascending (the
  host closes a session on a decreasing id), discovers a share backing that
  appeared after bind time, and never caps buffer size. The Boot #2 set
  (182,452,224 B plus four 19,680 B plus 352 B) is reproduced in tests and
  registers in one ascending sequence; copy, indirect and both descriptor-table
  paths use the same helper.
- **ADR-035 — bounded failure reason on Wire v3 error replies.** An error reply
  may carry `VMR3`, a u16 length and up to 256 UTF-8 bytes, truncated on a
  character boundary. Legacy four-byte payloads and v3 success replies stay
  byte-identical; the guest logs one `METAL_HOST_ERROR sequence=<n> reason=…`
  line and keeps a bounded count plus the last reason in telemetry.
- **Diagnostics that made the next failure visible.** A hermetic boot census
  (`tools/macos/metal-poc/boot-census.py`, 224 assertions, `--gate`), a bounded
  first-failure latch across descriptor/resource create and `Map`
  (`call_failures` / `first_call_failure`), a pipeline attempt census
  (`pipeline_attempts` by kind and outcome with the first bounded refusal), a
  descriptor-heap binding census (`heaps_bound`, `heaps_distinct`,
  `heaps_last`, `heap_bind_refusals`), and descriptor-heap memory taken from the
  title's own allocator when one is installed.
- **The native `SetDescriptorHeaps` gap.** The binding path routed through a
  state helper that returns early when no host D3D12 list exists, so on
  `VICTORY_BACKEND=metal` every title binding was dropped silently. The
  classification is now shared: the host path keeps its slot selection, the
  native path records the same binds.
- **DStorage duplicate lanes.** Eleven identical queue descriptions in one
  startup shared one scheduler lane (bounded dedup map of 64 descriptions,
  per-object status arrays, unchanged single-queue behaviour).

### 9.2 Boot progression (counts only)

| | Boot #1 | Boot #2 | Boot #3 | Boot #4 | Boot #5 |
|---|---|---|---|---|---|
| head | `44df1d70` | `c78ef5f5` | `6d8eb117` | `58319d96` | `c652a848` |
| host D3D12 (D3DMetal) device | yes | **no** | **no** | **no** | **no** |
| fallback count | 0 | 0 | 0 | 0 | 0 |
| host submissions / frames in window | 0 / 0 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| host p50 / p95 ms | — | 0.008 / 2.873 | 0.015 / 2.119 | 0.006 / 1.882 | 0.006 / 2.201 |
| guest p50 / p95 ms | — | 0.400 / 3.952 | 0.470 / 3.954 | 0.245 / 3.048 | 0.271 / 3.602 |
| unmapped-format refusals | 1 | 1 (DXGI 62) | **0** | **0** | **0** |
| buffers created before the resolver | 6 | 6 | 8 | 8 | 8 |
| descriptor heaps created / bound | — | — | 10 / not recorded | 10 / not recorded | 10 / 4 (2 distinct) |
| pipeline attempts / created | — | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| furthest event | first present attempt, then assert | 2 submissions, then DXGI 62 abort | 2 submissions, full RT set + 10 heaps, then title assert | same | same, with heaps bound and audited |
| surviving processes | 0 | 0 | 0 | 0 | 0 |

The 640x360 `R8_UINT` UAV+RT target that aborted Boot #2 now registers and is
accepted as one texture that is readable, writable and attachable. Boot #3
continued into a 1024x1024 `D32_FLOAT_S8X24` texture, three 994x624 and four
3840x2160 `B8G8R8A8` textures, ten CBV_SRV_UAV descriptor heaps of 3128
descriptors, one 460,800 B and one 16 B buffer, and a 4K `BGRA8` present
surface before it stopped.

### 9.3 Why the title still stops

The fault at title image `+0x2215c35` is not a dereference of one of our null
returns: it is the title's own fatal-assert stub — `movl $0,0x0; int3` —
reached from a guard loop over a table with a 0x48-byte stride whose addresses
come from globals next to the title's `__XMemAllocationHooks__`. At that point
`call_failures=0` (no instrumented export returned a failure or left a null
output), `pipeline_attempts=0` (no shader was attempted yet) and
`heap_bind_refusals=0` (the bound heaps were accepted). The remaining condition
is inside the title's own resource/heap validation, not an unmapped format, a
refused registration or a wire error. Resolving it needs the guard's predicate,
which this cycle did not establish.

### 9.4 Counts at the integration head

- `victory-metal` lib **308 passed / 0 failed / 6 ignored**; `victory-gpu`
  **303 / 0 / 1**; `victory-shader` **254 / 0 / 2**; Wine `victory-hle`
  **183 / 0 / 1**; `victory` bin **16 / 0 / 0**; DStorage subset under Wine
  **20 / 0 / 0** (three new lane-sharing tests included).
- Win64 `diag` build exit 0; `victory-metal` and `victory-hle` Windows
  all-targets exit 0; `victory-gpu` (diagnostics) Windows all-targets exit 0;
  x86_64 `victory-irconverter` exit 0.
- `boot-census.py --gate` exit 0 on Boot #3, #4 and #5 (fallback 0, host D3D12
  attempts 0, no surviving process); `git diff --check` clean; `pgrep` clean
  after every boot.
- GPU proofs: the format-table proof created 8 textures and every typed view on
  the M4 Max (6/6 ignored tests pass); heap-memory and heap-binding tests are
  host-side.

### 9.5 Remaining gaps (honest)

- **The title still stops before its first pipeline**, so no retail shader has
  been compiled, no 3D draw or compute dispatch has been submitted, and no
  world frame is validated. The pipeline census is ready to report the moment
  the title reaches one.
- **The assert predicate is unknown.** The next cycle should read the guard
  (the 0x48-stride table and the float comparison) or bisect it by supplying
  console-exact heap and allocation metadata; descriptor heap memory still
  landed at host allocator addresses, not in the title's XMem range.
- **Six create-time registrations still fail** (the pre-resolver buffers); the
  bind-time lazy path covers them, but the create-time counter still reports 6.
- **One submission per boot is refused guest-side** in every boot since #2; it
  is not a host refusal and is still unnamed.
- **Present counts are not asserted.** `present_status=0` and `graphics_status=0`
  are success codes; no frame checksum is read back in title mode.

## Cycle 10 — the assert was a headless predicate: real presents, real pipeline attempts, and a measured reference oracle

Base `ba05cee6` (cycle-09). This cycle removed the unnamed submission refusal,
made the guest-visible ABI measurable, and settled the Cycle 09 question — the
fatal assert at title image `+0x2215c35` is **not** an ABI/metadata delta. It is
the title's response to a headless session with no real swapchain. The same
tree, windowed, submits 72 command batches, presents 70 frames and reaches its
first real pipeline attempts. No proprietary binary, asset, key, shader or path
is recorded here: counts, DXGI values, address classes, categories and timings
only. Every number comes from a run in this workspace on the Apple M4 Max.

### 10.1 What changed

- **ADR-036 — differential guest-ABI parity, measured not guessed.** A bounded,
  opt-in trace (`VICTORY_GUEST_ABI_TRACE=1`) records the numeric answer of every
  guest-visible query: `CreateDescriptorHeap` (kind, count, flags, start halves,
  memory source), `GetCPU/GPUDescriptorHandleForHeapStart`,
  `GetDescriptorHandleIncrementSize`, `GetGPUVirtualAddress`,
  `CheckFeatureSupport`, the three resource creators, at-address creation (entry
  and answer), descriptor writes (destination halves, view kind) and the title's
  own `__XMemAllocationHooks__` path (size, attributes, address class, whether
  the hook answered). 64 records per call, 4,096 per process, nothing recorded
  when unset. `tools/macos/metal-poc/abi-diff.py` diffs two logs and names the
  first differing call, row and value; `test-abi-diff.sh` proves it hermetically.
  A zero-descriptor heap is refused (`E_INVALIDARG`) like the reference; the
  reserved-resource creator is now a named export instead of a silent stub.
- **ADR-037 — launcher and workspace hygiene.** Both launchers remove the
  session directory they own on every exit path (success, guest failure,
  timeout, signal, setup failure, a failure inside cleanup) unless
  `--keep-session` is passed; removal is ownership-proven, never a wildcard
  sweep, and a caller's `--out` is never touched. `VICTORY_BACKEND` is resolved
  once (`metal` default, `d3d12` reference) and printed in the dry run and
  summary. Integrated worker `target/` directories are pruned per wave.
- **The unnamed refusal.** Every boot since #2 counted one refused submission
  with no name. It was an empty list batch: `ExecuteCommandLists` carrying no
  recorded submission was enqueued, refused as `Empty`, and counted. An empty
  batch is now not a submission at all, every refusal carries a static name
  (`SUBMIT_REFUSAL_NAMES`, plus a session-wide `named_submit_refusals`
  histogram), and a no-client submission defers for a bounded streak before it
  is counted once. Boot #6 reports `submits=2/0` — zero refusals.
- **The frame probe.** `VICTORY_METAL_FRAME_PROBE=1` copies a 64x64 grid (8x8
  blocks of 8x8 texels) from the presented drawable into one 16 KiB staging
  texture on frame 0 and every 30th frame, with a 10 ms bounded wait and at most
  one probe in flight, and reports `non_black_pixels`, `unique_luma_buckets`,
  `frame_crc`, `sampled_frames`, `probe_pixels` and `failure`. Off by default:
  no command buffer, blit, readback or allocation.

### 10.2 The reference oracle

The D3D12 host reference path (same guest binary, same shim, `VICTORY_BACKEND=d3d12`)
was run under the same trace for the first time this cycle. It **passes**
`+0x2215c35` and runs to the launcher timeout: 9,413 submissions, real DXIL
translation ("shader translated" for vertex and pixel stages), host pipeline
creation and 168 MB/frame title-memory refreshes. Its first 22 `XMemAlloc`
rows, all 16 descriptor heaps and all 23 increment queries are **identical** to
the native run's. The ABI is therefore not the delta; device behaviour is.

### 10.3 Boot progression (counts only)

| | Boot #5 | Boot #6 | Ref #6 (D3D12) | Boot #9 | Boot #10 |
|---|---|---|---|---|---|
| head | `c652a848` | Wave 10A | Wave 10A | Wave 10A | +resolver `248cbe86` |
| session | headless | headless | headless | windowed | windowed |
| D3DMetal attempts / fallback | 0 / 0 | 0 / 0 | reference | 0 / 0 | 0 / 0 |
| submissions ok/refused | 2 / 1 | **2 / 0** | 9,413 | 72 / 0 | 67 / 0 |
| frames presented | 0 | 0 | n/a | **70** (71 drawables) | **65** (66) |
| assert `+0x2215c35` | yes | yes | no | **no** | **no** |
| pipeline attempts (graphics) | 0 | 0 | many | **3** | **3** |
| translator unavailable | 0 | 0 | 0 | 3 | 3 |
| "needs the guest range resolver" | n/a | n/a | n/a | many | **0** |
| buffers / PSOs registered | 0 / 0 | 0 / 0 | host n/a | 0 / 0 | 0 / 0 |
| non-black pixel probe | n/a | n/a (headless) | n/a | 0 of 4,096 | 0 of 4,096 |
| furthest event | assert | assert | frames | **real presents, first pipelines** | same, bindings resolve one layer deeper |

### 10.4 Why the headline is a headless predicate

Boots #6, #7 and #8 kept the assert while the trace showed identical call
sequences, identical increments and no request in flight at the fault: the
title faults between the last descriptor heap and its next at-address creation
without calling the shim. Boot #9 changed one variable — `--window` instead of
`--headless` — and the fault disappeared, with 72 submissions and 70 presented
frames. The predicate is the title's swapchain/present requirement. Headless
sessions remain useful for resource-path measurements but cannot cross this
gate; the harness now reports the mode in every summary.

### 10.5 Counts at the integration head

- `victory-metal` lib **314 passed / 0 failed / 7 ignored**; `victory-shader`
  **254 / 0 / 2**; `victory-gpu` **325 / 0 / 1**; `victory-hle` **81 / 0 / 0**
  (Wine suite 186 / 0 / 1); `victory` bin **20 / 0 / 0** under Wine; DStorage
  and the Wine share module green.
- Win64 `diag` build exit 0; `victory-metal` and `victory-hle` Windows
  all-targets exit 0; `victory-gpu` (diagnostics) exit 0; x86_64
  `victory-irconverter` exit 0.
- Pre-boot gate `--runs 1` **PASS exit 0**: p50 0.826 / p95 0.902 / p99 0.962 ms,
  1,211 FPS, model checksum `11527919792119713699` unchanged, xmem and retire
  proofs measured, `fallback=0`.
- Hermetic suites: `test-abi-diff` 8 checks, `test-launcher-session` 8,
  process-hygiene 13, GPU lock 10. `git diff --check` clean; no surviving
  process after any boot; 140 GB free.
- Frames are presented but still blank: `non_black_pixels=0`,
  `unique_luma_buckets=1`, `fallback_clear=1` on every sampled frame. **No
  loading screen, title screen or 3D frame is claimed.**

### 10.6 Remaining gaps (honest)

- **No real draw yet, for two named reasons.** Every graphics pipeline attempt
  is refused as `graphics/converter-unavailable`: the guest is a Win64 process
  under Wine and cannot run the macOS IRConverter, and the kind 4 v2 pipeline
  registration carries only a *ticket naming a pre-existing metallib* — no DXIL,
  no hash, no reflection and no capability advertisement (verified field by
  field; the host daemon owns no translator pool at all). With the resolver now
  installed, binds get one layer further and fail as
  `unsupported guest binding: the range's section has no host-visible backing`
  (134 vertex buffers, 67 root views, 67 descriptor tables per run), plus
  `a placed view is not a buffer descriptor` for 67 table enumerations.
- **Host-side translation is a real wire change**, not a wiring fix: a
  file-backed or chunked stage upload (the transport frame is 64 KiB against a
  16 MiB stage cap), a host translator-pool owner, a capability reply and a
  bounded rejection reason. That is the Cycle 11 headline.
- **Presents are blank**, so no loading screen, title screen or 3D frame is
  claimed; the probe is the instrument that will say so when it changes.
- **The headless gate is the title's**, not ours; the harness must run windowed
  to cross it, which is recorded rather than worked around.

## Cycle 11 — the three named gaps closed in code, host translation reached, and the next three named

Base `1dcafcd5` (cycle-10). This cycle implemented the three architectural gaps
Cycle 10 named precisely, measured each one at runtime, and removed two of them
from the boot log. The first real pipeline translation still does not complete:
the wire that carries it desynchronizes its reply. No proprietary binary, asset,
key, shader or path appears here — counts, codes, categories and timings only.
Every number comes from a run in this workspace on the Apple M4 Max.

### 11.1 What changed

- **ADR-038 — host-owned Rosetta DXIL translator pool.** `victory_metal_host`
  owns one bounded `CompilerPool` (`victory-irconverter --stdio`, VMRES2
  reflection) and the persistent content-hashed disk cache. The guest writes
  each stage's DXIL to `VICTORY_METAL_SHARE_DIR/stages/<blake3>.bin` (idempotent,
  up to the 16 MiB stage cap, never through the 64 KiB transport frame) and sends
  a new `VP3\0` kind 4 frame naming the stage files, entries, pipeline state and
  an optional root-signature file. The host validates the whole frame before any
  GPU work, translates uncached stages concurrently, derives the per-stage
  `PipelineAbi` from the reflection JSON, registers through the existing per-stage
  registry and replies `(pipeline_id, status, bounded <= 256 B reason)`. Host
  telemetry gained `pool_status`/`pool_*`, `host_translated_stage_*`,
  `host_pipeline_translations_ok/refused` and `host_translation_reason`. The
  launcher exports the converter worker and library paths to the host.
- **ADR-039 — universal host-visible XMem backing.** Every XMem section now takes
  the share path in the diagnostics build too (the launcher's guest is a
  diagnostics build), a size that is not a whole 64 KiB page is rounded up
  instead of silently downgrading to the pagefile, the directory is resolved once
  per process, and `share::live()` counts backings. `Section::size` stays the
  allocation's real size while `map_size` is the padded mapping, so padding never
  resolves.
- **ADR-040 — placed views as first-class table entries.** `View::Placed` now
  decodes the resource dimension, the view's memory address and the buffer
  union: a placed buffer resolves to `(address, len)` bounded by its footprint, a
  placed texture resolves to a registered host handle (auto-registering on first
  use), CBVs resolve from the footprint, and every remaining case is a named
  refusal. The blanket `a placed view is not a buffer descriptor` is gone.
- **Boot environment isolation.** The host-only converter variables were reaching
  the Wine guest, which then tried to run the macOS x86_64 worker; the launcher
  now unsets them for the guest and the Windows guest ignores a macOS worker path,
  so a converter-less guest reports `ConverterUnavailable` and takes the host path.
- **A stale mesh-ABI assertion** (pre-ea88d4cd 8-byte table entries) was corrected
  to the current 24-byte converter entry; it failed at cycle-10's head as well.

### 11.2 Boot progression (counts only)

| | Boot #10 | Boot #11 | Boot #12b (reverted) | Boot #13 |
|---|---|---|---|---|
| head | `1dcafcd5` | 11A | 11A + env + at-address | 11A + env |
| D3DMetal attempts / fallback | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| submissions ok/refused | 67 / 0 | 68 / 0 | 55 / 0 | 66 / 0 |
| frames presented | 65 | 67 | 1 | 65 |
| placed buffer / texture resolved | 0 / 0 | **64 / 0** | 64 / 0 | **64 / 0** |
| placed-view refusals | 67 | 0 | 0 | **0** |
| "no host-visible backing" | 201 | 0 | 0 | **0** |
| heaps bound | 0 | 406 | 394 | 394 |
| PSO attempts / created / refused | 3 / 0 / 3 | 3 / 0 / 3 | 3 / 0 / 3 | 3 / 0 / 3 |
| refusal class | `graphics/converter-unavailable` | `graphics/translation` (guest leak) | `graphics/host-translation` | `graphics/host-translation` (1-byte reply) |
| host-translated stages | 0 | 0 | 0 | 0 |
| `non_black_pixels` / `unique_luma_buckets` | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| furthest event | first pipelines | placed views + sections resolve | host aborts on a texture descriptor | VP3 reaches the host, reply lost |

`frame_crc` is unchanged (`11712555351105894739`) across every sampled frame in
#10, #11 and #13: the presented surface is the fallback clear.

### 11.3 Counts at the integration head

- `victory-metal` lib **319 passed / 0 failed / 7 ignored**, all integration
  test binaries green (mesh live ABI 3/3); `victory-gpu` **336 / 0 / 1**;
  `victory-shader` **254 / 0 / 2**; `victory-hle` **81 / 0 / 0**.
- Win64 `diag` build exit 0; `victory-metal`, `victory-hle` and `victory-gpu`
  (diagnostics) Windows all-targets exit 0; x86_64 `victory-irconverter` exit 0.
- Pre-boot gate `--runs 1` **PASS exit 0**: p50 0.811 / p95 0.898 / p99 0.964 ms,
  1,221 FPS, checksum unchanged, 283 submits / 1 refused, 140 presents / 0.
- `git diff --check` clean; no surviving process after any boot; 136 GB free.

### 11.4 Remaining gaps (honest)

- **The VP3 reply is lost, not the translation.** The host answers the pipeline
  frame with a one-byte rejection, which its decoder only emits when the payload
  does not start with `VP3\0`, while the guest demonstrably built that payload.
  The client socket is now shared (`Arc<Mutex<TcpStream>>`) between the pipeline
  worker and the render thread; request and reply are not serialized as one
  transaction, so a reply can be consumed out of order. Serializing the
  write+read pair (or giving the pipeline channel its own connection) is the
  single highest-leverage Cycle 12 change, and the host's bounded rejection
  reason must reach the log either way.
- **A 33x33 fp16 3D target crashes the host.** Registering that volume at
  creation (usage 3, unordered) makes the host build a
  `MTLTextureType2DMultisampleArray` with depth 33 and assert in
  `MTLTextureDescriptor`. The change that exposed it is reverted; the host's
  dimension/depth mapping for D3 and the multisample decision need fixing before
  at-address textures can be enabled.
- **XMem virtual allocations are still not share-backed.** Every guest-memory
  path was inventoried and the `XMemVirtualAlloc` region path was implemented,
  but the merged change left a reserved page uncommitted and the title faulted at
  frame 2 (`read ...: reserved, never committed`). It is reverted; 211
  registration failures per run still come from buffers in that memory.
- **No draw, no dispatch, no non-black pixel.** Frames are presented and blank;
  no loading screen, title screen or 3D frame is claimed.

## Cycle 12 — serialized pipeline channel, canonical texture types, live title draw progression

Base `0045347f`; ADRs **041** (dedicated transaction-isolated pipeline connection),
**042** (canonical `MTLTextureType` mapping and at-address textures), **043**
(exact `XMemVirtualAlloc` reserve/commit semantics — implemented, then **reverted**
after the title asserted; see 12.4) and **044** (per-entry kinds in a mixed
descriptor table).

### 12.1 What changed

- `victory_metal_host` serves concurrent loopback sessions; `victory-gpu` opens a
  dedicated, mutex-serialized pipeline connection, so a translation can never
  consume a render reply. Every pipeline error carries a bounded ≤256-byte preview
  of what the host sent, so a "1-byte rejection" now names itself. `--once` serves
  the whole guest (render session on the main thread — AppKit is main-thread-only —
  plus the pipeline session on a supervisor thread), which is what made the channel
  usable in a real boot.
- The nine `MTLTextureType` constants are canonical (`Cube=5`, `CubeArray=6`,
  `3D=7`, `2DMSArray=8`); one table is used by the descriptor, the view plan and
  texture creation. Proven on the M4 Max GPU: a 33×33×33 FP16 UAV volume is created
  as `MTLTextureType3D` and viewed (exit 0).
- At-address target textures register again and resource creation never fails on a
  host registration gap (the C11 `e4ef01cf` behaviour, re-applied on the fixed enum).
- The four trivial draw-list refusals (`IASetIndexBuffer(null)`,
  `IASetPrimitiveTopology`, `SetPixelShaderDepthOrder`, `SetShadingRate`) are named
  state changes with counters, not refusals.
- The host's stdio converter worker now always receives `--converter`, captures its
  bounded stderr, and the refusal carries that text.

### 12.2 Boot #13 → #14 → #17 (windowed, frame probe on)

| | #13 | #14 | #17 (final) |
|---|---|---|---|
| D3DMetal attempts / fallback | 0 / 0 | 0 / 0 | **0 / 0** |
| host submissions accepted / refused | 66 / 0 | 0 / 0 | **4223 / 0** (final run 5157 presents) |
| host frames presented | 66 | 0 | **5157** |
| registered buffers / psos | 0 / 0 | 0 / 0 | **112 / 0** |
| pipeline attempts / created / refused | 3 / 0 / 3 | 0 / 0 / 0 | **6 / 0 / 6** |
| pipeline refusal class | host-translation (1-byte) | — | **converter worker exited** (named, with its stderr) |
| host p50 / p95 / p99 ms | 0.009 / 0.013 / 1.114 | — | **0.016 / 0.032 / 0.038** |
| `non_black_pixels` / `unique_luma_buckets` | 0 / 1 | 0 / 1 | **0 / 1** |
| `fallback_clear` | 1 | 1 | **1** |

The surface is still the fallback clear and `frame_crc` is unchanged, so **no
non-black title, loading-screen or 3D frame is claimed**. What did change is that
the title now runs continuously (4223 submissions, no crash) instead of dying at
66 frames, and the VP3 pipeline frame finally reaches the host over its own
connection.

### 12.3 Counts at the integration head

- `victory-metal` lib **331 passed / 0 failed / 7 ignored** plus every integration
  binary; `victory-gpu` **354 / 0 / 1**; `victory-shader` **254 / 0 / 2**;
  Wine `victory-hle` **187 / 0 / 1** (`--test-threads=1`; four tests fail only under
  the harness's parallel default, which is pre-existing process-global contention).
- Win64 `diag` build exit 0; `victory-metal` all-targets, `victory-gpu`
  (diagnostics) and `victory-hle` Windows checks exit 0; x86_64
  `victory-irconverter` exit 0.
- Pre-boot gate `--runs 1` **PASS exit 0**: p50 0.827 / p95 0.917 / p99 0.945 ms,
  1,205.6 FPS, checksum unchanged.
- `git diff --check` clean; no surviving process after any boot.

### 12.4 Remaining gaps (honest)

- **The converter worker dies on a real job.** It starts with `--converter <path>`
  (its banner proves the argument reaches it), then exits while serving the first
  real vertex/compute stage: the only x86_64 `libmetalirconverter.dylib` is
  exercised under Rosetta and produces no stderr before dying. Every pipeline is
  therefore refused for a named reason and no PSO is created. Capturing the child's
  exit status/signal and either fixing the Rosetta path or exploring a native
  conversion path is the highest-leverage Cycle 13 change.
- **`XMemVirtualAlloc` backing is still absent.** Both ADR-043 attempts make the
  title assert at its first `CreateAtAddress` (guest VA in the low heap, before the
  xmem hooks are installed) and the crash handler then faults on a page reported as
  "reserved, never committed". Reverted again to keep the boot alive; the assertion
  needs to be identified from the title's own memory-init path, not guessed.
- **112 buffers register, but vertex/index ranges still refuse.** The new naming
  separates "heap has no host-visible backing" from "no shared xmem range"; the
  remaining blocker is the missing backing above.
- **`DrawIndexed` after an explicit null unbind is refused by name (18×).** Correct
  per D3D12, but it drops draws the title expects; it needs a per-list decision in
  Cycle 13.

---

## 13. Cycle 13 — real title shaders reach the converter; heap backing lands; the presented surface is finally sampled honestly

Cycle 13 attacked the three blockers Cycle 12 named and added a fifth track that the
Boot #17 log justified. Base `73701a70`; ADRs 045–047.

### 13.1 Decisions

- **ADR-045** — the `--stdio` supervisor never converts in-process (a short-lived
  re-exec per job), reports the child's exit code or signal plus its bounded, path-
  scrubbed stderr in a structured error frame, and keeps serving. Separately, the
  host now hands the converter the **complete `DXBC` container**, not a bare `DXIL`
  part, and takes the entry point from `PSV0`.
- **ADR-046** — `XMemVirtualAlloc` reserve/commit stays exactly as it was; committed
  regions are backed by share files at the same address through an additive share-
  layer hook with an in-place rollback. The allocation-semantics rewrite that asserted
  twice is not the path and stays reverted.
- **ADR-047** — `Present` reconfigures the drawable to the source extent before
  blitting, and `IASetIndexBuffer(null)` keeps the last non-null binding.
- **ADR-044 completion** — a descriptor table that mixes buffer and registered-
  resource entries is carried by the per-entry wire tag instead of refused.

### 13.2 Boots (windowed, frame probe on)

| | #17 | #18b | #19 | #20 |
|---|---|---|---|---|
| D3DMetal fallback / host D3D12 attempts | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| host frames, refused submits | 5157, 0 | 4257, 0 | 4175, 0 | 4199, 0 |
| buffers registered | 112 | 112 | **3371** | 3338 |
| pipelines attempted / created / refused | 6 / 0 / 6 | 6 / 0 / 6 | 6 / 0 / 6 | 6 / 0 / 6 |
| converter jobs / worker starts / restarts | 11 / 11 / 11 | 11 / 11 / 11 | **11 / 4 / 0** | 11 / 4 / 0 |
| converter outcome | worker exited (banner only) | container validation refusal | **child SIGSEGV, named** | SIGSEGV + `MSC unavailable` |
| vertex/index/descriptor refusals | present | present | **0** | **0** |
| `DrawIndexed` after null unbind | 18 | 18 | **0** | 0 |
| `SetStateX (0x470)` | 1/frame | 1/frame | **0** | 0 |
| `non_black_pixels` / buckets | 0 / 1 | 4096 / 1 | 0 / 1 | **0 / 1** |
| `uniform_color` | — | — | — | **transparent_black**, `nonzero_bytes=0` |
| `fallback_clear` | true | false | false | **false** |
| host p50/p95 ms, FPS | 0.017/0.036, 54.3k | 0.016/0.038, 51.5k | 0.011/0.027, 72.7k | 0.011/0.027, ~72k |

The round trip through #18b's `4096` non-black reading was a probe defect, not content:
#18b counted alpha as a colour. #20 replaced that with one classification pass and
self-describing fields (`nonzero_bytes`, min/max/mean RGBA, variance, distinct sample
values, exact sample byte offsets), which prove the drawable is fully transparent and
that the sampled bytes are inside the 3840×2160 source.

### 13.3 What the converter is actually given

`VICTORY_METAL_PART_CENSUS=1` (opt-in, numbers only) reports, per distinct container:
`parts=7 [SFI0 ISG1 OSG1 PSV0 HASH DXIL PRIV]`, `dxil=kind:1(Vertex) sm:6.6`,
`bitcode_offset/size` within a ≤1 KiB container. These are **stub** containers: the
`DXIL` payload carries 4 bytes of bitcode, and the legacy x86_64 converter segfaults on
it. `metal-shaderconverter` is not installed next to `libmetalirconverter.dylib` on this
host, so the new MSC fallback records a named one-line reason instead of panicking.

### 13.4 Counts at the integration head

- `victory-metal` **351 passed / 0 failed / 9 ignored** (`--test-threads=1`);
  `victory-gpu` **377 / 0 / 1**; Wine `victory-hle` **81 / 0 / 0** (lib,
  `--test-threads=1`); `victory-shader` clean.
- Win64 `diag` build exit 0; `victory-metal`, `victory-gpu` and `victory-hle` Windows
  all-targets checks exit 0; x86_64 `victory-irconverter` exit 0.
- Pre-boot gate `--runs 1` **PASS exit 0**: p50 0.825 / p95 0.892 / p99 0.966 ms,
  1,209.8 FPS.
- `git diff --check` clean; `pgrep` empty after every boot; worker `target/`
  directories pruned each wave; 122 GB free.

### 13.5 Remaining gaps (honest)

- **No PSO exists, so no draw is issued and no non-black frame is claimed.** The single
  remaining blocker for real geometry is translation of the title's container: the
  legacy converter dies on the stub `DXIL` payload and MSC is absent on this host.
- **`heaps_bound` grows every frame** (≈41k after 4.2k frames) — a per-frame heal
  census that needs bounding.
- **Probe cost** is ≈1.39 ms on each sampled frame; acceptable at `probe_every=30` but
  it should be measured against a real frame budget once draws exist.
