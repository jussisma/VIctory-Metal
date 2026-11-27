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
