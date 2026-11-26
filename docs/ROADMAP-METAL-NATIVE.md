# Roadmap: Native Metal Backend for VIctory-macOS (`victory-metal`) & Short-Term Optimizations

> **Handoff Note (LLM / DeepSeek-Ready Context):** This document captures the exact measured performance state, root-cause diagnosis, reuse boundaries, host Metal/IRConverter ABIs, and step-by-step milestones so any developer or agentic LLM can resume work cleanly without repeating experiments.

---

## 1. Current Baseline (`mac-turbo` @ `619204e5`, rebased onto `origin/main` `bf99e728`)

- **Host test machine**: Apple M4 Max, macOS, Rosetta 2 (`ROSETTA_ADVERTISE_AVX=1`), CrossOver Wine 26.3.0 (`target/wine-cx`) + Apple D3DMetal 4.0 beta 3.
- **Target game**: `game_scarlett_master.exe` v1.0.2.1.
- **Branch**: `mac-turbo` in `/Users/ismadev/developer/compatibility/VIctory-macOS` (commits above `origin/main` `bf99e728`: `9cde3793` `VICTORY_AMBIENT`, `641dc8a3` + `eba04b60` `VICTORY_BVH_NO_HIT`, `2e7e75f6` draw scratch/bitset/thin LTO, `f4ae5c7d` Windows `bvh_tests` fix, `9ae99c17` `VICTORY_PORT_DESCRIPTORS=1`, `619204e5` Metal-aligned mesh linking trigger).
- **Backup branches**: `mac-turbo-v3`, `mac-turbo-v4`, `mac-turbo-v5`.
- **Launcher**: `/Users/ismadev/developer/compatibility/run-with-winecx.sh` (must launch without PTY: `< /dev/null`).

### Measured Progression in World Gameplay

| Run | Profile & Key Settings | Frame Time (FPS) | Preparing / Copies / Textures | GPU Time | Copy Back | Refused >32 KiB Mesh / Untranslated Compute Indirects |
| --- | --- | --- | --- | --- | --- | --- |
| `v0` | Old baseline | ~7,000 ms (0.14 fps) | — | — | — | World unreadable / very slow |
| `v2` | Clean `ba4f374e` + `ambient` + `bvh_no_hit` (Series X `1920x1080`) | 629–645 ms (1.6 fps) | 332–338 ms | 274–289 ms | — | 207–239 dropped without pipeline |
| `v3` | `1280x720` + `ATOMIC_UMAX=1` + `ActivateRays` skip + thin LTO (Series X) | **494–508 ms (2.0 fps)** | 266–286 ms (`copies 70–78 ms / ~196 MB`, `textures 8–10 ms`) | 189–215 ms | 16.5 ms (0.0 MB) | **46** refused >32 KiB mesh (~100 dropped/frame), **111** untranslated CS (~270 indirect/frame), 2,829 textures (2.8 GB) |
| `v4` | `--console series-s` + `ntdll` patch + `PORT_DESCRIPTORS=1` (`render scale 0.67`) | 1,019–1,292 ms (0.8–1.0 fps) | 696–889 ms (`copies 688–1,134 ms / 176–491 MB`) | 184–253 ms | 130–219 ms (0.0 MB) | `ntdll` patch fixed intro to 30 fps, but Series S managed **6,913** textures (5.05 GB) and 87 missed memory pages |
| `v5` | Series X + `1280x720` + `render scale 0.5` + `PORT_DESCRIPTORS=1` + `619204e5` mesh linking fix | **544–585 ms (1.7–1.8 fps)** | 286–332 ms (`copies 80–98 ms / 253–264 MB`, `textures 4–5 ms`) | 234–251 ms | 17–21 ms (0.0 MB) | Refused >32 KiB mesh **halved (46 -> 25)**, dropped indirects **cut 62% (270 -> 101/frame)**, 8,670 textures (5.5 GB), `missed 12–16` pages |

### Sibling Forks Checked (Status as of 2026-11-26)
- `AvatarManz/VIctory-7900XTX` (16–22 fps) and `Belobug/VIctory-Belobug-AMD` (~8 fps) run natively on Windows x64 + AMD D3D12 (`OpenExistingHeapFromAddress` zero-copy), which macOS D3DMetal refuses (`crates/victory-gpu/src/alias.rs:4`).
- `VIctory-macOS` already reads table & heap CBVs directly as raw SRVs (`crates/victory-shader/src/emit/lower.rs:4574`, `crates/victory-gpu/src/descriptor/host.rs:1824`), so `AvatarManz`'s `VICTORY_PORT_CBV=1` does not apply to macOS. `VICTORY_PORT_DESCRIPTORS=1` was ported (`9ae99c17`, working at `0.33 ms/frame` taking slots with `0` draw-table misses).
- Sibling macOS repos in `/Users/ismadev/developer/compatibility/` (`xcl-mac`, `ScarlettMac`, `MacBox`) all run `xcl.exe` under CrossOver Wine + `D3DMetal.framework`; none have an in-process native Metal backend.

---

## 2. Why Performance & Visuals Hit a Wall Under D3DMetal

1. **No zero-copy buffer heaps under D3DMetal (`~250 MB/frame` CPU copies + page guards)**:
   - Apple's D3DMetal refuses `ID3D12Device3::OpenExistingHeapFromAddress*` (`crates/victory-gpu/src/alias.rs:4`).
   - Therefore `VIctory-macOS` maintains shadow D3D12 buffers, arming `VirtualProtect` page guards (`crates/victory-gpu/src/watch.rs`), fielding Rosetta write faults, copying ~250 MB/frame (`copies ~85 ms`), and burning `~200 ms/frame` in undocumented `preparing` overhead (Wine thunks / D3DMetal D3D12 translation / lock contention) plus `~18 ms/frame` scanning for `copying back` even when `0.0 MB` are written back due to `memory::withholding()`.
2. **Wave64 vs Wave32 (`~110` compute shaders untranslated in `v3`–`v5`)**:
   - Apple Silicon / Metal executes at `WaveSize = 32`; Xbox RDNA2/GFX10 runs `wave64`.
   - Breakdown in `v4`/`v5`: `56` `lane mask read as data (wave64 on a 32-lane host)`, `30` `LDS one console wave64 shares without barriers`, `12` `wave reduction over both 32-lane halves`, `8` `first-lane read`, `3` `lane read crossing halves`, `2` `lane mask read as data`, `1` `descriptor read only as data`.
3. **Metal 32 KiB Mesh Shader Output Limit (`25` pipelines still refused in `v5`)**:
   - Most of the remaining 25 refused pipelines in `v5` sit just **512–1,024 bytes** over `32,768` bytes when Metal-aligned (`33,280–33,792` bytes).

---

## 3. Reuse vs Replace Matrix (Critical Architectural Fact)

| Component | Native Metal Role | Why |
| --- | --- | --- |
| `victory-shader` (GFX10 -> DXIL, `halves.rs`, `barriers.rs`, `mesh.rs`) | **100% REUSED** | Wave32 and 32 KiB mesh-output constraints are **Apple Silicon / Metal hardware limits**, not Wine/D3DMetal limits. DXIL output feeds `libmetalirconverter.dylib` directly. |
| `victory-loader`, `victory-hle`, `victory-formats`, `victory-pe`, `resolution.rs`, `descriptor_sites.rs` | **100% REUSED** | `game_scarlett_master.exe` is Windows x86-64; Wine/Rosetta still loads the PE, resolves Xbox imports, and patches code sites. |
| D3D12 host layer in `victory-gpu` (`host.rs`, `pipeline/host.rs`, `descriptor/host.rs`, `draw.rs`, `shadow.rs`, `watch.rs`) | **REPLACED** (behind `VICTORY_BACKEND=metal`) | Eliminates D3DMetal, shadow buffer copies (`~250 MB/frame`), `VirtualProtect` page guards, and Wine D3D12 thunk overhead. |

---

## 4. Target Architecture: Native Metal Backend (`victory_metal.so` Wine Unixlib)

1. **In-Process Bridge via Wine Unixlib**:
   - Because CrossOver Wine already loads native x86-64 macOS `.so` libraries via `ntdll/unix` (`__wine_unix_call`), a thin Rust/ObjC CDylib (`victory_metal.so`, Mach-O `x86_64`) shares the **exact same 64-bit virtual address space** as `game_scarlett_master.exe` without IPC or sockets.
   - `victory.exe` calls a tiny bridge API (command buffer batch, resource creation, pipeline creation, present) across `__wine_unix_call` (or direct Mach-O helper callback).
2. **True Zero-Copy Unified Memory**:
   - All `memory.rs` / `xmem` / Xbox GPU heaps are allocated or rounded to Apple Silicon's 16 KiB page size (`vm_page_size = 16384`).
   - `victory_metal.so` wraps every GPU memory region once via `[device newBufferWithBytesNoCopy:ptr length:len options:MTLResourceStorageModeShared deallocator:nil]`.
   - **Result**: `copies` drops from ~250 MB/frame (`~85 ms`) -> **0 MB (`0 ms`)**, `copying back` drops to **0 ms**, and `watch.rs` page-guard faults vanish completely on buffers.
   - **Verified in M0 (below)**: works directly inside `x86_64` (Rosetta 2) processes on M4 Max.
3. **Direct DXIL -> `.metallib` via `libmetalirconverter.dylib`**:
   - Shipped inside `$WINE/../lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib` (Mach-O `x86_64`).
   - Verified C API symbols (checked directly via `otool -tV`):
     - `IRCompiler *IRCompilerCreate(void)` / `void IRCompilerDestroy(IRCompiler *)`
     - `IRObject *IRObjectCreateFromDXIL(const uint8_t *bytes, size_t size, IRBytecodeOwnership ownership)` (`1` = copy)
     - `IRShaderStage IRObjectGetMetalIRShaderStage(const IRObject *)`
     - `IRObject *IRCompilerAllocCompileAndLink(IRCompiler *, const char *entryPointOptional, const IRObject *input, IRError **err)`
     - `IRMetalLibBinary *IRMetalLibBinaryCreate(void)` / `void IRMetalLibBinaryDestroy(IRMetalLibBinary *)`
     - `bool IRObjectGetMetalLibBinary(const IRObject *, IRShaderStage stage, IRMetalLibBinary *out)`
     - `dispatch_data_t IRMetalLibGetBytecodeData(const IRMetalLibBinary *)` (pass directly to `[MTLDevice newLibraryWithData:err:]`)
     - `size_t IRMetalLibGetBytecodeSize(const IRMetalLibBinary *)`
     - `void IRCompilerSetGlobalRootSignature(IRCompiler *, const IRRootSignature *)` / `IRRootSignatureCreateFromDescriptor(...)`
     - `uint32_t IRErrorGetCode(const IRError *)` / `void IRErrorDestroy(IRError *)`

---

## 5. Parallel Work Tracks & Milestones

### Track 0 — Shared Quick Wins (Help Both D3DMetal & Native Metal)
1. **Pack/Sort `OSG1` Mesh Outputs (`crates/victory-shader/src/emit/lower/mesh.rs`)**:
   - `v5` showed ~20 pipelines sit at `33,280–33,792` bytes Metal-aligned (`512–1,024` B over `32,768`). Sorting arbitrary outputs (`kind == 0`) by decreasing component width (`vec4`, `vec2`, `scalar`) eliminates alignment padding between small scalars and `float4` vectors, recovering most remaining mesh pipelines.
2. **Continue Wave64 Lowering (`crates/victory-shader/src/emit/lower/{barriers,halves}.rs`)**:
   - Focus on the 30 `LDS one console wave64 shares without barriers` shaders (per-half disjoint LDS ranges) and 56 `lane mask read as data` shaders.
3. **`VICTORY_DIAGNOSTICS=1` Profile (D3DMetal visibility)**:
   - Run 1 session with `diag` build (`VICTORY_DIAGNOSTICS=1`) to break down the `~200 ms` unattributed `preparing` time (`SyncHeapsUs`, `HandlerUs`, `MirrorWaitUs`, host lock waits).

### Track 1 — Native Metal Backend (`victory_metal`) Milestones

- **Milestone M0 — Host Zero-Copy & IRConverter Proof-of-Concept** -> **COMPLETE (verified 2026-11-26)**
  - Deliverable: [`tools/macos/metal-poc/build-and-run.sh`](../tools/macos/metal-poc/build-and-run.sh).
  - Run: `tools/macos/metal-poc/build-and-run.sh`.
  - Verified M0 results on this machine (`Apple M4 Max`, `unified=1`, `maxThreadsPerGroup=1024`, under Rosetta 2 `x86_64`):
    - **Test 1 (Zero-Copy Unified Memory)**: `PASS` (`ptr=0x7fe848018000 len=65536 verified=1024 dwords host[0]=10 host[1023]=3079`) via `newBufferWithBytesNoCopy`.
    - **Test 2 (IRConverter DXIL -> `MTLLibrary`)**: `PASS` (`dxil_bytes=4288 stage=8 metallib_bytes=6748 functions=[main]`) converting a real cached compute shader via `libmetalirconverter.dylib`.

- **Milestone M1 — Wine Unixlib Bridge Skeleton + Headless Metal Device & Swapchain**
  - Goal: Prove `victory.exe` (PE `x86_64`) calls native `victory_metal.so` (Mach-O `x86_64`) in-process via Wine unixlib (`__wine_unix_call`) and presents a `CAMetalLayer` clear frame into Wine's HWND.
  - Key files to create/touch:
    - `tools/macos/patches/victory-metal-unixlib.patch` (or standalone unixlib loader shim tested via `build-crossover-wine.sh` — note `tools/macos/build-crossover-wine.sh` rebuilds cleanly in seconds if patching inside `dlls/` or loading a companion `.so` from `lib/wine/x86_64-unix/`).
    - `crates/victory-gpu/src/backend.rs` + `VICTORY_BACKEND=d3d12|metal` switch.
    - `metal-native/` native Mach-O crate (`objc2-metal` or direct ObjC C-ABI compiled to `victory_metal.so`).
  - Verification gate: Wine window clears to solid color at >= 60 fps and closes cleanly with 0 errors.

- **Milestone M2 — Zero-Copy Title Memory Buffers + Root Signatures + Compute Dispatch**
  - Goal: Wrap title memory allocations with `newBufferWithBytesNoCopy`, translate one cached compute DXIL via `libmetalirconverter.dylib`, bind IRConverter root descriptor/table buffers using Metal Argument Buffers (`IRRootSignatureCreateFromDescriptor`), and run a compute dispatch from `victory-gpu` writing back visible results into title memory with 0 `memcpy`.
  - Key invariants: Keep `lower.rs <= 524,288` bytes (`cargo xtask policy`); verify `victory-shader` + `victory-gpu` Wine tests pass.

- **Milestone M3 — Graphics + Textures + Indirect Draws & Mesh Pipelines**
  - Goal: Implement host texture cache (`MTLTexture` + tiled/linear uploads, depth twins, RTV/DSV render passes), `MTLRenderPipelineState` & `MTLMeshRenderPipelineDescriptor` (respecting 32 KiB limit), `executeIndirect` / Argument Buffers / heap indexing (`VICTORY_PORT_DESCRIPTORS=1` directly feeding Metal Argument Buffers).
  - Verification gate: GTA VI reaches the intro & title screen at 30+ fps, then loads into Vice City world.

- **Milestone M4 — Synchronization (`MTLSharedEvent`), Async Pipeline Warmup & Tuning**
  - Goal: Replace D3D12 `fence.rs` / `queue.rs` with `MTLSharedEvent` CPU/GPU wait & signal callbacks, warm up `MTLLibrary` & `MTLRenderPipelineState` asynchronously using `MTLBinaryArchive`, and benchmark world frame time vs D3DMetal (`v5` = 544–585 ms).

---

## 6. Hard Repository Rules (Never Violate)

1. Never commit anything from the game (`cargo xtask policy` must pass on every commit; `.m`/`.cpp`/`.c` files are blocked by `xtask/src/policy.rs`, so native macOS C/ObjC sources in `tools/macos/` either embed inline in `.sh` like `tools/macos/dxc-d3dmetal.sh` or live in an explicitly configured workspace crate only after updating policy by owner approval).
2. Keep `crates/victory-shader/src/emit/lower.rs <= 524,288` bytes.
3. Never modify `main` branch (stay on `mac-turbo` or feature branches off it) and never edit `victory.toml`.
4. Always run `victory.exe` detached without PTY (`< /dev/null`).
5. Never use `sudo` or weaken Wine/macOS security boundaries.
