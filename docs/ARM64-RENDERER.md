# ARM64 renderer handoff

Use `VIctory-Metal` to build an ARM64 renderer in its own process. Keep Wine as the x86-64 guest host and D3DMetal as the differential reference. The prior in-process Wine unixlib design uses the architecture of the Wine process and therefore keeps renderer CPU work under Rosetta on this stack.

## Verified on the M4 Max

Run from this checkout:

```sh
tools/macos/metal-poc/arm64-shared-buffer.sh
WINE="$PWD/../VIctory-macOS/target/wine-cx/bin/wine" tools/macos/metal-poc/arm64-shared-buffer.sh
VICTORY_PROBE_PLACEHOLDER=1 WINE="$PWD/../VIctory-macOS/target/wine-cx/bin/wine" \
  tools/macos/metal-poc/arm64-shared-buffer.sh
```

The probe builds synthetic executables with clang and, for Wine, `x86_64-w64-mingw32-gcc`. In the Wine variant the Win64 client uses two `MapViewOfFile` views of a 64 KiB file-backed section. The native ARM64 process maps that file with `MAP_SHARED`, wraps its own address with `newBufferWithBytesNoCopy` and runs an MSL kernel. The client verifies the GPU result through its second view after receiving completion. The processes exchange control bytes while accessing the shared payload in place. Each run verifies 2,293,760 words across 140 GPU dispatches. Timing excludes startup, CPU initialization and validation. Each reported phase has 120 samples after 20 warmup iterations.

| Client and transport | IPC round-trip p50 / p95 | IPC + GPU completion p50 / p95 |
| --- | ---: | ---: |
| Rosetta Mach-O, Unix socket pair | 8.0 / 9.3 µs | 132.4 / 464.3 µs |
| Win64 under CrossOver Wine, loopback TCP | 107.4 / 135.9 µs | 205.0 / 560.9 µs |
| Win64, placeholder at section offset 64 KiB | 108.6 / 159.0 µs | 204.2 / 570.6 µs |

The placeholder mode replaces the Win64 reservation at the requested virtual address with a 64 KiB view starting 64 KiB into the section. The client verifies the GPU writes through a separate alias and checks that the first 64 KiB retain their sentinel bytes. The renderer also draws a triangle into a private RGBA8 texture and uses a GPU blit to return the image through the shared buffer. The client verifies all 4,096 pixels, restores the placeholder, then remaps the view at the same address and verifies the pixels again. These are single-run synthetic latencies on macOS 27.0 (26A428). Five native `victory-metal` tests passed independently. The Wine prefix setup reports a Win32 `rundll32.exe` startup warning on this installed Wine build. The probe's Win64 client completes successfully.

Guest address translation, production pagefile-backed sections and on-screen presentation remain integration gates. A shared backing region can remove shadow copies for that region. The effect on frame time and on texture copies requires a complete, comparable frame.

### DXIL conversion across architectures

Run `tools/macos/metal-poc/arm64-dxil.sh` from this checkout with the local CrossOver Wine, DXC and IRConverter installations.

The synthetic compute shader compiles to 2,940 bytes of DXIL with an embedded `RootFlags(0), UAV(u0)` root signature. The x86-64 library runs in its own Rosetta process and serializes a 4,180-byte metallib. A separate ARM64 process loads it, binds the UAV by GPU address at byte 0 of Metal buffer slot 2 and verifies 1,024 results on each of two dispatches. This binding matches the probe's root signature. The title's root layouts, descriptor heaps, textures and graphics stages still need separate validation. [Apple Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)

## Implementation order

1. **Memory integration.** Prototype the file-backed allocation in `XMemAllocatePhysicalPages` and drive the real `map`, `map_alias` and `unmap` paths with synthetic pages. The existing `slice.index * PAGE` offset is relative to its section. Resolve guest virtual addresses in the Wine process and send backing allocation IDs, generations, byte offsets and lengths to ARM64. Keep each range alive until all referencing submission fences retire, including when the guest frees page numbers whose views are still mapped. Keep explicit copy handling for unshareable regions, and propagate missing mappings and renderer failure.
2. **Submission and display.** Version the protocol and batch command lists over a shared queue. Use IPC to signal work and completions. Specify ordering, queue backpressure and fence semantics before connecting the first caller in `victory-gpu`. Compare synthetic output pixels and GPU writes against the reference, then present a synthetic frame through the existing window. Evaluate IOSurface-backed presentation as a separate compatibility check.
3. **Shader conversion and binding.** Use the verified Rosetta compiler process as the starting point. Connect it to translated DXIL and cover root constants, descriptor tables, textures and samplers with synthetic shaders. Include the root layout, converter version, OS/GPU target and translation options in the cache key.
4. **Shader correctness in parallel.** Start with a paired mesh/pixel fixture that exceeds 32 KiB before packing. `container::signature` emits rows in register order, so reordering the input vector alone does not change this layout. Preserve guest export lookup and validate host register placement, DXIL element IDs, PSV0 and pixel linkage together. Measure byte counts and rendered pixels. For wave64, use lane-tagged synthetic tests with divergence and loops. Prove half-local LDS access or a legal convergence point before removing a rejection. Check upstream loop support against a pinned revision before extending it.
5. **Reference profiling.** The mac-turbo reference at `524d85a0` builds with `cargo build --locked --profile diag --target x86_64-pc-windows-gnullvm --package victory --features diagnostics`. Once authorized game inputs are available, repeat one fixed scene and collect descriptor refresh, page-guard wait, copies, queue submission and GPU time. Investigate missing region registration, addresses and lifetimes before changing readback withholding. The current completion path still handles bounded writes, overlapping held ranges and texture-read discovery during a hold. Measure `PORT_GUARD` or smaller root-view refreshes independently if counters identify these costs.

The game gate is correct geometry, materials, lighting and streaming with responsive controls, stable execution and at least 30 FPS in a repeatable world sequence. Use matched settings and report median and tail frame times. Loading screens and runs with diagnostic rendering shortcuts can establish individual milestones.

## Shader work from the first wave

- The experimental mesh worktree at `9ae9f920` reorders arbitrary elements within a register row. In the synthetic fixture the modelled OSG1 footprint falls from 33,280 to 32,768 bytes, with the same signature fields. The fixture passes on ARM64. The candidate stays isolated until a paired mesh and pixel pipeline uses the affected channels and produces the expected pixels on hardware. Review the shared pruning helper's effects on pixel inputs and primitive outputs before enabling it.
- The wave64 audit found a bounded case where `in_step` refuses a loop because a scalar derived from a lane read lacks a wave-uniformity proof. Test a uniform source and a lane-tagged divergent source before expanding the proof. Only certify a cross-half exchange after independently establishing that both halves reach it the same number of times.
