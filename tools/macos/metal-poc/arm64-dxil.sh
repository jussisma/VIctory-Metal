#!/bin/sh
# DS-DXIL: synthetic DXIL -> x86_64 Apple IRConverter -> arm64 Metal execution.
#
# Two separate address spaces, bytes only:
#   1. dcx (Win64 under Wine) compiles original synthetic HLSL to DXIL.
#   2. An x86_64 Mach-O process (Rosetta 2) loads libmetalirconverter.dylib,
#      converts the DXIL to a metallib, and serializes the bytes to a file.
#   3. A separate arm64 Mach-O process loads that metallib, creates a compute
#      PSO, binds the converted UAV through the IRConverter argument buffer and
#      dispatches. The GPU output is verified against known values.
#
# Root signature: authored in HLSL via [RootSignature(...)] and embedded in the
# DXIL. IRCompilerSetGlobalRootSignature is intentionally not called; the
# converter consumes the embedded root signature.
#
# Converted binding ABI observed for this root signature (compute):
#   buffer(2) = top-level global argument buffer (top_level_global_ab)
#   offset 0  = 8-byte device address of the UAV returned by gpuAddress
#   resource must be made resident on the encoder (useResource)
#
# Synthetic inputs only; no game assets, shaders or caches. No network.
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
COUNT=1024

WINE="${WINE:-$ROOT/target/wine-cx/bin/wine}"
if [ ! -x "$WINE" ]; then
    WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"
fi
if [ ! -x "$WINE" ]; then
    WINE="$(cd "$ROOT/../../.." && pwd)/VIctory-macOS/target/wine-cx/bin/wine"
fi
[ -x "$WINE" ] || {
    echo "Missing Wine binary; set WINE=/path/to/wine" >&2
    exit 1
}

DXC="${DXC:-$ROOT/toolchains/dxc/bin/x64/dxc.exe}"
if [ ! -f "$DXC" ]; then
    DXC="$(cd "$ROOT/../../.." && pwd)/toolchains/dxc/bin/x64/dxc.exe"
fi
[ -f "$DXC" ] || {
    echo "Missing dxc.exe; set DXC=/path/to/dxc.exe" >&2
    exit 1
}

D3DMETAL_RES="${D3DMETAL_RES:-$(cd "$(dirname "$WINE")/.." && pwd)/lib/external/D3DMetal.framework/Versions/A/Resources}"
IRCONV_LIB="${VICTORY_METAL_IRCONVERTER:-$D3DMETAL_RES/libmetalirconverter.dylib}"
[ -f "$IRCONV_LIB" ] || {
    echo "Missing libmetalirconverter.dylib at $IRCONV_LIB" >&2
    exit 1
}
WINESERVER="${WINESERVER:-$(dirname "$WINE")/wineserver}"

work="$(mktemp -d "${TMPDIR:-/tmp}/victory-dxil-arm64.XXXXXXXX")"
prefix="$work/wineprefix"
cleanup() {
    if [ -d "$prefix" ] && [ -x "$WINESERVER" ]; then
        WINEPREFIX="$prefix" "$WINESERVER" -w >/dev/null 2>&1 || true
    fi
    rm -rf "$work"
}
trap cleanup EXIT INT TERM

echo "== DS-DXIL synthetic DXIL -> x86_64 IRConverter -> arm64 execution =="
echo "host_arch=$(uname -m) library=$IRCONV_LIB"
echo "library_archs=$(/usr/bin/lipo -archs "$IRCONV_LIB" 2>/dev/null || echo unknown)"

# Original synthetic shader: one UAV at u0, arithmetic with known results.
cat > "$work/shader.hlsl" <<'HLSL'
#define RS "RootFlags(0), UAV(u0)"

RWStructuredBuffer<uint> gOut : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    gOut[tid.x] = tid.x * 3u + 7u;
}
HLSL

# --- Stage 1: Win64 dxc under a private Wine prefix -------------------------
export WINEPREFIX="$prefix"
export WINEDEBUG="-all"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d"
"$WINE" wineboot -i >/dev/null 2>&1 || true
(cd "$work" && "$WINE" "$DXC" -T cs_6_0 -E main -Fo shader.dxil shader.hlsl) || {
    echo "dxc failed" >&2
    exit 1
}
dxil_bytes=$(wc -c < "$work/shader.dxil" | tr -d ' ')
echo "dxc: -T cs_6_0 -E main stage=compute dxil_bytes=$dxil_bytes"

# --- Stage 2: x86_64 IRConverter process ------------------------------------
cat > "$work/convert.m" <<'OBJC'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysctl.h>

typedef struct IRCompiler IRCompiler;
typedef struct IRObject IRObject;
typedef struct IRMetalLibBinary IRMetalLibBinary;
typedef struct IRError IRError;

static int translated(void) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname("sysctl.proc_translated", &value, &size, NULL, 0) == 0 ? value : -1;
}

static void *need(void *dl, const char *name) {
    void *sym = dlsym(dl, name);
    if (!sym) {
        fprintf(stderr, "FAIL: missing symbol %s\n", name);
        exit(1);
    }
    return sym;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4) return 2;
        void *dl = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!dl) {
            fprintf(stderr, "FAIL: dlopen: %s\n", dlerror());
            return 1;
        }
        IRCompiler *(*compiler_create)(void) = need(dl, "IRCompilerCreate");
        void (*compiler_destroy)(IRCompiler *) = need(dl, "IRCompilerDestroy");
        IRObject *(*from_dxil)(const uint8_t *, size_t, int) = need(dl, "IRObjectCreateFromDXIL");
        void (*object_destroy)(IRObject *) = need(dl, "IRObjectDestroy");
        int (*get_stage)(const IRObject *) = need(dl, "IRObjectGetMetalIRShaderStage");
        IRObject *(*compile_and_link)(IRCompiler *, const char *, const IRObject *, IRError **) =
            need(dl, "IRCompilerAllocCompileAndLink");
        IRMetalLibBinary *(*binary_create)(void) = need(dl, "IRMetalLibBinaryCreate");
        void (*binary_destroy)(IRMetalLibBinary *) = need(dl, "IRMetalLibBinaryDestroy");
        bool (*get_binary)(const IRObject *, int, IRMetalLibBinary *) =
            need(dl, "IRObjectGetMetalLibBinary");
        size_t (*bytecode_size)(const IRMetalLibBinary *) = need(dl, "IRMetalLibGetBytecodeSize");
        dispatch_data_t (*bytecode_data)(const IRMetalLibBinary *) =
            need(dl, "IRMetalLibGetBytecodeData");
        uint32_t (*error_code)(const IRError *) = need(dl, "IRErrorGetCode");
        const char *(*error_payload)(const IRError *) = need(dl, "IRErrorGetPayload");
        void (*error_destroy)(IRError *) = need(dl, "IRErrorDestroy");

        FILE *f = fopen(argv[2], "rb");
        if (!f) { perror(argv[2]); return 1; }
        fseek(f, 0, SEEK_END);
        long dxil_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *dxil = malloc((size_t)dxil_size);
        if (!dxil || fread(dxil, 1, (size_t)dxil_size, f) != (size_t)dxil_size) {
            fprintf(stderr, "FAIL: reading %s\n", argv[2]);
            return 1;
        }
        fclose(f);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }

        IRCompiler *compiler = compiler_create();
        IRObject *input = from_dxil(dxil, (size_t)dxil_size, 1); // ownership copy
        free(dxil);
        if (!compiler || !input) { fprintf(stderr, "FAIL: compiler/input creation\n"); return 1; }
        int stage = get_stage(input);

        IRError *error = NULL;
        IRObject *air = compile_and_link(compiler, "main", input, &error);
        if (!air) {
            fprintf(stderr, "FAIL: compile+link code=%u payload=%s\n",
                    error ? error_code(error) : 0,
                    error ? error_payload(error) : "(none)");
            return 1;
        }
        IRMetalLibBinary *binary = binary_create();
        if (!binary || !get_binary(air, stage, binary)) {
            fprintf(stderr, "FAIL: IRObjectGetMetalLibBinary stage=%d\n", stage);
            return 1;
        }
        size_t bytes = bytecode_size(binary);
        dispatch_data_t data = bytecode_data(binary);
        NSError *ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:data error:&ns_error];
        if (!library) {
            fprintf(stderr, "FAIL: x86_64 library load: %s\n",
                    ns_error.localizedDescription.UTF8String);
            return 1;
        }

        const void *mapped = NULL;
        size_t mapped_len = 0;
        dispatch_data_t flat = dispatch_data_create_map(data, &mapped, &mapped_len);
        (void)flat;
        if (!mapped || mapped_len == 0) {
            fprintf(stderr, "FAIL: empty metallib\n");
            return 1;
        }
        FILE *out = fopen(argv[3], "wb");
        if (!out || fwrite(mapped, 1, mapped_len, out) != mapped_len) {
            fprintf(stderr, "FAIL: writing %s\n", argv[3]);
            return 1;
        }
        fclose(out);

        printf("IRCONVERT_OK pid=%d proc_translated=%d device=%s dxil_bytes=%ld "
               "stage=%d metallib_bytes=%zu functions=[%s]\n",
               getpid(), translated(), device.name.UTF8String, dxil_size, stage,
               bytes, [[library.functionNames componentsJoinedByString:@","] UTF8String]);
        binary_destroy(binary);
        object_destroy(air);
        object_destroy(input);
        compiler_destroy(compiler);
        if (error) error_destroy(error);
        dlclose(dl);
    }
    return 0;
}
OBJC

/usr/bin/clang -arch x86_64 -O2 -fobjc-arc -ObjC "$work/convert.m" \
    -framework Metal -framework Foundation -ldl -o "$work/convert"
"$work/convert" "$IRCONV_LIB" "$work/shader.dxil" "$work/shader.metallib" \
    | tee "$work/convert.log"
metallib_bytes=$(wc -c < "$work/shader.metallib" | tr -d ' ')
[ "$metallib_bytes" -gt 0 ] || {
    echo "FAIL: converter produced no metallib" >&2
    exit 1
}
convert_pid=$(sed -n 's/.*IRCONVERT_OK pid=\([0-9]*\).*/\1/p' "$work/convert.log")
[ -n "$convert_pid" ] || { echo "FAIL: could not parse compiler pid" >&2; exit 1; }

# --- Stage 3: separate arm64 renderer process -------------------------------
cat > "$work/render.m" <<'OBJC'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

static int translated(void) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname("sysctl.proc_translated", &value, &size, NULL, 0) == 0 ? value : -1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        const uint32_t count = 1024;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }

        NSData *bytes = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
        if (!bytes.length) { fprintf(stderr, "FAIL: reading metallib\n"); return 1; }
        dispatch_data_t dispatch_bytes =
            dispatch_data_create(bytes.bytes, bytes.length, dispatch_get_main_queue(), ^{});
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:dispatch_bytes error:&error];
        if (!library) {
            fprintf(stderr, "FAIL: arm64 library load: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        if (library.functionNames.count != 1) {
            fprintf(stderr, "FAIL: expected one function, got [%s]\n",
                    [[library.functionNames componentsJoinedByString:@","] UTF8String]);
            return 1;
        }
        NSString *entry = library.functionNames.firstObject;
        id<MTLFunction> function = [library newFunctionWithName:entry];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "FAIL: PSO: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLBuffer> data = [device newBufferWithLength:count * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> arguments = [device newBufferWithLength:32
                                                      options:MTLResourceStorageModeShared];
        if (!data || !arguments) { fprintf(stderr, "FAIL: buffer allocation\n"); return 1; }
        memset(data.contents, 0, count * sizeof(uint32_t));
        memset(arguments.contents, 0, 32);
        uint64_t address = data.gpuAddress;
        memcpy(arguments.contents, &address, sizeof(address));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (int repetition = 0; repetition < 2; repetition++) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            // Converted ABI: top-level argument buffer at buffer(2), UAV address at offset 0.
            [encoder setBuffer:arguments offset:0 atIndex:2];
            [encoder useResource:data usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "FAIL: GPU execution: %s\n",
                        command.error.localizedDescription.UTF8String);
                return 1;
            }
            uint32_t *out = data.contents;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t want = i * 3u + 7u;
                if (out[i] != want) {
                    fprintf(stderr, "FAIL: repetition=%d index=%u got=%u want=%u\n",
                            repetition, i, out[i], want);
                    return 1;
                }
            }
        }

        uint32_t *out = data.contents;
        printf("RENDER_OK pid=%d arch=arm64 proc_translated=%d device=%s entry=%s "
               "metallib_bytes=%lu words=%u arg_buffer_index=2 uav_offset=0 "
               "first=%u last=%u\n",
               getpid(), translated(), device.name.UTF8String, entry.UTF8String,
               (unsigned long)bytes.length, count, out[0], out[count - 1]);
    }
    return 0;
}
OBJC

/usr/bin/clang -arch arm64 -O2 -fobjc-arc "$work/render.m" \
    -framework Metal -framework Foundation -o "$work/render"
"$work/render" "$work/shader.metallib" | tee "$work/render.log"
render_pid=$(sed -n 's/.*RENDER_OK pid=\([0-9]*\).*/\1/p' "$work/render.log")
[ -n "$render_pid" ] || { echo "FAIL: could not parse renderer pid" >&2; exit 1; }
[ "$render_pid" != "$convert_pid" ] || {
    echo "FAIL: converter and renderer share a pid ($convert_pid)" >&2
    exit 1
}

echo "PASS separate_processes=1 converter_pid=$convert_pid renderer_pid=$render_pid"
echo "PASS stage=compute root_sig=embedded-dxil binding=argument-buffer(2)/uav0 gpu_verified=$COUNT"
