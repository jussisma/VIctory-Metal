#!/bin/sh
# Milestone M0: Proof-of-concept for a native Metal backend (VICTORY_BACKEND=metal).
# Tests under x86_64 (Rosetta 2, matching CrossOver Wine's architecture):
#   1. Zero-copy unified memory: posix_memalign(16384) -> newBufferWithBytesNoCopy
#      -> GPU compute shader -> direct CPU read back with 0 memcpy.
#   2. libmetalirconverter.dylib: dlopen Apple's x86_64 DXIL->MetalLib converter,
#      convert a cached DXIL shader from VIctory's cache (or a supplied .dxil path),
#      and create a real MTLLibrary on the Metal device.
#
# Kept inline in a shell script matching tools/macos/dxc-d3dmetal.sh so xtask
# asset policy stays untouched.
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WINE="${WINE:-$ROOT/target/wine-cx/bin/wine}"
[ -x "$WINE" ] || WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"
D3DMETAL_RES="${D3DMETAL_RES:-$(cd "$(dirname "$WINE")/.." && pwd)/lib/external/D3DMetal.framework/Versions/A/Resources}"
IRCONV_LIB="$D3DMETAL_RES/libmetalirconverter.dylib"

[ -f "$IRCONV_LIB" ] || {
    echo "Missing libmetalirconverter.dylib at $IRCONV_LIB" >&2
    exit 1
}

mkdir -p "$ROOT/target/macos"
BIN="$ROOT/target/macos/metal-poc"

/usr/bin/clang -arch x86_64 -O2 -fobjc-arc -ObjC -x objective-c - \
    -framework Metal -framework Foundation -ldl -o "$BIN" <<'OBJC'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct IRCompiler IRCompiler;
typedef struct IRObject IRObject;
typedef struct IRMetalLibBinary IRMetalLibBinary;
typedef struct IRError IRError;

// Metal Shader Converter C API signatures verified via otool on libmetalirconverter.dylib.
typedef IRCompiler *(*fn_IRCompilerCreate)(void);
typedef void (*fn_IRCompilerDestroy)(IRCompiler *);
typedef void (*fn_IRCompilerSetEntryPointName)(IRCompiler *, const char *);
typedef IRObject *(*fn_IRObjectCreateFromDXIL)(const uint8_t *, size_t, int ownership);
typedef void (*fn_IRObjectDestroy)(IRObject *);
typedef int (*fn_IRObjectGetMetalIRShaderStage)(const IRObject *);
typedef IRObject *(*fn_IRCompilerAllocCompileAndLink)(
    IRCompiler *, const char *entryPointOptional, const IRObject *input, IRError **error);
typedef IRMetalLibBinary *(*fn_IRMetalLibBinaryCreate)(void);
typedef void (*fn_IRMetalLibBinaryDestroy)(IRMetalLibBinary *);
typedef bool (*fn_IRObjectGetMetalLibBinary)(const IRObject *, int stage, IRMetalLibBinary *out);
typedef size_t (*fn_IRMetalLibGetBytecodeSize)(const IRMetalLibBinary *);
typedef dispatch_data_t (*fn_IRMetalLibGetBytecodeData)(const IRMetalLibBinary *);
typedef uint32_t (*fn_IRErrorGetCode)(const IRError *);
typedef const char *(*fn_IRErrorGetPayload)(const IRError *);
typedef void (*fn_IRErrorDestroy)(IRError *);

static int test_zero_copy(id<MTLDevice> device) {
    const size_t align = 16384; // Apple Silicon page size for NoCopy
    const size_t bytes = 65536;
    const uint32_t count = 1024;
    void *raw = NULL;
    if (posix_memalign(&raw, align, bytes) != 0 || !raw) {
        fprintf(stderr, "FAIL: posix_memalign failed\n");
        return 1;
    }
    memset(raw, 0, bytes);
    uint32_t *host = (uint32_t *)raw;
    for (uint32_t i = 0; i < count; i++) {
        host[i] = i + 1;
    }

    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:raw
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf) {
        fprintf(stderr, "FAIL: newBufferWithBytesNoCopy returned nil (Unified=%d)\n",
                (int)device.hasUnifiedMemory);
        free(raw);
        return 1;
    }

    NSString *src = @"#include <metal_stdlib>\n"
                     "using namespace metal;\n"
                     "kernel void m0_add(device uint *data [[buffer(0)]],\n"
                     "                   uint tid [[thread_position_in_grid]]) {\n"
                     "    data[tid] = data[tid] * 3u + 7u;\n"
                     "}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "FAIL: MSL compile: %s\n", err.localizedDescription.UTF8String);
        free(raw);
        return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"m0_add"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "FAIL: compute PSO: %s\n", err.localizedDescription.UTF8String);
        free(raw);
        return 1;
    }

    id<MTLCommandQueue> q = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "FAIL: GPU command buffer failed: %s\n",
                cb.error.localizedDescription.UTF8String);
        free(raw);
        return 1;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t want = (i + 1) * 3u + 7u;
        if (host[i] != want) {
            fprintf(stderr, "FAIL: zero-copy mismatch at %u: got=%u want=%u\n", i, host[i], want);
            free(raw);
            return 1;
        }
    }
    printf("PASS Test 1 (Zero-Copy Unified Memory): ptr=%p len=%zu verified=%u dwords host[0]=%u host[1023]=%u\n",
           raw, bytes, count, host[0], host[count - 1]);
    buf = nil;
    free(raw);
    return 0;
}

static int test_irconverter(id<MTLDevice> device, const char *lib_path, const char *dxil_path) {
    void *dl = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "FAIL: dlopen %s: %s\n", lib_path, dlerror());
        return 1;
    }
    #define LOAD(sym) fn_##sym p_##sym = (fn_##sym)dlsym(dl, #sym); \
        if (!p_##sym) { fprintf(stderr, "FAIL: missing symbol %s\n", #sym); return 1; }
    LOAD(IRCompilerCreate);
    LOAD(IRCompilerDestroy);
    LOAD(IRCompilerSetEntryPointName);
    LOAD(IRObjectCreateFromDXIL);
    LOAD(IRObjectDestroy);
    LOAD(IRObjectGetMetalIRShaderStage);
    LOAD(IRCompilerAllocCompileAndLink);
    LOAD(IRMetalLibBinaryCreate);
    LOAD(IRMetalLibBinaryDestroy);
    LOAD(IRObjectGetMetalLibBinary);
    LOAD(IRMetalLibGetBytecodeSize);
    LOAD(IRMetalLibGetBytecodeData);
    LOAD(IRErrorGetCode);
    LOAD(IRErrorGetPayload);
    LOAD(IRErrorDestroy);
    #undef LOAD

    IRCompiler *compiler = p_IRCompilerCreate();
    if (!compiler) {
        fprintf(stderr, "FAIL: IRCompilerCreate returned NULL\n");
        return 1;
    }

    FILE *f = fopen(dxil_path, "rb");
    if (!f) {
        perror(dxil_path);
        p_IRCompilerDestroy(compiler);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *dxil = (uint8_t *)malloc((size_t)sz);
    if (!dxil || fread(dxil, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "FAIL: reading %s\n", dxil_path);
        fclose(f);
        free(dxil);
        p_IRCompilerDestroy(compiler);
        return 1;
    }
    fclose(f);

    IRObject *input = p_IRObjectCreateFromDXIL(dxil, (size_t)sz, 1); // 1 = copy ownership
    free(dxil);
    if (!input) {
        fprintf(stderr, "FAIL: IRObjectCreateFromDXIL returned NULL\n");
        p_IRCompilerDestroy(compiler);
        return 1;
    }
    int stage = p_IRObjectGetMetalIRShaderStage(input);

    IRError *ir_err = NULL;
    IRObject *air = p_IRCompilerAllocCompileAndLink(compiler, NULL, input, &ir_err);
    if (!air) {
        uint32_t code = ir_err ? p_IRErrorGetCode(ir_err) : 0;
        const char *msg = ir_err ? p_IRErrorGetPayload(ir_err) : "unknown";
        fprintf(stderr, "FAIL: IRCompilerAllocCompileAndLink failed code=%u msg=%s\n",
                code, msg ? msg : "(null)");
        if (ir_err) p_IRErrorDestroy(ir_err);
        p_IRObjectDestroy(input);
        p_IRCompilerDestroy(compiler);
        return 1;
    }

    IRMetalLibBinary *bin = p_IRMetalLibBinaryCreate();
    if (!bin || !p_IRObjectGetMetalLibBinary(air, stage, bin)) {
        fprintf(stderr, "FAIL: IRObjectGetMetalLibBinary failed for stage=%d\n", stage);
        if (bin) p_IRMetalLibBinaryDestroy(bin);
        p_IRObjectDestroy(air);
        p_IRObjectDestroy(input);
        p_IRCompilerDestroy(compiler);
        return 1;
    }

    size_t metallib_bytes = p_IRMetalLibGetBytecodeSize(bin);
    dispatch_data_t data = p_IRMetalLibGetBytecodeData(bin);
    NSError *mtl_err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&mtl_err];
    if (!lib) {
        fprintf(stderr, "FAIL: newLibraryWithData (%zu bytes): %s\n",
                metallib_bytes, mtl_err.localizedDescription.UTF8String);
        p_IRMetalLibBinaryDestroy(bin);
        p_IRObjectDestroy(air);
        p_IRObjectDestroy(input);
        p_IRCompilerDestroy(compiler);
        return 1;
    }

    printf("PASS Test 2 (IRConverter DXIL -> MTLLibrary): dxil_bytes=%ld stage=%d metallib_bytes=%zu functions=[%s]\n",
           sz, stage, metallib_bytes,
           [lib.functionNames componentsJoinedByString:@", "].UTF8String);

    p_IRMetalLibBinaryDestroy(bin);
    p_IRObjectDestroy(air);
    p_IRObjectDestroy(input);
    p_IRCompilerDestroy(compiler);
    dlclose(dl);
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <libmetalirconverter.dylib> <sample.dxil>\n", argv[0]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "FAIL: MTLCreateSystemDefaultDevice returned nil\n");
            return 1;
        }
        printf("Metal Device: %s (unified=%d maxThreadsPerGroup=%lu)\n",
               device.name.UTF8String,
               (int)device.hasUnifiedMemory,
               (unsigned long)device.maxThreadsPerThreadgroup.width);
        if (test_zero_copy(device) != 0) return 1;
        if (test_irconverter(device, argv[1], argv[2]) != 0) return 1;
    }
    return 0;
}
OBJC

SAMPLE_DXIL="/tmp/victory-m0-sample.dxil"
if [ "${1:-}" != "" ]; then
    SAMPLE_DXIL="$1"
else
    CACHE_FILE=$(find "$HOME/Library/Caches/victory-macos/cache/shaders" -type f 2>/dev/null | head -n 1 || true)
    [ -n "$CACHE_FILE" ] || {
        echo "No cached DXIL shader found under $HOME/Library/Caches/victory-macos/cache/shaders; pass a .dxil path as arg 1" >&2
        exit 1
    }
    python3 - "$CACHE_FILE" "$SAMPLE_DXIL" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
magic = b"DX" + b"BC"
idx = data.find(magic)
assert idx >= 0, f"No container header in {src}"
total = struct.unpack_from("<I", data, idx + 24)[0]
open(dst, "wb").write(data[idx : idx + total])
PY
fi

exec "$BIN" "$IRCONV_LIB" "$SAMPLE_DXIL"
