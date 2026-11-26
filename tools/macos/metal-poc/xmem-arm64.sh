#!/bin/sh
# Cross-process proof for the native Metal gate: the actual Rust XMem HLE path
# runs under Wine (x86_64) and writes a payload through a map_alias view of a
# file-backed physical page at section offset 64 KiB. A native arm64 process
# waits for an atomic marker, then reads the same file through
# newBufferWithBytesNoCopy and verifies the bytes with a Metal compute shader.
#
# Usage: tools/macos/metal-poc/xmem-arm64.sh
# WINE is optional; tools/macos/test-wine.sh picks the checkout's Wine.
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; test-wine.sh expects it first on PATH too.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-xmem-arm64.XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

FIXTURE="$WORK/xmem-fixture.bin"
MARKER="$WORK/hle.ready"
RESULT="$WORK/gpu.result"

python3 - "$FIXTURE" <<'PY'
import sys

page = 65536
fixture = sys.argv[1]
page0 = b"\xa5" * page
# A distinct, independently reconstructible pattern for page 1: the guard
# word at 0x200 must survive the HLE write unchanged.
page1 = bytes(0x80 | (i & 0x3F) for i in range(page))
with open(fixture, "wb") as out:
    out.write(page0 + page1)
PY

cat > "$WORK/xmem_metal.m" <<'OBJC'
#ifdef __arm64__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

enum {
    page_bytes = 65536,
    guard_byte = 0x200,
    payload_byte = 0x400,
    payload_words = 1024,
    sentinel_byte = 0x2000,
};
static const uint32_t sentinel = 0xC0DED00D;

static uint32_t payload_word(uint32_t index) {
    return 0x51E30000u ^ (uint32_t)(index * 0x9E3779B9u);
}

static uint32_t guard_word(size_t at) {
    uint32_t word = 0;
    for (int k = 0; k < 4; k++)
        word |= (uint32_t)(0x80u | ((at + (size_t)k) & 0x3fu)) << (8 * k);
    return word;
}

static bool native_arm64(void) {
    int translated = 0;
    size_t size = sizeof(translated);
    if (sysctlbyname("sysctl.proc_translated", &translated, &size, NULL, 0) == 0 &&
        translated != 0) {
        fprintf(stderr, "FAIL: the process is translated, not native arm64\n");
        return false;
    }
    return true;
}

static bool wait_for_marker(const char *path, double seconds) {
    struct timespec start, now;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) return false;
    for (;;) {
        struct stat info;
        if (stat(path, &info) == 0) return true;
        if (errno != ENOENT) {
            fprintf(stderr, "marker stat failed: %s\n", strerror(errno));
            return false;
        }
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return false;
        double elapsed = (double)(now.tv_sec - start.tv_sec) +
                         (double)(now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed > seconds) {
            fprintf(stderr, "timed out after %.0f s waiting for %s\n", seconds, path);
            return false;
        }
        usleep(20000);
    }
}

static bool write_result(const char *path, uint32_t sum, uint32_t xor_sum,
                         uint32_t sentinel_value, uint32_t guard_value) {
    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", path) >= (int)sizeof(temporary)) {
        fprintf(stderr, "result path is too long\n");
        return false;
    }
    FILE *out = fopen(temporary, "w");
    if (!out) {
        fprintf(stderr, "opening %s failed: %s\n", temporary, strerror(errno));
        return false;
    }
    fprintf(out, "sum=%08x xor=%08x sentinel=%08x guard=%08x\n", sum, xor_sum,
            sentinel_value, guard_value);
    bool okay = fflush(out) == 0 && fsync(fileno(out)) == 0;
    okay = (fclose(out) == 0) && okay;
    if (!okay || rename(temporary, path) != 0) {
        fprintf(stderr, "publishing %s failed: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s <fixture> <marker> <result>\n", argv[0]);
            return 2;
        }
        if (!native_arm64()) return 1;
        // Atomic handoff: nothing below touches the file or Metal until the
        // HLE side has flushed its writes and published the marker.
        if (!wait_for_marker(argv[2], 180.0)) return 1;

        int file = open(argv[1], O_RDWR);
        if (file < 0) {
            fprintf(stderr, "opening %s failed: %s\n", argv[1], strerror(errno));
            return 1;
        }
        struct stat info;
        if (fstat(file, &info) != 0 || info.st_size < 2 * page_bytes) {
            fprintf(stderr, "FAIL: the fixture is not two 64 KiB pages\n");
            close(file);
            return 1;
        }
        void *mapping = mmap(NULL, (size_t)info.st_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             file, 0);
        close(file);
        if (mapping == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s\n", strerror(errno));
            return 1;
        }
        const uint8_t *page1 = (const uint8_t *)mapping + page_bytes;
        if (*(const uint32_t *)(page1 + guard_byte) != guard_word(guard_byte)) {
            fprintf(stderr, "FAIL: the original page-1 guard is missing at offset 64 KiB\n");
            munmap(mapping, (size_t)info.st_size);
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) {
            fprintf(stderr, "FAIL: no unified-memory Metal device\n");
            munmap(mapping, (size_t)info.st_size);
            return 1;
        }
        id<MTLBuffer> words = [device newBufferWithBytesNoCopy:(void *)page1
                                                        length:page_bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!words || words.contents != page1) {
            fprintf(stderr, "FAIL: newBufferWithBytesNoCopy rejected offset 64 KiB\n");
            munmap(mapping, (size_t)info.st_size);
            return 1;
        }

        NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void xmem_probe(device const uint *words [[buffer(0)]],\n"
             "                       device uint *out [[buffer(1)]],\n"
             "                       uint tid [[thread_position_in_grid]]) {\n"
             "    if (tid == 0) {\n"
             "        uint sum = 0, x = 0;\n"
             "        for (uint i = 0; i < 1024; ++i) {\n"
             "            uint w = words[256 + i];\n"
             "            sum += w; x ^= w;\n"
             "        }\n"
             "        out[0] = sum;\n"
             "        out[1] = x;\n"
             "        out[2] = words[2048];\n"
             "        out[3] = words[128];\n"
             "    }\n"
             "}\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"xmem_probe"];
        id<MTLComputePipelineState> pipeline =
            function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> output = [device newBufferWithLength:4 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        if (!library || !function || !pipeline || !queue || !output) {
            fprintf(stderr, "FAIL: Metal setup: %s\n",
                    error ? error.localizedDescription.UTF8String : "unavailable");
            words = nil;
            munmap(mapping, (size_t)info.st_size);
            return 1;
        }

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:words offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "FAIL: GPU execution: %s\n",
                    command.error.localizedDescription.UTF8String);
            words = nil;
            munmap(mapping, (size_t)info.st_size);
            return 1;
        }

        uint32_t expected_sum = 0, expected_xor = 0;
        for (uint32_t i = 0; i < payload_words; i++) {
            expected_sum += payload_word(i);
            expected_xor ^= payload_word(i);
        }
        const uint32_t *result = (const uint32_t *)output.contents;
        uint32_t sum = result[0], xor_sum = result[1];
        uint32_t seen_sentinel = result[2], seen_guard = result[3];
        bool okay = sum == expected_sum && xor_sum == expected_xor &&
                    seen_sentinel == sentinel && seen_guard == guard_word(guard_byte);
        printf("GPU result offset=%d sum=%08x xor=%08x sentinel=%08x guard=%08x "
               "expected_sum=%08x expected_xor=%08x\n",
               page_bytes, sum, xor_sum, seen_sentinel, seen_guard, expected_sum,
               expected_xor);
        fflush(stdout);
        if (!okay) {
            fprintf(stderr, "FAIL: the GPU did not read the HLE payload exactly\n");
        } else if (!write_result(argv[3], sum, xor_sum, seen_sentinel, seen_guard)) {
            okay = false;
        } else {
            printf("PASS native=arm64 device=%s offset=%d no_copy=1 checksum=%08x sentinel=%08x\n",
                   device.name.UTF8String, page_bytes, sum, seen_sentinel);
            fflush(stdout);
        }
        words = nil;
        munmap(mapping, (size_t)info.st_size);
        return okay ? 0 : 1;
    }
}
#else
#error "This probe must be built for native arm64"
#endif
OBJC

/usr/bin/clang -arch arm64 -O2 -Wall -Wextra -Werror -fobjc-arc \
    "$WORK/xmem_metal.m" -framework Metal -framework Foundation -o "$WORK/xmem-metal"

echo "Building the x86_64 Windows HLE test binary..."
"$ROOT/tools/macos/test-wine.sh" --locked -p victory-hle --lib \
    external_fixture_alias_write_for_native_gpu --no-run

python3 - "$ROOT" "$WORK" "$FIXTURE" "$MARKER" "$RESULT" <<'PY'
import os
import pathlib
import re
import struct
import subprocess
import sys

root, work, fixture, marker, result = map(pathlib.Path, sys.argv[1:])
page = 65536
payload_byte = 0x400
sentinel_byte = 0x2000
guard_byte = 0x200
sentinel = 0xC0DED00D


def windows_path(path):
    return "Z:" + str(path).replace("/", "\\")


def expected_payload():
    return [(0x51E30000 ^ ((i * 0x9E3779B9) & 0xFFFFFFFF)) for i in range(1024)]


def fail(message):
    sys.exit("FAIL: " + message)


env = dict(os.environ)
env["VICTORY_XMEM_GPU_FIXTURE"] = windows_path(fixture)
env["VICTORY_XMEM_GPU_MARKER"] = windows_path(marker)

native = subprocess.Popen(
    [str(work / "xmem-metal"), str(fixture), str(marker), str(result)],
    stdin=subprocess.DEVNULL,
)
try:
    test = subprocess.run(
        [
            str(root / "tools/macos/test-wine.sh"),
            "--locked",
            "-p",
            "victory-hle",
            "--lib",
            "external_fixture_alias_write_for_native_gpu",
            "--",
            "--nocapture",
            "--test-threads=1",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    if test.returncode:
        fail(f"the Rust HLE test exited with {test.returncode}")
    try:
        native_status = native.wait(timeout=60)
    except subprocess.TimeoutExpired:
        fail("the native arm64 process did not finish within 60 s of the HLE test")
    if native_status:
        fail(f"the native arm64 process exited with {native_status}")
finally:
    if native.poll() is None:
        native.kill()
        native.wait()

if not marker.is_file():
    fail("the HLE marker was not published")

data = fixture.read_bytes()
if data[:page] != b"\xa5" * page:
    fail("the HLE path modified page 0")
page1 = data[page : 2 * page]
words = list(struct.unpack_from("<1024I", page1, payload_byte))
if words != expected_payload():
    fail("the HLE payload is not in the fixture at offset 64 KiB")
if struct.unpack_from("<I", page1, sentinel_byte)[0] != sentinel:
    fail("the HLE sentinel is not in the fixture")
expected_guard = bytes(0x80 | ((guard_byte + i) & 0x3F) for i in range(4))
if page1[guard_byte : guard_byte + 4] != expected_guard:
    fail("the HLE write disturbed the guard bytes")

if not result.is_file():
    fail("the native GPU result file was not published")
text = result.read_text().strip()
match = re.fullmatch(
    r"sum=([0-9a-f]{8}) xor=([0-9a-f]{8}) sentinel=([0-9a-f]{8}) guard=([0-9a-f]{8})",
    text,
)
if not match:
    fail(f"unexpected GPU result file: {text!r}")
sum_gpu, xor_gpu, sentinel_gpu, guard_gpu = (int(v, 16) for v in match.groups())
expected_sum = sum(expected_payload()) & 0xFFFFFFFF
expected_xor = 0
for word in expected_payload():
    expected_xor ^= word
if (sum_gpu, xor_gpu, sentinel_gpu, guard_gpu) != (
    expected_sum,
    expected_xor,
    sentinel,
    int.from_bytes(expected_guard, "little"),
):
    fail(f"the GPU result does not match the HLE payload: {text}")

print(
    "PASS xmem-arm64 offset=65536 sum=%08x xor=%08x sentinel=%08x guard=%08x"
    % (sum_gpu, xor_gpu, sentinel_gpu, guard_gpu)
)
PY
