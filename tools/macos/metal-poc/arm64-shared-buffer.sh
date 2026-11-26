#!/bin/sh
# Synthetic x86_64 -> native arm64 GPU buffer test. Set WINE to exercise a Win64 client.
set -eu
work="$(mktemp -d "${TMPDIR:-/tmp}/victory-metal-shared.XXXXXXXX")"
trap 'rm -rf "$work"' EXIT

cat > "$work/probe.m" <<'OBJC'
#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#include <winsock2.h>
#include <windows.h>
#include <memoryapi.h>
#else
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <errno.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { word_count = 16384, samples = 120, warmup = 20, image_width = 64 };
static const size_t length = word_count * sizeof(uint32_t);

#ifdef _WIN32
typedef SOCKET Channel;
#else
typedef int Channel;
#endif

static bool exchange(Channel fd, bool sending, char tag) {
    char byte = tag;
#ifdef _WIN32
    int result = sending ? send(fd, &byte, 1, 0) : recv(fd, &byte, 1, 0);
#else
    ssize_t result;
    do {
        result = sending ? write(fd, &byte, 1) : read(fd, &byte, 1);
    } while (result < 0 && errno == EINTR);
#endif
    return result == 1 && byte == tag;
}

#if defined(__arm64__)
static bool draw_triangle(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                          id<MTLBuffer> buffer) {
    MTLRenderPipelineDescriptor *pipeline_desc = [MTLRenderPipelineDescriptor new];
    pipeline_desc.vertexFunction = [library newFunctionWithName:@"triangle_vertex"];
    pipeline_desc.fragmentFunction = [library newFunctionWithName:@"triangle_fragment"];
    pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipeline_desc
                                                                               error:&error];
    if (!pipeline) {
        fprintf(stderr, "Triangle pipeline failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }
    MTLTextureDescriptor *texture_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:image_width height:image_width
        mipmapped:NO];
    texture_desc.usage = MTLTextureUsageRenderTarget;
    texture_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [device newTextureWithDescriptor:texture_desc];
    if (!texture) return false;

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 1, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return false;
    [encoder setRenderPipelineState:pipeline];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];

    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!blit) return false;
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:MTLSizeMake(image_width, image_width, 1) toBuffer:buffer
        destinationOffset:0 destinationBytesPerRow:image_width * sizeof(uint32_t)
        destinationBytesPerImage:image_width * image_width * sizeof(uint32_t)];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "Triangle execution failed: %s\n", command.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static int render(int channel, uint32_t *words) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> buffer =
        [device newBufferWithBytesNoCopy:words length:length
        options:MTLResourceStorageModeShared deallocator:nil];
    if (!device.hasUnifiedMemory || !buffer || buffer.contents != words) {
        fprintf(stderr, "Metal rejected the shared mapping or returned a different address\n");
        return 1;
    }
    NSError *error = nil;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void update(device uint *words [[buffer(0)]], "
        "uint index [[thread_position_in_grid]]) { words[index] = words[index] * 3u + 7u; }\n"
        "vertex float4 triangle_vertex(uint index [[vertex_id]]) {\n"
        "  constexpr float2 vertices[] = { {-1,-1}, {1,-1}, {-1,1} };\n"
        "  return float4(vertices[index], 0, 1);\n"
        "}\n"
        "fragment float4 triangle_fragment() { return float4(1, 0, 0, 1); }\n";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"update"];
    if (!function) {
        fprintf(stderr, "MSL compilation failed: %s\n", error.localizedDescription.UTF8String);
        return 1;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!pipeline || !queue || pipeline.maxTotalThreadsPerThreadgroup < 64) {
        fprintf(stderr, "Metal pipeline/queue unavailable: %s\n",
                error ? error.localizedDescription.UTF8String : "device capability");
        return 1;
    }
    printf("renderer=arm64 device=%s buffer_bytes=%zu simd_width=%lu\n",
           device.name.UTF8String, length, (unsigned long)pipeline.threadExecutionWidth);
    fflush(stdout);
    if (!exchange(channel, true, 'R')) return 1;
    for (unsigned run = 0; run < 2 * (samples + warmup); run++) {
        const bool dispatch = run >= samples + warmup;
        if (!exchange(channel, false, dispatch ? 'G' : 'P')) return 1;
        if (dispatch) @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) return 1;
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(word_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "GPU execution failed: %s\n", command.error.localizedDescription.UTF8String);
                return 1;
            }
        }
        if (!exchange(channel, true, 'D')) return 1;
    }
    bool rendered = exchange(channel, false, 'T') && draw_triangle(device, queue, library, buffer) &&
                    exchange(channel, true, 'D');
    if (!rendered || !exchange(channel, false, 'Q')) return 1;
    return 0;
}
#define run_probe render
#elif defined(__x86_64__)
static int compare_ticks(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left, b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t clock_ticks(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    return (uint64_t)ticks.QuadPart;
#else
    return mach_absolute_time();
#endif
}

static bool verify_triangle(const uint32_t *pixels) {
    for (unsigned y = 0; y < image_width; y++) {
        for (unsigned x = 0; x < image_width; x++) {
            const uint32_t red = 0xff0000ff, blue = 0xffff0000;
            const uint32_t value = pixels[y * image_width + x];
            // Pixel centers on the diagonal may follow either edge convention.
            const bool valid = x == y ? value == red || value == blue : value == (x < y ? red : blue);
            if (!valid) {
                fprintf(stderr, "Triangle pixel mismatch at (%u, %u): %08x\n", x, y, value);
                return false;
            }
        }
    }
    return true;
}

static int drive(Channel channel, uint32_t *write_view, const uint32_t *read_view) {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency)) return 1;
    const double us = 1000000.0 / frequency.QuadPart;
    const char *client = "win64-wine";
#else
    int translated = 0;
    size_t size = sizeof(translated);
    if (sysctlbyname("sysctl.proc_translated", &translated, &size, NULL, 0) || translated != 1) {
        fprintf(stderr, "The x86_64 client must run under Rosetta\n");
        return 1;
    }
    mach_timebase_info_data_t timebase;
    if (mach_timebase_info(&timebase) != KERN_SUCCESS) return 1;
    const double us = (double)timebase.numer / timebase.denom / 1000.0;
    const char *client = "x86_64-rosetta";
#endif
    if (!exchange(channel, false, 'R')) return 1;
    for (unsigned dispatch = 0; dispatch <= 1; dispatch++) {
        uint64_t ticks[samples];
        for (unsigned run = 0; run < samples + warmup; run++) {
            if (dispatch) {
                for (uint32_t i = 0; i < word_count; i++) write_view[i] = run * word_count + i;
            }
            uint64_t start = clock_ticks();
            if (!exchange(channel, true, dispatch ? 'G' : 'P') || !exchange(channel, false, 'D')) return 1;
            uint64_t elapsed = clock_ticks() - start;
            if (dispatch) {
                for (uint32_t i = 0; i < word_count; i++) {
                    if (read_view[i] != (run * word_count + i) * 3u + 7u) {
                        fprintf(stderr, "GPU result mismatch: run=%u word=%u\n", run, i);
                        return 1;
                    }
                }
            }
            if (run >= warmup) ticks[run - warmup] = elapsed;
        }
        qsort(ticks, samples, sizeof(ticks[0]), compare_ticks);
        printf("client=%s scenario=%s samples=%u warmup=%u p50_us=%.3f p95_us=%.3f p99_us=%.3f\n",
               client, dispatch ? "ipc_gpu_completion" : "ipc_round_trip", samples, warmup,
               ticks[(50 * samples + 99) / 100 - 1] * us,
               ticks[(95 * samples + 99) / 100 - 1] * us,
               ticks[(99 * samples + 99) / 100 - 1] * us);
    }
    printf("PASS verified_words=%u app_copy_bytes=0\n", (samples + warmup) * word_count);
    if (!exchange(channel, true, 'T') || !exchange(channel, false, 'D') || !verify_triangle(read_view)) {
        return 1;
    }
    printf("PASS triangle_verified_pixels=%u\n", image_width * image_width);
    return 0;
}
#define run_probe(channel, mapping) drive(channel, mapping, mapping)
#else
#error Unsupported architecture
#endif

#ifdef _WIN32
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    const DWORD offset = strtoul(argv[3], NULL, 10);
    HANDLE file = CreateFileA(argv[1], GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFile failed: %lu\n", GetLastError());
        return 1;
    }
    HANDLE section = CreateFileMappingW(file, NULL, PAGE_READWRITE, 0, 0, NULL);
    CloseHandle(file);
    if (!section) {
        fprintf(stderr, "CreateFileMapping failed: %lu\n", GetLastError());
        return 1;
    }
    uint32_t *write_view;
    void *reserved = NULL;
    if (offset) {
        reserved = VirtualAlloc2(GetCurrentProcess(), NULL, length, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                 PAGE_NOACCESS, NULL, 0);
        if (!reserved) {
            fprintf(stderr, "VirtualAlloc2 placeholder failed: %lu\n", GetLastError());
            return 1;
        }
        write_view = MapViewOfFile3(section, GetCurrentProcess(), reserved, offset, length,
                                    MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
        if (write_view != reserved) {
            fprintf(stderr, "MapViewOfFile3 placeholder replacement failed: %lu\n", GetLastError());
            return 1;
        }
        printf("win64 placeholder replaced at requested address, section_offset=%lu\n", offset);
    } else {
        write_view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, length);
    }
    const uint32_t *read_view = MapViewOfFile(section, FILE_MAP_READ, 0, offset, length);
    if (!write_view || !read_view || write_view == read_view) {
        fprintf(stderr, "MapViewOfFile failed to create two views: %lu\n", GetLastError());
        return 1;
    }
    WSADATA winsock;
    if (WSAStartup(MAKEWORD(2, 2), &winsock)) return 1;
    SOCKET channel = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_port = htons(atoi(argv[2]))};
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int no_delay = 1;
    if (channel == INVALID_SOCKET ||
        setsockopt(channel, IPPROTO_TCP, TCP_NODELAY, (const char *)&no_delay, sizeof(no_delay)) ||
        connect(channel, (const struct sockaddr *)&address, sizeof(address))) {
        fprintf(stderr, "WinSock connection failed: %d\n", WSAGetLastError());
        return 1;
    }
    int result = drive(channel, write_view, read_view);
    if (!result && reserved) {
        if (!UnmapViewOfFile2(GetCurrentProcess(), write_view, MEM_PRESERVE_PLACEHOLDER)) {
            fprintf(stderr, "UnmapViewOfFile2 placeholder preservation failed: %lu\n", GetLastError());
            return 1;
        }
        if (!verify_triangle(read_view)) return 1;
        write_view = MapViewOfFile3(section, GetCurrentProcess(), reserved, offset, length,
                                    MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
        if (write_view != reserved || !verify_triangle(write_view)) {
            fprintf(stderr, "Restoring the mapping at its original address failed\n");
            return 1;
        }
        printf("PASS restored placeholder and preserved aliased data\n");
    }
    if (!result && !exchange(channel, true, 'Q')) result = 1;
    closesocket(channel);
    WSACleanup();
    UnmapViewOfFile(read_view);
    UnmapViewOfFile(write_view);
    CloseHandle(section);
    return result;
}
#else
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4) return 1;
        int channel = atoi(argv[1]), memory = atoi(argv[2]);
        off_t offset = strtol(argv[3], NULL, 10);
        void *mapping = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, memory, offset);
        if (mapping == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
        int result = run_probe(channel, mapping);
#if defined(__x86_64__)
        if (!result && !exchange(channel, true, 'Q')) result = 1;
#endif
        munmap(mapping, length);
        return result;
    }
}
#endif
OBJC

build_macos() {
    arch="$1"
    /usr/bin/clang -arch "$arch" -O2 -Wall -Wextra -Werror -fobjc-arc \
        "$work/probe.m" -framework Metal -framework Foundation -o "$work/$arch"
}

build_macos arm64
if [ -n "${WINE:-}" ]; then
    "${MINGW_CC:-x86_64-w64-mingw32-gcc}" -x c -O2 -Wall -Wextra -Werror \
        "$work/probe.m" -lws2_32 -lonecore -o "$work/win64.exe"
else
    build_macos x86_64
fi

python3 - "$work" <<'PY'
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager

directory = pathlib.Path(sys.argv[1])
wine = os.environ.get("WINE")
length = 65536
offset = length if os.environ.get("VICTORY_PROBE_PLACEHOLDER") == "1" else 0
if offset and not wine:
    sys.exit("VICTORY_PROBE_PLACEHOLDER requires WINE")

@contextmanager
def launch(command, **kwargs):
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, **kwargs) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()

def native(arch, endpoint, memory):
    descriptors = (endpoint.fileno(), memory.fileno())
    return launch([directory / arch, *map(str, (*descriptors, offset))], pass_fds=descriptors)

@contextmanager
def connect_driver(memory):
    if wine:
        env = dict(os.environ, WINEDEBUG="-all")
        isolated = not env.get("WINEPREFIX")
        if isolated:
            env["WINEPREFIX"] = str(directory / "wineprefix")
        env.setdefault("WINEDLLOVERRIDES", "mscoree,mshtml=;winedbg.exe=d")
        wineserver = os.environ.get("WINESERVER", str(pathlib.Path(wine).with_name("wineserver")))
        if isolated and not os.access(wineserver, os.X_OK):
            sys.exit("Set WINESERVER to the matching Wine server executable")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(30)
            port = listener.getsockname()[1]
            windows_path = "Z:" + memory.name.replace("/", "\\")
            try:
                command = [wine, directory / "win64.exe", windows_path, str(port), str(offset)]
                with launch(command, env=env) as driver:
                    renderer, _ = listener.accept()
                    with renderer:
                        renderer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        yield driver, renderer
            finally:
                if isolated:
                    try:
                        subprocess.run([wineserver, "-w"], env=env, timeout=10, check=True)
                    except subprocess.TimeoutExpired:
                        subprocess.run([wineserver, "-k"], env=env, timeout=5, check=True)
                        subprocess.run([wineserver, "-w"], env=env, timeout=5, check=True)
    else:
        client, renderer = socket.socketpair()
        with client, renderer, native("x86_64", client, memory) as driver:
            client.close()
            yield driver, renderer

with tempfile.NamedTemporaryFile(dir=directory) as memory:
    os.ftruncate(memory.fileno(), length + offset)
    sentinel = bytes([0xA5]) * offset
    if offset:
        os.pwrite(memory.fileno(), sentinel, 0)
    with connect_driver(memory) as (driver, endpoint), native("arm64", endpoint, memory) as worker:
        endpoint.close()
        driver_status = driver.wait(timeout=30)
        worker_status = worker.wait(timeout=5)
        if driver_status or worker_status:
            sys.exit(f"Probe failed (x86_64={driver_status}, arm64={worker_status})")
    if offset:
        if os.pread(memory.fileno(), offset, 0) != sentinel:
            sys.exit("The GPU or client modified the adjacent sentinel range")
        print(f"PASS preserved the adjacent {offset}-byte range")
PY
