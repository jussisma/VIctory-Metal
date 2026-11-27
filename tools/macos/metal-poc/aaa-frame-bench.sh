#!/bin/sh
# ADR-016 120-frame Wine guest -> ARM64 Metal end-to-end benchmark.
#
# Pipeline under test (the synthetic AAA-shaped frame):
#
#   compute dispatch -> MRT/depth graphics (direct 3-vertex draw)
#   -> post-FX compute -> present
#
# What this script actually does, in order:
#
#   1. builds `victory-metal-host` (ARM64) and a small Win64 guest harness;
#   2. starts the daemon on an ephemeral loopback port with a private secret,
#      a shared fence page and `--aaa-frame` (synthetic MSL compute + MRT/depth
#      graphics pipelines built from shaders already in the repository);
#   3. starts the guest under Wine with `VICTORY_BACKEND=metal`,
#      `VICTORY_METAL_PORT` and `VICTORY_METAL_SECRET` (no
#      `VICTORY_METAL_ENDPOINT`, so the guest exercises the ADR-016
#      auto-connect endpoint), then waits for `FRAME_BENCH_DONE`;
#   4. registers two RGBA8 colour targets and one D32F depth target over IPC
#      kind 8, aliases colour-0 as surface 1 over kind 9, then submits the
#      MRT+depth pass and the post-FX/present pair per measured frame;
#   5. reports measured frame-time p50/p95/p99 over 120 frames after 20 warmup
#      frames, sustained FPS, the 0-IPC shared-fence read cost, the pixel
#      checksum, the AlreadyExecuted replay guard, the texture/surface/
#      graphics/present reply codes, the present readback checksum and the
#      D3DMetal module count.
#
# The measured frames are the real authenticated loopback + ARM64 Metal path.
# Every stage status is reported raw: a refused registration or pass prints one
# PARTIAL line with the first failing step and its real reply code. The script
# never fabricates or extrapolates a number: if the 120-frame loop stops early
# the CSV holds the frames that actually ran and the script exits non-zero.
#
# Usage: tools/macos/metal-poc/aaa-frame-bench.sh
#   WINE=/path/to/wine          override the Wine binary
#   KEEP=1                      keep the session directory for inspection
#   VICTORY_FRAME_MEASURED=120  measured frames (default 120)
#   VICTORY_FRAME_WARMUP=20     excluded warmup frames (default 20)
#
# One GPU task at a time: the script takes /tmp/victory-gpu.lock.
#
# Process hygiene (C7 / Wave 7A): the host daemon and the Wine guest are started
# in the background and registered with `process-hygiene.sh`. The EXIT trap
# reaps them with TERM, a bounded wait and KILL for survivors, removes only the
# lock this run created, and exits 13 if any started PID is still alive after
# cleanup. `VICTORY_KEEP_LOGS=1` (or `KEEP=1`) keeps the session files but still
# reaps every child.
#
# Exit codes: 0 PASS, 1/2/3 named failures, 13 orphan survivor after cleanup.
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SELF_DIR/process-hygiene.sh"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; put it first like the other macOS harnesses.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

LOCK="/tmp/victory-gpu.lock"

if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$ROOT/../VIctory-macOS/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
export WINE
MINGW_CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-aaa-bench.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
HOST_OUT="$WORK/host.stdout"
HOST_ERR="$WORK/host.stderr"
GUEST_LOG="$WORK/guest.log"
GUEST_EXE="$WORK/aaa-guest.exe"
TELEMETRY="$WORK/telemetry.json"
FRAMES_CSV="$SHARE/frames.v1.csv"
GUEST_REPORT="$SHARE/guest.v1.txt"
HOST_BIN="$ROOT/target/aarch64-apple-darwin/debug/victory_metal_host"
HOST_PID=""
GUEST_PID=""
PORT=""
GUEST_RC=0
HOST_RC=0
victory_hygiene_work "$WORK"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    # Reap tracked PIDs, then use the unique temp dir as an ownership proof for
    # any grandchild the tracked-PID pass could not reach.
    victory_hygiene_reap
    victory_hygiene_reap_marker "$WORK"
    echo "PROCESS_CHECK host=0 guest=0 survivors=${VICTORY_HYGIENE_SURVIVORS:-0}"
    victory_hygiene_lock_release
    victory_hygiene_remove_work
    victory_hygiene_finalize "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# --- Serialise exclusive GPU work ------------------------------------------
victory_hygiene_lock_acquire "$LOCK"

mkdir -m 700 "$SHARE"
python3 -c 'import secrets; print(secrets.token_hex(32))' >"$SECRET"
chmod 600 "$SECRET"
SECRET_HEX="$(cat "$SECRET")"

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

fail_setup() {
    echo "FAIL: $1" >&2
    exit 2
}

if [ -z "${WINE:-}" ] || [ ! -x "$WINE" ]; then
    fail_setup "no Wine executable found (set WINE=...)"
fi
command -v "$MINGW_CC" >/dev/null 2>&1 || fail_setup "missing $MINGW_CC"

# --- Build the ARM64 daemon and the Win64 guest ----------------------------
echo "Building the ARM64 victory-metal-host daemon..."
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin victory_metal_host \
    || fail_setup "the ARM64 host daemon did not build"
[ -x "$HOST_BIN" ] || fail_setup "missing $HOST_BIN"

# The guest harness is generated from this script so the benchmark stays a
# single, reviewable file. It speaks the frozen Wire v3 envelope, creates the
# guest-visible allocation and shared fence page, and measures every frame.
cat >"$WORK/aaa_guest.c" <<'CEOF'
#define _WIN32_WINNT 0x0A00
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SECRET_LEN 32
#define ALLOC_ID 1ULL
#define ALLOC_GEN 1u
#define ALLOC_BYTES (64u * 1024u)
#define THREADS 16384u
#define TPG 64u
#define POSTFX_THREADS 64u
#define MAX_FRAMES 512u
#define TARGET_W 1280u
#define TARGET_H 720u
#define COLOR0_ID 4096ULL
#define COLOR1_ID 4097ULL
#define DEPTH_ID 4098ULL
#define SURFACE_ID 1ULL
#define COLOR_GEN 1u

enum {
    KIND_AUTH = 1,
    KIND_REGISTER = 2,
    KIND_SUBMISSION = 5,
    KIND_FENCE_PAGE = 6,
    KIND_SUBMISSION_ASYNC = 7,
    KIND_TEXTURE_REG = 8,
    KIND_SURFACE_REG = 9,
    KIND_ACK = 128
};

enum {
    STATUS_OK = 0,
    STATUS_ALREADY_EXECUTED = 3
};

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
} Buf;

static char g_share[1024];
static char g_report_path[2048];
static char g_csv_path[2048];
static char g_alloc_path[2048];
static char g_fence_path[2048];
static FILE *g_csv;
static unsigned g_measured;
static unsigned g_frames_ok;
static unsigned g_sequence;
static const char *g_reason = "completed";
static int g_ok = 0;
static double g_fence_read_ns;
static unsigned g_replay_status = 0xffff;
static int g_replay_unchanged = 0;
static unsigned g_texture_status[3] = {0xffff, 0xffff, 0xffff};
static unsigned g_surface_status = 0xffff;
static unsigned g_graphics_status = 0xffff;
static unsigned g_present_status = 0xffff;
static double g_enc_culling_us;
static double g_enc_graphics_us;
static double g_enc_postfx_us;
static double g_enc_present_us;
static uint64_t g_expected_present_checksum;
static unsigned g_guest_modules;
static int g_d3dmetal_modules = 1;
static uint64_t g_checksum;
static uint32_t g_first_word;
static uint32_t g_expected_first;
static unsigned g_executions;

static void ensure(Buf *buffer, size_t extra) {
    if (buffer->len + extra <= buffer->cap) {
        return;
    }
    size_t cap = buffer->cap ? buffer->cap : 256;
    while (cap < buffer->len + extra) {
        cap *= 2;
    }
    unsigned char *grown = (unsigned char *)realloc(buffer->data, cap);
    if (grown == NULL) {
        fprintf(stderr, "aaa-guest: out of memory\n");
        ExitProcess(3);
    }
    buffer->data = grown;
    buffer->cap = cap;
}

static void put(Buf *buffer, const void *bytes, size_t len) {
    ensure(buffer, len);
    memcpy(buffer->data + buffer->len, bytes, len);
    buffer->len += len;
}

static void put_u16(Buf *buffer, uint16_t value) { put(buffer, &value, 2); }
static void put_u32(Buf *buffer, uint32_t value) { put(buffer, &value, 4); }
static void put_u64(Buf *buffer, uint64_t value) { put(buffer, &value, 8); }
static void put_f32(Buf *buffer, float value) { put(buffer, &value, 4); }
static void put_f64(Buf *buffer, double value) { put(buffer, &value, 8); }
static void reset(Buf *buffer) { buffer->len = 0; }
static void release(Buf *buffer) {
    free(buffer->data);
    buffer->data = NULL;
    buffer->len = 0;
    buffer->cap = 0;
}

static double now_ms(void) {
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}

static void env_copy(const char *name, char *out, size_t len, const char *fallback) {
    DWORD read = GetEnvironmentVariableA(name, out, (DWORD)len);
    if (read == 0 || read >= len) {
        snprintf(out, len, "%s", fallback == NULL ? "" : fallback);
    }
}

static unsigned env_uint(const char *name, unsigned fallback) {
    char text[64];
    env_copy(name, text, sizeof(text), NULL);
    if (text[0] == '\0') {
        return fallback;
    }
    return (unsigned)strtoul(text, NULL, 10);
}

static void build_path(char *out, size_t len, const char *directory, const char *leaf) {
    int written = snprintf(out, len, "%s\\%s", directory, leaf);
    if (written < 0 || (size_t)written >= len) {
        fprintf(stderr, "aaa-guest: path is too long\n");
        ExitProcess(3);
    }
}

static void send_all(SOCKET client, const void *bytes, size_t len) {
    const char *cursor = (const char *)bytes;
    while (len > 0) {
        int chunk = (int)(len > (1u << 20) ? (1u << 20) : len);
        int sent = send(client, cursor, chunk, 0);
        if (sent <= 0) {
            g_reason = "socket write failed";
            return;
        }
        cursor += sent;
        len -= (size_t)sent;
    }
}

static void send_frame(SOCKET client, uint8_t kind, const void *payload, size_t len) {
    unsigned char header[5];
    header[0] = kind;
    header[1] = (unsigned char)(len & 0xff);
    header[2] = (unsigned char)((len >> 8) & 0xff);
    header[3] = (unsigned char)((len >> 16) & 0xff);
    header[4] = (unsigned char)((len >> 24) & 0xff);
    send_all(client, header, sizeof(header));
    send_all(client, payload, len);
}

static int recv_all(SOCKET client, void *bytes, size_t len) {
    char *cursor = (char *)bytes;
    while (len > 0) {
        int chunk = (int)(len > (1u << 20) ? (1u << 20) : len);
        int read = recv(client, cursor, chunk, 0);
        if (read <= 0) {
            g_reason = "socket read failed or the host closed the session";
            return 0;
        }
        cursor += read;
        len -= (size_t)read;
    }
    return 1;
}

static int recv_frame(SOCKET client, uint8_t *kind, unsigned char **payload, size_t *len) {
    unsigned char header[5];
    if (!recv_all(client, header, sizeof(header))) {
        return 0;
    }
    *kind = header[0];
    *len = (size_t)header[1] | ((size_t)header[2] << 8) | ((size_t)header[3] << 16) |
           ((size_t)header[4] << 24);
    *payload = (unsigned char *)malloc(*len ? *len : 1);
    if (*payload == NULL) {
        g_reason = "out of memory";
        return 0;
    }
    if (!recv_all(client, *payload, *len)) {
        free(*payload);
        *payload = NULL;
        return 0;
    }
    return 1;
}

/* Registers one host texture through kind 8 and returns the raw ack status. */
static int register_texture(SOCKET client, uint64_t id, uint32_t generation, uint8_t dimension,
                            uint8_t format, uint8_t usage, uint8_t sample_count, uint32_t width,
                            uint32_t height, uint32_t depth, uint16_t mip_levels,
                            uint16_t array_layers, unsigned *status) {
    unsigned char payload[32];
    uint8_t kind = 0;
    unsigned char *reply = NULL;
    size_t len = 0;
    memcpy(payload, &id, 8);
    memcpy(payload + 8, &generation, 4);
    payload[12] = dimension;
    payload[13] = format;
    payload[14] = usage;
    payload[15] = sample_count;
    memcpy(payload + 16, &width, 4);
    memcpy(payload + 20, &height, 4);
    memcpy(payload + 24, &depth, 4);
    memcpy(payload + 28, &mip_levels, 2);
    memcpy(payload + 30, &array_layers, 2);
    send_frame(client, KIND_TEXTURE_REG, payload, sizeof(payload));
    if (!recv_frame(client, &kind, &reply, &len)) {
        return 0;
    }
    if (kind != KIND_ACK || len != 1) {
        g_reason = "unexpected texture registration reply";
        free(reply);
        return 0;
    }
    *status = reply[0];
    free(reply);
    return 1;
}

/* Registers one present surface through kind 9 and returns the raw ack status. */
static int register_surface(SOCKET client, uint64_t surface_id, uint64_t texture_id,
                            unsigned *status) {
    unsigned char payload[16];
    uint8_t kind = 0;
    unsigned char *reply = NULL;
    size_t len = 0;
    memcpy(payload, &surface_id, 8);
    memcpy(payload + 8, &texture_id, 8);
    send_frame(client, KIND_SURFACE_REG, payload, sizeof(payload));
    if (!recv_frame(client, &kind, &reply, &len)) {
        return 0;
    }
    if (kind != KIND_ACK || len != 1) {
        g_reason = "unexpected surface registration reply";
        free(reply);
        return 0;
    }
    *status = reply[0];
    free(reply);
    return 1;
}

static uint64_t fence_value(void *page);

static int wait_fence(void *page, uint32_t sequence) {
    uint64_t deadline = GetTickCount64() + 30000;
    while (fence_value(page) < sequence) {
        if (GetTickCount64() > deadline) {
            g_reason = "timed out waiting for the shared completion fence";
            return 0;
        }
        SwitchToThread();
    }
    return 1;
}

/* FNV-1a 64 of the expected uniform colour-0 target (RGBA8, row-major). */
static uint64_t expected_present_checksum(void) {
    const unsigned char pixel[4] = {255, 128, 0, 255};
    size_t pixels = (size_t)TARGET_W * (size_t)TARGET_H;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < pixels; index++) {
        for (int channel = 0; channel < 4; channel++) {
            hash ^= pixel[channel];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static void build_dispatch(Buf *commands, uint64_t pipeline) {
    put_u16(commands, 1);
    put_u16(commands, 0);
    put_u64(commands, pipeline);
    put_u32(commands, THREADS);
    put_u32(commands, TPG);
    put_u16(commands, 1);
    put_u16(commands, 0);
    put_u64(commands, ALLOC_ID);
    put_u32(commands, ALLOC_GEN);
    put_u32(commands, 0);
    put_u64(commands, 0);
    put_u64(commands, ALLOC_BYTES);
}

/* Colour attachment: handle, generation, mip/slice, load/store, clear colour. */
static void build_color_attachment(Buf *commands, uint64_t id, uint32_t generation,
                                   uint16_t load, uint16_t store, double red, double green,
                                   double blue, double alpha) {
    put_u64(commands, id);
    put_u32(commands, generation);
    put_u16(commands, 0);
    put_u16(commands, 0);
    put_u16(commands, load);
    put_u16(commands, store);
    put_u32(commands, 0);
    put_f64(commands, red);
    put_f64(commands, green);
    put_f64(commands, blue);
    put_f64(commands, alpha);
}

/* MRT+depth graphics pass naming the two registered colour targets and the
   registered depth target. The host's built-in graphics pipeline was compiled
   for exactly this attachment set. */
static void build_graphics(Buf *commands, uint64_t pipeline) {
    /* BeginRenderPass: 2 colours + 1 depth, all Clear/Store. */
    put_u16(commands, 16);
    put_u16(commands, 0);
    put_u16(commands, 2);
    put_u16(commands, 1);
    put_u32(commands, 0);
    build_color_attachment(commands, COLOR0_ID, COLOR_GEN, 1, 0, 0.0, 0.0, 0.0, 1.0);
    build_color_attachment(commands, COLOR1_ID, COLOR_GEN, 1, 0, 0.0, 0.0, 0.0, 1.0);
    /* Depth attachment: D32F, clear to 1.0, store; stencil DontCare. */
    put_u64(commands, DEPTH_ID);
    put_u32(commands, COLOR_GEN);
    put_u16(commands, 0);
    put_u16(commands, 0);
    put_u16(commands, 1);
    put_u16(commands, 0);
    put_f32(commands, 1.0f);
    put_u16(commands, 2);
    put_u16(commands, 1);
    put_u32(commands, 0);
    put_u32(commands, 0);
    /* SetGraphicsPipeline(host graphics id). */
    put_u16(commands, 20);
    put_u16(commands, 0);
    put_u64(commands, pipeline);
    /* SetViewport(0, 0, TARGET_W, TARGET_H, 0, 1). */
    put_u16(commands, 18);
    put_u16(commands, 0);
    put_f64(commands, 0.0);
    put_f64(commands, 0.0);
    put_f64(commands, (double)TARGET_W);
    put_f64(commands, (double)TARGET_H);
    put_f64(commands, 0.0);
    put_f64(commands, 1.0);
    /* SetScissor(full target). */
    put_u16(commands, 19);
    put_u16(commands, 0);
    put_u32(commands, 0);
    put_u32(commands, 0);
    put_u32(commands, TARGET_W);
    put_u32(commands, TARGET_H);
    /* SetPrimitiveTopology(TriangleList). */
    put_u16(commands, 23);
    put_u16(commands, 0);
    put_u16(commands, 3);
    put_u16(commands, 0);
    /* SetRootConstants(slot 0, 32 bytes): the two exact RGBA tints. */
    put_u16(commands, 8);
    put_u16(commands, 0);
    put_u16(commands, 0);
    put_u16(commands, 32);
    {
        float tints[8] = {1.0f, 128.0f / 255.0f, 0.0f, 1.0f,
                          0.0f, 128.0f / 255.0f, 1.0f, 1.0f};
        put(commands, tints, sizeof(tints));
    }
    /* Draw(3 vertices, 1 instance). */
    put_u16(commands, 4);
    put_u16(commands, 0);
    put_u32(commands, 3);
    put_u32(commands, 1);
    put_u32(commands, 0);
    put_u32(commands, 0);
    /* EndRenderPass. */
    put_u16(commands, 17);
    put_u16(commands, 0);
}

/* Post-FX compute dispatch: the built-in mul-add kernel over the first
   POSTFX_THREADS words of the guest allocation. */
static void build_postfx(Buf *commands, uint64_t pipeline) {
    put_u16(commands, 22);
    put_u16(commands, 0);
    put_u64(commands, pipeline);
    put_u16(commands, 1);
    put_u16(commands, 0);
    put_u64(commands, pipeline);
    put_u32(commands, POSTFX_THREADS);
    put_u32(commands, TPG);
    put_u16(commands, 1);
    put_u16(commands, 0);
    put_u64(commands, ALLOC_ID);
    put_u32(commands, ALLOC_GEN);
    put_u32(commands, 0);
    put_u64(commands, 0);
    put_u64(commands, ALLOC_BYTES);
}

static void build_present(Buf *commands) {
    put_u16(commands, 31);
    put_u16(commands, 0);
    put_u64(commands, SURFACE_ID);
    put_u32(commands, 0);
    put_u32(commands, 0);
}

static void encode_submission(Buf *frame, uint32_t sequence, uint64_t list_id,
                              uint32_t list_generation, const Buf *commands, uint16_t count) {
    reset(frame);
    put(frame, "VMWF", 4);
    put_u16(frame, 3);
    put_u16(frame, 1);
    put_u32(frame, sequence);
    put_u16(frame, count);
    put_u16(frame, 0);
    put_u32(frame, (uint32_t)(12 + commands->len));
    put_u64(frame, list_id);
    put_u32(frame, list_generation);
    put(frame, commands->data, commands->len);
}

static void send_envelope(SOCKET client, uint8_t kind, uint32_t sequence, const Buf *frame) {
    Buf payload = {0};
    put(&payload, "V3RQ", 4);
    put_u16(&payload, 3);
    put_u32(&payload, sequence);
    put_u16(&payload, 0);
    put_u32(&payload, (uint32_t)frame->len);
    put(&payload, frame->data, frame->len);
    send_frame(client, kind, payload.data, payload.len);
    release(&payload);
}

static int read_reply(SOCKET client, uint16_t *status) {
    uint8_t kind = 0;
    unsigned char *payload = NULL;
    size_t len = 0;
    if (!recv_frame(client, &kind, &payload, &len)) {
        return 0;
    }
    /* The host's kind-5 payload is a complete Wire v2 reply frame: a 20-byte
       header (magic, version, kind=2, sequence, count, reserved, length)
       followed by the 4-byte status/reserved pair. */
    if (kind != KIND_SUBMISSION || len != 24 || memcmp(payload, "VMWF", 4) != 0 ||
        payload[6] != 2 || payload[7] != 0) {
        fprintf(stderr, "aaa-guest: reply kind=%u len=%zu head=%02x%02x%02x%02x %02x%02x %02x%02x\n",
                (unsigned)kind, len,
                len > 0 ? payload[0] : 0, len > 1 ? payload[1] : 0,
                len > 2 ? payload[2] : 0, len > 3 ? payload[3] : 0,
                len > 4 ? payload[4] : 0, len > 5 ? payload[5] : 0,
                len > 6 ? payload[6] : 0, len > 7 ? payload[7] : 0);
        g_reason = "unexpected version 3 reply frame";
        free(payload);
        return 0;
    }
    *status = (uint16_t)(payload[20] | (payload[21] << 8));
    free(payload);
    return 1;
}

static uint64_t file_checksum(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    unsigned char chunk[4096];
    size_t read;
    while ((read = fread(chunk, 1, sizeof(chunk), file)) > 0) {
        for (size_t index = 0; index < read; index++) {
            hash ^= chunk[index];
            hash *= 1099511628211ULL;
        }
    }
    fclose(file);
    return hash;
}

static int read_first_word(const char *path, uint32_t *word) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    unsigned char bytes[4];
    int ok = fread(bytes, 1, 4, file) == 4;
    fclose(file);
    if (ok) {
        *word = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
                ((uint32_t)bytes[3] << 24);
    }
    return ok;
}

static uint64_t fence_value(void *page) {
    return __atomic_load_n((volatile uint64_t *)page, __ATOMIC_ACQUIRE);
}

static int d3dmetal_loaded(unsigned *modules) {
    HMODULE handles[2048];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), handles, (DWORD)sizeof(handles), &needed)) {
        return 1;
    }
    unsigned count = needed / (unsigned)sizeof(HMODULE);
    if (count > 2048) {
        count = 2048;
    }
    *modules = count;
    int found = 0;
    for (unsigned index = 0; index < count; index++) {
        char name[MAX_PATH];
        DWORD len = GetModuleFileNameA(handles[index], name, (DWORD)sizeof(name));
        for (DWORD at = 0; at + 8 <= len; at++) {
            if (_strnicmp(name + at, "d3dmetal", 8) == 0) {
                found = 1;
            }
        }
    }
    if (GetModuleHandleA("d3dmetal.dll") != NULL) {
        found = 1;
    }
    return found;
}

static void write_report(void) {
    if (g_report_path[0] == '\0') {
        return;
    }
    FILE *report = fopen(g_report_path, "w");
    if (report == NULL) {
        return;
    }
    fprintf(report, "reason=%s\n", g_reason);
    fprintf(report, "frames_ok=%u\n", g_frames_ok);
    fprintf(report, "measured=%u\n", g_measured);
    fprintf(report, "executions=%u\n", g_executions);
    fprintf(report, "checksum=%016llx\n", (unsigned long long)g_checksum);
    fprintf(report, "first_word=%u\n", g_first_word);
    fprintf(report, "expected_first=%u\n", g_expected_first);
    fprintf(report, "replay_status=%u\n", g_replay_status);
    fprintf(report, "replay_unchanged=%d\n", g_replay_unchanged);
    fprintf(report, "texture0_status=%u\n", g_texture_status[0]);
    fprintf(report, "texture1_status=%u\n", g_texture_status[1]);
    fprintf(report, "depth_status=%u\n", g_texture_status[2]);
    fprintf(report, "surface_status=%u\n", g_surface_status);
    fprintf(report, "graphics_status=%u\n", g_graphics_status);
    fprintf(report, "present_status=%u\n", g_present_status);
    fprintf(report, "expected_present_checksum=%016llx\n",
            (unsigned long long)g_expected_present_checksum);
    fprintf(report, "encode_culling_us=%.3f\n", g_enc_culling_us);
    fprintf(report, "encode_graphics_us=%.3f\n", g_enc_graphics_us);
    fprintf(report, "encode_postfx_us=%.3f\n", g_enc_postfx_us);
    fprintf(report, "encode_present_us=%.3f\n", g_enc_present_us);
    fprintf(report, "fence_read_ns=%.2f\n", g_fence_read_ns);
    fprintf(report, "guest_modules=%u\n", g_guest_modules);
    fprintf(report, "d3dmetal_modules=%d\n", g_d3dmetal_modules);
    fclose(report);
}

static int finish(void) {
    if (g_csv != NULL) {
        fflush(g_csv);
        fclose(g_csv);
        g_csv = NULL;
    }
    write_report();
    if (g_ok) {
        printf("FRAME_BENCH_DONE ok=1 frames=%u measured=%u\n", g_frames_ok, g_measured);
        fflush(stdout);
        return 0;
    }
    printf("FRAME_BENCH_ERROR %s\n", g_reason);
    fflush(stdout);
    return 1;
}

int main(void) {
    WSADATA winsock;
    if (WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
        g_reason = "WSAStartup failed";
        return finish();
    }

    env_copy("VICTORY_METAL_SHARE_DIR", g_share, sizeof(g_share), NULL);
    if (g_share[0] == '\0') {
        g_reason = "VICTORY_METAL_SHARE_DIR is not set";
        return finish();
    }
    build_path(g_report_path, sizeof(g_report_path), g_share, "guest.v1.txt");
    build_path(g_csv_path, sizeof(g_csv_path), g_share, "frames.v1.csv");
    build_path(g_alloc_path, sizeof(g_alloc_path), g_share, "0000000000000001.bin");
    build_path(g_fence_path, sizeof(g_fence_path), g_share, "gpu-fence.v1.page");

    unsigned warmup = env_uint("VICTORY_FRAME_WARMUP", 20);
    unsigned measured = env_uint("VICTORY_FRAME_MEASURED", 120);
    if (warmup + measured > MAX_FRAMES || measured == 0) {
        g_reason = "invalid frame counts";
        return finish();
    }

    char compute_text[64];
    env_copy("VICTORY_FRAME_COMPUTE_ID", compute_text, sizeof(compute_text), NULL);
    uint64_t pipeline = _strtoui64(compute_text, NULL, 10);
    if (pipeline == 0) {
        g_reason = "VICTORY_FRAME_COMPUTE_ID is not set";
        return finish();
    }

    char graphics_text[64];
    env_copy("VICTORY_FRAME_GRAPHICS_ID", graphics_text, sizeof(graphics_text), NULL);
    uint64_t graphics_pipeline = _strtoui64(graphics_text, NULL, 10);
    if (graphics_pipeline == 0) {
        g_reason = "VICTORY_FRAME_GRAPHICS_ID is not set";
        return finish();
    }

    char secret_text[128];
    env_copy("VICTORY_METAL_SECRET", secret_text, sizeof(secret_text), NULL);
    if (strlen(secret_text) != SECRET_LEN * 2) {
        g_reason = "VICTORY_METAL_SECRET must be 64 hex characters";
        return finish();
    }
    unsigned char secret[SECRET_LEN];
    for (unsigned index = 0; index < SECRET_LEN; index++) {
        char pair[3] = {secret_text[index * 2], secret_text[index * 2 + 1], 0};
        secret[index] = (unsigned char)strtoul(pair, NULL, 16);
    }

    char endpoint[128];
    env_copy("VICTORY_METAL_ENDPOINT", endpoint, sizeof(endpoint), NULL);
    int auto_connect = endpoint[0] == '\0';
    if (auto_connect) {
        char backend[32];
        env_copy("VICTORY_BACKEND", backend, sizeof(backend), NULL);
        if (_stricmp(backend, "metal") != 0) {
            g_reason = "VICTORY_METAL_ENDPOINT is unset and VICTORY_BACKEND is not metal";
            return finish();
        }
        char port_text[32];
        env_copy("VICTORY_METAL_PORT", port_text, sizeof(port_text), "47654");
        int written = snprintf(endpoint, sizeof(endpoint), "127.0.0.1:%s", port_text);
        if (written < 0 || (size_t)written >= sizeof(endpoint)) {
            g_reason = "auto-connect endpoint is too long";
            return finish();
        }
    }

    char host[64];
    const char *colon = strrchr(endpoint, ':');
    if (colon == NULL || (size_t)(colon - endpoint) >= sizeof(host)) {
        g_reason = "endpoint is not host:port";
        return finish();
    }
    memcpy(host, endpoint, (size_t)(colon - endpoint));
    host[colon - endpoint] = 0;
    unsigned port = (unsigned)strtoul(colon + 1, NULL, 10);
    if (port == 0) {
        g_reason = "endpoint port is not usable";
        return finish();
    }

    FILE *allocation = fopen(g_alloc_path, "wb");
    if (allocation == NULL) {
        g_reason = "cannot create the guest allocation file";
        return finish();
    }
    for (uint32_t index = 0; index < THREADS; index++) {
        uint32_t value = index + 1;
        fwrite(&value, sizeof(value), 1, allocation);
    }
    for (unsigned index = THREADS; index < ALLOC_BYTES / 4; index++) {
        uint32_t value = 0;
        fwrite(&value, sizeof(value), 1, allocation);
    }
    fclose(allocation);

    HANDLE fence_file = CreateFileA(g_fence_path, GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, NULL);
    if (fence_file == INVALID_HANDLE_VALUE) {
        g_reason = "cannot create the shared fence page file";
        return finish();
    }
    LARGE_INTEGER page_size;
    page_size.QuadPart = 4096;
    if (!SetFilePointerEx(fence_file, page_size, NULL, FILE_BEGIN) || !SetEndOfFile(fence_file)) {
        g_reason = "cannot size the shared fence page";
        CloseHandle(fence_file);
        return finish();
    }
    HANDLE mapping = CreateFileMappingA(fence_file, NULL, PAGE_READWRITE, 0, 4096, NULL);
    if (mapping == NULL) {
        g_reason = "cannot map the shared fence page";
        CloseHandle(fence_file);
        return finish();
    }
    void *page = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 4096);
    if (page == NULL) {
        g_reason = "cannot view the shared fence page";
        CloseHandle(mapping);
        CloseHandle(fence_file);
        return finish();
    }

    SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (client == INVALID_SOCKET) {
        g_reason = "cannot create the guest socket";
        return finish();
    }
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        g_reason = "endpoint host is not an IPv4 address";
        closesocket(client);
        return finish();
    }
    if (connect(client, (struct sockaddr *)&address, sizeof(address)) != 0) {
        g_reason = "auto-connect to the victory-metal-host endpoint failed";
        closesocket(client);
        return finish();
    }
    DWORD timeout = 60000;
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof(timeout));
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeout, sizeof(timeout));

    send_frame(client, KIND_AUTH, secret, SECRET_LEN);
    {
        uint8_t kind = 0;
        unsigned char *payload = NULL;
        size_t len = 0;
        if (!recv_frame(client, &kind, &payload, &len)) {
            closesocket(client);
            return finish();
        }
        int accepted = kind == KIND_ACK && len == 1 && payload[0] == 0;
        free(payload);
        if (!accepted) {
            g_reason = "the host refused authentication";
            closesocket(client);
            return finish();
        }
    }

    {
        unsigned char registration[16];
        uint64_t id = ALLOC_ID;
        uint64_t bytes = ALLOC_BYTES;
        memcpy(registration, &id, 8);
        memcpy(registration + 8, &bytes, 8);
        send_frame(client, KIND_REGISTER, registration, sizeof(registration));
        uint8_t kind = 0;
        unsigned char *payload = NULL;
        size_t len = 0;
        if (!recv_frame(client, &kind, &payload, &len)) {
            closesocket(client);
            return finish();
        }
        int accepted = kind == KIND_ACK && len == 1 && payload[0] == 0;
        free(payload);
        if (!accepted) {
            g_reason = "the host refused the allocation registration";
            closesocket(client);
            return finish();
        }
    }

    {
        uint64_t bytes = 4096;
        send_frame(client, KIND_FENCE_PAGE, &bytes, sizeof(bytes));
        uint8_t kind = 0;
        unsigned char *payload = NULL;
        size_t len = 0;
        if (!recv_frame(client, &kind, &payload, &len)) {
            closesocket(client);
            return finish();
        }
        int accepted = kind == KIND_ACK && len == 1 && payload[0] == 0;
        free(payload);
        if (!accepted) {
            g_reason = "the host refused the shared fence page";
            closesocket(client);
            return finish();
        }
    }

    /* Register the two RGBA8 colour targets and the D32F depth target, then
       alias colour-0 as the present surface. Every status is reported raw so a
       refusal is attributed to the exact registration. */
    {
        static const struct {
            uint64_t id;
            uint8_t format;
            uint8_t usage;
        } targets[3] = {
            {COLOR0_ID, 0, 1},
            {COLOR1_ID, 0, 1},
            {DEPTH_ID, 7, 2},
        };
        for (unsigned index = 0; index < 3; index++) {
            unsigned status = 0xffff;
            if (!register_texture(client, targets[index].id, COLOR_GEN, 1, targets[index].format,
                                  targets[index].usage, 1, TARGET_W, TARGET_H, 1, 1, 1, &status)) {
                closesocket(client);
                return finish();
            }
            g_texture_status[index] = status;
            if (status != 0) {
                g_reason = "the host refused a texture registration";
                closesocket(client);
                return finish();
            }
        }
        unsigned status = 0xffff;
        if (!register_surface(client, SURFACE_ID, COLOR0_ID, &status)) {
            closesocket(client);
            return finish();
        }
        g_surface_status = status;
        if (status != 0) {
            g_reason = "the host refused the present-surface registration";
            closesocket(client);
            return finish();
        }
    }

    g_csv = fopen(g_csv_path, "w");
    if (g_csv == NULL) {
        g_reason = "cannot create the frame-time CSV";
        closesocket(client);
        return finish();
    }

    unsigned total = warmup + measured;
    Buf commands = {0};
    Buf frame = {0};
    for (unsigned index = 0; index < total; index++) {
        double started = now_ms();
        double stage_start;
        uint32_t sequence = ++g_sequence;

        /* Stage 1: compute culling (kind 7, fire-and-forget). */
        reset(&commands);
        stage_start = now_ms();
        build_dispatch(&commands, pipeline);
        encode_submission(&frame, sequence, 0, 0, &commands, 1);
        g_enc_culling_us += (now_ms() - stage_start) * 1000.0;
        send_envelope(client, KIND_SUBMISSION_ASYNC, sequence, &frame);
        if (!wait_fence(page, sequence)) {
            release(&commands);
            release(&frame);
            closesocket(client);
            return finish();
        }
        g_executions++;

        /* Stage 2: MRT+depth graphics pass (kind 5, explicit reply). */
        sequence = ++g_sequence;
        reset(&commands);
        stage_start = now_ms();
        build_graphics(&commands, graphics_pipeline);
        encode_submission(&frame, sequence, 0, 0, &commands, 8);
        g_enc_graphics_us += (now_ms() - stage_start) * 1000.0;
        send_envelope(client, KIND_SUBMISSION, sequence, &frame);
        {
            uint16_t status = 0xffff;
            if (!read_reply(client, &status)) {
                release(&commands);
                release(&frame);
                closesocket(client);
                return finish();
            }
            g_graphics_status = status;
            if (status != STATUS_OK) {
                g_reason = "the AAA graphics pass was refused";
                break;
            }
            /* ADR-017: the kind-5 reply means "committed", not "complete".
               Completion is observed only through the shared fence. */
            if (!wait_fence(page, sequence)) {
                break;
            }
        }

        /* Stage 3: post-FX compute dispatch plus present (kind 5). */
        sequence = ++g_sequence;
        reset(&commands);
        stage_start = now_ms();
        build_postfx(&commands, pipeline);
        g_enc_postfx_us += (now_ms() - stage_start) * 1000.0;
        stage_start = now_ms();
        build_present(&commands);
        g_enc_present_us += (now_ms() - stage_start) * 1000.0;
        encode_submission(&frame, sequence, 0, 0, &commands, 3);
        send_envelope(client, KIND_SUBMISSION, sequence, &frame);
        {
            uint16_t status = 0xffff;
            if (!read_reply(client, &status)) {
                release(&commands);
                release(&frame);
                closesocket(client);
                return finish();
            }
            g_present_status = status;
            if (status != STATUS_OK) {
                g_reason = "the post-FX/present submission was refused";
                break;
            }
            if (!wait_fence(page, sequence)) {
                break;
            }
        }
        g_executions++;

        g_frames_ok++;
        if (index >= warmup) {
            fprintf(g_csv, "%.6f\n", now_ms() - started);
            fflush(g_csv);
            g_measured++;
        }
    }

    {
        uint64_t sink = 0;
        double started = now_ms();
        for (int index = 0; index < 100000; index++) {
            sink += fence_value(page);
        }
        g_fence_read_ns = (now_ms() - started) * 1000000.0 / 100000.0;
        if (sink == 0xdeadbeefULL) {
            fprintf(stderr, "unreachable %llu\n", (unsigned long long)sink);
        }
    }

    {
        uint32_t first_sequence = ++g_sequence;
        reset(&commands);
        build_dispatch(&commands, pipeline);
        encode_submission(&frame, first_sequence, 9001, 1, &commands, 1);
        send_envelope(client, KIND_SUBMISSION, first_sequence, &frame);
        uint16_t status = 0xffff;
        if (!read_reply(client, &status)) {
            release(&commands);
            release(&frame);
            closesocket(client);
            return finish();
        }
        if (status == STATUS_OK) {
            g_executions++;
        }
        g_checksum = file_checksum(g_alloc_path);

        uint32_t replay_sequence = ++g_sequence;
        encode_submission(&frame, replay_sequence, 9001, 1, &commands, 1);
        send_envelope(client, KIND_SUBMISSION, replay_sequence, &frame);
        if (!read_reply(client, &status)) {
            release(&commands);
            release(&frame);
            closesocket(client);
            return finish();
        }
        g_replay_status = status;
        uint64_t after = file_checksum(g_alloc_path);
        g_replay_unchanged = after == g_checksum;
    }

    release(&commands);
    release(&frame);

    g_checksum = file_checksum(g_alloc_path);
    if (!read_first_word(g_alloc_path, &g_first_word)) {
        g_reason = "cannot read the guest-visible allocation";
        closesocket(client);
        return finish();
    }
    g_expected_first = 1;
    for (unsigned index = 0; index < g_executions; index++) {
        g_expected_first = g_expected_first * 3u + 7u;
    }
    g_expected_present_checksum = expected_present_checksum();

    g_d3dmetal_modules = d3dmetal_loaded(&g_guest_modules);

    g_ok = 1;
    if (g_texture_status[0] != 0 || g_texture_status[1] != 0 || g_texture_status[2] != 0) {
        g_ok = 0;
        g_reason = "a texture registration was refused";
    } else if (g_surface_status != 0) {
        g_ok = 0;
        g_reason = "the present-surface registration was refused";
    } else if (g_graphics_status != STATUS_OK) {
        g_ok = 0;
        g_reason = "the AAA graphics pass was refused";
    } else if (g_present_status != STATUS_OK) {
        g_ok = 0;
        g_reason = "the post-FX/present submission was refused";
    } else if (g_frames_ok != total) {
        g_ok = 0;
        g_reason = "the benchmark did not complete every frame";
    } else if (g_replay_status != STATUS_ALREADY_EXECUTED || !g_replay_unchanged) {
        g_ok = 0;
        g_reason = "the AlreadyExecuted replay guard did not hold";
    } else if (g_d3dmetal_modules) {
        g_ok = 0;
        g_reason = "the guest process loaded a D3DMetal module";
    } else if (g_first_word != g_expected_first) {
        g_ok = 0;
        g_reason = "the guest-visible compute result did not match";
    }
    if (g_ok) {
        g_reason = "completed";
    }
    closesocket(client);
    return finish();
}
CEOF

echo "Compiling the Win64 guest harness..."
victory_hygiene_run "$MINGW_CC" -x c -O2 -Wall -Wextra -Werror -o "$GUEST_EXE" "$WORK/aaa_guest.c" -lws2_32 -lpsapi \
    || fail_setup "the Win64 guest harness did not compile"

# --- Start the ARM64 daemon ------------------------------------------------
echo "Starting the ARM64 victory-metal-host daemon..."
: >"$HOST_OUT"
: >"$HOST_ERR"
"$HOST_BIN" \
    --secret-file "$SECRET" \
    --port 0 \
    --once \
    --aaa-frame \
    --share-dir "$SHARE" \
    --telemetry-json "$TELEMETRY" \
    >"$HOST_OUT" 2>"$HOST_ERR" &
HOST_PID=$!
victory_hygiene_track "$HOST_PID"

ready=0
for _ in $(seq 1 600); do
    if grep -q '^victory-metal-host: ready port=' "$HOST_OUT" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$HOST_PID" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "FAIL: the host daemon did not become ready" >&2
    cat "$HOST_OUT" "$HOST_ERR" >&2 || true
    exit 2
fi
PORT="$(sed -n 's/^victory-metal-host: ready port=\([0-9][0-9]*\).*$/\1/p' "$HOST_OUT" | tail -1)"
[ -n "$PORT" ] || fail_setup "the host did not publish its port"

compute_id=""
for _ in $(seq 1 200); do
    compute_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.compute.id" 2>/dev/null | tail -1)"
    [ -n "$compute_id" ] && break
    sleep 0.05
done
[ -n "$compute_id" ] || fail_setup "the frame gate did not publish a compute pipeline id"
graphics_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.graphics.id" 2>/dev/null | tail -1)"

# --- Run the Win64 guest under Wine ----------------------------------------
export WINEPREFIX="$WORK/wineprefix"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export VICTORY_BACKEND=metal
export VICTORY_METAL_PORT="$PORT"
export VICTORY_METAL_SECRET="$SECRET_HEX"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_FRAME_COMPUTE_ID="$compute_id"
export VICTORY_FRAME_GRAPHICS_ID="${graphics_id:-0}"
export VICTORY_FRAME_WARMUP="${VICTORY_FRAME_WARMUP:-20}"
export VICTORY_FRAME_MEASURED="${VICTORY_FRAME_MEASURED:-120}"
unset VICTORY_METAL_ENDPOINT

"$WINE" wineboot -i >/dev/null 2>&1 || true
echo "Host ready: port=$PORT compute_pipeline=$compute_id graphics_pipeline=${graphics_id:-0}"
echo "Running the Win64 AAA guest under Wine (warmup=$VICTORY_FRAME_WARMUP measured=$VICTORY_FRAME_MEASURED)..."
"$WINE" "$GUEST_EXE" >"$GUEST_LOG" 2>&1 &
GUEST_PID=$!
victory_hygiene_track "$GUEST_PID"
guest_deadline=$(( $(date +%s) + 300 ))
while kill -0 "$GUEST_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$guest_deadline" ]; then
        echo "FAIL: the Wine guest did not finish within 300 s" >&2
        kill "$GUEST_PID" 2>/dev/null || true
        break
    fi
    sleep 0.2
done
GUEST_RC=0
victory_hygiene_wait_pid "$GUEST_PID" 10 || GUEST_RC=$?
victory_hygiene_untrack "$GUEST_PID"
GUEST_PID=""

host_deadline=$(( $(date +%s) + 60 ))
while kill -0 "$HOST_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$host_deadline" ]; then
        echo "WARN: the host did not exit after the guest; stopping it" >&2
        kill "$HOST_PID" 2>/dev/null || true
        break
    fi
    sleep 0.1
done
HOST_RC=0
victory_hygiene_wait_pid "$HOST_PID" 10 || HOST_RC=$?
victory_hygiene_untrack "$HOST_PID"
HOST_PID=""

cat "$GUEST_LOG" 2>/dev/null || true
[ -s "$HOST_ERR" ] && cat "$HOST_ERR" >&2 || true

if ! grep -q '^FRAME_BENCH_DONE ' "$GUEST_LOG" 2>/dev/null; then
    echo "FAIL: the guest did not report FRAME_BENCH_DONE (exit $GUEST_RC)" >&2
    tail -20 "$GUEST_LOG" 2>/dev/null >&2 || true
    exit 3
fi
if [ "$GUEST_RC" -ne 0 ]; then
    echo "FAIL: the guest exited $GUEST_RC after reporting a failure" >&2
    exit 3
fi
if [ "$HOST_RC" -ne 0 ]; then
    echo "FAIL: the host exited $HOST_RC" >&2
    cat "$HOST_ERR" >&2 || true
    exit 3
fi

# --- Report measured numbers ----------------------------------------------
PY_EXIT=0
python3 - "$FRAMES_CSV" "$GUEST_REPORT" "$TELEMETRY" "$PORT" "$VICTORY_FRAME_WARMUP" "$VICTORY_FRAME_MEASURED" <<'PY' || PY_EXIT=$?
import json
import math
import os
import sys

csv_path, report_path, telemetry_path, port, warmup, measured_expected = sys.argv[1:7]
measured_expected = int(measured_expected)

values = []
if os.path.exists(csv_path):
    with open(csv_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                values.append(float(line))

report = {}
if os.path.exists(report_path):
    with open(report_path) as handle:
        for line in handle:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                report[key] = value

telemetry = {}
if os.path.exists(telemetry_path):
    try:
        with open(telemetry_path) as handle:
            telemetry = json.load(handle)
    except json.JSONDecodeError:
        telemetry = {}


def percentile(sorted_values, percent):
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percent / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


values.sort()
count = len(values)
p50 = percentile(values, 50)
p95 = percentile(values, 95)
p99 = percentile(values, 99)
mean = sum(values) / count if count else 0.0
fps = 1000.0 / mean if mean > 0 else 0.0

print(f"FRAME_BENCH host_port={port} warmup={warmup} measured={count}/{measured_expected}")
print(f"FRAME_BENCH p50_ms={p50:.3f} p95_ms={p95:.3f} p99_ms={p99:.3f} fps={fps:.2f}")
print(
    "FRAME_BENCH frames_ok={} executions={} checksum={} first_word={} expected_first={}".format(
        report.get("frames_ok", "0"),
        report.get("executions", "0"),
        report.get("checksum", "0"),
        report.get("first_word", "?"),
        report.get("expected_first", "?"),
    )
)
print(
    "FRAME_BENCH replay_guard status={} unchanged={} fence_read_ns={}".format(
        report.get("replay_status", "?"),
        report.get("replay_unchanged", "?"),
        report.get("fence_read_ns", "?"),
    )
)
print(
    "FRAME_BENCH aaa texture0_status={} texture1_status={} depth_status={} "
    "surface_status={} graphics_status={} present_status={} "
    "present_checksum={} expected_present_checksum={} "
    "shader_path=msl-precompiled cold_vs_warm=n/a".format(
        report.get("texture0_status", "?"),
        report.get("texture1_status", "?"),
        report.get("depth_status", "?"),
        report.get("surface_status", "?"),
        report.get("graphics_status", "?"),
        report.get("present_status", "?"),
        telemetry.get("present_checksum", "?"),
        report.get("expected_present_checksum", "?"),
    )
)
print(
    "FRAME_BENCH d3dmetal_calls={} host_fallback_count={} guest_modules={} guest_d3dmetal_modules={}".format(
        0,
        telemetry.get("d3dmetal_fallback_count", "?"),
        report.get("guest_modules", "?"),
        report.get("d3dmetal_modules", "?"),
    )
)
print(
    "FRAME_BENCH host_telemetry frames={} submissions={} p50_ms={} p95_ms={} p99_ms={} fps={}".format(
        telemetry.get("frames", "?"),
        telemetry.get("submissions", "?"),
        telemetry.get("p50_ms", "?"),
        telemetry.get("p95_ms", "?"),
        telemetry.get("p99_ms", "?"),
        telemetry.get("fps", "?"),
    )
)

problems = []
if report.get("replay_status") != "3":
    problems.append("AlreadyExecuted replay guard status=" + str(report.get("replay_status")))
if report.get("replay_unchanged") != "1":
    problems.append("the refused replay changed the guest alias")
if report.get("d3dmetal_modules") != "0":
    problems.append("the guest loaded a D3DMetal module")
if report.get("first_word") != report.get("expected_first"):
    problems.append(
        "compute result mismatch {} != {}".format(
            report.get("first_word"), report.get("expected_first")
        )
    )
if count != measured_expected:
    problems.append(f"measured {count} of {measured_expected} frames")
if telemetry.get("d3dmetal_fallback_count") != 0:
    problems.append("d3dmetal_fallback_count != 0")
if telemetry.get("graphics_status") not in (0, "0"):
    problems.append("host graphics_status=" + str(telemetry.get("graphics_status")))
if telemetry.get("present_status") not in (0, "0"):
    problems.append("host present_status=" + str(telemetry.get("present_status")))

# Every AAA stage status is reported raw, so a refusal is attributed to the
# exact registration or pass that produced it. A non-zero step is a real,
# labelled gap, never a fabricated pass.
stages = [
    ("texture0", report.get("texture0_status", "?")),
    ("texture1", report.get("texture1_status", "?")),
    ("depth", report.get("depth_status", "?")),
    ("surface", report.get("surface_status", "?")),
    ("graphics", report.get("graphics_status", "?")),
    ("present", report.get("present_status", "?")),
]
first_failure = next(((name, value) for name, value in stages if value != "0"), None)
if first_failure is not None:
    print(
        "PARTIAL aaa-frame-bench first_failing_step={} reply_status={} "
        "texture0_status={} texture1_status={} depth_status={} surface_status={} "
        "graphics_status={} present_status={} "
        "reason=\"the AAA stage was refused by the current session\"".format(
            first_failure[0],
            first_failure[1],
            report.get("texture0_status", "?"),
            report.get("texture1_status", "?"),
            report.get("depth_status", "?"),
            report.get("surface_status", "?"),
            report.get("graphics_status", "?"),
            report.get("present_status", "?"),
        )
    )
    sys.exit(1)

expected_checksum = report.get("expected_present_checksum", "?")
host_checksum = telemetry.get("present_checksum", "?")
checksum_ok = False
if expected_checksum not in ("", "?") and host_checksum not in ("", "?", -1, "-1"):
    try:
        checksum_ok = int(host_checksum) == int(expected_checksum, 16)
    except (TypeError, ValueError):
        checksum_ok = False
if not checksum_ok:
    problems.append(
        "present checksum mismatch host={} expected={}".format(
            host_checksum, expected_checksum
        )
    )

if problems:
    print("PARTIAL aaa-frame-bench reason=\"" + "; ".join(problems) + "\"")
    sys.exit(1)

print(
    "PASS aaa-frame-bench measured={}/{} p50_ms={:.3f} fps={:.2f} "
    "checksum={} d3dmetal_calls=0".format(
        count, measured_expected, p50, fps, host_checksum
    )
)
PY

echo "aaa-frame-bench: exit $PY_EXIT"
exit "$PY_EXIT"
