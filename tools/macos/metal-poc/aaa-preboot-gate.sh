#!/bin/sh
# C6 / Wave 6B: unified AAA pre-boot gate.
#
# One command boots the ARM64 `victory_metal_host` through the Wave 6A
# `run-title-metal.sh`, runs the Win64 `aaa_preboot_gate` guest twice under
# Wine, and reports both runs plus a per-feature verdict.
#
# The guest needs six prebuilt metallibs and their converter reflection JSONs
# (vertex, pixel, compute, object/amplification, mesh, mesh pixel). The launcher
# only prepares its own five stages, so this wrapper compiles the gate's exact
# shaders with the checked-in DXC and translates them through the same
# `victory-irconverter --stdio` protocol the launcher's pool uses. It never
# edits `run-title-metal.sh`.
#
# What the gate exercises (all in one submission stream, every frame) is
# documented in `crates/victory-gpu/tests/aaa_preboot_gate.rs`.
#
# Exclusive GPU: `run-title-metal.sh` takes `/tmp/victory-gpu.lock` for each
# run. If the caller already holds that lock (the validation command does), the
# wrapper hands the directory over to the launcher and restores it if free.
#
# Usage: tools/macos/metal-poc/aaa-preboot-gate.sh [--out DIR] [--runs N]
#
# Environment overrides: WINE, DXC_DIR, VICTORY_METAL_IRCONVERTER,
# VICTORY_FRAME_WARMUP (20), VICTORY_FRAME_MEASURED (120).
#
# Exit codes:
#   0   both runs observed every feature
#   12  the gate report exists but a Cycle 06 feature is not observed
#   3   setup failure; 10 a guest failed; 11 a D3DMetal fallback; 4 a leak
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
START_DIR="$(pwd)"
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

OUT=""
RUNS=2
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)
            [ "$#" -ge 2 ] || { echo "aaa-preboot-gate.sh: --out requires a value" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --runs)
            [ "$#" -ge 2 ] || { echo "aaa-preboot-gate.sh: --runs requires a value" >&2; exit 2; }
            RUNS="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "aaa-preboot-gate.sh: unknown option $1" >&2
            exit 2
            ;;
    esac
done
case "$RUNS" in
    *[!0-9]*|"") echo "aaa-preboot-gate.sh: --runs must be a positive integer" >&2; exit 2 ;;
    0) echo "aaa-preboot-gate.sh: --runs must be a positive integer" >&2; exit 2 ;;
esac

cd "$ROOT"
if [ -n "$OUT" ]; then
    case "$OUT" in
        /*) ;;
        *) OUT="$START_DIR/$OUT" ;;
    esac
else
    OUT="$(mktemp -d "${TMPDIR:-/tmp}/victory-aaa-gate.XXXXXXXX")"
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-aaa-pool.XXXXXXXX")"
HLSL_DIR="$WORK/hlsl"
DXIL_DIR="$WORK/dxil"
METALLIB_DIR="$WORK/metallib"
WINEPREFIX_DIR="$WORK/wineprefix"
mkdir -p "$HLSL_DIR" "$DXIL_DIR" "$METALLIB_DIR"

LOCK="/tmp/victory-gpu.lock"
CALLER_LOCK=0

cleanup() {
    status=$?
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    rm -rf "$WORK"
    # The validation command holds the lock across this script; restore it so
    # its EXIT trap can remove an owned directory. Never touch a lock another
    # task acquired in the meantime.
    if [ "$CALLER_LOCK" = 1 ] && [ ! -d "$LOCK" ]; then
        mkdir "$LOCK" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

fail_setup() {
    echo "aaa-preboot-gate.sh: FAIL: $1" >&2
    exit 3
}

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

# --- Locate the tools -------------------------------------------------------
if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$ROOT/../../../VIctory-macOS/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
[ -n "$WINE" ] && [ -x "$WINE" ] || fail_setup "no Wine executable found (set WINE=...)"
export WINE

find_tool() {
    for base in "${TOOLCHAIN_ROOT:-}" "$ROOT/../../.." "$ROOT" "$HOME/developer/compatibility"; do
        [ -n "$base" ] || continue
        if [ -e "$base/$1" ]; then
            printf '%s/%s' "$base" "$1"
            return 0
        fi
    done
    return 1
}

DXC_DIR="${DXC_DIR:-$(dirname "$(find_tool toolchains/dxc/bin/x64/dxc.exe 2>/dev/null || echo /nonexistent/dxc.exe)")}"
DXC_EXE="$DXC_DIR/dxc.exe"
[ -f "$DXC_EXE" ] || fail_setup "dxc.exe not found (set DXC_DIR)"
CONV="${VICTORY_METAL_IRCONVERTER:-$HOME/Library/Caches/victory-macos/target/wine-cx/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib}"
[ -f "$CONV" ] || fail_setup "libmetalirconverter.dylib not found (set VICTORY_METAL_IRCONVERTER)"
IRCONV="${CARGO_TARGET_DIR:-$ROOT/target}/x86_64-apple-darwin/debug/victory-irconverter"
case "$IRCONV" in
    /*) ;;
    *) IRCONV="$ROOT/$IRCONV" ;;
esac

# --- Hand the GPU lock to the launcher -------------------------------------
# run-title-metal.sh acquires /tmp/victory-gpu.lock itself. If the caller
# (the validation command) already holds it, release it now; the launcher
# reacquires immediately, so the gate stays serialised.
if [ -d "$LOCK" ]; then
    if rmdir "$LOCK" 2>/dev/null; then
        CALLER_LOCK=1
    fi
fi

# --- Build the x86_64 converter worker if it is missing --------------------
if [ ! -x "$IRCONV" ]; then
    echo "aaa-preboot-gate.sh: building the x86_64 victory-irconverter worker..."
    cargo build --locked -p victory-metal --target x86_64-apple-darwin --bin victory-irconverter \
        || fail_setup "victory-irconverter did not build"
fi
[ -x "$IRCONV" ] || fail_setup "missing $IRCONV"

# --- Write the gate's exact shaders ----------------------------------------
RS='RootFlags(0), DescriptorTable(CBV(b0, numDescriptors=2)), DescriptorTable(SRV(t0, numDescriptors=2)), DescriptorTable(UAV(u0, numDescriptors=2)), RootConstants(b2, space=0, num32BitConstants=4)'

cat >"$HLSL_DIR/vs.hlsl" <<EOS
// Geometry is selected by the root-constant selector (rc.w) so the draw does
// not depend on instance-id semantics; the full root signature is shared with
// the pixel stage so parameter index 3 is the same on both reflections.
cbuffer C0 : register(b0) { float4 c0v; };
cbuffer C1 : register(b1) { float4 c1v; };
StructuredBuffer<uint> s0 : register(t0);
StructuredBuffer<uint> s1 : register(t1);
RWStructuredBuffer<uint> u[2] : register(u0);
cbuffer RC : register(b2) { float4 rc; };
struct VSOut { float4 pos : SV_Position; };
[RootSignature("$RS")]
VSOut main(uint vid : SV_VertexID) {
    float2 c;
    if (rc.w > 1.5) {
        // Clockwise middle-left quad at z=0.25: culled by Back.
        float2 q[6] = { float2(-0.5,-1.0), float2(-0.5,1.0), float2(0.0,-1.0),
                        float2(-0.5,1.0), float2(0.0,1.0), float2(0.0,-1.0) };
        c = q[vid % 6u];
    } else if (rc.w < 0.5) {
        // Counter-clockwise bottom-right quad at z=0.75.
        float2 q[6] = { float2(0.0,-1.0), float2(1.0,-1.0), float2(1.0,0.0),
                        float2(0.0,-1.0), float2(1.0,0.0), float2(0.0,0.0) };
        c = q[vid % 6u];
    } else {
        // Full-screen triangle at z=1.0: rejected by Less.
        float2 t[3] = { float2(-1.0,-1.0), float2(3.0,-1.0), float2(-1.0,3.0) };
        c = t[vid % 3u];
    }
    float z = (rc.w > 0.5 && rc.w < 1.5) ? 1.0 : ((rc.w > 1.5) ? 0.25 : 0.75);
    VSOut o;
    o.pos = float4(c, z, 1.0);
    return o;
}
EOS

cat >"$HLSL_DIR/ps.hlsl" <<EOS
cbuffer C0 : register(b0) { float4 c0v; };
cbuffer C1 : register(b1) { float4 c1v; };
StructuredBuffer<uint> s0 : register(t0);
StructuredBuffer<uint> s1 : register(t1);
RWStructuredBuffer<uint> u[2] : register(u0);
cbuffer RC : register(b2) { float4 rc; };
struct VSOut { float4 pos : SV_Position; };
struct PSOut { float4 a : SV_Target0; float4 b : SV_Target1; };
[RootSignature("$RS")]
PSOut main(VSOut input) {
    PSOut o;
    float bias = float(s0[0] + s1[0]) * 0.0;
    o.a = c0v + float4(rc.x, rc.y, rc.z, 0.0) + float4(bias, 0.0, 0.0, 0.0);
    if (rc.w > 1.5) {
        o.a = float4(0.0, 1.0, 1.0, 1.0);       // cull probe colour
    } else if (rc.w > 0.5) {
        o.a = float4(1.0, 1.0, 0.0, 1.0);       // depth-fail probe colour
    }
    o.b = c1v;
    // Coverage counters: the guest reads them back from the shared allocation.
    uint px = (uint)input.pos.x;
    uint py = (uint)input.pos.y;
    InterlockedAdd(u[0][0], 1u);                                  // graphics pixels
    if (px >= 640u && py >= 360u) InterlockedAdd(u[0][1], 1u);    // near quad
    if (px >= 320u && px < 640u) InterlockedAdd(u[0][2], 1u);     // cull band
    if (rc.w > 0.5 && rc.w < 1.5) InterlockedAdd(u[0][3], 1u);    // far draw survived
    if (rc.w > 1.5) InterlockedAdd(u[0][4], 1u);                  // cull draw survived
    u[0][5] = 11u;
    u[1][0] = 22u;
    return o;
}
EOS

cat >"$HLSL_DIR/cs.hlsl" <<'EOS'
// The culling kernel writes the indirect mesh group-count record.
RWStructuredBuffer<uint> gArgs : register(u0);
[RootSignature("RootFlags(0), UAV(u0)")]
[numthreads(64,1,1)]
void main(uint tid : SV_DispatchThreadID) {
    uint lanes = WaveGetLaneCount();
    uint sum = WaveActiveSum(tid + 1u);
    if (tid == 0u) {
        gArgs[0] = 1u;
        gArgs[1] = 1u;
        gArgs[2] = 1u;
        gArgs[3] = 0xC0DEu;
    }
    if (sum == 0xdeadbeefu && lanes == 12345u) {
        gArgs[3] = 0xdeadbeefu;
    }
}
EOS

cat >"$HLSL_DIR/as.hlsl" <<'EOS'
// Amplification shader: fills an 8-byte payload and writes its own probe.
struct Payload { uint vertexCount; uint tint; };
RWStructuredBuffer<uint> gProbe : register(u0);
[outputtopology("triangle")]
[numthreads(1,1,1)]
[RootSignature("RootFlags(0), UAV(u0)")]
void main(in uint gid : SV_GroupID) {
    Payload p;
    p.vertexCount = 3u;
    p.tint = 1u;
    gProbe[0] = p.vertexCount;
    gProbe[1] = p.tint;
    gProbe[2] = 0xC0DEu;
    DispatchMesh(1, 1, 1, p);
}
EOS

cat >"$HLSL_DIR/ms.hlsl" <<'EOS'
struct Payload { uint vertexCount; uint tint; };
struct VOut { float4 pos : SV_Position; float4 color : COLOR0; };
[outputtopology("triangle")]
[numthreads(32,1,1)]
[RootSignature("RootFlags(0)")]
void main(in payload Payload p, out indices uint3 tri[1], out vertices VOut verts[3], uint gid : SV_GroupID) {
    SetMeshOutputCounts(3, 1);
    float2 c[3] = { float2(-1.0,-1.0), float2(3.0,-1.0), float2(-1.0,3.0) };
    for (uint i = 0; i < 3; i++) {
        verts[i].pos = float4(c[i], 0.25, 1.0);
        verts[i].color = float4(0.0, 1.0, 0.0, 1.0);
    }
    tri[0] = uint3(0, 1, 2);
}
EOS

cat >"$HLSL_DIR/psm.hlsl" <<'EOS'
// The mesh pixel stage binds the same probe UAV the object stage writes, so
// its coverage counter proves the amplified mesh geometry rasterised.
RWStructuredBuffer<uint> gMesh : register(u0);
struct PSOut { float4 a : SV_Target0; float4 b : SV_Target1; };
[RootSignature("RootFlags(0), UAV(u0)")]
PSOut main() {
    PSOut o;
    o.a = float4(0.0, 1.0, 0.0, 1.0);
    o.b = float4(0.0, 1.0, 1.0, 1.0);
    InterlockedAdd(gMesh[3], 1u);
    return o;
}
EOS

# --- Compile DXIL with the checked-in DXC -----------------------------------
echo "aaa-preboot-gate.sh: compiling the gate's six shader stages with dxc..."
export WINEPREFIX="$WINEPREFIX_DIR"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d"
"$WINE" wineboot -i >"$WORK/wineboot.log" 2>&1 || true

compile_dxil() {
    # $1 = stem, $2 = profile
    stem=$1
    profile=$2
    src="$(winpath "$HLSL_DIR/$stem.hlsl")"
    out="$(winpath "$DXIL_DIR/$stem.dxil")"
    if ! ( cd "$DXC_DIR" && "$WINE" ./dxc.exe -T "$profile" -E main -Od -Fo "$out" "$src" ) \
        >"$WORK/dxc-$stem.log" 2>&1; then
        echo "aaa-preboot-gate.sh: dxc failed for $stem ($profile)" >&2
        tail -20 "$WORK/dxc-$stem.log" >&2 || true
        exit 3
    fi
    [ -s "$DXIL_DIR/$stem.dxil" ] || fail_setup "dxc produced no $stem.dxil"
}
compile_dxil vs vs_6_0
compile_dxil ps ps_6_0
compile_dxil cs cs_6_6
compile_dxil as as_6_5
compile_dxil ms ms_6_5
compile_dxil psm ps_6_0

# --- Translate through the real --stdio reflection worker -------------------
echo "aaa-preboot-gate.sh: translating the stages with victory-irconverter..."
if ! python3 - "$IRCONV" "$CONV" "$DXIL_DIR" "$METALLIB_DIR" <<'PY'
import json
import os
import subprocess
import sys

worker, converter, dxil_dir, out_dir = sys.argv[1:5]
jobs = [("vs", 1), ("ps", 2), ("cs", 8), ("as", 6), ("ms", 5), ("psm", 2)]
env = dict(os.environ)
env["VICTORY_IRCONVERTER_STDIO_REFLECTION"] = "1"
proc = subprocess.Popen(
    [worker, "--stdio", "--converter", converter],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
)
for index, (name, stage) in enumerate(jobs):
    dxil = open(os.path.join(dxil_dir, name + ".dxil"), "rb").read()
    entry = b"main"
    header = f"VMJOB {index + 1} {1000 + index} {stage} {len(entry)} {len(dxil)}\n".encode()
    proc.stdin.write(header + entry + dxil)
    proc.stdin.flush()
    line = proc.stdout.readline()
    parts = line.split()
    if not parts or parts[0] != b"VMRES2":
        print(f"CONVERT_ERROR {name} protocol={line[:120]!r}", file=sys.stderr)
        sys.exit(3)
    status = int(parts[3])
    metallib_len = int(parts[4])
    reflection_len = int(parts[5])
    metallib = proc.stdout.read(metallib_len)
    reflection = proc.stdout.read(reflection_len)
    print(
        f"GATE_STAGE name={name} stage={stage} status={status} "
        f"metallib={len(metallib)} reflection={len(reflection)}"
    )
    if status != 0 or not metallib.startswith(b"MTLB"):
        print(f"CONVERT_ERROR {name} status={status} message={metallib[:200]!r}", file=sys.stderr)
        sys.exit(3)
    if not reflection:
        print(f"CONVERT_ERROR {name} empty reflection", file=sys.stderr)
        sys.exit(3)
    parsed = json.loads(reflection)
    if "TopLevelArgumentBuffer" not in parsed:
        print(f"CONVERT_ERROR {name} reflection has no TopLevelArgumentBuffer", file=sys.stderr)
        sys.exit(3)
    open(os.path.join(out_dir, name + ".metallib"), "wb").write(metallib)
    open(os.path.join(out_dir, name + ".reflection.json"), "wb").write(reflection)
proc.stdin.close()
proc.wait()
sys.exit(0 if proc.returncode == 0 else 3)
PY
then
    fail_setup "the reflection worker refused a gate stage"
fi

for stem in vs ps cs as ms psm; do
    [ -s "$METALLIB_DIR/$stem.metallib" ] || fail_setup "missing $stem.metallib"
    [ -s "$METALLIB_DIR/$stem.reflection.json" ] || fail_setup "missing $stem.reflection.json"
done

export VICTORY_AAA_GATE_METALLIB_VS="$(winpath "$METALLIB_DIR/vs.metallib")"
export VICTORY_AAA_GATE_METALLIB_PS="$(winpath "$METALLIB_DIR/ps.metallib")"
export VICTORY_AAA_GATE_METALLIB_CS="$(winpath "$METALLIB_DIR/cs.metallib")"
export VICTORY_AAA_GATE_METALLIB_AS="$(winpath "$METALLIB_DIR/as.metallib")"
export VICTORY_AAA_GATE_METALLIB_MS="$(winpath "$METALLIB_DIR/ms.metallib")"
export VICTORY_AAA_GATE_METALLIB_PSM="$(winpath "$METALLIB_DIR/psm.metallib")"
export VICTORY_AAA_GATE_REFLECTION_VS="$(winpath "$METALLIB_DIR/vs.reflection.json")"
export VICTORY_AAA_GATE_REFLECTION_PS="$(winpath "$METALLIB_DIR/ps.reflection.json")"
export VICTORY_AAA_GATE_REFLECTION_CS="$(winpath "$METALLIB_DIR/cs.reflection.json")"
export VICTORY_AAA_GATE_REFLECTION_AS="$(winpath "$METALLIB_DIR/as.reflection.json")"
export VICTORY_AAA_GATE_REFLECTION_MS="$(winpath "$METALLIB_DIR/ms.reflection.json")"
export VICTORY_AAA_GATE_REFLECTION_PSM="$(winpath "$METALLIB_DIR/psm.reflection.json")"

# --- Real xmem registry: the production bounded-retired-map test ------------
XMEM_STATUS="unavailable"
XMEM_LINE=""
echo "aaa-preboot-gate.sh: running the real victory-hle xmem registry stress under Wine..."
if tools/macos/test-wine.sh -p victory-hle --lib \
    sustained_streaming_bounds_the_retired_submission_map \
    -- --nocapture >"$OUT/xmem-registry.log" 2>&1; then
    XMEM_LINE="$(grep -E 'bounded share registry:' "$OUT/xmem-registry.log" | tail -1)"
    if [ -n "$XMEM_LINE" ]; then
        XMEM_STATUS="measured"
    else
        XMEM_STATUS="line_missing"
    fi
else
    XMEM_STATUS="test_failed"
    XMEM_LINE="$(tail -3 "$OUT/xmem-registry.log" | tr '\n' ' ')"
fi
echo "XMEM_REAL status=$XMEM_STATUS $XMEM_LINE"

# --- Build the Win64 guest once --------------------------------------------
echo "aaa-preboot-gate.sh: building the Win64 aaa_preboot_gate guest..."
GUEST_EXE="$(cargo test --locked --no-run -p victory-gpu \
    --target x86_64-pc-windows-gnullvm --test aaa_preboot_gate \
    --message-format=json 2>"$OUT/guest-build.log" \
    | python3 -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("reason") == "compiler-artifact" and record.get("executable"):
        exe = record["executable"]
print(exe or "")
')"
if [ -z "$GUEST_EXE" ] || [ ! -f "$GUEST_EXE" ]; then
    echo "aaa-preboot-gate.sh: FAIL: the Win64 guest did not build" >&2
    tail -20 "$OUT/guest-build.log" >&2 || true
    exit 3
fi

# --- Run the gate twice -----------------------------------------------------
SUMMARY="$OUT/aaa-preboot.summary.txt"
: >"$SUMMARY"
RUN_METRICS="$OUT/metrics.env"
: >"$RUN_METRICS"

run_once() {
    n=$1
    run_out="$OUT/run$n"
    mkdir -p "$run_out"
    log="$run_out/launcher.log"
    echo "aaa-preboot-gate.sh: run $n/2 through run-title-metal.sh..."
    rc=0
    VICTORY_LIVE_GATE=1 \
    VICTORY_AAA_PREBOOT_GATE=1 \
    VICTORY_FRAME_WARMUP="${VICTORY_FRAME_WARMUP:-20}" \
    VICTORY_FRAME_MEASURED="${VICTORY_FRAME_MEASURED:-120}" \
    "$ROOT/tools/macos/metal-poc/run-title-metal.sh" --headless --keep --out "$run_out" -- \
        "$GUEST_EXE" --ignored --exact aaa_preboot_guest_gate --test-threads=1 --nocapture \
        >"$log" 2>&1 || rc=$?
    echo "RUN_STATUS run=$n launcher_exit=$rc"
    RUN_RC="$rc"

    kept="$(sed -n 's/.*kept session files in //p' "$log" | tail -1)"
    if [ -n "$kept" ] && [ -d "$kept/share" ]; then
        cp "$kept/share/live-gate.report" "$run_out/report.txt" 2>/dev/null || true
        cp "$kept/share/live-gate.frames.csv" "$run_out/frames.csv" 2>/dev/null || true
        cp -R "$kept/share/pipelines" "$run_out/pipelines" 2>/dev/null || true
    fi
    if [ -f "$run_out/summary.txt" ]; then
        cp "$run_out/summary.txt" "$OUT/launcher-run$n-summary.txt"
    fi
    if [ -f "$run_out/report.txt" ]; then
        : >"$OUT/report-run$n.txt"
        cat "$run_out/report.txt" >>"$OUT/report-run$n.txt"
    fi
}

run_once 1
run_once 2

# --- Evaluate both runs -----------------------------------------------------
set +e
python3 - "$OUT" "$RUNS" "$XMEM_STATUS" "$XMEM_LINE" "$SUMMARY" <<'PY'
import hashlib
import json
import math
import os
import re
import sys

out, runs_text, xmem_status, xmem_line, summary_path = sys.argv[1:6]
runs = int(runs_text)
problems = []

W, H = 1280, 720
CLEAR0 = bytes([0, 0, 255, 255])
CLEAR1 = bytes([255, 0, 255, 255])
GRAPHICS0 = bytes([255, 64, 0, 255])
GRAPHICS1 = bytes([0, 255, 0, 255])
MESH = bytes([0, 255, 0, 255])
MESH1 = bytes([0, 255, 255, 255])


def model_image():
    image = bytearray(CLEAR0 * (W * H))
    def fill(x0, y0, x1, y1, colour):
        for y in range(y0, y1):
            base = (y * W + x0) * 4
            image[base:base + (x1 - x0) * 4] = colour * (x1 - x0)
    fill(0, 0, 320, H, MESH)             # indirect mesh
    fill(640, 0, 1280, 360, MESH)        # direct mesh
    fill(640, 360, 1280, H, GRAPHICS0)   # graphics near quad
    fill(320, 0, 324, 4, GRAPHICS1)      # copy of colour1's graphics region
    fill(320, 4, 324, 8, CLEAR1)         # copy of colour1's clear region
    for row, colour in enumerate([
        bytes([255, 0, 0, 255]),
        bytes([0, 255, 0, 255]),
        bytes([0, 0, 255, 255]),
        bytes([255, 255, 255, 255]),
    ]):
        fill(0, row, 4, row + 1, colour)  # pitched upload block
    return image


def fnv1a64(data):
    # The host's `checksum_rgba8` uses this exact (shortened) offset basis;
    # matching it is what makes the model comparable to `present_checksum`.
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def percentile(sorted_values, percent):
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percent / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def read_report(path):
    values = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def read_summary(path):
    values = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            line = line.strip()
            if "=" in line and not line.startswith("launcher="):
                key, value = line.split("=", 1)
                values[key] = value
    return values


def read_csv(path):
    values = []
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    values.sort()
    return values


model = model_image()
model_hash = fnv1a64(model)

print(f"model_checksum={model_hash}")
report_metrics = []
for run in range(1, runs + 1):
    report = read_report(os.path.join(out, f"report-run{run}.txt"))
    summary = read_summary(os.path.join(out, "launcher-run%d-summary.txt" % run))
    frames = read_csv(os.path.join(out, f"run{run}", "frames.csv"))
    p50 = percentile(frames, 50)
    p95 = percentile(frames, 95)
    p99 = percentile(frames, 99)
    mean = sum(frames) / len(frames) if frames else 0.0
    fps = 1000.0 / mean if mean > 0 else 0.0
    checksum_text = summary.get("present_checksum", "n/a")
    checksum = None
    try:
        checksum = int(checksum_text)
    except (TypeError, ValueError):
        checksum = None
    model_match = checksum == model_hash
    print(
        f"run{run}: status={summary.get('status','missing')} "
        f"launcher_exit={summary.get('exit_code','n/a')} "
        f"frames={len(frames)} p50_ms={p50:.3f} p95_ms={p95:.3f} p99_ms={p99:.3f} "
        f"fps={fps:.2f} present_checksum={checksum_text} model_match={int(model_match)} "
        f"(model is informational; the measured checksum is the exact evidence)"
    )
    print(
        f"run{run}_gate: ok={report.get('ok','missing')} "
        f"first_failure={report.get('first_failing_stage','n/a')} "
        f"reason={report.get('first_failing_reason','n/a')}"
    )
    for key in (
        "frames_ok",
        "fence_waits",
        "fence_timeouts",
        "d3dmetal_fallback_count",
        "guest_d3dmetal_module",
        "unsupported_opcodes",
        "compute_mesh_counts_after",
        "object_payload_after",
        "uav_table_writes_after",
        "xmem_churn_cycles",
        "xmem_retired_map_peak",
        "xmem_retired_map_capacity",
        "blend_probe_status",
        "texture_table_probe_status",
        "submits_ok",
        "submits_refused",
        "presents_ok",
        "presents_refused",
        "telemetry_p50_us",
        "telemetry_p95_us",
        "telemetry_p99_us",
        "telemetry_fps",
    ):
        print(f"run{run}_{key}={report.get(key, 'n/a')}")
    if checksum is not None and checksum < 0:
        problems.append(f"run {run}: no present checksum")
    try:
        frames_expected = int(report.get("measured_expected", "120"))
    except ValueError:
        frames_expected = 120
    if len(frames) != frames_expected:
        problems.append(
            f"run {run}: measured {len(frames)} frames, expected {frames_expected}"
        )
    not_observed = [
        key
        for key, value in report.items()
        if value.startswith("observed=false") or value.startswith("NOT_WIRED")
    ]
    print(f"run{run}_not_observed={','.join(not_observed) if not_observed else 'none'}")
    if report.get("ok") != "1":
        problems.append(
            f"run {run}: gate not ok ({report.get('first_failing_stage','?')}: "
            f"{report.get('first_failing_reason','?')}); not observed: "
            f"{','.join(not_observed) if not_observed else 'unknown'}"
        )
    report_metrics.append((p50, p95, p99, fps, checksum))

checksums = [metrics[4] for metrics in report_metrics if metrics[4] is not None]
if len(checksums) == runs and len(set(checksums)) != 1:
    problems.append(f"the two runs' present checksums differ: {checksums}")

if xmem_status != "measured":
    problems.append(f"real xmem registry measurement: {xmem_status} {xmem_line}")
else:
    match = re.search(r"cycles=(\d+) peak=(\d+) capacity=(\d+) watermark=(\d+)", xmem_line)
    if not match:
        problems.append(f"real xmem registry line unparsable: {xmem_line}")
    elif int(match.group(1)) < 10000 or int(match.group(2)) > int(match.group(3)):
        problems.append(f"real xmem registry not bounded: {xmem_line}")

status = "PASS" if not problems else "PARTIAL"
lines = [
    "launcher=aaa-preboot-gate",
    f"status={status}",
    f"runs={runs}",
    f"model_checksum={model_hash}",
    f"xmem_real_status={xmem_status}",
    f"xmem_real_line={xmem_line or 'none'}",
]
for index, (p50, p95, p99, fps, checksum) in enumerate(report_metrics, start=1):
    lines.append(
        f"run{index}_p50_ms={p50:.3f} run{index}_p95_ms={p95:.3f} "
        f"run{index}_p99_ms={p99:.3f} run{index}_fps={fps:.2f} "
        f"run{index}_checksum={checksum}"
    )
for problem in problems:
    lines.append(f"problem={problem}")
with open(summary_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if status == "PASS" else 12)
PY
summary_rc=$?
set -e

if [ "$summary_rc" -ne 0 ]; then
    echo "aaa-preboot-gate.sh: PARTIAL (exit $summary_rc); summary: $SUMMARY"
    exit "$summary_rc"
fi
echo "aaa-preboot-gate.sh: PASS; summary: $SUMMARY"
exit 0
