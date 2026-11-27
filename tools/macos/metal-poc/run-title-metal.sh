#!/bin/sh
# C6 / ADR-026b: one-command diagnostic launcher for a Win64 Wine guest on the
# native ARM64 Metal host.
#
# It starts `victory_metal_host`, waits for the machine-readable ready line,
# runs an arbitrary Wine Win64 executable with the full `VICTORY_*` environment,
# captures a structured `key=value` summary, and shuts the daemon down on
# exit/interrupt. When the guest is the existing `live_d3d12_gate` binary it
# also prepares the DXIL pool metallibs and reflection-derived `VAB1` ABIs the
# gate needs (the same stages `live-d3d12-gate.sh` runs), so the gate can be
# driven through this single entry point.
#
# Usage:
#   run-title-metal.sh [--window|--headless] [--frames N] [--out DIR] [--keep]
#                      -- <exe> [guest args...]
#
#   --window    pass `--window` to the host so the native window presents
#   --headless  force headless (default); refuses together with --window
#   --frames N  diagnostic frame count, exported as VICTORY_FRAME_MEASURED
#   --out DIR   directory for guest.log, host.log, telemetry.json, summary.txt
#               (default: a fresh mktemp directory, printed on exit)
#   --keep      keep the isolated session directory (share dir, wineprefix)
#
# Environment overrides are honoured where they are meaningful (WINE, DXC_DIR,
# TOOLCHAIN_ROOT, VICTORY_METAL_IRCONVERTER, VICTORY_FRAME_WARMUP,
# VICTORY_FRAME_MEASURED, and the gate asset paths). The connection variables
# (`VICTORY_BACKEND`, port, secret, share dir, fence page) are owned by this
# launcher.
#
# Exit codes:
#   0  guest exited 0, no D3DMetal fallback/module and, in gate mode, a clean
#      guest report
#   2  usage error
#   3  setup failure (missing Wine, host build, daemon not ready, ...)
#   4  a process from this session leaked after teardown
#  10  the guest exited non-zero
#  11  a D3DMetal fallback / module / image was observed
#  12  the gate guest report is incomplete or reports a failing stage
#  13  the diagnostic summary could not be produced (e.g. host telemetry lost)
#  14  the host daemon exited non-zero on its own
# 130  interrupted by SIGINT; 143 by SIGTERM
#
# One GPU task at a time: the script takes /tmp/victory-gpu.lock.
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
START_DIR="$(pwd)"
# Homebrew's rustup is keg-only; put it first like the other macOS harnesses.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

usage() {
    cat <<'USAGE'
usage: run-title-metal.sh [--window|--headless] [--frames N] [--out DIR] [--keep] -- <exe> [guest args...]

  --window      open the native window and present the guest's surface
  --headless    force headless presentation (default); conflicts with --window
  --frames N    diagnostic frame count (exported as VICTORY_FRAME_MEASURED)
  --out DIR     keep guest.log, host.log, telemetry.json and summary.txt here
  --keep        keep the isolated session directory (share dir, wineprefix)
  -h, --help
USAGE
}

WINDOW=0
WINDOW_SET=0
HEADLESS_SET=0
FRAMES=""
OUT=""
KEEP=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --window)
            WINDOW=1
            WINDOW_SET=1
            shift
            ;;
        --headless)
            WINDOW=0
            HEADLESS_SET=1
            shift
            ;;
        --frames)
            [ "$#" -ge 2 ] || { echo "run-title-metal.sh: --frames requires a value" >&2; exit 2; }
            FRAMES="$2"
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || { echo "run-title-metal.sh: --out requires a value" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --keep)
            KEEP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "run-title-metal.sh: unknown option $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            echo "run-title-metal.sh: expected -- before the guest executable" >&2
            exit 2
            ;;
    esac
done

if [ "$WINDOW_SET" = 1 ] && [ "$HEADLESS_SET" = 1 ]; then
    echo "run-title-metal.sh: --window conflicts with --headless" >&2
    exit 2
fi
if [ "$#" -lt 1 ]; then
    echo "run-title-metal.sh: missing -- <exe> [guest args...]" >&2
    usage >&2
    exit 2
fi
case "$FRAMES" in
    "") ;;
    *[!0-9]*) echo "run-title-metal.sh: --frames must be a positive integer" >&2; exit 2 ;;
    0) echo "run-title-metal.sh: --frames must be a positive integer" >&2; exit 2 ;;
esac

GUEST_EXE="$1"
shift
# The remaining positional parameters are the guest arguments and stay intact
# for `launch_guest`; functions get their own positional-parameter copy.
case "$GUEST_EXE" in
    */*)
        [ -f "$GUEST_EXE" ] || { echo "run-title-metal.sh: guest executable not found: $GUEST_EXE" >&2; exit 2; }
        GUEST_EXE="$(cd "$(dirname "$GUEST_EXE")" && pwd)/$(basename "$GUEST_EXE")"
        ;;
esac

cd "$ROOT"

if [ -n "$OUT" ]; then
    case "$OUT" in
        /*) ;;
        *) OUT="$START_DIR/$OUT" ;;
    esac
    mkdir -p "$OUT" || { echo "run-title-metal.sh: cannot create --out $OUT" >&2; exit 2; }
else
    OUT="$(mktemp -d "${TMPDIR:-/tmp}/victory-title-out.XXXXXXXX")"
fi
OUT="$(cd "$OUT" && pwd)"

if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$ROOT/../../../VIctory-macOS/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        if [ -x "$wine" ]; then
            WINE="$wine"
            break
        fi
    done
fi
if [ -z "${WINE:-}" ]; then
    WINE="$(command -v wine || true)"
fi
export WINE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-title.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
FENCE_PAGE="$SHARE/gpu-fence.v1.page"
TELEMETRY="$OUT/telemetry.json"
HOST_LOG="$OUT/host.log"
GUEST_LOG="$OUT/guest.log"
SUMMARY="$OUT/summary.txt"
GUEST_REPORT="$SHARE/live-gate.report"
FRAMES_CSV="$SHARE/live-gate.frames.csv"
HOST_PID=""
GUEST_PID=""
HOST_RC=0
GUEST_RC=0
GUEST_TIMED_OUT=0
HOST_TERMINATED=0
HOST_READY_LINE=""
PORT=""
MODE="title"
GATE_MODE=0
POOL_STATUS="skipped"
POOL_SUMMARY=""
ABI_STATUS="skipped"
ABI_SUMMARY=""
HOST_STARTED=0
LOCK="/tmp/victory-gpu.lock"
LOCK_HELD=0

fail_setup() {
    echo "run-title-metal.sh: FAIL: $1" >&2
    exit 3
}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ -n "$GUEST_PID" ] && kill -0 "$GUEST_PID" 2>/dev/null; then
        kill "$GUEST_PID" 2>/dev/null || true
        wait "$GUEST_PID" 2>/dev/null || true
    fi
    if [ -n "$HOST_PID" ] && kill -0 "$HOST_PID" 2>/dev/null; then
        kill "$HOST_PID" 2>/dev/null || true
        wait "$HOST_PID" 2>/dev/null || true
    fi
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    sleep 0.3
    leaked="$(pgrep -f "$WORK" 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "$leaked" ]; then
        echo "run-title-metal.sh: PROCESS_LEAK remaining=\"$leaked\"" >&2
        for pid in $leaked; do kill -9 "$pid" 2>/dev/null || true; done
        sleep 0.2
        leaked="$(pgrep -f "$WORK" 2>/dev/null | tr '\n' ' ' || true)"
    fi
    echo "PROCESS_CHECK host=0 guest=0 leaks=${leaked:-0}"
    if [ "$KEEP" = 1 ]; then
        echo "run-title-metal.sh: kept session files in $WORK"
        echo "run-title-metal.sh: diagnostics in $OUT"
    else
        rm -rf "$WORK"
    fi
    if [ "$LOCK_HELD" = 1 ]; then
        rmdir "$LOCK" 2>/dev/null || true
        LOCK_HELD=0
    fi
    if [ "$status" -eq 0 ] && [ -n "$leaked" ]; then
        exit 4
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- Serialise exclusive GPU work ------------------------------------------
acquired=0
while [ "$acquired" -eq 0 ]; do
    if mkdir "$LOCK" 2>/dev/null; then
        acquired=1
        LOCK_HELD=1
    else
        if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then
            rmdir "$LOCK" 2>/dev/null || true
        fi
        sleep 0.5
    fi
done

mkdir -m 700 "$SHARE"
if ! command -v openssl >/dev/null 2>&1; then
    fail_setup "openssl is required for the fresh host secret"
fi
openssl rand -hex 32 >"$SECRET" || fail_setup "cannot generate the host secret"
chmod 600 "$SECRET"
SECRET_HEX="$(cat "$SECRET")"

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

if [ -z "${WINE:-}" ] || [ ! -x "$WINE" ]; then
    fail_setup "no Wine executable found (set WINE=...)"
fi

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
case "$TARGET_DIR" in
    /*) ;;
    *) TARGET_DIR="$ROOT/$TARGET_DIR" ;;
esac
HOST_BIN="$TARGET_DIR/aarch64-apple-darwin/debug/victory_metal_host"

if [ ! -x "$HOST_BIN" ]; then
    echo "run-title-metal.sh: building the ARM64 victory_metal_host (missing $HOST_BIN)..."
    if ! cargo build --locked -p victory-metal --target aarch64-apple-darwin --bin victory_metal_host; then
        fail_setup "the ARM64 host daemon did not build"
    fi
fi
[ -x "$HOST_BIN" ] || fail_setup "missing $HOST_BIN after the build"

# --- Gate detection --------------------------------------------------------
GATE_MODE=0
case "$GUEST_EXE" in
    *live_d3d12_gate*) GATE_MODE=1 ;;
esac
for argument in "$@"; do
    if [ "$argument" = "live_d3d12_guest_gate" ]; then
        GATE_MODE=1
    fi
done
if [ "${VICTORY_LIVE_GATE:-}" = "1" ]; then
    GATE_MODE=1
fi
if [ "$GATE_MODE" = 1 ]; then
    MODE="gate"
fi

# --- Prepare the DXIL pool and reflection ABIs (gate guest only) -----------
# Adapted from tools/macos/metal-poc/live-d3d12-gate.sh so the gate can be
# launched through this single entry point. Generic titles skip this entirely.
METALLIB_DIR="$WORK/metallib"
HLSL_DIR="$WORK/hlsl"
DXIL_DIR="$WORK/dxil"
ABI_DIR="$WORK/abi"
SHARE_PIPE="$SHARE/pipelines"
IRCONV="$TARGET_DIR/x86_64-apple-darwin/debug/victory-irconverter"
CONV="${VICTORY_METAL_IRCONVERTER:-$HOME/Library/Caches/victory-macos/target/wine-cx/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib}"

find_tool() {
    # $1 = relative path under a candidate root
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

write_hlsl() {
    mkdir -p "$HLSL_DIR" "$DXIL_DIR" "$METALLIB_DIR"
    cat >"$HLSL_DIR/vs.hlsl" <<'EOS'
struct VSOut { float4 pos : SV_Position; };
[RootSignature("RootFlags(0)")]
VSOut main(uint vid : SV_VertexID) {
    float2 c[3] = { float2(-1.0,-1.0), float2(3.0,-1.0), float2(-1.0,3.0) };
    VSOut o; o.pos = float4(c[vid], 0.0, 1.0); return o;
}
EOS
    cat >"$HLSL_DIR/ps.hlsl" <<'EOS'
struct PSOut { float4 c0 : SV_Target0; float4 c1 : SV_Target1; };
[RootSignature("RootFlags(0)")]
PSOut main() {
    PSOut o;
    o.c0 = float4(1.0, 128.0/255.0, 0.0, 1.0);
    o.c1 = float4(0.0, 128.0/255.0, 1.0, 1.0);
    return o;
}
EOS
    cat >"$HLSL_DIR/psm.hlsl" <<'EOS'
[RootSignature("RootFlags(0)")]
float4 main() : SV_Target { return float4(0.0, 1.0, 1.0, 1.0); }
EOS
    cat >"$HLSL_DIR/cs.hlsl" <<'EOS'
// The culling kernel writes the indirect draw record the following
// ExecuteIndirect reads; the Wave64 values keep the console wave intrinsics.
RWStructuredBuffer<uint> gArgs : register(u0);
[RootSignature("RootFlags(0), UAV(u0)")]
[numthreads(64,1,1)]
void main(uint tid : SV_DispatchThreadID) {
    uint lanes = WaveGetLaneCount();
    uint sum = WaveActiveSum(tid + 1u);
    if (tid == 0u) {
        gArgs[0] = 3u;
        gArgs[1] = 1u;
        gArgs[2] = 0u;
        gArgs[3] = 0u;
    }
    if (sum == 0xdeadbeefu && lanes == 12345u) {
        gArgs[3] = 0xdeadbeefu;
    }
}
EOS
    # A mesh shader whose used constant table pushes the Mesh+Pixel pair well
    # over Metal's 32 KiB mesh-output budget class.
    python3 - "$HLSL_DIR/ms.hlsl" <<'PY'
import sys
n = 8192
vals = ",".join(str((i * 2654435761) & 0xffffffff) for i in range(n))
src = f'''struct VOut {{ float4 pos : SV_Position; }};
static const uint gTable[{n}] = {{{vals}}};
[outputtopology("triangle")]
[numthreads(32,1,1)]
[RootSignature("RootFlags(0)")]
void main(out indices uint3 tri[1], out vertices VOut verts[3], uint gid : SV_GroupID) {{
    SetMeshOutputCounts(3, 1);
    uint acc = gTable[gid % {n}] ^ gTable[(gid * 7u + 1u) % {n}];
    float2 c[3] = {{ float2(-1.0,-1.0), float2(3.0,-1.0), float2(-1.0,3.0) }};
    for (uint i = 0; i < 3; i++) {{
        verts[i].pos = float4(c[i] + float(acc & 1u), 0.0, 1.0);
    }}
    tri[0] = uint3(0, 1, 2);
}}
'''
open(sys.argv[1], "w").write(src)
PY
}

compile_dxil() {
    # $1 = source, $2 = profile, $3 = entry, $4 = output stem
    src_win="$("$WINE" winepath -w "$HLSL_DIR/$1" 2>/dev/null | tr -d '\r\n')"
    out_win="$("$WINE" winepath -w "$DXIL_DIR/$4.dxil" 2>/dev/null | tr -d '\r\n')"
    ( cd "$DXC_DIR" && "$WINE" ./dxc.exe -T "$2" -E "$3" -Od -Fo "$out_win" "$src_win" ) \
        >"$WORK/dxc-$4.log" 2>&1 || true
    if [ ! -s "$DXIL_DIR/$4.dxil" ]; then
        echo "DXC_ERROR stem=$4 profile=$2 entry=$3"
        tail -5 "$WORK/dxc-$4.log" >&2 || true
        return 1
    fi
    return 0
}

pool_stage() {
    POOL_STATUS="unavailable"
    [ -f "$DXC_EXE" ] || { echo "POOL_UNAVAILABLE dxc.exe missing"; return 0; }
    [ -f "$CONV" ] || { echo "POOL_UNAVAILABLE libmetalirconverter.dylib missing"; return 0; }
    if [ ! -x "$IRCONV" ]; then
        echo "run-title-metal.sh: building the x86_64 victory-irconverter worker..."
        if ! cargo build --locked -p victory-metal --target x86_64-apple-darwin --bin victory-irconverter; then
            echo "POOL_UNAVAILABLE victory-irconverter did not build"
            return 0
        fi
    fi
    [ -x "$IRCONV" ] || { echo "POOL_UNAVAILABLE victory-irconverter missing"; return 0; }
    write_hlsl
    compile_dxil vs.hlsl vs_6_0 main vs || return 0
    compile_dxil ps.hlsl ps_6_0 main ps || return 0
    compile_dxil ms.hlsl ms_6_5 main ms || return 0
    compile_dxil psm.hlsl ps_6_0 main psm || return 0
    compile_dxil cs.hlsl cs_6_6 main cs || return 0

    if ! python3 - "$DXIL_DIR" "$IRCONV" "$CONV" "$METALLIB_DIR" <<'PY'
import json
import os
import subprocess
import sys
import time

dxil_dir, worker, converter, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
jobs = [("vs", 1, "main"), ("ps", 2, "main"), ("ms", 5, "main"),
        ("psm", 2, "main"), ("cs", 8, "main")]
env = dict(os.environ)
env["VICTORY_IRCONVERTER_STDIO_REFLECTION"] = "1"
proc = subprocess.Popen([worker, "--stdio", "--converter", converter],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, env=env)
timings = []
written = {}
reflections = {}
started = time.time()
for index, (name, stage, entry) in enumerate(jobs):
    path = os.path.join(dxil_dir, name + ".dxil")
    dxil = open(path, "rb").read()
    header = f"VMJOB {index + 1} {1000 + index} {stage} {len(entry)} {len(dxil)}\n".encode()
    began = time.time()
    proc.stdin.write(header + entry.encode() + dxil)
    proc.stdin.flush()
    line = proc.stdout.readline()
    parts = line.split()
    if not parts or parts[0] != b"VMRES2":
        print(f"POOL_ERROR {name} protocol={line[:80]!r}")
        proc.kill()
        sys.exit(3)
    status = int(parts[3])
    metallib_len = int(parts[4])
    reflection_len = int(parts[5])
    metallib = proc.stdout.read(metallib_len)
    reflection = proc.stdout.read(reflection_len)
    elapsed = (time.time() - began) * 1000.0
    timings.append((name, elapsed, status, len(metallib)))
    if status != 0 or not metallib.startswith(b"MTLB"):
        print(f"POOL_ERROR {name} status={status} message={metallib[:160]!r}")
        proc.kill()
        sys.exit(3)
    if not reflection:
        print(f"POOL_ERROR {name} reflection JSON is empty")
        proc.kill()
        sys.exit(3)
    try:
        parsed = json.loads(reflection)
    except ValueError as error:
        print(f"POOL_ERROR {name} reflection JSON does not parse: {error}")
        proc.kill()
        sys.exit(3)
    if "TopLevelArgumentBuffer" not in parsed:
        print(f"POOL_ERROR {name} reflection has no TopLevelArgumentBuffer")
        proc.kill()
        sys.exit(3)
    written[name] = os.path.join(out_dir, name + ".metallib")
    open(written[name], "wb").write(metallib)
    reflection_path = os.path.join(out_dir, name + ".reflection.json")
    open(reflection_path, "wb").write(reflection)
    reflections[name] = {
        "bytes": len(reflection),
        "path": reflection_path,
        "params": len(parsed.get("TopLevelArgumentBuffer", [])),
        "entry": parsed.get("EntryPoint", ""),
    }
proc.stdin.close()
proc.wait()
total_s = max(time.time() - started, 1e-9)
jobs_per_s = len(jobs) / total_s
first_ms = timings[0][1]
warm = [t for _, t, _, _ in timings[1:]]
warm_avg = sum(warm) / len(warm) if warm else 0.0
summary = {
    "jobs": len(jobs),
    "seconds": round(total_s, 3),
    "jobs_per_s": round(jobs_per_s, 2),
    "first_job_ms": round(first_ms, 3),
    "subsequent_avg_ms": round(warm_avg, 3),
    "mesh_pair_bytes": os.path.getsize(written["ms"]) + os.path.getsize(written["psm"]),
    "metallib_bytes": {k: os.path.getsize(v) for k, v in written.items()},
    "reflection": reflections,
    "reflection_bytes": {k: v["bytes"] for k, v in reflections.items()},
}
print("POOL_SUMMARY " + json.dumps(summary))
PY
    then
        echo "POOL_UNAVAILABLE the --stdio worker refused a job (see above)"
        POOL_STATUS="failed"
        return 0
    fi
    POOL_STATUS="translated"
}

build_abis() {
    mkdir -p "$ABI_DIR" "$WORK/abi-helper/src" "$SHARE_PIPE"
    cat >"$WORK/abi-helper/Cargo.toml" <<EOS
[package]
name = "gate-abi"
version = "0.0.0"
edition = "2021"

[workspace]

[dependencies]
victory-metal = { path = "$ROOT/crates/victory-metal" }
EOS
    cat >"$WORK/abi-helper/src/main.rs" <<'EOS'
fn main() {
    let mut groups: Vec<Vec<String>> = vec![Vec::new()];
    for argument in std::env::args().skip(1) {
        if argument == "--" {
            groups.push(Vec::new());
        } else {
            groups.last_mut().expect("a group").push(argument);
        }
    }
    for group in groups.iter().filter(|group| !group.is_empty()) {
        let out = &group[0];
        let mut stages: Vec<(u8, String)> = Vec::new();
        let mut index = 1;
        while index + 1 < group.len() {
            let stage: u8 = group[index].parse().expect("wire stage code");
            let path = &group[index + 1];
            let json = std::fs::read_to_string(path).expect("read reflection JSON");
            let layout =
                victory_metal::macos::bindings::RootSignatureLayout::from_converter_reflection_json(
                    &json,
                )
                .unwrap_or_else(|error| panic!("{path}: {error}"));
            eprintln!(
                "ABI_STAGE path={path} stage={stage} params={} argbuf={} entry={}",
                layout.parameters().len(),
                layout.argument_buffer_len(),
                layout.entry_point()
            );
            stages.push((stage, json));
            index += 2;
        }
        let mut abi = Vec::with_capacity(6 + stages.len() * 8);
        abi.extend_from_slice(b"VAB1");
        abi.push(1);
        abi.push(u8::try_from(stages.len()).expect("stage count fits u8"));
        for (stage, json) in &stages {
            abi.push(*stage);
            abi.extend_from_slice(
                &u32::try_from(json.len())
                    .expect("reflection JSON length fits u32")
                    .to_le_bytes(),
            );
            abi.extend_from_slice(json.as_bytes());
        }
        std::fs::write(out, &abi).expect("write VAB1");
        eprintln!("ABI_WRITTEN path={out} bytes={} stages={}", abi.len(), stages.len());
    }
}
EOS
    if [ "$POOL_STATUS" = "translated" ]; then
        echo "run-title-metal.sh: encoding reflection-derived pipeline ABIs..."
        if CARGO_TARGET_DIR="$TARGET_DIR" cargo run --offline --quiet \
            --manifest-path "$WORK/abi-helper/Cargo.toml" --target aarch64-apple-darwin -- \
            "$ABI_DIR/graphics.vab1" 0 "$METALLIB_DIR/vs.reflection.json" \
                1 "$METALLIB_DIR/ps.reflection.json" -- \
            "$ABI_DIR/mesh.vab1" 2 "$METALLIB_DIR/ms.reflection.json" \
                1 "$METALLIB_DIR/psm.reflection.json" -- \
            "$ABI_DIR/compute.vab1" 4 "$METALLIB_DIR/cs.reflection.json" \
            >"$WORK/abi.log" 2>&1; then
            ABI_STATUS="built"
            ABI_SUMMARY="$(grep '^ABI_' "$WORK/abi.log" | tr '\n' ' ')"
        else
            ABI_STATUS="failed"
        fi
        cat "$WORK/abi.log" || true
    fi
    if [ "$POOL_STATUS" = "translated" ] && [ "$ABI_STATUS" = "built" ]; then
        cp "$METALLIB_DIR"/*.metallib "$METALLIB_DIR"/*.reflection.json "$ABI_DIR"/*.vab1 "$SHARE_PIPE"/
    fi
}

if [ "$GATE_MODE" = 1 ]; then
    echo "run-title-metal.sh: preparing the gate DXIL pool and reflection ABIs..."
    pool_stage >"$WORK/pool.log" 2>&1 || true
    cat "$WORK/pool.log"
    POOL_SUMMARY="$(sed -n 's/^POOL_SUMMARY //p' "$WORK/pool.log" | tail -1)"
    build_abis
fi

# --- Start the ARM64 daemon ------------------------------------------------
start_host() {
    set -- --secret-file "$SECRET" --port 0 --once --share-dir "$SHARE" \
        --telemetry-json "$TELEMETRY"
    if [ "$WINDOW" = 1 ]; then
        set -- "$@" --window
    else
        set -- "$@" --headless
    fi
    if [ "$GATE_MODE" = 1 ]; then
        set -- "$@" --aaa-frame
    fi
    : >"$HOST_LOG"
    "$HOST_BIN" "$@" >"$HOST_LOG" 2>&1 &
    HOST_PID=$!
    HOST_STARTED=1
}

echo "run-title-metal.sh: starting the ARM64 victory-metal-host daemon..."
start_host

ready=0
waited=0
while [ "$waited" -lt 600 ]; do
    if grep -q '^victory-metal-host: ready port=' "$HOST_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$HOST_PID" 2>/dev/null; then
        break
    fi
    sleep 0.1
    waited=$((waited + 1))
done
if [ "$ready" -ne 1 ]; then
    echo "run-title-metal.sh: FAIL: the host daemon did not become ready (pid $HOST_PID)" >&2
    cat "$HOST_LOG" >&2 || true
    exit 3
fi
HOST_READY_LINE="$(grep '^victory-metal-host: ready port=' "$HOST_LOG" | tail -1)"
PORT="$(printf '%s\n' "$HOST_READY_LINE" | sed -n 's/^victory-metal-host: ready port=\([0-9][0-9]*\).*$/\1/p')"
[ -n "$PORT" ] || fail_setup "the host did not publish its port"
WINDOW_SIZE="$(printf '%s\n' "$HOST_READY_LINE" | sed -n 's/.* window=\([0-9][0-9]*x[0-9][0-9]*\).*/\1/p')"
echo "run-title-metal.sh: host ready: $HOST_READY_LINE"

compute_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.compute.id" 2>/dev/null | tail -1)"
graphics_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.graphics.id" 2>/dev/null | tail -1)"

# --- Guest environment -----------------------------------------------------
export WINEPREFIX="$WORK/wineprefix"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"

# Host-owned connection values always win: they describe this exact daemon.
export VICTORY_BACKEND=metal
export VICTORY_METAL_PORT="$PORT"
export VICTORY_METAL_ENDPOINT="127.0.0.1:$PORT"
export VICTORY_METAL_SECRET="$SECRET_HEX"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_METAL_FENCE_PAGE="$(winpath "$FENCE_PAGE")"
export VICTORY_FRAME_MEASURED="${FRAMES:-${VICTORY_FRAME_MEASURED:-120}}"
export VICTORY_FRAME_WARMUP="${VICTORY_FRAME_WARMUP:-20}"

if [ "$GATE_MODE" = 1 ]; then
    export VICTORY_LIVE_GATE=1
    export VICTORY_LIVE_GATE_OUT="$(winpath "$GUEST_REPORT")"
    export VICTORY_LIVE_GATE_CSV="$(winpath "$FRAMES_CSV")"
    export VICTORY_LIVE_GATE_GRAPHICS_ID="${graphics_id:-0}"
    export VICTORY_LIVE_GATE_COMPUTE_ID="${compute_id:-0}"
    export VICTORY_LIVE_GATE_METALLIB_VS="$(winpath "$SHARE_PIPE/vs.metallib")"
    export VICTORY_LIVE_GATE_METALLIB_PS="$(winpath "$SHARE_PIPE/ps.metallib")"
    export VICTORY_LIVE_GATE_METALLIB_MS="$(winpath "$SHARE_PIPE/ms.metallib")"
    export VICTORY_LIVE_GATE_METALLIB_PSM="$(winpath "$SHARE_PIPE/psm.metallib")"
    export VICTORY_LIVE_GATE_METALLIB_CS="$(winpath "$SHARE_PIPE/cs.metallib")"
    export VICTORY_LIVE_GATE_ABI_GRAPHICS="$(winpath "$SHARE_PIPE/graphics.vab1")"
    export VICTORY_LIVE_GATE_ABI_MESH="$(winpath "$SHARE_PIPE/mesh.vab1")"
    export VICTORY_LIVE_GATE_ABI_COMPUTE="$(winpath "$SHARE_PIPE/compute.vab1")"
    export VICTORY_LIVE_GATE_REFLECTION_GRAPHICS="$(winpath "$SHARE_PIPE/vs.reflection.json")"
    export VICTORY_LIVE_GATE_REFLECTION_MESH="$(winpath "$SHARE_PIPE/ms.reflection.json")"
    export VICTORY_LIVE_GATE_REFLECTION_COMPUTE="$(winpath "$SHARE_PIPE/cs.reflection.json")"
    # The gate expects the 1280x720 present surface it records in its report.
    export VICTORY_METAL_SURFACE="${VICTORY_METAL_SURFACE:-1}"
    export VICTORY_METAL_PRESENT_TEXTURE="${VICTORY_METAL_PRESENT_TEXTURE:-4096}"
    export VICTORY_METAL_SURFACE_WIDTH="${VICTORY_METAL_SURFACE_WIDTH:-1280}"
    export VICTORY_METAL_SURFACE_HEIGHT="${VICTORY_METAL_SURFACE_HEIGHT:-720}"
fi

# --- Run the Win64 guest under Wine ----------------------------------------
echo "run-title-metal.sh: running the Win64 guest under Wine (mode=$MODE, frames=$VICTORY_FRAME_MEASURED, warmup=$VICTORY_FRAME_WARMUP)..."
GUEST_TIMEOUT="${GUEST_TIMEOUT:-900}"
# Capture the guest output to guest.log and echo it afterwards: POSIX sh has no
# pipefail, so a `| tee` pipeline would hide the guest exit status.
"$WINE" "$GUEST_EXE" "$@" >"$GUEST_LOG" 2>&1 &
GUEST_PID=$!
guest_deadline=$(( $(date +%s) + GUEST_TIMEOUT ))
while kill -0 "$GUEST_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$guest_deadline" ]; then
        echo "run-title-metal.sh: WARN: the Wine guest did not finish within ${GUEST_TIMEOUT}s; stopping it" >&2
        kill "$GUEST_PID" 2>/dev/null || true
        GUEST_TIMED_OUT=1
        break
    fi
    sleep 0.2
done
if wait "$GUEST_PID"; then
    GUEST_RC=0
else
    GUEST_RC=$?
fi
GUEST_PID=""
cat "$GUEST_LOG" 2>/dev/null || true

# --- Let the daemon finish its `--once` session, then stop it --------------
host_deadline=$(( $(date +%s) + 20 ))
while kill -0 "$HOST_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$host_deadline" ]; then
        echo "run-title-metal.sh: WARN: the host did not exit after the guest; stopping it" >&2
        kill "$HOST_PID" 2>/dev/null || true
        HOST_TERMINATED=1
        break
    fi
    sleep 0.1
done
if wait "$HOST_PID"; then
    HOST_RC=0
else
    HOST_RC=$?
fi
HOST_PID=""

# --- Structured summary ----------------------------------------------------
SUMMARY_RC=0
python3 - "$FRAMES_CSV" "$GUEST_REPORT" "$TELEMETRY" "$GUEST_RC" "$HOST_RC" \
    "$MODE" "$WINDOW" "$WINDOW_SIZE" "$FRAMES" "$SUMMARY" "$HOST_TERMINATED" \
    "$GUEST_TIMED_OUT" "$HOST_READY_LINE" "$GUEST_LOG" "$POOL_STATUS" \
    "$ABI_STATUS" "$graphics_id" "$compute_id" <<'PY' || SUMMARY_RC=$?
import json
import math
import os
import sys

(csv_path, report_path, telemetry_path, guest_rc, host_rc, mode, window,
 window_size, frames_cli, summary_path, host_terminated, guest_timed_out,
 ready_line, guest_log, pool_status, abi_status, graphics_id,
 compute_id) = sys.argv[1:19]
guest_rc = int(guest_rc)
host_rc = int(host_rc)
host_terminated = host_terminated == "1"
guest_timed_out = guest_timed_out == "1"

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
telemetry_ok = False
if os.path.exists(telemetry_path):
    try:
        with open(telemetry_path) as handle:
            telemetry = json.load(handle)
        telemetry_ok = True
    except (OSError, json.JSONDecodeError):
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

try:
    expected = int(frames_cli) if frames_cli else int(report.get("measured_expected", 0))
except ValueError:
    expected = 0

host_fallback = telemetry.get("d3dmetal_fallback_count", None)
guest_fallback = report.get("d3dmetal_fallback_count", None)
guest_module = report.get("guest_d3dmetal_module", None)
host_loaded = telemetry.get("d3dmetal_loaded", None)
host_images = telemetry.get("d3dmetal_images", []) or []
unsupported_host = telemetry.get("unsupported_opcodes", {}) or {}
unsupported_guest = report.get("unsupported_opcodes", "") or ""
log_mentions = 0
if os.path.exists(guest_log):
    try:
        with open(guest_log, "r", errors="replace") as handle:
            log_mentions = handle.read().lower().count("d3dmetal")
    except OSError:
        log_mentions = 0

def one_line(value):
    return str(value).replace("\n", " ").replace("\r", " ").strip()

problem = None
if guest_timed_out:
    problem = (10, "guest exceeded the launcher timeout")
elif guest_rc != 0:
    problem = (10, f"guest exited {guest_rc}")
elif not telemetry_ok:
    problem = (13, "host telemetry missing; cannot prove d3dmetal_fallback_count=0")
elif host_fallback not in (0, "0"):
    problem = (11, f"host d3dmetal_fallback_count={host_fallback}")
elif guest_fallback is not None and guest_fallback not in (0, "0"):
    problem = (11, f"guest d3dmetal_fallback_count={guest_fallback}")
elif guest_module is not None and guest_module not in (0, "0"):
    problem = (11, f"guest d3dmetal_module={guest_module}")
elif host_loaded not in (False, 0):
    problem = (11, f"host d3dmetal_loaded={host_loaded}")
elif host_images:
    problem = (11, "host d3dmetal_images=" + ",".join(map(str, host_images)))
elif host_rc != 0 and not host_terminated:
    problem = (14, f"host exited {host_rc}")
elif mode == "gate":
    if report.get("ok") != "1":
        reason = report.get("reason", "report missing or incomplete")
        problem = (12, "gate report ok=" + str(report.get("ok", "missing")) + " " + one_line(reason))
    elif count != expected:
        problem = (12, f"gate measured {count} of {expected} frames")

status = "PASS" if problem is None else "FAIL"
exit_code = 0 if problem is None else problem[0]
reason = "clean" if problem is None else one_line(problem[1])

window_value = window_size if (window == "1" and window_size) else ("none" if window != "1" else "unknown")

lines = [
    "launcher=run-title-metal",
    f"status={status}",
    f"exit_code={exit_code}",
    f"reason={reason}",
    f"mode={mode}",
    f"headless={0 if window == '1' else 1}",
    f"window={window_value}",
    f"host_ready_line={one_line(ready_line)}",
    f"graphics_pipeline_id={graphics_id or 'none'}",
    f"compute_pipeline_id={compute_id or 'none'}",
    f"guest_exit={guest_rc}",
    f"host_exit={host_rc}",
    f"host_terminated_by_launcher={1 if host_terminated else 0}",
    f"frames={count}",
    f"frames_expected={expected}",
    f"p50_ms={p50:.3f}",
    f"p95_ms={p95:.3f}",
    f"p99_ms={p99:.3f}",
    f"fps={fps:.2f}",
    f"guest_frames_ok={report.get('frames_ok', 'n/a')}",
    f"guest_fence_waits={report.get('fence_waits', 'n/a')}",
    f"guest_fence_timeouts={report.get('fence_timeouts', 'n/a')}",
    f"guest_shader_psos_registered={report.get('psos_registered', 'n/a')}",
    f"guest_submits_ok={report.get('submits_ok', 'n/a')}",
    f"guest_submits_refused={report.get('submits_refused', 'n/a')}",
    f"guest_presents_ok={report.get('presents_ok', 'n/a')}",
    f"guest_presents_refused={report.get('presents_refused', 'n/a')}",
    f"guest_registration_failures={report.get('registration_failures', 'n/a')}",
    f"guest_graphics_pipeline={one_line(report.get('graphics_pipeline', 'n/a'))}",
    f"guest_mesh_pipeline={one_line(report.get('mesh_pipeline', 'n/a'))}",
    f"guest_compute_pipeline={one_line(report.get('compute_pipeline', 'n/a'))}",
    f"guest_pipeline_ids_used={one_line(report.get('pipeline_ids_used', 'n/a'))}",
    f"host_refused_submissions={telemetry.get('refused_submissions', 'n/a')}",
    f"d3dmetal_fallback_count={host_fallback if host_fallback is not None else 'n/a'}",
    f"guest_d3dmetal_fallback_count={guest_fallback if guest_fallback is not None else 'n/a'}",
    f"guest_d3dmetal_module={guest_module if guest_module is not None else 'n/a'}",
    f"d3dmetal_loaded={host_loaded if host_loaded is not None else 'n/a'}",
    f"d3dmetal_images={','.join(map(str, host_images)) if host_images else 'none'}",
    f"guest_log_d3dmetal_mentions={log_mentions}",
    f"unsupported_opcodes={unsupported_guest or 'none'}",
    f"host_unsupported_opcodes={json.dumps(unsupported_host, sort_keys=True)}",
    f"host_frames={telemetry.get('frames', 'n/a')}",
    f"host_p50_ms={telemetry.get('p50_ms', 'n/a')}",
    f"host_p95_ms={telemetry.get('p95_ms', 'n/a')}",
    f"host_p99_ms={telemetry.get('p99_ms', 'n/a')}",
    f"host_fps={telemetry.get('fps', 'n/a')}",
    f"present_checksum={telemetry.get('present_checksum', 'n/a')}",
    f"host_window_presented={telemetry.get('window_presented', 'n/a')}",
    f"host_window_drawables={telemetry.get('window_drawables', 'n/a')}",
    f"host_window_error={one_line(telemetry.get('window_error') or 'none')}",
    f"pool_status={pool_status}",
    f"abi_status={abi_status}",
    f"telemetry={telemetry_path}",
    f"guest_log={guest_log}",
]
text = "\n".join(lines) + "\n"
try:
    with open(summary_path, "w") as handle:
        handle.write(text)
except OSError as error:
    print(f"run-title-metal.sh: FAIL: cannot write summary: {error}", file=sys.stderr)
    sys.exit(13)
sys.stdout.write(text)
sys.exit(exit_code)
PY

if [ "$SUMMARY_RC" -ne 0 ]; then
    echo "run-title-metal.sh: FAIL (exit $SUMMARY_RC); summary: $SUMMARY"
    exit "$SUMMARY_RC"
fi
echo "run-title-metal.sh: PASS; summary: $SUMMARY"
exit 0
