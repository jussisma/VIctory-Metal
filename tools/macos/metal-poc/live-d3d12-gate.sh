#!/bin/sh
# W5-LIVE-D3D12-TITLE-GATE: the live Wine guest -> ARM64 Metal gate driven
# through the real `victory-gpu` D3D12 API path.
#
# Unlike `aaa-frame-bench.sh` (which hand-encodes Wire v3 frames in a C guest),
# this gate builds a Win64 Rust guest from `crates/victory-gpu/tests/
# live_d3d12_gate.rs`, which drives the production device/queue/fence entry
# points, the version 3 producer, the kind 4/129 pipeline registrations and the
# native presenter. The 120 measured frames (after 20 warmup) submit a Wave64
# compute dispatch, an `ExecuteIndirect` MRT+depth draw and a real Present over
# the authenticated loopback session, and observe completion only on the shared
# `gpu-fence.v1.page`.
#
# Stages, and the first one the merged code cannot express, are printed raw.
# A required stage the merged code refuses prints one PARTIAL line with that
# stage and its real error, and the script exits non-zero. Nothing is
# interpolated: if a stage fails the frames that actually ran are reported.
#
# Usage: tools/macos/metal-poc/live-d3d12-gate.sh
#   WINE=/path/to/wine          override the Wine binary
#   KEEP=1                      keep the session directory
#   VICTORY_FRAME_MEASURED=120  measured frames (default 120)
#   VICTORY_FRAME_WARMUP=20     excluded warmup frames (default 20)
#
# One GPU task at a time: the script takes /tmp/victory-gpu.lock.
#
# Process hygiene (C7 / Wave 7A): every long-lived child (host daemon, Wine
# guest, the --stdio IRConverter pool helper) is started in the background and
# registered with `process-hygiene.sh`. The EXIT trap reaps them with TERM, a
# bounded wait and then KILL for survivors, removes only the lock this run
# created, and exits 13 if any started PID is still alive afterwards.
#
# Exit codes:
#   0   PASS (all measured assertions held)
#   1   PARTIAL (a required stage failed; the first failing stage is named)
#   2   setup failure
#   13  a child started by this run survived cleanup (orphan)
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SELF_DIR/process-hygiene.sh"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; put it first like the other macOS harnesses.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

LOCK="/tmp/victory-gpu.lock"

if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$ROOT/../../../VIctory-macOS/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
export WINE

# The checked-out Microsoft dxc and Apple's x86_64 IRConverter.
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
IRCONV="$ROOT/target/x86_64-apple-darwin/debug/victory-irconverter"
HOST_BIN="$ROOT/target/aarch64-apple-darwin/debug/victory_metal_host"
CONV="${VICTORY_METAL_IRCONVERTER:-$HOME/Library/Caches/victory-macos/target/wine-cx/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-live-gate.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
HOST_OUT="$WORK/host.stdout"
HOST_ERR="$WORK/host.stderr"
GUEST_LOG="$WORK/guest.log"
TELEMETRY="$WORK/telemetry.json"
METALLIB_DIR="$WORK/metallib"
HLSL_DIR="$WORK/hlsl"
DXIL_DIR="$WORK/dxil"
FRAMES_CSV="$SHARE/live-gate.frames.csv"
GUEST_REPORT="$SHARE/live-gate.report"
HOST_PID=""
GUEST_PID=""
PORT=""
GUEST_RC=0
HOST_RC=0
POOL_STATUS="skipped"
POOL_SUMMARY=""
victory_hygiene_work "$WORK"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    # Reap tracked PIDs, then use the unique temp dir as an ownership proof for
    # grandchildren (the Python pool helper's IRConverter worker).
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

mkdir -m 700 "$SHARE" "$METALLIB_DIR" "$HLSL_DIR" "$DXIL_DIR"
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
[ -f "$DXC_EXE" ] || echo "WARN: no dxc.exe at $DXC_EXE; the DXIL pool stage will be reported unavailable" >&2
[ -f "$CONV" ] || echo "WARN: no libmetalirconverter.dylib at $CONV; the DXIL pool stage will be reported unavailable" >&2

# --- Build the ARM64 daemon and the x86_64 converter worker ----------------
echo "Building the ARM64 victory-metal-host daemon..."
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin victory_metal_host \
    || fail_setup "the ARM64 host daemon did not build"
[ -x "$HOST_BIN" ] || fail_setup "missing $HOST_BIN"
echo "Building the x86_64 victory-irconverter worker..."
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target x86_64-apple-darwin --bin victory-irconverter \
    || fail_setup "the x86_64 converter worker did not build"
[ -x "$IRCONV" ] || fail_setup "missing $IRCONV"

# --- DXIL pool stage: compile the gate's shaders and translate them --------
write_hlsl() {
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
// The culling kernel: it writes the indirect draw record the following
// ExecuteIndirect reads, so a frame without this write draws nothing. The
// Wave64 values are read (and feed a never-taken branch) so the translated
// kernel still carries the console wave intrinsics; the store itself must not
// depend on them.
//
// `UAV(u0)` declares the argument buffer as a *root* UAV, not a descriptor
// table entry: the IRConverter then reflects a top-level argument-buffer field
// that the guest's root view at slot 0 fills.
RWStructuredBuffer<uint> gArgs : register(u0);
[RootSignature("RootFlags(0), UAV(u0)")]
[numthreads(64,1,1)]
void main(uint tid : SV_DispatchThreadID) {
    uint lanes = WaveGetLaneCount();
    uint sum = WaveActiveSum(tid + 1u);
    if (tid == 0u) {
        gArgs[0] = 3u; // vertex count
        gArgs[1] = 1u; // instance count
        gArgs[2] = 0u; // start vertex
        gArgs[3] = 0u; // start instance
    }
    if (sum == 0xdeadbeefu && lanes == 12345u) {
        gArgs[3] = 0xdeadbeefu;
    }
}
EOS
    # A mesh shader whose used constant table pushes the Mesh+Pixel pair well
    # over Metal's 32 KiB mesh-output budget class. The table is read so the
    # compiler keeps it in the metallib.
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
        >"$WORK/dxc-$4.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$DXIL_DIR/$4.dxil" ]; then
        echo "DXC_ERROR stem=$4 profile=$2 entry=$3 rc=$rc"
        tail -5 "$WORK/dxc-$4.log" >&2 || true
        return 1
    fi
    return 0
}

pool_stage() {
    POOL_STATUS="unavailable"
    [ -f "$DXC_EXE" ] || { echo "POOL_UNAVAILABLE dxc.exe missing"; return 0; }
    [ -f "$CONV" ] || { echo "POOL_UNAVAILABLE libmetalirconverter.dylib missing"; return 0; }
    [ -x "$IRCONV" ] || { echo "POOL_UNAVAILABLE victory-irconverter missing"; return 0; }
    write_hlsl
    compile_dxil vs.hlsl vs_6_0 main vs || return 0
    compile_dxil ps.hlsl ps_6_0 main ps || return 0
    compile_dxil ms.hlsl ms_6_5 main ms || return 0
    compile_dxil psm.hlsl ps_6_0 main psm || return 0
    compile_dxil cs.hlsl cs_6_6 main cs || return 0

    cat >"$WORK/pool_worker.py" <<'PY'
import atexit
import json
import os
import signal
import subprocess
import sys
import time

dxil_dir, worker, converter, out_dir = sys.argv[1:5]
jobs = [("vs", 1, "main"), ("ps", 2, "main"), ("ms", 5, "main"),
        ("psm", 2, "main"), ("cs", 8, "main")]
# The pool sets this on every child; the gate drives the same worker directly
# and must ask for the same `VMRES2` frame (metallib + converter reflection).
env = dict(os.environ)
env["VICTORY_IRCONVERTER_STDIO_REFLECTION"] = "1"
proc = subprocess.Popen([worker, "--stdio", "--converter", converter],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, env=env)


def _reap_worker():
    # Reap the stdio worker on normal exit, protocol error, SIGTERM or SIGHUP.
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


atexit.register(_reap_worker)


def _on_signal(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGHUP, _on_signal)

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
    pool_worker_rc=0
    victory_hygiene_run python3 "$WORK/pool_worker.py" \
        "$DXIL_DIR" "$IRCONV" "$CONV" "$METALLIB_DIR" \
        >"$WORK/pool-worker.log" 2>&1 || pool_worker_rc=$?
    cat "$WORK/pool-worker.log" || true
    if [ "$pool_worker_rc" -ne 0 ]; then
        echo "POOL_UNAVAILABLE the --stdio worker refused a job (see above)"
        POOL_STATUS="failed"
        return 0
    fi
    POOL_STATUS="translated"
}

echo "Compiling and translating the gate's DXIL shaders through the --stdio pool..."
pool_stage >"$WORK/pool.log" 2>&1 || true
cat "$WORK/pool.log"
POOL_SUMMARY="$(sed -n 's/^POOL_SUMMARY //p' "$WORK/pool.log" | tail -1)"

# --- Build the reflection-derived VAB1 ABIs on the host --------------------
#
# `victory_metal::macos` (and therefore `RootSignatureLayout`) only exists on
# macOS, so the Win64 guest cannot parse converter reflection itself. The
# x86_64 worker above produced each stage's reflection JSON; this one-shot
# host helper parses it with `RootSignatureLayout::from_converter_reflection_json`
# (rejecting an unparseable layout) and encodes the same `VAB1` container
# `pso::encode_abi` would, which the guest then registers verbatim.
ABI_DIR="$WORK/abi"
SHARE_PIPE="$SHARE/pipelines"
ABI_STATUS="skipped"
ABI_SUMMARY=""
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
        // Same container `pso::encode_abi` writes: magic, version, count, then
        // per entry `stage u8 | json_len u32 LE | converter reflection JSON`.
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
    echo "Encoding reflection-derived pipeline ABIs on the host..."
    if CARGO_TARGET_DIR="$ROOT/target" cargo run --offline --quiet \
        --manifest-path "$WORK/abi-helper/Cargo.toml" --target aarch64-apple-darwin -- \
        "$ABI_DIR/graphics.vab1" 0 "$METALLIB_DIR/vs.reflection.json" \
            1 "$METALLIB_DIR/ps.reflection.json" -- \
        "$ABI_DIR/mesh.vab1" 2 "$METALLIB_DIR/ms.reflection.json" \
            1 "$METALLIB_DIR/psm.reflection.json" -- \
        "$ABI_DIR/compute.vab1" 4 "$METALLIB_DIR/cs.reflection.json" \
        >"$WORK/abi.log" 2>&1
    then
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

# Disk-cache cold-vs-warm is owned by the pool itself; reuse the existing
# measurement rather than inventing a second cache. It is reported separately.
POOL_CACHE_LINE=""
POOL_CACHE_ERROR=""
if [ "$POOL_STATUS" = "translated" ]; then
    echo "Measuring the pool's cold-vs-warm disk cache..."
    if VICTORY_IRCONVERTER_WORKER="$IRCONV" VICTORY_METAL_IRCONVERTER="$CONV" \
        victory_hygiene_run cargo test --locked --quiet -p victory-metal --target aarch64-apple-darwin \
        --test irconverter_stdio_gpu cold_and_warm_throughput_amortise_rosetta_startup \
        -- --nocapture --test-threads=1 >"$WORK/pool-cache.log" 2>&1; then
        POOL_CACHE_LINE="$(grep -E 'cache speedup:' "$WORK/pool-cache.log" | tail -1)"
        [ -n "$POOL_CACHE_LINE" ] || POOL_CACHE_ERROR="cache speedup line missing"
    else
        POOL_CACHE_ERROR="$(tail -3 "$WORK/pool-cache.log" | tr '\n' ' ')"
    fi
fi

# --- Build the Win64 guest harness -----------------------------------------
echo "Building the Win64 live-gate guest..."
GUEST_EXE="$(cargo test --locked --no-run -p victory-gpu \
    --target x86_64-pc-windows-gnullvm --test live_d3d12_gate \
    --message-format=json 2>"$WORK/guest-build.log" \
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
    echo "FAIL: the Win64 guest harness did not build" >&2
    tail -20 "$WORK/guest-build.log" >&2 || true
    exit 2
fi

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
compute_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.compute.id" 2>/dev/null | tail -1)"
graphics_id="$(sed -n 's/^id=\([0-9][0-9]*\)$/\1/p' "$SHARE/victory-metal-host.graphics.id" 2>/dev/null | tail -1)"

# --- Run the Win64 guest under Wine ----------------------------------------
export WINEPREFIX="$WORK/wineprefix"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export VICTORY_BACKEND=metal
export VICTORY_METAL_ENDPOINT="127.0.0.1:$PORT"
export VICTORY_METAL_SECRET="$SECRET_HEX"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_METAL_SURFACE="1"
export VICTORY_METAL_PRESENT_TEXTURE="4096"
export VICTORY_METAL_SURFACE_WIDTH="1280"
export VICTORY_METAL_SURFACE_HEIGHT="720"
export VICTORY_LIVE_GATE="1"
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
export VICTORY_FRAME_WARMUP="${VICTORY_FRAME_WARMUP:-20}"
export VICTORY_FRAME_MEASURED="${VICTORY_FRAME_MEASURED:-120}"
unset VICTORY_METAL_ENDPOINT_OLD 2>/dev/null || true

"$WINE" wineboot -i >/dev/null 2>&1 || true
echo "Host ready: port=$PORT compute_builtin=${compute_id:-0} graphics_builtin=${graphics_id:-0}"
echo "Running the Win64 live gate under Wine (warmup=$VICTORY_FRAME_WARMUP measured=$VICTORY_FRAME_MEASURED)..."
"$WINE" "$GUEST_EXE" --ignored --exact live_d3d12_guest_gate --test-threads=1 --nocapture \
    >"$GUEST_LOG" 2>&1 &
GUEST_PID=$!
victory_hygiene_track "$GUEST_PID"
guest_deadline=$(( $(date +%s) + 600 ))
while kill -0 "$GUEST_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$guest_deadline" ]; then
        echo "FAIL: the Wine guest did not finish within 600 s" >&2
        kill "$GUEST_PID" 2>/dev/null || true
        break
    fi
    sleep 0.2
done
GUEST_RC=0
victory_hygiene_wait_pid "$GUEST_PID" 10 || GUEST_RC=$?
victory_hygiene_untrack "$GUEST_PID"
GUEST_PID=""

host_deadline=$(( $(date +%s) + 120 ))
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

# --- Report measured numbers and assertions --------------------------------
PY_EXIT=0
python3 - "$FRAMES_CSV" "$GUEST_REPORT" "$TELEMETRY" "$POOL_STATUS" "$POOL_SUMMARY" "$POOL_CACHE_LINE" "$POOL_CACHE_ERROR" "$VICTORY_FRAME_WARMUP" "$VICTORY_FRAME_MEASURED" "$ABI_STATUS" "$ABI_SUMMARY" <<'PY' || PY_EXIT=$?
import json
import math
import os
import sys

csv_path, report_path, telemetry_path, pool_status, pool_summary, pool_cache, pool_cache_error, warmup, measured_expected, abi_status, abi_summary = sys.argv[1:12]
measured_expected = int(measured_expected)
EXPECTED_PRESENT = 0xd5791f96cdb32383

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

print(f"LIVE_GATE host_port=ok warmup={warmup} measured={count}/{measured_expected}")
print(f"LIVE_GATE p50_ms={p50:.3f} p95_ms={p95:.3f} p99_ms={p99:.3f} fps={fps:.2f}")
print(f"LIVE_GATE frames_ok={report.get('frames_ok','?')} fence_waits={report.get('fence_waits','?')} "
      f"fence_timeouts={report.get('fence_timeouts','?')} in_flight_observed={report.get('max_in_flight_observed','?')}")
print(f"LIVE_GATE host_telemetry frames={telemetry.get('frames','?')} "
      f"p50_ms={telemetry.get('p50_ms','?')} p95_ms={telemetry.get('p95_ms','?')} "
      f"p99_ms={telemetry.get('p99_ms','?')} fps={telemetry.get('fps','?')}")
print(f"LIVE_GATE present_checksum host={telemetry.get('present_checksum','?')} "
      f"expected={EXPECTED_PRESENT} expected_hex={EXPECTED_PRESENT:#x}")
print(f"LIVE_GATE replay status={report.get('replay','?')}")
print(f"LIVE_GATE d3dmetal guest_fallback={report.get('d3dmetal_fallback_count','?')} "
      f"guest_module={report.get('guest_d3dmetal_module','?')} "
      f"host_fallback={telemetry.get('d3dmetal_fallback_count','?')} "
      f"host_images={telemetry.get('d3dmetal_images','?')}")
print(f"LIVE_GATE unsupported_opcodes={report.get('unsupported_opcodes','?') or '(none)'}")
print(f"LIVE_GATE pool status={pool_status} summary={pool_summary or '-'}")
print(f"LIVE_GATE pool_cache line={pool_cache or '-'} error={pool_cache_error or '-'}")
print(f"LIVE_GATE abi status={abi_status} {abi_summary or '-'}")
print(f"LIVE_GATE stages compute={report.get('compute_pipeline','?')} "
      f"graphics={report.get('graphics_pipeline','?')} mesh={report.get('mesh_pipeline','?')}")
print(f"LIVE_GATE stage_owner graphics={report.get('graphics_pipeline_owner','?')} "
      f"builtin_compute={report.get('builtin_compute_id','?')} "
      f"builtin_graphics={report.get('builtin_graphics_id','?')}")
print(f"LIVE_GATE pipeline_ids used={report.get('pipeline_ids_used','?')} "
      f"mesh_used={report.get('mesh_pipeline_used','?')} "
      f"compute_probe={report.get('compute_probe_pipeline_id','?')}")
print(f"LIVE_GATE probes compute={report.get('compute_probe','?')} "
      f"mesh={report.get('mesh_dispatch','?')} linkage={report.get('compute_indirect_linkage','?')}")
print(f"LIVE_GATE linkage_frames={report.get('compute_indirect_linkage_frames','?')} "
      f"unbound_dispatches={report.get('compute_dispatches_without_bindings','?')} "
      f"args_after_frames={report.get('args_after_frames','?')}")

problems = []
stage_failure = report.get("first_failing_stage")

# Exact pixels: the host read back the registered present texture.
expected_hex = f"{EXPECTED_PRESENT:x}"
host_checksum = telemetry.get("present_checksum", -1)
checksum_ok = False
try:
    checksum_ok = int(host_checksum) == EXPECTED_PRESENT
except (TypeError, ValueError):
    checksum_ok = False
if not checksum_ok:
    problems.append(f"present checksum host={host_checksum} expected={expected_hex}")

# Exactly-once replay.
if "status=AlreadyExecuted" not in report.get("replay", "") or "unchanged=true" not in report.get("replay", ""):
    problems.append("replay probe: " + report.get("replay", "missing"))

# Zero D3DMetal, three independent methods.
for key, value in (
    ("guest d3dmetal_fallback_count", report.get("d3dmetal_fallback_count")),
    ("guest d3dmetal module", report.get("guest_d3dmetal_module")),
    ("host d3dmetal_fallback_count", telemetry.get("d3dmetal_fallback_count")),
):
    if value not in ("0", 0):
        problems.append(f"{key}={value}")
if telemetry.get("d3dmetal_loaded") not in (False, 0):
    problems.append("host loaded a D3DMetal image")
if telemetry.get("d3dmetal_images"):
    problems.append("host d3dmetal_images=" + str(telemetry.get("d3dmetal_images")))

# Clean opcode set.
if report.get("unsupported_opcodes", ""):
    problems.append("unsupported opcodes: " + report["unsupported_opcodes"])

# Fence completion observed before the next frame.
if report.get("fence_timeouts") not in ("0", 0):
    problems.append("fence timeouts=" + str(report.get("fence_timeouts")))
if str(report.get("fence_waits", "0")) != str(report.get("frames_ok", "-1")):
    problems.append(
        f"fence waits {report.get('fence_waits')} != frames_ok {report.get('frames_ok')}"
    )
if count != measured_expected:
    problems.append(f"measured {count} of {measured_expected} frames")

# Pool translation and the >32 KiB Mesh+Pixel pair.
pool_data = {}
if pool_summary:
    try:
        pool_data = json.loads(pool_summary)
    except json.JSONDecodeError:
        pool_data = {}
if pool_status != "translated":
    problems.append("DXIL pool stage " + pool_status)
else:
    if pool_data.get("mesh_pair_bytes", 0) <= 32768:
        problems.append(f"Mesh+Pixel pair {pool_data.get('mesh_pair_bytes')} <= 32768")
    if not pool_cache:
        problems.append("pool cache measurement unavailable: " + (pool_cache_error or "missing"))

# Reflection-derived ABIs: the launcher must have parsed every stage's
# converter reflection with `RootSignatureLayout` and produced a VAB1.
if abi_status != "built":
    problems.append("reflection ABI stage " + abi_status)

# Required stages from the merged code. A translation refusal is a real gap.
if not report.get("compute_pipeline", "").startswith("v2_kind=0"):
    problems.append("compute pipeline: " + report.get("compute_pipeline", "missing"))
if not report.get("graphics_pipeline", "").startswith("v2_kind=1"):
    problems.append("graphics pipeline: " + report.get("graphics_pipeline", "missing"))
if not report.get("mesh_pipeline", "").startswith("v2_kind=2"):
    problems.append("mesh pipeline: " + report.get("mesh_pipeline", "missing"))
if report.get("graphics_pipeline_fallback"):
    problems.append("graphics fell back to the host built-in: " + report["graphics_pipeline_fallback"])
if report.get("graphics_pipeline_owner", "guest_v2") != "guest_v2":
    problems.append("graphics pipeline owner=" + report.get("graphics_pipeline_owner", "?"))
if not report.get("compute_probe", "").startswith("Ok("):
    problems.append("compute probe: " + report.get("compute_probe", "missing"))
if not report.get("mesh_dispatch", "").startswith("Ok("):
    problems.append("mesh dispatch probe: " + report.get("mesh_dispatch", "missing"))

# The recorded streams must name the ids the guest's own registrations
# returned, not a daemon built-in.
def stage_id(value):
    for field in (value or "").split():
        if field.startswith("id="):
            try:
                return int(field[3:])
            except ValueError:
                return None
    return None


graphics = stage_id(report.get("graphics_pipeline"))
mesh = stage_id(report.get("mesh_pipeline"))
compute = stage_id(report.get("compute_pipeline"))
used = report.get("pipeline_ids_used", "")
if f"graphics={graphics}" not in used.split():
    problems.append(f"frames did not name the guest graphics id={graphics}: {used}")
if compute is not None and f"compute={compute}" not in used.split():
    problems.append(f"frames did not name the guest compute id={compute}: {used}")
if str(report.get("graphics_pipeline_used", "")) != str(graphics):
    problems.append(
        f"graphics pipeline used {report.get('graphics_pipeline_used')} != registered {graphics}"
    )
if str(report.get("mesh_pipeline_used", "")) != str(mesh):
    problems.append(
        f"mesh pipeline used {report.get('mesh_pipeline_used')} != registered {mesh}"
    )

# The culling dispatch must carry the argument buffer as a binding, a barrier
# must order it before the indirect draw, and the GPU must have written the
# expected record.
linkage = report.get("compute_indirect_linkage", "")
if not linkage.startswith("true"):
    problems.append("compute_indirect_linkage: " + (linkage or "missing"))
if str(report.get("compute_indirect_linkage_frames", "0")) != str(report.get("frames_ok", "-1")):
    problems.append(
        f"linkage frames {report.get('compute_indirect_linkage_frames')} != "
        f"frames_ok {report.get('frames_ok')}"
    )
if str(report.get("compute_dispatches_without_bindings", "?")) not in ("0",):
    problems.append(
        "compute dispatches without bindings="
        + str(report.get("compute_dispatches_without_bindings", "?"))
    )
if report.get("args_after_frames", "") != "3,1,0,0":
    problems.append(
        "compute-written argument record=" + report.get("args_after_frames", "missing")
        + " expected=3,1,0,0"
    )

if problems:
    first = stage_failure or (problems[0].split("=")[0].strip() if problems else "unknown")
    print("PARTIAL live-d3d12-gate first_failing_stage=" + str(first) +
          " reason=\"" + "; ".join(problems) + "\"")
    sys.exit(1)

print(f"PASS live-d3d12-gate measured={count}/{measured_expected} p50_ms={p50:.3f} "
      f"p95_ms={p95:.3f} p99_ms={p99:.3f} fps={fps:.2f} "
      f"checksum={host_checksum} d3dmetal_calls=0 pool_jobs_s={pool_data.get('jobs_per_s','?')}")
PY

echo "live-d3d12-gate: exit $PY_EXIT"
exit "$PY_EXIT"
