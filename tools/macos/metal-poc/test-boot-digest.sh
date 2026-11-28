#!/bin/sh
# Hermetic tests for boot-digest.py (Cycle 15 comparison digest).
#
# No GPU, no Wine and no title: every fixture is a synthetic
# `run-title-metal.sh --out` directory under a mktemp root.  The fixtures cover
# a full Boot #23-shaped run with draw traces and a frame probe, two earlier
# boots (one without any of the new telemetry keys), a malformed/truncated run,
# a hostile-text run and a missing directory.
#
# A final Python verifier pins the exact key sets, the delta arithmetic, the
# sanitisation of hostile free text, determinism and the absence of every
# fixture path, title name, asset marker or raw log line in the output.
set -eu

SELF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOOL="$SELF_DIR/boot-digest.py"
CENSUS="$SELF_DIR/boot-census.py"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/boot-digest-test.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

fail() {
    echo "test-boot-digest.sh: FAIL: $*" >&2
    exit 1
}

[ -f "$TOOL" ] || fail "missing $TOOL"
[ -f "$CENSUS" ] || fail "missing $CENSUS (boot-digest.py reuses its parsers)"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "python3 3.11 or newer is required"

# --- Fixtures --------------------------------------------------------------

cat > "$TMP/make_fixtures.py" <<'PY'
import json
import os
import sys

root = sys.argv[1]


def write(case, name, text):
    directory = os.path.join(root, case)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(text)


def summary(case, lines):
    write(case, "summary.txt", "\n".join(lines) + "\n")


def telemetry(case, payload):
    write(case, "telemetry.json", json.dumps(payload, indent=2) + "\n")


# --- Boot #21: a legacy directory with no Cycle 14/15 host counters ---------
summary(
    "boot21",
    [
        "launcher=run-title-metal",
        "backend=metal",
        "status=PASS",
        "exit_code=0",
        "mode=gate",
        "frames=80",
        "frames_expected=80",
        "p50_ms=18.000",
        "p95_ms=22.000",
        "fps=55.00",
        "guest_submits_ok=80",
        "guest_submits_refused=1",
        "guest_shaders_refused=2",
        "host_frames=80",
        "host_p50_ms=17.000",
        "host_p95_ms=21.000",
        "host_fps=58.00",
        "d3dmetal_fallback_count=0",
        "guest_d3dmetal_fallback_count=0",
        "telemetry=/fixture/SecretTitle/telemetry.json",
    ],
)
telemetry(
    "boot21",
    {
        "host": "victory-metal-host",
        "submissions": 80,
        "frames": 80,
        "refused_submissions": 1,
        "d3dmetal_fallback_count": 0,
        "pipelines": {"p1": 1, "p2": 2},
        "d3dmetal_loaded": False,
    },
)
write("boot21", "host.log", "victory-metal-host: ready port=4560\n")

# --- Boot #22: stage counters land, draw counters do not --------------------
summary(
    "boot22",
    [
        "backend=metal",
        "status=PASS",
        "exit_code=0",
        "frames=100",
        "p50_ms=15.000",
        "p95_ms=18.000",
        "fps=66.66",
        "guest_submits_ok=100",
        "guest_submits_refused=5",
        "guest_shaders_refused=1",
        "host_frames=100",
        "host_p50_ms=14.000",
        "host_p95_ms=17.000",
        "host_fps=70.00",
        "host_refused_submissions=5",
        "d3dmetal_fallback_count=0",
        "guest_d3dmetal_fallback_count=0",
        "telemetry=/fixture/SecretTitle/telemetry.json",
    ],
)
telemetry(
    "boot22",
    {
        "host": "victory-metal-host",
        "submissions": 100,
        "frames": 100,
        "refused_submissions": 5,
        "d3dmetal_fallback_count": 0,
        "host_translated_stages": 9,
        "host_translated_stage_vertex": 4,
        "host_translated_stage_pixel": 4,
        "host_translated_stage_mesh": 0,
        "host_translated_stage_object": 1,
        "host_translated_stage_compute": 0,
        "host_pipeline_translations_ok": 4,
        "host_pipeline_translations_refused": 1,
        "pipelines": {},
        "d3dmetal_loaded": False,
    },
)
write("boot22", "host.log", "victory-metal-host: ready port=4561\n")

# --- Boot #23: exact encode counters, frame probe and draw traces -----------
summary(
    "boot23",
    [
        "backend=metal",
        "status=PASS",
        "exit_code=0",
        "mode=window",
        "frames=120",
        "frames_expected=120",
        "p50_ms=10.000",
        "p95_ms=12.000",
        "fps=100.00",
        "guest_submits_ok=118",
        "guest_submits_refused=2",
        "guest_shaders_refused=1",
        "host_frames=120",
        "host_p50_ms=9.500",
        "host_p95_ms=11.500",
        "host_fps=105.00",
        "host_refused_submissions=2",
        "d3dmetal_fallback_count=0",
        "guest_d3dmetal_fallback_count=0",
        "telemetry=/fixture/SecretTitle/telemetry.json",
        "guest_log=/fixture/SecretTitle/guest.log",
    ],
)
telemetry(
    "boot23",
    {
        "host": "victory-metal-host",
        "submissions": 120,
        "frames": 120,
        "refused_submissions": 2,
        "d3dmetal_fallback_count": 0,
        "host_translated_stages": 12,
        "host_translated_stage_vertex": 4,
        "host_translated_stage_pixel": 5,
        "host_translated_stage_mesh": 1,
        "host_translated_stage_object": 1,
        "host_translated_stage_compute": 1,
        "host_pipeline_translations_ok": 6,
        "host_pipeline_translations_refused": 1,
        "draws_encoded": 12,
        "draw_indexed_encoded": 8,
        "dispatches_encoded": 3,
        "mesh_dispatches_encoded": 1,
        "indirect_encoded": 2,
        "render_passes_encoded": 7,
        "clear_ops_encoded": 3,
        "unknown_handle_site": "buffer",
        "pipelines": {"a": 1, "b": 2, "c": 3},
        "d3dmetal_loaded": False,
        "frame_probe": {
            "sampled_frames": 3,
            "probe_pixels": 4096,
            "last_sampled_frame": 118,
            "non_black_pixels": 2048,
            "unique_luma_buckets": 7,
            "uniform_color": "swatch_0d1a33",
            "nonzero_bytes": 8192,
            "frame_crc": 123456789,
            "sample_cost_us": 500,
            "dropped_samples": 1,
            "fallback_clear": False,
            "probe_every": 10,
            "failure": None,
        },
    },
)
write(
    "boot23",
    "host.log",
    "\x1b[32m2026-11-28T10:00:00Z INFO victory_metal_host: "
    "draw_trace ordinal=1 kind=draw pipeline=7 count=3 instances=1 "
    "viewport=0,0,1920,1080 scissor=none abi_parameters=2 abi_bytes=64 "
    "rt=swapchain 1920x1080 bgra8\x1b[0m\n"
    "2026-11-28T10:00:01Z INFO victory_metal_host: "
    "draw_trace ordinal=2 kind=draw_indexed pipeline=8 count=36 instances=1 "
    "viewport=0,0,1920,1080 scissor=none abi_parameters=2 abi_bytes=64 "
    "rt=offscreen 1920x1080 rgba8\n"
    "2026-11-28T10:00:05Z INFO victory_metal_host: "
    "draw_trace_rt nonzero_bytes=1024 sampled_bytes=4096 distinct_values=42\n"
    "2026-11-28T10:00:06Z INFO victory_metal_host: "
    "draw_trace_rt failure=texture-unsupported\n"
    "asset C:\\Games\\SecretTitle\\secret-title.exe SECRET_PAYLOAD\n"
    "draw_trace ordinal=notanumber kind=draw SECRET_PAYLOAD\n"
    "draw_trace_rt ??? SECRET_PAYLOAD\n",
)

# --- Malformed telemetry and truncated trace lines --------------------------
summary(
    "malformed",
    [
        "backend=metal",
        "status=PASS",
        "frames=n/a",
        "host_frames=wat",
        "p50_ms=n/a",
        "telemetry=/fixture/SecretTitle/telemetry.json",
    ],
)
write("malformed", "telemetry.json", "{ this is not json\n")
write(
    "malformed",
    "host.log",
    "draw_trace ordinal=oops kind=draw SECRET_PAYLOAD\n"
    "draw_trace_rt ??? SECRET_PAYLOAD\n"
    "draw_trace_rt nonzero_bytes= SECRET_PAYLOAD\n",
)

# --- Hostile free text must become bounded classes -------------------------
summary(
    "hostile",
    [
        "backend=metal",
        "status=PASS",
        "frames=1",
        "host_frames=1",
        "telemetry=/fixture/SecretTitle/telemetry.json",
    ],
)
telemetry(
    "hostile",
    {
        "host": "victory-metal-host",
        "submissions": 1,
        "frames": 1,
        "refused_submissions": 0,
        "d3dmetal_fallback_count": 0,
        "unknown_handle_site": "C:\\Games\\SecretTitle",
        "frame_probe": {
            "uniform_color": "C:\\Games\\SecretTitle",
            "failure": "C:/Games/SecretTitle",
        },
    },
)
write(
    "hostile",
    "host.log",
    "draw_trace ordinal=1 kind=C:\\Games\\SecretTitle pipeline=99 count=1 "
    "instances=1 rt=C:\\Games\\SecretTitle SECRET_PAYLOAD\n"
    "draw_trace_rt failure=C:\\Games\\SecretTitle SECRET_PAYLOAD\n",
)
PY

python3 "$TMP/make_fixtures.py" "$TMP" || fail "fixture generation failed"

# --- Run the tool ----------------------------------------------------------

run_tool() {
    case_name=$1
    shift
    set +e
    python3 "$TOOL" "$@" > "$TMP/$case_name.out" 2> "$TMP/$case_name.err"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$TMP/$case_name.rc"
    if [ -s "$TMP/$case_name.err" ]; then
        echo "test-boot-digest.sh: stderr from $case_name:" >&2
        cat "$TMP/$case_name.err" >&2
        fail "$case_name wrote to stderr"
    fi
}

run_tool current --out "$TMP/boot23" --baseline "$TMP/boot22" --baseline "$TMP/boot21"
run_tool current-again --out "$TMP/boot23" --baseline "$TMP/boot22" --baseline "$TMP/boot21"
run_tool legacy --out "$TMP/boot22" --baseline "$TMP/boot21"
run_tool malformed --out "$TMP/malformed"
run_tool hostile --out "$TMP/hostile"
run_tool nodir --out "$TMP/no-such-directory"

cmp -s "$TMP/current.out" "$TMP/current-again.out" \
    || fail "JSON output is not deterministic across invocations"

# --- Exact assertions ------------------------------------------------------

cat > "$TMP/verify.py" <<'PY'
import json
import os
import sys

root = sys.argv[1]
checks = 0


def fail(message):
    raise SystemExit("test-boot-digest.sh: ASSERTION FAILED: " + message)


def read(name):
    path = os.path.join(root, name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as error:
        fail(f"cannot read {name}: {error}")


def load(name):
    try:
        value = json.loads(read(name))
    except json.JSONDecodeError as error:
        fail(f"{name} is not valid JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{name} is not a JSON object")
    return value


def get(document, path):
    if not path:
        return document
    value = document
    for part in path.split("."):
        if isinstance(value, list):
            try:
                index = int(part)
            except ValueError:
                fail(f"missing list index in path {path}")
            if index < 0 or index >= len(value):
                fail(f"missing path {path}")
            value = value[index]
            continue
        if not isinstance(value, dict) or part not in value:
            fail(f"missing path {path}")
        value = value[part]
    return value


def eq(document, path, expected, label=None):
    global checks
    actual = get(document, path)
    checks += 1
    if actual != expected:
        fail(f"{label or path}: expected {expected!r}, got {actual!r}")


def keys(document, path, expected, label=None):
    global checks
    actual = get(document, path)
    checks += 1
    if not isinstance(actual, dict):
        fail(f"{label or path}: expected an object, got {type(actual).__name__}")
    if set(actual) != set(expected):
        fail(
            f"{label or path}: expected keys {sorted(expected)}, got {sorted(actual)}"
        )


def rc(name, expected):
    global checks
    text = read(name + ".rc").strip()
    checks += 1
    if text != str(expected):
        fail(f"{name} exit: expected {expected}, got {text}")


def no_leak(name, needles):
    global checks
    text = read(name + ".out")
    for needle in needles:
        checks += 1
        if needle in text:
            fail(f"{name}.out leaks {needle!r}")
    checks += 1
    if "\x1b" in text:
        fail(f"{name}.out still contains ANSI escapes")


TOP_LEVEL = {
    "tool",
    "schema",
    "backend",
    "status",
    "frames",
    "submissions",
    "refused_submissions",
    "d3dmetal_fallback_count",
    "guest_d3dmetal_fallback_count",
    "host_pipeline_translations_ok",
    "host_pipeline_translations_refused",
    "host_translated_stages",
    "host_translated_stage_vertex",
    "host_translated_stage_pixel",
    "host_translated_stage_mesh",
    "host_translated_stage_object",
    "host_translated_stage_compute",
    "draws_encoded",
    "draw_indexed_encoded",
    "dispatches_encoded",
    "mesh_dispatches_encoded",
    "indirect_encoded",
    "render_passes_encoded",
    "clear_ops_encoded",
    "unknown_handle_site",
    "frame_probe",
    "draw_trace",
    "timing",
    "pipelines",
    "inputs",
    "comparison",
}

FRAME_PROBE_KEYS = {
    "present",
    "sampled_frames",
    "probe_pixels",
    "last_sampled_frame",
    "non_black_pixels",
    "unique_luma_buckets",
    "uniform_color",
    "nonzero_bytes",
    "frame_crc",
    "sample_cost_us",
    "dropped_samples",
    "fallback_clear",
    "probe_every",
    "failure",
}

DRAW_TRACE_KEYS = {
    "total",
    "malformed_lines",
    "by_kind",
    "by_rt",
    "rt_total",
    "rt_sampled_bytes",
    "rt_nonzero_bytes",
    "rt_distinct_values_max",
    "rt_nonzero_lines",
    "rt_zero_lines",
    "rt_failures",
}

# --- Boot #23 run, Boot #22 and Boot #21 baselines -------------------------
current = load("current.out")
keys(current, "", TOP_LEVEL, "top-level")
rc("current", 0)
eq(current, "tool", "boot-digest")
eq(current, "schema", 1)
eq(current, "backend", "metal")
eq(current, "status", "PASS")
eq(current, "frames", 120)
eq(current, "submissions", 120)
eq(current, "refused_submissions", 2)
eq(current, "d3dmetal_fallback_count", 0)
eq(current, "guest_d3dmetal_fallback_count", 0)
eq(current, "host_pipeline_translations_ok", 6)
eq(current, "host_pipeline_translations_refused", 1)
eq(current, "host_translated_stages", 12)
eq(current, "host_translated_stage_vertex", 4)
eq(current, "host_translated_stage_pixel", 5)
eq(current, "host_translated_stage_mesh", 1)
eq(current, "host_translated_stage_object", 1)
eq(current, "host_translated_stage_compute", 1)
eq(current, "draws_encoded", 12)
eq(current, "draw_indexed_encoded", 8)
eq(current, "dispatches_encoded", 3)
eq(current, "mesh_dispatches_encoded", 1)
eq(current, "indirect_encoded", 2)
eq(current, "render_passes_encoded", 7)
eq(current, "clear_ops_encoded", 3)
eq(current, "unknown_handle_site", "buffer")

keys(current, "frame_probe", FRAME_PROBE_KEYS)
eq(current, "frame_probe.present", True)
eq(current, "frame_probe.sampled_frames", 3)
eq(current, "frame_probe.probe_pixels", 4096)
eq(current, "frame_probe.last_sampled_frame", 118)
eq(current, "frame_probe.non_black_pixels", 2048)
eq(current, "frame_probe.unique_luma_buckets", 7)
eq(current, "frame_probe.uniform_color", "swatch_0d1a33")
eq(current, "frame_probe.nonzero_bytes", 8192)
eq(current, "frame_probe.frame_crc", 123456789)
eq(current, "frame_probe.sample_cost_us", 500)
eq(current, "frame_probe.dropped_samples", 1)
eq(current, "frame_probe.fallback_clear", False)
eq(current, "frame_probe.probe_every", 10)
eq(current, "frame_probe.failure", None)

keys(current, "draw_trace", DRAW_TRACE_KEYS)
eq(current, "draw_trace.total", 2)
eq(current, "draw_trace.malformed_lines", 2)
eq(current, "draw_trace.by_kind", {"draw": 1, "draw_indexed": 1})
eq(current, "draw_trace.by_rt", {"offscreen": 1, "swapchain": 1})
eq(current, "draw_trace.rt_total", 2)
eq(current, "draw_trace.rt_sampled_bytes", 4096)
eq(current, "draw_trace.rt_nonzero_bytes", 1024)
eq(current, "draw_trace.rt_distinct_values_max", 42)
eq(current, "draw_trace.rt_nonzero_lines", 1)
eq(current, "draw_trace.rt_zero_lines", 0)
eq(current, "draw_trace.rt_failures", {"texture-unsupported": 1})

keys(current, "timing", {
    "guest_p50_ms",
    "guest_p95_ms",
    "guest_fps",
    "host_p50_ms",
    "host_p95_ms",
    "host_fps",
})
eq(current, "timing.guest_p50_ms", 10.0)
eq(current, "timing.guest_p95_ms", 12.0)
eq(current, "timing.guest_fps", 100.0)
eq(current, "timing.host_p50_ms", 9.5)
eq(current, "timing.host_p95_ms", 11.5)
eq(current, "timing.host_fps", 105.0)

eq(current, "pipelines", {"created": 3, "refused": 1})
eq(current, "inputs", {
    "summary_present": True,
    "telemetry_present": True,
    "telemetry_valid": True,
    "host_log_present": True,
})

# --- Deltas against Boot #22 (index 1) and Boot #21 (index 2) --------------
eq(current, "comparison.baselines.0.index", 1)
eq(current, "comparison.baselines.0.delta.frames", 20)
eq(current, "comparison.baselines.0.delta.submissions", 20)
eq(current, "comparison.baselines.0.delta.refused_submissions", -3)
eq(current, "comparison.baselines.0.delta.host_pipeline_translations_ok", 2)
eq(current, "comparison.baselines.0.delta.host_pipeline_translations_refused", 0)
eq(current, "comparison.baselines.0.delta.host_translated_stages", 3)
eq(current, "comparison.baselines.0.delta.host_translated_stage_pixel", 1)
eq(current, "comparison.baselines.0.delta.host_translated_stage_compute", 1)
eq(current, "comparison.baselines.0.delta.draws_encoded", None)
eq(current, "comparison.baselines.0.delta.draw_trace.total", 2)
eq(current, "comparison.baselines.0.delta.draw_trace.by_kind.draw", 1)
eq(current, "comparison.baselines.0.delta.draw_trace.by_kind.draw_indexed", 1)
eq(current, "comparison.baselines.0.delta.draw_trace.by_rt.swapchain", 1)
eq(current, "comparison.baselines.0.delta.draw_trace.by_rt.offscreen", 1)
eq(current, "comparison.baselines.0.delta.draw_trace.by_rt.none", 0)
eq(current, "comparison.baselines.0.delta.draw_trace.rt_failures.texture-gone", 0)
eq(current, "comparison.baselines.0.delta.timing.guest_p50_ms", -5.0)
eq(current, "comparison.baselines.0.delta.timing.host_p50_ms", -4.5)
eq(current, "comparison.baselines.0.delta.pipelines.created", 3)
eq(current, "comparison.baselines.0.delta.pipelines.refused", 0)
eq(current, "comparison.baselines.0.fields.draws_encoded.baseline", None)
eq(current, "comparison.baselines.0.fields.draws_encoded.current", 12)
eq(current, "comparison.baselines.0.fields.draws_encoded.changed", True)
eq(current, "comparison.baselines.0.fields.frames.baseline", 100)
eq(current, "comparison.baselines.0.fields.frames.current", 120)
eq(current, "comparison.baselines.0.fields.frame_probe.present.baseline", False)
eq(current, "comparison.baselines.0.fields.frame_probe.present.changed", True)
eq(current, "comparison.baselines.0.fields.backend.changed", False)
eq(current, "comparison.baselines.0.fields.status.changed", False)

eq(current, "comparison.baselines.1.index", 2)
eq(current, "comparison.baselines.1.delta.frames", 40)
eq(current, "comparison.baselines.1.delta.submissions", 40)
eq(current, "comparison.baselines.1.delta.refused_submissions", 1)
eq(current, "comparison.baselines.1.delta.host_pipeline_translations_ok", None)
eq(current, "comparison.baselines.1.delta.timing.guest_p50_ms", -8.0)
eq(current, "comparison.baselines.1.delta.timing.host_p50_ms", -7.5)
eq(current, "comparison.baselines.1.delta.pipelines.created", 1)
eq(current, "comparison.baselines.1.delta.pipelines.refused", -1)
eq(current, "comparison.baselines.1.fields.frames.baseline", 80)

# --- A Boot #22 directory without the new keys still digests ---------------
legacy = load("legacy.out")
keys(legacy, "", TOP_LEVEL, "legacy top-level")
rc("legacy", 0)
eq(legacy, "frames", 100)
eq(legacy, "host_translated_stages", 9)
eq(legacy, "host_pipeline_translations_ok", 4)
eq(legacy, "host_pipeline_translations_refused", 1)
eq(legacy, "draws_encoded", None)
eq(legacy, "draw_indexed_encoded", None)
eq(legacy, "render_passes_encoded", None)
eq(legacy, "unknown_handle_site", None)
eq(legacy, "frame_probe.present", False)
eq(legacy, "frame_probe.non_black_pixels", None)
eq(legacy, "frame_probe.uniform_color", None)
eq(legacy, "draw_trace.total", 0)
eq(legacy, "draw_trace.by_kind", {})
eq(legacy, "comparison.baselines.0.delta.frames", 20)
eq(legacy, "comparison.baselines.0.delta.draws_encoded", None)

# --- Malformed input is counted, never a panic -----------------------------
malformed = load("malformed.out")
rc("malformed", 0)
eq(malformed, "backend", "metal")
eq(malformed, "status", "PASS")
eq(malformed, "frames", None)
eq(malformed, "timing.guest_p50_ms", None)
eq(malformed, "inputs.telemetry_valid", False)
eq(malformed, "frame_probe.present", False)
eq(malformed, "draw_trace.total", 0)
eq(malformed, "draw_trace.malformed_lines", 3)
eq(malformed, "draw_trace.rt_total", 0)
eq(malformed, "comparison.baselines", [])

# --- Hostile free text becomes a class, not content ------------------------
hostile = load("hostile.out")
rc("hostile", 0)
eq(hostile, "unknown_handle_site", "other")
eq(hostile, "frame_probe.present", True)
eq(hostile, "frame_probe.uniform_color", "other")
eq(hostile, "frame_probe.failure", "other")
eq(hostile, "draw_trace.total", 1)
eq(hostile, "draw_trace.by_kind", {"other": 1})
eq(hostile, "draw_trace.by_rt", {"other": 1})
eq(hostile, "draw_trace.rt_total", 1)
eq(hostile, "draw_trace.rt_failures", {"other": 1})

# --- Missing directory is null, not an error ------------------------------
nodir = load("nodir.out")
rc("nodir", 0)
eq(nodir, "backend", None)
eq(nodir, "status", None)
eq(nodir, "frames", None)
eq(nodir, "submissions", None)
eq(nodir, "refused_submissions", None)
eq(nodir, "host_translated_stages", None)
eq(nodir, "draws_encoded", None)
eq(nodir, "frame_probe.present", False)
eq(nodir, "draw_trace.total", 0)
eq(nodir, "inputs.summary_present", False)
eq(nodir, "inputs.telemetry_present", False)
eq(nodir, "inputs.host_log_present", False)

# --- No path, title, payload marker or raw trace line leaks ----------------
leaks = [
    "SecretTitle",
    "secret-title.exe",
    "SECRET_PAYLOAD",
    "C:\\Games",
    "C:/Games",
    "/fixture",
    root,
    "draw_trace ordinal",
    "draw_trace_rt nonzero_bytes",
]
for name in ("current", "legacy", "malformed", "hostile", "nodir"):
    no_leak(name, leaks)

print(f"checks={checks}")
PY

checks=$(python3 "$TMP/verify.py" "$TMP")
echo "test-boot-digest.sh: PASS ($checks)"
