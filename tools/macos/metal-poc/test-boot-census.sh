#!/bin/sh
# Hermetic tests for boot-census.py.
#
# No GPU, no Wine and no title are needed: every fixture is a synthetic
# `run-title-metal.sh --out` directory under a mktemp root.  The fixtures cover
# a clean run, a refusal-heavy run, missing telemetry, ANSI colour codes and a
# truncated log, a fallback gate failure, an orphan-survivor gate failure and a
# missing directory.  A final Python verifier asserts exact digest values,
# gate exit codes, determinism and that no fixture path or title name leaks
# into either output format.
set -eu

SELF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOOL="$SELF_DIR/boot-census.py"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/boot-census-test.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

fail() {
    echo "test-boot-census.sh: FAIL: $*" >&2
    exit 1
}

[ -f "$TOOL" ] || fail "missing $TOOL"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "python3 3.11 or newer is required"

# --- Fixtures --------------------------------------------------------------

cat > "$TMP/make_fixtures.py" <<'PY'
import json
import os
import sys

root = sys.argv[1]


def write(case, name, text, *, newline=True):
    directory = os.path.join(root, case)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        if newline and text and not text.endswith("\n"):
            handle.write("\n")


def summary(case, lines):
    write(case, "summary.txt", "\n".join(lines) + "\n")


# --- Clean run -------------------------------------------------------------
clean_lines = [
    "launcher=run-title-metal",
    "status=PASS",
    "exit_code=0",
    "reason=clean C:\\Games\\SecretTitle\\secret-title.exe",
    "mode=gate",
    "headless=1",
    "window=none",
    "host_ready_line=ready port=4567",
    "guest_exit=0",
    "host_exit=0",
    "host_terminated_by_launcher=1",
    "frames=120",
    "frames_expected=120",
    "p50_ms=11.111",
    "p95_ms=12.222",
    "p99_ms=13.333",
    "fps=89.99",
    "guest_frames_ok=120",
    "guest_submits_ok=120",
    "guest_submits_refused=0",
    "guest_presents_ok=120",
    "guest_presents_refused=0",
    "guest_shader_psos_registered=7",
    "guest_shaders_compiled=7",
    "guest_shaders_accepted=7",
    "guest_shaders_refused=0",
    "guest_buffers_registered=12",
    "guest_textures_registered=5",
    "guest_buffers_refused=0",
    "guest_textures_refused=0",
    "guest_registration_failures=0",
    "host_frames=120",
    "host_p50_ms=10.500",
    "host_p95_ms=11.500",
    "host_p99_ms=12.500",
    "host_fps=90.00",
    "host_refused_submissions=0",
    "host_window_drawables=120",
    "d3dmetal_fallback_count=0",
    "guest_d3dmetal_fallback_count=0",
    "guest_d3dmetal_module=0",
    "d3dmetal_loaded=0",
    "unsupported_opcodes=none",
    "host_unsupported_opcodes={}",
    "host_survived=0",
    "victim_survived=0",
    "orphan_survived=0",
    "guest_timed_out=0",
    "host_timed_out=0",
    "host_d3d12_creation_attempts=0",
    "ignored_states=0",
    "refused_states=0",
    "refusals=none",
    "milestones=host_ready,d3d12x_device,completion_page,dstorage_queue,"
    "first_native_submit,first_frame_presented,first_3d_draw,"
    "first_compute_dispatch,sustained_frames",
    "telemetry=/abs/fixture/SecretTitle/telemetry.json",
    "guest_log=/abs/fixture/SecretTitle/guest.log",
]
summary("clean", clean_lines)
write(
    "clean",
    "telemetry.json",
    json.dumps(
        {
            "host": "victory-metal-host",
            "pid": 4242,
            "sessions": 1,
            "submissions": 120,
            "frames": 120,
            "refused_submissions": 0,
            "p50_ms": 10.5,
            "p95_ms": 11.5,
            "p99_ms": 12.5,
            "fps": 90.0,
            "unsupported_opcodes": {},
            "pipelines": {},
            "d3dmetal_loaded": False,
            "d3dmetal_images": [],
            "d3dmetal_fallback_count": 0,
            "window_drawables": 120,
        }
    ),
)
write(
    "clean",
    "host.log",
    "\x1b[32m2026-11-27T10:00:00Z INFO victory_metal_host: "
    "guard=backend=metal host_d3d12_creation_attempts=0 "
    "host_d3d12_requests_refused=0\x1b[0m\n"
    "victory-metal-host: ready port=4567\n",
)
write(
    "clean",
    "guest.log",
    "\x1b[33mMETAL_HOST_READY\x1b[0m\n"
    "opened C:\\Games\\SecretTitle\\secret-title.exe\n"
    "asset /abs/fixture/SecretTitle/data.bin\n",
)

# --- Refusal-heavy run -----------------------------------------------------
refusal_lines = [
    "status=FAIL",
    "exit_code=12",
    "reason=gate report ok=0 unsupported_format",
    "mode=title",
    "headless=1",
    "window=none",
    "host_ready_line=ready port=4567",
    "guest_exit=0",
    "host_exit=0",
    "host_terminated_by_launcher=1",
    "frames=40",
    "frames_expected=40",
    "p50_ms=20.0",
    "p95_ms=25.0",
    "p99_ms=30.0",
    "fps=40.0",
    "guest_frames_ok=35",
    "guest_submits_ok=30",
    "guest_submits_refused=10",
    "guest_presents_ok=32",
    "guest_presents_refused=3",
    "guest_shader_psos_registered=2",
    "guest_shaders_compiled=9",
    "guest_shaders_accepted=2",
    "guest_shaders_refused=7",
    "guest_buffers_registered=4",
    "guest_textures_registered=1",
    "guest_buffers_refused=3",
    "guest_textures_refused=2",
    "guest_registration_failures=5",
    "host_frames=40",
    "host_p50_ms=n/a",
    "host_p95_ms=26.0",
    "host_p99_ms=31.0",
    "host_fps=39.0",
    "host_refused_submissions=10",
    "host_window_drawables=35",
    "d3dmetal_fallback_count=0",
    "guest_d3dmetal_fallback_count=0",
    "guest_d3dmetal_module=0",
    "unsupported_opcodes=3:4,7:2,dispatch_3d:1",
    "host_unsupported_opcodes={\"texture format\": 2, \"pipeline state\": 1}",
    "formats_refused=28:1,87:1",
    "ignored_states=3",
    "refused_states=4",
    "refusals=unsupported_format:5,missing_binding:3,other:1",
    "host_survived=0",
    "victim_survived=0",
    "orphan_survived=0",
    "guest_timed_out=0",
    "host_timed_out=0",
    "host_d3d12_creation_attempts=0",
    "milestones=host_ready,d3d12x_device,completion_page,first_native_submit,"
    "first_frame_presented,first_3d_draw,first_compute_dispatch",
    "telemetry=/abs/fixture/SecretTitle/telemetry.json",
    "guest_log=/abs/fixture/SecretTitle/guest.log",
]
summary("refusal", refusal_lines)
write(
    "refusal",
    "telemetry.json",
    json.dumps(
        {
            "submissions": 40,
            "frames": 40,
            "refused_submissions": 10,
            "p50_ms": 21.0,
            "p95_ms": 26.0,
            "p99_ms": 31.0,
            "fps": 39.0,
            "unsupported_opcodes": {"no such binding": 1},
            "d3dmetal_fallback_count": 0,
            "window_drawables": 35,
        }
    ),
)
write(
    "refusal",
    "host.log",
    "2026-11-27T10:00:00Z INFO victory_metal_host: "
    "guard=backend=metal host_d3d12_creation_attempts=0 "
    "host_d3d12_requests_refused=1\n"
    "victory-metal-host: ready port=4568\n",
)
write(
    "refusal",
    "guest.log",
    "\x1b[31mMETAL_HOST_ERROR: texture format 28 is not supported\x1b[0m\n"
    "refused_size_bytes=1024\n"
    "refused_size_bytes=2048\n"
    "refused_size_bytes=524288\n"
    "refused_size_bytes=8388608\n"
    "refused_size_bytes=134217728\n"
    "refused_format=28\n"
    "refused_format=28\n"
    "refused_format=87\n"
    "refused_pipeline=vs_main\n"
    "refused_pipeline=vs_main\n"
    "refused_pipeline=ps_lit\n"
    "opcode=3\n"
    "opcode=3\n"
    "opcode=7\n"
    "ignored_state=blend\n"
    "ignored_state=blend\n"
    "refused_state=rasterizer\n",
)

# --- Missing telemetry.json ------------------------------------------------
summary(
    "missing",
    [
        "status=PASS",
        "exit_code=0",
        "mode=title",
        "frames=3",
        "p50_ms=n/a",
        "p95_ms=n/a",
        "p99_ms=n/a",
        "fps=n/a",
        "guest_frames_ok=3",
        "guest_submits_ok=1",
        "guest_submits_refused=0",
        "guest_presents_ok=1",
        "guest_presents_refused=2",
        "guest_shader_psos_registered=n/a",
        "host_frames=n/a",
        "host_p50_ms=n/a",
        "host_p95_ms=n/a",
        "host_p99_ms=n/a",
        "host_fps=n/a",
        "d3dmetal_fallback_count=n/a",
        "guest_d3dmetal_fallback_count=n/a",
        "guest_d3dmetal_module=n/a",
        "host_d3d12_creation_attempts=n/a",
        "host_survived=0",
        "victim_survived=0",
        "orphan_survived=0",
        "guest_timed_out=0",
        "host_timed_out=0",
        "refusals=none",
    ],
)

# --- ANSI colour and truncated log ----------------------------------------
ansi_summary = (
    "\x1b[32mstatus=PASS\x1b[0m\n"
    "\x1b[1mmode=title\x1b[0m\n"
    "exit_code=0\n"
    "guest_exit=0\n"
    "host_exit=0\n"
    "frames=5\n"
    "p50_ms=1.5\n"
    "guest_frames_ok=5\n"
    "guest_submits_ok=5\n"
    "guest_submits_refused=0\n"
    "guest_presents_ok=5\n"
    "guest_presents_refused=0\n"
    "d3dmetal_fallback_count=0\n"
    "guest_d3dmetal_module=0\n"
    "host_d3d12_creation_attempts=0\n"
    "host_survived=0\n"
    "victim_survived=0\n"
    "orphan_survived=0\n"
    "guest_timed_out=0\n"
    "host_timed_out=0\n"
    "refused_size_bytes=4096\n"
    "refusals=none\n"
)
summary("ansi", ansi_summary.splitlines())
# Deliberately truncated JSON: present, invalid, must not crash.
write(
    "ansi",
    "telemetry.json",
    '{"frames": 5, "p50_ms": 2.0, "d3dmetal_fallback_count": 0,',
)
write(
    "ansi",
    "guest.log",
    "\x1b[31mMETAL_HOST_ERROR: incomplete frame "
    "C:\\Games\\My Title\\SecretTitle\\data.bin\x1b[0m\n"
    "refused_format=28\n"
    "refused_format=28:2\n"
    "refused_size_bytes=8388608\n"
    "unfinished refused_size_bytes=",
    newline=False,
)
write(
    "ansi",
    "host.log",
    "\x1b[36mvictory-metal-host: ready port=4569\x1b[0m\n"
    "guard=backend=metal host_d3d12_creation_attempts=0\n",
)

# --- Fallback gate failure -------------------------------------------------
summary(
    "fallback",
    [
        "status=FAIL",
        "exit_code=11",
        "mode=title",
        "frames=1",
        "guest_frames_ok=1",
        "guest_submits_ok=1",
        "guest_presents_ok=1",
        "d3dmetal_fallback_count=n/a",
        "guest_d3dmetal_fallback_count=0",
        "host_d3d12_creation_attempts=0",
        "host_survived=0",
        "victim_survived=0",
        "orphan_survived=0",
        "guest_timed_out=0",
        "host_timed_out=0",
        "refusals=none",
    ],
)
write(
    "fallback",
    "telemetry.json",
    json.dumps(
        {
            "frames": 1,
            "submissions": 1,
            "p50_ms": 12.0,
            "d3dmetal_fallback_count": 1,
        }
    ),
)

# --- Host D3D12 creation gate failure, from the host log guard line --------
summary(
    "creation",
    [
        "status=FAIL",
        "exit_code=11",
        "mode=title",
        "frames=1",
        "guest_frames_ok=1",
        "guest_submits_ok=1",
        "guest_presents_ok=1",
        "d3dmetal_fallback_count=0",
        "host_d3d12_creation_attempts=n/a",
        "host_survived=0",
        "victim_survived=0",
        "orphan_survived=0",
        "guest_timed_out=0",
        "host_timed_out=0",
        "refusals=none",
    ],
)
write(
    "creation",
    "host.log",
    "2026-11-27T10:00:00Z WARN victory_metal_host: "
    "guard=backend=metal host_d3d12_creation_attempts=1 "
    "host_d3d12_requests_refused=0\n"
    "native frame: 1 in window (1) 1.000/1.000/1.000 ms "
    "p50/p95/p99, 1.0 fps, fallback=0, submits=1/0 ok/refused, "
    "presents=1/0 ok/refused, registered=6 buffers/3 psos, "
    "pool=0 cold/0 warm, registration_failures=0, compute_unbound=0, "
    "opcodes=[], refusals=[], named_refusals=[]\n",
)

# --- Orphan-survivor gate failure, proven by the hygiene marker ------------
summary(
    "orphan",
    [
        "status=FAIL",
        "exit_code=0",
        "mode=title",
        "frames=1",
        "guest_frames_ok=1",
        "guest_submits_ok=1",
        "guest_presents_ok=1",
        "d3dmetal_fallback_count=0",
        "host_d3d12_creation_attempts=0",
        "host_survived=0",
        "victim_survived=0",
        "guest_timed_out=0",
        "host_timed_out=0",
        "refusals=none",
    ],
)
write(
    "orphan",
    "host.log",
    "PROCESS_ORPHAN_SURVIVED pids=4242\n"
    "victory-metal-host: ready port=4570\n",
)

# --- Exit 13 is the launcher's own survivor contract -----------------------
summary(
    "orphan-exit13",
    [
        "status=FAIL",
        "exit_code=13",
        "mode=title",
        "frames=1",
        "guest_frames_ok=1",
        "guest_submits_ok=1",
        "guest_presents_ok=1",
        "d3dmetal_fallback_count=0",
        "host_d3d12_creation_attempts=0",
        "host_survived=0",
        "victim_survived=0",
        "orphan_survived=0",
        "guest_timed_out=0",
        "host_timed_out=0",
        "refusals=none",
    ],
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
        echo "test-boot-census.sh: stderr from $case_name:" >&2
        cat "$TMP/$case_name.err" >&2
        fail "$case_name wrote to stderr"
    fi
}

run_tool clean-gate --out "$TMP/clean" --gate
run_tool clean-json --out "$TMP/clean"
run_tool refusal --out "$TMP/refusal" --gate
run_tool missing --out "$TMP/missing" --gate
run_tool ansi --out "$TMP/ansi" --gate
run_tool fallback --out "$TMP/fallback" --gate
run_tool creation --out "$TMP/creation" --gate
run_tool orphan --out "$TMP/orphan" --gate
run_tool orphan-exit13 --out "$TMP/orphan-exit13" --gate
run_tool nodir --out "$TMP/no-such-directory" --gate
run_tool text --out "$TMP/clean" --text

cmp -s "$TMP/clean-gate.out" "$TMP/clean-json.out" \
    || fail "JSON output is not deterministic across invocations"

# --- Exact assertions ------------------------------------------------------

cat > "$TMP/verify.py" <<'PY'
import json
import os
import sys

root = sys.argv[1]
checks = 0


def fail(message):
    raise SystemExit("test-boot-census.sh: ASSERTION FAILED: " + message)


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
    value = document
    for part in path.split("."):
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


def rc(name, expected):
    global checks
    text = read(name + ".rc").strip()
    checks += 1
    if text != str(expected):
        fail(f"{name} exit: expected {expected}, got {text}")


def contains(name, needle):
    global checks
    text = read(name + ".out")
    checks += 1
    if needle not in text:
        fail(f"{name}.out does not contain {needle!r}")


def no_leak(name, needles):
    global checks
    text = read(name + ".out")
    checks += 1
    for needle in needles:
        if needle in text:
            fail(f"{name}.out leaks {needle!r}")
    checks += 1
    if "\x1b" in text:
        fail(f"{name}.out still contains ANSI escapes")


# --- Clean run -------------------------------------------------------------
clean = load("clean-gate.out")
if set(clean) != {
    "run",
    "invariants",
    "frames",
    "resources",
    "shaders",
    "opcodes",
    "milestones",
    "refusals",
    "gate",
}:
    fail(f"unexpected top-level keys: {sorted(clean)}")
rc("clean-gate", 0)
eq(clean, "run.mode", "gate")
eq(clean, "run.status", "PASS")
eq(clean, "run.exit_code", 0)
eq(clean, "run.guest_exit", 0)
eq(clean, "run.host_exit", 0)
eq(clean, "run.host_terminated_by_launcher", True)
eq(clean, "run.guest_timed_out", False)
eq(clean, "run.host_timed_out", False)
eq(clean, "run.host_survived", False)
eq(clean, "run.victim_survived", False)
eq(clean, "run.orphan_survived", False)
eq(clean, "run.summary_present", True)
eq(clean, "run.telemetry_present", True)
eq(clean, "run.telemetry_valid", True)
eq(clean, "run.guest_log_present", True)
eq(clean, "run.host_log_present", True)

eq(clean, "invariants.d3dmetal_fallback_count", 0)
eq(clean, "invariants.host_d3dmetal_fallback_count", 0)
eq(clean, "invariants.guest_d3dmetal_fallback_count", 0)
eq(clean, "invariants.host_d3d12_creation_attempts", 0)
eq(clean, "invariants.guest_d3dmetal_module", 0)

eq(clean, "frames.presented", 120)
eq(clean, "frames.guest.p50_ms", 11.111)
eq(clean, "frames.guest.p95_ms", 12.222)
eq(clean, "frames.guest.p99_ms", 13.333)
eq(clean, "frames.guest.fps", 89.99)
eq(clean, "frames.guest.samples", 120)
eq(clean, "frames.guest.presented", 120)
eq(clean, "frames.host.p50_ms", 10.5)
eq(clean, "frames.host.p95_ms", 11.5)
eq(clean, "frames.host.p99_ms", 12.5)
eq(clean, "frames.host.fps", 90.0)
eq(clean, "frames.host.samples", 120)
eq(clean, "frames.host.presented", 120)
eq(clean, "frames.host.drawables", 120)
eq(clean, "frames.submissions.ok", 120)
eq(clean, "frames.submissions.refused", 0)
eq(clean, "frames.presents.ok", 120)
eq(clean, "frames.presents.refused", 0)

eq(clean, "resources.buffers.registered", 12)
eq(clean, "resources.buffers.refused", 0)
eq(clean, "resources.textures.registered", 5)
eq(clean, "resources.textures.refused", 0)
eq(clean, "resources.registration_failures", 0)
eq(clean, "resources.refused_sizes.lt_4kib", 0)
eq(clean, "resources.refused_sizes.lt_1mib", 0)
eq(clean, "resources.refused_sizes.lt_64mib", 0)
eq(clean, "resources.refused_sizes.ge_64mib", 0)
eq(clean, "resources.refused_sizes.worst_bytes", None)
eq(clean, "resources.refused_sizes.samples", 0)
eq(clean, "resources.formats_refused", {})

eq(clean, "shaders.compiled", 7)
eq(clean, "shaders.accepted", 7)
eq(clean, "shaders.refused", 0)
eq(clean, "shaders.top_refusals", [])
eq(clean, "shaders.top_refusal_names", [])
eq(clean, "opcodes.histogram", {})
eq(clean, "opcodes.ignored", 0)
eq(clean, "opcodes.refused", 0)
eq(clean, "opcodes.ignored_states", {})
eq(clean, "opcodes.refused_states", {})
eq(clean, "milestones.furthest", "sustained_frames")
eq(
    clean,
    "milestones.reached",
    [
        "host_ready",
        "d3d12x_device",
        "completion_page",
        "dstorage_queue",
        "first_native_submit",
        "first_frame_presented",
        "first_3d_draw",
        "first_compute_dispatch",
        "sustained_frames",
    ],
)
eq(clean, "refusals.top", [])
eq(clean, "refusals.total", 0)
eq(clean, "refusals.first_reason", None)
eq(clean, "gate.ok", True)
eq(clean, "gate.failures", [])

# --- Refusal-heavy run -----------------------------------------------------
refusal = load("refusal.out")
rc("refusal", 0)
eq(refusal, "run.status", "FAIL")
eq(refusal, "run.exit_code", 12)
eq(refusal, "frames.presented", 35)
eq(refusal, "frames.guest.p50_ms", 20.0)
eq(refusal, "frames.host.p50_ms", 21.0)
eq(refusal, "frames.submissions.ok", 30)
eq(refusal, "frames.submissions.refused", 10)
eq(refusal, "frames.presents.ok", 32)
eq(refusal, "frames.presents.refused", 3)
eq(refusal, "resources.buffers.registered", 4)
eq(refusal, "resources.buffers.refused", 3)
eq(refusal, "resources.textures.registered", 1)
eq(refusal, "resources.textures.refused", 2)
eq(refusal, "resources.registration_failures", 5)
eq(refusal, "resources.refused_sizes.lt_4kib", 2)
eq(refusal, "resources.refused_sizes.lt_1mib", 1)
eq(refusal, "resources.refused_sizes.lt_64mib", 1)
eq(refusal, "resources.refused_sizes.ge_64mib", 1)
eq(refusal, "resources.refused_sizes.worst_bytes", 134217728)
eq(refusal, "resources.refused_sizes.samples", 5)
eq(refusal, "resources.formats_refused", {"28": 3, "87": 2})
eq(refusal, "shaders.compiled", 9)
eq(refusal, "shaders.accepted", 2)
eq(refusal, "shaders.refused", 7)
eq(
    refusal,
    "shaders.top_refusals",
    [
        {"name": "vs_main", "count": 2},
        {"name": "ps_lit", "count": 1},
    ],
)
eq(refusal, "shaders.top_refusal_names", ["vs_main", "ps_lit"])
eq(
    refusal,
    "opcodes.histogram",
    {"dispatch_3d": 1, "draw_indexed": 6, "set_pipeline": 3},
)
eq(refusal, "opcodes.ignored", 3)
eq(refusal, "opcodes.refused", 4)
eq(refusal, "opcodes.ignored_states", {"blend": 2})
eq(refusal, "opcodes.refused_states", {"rasterizer": 1})
eq(
    refusal,
    "refusals.top",
    [
        {"name": "unsupported_format", "count": 5},
        {"name": "missing_binding", "count": 3},
        {"name": "format", "count": 2},
        {"name": "other", "count": 1},
        {"name": "pipeline", "count": 1},
    ],
)
eq(refusal, "refusals.total", 12)
eq(refusal, "refusals.first_reason", "texture format 28 is not supported")
eq(refusal, "milestones.furthest", "sustained_frames")
eq(
    refusal,
    "milestones.reached",
    [
        "host_ready",
        "d3d12x_device",
        "completion_page",
        "first_native_submit",
        "first_frame_presented",
        "first_3d_draw",
        "first_compute_dispatch",
        "sustained_frames",
    ],
)
eq(refusal, "gate.ok", True)
eq(refusal, "gate.failures", [])

# --- Missing telemetry -----------------------------------------------------
missing = load("missing.out")
rc("missing", 0)
eq(missing, "run.telemetry_present", False)
eq(missing, "run.telemetry_valid", False)
eq(missing, "run.guest_log_present", False)
eq(missing, "run.host_log_present", False)
eq(missing, "invariants.d3dmetal_fallback_count", None)
eq(missing, "invariants.host_d3dmetal_fallback_count", None)
eq(missing, "invariants.guest_d3dmetal_fallback_count", None)
eq(missing, "invariants.host_d3d12_creation_attempts", None)
eq(missing, "invariants.guest_d3dmetal_module", None)
eq(missing, "frames.presented", 3)
eq(missing, "frames.guest.p50_ms", None)
eq(missing, "frames.guest.p95_ms", None)
eq(missing, "frames.guest.p99_ms", None)
eq(missing, "frames.guest.fps", None)
eq(missing, "frames.host.p50_ms", None)
eq(missing, "frames.host.samples", None)
eq(missing, "frames.submissions.ok", 1)
eq(missing, "frames.presents.ok", 1)
eq(missing, "frames.presents.refused", 2)
eq(missing, "resources.buffers.registered", None)
eq(missing, "resources.textures.registered", None)
eq(missing, "resources.formats_refused", {})
eq(missing, "shaders.compiled", None)
eq(missing, "shaders.accepted", None)
eq(missing, "shaders.refused", None)
eq(missing, "opcodes.histogram", {})
eq(missing, "opcodes.ignored", None)
eq(missing, "opcodes.refused", None)
eq(missing, "milestones.furthest", "first_frame_presented")
eq(
    missing,
    "milestones.reached",
    ["first_native_submit", "first_frame_presented"],
)
eq(missing, "refusals.total", 0)
eq(missing, "refusals.first_reason", None)
eq(missing, "gate.ok", True)

# --- ANSI and truncated log ------------------------------------------------
ansi = load("ansi.out")
rc("ansi", 0)
eq(ansi, "run.status", "PASS")
eq(ansi, "run.mode", "title")
eq(ansi, "run.telemetry_present", True)
eq(ansi, "run.telemetry_valid", False)
eq(ansi, "invariants.d3dmetal_fallback_count", 0)
eq(ansi, "invariants.host_d3d12_creation_attempts", 0)
eq(ansi, "frames.guest.p50_ms", 1.5)
eq(ansi, "frames.host.p50_ms", None)
eq(ansi, "resources.refused_sizes.lt_1mib", 1)
eq(ansi, "resources.refused_sizes.lt_64mib", 1)
eq(ansi, "resources.refused_sizes.worst_bytes", 8388608)
eq(ansi, "resources.refused_sizes.samples", 2)
eq(ansi, "resources.formats_refused", {"28": 3})
eq(ansi, "refusals.first_reason", "incomplete frame <path>")
eq(ansi, "gate.ok", True)

# --- Fallback gate ---------------------------------------------------------
fallback = load("fallback.out")
rc("fallback", 2)
eq(fallback, "invariants.d3dmetal_fallback_count", 1)
eq(fallback, "invariants.host_d3dmetal_fallback_count", 1)
eq(fallback, "invariants.guest_d3dmetal_fallback_count", 0)
eq(fallback, "gate.ok", False)
eq(fallback, "gate.failures", ["d3dmetal_fallback"])

# --- Host D3D12 creation gate ---------------------------------------------
creation = load("creation.out")
rc("creation", 2)
eq(creation, "invariants.host_d3d12_creation_attempts", 1)
eq(creation, "invariants.d3dmetal_fallback_count", 0)
eq(creation, "resources.buffers.registered", 6)
eq(creation, "shaders.accepted", 3)
eq(creation, "gate.ok", False)
eq(creation, "gate.failures", ["host_d3d12_creation"])

# --- Orphan gate -----------------------------------------------------------
orphan = load("orphan.out")
rc("orphan", 2)
eq(orphan, "run.host_survived", False)
eq(orphan, "run.victim_survived", False)
eq(orphan, "run.orphan_survived", True)
eq(orphan, "gate.ok", False)
eq(orphan, "gate.failures", ["orphan_survived"])

orphan13 = load("orphan-exit13.out")
rc("orphan-exit13", 2)
eq(orphan13, "run.exit_code", 13)
eq(orphan13, "run.orphan_survived", True)
eq(orphan13, "gate.ok", False)
eq(orphan13, "gate.failures", ["orphan_survived"])

# --- Missing directory -----------------------------------------------------
nodir = load("nodir.out")
rc("nodir", 0)
eq(nodir, "run.summary_present", False)
eq(nodir, "run.telemetry_present", False)
eq(nodir, "run.guest_log_present", False)
eq(nodir, "run.host_log_present", False)
eq(nodir, "run.mode", None)
eq(nodir, "invariants.d3dmetal_fallback_count", None)
eq(nodir, "invariants.host_d3d12_creation_attempts", None)
eq(nodir, "frames.presented", None)
eq(nodir, "frames.guest.p50_ms", None)
eq(nodir, "resources.refused_sizes.worst_bytes", None)
eq(nodir, "resources.formats_refused", {})
eq(nodir, "milestones.furthest", None)
eq(nodir, "milestones.reached", [])
eq(nodir, "refusals.top", [])
eq(nodir, "refusals.total", None)
eq(nodir, "gate.ok", True)

# --- Text mode -------------------------------------------------------------
contains("text", "furthest=sustained_frames")
contains("text", "gate ok=true failures=none")
contains("text", "refusals total=0")

# --- No path, title or ANSI leak in any output -----------------------------
leaks = [
    "SecretTitle",
    "secret-title.exe",
    "C:\\Games",
    "My Title",
    "/abs/fixture",
    root,
]
for name in (
    "clean-gate",
    "clean-json",
    "refusal",
    "missing",
    "ansi",
    "fallback",
    "creation",
    "orphan",
    "orphan-exit13",
    "nodir",
    "text",
):
    no_leak(name, leaks)

print(f"checks={checks}")
PY

checks=$(python3 "$TMP/verify.py" "$TMP")
echo "test-boot-census.sh: PASS ($checks)"
