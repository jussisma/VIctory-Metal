#!/bin/sh
# C10: hermetic self-test for the live-title launchers' session hygiene and
# backend selection.
#
# No Wine, no GPU, no title, no network, no Rust build. `TMPDIR` points at a
# private mktemp directory and `VICTORY_LAUNCHER_RUNNER` points at a stub, so
# the wrapper's own behaviour is exercised without touching the delegate's
# daemon. The stub mimics the delegate's contract: it creates one
# `victory-title.*` session directory and removes it unless it is asked to keep
# it. The suite proves:
#   1. a failing delegated run (exit 10) leaves zero new `victory-title.*` and
#      `victory-metal-out.*` directories, and exit 10 is propagated unchanged;
#   2. `--keep-session` leaves exactly one of each and prints both paths;
#   3. any backend other than metal/d3d12 is a usage error (exit 2) that creates
#      nothing, in both the wrapper and run-title-metal.sh;
#   4. the default backend (metal) and a pre-set `VICTORY_BACKEND=d3d12` are
#      visible in the dry run, the summary and the delegate's
#      `LAUNCH_ENV backend=` line;
#   5. the historical keep aliases (`--keep`, `VICTORY_KEEP_LOGS=1`) keep the
#      same pair of directories;
#   6. a setup failure after the wrapper's diagnostics directory exists (a
#      missing --exe) still removes that directory, and a caller-supplied
#      --out is never removed;
#   7. the real run-title-metal.sh removes its own session directory after a
#      guest failure, a host setup failure that happens after WORK exists, the
#      internal guest timeout and a signal whose cleanup is interrupted by a
#      second signal (driven with a private GPU lock and WINE stubs);
#   8. ADR-038: the host-only converter variables reach victory_metal_host and
#      never the Wine guest even when they are exported by the caller;
#   9. `sh -n` parses both launchers, the hygiene helper and this test.
#
# Usage: sh tools/macos/metal-poc/test-launcher-session.sh
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WRAPPER="$ROOT/tools/macos/victory-metal.sh"
RUNNER="$HERE/run-title-metal.sh"

BASE="$(mktemp -d "${TMPDIR:-/tmp}/victory-launcher-session-test.XXXXXXXX")"
TMPDIR="$BASE/sessions"
mkdir -p "$TMPDIR"
export TMPDIR

PASSED=0
LAST_OUT=""
LAST_RC=0

cleanup_test() {
    rm -rf "$BASE"
}
trap cleanup_test EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

fail() {
    echo "FAIL: test-launcher-session: $1" >&2
    if [ -n "${LAST_OUT:-}" ] && [ -f "$LAST_OUT" ]; then
        echo "---- launcher output ($LAST_OUT) ----" >&2
        cat "$LAST_OUT" >&2 || true
        echo "-------------------------------------" >&2
    fi
    exit 1
}

pass() {
    PASSED=$((PASSED + 1))
    echo "PASS: $1"
}

[ -f "$WRAPPER" ] || fail "missing $WRAPPER"
[ -x "$WRAPPER" ] || fail "$WRAPPER is not executable"
[ -f "$RUNNER" ] || fail "missing $RUNNER"

# --- Private build target and a stub delegate -----------------------------
# `victory-metal.sh` only calls `build_one` when a binary is missing or older
# than the sources. Pre-creating the three expected binaries (now, hence newer
# than the checkout) keeps the test off cargo entirely.
TARGET="$BASE/target"
export CARGO_TARGET_DIR="$TARGET"
mkdir -p "$TARGET/x86_64-pc-windows-gnullvm/diag" \
    "$TARGET/aarch64-apple-darwin/debug" \
    "$TARGET/x86_64-apple-darwin/debug"
for stub_bin in \
    "$TARGET/x86_64-pc-windows-gnullvm/diag/victory.exe" \
    "$TARGET/aarch64-apple-darwin/debug/victory_metal_host" \
    "$TARGET/x86_64-apple-darwin/debug/victory-irconverter"; do
    printf '#!/bin/sh\nexit 0\n' >"$stub_bin"
    chmod +x "$stub_bin"
done
: >"$BASE/guest.exe"

STUB="$BASE/stub-runner.sh"
cat >"$STUB" <<'EOS'
#!/bin/sh
# Stand-in for run-title-metal.sh: same keep/backend contract, no daemon, no
# Wine, no GPU. It records the backend it received and removes its own session
# unless --keep-session/--keep was passed (the real delegate's guarantee).
set -eu
out=""
keep=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)
            out="$2"
            shift 2
            ;;
        --keep-session|--keep)
            keep=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done
if [ -n "${STUB_BACKEND_LOG:-}" ]; then
    printf '%s\n' "${VICTORY_BACKEND:-unset}" >"$STUB_BACKEND_LOG"
fi
printf 'LAUNCH_ENV backend=%s\n' "${VICTORY_BACKEND:-unset}"
printf 'LAUNCH_ENV endpoint=none\n'
printf 'LAUNCH_ENV share_dir=stub-share\n'
printf 'LAUNCH_ENV fence_page=stub-fence\n'
session="$(mktemp -d "${TMPDIR:-/tmp}/victory-title.XXXXXXXX")"
printf 'stub session\n' >"$session/guest.log"
if [ -n "$out" ]; then
    printf 'stub delegate out=%s\n' "$out" >"$out/delegate.txt" || true
fi
if [ "$keep" = 1 ]; then
    printf 'kept session files in %s\n' "$session"
else
    rm -rf -- "$session"
fi
exit "${STUB_RC:-0}"
EOS
chmod +x "$STUB"

export VICTORY_LAUNCHER_RUNNER="$STUB"
unset VICTORY_BACKEND
unset VICTORY_KEEP_LOGS
unset VICTORY_KEEP_SESSION
STUB_RC=0
STUB_BACKEND_LOG="$BASE/backend.txt"
export STUB_RC STUB_BACKEND_LOG

# Count the launcher-owned directories currently under the private TMPDIR.
count_dirs() {
    find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d -name "$1" | wc -l | tr -d ' '
}

count_entries() {
    find "$TMPDIR" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' '
}

reset_tmp() {
    find "$TMPDIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

# One wrapper invocation. Sets LAST_RC and LAST_OUT; never trips `set -e`.
run_wrapper() {
    last_label="$1"
    shift
    LAST_OUT="$BASE/$last_label.out"
    LAST_RC=0
    sh "$WRAPPER" "$@" >"$LAST_OUT" 2>&1 || LAST_RC=$?
}

expect_rc() {
    [ "$LAST_RC" -eq "$1" ] || fail "$2: exit $LAST_RC, expected $1"
}

expect_zero_sessions() {
    title_count="$(count_dirs 'victory-title.*')"
    out_count="$(count_dirs 'victory-metal-out.*')"
    [ "$title_count" -eq 0 ] || fail "$1: $title_count victory-title.* directories left behind"
    [ "$out_count" -eq 0 ] || fail "$1: $out_count victory-metal-out.* directories left behind"
}

expect_one_session_pair() {
    [ "$(count_dirs 'victory-title.*')" -eq 1 ] \
        || fail "$1: $(count_dirs 'victory-title.*') victory-title.* directories, expected 1"
    [ "$(count_dirs 'victory-metal-out.*')" -eq 1 ] \
        || fail "$1: $(count_dirs 'victory-metal-out.*') victory-metal-out.* directories, expected 1"
}

grep_out() {
    grep -q -- "$1" "$LAST_OUT" || fail "$2: '$1' missing from the output"
}

# --- 1. Default backend, failing run, nothing left behind ------------------
STUB_RC=10
STUB_BACKEND_LOG="$BASE/backend-default.txt"
export STUB_RC STUB_BACKEND_LOG
run_wrapper default --exe "$BASE/guest.exe" -- flag
expect_rc 10 "default failing run"
expect_zero_sessions "default failing run"
[ "$(count_entries)" -eq 0 ] || fail "default failing run created entries under TMPDIR"
grep_out 'LAUNCH_ENV backend=metal' "default failing run"
grep_out '  backend=metal' "default failing run"
grep_out 'removed the session directory' "default failing run"
[ "$(cat "$BASE/backend-default.txt")" = metal ] \
    || fail "the delegate received backend '$(cat "$BASE/backend-default.txt")', expected metal"
pass "default metal run: exit 10 propagated, no session or out directory left"

# --- 2. --keep-session keeps exactly one of each ---------------------------
STUB_RC=10
STUB_BACKEND_LOG="$BASE/backend-keep.txt"
export STUB_RC STUB_BACKEND_LOG
run_wrapper keep --exe "$BASE/guest.exe" --keep-session -- flag
expect_rc 10 "--keep-session run"
expect_one_session_pair "--keep-session run"
grep_out '^kept session files in ' "--keep-session run"
grep_out '^victory-metal.sh: kept session files in ' "--keep-session run"
kept_out="$(sed -n 's/^victory-metal.sh: kept session files in //p' "$LAST_OUT" | tail -n 1)"
[ -n "$kept_out" ] && [ -d "$kept_out" ] \
    || fail "--keep-session printed an out directory that does not exist: '$kept_out'"
grep_out '  out='"$kept_out" "--keep-session run"
reset_tmp
pass "--keep-session left exactly one session and one out directory and printed both"

# --- 3. Unknown backend is a usage error in both launchers -----------------
VICTORY_BACKEND=vulkan
export VICTORY_BACKEND
run_wrapper bad-backend --exe "$BASE/guest.exe" -- flag
expect_rc 2 "unknown backend (wrapper)"
expect_zero_sessions "unknown backend (wrapper)"
[ "$(count_entries)" -eq 0 ] || fail "a usage error created entries under TMPDIR"
grep_out 'unsupported VICTORY_BACKEND=vulkan' "unknown backend (wrapper)"

set +e
VICTORY_BACKEND=vulkan sh "$RUNNER" --headless -- /nonexistent-guest.exe \
    >"$BASE/run-title-bad-backend.out" 2>&1
bad_runner_rc=$?
set -e
[ "$bad_runner_rc" -eq 2 ] \
    || fail "run-title-metal.sh accepted VICTORY_BACKEND=vulkan (exit $bad_runner_rc)"
grep -q 'unsupported VICTORY_BACKEND=vulkan' "$BASE/run-title-bad-backend.out" \
    || fail "run-title-metal.sh did not name the rejected backend"
expect_zero_sessions "unknown backend (run-title-metal.sh)"
unset VICTORY_BACKEND
pass "unknown backend is exit 2 in both launchers and creates nothing"

# --- 4. Backend visibility: default, dry run and d3d12 ---------------------
run_wrapper dry-run-metal --dry-run --exe "$BASE/guest.exe" -- flag
expect_rc 0 "metal dry run"
grep_out '  VICTORY_BACKEND=metal' "metal dry run"
grep_out '  backend=metal' "metal dry run"
grep_out '  runner='"$STUB" "metal dry run"
expect_zero_sessions "metal dry run"

export VICTORY_BACKEND=d3d12
STUB_BACKEND_LOG="$BASE/backend-d3d12.txt"
export STUB_BACKEND_LOG
run_wrapper dry-run-d3d12 --dry-run --exe "$BASE/guest.exe" -- flag
expect_rc 0 "d3d12 dry run"
grep_out '  VICTORY_BACKEND=d3d12' "d3d12 dry run"
grep_out '  backend=d3d12' "d3d12 dry run"

STUB_RC=0
export STUB_RC
run_wrapper d3d12 --exe "$BASE/guest.exe" -- flag
expect_rc 0 "d3d12 run"
grep_out 'LAUNCH_ENV backend=d3d12' "d3d12 run"
grep_out '  backend=d3d12' "d3d12 run"
[ "$(cat "$BASE/backend-d3d12.txt")" = d3d12 ] \
    || fail "the delegate received backend '$(cat "$BASE/backend-d3d12.txt")', expected d3d12"
expect_zero_sessions "d3d12 run"
unset VICTORY_BACKEND
pass "backend resolves to metal by default and d3d12 when pre-set, visibly everywhere"

# --- 5. Historical keep aliases -------------------------------------------
STUB_RC=10
export STUB_RC
run_wrapper keep-flag --exe "$BASE/guest.exe" --keep -- flag
expect_rc 10 "keep alias --keep"
expect_one_session_pair "keep alias --keep"
reset_tmp

VICTORY_KEEP_LOGS=1
export VICTORY_KEEP_LOGS
run_wrapper keep-env --exe "$BASE/guest.exe" -- flag
expect_rc 10 "keep alias VICTORY_KEEP_LOGS=1"
expect_one_session_pair "keep alias VICTORY_KEEP_LOGS=1"
unset VICTORY_KEEP_LOGS
reset_tmp
pass "--keep and VICTORY_KEEP_LOGS=1 keep the same directories as --keep-session"

# --- 6. Setup failure after the out dir exists; caller --out is preserved --
run_wrapper missing-guest --exe "$BASE/does-not-exist.exe" -- flag
expect_rc 2 "missing explicit guest"
expect_zero_sessions "missing explicit guest"
[ "$(count_entries)" -eq 0 ] || fail "a setup failure created entries under TMPDIR"

CALLER_OUT="$BASE/caller-out"
mkdir -p "$CALLER_OUT"
STUB_RC=10
export STUB_RC
run_wrapper caller-out --exe "$BASE/guest.exe" --out "$CALLER_OUT" -- flag
expect_rc 10 "caller --out run"
[ -d "$CALLER_OUT" ] || fail "a caller-supplied --out directory was removed"
[ -f "$CALLER_OUT/delegate.txt" ] || fail "the delegate did not write into the caller --out directory"
expect_zero_sessions "caller --out run"
pass "setup failure cleans the owned out dir; a caller-supplied --out is never removed"

# --- 7. The real launcher's own exit paths (no Wine, no GPU) --------------
# `run-title-metal.sh` is driven with a private GPU lock, a private TARGET_DIR
# whose victory_metal_host exits immediately, and WINE pointing at a stub (or
# /usr/bin/false). No daemon reports ready, no guest is a Win64 title and no GPU is
# touched. These runs prove the session directory is removed after a guest
# failure, a setup failure that happens after WORK exists, the internal guest
# timeout, and a signal whose cleanup is interrupted by a second signal.
mkdir -p "$BASE/locks"
export VICTORY_GPU_LOCK="$BASE/locks/gpu.lock"
export VICTORY_HYGIENE_GRACE=1

run_runner() {
    last_label="$1"
    shift
    LAST_OUT="$BASE/$last_label.out"
    LAST_RC=0
    sh "$RUNNER" "$@" >"$LAST_OUT" 2>&1 || LAST_RC=$?
}

# A guest failure after WORK exists: d3d12 mode starts no daemon, WINE=/usr/bin/false
# exits 1 and the summary must report exit 10, not host telemetry exit 15.
VICTORY_BACKEND=d3d12
WINE=/usr/bin/false
export VICTORY_BACKEND WINE
run_runner real-guest-failure --headless -- /usr/bin/false
expect_rc 10 "real guest failure"
expect_zero_sessions "real guest failure"
[ "$(count_entries)" -eq 0 ] || fail "a real guest failure left entries under TMPDIR"
grep_out '^backend=d3d12' "real guest failure"
grep_out '^guest_exit=1' "real guest failure"
grep_out '^exit_code=10' "real guest failure"
grep_out 'removed the session directory' "real guest failure"

# A setup failure after WORK exists: metal mode starts the stub host, which
# never prints the ready line, so the launcher exits 3 after creating WORK.
VICTORY_BACKEND=metal
export VICTORY_BACKEND
run_runner real-host-not-ready --headless -- /usr/bin/false
expect_rc 3 "real host-not-ready failure"
expect_zero_sessions "real host-not-ready failure"
[ "$(count_entries)" -eq 0 ] || fail "a real setup failure left entries under TMPDIR"
grep_out 'the host daemon did not become ready' "real host-not-ready failure"
grep_out 'removed the session directory' "real host-not-ready failure"

# The internal guest timeout (the launcher maps it to exit 10; the wrapper's
# outer timeout(1) guard is the only source of 124).
cat >"$BASE/wine-slow.sh" <<'EOS'
#!/bin/sh
exec sleep 30
EOS
chmod +x "$BASE/wine-slow.sh"
VICTORY_BACKEND=d3d12
WINE="$BASE/wine-slow.sh"
GUEST_TIMEOUT=1
export VICTORY_BACKEND WINE GUEST_TIMEOUT
run_runner real-timeout --headless -- flag
expect_rc 10 "real internal timeout"
grep_out 'guest exceeded the launcher timeout' "real internal timeout"
expect_zero_sessions "real internal timeout"

# A signal during teardown, followed by a second one while cleanup is reaping a
# TERM-ignoring guest. The cleanup must ignore the second signal and still
# remove the session; with a default signal disposition it would be cut short.
cat >"$BASE/wine-stubborn.sh" <<'EOS'
#!/bin/sh
trap '' TERM
while :; do
    sleep 0.05
done
EOS
chmod +x "$BASE/wine-stubborn.sh"
WINE="$BASE/wine-stubborn.sh"
unset GUEST_TIMEOUT
export WINE
LAST_OUT="$BASE/real-signal.out"
LAST_RC=0
sh "$RUNNER" --headless -- /usr/bin/false >"$LAST_OUT" 2>&1 &
signal_pid=$!
sleep 0.5
kill -TERM "$signal_pid" 2>/dev/null || true
sleep 0.4
kill -TERM "$signal_pid" 2>/dev/null || true
wait "$signal_pid" || LAST_RC=$?
expect_rc 143 "real signal"
expect_zero_sessions "real signal"
[ "$(count_entries)" -eq 0 ] || fail "a signalled real run left entries under TMPDIR"
grep_out 'removed the session directory' "real signal"
unset VICTORY_GPU_LOCK VICTORY_HYGIENE_GRACE VICTORY_BACKEND WINE
reset_tmp
pass "real launcher: guest failure, host setup failure, timeout and double signal all remove the session"

# --- 8. ADR-038: converter paths reach the host, never the guest -----------
# `victory-metal.sh` exports the macOS converter worker and dylib for
# `victory_metal_host` (ADR-038). The launcher must keep them for the host and
# unset them before Wine: a guest that inherits a macOS worker path builds a
# pool whose child cannot start and refuses the pipeline as `Translation`
# instead of `ConverterUnavailable`, so the VP3 host-translation path is never
# reached. The recording host stub proves it saw both variables; the recording
# WINE stub proves the guest saw neither. No daemon, no GPU, no title.
mkdir -p "$BASE/locks"
VICTORY_GPU_LOCK="$BASE/locks/gpu.lock"
VICTORY_HYGIENE_GRACE=1
export VICTORY_GPU_LOCK VICTORY_HYGIENE_GRACE

HOST_ENV_OUT="$BASE/host-converter-env.txt"
GUEST_ENV_OUT="$BASE/guest-converter-env.txt"
WORKER_STUB="$BASE/irconverter-stub"
CONV_STUB="$BASE/libmetalirconverter.dylib"
printf '#!/bin/sh\nexit 0\n' >"$WORKER_STUB"
chmod +x "$WORKER_STUB"
: >"$CONV_STUB"

cat >"$TARGET/aarch64-apple-darwin/debug/victory_metal_host" <<EOS
#!/bin/sh
# Recording stand-in for victory_metal_host: captures the ADR-038 converter
# environment, writes minimal clean telemetry and reports ready.
telemetry=""
while [ "\$#" -gt 0 ]; do
    case "\$1" in
        --telemetry-json)
            telemetry="\$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
printf 'worker=%s\nconverter=%s\n' "\${VICTORY_IRCONVERTER_WORKER:-}" "\${VICTORY_METAL_IRCONVERTER:-}" >"$HOST_ENV_OUT"
if [ -n "\$telemetry" ]; then
    printf '{"d3dmetal_fallback_count":0,"d3dmetal_images":[],"d3dmetal_loaded":false}\n' >"\$telemetry"
fi
echo 'victory-metal-host: ready port=1'
exit 0
EOS
chmod +x "$TARGET/aarch64-apple-darwin/debug/victory_metal_host"

cat >"$BASE/wine-record-env.sh" <<EOS
#!/bin/sh
printf 'worker=%s\nconverter=%s\n' "\${VICTORY_IRCONVERTER_WORKER:-}" "\${VICTORY_METAL_IRCONVERTER:-}" >"$GUEST_ENV_OUT"
exit 0
EOS
chmod +x "$BASE/wine-record-env.sh"

VICTORY_IRCONVERTER_WORKER="$WORKER_STUB"
VICTORY_METAL_IRCONVERTER="$CONV_STUB"
VICTORY_BACKEND=metal
WINE="$BASE/wine-record-env.sh"
export VICTORY_IRCONVERTER_WORKER VICTORY_METAL_IRCONVERTER VICTORY_BACKEND WINE
run_runner real-converter-env --headless -- "$BASE/guest.exe"
expect_rc 0 "converter environment split"
grep -Fqx -- "worker=$WORKER_STUB" "$HOST_ENV_OUT" \
    || fail "victory_metal_host did not inherit VICTORY_IRCONVERTER_WORKER"
grep -Fqx -- "converter=$CONV_STUB" "$HOST_ENV_OUT" \
    || fail "victory_metal_host did not inherit VICTORY_METAL_IRCONVERTER"
grep -Fqx -- 'worker=' "$GUEST_ENV_OUT" \
    || fail "the Wine guest inherited VICTORY_IRCONVERTER_WORKER"
grep -Fqx -- 'converter=' "$GUEST_ENV_OUT" \
    || fail "the Wine guest inherited VICTORY_METAL_IRCONVERTER"
expect_zero_sessions "converter environment split"
unset VICTORY_IRCONVERTER_WORKER VICTORY_METAL_IRCONVERTER VICTORY_BACKEND
unset VICTORY_GPU_LOCK VICTORY_HYGIENE_GRACE WINE
reset_tmp
pass "ADR-038 converter variables reach the host and never the guest"

# --- 9. Every launcher and helper parses under `sh -n` --------------------
for script in "$WRAPPER" "$RUNNER" "$HERE/process-hygiene.sh" "$0"; do
    sh -n "$script" || fail "sh -n failed: $script"
done
pass "sh -n parsed both launchers, the hygiene helper and this test"

echo "PASS: test-launcher-session ($PASSED checks)"
