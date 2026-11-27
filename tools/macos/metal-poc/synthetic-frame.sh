#!/bin/sh
# Production-path Wire v3 gate: a real Win64 Wine guest and the native ARM64
# Metal host.
#
# Two parts, run one after the other on an exclusive GPU:
#
# 1. Guest ring (`metal-worker --guest-v3`): a real Win64 process creates a
#    file-backed allocation in the session share directory, authenticates to
#    the native host, registers the allocation, and sends Wire v3
#    (`KIND_SUBMISSION_V3`) compute lists over loopback. It asserts:
#      * the alias-visible compute result,
#      * a replay of the same list identity is refused as AlreadyExecuted and
#        the alias is byte-identical,
#      * an unknown allocation is UnknownHandle,
#      * an unsupported opcode is an explicit ExecutionFailed,
#      * no D3DMetal module is loaded in the guest process (psapi module scan).
#
# 2. Host gate (`producer_e2e`): the real `submit_v3` mixed encoder path with
#    MRT + depth graphics into IOSurface targets, exact pixels, host present,
#    and a Rosetta-converted DXIL compute shader whose output is read back
#    through the guest-visible allocation.
#
# Known missing links (reported as PARTIAL, never as pass):
#   * `victory-gpu`'s native queue installs `DisconnectedSink`; the producer does
#     not yet carry its v3 submissions over IPC.
#   * `victory_metal::guest_ipc::GuestIpcClient` has no version 3 encoder, so the
#     guest probe frames the same bytes itself.
#   * `ipc_host` cannot register render-target textures, so a v3 BeginRenderPass
#     over the wire is refused UnknownHandle; the graphics pass runs through
#     `DirectMetalTransport::submit_v3` directly.
#   * there is no graphics pipeline constructor from converted DXIL metallibs,
#     and `submit_v3` does not build the IRConverter top-level argument buffer,
#     so the graphics pass is MSL and the DXIL compute uses the verified MSC
#     single-UAV binder (`dispatch_msc_uav_1d`).
#
# Usage: tools/macos/metal-poc/synthetic-frame.sh
#
# The launcher creates a private temporary share directory, 32-byte secret,
# Wine prefix and native host, and removes them on exit (KEEP=1 or
# VICTORY_KEEP_LOGS=1 keeps them but still reaps every child).
# Nothing here launches the title. This is one frame, not a benchmark.
#
# Process hygiene (C7 / Wave 7A): the native host and the Wine guest are started
# in the background and registered with `process-hygiene.sh`; the EXIT trap
# reaps them with TERM, a bounded wait and KILL, then exits 13 if a started PID
# survived cleanup (0 = PASS, 1/2 = named failure, 13 = orphan survivor).
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SELF_DIR/process-hygiene.sh"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; put it first like the other macOS harnesses.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$ROOT/../VIctory-macOS/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
export WINE

IRCONVERTER_LIB="${VICTORY_METAL_IRCONVERTER:-$(cd "$(dirname "${WINE:-/usr/local/bin/wine}")/.." 2>/dev/null && pwd)/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib}"
export VICTORY_METAL_IRCONVERTER="$IRCONVERTER_LIB"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-frame-gate.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
HOST_LOG="$WORK/host.stdout"
HOST_ERR="$WORK/host.stderr"
GUEST_LOG="$WORK/guest.log"
HOST_PID=""
GUEST_PID=""
PORT=""
COMPUTE_PIPELINE=""
victory_hygiene_work "$WORK"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    victory_hygiene_reap
    victory_hygiene_reap_marker "$WORK"
    echo "PROCESS_CHECK host=0 guest=0 survivors=${VICTORY_HYGIENE_SURVIVORS:-0}"
    victory_hygiene_remove_work
    victory_hygiene_finalize "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

mkdir -m 700 "$SHARE"
python3 -c 'import secrets; print(secrets.token_hex(32))' >"$SECRET"
chmod 600 "$SECRET"

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

if [ -z "${WINE:-}" ] || [ ! -x "$WINE" ]; then
    echo "FAIL: no Wine executable found (set WINE=...)" >&2
    exit 1
fi
[ -f "$IRCONVERTER_LIB" ] || {
    echo "FAIL: missing libmetalirconverter.dylib at $IRCONVERTER_LIB" >&2
    exit 1
}

echo "Building the native ARM64 host worker, the x86_64 IRConverter worker and the Win64 guest probe..."
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin metal-worker
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target x86_64-apple-darwin --bin victory-irconverter
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target x86_64-pc-windows-gnullvm --bin metal-worker
WORKER="$ROOT/target/aarch64-apple-darwin/debug/metal-worker"
IRCONVERTER="$ROOT/target/x86_64-apple-darwin/debug/victory-irconverter"
GUEST="$ROOT/target/x86_64-pc-windows-gnullvm/debug/metal-worker.exe"
for artifact in "$WORKER" "$IRCONVERTER" "$GUEST"; do
    [ -x "$artifact" ] || { echo "FAIL: $artifact is missing" >&2; exit 1; }
done
export VICTORY_IRCONVERTER_WORKER="$IRCONVERTER"

# --- Part 1: Win64 guest drives the v3 ring ---------------------------------
: >"$HOST_LOG"
: >"$HOST_ERR"
"$WORKER" \
    --secret-file "$SECRET" \
    --share-dir "$SHARE" \
    --port 0 \
    --frame-gate \
    --io-timeout-secs 120 \
    --accept-timeout-secs 180 \
    >"$HOST_LOG" 2>"$HOST_ERR" &
HOST_PID=$!
victory_hygiene_track "$HOST_PID"

ready=0
for _ in $(seq 1 600); do
    if grep -q '^VICTORY_IPC_READY=1$' "$HOST_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$HOST_PID" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
if [ "$ready" -ne 1 ]; then
    echo "FAIL: the native host did not become ready" >&2
    cat "$HOST_LOG" "$HOST_ERR" >&2 || true
    exit 1
fi

PORT="$(sed -n 's/^VICTORY_IPC_PORT=\([0-9][0-9]*\)$/\1/p' "$HOST_LOG" | tail -1)"
COMPUTE_PIPELINE="$(sed -n 's/^VICTORY_PIPELINE_ID=\([0-9][0-9]*\)$/\1/p' "$HOST_LOG" | tail -1)"
[ -n "$PORT" ] || { echo "FAIL: no host port was published" >&2; exit 1; }
[ -n "$COMPUTE_PIPELINE" ] || { echo "FAIL: no compute pipeline was published" >&2; exit 1; }

export WINEPREFIX="$WORK/wineprefix"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_IPC_PORT="$PORT"
export VICTORY_IPC_PIPELINE_ID="$COMPUTE_PIPELINE"
export VICTORY_IPC_SECRET_FILE="$(winpath "$SECRET")"

"$WINE" wineboot -i >/dev/null 2>&1 || true
echo "Host ready: port=$PORT compute=$COMPUTE_PIPELINE"
echo "Running the Wine Win64 guest v3 probe..."
guest_rc=0
"$WINE" "$GUEST" --guest-v3 >"$GUEST_LOG" 2>&1 &
GUEST_PID=$!
victory_hygiene_track "$GUEST_PID"
victory_hygiene_wait_pid "$GUEST_PID" 600 || guest_rc=$?
victory_hygiene_untrack "$GUEST_PID"
GUEST_PID=""
cat "$GUEST_LOG" 2>/dev/null || true
if [ "$guest_rc" -ne 0 ]; then
    echo "FAIL: the Wine guest probe exited with $guest_rc" >&2
    cat "$HOST_LOG" "$HOST_ERR" >&2 || true
    exit 1
fi

host_rc=0
victory_hygiene_wait_pid "$HOST_PID" 180 || host_rc=$?
victory_hygiene_untrack "$HOST_PID"
HOST_PID=""
if [ "$host_rc" -ne 0 ]; then
    echo "FAIL: the native host exited with $host_rc" >&2
    cat "$HOST_ERR" >&2 || true
    exit 1
fi

fail() {
    echo "FAIL: $1" >&2
    echo "--- host stdout" >&2; cat "$HOST_LOG" >&2 || true
    echo "--- host stderr" >&2; cat "$HOST_ERR" >&2 || true
    echo "--- guest" >&2; cat "$GUEST_LOG" >&2 || true
    exit 1
}

grep -q 'GUEST_V3_PASS' "$GUEST_LOG" || fail "the guest did not report a v3 pass"
grep -q 'replay=AlreadyExecuted alias_unchanged=1' "$GUEST_LOG" \
    || fail "the guest did not observe the refused replay"
grep -q 'unknown_allocation=UnknownHandle' "$GUEST_LOG" \
    || fail "the guest did not observe the unknown-allocation refusal"
grep -q 'unsupported_opcode=ExecutionFailed' "$GUEST_LOG" \
    || fail "the guest did not observe the unsupported-opcode refusal"
grep -q 'd3dmetal_modules=0' "$GUEST_LOG" \
    || fail "the guest process loaded D3DMetal"
grep -q 'kind=compute value=10' "$HOST_LOG" \
    || fail "the host did not report the v3 compute result"
grep -q 'session complete submissions=4 allocations=1' "$HOST_ERR" \
    || fail "unexpected host session summary"

guest_line="$(grep 'GUEST_V3_PASS' "$GUEST_LOG" | head -1)"
compute_line="$(grep 'kind=compute' "$HOST_LOG" | head -1)"
summary_line="$(grep 'session complete' "$HOST_ERR" | head -1)"
echo "PASS synthetic-frame mode=wire-v3 guest=\"$guest_line\""
echo "PASS synthetic-frame host=\"$compute_line\" host=\"$summary_line\""

# --- Part 2: host mixed encoder, MRT/depth graphics and converted DXIL ------
echo "Running the host producer gate (MRT/depth graphics, converted DXIL compute)..."
gate_rc=0
victory_hygiene_run cargo test --locked -p victory-metal --target aarch64-apple-darwin --test producer_e2e \
    -- --ignored --test-threads=1 --nocapture >"$WORK/producer_e2e.log" 2>&1 || gate_rc=$?
cat "$WORK/producer_e2e.log" 2>/dev/null || true
if [ "$gate_rc" -ne 0 ]; then
    echo "FAIL: the host producer gate exited with $gate_rc" >&2
    exit 1
fi
grep -q 'producer_e2e PASS' "$WORK/producer_e2e.log" \
    || fail "the host producer gate did not print its PASS line"

echo "PASS synthetic-frame mode=host-v3 graphics=MRT+depth dxil=rosetta pixels=1024"
echo "PARTIAL synthetic-frame wine-guest-consumes-victory-gpu=0 reason=\"victory-gpu installs DisconnectedSink and GuestIpcClient has no v3 encoder\""
echo "NOTE one frame; this is not an FPS measurement."
