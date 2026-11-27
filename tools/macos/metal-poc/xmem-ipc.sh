#!/bin/sh
# End-to-end synthetic launcher: real Win64 Wine XMem file-backed physical
# pages + the real guest IPC client -> the native ARM64 Metal host.
#
# Two independent host sessions run in sequence, each serving exactly one
# Wine guest test (the worker serves one peer and exits):
#
# 1. bootstrap: the host compiles one MSL kernel; the guest submits two
#    ordered dispatches of it and reads the result through the real alias.
# 2. pipeline: Apple's offline Metal toolchain compiles a synthetic kernel to
#    `{ticket}.metallib` in the share directory; the guest registers it over
#    IPC by ticket and entry name only, dispatches the returned native id
#    after the bootstrap kernel, and verifies that a ticket without a backing
#    file is rejected without dispatching.
#
# Usage: tools/macos/metal-poc/xmem-ipc.sh
#
# The launcher creates everything the session needs in a private temporary
# directory (share directory, 32-byte secret, Wine prefix) and cleans it up
# again. The secret is never printed. Nothing here launches the title.
#
# Hardware note: this is the only GPU consumer of the run. Do not run it
# concurrently with another Metal benchmark or test; native GPU work is
# serialized by the caller.
#
# WINE is optional; tools/macos/test-wine.sh picks the checkout's Wine.
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SELF_DIR/process-hygiene.sh"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; test-wine.sh expects it first on PATH too.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
# Resolve Wine the same way test-wine.sh does, so cleanup can stop the private
# prefix's wineserver next to the Wine that ran the guest.
if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
export WINE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-xmem-ipc.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
HOST_LOG="$WORK/host.stdout"
HOST_ERR="$WORK/host.stderr"
HOST_PID=""
PORT=""
PIPELINE=""
# The ticket is the only name of the offline library that crosses the wire.
METALLIB_TICKET=1
METALLIB_ENTRY=vm_ipc_metallib_scale
SHARE_METALLIB="$(printf '%s/%016x.metallib' "$SHARE" "$METALLIB_TICKET")"
victory_hygiene_work "$WORK"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    # Stop the private prefix's wineserver before removing its directory, so
    # it cannot fail to save its registry into a deleted tree.
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
# 32 secret bytes as hex, readable only by this user. Never echoed.
python3 -c 'import secrets; print(secrets.token_hex(32))' >"$SECRET"
chmod 600 "$SECRET"

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

# The offline compiler artifact is a real launcher input: the guest never
# sends a path, so the ticket-named file must exist before registration.
echo "Compiling the synthetic precompiled kernel..."
METAL="$(xcrun -f metal 2>/dev/null || true)"
METALLIB_TOOL="$(xcrun -f metallib 2>/dev/null || true)"
if [ -z "$METAL" ] || [ ! -x "$METAL" ] || [ -z "$METALLIB_TOOL" ] || [ ! -x "$METALLIB_TOOL" ]; then
    echo "FAIL: the offline Apple Metal toolchain (metal/metallib) is unavailable" >&2
    exit 1
fi
cat >"$WORK/pipeline.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

// Selected by name over IPC: x -> 5x + 3.
kernel void vm_ipc_metallib_scale(device uint *data [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 5u + 3u;
}

// Never selected: a wrong-entry lookup would produce x + 1 instead.
kernel void vm_ipc_metallib_unused(device uint *data [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] + 1u;
}
EOF
"$METAL" -c "$WORK/pipeline.metal" -o "$WORK/pipeline.air"
"$METALLIB_TOOL" "$WORK/pipeline.air" -o "$SHARE_METALLIB"
[ -s "$SHARE_METALLIB" ] || { echo "FAIL: $SHARE_METALLIB is empty" >&2; exit 1; }
echo "Offline metallib: ticket=$METALLIB_TICKET entry=$METALLIB_ENTRY bytes=$(wc -c <"$SHARE_METALLIB" | tr -d ' ')"

echo "Building the native ARM64 host worker..."
victory_hygiene_run cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin metal-worker
WORKER="$ROOT/target/aarch64-apple-darwin/debug/metal-worker"
[ -x "$WORKER" ] || { echo "FAIL: $WORKER is missing" >&2; exit 1; }

# Starts the single-peer host and waits for its published port and pipeline.
start_host() {
    : >"$HOST_LOG"
    : >"$HOST_ERR"
    "$WORKER" \
        --secret-file "$SECRET" \
        --share-dir "$SHARE" \
        --port 0 \
        --bootstrap-compute \
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
    PIPELINE="$(sed -n 's/^VICTORY_PIPELINE_ID=\([0-9][0-9]*\)$/\1/p' "$HOST_LOG" | tail -1)"
    [ -n "$PORT" ] || { echo "FAIL: no host port was published" >&2; exit 1; }
    [ -n "$PIPELINE" ] || { echo "FAIL: no bootstrap pipeline was published" >&2; exit 1; }

    export WINEPREFIX="$WORK/wineprefix"
    export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
    export VICTORY_IPC_PORT="$PORT"
    export VICTORY_IPC_PIPELINE_ID="$PIPELINE"
    export VICTORY_IPC_SECRET_FILE="$(winpath "$SECRET")"
}

# Stops the host that served the case and checks its summary.
finish_host() {
    expected_submissions=$1
    host_rc=0
    victory_hygiene_wait_pid "$HOST_PID" 180 || host_rc=$?
    victory_hygiene_untrack "$HOST_PID"
    HOST_PID=""
    if [ "$host_rc" -ne 0 ]; then
        echo "FAIL: the native host exited with $host_rc" >&2
        cat "$HOST_ERR" >&2 || true
        exit 1
    fi
    grep -q "session complete submissions=$expected_submissions allocations=1" "$HOST_ERR" || {
        echo "FAIL: unexpected native host summary" >&2
        cat "$HOST_ERR" >&2 || true
        exit 1
    }
    cat "$HOST_ERR"
}

# Runs one guest test against the currently started host.
run_guest() {
    filter=$1
    echo "Running the Wine guest test $filter..."
    test_rc=0
    "$ROOT/tools/macos/test-wine.sh" --locked -p victory-hle --lib "$filter" \
        -- --test-threads=1 --nocapture || test_rc=$?
    if [ "$test_rc" -ne 0 ]; then
        echo "FAIL: the Wine guest test exited with $test_rc" >&2
        cat "$HOST_ERR" >&2 || true
        exit 1
    fi
}

# Case 1: the existing bootstrap pipeline path, unchanged.
start_host
echo "Host ready (bootstrap): port=$PORT pipeline=$PIPELINE share=$(winpath "$SHARE")"
unset VICTORY_IPC_PIPELINE_TICKET VICTORY_IPC_PIPELINE_ENTRY 2>/dev/null || true
run_guest xmem::e2e::two_ordered_host_dispatches_write_the_real_guest_alias
finish_host 2
BOOTSTRAP_PIPELINE="$PIPELINE"

# Case 2: offline-compiled pipeline registration over the same transport.
start_host
echo "Host ready (pipeline): port=$PORT pipeline=$PIPELINE share=$(winpath "$SHARE")"
export VICTORY_IPC_PIPELINE_TICKET="$METALLIB_TICKET"
export VICTORY_IPC_PIPELINE_ENTRY="$METALLIB_ENTRY"
run_guest xmem::e2e::registered_metallib_pipeline_dispatches_and_rejection_stays_inert
finish_host 2

echo "PASS xmem-ipc wine-guest=1 guest-alias=1 ordered-dispatches=2 pipeline=$BOOTSTRAP_PIPELINE"
echo "PASS xmem-ipc-pipeline wine-guest=1 offline-metallib=1 ticket=$METALLIB_TICKET entry=$METALLIB_ENTRY rejected-ticket=1"
