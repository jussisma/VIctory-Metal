#!/bin/sh
# End-to-end frame gate: a real Win64 Wine guest records one graphics list
# (indexed triangle) and one compute dispatch, the native ARM64 Metal host
# executes both through the same IPC session, and the run verifies:
#
#   * the host drew the triangle into its render target (centre pixel and the
#     number of pixels matching it),
#   * the guest's compute constant is visible through its own alias,
#   * a replay of the same list identity is refused as AlreadyExecuted,
#   * the guest alias is byte-identical after the refused replay.
#
# This is one synthetic frame, not a benchmark: no FPS claim is made.
#
# Usage: tools/macos/metal-poc/synthetic-frame.sh
#
# The launcher creates a private temporary share directory, 32-byte secret and
# Wine prefix, and removes them on exit (KEEP=1 keeps them). Nothing here
# launches the title.
#
# Hardware note: this is the only GPU consumer of the run. Do not run it
# concurrently with another Metal benchmark or test.
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Homebrew's rustup is keg-only; test-wine.sh expects it first on PATH too.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" \
        "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
export WINE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/victory-frame-gate.XXXXXXXX")"
SHARE="$WORK/share"
SECRET="$WORK/secret.hex"
HOST_LOG="$WORK/host.stdout"
HOST_ERR="$WORK/host.stderr"
GUEST_LOG="$WORK/guest.log"
HOST_PID=""
PORT=""
COMPUTE_PIPELINE=""
GRAPHICS_PIPELINE=""

cleanup() {
    if [ -n "$HOST_PID" ] && kill -0 "$HOST_PID" 2>/dev/null; then
        kill "$HOST_PID" 2>/dev/null || true
        wait "$HOST_PID" 2>/dev/null || true
    fi
    if [ -d "$WORK/wineprefix" ] && [ -n "${WINE:-}" ] && [ -x "${WINE%/*}/wineserver" ]; then
        WINEPREFIX="$WORK/wineprefix" "${WINE%/*}/wineserver" -k -w >/dev/null 2>&1 || true
    fi
    if [ "${KEEP:-0}" = "1" ]; then
        echo "kept harness files in $WORK"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT INT TERM

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

echo "Building the native ARM64 host worker..."
cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin metal-worker
WORKER="$ROOT/target/aarch64-apple-darwin/debug/metal-worker"
[ -x "$WORKER" ] || { echo "FAIL: $WORKER is missing" >&2; exit 1; }

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
GRAPHICS_PIPELINE="$(sed -n 's/^VICTORY_GRAPHICS_PIPELINE_ID=\([0-9][0-9]*\)$/\1/p' "$HOST_LOG" | tail -1)"
[ -n "$PORT" ] || { echo "FAIL: no host port was published" >&2; exit 1; }
[ -n "$COMPUTE_PIPELINE" ] || { echo "FAIL: no compute pipeline was published" >&2; exit 1; }
[ -n "$GRAPHICS_PIPELINE" ] || { echo "FAIL: no graphics pipeline was published" >&2; exit 1; }

export WINEPREFIX="$WORK/wineprefix"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_IPC_PORT="$PORT"
export VICTORY_IPC_PIPELINE_ID="$COMPUTE_PIPELINE"
export VICTORY_IPC_GRAPHICS_PIPELINE_ID="$GRAPHICS_PIPELINE"
export VICTORY_IPC_SECRET_FILE="$(winpath "$SECRET")"

echo "Host ready: port=$PORT compute=$COMPUTE_PIPELINE graphics=$GRAPHICS_PIPELINE"
echo "Running the Wine guest frame-gate test..."
guest_rc=0
"$ROOT/tools/macos/test-wine.sh" --locked -p victory-hle --lib \
    xmem::e2e::wine_guest_draws_and_computes_through_native_frame_gate \
    -- --test-threads=1 --nocapture 2>&1 | tee "$GUEST_LOG" || guest_rc=$?
if [ "$guest_rc" -ne 0 ]; then
    echo "FAIL: the Wine guest test exited with $guest_rc" >&2
    cat "$HOST_LOG" "$HOST_ERR" >&2 || true
    exit 1
fi

host_rc=0
wait "$HOST_PID" || host_rc=$?
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

grep -q 'kind=draw centre=0,255,0,255 pixels=4096' "$HOST_LOG" \
    || fail "the host did not report a full-target green draw"
grep -q 'kind=compute value=25' "$HOST_LOG" \
    || fail "the host did not report the computed constant 25"
grep -q 'session complete submissions=3 allocations=1' "$HOST_ERR" \
    || fail "unexpected host session summary"
grep -q 'replay=AlreadyExecuted' "$GUEST_LOG" \
    || fail "the guest did not observe the refused replay"
grep -q 'alias_unchanged=1' "$GUEST_LOG" \
    || fail "the guest did not confirm the alias was unchanged"

draw_line="$(grep 'kind=draw' "$HOST_LOG" | head -1)"
compute_line="$(grep 'kind=compute' "$HOST_LOG" | head -1)"
summary_line="$(grep 'session complete' "$HOST_ERR" | head -1)"
echo "PASS synthetic-frame wine-guest=1 draw=\"$draw_line\" compute=\"$compute_line\""
echo "PASS synthetic-frame host=\"$summary_line\" replay=AlreadyExecuted alias_unchanged=1 frame=1"
echo "NOTE one synthetic frame; this is not an FPS measurement."
