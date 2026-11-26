#!/bin/sh
# End-to-end synthetic launcher: real Win64 Wine XMem file-backed physical
# pages + the real guest IPC client -> the native ARM64 Metal host, whose two
# ordered dispatches write straight into the guest alias.
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

cleanup() {
    if [ -n "$HOST_PID" ] && kill -0 "$HOST_PID" 2>/dev/null; then
        kill "$HOST_PID" 2>/dev/null || true
        wait "$HOST_PID" 2>/dev/null || true
    fi
    # Stop the private prefix's wineserver before removing its directory, so
    # it cannot fail to save its registry into a deleted tree.
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
# 32 secret bytes as hex, readable only by this user. Never echoed.
python3 -c 'import secrets; print(secrets.token_hex(32))' >"$SECRET"
chmod 600 "$SECRET"

winpath() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

echo "Building the native ARM64 host worker..."
cargo build --locked --quiet -p victory-metal --target aarch64-apple-darwin --bin metal-worker
WORKER="$ROOT/target/aarch64-apple-darwin/debug/metal-worker"
[ -x "$WORKER" ] || { echo "FAIL: $WORKER is missing" >&2; exit 1; }

echo "Starting the native host (private share $SHARE)..."
"$WORKER" \
    --secret-file "$SECRET" \
    --share-dir "$SHARE" \
    --port 0 \
    --bootstrap-compute \
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
PIPELINE="$(sed -n 's/^VICTORY_PIPELINE_ID=\([0-9][0-9]*\)$/\1/p' "$HOST_LOG" | tail -1)"
[ -n "$PORT" ] || { echo "FAIL: no host port was published" >&2; exit 1; }
[ -n "$PIPELINE" ] || { echo "FAIL: no bootstrap pipeline was published" >&2; exit 1; }

echo "Host ready: port=$PORT pipeline=$PIPELINE share=$(winpath "$SHARE")"
echo "Running the Wine guest test..."

export WINEPREFIX="$WORK/wineprefix"
export VICTORY_METAL_SHARE_DIR="$(winpath "$SHARE")"
export VICTORY_IPC_PORT="$PORT"
export VICTORY_IPC_PIPELINE_ID="$PIPELINE"
export VICTORY_IPC_SECRET_FILE="$(winpath "$SECRET")"

test_rc=0
"$ROOT/tools/macos/test-wine.sh" --locked -p victory-hle --lib xmem::e2e \
    -- --test-threads=1 --nocapture || test_rc=$?

host_rc=0
wait "$HOST_PID" || host_rc=$?
HOST_PID=""

if [ "$test_rc" -ne 0 ]; then
    echo "FAIL: the Wine guest test exited with $test_rc" >&2
    cat "$HOST_ERR" >&2 || true
    exit 1
fi
if [ "$host_rc" -ne 0 ]; then
    echo "FAIL: the native host exited with $host_rc" >&2
    cat "$HOST_ERR" >&2 || true
    exit 1
fi
grep -q 'session complete submissions=2 allocations=1' "$HOST_ERR" || {
    echo "FAIL: unexpected native host summary" >&2
    cat "$HOST_ERR" >&2 || true
    exit 1
}
cat "$HOST_ERR"

echo "PASS xmem-ipc wine-guest=1 guest-alias=1 ordered-dispatches=2 pipeline=$PIPELINE"
