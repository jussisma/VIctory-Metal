#!/bin/sh
# ADR-031: turnkey, bounded live-title launcher for the native ARM64 Metal path.
#
# It wraps tools/macos/metal-poc/run-title-metal.sh, which owns the daemon, the
# machine-readable ready line, telemetry and process hygiene. This script only:
#   * builds the Win64 diag victory.exe, the ARM64 victory_metal_host and the
#     x86_64 victory-irconverter when they are missing or older than the tree,
#   * resolves VICTORY_BACKEND (metal by default, d3d12 for the D3D12 host
#     reference) and never forces a D3DMetal DLL override of its own,
#   * bounds the delegated run with GUEST_TIMEOUT and a hard outer guard,
#   * prints its own summary with backend, endpoint, share dir and fence page,
#   * removes the diagnostics directory it created itself unless the caller
#     passes --keep-session (a caller-supplied --out is never removed).
#
# Usage:
#   victory-metal.sh [--window|--headless] [--timeout SEC] [--frames N]
#                    [--out DIR] [--keep-session] [--dry-run] [--exe PATH]
#                    [-- guest args...]
#
#   --exe PATH    guest executable passed to `victory.exe run` (default: the
#                 built diag victory.exe, a start-up self-check). Relative
#                 paths are resolved against the caller's directory; a Windows
#                 path (`Z:\...`) is passed to Wine unchanged.
#   --timeout SEC bound the whole delegated run: GUEST_TIMEOUT=SEC and, when a
#                 timeout(1) is installed, a hard `timeout` of SEC+60 seconds
#                 (an internal watchdog otherwise). The 60 s margin lets the
#                 delegate reap the daemon and Wine after the guest stops.
#   --frames N    diagnostic frame count (VICTORY_FRAME_MEASURED, default 120)
#   --out DIR     build.log, launcher.log, host.log, guest.log, telemetry.json
#                 and summary.txt (default: this run's own mktemp directory,
#                 removed on exit unless --keep-session). A caller-supplied
#                 directory is kept and never removed.
#   --keep-session keep the session and the default --out directory. `--keep`,
#                 VICTORY_KEEP_LOGS=1 and VICTORY_KEEP_SESSION=1 are the same
#                 opt-in (historical aliases); removal is the default on every
#                 exit path.
#   --dry-run     print the resolved command and exported environment; build
#                 nothing, start no daemon and touch no GPU
#   --window      native window presentation
#   --headless    headless presentation (default); conflicts with --window
#   -h, --help
#
# Everything after `--` is forwarded to the guest after `--execute`.
#
# Backend (VICTORY_BACKEND):
#   metal (default)  the native ARM64 path: the delegate starts the daemon.
#   d3d12            the D3D12 host reference path, which is what
#                    metal-poc/abi-diff.py diffs the native guest ABI against.
#   Any other value is a usage error before a binary is built or a temp dir is
#   created. The resolved value is printed by --dry-run, in the summary and in
#   the delegate's `LAUNCH_ENV backend=` line.
#
# Delegation:
#   run-title-metal.sh <mode> --frames N --out DIR [--keep-session] --print-env \
#       -- <victory.exe> run <guest> --execute [guest args...]
#
# VICTORY_LAUNCHER_RUNNER overrides only the delegate executable (default:
# tools/macos/metal-poc/run-title-metal.sh) so the launcher contract can be
# tested hermetically. The wrapper still owns its temp directory, its signals,
# its outer guard and the exit code; the override cannot make it skip those.
#
# Exit codes:
#   0    the delegated run PASSed
#   2    usage error
#   3    a required binary is missing and did not build
#   124  the outer guard stopped the delegated run (timeout(1) or watchdog)
#   *    otherwise the delegated exit code, unchanged: 10 guest exited non-zero,
#        11 D3DMetal fallback observed, 12 gate report partial, 13 orphan
#        survived, 14 host exited, 15 summary lost, 130/143/129 signal.
#
# The session secret is never printed or logged. Teardown of the guest session
# belongs entirely to run-title-metal.sh; this wrapper waits for it, propagates
# its status and removes only the --out directory it created itself.
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SELF_DIR/../.." && pwd)"
# VICTORY_LAUNCHER_RUNNER replaces only the delegate executable (tests use a
# stub); the wrapper's own hygiene, guard and exit propagation stay in place.
RUNNER="${VICTORY_LAUNCHER_RUNNER:-$ROOT/tools/macos/metal-poc/run-title-metal.sh}"
START_DIR="$(pwd)"
# Homebrew's rustup is keg-only; put it first like the other macOS harnesses.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
case "$TARGET_DIR" in
    /*) ;;
    *) TARGET_DIR="$ROOT/$TARGET_DIR" ;;
esac
APP="$TARGET_DIR/x86_64-pc-windows-gnullvm/diag/victory.exe"
HOST_BIN="$TARGET_DIR/aarch64-apple-darwin/debug/victory_metal_host"
IRCONV_BIN="$TARGET_DIR/x86_64-apple-darwin/debug/victory-irconverter"

WINDOW=0
WINDOW_SET=0
HEADLESS_SET=0
TIMEOUT=""
FRAMES=120
OUT_ARG=""
OUT=""
OUT_OWNED=0
KEEP=0
DRY_RUN=0
EXE=""
GUEST=""
BACKEND="metal"
MODE="headless"
MODE_FLAG="--headless"
GUARD_SECONDS=""
TIMEOUT_BIN=""
TIMED_OUT=0
CHILD=""
WATCHDOG=""
ENDPOINT=""
SHARE_DIR_RESOLVED=""
FENCE_PAGE_RESOLVED=""

usage() {
    cat <<'USAGE'
usage: victory-metal.sh [--window|--headless] [--timeout SEC] [--frames N]
                        [--out DIR] [--keep-session] [--dry-run] [--exe PATH]
                        [-- guest args...]

  --exe PATH    guest executable passed to `victory.exe run` (default: the
                built diag victory.exe under the checkout)
  --timeout SEC bound the run with GUEST_TIMEOUT=SEC and a hard outer guard
  --frames N    diagnostic frame count (default 120)
  --out DIR     build/launcher/host/guest logs, telemetry and summary.txt here
  --keep-session keep the session and the default --out directory (--keep,
                VICTORY_KEEP_LOGS=1 and VICTORY_KEEP_SESSION=1 alias it)
  --dry-run     print the resolved command and environment; do nothing else
  --window      open the native window
  --headless    force headless presentation (default); conflicts with --window
  -h, --help

VICTORY_BACKEND selects the guest backend: metal (default) or d3d12 (the D3D12
host reference path). Any other value is a usage error.
USAGE
}

fail_usage() {
    echo "victory-metal.sh: $1" >&2
    usage >&2
    exit 2
}

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
        --timeout)
            [ "$#" -ge 2 ] || fail_usage "--timeout requires a value"
            TIMEOUT="$2"
            shift 2
            ;;
        --frames)
            [ "$#" -ge 2 ] || fail_usage "--frames requires a value"
            FRAMES="$2"
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || fail_usage "--out requires a value"
            OUT_ARG="$2"
            shift 2
            ;;
        --keep-session|--keep)
            KEEP=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --exe)
            [ "$#" -ge 2 ] || fail_usage "--exe requires a value"
            EXE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            fail_usage "unknown option $1"
            ;;
        *)
            fail_usage "unexpected argument $1 (guest arguments go after --)"
            ;;
    esac
done

if [ "$WINDOW_SET" = 1 ] && [ "$HEADLESS_SET" = 1 ]; then
    fail_usage "--window conflicts with --headless"
fi
if [ -n "$TIMEOUT" ]; then
    case "$TIMEOUT" in
        ""|*[!0-9]*|0) fail_usage "--timeout must be a positive integer" ;;
    esac
fi
case "$FRAMES" in
    ""|*[!0-9]*|0) fail_usage "--frames must be a positive integer" ;;
esac

# Resolve the delegate backend before a binary is built or a temp dir exists,
# so a typo is a usage error that leaves nothing behind. An inherited value
# selects the D3D12 host reference path; the default stays the native metal
# path exactly as before.
case "${VICTORY_BACKEND:-metal}" in
    metal) BACKEND=metal ;;
    d3d12) BACKEND=d3d12 ;;
    *)
        echo "victory-metal.sh: unsupported VICTORY_BACKEND=${VICTORY_BACKEND:-} (expected metal or d3d12)" >&2
        usage >&2
        exit 2
        ;;
esac
export VICTORY_BACKEND="$BACKEND"

# The delegate is exec'd directly, so it must be a real executable file. This
# only selects the process; signal forwarding, the outer guard and this
# wrapper's own temp-dir hygiene are unaffected.
if [ ! -f "$RUNNER" ] || [ ! -x "$RUNNER" ]; then
    echo "victory-metal.sh: VICTORY_LAUNCHER_RUNNER is not an executable file: $RUNNER" >&2
    exit 3
fi

if [ "$WINDOW" = 1 ]; then
    MODE=window
    MODE_FLAG=--window
else
    MODE=headless
    MODE_FLAG=--headless
fi

GUEST="${EXE:-$APP}"
case "$GUEST" in
    /*) ;;
    [A-Za-z]:*) ;;
    *\\*) ;;
    *) GUEST="$START_DIR/$GUEST" ;;
esac

# --- Teardown --------------------------------------------------------------
# Installed before this run creates any path, so every exit status (including a
# usage check that fails after the temp dir exists) removes what it owns. The
# delegate removes its own session; this wrapper removes only the --out
# directory it created itself.
stop_watchdog() {
    if [ -n "${WATCHDOG:-}" ]; then
        kill "$WATCHDOG" 2>/dev/null || true
        wait "$WATCHDOG" 2>/dev/null || true
        WATCHDOG=""
    fi
}

forward_and_exit() {
    status=$1
    trap '' INT TERM HUP
    if [ -n "${CHILD:-}" ]; then
        kill -TERM "$CHILD" 2>/dev/null || true
        waited=0
        while kill -0 "$CHILD" 2>/dev/null && [ "$waited" -lt 100 ]; do
            sleep 0.1
            waited=$((waited + 1))
        done
        kill -KILL "$CHILD" 2>/dev/null || true
        wait "$CHILD" 2>/dev/null || true
        CHILD=""
    fi
    stop_watchdog
    exit "$status"
}

# --keep-session and its historical aliases.
keep_session() {
    if [ "${VICTORY_KEEP_LOGS:-0}" = 1 ] || [ "${VICTORY_KEEP_SESSION:-0}" = 1 ] || [ "${KEEP:-0}" = 1 ]; then
        return 0
    fi
    return 1
}

wrapper_cleanup() {
    status=$?
    trap '' INT TERM HUP
    set +e
    stop_watchdog
    if [ "${OUT_OWNED:-0}" = 1 ] && [ -n "${OUT:-}" ] && [ -e "$OUT" ]; then
        case "$OUT" in
            */victory-metal-out.????????) ;;
            *)
                echo "victory-metal.sh: WARN: refusing to remove unexpected diagnostics path $OUT" >&2
                OUT=""
                ;;
        esac
        if [ -n "$OUT" ]; then
            if keep_session; then
                echo "victory-metal.sh: kept session files in $OUT"
            else
                rm -rf -- "$OUT" 2>/dev/null || true
                if [ -e "$OUT" ]; then
                    echo "victory-metal.sh: WARN: diagnostics directory survived removal: $OUT" >&2
                else
                    echo "victory-metal.sh: removed the session directory (use --keep-session to keep it)"
                fi
            fi
        fi
    fi
    trap - EXIT INT TERM HUP
    exit "$status"
}
trap wrapper_cleanup EXIT
trap 'forward_and_exit 130' INT
trap 'forward_and_exit 143' TERM
trap 'forward_and_exit 129' HUP

if [ -n "$OUT_ARG" ]; then
    case "$OUT_ARG" in
        /*) OUT="$OUT_ARG" ;;
        *) OUT="$START_DIR/$OUT_ARG" ;;
    esac
    OUT_OWNED=0
elif [ "$DRY_RUN" = 1 ]; then
    OUT="${TMPDIR:-/tmp}/victory-metal-out.DRYRUN"
    OUT_OWNED=0
else
    OUT="$(mktemp -d "${TMPDIR:-/tmp}/victory-metal-out.XXXXXXXX")" \
        || { echo "victory-metal.sh: cannot create the --out directory" >&2; exit 3; }
    OUT_OWNED=1
fi

if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$OUT" || { echo "victory-metal.sh: cannot create --out $OUT" >&2; exit 3; }
    OUT="$(cd "$OUT" && pwd)"
fi

if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="$(command -v gtimeout)"
fi
if [ -n "$TIMEOUT" ]; then
    GUARD_SECONDS=$((TIMEOUT + 60))
fi

# Cycle 08 Boot #1: without this, Rosetta does not report AVX/AVX2 in CPUID and
# victory.exe's own Xbox-Series preflight refuses to execute the title. The
# reference launcher (tools/macos/victory.sh) sets the same variable. An
# explicit value from the caller wins.
export ROSETTA_ADVERTISE_AVX="${ROSETTA_ADVERTISE_AVX:-1}"
if [ -n "$TIMEOUT" ]; then
    export GUEST_TIMEOUT="$TIMEOUT"
fi

# An explicit guest is checked before the (possibly slow) build; the default
# one is the app this launcher is about to build, so it is checked afterwards.
if [ "$DRY_RUN" != 1 ] && [ -n "$EXE" ]; then
    case "$GUEST" in
        [A-Za-z]:*) ;;
        *) [ -f "$GUEST" ] || { echo "victory-metal.sh: guest executable not found: $GUEST" >&2; exit 2; } ;;
    esac
fi

# True when any source of the workspace is newer than `bin`.
sources_are_newer() {
    bin=$1
    find "$ROOT/apps" "$ROOT/crates" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" \
        "$ROOT/rust-toolchain.toml" "$ROOT/.cargo" \
        -type f -newer "$bin" -print -quit 2>/dev/null | grep -q .
}

# Build `name` into `bin` when it is missing or stale. An existing binary is
# never a failure: a failed stale rebuild warns and keeps it, so a launch stays
# fast on a machine whose toolchain cannot rebuild right now.
build_one() {
    name=$1
    bin=$2
    shift 2
    if [ -x "$bin" ] && ! sources_are_newer "$bin"; then
        return 0
    fi
    echo "victory-metal.sh: building $name..." >&2
    {
        echo
        echo "== $(date -u '+%Y-%m-%dT%H:%M:%SZ') $name"
        printf '+'
        printf ' %s' "$@"
        echo
    } >>"$OUT/build.log"
    if ( cd "$ROOT" && "$@" ) >>"$OUT/build.log" 2>&1; then
        if [ ! -x "$bin" ]; then
            echo "victory-metal.sh: FAIL: $name built but $bin is missing" >&2
            return 1
        fi
        return 0
    fi
    if [ -x "$bin" ]; then
        echo "victory-metal.sh: WARN: $name did not rebuild; using the existing $bin (see $OUT/build.log)" >&2
        return 0
    fi
    echo "victory-metal.sh: FAIL: $name did not build and $bin is missing (see $OUT/build.log)" >&2
    tail -20 "$OUT/build.log" >&2 || true
    return 1
}

prepare_builds() {
    command -v cargo >/dev/null 2>&1 \
        || { echo "victory-metal.sh: cargo is required to build the launcher binaries" >&2; exit 3; }
    : >"$OUT/build.log"
    build_one "Win64 diag victory.exe" "$APP" \
        cargo build --locked --profile diag --target x86_64-pc-windows-gnullvm \
        -p victory --features diagnostics || exit 3
    build_one "ARM64 victory_metal_host" "$HOST_BIN" \
        cargo build --locked -p victory-metal --target aarch64-apple-darwin \
        --bin victory_metal_host || exit 3
    build_one "x86_64 victory-irconverter" "$IRCONV_BIN" \
        cargo build --locked -p victory-metal --target x86_64-apple-darwin \
        --bin victory-irconverter || exit 3
}

# The fallback timer is one Python process (run-title-metal.sh already needs
# python3): no long-lived `sleep` child can survive a killed watchdog.
start_watchdog() {
    seconds=$1
    pid=$2
    rm -f "$OUT/.timeout-fired"
    python3 - "$seconds" "$pid" "$OUT/.timeout-fired" <<'PY' &
import os
import signal
import sys
import time

seconds, pid, marker = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
time.sleep(seconds)
try:
    with open(marker, "w") as handle:
        handle.write("1\n")
except OSError:
    pass
try:
    os.kill(pid, signal.SIGTERM)
except OSError:
    sys.exit(0)
time.sleep(10)
try:
    os.kill(pid, signal.SIGKILL)
except OSError:
    pass
PY
    WATCHDOG=$!
}

# Run the delegated launcher, capturing its output to `$OUT/launcher.log` and
# returning its exit status. `seconds` empty means no outer guard.
run_guarded() {
    seconds=$1
    shift
    if [ -z "$seconds" ]; then
        "$@" >"$OUT/launcher.log" 2>&1 &
        CHILD=$!
        rc=0
        wait "$CHILD" || rc=$?
        CHILD=""
        return "$rc"
    fi
    if [ -n "$TIMEOUT_BIN" ]; then
        "$TIMEOUT_BIN" "$seconds" "$@" >"$OUT/launcher.log" 2>&1 &
    else
        "$@" >"$OUT/launcher.log" 2>&1 &
    fi
    CHILD=$!
    if [ -z "$TIMEOUT_BIN" ]; then
        if command -v python3 >/dev/null 2>&1; then
            start_watchdog "$seconds" "$CHILD"
        else
            echo "victory-metal.sh: WARN: no timeout(1) or python3; only GUEST_TIMEOUT bounds the run" >&2
        fi
    fi
    rc=0
    wait "$CHILD" || rc=$?
    CHILD=""
    stop_watchdog
    if [ -f "$OUT/.timeout-fired" ]; then
        TIMED_OUT=1
        if [ "$rc" -eq 0 ]; then
            rc=124
        fi
    elif [ "$rc" -eq 124 ]; then
        TIMED_OUT=1
    fi
    return "$rc"
}

print_args() {
    for arg in "$@"; do
        case "$arg" in
            '') printf "'' " ;;
            *[!A-Za-z0-9_./:=+@%-]*) printf "'%s' " "$arg" ;;
            *) printf '%s ' "$arg" ;;
        esac
    done
    printf '\n'
}

dry_run() {
    echo "victory-metal.sh: dry-run: nothing built, no daemon started, no GPU work"
    echo "COMMAND"
    if [ -n "$GUARD_SECONDS" ]; then
        if [ -n "$TIMEOUT_BIN" ]; then
            printf '  %s %s ' "$TIMEOUT_BIN" "$GUARD_SECONDS"
        else
            printf '  [internal watchdog %ss] ' "$GUARD_SECONDS"
        fi
    else
        printf '  '
    fi
    print_args "$@"
    echo "ENVIRONMENT"
    echo "  VICTORY_BACKEND=$BACKEND"
    echo "  ROSETTA_ADVERTISE_AVX=${ROSETTA_ADVERTISE_AVX:-1}"
    if [ -n "$TIMEOUT" ]; then
        echo "  GUEST_TIMEOUT=$TIMEOUT"
    else
        echo "  GUEST_TIMEOUT=<delegate default 900>"
    fi
    echo "  VICTORY_FRAME_MEASURED=$FRAMES"
    echo "  VICTORY_FRAME_WARMUP=<delegate default 20>"
    echo "  VICTORY_METAL_SECRET=REDACTED"
    if [ "$BACKEND" = metal ]; then
        echo "  VICTORY_METAL_ENDPOINT=<resolved by run-title-metal.sh>"
        echo "  VICTORY_METAL_SHARE_DIR=<resolved by run-title-metal.sh>"
        echo "  VICTORY_METAL_FENCE_PAGE=<share dir>/gpu-fence.v1.page"
        echo "  WINEDLLOVERRIDES=<delegate default: mscoree,mshtml=;winedbg.exe=d>"
        echo "  # no D3DMetal d3d12/d3d12core/dxgi override is set or required"
    else
        echo "  VICTORY_METAL_ENDPOINT=<unset: the d3d12 reference has no native daemon>"
        echo "  VICTORY_METAL_SHARE_DIR=<resolved by run-title-metal.sh>"
        echo "  VICTORY_METAL_FENCE_PAGE=<share dir>/gpu-fence.v1.page"
        echo "  WINEDLLOVERRIDES=<delegate default: mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b>"
    fi
    echo "  CARGO_TARGET_DIR=$TARGET_DIR"
    if [ -n "${WINE:-}" ]; then
        echo "  WINE=$WINE"
    else
        echo "  WINE=<auto-detected by run-title-metal.sh>"
    fi
    echo "  # inherited environment is passed through unchanged"
    echo "SUMMARY"
    echo "  launcher=victory-metal"
    echo "  backend=$BACKEND"
    echo "  runner=$RUNNER"
    echo "  mode=$MODE"
    echo "  endpoint=<resolved at launch>"
    echo "  share_dir=<resolved at launch>"
    echo "  fence_page=<resolved at launch>"
    echo "  app=$APP"
    echo "  guest=$GUEST"
    echo "  frames=$FRAMES"
    echo "  timeout=${TIMEOUT:-none}"
    echo "  outer_guard=${GUARD_SECONDS:-none}"
    echo "  out=$OUT"
}

read_resolved() {
    ENDPOINT=""
    SHARE_DIR_RESOLVED=""
    FENCE_PAGE_RESOLVED=""
    log="$OUT/launcher.log"
    if [ -f "$log" ]; then
        ENDPOINT="$(sed -n 's/^LAUNCH_ENV endpoint=//p' "$log" | tail -n 1)"
        SHARE_DIR_RESOLVED="$(sed -n 's/^LAUNCH_ENV share_dir=//p' "$log" | tail -n 1)"
        FENCE_PAGE_RESOLVED="$(sed -n 's/^LAUNCH_ENV fence_page=//p' "$log" | tail -n 1)"
    fi
    if [ -z "$ENDPOINT" ] && [ -f "$OUT/summary.txt" ]; then
        port="$(sed -n 's/.*ready port=\([0-9][0-9]*\).*/\1/p' "$OUT/summary.txt" | tail -n 1)"
        if [ -n "$port" ]; then
            ENDPOINT="127.0.0.1:$port"
        fi
    fi
}

print_summary() {
    rc=$1
    status=FAIL
    if [ "$TIMED_OUT" = 1 ]; then
        status=TIMEOUT
    elif [ "$rc" -eq 0 ]; then
        status=PASS
    fi
    echo "victory-metal.sh: summary"
    echo "  launcher=victory-metal"
    echo "  backend=$BACKEND"
    echo "  runner=$RUNNER"
    echo "  status=$status"
    echo "  exit_code=$rc"
    echo "  endpoint=${ENDPOINT:-none}"
    echo "  share_dir=${SHARE_DIR_RESOLVED:-none}"
    echo "  fence_page=${FENCE_PAGE_RESOLVED:-none}"
    echo "  app=$APP"
    echo "  guest=$GUEST"
    echo "  mode=$MODE"
    echo "  frames=$FRAMES"
    echo "  timeout=${TIMEOUT:-none}"
    echo "  outer_guard=${GUARD_SECONDS:-none}"
    echo "  timed_out=$TIMED_OUT"
    echo "  out=$OUT"
    if [ -f "$OUT/summary.txt" ]; then
        echo "  delegate_summary=$OUT/summary.txt"
    fi
}

delegate() {
    mode=$1
    keep=$2
    shift 2
    if [ "$keep" = 1 ]; then
        set -- "$RUNNER" "$mode" --frames "$FRAMES" --out "$OUT" --keep-session --print-env \
            -- "$APP" run "$GUEST" --execute "$@"
    else
        set -- "$RUNNER" "$mode" --frames "$FRAMES" --out "$OUT" --print-env \
            -- "$APP" run "$GUEST" --execute "$@"
    fi
    if [ "$DRY_RUN" = 1 ]; then
        dry_run "$@"
        return 0
    fi
    rc=0
    run_guarded "$GUARD_SECONDS" "$@" || rc=$?
    if [ -f "$OUT/launcher.log" ]; then
        cat "$OUT/launcher.log" || true
    fi
    read_resolved
    print_summary "$rc"
    return "$rc"
}

# The env aliases (VICTORY_KEEP_LOGS / VICTORY_KEEP_SESSION) must reach the
# delegate too, so the flag is computed once from every spelling.
KEEP_FLAG=0
if keep_session; then
    KEEP_FLAG=1
fi

if [ "$DRY_RUN" = 1 ]; then
    delegate "$MODE_FLAG" "$KEEP_FLAG" "$@"
    exit 0
fi

prepare_builds
case "$GUEST" in
    [A-Za-z]:*) ;;
    *) [ -f "$GUEST" ] || { echo "victory-metal.sh: guest executable not found: $GUEST" >&2; exit 2; } ;;
esac
rc=0
delegate "$MODE_FLAG" "$KEEP_FLAG" "$@" || rc=$?
exit "$rc"
