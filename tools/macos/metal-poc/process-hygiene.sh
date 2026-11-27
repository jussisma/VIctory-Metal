#!/bin/sh
# C7 / Wave 7A: shared process-lifecycle helper for the metal-poc harnesses.
#
# Source this file from a POSIX `sh` script *before* it starts any child:
#
#     SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
#     . "$SELF_DIR/process-hygiene.sh"
#     trap cleanup EXIT
#     trap 'exit 130' INT
#     trap 'exit 143' TERM
#     trap 'exit 129' HUP
#
# Ownership rules (non-negotiable):
#   * Only PIDs registered with `victory_hygiene_track` (direct children started
#     with `&` and `$!`, or `victory_hygiene_run`) are reaped. This helper never
#     runs `pkill victory-irconverter`, `pkill wine` or any global name match.
#   * `victory_hygiene_reap_marker` is the only name-based path. It matches the
#     run's own unique temp directory, which proves the process belongs to this
#     run and not to another agent's concurrent gate.
#   * `victory_hygiene_lock_acquire` records ownership; `victory_hygiene_lock_release`
#     removes the directory only when this process created it, exactly once.
#
# Cleanup contract for a script:
#   1. stop its private wineserver (if any);
#   2. `victory_hygiene_reap`          -> TERM, bounded wait, KILL, bounded wait;
#   3. `victory_hygiene_reap_marker "$WORK"` -> ownership-proven last resort;
#   4. `victory_hygiene_lock_release`  -> remove only an owned lock, once;
#   5. `victory_hygiene_remove_work`   -> remove only its own temp dir, once;
#   6. `victory_hygiene_finalize "$status"` -> exit 13 if a tracked PID survived,
#      otherwise exit with the run's real status.
#
# Exit code 13 means "a child started by this run survived cleanup"; scripts that
# use this helper document it in their header. It is deliberately distinct from
# the gate's 12 (PARTIAL with named gaps) and from ordinary setup failures.
#
# Requires: /bin/sh, kill(1), sleep(1) accepting fractional seconds (macOS BSD
# sleep does), and `wait` for direct children. It must be safe under `set -eu`:
# every expansion has a default and `wait` on an already-reaped PID is guarded.

VICTORY_HYGIENE_PIDS="${VICTORY_HYGIENE_PIDS:-}"
VICTORY_HYGIENE_LOCK="${VICTORY_HYGIENE_LOCK:-}"
VICTORY_HYGIENE_LOCK_OWNED="${VICTORY_HYGIENE_LOCK_OWNED:-0}"
VICTORY_HYGIENE_WORK="${VICTORY_HYGIENE_WORK:-}"
VICTORY_HYGIENE_WORK_REMOVED="${VICTORY_HYGIENE_WORK_REMOVED:-0}"
VICTORY_HYGIENE_GRACE="${VICTORY_HYGIENE_GRACE:-5}"
VICTORY_HYGIENE_SURVIVORS="${VICTORY_HYGIENE_SURVIVORS:-}"
VICTORY_HYGIENE_ESCALATED="${VICTORY_HYGIENE_ESCALATED:-0}"
VICTORY_HYGIENE_FINALIZED="${VICTORY_HYGIENE_FINALIZED:-0}"

# Record a direct child PID ($!) so cleanup owns it.
victory_hygiene_track() {
    if [ "$#" -ne 1 ] || [ -z "${1:-}" ]; then
        echo "process-hygiene: track requires one PID" >&2
        return 2
    fi
    case "$1" in
        *[!0-9]*) echo "process-hygiene: not a PID: $1" >&2; return 2 ;;
    esac
    VICTORY_HYGIENE_PIDS="${VICTORY_HYGIENE_PIDS} $1"
}

# Forget a PID that was already waited on normally.
victory_hygiene_untrack() {
    _vh_kept=""
    for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
        if [ "$_vh_pid" != "${1:-}" ]; then
            _vh_kept="$_vh_kept $_vh_pid"
        fi
    done
    VICTORY_HYGIENE_PIDS="$_vh_kept"
}

# kill -0 is safe on any PID: it only reports whether the process still exists.
victory_hygiene_alive() {
    if [ -z "${1:-}" ]; then
        return 1
    fi
    kill -0 "$1" 2>/dev/null
}

# Bind the session temp dir; cleanup removes exactly this dir and no other.
victory_hygiene_work() {
    VICTORY_HYGIENE_WORK="${1:-}"
    VICTORY_HYGIENE_WORK_REMOVED=0
}

# 1 when the caller asked to keep logs (KEEP=1 is the historical spelling).
victory_hygiene_keep_logs() {
    if [ "${VICTORY_KEEP_LOGS:-0}" = 1 ] || [ "${KEEP:-0}" = 1 ]; then
        return 0
    fi
    return 1
}

# mkdir-based exclusive GPU lock. Returns only after this process owns it.
# A fresh lock is never removed; only one older than 60 minutes is reclaimed.
victory_hygiene_lock_acquire() {
    _vh_lock="${1:-}"
    if [ -z "$_vh_lock" ]; then
        echo "process-hygiene: lock_acquire requires a directory" >&2
        return 2
    fi
    if [ "$VICTORY_HYGIENE_LOCK_OWNED" = 1 ] && [ "$VICTORY_HYGIENE_LOCK" = "$_vh_lock" ]; then
        return 0
    fi
    while :; do
        if mkdir "$_vh_lock" 2>/dev/null; then
            VICTORY_HYGIENE_LOCK="$_vh_lock"
            VICTORY_HYGIENE_LOCK_OWNED=1
            return 0
        fi
        if [ -d "$_vh_lock" ] && [ -n "$(find "$_vh_lock" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then
            rmdir "$_vh_lock" 2>/dev/null || true
        fi
        sleep 0.5
    done
}

# Remove the lock only if we created it; safe to call more than once.
victory_hygiene_lock_release() {
    if [ "$VICTORY_HYGIENE_LOCK_OWNED" = 1 ] && [ -n "${VICTORY_HYGIENE_LOCK:-}" ]; then
        rmdir "$VICTORY_HYGIENE_LOCK" 2>/dev/null || true
        VICTORY_HYGIENE_LOCK_OWNED=0
        VICTORY_HYGIENE_LOCK=""
    fi
}

# Wait for one tracked PID with a bounded timeout. On timeout: TERM, grace,
# KILL, reap; returns 124. Otherwise returns the child's own exit status.
# Call inside `if`/`||` when the script runs under `set -e`.
victory_hygiene_wait_pid() {
    _vh_pid="${1:-}"
    _vh_timeout="${2:-1800}"
    if [ -z "$_vh_pid" ]; then
        return 2
    fi
    _vh_deadline=$(( $(date +%s) + _vh_timeout ))
    while victory_hygiene_alive "$_vh_pid"; do
        if [ "$(date +%s)" -ge "$_vh_deadline" ]; then
            echo "process-hygiene: PID $_vh_pid exceeded ${_vh_timeout}s; stopping it" >&2
            kill -TERM "$_vh_pid" 2>/dev/null || true
            _vh_ticks=0
            _vh_limit=$(( ${VICTORY_HYGIENE_GRACE:-5} * 10 ))
            while [ "$_vh_ticks" -lt "$_vh_limit" ]; do
                if ! victory_hygiene_alive "$_vh_pid"; then
                    break
                fi
                sleep 0.1
                _vh_ticks=$((_vh_ticks + 1))
            done
            if victory_hygiene_alive "$_vh_pid"; then
                VICTORY_HYGIENE_ESCALATED=1
                kill -KILL "$_vh_pid" 2>/dev/null || true
            fi
            wait "$_vh_pid" 2>/dev/null || true
            return 124
        fi
        sleep 0.1
    done
    _vh_rc=0
    wait "$_vh_pid" 2>/dev/null || _vh_rc=$?
    return "$_vh_rc"
}

# Run a command as a tracked background child and wait for it with a bound.
# Redirections on the call site are inherited by the child.
victory_hygiene_run() {
    "$@" &
    _vh_pid=$!
    victory_hygiene_track "$_vh_pid"
    _vh_rc=0
    victory_hygiene_wait_pid "$_vh_pid" "${VICTORY_HYGIENE_RUN_TIMEOUT:-1800}" || _vh_rc=$?
    victory_hygiene_untrack "$_vh_pid"
    return "$_vh_rc"
}

# TERM every tracked PID, wait a bounded grace, KILL only survivors we started,
# then reap them. A PID that already exited is never waited on blindly.
victory_hygiene_reap() {
    _vh_limit=$(( ${VICTORY_HYGIENE_GRACE:-5} * 10 ))
    for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
        if victory_hygiene_alive "$_vh_pid"; then
            kill -TERM "$_vh_pid" 2>/dev/null || true
        fi
    done
    _vh_ticks=0
    while [ "$_vh_ticks" -lt "$_vh_limit" ]; do
        _vh_any=0
        for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
            if victory_hygiene_alive "$_vh_pid"; then
                _vh_any=1
                break
            fi
        done
        if [ "$_vh_any" -eq 0 ]; then
            break
        fi
        sleep 0.1
        _vh_ticks=$((_vh_ticks + 1))
    done
    for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
        if victory_hygiene_alive "$_vh_pid"; then
            VICTORY_HYGIENE_ESCALATED=1
            kill -KILL "$_vh_pid" 2>/dev/null || true
        fi
    done
    # Reap; guard every wait because a fast child may already have exited.
    for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
        wait "$_vh_pid" 2>/dev/null || true
    done
    # Anything still alive after KILL+wait is a real survivor.
    for _vh_pid in ${VICTORY_HYGIENE_PIDS:-}; do
        if victory_hygiene_alive "$_vh_pid"; then
            VICTORY_HYGIENE_SURVIVORS="$VICTORY_HYGIENE_SURVIVORS $_vh_pid"
        fi
    done
    VICTORY_HYGIENE_PIDS=""
}

# Ownership-proven last resort: processes whose argv names this run's unique
# temp directory. Never a global name match such as `victory-irconverter`.
victory_hygiene_reap_marker() {
    _vh_marker="${1:-}"
    if [ -z "$_vh_marker" ]; then
        return 0
    fi
    if ! command -v pgrep >/dev/null 2>&1; then
        return 0
    fi
    _vh_matches="$(pgrep -f "$_vh_marker" 2>/dev/null || true)"
    _vh_owned=""
    for _vh_pid in $_vh_matches; do
        if [ "$_vh_pid" != "$$" ]; then
            _vh_owned="$_vh_owned $_vh_pid"
        fi
    done
    if [ -z "$_vh_owned" ]; then
        return 0
    fi
    echo "process-hygiene: reaping marker-owned pids:$_vh_owned" >&2
    for _vh_pid in $_vh_owned; do
        kill -TERM "$_vh_pid" 2>/dev/null || true
    done
    _vh_ticks=0
    _vh_limit=$(( ${VICTORY_HYGIENE_GRACE:-5} * 10 ))
    while [ "$_vh_ticks" -lt "$_vh_limit" ]; do
        _vh_any=0
        for _vh_pid in $_vh_owned; do
            if victory_hygiene_alive "$_vh_pid"; then
                _vh_any=1
                break
            fi
        done
        if [ "$_vh_any" -eq 0 ]; then
            break
        fi
        sleep 0.1
        _vh_ticks=$((_vh_ticks + 1))
    done
    for _vh_pid in $_vh_owned; do
        if victory_hygiene_alive "$_vh_pid"; then
            VICTORY_HYGIENE_ESCALATED=1
            kill -KILL "$_vh_pid" 2>/dev/null || true
        fi
    done
    _vh_ticks=0
    while [ "$_vh_ticks" -lt 20 ]; do
        _vh_any=0
        for _vh_pid in $_vh_owned; do
            if victory_hygiene_alive "$_vh_pid"; then
                _vh_any=1
                break
            fi
        done
        if [ "$_vh_any" -eq 0 ]; then
            break
        fi
        sleep 0.1
        _vh_ticks=$((_vh_ticks + 1))
    done
    for _vh_pid in $_vh_owned; do
        if victory_hygiene_alive "$_vh_pid"; then
            VICTORY_HYGIENE_SURVIVORS="$VICTORY_HYGIENE_SURVIVORS $_vh_pid"
        fi
    done
}

# Remove the owned temp dir exactly once. A log-keeping caller only keeps it.
victory_hygiene_remove_work() {
    if [ "$VICTORY_HYGIENE_WORK_REMOVED" = 1 ]; then
        return 0
    fi
    if [ -z "${VICTORY_HYGIENE_WORK:-}" ]; then
        return 0
    fi
    if victory_hygiene_keep_logs; then
        echo "process-hygiene: kept session files in $VICTORY_HYGIENE_WORK"
        VICTORY_HYGIENE_WORK_REMOVED=1
        return 0
    fi
    rm -rf "$VICTORY_HYGIENE_WORK" 2>/dev/null || true
    VICTORY_HYGIENE_WORK_REMOVED=1
}

# Post-run assertion: fail with the dedicated code 13 if any PID this run
# started is still alive after the reap. Otherwise preserve the real status.
victory_hygiene_finalize() {
    _vh_status="${1:-0}"
    if [ "$VICTORY_HYGIENE_FINALIZED" = 1 ]; then
        return 0
    fi
    VICTORY_HYGIENE_FINALIZED=1
    if [ -n "${VICTORY_HYGIENE_SURVIVORS:-}" ]; then
        echo "PROCESS_ORPHAN_SURVIVED pids=${VICTORY_HYGIENE_SURVIVORS# }" >&2
        echo "process-hygiene: exit 13: a child this run started survived cleanup" >&2
        exit 13
    fi
    exit "$_vh_status"
}
