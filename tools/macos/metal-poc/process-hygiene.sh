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
#   * `victory_hygiene_lock_acquire` records ownership in `DIR/owner` as
#     "<pid> <epoch>" and reclaims a lock whose recorded PID is no longer
#     running. `victory_hygiene_lock_release` removes the owner record and the
#     directory only when this process created it, exactly once.
#
# Lock protocol (ADR-032, shared with tools/macos/metal-poc/gpu-lock):
#   * atomic `mkdir` acquisition, owner record written immediately after;
#   * a recorded PID that `kill -0` cannot find is stale and is reclaimed, at
#     most once per observed dead PID, under a `DIR/.reclaim` mutex;
#   * a directory with no readable owner record is a racer while it is younger
#     than one minute and a crashed creator after that;
#   * a live holder is waited for up to `VICTORY_HYGIENE_LOCK_WAIT` seconds
#     (default 900), then the acquire fails with exit 75 and a clear message.
#     It never spins forever and never bypasses a live holder;
#   * a lock owned by this process or one of its ancestors is re-entrant: the
#     acquire returns 0, sets VICTORY_HYGIENE_LOCK_INHERITED=1, and release
#     leaves that lock alone. A harness wrapped by `gpu-lock` on the default
#     lock path therefore never stalls on, or steals, its own wrapper's lock;
#   * the helper implements this protocol inline. It never invokes the gpu-lock
#     script, so a harness that is itself wrapped by `gpu-lock` cannot deadlock
#     on the default lock path.
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
VICTORY_HYGIENE_LOCK_INHERITED="${VICTORY_HYGIENE_LOCK_INHERITED:-0}"
VICTORY_HYGIENE_WORK="${VICTORY_HYGIENE_WORK:-}"
VICTORY_HYGIENE_WORK_REMOVED="${VICTORY_HYGIENE_WORK_REMOVED:-0}"
VICTORY_HYGIENE_GRACE="${VICTORY_HYGIENE_GRACE:-5}"
VICTORY_HYGIENE_LOCK_WAIT="${VICTORY_HYGIENE_LOCK_WAIT:-900}"
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

# --- PID-verified GPU lock (ADR-032) ---------------------------------------
# Seconds since the epoch, used for the owner record.
_vh_now() {
    date +%s
}

# True when DIR exists and its mtime is at least one minute old.
# `date -r` gives seconds resolution; BSD `find -mmin` rounds up, so it would
# call a one-second-old directory "one minute old" and reclaim a live racer.
_vh_lock_older_than_minute() {
    [ -n "${1:-}" ] || return 1
    [ -d "${1:-}" ] || return 1
    _vh_om_mtime="$(date -r "$1" +%s 2>/dev/null || true)"
    case "$_vh_om_mtime" in
        ''|*[!0-9]*)
            # No usable `date -r`: fall back to find's minute resolution but
            # demand more than one minute, so a fresh racer is never reclaimed.
            [ -n "$(find "$1" -maxdepth 0 -mmin +1 2>/dev/null)" ]
            return $?
            ;;
    esac
    [ $(( $(_vh_now) - _vh_om_mtime )) -ge 60 ]
}

# Print the recorded owner PID, or nothing when absent/unreadable. Succeeds.
_vh_lock_owner_pid() {
    _vh_op_line=""
    if [ -f "${1:-}/owner" ]; then
        _vh_op_line="$(cat "$1/owner" 2>/dev/null || true)"
    fi
    _vh_op_pid="${_vh_op_line%% *}"
    case "$_vh_op_pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    printf '%s\n' "$_vh_op_pid"
}

# True when PID is this process or one of its ancestors. Makes nesting safe:
# a wrapper that already holds the lock (for example `gpu-lock` running this
# harness on the default lock path) is not a competitor, and this process must
# not release its lock.
_vh_lock_pid_is_ancestor() {
    _vh_sa_want="${1:-}"
    [ -n "$_vh_sa_want" ] || return 1
    _vh_sa_cur="$$"
    _vh_sa_guard=0
    while [ "$_vh_sa_guard" -lt 64 ]; do
        if [ "$_vh_sa_cur" = "$_vh_sa_want" ]; then
            return 0
        fi
        [ "$_vh_sa_cur" -gt 1 ] || return 1
        _vh_sa_cur="$(ps -o ppid= -p "$_vh_sa_cur" 2>/dev/null | tr -d ' ')"
        case "$_vh_sa_cur" in
            ''|*[!0-9]*) return 1 ;;
        esac
        _vh_sa_guard=$((_vh_sa_guard + 1))
    done
    return 1
}

# Remove a stale lock directory. Returns 0 only when this call removed it.
# `DIR/.reclaim` serialises reclaimers; a new owner can only appear after DIR is
# gone, so a reclaimer can never delete a live holder's owner record.
_vh_lock_reclaim() {
    _vh_rd="${1:-}"
    if [ -z "$_vh_rd" ] || [ ! -d "$_vh_rd" ]; then
        return 1
    fi
    if ! mkdir "$_vh_rd/.reclaim" 2>/dev/null; then
        # A crashed reclaimer can leave the mutex behind. Only a mutex older
        # than a minute on an already-stale lock is removed, so a live
        # reclaimer (which completes in milliseconds) is never disturbed.
        if _vh_lock_older_than_minute "$_vh_rd/.reclaim"; then
            rmdir "$_vh_rd/.reclaim" 2>/dev/null || true
            mkdir "$_vh_rd/.reclaim" 2>/dev/null || return 1
        else
            return 1
        fi
    fi
    _vh_rpid="$(_vh_lock_owner_pid "$_vh_rd")"
    if [ -n "$_vh_rpid" ] && victory_hygiene_alive "$_vh_rpid"; then
        rmdir "$_vh_rd/.reclaim" 2>/dev/null || true
        return 1
    fi
    if [ -n "$_vh_rpid" ]; then
        echo "process-hygiene: reclaiming stale lock $_vh_rd (pid $_vh_rpid is not running)" >&2
    else
        echo "process-hygiene: reclaiming aged lock $_vh_rd with no readable owner record" >&2
    fi
    rm -f "$_vh_rd/owner" 2>/dev/null || true
    rmdir "$_vh_rd/.reclaim" 2>/dev/null || true
    rmdir "$_vh_rd" 2>/dev/null || true
    [ ! -d "$_vh_rd" ]
}

# mkdir-based exclusive GPU lock. Returns only after this process owns it.
# A dead recorded PID is reclaimed (once per observed PID); a live holder is
# waited for up to VICTORY_HYGIENE_LOCK_WAIT seconds, then this returns 75.
victory_hygiene_lock_acquire() {
    _vh_lock="${1:-}"
    if [ -z "$_vh_lock" ]; then
        echo "process-hygiene: lock_acquire requires a directory" >&2
        return 2
    fi
    if [ "$VICTORY_HYGIENE_LOCK_OWNED" = 1 ] && [ "$VICTORY_HYGIENE_LOCK" = "$_vh_lock" ]; then
        return 0
    fi
    _vh_wait="${VICTORY_HYGIENE_LOCK_WAIT:-900}"
    case "$_vh_wait" in
        ''|*[!0-9]*) _vh_wait=900 ;;
    esac
    _vh_deadline=$(( $(_vh_now) + _vh_wait ))
    _vh_reclaimed_key=""
    _vh_noent=0
    while :; do
        if mkdir "$_vh_lock" 2>/dev/null; then
            if ! printf '%s %s\n' "$$" "$(_vh_now)" >"$_vh_lock/owner" 2>/dev/null; then
                rmdir "$_vh_lock" 2>/dev/null || true
                echo "process-hygiene: cannot write the owner record in $_vh_lock" >&2
                return 2
            fi
            VICTORY_HYGIENE_LOCK="$_vh_lock"
            VICTORY_HYGIENE_LOCK_OWNED=1
            return 0
        fi
        if [ ! -d "$_vh_lock" ]; then
            if [ -e "$_vh_lock" ]; then
                echo "process-hygiene: lock path $_vh_lock exists but is not a directory" >&2
                return 2
            fi
            # mkdir failed and nothing is there: a racer removed it in this
            # instant, or the parent is missing/unwritable. Retry briefly.
            _vh_noent=$((_vh_noent + 1))
            if [ "$_vh_noent" -ge 5 ]; then
                echo "process-hygiene: cannot create the lock directory $_vh_lock (missing or unwritable parent)" >&2
                return 2
            fi
            sleep 0.1
            continue
        fi
        _vh_noent=0
        _vh_pid="$(_vh_lock_owner_pid "$_vh_lock")"
        _vh_stale=0
        if [ -n "$_vh_pid" ]; then
            if victory_hygiene_alive "$_vh_pid"; then
                if _vh_lock_pid_is_ancestor "$_vh_pid"; then
                    # Re-entrant: an ancestor (for example `gpu-lock` wrapping
                    # this harness) already holds the lock and will release it.
                    # Reuse its ownership; never release or reclaim it here.
                    VICTORY_HYGIENE_LOCK_INHERITED=1
                    return 0
                fi
            else
                _vh_stale=1
            fi
        elif _vh_lock_older_than_minute "$_vh_lock"; then
            _vh_stale=1
        fi
        if [ "$_vh_stale" = 1 ]; then
            _vh_key="${_vh_pid:-none}"
            # At most one reclaim attempt per observed dead PID: two waiters
            # cannot livelock each other, and a freshly created directory
            # without an owner record is still treated as a racer.
            if [ "$_vh_key" != "$_vh_reclaimed_key" ]; then
                _vh_reclaimed_key="$_vh_key"
                if _vh_lock_reclaim "$_vh_lock"; then
                    continue
                fi
            fi
        fi
        if [ "$(_vh_now)" -ge "$_vh_deadline" ]; then
            echo "process-hygiene: lock $_vh_lock is held by pid ${_vh_pid:-unknown} after ${_vh_wait}s; refusing to run" >&2
            return 75
        fi
        sleep 0.2
    done
}

# Remove the lock only if we created it; safe to call more than once.
victory_hygiene_lock_release() {
    if [ "$VICTORY_HYGIENE_LOCK_OWNED" = 1 ] && [ -n "${VICTORY_HYGIENE_LOCK:-}" ]; then
        _vh_rl_dir="$VICTORY_HYGIENE_LOCK"
        _vh_rl_pid="$(_vh_lock_owner_pid "$_vh_rl_dir")"
        VICTORY_HYGIENE_LOCK_OWNED=0
        VICTORY_HYGIENE_LOCK=""
        if [ "$_vh_rl_pid" = "$$" ]; then
            rm -f "$_vh_rl_dir/owner" 2>/dev/null || true
            rmdir "$_vh_rl_dir" 2>/dev/null || true
        fi
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
