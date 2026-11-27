#!/bin/sh
# ADR-032: hermetic self-test for `gpu-lock`.
#
# No GPU, no Wine, no Rust build, no network. Every lock and helper lives under
# a private $TMPDIR directory. It proves:
#   a. a live holder blocks a second acquire and `--status` reports held;
#   b. a holder killed with SIGKILL is reclaimed by the next acquire;
#   c. the lock is released on a normal exit and on SIGTERM of the wrapper,
#      and the wrapper stops the child it started before releasing;
#   d. concurrent acquirers are never inside the critical section together;
#   e. `--wait 1` against a live holder exits 75 within the bound;
#   f. `--status` is free on a missing lock and stale on a dead-PID lock, and it
#      never creates or removes anything;
#   g. a fresh ownerless directory is a racer: never reclaimed within a minute;
#   h. nested gpu-lock on one directory reuses its ancestor's lock instantly;
#   i. process-hygiene.sh and gpu-lock see one protocol in both directions.
#
# Usage: sh tools/macos/metal-poc/test-gpu-lock.sh
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
LOCK_BIN="$HERE/gpu-lock"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/victory-gpu-lock-test.XXXXXXXX")"
PIDS=""
PASSED=0

cleanup_test() {
    for p in $PIDS; do
        kill -KILL "$p" 2>/dev/null || true
        # Consume the job status so the shell does not print a "Killed" notice.
        wait "$p" 2>/dev/null || true
    done
    rm -rf "$TMP"
}
trap cleanup_test EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

fail() {
    echo "FAIL: test-gpu-lock: $1" >&2
    exit 1
}

pass() {
    PASSED=$((PASSED + 1))
    echo "PASS: $1"
}

track() {
    PIDS="$PIDS $1"
}

# Forget a PID already waited on, so cleanup never signals a recycled PID.
untrack() {
    _ut_kept=""
    for _ut_pid in $PIDS; do
        if [ "$_ut_pid" != "$1" ]; then
            _ut_kept="$_ut_kept $_ut_pid"
        fi
    done
    PIDS="$_ut_kept"
}

# wait_for_file PATH [TENTHS]
wait_for_file() {
    n=0
    while [ "$n" -lt "${2:-100}" ]; do
        [ -e "$1" ] && return 0
        sleep 0.1
        n=$((n + 1))
    done
    return 1
}

# wait_for_pid_gone PID [TENTHS] -- only for PIDs this shell cannot `wait` on.
wait_for_pid_gone() {
    n=0
    while [ "$n" -lt "${2:-50}" ]; do
        kill -0 "$1" 2>/dev/null || return 0
        sleep 0.1
        n=$((n + 1))
    done
    return 1
}

[ -f "$LOCK_BIN" ] || fail "missing $LOCK_BIN"
[ -x "$LOCK_BIN" ] || fail "$LOCK_BIN is not executable"

# The holder runs the lock, records the wrapped child's PID, touches a ready
# file and then sleeps so the lock stays held while the test probes it.
cat >"$TMP/holder.sh" <<'EOS'
#!/bin/sh
set -eu
lock_bin="$1"
lock_dir="$2"
ready="$3"
exec "$lock_bin" --lock "$lock_dir" -- sh -c \
    'printf "%s\n" "$$" > "$1"; : > "$2"; exec sleep 120' \
    sh "$ready.pid" "$ready"
EOS

# Start a holder and leave its wrapper PID in HOLDER_PID. A function is used
# instead of command substitution so the background child never inherits a
# capture pipe (that would block until the holder exits).
start_holder() {
    sh "$TMP/holder.sh" "$LOCK_BIN" "$1" "$2" &
    HOLDER_PID=$!
    track "$HOLDER_PID"
}

# --- a + e. Live holder blocks the acquirer; --status reports held ----------
A_LOCK="$TMP/a.lock"
A_READY="$TMP/a.ready"
start_holder "$A_LOCK" "$A_READY"
A_WRAPPER="$HOLDER_PID"
wait_for_file "$A_READY" 100 || fail "holder a did not acquire the lock"
[ -f "$A_READY.pid" ] || fail "holder a did not record its child PID"
A_CHILD="$(cat "$A_READY.pid")"
track "$A_CHILD"

set +e
a_status="$(sh "$LOCK_BIN" --lock "$A_LOCK" --status)"
a_status_rc=$?
set -e
[ "$a_status_rc" -eq 3 ] || fail "--status on a held lock exited $a_status_rc, expected 3: $a_status"
case "$a_status" in
    *status=held*) : ;;
    *) fail "--status did not report held: $a_status" ;;
esac
case "$a_status" in
    *"pid=$A_WRAPPER"*) : ;;
    *) fail "--status reported the wrong PID (want the acquiring wrapper $A_WRAPPER): $a_status" ;;
esac

set +e
sh "$LOCK_BIN" --lock "$A_LOCK" --wait 1 -- true >/dev/null 2>&1
a_acquire_rc=$?
set -e
[ "$a_acquire_rc" -eq 75 ] || fail "second acquire on a live holder exited $a_acquire_rc, expected 75"
[ -d "$A_LOCK" ] || fail "a failed acquire removed the live holder's lock"
pass "a: live holder blocks a second acquire (exit 75); --status held pid=$A_CHILD"

# e: the same holder, with the wait measured.
e_start="$(date +%s)"
set +e
sh "$LOCK_BIN" --lock "$A_LOCK" --wait 1 -- true >/dev/null 2>&1
e_rc=$?
set -e
e_end="$(date +%s)"
e_elapsed=$((e_end - e_start))
[ "$e_rc" -eq 75 ] || fail "--wait 1 exited $e_rc, expected 75"
[ "$e_elapsed" -ge 1 ] || fail "--wait 1 returned after ${e_elapsed}s; it did not wait for the holder"
[ "$e_elapsed" -le 15 ] || fail "--wait 1 waited ${e_elapsed}s; the bound is not enforced"
pass "e: --wait 1 on a live holder exited 75 after ${e_elapsed}s"

# --- c1. SIGTERM of the wrapper releases the lock and stops the child -------
kill -TERM "$A_WRAPPER" 2>/dev/null || true
a_wrapper_rc=0
wait "$A_WRAPPER" 2>/dev/null || a_wrapper_rc=$?
untrack "$A_WRAPPER"
[ "$a_wrapper_rc" -eq 143 ] || fail "wrapper exited $a_wrapper_rc on SIGTERM, expected 143"
if [ -d "$A_LOCK" ]; then
    fail "the lock survived SIGTERM of the wrapper"
fi
wait_for_pid_gone "$A_CHILD" 100 || fail "the wrapper did not stop its child on SIGTERM"
untrack "$A_CHILD"
pass "c1: SIGTERM released the lock and stopped the wrapped child"

# --- c2. Normal exit releases the lock -------------------------------------
C_LOCK="$TMP/c-normal.lock"
if ! sh "$LOCK_BIN" --lock "$C_LOCK" -- true; then
    fail "a normal locked run failed"
fi
if [ -d "$C_LOCK" ]; then
    fail "the lock survived a normal exit"
fi
pass "c2: normal exit released the lock"

# --- b. SIGKILLed holder leaves a stale lock that the next acquire reclaims -
B_LOCK="$TMP/b.lock"
B_READY="$TMP/b.ready"
start_holder "$B_LOCK" "$B_READY"
B_WRAPPER="$HOLDER_PID"
wait_for_file "$B_READY" 100 || fail "holder b did not acquire the lock"
B_CHILD="$(cat "$B_READY.pid")"
track "$B_CHILD"

set +e
b_held_status="$(sh "$LOCK_BIN" --lock "$B_LOCK" --status)"
b_held_rc=$?
set -e
[ "$b_held_rc" -eq 3 ] || fail "--status before the kill exited $b_held_rc, expected 3"
[ -f "$B_LOCK/owner" ] || fail "holder b wrote no owner record"

kill -KILL "$B_WRAPPER" 2>/dev/null || true
b_wrapper_rc=0
wait "$B_WRAPPER" 2>/dev/null || b_wrapper_rc=$?
untrack "$B_WRAPPER"
[ "$b_wrapper_rc" -eq 137 ] || fail "SIGKILLed wrapper exited $b_wrapper_rc, expected 137"
[ -d "$B_LOCK" ] || fail "SIGKILL left no stale lock to reclaim"

set +e
b_stale_status="$(sh "$LOCK_BIN" --lock "$B_LOCK" --status)"
b_stale_rc=$?
set -e
[ "$b_stale_rc" -eq 4 ] || fail "--status on a dead-PID lock exited $b_stale_rc, expected 4: $b_stale_status"
case "$b_stale_status" in
    *status=stale*) : ;;
    *) fail "--status did not report stale: $b_stale_status" ;;
esac

b_start="$(date +%s)"
if ! sh "$LOCK_BIN" --lock "$B_LOCK" --wait 5 -- true; then
    fail "acquire did not reclaim the SIGKILLed holder's lock"
fi
b_end="$(date +%s)"
b_elapsed=$((b_end - b_start))
if [ -d "$B_LOCK" ]; then
    fail "the reclaimed lock was not released after the run"
fi
[ "$b_elapsed" -le 5 ] || fail "reclaim took ${b_elapsed}s; the dead PID was treated as a live holder"
kill -KILL "$B_CHILD" 2>/dev/null || true
untrack "$B_CHILD"
pass "b: SIGKILLed holder reclaimed in ${b_elapsed}s and released"

# --- f. --status never creates or removes anything -------------------------
F_LOCK="$TMP/f.lock"
set +e
f_free="$(sh "$LOCK_BIN" --lock "$F_LOCK" --status)"
f_free_rc=$?
set -e
[ "$f_free_rc" -eq 0 ] || fail "--status on a missing lock exited $f_free_rc, expected 0: $f_free"
case "$f_free" in
    *status=free*) : ;;
    *) fail "--status did not report free: $f_free" ;;
esac
[ ! -e "$F_LOCK" ] || fail "--status created $F_LOCK"
pass "f: --status is free on a missing lock and creates nothing"

# --- g. A fresh ownerless directory is a racer, not a stale lock ----------
# Regression guard: a one-second-old directory must never be reclaimed, only a
# directory that has been ownerless for more than a minute.
G_LOCK="$TMP/g.lock"
mkdir "$G_LOCK"
set +e
g_held="$(sh "$LOCK_BIN" --lock "$G_LOCK" --status)"
g_held_rc=$?
set -e
[ "$g_held_rc" -eq 3 ] || fail "--status on a fresh ownerless lock exited $g_held_rc, expected 3: $g_held"
case "$g_held" in
    *status=held*owner-missing*) : ;;
    *) fail "--status did not report a held owner-missing lock: $g_held" ;;
esac
set +e
sh "$LOCK_BIN" --lock "$G_LOCK" --wait 1 -- true >/dev/null 2>&1
g_acquire_rc=$?
set -e
[ "$g_acquire_rc" -eq 75 ] || fail "fresh ownerless lock acquired with $g_acquire_rc, expected 75"
[ -d "$G_LOCK" ] || fail "a fresh ownerless lock was reclaimed (racer treated as stale)"
rmdir "$G_LOCK"
pass "g: fresh ownerless lock waited on, never reclaimed"

# --- h. Nested gpu-lock on one directory reuses the ancestor's lock -------
# A wrapper that already holds the lock must not stall its own child, and must
# not be released twice. Without ancestor reuse the inner call needs --wait to
# expire (75); with it the nested run is immediate.
H_LOCK="$TMP/h.lock"
h_start="$(date +%s)"
if ! sh "$LOCK_BIN" --lock "$H_LOCK" -- sh "$LOCK_BIN" --lock "$H_LOCK" --wait 2 -- true; then
    fail "nested gpu-lock on the same directory failed"
fi
h_end="$(date +%s)"
h_elapsed=$((h_end - h_start))
[ "$h_elapsed" -le 5 ] || fail "nested gpu-lock took ${h_elapsed}s; it did not reuse the ancestor's lock"
[ ! -d "$H_LOCK" ] || fail "nested gpu-lock left the lock behind"
pass "h: nested gpu-lock reused the ancestor's lock in ${h_elapsed}s"

# --- d. Concurrent acquirers never share the critical section --------------
D_LOCK="$TMP/d.lock"
D_SENTINEL="$TMP/d.sentinel"
D_LOG="$TMP/d.collisions"
cat >"$TMP/critical.sh" <<'EOS'
#!/bin/sh
set -eu
lock_bin="$1"
lock_dir="$2"
sentinel="$3"
log="$4"
exec "$lock_bin" --lock "$lock_dir" --wait 60 -- sh -c '
    if [ -e "$1" ]; then
        printf "COLLISION %s\n" "$$" >> "$2"
        exit 9
    fi
    : > "$1"
    sleep 0.3
    rm -f "$1"
' sh "$sentinel" "$log"
EOS

D_PIDS=""
d_i=0
while [ "$d_i" -lt 6 ]; do
    sh "$TMP/critical.sh" "$LOCK_BIN" "$D_LOCK" "$D_SENTINEL" "$D_LOG" &
    D_PIDS="$D_PIDS $!"
    d_i=$((d_i + 1))
done
d_failed=0
for d_pid in $D_PIDS; do
    d_rc=0
    wait "$d_pid" 2>/dev/null || d_rc=$?
    if [ "$d_rc" -ne 0 ]; then
        echo "test-gpu-lock: critical-section worker $d_pid exited $d_rc" >&2
        d_failed=1
    fi
done
[ "$d_failed" -eq 0 ] || fail "a concurrent acquirer did not complete cleanly"
if [ -s "$D_LOG" ]; then
    fail "two acquirers were inside the critical section together: $(cat "$D_LOG")"
fi
if [ -d "$D_LOCK" ]; then
    fail "the lock survived the concurrency test"
fi
pass "d: 6 concurrent acquirers serialised with no overlap (0 collisions)"

# --- i. The helper and gpu-lock share one lock protocol --------------------
# A harness acquires through process-hygiene.sh, a gate wraps it with gpu-lock:
# both must see the same directory, the same owner record and the same rules.
I_LOCK="$TMP/i.lock"
I_READY="$TMP/i.ready"
start_holder "$I_LOCK" "$I_READY"
I_WRAPPER="$HOLDER_PID"
wait_for_file "$I_READY" 100 || fail "holder i did not acquire the lock"
I_CHILD="$(cat "$I_READY.pid")"
track "$I_CHILD"
kill -KILL "$I_WRAPPER" 2>/dev/null || true
wait "$I_WRAPPER" 2>/dev/null || true
untrack "$I_WRAPPER"
[ -d "$I_LOCK" ] || fail "holder i left no stale lock for the helper to reclaim"

i_reclaim_out="$(sh -c '
    set -eu
    . "$1/process-hygiene.sh"
    VICTORY_HYGIENE_LOCK_WAIT=5
    victory_hygiene_lock_acquire "$2" || exit $?
    [ -d "$2" ] || exit 91
    victory_hygiene_lock_release
' sh "$HERE" "$I_LOCK" 2>&1)" || fail "the helper did not reclaim gpu-lock's stale lock: $i_reclaim_out"
[ ! -d "$I_LOCK" ] || fail "the helper left gpu-lock's reclaimed lock behind"
kill -KILL "$I_CHILD" 2>/dev/null || true
untrack "$I_CHILD"

# A lock the helper holds must look held to gpu-lock, not free.
sh -c '
    set -eu
    . "$1/process-hygiene.sh"
    VICTORY_HYGIENE_LOCK_WAIT=30
    victory_hygiene_lock_acquire "$2"
    sleep 30 &
    sleeper=$!
    trap "kill $sleeper 2>/dev/null || true; victory_hygiene_lock_release; exit 143" TERM
    wait "$sleeper" 2>/dev/null || true
    victory_hygiene_lock_release
' sh "$HERE" "$I_LOCK" &
I_HELPER=$!
track "$I_HELPER"
i_n=0
while [ "$i_n" -lt 100 ] && [ ! -f "$I_LOCK/owner" ]; do
    sleep 0.1
    i_n=$((i_n + 1))
done
[ -f "$I_LOCK/owner" ] || fail "the helper holder wrote no owner record"

set +e
sh "$LOCK_BIN" --lock "$I_LOCK" --wait 1 -- true >/dev/null 2>&1
i_blocked_rc=$?
set -e
[ "$i_blocked_rc" -eq 75 ] || fail "gpu-lock did not see the helper's lock as held (exit $i_blocked_rc)"

kill -TERM "$I_HELPER" 2>/dev/null || true
wait "$I_HELPER" 2>/dev/null || true
untrack "$I_HELPER"
[ ! -d "$I_LOCK" ] || fail "the helper holder did not release its own lock"
pass "i: helper and gpu-lock see one protocol (reclaim both ways, held both ways)"

echo "PASS: test-gpu-lock ($PASSED checks)"
