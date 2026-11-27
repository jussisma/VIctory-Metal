#!/bin/sh
# C7 / Wave 7A: hermetic self-test for `process-hygiene.sh`.
#
# No GPU, no Wine, no Rust build, no network. It proves:
#   1. a child that ignores SIGTERM is escalated to SIGKILL and reaped;
#   2. a lock directory this run created is removed exactly once, and a lock it
#      did not create is left alone;
#   3. a simulated mid-run `exit 1` still reaps the child, removes the lock and
#      preserves exit 1;
#   4. the dedicated orphan-survivor exit code 13 (and that 12 is preserved);
#   5. `sh -n` parses every script in this directory.
#
# Usage: sh tools/macos/metal-poc/test-process-hygiene.sh
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
HELPER="$HERE/process-hygiene.sh"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/victory-hygiene-test.XXXXXXXX")"

cleanup_test() {
    rm -rf "$TMP"
}
trap cleanup_test EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

fail() {
    echo "FAIL: test-process-hygiene: $1" >&2
    exit 1
}

pass() {
    echo "PASS: $1"
}

[ -f "$HELPER" ] || fail "missing $HELPER"
# The tests run the helper in three separate shells; keep the grace period short
# so the suite stays well under a minute.
VICTORY_HYGIENE_GRACE=1
export VICTORY_HYGIENE_GRACE

# --- 1. KILL escalation for a SIGTERM-ignoring child ------------------------
(
    . "$HELPER"
    sh -c 'trap "" TERM; while :; do :; done' &
    dummy=$!
    sleep 0.2
    if ! kill -0 "$dummy" 2>/dev/null; then
        fail "dummy child did not start"
    fi
    victory_hygiene_track "$dummy"
    victory_hygiene_reap
    if victory_hygiene_alive "$dummy"; then
        fail "SIGTERM-ignoring child survived reap"
    fi
    if [ "$VICTORY_HYGIENE_ESCALATED" != 1 ]; then
        fail "cleanup did not escalate to SIGKILL"
    fi
) || fail "KILL escalation test failed"
pass "SIGTERM-ignoring child escalated to SIGKILL and reaped"

# --- 2. Owned lock removed; foreign lock untouched --------------------------
(
    . "$HELPER"
    owned="$TMP/owned.lock"
    victory_hygiene_lock_acquire "$owned"
    if [ ! -d "$owned" ]; then
        fail "owned lock was not created"
    fi
    victory_hygiene_lock_release
    if [ -d "$owned" ]; then
        fail "owned lock was not removed"
    fi
) || fail "owned lock test failed"
mkdir "$TMP/foreign.lock"
(
    . "$HELPER"
    victory_hygiene_lock_release
) || fail "foreign lock release failed"
if [ ! -d "$TMP/foreign.lock" ]; then
    fail "cleanup removed a lock it did not create"
fi
rmdir "$TMP/foreign.lock"
pass "owned lock removed once; foreign lock preserved"

# --- 3. Simulated mid-run `exit 1` still reaps and preserves status ---------
cat >"$TMP/midrun.sh" <<'EOS'
#!/bin/sh
set -eu
HELPER_DIR="$1"
WORK="$2"
LOCKDIR="$3"
PIDFILE="$4"
. "$HELPER_DIR/process-hygiene.sh"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    victory_hygiene_reap
    victory_hygiene_lock_release
    victory_hygiene_remove_work
    victory_hygiene_finalize "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

mkdir -p "$WORK"
victory_hygiene_work "$WORK"
victory_hygiene_lock_acquire "$LOCKDIR"

# This child ignores SIGTERM, so only the KILL escalation can reap it.
sh -c 'trap "" TERM; while :; do :; done' &
child=$!
victory_hygiene_track "$child"
printf '%s' "$child" >"$PIDFILE"
sleep 0.2

# Simulate a failing run from the middle of the script.
exit 1
EOS
set +e
sh "$TMP/midrun.sh" "$HERE" "$TMP/midrun-work" "$TMP/midrun.lock" "$TMP/child.pid"
midrun_rc=$?
set -e
[ "$midrun_rc" -eq 1 ] || fail "mid-run exit 1 became exit $midrun_rc"
[ ! -d "$TMP/midrun.lock" ] || fail "mid-run lock survived cleanup"
[ ! -d "$TMP/midrun-work" ] || fail "mid-run temp dir survived cleanup"
midrun_child="$(cat "$TMP/child.pid")"
if kill -0 "$midrun_child" 2>/dev/null; then
    fail "mid-run child $midrun_child survived cleanup"
fi
pass "mid-run exit 1 reaped the child, removed the lock and preserved exit 1"

# --- 4. Exit-code contract: 13 orphan survivor, 12 preserved ---------------
set +e
sh -c '. "$1/process-hygiene.sh"; VICTORY_HYGIENE_SURVIVORS="4242"; victory_hygiene_finalize 0' \
    sh "$HERE" >/dev/null 2>&1
orphan_rc=$?
sh -c '. "$1/process-hygiene.sh"; victory_hygiene_finalize 12' sh "$HERE" >/dev/null 2>&1
partial_rc=$?
sh -c '. "$1/process-hygiene.sh"; victory_hygiene_finalize 0' sh "$HERE" >/dev/null 2>&1
clean_rc=$?
set -e
[ "$orphan_rc" -eq 13 ] || fail "orphan survivor exit code was $orphan_rc, expected 13"
[ "$partial_rc" -eq 12 ] || fail "PARTIAL exit code was $partial_rc, expected 12"
[ "$clean_rc" -eq 0 ] || fail "clean exit code was $clean_rc, expected 0"
pass "exit contract: 0 clean, 12 PARTIAL, 13 orphan survivor"

# --- 4b. VICTORY_KEEP_LOGS keeps the dir; reaping is unaffected ------------
(
    . "$HELPER"
    kept="$TMP/keep-work"
    mkdir -p "$kept"
    victory_hygiene_work "$kept"
    VICTORY_KEEP_LOGS=1
    export VICTORY_KEEP_LOGS
    victory_hygiene_remove_work
    if [ ! -d "$kept" ]; then
        fail "VICTORY_KEEP_LOGS=1 removed the session dir"
    fi
    sh -c 'trap "" TERM; while :; do :; done' &
    keep_child=$!
    sleep 0.2
    victory_hygiene_track "$keep_child"
    victory_hygiene_reap
    if victory_hygiene_alive "$keep_child"; then
        fail "VICTORY_KEEP_LOGS=1 skipped child reaping"
    fi
) || fail "keep-logs contract test failed"
pass "VICTORY_KEEP_LOGS=1 keeps logs and still reaps children"

# --- 5. Every script in the directory parses under `sh -n` -----------------
for script in "$HERE"/*.sh; do
    sh -n "$script" || fail "sh -n failed: $script"
done
pass "sh -n parsed every script in $HERE"

echo "PASS: test-process-hygiene"
