#!/bin/sh
# Hermetic tests for abi-diff.py: fixture logs only, no GPU, no Wine, no title.
#
# Covers the tool's contract: identical traces exit 0, one differing value
# exits 2 and names the call and value index, a missing file exits 1, empty
# logs exit 3, and no fixture path ever reaches stdout. It also feeds the
# parser ANSI colour, interleaved prefixes, two traces on one line, repeated
# calls and a row longer than the eight-value cap.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tool="$here/abi-diff.py"
work=$(mktemp -d "${TMPDIR:-/tmp}/abi-diff.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    printf 'ok: %s\n' "$*"
}

# run NAME EXPECTED-EXIT NATIVE REFERENCE [FLAGS...]
run() {
    name=$1
    expected=$2
    native=$3
    reference=$4
    shift 4
    if python3 "$tool" "$native" "$reference" "$@" >"$work/$name.out" 2>"$work/$name.err"; then
        code=0
    else
        code=$?
    fi
    [ "$code" -eq "$expected" ] ||
        fail "$name: exit $code, wanted $expected (stderr: $(cat "$work/$name.err"))"
}

# --- fixtures -------------------------------------------------------------

cat >"$work/same-a.log" <<'EOF'
2026-11-27T10:00:00Z INFO victory_abi_trace: abi_trace call=CreateDescriptorHeap values=[0, 4, 1, 4096, 0, 0]
2026-11-27T10:00:00Z INFO victory_abi_trace: abi_trace call=GetDescriptorHandleIncrementSize values=[0, 32]
2026-11-27T10:00:00Z INFO victory_abi_trace: abi_trace call=GetCPUDescriptorHandleForHeapStart values=[4096, 0, 0]
EOF
cp "$work/same-a.log" "$work/same-b.log"

cat >"$work/differ.log" <<'EOF'
2026-11-27T10:00:01Z INFO victory_abi_trace: abi_trace call=CreateDescriptorHeap values=[0, 4, 1, 8192, 0, 0]
2026-11-27T10:00:01Z INFO victory_abi_trace: abi_trace call=GetDescriptorHandleIncrementSize values=[0, 32]
2026-11-27T10:00:01Z INFO victory_abi_trace: abi_trace call=GetCPUDescriptorHandleForHeapStart values=[4096, 0, 0]
EOF

: >"$work/empty-a.log"
: >"$work/empty-b.log"

# ANSI colour, an interleaved prefix, two traces on one physical line, a
# repeated call, and a row of ten values (the trace caps at eight).
printf '%s\n' \
    "$(printf '\033[32m2026-11-27T10:00:02Z INFO victory_abi_trace: abi_trace call=CheckFeatureSupport values=[7, 4, 0, 102]\033[0m')" \
    "garbage before abi_trace call=CreateDescriptorHeap values=[0,4,1,4096,0,0] garbage after abi_trace call=GetGPUVirtualAddress values=[4660, 0]" \
    "abi_trace call=CreateDescriptorHeap values=[0, 4, 1, 4096, 0, 0]" \
    "abi_trace call=Wide values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" \
    >"$work/noisy.log"

# --- identical traces agree ----------------------------------------------

run identical 0 "$work/same-a.log" "$work/same-b.log"
grep -q '^same CreateDescriptorHeap' "$work/identical.out" ||
    fail "identical: the call row is not reported as same"
pass "identical traces exit 0"

# --- one differing value exits 2 and names the call and index ------------

run differ 2 "$work/same-a.log" "$work/differ.log"
grep -q 'DIFF CreateDescriptorHeap' "$work/differ.out" ||
    fail "differ: the differing call is not named"
grep -q 'value #3' "$work/differ.out" ||
    fail "differ: the differing value index is not named"
grep -q 'native=4096 reference=8192' "$work/differ.out" ||
    fail "differ: the differing values are not shown"
pass "a differing value exits 2 and names the call and index"

# --- missing file exits 1 -------------------------------------------------

run missing 1 "$work/does-not-exist.log" "$work/same-a.log"
[ -s "$work/missing.err" ] || fail "missing: no diagnostic on stderr"
pass "a missing file exits 1"

# --- empty logs exit 3 ----------------------------------------------------

run empty 3 "$work/empty-a.log" "$work/empty-b.log"
grep -q 'no abi_trace lines' "$work/empty.err" ||
    fail "empty: no explanation on stderr"
pass "empty logs exit 3"

# --- a usage error is 1, not the differ code ------------------------------

if python3 "$tool" "$work/same-a.log" >"$work/usage.out" 2>"$work/usage.err"; then
    code=0
else
    code=$?
fi
[ "$code" -eq 1 ] || fail "usage: exit $code, wanted 1"
pass "a usage error exits 1"

# --- ANSI, interleaving, repeats and the value cap ------------------------

run noisy 0 "$work/noisy.log" "$work/noisy.log"
grep -q 'same CheckFeatureSupport' "$work/noisy.out" ||
    fail "noisy: ANSI-wrapped trace not parsed"
grep -q 'same GetGPUVirtualAddress native=1 reference=1' "$work/noisy.out" ||
    fail "noisy: second trace on one line not parsed"
grep -q 'same CreateDescriptorHeap native=2 reference=2' "$work/noisy.out" ||
    fail "noisy: repeated calls not kept"
grep -q 'same Wide' "$work/noisy.out" ||
    fail "noisy: a row longer than the cap is dropped"
pass "ANSI, interleaved lines, repeated calls and the eight-value cap parse"

# --- no fixture path leaks into stdout -----------------------------------

for out in identical differ missing empty usage noisy; do
    if grep -Fq "$work" "$work/$out.out"; then
        fail "$out: a fixture path leaked into stdout"
    fi
done
pass "no fixture path appears on stdout"

# --- JSON keeps the same contract ----------------------------------------

run json-differ 2 "$work/same-a.log" "$work/differ.log" --json
python3 - "$work/json-differ.out" <<'PY' || fail "json: the report does not name the delta"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)
row = next(row for row in report["rows"] if row["call"] == "CreateDescriptorHeap")
difference = row["first_difference"]
assert difference["value"] == 3, difference
assert difference["native"] == 4096, difference
assert difference["reference"] == 8192, difference
PY
pass "the JSON report names the differing value"

printf 'all abi-diff tests passed\n'
