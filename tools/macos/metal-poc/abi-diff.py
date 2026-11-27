#!/usr/bin/env python3
"""ADR-036: diff the guest-ABI trace of two runs of the same title.

Both runs are the same title on the same machine; one is
`VICTORY_BACKEND=metal` (the native path) and one is the D3D12 host reference
(data collected by `crates/victory-gpu/src/native_metal/abi_trace.rs` when
`VICTORY_GUEST_ABI_TRACE=1`). For every export the title called, the tool
reports the value rows each backend returned, so the first differing row names
the guest-visible delta that decides a title predicate.

Counts and integers only: the input is `abi_trace call=<name> values=[...]`
lines plus optional ANSI colour codes and surrounding log text, and the output
never contains guest text, paths or asset bytes.

The trace is bounded by construction (at most eight values per line and at
most 64 lines per call name), so the parser truncates a row to the same eight
values and compares repeated calls by their order (row index) and each row's
values by their index. The first difference therefore names both.

Usage:
    abi-diff.py NATIVE_LOG REFERENCE_LOG [--json] [--text]

Exit codes: 0 when the traces agree, 2 when they differ, 1 on a usage or read
error, 3 when neither log contains a trace.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# SGR colour codes, which a log collector may leave in the line.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# One trace line. A line may carry more than one trace when writers interleave
# at chunk level, so every match on a line is taken, not just the first.
TRACE = re.compile(r"abi_trace\s+call=(\S+)\s+values=\[([^\]]*)\]")

# The trace format keeps at most this many values per row
# (`abi_trace::MAX_VALUES`).
MAX_VALUES = 8


def parse_values(body: str) -> list[int]:
    """The numeric values in one `values=[...]` body, capped like the trace."""
    values: list[int] = []
    for part in re.split(r"[,\s]+", body.strip()):
        if not part:
            continue
        try:
            values.append(int(part, 0))
        except ValueError:
            # Not a number: a malformed row contributes what it can, so one
            # noisy line never hides the rest of the trace.
            continue
    return values[:MAX_VALUES]


def read_trace(path: str) -> dict[str, list[list[int]]]:
    """Every trace row in `path`, by call name, in the order the title made them."""
    rows: dict[str, list[list[int]]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # `replace` covers CRLF logs and `search`-style scans below cover
            # interleaved prefixes; only the trace substring is read.
            line = ANSI.sub("", line).replace("\r", "")
            for match in TRACE.finditer(line):
                rows.setdefault(match.group(1), []).append(parse_values(match.group(2)))
    return rows


def first_difference(
    left: list[list[int]], right: list[list[int]]
) -> dict[str, object] | None:
    """The first differing (row, value) between two call-name traces."""
    for row_index in range(max(len(left), len(right))):
        a = left[row_index] if row_index < len(left) else None
        b = right[row_index] if row_index < len(right) else None
        if a == b:
            continue
        if a is None or b is None:
            # One run made fewer calls: the missing row itself is the delta.
            return {"row": row_index, "value": None, "native": a, "reference": b}
        for value_index in range(max(len(a), len(b))):
            av = a[value_index] if value_index < len(a) else None
            bv = b[value_index] if value_index < len(b) else None
            if av != bv:
                return {
                    "row": row_index,
                    "value": value_index,
                    "native": av,
                    "reference": bv,
                }
    return None


def compare(
    native: dict[str, list[list[int]]], reference: dict[str, list[list[int]]]
) -> list[dict[str, object]]:
    report = []
    for name in sorted(set(native) | set(reference)):
        left = native.get(name, [])
        right = reference.get(name, [])
        difference = first_difference(left, right)
        report.append(
            {
                "call": name,
                "native_rows": len(left),
                "reference_rows": len(right),
                "agree": difference is None,
                "first_difference": difference,
            }
        )
    return report


def describe(difference: dict[str, object]) -> str:
    """One line naming the call row and value index that differ."""
    where = f"row #{difference['row']}"
    if difference["value"] is not None:
        where += f" value #{difference['value']}"
    return (
        f"     first at {where}: native={difference['native']} "
        f"reference={difference['reference']}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("native", help="guest log from the VICTORY_BACKEND=metal run")
    parser.add_argument("reference", help="guest log from the D3D12 host reference run")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--text", action="store_true", help="one row per call (default)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_code:
        # argparse already printed the usage; 0 is `--help`, anything else is
        # a usage error (the tool reserves 2 for a differing trace).
        return 0 if exit_code.code == 0 else 1

    try:
        native = read_trace(args.native)
        reference = read_trace(args.reference)
    except OSError as error:
        print(f"abi-diff: {error}", file=sys.stderr)
        return 1
    if not native and not reference:
        print("abi-diff: no abi_trace lines in either log", file=sys.stderr)
        return 3

    report = compare(native, reference)
    differing = [row for row in report if not row["agree"]]

    if args.json:
        print(
            json.dumps(
                {
                    "calls": len(report),
                    "differing": len(differing),
                    "rows": report,
                },
                indent=1,
            )
        )
    else:
        for row in report:
            state = "same" if row["agree"] else "DIFF"
            print(
                f"{state:4} {row['call']} native={row['native_rows']} "
                f"reference={row['reference_rows']}"
            )
            if row["first_difference"]:
                print(describe(row["first_difference"]))
    return 0 if not differing else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
