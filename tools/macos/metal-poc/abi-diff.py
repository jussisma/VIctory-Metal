#!/usr/bin/env python3
"""ADR-036: diff the guest-ABI trace of two runs of the same title.

Both runs are the same title on the same machine; one is
`VICTORY_BACKEND=metal` (the native path) and one is the D3D12 host reference
(data collected by `crates/victory-gpu/src/native_metal/abi_trace.rs` when
`VICTORY_GUEST_ABI_TRACE=1`). For every export the title called, the tool
reports the value rows each backend returned, so the first differing row names
the guest-visible delta that decides a title predicate.

Counts and integers only: the input is `abi_trace call=<name> values=[...]`
lines plus optional ANSI colour codes, and the output never contains guest
text, paths or asset bytes.

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

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRACE = re.compile(r"abi_trace call=(\S+) values=\[([0-9, ]*)\]")


def read_trace(path: str) -> dict[str, list[list[int]]]:
    rows: dict[str, list[list[int]]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRACE.search(ANSI.sub("", line))
            if not match:
                continue
            name = match.group(1)
            body = match.group(2).strip()
            values = [int(part) for part in body.split(",") if part.strip()]
            rows.setdefault(name, []).append(values)
    return rows


def compare(native: dict[str, list[list[int]]], reference: dict[str, list[list[int]]]):
    names = sorted(set(native) | set(reference))
    report = []
    for name in names:
        left = native.get(name, [])
        right = reference.get(name, [])
        first = None
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else None
            b = right[index] if index < len(right) else None
            if a != b:
                first = {"index": index, "native": a, "reference": b}
                break
        report.append(
            {
                "call": name,
                "native_rows": len(left),
                "reference_rows": len(right),
                "agree": first is None,
                "first_difference": first,
            }
        )
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("native", help="guest log from the VICTORY_BACKEND=metal run")
    parser.add_argument("reference", help="guest log from the D3D12 host reference run")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--text", action="store_true", help="one row per call")
    args = parser.parse_args(argv)

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
                diff = row["first_difference"]
                print(
                    f"     first at #{diff['index']}: native={diff['native']} "
                    f"reference={diff['reference']}"
                )
    return 0 if not differing else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
