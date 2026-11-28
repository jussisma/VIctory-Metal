#!/usr/bin/env python3
"""Cycle 15 boot digest: one run directory -> one comparison JSON object.

This is the focused sibling of ``boot-census.py``.  It reuses that tool's
lossless-ish readers and sanitizers, then projects only the counters the
Cycle 15 devlog table needs:

    backend, status, frames, submissions/refused_submissions,
    d3dmetal fallback, host pipeline translations and translated stages,
    exact encode counters, unknown-handle refusal class,
    frame-probe fields, bounded draw/RT trace counts.

``--baseline DIR`` may be repeated (typically once or twice) to compare the
current run with earlier boots.  Every comparable field carries its baseline
value, current value, numeric delta and ``changed`` flag.

The digest is path-free by construction.  It never opens an asset, a title, a
GPU or a socket, and it never echoes a log line.  Free text is reduced to a
fixed whitelist of one-word classes, and unknown text becomes ``other``.  A
missing or malformed input becomes ``null`` (or a malformed-line count), not a
traceback.

Exit status:

    0   digest produced (missing keys and malformed lines are data, not errors)
    2   usage error
    1   unexpected internal failure

Stdlib only; Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Sequence

# --- Shared parsing helpers ------------------------------------------------

_CENSUS_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "boot-census.py"
)


def _load_census() -> Any:
    """Load the sibling census module without assuming a package or sys.path."""
    spec = importlib.util.spec_from_file_location("_victory_boot_census", _CENSUS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("cannot load boot-census.py")
    module = importlib.util.module_from_spec(spec)
    # Keep the tools tree clean: a digest invocation must not leave a .pyc.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


census = _load_census()

# --- Bounded vocabularies --------------------------------------------------

BACKENDS = {"metal", "d3d12"}
STATUSES = {"PASS", "FAIL", "PARTIAL", "SKIP", "ERROR", "UNKNOWN"}

# `UNKNOWN_SITES` from `crates/victory-metal/src/macos/device.rs` (ADR-052).
UNKNOWN_SITE_NAMES = {
    "unknown",
    "buffer",
    "texture",
    "pipeline:compute",
    "pipeline:mesh",
    "pipeline:graphics",
    "sampler",
    "surface",
}

# `DrawRecord::kind` from `crates/victory-metal/src/macos/trace.rs`.
DRAW_KINDS = (
    "draw",
    "draw_indexed",
    "draw_indirect",
    "dispatch",
    "dispatch_indirect",
    "mesh_dispatch",
)
RT_CLASSES = ("swapchain", "offscreen", "none")
RT_FAILURE_CLASSES = (
    "texture-gone",
    "texture-unsupported",
    "readback",
    "other",
)
DRAW_KIND_CLASSES = DRAW_KINDS + ("other",)
RT_TARGET_CLASSES = RT_CLASSES + ("other",)

# `probe_color_name` vocabulary from `crates/victory-metal/src/macos/present.rs`.
UNIFORM_COLOR_RE = re.compile(
    r"^(?:black|white|red|green|blue|mixed|no_samples|transparent_black"
    r"|transparent_[0-9a-f]{6}|grey_[0-9a-f]{6}|swatch_[0-9a-f]{6})$"
)

DRAW_TRACE_RE = re.compile(r"\bdraw_trace\b")
DRAW_TRACE_RT_RE = re.compile(r"\bdraw_trace_rt\b")

FRAME_PROBE_KEYS = (
    "sampled_frames",
    "probe_pixels",
    "last_sampled_frame",
    "non_black_pixels",
    "unique_luma_buckets",
    "nonzero_bytes",
    "frame_crc",
    "sample_cost_us",
    "dropped_samples",
    "probe_every",
)

# Every numeric field that carries a delta in `comparison.baselines[*]`.
NUMERIC_FIELDS = (
    "frames",
    "submissions",
    "refused_submissions",
    "d3dmetal_fallback_count",
    "guest_d3dmetal_fallback_count",
    "host_pipeline_translations_ok",
    "host_pipeline_translations_refused",
    "host_translated_stages",
    "host_translated_stage_vertex",
    "host_translated_stage_pixel",
    "host_translated_stage_mesh",
    "host_translated_stage_object",
    "host_translated_stage_compute",
    "draws_encoded",
    "draw_indexed_encoded",
    "dispatches_encoded",
    "mesh_dispatches_encoded",
    "indirect_encoded",
    "render_passes_encoded",
    "clear_ops_encoded",
    "abi_refusals",
    "memory_space_bound",
    "memory_space_bytes",
    "memory_space_updates",
    "memory_space_pages",
    "frame_probe.sampled_frames",
    "frame_probe.probe_pixels",
    "frame_probe.last_sampled_frame",
    "frame_probe.non_black_pixels",
    "frame_probe.unique_luma_buckets",
    "frame_probe.nonzero_bytes",
    "frame_probe.sample_cost_us",
    "frame_probe.dropped_samples",
    "frame_probe.probe_every",
    "draw_trace.total",
    "draw_trace.malformed_lines",
    "draw_trace.rt_total",
    "draw_trace.rt_sampled_bytes",
    "draw_trace.rt_nonzero_bytes",
    "draw_trace.rt_distinct_values_max",
    "draw_trace.rt_nonzero_lines",
    "draw_trace.rt_zero_lines",
    "timing.guest_p50_ms",
    "timing.guest_p95_ms",
    "timing.guest_fps",
    "timing.host_p50_ms",
    "timing.host_p95_ms",
    "timing.host_fps",
    "pipelines.created",
    "pipelines.refused",
) + tuple(f"draw_trace.by_kind.{kind}" for kind in DRAW_KIND_CLASSES) + tuple(
    f"draw_trace.by_rt.{name}" for name in RT_TARGET_CLASSES
) + tuple(f"draw_trace.rt_failures.{name}" for name in RT_FAILURE_CLASSES)

CATEGORICAL_FIELDS = (
    "backend",
    "status",
    "unknown_handle_site",
    "frame_probe.present",
    "frame_probe.uniform_color",
    "frame_probe.failure",
)

# Missing count keys mean "zero observations", not "unknown".
ZERO_DEFAULT_PREFIXES = (
    "draw_trace.by_kind.",
    "draw_trace.by_rt.",
    "draw_trace.rt_failures.",
)


# --- Sanitizers ------------------------------------------------------------

_NONE_TEXT = {
    "",
    "-",
    "n/a",
    "na",
    "nil",
    "none",
    "null",
    "unknown",
    "not-applicable",
    "not_applicable",
}


def _is_none_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NONE_TEXT


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {name: counter[name] for name in sorted(counter) if counter[name] > 0}


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = census.parse_int(value)
        if number is not None:
            return number
    return None


def _telemetry_int(inputs: Any, key: str) -> int | None:
    return census.parse_int(inputs.telemetry.get(key))


def _summary_int(inputs: Any, key: str) -> int | None:
    return census.parse_int(inputs.summary.get(key))


def _telemetry_number(inputs: Any, key: str) -> float | None:
    return census.parse_float(inputs.telemetry.get(key))


def _summary_number(inputs: Any, key: str) -> float | None:
    return census.parse_float(inputs.summary.get(key))


def _backend(inputs: Any) -> str | None:
    value = inputs.summary.get("backend")
    if value in BACKENDS:
        return value
    # Only the native host writes a valid telemetry.json; it never runs for
    # d3d12.  An invalid or absent file proves nothing.
    if inputs.telemetry_valid:
        return "metal"
    return None


def _status(inputs: Any) -> str | None:
    value = inputs.summary.get("status")
    return value if value in STATUSES else None


def _unknown_handle_site(inputs: Any) -> str | None:
    value = inputs.telemetry.get("unknown_handle_site")
    if value is None:
        return None
    if isinstance(value, str) and value in UNKNOWN_SITE_NAMES:
        return value
    return "other"


def _uniform_color(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and _is_none_text(value):
        return None
    if isinstance(value, str) and UNIFORM_COLOR_RE.match(value):
        return value
    return "other"


def _probe_failure(value: Any) -> str | None:
    if value is None:
        return None
    label = census.safe_label(value)
    if label is not None and label.startswith("probe_"):
        return label
    # `probe_not_sampled` is the synthetic never-sampled marker; it also
    # matches the `probe_` prefix.
    return "other"


def _rt_failure(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    if text in ("texture-gone", "texture-unsupported"):
        return text
    if text.startswith("readback"):
        return "readback"
    return "other"


def _draw_kind(value: Any) -> str:
    text = str(value).strip().strip("\"'") if value is not None else ""
    return text if text in DRAW_KINDS else "other"


def _rt_class(value: Any) -> str:
    text = str(value).strip().strip("\"'") if value is not None else ""
    return text if text in RT_CLASSES else "other"


# --- Input projections -----------------------------------------------------


def _frame_probe(inputs: Any) -> dict[str, Any]:
    raw = inputs.telemetry.get("frame_probe")
    present = isinstance(raw, dict)
    if not present:
        raw = {}
    probe: dict[str, Any] = {"present": present}
    for key in FRAME_PROBE_KEYS:
        probe[key] = census.parse_int(raw.get(key)) if present else None
    probe["uniform_color"] = _uniform_color(raw.get("uniform_color")) if present else None
    probe["fallback_clear"] = (
        census.parse_bool(raw.get("fallback_clear")) if present else None
    )
    probe["failure"] = _probe_failure(raw.get("failure")) if present else None
    return probe


def _draw_trace(host_lines: Sequence[str]) -> dict[str, Any]:
    """Count bounded trace lines.  Raw lines and payloads never leave here."""
    by_kind: Counter[str] = Counter()
    by_rt: Counter[str] = Counter()
    rt_failures: Counter[str] = Counter()
    total = 0
    malformed = 0
    rt_total = 0
    rt_sampled_bytes = 0
    rt_nonzero_bytes = 0
    rt_distinct_values_max = 0
    rt_nonzero_lines = 0
    rt_zero_lines = 0

    for line in host_lines:
        # `draw_trace_rt` is checked first; `\b` keeps the two apart anyway.
        if DRAW_TRACE_RT_RE.search(line):
            tokens = {
                match.group(1): match.group(2).strip().strip("\"'")
                for match in census.KV_TOKEN_RE.finditer(line)
            }
            recognised = False
            for key in ("nonzero_bytes", "sampled_bytes", "distinct_values"):
                number = census.parse_int(tokens.get(key))
                if number is None or number < 0:
                    continue
                recognised = True
                if key == "nonzero_bytes":
                    rt_nonzero_bytes += number
                    if number > 0:
                        rt_nonzero_lines += 1
                    else:
                        rt_zero_lines += 1
                elif key == "sampled_bytes":
                    rt_sampled_bytes += number
                else:
                    rt_distinct_values_max = max(rt_distinct_values_max, number)
            failure = _rt_failure(tokens.get("failure"))
            if failure is not None:
                recognised = True
                rt_failures[failure] += 1
            if recognised:
                rt_total += 1
            else:
                malformed += 1
            continue

        if DRAW_TRACE_RE.search(line):
            tokens = {
                match.group(1): match.group(2).strip().strip("\"'")
                for match in census.KV_TOKEN_RE.finditer(line)
            }
            # A trace line is defined by its ordinal.  Anything else is a
            # malformed or truncated line and is only counted as such.
            if census.parse_int(tokens.get("ordinal")) is None:
                malformed += 1
                continue
            total += 1
            by_kind[_draw_kind(tokens.get("kind"))] += 1
            rt_value = tokens.get("rt")
            if rt_value is not None:
                by_rt[_rt_class(rt_value)] += 1

    return {
        "total": total,
        "malformed_lines": malformed,
        "by_kind": _ordered_counts(by_kind),
        "by_rt": _ordered_counts(by_rt),
        "rt_total": rt_total,
        "rt_sampled_bytes": rt_sampled_bytes,
        "rt_nonzero_bytes": rt_nonzero_bytes,
        "rt_distinct_values_max": rt_distinct_values_max,
        "rt_nonzero_lines": rt_nonzero_lines,
        "rt_zero_lines": rt_zero_lines,
        "rt_failures": _ordered_counts(rt_failures),
    }


def _pipelines(inputs: Any) -> dict[str, int | None]:
    pipelines = inputs.telemetry.get("pipelines")
    created = len(pipelines) if isinstance(pipelines, dict) else None
    if created is None:
        created = _summary_int(inputs, "guest_shader_psos_registered")
    refused = _summary_int(inputs, "guest_shaders_refused")
    return {"created": created, "refused": refused}


def build_digest(out_dir: str) -> dict[str, Any]:
    inputs = census.Inputs(out_dir)

    frames = _first_int(
        _telemetry_int(inputs, "frames"),
        _summary_int(inputs, "host_frames"),
        _summary_int(inputs, "frames"),
    )
    submissions = _first_int(
        _telemetry_int(inputs, "submissions"),
        _summary_int(inputs, "guest_submits_ok"),
    )
    refused_submissions = _first_int(
        _telemetry_int(inputs, "refused_submissions"),
        _summary_int(inputs, "host_refused_submissions"),
        _summary_int(inputs, "guest_submits_refused"),
    )
    stages = {
        "vertex": _telemetry_int(inputs, "host_translated_stage_vertex"),
        "pixel": _telemetry_int(inputs, "host_translated_stage_pixel"),
        "mesh": _telemetry_int(inputs, "host_translated_stage_mesh"),
        "object": _telemetry_int(inputs, "host_translated_stage_object"),
        "compute": _telemetry_int(inputs, "host_translated_stage_compute"),
    }
    translated_total = _telemetry_int(inputs, "host_translated_stages")
    if translated_total is None:
        present = [value for value in stages.values() if value is not None]
        translated_total = sum(present) if present else None

    return {
        "tool": "boot-digest",
        "schema": 1,
        "backend": _backend(inputs),
        "status": _status(inputs),
        "frames": frames,
        "submissions": submissions,
        "refused_submissions": refused_submissions,
        "d3dmetal_fallback_count": _first_int(
            _telemetry_int(inputs, "d3dmetal_fallback_count"),
            _summary_int(inputs, "d3dmetal_fallback_count"),
        ),
        "guest_d3dmetal_fallback_count": _summary_int(
            inputs, "guest_d3dmetal_fallback_count"
        ),
        "host_pipeline_translations_ok": _telemetry_int(
            inputs, "host_pipeline_translations_ok"
        ),
        "host_pipeline_translations_refused": _telemetry_int(
            inputs, "host_pipeline_translations_refused"
        ),
        "host_translated_stages": translated_total,
        "host_translated_stage_vertex": stages["vertex"],
        "host_translated_stage_pixel": stages["pixel"],
        "host_translated_stage_mesh": stages["mesh"],
        "host_translated_stage_object": stages["object"],
        "host_translated_stage_compute": stages["compute"],
        "draws_encoded": _telemetry_int(inputs, "draws_encoded"),
        "draw_indexed_encoded": _telemetry_int(inputs, "draw_indexed_encoded"),
        "dispatches_encoded": _telemetry_int(inputs, "dispatches_encoded"),
        "mesh_dispatches_encoded": _telemetry_int(inputs, "mesh_dispatches_encoded"),
        "indirect_encoded": _telemetry_int(inputs, "indirect_encoded"),
        "render_passes_encoded": _telemetry_int(inputs, "render_passes_encoded"),
        "clear_ops_encoded": _telemetry_int(inputs, "clear_ops_encoded"),
        "abi_refusals": _telemetry_int(inputs, "abi_refusals"),
        "memory_space": {
            "bound": _telemetry_int(inputs, "memory_space_bound"),
            "bytes": _telemetry_int(inputs, "memory_space_bytes"),
            "updates": _telemetry_int(inputs, "memory_space_updates"),
            "pages": _telemetry_int(inputs, "memory_space_pages"),
        },
        "unknown_handle_site": _unknown_handle_site(inputs),
        "frame_probe": _frame_probe(inputs),
        "draw_trace": _draw_trace(inputs.host_lines),
        "timing": {
            "guest_p50_ms": _summary_number(inputs, "p50_ms"),
            "guest_p95_ms": _summary_number(inputs, "p95_ms"),
            "guest_fps": _summary_number(inputs, "fps"),
            "host_p50_ms": _first_number(
                _telemetry_number(inputs, "p50_ms"),
                _summary_number(inputs, "host_p50_ms"),
            ),
            "host_p95_ms": _first_number(
                _telemetry_number(inputs, "p95_ms"),
                _summary_number(inputs, "host_p95_ms"),
            ),
            "host_fps": _first_number(
                _telemetry_number(inputs, "fps"),
                _summary_number(inputs, "host_fps"),
            ),
        },
        "pipelines": _pipelines(inputs),
        "inputs": {
            "summary_present": inputs.summary_present,
            "telemetry_present": inputs.telemetry_present,
            "telemetry_valid": inputs.telemetry_valid,
            "host_log_present": inputs.host_present,
        },
        "comparison": {"baselines": []},
    }


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = census.parse_float(value)
        if number is not None:
            return number
    return None


# --- Comparison ------------------------------------------------------------


def _lookup(document: Any, path: str) -> Any:
    """Resolve a dotted path; missing bounded histogram keys count as zero."""
    parts = path.split(".")
    zero_default = path.startswith(ZERO_DEFAULT_PREFIXES)
    value = document
    for index, part in enumerate(parts):
        if not isinstance(value, dict):
            return None
        if part not in value:
            return 0 if zero_default and index == len(parts) - 1 else None
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _assign(tree: dict[str, Any], path: str, value: Any) -> None:
    """Place `value` at a dotted path, creating intermediate objects."""
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def compare_digests(
    current: dict[str, Any], baseline: dict[str, Any], index: int
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    delta: dict[str, Any] = {}
    for path in NUMERIC_FIELDS + CATEGORICAL_FIELDS:
        baseline_value = _lookup(baseline, path)
        current_value = _lookup(current, path)
        difference: int | float | None = None
        if _is_number(baseline_value) and _is_number(current_value):
            difference = current_value - baseline_value
            if isinstance(difference, float):
                difference = round(difference, 6)
        _assign(
            fields,
            path,
            {
                "baseline": baseline_value,
                "current": current_value,
                "delta": difference,
                "changed": baseline_value != current_value,
            },
        )
        if path in NUMERIC_FIELDS:
            _assign(delta, path, difference)
    return {"index": index, "fields": fields, "delta": delta}


def add_comparison(digest: dict[str, Any], baseline_dirs: Sequence[str]) -> None:
    baselines = []
    for index, directory in enumerate(baseline_dirs, start=1):
        baseline = build_digest(directory)
        baselines.append(compare_digests(digest, baseline, index))
    digest["comparison"] = {"baselines": baselines}


# --- CLI -------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="boot-digest.py",
        description=(
            "Emit one path-free Cycle 15 digest and per-field deltas for a "
            "run-title-metal.sh --out directory.  No title, asset, GPU or log "
            "line is ever printed; free text becomes a bounded class."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="current launcher --out directory (telemetry.json, summary.txt, host.log)",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="DIR",
        help="earlier --out directory; repeat once or twice for the Boot table",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        digest = build_digest(args.out)
        add_comparison(digest, args.baseline)
    except Exception as error:  # pragma: no cover - defensive last resort
        print(f"boot-digest.py: internal error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(digest, indent=2, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
