#!/usr/bin/env python3
"""Hermetic census of one `run-title-metal.sh --out` directory.

The launcher writes four files:

    summary.txt     canonical key=value lines from the launcher
    telemetry.json  the host daemon's JSON telemetry
    guest.log       the Wine guest's captured output
    host.log        the native host daemon's captured output

This tool turns those files into one deterministic, path-free digest.  It is a
counter and classifier, not a loader: it never opens a title, Wine, a GPU or a
network socket.  Guest memory, asset bytes, file paths and title names are not
part of the output.  Text it does keep (the first `METAL_HOST_ERROR` reason and
refusal labels) is bounded and safe-labelled; path-shaped tokens are redacted.

Exit status:

    0   digest produced (also when `--gate` sees no invariant violation)
    2   usage error, or `--gate` saw a fallback / host D3D12 creation attempt /
        orphan survival
    1   unexpected internal failure

Stdlib only; Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Iterable, Sequence

# --- Constants -------------------------------------------------------------

ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
KV_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z][A-Za-z0-9_.-]*)=([^\s,;]+)")
SUMMARY_LINE_RE = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_.-]*)=(.*)$")
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MILESTONE_LINE_RE = re.compile(
    r"\bMILESTONE(?:_REACHED)?[=: ]+([A-Za-z0-9_]+)", re.IGNORECASE
)
OPCODE_LINE_RE = re.compile(
    r"(?:^|\s)(unsupported_opcode|refused_opcode|opcode)=([^\s,;]+)"
)
OPCODE_LIST_RE = re.compile(
    r"(?:^|[\s,.])(?:unsupported_opcodes|opcodes)=\[([^\]]*)\]"
)
REFUSAL_TOKEN_RE = re.compile(
    r"(?:^|\s)(refusal|refused_category|refusal_category|category_refused)=([^\s,;]+)"
)
NAMED_REFUSALS_RE = re.compile(r"named_refusals=\[([^\]]*)\]")
REFUSALS_LIST_RE = re.compile(r"(?:^|[\s,.])refusals=\[([^\]]*)\]")
NATIVE_FRAME_COUNTS_RE = re.compile(
    r"registered=(\d+)\s+buffers/(\d+)\s+psos"
)

_NONE_VALUES = {
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

# Frozen wire v3 opcodes (`crates/victory-metal/src/wire_v3.rs`).  Numeric
# histogram keys are translated to these names; unknown numbers stay
# `opcode_<n>` so a new opcode is still countable without pretending to know it.
OPCODE_NAMES: dict[int, str] = {
    1: "dispatch_1d",
    2: "dispatch_3d",
    3: "draw_indexed",
    4: "draw",
    5: "dispatch_mesh",
    6: "resource_barrier",
    7: "set_pipeline",
    8: "root_constants",
    9: "descriptor_table",
    16: "begin_render_pass",
    17: "end_render_pass",
    18: "set_viewport",
    19: "set_scissor",
    20: "set_graphics_pipeline",
    21: "set_mesh_pipeline",
    22: "set_compute_pipeline",
    23: "set_primitive_topology",
    24: "set_vertex_buffers",
    25: "set_index_buffer",
    26: "set_root_buffer_view",
    27: "draw_instanced",
    28: "draw_indexed_instanced",
    29: "copy_buffer",
    30: "copy_texture",
    31: "present",
    40: "execute_indirect_draw",
    41: "execute_indirect_draw_indexed",
    42: "execute_indirect_dispatch_mesh",
    43: "execute_indirect_dispatch",
    44: "copy_texture_region",
    45: "resolve_subresource",
    46: "set_blend_state",
    47: "set_depth_stencil_state",
    48: "set_rasterizer_state",
    49: "set_stencil_ref",
    50: "set_blend_factor",
    51: "clear_render_target_view",
    52: "clear_depth_stencil_view",
    53: "copy_buffer_to_texture",
}
NO_OPCODE = 65535

MILESTONE_ORDER: tuple[str, ...] = (
    "host_ready",
    "d3d12x_device",
    "completion_page",
    "dstorage_queue",
    "first_native_submit",
    "first_frame_presented",
    "first_3d_draw",
    "first_compute_dispatch",
    "sustained_frames",
)
_MILESTONE_SET = set(MILESTONE_ORDER)

# A run is "sustained" once it presented at least one second at the 30 FPS game
# gate.  An explicit `sustained_frames` marker always wins.
SUSTAINED_FRAMES = 30

SIZE_SAMPLE_KEYS = {
    "refused_size",
    "refused_size_bytes",
    "refused_buffer_size",
    "refused_buffer_size_bytes",
    "refused_texture_size",
    "refused_texture_size_bytes",
    "texture_size_refused",
    "buffer_size_refused",
}
FORMAT_KEYS = {
    "refused_format",
    "format_refused",
    "refused_formats",
    "formats_refused",
    "refused_dxgi_format",
    "dxgi_format_refused",
    "dxgi_formats_refused",
}
REFUSAL_COUNT_KEYS = {
    "refusals",
    "refusal_histogram",
    "top_refusals",
    "named_refusals",
}
SHADER_REFUSAL_KEYS = {
    "refused_pipeline",
    "pipeline_refused",
    "refused_pipelines",
    "pipeline_refusals",
    "refused_shader",
    "shader_refused",
    "refused_shaders",
    "shader_refusals",
    "top_shader_refusals",
}
PATH_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Windows drive path, UNC path, then POSIX-style path.  Spaces are consumed
    # because proprietary paths may contain them; an over-redacted reason is
    # safer than a leaked one.  A redaction emits `<path>`, which contains no
    # slash, so the patterns cannot re-match.
    (re.compile(r"[A-Za-z]:\\[^\n\"'<>|,;]*"), "<path>"),
    (re.compile(r"\\\\[^\n\"'<>|,;]*"), "<path>"),
    (re.compile(r"(?<![A-Za-z0-9_])/[^\n\"'<>|,;]*"), "<path>"),
)


# --- Text helpers ----------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove ANSI colour/OSC sequences and normalise line endings."""
    text = ANSI_CSI_RE.sub("", text)
    text = ANSI_OSC_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: str) -> tuple[bool, str]:
    """Read a file lossily; absence and unreadability are both `(False, "")`."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return True, strip_ansi(handle.read())
    except OSError:
        return False, ""


def read_json(path: str) -> tuple[bool, bool, dict[str, Any]]:
    """Return (present, valid, object) for a telemetry JSON file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
    except OSError:
        return False, False, {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return True, False, {}
    if not isinstance(value, dict):
        return True, False, {}
    return True, True, value


def _is_none(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NONE_VALUES


def parse_int(value: Any) -> int | None:
    """Decimal integer or `None`; `n/a`, floats with a fraction, junk -> None."""
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip().strip("\"'")
    if _is_none(text):
        return None
    try:
        return int(text, 10)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else None


def parse_float(value: Any) -> float | None:
    """Finite float or `None`."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip("\"'")
    if _is_none(text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def parse_bool(value: Any) -> bool | None:
    """Boolean from a summary/telemetry flag; unknown -> None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    text = str(value).strip().strip("\"'").lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def safe_label(value: Any) -> str | None:
    """A bounded, path-free label suitable for a histogram key."""
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    if _is_none(text) or not SAFE_LABEL_RE.match(text):
        return None
    return text


def opcode_label(value: Any) -> str | None:
    """Translate a numeric opcode to its frozen name, or keep a safe name."""
    number = parse_int(value)
    if number is not None:
        if number == NO_OPCODE:
            return "unnamed"
        if number < 0:
            return None
        return OPCODE_NAMES.get(number, f"opcode_{number}")
    label = safe_label(value)
    if label is None:
        return None
    if label.lower() == "none":
        return "unnamed"
    return label


def classify_refusal(text: str) -> str:
    """Map a host refusal message to a small, path-free category."""
    lowered = text.lower()
    for needle, category in (
        ("format", "format"),
        ("texture", "texture"),
        ("buffer", "buffer"),
        ("pipeline", "pipeline"),
        ("shader", "pipeline"),
        ("descriptor", "descriptor"),
        ("binding", "binding"),
        ("state", "state"),
        ("opcode", "opcode"),
        ("unsupported", "unsupported"),
    ):
        if needle in lowered:
            return category
    return "other"


def sanitise_reason(text: str) -> str:
    """One bounded line, no ANSI/control bytes and no path-shaped tokens."""
    text = strip_ansi(text)
    text = " ".join(text.split())
    for pattern, replacement in PATH_REDACTIONS:
        text = pattern.sub(replacement, text)
    text = "".join(character for character in text if character >= " ")
    return text[:240]


# --- Input parsing ---------------------------------------------------------


def parse_summary(text: str) -> dict[str, str]:
    """Parse canonical `key=value` summary lines; the last value wins.

    A tracing prefix is tolerated.  Because a summary value (notably `reason`)
    may itself contain `=`, the whole remainder of the line is the value.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = SUMMARY_LINE_RE.search(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def parse_tokens(lines: Iterable[str]) -> dict[str, list[str]]:
    """Collect every `key=value` token in text, preserving repetition."""
    values: dict[str, list[str]] = {}
    for line in lines:
        for match in KV_TOKEN_RE.finditer(line):
            values.setdefault(match.group(1), []).append(
                match.group(2).strip().strip("\"'")
            )
    return values


def split_values(value: str) -> list[str]:
    """Split a CSV/JSON-array-ish value into tokens."""
    text = value.strip().strip("[]")
    return [token for token in re.split(r"[,;\s]+", text) if token]


def parse_json_object(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_named_counts(
    value: Any,
    *,
    label: Any = safe_label,
) -> tuple[Counter[str], bool]:
    """Parse `name:count`, `name=count`, `name`, a JSON object or `none`.

    Returns `(counts, seen)`.  `seen` distinguishes an explicit empty histogram
    (`none`) from a missing input.
    """
    counts: Counter[str] = Counter()
    if value is None:
        return counts, False
    if isinstance(value, dict):
        for key, count in value.items():
            number = parse_int(count)
            name = label(key)
            if number is None or number <= 0 or name is None:
                continue
            counts[name] += number
        return counts, True
    text = str(value).strip()
    if _is_none(text):
        return counts, True
    parsed = parse_json_object(text)
    if parsed is not None:
        return parse_named_counts(parsed, label=label)
    for token in split_values(text):
        token = token.strip("[]")
        if not token:
            continue
        match = re.match(r"^(.+?)(?::|=)(\d+)$", token)
        if match:
            number = int(match.group(2))
            name = label(match.group(1))
            if name is not None and number > 0:
                counts[name] += number
            continue
        name = label(token)
        if name is not None:
            counts[name] += 1
    return counts, True


def parse_sizes(value: Any) -> list[int]:
    """Parse a size list.  `size:count` is expanded (bounded)."""
    sizes: list[int] = []
    if value is None:
        return sizes
    text = str(value).strip()
    if _is_none(text):
        return sizes
    for token in split_values(text):
        match = re.match(r"^(\d+):(\d+)$", token)
        if match:
            size, count = int(match.group(1)), int(match.group(2))
            sizes.extend([size] * min(count, 100_000))
            continue
        number = parse_int(token)
        if number is not None and number >= 0:
            sizes.append(number)
    return sizes


def parse_format_counts(value: Any, counts: Counter[int]) -> bool:
    """Parse numeric DXGI format refusals only; names are deliberately ignored."""
    if value is None:
        return False
    if isinstance(value, dict):
        for key, count in value.items():
            number = parse_int(key)
            amount = parse_int(count)
            if (
                number is not None
                and amount is not None
                and number >= 0
                and amount > 0
            ):
                counts[number] += amount
        return True
    text = str(value).strip()
    if _is_none(text):
        return True
    parsed = parse_json_object(text)
    if parsed is not None:
        return parse_format_counts(parsed, counts)
    for token in split_values(text):
        match = re.match(r"^(\d+)(?::|=)(\d+)$", token)
        if match:
            counts[int(match.group(1))] += int(match.group(2))
            continue
        number = parse_int(token)
        if number is not None and number >= 0:
            counts[number] += 1
    return True


# --- Digest construction ---------------------------------------------------


class Inputs:
    """All four launcher artifacts, parsed and normalised."""

    def __init__(self, out_dir: str) -> None:
        summary_path = os.path.join(out_dir, "summary.txt")
        telemetry_path = os.path.join(out_dir, "telemetry.json")
        guest_path = os.path.join(out_dir, "guest.log")
        host_path = os.path.join(out_dir, "host.log")

        self.summary_present, summary_text = read_text(summary_path)
        self.guest_present, guest_text = read_text(guest_path)
        self.host_present, host_text = read_text(host_path)
        self.telemetry_present, self.telemetry_valid, self.telemetry = read_json(
            telemetry_path
        )

        self.summary = parse_summary(summary_text)
        self.guest_lines = guest_text.splitlines()
        self.host_lines = host_text.splitlines()
        self.guest_tokens = parse_tokens(self.guest_lines)
        self.host_tokens = parse_tokens(self.host_lines)

    def _source(self, source: str, key: str) -> Any:
        if source == "summary":
            return self.summary.get(key)
        if source == "telemetry":
            return self.telemetry.get(key)
        if source == "guest":
            values = self.guest_tokens.get(key)
            return values[-1] if values else None
        if source == "host":
            values = self.host_tokens.get(key)
            return values[-1] if values else None
        raise AssertionError(source)

    def raw(
        self,
        keys: Sequence[str],
        sources: Sequence[str] = ("summary", "telemetry", "guest", "host"),
    ) -> Any:
        for source in sources:
            for key in keys:
                value = self._source(source, key)
                # A launcher writes `n/a` when it could not produce a field
                # (for example `host_p50_ms` without telemetry).  That is a
                # missing value, not a source precedence stop: let a later
                # telemetry/log source answer instead.
                if value is not None and not _is_none(str(value)):
                    return value
        return None

    def integer(
        self,
        keys: Sequence[str],
        sources: Sequence[str] = ("summary", "telemetry", "guest", "host"),
    ) -> int | None:
        return parse_int(self.raw(keys, sources))

    def number(
        self,
        keys: Sequence[str],
        sources: Sequence[str] = ("summary", "telemetry", "guest", "host"),
    ) -> float | None:
        return parse_float(self.raw(keys, sources))

    def flag(
        self,
        keys: Sequence[str],
        sources: Sequence[str] = ("summary", "telemetry", "guest", "host"),
    ) -> bool | None:
        return parse_bool(self.raw(keys, sources))


def _named_list(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    """Top `limit` positive names by count desc, then name asc."""
    ordered = sorted(
        ((name, count) for name, count in counter.items() if count > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return [{"name": name, "count": count} for name, count in ordered[:limit]]


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {
        name: counter[name]
        for name in sorted(counter)
        if counter[name] > 0
    }


def collect_run(inputs: Inputs) -> dict[str, Any]:
    # Only the launcher's own vocabulary is echoed; arbitrary summary text
    # could otherwise carry a title or a path into the digest.
    raw_mode = inputs.summary.get("mode")
    mode = (
        raw_mode
        if raw_mode
        in {"title", "gate", "window", "headless", "bench", "frame", "diagnostic"}
        else None
    )
    raw_status = inputs.summary.get("status")
    status = (
        raw_status
        if raw_status in {"PASS", "FAIL", "PARTIAL", "SKIP", "ERROR", "UNKNOWN"}
        else None
    )
    exit_code = inputs.integer(["exit_code"], ("summary",))
    guest_exit = inputs.integer(["guest_exit"], ("summary",))
    host_exit = inputs.integer(["host_exit"], ("summary",))
    terminated = inputs.flag(
        ["host_terminated_by_launcher"], ("summary",)
    )

    reason = inputs.summary.get("reason", "").lower()
    timeout_by_reason = bool(
        re.search(r"\b(timed out|exceeded the launcher timeout)\b", reason)
    )
    guest_timed_out = inputs.flag(
        ["guest_timed_out"], ("summary",)
    )
    if guest_timed_out is None and timeout_by_reason:
        guest_timed_out = True
    host_timed_out = inputs.flag(["host_timed_out"], ("summary",))

    host_survived = inputs.flag(
        ["host_survived", "host_alive", "survived_host"], ("summary", "host")
    )
    victim_survived = inputs.flag(
        ["victim_survived", "victim_alive", "survived_victim"], ("summary", "host")
    )
    orphan_survived = inputs.flag(
        ["orphan_survived", "process_orphan_survived"], ("summary", "host", "guest")
    )
    marker = any(
        "PROCESS_ORPHAN_SURVIVED" in line
        for line in inputs.host_lines + inputs.guest_lines
    )
    # A survivor marker or the launcher's exit-13 contract is stronger evidence
    # than a contradictory summary flag; the gate must fail conservatively.
    if marker or exit_code == 13:
        orphan_survived = True
    elif orphan_survived is None and exit_code is not None:
        orphan_survived = False

    return {
        "mode": mode,
        "status": status,
        "exit_code": exit_code,
        "guest_exit": guest_exit,
        "host_exit": host_exit,
        "host_terminated_by_launcher": terminated,
        "guest_timed_out": guest_timed_out,
        "host_timed_out": host_timed_out,
        "host_survived": host_survived,
        "victim_survived": victim_survived,
        "orphan_survived": orphan_survived,
        "summary_present": inputs.summary_present,
        "telemetry_present": inputs.telemetry_present,
        "telemetry_valid": inputs.telemetry_valid,
        "guest_log_present": inputs.guest_present,
        "host_log_present": inputs.host_present,
    }


def collect_invariants(inputs: Inputs) -> dict[str, Any]:
    host_fallback = inputs.integer(
        ["d3dmetal_fallback_count", "host_d3dmetal_fallback_count"],
        ("summary", "telemetry", "host"),
    )
    guest_fallback = inputs.integer(
        ["guest_d3dmetal_fallback_count", "guest_d3dmetal_fallback"],
        ("summary", "guest"),
    )
    available = [value for value in (host_fallback, guest_fallback) if value is not None]
    fallback = max(available) if available else None

    module_raw = inputs.raw(["guest_d3dmetal_module"], ("summary", "guest"))
    module: int | None
    if module_raw is None or _is_none(str(module_raw)):
        module = None
    else:
        module = parse_int(module_raw)

    creation = inputs.integer(
        ["host_d3d12_creation_attempts", "d3d12_creation_attempts"],
        ("summary", "telemetry", "host", "guest"),
    )

    return {
        "d3dmetal_fallback_count": fallback,
        "host_d3dmetal_fallback_count": host_fallback,
        "guest_d3dmetal_fallback_count": guest_fallback,
        "host_d3d12_creation_attempts": creation,
        "guest_d3dmetal_module": module,
    }


def _guest_percentile(inputs: Inputs, name: str) -> float | None:
    value = inputs.number(
        [f"guest_{name}_ms"], ("summary", "guest")
    )
    if value is None:
        value = inputs.number([f"{name}_ms"], ("summary", "guest"))
    if value is None:
        micros = inputs.number(
            [f"guest_telemetry_{name}_us", f"telemetry_{name}_us"],
            ("summary", "guest"),
        )
        if micros is not None:
            value = micros / 1000.0
    return value


def _host_percentile(inputs: Inputs, name: str) -> float | None:
    value = inputs.number(
        [f"host_{name}_ms"], ("summary", "host")
    )
    if value is None:
        value = inputs.number([f"{name}_ms"], ("telemetry",))
    if value is None:
        micros = inputs.number(
            [f"host_{name}_us"], ("summary", "telemetry", "host")
        )
        if micros is not None:
            value = micros / 1000.0
    return value


def collect_frames(inputs: Inputs) -> dict[str, Any]:
    guest_samples = inputs.integer(["frames", "guest_frames"], ("summary", "guest"))
    host_samples = inputs.integer(["host_frames"], ("summary", "host"))
    if host_samples is None:
        host_samples = inputs.integer(["frames"], ("telemetry",))

    guest_presented = inputs.integer(["guest_frames_ok"], ("summary", "guest"))
    guest_presents_ok = inputs.integer(
        ["guest_presents_ok", "presents_ok"], ("summary", "guest")
    )
    if guest_presented is None:
        guest_presented = guest_presents_ok

    host_drawables = inputs.integer(
        ["host_window_drawables", "window_drawables"], ("summary", "telemetry", "host")
    )

    guest_fps = inputs.number(["guest_fps"], ("summary", "guest"))
    if guest_fps is None:
        guest_fps = inputs.number(["fps"], ("summary", "guest"))
    host_fps = inputs.number(["host_fps"], ("summary", "host"))
    if host_fps is None:
        host_fps = inputs.number(["fps"], ("telemetry",))

    submits_ok = inputs.integer(["guest_submits_ok"], ("summary", "guest"))
    if submits_ok is None:
        submits_ok = inputs.integer(["submits_ok"], ("summary", "guest"))
    if submits_ok is None:
        submits_ok = inputs.integer(["submissions"], ("telemetry",))
    submits_refused = inputs.integer(
        ["guest_submits_refused"], ("summary", "guest")
    )
    if submits_refused is None:
        submits_refused = inputs.integer(["submits_refused"], ("summary", "guest"))
    if submits_refused is None:
        submits_refused = inputs.integer(["refused_submissions"], ("telemetry",))
    if submits_refused is None:
        submits_refused = inputs.integer(["host_refused_submissions"], ("summary",))

    presents_ok = inputs.integer(
        ["guest_presents_ok", "presents_ok"], ("summary", "guest")
    )
    presents_refused = inputs.integer(
        ["guest_presents_refused", "presents_refused"], ("summary", "guest")
    )

    return {
        "presented": guest_presented,
        "guest": {
            "p50_ms": _guest_percentile(inputs, "p50"),
            "p95_ms": _guest_percentile(inputs, "p95"),
            "p99_ms": _guest_percentile(inputs, "p99"),
            "fps": guest_fps,
            "samples": guest_samples,
            "presented": guest_presented,
        },
        "host": {
            "p50_ms": _host_percentile(inputs, "p50"),
            "p95_ms": _host_percentile(inputs, "p95"),
            "p99_ms": _host_percentile(inputs, "p99"),
            "fps": host_fps,
            "samples": host_samples,
            "presented": host_drawables,
            "drawables": host_drawables,
        },
        "submissions": {"ok": submits_ok, "refused": submits_refused},
        "presents": {"ok": presents_ok, "refused": presents_refused},
    }


def _collect_size_samples(inputs: Inputs) -> list[int]:
    sizes: list[int] = []
    for key in SIZE_SAMPLE_KEYS:
        if key in inputs.summary:
            sizes.extend(parse_sizes(inputs.summary[key]))
    for tokens in (inputs.guest_tokens, inputs.host_tokens):
        for key, values in tokens.items():
            if key in SIZE_SAMPLE_KEYS:
                for value in values:
                    sizes.extend(parse_sizes(value))
    return [size for size in sizes if size >= 0]


def _size_bucket(size: int) -> str:
    if size < 4 * 1024:
        return "lt_4kib"
    if size < 1024 * 1024:
        return "lt_1mib"
    if size < 64 * 1024 * 1024:
        return "lt_64mib"
    return "ge_64mib"


def _native_frame_counts(inputs: Inputs) -> tuple[int | None, int | None]:
    """Last `registered=N buffers/M psos` from the host's native frame line."""
    buffers: int | None = None
    psos: int | None = None
    for line in inputs.host_lines:
        match = NATIVE_FRAME_COUNTS_RE.search(line)
        if match:
            buffers = int(match.group(1))
            psos = int(match.group(2))
    return buffers, psos


def collect_resources(inputs: Inputs) -> dict[str, Any]:
    native_buffers, _native_psos = _native_frame_counts(inputs)
    buffers_registered = inputs.integer(
        [
            "buffers_registered",
            "guest_buffers_registered",
            "buffer_registrations",
            "buffers_registered_total",
        ]
    )
    if buffers_registered is None:
        buffers_registered = native_buffers
    buffers_refused = inputs.integer(
        [
            "buffers_refused",
            "guest_buffers_refused",
            "buffer_registration_failures",
            "buffers_failed",
        ]
    )
    textures_registered = inputs.integer(
        [
            "textures_registered",
            "guest_textures_registered",
            "texture_registrations",
            "textures_registered_total",
        ]
    )
    textures_refused = inputs.integer(
        [
            "textures_refused",
            "guest_textures_refused",
            "texture_registration_failures",
            "textures_failed",
        ]
    )

    samples = _collect_size_samples(inputs)
    bucket_counts = {name: 0 for name in ("lt_4kib", "lt_1mib", "lt_64mib", "ge_64mib")}
    worst: int | None = None
    if samples:
        for size in samples:
            bucket_counts[_size_bucket(size)] += 1
        worst = max(samples)
    else:
        bucket_keys = {
            "lt_4kib": (
                "refused_sizes_lt_4kib",
                "refused_size_lt_4kib",
                "refused_sizes_under_4kib",
            ),
            "lt_1mib": (
                "refused_sizes_lt_1mib",
                "refused_size_lt_1mib",
                "refused_sizes_under_1mib",
            ),
            "lt_64mib": (
                "refused_sizes_lt_64mib",
                "refused_size_lt_64mib",
                "refused_sizes_under_64mib",
            ),
            "ge_64mib": (
                "refused_sizes_ge_64mib",
                "refused_size_ge_64mib",
                "refused_sizes_at_least_64mib",
            ),
        }
        any_bucket = False
        for name, keys in bucket_keys.items():
            value = inputs.integer(keys)
            if value is not None:
                bucket_counts[name] = value
                any_bucket = True
        if any_bucket:
            worst = inputs.integer(
                ["worst_refused_bytes", "refused_size_worst", "max_refused_size_bytes"]
            )

    formats: Counter[int] = Counter()
    for key in FORMAT_KEYS:
        if key in inputs.summary:
            parse_format_counts(inputs.summary[key], formats)
    for tokens in (inputs.guest_tokens, inputs.host_tokens):
        for key, values in tokens.items():
            if key in FORMAT_KEYS:
                for value in values:
                    parse_format_counts(value, formats)

    return {
        "buffers": {
            "registered": buffers_registered,
            "refused": buffers_refused,
        },
        "textures": {
            "registered": textures_registered,
            "refused": textures_refused,
        },
        "registration_failures": inputs.integer(
            ["guest_registration_failures", "registration_failures"]
        ),
        "refused_sizes": {
            "lt_4kib": bucket_counts["lt_4kib"],
            "lt_1mib": bucket_counts["lt_1mib"],
            "lt_64mib": bucket_counts["lt_64mib"],
            "ge_64mib": bucket_counts["ge_64mib"],
            "worst_bytes": worst,
            "samples": len(samples) if samples else sum(bucket_counts.values()),
        },
        "formats_refused": {
            str(number): formats[number] for number in sorted(formats)
        },
    }


def collect_shaders(inputs: Inputs) -> dict[str, Any]:
    compiled = inputs.integer(
        [
            "shaders_compiled",
            "guest_shaders_compiled",
            "psos_compiled",
            "pipelines_compiled",
            "shader_compiled",
        ]
    )
    accepted = inputs.integer(
        [
            "shaders_accepted",
            "guest_shaders_accepted",
            "psos_registered",
            "guest_shader_psos_registered",
            "pipelines_accepted",
            "pipeline_registrations",
        ]
    )
    if accepted is None:
        _native_buffers, native_psos = _native_frame_counts(inputs)
        accepted = native_psos
    refused = inputs.integer(
        [
            "shaders_refused",
            "guest_shaders_refused",
            "psos_refused",
            "pipelines_refused",
            "pipeline_registration_failures",
        ]
    )

    names: Counter[str] = Counter()
    for key in SHADER_REFUSAL_KEYS:
        if key in inputs.summary:
            counts, _ = parse_named_counts(inputs.summary[key])
            names.update(counts)
    for tokens in (inputs.guest_tokens, inputs.host_tokens):
        for key, values in tokens.items():
            if key in SHADER_REFUSAL_KEYS:
                for value in values:
                    counts, _ = parse_named_counts(value)
                    names.update(counts)
    # A `name:count` token in a shader-refusal key is rare but valid; the
    # generic parser above already handles it.  Keep unknown/path-shaped names
    # out of the output.
    cleaned: Counter[str] = Counter()
    for name, count in names.items():
        label = safe_label(name)
        cleaned[label if label is not None else "other"] += count

    top = _named_list(cleaned)
    return {
        "compiled": compiled,
        "accepted": accepted,
        "refused": refused,
        "top_refusals": top,
        "top_refusal_names": [item["name"] for item in top],
    }


def _collect_opcode_histogram(inputs: Inputs) -> Counter[str]:
    histogram: Counter[str] = Counter()

    def add_value(value: Any, count: int = 1) -> None:
        label = opcode_label(value)
        if label is not None:
            histogram[label] += count

    for key in ("unsupported_opcodes", "guest_unsupported_opcodes"):
        if key in inputs.summary:
            counts, seen = parse_named_counts(
                inputs.summary[key], label=opcode_label
            )
            if seen:
                histogram.update(counts)

    for line in inputs.guest_lines + inputs.host_lines:
        line_count = 1
        count_match = re.search(r"(?:^|\s)count=(\d+)", line)
        if count_match:
            line_count = int(count_match.group(1))
        for match in OPCODE_LINE_RE.finditer(line):
            add_value(match.group(2), line_count)
        for match in OPCODE_LIST_RE.finditer(line):
            counts, _ = parse_named_counts(match.group(1), label=opcode_label)
            histogram.update(counts)

    return histogram


def _collect_state_histograms(
    inputs: Inputs,
) -> tuple[Counter[str], Counter[str], int | None, int | None]:
    ignored: Counter[str] = Counter()
    refused: Counter[str] = Counter()

    for line in inputs.guest_lines + inputs.host_lines:
        count = 1
        count_match = re.search(r"(?:^|\s)count=(\d+)", line)
        if count_match:
            count = int(count_match.group(1))
        for match in KV_TOKEN_RE.finditer(line):
            key, value = match.group(1), match.group(2)
            if key == "ignored_state":
                label = safe_label(value)
                if label is not None:
                    ignored[label] += count
            elif key == "refused_state":
                label = safe_label(value)
                if label is not None:
                    refused[label] += count

    def total_and_maybe_histogram(
        keys: Sequence[str], histogram: Counter[str], summary_key: str
    ) -> tuple[int | None, Counter[str]]:
        aggregate = inputs.integer(keys, ("summary", "guest", "host"))
        if aggregate is not None:
            return aggregate, histogram
        # A summary may list state names (`ignored_states=blend,cull`) instead
        # of a count.  Only path-free labels are admitted.
        summary_value = inputs.summary.get(summary_key)
        if summary_value is not None and parse_int(summary_value) is None:
            counts, _ = parse_named_counts(summary_value)
            histogram.update(counts)
        if histogram:
            return sum(histogram.values()), histogram
        return None, histogram

    ignored_total, ignored = total_and_maybe_histogram(
        ["ignored_states", "states_ignored", "ignored_state_count"],
        ignored,
        "ignored_states",
    )
    refused_total, refused = total_and_maybe_histogram(
        ["refused_states", "states_refused", "refused_state_count"],
        refused,
        "refused_states",
    )
    return ignored, refused, ignored_total, refused_total


def collect_opcodes(inputs: Inputs) -> dict[str, Any]:
    histogram = _collect_opcode_histogram(inputs)
    ignored, refused, ignored_total, refused_total = _collect_state_histograms(inputs)
    return {
        "histogram": _ordered_counts(histogram),
        "ignored": ignored_total,
        "refused": refused_total,
        "ignored_states": _ordered_counts(ignored),
        "refused_states": _ordered_counts(refused),
    }


def _explicit_milestones(inputs: Inputs) -> set[str]:
    reached: set[str] = set()
    for key in ("milestones", "milestone", "furthest_milestone"):
        value = inputs.summary.get(key)
        if value is None:
            continue
        for token in split_values(value):
            if token in _MILESTONE_SET:
                reached.add(token)
    for line in inputs.guest_lines + inputs.host_lines:
        for match in MILESTONE_LINE_RE.finditer(line):
            name = match.group(1)
            if name in _MILESTONE_SET:
                reached.add(name)
    for tokens in (inputs.guest_tokens, inputs.host_tokens):
        for value in tokens.get("milestone", []):
            if value in _MILESTONE_SET:
                reached.add(value)
    return reached


def collect_milestones(
    inputs: Inputs, frames: dict[str, Any], opcodes: dict[str, Any]
) -> dict[str, Any]:
    reached = _explicit_milestones(inputs)

    ready_line = inputs.summary.get("host_ready_line", "")
    if ready_line and not _is_none(ready_line):
        reached.add("host_ready")
    if inputs.flag(["host_ready"], ("summary", "guest", "host")):
        reached.add("host_ready")
    if any(
        "victory-metal-host: ready" in line.lower()
        or "ready port=" in line.lower()
        for line in inputs.host_lines
    ):
        reached.add("host_ready")

    def any_line(needles: Sequence[str]) -> bool:
        haystack = "\n".join(inputs.guest_lines + inputs.host_lines).lower()
        return any(needle in haystack for needle in needles)

    # Only explicit flags or textual phrases prove these stages.  A bare
    # `dstorage_queue=0` token must not be read as a reached milestone.
    if inputs.flag(
        ["d3d12x_device"], ("summary", "guest", "host")
    ) or any_line(("d3d12x device",)):
        reached.add("d3d12x_device")
    if inputs.flag(
        ["completion_page", "fence_page"], ("summary", "guest", "host")
    ) or any_line(("completion page", "fence page")):
        reached.add("completion_page")
    if inputs.flag(
        ["dstorage_queue"], ("summary", "guest", "host")
    ) or any_line(("dstorage queue", "dstorage ready")):
        reached.add("dstorage_queue")

    submits_ok = frames["submissions"]["ok"]
    native_submits = inputs.integer(
        ["native_submits", "first_native_submit"], ("summary", "guest", "host")
    )
    if (
        (submits_ok is not None and submits_ok > 0)
        or (native_submits is not None and native_submits > 0)
        or inputs.flag(["first_native_submit"], ("summary", "guest", "host"))
    ):
        reached.add("first_native_submit")

    presented = frames.get("presented")
    presents_ok = frames["presents"]["ok"]
    if (
        (presented is not None and presented > 0)
        or (presents_ok is not None and presents_ok > 0)
        or inputs.flag(["first_frame_presented"], ("summary", "guest", "host"))
    ):
        reached.add("first_frame_presented")

    histogram_names = set(opcodes["histogram"])
    draw_names = {
        name
        for name in histogram_names
        if "draw" in name or "execute_indirect" in name
    }
    dispatch_names = {name for name in histogram_names if "dispatch" in name}
    draws = inputs.integer(["draws", "draw_calls"], ("summary", "guest", "host"))
    dispatches = inputs.integer(
        ["dispatches", "compute_dispatches"], ("summary", "guest", "host")
    )
    if (
        draw_names
        or (draws is not None and draws > 0)
        or inputs.flag(["first_3d_draw"], ("summary", "guest", "host"))
    ):
        reached.add("first_3d_draw")
    if (
        dispatch_names
        or (dispatches is not None and dispatches > 0)
        or inputs.flag(["first_compute_dispatch"], ("summary", "guest", "host"))
    ):
        reached.add("first_compute_dispatch")

    sustained_flag = inputs.flag(["sustained_frames"], ("summary", "guest", "host"))
    sample_count = presented
    if sample_count is None:
        sample_count = frames["guest"].get("samples")
    if sustained_flag is True or (
        sample_count is not None and sample_count >= SUSTAINED_FRAMES
    ):
        reached.add("sustained_frames")

    # Report only milestones with direct evidence.  A later milestone does not
    # prove an unrelated earlier one (a presented frame does not prove the
    # DStorage queue was created); `furthest` is still the last one observed.
    reached_in_order = [name for name in MILESTONE_ORDER if name in reached]
    return {
        "furthest": reached_in_order[-1] if reached_in_order else None,
        "reached": reached_in_order,
    }


def _collect_refusal_categories(inputs: Inputs) -> tuple[Counter[str], bool]:
    categories: Counter[str] = Counter()
    seen = False

    for key in REFUSAL_COUNT_KEYS:
        if key in inputs.summary:
            counts, explicit = parse_named_counts(
                inputs.summary[key], label=opcode_label
            )
            categories.update(counts)
            seen = seen or explicit

    host_unsupported = inputs.summary.get("host_unsupported_opcodes")
    if host_unsupported is not None:
        parsed = parse_json_object(host_unsupported)
        if parsed is None:
            counts, explicit = parse_named_counts(host_unsupported)
            for name, count in counts.items():
                categories[classify_refusal(name)] += count
            seen = seen or explicit
        else:
            seen = True
            for message, count in parsed.items():
                number = parse_int(count)
                if number is not None and number > 0:
                    categories[classify_refusal(str(message))] += number
    else:
        # The host telemetry carries the same message-keyed histogram when the
        # launcher summary is missing; classify it, never echo a message.
        telemetry_unsupported = inputs.telemetry.get("unsupported_opcodes")
        if isinstance(telemetry_unsupported, dict):
            seen = True
            for message, count in telemetry_unsupported.items():
                number = parse_int(count)
                if number is not None and number > 0:
                    categories[classify_refusal(str(message))] += number

    for tokens in (inputs.guest_tokens, inputs.host_tokens):
        for key, values in tokens.items():
            # `named_refusals=[...]` is parsed from the whole bracketed list
            # below; the token view would only see its first element.
            if key in REFUSAL_COUNT_KEYS and key != "named_refusals":
                for value in values:
                    counts, explicit = parse_named_counts(value, label=opcode_label)
                    categories.update(counts)
                    seen = seen or explicit

    for line in inputs.guest_lines + inputs.host_lines:
        line_count = 1
        count_match = re.search(r"(?:^|\s)count=(\d+)", line)
        if count_match:
            line_count = int(count_match.group(1))
        for match in REFUSAL_TOKEN_RE.finditer(line):
            label = safe_label(match.group(2))
            if label is not None:
                categories[label] += line_count
                seen = True
        for match in NAMED_REFUSALS_RE.finditer(line):
            counts, explicit = parse_named_counts(match.group(1))
            categories.update(counts)
            seen = seen or explicit
        for match in REFUSALS_LIST_RE.finditer(line):
            counts, explicit = parse_named_counts(
                match.group(1), label=opcode_label
            )
            categories.update(counts)
            seen = seen or explicit

    return categories, seen


def _first_reason(inputs: Inputs) -> str | None:
    for line in inputs.guest_lines + inputs.host_lines:
        index = line.upper().find("METAL_HOST_ERROR")
        if index < 0:
            continue
        rest = line[index + len("METAL_HOST_ERROR") :].strip()
        rest = rest.lstrip(":=- ").strip()
        if not rest:
            continue
        reason = sanitise_reason(rest)
        if reason:
            return reason
    return None


def collect_refusals(inputs: Inputs) -> dict[str, Any]:
    categories, seen = _collect_refusal_categories(inputs)
    return {
        "top": _named_list(categories),
        "total": sum(categories.values()) if seen else None,
        "first_reason": _first_reason(inputs),
    }


def build_digest(out_dir: str) -> dict[str, Any]:
    inputs = Inputs(out_dir)
    run = collect_run(inputs)
    invariants = collect_invariants(inputs)
    frames = collect_frames(inputs)
    resources = collect_resources(inputs)
    shaders = collect_shaders(inputs)
    opcodes = collect_opcodes(inputs)
    milestones = collect_milestones(inputs, frames, opcodes)
    refusals = collect_refusals(inputs)

    failures: list[str] = []
    if invariants["d3dmetal_fallback_count"] not in (None, 0):
        failures.append("d3dmetal_fallback")
    if invariants["host_d3d12_creation_attempts"] not in (None, 0):
        failures.append("host_d3d12_creation")
    if (
        run["orphan_survived"] is True
        or run["host_survived"] is True
        or run["victim_survived"] is True
    ):
        failures.append("orphan_survived")

    return {
        "run": run,
        "invariants": invariants,
        "frames": frames,
        "resources": resources,
        "shaders": shaders,
        "opcodes": opcodes,
        "milestones": milestones,
        "refusals": refusals,
        "gate": {"ok": not failures, "failures": failures},
    }


# --- Rendering -------------------------------------------------------------


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _histogram(counter: dict[str, int]) -> str:
    if not counter:
        return "{}"
    return ",".join(f"{name}={count}" for name, count in counter.items())


def render_text(digest: dict[str, Any]) -> str:
    """A compact human digest with the same values as the JSON form."""
    run = digest["run"]
    inv = digest["invariants"]
    frames = digest["frames"]
    resources = digest["resources"]
    shaders = digest["shaders"]
    opcodes = digest["opcodes"]
    milestones = digest["milestones"]
    refusals = digest["refusals"]
    gate = digest["gate"]

    lines = [
        "run "
        + " ".join(
            f"{key}={_scalar(run[key])}"
            for key in (
                "mode",
                "status",
                "exit_code",
                "guest_exit",
                "host_exit",
                "host_terminated_by_launcher",
                "guest_timed_out",
                "host_timed_out",
                "host_survived",
                "victim_survived",
                "orphan_survived",
                "summary_present",
                "telemetry_present",
                "telemetry_valid",
                "guest_log_present",
                "host_log_present",
            )
        ),
        "invariants "
        + " ".join(
            f"{key}={_scalar(value)}" for key, value in inv.items()
        ),
        "frames "
        f"presented={_scalar(frames['presented'])} "
        f"guest={_scalar(frames['guest']['p50_ms'])}/"
        f"{_scalar(frames['guest']['p95_ms'])}/"
        f"{_scalar(frames['guest']['p99_ms'])}ms "
        f"fps={_scalar(frames['guest']['fps'])} "
        f"samples={_scalar(frames['guest']['samples'])} "
        f"host={_scalar(frames['host']['p50_ms'])}/"
        f"{_scalar(frames['host']['p95_ms'])}/"
        f"{_scalar(frames['host']['p99_ms'])}ms "
        f"fps={_scalar(frames['host']['fps'])} "
        f"samples={_scalar(frames['host']['samples'])} "
        f"submissions={_scalar(frames['submissions']['ok'])}/"
        f"{_scalar(frames['submissions']['refused'])} "
        f"presents={_scalar(frames['presents']['ok'])}/"
        f"{_scalar(frames['presents']['refused'])}",
        "resources "
        f"buffers={_scalar(resources['buffers']['registered'])}/"
        f"{_scalar(resources['buffers']['refused'])} "
        f"textures={_scalar(resources['textures']['registered'])}/"
        f"{_scalar(resources['textures']['refused'])} "
        f"refused_sizes={_histogram({key: resources['refused_sizes'][key] for key in ('lt_4kib', 'lt_1mib', 'lt_64mib', 'ge_64mib')})} "
        f"worst_bytes={_scalar(resources['refused_sizes']['worst_bytes'])} "
        f"samples={_scalar(resources['refused_sizes']['samples'])} "
        f"formats={_histogram(resources['formats_refused'])}",
        "shaders "
        f"compiled={_scalar(shaders['compiled'])} "
        f"accepted={_scalar(shaders['accepted'])} "
        f"refused={_scalar(shaders['refused'])} "
        f"top={','.join(item['name'] + '=' + str(item['count']) for item in shaders['top_refusals']) or '{}'}",
        "opcodes "
        f"histogram={_histogram(opcodes['histogram'])} "
        f"ignored={_scalar(opcodes['ignored'])} "
        f"refused={_scalar(opcodes['refused'])} "
        f"ignored_states={_histogram(opcodes['ignored_states'])} "
        f"refused_states={_histogram(opcodes['refused_states'])}",
        "milestones "
        f"furthest={_scalar(milestones['furthest'])} "
        f"reached={','.join(milestones['reached']) or 'none'}",
        "refusals "
        f"total={_scalar(refusals['total'])} "
        f"top={','.join(item['name'] + '=' + str(item['count']) for item in refusals['top']) or '{}'} "
        f"first_reason={_scalar(refusals['first_reason'])}",
        "gate "
        f"ok={_scalar(gate['ok'])} "
        f"failures={','.join(gate['failures']) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


# --- CLI -------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="boot-census.py",
        description=(
            "Summarise one run-title-metal.sh --out directory as counts, "
            "categories and frame metrics.  No title is launched and no file "
            "path, guest memory or asset byte count is printed."
        ),
        epilog=(
            "Exit 2 under --gate when the D3DMetal fallback count, the host "
            "D3D12 creation attempt count or an orphan survival flag is "
            "non-zero; exit 0 otherwise.  Missing inputs are null."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="launcher --out directory (summary.txt, telemetry.json, *.log)",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        help="emit one JSON object (default)",
    )
    fmt.add_argument(
        "--text",
        dest="format",
        action="store_const",
        const="text",
        help="emit the same digest as compact text",
    )
    parser.set_defaults(format="json")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit 2 on a fallback, host D3D12 creation or orphan survivor",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        digest = build_digest(args.out)
    except Exception as error:  # pragma: no cover - defensive last resort
        print(f"boot-census.py: internal error: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        sys.stdout.write(
            json.dumps(digest, indent=2, ensure_ascii=True, sort_keys=False) + "\n"
        )
    else:
        sys.stdout.write(render_text(digest))

    if args.gate and not digest["gate"]["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
