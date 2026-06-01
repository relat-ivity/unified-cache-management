#!/usr/bin/env python3
"""Analyze FAWA start_load_kv wall-time records from vLLM logs."""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Iterable


START_LOAD_KV_PROFILE_RE = re.compile(
    r"FAWA connector start_load_kv profile (?P<fields>.*)"
)
FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")
NUMERIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)$")


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def parse_value(value: str) -> object:
    if NUMERIC_RE.match(value):
        number = float(value)
        if number.is_integer() and "." not in value:
            return int(number)
        return number
    return value


def parse_logs(paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                match = START_LOAD_KV_PROFILE_RE.search(line)
                if not match:
                    continue
                record: dict[str, object] = {
                    "source": str(path),
                    "line": line_no,
                }
                for field in FIELD_RE.finditer(match.group("fields")):
                    record[field.group("key")] = parse_value(field.group("value"))
                records.append(record)
    return records


def values(records: list[dict[str, object]], key: str) -> list[float]:
    vals: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return vals


def time_values_ms(
    records: list[dict[str, object]],
    key_ms: str,
    key_us: str | None = None,
) -> list[float]:
    if key_us is None:
        key_us = key_ms.replace("_ms", "_us")
    vals: list[float] = []
    for record in records:
        if isinstance(record.get(key_us), (int, float)):
            vals.append(float(record[key_us]) / 1000)
        elif isinstance(record.get(key_ms), (int, float)):
            vals.append(float(record[key_ms]))
    return vals


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return math.nan
    vals = sorted(vals)
    rank = (len(vals) - 1) * pct / 100
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - rank) + vals[hi] * (rank - lo)


def stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(vals),
        "sum": sum(vals),
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "min": min(vals),
        "max": max(vals),
    }


def total(records: list[dict[str, object]], key: str) -> float:
    return sum(values(records, key))


def summarize_group(label: str, paths: list[Path]) -> dict[str, object]:
    records = parse_logs(paths)
    return {
        "label": label,
        "paths": [str(path) for path in paths],
        "profiles": len(records),
        "requests": total(records, "requests"),
        "load_requests": total(records, "load_requests"),
        "tasks": total(records, "tasks"),
        "wall_ms": stats(time_values_ms(records, "wall_ms", "wall_us")),
        "errors": sum(1 for record in records if record.get("status") == "error"),
    }


def fmt_ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def print_summary(summary: dict[str, object]) -> None:
    wall = summary["wall_ms"]
    print(f"== {summary['label']} ==")
    print(f"logs: {', '.join(summary['paths'])}")
    print("start_load_kv:")
    print(
        f"  profiles: {summary['profiles']}, "
        f"requests: {int(summary['requests'])}, "
        f"load_requests: {int(summary['load_requests'])}, "
        f"tasks: {int(summary['tasks'])}, "
        f"errors: {summary['errors']}"
    )
    print(
        "  wall_ms: "
        f"mean={fmt_ms(wall['mean'])}, p50={fmt_ms(wall['p50'])}, "
        f"p90={fmt_ms(wall['p90'])}, p99={fmt_ms(wall['p99'])}, "
        f"sum={fmt_ms(wall['sum'])}"
    )


def print_comparison(baseline: dict[str, object], candidate: dict[str, object]) -> None:
    print("== comparison ==")
    base_wall = baseline["wall_ms"]
    cand_wall = candidate["wall_ms"]
    for metric in ("mean", "p50", "p90", "sum"):
        base = base_wall[metric]
        cand = cand_wall[metric]
        if math.isnan(base) or math.isnan(cand) or cand == 0:
            print(f"start_load_kv_wall.{metric}: n/a")
            continue
        speedup = base / cand
        reduction = (base - cand) / base * 100 if base else math.nan
        print(
            f"start_load_kv_wall.{metric}: baseline={base:.3f} ms, "
            f"candidate={cand:.3f} ms, speedup={speedup:.3f}x, "
            f"reduction={reduction:.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FAWA connector start_load_kv wall-time records."
    )
    parser.add_argument("logs", nargs="*", help="Log files or glob patterns.")
    parser.add_argument(
        "--baseline",
        nargs="+",
        help="Baseline log files or glob patterns for comparison.",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        help="Candidate log files or glob patterns for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline or args.candidate:
        if not args.baseline or not args.candidate:
            raise SystemExit("--baseline and --candidate must be used together.")
        baseline = summarize_group("baseline", expand_paths(args.baseline))
        candidate = summarize_group("candidate", expand_paths(args.candidate))
        print_summary(baseline)
        print()
        print_summary(candidate)
        print()
        print_comparison(baseline, candidate)
        return

    if not args.logs:
        raise SystemExit("No log files provided.")
    print_summary(summarize_group("logs", expand_paths(args.logs)))


if __name__ == "__main__":
    main()
