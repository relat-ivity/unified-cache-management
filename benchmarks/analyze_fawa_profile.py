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


def cross_rank_durations_ms(records: list[dict[str, object]]) -> tuple[list[float], list[int]]:
    by_request_ids: dict[str, list[dict[str, object]]] = {}
    for record in records:
        request_ids = record.get("request_ids")
        if not isinstance(request_ids, str) or not request_ids:
            continue
        by_request_ids.setdefault(request_ids, []).append(record)

    durations_ms: list[float] = []
    rank_counts: list[int] = []
    for group in by_request_ids.values():
        starts = values(group, "start_us")
        ends = values(group, "end_us")
        if not starts or not ends:
            continue
        ranks = {
            record.get("local_rank", record.get("tp_rank"))
            for record in group
            if record.get("local_rank", record.get("tp_rank")) is not None
        }
        durations_ms.append((max(ends) - min(starts)) / 1000)
        rank_counts.append(len(ranks) if ranks else len(group))
    return durations_ms, rank_counts


def summarize_group(label: str, paths: list[Path]) -> dict[str, object]:
    records = parse_logs(paths)
    cross_rank_ms, cross_rank_counts = cross_rank_durations_ms(records)
    return {
        "label": label,
        "paths": [str(path) for path in paths],
        "profiles": len(records),
        "requests": total(records, "requests"),
        "load_requests": total(records, "load_requests"),
        "tasks": total(records, "tasks"),
        "wall_ms": stats(time_values_ms(records, "wall_ms", "wall_us")),
        "cross_rank_groups": len(cross_rank_ms),
        "cross_rank_ranks": stats([float(count) for count in cross_rank_counts]),
        "cross_rank_wall_ms": stats(cross_rank_ms),
        "errors": sum(1 for record in records if record.get("status") == "error"),
    }


def fmt_ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def print_summary(summary: dict[str, object]) -> None:
    wall = summary["wall_ms"]
    cross_rank_wall = summary["cross_rank_wall_ms"]
    cross_rank_ranks = summary["cross_rank_ranks"]
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
    if summary["cross_rank_groups"]:
        print("cross_rank_start_load_kv:")
        print(
            f"  groups: {summary['cross_rank_groups']}, "
            f"ranks_mean={cross_rank_ranks['mean']:.3f}, "
            f"ranks_min={cross_rank_ranks['min']:.0f}, "
            f"ranks_max={cross_rank_ranks['max']:.0f}"
        )
        print(
            "  wall_ms: "
            f"mean={fmt_ms(cross_rank_wall['mean'])}, "
            f"p50={fmt_ms(cross_rank_wall['p50'])}, "
            f"p90={fmt_ms(cross_rank_wall['p90'])}, "
            f"p99={fmt_ms(cross_rank_wall['p99'])}, "
            f"sum={fmt_ms(cross_rank_wall['sum'])}"
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

    base_cross = baseline["cross_rank_wall_ms"]
    cand_cross = candidate["cross_rank_wall_ms"]
    if base_cross["count"] or cand_cross["count"]:
        for metric in ("mean", "p50", "p90", "sum"):
            base = base_cross[metric]
            cand = cand_cross[metric]
            if math.isnan(base) or math.isnan(cand) or cand == 0:
                print(f"cross_rank_start_load_kv_wall.{metric}: n/a")
                continue
            speedup = base / cand
            reduction = (base - cand) / base * 100 if base else math.nan
            print(
                f"cross_rank_start_load_kv_wall.{metric}: "
                f"baseline={base:.3f} ms, candidate={cand:.3f} ms, "
                f"speedup={speedup:.3f}x, reduction={reduction:.2f}%"
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
