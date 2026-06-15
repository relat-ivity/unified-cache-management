#!/usr/bin/env python3
"""Analyze connector start_load_kv wall-time records from vLLM logs."""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Iterable


START_LOAD_KV_PROFILE_RE = re.compile(
    r"(?:FAWA|UCM|HMA) connector start_load_kv profile (?P<fields>.*)"
)
HIT_EXTERNAL_RE = re.compile(
    r"(?:FAWA\s+)?request_id:\s*(?P<request_id>[^,\s]+),"
    r".*?hit external:\s*(?P<hit_external>-?\d+)"
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


def record_request_ids(record: dict[str, object]) -> list[str]:
    request_ids = record.get("request_ids")
    if not isinstance(request_ids, str) or not request_ids:
        return []
    return [request_id for request_id in request_ids.split(",") if request_id]


def matches_hit_external_filter(
    record: dict[str, object],
    hit_external: int | None,
    hit_external_by_request_id: dict[str, set[int]],
) -> bool:
    if hit_external is None:
        return True

    hit_values: list[int] = []
    for request_id in record_request_ids(record):
        hit_values.extend(hit_external_by_request_id.get(request_id, ()))
    return bool(hit_values) and all(value == hit_external for value in hit_values)


def parse_logs(
    paths: Iterable[Path],
    hit_external: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    path_lines: list[tuple[Path, list[str]]] = []
    hit_external_by_request_id: dict[str, set[int]] = {}
    records: list[dict[str, object]] = []
    stats = {
        "profiles_scanned": 0,
        "profiles_filtered_out": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = list(f)
        path_lines.append((path, lines))
        for line in lines:
            hit_match = HIT_EXTERNAL_RE.search(line)
            if not hit_match:
                continue
            hit_external_by_request_id.setdefault(
                hit_match.group("request_id"), set()
            ).add(int(hit_match.group("hit_external")))

    for path, lines in path_lines:
        for line_no, line in enumerate(lines, 1):
            match = START_LOAD_KV_PROFILE_RE.search(line)
            if not match:
                continue
            stats["profiles_scanned"] += 1
            record: dict[str, object] = {
                "source": str(path),
                "line": line_no,
            }
            for field in FIELD_RE.finditer(match.group("fields")):
                record[field.group("key")] = parse_value(field.group("value"))
            if not matches_hit_external_filter(
                record, hit_external, hit_external_by_request_id
            ):
                stats["profiles_filtered_out"] += 1
                continue
            records.append(record)
    return records, stats


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


def summarize_group(
    label: str,
    paths: list[Path],
    hit_external: int | None = None,
) -> dict[str, object]:
    records, parse_stats = parse_logs(paths, hit_external)
    cross_rank_ms, cross_rank_counts = cross_rank_durations_ms(records)
    return {
        "label": label,
        "paths": [str(path) for path in paths],
        "hit_external_filter": hit_external,
        **parse_stats,
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
    if summary["hit_external_filter"] is not None:
        print(
            f"filter: hit external={summary['hit_external_filter']} "
            f"(kept {summary['profiles']} of {summary['profiles_scanned']} profiles)"
        )
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
        description="Analyze connector start_load_kv wall-time records."
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
    parser.add_argument(
        "--hit-external",
        "--external-hit",
        "--hit-external-blocks",
        dest="hit_external",
        type=int,
        help=(
            "Only include transfer profiles whose request_ids have logged "
            "'hit external' equal to this block count."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline or args.candidate:
        if not args.baseline or not args.candidate:
            raise SystemExit("--baseline and --candidate must be used together.")
        baseline = summarize_group(
            "baseline", expand_paths(args.baseline), args.hit_external
        )
        candidate = summarize_group(
            "candidate", expand_paths(args.candidate), args.hit_external
        )
        print_summary(baseline)
        print()
        print_summary(candidate)
        print()
        print_comparison(baseline, candidate)
        return

    if not args.logs:
        raise SystemExit("No log files provided.")
    print_summary(summarize_group("logs", expand_paths(args.logs), args.hit_external))


if __name__ == "__main__":
    main()
