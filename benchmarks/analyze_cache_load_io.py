#!/usr/bin/env python3
"""Analyze CacheStore load H2D per-transfer IO stats from logs."""

from __future__ import annotations

import argparse
import glob
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


TASK_IO_STATS_RE = re.compile(
    r"Cache load H2D task IO stats: "
    r"task=(?P<task>\d+), "
    r"load_shard_count=(?P<shard_count>\d+), "
    r"load_io_count=(?P<io_count>\d+), "
    r"load_total_bytes=(?P<total_bytes>\d+), "
    r"load_io_size_counts=(?P<size_counts>\{[^}]*\})"
)
DETAIL_IO_RE = re.compile(
    r"Cache load H2D scatter IO: .*?size_bytes=(?P<size_bytes>\d+)"
)
SIZE_COUNT_RE = re.compile(r"(?P<size>\d+)\s*:\s*(?P<count>\d+)")


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.3f} {unit}" if unit != "B" else f"{value} B"
        size /= 1024
    return f"{value} B"


def parse_size_counts(text: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for match in SIZE_COUNT_RE.finditer(text):
        counts[int(match.group("size"))] += int(match.group("count"))
    return counts


def parse_logs(paths: list[Path]) -> dict[str, object]:
    task_records = 0
    detail_records = 0
    shard_count = 0
    io_count = 0
    total_bytes = 0
    computed_bytes = 0
    mismatches = 0
    counts_by_size: Counter[int] = Counter()
    detail_counts_by_size: Counter[int] = Counter()

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                task_match = TASK_IO_STATS_RE.search(line)
                if task_match:
                    task_records += 1
                    task_shards = int(task_match.group("shard_count"))
                    task_io_count = int(task_match.group("io_count"))
                    task_total_bytes = int(task_match.group("total_bytes"))
                    task_counts = parse_size_counts(task_match.group("size_counts"))
                    task_computed_count = sum(task_counts.values())
                    task_computed_bytes = sum(
                        size * count for size, count in task_counts.items()
                    )

                    shard_count += task_shards
                    io_count += task_io_count
                    total_bytes += task_total_bytes
                    computed_bytes += task_computed_bytes
                    counts_by_size.update(task_counts)
                    if (
                        task_computed_count != task_io_count
                        or task_computed_bytes != task_total_bytes
                    ):
                        mismatches += 1
                    continue

                detail_match = DETAIL_IO_RE.search(line)
                if detail_match:
                    detail_records += 1
                    detail_counts_by_size[int(detail_match.group("size_bytes"))] += 1

    if task_records == 0 and detail_records:
        counts_by_size = detail_counts_by_size
        io_count = sum(counts_by_size.values())
        total_bytes = sum(size * count for size, count in counts_by_size.items())
        computed_bytes = total_bytes

    return {
        "paths": paths,
        "task_records": task_records,
        "detail_records": detail_records,
        "shard_count": shard_count,
        "io_count": io_count,
        "total_bytes": total_bytes,
        "computed_bytes": computed_bytes,
        "mismatches": mismatches,
        "counts_by_size": counts_by_size,
        "mode": "task" if task_records else "detail",
    }


def print_summary(summary: dict[str, object]) -> None:
    counts_by_size: Counter[int] = summary["counts_by_size"]  # type: ignore[assignment]
    total_bytes = int(summary["total_bytes"])

    print("== Cache load H2D IO ==")
    print(f"logs: {', '.join(str(path) for path in summary['paths'])}")
    if summary["mode"] == "task":
        print(f"task_records: {summary['task_records']}")
        print(f"load_shard_count: {summary['shard_count']}")
        if summary["mismatches"]:
            print(f"warning: {summary['mismatches']} task records have count/byte mismatch")
    else:
        print("mode: per-transfer detail")
        print(f"detail_records: {summary['detail_records']}")
    print(f"load_io_count: {summary['io_count']}")
    print(f"load_total_bytes: {total_bytes} ({human_bytes(total_bytes)})")
    print("load_io_size_counts:")
    for size, count in sorted(counts_by_size.items()):
        bytes_for_size = size * count
        print(
            f"  {size}: count={count}, bytes={bytes_for_size} "
            f"({human_bytes(bytes_for_size)})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CacheStore load H2D per-transfer IO size/count totals."
    )
    parser.add_argument("logs", nargs="+", help="Log files or glob patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.logs)
    print_summary(parse_logs(paths))


if __name__ == "__main__":
    main()
