#!/usr/bin/env python3
"""Analyze UCM CacheStore transfer timing records from vLLM logs."""

from __future__ import annotations

import argparse
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROFILE_RE = re.compile(r"FAWA profile (?P<event>\w+) (?P<fields>.*)")
CONNECTOR_LOAD_PROFILE_RE = re.compile(
    r"FAWA connector load profile (?P<fields>.*)"
)
CONNECTOR_LOAD_TASK_RE = re.compile(r"FAWA connector load task (?P<fields>.*)")
START_LOAD_KV_PROFILE_RE = re.compile(
    r"FAWA connector start_load_kv profile (?P<fields>.*)"
)
TRANSFER_RE = re.compile(r"UCM transfer profile (?P<fields>.*)")
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


def parse_logs(paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    events: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                match = TRANSFER_RE.search(line)
                event = "transfer"
                if not match:
                    match = START_LOAD_KV_PROFILE_RE.search(line)
                    if match:
                        event = "start_load_kv"
                    else:
                        match = CONNECTOR_LOAD_PROFILE_RE.search(line)
                        if match:
                            event = "connector_load"
                        else:
                            match = CONNECTOR_LOAD_TASK_RE.search(line)
                            if match:
                                event = "connector_load_task"
                            else:
                                match = PROFILE_RE.search(line)
                                if not match:
                                    continue
                                event = match.group("event")
                record: dict[str, object] = {
                    "source": str(path),
                    "line": line_no,
                }
                for field in FIELD_RE.finditer(match.group("fields")):
                    record[field.group("key")] = parse_value(field.group("value"))
                events[event].append(record)
    return events

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


def total(records: list[dict[str, object]], key: str) -> float:
    return sum(values(records, key))


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


def gib_per_s(byte_count: float, duration_ms: float) -> float:
    if duration_ms <= 0:
        return math.nan
    return byte_count / duration_ms / 1024 / 1024


def summarize_operation(
    events: dict[str, list[dict[str, object]]],
    summary_event: str,
    task_event: str,
) -> dict[str, object]:
    summaries = events.get(summary_event, [])
    tasks = events.get(task_event, [])
    summary_wall = time_values_ms(summaries, "wall_ms", "wall_us")
    task_wait = time_values_ms(tasks, "wait_ms", "wait_us")
    duration_values = summary_wall if summary_wall else task_wait
    bytes_total = total(summaries, "bytes") if summaries else total(tasks, "bytes")
    keys_total = total(summaries, "keys") if summaries else total(tasks, "keys")
    duration_sum = sum(duration_values)

    if summaries:
        submit_sum_values = time_values_ms(
            summaries, "submit_ms_sum", "submit_us_sum"
        )
        wait_sum_values = time_values_ms(summaries, "wait_ms_sum", "wait_us_sum")
        fa_wait_values = time_values_ms(
            summaries, "fa_wait_ms_sum", "fa_wait_us_sum"
        )
        wa_wait_values = time_values_ms(
            summaries, "wa_wait_ms_sum", "wa_wait_us_sum"
        )
    else:
        submit_sum_values = time_values_ms(tasks, "submit_ms", "submit_us")
        wait_sum_values = task_wait
        fa_wait_values = time_values_ms(
            [task for task in tasks if task.get("label") == "FA"],
            "wait_ms",
            "wait_us",
        )
        wa_wait_values = time_values_ms(
            [task for task in tasks if task.get("label") == "WA"],
            "wait_ms",
            "wait_us",
        )

    result: dict[str, object] = {
        "summaries": len(summaries),
        "tasks": len(tasks),
        "duration_source": "summary_wall" if summaries else "task_wait",
        "keys": keys_total,
        "bytes": bytes_total,
        "wall_ms": stats(duration_values),
        "submit_ms_sum": stats(submit_sum_values),
        "wait_ms_sum": stats(wait_sum_values),
        "fa_wait_ms_sum": stats(fa_wait_values),
        "wa_wait_ms_sum": stats(wa_wait_values),
        "throughput_gib_s": gib_per_s(bytes_total, duration_sum),
        "task_submit_ms": stats(time_values_ms(tasks, "submit_ms", "submit_us")),
        "task_wait_ms": stats(task_wait),
        "errors": sum(1 for task in tasks if task.get("status") == "error"),
    }
    if task_event == "store_task":
        result["task_elapsed_since_submit_ms"] = stats(
            time_values_ms(
                tasks,
                "elapsed_since_submit_ms",
                "elapsed_since_submit_us",
            )
        )
    return result


def summarize_transfer_operation(
    events: dict[str, list[dict[str, object]]],
    op: str,
) -> dict[str, object]:
    records = [record for record in events.get("transfer", []) if record.get("op") == op]
    elapsed = time_values_ms(records, "elapsed_ms", "elapsed_us")
    submit = time_values_ms(records, "submit_ms", "submit_us")
    sync = time_values_ms(records, "sync_ms", "sync_us")
    bytes_total = total(records, "bytes")
    return {
        "summaries": 0,
        "tasks": len(records),
        "duration_source": "stream_elapsed",
        "shards": total(records, "shards"),
        "keys": total(records, "shards"),
        "bytes": bytes_total,
        "wall_ms": stats(elapsed),
        "submit_ms_sum": stats(submit),
        "wait_ms_sum": stats(sync),
        "fa_wait_ms_sum": stats([]),
        "wa_wait_ms_sum": stats([]),
        "throughput_gib_s": gib_per_s(bytes_total, sum(elapsed)),
        "task_submit_ms": stats(submit),
        "task_wait_ms": stats(sync),
        "task_elapsed_ms": stats(elapsed),
        "errors": sum(1 for record in records if record.get("status") == "error"),
    }


def summarize_connector_load(
    events: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    profiles = events.get("connector_load", [])
    tasks = events.get("connector_load_task", [])
    load_elapsed_by_task: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in events.get("transfer", []):
        if record.get("op") != "load":
            continue
        device_id = record.get("device_id")
        task_id = record.get("task_id")
        elapsed_us = record.get("elapsed_us")
        elapsed_ms = record.get("elapsed_ms")
        if not isinstance(device_id, (int, float)):
            continue
        if not isinstance(task_id, (int, float)):
            continue
        if isinstance(elapsed_us, (int, float)):
            elapsed = float(elapsed_us) / 1000
        elif isinstance(elapsed_ms, (int, float)):
            elapsed = float(elapsed_ms)
        else:
            continue
        load_elapsed_by_task[(int(device_id), int(task_id))].append(elapsed)

    tasks_by_request_rank: dict[tuple[object, object], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for task in tasks:
        tasks_by_request_rank[(task.get("request_id"), task.get("local_rank"))].append(
            task
        )

    cache_max_elapsed_ms: list[float] = []
    rough_overhead_ms: list[float] = []
    for profile in profiles:
        wall_us = profile.get("wall_us")
        wall_ms = profile.get("wall_ms")
        if isinstance(wall_us, (int, float)):
            connector_wall_ms = float(wall_us) / 1000
        elif isinstance(wall_ms, (int, float)):
            connector_wall_ms = float(wall_ms)
        else:
            continue

        cache_elapsed: list[float] = []
        for task in tasks_by_request_rank.get(
            (profile.get("request_id"), profile.get("local_rank")),
            [],
        ):
            local_rank = task.get("local_rank")
            task_id = task.get("task_id")
            if not isinstance(local_rank, (int, float)):
                continue
            if not isinstance(task_id, (int, float)):
                continue
            elapsed = load_elapsed_by_task.get((int(local_rank), int(task_id)), [])
            if elapsed:
                cache_elapsed.append(max(elapsed))
        if not cache_elapsed:
            continue
        cache_max = max(cache_elapsed)
        cache_max_elapsed_ms.append(cache_max)
        rough_overhead_ms.append(connector_wall_ms - cache_max)

    return {
        "profiles": len(profiles),
        "tasks": len(tasks),
        "fa_keys": total(profiles, "fa_keys"),
        "wa_keys": total(profiles, "wa_keys"),
        "wall_ms": stats(time_values_ms(profiles, "wall_ms", "wall_us")),
        "extract_ms": stats(time_values_ms(profiles, "extract_ms", "extract_us")),
        "submit_ms": stats(time_values_ms(profiles, "submit_ms", "submit_us")),
        "wait_ms": stats(time_values_ms(profiles, "wait_ms", "wait_us")),
        "task_submit_ms": stats(time_values_ms(tasks, "submit_ms", "submit_us")),
        "task_wait_ms": stats(time_values_ms(tasks, "wait_ms", "wait_us")),
        "task_elapsed_since_submit_ms": stats(
            time_values_ms(
                tasks,
                "elapsed_since_submit_ms",
                "elapsed_since_submit_us",
            )
        ),
        "cache_max_elapsed_ms": stats(cache_max_elapsed_ms),
        "rough_overhead_ms": stats(rough_overhead_ms),
        "errors": (
            sum(1 for profile in profiles if profile.get("status") == "error")
            + sum(1 for task in tasks if task.get("status") == "error")
        ),
    }


def summarize_start_load_kv(
    events: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    profiles = events.get("start_load_kv", [])
    return {
        "profiles": len(profiles),
        "requests": total(profiles, "requests"),
        "load_requests": total(profiles, "load_requests"),
        "tasks": total(profiles, "tasks"),
        "wall_ms": stats(time_values_ms(profiles, "wall_ms", "wall_us")),
        "errors": sum(1 for profile in profiles if profile.get("status") == "error"),
    }


def summarize_group(label: str, paths: list[Path]) -> dict[str, object]:
    events = parse_logs(paths)
    if events.get("transfer"):
        return {
            "label": label,
            "mode": "ucm_transfer",
            "paths": [str(path) for path in paths],
            "load": summarize_transfer_operation(events, "load"),
            "store": summarize_transfer_operation(events, "dump"),
            "connector_load": summarize_connector_load(events),
            "start_load_kv": summarize_start_load_kv(events),
        }
    return {
        "label": label,
        "mode": "fawa_profile",
        "paths": [str(path) for path in paths],
        "lookup": {
            "records": len(events.get("lookup", [])),
            "hash_ms": stats(
                time_values_ms(events.get("lookup", []), "hash_ms", "hash_us")
            ),
            "lookup_ms": stats(
                time_values_ms(events.get("lookup", []), "lookup_ms", "lookup_us")
            ),
            "external_hit_blocks": stats(
                values(events.get("lookup", []), "external_hit_blocks")
            ),
        },
        "load": summarize_operation(events, "load_summary", "load_task"),
        "store": summarize_operation(events, "store_summary", "store_task"),
        "connector_load": summarize_connector_load(events),
        "start_load_kv": summarize_start_load_kv(events),
    }


def fmt_ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def fmt_num(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def op_total_ms(summary: dict[str, object], op: str) -> float:
    return float(summary[op]["wall_ms"]["sum"])


def op_total_bytes(summary: dict[str, object], op: str) -> float:
    return float(summary[op]["bytes"])


def print_operation(name: str, summary: dict[str, object]) -> None:
    wall = summary["wall_ms"]
    submit = summary["submit_ms_sum"]
    wait = summary["wait_ms_sum"]
    fa_wait = summary["fa_wait_ms_sum"]
    wa_wait = summary["wa_wait_ms_sum"]
    print(f"{name}:")
    print(
        f"  summaries: {summary['summaries']}, tasks: {summary['tasks']}, "
        f"duration_source: {summary['duration_source']}"
    )
    if "shards" in summary:
        print(f"  shards: {int(summary['shards'])}, bytes: {int(summary['bytes'])}")
        print(
            "  elapsed_ms: "
            f"mean={fmt_ms(wall['mean'])}, p50={fmt_ms(wall['p50'])}, "
            f"p90={fmt_ms(wall['p90'])}, p99={fmt_ms(wall['p99'])}, "
            f"sum={fmt_ms(wall['sum'])}"
        )
        print(
            "  submit/sync ms: "
            f"submit_mean={fmt_ms(submit['mean'])}, "
            f"sync_mean={fmt_ms(wait['mean'])}, sync_sum={fmt_ms(wait['sum'])}"
        )
    else:
        print(f"  keys: {int(summary['keys'])}, bytes: {int(summary['bytes'])}")
        print(
            "  wall_ms: "
            f"mean={fmt_ms(wall['mean'])}, p50={fmt_ms(wall['p50'])}, "
            f"p90={fmt_ms(wall['p90'])}, p99={fmt_ms(wall['p99'])}, "
            f"sum={fmt_ms(wall['sum'])}"
        )
        print(
            "  submit/wait sum ms: "
            f"submit_mean={fmt_ms(submit['mean'])}, "
            f"wait_mean={fmt_ms(wait['mean'])}, "
            f"fa_wait_mean={fmt_ms(fa_wait['mean'])}, "
            f"wa_wait_mean={fmt_ms(wa_wait['mean'])}"
        )
    print(f"  throughput_gib_s: {fmt_num(summary['throughput_gib_s'])}")
    print(f"  errors: {summary['errors']}")

    task_wait = summary["task_wait_ms"]
    if "shards" in summary:
        task_elapsed = summary["task_elapsed_ms"]
        print(
            "  task_elapsed_ms: "
            f"mean={fmt_ms(task_elapsed['mean'])}, p50={fmt_ms(task_elapsed['p50'])}, "
            f"p90={fmt_ms(task_elapsed['p90'])}, p99={fmt_ms(task_elapsed['p99'])}"
        )
    else:
        print(
            "  task_wait_ms: "
            f"mean={fmt_ms(task_wait['mean'])}, p50={fmt_ms(task_wait['p50'])}, "
            f"p90={fmt_ms(task_wait['p90'])}, p99={fmt_ms(task_wait['p99'])}"
        )
    elapsed = summary.get("task_elapsed_since_submit_ms")
    if isinstance(elapsed, dict):
        print(
            "  task_elapsed_since_submit_ms: "
            f"mean={fmt_ms(elapsed['mean'])}, p50={fmt_ms(elapsed['p50'])}, "
            f"p90={fmt_ms(elapsed['p90'])}, p99={fmt_ms(elapsed['p99'])}"
        )


def print_transfer_totals(summary: dict[str, object]) -> None:
    load_ms = op_total_ms(summary, "load")
    dump_ms = op_total_ms(summary, "store")
    total_ms = load_ms + dump_ms
    load_bytes = op_total_bytes(summary, "load")
    dump_bytes = op_total_bytes(summary, "store")
    total_bytes = load_bytes + dump_bytes
    print("transfer totals:")
    print(f"  load_total_ms: {fmt_ms(load_ms)}")
    print(f"  dump_total_ms: {fmt_ms(dump_ms)}")
    print(f"  load_plus_dump_total_ms: {fmt_ms(total_ms)}")
    print(f"  load_total_bytes: {int(load_bytes)}")
    print(f"  dump_total_bytes: {int(dump_bytes)}")
    print(f"  load_plus_dump_total_bytes: {int(total_bytes)}")
    print(f"  load_plus_dump_gib_s: {fmt_num(gib_per_s(total_bytes, total_ms))}")


def print_connector_load(summary: dict[str, object]) -> None:
    connector = summary.get("connector_load")
    if not isinstance(connector, dict):
        return
    if not connector.get("profiles") and not connector.get("tasks"):
        return

    wall = connector["wall_ms"]
    extract = connector["extract_ms"]
    submit = connector["submit_ms"]
    wait = connector["wait_ms"]
    task_submit = connector["task_submit_ms"]
    task_wait = connector["task_wait_ms"]
    task_elapsed = connector["task_elapsed_since_submit_ms"]
    cache_max = connector["cache_max_elapsed_ms"]
    rough_overhead = connector["rough_overhead_ms"]
    print("connector load:")
    print(
        f"  profiles: {connector['profiles']}, tasks: {connector['tasks']}, "
        f"fa_keys: {int(connector['fa_keys'])}, "
        f"wa_keys: {int(connector['wa_keys'])}, errors: {connector['errors']}"
    )
    print(
        "  wall_ms: "
        f"mean={fmt_ms(wall['mean'])}, p50={fmt_ms(wall['p50'])}, "
        f"p90={fmt_ms(wall['p90'])}, p99={fmt_ms(wall['p99'])}, "
        f"sum={fmt_ms(wall['sum'])}"
    )
    print(
        "  extract/submit/wait ms: "
        f"extract_mean={fmt_ms(extract['mean'])}, "
        f"submit_mean={fmt_ms(submit['mean'])}, "
        f"wait_mean={fmt_ms(wait['mean'])}, wait_sum={fmt_ms(wait['sum'])}"
    )
    print(
        "  task submit/wait/elapsed ms: "
        f"submit_mean={fmt_ms(task_submit['mean'])}, "
        f"wait_mean={fmt_ms(task_wait['mean'])}, "
        f"elapsed_mean={fmt_ms(task_elapsed['mean'])}, "
        f"elapsed_p99={fmt_ms(task_elapsed['p99'])}"
    )
    if cache_max["count"]:
        print(
            "  rough overhead ms: "
            f"cache_max_elapsed_mean={fmt_ms(cache_max['mean'])}, "
            f"overhead_mean={fmt_ms(rough_overhead['mean'])}, "
            f"overhead_p50={fmt_ms(rough_overhead['p50'])}, "
            f"overhead_p90={fmt_ms(rough_overhead['p90'])}"
        )


def print_start_load_kv(summary: dict[str, object]) -> None:
    start_load = summary.get("start_load_kv")
    if not isinstance(start_load, dict):
        return
    if not start_load.get("profiles"):
        return

    wall = start_load["wall_ms"]
    print("start_load_kv:")
    print(
        f"  profiles: {start_load['profiles']}, "
        f"requests: {int(start_load['requests'])}, "
        f"load_requests: {int(start_load['load_requests'])}, "
        f"tasks: {int(start_load['tasks'])}, "
        f"errors: {start_load['errors']}"
    )
    print(
        "  wall_ms: "
        f"mean={fmt_ms(wall['mean'])}, p50={fmt_ms(wall['p50'])}, "
        f"p90={fmt_ms(wall['p90'])}, p99={fmt_ms(wall['p99'])}, "
        f"sum={fmt_ms(wall['sum'])}"
    )


def print_summary(summary: dict[str, object]) -> None:
    print(f"== {summary['label']} ==")
    print(f"logs: {', '.join(summary['paths'])}")
    print(f"mode: {summary['mode']}")
    if summary["mode"] != "ucm_transfer":
        lookup = summary["lookup"]
        print(
            "lookup: "
            f"records={lookup['records']}, "
            f"lookup_ms_mean={fmt_ms(lookup['lookup_ms']['mean'])}, "
            f"external_hit_blocks_mean={fmt_num(lookup['external_hit_blocks']['mean'])}"
        )
    print_operation("load", summary["load"])
    print_operation("dump/store", summary["store"])
    print_transfer_totals(summary)
    print_connector_load(summary)
    print_start_load_kv(summary)


def compare_metric(
    baseline: dict[str, object],
    candidate: dict[str, object],
    op: str,
    metric: str,
) -> None:
    base = baseline[op]["wall_ms"][metric]
    cand = candidate[op]["wall_ms"][metric]
    if math.isnan(base) or math.isnan(cand) or cand == 0:
        print(f"{op}.{metric}: n/a")
        return
    speedup = base / cand
    reduction = (base - cand) / base * 100 if base else math.nan
    print(
        f"{op}.{metric}: baseline={base:.3f} ms, "
        f"candidate={cand:.3f} ms, speedup={speedup:.3f}x, "
        f"reduction={reduction:.2f}%"
    )


def print_comparison(baseline: dict[str, object], candidate: dict[str, object]) -> None:
    print("== comparison ==")
    for op in ("load", "store"):
        for metric in ("mean", "p50", "p90", "sum"):
            compare_metric(baseline, candidate, op, metric)

    compare_connector_load(baseline, candidate)
    compare_start_load_kv(baseline, candidate)

    base_load = op_total_ms(baseline, "load")
    cand_load = op_total_ms(candidate, "load")
    base_dump = op_total_ms(baseline, "store")
    cand_dump = op_total_ms(candidate, "store")
    print_total_comparison("load_total_ms", base_load, cand_load)
    print_total_comparison("dump_total_ms", base_dump, cand_dump)
    print_total_comparison(
        "load_plus_dump_total_ms",
        base_load + base_dump,
        cand_load + cand_dump,
    )


def compare_connector_load(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> None:
    base_connector = baseline.get("connector_load")
    cand_connector = candidate.get("connector_load")
    if not isinstance(base_connector, dict) or not isinstance(cand_connector, dict):
        return
    if not base_connector.get("profiles") and not cand_connector.get("profiles"):
        return

    for stat_name, label in (
        ("wall_ms", "connector_load_wall"),
        ("extract_ms", "connector_extract"),
        ("submit_ms", "connector_submit"),
        ("wait_ms", "connector_wait"),
        ("rough_overhead_ms", "connector_rough_overhead"),
    ):
        for metric in ("mean", "p50", "p90", "sum"):
            base = base_connector[stat_name][metric]
            cand = cand_connector[stat_name][metric]
            if math.isnan(base) or math.isnan(cand) or cand == 0:
                print(f"{label}.{metric}: n/a")
                continue
            speedup = base / cand
            reduction = (base - cand) / base * 100 if base else math.nan
            print(
                f"{label}.{metric}: baseline={base:.3f} ms, "
                f"candidate={cand:.3f} ms, speedup={speedup:.3f}x, "
                f"reduction={reduction:.2f}%"
            )


def compare_start_load_kv(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> None:
    base_start = baseline.get("start_load_kv")
    cand_start = candidate.get("start_load_kv")
    if not isinstance(base_start, dict) or not isinstance(cand_start, dict):
        return
    if not base_start.get("profiles") and not cand_start.get("profiles"):
        return

    for metric in ("mean", "p50", "p90", "sum"):
        base = base_start["wall_ms"][metric]
        cand = cand_start["wall_ms"][metric]
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


def print_total_comparison(name: str, baseline_ms: float, candidate_ms: float) -> None:
    print(f"{name}:")
    if math.isnan(baseline_ms) or math.isnan(candidate_ms) or candidate_ms == 0:
        print("  n/a")
        return
    speedup = baseline_ms / candidate_ms
    reduction = (
        (baseline_ms - candidate_ms) / baseline_ms * 100 if baseline_ms else 0.0
    )
    print(
        f"  baseline={baseline_ms:.3f} ms, "
        f"candidate={candidate_ms:.3f} ms, "
        f"speedup={speedup:.3f}x, reduction={reduction:.2f}%"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze UCM transfer profile records from vLLM/UCM logs."
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
