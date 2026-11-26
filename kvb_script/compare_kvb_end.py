"""Compare artifacts collected by kvb_end_audit_connector across two builds."""

import argparse
import re
from pathlib import Path

import torch


def read_attempt(directory, phase, attempt):
    pattern = f"{phase}-{attempt or '*'}-tp*.pt"
    found = sorted(directory.glob(pattern))
    if not found:
        raise ValueError(f"No files match {directory / pattern}")
    rows = {}
    identity = None
    for path in found:
        row = torch.load(path, map_location="cpu", weights_only=True)
        current = (row["attempt"], row["key"], row["tokens"], row["tp_size"])
        rank = row["tp_rank"]
        if identity is not None and current != identity:
            raise ValueError(f"Multiple snapshot batches in {directory}; select an attempt ID")
        if row["phase"] != phase or (attempt is not None and row["attempt"] != attempt) or rank in rows:
            raise ValueError(f"Unexpected or duplicate TP rank file: {path}")
        identity = current
        rows[rank] = row
    if set(rows) != set(range(identity[-1])):
        raise ValueError(f"Incomplete TP ranks in {directory}: {sorted(rows)}")
    return identity, rows


def layer_order(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return (int(match.group(1)) if match else -1, name)


def compare(left, right):
    summary = {"ssm_entries": 0, "fa_pages": 0, "first_gdn_mismatch": None, "first_fa_mismatch": None,
               "ssm_mismatches": [], "fa_mismatch_count": 0}
    for rank, a in sorted(left.items()):
        b = right[rank]
        for kind in ("ssm", "fa"):
            if a[kind].keys() != b[kind].keys():
                raise ValueError(f"Different {kind} layer lists on TP rank {rank}")
        for name, pair in a["ssm"].items():
            other = b["ssm"][name]
            if len(pair) != 2 or len(other) != 2:
                raise ValueError(f"Expected conv and SSM components for {name}")
            for component, x, y in zip(("conv", "ssm"), pair, other):
                summary["ssm_entries"] += 1
                if x.shape != y.shape or x.dtype != y.dtype:
                    raise ValueError(f"Different {name}/{component} shapes or dtypes on TP rank {rank}")
                if torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
                    continue
                x, y = x.double(), y.double()
                delta = x - y
                summary["ssm_mismatches"].append((rank, name, component, delta.abs().max().item(),
                                                  (delta.norm() / x.norm().clamp_min(1e-12)).item()))
        for name in a["fa"]:
            pages, other = a["fa"][name], b["fa"][name]
            if len(pages) != len(other):
                raise ValueError(f"Different {name} FA page counts on TP rank {rank}")
            for block, (pair, other_pair) in enumerate(zip(pages, other)):
                summary["fa_pages"] += 1
                for component, x, y in zip(("K", "V"), pair, other_pair):
                    if x[:2] != y[:2]:
                        raise ValueError(f"Different {name}/{component} FA layout on TP rank {rank}")
                    if x != y:
                        summary["fa_mismatch_count"] += 1
                        entry = (rank, name, block, component)
                        first = summary["first_fa_mismatch"]
                        if first is None or (layer_order(name), block, rank) < (layer_order(first[1]), first[2], first[0]):
                            summary["first_fa_mismatch"] = entry
    diffs = summary["ssm_mismatches"]
    if diffs:
        summary["first_gdn_mismatch"] = min(diffs, key=lambda row: (layer_order(row[1]), row[0], row[2]))
    return summary


def report(label, summary, top):
    diffs = summary["ssm_mismatches"]
    print(f"\n{label}")
    print(f"GDN components differing byte for byte: {len(diffs)}/{summary['ssm_entries']}")
    if diffs:
        rank, name, component, max_abs, rel_l2 = summary["first_gdn_mismatch"]
        print(f"First changed GDN layer: tp={rank} {name}/{component} max_abs={max_abs:.6g} rel_l2={rel_l2:.6g}")
        for rank, name, component, max_abs, rel_l2 in sorted(diffs, key=lambda row: row[4], reverse=True)[:top]:
            print(f"  tp={rank} {name}/{component} max_abs={max_abs:.6g} rel_l2={rel_l2:.6g}")
    print(f"FA K/V components differing byte for byte: {summary['fa_mismatch_count']}/{2 * summary['fa_pages']}")
    if summary["first_fa_mismatch"] is not None:
        rank, name, block, component = summary["first_fa_mismatch"]
        print(f"First changed FA page: tp={rank} {name}/{component} logical_block={block}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, help="Optional old-build capture directory")
    parser.add_argument("--new", required=True, type=Path, help="New-build save and load capture directory")
    parser.add_argument("--old-attempt", help="Old save batch ID, if the directory has several")
    parser.add_argument("--new-attempt", help="New save batch ID, if the directory has several")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    try:
        new_id, new = read_attempt(args.new, "save", args.new_attempt)
        loaded_id, loaded = read_attempt(args.new, "load", new_id[0])
        if new_id != loaded_id:
            raise ValueError("Save and load snapshot identities differ")
        print(f"New snapshot: END tokens={new_id[2]} TP={new_id[3]} key={new_id[1]} attempt={new_id[0]}")
        round_trip = compare(new, loaded)
        report("New END capture vs restored state", round_trip, args.top)
        if args.old is not None:
            old_id, old = read_attempt(args.old, "save", args.old_attempt)
            if old_id[2:] != new_id[2:]:
                raise ValueError("Old and new captures have different END lengths or TP sizes")
            report("Old END capture vs new END capture", compare(old, new), args.top)
        return int(bool(round_trip["ssm_mismatches"] or round_trip["fa_mismatch_count"]))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
