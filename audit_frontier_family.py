#!/usr/bin/env python3
"""Family-wise audit of the established WAM/K-BSM compression frontier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aggregate_paired import load_run, paired_summary


def holm_adjust(items: list[tuple[str, float]]) -> dict[str, float]:
    finite = [(key, value) for key, value in items if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (key, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--r3637-root", type=Path)
    parser.add_argument("--mode", choices=("best", "final", "plateau"), default="plateau")
    parser.add_argument("--last-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    points = [
        ("2686", args.root, "KBSM_r20", "WAM_r20"),
        ("1944", args.root, "KBSM_r25", "WAM_r25"),
        ("616", args.root, "KBSM_L9_r25", "WAM_L9_r25"),
        ("261", args.root, "KBSM_L12_r25", "WAM_L12_r25"),
    ]
    if args.r3637_root:
        points.insert(0, ("3637", args.r3637_root, "KBSM_r015", "WAM_motion_r015"))
    metrics = ("ade", "fde", "velocity", "acceleration")
    records: list[dict[str, object]] = []
    for tokens, point_root, left, right in points:
        rows = []
        for seed in range(42, 54):
            left_path = point_root / f"{left}__s{seed}" / "metrics.json"
            right_path = point_root / f"{right}__s{seed}" / "metrics.json"
            if not left_path.exists() or not right_path.exists():
                continue
            rows.append(
                (
                    seed,
                    load_run(left_path, args.mode, args.last_k),
                    load_run(right_path, args.mode, args.last_k),
                )
            )
        for metric in metrics:
            if not rows:
                continue
            diffs = [right_values[metric] - left_values[metric] for _, left_values, right_values in rows]
            summary = paired_summary(diffs)
            records.append(
                {
                    "tokens": tokens,
                    "metric": metric,
                    "left": left,
                    "right": right,
                    **summary,
                }
            )

    primary_keys = [(f"{r['tokens']}:{r['metric']}", float(r["p_t_two_sided"])) for r in records if r["metric"] == "ade"]
    kinematic_keys = [
        (f"{r['tokens']}:{r['metric']}", float(r["p_t_two_sided"]))
        for r in records
        if r["metric"] in ("velocity", "acceleration")
    ]
    all_keys = [(f"{r['tokens']}:{r['metric']}", float(r["p_t_two_sided"])) for r in records]
    primary_holm = holm_adjust(primary_keys)
    kinematic_holm = holm_adjust(kinematic_keys)
    exploratory_holm = holm_adjust(all_keys)
    for record in records:
        key = f"{record['tokens']}:{record['metric']}"
        record["holm_primary_ade"] = primary_holm.get(key)
        record["holm_kinematic"] = kinematic_holm.get(key)
        record["holm_all_metrics"] = exploratory_holm.get(key)

    report = {
        "mode": args.mode,
        "last_k": args.last_k,
        "families": {
            "primary_ade": f"{len(points)} planned operating-point ADE comparisons",
            "kinematic": f"velocity and acceleration over {len(points)} operating points",
            "all_metrics": f"exploratory correction over all four metrics and {len(points)} points",
        },
        "records": records,
    }
    output = args.output or (args.root / f"frontier_family_audit_{args.mode}.json")
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("| tokens | metric | delta | p | Holm family p | Holm all-metrics p | WAM better |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for record in records:
        family_p = (
            record["holm_primary_ade"]
            if record["metric"] == "ade"
            else record["holm_kinematic"]
        )
        print(
            "| {tokens} | {metric} | {delta:+.8f} | {p:.5g} | {family} | {allp:.5g} | {better}/{n} |".format(
                tokens=record["tokens"],
                metric=record["metric"],
                delta=record["mean_delta_wam_minus_kbsm"],
                p=record["p_t_two_sided"],
                family=f"{family_p:.5g}" if family_p is not None else "--",
                allp=record["holm_all_metrics"],
                better=record["wam_better_count"],
                n=record["n"],
            )
        )


if __name__ == "__main__":
    main()
