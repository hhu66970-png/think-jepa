#!/usr/bin/env python3
"""Two-way clustered bootstrap over source-video groups and training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if payload.get("num_samples") != len(rows) or not rows:
        raise ValueError(f"invalid per-sample payload: {path}")
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate sample_id in {path}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--seeds", default="42-47")
    parser.add_argument("--bootstrap-seed", type=int, default=202718)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-", 1))
    seeds = list(range(lo, hi + 1))
    by_seed: dict[int, dict[str, dict[str, float | str]]] = {}
    canonical_ids: set[str] | None = None
    for seed in seeds:
        left = load_rows(args.root / f"{args.left}__s{seed}" / "per_sample_metrics.json")
        right = load_rows(args.root / f"{args.right}__s{seed}" / "per_sample_metrics.json")
        if set(left) != set(right):
            raise ValueError(f"left/right sample mismatch for seed {seed}")
        if canonical_ids is None:
            canonical_ids = set(left)
        elif set(left) != canonical_ids:
            raise ValueError(f"sample set differs across seeds at seed {seed}")
        paired = {}
        for sample_id in sorted(left):
            values = {}
            for metric in ("ade", "fde"):
                left_value = float(left[sample_id][metric])
                right_value = float(right[sample_id][metric])
                if not math.isfinite(left_value) or not math.isfinite(right_value):
                    raise ValueError(
                        f"non-finite {metric} for seed={seed}, sample={sample_id}: "
                        f"left={left_value!r}, right={right_value!r}"
                    )
                values[metric] = right_value - left_value
            paired[sample_id] = {
                "video_group": str(left[sample_id]["video_group"]),
                **values,
            }
        by_seed[seed] = paired

    assert canonical_ids is not None
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(canonical_ids):
        group = str(by_seed[seeds[0]][sample_id]["video_group"])
        if any(str(by_seed[seed][sample_id]["video_group"]) != group for seed in seeds):
            raise ValueError(f"video_group differs across seeds for {sample_id}")
        groups[group].append(sample_id)
    group_names = sorted(groups)

    raw_rows = []
    for seed in seeds:
        for sample_id in sorted(canonical_ids):
            raw_rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    **by_seed[seed][sample_id],
                }
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap: dict[str, list[float]] = {"ade": [], "fde": []}
    for _ in range(args.replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_ids = [
            sample_id for group in sampled_groups for sample_id in groups[str(group)]
        ]
        for metric in bootstrap:
            values = [
                float(by_seed[int(seed)][sample_id][metric])
                for seed in sampled_seeds
                for sample_id in sampled_ids
            ]
            bootstrap[metric].append(float(np.mean(values)))

    summaries = {}
    for metric in ("ade", "fde"):
        observed = np.array([float(row[metric]) for row in raw_rows], dtype=np.float64)
        draws = np.array(bootstrap[metric], dtype=np.float64)
        summaries[metric] = {
            "mean_delta_right_minus_left": float(observed.mean()),
            "cluster_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
            "cluster_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
            "right_better_clip_seed_count": int((observed < 0).sum()),
            "clip_seed_count": int(observed.size),
        }

    report = {
        "left": args.left,
        "right": args.right,
        "seeds": seeds,
        "num_samples": len(canonical_ids),
        "num_video_groups": len(group_names),
        "bootstrap_seed": args.bootstrap_seed,
        "replicates": args.replicates,
        "resampling": "training seeds and source-video groups with replacement",
        "summaries": summaries,
    }
    tag = f"clustered_{args.right}_minus_{args.left}"
    (args.root / f"{tag}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.root / f"{tag}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
