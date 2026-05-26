#!/usr/bin/env python
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Analyze token-merge decision overlap between methods.")
    parser.add_argument("--decision_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--baseline_prefix", type=str, default="A_")
    return parser.parse_args()


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No decision rows found in {path}")
    return rows


def decision_key(row):
    return (
        int(row.get("sample", -1)),
        int(row.get("repeat", -1)),
        int(row.get("layer", -1)),
        int(row.get("cell_id", -1)),
    )


def undirected_pair(row):
    src = int(row["source_token_id"])
    dst = int(row["receiver_token_id"])
    return tuple(sorted((src, dst)))


def directed_pair(row):
    return int(row["source_token_id"]), int(row["receiver_token_id"])


def config_groups(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[str(row["config"])][decision_key(row)] = row
    return groups


def group_ratio(rows):
    ratios = {round(float(row.get("merge_ratio", -1.0)), 8) for row in rows.values()}
    if len(ratios) != 1:
        raise ValueError(f"Expected one merge_ratio per config, got {sorted(ratios)}")
    return next(iter(ratios))


def baseline_configs_by_ratio(groups, baseline_prefix):
    baselines = {}
    for name, rows in groups.items():
        if name.startswith(baseline_prefix):
            baselines[group_ratio(rows)] = name
    if not baselines:
        raise ValueError(f"No baseline config starts with {baseline_prefix!r}")
    return baselines


def compare_configs(base_name, other_name, base_rows, other_rows):
    base_keys = set(base_rows)
    other_keys = set(other_rows)
    shared_keys = sorted(base_keys & other_keys)
    if not shared_keys:
        raise ValueError(f"No shared decision keys for {base_name} vs {other_name}")

    undirected_same = 0
    directed_same = 0
    selected_similarity_drop = []
    best_similarity_gap = []
    other_lower_than_base = 0
    source_importance_delta = []
    receiver_importance_delta = []
    for key in shared_keys:
        base = base_rows[key]
        other = other_rows[key]
        undirected_same += int(undirected_pair(base) == undirected_pair(other))
        directed_same += int(directed_pair(base) == directed_pair(other))
        base_sim = float(base.get("selected_similarity") or 0.0)
        other_sim = float(other.get("selected_similarity") or 0.0)
        drop = base_sim - other_sim
        selected_similarity_drop.append(drop)
        other_lower_than_base += int(drop > 1e-8)
        best_similarity_gap.append(
            float(other.get("best_similarity") or 0.0) - other_sim
        )
        if base.get("source_importance") is not None and other.get("source_importance") is not None:
            source_importance_delta.append(
                float(other["source_importance"]) - float(base["source_importance"])
            )
        if base.get("receiver_importance") is not None and other.get("receiver_importance") is not None:
            receiver_importance_delta.append(
                float(other["receiver_importance"]) - float(base["receiver_importance"])
            )

    count = len(shared_keys)
    mean_drop = sum(selected_similarity_drop) / count
    p10_drop = sorted(selected_similarity_drop)[max(0, int(count * 0.10) - 1)]
    mean_best_gap = sum(best_similarity_gap) / count
    return {
        "baseline_config": base_name,
        "config": other_name,
        "baseline_decisions": len(base_rows),
        "config_decisions": len(other_rows),
        "shared_cells": count,
        "cell_overlap": count / max(1, len(base_rows)),
        "undirected_pair_overlap": undirected_same / count,
        "directed_pair_overlap": directed_same / count,
        "mean_selected_similarity_drop_vs_baseline": mean_drop,
        "p10_selected_similarity_drop_vs_baseline": p10_drop,
        "mean_best_similarity_gap": mean_best_gap,
        "fraction_lower_similarity_than_baseline": other_lower_than_base / count,
        "mean_source_importance_delta_vs_baseline": (
            sum(source_importance_delta) / len(source_importance_delta)
            if source_importance_delta
            else ""
        ),
        "mean_receiver_importance_delta_vs_baseline": (
            sum(receiver_importance_delta) / len(receiver_importance_delta)
            if receiver_importance_delta
            else ""
        ),
    }


def write_csv(rows, path):
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, path):
    lines = ["# Merge Pair Overlap Analysis", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['config']} vs {row['baseline_config']}",
                "",
                f"- shared cells: {row['shared_cells']}",
                f"- cell overlap: {row['cell_overlap']:.4f}",
                f"- undirected pair overlap: {row['undirected_pair_overlap']:.4f}",
                f"- directed pair overlap: {row['directed_pair_overlap']:.4f}",
                f"- mean selected-similarity drop vs baseline: {row['mean_selected_similarity_drop_vs_baseline']:.6f}",
                f"- p10 selected-similarity drop vs baseline: {row['p10_selected_similarity_drop_vs_baseline']:.6f}",
                f"- mean best-similarity gap inside selected cells: {row['mean_best_similarity_gap']:.6f}",
                f"- lower-similarity fraction vs baseline: {row['fraction_lower_similarity_than_baseline']:.4f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = load_rows(args.decision_jsonl)
    groups = config_groups(rows)
    baselines = baseline_configs_by_ratio(groups, args.baseline_prefix)
    comparisons = []
    for name in sorted(groups):
        if name == "baseline":
            continue
        ratio = group_ratio(groups[name])
        baseline = baselines.get(ratio)
        if name == baseline:
            continue
        if baseline is None:
            raise ValueError(
                f"No baseline config with merge_ratio={ratio} for comparison against {name}"
            )
        comparisons.append(compare_configs(baseline, name, groups[baseline], groups[name]))
    if not comparisons:
        raise ValueError("Need at least one non-baseline config in decision dump")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparisons, out_dir / "pair_overlap.csv")
    write_summary(comparisons, out_dir / "summary.md")
    print(json.dumps({"baselines": baselines, "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
