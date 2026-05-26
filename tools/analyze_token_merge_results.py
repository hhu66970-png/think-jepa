#!/usr/bin/env python
import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Summarize ThinkJEPA token merge A/B/C results")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def extract_best_validation(path):
    text = Path(path).read_text(encoding="utf-8")
    patterns = {
        "best_epoch": r"`best_epoch`:\s*`([^`]+)`",
        "best_val_ADE": r"`best_val_avg_dist \(ADE\)`:\s*`([^`]+)`",
        "best_val_loss": r"`best_val_loss`:\s*`([^`]+)`",
        "best_val_pred_loss": r"`best_val_pred_loss`:\s*`([^`]+)`",
        "best_val_pred_latent_cosine_distance": (
            r"`best_val_pred_latent_cosine_distance`:\s*`([^`]+)`"
        ),
    }
    return {
        key: (float(match.group(1)) if match and key != "best_epoch" else int(match.group(1)))
        for key, pattern in patterns.items()
        for match in [re.search(pattern, text)]
        if match
    }


def summarize_encoder_dir(path):
    rows = read_csv(Path(path) / "full_pipeline_metrics.csv")
    baseline = [row for row in rows if row["config"] == "baseline"]
    baseline_latency = mean(to_float(row["latency_ms_mean"]) for row in baseline)
    baseline_memory = mean(
        to_float(row["peak_memory_mb_mean"])
        for row in baseline
        if row.get("peak_memory_mb_mean")
    )
    groups = {}
    for row in rows:
        if row["config"] == "baseline" or to_float(row.get("merge_ratio")) == 0.0:
            continue
        groups.setdefault(row["config"], []).append(row)

    out = []
    for config, items in groups.items():
        latency = mean(to_float(row["latency_ms_mean"]) for row in items)
        memory = mean(
            to_float(row["peak_memory_mb_mean"])
            for row in items
            if row.get("peak_memory_mb_mean")
        )
        out.append(
            {
                "scope": Path(path).name,
                "config": config,
                "method": items[0].get("method", ""),
                "strategy": items[0].get("strategy", ""),
                "importance_source": items[0].get("importance_source", ""),
                "protect_mode": items[0].get("protect_mode", ""),
                "merge_ratio": to_float(items[0].get("merge_ratio")),
                "actual_merge_ratio": mean(
                    to_float(row.get("actual_merge_ratio")) for row in items
                ),
                "tokens_after": mean(to_float(row["tokens_after"]) for row in items),
                "latency_ms_mean": latency,
                "speedup_vs_baseline": baseline_latency / latency if latency else 0.0,
                "peak_memory_mb_mean": memory,
                "memory_delta_mb": memory - baseline_memory if memory and baseline_memory else 0.0,
                "mean_cosine": mean(to_float(row["mean_cosine"]) for row in items),
                "p10_cosine": mean(to_float(row.get("p10_cosine")) for row in items),
                "p1_cosine": mean(to_float(row.get("p1_cosine")) for row in items),
                "relative_l2": mean(to_float(row.get("relative_l2")) for row in items),
                "any_fallback": any(str(row.get("any_fallback")) == "True" for row in items),
                "num_candidate_cells": mean(
                    to_float(row.get("num_candidate_cells")) for row in items
                ),
            }
        )
    return baseline_latency, baseline_memory, out


def write_csv(path, rows):
    fields = [
        "scope",
        "config",
        "method",
        "strategy",
        "importance_source",
        "protect_mode",
        "merge_ratio",
        "actual_merge_ratio",
        "tokens_after",
        "latency_ms_mean",
        "speedup_vs_baseline",
        "peak_memory_mb_mean",
        "memory_delta_mb",
        "mean_cosine",
        "p10_cosine",
        "p1_cosine",
        "relative_l2",
        "any_fallback",
        "num_candidate_cells",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_markdown(path, encoder_rows, downstream_rows):
    lines = [
        "# ThinkJEPA Token Merge A/B/C Result Summary",
        "",
        "## Claim Boundary",
        "",
        "- This is a full tiny-cache pipeline/stress evaluation, not a paper-scale benchmark.",
        "- Stage 2 importance diagnostics are not proof that importance improves downstream quality.",
        "- 512x512 results are encoder-only stress results, not downstream accuracy results.",
        "- `qk_global_hidden` is a global-hidden proxy, not a true attention map.",
        "- `dynamic_ratio_mode`, `score_delta`, and `debug_dump_scores` are metadata-only here.",
        "",
        "## Encoder Summary",
        "",
        "| scope | config | speedup | tokens | cosine | p10 | p1 | memory delta MiB | fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in encoder_rows:
        lines.append(
            "| {scope} | {config} | {speedup_vs_baseline:.3f}x | {tokens_after:.0f} | "
            "{mean_cosine:.4f} | {p10_cosine:.4f} | {p1_cosine:.4f} | "
            "{memory_delta_mb:.2f} | {any_fallback} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Downstream Small Compatibility",
            "",
            "| config | best epoch | ADE | val loss | pred loss | latent cosine distance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in downstream_rows:
        lines.append(
            "| {config} | {best_epoch} | {best_val_ADE:.4f} | {best_val_loss:.4f} | "
            "{best_val_pred_loss:.4f} | {best_val_pred_latent_cosine_distance:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Critical Read",
            "",
            "- A and B are nearly tied at the same actual merge ratio; current `norm_motion` protection is not strong evidence that importance is useful.",
            "- C is faster only because it merges the same amount, but its feature fidelity is worse than A/B at the same ratio.",
            "- The best conservative candidate remains `A r=0.05`; `B r=0.05` is comparable but not clearly better.",
            "- Downstream tiny-cache compatibility passes for all A/B/C r=0.05, but ADE is still worse than baseline on the one-sample validation split.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_rows = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and (item / "full_pipeline_metrics.csv").exists():
            _, _, rows = summarize_encoder_dir(item)
            encoder_rows.extend(rows)

    downstream_rows = []
    down_root = root / "downstream_small"
    if down_root.exists():
        for item in sorted(down_root.iterdir()):
            md = item / "test_results.md"
            if md.exists():
                row = {"config": item.name}
                row.update(extract_best_validation(md))
                downstream_rows.append(row)

    write_csv(out_dir / "pareto_table.csv", encoder_rows)
    best_by_cos = sorted(
        [row for row in encoder_rows if not row["any_fallback"]],
        key=lambda row: (row["scope"], -row["mean_cosine"], -row["speedup_vs_baseline"]),
    )
    best = {
        "best_encoder_rows_by_cosine": best_by_cos[:12],
        "downstream_small": downstream_rows,
    }
    (out_dir / "best_configs.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(out_dir / "summary_by_method.md", encoder_rows, downstream_rows)
    print(json.dumps({"encoder_rows": len(encoder_rows), "downstream_rows": len(downstream_rows)}, indent=2))


if __name__ == "__main__":
    main()
