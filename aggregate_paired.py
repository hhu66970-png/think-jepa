#!/usr/bin/env python3
"""Auditable paired statistics for matched WAM/K-BSM experiment directories."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev


def load_run(path: Path, mode: str, last_k: int) -> dict[str, float]:
    payload = json.loads(path.read_text())
    epochs = payload.get("epochs", [])
    if not epochs:
        raise ValueError(f"no epochs in {path}")

    if mode == "best":
        best_epoch = payload.get("best", {}).get("epoch")
        selected = [row for row in epochs if row.get("epoch") == best_epoch]
        if not selected:
            raise ValueError(f"best epoch {best_epoch!r} missing in {path}")
    elif mode == "final":
        selected = [epochs[-1]]
    else:
        selected = epochs[-last_k:]

    def avg(key: str) -> float:
        vals = [float(row[key]) for row in selected if row.get(key) is not None]
        return mean(vals) if vals else math.nan

    def temporal(key: str) -> float:
        vals = [
            float(row["temporal"][key])
            for row in selected
            if row.get("temporal") and row["temporal"].get(key) is not None
        ]
        return mean(vals) if vals else math.nan

    return {
        "ade": avg("val_avg_dist"),
        "fde": avg("val_final_dist"),
        "velocity": temporal("velocity_error"),
        "acceleration": temporal("accel_error"),
    }


def exact_signflip_pvalue(diffs: list[float]) -> float:
    observed = abs(mean(diffs))
    n = len(diffs)
    if n > 20:
        return math.nan
    extreme = 0
    total = 2**n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        value = abs(mean([s * d for s, d in zip(signs, diffs)]))
        if value >= observed - 1e-15:
            extreme += 1
    return extreme / total


def paired_summary(diffs: list[float]) -> dict[str, float | int]:
    n = len(diffs)
    delta = mean(diffs)
    sd = stdev(diffs) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    try:
        from scipy.stats import t as student_t

        crit = float(student_t.ppf(0.975, n - 1))
        t_stat = delta / se if se > 0 else math.nan
        p_t = float(2 * student_t.sf(abs(t_stat), n - 1)) if math.isfinite(t_stat) else math.nan
    except ImportError:
        t_stat = delta / se if se > 0 else math.nan
        try:
            import mpmath

            def two_sided_p(t_value: float) -> float:
                df = n - 1
                x = df / (df + t_value * t_value)
                return float(mpmath.betainc(df / 2, 0.5, 0, x, regularized=True))

            p_t = two_sided_p(abs(t_stat)) if math.isfinite(t_stat) else math.nan
            low, high = 0.0, 100.0
            for _ in range(100):
                mid = (low + high) / 2
                if two_sided_p(mid) > 0.05:
                    low = mid
                else:
                    high = mid
            crit = (low + high) / 2
        except ImportError:
            crit = 1.96
            p_t = math.nan
    return {
        "n": n,
        "mean_delta_right_minus_left": delta,
        "sd_delta": sd,
        "ci95_low": delta - crit * se,
        "ci95_high": delta + crit * se,
        "t_stat_two_sided": t_stat,
        "p_t_two_sided": p_t,
        "p_exact_signflip_two_sided": exact_signflip_pvalue(diffs),
        "cohen_dz": delta / sd if sd > 0 else math.nan,
        "right_better_count": sum(d < 0 for d in diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--left-root", type=Path, default=None)
    parser.add_argument("--right-root", type=Path, default=None)
    parser.add_argument("--left", default="KBSM_r015")
    parser.add_argument("--right", default="WAM_motion_r015")
    parser.add_argument("--seeds", default="42-53")
    parser.add_argument("--mode", choices=("best", "final", "plateau"), default="plateau")
    parser.add_argument("--last-k", type=int, default=10)
    args = parser.parse_args()

    left_root = args.left_root or args.root
    right_root = args.right_root or args.root

    lo, hi = (int(x) for x in args.seeds.split("-", 1))
    rows: list[dict[str, float | int]] = []
    missing: list[str] = []
    for seed in range(lo, hi + 1):
        left_path = left_root / f"{args.left}__s{seed}" / "metrics.json"
        right_path = right_root / f"{args.right}__s{seed}" / "metrics.json"
        if not left_path.exists() or not right_path.exists():
            missing.append(str(seed))
            continue
        left = load_run(left_path, args.mode, args.last_k)
        right = load_run(right_path, args.mode, args.last_k)
        row: dict[str, float | int] = {"seed": seed}
        for metric in ("ade", "fde", "velocity", "acceleration"):
            if not math.isfinite(left[metric]) or not math.isfinite(right[metric]):
                raise ValueError(
                    f"non-finite {metric} for seed {seed}: "
                    f"left={left[metric]!r}, right={right[metric]!r}"
                )
            row[f"left_{metric}"] = left[metric]
            row[f"right_{metric}"] = right[metric]
            row[f"delta_{metric}"] = right[metric] - left[metric]
        rows.append(row)

    if not rows:
        raise SystemExit("no complete seed pairs found")

    summaries = {
        metric: paired_summary([float(row[f"delta_{metric}"]) for row in rows])
        for metric in ("ade", "fde", "velocity", "acceleration")
    }
    report = {
        "root": str(args.root),
        "left_root": str(left_root),
        "right_root": str(right_root),
        "left": args.left,
        "right": args.right,
        "mode": args.mode,
        "last_k": args.last_k,
        "complete_seeds": [row["seed"] for row in rows],
        "missing_seeds": missing,
        "summaries": summaries,
    }

    pair_tag = f"{args.right}_minus_{args.left}".replace("/", "_")
    out_prefix = args.root / f"paired_{pair_tag}_{args.mode}"
    (out_prefix.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Paired {args.right} vs {args.left} ({args.mode})",
        "",
        f"Complete seed pairs: {len(rows)}; missing: {', '.join(missing) or 'none'}.",
        "",
        f"| Metric | Delta ({args.right}-{args.left}) | 95% CI | t | two-sided p | exact sign-flip p | Cohen dz | {args.right} better |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, summary in summaries.items():
        lines.append(
            "| {metric} | {delta:+.8f} | [{low:+.8f}, {high:+.8f}] | {t:+.3f} | "
            "{p:.5g} | {pex:.5g} | {dz:+.3f} | {better}/{n} |".format(
                metric=metric,
                delta=summary["mean_delta_right_minus_left"],
                low=summary["ci95_low"],
                high=summary["ci95_high"],
                t=summary["t_stat_two_sided"],
                p=summary["p_t_two_sided"],
                pex=summary["p_exact_signflip_two_sided"],
                dz=summary["cohen_dz"],
                better=summary["right_better_count"],
                n=summary["n"],
            )
        )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
