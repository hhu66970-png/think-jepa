#!/usr/bin/env python
"""Add reviewer-facing consistency counts to protocol_metrics.json."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text())
    rows = data["mechanism_rows"]
    lambda_keys = ["wam_lam000", "wam_lam025", "wam_lam050",
                   "wam_lam075", "wam_lam100"]
    counts = {
        "wam1_gt_kbsm": sum(
            row["correlation"]["wam_lam100"] > row["correlation"]["kbsm"]
            for row in rows
        ),
        "wam1_gt_pitome": sum(
            row["correlation"]["wam_lam100"] > row["correlation"]["pitome"]
            for row in rows
        ),
        "kbsm_gt_pitome": sum(
            row["correlation"]["kbsm"] > row["correlation"]["pitome"]
            for row in rows
        ),
        "strict_lambda_monotone": sum(
            all(
                row["correlation"][left] < row["correlation"][right]
                for left, right in zip(lambda_keys, lambda_keys[1:])
            )
            for row in rows
        ),
        "lambda1_is_max": sum(
            row["correlation"]["wam_lam100"]
            == max(row["correlation"][key] for key in lambda_keys)
            for row in rows
        ),
        "n_clips": len(rows),
    }
    data["consistency_counts"] = counts
    metrics_path.write_text(json.dumps(data, indent=2) + "\n")

    summary_path = Path(args.summary)
    with summary_path.open("a") as stream:
        stream.write("\n## Consistency counts\n\n")
        stream.write(f"- WAM(lambda=1) > K-BSM: {counts['wam1_gt_kbsm']}/25 clips.\n")
        stream.write(f"- WAM(lambda=1) > PiToMe: {counts['wam1_gt_pitome']}/25 clips.\n")
        stream.write(f"- K-BSM > PiToMe: {counts['kbsm_gt_pitome']}/25 clips.\n")
        stream.write(f"- Strictly increasing allocation correlation over lambda: "
                     f"{counts['strict_lambda_monotone']}/25 clips.\n")
        stream.write(f"- Lambda=1 has the largest allocation correlation: "
                     f"{counts['lambda1_is_max']}/25 clips.\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
