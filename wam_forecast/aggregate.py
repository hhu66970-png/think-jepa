"""Aggregate probe runs: per-config mean/SD over seeds and paired comparisons.

  python aggregate.py <exp_dir> [--pairs wam_L9:kbsm_L9,wam_s25:kbsm_s25] [--input vis]

Paired statistics use the SAME seeds and the SAME test clips for both arms:
  * seed level: d_s = ADE_a(s) - ADE_b(s); mean, SD, 95% t-interval, sign count;
  * clip level: hierarchical bootstrap (resample seeds, then clips) of the mean difference.
Negative = arm a is better (lower error).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045}


def tcrit(df):
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def load(exp, inp):
    runs = defaultdict(dict)                       # cfg -> seed -> (metrics, perclip)
    for mf in Path(exp).glob(f"*/{inp}/s*/metrics.json"):
        m = json.loads(mf.read_text())
        pc = np.load(mf.parent / "perclip_ade.npy")
        runs[m["cfg"]][m["seed"]] = (m, pc)
    return runs


def summarize(runs):
    rows = []
    for cfg, seeds in sorted(runs.items()):
        ms = [v[0] for v in seeds.values()]
        f = lambda k: np.array([m[k] for m in ms])
        rows.append(dict(cfg=cfg, K=ms[0]["K"], n=len(ms),
                         ade=f("ade_mm").mean(), ade_sd=f("ade_mm").std(ddof=1) if len(ms) > 1 else 0.0,
                         fde=f("fde_mm").mean(), wrist=f("wrist_ade_mm").mean()))
    return rows


def paired(runs, a, b, n_boot=4000, rng=np.random.default_rng(0)):
    seeds = sorted(set(runs[a]) & set(runs[b]))
    d_seed = np.array([runs[a][s][0]["ade_mm"] - runs[b][s][0]["ade_mm"] for s in seeds])
    D = np.stack([runs[a][s][1] - runs[b][s][1] for s in seeds])        # [S, clips]
    n = len(seeds)
    mean = d_seed.mean()
    sd = d_seed.std(ddof=1) if n > 1 else float("nan")
    half = tcrit(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, n, n)
        ci = rng.integers(0, D.shape[1], D.shape[1])
        boots[i] = D[si][:, ci].mean()
    return dict(a=a, b=b, seeds=n, mean=mean, sd=sd, t_lo=mean - half, t_hi=mean + half,
                wins=int((d_seed < 0).sum()), boot_lo=float(np.percentile(boots, 2.5)),
                boot_hi=float(np.percentile(boots, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--input", default="vis")
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    runs = load(a.exp, a.input)
    lines = [f"# {a.exp}  input={a.input}", "", "| cfg | K | seeds | ADE mm | SD | FDE mm | wrist ADE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summarize(runs):
        lines.append(f"| {r['cfg']} | {r['K']} | {r['n']} | {r['ade']:.2f} | {r['ade_sd']:.2f} | "
                     f"{r['fde']:.2f} | {r['wrist']:.2f} |")
    out = {"summary": summarize(runs), "pairs": []}
    if a.pairs:
        lines += ["", "| a − b | seeds | mean Δ mm | seed SD | 95% t-CI | a wins | bootstrap 95% CI |",
                  "|---|---:|---:|---:|---|---:|---|"]
        for pr in a.pairs.split(","):
            x, y = pr.split(":")
            if x in runs and y in runs:
                p = paired(runs, x, y)
                out["pairs"].append(p)
                lines.append(f"| {x} − {y} | {p['seeds']} | {p['mean']:+.3f} | {p['sd']:.3f} | "
                             f"[{p['t_lo']:+.3f}, {p['t_hi']:+.3f}] | {p['wins']}/{p['seeds']} | "
                             f"[{p['boot_lo']:+.3f}, {p['boot_hi']:+.3f}] |")
    txt = "\n".join(lines)
    print(txt)
    Path(a.exp, f"summary_{a.input}.md").write_text(txt + "\n")
    Path(a.exp, f"summary_{a.input}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
