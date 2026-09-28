"""Round D report: per-clip budget policies vs fixed K-BSM L9 (paired seeds 0-19)."""
import json
from pathlib import Path

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT

R = Path("/21231_data1/huhaoming_wam/runs/test_main")
MS = {972: 66.0, 309: 61.7, 131: 55.4}          # batch-8 ms/clip, bench_vitl.json (K-BSM)


def load(d):
    out = {}
    for mf in (R / d / "vis").glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        out[m["seed"]] = (m["ade_mm"], m["fde_mm"], np.load(mf.parent / "perclip_ade.npy"))
    return out


def pair(a, b, name):
    s = sorted(set(a) & set(b))
    d = np.array([a[k][0] - b[k][0] for k in s])
    half = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    D = np.stack([a[k][2] - b[k][2] for k in s])
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    print(f"{name:>28s}: n={len(s)} d={d.mean():+.3f} seedCI=[{d.mean()-half:+.3f},{d.mean()+half:+.3f}] "
          f"clipCI=[{np.percentile(bo,2.5):+.3f},{np.percentile(bo,97.5):+.3f}] wins={(d<0).sum()}/{len(s)}")


def main():
    is_test = np.load(FEAT_ROOT / "poses.npz")["is_test"]
    base = load("kbsm_L9")
    runs = {p: load(f"mix_{p}") for p in ("motion", "inverse", "random")}
    K = np.array([972, 309, 131])
    for p, r in runs.items():
        a = np.load(FEAT_ROOT / f"policy_{p}.npz")["assign"][is_test]
        ms = np.mean([MS[k] for k in K[a]])
        print(f"{p:>8s}: seeds={len(r)} ADE={np.mean([v[0] for v in r.values()]):.2f} "
              f"FDE={np.mean([v[1] for v in r.values()]):.2f} mean tokens={K[a].mean():.1f} "
              f"est. encoder {ms:.1f} ms/clip (L9 {MS[309]})")
    print(f"  kbsm_L9: ADE={np.mean([base[k][0] for k in range(20) if k in base]):.2f} (seeds 0-19)")
    for p in runs:
        pair(runs[p], base, f"{p} - kbsm_L9")
    pair(runs["motion"], runs["random"], "motion - random")
    pair(runs["motion"], runs["inverse"], "motion - inverse")


if __name__ == "__main__":
    main()
