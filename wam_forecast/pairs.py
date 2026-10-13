"""Paired comparison between arbitrary run directories (same seeds, same test clips).
  python pairs.py <root> A_dir:B_dir [...]     e.g. kbsm_L10g/vis_coords_raw:kbsm_L9/vis_coords
Prints n, mean difference (A - B, mm), seed-level 95 % t interval, clip x seed bootstrap, wins."""
import json
import sys
from pathlib import Path

import numpy as np

from aggregate import tcrit


def load(root, d):
    out = {}
    for mf in (Path(root) / d).glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        out[m["seed"]] = (m["ade_mm"], m["fde_mm"], np.load(mf.parent / "perclip_ade.npy"))
    return out


def main():
    root = sys.argv[1]
    for pr in sys.argv[2:]:
        a_d, b_d = pr.split(":")
        A, B = load(root, a_d), load(root, b_d)
        s = sorted(set(A) & set(B))
        if len(s) < 2:
            print(f"{pr}: not enough paired seeds ({len(s)})"); continue
        d = np.array([A[k][0] - B[k][0] for k in s])
        half = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
        D = np.stack([A[k][2] - B[k][2] for k in s])
        rng = np.random.default_rng(0)
        bo = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
        print(f"{a_d} - {b_d}: n={len(s)} A={np.mean([A[k][0] for k in s]):.2f} B={np.mean([B[k][0] for k in s]):.2f} "
              f"d={d.mean():+.3f} seedCI=[{d.mean()-half:+.3f},{d.mean()+half:+.3f}] "
              f"clipCI=[{np.percentile(bo,2.5):+.3f},{np.percentile(bo,97.5):+.3f}] wins={(d<0).sum()}/{len(s)} "
              f"FDE A={np.mean([A[k][1] for k in s]):.2f} B={np.mean([B[k][1] for k in s]):.2f}")


if __name__ == "__main__":
    main()
