"""Difference-in-differences for the readout experiment C.

gap(readout) = ADE(kbsm) - ADE(dense) per seed; DiD = gap(restore) - gap(new) per seed.
Positive DiD = the new readout closes the compression gap. Reports the seed-level t interval
and a seed x clip bootstrap. Usage: did.py <run_root> <budget cfg> <new readout tag>
"""
import json
import sys
from pathlib import Path

import numpy as np

from aggregate import tcrit


def load(root, cfg, tag):
    out = {}
    for mf in Path(root).glob(f"{cfg}/{tag}/s*/metrics.json"):
        m = json.loads(mf.read_text())
        out[m["seed"]] = (m["ade_mm"], np.load(mf.parent / "perclip_ade.npy"))
    return out


def main():
    root, cfg, new = sys.argv[1], sys.argv[2], sys.argv[3]
    old_c = load(root, cfg, "vis")
    old_d = load(root, "dense", "vis")
    new_c = load(root, cfg, new)
    new_d = load(root, "dense", "vis_compact") or old_d
    seeds = sorted(set(old_c) & set(old_d) & set(new_c) & set(new_d))
    did = np.array([(old_c[s][0] - old_d[s][0]) - (new_c[s][0] - new_d[s][0]) for s in seeds])
    gap_old = np.array([old_c[s][0] - old_d[s][0] for s in seeds])
    gap_new = np.array([new_c[s][0] - new_d[s][0] for s in seeds])
    n = len(seeds)
    half = tcrit(n - 1) * did.std(ddof=1) / np.sqrt(n)
    D = np.stack([(old_c[s][1] - old_d[s][1]) - (new_c[s][1] - new_d[s][1]) for s in seeds])
    rng = np.random.default_rng(0)
    boots = [D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    print(f"{cfg} {new}: seeds={n} gap_restore={gap_old.mean():.3f} gap_new={gap_new.mean():.3f} "
          f"DiD={did.mean():+.3f} seedCI=[{did.mean()-half:+.3f},{did.mean()+half:+.3f}] "
          f"bootCI=[{np.percentile(boots,2.5):+.3f},{np.percentile(boots,97.5):+.3f}] "
          f"new-arm ADE={np.mean([new_c[s][0] for s in seeds]):.2f}")


if __name__ == "__main__":
    main()
