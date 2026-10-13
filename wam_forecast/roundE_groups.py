"""Round E, descriptive: group concentration on the 16^3 probe grid (each probe cell counted once) for V-JEPA 2 and
VideoMAE at the main budget, all cached clips. -> runs/phaseA/roundE_groups.json"""
import glob

import numpy as np

from common import FEAT_ROOT, W, write_json


def gini(x):
    x = np.sort(x); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


out = {}
for name in ("kbsm_L9", "kbsmpa_L9", "vmae_kbsm_L9", "vmae_kbsmpa_L9"):
    g, m = [], []
    for f in sorted(glob.glob(str(FEAT_ROOT / name / "idx_*.npy"))):
        for r in np.load(f, mmap_mode="r"):
            c = np.bincount(np.asarray(r, dtype=np.int64)); c = c[c > 0]
            g.append(gini(c)); m.append(c.max() / c.sum())
    out[name] = dict(n=len(g), gini=float(np.mean(g)), max_share=float(np.mean(m)))
    print(name, out[name])
write_json(out, W / "runs" / "phaseA" / "roundE_groups.json")
