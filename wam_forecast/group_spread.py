"""Descriptive (sixth review): how far apart are the tokens a merger groups? For each cached merger, the mean distance (in
grid cells, over t, h, w) of an original cell to the centre of its merged group, on the first 150 clips of the first
shard.  python group_spread.py -> runs/phaseA/group_spread.json"""
import glob
import json

import numpy as np

from common import FEAT_ROOT, W

ARMS = ["kbsmpa_L9", "pospa_L9", "randmpa_L9", "wampa_L9", "wamshufpa_L9", "wamshuffpa_L9", "wamlam05pa_L9", "wamlam025pa_L9",
        "handmultpa_L9", "caps2pa_L9", "handcurpa_L9", "handcurshuffpa_L9", "vidtldrpa_L9", "fmpa_L9", "kbsmpa_L12", "pospa_L12",
        "randmpa_L12", "wampa_L12", "handmultpa_L12", "handcurpa_L12", "handcurshuffpa_L12", "handcurshufpa_L12", "handf31pa_L12",
        "handf31shuffpa_L12", "handf31shufpa_L12"]
o = np.arange(4096)
C = np.stack([o // 256, (o % 256) // 16, o % 16], -1).astype(float)
out = {}
for a in ARMS:
    fs = sorted(glob.glob(str(FEAT_ROOT / a / "idx_*.npy")))
    if not fs:
        continue
    sp = []
    for r in np.load(fs[0], mmap_mode="r")[:150]:
        r = np.asarray(r); K = r.max() + 1
        cnt = np.bincount(r, minlength=K).astype(float)
        cen = np.stack([np.bincount(r, weights=C[:, d], minlength=K) for d in range(3)], -1) / cnt[:, None]
        sp.append(np.sqrt(((C - cen[r]) ** 2).sum(1)).mean())
    out[a] = float(np.mean(sp))
    print(f"{a:20s} {out[a]:.2f} cells")
(W / "runs" / "phaseA" / "group_spread.json").write_text(json.dumps(out, indent=1))
