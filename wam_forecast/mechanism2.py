"""Eighth review, round M2 (descriptive): how much each hand gate mixes hand cells with background. For each cached
merger at 131 tokens, on 200 part 2 training clips (the first shard, as mechanism.py): the share of hand cells (last
two frames mask) whose merged group also contains non hand cells (contamination), the mean size of groups that contain
a hand cell, the share of hand cells left unmerged, and hand and non hand fidelity.
  python mechanism2.py -> runs/phaseA/mechanism2.json"""
import glob
import json

import numpy as np

from common import FEAT_ROOT, W

ARMS = ["kbsmpa_L12", "handcurpa_L12", "handcurshufpa_L12", "handcurq20pa_L12", "handcurq20shufpa_L12",
        "handcurq10pa_L12", "handcurq10shufpa_L12", "handcurrecvpa_L12", "handcurrecvshufpa_L12",
        "handcurexclpa_L12", "handcurexclshufpa_L12", "handcursegpa_L12", "handcursegshufpa_L12", "handfutpa_L12", "handfutshufpa_L12"]
N = 200


def load_all(name):
    """Every shard of a config: row -> (tokens, idx, position)."""
    d = FEAT_ROOT / name
    where = {}
    for f in sorted(glob.glob(str(d / "tokens_*.npy"))):
        tag = f.split("tokens_")[1][:-4]
        tok = np.load(f, mmap_mode="r")
        idx = np.load(d / f"idx_{tag}.npy", mmap_mode="r") if (d / f"idx_{tag}.npy").exists() else None
        for j, r in enumerate(np.load(d / f"rows_{tag}.npy")):
            where[int(r)] = (tok, idx, j)
    return where


DENSE = load_all("dense")
ROWS = sorted(DENSE)[:: max(1, len(DENSE) // 1000)][:N * 5]          # a fixed clip set shared by every arm
H = np.load(FEAT_ROOT / "hand_masks.npz")["mask_cur"].reshape(-1, 4096).astype(bool)
out = {}
for a in ARMS:
    if not glob.glob(str(FEAT_ROOT / a / "tokens_*.npy")):
        continue
    A = load_all(a)
    st = dict(contamination=[], hand_group_size=[], hand_unmerged=[], fid_hand=[], fid_other=[])
    for r in ROWS:
        hm = H[r]
        if r not in A or not hm.any() or len(st["contamination"]) >= N:
            continue
        tok, idx, j = A[r]
        ix = np.asarray(idx[j]).astype(np.int64)
        K = tok.shape[1]
        sz = np.bincount(ix, minlength=K)
        nh = np.bincount(ix, weights=hm.astype(float), minlength=K)        # hand cells per group
        mixed = (nh > 0) & (nh < sz)                                       # group with hand and non hand cells
        st["contamination"].append(float(mixed[ix][hm].mean()))
        st["hand_group_size"].append(float(sz[ix][hm].mean()))
        st["hand_unmerged"].append(float((sz[ix][hm] == 1).mean()))
        rest = np.asarray(tok[j]).astype(np.float32)[ix]
        dt, _, dj = DENSE[r]
        den = np.asarray(dt[dj]).astype(np.float32)
        cos = (rest * den).sum(1) / (np.linalg.norm(rest, axis=1) * np.linalg.norm(den, axis=1) + 1e-6)
        st["fid_hand"].append(float(cos[hm].mean())); st["fid_other"].append(float(cos[~hm].mean()))
    out[a] = {k: float(np.mean(v)) for k, v in st.items()}
    out[a]["n"] = len(st["contamination"])
    print(f"{a:24s} " + "  ".join(f"{k} {v:.3f}" for k, v in out[a].items()), flush=True)
(W / "runs" / "phaseA" / "mechanism2.json").write_text(json.dumps(out, indent=1))
