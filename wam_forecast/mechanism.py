"""Seventh review, descriptive: what does each hand gate do to the merge? For each cached merger at 131 tokens, on 200
part 2 training clips: Gini and maximum of the group sizes, the token weighted group size, the share of hand cells
(last two frames mask) left unmerged, and the fidelity (cosine of the restored token to the dense token) of hand and of
non hand cells.  python mechanism.py -> runs/phaseA/mechanism.json"""
import glob
import json

import numpy as np

from common import FEAT_ROOT, W

ARMS = ["kbsmpa_L12", "handcurpa_L12", "handcurshufpa_L12", "handcurq20pa_L12", "handcurq20shufpa_L12", "handcurq10pa_L12",
        "handcurq10shufpa_L12", "handcurq20cap2pa_L12", "handcurq20cap2shufpa_L12", "handcurcap2pa_L12", "caps2pa_L12",
        "handf31pa_L12", "handfutpa_L12", "handfutshufpa_L12", "pospa_L12", "wampa_L12"]
N = 200


def load(name):
    d = FEAT_ROOT / name
    f = sorted(glob.glob(str(d / "tokens_*.npy")))[0]
    tag = f.split("tokens_")[1][:-4]
    tok = np.load(f, mmap_mode="r")
    idx = np.load(d / f"idx_{tag}.npy", mmap_mode="r") if (d / f"idx_{tag}.npy").exists() else None
    return np.load(d / f"rows_{tag}.npy"), tok, idx


rows_d, tok_d, _ = load("dense")
H = np.load(FEAT_ROOT / "hand_masks.npz")["mask_cur"].reshape(-1, 4096)
pos_d = {int(r): i for i, r in enumerate(rows_d)}
out = {}
for a in ARMS:
    if not glob.glob(str(FEAT_ROOT / a / "tokens_*.npy")):
        continue
    rows, tok, idx = load(a)
    st = dict(gini=[], max=[], tokw=[], hand_unmerged=[], fid_hand=[], fid_other=[])
    for j, r in enumerate(rows[:N]):
        r = int(r)
        if r not in pos_d:
            continue
        ix = np.asarray(idx[j]).astype(np.int64)
        sz = np.bincount(ix, minlength=tok.shape[1]).astype(float)
        s = np.sort(sz[sz > 0]); n = len(s)
        st["gini"].append(float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())))
        st["max"].append(float(s.max())); st["tokw"].append(float(sz[ix].mean()))
        hm = H[r].astype(bool)
        st["hand_unmerged"].append(float((sz[ix][hm] == 1).mean()) if hm.any() else np.nan)
        rest = np.asarray(tok[j]).astype(np.float32)[ix]                 # [4096, D]
        den = np.asarray(tok_d[pos_d[r]]).astype(np.float32)
        cos = (rest * den).sum(1) / (np.linalg.norm(rest, axis=1) * np.linalg.norm(den, axis=1) + 1e-6)
        st["fid_hand"].append(float(cos[hm].mean()) if hm.any() else np.nan)
        st["fid_other"].append(float(cos[~hm].mean()))
    out[a] = {k: float(np.nanmean(v)) for k, v in st.items()}
    print(f"{a:24s} " + "  ".join(f"{k} {v:.3f}" for k, v in out[a].items()), flush=True)
(W / "runs" / "phaseA" / "mechanism.json").write_text(json.dumps(out, indent=1))
