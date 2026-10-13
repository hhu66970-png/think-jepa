"""Rebuild the primary contrasts of rounds DV2, Q, and M2 (eighth review) from the released raw archive alone and compare
them with roundDV2.json, roundQ.json, and roundM2.json.   python rebuild_eighth.py --data ../../data"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
RAW = np.load(DATA / "raw_results.npz")
KEYS = set(RAW.files)


def boot(D):
    n = D.shape[0]; rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    return float(D.mean()), [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]


def official(cfg, metric):
    out = {}
    for s in range(10):
        k = f"runs_DV2|off_cls|{cfg}|vis_official|s{s}"
        if f"{k}|{metric}" in KEYS:
            r = RAW[f"{k}|rows"].astype(np.int64); o = np.argsort(r); out[s] = RAW[f"{k}|{metric}"][o].astype(np.float64)
    return out


def cv(prefix, metric):
    out = {}
    for s in range(10):
        parts = []
        for f in range(5):
            k = f"{prefix}_cv{f}|s{s}"
            km = k if metric == "ade" else f"{k}|{metric}"
            if km not in KEYS:
                break
            parts.append((RAW[f"{k}|rows"].astype(np.int64), RAW[km].astype(np.float64)))
        if len(parts) == 5:
            r = np.concatenate([p[0] for p in parts]); x = np.concatenate([p[1] for p in parts]); out[s] = x[np.argsort(r)]
    return out


def pair(A, B, scale=1.0):
    s = sorted(set(A) & set(B))
    return boot(np.stack([(A[k] - B[k]) * scale for k in s]))


checks = []
D2 = json.loads((DATA / "roundDV2.json").read_text())
for name, a, b in [("DV2_1_motion_minus_shuffle_s25", "wampa_s25", "wamshufpa_s25"),
                   ("DV2_2_motion_minus_shuffle_in16f_L9", "in16f+wampa_L9", "in16f+wamshufpa_L9"),
                   ("DV2_3_motion_minus_shuffle_L9", "wampa_L9", "wamshufpa_L9"),
                   ("flowc_minus_shuffle_L9", "flowcpa_L9", "flowcshufpa_L9"),
                   ("kbsmpa_L9_minus_dense", "kbsmpa_L9", "dense")]:
    d, c = pair(official(a, "acc"), official(b, "acc"), 100.0)
    checks.append((name, d, c, D2[name]["acc"]["d"], D2[name]["acc"]["clip_ci"]))
Q = json.loads((DATA / "roundQ.json").read_text())
for cfg in ("kbsmpa_L9", "wampa_L9"):
    d, c = pair(cv(f"runs_Q|cv_cls|{cfg}|vis", "acc"), cv(f"runs_DV|cv_cls|{cfg}|vis", "acc"), 100.0)
    checks.append((f"Q {cfg}", d, c, Q[f"{cfg}_acc"]["d"], Q[f"{cfg}_acc"]["clip_ci"]))
M2 = json.loads((DATA / "roundM2.json").read_text())
d, c = pair(cv("runs|cv|handcurexclpa_L12|vispose_k1", "ade"), cv("runs|cv|handcurexclshufpa_L12|vispose_k1", "ade"))
checks.append(("M2_excl_minus_shuffle", d, c, M2["M2_excl_minus_shuffle"]["ade"]["d"], M2["M2_excl_minus_shuffle"]["ade"]["clip_ci"]))
dev = 0.0
for n, d, c, d0, c0 in checks:
    e = max(abs(d - d0), abs(c[0] - c0[0]), abs(c[1] - c0[1])); dev = max(dev, e)
    print(f"{n:40s} raw {d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]  released {d0:+.3f} [{c0[0]:+.3f}, {c0[1]:+.3f}]")
print(f"{len(checks)} contrasts, largest deviation {dev:.2e}")
