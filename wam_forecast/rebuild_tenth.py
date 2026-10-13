"""Rebuild the primary contrasts of rounds TOK and G48 (tenth review) from the released raw archive alone and compare
them with roundTOK.json and roundG48.json.   python rebuild_tenth.py --data ../../data"""
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


def official(cfg, sub, metric="acc"):
    out = {}
    for s in range(10):
        k = f"runs_DV2|off_cls|{cfg}|{sub}|s{s}"
        if f"{k}|{metric}" in KEYS:
            r = RAW[f"{k}|rows"].astype(np.int64); o = np.argsort(r); out[s] = RAW[f"{k}|{metric}"][o].astype(np.float64)
    return out


def cv(cfg, sub):
    out = {}
    for s in range(10):
        parts = []
        for f in range(5):
            k = f"runs|cv|{cfg}|{sub}_cv{f}|s{s}"
            if k not in KEYS:
                break
            parts.append((RAW[f"{k}|rows"].astype(np.int64), RAW[k].astype(np.float64)))
        if len(parts) == 5:
            r = np.concatenate([p[0] for p in parts]); x = np.concatenate([p[1] for p in parts]); out[s] = x[np.argsort(r)]
    return out


def pair(A, B, scale=1.0):
    s = sorted(set(A) & set(B))
    return boot(np.stack([(A[k] - B[k]) * scale for k in s]))


checks = []
TK = json.loads((DATA / "roundTOK.json").read_text())
E = "vispose_k1_cmp"
for name, a, b in [("TOK_1_kbsm131_minus_4f_pool128", "kbsmpa_L12", "in4f@p822"), ("TOK_2_hand131_minus_4f_pool128", "handcurpa_L12", "in4f@p822"),
                   ("TOK_3_hand131_minus_shuffle", "handcurpa_L12", "handcurshufpa_L12"), ("TOK_4_hand309_minus_4f", "handcurpa_L9", "in4f")]:
    d, c = pair(cv(a, E), cv(b, E))
    checks.append((name, d, c, TK["ego"][name]["ade"]["d"], TK["ego"][name]["ade"]["clip_ci"]))
for name, a, b in [("TOK_5_kbsm131_minus_4f_pool128", "kbsmpa_L12", "in4f@p822"), ("TOK_6_motion131_minus_4f_pool128", "wampa_L12", "in4f@p822"),
                   ("TOK_7_motion131_minus_shuffle", "wampa_L12", "wamshufpa_L12"), ("TOK_8_motion309_minus_4f", "wampa_L9", "in4f")]:
    d, c = pair(official(a, "vis_official_cmp"), official(b, "vis_official_cmp"), 100.0)
    checks.append((name, d, c, TK["dv"][name]["d"], TK["dv"][name]["clip_ci"]))
G = json.loads((DATA / "roundG48.json").read_text())
for name, a, b in [("G_1_motion_minus_shuffle", "vitg_wampa_gL9", "vitg_wamshufpa_gL9"), ("G_2_motion_minus_in12f", "vitg_wampa_gL9", "vitg_in12f")]:
    d, c = pair(official(a, "vis_official"), official(b, "vis_official"), 100.0)
    checks.append((name, d, c, G[name]["acc"]["d"], G[name]["acc"]["clip_ci"]))
dev = 0.0
for name, d, c, d0, c0 in checks:
    dev = max(dev, abs(d - d0), abs(c[0] - c0[0]), abs(c[1] - c0[1]))
    print(f"{name:34s} rebuilt {d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]  released {d0:+.3f} [{c0[0]:+.3f}, {c0[1]:+.3f}]")
print(f"{len(checks)} contrasts, largest deviation {dev:.2e}")
