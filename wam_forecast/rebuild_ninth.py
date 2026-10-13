"""Rebuild the primary contrasts of rounds DV3, VM, and SEG (ninth review) from the released raw archive alone and compare
them with roundDV3.json, roundVM.json, and roundSEG.json.   python rebuild_ninth.py --data ../../data"""
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
D3 = json.loads((DATA / "roundDV3.json").read_text())
for name, a, b in [("DV3_1_motion972_minus_in24f", "wampa_s25", "in24f"),
                   ("DV3_2_in24f_motion_minus_shuffle_s25", "in24f+wampa_s25", "in24f+wamshufpa_s25"),
                   ("DV3_3_in16f_motion_minus_shuffle_s25", "in16f+wampa_s25", "in16f+wamshufpa_s25")]:
    d, c = pair(official(a, "acc"), official(b, "acc"), 100.0)
    checks.append((name, d, c, D3[name]["acc"]["d"], D3[name]["acc"]["clip_ci"]))
VM = json.loads((DATA / "roundVM.json").read_text())
for name, a, b in [("VM_1_motion_minus_shuffle_L9", "vmae_wampa_L9", "vmae_wamshufpa_L9"),
                   ("VM_2_motion_minus_shuffle_s25", "vmae_wampa_s25", "vmae_wamshufpa_s25")]:
    d, c = pair(official(a, "acc"), official(b, "acc"), 100.0)
    checks.append((name, d, c, VM[name]["acc"]["d"], VM[name]["acc"]["clip_ci"]))
SG = json.loads((DATA / "roundSEG.json").read_text())
d, c = pair(cv("runs|cv|handcursegpa_L12|vispose_k1", "ade"), cv("runs|cv|handcursegshufpa_L12|vispose_k1", "ade"))
checks.append(("SEG_1_seg_minus_shuffle", d, c, SG["SEG_1_seg_minus_shuffle"]["ade"]["d"], SG["SEG_1_seg_minus_shuffle"]["ade"]["clip_ci"]))
dev = 0.0
for n, d, c, d0, c0 in checks:
    e = max(abs(d - d0), abs(c[0] - c0[0]), abs(c[1] - c0[1])); dev = max(dev, e)
    print(f"{n:40s} raw {d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]  released {d0:+.3f} [{c0[0]:+.3f}, {c0[1]:+.3f}]")
print(f"{len(checks)} contrasts, largest deviation {dev:.2e}")
