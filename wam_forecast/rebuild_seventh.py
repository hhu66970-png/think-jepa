"""Rebuild the primary contrasts of rounds F, M, O, FL (runs/cv), S (runs_S) and DV (runs_DV) from the released raw
archive alone and compare them with round5.json, roundS_{fixed30,uniform}.json and roundDV.json.
  python rebuild_seventh.py --data ../../data"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
RAW = np.load(DATA / "raw_results.npz")
KEYS = set(RAW.files)


def pooled(prefix, metric):
    out = {}
    for s in range(10):
        parts = []
        for f in range(5):
            k = f"{prefix}_cv{f}|s{s}"
            km = k if metric == "ade" else f"{k}|{metric}"
            if km not in KEYS or f"{k}|rows" not in KEYS:
                break
            parts.append((RAW[f"{k}|rows"].astype(np.int64), RAW[km].astype(np.float64)))
        if len(parts) == 5:
            r = np.concatenate([p[0] for p in parts]); x = np.concatenate([p[1] for p in parts]); o = np.argsort(r)
            out[s] = x[o]
    return out


def pair(a, b, metric, scale=1.0):
    A, B = pooled(a, metric), pooled(b, metric)
    s = sorted(set(A) & set(B))
    D = np.stack([(A[k] - B[k]) * scale for k in s]); n = len(s)
    rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    return float(D.mean()), [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]


checks = []
R5 = json.loads((DATA / "round5.json").read_text())
K = "vispose_k1"
for name, a, b in [("F1_in16f_merged_minus_kbsmpa_L9_vispose_k1", "in16f+kbsmpa_L9", "kbsmpa_L9"),
                   ("M1_recv_minus_shuffle", "handcurrecvpa_L12", "handcurrecvshufpa_L12"),
                   ("M1_q10_minus_shuffle", "handcurq10pa_L12", "handcurq10shufpa_L12"),
                   ("O1_fut_minus_shuffle_L12", "handfutpa_L12", "handfutshufpa_L12"),
                   ("FL1_flow_minus_shuffle_L9", "flowpa_L9", "flowshufpa_L9")]:
    d, c = pair(f"runs|cv|{a}|{K}", f"runs|cv|{b}|{K}", "ade")
    checks.append((name, d, c, R5[name]["ade"]["d"], R5[name]["ade"]["clip_ci"]))
for m in ("fixed30", "uniform"):
    S = json.loads((DATA / f"roundS_{m}.json").read_text())
    for name, a, b in [("S1_motion_minus_shuffle_L9", "wampa_L9", "wamshufpa_L9"), ("S1_flow_minus_shuffle_L9", "flowpa_L9", "flowshufpa_L9")]:
        d, c = pair(f"runs_S|{m}|cv|{a}|{K}", f"runs_S|{m}|cv|{b}|{K}", "ade")
        checks.append((f"{m} {name}", d, c, S[name]["ade"]["d"], S[name]["ade"]["clip_ci"]))
DV = json.loads((DATA / "roundDV.json").read_text())
for name, a, b in [("DV1_motion_minus_shuffle_L9", "wampa_L9", "wamshufpa_L9"), ("DV2_saliency_minus_shuffle_L9", "vidtldrpa_L9", "vidtldrshufpa_L9"),
                   ("DV3_flow_minus_shuffle_L9", "flowpa_L9", "flowshufpa_L9"), ("validity_kbsmpa_L12_minus_dense", "kbsmpa_L12", "dense")]:
    d, c = pair(f"runs_DV|cv_cls|{a}|vis", f"runs_DV|cv_cls|{b}|vis", "acc", 100.0)
    checks.append((name, d, c, DV[name]["acc"]["d"], DV[name]["acc"]["clip_ci"]))
dev = 0.0
for n, d, c, d0, c0 in checks:
    e = max(abs(d - d0), abs(c[0] - c0[0]), abs(c[1] - c0[1])); dev = max(dev, e)
    print(f"{n:48s} raw {d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]  released {d0:+.3f} [{c0[0]:+.3f}, {c0[1]:+.3f}]")
print(f"{len(checks)} contrasts, largest deviation {dev:.2e}")
