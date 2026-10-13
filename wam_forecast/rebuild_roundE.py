"""Rebuild round E (VideoMAE ViT-L, and the dense V-JEPA 2 reference) from the released raw archive alone.

  python rebuild_roundE.py --data ../../data
expects raw_results.npz, raw_index.json, clip_tasks.json and roundE.json in --data. Pools the five CV folds per seed,
recomputes every paired difference of roundE.json with the bootstraps of roundE_analysis.py (4000 replicates, seed 0;
clip x seed 95% and Bonferroni 97.5% intervals, task x seed 95% interval over the 13 tasks, fold effects), and prints
the largest deviation from the released aggregates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
IDX = json.loads((DATA / "raw_index.json").read_text())["runs"]
RAW = np.load(DATA / "raw_results.npz")
CT = json.loads((DATA / "clip_tasks.json").read_text())["part2"]
TASK = np.array([t for t, _ in CT["train"]] + [t for t, _ in CT["test"]])      # evaluation row -> task
REF = json.loads((DATA / "roundE.json").read_text())


def pooled(cfg):
    """{seed: (rows, ade, fold)} sorted by evaluation row, over the five folds of runs/cv/<cfg>/vis_cv{f}"""
    per = {}
    for k, m in IDX.items():
        if k.startswith(f"runs/cv/{cfg}/vis_cv"):
            f = int(k.split("/vis_cv")[1].split("/")[0])
            key = k.replace("/", "|")
            per.setdefault(m["seed"], {})[f] = (RAW[key + "|rows"], RAW[key].astype(np.float64))
    out = {}
    for s, v in per.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)]); ade = np.concatenate([v[f][1] for f in range(5)])
            fold = np.concatenate([np.full(len(v[f][0]), f) for f in range(5)])
            o = np.argsort(rows); out[s] = (rows[o], ade[o], fold[o])
    return out


def diff(A, B):
    s = sorted(set(A) & set(B))
    for k in s:
        assert np.array_equal(A[k][0], B[k][0])
    D = np.stack([A[k][1] - B[k][1] for k in s]); d = D.mean(1)
    h = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    rows, fold = A[s[0]][0], A[s[0]][2]
    tk = TASK[rows]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, len(s), len(s))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    return dict(d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                clip_ci_bonf=[float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                fold_d=[float(D[:, fold == f].mean()) for f in range(5)], wins=int((d < 0).sum()))


R = {a: pooled(f"vmae_{a}_L9") for a in ("kbsm", "kbsmpa", "wampa", "handmultpa")}
R["dense"] = pooled("vmae_dense")
V = {a: pooled(c) for a, c in (("dense", "dense"), ("kbsm", "kbsm_L9"), ("kbsmpa", "kbsmpa_L9"))}
NEW = {"pa": diff(R["kbsmpa"], R["kbsm"]),
       "vs_kbsmpa": {a: diff(R[a], R["kbsmpa"]) for a in ("kbsm", "wampa", "handmultpa")},
       "gap": {a: diff(R[a], R["dense"]) for a in ("kbsm", "kbsmpa")},
       "vjepa_gap": {a: diff(V[a], V["dense"]) for a in ("kbsm", "kbsmpa")}}
dev = 0.0


def walk(new, ref, name):
    global dev
    if isinstance(new, dict):
        for k in new:
            walk(new[k], ref[k], f"{name}.{k}")
    elif isinstance(new, list):
        dev = max(dev, float(np.max(np.abs(np.array(new) - np.array(ref)))))
    else:
        dev = max(dev, abs(new - ref))


for k in NEW:
    walk(NEW[k], REF[k], k)
fmt = lambda c: "[%+.2f, %+.2f]" % tuple(c)
for name, p in [("PA (K-BSM+PA - K-BSM)", NEW["pa"])] + [(f"{a} - K-BSM+PA", p) for a, p in NEW["vs_kbsmpa"].items()] \
        + [(f"VideoMAE {a} - dense", p) for a, p in NEW["gap"].items()] + [(f"V-JEPA 2 {a} - dense", p) for a, p in NEW["vjepa_gap"].items()]:
    print(f"{name:28s} {p['d']:+.2f}  clip {fmt(p['clip_ci'])}  97.5 {fmt(p['clip_ci_bonf'])}  task {fmt(p['task_ci'])}  seeds<0 {p['wins']}")
print(f"E1 (PA helps): {NEW['pa']['clip_ci'][1] < 0}; E2 (no gate advantage): "
      f"{all(NEW['vs_kbsmpa'][a]['clip_ci_bonf'][1] >= 0 for a in ('wampa', 'handmultpa'))}")
print(f"largest deviation from roundE.json: {dev:.4f} mm")
