"""Seed pooling and paired clip x seed bootstraps on the official Diving48 split (as roundDV2_analysis.py)."""
import json

import numpy as np

from common import RUN_ROOT


def seeds(cfg, metric):
    out = {}
    for mf in (RUN_ROOT / "off_cls" / cfg / "vis_official").glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        r = np.load(mf.parent / "eval_rows.npy"); x = np.load(mf.parent / f"perclip_{metric}.npy")
        o = np.argsort(r); out[m["seed"]] = (r[o], x[o].astype(np.float64))
    return out


def mean_acc(cfg):
    S = seeds(cfg, "acc")
    return float(np.mean([v[1].mean() for v in S.values()])) * 100 if S else None


def pair(a, b):
    o = {}
    for metric, scale in (("acc", 100.0), ("nll", 1.0)):
        A, B = seeds(a, metric), seeds(b, metric)
        s = sorted(set(A) & set(B))
        if len(s) < 2:
            o[metric] = None
            continue
        for k in s:
            assert np.array_equal(A[k][0], B[k][0])
        D = np.stack([(A[k][1] - B[k][1]) * scale for k in s]); n = len(s)
        rng = np.random.default_rng(0)
        bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
        pc = lambda q: float(np.percentile(bo, q))  # noqa: E731
        o[metric] = dict(n=n, d=float(D.mean()), clip_ci=[pc(2.5), pc(97.5)], ci90=[pc(5), pc(95)],
                         A=float(np.mean([A[k][1].mean() for k in s])) * scale, B=float(np.mean([B[k][1].mean() for k in s])) * scale)
    return o


def show(tag, out):
    for k, v in out.items():
        if isinstance(v, dict) and v.get("acc"):
            a_, n_ = v["acc"], v["nll"]
            print(f"{tag} {k:40s} acc {a_['d']:+.2f} pp 95[{a_['clip_ci'][0]:+.2f},{a_['clip_ci'][1]:+.2f}] | nll {n_['d']:+.3f} "
                  f"[{n_['clip_ci'][0]:+.3f},{n_['clip_ci'][1]:+.3f}] (A {a_['A']:.1f}%, B {a_['B']:.1f}%)")
