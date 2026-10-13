"""CV pooling and paired intervals shared by the seventh review analyses (same code as round5_analysis.py)."""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

TASK = np.load(FEAT_ROOT / "poses.npz")["task"]


def pooled(cfg, sub, metric):
    out = {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv" / cfg / f"{sub}_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            out.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
    res = {}
    for s, v in out.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
            o = np.argsort(rows); res[s] = (rows[o], x[o].astype(np.float64))
    return res


def stats(D, rows):
    n = D.shape[0]; d = D.mean(1); h = tcrit(n - 1) * d.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    tk = TASK[rows]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, n, n)]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    pc = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return dict(n=n, d=float(D.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)], clip_ci=[pc(bo, 2.5), pc(bo, 97.5)],
                clip_ci_bonf=[pc(bo, 1.25), pc(bo, 98.75)], ci90=[pc(bo, 5), pc(bo, 95)], task_ci=[pc(tb, 2.5), pc(tb, 97.5)],
                wins=int((d < 0).sum()))


def pair(a, b, sub, sub_b=None):
    o = {}
    for metric in ("ade", "fde"):
        A, B = pooled(a, sub, metric), pooled(b, sub_b or sub, metric)
        s = sorted(set(A) & set(B))
        if len(s) < 2:
            o[metric] = None; continue
        for k in s:
            assert np.array_equal(A[k][0], B[k][0])
        D = np.stack([A[k][1] - B[k][1] for k in s])
        r = stats(D, A[s[0]][0]); r["A"] = float(np.mean([A[k][1].mean() for k in s])); r["B"] = float(np.mean([B[k][1].mean() for k in s]))
        o[metric] = r
    return o


def mean(cfg, sub, metric="ade"):
    A = pooled(cfg, sub, metric)
    return float(np.mean([v[1].mean() for v in A.values()])) if A else None
