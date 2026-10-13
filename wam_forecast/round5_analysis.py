"""Rounds F, M, O, FL (PROTOCOL, seventh review).  python round5_analysis.py -> runs/phaseA/round5.json"""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

TASK = np.load(FEAT_ROOT / "poses.npz")["task"]
FLOPS = {"dense": 1.0, "kbsmpa_L9": 0.636, "kbsmpa_L12": 0.553, "in8f": 0.175, "in12f": 0.281, "in16f": 0.400, "in128": 0.175,
         "in8f+kbsmpa_L9": 0.133, "in8f+kbsmpa_L12": 0.113, "in12f+kbsmpa_L9": 0.200, "in12f+kbsmpa_L12": 0.172,
         "in16f+kbsmpa_L9": 0.274, "in16f+kbsmpa_L12": 0.237, "in24f+kbsmpa_L9": 0.442, "in24f+kbsmpa_L12": 0.383}


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


K, V = "vispose_k1", "vis"
out = {"F_frontier": {}}
for sub in (K, V):
    out["F_frontier"][sub] = {c: dict(flops=f, ade=mean(c, sub), fde=mean(c, sub, "fde")) for c, f in FLOPS.items()}
for s in ("L9", "L12"):
    for sub in (K, V):
        out[f"F1_in16f_merged_minus_kbsmpa_{s}_{sub}"] = pair(f"in16f+kbsmpa_{s}", f"kbsmpa_{s}", sub)
        for base in ("in8f", "in12f", "in24f"):
            out[f"F_{base}_merged_minus_kbsmpa_{s}_{sub}"] = pair(f"{base}+kbsmpa_{s}", f"kbsmpa_{s}", sub)
        out[f"F_in16f_merged_minus_in16f_{s}_{sub}"] = pair(f"in16f+kbsmpa_{s}", "in16f", sub)
for sub in (K, V):
    for c in ("in8f", "in12f", "in16f"):
        out[f"F_{c}_minus_kbsmpa_L9_{sub}"] = pair(c, "kbsmpa_L9", sub)
        out[f"F_{c}_minus_dense_{sub}"] = pair(c, "dense", sub)
    out[f"F_kbsmpa_L9_minus_dense_{sub}"] = pair("kbsmpa_L9", "dense", sub)
out["M1_recv_minus_shuffle"] = pair("handcurrecvpa_L12", "handcurrecvshufpa_L12", K)
out["M1_q10_minus_shuffle"] = pair("handcurq10pa_L12", "handcurq10shufpa_L12", K)
out["M_recv_minus_ref"] = pair("handcurrecvpa_L12", "kbsmpa_L12", K)
out["M_q10_minus_ref"] = pair("handcurq10pa_L12", "kbsmpa_L12", K)
out["O1_fut_minus_shuffle_L12"] = pair("handfutpa_L12", "handfutshufpa_L12", K)
out["O_fut_minus_shuffle_L9"] = pair("handfutpa_L9", "handfutshufpa_L9", K)
out["O_fut_minus_ref_L12"] = pair("handfutpa_L12", "kbsmpa_L12", K)
out["O_fut_minus_ref_L9"] = pair("handfutpa_L9", "kbsmpa_L9", K)
out["FL1_flow_minus_shuffle_L9"] = pair("flowpa_L9", "flowshufpa_L9", K)
out["FL_flow_minus_shuffle_L12"] = pair("flowpa_L12", "flowshufpa_L12", K)
out["FL_flow_minus_shuffle_L9_vis"] = pair("flowpa_L9", "flowshufpa_L9", V)
out["FL_flow_minus_ref_L9"] = pair("flowpa_L9", "kbsmpa_L9", K)
out["FL_flow_minus_ref_L12"] = pair("flowpa_L12", "kbsmpa_L12", K)
out["FL_flow_minus_ref_L9_vis"] = pair("flowpa_L9", "kbsmpa_L9", V)
for k, v in out.items():
    if isinstance(v, dict) and "ade" in v and v["ade"]:
        a, f = v["ade"], v["fde"]
        print(f"R5 {k:44s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] 97.5[{a['clip_ci_bonf'][0]:+.2f},{a['clip_ci_bonf'][1]:+.2f}] "
              f"task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] seeds<0 {a['wins']}/{a['n']}" + (f" | FDE {f['d']:+.2f} [{f['clip_ci'][0]:+.2f},{f['clip_ci'][1]:+.2f}]" if f else ""))
for sub, fr in out["F_frontier"].items():
    print("frontier", sub, {c: (v["flops"], None if v["ade"] is None else round(v["ade"], 2)) for c, v in fr.items()})
write_json(out, W / "runs" / "phaseA" / "round5.json")
