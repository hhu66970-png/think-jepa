"""Round N3 (late schedule: saliency rule versus hidden feature matching). Round N2: is the cost of the native early schedule specific to the vid-TLDR saliency rule? Round N: (merge between attention and MLP, official order; early or late schedule), CV.
  python roundN_analysis.py -> runs/phaseA/roundN.json"""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

TASK = np.load(FEAT_ROOT / "poses.npz")["task"]


def pooled(cfg, sub, readout, metric):
    out = {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv" / cfg / f"{sub}_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            assert m["args"]["pose_frames"] == 1 and m["args"]["input"] == "vispose" and m["args"]["readout"] == readout
            out.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
    res = {}
    for s, v in out.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
            fold = np.concatenate([np.full(len(v[f][0]), f) for f in range(5)])
            o = np.argsort(rows); res[s] = (rows[o], x[o], fold[o])
    return res


def diff(A, B):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    for k in s:
        assert np.array_equal(A[k][0], B[k][0])
    D = np.stack([A[k][1] - B[k][1] for k in s]); d = D.mean(1)
    h = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    rows, fold = A[s[0]][0], A[s[0]][2]
    tk = TASK[rows]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, len(s), len(s))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    return dict(n=len(s), n_clips=int(D.shape[1]), A=float(np.mean([A[k][1].mean() for k in s])),
                B=float(np.mean([B[k][1].mean() for k in s])), d=float(d.mean()),
                seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                clip_ci_bonf=[float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                fold_d=[float(D[:, fold == f].mean()) for f in range(5)], wins=int((d < 0).sum()))


def report(out, title):
    for k, v in out.items():
        if isinstance(v, dict) and "ade" in v and v["ade"]:
            a, f = v["ade"], v["fde"]
            print(f"{title} {k:28s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] 97.5[{a['clip_ci_bonf'][0]:+.2f},{a['clip_ci_bonf'][1]:+.2f}] "
                  f"task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] seeds<0 {a['wins']}/{a['n']} | FDE {f['d']:+.2f} [{f['clip_ci'][0]:+.2f},{f['clip_ci'][1]:+.2f}] (A {a['A']:.2f})")


def pair(a, b, sub, readout="restore"):
    o = {}
    for metric in ("ade", "fde"):
        if sub == "vis":
            A, B = pooled_vis(a, metric), pooled_vis(b, metric)
        else:
            A, B = pooled(a, sub, readout, metric), pooled(b, sub, readout, metric)
        o[metric] = diff(A, B)
    return o


def pooled_vis(cfg, metric):
    per, res = {}, {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv" / cfg / f"vis_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            assert m["args"]["input"] == "vis" and m["args"]["readout"] == "restore"
            per.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
    for s, v in per.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
            fold = np.concatenate([np.full(len(v[f][0]), f) for f in range(5)])
            o = np.argsort(rows); res[s] = (rows[o], x[o], fold[o])
    return res


def classify(p):
    lo, hi = p["clip_ci_bonf"]
    return "A" if lo > 0 else ("C*" if hi < 0 else "B")


out = {"vidtldrpa_minus_fmpa": pair("vidtldrpa_L9", "fmpa_L9", "vispose_k1"),
       "fmpa_minus_kbsmpa": pair("fmpa_L9", "kbsmpa_L9", "vispose_k1")}
report(out, "N3")
write_json(out, W / "runs" / "phaseA" / "roundN3.json")
