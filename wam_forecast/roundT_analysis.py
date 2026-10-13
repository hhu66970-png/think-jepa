"""Round T: vid-TLDR (keep salient tokens, drop the rest; ported from the official code) against K-BSM+PA, 309
tokens, pooled 5-fold CV over the 1620 non-selection clips, seeds 0-3, restore readout.
  python roundT_analysis.py -> runs/phaseA/roundT.json
Given the current pose: classification on the Bonferroni 97.5% clip x seed interval (A above 0, B contains 0, C* below
0). Video only: reported without criterion."""
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


out = {}
for name, sub in (("current_pose", "vispose_k1"), ("video_only", "vis")):
    o = out[name] = {}
    for metric in ("ade", "fde"):
        if sub == "vis":
            def pooled_vis(cfg, metric=metric):
                res = {}
                per = {}
                for f in range(5):
                    for mf in (RUN_ROOT / "cv" / cfg / f"vis_cv{f}").glob("s*/metrics.json"):
                        m = json.loads(mf.read_text())
                        assert m["args"]["input"] == "vis" and m["args"]["readout"] == "restore"
                        per.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
                for s, v in per.items():
                    if len(v) == 5:
                        rows = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
                        fold = np.concatenate([np.full(len(v[f][0]), f) for f in range(5)])
                        oo = np.argsort(rows); res[s] = (rows[oo], x[oo], fold[oo])
                return res
            A, B = pooled_vis("vidtldrpa_L9"), pooled_vis("kbsmpa_L9")
        else:
            A, B = pooled("vidtldrpa_L9", sub, "restore", metric), pooled("kbsmpa_L9", sub, "restore", metric)
        o[metric] = diff(A, B)
    p = o["ade"]
    if name == "current_pose" and p:
        lo, hi = p["clip_ci_bonf"]
        o["classification"] = "A" if lo > 0 else ("C*" if hi < 0 else "B")
    for metric in ("ade", "fde"):
        p = o[metric]
        if p:
            print(f"{name:12s} {metric} {p['d']:+.2f} clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] 97.5[{p['clip_ci_bonf'][0]:+.2f},{p['clip_ci_bonf'][1]:+.2f}] "
                  f"task[{p['task_ci'][0]:+.2f},{p['task_ci'][1]:+.2f}] folds {' '.join(f'{x:+.1f}' for x in p['fold_d'])} seeds<0 {p['wins']}/{p['n']} (A {p['A']:.2f}, ref {p['B']:.2f})")
    print(name, "classification", o.get("classification"))
write_json(out, W / "runs" / "phaseA" / "roundT.json")
