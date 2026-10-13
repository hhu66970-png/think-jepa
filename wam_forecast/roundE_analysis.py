"""Round E: VideoMAE ViT-L, pooled 5-fold CV over the 1620 non-selection clips, restore read-out.
  python roundE_analysis.py -> runs/phaseA/roundE.json
E1: vmae_kbsmpa - vmae_kbsm (PA) clip x seed 95% CI below 0. E2: neither gate with PA has an advantage over
vmae_kbsmpa (97.5% clip x seed CI below 0 would be one). Also reported: task x seed intervals and fold effects."""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

TASK = np.load(FEAT_ROOT / "poses.npz")["task"]


def pooled(cfg, inp="vis"):
    out = {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv" / cfg / f"{inp}_cv{f}").glob("s*/metrics.json"):
            s = json.loads(mf.read_text())["seed"]
            out.setdefault(s, {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / "perclip_ade.npy"))
    res = {}
    for s, v in out.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)]); ade = np.concatenate([v[f][1] for f in range(5)])
            fold = np.concatenate([np.full(len(v[f][0]), f) for f in range(5)])
            o = np.argsort(rows); res[s] = (rows[o], ade[o], fold[o])
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
    return dict(n=len(s), A=float(np.mean([A[k][1].mean() for k in s])), B=float(np.mean([B[k][1].mean() for k in s])),
                d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                clip_ci_bonf=[float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                fold_d=[float(D[:, fold == f].mean()) for f in range(5)], wins=int((d < 0).sum()))


R = {a: pooled(f"vmae_{a}_L9") for a in ("kbsm", "kbsmpa", "wampa", "handmultpa", "wamshufpa")}
R["dense"] = pooled("vmae_dense")       # descriptive reference (protocol addendum), enters no criterion
R = {a: v for a, v in R.items() if v}
out = {"pa": diff(R["kbsmpa"], R["kbsm"]), "vs_kbsmpa": {}}
for a in ("kbsm", "wampa", "handmultpa", "wamshufpa"):
    if a in R:
        out["vs_kbsmpa"][a] = diff(R[a], R["kbsmpa"])
if "dense" in R:
    out["gap"] = {a: diff(R[a], R["dense"]) for a in ("kbsm", "kbsmpa") if a in R}
V = {a: pooled(c) for a, c in (("dense", "dense"), ("kbsm", "kbsm_L9"), ("kbsmpa", "kbsmpa_L9"))}
if all(V.values()):                       # post hoc, descriptive: the V-JEPA 2 ViT-L gap on the same clips
    out["vjepa_gap"] = {a: diff(V[a], V["dense"]) for a in ("kbsm", "kbsmpa")}
    out["vjepa_ade"] = {a: float(np.mean([v[1].mean() for v in V[a].values()])) for a in V}
out["E1_pa_helps"] = bool(out["pa"]["clip_ci"][1] < 0)
out["E2_no_gate_advantage"] = {a: not (out["vs_kbsmpa"][a]["clip_ci_bonf"][1] < 0) for a in ("wampa", "handmultpa") if a in out["vs_kbsmpa"]}
out["ade"] = {a: float(np.mean([v[1].mean() for v in R[a].values()])) for a in R}
for k, p in [("pa (kbsmpa - kbsm)", out["pa"])] + [(f"{a} - kbsmpa", p) for a, p in out["vs_kbsmpa"].items()]:
    print(f"{k:22s} {p['d']:+.2f} clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] 97.5[{p['clip_ci_bonf'][0]:+.2f},{p['clip_ci_bonf'][1]:+.2f}] "
          f"task[{p['task_ci'][0]:+.2f},{p['task_ci'][1]:+.2f}] folds {' '.join(f'{x:+.1f}' for x in p['fold_d'])} n{p['n']}")
print("E1", out["E1_pa_helps"], "E2", out["E2_no_gate_advantage"], "ADE", {k: round(v, 2) for k, v in out["ade"].items()})
write_json(out, W / "runs" / "phaseA" / "roundE.json")
