"""Round CP: relevance gates in the current pose setting (input vispose, pose_frames 1), V-JEPA 2 ViT-L, 309 tokens,
pooled 5-fold CV over the 1620 non-selection clips, seeds 0-3, restore readout.
  python roundCP_analysis.py -> runs/phaseA/roundCP.json
Each arm minus kbsmpa_L9 and wampa minus each shuffled arm, paired by fold and seed, for ADE and FDE: seed t interval,
clip x seed bootstrap (95%, Bonferroni 97.5%), task x seed bootstrap, fold effects. Classification of the motion gate
as fixed in PROTOCOL.md (ROUND CP SPECIFICATION): A (97.5% interval above 0), B (contains 0), C (below 0 and the 95%
interval of wampa - wamshufpa below 0)."""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

TASK = np.load(FEAT_ROOT / "poses.npz")["task"]
INP = "vispose_k1"


def pooled(cfg, metric):
    out = {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv" / cfg / f"{INP}_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            assert m["args"]["pose_frames"] == 1 and m["args"]["input"] == "vispose" and m["args"]["readout"] == "restore"
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


ARMS = ("wampa_L9", "handmultpa_L9", "wamshufpa_L9", "wamshuffpa_L9")
out = {}
for metric in ("ade", "fde"):
    R = {a: pooled(a, metric) for a in ARMS + ("kbsmpa_L9",)}
    out[metric] = {"vs_kbsmpa": {a: diff(R[a], R["kbsmpa_L9"]) for a in ARMS},
                   "motion_vs_shuffled": {a: diff(R["wampa_L9"], R[a]) for a in ("wamshufpa_L9", "wamshuffpa_L9")},
                   "mean": {a: float(np.mean([v[1].mean() for v in R[a].values()])) for a in R if R[a]},
                   "n_seeds": {a: len(R[a]) for a in R}}
m = out["ade"]["vs_kbsmpa"]["wampa_L9"]; sh = out["ade"]["motion_vs_shuffled"]["wamshufpa_L9"]
lo, hi = m["clip_ci_bonf"]
out["classification"] = "A" if lo > 0 else ("C" if hi < 0 and sh["clip_ci"][1] < 0 else ("B" if lo <= 0 <= hi else "below0_not_signal"))
for metric in ("ade", "fde"):
    for k, p in list(out[metric]["vs_kbsmpa"].items()) + [(f"wampa - {a}", p) for a, p in out[metric]["motion_vs_shuffled"].items()]:
        print(f"{metric} {k:24s} {p['d']:+.2f} clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] "
              f"97.5[{p['clip_ci_bonf'][0]:+.2f},{p['clip_ci_bonf'][1]:+.2f}] task[{p['task_ci'][0]:+.2f},{p['task_ci'][1]:+.2f}] "
              f"folds {' '.join(f'{x:+.1f}' for x in p['fold_d'])} seeds<0 {p['wins']}/{p['n']}")
print("classification", out["classification"], "means", {k: round(v, 2) for k, v in out["ade"]["mean"].items()})
write_json(out, W / "runs" / "phaseA" / "roundCP.json")
