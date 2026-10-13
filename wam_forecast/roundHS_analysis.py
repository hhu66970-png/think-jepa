"""Round HS: hand shuffle controls given the current pose at 131 tokens (ViT-L, L12, PA), pooled 5-fold CV over the
1620 non-selection clips, seeds 0-3.  python roundHS_analysis.py -> runs/phaseA/roundHS.json
Primary (fixed in PROTOCOL.md): handmultpa_L12 - handshuffpa_L12, ADE, clip x seed 95% interval."""
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


ARMS = {"ref": "kbsmpa_L12", "hand": "handmultpa_L12", "shuf_clip": "handshufpa_L12", "shuf_frame": "handshuffpa_L12"}
out = {}
for metric in ("ade", "fde"):
    R = {k: pooled(c, "vispose_k1", "restore", metric) for k, c in ARMS.items()}
    out[metric] = {"hand_minus_shuf_frame": diff(R["hand"], R["shuf_frame"]), "hand_minus_shuf_clip": diff(R["hand"], R["shuf_clip"]),
                   "vs_ref": {k: diff(R[k], R["ref"]) for k in ("hand", "shuf_clip", "shuf_frame")},
                   "mean": {k: float(np.mean([v[1].mean() for v in R[k].values()])) for k in R if R[k]}}
p = out["ade"]["hand_minus_shuf_frame"]
out["reading"] = "position" if p["clip_ci"][1] < 0 else ("shuffled_better" if p["clip_ci"][0] > 0 else "not_attributable_to_location")
for metric in ("ade", "fde"):
    for k, p in [("hand-shuf_frame", out[metric]["hand_minus_shuf_frame"]), ("hand-shuf_clip", out[metric]["hand_minus_shuf_clip"])] + \
            [(f"{k}-ref", v) for k, v in out[metric]["vs_ref"].items()]:
        print(f"{metric} {k:16s} {p['d']:+.2f} clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] task[{p['task_ci'][0]:+.2f},{p['task_ci'][1]:+.2f}] "
              f"folds {' '.join(f'{x:+.1f}' for x in p['fold_d'])} seeds<0 {p['wins']}/{p['n']}")
print("reading", out["reading"], {k: round(v, 2) for k, v in out["ade"]["mean"].items()})
write_json(out, W / "runs" / "phaseA" / "roundHS.json")
