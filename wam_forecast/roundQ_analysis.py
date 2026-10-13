"""Round Q (PROTOCOL, eighth review): accuracy of probes on int8 per token quantized features (WAM_SIMQ=1, runs_Q)
minus the same probes on fp16 features (round DV, runs_DV), same clips, folds, and seeds.
  python roundQ_analysis.py -> runs/phaseA/roundQ.json"""
import json

import numpy as np

from common import W, write_json


def pooled(root, cfg, metric):
    per = {}
    for f in range(5):
        for mf in (W / root / "cv_cls" / cfg / f"vis_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            per.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
    res = {}
    for s, v in per.items():
        if len(v) == 5:
            r = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
            o = np.argsort(r); res[s] = (r[o], x[o].astype(np.float64))
    return res


out = {}
for cfg in ("kbsmpa_L9", "wampa_L9"):
    for metric, scale in (("acc", 100.0), ("nll", 1.0)):
        A, B = pooled("runs_Q", cfg, metric), pooled("runs_DV", cfg, metric)
        s = sorted(set(A) & set(B))
        if len(s) < 2:
            continue
        D = np.stack([(A[k][1] - B[k][1]) * scale for k in s]); n = len(s)
        rng = np.random.default_rng(0)
        bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
        out[f"{cfg}_{metric}"] = dict(n=n, d=float(D.mean()), clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                                      int8=float(np.mean([A[k][1].mean() for k in s])) * scale, fp16=float(np.mean([B[k][1].mean() for k in s])) * scale)
        v = out[f"{cfg}_{metric}"]
        print(f"Q {cfg} {metric}: int8 - fp16 {v['d']:+.3f} [{v['clip_ci'][0]:+.3f}, {v['clip_ci'][1]:+.3f}] (int8 {v['int8']:.2f}, fp16 {v['fp16']:.2f})")
write_json(out, W / "runs" / "phaseA" / "roundQ.json")
