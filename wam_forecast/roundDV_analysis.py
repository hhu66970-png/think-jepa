"""Round DV (PROTOCOL, seventh review): Diving48 action recognition, five fold CV over the 1970 test split clips.
  WAM_SLIM=... WAM_SPLITS=... WAM_FEAT_ROOT=feats_DV WAM_RUN_ROOT=runs_DV python roundDV_analysis.py -> runs/phaseA/roundDV.json
Differences in accuracy are in percentage points (positive = the first arm is better); cross entropy: negative = better."""
import json

import numpy as np

from common import RUN_ROOT, W, write_json


def pooled(cfg, metric):
    per = {}
    for f in range(5):
        for mf in (RUN_ROOT / "cv_cls" / cfg / f"vis_cv{f}").glob("s*/metrics.json"):
            m = json.loads(mf.read_text())
            per.setdefault(m["seed"], {})[f] = (np.load(mf.parent / "eval_rows.npy"), np.load(mf.parent / f"perclip_{metric}.npy"))
    res = {}
    for s, v in per.items():
        if len(v) == 5:
            r = np.concatenate([v[f][0] for f in range(5)]); x = np.concatenate([v[f][1] for f in range(5)])
            o = np.argsort(r); res[s] = (r[o], x[o].astype(np.float64))
    return res


def pair(a, b):
    o = {}
    for metric, scale in (("acc", 100.0), ("nll", 1.0)):
        A, B = pooled(a, metric), pooled(b, metric)
        s = sorted(set(A) & set(B))
        if len(s) < 2:
            o[metric] = None; continue
        D = np.stack([(A[k][1] - B[k][1]) * scale for k in s]); n = len(s)
        rng = np.random.default_rng(0)
        bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
        pc = lambda q: float(np.percentile(bo, q))  # noqa: E731
        o[metric] = dict(n=n, d=float(D.mean()), clip_ci=[pc(2.5), pc(97.5)], clip_ci_bonf=[pc(1.25), pc(98.75)], ci90=[pc(5), pc(95)],
                         A=float(np.mean([A[k][1].mean() for k in s])) * scale, B=float(np.mean([B[k][1].mean() for k in s])) * scale)
    return o


def quad(a, b, c, d):
    R = [pooled(x, "acc") for x in (a, b, c, d)]
    s = sorted(set.intersection(*[set(r) for r in R]))
    if len(s) < 2:
        return None
    D = np.stack([((R[0][k][1] - R[1][k][1]) - (R[2][k][1] - R[3][k][1])) * 100 for k in s]); n = len(s)
    rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    return dict(n=n, d=float(D.mean()), clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))])


acc = {c: (float(np.mean([v[1].mean() for v in pooled(c, "acc").values()])) * 100 if pooled(c, "acc") else None) for c in
       ["dense", "kbsmpa_L9", "kbsmpa_L12", "pospa_L9", "randmpa_L9", "in16f", "wampa_L9", "wamshufpa_L9", "wampa_L12", "wamshufpa_L12",
        "flowpa_L9", "flowshufpa_L9", "vidtldrpa_L9", "vidtldrshufpa_L9", "fmpa_L9", "vidtldrnat_L9", "fmnat_L9", "vidtldrnatshuf_L9"]}
out = {"accuracy": acc,
       "validity_kbsmpa_L12_minus_dense": pair("kbsmpa_L12", "dense"), "kbsmpa_L9_minus_dense": pair("kbsmpa_L9", "dense"),
       "DV1_motion_minus_shuffle_L9": pair("wampa_L9", "wamshufpa_L9"), "DV2_saliency_minus_shuffle_L9": pair("vidtldrpa_L9", "vidtldrshufpa_L9"),
       "DV3_flow_minus_shuffle_L9": pair("flowpa_L9", "flowshufpa_L9"), "DV_motion_minus_shuffle_L12": pair("wampa_L12", "wamshufpa_L12"),
       "DV_motion_minus_ref_L9": pair("wampa_L9", "kbsmpa_L9"), "DV_flow_minus_ref_L9": pair("flowpa_L9", "kbsmpa_L9"),
       "DV_saliency_minus_ref_L9": pair("vidtldrpa_L9", "kbsmpa_L9"), "DV_motion_minus_ref_L12": pair("wampa_L12", "kbsmpa_L12"),
       "DV_pospa_minus_ref": pair("pospa_L9", "kbsmpa_L9"), "DV_randm_minus_ref": pair("randmpa_L9", "kbsmpa_L9"),
       "DV_in16f_minus_ref": pair("in16f", "kbsmpa_L9"), "DV_in16f_minus_dense": pair("in16f", "dense"),
       "DV_early_saliency_shuffle": pair("vidtldrnat_L9", "vidtldrnatshuf_L9"),
       "DV_late_saliency": pair("vidtldrpa_L9", "fmpa_L9"), "DV_early_saliency": pair("vidtldrnat_L9", "fmnat_L9"),
       "DV_saliency_early_minus_late": quad("vidtldrnat_L9", "fmnat_L9", "vidtldrpa_L9", "fmpa_L9")}
v = out["validity_kbsmpa_L12_minus_dense"]["acc"]
out["valid"] = bool(v and v["clip_ci"][1] < 0)
for k, v in out.items():
    if isinstance(v, dict) and v.get("acc"):
        a_, n_ = v["acc"], v["nll"]
        print(f"DV {k:34s} acc {a_['d']:+.2f} pp 95[{a_['clip_ci'][0]:+.2f},{a_['clip_ci'][1]:+.2f}] | nll {n_['d']:+.3f} [{n_['clip_ci'][0]:+.3f},{n_['clip_ci'][1]:+.3f}] (A {a_['A']:.1f}%)")
print("DV accuracy", {k: (None if v is None else round(v, 1)) for k, v in acc.items()}, "valid", out["valid"])
print("DV saliency early minus late", out["DV_saliency_early_minus_late"])
write_json(out, W / "runs" / "phaseA" / "roundDV.json")
