"""Round DV2 (PROTOCOL, eighth review): Diving48 on the official split (train 15027, test 1970), seeds 0-3.
  WAM_SLIM=.../official/slim WAM_SPLITS=.../official/splits WAM_FEAT_ROOT=feats_DV2 WAM_RUN_ROOT=runs_DV2 \
  python roundDV2_analysis.py -> runs/phaseA/roundDV2.json
Accuracy differences in percentage points (positive = the first arm is better), cross entropy in nats (negative =
better); clip x seed bootstrap over the 1970 test clips (4000 replicates, seed 0)."""
import json

import numpy as np

from common import RUN_ROOT, W, write_json

CFGS = ["dense", "in16f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25", "kbsmpa_L9", "wampa_L9", "wamshufpa_L9",
        "kbsmpa_L12", "wampa_L12", "wamshufpa_L12", "in16f+kbsmpa_L9", "in16f+wampa_L9", "in16f+wamshufpa_L9",
        "vidtldrpa_L9", "vidtldrshufpa_L9", "pospa_L9", "flowpa_L9", "flowshufpa_L9", "flowcpa_L9", "flowcshufpa_L9"]


def seeds(cfg, metric):
    out = {}
    for mf in (RUN_ROOT / "off_cls" / cfg / "vis_official").glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        r = np.load(mf.parent / "eval_rows.npy"); x = np.load(mf.parent / f"perclip_{metric}.npy")
        o = np.argsort(r); out[m["seed"]] = (r[o], x[o].astype(np.float64))
    return out


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


acc = {}
for c in CFGS:
    S = seeds(c, "acc")
    acc[c] = dict(mean=float(np.mean([v[1].mean() for v in S.values()])) * 100 if S else None, n=len(S))
out = {"accuracy": acc,
       "DV2_1_motion_minus_shuffle_s25": pair("wampa_s25", "wamshufpa_s25"),
       "DV2_2_motion_minus_shuffle_in16f_L9": pair("in16f+wampa_L9", "in16f+wamshufpa_L9"),
       "DV2_3_motion_minus_shuffle_L9": pair("wampa_L9", "wamshufpa_L9"),
       "motion_minus_shuffle_L12": pair("wampa_L12", "wamshufpa_L12"),
       "motion_minus_ref_s25": pair("wampa_s25", "kbsmpa_s25"), "motion_minus_ref_L9": pair("wampa_L9", "kbsmpa_L9"),
       "motion_minus_ref_L12": pair("wampa_L12", "kbsmpa_L12"), "motion_minus_ref_in16f_L9": pair("in16f+wampa_L9", "in16f+kbsmpa_L9"),
       "kbsmpa_s25_minus_dense": pair("kbsmpa_s25", "dense"), "kbsmpa_L9_minus_dense": pair("kbsmpa_L9", "dense"),
       "kbsmpa_L12_minus_dense": pair("kbsmpa_L12", "dense"), "in16f_minus_dense": pair("in16f", "dense"),
       "in16f_merged_minus_in16f": pair("in16f+kbsmpa_L9", "in16f"), "in16f_minus_kbsmpa_L9": pair("in16f", "kbsmpa_L9"),
       "in16f_minus_kbsmpa_s25": pair("in16f", "kbsmpa_s25"),
       "in16f_merged_minus_kbsmpa_L9": pair("in16f+kbsmpa_L9", "kbsmpa_L9"),
       "saliency_minus_shuffle_L9": pair("vidtldrpa_L9", "vidtldrshufpa_L9"), "saliency_minus_ref_L9": pair("vidtldrpa_L9", "kbsmpa_L9"),
       "flow_minus_shuffle_L9": pair("flowpa_L9", "flowshufpa_L9"), "flowc_minus_shuffle_L9": pair("flowcpa_L9", "flowcshufpa_L9"),
       "flow_minus_ref_L9": pair("flowpa_L9", "kbsmpa_L9"), "flowc_minus_ref_L9": pair("flowcpa_L9", "kbsmpa_L9"),
       "pospa_minus_ref_L9": pair("pospa_L9", "kbsmpa_L9")}
for k, v in out.items():
    if isinstance(v, dict) and v.get("acc"):
        a_, n_ = v["acc"], v["nll"]
        print(f"DV2 {k:38s} acc {a_['d']:+.2f} pp 95[{a_['clip_ci'][0]:+.2f},{a_['clip_ci'][1]:+.2f}] | nll {n_['d']:+.3f} "
              f"[{n_['clip_ci'][0]:+.3f},{n_['clip_ci'][1]:+.3f}] (A {a_['A']:.1f}%, B {a_['B']:.1f}%)")
print("DV2 accuracy", {k: (None if v["mean"] is None else round(v["mean"], 1)) for k, v in acc.items()})
write_json(out, W / "runs" / "phaseA" / "roundDV2.json")
