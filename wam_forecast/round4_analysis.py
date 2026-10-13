"""Rounds B, P, D, H (PROTOCOL, sixth review): baselines, positive control, stronger readout, hard exemption.
  python round4_analysis.py -> runs/phaseA/round4.json"""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

P = np.load(FEAT_ROOT / "poses.npz")
TASK = P["task"]


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
    n = D.shape[0]; d = D.mean(1)
    h = tcrit(n - 1) * d.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    tk = TASK[rows]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, n, n)]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    pc = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return dict(n=n, d=float(D.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[pc(bo, 2.5), pc(bo, 97.5)], clip_ci_bonf=[pc(bo, 1.25), pc(bo, 98.75)],
                task_ci=[pc(tb, 2.5), pc(tb, 97.5)], wins=int((d < 0).sum()))


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


def mean(cfg, sub):
    A = pooled(cfg, sub, "ade")
    return float(np.mean([v[1].mean() for v in A.values()])) if A else None


K, KL, KD = "vispose_k1", "vispose_k1_local", "vispose_k1_d6"
out = {}
# B
for c in ("pospa", "randmpa", "randdroppa"):
    for s in ("L9", "L12"):
        out[f"B_{c}_{s}_minus_kbsmpa"] = pair(f"{c}_{s}", f"kbsmpa_{s}", K)
for c in ("in16f", "in128"):
    out[f"B_{c}_minus_kbsmpa_L9"] = pair(c, "kbsmpa_L9", K)
    out[f"B_{c}_minus_dense"] = pair(c, "dense", K)
out["B_pospa_L9_minus_kbsmpa_vis"] = pair("pospa_L9", "kbsmpa_L9", "vis")
out["B_in16f_minus_kbsmpa_L9_vis"] = pair("in16f", "kbsmpa_L9", "vis")
for s in ("L9", "L12"):
    a = out[f"B_pospa_{s}_minus_kbsmpa"]["ade"]
    out[f"B1_{s}_reading"] = None if not a else ("similarity_beats_uniform" if a["clip_ci_bonf"][0] > 0 else
                                                ("uniform_better" if a["clip_ci_bonf"][1] < 0 else "not_detected"))
out["B_means"] = {c: mean(c, K) for c in ("dense", "kbsmpa_L9", "kbsmpa_L12", "pospa_L9", "pospa_L12", "randmpa_L9", "randmpa_L12",
                                          "randdroppa_L9", "randdroppa_L12", "in16f", "in128")}
# P (local finger target)
out["P_validity_kbsmpa_L12_minus_dense"] = pair("kbsmpa_L12", "dense", KL)
out["P_kbsmpa_L9_minus_dense"] = pair("kbsmpa_L9", "dense", KL)
out["P1_frame_minus_shuffle"] = pair("handf31pa_L12", "handf31shufpa_L12", KL)
out["P_tubelet_minus_shuffle"] = pair("handcurpa_L12", "handcurshufpa_L12", KL)
out["P_frame_minus_ref"] = pair("handf31pa_L12", "kbsmpa_L12", KL)
out["P_tubelet_minus_ref"] = pair("handcurpa_L12", "kbsmpa_L12", KL)
out["P_motion_minus_shuffle"] = pair("wampa_L9", "wamshufpa_L9", KL)
out["P_motion_minus_ref"] = pair("wampa_L9", "kbsmpa_L9", KL)
loc_past = P["past"] - P["past"][:, :, [24] * 26 + [50] * 26]
loc_fut = P["future"] - P["future"][:, :, [24] * 26 + [50] * 26]
fj = [j for j in range(52) if j not in (24, 25, 50, 51)]
copy_local = np.linalg.norm(loc_fut[:, :, fj] - loc_past[:, -1:, fj], axis=-1).mean(axis=(1, 2)) * 1000
rows = next(iter(pooled("dense", KL, "ade").values()), (None,))[0]
out["P_means"] = {c: mean(c, KL) for c in ("dense", "kbsmpa_L9", "kbsmpa_L12", "handf31pa_L12", "handf31shufpa_L12", "wampa_L9")}
out["P_means"]["pose_only"] = mean("none", "pose_k1_local")
out["P_means"]["copy_last_local"] = float(copy_local[rows].mean()) if rows is not None else None
pv = out["P_validity_kbsmpa_L12_minus_dense"]["ade"]
out["P_valid"] = bool(pv and pv["clip_ci"][0] > 0)
p1 = out["P1_frame_minus_shuffle"]["ade"]
out["P1_reading"] = None if not p1 else ("detects_spatial_allocation" if p1["clip_ci"][1] < 0 else "not_detected")
# D (depth 6, width 512)
out["D_motion_minus_shuffle"] = pair("wampa_L9", "wamshufpa_L9", KD)
out["D_frame_minus_shuffle"] = pair("handf31pa_L12", "handf31shufpa_L12", KD)
out["D_motion_minus_ref"] = pair("wampa_L9", "kbsmpa_L9", KD)
out["D_frame_minus_ref"] = pair("handf31pa_L12", "kbsmpa_L12", KD)
out["D_kbsmpa_L9_minus_dense"] = pair("kbsmpa_L9", "dense", KD)
out["D_dense_deep_minus_small"] = pair("dense", "dense", KD, K)
out["D_kbsmpa_L9_deep_minus_small"] = pair("kbsmpa_L9", "kbsmpa_L9", KD, K)
dm, df = out["D_motion_minus_shuffle"]["ade"], out["D_frame_minus_shuffle"]["ade"]
out["D1_replicated"] = bool(dm and df and not dm["clip_ci"][1] < 0 and df["clip_ci"][1] < 0)
# H (hard exemption, quota 0.2)
out["H1_hand_minus_shuffle"] = pair("handcurq20pa_L12", "handcurq20shufpa_L12", K)
out["H2_motion_minus_shuffle"] = pair("wamq20pa_L9", "wamq20shufpa_L9", K)
out["H_hand_minus_ref"] = pair("handcurq20pa_L12", "kbsmpa_L12", K)
out["H_motion_minus_ref"] = pair("wamq20pa_L9", "kbsmpa_L9", K)
for k, v in out.items():
    if isinstance(v, dict) and "ade" in v and v["ade"]:
        a, f = v["ade"], v["fde"]
        print(f"R4 {k:34s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] 97.5[{a['clip_ci_bonf'][0]:+.2f},{a['clip_ci_bonf'][1]:+.2f}] "
              f"task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] seeds<0 {a['wins']}/{a['n']}" + (f" | FDE {f['d']:+.2f} [{f['clip_ci'][0]:+.2f},{f['clip_ci'][1]:+.2f}]" if f else ""))
    elif not isinstance(v, dict) or "ade" not in v:
        print(k, v)
write_json(out, W / "runs" / "phaseA" / "round4.json")
