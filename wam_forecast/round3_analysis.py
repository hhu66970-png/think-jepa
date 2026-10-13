"""Rounds C9, CF, SS, SM (PROTOCOL 2026-10-05 09:37 UTC), current pose, CV, written before the runs.
  python round3_analysis.py -> runs/phaseA/round3.json
  C9  handcurpa_L9 - handcurshuffpa_L9 (primary); - kbsmpa_L9 (97.5%, class); budget interaction 131 vs 309
  CF  handf31pa_L12 - handf31shuffpa_L12 (primary); - kbsmpa_L12; - handcurpa_L12
  SS  vidtldrpa - vidtldrshufpa (late), vidtldrnat - vidtldrnatshuf (early) (primary); shuffled - no saliency
  SM  vidtldrmid - fmmid; schedule interaction (vidtldrnat - fmnat) - (vidtldrmid - fmmid)"""
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


def diff2(A, B, C, D_):
    """(A - B) - (C - D_) on the same clips: an interaction contrast."""
    s = sorted(set(A) & set(B) & set(C) & set(D_))
    if len(s) < 2:
        return None
    X = {k: (A[k][0], (A[k][1] - B[k][1]) - (C[k][1] - D_[k][1]), A[k][2]) for k in s}
    for k in s:
        assert np.array_equal(A[k][0], B[k][0]) and np.array_equal(A[k][0], C[k][0]) and np.array_equal(A[k][0], D_[k][0])
    Z = {k: (X[k][0], np.zeros_like(X[k][1]), X[k][2]) for k in s}
    return diff(X, Z)


def quad(a, b, c, d, sub="vispose_k1", readout="restore"):
    return {m: diff2(*(pooled(x, sub, readout, m) for x in (a, b, c, d))) for m in ("ade", "fde")}


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


def cls(v):
    a = v["ade"] if v else None
    if not a:
        return None
    lo, hi = a["clip_ci_bonf"]
    return "A" if lo > 0 else ("C*" if hi < 0 else "B")


def read(v, neg, zero, pos):
    a = v["ade"] if v else None
    if not a:
        return None
    lo, hi = a["clip_ci"]
    return neg if hi < 0 else (pos if lo > 0 else zero)


K = "vispose_k1"
out = {}
# C9
out["C9_cur_minus_cur_shuffled_309"] = pair("handcurpa_L9", "handcurshuffpa_L9", K)
out["C9_cur_minus_ref_309"] = pair("handcurpa_L9", "kbsmpa_L9", K)
out["C9_cur_shuffled_minus_ref_309"] = pair("handcurshuffpa_L9", "kbsmpa_L9", K)
out["C9_budget_interaction_131_minus_309"] = quad("handcurpa_L12", "kbsmpa_L12", "handcurpa_L9", "kbsmpa_L9")
out["C9_class_cur_minus_ref_309"] = cls(out["C9_cur_minus_ref_309"])
out["C9_reading"] = (None if not out["C9_class_cur_minus_ref_309"] else
                     ("gain_at_309_drop_strong_compression" if out["C9_class_cur_minus_ref_309"] == "C*" else "report_as_131_result"))
# CF
out["CF_f31_minus_f31_shuffled"] = pair("handf31pa_L12", "handf31shuffpa_L12", K)
out["CF_f31_minus_ref"] = pair("handf31pa_L12", "kbsmpa_L12", K)
out["CF_f31_shuffled_minus_ref"] = pair("handf31shuffpa_L12", "kbsmpa_L12", K)
out["CF_f31_minus_cur"] = pair("handf31pa_L12", "handcurpa_L12", K)
out["CF_reading"] = read(out["CF_f31_minus_f31_shuffled"], "last_frame_location_suffices", "not_detected", "shuffled_better")
# SS
out["SS_late_real_minus_shuffled"] = pair("vidtldrpa_L9", "vidtldrshufpa_L9", K)
out["SS_early_real_minus_shuffled"] = pair("vidtldrnat_L9", "vidtldrnatshuf_L9", K)
out["SS_late_shuffled_minus_none"] = pair("vidtldrshufpa_L9", "fmpa_L9", K)
out["SS_early_shuffled_minus_none"] = pair("vidtldrnatshuf_L9", "fmnat_L9", K)
out["SS_late_reading"] = read(out["SS_late_real_minus_shuffled"], "true_positions_better", "position_not_detected", "shuffled_better")
out["SS_early_reading"] = read(out["SS_early_real_minus_shuffled"], "true_positions_better", "position_not_detected", "shuffled_better")
# SM
out["SM_late_mid_saliency_minus_none"] = pair("vidtldrmid_L9", "fmmid_L9", K)
out["SM_fmmid_minus_fmpa"] = pair("fmmid_L9", "fmpa_L9", K)
out["SM_schedule_interaction"] = quad("vidtldrnat_L9", "fmnat_L9", "vidtldrmid_L9", "fmmid_L9")
out["SM_reading"] = read(out["SM_schedule_interaction"], "schedule_flips_sign", "configuration_as_a_whole", "schedule_opposite")
report(out, "R3")
for k, v in out.items():
    if not isinstance(v, dict):
        print(k, v)
write_json(out, W / "runs" / "phaseA" / "round3.json")
