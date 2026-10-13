"""Round X (PROTOCOL): the three tests on 552 new episodes of the 13 part 2 tasks (official EgoDex test archive).
  WAM_FEAT_ROOT=feats_X WAM_RUN_ROOT=runs_X WAM_SPLITS=... python roundX_analysis.py -> runs/phaseA/roundX.json
Probes trained on the 1800 part 2 training clips (current pose, seeds 0-9), evaluated on the 552 test clips."""
import json
import os
from pathlib import Path

import numpy as np

from common import FEAT_ROOT, RUN_ROOT, W, read_split, write_json

TEST = read_split("test")
TASK = np.array([t for t, _ in TEST])
P = np.load(FEAT_ROOT / "poses.npz")
NTR = len(read_split("train"))


def load(cfg, sub="vispose_k1", metric="ade"):
    out = {}
    for mf in sorted((RUN_ROOT / "test_main" / cfg / sub).glob("s*/metrics.json")):
        m = json.loads(mf.read_text())
        x = np.load(mf.parent / f"perclip_{metric}.npy").astype(np.float64)
        r = np.load(mf.parent / "eval_rows.npy") if (mf.parent / "eval_rows.npy").exists() else np.arange(NTR, NTR + len(x))
        out[int(m["seed"])] = (r, x)
    return out


def boot(D, rows):
    """D: [seeds, clips] paired differences -> clip x seed and task x seed bootstraps."""
    rng = np.random.default_rng(0)
    n = D.shape[0]
    bo = [D[rng.integers(0, n, n)][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    tk = TASK[rows - NTR]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, n, n)]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    pc = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    per_task = {t: float(D[:, tk == t].mean()) for t in ut}
    return dict(clip_ci=[pc(bo, 2.5), pc(bo, 97.5)], clip_ci_bonf=[pc(bo, 1.25), pc(bo, 98.75)],
                task_ci=[pc(tb, 2.5), pc(tb, 97.5)], tasks_negative=int(sum(v < 0 for v in per_task.values())), n_tasks=len(ut))


def diff(a, b, metric="ade"):
    A, B = load(a, metric=metric), load(b, metric=metric)
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    for k in s:
        assert np.array_equal(A[k][0], B[k][0])
    D = np.stack([A[k][1] - B[k][1] for k in s])
    return dict(n=len(s), A=float(np.mean([A[k][1].mean() for k in s])), B=float(np.mean([B[k][1].mean() for k in s])),
                d=float(D.mean()), wins=int((D.mean(1) < 0).sum()), **boot(D, A[s[0]][0]))


def inter(a, b, c, d, metric="ade"):
    R = [load(x, metric=metric) for x in (a, b, c, d)]
    s = sorted(set.intersection(*[set(r) for r in R]))
    if len(s) < 2:
        return None
    D = np.stack([(R[0][k][1] - R[1][k][1]) - (R[2][k][1] - R[3][k][1]) for k in s])
    return dict(n=len(s), d=float(D.mean()), **boot(D, R[0][s[0]][0]))


def pair(a, b):
    return {m: diff(a, b, m) for m in ("ade", "fde")}


out = {}
dn = load("dense")
rows = next(iter(dn.values()))[0] if dn else np.arange(NTR, NTR + len(TEST))
past, fut = P["past"][rows], P["future"][rows]
copy_ade = np.linalg.norm(fut - past[:, -1:], axis=-1).mean(axis=(1, 2)) * 1000
copy_fde = np.linalg.norm(fut[:, -1] - past[:, -1], axis=-1).mean(-1) * 1000
mean = lambda c, m="ade", sub="vispose_k1": (float(np.mean([v[1].mean() for v in load(c, sub, m).values()])) if load(c, sub, m) else None)  # noqa: E731
out["reference"] = dict(n_clips=int(len(rows)), copy_last_ade=float(copy_ade.mean()), copy_last_fde=float(copy_fde.mean()),
                        dense_ade=mean("dense"), dense_fde=mean("dense", "fde"), pose_only_ade=mean("none", sub="pose_k1"),
                        kbsmpa_L9_ade=mean("kbsmpa_L9"), kbsmpa_L12_ade=mean("kbsmpa_L12"))
cg = pair("kbsmpa_L12", "dense")
out["gate_H"] = dict(VG=(out["reference"]["copy_last_ade"] - out["reference"]["dense_ade"]) if out["reference"]["dense_ade"] else None,
                     CG131=cg["ade"])
out["gate_H"]["met"] = bool(out["gate_H"]["VG"] is not None and out["gate_H"]["VG"] >= 10 and cg["ade"] and cg["ade"]["clip_ci"][0] > 0)
out["X1_motion_minus_shuffled"] = pair("wampa_L9", "wamshufpa_L9")
out["X1_motion_minus_ref"] = pair("wampa_L9", "kbsmpa_L9")
out["X2_tubelet_minus_shuffled"] = pair("handcurpa_L12", "handcurshuffpa_L12")
out["X2_frame_minus_shuffled"] = pair("handf31pa_L12", "handf31shuffpa_L12")
out["X2_tubelet_minus_ref"] = pair("handcurpa_L12", "kbsmpa_L12")
out["X2_frame_minus_ref"] = pair("handf31pa_L12", "kbsmpa_L12")
out["X3_late_saliency"] = pair("vidtldrpa_L9", "fmpa_L9")
out["X3_early_saliency"] = pair("vidtldrnat_L9", "fmnat_L9")
out["X3_early_schedule_minus_ref"] = pair("fmnat_L9", "kbsmpa_L9")
out["X3_interaction"] = {m: inter("vidtldrnat_L9", "fmnat_L9", "vidtldrpa_L9", "fmpa_L9", m) for m in ("ade", "fde")}
ok = lambda v, f: bool(v and v.get("ade") and f(v["ade"]))  # noqa: E731
out["X1_replicated"] = ok(out["X1_motion_minus_shuffled"], lambda a: a["clip_ci"][0] > -1e9 and not a["clip_ci"][1] < 0)
out["X2_replicated"] = ok(out["X2_tubelet_minus_shuffled"], lambda a: a["clip_ci"][1] < 0) and ok(out["X2_frame_minus_shuffled"], lambda a: a["clip_ci"][1] < 0)
out["X3_replicated"] = ok(out["X3_interaction"], lambda a: a["clip_ci"][1] < 0)
for k, v in out.items():
    if isinstance(v, dict) and "ade" in v and v["ade"]:
        a = v["ade"]; f = v.get("fde") or {}
        print(f"X {k:30s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] 97.5[{a['clip_ci_bonf'][0]:+.2f},{a['clip_ci_bonf'][1]:+.2f}] "
              f"task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] tasks<0 {a['tasks_negative']}/{a['n_tasks']} n {a['n']}"
              + (f" | FDE {f['d']:+.2f} [{f['clip_ci'][0]:+.2f},{f['clip_ci'][1]:+.2f}]" if f else ""))
print("reference", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["reference"].items()})
print("gate H", round(out["gate_H"]["VG"], 2) if out["gate_H"]["VG"] else None, out["gate_H"]["met"])
print("replicated: X1", out["X1_replicated"], "X2", out["X2_replicated"], "X3", out["X3_replicated"])
write_json(out, W / "runs" / "phaseA" / "roundX.json")
