"""Round L: every merger with PA (and K-BSM without PA) against K-BSM+PA at 309 tokens, pooled 5-fold CV over
the 1620 non-selection clips, restore and compact read-out.
  python roundL_analysis.py  -> runs/phaseA/roundL.json and a printed summary
Primary (confirmatory, specified 2026-09-29 before any run): wampa and handmultpa minus kbsmpa, restore read-out,
advantage confirmed if the clip x seed 97.5% interval (Bonferroni over the two gates) lies below 0.
"""
import json

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

ARMS = ["kbsm", "wampa", "handmultpa", "tomepa", "pitomepa", "caps2pa"]
if (RUN_ROOT / "cv" / "pitomeopa_L9").exists():
    ARMS.append("pitomeopa")          # round P: published PiToMe
for _a in ("wamshufpa", "wamshuffpa", "waminvpa", "wamcomppa", "randrankpa"):   # round S (restore only)
    if (RUN_ROOT / "cv" / f"{_a}_L9").exists():
        ARMS.append(_a)
TASK = np.load(FEAT_ROOT / "poses.npz")["task"]
PRIMARY = ["wampa", "handmultpa"]
INP = {"restore": "vis", "compact": "vis_compact"}


def pooled(cfg, inp):
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


def diff(A, B, level=95.0):
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
    tk = TASK[rows]
    ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, len(s), len(s))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    folds = [float(D[:, fold == f].mean()) for f in range(5)]
    return dict(task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                task_ci_bonf=[float(np.percentile(tb, 1.25)), float(np.percentile(tb, 98.75))],
                n_tasks=int(len(ut)), fold_d=folds,n=len(s), A=float(np.mean([A[k][1].mean() for k in s])), B=float(np.mean([B[k][1].mean() for k in s])),
                d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                clip_ci_bonf=[float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))],
                wins=int((d < 0).sum()))


R = {a: {i: pooled(f"{a}_L9", v) for i, v in INP.items()} for a in ARMS + ["kbsmpa"]}
out = {"vs_kbsmpa": {}, "compact_minus_restore": {}, "primary": {}}
for a in ARMS:
    out["vs_kbsmpa"][a] = {i: diff(R[a][i], R["kbsmpa"][i]) for i in INP}
for a in ARMS + ["kbsmpa"]:
    out["compact_minus_restore"][a] = diff(R[a]["compact"], R[a]["restore"])
for a in PRIMARY:
    p = out["vs_kbsmpa"][a]["restore"]
    out["primary"][a] = dict(d=p["d"], ci_97_5=p["clip_ci_bonf"], advantage_confirmed=bool(p["clip_ci_bonf"][1] < 0),
                             above_zero=bool(p["clip_ci_bonf"][0] > 0), task_ci_97_5=p["task_ci_bonf"])
agree = [np.sign(v["restore"]["d"]) == np.sign(v["compact"]["d"]) for v in out["vs_kbsmpa"].values()
         if v["restore"] and v["compact"]]
out["sign_agreement"] = f"{sum(agree)}/{len(agree)}"
f = lambda p: "-" if not p else f"{p['d']:+.2f} clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] task[{p['task_ci'][0]:+.2f},{p['task_ci'][1]:+.2f}] folds {' '.join(f'{x:+.1f}' for x in p['fold_d'])}"  # noqa: E731
for a, v in out["vs_kbsmpa"].items():
    print(f"{a:11s} - kbsmpa  restore {f(v['restore'])}\n{'':21s}compact {f(v['compact'])}")
S_ARMS = [a for a in ("wampa", "wamshufpa", "wamshuffpa", "waminvpa", "wamcomppa", "randrankpa") if a in ARMS]
out["roundS_vs_wampa"] = {a: diff(R[a]["restore"], R["wampa"]["restore"]) for a in S_ARMS if a != "wampa"}
print("round S, arm - wampa:", {a: (round(p["d"], 2), [round(x, 2) for x in p["clip_ci"]]) for a, p in out["roundS_vs_wampa"].items() if p})
print("compact - restore:", {a: (round(p["d"], 2) if p else None) for a, p in out["compact_minus_restore"].items()})
print("primary:", out["primary"]); print("sign agreement:", out["sign_agreement"])
write_json(out, W / "runs" / "phaseA" / "roundL.json")
