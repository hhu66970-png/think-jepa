"""Recompute every Round V summary from the released per-clip archive alone.

  python rebuild_roundV.py --data ../../data
Inputs: roundV_perclip.npz (keys '<input>|s<seed>|<intervention>', per-clip ADE of the 200 part-2 test clips in
evaluation order), clip_tasks.json (task of each of those clips), roundV.json (released summaries). Prints, for each
intervention and each paired contrast between masked regions, the mean change with its seed, clip x seed, and
task x seed 95% intervals (4000 replicates, seed 0), a Holm adjustment over the six region contrasts, and the
largest deviation from roundV.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
D = Path(ap.parse_args().data)
Z = np.load(D / "roundV_perclip.npz")
TASK = np.array(json.loads((D / "clip_tasks.json").read_text())["part2_test_tasks_eval_order"])
REF = json.loads((D / "roundV.json").read_text())
runs = {}
for k in Z.files:
    inp, s, kind = k.split("|")
    runs.setdefault(inp, {}).setdefault(int(s[1:]), {})[kind] = Z[k].astype(np.float64)


def summ(A, B):
    """paired A - B; A, B: {seed: per-clip ADE}"""
    s = sorted(set(A) & set(B))
    Dm = np.stack([A[k] - B[k] for k in s])
    d = Dm.mean(1)
    h = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    r = np.random.default_rng(0)
    clip = [Dm[r.integers(0, len(s), len(s))][:, r.integers(0, Dm.shape[1], Dm.shape[1])].mean() for _ in range(4000)]
    ut = np.unique(TASK); idx = [np.flatnonzero(TASK == t) for t in ut]
    r = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = Dm[r.integers(0, len(s), len(s))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in r.integers(0, len(ut), len(ut))])].mean())
    clip = np.array(clip)
    p = 2 * min((clip <= 0).mean(), (clip >= 0).mean())
    return dict(d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(clip, 2.5)), float(np.percentile(clip, 97.5))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))], p_boot=float(p))


ci = lambda c: "[%+.2f, %+.2f]" % tuple(c)  # noqa: E731
worst = 0.0
for inp, per in runs.items():
    print(f"== {inp} (normal ADE {np.mean([v['normal'].mean() for v in per.values()]):.2f})")
    for kind in sorted(next(iter(per.values()))):
        if kind == "normal":
            continue
        r = summ({s: v[kind] for s, v in per.items()}, {s: v["normal"] for s, v in per.items()})
        ref = REF[inp][kind]
        worst = max(worst, abs(r["d"] - ref["d"]), *[abs(a - b) for a, b in zip(r["clip_ci"], ref["clip_ci"])])
        print("  %-16s %+7.2f seed %s clip %s task %s" % (kind, r["d"], ci(r["seed_ci"]), ci(r["clip_ci"]), ci(r["task_ci"])))
    rows = []
    for f in ("0.1", "0.25"):
        for reg in ("change", "hand", "lowchange"):
            r = summ({s: v[f"{reg}@{f}"] for s, v in per.items()}, {s: v[f"random@{f}"] for s, v in per.items()})
            rows.append((f"{reg}@{f} - random@{f}", r))
    ps = np.array([r["p_boot"] for _, r in rows]); order = np.argsort(ps)
    holm = np.empty_like(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - rank))); holm[i] = run
    for (name, r), hp in zip(rows, holm):
        print("  %-26s %+6.2f clip %s task %s  p %.3f  Holm %.3f" % (name, r["d"], ci(r["clip_ci"]), ci(r["task_ci"]), r["p_boot"], hp))
print("largest deviation from roundV.json: %.4f mm" % worst)
