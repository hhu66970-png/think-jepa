"""Rebuild the paper's paired comparisons from the released raw archive alone (no run tree, no features).

  python rebuild_from_raw.py --data ../../data
expects raw_results.npz, raw_index.json, clip_tasks.json and paper_data2.json in --data. Recomputes every
test comparison of paper_data2.json["cmp"] (video only and current pose, ViT-L and ViT-g; paired means, seed
level t interval, clip x seed and task x seed bootstraps with 4000 replicates and seed 0, as in
paper_data2.py) and the pooled 5-fold CV comparisons at 309 tokens (rounds A2b, K, L, P), and prints the
largest deviation from the released aggregates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
IDX = json.loads((DATA / "raw_index.json").read_text())["runs"]
RAW = np.load(DATA / "raw_results.npz")
TASKS = np.array(json.loads((DATA / "clip_tasks.json").read_text())["part2_test_tasks_eval_order"])
BY_DIR = {}
for k, m in IDX.items():
    BY_DIR.setdefault(k.rsplit("/", 1)[0], []).append((k.replace("/", "|"), m))


def load(run_dir):
    """run_dir such as runs/test_main/kbsm_L9/vis -> {seed: (ade, perclip)}"""
    return {m["seed"]: (m["ade"], RAW[key].astype(np.float64)) for key, m in BY_DIR.get(run_dir, []) if key in RAW.files}


def load_cv(run_dir):
    """pool the five folds per seed: runs/cv/<cfg>/<input>_cv{f}; per-clip order = sorted evaluation rows"""
    per = {}
    for f in range(5):
        for key, m in BY_DIR.get(f"{run_dir}_cv{f}", []):
            per.setdefault(m["seed"], {})[f] = (RAW[key + "|rows"], RAW[key].astype(np.float64))
    out = {}
    for s, v in per.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)])
            ade = np.concatenate([v[f][1] for f in range(5)])[np.argsort(rows)]
            out[s] = (float(ade.mean()), ade)
    return out


def paired(A, B, tasks=None, nboot=4000):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    d = np.array([A[k][0] - B[k][0] for k in s])
    half = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    D = np.stack([A[k][1] - B[k][1] for k in s])
    rng = np.random.default_rng(0)
    clip = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(nboot)]
    res = dict(n=len(s), d=float(d.mean()), seed_ci=[float(d.mean() - half), float(d.mean() + half)],
               clip_ci=[float(np.percentile(clip, 2.5)), float(np.percentile(clip, 97.5))])
    if tasks is not None and D.shape[1] == len(tasks):
        ut = np.unique(tasks)
        idx = [np.flatnonzero(tasks == t) for t in ut]
        rng = np.random.default_rng(0)
        tb = []
        for _ in range(nboot):
            Ds = D[rng.integers(0, len(s), len(s))]
            tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
        res["task_ci"] = [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))]
    return res


def ci(c):
    return "[%+.2f, %+.2f]" % tuple(c)


REF = json.loads((DATA / "paper_data2.json").read_text())["cmp"]


def run_dirs(name):
    """comparison key of paper_data2.json -> (run dir A, run dir B)"""
    if name.startswith("vitg/"):
        x, y = name[5:].split("-")
        return f"runs/test_vitg/vitg_{x}_gL9/vis", f"runs/test_vitg/vitg_{y}_gL9/vis"
    if name == "P1/dense-pose":
        return "runs/test_main/dense/vispose_k1", "runs/test_main/none/pose_k1"
    if name.startswith("P1/"):
        pair, b = name[3:].split("/")
        x, y = pair.split("-")
        d = lambda a: "runs/test_main/dense/vispose_k1" if a == "dense" else f"runs/test_main/{a}_{b}/vispose_k1"  # noqa: E731
        return d(x), d(y)
    pair, b = name.split("/")
    x, y = pair.split("-")
    return f"runs/test_main/{x}_{b}/vis", f"runs/test_main/{y}_{b}/vis"


worst, n_done, n_skip = 0.0, 0, 0
print("%-30s %7s %17s %17s %17s  %s" % ("test comparison", "delta", "seed CI", "clip CI", "task CI", "max dev"))
for name, ref in REF.items():
    if ref is None:
        n_skip += 1
        continue
    da, db = run_dirs(name)
    r = paired(load(da), load(db), TASKS)
    if r is None:
        print("%-30s missing runs" % name)
        n_skip += 1
        continue
    keys = [k for k in ("seed_ci", "clip_ci", "task_ci") if k in ref and k in r]
    dev = max([abs(r["d"] - ref["d"])] + [abs(u - v) for k in keys for u, v in zip(r[k], ref[k])])
    worst = max(worst, dev)
    n_done += 1
    print("%-30s %+7.2f %17s %17s %17s  %.4f" % (name, r["d"], ci(r["seed_ci"]), ci(r["clip_ci"]),
                                                   ci(r["task_ci"]) if "task_ci" in r else "-", dev))
print("recomputed %d comparisons (%d entries without runs); largest deviation: %.4f mm" % (n_done, n_skip, worst))

print("\npooled 5-fold CV at 309 tokens (1620 clips): arm minus K-BSM+PA")
for inp in ("vis", "vis_compact"):
    ref = load_cv(f"runs/cv/kbsmpa_L9/{inp}")
    for arm in ("kbsm", "wampa", "handmultpa", "tomepa", "pitomepa", "pitomeopa", "caps2pa", "wamshufpa", "wamshuffpa", "waminvpa", "wamcomppa", "randrankpa"):
        r = paired(load_cv(f"runs/cv/{arm}_L9/{inp}"), ref)
        if r:
            print("  %-12s %-11s %+.2f  clip %s  seeds %d" % (inp, arm, r["d"], ci(r["clip_ci"]), r["n"]))
