"""Phase A analysis: validation selection (A1) and group-size-balance vs ADE (A3).

  python phaseA_analysis.py select            # A1: val table + pre-registered choice of c*
  python phaseA_analysis.py corr <split>      # A3: Gini vs ADE across PA arms (split: val | test)

Val runs: runs/select/<arm>_<b>/vis_val ; test runs: runs/test_main/<arm>_<b>/vis.
Allocation metrics: feats/alloc_phaseA_<b>.json (train clips).
"""
import json
import sys
from pathlib import Path

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json
from pairs import load

CAPS = ["capc1pa", "capc2pa", "capc4pa", "caps2pa", "caps4pa", "caps8pa"]
ALL_PA = ["kbsmpa"] + CAPS + ["rrecv15pa", "mrecv15pa", "handmultpa", "wampa", "kbsmlmpa", "b2pa"]
OUT = W / "runs" / "phaseA"


def paired(A, B):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    d = np.array([A[k][0] - B[k][0] for k in s])
    half = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    D = np.stack([A[k][2] - B[k][2] for k in s])
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean()
          for _ in range(4000)]
    return dict(n=len(s), A=float(np.mean([A[k][0] for k in s])), B=float(np.mean([B[k][0] for k in s])),
                d=float(d.mean()), seed_ci=[float(d.mean() - half), float(d.mean() + half)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                wins=int((d < 0).sum()),
                fde_d=float(np.mean([A[k][1] - B[k][1] for k in s])))


def runs(split, arm, b):
    if split == "val":
        return load(RUN_ROOT / "select", f"{arm}_{b}/vis_val")
    return load(RUN_ROOT / "test_main", f"{arm}_{b}/vis")


def select():
    res = {}
    for b in ("L9", "L12"):
        ref = runs("val", "kbsmpa", b)
        for arm in CAPS + ["rrecv15pa"]:
            p = paired(runs("val", arm, b), ref)
            if p:
                res[f"{arm}_{b}"] = p
    print(f"{'arm':>12s} {'budget':>6s} {'ADE':>7s} {'kbsmpa':>7s} {'d':>7s} {'seed CI':>18s} {'clip CI':>18s} wins")
    for k, p in res.items():
        arm, b = k.rsplit("_", 1)
        print(f"{arm:>12s} {b:>6s} {p['A']:7.2f} {p['B']:7.2f} {p['d']:+7.2f} "
              f"[{p['seed_ci'][0]:+6.2f},{p['seed_ci'][1]:+6.2f}] "
              f"[{p['clip_ci'][0]:+6.2f},{p['clip_ci'][1]:+6.2f}] {p['wins']}/{p['n']}")
    avg = {a: np.mean([res[f"{a}_{b}"]["A"] for b in ("L9", "L12")]) for a in CAPS
           if all(f"{a}_{b}" in res for b in ("L9", "L12"))}
    if avg:
        c = min(avg, key=avg.get)
        print("PRE-REGISTERED SELECTION c* =", c, {k: round(v, 2) for k, v in avg.items()})
        res["selected"] = c
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(res, OUT / "select_val.json")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def corr(split):
    out = {}
    for b in ("L9", "L12", "s25"):
        f = FEAT_ROOT / f"alloc_phaseA_{b}.json"
        if not f.exists():
            continue
        al = json.loads(f.read_text())["configs"]
        ref = runs(split, "kbsmpa", b)
        rows = []
        for arm in ALL_PA:
            c = f"{arm}_{b}"
            R = runs(split, arm, b)
            if c not in al or not R:
                continue
            p = paired(R, ref) if arm != "kbsmpa" else dict(d=0.0, A=float(np.mean([v[0] for v in R.values()])), n=len(R))
            if p is None:
                continue
            rows.append(dict(arm=arm, ade=p["A"], d=p["d"], n=p["n"], **{k: al[c][k] for k in
                        ("gini", "gs_max", "gs_var", "gs_median", "gs_tok", "fid_hand", "fid_bg")}))
        if len(rows) < 4:
            continue
        rng = np.random.default_rng(0)
        stats = {}
        for key in ("gini", "gs_max", "gs_tok", "fid_hand", "fid_bg"):
            x = np.array([r[key] for r in rows]); y = np.array([r["d"] for r in rows])
            rho = spearman(x, y)
            null = [spearman(x, rng.permutation(y)) for _ in range(5000)]
            stats[key] = dict(rho=rho, p=float(np.mean(np.abs(null) >= abs(rho) - 1e-12)))
        # capacity arms only (the causal manipulation): same similarity rule, only balance changes
        cap = [r for r in rows if r["arm"] in CAPS + ["kbsmpa"]]
        if len(cap) >= 4:
            x = np.array([r["gini"] for r in cap]); y = np.array([r["d"] for r in cap])
            rho = spearman(x, y)
            null = [spearman(x, rng.permutation(y)) for _ in range(5000)]
            stats["gini_caps_only"] = dict(rho=rho, p=float(np.mean(np.abs(null) >= abs(rho) - 1e-12)), n=len(cap))
        out[b] = dict(rows=rows, stats=stats)
        print(f"== {b} ({split}), n arms = {len(rows)}")
        for r in sorted(rows, key=lambda r: r["gini"]):
            print(f"  {r['arm']:>11s} gini {r['gini']:.3f} max {r['gs_max']:6.1f} tok-mean {r['gs_tok']:6.1f} "
                  f"fid_h {r['fid_hand']:.4f} fid_bg {r['fid_bg']:.4f}  ADE {r['ade']:.2f}  d {r['d']:+.2f} (n={r['n']})")
        for k, s in stats.items():
            print(f"  Spearman({k}, dADE) = {s['rho']:+.2f}  perm p = {s['p']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(out, OUT / f"corr_{split}.json")


def cv(arm="caps2pa", ref="kbsmpa", budgets="L9,L12,s25", inp="vis"):
    """A2b: pooled 5-fold CV over the 1620 non-selection train clips (runs/cv/<cfg>/vis_cv<f>)."""
    res = {}
    for b in budgets.split(","):
        per = {}
        for m in (arm, ref):
            seeds = {}
            for f in range(5):
                for mf in (RUN_ROOT / "cv" / f"{m}_{b}" / f"{inp}_cv{f}").glob("s*/metrics.json"):
                    sd = json.loads(mf.read_text())["seed"]
                    seeds.setdefault(sd, {})[f] = (np.load(mf.parent / "eval_rows.npy"),
                                                  np.load(mf.parent / "perclip_ade.npy"))
            per[m] = {sd: v for sd, v in seeds.items() if len(v) == 5}
        common = sorted(set(per[arm]) & set(per[ref]))
        if len(common) < 2:
            print(b, "not enough complete seeds", len(common)); continue
        D = []
        for sd in common:
            ra = np.concatenate([per[arm][sd][f][0] for f in range(5)])
            rb = np.concatenate([per[ref][sd][f][0] for f in range(5)])
            assert np.array_equal(ra, rb)
            D.append(np.concatenate([per[arm][sd][f][1] for f in range(5)]) -
                     np.concatenate([per[ref][sd][f][1] for f in range(5)]))
        D = np.stack(D)
        d = D.mean(1)
        half = tcrit(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rng = np.random.default_rng(0)
        bo = [D[rng.integers(0, len(d), len(d))][:, rng.integers(0, D.shape[1], D.shape[1])].mean()
              for _ in range(4000)]
        A = np.mean([np.concatenate([per[arm][sd][f][1] for f in range(5)]).mean() for sd in common])
        B = np.mean([np.concatenate([per[ref][sd][f][1] for f in range(5)]).mean() for sd in common])
        res[b] = dict(n_seeds=len(d), n_clips=int(D.shape[1]), A=float(A), B=float(B), d=float(d.mean()),
                      seed_ci=[float(d.mean() - half), float(d.mean() + half)],
                      clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                      wins=int((d < 0).sum()))
        r = res[b]
        print(f"{arm}-{ref} {b}: clips={r['n_clips']} seeds={r['n_seeds']} A={A:.2f} B={B:.2f} d={r['d']:+.3f} "
              f"seedCI=[{r['seed_ci'][0]:+.3f},{r['seed_ci'][1]:+.3f}] clipCI=[{r['clip_ci'][0]:+.3f},"
              f"{r['clip_ci'][1]:+.3f}] wins={r['wins']}/{r['n_seeds']}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(res, OUT / f"cv_{arm}_vs_{ref}{'' if inp == 'vis' else '_' + inp}.json")


if __name__ == "__main__":
    {"select": lambda: select(), "corr": lambda: corr(sys.argv[2]), "cv": lambda: cv(*sys.argv[2:])}[sys.argv[1]]()
