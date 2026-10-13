"""Gini vs dADE in the older no-PA arm family (reference kbsm), val (runs/select) and test."""
import json

import numpy as np

from common import FEAT_ROOT, RUN_ROOT, W, write_json
from pairs import load

G = {}
for f in ("alloc_full.json", "alloc_v3.json", "alloc_p4.json", "alloc_parel.json"):
    for k, v in json.loads((FEAT_ROOT / f).read_text())["configs"].items():
        G.setdefault(k, v)


def sp(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


rng = np.random.default_rng(0)
out = {}
for split, root in (("val", RUN_ROOT / "select"), ("test", RUN_ROOT / "test_main")):
    for b in ("L9", "L12", "s25"):
        ref = load(root, f"kbsm_{b}/vis")
        rows = []
        for c, g in G.items():
            if not c.endswith("_" + b) or c.startswith("kbsm_") or "pa_" in c:
                continue
            R = load(root, f"{c}/vis")
            s = sorted(set(R) & set(ref))
            if len(s) < 3:
                continue
            rows.append((c, g["gini"], g["fid_hand"], float(np.mean([R[k][0] - ref[k][0] for k in s])), len(s)))
        if f"kbsm_{b}" not in G or not ref:
            continue
        rows.append((f"kbsm_{b}", G[f"kbsm_{b}"]["gini"], G[f"kbsm_{b}"]["fid_hand"], 0.0, len(ref)))
        if len(rows) < 5:
            continue
        y = np.array([r[3] for r in rows])
        res = {}
        for nm, j in (("gini", 1), ("fid_hand", 2)):
            v = np.array([r[j] for r in rows])
            rho = sp(v, y)
            p = float(np.mean([abs(sp(v, rng.permutation(y))) >= abs(rho) - 1e-12 for _ in range(5000)]))
            res[nm] = dict(rho=rho, p=p)
        out[f"{split}_{b}"] = dict(n=len(rows), stats=res, rows=rows)
        print(f"no-PA family {split} {b}: n={len(rows)}  Gini rho {res['gini']['rho']:+.2f} (p {res['gini']['p']:.3f})"
              f"  fid_hand rho {res['fid_hand']['rho']:+.2f} (p {res['fid_hand']['p']:.3f})")
        if b == "L9":
            for r in sorted(rows, key=lambda r: r[1]):
                print(f"    {r[0]:>14s} gini {r[1]:.3f} fid_h {r[2]:.4f} d {r[3]:+.2f} (n={r[4]})")
write_json(out, W / "runs" / "phaseA" / "corr_nopa_family.json")
