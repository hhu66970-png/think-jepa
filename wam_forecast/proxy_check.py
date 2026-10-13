"""Proxy check on common ground (descriptive): do allocation statistics track forecasting error once the arm set and the
split are held fixed? Reads corr_test.json (mergers with PA, test, 309 tokens) and corr_nopa_family.json (mergers without
PA, test, 309 tokens) and reports Spearman correlations of Gini and hand fidelity with the change in ADE
(a) for the PA family without the six group cap arms, and (b) for the five arms run in both families on the same split.
  python proxy_check.py --data ../../data -> <data>/proxy_check.json"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
pa = {r["arm"]: (r["gini"], r["fid_hand"], r["d"]) for r in json.loads((DATA / "corr_test.json").read_text())["L9"]["rows"]}
nopa = {r[0].replace("_L9", ""): (r[1], r[2], r[3]) for r in json.loads((DATA / "corr_nopa_family.json").read_text())["test_L9"]["rows"]}


def sp(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def stats(rows):
    g, h, d = (np.array([r[i] for r in rows]) for i in range(3))
    rng = np.random.default_rng(0)
    out = {}
    for nm, v in (("gini", g), ("fid_hand", h)):
        rho = sp(v, d)
        p = float(np.mean([abs(sp(v, rng.permutation(d))) >= abs(rho) - 1e-12 for _ in range(5000)]))
        out[nm] = dict(rho=rho, p=p)
    return out


no_cap = {a: v for a, v in pa.items() if not a.startswith("cap")}
common = [a for a in ("kbsm", "wam", "handmult", "mrecv15", "kbsmlm") if a in nopa and a + "pa" in pa]
res = {"pa_all": dict(n=len(pa), **stats(list(pa.values()))),
       "pa_without_caps": dict(n=len(no_cap), arms=sorted(no_cap), **stats(list(no_cap.values()))),
       "common_arms": common,
       "common_pa": dict(n=len(common), **stats([pa[a + "pa"] for a in common])),
       "common_nopa": dict(n=len(common), **stats([nopa[a] for a in common])),
       "pa_effect_on_gini_and_ade": {a: dict(gini_nopa=nopa[a][0], gini_pa=pa[a + "pa"][0]) for a in common}}
(DATA / "proxy_check.json").write_text(json.dumps(res, indent=1))
for k in ("pa_all", "pa_without_caps", "common_pa", "common_nopa"):
    v = res[k]
    print(f"{k:16s} n={v['n']:2d}  Gini rho {v['gini']['rho']:+.2f} (p {v['gini']['p']:.3f})  hand fidelity rho {v['fid_hand']['rho']:+.2f} (p {v['fid_hand']['p']:.3f})")
