"""Round K: compact-token versus restore interface, pooled 5-fold CV over the 1620 non-selection clips.
  python roundK_analysis.py  -> runs/phaseA/roundK.json and a printed summary
"""
import json
from pathlib import Path

import numpy as np

from aggregate import tcrit
from common import RUN_ROOT, W, write_json

ARMS = ["kbsm", "kbsmpa", "wam", "handmult", "caps2pa"]
REF = {"wam": "kbsm", "handmult": "kbsm", "kbsmpa": "kbsm", "caps2pa": "kbsmpa"}


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
            o = np.argsort(rows); res[s] = (rows[o], ade[o])
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
    return dict(n=len(s), A=float(np.mean([A[k][1].mean() for k in s])), B=float(np.mean([B[k][1].mean() for k in s])),
                d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))])


R = {a: {"restore": pooled(f"{a}_L9", "vis"), "compact": pooled(f"{a}_L9", "vis_compact")} for a in ARMS}
out = {"per_arm": {}, "vs_ref": {}}
for a in ARMS:
    out["per_arm"][a] = diff(R[a]["compact"], R[a]["restore"])
for a, r in REF.items():
    out["vs_ref"][a] = {i: diff(R[a][i], R[r][i]) for i in ("restore", "compact")}
f = lambda p: "-" if not p else f"{p['d']:+.2f} seed[{p['seed_ci'][0]:+.2f},{p['seed_ci'][1]:+.2f}] clip[{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] (A {p['A']:.2f}, B {p['B']:.2f}, n{p['n']})"
print("compact - restore per arm (309 tokens, CV 1620 clips):")
for a, p in out["per_arm"].items():
    print(f"  {a:9s}", f(p))
print("arm - reference under each interface:")
agree = []
for a, v in out["vs_ref"].items():
    print(f"  {a:9s} vs {REF[a]:7s} restore {f(v['restore'])}\n  {'':9s}    {'':7s} compact {f(v['compact'])}")
    if v["restore"] and v["compact"]:
        agree.append(np.sign(v["restore"]["d"]) == np.sign(v["compact"]["d"]))
out["sign_agreement"] = f"{sum(agree)}/{len(agree)}"
print("sign agreement:", out["sign_agreement"])
write_json(out, W / "runs" / "phaseA" / "roundK.json")
