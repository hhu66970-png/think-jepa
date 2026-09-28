"""Round M: P1 (video + current pose) comparison of BPM (caps2pa) with K-BSM+PA, ToMe+PA,
PiToMe+PA on ViT-L and ViT-g, plus the pre-registered method-paper rule.

  python roundM_analysis.py
"""
import json

import numpy as np

from common import RUN_ROOT, W, write_json
from phaseA_analysis import paired
from pairs import load

ARMS = ["kbsmpa", "tomepa", "pitomepa", "caps2pa"]
BACK = {
    "ViT-L": dict(root=RUN_ROOT / "test_main", pre="", bud=["s25", "L9", "L12", "L15"], dense="dense/vispose_k1"),
    "ViT-g": dict(root=RUN_ROOT / "test_vitg", pre="vitg_", bud=["gs25", "gL9", "gL12", "gL15"], dense=None),
}
out, verdict = {}, {}
for name, B in BACK.items():
    out[name] = {}
    for b in B["bud"]:
        R = {a: load(B["root"], f"{B['pre']}{a}_{b}/vispose_k1") for a in ARMS}
        if not any(R.values()):
            continue
        row = {a: dict(n=len(v), ade=float(np.mean([x[0] for x in v.values()])) if v else None,
                       fde=float(np.mean([x[1] for x in v.values()])) if v else None) for a, v in R.items()}
        if B["dense"]:
            D = load(B["root"], B["dense"])
            row["dense"] = dict(n=len(D), ade=float(np.mean([x[0] for x in D.values()])))
        cmp = {ref: paired(R["caps2pa"], R[ref]) for ref in ("kbsmpa", "tomepa", "pitomepa") if R[ref] and R["caps2pa"]}
        out[name][b] = dict(arms=row, caps2pa_vs=cmp)
        line = "  ".join(f"{a} {row[a]['ade']:.2f}(n{row[a]['n']})" for a in ARMS if row[a]["ade"] is not None)
        if "dense" in row:
            line += f"  dense {row['dense']['ade']:.2f}"
        print(f"[{name} {b}] {line}")
        for ref, p in cmp.items():
            if p:
                print(f"     caps2pa - {ref:8s}: d {p['d']:+.2f} seed [{p['seed_ci'][0]:+.2f},{p['seed_ci'][1]:+.2f}] "
                      f"clip [{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] wins {p['wins']}/{p['n']}")

# pre-registered rule
def ok_mean_below_all(name, budgets):
    return [all(out[name][b]["caps2pa_vs"].get(r) and out[name][b]["caps2pa_vs"][r]["d"] < 0
                for r in ("kbsmpa", "tomepa", "pitomepa")) for b in budgets]

try:
    i = ok_mean_below_all("ViT-L", BACK["ViT-L"]["bud"][:3])
    g = BACK["ViT-g"]["bud"][:3]
    gk = [out["ViT-g"][b]["caps2pa_vs"]["kbsmpa"] for b in g]
    ii_a = all(p["d"] < 0 for p in gk) and sum(p["seed_ci"][1] < 0 for p in gk) >= 2
    ii_b = sum(ok_mean_below_all("ViT-g", g)) >= 2
    cv = json.loads((W / "runs" / "phaseA" / "cv_caps2pa_vs_kbsmpa_vispose_k1.json").read_text())
    iii = all(cv[b]["d"] < 0 for b in cv) and sum(cv[b]["clip_ci"][1] < 0 for b in cv) >= 2
    verdict = dict(i=all(i), i_per_budget=i, ii_vs_kbsmpa=ii_a, ii_vs_others=ii_b, iii_cv=iii,
                   cv=cv, method_paper=bool(all(i) and ii_a and ii_b and iii))
    print("RULE:", json.dumps({k: v for k, v in verdict.items() if k != "cv"}))
    for b, r in cv.items():
        print(f"  CV P1 {b}: d {r['d']:+.2f} clip [{r['clip_ci'][0]:+.2f},{r['clip_ci'][1]:+.2f}] wins {r['wins']}/{r['n_seeds']}")
except (KeyError, TypeError, FileNotFoundError) as e:
    print("rule not evaluable yet:", repr(e))
# round N rule (57 tokens, both backbones)
verdictN = {}
try:
    nl, ng = out["ViT-L"]["L15"]["caps2pa_vs"], out["ViT-g"]["gL15"]["caps2pa_vs"]
    best_l = max(("kbsmpa", "tomepa", "pitomepa"), key=lambda r: nl[r]["d"])   # closest baseline
    best_g = max(("kbsmpa", "tomepa", "pitomepa"), key=lambda r: ng[r]["d"])
    verdictN = dict(below_all_L=all(nl[r]["d"] < 0 for r in nl), below_all_g=all(ng[r]["d"] < 0 for r in ng),
                    best_L=best_l, clip_vs_best_L=nl[best_l]["clip_ci"], best_g=best_g, seed_vs_best_g=ng[best_g]["seed_ci"])
    verdictN["confirmed"] = bool(verdictN["below_all_L"] and verdictN["below_all_g"]
                                 and nl[best_l]["clip_ci"][1] < 0 and ng[best_g]["seed_ci"][1] < 0)
    print("ROUND N RULE:", json.dumps(verdictN))
except (KeyError, TypeError) as e:
    print("round N not evaluable yet:", repr(e))
write_json(dict(results=out, verdict=verdict, verdictN=verdictN), W / "runs" / "phaseA" / "roundM.json")
