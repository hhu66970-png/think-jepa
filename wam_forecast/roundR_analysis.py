"""Round R (second dataset): pre-registered hypotheses H1-H3 in P1.
Primary cells: 131 and 57 tokens x {ViT-L, ViT-g}; 309 tokens descriptive.
Run with the round-R environment (WAM_FEAT_ROOT / WAM_RUN_ROOT pointing at feats_R / runs_R).
"""
import json

import numpy as np

from common import RUN_ROOT, write_json
from pairs import load
from phaseA_analysis import paired

ARMS = ["kbsmpa", "tomepa", "pitomepa", "caps2pa"]
BACK = {"ViT-L": (RUN_ROOT / "test_main", "", ["L9", "L12", "L15"]),
        "ViT-g": (RUN_ROOT / "test_vitg", "vitg_", ["gL9", "gL12", "gL15"])}
out = {}
for name, (root, pre, buds) in BACK.items():
    for b in buds:
        R = {a: load(root, f"{pre}{a}_{b}/vispose_k1") for a in ARMS}
        cell = dict(ade={a: float(np.mean([v[0] for v in R[a].values()])) for a in ARMS if R[a]})
        for a, ref in (("kbsmpa", "tomepa"), ("caps2pa", "tomepa"), ("caps2pa", "pitomepa"), ("caps2pa", "kbsmpa")):
            cell[f"{a}-{ref}"] = paired(R[a], R[ref]) if R[a] and R[ref] else None
        out[f"{name}/{b}"] = cell
        print(f"[{name} {b}] " + "  ".join(f"{a} {v:.2f}" for a, v in cell["ade"].items()))
        for k, p in cell.items():
            if k != "ade" and p:
                print(f"     {k:18s} d {p['d']:+.2f} seed [{p['seed_ci'][0]:+.2f},{p['seed_ci'][1]:+.2f}] "
                      f"clip [{p['clip_ci'][0]:+.2f},{p['clip_ci'][1]:+.2f}] wins {p['wins']}/{p['n']}")
dense = load(RUN_ROOT / "test_main", "dense/vispose_k1")
pose = load(RUN_ROOT / "test_main", "none/pose_k1")
desc = {}
if dense and pose:
    P = np.load(__import__("common").FEAT_ROOT / "poses.npz")
    copylast = float((np.linalg.norm(P["future"] - P["past"][:, -1:], axis=-1).mean(axis=(1, 2)) * 1000)[P["is_test"]].mean())
    dn = float(np.mean([v[0] for v in dense.values()]))
    ref = min(copylast, float(np.mean([v[0] for v in pose.values()])))
    desc = dict(dense=dn, copy_last=copylast, pose_only=float(np.mean([v[0] for v in pose.values()])), VG=ref - dn,
                CG={b: out[f"ViT-L/{b}"]["ade"].get("kbsmpa", np.nan) - dn for b in ("L9", "L12", "L15")})
    print("DESCRIPTIVE:", json.dumps(desc))

prim = ["ViT-L/L12", "ViT-L/L15", "ViT-g/gL12", "ViT-g/gL15"]


def get(c, k):
    return out[c][k]


verdict = {}
try:
    h1 = [get(c, "kbsmpa-tomepa") for c in prim]
    verdict["H1"] = all(p["d"] < 0 for p in h1) and sum(p["seed_ci"][1] < 0 for p in h1) >= 3
    h2 = []
    for c in prim:
        ref = "tomepa" if out[c]["ade"]["tomepa"] <= out[c]["ade"]["pitomepa"] else "pitomepa"
        h2.append((c, ref, get(c, f"caps2pa-{ref}")))
    verdict["H2_refs"] = {c: r for c, r, _ in h2}
    verdict["H2"] = (all(p["d"] < 0 for _, _, p in h2)
                     and all(p["clip_ci"][1] < 0 for c, _, p in h2 if c.startswith("ViT-L"))
                     and all(p["seed_ci"][1] < 0 for c, _, p in h2 if c.startswith("ViT-g")))
    h3 = [get(c, "caps2pa-kbsmpa") for c in prim]
    verdict["H3"] = all(p["d"] < 0 for p in h3) and sum(p["seed_ci"][1] < 0 for p in h3) >= 3
    verdict["decision"] = ("method paper defensible (<=131 tokens, RoPE video encoders)" if verdict["H2"] and verdict["H3"]
                           else "RoPE-aware matching + PA as secondary contribution of the analysis paper"
                           if verdict["H2"] else "analysis paper (plan 2); negative replication reported")
    print("VERDICT:", json.dumps(verdict))
except (KeyError, TypeError) as e:
    print("not evaluable yet:", repr(e))
write_json(dict(cells=out, descriptive=desc, verdict=verdict), RUN_ROOT / "roundR.json")
