"""Phase B analysis: vision gain (VG) and compression gap (CG) per task setting and stratum.

  python phaseB_analysis.py [val|test] [extra_arm ...]

Settings: P32 / P4 / P1 (pose history of 32 / 4 / 1 frames, input vispose vs pose) and P0
(no pose: vis vs copy-last). VG = E(no-vision reference) - E(dense vision);
CG_b = E(compressed_b) - E(dense). Seed-paired 95 % t intervals. Strata from clip_strata.npz
with thresholds from TRAIN quartiles/tertiles.
"""
import json
import sys
from pathlib import Path

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

split = sys.argv[1] if len(sys.argv) > 1 else "val"
extra = sys.argv[2:]
root = RUN_ROOT / ("select" if split == "val" else "test_main")
sfx = "_val" if split == "val" else ""
Z = np.load(FEAT_ROOT / "clip_strata.npz")
tr = ~Z["is_test"]
mq = np.nanpercentile(Z["motion"][tr], [25, 50, 75])
hq = np.nanpercentile(Z["horizon_s"][tr], [100 / 3, 200 / 3])
STRATA = {
    "all": lambda r: np.ones(len(r), bool),
    "motion_q1(low)": lambda r: Z["motion"][r] < mq[0],
    "motion_q4(high)": lambda r: Z["motion"][r] >= mq[2],
    "horizon_short": lambda r: Z["horizon_s"][r] < hq[0],
    "horizon_long": lambda r: Z["horizon_s"][r] >= hq[1],
}


def load(d):
    out = {}
    for mf in (root / d).glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        f = mf.parent
        ade = np.load(f / "perclip_ade.npy")
        if (f / "eval_rows.npy").exists():
            rows = np.load(f / "eval_rows.npy")
            CANON.setdefault(len(rows), rows)
            assert np.array_equal(CANON[len(rows)], rows), f"eval rows differ: {f}"
        else:                       # older run: every arm of a split evaluates the same rows
            rows = None
        step = np.load(f / "perclip_step.npy").astype(np.float32) if (f / "perclip_step.npy").exists() else None
        out[m["seed"]] = dict(rows=rows, ade=ade, step=step)
    return out


CANON = {}


def fill_rows(*groups):
    for G in groups:
        for v in G.values():
            if v["rows"] is None:
                v["rows"] = CANON[len(v["ade"])]


def copylast(rows_ref):
    """Copy-last per-clip ADE = motion; identical for every seed."""
    return {s: dict(rows=v["rows"], ade=Z["motion"][v["rows"]], step=None) for s, v in rows_ref.items()}


def diff(A, B, stratum):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    d = []
    for k in s:
        assert np.array_equal(A[k]["rows"], B[k]["rows"])
        m = STRATA[stratum](A[k]["rows"])
        d.append(A[k]["ade"][m].mean() - B[k]["ade"][m].mean())
    d = np.array(d)
    half = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    return dict(d=float(d.mean()), ci=[float(d.mean() - half), float(d.mean() + half)], n=len(s),
                nclips=int(STRATA[stratum](A[s[0]]["rows"]).sum()))


def mean_ade(A, stratum="all"):
    return float(np.mean([v["ade"][STRATA[stratum](v["rows"])].mean() for v in A.values()])) if A else float("nan")


settings = {}
for k in (32, 4, 1):
    T = sfx if k == 32 else f"_k{k}{sfx}"
    settings[f"P{k}"] = dict(ref=load(f"none/pose{T}"), dense=load(f"dense/vispose{T}"),
                             **{c: load(f"{c}/vispose{T}") for c in ["kbsmpa_L9", "kbsmpa_L12"] + extra})
dv = load(f"dense/vis{sfx}")
fill_rows(dv)
settings["P0"] = dict(ref=copylast(dv), dense=dv,
                      **{c: load(f"{c}/vis{sfx}") for c in ["kbsmpa_L9", "kbsmpa_L12"] + extra})

for S in settings.values():
    fill_rows(*S.values())
report = {}
print(f"split={split}; motion quartiles {mq.round(1)}, horizon tertiles {hq.round(2)} s")
for name, S in settings.items():
    if not S["dense"] or not S["ref"]:
        print(f"{name}: missing runs"); continue
    rep = {}
    for st in STRATA:
        row = dict(ref=mean_ade(S["ref"], st), dense=mean_ade(S["dense"], st), VG=diff(S["ref"], S["dense"], st))
        for c in S:
            if c not in ("ref", "dense") and S[c]:
                row[c] = mean_ade(S[c], st)
                row[f"CG_{c}"] = diff(S[c], S["dense"], st)
        rep[st] = row
    # per-step VG / CG (all clips)
    if S["ref"] and all(v["step"] is not None for G in (S["ref"], S["dense"]) for v in G.values()):
        sd = sorted(set(S["ref"]) & set(S["dense"]))
        rep["VG_per_step"] = np.mean([S["ref"][s]["step"].mean(0) - S["dense"][s]["step"].mean(0) for s in sd], 0).tolist()
    for c in S:
        if c not in ("ref", "dense") and S[c] and all(v["step"] is not None for v in S[c].values()):
            sd = sorted(set(S[c]) & set(S["dense"]))
            rep[f"CG_per_step_{c}"] = np.mean([S[c][s]["step"].mean(0) - S["dense"][s]["step"].mean(0) for s in sd], 0).tolist()
    report[name] = rep
    print(f"== {name}")
    for st in STRATA:
        r = rep[st]
        vg = r["VG"]
        line = f"  {st:>16s} ref {r['ref']:6.2f} dense {r['dense']:6.2f} VG {vg['d']:+6.2f} [{vg['ci'][0]:+.2f},{vg['ci'][1]:+.2f}]" if vg else f"  {st:>16s} (no VG)"
        for c in r:
            if c.startswith("CG_") and r[c]:
                line += f" | {c[3:]} {r[c[3:]]:6.2f} CG {r[c]['d']:+5.2f} [{r[c]['ci'][0]:+.2f},{r[c]['ci'][1]:+.2f}]"
        if vg:
            line += f" (clips {vg['nclips']})"
        print(line)
out = W / "runs" / "phaseB"
out.mkdir(parents=True, exist_ok=True)
write_json(report, out / f"phaseB_{split}.json")
