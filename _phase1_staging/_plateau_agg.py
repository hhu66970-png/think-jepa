#!/usr/bin/env python3
"""Aggregate ext30 downstream ADE: per-seed plateau = mean of last-10-epoch
val_avg_dist; report PiToMe mean +/- std and the paired Delta vs dense (and vs
K-BSM r0.15) over the common seeds. Reads metrics.json per run dir."""
import json, os, statistics as st, sys

BASE = "/root/autodl-tmp/thinkjepa-work/ThinkJEPA/outputs/downstream_ext30_20260531"
SEEDS = [42, 43, 44, 45, 46, 47]
LASTN = 10

def plateau(run_dir):
    mp = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mp):
        return None, "no metrics.json"
    d = json.load(open(mp))
    eps = d.get("epochs", [])
    vals = [e.get("val_avg_dist") for e in eps if e.get("val_avg_dist") is not None]
    if len(vals) < LASTN:
        return None, f"only {len(vals)} epochs"
    last = vals[-LASTN:]
    return st.mean(last), f"n={len(vals)} last{LASTN}_mean"

def best_ade(run_dir):
    mp = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mp): return None
    d = json.load(open(mp))
    return (d.get("best") or {}).get("val_avg_dist")

def collect(prefix):
    out = {}
    for s in SEEDS:
        p = os.path.join(BASE, f"{prefix}__s{s}")
        pl, note = plateau(p)
        out[s] = (pl, best_ade(p), note)
    return out

def summarize(name, d):
    vals = [v[0] for v in d.values() if v[0] is not None]
    print(f"\n[{name}] per-seed plateau (last{LASTN} val_avg_dist):")
    for s in SEEDS:
        pl, be, note = d[s]
        print(f"   s{s}: plateau={('%.5f'%pl) if pl is not None else 'MISSING'} best={('%.5f'%be) if be else 'NA'} ({note})")
    if vals:
        m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"   -> mean={m:.5f} std={sd:.5f} n={len(vals)}")
    return d

def paired_delta(a, b, na, nb):
    # Delta = a - b per seed on common seeds with both present
    common = [s for s in SEEDS if a[s][0] is not None and b[s][0] is not None]
    if not common:
        print(f"\n[paired {na} - {nb}] no common seeds")
        return
    deltas = [a[s][0] - b[s][0] for s in common]
    m = st.mean(deltas); sd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
    print(f"\n[paired Delta: {na} - {nb}] common seeds={common}")
    for s, dl in zip(common, deltas):
        print(f"   s{s}: {na}={a[s][0]:.5f} {nb}={b[s][0]:.5f} delta={dl:+.5f}")
    print(f"   -> mean_delta={m:+.5f} std={sd:.5f}  ({'PiToMe WORSE' if m>0 else 'PiToMe BETTER/eq'} vs {nb})")

pit = summarize("PITOME_r015", collect("PITOME_r015"))
den = summarize("dense", collect("dense"))
bsm = summarize("BSM_r015 (K-BSM)", collect("BSM_r015"))
paired_delta(pit, den, "PITOME_r015", "dense")
paired_delta(pit, bsm, "PITOME_r015", "BSM_r015")
