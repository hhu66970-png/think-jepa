#!/usr/bin/env python
"""Paired Δ vs dense + t-test for D-temporal (velocity_error, accel_error),
same plateau(last10) machinery as agg_aligned.py. The aligned agg only printed
point estimates; this adds significance so we know if WAM's temporal win is real.
"""
import json, os, math
ROOT = "outputs/downstream_ext30_aligned_20260602"
CFGS = [("dense", "dense"), ("A_l8_r025", "方案A r0.25"), ("BSM_r015", "K-BSM r0.15"),
        ("PITOME_r015", "PiToMe r0.15"), ("WAM_r015", "WAM r0.15")]
SEEDS = [42, 43, 44, 45, 46, 47]; LASTK = 10
TCRIT = {5: 2.015, 4: 2.132, 3: 2.353}  # one-sided 95%


def epsT(c, s):
    p = f"{ROOT}/{c}__s{s}/metrics.json"
    if not os.path.exists(p):
        return []
    return json.load(open(p)).get("epochs", [])


def platT(c, s, k):
    e = epsT(c, s)
    v = [x["temporal"][k] for x in e[-LASTK:]
         if x.get("temporal") and x["temporal"].get(k) is not None]
    return sum(v) / len(v) if v else None


def mn(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]; m = mn(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")


for key, name in [("velocity_error", "VEL"), ("accel_error", "ACCEL")]:
    print(f"\n=== {name} aligned plateau + pairedΔ vs dense + one-sided t (n<=6) ===")
    dp = {s: platT("dense", s, key) for s in SEEDS}; dm = mn(list(dp.values()))
    print(f"{'dense':14s} mean={dm:.5f}  --")
    for c, lab in CFGS:
        if c == "dense":
            continue
        d = [platT(c, s, key) - dp[s] for s in SEEDS
             if platT(c, s, key) is not None and dp.get(s) is not None]
        m = mn([platT(c, s, key) for s in SEEDS])
        dmd, dsd = mn(d), sd(d); se = dsd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = dmd / se if se and se > 0 else float("nan")
        up = dmd + TCRIT.get(len(d) - 1, 2.0) * se
        sig = "SIG✓" if (t == t and abs(t) > TCRIT.get(len(d) - 1, 2.0) and dmd < 0) else ""
        print(f"{lab:14s} mean={m:.5f} Δ={dmd:+.5f} t={t:+.2f} UP95={up:+.5f}({100*up/dm:+.1f}%) {sig}")
