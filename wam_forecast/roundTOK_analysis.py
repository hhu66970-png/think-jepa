"""Round TOK (PROTOCOL, tenth review): accuracy against the number of tokens handed to the downstream module (compact
readout), on EgoDex (current pose CV) and Diving48 (official split).  -> runs/phaseA/roundTOK.json"""
import json
import math
import os

import numpy as np

from common import SCHEDULES, W, write_json
import cvstats
from extract import INPUT_TOKENS, composite_counts, composite_target

EGO_C = ["in16f", "in12f", "in8f", "in4f", "kbsmpa_L9", "kbsmpa_L12", "handcurpa_L9", "handcurshufpa_L9", "handcurpa_L12",
         "handcurshufpa_L12", "in16f+kbsmpa_L9", "in16f+kbsmpa_L12", "in12f+kbsmpa_L9", "in12f+kbsmpa_L12",
         "in8f+kbsmpa_L9", "in8f+kbsmpa_L12", "dense@p222", "dense@p244", "in4f@p822", "in8f@p422"]
DV_C = ["in24f", "in16f", "in12f", "in8f", "in4f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25", "kbsmpa_L9", "wampa_L9",
        "wamshufpa_L9", "kbsmpa_L12", "wampa_L12", "wamshufpa_L12", "in16f+kbsmpa_L9", "in16f+wampa_L9",
        "in16f+wamshufpa_L9", "dense@p222", "dense@p244", "in4f@p822", "in8f@p422"]
DV_ROOT = W / "runs_DV2"


def flops_tokens(arm):
    """(encoder FLOPs relative to dense, output tokens) for ViT-L (24 blocks, D 1024; per block 24 N D^2 + 4 N^2 D)."""
    D, L = 1024, 24
    pool = None
    if "@p" in arm:
        arm, p = arm.split("@p")
        pool = 4096 // (int(p[0]) * int(p[1]) * int(p[2]))
    base, rest = (arm.split("+", 1) if "+" in arm else (None, arm))
    if arm == "dense" or arm in INPUT_TOKENS:
        n0, layers, counts = (4096 if arm == "dense" else INPUT_TOKENS[arm]), [], []
    else:
        sched = rest.split("_", 1)[1]
        layers = [int(v) for v in SCHEDULES[sched][0].split(",")]
        n0 = INPUT_TOKENS[base] if base else 4096
        if base:
            counts = composite_counts(n0, layers, composite_target(base, sched))
        else:
            counts, n = [], n0
            for _ in layers:
                r = min(int(math.floor(n * SCHEDULES[sched][1])), (n - 1) // 2); counts.append(r); n -= r
    f = lambda n: 24 * n * D * D + 4 * n * n * D  # noqa: E731
    n, tot = n0, 0.0
    for b in range(L):
        tot += f(n)
        if b in layers:
            n -= counts[layers.index(b)]
    return tot / (L * f(4096)), (pool if pool else n)


def dv_seeds(cfg, sub, metric):
    out = {}
    for mf in (DV_ROOT / "off_cls" / cfg / sub).glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        r = np.load(mf.parent / "eval_rows.npy"); x = np.load(mf.parent / f"perclip_{metric}.npy")
        o = np.argsort(r); out[m["seed"]] = (r[o], x[o].astype(np.float64))
    return out


def dv_pair(a, b, sa="vis_official_cmp", sb=None):
    sb = sb or sa
    A, B = dv_seeds(a, sa, "acc"), dv_seeds(b, sb, "acc")
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    for k in s:
        assert np.array_equal(A[k][0], B[k][0])
    Dm = np.stack([(A[k][1] - B[k][1]) * 100 for k in s]); n = len(s)
    rng = np.random.default_rng(0)
    bo = np.array([Dm[rng.integers(0, n, n)][:, rng.integers(0, Dm.shape[1], Dm.shape[1])].mean() for _ in range(4000)])
    pc = lambda q: float(np.percentile(bo, q))  # noqa: E731
    return dict(n=n, d=float(Dm.mean()), clip_ci=[pc(2.5), pc(97.5)], ci90=[pc(5), pc(95)],
                A=float(np.mean([A[k][1].mean() for k in s])) * 100, B=float(np.mean([B[k][1].mean() for k in s])) * 100)


def dv_mean(cfg, sub="vis_official_cmp"):
    S = dv_seeds(cfg, sub, "acc")
    return float(np.mean([v[1].mean() for v in S.values()])) * 100 if len(S) == 4 else None


def frontier(pts, better):
    """pts: {arm: (cost tuple, score)}; an arm is dominated if another has every cost <= and a strictly better score, or
    every cost <= with one strictly lower and a score at least as good (point estimates)."""
    on = []
    for a, (c, s) in pts.items():
        dom = False
        for b, (c2, s2) in pts.items():
            if b == a or s2 is None or s is None:
                continue
            le = all(x2 <= x for x2, x in zip(c2, c))
            lt = any(x2 < x for x2, x in zip(c2, c))
            if le and (better(s2, s) or (lt and not better(s, s2))):
                dom = True; break
        if not dom and s is not None:
            on.append(a)
    return sorted(on, key=lambda a: pts[a][0])


def main():
    out = {"ego": {}, "dv": {}}
    E = out["ego"]
    E["arms"] = {}
    for c in ["dense"] + EGO_C:
        sub = "vispose_k1" if c == "dense" else "vispose_k1_cmp"
        fl, tk = flops_tokens(c)
        E["arms"][c] = dict(flops=fl, tokens=tk, ade=cvstats.mean(c, sub), ade_restore=cvstats.mean(c, "vispose_k1"))
    K = "vispose_k1_cmp"
    E["TOK_1_kbsm131_minus_4f_pool128"] = cvstats.pair("kbsmpa_L12", "in4f@p822", K)
    E["TOK_2_hand131_minus_4f_pool128"] = cvstats.pair("handcurpa_L12", "in4f@p822", K)
    E["TOK_3_hand131_minus_shuffle"] = cvstats.pair("handcurpa_L12", "handcurshufpa_L12", K)
    E["TOK_4_hand309_minus_4f"] = cvstats.pair("handcurpa_L9", "in4f", K)
    E["hand309_minus_shuffle"] = cvstats.pair("handcurpa_L9", "handcurshufpa_L9", K)
    E["kbsm131_minus_dense_pool128"] = cvstats.pair("kbsmpa_L12", "dense@p244", K)
    E["kbsm309_minus_4f"] = cvstats.pair("kbsmpa_L9", "in4f", K)
    E["hand131_minus_8f_pool256"] = cvstats.pair("handcurpa_L12", "in8f@p422", K)
    E["4f_pool128_minus_dense_pool128"] = cvstats.pair("in4f@p822", "dense@p244", K)
    E["hand131_minus_ref"] = cvstats.pair("handcurpa_L12", "kbsmpa_L12", K)
    E["hand309_minus_ref"] = cvstats.pair("handcurpa_L9", "kbsmpa_L9", K)
    E["compact_minus_restore"] = {c: cvstats.pair(c, c, K, "vispose_k1") for c in EGO_C}
    pts = {c: ((v["tokens"],), v["ade"]) for c, v in E["arms"].items()}
    E["frontier_tokens"] = frontier(pts, lambda x, y: x < y)
    pts2 = {c: ((v["flops"], v["tokens"]), v["ade"]) for c, v in E["arms"].items()}
    E["frontier_flops_tokens"] = frontier(pts2, lambda x, y: x < y)
    Dv = out["dv"]
    Dv["arms"] = {}
    for c in ["dense"] + DV_C:
        fl, tk = flops_tokens(c)
        Dv["arms"][c] = dict(flops=fl, tokens=tk, acc=dv_mean(c, "vis_official" if c == "dense" else "vis_official_cmp"),
                             acc_restore=dv_mean(c, "vis_official"))
    Dv["TOK_5_kbsm131_minus_4f_pool128"] = dv_pair("kbsmpa_L12", "in4f@p822")
    Dv["TOK_6_motion131_minus_4f_pool128"] = dv_pair("wampa_L12", "in4f@p822")
    Dv["TOK_7_motion131_minus_shuffle"] = dv_pair("wampa_L12", "wamshufpa_L12")
    Dv["TOK_8_motion309_minus_4f"] = dv_pair("wampa_L9", "in4f")
    Dv["motion309_minus_shuffle"] = dv_pair("wampa_L9", "wamshufpa_L9")
    Dv["motion972_minus_shuffle"] = dv_pair("wampa_s25", "wamshufpa_s25")
    Dv["in16f_motion309_minus_shuffle"] = dv_pair("in16f+wampa_L9", "in16f+wamshufpa_L9")
    Dv["kbsm131_minus_dense_pool128"] = dv_pair("kbsmpa_L12", "dense@p244")
    Dv["kbsm309_minus_4f"] = dv_pair("kbsmpa_L9", "in4f")
    Dv["motion131_minus_8f_pool256"] = dv_pair("wampa_L12", "in8f@p422")
    Dv["4f_pool128_minus_dense_pool128"] = dv_pair("in4f@p822", "dense@p244")
    Dv["motion131_minus_ref"] = dv_pair("wampa_L12", "kbsmpa_L12")
    Dv["motion309_minus_ref"] = dv_pair("wampa_L9", "kbsmpa_L9")
    Dv["motion972_minus_ref"] = dv_pair("wampa_s25", "kbsmpa_s25")
    Dv["motion972_minus_in24f"] = dv_pair("wampa_s25", "in24f")
    Dv["compact_minus_restore"] = {c: dv_pair(c, c, "vis_official_cmp", "vis_official") for c in DV_C}
    pts = {c: ((v["tokens"],), v["acc"]) for c, v in Dv["arms"].items()}
    Dv["frontier_tokens"] = frontier(pts, lambda x, y: x > y)
    pts2 = {c: ((v["flops"], v["tokens"]), v["acc"]) for c, v in Dv["arms"].items()}
    Dv["frontier_flops_tokens"] = frontier(pts2, lambda x, y: x > y)
    for c, v in E["arms"].items():
        print(f"EGO {c:20s} FLOPs {v['flops']:.2f} tokens {v['tokens']:5d} ADE {v['ade'] if v['ade'] is None else round(v['ade'], 2)}")
    for k, v in E.items():
        if k.startswith(("TOK", "hand", "kbsm", "4f")) and v and v.get("ade"):
            a = v["ade"]
            print(f"EGO {k:34s} {a['d']:+.2f} mm 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}]")
    print("EGO frontier tokens", E["frontier_tokens"]); print("EGO frontier FLOPs x tokens", E["frontier_flops_tokens"])
    for c, v in Dv["arms"].items():
        print(f"DV {c:20s} FLOPs {v['flops']:.2f} tokens {v['tokens']:5d} acc {v['acc'] if v['acc'] is None else round(v['acc'], 1)}")
    for k, v in Dv.items():
        if k.startswith(("TOK", "motion", "in16f", "kbsm", "4f")) and v:
            print(f"DV {k:34s} {v['d']:+.2f} pp 95[{v['clip_ci'][0]:+.2f},{v['clip_ci'][1]:+.2f}] (A {v['A']:.1f}, B {v['B']:.1f})")
    print("DV frontier tokens", Dv["frontier_tokens"]); print("DV frontier FLOPs x tokens", Dv["frontier_flops_tokens"])
    write_json(out, W / "runs" / "phaseA" / "roundTOK.json")


if __name__ == "__main__":
    main()
