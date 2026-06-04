#!/usr/bin/env python
"""E1 聚合:每压缩点 WAM vs K-BSM 配对-t(同seed)+ 各自 vs dense(复用 aligned dense)。
plateau 末10 val_avg_dist。决策门:WAM−K-BSM 配对Δ<0 且单边 t< −2.015 → WAM 干净显著赢。"""
import json, os, math
E1 = "outputs/frontier_retrain_E1"
DENSE = "outputs/downstream_ext30_aligned_20260602"   # 复用 dense 基线
SEEDS = [42, 43, 44, 45, 46, 47]; LASTK = 10
TCRIT = {5: 2.015, 4: 2.132, 3: 2.353, 2: 2.920}


def plat(base, cfg, s):
    p = f"{base}/{cfg}__s{s}/metrics.json"
    if not os.path.exists(p): return None
    ep = json.load(open(p)).get("epochs", [])
    v = [e.get("val_avg_dist") for e in ep[-LASTK:] if e.get("val_avg_dist") is not None]
    return sum(v)/len(v) if v else None


def stats(d):
    d = [x for x in d if x is not None]
    if len(d) < 2: return (float('nan'),)*4 + (len(d),)
    m = sum(d)/len(d); sd = math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1))
    se = sd/math.sqrt(len(d)); t = m/se if se > 0 else float('nan')
    return m, sd, se, t, len(d)


POINTS = [("L12_r25","~261"), ("L9_r25","~616"), ("r25","1944"), ("r20","2686")]
dp = {s: plat(DENSE, "dense", s) for s in SEEDS}
dmean = stats([v for v in dp.values()])[0]
print(f"dense plateau ADE mean={dmean:.5f} (n={sum(v is not None for v in dp.values())})\n")
print(f"{'point':9s}{'method':6s}{'mean':>9}{'Δvs_dense':>11}{'t':>7}{'  | WAM−KBSM 配对':>22}")
for pt, tok in POINTS:
    kb = {s: plat(E1, f"KBSM_{pt}", s) for s in SEEDS}
    wm = {s: plat(E1, f"WAM_{pt}", s) for s in SEEDS}
    nk = sum(v is not None for v in kb.values()); nw = sum(v is not None for v in wm.values())
    if nk == 0 and nw == 0:
        print(f"{pt:9s}(未跑)"); continue
    # vs dense
    for nm, dd in [("KBSM", kb), ("WAM", wm)]:
        diff = [dd[s]-dp[s] for s in SEEDS if dd.get(s) is not None and dp.get(s) is not None]
        m, sd, se, t, n = stats(dd.values()); _, _, _, td, _ = stats(diff)
        dm = (sum(diff)/len(diff)) if diff else float('nan')
        print(f"{pt:9s}{nm:6s}{m:>9.5f}{dm:>+11.5f}{td:>7.2f}", end="")
        if nm == "KBSM": print()
    # WAM vs KBSM paired
    wk = [wm[s]-kb[s] for s in SEEDS if wm.get(s) is not None and kb.get(s) is not None]
    if len(wk) >= 2:
        m, sd, se, t, n = stats(wk)
        tc = TCRIT.get(n-1, 2.0)
        sig = "  ✅WAM显著优于K-BSM" if (t == t and t < -tc) else ("  (趋势)" if (t==t and m<0) else "  (无优势)")
        up = m + tc*se
        print(f"   ⇒ WAM−KBSM Δ={m:+.5f} t={t:+.2f} (n={n},单边阈 {tc}) UP95={up:+.5f}{sig}")
    print()
print("决策门:某点 'WAM−KBSM Δ<0 且 t<−2.015' → WAM 在高压缩干净显著赢 K-BSM → 方法立住。")
