import json, os, statistics as st
ROOT = "outputs/downstream_c6_heldout"
CFGS = [("dense", "dense"), ("A_l8_r025", "方案A r0.25"), ("BSM_r015", "K-BSM r0.15"),
        ("PITOME_r015", "PiToMe r0.15"),
        ("WAM_r015", "WAM(feature旧)"), ("WAM_r015_key", "WAM(Key新)")]
SEEDS = [42, 43, 44]; LASTK = 10


def plat(cfg, s):
    p = f"{ROOT}/{cfg}__s{s}/metrics.json"
    if not os.path.exists(p):
        return None
    ep = json.load(open(p)).get("epochs", [])
    v = [e.get("val_avg_dist") for e in ep[-LASTK:] if e.get("val_avg_dist") is not None]
    return sum(v) / len(v) if v else None


dp = {s: plat("dense", s) for s in SEEDS}
print(f"\n=== C6 held-out drawer: plateau(last{LASTK}) ADE + pairedΔ vs dense (n=3) ===")
print(f"{'method':16s}" + "".join(f"{('s'+str(s)):>9}" for s in SEEDS) + f"{'mean':>9}{'pairedΔ':>10}")
for cfg, lab in CFGS:
    vs = [plat(cfg, s) for s in SEEDS]; ok = [v for v in vs if v is not None]
    m = st.mean(ok) if ok else float('nan')
    cells = "".join((f"{v:>9.4f}" if v is not None else f"{'--':>9}") for v in vs)
    if cfg == "dense":
        print(f"{lab:16s}{cells}{m:>9.4f}{'--':>10}"); continue
    d = [vs[i] - dp[SEEDS[i]] for i in range(len(SEEDS)) if vs[i] is not None and dp.get(SEEDS[i]) is not None]
    dm = st.mean(d) if d else float('nan')
    print(f"{lab:16s}{cells}{m:>9.4f}{dm:>+10.5f}")
