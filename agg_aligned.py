import json,os,math,statistics as st
ROOT="outputs/downstream_ext30_aligned_20260602"
CFGS=[("dense","dense"),("A_l8_r025","方案A r0.25"),("BSM_r015","K-BSM r0.15"),
      ("PITOME_r015","PiToMe r0.15"),("WAM_r015","WAM r0.15")]
SEEDS=[42,43,44,45,46,47]; LASTK=10; TCRIT={5:2.015,4:2.132,3:2.353}
def eps(c,s):
    p=f"{ROOT}/{c}__s{s}/metrics.json"; return json.load(open(p)).get("epochs",[]) if os.path.exists(p) else []
def plat(c,s,k):
    e=eps(c,s); v=[x.get(k) for x in e[-LASTK:] if x.get(k) is not None]; return sum(v)/len(v) if v else None
def platT(c,s,k):
    e=eps(c,s); v=[x["temporal"][k] for x in e[-LASTK:] if x.get("temporal") and x["temporal"].get(k) is not None]; return sum(v)/len(v) if v else None
def mn(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float("nan")
def sd(xs): xs=[x for x in xs if x is not None]; m=mn(xs); return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else float("nan")
print(f"ROOT={ROOT}")
for metric,name in [("val_avg_dist","ADE"),("val_final_dist","FDE")]:
    print(f"\n=== {name} aligned plateau(last{LASTK}) + pairedΔ vs dense + TOST UP95, n<=6 ===")
    dp={s:plat("dense",s,metric) for s in SEEDS}; dm=mn(list(dp.values()))
    for c,lab in CFGS:
        vs=[plat(c,s,metric) for s in SEEDS]; m=mn(vs); n=len([v for v in vs if v is not None])
        if c=="dense": print(f"{lab:14s} n={n} mean={m:.5f}  --"); continue
        d=[plat(c,s,metric)-dp[s] for s in SEEDS if plat(c,s,metric) is not None and dp.get(s) is not None]
        dmd,dsd=mn(d),sd(d); se=dsd/math.sqrt(len(d)) if len(d)>1 else float("nan")
        t=dmd/se if se and se>0 else float("nan"); up=dmd+TCRIT.get(len(d)-1,2.0)*se
        print(f"{lab:14s} n={n} mean={m:.5f} Δ={dmd:+.5f} t={t:+.2f} UP95={up:+.5f}({100*up/dm:+.2f}%)")
print(f"\n=== D temporal aligned (plateau vel/accel) ===")
for c,lab in CFGS:
    print(f"{lab:14s} vel={mn([platT(c,s,'velocity_error') for s in SEEDS]):.4f} acc={mn([platT(c,s,'accel_error') for s in SEEDS]):.4f}")
