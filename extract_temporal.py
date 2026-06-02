import json, os, statistics as st
ROOT="outputs/downstream_ext30_20260531"
CFGS=[("dense","dense"),("A_l8_r025","方案A r0.25"),("BSM_r008","K-BSM r0.08"),
      ("BSM_r015","K-BSM r0.15"),("PITOME_r015","PiToMe r0.15"),("WAM_r015","WAM r0.15")]
SEEDS=[42,43,44,45,46,47]; LASTK=10
def eps(cfg,s):
    p=f"{ROOT}/{cfg}__s{s}/metrics.json"
    return json.load(open(p)).get("epochs",[]) if os.path.exists(p) else []
res={}
print(f"{'method':14s} {'n':>2} {'vel_err(plateau)':>16} {'acc_err(plateau)':>16}")
for cfg,lab in CFGS:
    vel,acc,psd42=[],[],None
    for s in SEEDS:
        ep=eps(cfg,s); temps=[e.get("temporal") for e in ep if e.get("temporal")]
        if not temps: continue
        v=[t["velocity_error"] for t in temps[-LASTK:] if t.get("velocity_error") is not None]
        a=[t["accel_error"] for t in temps[-LASTK:] if t.get("accel_error") is not None]
        if v: vel.append(sum(v)/len(v))
        if a: acc.append(sum(a)/len(a))
        if s==42 and temps and temps[-1].get("per_step_displacement"): psd42=temps[-1]["per_step_displacement"]
    res[cfg]=dict(vel=vel,acc=acc,psd=psd42)
    if vel: print(f"{lab:14s} {len(vel):>2} {st.mean(vel):>16.4f} {st.mean(acc):>16.4f}")
    else:   print(f"{lab:14s}  0   *** NO temporal block in metrics.json ***")
dpsd=res.get("dense",{}).get("psd")
if dpsd:
    print("\nper-step displacement max|Δ vs dense| (seed42, last epoch):")
    for cfg,lab in CFGS:
        psd=res[cfg]["psd"]
        if psd and cfg!="dense" and len(psd)==len(dpsd):
            print(f"  {lab:14s} max|Δ|={max(abs(a-b) for a,b in zip(psd,dpsd)):.4f}")
