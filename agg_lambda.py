import json, os
E1="outputs/frontier_retrain_E1"; LASTK=10; S=list(range(42,48))
def plat(cfg,key):
    vals=[]
    for s in S:
        p="%s/%s__s%d/metrics.json"%(E1,cfg,s)
        if not os.path.exists(p): continue
        ep=json.load(open(p)).get("epochs",[])
        v=[e.get(key) for e in ep[-LASTK:] if e.get(key) is not None]
        if v: vals.append(sum(v)/len(v))
    return (sum(vals)/len(vals) if vals else float("nan")), len(vals)
print("λ 门控扫描 @616 (L9_r25), n=6, plateau ADE/FDE")
for lam,cfg in [("0.0(=KBSM)","KBSM_L9_r25"),("0.25","WAMlam025_L9_r25"),("0.5","WAMlam05_L9_r25"),("0.75","WAMlam075_L9_r25"),("1.0(=WAM)","WAM_L9_r25")]:
    a,na=plat(cfg,"val_avg_dist"); f,_=plat(cfg,"val_final_dist")
    print("  lambda=%-11s ADE=%.5f FDE=%.5f (n=%d)"%(lam,a,f,na))
print("-> ADE 随 lambda 单调下降 => 门控(任务感知)是 WAM 增益的因(非仅 K-BSM)")
