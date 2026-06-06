import json, os, math
E1="outputs/frontier_retrain_E1"; AL="outputs/downstream_ext30_aligned_20260602"; LASTK=10
TCRIT={5:2.015,4:2.132,3:2.353,2:2.920}
def plat(base,cfg,s,key,temporal=False):
    p="%s/%s__s%d/metrics.json"%(base,cfg,s)
    if not os.path.exists(p): return None
    e=json.load(open(p)).get("epochs",[])
    vals=[]
    for x in e[-LASTK:]:
        src=x.get("temporal") if temporal else x
        if not src or src.get(key) is None: continue
        v=src[key]
        if isinstance(v,list): v=(sum(v)/len(v)) if v else None
        if v is not None: vals.append(v)
    return (sum(vals)/len(vals)) if vals else None
def st(d):
    d=[x for x in d if x is not None]
    if len(d)<2: return float("nan"),float("nan"),len(d)
    m=sum(d)/len(d); sd=math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)); se=sd/math.sqrt(len(d))
    return m,(m/se if se>0 else float("nan")),len(d)
SEEDS=list(range(42,48))
METRICS=[("val_avg_dist","ADE",False),("val_final_dist","FDE",False),("velocity_error","VEL",True),("accel_error","ACCEL",True)]
# point: (cfg_suffix, token, label).  L12=已有261参照; L14/L16=新极端点
POINTS=[("L12_r25","261(参照)"),("L14_r25","~146"),("L16_r25","~82")]
print("E1-EXTREME 聚合:WAM/KBSM vs dense(Δ<0=仍优于dense=未崩溃),+ WAM-KBSM 配对。seeds 42-47")
for key,nm,tp in METRICS:
    dn={s:plat(AL,"dense",s,key,tp) for s in SEEDS}; dm=st(list(dn.values()))[0]
    print("\n===== %s (dense=%.5f) ====="%(nm,dm))
    for pt,tok in POINTS:
        kb={s:plat(E1,"KBSM_"+pt,s,key,tp) for s in SEEDS}; wm={s:plat(E1,"WAM_"+pt,s,key,tp) for s in SEEDS}
        nk=sum(v is not None for v in kb.values()); nw=sum(v is not None for v in wm.values())
        if nk==0 and nw==0: print("  %-9s tok=%-9s (未跑)"%(pt,tok)); continue
        km=st(list(kb.values()))[0]; wmn=st(list(wm.values()))[0]
        wd=[wm[s]-dn[s] for s in SEEDS if wm.get(s) is not None and dn.get(s) is not None]; wdm,wdt,_=st(wd)
        kd=[kb[s]-dn[s] for s in SEEDS if kb.get(s) is not None and dn.get(s) is not None]; kdm,kdt,_=st(kd)
        wk=[wm[s]-kb[s] for s in SEEDS if wm.get(s) is not None and kb.get(s) is not None]; wkm,wkt,wkn=st(wk)
        wv="优于dense" if wdm<0 else "已崩溃(>dense)"
        print("  %-9s tok=%-9s KBSM=%.5f(vsD%+.5f,t%.2f) WAM=%.5f(vsD%+.5f,t%.2f %s) | W-K%+.5f t%.2f n%d"%(pt,tok,km,kdm,kdt,wmn,wdm,wdt,wv,wkm,wkt,wkn))
