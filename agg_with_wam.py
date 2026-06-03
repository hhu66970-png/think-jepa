"""四方法下游最终聚合:dense / 方案A / K-BSM / PiToMe，plateau(末10epoch)ADE+FDE，
配对Δ vs dense，单边95%上界(TOST)。读远端 downstream_ext30_20260531/<cfg>__s<seed>/test_results.md。"""
import os, math, glob
ROOT = "outputs/downstream_ext30_20260531"   # run on remote, cwd=repo
SEEDS = [42, 43, 44, 45, 46, 47]
CFGS = {  # cfg_dir : (label, cos, speedup)
    "dense":       ("dense(基线)",   1.000, 1.00),
    "A_l8_r025":   ("方案A r0.25",   0.804, 1.25),
    "BSM_r008":    ("K-BSM r0.08",   0.914, 1.14),
    "BSM_r015":    ("K-BSM r0.15",   0.844, 1.26),
    "PITOME_r015": ("PiToMe r0.15",  0.826, 1.24),
    "WAM_r015":    ("WAM r0.15",     0.851, 1.25),
}
LASTK = 10
TCRIT = {2:2.920,3:2.353,4:2.132,5:2.015,6:1.943}

def col(path, name):
    rows=[l for l in open(path).read().splitlines() if l.strip().startswith("|")]
    if len(rows)<3: return None
    hdr=[c.strip() for c in rows[0].strip().strip("|").split("|")]
    if name not in hdr: return None
    i=hdr.index(name); out=[]
    for l in rows[2:]:
        c=[x.strip() for x in l.strip().strip("|").split("|")]
        if len(c)>i:
            try: out.append(float(c[i]))
            except: pass
    return out or None

def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float("nan")
def std(xs):
    xs=[x for x in xs if x is not None]
    if len(xs)<2: return float("nan")
    m=mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def plat(cfg, seed, metric):
    p=os.path.join(ROOT, f"{cfg}__s{seed}", "test_results.md")
    if not os.path.exists(p): return None
    v=col(p, metric)
    return mean(v[-LASTK:]) if v else None

for metric,name in [("val_avg_dist","ADE"),("val_final_dist","FDE")]:
    print(f"\n=== {name}  四方法 plateau(末{LASTK}) + 配对Δ vs dense + 95%上界 ===")
    print(f"{'方法':<14}{'cos':>6}{'spd':>6}{'mean':>9}{'n':>3}{'配对Δ':>10}{'sd':>9}{'t':>7}{'UP95':>10}{'UP%':>8}")
    dpl={s:plat("dense",s,metric) for s in SEEDS}
    dmean=mean([v for v in dpl.values() if v is not None])
    for cfg,(lab,cos,sp) in CFGS.items():
        vals=[plat(cfg,s,metric) for s in SEEDS]
        n=len([v for v in vals if v is not None]); m=mean(vals)
        if cfg=="dense":
            print(f"{lab:<14}{cos:>6.3f}{sp:>5.2f}x{m:>9.5f}{n:>3}{'--':>10}"); continue
        d=[plat(cfg,s,metric)-dpl[s] for s in SEEDS if plat(cfg,s,metric) is not None and dpl.get(s) is not None]
        dm,dsd=mean(d),std(d); se=dsd/math.sqrt(len(d)) if len(d)>1 else float("nan")
        t=dm/se if se and se>0 else float("nan"); up=dm+TCRIT.get(len(d)-1,2.0)*se
        print(f"{lab:<14}{cos:>6.3f}{sp:>5.2f}x{m:>9.5f}{n:>3}{dm:>+10.5f}{dsd:>9.5f}{t:>7.2f}{up:>+10.5f}{100*up/dmean:>+7.2f}%")
print(f"\ndense 种子噪声(ADE sd)={std([dpl_v for dpl_v in [plat('dense',s,'val_avg_dist') for s in SEEDS] if dpl_v is not None]):.5f}")
