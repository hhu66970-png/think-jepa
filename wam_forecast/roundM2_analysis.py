"""Round M2 (PROTOCOL, eighth review): the excl hand gate (top 20% hand tokens leave both sides, the rest re-matched)
against its shuffle and against K-BSM+PA, EgoDex current pose CV.  python roundM2_analysis.py -> runs/phaseA/roundM2.json"""
from cvstats import pair
from common import W, write_json

K = "vispose_k1"
out = {"M2_excl_minus_shuffle": pair("handcurexclpa_L12", "handcurexclshufpa_L12", K),
       "M2_excl_minus_ref": pair("handcurexclpa_L12", "kbsmpa_L12", K),
       "M2_excl_shuffle_minus_ref": pair("handcurexclshufpa_L12", "kbsmpa_L12", K)}
for k, v in out.items():
    a = v["ade"]
    print(f"M2 {k:28s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] seeds<0 {a['wins']}/{a['n']}")
write_json(out, W / "runs" / "phaseA" / "roundM2.json")
