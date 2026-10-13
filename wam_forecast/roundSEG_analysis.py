"""Round SEG (PROTOCOL, ninth review): the hand mask only forbids hand and background merges (131 tokens, EgoDex
current pose CV).  python roundSEG_analysis.py -> runs/phaseA/roundSEG.json"""
from cvstats import pair
from common import W, write_json

K = "vispose_k1"
out = {"SEG_1_seg_minus_shuffle": pair("handcursegpa_L12", "handcursegshufpa_L12", K),
       "seg_minus_ref": pair("handcursegpa_L12", "kbsmpa_L12", K),
       "seg_minus_reorder": pair("handcursegpa_L12", "handcurpa_L12", K),
       "seg_shuffle_minus_ref": pair("handcursegshufpa_L12", "kbsmpa_L12", K)}
for k, v in out.items():
    a = v["ade"]
    print(f"SEG {k:28s} ADE {a['d']:+.2f} 95[{a['clip_ci'][0]:+.2f},{a['clip_ci'][1]:+.2f}] task[{a['task_ci'][0]:+.2f},{a['task_ci'][1]:+.2f}] seeds<0 {a['wins']}/{a['n']}")
write_json(out, W / "runs" / "phaseA" / "roundSEG.json")
