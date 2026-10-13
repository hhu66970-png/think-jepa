"""Round S part (b) (PROTOCOL, seventh review): fixed 30 fps versus uniform sampling of the same held out episodes.
Run once per sampling with its WAM_SLIM / WAM_SPLITS / WAM_FEAT_ROOT / WAM_RUN_ROOT:
  python roundS_analysis.py --tag fixed30  -> runs/phaseA/roundS_fixed30.json"""
import argparse
import json

import numpy as np

from cvstats import mean, pair  # this sampling's RUN_ROOT and poses
from common import FEAT_ROOT, W, write_json

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
K, V = "vispose_k1", "vis"
P = np.load(FEAT_ROOT / "poses.npz")
out = {"S1_motion_minus_shuffle_L9": pair("wampa_L9", "wamshufpa_L9", K), "S1_flow_minus_shuffle_L9": pair("flowpa_L9", "flowshufpa_L9", K),
       "S_motion_minus_shuffle_L12": pair("wampa_L12", "wamshufpa_L12", K), "S_motion_minus_shuffle_L9_vis": pair("wampa_L9", "wamshufpa_L9", V),
       "S_motion_minus_ref_L9": pair("wampa_L9", "kbsmpa_L9", K), "S_flow_minus_ref_L9": pair("flowpa_L9", "kbsmpa_L9", K),
       "S_pospa_minus_ref_L9": pair("pospa_L9", "kbsmpa_L9", K), "S_motion_minus_ref_L12": pair("wampa_L12", "kbsmpa_L12", K)}
past, fut = P["past"], P["future"]
copy = np.linalg.norm(fut - past[:, -1:], axis=-1).mean(axis=(1, 2)) * 1000
out["means"] = dict(kbsmpa_L9=mean("kbsmpa_L9", K), kbsmpa_L12=mean("kbsmpa_L12", K), pose_only=mean("none", "pose_k1"),
                    kbsmpa_L9_vis=mean("kbsmpa_L9", V), copy_last_all=float(copy[~P["is_test"]].mean()))
for k, v in out.items():
    if isinstance(v, dict) and v.get("ade"):
        a_ = v["ade"]
        print(f"S[{a.tag}] {k:34s} ADE {a_['d']:+.2f} 95[{a_['clip_ci'][0]:+.2f},{a_['clip_ci'][1]:+.2f}] task[{a_['task_ci'][0]:+.2f},{a_['task_ci'][1]:+.2f}] seeds<0 {a_['wins']}/{a_['n']}")
print(f"S[{a.tag}] means", {k: (None if v is None else round(v, 2)) for k, v in out["means"].items()})
write_json(out, W / "runs" / "phaseA" / f"roundS_{a.tag}.json")
