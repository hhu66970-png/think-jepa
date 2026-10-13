"""Round DV3 (PROTOCOL, ninth review): frame rate frontier on the official Diving48 split.  -> runs/phaseA/roundDV3.json"""
from common import W, write_json
from offsplit import mean_acc, pair, show

ARMS = ["dense", "in24f", "in16f", "in12f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25", "kbsmpa_L9", "wampa_L9",
        "wamshufpa_L9", "in24f+kbsmpa_s25", "in24f+wampa_s25", "in24f+wamshufpa_s25", "in16f+kbsmpa_s25",
        "in16f+wampa_s25", "in16f+wamshufpa_s25", "in16f+kbsmpa_L9", "in16f+wampa_L9", "in16f+wamshufpa_L9"]
out = {"accuracy": {a: mean_acc(a) for a in ARMS},
       "DV3_1_motion972_minus_in24f": pair("wampa_s25", "in24f"),
       "DV3_2_in24f_motion_minus_shuffle_s25": pair("in24f+wampa_s25", "in24f+wamshufpa_s25"),
       "DV3_3_in16f_motion_minus_shuffle_s25": pair("in16f+wampa_s25", "in16f+wamshufpa_s25"),
       "in24f_minus_dense": pair("in24f", "dense"), "in12f_minus_dense": pair("in12f", "dense"),
       "in24f_minus_kbsmpa_s25": pair("in24f", "kbsmpa_s25"), "in12f_minus_kbsmpa_L9": pair("in12f", "kbsmpa_L9"),
       "in24f_motion_minus_ref_s25": pair("in24f+wampa_s25", "in24f+kbsmpa_s25"),
       "in16f_motion_minus_ref_s25": pair("in16f+wampa_s25", "in16f+kbsmpa_s25"),
       "in24f_motion_s25_minus_in16f": pair("in24f+wampa_s25", "in16f"),
       "in24f_motion_s25_minus_in24f": pair("in24f+wampa_s25", "in24f"),
       "in16f_motion_s25_minus_in16f": pair("in16f+wampa_s25", "in16f"),
       "motion972_minus_in16f": pair("wampa_s25", "in16f")}
show("DV3", out)
print("DV3 accuracy", {k: (None if v is None else round(v, 1)) for k, v in out["accuracy"].items()})
write_json(out, W / "runs" / "phaseA" / "roundDV3.json")
