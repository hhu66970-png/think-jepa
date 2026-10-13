"""Round G48 (PROTOCOL, tenth review): V-JEPA 2 ViT-g on the official Diving48 split.  -> runs/phaseA/roundG48.json"""
from common import W, write_json
from offsplit import mean_acc, pair, show

ARMS = ["vitg_in12f", "vitg_kbsmpa_gL9", "vitg_wampa_gL9", "vitg_wamshufpa_gL9"]
out = {"accuracy": {a: mean_acc(a) for a in ARMS},
       "G_1_motion_minus_shuffle": pair("vitg_wampa_gL9", "vitg_wamshufpa_gL9"),
       "G_2_motion_minus_in12f": pair("vitg_wampa_gL9", "vitg_in12f"),
       "kbsm_minus_in12f": pair("vitg_kbsmpa_gL9", "vitg_in12f"),
       "motion_minus_kbsm": pair("vitg_wampa_gL9", "vitg_kbsmpa_gL9"),
       "shuffle_minus_kbsm": pair("vitg_wamshufpa_gL9", "vitg_kbsmpa_gL9")}
show("G48", out)
print("G48 accuracy", {k: (None if v is None else round(v, 1)) for k, v in out["accuracy"].items()})
write_json(out, W / "runs" / "phaseA" / "roundG48.json")
