"""Round VM (PROTOCOL, ninth review): VideoMAE ViT-L on the official Diving48 split.  -> runs/phaseA/roundVM.json"""
from common import W, write_json
from offsplit import mean_acc, pair, show

ARMS = ["vmae_dense", "vmae_kbsmpa_s25", "vmae_wampa_s25", "vmae_wamshufpa_s25", "vmae_kbsmpa_L9", "vmae_wampa_L9",
        "vmae_wamshufpa_L9"]
out = {"accuracy": {a: mean_acc(a) for a in ARMS},
       "VM_1_motion_minus_shuffle_L9": pair("vmae_wampa_L9", "vmae_wamshufpa_L9"),
       "VM_2_motion_minus_shuffle_s25": pair("vmae_wampa_s25", "vmae_wamshufpa_s25"),
       "motion_minus_ref_L9": pair("vmae_wampa_L9", "vmae_kbsmpa_L9"), "motion_minus_ref_s25": pair("vmae_wampa_s25", "vmae_kbsmpa_s25"),
       "kbsmpa_s25_minus_dense": pair("vmae_kbsmpa_s25", "vmae_dense"), "kbsmpa_L9_minus_dense": pair("vmae_kbsmpa_L9", "vmae_dense")}
show("VM", out)
print("VM accuracy", {k: (None if v is None else round(v, 1)) for k, v in out["accuracy"].items()})
write_json(out, W / "runs" / "phaseA" / "roundVM.json")
