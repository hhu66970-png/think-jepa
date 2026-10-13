"""Tenth review, round G48: V-JEPA 2 ViT-g on the official Diving48 split (restore readout, seeds 0-3, 20 epochs)."""
import json

W = "$WAM_ROOT"
D = f"{W}/data/diving48/official"
jobs = []
for c in ["vitg_in12f", "vitg_kbsmpa_gL9", "vitg_wampa_gL9", "vitg_wamshufpa_gL9"]:
    for s in range(4):
        out = f"{W}/runs_DV2/off_cls/{c}/vis_official/s{s}"
        cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={D}/slim WAM_SPLITS={D}/splits WAM_FEAT_ROOT={W}/feats_DV2 "
               f"WAM_RUN_ROOT={W}/runs_DV2 {W}/conda/bin/python train_cls.py --cfg {c} --seed {s} --cv_fold -1 --epochs 20 "
               f"--rows_from dense --out {out}")
        jobs.append(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt"))
with open(f"{W}/runs_DV2/jobs_g48.jsonl", "w") as fh:
    for j in jobs:
        fh.write(json.dumps(j) + "\n")
print("jobs", len(jobs))
