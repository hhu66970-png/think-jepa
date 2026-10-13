"""Ninth review jobs: round DV3 (8 configs), VM (7 VideoMAE configs) on the official Diving48 split, seeds 0-3, 20
epochs; round SEG (2 configs) on EgoDex CV, current pose, 5 folds x seeds 0-3. Large banks first."""
import json

W = "$WAM_ROOT"
D = f"{W}/data/diving48/official"
DV3 = ["in24f", "in12f", "in24f+kbsmpa_s25", "in24f+wampa_s25", "in24f+wamshufpa_s25", "in16f+kbsmpa_s25",
       "in16f+wampa_s25", "in16f+wamshufpa_s25"]
VM = ["vmae_dense", "vmae_kbsmpa_s25", "vmae_wampa_s25", "vmae_wamshufpa_s25", "vmae_kbsmpa_L9", "vmae_wampa_L9",
      "vmae_wamshufpa_L9"]
jobs = []
for c in DV3 + VM:
    rows_from = "vmae_dense" if c.startswith("vmae_") else "dense"
    for s in range(4):
        out = f"{W}/runs_DV2/off_cls/{c}/vis_official/s{s}"
        cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={D}/slim WAM_SPLITS={D}/splits WAM_FEAT_ROOT={W}/feats_DV2 "
               f"WAM_RUN_ROOT={W}/runs_DV2 {W}/conda/bin/python train_cls.py --cfg {c} --seed {s} --cv_fold -1 --epochs 20 "
               f"--rows_from {rows_from} --out {out}")
        jobs.append(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt"))
for c in ("handcursegpa_L12", "handcursegshufpa_L12"):
    for f in range(5):
        for s in range(4):
            out = f"{W}/runs/cv/{c}/vispose_k1_cv{f}/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py --cfg {c} "
                   f"--input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f} --pose_frames 1")
            jobs.append(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt"))
big = ("in24f", "in12f", "vmae_dense")
jobs.sort(key=lambda j: 0 if any(f"/{b}/" in j["done"] for b in big) else 1)
with open(f"{W}/runs_DV2/jobs_r9.jsonl", "w") as fh:
    for j in jobs:
        fh.write(json.dumps(j) + "\n")
print("jobs", len(jobs))
