"""Tenth review jobs: round TOK (compact readout; EgoDex current pose CV 5 folds x seeds 0-3, 80 epochs; Diving48 official
split seeds 0-3, 20 epochs) and the restore runs of the new frame counts. Large banks first."""
import json

W = "$WAM_ROOT"
D = f"{W}/data/diving48/official"
EGO_C = ["in16f", "in12f", "in8f", "in4f", "kbsmpa_L9", "kbsmpa_L12", "handcurpa_L9", "handcurshufpa_L9", "handcurpa_L12",
         "handcurshufpa_L12", "in16f+kbsmpa_L9", "in16f+kbsmpa_L12", "in12f+kbsmpa_L9", "in12f+kbsmpa_L12",
         "in8f+kbsmpa_L9", "in8f+kbsmpa_L12", "dense@p222", "dense@p244", "in4f@p822", "in8f@p422"]
EGO_R = ["in4f"]
DV_C = ["in24f", "in16f", "in12f", "in8f", "in4f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25", "kbsmpa_L9", "wampa_L9",
        "wamshufpa_L9", "kbsmpa_L12", "wampa_L12", "wamshufpa_L12", "in16f+kbsmpa_L9", "in16f+wampa_L9",
        "in16f+wamshufpa_L9", "dense@p222", "dense@p244", "in4f@p822", "in8f@p422"]
DV_R = ["in8f", "in4f"]
BIG = {"in24f", "in16f", "in12f", "in8f", "dense@p222"}
jobs = []
for c, ro in [(c, "compact") for c in DV_C] + [(c, "restore") for c in DV_R]:
    for s in range(4):
        out = f"{W}/runs_DV2/off_cls/{c}/vis_official{'_cmp' if ro == 'compact' else ''}/s{s}"
        cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={D}/slim WAM_SPLITS={D}/splits WAM_FEAT_ROOT={W}/feats_DV2 "
               f"WAM_RUN_ROOT={W}/runs_DV2 {W}/conda/bin/python train_cls.py --cfg '{c}' --seed {s} --cv_fold -1 --epochs 20 "
               f"--rows_from dense --readout {ro} --out '{out}'")
        jobs.append(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt", big=c in BIG))
for c, ro in [(c, "compact") for c in EGO_C] + [(c, "restore") for c in EGO_R]:
    for f in range(5):
        for s in range(4):
            sub = f"vispose_k1{'_cmp' if ro == 'compact' else ''}_cv{f}"
            out = f"{W}/runs/cv/{c}/{sub}/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py --cfg '{c}' "
                   f"--input vispose --seed {s} --out '{out}' --lr 2e-4 --epochs 80 --cv_fold {f} --pose_frames 1 --readout {ro}")
            jobs.append(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt", big=c in BIG))
jobs.sort(key=lambda j: 0 if j["big"] else 1)
with open(f"{W}/runs_DV2/jobs_r10.jsonl", "w") as fh:
    for j in jobs:
        fh.write(json.dumps(j) + "\n")
print("jobs", len(jobs), "big", sum(j["big"] for j in jobs))
