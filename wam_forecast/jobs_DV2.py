"""Round DV2 jobs: 21 configs x seeds 0-3, official split, 20 epochs; the large banks first."""
import json
from pathlib import Path

W = "$WAM_ROOT"
D = f"{W}/data/diving48/official"
CFGS = ["dense", "in16f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25", "kbsmpa_L9", "wampa_L9", "wamshufpa_L9", "kbsmpa_L12",
        "wampa_L12", "wamshufpa_L12", "in16f+kbsmpa_L9", "in16f+wampa_L9", "in16f+wamshufpa_L9", "vidtldrpa_L9",
        "vidtldrshufpa_L9", "pospa_L9", "flowpa_L9", "flowshufpa_L9", "flowcpa_L9", "flowcshufpa_L9"]
Path(f"{W}/runs_DV2").mkdir(parents=True, exist_ok=True)
with open(f"{W}/runs_DV2/jobs_DV2.jsonl", "w") as fh:
    for c in CFGS:
        for s in range(4):
            out = f"{W}/runs_DV2/off_cls/{c}/vis_official/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={D}/slim WAM_SPLITS={D}/splits WAM_FEAT_ROOT={W}/feats_DV2 "
                   f"WAM_RUN_ROOT={W}/runs_DV2 {W}/conda/bin/python train_cls.py --cfg {c} --seed {s} --cv_fold -1 --epochs 20 --out {out}")
            fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
print("jobs", len(CFGS) * 4)
