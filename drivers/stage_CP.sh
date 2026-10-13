#!/bin/bash
# Round CP (current pose gates): 4 arms x 5 folds x seeds 0-3 = 80 probe runs, cached L9 features, vispose with
# pose_frames 1, restore readout; arguments identical to the existing kbsmpa_L9 vispose_k1 CV runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
J=$W/runs/cv/jobs_CP.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for cfg in ("wampa_L9", "handmultpa_L9", "wamshufpa_L9", "wamshuffpa_L9"):
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/vispose_k1_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${CP_GPUS:-1,1,2,2,3,3,5,5} --per_gpu 1
echo "CP DONE"
