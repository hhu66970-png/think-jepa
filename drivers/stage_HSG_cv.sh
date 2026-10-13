#!/bin/bash
# Rounds HS and G: CV probe runs with the current pose (vispose, pose_frames 1), seeds 0-3, after the extraction.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "G EXTRACT DONE" $W/tmp/stage_HSG_extract.log; do sleep 60; done
J=$W/runs/cv/jobs_HSG.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for cfg in ("handshuffpa_L12", "handshufpa_L12", "vitg_wampa_gL9", "vitg_handmultpa_gL9", "vitg_kbsmpa_gL9"):
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/vispose_k1_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${HSG_GPUS:-1,1,2,2,4,4,5,5} --per_gpu 1
echo "HSG CV DONE"
