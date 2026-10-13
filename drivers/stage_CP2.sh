#!/bin/bash
# Round CP2: replications of round CP with the current pose (vispose, pose_frames 1), CV, seeds 0-3, cached features.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
J=$W/runs/cv/jobs_CP2.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
arms = [(c, "vispose_k1", "") for c in ("wampa_L12", "handmultpa_L12", "vmae_kbsmpa_L9", "vmae_wampa_L9", "vmae_handmultpa_L9", "vmae_dense")]
arms += [(c, "vispose_k1_compact", " --readout compact_logsize") for c in ("kbsmpa_L9", "wampa_L9", "handmultpa_L9")]
with open(J, "w") as fh:
    for cfg, sub, extra in arms:
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/{sub}_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}{extra}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${CP2_GPUS:-0,0,1,1,2,3,4,5} --per_gpu 1
echo "CP2 DONE"
