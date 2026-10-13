#!/bin/bash
# Round CP replication on the part 2 test split (run only after the CV classification, as the protocol fixes):
# the same 4 arms, vispose with pose_frames 1, seeds 0-19 as the existing kbsmpa_L9 vispose_k1 test runs, no tuning.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
J=$W/runs/test_main/jobs_CPtest.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for cfg in ("wampa_L9", "handmultpa_L9", "wamshufpa_L9", "wamshuffpa_L9"):
        for s in range(20):
            out = f"{W}/runs/test_main/{cfg}/vispose_k1/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                   f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1")
            fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${CP_GPUS:-1,1,2,2,3,3,5,5} --per_gpu 1
echo "CPTEST DONE"
