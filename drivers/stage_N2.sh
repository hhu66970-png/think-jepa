#!/bin/bash
# Round N2: fmnat_L9 features (6 shards) and 20 current pose CV runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs fmnat_L9 --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_N2_$s.log 2>&1 &
done
wait
echo "N2 EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_N2_0.log
J=$W/runs/cv/jobs_N2.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for f in range(5):
        for s in range(4):
            out = f"{W}/runs/cv/fmnat_L9/vispose_k1_cv{f}/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                   f"--cfg fmnat_L9 --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}")
            fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
$PY gpu_pool_claim.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
echo "N2 CV DONE"
