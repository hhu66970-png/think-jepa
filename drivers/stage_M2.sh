#!/bin/bash
# Round M2 (eighth review): excl hand gate and its shuffle at 131 tokens, EgoDex current pose CV.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s+1)) $PY extract.py --configs handcurexclpa_L12,handcurexclshufpa_L12 --shard $s --nshards 3 --batch 4 > $W/feats/logs/extract_M2_$s.log 2>&1 &
done
wait
grep -il traceback $W/feats/logs/extract_M2_*.log && { echo "M2 EXTRACT FAILED"; exit 1; }
echo "M2 EXTRACT DONE"
J=$W/runs/cv/jobs_M2.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"
with open("$J", "w") as fh:
    for cfg in ("handcurexclpa_L12", "handcurexclshufpa_L12"):
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/vispose_k1_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py --cfg {cfg} "
                       f"--input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f} --pose_frames 1")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
$PY gpu_pool_claim.py $J --gpus 1,1,2,2,3,3 --per_gpu 1
echo "M2 CV DONE"
