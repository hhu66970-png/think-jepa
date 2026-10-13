#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
until grep -q "AFTER DONE" $W/tmp/stage_after.log; do sleep 60; done
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$S
# kinematic baselines on the test split
$PY - <<'P'
import numpy as np
P = np.load("$WAM_SHM/feats/poses.npz")
t = P["is_test"]; past, fut = P["past"][t], P["future"][t]
last = past[:, -1:]
copy = np.linalg.norm(fut - last, axis=-1) * 1000
vel = past[:, -1:] - past[:, -2:-1]
steps = np.arange(1, 33).reshape(1, 32, 1, 1)
cv = np.linalg.norm(fut - (last + vel * steps), axis=-1) * 1000
v4 = (past[:, -1:] - past[:, -5:-4]) / 4
cv4 = np.linalg.norm(fut - (last + v4 * steps), axis=-1) * 1000
for n, e in [("copy_last", copy), ("const_vel_1", cv), ("const_vel_4", cv4)]:
    print(f"BASELINE {n}: ADE {e.mean():.2f} FDE {e[:, -1].mean():.2f} wrist {e[:, :, [24, 50]].mean():.2f}")
P
J=$W/runs/test_main/jobs_g1full.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs dense --inputs zero --seeds 0-4 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs dense --inputs vispose --seeds 0-9 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
$PY aggregate.py $W/runs/test_main --input vispose --pairs handmult_L9:kbsm_L9,wam_L9:kbsm_L9,kbsm_L9:dense
$PY aggregate.py $W/runs/test_main --input zero
$PY aggregate.py $W/runs/test_main --input pose
# allocation metrics on the full train split
CUDA_VISIBLE_DEVICES=0 WAM_FEAT_ROOT=$W/feats $PY alloc_metrics.py --split train \
  --cfgs kbsm_L9,pitome_L9,wam_L9,randrank_L9,handmult_L9,handsrc_L9,handq10_L9,handq20_L9,wamq10_L9,wamsrc_L9,wamrank_L9,wamff_L9,wampmax_L9,wampmean_L9,wamcomp_L9,wamss_L9,wamadd_L9,kbsm_s25,wam_s25,handmult_s25,randrank_s25,kbsm_L12,wam_L12,handmult_L12,randrank_L12 \
  --out $W/feats/alloc_full.json 2>&1 | grep -v Warn | grep -v "tok = "
echo "EXTRA DONE"
