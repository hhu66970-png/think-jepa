#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
while ! grep -q "A4 DONE\|A4 EXTRACT FAILED" $W/tmp/stage_A4.log; do sleep 60; done
export WAM_FEAT_ROOT=$WAM_SHM/feats
[ -d $WAM_SHM/feats/kbsm_L9 ] || echo "kbsm_L9 missing in shm"
J=$W/runs/test_main/jobs_Btest.jsonl; JD=$W/runs/test_main/jobs_Btest_dense.jsonl; rm -f $J $JD
X="$HP --pose_frames 1"
$PY make_jobs.py test_main --cfgs kbsmpa_L9,kbsmpa_L12,kbsmpa_s25,caps2pa_L9,caps2pa_L12,caps2pa_s25 --inputs vispose --seeds 0-19 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L9 --inputs pose,vispose --seeds 0-9 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_main --cfgs dense --inputs vispose --seeds 0-9 --tag _k1 --extra="$X" --out $JD
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY gpu_pool.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
for b in L9 L12 s25; do $PY pairs.py $W/runs/test_main caps2pa_$b/vispose_k1:kbsmpa_$b/vispose_k1 kbsmpa_$b/vispose_k1:dense/vispose_k1 caps2pa_$b/vispose_k1:dense/vispose_k1; done
$PY pairs.py $W/runs/test_main none/pose_k1:dense/vispose_k1 kbsmpa_L9/vispose_k1:kbsm_L9/vispose_k1
echo "B TEST DONE"
