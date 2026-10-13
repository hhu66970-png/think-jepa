#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
while ! grep -q "A2b DONE" $W/tmp/stage_A2b.log; do sleep 60; done
export WAM_FEAT_ROOT=$WAM_SHM/feats
J=$W/runs/select/jobs_P1.jsonl; JD=$W/runs/select/jobs_P1_dense.jsonl; rm -f $J $JD
rm -rf $W/runs/select/none/pose_k1_val $W/runs/select/*/vispose_k1_val
$PY make_jobs.py select --cfgs kbsmpa_L9,kbsmpa_L12,caps2pa_L9,caps2pa_L12 --inputs pose,vispose --seeds 0-4 --tag _k1_val --extra="$HP --val --pose_frames 1" --out $J
$PY make_jobs.py select --cfgs dense --inputs vispose --seeds 0-4 --tag _k1_val --extra="$HP --val --pose_frames 1" --out $JD
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY gpu_pool.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
$PY phaseB_analysis.py val caps2pa_L9 caps2pa_L12
echo "P1 DONE"
