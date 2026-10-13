#!/bin/bash
# Rerun the round 4 jobs that failed with CUDA out of memory: dense banks one per GPU (0-2), others two per GPU (3-5).
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
$PY $W/tmp/fix_R4.py
$PY gpu_pool_claim.py $W/runs/cv/jobs_R4_fix_dense.jsonl --gpus 0,1,2 --per_gpu 1 &
$PY gpu_pool_claim.py $W/runs/cv/jobs_R4_fix_other.jsonl --gpus 3,3,4,4,5,5 --per_gpu 1 &
wait
echo "R4 FIX DONE"
WAM_FEAT_ROOT=$W/feats $PY round4_analysis.py
echo "R4 FIX ANALYSIS DONE"
