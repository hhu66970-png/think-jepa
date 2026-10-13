#!/bin/bash
# Phase B (validation): vision gain and compression gap under weakened pose input.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$WAM_SHM/feats
J=$W/runs/select/jobs_phaseB.jsonl; JD=$W/runs/select/jobs_phaseB_dense.jsonl; rm -f $J $JD
for k in 32 4 1; do
  if [ $k = 32 ]; then T=_val; else T=_k${k}_val; fi
  $PY make_jobs.py select --cfgs kbsmpa_L9,kbsmpa_L12,caps2pa_L9,caps2pa_L12 --inputs pose,vispose --seeds 0-4 --tag $T --extra="$HP --val --pose_frames $k" --out $J
  $PY make_jobs.py select --cfgs dense --inputs vispose --seeds 0-4 --tag $T --extra="$HP --val --pose_frames $k" --out $JD
done
# vis-only dense on val with the same seeds (P0)
$PY make_jobs.py select --cfgs dense --inputs vis --seeds 0-4 --tag _val --extra="$HP --val" --out $JD
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY gpu_pool.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
echo "PHASE B VAL DONE"
