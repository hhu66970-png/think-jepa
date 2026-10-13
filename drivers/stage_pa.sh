#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs kbsmpa_s25,kbsmpa_L9,kbsmpa_L12 --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_pa_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_pa_0.log
for c in kbsmpa_s25 kbsmpa_L9 kbsmpa_L12; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/test_main/jobs_pa.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs kbsmpa_s25,kbsmpa_L9,kbsmpa_L12 --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY pairs.py $W/runs/test_main kbsmpa_s25/vis:kbsm_s25/vis kbsmpa_L9/vis:kbsm_L9/vis kbsmpa_L12/vis:kbsm_L12/vis kbsmpa_L9/vis:dense/vis
echo "PA DONE"
