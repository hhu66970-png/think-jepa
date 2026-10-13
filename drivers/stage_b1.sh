#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
until grep -q "PA DONE" $W/tmp/stage_pa.log; do sleep 60; done
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs kbsmlm_s25,kbsmlm_L9,kbsmlm_L12 --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_b1_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_b1_0.log
grep -il traceback $W/feats/logs/extract_b1_*.log
for c in kbsmlm_s25 kbsmlm_L9 kbsmlm_L12; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
CUDA_VISIBLE_DEVICES=0 $PY alloc_metrics.py --split train --cfgs kbsm_L9,kbsmpa_L9,kbsmlm_L9,kbsm_s25,kbsmlm_s25,kbsm_L12,kbsmlm_L12 \
  --out $W/feats/alloc_b1.json 2>&1 | grep -v Warn | grep -v "tok = "
until grep -q "PA DONE" $W/tmp/stage_pa.log; do sleep 60; done
export WAM_FEAT_ROOT=$S
J=$W/runs/test_main/jobs_b1.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs kbsmlm_s25,kbsmlm_L9,kbsmlm_L12 --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY pairs.py $W/runs/test_main kbsmlm_s25/vis:kbsm_s25/vis kbsmlm_L9/vis:kbsm_L9/vis kbsmlm_L12/vis:kbsm_L12/vis \
  kbsmlm_L9/vis:kbsmpa_L9/vis kbsmlm_L9/vis:dense/vis
echo "B1 DONE"
