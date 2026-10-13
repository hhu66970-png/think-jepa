#!/bin/bash
# Round P: the published PiToMe algorithm (official code) with PA, fresh 5-fold CV at 309 tokens, both read-outs.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
grep -q "ROUND P SPECIFICATION" PROTOCOL.md || echo "missing spec"
G=(0 1 4 5 0 1)   # GPUs 2 and 3 host another user's services
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${G[$s]} $PY extract.py --configs pitomeopa_L9 --shard $s --nshards 6 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_P_$s.log 2>&1 &
done
wait
ls $WAM_FEAT_ROOT/pitomeopa_L9 | wc -l
echo "P EXTRACT DONE"
J=$W/runs/cv/jobs_P.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs pitomeopa_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
  $PY make_jobs.py cv --cfgs pitomeopa_L9 --inputs vis --seeds 0-3 --tag _compact_cv$f --extra="$HP --cv_fold $f --readout compact_logsize" --out $J
done
$PY gpu_pool.py $J --gpus 0,0,0,1,1,1,4,4,4,5,5,5 --per_gpu 1
echo "P DONE"
