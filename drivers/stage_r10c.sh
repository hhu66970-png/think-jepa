#!/bin/bash
# Tenth review: rerun the round TOK jobs that failed (a code slip that cut FeatureBank.batch, fixed at 05:35 UTC, and
# out of memory when two large compact banks shared a GPU), one job per GPU, then the analysis.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "R10 RUNS DONE" $W/tmp/stage_r10b.log 2>/dev/null; do sleep 60; done
for i in 1 2; do
  $PY $W/tmp/retry_claims.py $W/runs_DV2/jobs_r10.jsonl
  $PY gpu_pool_claim.py $W/runs_DV2/jobs_r10.jsonl --gpus 0,1,2,3,4,5 --per_gpu 1 > $W/tmp/pool_r10_retry$i.log 2>&1 && break
done
echo "R10 RETRY DONE"
D=$W/data/diving48/official
WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats $PY roundTOK_analysis.py > $W/tmp/roundTOK.txt 2>&1
echo "TOK ANALYSIS DONE"
