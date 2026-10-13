#!/bin/bash
# Tenth review, round TOK runs (restarted with *_cmp output names), then analysis.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
$PY gpu_pool_claim.py $W/runs_DV2/jobs_r10.jsonl --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1 > $W/tmp/pool_r10_A.log 2>&1 &
until [ $($PY -c "import json,os;J=[json.loads(l) for l in open('$W/runs_DV2/jobs_r10.jsonl')];print(sum(1 for j in J if j['big'] and not (os.path.exists(j['done']+'.claim') or os.path.exists(j['done']))))") -eq 0 ]; do sleep 60; done
$PY gpu_pool_claim.py $W/runs_DV2/jobs_r10.jsonl --gpus 0,1,2,3,4,5 --per_gpu 1 > $W/tmp/pool_r10_B.log 2>&1 &
wait
echo "R10 RUNS DONE"
D=$W/data/diving48/official
WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats $PY roundTOK_analysis.py > $W/tmp/roundTOK.txt 2>&1
echo "TOK ANALYSIS DONE"
