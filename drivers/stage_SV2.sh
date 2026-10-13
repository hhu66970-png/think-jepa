#!/bin/bash
# Continuation of stage_SV.sh: wait for the round-S extraction, then run round-S CV on GPUs 4,5 and the
# round-V probes (dense banks need ~17 GB, one job per GPU) on GPUs 0,1.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
while [ "$(ps -u $(whoami) -o args= | grep -c '^$WAM_ROOT/conda/bin/python extract.py')" -gt 0 ]; do sleep 60; done
echo "S EXTRACT DONE"; grep -h "K=" $WAM_FEAT_ROOT/logs/extract_S_0.log
JV=$W/runs/viz/jobs_V.jsonl; mkdir -p $W/runs/viz; rm -f $JV
$PY make_jobs.py viz --cfgs dense --inputs vis --seeds 0-4 --extra="$HP --save_model" --out $JV
$PY make_jobs.py viz --cfgs dense,kbsmpa_L9,wampa_L9,handmultpa_L9 --inputs vispose --seeds 0-4 --tag _k1 --extra="$HP --pose_frames 1 --save_model" --out $JV
$PY gpu_pool.py $JV --gpus 0,1 --per_gpu 1 > $W/tmp/stage_V_pool.log 2>&1 &
J=$W/runs/cv/jobs_S.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs wamshufpa_L9,wamshuffpa_L9,waminvpa_L9,wamcomppa_L9,randrankpa_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
done
$PY gpu_pool.py $J --gpus 4,4,4,5,5,5 --per_gpu 1 > $W/tmp/stage_S_pool.log 2>&1 &
wait
echo "SV DONE"
