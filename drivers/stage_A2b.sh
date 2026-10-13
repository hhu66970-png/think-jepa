#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
while ! grep -q "PHASE B VAL DONE" $W/tmp/stage_phaseA2.log; do sleep 60; done
export WAM_FEAT_ROOT=$WAM_SHM/feats
J=$W/runs/cv/jobs_A2b.jsonl; mkdir -p $W/runs/cv; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs caps2pa_L9,kbsmpa_L9,caps2pa_L12,kbsmpa_L12,caps2pa_s25,kbsmpa_s25 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
done
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY phaseA_analysis.py cv caps2pa kbsmpa
echo "A2b DONE"
