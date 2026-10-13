#!/bin/bash
# Phase A2: one-time test of c* = caps2pa (+ other caps for the correlation analysis), then Phase B.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
while ! grep -q "PHASE A1 DONE" $W/tmp/stage_phaseA.log; do sleep 30; done
export WAM_FEAT_ROOT=$WAM_SHM/feats
J=$W/runs/test_main/jobs_phaseA2.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs caps2pa_L9,caps2pa_L12,caps2pa_s25 --inputs vis --seeds 0-19 --extra="$HP" --out $J
O=""; for m in capc1pa capc2pa capc4pa caps4pa caps8pa; do O="$O,${m}_L9,${m}_L12,${m}_s25"; done
$PY make_jobs.py test_main --cfgs ${O#,} --inputs vis --seeds 0-9 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
for b in L9 L12 s25; do $PY pairs.py $W/runs/test_main caps2pa_$b/vis:kbsmpa_$b/vis; done
echo "PHASE A2 DONE"
bash $W/tmp/stage_phaseB.sh
