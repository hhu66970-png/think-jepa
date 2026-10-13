#!/bin/bash
# Rounds LAT2 and PROF (ninth review), on an otherwise idle GPU 0.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
CUDA_VISIBLE_DEVICES=0 $PY bench.py --configs dense,kbsmpa_s25,wampa_s25,in24f,in12f,in24f+kbsmpa_s25,in24f+wampa_s25,in16f+kbsmpa_s25,in16f+wampa_s25 --batch 1,8 --out $W/runs/phaseA/bench_lat2.json
CUDA_VISIBLE_DEVICES=0 $PY profile_merge.py
echo "LAT2 PROF DONE"
