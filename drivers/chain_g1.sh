#!/bin/bash
# 2026-10-05: a no-op skip line rewritten as a comment so that the script parses; behaviour unchanged.
W=$WAM_ROOT
PY=$W/conda/bin/python
export WAM_FEAT_ROOT=$W/feats_pilot

cd $W/repo/wam_forecast
# (skipped at the time: alloc_metrics was still running)
# : skip alloc (still running) \
#   --cfgs kbsm_L9,pitome_L9,wam_L9,wamrank_L9,wamff_L9,wampmax_L9,wampmean_L9,wamcomp_L9,wamss_L9,wamsrc_L9,wamadd_L9,wamq10_L9,randmax_L9,randrank_L9,kbsm_s25,wam_s25,kbsm_L12,wam_L12 \
#   > $W/feats_pilot/alloc_metrics.txt 2>&1 &
$PY gpu_pool.py $W/runs/pilot_g1/jobs.jsonl --gpus 0,1,2,3,4,5 --per_gpu 2
wait
echo "CHAIN G1 DONE $(date -u +%FT%TZ)"
