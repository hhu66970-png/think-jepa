#!/bin/bash
W=$WAM_ROOT
export WAM_FEAT_ROOT=$W/feats_pilot
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
$PY hand_masks.py
CFGS=dense,kbsm_s25,kbsm_L9,kbsm_L12,pitome_L9,wam_s25,wam_L9,wam_L12,wamrank_L9,wamff_L9,wampmax_L9,wampmean_L9,wamcomp_L9,wamss_L9,wamsrc_L9,wamadd_L9,wamq10_L9,randmax_L9,randrank_L9
mkdir -p $WAM_FEAT_ROOT/logs
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s nohup $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_$s.log 2>&1 &
done
wait
echo "PILOT EXTRACT DONE $(date -u +%FT%TZ)"
