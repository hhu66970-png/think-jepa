#!/bin/bash
W=$WAM_ROOT
export WAM_FEAT_ROOT=$W/feats
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
mkdir -p $WAM_FEAT_ROOT/logs
$PY prep_poses.py
$PY hand_masks.py
M=""
for s in s15 s20 s25 L9 L12; do M="$M,kbsm_$s,wam_$s"; done
for s in s25 L9 L12; do M="$M,pitome_$s,handmult_$s,handq10_$s,handq20_$s,handsrc_$s,randmax_$s,randrank_$s,wamq10_$s,wamsrc_$s"; done
for v in wamrank wamff wampmax wampmean wamcomp wamss wamadd; do M="$M,${v}_L9"; done
CFGS="dense$M"
echo "configs: $CFGS"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_$s.log 2>&1 &
done
wait
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_0.log
echo "FULL EXTRACT DONE $(date -u +%FT%TZ)"
