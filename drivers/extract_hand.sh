#!/bin/bash
W=$WAM_ROOT
export WAM_FEAT_ROOT=$W/feats_pilot
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
CFGS=handmult_L9,handq10_L9,handq20_L9,handsrc_L9
# use exactly the clips of the first pilot extraction: same shard assignment, same files
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_hand_$s.log 2>&1 &
done
wait
for c in ${CFGS//,/ }; do mkdir -p $WAM_SHM/feats_pilot/$c && cp $WAM_FEAT_ROOT/$c/* $WAM_SHM/feats_pilot/$c/; done
echo "HAND EXTRACT DONE"
