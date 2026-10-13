#!/bin/bash
# Rounds HS and G: feature extraction on GPUs 1,2,4,5 (0 and 3 are used by other users).
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
G=(1 2 4 5)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY extract.py --configs handshufpa_L12,handshuffpa_L12 --shard $i --nshards 4 --batch 8 > $W/feats/logs/extract_HS_$i.log 2>&1 &
done
wait
echo "HS EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_HS_*.log | head -4
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY extract.py --arch vitg --configs wampa_gL9,handmultpa_gL9 --shard $i --nshards 4 --batch 4 > $W/feats/logs/extract_G_$i.log 2>&1 &
done
wait
echo "G EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_G_*.log | head -4
