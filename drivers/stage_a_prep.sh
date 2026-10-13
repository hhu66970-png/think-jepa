#!/bin/bash
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
until grep -q "FIXCD DONE" $W/tmp/fix_cd.log; do sleep 60; done
export WAM_FEAT_ROOT=$WAM_SHM/feats
# 1) restore path must still reproduce the published run bit for bit
CUDA_VISIBLE_DEVICES=0 $PY train_probe.py --cfg kbsm_L9 --input vis --seed 0 --lr 2e-4 --epochs 80 --out /tmp/wam_repro > /tmp/wam_repro.log 2>&1 &
# 2) global path at ~232 tokens (identical sharding as every other arm)
export WAM_FEAT_ROOT=$W/feats
for s in 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs kbsm_L10g --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_L10g_$s.log 2>&1 & done
wait
CUDA_VISIBLE_DEVICES=0 $PY extract.py --configs kbsm_L10g --shard 0 --nshards 6 --batch 8 > $W/feats/logs/extract_L10g_0.log 2>&1
cmp /tmp/wam_repro/perclip_ade.npy $W/runs/test_main/kbsm_L9/vis/s0/perclip_ade.npy && echo "REPRO OK" || echo "REPRO MISMATCH"
grep -h "K=" $W/feats/logs/extract_L10g_0.log
# 3) crops (needs all raw videos)
until grep -q "RAW DONE" $W/tmp/fetch_raw.log; do sleep 60; done
grep "RAW DONE" $W/tmp/fetch_raw.log
$PY crop_hands.py --S 128 --workers 16
# 4) encode crops
for s in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$s $PY extract_crops.py --S 128 --scheds dense,c11,c9 --srcs raw,d256 --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_crops_$s.log 2>&1 & done
wait
grep -h "K=" $W/feats/logs/extract_crops_0.log
echo "A PREP DONE"
