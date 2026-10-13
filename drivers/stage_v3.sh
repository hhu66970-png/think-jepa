#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
NEW="hrecv15_L9,hrecv30_L9,hexcl15_L9,hexcl30_L9,hanc1_L9,hanc4_L9,hrecvanc_L9,mrecv15_L9,manc4_L9"
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $NEW --shard $s --nshards 6 --batch 8 \
    > $W/feats/logs/extract_v3_$s.log 2>&1 &
done
wait
echo "V3 EXTRACT DONE"
CUDA_VISIBLE_DEVICES=0 $PY alloc_metrics.py --split train --cfgs kbsm_L9,$NEW \
  --out $W/feats/alloc_v3.json 2>&1 | grep -v Warn | grep -v "tok = "
for c in ${NEW//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
cp $W/feats/hand_masks.npz $S/
export WAM_FEAT_ROOT=$S
J=$W/runs/select/jobs_v3.jsonl; rm -f $J
$PY make_jobs.py select --cfgs $NEW --inputs vis --seeds 0-4 --extra="--val $HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
PAIRS=$(for c in ${NEW//,/ }; do printf "%s:kbsm_L9," $c; done)
$PY aggregate.py $W/runs/select --input vis --pairs ${PAIRS%,}
echo "V3 SELECT DONE"
BEST=$($PY - <<'P'
import json
d = json.load(open("$WAM_ROOT/runs/select/summary_vis.json"))
new = "hrecv15_L9,hrecv30_L9,hexcl15_L9,hexcl30_L9,hanc1_L9,hanc4_L9,hrecvanc_L9,mrecv15_L9,manc4_L9".split(",")
rows = sorted([r for r in d["summary"] if r["cfg"] in new], key=lambda r: r["ade"])
print(",".join(r["cfg"] for r in rows[:2]))
P
)
echo "V3 BEST (val): $BEST"
J=$W/runs/test_main/jobs_v3.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs $BEST --inputs vis --seeds 0-29 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
B1=${BEST%,*}; B2=${BEST#*,}
$PY aggregate.py $W/runs/test_main --input vis --pairs $B1:kbsm_L9,$B2:kbsm_L9,$B1:dense,$B2:dense,handmult_L9:kbsm_L9,wam_L9:kbsm_L9
echo "V3 TEST DONE"
