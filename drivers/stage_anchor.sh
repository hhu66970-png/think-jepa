#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
until grep -q "P4 TEST DONE" $W/tmp/stage_p4b.log; do sleep 60; done
export WAM_FEAT_ROOT=$W/feats
NEW="rrecv15_s25,rrecv15_L12,lat4_s25,lat4_L9,lat4_L12,lat2_s25,lat2_L9,lat2_L12"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $NEW --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_anc_$s.log 2>&1 &
done
wait
echo "ANCHOR EXTRACT DONE"
for c in ${NEW//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/select/jobs_anchor.jsonl; rm -f $J
$PY make_jobs.py select --cfgs $NEW,kbsm_s25,kbsm_L12 --inputs vis --seeds 0-4 --extra="--val $HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 2
$PY aggregate.py $W/runs/select --input vis --pairs rrecv15_s25:kbsm_s25,rrecv15_L9:kbsm_L9,rrecv15_L12:kbsm_L12,lat4_s25:kbsm_s25,lat4_L9:kbsm_L9,lat4_L12:kbsm_L12,lat2_s25:kbsm_s25,lat2_L9:kbsm_L9,lat2_L12:kbsm_L12
BEST=$($PY - <<'P'
import json
d = {r["cfg"]: r["ade"] for r in json.load(open("$WAM_ROOT/runs/select/summary_vis.json"))["summary"]}
score = {}
for arm in ("rrecv15", "lat4", "lat2"):
    score[arm] = sum(d[f"{arm}_{b}"] - d[f"kbsm_{b}"] for b in ("s25", "L9", "L12")) / 3
print(min(score, key=score.get))
P
)
echo "ANCHOR BEST (val): $BEST"
J=$W/runs/test_main/jobs_anchor.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs ${BEST}_s25,${BEST}_L9,${BEST}_L12 --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 2
$PY aggregate.py $W/runs/test_main --input vis --pairs ${BEST}_s25:kbsm_s25,${BEST}_L9:kbsm_L9,${BEST}_L12:kbsm_L12,lrecv30_L9:kbsm_L9,mrecv15_L9:kbsm_L9
echo "ANCHOR TEST DONE"
