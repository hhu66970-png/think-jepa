#!/bin/bash
# P4 learned relevance: dense probe -> train saliency -> layer-12 cache -> scorer -> new arms.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
until grep -q "CONFIRM DONE" $W/tmp/stage_confirm.log 2>/dev/null; do sleep 60; done
export WAM_FEAT_ROOT=$S
CUDA_VISIBLE_DEVICES=0 $PY train_probe.py --cfg dense --input vis --seed 100 $HP --save_model \
  --out $W/runs/p4/dense_probe_train > $W/tmp/p4_probe.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY saliency.py --model $W/runs/p4/dense_probe_train/model.pt
cp $S/saliency_train.npz $W/feats/
export WAM_FEAT_ROOT=$W/feats
CUDA_VISIBLE_DEVICES=1 $PY cache_hidden.py --layer 12
CUDA_VISIBLE_DEVICES=1 $PY fit_scorer.py --layer 12
echo "P4 SCORER DONE"
NEW="lrecv15_L9,lrecv30_L9,lexcl15_L9,lmult_L9"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $NEW --shard $s --nshards 6 --batch 8 \
    > $W/feats/logs/extract_p4_$s.log 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 $PY alloc_metrics.py --split train --cfgs kbsm_L9,$NEW --out $W/feats/alloc_p4.json 2>&1 | grep -v Warn | grep -v "tok = "
for c in ${NEW//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/select/jobs_p4.jsonl; rm -f $J
$PY make_jobs.py select --cfgs $NEW --inputs vis --seeds 0-4 --extra="--val $HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
PAIRS=$(for c in ${NEW//,/ }; do printf "%s:kbsm_L9," $c; done)
$PY aggregate.py $W/runs/select --input vis --pairs ${PAIRS%,}
BEST=$($PY - <<'P'
import json
d = json.load(open("$WAM_ROOT/runs/select/summary_vis.json"))
new = "lrecv15_L9,lrecv30_L9,lexcl15_L9,lmult_L9".split(",")
print(min((r for r in d["summary"] if r["cfg"] in new), key=lambda r: r["ade"])["cfg"])
P
)
echo "P4 BEST (val): $BEST"
J=$W/runs/test_main/jobs_p4.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs $BEST --inputs vis --seeds 0-29 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY aggregate.py $W/runs/test_main --input vis --pairs $BEST:kbsm_L9,$BEST:dense
echo "P4 TEST DONE"
