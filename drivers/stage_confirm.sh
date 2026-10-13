#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
until grep -q "FIXUP DONE" $W/tmp/fixup_g1.log; do sleep 60; done
export WAM_FEAT_ROOT=$W/feats
L="mrecv15_s25,mrecv15_L12,rrecv15_L9,mrecv10_L9,mrecv20_L9"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $L --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_cf_$s.log 2>&1 &
done
wait
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs mrecv15_gL9 --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_cfg_$s.log 2>&1 &
done
wait
echo "CONFIRM EXTRACT DONE"
for c in mrecv15_s25 mrecv15_L12 rrecv15_L9 mrecv10_L9 mrecv20_L9 vitg_mrecv15_gL9; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/test_main/jobs_confirm.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs rrecv15_L9 --inputs vis --seeds 0-29 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs mrecv15_s25,mrecv15_L12 --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs kbsm_s25,kbsm_L12 --inputs vis --seeds 10-19 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs mrecv15_L9 --inputs vispose --seeds 0-9 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 2
J=$W/runs/test_vitg/jobs_confirm.jsonl; rm -f $J
$PY make_jobs.py test_vitg --cfgs vitg_mrecv15_gL9 --inputs vis --seeds 0-9 --extra="$HP --rows_from vitg_kbsm_gL9" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
J=$W/runs/select/jobs_cf.jsonl; rm -f $J
$PY make_jobs.py select --cfgs mrecv10_L9,mrecv20_L9,rrecv15_L9 --inputs vis --seeds 0-4 --extra="--val $HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY aggregate.py $W/runs/test_main --input vis --pairs mrecv15_L9:kbsm_L9,mrecv15_L9:rrecv15_L9,rrecv15_L9:kbsm_L9,mrecv15_s25:kbsm_s25,mrecv15_L12:kbsm_L12,mrecv15_L9:dense,hrecv15_L9:kbsm_L9,handmult_L9:kbsm_L9,wam_L9:kbsm_L9
$PY aggregate.py $W/runs/test_main --input vispose --pairs mrecv15_L9:kbsm_L9,handmult_L9:kbsm_L9,wam_L9:kbsm_L9,kbsm_L9:dense
$PY aggregate.py $W/runs/test_vitg --input vis --pairs vitg_mrecv15_gL9:vitg_kbsm_gL9,vitg_handmult_gL9:vitg_kbsm_gL9,vitg_wam_gL9:vitg_kbsm_gL9,vitg_randrank_gL9:vitg_kbsm_gL9
$PY aggregate.py $W/runs/select --input vis --pairs mrecv10_L9:kbsm_L9,mrecv15_L9:kbsm_L9,mrecv20_L9:kbsm_L9,rrecv15_L9:kbsm_L9
echo "CONFIRM DONE"
