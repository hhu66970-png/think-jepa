#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
until grep -q "TEST DONE" $W/tmp/stage_test.log; do sleep 60; done
cd $W/repo/wam_forecast
WAM_FEAT_ROOT=$WAM_SHM/feats $PY gpu_pool.py $W/runs/test_main/jobs_budget.jsonl --gpus 0,1,2,3,4,5 --per_gpu 2
WAM_FEAT_ROOT=$WAM_SHM/feats $PY gpu_pool.py $W/runs/test_main/jobs_vispose.jsonl --gpus 0,1,2,3,4,5 --per_gpu 2
WAM_FEAT_ROOT=$WAM_SHM/feats $PY aggregate.py $W/runs/test_main --input vis --pairs handmult_L9:kbsm_L9,wam_L9:kbsm_L9,randrank_L9:kbsm_L9,handsrc_L9:kbsm_L9,handmult_L9:randrank_L9,kbsm_L9:dense,handmult_L9:dense,wam_L9:dense,handmult_s25:kbsm_s25,wam_s25:kbsm_s25,randrank_s25:kbsm_s25,handmult_L12:kbsm_L12,wam_L12:kbsm_L12,randrank_L12:kbsm_L12
echo RERUN DONE
# 1) efficiency on idle GPUs
export WAM_FEAT_ROOT=$W/feats
CUDA_VISIBLE_DEVICES=0 $PY bench.py --configs dense,kbsm_s25,wam_s25,handmult_s25,kbsm_L9,wam_L9,handmult_L9,randrank_L9,pitome_L9,kbsm_L12,wam_L12,handmult_L12 --batch 1,8 --out $W/feats/bench_vitl.json > $W/tmp/bench_vitl.log 2>&1
echo "BENCH DONE"
# 2) ViT-g extraction (all 2000 clips)
G=dense,kbsm_gL9,wam_gL9,handmult_gL9,randrank_gL9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs $G --shard $s --nshards 6 --batch 4 \
    > $W/feats/logs/extract_vitg_$s.log 2>&1 &
done
wait
echo "VITG EXTRACT DONE"
# 3) ViT-g probes: merged on shm, dense (2000x4096x1408 fp16 = 23 GB) does not fit one 3090 -> skip dense-g probe
S=$WAM_SHM/feats
for c in vitg_kbsm_gL9 vitg_wam_gL9 vitg_handmult_gL9 vitg_randrank_gL9; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/test_vitg/jobs.jsonl; rm -f $J
$PY make_jobs.py test_vitg --cfgs vitg_kbsm_gL9,vitg_wam_gL9,vitg_handmult_gL9,vitg_randrank_gL9 --inputs vis --seeds 0-9 --extra="$HP --rows_from vitg_kbsm_gL9" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
$PY aggregate.py $W/runs/test_vitg --input vis --pairs vitg_handmult_gL9:vitg_kbsm_gL9,vitg_wam_gL9:vitg_kbsm_gL9,vitg_randrank_gL9:vitg_kbsm_gL9
echo "AFTER DONE"
