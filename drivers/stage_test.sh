#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
S=$WAM_SHM/feats
PY=$W/conda/bin/python
for c in kbsm_s25 handmult_s25 wam_s25 randrank_s25 kbsm_L12 handmult_L12 wam_L12 randrank_L12; do
  mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/
done
du -sh $S
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$S
T=$W/runs/test_main; mkdir -p $T
J1=$T/jobs_merged.jsonl; J2=$T/jobs_dense.jsonl; J3=$T/jobs_budget.jsonl; J4=$T/jobs_vispose.jsonl
rm -f $J1 $J2 $J3 $J4
$PY make_jobs.py test_main --cfgs kbsm_L9,handmult_L9,wam_L9,randrank_L9,handsrc_L9 --inputs vis --seeds 0-29 --extra="$HP" --out $J1
$PY make_jobs.py test_main --cfgs dense --inputs vis --seeds 0-29 --extra="$HP" --out $J2
$PY make_jobs.py test_main --cfgs kbsm_s25,handmult_s25,wam_s25,randrank_s25,kbsm_L12,handmult_L12,wam_L12,randrank_L12 --inputs vis --seeds 0-9 --extra="$HP" --out $J3
$PY make_jobs.py test_main --cfgs kbsm_L9,handmult_L9,wam_L9 --inputs vispose,pose --seeds 0-9 --extra="$HP" --out $J4
$PY gpu_pool.py $J1 --gpus 0,1,2,3,4,5 --per_gpu 3
$PY aggregate.py $T --input vis --pairs handmult_L9:kbsm_L9,wam_L9:kbsm_L9,randrank_L9:kbsm_L9,handsrc_L9:kbsm_L9,handmult_L9:randrank_L9
echo "PRIMARY DONE"
$PY gpu_pool.py $J2 --gpus 0,1,2,3,4,5 --per_gpu 1
$PY gpu_pool.py $J3 --gpus 0,1,2,3,4,5 --per_gpu 3
$PY gpu_pool.py $J4 --gpus 0,1,2,3,4,5 --per_gpu 3
$PY aggregate.py $T --input vis --pairs handmult_L9:kbsm_L9,wam_L9:kbsm_L9,randrank_L9:kbsm_L9,handsrc_L9:kbsm_L9,handmult_L9:randrank_L9,kbsm_L9:dense,handmult_L9:dense,handmult_s25:kbsm_s25,wam_s25:kbsm_s25,randrank_s25:kbsm_s25,handmult_L12:kbsm_L12,wam_L12:kbsm_L12,randrank_L12:kbsm_L12
$PY aggregate.py $T --input vispose --pairs handmult_L9:kbsm_L9,wam_L9:kbsm_L9
echo "TEST DONE"
