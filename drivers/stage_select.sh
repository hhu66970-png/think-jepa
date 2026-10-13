#!/bin/bash
# Stage 1 (method selection, validation split of TRAIN only). Usage: stage_select.sh "<probe hparams>"
HP="$1"
W=$WAM_ROOT
S=$WAM_SHM/feats
PY=$W/conda/bin/python
until grep -q "FULL EXTRACT DONE" $W/tmp/full_extract.log; do sleep 30; done
rm -rf $WAM_SHM/feats_pilot
mkdir -p $S
cp $W/feats/poses.npz $W/feats/hand_masks.npz $S/
MER="kbsm_L9,wam_L9,handmult_L9,handq10_L9,handq20_L9,handsrc_L9,wamq10_L9,wamsrc_L9,randrank_L9,pitome_L9"
for c in dense ${MER//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
du -sh $S
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$S
J1=$W/runs/select/jobs_merged.jsonl; J2=$W/runs/select/jobs_dense.jsonl; rm -f $J1 $J2
$PY make_jobs.py select --cfgs $MER --inputs vis --seeds 0-4 --extra="--val $HP" --out $J1
$PY make_jobs.py select --cfgs dense --inputs vis --seeds 0-4 --extra="--val $HP" --out $J2
$PY gpu_pool.py $J1 --gpus 0,1,2,3,4,5 --per_gpu 3
$PY gpu_pool.py $J2 --gpus 0,1,2,3,4,5 --per_gpu 1
$PY aggregate.py $W/runs/select --input vis --pairs wam_L9:kbsm_L9,handmult_L9:kbsm_L9,handq10_L9:kbsm_L9,handq20_L9:kbsm_L9,handsrc_L9:kbsm_L9,wamq10_L9:kbsm_L9,wamsrc_L9:kbsm_L9,randrank_L9:kbsm_L9,pitome_L9:kbsm_L9,kbsm_L9:dense
echo "SELECT DONE"
