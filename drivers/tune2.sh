#!/bin/bash
W=$WAM_ROOT
export WAM_FEAT_ROOT=$WAM_SHM/feats_pilot
cd $W/repo/wam_forecast
PY=$W/conda/bin/python
J=$W/runs/pilot_tune/jobs2.jsonl; rm -f $J
$PY make_jobs.py pilot_tune/A --cfgs dense --inputs vis --seeds 2-3 --extra="--val " --out $J
$PY make_jobs.py pilot_tune/F_lr2e4 --cfgs dense --inputs vis --seeds 2-3 --extra="--val --lr 2e-4 --epochs 80" --out $J
$PY make_jobs.py pilot_tune/G_lr2e4_ep40 --cfgs dense --inputs vis --seeds 0-3 --extra="--val --lr 2e-4 --epochs 40" --out $J
$PY make_jobs.py pilot_tune/H_lr1e4_ep80 --cfgs dense --inputs vis --seeds 0-3 --extra="--val --lr 1e-4 --epochs 80" --out $J
$PY make_jobs.py pilot_tune/I_lr2e4_ep80_wd01 --cfgs dense --inputs vis --seeds 0-3 --extra="--val --lr 2e-4 --epochs 80 --wd 0.1" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
echo "TUNE2 DONE"
