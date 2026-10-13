#!/bin/bash
W=$WAM_ROOT
export WAM_FEAT_ROOT=$WAM_SHM/feats_pilot
cd $W/repo/wam_forecast
PY=$W/conda/bin/python
J=$W/runs/pilot_tune/jobs.jsonl; rm -f $J
$PY make_jobs.py pilot_tune/A --cfgs dense --inputs vis --seeds 0-1 --extra "--val" --out $J
$PY make_jobs.py pilot_tune/B_ep80 --cfgs dense --inputs vis --seeds 0-1 --extra "--val --epochs 80" --out $J
$PY make_jobs.py pilot_tune/C_lr1e3 --cfgs dense --inputs vis --seeds 0-1 --extra "--val --lr 1e-3" --out $J
$PY make_jobs.py pilot_tune/D_dim384 --cfgs dense --inputs vis --seeds 0-1 --extra "--val --dim 384" --out $J
$PY make_jobs.py pilot_tune/E_depth2 --cfgs dense --inputs vis --seeds 0-1 --extra "--val --depth 2" --out $J
$PY make_jobs.py pilot_tune/F_lr2e4 --cfgs dense --inputs vis --seeds 0-1 --extra "--val --lr 2e-4 --epochs 80" --out $J
until ! pgrep -f "gpu_pool.py $W/runs/pilot_g1" >/dev/null; do sleep 20; done
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
echo "TUNE DONE"
