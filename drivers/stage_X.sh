#!/bin/bash
# Round X: the three tests on 552 new episodes (PROTOCOL). Split, poses, masks, features (6 GPUs), 150 probe runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_SLIM=$W/data/egodex_x/slim WAM_SPLITS=$W/data/egodex_x/splits WAM_FEAT_ROOT=$W/feats_X WAM_RUN_ROOT=$W/runs_X
mkdir -p $WAM_FEAT_ROOT/logs $WAM_RUN_ROOT
$PY build_xsplit.py
[ -f $WAM_FEAT_ROOT/poses.npz ] || $PY prep_poses.py
[ -f $WAM_FEAT_ROOT/hand_masks.npz ] || $PY hand_masks.py
L="dense,kbsmpa_L9,kbsmpa_L12,wampa_L9,wamshufpa_L9,handcurpa_L12,handcurshuffpa_L12,handf31pa_L12,handf31shuffpa_L12,vidtldrpa_L9,fmpa_L9,vidtldrnat_L9,fmnat_L9"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $L --shard $s --nshards 6 --batch 4 > $WAM_FEAT_ROOT/logs/extract_X_$s.log 2>&1 &
done
wait
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_X_0.log
grep -il traceback $WAM_FEAT_ROOT/logs/extract_X_*.log && { echo "X EXTRACT FAILED"; exit 1; }
echo "X EXTRACT DONE"
X="--lr 2e-4 --epochs 80 --pose_frames 1"
J=$WAM_RUN_ROOT/jobs_X.jsonl; JD=$WAM_RUN_ROOT/jobs_X_dense.jsonl; rm -f $J $JD
$PY make_jobs.py test_main --cfgs ${L#dense,} --inputs pose,vispose --seeds 0-9 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_main --cfgs dense --inputs vispose --seeds 0-9 --tag _k1 --extra="$X" --out $JD
$PY gpu_pool_claim.py $J --gpus ${X_GPUS:-0,0,1,1,2,2,3,3,4,4,5,5} --per_gpu 1
$PY gpu_pool_claim.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
echo "X CV DONE"
$PY roundX_analysis.py
echo "X ANALYSIS DONE"
