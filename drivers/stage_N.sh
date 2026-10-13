#!/bin/bash
# Round N: fresh 57-token budget, P1, ViT-L + ViT-g. Other users' services occupy ~8.4 GB on GPUs 0-3.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
F=$W/feats
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$F
while ! grep -q "finished" $W/tmp/stage_M_rerun.log; do sleep 60; done
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs kbsmpa_gL15,caps2pa_gL15,tomepa_gL15,pitomepa_gL15 --shard $s --nshards 6 --batch 4 > $F/logs/extract_Ng_$s.log 2>&1 &
done
wait
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs kbsmpa_L15,caps2pa_L15,tomepa_L15,pitomepa_L15 --shard $s --nshards 6 --batch 8 > $F/logs/extract_Nl_$s.log 2>&1 &
done
wait
grep -h "K=" $F/logs/extract_Ng_0.log $F/logs/extract_Nl_0.log
grep -il traceback $F/logs/extract_N*_*.log && { echo "N EXTRACT FAILED"; exit 1; }
X="$HP --pose_frames 1"
J=$W/runs/jobs_N.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs kbsmpa_L15,caps2pa_L15 --inputs vispose --seeds 0-19 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_main --cfgs tomepa_L15,pitomepa_L15 --inputs vispose --seeds 0-9 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_vitg --cfgs vitg_kbsmpa_gL15,vitg_caps2pa_gL15 --inputs vispose --seeds 0-19 --tag _k1 --extra="$X --rows_from vitg_kbsm_gL9" --out $J
$PY make_jobs.py test_vitg --cfgs vitg_tomepa_gL15,vitg_pitomepa_gL15 --inputs vispose --seeds 0-9 --tag _k1 --extra="$X --rows_from vitg_kbsm_gL9" --out $J
$PY gpu_pool.py $J --gpus 0,0,1,1,2,2,3,3,4,4,4,5,5,5 --per_gpu 1
echo "N DONE"
