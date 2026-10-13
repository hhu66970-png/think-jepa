#!/bin/bash
# Round R: second dataset (EgoDex official test set), P1, ViT-L + ViT-g. Pre-registered in PROTOCOL.md.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
D=$W/data/egodex_test
cd $W/repo/wam_forecast
while [ -f $D/test.zip.aria2 ] || pgrep -f "aria2c -c -x 8" > /dev/null; do sleep 60; done
[ "$(stat -c %s $D/test.zip)" = "17304529397" ] || { echo "DOWNLOAD SIZE MISMATCH"; exit 1; }
echo "download ok"
cd $D && unzip -q -o test.zip -d $D/unzipped || { echo "UNZIP FAILED"; exit 1; }
cd $W/repo/wam_forecast
$PY build_testset.py --src $D/unzipped --out $W/data/egodex_testset --workers 16 || { echo "BUILD FAILED"; exit 1; }
export WAM_SLIM=$W/data/egodex_testset/slim WAM_SPLITS=$W/data/egodex_testset/splits
export WAM_FEAT_ROOT=$W/feats_R WAM_RUN_ROOT=$W/runs_R
mkdir -p $WAM_FEAT_ROOT/logs $WAM_RUN_ROOT
$PY prep_poses.py || { echo "POSES FAILED"; exit 1; }
$PY clip_strata.py
L="dense,kbsmpa_L9,kbsmpa_L12,kbsmpa_L15,tomepa_L9,tomepa_L12,tomepa_L15,pitomepa_L9,pitomepa_L12,pitomepa_L15,caps2pa_L9,caps2pa_L12,caps2pa_L15"
G="kbsmpa_gL9,kbsmpa_gL12,kbsmpa_gL15,tomepa_gL9,tomepa_gL12,tomepa_gL15,pitomepa_gL9,pitomepa_gL12,pitomepa_gL15,caps2pa_gL9,caps2pa_gL12,caps2pa_gL15"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $L --shard $s --nshards 6 --batch 8 > $WAM_FEAT_ROOT/logs/extract_Rl_$s.log 2>&1 &
done
wait
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs $G --shard $s --nshards 6 --batch 4 > $WAM_FEAT_ROOT/logs/extract_Rg_$s.log 2>&1 &
done
wait
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_Rl_0.log $WAM_FEAT_ROOT/logs/extract_Rg_0.log
grep -il traceback $WAM_FEAT_ROOT/logs/extract_R*_*.log && { echo "R EXTRACT FAILED"; exit 1; }
X="$HP --pose_frames 1"
J=$WAM_RUN_ROOT/jobs_R.jsonl; JD=$WAM_RUN_ROOT/jobs_R_dense.jsonl; rm -f $J $JD
$PY make_jobs.py test_main --cfgs ${L#dense,} --inputs pose,vispose --seeds 0-9 --tag _k1 --extra="$X" --out $J
$PY make_jobs.py test_vitg --cfgs $(echo $G | sed "s/\([a-z0-9]*_g[A-Z0-9]*\)/vitg_\1/g") --inputs vispose --seeds 0-9 --tag _k1 --extra="$X --rows_from vitg_kbsmpa_gL9" --out $J
$PY make_jobs.py test_main --cfgs dense --inputs vispose --seeds 0-9 --tag _k1 --extra="$X" --out $JD
$PY gpu_pool.py $J --gpus 0,0,1,1,2,2,3,3,4,4,4,5,5,5 --per_gpu 1
$PY gpu_pool.py $JD --gpus 4,5 --per_gpu 1
$PY roundR_analysis.py
echo "R DONE"
