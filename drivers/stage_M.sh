#!/bin/bash
# Round M: P1 baselines (ToMe+PA, PiToMe+PA), ViT-g P1, CV confirmation. Features read from disk.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
F=$W/feats
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$F
G="kbsmpa_gs25,kbsmpa_gL12,caps2pa_gs25,caps2pa_gL12,tomepa_gs25,tomepa_gL9,tomepa_gL12,pitomepa_gs25,pitomepa_gL9,pitomepa_gL12"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs $G --shard $s --nshards 6 --batch 4 > $F/logs/extract_Mg_$s.log 2>&1 &
done
wait
L="tomepa_s25,tomepa_L9,tomepa_L12,pitomepa_s25,pitomepa_L9,pitomepa_L12"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $L --shard $s --nshards 6 --batch 8 > $F/logs/extract_Ml_$s.log 2>&1 &
done
wait
grep -h "K=" $F/logs/extract_Mg_0.log $F/logs/extract_Ml_0.log
grep -il traceback $F/logs/extract_M*_*.log && { echo "M EXTRACT FAILED"; exit 1; }
X="$HP --pose_frames 1"
J=$W/runs/jobs_M12.jsonl; rm -f $J
# M1 ViT-L P1 baselines
$PY make_jobs.py test_main --cfgs tomepa_s25,tomepa_L9,tomepa_L12,pitomepa_s25,pitomepa_L9,pitomepa_L12 --inputs vispose --seeds 0-19 --tag _k1 --extra="$X" --out $J
# M2 ViT-g P1
XG="$X --rows_from vitg_kbsm_gL9"
$PY make_jobs.py test_vitg --cfgs vitg_kbsmpa_gs25,vitg_kbsmpa_gL9,vitg_kbsmpa_gL12,vitg_caps2pa_gs25,vitg_caps2pa_gL9,vitg_caps2pa_gL12 --inputs vispose --seeds 0-19 --tag _k1 --extra="$XG" --out $J
$PY make_jobs.py test_vitg --cfgs vitg_tomepa_gs25,vitg_tomepa_gL9,vitg_tomepa_gL12,vitg_pitomepa_gs25,vitg_pitomepa_gL9,vitg_pitomepa_gL12,vitg_kbsm_gL9 --inputs vispose --seeds 0-9 --tag _k1 --extra="$XG" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
echo "M12 DONE"
# M3 CV (ViT-L, P1)
J3=$W/runs/cv/jobs_M3.jsonl; rm -f $J3
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs caps2pa_s25,kbsmpa_s25,caps2pa_L9,kbsmpa_L9,caps2pa_L12,kbsmpa_L12 --inputs vispose --seeds 0-3 --tag _k1_cv$f --extra="$X --cv_fold $f" --out $J3
done
$PY gpu_pool.py $J3 --gpus 0,1,2,3,4,5 --per_gpu 3
$PY phaseA_analysis.py cv caps2pa kbsmpa L9,L12,s25 vispose_k1
echo "M DONE"
