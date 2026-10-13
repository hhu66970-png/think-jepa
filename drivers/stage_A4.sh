#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
while ! grep -q "P1 DONE" $W/tmp/stage_P1.log; do sleep 60; done
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --arch vitg --configs kbsmpa_gL9,caps2pa_gL9 --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_A4g_$s.log 2>&1 &
done
wait
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs caps2_L9,caps2_L12 --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_A4b_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_A4g_0.log $W/feats/logs/extract_A4b_0.log
grep -il traceback $W/feats/logs/extract_A4*_*.log && { echo "A4 EXTRACT FAILED"; exit 1; }
for c in vitg_kbsmpa_gL9 vitg_caps2pa_gL9 caps2_L9 caps2_L12; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
[ -d $S/vitg_kbsm_gL9 ] || { mkdir -p $S/vitg_kbsm_gL9 && cp $W/feats/vitg_kbsm_gL9/* $S/vitg_kbsm_gL9/; }
df -h /dev/shm | tail -1
export WAM_FEAT_ROOT=$S
J=$W/runs/test_vitg/jobs_A4a.jsonl; rm -f $J
$PY make_jobs.py test_vitg --cfgs vitg_kbsmpa_gL9,vitg_caps2pa_gL9 --inputs vis --seeds 0-19 --extra="$HP --rows_from vitg_kbsm_gL9" --out $J
J2=$W/runs/cv/jobs_A4b.jsonl; rm -f $J2
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs caps2_L9,kbsm_L9,caps2_L12,kbsm_L12 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J2
done
cat $J >> $J2.all; cat $J2 >> $J2.all
$PY gpu_pool.py $J2.all --gpus 0,1,2,3,4,5 --per_gpu 3
$PY pairs.py $W/runs/test_vitg vitg_caps2pa_gL9/vis:vitg_kbsmpa_gL9/vis vitg_kbsmpa_gL9/vis:vitg_kbsm_gL9/vis
$PY phaseA_analysis.py cv caps2 kbsm L9,L12
echo "A4 DONE"
