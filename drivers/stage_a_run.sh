#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
until grep -q "A PREP DONE" $W/tmp/stage_a_prep.log; do sleep 60; done
mkdir -p $S/crops_S128
cp $W/feats/crops_S128/boxes.npy $W/feats/crops_S128/hand_ok.npy $W/feats/crops_S128/align.npy $S/crops_S128/
for c in kbsm_L10g crop_raw_h0_c11 crop_raw_h1_c11 crop_d256_h0_c11 crop_d256_h1_c11 crop_raw_h0_dense crop_raw_h1_dense; do
  mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/
done
du -sh $S
export WAM_FEAT_ROOT=$S
# ---- regression gate (validation): coordinate read-out vs published restore read-out
J=$W/runs/select/jobs_agate.jsonl; rm -f $J
$PY make_jobs.py select --cfgs kbsm_L9 --inputs vis --seeds 0-4 --tag _coords --extra="--val $HP --coords" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
GATE=$($PY pairs.py $W/runs/select kbsm_L9/vis_coords:kbsm_L9/vis)
echo "GATE $GATE"
PASS=$($PY -c "
import re,sys
m=re.search(r'd=([+-][0-9.]+)', '''$GATE'''); print('yes' if m and abs(float(m.group(1)))<1.0 else 'no')")
echo "GATE PASS: $PASS"
if [ "$PASS" != "yes" ]; then echo "A STOPPED AT GATE"; exit 0; fi
# ---- test arms
J=$W/runs/test_main/jobs_a.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs kbsm_L9 --inputs vis --seeds 0-19 --tag _coords --extra="$HP --coords" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L10g --inputs vis --seeds 0-19 --tag _coords_d256 --extra="$HP --coords --crops d256 --crop_sched c11" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L10g --inputs vis --seeds 0-19 --tag _coords_raw --extra="$HP --coords --crops raw --crop_sched c11" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L10g --inputs vis --seeds 0-19 --tag _coords_rawdense --extra="$HP --coords --crops raw --crop_sched dense" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L9 --inputs vispose --seeds 0-9 --tag _coords --extra="$HP --coords" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L10g --inputs vispose --seeds 0-9 --tag _coords_raw --extra="$HP --coords --crops raw --crop_sched c11" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 2
J=$W/runs/test_main/jobs_a_dense.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs dense --inputs vis --seeds 0-19 --tag _coords --extra="$HP --coords" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
echo "== A results"
$PY pairs.py $W/runs/test_main \
  kbsm_L10g/vis_coords_raw:kbsm_L9/vis_coords \
  kbsm_L10g/vis_coords_raw:kbsm_L10g/vis_coords_d256 \
  kbsm_L10g/vis_coords_d256:kbsm_L9/vis_coords \
  kbsm_L10g/vis_coords_rawdense:kbsm_L10g/vis_coords_raw \
  kbsm_L10g/vis_coords_raw:dense/vis_coords \
  kbsm_L9/vis_coords:dense/vis_coords \
  kbsm_L9/vis_coords:kbsm_L9/vis \
  kbsm_L10g/vispose_coords_raw:kbsm_L9/vispose_coords \
  kbsm_L9/vispose_coords:kbsm_L9/vispose
echo "A RUN DONE"
