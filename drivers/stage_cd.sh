#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$WAM_SHM/feats
J=$W/runs/test_main/jobs_c.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs kbsm_L9 --inputs vis --seeds 0-19 --tag _compact --extra="$HP --readout compact" --out $J
$PY make_jobs.py test_main --cfgs kbsm_L9,kbsm_s25,kbsm_L12 --inputs vis --seeds 0-19 --tag _compact_logsize --extra="$HP --readout compact_logsize" --out $J
$PY make_jobs.py select --cfgs kbsm_L9 --inputs vis --seeds 0-4 --tag _compact --extra="--val $HP --readout compact" --out $J
$PY make_jobs.py select --cfgs kbsm_L9 --inputs vis --seeds 0-4 --tag _compact_logsize --extra="--val $HP --readout compact_logsize" --out $J
J2=$W/runs/test_main/jobs_d.jsonl; rm -f $J2
$PY make_jobs.py test_main --cfgs mix:motion,mix:inverse,mix:random --inputs vis --seeds 0-19 --extra="$HP" --out $J2
cat $J $J2 > $W/runs/test_main/jobs_cd.jsonl
$PY gpu_pool.py $W/runs/test_main/jobs_cd.jsonl --gpus 0,1,2,3,4,5 --per_gpu 3
echo "== C: difference in differences"
$PY did.py $W/runs/test_main kbsm_L9 vis_compact_logsize
$PY did.py $W/runs/test_main kbsm_L9 vis_compact
$PY did.py $W/runs/test_main kbsm_s25 vis_compact_logsize
$PY did.py $W/runs/test_main kbsm_L12 vis_compact_logsize
echo "== D"
$PY aggregate.py $W/runs/test_main --input vis --pairs mix:motion:kbsm_L9,mix:motion:mix:random,mix:inverse:kbsm_L9,mix:random:kbsm_L9 | tail -8
echo "CD DONE"
