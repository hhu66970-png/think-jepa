#!/bin/bash
# Round K: compact-token interface (probe reads only the surviving tokens), 5-fold CV over non-selection clips.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
cat >> PROTOCOL.md <<'P'
- 2026-09-29, ROUND K PRE-REGISTRATION (compact-token interface; answers the review point that the
  restore read-out still attends to 4096 positions). Read-out compact_logsize (the probe attends only to
  the surviving tokens, position = mean embedding of member cells, key bias log size). Arms at 309 tokens:
  kbsm, kbsmpa, wam (motion gate), handmult (hand gate), caps2pa; restore read-out added for wam and
  handmult so that every arm has both interfaces. Protocol: 5-fold CV over the 1620 non-selection train
  clips (the same folds as A2b), seeds 0-3; the test split is not used. Question (descriptive, no method
  claim): is the sign of every arm's difference to its reference (gates vs kbsm, caps2pa vs kbsmpa,
  kbsmpa vs kbsm) the same under both interfaces, and how large is compact - restore per arm?
P
J=$W/runs/cv/jobs_K.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs kbsm_L9,kbsmpa_L9,wam_L9,handmult_L9,caps2pa_L9 --inputs vis --seeds 0-3 --tag _compact_cv$f --extra="$HP --cv_fold $f --readout compact_logsize" --out $J
  $PY make_jobs.py cv --cfgs wam_L9,handmult_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
done
$PY gpu_pool.py $J --gpus 0,0,1,1,2,2,3,3,4,4,4,5,5,5 --per_gpu 1
echo "K DONE"
