#!/bin/bash
# Round L: gates with PA and public baselines with PA vs K-BSM+PA, fresh 5-fold CV, both interfaces.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
cat >> PROTOCOL.md <<'P'
- 2026-09-29, ROUND L SPECIFICATION (written before any Round L run; answers the review point that the
  gate+PA comparisons exist only on the reused test split). Arms at 309 tokens: wampa (motion gate + PA),
  handmultpa (hand gate + PA), tomepa (ToMe + PA, pre-rotary keys), pitomepa (PiToMe + PA); reference kbsmpa
  (existing CV runs, same folds and seeds). Protocol: 5-fold CV over the 1620 non-selection train clips (the
  folds of A2b/K), seeds 0-3, restore read-out (vis) and compact read-out (compact_logsize); the test split is
  not used. Primary question (confirmatory): does either relevance gate with PA have an advantage over kbsmpa
  under the restore read-out? Criterion: clip x seed 95% interval of (gate - kbsmpa) entirely below 0 after a
  Bonferroni correction over the two gates (97.5% interval). Secondary (descriptive, no success criterion):
  the same differences under the compact read-out, and tomepa / pitomepa vs kbsmpa under both read-outs.
P
J=$W/runs/cv/jobs_L.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs wampa_L9,handmultpa_L9,tomepa_L9,pitomepa_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
  $PY make_jobs.py cv --cfgs wampa_L9,handmultpa_L9,tomepa_L9,pitomepa_L9 --inputs vis --seeds 0-3 --tag _compact_cv$f --extra="$HP --cv_fold $f --readout compact_logsize" --out $J
done
wc -l $J
$PY gpu_pool.py $J --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --per_gpu 1
echo "L DONE"
