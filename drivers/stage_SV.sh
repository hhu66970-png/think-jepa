#!/bin/bash
# Round S (priority-signal controls, CV) and Round V (probes saved for inference-time interventions and
# trajectory / attention visualisation). GPUs 2 and 3 host another user's services and are not used.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
cat >> PROTOCOL.md <<'P'
- 2026-09-30, ROUND S SPECIFICATION (written before any Round S run; advisor review of the earlier
  submission: is the gain of a priority signal due to the signal, its spatial correspondence, or its
  direction?). One framework, all with PA, 309 tokens, restore read-out, 5-fold CV over the 1620
  non-selection clips (folds of A2b/K/L/P), seeds 0-3, test split unused. Arms: kbsmpa (similarity order),
  wampa (protect high feature change), wamshufpa (the same scores permuted over all positions of the clip:
  same distribution, no correspondence), wamshuffpa (permuted within each frame: keeps the temporal profile),
  waminvpa (inverted: protect the least changing tokens), wamcomppa (camera-motion compensated: per-frame
  spatial median subtracted), randrankpa (uniform random ranks), handmultpa (task-relevant hand prior).
  Unit test test_priority_controls.py checks the controls. Descriptive, no success criterion; the reading is
  fixed now: if wampa is not better than wamshufpa and waminvpa, the feature-change signal carries no useful
  spatial information for this task; if wamcomppa is better than wampa, camera motion contaminates it.
- 2026-09-30, ROUND V SPECIFICATION (inference-time interventions; exploratory on the reused part-2 test
  split, no model selection). Probes retrained with --save_model (deterministic, so they reproduce the
  published runs): dense vis and dense vispose_k1 (current pose), kbsmpa/wampa/handmultpa L9 vispose_k1,
  seeds 0-4. Interventions on the frozen probes, no retraining: (1) video swapped with another test clip's
  (fixed derangement), poses kept; (2) all visual tokens replaced by the training-mean token; (3) at the same
  budget (25% and 10% of the 4096 positions) the tokens of the highest feature-change cells, random cells,
  hand cells, or lowest-change cells replaced by the training-mean token of that position. Also recorded:
  the probe's cross-attention and the predicted trajectories, for visualisation.
P
J=$W/runs/viz/jobs_V.jsonl; rm -f $J
$PY make_jobs.py viz --cfgs dense --inputs vis --seeds 0-4 --extra="$HP --save_model" --out $J
$PY make_jobs.py viz --cfgs dense,kbsmpa_L9,wampa_L9,handmultpa_L9 --inputs vispose --seeds 0-4 --tag _k1 --extra="$HP --pose_frames 1 --save_model" --out $J
$PY gpu_pool.py $J --gpus 0,0,1,1,4,4,5,5 --per_gpu 1 > $W/tmp/stage_V_pool.log 2>&1 &
POOLV=$!
CF=wamshufpa_L9,wamshuffpa_L9,waminvpa_L9,wamcomppa_L9,randrankpa_L9
G=(0 1 4 5 0 1)
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${G[$s]} $PY extract.py --configs $CF --shard $s --nshards 6 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_S_$s.log 2>&1 &
done
wait
echo "S EXTRACT + V TRAIN DONE"
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_S_0.log
J=$W/runs/cv/jobs_S.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs wamshufpa_L9,wamshuffpa_L9,waminvpa_L9,wamcomppa_L9,randrankpa_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
done
$PY gpu_pool.py $J --gpus 0,0,0,1,1,1,4,4,4,5,5,5 --per_gpu 1
echo "S DONE"
