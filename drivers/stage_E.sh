#!/bin/bash
# Round E: a second encoder family (VideoMAE ViT-L, self-supervised, absolute positions, no rotary embedding).
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
# pinned: revision 12da269a02d3e1fbbb7011e610e591fea8061dca, the only snapshot in the cache when this round ran
export VIDEOMAE_PATH=$W/cache/hf/hub/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca/
grep -q "ROUND E SPECIFICATION" PROTOCOL.md || cat >> PROTOCOL.md <<'P'
- 2026-10-01, ROUND E SPECIFICATION (written before any Round E feature or probe run; answers the review point
  that every result comes from one encoder family). Encoder: VideoMAE ViT-L (MCG-NJU/videomae-large, self-supervised
  on Kinetics-400, absolute sinusoidal positions, no rotary embedding), frozen; input = every second observed frame
  (16 frames, 224 px), grid 8 x 14 x 14 = 1568 tokens; the probe reads the 16^3 grid through the covering cell.
  Merger: the identical token_merge_diagnostics implementation (head-mean attention key of the merge layer; without
  rotary positions K-BSM and ToMe coincide), layers 12-20, r = 25% (1568 -> 118 tokens [estimate; integer rounding gives 119, see the outcome entry], the same 7.5% as 4096 -> 309),
  PA as for V-JEPA. Arms: vmae_kbsm_L9, vmae_kbsmpa_L9, vmae_wampa_L9 (motion gate + PA), vmae_handmultpa_L9 (hand
  gate + PA, hand mask max-pooled to the VideoMAE grid). Protocol: 5-fold CV over the 1620 non-selection clips
  (folds of A2b/K/L), seeds 0-3, restore read-out, video only; test split unused. Unit check before the runs: the
  hand-written layer reproduces VideoMAEModel on real clips. Criteria (replication of the paper's conclusions on a
  second family): E1 PA helps: kbsmpa - kbsm clip x seed 95% CI below 0. E2 neither gate has an advantage over
  kbsmpa (clip x seed 97.5% CI below 0 would be an advantage). Reported whatever the outcome.
P
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s $PY extract_videomae.py --configs dense,kbsm_L9,kbsmpa_L9,wampa_L9,handmultpa_L9 --shard $s --nshards 2 --batch 8 \
    > $WAM_FEAT_ROOT/logs/extract_E_$s.log 2>&1 &
done
wait
echo "E EXTRACT DONE"; grep -h "K=" $WAM_FEAT_ROOT/logs/extract_E_*.log
J=$W/runs/cv/jobs_E.jsonl; rm -f $J
for f in 0 1 2 3 4; do
  $PY make_jobs.py cv --cfgs vmae_kbsm_L9,vmae_kbsmpa_L9,vmae_wampa_L9,vmae_handmultpa_L9 --inputs vis --seeds 0-3 --tag _cv$f --extra="$HP --cv_fold $f" --out $J
done
$PY gpu_pool.py $J --gpus ${E_GPUS:-0,0,0,1,1,1} --per_gpu 1
echo "E DONE"
