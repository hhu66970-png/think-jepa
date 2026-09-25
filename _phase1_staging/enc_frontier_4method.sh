#!/usr/bin/env bash
# Phase1 encoder frontier: 4 methods (dense implicit / 方案A simple-ToMe / K-BSM / PiToMe) + axis ablation (C1).
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0
BASE=outputs/phase1_enc_20260601
mkdir -p "$BASE"
REP=3
echo "==== ENC FRONTIER START $(date +%H:%M:%S) ===="

# 方案A (simple ToMe): early L8 (more merge) + late L16 (high fidelity)
$PY tools/run_token_merge_pca_experiment.py --strategy local_2x2_same_time_vec \
  --baseline_scheme_a_layer 8  --scheme_a_ratios "0.125,0.25" \
  --repeats $REP --warmup 2 --out_dir "$BASE/A_L8"  2>&1 | tail -3
$PY tools/run_token_merge_pca_experiment.py --strategy local_2x2_same_time_vec \
  --baseline_scheme_a_layer 16 --scheme_a_ratios "0.125,0.25" \
  --repeats $REP --warmup 2 --out_dir "$BASE/A_L16" 2>&1 | tail -3

# K-BSM (existing, byte-unchanged): gradual late, free axis
$PY tools/run_token_merge_pca_experiment.py --strategy bsm_ksim_gradual_vec \
  --merge_layers "12-20:2" --r_per_layer "0.08,0.15" --bsm_match_metric key \
  --repeats $REP --warmup 2 --out_dir "$BASE/KBSM_free" 2>&1 | tail -3

# PiToMe (NEW): gradual late, free axis  -> compare vs K-BSM at matched r
$PY tools/run_token_merge_pca_experiment.py --strategy bsm_pitome_gradual_vec \
  --merge_layers "12-20:2" --r_per_layer "0.08,0.15" --bsm_match_metric key \
  --repeats $REP --warmup 2 --out_dir "$BASE/PITOME_free" 2>&1 | tail -3

# Axis ablation (C1 / F2 controlled): same strat+r, vary axis. Single early L8 -> token cut concentrated.
for AX in free spatial temporal; do
  echo "---- AXIS $AX $(date +%H:%M:%S) ----"
  $PY tools/run_token_merge_pca_experiment.py --strategy bsm_ksim_gradual_vec \
    --merge_layers "8" --r_per_layer "0.15,0.25" --bsm_match_metric key --merge_axis $AX \
    --repeats $REP --warmup 2 --out_dir "$BASE/AXIS_$AX" 2>&1 | tail -3
done
echo "==== ENC FRONTIER DONE $(date +%H:%M:%S) ===="
