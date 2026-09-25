#!/usr/bin/env bash
# Phase1 motivation figures C2/C4/C5 (single clip, no model-code integration needed).
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0
H=tools/run_token_merge_pca_experiment.py
NPZ=/root/autodl-tmp/thinkjepa-work/tiny-cache/part2/assemble_disassemble_furniture_bench_chair/1873_L8_nf32_res256_new16_s00of08.npz
CK=vjepa2/vitl.pt
OUT=outputs/phase1_motiv_20260601
mkdir -p "$OUT"
echo "==== VIZ START $(date +%H:%M:%S) ===="
echo "== C5 attn/rank-by-depth =="
$PY _phase1_staging/viz_attn_entropy.py --harness $H --ckpt $CK --npz $NPZ --out $OUT/c5_attn --plot 2>&1 | tail -6
echo "== C2 similarity (same-frame vs cross-frame, by depth) =="
$PY _phase1_staging/viz_similarity_heatmap.py --harness $H --ckpt $CK --npz $NPZ --out $OUT/c2_sim --plot 2>&1 | tail -6
echo "== C4 merge-tree overlay (which patches K-BSM merges) =="
$PY _phase1_staging/viz_merge_tree.py --harness $H --ckpt $CK --npz $NPZ --merge_layers "12,14,16,18,20" --r_per_layer 0.15 --out $OUT/c4_tree 2>&1 | tail -6
echo "==== VIZ DONE $(date +%H:%M:%S) ===="
