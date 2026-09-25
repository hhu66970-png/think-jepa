#!/usr/bin/env bash
# Phase2 downstream: add PiToMe (4th method) to the SAME base dir as dense/方案A/K-BSM
# so ext30_temporal_probe / ext30_robust can paired-compare vs dense at same train20/test10 + seeds.
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
export HF_ENDPOINT=https://hf-mirror.com
export PYBIN=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1
export DATA_DIR=/root/autodl-tmp/thinkjepa-work/cache_ext30/part2
export CACHE_DIR=/root/autodl-tmp/thinkjepa-work/cache_ext30/part2
export EPOCHS="${EPOCHS:-50}" FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=/root/autodl-tmp/thinkjepa-work/downstream_ext30/manifests
TRAIN="$MAN/train20.txt"; TEST="$MAN/test10.txt"
BASE=outputs/downstream_ext30_20260531
mkdir -p "$BASE"
run() {  # cfg merge strat layers ratio seed
  local cfg=$1 merge=$2 strat=$3 layers=$4 ratio=$5 seed=$6
  local out="$BASE/${cfg}__s${seed}"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=$merge DENSE_JEPA_MERGE_STRATEGY=$strat \
    DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
    TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1
  local ade=$(grep "best_val_avg_dist" "$out/test_results.md" 2>/dev/null | grep -oE "[0-9.]+" | head -1)
  echo "[RES] $cfg seed=$seed ADE=$ade"
}
echo "==== PITOME DOWNSTREAM START $(date +%H:%M:%S) ===="
for seed in ${SEEDS:-42 43 44 45 46 47}; do
  echo "---- seed=$seed $(date +%H:%M:%S) ----"
  run PITOME_r015 1 bsm_pitome_gradual_vec 12,14,16,18,20 0.15 "$seed"   # PiToMe 对照 K-BSM r0.15
done
echo "==== PITOME DOWNSTREAM DONE $(date +%H:%M:%S) ===="
