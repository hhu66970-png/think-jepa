#!/usr/bin/env bash
# WAM (5th method) full downstream training, SAME base dir / manifests / seeds as
# dense/方案A/K-BSM/PiToMe so it joins the n=6 paired comparison (02 table C).
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
run() { local cfg=$1 strat=$2 layers=$3 ratio=$4 seed=$5
  local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat \
    DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
    TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] rc=$?"
  local ade=$(grep "best_val_avg_dist" "$out/test_results.md" 2>/dev/null | grep -oE "[0-9.]+" | head -1)
  echo "[RES] $cfg s$seed ADE=$ade $(date +%H:%M:%S)"
}
echo "==== WAM DOWNSTREAM START $(date +%H:%M:%S) ===="
for seed in ${SEEDS:-42 43 44 45 46 47}; do
  run WAM_r015 bsm_taware_gradual_vec 12,14,16,18,20 0.15 "$seed"
done
echo "==== WAM DOWNSTREAM DONE $(date +%H:%M:%S) ===="
