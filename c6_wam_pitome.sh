#!/usr/bin/env bash
# C6 generalization (held-out drawer): add PiToMe + WAM (seeds 42-44) to match the
# 3 existing methods, into the SAME base dir for paired comparison vs dense.
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=$B/downstream_ext30/manifests; TRAIN="$MAN/heldout_train.txt"; TEST="$MAN/heldout_test.txt"
BASE=outputs/downstream_c6_heldout
run() { local cfg=$1 strat=$2 seed=$3; local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  echo "[run $(date +%H:%M:%S)] $cfg s$seed"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=12,14,16,18,20 \
    DENSE_JEPA_MERGE_RATIO=0.15 TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" OUT_DIR="$out" \
    bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] rc=$?"
  echo "[RES] $cfg s$seed ADE=$(grep best_val_avg_dist "$out/test_results.md" 2>/dev/null|grep -oE "[0-9.]+"|head -1) $(date +%H:%M:%S)"
}
echo "==== C6 PITOME+WAM START $(date +%H:%M:%S) ===="
for seed in 42 43 44; do
  run PITOME_r015 bsm_pitome_gradual_vec $seed
  run WAM_r015    bsm_taware_gradual_vec $seed
done
echo "==== C6 PITOME+WAM DONE $(date +%H:%M:%S) ===="
