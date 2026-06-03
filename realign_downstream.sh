#!/usr/bin/env bash
# Fully-aligned re-run: 5 methods x seeds 42-47, ONE batch, bjb1, current code,
# new dir (non-destructive). Eliminates cross-host(bjb2->bjb1)/cross-day confound.
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=$B/downstream_ext30/manifests; TRAIN="$MAN/train20.txt"; TEST="$MAN/test10.txt"
BASE=outputs/downstream_ext30_aligned_20260602; mkdir -p $BASE
run() { local cfg=$1 tm=$2 strat=$3 layers=$4 ratio=$5 seed=$6; local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  echo "[run $(date +%H:%M:%S)] $cfg s$seed"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=$tm DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=$layers \
    DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] rc=$?"
  echo "[RES] $cfg s$seed ADE=$(grep best_val_avg_dist "$out/test_results.md" 2>/dev/null|grep -oE "[0-9.]+"|head -1) $(date +%H:%M:%S)"
}
echo "==== REALIGN START $(date +%H:%M:%S) ===="
for seed in 42 43 44 45 46 47; do
  run dense       0 bsm_ksim_gradual_vec    12             0.0  $seed
  run A_l8_r025   1 local_2x2_same_time_vec 8              0.25 $seed
  run BSM_r015    1 bsm_ksim_gradual_vec    12,14,16,18,20 0.15 $seed
  run PITOME_r015 1 bsm_pitome_gradual_vec  12,14,16,18,20 0.15 $seed
  run WAM_r015    1 bsm_taware_gradual_vec  12,14,16,18,20 0.15 $seed
done
echo "==== REALIGN DONE $(date +%H:%M:%S) ===="
