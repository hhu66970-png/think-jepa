#!/usr/bin/env bash
# C6 generalization: task-held-out split. 3 ready methods (dense / 方案A / K-BSM) x seeds 42,43,44.
# Serial, GPU-polling (wait until memory.used < 5000 MiB), OOM backoff.
# PiToMe (4th method) intentionally NOT run here (waiting on other agent fix).
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
TRAIN="$MAN/heldout_train.txt"; TEST="$MAN/heldout_test.txt"
BASE=outputs/downstream_c6_heldout
mkdir -p "$BASE"

GPU_FREE_MIB=5000
wait_for_gpu() {
  # block until GPU memory.used < GPU_FREE_MIB; report waits
  local waited=0
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
    if [ -z "$used" ]; then used=999999; fi
    if [ "$used" -lt "$GPU_FREE_MIB" ]; then
      echo "[GPU] free (used=${used} MiB < ${GPU_FREE_MIB}); proceeding $(date +%H:%M:%S)"
      return 0
    fi
    echo "[GPU] busy (used=${used} MiB >= ${GPU_FREE_MIB}); waiting... $(date +%H:%M:%S)"
    waited=$((waited+1))
    sleep 60
  done
}

run() {  # cfg merge strat layers ratio seed
  local cfg=$1 merge=$2 strat=$3 layers=$4 ratio=$5 seed=$6
  local out="$BASE/${cfg}__s${seed}"
  # idempotent: skip if already completed
  if [ -f "$out/test_results.md" ] && grep -q "best_val_avg_dist" "$out/test_results.md"; then
    local ade0=$(grep "best_val_avg_dist" "$out/test_results.md" | grep -oE "[0-9.]+" | head -1)
    echo "[SKIP] $cfg seed=$seed already done ADE=$ade0"
    return 0
  fi
  local attempt=1
  while [ $attempt -le 3 ]; do
    wait_for_gpu
    echo "==== RUN $cfg seed=$seed attempt=$attempt $(date +%H:%M:%S) ===="
    SEED=$seed DENSE_JEPA_TOKEN_MERGE=$merge DENSE_JEPA_MERGE_STRATEGY=$strat \
      DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
      TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
      OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$out/test_results.md" ]; then
      local ade=$(grep "best_val_avg_dist" "$out/test_results.md" | grep -oE "[0-9.]+" | head -1)
      echo "[RES] $cfg seed=$seed ADE=$ade rc=$rc $(date +%H:%M:%S)"
      return 0
    fi
    if grep -qiE "out of memory|CUDA error|OutOfMemory" "$BASE/${cfg}__s${seed}.log"; then
      echo "[OOM] $cfg seed=$seed attempt=$attempt rc=$rc; backing off 120s then retry"
      sleep 120
      attempt=$((attempt+1))
      continue
    fi
    echo "[FAIL] $cfg seed=$seed rc=$rc (non-OOM); see log. Not retrying."
    return 1
  done
  echo "[GIVEUP] $cfg seed=$seed after 3 OOM attempts"
  return 1
}

echo "==== C6 HELDOUT START $(date +%H:%M:%S) ===="
for seed in ${SEEDS:-42 43 44}; do
  echo "---- seed=$seed $(date +%H:%M:%S) ----"
  run dense        0 local_2x2_same_time_vec 8             0.125 "$seed"
  run A_l8_r025    1 local_2x2_same_time_vec 8             0.25  "$seed"
  run BSM_r015     1 bsm_ksim_gradual_vec    12,14,16,18,20 0.15  "$seed"
done
echo "==== C6 HELDOUT DONE $(date +%H:%M:%S) ===="
