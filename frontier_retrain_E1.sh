#!/usr/bin/env bash
# E1: 高压缩 frontier 全量重训(WAM vs K-BSM)。原样复制 realign_downstream.sh 的
# 调用,仅改 MERGE_LAYERS/RATIO(控制变量)。dense 基线复用 aligned_20260602。
# 顺序:最激进 L12_r25 → L9_r25 → 中等 r25/r20。每点 WAM/K-BSM 按 seed 配对。
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=$B/downstream_ext30/manifests; TRAIN="$MAN/train20.txt"; TEST="$MAN/test10.txt"
BASE=outputs/frontier_retrain_E1; mkdir -p $BASE
run() { local cfg=$1 strat=$2 layers=$3 ratio=$4 seed=$5; local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  echo "[run $(date +%H:%M:%S)] $cfg s$seed strat=$strat layers=$layers r=$ratio"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=$layers \
    DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  echo "[RES] $cfg s$seed ADE=$(grep best_val_avg_dist "$out/test_results.md" 2>/dev/null|grep -oE '[0-9.]+'|head -1) $(date +%H:%M:%S)"
}
L9=12,13,14,15,16,17,18,19,20
L12=10,11,12,13,14,15,16,17,18,19,20,21
G5=12,14,16,18,20
echo "==== E1 START $(date +%H:%M:%S) ===="
# ---- 最激进 L12_r25(~261 token,gap 应最大,先出决策门答案)----
for s in 42 43 44 45 46 47; do
  run KBSM_L12_r25 bsm_ksim_gradual_vec   $L12 0.25 $s
  run WAM_L12_r25  bsm_taware_gradual_vec $L12 0.25 $s
done
# ---- L9_r25(~616 token)----
for s in 42 43 44 45 46 47; do
  run KBSM_L9_r25  bsm_ksim_gradual_vec   $L9 0.25 $s
  run WAM_L9_r25   bsm_taware_gradual_vec $L9 0.25 $s
done
# ---- 中等 r0.25 / r0.20 @ L12-20(若时间允许)----
for s in 42 43 44 45 46 47; do
  run KBSM_r25 bsm_ksim_gradual_vec   $G5 0.25 $s
  run WAM_r25  bsm_taware_gradual_vec $G5 0.25 $s
done
for s in 42 43 44 45 46 47; do
  run KBSM_r20 bsm_ksim_gradual_vec   $G5 0.20 $s
  run WAM_r20  bsm_taware_gradual_vec $G5 0.20 $s
done
echo "==== E1 DONE $(date +%H:%M:%S) ===="
