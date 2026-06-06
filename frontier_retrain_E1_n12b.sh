#!/usr/bin/env bash
# E1 强化主结果:把 r20(2686) 与 L12_r25(261) 两个 n=6 点扩 seed 48-53 → n=12。
# 配置与 frontier_retrain_E1_n12.sh 完全一致(控制变量);WAM 用 bsm_taware 默认(motion λ=1),与 s42-47 同。
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
  echo "[run $(date +%H:%M:%S)] $cfg s$seed"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=$layers \
    DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  [ -f "$out/test_results.md" ] && rm -f "$out/ckpt_latest.pt"
}
G5=12,14,16,18,20
L12=10,11,12,13,14,15,16,17,18,19,20,21
echo "==== E1 n12b START $(date +%H:%M:%S) ===="
for s in 48 49 50 51 52 53; do
  run KBSM_r20     bsm_ksim_gradual_vec   $G5  0.20 $s
  run WAM_r20      bsm_taware_gradual_vec $G5  0.20 $s
  run KBSM_L12_r25 bsm_ksim_gradual_vec   $L12 0.25 $s
  run WAM_L12_r25  bsm_taware_gradual_vec $L12 0.25 $s
done
echo "==== E1 n12b DONE $(date +%H:%M:%S) ===="
