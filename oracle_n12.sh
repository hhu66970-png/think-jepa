#!/usr/bin/env bash
# oracle 扩 n=12:WAMhand(手区先验)在两赢点加 seed 48-53,与 42-47 合成 n=12 坐实"手vs运动"趋势
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
export DENSE_JEPA_RELEVANCE_SOURCE=handjoint DENSE_JEPA_RELEVANCE_PATH=$B/oracle_handprior/handprior_avg.npz DENSE_JEPA_RELEVANCE_LAMBDA=1.0
BASE=outputs/frontier_retrain_E1; mkdir -p $BASE
run() { local cfg=$1 layers=$2 ratio=$3 seed=$4; local out="$BASE/${cfg}__s${seed}"
  [ -f "$out/test_results.md" ] && { echo "[skip] $cfg s$seed"; return; }
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_taware_gradual_vec DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  grep -q "falling back to MOTION" "$BASE/${cfg}__s${seed}.log" 2>/dev/null && echo "[!!] $cfg s$seed prior回退motion"
  [ -f "$out/test_results.md" ] && rm -f "$out/ckpt_latest.pt"
}
L9=12,13,14,15,16,17,18,19,20; G5=12,14,16,18,20
for s in 48 49 50 51 52 53; do run WAMhand_r25 $G5 0.25 $s; run WAMhand_L9_r25 $L9 0.25 $s; done
