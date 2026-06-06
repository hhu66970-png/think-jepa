#!/usr/bin/env bash
# λ 门控扫描 @616(L9_r25), n=6:λ∈{0.25,0.5,0.75}。端点复用 λ=0≡KBSM_L9_r25 / λ=1=WAM_L9_r25(已跑)。
# 证 "是门控(任务感知)起作用、非仅 K-BSM" → ADE 应随 λ 单调改善。
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
BASE=outputs/frontier_retrain_E1; mkdir -p $BASE
L9=12,13,14,15,16,17,18,19,20
run() { local cfg=$1 lam=$2 seed=$3; local out="$BASE/${cfg}__s${seed}"
  [ -f "$out/test_results.md" ] && { echo "[skip] $cfg s$seed"; return; }
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_taware_gradual_vec DENSE_JEPA_MERGE_LAYERS=$L9 DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 DENSE_JEPA_RELEVANCE_SOURCE=motion DENSE_JEPA_RELEVANCE_LAMBDA=$lam OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  [ -f "$out/test_results.md" ] && rm -f "$out/ckpt_latest.pt"
}
for s in 48 49 50 51 52 53; do
  run WAMlam025_L9_r25 0.25 $s
  run WAMlam05_L9_r25  0.5  $s
  run WAMlam075_L9_r25 0.75 $s
done
