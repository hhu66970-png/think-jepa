#!/usr/bin/env bash
# Oracle-WAM:在 E1 的两个赢点(r0.25=1944, L9_r25=616)用 handjoint 平均手区先验
# 跑 WAM,与 E1 已有的 V1-motion WAM / K-BSM 同 seed 配对对比。
# 需先应用 relevance 透传 patch(--dense_jepa_relevance_*)。
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt
export TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
# === V2 handjoint 先验(平均手区,[h*w]=256,接线 tile 到 t)===
export DENSE_JEPA_RELEVANCE_SOURCE=handjoint
export DENSE_JEPA_RELEVANCE_PATH=$B/oracle_handprior/handprior_avg.npz
export DENSE_JEPA_RELEVANCE_LAMBDA=1.0
BASE=outputs/frontier_retrain_E1; mkdir -p $BASE
run() { local cfg=$1 layers=$2 ratio=$3 seed=$4; local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  echo "[run $(date +%H:%M:%S)] $cfg s$seed (handjoint prior)"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_taware_gradual_vec \
    DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  # 验证真用了 prior(grep 日志:不该出现回退告警)
  if grep -q "falling back to MOTION" "$BASE/${cfg}__s${seed}.log" 2>/dev/null; then
    echo "  [!!] $cfg s$seed: prior 回退到 motion 了!检查 npz"; fi
  rm -f "$out/ckpt_latest.pt"
}
L9=12,13,14,15,16,17,18,19,20
G5=12,14,16,18,20
echo "==== ORACLE-WAM START $(date +%H:%M:%S) ===="
for s in 42 43 44 45 46 47; do
  run WAMhand_r25    $G5 0.25 $s
  run WAMhand_L9_r25 $L9 0.25 $s
done
echo "==== ORACLE-WAM DONE $(date +%H:%M:%S) ===="
