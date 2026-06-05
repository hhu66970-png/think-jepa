#!/usr/bin/env bash
# E1-extreme: 把压缩推到 261(L12_r25)以下,找"崩溃点"——ADE/FDE/运动学何时才 > dense——
# 并测 WAM 是否比 K-BSM 晚崩溃 => 直接检验"WAM 能否进一步压缩加速"。
# 完全沿用 frontier_retrain_E1_n12.sh 的 run()/env(控制变量),写入同 BASE。
#
# 关键:token_merge.py(L524/L990)把 per-layer merge_ratio 硬 cap 在 0.25
# (min(merge_ratio,0.25)),所以"加大 r"无效(会被钳回 0.25=261)。
# 因此沿用 frontier 既有的压缩杠杆——"加合并层"(G5→L9→L12 就是这么压下来的):
#   L14 = 8..21 (14层) ~ 8192*0.75^14 ≈ 146 token
#   L16 = 6..21 (16层) ~ 8192*0.75^16 ≈ 82  token
# (参照:L12=10..21=261,已验证;r 全程 0.25,在 cap 内,无需改 merge 码)
# 诚实边界:加层=从更早层开始合并,"更少token"与"更早合并"会耦合(F5 U形);如需隔离
# 该混淆,再单独做"抬 cap 在 L12 上加大 r"的对照(改码+审查),本脚本先走零改码路径。
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=$B/downstream_ext30/manifests; TRAIN="$MAN/train20.txt"; TEST="$MAN/test10.txt"
BASE=outputs/frontier_retrain_E1; mkdir -p $BASE
SMOKE="${SMOKE:-0}"   # SMOKE=1 => 只跑 1 个 config 1 epoch,验证不崩 + 看真实 token 数

run() { local cfg=$1 strat=$2 layers=$3 ratio=$4 seed=$5; local out="$BASE/${cfg}__s${seed}"
  if [ -f "$out/test_results.md" ]; then echo "[skip] $cfg s$seed"; return; fi
  echo "[run $(date +%H:%M:%S)] $cfg s$seed strat=$strat layers=$layers r=$ratio"
  SEED=$seed DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=$layers \
    DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 TRAIN_MANIFEST="$TRAIN" TEST_MANIFEST="$TEST" \
    OUT_DIR="$out" bash scripts/train.sh > "$BASE/${cfg}__s${seed}.log" 2>&1 || echo "[warn] $cfg s$seed rc=$?"
  echo "[RES] $cfg s$seed ADE=$(grep best_val_avg_dist "$out/test_results.md" 2>/dev/null|grep -oE '[0-9.]+'|head -1) $(date +%H:%M:%S)"
  [ -f "$out/test_results.md" ] && rm -f "$out/ckpt_latest.pt"   # 省盘:保留 best 供 E2/E4
}

L14=8,9,10,11,12,13,14,15,16,17,18,19,20,21
L16=6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21

if [ "$SMOKE" = "1" ]; then
  echo "==== E1-EXTREME SMOKE (1 cfg, EPOCHS=1) $(date +%H:%M:%S) ===="
  EPOCHS=1 run SMOKE_WAM_L16_r25 bsm_taware_gradual_vec $L16 0.25 42
  echo "  实际 token(num_tokens_after,应 ~82):"; grep -oE "num_tokens_after[^,}]*" "$BASE/SMOKE_WAM_L16_r25__s42.log" 2>/dev/null | tail -1
  echo "  错误自检(应空):"; grep -iE "error|assert|nan|traceback" "$BASE/SMOKE_WAM_L16_r25__s42.log" 2>/dev/null | head -3
  echo "==== SMOKE DONE $(date +%H:%M:%S) ===="
  exit 0
fi

echo "==== E1-EXTREME START $(date +%H:%M:%S) ===="
# L14 ~146 token
for s in 42 43 44 45 46 47; do
  run KBSM_L14_r25 bsm_ksim_gradual_vec   $L14 0.25 $s
  run WAM_L14_r25  bsm_taware_gradual_vec $L14 0.25 $s
done
# L16 ~82 token
for s in 42 43 44 45 46 47; do
  run KBSM_L16_r25 bsm_ksim_gradual_vec   $L16 0.25 $s
  run WAM_L16_r25  bsm_taware_gradual_vec $L16 0.25 $s
done
echo "==== E1-EXTREME DONE $(date +%H:%M:%S) ===="
