#!/usr/bin/env bash
# Goal A temporal audit: frozen single-pass eval on EXISTING ckpts (no retrain).
# A1/A2: per-step displacement + velocity/accel curves for dense (own) vs K-BSM r0.15 (own).
# A3: dense-trained predictor on K-BSM r0.15 features (restore_dense=True -> same 8192 shape).
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
export HF_ENDPOINT=https://hf-mirror.com
export PYBIN=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1
export DATA_DIR=/root/autodl-tmp/thinkjepa-work/cache_ext30/part2
export CACHE_DIR=/root/autodl-tmp/thinkjepa-work/cache_ext30/part2
export FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=/root/autodl-tmp/thinkjepa-work/downstream_ext30/manifests
export TRAIN_MANIFEST="$MAN/train20.txt" TEST_MANIFEST="$MAN/test10.txt"
export FROZEN_PREDICTOR=1
CK=outputs/downstream_ext30_20260531
AB=outputs/phase1_audit_20260601
mkdir -p "$AB"
echo "==== AUDIT START $(date +%H:%M:%S) ===="

# A1/A2 dense (own predictor + dense features) = baseline temporal curves
SEED=42 DENSE_JEPA_TOKEN_MERGE=0 DENSE_JEPA_MERGE_STRATEGY=local_2x2_same_time_vec \
  PREDICTOR_CKPT=$CK/dense__s42/ckpt_best.pt OUT_DIR=$AB/dense_own \
  bash scripts/train.sh > $AB/dense_own.log 2>&1
echo "[A1/A2 dense_own] $(grep -o 'best_val_avg_dist.*' $AB/dense_own/test_results.md 2>/dev/null | head -1)"

# A1/A2 K-BSM r0.15 (own predictor + K-BSM features) = adapted merged temporal curves
SEED=42 DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_ksim_gradual_vec \
  DENSE_JEPA_MERGE_LAYERS=12,14,16,18,20 DENSE_JEPA_MERGE_RATIO=0.15 \
  PREDICTOR_CKPT=$CK/BSM_r015__s42/ckpt_best.pt OUT_DIR=$AB/kbsm_own \
  bash scripts/train.sh > $AB/kbsm_own.log 2>&1
echo "[A1/A2 kbsm_own] $(grep -o 'best_val_avg_dist.*' $AB/kbsm_own/test_results.md 2>/dev/null | head -1)"

# A3 cross: dense-trained predictor on K-BSM r0.15 features (isolates representation compatibility)
SEED=42 DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_ksim_gradual_vec \
  DENSE_JEPA_MERGE_LAYERS=12,14,16,18,20 DENSE_JEPA_MERGE_RATIO=0.15 \
  PREDICTOR_CKPT=$CK/dense__s42/ckpt_best.pt OUT_DIR=$AB/kbsm_feats_dense_pred \
  bash scripts/train.sh > $AB/kbsm_feats_dense_pred.log 2>&1
echo "[A3 cross] $(grep -o 'best_val_avg_dist.*' $AB/kbsm_feats_dense_pred/test_results.md 2>/dev/null | head -1)"
echo "==== AUDIT DONE $(date +%H:%M:%S) ===="
