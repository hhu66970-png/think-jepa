#!/usr/bin/env bash
set -uo pipefail

# Train on 52 clips, select checkpoints on 13 validation clips, then evaluate
# each method's own checkpoint exactly once on 26 untouched official-test clips.

REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/thinkjepa-work}"
SUBSET="${SUBSET:-${WORK_ROOT}/egodex_balanced_52_13_26_seed202707}"
BASE="${BASE:-${REPO}/outputs/revision_20260718_clean_616_n6}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-phase2-wam-regime}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-c74a54b}"
DRY_RUN="${DRY_RUN:-0}"

cd "${REPO}" || exit 1
branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
[[ "${branch}" == "${EXPECTED_BRANCH}" ]] || { echo "[fatal] branch ${branch}" >&2; exit 2; }
[[ "${commit}" == "${EXPECTED_COMMIT}"* ]] || { echo "[fatal] commit ${commit}" >&2; exit 3; }

TRAIN_MANIFEST="${SUBSET}/manifests/train.txt"
VAL_MANIFEST="${SUBSET}/manifests/validation.txt"
TEST_MANIFEST_CLEAN="${SUBSET}/manifests/test.txt"
DATA_DIR="${SUBSET}/cache/part2"
PYBIN="${PYBIN:-${WORK_ROOT}/miniconda3/envs/thinkjepa-train/bin/python}"
for path in "${PYBIN}" "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${TEST_MANIFEST_CLEAN}"; do
  [[ -s "${path}" ]] || { echo "[fatal] required path missing: ${path}" >&2; exit 4; }
done

for pair in "${TRAIN_MANIFEST}:${VAL_MANIFEST}" "${TRAIN_MANIFEST}:${TEST_MANIFEST_CLEAN}" "${VAL_MANIFEST}:${TEST_MANIFEST_CLEAN}"; do
  left="${pair%%:*}"; right="${pair##*:}"
  if comm -12 <(sort -u "${left}") <(sort -u "${right}") | grep -q .; then
    echo "[fatal] manifest overlap: ${left} ${right}" >&2
    exit 5
  fi
done

mkdir -p "${BASE}/protocol" "${BASE}/train" "${BASE}/test"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=${branch}"
  echo "commit=${commit}"
  echo "train_count=$(wc -l < "${TRAIN_MANIFEST}")"
  echo "validation_count=$(wc -l < "${VAL_MANIFEST}")"
  echo "test_count=$(wc -l < "${TEST_MANIFEST_CLEAN}")"
  echo "train_sha256=$(sha256sum "${TRAIN_MANIFEST}" | awk '{print $1}')"
  echo "validation_sha256=$(sha256sum "${VAL_MANIFEST}" | awk '{print $1}')"
  echo "test_sha256=$(sha256sum "${TEST_MANIFEST_CLEAN}" | awk '{print $1}')"
  echo "layers=12,13,14,15,16,17,18,19,20"
  echo "ratio=0.25"
  echo "expected_tokens=616"
  echo "seeds=42-47"
} > "${BASE}/protocol/run_manifest.txt"
cp "${SUBSET}/subset_metadata.json" "${BASE}/protocol/subset_metadata.json"
cp "${TRAIN_MANIFEST}" "${BASE}/protocol/train_manifest.txt"
cp "${VAL_MANIFEST}" "${BASE}/protocol/validation_manifest.txt"
cp "${TEST_MANIFEST_CLEAN}" "${BASE}/protocol/test_manifest.txt"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYBIN GPU_LIST="${GPU_LIST:-0}" NPROC_PER_NODE=1
export DATA_DIR CACHE_DIR="${DATA_DIR}"
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=1 NUM_WORKERS="${NUM_WORKERS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_MANIFEST

failures=0
train_one() {
  local cfg="$1" strategy="$2" relevance="$3" seed="$4"
  local out="${BASE}/train/${cfg}__s${seed}"
  local log="${BASE}/train/${cfg}__s${seed}.log"
  if [[ -s "${out}/metrics.json" && -s "${out}/ckpt_best.pt" ]]; then
    echo "[skip train] ${cfg} seed=${seed}"
    return 0
  fi
  [[ ! -e "${out}" ]] || mv "${out}" "${out}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[train] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
  [[ "${DRY_RUN}" != "1" ]] || return 0
  if SEED="${seed}" TEST_MANIFEST="${VAL_MANIFEST}" \
      DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY="${strategy}" \
      DENSE_JEPA_MERGE_LAYERS=12,13,14,15,16,17,18,19,20 \
      DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 \
      DENSE_JEPA_RELEVANCE_SOURCE="${relevance}" DENSE_JEPA_RELEVANCE_LAMBDA=1.0 \
      OUT_DIR="${out}" bash scripts/train.sh > "${log}" 2>&1; then
    [[ -s "${out}/ckpt_best.pt" ]] || { failures=$((failures + 1)); return 1; }
    rm -f "${out}/ckpt_latest.pt"
  else
    failures=$((failures + 1)); return 1
  fi
}

test_one() {
  local cfg="$1" strategy="$2" relevance="$3" seed="$4"
  local ckpt="${BASE}/train/${cfg}__s${seed}/ckpt_best.pt"
  local out="${BASE}/test/${cfg}__s${seed}"
  local log="${BASE}/test/${cfg}__s${seed}.log"
  [[ -s "${out}/metrics.json" ]] && { echo "[skip test] ${cfg} seed=${seed}"; return 0; }
  [[ -s "${ckpt}" ]] || { echo "[error] missing checkpoint ${ckpt}" >&2; failures=$((failures + 1)); return 1; }
  [[ ! -e "${out}" ]] || mv "${out}" "${out}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[test] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
  [[ "${DRY_RUN}" != "1" ]] || return 0
  if SEED="${seed}" TEST_MANIFEST="${TEST_MANIFEST_CLEAN}" FROZEN_PREDICTOR=1 \
      PREDICTOR_CKPT="${ckpt}" DENSE_JEPA_TOKEN_MERGE=1 \
      DENSE_JEPA_MERGE_STRATEGY="${strategy}" \
      DENSE_JEPA_MERGE_LAYERS=12,13,14,15,16,17,18,19,20 \
      DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 \
      DENSE_JEPA_RELEVANCE_SOURCE="${relevance}" DENSE_JEPA_RELEVANCE_LAMBDA=1.0 \
      OUT_DIR="${out}" bash scripts/train.sh > "${log}" 2>&1; then
    [[ -s "${out}/metrics.json" ]] || { failures=$((failures + 1)); return 1; }
    rm -f "${out}/ckpt_latest.pt" "${out}/ckpt_best.pt"
  else
    failures=$((failures + 1)); return 1
  fi
}

for seed in $(seq 42 47); do
  train_one KBSM_L9_r25 bsm_ksim_gradual_vec none "${seed}" || true
  test_one KBSM_L9_r25 bsm_ksim_gradual_vec none "${seed}" || true
  train_one WAM_motion_L9_r25 bsm_taware_gradual_vec motion "${seed}" || true
  test_one WAM_motion_L9_r25 bsm_taware_gradual_vec motion "${seed}" || true
done

echo "failures=${failures}" > "${BASE}/protocol/completion.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${BASE}/protocol/completion.txt"
(( failures == 0 ))

