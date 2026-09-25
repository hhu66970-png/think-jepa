#!/usr/bin/env bash
set -euo pipefail

# Incremental matched spatial-axis baseline on the clean 52/13/26 protocol.
# This complements the independently trained temporal-axis control so their
# only intended difference is merge_axis=spatial versus merge_axis=temporal.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/thinkjepa-work}"
SUBSET="${SUBSET:-${WORK_ROOT}/egodex_balanced_52_13_26_seed202707}"
REFERENCE="${REFERENCE:-${REPO}/outputs/revision_20260718_clean_616_n6}"
TEMPORAL="${TEMPORAL:-${REPO}/outputs/revision_20260718_clean_temporal_616_n6}"
BASE="${BASE:-${REPO}/outputs/revision_20260719_clean_spatial_616_n6}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-phase2-wam-regime}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-c74a54b}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "${BASE}")"
exec 9>"${BASE}.lock"
flock -n 9 || { echo "[fatal] another runner holds ${BASE}.lock" >&2; exit 6; }

cd "${REPO}" || exit 1
branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
[[ "${branch}" == "${EXPECTED_BRANCH}" ]] || { echo "[fatal] branch ${branch}" >&2; exit 2; }
[[ "${commit}" == "${EXPECTED_COMMIT}"* ]] || { echo "[fatal] commit ${commit}" >&2; exit 3; }
grep -q 'dense_jepa_merge_axis' cache_train/thinker_train.py || {
  echo "[fatal] merge-axis CLI patch is not installed" >&2; exit 7;
}
grep -q 'DENSE_JEPA_MERGE_AXIS' scripts/train.sh || {
  echo "[fatal] merge-axis train.sh patch is not installed" >&2; exit 8;
}

TRAIN_MANIFEST="${SUBSET}/manifests/train.txt"
VAL_MANIFEST="${SUBSET}/manifests/validation.txt"
TEST_MANIFEST_CLEAN="${SUBSET}/manifests/test.txt"
DATA_DIR="${SUBSET}/cache/part2"
PYBIN="${PYBIN:-${WORK_ROOT}/miniconda3/envs/thinkjepa-train/bin/python}"
for path in "${PYBIN}" "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${TEST_MANIFEST_CLEAN}"; do
  [[ -s "${path}" ]] || { echo "[fatal] required path missing: ${path}" >&2; exit 4; }
done
grep -qx 'failures=0' "${REFERENCE}/protocol/completion.txt" || {
  echo "[fatal] reference clean experiment is incomplete" >&2; exit 9;
}
grep -qx 'failures=0' "${TEMPORAL}/protocol/completion.txt" || {
  echo "[fatal] temporal-axis experiment is incomplete" >&2; exit 10;
}

mkdir -p "${BASE}/protocol" "${BASE}/train" "${BASE}/test"
protocol_fingerprint="$({
  printf '%s\n' \
    "train_manifest=$(sha256sum "${TRAIN_MANIFEST}" | awk '{print $1}')" \
    "validation_manifest=$(sha256sum "${VAL_MANIFEST}" | awk '{print $1}')" \
    "test_manifest=$(sha256sum "${TEST_MANIFEST_CLEAN}" | awk '{print $1}')" \
    "method=KBSM_spatial_L9_r25" \
    "axis=spatial" \
    "layers=12,13,14,15,16,17,18,19,20" \
    "ratio=0.25" \
    "seeds=42-47"
  sha256sum cache_train/thinker_train.py scripts/train.sh \
    vjepa2/src/models/utils/token_merge.py \
    vjepa2/src/models/utils/token_merge_diagnostics.py
} | sha256sum | awk '{print $1}')"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=${branch}"
  echo "commit=${commit}"
  echo "method=KBSM_spatial_L9_r25"
  echo "merge_axis=spatial"
  echo "layers=12,13,14,15,16,17,18,19,20"
  echo "ratio=0.25"
  echo "expected_tokens=616"
  echo "seeds=42-47"
  echo "protocol_fingerprint=${protocol_fingerprint}"
  echo "train_sha256=$(sha256sum "${TRAIN_MANIFEST}" | awk '{print $1}')"
  echo "validation_sha256=$(sha256sum "${VAL_MANIFEST}" | awk '{print $1}')"
  echo "test_sha256=$(sha256sum "${TEST_MANIFEST_CLEAN}" | awk '{print $1}')"
} > "${BASE}/protocol/run_manifest.txt"
cp "${SUBSET}/subset_metadata.json" "${BASE}/protocol/subset_metadata.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYBIN GPU_LIST="${GPU_LIST:-0}" NPROC_PER_NODE=1
export DATA_DIR CACHE_DIR="${DATA_DIR}"
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=1 NUM_WORKERS="${NUM_WORKERS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_MANIFEST

failures=0
cfg="KBSM_spatial_L9_r25"
for seed in $(seq 42 47); do
  train_out="${BASE}/train/${cfg}__s${seed}"
  train_log="${BASE}/train/${cfg}__s${seed}.log"
  if [[ ! -s "${train_out}/metrics.json" || ! -s "${train_out}/ckpt_best.pt" || \
        ! -s "${train_out}/provenance.txt" || \
        "$(cat "${train_out}/provenance.txt" 2>/dev/null)" != "protocol=${protocol_fingerprint}" ]]; then
    auto_resume=0
    if [[ -s "${train_out}/ckpt_latest.pt" ]]; then
      auto_resume=1
    elif [[ -e "${train_out}" ]]; then
      mv "${train_out}" "${train_out}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    echo "[train] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
    if [[ "${DRY_RUN}" != "1" ]] && ! SEED="${seed}" TEST_MANIFEST="${VAL_MANIFEST}" \
        DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_ksim_gradual_vec \
        DENSE_JEPA_MERGE_AXIS=spatial \
        DENSE_JEPA_MERGE_LAYERS=12,13,14,15,16,17,18,19,20 \
        DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 \
        DENSE_JEPA_RELEVANCE_SOURCE=none AUTO_RESUME="${auto_resume}" \
        OUT_DIR="${train_out}" bash scripts/train.sh > "${train_log}" 2>&1; then
      failures=$((failures + 1)); continue
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
      rm -f "${train_out}/ckpt_latest.pt"
      printf '%s\n' "protocol=${protocol_fingerprint}" > "${train_out}/provenance.txt"
    fi
  fi

  test_out="${BASE}/test/${cfg}__s${seed}"
  test_log="${BASE}/test/${cfg}__s${seed}.log"
  ckpt="${train_out}/ckpt_best.pt"
  checkpoint_sha="dry-run"
  [[ "${DRY_RUN}" == "1" ]] || checkpoint_sha="$(sha256sum "${ckpt}" | awk '{print $1}')"
  expected_stamp="protocol=${protocol_fingerprint};checkpoint=${checkpoint_sha}"
  if [[ ! -s "${test_out}/metrics.json" || ! -s "${test_out}/test_results.md" || \
        ! -s "${test_out}/temporal_metrics.json" || ! -s "${test_out}/per_sample_metrics.json" || \
        ! -s "${test_out}/provenance.txt" || \
        "$(cat "${test_out}/provenance.txt" 2>/dev/null)" != "${expected_stamp}" ]]; then
    [[ "${DRY_RUN}" == "1" || -s "${ckpt}" ]] || { failures=$((failures + 1)); continue; }
    [[ ! -e "${test_out}" ]] || mv "${test_out}" "${test_out}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[test] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
    if [[ "${DRY_RUN}" != "1" ]] && ! SEED="${seed}" TEST_MANIFEST="${TEST_MANIFEST_CLEAN}" \
        FROZEN_PREDICTOR=1 PREDICTOR_CKPT="${ckpt}" \
        DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_ksim_gradual_vec \
        DENSE_JEPA_MERGE_AXIS=spatial \
        DENSE_JEPA_MERGE_LAYERS=12,13,14,15,16,17,18,19,20 \
        DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 \
        DENSE_JEPA_RELEVANCE_SOURCE=none OUT_DIR="${test_out}" \
        bash scripts/train.sh > "${test_log}" 2>&1; then
      failures=$((failures + 1)); continue
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
      rm -f "${test_out}/ckpt_latest.pt" "${test_out}/ckpt_best.pt"
      printf '%s\n' "${expected_stamp}" > "${test_out}/provenance.txt"
    fi
  fi
done

if (( failures != 0 )); then
  printf 'failures=%s\nfinished_utc=%s\n' "${failures}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BASE}/protocol/completion.txt"
  exit 1
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${BASE}/protocol/dry_run.txt"
  exit 0
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYBIN}" "${SCRIPT_DIR}/aggregate_paired.py" "${BASE}" \
    --left-root "${REFERENCE}/test" --right-root "${BASE}/test" \
    --left KBSM_L9_r25 --right KBSM_spatial_L9_r25 --seeds 42-47 --mode final
  "${PYBIN}" "${SCRIPT_DIR}/aggregate_paired.py" "${BASE}" \
    --left-root "${BASE}/test" --right-root "${TEMPORAL}/test" \
    --left KBSM_spatial_L9_r25 --right KBSM_temporal_L9_r25 --seeds 42-47 --mode final
fi
printf 'failures=0\nfinished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${BASE}/protocol/completion.txt"
