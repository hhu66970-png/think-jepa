#!/usr/bin/env bash
set -uo pipefail

# Fresh matched-budget confirmation for the 3,637-token operating point.
# This wrapper never writes into historical result directories.

REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/thinkjepa-work}"
BASE="${BASE:-${REPO}/outputs/revision_20260718_3637_n12}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-phase2-wam-regime}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-c74a54b}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_LATEST="${KEEP_LATEST:-0}"

cd "${REPO}" || exit 1

branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
if [[ "${branch}" != "${EXPECTED_BRANCH}" ]]; then
  echo "[fatal] expected branch ${EXPECTED_BRANCH}, found ${branch}" >&2
  exit 2
fi
if [[ "${commit}" != "${EXPECTED_COMMIT}"* ]]; then
  echo "[fatal] expected commit ${EXPECTED_COMMIT}, found ${commit}" >&2
  exit 3
fi

PYBIN="${PYBIN:-${WORK_ROOT}/miniconda3/envs/thinkjepa-train/bin/python}"
DATA_DIR="${DATA_DIR:-${WORK_ROOT}/cache_ext30/part2}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${WORK_ROOT}/downstream_ext30/manifests/train20.txt}"
TEST_MANIFEST="${TEST_MANIFEST:-${WORK_ROOT}/downstream_ext30/manifests/test10.txt}"

for path in "${PYBIN}" "${TRAIN_MANIFEST}" "${TEST_MANIFEST}"; do
  if [[ ! -s "${path}" ]]; then
    echo "[fatal] required path missing or empty: ${path}" >&2
    exit 4
  fi
done

if comm -12 \
  <(sed '/^[[:space:]]*$/d' "${TRAIN_MANIFEST}" | sort -u) \
  <(sed '/^[[:space:]]*$/d' "${TEST_MANIFEST}" | sort -u) \
  | grep -q .; then
  echo "[fatal] train and evaluation manifests overlap" >&2
  exit 5
fi

mkdir -p "${BASE}/protocol"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "branch=${branch}"
  echo "commit=${commit}"
  echo "python=${PYBIN}"
  echo "data_dir=${DATA_DIR}"
  echo "train_manifest=${TRAIN_MANIFEST}"
  echo "test_manifest=${TEST_MANIFEST}"
  echo "train_sha256=$(sha256sum "${TRAIN_MANIFEST}" | awk '{print $1}')"
  echo "test_sha256=$(sha256sum "${TEST_MANIFEST}" | awk '{print $1}')"
  echo "layers=12,14,16,18,20"
  echo "ratio=0.15"
  echo "expected_tokens=3637"
  echo "seeds=42-53"
  echo "epochs=50"
  echo "past_t=32"
  echo "future_t=32"
  echo "train_batch_size=2"
  echo "test_batch_size=2"
  echo "wam_relevance_source=motion"
  echo "wam_lambda=1.0"
} > "${BASE}/protocol/run_manifest.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader > "${BASE}/protocol/gpu.txt" 2>/dev/null || true
git status --short > "${BASE}/protocol/git_status.txt"
cp "${TRAIN_MANIFEST}" "${BASE}/protocol/train_manifest.txt"
cp "${TEST_MANIFEST}" "${BASE}/protocol/test_manifest.txt"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYBIN GPU_LIST="${GPU_LIST:-0}" NPROC_PER_NODE=1
export DATA_DIR CACHE_DIR="${DATA_DIR}"
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 NUM_WORKERS="${NUM_WORKERS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_MANIFEST TEST_MANIFEST

failures=0
run_one() {
  local cfg="$1" strategy="$2" relevance="$3" seed="$4"
  local out="${BASE}/${cfg}__s${seed}"
  local log="${BASE}/${cfg}__s${seed}.log"

  if [[ -s "${out}/metrics.json" && -s "${out}/test_results.md" ]]; then
    echo "[skip] ${cfg} seed=${seed} already complete"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${out}" "${out}.incomplete.${stamp}"
  fi

  echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'SEED=%q strategy=%q relevance=%q OUT_DIR=%q bash scripts/train.sh\n' \
      "${seed}" "${strategy}" "${relevance}" "${out}"
    return 0
  fi

  if SEED="${seed}" \
      DENSE_JEPA_TOKEN_MERGE=1 \
      DENSE_JEPA_MERGE_STRATEGY="${strategy}" \
      DENSE_JEPA_MERGE_LAYERS=12,14,16,18,20 \
      DENSE_JEPA_MERGE_RATIO=0.15 \
      DENSE_JEPA_RESTORE_DENSE=1 \
      DENSE_JEPA_RELEVANCE_SOURCE="${relevance}" \
      DENSE_JEPA_RELEVANCE_LAMBDA=1.0 \
      OUT_DIR="${out}" \
      bash scripts/train.sh > "${log}" 2>&1; then
    if [[ ! -s "${out}/metrics.json" || ! -s "${out}/test_results.md" ]]; then
      echo "[error] ${cfg} seed=${seed} exited cleanly but outputs are incomplete" >&2
      failures=$((failures + 1))
      return 1
    fi
    if [[ "${KEEP_LATEST}" != "1" ]]; then
      rm -f "${out}/ckpt_latest.pt"
    fi
    echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
  else
    local rc=$?
    echo "[error] ${cfg} seed=${seed} rc=${rc}; see ${log}" >&2
    failures=$((failures + 1))
    return 1
  fi
}

for seed in $(seq 42 53); do
  run_one KBSM_r015 bsm_ksim_gradual_vec none "${seed}" || true
  run_one WAM_motion_r015 bsm_taware_gradual_vec motion "${seed}" || true
done

echo "failures=${failures}" > "${BASE}/protocol/completion.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${BASE}/protocol/completion.txt"
if (( failures > 0 )); then
  exit 10
fi

