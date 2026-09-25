#!/usr/bin/env bash
set -euo pipefail

# Reuse checkpoints selected on the 13-clip validation set and evaluate them
# once on the complete official 200-clip test split.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/thinkjepa-work}"
SUBSET="${SUBSET:-${WORK_ROOT}/egodex_balanced_52_13_26_seed202707}"
TEST200="${TEST200:-${WORK_ROOT}/egodex_official_test200}"
REFERENCE="${REFERENCE:-${REPO}/outputs/revision_20260718_clean_616_n6}"
BASE="${BASE:-${REPO}/outputs/revision_20260718_frozen_test200_616_n6}"
PYBIN="${PYBIN:-${WORK_ROOT}/miniconda3/envs/thinkjepa-train/bin/python}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-phase2-wam-regime}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-c74a54b}"
EXPECTED_THINKER_SHA="${EXPECTED_THINKER_SHA:-4d2ffa846cd5c34eda6323c33402d5b17683a4e3dc0872fbd015ca23edd619d0}"
EXPECTED_TRAIN_SHA="${EXPECTED_TRAIN_SHA:-bd0371f82811a88aed20fbcf26ca156cebe143497dc7f0077b8f6ccfa37c48a0}"
TEST_MANIFEST_CLEAN="${TEST200}/manifests/test200.txt"
TRAIN_MANIFEST="${SUBSET}/manifests/train.txt"
DATA_DIR="${SUBSET}/cache/part2"

mkdir -p "$(dirname "${BASE}")"
exec 9>"${BASE}.lock"
flock -n 9 || { echo "[fatal] another runner holds ${BASE}.lock" >&2; exit 6; }

cd "${REPO}" || exit 1
branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
thinker_sha="$(sha256sum cache_train/thinker_train.py | awk '{print $1}')"
train_sha="$(sha256sum scripts/train.sh | awk '{print $1}')"
[[ "${branch}" == "${EXPECTED_BRANCH}" ]] || {
  echo "[fatal] branch ${branch}, expected ${EXPECTED_BRANCH}" >&2; exit 2;
}
[[ "${commit}" == "${EXPECTED_COMMIT}"* ]] || {
  echo "[fatal] commit ${commit}, expected prefix ${EXPECTED_COMMIT}" >&2; exit 3;
}
[[ "${thinker_sha}" == "${EXPECTED_THINKER_SHA}" ]] || {
  echo "[fatal] thinker_train.py hash ${thinker_sha}" >&2; exit 7;
}
[[ "${train_sha}" == "${EXPECTED_TRAIN_SHA}" ]] || {
  echo "[fatal] train.sh hash ${train_sha}" >&2; exit 8;
}
for path in "${PYBIN}" "${TRAIN_MANIFEST}" "${TEST_MANIFEST_CLEAN}" \
            "${TEST200}/test200_metadata.json"; do
  [[ -s "${path}" ]] || { echo "[fatal] required path missing: ${path}" >&2; exit 4; }
done
"${PYBIN}" - "${TEST200}/test200_metadata.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
assert meta.get("official_test_count") == 200
assert meta.get("required_guidance") == "both"
assert set(meta.get("required_npz_keys", [])) >= {"imgs", "vjepa_feats", "vlm_old", "vlm_new"}
assert all(int(meta.get("guidance_feature_dims", {}).get(k, 0)) > 0 for k in ("vlm_old", "vlm_new"))
assert len(str(meta.get("resolved_revision", ""))) >= 7
files = meta.get("files", [])
assert len(files) == 200
assert all(len(str(row.get("sha256", ""))) == 64 for row in files)
PY
grep -qx 'failures=0' "${REFERENCE}/protocol/completion.txt" || {
  echo "[fatal] reference clean experiment is incomplete" >&2; exit 9;
}
[[ "$(wc -l < "${TEST_MANIFEST_CLEAN}")" -eq 200 ]] || {
  echo "[fatal] test manifest is not 200 clips" >&2; exit 10;
}

mkdir -p "${BASE}/protocol" "${BASE}/test"
protocol_fingerprint="$({
  printf '%s\n' \
    "test_manifest=$(sha256sum "${TEST_MANIFEST_CLEAN}" | awk '{print $1}')" \
    "methods=KBSM_L9_r25,WAM_motion_L9_r25" \
    "layers=12,13,14,15,16,17,18,19,20" \
    "ratio=0.25" \
    "seeds=42-47"
  sha256sum "${REPO}/cache_train/thinker_train.py" "${REPO}/scripts/train.sh" \
    "${REPO}/vjepa2/src/models/utils/token_merge.py" \
    "${REPO}/vjepa2/src/models/utils/token_merge_diagnostics.py"
} | sha256sum | awk '{print $1}')"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=${branch}"
  echo "commit=${commit}"
  echo "thinker_train_sha256=${thinker_sha}"
  echo "train_sh_sha256=${train_sha}"
  echo "checkpoint_source=${REFERENCE}/train"
  echo "checkpoint_selection=validation13"
  echo "test_count=200"
  echo "test_manifest_sha256=$(sha256sum "${TEST_MANIFEST_CLEAN}" | awk '{print $1}')"
  echo "methods=KBSM_L9_r25,WAM_motion_L9_r25"
  echo "seeds=42-47"
  echo "protocol_fingerprint=${protocol_fingerprint}"
} > "${BASE}/protocol/run_manifest.txt"
cp "${TEST200}/test200_metadata.json" "${BASE}/protocol/test200_metadata.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYBIN GPU_LIST="${GPU_LIST:-0}" NPROC_PER_NODE=1
export DATA_DIR CACHE_DIR="${DATA_DIR}"
export EPOCHS=50 FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=1 NUM_WORKERS="${NUM_WORKERS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_MANIFEST

failures=0
for seed in $(seq 42 47); do
  for spec in "KBSM_L9_r25:bsm_ksim_gradual_vec:none" \
              "WAM_motion_L9_r25:bsm_taware_gradual_vec:motion"; do
    IFS=: read -r cfg strategy relevance <<< "${spec}"
    ckpt="${REFERENCE}/train/${cfg}__s${seed}/ckpt_best.pt"
    out="${BASE}/test/${cfg}__s${seed}"
    log="${BASE}/test/${cfg}__s${seed}.log"
    [[ -s "${ckpt}" ]] || { failures=$((failures + 1)); continue; }
    checkpoint_sha="$(sha256sum "${ckpt}" | awk '{print $1}')"
    expected_stamp="protocol=${protocol_fingerprint};checkpoint=${checkpoint_sha}"
    if [[ -s "${out}/metrics.json" && -s "${out}/test_results.md" && \
          -s "${out}/temporal_metrics.json" && -s "${out}/per_sample_metrics.json" && \
          -s "${out}/provenance.txt" && \
          "$(cat "${out}/provenance.txt")" == "${expected_stamp}" ]]; then
      echo "[skip] ${cfg} seed=${seed}"
      continue
    fi
    [[ ! -e "${out}" ]] || mv "${out}" "${out}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[test200] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cfg} seed=${seed}"
    if ! SEED="${seed}" TEST_MANIFEST="${TEST_MANIFEST_CLEAN}" \
        FROZEN_PREDICTOR=1 PREDICTOR_CKPT="${ckpt}" \
        DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY="${strategy}" \
        DENSE_JEPA_MERGE_LAYERS=12,13,14,15,16,17,18,19,20 \
        DENSE_JEPA_MERGE_RATIO=0.25 DENSE_JEPA_RESTORE_DENSE=1 \
        DENSE_JEPA_RELEVANCE_SOURCE="${relevance}" DENSE_JEPA_RELEVANCE_LAMBDA=1.0 \
        OUT_DIR="${out}" bash scripts/train.sh > "${log}" 2>&1; then
      failures=$((failures + 1)); continue
    fi
    "${PYBIN}" - "${out}/per_sample_metrics.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
rows = payload.get("rows", [])
assert payload.get("num_samples") == 200 == len(rows), payload.get("num_samples")
assert len({row["sample_id"] for row in rows}) == 200
PY
    rm -f "${out}/ckpt_latest.pt" "${out}/ckpt_best.pt"
    printf '%s\n' "${expected_stamp}" > "${out}/provenance.txt"
  done
done

if (( failures != 0 )); then
  printf 'failures=%s\nfinished_utc=%s\n' "${failures}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BASE}/protocol/completion.txt"
  exit 1
fi
"${PYBIN}" "${SCRIPT_DIR}/aggregate_paired.py" "${BASE}/test" \
  --left KBSM_L9_r25 --right WAM_motion_L9_r25 --seeds 42-47 --mode final
"${PYBIN}" "${SCRIPT_DIR}/aggregate_clustered_clips.py" "${BASE}/test" \
  --left KBSM_L9_r25 --right WAM_motion_L9_r25 --seeds 42-47 \
  --bootstrap-seed 202718 --replicates 10000
printf 'failures=0\nfinished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${BASE}/protocol/completion.txt"
