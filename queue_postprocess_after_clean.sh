#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
PYBIN="${PYBIN:-/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python}"
OLD_FRONTIER="${OLD_FRONTIER:-${REPO}/outputs/frontier_retrain_E1}"
R3637="${R3637:-${REPO}/outputs/revision_20260718_3637_n12}"
CLEAN="${CLEAN:-${REPO}/outputs/revision_20260718_clean_616_n6}"
SUBSET="${SUBSET:-/root/autodl-tmp/thinkjepa-work/egodex_balanced_52_13_26_seed202707}"
LOG="${CLEAN}/postprocess_queue.log"

mkdir -p "${CLEAN}"
exec >> "${LOG}" 2>&1
echo "[postprocess] queued $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while [[ ! -s "${CLEAN}/protocol/completion.txt" ]]; do
  echo "[wait] clean experiment $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 120
done
grep -qx 'failures=0' "${CLEAN}/protocol/completion.txt" || {
  echo "[fatal] clean experiment recorded failures"
  exit 30
}

while pgrep -f '[t]hinker_train.py' >/dev/null; do
  echo "[wait] GPU trainer still active $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

cd "${REPO}" || exit 31
"${PYBIN}" aggregate_paired.py "${R3637}" \
  --left KBSM_r015 --right WAM_motion_r015 --seeds 42-53 --mode plateau
"${PYBIN}" aggregate_paired.py "${CLEAN}/test" \
  --left KBSM_L9_r25 --right WAM_motion_L9_r25 --seeds 42-47 --mode final
"${PYBIN}" audit_frontier_family.py "${OLD_FRONTIER}" \
  --r3637-root "${R3637}" --mode plateau \
  --output "${R3637}/frontier_family_audit_5point_plateau.json"

"${PYBIN}" audit_wam_similarity_sign.py \
  --repo "${REPO}" \
  --manifest "${SUBSET}/manifests/test.txt" \
  --output "${CLEAN}/protocol/wam_selected_similarity_sign_test26.json" \
  --layers 12,13,14,15,16,17,18,19,20 \
  --ratio 0.25

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${CLEAN}/protocol/postprocess_complete.txt"
echo "[postprocess] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"

