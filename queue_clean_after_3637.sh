#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
SUBSET="${SUBSET:-/root/autodl-tmp/thinkjepa-work/egodex_balanced_52_13_26_seed202707}"
BASE_3637="${BASE_3637:-${REPO}/outputs/revision_20260718_3637_n12}"
BASE_CLEAN="${BASE_CLEAN:-${REPO}/outputs/revision_20260718_clean_616_n6}"
LOG="${BASE_CLEAN}/queue.log"

mkdir -p "${BASE_CLEAN}"
exec >> "${LOG}" 2>&1
echo "[queue] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while [[ ! -s "${SUBSET}/subset_metadata.json" ]]; do
  if ! pgrep -f '[p]repare_egodex_balanced_subset.py' >/dev/null; then
    echo "[fatal] subset preparation stopped without metadata"
    exit 20
  fi
  echo "[wait] subset preparation $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

for name in train validation test; do
  [[ -s "${SUBSET}/manifests/${name}.txt" ]] || {
    echo "[fatal] missing ${name} manifest"
    exit 21
  }
done

while pgrep -f '[r]un_3637_n12.sh' >/dev/null; do
  echo "[wait] 3637-token run $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

if [[ ! -s "${BASE_3637}/protocol/completion.txt" ]]; then
  echo "[fatal] 3637-token runner exited without completion record"
  exit 22
fi
if ! grep -qx 'failures=0' "${BASE_3637}/protocol/completion.txt"; then
  echo "[fatal] 3637-token runner recorded failures"
  exit 23
fi

free_kb="$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
if (( free_kb < 20 * 1024 * 1024 )); then
  echo "[fatal] less than 20 GiB free on data volume"
  exit 24
fi

echo "[run] clean 616-token experiment $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "${REPO}" || exit 25
SUBSET="${SUBSET}" BASE="${BASE_CLEAN}" ./run_clean_616_n6.sh
rc=$?
if (( rc != 0 )); then
  echo "[fatal] clean experiment failed rc=${rc}"
  exit "${rc}"
fi

./aggregate_paired.py "${BASE_CLEAN}/test" \
  --left KBSM_L9_r25 --right WAM_motion_L9_r25 \
  --seeds 42-47 --mode final
echo "[queue] completed $(date -u +%Y-%m-%dT%H:%M:%SZ)"

