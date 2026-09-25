#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/thinkjepa-work/ThinkJEPA}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/thinkjepa-work}"
PYBIN="${PYBIN:-${WORK_ROOT}/miniconda3/envs/thinkjepa-train/bin/python}"
SUBSET="${SUBSET:-${WORK_ROOT}/egodex_balanced_52_13_26_seed202707}"
TEST200="${TEST200:-${WORK_ROOT}/egodex_official_test200}"
PREP_PID_FILE="${PREP_PID_FILE:-/root/autodl-tmp/test200_prepare.pid}"
TEMPORAL_PID_FILE="${TEMPORAL_PID_FILE:-/root/autodl-tmp/temporal_formal_616_n6.pid}"
STATUS="${STATUS:-/root/autodl-tmp/test200_queue_status.txt}"

exec 9>/root/autodl-tmp/test200_queue.lock
flock -n 9 || { echo "[fatal] test200 queue already active" >&2; exit 6; }

wait_for_pid_file() {
  local label="$1"
  local pid_file="$2"
  local pid
  [[ -s "${pid_file}" ]] || { echo "[fatal] missing ${label} pid file" >&2; exit 4; }
  pid="$(cat "${pid_file}")"
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${label} pid=${pid}"
    sleep 30
  done
  echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${label} pid=${pid}"
}

cd "${REPO}" || exit 1

wait_for_pid_file "initial test200 download" "${PREP_PID_FILE}"
[[ -s "${TEST200}/test200_metadata.json" ]] || {
  echo "[fatal] initial test200 preparation did not produce metadata" >&2; exit 10;
}

echo "[verify] resolving dataset revision and hashing all 200 archives"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  "${PYBIN}" prepare_egodex_official_test200.py \
  --destination "${TEST200}" \
  --reference-subset "${SUBSET}" \
  --workers 4 \
  --required-guidance both

"${PYBIN}" - "${TEST200}/test200_metadata.json" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1]))
assert meta["official_test_count"] == 200
assert meta["downloaded_files"] == 200
assert meta["video_group_overlap"] == {
    "official_train_test": 0,
    "selected_train_validation_vs_test": 0,
}
assert len(meta["resolved_revision"]) >= 7
assert len(meta["files"]) == 200
assert all(len(row["sha256"]) == 64 for row in meta["files"])
PY

wait_for_pid_file "matched temporal experiment" "${TEMPORAL_PID_FILE}"
if ! grep -qx 'failures=0' \
    "${REPO}/outputs/revision_20260718_clean_temporal_616_n6/protocol/completion.txt"; then
  echo "[warn] temporal experiment did not complete cleanly; continuing with independent test200"
fi

echo "[test200] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting frozen evaluation"
bash run_frozen_test200_616_n6.sh

grep -qx 'failures=0' \
  "${REPO}/outputs/revision_20260718_frozen_test200_616_n6/protocol/completion.txt"
printf 'failures=0\nfinished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
