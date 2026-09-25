#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/thinkjepa-work/ThinkJEPA"
TEST200_PID_FILE="/root/autodl-tmp/test200_queue.pid"
TEST200_BASE="${REPO}/outputs/revision_20260718_frozen_test200_616_n6"
LOG="/root/autodl-tmp/spatial_formal_616_n6.log"
PID_FILE="/root/autodl-tmp/spatial_formal_616_n6.pid"

test200_pid="$(cat "${TEST200_PID_FILE}")"
while kill -0 "${test200_pid}" 2>/dev/null; do
  echo "[wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) official test200 pid=${test200_pid}"
  sleep 30
done

grep -qx 'failures=0' "${TEST200_BASE}/protocol/completion.txt" || {
  echo "[fatal] official test200 did not complete cleanly" >&2
  exit 1
}

cd "${REPO}"
DRY_RUN=1 bash run_clean_spatial_616_n6.sh > /root/autodl-tmp/spatial_616_n6_dryrun.log 2>&1
PYTHONPATH=vjepa2:tools \
  /root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python - \
  > /root/autodl-tmp/spatial_encoder_audit.log 2>&1 <<'PY'
import torch
import run_token_merge_pca_experiment as experiment

checkpoint = "vjepa2/vitl.pt"
manifest = "/root/autodl-tmp/thinkjepa-work/egodex_balanced_52_13_26_seed202707/manifests/train.txt"
clip = open(manifest).readline().strip()
model = experiment.build_model(checkpoint, 64, 256, 16, "bsm_ksim_gradual_vec", "cuda")
model.out_layers = [23]
experiment.apply_merge_config(
    model,
    enabled=True,
    strategy="bsm_ksim_gradual_vec",
    merge_layers=list(range(12, 21)),
    merge_ratio=0.25,
    restore_dense=True,
    bsm_match_metric="key",
    merge_axis="spatial",
)
video, _ = experiment.load_video(clip, 64, 256, "cuda")
with torch.no_grad():
    _, infos = model(video, return_merge_info=True, restore_dense=True)
tokens = [row.get("num_tokens_after") for row in infos]
axes = [row.get("merge_axis") for row in infos]
fallbacks = [row.get("fallback_reason") for row in infos]
print("clip", clip)
print("tokens_after", tokens)
print("axes", axes)
print("fallbacks", fallbacks)
assert tokens[-1] == 616, tokens
assert all(axis == "spatial" for axis in axes), axes
assert not any(fallbacks), fallbacks
print("SPATIAL_AXIS_ENCODER_AUDIT_PASS")
PY
nohup bash run_clean_spatial_616_n6.sh > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) spatial pid=$(cat "${PID_FILE}")"
