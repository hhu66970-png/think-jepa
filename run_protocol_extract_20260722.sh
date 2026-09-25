#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA

PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
MANIFEST=outputs/protocol64x256_clip_manifest_20260722.txt
OUT=outputs/redundancy_protocol64x256_lambda_20260722
LOG=outputs/redundancy_protocol64x256_lambda_20260722.log

if [[ ! -s "$MANIFEST" ]]; then
  echo "missing manifest: $MANIFEST" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing output: $OUT" >&2
  exit 3
fi

CLIPS=$(paste -sd, "$MANIFEST")
export PYTHONPATH=vjepa2

"$PY" tools/redundancy_extract_protocol_20260722.py \
  --clips "$CLIPS" \
  --num_frames 64 \
  --img_size 256 \
  --patch_size 16 \
  --cap_layers 2,5,8,11,12,14,17,20,23 \
  --deep_layer 20 \
  --bsm_layers 12,13,14,15,16,17,18,19,20 \
  --bsm_ratio 0.25 \
  --wam_lambdas 0,0.25,0.5,0.75,1 \
  --out_dir "$OUT" 2>&1 | tee "$LOG"

"$PY" - <<'PY'
from pathlib import Path
import hashlib
import json
import numpy as np

out = Path('outputs/redundancy_protocol64x256_lambda_20260722')
files = sorted(out.glob('feats_*.npz'))
rows = []
for path in files:
    data = np.load(path, allow_pickle=True)
    rows.append({
        'file': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'cid': str(data['cid'].item()),
        'task': str(data['task'].item()),
        'grid': [int(data['t']), int(data['h']), int(data['w'])],
        'token_count': int(data['token_count']),
        'lambda0_same_partition_kbsm': bool(data['lambda0_same_partition_kbsm']),
        'groups': {
            key[:-3]: int(data[key])
            for key in data.files if key.endswith('_ng')
        },
    })

summary = {
    'protocol': '64 frames, 256 px, patch 16, 32x16x16=8192',
    'n_files': len(files),
    'all_lambda0_same_partition_kbsm': all(
        row['lambda0_same_partition_kbsm'] for row in rows
    ),
    'rows': rows,
}
(out / 'MANIFEST.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
if len(files) != 25:
    raise SystemExit(f'expected 25 files, got {len(files)}')
if not summary['all_lambda0_same_partition_kbsm']:
    raise SystemExit('lambda=0 partition mismatch')
PY
