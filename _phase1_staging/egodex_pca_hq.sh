#!/usr/bin/env bash
# High-quality PCA on EgoDex clips: 512px x 16-frame window (=8192 tokens, fits) + L4-6 + two-stage foreground + bicubic.
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "==== EGODEX HQ PCA START $(date +%H:%M:%S) ===="
for id in 542 1873 5522; do
  f=$(ls /root/autodl-tmp/thinkjepa-work/tiny-cache/part2/*/${id}_*.npz 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "[skip] no npz for $id"; continue; }
  echo "---- $id  $f ----"
  $PY tools/dense_jepa_pca_vis.py --npz "$f" --checkpoint vjepa2/vitl.pt --model_arch vit_large_rope \
    --img_size 512 --num_frames 16 --out_layers 4,5,6 \
    --pca_recipe foreground --foreground_method hybrid --foreground_quantile 0.7 \
    --background_mode desaturate --rgb_smooth 0.3 --rgb_saturation 1.35 --rgb_gamma 0.9 \
    --render_interp bicubic --out_dir outputs/F8b_PCA_高清/egodex_$id 2>&1 | tail -4
done
echo "==== EGODEX HQ PCA DONE $(date +%H:%M:%S) ===="
