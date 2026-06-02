#!/bin/bash
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
while pgrep -f wam_frontier2.sh >/dev/null; do sleep 8; done
echo "[chain] sweep done $(date +%H:%M:%S); starting enc_cos"
B=/root/autodl-tmp/thinkjepa-work
export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$B/miniconda3/envs/thinkjepa-train/bin/python
$PY tools/enc_cos_frontier.py \
  --clips "$B/tiny-cache/part2/assemble_disassemble_furniture_bench_chair/1873_L8_nf32_res256_new16_s00of08.npz,$B/tiny-cache/part2/assemble_disassemble_furniture_bench_drawer/5522_L8_nf32_res256_new16_s02of08.npz,$B/tiny-cache/part2/insert_remove_furniture_bench_cabinet/542_L8_nf32_res256_new16_s01of08.npz" \
  --out outputs/wam_frontier/enc_cos.tsv
echo "[ENC_COS DONE $(date +%H:%M:%S)]"
