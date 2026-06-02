#!/bin/bash
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
while pgrep -f "wam_frontier2.sh|enc_cos_frontier.py" >/dev/null; do sleep 8; done
echo "[chain2] start speed+protection-dump $(date +%H:%M:%S)"
B=/root/autodl-tmp/thinkjepa-work
export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$B/miniconda3/envs/thinkjepa-train/bin/python
C1=$B/tiny-cache/part2/assemble_disassemble_furniture_bench_chair/1873_L8_nf32_res256_new16_s00of08.npz
C2=$B/tiny-cache/part2/assemble_disassemble_furniture_bench_drawer/5522_L8_nf32_res256_new16_s02of08.npz
C3=$B/cache_ext30/part2/fold_unfold_paper_basic/1082_L8_nf32_res256_new16_s03of08.npz
echo "[chain2] enc_speed"; $PY tools/enc_speed.py --clip "$C1" --out outputs/wam_frontier/enc_speed.tsv 2>&1 | tail -8
echo "[chain2] protection dump (5 methods incl WAM, 512/16)"; $PY tools/dump_pca_feats.py --out_dir $B/pca_dump_wam --clips "$C1,$C2,$C3" 2>&1 | tail -20
echo "[CHAIN2 DONE $(date +%H:%M:%S)]"
