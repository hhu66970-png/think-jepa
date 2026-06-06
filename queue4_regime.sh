#!/usr/bin/env bash
# 等 λ(queue3)跑完 → 在 616 token 重新 dump 各方法 merge gids(供 regime 保护可视化 D14)。复用 dump_pca_feats.py。
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
Q=outputs/_queue.log
B=/root/autodl-tmp/thinkjepa-work/cache_ext30/part2
C1=$B/cache_ext30/part2/assemble_disassemble_furniture_bench_chair/1013_L8_nf32_res256_new16_s03of08.npz
C2=$B/cache_ext30/part2/fold_unfold_paper_basic/1082_L8_nf32_res256_new16_s03of08.npz
C3=$B/cache_ext30/part2/assemble_disassemble_furniture_bench_chair/1098_L8_nf32_res256_new16_s00of08.npz
while ! grep -q QUEUE3_LAMBDA_DONE "$Q" 2>/dev/null; do sleep 60; done
echo "[queue4 regime-dump start $(date +%H:%M:%S)]" >> $Q
env PYTHONPATH=/root/autodl-tmp/thinkjepa-work/ThinkJEPA/vjepa2 $PY tools/dump_pca_feats.py --clips "$C1,$C2,$C3" --bsm_layers 12,13,14,15,16,17,18,19,20 --bsm_ratio 0.25 --out_dir outputs/pca_dump_regime616 >> $Q 2>&1
echo "[QUEUE4_REGIME_DUMP_DONE $(date +%H:%M:%S)]" >> $Q
