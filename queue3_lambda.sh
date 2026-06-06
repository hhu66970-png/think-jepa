#!/usr/bin/env bash
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
Q=outputs/_queue.log
while pgrep -f gpu_queue.sh >/dev/null; do sleep 60; done
echo "[queue3 lambda start $(date +%H:%M:%S)]" >> $Q
bash lambda_ablation.sh >> $Q 2>&1
$PY agg_lambda.py > outputs/_agg_lambda.txt 2>&1
echo "[QUEUE3_LAMBDA_DONE $(date +%H:%M:%S)]" >> $Q
