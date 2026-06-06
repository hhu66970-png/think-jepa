#!/usr/bin/env bash
# 远端自驱 GPU 队列:E1-extreme -> agg -> oracle_n12 -> agg(n12)。不依赖本地 SSH 存活。
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
PY=/root/autodl-tmp/thinkjepa-work/miniconda3/envs/thinkjepa-train/bin/python
Q=outputs/_queue.log
echo "[queue start $(date +%H:%M:%S)]" > $Q
while pgrep -f frontier_extreme.sh >/dev/null; do sleep 60; done
echo "[E1-extreme done $(date +%H:%M:%S)]" >> $Q
$PY agg_extreme.py > outputs/_agg_extreme.txt 2>&1; echo "[agg_extreme done]" >> $Q
echo "[oracle_n12 start $(date +%H:%M:%S)]" >> $Q
bash oracle_n12.sh >> $Q 2>&1
echo "[oracle_n12 done $(date +%H:%M:%S)]" >> $Q
$PY /tmp/agg_oracle.py > outputs/_agg_oracle_n12.txt 2>&1
echo "[QUEUE_ALL_DONE $(date +%H:%M:%S)]" >> $Q
