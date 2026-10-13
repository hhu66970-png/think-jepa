#!/bin/bash
# After all pools: rerun the R5 jobs that failed for GPU memory (in12f video only), then the R5 analysis again.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "DV CV DONE" $W/tmp/stage_DV.log && grep -q "R5 ANALYSIS DONE" $W/tmp/stage_R5.log; do sleep 120; done
until grep -q "\[pool\] finished" $W/tmp/pool_S.log; do sleep 120; done
$PY - <<PYEOF
import json, os
W = "$W"
out = open(f"{W}/runs/cv/jobs_R5_rerun.jsonl", "w"); n = 0
for f in ("jobs_R5.jsonl", "jobs_R5_dense.jsonl"):
    for line in open(f"{W}/runs/cv/{f}"):
        j = json.loads(line)
        if not os.path.exists(j["done"]):
            if os.path.exists(j["done"] + ".claim"):
                os.remove(j["done"] + ".claim")
            out.write(line); n += 1
out.close(); print("rerun jobs", n)
PYEOF
$PY gpu_pool_claim.py $W/runs/cv/jobs_R5_rerun.jsonl --gpus 0,1,2,3,4,5 --per_gpu 1
$PY round5_analysis.py > $W/tmp/round5_analysis.log 2>&1
echo "TAIL DONE"
