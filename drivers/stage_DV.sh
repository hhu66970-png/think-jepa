#!/bin/bash
# Round DV (seventh review): Diving48 features (capped, sharing GPUs with R5), 360 CV classifier runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_SLIM=$W/data/diving48/v2test/slim WAM_SPLITS=$W/data/diving48/v2test/splits WAM_FEAT_ROOT=$W/feats_DV WAM_RUN_ROOT=$W/runs_DV
CFGS=dense,kbsmpa_L9,kbsmpa_L12,pospa_L9,randmpa_L9,in16f,wampa_L9,wamshufpa_L9,wampa_L12,wamshufpa_L12,flowpa_L9,flowshufpa_L9,vidtldrpa_L9,vidtldrshufpa_L9,fmpa_L9,vidtldrnat_L9,fmnat_L9,vidtldrnatshuf_L9
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
for s in 0 1 2 3 4 5; do
  ( until [ "$(free_mb $s)" -ge 8500 ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY capfrac.py 0.25 extract.py --configs $CFGS --shard $s --nshards 6 --batch 2 > $WAM_FEAT_ROOT/logs/extract_DV_$s.log 2>&1 ) &
  sleep 20
done
wait
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_DV_0.log
grep -il traceback $WAM_FEAT_ROOT/logs/extract_DV_*.log && { echo "DV EXTRACT FAILED"; exit 1; }
echo "DV EXTRACT DONE"
J=$W/runs_DV/jobs_DV.jsonl; JD=$W/runs_DV/jobs_DV_big.jsonl; mkdir -p $W/runs_DV; rm -f $J $JD
$PY - <<PYEOF
import json
W = "$W"
def job(cfg, s, f, cap):
    out = f"{W}/runs_DV/cv_cls/{cfg}/vis_cv{f}/s{s}"
    pre = f"capfrac.py {cap} " if cap else ""
    cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={W}/data/diving48/v2test/slim WAM_SPLITS={W}/data/diving48/v2test/splits "
           f"WAM_FEAT_ROOT={W}/feats_DV WAM_RUN_ROOT={W}/runs_DV {W}/conda/bin/python {pre}train_cls.py --cfg {cfg} --seed {s} --cv_fold {f} --out {out}")
    return json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n"
small = "$CFGS".split(",")
big = ["dense", "in16f"]
with open("$J", "w") as fh:
    for c in small:
        if c not in big:
            for f in range(5):
                for s in range(4):
                    fh.write(job(c, s, f, 0.25))
with open("$JD", "w") as fh:
    for c in big:
        for f in range(5):
            for s in range(4):
                fh.write(job(c, s, f, None))
PYEOF
wc -l $J $JD
$PY gpu_pool_claim.py $J --gpus 0,1,2,3,4,5 --per_gpu 1
echo "DV SMALL DONE"
until grep -q "R5 CV DONE" $W/tmp/stage_R5.log; do sleep 60; done
$PY gpu_pool_claim.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
echo "DV CV DONE"
$PY roundDV_analysis.py
echo "DV ANALYSIS DONE"
