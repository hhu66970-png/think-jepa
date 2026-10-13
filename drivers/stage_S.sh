#!/bin/bash
# Round S(b) (seventh review): features of the fixed30 and uniform sets (capped), then the jobs file; pools started separately.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
CFGS=kbsmpa_L9,wampa_L9,wamshufpa_L9,flowpa_L9,flowshufpa_L9,pospa_L9,kbsmpa_L12,wampa_L12,wamshufpa_L12
until grep -q "DV EXTRACT" $W/tmp/stage_DV.log; do sleep 60; done
for m in fixed30 uniform; do
  export WAM_SLIM=$W/data/egodex_fps/$m/slim WAM_SPLITS=$W/data/egodex_fps/$m/splits WAM_FEAT_ROOT=$W/feats_fps/$m
  for s in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY capfrac.py 0.25 extract.py --configs $CFGS --shard $s --nshards 6 --batch 2 > $WAM_FEAT_ROOT/logs/extract_S_$s.log 2>&1 &
  done
  wait
  grep -il traceback $WAM_FEAT_ROOT/logs/extract_S_*.log && { echo "S EXTRACT $m FAILED"; exit 1; }
done
echo "S EXTRACT DONE"
J=$W/runs_S/jobs_S.jsonl; mkdir -p $W/runs_S; rm -f $J
$PY - <<PYEOF
import json
W = "$W"
def job(m, cfg, inp, sub, extra, s, f):
    out = f"{W}/runs_S/{m}/cv/{none if inp == pose else cfg}/{sub}_cv{f}/s{s}"
    cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={W}/data/egodex_fps/{m}/slim WAM_SPLITS={W}/data/egodex_fps/{m}/splits "
           f"WAM_FEAT_ROOT={W}/feats_fps/{m} WAM_RUN_ROOT={W}/runs_S/{m} CAPFRAC "
           f"train_probe.py --cfg {cfg} --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f} --rows_from kbsmpa_L9{extra}")
    return dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")
K1 = " --pose_frames 1"
cp = ["kbsmpa_L9", "wampa_L9", "wamshufpa_L9", "flowpa_L9", "flowshufpa_L9", "pospa_L9", "kbsmpa_L12", "wampa_L12", "wamshufpa_L12"]
arms = [(c, "vispose", "vispose_k1", K1) for c in cp] + [("kbsmpa_L9", "pose", "pose_k1", K1)]
arms += [(c, "vis", "vis", "") for c in ["kbsmpa_L9", "wampa_L9", "wamshufpa_L9"]]
with open("$J", "w") as fh:
    for f in range(5):
        for s in range(4):
            for m in ("fixed30", "uniform"):
                for cfg, inp, sub, extra in arms:
                    fh.write(json.dumps(job(m, cfg, inp, sub, extra, s, f)) + "\n")
PYEOF
wc -l $J
echo "S JOBS READY"
