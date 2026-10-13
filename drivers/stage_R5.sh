#!/bin/bash
# Rounds F, M, O, FL (seventh review): features (6 GPUs), 680 CV probe runs (12 slots), dense runs (one per GPU), analysis.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
CFGS=in8f,in12f,in8f+kbsmpa_L9,in8f+kbsmpa_L12,in12f+kbsmpa_L9,in12f+kbsmpa_L12,in16f+kbsmpa_L9,in16f+kbsmpa_L12,in24f+kbsmpa_L9,in24f+kbsmpa_L12,handcurq10pa_L12,handcurq10shufpa_L12,handcurrecvpa_L12,handcurrecvshufpa_L12,handfutpa_L12,handfutshufpa_L12,handfutpa_L9,handfutshufpa_L9,flowpa_L9,flowshufpa_L9,flowpa_L12,flowshufpa_L12
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_R5_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_R5_0.log
grep -il traceback $W/feats/logs/extract_R5_*.log && { echo "R5 EXTRACT FAILED"; exit 1; }
echo "R5 EXTRACT DONE"
$PY mechanism.py > $W/tmp/mechanism.log 2>&1 &
J=$W/runs/cv/jobs_R5.jsonl; JD=$W/runs/cv/jobs_R5_dense.jsonl; rm -f $J $JD
$PY - <<PYEOF
import json
W = "$W"
def job(cfg, inp, sub, extra, s, f):
    out = f"{W}/runs/cv/{cfg}/{sub}_cv{f}/s{s}"
    cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
           f"--cfg {cfg} --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f}{extra}")
    return json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n"
K1 = " --pose_frames 1"
F = ["in8f", "in12f", "in8f+kbsmpa_L9", "in8f+kbsmpa_L12", "in12f+kbsmpa_L9", "in12f+kbsmpa_L12", "in16f+kbsmpa_L9",
     "in16f+kbsmpa_L12", "in24f+kbsmpa_L9", "in24f+kbsmpa_L12"]
arms = [(c, "vispose", "vispose_k1", K1) for c in F] + [(c, "vis", "vis", "") for c in F]
arms += [(c, "vispose", "vispose_k1", K1) for c in ["handcurq10pa_L12", "handcurq10shufpa_L12", "handcurrecvpa_L12",
         "handcurrecvshufpa_L12", "handfutpa_L12", "handfutshufpa_L12", "handfutpa_L9", "handfutshufpa_L9", "flowpa_L9",
         "flowshufpa_L9", "flowpa_L12", "flowshufpa_L12"]]
arms += [(c, "vis", "vis", "") for c in ["flowpa_L9", "flowshufpa_L9"]]
with open("$J", "w") as fh:
    for cfg, inp, sub, extra in arms:
        for f in range(5):
            for s in range(4):
                fh.write(job(cfg, inp, sub, extra, s, f))
with open("$JD", "w") as fh:
    for f in range(5):
        for s in range(4):
            fh.write(job("dense", "vispose", "vispose_k1", K1, s, f))
PYEOF
wc -l $J $JD
$PY gpu_pool_claim.py $J --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
$PY gpu_pool_claim.py $JD --gpus 0,1,2,3,4,5 --per_gpu 1
echo "R5 CV DONE"
$PY round5_analysis.py
echo "R5 ANALYSIS DONE"
