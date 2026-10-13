#!/bin/bash
# Rounds C9, CF, SS, SM: feature extraction (6 GPUs), then 140 CV probe runs (12 slots), then round3_analysis.py.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
CFGS=handcurpa_L9,handcurshuffpa_L9,handf31pa_L12,handf31shuffpa_L12,vidtldrshufpa_L9,vidtldrnatshuf_L9,fmmid_L9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_R3_$s.log 2>&1 &
done
wait
echo "R3 EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_R3_0.log
J=$W/runs/cv/jobs_R3.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
arms = ["handcurpa_L9", "handcurshuffpa_L9", "handf31pa_L12", "handf31shuffpa_L12", "vidtldrshufpa_L9", "vidtldrnatshuf_L9", "fmmid_L9"]
with open(J, "w") as fh:
    for cfg in arms:
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/vispose_k1_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${R3_GPUS:-0,0,1,1,2,2,3,3,4,4,5,5} --per_gpu 1
echo "R3 CV DONE"
$PY round3_analysis.py
echo "R3 ANALYSIS DONE"
