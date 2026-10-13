#!/bin/bash
# Round T: vid-TLDR features (after the round G extraction), then 40 CV probe runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
until grep -q "G EXTRACT DONE" $W/tmp/stage_HSG_extract.log; do sleep 60; done
G=(1 2 4 5)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY extract.py --configs vidtldrpa_L9 --shard $i --nshards 4 --batch 4 > $W/feats/logs/extract_T_$i.log 2>&1 &
done
wait
echo "T EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_T_*.log | head -4
J=$W/runs/cv/jobs_T.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for sub, extra in (("vispose_k1", " --pose_frames 1"), ("vis", "")):
        inp = "vispose" if sub == "vispose_k1" else "vis"
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/vidtldrpa_L9/{sub}_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg vidtldrpa_L9 --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80{extra} --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus 1,2,4,5 --per_gpu 1
echo "T DONE"
