#!/bin/bash
# Rounds B, P, D, H (sixth review): features for the new arms, then 620 CV probe runs, then round4_analysis.py.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
CFGS=pospa_L9,pospa_L12,randmpa_L9,randmpa_L12,randdroppa_L9,randdroppa_L12,in16f,in128,handcurq20pa_L12,handcurq20shufpa_L12,wamq20pa_L9,wamq20shufpa_L9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_R4_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_R4_0.log
grep -il traceback $W/feats/logs/extract_R4_*.log && { echo "R4 EXTRACT FAILED"; exit 1; }
echo "R4 EXTRACT DONE"
J=$W/runs/cv/jobs_R4.jsonl; JD=$W/runs/cv/jobs_R4_deep.jsonl; rm -f $J $JD
$PY - <<PYEOF
import json
W = "$W"
def job(cfg, inp, sub, extra, s, f):
    out = f"{W}/runs/cv/{cfg if inp != 'pose' else 'none'}/{sub}_cv{f}/s{s}"
    cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
           f"--cfg {cfg} --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f}{extra}")
    return json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n"
K1 = " --pose_frames 1"
arms = []
for c in ["pospa_L9", "pospa_L12", "randmpa_L9", "randmpa_L12", "randdroppa_L9", "randdroppa_L12", "in16f", "in128"]:
    arms.append((c, "vispose", "vispose_k1", K1))
for c in ["pospa_L9", "in16f"]:
    arms.append((c, "vis", "vis", ""))
for c in ["dense", "kbsmpa_L9", "kbsmpa_L12", "handf31pa_L12", "handf31shufpa_L12", "handcurpa_L12", "handcurshufpa_L12", "wampa_L9", "wamshufpa_L9"]:
    arms.append((c, "vispose", "vispose_k1_local", K1 + " --target local"))
arms.append(("dense", "pose", "pose_k1_local", K1 + " --target local"))
for c in ["handcurq20pa_L12", "handcurq20shufpa_L12", "wamq20pa_L9", "wamq20shufpa_L9"]:
    arms.append((c, "vispose", "vispose_k1", K1))
deep = [(c, "vispose", "vispose_k1_d6", K1 + " --depth 6 --dim 512") for c in
        ["dense", "kbsmpa_L9", "wampa_L9", "wamshufpa_L9", "kbsmpa_L12", "handf31pa_L12", "handf31shufpa_L12"]]
for path, A in (("$J", arms), ("$JD", deep)):
    with open(path, "w") as fh:
        for cfg, inp, sub, extra in A:
            for f in range(5):
                for s in range(4):
                    fh.write(job(cfg, inp, sub, extra, s, f))
PYEOF
wc -l $J $JD
$PY gpu_pool_claim.py $J --gpus ${R4_GPUS:-0,0,1,1,2,2,3,3,4,4,5,5} --per_gpu 1
$PY gpu_pool_claim.py $JD --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
echo "R4 CV DONE"
$PY round4_analysis.py
echo "R4 ANALYSIS DONE"
