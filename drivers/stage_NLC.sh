#!/bin/bash
# Rounds N, LS, C: regression test, feature extraction (6 GPUs), then 140 CV probe runs (12 slots).
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
BASELINE_DIR=$W/tmp/baseline_original CUDA_VISIBLE_DEVICES=0 $PY test_prop_attn.py > $W/runs/phaseA/test_prop_attn_roundN.log 2>&1
grep -E "ALL PASS|FAIL" $W/runs/phaseA/test_prop_attn_roundN.log | tail -2
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs vidtldrnat_L9,vidtldrmid_L9,wamlam05pa_L9,wamlam025pa_L9,handcurpa_L12,handcurshuffpa_L12 \
    --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_NLC_$s.log 2>&1 &
done
wait
echo "NLC EXTRACT DONE"; grep -h "K=" $W/feats/logs/extract_NLC_0.log
J=$W/runs/cv/jobs_NLC.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
arms = [("vidtldrnat_L9", "vispose_k1"), ("vidtldrnat_L9", "vis"), ("vidtldrmid_L9", "vispose_k1"),
        ("wamlam05pa_L9", "vispose_k1"), ("wamlam025pa_L9", "vispose_k1"),
        ("handcurpa_L12", "vispose_k1"), ("handcurshuffpa_L12", "vispose_k1")]
with open(J, "w") as fh:
    for cfg, sub in arms:
        inp, extra = ("vispose", " --pose_frames 1") if sub == "vispose_k1" else ("vis", "")
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/{sub}_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80{extra} --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l < $J
$PY gpu_pool_claim.py $J --gpus ${NLC_GPUS:-0,0,1,1,2,2,3,3,4,4,5,5} --per_gpu 1
echo "NLC CV DONE"
