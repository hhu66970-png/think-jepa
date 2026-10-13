#!/bin/bash
# Round U: hand masks against the clip wide shuffle. (a) part 2 CV, (b) round X split; runs after round X finishes.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
while ! grep -q "X ANALYSIS DONE\|FAILED" $W/tmp/stage_X.log; do sleep 60; done
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs handcurshufpa_L12,handf31shufpa_L12 --shard $s --nshards 6 --batch 4 > $W/feats/logs/extract_U_$s.log 2>&1 &
done
wait
(export WAM_SLIM=$W/data/egodex_x/slim WAM_SPLITS=$W/data/egodex_x/splits WAM_FEAT_ROOT=$W/feats_X WAM_RUN_ROOT=$W/runs_X
 for s in 0 1 2 3 4 5; do
   CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs handcurshufpa_L12,handf31shufpa_L12 --shard $s --nshards 6 --batch 4 > $WAM_FEAT_ROOT/logs/extract_U_$s.log 2>&1 &
 done; wait)
grep -il traceback $W/feats/logs/extract_U_*.log $W/feats_X/logs/extract_U_*.log && { echo "U EXTRACT FAILED"; exit 1; }
echo "U EXTRACT DONE"
J=$W/runs/cv/jobs_U.jsonl; rm -f $J
$PY - <<PYEOF
import json
W = "$W"; J = "$J"
with open(J, "w") as fh:
    for cfg in ("handcurshufpa_L12", "handf31shufpa_L12"):
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs/cv/{cfg}/vispose_k1_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_FEAT_ROOT={W}/feats {W}/conda/bin/python train_probe.py "
                       f"--cfg {cfg} --input vispose --seed {s} --out {out} --lr 2e-4 --epochs 80 --pose_frames 1 --cv_fold {f}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
$PY gpu_pool_claim.py $J --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
(export WAM_SLIM=$W/data/egodex_x/slim WAM_SPLITS=$W/data/egodex_x/splits WAM_FEAT_ROOT=$W/feats_X WAM_RUN_ROOT=$W/runs_X
 JX=$WAM_RUN_ROOT/jobs_U.jsonl; rm -f $JX
 $PY make_jobs.py test_main --cfgs handcurshufpa_L12,handf31shufpa_L12 --inputs vispose --seeds 0-9 --tag _k1 --extra="--lr 2e-4 --epochs 80 --pose_frames 1" --out $JX
 $PY gpu_pool_claim.py $JX --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
 $PY roundX_analysis.py > $W/tmp/roundX_with_U.log 2>&1)
echo "U CV DONE"
$PY roundU_analysis.py
echo "U ANALYSIS DONE"
