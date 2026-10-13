#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
R=$W/runs/b2_train
cd $W/repo/wam_forecast
export WAM_FEAT_ROOT=$W/feats
mkdir -p $R
i=0
for pa in "" "--pa"; do
  for hp in "--tau 0.05 --lr_g 1e-4" "--tau 0.1 --lr_g 1e-4" "--tau 0.05 --lr_g 3e-4"; do
    tag=$(echo "pa${pa:+1}_$hp" | tr -d ' -' | tr '.' 'p')
    CUDA_VISIBLE_DEVICES=$i $PY train_dtem.py $pa $hp --out $R/$tag > $R/$tag.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo "B2 TRAINING FINISHED"
$PY - <<'P'
import json, glob, shutil, os
R = "$WAM_ROOT/runs/b2_train"
best = {}
for f in glob.glob(f"{R}/*/log.json"):
    d = json.load(open(f)); tag = f.split("/")[-2]; pa = d["args"]["pa"]
    print(tag, "best val ADE (soft)", round(d["best_val_ade_soft"], 2), "kept", d["log"][-1]["kept"])
    if pa not in best or d["best_val_ade_soft"] < best[pa][0]:
        best[pa] = (d["best_val_ade_soft"], os.path.dirname(f))
for pa, (v, d) in best.items():
    dst = "$WAM_ROOT/feats/" + ("b2_best_pa.pt" if pa else "b2_best_nopa.pt")
    shutil.copy(d + "/matchers.pt", dst); print("selected", "PA" if pa else "noPA", d, round(v, 2))
P
C="b2_s25,b2_L9,b2_L12,b2pa_s25,b2pa_L9,b2pa_L12"
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $C --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_b2_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_b2_0.log
grep -il traceback $W/feats/logs/extract_b2_*.log
for c in ${C//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
export WAM_FEAT_ROOT=$S
J=$W/runs/test_main/jobs_b2.jsonl; rm -f $J
$PY make_jobs.py test_main --cfgs b2_L9,b2_L12,b2pa_L9,b2pa_L12 --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs b2_s25,b2pa_s25 --inputs vis --seeds 0-9 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
for b in L9 L12 s25; do
  $PY pairs.py $W/runs/test_main b2pa_$b/vis:kbsmpa_$b/vis b2_$b/vis:kbsm_$b/vis
done
echo "B2 DONE"
