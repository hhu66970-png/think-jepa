#!/bin/bash
# AGGRESSIVE compression sweep: push token count down (more merge layers) until
# task-agnostic ADE breaks, to expose WAM's protection advantage. Frozen dense
# predictor (s42). Same env as wam_frozen_frontier.sh.
set -u
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com
export PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1
export DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32
export TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt
export TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
export FROZEN_PREDICTOR=1
CKPT=outputs/downstream_ext30_20260531/dense__s42/ckpt_best.pt
ROOT=outputs/wam_frontier
RES=$ROOT/results2.tsv
[ -f "$RES" ] || printf "name\tstrategy\tlayers\tratio\tADE\tFDE\n" > $RES

run() {
  local name=$1 strat=$2 layers=$3 ratio=$4
  local out=$ROOT/$name
  if [ -f "$out/metrics.json" ]; then echo "[skip] $name"; else
    echo "[run $(date +%H:%M:%S)] $name layers=$layers r=$ratio"
    SEED=42 DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=$strat \
      DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
      DENSE_JEPA_RESTORE_DENSE=1 PREDICTOR_CKPT=$CKPT OUT_DIR=$out \
      bash scripts/train.sh > $out.log 2>&1 || echo "  [warn] rc=$?"
  fi
  $PYBIN - "$name" "$strat" "$layers" "$ratio" "$out" "$RES" <<'PY'
import json,sys,os
name,strat,layers,ratio,out,res=sys.argv[1:7]
ade=fde="NA"
mj=os.path.join(out,"metrics.json")
if os.path.exists(mj):
    try:
        ep=json.load(open(mj)).get("epochs",[])
        if ep: ade=ep[-1].get("val_avg_dist"); fde=ep[-1].get("val_final_dist")
    except Exception as e: ade=f"ERR:{e}"
open(res,"a").write(f"{name}\t{strat}\t{layers}\t{ratio}\t{ade}\t{fde}\n")
print(f"  -> {name}: ADE={ade} FDE={fde}")
PY
}

declare -A LAYERSETS=(
  [L9]="12,13,14,15,16,17,18,19,20"
  [L10]="12,13,14,15,16,17,18,19,20,21"
  [L12]="10,11,12,13,14,15,16,17,18,19,20,21"
)
for ln in L9 L10 L12; do
  layers=${LAYERSETS[$ln]}
  for r in 0.20 0.25; do
    rr=${r#0.}   # 0.20->20, 0.25->25
    run kbsm_${ln}_r${rr}   bsm_ksim_gradual_vec   $layers $r
    run pitome_${ln}_r${rr} bsm_pitome_gradual_vec $layers $r
    run wam_${ln}_r${rr}    bsm_taware_gradual_vec $layers $r
  done
done
echo "[DONE2 $(date +%H:%M:%S)]"; cat $RES
