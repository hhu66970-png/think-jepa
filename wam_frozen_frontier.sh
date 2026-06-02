#!/bin/bash
# Frozen-predictor downstream ADE/FDE frontier (the WAM money plot).
# One converged DENSE predictor (s42) eats token-merged features from each method
# at several compression ratios; no retrain (isolates info preserved). Pushes r
# high to find where task-agnostic methods break and whether WAM holds longer.
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
mkdir -p $ROOT
RES=$ROOT/results.tsv
[ -f "$RES" ] || printf "name\tstrategy\tlayers\tratio\tADE\tFDE\ttokens\n" > $RES

run() {
  local name=$1 tm=$2 strat=$3 layers=$4 ratio=$5
  local out=$ROOT/$name
  if [ -f "$out/metrics.json" ]; then echo "[skip] $name"; else
    echo "[run $(date +%H:%M:%S)] $name strat=$strat layers=$layers r=$ratio"
    SEED=42 DENSE_JEPA_TOKEN_MERGE=$tm DENSE_JEPA_MERGE_STRATEGY=$strat \
      DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
      DENSE_JEPA_RESTORE_DENSE=1 PREDICTOR_CKPT=$CKPT OUT_DIR=$out \
      bash scripts/train.sh > $out.log 2>&1 || echo "  [warn] $name rc=$?"
  fi
  $PYBIN - "$name" "$strat" "$layers" "$ratio" "$out" "$RES" <<'PY'
import json,sys,os,re
name,strat,layers,ratio,out,res=sys.argv[1:7]
ade=fde=tok="NA"
mj=os.path.join(out,"metrics.json")
if os.path.exists(mj):
    try:
        d=json.load(open(mj)); ep=d.get("epochs",[])
        if ep: ade=ep[-1].get("val_avg_dist","NA"); fde=ep[-1].get("val_final_dist","NA")
        if isinstance(d.get("best"),dict): ade=d["best"].get("ade",ade)
    except Exception as e: ade=f"ERR:{e}"
log=out+".log"
if os.path.exists(log):
    m=re.findall(r'num_tokens_after["\']?\s*[:=]\s*(\d+)', open(log,errors="ignore").read())
    if m: tok=m[-1]
open(res,"a").write(f"{name}\t{strat}\t{layers}\t{ratio}\t{ade}\t{fde}\t{tok}\n")
print(f"  -> {name}: ADE={ade} FDE={fde} tokens={tok}")
PY
}

# dense sanity first (should reproduce ~0.130), then r0.15 5-way, then aggressive.
run dense      0 bsm_ksim_gradual_vec    12             0.0
run A_r25      1 local_2x2_same_time_vec 8              0.25
run kbsm_r15   1 bsm_ksim_gradual_vec    12,14,16,18,20 0.15
run pitome_r15 1 bsm_pitome_gradual_vec  12,14,16,18,20 0.15
run wam_r15    1 bsm_taware_gradual_vec  12,14,16,18,20 0.15
run kbsm_r20   1 bsm_ksim_gradual_vec    12,14,16,18,20 0.20
run pitome_r20 1 bsm_pitome_gradual_vec  12,14,16,18,20 0.20
run wam_r20    1 bsm_taware_gradual_vec  12,14,16,18,20 0.20
run kbsm_r25   1 bsm_ksim_gradual_vec    12,14,16,18,20 0.25
run pitome_r25 1 bsm_pitome_gradual_vec  12,14,16,18,20 0.25
run wam_r25    1 bsm_taware_gradual_vec  12,14,16,18,20 0.25
echo "[DONE $(date +%H:%M:%S)] frontier -> $RES"; cat $RES
