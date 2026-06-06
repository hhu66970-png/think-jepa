#!/bin/bash
# WAM-only frozen frontier with CORRECT Key metric, into a FRESH root (no deletion
# of old feature-metric wam_frontier dirs). dense/K-BSM/PiToMe reuse old wam_frontier
# (already key). Reads val_avg_dist (the real ADE; old results.tsv had inf bug).
set -u
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt
export TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
export FROZEN_PREDICTOR=1
CKPT=outputs/downstream_ext30_20260531/dense__s42/ckpt_best.pt
ROOT=outputs/wam_frontier_key; mkdir -p $ROOT
RES=$ROOT/wam_key_frozen.tsv
[ -f "$RES" ] || printf "name\tlayers\tratio\tADE_valavgdist\tFDE\ttokens\n" > $RES
run() {
  local name=$1 layers=$2 ratio=$3; local out=$ROOT/$name
  if [ -f "$out/metrics.json" ]; then echo "[skip] $name"; else
    echo "[run $(date +%H:%M:%S)] $name layers=$layers r=$ratio"
    SEED=42 DENSE_JEPA_TOKEN_MERGE=1 DENSE_JEPA_MERGE_STRATEGY=bsm_taware_gradual_vec \
      DENSE_JEPA_MERGE_LAYERS=$layers DENSE_JEPA_MERGE_RATIO=$ratio \
      DENSE_JEPA_RESTORE_DENSE=1 PREDICTOR_CKPT=$CKPT OUT_DIR=$out \
      bash scripts/train.sh > $out.log 2>&1 || echo "  [warn] $name rc=$?"
  fi
  $PYBIN - "$name" "$layers" "$ratio" "$out" "$RES" <<'PY'
import json,sys,os,re
name,layers,ratio,out,res=sys.argv[1:6]
ade=fde=tok="NA"
mj=os.path.join(out,"metrics.json")
if os.path.exists(mj):
    try:
        ep=json.load(open(mj)).get("epochs",[])
        if ep: ade=ep[-1].get("val_avg_dist","NA"); fde=ep[-1].get("val_final_dist","NA")
    except Exception as e: ade=f"ERR:{e}"
log=out+".log"
if os.path.exists(log):
    m=re.findall(r'num_tokens_after["\']?\s*[:=]\s*(\d+)', open(log,errors="ignore").read())
    if m: tok=m[-1]
open(res,"a").write(f"{name}\t{layers}\t{ratio}\t{ade}\t{fde}\t{tok}\n")
print(f"  -> {name}: ADE(valavg)={ade} FDE={fde} tokens={tok}")
PY
}
echo "==== WAM frozen (key) START $(date +%H:%M:%S) ===="
run wam_r15 12,14,16,18,20 0.15
run wam_r20 12,14,16,18,20 0.20
run wam_r25 12,14,16,18,20 0.25
run wam_L9_r20  12,13,14,15,16,17,18,19,20 0.20
run wam_L9_r25  12,13,14,15,16,17,18,19,20 0.25
run wam_L10_r20 12,13,14,15,16,17,18,19,20,21 0.20
run wam_L10_r25 12,13,14,15,16,17,18,19,20,21 0.25
run wam_L12_r20 10,11,12,13,14,15,16,17,18,19,20,21 0.20
run wam_L12_r25 10,11,12,13,14,15,16,17,18,19,20,21 0.25
echo "==== WAM frozen (key) DONE $(date +%H:%M:%S) ===="; cat $RES
