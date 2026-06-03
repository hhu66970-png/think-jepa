#!/bin/bash
# Real temporal audit (A1 per-step displacement, A2 velocity/accel error) for ALL
# 5 methods, SAME protocol: frozen eval of each method's OWN best ckpt (1 epoch,
# native temporal metrics), seed42, test10. Older runs lacked the temporal block.
set -uo pipefail
cd /root/autodl-tmp/thinkjepa-work/ThinkJEPA
B=/root/autodl-tmp/thinkjepa-work
export HF_ENDPOINT=https://hf-mirror.com PYBIN=$B/miniconda3/envs/thinkjepa-train/bin/python
export GPU_LIST=0 NPROC_PER_NODE=1 DATA_DIR=$B/cache_ext30/part2 CACHE_DIR=$B/cache_ext30/part2
export FORCE_ONLINE_VJEPA=1 PAST_T=32 FUTURE_T=32 TRAIN_BATCH_SIZE=2 TEST_BATCH_SIZE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MANIFEST=$B/downstream_ext30/manifests/train20.txt TEST_MANIFEST=$B/downstream_ext30/manifests/test10.txt
export FROZEN_PREDICTOR=1
BASE=outputs/downstream_ext30_20260531; OUT=outputs/temporal_audit; mkdir -p $OUT
ev() { local name=$1 tm=$2 strat=$3 layers=$4 ratio=$5 ckpt=$6
  if [ -f "$OUT/$name/metrics.json" ]; then echo "[skip] $name"; return; fi
  echo "[run $(date +%H:%M:%S)] $name"
  SEED=42 DENSE_JEPA_TOKEN_MERGE=$tm DENSE_JEPA_MERGE_STRATEGY=$strat DENSE_JEPA_MERGE_LAYERS=$layers \
    DENSE_JEPA_MERGE_RATIO=$ratio DENSE_JEPA_RESTORE_DENSE=1 PREDICTOR_CKPT=$BASE/${ckpt}__s42/ckpt_best.pt \
    OUT_DIR=$OUT/$name bash scripts/train.sh > $OUT/$name.log 2>&1 || echo "[warn] $name rc=$?"
}
ev dense   0 bsm_ksim_gradual_vec    12             0.0  dense
ev A_r25   1 local_2x2_same_time_vec 8              0.25 A_l8_r025
ev kbsm    1 bsm_ksim_gradual_vec    12,14,16,18,20 0.15 BSM_r015
ev pitome  1 bsm_pitome_gradual_vec  12,14,16,18,20 0.15 PITOME_r015
ev wam     1 bsm_taware_gradual_vec  12,14,16,18,20 0.15 WAM_r015
echo "==== TEMPORAL AUDIT DONE $(date +%H:%M:%S) ===="
$PYBIN - <<'PY'
import json,os
OUT="outputs/temporal_audit"
order=[("dense","dense"),("A_r25","方案A r0.25"),("kbsm","K-BSM r0.15"),("pitome","PiToMe r0.15"),("wam","WAM r0.15")]
def temp(n):
    p=f"{OUT}/{n}/metrics.json"
    if not os.path.exists(p): return None
    ep=json.load(open(p)).get("epochs",[])
    return ep[-1].get("temporal") if ep else None
T={n:temp(n) for n,_ in order}
print("\n=== TEMPORAL AUDIT (frozen own-ckpt eval, seed42, test10) ===")
print(f"{'method':14s}{'ADE':>9}{'FDE':>9}{'vel_err':>10}{'acc_err':>10}{'maxΔstep vs dense':>20}")
dpsd=(T.get('dense') or {}).get('per_step_displacement')
for n,lab in order:
    t=T.get(n)
    if not t: print(f"{lab:14s}  (no data)"); continue
    ep=json.load(open(f"{OUT}/{n}/metrics.json"))["epochs"][-1]
    ade=ep.get("val_avg_dist"); fde=ep.get("val_final_dist")
    psd=t.get("per_step_displacement")
    md=max(abs(a-b) for a,b in zip(psd,dpsd)) if (psd and dpsd and n!="dense") else 0.0
    print(f"{lab:14s}{ade:>9.4f}{fde:>9.4f}{t['velocity_error']:>10.4f}{t['accel_error']:>10.4f}{md:>20.4f}")
PY
