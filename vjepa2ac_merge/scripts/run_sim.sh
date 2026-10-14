#!/bin/bash
# The whole simulated pipeline in order, on one machine with 4+ GPUs (RTX 3090 timings in the comments).
# Run from the package root. Every step is restartable; each writes into data/, ckpt/, results/ or $VJEPA2_CACHE.
set -e
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
PY=${PY:-python}
NG=${NG:-4}                                        # GPUs for training (batch 32 is split over them)

# 0. official repo and checkpoint (once)
[ -d vjepa2_official ] || git clone https://github.com/facebookresearch/vjepa2 vjepa2_official
mkdir -p ckpt data/sim results cache
[ -s ckpt/vjepa2-ac-vitg.pt ] || curl -L -o ckpt/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt

# 1. installation check: merged predictor with identity groups == official predictor; timing per token count
$PY t0_offline.py ckpt/vjepa2-ac-vitg.pt                                        # -> results/t0.json

# 2. zero shot check in the simulator (expected to FAIL: one step L1 worse than copying the frame)
$PY t1_landscape.py --states 100-199 --out results/t1.jsonl

# 3. random play data, 800 episodes x 17 frames (CPU, 16 processes)
for k in $(seq 0 15); do
  lo=$((1000 + k * 50))
  $PY s3_collect.py --seeds $lo-$((lo + 49)) --out data/sim/play_$lo.npz > results/collect_$lo.log 2>&1 &
done; wait

# 4. S3: plain fine tune of the predictor (encoder frozen; tokens cached once)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k $PY s3_train.py encode --part $k/4 & done; wait   # ~10 GB fp16 memmap
$PY -m torch.distributed.run --nproc_per_node $NG s3_train.py train                           # 2000 steps, ~1 h on 4 GPUs
$PY s3_train.py eval      # gate: fine tuned L1 <= 0.9 x copy L1 and argmin cosine >= 0.5 (results/s3_eval.json)

# 5. S4: merge aware fine tune from S3 (random merger and n in {64, 16} per sample)
$PY s4_train.py rel                                                              # gripper relevance for gate
$PY -m torch.distributed.run --nproc_per_node $NG s4_train.py train --seed 0 --out ckpt/s4_k0.pt   # ~1.4 h
#    equal training control (same recipe, full tokens, no merging): S3b
$PY -m torch.distributed.run --nproc_per_node $NG s4_train.py train --seed 0 --plain --out ckpt/s3b_k0.pt
$PY s4_train.py eval --out ckpt/s4_k0.pt                                         # S3 vs S4 on every merged arm

# 6. closed loop reach (one GPU per process; 25 episodes per job, split across GPUs as you like)
#    METHOD: S4, sim n = 64, 102 samples   |   CONTROL: S3b, 256 tokens, 28 samples   (equal predictor tokens)
mkdir -p results/c1
CUDA_VISIBLE_DEVICES=0 $PY sim_run.py --arm sim --n 64 --samples 102 --episodes 2000-2099 \
    --pred_ckpt ckpt/s4_k0.pt --out results/c1/s4k0_stack_sim_e2000.jsonl &
CUDA_VISIBLE_DEVICES=1 $PY sim_run.py --arm full --n 256 --samples 28 --episodes 2000-2099 \
    --pred_ckpt ckpt/s3b_k0.pt --out results/c1/s3bk0_few28_e2000.jsonl &
wait

# 7. paired bootstrap analysis (file names <tag>_<arm>_e<first episode>.jsonl, as analyze.py expects)
$PY analyze.py                                                                   # -> results/v_summary.txt
