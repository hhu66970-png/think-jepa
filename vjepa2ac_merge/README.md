# Token merging for V-JEPA 2-AC planning

Code and results of a one day study: can the V-JEPA 2-AC planner get cheaper by merging the visual tokens the
predictor reads, and is that better than simply using fewer CEM samples? Simulated Franka reach (robosuite) plus an
offline test on real DROID frames. Everything was pre-registered in `PROTOCOL.md` before it ran.

## Short answer

| | result |
|---|---|
| Zero shot (official predictor, merged tokens) | fails: the official predictor does not even beat "copy the current frame" in our simulator |
| After plain fine tuning (S3) + merge aware fine tuning (S4) | merged planning works |
| Simulation, equal predictor compute, vs fewer samples alone | **better at every budget**: +16 points success (C1, ~3 s/step), +51 (C3), +45 (C4) |
| Simulation, equal wall clock (C4 budget, 1.4 s/step) | **better**: +31 and +39 points |
| Simulation, vs 4x more compute (102 samples, full tokens) | not shown: -4 points [-10, +2] (non-inferiority margin -10 just touched) |
| Real DROID frames, offline (official predictor is in domain) | **does not help**: the untouched official predictor with 28 samples is best |
| Merging without merge aware training | breaks planning (0.18 vs 0.92 success, H5) |

Take away for a real robot: fine tune the predictor on your own robot's logs first (plain, then merge aware), and
only then plan with `sim` merging to n = 64 groups plus about 4x fewer CEM samples. Before deploying, run the same
paired comparison as here on your own held out episodes: on DROID the official predictor was already good and the
method lost.

## Data flow

```
 camera frame (256x256)                      goal frame (256x256)
        |                                           |
 frozen V-JEPA 2 ViT-g encoder (once per step)   frozen encoder (once per episode)
        | 256 tokens x 1408                         | 256 goal tokens
        v                                           |
 MERGE: 256 tokens -> n groups (n = 64 or 16)       |
   sim  = bipartite matching on cosine              |
   pool = fixed 2x2 / 4x4 grid blocks               |
   group token = mean; position = centroid (RoPE takes fractional positions);
   attention adds log(group size) to the key logits |
        | n tokens                                  |
        v                                           |
 AC predictor (300M), rollout 2, conditioned on     |
 the sampled actions and the robot state            |
        | predicted n group tokens                  |
        v                                           |
 RESTORE: copy each group's prediction back to its member positions -> 256 tokens
        |                                           |
        +-------------> energy = mean L1 <----------+
                            |
 CEM: samples x iterations (official: 400 x 10, top 10, maxnorm 0.05)
                            |
        first action [dx, dy, dz, 0, 0, 0, dgrip] -> robot
```

Cost model (hardware independent): predictor tokens per CEM iteration = samples x 3 (n + 2) for rollout 2.
Full tokens with 400 samples: 310k. METHOD (n = 64, 102 samples): 20.2k. CONTROL (256 tokens, 28 samples): 21.7k.

## Files

| file | what |
|---|---|
| `acmerge.py` | the method: mergers (`sim`, `gate`, `gate_shuf`, `rand`, `pool`), predictor on groups, restore readout, CEM |
| `example_plan.py` | `MergedPlanner`: one control step on any robot; smoke test with timing |
| `ftlib.py` | fine tuning: official teacher forcing + 2 step rollout L1, merge aware or plain, torchrun data parallel |
| `sim_env.py` | robosuite Panda with V-JEPA 2-AC state and action conventions, front camera |
| `s3_collect.py` | random play data in the simulator |
| `s3_train.py` | S3: encode play data once, plain fine tune, held out check |
| `s4_train.py` | S4: merge aware fine tune (and `--plain` for the equal training control S3b), held out check |
| `sim_run.py` | closed loop reach episodes, one JSON line per episode |
| `d1_droid.py` | the same recipe on real DROID (droid_100): prep, encode, train, validation choice, offline test |
| `t0_offline.py` | installation check (identity merge == official predictor) and timing per token count |
| `t1_landscape.py` | zero shot check: one step L1 vs copy, energy landscape argmin vs true action |
| `analyze.py` | paired bootstrap analysis of `results/` -> `results/v_summary.txt` |
| `scripts/run_sim.sh` | the whole simulated pipeline in order |
| `tools/slim_ckpt.py` | encoder + predictor only checkpoint |
| `PROTOCOL.md` | pre-registration log: every plan, amendment and outcome with UTC time (cluster job files it mentions are not included) |
| `results/` | raw episode records of C1 to C4, D1, T3, and the summaries (rerun `python analyze.py` to reproduce) |

## Setup

```bash
pip install -r requirements.txt                    # torch for your CUDA first
git clone https://github.com/facebookresearch/vjepa2 vjepa2_official
mkdir -p ckpt && curl -L -o ckpt/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
python t0_offline.py ckpt/vjepa2-ac-vitg.pt        # "equivalence max abs diff" must be tiny
```

Environment variables (all optional): `VJEPA2_REPO` (official repo, default `./vjepa2_official`), `VJEPA2_AC_CKPT`
(default `./ckpt/vjepa2-ac-vitg.pt`), `VJEPA2_CACHE` (token memmaps, default `./cache`; use a fast disk),
`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` for headless rendering. One RTX 3090 (24 GB) per planning process; training
uses 4 GPUs (batch 32).

## Pipeline

`scripts/run_sim.sh` has every command. In short:

1. **Zero shot check** (`t1_landscape.py`). In our simulator the official predictor is worse than copying the frame
   (one step L1 0.43 vs 0.35) and its energy minimum points the wrong way (cosine with the true action 0.05). Closed
   loop: 0% success. Do this check on your robot first.
2. **S3, plain fine tune** on 800 random play episodes x 17 frames (`s3_collect.py`, `s3_train.py`). Gate:
   held out L1 at least 10% below copy and argmin cosine at least 0.5. We got 0.21 vs 0.29 and 0.73.
3. **S4, merge aware fine tune** from S3 (`s4_train.py train`): every training sample draws a random merger and
   n in {64, 16}; the loss is computed after restore, exactly as at planning time. 2000 steps, batch 32, lr 1e-4.
   Equal training control S3b: the same run with `--plain`.
4. **Plan** with S4, `--arm sim --n 64 --samples 102` (`sim_run.py`, or `MergedPlanner` on a robot).

## Results (simulation: 100 fresh episodes per set, paired, 95% bootstrap, 3 training seed pairs)

Success = final distance < 5 cm after 10 steps (goal 10 to 20 cm away). Plan seconds measured on an unshared RTX 3090.

| budget | arm | success (seeds 0/1/2) | s/step |
|---|---|---|---|
| reference | full tokens, 400 samples | (not run closed loop) | 45.3 |
| reference | full tokens, 102 samples (S3b) | 0.96 (seed 0) | 11.6 |
| C1 ~21k tok/iter | **METHOD** S4, sim n=64, 102 samples | 0.92 / 0.95 / 0.93 | 3.1 |
| | CONTROL S3b, 256 tokens, 28 samples | 0.83 / 0.70 / 0.78 | 3.3 |
| C3 ~10.8k | **METHOD3** S4, sim n=64, 51 samples | 0.92 / 0.89 / 0.90 | 1.8 |
| | S3b, 14 samples | 0.40 / 0.37 / 0.41 | 1.9 |
| | S3b, 28 samples x 5 iterations | 0.55 / 0.46 / 0.53 | 1.7 |
| C4 ~5.4k | **METHOD4** S4, sim n=64, 26 samples | 0.73 / 0.71 / 0.84 | 1.4 |
| | S3b, 14 x 5 / 28 x 2 (equal tokens) | 0.29-0.33 / 0.39-0.46 | 1.0 / 0.7 |
| | S3b, 28 x 4 / 14 x 7 (equal wall clock) | 0.41-0.53 / 0.36-0.38 | ~1.4 |

| test | difference in success [95% CI] | verdict |
|---|---|---|
| H1: METHOD - CONTROL (C1) | +0.163 [+0.090, +0.240]; final distance -1.41 cm | holds |
| H2: METHOD - full tokens 102 samples (4x cost) | -0.040 [-0.100, +0.020] | not shown (margin -0.10) |
| H5: METHOD - same merging on plain S3b | +0.740 [+0.650, +0.820] | holds: merge aware training is necessary |
| H6: METHOD3 - 14 samples / - 28x5 | +0.510 [+0.407, +0.610] / +0.390 [+0.293, +0.487] | holds |
| H7: METHOD4 - 14x5 / - 28x2 | +0.453 [+0.357, +0.550] / +0.330 [+0.227, +0.430] | holds |
| H8: METHOD4 - 28x4 / - 14x7 (equal wall clock) | +0.307 [+0.213, +0.403] / +0.387 [+0.290, +0.483] | holds |
| C2: harder reach (20-30 cm, 15 steps) | +0.040 [-0.040, +0.120]; final distance -0.68 cm [-1.10, -0.24] | success not shown |

Secondary (C1): `pool` n=64 also beats CONTROL (+0.12); merging to n=16 with 400 samples is worse than METHOD
(-0.11), so combining moderate merging with fewer samples beats either extreme.

**Real DROID (D1)**, offline: 70 / 10 / 20 episodes train / validation / test; checkpoint chosen on validation loss;
200 test transitions, goal = next frame, metric = cosine between planned and true xyz displacement.

| arm | cosine |
|---|---|
| official predictor, 28 samples | **0.551** |
| DP (plain fine tune), 28 samples | 0.426 |
| DP, 102 samples | 0.370 |
| DM (merge aware) pool n=64, 102 | 0.414 |
| DM sim n=64, 102 (METHOD) | 0.307 |
| DM sim n=16, 400 | 0.155 |
| official predictor, sim n=64, 102 (frozen merging) | 0.033 |

H3 (METHOD beats the 28 sample baselines) and H4 not shown. Fine tuning on 70 episodes already hurts the in domain
official predictor (DP - official -0.125). Merge aware training recovers most of what frozen merging loses (+0.274),
but not enough.

## Using it on your robot

1. Log your own robot: frames from the planning camera (crop and resize to 256 the same way at train and test
   time), state `[x, y, z, roll, pitch, yaw, closedness]` in the base frame, at the control period you plan with.
2. Write a `sample(rng)` that returns `(tokens [3, 256, 1408] fp16, states [3, 7], relevance [256])` for a random
   clip of 3 consecutive frames (see `d1_droid.py train`, which does exactly this for DROID; relevance only matters
   for `gate`, any per patch saliency works).
3. Check zero shot first (one step L1 vs copy, argmin cosine, as in `t1_landscape.py` / `s3_train.py eval`). If the
   official predictor already passes, compare against it and expect the DROID outcome.
4. `ftlib.train(P, sample, out, log, plain=True, ...)` then `plain=False` from that checkpoint; choose the checkpoint
   on held out validation loss only.
5. `MergedPlanner(pred_ckpt=out, method="sim", n=64, samples=102)`; compare paired with the plain predictor at 28
   samples on the same held out goals before trusting it.

## Caveats

- One simulated task (reach, gripper open, no objects), one camera, one robot. No real closed loop runs.
- H2 (no worse than 4x more compute) is not established.
- C1 to C4 planning times inside `results/` were recorded while GPUs were shared with other jobs; use the T3 numbers
  above (unshared) for wall clock.
- D1 trained 1000 steps instead of the planned 2000 (GPU sharing; recorded in `PROTOCOL.md` before any D1 result).
- Seeds: every training run is reported; nothing was selected on test episodes.
