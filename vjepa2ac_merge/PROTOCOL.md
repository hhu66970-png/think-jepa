# V-JEPA 2-AC with token merging: protocol (written before any run)

Carrier: official V-JEPA 2-AC (ViT-g encoder, AC predictor, `vjepa2-ac-vitg.pt`), frozen, no training.
Merging sits between the encoder output and the predictor (`acmerge.py`): the 256 tokens of the current frame become
n groups; the predictor reads n tokens per frame at the group centroids with proportional attention; the energy is the
official L1 to the goal frame, read out by copying group predictions back to their positions (restore, primary) or
against the goal merged with the same groups (compact, secondary). CEM is the official one (`mpc_utils.cem`):
rollout 2, 400 samples, top 10, 10 iterations, maxnorm 0.05, momenta 0.15 (WorldModel defaults).

Arms (n = 128, 64, 32, 16 tokens per frame unless stated):
- full: 256 tokens (no merging)
- sim: bipartite matching on encoder output cosine
- gate: sim with edge score s (1 - r_a)(1 - r_b), relevance r = projected gripper position (Gaussian, 2 patch sigma)
- gate_shuf: the same relevance values at permuted positions (seeded per step)
- rand: same bipartite procedure with random edges
- pool: fixed grid blocks (1x2, 2x2, 2x4, 4x4)
- fewer samples: full tokens with CEM samples cut to match the predictor token count of the n = 64 arm

## T0 offline (official example trajectory, 2 frames)
1. Equivalence: our predictor path with identity groups equals the official forward (max abs diff < 1e-3 in fp32).
2. Fidelity: L1 of the predicted next frame (restore) to the encoded next frame, per arm and n; reference = copy of
   the current frame and full prediction.
3. Energy landscape: 5 x 5 x 5 grid of xyz actions in [-0.075, 0.075]; Spearman correlation of each arm's energies
   with the full arm, and distance of each arm's argmin to the full argmin.
4. Timing on one RTX 3090: encoder per frame, one CEM call per arm (bf16 autocast, batch chunk 200), median of 3.

## S1 simulation baseline (MuJoCo, robosuite Panda, before any merging)
Franka Panda in robosuite, camera in front of the robot facing it (as in the official example frames), 256 x 256.
State = end effector xyz, extrinsic xyz Euler (gripper down), gripper closedness 0 to 1, robot base frame.
One action = target end effector displacement applied with the OSC controller for 0.25 s (4 Hz like training).
Task reach: start pose fixed, goal = end effector moved by a random displacement of 10 to 20 cm (seeded), goal image
rendered at that pose; 10 CEM steps per episode; success = final distance < 5 cm. 20 episodes for the baseline.
Decision rule: the merging comparison in S2 is run on success only if full reaches >= 40% success; otherwise S2 reports
the per step metrics below, which do not need task success.

## S2 simulation, arms
Same 20 episodes (same seeds, same goals) for every arm and n. Per step: planned action, cosine of its xyz with the
true direction to the goal, cosine with the full arm's action from the same state (the full arm is re planned from each
visited state of every arm), planning time. Per episode: final distance, success.
Primary contrasts: sim vs pool at equal n; gate vs gate_shuf at equal n; each merger vs fewer samples at equal
predictor tokens. Paired by episode; report mean differences with 95% bootstrap intervals (10k resamples over episodes).
No arm, n, or threshold is chosen after looking at S2 results.

## Amendments before S2 (written after T0 and the 2 episode sanity run of the full arm, before any merged arm ran in
## closed loop)
- Simulation settings fixed while calibrating with the full arm only: an action is tracked for 0.5 s (20 Hz OSC, the
  0.25 s window reached under 80% of a 5 cm step); state z is measured from the table top; orientation is held
  straight down (roll -pi, pitch 0, yaw 0, as in the official example); table top dark grey; camera 1.05 m in front,
  0.25 m to the side, 0.28 m above the base, looking at (0.45, 0, 0.25); start pose (0.50, 0, 0.26).
- Dropped the re planning of the full arm from every visited state (it would double the cost of every arm); agreement
  with the goal direction (cos) stays as the per step measure.
- Fewer samples arms: 102 samples (predictor tokens of n = 64) and 28 samples (n = 16), top 10, otherwise identical.
- n = 64 and 16 only (128 and 32 dropped for compute); 13 arms x 20 episodes, run as 52 jobs of 5 episodes on 6 GPUs.
- Primary outcome: final distance and progress (1 - final / start distance); success < 5 cm is reported but the
  20 episodes give it little power.
- T1 added (before it ran; T0 showed pooling keeping the one example's landscape far better than similarity merging,
  and one example cannot carry that): T0's fidelity and landscape measures on 100 simulated transitions (seeds 100 to
  199, disjoint from S2): random start offset up to (12, 15, 10) cm, random action up to 5 cm per axis, next frame as
  goal, 5^3 grid in [-0.05, 0.05], arms sim / gate / gate_shuf / rand / pool at n = 128, 64, 32, 16. Measures per
  arm: L1 of the one step prediction (restore), Spearman rho with the full landscape, cosine of the argmin with the
  reached displacement. Paired over the 100 transitions, bootstrap 95% intervals. GPU 0 (`t1_landscape.py`).

## Decision after the first S2 steps and 26 T1 transitions (2026-10-13 06:15 UTC)
- The unmerged model does not transfer zero shot to this simulation: on T1 its one step prediction is further from
  the next frame than the current frame itself (L1 0.42 against 0.35 for a copy), and the argmin of its landscape has
  cosine 0.17 with the reached displacement (0.93 on the official real example). In S2 the full arm moves away from the
  goal (4 of 4 first episodes end farther than they started). This is the failure reported in vjepa2 issue #110.
- So closed loop merged arms in this simulation would compare noise: S2 is stopped after the 4 full arm jobs (20
  episodes, kept as the record of the baseline failure); merged S2 jobs are cancelled (the two partial merged
  episodes are discarded unread). T1 runs to 100 transitions and is reported as is.
- Bug fixed before the cancelled jobs and T1 restarted: the iterated bipartite matching merged every source in each
  round (r = |A|), so the ranking, and with it the gate, had no effect; now r <= |A| / 2.
- T2 (new, before it ran): the T1 measures on real DROID transitions, which the model was trained on (droid_100,
  shards 0 to 5, exterior camera 1 left, 15 Hz subsampled by 4, centre crop 240 x 180 resized to 256, state =
  cartesian position + gripper position, action = official poses_to_diff). Relevance for the gate = motion (mean
  absolute pixel change of each patch from the previous subsampled frame), shuffled for gate_shuf. Up to 300
  transitions (every 2nd subsampled step); the argmin cosine only on transitions moving >= 2 cm. Plus T2b: CEM with goal =
  the next frame on 60 of these transitions (moving >= 2 cm), arms full / sim / gate / gate_shuf / pool at n = 64 and
  fewer samples (102), cosine of the planned xyz with the true displacement.
- S3 (planned, not started): fine tune the AC predictor on simulated random play so the closed loop works in this
  simulation, then rerun S2 with it. Starts only after T2.
- S3 details (before any S3 run). This fine tunes the predictor for the simulation only; it is a VLA side tool and
  not part of the paper, whose frozen setting is unchanged.
  Data: random play, seeds 1000 to 1799 (disjoint from T1 and S2), 800 episodes x 17 frames (`s3_collect.py`).
  Training: official AC predictor as the start, official losses (teacher forcing L1 + 2 step rollout L1, layer
  normed targets from the frozen encoder), clips of 8 frames, AdamW lr 1e-4 (100 step warmup), weight decay 0.04,
  bf16, 2000 steps of batch 32; seeds 1000 to 1759 train, 1760 to 1799 held out (one step L1 reported against the
  copy baseline). No merging during training.
  Use: S2 rerun with the fine tuned predictor (same 20 episodes, same arms) only if the full arm's held out one step
  L1 beats copy by >= 10% and its T1 style argmin cosine on the held out transitions is >= 0.5; otherwise report S3
  as failed.

## S3 outcome and S2 rerun (2026-10-13 07:52 UTC)
- Training ran as specified (2000 steps, 4 GPUs x batch 8, about 1 h; train loss 0.37 -> 0.17 teacher forcing).
- Held out (seeds 1760 to 1799, 160 transitions, 107 moving >= 2 cm): one step L1 0.213 against copy 0.295 (official
  predictor 0.403); argmin cosine 0.73 (official -0.06). Both gate criteria met, so S2 is rerun with the fine tuned
  predictor as specified (same 20 episodes, same 13 arms, `jobs/s2ft.txt`, results in `results/s2ft`), on GPUs 1 to 5
  (GPU 0 joins after T2b).
- T2b: a second worker taking the same 60 picks from the other end failed at start (GPU still held); T2b continues
  on one GPU, files merged by transition index if a second worker is used.
- T1ft (added before it ran): T1 repeated with the fine tuned predictor (same seeds 100 to 199, same arms and
  measures, `results/t1ft.jsonl`), so the open loop merging measures in simulation come from a model that works
  there. Runs on GPU 0 after T2b. Seeds 100 to 199 are disjoint from the S3 training seeds (1000 to 1759).

## S2ft outcome and S4 (written after S2ft, before any S4 code ran; 2026-10-13 09:45 UTC)
- S2ft (260 episodes): full 90% success, fewer samples 102 100% and 28 75%; every merged arm 0 to 40%, all worse
  than fewer samples at equal predictor tokens (paired intervals exclude 0); gate beats gate_shuf at n = 16 and on
  per step cosine at n = 64; pool beats sim at both n. T1ft: every merged arm's one step L1 (about 0.45) is worse than
  the copy (0.35) although the full arm's is 0.22, so the predictor reads merged tokens off its training distribution.
- S4 asks whether that distribution shift, rather than the information lost by merging, causes the closed loop gap
  (VLA side only, not part of the paper). Merge aware fine tuning: start from the S3 predictor, same play data and
  split (seeds 1000 to 1759 train, 1760 to 1799 held out), clips of 3 frames (the context frame and the 2 rollout
  frames CEM uses); each sample draws a merger uniformly from sim / gate / gate_shuf / rand / pool and n from {64, 16},
  groups from the context frame's tokens (gate relevance = projected gripper as in S2, shuffled for gate_shuf), all 3
  frames reduced with these groups as at planning time; losses = the official pair with the restore readout
  (teacher forcing L1 and 2 step rollout L1 of restored predictions against the layer normed full tokens). AdamW lr
  1e-4, 100 step warmup, wd 0.04, bf16, 2000 steps of 32 samples on 4 GPUs. All five mergers are trained equally so
  the gate vs gate_shuf and pool vs sim contrasts stay fair.
- S4 evaluation, reported whatever the outcome: (a) held out, per merged arm, one step L1 (restore) and argmin cosine
  on the T1 grid, S3 predictor vs S4 predictor; (b) S2m: the 10 merged arms of S2 (n = 64 and 16) on the same 20
  episodes with the S4 predictor, compared with S2ft's fewer samples arms at equal predictor tokens (few102 vs n = 64,
  few28 vs n = 16) and with the same merged arms under the S3 predictor; same paired bootstrap.
- Correction (10:15 UTC, S4 training at step 800, before its evaluation ran): the restore readout cannot recover
  variation inside a group, so a merged arm's one step L1 has a floor, L1(restore(reduce(next frame)), next frame),
  which T0 put at 0.37 to 0.45 for n = 64 to 16. Merged L1 above the copy is therefore not by itself evidence of
  distribution shift. The S4 evaluation now also reports this floor per arm, so the S3 and S4 gaps to it can be read.

## S4 outcome and S5 (2026-10-13 12:10 UTC, before any S5 run)
- S4 held out: argmin cosine of every merged arm rises from 0.02 to 0.20 (S3 predictor) to 0.53 to 0.73 (S4; full arm
  0.73), and the L1 gap above the restore floor roughly halves. S2m (200 episodes): every merged arm 65% to 95% success
  (S2ft 0% to 40%), each paired gain over its S2ft counterpart excludes 0; against fewer samples at equal predictor
  tokens no difference excludes 0 (n = 64 vs few102, n = 16 vs few28); gate vs gate_shuf and pool vs sim no longer
  differ. The closed loop gap of merging was mostly the predictor reading merged tokens off distribution.
- S5 (stacking, S4 predictor): merging and fewer samples together at the predictor cost of n = 16 with 400 samples
  (21.6k tokens per CEM iteration): n = 64 with 102 samples (20.2k) for sim, gate and pool, same 20 episodes,
  `results/s5`. Compared paired with few28 (S2ft, 21.7k) and with the same mergers at n = 16, 400 samples (S2m).
- S5 outcome (12:25 UTC): sim / pool / gate at n = 64 with 102 samples (3.0 to 3.2 s per step): 100% / 90% / 85%
  success, 2.2 / 2.8 / 3.3 cm. Against few28: sim +0.25 [0.10, 0.45] success and -1.7 [-2.8, -0.7] cm; pool and gate
  intervals include 0. Against the same merger at n = 16 with 400 samples: sim +0.20 [0.05, 0.40], pool -0.9 cm
  [-1.7, -0.2], gate includes 0. Which merger stacks best was not a planned contrast.

## Validation of merge aware fine tuning: C1, D1, C2 (2026-10-13 12:58 UTC, before any of it ran)
Claim under test (VLA side only, not part of the paper): after merge aware fine tuning, planning with merged tokens
plus fewer CEM samples beats planning with fewer samples alone at equal predictor cost, and is not worse than 4x
costlier planning. S5 suggested it on 20 episodes with one training run, sim chosen after the fact, and a control
predictor with 2000 fewer training steps. C1 removes these weaknesses; D1 tests it on real images; C2 on a harder task.
Recipes, arms, episodes, hypotheses and analysis are fixed here; nothing below is tuned on test results.

- Equal training control (S3b): the S4 recipe exactly (start from S3, same data and split, clips of 3 frames, the
  official pair of L1 losses, AdamW lr 1e-4, warmup 100, wd 0.04, bf16, 2000 steps of 32 samples, 4 GPUs) on the full
  256 tokens, without merging (`s4_train.py train --plain`, shared code in `ftlib.py`). The S4 predictor and S3b have
  the same training budget, so a difference between them is due to merging during training.
- Training seeds: S4 k0 is the existing S4 predictor; S4 k1, k2 and S3b k0, k1, k2 are new runs that differ only in
  the random seed (sampling of clips, mergers, n). Seeds are not selected; every run is used.
- C1 episodes: 2000 to 2099 (100 new episodes, goals from rng 3000 to 3099, disjoint from the play data seeds 1000 to
  1799 and from every earlier episode). Same goals for every arm; 10 steps; success = final distance < 5 cm.
- C1 arms (about 3 s per step budget unless noted): METHOD = S4 k with sim, n = 64, 102 samples (20.2k predictor
  tokens per CEM iteration); CONTROL = S3b k with 256 tokens, 28 samples (21.7k). Both for k = 0, 1, 2. For k = 0 only:
  S3b with 102 samples (78.9k, the 4x costlier reference), S4 with pool n = 64 and 102 samples, S4 with sim n = 16 and
  400 samples (merging alone, 21.6k), and S3 with 28 samples (the control used in S5).
- C1 hypotheses. H1 (primary): METHOD success minus CONTROL success, each averaged over the 3 seed pairs per episode,
  paired over the 100 episodes, 95% bootstrap interval (10k resamples of episodes) above 0. H1b: same for final
  distance, below 0. H2: METHOD k0 minus S3b k0 with 102 samples, success interval lower bound above -0.10
  (non-inferiority at a quarter of the cost). Per seed results are reported too. Secondary (reported, not claimed
  unless the interval excludes 0): pool vs CONTROL, n = 16 merging alone vs METHOD, S3 vs S3b control.
- D1 data: all 100 droid_100 episodes (the 25 missing shards downloaded from the official bucket,
  gs://gresearch/robotics/droid_100/1.0.0, about 1.8 GB), exterior camera 1 left, 15 Hz subsampled by 4, T2 crop,
  state = cartesian position + gripper, gate relevance = frame difference motion as in T2. Split by episode index e in
  file order: e % 5 == 4 test (20), e % 10 == 3 validation (10), rest train (70).
- D1 predictors, both from the official predictor with the S4 recipe and the same seed: DP plain (full tokens) and DM
  merge aware (five mergers, n in {64, 16}). Checkpoints at 250, 500, 1000, 2000 steps; for each of DP and DM the
  checkpoint (or the official predictor, step 0) with the lowest loss on 300 fixed validation clips (its own objective)
  is used. The test episodes are not touched by this choice.
- D1 test: every subsampled transition of the test episodes moving at least 2 cm, 200 drawn with rng seed 2; goal =
  the next frame; official CEM; measure = cosine of the planned xyz with the true displacement (as T2b). Arms: official
  few28, DP few28, DM sim n = 64 with 102 samples (METHOD), DM pool n = 64 with 102, official sim n = 64 with 102
  (frozen merging), DP few102 (reference), DM sim n = 16 with 400.
- D1 hypotheses. H3 (primary): METHOD minus DP few28 and METHOD minus official few28, both paired 95% intervals
  (10k resamples of transitions) above 0. H4: METHOD minus DP few102 lower bound above -0.10. Secondary: METHOD minus
  official sim n = 64 with 102 (does merge aware training help on real data).
- C2 (harder reach, run after C1 and D1 if GPUs are free): episodes 3000 to 3099, goals 20 to 30 cm away (instead of
  10 to 20), 15 steps; S4 k0 METHOD vs S3b k0 CONTROL; same H1 analysis (single seed pair).
- Verdict rule: the claim counts as supported in simulation if H1 and H2 hold, and on real images if H3 and H4 hold.
  Any other outcome is reported as it is, with the per seed and secondary numbers.
- Jobs: `jobs/v.txt` run by `qworker.sh` (jobs wait for their checkpoint); training chain `tmp/v_train.sh` on GPUs 0
  to 3; results in `results/c1`, `results/c2`, `results/d1`.

## Addendum: H5 and C3 (2026-10-13 14:00 UTC; seen so far: C1 seed pair 0 on 63 of 100 episodes only)
Nothing above changes (H1 to H4, arms, episodes, analysis). Two questions are added, each on its own arms:
- H5 (C1 episodes 2000 to 2099, new arm): is merge aware training what makes merged planning work, at equal training?
  S3b k0 with sim n = 64 and 102 samples (the METHOD's merging, plain fine tuned predictor) vs METHOD k0. H5: METHOD
  k0 minus this arm, success interval above 0.
- C3 (tighter budget, about 10.8k predictor tokens per CEM iteration, half of C1): new episodes 4000 to 4099 (goal
  rng 5000 to 5099), 10 cm to 20 cm goals, 10 steps. METHOD3 = S4 k, sim n = 64, 51 samples (10.1k); CONTROL3 = S3b
  k, 256 tokens, 14 samples (10.8k); for k = 0, 1, 2. Because 14 samples with the official top 10 is a weak CEM, a
  second baseline at the same total planning cost is added for k = 0: S3b k0, 256 tokens, 28 samples, 5 CEM
  iterations instead of 10. Secondary for k = 0: S4 k0 sim n = 16 with 200 samples (10.8k).
  H6 (primary for C3): METHOD3 minus CONTROL3, pooled over the 3 seed pairs, success interval above 0; and METHOD3 k0
  minus the 5 iteration baseline k0, success interval above 0. Same bootstrap as H1.
- Reading: H1 asks whether merging helps at C1's budget; C3 asks the same at a tighter budget, where fewer samples is
  expected to degrade faster. Both are reported whatever they show; neither replaces the other.

## Addendum: C4, budget sweep (2026-10-13 15:02 UTC; seen: C1 seed pair 0 complete, C3 seed pair 0 on 25 to 30 episodes)
- C3's first episodes suggest the gap over fewer samples widens as the budget tightens. C4 tests one tighter budget
  on new episodes, so the speed and success trade off can be read as a curve (C1 about 21k, C3 about 10.8k, C4 about
  5.4k predictor tokens per CEM iteration), fixed before C4 runs.
- Episodes 5000 to 5099 (goal rng 6000 to 6099), 10 to 20 cm goals, 10 steps; seed pair 0 only.
  METHOD4 = S4 k0, sim n = 64, 26 samples (5.1k per iteration); METHOD4b = S4 k0, sim n = 16, 100 samples (5.4k).
  Baselines at no more total planning cost (the official top 10 needs more than 10 samples, so fewer iterations is the
  other way to cut cost): S3b k0 with 14 samples and 5 CEM iterations (54k per plan vs METHOD4's 51k), and S3b k0 with
  28 samples and 2 iterations (43k).
- H7: METHOD4 minus each baseline, success interval above 0 (both required). METHOD4b reported as secondary.

## Deviation: D1 training length (2026-10-13 16:08 UTC, before any D1 result exists)
- Another user's jobs started sharing GPUs 0 to 3 and 5 at about 15:38, and training runs about 2x slower. To keep
  the pre-registered C1 seed pair 2 inside the time window, D1's two predictors are trained for 1000 steps instead of
  2000 (checkpoints 250, 500, 1000; the validation choice among the official predictor and these is unchanged).
  DP is stopped after its 1000 step checkpoint; DM runs 1000 steps. Nothing about D1 has been evaluated yet.

## Addendum: more seed pairs for C3 and C4 (2026-10-13 16:30 UTC; C4 not started, C3 seed pairs 1 and 2 not started)
- To use idle GPU time (two job workers per GPU from now on), appended at the end of the job list: C4 seed pairs 1
  and 2 (METHOD4 with S4 k, both baselines with S3b k) and the C3 5 iteration baseline for S3b k1 and k2. H6 and H7
  are then also reported pooled over the available seed pairs (H6b and H7 pooled the same way as H1); the single seed
  versions stay as defined above.

## D1 outcome (2026-10-13 19:01 UTC, all 200 test transitions)
- Validation choice (by the rule above): DP step 250 (loss 0.666; official 0.747, 500: 0.677, 1000: 0.696), DM step
  500 (0.906; official under merging 0.956, 250: 0.906, 1000: 0.906). 293 validation clips.
- Test, 200 transitions from 19 test episodes, cosine of planned xyz: official few28 0.551, DP few28 0.426, METHOD (DM
  sim n = 64, 102) 0.307, DM pool n = 64 102 0.414, official sim n = 64 102 0.033, DP few102 0.370, DM sim n = 16 400
  0.155.
- H3 not shown: METHOD minus DP few28 -0.119 [-0.225, -0.012]; METHOD minus official few28 -0.244 [-0.351, -0.137].
  H4 not shown (lower bound -0.168). By the verdict rule the claim is not supported on real images in this setting.
- Reading (secondary numbers, not tests): the official predictor was trained on DROID, and fine tuning on 70 episodes
  lowers even plain planning (DP few28 minus official few28 -0.125 [-0.212, -0.041]). Merge aware training recovers
  much of what frozen merging loses (METHOD minus official sim n = 64 +0.274 [+0.157, +0.389]), but not the gap to
  the untouched predictor with fewer samples, which on DROID already matches 400 samples (T2b).
- No D1 follow up runs in this window; C1, C3 and C4 continue unchanged.

## Outcomes so far (2026-10-13 19:35 UTC; C1 seed pair 2 and the C3/C4 addenda still running)
- H2 (final, 100 episodes): METHOD k0 minus S3b k0 with 102 samples, success -0.040 [-0.100, +0.020] (3 episodes
  only METHOD succeeds, 7 only the reference); final distance -0.07 cm [-0.46, +0.40]. The lower bound equals the
  margin, so by the strict rule (above -0.10) H2 is not shown. The verdict rule's simulation claim therefore rests on H1
  alone and is reported as partly supported.
- H5 (final): +0.740 [+0.650, +0.820]. H7 (seed pair 0, final): METHOD4 minus 14 x 5 +0.440 [+0.330, +0.550], minus
  28 x 2 +0.290 [+0.160, +0.410]; holds. H6 seed pairs 0 and 1 pooled: +0.520 [+0.420, +0.620]; H6b k0 +0.370
  [+0.260, +0.470].

## T3: wall clock on unshared GPUs (2026-10-13 22:10 UTC; descriptive only, no hypothesis)
- All C1 to C4 jobs are done; the other user's jobs have left, so planning time can be measured without sharing.
  Plan times in C1 to C4 were taken while GPUs were shared and are not comparable across arms.
- One process per GPU, arms run one after another: the C1, C3 and C4 METHOD arms (S4 k0) and their baselines
  (S3b k0), merging alone (S4 k0, sim n = 16 with 400, 200, 100 samples), S3b k0 with 102 samples, and S3b k0 with
  400 samples. First 3 episodes of the matching C set (2 for 400 samples). Reported: mean CEM time per step. Outputs
  in results/t3 (not used by any test).

## Final outcomes of C1, C2, C3, C4 (2026-10-13 22:12 UTC, all 142 jobs done, none failed)
- H1 (3 seed pairs, 100 episodes): METHOD minus CONTROL success +0.163 [+0.090, +0.240]; per seed pair +0.090
  [0.000, 0.180], +0.250 [0.150, 0.350], +0.150 [0.060, 0.240]. H1b: final distance -1.41 cm [-1.84, -0.98]. Both hold.
- H2: not shown (lower bound -0.100, see above). H5 holds (+0.740 [+0.650, +0.820]).
- Verdict rule: in simulation H1 holds and H2 is not shown, so the claim is partly supported: merging plus fewer
  samples beats fewer samples alone at equal cost, but being no worse than the 4x costlier reference is not
  established at the -0.10 margin (estimate -0.04). On real images (D1) it is not supported.
- H6 (3 seed pairs): +0.510 [+0.407, +0.610]; per seed +0.52, +0.52, +0.49. H6b k0 +0.370 [+0.260, +0.470]; pooled
  over 3 seed pairs +0.390 [+0.293, +0.487]. H6 holds.
- H7 (seed pair 0): +0.440 [+0.330, +0.550] and +0.290 [+0.160, +0.410]; pooled over 3 seed pairs +0.453 [+0.357,
  +0.550] and +0.330 [+0.227, +0.430]. H7 holds.
- C2 (harder reach, 15 steps): success +0.040 [-0.040, +0.120], not shown; final distance -0.68 cm [-1.10, -0.24].
- Secondary, C1: pool n = 64 minus CONTROL k0 +0.120 [+0.040, +0.200]; merging alone (sim n = 16, 400) minus
  METHOD k0 -0.110 [-0.190, -0.030]; S3 minus S3b k0 at 28 samples -0.250 [-0.340, -0.160] (the S5 control was weak).
  C3: sim n = 16 with 200 minus METHOD3 -0.150 [-0.240, -0.060]. C4: sim n = 16 with 100 minus METHOD4 +0.090
  [-0.010, +0.190].

## T3 outcome (2026-10-13 22:29 UTC)
- Mean CEM time per step on an unshared RTX 3090 (3 episodes; 2 for 400 samples): S3b 400 samples 45.3 s, 102
  samples 11.6 s, 28 samples 3.3 s; METHOD (sim n = 64, 102) 3.1 s; sim n = 16 with 400 3.6 s. C3 budget: METHOD3 1.8
  s, 14 samples 1.9 s, 28 x 5 1.7 s, sim n = 16 with 200 1.9 s. C4 budget: METHOD4 1.4 s, 14 x 5 1.0 s, 28 x 2 0.7 s,
  sim n = 16 with 100 1.4 s.
- At the C1 and C3 budgets equal predictor tokens are also about equal wall clock. At the C4 budget they are not:
  METHOD4 takes 1.4 to 2x the time of its baselines (fixed cost of merging, restore and small batches), so H7 holds
  at equal predictor tokens but has not been tested at equal wall clock.

## Addendum: H8, C4 at equal wall clock (2026-10-13 22:31 UTC; seen: all C1 to C4 results and T3)
- T3 shows METHOD4 (1.4 s per step) is slower than the C4 baselines (0.7 and 1.0 s). H8 asks whether H7 survives when
  the baselines get the same wall clock instead of the same predictor tokens.
- New arms on the C4 episodes 5000 to 5099, for k = 0, 1, 2: S3b k with 256 tokens, 28 samples and 4 CEM iterations
  (about 1.36 s by T3 scaling), and with 14 samples and 7 iterations (about 1.33 s). METHOD4 results are the existing
  ones (S4 k, sim n = 64, 26 samples, 10 iterations).
- H8: METHOD4 minus each new baseline, pooled over the 3 seed pairs (as H1), success interval above 0 (both required).
  Per seed numbers reported too. Results in results/c4 (arm names few28c4, few14c7).

## H8 outcome (2026-10-13 23:06 UTC, 24 jobs, none failed)
- Success, S3b k with 28 samples x 4 iterations: 0.53 / 0.41 / 0.42; with 14 x 7: 0.38 / 0.36 / 0.38 (k = 0, 1, 2).
- METHOD4 minus 28 x 4, pooled over 3 seed pairs: +0.307 [+0.213, +0.403] (per seed +0.20, +0.30, +0.42); minus 14 x 7:
  +0.387 [+0.290, +0.483] (per seed +0.35, +0.35, +0.46). H8 holds: at the C4 budget the method also wins at equal wall
  clock on this GPU.
