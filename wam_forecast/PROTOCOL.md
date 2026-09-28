# WAM forecasting protocol (CVPR 2027) — frozen before any downstream result

Written 2026-09-25, before G1. Changes after this date are logged at the bottom with a reason.

## Why a new protocol
Code audit of the AAAI pipeline (`cache_train/thinker_train.py`):
- `--trajmode` defaulted to `track` and was never passed, so the task was same-window pose
  regression, not the 32→32 forecasting the paper describes.
- In `traj` mode the encoder still received all 64 frames with full space-time attention, so
  "past" tokens had already seen the future.
- The predictor was `thinkjepa` with an uncompressed Qwen-VL FiLM side channel computed on the
  whole clip (another path around the compressed tokens, and a future leak for forecasting).
- The readout head pools each "frame" of `view(B,64,128,D)`, i.e. half of a tubelet's patches.
- `cudnn.benchmark=True` and no deterministic mode, so fixed-seed reruns differed by ~6.6e-4.

## Task
- Data: official EgoDex part2 subset, 1800 train / 200 test (`train_cache.txt`, `test_cache.txt`).
- Input: video frames 0–31 only → frozen V-JEPA 2 ViT-L (256 px) → 16×16×16 = 4096 tokens.
- Target: 52 hand joints of frames 32–63 in the camera frame of frame 31 (no future camera pose).
- Metrics: ADE / FDE / wrist ADE in mm, error per horizon step; per-clip ADE saved.

## Compression
Frozen encoder, run once per (method, schedule); compressed tokens + representative map are
cached (`extract.py`) and restored to the dense grid at load (identical to `restore_dense`).
Schedules: s15/s20/s25 (layers 12,14,16,18,20; r = 0.15/0.20/0.25), L9 (12–20, r 0.25),
L12 (10–21, r 0.25). Main budget: L9.

## Read-out
`train_probe.py`: cross-attention probe (3 blocks, d = 256, 32 future-frame queries), AdamW
5e-4, wd 0.05, 40 epochs, batch 32, bf16, fully deterministic (`use_deterministic_algorithms`,
explicit-math attention, TF32 off). Same seed list for every arm → paired comparisons.

## Gates
- G0: identical metrics on a same-seed rerun (|ΔADE| < 1e-6 mm).
- G1 (dynamic range): with 5 seeds, dense `vis` must beat `zero` by ≥ 5× the paired seed SD of
  dense-vs-dense reruns across seeds, and `vispose` must beat `pose`. If `vis` does not beat
  `pose`-only, the main read-out becomes `vispose`, and the paper says so.
- G2: primary endpoint below.

## Pre-registered primary endpoint
Test ADE (mm) of WAM-v2 − K-BSM at schedule L9, `vis` input, n seeds chosen from the G0/G1
paired SD so that the detectable effect at 80 % power is ≤ 0.5 mm. Secondary endpoints (Holm):
the same at s25 and L12; WAM-v1 − K-BSM; WAM-v2 − random gate. WAM-v2 is chosen from the
allocation-level ablation (`alloc_metrics.py`, hand-token fidelity on the train split only)
before any test-set downstream run.

## Change log
- 2026-09-25, after G0/G1 (dense, zero, pose, vispose only; NO compressed-arm downstream
  result had been computed). G0 passed (bit-identical rerun). G1 on the pilot (1123 train /
  200 test): vis−zero = −70.0 mm (paired SD 0.92, 76×); vispose−pose = −2.8 mm (SD 1.64).
  vis (76.3 mm) does not beat pose-only (57.0 mm), which by the rule above would make vispose
  the main read-out. Change: the primary read-out stays `vis`, because it is the only setting
  in which the compressed visual tokens are the sole information source (in vispose, vision
  contributes 2.8 mm in total, so any compression effect is bounded by that margin and cannot
  be resolved with a feasible number of seeds). `vispose` is reported for every arm as the
  pre-registered secondary read-out, and the paper states both.
- 2026-09-25, before any downstream compressed-arm run: allocation-level results (train split)
  showed the motion signal does not track hands (WAM-v1 hand fidelity −0.020 vs K-BSM) while
  the observed-past hand prior does (+0.069 with multiplicative gate). The hand-prior arms
  (handmult, handq10, handq20, handsrc) are added to the candidate set for WAM-v2.
- 2026-09-25, probe hyper-parameters chosen on the validation split (10 % of train, dense
  features only, 2–4 seeds): lr 2e-4, 80 epochs (val ADE 68.3 mm vs 70.6 mm for the initial
  5e-4/40 ep; dim 384, depth 2, lr 1e-3/1e-4 and wd 0.1 were not better). Frozen for all arms.
- 2026-09-25, WAM-v2 selected on the validation split (vis, L9, 5 seeds, 180 val clips):
  handmult (observed-past hand prior, multiplicative two-sided gate) had the lowest val ADE
  among gated arms (−1.49 mm vs K-BSM, 95 % t-CI [−4.72, +1.74]). WAM-v1 was worse than K-BSM
  (+3.56 mm, CI [+0.64, +6.47]), as was the random gate (+3.30). Paired seed SD of method
  differences ≈ 1.6–2.6 mm, so the test stage uses 30 seeds per arm for the primary pair
  (detectable effect ≈ 1.3 mm at 80 % power). Test arms: dense, K-BSM, WAM-v2, WAM-v1,
  random gate, handsrc (all L9, vis, seeds 0–29); secondary: s25 and L12 (10 seeds), vispose.
- 2026-09-25, determinism scope: re-encoding the same clips with the same batch composition
  reproduces the cache exactly; a different batch composition changes fp16 GEMM numerics
  slightly (mean cosine 0.9995), which flips some discrete merge decisions. Every arm is
  extracted with the identical sharding (6 shards, batch 8, clip order), so arms remain paired.
- 2026-09-25, POST-HOC method round (after the test results of WAM-v1/v2 were known). Diagnosis:
  every gate that reorders K-BSM edges forces lower-similarity merges (a random gate costs as
  much as the motion gate), so new variants keep K-BSM's similarity order (lambda = 0) and act
  only on the partition or the update: recv (densest hand cells, quota q, can absorb but are
  never removed), excl (the same cells are excluded from matching), anchored averaging
  (merge weights size*(1+beta*relevance)), and combinations; hand relevance is a graded density
  of observed past joints. Selection: the two arms with the lowest validation ADE (5 seeds,
  same seeds and hyper-parameters as the first selection round) go to the test split with 30
  seeds. These results are reported as a post-hoc exploration, separately from the
  pre-registered comparison.
- 2026-09-25, post-hoc round result (test, L9, vis, 30 seeds): mrecv15 (motion relevance,
  similarity order kept, top-15 % motion tokens receiver-only) − K-BSM = −1.41 mm, seed CI
  [−1.86, −0.97], clip×seed CI [−2.75, −0.11], 27/30; hrecv15 − K-BSM = −0.71 mm [−1.25, −0.18],
  clip×seed [−2.30, +0.85].
- 2026-09-25, CONFIRMATION plan for mrecv15, fixed BEFORE running any of it. Each test compares
  with K-BSM at the same budget, same seeds; success = seed-level CI below zero.
  C1 random control rrecv15 (random 15 % receiver-only) at L9, 30 seeds: mrecv15 − rrecv15 < 0.
  C2 s25 (972 tokens), 20 seeds. C3 L12 (131 tokens), 20 seeds. C4 ViT-g gL9, 10 seeds.
  C5 vispose at L9, 10 seeds (secondary). Quota sensitivity (q = 0.10, 0.20) on the VALIDATION
  split only, 5 seeds, reported descriptively. All results are reported whatever their sign.
- 2026-09-25, CONFIRMATION RESULT for mrecv15 (all pre-registered tests): C1 mrecv15 − rrecv15
  = −0.33 mm [−0.77, +0.11] (random receivers alone give −1.08 [−1.57, −0.59] vs K-BSM): FAIL.
  C2 s25: +0.86 [+0.28, +1.44]: FAIL (worse). C3 L12: +1.49 [+0.97, +2.02]: FAIL (worse).
  C4 ViT-g: +0.19 [−0.61, +0.99]: FAIL (null). C5 vispose: −0.02 [−0.64, +0.60]: null.
  Conclusion: the post-hoc L9 gain is a partition effect specific to that budget, not a
  benefit of motion relevance. mrecv15 is NOT adopted as a method.
- 2026-09-25, P4 (learned relevance: scorer on layer-12 tokens trained on train-split
  probe saliency, val Spearman 0.37) on validation: all four arms worse than K-BSM (best
  lrecv30 +0.49 mm); lrecv30 goes to test with 30 seeds per the pre-set rule.
- 2026-09-25, ANCHOR round (content-free partition, prompted by rrecv15 < K-BSM at L9), fixed
  before running: arms rrecv15 (random 15 %), lat4 (every 4th row/column, 6.25 %), lat2 (every
  2nd, 25 %), receiver-only, similarity order kept. Validation (5 seeds) at s25, L9, L12 with
  K-BSM validation runs at s25 and L12 added. The arm with the best mean validation rank across
  the three budgets goes to test with 20 seeds per budget. Success = seed-level CI below zero
  vs K-BSM at ALL three budgets; anything less is reported as a failed generalization.
- 2026-09-25, P4 test (lrecv30, L9, 30 seeds): −0.78 mm, seed CI [−1.26, −0.31], clip×seed
  [−2.19, +0.54], 22/30. Same size as random receivers (−1.08), i.e. the partition effect at L9;
  no evidence that the learned signal adds to it.
- 2026-09-25, ANCHOR round result. Validation: lattice anchors worse than K-BSM (lat2 at L12
  +5.1 mm); random 15 % best on average and sent to test. Test (20 seeds; L9 has 30):
  s25 +0.19 [−0.34, +0.72], L9 −1.08 [−1.57, −0.59], L12 +0.87 [+0.34, +1.41]. FAIL: the
  receiver-only partition helps only at 309 tokens. No generalizable improvement over K-BSM was
  found; exploration stops here and the paper reports all rounds.
- 2026-09-26, code snapshot before new rounds: /21231_data1/huhaoming_wam/snapshots/
  tracked_20260926T0732Z.patch (sha256 e36a79b8...) and wam_forecast_20260926T0732Z.tgz
  (sha256 6c154c5e...). train_probe.py gained --readout and --mix_policy; the restore path keeps
  the published arithmetic and reproduces test_main/kbsm_L9/vis/s0 bit for bit (per-clip ADE
  identical). Note: reordering the position-embedding additions alone changed that run by
  0.45 mm ADE, i.e. seed-level spread comes from chaotic amplification of tiny numerics.
- 2026-09-26, ROUND C (read-out of merged tokens), fixed before running. Arms on the test split,
  seeds 0-19, vis input: kbsm_L9 with readout compact (309 merged tokens read directly, position
  = mean embedding of member cells) and compact_logsize (+ log cluster size key bias, ToMe
  proportional attention); kbsm_s25 and kbsm_L12 with compact_logsize. Dense needs no new run:
  with all sizes 1 the compact path is the restore computation (identical code path).
  Statistic: DiD = [ADE(kbsm,restore) - ADE(dense)] - [ADE(kbsm,new) - ADE(dense)] per seed.
  PRIMARY: L9 compact_logsize, success = seed-level 95 % CI of DiD above 0; the clip x seed
  bootstrap is reported and must also be above 0 for a "robust" claim. Secondary: L9 compact,
  s25 and L12 compact_logsize. Validation runs (5 seeds, L9) are descriptive only.
  Distillation (C2) only if C succeeds.
- 2026-09-26, ROUND D (per-clip adaptive budget), an EXCLUSION test, fixed before running.
  Motion score = mean |gray(t) - gray(t-1)| over observed frames 0..31. Policies (train-split
  quantiles, applied unchanged to test): motion (top 10 % -> 972, bottom 37.25 % -> 131, rest
  309; expected train mean 309), inverse, random (same fractions). Test mean tokens: motion
  324.7, inverse 295.9, random 309.4 (not forced to 309). Arms: mix:motion, mix:inverse,
  mix:random, seeds 0-19, restore readout, compared with kbsm_L9 seeds 0-19. Success (i.e. D
  stays open) = motion - kbsm_L9 clip x seed CI below 0 AND motion - random seed CI below 0.
  Estimated encoder cost reported from bench_vitl.json (batch 8 ms/clip: 972 66.0, 309 61.7,
  131 55.4, dense 84.9).
- 2026-09-26, ROUND C RESULT (test, 20 seeds): DiD (positive = gap closed) L9 compact_logsize
  +0.10 mm, seed CI [-0.30, +0.50], clip x seed [-0.51, +0.72]: FAIL (primary). L9 compact +0.25
  [-0.37, +0.87]; s25 compact_logsize +0.14 [-0.18, +0.46]; L12 compact_logsize -0.06
  [-0.45, +0.33]. Reading the merged tokens directly, with or without proportional attention,
  does not recover the merge loss. Per plan: no distillation round; direction A next.
- 2026-09-26, data facts for A: raw_videos/<task>/<id>.mp4 on HF main are 1920x1080 at 30 fps;
  npz frame_indices index those frames; 50-clip sample: sampling stride median 2 frames, past
  and future windows median 1.63 s and 1.67 s (range 0.7-8.0 s). The 256x256 frames are a
  non-uniform resize of the full 16:9 frame.
- 2026-09-26, direction A, box statistics (all 2000 clips, past joints only): a static
  two-hand union box has median side 1080 px (60 % of clips span the full height); a 128-px crop
  of it has 0.89x the horizontal pixel density of the global 256 view, i.e. no gain. Per-hand,
  per-tubelet tracked boxes have median side 404 px (right) / 339 px (left) -> 2.38x / 2.83x.
  Design: one tracked crop per hand (centre = mean of that hand's visible past joints per
  tubelet, 3-tubelet smoothing, fixed side = 1.3 x median extent in [192, 720] px), 128 px,
  from the raw 1080p frames (frame alignment with the 256 frames verified, corr 0.999-1.000).
- 2026-09-26, ROUND A, fixed before running any A result. Read-out for every A arm: compact
  tokens with continuous coordinate encoding (--coords: Fourier(t,x,y) + log cell w/h + log
  cluster size + branch embedding), vis input, test split, seeds 0-19.
  A0 kbsm_L9 global only (309 tokens) | A1 kbsm_L10g global (~232) + two d256 crops c11 (~43
  each, same pixels as the global view) | A2 kbsm_L10g + two RAW 1080p crops c11 | A3 dense
  global + two raw dense crops (upper reference) | A4 dense global only.
  Regression gate (validation, 5 seeds): A0 must match the published restore read-out,
  |mean(A0 - restore)| < 1 mm, otherwise the coordinate encoding is fixed first.
  PRIMARY: A2 - A0 clip x seed 95 % CI below 0. NECESSARY: A2 - A1 seed CI below 0 (gain comes
  from new pixels, not sampling density). SECONDARY: A2 - A0 with vispose input (10 seeds);
  budget extension (global L12 + crops) only if the primary succeeds. Token totals and measured
  encoder latency (global + two crop passes) are reported for every arm.
- 2026-09-26, ROUND D RESULT (test, 20 seeds, vs fixed kbsm_L9 74.38 mm): motion policy
  (mean 324.7 tokens) +0.66 [+0.07, +1.24], clip x seed [-0.94, +2.09]; inverse (295.9) +0.05
  [-0.51, +0.61]; random (309.4) +1.34 [+0.93, +1.76]; motion - random -0.69 [-1.43, +0.05];
  motion - inverse +0.61 [+0.08, +1.14]. Success criterion not met: D is CLOSED. Allocating
  more tokens to high-motion clips does not help, and mixing budgets across clips costs
  accuracy (random mixing +1.3 mm).
- 2026-09-26, A regression gate FAILED with the first coordinate encoding (Fourier + MLP):
  kbsm_L9 coords - restore on validation = +5.19 mm [+3.47, +6.90], 0/5 (runs kept under
  select/kbsm_L9/vis_coords). Per the gate rule, the encoding was fixed before any test run:
  v2 interpolates the probe's own per-axis grid tables at continuous (t, y, x) and adds scale,
  cluster-size and branch cues through zero-initialised layers, so a global-only, unmerged
  token set is mathematically the published grid embedding. Gate re-run with tag _cv2; all A
  test arms use v2 (tags *_cv2*). Arms, criteria and seeds otherwise unchanged.
- 2026-09-26, A regression gate v2 PASSED: kbsm_L9 cv2 - restore (validation) +0.14 mm
  [-1.29, +1.57]; on test -0.44 [-0.96, +0.08].
- 2026-09-26, ROUND A RESULT (test, 20 seeds; vispose 10): A2 raw crops (232 + 2 x 45 = 322
  tokens) - A0 (309) = -0.66 mm, seed CI [-1.18, -0.14], clip x seed [-1.87, +0.54]: PRIMARY
  FAIL. A2 - A1 (raw vs 256-derived crops) -0.12 [-0.48, +0.25]: NECESSARY condition FAIL (no
  benefit from new pixels). A1 - A0 -0.55 [-1.10, +0.00]. Unmerged crops (2 x 1024) are worse
  than merged ones: +1.32 [+0.65, +1.99]. vispose A2 - A0 -0.04 [-0.43, +0.36]. The crops recover
  ~0.66 of the 5.27 mm compression gap (dense cv2 68.66) at the seed level only. A is closed;
  next per plan: B (learned decoupled matching embedding, DTEM-style).
- 2026-09-26, BASELINE AUDIT: the encoder never used token sizes in attention (blocks are called
  with attn_mask=None; token_size only weights the averaging), although the AAAI text states that
  sizes feed proportional attention. Added optional ToMe proportional attention (prop_attn,
  log-size key bias after the first merge; default off, published configs unchanged).
  Round PA, fixed before running: kbsmpa vs kbsm at s25, L9, L12, test, seeds 0-19, vis restore
  read-out. Reported whatever the sign; if PA helps at all budgets (seed CI below 0), K-BSM+PA
  becomes the reference baseline for B and for the paper.
- 2026-09-26, ROUND B1 (learned decoupled matching embedding by distillation), fixed before any
  downstream B1 result. g = LN + Linear(1024 -> 64) on the dense hidden state after block 12,
  trained on the train split to regress, for sampled pairs (8 feature-nearest + 8 random per
  anchor), the cosine of the pair's DENSE final-layer features. Held-out pair correlation 0.875
  (layer-12 feature cosine: 0.764). The same g replaces the Key cosine at every merge layer
  (limitation: trained on layer-12 states only); partition, order rule and update unchanged.
  Arms: kbsmlm at s25, L9, L12, test, seeds 0-19, restore read-out, vs kbsm (and vs kbsmpa).
  PRIMARY: L9 kbsmlm - kbsm clip x seed CI below 0. METHOD condition: seed CI not above 0 at
  s25 and L12. Full soft-merge DTEM (B2) only if B1 shows a consistent direction.
- 2026-09-26, ROUND PA RESULT (test, 20 seeds): kbsmpa - kbsm = s25 +0.31 [-0.23, +0.85];
  L9 -1.71 [-2.21, -1.22], clip x seed [-3.20, -0.35], 19/20; L12 -1.63 [-2.34, -0.91], clip x
  seed [-3.21, -0.01], 17/20. FDE also lower. Proportional attention closes ~30 % of the L9
  compression gap (5.74 -> 4.03 mm vs dense). K-BSM+PA becomes the reference baseline.
- 2026-09-26, ROUND PA-REL, fixed before running: relevance variants re-tested WITH proportional
  attention, reference kbsmpa at the same budget: wampa, handmultpa, mrecv15pa, rrecv15pa,
  kbsmlmpa (B1 + PA); plus B1 without PA (kbsmlm, vs kbsm). Budgets L9 and L12 with seeds 0-19,
  s25 with seeds 0-9. Primary family: the five PA arms at L9 vs kbsmpa, Bonferroni (5): a
  clip x seed 99 % bootstrap interval below 0. Method condition as before (not worse at the
  other budgets, seed CI). Motivation recorded: WAM enlarges background groups (mean group
  size 82 vs 62), which proportional attention would stop from being under-weighted.
- 2026-09-26, ROUND PA-REL + B1 RESULT (test; L9/L12 20 seeds, s25 10 seeds).
  B1 (learned matching, no PA) - kbsm: s25 +1.31 [+0.31, +2.31]; L9 +0.54 [-0.15, +1.22];
  L12 +0.47 [-0.16, +1.11]: FAIL. B1+PA - kbsmpa: s25 +0.32; L9 +1.81 [+1.31, +2.30], clip x seed
  [+0.15, +3.34]; L12 +3.20 [+2.64, +3.75], clip [+1.51, +4.88]: FAIL (worse). Despite the best
  allocation fidelity (hand +0.040, background +0.031 vs kbsm), learned matching hurts.
  vs kbsmpa (s25 / L9 / L12): wampa -0.13 / +1.83 [+1.36, +2.30] / +3.39 [+2.59, +4.19];
  handmultpa -0.32 [-0.62, -0.02] / +0.31 / -0.89 [-1.51, -0.27]; mrecv15pa -0.78 / -0.43 /
  -0.38; rrecv15pa -0.89 [-1.57, -0.20] / -0.53 [-1.12, +0.06] / -0.92 [-1.34, -0.49].
  All clip x seed intervals cover 0. PRIMARY (Bonferroni over 5 arms at L9, clip x seed 99 %
  CI below 0): no arm passes. WAM's deficit is NOT explained by the missing proportional
  attention. Only random receiver-only anchors + PA are directionally better at all three
  budgets (seed level at s25 and L12); allocation shows they lower the group-size Gini
  (0.628 vs 0.688 for kbsmpa at L9). This is a post-hoc observation, not a confirmed effect.
- 2026-09-26, ROUND B2 (end-to-end DTEM-style matching), fixed before training. Feasibility:
  soft merge over blocks 13-23 from the cached post-block-12 states, batch 4: 0.8 s per
  forward+backward, 7.3 GB. Training: per-layer 64-d matchers (9 layers, L9 schedule), init
  from the B1 matcher, joint with a temporary probe, forecasting MSE, 12 epochs, train split
  minus the fixed 10 % validation clips. Grid (one GPU each): {no PA, PA} x {(tau 0.05, lr_g
  1e-4), (tau 0.1, lr_g 1e-4), (tau 0.05, lr_g 3e-4)}. Selection: lowest validation ADE of the
  soft-merge training model, separately for no-PA and PA. Evaluation: hard merge with the
  selected matchers (layer l uses g_(l-12), clamped to 0..8, for s25/L12), extraction for all
  clips, test seeds 0-19 at L9 and L12, 0-9 at s25; B2 vs kbsm and B2-PA vs kbsmpa.
  PRIMARY: B2-PA - kbsmpa at L9, clip x seed 95 % CI below 0. METHOD condition: seed CI not
  above 0 at s25 and L12.
- 2026-09-27, B2 debugging: first training attempt produced NaN in epoch 1 (fp16 overflow of
  size-weighted sums inside the soft merge under autocast, plus near-empty receivers). Fixed
  before any B2 result: the soft merge runs in fp32 and absorbed tokens (presence < 0.05) cannot
  receive. Grid re-run unchanged.
- 2026-09-27, ROUND B2 RESULT. Selected on validation (soft-merge model): no-PA tau 0.05 lr_g 3e-4
  (val 80.9 mm), PA tau 0.1 lr_g 1e-4 (81.2 mm). Test, hard merge with the learned matchers:
  b2 - kbsm: s25 +1.35 [+0.38, +2.33]; L9 +2.50 [+1.75, +3.25], clip x seed [+0.65, +4.25];
  L12 +0.87 [+0.27, +1.47]. b2pa - kbsmpa: s25 +0.52 [-0.35, +1.39]; L9 +1.85 [+1.26, +2.44],
  clip [+0.38, +3.24]; L12 +4.70 [+4.05, +5.36], clip [+2.79, +6.75]. FAIL at every budget:
  end-to-end learned matching is worse than the post-RoPE Key cosine. The planned sequence
  C -> D -> A -> B is complete; the only robust improvement found is proportional attention.
- 2026-09-27, PA VERIFICATION (code audit, before Phase A). At git HEAD ee09031 the encoder
  block loop calls blk(x, mask=token_ids, attn_mask=None, ...) (vision_transformer.py:306):
  token_size is carried through every merge but only used for the size-weighted feature
  average, never in attention. The AAAI text ("token sizes are passed to proportional attention")
  therefore did NOT describe the code; every run before round PA is "K-BSM without PA".
  Fix (already used since round PA, config prop_attn=True, default False): after the first
  merge each key gets bias log(token_size). Activation checkpointing path now raises if PA is
  requested (it would silently drop the bias). test_prop_attn.py (all pass): T1 a size-2 token
  with log 2 bias equals the duplicated-token sequence (max|diff| 4.9e-4 vs 7.3 without bias);
  T2 token size changes the block output only with PA; T3 prop_attn=False is bit-identical to the
  HEAD encoder, prop_attn=True changes it; T4 checkpointing guard. smoke_exact.py still passes.
  RESULT STATUS: kbsmpa (K-BSM + PA) is the corrected baseline from now on. All earlier
  "vs kbsm" comparisons are internally valid but are comparisons against the no-PA baseline
  and are reported in a separate block, never in the same table column as PA results.
- 2026-09-27, PHASE A PRE-REGISTRATION (balanced group size), fixed before any Phase A result.
  Hypothesis H_A: extreme merge-group-size imbalance harms forecasting; more balanced groups
  (lower Gini) improve ADE at equal token budget. Motivated POST-HOC by rrecv15pa (test).
  Intervention (one factor vs kbsmpa, PA on, K-BSM Key-cosine rule unchanged): capacity-
  constrained edge selection, greedy in K-BSM score order, rejected sources re-propose to their
  best non-full receiver; exactly r tokens removed per layer (same K). Arms:
    capc{1,2,4}pa  <= c sources per receiver per merge layer
    caps{2,4,8}pa  group size after each layer <= k x that layer's mean group size
  Sanity (test_capacity.py, all pass): non-binding cap == kbsmpa bit-exactly; caps respected;
  4-clip Gini at L9: kbsmpa .698, capc4 .681, capc2 .664, caps8 .647, caps4 .545, capc1 .501,
  caps2 .307 (a dose range).
  A1 SELECTION (validation only: fixed 10 % of train, train_probe --val): kbsmpa, the 6 cap arms
  and rrecv15pa at L9 and L12, seeds 0-9. Select c* = lowest val ADE averaged over L9 and L12.
  rrecv15pa on val = independent re-test of the post-hoc observation.
  A2 TEST (run once): c* vs kbsmpa at s25, L9, L12, seeds 0-19. The other cap arms are also run
  on test (seeds 0-9) ONLY for the pre-registered Gini/ADE dose-response analysis, never for
  selection. SUCCESS for H_A as a method: c* - kbsmpa mean < 0 at all three budgets AND clip x
  seed 95 % CI below 0 at >= 2 budgets. STOP RULE: otherwise the branch is closed (no further
  cap variants, no tuning on test).
  A3 CORRELATION: per budget, Spearman(arm Gini on train clips, arm test ADE) over all PA arms
  (kbsmpa, 6 caps, rrecv15pa, mrecv15pa, handmultpa, wampa, kbsmlmpa, b2pa); also on val.
  H_A predicts a positive correlation. Reported with a permutation p-value; n is small, so it
  is descriptive unless it is consistent across budgets and splits.
- 2026-09-27, PHASE B PRE-REGISTRATION (vision dependence), fixed before any Phase B result.
  Question: in which task setting does vision carry meaningful marginal information AND leave
  room between dense and compressed tokens? Selection on VALIDATION only (fixed 10 % of train),
  seeds 0-4; the selected setting is then run once on test.
  Settings: P32 past pose, all 32 frames (current); P4 only the last 4 pose frames (older frames
  replaced by the oldest kept one; train_probe --pose_frames 4); P1 last pose only (no pose
  motion); P0 no pose (vis only; baseline = copy-last). Longer horizons cannot be created
  (EgoDex hdf5 not on this host; clips hold 64 frames), so horizon is studied by the per-step
  curve and by strata of the real forecast duration.
  Arms per setting: pose-only, pose+dense, pose+kbsmpa_L9, pose+kbsmpa_L12 (+ the Phase A
  winner if one survives). Quantities: VG = E(pose-only) - E(pose+dense);
  CG_b = E(pose+kbsmpa_b) - E(pose+dense). Strata (clip_strata.py, thresholds from TRAIN
  quartiles): motion = copy-last ADE (21.2 / 52.8 / 86.9 mm), horizon_s tertiles; per-step ADE.
  HEADROOM CRITERION (fixed now): a setting qualifies if VG >= 10 mm (about 3x the current 3.4)
  and CG_L9 >= 1.5 mm with the seed-level 95 % CI above 0. Among qualifying settings prefer the
  least artificial (P32 > P4 > P1 > P0); strata are diagnostics, not the benchmark definition.
- 2026-09-27, PHASE A1 RESULT (validation, seeds 0-9, all 160 runs OK). d = arm - kbsmpa (mm),
  seed 95 % CI: L9: capc1 -0.62 [-1.93,+0.70]; capc2 -1.04 [-2.04,-0.03]; capc4 -0.39; caps2
  -2.41 [-3.32,-1.49] (10/10 seeds); caps4 -0.85; caps8 -0.98; rrecv15pa -1.47 [-2.74,-0.20].
  L12: capc1 +1.26; capc2 -0.35; capc4 +2.80 [+1.59,+4.00]; caps2 -0.82 [-1.76,+0.12]; caps4
  -0.78; caps8 +1.67 [+0.63,+2.70]; rrecv15pa +0.46. Pre-registered choice c* = caps2pa
  (mean val ADE 76.47 vs kbsmpa 78.08). rrecv15pa independent re-test: replicates at L9 only.
  Next (unchanged plan): A2 test once, caps2pa seeds 0-19 at s25/L9/L12; other caps seeds 0-9
  (correlation only). caps2pa is added to Phase B as the Phase A candidate (chosen on val).
- 2026-09-27, PHASE A2 RESULT (test, run once, seeds 0-19). caps2pa - kbsmpa: L9 -1.38 seed CI
  [-2.04,-0.72] clip x seed [-2.66,-0.09] 18/20; L12 -1.50 [-2.12,-0.88] clip [-3.01,+0.09] 17/20;
  s25 -0.69 [-1.37,-0.004] clip [-1.75,+0.35] 13/20. FDE also lower (L9 69.81 vs 70.11; L12 71.30
  vs 71.76). Pre-registered success (mean < 0 at all three AND clip CI < 0 at >= 2): NOT MET
  (clip CI below 0 at L9 only; L12 misses by +0.09). Mean < 0 and seed CI < 0 at all three.
  Per the stop rule: no further cap variants and no further test runs of caps.
  A3 (test, 13 PA arms): Spearman(Gini, dADE) L9 +0.93 (perm p < 0.001), L12 +0.52 (p 0.08), s25
  +0.62 (p 0.03); max group size similar; hand fidelity does NOT predict dADE (L9 -0.01, s25 +0.05,
  L12 -0.58). Cap arms only: L9 +0.93 (p 0.007), L12 +0.36, s25 +0.39.
- 2026-09-27, A2b REPLICATION (decided AFTER seeing A2; labelled as such; reported whatever the
  outcome). Motivation: the test set has 200 clips, which bounds the clip-level CI. Design:
  5-fold CV over the 1620 train clips outside the selection hold-out (train_probe --cv_fold;
  fold split RandomState(777); each fold trains on all other train clips incl. the hold-out;
  test clips are never used). Only c* = caps2pa vs kbsmpa, budgets L9 / L12 / s25, seeds 0-3 per
  fold, pooled per seed over folds (1620 clips). CONFIRMED if mean < 0 at all three budgets and
  the pooled clip x seed 95 % CI is below 0 at >= 2 budgets; otherwise the balanced-group effect
  is reported as "consistent in sign, not confirmed" and the method claim is dropped.
- 2026-09-27, PHASE B VALIDATION (partial). BUG: --pose_frames 1 sliced past[:, -1:0] (empty) and
  all 30 P1 jobs crashed; fixed (explicit T0 - k indexing, k=4 unchanged, verified) and P1 is
  re-queued after A2b. Split check: train/test share no video (1 clip per video), so the large
  value of long pose history is not a video-level leak. Results (val, seeds 0-4, mm):
  P32: pose-only 54.86, dense 50.05, VG +4.81 [+3.83,+5.78]; CG kbsmpa L9 +0.75 [-0.19,+1.69],
       L12 +1.74 [+1.48,+2.00]; caps2pa L9 +0.58, L12 +0.72.
  P4:  pose-only 73.84 (worse than copy-last 70.68: the pose-only probe overfits), dense 53.89,
       VG +19.95 (+16.8 vs copy-last); CG kbsmpa L9 +0.97 [+0.01,+1.93], L12 +3.18 [+2.13,+4.22];
       caps2pa L9 +0.85, L12 +2.94. High-motion quartile: VG +37.9, CG_L12 +7.2.
  P0:  copy-last 70.68, dense vis 73.63 (VG -2.95: vision alone is worse than copy-last);
       CG kbsmpa L9 +3.46 [+1.84,+5.08], L12 +4.29; caps2pa L9 +1.70, L12 +3.97.
  Headroom criterion (VG >= 10 and CG_L9 >= 1.5 with CI > 0): P32 no, P4 no (CG_L9 0.97), P0 no
  (VG < 0). P1 pending. Observation: CG grows with compression (L12 > L9) and with weaker pose.
- 2026-09-27, A2b RESULT (5-fold CV over 1620 non-selection train clips, seeds 0-3, pooled):
  caps2pa - kbsmpa: s25 -0.97 seed [-1.72,-0.22] clip x seed [-1.71,-0.25] 4/4; L9 -0.51
  [-1.57,+0.55] clip [-1.41,+0.43] 3/4; L12 -1.79 [-2.58,-0.99] clip [-2.75,-0.82] 4/4.
  CONFIRMED by the pre-registered rule (mean < 0 at all three, clip CI < 0 at 2/3). Across the
  three disjoint evaluation sets (val 180, test 200, CV 1620 clips) all 9 budget x set means are
  negative; effect size 0.5-1.8 mm (1-2 %). caps2pa becomes the method candidate.
- 2026-09-27, A4 PRE-REGISTRATION (method development, no tuning; cap value fixed at 2):
  A4a backbone replication: ViT-g/16 (gL9 schedule, 309 tokens) kbsmpa vs caps2pa, test once,
  seeds 0-19, rows_from vitg_kbsm_gL9 (as all earlier ViT-g runs). Replicates if mean < 0 and
  clip x seed CI < 0. Also reported: kbsmpa_gL9 - kbsm_gL9 (PA on ViT-g, seeds 0-9 overlap).
  A4b PA dependence (mechanism): caps2 (cap, no PA) vs kbsm, CV protocol (as A2b), L9 and L12,
  seeds 0-3. Question: does balancing help without PA? (Under PA, a giant group gets a large
  log-size logit and can dominate attention; the cap bounds that.) No success criterion: this
  is descriptive; interaction = (caps2pa - kbsmpa) - (caps2 - kbsm) with a seed-level CI.
- 2026-09-27, PHASE B P1 RESULT (val, seeds 0-4, after the slicing fix): last-pose-only
  pose-only 72.22 (copy-last 70.68), pose+dense 53.49, VG +18.73 [+17.58,+19.88] (+17.2 vs
  copy-last); CG kbsmpa L9 +2.20 [+0.58,+3.82], L12 +4.44 [+3.36,+5.51]; caps2pa L9 +1.77
  [-0.37,+3.92], L12 +4.08. High-motion quartile: VG +51.2, CG L9 +6.0, L12 +10.6.
  HEADROOM CRITERION: only P1 qualifies (VG >= 10, CG_L9 >= 1.5 with CI > 0) -> P1 is selected.
- 2026-09-27, PHASE B TEST PRE-REGISTRATION (P1, run once). Arms: pose-only (k=1), pose+dense,
  pose+kbsmpa and pose+caps2pa at s25/L9/L12 (seeds 0-19), pose+kbsm_L9 (no PA, seeds 0-9),
  dense and pose-only seeds 0-9; copy-last analytic. Setting CONFIRMED if VG (vs the better of
  pose-only and copy-last) >= 10 mm and CG_L9 seed CI > 0. Method in P1: caps2pa - kbsmpa, same
  rule as A2 (mean < 0 at all three budgets AND clip x seed CI < 0 at >= 2). Also reported:
  fraction of the compression gap closed = (CG_kbsmpa - CG_caps2pa) / CG_kbsmpa.
- 2026-09-27, A4 RESULT. A4a ViT-g/16 (gL9, test, seeds 0-19): caps2pa - kbsmpa = -0.04, seed CI
  [-0.58,+0.50], clip x seed [-1.48,+1.37], 13/20: DOES NOT REPLICATE on the second backbone.
  PA itself on ViT-g: kbsmpa - kbsm = -2.10 [-2.90,-1.30], clip [-3.87,-0.46], 10/10.
  A4b (CV, 1620 clips, seeds 0-3): caps2 (cap, no PA) - kbsm: L9 -2.12 clip [-2.93,-1.29] 4/4;
  L12 -1.63 clip [-2.85,-0.51] 4/4. Cell means L9: kbsm 76.47, caps2 74.36, kbsmpa 74.32,
  caps2pa 73.81; L12: 78.99, 77.35, 77.02, 75.23. Interaction (caps2pa-kbsmpa)-(caps2-kbsm):
  L9 +1.61 clip [+0.42,+2.79] (sub-additive: PA effect -2.15 without cap, -0.54 with cap);
  L12 -0.15 [-1.72,+1.43] (additive). Reading: PA and the group-size cap largely correct the
  same failure (a giant merged group is one key and is under-weighted in attention); the cap is
  most useful when PA is absent or at very high compression.
  DECISION (stop rule): the balanced-group method does not give a backbone-robust improvement
  over the corrected baseline -> no further cap variants; caps2pa is reported as a ViT-L result
  and as mechanism evidence, not as the paper's main method.
- 2026-09-27, A3 ROBUSTNESS (descriptive, existing runs; corr_family.py). In the older NO-PA arm
  family (reference kbsm) Gini does NOT predict dADE: val L9 (23 arms, 5 seeds) rho +0.17
  (perm p 0.45); test L9 (9 arms) +0.45 (p 0.24). Hand fidelity does on val: rho -0.62 (p 0.002);
  test -0.43 (p 0.27). In the PA family the reverse holds (Gini +0.93 / fidelity -0.01 at L9 test).
  Conclusion: no single allocation metric predicts forecasting error across method families;
  the strong PA-family Gini correlation is partly the designed cap dose-response. "Gini predicts
  ADE" is stated only conditionally (PA on, similarity-order-preserving arms).
- 2026-09-27, PHASE B TEST RESULT (P1 = video + current hand pose only; run once). copy-last
  68.37, pose-only 69.96, pose+dense 53.20 -> VG +15.2 mm vs copy-last (+16.76 [+16.01,+17.51]
  vs pose-only). CG (kbsmpa - dense): s25 +0.18 [-0.68,+1.04]; L9 +1.70 [+0.57,+2.82] clip
  [+0.18,+3.37]; L12 +3.34 [+2.32,+4.35] clip [+1.64,+5.14]. SETTING CONFIRMED (VG >= 10,
  CG_L9 seed CI > 0). Method in P1, caps2pa - kbsmpa (20 seeds): s25 -0.24 [-0.78,+0.30] clip
  [-1.14,+0.61] 10/20; L9 -1.03 [-1.45,-0.62] clip [-2.02,-0.16] 18/20; L12 -1.17 [-1.58,-0.75]
  clip [-2.51,+0.15] 19/20. Pre-registered rule NOT MET (clip CI < 0 at 1/3 budgets), the
  same pattern as A2. Gap closed by caps2pa: L9 73 % (caps2pa - dense +0.46 n.s.), L12 31 %.
  PA in P1: kbsmpa_L9 - kbsm_L9 = +0.34 [-0.44,+1.12] (no PA benefit in this setting).
- 2026-09-28, ROUND M PRE-REGISTRATION (P1 baselines + decisive method-paper test), fixed
  before any result of this round. Setting P1 (video + current hand pose, --pose_frames 1).
  New baselines: tomepa = ToMe metric (head-mean key BEFORE RoPE, bsm_match_metric=key_prerope)
  + PA; pitomepa = PiToMe (codebase implementation) + PA. Sanity (test_tome.py): same K at all
  budgets, finite, groupings differ from K-BSM (rep agreement 0.4-11 %); default paths still
  bit-exact (test_prop_attn T3 vs ee09031, smoke_exact, test_capacity C3).
  Candidate method BPM = caps2pa (k = 2 fixed; no new variants, no tuning).
  M1 ViT-L P1 test: tomepa, pitomepa at s25/L9/L12, seeds 0-19 (kbsmpa, caps2pa, dense exist).
  M2 ViT-g P1 test (schedules gs25 / gL9 / gL12 = 972 / 309 / ~131 tokens, rows_from
     vitg_kbsm_gL9): kbsmpa and caps2pa seeds 0-19; tomepa, pitomepa seeds 0-9; kbsm_gL9 seeds
     0-9. Dense ViT-g cannot be probed on a 24 GB GPU (23 GB bank) -> no ViT-g CG.
  M3 ViT-L P1 5-fold CV over the 1620 non-selection train clips: caps2pa vs kbsmpa, s25/L9/L12,
     seeds 0-3 (independent of every test result).
  METHOD-PAPER RULE for BPM (all three needed): (i) ViT-L P1: BPM mean below each of kbsmpa,
  tomepa, pitomepa at all three budgets; (ii) ViT-g P1: BPM - kbsmpa mean < 0 at all three
  budgets with seed CI < 0 at >= 2, and BPM mean below tomepa and pitomepa at >= 2 budgets;
  (iii) M3: mean < 0 at all three budgets and clip x seed CI < 0 at >= 2. The already known
  P1 test result (clip CI < 0 vs kbsmpa at 1/3 budgets) stays reported as is.
