# Integration spec — temporal-audit eval (A1/A2/A3 + A4/A5) for `thinker_train.py`

**Status:** code-only deliverable, no GPU run performed. All line numbers below
refer to the local read-only copy at
`/Users/huhaoming/Desktop/wm/tmp/remote_review/thinker_train.py`
(4040 lines). **Re-confirm the line numbers against the live remote copy before
editing** — the remote file may have drifted. Anchors are quoted as code
snippets so you can `grep` for them regardless of exact line numbers.

The new functions live in
`tmp/phase0/staging/eval_temporal_metrics.py` and are pure torch/numpy with no
side effects. Drop that file next to `thinker_train.py` (or anywhere on
`sys.path`) and `from eval_temporal_metrics import (...)`.

---

## 0. What we are auditing and why these three additions

Current per-epoch eval records exactly two trajectory scalars (see
`logs["epochs"].append({... "val_avg_dist": avg_test_avgdist,
"val_final_dist": avg_test_finaldist ...})`, ~line 3456–3489):

- `val_avg_dist` = **ADE** = `err.mean(dim=(1,2))` batch-mean
- `val_final_dist` = **FDE** = `err[:, -1, :].mean(dim=1)` batch-mean

where `err = torch.linalg.norm(pred_world - xyz_world_slice, dim=-1)` has shape
`[B, future_T, num_joints]` (`compute_trajectory_loss_and_accuracy`, line 148).

Two blind spots this leaves, and the fix:

| ID | Blind spot | New metric |
|----|-----------|-----------|
| A1 | A single ADE scalar hides *where* error accrues. Token-merge may be fine early and **drift late**. | `per_step_displacement` → `[future_T]` error-vs-step curve. |
| A2 | Correct mean position ≠ correct *dynamics*. Merge can smooth/jitter motion while ADE looks OK. | `velocity_accel_error` → 1st/2nd temporal-difference error (overall + per-step + final). |
| A3 | Each config **retrains** its predictor 50 epochs, so the predictor compensates for the merge. That answers "can a fresh head cope?" not "is the encoder rep still usable?". | `eval_frozen` → run the **dense-trained** predictor, frozen, on **merge** features. |

And two horizon/sensitivity controls driven entirely by existing args + a tiny shuffle:

| ID | Question | Change |
|----|----------|--------|
| A4 | Does the gap widen at a **longer horizon**? | `--future_T 64` (+ ensure dataset supplies enough frames). |
| A5 | Is the task even **temporally sensitive** (baseline)? | shuffle past frames before the predictor; a model insensitive to time barely degrades. |

---

## 1. Tensor-shape facts this integration relies on (VERIFY ON GPU)

Read off the source, **not** confirmed on a running GPU:

- `pred_world`, `xyz_world_slice` are both `[B, future_T, num_joints, 3]`.
  - `pred_cam = predict_trajectory_from_latents(cls_model, feats_task_in).view(B, Tpred, J, 3)` (line ~3225–3229).
  - `pred_world = project_camera_points_to_world(pred_cam, cam_ext_slice)` (line ~3246) preserves shape.
  - `xyz_world_slice = xyz_world[:, f0:f1, ...]` (line ~3060), `xyz_world` is `[B, Tall, J, Dj]` (`B, Tall, J, Dj = xyz_world.shape`, line ~3043).
- `Tpred = f1 - f0` (line ~3053); for `trajmode=track` the brief says future_T=32. **VERIFY `Tpred == args.future_T`** on the run (it can be clipped by `split_context_and_future_windows` if the clip is short).
- `num_joints (J)`: comes from the dataset, brief says **52** for both hands. **VERIFY `J == 52`.**
- `coord_dim = 3`, hard-coded in the `.view(B, Tpred, J, 3)`. **VERIFY** no config emits 6-D.

`eval_temporal_metrics.py` accepts either the 4-D `[B,T,J,C]` layout (default,
what the source uses) or a collapsed 3-D `[B,T,J*C]` (pass `coord_dim=`). Use 4-D.

**Consistency guarantee (verified numerically with a numpy reference):**
`per_step_displacement(pred, gt, reduction="mean").mean() == ADE` and
`per_step_displacement(...)[-1] == FDE` for the same tensors. So A1 cannot
contradict the existing scalars; it is a strict refinement. Use this as a unit
assertion the first time you run on GPU (see §6).

---

## 2. A1 + A2 — wire into the live eval loop

### 2a. Import (top of file, near the other `cache_train` imports ~line 49–81)

```python
from eval_temporal_metrics import summarize_temporal_metrics
```

### 2b. Per-batch call — eval loop, immediately AFTER the existing metric call

Anchor (eval loop, ~line 3248):

```python
                loss, avg_dist, final_dist, acc = compute_trajectory_loss_and_accuracy(
                    pred_world, xyz_world_slice, Crit, thr=0.05
                )
                test_loss_sum += float(loss.item())
                ...
                test_count += 1
```

Insert **right after `test_count += 1`** (still inside `with torch.no_grad():`,
still inside the `while True:` batch loop):

```python
                # --- A1/A2 temporal audit (per-batch) ---
                # pred_world / xyz_world_slice are [B, future_T, num_joints, 3].
                # Pass the SAME joint_keep_idx the loss used if you enabled a
                # subset (here None == all joints, matching thr=0.05 call above).
                _tm = summarize_temporal_metrics(
                    pred_world, xyz_world_slice, coord_dim=3, dt=1.0
                )
                # accumulate batch-weighted, exactly like ADE/FDE above
                _T = len(_tm["per_step_displacement"])
                if temporal_step_sum is None and _T > 0:
                    temporal_step_sum = [0.0] * _T
                    temporal_vel_step_sum = [0.0] * max(0, _T - 1)
                    temporal_acc_step_sum = [0.0] * max(0, _T - 2)
                if _T > 0 and len(temporal_step_sum) == _T:
                    for i, v in enumerate(_tm["per_step_displacement"]):
                        temporal_step_sum[i] += v
                    for i, v in enumerate(_tm["velocity_error_per_step"]):
                        temporal_vel_step_sum[i] += v
                    for i, v in enumerate(_tm["accel_error_per_step"]):
                        temporal_acc_step_sum[i] += v
                    temporal_step_count += 1
                temporal_vel_sum += _tm["velocity_error"]
                temporal_vel_final_sum += _tm["velocity_error_final"]
                temporal_acc_sum += _tm["accel_error"]
                temporal_acc_final_sum += _tm["accel_error_final"]
```

### 2c. Initialise the accumulators — next to the other eval sums

Anchor (just before `with torch.no_grad():` of the eval loop, ~line 2953–2956):

```python
        test_loss_sum = test_acc_sum = test_avgdist_sum = test_finaldist_sum = 0.0
        test_lat_metric_sums = initialize_latent_metric_totals()
        test_count = 0
        test_pred_count = 0
```

Add right after:

```python
        # A1/A2 temporal-audit accumulators (per-batch sums; divide by count later)
        temporal_step_sum = None          # list[future_T], lazily sized
        temporal_vel_step_sum = None      # list[future_T-1]
        temporal_acc_step_sum = None      # list[future_T-2]
        temporal_step_count = 0
        temporal_vel_sum = 0.0
        temporal_vel_final_sum = 0.0
        temporal_acc_sum = 0.0
        temporal_acc_final_sum = 0.0
```

### 2d. Reduce + record — where the epoch metrics dict is built

The cleanest place to persist is the per-epoch `logs["epochs"].append({...})`
block (anchor `"val_avg_dist": avg_test_avgdist,` ~line 3484), guarded by
`is_primary_process(rank)`. Compute the means just before it and add keys:

```python
            # --- reduce A1/A2 over the epoch (batch-weighted, like ADE/FDE) ---
            _tc = max(temporal_step_count, 1)
            _bc = max(test_count, 1)
            per_step_disp = (
                [s / _tc for s in temporal_step_sum] if temporal_step_sum else []
            )
            vel_per_step = (
                [s / _tc for s in temporal_vel_step_sum] if temporal_vel_step_sum else []
            )
            acc_per_step = (
                [s / _tc for s in temporal_acc_step_sum] if temporal_acc_step_sum else []
            )
            temporal_block = {
                "per_step_displacement": per_step_disp,        # [future_T]
                "velocity_error": temporal_vel_sum / _bc,
                "velocity_error_final": temporal_vel_final_sum / _bc,
                "velocity_error_per_step": vel_per_step,        # [future_T-1]
                "accel_error": temporal_acc_sum / _bc,
                "accel_error_final": temporal_acc_final_sum / _bc,
                "accel_error_per_step": acc_per_step,           # [future_T-2]
            }
```

Then either:

**(i) inline into the epoch dict** (so it lands in the existing `metrics.json`
written by `write_json_atomic(logs, metrics_json_path)`, line ~3492):

```python
            logs["epochs"].append(
                {
                    "epoch": epoch + 1,
                    ...
                    "val_avg_dist": avg_test_avgdist,
                    "val_final_dist": avg_test_finaldist,
                    "temporal": temporal_block,   # <-- ADD THIS LINE
                    ...
                }
            )
```

`metrics.json` already exists (`metrics_json_path = out_dir / "metrics.json"`,
line ~2195) and is rewritten every epoch — nesting under `"temporal"` is
backward-compatible (old readers ignore the extra key).

**(ii) OR a sidecar `temporal_metrics.json`** if you want the audit decoupled
from the main metrics file (recommended for diffing dense-vs-merge runs):

```python
            try:
                write_json_atomic(
                    {
                        "epoch": epoch + 1,
                        "future_T": int(args.future_T),
                        "num_joints_verified": None,  # fill once confirmed on GPU
                        **temporal_block,
                    },
                    out_dir / "temporal_metrics.json",
                )
            except Exception as ex:
                print(f"[WARN] failed to save temporal_metrics.json: {ex}", flush=True)
```

`write_json_atomic` already exists (line 189). Use **(i)** for minimal surface;
use **(ii)** if you plan to plot many configs' curves side by side.

### 2e. (optional) surface ADE/FDE-by-step in `test_results.md`

`test_results.md` is written once at the end by
`write_markdown_experiment_report` (called ~line 3528; function at line ~1450).
The epoch table is built from `logs["epochs"]`. To add a compact temporal
summary, append a section inside that function after the "## Epoch Table" block
(anchor the `lines.append("## Epoch Table")` at line ~1533). Minimal add — dump
the **last epoch's** curve as a one-line sparkline-ish row:

```python
        last_temporal = epochs[-1].get("temporal") if epochs else None
        if last_temporal:
            lines.append("## Temporal Audit (last epoch)")
            lines.append("")
            psd = last_temporal.get("per_step_displacement", [])
            lines.append(f"- `per_step_displacement` (ADE-by-step, len={len(psd)}): `{psd}`")
            lines.append(f"- `velocity_error`: `{last_temporal.get('velocity_error')}`")
            lines.append(f"- `velocity_error_final`: `{last_temporal.get('velocity_error_final')}`")
            lines.append(f"- `accel_error`: `{last_temporal.get('accel_error')}`")
            lines.append(f"- `accel_error_final`: `{last_temporal.get('accel_error_final')}`")
            lines.append("")
```

(The dense-vs-merge **curve plot** itself is better done offline from the JSON
than drawn in-process; the existing `plot_latent_metric_curves` (line ~291) is a
good template if you want an in-run PNG.)

---

## 3. A3 — `--frozen_predictor` (eval-only, no retrain)

Goal: load a predictor checkpoint **trained on dense features** and evaluate it,
**unchanged**, on **merge** features. This isolates "is the encoder rep itself
trajectory-usable?" from "can a fresh predictor be retrained to cope?".

### 3a. Add CLI args (argparser, near `--predictor`, ~line 3610)

```python
    parser.add_argument(
        "--frozen_predictor",
        action="store_true",
        help="EVAL-ONLY: load a dense-trained predictor (and cls_model) from "
             "--predictor_ckpt, freeze them, and run a single eval pass on the "
             "current (possibly token-merged) features. No training, no optimizer.",
    )
    parser.add_argument(
        "--predictor_ckpt",
        type=str,
        default="",
        help="path to the DENSE-trained checkpoint (ckpt_best.pt / ckpt_latest.pt) "
             "whose 'predictor' (and 'cls_model') weights are loaded for "
             "--frozen_predictor. Must have been trained with the SAME --predictor "
             "type and architecture hyperparameters.",
    )
```

### 3b. Why a dedicated branch (vs. reusing `--resume_ckpt`)

`--resume_ckpt` (`resolve_resume_checkpoint`, line 499; load at line ~2221–2255)
loads `cls_model` strictly and *continues training* from `start_epoch`. The
predictor there is loaded lazily inside the **train** loop at the first batch
(line ~2568–2573) — it never loads if you skip training. For a frozen eval you
must (a) build + load the predictor **before** eval, (b) **not** step any
optimizer/scheduler, (c) run exactly **one** eval pass. Hence a separate branch.

### 3c. Eval-only control flow

The training loop is `for epoch in range(start_epoch, num_epoch):` (line ~2260)
with eval at the bottom (`cls_model.eval()`, line ~2950). The lowest-risk
integration is to **force a single epoch that skips the train body and loads the
frozen weights up front**:

1. **Force one eval-only epoch.** After `num_epoch = int(getattr(args, "epochs", 300))`
   (line ~1980) add:

   ```python
   frozen_eval = bool(getattr(args, "frozen_predictor", False))
   if frozen_eval:
       num_epoch = 1            # single pass
       start_epoch = 0
   ```

   (Place the `start_epoch` override *after* the resume block at line ~2248 sets
   it, or simply gate the resume block with `if not frozen_eval:`.)

2. **Load the frozen checkpoint up front and build the predictor eagerly.** The
   predictor is normally built lazily in the train loop (line ~2428, needs `D`,
   `P`, `total_frames` from `feats_teacher_full`). For frozen eval you need those
   dims *before* training runs. Two options:

   - **Option A (preferred, minimal):** keep the lazy build, but **skip the train
     body** when `frozen_eval` and let the predictor build on the **first eval
     batch** instead. Factor the predictor-construction block (line ~2437–2541,
     the `if args.predictor == "tiny"/"official"/"thinkjepa"` ladder) into a
     helper `build_predictor(args, D, P, total_frames, extras, rank) -> nn.Module`
     and call it once at the top of the eval loop when `predictor is None`. Then
     load weights:

     ```python
     if frozen_eval and predictor is not None and not _frozen_loaded:
         blob = torch_load_checkpoint(args.predictor_ckpt, map_location="cpu")
         if "predictor" not in blob:
             raise KeyError(f"--predictor_ckpt has no 'predictor' state: {args.predictor_ckpt}")
         unwrap_ddp_module(predictor).load_state_dict(blob["predictor"], strict=True)
         # also load the dense-trained readout head so the WHOLE downstream path
         # is the dense one (otherwise a fresh cls_model contaminates the test):
         unwrap_ddp_module(cls_model).load_state_dict(blob["cls_model"], strict=True)
         for p in predictor.parameters():
             p.requires_grad_(False)
         for p in cls_model.parameters():
             p.requires_grad_(False)
         predictor.eval(); cls_model.eval()
         _frozen_loaded = True
     ```

     `torch_load_checkpoint` (line 438), `unwrap_ddp_module` (line 380),
     `save_training_checkpoint` payload keys `"predictor"` / `"cls_model"`
     (line 403–408) are all already there.

   - **Option B (cleanest separation):** add a standalone `run_frozen_eval(args)`
     that builds dataloaders (`build_egodex_dataloaders`, line ~2000), encoder
     (`load_dense_jepa_encoder` **with the merge_config**, line 607 +
     `build_dense_jepa_merge_config`, line 567), builds+loads the predictor, then
     calls `eval_frozen(...)` from `eval_temporal_metrics.py` with an
     `encode_to_pred_fn` closure that reproduces the eval-loop body
     (line ~3000–3246). Dispatch at the top of `main`:
     `if getattr(args, "frozen_predictor", False): return run_frozen_eval(args)`.

3. **Skip the train body.** Guard the training section
   (`cls_model.train()` … through the optimizer steps, line ~2267–2948) with
   `if not frozen_eval:`. The eval block (line ~2950+) then runs once and writes
   metrics including the §2 temporal block — now reflecting the **frozen** path.

4. **Block all weight updates.** When `frozen_eval`: do **not** call
   `optimizer.step()` / `optimizer_pred.step()` / `scheduler.step()`
   (line ~2938–2943, 3517) and do **not** save checkpoints (line ~3395–3453) —
   gate those with `if not frozen_eval:`. We are measuring, not training.

### 3d. Using the library `eval_frozen` (Option B closure)

`eval_frozen(predictor, encode_to_pred_fn, data_iter, coord_dim=3,
joint_keep_idx=None, dt=1.0, max_batches=None)` wants a closure
`encode_to_pred_fn(batch) -> (pred_world, xyz_world_slice)`. Build it by copying
the eval-loop body verbatim:

- features: `out_feats_full = encode_dense_jepa_video(img, model_pt)` (line ~3034)
  **— ensure `model_pt` was built with the merge `merge_config`** so you are
  feeding MERGE features (that is the whole experiment).
- window split: `split_context_and_future_windows` (line 1919);
  `flatten_temporal_patch_tokens` (line 1942); `build_temporal_patch_indices`
  (line 1948).
- predictor forward per type:
  - thinkjepa: `y_future_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)`
    then `.view(B_, T_tgt, P_, D_)` (line ~3190–3192).
  - official: `predictor(x_ctxt, masks_x, masks_y)` (line ~3136).
  - tiny: `predictor(feats_ar_in)` (line ~3086).
- head + frame: `pred_cam = predict_trajectory_from_latents(cls_model,
  feats_task_in).view(B, Tpred, J, 3)` (line ~3225); apply `ref_mode`
  offsets (line ~3231–3244); `pred_world = project_camera_points_to_world(
  pred_cam, cam_ext_slice)` (line ~3246).
- return `(pred_world, xyz_world_slice)`.

`eval_frozen` then returns a dict with `ade`, `fde`, `per_step_displacement`,
`velocity_error[/_final/_per_step]`, `accel_error[/_final/_per_step]`. Dump it
to `out_dir / "frozen_eval_metrics.json"` via `write_json_atomic`.

### 3e. Reading the result

Compare three numbers for the same merge config:
- **dense-retrained ADE** (existing run on dense features),
- **merge-retrained ADE** (existing run on merge features, predictor retrained),
- **frozen ADE** (this branch: dense predictor on merge features).

`frozen_ADE ≫ merge_retrained_ADE ≈ dense_ADE` ⇒ the encoder representation DID
shift; downstream parity was bought purely by retraining (the audit's "yes, time
is being hurt at the representation level, the head just papers over it"). A
small frozen gap ⇒ the merged rep is natively compatible. Read the **per-step**
and **velocity** gaps the same way to localise it to late horizon / dynamics.

---

## 4. A4 — longer horizon (future_T = 64)

No new code. `--future_T` already exists (line 3586, default 32) and flows into
`split_context_and_future_windows(Tall, args.past_T, args.future_T)`
(line ~3046) for both train and eval, and into the predictor build's
`total_frames = (p1 - p0) + (f1 - f0)` (line ~2432) and
`num_frames=total_frames` (line ~2445 / 2515). So:

```
--past_T 32 --future_T 64
```

**Caveats to verify on GPU:**
- `future_T` is clamped by data: `split_context_and_future_windows` does
  `future_T = min(future_T, T - past_T)` (line 1923). So you need
  `Tall >= past_T + 64 = 96` frames per clip *after* `temporal_stride`
  (`stride_time_tensor`, line 1929, subsamples time first, line ~3035–3041).
  **VERIFY the dataset/cache actually delivers ≥96 frames**; otherwise future_T
  silently shrinks and A4 is a no-op. Check the printed `Tpred` / log `future_T`.
- The encoder is built `num_frames=64` (`load_dense_jepa_encoder`, line 614) and
  `vit_large_rope(..., num_frames=64)`. For past+future = 96 frames of *latents*
  the predictor's positional handling must cover 96 — confirm RoPE handles it
  (it should, `use_rope=True`), but **VERIFY no fixed-size pos-embed assert trips
  at 96**.
- Train + eval must use the **same** `future_T` (they read the same arg, so this
  is automatic). The retrained predictor for future_T=64 is a *different* model
  than future_T=32 — do not load a 32-step ckpt into a 64-step build (state_dict
  mask-token / num_frames mismatch). For the A4×A3 cross (frozen at long horizon)
  you must have a *dense* future_T=64 checkpoint.
- Record both horizons' `per_step_displacement`: the **slope** of the curve past
  step 32 is the headline A4 signal (does merge diverge faster at long range?).

---

## 5. A5 — temporal-shuffle control (sensitivity baseline)

Purpose: a sanity baseline. If you randomly **shuffle the past frames'** order
before the predictor and ADE barely changes, the task/model is not using fine
temporal order, so A1/A2 differences would be meaningless. A5 calibrates how much
"temporal information" is even on the table.

### 5a. Minimal arg

```python
    parser.add_argument(
        "--shuffle_past_frames",
        action="store_true",
        help="A5 control: randomly permute the PAST (context) frames along time "
             "before the predictor at EVAL. Sanity baseline for temporal sensitivity.",
    )
```

### 5b. Where to shuffle

Shuffle the **context latents** after slicing, before they are flattened/fed to
the predictor — i.e. permute the time axis of the context window only, never the
future/target. In the eval loop, the context is taken at:

- thinkjepa: `feats_total = feats_eval_full[:, :total_frames, ...]` then
  `x_seq = flatten_temporal_patch_tokens(feats_total)` and the context is gathered
  via `masks_x` from `idx_ctx_1d = build_temporal_patch_indices(P_, 0, ctx_len)`
  (line ~3160–3182).
- official: analogous (line ~3104–3126).

The cleanest, predictor-agnostic hook is to permute the **time slice of
`feats_eval_full` that becomes context** right after `feats_eval_full` is defined
(line ~3055–3059) and before any predictor branch:

```python
                if getattr(args, "shuffle_past_frames", False):
                    # A5: permute ONLY the context frames [p0:p1] along time.
                    ctx = feats_eval_full[:, p0:p1, ...]
                    perm = torch.randperm(ctx.shape[1], device=ctx.device)
                    feats_eval_full = feats_eval_full.clone()
                    feats_eval_full[:, p0:p1, ...] = ctx.index_select(1, perm)
```

**Important correctness notes (VERIFY):**
- Apply to `feats_eval_full` (the predictor input), **not** `feats_gt_full`
  (the latent target) — keep the target unshuffled (line ~3054 keeps
  `feats_gt_full = out_feats_full.contiguous()` separate; do not touch it).
- For `trajmode=track`, double-check `p0:p1` is the true context window and not
  overlapping the future slice `f0:f1` (`split_context_and_future_windows` gives
  disjoint `(0,past_T)` and `(past_T, past_T+future_T)`, so for traj they are
  disjoint — fine).
- Use a **fixed seed** per eval (`configure_reproducibility_seed`, line 85, is
  called once at start) or the A5 number is noisy across batches; consider a
  per-batch `torch.Generator` seeded by batch index for reproducibility.
- A5 is **eval-only**; do not shuffle during training.

### 5c. Reading A5

`ADE(shuffle_past) - ADE(normal)` ≈ 0 ⇒ task barely uses temporal order ⇒
A1/A2 audit has little to detect (report this caveat). Large positive gap ⇒ order
matters ⇒ A1/A2 late-horizon/velocity comparisons are meaningful. Run A5 on the
**dense** config to establish the ceiling of temporal sensitivity, then compare
to the merge configs.

---

## 6. First-run GPU sanity asserts (cheap, do these before trusting numbers)

Put these behind `if args.debug:` (the flag exists, line ~3579) on the first
eval batch:

```python
        assert pred_world.shape == xyz_world_slice.shape, (pred_world.shape, xyz_world_slice.shape)
        assert pred_world.dim() == 4 and pred_world.shape[-1] == 3, pred_world.shape
        # A1 must reproduce the existing scalars on this batch:
        _curve = summarize_temporal_metrics(pred_world, xyz_world_slice, coord_dim=3)["per_step_displacement"]
        import torch as _t
        _err = _t.linalg.norm(pred_world - xyz_world_slice, dim=-1)  # [B,T,J]
        assert abs(sum(_curve) / len(_curve) - float(_err.mean())) < 1e-4   # curve.mean() == ADE
        assert abs(_curve[-1] - float(_err[:, -1, :].mean())) < 1e-4         # curve[-1]  == FDE
        print(f"[A1-CHECK] future_T={len(_curve)} num_joints={pred_world.shape[2]} (expect 52?)")
```

That last print is where you **confirm `num_joints == 52`** and
`future_T == args.future_T` on real data — the two starred shape facts.

---

## 7. Open items to confirm on the GPU box / against the live source

1. **`num_joints == 52`** for `trajmode=track` (read off `J` in the [A1-CHECK] print).
2. **`coord_dim == 3`** always (no 6-D pos+rot variant feeds these metrics).
3. **`Tpred == args.future_T`** (not silently clipped by short clips / stride).
4. **A4:** dataset delivers `>= past_T + future_T` frames after `temporal_stride`; RoPE/predictor handle the longer `total_frames` with no fixed pos-embed assert.
5. **Checkpoint keys:** dense ckpt has both `"predictor"` and `"cls_model"` (it does in `save_training_checkpoint`, line 403–408) and the dense build's predictor **architecture hyperparameters match** the frozen build (depth=12, num_heads=6, embed_dim=D, num_mask_tokens=2 for thinkjepa, `use_vlm_merge`/`vlm_cond_mode`/`vlm_*_dim` — line ~2512–2541). A mismatch → `strict=True` load error.
6. **thinkjepa `ext` guidance:** frozen eval must build `ext` the same way
   (`build_thinkjepa_guidance_inputs`, line 1400; payload via
   `ensure_thinker_guidance_payload`, line ~3063) — and decide whether the frozen
   experiment keeps VLM guidance on (it should, to match dense train).
7. **Averaging convention:** confirm you want **batch-weighted** means (sum per
   batch / batch count), which is what thinker_train uses for ADE/FDE
   (`test_avgdist_sum += float(avg_dist.mean().item()); ... / max(test_count,1)`,
   line ~3253, 3305). The library follows the same convention. If you instead
   want sample-weighted, weight each batch by `B`.
8. **DDP:** the snippets above record on `is_primary_process(rank)` only and do
   not all-reduce the temporal curves. For multi-GPU eval, either run the audit
   on rank 0 over the full `test_loader` (simplest) or all-reduce the per-step
   sums like `distributed_average_from_sum_count` (line 658) does for latents.
