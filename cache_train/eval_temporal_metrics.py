# ThinkJEPA downstream temporal-audit metrics (A1/A2/A3)
# -----------------------------------------------------------------------------
# Purpose
#   The existing trajectory eval in `thinker_train.py` only logs two scalars per
#   epoch: `val_avg_dist` (ADE = mean L2 over future steps & joints) and
#   `val_final_dist` (FDE = L2 at the last future step). That is enough to rank
#   configs but blind to *where* a token-merge config hurts: it cannot show late-
#   horizon drift, and it cannot show whether the *dynamics* (velocity /
#   acceleration) are preserved even when mean position looks fine.
#
#   This module adds three import-able, side-effect-free helpers that operate on
#   the SAME (pred, gt) tensors `thinker_train.py` already builds at eval time
#   (`pred_world`, `xyz_world_slice`), so integration is a few lines, no new
#   forward passes:
#     A1  per_step_displacement   -> [future_T]   error-vs-timestep curve
#     A2  velocity_accel_error    -> dict         1st/2nd temporal-difference err
#     A3  eval_frozen             -> dict          frozen-predictor eval skeleton
#
#   Everything is pure torch (numpy only for the optional .tolist()/np path) and
#   has no GPU dependency in its math; the functions run on whatever device the
#   input tensors live on.
#
# -----------------------------------------------------------------------------
# TENSOR-SHAPE CONTRACT  ***VERIFY ON GPU / AGAINST SOURCE BEFORE TRUSTING***
# -----------------------------------------------------------------------------
#   This contract was read off `thinker_train.py` (local copy at
#   tmp/remote_review/thinker_train.py), NOT confirmed on a running GPU. Re-check
#   the two starred facts before relying on any number this produces.
#
#   pred, gt : torch.Tensor of shape [B, future_T, num_joints, coord_dim]
#       B          = batch size
#       future_T   = number of predicted future steps (default 32; A4 raises to 64)
#       num_joints = 52 for trajmode=track (both hands)   *** VERIFY: the head is
#                    built with `.view(B, Tpred, J, 3)` so J comes from the
#                    dataset's xyz_world[...,J,:]; confirm J==52 for your run ***
#       coord_dim  = 3 (xyz)                               *** VERIFY: hard-coded 3
#                    in thinker_train.py `predict_trajectory_from_latents(...)
#                    .view(B, Tpred, J, 3)` ; if a future config emits 6-D
#                    (pos+rot) the L2 over the last dim still works but is no
#                    longer a pure positional error ***
#
#   Reference (ground truth these metrics must stay consistent with), from
#   thinker_train.py::compute_trajectory_loss_and_accuracy:
#       err        = torch.linalg.norm(pred - target, dim=-1)   # [B, T, J]
#       avg_dist   = err.mean(dim=(1, 2))                        # [B]  -> ADE
#       final_dist = err[:, -1, :].mean(dim=1)                   # [B]  -> FDE
#   i.e. the upstream ADE/FDE are *per-sample* then `.mean()`-ed over the batch
#   in the eval loop. A1 below returns the per-step curve whose mean over t
#   equals that same ADE (when each step is weighted equally), so the curve is a
#   strict refinement of the existing scalar -- it will not contradict it.
#
#   A 2-D / collapsed-coord layout [B, future_T, num_joints*coord_dim] is also
#   supported by passing `coord_dim=` explicitly (see `_split_last_coord`). This
#   matches the alternative "[B,32,52*C]" hypothesis in the task brief. Default
#   behaviour assumes the 4-D layout that thinker_train.py actually uses.
# -----------------------------------------------------------------------------

from __future__ import annotations

# torch is the only hard dependency for the math. Guard the import so this file
# stays `py_compile`-clean and import-able for signature inspection on a box
# without torch installed (e.g. the local no-GPU dev machine).
try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only on torch-less machines
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


__all__ = [
    "per_step_displacement",
    "velocity_accel_error",
    "eval_frozen",
    "summarize_temporal_metrics",
]


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "eval_temporal_metrics: torch is not importable in this environment; "
            "these functions are meant to run inside thinker_train.py on the GPU box."
        )


def _check_pair(pred, gt, *, name="pred/gt"):
    """Basic shape/type validation shared by A1/A2.

    Expects 4-D [B, T, J, C] OR 3-D [B, T, J*C] (then call sites pass coord_dim).
    Does NOT move devices or change dtype -- caller controls that (we cast to
    float internally only for the arithmetic to avoid integer/half surprises).
    """
    _require_torch()
    if not torch.is_tensor(pred) or not torch.is_tensor(gt):
        raise TypeError(f"{name} must both be torch.Tensor, got {type(pred)} / {type(gt)}")
    if pred.shape != gt.shape:
        # Mirror thinker_train.py tolerance: it only broadcasts when last dim
        # matches; here we are stricter and fail loudly because a silent
        # broadcast would corrupt a per-step curve.
        raise ValueError(f"{name} shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.dim() not in (3, 4):
        raise ValueError(
            f"{name} must be 3-D [B,T,J*C] or 4-D [B,T,J,C], got {pred.dim()}-D "
            f"{tuple(pred.shape)}"
        )


def _split_last_coord(x, coord_dim):
    """Reshape [B, T, J*C] -> [B, T, J, C] when a collapsed layout is fed.

    If x is already 4-D it is returned unchanged. `coord_dim` (C) must divide the
    last dim. This is the bridge for the alternative [B,32,52*C] hypothesis.
    """
    if x.dim() == 4:
        return x
    B, T, JC = x.shape
    if coord_dim is None or coord_dim <= 0:
        raise ValueError(
            "coord_dim must be a positive int when passing a 3-D [B,T,J*C] tensor; "
            f"got coord_dim={coord_dim} for last dim {JC}"
        )
    if JC % coord_dim != 0:
        raise ValueError(
            f"last dim {JC} is not divisible by coord_dim={coord_dim}; cannot infer num_joints"
        )
    J = JC // coord_dim
    return x.reshape(B, T, J, coord_dim)


def _l2_over_coord(pred, gt):
    """L2 distance over the final (coord) axis -> [B, T, J]. float32 math."""
    diff = (pred.float() - gt.float())
    return torch.linalg.norm(diff, dim=-1)


# ---------------------------------------------------------------------------
# A1: per-step displacement error  (error-vs-timestep curve)
# ---------------------------------------------------------------------------
def per_step_displacement(pred, gt, *, coord_dim=None, joint_keep_idx=None, reduction="mean"):
    """A1 -- per future-step mean L2 displacement error.

    For each future step t, compute the L2 position error per joint, then average
    over (batch, joints) -> one scalar per t. The resulting [future_T] vector is
    the "error vs. time-step" curve used to compare dense vs. token-merge configs:
    a curve that fans out at large t indicates temporal drift (the merge hurt the
    model's ability to roll the latent world-model forward), even if the single
    ADE scalar looks acceptable.

    Consistency guarantee: with reduction="mean", `per_step_displacement(...).mean()`
    equals the upstream ADE (`val_avg_dist`) for the same (pred, gt), because
    upstream ADE = err.mean(dim=(1,2)).mean() = mean over (B,T,J) and this is the
    same mean re-associated as mean_t( mean_{B,J} ). (Exactly equal when T is the
    full future window; equal up to float associativity.)

    Args:
        pred, gt: [B, future_T, num_joints, coord_dim]  (4-D, thinker_train layout)
                  OR [B, future_T, num_joints*coord_dim] (3-D; then pass coord_dim).
                  ***Shapes per the VERIFY block at top of file.***
        coord_dim: int C, required only if a 3-D collapsed tensor is passed.
        joint_keep_idx: optional 1-D LongTensor / list of joint indices to score a
                  subset (e.g. only right-hand joints). Mirrors thinker_train.py's
                  `select_joint_subset_if_needed`. None = all joints.
        reduction: "mean" (default) -> mean over (batch, joints) per step.
                   "none"          -> return the full [B, future_T, num_joints]
                                      per-joint error cube (caller reduces).

    Returns:
        torch.Tensor:
            reduction="mean": shape [future_T], float32, on pred.device.
            reduction="none": shape [B, future_T, num_joints], float32.

    Notes:
        - No gradients are tracked (call under torch.no_grad() in eval anyway).
        - Empty future window (future_T == 0) returns an empty [0] tensor rather
          than raising, so the eval loop can stay branch-free.
    """
    _check_pair(pred, gt, name="per_step_displacement pred/gt")
    pred4 = _split_last_coord(pred, coord_dim)
    gt4 = _split_last_coord(gt, coord_dim)

    if joint_keep_idx is not None:
        if not torch.is_tensor(joint_keep_idx):
            joint_keep_idx = torch.as_tensor(joint_keep_idx, dtype=torch.long, device=pred4.device)
        pred4 = pred4.index_select(dim=2, index=joint_keep_idx.to(pred4.device))
        gt4 = gt4.index_select(dim=2, index=joint_keep_idx.to(gt4.device))

    err = _l2_over_coord(pred4, gt4)  # [B, T, J]
    if reduction == "none":
        return err
    if reduction != "mean":
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    # mean over (batch, joints) -> [T]
    if err.numel() == 0:
        return err.new_zeros((err.shape[1],))
    return err.mean(dim=(0, 2))


# ---------------------------------------------------------------------------
# A2: velocity / acceleration error  (dynamics, not just position)
# ---------------------------------------------------------------------------
def velocity_accel_error(pred, gt, *, coord_dim=None, joint_keep_idx=None, dt=1.0):
    """A2 -- temporal-derivative (velocity & acceleration) error.

    Position being right does NOT mean the dynamics are right: a model can hit the
    correct mean position while smoothing or jittering the motion. Fine temporal
    structure lives in the derivatives. We therefore take finite differences along
    the TIME axis of both pred and gt and measure how far the predicted dynamics
    are from ground truth.

        velocity[t]      = (x[t+1]   - x[t])   / dt           # 1st difference
        acceleration[t]  = (v[t+1]   - v[t])   / dt           # 2nd difference
                         = (x[t+2] - 2 x[t+1] + x[t]) / dt^2

    The error is the L2 norm (over coord_dim) of (pred_derivative - gt_derivative),
    i.e. we compare the *velocity vectors*, not just speed magnitudes. This catches
    direction errors a speed-only metric would miss.

    Args:
        pred, gt: same layout/contract as `per_step_displacement` (4-D preferred).
        coord_dim: required only for the 3-D collapsed layout.
        joint_keep_idx: optional joint subset (see per_step_displacement).
        dt: time step between consecutive frames. Default 1.0 = report errors in
            "per-frame" units (a pure unit choice; relative dense-vs-merge
            comparisons are invariant to dt). ***If you need physical units, set
            dt to the real inter-frame interval AFTER temporal_stride is applied
            (stride changes the effective dt) -- VERIFY the stride on the GPU run.***

    Returns:
        dict with keys (all values are plain python floats or torch.Tensors on
        pred.device; per-step tensors are float32):
            "velocity_error"            : float  -- mean over (B, T-1, J)
            "velocity_error_per_step"   : [T-1]  -- mean over (B, J) per step
            "velocity_error_final"      : float  -- velocity err at the last vel step
            "accel_error"               : float  -- mean over (B, T-2, J)
            "accel_error_per_step"      : [T-2]  -- mean over (B, J) per step
            "accel_error_final"         : float  -- accel err at the last accel step
            "future_T"                  : int    -- T (for sanity/logging)
        If future_T < 2 (no velocity definable) velocity_* are 0.0 / empty and
        accel_* likewise; if future_T < 3 accel_* are 0.0 / empty. This keeps the
        eval loop branch-free for short-horizon smoke runs.

    Implementation detail:
        We diff BEFORE selecting joints? No -- we select joints first (so the diff
        is taken on the kept joints' own trajectories), then diff along time. Order
        matters: time-diff and joint-select commute, but doing select first avoids
        allocating the full cube twice.
    """
    _check_pair(pred, gt, name="velocity_accel_error pred/gt")
    pred4 = _split_last_coord(pred, coord_dim)
    gt4 = _split_last_coord(gt, coord_dim)

    if joint_keep_idx is not None:
        if not torch.is_tensor(joint_keep_idx):
            joint_keep_idx = torch.as_tensor(joint_keep_idx, dtype=torch.long, device=pred4.device)
        pred4 = pred4.index_select(dim=2, index=joint_keep_idx.to(pred4.device))
        gt4 = gt4.index_select(dim=2, index=joint_keep_idx.to(gt4.device))

    pred4 = pred4.float()
    gt4 = gt4.float()
    T = pred4.shape[1]
    dt = float(dt)
    if dt == 0.0:
        raise ValueError("dt must be non-zero")

    def _empty_scalar():
        return pred4.new_zeros(())

    out = {"future_T": int(T)}

    # ---- velocity (1st temporal difference) ----
    if T >= 2:
        # diff along time axis (dim=1): [B, T-1, J, C]
        pred_v = (pred4[:, 1:, ...] - pred4[:, :-1, ...]) / dt
        gt_v = (gt4[:, 1:, ...] - gt4[:, :-1, ...]) / dt
        v_err = torch.linalg.norm(pred_v - gt_v, dim=-1)  # [B, T-1, J]
        v_per_step = v_err.mean(dim=(0, 2))  # [T-1]
        out["velocity_error"] = float(v_err.mean().item())
        out["velocity_error_per_step"] = v_per_step
        out["velocity_error_final"] = float(v_per_step[-1].item())
    else:
        out["velocity_error"] = 0.0
        out["velocity_error_per_step"] = pred4.new_zeros((0,))
        out["velocity_error_final"] = 0.0

    # ---- acceleration (2nd temporal difference) ----
    if T >= 3:
        pred_a = (pred4[:, 2:, ...] - 2.0 * pred4[:, 1:-1, ...] + pred4[:, :-2, ...]) / (dt * dt)
        gt_a = (gt4[:, 2:, ...] - 2.0 * gt4[:, 1:-1, ...] + gt4[:, :-2, ...]) / (dt * dt)
        a_err = torch.linalg.norm(pred_a - gt_a, dim=-1)  # [B, T-2, J]
        a_per_step = a_err.mean(dim=(0, 2))  # [T-2]
        out["accel_error"] = float(a_err.mean().item())
        out["accel_error_per_step"] = a_per_step
        out["accel_error_final"] = float(a_per_step[-1].item())
    else:
        out["accel_error"] = 0.0
        out["accel_error_per_step"] = pred4.new_zeros((0,))
        out["accel_error_final"] = 0.0

    return out


# ---------------------------------------------------------------------------
# A3: frozen-predictor evaluation skeleton
# ---------------------------------------------------------------------------
def eval_frozen(
    predictor,
    encode_to_pred_fn,
    data_iter,
    *,
    coord_dim=3,
    joint_keep_idx=None,
    device=None,
    dt=1.0,
    max_batches=None,
):
    """A3 -- evaluate a predictor trained on DENSE features, *unchanged*, on
    token-MERGE features. No retraining, no optimizer, no grad.

    Why: when each token-merge config retrains its own predictor for 50 epochs,
    the predictor can silently *compensate* for whatever the merge did to the
    encoder representation. That conflates two very different questions:
        (1) "is the merged encoder representation itself still trajectory-usable?"
        (2) "can a fresh predictor be trained to cope with the merged representation?"
    The retrained ADE/FDE only answer (2). To isolate (1), you must take the
    predictor weights learned on dense features and run them, frozen, on merge
    features. A large frozen-eval gap (merge >> dense) with a small retrained gap
    means the encoder DID move the representation, and downstream parity is only
    bought by re-training -- exactly the "is time being hurt" audit we want.

    This function is deliberately a *thin orchestration skeleton*: it does not know
    how thinker_train builds latents (that logic -- masks_x / masks_y / ext, the
    `.view(B, Tpred, J, 3)`, the camera->world projection -- lives in the eval loop
    and differs per predictor type: tiny / official / thinkjepa). You inject that
    via `encode_to_pred_fn`, so the SAME positional/derivative metrics are applied
    no matter which predictor is frozen.

    Args:
        predictor: an already-loaded nn.Module with weights from the DENSE-trained
            checkpoint. ***Caller is responsible for loading the right ckpt and
            for building it with the SAME architecture hyperparameters used at
            dense train time*** (depth/num_heads/embed_dim/use_vlm_merge/... see
            thinker_train.py CortexGuidedVideoPredictor / VisionTransformerPredictor
            construction). A mismatched arch will fail state_dict load upstream,
            not here. This fn only flips it to eval()+no_grad and runs it.
        encode_to_pred_fn: callable(batch) -> (pred, gt) where both are
            [B, future_T, num_joints, coord_dim] in the SAME world/camera frame
            and joint ordering that `compute_trajectory_loss_and_accuracy` uses.
            This closure must reproduce the eval-loop body from
            `out_feats_full` (with MERGE features!) down to `pred_world` /
            `xyz_world_slice`, calling `predictor` internally. See the INTEGRATION
            doc for the exact lines to factor out. May return (None, None) to skip
            a batch (e.g. bad camera geometry); such batches are ignored.
        data_iter: any iterable yielding batches accepted by encode_to_pred_fn
            (typically the existing `test_loader`).
        coord_dim: C, passed through to the metric fns (default 3).
        joint_keep_idx: optional joint subset, passed through.
        device: device for the running accumulators; default = infer from first
            non-None pred, else "cpu".
        dt: passed to velocity_accel_error.
        max_batches: optional cap on number of batches (smoke testing).

    Returns:
        dict of aggregate frozen-eval metrics (batch-count-weighted means, matching
        how thinker_train.py averages: it sums per-batch scalars and divides by the
        number of batches, NOT by sample count -- see test_*_sum / test_count). Keys:
            "ade", "fde",                      # parity with val_avg_dist/val_final_dist
            "per_step_displacement"  : [future_T]  list of floats
            "velocity_error", "velocity_error_final",
            "accel_error",    "accel_error_final",
            "velocity_error_per_step" : [future_T-1] list,
            "accel_error_per_step"    : [future_T-2] list,
            "num_batches", "num_samples",
            "future_T",
        All curve values are python lists (json-serializable). If no batch yielded
        a valid (pred, gt), returns {"num_batches": 0, ...} with empty curves so
        the caller can detect the no-op case.

    NOTE: This is an EVAL-ONLY routine. It calls `predictor.eval()` and wraps the
    whole loop in torch.no_grad(). It never touches an optimizer or scheduler and
    never calls .backward(). That is the entire point of "frozen".
    """
    _require_torch()

    predictor.eval()

    # running sums over batches (batch-weighted to match thinker_train averaging)
    n_batches = 0
    n_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0
    vel_sum = 0.0
    vel_final_sum = 0.0
    acc_sum = 0.0
    acc_final_sum = 0.0
    # per-step curve accumulators (lazily sized once future_T is known)
    step_sum = None          # [future_T]
    step_batches = 0
    vel_step_sum = None      # [future_T-1]
    acc_step_sum = None      # [future_T-2]
    seen_future_T = None

    with torch.no_grad():
        for bi, batch in enumerate(data_iter):
            if max_batches is not None and bi >= int(max_batches):
                break
            try:
                pred, gt = encode_to_pred_fn(batch)
            except Exception as ex:  # be robust like the real eval loop
                # The real loop re-raises fatal camera-geometry errors; here we
                # leave that policy to encode_to_pred_fn and just skip on failure.
                # Surface the error type for debuggability without aborting.
                print(f"[eval_frozen][batch {bi}] skipped: {type(ex).__name__}: {ex}")
                continue
            if pred is None or gt is None:
                continue

            if device is None:
                device = pred.device

            # ---- positional ADE / FDE (mirror compute_trajectory_loss_and_accuracy) ----
            pred4 = _split_last_coord(pred, coord_dim)
            gt4 = _split_last_coord(gt, coord_dim)
            if joint_keep_idx is not None:
                jk = (
                    joint_keep_idx
                    if torch.is_tensor(joint_keep_idx)
                    else torch.as_tensor(joint_keep_idx, dtype=torch.long)
                )
                pred4 = pred4.index_select(2, jk.to(pred4.device))
                gt4 = gt4.index_select(2, jk.to(gt4.device))

            err = _l2_over_coord(pred4, gt4)  # [B, T, J]
            ade_sum += float(err.mean(dim=(1, 2)).mean().item())     # == val_avg_dist
            fde_sum += float(err[:, -1, :].mean(dim=1).mean().item())  # == val_final_dist

            # ---- A1 curve ----
            curve = per_step_displacement(pred4, gt4, reduction="mean")  # [T]
            T = int(curve.shape[0])
            if seen_future_T is None:
                seen_future_T = T
                step_sum = curve.new_zeros((T,))
            if T != seen_future_T:
                # ragged future_T across batches should not happen for a fixed
                # config; warn and skip the curve accumulation rather than crash.
                print(
                    f"[eval_frozen][batch {bi}] future_T changed "
                    f"{seen_future_T}->{T}; skipping curve accumulation"
                )
            else:
                step_sum = step_sum + curve
                step_batches += 1

            # ---- A2 derivatives ----
            dyn = velocity_accel_error(pred4, gt4, dt=dt)
            vel_sum += dyn["velocity_error"]
            vel_final_sum += dyn["velocity_error_final"]
            acc_sum += dyn["accel_error"]
            acc_final_sum += dyn["accel_error_final"]
            v_ps = dyn["velocity_error_per_step"]
            a_ps = dyn["accel_error_per_step"]
            if v_ps.numel() > 0:
                if vel_step_sum is None:
                    vel_step_sum = v_ps.clone()
                elif vel_step_sum.shape == v_ps.shape:
                    vel_step_sum = vel_step_sum + v_ps
            if a_ps.numel() > 0:
                if acc_step_sum is None:
                    acc_step_sum = a_ps.clone()
                elif acc_step_sum.shape == a_ps.shape:
                    acc_step_sum = acc_step_sum + a_ps

            n_batches += 1
            n_samples += int(pred4.shape[0])

    if n_batches == 0:
        return {
            "ade": None,
            "fde": None,
            "per_step_displacement": [],
            "velocity_error": None,
            "velocity_error_final": None,
            "velocity_error_per_step": [],
            "accel_error": None,
            "accel_error_final": None,
            "accel_error_per_step": [],
            "num_batches": 0,
            "num_samples": 0,
            "future_T": 0,
        }

    def _curve_to_list(s, denom):
        if s is None or denom <= 0:
            return []
        return (s / float(denom)).detach().cpu().tolist()

    return {
        "ade": ade_sum / n_batches,
        "fde": fde_sum / n_batches,
        "per_step_displacement": _curve_to_list(step_sum, step_batches),
        "velocity_error": vel_sum / n_batches,
        "velocity_error_final": vel_final_sum / n_batches,
        "velocity_error_per_step": _curve_to_list(vel_step_sum, n_batches),
        "accel_error": acc_sum / n_batches,
        "accel_error_final": acc_final_sum / n_batches,
        "accel_error_per_step": _curve_to_list(acc_step_sum, n_batches),
        "num_batches": int(n_batches),
        "num_samples": int(n_samples),
        "future_T": int(seen_future_T or 0),
    }


# ---------------------------------------------------------------------------
# convenience: bundle A1 + A2 for the live eval loop (per-batch accumulation)
# ---------------------------------------------------------------------------
def summarize_temporal_metrics(pred, gt, *, coord_dim=None, joint_keep_idx=None, dt=1.0):
    """One-call wrapper computing A1 + A2 for a single (pred, gt) batch.

    Intended to be called once per eval batch in thinker_train.py right after the
    existing `compute_trajectory_loss_and_accuracy(...)` call, with `pred_world`
    and `xyz_world_slice`. Returns json-friendly python objects (lists/floats) so
    the caller can sum curves across batches and dump to temporal_metrics.json.

    Returns dict:
        "per_step_displacement"   : [future_T] list of floats   (A1)
        "velocity_error"          : float                       (A2)
        "velocity_error_final"    : float
        "velocity_error_per_step" : [future_T-1] list
        "accel_error"             : float
        "accel_error_final"       : float
        "accel_error_per_step"    : [future_T-2] list
        "future_T"                : int

    These are PER-BATCH values; aggregate them across batches the same way
    thinker_train aggregates ADE/FDE (sum per-batch, divide by batch count).
    """
    _require_torch()
    curve = per_step_displacement(
        pred, gt, coord_dim=coord_dim, joint_keep_idx=joint_keep_idx, reduction="mean"
    )
    dyn = velocity_accel_error(
        pred, gt, coord_dim=coord_dim, joint_keep_idx=joint_keep_idx, dt=dt
    )
    return {
        "per_step_displacement": curve.detach().cpu().tolist(),
        "velocity_error": dyn["velocity_error"],
        "velocity_error_final": dyn["velocity_error_final"],
        "velocity_error_per_step": dyn["velocity_error_per_step"].detach().cpu().tolist(),
        "accel_error": dyn["accel_error"],
        "accel_error_final": dyn["accel_error_final"],
        "accel_error_per_step": dyn["accel_error_per_step"].detach().cpu().tolist(),
        "future_T": dyn["future_T"],
    }
