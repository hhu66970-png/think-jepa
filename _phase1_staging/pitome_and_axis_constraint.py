"""STAGING — PiToMe energy-score selection + axis-constraint (merge_axis) for K-BSM.

This file is NOT a drop-in module. It contains the *exact* methods and config
fields to splice into the live source. Nothing here imports/extends the real
``LocalTokenMerger`` so that ``python3 -m py_compile`` passes even on a box with
no ``torch``. We guard the torch imports in a try/except (see below) purely so
the syntax self-check runs; on the GPU box torch is always importable and the
guard is a no-op.

What it implements (both are NEW, additive extensions of the diagnostic-only
``bsm_ksim_gradual_vec`` path; the existing K-BSM behaviour is byte-unchanged):

  (1) PiToMe energy-score partition  -> new strategy ``bsm_pitome_gradual_vec``
      (PiToMe, NeurIPS'24, arXiv:2405.16148). Instead of ToMe's arbitrary
      even/odd A/B split, an energy score decides who is a *source* (high
      energy = redundant = merged away) vs a *receiver/protected* (low energy =
      isolated = kept). Same size-weighting, same multi-layer rep_for_orig
      carry-forward, same rectangular [B, N-r, D] invariant as K-BSM.

  (2) Axis constraint  -> new config field ``merge_axis in {free, spatial, temporal}``
      Masks the [B, Na, Nb] cosine score matrix BEFORE the per-A argmax so BSM
      merges only within a frame (spatial), only across frames (temporal), or
      freely (default). Works for BOTH ``bsm_ksim_gradual_vec`` and
      ``bsm_pitome_gradual_vec``.

id<->(t,h,w) mapping — VERIFIED against token_merge.py:
  * init_token_merge_state (L222-227): token_ids = arange(num_tokens) at L_start,
    so the original id == flat grid position t*(H*W)+h*W+w.
  * ids_to_coords (L230-236): tokens_per_frame = h_grid*w_grid; t = id // tpf.
  => frame(id) = id // (h_grid * w_grid).  CONFIRMED.
  IMPORTANT: we derive the frame from ``token_ids`` (the ORIGINAL id), NOT from
  the current position, because by layers > L_start the position has been
  compacted but the id still encodes the original (t,h,w). A *merged* token keeps
  the receiver's id, hence the receiver's original frame — a sound, well-defined
  choice for the axis mask (see INTEGRATION doc, "frame of a merged token").

See INTEGRATION_pitome_axis.md for the precise file/method/line splice list.
"""

import math

# torch is mandatory on the GPU box; the guard exists ONLY so this staging file
# compiles for `python3 -m py_compile` on a torch-less laptop. All functions
# below reference torch/F at call time, so a missing torch never affects syntax.
try:  # pragma: no cover - environment shim for py_compile
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


# ==========================================================================
# PART A — config (token_merge.py) ::: NEW DATACLASS FIELDS + normalize/validate
# ==========================================================================
# ---- A.1  Add to `@dataclass MergeConfig` (token_merge.py, after the existing
#           `bsm_match_metric: str = "key"` line ~L41). These four fields are
#           ALSO already referenced by run_token_merge_pca_experiment.py
#           (bsm_partition / pre_merge_ratio / bsm_protect_ratio) but are MISSING
#           from the dataclass + normalizer today, so the harness values are
#           currently dropped. Adding them here fixes that latent bug too.
#
#     # --- Axis constraint for the BSM family (bsm_ksim_gradual_vec and
#     # bsm_pitome_gradual_vec). "free" = no constraint (default, unchanged
#     # behaviour); "spatial" = only merge tokens in the SAME frame; "temporal"
#     # = only merge tokens in DIFFERENT frames. Frame is derived from the
#     # ORIGINAL token id (id // (h_grid*w_grid)). Ignored by every non-BSM
#     # strategy, so A/B/C/B2/C2 are unaffected.
#     merge_axis: str = "free"
#
#     # --- PiToMe energy score (bsm_pitome_gradual_vec only). Margin m in the
#     # energy formula energy_i = mean_j relu(cos(i,j) - m). Larger m => only
#     # strongly-similar neighbours count as redundancy. 0.0 reproduces a plain
#     # mean-cosine energy. Ignored by all other strategies.
#     pitome_margin: float = 0.0
#
#     # --- PiToMe energy computation budget. If the current token count N exceeds
#     # this, energy is computed on a uniformly-subsampled set of anchor columns
#     # of this size (O(N * anchors) instead of O(N^2)); <=0 disables subsampling
#     # (always full O(N^2)). See _bsm_energy_scores for the complexity note.
#     pitome_energy_max_anchors: int = 2048
#
#     # --- (pre-existing harness fields, currently missing from the dataclass)
#     # bsm_partition: A/B split rule for the BSM family. "positional" = ToMe
#     # even/odd over the current order (default, unchanged). "energy" is set
#     # AUTOMATICALLY for bsm_pitome_gradual_vec and need not be passed. ("temporal"
#     # is accepted for forward-compat with the harness's even/odd-by-tubelet idea
#     # but is treated as "positional" here; prefer merge_axis=temporal instead.)
#     bsm_partition: str = "positional"
#     # RLT-style single pre-encoder merge ratio (unused by these two features;
#     # carried so the harness's value is no longer silently dropped).
#     pre_merge_ratio: float = 0.0
#     # Protect this fraction of most-salient A-tokens (feature-norm) from BSM
#     # merging. 0.0 = no protection (default). Used by _forward_bsm protection
#     # block (optional; orthogonal to PiToMe energy protection).
#     bsm_protect_ratio: float = 0.0
#
# ---- A.2  Add to `normalize_merge_config(...)`'s `MergeConfig(...)` kwargs
#           (token_merge.py ~L98-127), reading from the incoming dict:
#
#     merge_axis=str(config.get("merge_axis", "free")),
#     pitome_margin=float(config.get("pitome_margin", 0.0)),
#     pitome_energy_max_anchors=int(config.get("pitome_energy_max_anchors", 2048)),
#     bsm_partition=str(config.get("bsm_partition", "positional")),
#     pre_merge_ratio=float(config.get("pre_merge_ratio", 0.0)),
#     bsm_protect_ratio=float(config.get("bsm_protect_ratio", 0.0)),
#
# ---- A.3  Add to `normalize_merge_config`'s `grid_agnostic_multilayer_strategies`
#           tuple (~L69-71) so the new strategy may also span multiple layers:
#
#     grid_agnostic_multilayer_strategies = (
#         "bsm_ksim_gradual_vec",
#         "bsm_pitome_gradual_vec",   # NEW: same multi-layer allowance as K-BSM
#     )
#
# ---- A.4  Add to `_validate_merge_config(config)` (token_merge.py ~L132):
#
#     if config.merge_axis not in ("free", "spatial", "temporal"):
#         raise ValueError(
#             f"merge_axis must be one of free|spatial|temporal, got {config.merge_axis!r}"
#         )
#     if config.strategy == "bsm_pitome_gradual_vec" and config.pitome_margin < 0.0:
#         raise ValueError("pitome_margin must be >= 0.0")
# ==========================================================================


# ==========================================================================
# PART B — DiagnosticTokenMerger (token_merge_diagnostics.py) ::: registration
# ==========================================================================
# ---- B.1  Extend the two class-level tuples (token_merge_diagnostics.py
#           ~L36-45) to register the new strategy:
#
#     VECTORIZED_STRATEGIES = LocalTokenMerger.VECTORIZED_STRATEGIES + (
#         "local_keep_then_merge_vec",
#         "local_2x2_similarity_gated_importance_vec",
#         "bsm_ksim_gradual_vec",
#         "bsm_pitome_gradual_vec",   # NEW: PiToMe energy-score partition
#     )
#     NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
#         "local_keep_then_merge_vec",
#         "local_2x2_similarity_gated_importance_vec",
#         "bsm_ksim_gradual_vec",
#         "bsm_pitome_gradual_vec",   # NEW: never hit the 2x2 Python fallback
#     )
#
# ---- B.2  Add a branch to `_method_name(self)` (token_merge_diagnostics.py
#           ~L47-54), BEFORE `return super()._method_name()`:
#
#     if self.config.strategy == "bsm_pitome_gradual_vec":
#         return "BSM_pitome_energy"
#
# ---- B.3  forward() dispatch.  bsm_pitome_gradual_vec must reach `_forward_bsm`
#           exactly like bsm_ksim_gradual_vec. The cleanest splice keeps a single
#           dispatch site and lets `_forward_bsm` branch on the strategy name
#           internally (the partition mode is read from self.config inside
#           _forward_bsm, see PART C). In token_merge.py forward() (~L299-303),
#           widen the existing condition:
#
#         if self.config.strategy in ("bsm_ksim_gradual_vec", "bsm_pitome_gradual_vec"):
#             return self._forward_bsm(
#                 x, token_ids, token_size, rep_for_orig,
#                 int(t_grid), int(h_grid), int(w_grid), attn_key,
#             )
#
#           AND in token_merge.py `_can_vectorize_dense_grid` (~L453) widen the
#           defensive early-return likewise:
#
#         if self.config.strategy in ("bsm_ksim_gradual_vec", "bsm_pitome_gradual_vec"):
#             return True, None
# ==========================================================================


# ==========================================================================
# PART C — REPLACE DiagnosticTokenMerger._forward_bsm  (token_merge_diagnostics.py
#          L251-389) with the version below, plus the two NEW helper methods that
#          follow it (_bsm_energy_partition, _bsm_axis_mask).
#
# Diff summary vs the current _forward_bsm:
#   * unchanged: metric selection (key vs feature fallback), size-weighted
#     scatter-add merge, multi-layer rep_for_orig remap, rectangular invariant,
#     the r-cap logic, and the K-BSM (positional) code path.
#   * NEW: a partition step. For bsm_pitome_gradual_vec we build A (sources pool)
#     / B (dst pool) from the ENERGY score instead of even/odd. For
#     bsm_ksim_gradual_vec the even/odd split is preserved EXACTLY.
#   * NEW: after `scores = bmm(a, b^T)` and BEFORE `scores.max(dim=2)` we apply
#     the merge_axis mask, and exclude any A-row that becomes all -inf from top-r.
# ==========================================================================
class _PiToMeAxisForwardBSM:
    """Container for the spliced `_forward_bsm` + helpers.

    These three methods are written to live on ``DiagnosticTokenMerger``; the
    class wrapper here exists only so this staging file is importable/compilable
    in isolation. Copy the method bodies (the `self`-methods) into
    DiagnosticTokenMerger; do NOT copy this wrapper class.
    """

    @torch.no_grad() if torch is not None else (lambda f: f)
    def _forward_bsm(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid, attn_key):
        batch_size, num_tokens, dim = x.shape

        # r = tokens to remove THIS layer. Same per-layer ratio cap (<=25%) as the
        # 2x2 path; also cap at floor((N-1)/2) so even the alternating A/B split
        # always has a partner for every kept A-token and >=1 token survives.
        ratio = max(0.0, min(self.config.merge_ratio, 0.25))
        r = int(math.floor(num_tokens * ratio))
        r = min(r, (num_tokens - 1) // 2)
        if r <= 0 or num_tokens < 2:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        # -- Matching metric (UNCHANGED from K-BSM) --------------------------
        want_key = str(getattr(self.config, "bsm_match_metric", "key")) == "key"
        if (
            want_key
            and attn_key is not None
            and attn_key.dim() == 3
            and attn_key.shape[0] == batch_size
            and attn_key.shape[1] == num_tokens
        ):
            metric_src = attn_key.float()
            match_metric = "key"
            fallback_reason = None
        else:
            metric_src = x.float()
            match_metric = "feature_fallback"
            if not want_key:
                fallback_reason = "feature_metric_requested"
            elif attn_key is None:
                fallback_reason = "attn_key_unavailable"
            else:
                fallback_reason = "attn_key_shape_mismatch"
        metric = F.normalize(metric_src, dim=-1, eps=1e-6)  # [B, N, d]

        # -- Bipartite partition over the CURRENT token order ----------------
        # K-BSM: ToMe even/odd. PiToMe: energy-score split. Both yield
        # 1-D LongTensors a_idx [Na] / b_idx [Nb] of ABSOLUTE positions in the
        # current [0, num_tokens) order, disjoint and covering all tokens. We
        # also surface a per-A "energy" tensor for info/debug (None for K-BSM).
        use_pitome = (self.config.strategy == "bsm_pitome_gradual_vec")
        partition_mode = str(getattr(self.config, "bsm_partition", "positional"))
        if use_pitome:
            a_idx, b_idx, energy_a = self._bsm_energy_partition(metric, num_tokens)
            partition_name = "energy"
        else:
            # positional even/odd (default K-BSM). "temporal" partition from the
            # harness is treated as positional here — use merge_axis="temporal"
            # for cross-frame-only merging instead (cleaner + grid-correct).
            pos = torch.arange(num_tokens, device=x.device)
            a_idx = pos[0::2]
            b_idx = pos[1::2]
            energy_a = None
            partition_name = "positional" if partition_mode != "temporal" else "positional_from_temporal_req"

        na = a_idx.numel()
        nb = b_idx.numel()
        # r cannot exceed #A-tokens (each A contributes <=1 edge).
        r = min(r, na)
        if r <= 0 or nb == 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        # Gather A/B metric rows. a_idx/b_idx are 1-D [Na]/[Nb] for K-BSM
        # (even/odd, shared across the batch) but per-sample [B, Na]/[B, Nb] for
        # PiToMe (energy ranking differs per sample). Handle both shapes inline so
        # this method is correct as-written for either partition.
        if a_idx.dim() == 2:
            a_metric = metric.gather(1, a_idx.unsqueeze(-1).expand(-1, -1, dim))  # [B, Na, d]
            b_metric = metric.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, dim))  # [B, Nb, d]
        else:
            a_metric = metric.index_select(1, a_idx)   # [B, Na, d]
            b_metric = metric.index_select(1, b_idx)   # [B, Nb, d]
        scores = torch.bmm(a_metric, b_metric.transpose(1, 2))   # [B, Na, Nb] cosine

        # -- Axis constraint mask (NEW) --------------------------------------
        # Applied AFTER scores, BEFORE the per-A argmax. Frames come from the
        # ORIGINAL token id (id // (h_grid*w_grid)), verified above.
        axis = str(getattr(self.config, "merge_axis", "free"))
        a_valid_row = None  # [B, Na] bool; an A-row with no allowed B becomes False
        if axis in ("spatial", "temporal"):
            scores, a_valid_row = self._bsm_axis_mask(
                scores, token_ids, a_idx, b_idx, int(h_grid), int(w_grid), axis
            )

        # each A-token -> most similar ALLOWED B-token
        best_sim, best_b_local = scores.max(dim=2)   # [B, Na], [B, Na]

        # If an A-row was fully masked (-inf) by the axis constraint, it has no
        # valid edge: its best_sim is -inf. We must keep it OUT of the top-r so
        # we never "select" a -inf edge. Because best_sim==-inf sorts to the
        # bottom, topk will only pick a masked row if there are fewer than r rows
        # with a finite edge. So we cap r at the per-sample count of valid rows
        # (min over the batch keeps r a single scalar -> rectangular [B,N-r,D]).
        if a_valid_row is not None:
            valid_rows_per_sample = a_valid_row.sum(dim=1)            # [B]
            r_eff = int(min(int(r), int(valid_rows_per_sample.min().item())))
        else:
            r_eff = int(r)
        if r_eff <= 0:
            info = self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )
            info["bsm_partition"] = partition_name
            info["merge_axis"] = axis
            if fallback_reason is not None:
                info["fallback_reason"] = fallback_reason
            return x, token_ids, token_size, rep_for_orig, info
        r = r_eff

        # keep the r strongest A->B edges per sample (top-r over A-tokens). r is a
        # single scalar across the batch -> exactly r sources dropped per sample
        # -> x_new stays a dense [B, N-r, D] (the invariant A/B/C rely on). Every
        # selected edge is finite because r <= #valid rows.
        edge_sim, edge_a_local = best_sim.topk(r, dim=1)        # [B, r]
        edge_b_local = best_b_local.gather(1, edge_a_local)     # [B, r]

        # map A/B local indices back to ABSOLUTE token positions (per-sample or
        # shared, mirroring the a_idx/b_idx shape handled above).
        if a_idx.dim() == 2:
            source_pos = a_idx.gather(1, edge_a_local)              # [B, r]
            receiver_pos = b_idx.gather(1, edge_b_local)            # [B, r]
        else:
            source_pos = a_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_a_local)   # [B, r]
            receiver_pos = b_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_b_local) # [B, r]

        # NOTE on collisions (UNCHANGED): two A-tokens may pick the same B-token.
        # We scatter-add BOTH into that B (size-weighting stays exact). Sources are
        # unique (A disjoint, never overlaps B), so "drop sources, keep rest" stays
        # rectangular with exactly r removed per sample regardless of collisions.

        # -- Size-weighted merge (UNCHANGED) ---------------------------------
        source_ids = token_ids.gather(1, source_pos)            # [B, r]
        receiver_ids = token_ids.gather(1, receiver_pos)        # [B, r]
        source_weight = token_size.gather(1, source_pos)        # [B, r]
        source_x = x.gather(1, source_pos.unsqueeze(-1).expand(-1, -1, dim))  # [B, r, D]
        weighted_src = source_x * source_weight.unsqueeze(-1)   # [B, r, D]

        acc_x = torch.zeros_like(x)
        acc_w = torch.zeros_like(token_size)
        acc_x.scatter_add_(1, receiver_pos.unsqueeze(-1).expand(-1, -1, dim), weighted_src)
        acc_w.scatter_add_(1, receiver_pos, source_weight)
        recv_mask = torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        recv_mask.scatter_(1, receiver_pos, True)
        old_w = token_size                                       # [B, N]
        new_w = old_w + acc_w
        merged_x = (x * old_w.unsqueeze(-1) + acc_x) / new_w.clamp_min(1e-6).unsqueeze(-1)
        x_updated = torch.where(recv_mask.unsqueeze(-1), merged_x, x)
        token_size_updated = torch.where(recv_mask, new_w, old_w)

        # -- Drop sources, keep rest (rectangular, UNCHANGED) ----------------
        keep = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        keep.scatter_(1, source_pos, False)
        num_after = num_tokens - r
        x_new = x_updated[keep].reshape(batch_size, num_after, dim)
        ids_new = token_ids[keep].reshape(batch_size, num_after)
        size_new = token_size_updated[keep].reshape(batch_size, num_after)

        # -- Multi-layer rep_for_orig carry-forward (UNCHANGED) --------------
        num_original_tokens = rep_for_orig.shape[1]
        remap = (
            torch.arange(num_original_tokens, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        remap.scatter_(1, source_ids, receiver_ids)
        rep_new = remap.gather(1, rep_for_orig)

        info = self._info(
            x, x_new, r, r,
            selected_scores=edge_sim,
            implementation="vectorized_bsm",
            num_candidates=int(batch_size * na),
            num_candidate_cells=int(batch_size * na),
            num_accepted=r,
        )
        info["bsm_match_metric"] = match_metric
        info["matching_metric"] = match_metric
        info["bsm_partition"] = partition_name
        info["merge_axis"] = axis
        if energy_a is not None:
            e = energy_a.detach().float()
            info["pitome_energy_mean"] = float(e.mean().item())
            info["pitome_energy_min"] = float(e.min().item())
            info["pitome_energy_max"] = float(e.max().item())
            info["pitome_margin"] = float(getattr(self.config, "pitome_margin", 0.0))
        if fallback_reason is not None:
            info["fallback_reason"] = fallback_reason
        return x_new, ids_new, size_new, rep_new, info

    # ---------------------------------------------------------------------
    # NEW HELPER 1 — PiToMe energy partition.
    # ---------------------------------------------------------------------
    @torch.no_grad() if torch is not None else (lambda f: f)
    def _bsm_energy_partition(self, metric, num_tokens):
        """Split tokens into A (high-energy sources) / B (low-energy receivers).

        metric: [B, N, d] L2-normalized matching vectors (unit rows).

        ENERGY (PiToMe, arXiv:2405.16148):
            energy_i = mean_j  relu( cos(i, j) - m )          (i != j)
        High energy <=> similar to many tokens <=> redundant => prefer SOURCE.
        Low energy  <=> isolated/unique             => protect  => RECEIVER pool.

        We rank tokens by energy and take the top-Na as A (sources pool) and the
        rest as B (dst pool). To keep K-BSM's invariants we use the SAME split
        sizes as the even/odd partition: Na = ceil(N/2), Nb = floor(N/2). This
        guarantees na >= nb >= 1 and r <= na exactly as before, and the existing
        r-cap floor((N-1)/2) <= na holds, so the rectangular [B, N-r, D] invariant
        is preserved.

        Energy is a per-sample [B, N] score; the A/B membership can therefore
        differ across the batch. We DON'T need a shared partition across the
        batch — only a shared scalar r — because the merge gathers by per-sample
        positions and drops exactly r sources per sample. (a_idx/b_idx below are
        per-sample [B, Na]/[B, Nb]; the caller's index_select/gather all accept
        per-sample indices via gather, see note.)

        COMPLEXITY: the full energy needs the N x N cosine gram = O(N^2 d) and
        O(N^2) memory. For V-JEPA2 ViT-L at 256px/64f, N starts at
        t_grid*h_grid*w_grid = 32*16*16 = 8192 and SHRINKS each merge layer, so
        N^2 is ~6.7e7 entries at the first merge layer — feasible but heavy. We
        therefore subsample anchor columns when N > pitome_energy_max_anchors:
        pick `A` uniformly-spaced anchor tokens and approximate
            energy_i ~= mean over anchors relu(cos(i, anchor) - m)
        which is O(N * A) time / O(N * A) memory and an unbiased estimate of the
        full mean (uniform anchors). Set pitome_energy_max_anchors<=0 to force
        full O(N^2). Default 2048 keeps the gram <= N*2048.

        Returns:
            a_idx_pos: [B, Na] LongTensor of ABSOLUTE positions (sources pool)
            b_idx_pos: [B, Nb] LongTensor of ABSOLUTE positions (dst pool)
            energy_a:  [B, Na] energy of the A tokens (for info/debug)
        NOTE — these are PER-SAMPLE [B, Na]/[B, Nb] indices (energy ranking
        differs per sample). The spliced `_forward_bsm` already dispatches on
        `a_idx.dim()`: for the 2-D (PiToMe) case it uses batched `gather` for
        a_metric/b_metric and for source_pos/receiver_pos; for the 1-D (K-BSM
        even/odd) case it uses `index_select`/expand. Nothing else differs.
        """
        batch_size, n, d = metric.shape
        margin = float(getattr(self.config, "pitome_margin", 0.0))
        max_anchors = int(getattr(self.config, "pitome_energy_max_anchors", 2048))

        if max_anchors > 0 and n > max_anchors:
            # uniformly-spaced anchor columns -> [B, A, d]
            anchor_idx = torch.linspace(
                0, n - 1, steps=max_anchors, device=metric.device
            ).round().long()
            anchor = metric.index_select(1, anchor_idx)               # [B, A, d]
            sim = torch.bmm(metric, anchor.transpose(1, 2))           # [B, N, A] cosine
            contrib = torch.relu(sim - margin)
            # subtract a token's self-contribution when it is itself an anchor so
            # the diagonal (cos=1) doesn't bias energy; tokens not in anchors are
            # unaffected. self_in_anchor[b,i] = relu(1 - m) if i is an anchor.
            self_mask = torch.zeros(batch_size, n, device=metric.device, dtype=metric.dtype)
            self_mask.index_fill_(1, anchor_idx, float(max(0.0, 1.0 - margin)))
            energy = (contrib.sum(dim=2) - self_mask) / float(max(1, max_anchors - 1))
        else:
            sim = torch.bmm(metric, metric.transpose(1, 2))           # [B, N, N] cosine
            contrib = torch.relu(sim - margin)
            # zero the diagonal (self cos=1) so it never inflates energy.
            diag = torch.arange(n, device=metric.device)
            contrib[:, diag, diag] = 0.0
            energy = contrib.sum(dim=2) / float(max(1, n - 1))        # [B, N]

        # SAME split sizes as even/odd so all downstream r-caps still hold.
        na = (n + 1) // 2
        # rank by DESCENDING energy: top-na = high energy = sources pool (A);
        # the remainder = low energy = receivers pool (B). stable sort keeps the
        # order deterministic for ties.
        order = torch.argsort(energy, dim=1, descending=True, stable=True)  # [B, N]
        a_idx_pos = order[:, :na].contiguous()                              # [B, Na]
        b_idx_pos = order[:, na:].contiguous()                              # [B, Nb]
        energy_a = energy.gather(1, a_idx_pos)                              # [B, Na]
        return a_idx_pos, b_idx_pos, energy_a

    # ---------------------------------------------------------------------
    # NEW HELPER 2 — axis-constraint mask.
    # ---------------------------------------------------------------------
    @torch.no_grad() if torch is not None else (lambda f: f)
    def _bsm_axis_mask(self, scores, token_ids, a_idx, b_idx, h_grid, w_grid, axis):
        """Mask the [B, Na, Nb] cosine matrix by frame relationship.

        Frame of a token = ORIGINAL id // (h_grid*w_grid)  (VERIFIED mapping).
        token_ids: [B, N] original ids of the CURRENT tokens.
        a_idx/b_idx: either 1-D [Na]/[Nb] (positional/K-BSM) OR per-sample
            [B, Na]/[B, Nb] (PiToMe energy). Handled uniformly below.

        spatial  -> keep only SAME-frame edges  (mask where frame_a != frame_b)
        temporal -> keep only CROSS-frame edges  (mask where frame_a == frame_b)

        Returns (scores_masked, a_valid_row) where a_valid_row [B, Na] is True iff
        that A-row still has >=1 finite (allowed) edge. Rows that are all -inf are
        excluded from top-r by the caller (it caps r at the #valid rows).
        """
        batch_size, na, nb = scores.shape
        tokens_per_frame = int(h_grid * w_grid)

        # frame id per CURRENT token, then gather to A-side / B-side frames.
        frame_all = token_ids // tokens_per_frame                 # [B, N]
        if a_idx.dim() == 1:
            a_idx_b = a_idx.unsqueeze(0).expand(batch_size, -1)   # [B, Na]
            b_idx_b = b_idx.unsqueeze(0).expand(batch_size, -1)   # [B, Nb]
        else:
            a_idx_b = a_idx
            b_idx_b = b_idx
        frame_a = frame_all.gather(1, a_idx_b)                    # [B, Na]
        frame_b = frame_all.gather(1, b_idx_b)                    # [B, Nb]

        same = frame_a.unsqueeze(2) == frame_b.unsqueeze(1)       # [B, Na, Nb] bool
        if axis == "spatial":
            allowed = same          # keep same-frame
        else:                       # "temporal"
            allowed = ~same         # keep cross-frame

        neg_inf = torch.finfo(scores.dtype).min
        scores_masked = scores.masked_fill(~allowed, neg_inf)
        a_valid_row = allowed.any(dim=2)                          # [B, Na]
        return scores_masked, a_valid_row


# --------------------------------------------------------------------------
# Tiny pure-python sanity helpers (no torch) so reviewers can eyeball the math
# and so py_compile exercises real code paths. NOT part of the integration.
# --------------------------------------------------------------------------
def _energy_reference(cos_matrix, margin=0.0):
    """Pure-python reference for energy_i = mean_{j!=i} relu(cos_ij - m).

    cos_matrix: list[list[float]] NxN. Returns list[float] length N.
    """
    n = len(cos_matrix)
    out = []
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if j == i:
                continue
            acc += max(0.0, cos_matrix[i][j] - margin)
        out.append(acc / max(1, n - 1))
    return out


def _split_sizes(n):
    """A/B split sizes matching torch even/odd: na=ceil(n/2), nb=floor(n/2)."""
    na = (n + 1) // 2
    nb = n // 2
    return na, nb


if __name__ == "__main__":  # pragma: no cover - smoke check w/o torch
    cm = [
        [1.0, 0.9, 0.1, 0.0],
        [0.9, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.95],
        [0.0, 0.1, 0.95, 1.0],
    ]
    e = _energy_reference(cm, margin=0.0)
    # token 0 and 1 are a tight pair, 2 and 3 are a tight pair; all four have a
    # close partner so energies are similar — exercises the formula end-to-end.
    print("energy(no margin) =", [round(v, 4) for v in e])
    print("energy(margin .5) =", [round(v, 4) for v in _energy_reference(cm, margin=0.5)])
    print("split_sizes(8192) =", _split_sizes(8192))
    print("split_sizes(7)    =", _split_sizes(7))
