"""Diagnostic / research-only token merger: home of the Gradual K-BSM strategy.

``DiagnosticTokenMerger`` subclasses ``LocalTokenMerger`` and adds exactly one
extra strategy: ``bsm_ksim_gradual_vec`` (global Bipartite Soft Matching on
post-RoPE attention-Key cosine, gradual multi-layer, size-weighted; the verified
encoder-speedup winner). A/B/C behaviour is inherited unchanged from the base.

Kept out of the main path: the encoder (``vision_transformer.py``) /
``scripts/train.sh`` never instantiate this subclass unless
``bsm_ksim_gradual_vec`` is explicitly requested.

History: the No-Go strategies B2 (``local_keep_then_merge_vec``) and
C2 (``local_2x2_similarity_gated_importance_vec``), plus the dead-end research
knobs (temporal partition, RLT pre-merge, norm protection), were removed
2026-05-31 after experiments confirmed they offered no Pareto improvement.
"""

import math

import torch
import torch.nn.functional as F

from src.models.utils.token_merge import LocalTokenMerger, compute_importance


class DiagnosticTokenMerger(LocalTokenMerger):
    """``LocalTokenMerger`` extended with the Gradual K-BSM strategy.

    A/B/C behaviour is inherited unchanged from the base class; this subclass
    adds only ``bsm_ksim_gradual_vec`` (the verified encoder-speedup winner).
    """

    VECTORIZED_STRATEGIES = LocalTokenMerger.VECTORIZED_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # Gradual K-BSM (grid-agnostic, multi-layer)
        "bsm_pitome_gradual_vec",  # NEW: PiToMe energy-score partition
        "bsm_taware_gradual_vec",  # WAM: K-BSM + task-relevance (motion) gate
    )
    NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # must never hit the 2x2 Python fallback path
        "bsm_pitome_gradual_vec",  # NEW: never hit the 2x2 Python fallback
        "bsm_taware_gradual_vec",  # WAM: never hit the 2x2 Python fallback
    )

    def _method_name(self):
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return "BSM_ksim_gradual"
        if self.config.strategy == "bsm_pitome_gradual_vec":
            return "BSM_pitome_energy"
        if self.config.strategy == "bsm_taware_gradual_vec":
            return "BSM_taware_motion"
        return super()._method_name()

    # Gradual K-BSM: global bipartite soft matching (training-free, SDPA-safe).
    # Grid-agnostic; runs on compressed token sets across MULTIPLE layers. NEVER
    # routes through the dense 2x2 _forward_vectorized path (no _reshape_2x2_cells,
    # no compute_importance, no flat_cell_positions), so compressed inputs at
    # layers > L_start are handled natively. Reached only from
    # LocalTokenMerger.forward's bsm dispatch (subclass-only strategy).
    # ------------------------------------------------------------------
    @torch.no_grad()
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
        # Neutralize any non-finite metric row at the source so neither the
        # cosine score bmm nor the PiToMe energy bmm can produce nan/inf; a
        # genuine zero row then normalizes to a finite ~0 vector (eps guard).
        # No-op for the finite metrics K-BSM always sees -> K-BSM unchanged.
        metric_src = torch.nan_to_num(metric_src, nan=0.0, posinf=0.0, neginf=0.0)
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
        # Bugfix: gather along the METRIC's own feature width (post-RoPE key
        # head_dim, e.g. 64) -- NOT x's hidden dim (1024). They are equal only on
        # the feature_fallback path (metric == x); with the key metric they differ,
        # so expanding to `dim` mis-sized the gather index and raised
        # "Size does not match at dimension 2". K-BSM took the index_select branch
        # and never hit this, which is why it surfaced only once PiToMe used key.
        d_metric = metric.shape[-1]
        if a_idx.dim() == 2:
            a_metric = metric.gather(1, a_idx.unsqueeze(-1).expand(-1, -1, d_metric))  # [B, Na, d]
            b_metric = metric.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, d_metric))  # [B, Nb, d]
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

        # -- WAM task-relevance gate (NEW; bsm_taware_gradual_vec only) -------
        # Down-weight the merge PRIORITY of any edge touching a high-relevance
        # (e.g. high-motion / task-relevant) token, so such tokens are dropped
        # LAST. Partner CHOICE (best_b_local) is UNCHANGED — only which edges the
        # top-r selects changes. relevance_lambda=0 / strategy!=taware => identical
        # to K-BSM (rank_score == best_sim).
        rank_score = best_sim
        rel_source_used = None
        if self.config.strategy == "bsm_taware_gradual_vec":
            lam = float(getattr(self.config, "relevance_lambda", 1.0))
            if lam > 0.0:
                rel = self._wam_relevance(x, token_ids, rep_for_orig,
                                          int(t_grid), int(h_grid), int(w_grid))  # [B,N] in [0,1]
                rel_source_used = str(getattr(self.config, "relevance_source", "motion") or "motion")
                pw = float(getattr(self.config, "relevance_power", 1.0))
                gate = (1.0 - lam * rel).clamp(0.0, 1.0).pow(pw)        # [B, N]
                if a_idx.dim() == 2:
                    gate_a = gate.gather(1, a_idx)                      # [B, Na]
                    gate_b_all = gate.gather(1, b_idx)                  # [B, Nb]
                else:
                    gate_a = gate.index_select(1, a_idx)               # [B, Na]
                    gate_b_all = gate.index_select(1, b_idx)           # [B, Nb]
                gate_b_chosen = gate_b_all.gather(1, best_b_local)    # [B, Na]
                gated = best_sim * gate_a * gate_b_chosen
                # keep axis-masked (-inf) rows at -inf (avoid -inf*0=nan promotion)
                rank_score = torch.where(torch.isfinite(best_sim), gated, best_sim)

        # keep the r strongest A->B edges per sample (top-r by rank_score). r is a
        # single scalar across the batch -> exactly r sources dropped per sample
        # -> x_new stays a dense [B, N-r, D] (the invariant A/B/C rely on). Every
        # selected edge is finite because r <= #valid rows.
        _, edge_a_local = rank_score.topk(r, dim=1)             # [B, r]
        edge_b_local = best_b_local.gather(1, edge_a_local)     # [B, r]
        edge_sim = best_sim.gather(1, edge_a_local)             # [B, r] ACTUAL cosine

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

        # -- Size-weighted merge (NUMERICALLY HARDENED) ----------------------
        # Bugfix (PiToMe NaN): the energy partition can funnel many redundant
        # sources into a FEW low-energy receivers, so a receiver token_size can
        # reach ~1e3 (vs ~4e2 for K-BSM even/odd). Under fp16 AMP (downstream
        # training) the weighted sum x*size then exceeds the fp16 max (65504) and
        # overflows to +/-inf -> inf/inf = nan, which poisons the encoder output
        # and the downstream loss (epoch1 non-finite). We therefore accumulate the
        # weighted average in FP32 and only cast back to x.dtype at the end. This
        # is a NO-OP for the fp32 encoder harness (x already fp32) -> K-BSM encoder
        # numbers are byte-identical; it only removes the fp16 overflow under AMP.
        x_f = x.float()
        size_f = token_size.float()
        source_ids = token_ids.gather(1, source_pos)            # [B, r]
        receiver_ids = token_ids.gather(1, receiver_pos)        # [B, r]
        source_weight = size_f.gather(1, source_pos)            # [B, r] fp32
        source_x = x_f.gather(1, source_pos.unsqueeze(-1).expand(-1, -1, dim))  # [B, r, D] fp32
        weighted_src = source_x * source_weight.unsqueeze(-1)   # [B, r, D] fp32

        acc_x = torch.zeros_like(x_f)
        acc_w = torch.zeros_like(size_f)
        acc_x.scatter_add_(1, receiver_pos.unsqueeze(-1).expand(-1, -1, dim), weighted_src)
        acc_w.scatter_add_(1, receiver_pos, source_weight)
        recv_mask = torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        recv_mask.scatter_(1, receiver_pos, True)
        old_w = size_f                                           # [B, N] fp32
        new_w = old_w + acc_w
        merged_x = (x_f * old_w.unsqueeze(-1) + acc_x) / new_w.clamp_min(1e-6).unsqueeze(-1)
        # guard against any residual non-finite (e.g. a non-finite incoming x row)
        merged_x = torch.nan_to_num(merged_x, nan=0.0, posinf=0.0, neginf=0.0)
        x_updated = torch.where(recv_mask.unsqueeze(-1), merged_x.to(x.dtype), x)
        token_size_updated = torch.where(recv_mask, new_w.to(token_size.dtype), token_size)

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
        if rel_source_used is not None:
            info["relevance_source"] = rel_source_used
            info["relevance_lambda"] = float(getattr(self.config, "relevance_lambda", 1.0))
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
    @torch.no_grad()
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
        # Sanitize before ranking: a non-finite incoming key/feature row would make
        # cos -> nan -> energy nan and torch.argsort with nan is UNDEFINED (garbage
        # permutation, could even drop/duplicate positions). Map non-finite energy
        # to a very low finite value so such tokens sort to the receiver (B) pool
        # and the partition stays a valid permutation. Finite energies are
        # untouched, so K-BSM is unaffected and normal PiToMe runs identically.
        energy = torch.nan_to_num(energy, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
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
    @torch.no_grad()
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

    # ---------------------------------------------------------------------
    # NEW HELPER 3 — WAM per-token task relevance (motion saliency).
    # ---------------------------------------------------------------------
    @torch.no_grad()
    @torch.no_grad()
    def _load_prior_relevance(self, source, batch_size, num_tokens, t_grid, h_grid, w_grid, device):
        """NEW(V2): load a precomputed per-token relevance PRIOR from config.relevance_path.

        Returns [batch_size, num_tokens] float on `device`, or None to signal
        "no prior -> fall back to the in-place motion signal (V1)". The prior is a
        CLIP-AGNOSTIC normalized importance map (e.g. offline predictor saliency or
        hand-joint density), shape [t*h*w] or [h*w] (tiled over time). Only consulted
        at the first merge layer. Robust: any missing/unreadable/mismatched file
        returns None (=> motion fallback), so this can NEVER crash a run.
        """
        if source not in ("predictor_saliency", "handjoint"):
            return None
        path = str(getattr(self.config, "relevance_path", "") or "")
        if not path:
            return None
        import os as _os
        import numpy as _np
        cache = getattr(self, "_wam_prior_cache", None)
        if cache is None or cache.get("path") != path:
            if not _os.path.exists(path):
                return None
            try:
                arr = _np.load(path)
                rel = arr["rel"] if (hasattr(arr, "files") and "rel" in arr.files) else arr
                rel = _np.asarray(rel, dtype="float32").reshape(-1)
            except Exception:
                return None
            cache = {"path": path, "rel": rel}
            self._wam_prior_cache = cache
        rel = cache["rel"]
        n_thw, n_hw = int(t_grid * h_grid * w_grid), int(h_grid * w_grid)
        if rel.shape[0] == n_thw:
            vec = rel
        elif rel.shape[0] == n_hw:                      # spatial prior -> tile over time
            vec = _np.tile(rel, int(t_grid))
        else:
            return None                                 # shape mismatch -> motion fallback
        if int(vec.shape[0]) != int(num_tokens):
            return None
        t = torch.from_numpy(_np.ascontiguousarray(vec)).to(device=device, dtype=torch.float32)
        return t.unsqueeze(0).expand(int(batch_size), -1).contiguous()   # [B, t*h*w]

    def _wam_relevance(self, x, token_ids, rep_for_orig, t_grid, h_grid, w_grid):
        """Per-CURRENT-token task relevance in [0,1] (1 = protect hardest).

        Computed ONCE at the first merge layer (where N == num_original == t*h*w)
        as adjacent-frame feature change (motion); cached per ORIGINAL token id.
        At later (already-compressed) layers we GATHER the cached relevance by the
        current tokens' original ids (token_ids), since N != t*h*w there and the
        grid reshape in compute_importance would be invalid. Each layer is then
        per-sample min-max normalized to [0,1] (max -> 1) so the gate is consistent.

        relevance_source="motion" => compute_importance(..., "motion"); other
        sources fall through to the same motion default for now (V1).
        """
        batch_size, num_tokens, _ = x.shape
        num_original = int(rep_for_orig.shape[1])
        source = str(getattr(self.config, "relevance_source", "motion") or "motion")
        if source == "none":
            source = "motion"

        def _unit01(rel):
            rel = rel.float().clamp_min(0.0)
            rel = rel / rel.amax(dim=1, keepdim=True).clamp_min(1e-6)
            return rel

        # First merge layer: dense grid -> compute & cache (token_ids == arange,
        # so the per-current-token vector IS indexed by original id).
        if num_tokens == num_original and num_tokens == int(t_grid * h_grid * w_grid):
            # NEW(V2): if a precomputed per-token prior is configured for this source
            # (predictor_saliency / handjoint + relevance_path), use it; otherwise
            # fall back to the in-place adjacent-frame motion signal (V1). Everything
            # downstream (cache, _unit01, gate, top-r, merge) is IDENTICAL either way,
            # so this is a pure signal-source swap (clean V1-vs-V2 ablation).
            rel = self._load_prior_relevance(source, batch_size, num_tokens,
                                             int(t_grid), int(h_grid), int(w_grid), x.device)
            if rel is None:
                rel = compute_importance(x, int(t_grid), int(h_grid), int(w_grid),
                                         "motion" if source not in ("norm", "norm_motion",
                                                                    "qk_global_hidden") else source)
            if rel is None:
                rel = torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.float32)
            self._wam_rel_orig = rel.detach().float()              # [B, num_original]
            return _unit01(rel)

        # Later layers: gather cached relevance by current tokens' original ids.
        cache = getattr(self, "_wam_rel_orig", None)
        if (cache is not None and cache.shape[0] == batch_size
                and cache.shape[1] == num_original):
            return _unit01(cache.gather(1, token_ids))
        # Fallback: no cache (first merge layer wasn't dense) -> no protection.
        return torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.float32)
