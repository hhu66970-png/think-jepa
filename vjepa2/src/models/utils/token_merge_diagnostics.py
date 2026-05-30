"""Research-only diagnostic token-merge strategies (B2 / C2).

These two strategies were evaluated against the similarity-only baseline
(method A) and the importance-protected / hybrid variants (B / C). The closure
reports concluded they form **No-Go** points: they never produced a Pareto
improvement over A, and they are NOT meant to enter the main ThinkJEPA training
or inference path.

To keep the main path clean (see ``LocalTokenMerger`` in ``token_merge.py``),
the B2/C2 algorithms live here, in a subclass that is only instantiated by the
research diagnostic pipeline (``tools/run_encoder_token_merge_full_pipeline.py``)
when one of these strategies is explicitly requested. The encoder
(``vision_transformer.py``) and ``scripts/train.sh`` never reference them.

Strategy crosswalk:
    B2 = ``local_keep_then_merge_vec``               (keep-score then merge)
    C2 = ``local_2x2_similarity_gated_importance_vec`` (similarity-gated importance)
"""

import math

import torch
import torch.nn.functional as F

from src.models.utils.token_merge import LocalTokenMerger


class DiagnosticTokenMerger(LocalTokenMerger):
    """``LocalTokenMerger`` extended with the diagnostic-only B2/C2 strategies.

    A/B/C behaviour is inherited unchanged from the base class; only the two
    extra ``*_vec`` strategies are added here. This class is the *only* place
    the B2/C2 pair-selection algorithms exist.
    """

    VECTORIZED_STRATEGIES = LocalTokenMerger.VECTORIZED_STRATEGIES + (
        "local_keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec",
        "bsm_ksim_gradual_vec",  # NEW: Gradual K-BSM (grid-agnostic, multi-layer)
    )
    NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
        "local_keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec",
        "bsm_ksim_gradual_vec",  # NEW: must never hit the 2x2 Python fallback path
    )

    def _method_name(self):
        if self.config.strategy == "local_keep_then_merge_vec":
            return "B2_keep_then_merge"
        if self.config.strategy == "local_2x2_similarity_gated_importance_vec":
            return "C2_similarity_gated_importance"
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return "BSM_ksim_gradual"
        return super()._method_name()

    def _dispatch_pair_selection(
        self,
        similarities,
        flat_cell_positions,
        pair_left,
        pair_right,
        norm_cells,
        importance_cells,
        protected_cells,
        target_merges,
    ):
        if self.config.strategy == "local_keep_then_merge_vec":
            return self._select_keep_then_merge_pairs(
                similarities,
                flat_cell_positions,
                pair_left,
                pair_right,
                importance_cells,
                protected_cells,
                target_merges,
            )
        if self.config.strategy == "local_2x2_similarity_gated_importance_vec":
            return self._select_similarity_gated_importance_pairs(
                similarities,
                flat_cell_positions,
                pair_left,
                pair_right,
                norm_cells,
                importance_cells,
                protected_cells,
                target_merges,
            )
        return super()._dispatch_pair_selection(
            similarities,
            flat_cell_positions,
            pair_left,
            pair_right,
            norm_cells,
            importance_cells,
            protected_cells,
            target_merges,
        )

    # ------------------------------------------------------------------
    # B2/C2 pair-selection algorithms (moved verbatim from LocalTokenMerger)
    # ------------------------------------------------------------------
    def _local_similarity_matrix(self, similarities, pair_left, pair_right):
        batch_size, num_cells = similarities.shape[:2]
        sim_matrix = torch.full(
            (batch_size, num_cells, 4, 4),
            -torch.inf,
            device=similarities.device,
            dtype=similarities.dtype,
        )
        sim_matrix[:, :, pair_left, pair_right] = similarities
        sim_matrix[:, :, pair_right, pair_left] = similarities
        return sim_matrix

    def _select_keep_then_merge_pairs(
        self,
        similarities,
        flat_cell_positions,
        pair_left,
        pair_right,
        importance_cells,
        protected_cells,
        target_merges,
    ):
        sim_matrix = self._local_similarity_matrix(similarities, pair_left, pair_right)
        redundancy = sim_matrix.max(dim=-1).values
        novelty = 1.0 - redundancy
        keep_source = str(self.config.keep_source)
        if keep_source == "redundancy":
            keep_score = novelty
        elif keep_source == "importance":
            keep_score = importance_cells
        elif keep_source == "importance_redundancy":
            keep_score = (
                float(self.config.keep_score_alpha) * novelty
                + float(self.config.keep_score_beta) * importance_cells
            )
        elif keep_source == "random":
            keep_score = torch.rand_like(novelty)
        else:
            raise ValueError(f"Unsupported keep_source: {keep_source}")

        source_local = keep_score.argmin(dim=-1)
        source_protected = protected_cells.gather(2, source_local.unsqueeze(-1)).squeeze(-1)
        sim_to_receiver = sim_matrix.gather(
            2, source_local.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 4)
        ).squeeze(2)
        local_index = torch.arange(4, device=similarities.device, dtype=torch.long)
        valid_receiver = local_index.view(1, 1, 4) != source_local.unsqueeze(-1)
        sim_to_receiver = sim_to_receiver.masked_fill(~valid_receiver, -torch.inf)
        best_receiver_sim, receiver_local = sim_to_receiver.max(dim=-1)

        valid = ~source_protected & torch.isfinite(best_receiver_sim)
        if float(self.config.similarity_threshold) >= 0.0:
            valid = valid & (best_receiver_sim >= float(self.config.similarity_threshold))

        num_accepted = self._candidate_target_merges(valid, target_merges)
        candidate_count = int(valid.sum().item())
        candidate_cell_count = self._candidate_cell_count(valid)
        if num_accepted <= 0:
            empty = torch.empty(similarities.shape[0], 0, device=similarities.device, dtype=torch.long)
            return empty, empty, torch.empty_like(empty, dtype=similarities.dtype), 0, candidate_count, candidate_cell_count

        cell_score = best_receiver_sim.masked_fill(~valid, -torch.inf)
        selected_scores, selected_cells = cell_score.topk(num_accepted, dim=1)
        if not bool(torch.isfinite(selected_scores).all().item()):
            raise RuntimeError("Invalid keep-then-merge selection produced non-finite scores")
        selected_positions = flat_cell_positions[selected_cells]
        source_local = source_local.gather(1, selected_cells)
        receiver_local = receiver_local.gather(1, selected_cells)
        source_pos = selected_positions.gather(2, source_local.unsqueeze(-1)).squeeze(-1)
        receiver_pos = selected_positions.gather(2, receiver_local.unsqueeze(-1)).squeeze(-1)
        return source_pos, receiver_pos, selected_scores, num_accepted, candidate_count, candidate_cell_count

    def _select_similarity_gated_importance_pairs(
        self,
        similarities,
        flat_cell_positions,
        pair_left,
        pair_right,
        norm_cells,
        importance_cells,
        protected_cells,
        target_merges,
    ):
        left_importance = importance_cells[:, :, pair_left]
        right_importance = importance_cells[:, :, pair_right]
        left_norm = norm_cells[:, :, pair_left]
        right_norm = norm_cells[:, :, pair_right]
        if bool(self.config.direction_by_importance):
            left_is_receiver = torch.where(
                left_importance == right_importance,
                left_norm >= right_norm,
                left_importance >= right_importance,
            )
        elif self.config.receiver == "max_norm":
            left_is_receiver = left_norm >= right_norm
        else:
            left_is_receiver = torch.ones_like(similarities, dtype=torch.bool)

        source_local_all = torch.where(left_is_receiver, pair_right, pair_left)
        receiver_local_all = torch.where(left_is_receiver, pair_left, pair_right)
        best_sim = similarities.max(dim=-1, keepdim=True).values
        epsilon = max(0.0, float(self.config.similarity_gate_epsilon))
        valid = similarities >= (best_sim - epsilon)
        source_protected = protected_cells.gather(2, source_local_all)
        valid = valid & ~source_protected
        if float(self.config.similarity_threshold) >= 0.0:
            valid = valid & (similarities >= float(self.config.similarity_threshold))

        num_accepted = self._candidate_target_merges(valid, target_merges)
        candidate_count = int(valid.sum().item())
        candidate_cell_count = self._candidate_cell_count(valid)
        if num_accepted <= 0:
            empty = torch.empty(similarities.shape[0], 0, device=similarities.device, dtype=torch.long)
            return empty, empty, torch.empty_like(empty, dtype=similarities.dtype), 0, candidate_count, candidate_cell_count

        importance_gap = (left_importance - right_importance).abs()
        tie_weight = float(self.config.score_beta) * max(epsilon, 1e-6)
        score = similarities + tie_weight * importance_gap
        score = score.masked_fill(~valid, -torch.inf)
        best_score, best_pair_index = score.max(dim=-1)
        selected_scores_for_cells, selected_cells = best_score.topk(num_accepted, dim=1)
        if not bool(torch.isfinite(selected_scores_for_cells).all().item()):
            raise RuntimeError("Invalid similarity-gated selection produced non-finite scores")
        selected_pair_index = best_pair_index.gather(1, selected_cells)
        source_local = source_local_all.gather(
            1, selected_cells.unsqueeze(-1).expand(-1, -1, 6)
        )
        receiver_local = receiver_local_all.gather(
            1, selected_cells.unsqueeze(-1).expand(-1, -1, 6)
        )
        source_local = source_local.gather(2, selected_pair_index.unsqueeze(-1)).squeeze(-1)
        receiver_local = receiver_local.gather(2, selected_pair_index.unsqueeze(-1)).squeeze(-1)
        selected_positions = flat_cell_positions[selected_cells]
        source_pos = selected_positions.gather(2, source_local.unsqueeze(-1)).squeeze(-1)
        receiver_pos = selected_positions.gather(2, receiver_local.unsqueeze(-1)).squeeze(-1)
        selected_similarities = similarities.gather(
            1, selected_cells.unsqueeze(-1).expand(-1, -1, similarities.shape[-1])
        )
        selected_cos = selected_similarities.gather(2, selected_pair_index.unsqueeze(-1)).squeeze(-1)
        return source_pos, receiver_pos, selected_cos, num_accepted, candidate_count, candidate_cell_count

    # ------------------------------------------------------------------
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
        # 2x2 path; also cap at floor((N-1)/2) so the alternating A/B split always
        # has a partner for every kept A-token and at least one token survives.
        ratio = max(0.0, min(self.config.merge_ratio, 0.25))
        r = int(math.floor(num_tokens * ratio))
        r = min(r, (num_tokens - 1) // 2)
        if r <= 0 or num_tokens < 2:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        # -- Matching metric -------------------------------------------------
        # Prefer the post-RoPE attention Key (SDPA-safe stash; mean-over-heads
        # [B, N, head_dim]). Fall back to block-output hidden features and FLAG
        # it clearly in info. Speed is identical; only fidelity differs.
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

        # -- Alternating bipartite partition over the CURRENT token order -----
        # ToMe: even positions -> set A (sources), odd positions -> set B (dst).
        # Purely positional over num_tokens; never references t/h/w grid, so it is
        # correct whether token_ids is the dense arange (L_start) or an arbitrary
        # survivor subset (layers > L_start).
        pos = torch.arange(num_tokens, device=x.device)
        if str(getattr(self.config, "bsm_partition", "positional")) == "temporal":
            # Temporal split: even TUBELET -> A, odd -> B, so each A-token matches
            # the most-similar token in a DIFFERENT time step (inter-frame merge).
            # t recovered from ORIGINAL ids (id // tokens_per_frame); exact at a
            # single merge layer where token_ids == arange.
            tokens_per_frame = max(1, int(h_grid) * int(w_grid))
            t_of_pos = (token_ids[0].to(torch.long) // tokens_per_frame)
            a_mask = (t_of_pos % 2) == 0
            a_idx = pos[a_mask]            # [Na] even-tubelet positions
            b_idx = pos[~a_mask]          # [Nb] odd-tubelet positions
        else:
            a_idx = pos[0::2]                 # [Na]
            b_idx = pos[1::2]                 # [Nb]
        na = a_idx.numel()
        nb = b_idx.numel()
        # r cannot exceed the number of A-tokens (each A contributes <=1 edge).
        r = min(r, na)
        if r <= 0 or nb == 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        a_metric = metric.index_select(1, a_idx)   # [B, Na, d]
        b_metric = metric.index_select(1, b_idx)   # [B, Nb, d]
        # cosine sim A x B (unit vectors -> dot product). [B, Na, Nb]
        scores = torch.bmm(a_metric, b_metric.transpose(1, 2))
        # each A-token -> most similar B-token
        best_sim, best_b_local = scores.max(dim=2)   # [B, Na], [B, Na]
        # keep the r strongest A->B edges per sample (top-r over A-tokens). r is a
        # single scalar across the batch, so exactly r sources are dropped per
        # sample -> x_new stays a dense [B, N-r, D] (same invariant A/B/C rely on).
        edge_sim, edge_a_local = best_sim.topk(r, dim=1)        # [B, r]
        edge_b_local = best_b_local.gather(1, edge_a_local)     # [B, r]

        # map A/B local indices back to ABSOLUTE token positions
        source_pos = a_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_a_local)   # [B, r]
        receiver_pos = b_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_b_local) # [B, r]

        # NOTE on collisions: two A-tokens may pick the same B-token. Like ToMe's
        # vectorized form we average BOTH into that B token via scatter-add, so
        # size-weighting stays exact. Each source is unique (A-tokens are disjoint
        # and never overlap B) so "drop sources, keep rest" stays rectangular with
        # exactly r removed per sample regardless of receiver collisions.

        # -- Size-weighted merge (req 4); scatter-add to combine collisions ----
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

        # -- Drop source tokens, keep the rest (rectangular: r removed/sample) -
        keep = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        keep.scatter_(1, source_pos, False)
        num_after = num_tokens - r
        x_new = x_updated[keep].reshape(batch_size, num_after, dim)
        ids_new = token_ids[keep].reshape(batch_size, num_after)
        size_new = token_size_updated[keep].reshape(batch_size, num_after)

        # -- Multi-layer-correct rep_for_orig carry-forward --------------------
        # Remap EVERY original whose CURRENT rep is one of this layer's sources
        # (not just originals whose id == source_id). Source ids are distinct
        # (A-tokens disjoint) and never coincide with receivers in the same layer,
        # so an identity-LUT scatter + gather is exact and composes across layers.
        # Required so restore_dense stays correct under gradual multi-layer BSM;
        # for the default compressed path (restore_dense=False) it is harmless.
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
        if fallback_reason is not None:
            info["fallback_reason"] = fallback_reason
        return x_new, ids_new, size_new, rep_new, info
