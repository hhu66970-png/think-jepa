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

from src.models.utils.token_merge import LocalTokenMerger


class DiagnosticTokenMerger(LocalTokenMerger):
    """``LocalTokenMerger`` extended with the Gradual K-BSM strategy.

    A/B/C behaviour is inherited unchanged from the base class; this subclass
    adds only ``bsm_ksim_gradual_vec`` (the verified encoder-speedup winner).
    """

    VECTORIZED_STRATEGIES = LocalTokenMerger.VECTORIZED_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # Gradual K-BSM (grid-agnostic, multi-layer)
    )
    NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # must never hit the 2x2 Python fallback path
    )

    def _method_name(self):
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return "BSM_ksim_gradual"
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
