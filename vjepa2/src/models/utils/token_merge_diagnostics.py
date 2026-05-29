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

import torch

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
    )
    NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
        "local_keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec",
    )

    def _method_name(self):
        if self.config.strategy == "local_keep_then_merge_vec":
            return "B2_keep_then_merge"
        if self.config.strategy == "local_2x2_similarity_gated_importance_vec":
            return "C2_similarity_gated_importance"
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
