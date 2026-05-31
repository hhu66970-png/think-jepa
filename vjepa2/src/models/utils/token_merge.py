import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MergeConfig:
    enabled: bool = False
    merge_layers: tuple[int, ...] = ()
    merge_ratio: float = 0.0
    strategy: str = "local_2x2_same_time"
    receiver: str = "max_norm"
    restore_dense: bool = True
    profile: bool = False
    importance_source: str = "none"
    protect_mode: str = "none"
    protect_ratio: float = 0.0
    similarity_threshold: float = -1.0
    dynamic_ratio_mode: str = "none"
    score_alpha: float = 1.0
    score_beta: float = 0.3
    score_gamma: float = 0.5
    score_delta: float = 0.0
    lambda_norm: float = 0.3
    lambda_motion: float = 0.7
    debug_dump_scores: bool = False
    dump_merge_decisions: bool = False
    max_decision_dump: int = 8192
    keep_source: str = "redundancy"
    receiver_search: str = "cell"
    keep_score_alpha: float = 1.0
    keep_score_beta: float = 0.0
    similarity_gate_epsilon: float = 0.01
    direction_by_importance: bool = True
    # Matching metric for bsm_ksim_gradual_vec: "key" => post-RoPE attention Key
    # cosine (SDPA-safe stash), "feature" => block-output hidden-feature cosine.
    # Ignored by every other strategy. Defaulted so A/B/C/B2/C2 are unaffected.
    bsm_match_metric: str = "key"


def normalize_merge_config(config):
    if config is None:
        return MergeConfig()
    if isinstance(config, MergeConfig):
        _validate_merge_config(config)
        return config
    if not isinstance(config, dict):
        raise TypeError(f"merge_config must be a dict or MergeConfig, got {type(config)!r}")
    layers = config.get("merge_layers", ())
    if isinstance(layers, str):
        layers = tuple(int(item) for item in layers.split(",") if item.strip())
    strategy = str(config.get("strategy", "local_2x2_same_time"))
    layers = tuple(int(item) for item in layers)
    vectorized_strategies = (
        "local_2x2_same_time_vec",
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
        "local_keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec",
    )
    # Grid-agnostic strategies operate on arbitrary (already-compressed) token
    # sets and are explicitly ALLOWED to merge across MULTIPLE layers. The 2x2
    # cell strategies above still require a single dense merge layer. bsm is not
    # in vectorized_strategies, so the guard already skips it; this allow-set
    # documents intent and stays robust if it is ever added to the tuple above.
    grid_agnostic_multilayer_strategies = (
        "bsm_ksim_gradual_vec",
    )
    if (
        strategy in vectorized_strategies
        and strategy not in grid_agnostic_multilayer_strategies
        and len(layers) > 1
    ):
        raise ValueError(
            f"{strategy} currently supports exactly one merge layer. "
            "Use a single layer or implement sparse-state vectorized merging before "
            "enabling multi-layer vectorized merge."
        )
    importance_source = str(config.get("importance_source", "none"))
    if strategy in (
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
        "local_2x2_similarity_gated_importance_vec",
    ) and importance_source == "none":
        raise ValueError(
            f"{strategy} requires importance_source != 'none'. "
            "Use norm, motion, norm_motion, or qk_global_hidden."
        )
    keep_source = str(config.get("keep_source", "redundancy"))
    if strategy == "local_keep_then_merge_vec" and "importance" in keep_source and importance_source == "none":
        raise ValueError(
            "local_keep_then_merge_vec with an importance-based keep_source requires "
            "importance_source != 'none'."
        )
    normalized = MergeConfig(
        enabled=bool(config.get("enabled", False)),
        merge_layers=layers,
        merge_ratio=float(config.get("merge_ratio", 0.0)),
        strategy=strategy,
        receiver=str(config.get("receiver", "max_norm")),
        restore_dense=bool(config.get("restore_dense", True)),
        profile=bool(config.get("profile", False)),
        importance_source=importance_source,
        protect_mode=str(config.get("protect_mode", "none")),
        protect_ratio=float(config.get("protect_ratio", 0.0)),
        similarity_threshold=float(config.get("similarity_threshold", -1.0)),
        dynamic_ratio_mode=str(config.get("dynamic_ratio_mode", "none")),
        score_alpha=float(config.get("score_alpha", 1.0)),
        score_beta=float(config.get("score_beta", 0.3)),
        score_gamma=float(config.get("score_gamma", 0.5)),
        score_delta=float(config.get("score_delta", 0.0)),
        lambda_norm=float(config.get("lambda_norm", 0.3)),
        lambda_motion=float(config.get("lambda_motion", 0.7)),
        debug_dump_scores=bool(config.get("debug_dump_scores", False)),
        dump_merge_decisions=bool(config.get("dump_merge_decisions", False)),
        max_decision_dump=int(config.get("max_decision_dump", 8192)),
        keep_source=keep_source,
        receiver_search=str(config.get("receiver_search", "cell")),
        keep_score_alpha=float(config.get("keep_score_alpha", 1.0)),
        keep_score_beta=float(config.get("keep_score_beta", 0.0)),
        similarity_gate_epsilon=float(config.get("similarity_gate_epsilon", 0.01)),
        direction_by_importance=bool(config.get("direction_by_importance", True)),
        bsm_match_metric=str(config.get("bsm_match_metric", "key")),
    )
    _validate_merge_config(normalized)
    return normalized


def _validate_merge_config(config):
    if config.strategy in (
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
        "local_2x2_similarity_gated_importance_vec",
    ) and config.importance_source == "none":
        raise ValueError(
            f"{config.strategy} requires importance_source != 'none'. "
            "Use norm, motion, norm_motion, or qk_global_hidden."
        )
    if config.strategy == "local_keep_then_merge_vec":
        if config.receiver_search != "cell":
            raise ValueError("local_keep_then_merge_vec currently supports receiver_search='cell'")
        if config.keep_source not in ("redundancy", "importance", "importance_redundancy", "random"):
            raise ValueError(
                "keep_source must be one of redundancy|importance|importance_redundancy|random, "
                f"got {config.keep_source!r}"
            )
        if "importance" in config.keep_source and config.importance_source == "none":
            raise ValueError(
                "local_keep_then_merge_vec with an importance-based keep_source requires "
                "importance_source != 'none'."
            )


def _normalize_per_sample(score, eps=1e-6):
    score = score.float()
    min_val = score.amin(dim=1, keepdim=True)
    max_val = score.amax(dim=1, keepdim=True)
    return (score - min_val) / (max_val - min_val + eps)


def compute_importance(
    x,
    t_grid,
    h_grid,
    w_grid,
    source: str,
    lambda_norm: float = 0.3,
    lambda_motion: float = 0.7,
    eps: float = 1e-6,
):
    """Return per-token importance [B, N] for dense video patch tokens."""
    source = str(source or "none")
    if source == "none":
        return None

    batch_size, num_tokens, dim = x.shape
    expected = int(t_grid * h_grid * w_grid)
    if num_tokens != expected:
        raise ValueError(
            f"importance_source={source} requires dense tokens, got {num_tokens} vs {expected}"
        )

    x_float = x.float()
    norm_importance = _normalize_per_sample(torch.linalg.norm(x_float, dim=-1), eps=eps)

    if source == "norm":
        return norm_importance

    if source in ("motion", "norm_motion"):
        x_grid = x_float.reshape(batch_size, int(t_grid), int(h_grid), int(w_grid), dim)
        motion = torch.zeros(
            batch_size,
            int(t_grid),
            int(h_grid),
            int(w_grid),
            device=x.device,
            dtype=torch.float32,
        )
        if int(t_grid) > 1:
            motion[:, 1:] = torch.linalg.norm(x_grid[:, 1:] - x_grid[:, :-1], dim=-1)
        motion_importance = _normalize_per_sample(motion.reshape(batch_size, num_tokens), eps=eps)
        if source == "motion":
            return motion_importance
        mixed = float(lambda_norm) * norm_importance + float(lambda_motion) * motion_importance
        return _normalize_per_sample(mixed, eps=eps)

    if source == "qk_global_hidden":
        x_norm = F.normalize(x_float, dim=-1, eps=eps)
        global_query = F.normalize(x_norm.mean(dim=1, keepdim=True), dim=-1, eps=eps)
        score = (x_norm * global_query).sum(dim=-1)
        return _normalize_per_sample(score, eps=eps)

    raise ValueError(
        "importance_source must be one of none|norm|motion|norm_motion|qk_global_hidden, "
        f"got {source!r}"
    )


def init_token_merge_state(batch_size, num_tokens, device, dtype):
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
    token_ids = token_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    token_size = torch.ones(batch_size, num_tokens, device=device, dtype=dtype)
    rep_for_orig = token_ids.clone()
    return token_ids, token_size, rep_for_orig


def ids_to_coords(token_ids, h_grid, w_grid):
    tokens_per_frame = int(h_grid * w_grid)
    t = token_ids // tokens_per_frame
    rem = token_ids - t * tokens_per_frame
    h = rem // int(w_grid)
    w = rem - h * int(w_grid)
    return t, h, w


def restore_dense_tokens(x, token_ids, rep_for_orig, num_original_tokens):
    """Scatter compressed tokens back to the original dense grid shape."""
    batch_size, num_tokens = token_ids.shape
    pos_by_original_id = torch.empty(
        batch_size, num_original_tokens, device=x.device, dtype=torch.long
    )
    token_positions = torch.arange(num_tokens, device=x.device, dtype=torch.long)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    pos_by_original_id.scatter_(1, token_ids, token_positions)
    gather_pos = pos_by_original_id.gather(1, rep_for_orig)
    gather_pos = gather_pos.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(1, gather_pos)


class LocalTokenMerger(nn.Module):
    """Training-free local same-time 2x2 token merging for video ViT tokens."""

    # Vectorized strategies handled on the MAIN path: A (similarity-only),
    # B (importance-protected), C (hybrid score). The diagnostic-only B2/C2
    # strategies are intentionally NOT listed here; they live in
    # DiagnosticTokenMerger (token_merge_diagnostics.py), which extends these
    # tuples so the main training/inference path can never select them.
    VECTORIZED_STRATEGIES = (
        "local_2x2_same_time_vec",
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
    )
    # Vectorized strategies that must NOT silently fall back to the slow Python
    # similarity path (doing so would drop importance/protection semantics).
    NO_PYTHON_FALLBACK_STRATEGIES = (
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
    )

    def __init__(self, config):
        super().__init__()
        self.config = normalize_merge_config(config)

    @torch.no_grad()
    def forward(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid, attn_key=None):
        if self.config.strategy in ("local_2x2_same_time", "local_2x2_same_time_python"):
            return self._forward_python(
                x,
                token_ids,
                token_size,
                rep_for_orig,
                t_grid,
                h_grid,
                w_grid,
                implementation="python",
            )
        if self.config.strategy not in self.VECTORIZED_STRATEGIES:
            raise ValueError(f"Unsupported merge strategy: {self.config.strategy}")

        # Grid-agnostic global bipartite soft matching (BSM). Handled entirely
        # separately from the dense 2x2 path; works on compressed token sets and
        # across multiple merge layers. _forward_bsm is defined ONLY on
        # DiagnosticTokenMerger and the strategy name is registered ONLY in that
        # subclass's VECTORIZED_STRATEGIES, so the base LocalTokenMerger raises
        # "Unsupported merge strategy" above and never reaches this branch.
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return self._forward_bsm(
                x, token_ids, token_size, rep_for_orig,
                int(t_grid), int(h_grid), int(w_grid), attn_key,
            )

        if self.config.merge_ratio <= 0.0:
            importance = None
            if self.config.importance_source != "none":
                importance = compute_importance(
                    x,
                    int(t_grid),
                    int(h_grid),
                    int(w_grid),
                    self.config.importance_source,
                    lambda_norm=self.config.lambda_norm,
                    lambda_motion=self.config.lambda_motion,
                )
            return x, token_ids, token_size, rep_for_orig, self._info(
                x,
                x,
                0,
                0,
                implementation="vectorized",
                importance=importance,
                protected_mask=None,
            )

        can_vectorize, fallback_reason = self._can_vectorize_dense_grid(
            x, token_ids, rep_for_orig, int(t_grid), int(h_grid), int(w_grid)
        )
        if not can_vectorize:
            if self.config.strategy in self.NO_PYTHON_FALLBACK_STRATEGIES:
                raise RuntimeError(
                    f"{self.config.strategy} cannot fall back to the Python similarity path "
                    f"because that would drop importance/protection semantics: {fallback_reason}"
                )
            return self._forward_python(
                x,
                token_ids,
                token_size,
                rep_for_orig,
                t_grid,
                h_grid,
                w_grid,
                implementation="python_fallback",
                fallback_reason=fallback_reason,
            )

        return self._forward_vectorized(
            x,
            token_ids,
            token_size,
            rep_for_orig,
            int(t_grid),
            int(h_grid),
            int(w_grid),
        )

    def _forward_python(
        self,
        x,
        token_ids,
        token_size,
        rep_for_orig,
        t_grid,
        h_grid,
        w_grid,
        implementation,
        fallback_reason=None,
    ):
        if self.config.merge_ratio <= 0.0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x,
                x,
                0,
                0,
                implementation=implementation,
                fallback_reason=fallback_reason,
            )

        pairs_by_batch = []
        max_merge = 0
        selected_scores = []
        for batch_idx in range(x.shape[0]):
            pairs = self._select_pairs_one_batch(
                x[batch_idx],
                token_ids[batch_idx],
                token_size[batch_idx],
                int(t_grid),
                int(h_grid),
                int(w_grid),
            )
            pairs_by_batch.append(pairs)
            max_merge = max(max_merge, len(pairs))
            if pairs:
                selected_scores.extend(pair[2] for pair in pairs)

        if max_merge == 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x,
                x,
                0,
                0,
                selected_scores=selected_scores,
                implementation=implementation,
                fallback_reason=fallback_reason,
            )

        merged = []
        merged_ids = []
        merged_size = []
        merged_rep = []
        num_merged = []
        for batch_idx, pairs in enumerate(pairs_by_batch):
            x_b, ids_b, size_b, rep_b = self._apply_pairs_one_batch(
                x[batch_idx],
                token_ids[batch_idx],
                token_size[batch_idx],
                rep_for_orig[batch_idx],
                pairs,
            )
            merged.append(x_b)
            merged_ids.append(ids_b)
            merged_size.append(size_b)
            merged_rep.append(rep_b)
            num_merged.append(len(pairs))

        # The current local cell strategy produces the same count per batch for normal dense inputs.
        # Keep this explicit check so shape bugs surface early instead of silently padding tokens.
        lengths = {item.shape[0] for item in merged}
        if len(lengths) != 1:
            raise RuntimeError(f"Batch elements produced different compressed lengths: {sorted(lengths)}")

        x_new = torch.stack(merged, dim=0)
        ids_new = torch.stack(merged_ids, dim=0)
        size_new = torch.stack(merged_size, dim=0)
        rep_new = torch.stack(merged_rep, dim=0)
        info = self._info(
            x,
            x_new,
            min(num_merged),
            max(num_merged),
            selected_scores=selected_scores,
            implementation=implementation,
            fallback_reason=fallback_reason,
        )
        return x_new, ids_new, size_new, rep_new, info

    def _can_vectorize_dense_grid(self, x, token_ids, rep_for_orig, t_grid, h_grid, w_grid):
        # Grid-agnostic global BSM does not require a dense contiguous 2x2 grid;
        # it can run on already-compressed token sets at every merge layer. (The
        # forward() dispatch routes bsm before this is reached; this guard is a
        # defensive no-op for that strategy.)
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return True, None
        expected_tokens = int(t_grid * h_grid * w_grid)
        if h_grid % 2 != 0 or w_grid % 2 != 0:
            return False, "odd_spatial_grid"
        if x.shape[1] != expected_tokens or token_ids.shape[1] != expected_tokens:
            return False, "not_full_dense_grid"
        if rep_for_orig.shape[1] != expected_tokens:
            return False, "rep_for_orig_not_dense"

        expected_ids = torch.arange(expected_tokens, device=token_ids.device, dtype=torch.long)
        expected_ids = expected_ids.unsqueeze(0).expand(token_ids.shape[0], -1)
        if not torch.equal(token_ids, expected_ids):
            return False, "token_ids_not_contiguous_dense_grid"
        if not torch.equal(rep_for_orig, expected_ids):
            return False, "rep_for_orig_not_identity_dense_grid"
        return True, None

    def _forward_vectorized(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid):
        batch_size, num_tokens, dim = x.shape
        target_merges = int(math.floor(num_tokens * max(0.0, min(self.config.merge_ratio, 0.25))))
        if target_merges <= 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, implementation="vectorized"
            )

        h_cells = h_grid // 2
        w_cells = w_grid // 2
        num_cells = int(t_grid * h_cells * w_cells)
        target_merges = min(target_merges, num_cells)

        x_cells = self._reshape_2x2_cells(x, batch_size, t_grid, h_cells, w_cells, dim)
        x_cells = x_cells.reshape(batch_size, num_cells, 4, dim)

        pair_left = torch.tensor([0, 0, 0, 1, 1, 2], device=x.device, dtype=torch.long)
        pair_right = torch.tensor([1, 2, 3, 2, 3, 3], device=x.device, dtype=torch.long)
        x_normed = F.normalize(x_cells.float(), dim=-1, eps=1e-6)
        similarities = (x_normed[:, :, pair_left] * x_normed[:, :, pair_right]).sum(dim=-1)

        flat_cell_positions = torch.arange(num_tokens, device=x.device, dtype=torch.long)
        flat_cell_positions = flat_cell_positions.reshape(t_grid, h_cells, 2, w_cells, 2)
        flat_cell_positions = flat_cell_positions.permute(0, 1, 3, 2, 4).reshape(num_cells, 4)

        token_norm = torch.linalg.norm(x.float(), dim=-1)
        norm_cells = token_norm.reshape(batch_size, num_cells, 4)
        importance = compute_importance(
            x,
            t_grid,
            h_grid,
            w_grid,
            self.config.importance_source,
            lambda_norm=self.config.lambda_norm,
            lambda_motion=self.config.lambda_motion,
        )
        if importance is None:
            importance_cells = torch.zeros(
                batch_size, num_cells, 4, device=x.device, dtype=torch.float32
            )
        else:
            importance_cells = importance.reshape(batch_size, num_cells, 4)
        protected_cells = self._compute_protected_cells(
            importance, importance_cells, flat_cell_positions
        )

        (
            source_pos,
            receiver_pos,
            selected_scores,
            num_accepted,
            candidate_count,
            candidate_cell_count,
        ) = self._dispatch_pair_selection(
            similarities,
            flat_cell_positions,
            pair_left,
            pair_right,
            norm_cells,
            importance_cells,
            protected_cells,
            target_merges,
        )

        if num_accepted <= 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x,
                x,
                0,
                0,
                selected_scores=None,
                implementation="vectorized",
                importance=importance,
                protected_mask=protected_cells.reshape(batch_size, -1),
                num_candidates=candidate_count,
                num_candidate_cells=candidate_cell_count,
                num_accepted=0,
            )

        # Shared, grid-agnostic merge/compaction (gather/scatter by token
        # position, size-weighted average, build compressed tensors, update
        # rep_for_orig). Reused verbatim by the global BSM path. multi_layer_rep
        # is False here so the A/B/C/B2/C2 tensors are byte-identical.
        x_new, ids_new, size_new, rep_new = self._apply_merge_from_positions(
            x,
            token_ids,
            token_size,
            rep_for_orig,
            source_pos,
            receiver_pos,
            num_accepted,
            multi_layer_rep=False,
        )

        info = self._info(
            x,
            x_new,
            num_accepted,
            num_accepted,
            selected_scores=selected_scores,
            implementation="vectorized",
            importance=importance,
            protected_mask=protected_cells.reshape(batch_size, -1),
            source_importance=(
                importance.gather(1, source_pos) if importance is not None else None
            ),
            receiver_importance=(
                importance.gather(1, receiver_pos) if importance is not None else None
            ),
            num_candidates=candidate_count,
            num_candidate_cells=candidate_cell_count,
            num_accepted=num_accepted,
            decision_dump=self._build_decision_dump(
                source_pos,
                receiver_pos,
                selected_scores,
                similarities,
                flat_cell_positions,
                token_ids,
                importance,
                t_grid,
                h_cells,
                w_cells,
            ),
        )
        return x_new, ids_new, size_new, rep_new, info

    def _apply_merge_from_positions(
        self,
        x,
        token_ids,
        token_size,
        rep_for_orig,
        source_pos,
        receiver_pos,
        num_accepted,
        multi_layer_rep=False,
    ):
        """Shared post-pair-selection merge/compaction (grid-agnostic).

        Gather source/receiver by token POSITION, size-weighted average into the
        receiver, drop the source positions, and compact to a dense
        ``[B, N-num_accepted, D]``. This assumes source positions are disjoint
        from each other AND from receiver positions, and that each receiver is
        targeted at most once (true for the 2x2 cell strategies). Extracted from
        ``_forward_vectorized`` so the 2x2 path and global BSM share ONE
        implementation; with ``multi_layer_rep=False`` the produced tensors are
        byte-identical to the original A/B/C/B2/C2 behaviour.
        """
        batch_size, num_tokens, dim = x.shape

        source_ids = token_ids.gather(1, source_pos)
        receiver_ids = token_ids.gather(1, receiver_pos)
        source_weight = token_size.gather(1, source_pos)
        receiver_weight = token_size.gather(1, receiver_pos)
        total_weight = source_weight + receiver_weight

        source_x = x.gather(1, source_pos.unsqueeze(-1).expand(-1, -1, dim))
        receiver_x = x.gather(1, receiver_pos.unsqueeze(-1).expand(-1, -1, dim))
        merged_x = (
            receiver_x * receiver_weight.unsqueeze(-1)
            + source_x * source_weight.unsqueeze(-1)
        ) / total_weight.unsqueeze(-1)

        x_updated = x.clone()
        token_size_updated = token_size.clone()
        x_updated.scatter_(1, receiver_pos.unsqueeze(-1).expand(-1, -1, dim), merged_x)
        token_size_updated.scatter_(1, receiver_pos, total_weight)

        keep = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        keep.scatter_(1, source_pos, False)
        num_after = num_tokens - num_accepted
        x_new = x_updated[keep].reshape(batch_size, num_after, dim)
        ids_new = token_ids[keep].reshape(batch_size, num_after)
        size_new = token_size_updated[keep].reshape(batch_size, num_after)

        if multi_layer_rep:
            # Remap EVERY original whose CURRENT rep is one of this layer's
            # sources (not just originals whose id == source_id, which is all the
            # base scatter did). Within one layer source positions are disjoint
            # and never coincide with receivers, so a single identity-LUT scatter
            # + gather is exact; across layers it composes because each layer
            # applies its own remap to the running rep_for_orig. This is the
            # vectorized equivalent of the Python path's
            # ``rep[rep == source_id] = receiver_id`` and is required for correct
            # restore_dense under gradual multi-layer merging.
            num_original_tokens = rep_for_orig.shape[1]
            remap = (
                torch.arange(num_original_tokens, device=x.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .clone()
            )
            remap.scatter_(1, source_ids, receiver_ids)
            rep_new = remap.gather(1, rep_for_orig)
        else:
            # BYTE-IDENTICAL to the original A/B/C/B2/C2 behaviour.
            rep_new = rep_for_orig.clone()
            rep_new.scatter_(1, source_ids, receiver_ids)

        return x_new, ids_new, size_new, rep_new

    def _compute_protected_cells(self, importance, importance_cells, flat_cell_positions):
        protected = torch.zeros_like(importance_cells, dtype=torch.bool)
        mode = str(self.config.protect_mode or "none")
        if mode in ("local_top1", "local_top1_global_topk"):
            top_local = importance_cells.argmax(dim=-1, keepdim=True)
            protected.scatter_(-1, top_local, True)
        if mode in ("global_topk", "local_top1_global_topk") and importance is not None:
            ratio = max(0.0, min(float(self.config.protect_ratio), 1.0))
            k = int(math.floor(importance.shape[1] * ratio))
            if k > 0:
                _, top_idx = importance.topk(k, dim=1)
                global_mask = torch.zeros_like(importance, dtype=torch.bool)
                global_mask.scatter_(1, top_idx, True)
                grouped_positions = flat_cell_positions.reshape(1, -1).expand(
                    importance.shape[0], -1
                )
                protected_global = global_mask.gather(1, grouped_positions).reshape_as(protected)
                protected = protected | protected_global
        return protected

    def _candidate_target_merges(self, valid, target_merges):
        valid_cells = valid.any(dim=-1) if valid.ndim == 3 else valid
        valid_count = valid_cells.reshape(valid_cells.shape[0], -1).sum(dim=1)
        return int(min(target_merges, int(valid_count.min().item())))

    def _candidate_cell_count(self, valid):
        valid_cells = valid.any(dim=-1) if valid.ndim == 3 else valid
        return int(valid_cells.sum().item())

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
        """Select (source, receiver) merge pairs for the configured strategy.

        Base path handles A (``local_2x2_same_time_vec``),
        B (``local_2x2_importance_protected_vec``) and C
        (``local_2x2_hybrid_score_vec``). DiagnosticTokenMerger overrides this
        to add the research-only B2/C2 strategies.
        """
        if self.config.strategy == "local_2x2_hybrid_score_vec":
            return self._select_hybrid_pairs(
                similarities,
                flat_cell_positions,
                pair_left,
                pair_right,
                importance_cells,
                protected_cells,
                target_merges,
            )
        return self._select_similarity_pairs(
            similarities,
            flat_cell_positions,
            pair_left,
            pair_right,
            norm_cells,
            importance_cells,
            protected_cells,
            target_merges,
        )

    def _select_similarity_pairs(
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

        if self.config.strategy == "local_2x2_importance_protected_vec":
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
        if self.config.strategy == "local_2x2_same_time_vec":
            valid = torch.ones_like(similarities, dtype=torch.bool)
        else:
            source_protected = protected_cells.gather(2, source_local_all)
            valid = ~source_protected
            if float(self.config.similarity_threshold) >= 0.0:
                valid = valid & (similarities >= float(self.config.similarity_threshold))

        num_accepted = self._candidate_target_merges(valid, target_merges)
        candidate_count = int(valid.sum().item())
        candidate_cell_count = self._candidate_cell_count(valid)
        if num_accepted <= 0:
            empty = torch.empty(similarities.shape[0], 0, device=similarities.device, dtype=torch.long)
            return empty, empty, torch.empty_like(empty, dtype=similarities.dtype), 0, candidate_count, candidate_cell_count

        score = similarities.masked_fill(~valid, -torch.inf)
        best_scores, best_pair_index = score.max(dim=-1)
        selected_scores, selected_cells = best_scores.topk(num_accepted, dim=1)
        if not bool(torch.isfinite(selected_scores).all().item()):
            raise RuntimeError("Invalid similarity merge selection produced non-finite scores")
        selected_pair_index = best_pair_index.gather(1, selected_cells)
        source_local = source_local_all.gather(1, selected_cells.unsqueeze(-1).expand(-1, -1, 6))
        receiver_local = receiver_local_all.gather(1, selected_cells.unsqueeze(-1).expand(-1, -1, 6))
        source_local = source_local.gather(2, selected_pair_index.unsqueeze(-1)).squeeze(-1)
        receiver_local = receiver_local.gather(2, selected_pair_index.unsqueeze(-1)).squeeze(-1)
        selected_positions = flat_cell_positions[selected_cells]
        source_pos = selected_positions.gather(2, source_local.unsqueeze(-1)).squeeze(-1)
        receiver_pos = selected_positions.gather(2, receiver_local.unsqueeze(-1)).squeeze(-1)
        return source_pos, receiver_pos, selected_scores, num_accepted, candidate_count, candidate_cell_count

    def _select_hybrid_pairs(
        self,
        similarities,
        flat_cell_positions,
        pair_left,
        pair_right,
        importance_cells,
        protected_cells,
        target_merges,
    ):
        src_local = torch.cat([pair_left, pair_right], dim=0)
        dst_local = torch.cat([pair_right, pair_left], dim=0)
        directed_cos = torch.cat([similarities, similarities], dim=-1)
        src_importance = importance_cells[:, :, src_local]
        dst_importance = importance_cells[:, :, dst_local]
        score = (
            float(self.config.score_alpha) * directed_cos
            + float(self.config.score_beta) * dst_importance
            - float(self.config.score_gamma) * src_importance
        )
        src_protected = protected_cells.gather(2, src_local.unsqueeze(0).unsqueeze(0).expand_as(src_importance))
        valid = ~src_protected
        if float(self.config.similarity_threshold) >= 0.0:
            valid = valid & (directed_cos >= float(self.config.similarity_threshold))

        num_accepted = self._candidate_target_merges(valid, target_merges)
        candidate_count = int(valid.sum().item())
        candidate_cell_count = self._candidate_cell_count(valid)
        if num_accepted <= 0:
            empty = torch.empty(similarities.shape[0], 0, device=similarities.device, dtype=torch.long)
            return empty, empty, torch.empty_like(empty, dtype=similarities.dtype), 0, candidate_count, candidate_cell_count

        score = score.masked_fill(~valid, -torch.inf)
        best_score, best_dir_index = score.max(dim=-1)
        selected_scores, selected_cells = best_score.topk(num_accepted, dim=1)
        if not bool(torch.isfinite(selected_scores).all().item()):
            raise RuntimeError("Invalid hybrid merge selection produced non-finite scores")
        selected_dir_index = best_dir_index.gather(1, selected_cells)
        source_local = src_local[selected_dir_index]
        receiver_local = dst_local[selected_dir_index]
        selected_positions = flat_cell_positions[selected_cells]
        source_pos = selected_positions.gather(2, source_local.unsqueeze(-1)).squeeze(-1)
        receiver_pos = selected_positions.gather(2, receiver_local.unsqueeze(-1)).squeeze(-1)
        selected_directed_cos = directed_cos.gather(
            1, selected_cells.unsqueeze(-1).expand(-1, -1, directed_cos.shape[-1])
        )
        selected_cos = selected_directed_cos.gather(2, selected_dir_index.unsqueeze(-1)).squeeze(-1)
        return source_pos, receiver_pos, selected_cos, num_accepted, candidate_count, candidate_cell_count

    # NOTE: The diagnostic-only B2/C2 pair-selection helpers
    # (``_local_similarity_matrix``, ``_select_keep_then_merge_pairs``,
    # ``_select_similarity_gated_importance_pairs``) were moved to
    # DiagnosticTokenMerger in token_merge_diagnostics.py. They are research-only
    # (No-Go: never beat similarity-only baseline A) and are intentionally kept
    # out of this main-path class.

    def _build_decision_dump(
        self,
        source_pos,
        receiver_pos,
        selected_scores,
        similarities,
        flat_cell_positions,
        token_ids,
        importance,
        t_grid,
        h_cells,
        w_cells,
    ):
        if not bool(self.config.dump_merge_decisions) or source_pos.numel() == 0:
            return None

        max_items = max(0, int(self.config.max_decision_dump))
        if max_items == 0:
            return None
        limit = min(max_items, source_pos.shape[1])
        source_pos = source_pos[:, :limit]
        receiver_pos = receiver_pos[:, :limit]
        selected_scores = selected_scores[:, :limit]

        tokens_per_frame = int(h_cells * 2 * w_cells * 2)
        source_t = source_pos // tokens_per_frame
        source_rem = source_pos - source_t * tokens_per_frame
        source_h = source_rem // int(w_cells * 2)
        source_w = source_rem - source_h * int(w_cells * 2)
        receiver_t = receiver_pos // tokens_per_frame
        receiver_rem = receiver_pos - receiver_t * tokens_per_frame
        receiver_h = receiver_rem // int(w_cells * 2)
        receiver_w = receiver_rem - receiver_h * int(w_cells * 2)
        cell_h = source_h // 2
        cell_w = source_w // 2
        cell_id = source_t * int(h_cells * w_cells) + cell_h * int(w_cells) + cell_w
        source_local = (source_h % 2) * 2 + (source_w % 2)
        receiver_local = (receiver_h % 2) * 2 + (receiver_w % 2)

        selected_cell_sims = similarities.gather(
            1, cell_id.unsqueeze(-1).expand(-1, -1, similarities.shape[-1])
        )
        best_two = selected_cell_sims.topk(k=2, dim=-1).values
        best_similarity = best_two[:, :, 0]
        second_best_similarity = best_two[:, :, 1]

        source_token_id = token_ids.gather(1, source_pos)
        receiver_token_id = token_ids.gather(1, receiver_pos)
        source_importance = None
        receiver_importance = None
        if importance is not None:
            source_importance = importance.gather(1, source_pos)
            receiver_importance = importance.gather(1, receiver_pos)

        return {
            "strategy": self.config.strategy,
            "cell_id": cell_id.detach(),
            "time_id": source_t.detach(),
            "cell_h": cell_h.detach(),
            "cell_w": cell_w.detach(),
            "source_pos": source_pos.detach(),
            "receiver_pos": receiver_pos.detach(),
            "source_token_id": source_token_id.detach(),
            "receiver_token_id": receiver_token_id.detach(),
            "source_local_id": source_local.detach(),
            "receiver_local_id": receiver_local.detach(),
            "selected_similarity": selected_scores.detach(),
            "best_similarity": best_similarity.detach(),
            "second_best_similarity": second_best_similarity.detach(),
            "source_importance": source_importance.detach() if source_importance is not None else None,
            "receiver_importance": receiver_importance.detach() if receiver_importance is not None else None,
            "num_dumped_per_batch": int(limit),
            "num_selected_per_batch": int(source_pos.shape[1]),
        }

    def _reshape_2x2_cells(self, tensor, batch_size, t_grid, h_cells, w_cells, dim):
        return (
            tensor.reshape(batch_size, t_grid, h_cells, 2, w_cells, 2, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch_size, t_grid, h_cells, w_cells, 4, dim)
        )

    def _select_pairs_one_batch(self, x, token_ids, token_size, t_grid, h_grid, w_grid):
        target_merges = int(math.floor(x.shape[0] * max(0.0, min(self.config.merge_ratio, 0.25))))
        if target_merges <= 0:
            return []

        t, h, w = ids_to_coords(token_ids, h_grid, w_grid)
        cell_to_positions = {}
        for pos in range(token_ids.numel()):
            key = (int(t[pos].item()), int((h[pos] // 2).item()), int((w[pos] // 2).item()))
            cell_to_positions.setdefault(key, []).append(pos)

        candidates = []
        x_norm = F.normalize(x.float(), dim=-1, eps=1e-6)
        x_norm_value = torch.linalg.norm(x.float(), dim=-1)
        for positions in cell_to_positions.values():
            if len(positions) < 2:
                continue
            best = None
            for left_index in range(len(positions)):
                for right_index in range(left_index + 1, len(positions)):
                    left = positions[left_index]
                    right = positions[right_index]
                    score = float(torch.dot(x_norm[left], x_norm[right]).item())
                    if self.config.receiver == "max_norm":
                        receiver, source = (left, right) if x_norm_value[left] >= x_norm_value[right] else (right, left)
                    else:
                        receiver, source = min(left, right), max(left, right)
                    if best is None or score > best[0]:
                        best = (score, source, receiver)
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda item: item[0], reverse=True)
        used = set()
        pairs = []
        for score, source, receiver in candidates:
            if len(pairs) >= target_merges:
                break
            if source in used or receiver in used:
                continue
            used.add(source)
            used.add(receiver)
            pairs.append((source, receiver, score))
        return pairs

    def _apply_pairs_one_batch(self, x, token_ids, token_size, rep_for_orig, pairs):
        if not pairs:
            return x, token_ids, token_size, rep_for_orig

        x = x.clone()
        token_size = token_size.clone()
        token_ids_new = token_ids.clone()
        rep = rep_for_orig.clone()
        keep = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)

        for source_pos, receiver_pos, _score in pairs:
            source_id = token_ids[source_pos]
            receiver_id = token_ids[receiver_pos]
            source_weight = token_size[source_pos]
            receiver_weight = token_size[receiver_pos]
            total = source_weight + receiver_weight
            x[receiver_pos] = (x[receiver_pos] * receiver_weight + x[source_pos] * source_weight) / total
            token_size[receiver_pos] = total
            keep[source_pos] = False
            rep[rep == source_id] = receiver_id

        return x[keep], token_ids_new[keep], token_size[keep], rep

    def _method_name(self):
        """Human-readable method label for the configured strategy.

        Base path covers A/B/C; DiagnosticTokenMerger overrides to add B2/C2.
        """
        if self.config.strategy == "local_2x2_importance_protected_vec":
            return "B_importance_protected"
        if self.config.strategy == "local_2x2_hybrid_score_vec":
            return "C_hybrid_similarity_importance"
        if self.config.merge_ratio <= 0 and self.config.importance_source != "none":
            return "importance_diagnostic"
        return "A_similarity_only"

    def _info(
        self,
        x_before,
        x_after,
        min_merged,
        max_merged,
        selected_scores=None,
        implementation=None,
        fallback_reason=None,
        importance=None,
        protected_mask=None,
        source_importance=None,
        receiver_importance=None,
        num_candidates=None,
        num_candidate_cells=None,
        num_accepted=None,
        decision_dump=None,
    ):
        before = int(x_before.shape[1])
        after = int(x_after.shape[1])
        ratio = float(after / max(1, before))
        method = self._method_name()
        info = {
            "method": method,
            "strategy": self.config.strategy,
            "receiver": self.config.receiver,
            "importance_source": self.config.importance_source,
            "protect_mode": self.config.protect_mode,
            "protect_ratio": float(self.config.protect_ratio),
            "similarity_threshold": float(self.config.similarity_threshold),
            "dynamic_ratio_mode": self.config.dynamic_ratio_mode,
            "score_alpha": float(self.config.score_alpha),
            "score_beta": float(self.config.score_beta),
            "score_gamma": float(self.config.score_gamma),
            "score_delta": float(self.config.score_delta),
            "lambda_norm": float(self.config.lambda_norm),
            "lambda_motion": float(self.config.lambda_motion),
            "dump_merge_decisions": bool(self.config.dump_merge_decisions),
            "max_decision_dump": int(self.config.max_decision_dump),
            "keep_source": self.config.keep_source,
            "receiver_search": self.config.receiver_search,
            "keep_score_alpha": float(self.config.keep_score_alpha),
            "keep_score_beta": float(self.config.keep_score_beta),
            "similarity_gate_epsilon": float(self.config.similarity_gate_epsilon),
            "direction_by_importance": bool(self.config.direction_by_importance),
            "num_tokens_before": before,
            "num_tokens_after": after,
            "num_merged_min_batch": int(min_merged),
            "num_merged_max_batch": int(max_merged),
            "actual_merge_ratio": float(int(min_merged) / max(1, before)),
            "kept_ratio": ratio,
            "theoretical_attention_ratio": ratio * ratio,
            "num_candidates": int(num_candidates) if num_candidates is not None else None,
            "num_candidate_cells": (
                int(num_candidate_cells) if num_candidate_cells is not None else None
            ),
            "num_accepted": int(num_accepted) if num_accepted is not None else int(min_merged),
            "mean_source_importance": None,
            "mean_receiver_importance": None,
            "protected_token_fraction": None,
            "importance_mean": None,
            "importance_std": None,
            "importance_min": None,
            "importance_max": None,
            "importance_entropy": None,
        }
        if implementation is not None:
            info["implementation"] = implementation
        if fallback_reason is not None:
            info["fallback_reason"] = fallback_reason
        if decision_dump is not None:
            info["decision_dump"] = decision_dump

        if importance is not None:
            score = importance.detach().float()
            score_sum = score.sum(dim=1, keepdim=True).clamp_min(1e-12)
            probs = score / score_sum
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            info["importance_mean"] = float(score.mean().item())
            info["importance_std"] = float(score.std(unbiased=False).item())
            info["importance_min"] = float(score.min().item())
            info["importance_max"] = float(score.max().item())
            info["importance_entropy"] = float(entropy.mean().item())
        if protected_mask is not None:
            info["protected_token_fraction"] = float(protected_mask.float().mean().item())
        if source_importance is not None and source_importance.numel() > 0:
            info["mean_source_importance"] = float(source_importance.detach().float().mean().item())
        if receiver_importance is not None and receiver_importance.numel() > 0:
            info["mean_receiver_importance"] = float(receiver_importance.detach().float().mean().item())

        if selected_scores is None:
            info["mean_selected_similarity"] = None
            info["min_selected_similarity"] = None
            info["max_selected_similarity"] = None
            return info

        if torch.is_tensor(selected_scores):
            if selected_scores.numel() == 0:
                info["mean_selected_similarity"] = None
                info["min_selected_similarity"] = None
                info["max_selected_similarity"] = None
            else:
                scores = selected_scores.detach().float()
                info["mean_selected_similarity"] = float(scores.mean().item())
                info["min_selected_similarity"] = float(scores.min().item())
                info["max_selected_similarity"] = float(scores.max().item())
            return info

        if len(selected_scores) == 0:
            info["mean_selected_similarity"] = None
            info["min_selected_similarity"] = None
            info["max_selected_similarity"] = None
            return info

        scores = torch.tensor(selected_scores, dtype=torch.float32)
        info["mean_selected_similarity"] = float(scores.mean().item())
        info["min_selected_similarity"] = float(scores.min().item())
        info["max_selected_similarity"] = float(scores.max().item())
        return info
