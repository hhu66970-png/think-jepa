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


def normalize_merge_config(config):
    if config is None:
        return MergeConfig()
    if isinstance(config, MergeConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"merge_config must be a dict or MergeConfig, got {type(config)!r}")
    layers = config.get("merge_layers", ())
    if isinstance(layers, str):
        layers = tuple(int(item) for item in layers.split(",") if item.strip())
    strategy = str(config.get("strategy", "local_2x2_same_time"))
    layers = tuple(int(item) for item in layers)
    if strategy == "local_2x2_same_time_vec" and len(layers) > 1:
        raise ValueError(
            "local_2x2_same_time_vec currently supports exactly one merge layer. "
            "Use a single layer or implement sparse-state vectorized merging before "
            "enabling multi-layer vectorized merge."
        )
    return MergeConfig(
        enabled=bool(config.get("enabled", False)),
        merge_layers=layers,
        merge_ratio=float(config.get("merge_ratio", 0.0)),
        strategy=strategy,
        receiver=str(config.get("receiver", "max_norm")),
        restore_dense=bool(config.get("restore_dense", True)),
        profile=bool(config.get("profile", False)),
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

    def __init__(self, config):
        super().__init__()
        self.config = normalize_merge_config(config)

    @torch.no_grad()
    def forward(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid):
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
        if self.config.strategy != "local_2x2_same_time_vec":
            raise ValueError(f"Unsupported merge strategy: {self.config.strategy}")

        if self.config.merge_ratio <= 0.0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, implementation="vectorized"
            )

        can_vectorize, fallback_reason = self._can_vectorize_dense_grid(
            x, token_ids, rep_for_orig, int(t_grid), int(h_grid), int(w_grid)
        )
        if not can_vectorize:
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
        best_scores, best_pair_index = similarities.max(dim=-1)
        selected_scores, selected_cells = best_scores.topk(target_merges, dim=1)

        selected_pair_index = best_pair_index.gather(1, selected_cells)
        left_local = pair_left[selected_pair_index]
        right_local = pair_right[selected_pair_index]

        flat_cell_positions = torch.arange(num_tokens, device=x.device, dtype=torch.long)
        flat_cell_positions = flat_cell_positions.reshape(t_grid, h_cells, 2, w_cells, 2)
        flat_cell_positions = flat_cell_positions.permute(0, 1, 3, 2, 4).reshape(num_cells, 4)
        selected_positions = flat_cell_positions[selected_cells]
        left_pos = selected_positions.gather(2, left_local.unsqueeze(-1)).squeeze(-1)
        right_pos = selected_positions.gather(2, right_local.unsqueeze(-1)).squeeze(-1)

        if self.config.receiver == "max_norm":
            token_norm = torch.linalg.norm(x.float(), dim=-1)
            left_norm = token_norm.gather(1, left_pos)
            right_norm = token_norm.gather(1, right_pos)
            left_is_receiver = left_norm >= right_norm
            receiver_pos = torch.where(left_is_receiver, left_pos, right_pos)
            source_pos = torch.where(left_is_receiver, right_pos, left_pos)
        else:
            receiver_pos = torch.minimum(left_pos, right_pos)
            source_pos = torch.maximum(left_pos, right_pos)

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
        num_after = num_tokens - target_merges
        x_new = x_updated[keep].reshape(batch_size, num_after, dim)
        ids_new = token_ids[keep].reshape(batch_size, num_after)
        size_new = token_size_updated[keep].reshape(batch_size, num_after)

        rep_new = rep_for_orig.clone()
        rep_new.scatter_(1, source_ids, receiver_ids)

        info = self._info(
            x,
            x_new,
            target_merges,
            target_merges,
            selected_scores=selected_scores,
            implementation="vectorized",
        )
        return x_new, ids_new, size_new, rep_new, info

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

    def _info(
        self,
        x_before,
        x_after,
        min_merged,
        max_merged,
        selected_scores=None,
        implementation=None,
        fallback_reason=None,
    ):
        before = int(x_before.shape[1])
        after = int(x_after.shape[1])
        ratio = float(after / max(1, before))
        info = {
            "strategy": self.config.strategy,
            "receiver": self.config.receiver,
            "num_tokens_before": before,
            "num_tokens_after": after,
            "num_merged_min_batch": int(min_merged),
            "num_merged_max_batch": int(max_merged),
            "kept_ratio": ratio,
            "theoretical_attention_ratio": ratio * ratio,
        }
        if implementation is not None:
            info["implementation"] = implementation
        if fallback_reason is not None:
            info["fallback_reason"] = fallback_reason

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
