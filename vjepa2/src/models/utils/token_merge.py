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
    return MergeConfig(
        enabled=bool(config.get("enabled", False)),
        merge_layers=tuple(int(item) for item in layers),
        merge_ratio=float(config.get("merge_ratio", 0.0)),
        strategy=str(config.get("strategy", "local_2x2_same_time")),
        receiver=str(config.get("receiver", "max_norm")),
        restore_dense=bool(config.get("restore_dense", True)),
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
    restored = []
    for batch_idx in range(x.shape[0]):
        pos_by_original_id = torch.empty(num_original_tokens, device=x.device, dtype=torch.long)
        pos_by_original_id[token_ids[batch_idx]] = torch.arange(
            token_ids.shape[1], device=x.device, dtype=torch.long
        )
        gather_pos = pos_by_original_id[rep_for_orig[batch_idx]]
        restored.append(x[batch_idx].index_select(0, gather_pos))
    return torch.stack(restored, dim=0)


class LocalTokenMerger(nn.Module):
    """Training-free local same-time 2x2 token merging for video ViT tokens."""

    def __init__(self, config):
        super().__init__()
        self.config = normalize_merge_config(config)

    @torch.no_grad()
    def forward(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid):
        if self.config.strategy != "local_2x2_same_time":
            raise ValueError(f"Unsupported merge strategy: {self.config.strategy}")
        if self.config.merge_ratio <= 0.0:
            return x, token_ids, token_size, rep_for_orig, self._info(x, x, 0, 0)

        pairs_by_batch = []
        max_merge = 0
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

        if max_merge == 0:
            return x, token_ids, token_size, rep_for_orig, self._info(x, x, 0, 0)

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
        info = self._info(x, x_new, min(num_merged), max(num_merged))
        return x_new, ids_new, size_new, rep_new, info

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

    def _info(self, x_before, x_after, min_merged, max_merged):
        before = int(x_before.shape[1])
        after = int(x_after.shape[1])
        ratio = float(after / max(1, before))
        return {
            "strategy": self.config.strategy,
            "receiver": self.config.receiver,
            "num_tokens_before": before,
            "num_tokens_after": after,
            "num_merged_min_batch": int(min_merged),
            "num_merged_max_batch": int(max_merged),
            "kept_ratio": ratio,
            "theoretical_attention_ratio": ratio * ratio,
        }
