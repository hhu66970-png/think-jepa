#!/usr/bin/env python
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_int_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        "Run a full encoder-side token merge benchmark over a complete NPZ set."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz_glob", type=str)
    source.add_argument("--manifest", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="0 means use all cached frames from each NPZ clip.",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--model_arch", type=str, default="vit_large_rope")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--merge_layers", type=str, default="8")
    parser.add_argument("--merge_ratios", type=str, default="0,0.05,0.10,0.125,0.25")
    parser.add_argument("--strategy", type=str, default="local_2x2_same_time")
    parser.add_argument("--merge_strategy", dest="strategy", type=str)
    parser.add_argument("--receiver", type=str, default="max_norm")
    parser.add_argument("--importance_source", type=str, default="none")
    parser.add_argument("--protect_mode", type=str, default="none")
    parser.add_argument("--protect_ratio", type=float, default=0.0)
    parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    parser.add_argument("--dynamic_ratio_mode", type=str, default="none")
    parser.add_argument("--score_alpha", type=float, default=1.0)
    parser.add_argument("--score_beta", type=float, default=0.3)
    parser.add_argument("--score_gamma", type=float, default=0.5)
    parser.add_argument("--score_delta", type=float, default=0.0)
    parser.add_argument("--lambda_norm", type=float, default=0.3)
    parser.add_argument("--lambda_motion", type=float, default=0.7)
    parser.add_argument("--debug_dump_scores", action="store_true")
    parser.add_argument("--dump_merge_decisions", action="store_true")
    parser.add_argument("--max_decision_dump", type=int, default=8192)
    parser.add_argument(
        "--keep_source",
        type=str,
        default="redundancy",
        choices=["redundancy", "importance", "importance_redundancy", "random"],
    )
    parser.add_argument("--receiver_search", type=str, default="cell", choices=["cell"])
    parser.add_argument("--keep_score_alpha", type=float, default=1.0)
    parser.add_argument("--keep_score_beta", type=float, default=0.0)
    parser.add_argument("--similarity_gate_epsilon", type=float, default=0.01)
    parser.add_argument("--direction_by_importance", action="store_true", default=True)
    parser.add_argument("--no_direction_by_importance", dest="direction_by_importance", action="store_false")
    parser.add_argument("--run_no_merge_baseline", action="store_true")
    parser.add_argument("--run_r0_sanity", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--profile_segments",
        action="store_true",
        help="record patch/pre-merge/merge/post-merge/norm/restore/total timing inside the encoder",
    )
    parser.add_argument(
        "--no_profile_segments",
        dest="profile_segments",
        action="store_false",
        help="disable internal segment timing",
    )
    parser.set_defaults(profile_segments=False)
    return parser.parse_args()


def configure_paths():
    repo_root = Path(__file__).resolve().parents[1]
    vjepa_root = Path(os.environ.get("VJEPA2_ROOT", repo_root / "vjepa2")).resolve()
    for path in (repo_root, repo_root / "cache_train", vjepa_root, vjepa_root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("VJEPA2_ROOT", str(vjepa_root))
    return repo_root


def resolve_checkpoint(repo_root, explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_path = os.environ.get("THINKJEPA_JEPA_VITL_PT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            repo_root / "vjepa2" / "vitl.pt",
            repo_root / "checkpoints" / "vitl.pt",
            repo_root / "checkpoints" / "thinkjepa_jepa_vitl.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No V-JEPA2 ViT-L checkpoint found")


def list_npz_files(args):
    if args.manifest:
        paths = []
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    else:
        paths = sorted(glob.glob(args.npz_glob, recursive=True))
    unique = []
    seen = set()
    for item in paths:
        item = os.path.abspath(os.path.expanduser(item))
        if item not in seen:
            seen.add(item)
            unique.append(item)
    if not unique:
        raise FileNotFoundError("No NPZ inputs found for full pipeline benchmark")
    return unique


def load_npz_frames(path, max_frames):
    data = np.load(path, allow_pickle=False)
    frames = data["imgs"]
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if max_frames and max_frames > 0 and frames.shape[0] > max_frames:
        keep = np.linspace(0, frames.shape[0] - 1, max_frames)
        keep = np.clip(np.round(keep), 0, frames.shape[0] - 1).astype(np.int64)
        frames = frames[keep]
    else:
        keep = np.arange(frames.shape[0], dtype=np.int64)
    meta = {
        "raw_frames_shape": [int(v) for v in frames.shape],
        "selected_npz_indices": [int(v) for v in keep.tolist()],
    }
    for key in ("video_path", "path", "lang_instruct"):
        if key in data.files:
            try:
                meta[key] = str(data[key].item())
            except Exception:
                meta[key] = str(data[key])
    return frames, meta


def preprocess_frames(frames_np, img_size):
    frames = torch.from_numpy(frames_np).float()
    if frames.max() > 1.5:
        frames = frames / 255.0
    frames = frames.permute(0, 3, 1, 2).contiguous()
    _, _, height, width = frames.shape
    short_side = int(256.0 / 224.0 * img_size)
    if height <= width:
        new_h = short_side
        new_w = int(round(width * short_side / height))
    else:
        new_w = short_side
        new_h = int(round(height * short_side / width))
    resized = F.interpolate(
        frames, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    cropped = resized[:, :, top : top + img_size, left : left + img_size]
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32)[None, :, None, None]
    return ((cropped - mean) / std).contiguous()


def load_model(args, checkpoint_path, num_frames, merge_layers):
    from src.models import vision_transformer

    merge_config = {
        "enabled": True,
        "merge_layers": tuple(int(v) for v in merge_layers),
        "merge_ratio": 0.0,
        "strategy": args.strategy,
        "receiver": args.receiver,
        "restore_dense": True,
        "profile": bool(args.profile_segments),
        "importance_source": args.importance_source,
        "protect_mode": args.protect_mode,
        "protect_ratio": float(args.protect_ratio),
        "similarity_threshold": float(args.similarity_threshold),
        "dynamic_ratio_mode": args.dynamic_ratio_mode,
        "score_alpha": float(args.score_alpha),
        "score_beta": float(args.score_beta),
        "score_gamma": float(args.score_gamma),
        "score_delta": float(args.score_delta),
        "lambda_norm": float(args.lambda_norm),
        "lambda_motion": float(args.lambda_motion),
        "debug_dump_scores": bool(args.debug_dump_scores),
        "dump_merge_decisions": bool(args.dump_merge_decisions),
        "max_decision_dump": int(args.max_decision_dump),
        "keep_source": str(args.keep_source),
        "receiver_search": str(args.receiver_search),
        "keep_score_alpha": float(args.keep_score_alpha),
        "keep_score_beta": float(args.keep_score_beta),
        "similarity_gate_epsilon": float(args.similarity_gate_epsilon),
        "direction_by_importance": bool(args.direction_by_importance),
    }
    model = getattr(vision_transformer, args.model_arch)(
        img_size=(args.img_size, args.img_size),
        num_frames=num_frames,
        patch_size=args.patch_size,
        tubelet_size=2,
        out_layers=None,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        merge_config=merge_config,
    )
    try:
        blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(blob, dict) and "encoder" in blob:
        state = blob["encoder"]
    elif isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    else:
        state = blob
    clean = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] checkpoint={checkpoint_path}", flush=True)
    print(f"[INFO] load_state_dict={msg}", flush=True)
    model.to(args.device)
    model.eval()
    return model


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def set_merge_config(model, *, enabled, merge_layers, merge_ratio, strategy, receiver, profile):
    vectorized_strategies = {
        "local_2x2_same_time_vec",
        "local_2x2_importance_protected_vec",
        "local_2x2_hybrid_score_vec",
        "local_keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec",
    }
    if str(strategy) in vectorized_strategies and len(tuple(merge_layers)) > 1:
        raise ValueError(
            f"{strategy} currently supports exactly one merge layer. "
            "Use a single layer for vectorized benchmarking."
        )
    model.merge_config.enabled = bool(enabled)
    model.merge_config.merge_layers = tuple(int(v) for v in merge_layers)
    model.merge_config.merge_ratio = float(merge_ratio)
    model.merge_config.strategy = str(strategy)
    model.merge_config.receiver = str(receiver)
    model.merge_config.restore_dense = True
    model.merge_config.profile = bool(profile)
    if getattr(model, "token_merger", None) is not None:
        model.token_merger.config = model.merge_config


def set_merge_experiment_fields(model, args):
    model.merge_config.importance_source = str(args.importance_source)
    model.merge_config.protect_mode = str(args.protect_mode)
    model.merge_config.protect_ratio = float(args.protect_ratio)
    model.merge_config.similarity_threshold = float(args.similarity_threshold)
    model.merge_config.dynamic_ratio_mode = str(args.dynamic_ratio_mode)
    model.merge_config.score_alpha = float(args.score_alpha)
    model.merge_config.score_beta = float(args.score_beta)
    model.merge_config.score_gamma = float(args.score_gamma)
    model.merge_config.score_delta = float(args.score_delta)
    model.merge_config.lambda_norm = float(args.lambda_norm)
    model.merge_config.lambda_motion = float(args.lambda_motion)
    model.merge_config.debug_dump_scores = bool(args.debug_dump_scores)
    model.merge_config.dump_merge_decisions = bool(args.dump_merge_decisions)
    model.merge_config.max_decision_dump = int(args.max_decision_dump)
    model.merge_config.keep_source = str(args.keep_source)
    model.merge_config.receiver_search = str(args.receiver_search)
    model.merge_config.keep_score_alpha = float(args.keep_score_alpha)
    model.merge_config.keep_score_beta = float(args.keep_score_beta)
    model.merge_config.similarity_gate_epsilon = float(args.similarity_gate_epsilon)
    model.merge_config.direction_by_importance = bool(args.direction_by_importance)
    if getattr(model, "token_merger", None) is not None:
        model.token_merger.config = model.merge_config


@torch.no_grad()
def run_timed_forward(model, video, args, *, enabled, merge_layers, merge_ratio):
    set_merge_config(
        model,
        enabled=enabled,
        merge_layers=merge_layers,
        merge_ratio=merge_ratio,
        strategy=args.strategy,
        receiver=args.receiver,
        profile=bool(args.profile_segments),
    )
    set_merge_experiment_fields(model, args)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(args.device)
    start = time.perf_counter()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(args.amp and str(args.device).startswith("cuda")),
    ):
        out, infos = model(video, return_merge_info=True, restore_dense=True)
    synchronize(args.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    profile = dict(getattr(model, "last_forward_profile", {}) or {})
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if str(args.device).startswith("cuda")
        else None
    )
    return out.detach().float().cpu(), infos, elapsed_ms, peak_mb, profile


def summarize_times(values):
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


PROFILE_FIELDS = [
    "patch_embed_ms",
    "pre_merge_blocks_ms",
    "merge_module_ms",
    "post_merge_blocks_ms",
    "norm_ms",
    "restore_dense_ms",
    "total_profiled_ms",
]


def summarize_profile(profiles):
    out = {}
    for key in PROFILE_FIELDS:
        vals = [float(item.get(key, 0.0)) for item in profiles if item]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_median"] = float(np.median(vals)) if vals else 0.0
    return out


def write_csv(rows, out_dir):
    path = out_dir / "full_pipeline_metrics.csv"
    fields = [
        "config",
        "sample",
        "method",
        "strategy",
        "importance_source",
        "protect_mode",
        "protect_ratio",
        "similarity_threshold",
        "dynamic_ratio_mode",
        "score_alpha",
        "score_beta",
        "score_gamma",
        "score_delta",
        "lambda_norm",
        "lambda_motion",
        "keep_source",
        "receiver_search",
        "keep_score_alpha",
        "keep_score_beta",
        "similarity_gate_epsilon",
        "direction_by_importance",
        "merge_ratio",
        "merge_layers",
        "actual_merge_ratio",
        "tokens_before",
        "tokens_after",
        "tokens_removed",
        "latency_ms_mean",
        "latency_ms_median",
        "patch_embed_ms_mean",
        "pre_merge_blocks_ms_mean",
        "merge_module_ms_mean",
        "post_merge_blocks_ms_mean",
        "norm_ms_mean",
        "restore_dense_ms_mean",
        "total_profiled_ms_mean",
        "peak_memory_mb_mean",
        "mean_cosine",
        "median_cosine",
        "p10_cosine",
        "p1_cosine",
        "min_cosine",
        "relative_l2",
        "mse",
        "max_token_error",
        "p95_token_error",
        "p99_token_error",
        "mean_selected_similarity",
        "min_selected_similarity",
        "max_selected_similarity",
        "mean_source_importance",
        "mean_receiver_importance",
        "protected_token_fraction",
        "num_candidates",
        "num_candidate_cells",
        "num_accepted",
        "importance_mean",
        "importance_std",
        "importance_min",
        "importance_max",
        "importance_entropy",
        "sequence_reduced",
        "final_implementation",
        "any_fallback",
        "fallback_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_value(tensor, batch_idx, item_idx):
    if tensor is None:
        return None
    value = tensor[batch_idx, item_idx]
    if value.dtype.is_floating_point:
        return float(value.detach().cpu().item())
    return int(value.detach().cpu().item())


def decision_dump_rows(config_name, sample_idx, repeat_idx, merge_ratio, info):
    dump = (info or {}).get("decision_dump")
    if not dump:
        return []
    cell_id = dump.get("cell_id")
    if cell_id is None:
        return []
    rows = []
    batch_size, count = cell_id.shape
    layer = int(info.get("layer", -1))
    strategy = str(info.get("strategy", ""))
    method = str(info.get("method", ""))
    for batch_idx in range(batch_size):
        for item_idx in range(count):
            rows.append(
                {
                    "config": config_name,
                    "sample": int(sample_idx),
                    "repeat": int(repeat_idx),
                    "merge_ratio": float(merge_ratio),
                    "batch": int(batch_idx),
                    "layer": layer,
                    "method": method,
                    "strategy": strategy,
                    "cell_id": tensor_value(cell_id, batch_idx, item_idx),
                    "time_id": tensor_value(dump.get("time_id"), batch_idx, item_idx),
                    "cell_h": tensor_value(dump.get("cell_h"), batch_idx, item_idx),
                    "cell_w": tensor_value(dump.get("cell_w"), batch_idx, item_idx),
                    "source_pos": tensor_value(dump.get("source_pos"), batch_idx, item_idx),
                    "receiver_pos": tensor_value(dump.get("receiver_pos"), batch_idx, item_idx),
                    "source_token_id": tensor_value(dump.get("source_token_id"), batch_idx, item_idx),
                    "receiver_token_id": tensor_value(dump.get("receiver_token_id"), batch_idx, item_idx),
                    "source_local_id": tensor_value(dump.get("source_local_id"), batch_idx, item_idx),
                    "receiver_local_id": tensor_value(dump.get("receiver_local_id"), batch_idx, item_idx),
                    "selected_similarity": tensor_value(
                        dump.get("selected_similarity"), batch_idx, item_idx
                    ),
                    "best_similarity": tensor_value(
                        dump.get("best_similarity"), batch_idx, item_idx
                    ),
                    "second_best_similarity": tensor_value(
                        dump.get("second_best_similarity"), batch_idx, item_idx
                    ),
                    "source_importance": tensor_value(
                        dump.get("source_importance"), batch_idx, item_idx
                    ),
                    "receiver_importance": tensor_value(
                        dump.get("receiver_importance"), batch_idx, item_idx
                    ),
                }
            )
    return rows


def strip_decision_dump(info):
    if not isinstance(info, dict):
        return info
    return {key: value for key, value in info.items() if key != "decision_dump"}


def strip_decision_dumps(infos):
    return [strip_decision_dump(item) for item in (infos or [])]


def method_name_for_strategy(strategy):
    if strategy == "local_2x2_importance_protected_vec":
        return "B_importance_protected"
    if strategy == "local_2x2_hybrid_score_vec":
        return "C_hybrid_similarity_importance"
    if strategy == "local_keep_then_merge_vec":
        return "B2_keep_then_merge"
    if strategy == "local_2x2_similarity_gated_importance_vec":
        return "C2_similarity_gated_importance"
    return "A_similarity_only"


def strategy_label(strategy):
    mapping = {
        "local_2x2_same_time": "same_time_python",
        "local_2x2_same_time_python": "same_time_python",
        "local_2x2_same_time_vec": "same_time_vec",
        "local_2x2_importance_protected_vec": "protect_vec",
        "local_2x2_hybrid_score_vec": "hybrid_vec",
        "local_keep_then_merge_vec": "keep_then_merge_vec",
        "local_2x2_similarity_gated_importance_vec": "similarity_gated_vec",
    }
    return mapping.get(str(strategy), str(strategy).replace("/", "_"))


def make_config_name(args, merge_layers, ratio):
    layers = "-".join(str(v) for v in merge_layers)
    method = method_name_for_strategy(str(args.strategy))
    strategy = strategy_label(args.strategy)
    if float(ratio) == 0.0 and bool(args.run_r0_sanity):
        return f"B0_r0_sanity_{strategy}_l{layers}_{args.img_size}"
    if method.startswith("A_"):
        return f"A_sim_{strategy}_l{layers}_r{float(ratio):g}_{args.img_size}"
    if method.startswith("B_"):
        return (
            f"B_protect_{args.importance_source}_{args.protect_mode}_"
            f"l{layers}_r{float(ratio):g}_{args.img_size}"
        )
    if method.startswith("B2_"):
        return (
            f"B2_keep_{args.keep_source}_{args.receiver_search}_"
            f"l{layers}_r{float(ratio):g}_{args.img_size}"
        )
    if method.startswith("C2_"):
        return (
            f"C2_gated_{args.importance_source}_eps{float(args.similarity_gate_epsilon):g}_"
            f"l{layers}_r{float(ratio):g}_{args.img_size}"
        )
    return (
        f"C_hybrid_{args.importance_source}_{args.protect_mode}_"
        f"l{layers}_r{float(ratio):g}_{args.img_size}"
    )


def experiment_fields(args, info=None):
    method = method_name_for_strategy(str(args.strategy))
    if info:
        method = str(info.get("method", method))
    return {
        "method": method,
        "strategy": str(args.strategy),
        "importance_source": str(args.importance_source),
        "protect_mode": str(args.protect_mode),
        "protect_ratio": float(args.protect_ratio),
        "similarity_threshold": float(args.similarity_threshold),
        "dynamic_ratio_mode": str(args.dynamic_ratio_mode),
        "score_alpha": float(args.score_alpha),
        "score_beta": float(args.score_beta),
        "score_gamma": float(args.score_gamma),
        "score_delta": float(args.score_delta),
        "lambda_norm": float(args.lambda_norm),
        "lambda_motion": float(args.lambda_motion),
        "keep_source": str(args.keep_source),
        "receiver_search": str(args.receiver_search),
        "keep_score_alpha": float(args.keep_score_alpha),
        "keep_score_beta": float(args.keep_score_beta),
        "similarity_gate_epsilon": float(args.similarity_gate_epsilon),
        "direction_by_importance": bool(args.direction_by_importance),
    }


def write_plots(summary, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        return

    configs = [item for item in summary["configs"] if item["config"] != "baseline"]
    labels = [item["config"] for item in configs]
    latencies = [item["aggregate"]["latency"]["mean_ms"] for item in configs]
    baseline_latency = summary["baseline"]["aggregate"]["latency"]["mean_ms"]
    speedups = [baseline_latency / max(value, 1e-9) for value in latencies]
    cosines = [item["aggregate"].get("mean_cosine", 1.0) for item in configs]
    tokens_after = [item["aggregate"].get("tokens_after", 0) for item in configs]

    plt.figure(figsize=(max(7, len(labels) * 1.25), 4.2), dpi=160)
    plt.axhline(1.0, color="#6b7280", linewidth=1, linestyle="--")
    plt.bar(labels, speedups, color="#2563eb")
    plt.ylabel("speedup vs baseline")
    plt.xticks(rotation=30, ha="right")
    plt.title("Full Encoder Pipeline Speedup")
    plt.tight_layout()
    plt.savefig(out_dir / "full_pipeline_speedup.png")
    plt.close()

    plt.figure(figsize=(max(7, len(labels) * 1.25), 4.2), dpi=160)
    plt.plot(tokens_after, cosines, marker="o", color="#059669")
    for x, y, label in zip(tokens_after, cosines, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))
    plt.xlabel("tokens after merge")
    plt.ylabel("mean cosine vs baseline")
    plt.title("Token Count vs Feature Fidelity")
    plt.tight_layout()
    plt.savefig(out_dir / "full_pipeline_fidelity_tradeoff.png")
    plt.close()


def main():
    args = parse_args()
    repo_root = configure_paths()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list_npz_files(args)
    merge_layers = parse_int_list(args.merge_layers)
    merge_ratios = parse_float_list(args.merge_ratios)
    if not merge_layers:
        raise ValueError("--merge_layers must contain at least one layer index")
    if not merge_ratios:
        raise ValueError("--merge_ratios must contain at least one ratio")
    if bool(args.run_r0_sanity) and all(abs(float(r)) > 1e-12 for r in merge_ratios):
        merge_ratios = [0.0] + merge_ratios

    videos = []
    samples = []
    for path in npz_files:
        frames, meta = load_npz_frames(path, args.max_frames)
        frames_norm = preprocess_frames(frames, args.img_size)
        video = frames_norm.permute(1, 0, 2, 3).unsqueeze(0).to(
            args.device, non_blocking=True
        )
        videos.append(video)
        meta.update(
            {
                "npz": path,
                "video_shape": [int(v) for v in video.shape],
                "num_frames_used": int(video.shape[2]),
            }
        )
        samples.append(meta)

    num_frames_set = {int(video.shape[2]) for video in videos}
    if len(num_frames_set) != 1:
        raise ValueError(f"All inputs must use the same frame count, got {num_frames_set}")
    checkpoint_path = resolve_checkpoint(repo_root, args.checkpoint)
    model = load_model(args, checkpoint_path, int(videos[0].shape[2]), merge_layers)

    baseline_outputs = []
    baseline_rows = []
    baseline_latencies = []
    baseline_memories = []
    per_run_rows = []
    decision_rows = []
    print("[RUN] baseline full encoder pipeline", flush=True)
    for sample_idx, video in enumerate(videos):
        for _ in range(max(0, int(args.warmup))):
            run_timed_forward(
                model,
                video,
                args,
                enabled=False,
                merge_layers=merge_layers,
                merge_ratio=0.0,
            )
        outputs = []
        latencies = []
        memories = []
        profiles = []
        for _ in range(max(1, int(args.repeats))):
            repeat_idx = len(latencies)
            out, infos, elapsed_ms, peak_mb, profile = run_timed_forward(
                model,
                video,
                args,
                enabled=False,
                merge_layers=merge_layers,
                merge_ratio=0.0,
            )
            outputs.append(out)
            latencies.append(elapsed_ms)
            profiles.append(profile)
            if peak_mb is not None:
                memories.append(peak_mb)
            per_run_rows.append(
                {
                    "config": "baseline",
                    "sample": sample_idx,
                    "repeat": repeat_idx,
                    "enabled": False,
                    "merge_ratio": 0.0,
                    "latency_ms": float(elapsed_ms),
                    "peak_memory_mb": float(peak_mb) if peak_mb is not None else None,
                    "profile": profile,
                    "merge_info": strip_decision_dumps(infos),
                }
            )
            if bool(args.dump_merge_decisions):
                for info in infos:
                    decision_rows.extend(
                        decision_dump_rows("baseline", sample_idx, repeat_idx, 0.0, info)
                    )
        profile_summary = summarize_profile(profiles)
        baseline_outputs.append(outputs[-1])
        row = {
            "config": "baseline",
            "sample": sample_idx,
            **{
                "method": "B0_no_merge",
                "strategy": "none",
                "importance_source": "none",
                "protect_mode": "none",
                "protect_ratio": 0.0,
                "similarity_threshold": -1.0,
                "dynamic_ratio_mode": "none",
                "score_alpha": "",
                "score_beta": "",
                "score_gamma": "",
                "score_delta": "",
                "lambda_norm": "",
                "lambda_motion": "",
                "keep_source": "",
                "receiver_search": "",
                "keep_score_alpha": "",
                "keep_score_beta": "",
                "similarity_gate_epsilon": "",
                "direction_by_importance": "",
                "actual_merge_ratio": 0.0,
            },
            "merge_ratio": 0.0,
            "merge_layers": "",
            "tokens_before": int(outputs[-1].shape[1]),
            "tokens_after": int(outputs[-1].shape[1]),
            "tokens_removed": 0,
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_median": float(np.median(latencies)),
            **profile_summary,
            "peak_memory_mb_mean": float(np.mean(memories)) if memories else None,
            "mean_cosine": 1.0,
            "median_cosine": 1.0,
            "p10_cosine": 1.0,
            "p1_cosine": 1.0,
            "min_cosine": 1.0,
            "relative_l2": 0.0,
            "mse": 0.0,
            "max_token_error": 0.0,
            "p95_token_error": 0.0,
            "p99_token_error": 0.0,
            "mean_selected_similarity": "",
            "min_selected_similarity": "",
            "max_selected_similarity": "",
            "mean_source_importance": "",
            "mean_receiver_importance": "",
            "protected_token_fraction": "",
                "num_candidates": "",
                "num_candidate_cells": "",
                "num_accepted": "",
            "importance_mean": "",
            "importance_std": "",
            "importance_min": "",
            "importance_max": "",
            "importance_entropy": "",
            "sequence_reduced": False,
            "latency": summarize_times(latencies),
            "profile": profile_summary,
        }
        baseline_rows.append(row)
        baseline_latencies.extend(latencies)
        baseline_memories.extend(memories)

    config_summaries = []
    all_rows = list(baseline_rows)
    for ratio in merge_ratios:
        config_name = make_config_name(args, merge_layers, ratio)
        print(f"[RUN] merge config {config_name}", flush=True)
        config_rows = []
        for sample_idx, video in enumerate(videos):
            for _ in range(max(0, int(args.warmup))):
                run_timed_forward(
                    model,
                    video,
                    args,
                    enabled=True,
                    merge_layers=merge_layers,
                    merge_ratio=ratio,
                )
            outputs = []
            infos_last = []
            latencies = []
            memories = []
            profiles = []
            for _ in range(max(1, int(args.repeats))):
                repeat_idx = len(latencies)
                out, infos, elapsed_ms, peak_mb, profile = run_timed_forward(
                    model,
                    video,
                    args,
                    enabled=True,
                    merge_layers=merge_layers,
                    merge_ratio=ratio,
                )
                outputs.append(out)
                infos_last = infos
                latencies.append(elapsed_ms)
                profiles.append(profile)
                if peak_mb is not None:
                    memories.append(peak_mb)
                per_run_rows.append(
                    {
                        "config": config_name,
                        "sample": sample_idx,
                        "repeat": repeat_idx,
                        "enabled": True,
                        "merge_ratio": float(ratio),
                        "latency_ms": float(elapsed_ms),
                        "peak_memory_mb": float(peak_mb) if peak_mb is not None else None,
                        "profile": profile,
                        "merge_info": strip_decision_dumps(infos),
                    }
                )
                if bool(args.dump_merge_decisions):
                    for info in infos:
                        decision_rows.extend(
                            decision_dump_rows(config_name, sample_idx, repeat_idx, ratio, info)
                        )
            profile_summary = summarize_profile(profiles)
            y = baseline_outputs[sample_idx]
            y_merge = outputs[-1]
            cos = F.cosine_similarity(y, y_merge, dim=-1).reshape(-1)
            diff = y - y_merge
            token_error = diff.norm(dim=-1).reshape(-1)
            rel_l2 = float(diff.norm().item() / y.norm().clamp_min(1e-12).item())
            info = strip_decision_dump(infos_last[-1]) if infos_last else {}
            fallback_reasons = [
                str(item.get("fallback_reason"))
                for item in infos_last
                if item.get("fallback_reason") is not None
            ]
            tokens_before = int(info.get("num_tokens_before", y.shape[1]))
            tokens_after = int(info.get("num_tokens_after", y_merge.shape[1]))
            row = {
                "config": config_name,
                "sample": sample_idx,
                **experiment_fields(args, info),
                "merge_ratio": float(ratio),
                "merge_layers": ",".join(str(v) for v in merge_layers),
                "actual_merge_ratio": float(info.get("actual_merge_ratio", 0.0)),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_removed": int(tokens_before - tokens_after),
                "latency_ms_mean": float(np.mean(latencies)),
                "latency_ms_median": float(np.median(latencies)),
                **profile_summary,
                "peak_memory_mb_mean": float(np.mean(memories)) if memories else None,
                "mean_cosine": float(cos.mean().item()),
                "median_cosine": float(cos.median().item()),
                "p10_cosine": float(torch.quantile(cos, 0.10).item()),
                "p1_cosine": float(torch.quantile(cos, 0.01).item()),
                "min_cosine": float(cos.min().item()),
                "relative_l2": rel_l2,
                "mse": float(diff.square().mean().item()),
                "max_token_error": float(token_error.max().item()),
                "p95_token_error": float(torch.quantile(token_error, 0.95).item()),
                "p99_token_error": float(torch.quantile(token_error, 0.99).item()),
                "mean_selected_similarity": info.get("mean_selected_similarity"),
                "min_selected_similarity": info.get("min_selected_similarity"),
                "max_selected_similarity": info.get("max_selected_similarity"),
                "mean_source_importance": info.get("mean_source_importance"),
                "mean_receiver_importance": info.get("mean_receiver_importance"),
                "protected_token_fraction": info.get("protected_token_fraction"),
                "num_candidates": info.get("num_candidates"),
                "num_candidate_cells": info.get("num_candidate_cells"),
                "num_accepted": info.get("num_accepted"),
                "importance_mean": info.get("importance_mean"),
                "importance_std": info.get("importance_std"),
                "importance_min": info.get("importance_min"),
                "importance_max": info.get("importance_max"),
                "importance_entropy": info.get("importance_entropy"),
                "sequence_reduced": bool(info.get("sequence_reduced", False)),
                "final_implementation": str(info.get("implementation", "")),
                "any_fallback": bool(fallback_reasons),
                "fallback_reasons": ";".join(fallback_reasons),
                "final_merge_info": info,
                "merge_info": strip_decision_dumps(infos_last),
                "latency": summarize_times(latencies),
                "profile": profile_summary,
            }
            config_rows.append(row)
            all_rows.append(row)

        lat_all = [row["latency_ms_mean"] for row in config_rows]
        mem_all = [
            row["peak_memory_mb_mean"]
            for row in config_rows
            if row["peak_memory_mb_mean"] is not None
        ]
        profile_aggregate = {
            key: float(np.mean([row.get(key, 0.0) for row in config_rows]))
            for key in [f"{field}_mean" for field in PROFILE_FIELDS]
        }
        config_summaries.append(
            {
                "config": config_name,
                "merge_ratio": float(ratio),
                "merge_layers": merge_layers,
                "samples": config_rows,
                "aggregate": {
                    "latency": summarize_times(lat_all),
                    "profile": profile_aggregate,
                    "peak_memory_mb_mean": float(np.mean(mem_all)) if mem_all else None,
                    "tokens_before": int(np.mean([r["tokens_before"] for r in config_rows])),
                    "tokens_after": int(np.mean([r["tokens_after"] for r in config_rows])),
                    "tokens_removed": int(np.mean([r["tokens_removed"] for r in config_rows])),
                    "mean_cosine": float(np.mean([r["mean_cosine"] for r in config_rows])),
                    "median_cosine": float(np.mean([r["median_cosine"] for r in config_rows])),
                    "p10_cosine": float(np.mean([r["p10_cosine"] for r in config_rows])),
                    "p1_cosine": float(np.mean([r["p1_cosine"] for r in config_rows])),
                    "relative_l2": float(np.mean([r["relative_l2"] for r in config_rows])),
                    "mse": float(np.mean([r["mse"] for r in config_rows])),
                    "sequence_reduced_all": bool(
                        all(r["sequence_reduced"] for r in config_rows)
                    ),
                    "any_fallback": bool(any(r.get("any_fallback") for r in config_rows)),
                    "final_implementations": sorted(
                        {str(r.get("final_implementation", "")) for r in config_rows}
                    ),
                },
            }
        )

    baseline_summary = {
        "config": "baseline",
        "samples": baseline_rows,
        "aggregate": {
            "latency": summarize_times(baseline_latencies),
            "profile": {
                key: float(np.mean([row.get(key, 0.0) for row in baseline_rows]))
                for key in [f"{field}_mean" for field in PROFILE_FIELDS]
            },
            "peak_memory_mb_mean": (
                float(np.mean(baseline_memories)) if baseline_memories else None
            ),
            "tokens": int(baseline_outputs[0].shape[1]),
        },
    }
    summary = {
        "experiment": "thinkjepa_encoder_token_merge_full_pipeline",
        "notes": [
            "Uses all selected NPZ samples, not a single-sample smoke test.",
            "Reads npz['imgs'] and reruns the dense V-JEPA encoder online.",
            "Does not use cached npz['vjepa_feats'] as encoder output.",
            "Compares no-merge baseline with encoder-side local token merge.",
            "dynamic_ratio_mode, score_delta, and debug_dump_scores are metadata-only in this implementation.",
            "Use final_implementation and any_fallback fields to distinguish vectorized runs from Python fallback.",
            "Decision dumps are optional debug artifacts and are written outside the encoder hot path.",
        ],
        "input": {
            "num_samples": len(samples),
            "samples": samples,
            "img_size": int(args.img_size),
            "patch_size": int(args.patch_size),
            "tubelet_size": 2,
        },
        "model": {
            "arch": args.model_arch,
            "checkpoint": str(checkpoint_path),
            "amp": bool(args.amp),
        },
        "merge_experiment": {
            "strategy": str(args.strategy),
            "importance_source": str(args.importance_source),
            "protect_mode": str(args.protect_mode),
            "protect_ratio": float(args.protect_ratio),
            "similarity_threshold": float(args.similarity_threshold),
            "dynamic_ratio_mode": str(args.dynamic_ratio_mode),
            "score_alpha": float(args.score_alpha),
            "score_beta": float(args.score_beta),
            "score_gamma": float(args.score_gamma),
            "score_delta": float(args.score_delta),
            "lambda_norm": float(args.lambda_norm),
            "lambda_motion": float(args.lambda_motion),
            "dump_merge_decisions": bool(args.dump_merge_decisions),
            "max_decision_dump": int(args.max_decision_dump),
            "keep_source": str(args.keep_source),
            "receiver_search": str(args.receiver_search),
            "keep_score_alpha": float(args.keep_score_alpha),
            "keep_score_beta": float(args.keep_score_beta),
            "similarity_gate_epsilon": float(args.similarity_gate_epsilon),
            "direction_by_importance": bool(args.direction_by_importance),
        },
        "baseline": baseline_summary,
        "configs": [{"config": "baseline", **baseline_summary}] + config_summaries,
    }
    write_csv(all_rows, out_dir)
    write_jsonl(per_run_rows, out_dir / "per_run_profile.jsonl")
    if decision_rows:
        write_jsonl(decision_rows, out_dir / "merge_decisions.jsonl")
    write_plots(summary, out_dir)
    with (out_dir / "full_pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
