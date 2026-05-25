#!/usr/bin/env python
import argparse
import csv
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
    if value is None or value == "":
        return []
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser("Run a V-JEPA2 encoder-side token merge smoke experiment.")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--model_arch", type=str, default="vit_large_rope")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--merge_layers", type=str, default="8")
    parser.add_argument("--merge_ratio", type=float, default=0.125)
    parser.add_argument("--strategy", type=str, default="local_2x2_same_time")
    parser.add_argument("--receiver", type=str, default="max_norm")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
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
    candidates.extend([
        repo_root / "vjepa2" / "vitl.pt",
        repo_root / "checkpoints" / "vitl.pt",
        repo_root / "checkpoints" / "thinkjepa_jepa_vitl.pt",
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No V-JEPA2 ViT-L checkpoint found")


def load_npz_frames(path, max_frames):
    data = np.load(path, allow_pickle=False)
    frames = data["imgs"]
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if max_frames and frames.shape[0] > max_frames:
        keep = np.linspace(0, frames.shape[0] - 1, max_frames)
        keep = np.clip(np.round(keep), 0, frames.shape[0] - 1).astype(np.int64)
        frames = frames[keep]
    return frames


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
    resized = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    cropped = resized[:, :, top:top + img_size, left:left + img_size]
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32)[None, :, None, None]
    return ((cropped - mean) / std).contiguous()


def load_model(args, checkpoint_path, num_frames):
    from src.models import vision_transformer

    merge_config = {
        "enabled": True,
        "merge_layers": parse_int_list(args.merge_layers),
        "merge_ratio": float(args.merge_ratio),
        "strategy": args.strategy,
        "receiver": args.receiver,
        "restore_dense": True,
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
    clean = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] checkpoint={checkpoint_path}")
    print(f"[INFO] load_state_dict={msg}")
    model.to(args.device)
    model.eval()
    return model


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def timed_forward(model, video, enabled, repeats, warmup, device):
    model.merge_config.enabled = bool(enabled)
    times = []
    peak_memory = []
    output = None
    merge_infos = []
    total_runs = warmup + repeats
    for run_idx in range(total_runs):
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        out, infos = model(video, return_merge_info=True, restore_dense=True)
        synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if run_idx >= warmup:
            times.append(elapsed)
            if str(device).startswith("cuda"):
                peak_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
        output = out.detach().float().cpu()
        merge_infos = infos
    return {
        "output": output,
        "merge_infos": merge_infos,
        "latency_ms": times,
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "peak_memory_mb_mean": float(np.mean(peak_memory)) if peak_memory else None,
    }


def write_plots(summary, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        return

    labels = ["baseline", "after merge"]
    counts = [
        summary["baseline"]["num_tokens"],
        summary["merge"]["num_tokens_after"],
    ]
    plt.figure(figsize=(4.5, 3.2), dpi=160)
    plt.bar(labels, counts, color=["#6b7280", "#2563eb"])
    plt.ylabel("tokens")
    plt.title("Encoder Token Count")
    for idx, value in enumerate(counts):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "token_count_before_after.png")
    plt.close()

    plt.figure(figsize=(4.5, 3.2), dpi=160)
    plt.bar(["baseline", "merge"], [summary["baseline"]["latency_ms_mean"], summary["merge"]["latency_ms_mean"]])
    plt.ylabel("ms")
    plt.title("End-to-end Encoder Latency")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_before_after.png")
    plt.close()


def main():
    args = parse_args()
    repo_root = configure_paths()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_npz_frames(args.npz, args.max_frames)
    frames_norm = preprocess_frames(frames, args.img_size)
    video = frames_norm.permute(1, 0, 2, 3).unsqueeze(0).to(args.device, non_blocking=True)
    checkpoint_path = resolve_checkpoint(repo_root, args.checkpoint)
    model = load_model(args, checkpoint_path, int(frames_norm.shape[0]))

    baseline = timed_forward(model, video, enabled=False, repeats=args.repeats, warmup=args.warmup, device=args.device)
    merged = timed_forward(model, video, enabled=True, repeats=args.repeats, warmup=args.warmup, device=args.device)

    y = baseline["output"]
    y_merge = merged["output"]
    cos = F.cosine_similarity(y, y_merge, dim=-1).reshape(-1)
    diff = y - y_merge
    rel_l2 = float(diff.norm().item() / y.norm().clamp_min(1e-12).item())
    mse = float(diff.square().mean().item())
    merge_info = merged["merge_infos"][0] if merged["merge_infos"] else {}
    num_tokens = int(y.shape[1])
    num_after = int(merge_info.get("num_tokens_after", num_tokens))
    kept_ratio = float(num_after / max(1, num_tokens))

    summary = {
        "input": {
            "npz": args.npz,
            "raw_frames_shape": [int(v) for v in frames.shape],
            "video_shape": [int(v) for v in video.shape],
            "img_size": int(args.img_size),
            "patch_size": int(args.patch_size),
            "tubelet_size": 2,
        },
        "model": {
            "arch": args.model_arch,
            "checkpoint": str(checkpoint_path),
            "output_shape": [int(v) for v in y.shape],
        },
        "baseline": {
            "num_tokens": num_tokens,
            "latency_ms": baseline["latency_ms"],
            "latency_ms_mean": baseline["latency_ms_mean"],
            "latency_ms_std": baseline["latency_ms_std"],
            "peak_memory_mb_mean": baseline["peak_memory_mb_mean"],
        },
        "merge": {
            "enabled": True,
            "strategy": args.strategy,
            "merge_layers": parse_int_list(args.merge_layers),
            "merge_ratio_requested": float(args.merge_ratio),
            "merge_info": merged["merge_infos"],
            "num_tokens_before": num_tokens,
            "num_tokens_after": num_after,
            "tokens_removed": int(num_tokens - num_after),
            "kept_ratio": kept_ratio,
            "theoretical_attention_ratio": kept_ratio * kept_ratio,
            "latency_ms": merged["latency_ms"],
            "latency_ms_mean": merged["latency_ms_mean"],
            "latency_ms_std": merged["latency_ms_std"],
            "peak_memory_mb_mean": merged["peak_memory_mb_mean"],
        },
        "fidelity": {
            "mean_cosine": float(cos.mean().item()),
            "median_cosine": float(cos.median().item()),
            "min_cosine": float(cos.min().item()),
            "relative_l2": rel_l2,
            "mse": mse,
        },
    }
    if baseline["latency_ms_mean"] > 0:
        summary["merge"]["wall_clock_speedup_vs_baseline"] = float(
            baseline["latency_ms_mean"] / merged["latency_ms_mean"]
        )

    with (out_dir / "token_merge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (out_dir / "token_merge_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["baseline_tokens", num_tokens])
        writer.writerow(["merge_tokens_after", num_after])
        writer.writerow(["tokens_removed", num_tokens - num_after])
        writer.writerow(["kept_ratio", kept_ratio])
        writer.writerow(["theoretical_attention_ratio", kept_ratio * kept_ratio])
        writer.writerow(["baseline_latency_ms_mean", baseline["latency_ms_mean"]])
        writer.writerow(["merge_latency_ms_mean", merged["latency_ms_mean"]])
        writer.writerow(["mean_cosine", summary["fidelity"]["mean_cosine"]])
        writer.writerow(["relative_l2", rel_l2])
    write_plots(summary, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
