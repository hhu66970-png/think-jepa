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
    parser.add_argument("--receiver", type=str, default="max_norm")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
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


def set_merge_config(model, *, enabled, merge_layers, merge_ratio, strategy, receiver):
    model.merge_config.enabled = bool(enabled)
    model.merge_config.merge_layers = tuple(int(v) for v in merge_layers)
    model.merge_config.merge_ratio = float(merge_ratio)
    model.merge_config.strategy = str(strategy)
    model.merge_config.receiver = str(receiver)
    model.merge_config.restore_dense = True
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
    )
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
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if str(args.device).startswith("cuda")
        else None
    )
    return out.detach().float().cpu(), infos, elapsed_ms, peak_mb


def summarize_times(values):
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def write_csv(rows, out_dir):
    path = out_dir / "full_pipeline_metrics.csv"
    fields = [
        "config",
        "sample",
        "merge_ratio",
        "merge_layers",
        "tokens_before",
        "tokens_after",
        "tokens_removed",
        "latency_ms_mean",
        "latency_ms_median",
        "peak_memory_mb_mean",
        "mean_cosine",
        "relative_l2",
        "sequence_reduced",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


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
        for _ in range(max(1, int(args.repeats))):
            out, infos, elapsed_ms, peak_mb = run_timed_forward(
                model,
                video,
                args,
                enabled=False,
                merge_layers=merge_layers,
                merge_ratio=0.0,
            )
            outputs.append(out)
            latencies.append(elapsed_ms)
            if peak_mb is not None:
                memories.append(peak_mb)
        baseline_outputs.append(outputs[-1])
        row = {
            "config": "baseline",
            "sample": sample_idx,
            "merge_ratio": 0.0,
            "merge_layers": "",
            "tokens_before": int(outputs[-1].shape[1]),
            "tokens_after": int(outputs[-1].shape[1]),
            "tokens_removed": 0,
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_median": float(np.median(latencies)),
            "peak_memory_mb_mean": float(np.mean(memories)) if memories else None,
            "mean_cosine": 1.0,
            "relative_l2": 0.0,
            "sequence_reduced": False,
            "latency": summarize_times(latencies),
        }
        baseline_rows.append(row)
        baseline_latencies.extend(latencies)
        baseline_memories.extend(memories)

    config_summaries = []
    all_rows = list(baseline_rows)
    for ratio in merge_ratios:
        config_name = f"l{'-'.join(str(v) for v in merge_layers)}_r{ratio:g}"
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
            for _ in range(max(1, int(args.repeats))):
                out, infos, elapsed_ms, peak_mb = run_timed_forward(
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
                if peak_mb is not None:
                    memories.append(peak_mb)
            y = baseline_outputs[sample_idx]
            y_merge = outputs[-1]
            cos = F.cosine_similarity(y, y_merge, dim=-1).reshape(-1)
            diff = y - y_merge
            rel_l2 = float(diff.norm().item() / y.norm().clamp_min(1e-12).item())
            info = infos_last[0] if infos_last else {}
            tokens_before = int(info.get("num_tokens_before", y.shape[1]))
            tokens_after = int(info.get("num_tokens_after", y_merge.shape[1]))
            row = {
                "config": config_name,
                "sample": sample_idx,
                "merge_ratio": float(ratio),
                "merge_layers": ",".join(str(v) for v in merge_layers),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_removed": int(tokens_before - tokens_after),
                "latency_ms_mean": float(np.mean(latencies)),
                "latency_ms_median": float(np.median(latencies)),
                "peak_memory_mb_mean": float(np.mean(memories)) if memories else None,
                "mean_cosine": float(cos.mean().item()),
                "median_cosine": float(cos.median().item()),
                "min_cosine": float(cos.min().item()),
                "relative_l2": rel_l2,
                "mse": float(diff.square().mean().item()),
                "sequence_reduced": bool(info.get("sequence_reduced", False)),
                "merge_info": infos_last,
                "latency": summarize_times(latencies),
            }
            config_rows.append(row)
            all_rows.append(row)

        lat_all = [row["latency_ms_mean"] for row in config_rows]
        mem_all = [
            row["peak_memory_mb_mean"]
            for row in config_rows
            if row["peak_memory_mb_mean"] is not None
        ]
        config_summaries.append(
            {
                "config": config_name,
                "merge_ratio": float(ratio),
                "merge_layers": merge_layers,
                "samples": config_rows,
                "aggregate": {
                    "latency": summarize_times(lat_all),
                    "peak_memory_mb_mean": float(np.mean(mem_all)) if mem_all else None,
                    "tokens_before": int(np.mean([r["tokens_before"] for r in config_rows])),
                    "tokens_after": int(np.mean([r["tokens_after"] for r in config_rows])),
                    "tokens_removed": int(np.mean([r["tokens_removed"] for r in config_rows])),
                    "mean_cosine": float(np.mean([r["mean_cosine"] for r in config_rows])),
                    "relative_l2": float(np.mean([r["relative_l2"] for r in config_rows])),
                    "sequence_reduced_all": bool(
                        all(r["sequence_reduced"] for r in config_rows)
                    ),
                },
            }
        )

    baseline_summary = {
        "config": "baseline",
        "samples": baseline_rows,
        "aggregate": {
            "latency": summarize_times(baseline_latencies),
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
        "baseline": baseline_summary,
        "configs": [{"config": "baseline", **baseline_summary}] + config_summaries,
    }
    write_csv(all_rows, out_dir)
    write_plots(summary, out_dir)
    with (out_dir / "full_pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
