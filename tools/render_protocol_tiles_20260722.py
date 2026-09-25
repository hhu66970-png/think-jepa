#!/usr/bin/env python
"""Render truthful, label-free panel tiles from the protocol-locked dump."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


GREEN = "#36B98B"


def norm01(values, low=None, high=None):
    if low is None:
        low = float(np.min(values))
    if high is None:
        high = float(np.max(values))
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def motion_change(array, t, h, w):
    grid = array.astype(np.float32, copy=False).reshape(t, h, w, -1)
    motion = np.zeros((t, h, w), dtype=np.float32)
    motion[1:] = np.linalg.norm(grid[1:] - grid[:-1], axis=-1)
    return norm01(motion)


def preservation(group_ids, t, h, w):
    _, inverse, counts = np.unique(
        group_ids.astype(np.int64, copy=False), return_inverse=True, return_counts=True
    )
    return (1.0 / counts[inverse]).reshape(t, h, w).astype(np.float32)


def upsample(values, size=700, blur=5.0):
    image = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_input(frame, path, size=700):
    Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS).save(path)


def save_overlay(gray, detail, motion, path, detail_limits):
    detail_up = upsample(norm01(detail, *detail_limits))
    motion_up = upsample(norm01(motion), blur=3.0)
    contour_level = float(np.quantile(motion_up, 0.82))
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=200)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(detail_up, cmap="inferno", alpha=0.72, vmin=0, vmax=1)
    ax.contour(motion_up, levels=[contour_level], colors=GREEN,
               linewidths=1.35, alpha=0.95)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_motion(gray, motion, path):
    motion_up = upsample(norm01(motion), blur=3.0)
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=200)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(motion_up, cmap="viridis", alpha=0.74, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", default="1082,1013")
    args = parser.parse_args()

    dump = Path(args.dump)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [item.strip() for item in args.clips.split(",") if item.strip()]
    methods = ["kbsm", "pitome", "wam_lam100"]

    for cid in clips:
        data = np.load(dump / f"feats_{cid}.npz", allow_pickle=True)
        t, h, w = int(data["t"]), int(data["h"]), int(data["w"])
        relevance_layer = int(data["relevance_layer"])
        motion_all = motion_change(data[f"dense_L{relevance_layer}"], t, h, w)
        motion_mean = motion_all[1:].mean(axis=0)
        token_t = int(data["token_t"])
        motion_slice = motion_all[token_t]
        frame = data["input_frame"]
        frame_700 = np.asarray(
            Image.fromarray(frame).resize((700, 700), Image.Resampling.LANCZOS)
        )
        gray = np.asarray(Image.fromarray(frame_700).convert("L"))
        save_input(frame, out / f"clip_{cid}_input.png")
        save_motion(gray, motion_slice, out / f"clip_{cid}_motion.png")

        mean_maps = {
            method: preservation(data[f"{method}_gids"], t, h, w).mean(axis=0)
            for method in methods
        }
        all_detail = np.concatenate([value.reshape(-1) for value in mean_maps.values()])
        detail_limits = (
            float(np.quantile(all_detail, 0.01)),
            float(np.quantile(all_detail, 0.99)),
        )
        for method, values in mean_maps.items():
            label = {"kbsm": "kbsm", "pitome": "pitome", "wam_lam100": "wam"}[method]
            save_overlay(
                gray, values, motion_mean, out / f"clip_{cid}_{label}.png", detail_limits
            )
        np.savez_compressed(
            out / f"clip_{cid}_maps.npz",
            motion_mean=motion_mean,
            motion_slice=motion_slice,
            kbsm=mean_maps["kbsm"],
            pitome=mean_maps["pitome"],
            wam=mean_maps["wam_lam100"],
            detail_limits=np.asarray(detail_limits),
            relevance_layer=relevance_layer,
            grid=np.asarray([t, h, w]),
        )

    metrics = json.loads(Path(args.metrics).read_text())
    with (out / "mechanism_clip_values.csv").open("w", newline="") as stream:
        fieldnames = ["cid", "task", "kbsm", "pitome", "lambda_0",
                      "lambda_025", "lambda_05", "lambda_075", "lambda_1"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["mechanism_rows"]:
            corr = row["correlation"]
            writer.writerow({
                "cid": row["cid"], "task": row["task"],
                "kbsm": corr["kbsm"], "pitome": corr["pitome"],
                "lambda_0": corr["wam_lam000"],
                "lambda_025": corr["wam_lam025"],
                "lambda_05": corr["wam_lam050"],
                "lambda_075": corr["wam_lam075"],
                "lambda_1": corr["wam_lam100"],
            })
    with (out / "layer_summary.csv").open("w", newline="") as stream:
        fieldnames = ["layer", "twonn", "twonn_low", "twonn_high",
                      "effective_rank", "effective_rank_low", "effective_rank_high",
                      "adjacent_cosine", "distant_cosine", "cosine_gap"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for layer in metrics["protocol"]["layers"]:
            summary = metrics["layer_summary"][str(layer)]
            writer.writerow({
                "layer": layer,
                "twonn": summary["twonn"]["mean"],
                "twonn_low": summary["twonn"]["ci95"][0],
                "twonn_high": summary["twonn"]["ci95"][1],
                "effective_rank": summary["effective_rank"]["mean"],
                "effective_rank_low": summary["effective_rank"]["ci95"][0],
                "effective_rank_high": summary["effective_rank"]["ci95"][1],
                "adjacent_cosine": summary["adjacent_cosine"]["mean"],
                "distant_cosine": summary["distant_cosine"]["mean"],
                "cosine_gap": summary["cosine_gap"]["mean"],
            })
    print(f"rendered tiles and CSVs to {out}")


if __name__ == "__main__":
    main()
