#!/usr/bin/env python
"""Analyze the protocol-locked 64-frame/256-px WAM figure dump.

Outputs machine-readable per-clip values and compact summaries for:
  * layer-wise effective rank, participation ratio, and TwoNN dimension;
  * layer-wise adjacent-vs-distant same-frame cosine;
  * preservation/change correlation for K-BSM, PiToMe, and WAM lambda sweep.

The motion/change signal is computed from dense features at the first merge
layer, matching the implementation used by WAM.
"""

import argparse
import hashlib
import json
import math
import os
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


T_CRIT_95 = {24: 2.0639}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_ci95(values):
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    tcrit = T_CRIT_95.get(len(a) - 1, 1.96)
    half = float(tcrit * sd / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return {
        "n": int(len(a)),
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
    }


def linear_dimensions(array, device):
    x = torch.from_numpy(array.astype(np.float32, copy=False)).to(device)
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    singular = singular[singular > 1e-10]
    prob = singular / singular.sum()
    effective_rank = torch.exp(-(prob * prob.log()).sum())
    eig = singular.square()
    participation_ratio = eig.sum().square() / eig.square().sum().clamp_min(1e-20)
    return float(effective_rank.cpu()), float(participation_ratio.cpu())


def twonn_dimension(array, device, nsub, seed):
    rng = np.random.RandomState(seed)
    if len(array) > nsub:
        indices = rng.choice(len(array), nsub, replace=False)
        array = array[indices]
    x = torch.from_numpy(array.astype(np.float32, copy=False)).to(device)
    distances = torch.cdist(x, x)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.topk(distances, k=2, dim=1, largest=False).values
    r1, r2 = nearest[:, 0], nearest[:, 1]
    valid = (r1 > 1e-9) & (r2 > r1)
    ratios = (r2[valid] / r1[valid]).sort().values
    ratios = ratios[: max(1, int(len(ratios) * 0.9))]
    value = float(len(ratios) / ratios.log().sum().cpu())
    return value


def spatial_cosines(array, t, h, w, seed, n_pairs=6000):
    x = torch.from_numpy(array.astype(np.float32, copy=False))
    grid = F.normalize(x, dim=-1).reshape(t, h, w, -1)
    horizontal = (grid[:, :, :-1] * grid[:, :, 1:]).sum(dim=-1)
    vertical = (grid[:, :-1, :] * grid[:, 1:, :]).sum(dim=-1)
    adjacent = float(torch.cat([horizontal.reshape(-1), vertical.reshape(-1)]).mean())

    rng = np.random.RandomState(seed)
    threshold = max(h, w) // 2
    accepted = []
    while sum(len(item) for item in accepted) < n_pairs:
        pairs = rng.randint(0, h * w, size=(n_pairs, 2))
        row_gap = np.abs(pairs[:, 0] // w - pairs[:, 1] // w)
        col_gap = np.abs(pairs[:, 0] % w - pairs[:, 1] % w)
        keep = pairs[(row_gap + col_gap) >= threshold]
        if len(keep):
            accepted.append(keep)
    pairs = np.concatenate(accepted, axis=0)[:n_pairs]
    flat = grid.reshape(t, h * w, -1)
    distant_values = []
    for time_index in range(t):
        distant_values.append(
            (flat[time_index, pairs[:, 0]] * flat[time_index, pairs[:, 1]])
            .sum(dim=-1)
        )
    distant = float(torch.cat(distant_values).mean())
    return adjacent, distant


def motion_change(array, t, h, w):
    grid = array.astype(np.float32, copy=False).reshape(t, h, w, -1)
    motion = np.zeros((t, h, w), dtype=np.float32)
    motion[1:] = np.linalg.norm(grid[1:] - grid[:-1], axis=-1)
    flat = motion.reshape(-1).astype(np.float64)
    flat -= flat.min()
    flat /= max(flat.max(), 1e-12)
    return flat


def preservation(group_ids):
    _, inverse, counts = np.unique(
        group_ids.astype(np.int64, copy=False), return_inverse=True, return_counts=True
    )
    return (1.0 / counts[inverse]).astype(np.float64)


def pearson(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--twonn_subsample", type=int, default=2500)
    args = parser.parse_args()

    dump = Path(args.dump)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(dump.glob("feats_*.npz"))
    if len(files) != 25:
        raise SystemExit(f"expected 25 dumps, got {len(files)}")

    first = np.load(files[0], allow_pickle=True)
    layers = [int(v) for v in first["cap_layers"]]
    relevance_layer = int(first["relevance_layer"])
    if relevance_layer not in layers:
        raise SystemExit(f"relevance layer L{relevance_layer} was not captured")
    expected_geometry = (32, 16, 16)
    methods = ["kbsm", "pitome", "wam_lam000", "wam_lam025",
               "wam_lam050", "wam_lam075", "wam_lam100"]

    layer_rows = {str(layer): [] for layer in layers}
    mechanism_rows = []
    for file_index, path in enumerate(files):
        data = np.load(path, allow_pickle=True)
        geometry = (int(data["t"]), int(data["h"]), int(data["w"]))
        if geometry != expected_geometry or int(data["token_count"]) != 8192:
            raise SystemExit(f"geometry mismatch in {path}: {geometry}")
        if not bool(data["lambda0_same_partition_kbsm"]):
            raise SystemExit(f"lambda=0 mismatch in {path}")
        cid = str(data["cid"].item())
        task = str(data["task"].item())

        for layer in layers:
            dense = data[f"dense_L{layer}"].astype(np.float32)
            seed = zlib.crc32(f"{cid}:L{layer}".encode()) & 0xFFFFFFFF
            effective_rank, ratio = linear_dimensions(dense, args.device)
            twonn = twonn_dimension(dense, args.device, args.twonn_subsample, seed)
            adjacent, distant = spatial_cosines(
                dense, *geometry, seed=(seed ^ 0xA5A5A5A5)
            )
            layer_rows[str(layer)].append({
                "cid": cid,
                "task": task,
                "effective_rank": effective_rank,
                "participation_ratio": ratio,
                "twonn": twonn,
                "adjacent_cosine": adjacent,
                "distant_cosine": distant,
                "cosine_gap": adjacent - distant,
            })
            print(
                f"[{file_index + 1:02d}/25 {cid}] L{layer}: "
                f"er={effective_rank:.2f} TwoNN={twonn:.2f} "
                f"cos={adjacent:.3f}/{distant:.3f}",
                flush=True,
            )

        motion = motion_change(
            data[f"dense_L{relevance_layer}"], *geometry
        )
        correlations = {}
        for method in methods:
            correlations[method] = pearson(
                preservation(data[f"{method}_gids"]), motion
            )
        mechanism_rows.append({
            "cid": cid,
            "task": task,
            "relevance_layer": relevance_layer,
            "correlation": correlations,
        })
        print(f"[{cid}] mechanism {correlations}", flush=True)

    layer_summary = {}
    for layer in layers:
        rows = layer_rows[str(layer)]
        layer_summary[str(layer)] = {
            metric: mean_ci95([row[metric] for row in rows])
            for metric in ["effective_rank", "participation_ratio", "twonn",
                           "adjacent_cosine", "distant_cosine", "cosine_gap"]
        }

    mechanism_summary = {
        method: mean_ci95([
            row["correlation"][method] for row in mechanism_rows
        ])
        for method in methods
    }
    ordering = sum(
        row["correlation"]["wam_lam100"]
        > row["correlation"]["pitome"]
        > row["correlation"]["kbsm"]
        for row in mechanism_rows
    )

    result = {
        "protocol": {
            "num_frames": 64,
            "img_size": 256,
            "patch_size": 16,
            "grid": list(expected_geometry),
            "token_count": 8192,
            "n_clips": len(files),
            "layers": layers,
            "relevance_layer": relevance_layer,
            "motion_definition": "per-token L2 change between adjacent temporal tokens",
            "distant_definition": "same-frame pairs with Manhattan distance >= half the spatial grid width",
            "twonn_subsample": args.twonn_subsample,
        },
        "source": {
            "dump": str(dump.resolve()),
            "files": [{"file": p.name, "sha256": sha256(p)} for p in files],
        },
        "layer_rows": layer_rows,
        "layer_summary": layer_summary,
        "mechanism_rows": mechanism_rows,
        "mechanism_summary": mechanism_summary,
        "ordering_wam_gt_pitome_gt_kbsm": int(ordering),
    }
    json_path = out / "protocol_metrics.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Protocol-Locked Figure Metrics",
        "",
        "Geometry: 64 frames, 256 px, 32x16x16 = 8192 tokens.",
        f"Mechanism relevance layer: L{relevance_layer}.",
        "",
        "## Layer summary",
        "",
        "| Layer | TwoNN | Effective rank | Adjacent cos | Distant cos | Gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        summary = layer_summary[str(layer)]
        lines.append(
            f"| L{layer} | {summary['twonn']['mean']:.2f} | "
            f"{summary['effective_rank']['mean']:.1f} | "
            f"{summary['adjacent_cosine']['mean']:.3f} | "
            f"{summary['distant_cosine']['mean']:.3f} | "
            f"{summary['cosine_gap']['mean']:+.3f} |"
        )
    lines.extend([
        "",
        "## Preservation-change correlation",
        "",
        "| Method | Mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for method in methods:
        summary = mechanism_summary[method]
        lines.append(
            f"| {method} | {summary['mean']:+.4f} | "
            f"[{summary['ci95'][0]:+.4f}, {summary['ci95'][1]:+.4f}] |"
        )
    lines.extend([
        "",
        f"WAM(lambda=1) > PiToMe > K-BSM: {ordering}/25 clips.",
        "",
    ])
    (out / "protocol_summary.md").write_text("\n".join(lines))
    print(json.dumps({
        "json": str(json_path),
        "ordering": ordering,
        "mechanism_summary": mechanism_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
