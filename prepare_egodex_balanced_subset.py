#!/usr/bin/env python3
"""Build a deterministic, category-balanced local subset from the public EgoDex cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


SOURCE_PREFIX = "egodex_part2_video_cache_subset2000_ratio0.9_seed42/splits"


def category(path: str) -> str:
    return Path(path).parent.name


def video_group(path: str) -> tuple[str, str]:
    p = Path(path)
    return category(path), p.name.split("_L8_", 1)[0]


def stable_shuffle(items: list[str], seed: int, label: str) -> list[str]:
    digest = hashlib.sha256(label.encode()).digest()
    category_seed = seed ^ int.from_bytes(digest[:8], "big")
    result = list(items)
    random.Random(category_seed).shuffle(result)
    return result


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def repo_npz_path(source_path: str) -> str:
    p = Path(source_path)
    return f"part2/{p.parent.name}/{p.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="haichaozhang/cache")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202707)
    parser.add_argument("--train-per-category", type=int, default=4)
    parser.add_argument("--val-per-category", type=int, default=1)
    parser.add_argument("--test-per-category", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    source_dir = args.destination / "source_splits"
    source_dir.mkdir(exist_ok=True)
    source_files: dict[str, Path] = {}
    for name in ("train_cache.txt", "test_cache.txt", "meta.json", "size_summary.json"):
        downloaded = Path(
            hf_hub_download(
                args.repo,
                f"{SOURCE_PREFIX}/{name}",
                repo_type="dataset",
                endpoint=args.endpoint,
                local_dir=source_dir,
            )
        )
        source_files[name] = downloaded

    official_train = read_lines(source_files["train_cache.txt"])
    official_test = read_lines(source_files["test_cache.txt"])
    overlap = set(map(video_group, official_train)) & set(map(video_group, official_test))
    if overlap:
        raise RuntimeError(f"official train/test video-group overlap: {sorted(overlap)[:5]}")

    train_by_category: dict[str, list[str]] = defaultdict(list)
    test_by_category: dict[str, list[str]] = defaultdict(list)
    for path in official_train:
        train_by_category[category(path)].append(path)
    for path in official_test:
        test_by_category[category(path)].append(path)
    categories = sorted(set(train_by_category) & set(test_by_category))

    selected: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for cat in categories:
        train_pool = stable_shuffle(train_by_category[cat], args.seed, f"train:{cat}")
        test_pool = stable_shuffle(test_by_category[cat], args.seed, f"test:{cat}")
        train_need = args.train_per_category + args.val_per_category
        if len(train_pool) < train_need or len(test_pool) < args.test_per_category:
            raise RuntimeError(
                f"insufficient files for {cat}: train={len(train_pool)}, test={len(test_pool)}"
            )
        selected["train"].extend(train_pool[: args.train_per_category])
        selected["validation"].extend(
            train_pool[args.train_per_category : train_need]
        )
        selected["test"].extend(test_pool[: args.test_per_category])

    split_groups = {name: set(map(video_group, paths)) for name, paths in selected.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        common = split_groups[left] & split_groups[right]
        if common:
            raise RuntimeError(f"{left}/{right} video-group overlap: {sorted(common)[:5]}")

    requested = sorted({repo_npz_path(path) for paths in selected.values() for path in paths})
    data_root = args.destination / "cache"
    data_root.mkdir(exist_ok=True)

    def download(repo_path: str) -> tuple[str, Path]:
        local_path = Path(
            hf_hub_download(
                args.repo,
                repo_path,
                repo_type="dataset",
                endpoint=args.endpoint,
                local_dir=data_root,
            )
        )
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise RuntimeError(f"empty download: {repo_path}")
        return repo_path, local_path

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, path): path for path in requested}
        for index, future in enumerate(as_completed(futures), start=1):
            repo_path, local_path = future.result()
            downloaded[repo_path] = local_path
            print(f"[{index:03d}/{len(requested):03d}] {repo_path}", flush=True)

    manifest_dir = args.destination / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    local_splits: dict[str, list[str]] = {}
    for name, paths in selected.items():
        local_paths = [str(downloaded[repo_npz_path(path)].resolve()) for path in paths]
        local_splits[name] = local_paths
        (manifest_dir / f"{name}.txt").write_text("\n".join(local_paths) + "\n")

    # Lightweight structural validation: opening an NPZ checks its archive table
    # without loading every array into memory.
    import numpy as np

    required_keys = {"imgs", "vjepa_feats"}
    for path in requested:
        with np.load(downloaded[path], allow_pickle=False) as payload:
            missing = required_keys - set(payload.files)
            if missing:
                raise RuntimeError(f"{path} missing keys: {sorted(missing)}")

    metadata = {
        "repo": args.repo,
        "endpoint": args.endpoint,
        "source_prefix": SOURCE_PREFIX,
        "selection_seed": args.seed,
        "categories": categories,
        "counts": {name: len(paths) for name, paths in local_splits.items()},
        "per_category": {
            "train": args.train_per_category,
            "validation": args.val_per_category,
            "test": args.test_per_category,
        },
        "official_counts": {"train": len(official_train), "test": len(official_test)},
        "video_group_overlap": {
            "official_train_test": 0,
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "downloaded_files": len(downloaded),
        "downloaded_bytes": sum(path.stat().st_size for path in downloaded.values()),
    }
    (args.destination / "subset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

