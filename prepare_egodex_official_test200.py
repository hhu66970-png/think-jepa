#!/usr/bin/env python3
"""Download and validate the complete official 200-clip EgoDex test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


SOURCE_PREFIX = "egodex_part2_video_cache_subset2000_ratio0.9_seed42/splits"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def category(path: str) -> str:
    return Path(path).parent.name


def video_group(path: str) -> tuple[str, str]:
    p = Path(path)
    return category(path), p.name.split("_L8_", 1)[0]


def repo_npz_path(source_path: str) -> str:
    p = Path(source_path)
    return f"part2/{p.parent.name}/{p.name}"


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="haichaozhang/cache")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reference-subset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--required-guidance",
        choices=("both", "old", "new", "none"),
        default="both",
        help="ThinkJEPA guidance streams that every test archive must contain",
    )
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    )
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    dataset_info = api.dataset_info(args.repo, revision=args.revision)
    resolved_revision = str(dataset_info.sha)
    source_dir = args.destination / "source_splits"
    source_dir.mkdir(exist_ok=True)
    train_source = Path(
        hf_hub_download(
            args.repo,
            f"{SOURCE_PREFIX}/train_cache.txt",
            repo_type="dataset",
            revision=resolved_revision,
            endpoint=args.endpoint,
            local_dir=source_dir,
        )
    )
    test_source = Path(
        hf_hub_download(
            args.repo,
            f"{SOURCE_PREFIX}/test_cache.txt",
            repo_type="dataset",
            revision=resolved_revision,
            endpoint=args.endpoint,
            local_dir=source_dir,
        )
    )
    official_train = read_lines(train_source)
    official_test = read_lines(test_source)
    if len(official_test) != 200:
        raise RuntimeError(f"expected 200 official test clips, found {len(official_test)}")
    official_overlap = set(map(video_group, official_train)) & set(map(video_group, official_test))
    if official_overlap:
        raise RuntimeError(f"official train/test video-group overlap: {sorted(official_overlap)[:5]}")

    train_manifest = read_lines(args.reference_subset / "manifests/train.txt")
    validation_manifest = read_lines(args.reference_subset / "manifests/validation.txt")
    selected_groups = set(map(video_group, train_manifest + validation_manifest))
    test_groups = set(map(video_group, official_test))
    overlap = selected_groups & test_groups
    if overlap:
        raise RuntimeError(f"selected train/validation overlaps official test: {sorted(overlap)[:5]}")

    requested = [repo_npz_path(path) for path in official_test]
    if len(set(requested)) != 200:
        raise RuntimeError("official test paths are not unique")
    data_root = args.destination / "cache"
    data_root.mkdir(exist_ok=True)

    def download(repo_path: str) -> tuple[str, Path]:
        local = Path(
            hf_hub_download(
                args.repo,
                repo_path,
                repo_type="dataset",
                revision=resolved_revision,
                endpoint=args.endpoint,
                local_dir=data_root,
            )
        )
        if not local.is_file() or local.stat().st_size == 0:
            raise RuntimeError(f"empty download: {repo_path}")
        return repo_path, local

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, path): path for path in requested}
        for index, future in enumerate(as_completed(futures), start=1):
            repo_path, local_path = future.result()
            downloaded[repo_path] = local_path
            print(f"[{index:03d}/200] {repo_path}", flush=True)

    import numpy as np

    required_keys = {"imgs", "vjepa_feats"}
    if args.required_guidance in ("both", "old"):
        required_keys.add("vlm_old")
    if args.required_guidance in ("both", "new"):
        required_keys.add("vlm_new")

    reference_dims: dict[str, int] = {}
    reference_archive = Path(train_manifest[0])
    with np.load(reference_archive, allow_pickle=False) as reference_payload:
        reference_missing = required_keys - set(reference_payload.files)
        if reference_missing:
            raise RuntimeError(
                f"reference training archive missing keys: {sorted(reference_missing)}"
            )
        for key in ("vlm_old", "vlm_new"):
            if key in required_keys:
                if reference_payload[key].ndim < 2:
                    raise RuntimeError(f"reference {key} has invalid shape")
                reference_dims[key] = int(reference_payload[key].shape[-1])

    local_paths: list[str] = []
    for source_path, repo_path in zip(official_test, requested):
        local_path = downloaded[repo_path]
        with np.load(local_path, allow_pickle=False) as payload:
            missing = required_keys - set(payload.files)
            if missing:
                raise RuntimeError(f"{repo_path} missing keys: {sorted(missing)}")
            for key in required_keys:
                if payload[key].size == 0:
                    raise RuntimeError(f"{repo_path} has empty required key: {key}")
            for key, expected_dim in reference_dims.items():
                if payload[key].ndim < 2 or int(payload[key].shape[-1]) != expected_dim:
                    raise RuntimeError(
                        f"{repo_path} {key} dim mismatch: "
                        f"shape={payload[key].shape}, expected last dim={expected_dim}"
                    )
        local_paths.append(str(local_path.resolve()))

    manifest_dir = args.destination / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "test200.txt").write_text("\n".join(local_paths) + "\n")
    file_records = [
        {
            "source_path": source_path,
            "repo_path": repo_path,
            "local_path": str(downloaded[repo_path].resolve()),
            "size_bytes": downloaded[repo_path].stat().st_size,
            "sha256": sha256_file(downloaded[repo_path]),
        }
        for source_path, repo_path in zip(official_test, requested)
    ]
    metadata = {
        "repo": args.repo,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "endpoint": args.endpoint,
        "source_prefix": SOURCE_PREFIX,
        "required_guidance": args.required_guidance,
        "required_npz_keys": sorted(required_keys),
        "reference_training_archive": str(reference_archive),
        "guidance_feature_dims": reference_dims,
        "official_train_count": len(official_train),
        "official_test_count": len(official_test),
        "official_test_source_sha256": sha256_lines(official_test),
        "local_test_manifest_sha256": sha256_lines(local_paths),
        "categories": dict(sorted(Counter(map(category, official_test)).items())),
        "source_video_groups": len(test_groups),
        "video_group_overlap": {
            "official_train_test": 0,
            "selected_train_validation_vs_test": 0,
        },
        "downloaded_files": len(downloaded),
        "downloaded_bytes": sum(path.stat().st_size for path in downloaded.values()),
        "files": file_records,
    }
    (args.destination / "test200_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
