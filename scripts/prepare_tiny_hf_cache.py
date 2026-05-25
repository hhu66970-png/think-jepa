#!/usr/bin/env python
"""Prepare a tiny real ThinkJEPA cache subset from the gated HF release."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Set an authorized token after accepting "
            "https://huggingface.co/datasets/haichaozhang/cache access conditions."
        )
    return token


def _read_manifest(path: Path, limit: int) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.lstrip().startswith("#"):
                rows.append(text)
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise RuntimeError(f"{path} has only {len(rows)} usable rows, need {limit}")
    return rows


def _suffix_from_part2(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    marker = "/part2/"
    if marker in normalized:
        return "part2/" + normalized.split(marker, 1)[1]
    if normalized.startswith("part2/"):
        return normalized
    parts = [p for p in normalized.split("/") if p]
    if "part2" in parts:
        idx = parts.index("part2")
        return "/".join(parts[idx:])
    return None


def _cache_stem(path: str) -> str:
    stem = Path(path).stem
    if "_L" in stem and "_nf" in stem:
        return stem.split("_L", 1)[0]
    return stem


def _candidate_exact_repo_paths(sample_path: str) -> list[str]:
    suffix = _suffix_from_part2(sample_path)
    if suffix is None:
        return []
    suffix_no_ext = str(Path(suffix).with_suffix("")).replace("\\", "/")
    candidates = []
    if suffix.endswith(".npz"):
        candidates.append(suffix)
    candidates.append(suffix_no_ext + ".npz")
    return list(dict.fromkeys(candidates))


def _find_npz_by_directory(api: HfApi, repo: str, token: str, sample_path: str) -> str:
    suffix = _suffix_from_part2(sample_path)
    if suffix is None:
        raise FileNotFoundError(f"Could not find part2/ suffix in manifest row: {sample_path}")
    sample_stem = _cache_stem(suffix)
    rel_dir = str(Path(suffix).parent).replace("\\", "/")
    for item in api.list_repo_tree(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=rel_dir,
        recursive=False,
        token=token,
    ):
        item_path = getattr(item, "path", "")
        if not item_path.endswith(".npz"):
            continue
        if _cache_stem(item_path) == sample_stem:
            return item_path
    raise FileNotFoundError(f"No matching .npz for {sample_path} under {rel_dir}")


def _resolve_repo_npz(api: HfApi, repo: str, token: str, sample_path: str) -> str:
    for candidate in _candidate_exact_repo_paths(sample_path):
        try:
            if api.file_exists(
                repo_id=repo,
                repo_type="dataset",
                filename=candidate,
                token=token,
            ):
                return candidate
        except TypeError:
            if api.file_exists(repo_id=repo, repo_type="dataset", path_in_repo=candidate, token=token):
                return candidate
        except Exception:
            continue
        try:
            if api.file_exists(repo_id=repo, repo_type="dataset", path_in_repo=candidate, token=token):
                return candidate
        except Exception:
            continue
    return _find_npz_by_directory(api, repo, token, sample_path)


def _download_repo_file(repo: str, token: str, filename: str, out_root: Path) -> Path:
    cached = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        token=token,
    )
    cached_path = Path(cached).resolve(strict=True)
    dst = out_root / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst) and not dst.exists():
        dst.unlink()
    if not dst.exists():
        try:
            os.link(cached_path, dst)
        except OSError:
            shutil.copy2(cached_path, dst)
    return dst


def _write_local_manifest(rows: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row.resolve()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Prepare tiny ThinkJEPA HF cache subset")
    parser.add_argument("--repo", default="haichaozhang/cache")
    parser.add_argument(
        "--split-dir",
        default="egodex_part2_video_cache_subset2000_ratio0.9_seed42/splits",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=1)
    args = parser.parse_args()

    token = _require_token()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    train_manifest = _download_repo_file(
        args.repo, token, f"{args.split_dir.strip('/')}/train_cache.txt", out_root
    )
    test_manifest = _download_repo_file(
        args.repo, token, f"{args.split_dir.strip('/')}/test_cache.txt", out_root
    )

    train_rows = _read_manifest(train_manifest, int(args.train_count))
    test_rows = _read_manifest(test_manifest, int(args.test_count))

    train_local = []
    test_local = []
    for sample in train_rows:
        repo_npz = _resolve_repo_npz(api, args.repo, token, sample)
        train_local.append(_download_repo_file(args.repo, token, repo_npz, out_root))
        print(f"[train] {repo_npz}")
    for sample in test_rows:
        repo_npz = _resolve_repo_npz(api, args.repo, token, sample)
        test_local.append(_download_repo_file(args.repo, token, repo_npz, out_root))
        print(f"[test] {repo_npz}")

    _write_local_manifest(train_local, out_root / "train_tiny.txt")
    _write_local_manifest(test_local, out_root / "test_tiny.txt")
    print(f"TINY_CACHE_READY out_dir={out_root} train={len(train_local)} test={len(test_local)}")


if __name__ == "__main__":
    main()
