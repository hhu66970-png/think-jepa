"""Per-clip relevance-prior store for WAM score-family experiments.

Why this exists
---------------
``DiagnosticTokenMerger._load_prior_relevance`` (V2) can only load a SINGLE
``[t*h*w]`` vector and broadcasts it to every sample in the batch, i.e. a
clip-agnostic prior. Comparing relevance *score families* (motion vs. optical
flow vs. ego-motion-compensated flow vs. hand-joint proximity vs. random)
requires a DIFFERENT map per clip, so this module adds a per-clip store keyed by
the clip's cache path.

Contract
--------
* ``set_current_clip_keys(paths)`` is called by the training/eval loop
  immediately before the encoder forward, with the batch's per-sample cache
  paths (same order as the batch dimension).
* ``PriorStore.lookup(keys, n_tokens)`` returns ``[B, n_tokens]`` float32, or
  ``None`` if the store cannot serve the request. ``None`` ALWAYS means "fall
  back to the in-place motion signal" -- this module must never raise into the
  encoder.

Store layout on disk::

    <root>/index.json            {"family": str, "t":32, "h":16, "w":16,
                                  "entries": {clip_key: "rel/<sha1>.npy"}}
    <root>/rel/<sha1>.npy        float32 [t*h*w] in [0,1]

``clip_key`` is the basename-with-parent of the cache file, e.g.
``assemble_disassemble_furniture_bench_chair/3503_L8_nf32_res256_new16_s01of08.npz``
so the store survives the tree being moved.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np

__all__ = [
    "clip_key_for_path",
    "set_current_clip_keys",
    "get_current_clip_keys",
    "clear_current_clip_keys",
    "PriorStore",
    "get_store",
]

_TLS = threading.local()


def clip_key_for_path(path) -> str:
    """Stable, relocation-proof key: '<parent_dir>/<filename>'."""
    p = str(path)
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{os.path.basename(p)}" if parent else os.path.basename(p)


def set_current_clip_keys(paths) -> None:
    """Record the batch's clip paths. Safe to call with None / empty."""
    if paths is None:
        _TLS.keys = None
        return
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    try:
        _TLS.keys = [clip_key_for_path(p) for p in paths]
    except Exception:
        _TLS.keys = None


def get_current_clip_keys():
    return getattr(_TLS, "keys", None)


def clear_current_clip_keys() -> None:
    _TLS.keys = None


class PriorStore:
    """Read-only per-clip prior store. Never raises on lookup."""

    def __init__(self, root: str):
        self.root = str(root)
        self.ok = False
        self.family = None
        self.shape = None            # (t, h, w)
        self._entries = {}
        self._cache = {}
        self._miss_warned = set()
        try:
            with open(os.path.join(self.root, "index.json"), "r") as fh:
                meta = json.load(fh)
            self.family = meta.get("family")
            self.shape = (int(meta["t"]), int(meta["h"]), int(meta["w"]))
            self._entries = dict(meta.get("entries", {}))
            self.ok = len(self._entries) > 0
        except Exception:
            self.ok = False

    # ------------------------------------------------------------------
    def _vec(self, key: str):
        if key in self._cache:
            return self._cache[key]
        rel = self._entries.get(key)
        if rel is None:
            return None
        try:
            arr = np.load(os.path.join(self.root, rel))
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        self._cache[key] = arr
        return arr

    def lookup(self, keys, n_tokens: int):
        """Return [B, n_tokens] float32, or None to trigger the motion fallback.

        Partial coverage is treated as failure on purpose: silently mixing a
        prior for some clips with the motion signal for others would corrupt the
        score-family comparison this store exists to enable.
        """
        if not self.ok or not keys:
            return None
        t, h, w = self.shape
        n_thw, n_hw = t * h * w, h * w
        out = np.empty((len(keys), int(n_tokens)), dtype=np.float32)
        for i, key in enumerate(keys):
            vec = self._vec(key)
            if vec is None:
                if key not in self._miss_warned and len(self._miss_warned) < 5:
                    self._miss_warned.add(key)
                    print(f"[WAM][prior] MISS key={key!r} in {self.root} "
                          f"-> whole batch falls back to motion", flush=True)
                return None
            if vec.shape[0] == n_hw and n_thw == int(n_tokens):
                vec = np.tile(vec, t)
            if vec.shape[0] != int(n_tokens):
                return None
            out[i] = vec
        return out


_STORES = {}
_STORES_LOCK = threading.Lock()


def get_store(root: str):
    """Process-wide cached store. Returns None when `root` is not a store."""
    if not root:
        return None
    root = str(root)
    with _STORES_LOCK:
        st = _STORES.get(root)
        if st is None:
            st = PriorStore(root)
            _STORES[root] = st
    return st if st.ok else None
