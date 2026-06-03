#!/usr/bin/env python
"""Thin wrapper around tools/dense_jepa_pca_vis.py that builds a V-JEPA2 ViT-g
encoder with the *correct* giant config (use_rope=True, uniform_power=True) and
runs the existing single-method dense-PCA pipeline.

We do NOT touch any shared / token-merge code.  We only override
`load_dense_jepa_model` so the giant checkpoint loads cleanly, then call the
original `main()`.  All other behaviour (foreground two-stage PCA, rendering,
layer auto-selection, manifests) is reused verbatim.

Run exactly like dense_jepa_pca_vis.py, e.g.:
  python tools/pca_vitg_driver.py --npz <clip>.npz --checkpoint vjepa2/vitg.pt \
      --model_arch vit_giant_xformers --img_size 384 --patch_size 16 \
      --num_frames 32 --out_layers 24,29,34,39 ... --out_dir <dir>
"""
import sys
import torch

import tools.dense_jepa_pca_vis as D


GIANT_ARCHES = {
    "vit_giant", "vit_giant_rope", "vit_giant_xformers",
    "vit_giant_xformers_rope", "vit_gigantic", "vit_gigantic_xformers",
}


def load_giant_model(checkpoint_path, img_size, num_frames, patch_size,
                     out_layers, device, model_arch):
    """Build giant encoder with rope+uniform_power and load weights leniently."""
    from vjepa2.src.models import vision_transformer

    if not hasattr(vision_transformer, model_arch):
        available = [n for n in dir(vision_transformer) if n.startswith("vit_")]
        raise ValueError(f"Unknown --model_arch {model_arch!r}; available={available}")

    is_giant = model_arch in GIANT_ARCHES
    factory = getattr(vision_transformer, model_arch)
    model = factory(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=2,
        out_layers=out_layers,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        # ViT-g 384 official pretrain: uniform_power=True, use_rope=True
        uniform_power=True if is_giant else False,
        use_rope=True if is_giant else False,
    )

    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (TypeError, Exception):  # noqa: BLE001 - some ckpts need full unpickle
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # locate weights: V-JEPA2 release uses 'target_encoder'/'encoder'
    if isinstance(blob, dict):
        for key in ("target_encoder", "encoder", "model"):
            if key in blob and isinstance(blob[key], dict):
                state = blob[key]
                print(f"[INFO] using checkpoint key: {key}", flush=True)
                break
        else:
            state = blob
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(blob)!r}")

    clean = {}
    for k, v in state.items():
        clean[k.replace("module.", "").replace("backbone.", "")] = v

    msg = model.load_state_dict(clean, strict=False)
    n_missing = len(msg.missing_keys)
    n_unexp = len(msg.unexpected_keys)
    print(f"[INFO] loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"[INFO] load_state_dict: missing={n_missing} unexpected={n_unexp}", flush=True)
    if n_missing:
        print(f"[INFO] first missing keys: {msg.missing_keys[:8]}", flush=True)
    if n_unexp:
        print(f"[INFO] first unexpected keys: {msg.unexpected_keys[:8]}", flush=True)
    # sanity: a giant load should match the vast majority of params
    total = sum(1 for _ in model.state_dict())
    if n_missing > 0.15 * total:
        print(f"[WARN] many missing keys ({n_missing}/{total}); arch/ckpt mismatch?", flush=True)

    model.to(device)
    model.eval()
    return model


def _patch_param_count():
    """Optional: print param count once at build (informational)."""
    orig = load_giant_model

    def wrapped(*a, **k):
        m = orig(*a, **k)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"[INFO] model params: {n:.1f}M", flush=True)
        return m

    return wrapped


if __name__ == "__main__":
    D.load_dense_jepa_model = _patch_param_count()
    D.main()
