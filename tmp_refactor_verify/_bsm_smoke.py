"""HARD GATE B smoke test for bsm_ksim_gradual_vec (Gradual K-BSM).

Builds the RoPE encoder, loads vitl.pt, runs one compressed forward and one
restore_dense forward on a random video, and asserts:
  (a) it runs,
  (b) final tokens < initial tokens AND tokens were dropped at MULTIPLE layers,
  (c) fidelity (cosine + relL2) vs the restore_dense=True pass is finite.
Reports the per-layer trajectory and the matching metric actually used (key vs
feature fallback) to confirm SDPA-safe K exposure.
"""
import json

import torch

from src.models import vision_transformer

CKPT = "vjepa2/vitl.pt"
MERGE_LAYERS = (4, 6, 8)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    merge_config = {
        "enabled": True,
        "strategy": "bsm_ksim_gradual_vec",
        "merge_layers": MERGE_LAYERS,
        "merge_ratio": 0.125,
        "restore_dense": False,
        "profile": True,
        "bsm_match_metric": "key",
    }
    model = vision_transformer.vit_large_rope(
        img_size=(256, 256),
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        out_layers=None,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        merge_config=merge_config,
    )
    print(f"[INFO] token_merger class = {type(model.token_merger).__name__}")
    try:
        blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(CKPT, map_location="cpu")
    if isinstance(blob, dict) and "encoder" in blob:
        state = blob["encoder"]
    elif isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    else:
        state = blob
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model = model.to(dev).eval()

    video = torch.randn(1, 3, 16, 256, 256, device=dev)

    # (a) compressed forward. With out_layers=None the encoder returns the tensor
    # x of shape [B, N, D]; index batch 0 -> [N, D] (matches the runner).
    with torch.no_grad():
        out_c, infos = model(video, return_merge_info=True, restore_dense=False)
    feat_c = out_c[0]
    tokens_final = int(feat_c.shape[0])
    original = int(model.last_forward_profile["tokens_original"])

    print(f"[RESULT] original_tokens={original} final_tokens={tokens_final}")
    traj = []
    layers_with_drops = 0
    for info in infos:
        nb = int(info.get("num_tokens_before"))
        na = int(info.get("num_tokens_after"))
        acc = int(info.get("num_accepted") or 0)
        metric = info.get("matching_metric", info.get("bsm_match_metric"))
        impl = info.get("implementation")
        fb = info.get("fallback_reason")
        dropped = nb - na
        if dropped > 0:
            layers_with_drops += 1
        traj.append({
            "layer": int(info.get("layer")), "before": nb, "after": na,
            "accepted": acc, "dropped": dropped, "metric": metric,
            "impl": impl, "fallback_reason": fb,
        })
        print(f"  layer {info.get('layer')}: {nb} -> {na} (dropped {dropped}, "
              f"accepted {acc}) metric={metric} impl={impl} fb={fb}")

    # (c) fidelity vs restore_dense pass (aligned to ORIGINAL grid -> [N, D]).
    with torch.no_grad():
        out_d, _ = model(video, return_merge_info=True, restore_dense=True)
    feat_d = out_d[0].float()
    feat_c_dense_tokens = int(feat_d.shape[0])

    # Build a dense reference by running with merge disabled to compare fidelity
    # of the restored features.
    model.merge_config.enabled = False
    with torch.no_grad():
        out_ref, _ = model(video, return_merge_info=True, restore_dense=True)
    model.merge_config.enabled = True
    feat_ref = out_ref[0].float()

    cos = torch.nn.functional.cosine_similarity(feat_ref, feat_d, dim=-1).mean().item()
    rel_l2 = ((feat_d - feat_ref).norm() / feat_ref.norm().clamp_min(1e-12)).item()
    cos_finite = bool(torch.isfinite(torch.tensor(cos)).item())
    l2_finite = bool(torch.isfinite(torch.tensor(rel_l2)).item())
    feat_c_finite = bool(torch.isfinite(feat_c).all().item())

    # metric provenance
    metrics_used = sorted({t["metric"] for t in traj})
    used_key = all(t["metric"] == "key" for t in traj) and len(traj) > 0

    summary = {
        "ran": True,
        "original_tokens": original,
        "final_tokens": tokens_final,
        "tokens_reduced": tokens_final < original,
        "num_merge_layers_with_drops": layers_with_drops,
        "multi_layer_drop": layers_with_drops >= 2,
        "restore_dense_tokens": feat_c_dense_tokens,
        "restore_dense_eq_original": feat_c_dense_tokens == original,
        "fidelity_cosine_vs_dense": cos,
        "fidelity_rel_l2_vs_dense": rel_l2,
        "cos_finite": cos_finite,
        "l2_finite": l2_finite,
        "compressed_feat_finite": feat_c_finite,
        "metrics_used": metrics_used,
        "used_key_metric_all_layers": used_key,
        "trajectory": traj,
    }
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False))

    # ---- assertions ----
    assert tokens_final < original, "FAIL: final tokens not < original"
    assert layers_with_drops >= 2, f"FAIL: tokens dropped at only {layers_with_drops} layer(s); need >=2"
    assert cos_finite and l2_finite, "FAIL: fidelity not finite"
    assert feat_c_finite, "FAIL: compressed features contain non-finite values"
    assert feat_c_dense_tokens == original, "FAIL: restore_dense did not return original token count"
    print("[GATE_B_PASS] bsm_ksim_gradual_vec runs, reduces tokens across multiple layers, fidelity finite")


if __name__ == "__main__":
    main()
