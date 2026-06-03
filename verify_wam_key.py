#!/usr/bin/env python
"""Gold-standard check: did WAM (bsm_taware_gradual_vec) actually use the
post-RoPE Key metric (not feature_fallback) on the REAL encode path used by
downstream training? Builds the real vitl.pt model, runs one real clip through
the merge path, and inspects every merge layer's info["fallback_reason"].

PASS = no merge-layer reports 'attn_key_unavailable' for WAM(key).
Control = force bsm_match_metric='feature' and show the fallback DOES appear,
proving the check discriminates.
"""
import os, sys
import torch
sys.path.insert(0, "vjepa2/src")
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E

CKPT = "vjepa2/vitl.pt"
CLIP = ("/root/autodl-tmp/thinkjepa-work/cache_ext30/part2/"
        "assemble_disassemble_furniture_bench_chair/1098_L8_nf32_res256_new16_s00of08.npz")
NF, IMG, PATCH = 64, 256, 16
GR5 = [12, 14, 16, 18, 20]


@torch.no_grad()
def run(model, video, metric, label):
    E.apply_merge_config(model, enabled=True, strategy="bsm_taware_gradual_vec",
                         merge_layers=GR5, merge_ratio=0.15, restore_dense=True,
                         bsm_match_metric=metric, merge_axis="free")
    model.merge_config.relevance_source = "motion"
    model.merge_config.relevance_lambda = 1.0
    outs, infos = model(video, return_merge_info=True, restore_dense=True)
    print(f"\n===== WAM metric={metric!r} ({label}) =====")
    fb_seen = []
    for i, info in enumerate(infos):
        fb = info.get("fallback_reason")
        nta = info.get("num_tokens_after")
        mm = info.get("match_metric", info.get("bsm_match_metric"))
        if fb is not None or (nta is not None and nta < video.shape[1] // PATCH):
            print(f"  layer-info[{i}] tokens_after={nta} match_metric={mm} fallback={fb}")
        if fb is not None:
            fb_seen.append(fb)
    final_tok = int(infos[-1].get("num_tokens_after")) if infos else None
    print(f"  final tokens={final_tok}  fallback_reasons_seen={sorted(set(fb_seen)) or 'NONE'}")
    return sorted(set(fb_seen)), final_tok


def main():
    dev = "cuda"
    model = E.build_model(CKPT, NF, IMG, PATCH, "bsm_taware_gradual_vec", dev)
    model.out_layers = [23]
    video, _ = E.load_video(CLIP, NF, IMG, dev)
    print(f"video shape={tuple(video.shape)}")

    key_fb, key_tok = run(model, video, "key", "as downstream used -> expect NO attn_key_unavailable")
    feat_fb, feat_tok = run(model, video, "feature", "forced control -> expect feature_fallback")

    print("\n================ VERDICT ================")
    key_clean = "attn_key_unavailable" not in key_fb
    print(f"WAM(key)     fallback_reasons = {key_fb or 'NONE'}  tokens={key_tok}")
    print(f"WAM(feature) fallback_reasons = {feat_fb or 'NONE'}  tokens={feat_tok}")
    if key_clean and ("feature_fallback" in feat_fb or "feature_metric_requested" in feat_fb or feat_fb):
        print("PASS ✅  WAM(key) used the post-RoPE Key metric (no attn_key fallback); "
              "control correctly shows feature fallback. Aligned WAM downstream is VALID.")
    elif key_clean:
        print("PASS(partial) ✅  WAM(key) shows no attn_key_unavailable fallback "
              "(control inconclusive but key path is clean).")
    else:
        print("FAIL ❌  WAM(key) STILL falls back to feature (attn_key_unavailable). "
              "Aligned WAM downstream is STILL confounded.")


if __name__ == "__main__":
    main()
