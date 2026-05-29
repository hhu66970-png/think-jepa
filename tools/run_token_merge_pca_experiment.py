"""Step 2 experiment: encoder-side token merge (方案A) vs no-merge, then PCA.

For each tiny-cache video we run the ThinkJEPA dense encoder twice:
  Group A (baseline): video -> encoder (NO token merge)            -> features -> PCA
  Group B (方案A)   : video -> encoder (local_2x2_same_time_vec)  -> features -> PCA

Both groups use the SAME video input and the SAME PCA settings. We record, per
group: explained-variance ratio per principal component + cumulative variance,
reconstruction relative error at fixed component counts, #components to reach
{90,95,99}% variance, token count (sequence length) before/after merge, encoder
wall-clock time and peak GPU memory. For group B we additionally run a
restore-dense pass to measure feature fidelity (cosine / relative L2) vs A.

Everything is computed from real forward passes; nothing is estimated.
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed, identical PCA reporting settings for BOTH groups.
PCA_CENTER = True          # subtract per-feature mean (standard PCA), no scaling
K_LIST = [1, 2, 3, 5, 8, 16, 32, 64, 128]
VAR_THRESHOLDS = [0.90, 0.95, 0.99]


def build_model(checkpoint, num_frames, img_size, patch_size, merge_layer, device):
    from src.models import vision_transformer

    merge_config = {
        "enabled": True,
        "merge_layers": (int(merge_layer),),
        "merge_ratio": 0.0,
        "strategy": "local_2x2_same_time_vec",  # 方案A
        "receiver": "max_norm",
        "restore_dense": False,
        "profile": False,
        "importance_source": "none",
    }
    model = vision_transformer.vit_large_rope(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=2,
        out_layers=None,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        merge_config=merge_config,
    )
    try:
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(blob, dict) and "encoder" in blob:
        state = blob["encoder"]
    elif isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    else:
        state = blob
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    assert type(model.token_merger).__name__ == "LocalTokenMerger", "main path must use base merger"
    return model.to(device).eval()


def load_video(npz_path, num_frames, img_size, device):
    d = np.load(npz_path, allow_pickle=True)
    imgs = d["imgs"]  # [F, H, W, 3] uint8
    total = int(imgs.shape[0])
    idx = np.linspace(0, total - 1, num_frames).round().astype(int)
    sel = imgs[idx].astype(np.float32) / 255.0          # [T, H, W, 3]
    t = torch.from_numpy(sel).permute(0, 3, 1, 2)       # [T, 3, H, W]
    if t.shape[-1] != img_size or t.shape[-2] != img_size:
        t = torch.nn.functional.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = (t - mean) / std
    v = t.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1, 3, T, H, W]
    return v.to(device), total


def pca_metrics(features):
    """features: [N, D] torch tensor. Returns dict + singular values (numpy)."""
    x = features.float()
    if PCA_CENTER:
        x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)                  # [min(N, D)]
    var = s ** 2
    total = var.sum().clamp_min(1e-12)
    evr = var / total
    cum = torch.cumsum(evr, dim=0)
    out = {
        "num_tokens": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "n_components_total": int(s.numel()),
        "evr_top16": [float(v) for v in evr[:16].tolist()],
        "cum_evr_top16": [float(v) for v in cum[:16].tolist()],
    }
    for th in VAR_THRESHOLDS:
        n = int((cum < th).sum().item()) + 1
        out[f"ncomp_{int(th * 100)}pct"] = int(min(n, s.numel()))
    for k in K_LIST:
        if k <= s.numel():
            retained = float(cum[k - 1].item())
            out[f"cum_evr_k{k}"] = retained
            out[f"recon_relerr_k{k}"] = float(max(0.0, 1.0 - retained) ** 0.5)
    return out, s.detach().cpu().numpy()


def timed_forward(model, video, *, enabled, ratio, restore_dense, repeats):
    model.merge_config.enabled = bool(enabled)
    model.merge_config.merge_ratio = float(ratio)
    model.merge_config.restore_dense = bool(restore_dense)
    with torch.no_grad():  # warmup
        out, infos = model(video, return_merge_info=True, restore_dense=restore_dense)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out, infos = model(video, return_merge_info=True, restore_dense=restore_dense)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return out, infos, times, peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--npz_glob", default="/root/autodl-tmp/thinkjepa-work/tiny-cache/part2/*/*.npz")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--merge_layer", type=int, default=8)
    ap.add_argument("--ratios", default="0.05,0.125,0.25")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    npz_files = sorted(glob.glob(args.npz_glob))
    assert npz_files, f"no npz under {args.npz_glob}"
    print(f"[INFO] videos={len(npz_files)} ratios={ratios} layer={args.merge_layer} "
          f"frames={args.num_frames} res={args.img_size}")

    model = build_model(args.checkpoint, args.num_frames, args.img_size,
                        args.patch_size, args.merge_layer, args.device)

    results = {
        "meta": {
            "checkpoint": args.checkpoint, "num_frames": args.num_frames,
            "img_size": args.img_size, "patch_size": args.patch_size,
            "merge_layer": args.merge_layer, "ratios": ratios, "repeats": args.repeats,
            "strategy": "local_2x2_same_time_vec",
            "pca_center": PCA_CENTER, "k_list": K_LIST, "var_thresholds": VAR_THRESHOLDS,
            "device_name": torch.cuda.get_device_name(0),
        },
        "samples": {},
    }
    spectra = {}
    csv_rows = []

    for npz in npz_files:
        name = os.path.splitext(os.path.basename(npz))[0]
        video, total_frames = load_video(npz, args.num_frames, args.img_size, args.device)
        print(f"\n[VIDEO] {name} src_frames={total_frames} input={tuple(video.shape)}")
        sample = {"npz": npz, "total_frames_in_npz": total_frames}

        # ---- Group A: no merge ----
        outA, _, tA, memA = timed_forward(model, video, enabled=False, ratio=0.0,
                                          restore_dense=True, repeats=args.repeats)
        featA = outA[0]  # [N, D]
        pcaA, sA = pca_metrics(featA)
        sample["A_baseline"] = {
            "tokens": int(featA.shape[0]), "time_ms_mean": float(np.mean(tA)),
            "time_ms_median": float(np.median(tA)), "peak_mem_mb": float(memA),
            "pca": pcaA,
        }
        spectra[f"{name}__A"] = sA
        print(f"  A: tokens={featA.shape[0]} time={np.mean(tA):.1f}ms mem={memA:.0f}MB "
              f"ncomp95={pcaA['ncomp_95pct']} recon_k8={pcaA.get('recon_relerr_k8')}")
        csv_rows.append([name, "A_baseline", "-", featA.shape[0], f"{np.mean(tA):.2f}",
                         f"{memA:.0f}", pcaA["ncomp_90pct"], pcaA["ncomp_95pct"],
                         pcaA["ncomp_99pct"], pcaA.get("recon_relerr_k8"), "-", "-"])

        featA_f = featA.float()
        normA = featA_f.norm()

        # ---- Group B: 方案A merge at each ratio ----
        sample["B_scheme_a"] = {}
        for r in ratios:
            outB, infoB, tB, memB = timed_forward(model, video, enabled=True, ratio=r,
                                                  restore_dense=False, repeats=args.repeats)
            featB = outB[0]  # [N', D] compressed
            pcaB, sB = pca_metrics(featB)
            info = infoB[0] if infoB else {}
            # fidelity vs A (aligned via restore_dense=True)
            outBr, _, _, _ = timed_forward(model, video, enabled=True, ratio=r,
                                           restore_dense=True, repeats=1)
            featBr = outBr[0].float()
            cos = torch.nn.functional.cosine_similarity(featA_f, featBr, dim=-1).mean().item()
            rel_l2 = ((featBr - featA_f).norm() / normA.clamp_min(1e-12)).item()
            entry = {
                "ratio": r,
                "tokens": int(featB.shape[0]),
                "tokens_merged": int(info.get("num_accepted", 0)),
                "seq_reduction": 1.0 - featB.shape[0] / float(featA.shape[0]),
                "time_ms_mean": float(np.mean(tB)),
                "time_ms_median": float(np.median(tB)),
                "peak_mem_mb": float(memB),
                "fidelity_mean_cosine_vs_A": float(cos),
                "fidelity_rel_l2_vs_A": float(rel_l2),
                "pca": pcaB,
            }
            sample["B_scheme_a"][f"r{r}"] = entry
            spectra[f"{name}__B_r{r}"] = sB
            print(f"  B r={r}: tokens={featB.shape[0]} (-{entry['seq_reduction']*100:.1f}%) "
                  f"time={np.mean(tB):.1f}ms mem={memB:.0f}MB cos={cos:.4f} relL2={rel_l2:.4f} "
                  f"ncomp95={pcaB['ncomp_95pct']} recon_k8={pcaB.get('recon_relerr_k8')}")
            csv_rows.append([name, "B_scheme_a", r, featB.shape[0], f"{np.mean(tB):.2f}",
                             f"{memB:.0f}", pcaB["ncomp_90pct"], pcaB["ncomp_95pct"],
                             pcaB["ncomp_99pct"], pcaB.get("recon_relerr_k8"),
                             f"{cos:.4f}", f"{rel_l2:.4f}"])

        results["samples"][name] = sample

    # ---- save ----
    json_path = os.path.join(args.out_dir, "pca_experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(args.out_dir, "pca_singular_values.npz"), **spectra)
    csv_path = os.path.join(args.out_dir, "pca_experiment_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "group", "ratio", "tokens", "enc_time_ms", "peak_mem_mb",
                    "ncomp_90pct", "ncomp_95pct", "ncomp_99pct", "recon_relerr_k8",
                    "cos_vs_A", "relL2_vs_A"])
        w.writerows(csv_rows)
    print(f"\n[SAVED] {json_path}\n[SAVED] {csv_path}\n[SAVED] {os.path.join(args.out_dir, 'pca_singular_values.npz')}")


if __name__ == "__main__":
    main()
