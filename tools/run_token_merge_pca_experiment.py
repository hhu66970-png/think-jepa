"""Encoder-side token-merge benchmark + PCA: dense baseline, 方案A reference,
and Gradual K-BSM (``bsm_ksim_gradual_vec``).

For each tiny-cache video we run the ThinkJEPA dense encoder under several
configs that share the SAME video input and the SAME PCA settings:
  dense baseline : video -> encoder (NO token merge, restore_dense=True) -> PCA
  方案A reference: video -> encoder (local_2x2_same_time_vec, single layer)  -> PCA
  K-BSM          : video -> encoder (bsm_ksim_gradual_vec, gradual multi-layer) -> PCA

We record, per config: explained-variance ratio per principal component +
cumulative variance, reconstruction relative error at fixed component counts,
#components to reach {90,95,99}% variance, token count (sequence length)
before/after merge, PER-LAYER token trajectory, encoder wall-clock time, peak GPU
memory, speedup/mem-saving vs dense, and feature fidelity (cosine / relative L2)
vs the dense baseline via a restore_dense=True alignment pass.

Everything is computed from real forward passes; nothing is estimated.

K-EXPOSURE: bsm_ksim_gradual_vec prefers the post-RoPE attention Key (mean over
heads) captured SDPA-safely by RoPEAttention.forward when bsm_match_metric="key".
If the stash is unavailable it falls back to block-output hidden-feature cosine
and the per-layer info record flags it via matching_metric="feature_fallback".
This runner reports whatever the merger actually used (per-layer matching_metric).
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

# Fixed, identical PCA reporting settings for ALL configs.
PCA_CENTER = True          # subtract per-feature mean (standard PCA), no scaling
K_LIST = [1, 2, 3, 5, 8, 16, 32, 64, 128]
VAR_THRESHOLDS = [0.90, 0.95, 0.99]


# --------------------------------------------------------------------------
# Small parsing / config helpers
# --------------------------------------------------------------------------
def parse_layers(spec):
    """'4,8,12' -> [4,8,12]; '4-12' -> [4..12]; '4-12:2' -> [4,6,8,10,12]."""
    spec = str(spec).strip()
    if not spec:
        return []
    if "-" in spec and "," not in spec:
        body, _, step_s = spec.partition(":")
        lo_s, _, hi_s = body.partition("-")
        lo, hi = int(lo_s), int(hi_s)
        step = int(step_s) if step_s else 1
        if step <= 0:
            raise ValueError(f"step must be >0 in --merge_layers {spec!r}")
        return list(range(lo, hi + 1, step))
    return [int(v) for v in spec.split(",") if v.strip()]


def parse_floats(spec):
    return [float(v) for v in str(spec).split(",") if str(v).strip()]


def ensure_token_merger(model):
    """Single opt-in point that (re)binds the correct merger to the model.

    bsm_ksim_gradual_vec is a diagnostic-only strategy and requires
    DiagnosticTokenMerger; main-path strategies (A/B/C) use LocalTokenMerger.
    """
    from src.models.utils.token_merge import LocalTokenMerger
    from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger

    diagnostic_only = set(DiagnosticTokenMerger.VECTORIZED_STRATEGIES) - set(
        LocalTokenMerger.VECTORIZED_STRATEGIES
    )
    strategy = str(model.merge_config.strategy)
    want_cls = DiagnosticTokenMerger if strategy in diagnostic_only else LocalTokenMerger
    current = getattr(model, "token_merger", None)
    if current is None or type(current) is not want_cls:
        model.token_merger = want_cls(model.merge_config)
    else:
        current.config = model.merge_config
    return type(model.token_merger).__name__


def apply_merge_config(model, *, enabled, strategy, merge_layers, merge_ratio,
                       restore_dense, receiver="max_norm", bsm_match_metric="key",
                       bsm_partition="positional", pre_merge_ratio=0.0):
    """Mutate model.merge_config in place and (re)bind the right merger.

    Re-runs normalize_merge_config so the strategy's validation (multi-layer
    allowance, etc.) is exercised exactly as in production. Raises if the config
    is rejected -> caller decides fail-loud vs record.
    """
    from src.models.utils.token_merge import normalize_merge_config
    cfg = {
        "enabled": bool(enabled),
        "merge_layers": tuple(int(v) for v in merge_layers),
        "merge_ratio": float(merge_ratio),
        "strategy": str(strategy),
        "receiver": str(receiver),
        "restore_dense": bool(restore_dense),
        "profile": True,                 # enables per-segment timing in forward()
        "importance_source": "none",     # BSM K/hidden-cosine needs no importance
        "bsm_match_metric": str(bsm_match_metric),
        "bsm_partition": str(bsm_partition),
        "pre_merge_ratio": float(pre_merge_ratio),
    }
    model.merge_config = normalize_merge_config(cfg)   # may raise ValueError
    merger_cls = ensure_token_merger(model)
    return merger_cls


def build_model(checkpoint, num_frames, img_size, patch_size, initial_strategy, device):
    from src.models import vision_transformer

    merge_config = {
        "enabled": True,
        "merge_layers": (),
        "merge_ratio": 0.0,
        "strategy": str(initial_strategy),
        "receiver": "max_norm",
        "restore_dense": False,
        "profile": True,
        "importance_source": "none",
        "bsm_match_metric": "key",
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
    print(f"[INFO] initial merger={type(model.token_merger).__name__} strategy={initial_strategy}")
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


def timed_forward(model, video, *, restore_dense, repeats, warmup):
    """Returns (out, merge_infos, profile, times_ms, peak_mem_mb).

    Timing covers the FULL encoder forward on the chosen path. sync before every
    t0 and after the call; warmup passes excluded; peak mem reset after warmup so
    it reflects the steady-state of the timed region only.
    """
    infos = None
    for _ in range(max(1, warmup)):
        with torch.no_grad():
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
    profile = dict(getattr(model, "last_forward_profile", {}) or {})
    return out, (infos or []), profile, times, peak_mb


def layer_trajectory(merge_infos, original_tokens):
    """[{layer, tokens_before, tokens_after, num_accepted, kept_ratio, ...}, ...]"""
    traj = []
    prev = int(original_tokens)
    for info in merge_infos:
        after = int(info.get("num_tokens_after", prev))
        traj.append({
            "layer": int(info.get("layer", -1)),
            "tokens_before": int(info.get("num_tokens_before", prev)),
            "tokens_after": after,
            "num_accepted": int(info.get("num_accepted") or 0),
            "kept_ratio": float(after / max(1, original_tokens)),
            "implementation": info.get("implementation"),
            "fallback_reason": info.get("fallback_reason"),
            "matching_metric": info.get("matching_metric", info.get("bsm_match_metric")),
        })
        prev = after
    return traj


def fidelity_vs_dense(model, video, dense_feat_f, dense_norm, *, warmup):
    """restore_dense=True alignment pass; cosine (mean over tokens) + relL2."""
    out, _, _, _, _ = timed_forward(model, video, restore_dense=True,
                                    repeats=1, warmup=max(1, warmup))
    feat_r = out[0].float()
    cos = torch.nn.functional.cosine_similarity(dense_feat_f, feat_r, dim=-1).mean().item()
    rel_l2 = ((feat_r - dense_feat_f).norm() / dense_norm.clamp_min(1e-12)).item()
    return float(cos), float(rel_l2)


def build_config_matrix(args):
    """Returns list of merge configs (excludes the dense baseline)."""
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    a_layer = int(args.baseline_scheme_a_layer)
    gradual_layers = parse_layers(args.merge_layers)
    r_list = parse_floats(args.r_per_layer)
    a_ratios = parse_floats(args.scheme_a_ratios)
    configs = []
    # optional single-layer 方案A reference
    if a_layer >= 0:
        for r in a_ratios:
            configs.append({
                "tag": f"A_ref__L{a_layer}__r{r}",
                "group": "A_scheme_a",
                "strategy": "local_2x2_same_time_vec",
                "merge_layers": [a_layer],
                "merge_ratio": float(r),
                "r_per_layer": float(r),
            })
    # the strategy(ies) under test (gradual, multi-layer)
    for strat in strategies:
        if strat == "local_2x2_same_time_vec":
            continue  # already covered as the A reference above
        for r in r_list:
            # ratio<1 -> direct per-layer ratio; int>=1 -> absolute (resolved later)
            ratio = r if r < 1.0 else None
            lo = gradual_layers[0] if gradual_layers else -1
            hi = gradual_layers[-1] if gradual_layers else -1
            configs.append({
                "tag": f"{strat}__L{lo}-{hi}__r{r}",
                "group": strat,
                "strategy": strat,
                "merge_layers": list(gradual_layers),
                "merge_ratio": ratio,           # may be None -> resolve later
                "r_per_layer": float(r),
            })
    # RLT-style pre-encoder temporal merge (single merge before block 0)
    for r in parse_floats(getattr(args, "pre_merge_ratios", "") or ""):
        configs.append({
            "tag": f"PRE_temporal__r{r}",
            "group": "pre_merge_temporal",
            "strategy": "bsm_ksim_gradual_vec",
            "merge_layers": [],
            "merge_ratio": float(r),
            "r_per_layer": float(r),
            "pre_merge_ratio": float(r),
        })
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--npz_glob", default="/root/autodl-tmp/thinkjepa-work/tiny-cache/part2/*/*.npz")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)

    # --- strategy selection -------------------------------------------------
    ap.add_argument("--strategy", default="bsm_ksim_gradual_vec",
                    help="comma-separated strategies to benchmark, e.g. "
                         "'bsm_ksim_gradual_vec' or "
                         "'local_2x2_same_time_vec,bsm_ksim_gradual_vec'")
    ap.add_argument("--baseline_scheme_a_layer", type=int, default=8,
                    help="layer for the single-layer 方案A reference; <0 disables it")
    ap.add_argument("--scheme_a_ratios", default="0.125",
                    help="merge_ratio(s) for the 方案A reference, comma list")

    # --- gradual multi-layer schedule for BSM ------------------------------
    ap.add_argument("--merge_layers", default="4-12:2",
                    help="merge layers: comma list '4,8,12', range '4-12', "
                         "or range:step '4-12:2'")
    ap.add_argument("--r_per_layer", default="0.15",
                    help="per-layer reduction; float<1 = ratio of current tokens, "
                         "int>=1 = absolute tokens removed per layer. Comma list "
                         "to sweep, e.g. '0.1,0.15,0.2'")
    ap.add_argument("--bsm_match_metric", default="key", choices=["key", "feature"],
                    help="BSM matching metric: 'key' = post-RoPE attention Key "
                         "cosine (SDPA-safe stash, falls back to feature if "
                         "unavailable), 'feature' = block-output hidden cosine")
    ap.add_argument("--bsm_partition", default="positional",
                    choices=["positional", "temporal"],
                    help="BSM bipartite split: 'positional' = ToMe even/odd; "
                         "'temporal' = even/odd by tubelet so tokens merge ACROSS "
                         "time (exploits inter-frame redundancy; single merge layer)")
    ap.add_argument("--pre_merge_ratios", default="",
                    help="comma list of ratios for RLT-style pre-encoder temporal "
                         "merge (one merge before block 0); use with --bsm_partition temporal")

    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--allow_fallback", action="store_true",
                    help="if set, do NOT hard-fail when a strategy/multi-layer "
                         "config is rejected by normalize_merge_config; record the "
                         "error instead (default: fail loudly so a misconfigured "
                         "BSM never silently degrades to baseline A)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    # first strategy under test that is not the A reference -> binds initial merger
    initial_strategy = next(
        (s for s in strategies if s != "local_2x2_same_time_vec"),
        "local_2x2_same_time_vec",
    )
    npz_files = sorted(glob.glob(args.npz_glob))
    assert npz_files, f"no npz under {args.npz_glob}"
    print(f"[INFO] videos={len(npz_files)} strategies={strategies} "
          f"merge_layers={args.merge_layers} r_per_layer={args.r_per_layer} "
          f"frames={args.num_frames} res={args.img_size} metric={args.bsm_match_metric}")

    model = build_model(args.checkpoint, args.num_frames, args.img_size,
                        args.patch_size, initial_strategy, args.device)

    results = {
        "meta": {
            "checkpoint": args.checkpoint, "num_frames": args.num_frames,
            "img_size": args.img_size, "patch_size": args.patch_size,
            "strategies": strategies,
            "merge_layers_spec": args.merge_layers,
            "merge_layers_resolved": parse_layers(args.merge_layers),
            "r_per_layer": parse_floats(args.r_per_layer),
            "scheme_a_layer": args.baseline_scheme_a_layer,
            "scheme_a_ratios": parse_floats(args.scheme_a_ratios),
            "bsm_match_metric_requested": args.bsm_match_metric,
            "repeats": args.repeats, "warmup": args.warmup,
            "pca_center": PCA_CENTER, "k_list": K_LIST, "var_thresholds": VAR_THRESHOLDS,
            "device_name": torch.cuda.get_device_name(0),
            "k_exposure": ("post_rope_key_stash_sdpa_safe (RoPEAttention.forward "
                           "stashes mean-over-heads post-RoPE key before SDPA; the "
                           "merger flags per-layer matching_metric, falling back to "
                           "hidden-feature cosine if the stash is unavailable)"),
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

        # ---- dense baseline (no merge) ----
        apply_merge_config(model, enabled=False, strategy="local_2x2_same_time_vec",
                           merge_layers=[], merge_ratio=0.0, restore_dense=True)
        model.merge_config.enabled = False  # force true dense
        outD, _, profD, tD, memD = timed_forward(
            model, video, restore_dense=True, repeats=args.repeats, warmup=args.warmup)
        featD = outD[0]
        pcaD, sD = pca_metrics(featD)
        dense_feat_f = featD.float()
        dense_norm = dense_feat_f.norm()
        sample["dense_baseline"] = {
            "tokens": int(featD.shape[0]),
            "time_ms_mean": float(np.mean(tD)), "time_ms_median": float(np.median(tD)),
            "time_ms_std": float(np.std(tD)), "peak_mem_mb": float(memD),
            "profile": profD, "pca": pcaD,
        }
        spectra[f"{name}__dense"] = sD
        print(f"  dense: tokens={featD.shape[0]} time={np.median(tD):.2f}ms "
              f"mem={memD:.0f}MB ncomp95={pcaD['ncomp_95pct']} "
              f"recon_k8={pcaD.get('recon_relerr_k8')}")
        csv_rows.append([name, "dense_baseline", "-", "-", featD.shape[0],
                         f"{np.mean(tD):.3f}", f"{np.median(tD):.3f}", f"{memD:.0f}",
                         pcaD["ncomp_90pct"], pcaD["ncomp_95pct"], pcaD["ncomp_99pct"],
                         pcaD.get("recon_relerr_k8"), "1.0", "0.0"])

        # ---- each merge config ----
        for cfg in build_config_matrix(args):
            ratio = cfg["merge_ratio"]
            if ratio is None:  # absolute r_per_layer -> ratio vs ORIGINAL token count
                ratio = float(cfg["r_per_layer"]) / float(featD.shape[0])
            try:
                merger_cls = apply_merge_config(
                    model, enabled=True, strategy=cfg["strategy"],
                    merge_layers=cfg["merge_layers"], merge_ratio=ratio,
                    restore_dense=False, bsm_match_metric=args.bsm_match_metric,
                    bsm_partition=args.bsm_partition,
                    pre_merge_ratio=cfg.get("pre_merge_ratio", 0.0))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if not args.allow_fallback:
                    raise RuntimeError(
                        f"[CONFIG REJECTED] {cfg['tag']}: {msg}. "
                        f"Refusing to silently fall back to baseline A. "
                        f"Likely normalize_merge_config still rejects multi-layer "
                        f"for {cfg['strategy']}.") from e
                sample.setdefault("errors", {})[cfg["tag"]] = msg
                print(f"  [SKIP] {cfg['tag']} -> {msg}")
                continue

            # timed COMPRESSED path
            outM, infoM, profM, tM, memM = timed_forward(
                model, video, restore_dense=False,
                repeats=args.repeats, warmup=args.warmup)
            featM = outM[0]
            pcaM, sM = pca_metrics(featM)
            traj = layer_trajectory(infoM, original_tokens=int(featD.shape[0]))
            cos, rel_l2 = fidelity_vs_dense(
                model, video, dense_feat_f, dense_norm, warmup=args.warmup)
            metric_used = (infoM[-1].get("matching_metric",
                           infoM[-1].get("bsm_match_metric", "n/a"))
                           if infoM else "n/a")
            num_merge_layers_active = sum(1 for s in traj if s["num_accepted"] > 0)

            entry = {
                "tag": cfg["tag"], "group": cfg["group"], "strategy": cfg["strategy"],
                "merger_class": merger_cls,
                "merge_layers": cfg["merge_layers"],
                "r_per_layer_requested": cfg["r_per_layer"],
                "merge_ratio_used": float(ratio),
                "matching_metric": metric_used,
                "num_merge_layers_active": int(num_merge_layers_active),
                "tokens_final": int(featM.shape[0]),
                "seq_reduction_total": 1.0 - featM.shape[0] / float(featD.shape[0]),
                "layer_trajectory": traj,
                "speedup_vs_dense": float(np.median(tD) / max(1e-9, np.median(tM))),
                "time_ms_mean": float(np.mean(tM)), "time_ms_median": float(np.median(tM)),
                "time_ms_std": float(np.std(tM)), "peak_mem_mb": float(memM),
                "mem_saving_vs_dense": 1.0 - memM / max(1e-9, memD),
                "fidelity_mean_cosine_vs_dense": cos,
                "fidelity_rel_l2_vs_dense": rel_l2,
                "profile": profM, "pca": pcaM,
            }
            sample.setdefault("merge_configs", {})[cfg["tag"]] = entry
            spectra[f"{name}__{cfg['tag']}"] = sM
            tok_path = ">".join(str(s["tokens_after"]) for s in traj) or "(none)"
            print(f"  {cfg['tag']}: tokens {featD.shape[0]}>{tok_path} "
                  f"layers_active={num_merge_layers_active} "
                  f"time={np.median(tM):.2f}ms (x{entry['speedup_vs_dense']:.2f}) "
                  f"mem={memM:.0f}MB cos={cos:.4f} relL2={rel_l2:.4f} "
                  f"metric={metric_used}")
            csv_rows.append([name, cfg["group"], cfg["tag"],
                             ";".join(map(str, cfg["merge_layers"])),
                             featM.shape[0], f"{np.mean(tM):.3f}", f"{np.median(tM):.3f}",
                             f"{memM:.0f}", pcaM["ncomp_90pct"], pcaM["ncomp_95pct"],
                             pcaM["ncomp_99pct"], pcaM.get("recon_relerr_k8"),
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
        w.writerow(["sample", "group", "tag", "merge_layers", "tokens_final",
                    "enc_time_ms_mean", "enc_time_ms_median", "peak_mem_mb",
                    "ncomp_90pct", "ncomp_95pct", "ncomp_99pct", "recon_relerr_k8",
                    "cos_vs_dense", "relL2_vs_dense"])
        w.writerows(csv_rows)

    # per-LAYER token trajectory (one row per merge layer per config)
    traj_path = os.path.join(args.out_dir, "pca_layer_trajectory.csv")
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "tag", "strategy", "layer", "tokens_before",
                    "tokens_after", "num_accepted", "kept_ratio", "matching_metric"])
        for sname, s in results["samples"].items():
            for tag, e in (s.get("merge_configs") or {}).items():
                for step in e["layer_trajectory"]:
                    w.writerow([sname, tag, e["strategy"], step["layer"],
                                step["tokens_before"], step["tokens_after"],
                                step["num_accepted"], f"{step['kept_ratio']:.4f}",
                                step["matching_metric"]])
    print(f"\n[SAVED] {json_path}\n[SAVED] {csv_path}\n[SAVED] {traj_path}\n"
          f"[SAVED] {os.path.join(args.out_dir, 'pca_singular_values.npz')}")


if __name__ == "__main__":
    main()
