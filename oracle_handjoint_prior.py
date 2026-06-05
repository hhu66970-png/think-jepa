#!/usr/bin/env python
"""Oracle 手关节相关度先验:把 xyz_cam 52 关节投影到 token 网格(32t×16h×16w=8192),
得每-clip "手在这" 相关度 [t*h*w];并对校准 clip 求平均得 clip 无关先验(喂现有接线)。
纯几何,无 predictor、无梯度 → 零前向风险。reuse egodex 投影。"""
import os, sys, glob, argparse
import numpy as np
sys.path.insert(0, ".")
from egodex.utils.draw_utils import _project_point_to_image_plane

T_BIN, H, W, RES, PATCH = 32, 16, 16, 256, 16   # 在线编码几何:64帧 tubelet2→32, 256/16=16


def clip_handmap(npz_path, sigma=1.0, conf_thr=0.3):
    """返回 [T_BIN, H, W] in [0,1]:每个 token 网格被手关节命中的(高斯铺开)密度。"""
    d = np.load(npz_path, allow_pickle=True)
    xyz = d["xyz_cam"].astype(np.float32)        # [64,52,3] 相机系
    K = d["cam_int"].astype(np.float32)          # [3,3], cx=W/2 cy=H/2 (原图尺寸)
    conf = d["confs"].astype(np.float32) if "confs" in d.files else np.ones(xyz.shape[:2], np.float32)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    sx, sy = RES / (2.0 * cx), RES / (2.0 * cy)  # 原图(2cx×2cy)direct-resize→256×256(无crop)
    F = xyz.shape[0]
    grid = np.zeros((T_BIN, H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for f in range(F):
        tb = min(f // (F // T_BIN), T_BIN - 1)   # 64→32: f//2
        for j in range(xyz.shape[1]):
            if conf[f, j] < conf_thr:
                continue
            p = xyz[f, j]
            if p[2] <= 1e-4:                      # 在相机后方/无效
                continue
            uv = _project_point_to_image_plane(p[None, :], K)  # 原图像素
            uv = np.asarray(uv).reshape(-1)
            if uv.shape[0] < 2 or not np.all(np.isfinite(uv[:2])):
                continue
            u, v = float(uv[0]) * sx, float(uv[1]) * sy        # → 256×256 像素
            if not (0 <= u < RES and 0 <= v < RES):
                continue
            gx, gy = u / PATCH, v / PATCH         # 像素→网格坐标(连续)
            blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))
            grid[tb] = np.maximum(grid[tb], blob)  # 取 max(命中即亮)
    m = grid.max()
    if m > 1e-6:
        grid /= m
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/autodl-tmp/thinkjepa-work/downstream_ext30/manifests/train20.txt")
    ap.add_argument("--cache_root", default="/root/autodl-tmp/thinkjepa-work/cache_ext30/part2")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/thinkjepa-work/oracle_handprior")
    ap.add_argument("--max_clips", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # manifest 行 = npz 路径(或可定位 npz)。容错:直接用 cache_root 下所有 npz 也行。
    paths = []
    if os.path.exists(args.manifest):
        for ln in open(args.manifest):
            ln = ln.strip()
            if ln.endswith(".npz") and os.path.exists(ln):
                paths.append(ln)
    if not paths:
        paths = sorted(glob.glob(args.cache_root + "/*/*.npz"))
    paths = paths[:args.max_clips]
    print(f"[oracle] {len(paths)} clips")
    # --- 叠图验证:第一个 clip 的关节投影画到 256 帧上(看手对齐)---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d0 = np.load(paths[0], allow_pickle=True)
        K0 = d0["cam_int"].astype(np.float32); cx0, cy0 = float(K0[0, 2]), float(K0[1, 2])
        sx0, sy0 = RES / (2 * cx0), RES / (2 * cy0)
        fr = d0["imgs"].shape[0] // 2
        img = d0["imgs"][fr]
        xc = d0["xyz_cam"][fr]; cf = d0["confs"][fr] if "confs" in d0.files else np.ones(52)
        fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img); ax.set_title("oracle 投影验证:点应落在手上")
        for j in range(xc.shape[0]):
            if cf[j] < 0.3 or xc[j, 2] <= 1e-4:
                continue
            uv = np.asarray(_project_point_to_image_plane(xc[j][None, :], K0)).reshape(-1)
            u, v = uv[0] * sx0, uv[1] * sy0
            if 0 <= u < RES and 0 <= v < RES:
                ax.plot(u, v, "o", ms=4, color="lime", mec="k", mew=0.5)
        ax.set_xlim(0, RES); ax.set_ylim(RES, 0); ax.axis("off")
        op = "/root/autodl-tmp/thinkjepa-work/oracle_handprior/overlay_verify.png"
        fig.savefig(op, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[oracle] overlay saved -> {op}")
    except Exception as e:
        print(f"[oracle] overlay skip: {e}")
    acc = np.zeros((T_BIN, H, W), np.float32); n = 0
    for p in paths:
        try:
            g = clip_handmap(p)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}"); continue
        cid = os.path.basename(p).split("_")[0]
        np.savez_compressed(os.path.join(args.out_dir, f"rel_{cid}.npz"),
                            rel=g.reshape(-1).astype(np.float32), t=T_BIN, h=H, w=W)
        acc += g; n += 1
        print(f"  {cid}: hand-cells>{0.5}: {int((g>0.5).sum())}/{g.size}  peakframe_mean={g.mean():.3f}")
    if n:
        avg_thw = acc / n
        avg_hw = avg_thw.mean(0)                  # 空间平均(clip+时间无关)
        np.savez_compressed(os.path.join(args.out_dir, "handprior_avg.npz"),
                            rel=avg_hw.reshape(-1).astype(np.float32), t=T_BIN, h=H, w=W)
        np.savez_compressed(os.path.join(args.out_dir, "handprior_avg_thw.npz"),
                            rel=avg_thw.reshape(-1).astype(np.float32), t=T_BIN, h=H, w=W)
        # ASCII 看手区集中度
        print(f"\n[oracle] 平均空间手区 [{H}x{W}](clip+时间平均,越满=越分散→越需per-clip):")
        vis = avg_hw / (avg_hw.max() + 1e-6)
        chars = " .:-=+*#%@"
        for row in vis:
            print("  " + "".join(chars[min(len(chars) - 1, int(x * (len(chars) - 1)))] for x in row))
        print(f"[oracle] saved per-clip rel_*.npz + handprior_avg.npz(spatial) to {args.out_dir}")


if __name__ == "__main__":
    main()
