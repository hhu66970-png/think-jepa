#!/usr/bin/env python
"""Dense JEPA/V-JEPA feature PCA visualization for ThinkJEPA.

此脚本用于将 ThinkJEPA / V-JEPA 模型提取的密集特征 (dense features) 
进行主成分分析 (PCA) 降维，并映射到 RGB 颜色空间进行可视化。
默认渲染器使用双阶段 (two-stage) PCA 配方：
第一阶段 PCA 估计前景掩码 (foreground mask)，
第二阶段 PCA 仅在前景 token 上拟合 RGB 颜色，以排除背景的特征干扰。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ImageNet 默认归一化参数，用于输入模型前的图像预处理
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# 默认的 VJEPA2 官方权重下载链接
VJEPA2_VITL_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    """将输入的逗号/分号分隔字符串解析为整数列表"""
    if value is None:
        return list(default)
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    """将输入的逗号/分号分隔字符串解析为浮点数列表"""
    if value is None:
        return list(default)
    text = str(value).strip()
    if not text:
        return []
    return [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Create V-JEPA-style dense feature PCA visualizations for ThinkJEPA."
    )
    # 互斥组：这几个输入方式只能选一个
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", type=str, help="原始输入的 .mp4 视频路径.")
    src.add_argument("--npz", type=str, help="ThinkJEPA 缓存的 .npz 数据路径.")
    src.add_argument("--npz_glob", type=str, help="批量处理 ThinkJEPA 缓存 .npz 文件的 glob 匹配模式.")

    # 基础运行选项
    parser.add_argument("--self_test", action="store_true", help="跑一个基于生成数据的 PCA 测试并退出.")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径.")
    parser.add_argument("--num_frames", type=int, default=64, help="编码的帧数.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="在编码之前对帧进行均匀下采样的最大保留帧数。适用于显存不够或者处理过长视频.",
    )
    parser.add_argument("--img_size", type=int, default=256, help="输入模型的空间分辨率 (高和宽).")
    parser.add_argument("--patch_size", type=int, default=16, help="模型把图像切分成 token 的 patch 大小.")
    
    # 裁剪控制
    parser.add_argument(
        "--crop_mode",
        choices=("none", "motion", "motion_window", "center"),
        default="none",
        help="可选的预编码空间裁剪。motion 可以将注意力集中在画面中处于运动的部分.",
    )
    parser.add_argument(
        "--crop_padding",
        type=float,
        default=0.30,
        help="在提取出来的运动选框外围扩充的 padding 比例.",
    )
    parser.add_argument(
        "--crop_quantile",
        type=float,
        default=0.82,
        help="使用 --crop_mode motion 时用于生成运动热力图的激活阈值分位数.",
    )
    parser.add_argument(
        "--crop_window_fraction",
        type=float,
        default=0.68,
        help="使用 --crop_mode motion_window 时的裁剪框长度占比 (相对于短边).",
    )
    
    # 模型架构与采样
    parser.add_argument(
        "--model_arch",
        type=str,
        default="vit_large_rope",
        help="使用的 vjepa2 Transformer encoder 结构名, 如 vit_large_rope.",
    )
    parser.add_argument(
        "--sample_mode",
        choices=("uniform", "centered"),
        default="uniform",
        help="针对 --video 视频输入的帧采样模式 (均匀跳帧采样 or 以目标帧为中心局部采样).",
    )
    parser.add_argument("--target_frame", type=int, default=None, help="目标帧索引 (用于 centered 采样).")
    parser.add_argument("--stride", type=int, default=1, help="处理帧时的步长 (用于 centered 采样).")
    parser.add_argument("--token_t", type=int, default=None, help="强行指定要输出 PCA 单图的时间 token 索引.")
    
    # PCA 和特征配方配置
    parser.add_argument(
        "--pca_norm",
        choices=("center", "l2_center", "standardize"),
        default="l2_center",
        help="执行 PCA 前的特征归一化方式：简单减均值、L2归一化后减均值、或者标准化.",
    )
    parser.add_argument(
        "--pca_recipe",
        choices=("foreground", "simple"),
        default="foreground",
        help="foreground(两阶段前景分离 PCA); simple(所有 tokens 参与单次 PCA).",
    )
    parser.add_argument(
        "--foreground_method",
        choices=("pc1", "norm", "hybrid"),
        default="hybrid",
        help="前景计算方式: pc1(按绝对值最大的第一主成分), norm(特征向量长度), hybrid(两者结合).",
    )
    parser.add_argument(
        "--foreground_quantile",
        type=float,
        default=0.70,
        help="前景阈值的切割分位数。如 0.7 意味着把得分前 30% 的 tokens 视为前景.",
    )
    parser.add_argument(
        "--auto_quantiles",
        type=str,
        default="0.62,0.68,0.74,0.80",
        help="逗号分隔的多个分位数，用于自动化选出视觉效果最佳的分位数跑出来的掩码结果.",
    )
    parser.add_argument(
        "--smooth_foreground_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 token 的二维空间网格上对前景打分结果做 3x3 的最大/平均池化平滑去噪.",
    )
    parser.add_argument(
        "--out_layers",
        type=str,
        default="5,11,17,23",
        help="逗号分隔需要抽取做对比的 0-based Transformer 编码器层下标.",
    )
    
    # 后渲染输出参数
    parser.add_argument(
        "--render_interp",
        choices=("nearest", "bicubic"),
        default="nearest",
        help="渲染主图使用的插值方式。nearest (保留马赛克格子的原始感)，此外也会保留一张 bicubic 图做比对.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="RGB 映射健壮性的百分位数。1.0 代表将每个通道投影按 [1%%, 99%%] 线性映射到 0~255.",
    )
    parser.add_argument(
        "--background_gray",
        type=int,
        default=145,
        help="渲染前景 PCA 模式下的背景部分的灰色默认值 (0-255).",
    )
    parser.add_argument(
        "--background_mode",
        choices=("gray", "pca", "desaturate"),
        default="pca",
        help="非前景 token 的渲染方案 (涂灰、仍旧展示基底 PCA 色、或者做降饱和偏灰处理).",
    )
    parser.add_argument(
        "--background_desaturate",
        type=float,
        default=0.35,
        help="如果 bg_mode 为 desaturate 时的混合系数.",
    )
    parser.add_argument(
        "--rgb_smooth",
        type=float,
        default=0.18,
        help="混合原 PCA RGB 特征图与经过 3x3 空间平滑模糊后的 RGB 的比例，消除突刺.",
    )
    parser.add_argument(
        "--rgb_saturation",
        type=float,
        default=1.25,
        help="RGB图的整体色彩饱和度乘数.",
    )
    parser.add_argument(
        "--rgb_gamma",
        type=float,
        default=0.92,
        help="整体 Gamma 值校正。低于 1.0 的值能提亮画面.",
    )
    parser.add_argument(
        "--fusion_layers",
        type=str,
        default="",
        help="可选，将多个指定曾拼接成一个超长特征后一同 PCA (融合通道).",
    )
    parser.add_argument(
        "--use_cached_feats",
        action="store_true",
        help="是否跳过前向网络提取，直接使用 .npz 里存好的 vjepa_feats 二进制文件.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="使用的计算设备，由于都是前向计算，CPU也可行。")
    parser.add_argument("--out_dir", type=str, required=False, help="图像结果输出目标目录.")
    parser.add_argument("--sheet_cols", type=int, default=5, help="在联系表 (contact sheet) 中要展示最多几张横向图.")
    return parser.parse_args()


def configure_paths():
    """解析执行根目录并注入对应模块所需的 sys.path """
    repo_root = Path(__file__).resolve().parents[1]
    vjepa_root = Path(os.environ.get("VJEPA2_ROOT", repo_root / "vjepa2")).resolve()
    for path in (repo_root, repo_root / "cache_train", vjepa_root, vjepa_root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("VJEPA2_ROOT", str(vjepa_root))
    return repo_root, vjepa_root


def resolve_checkpoint(repo_root: Path, explicit_path: str | None):
    """确定应该拿哪个检查点去跑推理"""
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    env_path = os.environ.get("THINKJEPA_JEPA_VITL_PT")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"THINKJEPA_JEPA_VITL_PT does not exist: {path}")

    candidates = [
        repo_root / "vjepa2" / "vitl.pt",
        repo_root / "checkpoints" / "vitl.pt",
        repo_root / "checkpoints" / "thinkjepa_jepa_vitl.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    checked = "\n".join(f"  - {item}" for item in candidates)
    raise FileNotFoundError(
        "Cannot resolve V-JEPA2 ViT-L checkpoint.\n"
        f"Download it with:\n  wget {VJEPA2_VITL_URL} -O {repo_root / 'vjepa2' / 'vitl.pt'}\n"
        "Or pass --checkpoint /path/to/vitl.pt.\n"
        f"Checked:\n{checked}"
    )


def load_dense_jepa_model(checkpoint_path, img_size, num_frames, patch_size, out_layers, device, model_arch):
    """
    实例化 V-JEPA2 encoder 架构，并加载指定的预训练权重 (只用其编码器部分以提取特征)。
    """
    from vjepa2.src.models import vision_transformer

    # 参数里的架构如果不在 vision_transformer 包里，报错返回
    if not hasattr(vision_transformer, model_arch):
        available = [name for name in dir(vision_transformer) if name.startswith("vit_")]
        raise ValueError(f"Unknown --model_arch {model_arch!r}; available={available}")

    factory = getattr(vision_transformer, model_arch)
    # 实例化模型
    model = factory(
        img_size=(img_size, img_size),  # 模型支持非正方面积但这里写死了 S x S
        num_frames=num_frames,
        patch_size=patch_size,       # 切 patch 用到的尺寸，如 16x16
        tubelet_size=2,              # 时间帧打组单位，2 意味着每两帧合为一个时空 Token
        out_layers=out_layers,       # 我们要抽取的层的注册字典
        use_sdpa=True,               # 默认使用原生的 Scaled Dot Product Attention 硬件加速
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
    )

    # 尝试加载只含权重的字典并规避可能的序列化冲突
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(checkpoint_path, map_location="cpu")

    # 定位有效权重的 root key (考虑到有时是 checkpoint字典，有时包含 optimizer state)
    if isinstance(blob, dict) and "encoder" in blob:
        state = blob["encoder"]
    elif isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(blob)!r}")

    # 清理多卡或特定套件留下的多级命名前缀
    clean = {}
    for key, value in state.items():
        clean[key.replace("module.", "").replace("backbone.", "")] = value

    # 加载状态，并不做严格匹配 (因为只是需要 encoder 部分，没必要配分类器头)
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"[INFO] load_state_dict: {msg}", flush=True)
    
    # 进入验证模式 (消除 Dropout 随机性等)，放到计算设备
    model.to(device)
    model.eval()
    return model


def sample_frame_indices(total, num_frames, mode, target_frame, stride):
    """计算下采样需要的帧下标"""
    if mode == "uniform":
        values = np.linspace(0, total - 1, num=num_frames)
        return np.clip(np.round(values), 0, total - 1).astype(np.int64)

    if target_frame is None:
        raise ValueError("--target_frame is required for --sample_mode centered")
    center = int(np.clip(target_frame, 0, total - 1))
    offsets = (np.arange(num_frames) - num_frames // 2) * max(1, int(stride))
    return np.clip(center + offsets, 0, total - 1).astype(np.int64)


def load_frames_from_video(video_path, num_frames, mode, target_frame, stride):
    """利用 decord 将视频读取为 NumPy 数组帧批次 [T, H, W, 3]"""
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    if total <= 0:
        raise RuntimeError(f"Empty video: {video_path}")
    indices = sample_frame_indices(total, num_frames, mode, target_frame, stride)
    return reader.get_batch(indices).asnumpy(), indices, total


def load_frames_from_npz(npz_path):
    """从 npz 缓存里拿处理存好的图片数组 [T, H, W, 3]"""
    data = np.load(npz_path, allow_pickle=False)
    if "imgs" not in data:
        raise KeyError(f"{npz_path} has no imgs field")
    frames = data["imgs"]
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"Expected imgs [T,H,W,C], got {frames.shape}")
    if frames.shape[-1] == 4:
        # 去掉偶尔混进来的 alpha 通道
        frames = frames[..., :3]
    indices = np.arange(frames.shape[0], dtype=np.int64)
    return frames, indices, frames.shape[0], data


def maybe_subsample_frames(frames_np, sampled_indices, max_frames):
    """根据需求，对提取完的帧二次均匀采样（方便小规模显存推理高帧率视频）"""
    if max_frames is None or int(max_frames) <= 0 or frames_np.shape[0] <= int(max_frames):
        return frames_np, sampled_indices
    keep = np.linspace(0, frames_np.shape[0] - 1, int(max_frames))
    keep = np.clip(np.round(keep), 0, frames_np.shape[0] - 1).astype(np.int64)
    return frames_np[keep], sampled_indices[keep]


def clamp_square_box(x0, y0, x1, y1, width, height):
    """以选框重心为基准，将其强行调整为等边长正方形，并不越过图片边界。"""
    side = max(int(x1 - x0), int(y1 - y0), 1)
    cx = 0.5 * (float(x0) + float(x1))
    cy = 0.5 * (float(y0) + float(y1))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        shift = x1 - width
        x0 = max(0, x0 - shift)
        x1 = width
    if y1 > height:
        shift = y1 - height
        y0 = max(0, y0 - shift)
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def apply_spatial_crop(frames_np, args):
    """可选特性：利用运动热力图裁剪出真正带有"主运动体"区域，让分辨率聚焦该对象"""
    height, width = frames_np.shape[1], frames_np.shape[2]
    if args.crop_mode == "none":
        return frames_np, {
            "mode": "none",
            "box_xyxy": [0, 0, int(width), int(height)],
            "original_hw": [int(height), int(width)],
        }

    if args.crop_mode == "center":
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2
        x1, y1 = x0 + side, y0 + side
    else:
        frames = frames_np.astype(np.float32)
        if frames.shape[0] < 2:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            gray = frames[..., :3].mean(axis=-1)
            # 作差分得到时间切片的运动绝对值热力谱
            heat = np.abs(np.diff(gray, axis=0)).mean(axis=0)
            heat_t = torch.from_numpy(heat).float()[None, None]
            # 采用平滑核放大运动特征区域边界
            heat_t = F.avg_pool2d(F.pad(heat_t, (5, 5, 5, 5), mode="replicate"), kernel_size=11, stride=1)
            heat = heat_t[0, 0].numpy()
            
            # 滤掉轻微抖动噪音的运动区块
            threshold = np.quantile(heat, float(args.crop_quantile))
            motion = np.maximum(heat - threshold, 0.0)
            if float(motion.sum()) <= 1e-6:
                x0, y0, x1, y1 = 0, 0, width, height
            elif args.crop_mode == "motion_window":
                yy, xx = np.mgrid[0:height, 0:width]
                total = float(motion.sum())
                # 寻找运动势能质心 cx, cy
                cx = float((motion * xx).sum() / total)
                cy = float((motion * yy).sum() / total)
                side = int(round(min(height, width) * float(args.crop_window_fraction)))
                side = int(np.clip(side, min(height, width) * 0.45, min(height, width)))
                x0 = int(round(cx - side / 2.0))
                y0 = int(round(cy - side / 2.0))
                x1, y1 = x0 + side, y0 + side
                x0, y0, x1, y1 = clamp_square_box(x0, y0, x1, y1, width, height)
            else:
                ys, xs = np.where(motion > 0.0)
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                pad = int(round(max(x1 - x0, y1 - y0) * float(args.crop_padding)))
                x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
                x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
                x0, y0, x1, y1 = clamp_square_box(x0, y0, x1, y1, width, height)

    cropped = frames_np[:, y0:y1, x0:x1, :]
    return cropped, {
        "mode": args.crop_mode,
        "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "original_hw": [int(height), int(width)],
        "cropped_hw": [int(cropped.shape[1]), int(cropped.shape[2])],
        "padding": float(args.crop_padding),
        "quantile": float(args.crop_quantile),
        "window_fraction": float(args.crop_window_fraction),
    }


def preprocess_frames(frames_np, img_size):
    """
    ImageNet 标准视角的图像处理管线。
    1. 由 NHWC 转为 NCHW  (T维度被放前面)
    2. Resize 让短边等于 (256/224 * img_size) 以保持常规的中心感受野
    3. CenterCrop 抠出正好是 img_size × img_size
    4. 执行 (x - mean) / std，将其推送到 [-2, 2] 左右浮动的浮点区间
    """
    frames = torch.from_numpy(frames_np).float()
    if frames.max() > 1.5:
        # Normalize 到 0~1 的浮点色阶
        frames = frames / 255.0
    # [T, H, W, 3] -> [T, 3, H, W]
    frames = frames.permute(0, 3, 1, 2).contiguous()
    _, _, height, width = frames.shape
    
    # 动态适应指定的短边分辨率倍率
    short_side = int(256.0 / 224.0 * img_size)
    if height <= width:
        new_h = short_side
        new_w = int(round(width * short_side / height))
    else:
        new_w = short_side
        new_h = int(round(height * short_side / width))

    # 执行插值改变空间形状
    resized = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    
    # 固定取中心区块
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    if top < 0 or left < 0:
        raise ValueError(f"Cannot center-crop {img_size} from resized {(new_h, new_w)}")
    cropped = resized[:, :, top : top + img_size, left : left + img_size]

    # 保存一份用作输出和展示对照的原图 (未 normalization 的干净区块)
    cropped_uint8 = (
        cropped.permute(0, 2, 3, 1).mul(255.0).clamp(0, 255).byte().cpu().numpy()
    )
    
    # 正态分布 Normalization，为输入预训练大模型准备
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32)[None, :, None, None]
    normalized = (cropped - mean) / std
    return normalized.contiguous(), cropped_uint8


@torch.no_grad()
def encode_feature_layers(model, frames_norm, device, layer_names):
    """
    把做完规范化的所有图像一起抛给模型前推 (inference)。
    注意 VJEPA 等视频预训练模型需要 batch 处理的视频输入通常为 [B, C, T, H, W] 的 5 维数据体。
    """
    # frames_norm 形如 [T, 3, H, W] -> 转换成 [1_batch, 3_channels, T, H, W]
    video = frames_norm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    video = video.to(device, non_blocking=True)
    with torch.inference_mode():
        # 执行前向网络特征抽取
        features = model(video)
    if isinstance(features, torch.Tensor):
        features = [features]
        
    # 返回指定层的 tensor 信息列表字典 (把没用的 batch=1 这个壳扒掉并迁回CPU)
    return {
        name: feat[0].detach().float().cpu()
        for name, feat in zip(layer_names, features)
    }


def infer_token_grid(model, num_frames, img_size, feat_tokens):
    """
    【核心计算】根据模型预设推理特征维度将 [N, Dimension] 中的 
    N(Token 序列长度) 逆向推算回空间网格和时间管长 (Tubelet)。
    
    推断一维序列 [N, D] 应该如何折叠回 [T_token, H_patch, W_patch, D] 的四维时空网格。
    这是JEPA/ViT dense特征可视化最底层的数学校验与变换规则保障。
    """
    patch_size = int(getattr(model, "patch_size", 16))
    tubelet_size = int(getattr(model, "tubelet_size", 1))
    
    # 1. 计算时间维度 token 数量 T_token (由几帧合成一个序列元素？)
    t_grid = int(num_frames) // tubelet_size
    
    # 2. 计算每个空间轴能被 patch size 整除出多少个块
    h_grid = int(img_size) // patch_size
    w_grid = int(img_size) // patch_size
    
    # 理论期待的模型出来的特征词数量
    expected = t_grid * h_grid * w_grid
    
    # 核对不符合的情况：例如包含 [CLS] Token 忘记清理，或者填充问题。
    if expected != int(feat_tokens):
        raise RuntimeError(
            "Cannot reshape encoder output into [T_token,H_patch,W_patch,D]. "
            f"feat_tokens={feat_tokens}, expected={expected}, "
            f"T_token={t_grid}, H_patch={h_grid}, W_patch={w_grid}, "
            f"patch_size={patch_size}, tubelet_size={tubelet_size}"
        )
    return t_grid, h_grid, w_grid, patch_size, tubelet_size


def maybe_use_cached_feats(npz_data, img_size, patch_size):
    """当用户传递了 npz 缓存且开启选项时，不重新跑一遍推演而是直接读取 npz 里的特征层序列。"""
    if "vjepa_feats" not in npz_data:
        raise KeyError("npz has no vjepa_feats field")
    feats = torch.from_numpy(npz_data["vjepa_feats"]).float()
    if feats.ndim != 3:
        raise ValueError(f"Expected cached vjepa_feats [T,P,D], got {tuple(feats.shape)}")
    t_count, patch_count, dim = feats.shape
    h_grid = int(img_size) // int(patch_size)
    w_grid = int(img_size) // int(patch_size)
    if patch_count != h_grid * w_grid:
        raise ValueError(
            "Cached vjepa_feats is not a recoverable spatial grid for this image size. "
            f"shape={tuple(feats.shape)}, expected P={h_grid * w_grid}. "
            "Re-run without --use_cached_feats to encode npz['imgs']."
        )
    # 同样的由于我们已经校验了尺寸，将一维铺开的空间序列还原为网格特征：[T_count, H_grid, W_grid, Dimension]
    return feats.reshape(t_count, h_grid, w_grid, dim)


def normalize_features_for_pca(tokens, mode):
    """
    PCA前必须的归一化过程选择器。
    通常特征提取出来范数差距大，最鲁棒的是使用 "l2_center" 规避极端范数的 token 对 PCA 方差产生致命的主导影响。
    """
    tokens = tokens.float()
    if mode == "center":
        # 简单均值中心化
        return tokens - tokens.mean(dim=0, keepdim=True)
    if mode == "l2_center":
        # 先沿特征维度D将其压缩成单位向量，消除整体长度只计算投影方向的一致点，再做中心化
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        return tokens - tokens.mean(dim=0, keepdim=True)
    if mode == "standardize":
        # 标准正态化 (除以标准差)，让每个维度的贡献变均等
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
        return tokens / tokens.std(dim=0, keepdim=True).clamp_min(1e-6)
    raise ValueError(mode)


def fit_pca(tokens, components=3):
    """
    使用 SVD 奇异值分解来进行 PCA 基准空间的提取。
    要求输入序列 [N, D]，我们将提取出方差最大的头几套（对应最终颜色，组件固定是3(RGB)）特征映射向量。
    """
    if tokens.size(0) < components:
        raise ValueError(f"Need at least {components} tokens for PCA, got {tokens.size(0)}")
    
    # 再次减去均值保障中心在原点
    center = tokens.mean(dim=0, keepdim=True)
    centered = tokens - center
    
    # 对 [N, D] 的残差空间施行无缩截 SVD (奇异值分解)。Vh(V的转置矩阵) 是包含了各主成分的正交基向量。
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    
    # 抽取特征最丰富的前三个基底（转置为 [D, 3]）
    basis = vh[:components].T
    
    # 计算三个成分对总量方差的解释度占比
    total = float(torch.sum(singular_values.square()).clamp_min(1e-12).item())
    explained = [
        float(v.item() / total)
        for v in singular_values[:components].square()
    ]
    # 返回：数据的均值，用于PCA变换的空间基，前三个成分各自的信息表征能力解释率
    return center, basis, explained


def apply_pca(tokens, center, basis):
    """将特征通过所学的特征基进行运算并映射到低维特征系 (通常是3D空间 RGB)。"""
    # [N, D] 乘以 [D, 3] = [N, 3]
    projection = (tokens - center) @ basis
    
    # ! 统一翻转处理，用来稳固同一视频流各层跑完后 PCA 的整体颜色方向，不然稍微变动一点颜色可能会全局逆翻转！
    for channel in range(projection.size(1)):
        # 寻找拥有绝对值最大影响的 token 位置
        idx = torch.argmax(torch.abs(projection[:, channel]))
        # 假如它在该影响下处于负维象标位置，我们把该轴所有投影符号反转（由于PCA特征基是对称不分左右的，这个操作属于等价调整）
        if projection[idx, channel] < 0:
            projection[:, channel] *= -1
    return projection


def rank01(values):
    """
    用于把一些未归一化的大范围分数基于其在整体分发里的排位分位情况强行线性平铺到 [0.0, 1.0]。
    剔除最高最低极端点的影响。
    """
    values = values.float().reshape(-1)
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    return ((values - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def robust_rgb_from_projection(projection, normalize_mask, percentile):
    """
    【后处理色彩】：将提取到的 [N, 3] 的三向主成份降维实数转为稳健的 RGB 数值。
    不使用简单的 min/max，是为了防备画面中某一个过亮的噪音 Token 拉垮剩下的全部亮度层级。
    """
    low_q = float(percentile) / 100.0
    high_q = 1.0 - low_q
    channels = []
    # 有时候只需要在“前对象区域”找它的统计分布 (避免被背景干扰颜色值域映射)
    values_for_norm = projection[normalize_mask] if normalize_mask.any() else projection
    
    for channel in range(3):
        vals = values_for_norm[:, channel]
        # 按参数截断比如 (0.01 和 0.99) 的分位值作为 min 和 max 的定义。
        lo = torch.quantile(vals, low_q)
        hi = torch.quantile(vals, high_q)
        # 用截出的区间重新将该通道归于到 [0.0, 1.0] 空间。
        mapped = (projection[:, channel] - lo) / (hi - lo).clamp_min(1e-6)
        channels.append(mapped.clamp(0.0, 1.0))
    return torch.stack(channels, dim=-1)


def smooth_score_grid(score_grid):
    """将预测的分数网格平滑一下，让边缘和中心连接更温和，减少跳动孤岛"""
    values = score_grid.float().unsqueeze(1)
    return F.avg_pool2d(values, kernel_size=3, stride=1, padding=1).squeeze(1)


def denoise_mask(mask_grid):
    """【形态学操作】基于形态学的最大后均值池化，主要作用是在空间层面将稀碎的背景噪点mask消掉并填补对象的微小孔洞"""
    values = mask_grid.float().unsqueeze(1)
    values = F.max_pool2d(values, kernel_size=3, stride=1, padding=1)
    values = F.avg_pool2d(values, kernel_size=3, stride=1, padding=1)
    # 若平均覆盖比例超过其面积一部分，就维持 1 状态
    return values.squeeze(1) >= 0.35


def foreground_mask_from_scores(score_grid, quantile, smooth):
    """从热力分数得分获取布尔二进制前景掩码 (Mask). 超出参数 quantile 给出的范围就会被作为前景."""
    work = smooth_score_grid(score_grid) if smooth else score_grid.float()
    threshold = torch.quantile(work.reshape(-1), float(quantile))
    mask = work >= threshold
    if smooth:
        mask = denoise_mask(mask)
    return mask


def simple_pca_rgb(
    feat_grid,
    mode,
    percentile,
    rgb_smooth=0.0,
    rgb_saturation=1.0,
    rgb_gamma=1.0,
):
    """
    【简易备用 PCA】
    不做任何去除背景的优化，所有 token 一起丢里做PCA处理。如果 mask 识别坏了，会回退到这个方法兜底。
    由于包含了大片特征相对趋同时无用的背景计算权重，其对象高亮往往不如前景提取模式好看。
    """
    t_count, h_grid, w_grid, dim = feat_grid.shape
    tokens = normalize_features_for_pca(feat_grid.reshape(-1, dim), mode=mode)
    
    # 不分青红皂白训练基元
    center, basis, explained = fit_pca(tokens, components=3)
    projection = apply_pca(tokens, center, basis)
    
    rgb = robust_rgb_from_projection(
        projection,
        torch.ones(projection.size(0), dtype=torch.bool),
        percentile,
    )
    rgb = postprocess_rgb(rgb, t_count, h_grid, w_grid, rgb_smooth, rgb_saturation, rgb_gamma)
    rgb_grid = (rgb * 255.0).byte().cpu().numpy()
    mask = torch.ones(t_count, h_grid, w_grid, dtype=torch.bool)  # 全部标记为有效前景
    return rgb_grid, mask.numpy(), explained, False


def postprocess_rgb(rgb, t_count, h_grid, w_grid, rgb_smooth, rgb_saturation, rgb_gamma):
    """图像色彩的后置美化处理（包含颜色平滑、色彩强度拉升、灰阶还原亮度等算法）."""
    rgb = rgb.reshape(t_count, h_grid, w_grid, 3).float().clamp(0.0, 1.0)
    smooth_alpha = float(np.clip(rgb_smooth, 0.0, 1.0))
    if smooth_alpha > 0.0:
        values = rgb.permute(0, 3, 1, 2)
        # 用 3x3 池化层混一点低频数据作为反走样
        pooled = F.avg_pool2d(F.pad(values, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        rgb = ((1.0 - smooth_alpha) * values + smooth_alpha * pooled).permute(0, 2, 3, 1)

    if float(rgb_saturation) != 1.0:
        gray = rgb.mean(dim=-1, keepdim=True)
        # 饱和提升等式
        rgb = gray + (rgb - gray) * float(rgb_saturation)
    if float(rgb_gamma) > 0.0 and float(rgb_gamma) != 1.0:
        # gamma 校准：低于 1 让暗部变明亮
        rgb = rgb.clamp(0.0, 1.0).pow(float(rgb_gamma))
    return rgb.clamp(0.0, 1.0)


def foreground_pca_rgb(
    feat_grid,
    mode,
    percentile,
    method,
    quantile,
    smooth,
    background_gray,
    background_mode,
    background_desaturate,
    rgb_smooth,
    rgb_saturation,
    rgb_gamma,
):
    """
    【核心实现核心】：这函数实现的是大名鼎鼎的双阶段 PCA 流程，解决背景喧宾夺主问题。
    第一阶段：在所有 tokens 上做宏观 PCA 并结合特征向量范数寻找画面中与众不同 (更发亮突出) 且包含主要特征的区域（视为对象 Foreground）
    第二阶段：剥离出单纯 foreground 的 tokens，只基于它们训练 PCA，以便提取对象身上更加细腻独特的色彩纹理分布！
    第三阶段：把这个特征网再强行覆盖作用全部原图中完成渲染。
    """
    # 拆包网络形状信息
    t_count, h_grid, w_grid, dim = feat_grid.shape
    
    # 得到 1 维 token 序列总队 [N, D]
    flat = feat_grid.reshape(-1, dim).float()
    tokens = normalize_features_for_pca(flat, mode=mode)

    # ---------- 一阶段：全局 PCA 以判定前后台评分 ----------
    center1, basis1, explained1 = fit_pca(tokens, components=3)
    projection1 = apply_pca(tokens, center1, basis1)
    
    # 提取在主要变化成分特征（代表它是特殊的画面主体）以及单纯范数列里比较凸显的位置。
    pc1_score = rank01(torch.abs(projection1[:, 0]))
    norm_score = rank01(torch.linalg.norm(flat - flat.mean(dim=0, keepdim=True), dim=1))
    
    # 选择打分方式（默认使用两边混合结果作为前景评分权值）
    if method == "pc1":
        score = pc1_score
    elif method == "norm":
        score = norm_score
    else:
        score = 0.65 * pc1_score + 0.35 * norm_score

    # 把打好分的得分板重排为时空结构以防后续二维区域滤波使用。
    score_grid = score.reshape(t_count, h_grid, w_grid)
    mask_grid = foreground_mask_from_scores(score_grid, quantile, smooth)
    
    # 把 mask 降维
    mask = mask_grid.reshape(-1)
    
    # 兜底防御保护判定：假如果抠出来的前景大到填满屏幕（过平庸）或小到几个像素。直接中止复杂化处理交由单阶段接管。
    coverage = float(mask.float().mean().item())
    if coverage < 0.15 or coverage > 0.85 or int(mask.sum().item()) < 4:
        rgb_grid, simple_mask, explained, _ = simple_pca_rgb(
            feat_grid,
            mode,
            percentile,
            rgb_smooth,
            rgb_saturation,
            rgb_gamma,
        )
        return rgb_grid, simple_mask, explained, True

    # ---------- 二阶段：剔除噪音专注入微的 PCA ----------
    # 仅把布尔数组代表为True的重要物件拿出来练一次PCA，获取到对于主体极度敏感能切分色彩细微差距的 basis2.
    foreground_tokens = tokens[mask]
    center2, basis2, explained2 = fit_pca(foreground_tokens, components=3)
    
    # 重要：虽然用小圈子练的权重（以抗外部乱七八糟背景牵引带崩），但是最后运用渲染仍然给整个世界染！
    projection2 = apply_pca(tokens, center2, basis2)
    rgb = robust_rgb_from_projection(projection2, mask, percentile)
    
    # 根据指定的不同背景渲染样式处理不要凸显的背景部分。
    gray = float(np.clip(background_gray, 0, 255)) / 255.0
    if background_mode == "gray":
        rgb[~mask] = gray # 一抹纯灰处理
    elif background_mode == "desaturate":
        mix = float(np.clip(background_desaturate, 0.0, 1.0))
        # （降饱和偏灰算法混合模式，降低存在感保留些许细节）
        rgb[~mask] = (1.0 - mix) * rgb[~mask] + mix * gray
    elif background_mode != "pca":
        raise ValueError(background_mode)

    # 最后美化和色彩修正步骤
    rgb = postprocess_rgb(
        rgb,
        t_count,
        h_grid,
        w_grid,
        rgb_smooth,
        rgb_saturation,
        rgb_gamma,
    )
    rgb_grid = (rgb * 255.0).byte().cpu().numpy()
    return rgb_grid, mask_grid.cpu().numpy(), explained2 or explained1, False


def spatial_smoothness(rgb_grid):
    """【评价指标】检查相邻 patch 块之间的颜色是否温和渐变，如果是噪音则越跳变。用于自动化寻找更好的网络层效果打分."""
    values = torch.from_numpy(rgb_grid).float() / 255.0
    dx = (values[:, :, 1:, :] - values[:, :, :-1, :]).abs().mean()
    dy = (values[:, 1:, :, :] - values[:, :-1, :, :]).abs().mean()
    diff = float(((dx + dy) * 0.5).item())
    return float(1.0 / (1.0 + 6.0 * diff))


def temporal_consistency(rgb_grid):
    """【评价指标】检查各个帧间时序颜色变化是否太跳脱抽风。"""
    values = torch.from_numpy(rgb_grid).float() / 255.0
    if values.size(0) < 2:
        return 1.0
    diff = float((values[1:] - values[:-1]).abs().mean().item())
    return float(1.0 / (1.0 + 6.0 * diff))


def foreground_colorfulness(rgb_grid, mask_grid):
    """【评价指标】测量对象身上是不是展现了复杂的纹理结构（RGB多元化且非单一色块）. 标准差表征其色彩缤纷程度."""
    values = torch.from_numpy(rgb_grid).float() / 255.0
    mask = torch.from_numpy(mask_grid.astype(np.bool_))
    if not mask.any():
        return 0.0
    fg = values[mask]
    return float(fg.std(dim=0).mean().clamp(0.0, 1.0).item())


def score_layer_metrics(metrics, layer_index, max_layer):
    """【主评分函数】结合空间噪音，时间抖动，提取对象比重以及颜色丰富度作为联合打分用于最终在参数多选项间定音选出的最佳层。"""
    coverage = float(metrics["foreground_coverage"])
    # Coverage 占比 45% 的表现会被认为是理想比例 (过满像空图，过少太微小)
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.45) / 0.45)
    explained = min(1.0, float(sum(metrics.get("pca_explained_variance", [])[:3])))
    # 如果效果一致倾向于使用偏后期的特征深层网络。
    layer_bias = 0.03 * (float(layer_index) / max(1.0, float(max_layer)))
    return float(
        0.30 * metrics["spatial_smoothness"]
        + 0.25 * metrics["temporal_consistency"]
        + 0.20 * coverage_score
        + 0.15 * metrics["foreground_colorfulness"]
        + 0.10 * explained
        + layer_bias
    )


def layer_index_from_name(name):
    """从网络指定出来的字典里提取出层级索引数做评分偏重"""
    if name.startswith("layer"):
        return int(name.replace("layer", ""))
    if name.startswith("fusion"):
        parts = [int(part) for part in name.replace("fusion", "").replace("_", ",").split(",") if part]
        return max(parts) if parts else 0
    return 0


def render_feature_grid(feat_grid, args, foreground_quantile=None):
    """封装特征层的分发，调用指定的方法绘制最终网格，并顺带进行相关度量收集工作。"""
    quantile = args.foreground_quantile if foreground_quantile is None else float(foreground_quantile)
    if args.pca_recipe == "simple":
        rgb_grid, mask_grid, explained, fallback = simple_pca_rgb(
            feat_grid,
            args.pca_norm,
            args.percentile,
            args.rgb_smooth,
            args.rgb_saturation,
            args.rgb_gamma,
        )
    else:
        rgb_grid, mask_grid, explained, fallback = foreground_pca_rgb(
            feat_grid,
            args.pca_norm,
            args.percentile,
            args.foreground_method,
            quantile,
            args.smooth_foreground_mask,
            args.background_gray,
            args.background_mode,
            args.background_desaturate,
            args.rgb_smooth,
            args.rgb_saturation,
            args.rgb_gamma,
        )
    coverage = float(np.asarray(mask_grid, dtype=np.float32).mean())
    metrics = {
        "feature_grid_shape": [int(v) for v in feat_grid.shape],
        "pca_rgb_grid_shape": [int(v) for v in rgb_grid.shape],
        "foreground_mask_shape": [int(v) for v in mask_grid.shape],
        "foreground_coverage": coverage,
        "pca_explained_variance": [float(v) for v in explained],
        "spatial_smoothness": spatial_smoothness(rgb_grid),
        "temporal_consistency": temporal_consistency(rgb_grid),
        "foreground_colorfulness": foreground_colorfulness(rgb_grid, mask_grid),
        "fallback_to_simple_pca": bool(fallback),
        "foreground_quantile_effective": float(quantile),
        "background_mode": args.background_mode,
        "rgb_smooth": float(args.rgb_smooth),
        "rgb_saturation": float(args.rgb_saturation),
        "rgb_gamma": float(args.rgb_gamma),
    }
    return rgb_grid, mask_grid, metrics


def select_token_time(args, sampled_indices, total_frames, t_grid, tubelet_size):
    """判定在做主图展示时要选用特征抽取时序的哪一时段 (比如如果有要求目标帧则找最接近帧代表层的特征图)"""
    if args.token_t is not None:
        return int(np.clip(args.token_t, 0, t_grid - 1))
    if args.target_frame is not None:
        raw_target = int(np.clip(args.target_frame, 0, total_frames - 1))
        sample_pos = int(np.argmin(np.abs(sampled_indices - raw_target)))
        return int(np.clip(sample_pos // max(1, tubelet_size), 0, t_grid - 1))
    return int(t_grid // 2)


def background_frame_index(token_t, tubelet_size, num_frames):
    """
    【重要逆推索引】: 当特征网络管长为 T 时，某个给定的空间 Token 该映射为原本视像上的第几帧背景。
    以此保证左图视频动作与右图特征渲染所发生的时间点恰好是对齐匹配在重合部位。
    """
    pos = token_t * max(1, tubelet_size) + max(0, tubelet_size // 2)
    return int(np.clip(pos, 0, num_frames - 1))


def pil_resample(name):
    """转换字串名为库对应重插参数项常量"""
    return Image.Resampling.NEAREST if name == "nearest" else Image.Resampling.BICUBIC


def resize_patch_map(rgb_small, size, interp):
    """改变像素图空间尺寸"""
    return Image.fromarray(rgb_small).resize((size, size), pil_resample(interp))


def save_overlay(frame_uint8, pca_image, out_path, alpha=0.65):
    """绘制源视频图像和特征渲染 PCA 色的透明度 Alpha 叠加图"""
    base = np.asarray(Image.fromarray(frame_uint8).convert("RGB")).astype(np.float32)
    top = np.asarray(pca_image.convert("RGB")).astype(np.float32)
    out = (1.0 - alpha) * base + alpha * top
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)


def load_label_font(size):
    """寻找内置或系统适用的支持绘字字体."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_vertical_label(sheet, text, x, y, width, height, font):
    """绘制联系表左侧的旋转 90度垂直文字列"""
    label = Image.new("RGB", (height, width), "white")
    draw = ImageDraw.Draw(label)
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (height - (bbox[2] - bbox[0])) // 2
    ty = (width - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    sheet.paste(label.rotate(90, expand=True), (x, y))


def make_sheet(row_specs, token_indices, tubelet_size, out_path, cell_size, interp):
    """
    绘制大画幅横行展现全貌表格 (contact sheet)。即左边是说明字，上面标 t 下面一行行横向展示。
    """
    label_w = 110
    gutter = 8
    header_h = 24
    rows = len(row_specs)
    cols = len(token_indices)
    width = label_w + cols * cell_size + (cols + 1) * gutter
    height = header_h + rows * cell_size + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = load_label_font(size=24)
    header_font = load_label_font(size=13)

    for row_idx, (label, grid_or_frames, is_frames) in enumerate(row_specs):
        y0 = header_h + gutter + row_idx * (cell_size + gutter)
        draw_vertical_label(sheet, label, 0, y0, label_w, cell_size, label_font)
        for col, token_t in enumerate(token_indices):
            x0 = label_w + gutter + col * (cell_size + gutter)
            if is_frames:
                # 若是给原图，算出原本的对标时机。
                bg_idx = background_frame_index(token_t, tubelet_size, grid_or_frames.shape[0])
                img = Image.fromarray(grid_or_frames[bg_idx]).resize(
                    (cell_size, cell_size), Image.Resampling.BICUBIC
                )
            else:
                img = resize_patch_map(grid_or_frames[token_t], cell_size, interp)
            sheet.paste(img, (x0, y0))
            if row_idx == 0:
                draw.text((x0 + 4, 4), f"t={token_t}", fill=(0, 0, 0), font=header_font)

    sheet.save(out_path)


def mask_to_rgb(mask_grid):
    """把用于做筛选用的2值黑白区域掩码转化成用于直接呈现可视化的 3 通道图片黑白区域"""
    mask = np.asarray(mask_grid, dtype=np.uint8) * 255
    return np.repeat(mask[..., None], 3, axis=-1)


def write_manifest(path, payload):
    """落地存储一些生成日志清单，以便其它测试和报表代码解析使用分析效果"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sample_motion_score(frames_np):
    """粗略评价全局动作复杂程度用于给偏静态平稳特征加偏科分数。"""
    frames = frames_np.astype(np.float32)
    if frames.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(frames, axis=0)).mean())


def choose_tokens(args, rgb_shape, sampled_indices, total_frames, tubelet_size):
    """推算我们需要拿来放展示矩阵的主推 token """
    token_t = select_token_time(args, sampled_indices, total_frames, rgb_shape[0], tubelet_size)
    n_show = min(max(1, int(args.sheet_cols)), int(rgb_shape[0]))
    token_indices = [int(round(x)) for x in np.linspace(0, rgb_shape[0] - 1, n_show)]
    if token_t not in token_indices:
        token_indices[-1] = token_t
        token_indices = sorted(set(token_indices))[:n_show]
    return token_t, token_indices


def load_input_for_path(path, args):
    """根据类型加载数据返回其基本形状元数据."""
    if path.suffix.lower() == ".npz":
        frames_np, sampled_indices, total_frames, npz_data = load_frames_from_npz(str(path))
        frames_np, sampled_indices = maybe_subsample_frames(frames_np, sampled_indices, args.max_frames)
        args.num_frames = int(frames_np.shape[0])
        return frames_np, sampled_indices, total_frames, npz_data, path.stem
    frames_np, sampled_indices, total_frames = load_frames_from_video(
        str(path),
        args.num_frames,
        args.sample_mode,
        args.target_frame,
        args.stride,
    )
    frames_np, sampled_indices = maybe_subsample_frames(frames_np, sampled_indices, args.max_frames)
    args.num_frames = int(frames_np.shape[0])
    return frames_np, sampled_indices, total_frames, None, path.stem


def build_feature_grids(model, npz_data, frames_norm, args, layer_names):
    """
    【总装配流水线一局】：根据送进网络里的裁剪规范格式流提取所需的特征列表，并且通过映射还原算法把一维网重新化作空间。
    并且如果存在要融合（融合两特征段落），便把结果级联相交起来！
    """
    if args.use_cached_feats:
        if npz_data is None:
            raise ValueError("--use_cached_feats is valid only with --npz")
        feat_grid = maybe_use_cached_feats(npz_data, args.img_size, args.patch_size)
        return {"cached": feat_grid}, 1, int(args.patch_size)

    # 1. 过网络得到特征数组字典。如 {"layer5": [序列长, 特征空间], ...}
    layer_features = encode_feature_layers(model, frames_norm, args.device, layer_names)
    
    feature_grids = {}
    tubelet_size = None
    patch_size = None
    
    for name, feat in layer_features.items():
        # 2. 判断映射网格形状
        t_grid, h_grid, w_grid, patch_size, tubelet_size = infer_token_grid(
            model,
            args.num_frames,
            args.img_size,
            feat.shape[0],
        )
        # 3. 将对应信息归档，存为真正的基于时空感知的四维特征域，不再是一维链条 [T, H, W, D]。
        feature_grids[name] = feat.reshape(t_grid, h_grid, w_grid, int(feat.shape[-1]))

    # 把某些指定的层特征用类似早期混合残差结构那样接轨连到一块成为混合 PCA 判断标准
    fusion_layers = parse_int_list(args.fusion_layers, [])
    if fusion_layers:
        fusion_names = [f"layer{idx}" for idx in fusion_layers]
        missing = [name for name in fusion_names if name not in feature_grids]
        if missing:
            raise ValueError(
                "--fusion_layers must be included in --out_layers; "
                f"missing={missing}, out_layers={sorted(feature_grids)}"
            )
        normalized = []
        for name in fusion_names:
            grid = feature_grids[name].float()
            flat = grid.reshape(-1, grid.shape[-1])
            flat = F.normalize(flat, dim=-1, eps=1e-6)
            normalized.append(flat.reshape(*grid.shape[:-1], grid.shape[-1]))
        feature_grids["fusion" + "_".join(str(v) for v in fusion_layers)] = torch.cat(
            normalized,
            dim=-1,
        )
    return feature_grids, int(tubelet_size), int(patch_size)


def process_one_source(path, args, model, layer_names, out_dir):
    """
    【总装配流水线二局】：从拿到特征结果后的第二阶段，利用评分、不同层之间的试探进行结果选取，直到执行绘图落盘到文件。
    """
    frames_np, sampled_indices, total_frames, npz_data, source_name = load_input_for_path(path, args)
    frames_np, crop_meta = apply_spatial_crop(frames_np, args)
    source_out = out_dir
    source_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] source={source_name}", flush=True)
    print(f"[INFO] input frames shape={tuple(frames_np.shape)}", flush=True)
    print(f"[INFO] crop={crop_meta}", flush=True)
    frames_norm, frames_crop_uint8 = preprocess_frames(frames_np, img_size=args.img_size)

    # 第一大部分：从图片流生成提取空间感官四维特征群。
    feature_grids, tubelet_size, patch_size = build_feature_grids(
        model,
        npz_data,
        frames_norm,
        args,
        layer_names,
    )

    layer_outputs = {}
    max_layer = max([layer_index_from_name(name) for name in feature_grids] or [0])
    
    # 不同的分数阈值试探
    quantile_candidates = (
        parse_float_list(args.auto_quantiles, [args.foreground_quantile])
        if args.pca_recipe == "foreground"
        else [args.foreground_quantile]
    )
    if not quantile_candidates:
        quantile_candidates = [args.foreground_quantile]
        
    for name, feat_grid in feature_grids.items():
        layer_idx = layer_index_from_name(name)
        variants = []
        for quantile in quantile_candidates:
            # 依次去测试不同切割分位点（用来确定对象与背景）绘制出来的渲染图结果如何
            rgb_grid, mask_grid, metrics = render_feature_grid(
                feat_grid,
                args,
                foreground_quantile=quantile,
            )
            # 通过评选策略来评价此次的降维着色观感质量优劣
            metrics["layer_score"] = score_layer_metrics(metrics, layer_idx, max_layer)
            metrics["layer_index"] = layer_idx
            variants.append(
                {
                    "rgb_grid": rgb_grid,
                    "mask_grid": mask_grid,
                    "metrics": metrics,
                }
            )
        # 用跑出最高的分来定调当前测试属于该模型网络层下的最优生成
        best_variant = max(variants, key=lambda item: item["metrics"]["layer_score"])
        metrics = best_variant["metrics"]
        layer_outputs[name] = {
            "rgb_grid": best_variant["rgb_grid"],
            "mask_grid": best_variant["mask_grid"],
            "metrics": metrics,
            "variant_metrics": [item["metrics"] for item in variants],
        }
        print(
            f"[INFO] {source_name} {name} score={metrics['layer_score']:.4f} "
            f"q={metrics['foreground_quantile_effective']:.2f} "
            f"coverage={metrics['foreground_coverage']:.3f} "
            f"smooth={metrics['spatial_smoothness']:.3f} "
            f"temporal={metrics['temporal_consistency']:.3f}",
            flush=True,
        )

    # 接着，在所有的评估抽取层(如深层、中层等)里面，再选出一个具有最明显的主体识别与时间一致连贯的最佳网络产出。
    best_layer = max(layer_outputs, key=lambda key: layer_outputs[key]["metrics"]["layer_score"])
    best_rgb = layer_outputs[best_layer]["rgb_grid"]
    best_mask = layer_outputs[best_layer]["mask_grid"]
    
    # 获取我们最终该拿来绘制表格图的时间流
    token_t, token_indices = choose_tokens(args, best_rgb.shape, sampled_indices, total_frames, tubelet_size)
    bg_idx = background_frame_index(token_t, tubelet_size, frames_crop_uint8.shape[0])
    raw_idx = int(sampled_indices[bg_idx]) if len(sampled_indices) > bg_idx else int(bg_idx)

    # 设定所有的保存路径与名称
    frame_out = source_out / f"{source_name}_token{token_t:03d}_frame{raw_idx:06d}_frame.png"
    pca_out = source_out / f"{source_name}_token{token_t:03d}_{best_layer}_pca.png"
    overlay_out = source_out / f"{source_name}_token{token_t:03d}_{best_layer}_overlay.png"
    sheet_out = source_out / f"{source_name}_best_contact_sheet.png"
    bicubic_sheet_out = source_out / f"{source_name}_best_contact_sheet_bicubic.png"
    comparison_out = source_out / f"{source_name}_layer_comparison.png"
    mask_out = source_out / f"{source_name}_mask_preview.png"
    manifest_out = source_out / f"{source_name}_manifest.json"

    # 把各个状态的信息分别制卡为图写往磁盘...
    Image.fromarray(frames_crop_uint8[bg_idx]).save(frame_out)
    best_pca_image = resize_patch_map(best_rgb[token_t], args.img_size, args.render_interp)
    best_pca_image.save(pca_out)
    save_overlay(frames_crop_uint8[bg_idx], best_pca_image, overlay_out)

    make_sheet(
        [
            ("Image/Video", frames_crop_uint8, True),
            ("ThinkJEPA Dense PCA", best_rgb, False),
        ],
        token_indices,
        tubelet_size,
        sheet_out,
        args.img_size,
        args.render_interp,
    )
    make_sheet(
        [
            ("Image/Video", frames_crop_uint8, True),
            ("ThinkJEPA Dense PCA", best_rgb, False),
        ],
        token_indices,
        tubelet_size,
        bicubic_sheet_out,
        args.img_size,
        "bicubic",
    )
    make_sheet(
        [("Image/Video", frames_crop_uint8, True)]
        + [(name, payload["rgb_grid"], False) for name, payload in layer_outputs.items()],
        token_indices,
        tubelet_size,
        comparison_out,
        args.img_size,
        args.render_interp,
    )
    make_sheet(
        [
            ("Image/Video", frames_crop_uint8, True),
            ("FG mask", mask_to_rgb(best_mask), False),
        ],
        token_indices,
        tubelet_size,
        mask_out,
        args.img_size,
        "nearest",
    )

    # 构建汇总清单写入磁盘。以便未来做基于统计分数的评估
    layer_metrics = {
        name: payload["metrics"]
        for name, payload in layer_outputs.items()
    }
    motion = sample_motion_score(frames_np)
    best_score = float(layer_metrics[best_layer]["layer_score"])
    sample_score = float(best_score + min(motion / 2.0, 1.0) * 0.08)
    manifest = {
        "source": str(path),
        "source_name": source_name,
        "input_frames_shape": [int(v) for v in frames_np.shape],
        "processed_frames_shape": [int(v) for v in frames_crop_uint8.shape],
        "crop": crop_meta,
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "img_size": int(args.img_size),
        "model_arch": args.model_arch,
        "patch_size": int(patch_size),
        "tubelet_size": int(tubelet_size),
        "selected_token_t": int(token_t),
        "background_sample_position": int(bg_idx),
        "background_raw_frame_index": int(raw_idx),
        "contact_sheet_tokens": [int(v) for v in token_indices],
        "pca_recipe": args.pca_recipe,
        "foreground_method": args.foreground_method,
        "foreground_quantile": float(args.foreground_quantile),
        "auto_quantiles": [float(v) for v in quantile_candidates],
        "smooth_foreground_mask": bool(args.smooth_foreground_mask),
        "background_mode": args.background_mode,
        "rgb_smooth": float(args.rgb_smooth),
        "rgb_saturation": float(args.rgb_saturation),
        "rgb_gamma": float(args.rgb_gamma),
        "fusion_layers": parse_int_list(args.fusion_layers, []),
        "render_interp": args.render_interp,
        "sample_motion_score": motion,
        "best_layer": best_layer,
        "best_layer_score": best_score,
        "sample_score": sample_score,
        "layers": layer_metrics,
        "layer_variants": {
            name: payload["variant_metrics"]
            for name, payload in layer_outputs.items()
        },
        "outputs": {
            "frame": str(frame_out),
            "pca": str(pca_out),
            "overlay": str(overlay_out),
            "contact_sheet": str(sheet_out),
            "contact_sheet_bicubic": str(bicubic_sheet_out),
            "layer_comparison": str(comparison_out),
            "mask_preview": str(mask_out),
            "manifest": str(manifest_out),
        },
    }
    write_manifest(manifest_out, manifest)
    print(f"[DONE] best layer for {source_name}: {best_layer}", flush=True)
    print(f"[DONE] sheet: {sheet_out}", flush=True)
    return manifest


def copy_best_outputs(best_manifest, out_dir):
    """复制所有当前评估为表现最佳的内容，防止其在临时夹里无法快速浏览"""
    outputs = best_manifest["outputs"]
    for key in ("contact_sheet", "contact_sheet_bicubic", "layer_comparison", "mask_preview", "manifest"):
        src = Path(outputs[key])
        if not src.exists():
            continue
        suffix = src.suffix
        dst = out_dir / f"best_{key}{suffix}"
        shutil.copy2(src, dst)


def run_self_test():
    """造一个假数据，试运行特征测试通道看会不会出逻辑 bug 或越出边界发生断言违背"""
    torch.manual_seed(7)
    t_count, h_grid, w_grid, dim = 4, 16, 16, 64
    features = torch.randn(t_count, h_grid, w_grid, dim) * 0.15
    yy, xx = torch.meshgrid(torch.arange(h_grid), torch.arange(w_grid), indexing="ij")
    mask = ((yy - 8).square() + (xx - 8).square()) < 20
    features[:, mask, :8] += torch.linspace(1.0, 2.0, t_count).view(t_count, 1, 1)
    args = argparse.Namespace(
        pca_recipe="foreground",
        pca_norm="l2_center",
        percentile=1.0,
        foreground_method="hybrid",
        foreground_quantile=0.70,
        smooth_foreground_mask=True,
        background_gray=145,
        background_mode="pca",
        background_desaturate=0.35,
        rgb_smooth=0.18,
        rgb_saturation=1.25,
        rgb_gamma=0.92,
    )
    rgb_grid, fg_mask, metrics = render_feature_grid(features, args)
    assert list(rgb_grid.shape) == [t_count, h_grid, w_grid, 3], rgb_grid.shape
    assert list(fg_mask.shape) == [t_count, h_grid, w_grid], fg_mask.shape
    assert 0.15 <= metrics["foreground_coverage"] <= 0.85, metrics
    assert not metrics["fallback_to_simple_pca"], metrics
    print("SELF_TEST_OK")


def validate_source_args(args):
    """验证终端调用者的传参互相是否不排斥不矛盾。"""
    if args.self_test:
        return
    if not args.out_dir:
        raise ValueError("--out_dir is required unless --self_test is used")
    selected = [bool(args.video), bool(args.npz), bool(args.npz_glob)]
    if sum(selected) != 1:
        raise ValueError("Pass exactly one of --video, --npz, or --npz_glob")
    if args.use_cached_feats and args.npz_glob:
        raise ValueError("--use_cached_feats is not supported with --npz_glob")
    if not (0.0 < float(args.foreground_quantile) < 1.0):
        raise ValueError("--foreground_quantile must be in (0, 1)")
    if not (0.0 < float(args.crop_quantile) < 1.0):
        raise ValueError("--crop_quantile must be in (0, 1)")
    if not (0.0 < float(args.crop_window_fraction) <= 1.0):
        raise ValueError("--crop_window_fraction must be in (0, 1]")
    for value in parse_float_list(args.auto_quantiles, [args.foreground_quantile]):
        if not (0.0 < float(value) < 1.0):
            raise ValueError("--auto_quantiles values must be in (0, 1)")


def main():
    """主函数挂载入口"""
    args = parse_args()
    validate_source_args(args)
    if args.self_test:
        run_self_test()
        return

    repo_root, _ = configure_paths()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_frames is not None and int(args.max_frames) > 0:
        args.num_frames = int(args.max_frames)

    # 检查硬件兜底方案
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU", flush=True)
        args.device = "cpu"

    # 如果没用使用已有的预存特征数据，就去初始化并加载特征网络抽取模型对象
    out_layers = [] if args.use_cached_feats else parse_int_list(args.out_layers, [5, 11, 17, 23])
    layer_names = [f"layer{idx}" for idx in out_layers] if out_layers else ["cached"]
    model = None
    checkpoint_path = None
    if not args.use_cached_feats:
        checkpoint_path = resolve_checkpoint(repo_root, args.checkpoint)
        model = load_dense_jepa_model(
            checkpoint_path,
            args.img_size,
            args.num_frames,
            args.patch_size,
            out_layers,
            args.device,
            args.model_arch,
        )

    # glob 多路径下的统一串行批量执行机制
    if args.npz_glob:
        sources = [Path(path) for path in sorted(glob.glob(args.npz_glob, recursive=True))]
        if not sources:
            raise FileNotFoundError(f"No files matched --npz_glob {args.npz_glob!r}")
        manifests = []
        for src in sources:
            sample_out = out_dir / src.stem
            manifests.append(process_one_source(src, args, model, layer_names, sample_out))
            
        # 根据指标找到效果跑出来最好打分最高的案例图
        best_manifest = max(manifests, key=lambda item: item["sample_score"])
        batch_manifest = {
            "checkpoint": checkpoint_path,
            "npz_glob": args.npz_glob,
            "sample_count": len(manifests),
            "best_sample": best_manifest["source_name"],
            "best_layer": best_manifest["best_layer"],
            "best_sample_score": best_manifest["sample_score"],
            "samples": manifests,
        }
        write_manifest(out_dir / "batch_manifest.json", batch_manifest)
        copy_best_outputs(best_manifest, out_dir)
        print(
            f"[DONE] best sample: {best_manifest['source_name']} "
            f"layer={best_manifest['best_layer']}",
            flush=True,
        )
    else:
        # 单视频/图提取工作流执行
        src = Path(args.npz or args.video)
        manifest = process_one_source(src, args, model, layer_names, out_dir)
        manifest["checkpoint"] = checkpoint_path
        write_manifest(Path(manifest["outputs"]["manifest"]), manifest)


if __name__ == "__main__":
    main()

