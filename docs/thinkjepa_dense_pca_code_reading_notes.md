# ThinkJEPA Dense PCA 可视化代码阅读笔记

> 远端项目路径：`/root/autodl-tmp/thinkjepa-work/ThinkJEPA`  
> 阅读对象：`tools/dense_jepa_pca_vis.py`  
> 目的：解释 ThinkJEPA 当前 dense feature PCA 可视化脚本的数据流、两阶段 PCA、foreground mask、多层选择和输出文件。  
> 说明：这份笔记只阅读和总结代码，不重新跑模型；当前远端是无卡模式。

---

## 1. 这段代码在做什么

`tools/dense_jepa_pca_vis.py` 的目标是把 ThinkJEPA / V-JEPA encoder 产生的 dense patch token feature 映射成 RGB 图，形成类似 V-JEPA README 中 dense PCA teaser 的可视化：

1. 从视频 `--video` 或 ThinkJEPA cache `--npz` 读取帧。
2. 按 ImageNet 规范 resize、center crop、normalize。
3. 用当前 ThinkJEPA 使用的 JEPA encoder checkpoint 编码视频，抽取指定 transformer layer 的 dense token。
4. 把 token reshape 成 `[T_token, H_patch, W_patch, D]`。
5. 对所有 patch token 做 PCA，把前 3 个主成分映射成 RGB。
6. 输出单帧图、PCA 图、overlay、2 行 contact sheet、layer comparison、mask preview 和 manifest。

注意：这里可视化的是 **dense visual feature 的 PCA 投影**，不是 attention heatmap，也不是 VLM token attention。

---

## 2. 主要入口和参数

入口函数是 `main()`，核心路径如下：

```text
main
  -> parse_args
  -> validate_source_args
  -> configure_paths
  -> resolve_checkpoint
  -> load_dense_jepa_model
  -> process_one_source
       -> load_input_for_path
       -> apply_spatial_crop
       -> preprocess_frames
       -> build_feature_grids
       -> render_feature_grid
       -> make_sheet / save_overlay / write_manifest
```

关键输入参数：

| 参数 | 含义 |
|---|---|
| `--video` | 直接输入原始视频文件，例如猫猫 `.webm`。 |
| `--npz` | 输入 ThinkJEPA cache `.npz`，读取其中 `imgs`。 |
| `--npz_glob` | 批量读取多个 `.npz`，自动挑 best sample。 |
| `--checkpoint` | 显式指定 encoder checkpoint；否则按环境变量和默认路径查找。 |
| `--img_size` | encoder 输入分辨率，例如 256/384/512。 |
| `--patch_size` | patch 大小，当前默认 16。 |
| `--max_frames` | 对输入帧均匀下采样，降低显存/算力需求。 |
| `--out_layers` | 抽取哪些 encoder block 的特征，如 `5,11,17,23`。 |
| `--fusion_layers` | 可选，把多个层的 L2-normalized feature 拼接后再 PCA。 |
| `--pca_recipe` | `foreground` 为两阶段 PCA；`simple` 为全 token 单次 PCA。 |
| `--foreground_method` | `pc1`、`norm` 或 `hybrid`，决定前景分数来源。 |
| `--foreground_quantile` | foreground score 阈值分位数。 |
| `--auto_quantiles` | 对每层尝试多个 quantile，自动选分数最高的变体。 |
| `--smooth_foreground_mask` | 是否对前景 mask 做空间平滑和去噪。 |
| `--background_mode` | 非前景 token 的颜色处理：`gray`、`pca`、`desaturate`。 |
| `--rgb_smooth` | 对 PCA RGB patch grid 做 3x3 空间平滑混合。 |
| `--render_interp` | 主图放大方式，`nearest` 保留 patch-grid 颗粒感，`bicubic` 更平滑。 |

---

## 3. Checkpoint 解析逻辑

`resolve_checkpoint()` 的优先级是：

1. 命令行 `--checkpoint`。
2. 环境变量 `THINKJEPA_JEPA_VITL_PT`。
3. 默认候选：
   - `vjepa2/vitl.pt`
   - `checkpoints/vitl.pt`
   - `checkpoints/thinkjepa_jepa_vitl.pt`

这份代码当前使用的是 ThinkJEPA 目录下已有的 `vjepa2/vitl.pt`，并不是切换到 V-JEPA 2.1 checkpoint。代码里 help 文案中残留的 “V-JEPA2.1 runs” 只是参数说明文字残留，不代表运行时加载了 2.1 权重。

模型加载函数 `load_dense_jepa_model()` 会：

1. 从 `vjepa2.src.models.vision_transformer` 中取 `--model_arch`，默认 `vit_large_rope`。
2. 构造 encoder：
   - `img_size=(S,S)`
   - `num_frames=T`
   - `patch_size=16`
   - `tubelet_size=2`
   - `out_layers=[...]`
3. 加载 checkpoint 中的 `encoder` 或 `model` state dict。
4. 清理 key 前缀 `module.` 和 `backbone.`。
5. `model.eval()` 后用于抽取 dense feature。

---

## 4. 端到端数据流 shape

通用 shape ledger：

```text
frames_np          [T, H, W, 3]
frames_norm        [T, 3, S, S]
video              [1, 3, T, S, S]
features           [N, D]
feature_grid       [T/2, S/16, S/16, D]
pca_rgb_grid       [T/2, S/16, S/16, 3]
foreground_mask    [T/2, S/16, S/16]
contact_sheet      2 行：Image/Video + ThinkJEPA Dense PCA
```

为什么是 `T/2`：当前 encoder 构造时 `tubelet_size=2`，所以每 2 帧对应 1 个时间 token。

为什么是 `S/16`：当前默认 `patch_size=16`，因此 512 输入对应 `32x32` patch grid，256 输入对应 `16x16` patch grid。

以当前猫猫样本 manifest 为例：

```text
source: inputs/cat_plays_wikimedia.webm
input_frames_shape:      [16, 720, 720, 3]
processed_frames_shape:  [16, 512, 512, 3]
feature_grid_shape:      [8, 32, 32, 1024]
pca_rgb_grid_shape:      [8, 32, 32, 3]
foreground_mask_shape:   [8, 32, 32]
best_layer:              layer5
foreground_coverage:     0.3101
checkpoint:              vjepa2/vitl.pt
```

这说明本次可视化的精细程度上限是 `32x32` patch token，而不是像素级 mask。最终图片看起来仍有 patch 颗粒感是正常的。

---

## 5. 输入读取与预处理

输入分两类：

1. `--video`：用 `decord.VideoReader` 读取视频，按 `uniform` 或 `centered` 采样。
2. `--npz`：读取 `npz["imgs"]`，要求 shape 是 `[T,H,W,C]`，C 可以是 3 或 4。

`preprocess_frames()` 做了三件事：

1. 将 uint8 图像转到 `[0,1]` float。
2. resize 到短边 `256/224 * img_size`，再 center crop 到 `[S,S]`。
3. 使用 ImageNet mean/std normalize。

输出：

```text
frames_norm       [T, 3, S, S]    # 给 encoder
frames_crop_uint8 [T, S, S, 3]    # 给 contact sheet 和 overlay
```

`encode_feature_layers()` 会把 `[T,3,S,S]` 改成视频模型需要的：

```text
video = frames_norm.permute(1,0,2,3).unsqueeze(0)
shape = [1, 3, T, S, S]
```

---

## 6. Dense feature grid 构建

`build_feature_grids()` 有两种模式：

1. 默认模式：重新用 encoder 编码帧，得到指定层 feature。
2. `--use_cached_feats`：直接用 `.npz` 中的 `vjepa_feats`，但只有当 `P == H_patch * W_patch` 时才允许。

默认编码后，`infer_token_grid()` 根据模型参数推断：

```text
T_token = num_frames // tubelet_size
H_patch = img_size // patch_size
W_patch = img_size // patch_size
expected_tokens = T_token * H_patch * W_patch
```

如果 encoder 输出 token 数和这个 expected 不一致，代码会直接报错，避免把 token 错 reshape 成错误的空间网格。

---

## 7. PCA 的核心逻辑

### 7.1 特征归一化

`normalize_features_for_pca()` 支持三种模式：

| 模式 | 处理 |
|---|---|
| `center` | 每个维度减均值。 |
| `l2_center` | 先对 token 做 L2 normalize，再减均值；当前默认。 |
| `standardize` | 减均值后除标准差。 |

默认 `l2_center` 更接近 dense feature visualization 的常见做法，因为它先消除 feature norm 尺度差异，再看方向结构。

### 7.2 PCA 分解

`fit_pca()` 用的是：

```text
centered = tokens - tokens.mean(0)
U, S, Vh = torch.linalg.svd(centered)
basis = Vh[:3].T
```

前 3 个主成分作为 RGB 三个通道的投影方向。

`apply_pca()` 会固定 PCA sign：每个主成分找到绝对值最大的 token，如果这个 token 在该通道为负，就把整列乘以 -1。这样可以减少不同运行或不同层之间颜色翻转的问题。

### 7.3 robust RGB normalization

`robust_rgb_from_projection()` 不直接用 min/max，而是按分位数做 channel normalization，例如默认 `1%~99%`。这可以减少少数异常 token 把颜色范围拉爆。

---

## 8. 两阶段 PCA 和 foreground mask

当前默认 `--pca_recipe foreground`，不是简单全 token PCA。

两阶段流程：

```text
第 1 阶段：
  所有 token -> L2 normalize/center -> PCA
  取 PC1 或 feature norm 构造 foreground score

foreground mask：
  score reshape 成 [T_token,H_patch,W_patch]
  按 quantile 阈值取前景 token
  可选 3x3 平滑和去噪

第 2 阶段：
  只用 foreground token 重新 fit PCA
  再把所有 token 投影到这个 PCA basis
  输出 RGB patch map
```

foreground score 的三种方式：

| `foreground_method` | 含义 |
|---|---|
| `pc1` | 用第一阶段 PC1 绝对值作为前景强度。 |
| `norm` | 用 feature norm 偏离均值的程度作为前景强度。 |
| `hybrid` | `0.65 * pc1_score + 0.35 * norm_score`，当前默认。 |

前景覆盖率保护：

```text
coverage < 0.15
coverage > 0.85
foreground token 数 < 4
```

出现这些情况时，代码会 fallback 到 `simple_pca_rgb()`，避免 mask 太小或太大导致 PCA 不稳定。

背景 token 的颜色处理由 `--background_mode` 控制：

| 模式 | 效果 |
|---|---|
| `gray` | 非前景 token 直接置灰。 |
| `pca` | 非前景 token 也保留 PCA 颜色。 |
| `desaturate` | 非前景 token 颜色向灰色混合。 |

当前猫猫最终图使用的是 `desaturate`，这比纯灰背景更自然，也比全 PCA 背景更不容易让噪声抢主体。

---

## 9. 多层 feature 对比与 best layer 选择

默认 `--out_layers 5,11,17,23`，每一层都会单独执行 PCA 和 mask 逻辑。每层还可以通过 `--auto_quantiles` 尝试多个 foreground quantile。

每个候选结果记录：

```text
feature_grid_shape
pca_rgb_grid_shape
foreground_mask_shape
foreground_coverage
pca_explained_variance
spatial_smoothness
temporal_consistency
foreground_colorfulness
fallback_to_simple_pca
foreground_quantile_effective
```

`score_layer_metrics()` 的打分组成：

```text
0.30 * spatial_smoothness
0.25 * temporal_consistency
0.20 * coverage_score
0.15 * foreground_colorfulness
0.10 * explained_variance_sum
+ small layer_bias
```

直观理解：

1. 空间上越连续越好。
2. 时间上颜色越稳定越好。
3. foreground coverage 接近中等比例更好。
4. 前景颜色有区分度更好。
5. PCA 前 3 维解释方差越高越好。
6. 有一个很小的后层偏置，但不会硬编码一定选后层。

当前猫猫 512 clean 结果只跑了 `layer5`，所以 best layer 就是 `layer5`。

---

## 10. 渲染输出

每个 source 会输出：

| 文件 | 说明 |
|---|---|
| `*_frame.png` | 选中的原始背景帧。 |
| `*_pca.png` | 单个 token time 的 PCA RGB 图。 |
| `*_overlay.png` | 原图和 PCA 图的叠加。 |
| `*_best_contact_sheet.png` | 主图，2 行：`Image/Video` + `ThinkJEPA Dense PCA`。 |
| `*_best_contact_sheet_bicubic.png` | bicubic 放大诊断版。 |
| `*_layer_comparison.png` | 多层 PCA 对比图。 |
| `*_mask_preview.png` | foreground mask 可视化。 |
| `*_manifest.json` | 记录 shape、参数、指标和输出路径。 |

主图示例路径：

```text
../outputs/dense_pca_cat_vitl512_layer5_clean/cat_plays_wikimedia_best_contact_sheet.png
```

对应 mask：

```text
../outputs/dense_pca_cat_vitl512_layer5_clean/cat_plays_wikimedia_mask_preview.png
```

---

## 11. 为什么和 V-JEPA README 的图还有差距

这份可视化链路已经在算法上接近常见 dense PCA 展示：L2 normalize、foreground PCA、robust normalization、前景 mask、多层对比、patch-grid 渲染。但和 V-JEPA README 里的展示仍有客观差距：

1. **空间分辨率上限不同**：512 输入、16 patch 只有 `32x32` token grid。边界不可能像像素级 segmentation 一样锐利。
2. **PCA 不是监督分割**：颜色代表 feature 主方向，不代表类别标签；语义连续性取决于 encoder feature 本身是否把主体和背景分开。
3. **样本影响很大**：主体大、边界清楚、运动明显的视频会显著好于 tiny-cache 中主体小或纹理复杂的样本。
4. **当前不能换 checkpoint**：这里坚持使用 ThinkJEPA 当前 encoder，不切到 V-JEPA2.1，因此不会伪装成 README 原模型的效果。
5. **后处理只能改善观感**：`rgb_smooth`、`desaturate`、`bicubic` 可以减少噪声，但不能凭空创造更高语义精度。

因此这张图应该被解释为：

> ThinkJEPA 当前 dense visual token feature 的 PCA 可解释投影，而不是 V-JEPA2.1 原论文 teaser 的严格复现。

---

## 12. 当前猫猫样本的真实结论

当前已有猫猫结果是一个比 tiny-cache 更适合展示 dense PCA 的样本：

```text
input_frames_shape:      [16, 720, 720, 3]
processed_frames_shape:  [16, 512, 512, 3]
feature_grid_shape:      [8, 32, 32, 1024]
best_layer:              layer5
foreground_coverage:     0.3101
spatial_smoothness:      0.7007
temporal_consistency:    0.5987
foreground_colorfulness: 0.2716
checkpoint:              vjepa2/vitl.pt
```

这说明：

1. 链路确实是在 ThinkJEPA 当前 encoder 上跑出的 dense PCA。
2. 512 输入带来了比 256 输入更细的 `32x32` patch grid。
3. foreground coverage 约 31%，主体区域没有被全背景淹没。
4. 但视觉精度仍受 patch token 粒度和 encoder 表征质量限制。

---

## 13. 建议后续阅读代码时重点看哪里

如果只看最重要的函数，建议按这个顺序：

1. `parse_args()`：了解脚本能力和可调参数。
2. `load_dense_jepa_model()`：确认加载的是哪个 encoder 和 checkpoint。
3. `preprocess_frames()`：确认输入对齐方式。
4. `encode_feature_layers()`：确认视频 tensor layout。
5. `infer_token_grid()`：确认 token 如何恢复成空间网格。
6. `foreground_pca_rgb()`：理解两阶段 PCA 的核心。
7. `score_layer_metrics()`：理解自动选层指标。
8. `process_one_source()`：串起输入、PCA、渲染和 manifest。

最关键的一句话：

> 这份代码不是直接“把图片降维”，而是先用 ThinkJEPA/V-JEPA encoder 把视频变成 dense patch feature，再在整段 clip 的 token feature 空间里做 PCA，把前三个主成分渲染成 RGB。

---

## 14. 详细补充：Token映射与核心实现批注

为了帮助精确理解 **"将 patch token reshape 为 [T_token, H_patch, W_patch, D]"** 这一过程及其上下文特征提取机制，以下抽取了代码中最重要的三个模块，并给出了*非常详细*的中文逐行说明和索引映射公式注释。

### 14.1 核心一：如何确定 Token Grid 的形状和索引映射

```python
def infer_token_grid(model, num_frames, img_size, feat_tokens):
    """
    推断一维序列 [N, D] 应该如何折叠回 [T_token, H_patch, W_patch, D] 的四维时空网格。
    这是JEPA/ViT dense特征可视化最底层的核心操作。
    """
    # 拿到模型切 patch 的参数，默认 patch_size=16
    patch_size = int(getattr(model, "patch_size", 16))
    
    # 获取时间维度的 tubelet 大小（将时间上相邻的帧打包），默认 tubelet_size=1 或 2
    tubelet_size = int(getattr(model, "tubelet_size", 1))
    
    # 1. 计算时间维度 token 数量 T_token
    # 原视频有 num_frames 帧，每 tubelet_size 帧压成1个token，于是 T_token = T / 2
    t_grid = int(num_frames) // tubelet_size
    
    # 2. 计算空间维度 token 数量 H_patch, W_patch
    # 输入图像边长是 img_size，每个 patch 大小是 16，所以 H_patch = img_size / 16
    h_grid = int(img_size) // patch_size
    w_grid = int(img_size) // patch_size
    
    # 3. 理论上应该产生的 token 总数 N
    expected = t_grid * h_grid * w_grid
    
    # 4. 安全校验：如果模型实际出来的 token 数量 =/= 理论数量
    # (例如输入没裁切干净或包含了额外 CLS_token 等没去掉的情况)，立即报错
    if expected != int(feat_tokens):
        raise RuntimeError(...)
        
    return t_grid, h_grid, w_grid, patch_size, tubelet_size

'''
【深入理解：这里的数学映射逻辑】
假设 T_token=8, H_patch=14, W_patch=14。
一维 token 序列总长 N = 8 * 14 * 14 = 1568。
对于序列中第 i 个 token (其中 i 在 0 包含至 1567 之间)：
  - 它所在的时刻 (t): t = i // (14 * 14) 
  - 减去历史时刻占据的量: rem = i % (14 * 14)
  - 它在图像上的纵向位置 (h_coord): h = rem // 14
  - 它在图像上的横向位置 (w_coord): w = rem % 14
当我们用 `feat.reshape(t_grid, h_grid, w_grid, dim)` 这个操作时，
PyTorch在内存中实际上就是自动遵循了这个字典级别的索引映射公式，将序列切分为三维空间。
'''
```

### 14.2 核心二：两阶段特征 PCA 与 RGB 映射

这段包含了如何把高维特征映射为RGB并在两阶段实现“排除背景干扰”的核心方法：

```python
def foreground_pca_rgb(
    feat_grid,
    mode,
    percentile,
    method,
    quantile,
    smooth,
    ...
):
    """
    两阶段 PCA 处理：
    第一阶段：在所有 tokens 空间上找出 foreground (前景对象)。
    第二阶段：只对 foreground token 运用 PCA 计算颜色基底，再把所有的 token 按照基底着色。
    """
    # 拆包网络：[T_token, H_patch, W_patch, D]
    t_count, h_grid, w_grid, dim = feat_grid.shape
    
    # 1. 展平回一维，以备计算 PCA: shape 变成 [N, D]
    flat = feat_grid.reshape(-1, dim).float()
    
    # 2. 特征归一化（推荐对每个token进行 l2 normalize后减总体均值）
    tokens = normalize_features_for_pca(flat, mode=mode)

    # -----------------------------------------------
    # 阶段一：用所有 token 去做 PCA 拟合获取第一组特征基
    # fit_pca 是做 SVD(主成分分解)，并返回 3 个最大的方差方向
    center1, basis1, explained1 = fit_pca(tokens, components=3)
    
    # 求整个张量投射过去的前三成分 
    projection1 = apply_pca(tokens, center1, basis1)
    
    # 使用"大绝对值的主成分PC1"(模型学习到的最明显差异通常反映了主体vs背景) 以及 "特征范数差" 作为候选分
    pc1_score = rank01(torch.abs(projection1[:, 0]))
    norm_score = rank01(torch.linalg.norm(flat - flat.mean(dim=0, keepdim=True), dim=1))
    
    # 根据指定的 method（通常 hybrid 结合两者）得出一个 [N] 大小的 foreground 评分序列
    score = 0.65 * pc1_score + 0.35 * norm_score

    # 把长度为 N 的分数序列重新 reshape 成空间状态 [T_token, H_patch, W_patch] 
    # 从一维拉成空间网格，以便后续可能会在这一步做 3x3 的均值平滑/中值去噪 (smooth=True)
    score_grid = score.reshape(t_count, h_grid, w_grid)
    mask_grid = foreground_mask_from_scores(score_grid, quantile, smooth)
    
    # 展平以便取前景的索引
    mask = mask_grid.reshape(-1)
    
    # -----------------------------------------------
    # 阶段二：重点只对选中的 foreground mask 再做一次精确 PCA。
    # 把不含背景的主体取出来（从而排除了背景对基底成分权重的干扰，突出主体结构颜色）
    foreground_tokens = tokens[mask]
    center2, basis2, explained2 = fit_pca(foreground_tokens, components=3)
    
    # 但是！做完投影特征基（basis2）后，应用投射对象变回【全部tokens】
    # 以保证连背景都有这套基准颜色的映射结果。
    projection2 = apply_pca(tokens, center2, basis2)
    
    # 按照 percentile（比如 1%~99%）作为(0, 1)边界对 projection 进一步做归一化(抗锯齿和突刺)，映射为 RGB
    rgb = robust_rgb_from_projection(projection2, mask, percentile)

    # 根据设定弱化背景：将非前景区域按设定的不同方法压成灰色或降饱和。
    # 比如 background_mode="desaturate" 让背景颜色偏灰。
    if background_mode == "desaturate":
        mix = float(np.clip(background_desaturate, 0.0, 1.0))
        rgb[~mask] = (1.0 - mix) * rgb[~mask] + mix * gray
        
    ...
    # 恢复原状，最终输出为图像网格 [T, H, W, 3] 范围 0-255 字节值。
    rgb_grid = (rgb * 255.0).byte().cpu().numpy()
    return rgb_grid, mask_grid.cpu().numpy(), explained2 or explained1, False
```

### 14.3 核心三：在原图和 Patch Array 上的逆向关联映射操作

渲染阶段的 `make_sheet()` 调用过程解释了怎么把 patch 网格坐标（token time 等）切回原始的时间索引：

```python
def background_frame_index(token_t, tubelet_size, num_frames):
    """
    此方法用于逆向溯源：这个 patch token (时间 t=token_t) 代表的是原视频中的哪一帧?
    举例，如果 tubelet_size = 2，代表每两个帧产生1个时间 token。
    假设我们选中了 token_t = 3 （即第 4 个 token），
    在原视频它的基础索引就是：3 * 2 + (2//2) = 6 + 1 = 7。
    说明这个时间层在原视频的最代表帧是第 8 帧。
    """
    pos = token_t * max(1, tubelet_size) + max(0, tubelet_size // 2)
    return int(np.clip(pos, 0, num_frames - 1))
```

所有上面的操作归总为一个目标：**从网络的一维嵌入向量空间（毫无空间结构），通过已知的 `t`, `h`, `w` 格子属性复原为有实际几何关系的网格，在这个结构网格内分别实施 PCA、二维高斯平滑以及图像插值技术。**

