# Encoder-side Token Merge (方案A) vs No-Merge → PCA:对照实验分析

**日期**: 2026-05-29 · **分支**: `refactor-and-exp` · **机器**: NVIDIA A800-80GB
**脚本**: `tools/run_token_merge_pca_experiment.py`
**原始结果**: `outputs/token_merge_pca_AvsB_20260529/`
(`pca_experiment_results.json` / `pca_experiment_summary.csv` / `pca_singular_values.npz`)

## 实验设置(两组完全可比)

- **输入**:tiny-cache `part2` 全部 3 个视频(`1873` 椅子 / `5522` 抽屉 / `542` 柜子),每个取 `imgs` 的 64 帧、256×256,在线跑 encoder(**不用** 缓存的 `vjepa_feats`)。
- **Encoder**:ThinkJEPA `vjepa2/vitl.pt`(ViT-L, RoPE),`patch=16, tubelet=2` → dense token 网格 `32×16×16 = 8192`。checkpoint 完全匹配(missing=0, unexpected=0)。
- **组 A(baseline)**:视频 → encoder(**不 merge**)→ 最终 8192 token 特征 → PCA。
- **组 B(方案A)**:同视频 → encoder,在 **layer 8** 用 `local_2x2_same_time_vec`(纯相似度 2×2 同时刻 merge,`restore_dense=False` 真实缩短序列)→ 特征 → PCA。比例 `r ∈ {0.05, 0.125, 0.25}`。
- **PCA 设置(A/B 完全一致)**:对 token 特征矩阵 `[N, 1024]` 仅做**去均值**(标准 PCA,不缩放),`torch.linalg.svd`;统一报告:各主成分解释方差比、累计方差、达到 90/95/99% 方差所需分量数、固定分量数 `k` 的相对重建误差。
- 计时为 encoder 前向 wall-clock(fp32,3 次取均值,含 1 次 warmup),显存为该次前向 `max_memory_allocated`。**所有数字来自真实前向,无估算。**

## 对照结果(3 视频平均)

| 配置 | token 数 | 序列↓ | encoder 时间 | 加速 | 峰值显存 | cos vs A | relL2 vs A | ncomp@90% | ncomp@95% | ncomp@99% | 重建误差@k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A baseline** | 8192 | — | 603.6 ms | — | 1691 MB | — | — | 160.0 | 277.0 | 585.3 | 0.771 |
| **B  r=0.05** | 7783 | −5.0% | 591.0 ms | −2.1% | 1700 MB | 0.955 | 0.302 | 159.0 | 275.7 | 583.7 | 0.771 |
| **B  r=0.125** | 7168 | −12.5% | 546.0 ms | −9.5% | 1723 MB | 0.894 | 0.465 | 158.0 | 273.7 | 580.7 | 0.770 |
| **B  r=0.25** | 6144 | −25.0% | 489.8 ms | −18.9% | 1717 MB | 0.804 | 0.627 | 156.0 | 271.0 | 575.0 | 0.769 |

> 单视频数据见 `pca_experiment_summary.csv`,三个视频趋势完全一致(差异 < 2%)。
> 前 3 个主成分累计方差仅 ~0.25(A/B 都是),k=32 ~0.666,k=64 ~0.781 —— 三组几乎不变。

## 分析

### 1. token merge 对 PCA 降维 / 特征质量的影响:几乎无损,甚至略更紧凑

依据:上表 `ncomp@90/95/99%` 与 `重建误差@k=8`(来自 `pca_experiment_results.json` 各 sample 的 `pca` 字段)。

- 达到同样方差所需的主成分数,A→B 只微降(95%:277→271,90%:160→156),方向是**更少**分量即可解释同样方差——说明 merge 去掉的是冗余 token,主成分结构没有被破坏。
- 固定 k 的重建误差几乎不动(k=8:0.771→0.769;各 k 见 JSON 的 `recon_relerr_k*`)。各主成分解释方差比逐条几乎重合(PC1 0.109→0.111)。
- **结论**:在 layer 8 用方案A 砍掉 5–25% 的 dense token,对最终 encoder 特征的 PCA 谱/低维结构**几乎没有负面影响**。这与"dense video token 存在大量冗余"的判断一致。

### 2. 加速 / 收束 encoder:有真实 wall-clock 加速,但代价是保真度,且不省显存

依据:上表 `encoder 时间`、`峰值显存`、`cos/relL2 vs A`(来自 `summary.csv`)。

- **划算的部分**:layer 8 merge 后,9–24 层在更短序列上算,encoder 前向**真实加速**,且随比例增大:r=0.05 ≈ −2%,r=0.125 ≈ −9.5%,r=0.25 ≈ −19%。序列长度按比例精确下降(8192→6144)。
- **代价 1 — 保真度**:与 baseline 的逐 token cosine 随比例下降(0.955 → 0.894 → 0.804),relL2 上升(0.30 → 0.47 → 0.63)。即比例越高,合并后的特征集合越偏离原始 dense 表征。
- **代价 2 — 不省显存**:峰值显存基本不变(~1.7 GB),B 甚至略高。原因:0–8 层仍处理完整 8192 token,且 merge 的 gather/scatter + restore 有额外分配;峰值出现在 merge 之前。**所以"省显存"在当前单层 merge 配置下不成立。**

### 3. 明确结论与建议

| 问题 | 结论 | 依据 |
|---|---|---|
| 保留多少 PCA 维? | dense 特征本征维偏高:90% 需 ~160 维、95% 需 ~275 维、99% 需 ~585 维(/1024)。前 3 维仅 ~25%(故 RGB-PCA 可视化天然"块状")。merge 不改变这些阈值。 | 上表 + JSON `evr_top16` |
| 是否上 token merge? | **低比例(r=0.05–0.125)值得上**:换来 2–10% encoder 加速、序列缩短,而 PCA 结构几乎无损、保真度仍有 0.89–0.96。**r=0.25 谨慎**:加速 ~19% 但保真度掉到 ~0.80。 | `summary.csv` 时间 + cos 列 |
| 方案A 划算吗? | 在"只看 encoder 前向延迟 + 特征 PCA 质量"目标下,**低比例划算**:加速真实、PCA 几乎不掉。但若目标是省显存,或需要高保真逐 token 特征,则收益有限。 | 时间↓ vs 显存持平 vs cos↓ |
| 推荐工作点 | **layer 8 + 方案A + r≈0.125**:序列 −12.5%、encoder −9.5%、cos 0.89、PCA 95% 分量 277→274(几乎不变)。要更稳取 r=0.05;要更快、能接受偏差取 r=0.25。 | 上表整行对比 |

## 诚实的局限(不要过度外推)

- **规模**:tiny-cache 仅 3 个视频、encoder-only、fp32、单 GPU、单 merge 层(layer 8)。这是诊断级对照,**不是完整 ThinkJEPA 大规模 benchmark**。
- 加速是 **encoder 前向** wall-clock,**不等于**完整 ThinkJEPA 下游(predictor/trajectory)端到端加速;下游仍可能因 `restore_dense` 或表征偏差受影响,本实验未评估下游 ADE。
- `cos/relL2 vs A` 衡量的是**合并表征相对 baseline 的偏差**,不是下游任务精度;它说明"特征变了多少",不直接等于"任务掉了多少"。
- 显存结论绑定"单层 layer-8 merge + restore"配置;更早 merge 或多层 merge 的显存/速度权衡需另测。
- PCA 在**最终 encoder 特征**上做;若在 merge 当层或其它层取特征,数值会不同。

## 一句话总结

> 在 ThinkJEPA dense encoder 上,方案A(纯相似度 2×2 merge)能在 layer 8 真实缩短序列并带来 encoder 前向加速(r=0.25 约 −19%),且对最终特征的 PCA 结构几乎无损(95% 方差所需分量 277→271,重建误差不变);代价是逐 token 保真度随比例下降(cos 0.96→0.80)、且当前单层配置下不省显存。**推荐低比例(r≈0.05–0.125)作为划算工作点**,但这是 tiny-cache encoder-only 结论,尚不能外推到完整下游 benchmark。
