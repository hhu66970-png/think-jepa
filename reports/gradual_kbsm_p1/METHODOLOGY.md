# Token Merge 优化方法论(速度优先,精度其次)

> 目标排序(按你的要求):**① 先最大化推理 wall-clock 加速;② 精度作为护栏**——只要不跌破阈值即可,不追求"更准"。
> 范围:**训练无关 / 冻结 V-JEPA encoder / SDPA**。先在实验 harness 验证,跑赢现状(方案A)再进主路径。

---

## 0. 先认清"加速到底来自哪"(基于我们 Step 2 的实测)

每层 encoder 成本 ≈ attention `O(N²·d)` + MLP `O(N·d²)`,`N`=token 数。**merge 只对"merge 层之后"的层省时间。**

我们实测(单层 @layer8,8192 tokens,A800 fp32):

| 配置 | token | 序列↓ | encoder | 加速 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| A baseline | 8192 | — | 603.6ms | — | 1691MB |
| 方案A r=0.125 | 7168 | −12.5% | 546.0ms | −9.5% | 1723MB |
| 方案A r=0.25 | 6144 | −25% | 489.8ms | **−18.9%** | 1717MB |

**三条结论直接定义了"速度优先"该怎么改:**
1. 单层 @layer8 → 只有 16/24 层受益,加速封顶 ~19%。**要更快 = merge 更早 + 跨更多层渐进 merge。**
2. `restore_dense=True` 把序列还原回 8192 → 下游(predictor)完全没省 → 加速停在 encoder。**要端到端加速 = 走 compressed path,不 restore。**
3. 峰值显存没降(前 8 层仍满 token + restore 额外分配)。**要省显存 = 早层就把 N 降下来。**

---

## 1. 核心方法:Gradual K-BSM(渐进式 ToMe,compressed path)

新策略(实验名暂定 `bsm_ksim_gradual_vec`),5 个组件,**全部训练无关、SDPA-safe(不需要 attention 概率矩阵)**:

### (1) 匹配度量:attention **Key (K)** 的 cosine,而非原始 hidden cosine
- 依据:ToMe 消融 K=84.25% > 原始特征 83.70%(同速),**K 本来就在前向里算了,零额外算力**。
- 速度视角:K 让"同样 ratio 下保真更高" → 我们能**在不掉精度的前提下 merge 得更狠** → 间接换更多速度。
- SDPA 安全性:K 只是个线性投影,我们在 block 里 hook 出 `k`(SDPA 之前),**不碰 attention 概率矩阵**,不破坏 fused kernel。

### (2) 匹配拓扑:ToMe **Bipartite Soft Matching (BSM)**,替代 2×2 局部
- 把 token 交替分成 A/B 两组 → 每个 A 连到最相似的 B → 保留 top-r 边 → 合并。我们的 2×2 cell 是它的"严格受限特例"。
- 速度视角:全局匹配在同 ratio 下找到更好的 pair → 同上,允许更狠地砍。非迭代、可向量化、保持 batched 推理(r 与内容无关)。

### (3) 调度:**每层砍 r 个,跨层范围 `[L_start, L_end]` 渐进 merge**(替代单层一次砍)
- 这是**速度优先的最大杠杆**:越早开始 + 跨越多层,后面所有层都在更短序列上算,加速**累乘**。
- 旋钮:`L_start`(越早越快、但越靠前特征越不聚类、风险越高)、每层 `r`(可常数或递减 schedule)。
- 例:从 L4 起每层砍一点、到 L12 累计 −40%,远比"L8 一次 −25%"省时间,且峰值显存也降。

### (4) 聚合:**size 加权平均**(我们已在维护 `token_size`,只是没用进聚合)
- 零成本、SDPA-safe 的廉价保真保险,替代"并入高 norm token"的启发式。

### (5) 默认 **不 restore_dense(compressed path)**
- 保留缩短后的序列跑完 encoder。端到端加速需要 predictor 也吃 compressed tokens(Phase 3,涉及 predictor,单独做)。
- `restore_dense=True` 仅作为"要和 dense baseline 对齐算保真"时的诊断开关。

---

## 2. 明确**不做**什么(速度优先的取舍)

| 方法 | 为什么速度优先下先不做 |
|---|---|
| **proportional attention**(softmax 加 log s) | 需要 attention 矩阵 → **SDPA 冲突**,会拖慢 wall-clock;且 ToMe 报告自监督/MAE 模型 off-the-shelf **不需要它**,V-JEPA 很可能也不需要。**仅当护栏被击穿且愿付代价时再考虑。** |
| **attention-熵 saliency(vid-TLDR)** | 需要 full attention map → SDPA 上有真实开销,**吃掉加速**。 |
| **ToSA(特征+空间融合)** | 依赖 Depth-Anything-v2 深度图,V-JEPA 没有 → 不可迁移。 |
| **TempMe 原样** | 需 LoRA/微调,非 plug-and-play。 |

> 注:跨帧/时间 merge(吃视频最大冗余)是**有吸引力但训练无关证据弱**的方向,作为 Phase 2 的可选 toggle(改 BSM 的 A/B 邻域允许跨相邻帧),**必须用我们自己的 harness 验证**,不直接信文献数字。

---

## 3. 衡量协议(速度优先 → 看 wall-clock,不看 FLOPs)

复用并扩展 `tools/run_token_merge_pca_experiment.py`:

- **主指标**:encoder 前向 **wall-clock**(均值/中位,含 warmup)+ **峰值显存**,以及**逐层 token 轨迹**(看序列怎么一层层降)。
- **护栏指标(精度)**:与 dense baseline 的 token 特征 cosine、PCA 解释方差/重建误差(restore 对齐后算)。
- **产出**:画出**速度–保真 Pareto 曲线**,报告"保真 ≥ 阈值 下的最大加速"工作点。
- **诚实**:只认实测 wall-clock 与显存;FLOPs 降 ≠ 时间降(SDPA 已经很快)。merge op 本身必须保持向量化(早期 naive 版曾 10× 变慢,已修,不能回退)。

**成功判据**:在相同保真护栏下,Gradual K-BSM 的**加速明显优于现状方案A**(单层 @L8)。

---

## 4. 分阶段计划(每阶段先速度、后看护栏)

| 阶段 | 做什么 | 主看 | 涉及 |
|---|---|---|---|
| **P1 纯 encoder 加速** | Gradual K-BSM + size 加权,不 restore;扫 `L_start`/`r-schedule` | wall-clock↑ / 显存↓ vs 方案A,护栏=cos | 仅 encoder + harness |
| **P2 再加速** | merge 更早 / 累计 r 更高 /(可选)跨帧匹配 | 速度–保真 Pareto 推进 | encoder + harness |
| **P3 端到端加速** | compressed tokens 直送 predictor(不 restore),量全链路 + 下游 ADE | 全链路 wall-clock + ADE | **改 predictor** |
| **P4 精度回收(仅护栏被破时)** | 调 size 加权 /(最后手段)proportional attention 即便有 SDPA 代价 | 在守住速度的前提下补保真 | 视情况 |

---

## 5. 落地形态(代码层)

- 新增策略字符串 `bsm_ksim_gradual_vec`,**先放进 `DiagnosticTokenMerger`(实验侧)**,用 harness 跑赢方案A 后再考虑进主路径——延续我们 Step 1 的"主路径干净 + 诊断隔离"结构。
- 关键改点:
  - `vision_transformer` block:暴露/hook 出该层的 `k`(SDPA 前),传给 merger 当匹配度量;**不改 SDPA 调用**。
  - merger:BSM 选 pair(K cosine)+ size 加权聚合;`merge_layers` 扩成范围 + 每层 r 调度;默认 `restore_dense=False`。
  - harness:加 `--strategy bsm_ksim_gradual_vec --merge_layers L_start..L_end --r_per_layer ...`,输出逐层 token 轨迹 + wall-clock + 显存 + 保真。

---

## 6. 风险与未知(诚实标注,避免重蹈上一轮过度声称)

- 早层 merge 可能更伤保真(早层特征不够聚类)——**经验问题,靠 P1 扫描定 L_start**。
- 跨帧训练无关是否可行 **未被文献证实**(TempMe 需 LoRA)——只当假设,用我们数据验。
- 端到端加速**依赖 predictor 改造**;不改 predictor、只在 encoder 不 restore,也能省 encoder 后段 + 显存,但下游仍按 dense。
- K 的 hook 不能意外关掉 fused SDPA(我们只读 k、照常调 SDPA,应安全,需实测确认)。
- 这套结论目前是 tiny-cache / encoder 级;**P3 前不声称完整 benchmark 加速**。

---

## 7. 与上一轮 B/C 失败的关系

上一轮 B/C 失败根因是 **importance proxy 不准**(norm/motion/qk_global_hidden)。本方法论**刻意避开 importance proxy 路线**,改用 ToMe 实证有效、且训练无关的三件套:**K 相似度 + BSM 全局匹配 + 渐进多层调度**;精度护栏用**零成本的 size 加权**,而不是昂贵且 SDPA 冲突的 attention 机制。
