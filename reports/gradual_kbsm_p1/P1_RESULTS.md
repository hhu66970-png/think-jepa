# P1 结果:Gradual K-BSM token merge(速度优先)

**日期** 2026-05-29 · **分支** `refactor-and-exp` · **机器** A800-80GB · **策略** `bsm_ksim_gradual_vec`(诊断侧,未进主路径)
**原始结果** `outputs/gradual_kbsm_p1/{run_3layer,run_5layer}/`(gitignored)

## 方法(全部训练无关 + SDPA-safe)
全局 ToMe **Bipartite Soft Matching**(偶→A 源 / 奇→B 收,A→最相似 B 的 cosine,top-r 合并)+ **attention Key(K)相似度**(RoPEAttention 在 SDPA 调用前 stash `k.mean(head)`,**不改 SDPA、不物化 NxN**)+ **跨层渐进调度**(每层砍 r,layer 范围)+ **size 加权平均** + **默认不 restore_dense**(压缩态跑完 encoder)。多层压缩态的 rep 重映射用 identity-LUT,`restore_dense=True` 验证能精确还原回 8192。

## 基准结果(3 视频均值,8192 dense token,repeats=3)

| 配置 | token | 序列↓ | encoder | 加速 | cos vs dense | 95%维 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense baseline | 8192 | — | 606.0ms | 1.00× | 1.000 | 277 | 1671MB |
| 方案A L8 r0.125 | 7168 | 12.5% | 547.5ms | 1.11× | 0.894 | 274 | 1673MB |
| 方案A L8 r0.25 | 6144 | 25.0% | 490.8ms | 1.23× | 0.804 | 271 | 1685MB |
| **BSM L6-10 r0.092** | 6134 | 25.1% | 489.8ms | 1.24× | 0.813 | 270 | 1682MB |
| **BSM L4-12 r0.056** | 6143 | 25.0% | 488.7ms | **1.24×** | **0.816** | 270 | 1672MB |
| **BSM L6-10 r0.157** | 4908 | 40.1% | 419.0ms | **1.45×** | 0.732 | 264 | 1678MB |
| **BSM L4-12 r0.097** | 4920 | 39.9% | 419.7ms | **1.45×** | 0.736 | 264 | 1678MB |

逐层 token 轨迹(positional ToMe,确定性,3 视频一致;metric=key 每层):
```
BSM L4-12 r0.056 (~25%): 8192→7734→7301→6893→6507→6143
BSM L4-12 r0.097 (~40%): 8192→7398→6681→6033→5448→4920
```

## 结论(速度优先)
1. **同 ~25% 缩减下,BSM 严格优于方案A**:BSM L4-12 r0.056(488.7ms / 1.24× / cos 0.816)在**速度(~2ms)和保真(+0.013 cos)两个轴上都压过**单层方案A r0.25(490.8ms / 1.23× / cos 0.804)。
2. **BSM 解锁单层方案A 够不到的更快档**:~40% 缩减 → **1.45× 加速**(606→419ms),因为从 L4/L6 就开始砍,更深的层都在更短序列上跑。
3. **PCA 结构几乎不动**:95% 方差所需分量 270–264 vs dense 277,重建误差不变——merge 不破坏主成分结构。
4. **3 层 vs 5 层调度**:同缩减下 final token/加速相同,5 层渐进保真略高(removal 更平滑)。
5. **K 相似度确认 live**(metric=key、无 fallback,每层每视频)。

## 诚实的局限(不外推)
- ⚠️ **cos≥0.90 护栏:本次扫描的网格内没有任何配置达到**(最高是方案A r0.125 的 0.894,且只 1/3 视频过 0.90)。要 BSM 在 cos≥0.90,需要比本次更小的 per-layer r(更低总缩减)——**那个点还没跑**,不声称它达标。
- **峰值显存基本不变**(~1.67GB):8192 token 下激活相对权重很小,merge 省的是 **wall-clock 不是显存**(这个规模)。
- tiny-cache / encoder-only / fp32 / 单 A800;**P3(compressed tokens 进 predictor + 下游 ADE)前不声称完整 benchmark 加速**。

## 验证状态
- **A/B/C 逐位不变**(equivalence oracle,主路径未动);主路径拒绝 `bsm_ksim_gradual_vec`,仅 DiagnosticTokenMerger 可跑。
- **SDPA fused 完好**:24/24 attn 走 flash+mem-efficient,K-stash 对 attention 输出影响 = 0.0,SDPA 代码块 git diff 无变化。
- **加速是真 wall-clock**(warmup+cuda.synchronize+perf_counter;对抗复核独立复现 1.440×)。
- 对抗复核结论:**CONFIRMED — 无法证伪**。

## 下一步候选
- **P1b**:扫更小 per-layer r,找 BSM 在 cos≥0.90 的工作点(补上护栏内的速度点)。
- **P3**:让 predictor 吃 compressed tokens(不 restore)→ 量全链路加速 + 下游 ADE(把 encoder 加速兑现到端到端)。
- 可选:proportional attention(仅当护栏在更高缩减下被击穿;注意 SDPA 代价)。
