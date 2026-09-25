"""Figure A/B: temporal token-merge fidelity is far below spatial at matched token budget.
Recovers all numbers from recovered/outputs/**/pca_experiment_results.json (encoder-side PCA exp).
Figure C: Step0 pixel-level temporal redundancy (real values from 隔夜_20260531 delivery JSON).
Pure matplotlib (Agg), no GPU/torch. English labels to avoid CJK tofu.

NO fabricated numbers: every plotted point comes from a JSON field
(fidelity_mean_cosine_vs_dense / tokens_final / speedup_vs_dense), averaged over samples per tag.
"""
import json, glob, os, sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except Exception as e:
    print("NO_MPL", e); sys.exit(3)

PCA_GLOB = "/Users/huhaoming/Desktop/wm/tmp/phase0/recovered/outputs/**/pca_experiment_results.json"
STEP0_AVG = "/Users/huhaoming/Desktop/wm/token_merge_交付包/实验结果数据/隔夜_20260531/Step0_时间冗余_均值.json"
OUT = "/Users/huhaoming/Desktop/wm/token_merge_交付包/实验结果数据/时序冗余_20260601"
DENSE_TOKENS = 8192
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- load & aggregate
agg = defaultdict(list)  # tag -> [(tok, cos, sp, group), ...] across all samples
files = sorted(glob.glob(PCA_GLOB, recursive=True))
for f in files:
    d = json.load(open(f))
    for clipid, s in d.get("samples", {}).items():
        for tag, mc in s.get("merge_configs", {}).items():
            cos = mc.get("fidelity_mean_cosine_vs_dense")
            tok = mc.get("tokens_final")
            sp = mc.get("speedup_vs_dense")
            grp = mc.get("group")
            if cos is None or tok is None:
                continue
            agg[tag].append((tok, cos, sp, grp))

def mean(xs): return sum(xs) / len(xs)
recs = []
for tag, lst in agg.items():
    recs.append(dict(group=lst[0][3], tag=tag,
                     tok=round(mean([x[0] for x in lst])),
                     cos=mean([x[1] for x in lst]),
                     sp=mean([x[2] for x in lst]),
                     n=len(lst)))

TEMP = sorted([r for r in recs if r["group"] == "pre_merge_temporal"], key=lambda r: -r["tok"])
SPAT_A = [r for r in recs if r["group"] == "A_scheme_a"]
SPAT_B = [r for r in recs if r["group"] == "bsm_ksim_gradual_vec"]
SPAT = SPAT_A + SPAT_B

assert TEMP, "no temporal configs found"
assert SPAT, "no spatial configs found"

# SPATIAL envelope = best (max) cos per token BUDGET (max over A_ref + BSM).
# Near-identical token counts (within TOL, e.g. 6144 vs 6143) are merged into one budget
# bin so the envelope is a true "best achievable at this budget" line, not an artifact
# of one isolated config sitting at a singleton token count.
TOL = 24
spat_sorted = sorted(SPAT, key=lambda r: -r["tok"])
bins = []  # list of (bin_anchor_tok, [recs])
for r in spat_sorted:
    if bins and abs(bins[-1][0] - r["tok"]) <= TOL:
        bins[-1][1].append(r)
    else:
        bins.append((r["tok"], [r]))
spat_env = []
for anchor, grp in bins:
    best = max(grp, key=lambda r: r["cos"])
    spat_env.append(best)  # keep the actual best config (its real tok & cos)

def red(tok): return 1.0 - tok / DENSE_TOKENS  # reduction ratio

# matched-token deltas (exact token match temporal-vs-spatial-best) for annotation
matched = {}
for t in TEMP:
    same = [r for r in SPAT if r["tok"] == t["tok"]]
    if same:
        b = max(same, key=lambda r: r["cos"])
        matched[t["tok"]] = (t, b)

print("=== TEMPORAL (pre_merge_temporal) ===")
for t in TEMP:
    print(f"  tok={t['tok']:5d} red={red(t['tok']):.3f} cos={t['cos']:.4f} sp={t['sp']:.3f}x n={t['n']} {t['tag']}")
print("=== SPATIAL envelope (max cos over A_ref+BSM per token) ===")
for r in spat_env:
    print(f"  tok={r['tok']:5d} red={red(r['tok']):.3f} cos={r['cos']:.4f} sp={r['sp']:.3f}x {r['tag']}")
print("=== matched-token (exact) temporal vs spatial-best ===")
for tok, (t, b) in sorted(matched.items(), reverse=True):
    print(f"  tok={tok}: TEMPORAL cos={t['cos']:.4f}  vs  SPATIAL-best cos={b['cos']:.4f} ({b['tag']})  Δ={b['cos']-t['cos']:.3f}")

# ================================================================ FIGURE A
fig, ax = plt.subplots(figsize=(9.2, 6.2))
tx = [t["tok"] for t in TEMP]; ty = [t["cos"] for t in TEMP]
sx = [r["tok"] for r in spat_env]; sy = [r["cos"] for r in spat_env]
ax.plot(tx, ty, "X-", color="#d62728", ms=11, lw=2.4, zorder=4,
        label="TEMPORAL merge (RLT-style pre_merge_temporal)")
ax.plot(sx, sy, "o-", color="#1f77b4", ms=8, lw=2.2, zorder=3,
        label="SPATIAL merge (best of scheme-A / BSM @ same #tokens)")

# faint scatter of every individual spatial config (shows envelope is a real max)
ax.scatter([r["tok"] for r in SPAT_A], [r["cos"] for r in SPAT_A], c="#9ecae1",
           s=18, alpha=0.55, zorder=1, label="scheme-A configs (all layers/ratios)")
ax.scatter([r["tok"] for r in SPAT_B], [r["cos"] for r in SPAT_B], marker="^",
           c="#c6dbef", s=20, alpha=0.7, zorder=1, edgecolors="none", label="BSM configs (all)")

ax.axhline(0.90, color="darkgreen", ls="--", lw=1.3, alpha=0.8)
ax.text(4150, 0.905, "cos=0.90 fidelity guardrail", color="darkgreen", fontsize=8.5, va="bottom")

# headline annotation at 7168 tokens: temporal 0.84 vs spatial best 0.983 (A@L20),
# also explicitly mark the representative spatial point A@L16 r0.125 = 0.975.
L16_7168 = next((r for r in SPAT_A if r["tag"] == "A_ref__L16__r0.125"), None)
if 7168 in matched:
    t7, b7 = matched[7168]
    ax.annotate("", xy=(7168, b7["cos"]), xytext=(7168, t7["cos"]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.6), zorder=6)
    ax.scatter([7168, 7168], [t7["cos"], b7["cos"]], c=["#d62728", "#1f77b4"],
               s=90, zorder=7, edgecolors="k", linewidths=0.7)
    l16txt = f"\n(A@L16 r0.125 = {L16_7168['cos']:.3f})" if L16_7168 else ""
    if L16_7168:
        ax.scatter([7168], [L16_7168["cos"]], marker="D", c="#6baed6",
                   s=55, zorder=7, edgecolors="k", linewidths=0.5)
    ax.annotate(
            f"@7168 tokens (1/8 merged):\nspatial best cos={b7['cos']:.3f}{l16txt}\ntemporal cos={t7['cos']:.3f}\nΔ = {b7['cos']-t7['cos']:.3f}",
            xy=(7168, (t7["cos"] + b7["cos"]) / 2), xytext=(6700, 0.70),
            ha="left", va="center", fontsize=9.3,
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff4e6", ec="orange", alpha=0.95), zorder=8)

# secondary annotation at 6144 tokens (0.727 vs 0.952)
if 6144 in matched:
    t6, b6 = matched[6144]
    ax.annotate("", xy=(6144, b6["cos"]), xytext=(6144, t6["cos"]),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2), zorder=5)
    ax.text(6144, t6["cos"] - 0.018,
            f"@6144 tok: {b6['cos']:.2f} vs {t6['cos']:.2f}",
            ha="center", va="top", fontsize=8.5, color="dimgray", zorder=6)

ax.set_xlabel("tokens after merge  (dense = 8192;  fewer = more aggressive merge)", fontsize=11)
ax.set_ylabel("cos_vs_dense  (encoder representation fidelity)", fontsize=11)
ax.set_title("Temporal token-merge is far less faithful than spatial at the SAME token budget\n"
             "V-JEPA2 ViT-L encoder, 3-video avg, PCA-aligned cosine vs dense",
             fontsize=12.5)
ax.invert_xaxis()  # left = aggressive merge, matches "reduction increases leftward"
ax.set_ylim(0.55, 1.005)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.6, framealpha=0.95)

# top secondary axis: reduction ratio
def tok2red(x): return 1.0 - x / DENSE_TOKENS
def red2tok(x): return (1.0 - x) * DENSE_TOKENS
secax = ax.secondary_xaxis("top", functions=(tok2red, red2tok))
secax.set_xlabel("token reduction ratio (1 - tokens/8192)", fontsize=9.5)

plt.tight_layout()
pA = os.path.join(OUT, "图A_时间vs空间合并_同token保真度.png")
plt.savefig(pA, dpi=135, bbox_inches="tight")
plt.close(fig)
print("SAVED", pA)

# ================================================================ FIGURE B (speed-fidelity)
fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.scatter([r["sp"] for r in SPAT_A], [r["cos"] for r in SPAT_A], marker="o",
           c="#1f77b4", s=55, alpha=0.85, edgecolors="k", linewidths=0.3,
           zorder=3, label="SPATIAL scheme-A")
ax.scatter([r["sp"] for r in SPAT_B], [r["cos"] for r in SPAT_B], marker="^",
           c="#2ca02c", s=60, alpha=0.85, edgecolors="k", linewidths=0.3,
           zorder=3, label="SPATIAL BSM (gradual K-sim)")
ax.scatter([t["sp"] for t in TEMP], [t["cos"] for t in TEMP], marker="X",
           c="#d62728", s=130, edgecolors="k", linewidths=0.6,
           zorder=4, label="TEMPORAL (pre_merge_temporal)")
# connect temporal points to show its frontier sits below/right
to = sorted(TEMP, key=lambda r: r["sp"])
ax.plot([t["sp"] for t in to], [t["cos"] for t in to], "--", color="#d62728", lw=1.2, alpha=0.7, zorder=2)

ax.axhline(0.90, color="darkgreen", ls="--", lw=1.2, alpha=0.8)
ax.text(ax.get_xlim()[1], 0.905, " cos=0.90 guardrail", color="darkgreen",
        fontsize=8.5, va="bottom", ha="right")
ax.set_xlabel("encoder speedup  (x vs dense)", fontsize=11)
ax.set_ylabel("cos_vs_dense  (representation fidelity)", fontsize=11)
ax.set_title("Speed-Fidelity frontier: temporal merge is on the inferior side\n"
             "(for any given speedup, temporal loses much more fidelity than spatial)",
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
plt.tight_layout()
pB = os.path.join(OUT, "图B_速度保真前沿_时间vs空间.png")
plt.savefig(pB, dpi=135, bbox_inches="tight")
plt.close(fig)
print("SAVED", pB)

# ================================================================ FIGURE C (Step0 pixel redundancy)
pC = None
if os.path.exists(STEP0_AVG):
    s0 = json.load(open(STEP0_AVG))
    taus = [2, 4, 8, 16, 32]
    fk = [s0.get(f"frame_static@{t}") for t in taus]
    tk = [s0.get(f"tubelet_static@{t}") for t in taus]
    if all(v is not None for v in fk + tk):
        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        ax.plot(taus, fk, "o-", color="#ff7f0e", lw=2.2, ms=9, label="frame-level (adjacent frames)")
        ax.plot(taus, tk, "s-", color="#8c564b", lw=2.2, ms=8, label="tubelet-level (adjacent tubelets)")
        ax.set_ylim(0.75, 1.01)
        ax.set_xlabel("threshold tau  (mean abs pixel diff per patch, 0-255 scale)", fontsize=11)
        ax.set_ylabel("near-static patch fraction\n(RLT run-length reduction upper bound)", fontsize=11)
        ax.set_title("Step0: PIXEL-level temporal redundancy is high (static-camera clips)\n"
                     "BUT high pixel redundancy != feature-mergeable  (see Figure A negative result)",
                     fontsize=11.5)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9.5)
        ax.annotate(f"tau=2: tubelet {tk[0]:.0%} 'static'\nyet temporal merge to 7168 tok\nalready drops cos to 0.84",
                    (2, tk[0]), xytext=(7.0, 0.80), fontsize=9,
                    arrowprops=dict(arrowstyle="->"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff4e6", ec="orange", alpha=0.9))
        plt.tight_layout()
        pC = os.path.join(OUT, "图C_像素级时间冗余_vs_阈值.png")
        plt.savefig(pC, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print("SAVED", pC)
    else:
        print("STEP0_FIELDS_MISSING", STEP0_AVG)
else:
    print("STEP0_NOT_FOUND", STEP0_AVG)

print("\nDONE. Files in:", OUT)
