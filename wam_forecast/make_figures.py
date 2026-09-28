"""Key figures for the Phase A/B report (reads the JSON outputs of phaseA_analysis / phaseB_analysis).

  python make_figures.py      -> runs/figures/*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import W  # noqa: E402

R = W / "runs"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CAPS = {"capc1pa", "capc2pa", "capc4pa", "caps2pa", "caps4pa", "caps8pa"}

# 1. Gini vs dADE (test), one panel per budget
corr = json.loads((R / "phaseA" / "corr_test.json").read_text())
corr = {b: corr[b] for b in ("s25", "L9", "L12") if b in corr}
fig, ax = plt.subplots(1, len(corr), figsize=(4.2 * len(corr), 3.6))
for a, (b, v) in zip(np.atleast_1d(ax), corr.items()):
    for r in v["rows"]:
        col = "tab:blue" if r["arm"] in CAPS else ("k" if r["arm"] == "kbsmpa" else "tab:orange")
        a.scatter(r["gini"], r["d"], c=col, s=28)
        a.annotate("kbsmpa (ref)" if r["arm"] == "kbsmpa" else r["arm"][:-2] if r["arm"].endswith("pa") else r["arm"], (r["gini"], r["d"]), fontsize=6, xytext=(2, 2),
                   textcoords="offset points")
    st = v["stats"]["gini"]
    a.axhline(0, c="gray", lw=0.6)
    a.set_title(f"{b}: Spearman {st['rho']:+.2f} (p={st['p']:.3f})", fontsize=9)
    a.set_xlabel("group-size Gini (train clips)")
    a.set_ylabel("ADE - kbsmpa (mm, test)")
fig.suptitle("Group-size concentration vs forecasting error (blue: capacity arms, orange: other PA arms)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "gini_vs_dade_test.png", dpi=160)

# 2. caps2pa - kbsmpa on three disjoint evaluation sets
sel = json.loads((R / "phaseA" / "select_val.json").read_text())
cvr = json.loads((R / "phaseA" / "cv_caps2pa_vs_kbsmpa.json").read_text())
test = {  # A2 (pairs.py output, recorded in PROTOCOL.md)
    "s25": (-0.687, -1.752, 0.350), "L9": (-1.380, -2.659, -0.086), "L12": (-1.500, -3.009, 0.087)}
fig, a = plt.subplots(figsize=(6, 3.4))
xs = {"s25": 0, "L9": 1, "L12": 2}
for k, (lab, off, col) in enumerate((("val (180 clips, 10 seeds)", -0.2, "tab:gray"),
                                      ("test (200 clips, 20 seeds)", 0.0, "tab:red"),
                                      ("CV (1620 clips, 4 seeds)", 0.2, "tab:blue"))):
    for b, x in xs.items():
        if k == 0:
            v = sel.get(f"caps2pa_{b}")
            if not v:
                continue
            m, lo, hi = v["d"], v["clip_ci"][0], v["clip_ci"][1]
        elif k == 1:
            m, lo, hi = test[b]
        else:
            v = cvr[b]; m, lo, hi = v["d"], v["clip_ci"][0], v["clip_ci"][1]
        a.errorbar(x + off, m, yerr=[[m - lo], [hi - m]], fmt="o", c=col, capsize=3,
                   label=lab if b == "L9" else None)
a.axhline(0, c="k", lw=0.6)
a.set_xticks(list(xs.values()), [f"{b}\n({t} tok)" for b, t in (("s25", 972), ("L9", 309), ("L12", 131))])
a.set_ylabel("ADE caps2pa - kbsmpa (mm)\nclip x seed 95% CI")
a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "caps2pa_three_sets.png", dpi=160)

# 3. Phase B: VG and CG per setting (val)
pb = json.loads((R / "phaseB" / "phaseB_val.json").read_text())
names = [s for s in ("P32", "P4", "P1", "P0") if s in pb]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.arange(len(names))
for j, st in enumerate(("all", "motion_q4(high)")):
    a = ax[j]
    vg = [pb[n][st]["VG"]["d"] for n in names]
    a.bar(x - 0.27, vg, 0.25, label="vision gain (ref - dense)", color="tab:green")
    for i, (c, col) in enumerate((("kbsmpa_L9", "tab:blue"), ("kbsmpa_L12", "tab:purple"))):
        cg = [pb[n][st].get(f"CG_{c}", {}) or {} for n in names]
        a.bar(x + (i * 0.27), [v.get("d", np.nan) for v in cg], 0.25, label=f"compression gap {c}", color=col)
    a.axhline(0, c="k", lw=0.6)
    a.set_xticks(x, names)
    a.set_title(f"clips: {st}", fontsize=9)
    a.set_ylabel("mm (validation)")
ax[0].legend(fontsize=7)
fig.suptitle("Vision dependence: P32/P4/P1 = pose history frames, P0 = no pose (ref = copy-last)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "phaseB_vg_cg.png", dpi=160)

# 4. per-step curves
fig, a = plt.subplots(figsize=(6, 3.4))
for n, col in zip(names, ("tab:gray", "tab:green", "tab:orange", "tab:red")):
    if "VG_per_step" in pb[n]:
        a.plot(np.arange(1, 33), pb[n]["VG_per_step"], c=col, label=f"{n} vision gain")
    if "CG_per_step_kbsmpa_L12" in pb[n]:
        a.plot(np.arange(1, 33), pb[n]["CG_per_step_kbsmpa_L12"], c=col, ls="--", label=f"{n} CG L12")
a.axhline(0, c="k", lw=0.6)
a.set_xlabel("future step (of 32)")
a.set_ylabel("mm (validation)")
a.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "phaseB_per_step.png", dpi=160)
print("figures ->", OUT)

# 5. original task (vision only) vs P1 (video + current pose), test: dense / kbsmpa / caps2pa per budget
from pairs import load  # noqa: E402


def mean_ade(d):
    v = load(R / "test_main", d)
    return float(np.mean([x[0] for x in v.values()])) if v else np.nan


fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, (title, sfx, dense_dir, base, base_lab) in zip(ax, (
        ("original task (vision only)", "vis", "dense/vis", 68.37, "copy-last"),
        ("P1: video + current hand pose", "vispose_k1", "dense/vispose_k1", 68.37, "copy-last"))):
    bud = ["s25", "L9", "L12"]
    x = np.arange(3)
    k = [mean_ade(f"kbsmpa_{b}/{sfx}") for b in bud]
    c = [mean_ade(f"caps2pa_{b}/{sfx}") for b in bud]
    dn = mean_ade(dense_dir)
    a.plot(x, k, "o-", label="K-BSM + PA (corrected baseline)")
    a.plot(x, c, "s-", label="caps2pa (balanced groups)")
    a.axhline(dn, c="k", ls="--", lw=1, label=f"dense ({dn:.1f})")
    a.set_xticks(x, ["s25\n972 tok", "L9\n309 tok", "L12\n131 tok"])
    a.set_title(f"{title}; {base_lab} = {base:.1f}", fontsize=8)
    a.set_ylabel("test ADE (mm)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "headroom_original_vs_P1.png", dpi=160)
print("figure 5 done")
