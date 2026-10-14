"""C1 / C2 / D1 analysis as fixed in PROTOCOL.md -> results/v_summary.txt (works on partial results too)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
B = 10000
out = []


def say(*a):
    out.append(" ".join(str(x) for x in a))


def boot(d, groups=None):
    """Mean and 95% bootstrap interval of d [N]; groups [N] resamples clusters (episodes) instead of rows."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    if groups is None:
        bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    else:
        g = np.asarray(groups)
        u = np.unique(g)
        sums = np.array([d[g == x].sum() for x in u])
        cnt = np.array([(g == x).sum() for x in u])
        idx = rng.integers(0, len(u), (B, len(u)))
        bs = sums[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi


def fmt(m, lo, hi, scale=1.0):
    return f"{m * scale:+.3f} [{lo * scale:+.3f}, {hi * scale:+.3f}]"


# ------------------------------------------------------------------------------------------------ C1 / C2 (closed loop)
def load_sim(res):
    arms = defaultdict(dict)
    for f in glob.glob(os.path.join(R, res, "*.jsonl")):
        tag_arm = os.path.basename(f).rsplit("_e", 1)[0]
        for line in open(f):
            r = json.loads(line)
            arms[tag_arm][r["episode"]] = r
    return arms


def sim_section(res, title):
    A = load_sim(res)
    if not A:
        return
    say(f"== {title} ({res})")
    say(f"{'arm':24s} {'eps':>4s} {'success':>8s} {'final cm':>9s} {'plan s':>7s}")
    for k in sorted(A):
        rs = list(A[k].values())
        say(f"{k:24s} {len(rs):4d} {np.mean([r['success'] for r in rs]):8.2f} "
            f"{100 * np.mean([r['final_dist'] for r in rs]):9.2f} "
            f"{np.mean([s['t_plan'] for r in rs for s in r['steps']]):7.2f}")

    def pair(a, b, label, seeds=None):
        """a, b: arm keys, or lists of arm keys (seed pairs averaged per episode)."""
        a, b = ([a], [b]) if isinstance(a, str) else (a, b)
        if not all(x in A for x in a + b):
            say(f"{label}: missing arms")
            return None
        eps = sorted(set.intersection(*[set(A[x]) for x in a + b]))
        if not eps:
            say(f"{label}: no common episodes")
            return None
        res_ = {}
        for key, sc in (("success", 1.0), ("final_dist", 100.0)):
            d = [np.mean([float(A[x][e][key]) for x in a]) - np.mean([float(A[x][e][key]) for x in b]) for e in eps]
            res_[key] = boot(d)
        say(f"{label} (n = {len(eps)} episodes{', ' + str(len(a)) + ' seed pairs' if len(a) > 1 else ''}): "
            f"success {fmt(*res_['success'])}, final cm {fmt(*res_['final_dist'], scale=100)}")
        return res_

    if res == "c1":
        ks = [k for k in range(3) if f"s4k{k}_stack_sim" in A and f"s3bk{k}_few28" in A]
        say("-- H1 (primary): METHOD (S4 k, sim n=64, 102 samples) - CONTROL (S3b k, 256 tokens, 28 samples)")
        h1 = pair([f"s4k{k}_stack_sim" for k in ks], [f"s3bk{k}_few28" for k in ks], f"H1 pooled seeds {ks}") if ks else None
        for k in ks:
            pair(f"s4k{k}_stack_sim", f"s3bk{k}_few28", f"   seed pair {k}")
        say("-- H2: METHOD k0 - S3b k0 with 102 samples (non inferiority margin -0.10 success)")
        h2 = pair("s4k0_stack_sim", "s3bk0_few102", "H2")
        say("-- H5 (addendum): METHOD k0 - same merging with the plain (S3b k0) predictor")
        h5 = pair("s4k0_stack_sim", "s3bk0_stack_sim", "H5")
        if h5:
            say(f"VERDICT H5: {'holds' if h5['success'][1] > 0 else 'not shown'}")
        say("-- secondary")
        pair("s4k0_stack_pool", "s3bk0_few28", "pool n=64 102 (S4 k0) - CONTROL k0")
        pair("s4k0_sim16", "s4k0_stack_sim", "sim n=16 400 (merging alone) - METHOD k0")
        pair("s3_few28", "s3bk0_few28", "S3 28 samples - S3b k0 28 samples (extra plain training)")
        pair("s3bk0_few102", "s3bk0_few28", "S3b k0: 102 - 28 samples")
        if h1:
            say(f"VERDICT H1: {'holds' if h1['success'][1] > 0 else 'not shown'} (success lower bound "
                f"{h1['success'][1]:+.3f}); H1b: {'holds' if h1['final_dist'][2] < 0 else 'not shown'}")
        if h2:
            say(f"VERDICT H2: {'holds' if h2['success'][1] > -0.10 else 'not shown'} (lower bound {h2['success'][1]:+.3f})")
    if res == "c3":
        ks = [k for k in range(3) if f"s4k{k}_stack51_sim" in A and f"s3bk{k}_few14" in A]
        say("-- H6 (primary for C3): METHOD3 (S4 k, sim n=64, 51 samples) - CONTROL3 (S3b k, 14 samples)")
        h6 = pair([f"s4k{k}_stack51_sim" for k in ks], [f"s3bk{k}_few14" for k in ks], f"H6 pooled seeds {ks}") if ks else None
        for k in ks:
            pair(f"s4k{k}_stack51_sim", f"s3bk{k}_few14", f"   seed pair {k}")
        h6b = pair("s4k0_stack51_sim", "s3bk0_few28c5", "H6b: METHOD3 k0 - 28 samples x 5 iterations k0")
        kb = [k for k in range(3) if f"s4k{k}_stack51_sim" in A and f"s3bk{k}_few28c5" in A]
        if len(kb) > 1:
            pair([f"s4k{k}_stack51_sim" for k in kb], [f"s3bk{k}_few28c5" for k in kb], f"H6b pooled seeds {kb}")
        say("-- secondary")
        pair("s4k0_sim16s200", "s4k0_stack51_sim", "sim n=16 200 - METHOD3 k0")
        pair("s3bk0_few28c5", "s3bk0_few14", "baselines: 28 x 5 - 14 x 10")
        if h6 and h6b:
            say(f"VERDICT H6: {'holds' if h6['success'][1] > 0 and h6b['success'][1] > 0 else 'not shown'}")
    if res == "c4":
        a = pair("s4k0_stack26_sim", "s3bk0_few14c5", "H7a: METHOD4 - 14 samples x 5 iterations")
        b = pair("s4k0_stack26_sim", "s3bk0_few28c2", "H7b: METHOD4 - 28 samples x 2 iterations")
        for base in ("few14c5", "few28c2"):
            kb = [k for k in range(3) if f"s4k{k}_stack26_sim" in A and f"s3bk{k}_{base}" in A]
            if len(kb) > 1:
                pair([f"s4k{k}_stack26_sim" for k in kb], [f"s3bk{k}_{base}" for k in kb], f"H7 vs {base} pooled seeds {kb}")
        pair("s4k0_sim16s100", "s4k0_stack26_sim", "secondary: sim n=16 100 - METHOD4")
        if a and b:
            say(f"VERDICT H7: {'holds' if a['success'][1] > 0 and b['success'][1] > 0 else 'not shown'}")
        say("-- H8 (addendum): METHOD4 - baselines at equal wall clock (T3), pooled over seed pairs")
        h8 = []
        for base in ("few28c4", "few14c7"):
            kb = [k for k in range(3) if f"s4k{k}_stack26_sim" in A and f"s3bk{k}_{base}" in A]
            if kb:
                h8.append(pair([f"s4k{k}_stack26_sim" for k in kb], [f"s3bk{k}_{base}" for k in kb],
                               f"H8 vs {base} pooled seeds {kb}"))
                for k in kb:
                    pair(f"s4k{k}_stack26_sim", f"s3bk{k}_{base}", f"   seed pair {k}")
        if len(h8) == 2 and all(h8):
            say(f"VERDICT H8: {'holds' if all(h['success'][1] > 0 for h in h8) else 'not shown'}")
    if res == "c2":
        h = pair("s4k0_stack_sim", "s3bk0_few28", "C2: METHOD k0 - CONTROL k0")
        if h:
            say(f"VERDICT C2: {'holds' if h['success'][1] > 0 else 'not shown'}")
    say("")


# ----------------------------------------------------------------------------------------------------------- D1
def d1_section():
    rows = {}
    for f in glob.glob(os.path.join(R, "d1", "eval_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[r["i"]] = r
    vf = os.path.join(R, "d1_val.json")
    if os.path.exists(vf):
        v = json.load(open(vf))
        say("== D1 validation loss (step: loss); chosen", v["chosen"])
        for k, d in v["loss"].items():
            say(f"   {k}: " + ", ".join(f"{s}: {x:.4f}" for s, x in d.items()))
    if not rows:
        return
    rs = [rows[i] for i in sorted(rows)]
    arms = list(rs[0]["arms"])
    say(f"== D1 test (real DROID, {len(rs)} transitions from {len({r['ep'] for r in rs})} episodes): cosine of plan")
    for a in arms:
        say(f"{a:16s} cos {np.mean([r['arms'][a]['cos'] for r in rs]):.3f}  plan s "
            f"{np.mean([r['arms'][a]['t'] for r in rs]):.2f}")
    ep = [r["ep"] for r in rs]

    def pair(a, b, label):
        d = [r["arms"][a]["cos"] - r["arms"][b]["cos"] for r in rs]
        t = boot(d)
        c = boot(d, ep)
        say(f"{label}: {fmt(*t)}   (episode cluster bootstrap {fmt(*c)})")
        return t

    say("-- H3 (primary)")
    h3a = pair("dm_sim64s102", "dp_few28", "METHOD - DP few28")
    h3b = pair("dm_sim64s102", "off_few28", "METHOD - official few28")
    say("-- H4")
    h4 = pair("dm_sim64s102", "dp_few102", "METHOD - DP few102")
    say("-- secondary")
    pair("dm_sim64s102", "off_sim64s102", "METHOD - official sim n=64 102 (frozen merging)")
    pair("dm_pool64s102", "dp_few28", "DM pool n=64 102 - DP few28")
    pair("dm_sim16", "dm_sim64s102", "DM sim n=16 400 - METHOD")
    pair("dp_few102", "dp_few28", "DP 102 - 28 samples")
    pair("dp_few28", "off_few28", "DP few28 - official few28")
    say(f"VERDICT H3: {'holds' if h3a[1] > 0 and h3b[1] > 0 else 'not shown'}; "
        f"H4: {'holds' if h4[1] > -0.10 else 'not shown'} (lower bound {h4[1]:+.3f})")
    say("")


sim_section("c1", "C1 closed loop, fresh episodes 2000-2099")
d1_section()
sim_section("c3", "C3 tighter budget (about 10.8k tokens per CEM iteration), episodes 4000-4099")
sim_section("c4", "C4 tighter still (about 5.4k tokens per CEM iteration), episodes 5000-5099")
sim_section("c2", "C2 harder reach (20-30 cm, 15 steps)")
open(os.path.join(R, "v_summary.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
