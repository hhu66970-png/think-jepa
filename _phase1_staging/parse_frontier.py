import json, glob, statistics as st
from collections import defaultdict
rows = []
for f in sorted(glob.glob("outputs/phase1_enc_20260601/*/pca_experiment_results.json")):
    cfg = f.split("/")[-2]
    try:
        d = json.load(open(f))
    except Exception as e:
        print("ERR", f, e); continue
    agg = defaultdict(lambda: {"cos": [], "tok": [], "spd": [], "metric": ""})
    dense_t = []
    for s in d.get("samples", {}).values():
        db = s.get("dense_baseline", {})
        if db.get("time_ms_median"): dense_t.append(db["time_ms_median"])
        mcs = s.get("merge_configs", {}) or {}
        for tag, mc in mcs.items():
            if isinstance(mc, dict) and mc.get("fidelity_mean_cosine_vs_dense") is not None:
                agg[tag]["cos"].append(mc["fidelity_mean_cosine_vs_dense"])
                agg[tag]["tok"].append(mc.get("tokens_final"))
                agg[tag]["spd"].append(mc.get("speedup_vs_dense"))
                agg[tag]["metric"] = mc.get("matching_metric", mc.get("bsm_match_metric", ""))
    for tag, v in agg.items():
        rows.append((cfg, tag, st.mean(v["tok"]), st.mean(v["cos"]), st.mean(v["spd"]), v["metric"]))
print(f"{'dir':14s} {'tag':46s} {'tokens':>7} {'cos':>7} {'speed':>6} metric")
for cfg, tag, tok, cos, spd, met in rows:
    print(f"{cfg:14s} {tag:46s} {tok:7.0f} {cos:7.4f} {spd:5.2f}x {met}")
