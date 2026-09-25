import json
B = "outputs/phase1_motiv_20260601"
c5 = json.load(open(f"{B}/c5_attn/attn_entropy_stats.json"))
print("=== C5 redundancy-by-depth (lower PR/effrank = more redundant) ===")
for L, v in c5["per_layer"].items():
    r = v["rank_proxy"]
    print(f"  L{int(L):>2}: PR={r['participation_ratio']:6.2f}  effrank={r['effective_rank']:6.1f}  nc95={r['ncomp_95pct']:>3}  nc99={r['ncomp_99pct']:>3}")
c2 = json.load(open(f"{B}/c2_sim/similarity_heatmap_stats.json"))
print("\n=== C2 top-level keys ===", list(c2.keys()))
pl = c2.get("per_layer", c2.get("by_depth", {}))
if isinstance(pl, dict) and pl:
    k0 = list(pl.keys())[0]
    print("C2 per-layer group keys:", list(pl[k0].keys()) if isinstance(pl[k0], dict) else type(pl[k0]))
    for L, v in pl.items():
        if isinstance(v, dict):
            out = {}
            for g, x in v.items():
                if isinstance(x, dict):
                    out[g] = round(x.get("mean", x.get("cos_mean", 0)), 3)
                elif isinstance(x, (int, float)):
                    out[g] = round(x, 3)
            print(f"  L{L}: {out}")
