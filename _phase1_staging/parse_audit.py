import json
B = "outputs/phase1_audit_20260601"
runs = ["dense_own", "kbsm_own", "kbsm_feats_dense_pred"]
data = {}
for r in runs:
    try:
        data[r] = json.load(open(f"{B}/{r}/temporal_metrics.json"))
    except Exception as e:
        print(r, "ERR", e)
if data:
    print("temporal_metrics keys:", list(next(iter(data.values())).keys()))
for r in runs:
    d = data.get(r, {})
    print(f"\n== {r} ==")
    for k, v in d.items():
        if isinstance(v, (int, float)):
            print(f"  {k} = {v:.5f}")
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            print(f"  {k} = first={v[0]:.4f} last={v[-1]:.4f} max={max(v):.4f} (len {len(v)})")
        elif isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={vv:.5f}" for kk, vv in v.items() if isinstance(vv, (int, float))))

def curve(d):
    for k in ("per_step_displacement", "per_step", "ade_curve", "displacement_per_step", "per_step_l2"):
        if k in d and isinstance(d[k], list):
            return d[k]
    return None
c0, c1 = curve(data.get("dense_own", {})), curve(data.get("kbsm_own", {}))
if c0 and c1 and len(c0) == len(c1):
    diffs = [abs(a - b) for a, b in zip(c0, c1)]
    print(f"\n[A1] per-step |dense - kbsm_own|: max={max(diffs):.5f} mean={sum(diffs)/len(diffs):.5f}")
    print("  dense per-step:", [round(x, 4) for x in c0])
    print("  kbsm  per-step:", [round(x, 4) for x in c1])
