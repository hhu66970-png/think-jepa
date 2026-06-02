#!/usr/bin/env python
"""Parse all outputs/wam_frontier*/<cfg>/metrics.json -> real ADE(val_avg_dist)/FDE."""
import json, os, glob, sys

roots = sys.argv[1:] or ["outputs/wam_frontier"]
rows = []
for root in roots:
    for mj in sorted(glob.glob(root + "/*/metrics.json")):
        name = os.path.basename(os.path.dirname(mj))
        try:
            d = json.load(open(mj)); ep = d.get("epochs", [])
            if not ep:
                continue
            e = ep[-1]
            rows.append((name, e.get("val_avg_dist"), e.get("val_final_dist"),
                         e.get("time_sec")))
        except Exception as ex:
            rows.append((name, f"ERR:{ex}", "", ""))
rows.sort()
print(f"{'name':16s}\t{'ADE':>9}\t{'FDE':>9}\t{'t(s)':>6}")
for n, a, f, t in rows:
    try:
        print(f"{n:16s}\t{a:9.5f}\t{f:9.5f}\t{t:6.1f}")
    except Exception:
        print(f"{n:16s}\t{a}\t{f}\t{t}")
out = os.path.join(roots[0], "ADE_FDE.tsv")
with open(out, "w") as fh:
    fh.write("name\tADE\tFDE\n")
    for n, a, f, t in rows:
        fh.write(f"{n}\t{a}\t{f}\n")
print("saved", out)
