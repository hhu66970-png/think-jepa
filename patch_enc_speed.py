#!/usr/bin/env python
"""给 tools/enc_speed.py 加 E1 frontier 缺的工作点:r0.20(2686)/r0.25(1944)/L12_r25(261)。"""
p = "tools/enc_speed.py"
s = open(p).read()
if "L12 = list(range(10, 22))" not in s:
    s = s.replace("GR5 = [12, 14, 16, 18, 20]; L9 = list(range(12, 21))",
                  "GR5 = [12, 14, 16, 18, 20]; L9 = list(range(12, 21)); L12 = list(range(10, 22))")
anchor = '        ("wam_L9_r25",   dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=L9, ratio=0.25, relevance=("motion", 1.0))),'
add = '''        ("kbsm_r20",     dict(enabled=True,  strategy="bsm_ksim_gradual_vec", layers=GR5, ratio=0.20)),
        ("pitome_r20",   dict(enabled=True,  strategy="bsm_pitome_gradual_vec", layers=GR5, ratio=0.20)),
        ("wam_r20",      dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=GR5, ratio=0.20, relevance=("motion", 1.0))),
        ("kbsm_r25",     dict(enabled=True,  strategy="bsm_ksim_gradual_vec", layers=GR5, ratio=0.25)),
        ("pitome_r25",   dict(enabled=True,  strategy="bsm_pitome_gradual_vec", layers=GR5, ratio=0.25)),
        ("wam_r25",      dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=GR5, ratio=0.25, relevance=("motion", 1.0))),
        ("kbsm_L12_r25", dict(enabled=True,  strategy="bsm_ksim_gradual_vec", layers=L12, ratio=0.25)),
        ("pitome_L12_r25",dict(enabled=True, strategy="bsm_pitome_gradual_vec", layers=L12, ratio=0.25)),
        ("wam_L12_r25",  dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=L12, ratio=0.25, relevance=("motion", 1.0))),'''
if '("wam_r25",' not in s:
    s = s.replace(anchor, anchor + "\n" + add)
open(p, "w").write(s)
print("patched enc_speed.py; configs now:")
import re
for m in re.findall(r'\("(\w+)",\s+dict', s):
    print("  ", m)
