"""Quick manual check: run the pipeline on a few hundred Dolly rows and print what it did."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

from common import dolly_slice
from sftforge import pipeline
from sftforge.loaders import load_lines
from sftforge.schema import Config

n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
rows = dolly_slice("dev")[:n]
df = load_lines(json.dumps(r) for r in rows).df
t = time.time()
res = pipeline.run(df, Config())
print("time", round(time.time() - t, 1))
for g in res.audit["gates"]:
    print(g["gate"], g["in"], g["rejected"], g["reasons"])
print(res.audit["timings_s"])
rej = res.df[res.df["status"] == "rejected"]
for _, r in rej.head(25).iterrows():
    print("-", r["id"], r["gate"], [x["c"] + ":" + x["m"] for x in r["reasons"]], "|", r["instruction"][:60].replace("\n", " "), "=>", r["response"][:60].replace("\n", " "))
print(Counter(len(x) > 0 for x in res.df["redactions"]))
