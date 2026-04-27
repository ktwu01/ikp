#!/usr/bin/env python3
"""Show evaluation progress from saved result files."""

import json
from pathlib import Path
from collections import Counter

results_dir = Path("data/results")

results = []
for f in sorted(results_dir.glob("*.json")):
    if f.name in ["evaluation_summary.json", "analysis.json"]:
        continue
    try:
        d = json.load(open(f))
        if "model_name" in d:
            results.append(d)
    except:
        pass

results.sort(key=lambda d: (0 if d.get("params_B") else 1, d.get("params_B") or 0))

config = json.load(open("configs/all_models.json"))
total_models = len(config["models"])
print(f"Models evaluated: {len(results)}/{total_models}\n")
print(f"{'Model':30s} {'Params':>8s} {'Vendor':>10s} {'Agg':>6s}", end="")
for t in ["T1","T2","T3","T4","T5","T6","T7"]:
    print(f" {t:>5s}", end="")
print()
print("-" * 100)

for d in results:
    params = d.get("params_B")
    pstr = f"{params:.0f}B" if params else "?"
    vendor = (d.get("vendor") or "?")[:10]
    acc = d.get("accuracy", 0)
    ta = d.get("tier_accuracy", {})
    print(f"{d['model_name']:30s} {pstr:>8s} {vendor:>10s} {acc:6.1%}", end="")
    for t in ["T1","T2","T3","T4","T5","T6","T7"]:
        print(f" {ta.get(t, 0):5.0%}", end="")
    print()
