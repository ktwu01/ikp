"""Inspect smoke-test results manually.

Reads data/results_v2/<model>.json for the 4 smoke-test models and prints:
  - Per-tier verdict counts (STRONG/WEAK/WRONG/REFUSAL)
  - Sample CORRECT_STRONG, CORRECT_WEAK, WRONG verdicts to spot-check
  - Counts by no_cs_match probes
  - Sanity: Pro should have more STRONG than Flash; Flash should have more WRONG
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results_v2"

MODELS = ["deepseek-v4-pro", "deepseek-v4-flash",
          "gemini-3.1-pro", "gemini-3-flash"]


def main():
    data = {}
    for m in MODELS:
        f = RESULTS / f"{m}.json"
        if not f.exists():
            print(f"  [MISSING] {f}")
            continue
        with open(f) as fp:
            data[m] = json.load(fp)

    # Per-tier verdict counts
    print("=== Per-tier verdict counts ===")
    for m, d in data.items():
        print(f"\n  {m}:  overall_score={d['overall_score']:.3f}")
        ts = d["tier_stats"]
        print(f"    {'tier':4s}  {'STRONG':>7s}  {'WEAK':>5s}  {'REFUSAL':>8s}  {'WRONG':>5s}  {'no_cs ok':>9s}  {'total':>5s}")
        for t in ["T3", "T4", "T5", "T6", "T7"]:
            s = ts.get(t, {})
            print(f"    {t}      {s.get('strong',0):5d}  {s.get('weak',0):5d}  "
                  f"{s.get('refusal',0):7d}  {s.get('wrong',0):5d}  "
                  f"{s.get('no_cs_match_correct',0):9d}  {s.get('total',0):5d}")

    # Compare Pro vs Flash STRONG/WRONG ratios
    print("\n=== Pro vs Flash bluff diagnostics ===")
    for pair in [("deepseek-v4-pro", "deepseek-v4-flash"),
                 ("gemini-3.1-pro", "gemini-3-flash")]:
        p, f = pair
        if p not in data or f not in data: continue
        p_strong = sum(s.get("strong", 0) for s in data[p]["tier_stats"].values())
        f_strong = sum(s.get("strong", 0) for s in data[f]["tier_stats"].values())
        p_wrong = sum(s.get("wrong", 0) for s in data[p]["tier_stats"].values())
        f_wrong = sum(s.get("wrong", 0) for s in data[f]["tier_stats"].values())
        p_weak = sum(s.get("weak", 0) for s in data[p]["tier_stats"].values())
        f_weak = sum(s.get("weak", 0) for s in data[f]["tier_stats"].values())
        print(f"\n  {p} vs {f}:")
        print(f"    STRONG:   Pro={p_strong}, Flash={f_strong}")
        print(f"    WEAK:     Pro={p_weak}, Flash={f_weak}")
        print(f"    WRONG:    Pro={p_wrong}, Flash={f_wrong}  (bluff ratio: {f_wrong/max(p_wrong,1):.2f}x)")
        print(f"    Score:    Pro={data[p]['overall_score']:.3f}, Flash={data[f]['overall_score']:.3f}")

    # Sample for manual inspection
    print("\n=== Sample STRONG verdicts (V4 Flash, T6) ===")
    if "deepseek-v4-flash" in data:
        for r in data["deepseek-v4-flash"]["results"]:
            if r["tier"] == "T6" and r["verdict"] == "CORRECT_STRONG":
                print(f"  [{r['probe_id']}] {r.get('researcher_name','?')}")
                print(f"    response: {r['model_response'][:200]}")

    print("\n=== Sample WEAK verdicts (V4 Flash, T6) — bluffers should land here ===")
    if "deepseek-v4-flash" in data:
        n = 0
        for r in data["deepseek-v4-flash"]["results"]:
            if r["tier"] == "T6" and r["verdict"] == "CORRECT_WEAK":
                print(f"  [{r['probe_id']}] {r.get('researcher_name','?')}")
                print(f"    response: {r['model_response'][:200]}")
                n += 1
                if n >= 3: break

    print("\n=== Sample WRONG verdicts (V4 Flash, any tier) ===")
    if "deepseek-v4-flash" in data:
        n = 0
        for r in data["deepseek-v4-flash"]["results"]:
            if r["verdict"] == "WRONG":
                print(f"  [{r['probe_id']} {r['tier']}] {r.get('researcher_name','?')}")
                print(f"    response: {r['model_response'][:200]}")
                n += 1
                if n >= 4: break

    # no_cs_match cases
    print("\n=== no_cs_match probes — model behavior ===")
    for m, d in data.items():
        nm_stats = Counter()
        for r in d["results"]:
            if r.get("no_cs_match"):
                nm_stats[r["verdict"]] += 1
        print(f"  {m}: {dict(nm_stats)}")


if __name__ == "__main__":
    main()
