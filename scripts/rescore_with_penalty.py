"""Replay scoring offline against existing per-model result JSONs.

Re-applies the IKP scoring formula with a configurable hallucination penalty,
without re-querying any model. Uses the stored verdicts in
data/results/<model>.json (CORRECT / REFUSAL / WRONG).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def rescore(results, penalty):
    by_tier = defaultdict(lambda: {"correct": 0, "wrong": 0, "refusal": 0, "total": 0})
    for r in results:
        t = r["tier"]
        by_tier[t]["total"] += 1
        v = r["verdict"]
        if v == "CORRECT":
            by_tier[t]["correct"] += 1
        elif v == "WRONG":
            by_tier[t]["wrong"] += 1
        else:
            by_tier[t]["refusal"] += 1

    tier_accs = {}
    for t in TIERS:
        s = by_tier[t]
        if s["total"] > 0:
            score = (s["correct"] + penalty * s["wrong"]) / s["total"]
            tier_accs[t] = max(score, 0.0)
        else:
            tier_accs[t] = 0.0

    accuracy = sum(tier_accs.values()) / len(tier_accs)
    correct = sum(b["correct"] for b in by_tier.values())
    wrong = sum(b["wrong"] for b in by_tier.values())
    total = sum(b["total"] for b in by_tier.values())
    raw = correct / total if total else 0
    return accuracy, raw, tier_accs, correct, wrong, total


def main():
    results_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(results_dir))
    ap.add_argument("--penalties", default="-0.5,-1.0", help="Comma-separated penalties")
    ap.add_argument("--out", default=str(results_dir / "rescore_penalty_sweep.csv"))
    args = ap.parse_args()

    penalties = [float(x) for x in args.penalties.split(",")]
    rows = []
    skipped = 0
    files = sorted(Path(args.results_dir).glob("*.json"))
    for fp in files:
        if fp.name in {"analysis.json", "analysis_before_v4.json", "evaluation_summary.json"}:
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            skipped += 1
            continue
        if "results" not in d or not isinstance(d["results"], list):
            skipped += 1
            continue
        model = d.get("model_name") or fp.stem
        params_B = d.get("params_B")
        row = {
            "model": model,
            "params_B": params_B,
            "family": d.get("family"),
            "vendor": d.get("vendor"),
            "thinking": d.get("thinking", "?"),
        }
        for p in penalties:
            acc, raw, tier_accs, c, w, n = rescore(d["results"], p)
            tag = f"p{p:+.2f}".replace(".", "_").replace("+", "p").replace("-", "m")
            row[f"acc_{tag}"] = round(acc, 4)
            row[f"raw_{tag}"] = round(raw, 4)
            row[f"correct_{tag}"] = c
            row[f"wrong_{tag}"] = w
            row[f"total_{tag}"] = n
            for t, v in tier_accs.items():
                row[f"{t}_{tag}"] = round(v, 4)
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out} (skipped {skipped})")

    # Print key comparisons
    of_interest = {
        "deepseek-v4-flash", "deepseek-v4-pro",
        "deepseek-v4-flash-think", "deepseek-v4-pro-think",
        "gemini-3-flash", "gemini-3.1-pro",
        "gemini-3-flash-think", "gemini-3.1-flash-lite",
        "gemini-2.5-flash", "gemini-2.5-pro",
    }
    print(f"\n{'model':30s} {'acc@-0.5':>10s} {'acc@-1.0':>10s} {'Δ':>8s}  {'T6@-0.5':>9s} {'T6@-1.0':>9s}")
    for r in rows:
        if r["model"] in of_interest:
            a05 = r.get("acc_pm0_50", r.get("acc_p0_50"))
            a10 = r.get("acc_pm1_00", r.get("acc_p1_00"))
            t6_05 = r.get("T6_pm0_50")
            t6_10 = r.get("T6_pm1_00")
            d = (a10 - a05) if (a05 is not None and a10 is not None) else None
            print(f"  {r['model']:28s} {a05:10.4f} {a10:10.4f} {d:+8.4f}  {t6_05:9.4f} {t6_10:9.4f}")


if __name__ == "__main__":
    main()
