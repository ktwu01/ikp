"""Final assembly: merge researcher v2 verdicts with non-researcher v1 verdicts,
score every probe at lambda=-1, compute per-tier accuracy floored at 0,
take the mean across 7 tiers per model.

Per user spec:
  - All 200 questions per tier (1400 total).
  - Researcher probes (T3-T7, source=researcher): use data/results_v2/<model>.json
    with 4-way evidence judge (CORRECT_STRONG=+1, CORRECT_WEAK=+0.5,
    REFUSAL=0, WRONG=lambda).
  - Non-researcher probes: use data/results/<model>.json verdicts, score with
    same lambda (CORRECT=+1, REFUSAL=0, WRONG=lambda).
  - lambda = -1.0 across all probe types.
  - Per-tier score = max(sum_of_per_probe_scores / 200, 0).
  - Overall = mean of 7 floored per-tier scores.

Outputs:
  data/results/final_assembly.json  — one entry per model with:
    {model, params_B, family, vendor, accuracy, raw_accuracy, tier_accuracy,
     tier_stats, has_researcher_v2}
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_V1 = ROOT / "data" / "results"
RESULTS_V2 = ROOT / "data" / "results_v2"
CONFIG = ROOT / "configs" / "all_models.json"
OUT = ROOT / "data" / "results" / "final_assembly.json"

LAMBDA = -1.0
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def score_verdict(verdict: str, lam: float = LAMBDA) -> float:
    """Map any verdict (3-way or 4-way) to a per-probe score."""
    if verdict == "CORRECT_STRONG" or verdict == "CORRECT":
        return 1.0
    if verdict == "CORRECT_WEAK":
        return 0.5
    if verdict == "REFUSAL":
        return 0.0
    if verdict == "WRONG":
        return lam
    return 0.0  # ERROR or unknown


def assemble_one(model_name: str):
    """Compute the final assembled accuracy for one model.

    Returns dict or None if v1 results are missing.
    """
    v1_file = RESULTS_V1 / f"{model_name}.json"
    if not v1_file.exists():
        return None
    with open(v1_file) as f:
        v1 = json.load(f)

    # Load v2 researcher results if available
    v2_file = RESULTS_V2 / f"{model_name}.json"
    v2_results_by_pid = {}
    has_v2 = False
    v2_failed = False
    if v2_file.exists():
        try:
            with open(v2_file) as f:
                v2 = json.load(f)
            results = v2.get("results", [])
            err_count = sum(1 for r in results if r.get("model_query_error"))
            if err_count >= len(results) * 0.95:
                v2_failed = True
            else:
                has_v2 = True
                for r in results:
                    v2_results_by_pid[r["probe_id"]] = r
        except Exception:
            pass

    # Per-tier accumulator: sum of per-probe scores, count of probes
    by_tier_sum = defaultdict(float)
    by_tier_count = defaultdict(int)
    by_tier_correct = defaultdict(int)  # for raw accuracy (unweighted CORRECT count)
    by_tier_stats = defaultdict(lambda: {"strong": 0, "weak": 0, "wrong": 0,
                                          "refusal": 0, "no_cs_match_correct": 0,
                                          "total": 0})

    for r in v1.get("results", []):
        pid = r.get("probe_id", "")
        tier = r.get("tier")
        if not tier:
            continue
        is_researcher = (r.get("source_type") == "researcher")

        # Use v2 score if researcher probe AND v2 has it
        if is_researcher and has_v2 and pid in v2_results_by_pid:
            v2r = v2_results_by_pid[pid]
            verdict = v2r.get("verdict")
            score = v2r.get("score")
            if score is None:
                score = score_verdict(verdict, LAMBDA)
        else:
            verdict = r.get("verdict")
            score = score_verdict(verdict, LAMBDA)

        by_tier_sum[tier] += score
        by_tier_count[tier] += 1
        if verdict == "CORRECT" or verdict == "CORRECT_STRONG":
            by_tier_correct[tier] += 1
        ts = by_tier_stats[tier]
        ts["total"] += 1
        if verdict == "CORRECT_STRONG":
            ts["strong"] += 1
        elif verdict == "CORRECT_WEAK":
            ts["weak"] += 1
        elif verdict == "CORRECT":
            ts["strong"] += 1
        elif verdict == "REFUSAL":
            ts["refusal"] += 1
        elif verdict == "WRONG":
            ts["wrong"] += 1
        # no_cs_match handled by v2 score directly

    # Per-tier penalized accuracy (floor at 0)
    tier_acc = {}
    for t in TIERS:
        if by_tier_count[t] == 0:
            tier_acc[t] = 0.0
        else:
            tier_acc[t] = max(by_tier_sum[t] / by_tier_count[t], 0.0)

    # Overall = mean of 7 floored per-tier scores
    overall = sum(tier_acc[t] for t in TIERS) / 7.0

    # Raw accuracy (no penalty, no floor)
    total_correct = sum(by_tier_correct[t] for t in TIERS)
    total_count = sum(by_tier_count[t] for t in TIERS)
    raw_acc = total_correct / total_count if total_count else 0

    return {
        "model": model_name,
        "accuracy": overall,
        "raw_accuracy": raw_acc,
        "tier_accuracy": tier_acc,
        "tier_stats": {t: dict(by_tier_stats[t]) for t in TIERS},
        "has_researcher_v2": has_v2,
        "v2_failed": v2_failed,
    }


def main():
    with open(CONFIG) as f:
        cfg = json.load(f)["models"]

    rows = []
    skipped = []
    for model_name in sorted(cfg.keys()):
        out = assemble_one(model_name)
        if out is None:
            skipped.append(model_name)
            continue
        info = cfg[model_name]
        out.update({
            "params_B": info.get("params_B"),
            "active_B": info.get("active_B"),
            "family": info.get("family"),
            "vendor": info.get("vendor"),
            "arch": info.get("arch"),
            "type": info.get("type"),
            "thinking": info.get("thinking", False),
            "release_date": info.get("release_date"),
        })
        rows.append(out)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    n_v2 = sum(1 for r in rows if r["has_researcher_v2"])
    n_v2_failed = sum(1 for r in rows if r["v2_failed"])
    print(f"Assembled {len(rows)} models")
    print(f"  with researcher_v2: {n_v2}")
    print(f"  v2 failed/missing:  {len(rows) - n_v2}")
    print(f"  skipped (no v1):    {len(skipped)}")

    print(f"\nSpotlight:")
    spotlight = ["deepseek-v4-flash", "deepseek-v4-pro",
                 "deepseek-v4-flash-think", "deepseek-v4-pro-think",
                 "gemini-3-flash", "gemini-3-flash-think", "gemini-3.1-pro",
                 "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
                 "claude-opus-4.6-think", "claude-opus-4.7-think",
                 "claude-sonnet-4.6-think", "gpt-5-think", "gpt-5", "o3", "o1",
                 "kimi-k2.5-think", "kimi-k2", "v3-pro-think", "deepseek-v3",
                 "grok-3", "grok-4"]
    by_name = {r["model"]: r for r in rows}
    for name in spotlight:
        if name not in by_name: continue
        r = by_name[name]
        ta = r["tier_accuracy"]
        tier_str = " ".join(f"{ta[t]:.3f}" for t in TIERS)
        v2 = "v2" if r["has_researcher_v2"] else "v1-only"
        print(f"  {name:30s} acc={r['accuracy']:.4f} raw={r['raw_accuracy']:.4f} "
              f"[{tier_str}] {v2}")


if __name__ == "__main__":
    main()
