#!/usr/bin/env python3
"""Re-score every model at lambda=0 on the CLEANED probe subset.

Keeps the full per-probe verdicts in each result file (source of truth) but
recomputes accuracy/tier_accuracy/tier_stats over the clean_ids only, at
lambda=0 (no penalty; floor is a no-op since scores are >=0). Also drops records
flagged as query ERRORs (empty response due to network/API failure) rather than
scoring them as refusals. Rebuilds evaluation_summary.json.
"""
import json, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
CFG = json.load(open(ROOT / "configs" / "all_models.json"))["models"]
MASK = json.load(open(ROOT / "data" / "probes" / "clean_mask.json"))
CLEAN = set(MASK["clean_ids"])
TIERS = ["T1","T2","T3","T4","T5","T6","T7"]
SKIP = {"evaluation_summary.json","evaluation_summary_pre_v2.json","analysis.json",
        "analysis_before_v4.json","calibration_refit_v2.json","final_assembly.json",
        "deepseek-v4-flash_strict_rejudge.json"}

def is_error(r):
    # empty response with no genuine refusal text -> treat as query error, exclude
    return (not (r.get("model_response") or "").strip()) and r.get("verdict") == "REFUSAL"

def score(records):
    ts = {t: {"total":0,"correct":0,"wrong":0,"refusal":0} for t in TIERS}
    n_excluded = 0
    for r in records:
        if r.get("probe_id") not in CLEAN:
            continue
        if is_error(r):
            n_excluded += 1
            continue
        t = r.get("tier")
        if t not in ts: continue
        ts[t]["total"] += 1
        if r.get("refusal"): ts[t]["refusal"] += 1
        elif r.get("correct"): ts[t]["correct"] += 1
        else: ts[t]["wrong"] += 1
    tacc = {}
    for t in TIERS:
        s = ts[t]
        s["score"] = (s["correct"] / s["total"]) if s["total"] else 0.0  # lambda=0
        tacc[t] = s["score"]
    acc = sum(tacc.values()) / 7
    tot = sum(s["total"] for s in ts.values()); corr = sum(s["correct"] for s in ts.values())
    return ts, tacc, acc, (corr/tot if tot else 0.0), corr, tot, n_excluded

summary = []; tot_excluded = 0
for f in sorted(glob.glob(str(RES / "*.json"))):
    if os.path.basename(f) in SKIP: continue
    d = json.load(open(f))
    if "model_name" not in d or "results" not in d: continue
    ts, tacc, acc, raw, corr, tot, nx = score(d["results"])
    tot_excluded += nx
    d.update({"accuracy": acc, "raw_accuracy": raw, "tier_accuracy": tacc, "tier_stats": ts,
              "hallucination_penalty": 0.0, "correct": corr, "total": tot,
              "scored_on": "clean_subset", "n_clean_probes": tot, "n_query_errors_excluded": nx})
    json.dump(d, open(f, "w"), indent=2, ensure_ascii=False)
    c = CFG.get(d["model_name"], {})
    summary.append({"model": d["model_name"], "params_B": c.get("params_B", d.get("params_B")),
                    "family": c.get("family", d.get("family")), "vendor": c.get("vendor", d.get("vendor")),
                    "accuracy": acc, "raw_accuracy": raw, "tier_accuracy": tacc, "tier_stats": ts})

json.dump(summary, open(RES / "evaluation_summary.json", "w"), indent=2, ensure_ascii=False)
print(f"re-scored {len(summary)} models at λ=0 on {len(CLEAN)} clean probes; "
      f"excluded {tot_excluded} silent query-errors across models")
for name in ["claude-fable-5","gpt-5.5","claude-opus-4.7","nemotron-3-ultra","glm-5","gpt-4.1"]:
    m = next((x for x in summary if x["model"]==name), None)
    if m: print(f"  {name:18s} acc(clean,λ0)={m['accuracy']:.4f}")
