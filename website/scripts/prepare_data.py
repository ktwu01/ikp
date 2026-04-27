#!/usr/bin/env python3
"""Prepare static JSON files for the IKP visualization website.

Reads from /Users/boj/ikp-paper/data/ and writes compact, web-friendly
JSON to /Users/boj/ikp-paper/website/public/data/.

Outputs:
  models.json              — all models with summary stats (~50KB)
  probes.json              — all 1400 probes with metadata (~500KB)
  calibration.json         — fit params, scatter data, LOO-CV
  pipeline.json            — pipeline stage descriptions + sample artifacts
  tiers/T1.json ... T7.json — per-tier full responses (each ~5MB)
  models/<name>.json       — per-model summary + sample probes (165 files)
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = ROOT / "website" / "public" / "data"

CALIBRATION_EXCLUDE = {
    "minimax-m1-think",       # broken API: most responses empty
    "hunyuan-a13b",           # extreme outlier: 80B scores below 12B dense models
    "hunyuan-a13b-think",     # same issue as hunyuan-a13b
    "hermes-3-405b",          # superseded by hermes-4-405b (3.5σ outlier)
    "ling-2.6-flash",         # extreme outlier: 104B MoE scores like a 1B dense model
    "nemotron-ultra-253b",    # post-training; separate regime
    "deepseek-v3.1-nex-n1",   # post-training: -5.58σ below-trend
    "intellect-3-think",      # post-training (GLM-4.5-Air-Base SFT+RL)
}

# Single-regime calibration. Earlier versions of this code applied a
# two-regime correction to "Flash"-class proprietary variants; this
# correction was retired after the V4 Pro / V4 Flash anchor pair was
# shown to fall within the calibration's natural prediction interval
# at λ = -1.0 (paper §3.5).
DISTILL_BOOST = 1.0
DISTILLED_MODELS = {}


def load_summary():
    return json.load(open(DATA / "results" / "evaluation_summary.json"))


def load_probes():
    # Use v9 probe set (final researcher probes with extended two-part questions);
    # v8 had the short one-part questions but the same probe IDs and gold answers.
    return json.load(open(DATA / "probes" / "final_probe_set_v9.json"))


def load_per_model_results(model_name):
    f = DATA / "results" / f"{model_name}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def fit_linear(open_models):
    log_p = np.array([math.log10(m["params_B"]) for m in open_models])
    acc = np.array([m["accuracy"] for m in open_models])
    sl, b, r, _, se = stats.linregress(log_p, acc)
    pred = sl * log_p + b
    resid = acc - pred
    return {
        "slope": float(sl),
        "intercept": float(b),
        "r_squared": float(r ** 2),
        "residual_se": float(np.std(resid, ddof=2)),
    }


def loo_cv(open_models):
    log_p = np.array([math.log10(m["params_B"]) for m in open_models])
    acc = np.array([m["accuracy"] for m in open_models])
    fold_errors = []
    preds = []
    for i in range(len(open_models)):
        mask = np.ones(len(open_models), dtype=bool)
        mask[i] = False
        sl, b, _, _, _ = stats.linregress(log_p[mask], acc[mask])
        pred_acc = sl * log_p[i] + b
        pred_param = 10 ** ((acc[i] - b) / sl) if sl > 0 else 0
        actual_param = open_models[i]["params_B"]
        fold_err = None
        if pred_param > 0:
            fold_err = max(actual_param / pred_param, pred_param / actual_param)
            fold_errors.append(fold_err)
        preds.append({
            "model": open_models[i]["model"],
            "vendor": open_models[i].get("vendor"),
            "arch": open_models[i].get("arch"),
            "actual_B": float(actual_param),
            "pred_B": float(pred_param),
            "actual_acc": float(acc[i]),
            "pred_acc": float(pred_acc),
            "fold_err": float(fold_err) if fold_err is not None else None,
        })
    actual = np.array([p["actual_acc"] for p in preds])
    predicted = np.array([p["pred_acc"] for p in preds])
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "r_squared": float(1 - ss_res / ss_tot),
        "median_fold_err": float(np.median(fold_errors)),
        "within_2x": float(np.mean(np.array(fold_errors) <= 2)),
        "within_3x": float(np.mean(np.array(fold_errors) <= 3)),
        "predictions": preds,
    }


def fit_moe(moe_models, active=False):
    """Fit log-linear accuracy ~ log10(params) for MoE models on either total or active."""
    key = "active_B" if active else "params_B"
    xs = np.array([math.log10(m[key]) for m in moe_models])
    ys = np.array([m["accuracy"] for m in moe_models])
    sl, b, r, _, _ = stats.linregress(xs, ys)
    return {"slope": float(sl), "intercept": float(b), "r_squared": float(r ** 2)}


def write_calibration(summary):
    open_models = [
        m for m in summary
        if m.get("type") == "open"
        and m.get("params_B")
        and m["params_B"] > 0
        and m["model"] not in CALIBRATION_EXCLUDE
    ]
    fit = fit_linear(open_models)
    cv = loo_cv(open_models)

    # 90% PI half-width in log10(params_B) space, derived from the
    # calibration residual standard error (matches Table 1 of the paper).
    log_p = np.array([math.log10(m["params_B"]) for m in open_models])
    accs = np.array([m["accuracy"] for m in open_models])
    resid = accs - (fit["slope"] * log_p + fit["intercept"])
    n = len(open_models)
    residual_se = float(math.sqrt(float(np.sum(resid ** 2)) / max(n - 2, 1)))
    pi_half_log10 = 1.645 * residual_se / abs(fit["slope"]) if fit["slope"] else 0.0
    fit["residual_se"] = residual_se
    fit["pi_half_log10"] = pi_half_log10
    fit["pi_factor"] = 10 ** pi_half_log10

    # MoE: total vs active parameter fits
    moe = [m for m in open_models if m.get("arch") == "moe" and m.get("active_B")]
    moe_total = fit_moe(moe, active=False) if moe else None
    moe_active = fit_moe(moe, active=True) if moe else None
    moe_points = [
        {
            "model": m["model"],
            "params_B": m["params_B"],
            "active_B": m["active_B"],
            "accuracy": m["accuracy"],
            "vendor": m.get("vendor"),
            "thinking": m.get("thinking", False),
        }
        for m in moe
    ]

    points = [
        {
            "model": m["model"],
            "params_B": m["params_B"],
            "active_B": m.get("active_B"),
            "accuracy": m["accuracy"],
            "raw_accuracy": m["raw_accuracy"],
            "vendor": m.get("vendor"),
            "family": m.get("family"),
            "arch": m.get("arch"),
            "thinking": m.get("thinking", False),
        }
        for m in open_models
    ]

    # Excluded outliers (still plottable but flagged)
    excluded = [
        {
            "model": m["model"],
            "params_B": m.get("params_B"),
            "accuracy": m["accuracy"],
            "vendor": m.get("vendor"),
            "reason": "pathological refusal" if m["model"] in {"minimax-m1-think", "ling-2.6-flash"}
                      else "extreme outlier" if "hunyuan" in m["model"]
                      else "superseded",
        }
        for m in summary
        if m["model"] in CALIBRATION_EXCLUDE and m.get("params_B")
    ]

    def estimate(m):
        if not (fit["slope"] and fit["slope"] > 0):
            return None
        log_eff = (m["accuracy"] - fit["intercept"]) / fit["slope"]
        eff_B = 10 ** log_eff
        is_distilled = m["model"] in DISTILLED_MODELS
        actual_B = eff_B / DISTILL_BOOST if is_distilled else eff_B
        # PI in log10 space, around the actual estimate.
        log_actual = math.log10(actual_B) if actual_B > 0 else 0
        pi_lo = 10 ** (log_actual - pi_half_log10)
        pi_hi = 10 ** (log_actual + pi_half_log10)
        return {
            "model": m["model"],
            "accuracy": m["accuracy"],
            "raw_accuracy": m["raw_accuracy"],
            "vendor": m.get("vendor"),
            "family": m.get("family"),
            "thinking": m.get("thinking", False),
            "estimated_B": actual_B,
            "estimated_B_eff": eff_B,
            "regime": "distilled" if is_distilled else "pretraining",
            "distill_anchor": DISTILLED_MODELS.get(m["model"]) if is_distilled else None,
            "pi_lo": pi_lo,
            "pi_hi": pi_hi,
        }

    # Exclude landmark models from the proprietary-estimates table: Gemini 3.1 Pro
    # is the T6 calibration landmark (T6 inflated by construction) and the Gemini 3.x
    # Flash / Flash-Lite siblings inherit that inflation through within-family score
    # saturation, so single-curve estimates for them are systematically over-stated.
    LANDMARK_EXCLUDE = {
        "gemini-3.1-pro", "gemini-3.1-pro-think",
        "gemini-3-flash", "gemini-3-flash-think",
        "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-think",
    }
    proprietary = [
        e for e in (estimate(m) for m in summary
                    if m.get("type") == "proprietary"
                    and m["model"] not in LANDMARK_EXCLUDE)
        if e
    ]

    out = {
        "fit": fit,
        "loo_cv": cv,
        "n_calibration": len(open_models),
        "n_proprietary": len(proprietary),
        "vendors": sorted({p["vendor"] for p in points if p["vendor"]}),
        "calibration_points": points,
        "excluded_points": excluded,
        "proprietary_estimates": sorted(proprietary, key=lambda p: -p["accuracy"]),
        "distillation": {
            "boost": DISTILL_BOOST,
            "boost_range": [3.44, 4.58],
            "anchor_pair": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "students": sorted(DISTILLED_MODELS.keys()),
        },
        "moe": {
            "total": moe_total,
            "active": moe_active,
            "points": moe_points,
        },
    }
    (OUT / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"calibration.json: {len(points)} calibration, {len(proprietary)} proprietary, R²={fit['r_squared']:.4f}")


def write_models(summary):
    """Compact list of all models for the table view."""
    rows = []
    for m in summary:
        rows.append({
            "model": m["model"],
            "vendor": m.get("vendor"),
            "family": m.get("family"),
            "type": m.get("type"),
            "arch": m.get("arch"),
            "params_B": m.get("params_B"),
            "active_B": m.get("active_B"),
            "thinking": m.get("thinking", False),
            "accuracy": m["accuracy"],
            "raw_accuracy": m["raw_accuracy"],
            "tier_accuracy": m.get("tier_accuracy"),
            "tier_stats": m.get("tier_stats"),
        })
    rows.sort(key=lambda r: (r["vendor"] or "", -r["accuracy"]))
    (OUT / "models.json").write_text(json.dumps(rows, indent=2))
    print(f"models.json: {len(rows)} models")


def write_probes(probes):
    """All probes with metadata, plus across-model correct- and hallucination-rates.

    correct_rate     = fraction of evaluated models that answered correctly
    halluc_rate      = fraction that answered confidently wrong (verdict=WRONG)
    refusal_rate     = fraction that refused
    n_models         = number of models that answered the probe
    """
    # Aggregate across all per-model results.
    agg: dict[str, dict] = {}
    for f in sorted((DATA / "results").glob("*.json")):
        if f.name in ("evaluation_summary.json", "evaluation_summary_pre_v2.json",
                       "analysis.json", "analysis_before_v4.json", "final_assembly.json",
                       "calibration_refit_v2.json", "deepseek-v4-flash_strict_rejudge.json"):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for r in data.get("results", []):
            pid = r.get("probe_id")
            if not pid:
                continue
            row = agg.setdefault(pid, {"n": 0, "correct": 0, "wrong": 0, "refusal": 0})
            row["n"] += 1
            if r.get("correct"):
                row["correct"] += 1
            elif r.get("refusal"):
                row["refusal"] += 1
            else:
                row["wrong"] += 1

    rows = []
    for p in probes:
        pid = p["id"]
        a = agg.get(pid)
        if a and a["n"] > 0:
            n = a["n"]
            correct_rate = a["correct"] / n
            halluc_rate = a["wrong"] / n
            refusal_rate = a["refusal"] / n
            n_models = n
        else:
            correct_rate = halluc_rate = refusal_rate = 0.0
            n_models = 0
        rows.append({
            "id": pid,
            "tier": p["tier"],
            "domain": p.get("domain"),
            "source_type": p.get("source_type"),
            "question": p["question"],
            "answer": p["answer"],
            "n_models": n_models,
            "correct_rate": correct_rate,
            "halluc_rate": halluc_rate,
            "refusal_rate": refusal_rate,
        })
    (OUT / "probes.json").write_text(json.dumps(rows, indent=2))
    print(f"probes.json: {len(rows)} probes (with cross-model stats from {len(agg)} aggregated)")


def _strip_tool_handlers(text: str) -> str:
    """Remove tool-call / invoke wrappers some models emit when they treat the
    factual prompt as a tool-use task. Strips patterns like
    '<minimax:tool_call>...</minimax:tool_call>', '<invoke name="...">',
    'tool_call>' fragments. Idempotent and leaves normal prose untouched."""
    if not text:
        return text
    import re
    text = re.sub(r"<[^<>]*?:?tool_call>.*?</[^<>]*?:?tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?[^<>]*?:?tool_call>", "", text)
    text = re.sub(r"<invoke[^>]*>.*?</invoke>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?invoke[^>]*>", "", text)
    text = re.sub(r"<parameter[^>]*>.*?</parameter>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?parameter[^>]*>", "", text)
    return text.strip()


def _load_researcher_evidence():
    """Load the per-researcher evidence bundle (subfield + named systems +
    co-authors + top works + venues + affiliations) used by the 4-way judge.
    Returns dict keyed by probe_id."""
    p = DATA / "probes" / "researcher_gold_enriched.json"
    if not p.exists():
        return {}
    try:
        rows = json.load(open(p))
    except Exception:
        return {}
    out = {}
    for r in rows:
        pid = r.get("probe_id")
        if not pid:
            continue
        out[pid] = {
            "primary_subfield": r.get("primary_subfield"),
            "secondary_subfields": r.get("secondary_subfields", []),
            "affiliations": r.get("affiliations", []),
            "venues": r.get("venues", []),
            "named_systems": r.get("named_systems", []),
            "co_authors": r.get("co_authors", []),
            "top_works": r.get("top_works", []),
        }
    return out


def write_per_tier_responses(summary, probes):
    """Pivot all model responses by tier — each file ~5MB.

    Schema: { probes: { probe_id: { question, answer, domain, responses: { model: {response, verdict, correct, refusal} } } } }
    """
    by_tier = defaultdict(dict)
    probe_by_id = {p["id"]: p for p in probes}
    researcher_evidence = _load_researcher_evidence()

    # Initialize probe entries
    for p in probes:
        entry = {
            "question": p["question"],
            "answer": p["answer"],
            "domain": p.get("domain"),
            "source_type": p.get("source_type"),
            "responses": {},
        }
        # Attach researcher evidence bundle when this probe sources from a
        # researcher (so the 4-way-judge gold context is visible to readers).
        if p.get("source_type") == "researcher" and p["id"] in researcher_evidence:
            entry["evidence"] = researcher_evidence[p["id"]]
        by_tier[p["tier"]][p["id"]] = entry

    # Fill responses
    for m in summary:
        result = load_per_model_results(m["model"])
        if not result or not result.get("results"):
            continue
        for r in result["results"]:
            tier = r["tier"]
            pid = r["probe_id"]
            if pid not in by_tier[tier]:
                continue
            by_tier[tier][pid]["responses"][m["model"]] = {
                "response": _strip_tool_handlers(r.get("model_response", "")),
                "verdict": r.get("verdict"),
                "correct": r.get("correct", False),
                "refusal": r.get("refusal", False),
            }

    tier_dir = OUT / "tiers"
    tier_dir.mkdir(exist_ok=True)
    for tier, data in by_tier.items():
        path = tier_dir / f"{tier}.json"
        path.write_text(json.dumps({"tier": tier, "probes": data}, ensure_ascii=False))
        size_mb = path.stat().st_size / 1e6
        print(f"  tiers/{tier}.json: {len(data)} probes, {size_mb:.1f}MB")


def write_per_model_summary(summary):
    """Per-model summary including sample probes per tier (correct/wrong/refusal samples)."""
    out_dir = OUT / "models"
    out_dir.mkdir(exist_ok=True)
    for m in summary:
        result = load_per_model_results(m["model"])
        if not result:
            continue
        # Sample 5 of each verdict type per tier for the drill-down page
        samples = defaultdict(lambda: {"CORRECT": [], "WRONG": [], "REFUSAL": []})
        for r in result.get("results", []):
            tier = r["tier"]
            v = r.get("verdict")
            if v in ("CORRECT", "WRONG", "REFUSAL") and len(samples[tier][v]) < 5:
                samples[tier][v].append({
                    "probe_id": r["probe_id"],
                    "question": r["question"],
                    "gold_answer": r["gold_answer"],
                    "model_response": _strip_tool_handlers(r.get("model_response", "")),
                })
        out = {
            "model": m["model"],
            "vendor": m.get("vendor"),
            "family": m.get("family"),
            "type": m.get("type"),
            "arch": m.get("arch"),
            "params_B": m.get("params_B"),
            "active_B": m.get("active_B"),
            "thinking": m.get("thinking", False),
            "accuracy": m["accuracy"],
            "raw_accuracy": m["raw_accuracy"],
            "tier_accuracy": m.get("tier_accuracy"),
            "tier_stats": m.get("tier_stats"),
            "samples": dict(samples),
        }
        (out_dir / f"{m['model']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"models/: {len(summary)} per-model files")


def write_pipeline():
    """Pipeline narrative — describes the 7-stage pipeline with sample artifacts."""
    stages = [
        {
            "id": "1_seed",
            "name": "1. Seed question generation",
            "description": "Frontier LLMs (Claude, GPT-5, Gemini) generate ~10K candidate factual questions across diverse domains. Each question includes a single canonical answer.",
            "output": "Raw LLM-generated questions",
            "count": "~10,000 candidates",
            "script": "scripts/01_generate_probes.py",
        },
        {
            "id": "2_external",
            "name": "2. External knowledge sources",
            "description": "Augment LLM-generated questions with externally-sourced facts: Wikidata entities (filtered by sitelink count for tier assignment), DBLP researcher publications (filtered by citation count), Common Crawl rare facts.",
            "output": "Wikidata + researcher + corpus probes",
            "count": "~5,000 candidates",
            "script": "scripts/10_web_grounded_probes.py",
        },
        {
            "id": "3_calibration",
            "name": "3. Tier calibration",
            "description": "Six landmark models of varying sizes (1B, 8B, 70B, 235B-A22B, 405B, Gemini 3.1 Pro) answer each candidate question. The pattern of correct answers determines the question's tier (T1=easy, all-correct; T7=hardest, only Gemini 3.1 Pro correct).",
            "output": "Calibration matrix (probe × landmark)",
            "count": "6 landmarks × ~15K candidates",
            "script": "scripts/02_run_calibration.py",
        },
        {
            "id": "4_filter",
            "name": "4. Quality filtering",
            "description": "Filter out: computable knowledge (math, derivable facts), 2-character Chinese names (transcription ambiguity), researchers in ML/AI fields (memorization shortcut), and questions with ambiguous gold answers.",
            "output": "Filtered candidate pool",
            "count": "~3,000 candidates",
            "script": "scripts/01_generate_probes.py + manual review",
        },
        {
            "id": "5_assemble",
            "name": "5. Final probe assembly",
            "description": "Sample 200 probes per tier (T1–T7) balanced across LLM-generated, researcher, and Wikidata sources. T3/T4 split: LLM=50, Researcher=50, Wikidata=100.",
            "output": "1,400 final probes (200 × 7 tiers)",
            "count": "1,400 probes",
            "script": "scripts/assemble_final_dataset.py",
        },
        {
            "id": "6_evaluate",
            "name": "6. Model evaluation",
            "description": "Each target model answers all 1,400 probes via OpenRouter or local Ollama. Strict factual grading by Gemini 3 Flash judge with REFUSAL/CORRECT/WRONG verdicts.",
            "output": "Per-model probe-level responses",
            "count": "188 models × 1,400 probes = 263K responses",
            "script": "scripts/run_evaluation.py",
        },
        {
            "id": "7_calibrate_curve",
            "name": "7. Scaling curve fit",
            "description": "Fit log-linear regression: accuracy = α·log10(N_B) + β on 89 open-weight models. Invert to estimate proprietary model parameter counts. Validate with leave-one-out cross-validation.",
            "output": "Calibration curve + parameter estimates",
            "count": "89 calibration → 92 proprietary estimates",
            "script": "scripts/analyze_results.py",
        },
    ]
    (OUT / "pipeline.json").write_text(json.dumps({"stages": stages}, indent=2))
    print(f"pipeline.json: {len(stages)} stages")


def write_thinking_pairs(summary):
    """Thinking mode pair comparisons."""
    by_model = {m["model"]: m for m in summary}
    pairs = []
    for name, m in by_model.items():
        if name.endswith("-think"):
            base_name = name[:-6]
            if base_name in by_model:
                base = by_model[base_name]
                pairs.append({
                    "base": base_name,
                    "think": name,
                    "base_acc": base["accuracy"],
                    "think_acc": m["accuracy"],
                    "delta": m["accuracy"] - base["accuracy"],
                    "vendor": base.get("vendor"),
                    "params_B": base.get("params_B"),
                })
    pairs.sort(key=lambda p: -p["delta"])
    (OUT / "thinking_pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"thinking_pairs.json: {len(pairs)} pairs")


def write_hallucination(summary):
    """Vendor-level hallucination rate on T5-T7 probes.

    hallucination rate per-model per-tier = wrong / total (NOT wrong / (wrong+correct+refusal) since total already includes all).
    Reporting per-model (for scatter) and per-vendor aggregate (for bars).
    """
    per_model = []
    for m in summary:
        tier_stats = m.get("tier_stats", {})
        # Aggregate across T5-T7
        w = sum(tier_stats.get(t, {}).get("wrong", 0) for t in ("T5", "T6", "T7"))
        tot = sum(tier_stats.get(t, {}).get("total", 0) for t in ("T5", "T6", "T7"))
        if tot == 0:
            continue
        per_model.append({
            "model": m["model"],
            "vendor": m.get("vendor"),
            "thinking": m.get("thinking", False),
            "accuracy": m["accuracy"],
            "t5_t7_wrong": w,
            "t5_t7_total": tot,
            "halluc_rate": w / tot,
            "per_tier": {
                t: (
                    (tier_stats[t]["wrong"] / tier_stats[t]["total"])
                    if t in tier_stats and tier_stats[t]["total"] else None
                )
                for t in ("T5", "T6", "T7")
            },
        })

    # Vendor aggregate
    by_vendor = defaultdict(list)
    for r in per_model:
        if r["vendor"]:
            by_vendor[r["vendor"]].append(r)
    vendors = []
    for v, rows in by_vendor.items():
        rates = [r["halluc_rate"] for r in rows]
        vendors.append({
            "vendor": v,
            "n_models": len(rows),
            "mean_halluc": float(np.mean(rates)),
            "median_halluc": float(np.median(rates)),
            "min": float(np.min(rates)),
            "max": float(np.max(rates)),
        })
    vendors.sort(key=lambda v: -v["mean_halluc"])

    out = {
        "per_model": per_model,
        "vendors": vendors,
    }
    (OUT / "hallucination.json").write_text(json.dumps(out, indent=2))
    print(f"hallucination.json: {len(per_model)} models, {len(vendors)} vendors")


def write_generations(summary):
    """Within-family generation trajectories + GPT-5 family stratification.

    A generation is identified by family name. We extract all models in the same
    family ordered by a rough generation key inferred from the model name.
    """
    FAMILY_GROUPS = {
        "GPT (base)": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4.1", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4"],
        "Claude Opus": ["claude-opus-4", "claude-opus-4.1", "claude-opus-4.5", "claude-opus-4.6", "claude-opus-4.7"],
        "Claude Sonnet": ["claude-3.7-sonnet", "claude-sonnet-4", "claude-sonnet-4.5", "claude-sonnet-4.6"],
        "Claude Haiku": ["claude-3-haiku", "claude-3.5-haiku", "claude-haiku-4.5"],
        "Gemini Flash": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash"],
        "Gemini Pro": ["gemini-2.5-pro", "gemini-3.1-pro"],
        "Gemma": ["gemma-2-2b", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-2-27b", "gemma-3-27b", "gemma-4-31b"],
        "DeepSeek V3": ["deepseek-v3", "deepseek-v3.1", "deepseek-v3.2"],
        "GLM": ["glm-4.5-think", "glm-4.6-think", "glm-4.7-think", "glm-5-think", "glm-5.1-think"],
        "Qwen Plus": ["qwen-plus", "qwen3.5-plus-think", "qwen3.6-plus-think"],
        "Llama 3 (70B)": ["llama-3-70b", "llama-3.1-70b", "llama-3.3-70b"],
        "Grok": ["grok-3", "grok-4", "grok-4.20"],
        "Kimi K2": ["kimi-k2", "kimi-k2.5-think", "kimi-k2.6-think"],
    }
    by_model = {m["model"]: m for m in summary}
    families_out = []
    for label, names in FAMILY_GROUPS.items():
        chain = []
        for n in names:
            m = by_model.get(n)
            if not m:
                continue
            chain.append({
                "model": n,
                "vendor": m.get("vendor"),
                "accuracy": m["accuracy"],
                "raw_accuracy": m["raw_accuracy"],
                "tier_accuracy": m.get("tier_accuracy"),
                "params_B": m.get("params_B"),
            })
        if len(chain) >= 2:
            families_out.append({"family": label, "chain": chain})

    # GPT-5 family stratification (base / mini / nano / pro / think)
    gpt5_variants = [
        "gpt-5-nano", "gpt-5.4-nano",
        "gpt-5-mini", "gpt-5.4-mini",
        "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4",
        "gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro",
        "gpt-5-think",
    ]
    gpt5 = []
    for n in gpt5_variants:
        m = by_model.get(n)
        if not m:
            continue
        # Extract variant class from name
        if n.endswith("-nano"):
            variant = "nano"
        elif n.endswith("-mini"):
            variant = "mini"
        elif n.endswith("-pro"):
            variant = "pro"
        elif n.endswith("-think"):
            variant = "think"
        else:
            variant = "base"
        gpt5.append({
            "model": n,
            "variant": variant,
            "accuracy": m["accuracy"],
            "raw_accuracy": m["raw_accuracy"],
            "tier_accuracy": m.get("tier_accuracy"),
        })

    out = {"families": families_out, "gpt5_family": gpt5}
    (OUT / "generations.json").write_text(json.dumps(out, indent=2))
    print(f"generations.json: {len(families_out)} families, {len(gpt5)} GPT-5 variants")


def _pair_id(a: str, b: str) -> str:
    """Stable URL-safe id for an (a, b) pair, normalised so that
    pair_id(a, b) == pair_id(b, a). Frontend uses this id as the route
    parameter and as the per-pair JSON filename."""
    x, y = sorted([a, b])
    return f"{x}__{y}"


def write_fingerprint():
    """Within-family + cross-vendor fingerprint metrics for the Fingerprint page.

    Source: results/comprehensive_fingerprint_results.json
    """
    src = json.load(open(RESULTS / "comprehensive_fingerprint_results.json"))
    families = []
    for family_name, rows in src["series"].items():
        families.append({
            "family": family_name,
            "pairs": [
                {
                    "pair_id": _pair_id(r["a"], r["b"]),
                    "a": r["a"],
                    "b": r["b"],
                    "jaccard": r["jaccard"],
                    "lift": r["lift"],
                    "hss": r["hss"],
                    "n_a": r["n_a"],
                    "n_b": r["n_b"],
                    "inter": r["inter"],
                    "both_wrong": r["both_wrong"],
                    "same_wrong": r["same_wrong"],
                    "class": r["class"],
                }
                for r in rows
            ],
        })

    cross = [
        {
            "pair_id": _pair_id(r["a"], r["b"]),
            "a": r["a"],
            "b": r["b"],
            "jaccard": r["jaccard"],
            "lift": r["lift"],
            "hss": r["hss"],
            "both_wrong": r["both_wrong"],
            "same_wrong": r.get("same_wrong", 0),
            "class": r.get("class"),
            "vendor_a": r.get("vendor_a"),
            "vendor_b": r.get("vendor_b"),
        }
        for r in src.get("cross_vendor_outliers", [])
    ]

    # Jaccard heatmap data for 15 frontier models -- pulled from all_pairs.
    FRONTIER = [
        "gemini-3.1-pro", "gemini-3-flash", "gpt-5", "gpt-5.4", "gpt-5.4-pro",
        "claude-opus-4.6", "claude-opus-4.7", "grok-4", "deepseek-v3.2",
        "qwen3-max", "kimi-k2.6-think", "glm-5.1-think", "llama-3.1-70b",
        "mistral-large", "ernie-4.5-300b-a47b",
    ]
    all_pairs = src.get("all_pairs", {})
    heatmap = []
    for a in FRONTIER:
        row = []
        for b in FRONTIER:
            if a == b:
                row.append({"j": 1.0, "hss": None, "both_w": 0})
            else:
                key = f"{a}|{b}" if f"{a}|{b}" in all_pairs else f"{b}|{a}"
                r = all_pairs.get(key)
                row.append({
                    "j": r["jaccard"] if r else None,
                    "hss": r["hss"] if r else None,
                    "both_w": r["both_wrong"] if r else None,
                } if r else None)
        heatmap.append(row)

    # Consecutive-pair flat list for the Fig 8a bar chart (HSS per pair).
    consecutive_pairs = []
    for family_name, rows in src["series"].items():
        for r in rows:
            consecutive_pairs.append({
                "pair_id": _pair_id(r["a"], r["b"]),
                "family": family_name,
                "a": r["a"],
                "b": r["b"],
                "hss": r["hss"],
                "jaccard": r["jaccard"],
                "both_wrong": r["both_wrong"],
                "same_wrong": r["same_wrong"],
                "class": r["class"],
            })

    out = {
        "n_probes": src["n_probes_T5T6"],
        "n_models": src["n_models"],
        "thresholds": {
            "shared_base_hss": 0.30,
            "lineage_hss": 0.10,
            "min_both_wrong": 10,
        },
        "families": families,
        "cross_vendor": cross,
        "heatmap": {
            "models": FRONTIER,
            "matrix": heatmap,
        },
        "consecutive_pairs": consecutive_pairs,
    }
    (OUT / "fingerprint.json").write_text(json.dumps(out, indent=2))
    print(f"fingerprint.json: {len(families)} families, {len(cross)} cross-vendor outliers, {len(consecutive_pairs)} pairs")

    # ── Per-pair detail files ────────────────────────────────────────────────
    # One JSON per (a, b) pair surfaced in the Fingerprint UI: lists every
    # probe where both models are wrong, with each model's response and
    # whether they hallucinated the same wrong answer. Used by the click-
    # through pair-detail page.
    pair_index: dict[tuple[str, str], dict] = {}
    seen_ids: set[str] = set()

    def _seed_pair(a: str, b: str, source: str, extra: dict | None = None) -> str:
        pid = _pair_id(a, b)
        if pid not in pair_index:
            pair_index[pid] = {"a": a, "b": b, "sources": [source], "meta": dict(extra or {})}
        else:
            if source not in pair_index[pid]["sources"]:
                pair_index[pid]["sources"].append(source)
            if extra:
                pair_index[pid]["meta"].update(extra)
        seen_ids.add(pid)
        return pid

    for r in src.get("cross_vendor_outliers", []):
        _seed_pair(r["a"], r["b"], "cross_vendor", {
            "hss": r["hss"], "jaccard": r["jaccard"], "lift": r.get("lift"),
            "both_wrong": r["both_wrong"], "same_wrong": r.get("same_wrong", 0),
            "class": r.get("class"), "vendor_a": r.get("vendor_a"), "vendor_b": r.get("vendor_b"),
        })
    for family_name, rows in src["series"].items():
        for r in rows:
            _seed_pair(r["a"], r["b"], "consecutive", {
                "family": family_name,
                "hss": r["hss"], "jaccard": r["jaccard"], "lift": r.get("lift"),
                "both_wrong": r["both_wrong"], "same_wrong": r["same_wrong"],
                "class": r["class"],
            })

    # Load per-model results once. {model -> {probe_id -> {response, correct, refusal, verdict, tier, question, gold}}}
    needed_models = set()
    for entry in pair_index.values():
        needed_models.add(entry["a"]); needed_models.add(entry["b"])

    model_probes: dict[str, dict] = {}
    for name in needed_models:
        f = DATA / "results" / f"{name}.json"
        if not f.exists():
            continue
        data = json.load(open(f))
        idx: dict[str, dict] = {}
        for r in data.get("results", []):
            pid = r.get("probe_id")
            if not pid:
                continue
            idx[pid] = {
                "response": r.get("model_response"),
                "correct": bool(r.get("correct")),
                "refusal": bool(r.get("refusal")),
                "verdict": r.get("verdict"),
                "tier": r.get("tier"),
                "question": r.get("question"),
                "gold_answer": r.get("gold_answer"),
                "domain": r.get("domain"),
            }
        model_probes[name] = idx

    pair_dir = OUT / "fingerprint_pairs"
    pair_dir.mkdir(exist_ok=True)
    n_pair_files = 0
    for pid, entry in pair_index.items():
        a, b = entry["a"], entry["b"]
        ap = model_probes.get(a, {})
        bp = model_probes.get(b, {})

        joint_wrong = []
        for pid_probe, ra in ap.items():
            rb = bp.get(pid_probe)
            if rb is None:
                continue
            # "wrong" excludes refusals (paper's HSS denominator)
            a_wrong = (not ra["correct"]) and (not ra["refusal"])
            b_wrong = (not rb["correct"]) and (not rb["refusal"])
            if not (a_wrong and b_wrong):
                continue
            same = (ra["response"] or "").strip().lower() == (rb["response"] or "").strip().lower()
            joint_wrong.append({
                "probe_id": pid_probe,
                "tier": ra["tier"],
                "domain": ra.get("domain"),
                "question": ra["question"],
                "gold_answer": ra["gold_answer"],
                "response_a": ra["response"],
                "response_b": rb["response"],
                "verdict_a": ra["verdict"],
                "verdict_b": rb["verdict"],
                "same_wrong": same,
            })

        # also surface "either-wrong" disagreement probes (one wrong, one right)
        # capped — useful colour for distinguishing the pair's strengths.
        disagreement = []
        for pid_probe, ra in ap.items():
            rb = bp.get(pid_probe)
            if rb is None:
                continue
            if ra["correct"] == rb["correct"]:
                continue
            disagreement.append({
                "probe_id": pid_probe,
                "tier": ra["tier"],
                "question": ra["question"],
                "gold_answer": ra["gold_answer"],
                "correct_a": ra["correct"],
                "correct_b": rb["correct"],
                "response_a": ra["response"],
                "response_b": rb["response"],
                "verdict_a": ra["verdict"],
                "verdict_b": rb["verdict"],
            })

        # Sort joint_wrong by tier (T7→T1) so the rarest probes come first.
        tier_order = {f"T{i}": -i for i in range(1, 8)}
        joint_wrong.sort(key=lambda r: (tier_order.get(r.get("tier") or "", 0), r["probe_id"]))
        # Cap disagreement to keep file size in check.
        disagreement.sort(key=lambda r: (tier_order.get(r.get("tier") or "", 0), r["probe_id"]))
        disagreement = disagreement[:60]

        out_pair = {
            "pair_id": pid,
            "a": a,
            "b": b,
            **entry["meta"],
            "sources": entry["sources"],
            "n_joint_wrong": len(joint_wrong),
            "n_same_wrong": sum(1 for r in joint_wrong if r["same_wrong"]),
            "joint_wrong": joint_wrong,
            "disagreement": disagreement,
        }
        (pair_dir / f"{pid}.json").write_text(json.dumps(out_pair, indent=2, ensure_ascii=False))
        n_pair_files += 1
    print(f"fingerprint_pairs/: {n_pair_files} pair files")


def write_densing():
    """Densing-Law falsification data.

    Source: data/densing_analysis_data.csv
    Fits pen_acc ~ beta0 + beta1*log10(N_B) + beta2*months and compares beta2 with
    the Densing prediction beta1 * log10(2) / 3.5 per month.
    """
    rows = []
    with open(DATA / "densing_analysis_data.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "vendor": r["vendor"],
                "family": r["family"],
                "arch": r["arch"],
                "thinking": r["thinking"] == "True",
                "params_B": float(r["params_B"]) if r["params_B"] else None,
                "active_B": float(r["active_B"]) if r["active_B"] else None,
                "release_date": r["release_date"],
                "pen_acc": float(r["pen_acc"]),
                "raw_acc": float(r["raw_acc"]),
                "log10_params": float(r["log10_params"]),
                "months": float(r["months"]),
            })

    # Fit pen_acc = b0 + b1*log10(N) + b2*months
    X = np.array([[1.0, r["log10_params"], r["months"]] for r in rows])
    y = np.array([r["pen_acc"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1, b2 = float(coef[0]), float(coef[1]), float(coef[2])
    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot

    # Log-only baseline (no time term) for reporting
    X0 = X[:, :2]
    coef0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    y0_pred = X0 @ coef0
    r2_no_time = 1 - float(np.sum((y - y0_pred) ** 2)) / ss_tot
    b1_only = float(coef0[1])

    # Bootstrap CI for b2.
    rng = np.random.default_rng(0)
    boot = []
    n = len(rows)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        yb = y[idx]
        try:
            cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boot.append(cb[2])
        except np.linalg.LinAlgError:
            continue
    boot = np.array(boot)
    ci_lo = float(np.quantile(boot, 0.025))
    ci_hi = float(np.quantile(boot, 0.975))

    # Densing prediction: 3.5-month half-life on capability density
    densing_b2 = b1_only * math.log10(2) / 3.5

    # Partial residuals: residual of pen_acc after partialling out log10(N)
    # Compute slope and intercept of pen_acc ~ log10(N); take residuals
    x_logN = X[:, 1]
    resid_time = y - (coef0[0] + coef0[1] * x_logN)
    partial = [
        {"model": r["model"], "months": r["months"], "resid": float(resid_time[i]),
         "log10_params": r["log10_params"], "vendor": r["vendor"], "thinking": r["thinking"]}
        for i, r in enumerate(rows)
    ]

    out = {
        "n": n,
        "fit": {
            "b0": b0, "b1_logN": b1, "b2_time": b2, "r_squared": r2,
            "ci95_b2": [ci_lo, ci_hi],
            "baseline_b1": b1_only,
            "baseline_r_squared": r2_no_time,
            "r2_gain_from_time": r2 - r2_no_time,
        },
        "densing_prediction": {
            "b2": densing_b2,
            "doubling_time_months": 3.5,
            "comment": "Densing-Law prediction: capability density doubles every 3.5 months.",
        },
        "points": rows,
        "partial_residuals": partial,
    }
    (OUT / "densing.json").write_text(json.dumps(out, indent=2))
    print(f"densing.json: n={n}, b2={b2:+.4f}/month (CI [{ci_lo:+.4f},{ci_hi:+.4f}]), Densing predicts {densing_b2:+.4f}")


def write_benchmark_comparison():
    """Compare official MMLU/MMLU-Pro/GPQA Diamond/SimpleQA scores vs IKP as
    a parameter-count proxy. Source data: data/benchmarks/benchmark_scores.csv
    (vendor-published official numbers) joined with densing_analysis_data.csv.
    """
    bench_path = DATA / "benchmarks" / "benchmark_scores.csv"
    if not bench_path.exists():
        print("benchmarks.json: skipped (data/benchmarks/benchmark_scores.csv missing)")
        return

    # Load IKP rows.
    ikp_rows = []
    with open(DATA / "densing_analysis_data.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] in CALIBRATION_EXCLUDE:
                continue
            ikp_rows.append({
                "model": r["model"],
                "vendor": r["vendor"],
                "params_B": float(r["params_B"]) if r["params_B"] else None,
                "log10_params": float(r["log10_params"]),
                "months": float(r["months"]),
                "release_date": r["release_date"],
                "ikp": float(r["pen_acc"]) * 100.0,
            })
    by_model = {r["model"]: r for r in ikp_rows}

    # Load benchmark scores and merge.
    bench_cols = ["mmlu", "mmlu_pro", "gpqa_diamond", "simpleqa"]
    with open(bench_path) as f:
        for r in csv.DictReader(f):
            m = by_model.get(r["model"])
            if not m:
                continue
            for c in bench_cols:
                v = r.get(c, "").strip()
                if v:
                    m[c] = float(v)

    # Per-benchmark fits: score ~ log10(N), and joint with months.
    def fit_simple(xs, ys):
        a = np.array(xs); b = np.array(ys)
        slope, intercept, rval, _, _ = stats.linregress(a, b)
        return {"n": len(a), "slope": float(slope), "intercept": float(intercept), "r2": float(rval ** 2)}

    def fit_joint(x_p, x_m, y):
        X = np.column_stack([np.ones(len(y)), np.array(x_p), np.array(x_m)])
        beta, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((np.array(y) - yhat) ** 2))
        ss_tot = float(np.sum((np.array(y) - np.mean(y)) ** 2))
        return {"intercept": float(beta[0]), "slope_params": float(beta[1]),
                "slope_months": float(beta[2]),
                "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0}

    bench_labels = {
        "mmlu": "MMLU", "mmlu_pro": "MMLU-Pro",
        "gpqa_diamond": "GPQA Diamond", "simpleqa": "SimpleQA",
    }

    benchmarks_out = []
    for c in bench_cols:
        sub = [m for m in ikp_rows if c in m]
        if len(sub) < 4:
            continue
        x_p = [m["log10_params"] for m in sub]
        x_m = [m["months"] for m in sub]
        y_b = [m[c] for m in sub]
        y_i = [m["ikp"] for m in sub]
        fit_b = fit_simple(x_p, y_b)
        fit_i = fit_simple(x_p, y_i)
        joint_b = fit_joint(x_p, x_m, y_b)
        joint_i = fit_joint(x_p, x_m, y_i)
        benchmarks_out.append({
            "key": c,
            "label": bench_labels[c],
            "n": len(sub),
            "benchmark_fit": fit_b,
            "ikp_fit_same_subset": fit_i,
            "benchmark_joint": joint_b,
            "ikp_joint_same_subset": joint_i,
            "points": [
                {"model": m["model"], "vendor": m["vendor"],
                 "log10_params": m["log10_params"], "months": m["months"],
                 "release_date": m["release_date"],
                 "score": m[c], "ikp": m["ikp"]}
                for m in sub
            ],
        })

    # Full IKP fit (all 89) as reference.
    ikp_full_fit = fit_simple([m["log10_params"] for m in ikp_rows],
                              [m["ikp"] for m in ikp_rows])
    ikp_full_joint = fit_joint([m["log10_params"] for m in ikp_rows],
                               [m["months"] for m in ikp_rows],
                               [m["ikp"] for m in ikp_rows])

    out = {
        "n_total": len(ikp_rows),
        "ikp_full_fit": ikp_full_fit,
        "ikp_full_joint": ikp_full_joint,
        "benchmarks": benchmarks_out,
    }
    (OUT / "benchmarks.json").write_text(json.dumps(out, indent=2))
    print(f"benchmarks.json: {len(benchmarks_out)} benchmarks, IKP R²={ikp_full_fit['r2']:.3f} (n={len(ikp_rows)})")


def write_recognition():
    """Researcher recognition vs citations for the Recognition page."""
    rates = json.load(open(DATA / "researcher_recognition_rates.json"))
    cits = json.load(open(DATA / "researcher_citations.json"))
    by_id = {c["probe_id"]: c for c in cits}

    rows = []
    for r in rates:
        c = by_id.get(r["probe_id"])
        if not c:
            continue
        rows.append({
            "probe_id": r["probe_id"],
            "name": r["name"],
            "tier": r["tier"],
            "recognition_rate": r["recognition_rate"],
            "correct": r["correct"],
            "total": r["total"],
            "refusal": r["refusal"],
            "wrong": r["wrong"],
            "field": c.get("answer"),
            "domain": c.get("domain"),
            "works_count": c.get("works_count"),
            "cited_by_count": c.get("cited_by_count"),
            "h_index": c.get("h_index"),
            "i10_index": c.get("i10_index"),
        })

    # Quantile buckets on log10(citations+1) for the summary band.
    valid = [r for r in rows if r["cited_by_count"]]
    log_c = np.array([math.log10(max(r["cited_by_count"], 1)) for r in valid])
    # 5 buckets
    edges = np.quantile(log_c, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    buckets = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        bucket = [r for r in valid
                  if log_c[valid.index(r)] >= lo and
                  (log_c[valid.index(r)] <= hi if i == 4 else log_c[valid.index(r)] < hi)]
        # Above is O(n^2); simpler:
    # Rebuild cleanly
    idx = np.argsort(log_c)
    sorted_rows = [valid[i] for i in idx]
    per_bucket_n = max(1, len(sorted_rows) // 5)
    buckets = []
    for i in range(5):
        lo = i * per_bucket_n
        hi = (i + 1) * per_bucket_n if i < 4 else len(sorted_rows)
        b = sorted_rows[lo:hi]
        if not b:
            continue
        log_range = (math.log10(max(b[0]["cited_by_count"], 1)),
                     math.log10(max(b[-1]["cited_by_count"], 1)))
        median_rec = float(np.median([r["recognition_rate"] for r in b]))
        median_cit = int(np.median([r["cited_by_count"] for r in b]))
        buckets.append({
            "index": i,
            "n": len(b),
            "log_cit_range": log_range,
            "citations_range": [b[0]["cited_by_count"], b[-1]["cited_by_count"]],
            "median_citations": median_cit,
            "median_recognition": median_rec,
        })

    # Pearson and Spearman on log citations vs recognition
    logc_arr = np.array([math.log10(max(r["cited_by_count"], 1)) for r in valid])
    rec_arr = np.array([r["recognition_rate"] for r in valid])
    pearson = float(stats.pearsonr(logc_arr, rec_arr).statistic)
    spearman = float(stats.spearmanr(logc_arr, rec_arr).statistic)

    out = {
        "n": len(rows),
        "n_with_citations": len(valid),
        "pearson_log_citations": pearson,
        "spearman_log_citations": spearman,
        "quintile_buckets": buckets,
        "points": rows,
    }
    (OUT / "recognition.json").write_text(json.dumps(out, indent=2))
    print(f"recognition.json: n={len(rows)} ({len(valid)} with citations), ρ={spearman:.3f}")


def write_index():
    """Top-level index summarizing what data is available."""
    out = {
        "schema_version": 1,
        "generated_from": str(DATA),
        "files": {
            "models": "models.json",
            "probes": "probes.json",
            "calibration": "calibration.json",
            "pipeline": "pipeline.json",
            "thinking_pairs": "thinking_pairs.json",
            "hallucination": "hallucination.json",
            "generations": "generations.json",
            "fingerprint": "fingerprint.json",
            "densing": "densing.json",
            "recognition": "recognition.json",
            "tier_responses": "tiers/{T1..T7}.json",
            "model_detail": "models/{model_name}.json",
        },
    }
    (OUT / "index.json").write_text(json.dumps(out, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    probes = load_probes()
    print(f"Loaded {len(summary)} models, {len(probes)} probes")

    # Enrich summary with config metadata (type, arch, active_B, thinking).
    # evaluation_summary.json from run_evaluation.py does not include these.
    cfg = json.load(open(ROOT / "configs" / "all_models.json"))["models"]
    for m in summary:
        info = cfg.get(m["model"])
        if info:
            m.setdefault("type", info.get("type", "unknown"))
            m.setdefault("arch", info.get("arch", "unknown"))
            m.setdefault("active_B", info.get("active_B"))
            m.setdefault("thinking", info.get("thinking", False))
            if m.get("params_B") is None:
                m["params_B"] = info.get("params_B")
        else:
            m.setdefault("type", "unknown")
            m.setdefault("arch", "unknown")
            m.setdefault("active_B", None)
            m.setdefault("thinking", False)

    write_calibration(summary)
    write_models(summary)
    write_probes(probes)
    write_pipeline()
    write_thinking_pairs(summary)
    write_hallucination(summary)
    write_generations(summary)
    write_fingerprint()
    write_densing()
    write_benchmark_comparison()
    write_recognition()
    write_per_model_summary(summary)
    write_per_tier_responses(summary, probes)
    write_index()

    print("\nDone.")


if __name__ == "__main__":
    main()
