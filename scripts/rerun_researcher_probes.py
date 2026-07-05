"""Re-run researcher subfield probes with the new evidence-aware judge.

Scope: ONLY researcher probes (T3-T7, source_type=researcher). Other probe
types (wikidata, llm, T*_final, manual) are NOT touched — their stored
verdicts remain valid.

Concurrency (per user spec):
  - OpenRouter models: 16 models concurrent × 8 requests per model = 128 in-flight
  - Ollama (local) models: serial (one at a time) × 16 requests per model

Reads:
  data/probes/final_probe_set_v9.json          — rewritten probe questions
  data/probes/researcher_gold_enriched.json    — manual + OpenAlex enrichment
  configs/all_models.json                      — model config

Writes:
  data/results_v2/<model>.json                 — fresh per-model researcher results
  data/results_v2/researcher_summary.json      — aggregated summary
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evidence_judge import judge_evidence, score_4way, JUDGE_MODEL_DEFAULT

PROBES_PATH = ROOT / "data" / "probes" / "final_probe_set_v9.json"
ENRICHED_PATH = ROOT / "data" / "probes" / "researcher_gold_enriched.json"
CONFIG_PATH = ROOT / "configs" / "all_models.json"
OUT_DIR = ROOT / "data" / "results_v2"
SUMMARY = OUT_DIR / "researcher_summary.json"
LOG_FILE = ROOT / "data" / "evaluation_v9_researcher.log"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OLLAMA_BASE = "http://localhost:11434/v1/chat/completions"
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."

# Concurrency settings (per user spec)
OPENROUTER_MODELS_CONCURRENT = 16
OPENROUTER_REQUESTS_PER_MODEL = 8
OLLAMA_REQUESTS_PER_MODEL = 16  # serial models, but parallel requests per model


def log(msg):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def strip_thinking(text):
    if not text: return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        cleaned = cleaned[end + 8:].strip() if end >= 0 else ""
    return cleaned or text


def query_model(model_id, question, is_thinking=False, http=None):
    """Query OpenRouter or local Ollama. Returns dict with response, attempts, error."""
    is_local = model_id.startswith("ollama/")
    out = {"response": "", "attempts": 0, "status_code": None, "error": None}
    own_http = http is None
    if own_http:
        http = httpx.Client(timeout=120)

    if is_local:
        api_url = OLLAMA_BASE
        actual_id = model_id[len("ollama/"):]
    else:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        actual_id = model_id

    payload = {
        "model": actual_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    if is_thinking and not is_local:
        payload["reasoning"] = {"effort": "low"}
    if not is_local:
        # Override account-level provider allowlist by explicitly listing a wide
        # set of providers. Without this, models hosted only on Together/Phala/
        # Atlas/Cloudflare/Mistral get HTTP 404 due to a default account allowlist.
        payload["provider"] = {
            "only": [
                "together", "phala", "atlas-cloud", "novita", "fireworks",
                "cloudflare", "mistral", "lambda", "kimi-k2", "alibaba",
                "deepinfra", "siliconflow", "deepseek", "anthropic", "openai",
                "google-vertex", "google-ai-studio", "nebius", "xiaomi",
                "x-ai", "xai", "groq", "perplexity", "cohere", "mancer", "azure",
                "amazon-bedrock", "infermatic", "openinference", "minimax",
                "sambanova", "hyperbolic", "chutes", "nextbit",
                "parasail", "crusoe", "targon", "ubicloud", "featherless",
                "moonshotai", "stepfun", "zhipu", "01ai", "baichuan",
                "ai21", "inflection", "reka", "mancer", "openrouter",
                "openchat", "deepseek-cloud", "moonshot", "tencent",
                "seed", "z-ai", "bytedance", "volcengine", "nineteen",
                "google", "meta", "microsoft", "ibm", "amazon",
            ],
            "allow_fallbacks": True,
        }

    headers = {"Content-Type": "application/json"}
    if not is_local:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

    try:
        for attempt in range(3):
            out["attempts"] = attempt + 1
            try:
                r = http.post(api_url, headers=headers, json=payload, timeout=120)
                out["status_code"] = r.status_code
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    if is_thinking:
                        content = strip_thinking(content)
                    out["response"] = content
                    return out
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                else:
                    out["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    return out
            except Exception as e:
                out["error"] = f"exception: {e}"
                time.sleep(2)
        return out
    finally:
        if own_http:
            http.close()


def evaluate_model_on_researcher(model_name, model_info, researcher_probes, enriched_by_pid,
                                  results_dir):
    """Evaluate a single model on the 345 researcher probes."""
    model_id = model_info["id"]
    is_local = model_id.startswith("ollama/")
    is_thinking = model_info.get("thinking", False)
    out_file = results_dir / f"{model_name}.json"

    # Skip if already done with the new schema (has CORRECT_STRONG/WEAK)
    # AND not silently failed via provider 404s.
    if out_file.exists():
        try:
            existing = json.load(open(out_file))
            results = existing.get("results", [])
            if (existing.get("schema_version") == "v2_evidence"
                and len(results) == len(researcher_probes)):
                # Sanity: detect catastrophic failure (all-error, all-empty)
                error_count = sum(1 for r in results if r.get("model_query_error"))
                if error_count >= len(results) * 0.95:
                    log(f"  {model_name}: existing file is failed (errors={error_count}/{len(results)}), re-evaluating")
                else:
                    log(f"  {model_name}: already done, skipping")
                    return existing
        except Exception:
            pass

    n_workers = OLLAMA_REQUESTS_PER_MODEL if is_local else OPENROUTER_REQUESTS_PER_MODEL
    log(f"  Starting {model_name} (n={len(researcher_probes)}, workers={n_workers})")

    def eval_one(probe):
        q = probe["question"]
        pid = probe["id"]
        gold = enriched_by_pid.get(pid, {})

        # 1. Query model
        with httpx.Client(timeout=120) as http:
            mres = query_model(model_id, q, is_thinking=is_thinking, http=http)
        response = mres["response"]

        # 2. Judge with evidence-aware judge
        gold_for_judge = dict(gold)
        gold_for_judge["name"] = gold.get("name") or probe.get("researcher_name", "")
        jres = judge_evidence(q, response, gold_for_judge)
        verdict = jres["verdict"]
        no_cs_match = bool(gold.get("no_cs_match"))
        score = score_4way(verdict, lam=-1.0, no_cs_match=no_cs_match)

        return {
            "probe_id": pid,
            "tier": probe["tier"],
            "source_type": probe.get("source_type", "researcher"),
            "domain": probe.get("domain", "computer_science"),
            "question": q,
            "researcher_name": gold.get("name"),
            "no_cs_match": no_cs_match,
            "model_response": (response or "")[:600],
            "verdict": verdict,
            "score": score,
            "judge_raw": jres.get("judge_raw_output"),
            "model_query_status": mres["status_code"],
            "model_query_error": mres["error"],
        }

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(eval_one, p): p for p in researcher_probes}
        done = 0
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                results.append(rec)
            except Exception as e:
                log(f"    {model_name}: probe exception: {e}")
            done += 1
            if done % 50 == 0:
                log(f"    {model_name}: {done}/{len(researcher_probes)}")

    # Aggregate by tier
    from collections import defaultdict
    tier_stats = defaultdict(lambda: {"strong": 0, "weak": 0, "wrong": 0, "refusal": 0,
                                      "no_cs_match_correct": 0, "total": 0})
    for r in results:
        t = r["tier"]
        v = r["verdict"]
        tier_stats[t]["total"] += 1
        if r["no_cs_match"]:
            if v == "REFUSAL":
                tier_stats[t]["no_cs_match_correct"] += 1
            else:
                tier_stats[t]["wrong"] += 1
        else:
            if v == "CORRECT_STRONG": tier_stats[t]["strong"] += 1
            elif v == "CORRECT_WEAK": tier_stats[t]["weak"] += 1
            elif v == "REFUSAL":      tier_stats[t]["refusal"] += 1
            else:                     tier_stats[t]["wrong"] += 1

    # Per-tier mean score (penalty=-1.0)
    tier_score = {}
    for t in ["T3", "T4", "T5", "T6", "T7"]:
        s = tier_stats.get(t, {"total": 0})
        if s["total"] == 0:
            tier_score[t] = 0.0
        else:
            score_sum = sum(r["score"] for r in results if r["tier"] == t)
            tier_score[t] = score_sum / s["total"]

    overall_score = sum(r["score"] for r in results) / max(len(results), 1)
    out = {
        "schema_version": "v2_evidence",
        "model_name": model_name,
        "model_id": model_id,
        "params_B": model_info.get("params_B"),
        "family": model_info.get("family"),
        "vendor": model_info.get("vendor"),
        "thinking": is_thinking,
        "judge_model": JUDGE_MODEL_DEFAULT,
        "hallucination_penalty": -1.0,
        "tier_stats": dict(tier_stats),
        "tier_score": tier_score,
        "overall_score": overall_score,
        "results": results,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  {model_name}: overall_score={overall_score:.3f}  "
        f"tier_score={ {t: round(v,3) for t,v in tier_score.items()} }")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="Subset of model names to run; empty = all")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit probes for smoke testing")
    ap.add_argument("--stratified", type=int, default=None,
                    help="Stratified per-tier sample for smoke testing (e.g. 6 = 6 per tier)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROBES_PATH) as f:
        all_probes = json.load(f)
    with open(ENRICHED_PATH) as f:
        enriched = json.load(f)
    enriched_by_pid = {r["probe_id"]: r for r in enriched}

    researcher_probes = [p for p in all_probes if p.get("source_type") == "researcher"]
    if args.stratified:
        from collections import defaultdict
        import random
        by_tier = defaultdict(list)
        for p in researcher_probes: by_tier[p["tier"]].append(p)
        random.seed(42)
        sampled = []
        # Always include all 16 T7 substitutions for verification
        sub_pids = {p["id"] for p in researcher_probes if p.get("_replaced_via_t7_substitution")}
        for t in ["T3","T4","T5","T6","T7"]:
            sub_in_tier = [p for p in by_tier[t] if p["id"] in sub_pids]
            others = [p for p in by_tier[t] if p["id"] not in sub_pids]
            n_other = max(0, args.stratified - len(sub_in_tier))
            sampled.extend(sub_in_tier)
            sampled.extend(random.sample(others, min(n_other, len(others))))
        researcher_probes = sampled
    elif args.limit:
        researcher_probes = researcher_probes[:args.limit]
    log(f"Loaded {len(researcher_probes)} researcher probes")

    with open(CONFIG_PATH) as f:
        models = json.load(f)["models"]
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    log(f"Evaluating {len(models)} models")

    # Split: openrouter (parallel models) vs ollama (serial models)
    openrouter_models = [(k, v) for k, v in models.items() if not v["id"].startswith("ollama/")]
    ollama_models = [(k, v) for k, v in models.items() if v["id"].startswith("ollama/")]
    log(f"  OpenRouter: {len(openrouter_models)}; Ollama: {len(ollama_models)}")

    summaries = {}
    summaries_lock = __import__("threading").Lock()

    def store_summary(mn, res):
        if not res: return
        with summaries_lock:
            summaries[mn] = {
                "overall_score": res["overall_score"],
                "tier_score": res["tier_score"],
                "params_B": res["params_B"],
                "family": res["family"],
                "vendor": res["vendor"],
                "thinking": res["thinking"],
            }
            with open(SUMMARY, "w") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)

    # Run OpenRouter and Ollama IN PARALLEL.
    # OpenRouter: 16 models concurrent × 8 requests each.
    # Ollama: 1 model at a time (serial models to avoid GPU contention)
    #         × 16 requests per model.
    log(f"\n=== Launching OpenRouter ({OPENROUTER_MODELS_CONCURRENT} concurrent × {OPENROUTER_REQUESTS_PER_MODEL} requests) ===")
    log(f"=== AND Ollama (serial models × {OLLAMA_REQUESTS_PER_MODEL} requests) IN PARALLEL ===")

    openrouter_pool = ThreadPoolExecutor(max_workers=OPENROUTER_MODELS_CONCURRENT,
                                          thread_name_prefix="openrouter")
    ollama_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama")

    def run_one(mn, mi):
        try:
            res = evaluate_model_on_researcher(mn, mi, researcher_probes,
                                                enriched_by_pid, OUT_DIR)
            store_summary(mn, res)
        except Exception as e:
            log(f"  ERR {mn}: {e}")

    futs = []
    for mn, mi in openrouter_models:
        futs.append(openrouter_pool.submit(run_one, mn, mi))
    for mn, mi in ollama_models:
        futs.append(ollama_pool.submit(run_one, mn, mi))

    # Wait for everything
    n_done = 0
    for fut in as_completed(futs):
        n_done += 1
        if n_done % 5 == 0:
            log(f"  Progress: {n_done}/{len(futs)} models done")
    openrouter_pool.shutdown(wait=True)
    ollama_pool.shutdown(wait=True)

    log(f"\n=== Done. Summaries: {len(summaries)} models ===")


if __name__ == "__main__":
    main()
