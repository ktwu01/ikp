"""Patch failed probes (status != 200) in existing v2 result files.

Re-evaluates only the failed probes with low concurrency and long backoff
so we don't trigger fresh rate limits. Leaves successful (200 OK) results
unchanged.

Usage:
  python3 scripts/patch_429_failures.py --models deepseek-v4-pro deepseek-v4-pro-think
"""

import argparse
import json
import os
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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."

# Conservative settings to avoid 429
N_WORKERS = 2  # very conservative
MAX_RETRIES = 10
INITIAL_BACKOFF = 5.0


def query_model_robust(model_id, question, is_thinking=False):
    """Query OpenRouter with aggressive 429 handling."""
    out = {"response": "", "attempts": 0, "status_code": None, "error": None}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "provider": {
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
                "ai21", "inflection", "reka", "openrouter",
                "openchat", "deepseek-cloud", "moonshot", "tencent",
                "seed", "z-ai", "bytedance", "volcengine", "nineteen",
                "google", "meta", "microsoft", "ibm", "amazon",
            ],
            "allow_fallbacks": True,
        },
    }
    if is_thinking:
        payload["reasoning"] = {"effort": "low"}

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}",
               "Content-Type": "application/json"}

    with httpx.Client(timeout=180) as http:
        for attempt in range(MAX_RETRIES):
            out["attempts"] = attempt + 1
            try:
                r = http.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=180)
                out["status_code"] = r.status_code
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    out["response"] = content
                    return out
                elif r.status_code == 429:
                    backoff = INITIAL_BACKOFF * (2 ** attempt)  # 5, 10, 20, 40, 80, ...
                    backoff = min(backoff, 300)
                    print(f"    429, sleeping {backoff:.0f}s (attempt {attempt+1}/{MAX_RETRIES})", flush=True)
                    time.sleep(backoff)
                else:
                    out["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    return out
            except Exception as e:
                out["error"] = f"exception: {e}"
                time.sleep(5)
    return out


def patch_one(model_name, model_info, all_probes_by_pid, enriched_by_pid):
    """Patch failed probes in the existing v2 result file."""
    out_file = OUT_DIR / f"{model_name}.json"
    if not out_file.exists():
        print(f"  [SKIP] {model_name}: no v2 result file")
        return
    with open(out_file) as f:
        existing = json.load(f)

    failed = [r for r in existing["results"]
              if r.get("model_query_status") != 200
              or r.get("model_query_error")]
    print(f"  {model_name}: {len(failed)} failed probes to patch")
    if not failed:
        return

    model_id = model_info["id"]
    is_thinking = model_info.get("thinking", False)

    def patch_one_probe(rec):
        pid = rec["probe_id"]
        question = rec["question"]
        gold = enriched_by_pid.get(pid, {})

        # Re-query with robust retries
        mres = query_model_robust(model_id, question, is_thinking=is_thinking)
        response = mres["response"]
        if not response.strip():
            return rec  # still failed, leave as is

        # Re-judge
        gold_for_judge = dict(gold)
        gold_for_judge["name"] = gold.get("name") or rec.get("researcher_name", "")
        jres = judge_evidence(question, response, gold_for_judge)
        verdict = jres["verdict"]
        no_cs_match = bool(gold.get("no_cs_match"))
        score = score_4way(verdict, lam=-1.0, no_cs_match=no_cs_match)

        new_rec = {
            "probe_id": pid,
            "tier": rec["tier"],
            "source_type": rec.get("source_type", "researcher"),
            "domain": rec.get("domain"),
            "question": question,
            "researcher_name": rec.get("researcher_name"),
            "no_cs_match": no_cs_match,
            "model_response": (response or "")[:600],
            "verdict": verdict,
            "score": score,
            "judge_raw": jres.get("judge_raw_output"),
            "model_query_status": mres["status_code"],
            "model_query_error": mres["error"],
        }
        return new_rec

    patched_results = list(existing["results"])
    pid_to_idx = {r["probe_id"]: i for i, r in enumerate(patched_results)}

    n_recovered = 0
    n_still_failed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(patch_one_probe, r): r["probe_id"] for r in failed}
        done = 0
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                new_rec = fut.result()
                idx = pid_to_idx[pid]
                if new_rec.get("model_query_status") == 200 and new_rec.get("model_response"):
                    patched_results[idx] = new_rec
                    n_recovered += 1
                else:
                    n_still_failed += 1
            except Exception as e:
                print(f"    patch err {pid}: {e}")
                n_still_failed += 1
            done += 1
            if done % 20 == 0:
                print(f"    {model_name}: {done}/{len(failed)} patched (recovered={n_recovered})", flush=True)

    print(f"  {model_name}: recovered {n_recovered}/{len(failed)}, still failed {n_still_failed}")

    # Recompute aggregate
    from collections import defaultdict
    tier_stats = defaultdict(lambda: {"strong":0,"weak":0,"wrong":0,"refusal":0,
                                      "no_cs_match_correct":0,"total":0})
    for r in patched_results:
        t = r["tier"]
        v = r["verdict"]
        tier_stats[t]["total"] += 1
        if r.get("no_cs_match"):
            if v == "REFUSAL": tier_stats[t]["no_cs_match_correct"] += 1
            else: tier_stats[t]["wrong"] += 1
        else:
            if v == "CORRECT_STRONG": tier_stats[t]["strong"] += 1
            elif v == "CORRECT_WEAK": tier_stats[t]["weak"] += 1
            elif v == "REFUSAL": tier_stats[t]["refusal"] += 1
            else: tier_stats[t]["wrong"] += 1

    tier_score = {}
    for t in ["T3","T4","T5","T6","T7"]:
        rs = [r for r in patched_results if r["tier"] == t]
        if not rs:
            tier_score[t] = 0.0
        else:
            tier_score[t] = sum(r.get("score", 0) for r in rs) / len(rs)

    overall_score = sum(r.get("score", 0) for r in patched_results) / max(len(patched_results), 1)

    existing["results"] = patched_results
    existing["tier_stats"] = dict(tier_stats)
    existing["tier_score"] = tier_score
    existing["overall_score"] = overall_score
    existing["patched_at"] = datetime.now(timezone.utc).isoformat()
    existing["n_patched"] = n_recovered

    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  {model_name}: new overall_score={overall_score:.4f}, "
          f"tier_score={ {t: round(v,3) for t,v in tier_score.items()} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    with open(PROBES_PATH) as f:
        probes = json.load(f)
    all_probes_by_pid = {p["id"]: p for p in probes}
    with open(ENRICHED_PATH) as f:
        enriched = json.load(f)
    enriched_by_pid = {r["probe_id"]: r for r in enriched}
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)["models"]

    for model_name in args.models:
        if model_name not in cfg:
            print(f"  [WARN] {model_name} not in config")
            continue
        print(f"\n=== Patching {model_name} ===")
        patch_one(model_name, cfg[model_name], all_probes_by_pid, enriched_by_pid)


if __name__ == "__main__":
    main()
