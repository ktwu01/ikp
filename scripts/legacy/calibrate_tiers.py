#!/usr/bin/env python3
"""Calibrate tier boundaries by running probes against known-size models.

Tests 140 calibration probes (20 per tier) against 7 models spanning
0.5B to frontier. The results tell us whether our tier assignments
(sitelink counts, citation counts, paper counts) actually match
the models' knowledge boundaries.
"""

import json
import sys
import time
import logging
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from api_client import OpenRouterClient
from scorer import score_with_llm_judge, create_openrouter_judge, is_refusal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def query_ollama(model_name: str, prompt: str, system: str = "") -> str:
    """Query a local Ollama model."""
    http = httpx.Client(timeout=120)
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system} if system else None,
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        payload["messages"] = [m for m in payload["messages"] if m]

        r = http.post("http://localhost:11434/api/chat", json=payload)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "")
        return ""
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return ""
    finally:
        http.close()


def run_probe(probe, model_name, model_type, openrouter_client=None, judge_fn=None):
    """Run a single probe against a model and score the result."""
    system_msg = (
        "You are answering factual knowledge questions. "
        "Give a direct, concise answer. "
        "If the question asks for a number, give just the number. "
        "If it asks for a name or field, give just the name or field. "
        "If you genuinely do not know, say 'I don't know'. "
        "Think carefully before answering."
    )

    question = probe["question_direct"]
    gold = probe["answer"]
    answer_type = probe.get("answer_type", "auto")
    is_researcher = "researcher_name" in probe

    # Get response
    if model_type == "ollama":
        response = query_ollama(model_name, question, system_msg)
    else:
        response = openrouter_client.get_response_text(
            model_name,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=200,
        )

    # Score using LLM-as-judge
    refusal = is_refusal(response)
    if refusal:
        correct = False
    elif judge_fn:
        correct = score_with_llm_judge(question, gold, response, judge_fn)
    else:
        correct = False

    return {
        "probe_id": probe["id"],
        "tier": probe["tier"],
        "question": question,
        "gold_answer": gold,
        "response": response[:300] if response else "",
        "correct": correct,
        "refusal": refusal,
        "is_researcher": is_researcher,
        "is_wikidata": "wikidata_id" in probe,
    }


def main():
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "calibration_sample.json"))
    logger.info(f"Loaded {len(probes)} calibration probes")

    # Models to test — spanning 0.5B to frontier
    models = [
        ("qwen2.5:0.5b", "ollama", 0.5),
        ("qwen3:4b", "ollama", 4.0),
        ("qwen/qwen-2.5-7b-instruct", "openrouter", 7.6),
        ("qwen/qwen3-32b", "openrouter", 32.0),
        ("qwen/qwen-2.5-72b-instruct", "openrouter", 72.7),
        ("qwen/qwen3-235b-a22b", "openrouter", 235.0),       # Large MoE
        ("moonshotai/kimi-k2.5", "openrouter", None),          # ~1T frontier
        ("google/gemini-3.1-pro-preview", "openrouter", None),  # Top frontier
    ]

    or_client = OpenRouterClient(requests_per_minute=50, max_retries=3, timeout=120)
    judge_fn = create_openrouter_judge(or_client, model="anthropic/claude-sonnet-4.6")

    all_results = {}

    for model_id, model_type, params_B in models:
        model_label = model_id.split("/")[-1] if "/" in model_id else model_id
        logger.info(f"\n{'='*60}")
        logger.info(f"  {model_label} ({params_B}B, {model_type})")
        logger.info(f"{'='*60}")

        results = []
        for i, probe in enumerate(probes):
            try:
                result = run_probe(
                    probe, model_id, model_type,
                    openrouter_client=or_client if model_type == "openrouter" else None,
                    judge_fn=judge_fn,
                )
                results.append(result)

                if (i + 1) % 20 == 0:
                    tier_acc = {}
                    for r in results:
                        t = r["tier"]
                        if t not in tier_acc:
                            tier_acc[t] = {"correct": 0, "total": 0, "refusal": 0}
                        if r["refusal"]:
                            tier_acc[t]["refusal"] += 1
                        else:
                            tier_acc[t]["total"] += 1
                            if r["correct"]:
                                tier_acc[t]["correct"] += 1

                    logger.info(f"  {i+1}/{len(probes)} done")
                    for t in sorted(tier_acc.keys()):
                        s = tier_acc[t]
                        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
                        logger.info(f"    {t}: {acc:.0%} ({s['correct']}/{s['total']}, {s['refusal']} refusals)")

            except Exception as e:
                logger.warning(f"  Error on probe {probe['id']}: {e}")
                results.append({
                    "probe_id": probe["id"], "tier": probe["tier"],
                    "correct": False, "refusal": True, "error": str(e),
                })

        all_results[model_label] = {
            "params_B": params_B,
            "model_type": model_type,
            "results": results,
        }

    # Save results
    output = PROJECT_ROOT / "data" / "calibration" / "tier_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    logger.info(f"\n{'='*90}")
    logger.info(f"  TIER CALIBRATION RESULTS")
    logger.info(f"{'='*90}")

    header = f"{'Model':25s} {'Params':>7s}"
    for tier in ["T1","T2","T3","T4","T5","T6","T7"]:
        header += f" {tier:>6s}"
    header += f" {'Agg':>6s}"
    logger.info(header)
    logger.info("-" * 90)

    for model_label, data in sorted(all_results.items(), key=lambda x: x[1].get("params_B") or 9999):
        params = data["params_B"]
        pstr = f"{params:.0f}B" if params else "?"

        tier_accs = {}
        for r in data["results"]:
            t = r["tier"]
            if t not in tier_accs:
                tier_accs[t] = {"c": 0, "n": 0}
            if not r.get("refusal"):
                tier_accs[t]["n"] += 1
                if r.get("correct"):
                    tier_accs[t]["c"] += 1

        row = f"{model_label:25s} {pstr:>7s}"
        accs = []
        for tier in ["T1","T2","T3","T4","T5","T6","T7"]:
            s = tier_accs.get(tier, {"c": 0, "n": 0})
            acc = s["c"] / s["n"] if s["n"] > 0 else 0
            accs.append(acc)
            row += f" {acc:6.0%}"
        agg = sum(accs) / len(accs)
        row += f" {agg:6.0%}"
        logger.info(row)

    # Per-source analysis
    logger.info(f"\n  PER-SOURCE BREAKDOWN:")
    for source_type in ["llm", "wikidata", "researcher"]:
        logger.info(f"\n  Source: {source_type}")
        for model_label, data in sorted(all_results.items(), key=lambda x: x[1].get("params_B") or 9999):
            params = data["params_B"]
            pstr = f"{params:.0f}B" if params else "?"

            tier_accs = {}
            for r in data["results"]:
                # Filter by source
                if source_type == "researcher" and not r.get("is_researcher"):
                    continue
                if source_type == "wikidata" and not r.get("is_wikidata"):
                    continue
                if source_type == "llm" and (r.get("is_researcher") or r.get("is_wikidata")):
                    continue

                t = r["tier"]
                if t not in tier_accs:
                    tier_accs[t] = {"c": 0, "n": 0}
                if not r.get("refusal"):
                    tier_accs[t]["n"] += 1
                    if r.get("correct"):
                        tier_accs[t]["c"] += 1

            row = f"    {model_label:23s} {pstr:>5s}"
            for tier in ["T1","T2","T3","T4","T5","T6","T7"]:
                s = tier_accs.get(tier, {"c": 0, "n": 0})
                if s["n"] > 0:
                    acc = s["c"] / s["n"]
                    row += f" {acc:5.0%}"
                else:
                    row += f"    —"
            logger.info(row)


if __name__ == "__main__":
    main()
