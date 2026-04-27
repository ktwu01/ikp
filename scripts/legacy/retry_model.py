#!/usr/bin/env python3
"""Retry evaluation for specific models without touching other results.

Usage:
  python scripts/retry_model.py kimi-k2           # retry entire model
  python scripts/retry_model.py qwq-32b-think     # retry with fix
  python scripts/retry_model.py --list-anomalies   # find models needing retry
"""

import json
import sys
import os
import re
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."


def strip_thinking(text):
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return cleaned or text


def query_model(model_id, question, is_thinking=False):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    if is_thinking:
        payload["reasoning"] = {"effort": "low"}

    with httpx.Client(timeout=120) as http:
        for attempt in range(3):
            try:
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload)
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    if is_thinking and content:
                        content = strip_thinking(content)
                    return content
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                else:
                    return ""
            except:
                time.sleep(2)
    return ""


def judge_response(question, gold, response):
    if not response or not response.strip():
        return False
    lower = response.lower().strip()
    for phrase in ["i don't know", "i do not know", "i'm not sure", "i cannot", "i can't", "not aware"]:
        if phrase in lower:
            return False

    prompt = f"""You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold}
Model's response: {response}

Strict rules:
1. If the correct answer is a YEAR, the model must state that EXACT year. A different year is INCORRECT.
2. If the correct answer is a NUMBER, the numeric value must match exactly or be very close (within 1-2%). Ignore formatting differences like commas, spaces, units, or words like "approximately" — only the numeric value matters (e.g. "approximately 299,792,458 m/s" matches "299792458").
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling OK.
4. If the correct answer is a RESEARCH FIELD, accept synonyms (e.g. "networking" = "computer networking") but reject unrelated fields.
5. If the model refuses or doesn't know, that is INCORRECT.
6. If the model gives a DIFFERENT answer, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""

    with httpx.Client(timeout=60) as http:
        for attempt in range(3):
            try:
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": "google/gemini-3-flash-preview",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0,
                          "reasoning": {"effort": "low"}})
                if r.status_code == 200:
                    result = r.json()["choices"][0]["message"]["content"].strip().upper()
                    return result == "CORRECT" or (result.startswith("CORRECT") and not result.startswith("INCORRECT"))
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
            except:
                time.sleep(1)
    return False


def find_anomalies():
    """Find models with suspicious results."""
    results_dir = PROJECT_ROOT / "data" / "results"
    anomalies = []

    for f in sorted(results_dir.glob("*.json")):
        if f.name in ["evaluation_summary.json", "analysis.json"]:
            continue
        try:
            d = json.load(open(f))
            if "results" not in d:
                continue

            ta = d.get("tier_accuracy", {})
            name = d.get("model_name", f.stem)

            # All zeros
            if all(ta.get(t, 0) == 0 for t in ["T1","T2","T3","T4","T5","T6","T7"]):
                anomalies.append((name, "all_zero", "0% on every tier"))
                continue

            # High refusal rate
            refusals = sum(1 for r in d["results"] if r.get("refusal"))
            if refusals > len(d["results"]) * 0.5:
                anomalies.append((name, "high_refusal", f"{refusals}/{len(d['results'])} refusals"))

            # Non-monotonic tier accuracy (T(n) >> T(n-1))
            for i, (t1, t2) in enumerate(zip(["T1","T2","T3","T4","T5","T6"], ["T2","T3","T4","T5","T6","T7"])):
                a1 = ta.get(t1, 0)
                a2 = ta.get(t2, 0)
                if a2 > a1 + 0.1 and a1 < 0.5:  # next tier much higher than current
                    anomalies.append((name, "non_monotonic", f"{t1}={a1:.0%} but {t2}={a2:.0%}"))
                    break

            # Sudden drop to 0 in middle
            for t in ["T2", "T3", "T4"]:
                if ta.get(t, 0) == 0 and ta.get("T1", 0) > 0.5:
                    tier_results = [r for r in d["results"] if r["tier"] == t]
                    tier_refusals = sum(1 for r in tier_results if r.get("refusal"))
                    if tier_refusals > len(tier_results) * 0.8:
                        anomalies.append((name, "tier_dropout", f"{t}=0% ({tier_refusals} refusals)"))
                        break

        except Exception as e:
            anomalies.append((f.stem, "error", str(e)[:50]))

    return anomalies


def retry_model(model_name):
    """Re-evaluate a single model."""
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    models = config["models"]

    if model_name not in models:
        print(f"Model '{model_name}' not found in config")
        return

    model_info = models[model_name]
    model_id = model_info["id"]
    is_thinking = model_info.get("thinking", False)
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "final_probe_set_v7.json"))

    result_file = PROJECT_ROOT / "data" / "results" / f"{model_name}.json"

    print(f"Retrying {model_name} ({model_id}) thinking={is_thinking}")
    print(f"  {len(probes)} probes, 16 workers", flush=True)

    results = []
    correct_count = 0

    def eval_one(probe):
        q = probe["question"]
        gold = probe["answer"]
        response = query_model(model_id, q, is_thinking)
        correct = judge_response(q, gold, response) if response else False
        return {
            "probe_id": probe.get("id", ""),
            "tier": probe["tier"],
            "source_type": probe.get("source_type", ""),
            "question": q,
            "gold_answer": gold,
            "correct": correct,
            "refusal": not response or not response.strip(),
            "response": (response or "")[:300],
        }

    done = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(eval_one, p): p for p in probes}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["correct"]:
                correct_count += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(probes)}, {correct_count} correct", flush=True)

    # Compute stats
    from collections import defaultdict
    tier_stats = defaultdict(lambda: {"correct": 0, "total": 0, "refusal": 0})
    for r in results:
        t = r["tier"]
        if r["refusal"]:
            tier_stats[t]["refusal"] += 1
        else:
            tier_stats[t]["total"] += 1
            if r["correct"]:
                tier_stats[t]["correct"] += 1

    tier_accs = {}
    for t in ["T1","T2","T3","T4","T5","T6","T7"]:
        s = tier_stats[t]
        tier_accs[t] = s["correct"] / s["total"] if s["total"] > 0 else 0.0

    accuracy = correct_count / len(probes)

    output = {
        "model_name": model_name,
        "model_id": model_id,
        "params_B": model_info.get("params_B"),
        "family": model_info.get("family"),
        "vendor": model_info.get("vendor"),
        "arch": model_info.get("arch"),
        "thinking": is_thinking,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(probes),
        "tier_accuracy": tier_accs,
        "tier_stats": dict(tier_stats),
        "results": results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    tier_str = " ".join(f"{t}={tier_accs[t]:.0%}" for t in ["T1","T2","T3","T4","T5","T6","T7"])
    print(f"\n  {model_name}: {accuracy:.1%} ({correct_count}/{len(probes)})")
    print(f"  Per-tier: {tier_str}")
    print(f"  Saved to {result_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list-anomalies":
        anomalies = find_anomalies()
        if anomalies:
            print("Models with anomalies:")
            for name, atype, detail in anomalies:
                print(f"  {name:35s} [{atype:15s}] {detail}")
        else:
            print("No anomalies found.")
    else:
        for model_name in sys.argv[1:]:
            retry_model(model_name)
