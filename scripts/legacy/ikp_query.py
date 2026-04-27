#!/usr/bin/env python3
"""IKP Query CLI — probe models with factual questions.

Query landmark models and flagship models to gauge how well-known a fact,
researcher, or entity is across the LLM knowledge spectrum.

Usage:
  # Default: query all landmark + flagship models
  python scripts/ikp_query.py "What is the capital of France?"

  # With gold answer for automatic judging
  python scripts/ikp_query.py --gold Paris "What is the capital of France?"

  # Researcher shorthand
  python scripts/ikp_query.py --researcher "Bojie Li"
  python scripts/ikp_query.py --researcher "Bojie Li" --gold "computer networking"

  # Founding year shorthand
  python scripts/ikp_query.py --founding "Jaume I University"
  python scripts/ikp_query.py --founding "Jaume I University" --gold 1991

  # Pick specific models
  python scripts/ikp_query.py -m gpt-4.1 -m claude-opus-4.6 "Who wrote Faust?"

  # Landmarks only (skip flagships)
  python scripts/ikp_query.py --landmarks-only "What is the capital of France?"

  # Sequential mode (no parallelism, for rate-limited accounts)
  python scripts/ikp_query.py --sequential "What is the capital of France?"

  # List available models
  python scripts/ikp_query.py --list
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."
JUDGE_MODEL = "google/gemini-3-flash-preview"

# ── Landmark models (define tier boundaries) ──────────────────
LANDMARKS = [
    {"name": "qwen2.5-0.5b",   "id": "qwen2.5:0.5b",                 "type": "ollama",     "thinking": False, "tier": "T1", "params": "0.5B"},
    {"name": "qwen2.5-7b",     "id": "qwen/qwen-2.5-7b-instruct",    "type": "openrouter", "thinking": False, "tier": "T2", "params": "7.6B"},
    {"name": "qwen3-32b",      "id": "qwen/qwen3-32b",                "type": "openrouter", "thinking": True,  "tier": "T3", "params": "32B"},
    {"name": "qwen3-235b",     "id": "qwen/qwen3-235b-a22b",          "type": "openrouter", "thinking": True,  "tier": "T4", "params": "235B"},
    {"name": "kimi-k2.5",      "id": "moonshotai/kimi-k2.5",          "type": "openrouter", "thinking": True,  "tier": "T5", "params": "1040B"},
    {"name": "gemini-3.1-pro", "id": "google/gemini-3.1-pro-preview",  "type": "openrouter", "thinking": False, "tier": "T6", "params": "~5T?"},
]

# ── Flagship models (latest from top vendors) ────────────────
FLAGSHIPS = [
    {"name": "gpt-5.4-think",   "id": "openai/gpt-5.4",                "type": "openrouter", "thinking": True,  "params": "?"},
    {"name": "claude-opus-4.6", "id": "anthropic/claude-opus-4.6",     "type": "openrouter", "thinking": False, "params": "?"},
    {"name": "gemini-3-flash",  "id": "google/gemini-3-flash-preview",  "type": "openrouter", "thinking": False, "params": "?"},
    {"name": "grok-4",          "id": "x-ai/grok-4",                    "type": "openrouter", "thinking": False, "params": "?"},
]

ALL_MODELS = {m["name"]: m for m in LANDMARKS + FLAGSHIPS}

# ── Additional well-known models users can select ────────────
EXTRA_MODELS = {
    "gpt-3.5-turbo":       {"id": "openai/gpt-3.5-turbo",            "type": "openrouter", "thinking": False},
    "gpt-4":               {"id": "openai/gpt-4",                     "type": "openrouter", "thinking": False},
    "gpt-4o":              {"id": "openai/gpt-4o",                    "type": "openrouter", "thinking": False},
    "gpt-4o-mini":         {"id": "openai/gpt-4o-mini",              "type": "openrouter", "thinking": False},
    "gpt-5":               {"id": "openai/gpt-5",                     "type": "openrouter", "thinking": False},
    "gpt-5.4":             {"id": "openai/gpt-5.4",                   "type": "openrouter", "thinking": False},
    "o3":                  {"id": "openai/o3",                         "type": "openrouter", "thinking": True},
    "o3-pro":              {"id": "openai/o3-pro",                     "type": "openrouter", "thinking": True},
    "claude-3-haiku":      {"id": "anthropic/claude-3-haiku-20240307", "type": "openrouter", "thinking": False},
    "claude-3.5-haiku":    {"id": "anthropic/claude-3-5-haiku-20241022","type": "openrouter","thinking": False},
    "claude-sonnet-4":     {"id": "anthropic/claude-sonnet-4",         "type": "openrouter", "thinking": False},
    "claude-sonnet-4.6":   {"id": "anthropic/claude-sonnet-4-6",       "type": "openrouter", "thinking": False},
    "claude-opus-4.5":     {"id": "anthropic/claude-opus-4-5",         "type": "openrouter", "thinking": False},
    "gemini-2.5-flash":    {"id": "google/gemini-2.5-flash-preview-04-17","type":"openrouter","thinking": False},
    "gemini-2.5-pro":      {"id": "google/gemini-2.5-pro-preview",     "type": "openrouter", "thinking": False},
    "deepseek-v3":         {"id": "deepseek/deepseek-chat",            "type": "openrouter", "thinking": False},
    "deepseek-r1":         {"id": "deepseek/deepseek-reasoner",        "type": "openrouter", "thinking": True},
    "grok-3":              {"id": "x-ai/grok-3-beta",                  "type": "openrouter", "thinking": False},
    "grok-4":              {"id": "x-ai/grok-4",                       "type": "openrouter", "thinking": False},
    "llama-3.3-70b":       {"id": "meta-llama/llama-3.3-70b-instruct", "type": "openrouter", "thinking": False},
}
ALL_MODELS.update(EXTRA_MODELS)


# ── Query functions ───────────────────────────────────────────
def strip_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned.startswith('<think>'):
        end = cleaned.find('</think>')
        cleaned = cleaned[end + 8:].strip() if end >= 0 else ''
    return cleaned or text


def query_model(model: dict, question: str) -> str:
    if model["type"] == "ollama":
        return _query_ollama(model["id"], question)
    else:
        return _query_openrouter(model["id"], question, model.get("thinking", False))


def _query_ollama(model_id: str, question: str) -> str:
    try:
        with httpx.Client(timeout=120) as http:
            r = http.post("http://localhost:11434/v1/chat/completions", json={
                "model": model_id,
                "messages": [{"role": "system", "content": SYSTEM_MSG},
                             {"role": "user", "content": question}],
                "temperature": 0,
            })
            if r.status_code == 200:
                return r.json()["choices"][0]["message"].get("content", "")
    except Exception:
        pass
    return ""


def _query_openrouter(model_id: str, question: str, thinking: bool = False) -> str:
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": SYSTEM_MSG},
                     {"role": "user", "content": question}],
        "temperature": 0,
    }
    if thinking:
        payload["reasoning"] = {"effort": "medium"}

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
                    if thinking and content:
                        content = strip_thinking(content)
                    return content
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                return ""
            except Exception:
                time.sleep(1)
    return ""


def judge_response(question: str, gold: str, response: str) -> str:
    """Judge using Gemini 3 Flash. Returns CORRECT, WRONG, or REFUSAL."""
    if not response or not response.strip():
        return "REFUSAL"

    if ';' in gold:
        gold_display = ' OR '.join(a.strip() for a in gold.split(';'))
        co_note = "\nNOTE: Any of the listed names (alone or together) counts as CORRECT."
    else:
        gold_display = gold
        co_note = ""

    prompt = f"""You are a strict factual judge. Classify the model's response.

Question: {question}
Correct answer: {gold_display}
Model's response: {response}

Rules:
1. YEAR must match exactly. 2. NUMBER within 1-2%. 3. NAME: same entity, minor spelling OK, co-contributors OK.
4. RESEARCH FIELD: accept adjacent/related subfields, reject unrelated.
5. If the model refuses or doesn't know: REFUSAL.
6. If different/wrong answer: WRONG.
{co_note}
Reply with one word: CORRECT, REFUSAL, or WRONG"""

    try:
        with httpx.Client(timeout=60) as http:
            r = http.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": JUDGE_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "reasoning": {"effort": "low"}})
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"].strip().upper()
                if raw.startswith("CORRECT"): return "CORRECT"
                if raw.startswith("REFUSAL"): return "REFUSAL"
                return "WRONG"
    except Exception:
        pass
    return "UNKNOWN"


# ── Display ───────────────────────────────────────────────────
VERDICT_SYMBOLS = {
    "CORRECT": "\033[92m✓\033[0m",
    "WRONG":   "\033[91m✗\033[0m",
    "REFUSAL": "\033[93m?\033[0m",
    "UNKNOWN": "\033[90m-\033[0m",
}


def display_results(question: str, results: list, gold: str = None):
    """Pretty-print results table."""
    print()
    print(f"  Question: {question}")
    if gold:
        print(f"  Gold answer: {gold}")
    print()

    # Determine tier (from landmark responses)
    tier_result = "T7+"
    for lm in LANDMARKS:
        match = next((r for r in results if r["name"] == lm["name"]), None)
        if not match:
            continue
        resp = match.get("response", "")
        if not resp or not resp.strip():
            continue
        if gold:
            if match.get("verdict") == "CORRECT":
                tier_result = lm["tier"]
                break
        else:
            lower = resp.lower()
            if not any(p in lower for p in ["i don't know", "i'm not sure", "i cannot", "i'm unable"]):
                tier_result = lm["tier"]
                break

    max_name = max(len(r["name"]) for r in results)

    # Print landmarks first, then flagships, then others
    landmark_names = [m["name"] for m in LANDMARKS]
    flagship_names = [m["name"] for m in FLAGSHIPS]

    sections = [
        ("Landmarks", [r for r in results if r["name"] in landmark_names]),
        ("Flagships", [r for r in results if r["name"] in flagship_names]),
        ("Other",     [r for r in results if r["name"] not in landmark_names and r["name"] not in flagship_names]),
    ]

    for section_name, section_results in sections:
        if not section_results:
            continue
        print(f"  \033[90m── {section_name} ──\033[0m")
        for r in section_results:
            name = r["name"].ljust(max_name)
            resp = (r.get("response") or "").strip()
            resp_display = resp
            if not resp:
                resp_display = "\033[90m(no response)\033[0m"

            if gold and "verdict" in r:
                symbol = VERDICT_SYMBOLS.get(r["verdict"], "-")
                print(f"  {symbol} {name}  {r['verdict']:>8}  {resp_display}")
            else:
                # No gold: just show response
                print(f"    {name}  {resp_display}")

    # Summary
    print()
    if gold:
        correct = sum(1 for r in results if r.get("verdict") == "CORRECT")
        wrong = sum(1 for r in results if r.get("verdict") == "WRONG")
        refusal = sum(1 for r in results if r.get("verdict") == "REFUSAL")
        print(f"  Estimated tier: {tier_result}")
        print(f"  {correct} correct, {wrong} wrong, {refusal} refusal (out of {len(results)})")
    print()


# ── Progress indicator ────────────────────────────────────────
_print_lock = threading.Lock()
_completed_count = 0


def _progress_callback(model_name: str, total: int):
    global _completed_count
    with _print_lock:
        _completed_count += 1
        sys.stderr.write(f"\r  [{_completed_count}/{total}] {model_name} done")
        sys.stderr.flush()


# ── Main ──────────────────────────────────────────────────────
def main():
    global _completed_count

    parser = argparse.ArgumentParser(
        description="IKP Query CLI — probe LLMs with factual questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="?", help="The question to ask")
    parser.add_argument("--researcher", "-r", metavar="NAME",
                        help="Shorthand: 'In computer science, what is the research subfield of NAME?'")
    parser.add_argument("--founding", "-f", metavar="ENTITY",
                        help="Shorthand: 'In what year was ENTITY founded?'")
    parser.add_argument("--gold", "-g", metavar="ANSWER",
                        help="Gold answer for automatic judging")
    parser.add_argument("--models", "-m", action="append", metavar="MODEL",
                        help="Specific model(s) to query (can repeat)")
    parser.add_argument("--landmarks-only", "-L", action="store_true",
                        help="Only query landmark models")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Query all available models")
    parser.add_argument("--sequential", "-s", action="store_true",
                        help="Disable parallel queries (for rate-limited accounts)")
    parser.add_argument("--list", action="store_true",
                        help="List available models and exit")

    args = parser.parse_args()

    if args.list:
        print("\nLandmark models (define tier boundaries):")
        for m in LANDMARKS:
            print(f"  {m['name']:25s} {m['params']:>8s}  tier={m['tier']}")
        print("\nFlagship models (latest from top vendors):")
        for m in FLAGSHIPS:
            print(f"  {m['name']:25s} {m.get('params','?'):>8s}")
        print("\nAdditional models (use with -m):")
        for name in sorted(EXTRA_MODELS):
            print(f"  {name}")
        return

    # Build question
    if args.researcher:
        question = f"In computer science, what is the research subfield of {args.researcher}?"
    elif args.founding:
        question = f"In what year was {args.founding} founded?"
    elif args.question:
        question = args.question
    else:
        parser.print_help()
        return

    # Select models
    if args.models:
        selected = []
        for name in args.models:
            if name in ALL_MODELS:
                selected.append({"name": name, **ALL_MODELS[name]})
            else:
                print(f"Unknown model: {name}. Use --list to see options.")
                return
    elif args.all:
        selected = [{"name": n, **m} for n, m in ALL_MODELS.items()]
    elif args.landmarks_only:
        selected = list(LANDMARKS)
    else:
        selected = list(LANDMARKS) + list(FLAGSHIPS)

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    total = len(selected)
    gold = args.gold
    _completed_count = 0

    print(f"\n  Querying {total} models{'  (sequential)' if args.sequential else ''}...")

    # Phase 1: Query all models
    responses = {}  # name -> response text

    def query_one(model):
        resp = query_model(model, question)
        _progress_callback(model["name"], total)
        return model["name"], resp

    if args.sequential:
        for model in selected:
            name, resp = query_one(model)
            responses[name] = resp
    else:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {executor.submit(query_one, m): m for m in selected}
            for future in as_completed(futures):
                name, resp = future.result()
                responses[name] = resp

    sys.stderr.write("\r" + " " * 60 + "\r")  # Clear progress line
    sys.stderr.flush()

    # Phase 2: Judge all responses (if gold provided), also in parallel
    results = []
    if gold:
        to_judge = [(name, resp) for name, resp in responses.items()
                    if resp and resp.strip()]
        verdicts = {}

        # Pre-fill refusals (empty responses don't need judging)
        for name, resp in responses.items():
            if not resp or not resp.strip():
                verdicts[name] = "REFUSAL"

        def judge_one(args_tuple):
            name, resp = args_tuple
            return name, judge_response(question, gold, resp)

        if to_judge:
            judge_total = len(to_judge)
            _completed_count = 0
            sys.stderr.write(f"  Judging {judge_total} responses...\n")

            if args.sequential:
                for name, resp in to_judge:
                    _, verdict = judge_one((name, resp))
                    verdicts[name] = verdict
            else:
                with ThreadPoolExecutor(max_workers=len(to_judge)) as executor:
                    futures = {executor.submit(judge_one, t): t for t in to_judge}
                    for future in as_completed(futures):
                        name, verdict = future.result()
                        verdicts[name] = verdict

            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()

        for model in selected:
            name = model["name"]
            results.append({
                "name": name,
                "response": responses.get(name, ""),
                "verdict": verdicts.get(name, "UNKNOWN"),
            })
    else:
        for model in selected:
            name = model["name"]
            results.append({"name": name, "response": responses.get(name, "")})

    display_results(question, results, gold)


if __name__ == "__main__":
    main()
