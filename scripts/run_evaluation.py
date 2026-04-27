#!/usr/bin/env python3
"""Run IKP probes against all models via OpenRouter.

Priority ordering:
1. Known-size open models (for fitting the scaling curve)
2. Proprietary frontier models (for parameter estimation)

Uses parallel execution (16 workers) and Gemini 3 Flash Preview as judge.
Saves results incrementally per model.

Comprehensive logging: every probe interaction (question, model response,
judge prompt, judge raw output, verdict) is logged to data/evaluation_v8.log
and per-model JSONL transcript files in data/results/transcripts/.
"""

import json
import os
import sys
import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# ── Logging setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "results" / "transcripts"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROJECT_ROOT / "data" / "evaluation_v8.log"

# Console + file logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Suppress noisy httpx/httpcore request-level logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
JUDGE_MODEL = "google/gemini-3-flash-preview"
SYSTEM_MSG = "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."
HALLUCINATION_PENALTY = -0.5  # correct=+1, refusal=0, wrong=this

# Thread-local transcript writers (one JSONL file per model)
_transcript_lock = threading.Lock()
_transcript_files = {}


def _get_transcript_writer(model_name: str):
    """Get or create a JSONL transcript file handle for a model."""
    if model_name not in _transcript_files:
        path = LOG_DIR / f"{model_name}.jsonl"
        _transcript_files[model_name] = open(path, "w", encoding="utf-8")
    return _transcript_files[model_name]


def _write_transcript(model_name: str, record: dict):
    """Append one JSON record to the model's transcript file (thread-safe)."""
    with _transcript_lock:
        f = _get_transcript_writer(model_name)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ── Thinking-mode helpers ──────────────────────────────────────
def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking model output."""
    if not text:
        return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned.startswith('<think>'):
        cleaned = ''
    return cleaned or text


# ── Model query ────────────────────────────────────────────────
# Ollama exposes an OpenAI-compatible API at localhost:11434/v1/.
# We use the same code path for both, just different base URL and headers.
OLLAMA_BASE = "http://localhost:11434/v1/chat/completions"
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"


def query_model(model_id: str, question: str, is_thinking: bool = False) -> dict:
    """Query a model via OpenRouter or local Ollama (OpenAI-compatible API).

    Models with "ollama/" prefix are routed to the local Ollama server.
    Both use the same OpenAI chat completions format.
    Returns dict with keys: response, status_code, error, attempts.
    """
    is_local = model_id.startswith("ollama/")
    if is_local:
        api_url = OLLAMA_BASE
        model_name = model_id[len("ollama/"):]
        headers = {"Content-Type": "application/json"}
    else:
        api_url = OPENROUTER_BASE
        model_name = model_id
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    if is_local:
        # Ollama defaults to ~50 predict tokens; many Qwen3-family models
        # reason by default and consume all tokens on reasoning before writing
        # a content answer. Grant a generous budget so the content field is
        # actually populated.
        payload["max_tokens"] = 1500
    if not is_local:
        if model_id.startswith("anthropic/"):
            # Anthropic via OpenRouter: {"effort": "high"} to enable extended thinking,
            # {"enabled": False} to explicitly disable (some Claude models reason by default).
            payload["reasoning"] = {"effort": "high"} if is_thinking else {"enabled": False}
        elif is_thinking:
            payload["reasoning"] = {"effort": "medium"}
        elif model_id.startswith("deepseek/deepseek-v4"):
            # DeepSeek V4 Pro/Flash reason by default on OpenRouter; explicitly disable for non-thinking variant.
            payload["reasoning"] = {"enabled": False}

    result = {"response": "", "status_code": None, "error": None, "attempts": 0}
    http = httpx.Client(timeout=120)

    try:
        for attempt in range(3):
            result["attempts"] = attempt + 1
            try:
                r = http.post(api_url, headers=headers, json=payload)
                result["status_code"] = r.status_code
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    if is_thinking and content:
                        content = strip_thinking(content)
                    result["response"] = content
                    return result
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                elif r.status_code == 400:
                    result["error"] = f"HTTP 400: {r.text[:200]}"
                    return result
                else:
                    result["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    if attempt == 2:
                        logger.warning(f"  Model {model_id} returned {r.status_code}")
                    time.sleep(2)
            except Exception as e:
                result["error"] = str(e)
    finally:
        http.close()

    return result


# ── Judge ──────────────────────────────────────────────────────
def judge_response(question: str, gold: str, response: str) -> dict:
    """Judge a response using Gemini 3 Flash Preview as a 3-way classifier.

    Returns dict with keys: verdict, judge_prompt, judge_raw_output, error.
    verdict is one of: "CORRECT", "REFUSAL", "WRONG".
    """
    out = {"verdict": None, "judge_prompt": None, "judge_raw_output": None, "error": None}

    if not response or not response.strip():
        out["verdict"] = "REFUSAL"
        out["judge_raw_output"] = "(empty response — auto-REFUSAL)"
        return out

    # Handle co-credit answers (semicolon-separated alternatives)
    if ';' in gold:
        gold_display = ' OR '.join(a.strip() for a in gold.split(';'))
        co_credit_note = (
            "\nNOTE: The correct answer lists multiple acceptable names separated by OR. "
            "The model's response is CORRECT if it names ANY of these people (alone or together). "
            "Naming additional correct co-contributors is also fine."
        )
    else:
        gold_display = gold
        co_credit_note = ""

    prompt = f"""You are a strict factual judge. Classify the model's response into one of three categories.

Question: {question}
Correct answer: {gold_display}
Model's response: {response}

First, determine if the model REFUSED to answer (said "I don't know", expressed uncertainty, hedged, declined, or otherwise avoided giving a definitive answer). If so, reply REFUSAL.

If the model gave a definitive answer, judge whether it is correct using these rules:
1. If the correct answer is a YEAR, the model must state that EXACT year. A different year is WRONG.
2. If the correct answer is a NUMBER, the numeric value must match exactly or be very close (within 1-2%). Ignore formatting differences like commas, spaces, units, or words like "approximately" — only the numeric value matters.
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling OK. If the model names the correct person PLUS additional co-contributors, that is still CORRECT.
4. If the correct answer is a RESEARCH FIELD, accept the answer if the model identifies a field that is closely related or adjacent to the correct field. Sub-fields, parent fields, and overlapping areas all count as correct (e.g. "wireless networking" matches "computer networking", "cybersecurity" matches "computer security", "database indexing" matches "database systems"). However, if the model lists MULTIPLE unrelated fields as guesses without committing to one, that is WRONG — it indicates the model is guessing rather than demonstrating knowledge.
5. If the model gives a DIFFERENT, unrelated answer, that is WRONG.
{co_credit_note}
Reply with exactly one word: CORRECT, REFUSAL, or WRONG"""

    out["judge_prompt"] = prompt

    http = httpx.Client(timeout=60)
    try:
        for attempt in range(3):
            try:
                r = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": JUDGE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "reasoning": {"effort": "low"},
                    },
                )
                if r.status_code == 200:
                    raw = r.json()["choices"][0]["message"]["content"].strip()
                    out["judge_raw_output"] = raw
                    raw_upper = raw.upper()
                    if raw_upper.startswith("CORRECT"):
                        out["verdict"] = "CORRECT"
                    elif raw_upper.startswith("REFUSAL"):
                        out["verdict"] = "REFUSAL"
                    else:
                        out["verdict"] = "WRONG"
                    return out
                elif r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                else:
                    out["error"] = f"Judge HTTP {r.status_code}: {r.text[:200]}"
                    out["verdict"] = "WRONG"
                    return out
            except Exception as e:
                out["error"] = f"Judge exception: {e}"
        out["verdict"] = "WRONG"
        out["judge_raw_output"] = "(all retries failed)"
        return out
    finally:
        http.close()


# ── Per-model evaluation ──────────────────────────────────────
def evaluate_model(model_name: str, model_info: dict, probes: list, results_dir: Path):
    """Evaluate a single model against all probes."""
    model_id = model_info["id"]
    result_file = results_dir / f"{model_name}.json"

    # Check for existing results
    if result_file.exists():
        existing = json.load(open(result_file))
        if len(existing.get("results", [])) == len(probes):
            acc = existing.get("accuracy", 0)
            logger.info(f"  {model_name}: already done (accuracy={acc:.1%})")
            return existing

    is_thinking = model_info.get("thinking", False)
    results = []
    correct_count = 0

    def eval_one(probe):
        q = probe["question"]
        gold = probe["answer"]

        # 1. Query the model
        model_result = query_model(model_id, q, is_thinking)
        response = model_result["response"]

        # 2. Judge the response
        judge_result = judge_response(q, gold, response)
        verdict = judge_result["verdict"]

        # 3. Build result record
        record = {
            "probe_id": probe.get("id", ""),
            "tier": probe["tier"],
            "source_type": probe.get("source_type", ""),
            "domain": probe.get("domain", ""),
            "question": q,
            "gold_answer": gold,
            "model_response": (response or "")[:500],
            "correct": verdict == "CORRECT",
            "refusal": verdict == "REFUSAL",
            "verdict": verdict,
        }

        # 4. Write comprehensive transcript
        transcript = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "model_id": model_id,
            "probe_id": probe.get("id", ""),
            "tier": probe["tier"],
            "domain": probe.get("domain", ""),
            "question": q,
            "gold_answer": gold,
            "model_response_full": (response or "")[:1000],
            "model_query_attempts": model_result["attempts"],
            "model_query_status": model_result["status_code"],
            "model_query_error": model_result["error"],
            "judge_model": JUDGE_MODEL,
            "judge_prompt": judge_result["judge_prompt"],
            "judge_raw_output": judge_result["judge_raw_output"],
            "judge_error": judge_result["error"],
            "verdict": verdict,
        }
        _write_transcript(model_name, transcript)

        return record

    # Ollama batches concurrent requests internally; use fewer workers than OpenRouter
    # to avoid overwhelming local GPU, but still benefit from batching
    is_local = model_id.startswith("ollama/")
    max_workers = 16 if is_local else 16

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, p): i for i, p in enumerate(probes)}
        done = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if result["correct"]:
                    correct_count += 1
                done += 1
                if done % 100 == 0:
                    logger.info(f"    {model_name}: {done}/{len(probes)}, {correct_count} correct so far")
            except Exception as e:
                done += 1
                logger.error(f"    {model_name}: probe exception: {e}")
                results.append({
                    "probe_id": "", "tier": "", "correct": False,
                    "refusal": True, "verdict": "ERROR", "error": str(e),
                })

    # ── Scoring ────────────────────────────────────────────────
    from collections import defaultdict
    tier_stats = defaultdict(lambda: {"correct": 0, "total": 0, "refusal": 0, "wrong": 0})
    for r in results:
        t = r["tier"]
        tier_stats[t]["total"] += 1
        if r.get("refusal"):
            tier_stats[t]["refusal"] += 1
        elif r["correct"]:
            tier_stats[t]["correct"] += 1
        else:
            tier_stats[t]["wrong"] += 1

    tier_accs = {}
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        s = tier_stats[t]
        if s["total"] > 0:
            score = (s["correct"] + HALLUCINATION_PENALTY * s["wrong"]) / s["total"]
            tier_accs[t] = max(score, 0.0)  # Floor at 0
        else:
            tier_accs[t] = 0.0

    # Overall = mean of per-tier scores (each already floored at 0)
    accuracy = sum(tier_accs.values()) / len(tier_accs) if tier_accs else 0.0
    raw_accuracy = correct_count / len(probes) if probes else 0

    output = {
        "model_name": model_name,
        "model_id": model_id,
        "params_B": model_info.get("params_B"),
        "family": model_info.get("family"),
        "vendor": model_info.get("vendor"),
        "arch": model_info.get("arch"),
        "accuracy": accuracy,
        "raw_accuracy": raw_accuracy,
        "hallucination_penalty": HALLUCINATION_PENALTY,
        "judge_model": JUDGE_MODEL,
        "correct": correct_count,
        "total": len(probes),
        "tier_accuracy": tier_accs,
        "tier_stats": dict(tier_stats),
        "results": results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"  {model_name}: accuracy={accuracy:.1%} raw={raw_accuracy:.1%} ({correct_count}/{len(probes)})")
    tier_str = " ".join(f"{t}={tier_accs[t]:.0%}" for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])
    logger.info(f"    Per-tier: {tier_str}")
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        s = tier_stats[t]
        logger.info(f"    {t}: correct={s['correct']} wrong={s['wrong']} refusal={s['refusal']} total={s['total']}")

    return output


# ── Main ──────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("IKP EVALUATION v8 — starting fresh run")
    logger.info(f"  Judge model: {JUDGE_MODEL}")
    logger.info(f"  Hallucination penalty: {HALLUCINATION_PENALTY}")
    logger.info(f"  Log file: {LOG_FILE}")
    logger.info(f"  Transcript dir: {LOG_DIR}")
    logger.info("=" * 60)

    # Load probes
    probes = json.load(open(PROJECT_ROOT / "data" / "probes" / "final_probe_set_v8.json"))
    logger.info(f"Loaded {len(probes)} probes")
    from collections import Counter
    tier_counts = Counter(p["tier"] for p in probes)
    logger.info(f"  Per tier: {dict(sorted(tier_counts.items()))}")

    # Load model config
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    models = config["models"]
    logger.info(f"Loaded {len(models)} models")

    # Create results directory
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Priority ordering: known-size open models first
    priority = []
    for name, info in models.items():
        params = info.get("params_B")
        is_open = info.get("type") == "open"
        if params and is_open:
            priority.append((0, params, name, info))
        elif params:
            priority.append((1, params, name, info))
        else:
            priority.append((2, 0, name, info))
    priority.sort()

    # Evaluate models concurrently (4 models at a time, each with 16 internal workers)
    CONCURRENT_MODELS = 4
    all_summaries = []

    def eval_model_wrapper(args):
        _, _, model_name, model_info = args
        logger.info(f"Evaluating: {model_name} ({model_info['id']})")
        try:
            summary = evaluate_model(model_name, model_info, probes, results_dir)
            return {
                "model": model_name,
                "params_B": model_info.get("params_B"),
                "family": model_info.get("family"),
                "vendor": model_info.get("vendor"),
                "accuracy": summary["accuracy"],
                "raw_accuracy": summary["raw_accuracy"],
                "tier_accuracy": summary["tier_accuracy"],
                "tier_stats": summary["tier_stats"],
            }
        except Exception as e:
            logger.error(f"  Error evaluating {model_name}: {e}", exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=CONCURRENT_MODELS) as model_executor:
        futures = {model_executor.submit(eval_model_wrapper, p): p for p in priority}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_summaries.append(result)
                with open(results_dir / "evaluation_summary.json", "w") as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Close all transcript files
    for f in _transcript_files.values():
        f.close()

    logger.info(f"\n{'=' * 60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Models evaluated: {len(all_summaries)}")
    for s in sorted(all_summaries, key=lambda x: x.get("accuracy", 0), reverse=True):
        logger.info(f"  {s['model']:25s}  accuracy={s['accuracy']:.1%}  raw={s['raw_accuracy']:.1%}")


if __name__ == "__main__":
    main()
