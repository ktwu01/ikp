"""Strict re-judge of V4 Flash researcher-subfield probes.

Uses the same judge model (Gemini 3 Flash Preview) but with a stricter rubric:
the response is CORRECT only if it either (a) matches the gold subfield exactly
or in standard synonym, OR (b) demonstrates person-specific knowledge by naming
a co-author, paper, system, or institutional affiliation. Generic
'computer X with focus on Y' answers without person-specific evidence are WRONG.

This re-judges only V4 Flash's CORRECT verdicts on researcher probes (T3-T7)
to quantify how many flip CORRECT -> WRONG under the stricter rubric.

Reads /Users/boj/ikp-paper/data/results/deepseek-v4-flash.json and writes
/Users/boj/ikp-paper/data/results/deepseek-v4-flash_strict_rejudge.json.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
JUDGE_MODEL = "google/gemini-3-flash-preview"
N_WORKERS = 16

STRICT_PROMPT_TEMPLATE = """You are a STRICT factual judge for a research-knowledge probe. The probe asks a model to identify a specific researcher's primary CS subfield. Your job is to decide whether the model demonstrates real knowledge of THIS specific person, not generic guessing.

Question: {question}
Gold subfield: {gold}
Model's response: {response}

CLASSIFY into one of three categories:

REFUSAL: The model said "I don't know", expressed uncertainty, hedged, or declined to answer.

CORRECT: The model's response satisfies AT LEAST ONE of:
  (a) Names the gold subfield exactly, or a direct standard synonym (e.g., "networking" = "computer networking", "cybersecurity" = "computer security", "OS" = "operating systems"). A SUBSET of the gold subfield is OK (e.g., "wireless networking" matches "computer networking").
  (b) Demonstrates person-specific knowledge: names a real paper/system/artifact by this researcher, names a verifiable co-author or institutional affiliation, OR describes a specific research contribution that is plausible only for this exact person.

WRONG: Everything else. In particular:
  - Generic "computer X with focus on Y" answers that name an ADJACENT or PARENT subfield without person-specific evidence are WRONG. (Example: gold="embedded systems", response="computer architecture, focusing on memory systems and cache coherence" — WRONG, because the response invents specifics not connected to the actual person.)
  - Answers that name a DIFFERENT subfield (even within CS) without person-specific evidence are WRONG.
  - Answers that hedge with multiple unrelated guesses ("X, Y, or Z") are WRONG.

Reply with EXACTLY one word: CORRECT, REFUSAL, or WRONG"""


def call_judge(probe_id, question, gold, response, http):
    prompt = STRICT_PROMPT_TEMPLATE.format(question=question, gold=gold, response=response)
    for attempt in range(3):
        try:
            r = http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": JUDGE_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0,
                      "reasoning": {"effort": "low"}},
                timeout=60,
            )
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"].strip()
                u = raw.upper()
                if u.startswith("CORRECT"):
                    return probe_id, "CORRECT", raw
                if u.startswith("REFUSAL"):
                    return probe_id, "REFUSAL", raw
                return probe_id, "WRONG", raw
            elif r.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                return probe_id, "ERROR", f"HTTP {r.status_code}"
        except Exception as e:
            time.sleep(2)
    return probe_id, "ERROR", "all retries failed"


def main():
    src = Path("/Users/boj/ikp-paper/data/results/deepseek-v4-flash.json")
    out = Path("/Users/boj/ikp-paper/data/results/deepseek-v4-flash_strict_rejudge.json")
    with open(src) as f:
        data = json.load(f)

    targets = [r for r in data["results"]
               if r["source_type"] == "researcher"
               and r["tier"] in ("T3", "T4", "T5", "T6", "T7")
               and r["verdict"] == "CORRECT"]
    print(f"Re-judging {len(targets)} V4 Flash CORRECT researcher probes (T3-T7) with strict rubric")

    http = httpx.Client(timeout=60)
    new_verdicts = {}
    raw_outputs = {}
    flips = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(call_judge, r["probe_id"], r["question"],
                          r["gold_answer"], r["model_response"], http)
                for r in targets]
        for i, fut in enumerate(as_completed(futs)):
            pid, verdict, raw = fut.result()
            new_verdicts[pid] = verdict
            raw_outputs[pid] = raw
            if verdict != "CORRECT":
                flips += 1
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(targets)}  flips so far: {flips}")
    http.close()

    # Per-tier breakdown
    by_tier = {}
    for r in targets:
        pid = r["probe_id"]
        t = r["tier"]
        nv = new_verdicts.get(pid, "?")
        by_tier.setdefault(t, {"CORRECT": 0, "WRONG": 0, "REFUSAL": 0, "ERROR": 0, "?": 0, "_n": 0})
        by_tier[t][nv] = by_tier[t].get(nv, 0) + 1
        by_tier[t]["_n"] += 1

    print("\n=== Strict re-judge results (V4 Flash, researcher probes only) ===")
    print(f"  Originally {len(targets)} judged CORRECT under lenient rubric")
    print(f"  Per-tier strict verdicts:")
    for t in ("T3", "T4", "T5", "T6", "T7"):
        if t not in by_tier: continue
        b = by_tier[t]
        n = b["_n"]
        c = b.get("CORRECT", 0)
        w = b.get("WRONG", 0)
        r = b.get("REFUSAL", 0)
        e = b.get("ERROR", 0)
        flip_rate = (n - c) / max(n, 1) * 100
        print(f"    {t}: n={n:3d}  strict-CORRECT={c:3d}  strict-WRONG={w:3d}  REFUSAL={r:2d}  ERROR={e:2d}  flip={flip_rate:.0f}%")

    # Save
    with open(out, "w") as f:
        json.dump({
            "source_file": str(src),
            "judge_model": JUDGE_MODEL,
            "rubric": "strict",
            "n_rejudged": len(targets),
            "new_verdicts": new_verdicts,
            "raw_outputs": raw_outputs,
            "per_tier": by_tier,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
