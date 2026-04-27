#!/usr/bin/env python3
"""Test all 115 models for thinking mode support.

For each model, tests:
1. Normal mode (no reasoning parameter)
2. Thinking mode (reasoning.effort=medium)

Reports which models:
- Support thinking (reasoning field populated)
- Support both modes (different behavior with/without reasoning)
- Fail with reasoning parameter (HTTP 400)
- Always think (reasoning populated even without parameter)
"""

import json
import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
PROJECT_ROOT = Path(__file__).parent.parent.parent
Q = "What is the capital of France? Answer in one word."


def test_model(name: str, model_id: str):
    """Test a model in both normal and thinking modes."""
    result = {
        "name": name,
        "model_id": model_id,
        "normal": {"status": None, "content": None, "has_reasoning": False},
        "thinking": {"status": None, "content": None, "has_reasoning": False},
    }

    for mode in ["normal", "thinking"]:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": Q}],
            "temperature": 0,
        }
        if mode == "thinking":
            payload["reasoning"] = {"effort": "medium"}

        try:
            with httpx.Client(timeout=30) as http:
                r = http.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload)

                result[mode]["status"] = r.status_code
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    result[mode]["content"] = msg.get("content")
                    result[mode]["has_reasoning"] = bool(msg.get("reasoning"))
                    result[mode]["content_is_none"] = msg.get("content") is None
                else:
                    err = r.json().get("error", {}).get("message", "")[:80]
                    result[mode]["error"] = err
        except Exception as e:
            result[mode]["status"] = "error"
            result[mode]["error"] = str(e)[:80]

    return result


def main():
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))
    models = config["models"]

    print(f"Testing {len(models)} models in both normal and thinking modes...\n")

    results = []
    done = 0
    total = len(models)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(test_model, name, info["id"]): name
                   for name, info in models.items()}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            n = r["normal"]
            t = r["thinking"]
            n_reason = "think" if n["has_reasoning"] else "no"
            t_status = f"think" if t["has_reasoning"] else ("FAIL" if t["status"] != 200 else "no")
            t_null = " NULL!" if t.get("content_is_none") else ""
            print(f"  [{done:3d}/{total}] {r['name']:35s} normal={n_reason:5s}  +param={t_status:5s}{t_null}", flush=True)

    results.sort(key=lambda r: r["name"])

    # Classify models
    always_thinks = []      # reasoning in normal mode (no param needed)
    supports_thinking = []  # reasoning only with param
    thinking_fails = []     # HTTP error with reasoning param
    never_thinks = []       # no reasoning in either mode
    content_null = []       # content is None in some mode

    for r in results:
        n = r["normal"]
        t = r["thinking"]

        if n.get("content_is_none") or t.get("content_is_none"):
            content_null.append(r)

        if t["status"] != 200:
            thinking_fails.append(r)
        elif n["has_reasoning"] and t["has_reasoning"]:
            always_thinks.append(r)
        elif not n["has_reasoning"] and t["has_reasoning"]:
            supports_thinking.append(r)
        elif not n["has_reasoning"] and not t["has_reasoning"]:
            never_thinks.append(r)
        else:
            always_thinks.append(r)  # edge case

    # Print report
    print(f"{'='*90}")
    print(f"THINKING MODE SUPPORT REPORT")
    print(f"{'='*90}")

    print(f"\n1. ALWAYS THINKS (reasoning even without param) — {len(always_thinks)} models:")
    print(f"   These models produce reasoning natively. No need for reasoning param.")
    for r in always_thinks:
        c = (r["normal"]["content"] or "")[:30]
        print(f"   {r['name']:35s} normal_content='{c}'")

    print(f"\n2. SUPPORTS THINKING (reasoning only with param) — {len(supports_thinking)} models:")
    print(f"   These can run in BOTH modes — good candidates for ablation study.")
    for r in supports_thinking:
        print(f"   {r['name']:35s}")

    print(f"\n3. THINKING PARAM FAILS — {len(thinking_fails)} models:")
    for r in thinking_fails:
        print(f"   {r['name']:35s} error={r['thinking'].get('error','')[:50]}")

    print(f"\n4. NEVER THINKS — {len(never_thinks)} models:")
    for r in never_thinks:
        print(f"   {r['name']:35s}")

    print(f"\n5. CONTENT IS NULL in some mode — {len(content_null)} models:")
    for r in content_null:
        nm = "NULL" if r["normal"].get("content_is_none") else "ok"
        tm = "NULL" if r["thinking"].get("content_is_none") else "ok"
        print(f"   {r['name']:35s} normal={nm} thinking={tm}")

    # Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"  Always thinks:      {len(always_thinks)}")
    print(f"  Supports thinking:  {len(supports_thinking)}")
    print(f"  Thinking fails:     {len(thinking_fails)}")
    print(f"  Never thinks:       {len(never_thinks)}")
    print(f"  Content null:       {len(content_null)}")

    # Save full results
    output = PROJECT_ROOT / "data" / "thinking_mode_report.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output}")

    # Generate recommended model config update
    print(f"\n{'='*90}")
    print(f"RECOMMENDED ABLATION VARIANTS")
    print(f"{'='*90}")
    print(f"Models that support BOTH thinking and non-thinking modes:")
    for r in supports_thinking:
        print(f'    "{r["name"]}-think":  {{"id": "{r["model_id"]}", ..., "thinking": true}}')
        print(f'    "{r["name"]}-nothink": {{"id": "{r["model_id"]}", ..., "thinking": false}}')


if __name__ == "__main__":
    main()
