#!/usr/bin/env python3
"""Generate and run Chinese-language IKP probes.

Tests geographic/linguistic bias: do Chinese models perform better on
Chinese-language probes about Chinese topics?

Generates probes in Chinese about:
- Chinese geography (city populations, provincial capitals, landmark heights)
- Chinese history (specific dates, figures, events)
- Chinese academia (university departments, researchers, publications)
- Chinese culture (literary works, historical figures, regional specialties)
"""

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_client import OpenRouterClient
from probe_runner import run_single_probe, compute_summary
from scorer import score_probe, is_refusal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "chinese_probes.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def generate_chinese_probes(client, model, n_per_category=30):
    """Generate Chinese-language probes across difficulty tiers."""

    categories = [
        {
            "name": "easy",
            "tier_equiv": "T1-T2",
            "prompt": """请生成{n}个关于中国的常识性事实问题，用中文提问和回答。
这些应该是大多数中国人都知道的事实。

每个问题需要包含：
- question_direct: 直接问题形式
- question_fill_blank: 填空形式
- answer: 简短准确的答案
- answer_type: "text" 或 "numeric"
- category: "geography" / "history" / "culture" / "science"

示例：
- "中国的首都是哪里？" 答案："北京"
- "长江全长多少公里？" 答案："6300"
- "《红楼梦》的作者是谁？" 答案："曹雪芹"

返回JSON数组，每个元素格式：
{{"id": "cn_easy_001", "question_direct": "...", "question_fill_blank": "...是___。", "answer": "...", "answer_type": "text|numeric", "category": "..."}}

只返回JSON数组，不要其他文字。"""
        },
        {
            "name": "medium",
            "tier_equiv": "T3-T4",
            "prompt": """请生成{n}个关于中国的专业性事实问题，用中文提问和回答。
这些应该是需要一定专业知识才能回答的问题。

类型包括：
- 中国各省会城市的具体人口数据
- 中国历史上具体事件的准确日期
- 中国大学特定院系的成立年份
- 中国特定建筑的高度
- 中国特定行政区划的面积
- 中国科学家的具体成就和年份

示例：
- "贵州省的省会是哪个城市？" 答案："贵阳"
- "南京长江大桥建成于哪一年？" 答案："1968"
- "中国科学技术大学位于哪个城市？" 答案："合肥"

返回JSON数组，格式同上，id以"cn_medium_"开头。"""
        },
        {
            "name": "hard",
            "tier_equiv": "T5-T6",
            "prompt": """请生成{n}个关于中国的非常专业和冷门的事实问题，用中文提问和回答。
这些应该是只有相关领域专家才可能知道的事实。

类型包括：
- 中国某个小城市的具体人口（如：2020年第七次人口普查数据）
- 某个不太知名的中国学者及其具体研究方向
- 中国某个具体的地方政策或法规的颁布年份
- 中国某个小众历史事件的具体日期
- 中国某个不太出名的建筑或桥梁的具体参数
- 中国某个地级市的GDP或面积

示例：
- "2020年人口普查中，甘肃省天水市的常住人口是多少？" 答案："约332万"
- "清华大学计算机科学与技术系成立于哪一年？" 答案："1958"

这些必须是真实可验证的事实，不要编造。
返回JSON数组，格式同上，id以"cn_hard_"开头。"""
        },
    ]

    all_probes = []
    for cat in categories:
        prompt = cat["prompt"].format(n=n_per_category)
        messages = [
            {"role": "system", "content": "你是一个研究助手，负责生成事实性知识探针。请只返回有效的JSON。"},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"Generating {cat['name']} Chinese probes...")
        text = client.get_response_text(model=model, messages=messages, temperature=0.7, max_tokens=16000)

        # Parse JSON
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            if text.startswith("json"):
                text = text[4:]

        try:
            probes = json.loads(text)
            valid_probes = []
            for i, p in enumerate(probes):
                # Skip probes missing required fields
                if "question_direct" not in p and "answer" not in p:
                    continue
                p["id"] = f"cn_{cat['name']}_{i:03d}"
                p["tier_equiv"] = cat["tier_equiv"]
                p["language"] = "zh"
                if "question_direct" not in p:
                    p["question_direct"] = p.get("question_fill_blank", p.get("question", ""))
                if "question_fill_blank" not in p:
                    p["question_fill_blank"] = p["question_direct"]
                if "question_contextual" not in p:
                    p["question_contextual"] = p["question_fill_blank"]
                if "answer" not in p:
                    continue
                valid_probes.append(p)
            probes = valid_probes
            all_probes.extend(probes)
            logger.info(f"  Got {len(probes)} {cat['name']} probes")
        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse {cat['name']} probes: {e}")

    return all_probes


def run_chinese_probes_on_model(client, model_name, model_id, probes):
    """Run Chinese probes against a single model."""
    results = []

    for i, probe in enumerate(probes):
        phrasings = [
            ("direct", probe.get("question_direct", "")),
            ("fill_blank", probe.get("question_fill_blank", probe.get("question_direct", ""))),
            ("contextual", probe.get("question_contextual", probe.get("question_direct", ""))),
        ]

        responses = {}
        for ptype, ptext in phrasings:
            if not ptext:
                continue
            messages = [
                {"role": "system", "content": "请直接简洁地回答以下事实问题。如果问的是数字，只给数字。如果问的是名称，只给名称。不要解释。"},
                {"role": "user", "content": ptext},
            ]
            try:
                resp = client.get_response_text(model=model_id, messages=messages, temperature=0, max_tokens=150)
                responses[ptype] = {"text": resp, "is_refusal": is_refusal(resp)}
            except Exception as e:
                responses[ptype] = {"text": None, "error": str(e), "is_refusal": True}

        resp_texts = [responses[pt]["text"] for pt in responses if responses[pt].get("text")]
        correct = score_probe(resp_texts, probe.get("answer", ""), probe.get("answer_type", "auto"))
        all_refusal = all(responses[pt].get("is_refusal", True) for pt in responses)

        results.append({
            "probe_id": probe["id"],
            "tier_equiv": probe.get("tier_equiv", "?"),
            "category": probe.get("category", "?"),
            "gold_answer": probe.get("answer", ""),
            "responses": responses,
            "correct": correct,
            "excluded": all_refusal,
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  {model_name}: {i+1}/{len(probes)} Chinese probes done")

    return results


def main():
    config = json.load(open(PROJECT_ROOT / "configs" / "all_models.json"))["models"]

    client = OpenRouterClient(requests_per_minute=50, max_retries=5, timeout=120)

    # Step 1: Generate Chinese probes
    probes_file = PROJECT_ROOT / "data" / "probes" / "chinese_probes.json"
    if probes_file.exists():
        probes = json.load(open(probes_file))
        logger.info(f"Loaded {len(probes)} existing Chinese probes")
    else:
        probes = generate_chinese_probes(client, "anthropic/claude-sonnet-4", n_per_category=30)
        probes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(probes_file, "w") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated and saved {len(probes)} Chinese probes")

    # Step 2: Select models to test — mix of Chinese and Western
    test_models = [
        # Chinese-origin
        "qwen-2.5-7b", "qwen-2.5-72b", "deepseek-v3",
        # Western — similar sizes for comparison
        "llama-3.1-8b", "llama-3.3-70b", "hermes-3-405b",
        # Frontier proprietary
        "gpt-4o", "gpt-4o-mini", "gpt-4.1",
        "claude-sonnet-4", "claude-opus-4",
        "gemini-2.5-pro", "gemini-3.1-pro",
    ]

    output_dir = PROJECT_ROOT / "data" / "chinese_responses"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in test_models:
        if model_name not in config:
            logger.warning(f"Model {model_name} not in config, skipping")
            continue

        model_info = config[model_name]
        results_file = output_dir / f"{model_name}_cn_results.json"

        # Skip if already done
        if results_file.exists():
            existing = json.load(open(results_file))
            if existing.get("total_probes", 0) >= len(probes) * 0.9:
                logger.info(f"Skipping {model_name} (already complete)")
                all_summaries.append(existing)
                continue

        logger.info(f"\n=== {model_name} ({model_info['vendor']}) — Chinese probes ===")

        try:
            results = run_chinese_probes_on_model(
                client, model_name, model_info["id"], probes
            )

            # Compute per-difficulty accuracy
            cat_correct = {}
            cat_total = {}
            for r in results:
                cat = r.get("tier_equiv", "?")
                if r.get("excluded"):
                    continue
                cat_total[cat] = cat_total.get(cat, 0) + 1
                if r["correct"]:
                    cat_correct[cat] = cat_correct.get(cat, 0) + 1

            cat_accuracy = {c: cat_correct.get(c, 0) / cat_total[c] if cat_total.get(c, 0) > 0 else 0
                          for c in sorted(set(list(cat_total.keys()) + list(cat_correct.keys())))}

            total_scored = sum(cat_total.values())
            total_correct = sum(cat_correct.values())
            agg = total_correct / total_scored if total_scored > 0 else 0

            summary = {
                "model_name": model_name,
                "vendor": model_info["vendor"],
                "params_billion": model_info.get("params_B"),
                "language": "zh",
                "total_probes": len(results),
                "per_difficulty_accuracy": cat_accuracy,
                "aggregate_accuracy": agg,
                "total_scored": total_scored,
                "total_correct": total_correct,
            }

            with open(results_file, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            all_summaries.append(summary)
            logger.info(f"  {model_name}: agg={agg:.3f}, per_diff={cat_accuracy}")

        except Exception as e:
            logger.error(f"  FAILED {model_name}: {e}")

    # Final comparison
    logger.info(f"\n{'='*90}")
    logger.info(f"  CHINESE VS ENGLISH PROBE COMPARISON")
    logger.info(f"{'='*90}")
    logger.info(f"  {'Model':25s} {'Vendor':>10s} {'CN Agg':>8s} {'EN Agg':>8s} {'Delta':>8s}")
    logger.info(f"  {'─'*65}")

    for s in sorted(all_summaries, key=lambda x: x.get("aggregate_accuracy", 0)):
        name = s["model_name"]
        cn_agg = s["aggregate_accuracy"]
        # Load English results for comparison
        en_file = PROJECT_ROOT / "data" / "raw_responses" / f"{name}_results.json"
        en_agg = 0
        if en_file.exists():
            en = json.load(open(en_file))
            en_agg = en.get("aggregate_accuracy", 0)

        delta = cn_agg - en_agg
        logger.info(f"  {name:25s} {s['vendor']:>10s} {cn_agg:8.3f} {en_agg:8.3f} {delta:+8.3f}")


if __name__ == "__main__":
    main()
