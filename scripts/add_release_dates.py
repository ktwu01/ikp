#!/usr/bin/env python3
"""Add release_date field to every model in configs/all_models.json.

Dates are official first public release (not distilled/finetuned re-release).
For '-think' variants that share weights with the base model, the date equals
the base model's release. All dates verified against authoritative sources
(vendor blog posts, HuggingFace model cards, primary news coverage) via web
search. Notes/uncertainty documented inline.
"""
import json
from pathlib import Path

RELEASE_DATES = {
    # Meta Llama
    "llama-3-8b": "2024-04-18",
    "llama-3.1-8b": "2024-07-23",
    "llama-3.2-1b": "2024-09-25",
    "llama-3.2-3b": "2024-09-25",
    "llama-3-70b": "2024-04-18",
    "llama-3.1-70b": "2024-07-23",
    "llama-3.3-70b": "2024-12-06",
    "hermes-3-405b": "2024-08-16",
    "hermes-4-405b": "2025-08-26",
    "llama-4-scout": "2025-04-05",
    "llama-4-maverick": "2025-04-05",

    # Alibaba Qwen
    "qwen-2.5-7b": "2024-09-19",
    "qwen-2.5-72b": "2024-09-19",
    "qwq-32b-think": "2025-03-06",
    "qwen3-8b-think": "2025-04-29",
    "qwen3-14b-think": "2025-04-29",
    "qwen3-30b-a3b-think": "2025-04-29",
    "qwen3-32b-think": "2025-04-29",
    "qwen3-235b-a22b-think": "2025-04-29",
    "qwen3.5-9b-think": "2026-03-02",
    "qwen3.5-27b-think": "2026-02-24",
    "qwen3.5-35b-a3b-think": "2026-02-24",
    "qwen3.5-122b-a10b-think": "2026-02-24",
    "qwen3.5-397b-a17b-think": "2026-02-16",
    "qwen-turbo": "2024-11-01",
    "qwen-plus": "2024-11-01",
    "qwen-max": "2025-01-27",
    "qwen3-max": "2025-09-23",
    "qwen3-next-80b-a3b": "2025-09-11",
    "qwen3.5-flash-think": "2026-02-23",
    "qwen3.5-plus-think": "2026-02-16",
    "qwen3.6-plus-think": "2026-04-02",

    # Mistral
    "mistral-7b": "2023-09-27",
    "mixtral-8x7b": "2023-12-11",
    "mixtral-8x22b": "2024-04-17",
    "mistral-nemo-12b": "2024-07-18",
    "mistral-large": "2024-11-18",
    "mistral-small-24b": "2025-01-30",
    "ministral-3b": "2024-10-16",
    "ministral-8b": "2024-10-16",
    "mistral-medium-3.1": "2025-08-13",

    # Google Gemma/Gemini
    "gemma-2-2b": "2024-07-31",
    "gemma-2-27b": "2024-06-27",
    "gemma-3-1b": "2025-03-12",
    "gemma-3-4b": "2025-03-12",
    "gemma-3-12b": "2025-03-12",
    "gemma-3-27b": "2025-03-12",
    "gemma-3n-e4b": "2025-06-26",
    "gemma-4-26b-a4b": "2026-04-02",
    "gemma-4-31b": "2026-04-02",
    "gemini-2.0-flash": "2025-02-05",
    "gemini-2.5-flash": "2025-04-17",
    "gemini-2.5-flash-think": "2025-04-17",
    "gemini-2.5-flash-lite": "2025-06-17",
    "gemini-2.5-flash-lite-think": "2025-06-17",
    "gemini-2.5-pro": "2025-03-25",
    "gemini-2.5-pro-think": "2025-03-25",
    "gemini-3-flash": "2025-12-17",
    "gemini-3-flash-think": "2025-12-17",
    "gemini-3.1-pro": "2026-02-19",
    "gemini-3.1-flash-lite": "2026-03-03",

    # DeepSeek
    "deepseek-v3": "2024-12-26",
    "deepseek-r1-think": "2025-01-20",
    "deepseek-r1-distill-qwen-32b-think": "2025-01-20",
    "deepseek-r1-distill-llama-70b-think": "2025-01-20",
    "deepseek-v3.1": "2025-08-21",
    "deepseek-v3.2": "2025-09-29",
    "deepseek-v3.2-think": "2025-09-29",

    # Microsoft
    "phi-3-mini": "2024-04-23",
    "phi-4": "2024-12-12",

    # AllenAI
    "olmo-3.1-32b": "2025-11-20",  # Olmo 3 base family; 3.1 follow-up within same window

    # NVIDIA
    "nemotron-70b": "2024-10-15",
    "nemotron-super-49b-think": "2025-03-18",

    # OpenAI
    "gpt-3.5-turbo": "2022-11-30",
    "gpt-4": "2023-03-14",
    "gpt-4-turbo": "2023-11-06",
    "gpt-4o": "2024-05-13",
    "gpt-4o-mini": "2024-07-18",
    "gpt-4.1": "2025-04-14",
    "gpt-4.1-mini": "2025-04-14",
    "gpt-4.1-nano": "2025-04-14",
    "o1": "2024-12-05",
    "o3": "2025-04-16",
    "o3-mini": "2025-01-31",
    "o4-mini-think": "2025-04-16",
    "gpt-5": "2025-08-07",
    "gpt-5-think": "2025-08-07",
    "gpt-5-mini": "2025-08-07",
    "gpt-5-mini-think": "2025-08-07",
    "gpt-5-nano": "2025-08-07",
    "gpt-5-nano-think": "2025-08-07",
    "gpt-5-pro": "2025-10-06",
    "gpt-5.1": "2025-11-12",
    "gpt-5.2": "2025-12-11",
    "gpt-5.2-pro": "2025-12-11",
    "gpt-5.3": "2026-02-05",
    "gpt-5.4": "2026-03-05",
    "gpt-5.4-mini": "2026-03-17",
    "gpt-5.4-nano": "2026-03-17",
    "gpt-5.4-pro": "2026-03-05",
    "gpt-oss-20b-think": "2025-08-05",
    "gpt-oss-120b-think": "2025-08-05",

    # Anthropic Claude
    "claude-3-haiku": "2024-03-07",
    "claude-3.5-haiku": "2024-10-22",
    "claude-3.7-sonnet": "2025-02-24",
    "claude-3.7-sonnet-think": "2025-02-24",
    "claude-sonnet-4": "2025-05-22",
    "claude-sonnet-4-think": "2025-05-22",
    "claude-opus-4": "2025-05-22",
    "claude-opus-4-think": "2025-05-22",
    "claude-opus-4.1": "2025-08-05",
    "claude-opus-4.1-think": "2025-08-05",
    "claude-sonnet-4.5": "2025-09-29",
    "claude-sonnet-4.5-think": "2025-09-29",
    "claude-haiku-4.5": "2025-10-15",
    "claude-haiku-4.5-think": "2025-10-15",
    "claude-opus-4.5": "2025-11-24",
    "claude-opus-4.5-think": "2025-11-24",
    "claude-sonnet-4.6": "2026-02-17",
    "claude-sonnet-4.6-think": "2026-02-17",
    "claude-opus-4.6": "2026-02-05",
    "claude-opus-4.6-think": "2026-02-05",
    "claude-opus-4.7": "2026-04-16",
    "claude-opus-4.7-think": "2026-04-16",

    # xAI
    "grok-3": "2025-02-17",
    "grok-3-mini-think": "2025-02-17",
    "grok-4": "2025-07-09",
    "grok-4.20": "2026-02-17",
    "grok-4.20-think": "2026-02-17",

    # Cohere
    "command-r-plus": "2024-04-04",
    "command-r7b": "2024-12-13",
    "command-a": "2025-03-13",

    # Amazon
    "nova-micro": "2024-12-03",
    "nova-pro": "2024-12-03",
    "nova-premier": "2025-04-30",

    # AI21
    "jamba-large": "2024-08-22",

    # Moonshot
    "kimi-k2": "2025-07-11",
    "kimi-k2.5-think": "2026-01-27",
    "kimi-k2.6-think": "2026-04-20",

    # Baidu
    "ernie-4.5-300b-a47b": "2025-06-30",

    # ByteDance
    "seed-1.6-think": "2025-06-11",
    "seed-1.6-flash-think": "2025-06-11",
    "seed-2.0-lite-think": "2026-03-10",
    "seed-2.0-mini-think": "2026-02-26",

    # MiniMax
    "minimax-m1-think": "2025-06-16",
    "minimax-m2.7-think": "2026-03-18",

    # Zhipu
    "glm-4-32b": "2025-04-15",
    "glm-4.5-think": "2025-07-28",
    "glm-4.5-air-think": "2025-07-28",
    "glm-4.6-think": "2025-09-30",
    "glm-4.7-think": "2025-12-22",
    "glm-4.7-flash-think": "2025-12-22",
    "glm-5-think": "2026-02-11",
    "glm-5-turbo-think": "2026-02-11",
    "glm-5.1-think": "2026-04-07",

    # Tencent
    "hunyuan-a13b": "2025-06-27",
    "hunyuan-a13b-think": "2025-06-27",

    # StepFun
    "step-3.5-flash-think": "2026-01-29",

    # Xiaomi
    "mimo-v2-flash": "2025-12-16",
    "mimo-v2-flash-think": "2025-12-16",
    "mimo-v2-pro-think": "2026-03-18",

    # HuggingFace
    "smollm2-1.7b": "2024-11-01",

    # IBM
    "granite-3.3-2b": "2025-04-15",

    # Ant/Inclusion AI
    "ling-2.6-flash": "2026-04-22",
}


def main():
    cfg_path = Path("configs/all_models.json")
    cfg = json.load(open(cfg_path))
    models = cfg["models"]

    missing = [k for k in models if k not in RELEASE_DATES]
    if missing:
        print(f"WARNING: {len(missing)} models have no release_date entry:")
        for m in missing:
            print(f"  - {m}")

    extra = [k for k in RELEASE_DATES if k not in models]
    if extra:
        print(f"NOTE: {len(extra)} release_date entries don't match any model:")
        for m in extra:
            print(f"  - {m}")

    updated = 0
    for name, cfg_entry in models.items():
        if name in RELEASE_DATES:
            cfg_entry["release_date"] = RELEASE_DATES[name]
            updated += 1

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"Updated {updated}/{len(models)} models with release_date.")


if __name__ == "__main__":
    main()
