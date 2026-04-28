"""Preset model lists for the IKP CLI.

Six landmarks (one per tier boundary, T1–T6) plus three state-of-the-art
frontier models. The landmark list mirrors `pipeline/landmarks.py`. The
SOTA list intentionally uses GPT-5.5 (not GPT-5.4); IDs are looked up
against `configs/all_models.json`.
"""

LANDMARKS = [
    {"name": "qwen2.5-0.5b",   "id": "qwen2.5:0.5b",                 "type": "ollama",     "thinking": False, "tier": "T1"},
    {"name": "qwen2.5-7b",     "id": "qwen/qwen-2.5-7b-instruct",    "type": "openrouter", "thinking": False, "tier": "T2"},
    {"name": "qwen3-32b",      "id": "qwen/qwen3-32b",               "type": "openrouter", "thinking": True,  "tier": "T3"},
    {"name": "qwen3-235b",     "id": "qwen/qwen3-235b-a22b",         "type": "openrouter", "thinking": True,  "tier": "T4"},
    {"name": "kimi-k2.5",      "id": "moonshotai/kimi-k2.5",         "type": "openrouter", "thinking": True,  "tier": "T5"},
    {"name": "gemini-3.1-pro", "id": "google/gemini-3.1-pro-preview", "type": "openrouter", "thinking": False, "tier": "T6"},
]

SOTA = [
    {"name": "gpt-5.5",         "id": "openai/gpt-5.5",            "type": "openrouter", "thinking": False, "tier": "SOTA"},
    {"name": "deepseek-v4-pro", "id": "deepseek/deepseek-v4-pro",  "type": "openrouter", "thinking": False, "tier": "SOTA"},
    {"name": "claude-opus-4.7", "id": "anthropic/claude-opus-4.7", "type": "openrouter", "thinking": False, "tier": "SOTA"},
]

PRESET = LANDMARKS + SOTA
