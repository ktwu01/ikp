#!/usr/bin/env python3
"""Generate the proprietary frontier-model estimate table (Table tab:frontier)
from website/public/data/calibration.json at the current operating penalty (λ=0).

Dedups thinking/non-thinking variants (keeps higher accuracy), excludes the
Gemini T6 landmark family, and emits paper/tables/frontier_estimates.tex.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAL = json.load(open(ROOT / "website" / "public" / "data" / "calibration.json"))
OUT = ROOT / "paper" / "tables" / "frontier_estimates.tex"

# Excluded from the estimate table (landmarks whose T6 score is inflated by construction).
EXCLUDE = {"gemini-3.1-pro", "gemini-3-flash", "gemini-3-flash-think",
           "gemini-3.1-flash-lite", "gemini-3-flash-lite", "gemini-3-flash-lite-think"}
MIN_EST_B = 60  # only frontier-scale rows

VENDOR = {"openai":"OpenAI","anthropic":"Anthropic","google":"Google","xai":"xAI",
          "alibaba":"Alibaba","deepseek":"DeepSeek","moonshot":"Moonshot","zhipu":"Z.ai",
          "minimax":"MiniMax","stepfun":"StepFun","xiaomi":"Xiaomi","amazon":"Amazon",
          "bytedance":"ByteDance","tencent":"Tencent"}

FAMILY = {"gpt":"GPT-", "grok":"Grok-", "o1":"o1", "o3":"o3", "o4":"o4",
          "claude":"Claude", "gemini":"Gemini", "qwen":"Qwen", "qwen3":"Qwen3",
          "glm":"GLM-", "kimi":"Kimi", "minimax":"MiniMax", "nemotron":"Nemotron",
          "nova":"Nova", "seed":"Seed", "mimo":"MiMo", "step":"Step-", "hunyuan":"Hunyuan",
          "deepseek":"DeepSeek", "ernie":"ERNIE", "mistral":"Mistral"}
CAP = {"pro","mini","nano","turbo","flash","max","plus","lite","air","code","preview",
       "opus","sonnet","haiku","premier","turbo","chat","instruct"}
def pretty(name):
    n = re.sub(r"-think$", "", name)
    parts = n.split("-")
    out = []
    for i, p in enumerate(parts):
        low = p.lower()
        if i == 0 and low in FAMILY:
            fam = FAMILY[low]
            out.append(fam)  # may end with '-' to attach the version
        elif low in CAP:
            out.append((" " if out and not out[-1].endswith("-") else "") + p.capitalize())
        elif re.match(r"^v?\d", low):  # version token: 5.5, 4.1, 4o, 2.5, v3
            out.append((" " if out and out[-1][-1:].isalpha() else "") + p)
        else:
            out.append((" " if out and not out[-1].endswith("-") else "") + (p.upper() if len(p) <= 2 else p.capitalize()))
    s = "".join(out)
    s = re.sub(r"-\s+", "-", s)            # 'GPT- 5.5' -> 'GPT-5.5'
    return s.strip()

def fsize(b):
    if b is None: return "--"
    if b >= 1000: return f"${{\\sim}}{b/1000:.1f}$T"
    return f"${{\\sim}}{b:.0f}$B"
def fpi(lo, hi):
    def f(v): return f"{v/1000:.1f}T" if v and v>=1000 else (f"{v:.0f}B" if v else "--")
    return f"[{f(lo)}--{f(hi)}]"

# dedup think/non-think by base name, keep higher accuracy
best = {}
for e in CAL["proprietary_estimates"]:
    if e["model"] in EXCLUDE: continue
    if not e.get("estimated_B") or e["estimated_B"] < MIN_EST_B: continue
    base = re.sub(r"-think$", "", e["model"])
    if base not in best or e["accuracy"] > best[base]["accuracy"]:
        best[base] = e

# Drop superseded legacy models to keep the table current-frontier.
def is_legacy(n):
    n = re.sub(r"-think$", "", n)
    if re.match(r"gpt-(3\.5|4($|-turbo|o|\.1-mini|\.1-nano))", n): return True
    if n in {"o1", "o1-mini", "o3-mini", "o4-mini",
             "gpt-5-mini", "gpt-5-nano", "gpt-5.4-mini",
             "gemini-2.0-flash", "gemini-2.5-flash",
             "claude-3.5-sonnet", "claude-3.7-sonnet"}: return True
    if n.startswith("claude-3-") or n.startswith("claude-3.5-haiku"): return True
    if n.startswith("nova-") or n.startswith("seed-") or re.match(r"mimo-v2($|-)", n): return True
    return False

# Always surface the frontier models added in this round, even if below the cut.
MUST_INCLUDE = {"claude-opus-4.8", "claude-sonnet-5", "claude-fable-5", "qwen3.7-plus",
                "qwen3.7-max", "minimax-m3", "glm-5.2", "glm-5-turbo", "step-3.7-flash"}
MAX_ROWS = 30
ranked = [e for e in sorted(best.values(), key=lambda e: -e["estimated_B"])
          if not is_legacy(e["model"])]
rows = ranked[:MAX_ROWS]
have = {e["model"] for e in rows}
for e in ranked:
    if e["model"] in MUST_INCLUDE and e["model"] not in have:
        rows.append(e)
rows.sort(key=lambda e: -e["estimated_B"])
lines = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Model & Vendor & Accuracy & Est.\ Size & 90\% PI \\", r"\midrule"]
for e in rows:
    v = VENDOR.get(e.get("vendor",""), (e.get("vendor") or "").capitalize())
    thinking = " (think)" if e.get("thinking") else ""
    lines.append(f"{pretty(e['model'])}{thinking} & {v} & {e['accuracy']*100:.1f}\\% "
                 f"& {fsize(e['estimated_B'])} & {fpi(e.get('pi_lo'), e.get('pi_hi'))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} with {len(rows)} rows (λ=0). Top 6:")
for e in rows[:6]:
    print(f"  {pretty(e['model']):22s} {e['accuracy']*100:.1f}%  {fsize(e['estimated_B'])}")
