#!/usr/bin/env python3
"""Generate the appendix full-results longtables (per-tier accuracy + hallucination
rate) for all evaluated models from the current lambda=0 data.

Writes paper/tables/full_accuracy.tex and paper/tables/full_hallucination.tex
(complete longtable environments, \\input by paper/appendix.tex).
"""
import json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
summ = json.load(open(ROOT / "data/results/evaluation_summary.json"))
cfg = json.load(open(ROOT / "configs/all_models.json"))["models"]
fit = json.load(open(ROOT / "website/public/data/calibration.json"))["fit"]
SL, IC = fit["slope"], fit["intercept"]
TIERS = ["T1","T2","T3","T4","T5","T6","T7"]

def esc(s): return s.replace("_", r"\_").replace("&", r"\&")
def fp(b):
    if b is None: return "--"
    if b >= 1000: return f"{b/1000:.1f}T"
    if b >= 1: return f"{b:.0f}B"
    return f"{b*1000:.0f}M"

# enrich rows
rows = []
for m in summ:
    c = cfg.get(m["model"], {})
    rows.append({**m, "params_B": c.get("params_B", m.get("params_B")),
                 "arch": c.get("arch","unknown"), "type": c.get("type","unknown"),
                 "thinking": c.get("thinking", False), "vendor": c.get("vendor", m.get("vendor","?"))})

# group by vendor, order vendors by best accuracy, models within vendor by accuracy desc
byv = defaultdict(list)
for r in rows: byv[r["vendor"]].append(r)
vendors = sorted(byv, key=lambda v: -max(x["accuracy"] for x in byv[v]))

def ikp_pred(r):
    if r["type"] != "open" or not r["params_B"]:
        return "--"
    pred = 10 ** ((r["accuracy"] - IC) / SL)
    ratio = pred / r["params_B"]
    cell = f"{fp(pred)} ({ratio:.1f}$\\times$)"
    return f"\\textbf{{{cell}}}" if (ratio > 2 or ratio < 0.5) else cell

VENDOR = {"openai":"OpenAI","anthropic":"Anthropic","google":"Google","xai":"xAI",
          "alibaba":"Alibaba","deepseek":"DeepSeek","moonshot":"Moonshot","zhipu":"Z.ai",
          "minimax":"MiniMax","nvidia":"NVIDIA","stepfun":"StepFun","xiaomi":"Xiaomi",
          "meta":"Meta","mistral":"Mistral","microsoft":"Microsoft","amazon":"Amazon",
          "bytedance":"ByteDance","tencent":"Tencent","huggingface":"HuggingFace",
          "cohere":"Cohere","ai21":"AI21","allenai":"AllenAI","01ai":"01.AI","baidu":"Baidu"}

def hallu(ts, t):
    s = ts.get(t)
    if not s: return 0.0
    d = s.get("wrong",0)+s.get("correct",0)+s.get("refusal",0)
    return s.get("wrong",0)/d if d else 0.0

acc_rows, hal_rows = [], []
for v in vendors:
    grp = sorted(byv[v], key=lambda x: -x["accuracy"])
    for i, r in enumerate(grp):
        vlabel = VENDOR.get(v, v.capitalize()) if i == 0 else ""
        think = "Y" if r["thinking"] else ""
        arch = r["arch"] if r["arch"] in ("dense","moe") else "--"
        tvals = " & ".join(f"{r['tier_accuracy'].get(t,0):.2f}" for t in TIERS)
        acc_rows.append(f"{vlabel} & {esc(r['model'])} & {fp(r['params_B'])} & {ikp_pred(r)} & {arch} & {think} "
                        f"& {r['accuracy']:.3f} & {r['raw_accuracy']:.3f} & {tvals} \\\\")
        hvals = " & ".join(f"{hallu(r['tier_stats'], t):.2f}" for t in TIERS)
        hal_rows.append(f"{vlabel} & {esc(r['model'])} & {r['accuracy']:.3f} & {r['raw_accuracy']:.3f} & {hvals} \\\\")

ACC_HDR = r"\textbf{Vendor} & \textbf{Model} & \textbf{Params} & \textbf{IKP Pred.} & \textbf{Arch} & \textbf{Think} & \textbf{Acc.} & \textbf{Raw} & \textbf{T1} & \textbf{T2} & \textbf{T3} & \textbf{T4} & \textbf{T5} & \textbf{T6} & \textbf{T7} \\"
acc = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{longtable}{@{}ll r r l c rr rrrrrrr@{}}",
  r"\caption{Full model results: per-tier accuracy at $\lambda=0$ (no penalty). Models grouped by vendor, sorted by accuracy. \textbf{Acc.}\ is the tier-mean accuracy used throughout; \textbf{Raw} is overall correct/total. The \textbf{IKP Pred.} column shows the calibration's predicted parameter count and ratio to actual for open-weight models; ratios $>2\times$ or $<0.5\times$ are bolded as systematic outliers. Proprietary models show `--'.}",
  r"\label{tab:full-accuracy} \\", r"\toprule", ACC_HDR, r"\midrule", r"\endfirsthead",
  r"\toprule", ACC_HDR, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"] + acc_rows + [r"\end{longtable}"]
(ROOT / "paper/tables/full_accuracy.tex").write_text("\n".join(acc) + "\n")

HAL_HDR = r"\textbf{Vendor} & \textbf{Model} & \textbf{Acc.} & \textbf{Raw} & \textbf{T1} & \textbf{T2} & \textbf{T3} & \textbf{T4} & \textbf{T5} & \textbf{T6} & \textbf{T7} \\"
hal = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{longtable}{@{}ll rr rrrrrrr@{}}",
  r"\caption{Per-tier hallucination rate = wrong / (wrong + correct + refusal). Higher = more confident incorrect answers. Models sorted as in Table~\ref{tab:full-accuracy}.}",
  r"\label{tab:full-hallucination} \\", r"\toprule", HAL_HDR, r"\midrule", r"\endfirsthead",
  r"\toprule", HAL_HDR, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"] + hal_rows + [r"\end{longtable}"]
(ROOT / "paper/tables/full_hallucination.tex").write_text("\n".join(hal) + "\n")
print(f"wrote full_accuracy.tex ({len(acc_rows)} rows) + full_hallucination.tex ({len(hal_rows)} rows)")
print("vendors:", len(vendors))
