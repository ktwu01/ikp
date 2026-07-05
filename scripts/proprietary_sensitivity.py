"""Penalty-sensitivity analysis for proprietary model parameter estimates.

For each lambda in {0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0}:
  1. Refit log-linear calibration on the open-weight set (with that lambda).
  2. Predict effective capacity (in B) for each proprietary spotlight model.
  3. Tabulate predictions across lambda values to show estimate sensitivity.

Outputs a CSV: data/results/proprietary_sensitivity.csv
And a printed table on stdout.
"""

import json
import math
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ASSEMBLED = ROOT / "data" / "results" / "final_assembly.json"
RESULTS_V1 = ROOT / "data" / "results"
RESULTS_V2 = ROOT / "data" / "results_v2"
OUT = ROOT / "data" / "results" / "proprietary_sensitivity.csv"

CALIBRATION_EXCLUDE = {"minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think",
                       "hermes-3-405b", "ling-2.6-flash", "deepseek-v3.1-nex-n1",
                       "intellect-3-think"}
TIERS = ["T1","T2","T3","T4","T5","T6","T7"]
LAMBDAS = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0]

SPOTLIGHT = [
    # Frontier proprietary
    "gemini-3.1-pro", "gemini-3-flash", "gemini-3-flash-think",
    "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash",
    "claude-opus-4.7-think", "claude-opus-4.6-think", "claude-opus-4.5-think",
    "claude-sonnet-4.6-think", "claude-sonnet-4-think", "claude-haiku-4.5",
    "gpt-5-think", "gpt-5", "gpt-5.5-think", "gpt-5.4", "gpt-5.1",
    "gpt-5-mini", "gpt-5-nano", "o3", "o1", "o3-mini",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4-turbo", "gpt-4",
    "grok-3", "grok-4", "grok-4.20",
    "qwen3-max", "qwen3.6-plus-think", "qwen-plus", "qwen-max", "qwen-turbo",
    "kimi-k2", "kimi-k2.5-think", "kimi-k2.6-think",
    "mistral-medium-3.1", "mistral-large",
    "ernie-4.5-300b-a47b", "command-a", "hy3-preview",
    "nova-premier", "nova-pro", "nova-micro", "jamba-large",
    # Open with known params (for sanity)
    "deepseek-v4-flash", "deepseek-v4-pro",
    "deepseek-v4-flash-think", "deepseek-v4-pro-think",
    "deepseek-v3", "kimi-k2.5-think",
]


def score_verdict(verdict, lam):
    if verdict in ("CORRECT_STRONG","CORRECT"): return 1.0
    if verdict == "CORRECT_WEAK": return 0.5
    if verdict == "REFUSAL": return 0.0
    if verdict == "WRONG": return lam
    return 0.0


def assemble_one(model_name, lam):
    v1f = RESULTS_V1 / f"{model_name}.json"
    if not v1f.exists(): return None
    with open(v1f) as f: v1 = json.load(f)
    v2f = RESULTS_V2 / f"{model_name}.json"
    v2_results = {}
    if v2f.exists():
        try:
            with open(v2f) as f: v2 = json.load(f)
            results = v2.get("results", [])
            if sum(1 for r in results if r.get("model_query_error")) < len(results)*0.95:
                for r in results: v2_results[r["probe_id"]] = r
        except: pass
    by_t_sum = {t:0.0 for t in TIERS}; by_t_n = {t:0 for t in TIERS}
    for r in v1.get("results", []):
        pid = r.get("probe_id",""); tier = r.get("tier")
        if not tier: continue
        is_res = (r.get("source_type") == "researcher")
        if is_res and pid in v2_results:
            verdict = v2_results[pid].get("verdict")
        else:
            verdict = r.get("verdict")
        by_t_sum[tier] += score_verdict(verdict, lam)
        by_t_n[tier] += 1
    tacc = {t: max(by_t_sum[t]/by_t_n[t], 0.0) if by_t_n[t] else 0.0 for t in TIERS}
    return sum(tacc[t] for t in TIERS) / 7.0


def main():
    with open(ASSEMBLED) as f:
        rows = json.load(f)
    cfg_by_name = {r["model"]: r for r in rows}

    # For each lambda: fit calibration on open models, then predict spotlight
    fits = {}
    accs_by_lam = {}
    for lam in LAMBDAS:
        accs = {}
        log_params = []
        cal_accs = []
        for r in rows:
            acc = assemble_one(r["model"], lam)
            if acc is None: continue
            accs[r["model"]] = acc
            if (r.get("type") == "open" and r.get("params_B") and r["params_B"] > 0
                and r["model"] not in CALIBRATION_EXCLUDE):
                log_params.append(math.log10(r["params_B"]))
                cal_accs.append(acc)
        log_params = np.array(log_params); cal_accs = np.array(cal_accs)
        slope, intercept, rval, _, _ = stats.linregress(log_params, cal_accs)
        fits[lam] = {"slope": float(slope), "intercept": float(intercept),
                     "r_squared": float(rval**2), "n": len(log_params)}
        accs_by_lam[lam] = accs

    # Build prediction table: rows=models, cols=lambda
    out_rows = []
    for name in SPOTLIGHT:
        if name not in cfg_by_name: continue
        m = cfg_by_name[name]
        row = {"model": name, "actual_B": m.get("params_B"), "type": m.get("type")}
        for lam in LAMBDAS:
            acc = accs_by_lam[lam].get(name)
            if acc is None:
                row[f"acc_{lam}"] = None; row[f"pred_B_{lam}"] = None
                continue
            f = fits[lam]
            if f["slope"] == 0:
                row[f"acc_{lam}"] = acc; row[f"pred_B_{lam}"] = None; continue
            pred_log = (acc - f["intercept"]) / f["slope"]
            pred_B = 10 ** pred_log
            row[f"acc_{lam}"] = round(acc, 4)
            row[f"pred_B_{lam}"] = round(pred_B, 1)
        out_rows.append(row)

    # Save CSV
    fieldnames = ["model", "actual_B", "type"]
    for lam in LAMBDAS:
        fieldnames += [f"acc_{lam}", f"pred_B_{lam}"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(out_rows)

    # Print table — predicted B at each lambda
    print(f"\n=== Predicted effective capacity (B) at each lambda ===")
    print(f"\nCalibration fits (n=89 open models each):")
    print(f"  {'lambda':>8s} {'R²':>7s} {'slope':>7s} {'intercept':>10s}")
    for lam in LAMBDAS:
        f = fits[lam]
        print(f"  {lam:+6.2f}  {f['r_squared']:6.3f} {f['slope']:+7.3f} {f['intercept']:+10.4f}")

    print(f"\n=== Per-model predicted B by lambda ===")
    hdr = f"  {'model':30s} {'actual':>9s}  " + "  ".join(f"λ={l:+5.2f}" for l in LAMBDAS)
    print(hdr)
    for row in out_rows:
        actual = row["actual_B"]
        actual_s = f"{actual}B" if actual else "?"
        line = f"  {row['model']:30s} {actual_s:>9s}  "
        cells = []
        for lam in LAMBDAS:
            v = row.get(f"pred_B_{lam}")
            cells.append(f"{v:9.0f}" if v is not None else "        ?")
        line += "  ".join(cells)
        print(line)

    # ECR table for models with known params
    print(f"\n=== ECR (predicted / actual) for spotlight open models ===")
    print(hdr.replace("actual", "actual"))
    for row in out_rows:
        actual = row["actual_B"]
        if not actual: continue
        line = f"  {row['model']:30s} {actual:>8}B   "
        cells = []
        for lam in LAMBDAS:
            v = row.get(f"pred_B_{lam}")
            if v is not None:
                ecr = v / actual
                cells.append(f"{ecr:6.2f}×")
            else:
                cells.append("    ?")
        line += "    ".join(cells)
        print(line)

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
