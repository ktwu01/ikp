#!/usr/bin/env python3
"""Dense vs MoE calibration analysis (λ=0, cleaned set).

Emits:
  paper/tables/moe_dense_fits.tex   -- dense / MoE-total / MoE-active / combined fits
  paper/tables/moe_dense_sens.tex   -- frontier estimates: combined vs MoE-total curve
and prints the LOO recovery comparison used in the appendix text.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
s = {m["model"]: m for m in json.load(open(ROOT / "data/results/evaluation_summary.json"))}
cfg = json.load(open(ROOT / "configs/all_models.json"))["models"]
cal = json.load(open(ROOT / "website/public/data/calibration.json"))
pts = cal["calibration_points"]
prop = {e["model"]: e for e in cal["proprietary_estimates"]}

def logN(p, key="params_B"): return math.log10(cfg[p["model"]][key])
def acc(p): return p["accuracy"]
def fit(items, key="params_B"):
    x = np.array([logN(p, key) for p in items]); y = np.array([acc(p) for p in items])
    sl, ic, r, _, _ = stats.linregress(x, y); return sl, ic, r**2, len(items)

dense = [p for p in pts if cfg[p["model"]].get("arch") == "dense"]
moe   = [p for p in pts if cfg[p["model"]].get("arch") == "moe"]
moe_a = [p for p in moe if cfg[p["model"]].get("active_B")]

Dsl, Dic, Dr, Dn = fit(dense)
Msl, Mic, Mr, Mn = fit(moe)
Asl, Aic, Ar, An = fit(moe_a, "active_B")
Csl, Cic, Cr, Cn = fit(pts)

# ---- fit table ----
fits = [("Dense", Dn, Dsl, Dic, Dr),
        ("MoE (total params)", Mn, Msl, Mic, Mr),
        ("MoE (active params)", An, Asl, Aic, Ar),
        ("Combined (headline)", Cn, Csl, Cic, Cr)]
t1 = [r"\begin{tabular}{lrrrr}", r"\toprule",
      r"Calibration subset & $n$ & Slope & Intercept & $R^2$ \\", r"\midrule"]
for name, n, sl, ic, r2 in fits:
    b = r"\textbf" if name.startswith("Combined") else lambda x: x
    t1.append(f"{name} & {n} & {sl:.3f} & {ic:.3f} & {r2:.3f} \\\\")
t1 += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "paper/tables/moe_dense_fits.tex").write_text("\n".join(t1) + "\n")

# ---- LOO recovery: MoE-curve vs combined, on known-size MoE ----
def loo_pred(target, pool, key="params_B"):
    others = [p for p in pool if p["model"] != target["model"]]
    x = np.array([logN(p, key) for p in others]); y = np.array([acc(p) for p in others])
    sl, ic, _, _, _ = stats.linregress(x, y)
    return 10 ** ((acc(target) - ic) / sl)
def fold(pred, act): return max(pred / act, act / pred)
fm, fc = [], []
for p in moe:
    a = cfg[p["model"]]["params_B"]
    fm.append(fold(loo_pred(p, moe), a)); fc.append(fold(loo_pred(p, pts), a))
fm, fc = np.array(fm), np.array(fc)
large = np.array([cfg[p["model"]]["params_B"] >= 100 for p in moe])
print("=== LOO recovery of known-size MoE (median fold / within-2x) ===")
print(f"  all MoE (n={len(fm)}):   MoE-curve {np.median(fm):.2f}x/{np.mean(fm<=2)*100:.0f}%   combined {np.median(fc):.2f}x/{np.mean(fc<=2)*100:.0f}%")
print(f"  large>=100B (n={large.sum()}): MoE-curve {np.median(fm[large]):.2f}x/{np.mean(fm[large]<=2)*100:.0f}%   combined {np.median(fc[large]):.2f}x/{np.mean(fc[large]<=2)*100:.0f}%")

# ---- frontier sensitivity: combined vs MoE curve ----
SPOT = ["claude-fable-5","gpt-5.5-pro","gpt-5.5","gemini-2.5-pro","gpt-4.1","claude-opus-4.7"]
NAME = {"claude-fable-5":"Claude Fable 5","gpt-5.5-pro":"GPT-5.5 Pro","gpt-5.5":"GPT-5.5",
        "gemini-2.5-pro":"Gemini 2.5 Pro","gpt-4.1":"GPT-4.1","claude-opus-4.7":"Claude Opus 4.7"}
def fs(b): return f"${{\\sim}}{b/1000:.1f}$T" if b >= 1000 else f"${{\\sim}}{b:.0f}$B"
inv = lambda a, sl, ic: 10 ** ((a - ic) / sl)
t2 = [r"\begin{tabular}{lrrr}", r"\toprule",
      r"Model & Accuracy & Combined & MoE-curve \\", r"\midrule"]
for m in SPOT:
    a = s[m]["accuracy"]
    t2.append(f"{NAME[m]} & {a*100:.1f}\\% & {fs(inv(a,Csl,Cic))} & {fs(inv(a,Msl,Mic))} \\\\")
t2 += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "paper/tables/moe_dense_sens.tex").write_text("\n".join(t2) + "\n")
print("\nwrote paper/tables/moe_dense_fits.tex, moe_dense_sens.tex")
print(f"combined slope {Csl:.3f} vs dense {Dsl:.3f} / MoE {Msl:.3f}; "
      f"MoE-curve raises frontier est by ~{(inv(0.83,Msl,Mic)/inv(0.83,Csl,Cic)-1)*100:.0f}% at acc=0.83")
