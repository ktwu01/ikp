#!/usr/bin/env python3
"""Thorough ablation of the two scoring hyperparameters flagged by the LessWrong
critique: the hallucination penalty lambda AND per-tier flooring at zero.

For each (lambda, floor) cell it recomputes every model's accuracy from stored
per-tier verdict counts, refits the log-linear calibration on the 93-model
open-weight set, and reports fit quality + frontier estimates with 90% bands.

Key point demonstrated: at lambda=0 the floor is a no-op (tier score =
correct/total >= 0), so lambda=0 removes BOTH hyperparameters at once.

Writes results/tables/lambda_floor_ablation.{json,txt}.
"""
import json, math
from itertools import product
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = json.load(open(ROOT / "data" / "results" / "evaluation_summary.json"))
CFG = json.load(open(ROOT / "configs" / "all_models.json"))["models"]
OUTJSON = ROOT / "results" / "tables" / "lambda_floor_ablation.json"
OUTTXT = ROOT / "results" / "tables" / "lambda_floor_ablation.txt"

TIERS = ["T1","T2","T3","T4","T5","T6","T7"]
LAMBDAS = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0]
EXCLUDE = {"minimax-m1-think","hunyuan-a13b","hunyuan-a13b-think","hermes-3-405b",
           "ling-2.6-flash","nemotron-ultra-253b","deepseek-v3.1-nex-n1","intellect-3-think"}
# critique's spotlight + our new frontier targets
SPOT = ["gpt-5.5","gpt-5.5-pro","claude-opus-4.7","gemini-3.1-pro","claude-fable-5",
        "gpt-4.1","claude-opus-4.8","claude-sonnet-5"]

by = {m["model"]: m for m in SUMMARY}

def acc(ts, lam, floor):
    xs = []
    for t in TIERS:
        s = ts.get(t)
        if not s or not s.get("total"): xs.append(0.0); continue
        v = (s["correct"] + lam*s["wrong"]) / s["total"]
        xs.append(max(v, 0.0) if floor else v)
    return float(np.mean(xs))

opens = [m for m in SUMMARY if CFG.get(m["model"],{}).get("type")=="open"
         and CFG.get(m["model"],{}).get("params_B") and CFG[m["model"]]["params_B"]>0
         and m["model"] not in EXCLUDE]
logN = np.array([math.log10(CFG[m["model"]]["params_B"]) for m in opens])

def cell(lam, floor):
    a = np.array([acc(m["tier_stats"], lam, floor) for m in opens])
    # forward fit A = alpha*log10N + beta
    alpha, beta, r, _, _ = stats.linregress(logN, a)
    resid = a - (alpha*logN + beta)
    se = math.sqrt(float(np.sum(resid**2))/max(len(a)-2,1))
    pi = 10**(1.645*se/abs(alpha)) if alpha else float("inf")
    # estimator slope S in log10N = S*acc + I (what ikp_estimate reports; critique's 6.79)
    S, I, rS, _, _ = stats.linregress(a, logN)
    # LOO
    folds=[]; n=len(opens)
    for i in range(n):
        mk=np.ones(n,bool); mk[i]=False
        al,be,_,_,_=stats.linregress(logN[mk],a[mk])
        if al>0: folds.append(10**abs((a[i]-be)/al - logN[i]))
    med=float(np.median(folds)); w2=float(np.mean(np.array(folds)<=2)); w3=float(np.mean(np.array(folds)<=3))
    # estimates
    est={}
    for name in SPOT:
        m=by.get(name)
        if m and alpha>0:
            av=acc(m["tier_stats"], lam, floor)
            e=10**((av-beta)/alpha)
            est[name]={"acc":av,"est_B":e,"lo":e/pi,"hi":e*pi}
        else: est[name]=None
    return {"lambda":lam,"floor":floor,"n":n,"R2":r**2,"alpha_pp":alpha*100,
            "estimator_slope_S":S,"pi_factor":pi,"loo_med":med,"within2":w2,"within3":w3,
            "estimates":est}

results=[cell(l,f) for l,f in product(LAMBDAS,[True,False])]
OUTJSON.parent.mkdir(parents=True, exist_ok=True)
OUTJSON.write_text(json.dumps(results, indent=2))

def fs(b): return f"{b/1000:.1f}T" if b and b>=1000 else (f"{b:.0f}B" if b else "--")
lines=[]
lines.append(f"{'λ':>6} {'floor':>6} {'R²':>6} {'α(pp)':>6} {'S':>6} {'PI×':>5} {'LOO×':>5} {'≤2×':>5} | "
             + " ".join(f"{s.split('-')[0][:7]:>7}" for s in ["gpt-5.5","opus-4.7","gemini-3.1","fable-5","gpt-4.1"]))
for r in results:
    e=r["estimates"]
    def g(n): return fs(e[n]["est_B"]) if e.get(n) else "--"
    lines.append(f"{r['lambda']:>6.2f} {str(r['floor']):>6} {r['R2']:>6.3f} {r['alpha_pp']:>6.1f} "
                 f"{r['estimator_slope_S']:>6.2f} {r['pi_factor']:>5.2f} {r['loo_med']:>5.2f} {r['within2']*100:>4.0f}% | "
                 f"{g('gpt-5.5'):>7} {g('claude-opus-4.7'):>7} {g('gemini-3.1-pro'):>7} {g('claude-fable-5'):>7} {g('gpt-4.1'):>7}")
txt="\n".join(lines)
OUTTXT.write_text(txt+"\n")
print(txt)

# LaTeX table for the appendix (shows the floor's effect on fit + two estimates)
def fs2(b): return (f"{b/1000:.1f}T" if b>=1000 else f"{b:.0f}B") if b else "--"
tex=[r"\begin{tabular}{rlrrrrrr}", r"\toprule",
     r"$\lambda$ & floor & $R^2$ & Slope & PI$\times$ & LOO$\times$ & GPT-5.5 & Gemini 3.1 Pro \\",
     r" & & & (pp/dec) & & & est. & est. \\", r"\midrule"]
for r in results:
    e=r["estimates"]
    g=lambda n: fs2(e[n]["est_B"]) if e.get(n) else "--"
    fl = "on" if r["floor"] else "off"
    star = r"$^\star$" if (r["lambda"]==0.0 and r["floor"]) else ""
    tex.append(f"${r['lambda']:.2f}${star} & {fl} & {r['R2']:.3f} & {r['alpha_pp']:.1f} & "
               f"{r['pi_factor']:.2f} & {r['loo_med']:.2f} & {g('gpt-5.5')} & {g('gemini-3.1-pro')} \\\\")
    if r["lambda"]==0.0 and r["floor"]:  # note the no-op
        tex.append(r"\multicolumn{8}{l}{\footnotesize\quad(at $\lambda=0$ the floor is a no-op: scores are $\text{correct}/\text{total}\geq 0$)} \\")
tex+=[r"\bottomrule", r"\end{tabular}"]
TEX = ROOT / "paper" / "tables" / "lambda_floor_ablation.tex"
TEX.write_text("\n".join(tex)+"\n")
print(f"[wrote {TEX}]")
print(f"\n[wrote {OUTJSON} and {OUTTXT}]")
print("\nNote: at λ=0 the floor is a no-op (scores are correct/total ≥ 0); the two λ=0 rows are identical.")
