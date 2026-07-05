#!/usr/bin/env python3
"""Fig: what changes across the hallucination penalty lambda.

Left  — calibration fit quality (R^2, 90% PI factor) vs lambda: nearly flat.
Right — flagship frontier estimates (with 90% bands) vs lambda: move a lot.
Together they motivate the parsimony principle: the choice does not change
conclusions, so we take the parameter-free point (lambda=0).
"""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
s = {m["model"]: m for m in json.load(open(ROOT / "data/results/evaluation_summary.json"))}
cfg = json.load(open(ROOT / "configs/all_models.json"))["models"]
OUT = ROOT / "paper" / "figures"
EXCL = {"minimax-m1-think","hunyuan-a13b","hunyuan-a13b-think","hermes-3-405b","ling-2.6-flash",
        "nemotron-ultra-253b","deepseek-v3.1-nex-n1","intellect-3-think"}
TIERS = ["T1","T2","T3","T4","T5","T6","T7"]
LAM = [0.0,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0]
SPOT = [("Claude Fable 5","claude-fable-5"),("GPT-5.5","gpt-5.5"),
        ("GPT-4.1","gpt-4.1"),("Claude Opus 4.7","claude-opus-4.7")]
COL = {"Claude Fable 5":"#7570B3","GPT-5.5":"#1B9E77","GPT-4.1":"#D95F02","Claude Opus 4.7":"#E7298A"}

opens = [m for m in s.values() if cfg.get(m["model"],{}).get("type")=="open"
         and cfg.get(m["model"],{}).get("params_B") and cfg[m["model"]]["params_B"]>0
         and m["model"] not in EXCL]
logN = np.array([math.log10(cfg[m["model"]]["params_B"]) for m in opens])

def acc(ts, lam, floor=True):
    xs=[]
    for t in TIERS:
        v=ts.get(t)
        if not v or not v.get("total"): xs.append(0.0); continue
        sc=(v["correct"]+lam*v["wrong"])/v["total"]
        xs.append(max(sc,0.0) if floor else sc)
    return float(np.mean(xs))

R2=[]; PIF=[]; EST={l[0]:[] for l in SPOT}; LO={l[0]:[] for l in SPOT}; HI={l[0]:[] for l in SPOT}
for lam in LAM:
    a=np.array([acc(m["tier_stats"],lam) for m in opens])
    sl,ic,r,_,_=stats.linregress(logN,a)
    resid=a-(sl*logN+ic); se=math.sqrt(float(np.sum(resid**2))/(len(a)-2))
    pif=10**(1.645*se/abs(sl)); R2.append(r**2); PIF.append(pif)
    for label,name in SPOT:
        av=acc(s[name]["tier_stats"],lam); e=10**((av-ic)/sl)
        EST[label].append(e); LO[label].append(e/pif); HI[label].append(e*pif)

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"figure.dpi":150,"savefig.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6))
x=np.array(LAM)

# Left: fit quality
c1="#2166AC"; c2="#D95F02"
ax1.plot(x,R2,"o-",color=c1,lw=2,label=r"$R^2$")
ax1.set_ylim(0.85,0.95); ax1.set_ylabel(r"Calibration $R^2$",color=c1)
ax1.tick_params(axis="y",labelcolor=c1)
ax1.set_xlabel(r"Hallucination penalty $\lambda$")
axb=ax1.twinx(); axb.plot(x,PIF,"s--",color=c2,lw=2,label="90% PI factor")
axb.set_ylabel(r"90% PI factor ($\times$)",color=c2); axb.tick_params(axis="y",labelcolor=c2)
axb.spines["top"].set_visible(False)
ax1.axvline(0,color="gray",ls=":",lw=1); ax1.text(0.02,0.86,"used",fontsize=8,color="gray")
ax1.set_title("Fit quality is nearly flat across $\\lambda$")

# Right: estimates
for label,_ in SPOT:
    c=COL[label]
    ax2.plot(x,np.array(EST[label])/1000,"o-",color=c,lw=2,label=label)
    ax2.fill_between(x,np.array(LO[label])/1000,np.array(HI[label])/1000,color=c,alpha=0.08)
ax2.set_yscale("log"); ax2.set_xlabel(r"Hallucination penalty $\lambda$")
ax2.set_ylabel("Estimated parameters (T)")
ax2.axvline(0,color="gray",ls=":",lw=1)
ax2.set_title("Individual estimates move by $2$--$3\\times$")
ax2.legend(fontsize=8,loc="upper left")
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(OUT/f"fig_lambda_sensitivity.{ext}",bbox_inches="tight",dpi=300)
print("saved fig_lambda_sensitivity.pdf/.png")
print("R2 range %.3f-%.3f; PI range %.2f-%.2f"%(min(R2),max(R2),min(PIF),max(PIF)))
