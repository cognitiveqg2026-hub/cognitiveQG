import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U=str(OUT / "causation_exact_s-1over4") + "/"
means={(r[0],r[1]):(float(r[3]),float(r[4])) for r in list(csv.reader(open(U+"condition_means_s-1over4.csv")))[1:]}
M=["LLaMA-2","LLaMA-3","Mistral","OLMo","Qwen3","Qwen2.5","GPT-4o","GPT-5.5"]
conds=["vanilla","baseline","noise","oracle"]; labels=["Vanilla","Baseline","Noise","Oracle"]
cols=["#55A868","#4C72B0","#C44E52","#8172B2"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig,axes=plt.subplots(4,2,figsize=(8.0,10.6))
for ax,m in zip(axes.flat,M):
    v=[means[(m,c)][0] for c in conds]; e=[means[(m,c)][1] for c in conds]
    bars=ax.bar(range(4),v,yerr=e,color=cols,width=0.7,capsize=3,error_kw=dict(elinewidth=0.8,ecolor="#333333"))
    for i,(x,er) in enumerate(zip(v,e)):
        ax.text(i,x+er+0.01,f"{x:.3f}",ha="center",va="bottom",fontsize=7.5)
    ax.set_title(m,fontweight="bold",fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylim(0,0.75); ax.set_yticks([i/10 for i in range(8)])
    ax.set_ylabel("Mean Semantic Equivalence (0–1)",fontsize=7.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(h_pad=1.2,w_pad=2.0)
plt.savefig(str(OUT / "causation_judge_120_FIXED.pdf")); plt.savefig(str(OUT / "causation_judge_120_FIXED.png"),dpi=150)
