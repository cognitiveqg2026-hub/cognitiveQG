import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
import json, os, sys
import numpy as np
sys.argv=[sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_wf1.py')).read().split("mode = sys.argv")[0])
ROLES=["major-claim","minor-claim","premise"]
SLOTS=["span1","span2","span3"]
def pooled(gold_rows, pred_rows):
    pairs=[]
    for g,p in zip(gold_rows,pred_rows):
        if p=="UNMATCHED": continue
        for s in SLOTS:
            gv=norm_str(g.get(s,""))
            if gv in ("","n/a"): continue
            pairs.append((gv, norm_str(p.get(s,""))))
    out={}
    for r in ROLES:
        tp=sum(1 for a,b in pairs if a==r and b==r); fp=sum(1 for a,b in pairs if a!=r and b==r); fn=sum(1 for a,b in pairs if a==r and b!=r)
        pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
        out[r]=(2*pr*rc/(pr+rc)*100 if pr+rc else 0.0, tp+fn)
    out["wtf1"]=(macro_f1_categorical([a for a,_ in pairs],[b for _,b in pairs])*100, len(pairs))
    return out
golds={"A1":[a1[i] for i in idx],"A2":[a2[i] for i in idx]}
res={}
print("HUMAN A2 vs A1:", {k:round(v[0],1) for k,v in pooled(golds["A1"],golds["A2"]).items()}, "support", {k:v[1] for k,v in pooled(golds["A1"],golds["A2"]).items()})
print("HUMAN A1 vs A2:", {k:round(v[0],1) for k,v in pooled(golds["A2"],golds["A1"]).items()})
MODELS=["llama2","llama3","qwen","mistral","olmo","qwen3","gpt-4o","gpt-5.5"]
for m in MODELS:
    d=json.load(open(R+f"/data/baseline_complete_dev1-12/raw/{m}/raw_output.json"))
    by={int(s["argument_index"]):s for s in d["per_sample"]}
    for gname,grows in golds.items():
        acc={k:[] for k in ROLES+["wtf1"]}
        for seed in ("44","45","46"):
            preds=[]
            for i in idx:
                t=by.get(i,{}).get("per_seed",{}).get(seed,{}).get("generated_text","")
                preds.append(parse_model_output(t) if t else "UNMATCHED")
            o=pooled(grows,preds)
            for k in acc: acc[k].append(o[k][0])
        res[(m,gname)]={k:(float(np.mean(v)),float(np.std(v))) for k,v in acc.items()}
        print(m,gname," ".join(f"{k}={res[(m,gname)][k][0]:.1f}±{res[(m,gname)][k][1]:.1f}" for k in acc))
json.dump({f"{m}|{g}":v for (m,g),v in res.items()},open(str(OUT / "span_roles_human_check.json"),"w"))
