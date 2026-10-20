import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
import json, os, sys
sys.argv=[sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_wf1.py')).read().split("mode = sys.argv")[0])
# A1 vs A2 (gold=A2, pred=A1)
g=[a2[i] for i in idx]; p=[a1[i] for i in idx]
for f in ROWS:
    v,n=score(f,g,p); print(f"{f:22s} {'--' if v is None else f'{v:5.1f}'} n={n}")
print("--- validate LLaMA-2 vs A2 gold (Table 27 says Stance 40.0, Knowledge 2.0)")
d=json.load(open(R+"/data/baseline_complete_dev1-12/raw/llama2/raw_output.json"))
by={int(s["argument_index"]):s for s in d["per_sample"]}
for f in ["stanceTarget","knowledgeDomain","reasoningType","logicalFallacy"]:
    vals=[]
    for seed in ("44","45","46"):
        preds=[parse_model_output(by[i]["per_seed"][seed]["generated_text"]) if by.get(i,{}).get("per_seed",{}).get(seed,{}).get("generated_text") else "UNMATCHED" for i in idx]
        v,_=score(f,g,preds); vals.append(v)
    print(f, round(sum(vals)/3,1))
