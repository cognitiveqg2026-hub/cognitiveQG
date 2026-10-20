#!/usr/bin/env python3
"""Span Eval. (span1/span2/span3) rescaled BERTScore F1 -- human ceiling (dev1-12) and LSAT (models + human).

Uses exactly the per-field BERTScore rule of scripts/compute_field_metrics_latex_rescaled.py
(precompute_bert_scores / compute_ft_score), which produced the Span Eval rows of Tables 27-30:
  * values normalised with norm_str (lowercase, whitespace-collapsed; first item if a list)
  * gold "" or "n/a"  -> argument skipped
  * no prediction for the argument (no generated text / empty parse) -> skipped (the "UNMATCHED" rule)
  * empty predicted value -> score 0
  * otherwise bert_score.score(cands=[pred], refs=[gold], model_type="roberta-large", lang="en",
    rescale_with_baseline=True); field score = mean over scored arguments (reported x100)
Run in the BERTScore environment used for the paper: bert-score 0.3.12, transformers 5.9.0, tokenizers 0.22.2.

Outputs results/revision/span_eval_bertscore.json:
  dev1-12 : HUMAN_A2vsA1 / HUMAN_A1vsA2  {span1, span2, span3}           (Tables 3, 27, 28)
  lsat    : A1 / A2  {"span<k>|<model>": {mean, std, per_seed}}           (Tables 11, 32, 33)
            HUMAN_A2vsA1 / HUMAN_A1vsA2 {span1, span2, span3}
STOP-AND-REPORT: the script first reproduces published cells; if any differs by more than 0.1 it
writes nothing and exits with status 1. Do not adjust anything to make a gate pass.
"""
import importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # scripts/revision (paths.py)
from paths import REPO, OUT
sys.path.insert(0, os.path.join(str(REPO), "scripts"))
from compute_field_metrics_latex_rescaled import norm_str      # same normaliser as the tables
import torch
from bert_score import score as bert_score_fn
import transformers, bert_score

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

T3 = _load("t3", os.path.join(os.path.dirname(HERE), "span_roles", "compute_span_roles_table3.py"))
LS = _load("ls", os.path.join(os.path.dirname(HERE), "span_roles", "compute_span_roles_lsat.py"))
MODELS, SEEDS, SLOTS = T3.MODELS, T3.SEEDS, ["span1", "span2", "span3"]
ROOT = str(REPO)

PAIRS, CACHE = [], {}
def need(p, g):
    if (p, g) not in CACHE: CACHE[(p, g)] = None; PAIRS.append((p, g))

def plan(gold, pred_by_idx, field, indices):
    """Return list of ('skip'|'zero'|(p,g)) per argument, registering BERTScore pairs."""
    out = []
    for i in indices:
        g = norm_str(gold.get(i, {}).get(field, "")) if i in gold else ""
        pr = pred_by_idx.get(i)
        if g in ("", "n/a") or pr is None:
            out.append("skip"); continue
        p = norm_str(pr.get(field, ""))
        if p == "":
            out.append("zero"); continue
        need(p, g); out.append((p, g))
    return out

def value(plan_list):
    xs = [0.0 if x == "zero" else CACHE[x] for x in plan_list if x != "skip"]
    return 100.0 * sum(xs) / len(xs) if xs else None

def run_bertscore():
    todo = [pg for pg in PAIRS if CACHE[pg] is None]
    if not todo: return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, F = bert_score_fn([p for p, g in todo], [g for p, g in todo], model_type="roberta-large",
                            lang="en", rescale_with_baseline=True, verbose=False, device=device)
    for pg, f in zip(todo, F.tolist()): CACHE[pg] = f

def main():
    print("bert-score", bert_score.__version__, "| transformers", transformers.__version__)
    # ---------- dev1-12 ----------
    raw_root = os.path.join(ROOT, "data/baseline_complete_dev1-12/raw")
    a1 = T3.load_gold(os.path.join(ROOT, "data/a1_dev1-12_gold.json"))
    a2 = T3.load_gold(os.path.join(ROOT, "data/a2_dev1-12_gold.json"))
    idx = sorted(T3.DEV1_12_IDX)
    dev_preds = {m: T3.load_preds(m, raw_root) for m in ["llama3", "gpt-4o", "gpt-5.5"]}
    dev_plans = {}
    for m, preds in dev_preds.items():
        for gname, gold in [("A1", a1), ("A2", a2)]:
            for f in SLOTS:
                dev_plans[(m, gname, f)] = [plan(gold, preds[s], f, idx) for s in SEEDS]
    hum_dev = {("HUMAN_A2vsA1", f): plan(a1, a2, f, idx) for f in SLOTS}
    hum_dev.update({("HUMAN_A1vsA2", f): plan(a2, a1, f, idx) for f in SLOTS})

    # ---------- LSAT ----------
    review = LS.load_review()
    la1 = T3.load_gold(os.path.join(ROOT, "data/lsat/lsat_a1_gold.json"))
    la2 = T3.load_gold(os.path.join(ROOT, "data/lsat/lsat_a2_gold.json"))
    for r in review:   # same alignment assertion as compute_span_roles_lsat.py
        assert int(r["master_argument_id"]) == int(la1[int(r["arg_index"])]["master_argument_id"])
    lpreds = {}
    for m in MODELS:
        km = {LS.norm_join(s["argument"]): s for s in LS.load_parsed(m)}
        lpreds[m] = {s: {} for s in SEEDS}
        for r in review:
            smp = km[LS.norm_join(r["argument"])]
            for s in SEEDS:
                ps = smp.get("per_seed", {}).get(s)
                parsed = ps.get("parsed") if ps else None
                lpreds[m][s][int(r["arg_index"])] = parsed if parsed else None
    lidx = list(range(20))
    lsat_plans = {(m, g, f): [plan(gold, lpreds[m][s], f, lidx) for s in SEEDS]
                  for m in MODELS for g, gold in [("A1", la1), ("A2", la2)] for f in SLOTS + ["goodEvidence"]}
    hum_lsat = {("HUMAN_A2vsA1", f): plan(la1, la2, f, lidx) for f in SLOTS}
    hum_lsat.update({("HUMAN_A1vsA2", f): plan(la2, la1, f, lidx) for f in SLOTS})

    print(f"BERTScore on {len(PAIRS)} distinct (pred, gold) pairs ...")
    run_bertscore()

    def ms(plans):
        v = [value(p) for p in plans]; v = [x for x in v if x is not None]
        return (float(np.mean(v)), float(np.std(v)), v) if v else (None, None, [])

    # ---------- gates: reproduce published cells ----------
    expected = {("llama3", "A1"): (69.4, 47.8, 98.2), ("gpt-4o", "A1"): (68.8, 47.5, 98.4),
                ("gpt-5.5", "A1"): (69.6, 54.1, 96.9), ("llama3", "A2"): (75.1, 51.1, 98.3),
                ("gpt-4o", "A2"): (75.0, 50.9, 98.5), ("gpt-5.5", "A2"): (73.0, 54.3, 97.2)}
    ok = True
    print("\n=== Gate 1: dev1-12 Span Eval cells (Tables 27/28) ===")
    for (m, g), exp in expected.items():
        for f, e in zip(SLOTS, exp):
            got = ms(dev_plans[(m, g, f)])[0]
            good = got is not None and abs(got - e) <= 0.1; ok &= good
            print(f"  {m:8} {g} {f}: got {got:.2f} expected {e:.1f} {'OK' if good else '*** MISMATCH ***'}")
    print("\n=== Gate 2: LSAT Good Evidence vs lsat_meanref_result.json ===")
    ref = json.load(open(str(OUT / "lsat_meanref_result.json")))
    for m in MODELS:
        for g in ["A1", "A2"]:
            got = ms(lsat_plans[(m, g, "goodEvidence")])[0]
            e = ref[g]["fields"][f"goodEvidence|{m}"]["mean"] * 100
            good = got is not None and abs(got - e) <= 0.1; ok &= good
            print(f"  {m:8} {g}: got {got:.2f} expected {e:.2f} {'OK' if good else '*** MISMATCH ***'}")
    if not ok:
        print("\n*** GATE FAILED -- nothing written. Report the mismatches; do not adjust the method. ***")
        sys.exit(1)

    res = {"env": {"bert_score": bert_score.__version__, "transformers": transformers.__version__},
           "dev1-12": {h: {f: value(hum_dev[(h, f)]) for f in SLOTS} for h in ["HUMAN_A2vsA1", "HUMAN_A1vsA2"]},
           "lsat": {}}
    for g in ["A1", "A2"]:
        res["lsat"][g] = {}
        for m in MODELS:
            for f in SLOTS:
                mean, std, per = ms(lsat_plans[(m, g, f)])
                res["lsat"][g][f"{f}|{m}"] = {"mean": mean, "std": std, "per_seed": per}
    for h in ["HUMAN_A2vsA1", "HUMAN_A1vsA2"]:
        res["lsat"][h] = {f: value(hum_lsat[(h, f)]) for f in SLOTS}
    path = str(OUT / "span_eval_bertscore.json")
    json.dump(res, open(path, "w"), indent=2)
    print("\nALL GATES PASS. Wrote", path)
    print("dev1-12 human:", json.dumps(res["dev1-12"]))
    print("LSAT human:", json.dumps({h: res["lsat"][h] for h in ["HUMAN_A2vsA1", "HUMAN_A1vsA2"]}))
    for g in ["A1", "A2"]:
        for f in SLOTS:
            print(f"LSAT {g} {f}: " + "  ".join(f"{m}={res['lsat'][g][f'{f}|{m}']['mean']:.1f}" for m in MODELS))

if __name__ == "__main__":
    main()
