#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
"""
Span-role F1 (one-vs-rest, pooled over span1/span2/span3 and all arguments)
for the Table 3 variants, replacing the old per-slot BERTScore "Span Eval."
rows -- scoring a one-word categorical role label with BERTScore is not
meaningful; this computes a real classification metric instead.

Per role, the score is plain (not support-weighted) one-vs-rest binary F1
over pairs pooled across all three span slots.

"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = str(REPO)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from parse_model_output import parse_model_output  # noqa: E402
from compute_field_metrics_latex_rescaled import norm_str  # noqa: E402

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
SEEDS = ["44", "45", "46"]
SLOTS = ["span1", "span2", "span3"]
ROLES = ["major-claim", "minor-claim", "premise"]
ROLE_DISPLAY = {"major-claim": "Major claim", "minor-claim": "Minor claim", "premise": "Premise"}

# dev3-12 excludes the dev1-2 calibration round (argumentIndex 35..54).
EXCLUDE_DEV1_2 = set(range(35, 55))
DEV1_12_IDX = set(range(120))
DEV3_12_IDX = DEV1_12_IDX - EXCLUDE_DEV1_2


def load_gold(path):
    d = json.load(open(path))
    anns = d.get("annotations", d.get("completedAnnotations", []))
    return {int(a["argumentIndex"]): a for a in anns}


def load_preds(model, raw_root):
    path = os.path.join(raw_root, model, "raw_output.json")
    raw = json.load(open(path))
    preds = {s: {} for s in SEEDS}
    for sample in raw["per_sample"]:
        idx = int(sample["argument_index"])
        for s in SEEDS:
            txt = sample.get("per_seed", {}).get(s, {}).get("generated_text", "")
            preds[s][idx] = parse_model_output(txt) if txt else None
    return preds


def pooled_f1(gold_by_idx, pred_by_idx, indices):
    """pred_by_idx[idx] is a dict-like (parsed prediction, or another
    annotator's gold dict) or None (no prediction at all for that
    argument -> excluded entirely, mirrors UNMATCHED). Returns
    {role: (f1_pct, n_pairs)}."""
    pairs = []
    for idx in indices:
        g = gold_by_idx.get(idx)
        if g is None:
            continue
        p = pred_by_idx.get(idx)
        for slot in SLOTS:
            gv = norm_str(g.get(slot, ""))
            if gv in ("", "n/a"):
                continue
            if p is None:
                continue
            pv = norm_str(p.get(slot, ""))
            pairs.append((gv, pv))
    out = {}
    for role in ROLES:
        tp = sum(1 for a, b in pairs if a == role and b == role)
        fp = sum(1 for a, b in pairs if a != role and b == role)
        fn = sum(1 for a, b in pairs if a == role and b != role)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) * 100 if (pr + rc) else 0.0
        out[role] = (f1, tp + fn)
    return out


def model_mean_std(model, gold_by_idx, raw_root, indices):
    """{role: (mean_f1, std_f1, mean_n_pairs)} across 3 seeds."""
    preds = load_preds(model, raw_root)
    per_seed = [pooled_f1(gold_by_idx, preds[s], indices) for s in SEEDS]
    out = {}
    for role in ROLES:
        f1s = [ps[role][0] for ps in per_seed]
        ns = [ps[role][1] for ps in per_seed]
        out[role] = (float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(ns)))
    return out


def main():
    raw_root = os.path.join(ROOT, "data/baseline_complete_dev1-12/raw")
    a1 = load_gold(os.path.join(ROOT, "data/a1_dev1-12_gold.json"))
    a2 = load_gold(os.path.join(ROOT, "data/a2_dev1-12_gold.json"))

    results = {}  # (model_or_HUMAN, gold_ref, batch) -> {role: (mean,std,n)}

    for batch_name, indices in [("dev1-12", DEV1_12_IDX), ("dev3-12", DEV3_12_IDX)]:
        for gold_name, gold in [("A1", a1), ("A2", a2)]:
            for m in MODELS:
                results[(m, gold_name, batch_name)] = model_mean_std(m, gold, raw_root, indices)

    # Human, dev1-12 only (A1/A2 full annotations, both directions), no seeds/std.
    for batch_name, indices in [("dev1-12", DEV1_12_IDX), ("dev3-12", DEV3_12_IDX)]:
        h_a2vsa1 = pooled_f1(a1, a2, indices)
        h_a1vsa2 = pooled_f1(a2, a1, indices)
        results[("HUMAN_A2vsA1", "-", batch_name)] = {r: (v[0], 0.0, v[1]) for r, v in h_a2vsa1.items()}
        results[("HUMAN_A1vsA2", "-", batch_name)] = {r: (v[0], 0.0, v[1]) for r, v in h_a1vsa2.items()}

    out_path = str(OUT / "span_roles_table3.json")
    dump = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}
    json.dump(dump, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")

    # ---- spot-check against the 6 known reference numbers ----
    print("\n=== spot-check vs given reference values (dev1-12) ===")
    checks = [
        ("llama3", "A1", "dev1-12", {"major-claim": 95.5, "minor-claim": 0.0, "premise": 63.0}),
        ("gpt-5.5", "A1", "dev1-12", {"major-claim": 88.7, "minor-claim": 40.4, "premise": 66.5}),
    ]
    all_ok = True
    for key0, key1, key2, expected in checks:
        got = results[(key0, key1, key2)]
        for role, exp in expected.items():
            g = got[role][0]
            d = abs(g - exp)
            ok = d <= 0.1
            all_ok &= ok
            print(f"  {key0:8} {role:12} got={g:6.1f}  expected={exp:6.1f}  diff={d:.3f}  {'OK' if ok else '*** MISMATCH ***'}")

    human_checks = [
        ("HUMAN_A2vsA1", {"major-claim": 93.8, "minor-claim": 75.5, "premise": 86.9}),
        ("HUMAN_A1vsA2", {"major-claim": 94.6, "minor-claim": 77.1, "premise": 88.5}),
    ]
    for key0, expected in human_checks:
        got = results[(key0, "-", "dev1-12")]
        for role, exp in expected.items():
            g = got[role][0]
            d = abs(g - exp)
            ok = d <= 0.1
            all_ok &= ok
            print(f"  {key0:14} {role:12} got={g:6.1f}  expected={exp:6.1f}  diff={d:.3f}  {'OK' if ok else '*** MISMATCH ***'}")

    print("\nALL SPOT-CHECKS PASS" if all_ok else "\n*** SOME SPOT-CHECKS FAILED -- STOP, DO NOT PROCEED TO TABLE EDITS ***")

    # ---- full dev1-12 / A1 table for all 8 models (what feeds the tables) ----
    print(f"\n{'model':10} {'major':>18} {'minor':>18} {'premise':>18}")
    for m in MODELS:
        r = results[(m, "A1", "dev1-12")]
        print(f"{m:10} " + " ".join(f"{r[role][0]:7.1f}+-{r[role][1]:4.1f}(n={r[role][2]:.0f})" for role in ROLES))
    print("HUMAN A2vsA1  " + " ".join(f"{results[('HUMAN_A2vsA1','-','dev1-12')][role][0]:7.1f}(n={results[('HUMAN_A2vsA1','-','dev1-12')][role][2]:.0f})" for role in ROLES))
    print("HUMAN A1vsA2  " + " ".join(f"{results[('HUMAN_A1vsA2','-','dev1-12')][role][0]:7.1f}(n={results[('HUMAN_A1vsA2','-','dev1-12')][role][2]:.0f})" for role in ROLES))

    return results, all_ok


if __name__ == "__main__":
    main()
