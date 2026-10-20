#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
"""
LSAT (n=20) version of compute_span_roles_table3.py's span-role F1 (one-vs-
rest, pooled over span1/span2/span3), using the 3-seed parsed predictions.

The parsed predictions (results/baseline_lsat/parsed/<model>/parsed.json) and
the argument review (results/baseline_lsat/annotation_review.json) contain the
LSAT argument text, which is not redistributed here. Point CQG_PROD_ROOT at a
folder that holds them; the script only reads from it. Outputs go to
results/revision/.

Join logic mirrors scripts/rescore_lsat_field_metrics_3seed.py exactly
(lines 46-69 there): predictions are joined to the review by normalized
FULL ARGUMENT TEXT (norm() = whitespace-collapse only, no lowercasing --
distinct from norm_str, which is applied to VALUES, not join keys), not by
numeric index. Gold span values are pulled separately, by argumentIndex,
from this repo's own data/lsat/lsat_a1_gold.json / lsat_a2_gold.json
. Because
the join key (review arg_index <-> parsed.json argument text) and the gold
key (argumentIndex) are different fields from different files, this script
explicitly asserts they identify the same argument for all 20 items before
scoring anything (via master_argument_id, present in both).

"No prediction" semantics are defined to match compute_span_roles_table3.py
exactly: there, an argument is skipped for a model/seed only when there is
literally no generated text for that seed (per_seed[s] absent). parsed.json
has no raw text, only the already-parsed dict, so the equivalent rule here
is: skip (treat as no prediction) only when per_seed is missing the seed
key entirely, or per_seed[s]["parsed"] is an empty dict -- NOT when parsed
exists but span1/span2/span3 happen to be blank inside it (that's a real
miss, scored like any other blank prediction, exactly as in Table 3).

"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = str(REPO)
PROD_ROOT = os.environ.get("CQG_PROD_ROOT", ROOT)  # read only
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Reuse the already-verified pooled_f1 / norm_str / ROLES / SLOTS from the
# Table 3 script, rather than reimplementing the metric.
_spec = importlib.util.spec_from_file_location(
    "span_roles_table3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_span_roles_table3.py"))
t3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t3)  # safe: its own work only happens under __main__

SLOTS = t3.SLOTS
ROLES = t3.ROLES
norm_str = t3.norm_str if hasattr(t3, "norm_str") else None
pooled_f1 = t3.pooled_f1
load_gold = t3.load_gold

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
SEEDS = ["44", "45", "46"]


def norm_join(t):
    """Exactly rescore_lsat_field_metrics_3seed.py's norm(): whitespace-
    collapse only, no lowercasing -- a different function from norm_str,
    used only for the argument-text JOIN key, never for value comparison."""
    return " ".join(str(t).split())


def load_review():
    path = os.path.join(PROD_ROOT, "results/baseline_lsat/annotation_review.json")
    with open(path) as f:  # read-only
        return json.load(f)


def load_parsed(model):
    path = os.path.join(PROD_ROOT, "results/baseline_lsat/parsed", model, "parsed.json")
    with open(path) as f:  # read-only
        return json.load(f)["samples"]


def main():
    report = {"skip_counts": {}, "join_failures": {}, "index_alignment_ok": None}

    review = load_review()  # [{arg_index, master_argument_id, argument, fields}]
    a1 = load_gold(os.path.join(ROOT, "data/lsat/lsat_a1_gold.json"))
    a2 = load_gold(os.path.join(ROOT, "data/lsat/lsat_a2_gold.json"))

    # ---- index-alignment check: review arg_index <-> gold argumentIndex ----
    review_by_idx = {int(r["arg_index"]): r for r in review}
    mismatches = []
    for idx in range(20):
        rv = review_by_idx.get(idx)
        gv = a1.get(idx)
        if rv is None or gv is None:
            mismatches.append((idx, "missing on one side"))
            continue
        if int(rv["master_argument_id"]) != int(gv["master_argument_id"]):
            mismatches.append((idx, f"master_argument_id {rv['master_argument_id']} != {gv['master_argument_id']}"))
    report["index_alignment_ok"] = (len(mismatches) == 0)
    report["index_mismatches"] = mismatches
    print("=== index-alignment check (review arg_index vs gold argumentIndex, via master_argument_id) ===")
    if mismatches:
        print(f"*** {len(mismatches)} MISMATCHES -- STOP, do not proceed ***")
        for m in mismatches:
            print(" ", m)
        json.dump(report, open(str(OUT / "span_roles_lsat.json"), "w"), indent=2)
        return report
    print(f"OK -- all 20 items aligned (arg_index == argumentIndex, master_argument_id matches)")

    # ---- join predictions to review by normalized argument text ----
    pmap = {}
    for m in MODELS:
        samples = load_parsed(m)
        key_to_sample = {norm_join(s["argument"]): s for s in samples}
        missing = [r["arg_index"] for r in review if norm_join(r["argument"]) not in key_to_sample]
        report["join_failures"][m] = missing
        assert not missing, f"join failure for {m}: args {missing}"
        pmap[m] = key_to_sample
    print("\n=== join check ===")
    print("OK -- zero join failures across all 8 models")

    # ---- build {model: {seed: {arg_index: parsed-dict-or-None}}}, honoring
    # the "no prediction" rule: None only if per_seed[s] missing or parsed=={} ----
    skip_counts = {m: {s: 0 for s in SEEDS} for m in MODELS}
    preds = {m: {s: {} for s in SEEDS} for m in MODELS}
    for m in MODELS:
        for r in review:
            idx = int(r["arg_index"])
            sample = pmap[m][norm_join(r["argument"])]
            for s in SEEDS:
                ps = sample.get("per_seed", {}).get(s)
                parsed = ps.get("parsed") if ps else None
                if not ps or not parsed:
                    preds[m][s][idx] = None
                    skip_counts[m][s] += 1
                else:
                    preds[m][s][idx] = parsed
    report["skip_counts"] = skip_counts
    print("\n=== 'no prediction' skip counts (per_seed missing or parsed=={}) ===")
    any_skips = False
    for m in MODELS:
        for s in SEEDS:
            if skip_counts[m][s]:
                any_skips = True
                print(f"  {m:10} seed {s}: {skip_counts[m][s]} args skipped")
    if not any_skips:
        print("  none -- every model has a non-empty parse for every argument, every seed")

    # ---- score: per model, per seed, pooled F1 per role; mean/std over seeds ----
    indices = list(range(20))
    results = {}
    for m in MODELS:
        per_seed_scores = [pooled_f1(a1, preds[m][s], indices) for s in SEEDS]
        out = {}
        for role in ROLES:
            f1s = [ps[role][0] for ps in per_seed_scores]
            ns = [ps[role][1] for ps in per_seed_scores]
            out[role] = {
                "mean": float(np.mean(f1s)), "std": float(np.std(f1s)),
                "per_seed": f1s, "n_pairs_mean": float(np.mean(ns)),
            }
        results[m] = out

    # ---- Human: gold-vs-gold, no seeds ----
    h_a2a1 = pooled_f1(a1, a2, indices)
    h_a1a2 = pooled_f1(a2, a1, indices)
    results["HUMAN_A2vsA1"] = {r: {"mean": h_a2a1[r][0], "n_pairs": h_a2a1[r][1]} for r in ROLES}
    results["HUMAN_A1vsA2"] = {r: {"mean": h_a1a2[r][0], "n_pairs": h_a1a2[r][1]} for r in ROLES}

    report["results"] = results
    out_path = str(OUT / "span_roles_lsat.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")

    # ---- CSV ----
    import csv
    csv_path = str(OUT / "span_roles_lsat.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "role", "seed44", "seed45", "seed46", "mean", "std", "n_pairs_mean"])
        for m in MODELS:
            for role in ROLES:
                r = results[m][role]
                w.writerow([m, role, *[round(x, 3) for x in r["per_seed"]],
                            round(r["mean"], 3), round(r["std"], 3), round(r["n_pairs_mean"], 1)])
        for h in ["HUMAN_A2vsA1", "HUMAN_A1vsA2"]:
            for role in ROLES:
                r = results[h][role]
                w.writerow([h, role, "", "", "", round(r["mean"], 3), 0.0, round(r["n_pairs"], 1)])
    print(f"Wrote {csv_path}")

    # ---- sanity gates ----
    print("\n=== sanity gate 1: re-run dev1-12 via compute_span_roles_table3.py ===")
    t3_results, t3_ok = t3.main()
    print("dev1-12 spot-checks:", "PASS" if t3_ok else "*** FAIL ***")

    print("\n=== sanity gate 2: Human LSAT vs expected ===")
    expected_a2a1 = {"major-claim": 97.0, "minor-claim": 82.8, "premise": 87.5}
    expected_a1a2 = {"major-claim": 97.0, "minor-claim": 85.7, "premise": 87.5}
    all_ok = t3_ok
    for role, exp in expected_a2a1.items():
        got = results["HUMAN_A2vsA1"][role]["mean"]
        d = abs(got - exp)
        ok = d <= 0.1
        all_ok &= ok
        print(f"  A2vsA1 {role:12} got={got:.1f} expected={exp:.1f} diff={d:.3f} {'OK' if ok else '*** MISMATCH ***'}")
    for role, exp in expected_a1a2.items():
        got = results["HUMAN_A1vsA2"][role]["mean"]
        d = abs(got - exp)
        ok = d <= 0.1
        all_ok &= ok
        print(f"  A1vsA2 {role:12} got={got:.1f} expected={exp:.1f} diff={d:.3f} {'OK' if ok else '*** MISMATCH ***'}")

    print("\n=== LSAT per-model results (3-seed mean+-std) ===")
    for m in MODELS:
        print(f"{m:10}", " | ".join(f"{role}={results[m][role]['mean']:5.1f}+-{results[m][role]['std']:.1f}" for role in ROLES))

    print("\nALL SANITY GATES PASS" if all_ok else "\n*** SOME SANITY GATES FAILED -- STOP, DO NOT WRITE TABLE ***")
    report["all_ok"] = all_ok
    json.dump(report, open(out_path, "w"), indent=2)
    return report


if __name__ == "__main__":
    main()
