#!/usr/bin/env python3
"""
Tables 6, 21, 35, 43 and Figure 2 data: per-model Pearson r AND Spearman
rho (each with its own Fisher-z/bootstrap CI and Holm/BH correction family,
per the Table 6 caption: "Corrections are applied separately across the
eight Pearson and eight Spearman tests") between the per-argument Combined
Trace Score and GPT-Sem, under several score variants and judges.

Writes:
  out/table6_21_figure2_under_A.csv      -- per-model Pearson+Spearman r/rho/CI/Holm/BH under A (primary judge)
  out/table35_under_A_and_Aeq.csv        -- Pearson r/CI/Holm under A and A_eq
  out/table43_judge_robustness.csv       -- Holm-significant count per judge, under A
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stats_utils as su
from judge_io import load_judge_se

ALPHA = 0.05
SEED = 42
N_BOOT = 10000

MODELS = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]
JUDGES = {"gpt55": "GPT-5.5", "gpt6astra": "GPT-6 Astra", "gpt56sol": "GPT-5.6 Sol"}


def load_per_arg_scores(scores_csv: Path) -> dict:
    """Returns {model: {arg_index: {col: value}}}."""
    out = {}
    with open(scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["model"]
            idx = int(row["arg_index"])
            out.setdefault(m, {})[idx] = {k: float(v) for k, v in row.items()
                                          if k not in ("arg_index", "model")}
    return out


def correlate(score_by_model, judge_se_by_model, models, score_col):
    """Per-model Pearson AND Spearman r/rho vs. GPT-Sem, each with its own
    Holm/BH correction family (Table 6 / 21: 'Corrections are applied
    separately across the eight Pearson and eight Spearman tests')."""
    rows = []
    pear_pvals, spear_pvals, ms = [], [], []
    for model in models:
        se = judge_se_by_model.get(model, {})
        sc = score_by_model.get(model, {})
        common = sorted(set(sc.keys()) & set(se.keys()))
        if len(common) < 4:
            continue
        x = [sc[i][score_col] for i in common]
        y = [se[i] for i in common]
        pear = su.pearson_with_ci(x, y, ALPHA, N_BOOT, SEED)
        spear = su.spearman_with_ci(x, y, ALPHA, N_BOOT, SEED)
        rows.append({"model": model, "n": pear["n"], "pearson_r": pear["r"],
                     "pearson_p": pear["p"], "ci95_lo": pear["ci95_fisherz"][0],
                     "ci95_hi": pear["ci95_fisherz"][1],
                     "spearman_rho": spear["rho"], "spearman_p": spear["p"],
                     "spearman_ci95_lo": spear["ci95_bootstrap"][0],
                     "spearman_ci95_hi": spear["ci95_bootstrap"][1]})
        pear_pvals.append(pear["p"])
        spear_pvals.append(spear["p"])
        ms.append(model)
    holm_p, holm_rej = su.holm_bonferroni(pear_pvals, ALPHA) if pear_pvals else ([], [])
    bh_p, bh_rej = su.benjamini_hochberg(pear_pvals, ALPHA) if pear_pvals else ([], [])
    s_holm_p, s_holm_rej = su.holm_bonferroni(spear_pvals, ALPHA) if spear_pvals else ([], [])
    s_bh_p, s_bh_rej = su.benjamini_hochberg(spear_pvals, ALPHA) if spear_pvals else ([], [])
    for i, m in enumerate(ms):
        for row in rows:
            if row["model"] == m:
                row["holm_p"] = holm_p[i]
                row["holm_reject"] = holm_rej[i]
                row["bh_p"] = bh_p[i]
                row["bh_reject"] = bh_rej[i]
                row["spearman_holm_p"] = s_holm_p[i]
                row["spearman_holm_reject"] = s_holm_rej[i]
                row["spearman_bh_p"] = s_bh_p[i]
                row["spearman_bh_reject"] = s_bh_rej[i]
    return rows


def write_csv(rows, path: Path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(repr(r[c]) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")


def main(scores_csv: str, judge_dir: str, out_dir: str):
    scores_csv = Path(scores_csv)
    judge_dir = Path(judge_dir)
    out_dir = Path(out_dir)

    score_by_model = load_per_arg_scores(scores_csv)

    # --- Table 6 / 21 / Figure 2: under A, primary judge (gpt55) ---
    judge_se = {m: load_judge_se(judge_dir / f"{m}_gpt55_judge.json") for m in MODELS}
    rows_a = correlate(score_by_model, judge_se, MODELS, "A")
    write_csv(rows_a, out_dir / "table6_21_figure2_under_A.csv",
              ["model", "n", "pearson_r", "ci95_lo", "ci95_hi", "pearson_p", "holm_p", "holm_reject", "bh_p", "bh_reject",
               "spearman_rho", "spearman_ci95_lo", "spearman_ci95_hi", "spearman_p",
               "spearman_holm_p", "spearman_holm_reject", "spearman_bh_p", "spearman_bh_reject"])
    print("=== Table 6/21/Figure2 (under A, GPT-5.5 judge) ===")
    for r in rows_a:
        print(f"  {r['model']:10s} r={r['pearson_r']:+.4f} p={r['pearson_p']:.4f} Holm={r['holm_p']:.4f}  "
              f"rho={r['spearman_rho']:+.4f} p={r['spearman_p']:.4f} Holm={r['spearman_holm_p']:.4f}")

    # --- Table 35: under A and A_eq ---
    rows_eq = correlate(score_by_model, judge_se, MODELS, "A_eq")
    with open(out_dir / "table35_under_A_and_Aeq.csv", "w") as f:
        f.write("model,A_r,A_ci95_lo,A_ci95_hi,A_holm_p,Aeq_r,Aeq_ci95_lo,Aeq_ci95_hi,Aeq_holm_p\n")
        a_by_m = {r["model"]: r for r in rows_a}
        eq_by_m = {r["model"]: r for r in rows_eq}
        for m in MODELS:
            ra, re = a_by_m.get(m), eq_by_m.get(m)
            if not ra or not re:
                continue
            f.write(f"{m},{ra['pearson_r']!r},{ra['ci95_lo']!r},{ra['ci95_hi']!r},{ra['holm_p']!r},"
                    f"{re['pearson_r']!r},{re['ci95_lo']!r},{re['ci95_hi']!r},{re['holm_p']!r}\n")
    print("\n=== Table 35 (A vs A_eq) written ===")

    # --- A_old correlation (previously-published score, for comparison) ---
    rows_old = correlate(score_by_model, judge_se, MODELS, "A_old")
    write_csv(rows_old, out_dir / "table35_under_Aold.csv",
              ["model", "n", "pearson_r", "ci95_lo", "ci95_hi", "pearson_p", "holm_p", "holm_reject", "bh_p", "bh_reject"])
    n_sig_old = sum(1 for r in rows_old if r["holm_reject"])
    print(f"\n=== A_old (previously published): {n_sig_old}/8 Holm-significant ===")

    # --- Table 43: 3-judge robustness, under A ---
    robustness_rows = []
    for jkey, jlabel in JUDGES.items():
        jse = {m: load_judge_se(judge_dir / f"{m}_{jkey}_judge.json") for m in MODELS}
        r = correlate(score_by_model, jse, MODELS, "A")
        n_sig = sum(1 for row in r if row["holm_reject"])
        sig_models = [row["model"] for row in r if row["holm_reject"]]
        robustness_rows.append({"judge": jlabel, "n_holm_significant": n_sig, "models": ";".join(sig_models)})
    with open(out_dir / "table43_judge_robustness.csv", "w") as f:
        f.write("judge,n_holm_significant_of_8,models\n")
        for r in robustness_rows:
            f.write(f"{r['judge']},{r['n_holm_significant']},{r['models']}\n")
    print("\n=== Table 43 (3-judge robustness, under A) ===")
    for r in robustness_rows:
        print(f"  {r['judge']:12s} {r['n_holm_significant']}/8 Holm-significant: {r['models']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    main(args.scores_csv, args.judge_dir, args.out_dir)
