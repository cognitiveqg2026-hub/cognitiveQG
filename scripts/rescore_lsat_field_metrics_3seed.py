#!/usr/bin/env python3
"""3-seed (44/45/46) mean +/- std rescore of the LSAT per-field baseline tables.

The original tables scored one prediction per argument (parse_model_output
'preferred' seed). This script builds one annotation review per seed from
parsed/<m>/parsed.json (per_seed[S].parsed), scores every seed with the
UNCHANGED scoring functions of compute_field_metrics_latex_rescaled_lsat6.py
and compute_field_metrics_jac_gwet_lsat6.py, and writes mean{\tiny$\pm$std}
tables with the same layouts and \\labels (drop-in replacements):

  results/baseline_lsat/field_metrics_lsat_rescaled_3seed.tex
  results/baseline_lsat/field_metrics_lsat_jac_gwet_3seed.tex

Run on the GPU host (BERTScore roberta-large):
  python scripts/rescore_lsat_field_metrics_3seed.py
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cfm = load_mod("cfm_lsat", "compute_field_metrics_latex_rescaled_lsat6.py")
cjg = load_mod("cjg_lsat", "compute_field_metrics_jac_gwet_lsat6.py")

SEEDS = ["44", "45", "46"]
MODELS = cfm.MODELS
# review field name -> parsed field name where they differ; fields absent from
# the parses (revisionPhases, revisionReason) fall back to "" for every seed,
# matching the original single-seed table (all-empty rows).
FIELD_ALIASES = {"trustworthinessExplanation": "trustExplanation"}


def norm(t):
    return " ".join(str(t).split())


review = json.loads((ROOT / "results/baseline_lsat/annotation_review.json").read_text())
pmap = {}
for m in MODELS:
    samples = json.loads(
        (ROOT / f"results/baseline_lsat/parsed/{m}/parsed.json").read_text())["samples"]
    pmap[m] = {norm(s["argument"]): s for s in samples}
    missing = [r["arg_index"] for r in review if norm(r["argument"]) not in pmap[m]]
    assert not missing, f"join failure for {m}: args {missing}"


def review_for_seed(seed):
    rev = copy.deepcopy(review)
    for row in rev:
        sample = {m: pmap[m][norm(row["argument"])] for m in MODELS}
        for fname, cell in row["fields"].items():
            pf = FIELD_ALIASES.get(fname, fname)
            for m in MODELS:
                val = sample[m]["per_seed"].get(seed, {}).get("parsed", {}).get(pf, "")
                cell[m] = "" if val is None else val
    return rev


def score_review(rev):
    out = {"rescaled": {}, "jacgwet": {}}
    bert_cache = cfm.precompute_bert_scores(rev)
    for phase, fields in cfm.PHASE_FIELDS.items():
        for field, display, metric in fields:
            for m in MODELS:
                if metric == "CAT":
                    s = cfm.compute_cat_score(rev, field, m)
                elif metric == "ML":
                    s = cfm.compute_ml_score(rev, field, m)
                else:
                    s = cfm.compute_ft_score(bert_cache, field, m)
                out["rescaled"][(field, m)] = s
    for phase, fields in cjg.PHASE_FIELDS.items():
        for field, display, metric in fields:
            for m in MODELS:
                out["jacgwet"][(field, m)] = cjg.field_score(rev, field, metric, m)
    return out


per_seed_scores = {}
for seed in SEEDS:
    print(f"=== scoring seed {seed} ===", flush=True)
    per_seed_scores[seed] = score_review(review_for_seed(seed))

# Reference: the original single-prediction review ('preferred' seed) for a
# drift report against the published single-seed tables.
print("=== scoring original (preferred-seed) review for comparison ===", flush=True)
orig_scores = score_review(review)


def agg(kind, field, m):
    vals = [per_seed_scores[s][kind].get((field, m)) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def cell(kind, field, m):
    mean, std = agg(kind, field, m)
    if mean is None:
        return "--", None
    if kind == "rescaled":
        return f"{mean * 100:.1f}" + r"{\tiny$\pm$" + f"{std * 100:.1f}" + "}", mean
    return f"{mean:.3f}" + r"{\tiny$\pm$" + f"{std:.3f}" + "}", mean


def build_rescaled():
    total_cols = 2 + len(MODELS)
    model_headers = " & ".join(r"\textbf{" + cfm.MODEL_LABELS[m] + r"}" for m in MODELS)
    lines = [
        r"% 3-seed mean+/-std rescore (seeds 44/45/46, per-seed parses).",
        r"% Source: scripts/rescore_lsat_field_metrics_3seed.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{3.2cm} l r r r r r r r r}",
        r"\hline",
        rf"\textbf{{Annotation Field}} & \textbf{{Metric}} & {model_headers} \\",
        r"\hline",
    ]
    for phase, fields in cfm.PHASE_FIELDS.items():
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")
        for field, display, metric in fields:
            cells = [cell("rescaled", field, m)[0] for m in MODELS]
            lines.append(f"{display} & {cfm.METRIC_LABEL[metric]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Per-field scores by cognitive phase on the LSAT batch "
        r"($n=20$, A1 gold; P1 prompt, mean\,$\pm$\,std over 3 inference seeds: "
        r"44, 45, 46). Wt.\,F1 = categorical weighted macro-F1; "
        r"Micro-F1 = multi-label set micro-F1; "
        r"BERTScore = BERTScore F1 (roberta-large). "
        r"Field ordering matches Table~\ref{tab:iaa-human}.}",
        r"\label{tab:model-f1-LSATbaseline}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def build_jacgwet():
    total_cols = 3 + len(MODELS)
    model_headers = " & ".join(r"\textbf{" + cjg.MODEL_LABELS[m] + r"}" for m in MODELS)
    lines = [
        r"% 3-seed mean+/-std rescore (seeds 44/45/46, per-seed parses).",
        r"% Source: scripts/rescore_lsat_field_metrics_3seed.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{3.2cm} l r r r r r r r r r}",
        r"\hline",
        rf"\textbf{{Annotation Field}} & \textbf{{Metric}} & {model_headers} & \textbf{{Avg}} \\",
        r"\hline",
    ]
    for phase, fields in cjg.PHASE_FIELDS.items():
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")
        for field, display, metric in fields:
            cells, means = [], []
            for m in MODELS:
                c, mean = cell("jacgwet", field, m)
                cells.append(c)
                if mean is not None:
                    means.append(mean)
            avg = f"{np.mean(means):.3f}" if means else "--"
            lines.append(f"{display} & {cjg.METRIC_LABEL[metric]} & "
                         + " & ".join(cells) + f" & {avg}" + r" \\")
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Per-field model-vs-gold scores by cognitive phase on the LSAT "
        r"batch ($n=20$, A1 gold; P1 prompt, mean\,$\pm$\,std over 3 inference "
        r"seeds: 44, 45, 46). Gwet's AC for categorical fields; Jaccard "
        r"similarity for span/text and multi-label fields. Metrics match the "
        r"human IAA study (Table~\ref{tab:iaa-human}) to enable direct "
        r"human--model comparison. Jaccard for span fields equals the span "
        r"component of the reasoning trace alignment score (\S\ref{sec:trace}).}",
        r"\label{tab:model-baseline-jac-gwet-LSATbaseline}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


out1 = ROOT / "results/baseline_lsat/field_metrics_lsat_rescaled_3seed.tex"
out2 = ROOT / "results/baseline_lsat/field_metrics_lsat_jac_gwet_3seed.tex"
out1.write_text(build_rescaled(), encoding="utf-8")
out2.write_text(build_jacgwet(), encoding="utf-8")
print(f"wrote {out1}\nwrote {out2}")

# ---- drift report: seed-44 column vs the published preferred-seed scores ----
drift = []
for kind in ("rescaled", "jacgwet"):
    for (field, m), orig in orig_scores[kind].items():
        s44 = per_seed_scores["44"][kind].get((field, m))
        if orig is None and s44 is None:
            continue
        if orig is None or s44 is None or abs(orig - s44) > 1e-9:
            drift.append((kind, field, m, orig, s44))
print(f"\n(field, model) cells where seed-44 differs from the published "
      f"preferred-seed value: {len(drift)}")
for row in drift[:12]:
    print("  ", row)

# per-seed macro means for a stability summary
for kind in ("rescaled", "jacgwet"):
    for s in SEEDS:
        vals = [v for v in per_seed_scores[s][kind].values() if v is not None]
        print(f"{kind} seed {s}: macro mean over cells = {np.mean(vals):.4f} (n={len(vals)})")
