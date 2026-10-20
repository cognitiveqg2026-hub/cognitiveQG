# CognitiveQG Combined Trace Score — reproduction package

## What this is

A self-contained reproduction of the paper's per-argument **Combined
Reasoning Trace Score**, `A`, and every published table/figure built from
it. No API keys, no GPU, no internet access needed — everything runs from
cached model outputs and judge scores already shipped in `data/`.

## The formula

Eq. (2) of the manuscript, exactly:

```
A_i = 16/41 * EM_i + 4/41 * J_multi_i + 21/41 * J_span_i
```

for argument `i`, model vs. expert (A1) gold:

- `EM_i` — mean **exact match** over 16 single-label fields (`revisionPhases`
  compared as a label **set**: exact match = identical sets, not string
  equality — its gold is list-valued)
- `J_multi_i` — mean **set Jaccard** over 4 multi-label fields
- `J_span_i` — mean **token-level Jaccard** over 21 span/text fields

A field is skipped for an argument when its gold value is `n/a` or empty.
`A` does **not** use Gwet's AC1 anywhere — AC1 is a corpus-level agreement
statistic (used only in the paper's human–human IAA table, out of scope
here) and cannot be computed for a single argument.

**Field accounting:** 44 annotation fields = 41 trace fields + 3 Socratic
question fields (excluded as the dependent variable). All 41 trace fields
are scored — every one of them weighs exactly `1/41`, split across the
three metric groups by field count (16/41 EM + 4/41 ML + 21/41 FT = 41/41).
An earlier version of this package (and an earlier draft of the paper)
left `revisionPhases` and `revisionReason` out of the score entirely; both
are now scored like every other field.

**Why field-count weights now, not the earlier fixed 0.390/0.098/0.512?**
Because Eq. (2) defines the weights as each group's literal share of the
41 scored fields — 16/41 ≈ 0.390, 4/41 ≈ 0.098, 21/41 ≈ 0.512 are the
*exact fractions*, not independently-chosen rounded constants. The
previously-published score (0.390/0.098/0.512 over a 12/4/23, 39-field
basis that excluded `revisionPhases`/`revisionReason`) is still computed
and shipped as `A_old`, purely for side-by-side comparison — it is not
part of Eq. (2) and is not the paper's primary score going forward.

## What's in the package

```
src/score_trace.py        the scorer (no imports from the original repo)
src/stats_utils.py        Pearson+Fisher-z CI, Spearman+bootstrap CI, Holm, BH
src/judge_io.py           loads judge-score files -> per-argument GPT-Sem
src/analysis_correlation.py   Tables 6, 21, 35, 43 + Figure 2 data
src/analysis_2x2.py           Tables 9, 20
src/analysis_weighting.py     Table 34 + corrected Table 5
src/run_all.py             orchestrates everything, deterministic, writes env.txt
data/annotation_review.json   gold (A1) + 8 models' aligned field predictions, dev1-12 (n=120).
                               Raw argument text removed (not needed for scoring; keeps
                               the file small and avoids any question about redistributing
                               source-corpus text).
data/judge/*.json          pruned judge scores (semantic_equivalence only --
                            no question text, no judge rationale) for all 8
                            models x 3 judges (GPT-5.5 / GPT-6 Astra / GPT-5.6 Sol)
data/SHA256SUMS            checksums for everything above
tests/test_reproduce.py    asserts the values below at printed precision
run.sh                     one-command runner
```

## One command

```bash
./run.sh
```

This creates a fresh virtual environment, installs the two pinned
dependencies (`numpy==2.2.6`, `scipy==1.15.3`), verifies every input file's
checksum, runs the full pipeline and the test suite, and writes the outputs
to `out/`. It takes about 1-2 minutes on a laptop; most of that is the
Spearman bootstrap CIs (10,000 resamples each).

## Determinism

- Every random process (the Spearman bootstrap CIs) uses a **fixed seed,
  42**, written into `out/env.txt` and reused identically every run.
- `run_all.py` sets `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL/Accelerate/
  NumExpr equivalents) *before* numpy/scipy are imported, so floating-point
  results don't drift across machines with different core counts.
- Outputs are written with `repr()`-precision floats (not rounded) so a
  byte-for-byte diff is meaningful.

## Output file → paper table/figure map

| Output file | Paper item |
|---|---|
| `out/per_argument_scores.csv` | underlying per-argument `A` (and all 6 variants) — not itself a table, everything below is derived from it |
| `out/table6_21_figure2_under_A.csv` | Tables 6 & 21 (trace–question Pearson r AND Spearman ρ, each with its own CI, Holm, BH), Figure 2 |
| `out/table35_under_A_and_Aeq.csv`, `out/table35_under_Aold.csv` | Table 35 (equal-phase weighting robustness; `A_old` is the previously-published score, shown only for comparison) |
| `out/table43_judge_robustness.csv` | Table 43 (3-judge robustness of the trace–question significance pattern) |
| `out/table9_2x2_3model.csv` | Table 9 (3-model 2×2 trace-grounding × alignment diagnostic) |
| `out/table20_2x2_sensitivity.csv` | Table 20 (8-model 2×2, threshold-sensitivity sweep) |
| `out/table34_weighting_sensitivity.csv`, `out/table34_rank_agreement.csv` | Table 34 (A/A_eq/A_eq_noSR/A_old/A_abl/A_eq_abl means, ranks, rank agreement) |
| `out/table5_ablated_score.csv` | Table 5 — identical to Table 34's `A_abl` column (13/36, 2/36, 21/36 over the 36 fields remaining after excluding span1/2/3 and biasDetection/heuristicDetection) |

## Known scope limits

- **S2 (A2-gold) and S3 (dev3-12 A1-gold subset) rank-agreement settings
  are not included.** The paper's published numbers are all S1 (dev1-12,
  A1 gold, n=120), which this package reproduces in full. S2/S3 would
  require additional raw model-output data not otherwise needed for any
  number in scope here.
- **Table 2 (IAA) and Table 3 (per-field baseline table) are not included.**
  They use different metrics entirely (Cohen's κ/PABAK/Gwet-AC1 for IAA;
  support-weighted F1 + BERTScore F1, which needs a local roberta-large
  model, for Table 3) — out of scope for a no-GPU, Combined-Trace-Score-only
  package.
- **LSAT-derived numbers are not included.** LSAT item text is copyrighted
  (LSAC) and is not redistributed in this package; nothing in the scope
  list above (Tables 5/6/9/20/21/34/35/43, Figure 2) uses LSAT data, so
  this has no effect on anything this package reproduces.
