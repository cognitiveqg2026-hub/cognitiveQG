# CognitiveQG

Code and data for the paper:

> **CognitiveQG: A Diagnostic Benchmark for Reasoning-Trace Evaluation in Socratic Question Generation**
> Anonymous Author, Anonymous Author, Anonymous Author, Anonymous Author, Anonymous Author
> *IEEE Access*, 2026

CognitiveQG evaluates whether language models produce structured reasoning traces that
agree with expert annotations under Facione's critical-thinking framework, and whether
those traces are reflected in the Socratic questions the models generate. It combines
three complementary constructs:

1. **Cognitive alignment (RQ1a)** — phase-level agreement between model and expert reasoning traces.
2. **Trace–question consistency (RQ1b)** — whether a generated question targets the same weakness the model's own trace identified (**consistent** vs. **inconsistent**).
3. **Functional trace influence (RQ2)** — whether the *content* of the trace causally shifts question quality (oracle vs. noise substitution).

**Benchmark at a glance:** 120 argumentative texts, two expert annotators (median Gwet's
AC₁ = 0.88 on categorical fields), 45 annotation fields across seven phases, eight models
evaluated (LLaMA-2-7B, LLaMA-3.1-8B, Qwen2.5-7B, Mistral-7B, OLMo-2-7B, Qwen3-8B, GPT-4o, GPT-5.5).

---

## Repository structure

```
.
├── data/                           # expert annotations (argument text withheld; see below)
│   ├── a1_dev1-12_gold.json         # 120-argument expert gold annotations (A1)
│   ├── a2_dev1-12_combined.json     # second annotator (A2), 120 args, A1-aligned -> IAA
│   ├── guidelines/                  # annotation guidelines (Markdown + PDF)
│   ├── annotation_tool/             # self-contained HTML annotation interface
│   └── README.md                    # data access + FOCUS source pointer
├── results/                        # model outputs + scores (see results/README.md)
│   ├── baseline_dev1-12/            # RQ1: annotation review + trace-score CSV (8 models x 120)
│   ├── causation/                   # RQ2: oracle / noise / vanilla generations
│   └── causation_judge/             # RQ2: GPT-5.5 judge scores
├── code/                           # analysis scripts that reproduce the reported results
│   ├── compute_run2_trace_scores_complete.py    # Combined Trace Score (RQ1)
│   ├── compute_field_metrics_jac_gwet.py        # per-field agreement, Gwet AC1 / Jaccard (RQ1a)
│   ├── compute_field_metrics_latex_rescaled.py  # rescaled-BERTScore field table (RQ1a)
│   ├── compute_iaa_generic_rescaled.py          # inter-annotator agreement
│   ├── compute_trace_question_consistency.py    # RQ1b consistent / inconsistent
│   ├── generate_2x2_annotation.py               # 2x2 diagnostic assignment (RQ1b)
│   ├── build_socratic_format_causation.py       # RQ2 oracle/noise/vanilla formatting
│   └── evaluate_llm_judge.py                     # RQ2 GPT-5.5 judge scoring
├── iaa_core/                       # shared metric library imported by code/ (Gwet AC1, Jaccard, judge)
├── requirements.txt
├── CITATION.bib
├── LICENSE
└── MANIFEST.md
```

> **Scope.** This release contains the analysis code that reproduces the paper's reported
> results from the released annotations and model outputs. Each model is evaluated at its base
> instruction-tuned checkpoint (Table 8) under three prompting conditions (vanilla, oracle,
> noise); no fine-tuning is involved in any reported result. Model-generation and fine-tuning
> code are not included here: the raw generation stage additionally requires the FOCUS source
> texts (not redistributed; see below) and GPU inference.

---

## Metrics (current)

| Field type | Primary metric | Also reported |
|---|---|---|
| Categorical (e.g. `logicalFallacy`, `reasoningType`) | **Gwet's AC₁**, Weighted F1 | Cohen's κ, PABAK, OA |
| Multi-label (e.g. `credibilityFactors`, `biasDetection`) | Weighted F1, Gwet's AC₁ | Jaccard, exact match |
| Span / text (e.g. `span1–3`, `coreClaim`) | **Rescaled BERTScore F1** (roberta-large) | Jaccard, ROUGE-L |
| Question quality (downstream) | **GPT-5.5 Semantic Equivalence (GPT-Sem)** | Hungarian-corrected BERT |

The **Combined Trace Score** aggregates categorical (exact-match), multi-label (set-Jaccard),
and span/text (token-Jaccard) fields; the **ablated** variant drops `span1–3`,
`biasDetection`, and `heuristicDetection` (majority-class / format artifacts).

---

## Which script answers which research question?

The eight scripts in `code/` map to the research questions as follows.

| RQ / step | Script(s) in `code/` | Output |
|---|---|---|
| **RQ1a** — model-vs-expert trace agreement | `compute_field_metrics_latex_rescaled.py`, `compute_field_metrics_jac_gwet.py` | per-field agreement (Gwet AC₁ / Wt.F1 / Rescaled BERT) |
| **IAA** — inter-annotator agreement | `compute_iaa_generic_rescaled.py` | Gwet AC₁ / κ / Jaccard per field |
| **Trace score** — Combined Trace Score | `compute_run2_trace_scores_complete.py` | per-(model, arg) trace-score CSV |
| **RQ1b** — trace–question consistency + 2×2 | `compute_trace_question_consistency.py`, `generate_2x2_annotation.py` | per-question **consistent / inconsistent** + 2×2 assignment |
| **RQ2** — functional trace influence | `build_socratic_format_causation.py`, `evaluate_llm_judge.py` | oracle / noise / vanilla generations + GPT-5.5 judge |

> ⚠️ **Some scripts in `code/` still use older conventions** (`FOLLOWS / BYPASSES / CONTRADICTS`
> labels, a GPT-4o judge, or Micro-F1). The final paper uses **consistent / inconsistent**, the
> **GPT-5.5** judge, and **Weighted-F1 / Gwet's AC₁ / Rescaled BERTScore** — verify the vocabulary
> and metric flags before publishing. See `MANIFEST.md`.

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=<your-key>     # GPT-5.5 Semantic-Equivalence judge (RQ1b, RQ2)
export HF_TOKEN=<your-token>         # gated open-weight checkpoints (e.g. LLaMA-2)
```

---

## Reproducing the reported results

The following scripts regenerate the paper's core quantitative results directly from the
released `data/` and `results/`, without GPU or model inference (BERTScore downloads
`roberta-large` on first use). Run them from the repository root:

```bash
# Combined Trace Score (RQ1) — per (model, argument); regenerates
# results/baseline_dev1-12/reasoning_trace_scores.csv (960 rows, 8 models x 120 args)
python3 code/compute_run2_trace_scores_complete.py \
    --models gpt-4o gpt-5.5 llama2 llama3 mistral olmo qwen qwen3

# Per-field annotator-vs-model agreement (RQ1a): Gwet's AC1 / Weighted-F1 / Jaccard,
# and the rescaled-BERTScore LaTeX table
python3 code/compute_field_metrics_jac_gwet.py
python3 code/compute_field_metrics_latex_rescaled.py

# Inter-annotator agreement (Gwet's AC1 / Cohen's kappa / rescaled BERTScore)
python3 code/compute_iaa_generic_rescaled.py \
    --a1 data/a1_dev1-12_gold.json --a2 data/a2_dev1-12_combined.json \
    --out results/baseline_dev1-12/iaa.json
```

The GPT-5.5 judge stages (RQ1b trace–question consistency; RQ2 functional trace influence)
require `OPENAI_API_KEY`. Their judged outputs are already provided under `results/`, so the
downstream numbers can be inspected and re-scored without re-querying the model.

Model generation (oracle / noise / vanilla) and figure plotting are not part of this release:
the generation stage requires the FOCUS source texts (not redistributed; see below) and GPU
inference, and the figures are produced from the complete experiment tree. This package provides
the analysis code and the released outputs needed to reproduce the reported numbers.

---

## Data & license

- **Code** (`code/`, scripts) — **MIT** (see `LICENSE`).
- **Annotations** (`data/`) — **CC BY 4.0** (see `data/LICENSE`).

The source **argument texts are not redistributed** here: the `argument` field is
intentionally blank throughout `data/` and `results/`. The arguments come from the
**FOCUS** development split — obtain them from the FOCUS release and re-join via
`argumentIndex` (see `data/README.md`). FOCUS arguments remain under FOCUS's own terms.

## Citation

See `CITATION.bib`.
