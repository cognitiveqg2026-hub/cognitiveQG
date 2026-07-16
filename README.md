# CognitiveQG

Dataset, annotation guidelines, and evaluation code for the paper:

> **CognitiveQG: A Diagnostic Benchmark for Reasoning-Trace Evaluation in Socratic Question Generation**
> Anonymous Author, Anonymous Author, Anonymous Author, Anonymous Author, Anonymous Author
> *IEEE Access*, 2026

CognitiveQG evaluates whether language models produce structured reasoning traces that
agree with expert annotations under Facione's critical-thinking framework, and whether
those traces are reflected in the Socratic questions the models generate.

**Benchmark at a glance:** 120 argumentative texts (FOCUS development split, dev1–dev12),
two expert annotators (median Gwet's AC₁ = 0.88 on categorical fields), 45 annotation
fields across seven phases (Facione's six critical-thinking phases + Socratic question
generation), 10,800 expert annotation decisions in total.

---

## Repository structure

```
.
├── data/
│   ├── a1_dev1-12_gold.json         # 120-argument expert gold annotations (A1)
│   ├── a2_dev1-12_gold.json         # second annotator (A2), 120 args, A1-aligned -> IAA
│   ├── guidelines/                  # annotation guidelines (PDF)
│   │   ├── CognitiveQG-Annotaton_Guideline_final.pdf   # full seven-phase guideline
│   │   └── Error_Analysis_guideline.pdf                # trace-question consistency (Case 3 & 4)
│   ├── annotation_tool/             # self-contained HTML annotation interface
│   ├── README.md                    # data format + FOCUS source pointer
│   └── LICENSE                      # CC BY 4.0 (annotations)
├── code/                            # evaluation scripts (see table below)
├── iaa_core/                        # shared metric library (Gwet's AC1, Jaccard, LLM judge)
├── prompts/                         # verbatim prompt templates (P1 baseline, GPT-5.5 judge rubric)
├── requirements.txt
├── CITATION.bib
├── LICENSE                          # MIT (code + annotation tool)
└── MANIFEST.md
```

---

## Annotation scheme

Each argument is annotated across seven phases following Facione's critical-thinking
framework: Interpretation, Analysis, Inference, Evaluation, Explanation, Self-Regulation,
and Socratic Question Generation (up to three priority-ranked questions per argument).
The 45 fields span categorical labels (e.g., `reasoningType`, `logicalFallacy`),
multi-label fields (e.g., `credibilityFactors`, `biasDetection`), and span/free-text
fields (e.g., `coreClaim`, `premise`, `span1–3`). Field definitions, decision trees, and
worked examples are provided in `data/guidelines/`.

The two annotator files share the same `{"annotations": [...]}` structure and
`argumentIndex` (0–119) convention, so they are directly joinable for inter-annotator
agreement analysis.

---

## Evaluation code

The `code/` directory contains the evaluation scripts used in the paper, with the shared
metric routines in `iaa_core/`:

| Script | Role in the paper |
|---|---|
| `compute_iaa_generic_rescaled.py` | inter-annotator agreement (Table 2) |
| `compute_field_metrics_jac_gwet.py`, `compute_field_metrics_latex_rescaled.py` | per-field model–human alignment (RQ1a) |
| `compute_run2_trace_scores_complete.py` | Combined Reasoning Trace Score |
| `compute_trace_question_consistency.py`, `generate_2x2_annotation.py` | trace–question consistency + 2×2 diagnostic (RQ1b) |
| `build_socratic_format_causation.py`, `evaluate_llm_judge.py` | trace-substitution formatting + GPT-5.5 judge scoring (RQ2) |

Inter-annotator agreement is directly reproducible from the released data:

```bash
pip install -r requirements.txt
python3 code/compute_iaa_generic_rescaled.py \
    --a1 data/a1_dev1-12_gold.json --a2 data/a2_dev1-12_gold.json \
    --out iaa.json
```

The remaining scripts operate on model-generated outputs (reasoning traces and Socratic
questions), which are not distributed in this repository; they are provided to document
the exact metric implementations used in the paper. The GPT-5.5 judge scripts additionally
require `OPENAI_API_KEY`.

---

## Data & license

- **Annotations** (`data/`) — **CC BY 4.0** (see `data/LICENSE`).
- **Code and annotation tool** — **MIT** (see `LICENSE`).

The source **argument texts are not redistributed** here: the `argument` field is
intentionally blank throughout `data/`. The arguments come from the **FOCUS**
development split — obtain them from the FOCUS release and re-join via the
`argumentIndex` field (see `data/README.md`). FOCUS arguments remain under FOCUS's
own terms.

## Citation

See `CITATION.bib`.
