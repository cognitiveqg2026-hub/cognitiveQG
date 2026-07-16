# CognitiveQG

Dataset and annotation guidelines for the paper:

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
│   ├── guidelines/                  # annotation guidelines (Markdown + PDF)
│   │   ├── CognitiveQG-Annotaton_Guideline_final.pdf   # full seven-phase guideline
│   │   └── Error_Analysis_guideline.pdf                # trace-question consistency (Case 3 & 4)
│   ├── annotation_tool/             # self-contained HTML annotation interface
│   ├── README.md                    # data format + FOCUS source pointer
│   └── LICENSE                      # CC BY 4.0 (annotations)
├── CITATION.bib
├── LICENSE                          # MIT (annotation tool)
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

## Data & license

- **Annotations** (`data/`) — **CC BY 4.0** (see `data/LICENSE`).
- **Annotation tool** (`data/annotation_tool/`) — **MIT** (see `LICENSE`).

The source **argument texts are not redistributed** here: the `argument` field is
intentionally blank throughout `data/`. The arguments come from the **FOCUS**
development split — obtain them from the FOCUS release and re-join via the
`argumentIndex` field (see `data/README.md`). FOCUS arguments remain under FOCUS's
own terms.

## Citation

See `CITATION.bib`.
