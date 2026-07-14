# Release manifest

A curated, reproduction-focused release for the 120-instance CognitiveQG study. Contents are
limited to the data, model outputs, and analysis code required to reproduce the paper's reported
results. The model-generation and fine-tuning pipeline is not included (see **Scope** in the
README); all file sizes are well under GitHub's 100 MB limit, so no git-LFS is required.

## Contents

| Path | Contents |
|---|---|
| `data/a1_dev1-12_gold.json` | 120-argument expert gold annotations (annotator A1) |
| `data/a2_dev1-12_combined.json` | second annotator (A2), 120 args, A1-aligned for IAA |
| `data/guidelines/` | annotation guidelines (Markdown + PDF) |
| `data/annotation_tool/` | self-contained HTML annotation interface |
| `results/baseline_dev1-12/` | RQ1 — annotation review + Combined Trace Score CSV (8 models × 120) |
| `results/causation/`, `causation_judge/` | RQ2 — oracle/noise/vanilla generations + GPT-5.5 judge scores |
| `code/` | eight analysis scripts (see README table) |
| `iaa_core/` | shared metric library imported by `code/` (Gwet's AC₁, Jaccard, LLM judge) |
| `requirements.txt`, `LICENSE`, `data/LICENSE`, `CITATION.bib`, `README.md` | MIT (code) + CC BY 4.0 (data) + docs |

## Reproduction status

The Combined Trace Score, per-field agreement, and inter-annotator agreement reproduce directly
from the released `data/` and `results/`; the trace-score CSV has been verified to regenerate
bit-for-bit. The GPT-5.5 judge stages (RQ1b, RQ2) require `OPENAI_API_KEY`, and their outputs are
already provided under `results/`. See "Reproducing the reported results" in the README.

## Anonymization

Author names, institutional affiliations, cluster usernames, and personal filesystem paths have
been removed from the code and documentation. Annotators are pseudonymized (A1 / A2). The FOCUS
source `argument` text is withheld throughout `data/` and `results/` — the `argument` field is
blank, while annotations and `argumentIndex` are retained; rejoin from the FOCUS release via
`argumentIndex` (see `data/README.md`).

## Author to-do before publishing

- [ ] Set the public repository URL in the manuscript footnote and `CITATION.bib`.
- [ ] Resolve `code/compute_trace_question_consistency.py`: it still emits the legacy
      `FOLLOWS / BYPASSES / CONTRADICTS` labels via a GPT-4o judge, whereas RQ1b in the paper uses
      `consistent / inconsistent` (carried by `generate_2x2_annotation.py`). Update it to the final
      convention or remove it.
- [x] Licenses — MIT (code) + CC BY 4.0 (annotations).
- [x] PII, FOCUS source text, and annotator anonymization — done.
