# Release manifest

Dataset, annotation guidelines, and evaluation code for the 120-instance CognitiveQG
study. Model-generated outputs (reasoning traces, Socratic questions, judge scores) are
not distributed in this repository.

## Contents

| Path | Contents |
|---|---|
| `data/a1_dev1-12_gold.json` | 120-argument expert gold annotations (annotator A1) |
| `data/a2_dev1-12_gold.json` | second annotator (A2), 120 args, A1-aligned for IAA |
| `data/guidelines/` | annotation guidelines: full seven-phase guideline (PDF) + trace–question consistency guideline (PDF) |
| `data/annotation_tool/` | self-contained HTML annotation interface |
| `code/` | eight evaluation scripts (see README table) |
| `iaa_core/` | shared metric library imported by `code/` (Gwet's AC₁, Jaccard, LLM judge) |
| `requirements.txt`, `LICENSE`, `data/LICENSE`, `CITATION.bib`, `README.md` | MIT (code + annotation tool) + CC BY 4.0 (annotations) + docs |

## Reproduction status

Inter-annotator agreement is directly reproducible from the released data
(`code/compute_iaa_generic_rescaled.py`; see README). The remaining scripts document the
exact metric implementations used in the paper and operate on model-generated outputs
that are not distributed here. The GPT-5.5 judge stages require `OPENAI_API_KEY`.

## Anonymization

Author names, institutional affiliations, and personal paths have been removed.
Annotators are pseudonymized (A1 / A2). The FOCUS source `argument` text is withheld
throughout `data/` — the `argument` field is blank, while annotations and
`argumentIndex` are retained; rejoin from the FOCUS release via `argumentIndex`
(see `data/README.md`).
