# Release manifest

Dataset-and-guidelines release for the 120-instance CognitiveQG study. This repository
contains the expert annotations and the annotation guidelines; model outputs and analysis
code are not distributed here.

## Contents

| Path | Contents |
|---|---|
| `data/a1_dev1-12_gold.json` | 120-argument expert gold annotations (annotator A1) |
| `data/a2_dev1-12_gold.json` | second annotator (A2), 120 args, A1-aligned for IAA |
| `data/guidelines/` | annotation guidelines: full seven-phase guideline (PDF) + trace–question consistency guideline (PDF) |
| `data/annotation_tool/` | self-contained HTML annotation interface |
| `LICENSE`, `data/LICENSE`, `CITATION.bib`, `README.md` | MIT (annotation tool) + CC BY 4.0 (annotations) + docs |

## Anonymization

Author names, institutional affiliations, and personal paths have been removed.
Annotators are pseudonymized (A1 / A2). The FOCUS source `argument` text is withheld
throughout `data/` — the `argument` field is blank, while annotations and
`argumentIndex` are retained; rejoin from the FOCUS release via `argumentIndex`
(see `data/README.md`).
