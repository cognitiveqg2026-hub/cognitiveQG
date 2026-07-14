# Data

## `a1_dev1-12_gold.json`

Expert gold annotations for the **120 CognitiveQG arguments** (dev1–dev12), produced by
expert annotator *A1*. Each entry contains the argument text and the full seven-phase
trace (45 fields: interpretation, analysis, inference, evaluation, explanation,
self-regulation, and up to three grounded Socratic questions).

Inter-annotator agreement (with the second annotator, *A2*) is reported in
`../tables/cognitiveqg-iaa-dev1-12_RESCALED.tex` — median Gwet's AC₁ = 0.88 on categorical fields.

## `a2_dev1-12_gold.json`

Second annotator (*A2*) annotations for all **120 arguments**, in the same
`{"annotations": [...]}` shape and `argumentIndex` (0–119) convention as
`a1_dev1-12_gold.json` — the two files are directly joinable by `argumentIndex` for
IAA reproduction. Built by concatenating A2's per-batch source files in the pipeline's
fixed unit order (dev1–6 combined, then dev7–8–…–12 in sequence) and renumbering
positionally; the per-batch source files are available from the authors on request.
Together with `a1_dev1-12_gold.json`, this covers all 120 arguments for both annotators.

## Source arguments (FOCUS) — not redistributed here

CognitiveQG annotates argumentative texts from the **FOCUS** benchmark. **The source
argument text is not included in this release**: the `argument` field in every data file
is intentionally left blank. Re-join the arguments to these annotations via the
`argumentIndex` field.

- Paper: <https://aclanthology.org/2025.ijcnlp-long.157/> (IJCNLP-AACL 2025)
- The source arguments are governed by the FOCUS release terms; obtain them from the FOCUS
  repository. This release adds only the CognitiveQG reasoning-trace annotations on top.

## Guidelines

- `guidelines/Error_Analysis_guideline.pdf` — trace–question consistency (Case 3 & 4)
  labeling rules: consistent vs. inconsistent, with worked examples.
- `guidelines/CognitiveQG-Annotaton_Guideline_final.pdf` — full seven-phase annotation guideline.

## Annotation tools

Self-contained HTML interface the annotators used (open in any browser — no server needed):

- `annotation_tool/cognitiveqg_annotator.html` — the main seven-phase trace + Socratic-question annotation interface.

## License

The CognitiveQG annotations and guidelines are released under **CC BY 4.0** — see
`data/LICENSE`. (The repository code under `code/` is MIT-licensed; see the top-level `LICENSE`.)
