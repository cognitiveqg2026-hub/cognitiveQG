# Data

## `a1_dev1-12_gold.json`

Expert gold annotations for the **120 CognitiveQG arguments** (dev1–dev12), produced by
expert annotator *A1*. Each entry contains the full seven-phase annotation (44 fields:
41 reasoning-trace fields across interpretation, analysis, inference, evaluation,
explanation and self-regulation, plus up to three grounded Socratic questions).

Inter-annotator agreement with the second annotator, *A2*, is reported in the paper
(Table 2).

## `a2_dev1-12_gold.json`

Second annotator (*A2*) annotations for all **120 arguments**, in the same
`{"annotations": [...]}` shape and `argumentIndex` (0–119) convention as
`a1_dev1-12_gold.json`. The two files are directly joinable by `argumentIndex` for
IAA reproduction. Built by concatenating A2's per-batch source files in the pipeline's
fixed unit order (dev1–6 combined, then dev7–8–…–12 in sequence) and renumbering
positionally; the per-batch source files are available from the authors on request.
Together with `a1_dev1-12_gold.json`, this covers all 120 arguments for both annotators.

## Other data used by the evaluation code

- `baseline_complete_dev1-12/`: model outputs (Tier 1) and the scores from the three LLM judges.
- `causation/`: trace-substitution outputs (Vanilla, Oracle, Noise) with judge scores, and the summary statistics.
- `dev1-12_a2/`: A2 annotations in the format used by the Table 34 S2/S3 script.
- `human_validation/`: the 400 human semantic-equivalence ratings used to validate the GPT-5.5 judge (Table 4).
- `lsat/`: LSAT extension (20 arguments): A1 and A2 annotations and judge scores.

See `../DATA.md` for the source and license of each.

## Source arguments (FOCUS): not redistributed here

CognitiveQG annotates argumentative texts from the **FOCUS** benchmark. **The source
argument text is not included in this release**: the `argument` field in every data file
is intentionally left blank. Re-join the arguments to these annotations via the
`argumentIndex` field.

- Paper: <https://aclanthology.org/2025.ijcnlp-long.157/> (IJCNLP-AACL 2025)
- The source arguments are governed by the FOCUS release terms; obtain them from the FOCUS
  repository. This release adds only the CognitiveQG reasoning-trace annotations on top.

## Guidelines

- `guidelines/Error_Analysis_guideline.pdf`: trace–question consistency (Case 3 & 4)
  labeling rules: consistent vs. inconsistent, with worked examples.
- `guidelines/CognitiveQG-Annotaton_Guideline_final.pdf`: full seven-phase annotation guideline.

## Annotation tools

Self-contained HTML interface the annotators used (open in any browser, no server needed):

- `annotation_tool/cognitiveqg_annotator.html`: the main seven-phase trace + Socratic-question annotation interface.

## License

The CognitiveQG annotations and guidelines are released under **CC BY 4.0**; see
`data/LICENSE`. (The HTML annotation tool is MIT-licensed; see the top-level `LICENSE`.)
