# Data sources and licenses

## Source arguments (FOCUS / SoQG-2023)

**Not redistributed here.** The 120 dev1-12 arguments and 20 LSAT arguments
are drawn from the FOCUS benchmark (Pothong et al., IJCNLP-AACL 2025), which
itself reuses arguments from SoQG-2023 (Ang et al., EACL 2023). The raw
argument text has been stripped from every annotation/gold/judge file in
`data/` (confirmed: 32,680 `"argument"` text fields removed across 217
files); only the structured field-level annotations and scores remain.
Obtain the source arguments directly from the FOCUS/SoQG-2023 releases if
needed; see their respective publications for license terms.

**LSAT items specifically**: sourced via AGIEval (Zhong et al., 2024), whose
underlying LSAT item text is LSAC-copyrighted. Not redistributed here for
the same reason, independent of the FOCUS/SoQG-2023 point above.

## Human expert annotations (A1 and A2)

`data/a1_dev1-12_gold.json`, `data/a2_dev1-12_gold.json`: 120
arguments × 41 trace fields + 3 Socratic questions each, two independent
annotators. `data/lsat/lsat_a1_gold.json`, `lsat_a2_gold.json`: same
scheme, 20 LSAT arguments. **Original to this study.** Collected under JAIST
Institutional Review Board approval (HU26-007); annotators provided
informed consent and were compensated (JPY 4,000/task). Authors' own data,
no external license restriction, but please cite the paper if reused.

## Model-generated reasoning traces and judge scores

`data/baseline_complete_dev1-12/`, `data/lsat/`, `data/causation/`: outputs
from 8 LLMs (LLaMA-2-7B-Chat, LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct,
Qwen3-8B, Mistral-7B-Instruct-v0.3, OLMo-2-7B-Instruct, GPT-4o, GPT-5.5) and
3 LLM judges (GPT-5.5, GPT-6 Astra, GPT-5.6 Sol). Open-weight model outputs:
original to this study, generated under each model's own license (Llama,
Apache 2.0/OpenRAIL variants; see each model card). GPT-4o/GPT-5.5
generations and all three judges' outputs: subject to OpenAI's API terms of
use; included here as derived research artifacts (scores/labels), not as a
redistribution of OpenAI's model weights.

## Human judge validation (400 pairs)

`data/human_validation/`: one domain expert's 1–5 semantic-equivalence
ratings on 400 model-question/gold-question pairs (50 per model, stratified
sample), used to validate GPT-5.5 as an automated judge (Table 4, ρ=0.794).
Original to this study, same IRB/consent basis as the expert annotations
above.

## Trace-question consistency labels (263 pairs)

Not included in this release. The labelling guideline is in
`data/guidelines/Error_Analysis_guideline.pdf`.

## Derived scores and statistics

Everything under `results/` (once `run.sh` is executed) is computed by the
scripts in `src/` and `scripts/` from the data above, with no additional license
considerations beyond the inputs already covered.

## requirements.txt pin rationale

`numpy==2.2.6` / `scipy==1.15.3` are pinned exactly (matching the original
reproduction package) because the Spearman bootstrap CIs and Holm/BH
corrections are floating-point sensitive; other packages are pinned to
minimum versions only.
