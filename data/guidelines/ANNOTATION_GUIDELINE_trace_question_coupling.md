# CognitiveQG Annotation Guideline — Trace–Question Consistency

## Background

Cognition refers to the mental processes people use to understand information — perceiving,
remembering, interpreting, and reasoning. Facione's critical thinking framework organizes these
into six skills: interpretation, analysis, inference, evaluation, explanation, and self-regulation.
In the **first** CognitiveQG task, annotators reconstruct this reasoning across the six phases and
write Socratic questions from it. **This guideline covers the second task:** judging whether a
*model's* generated Socratic question genuinely **follows from the model's own reasoning trace**.

A fluent question can hide a broken reasoning chain, and a correct question can be produced from a
wrong reasoning path. Scoring the final question alone cannot tell these apart, so we evaluate the
trace and the question **together** — as one `(argument, trace, question)` triple.

- **Trace as control variable:** If the six-phase trace truly drives question generation, the
  question should be recoverable from the trace. We test whether the trace actually functioned as a
  reasoning path or merely as decoration.
- **Consistency, not correctness:** Whether the question is *correct* (probes the intended
  weakness) is decided by the LLM judge. The annotator decides whether the question is *consistent*
  with the model's own trace. A question can be correct **and** unsupported at the same time.

## Research Question

> "Given a model's stated reasoning trace, should its Socratic question naturally follow — and
> where does the reasoning path break when it does not?"

> "When a model produces a correct question, is that question actually justified by its reasoning
> trace, or did it appear *despite* the trace?"

## Objective

Create an annotation scheme and task setting that best: 1) captures whether the model's reasoning
and its question are internally consistent, 2) allows easy annotation and faithful evaluation by
reading the model's own fields, and 3) provides a structured, field-level diagnosis of *where* and
*why* a reasoning path breaks.

## Based Dataset

CognitiveQG model outputs (derived from **FOCUS2025**). Nine models — LLaMA-2, LLaMA-3, Qwen2.5,
Qwen3, Mistral, OLMo, DeepSeek-Chat, GPT-4o, GPT-5.5 — over the 60-argument set; **466 instances**,
of which **443** are pre-binned into the four coupling cases by the manuscript's threshold scores.

## CognitiveQG Dataset Component

Each instance contains:

- **Argument** — the source text.
- **Model reasoning trace** — the model's own Facione 6-phase fields (Interpretation, Analysis,
  Inference, Evaluation, Explanation, Self-Regulation).
- **Model Socratic question(s)** — up to 3, generated after the trace.
- **Human reference / LLM-judge correctness signal** — establishes the intended weakness and
  whether the question is correct.
- **Pre-assigned coupling case** — from the threshold scores `trace_jac` (≈0.24) × judge semantic
  equivalence (≈0.27), together with those two scores.

## Annotation Scheme

- **Consistency label** (per question): **FOLLOWS** or **CONTRADICTS** — judged against the model's
  own **primary trace field** (see §3).
- **Primary trace field priority:** `missingComponent → logicalFallacy → hasAssumption → premise →
  coreClaim`, then any other reasoning field the question targets (see §6).
- **Coupling cases:** Case 1–4 (pre-assigned; the annotator analyzes, not assigns).
- **Annotation Motivation:** *Critical Thinking: What It Is and Why It Counts* (Facione), and the
  CognitiveQG manuscript's trace–question coupling analysis.

## Annotation Resource

1. Open the annotation tool (`cognitiveqg_annotator.html`) in a browser (no install required).
2. Click **"Load JSON"** and upload one file from `annotation_sets/` (one per model) or from
   `annotation_sets/by_case/caseN.json` (instances grouped by pre-assigned case).
3. Step through instances with **Prev / Next**, complete the scheme, then click **"Export current"**
   (and **"Export all"** at the end). Submit the exported JSON.

---

## 1. The 2×2 Coupling Cases

> **The case is pre-assigned, not computed by the annotator.** Each instance already carries its
> case from the manuscript's threshold scores — the trace axis from `trace_jac` (≈0.24) and the
> question axis from the judge's semantic-equivalence score (≈0.27). You **pick instances from each
> case and analyze them**; your consistency label (FOLLOWS / CONTRADICTS, §3) explains the case and
> may flag where it disagrees with the threshold bin (`trace_jac` is only a proxy for "trace
> supports the question").

The 2×2 crosses the **trace axis** (does the trace support the question) with the **question axis**
(judge correctness):

|                              | **Question Wrong** | **Question Correct** |
|------------------------------|--------------------|----------------------|
| **Trace CONTRADICTS question** | **Case 1 — Wrong Question, Unsupported Path** | **Case 4 — Correct Question, Unsupported Path** ← key finding |
| **Trace FOLLOWS question**     | **Case 3 — Consistent but Wrong** | **Case 2 — Consistent and Correct** |

**Threshold for assigning Cases 1–4.** Each instance is sorted into one of the four cases by crossing
two continuous scores against their dataset-wide **mean thresholds**, computed over all 443 valid
`(model, argument)` pairs: the **reasoning-trace score** `trace_jac` (mean threshold ≈ **0.24**),
which measures how closely the model's trace matches the human annotation and serves as a *proxy*
for whether the trace supports the question; and the **judge's semantic-equivalence score**
(normalized to [0, 1], mean threshold ≈ **0.27**), which measures whether the question is *correct*
relative to the human reference. An instance is "above" an axis when its score is **at or above**
that axis's threshold. The four cases follow directly: **Case 1** = trace *below* **and** question
*below*; **Case 2** = trace *at/above* **and** question *at/above*; **Case 3** = trace *at/above*
but question *below*; **Case 4** = trace *below* but question *at/above*. Because `trace_jac` is only
a proxy for genuine trace–question support, the annotator's FOLLOWS / CONTRADICTS judgment is what
ultimately confirms or overturns each threshold-assigned case.

- **Case 1 —** the trace does not support the question and the judge marks it wrong; the misreading
  propagated into a bad question.
- **Case 2 —** the trace supports the question and the judge marks it correct; the pipeline worked
  as intended.
- **Case 3 —** the trace supports the question but the judge marks it wrong; a usable trace produced
  an off-target question.
- **Case 4 (key finding) —** the question is correct, but one or more required reasoning fields fail
  to support it (or contradict it); the right question appeared **despite** the trace.

### Annotation Scheme and Task with Examples (3 per case)

Drawn from the CognitiveQG model runs; Case 4 includes the manuscript's named Arg 15 / Arg 18. Each
example follows the same structure: **trace fields → Question → reading → Primary field →
Trace–question check → Label**.

#### Case 1 — Wrong Question, Unsupported Path *(CONTRADICTS + judge Wrong)*

**1. [DeepSeek] weapons-design** — both `coreClaim` and `missingComponent` are empty, and
`logicalFallacy = "no-fallacy"`.
- **Question:** "What might be a weakness in the argument's claim?"
- **Reading:** the question is generic, does not follow any trace field, and misses the speaker's
  moral-qualms target.
- **Primary field:** Analysis/`coreClaim` (empty).
- **Trace–question check:** **CONTRADICTS** — no trace field contains the moral-qualms weakness the
  argument raises; the reasoning path breaks because the primary field is empty.
- **Label:** CONTRADICTS · Wrong (Case 1).

**2. [LLaMA-2] gender-as-construct** — primary field `coreClaim = "Gender is a social construct."`
- **Question:** "How does the argument address the potential for individual variation in gender
  identity?"
- **Reading:** the question probes individual variation in gender identity, which is not reflected
  in the trace and misses the intended dysphoria-as-medical-condition target.
- **Primary field:** Analysis/`coreClaim = "Gender is a social construct."`
- **Trace–question check:** **CONTRADICTS** — the question targets a different element (individual
  variation) than the trace named; the path breaks at Analysis (the coreClaim states the claim but
  not the dysphoria weakness).
- **Label:** CONTRADICTS · Wrong (Case 1).

**3. [Qwen2.5] weapons-design** — primary field `missingComponent = "[leave blank]"`, effectively a
filler field.
- **Question:** "Can you provide examples of how your past desire for working on fighter jets aligns
  with your current moral qualms?"
- **Reading:** the question asks how a past desire for fighter jets aligns with current moral qualms,
  but this relationship is not captured in any trace field.
- **Primary field:** Analysis/`missingComponent = "[leave blank]"` (filler).
- **Trace–question check:** **CONTRADICTS** — the primary field is a filler placeholder; no trace
  field captures the past-vs-present relationship the question relies on; the path breaks at
  Analysis/`missingComponent`.
- **Label:** CONTRADICTS · Wrong (Case 1).

#### Case 2 — Consistent and Correct *(FOLLOWS + judge Correct)*

**1. [GPT-5.5] judiciary** — `missingComponent = "broader evidence; multiple rulings; institutional
pressure"`.
- **Question:** "Does one upheld temporary block provide enough evidence to conclude the judiciary
  is not being weakened?"
- **Reading:** the question directly addresses the missing evidence identified in the trace and
  matches the human reference question, "Do you have any other evidence?"
- **Primary field:** Analysis/`missingComponent = "broader evidence; multiple rulings; institutional
  pressure"`.
- **Trace–question check:** **FOLLOWS** — the question probes exactly the missing-evidence gap named
  in the trace; a clear path runs `missingComponent` → weakness → question.
- **Label:** FOLLOWS · Correct (Case 2).

**2. [GPT-5.5] abortion** — `missingComponent = "personhood standard, legal rights, moral status"`.
- **Question:** "What definition of 'human' or 'person' is being used, and why should that
  definition determine legality?"
- **Reading:** the question probes which definition of human or person should determine legal status,
  directly addressing the missing component identified in the trace and matching the human reference.
- **Primary field:** Analysis/`missingComponent = "personhood standard, legal rights, moral status"`.
- **Trace–question check:** **FOLLOWS** — the question follows the personhood-standard gap named in
  the trace; path runs `missingComponent` → weakness → question.
- **Label:** FOLLOWS · Correct (Case 2).

**3. [GPT-4o] abortion** — `logicalFallacy = "Hasty generalization"`.
- **Question:** "What scientific evidence supports the view that a fetus is just a bundle of cells?"
- **Reading:** the question directly probes the evidential weakness identified in the trace and
  aligns with the human reference question.
- **Primary field:** Evaluation/`logicalFallacy = "Hasty generalization"`.
- **Trace–question check:** **FOLLOWS** — the question targets the unsupported-claim weakness the
  fallacy field flags; path runs `logicalFallacy` → weakness → question.
- **Label:** FOLLOWS · Correct (Case 2).

#### Case 3 — Consistent but Wrong *(FOLLOWS + judge Wrong)*

**1. [GPT-4o] slippery-slope** — `coreClaim = "…the removal of hateful subreddits is not censorship."`
- **Question:** "What are the risks of equating content moderation with censorship?"
- **Reading:** the question directly follows the claim identified in the trace. However, the human
  reference question targeted the meaning of the slippery slope fallacy, making the generated
  question off-target.
- **Primary field:** Analysis/`coreClaim = "…the removal of hateful subreddits is not censorship."`
- **Trace–question check:** **FOLLOWS** — the question follows the `coreClaim`; the trace supports
  it, but it lands off the human's intended target (slippery-slope clarification).
- **Label:** FOLLOWS · Wrong (Case 3).

**2. [GPT-5.5] slippery-slope** — `missingComponent = "definition of censorship, moderation criteria,
limits of platform authority"`.
- **Question:** "What standard distinguishes content moderation from actual censorship?"
- **Reading:** the question directly addresses the missing component identified in the trace.
  However, it does not target the weakness identified in the human reference question and is
  therefore off-target.
- **Primary field:** Analysis/`missingComponent = "definition of censorship, moderation criteria…"`.
- **Trace–question check:** **FOLLOWS** — the question follows the missing-definition gap; the trace
  supports it, but it misses the human's slippery-slope target.
- **Label:** FOLLOWS · Wrong (Case 3).

**3. [Qwen3] child-support** — `missingComponent = "legal framework, societal standards, economic
realities"`.
- **Question:** "What criteria determine who should bear the costs of raising a child?"
- **Reading:** the question follows the gap identified in the trace and addresses the missing
  supporting information. However, the human reference question focused on the meaning of "cutting
  the losses," making the generated question off-target.
- **Primary field:** Analysis/`missingComponent = "legal framework; societal standards; economic
  realities"`.
- **Trace–question check:** **FOLLOWS** — the question follows the missing-criteria gap; it is
  trace-supported but off the human's "cutting the losses" target.
- **Label:** FOLLOWS · Wrong (Case 3).

#### Case 4 — Correct Question, Unsupported Path *(CONTRADICTS + judge Correct)* ← key finding

**1. [LLaMA-2] Arg 15 ("wet") (manuscript)** — `coreClaim = "nothing"`.
- **Question:** "Can you provide more justification for why water cannot be soaked in itself?"
- **Reading:** the question probes the contested definition of "wet," which is the intended human
  target. However, the trace does not identify any claim, ambiguity, or weakness related to the
  concept of wetness.
- **Primary field:** Analysis/`coreClaim = "nothing"` (empty).
- **Trace–question check:** **CONTRADICTS** — the question targets the definition of "wet," but no
  trace field records this ambiguity; the reasoning path breaks at Analysis/`coreClaim`.
- **Label:** CONTRADICTS · Correct (Case 4).

**2. [LLaMA-2] Arg 18 ("voting") (manuscript)** — `coreClaim = "Voting"`, `missingComponent = "none"`.
- **Question:** "What do you mean by 'voting'?"
- **Reading:** the question matches the human reference and correctly targets the need for
  clarification. However, the one-word trace provides insufficient reasoning support and cannot
  justify the specific clarification requested.
- **Primary field:** Analysis/`coreClaim = "Voting"` (one-word filler; `missingComponent = "none"`).
- **Trace–question check:** **CONTRADICTS** — the primary field is a single token and no field
  articulates the ambiguity; the reasoning path breaks at Analysis (nothing specifies *what* about
  voting to ask).
- **Label:** CONTRADICTS · Correct (Case 4).

**3. [Qwen2.5] gender-as-construct** — `missingComponent = ""` (filler), so the primary field falls
back to `coreClaim = "Gender is a social construct."`
- **Question:** "How does the presence of dysphoria challenge the assertion that gender is solely a
  social construct?"
- **Reading:** the question correctly targets the human reference by introducing the dysphoria
  counterargument. However, no trace field records dysphoria, medical considerations, or any related
  counterpoint, so the question relies on information absent from the trace.
- **Primary field:** Analysis/`coreClaim = "Gender is a social construct."` (after `missingComponent
  = ""` filler).
- **Trace–question check:** **CONTRADICTS** — the question introduces the dysphoria counterpoint,
  which no trace field captures; the reasoning path breaks because the trace records nothing about
  dysphoria / medical condition.
- **Label:** CONTRADICTS · Correct (Case 4).

---

## 2. Case 4 Deep-Dive: Reasoning-Path Inconsistency

**Definition:** "Correct question, unsupported reasoning path." The judge marks the question
correct, but the trace contains none of the claim, gap, fallacy, ambiguity, assumption, or missing
evidence needed to justify it. Inspect the trace **field by field** and complete the template.

**Annotation Scheme and Task with Example:**

```
Case 4: Good Question, Bad Reasoning Path
- Good question target:
- Field(s) that support the question:
- Field(s) that contradict the question:
- Field(s) that are missing / filler:
- Main break point in the reasoning path:
- Why the final question cannot be fully derived from the trace:
- Final judgment:   Fully supported | Partially supported | Unsupported / contradicted
```

*Eg (Arg 15 — "definition of wet"). Question: "Can you provide more justification for why water
cannot be soaked in itself?"*

```
Case 4: Good Question, Bad Reasoning Path
- Good question target:               the definitional claim about "wet" / water.
- Field(s) that support the question: none.
- Field(s) that contradict the question: Evaluation/logicalFallacy="false cause" (unrelated).
- Field(s) that are missing / filler: Analysis/coreClaim="nothing", minorClaim="nothing",
                                      premise="nothing", missingComponent="none".
- Main break point in the reasoning path: Analysis — no claim is named, so there is nothing to ask.
- Why the final question cannot be fully derived from the trace: the question's content is lifted
                                      from the argument's first sentence; no trace field contains it.
- Final judgment:   Unsupported / contradicted
```

---

## 3. Trace–Question Consistency (FOLLOWS / CONTRADICTS)

**Definition:** A question is consistent with the trace when it probes the *same* claim, gap,
fallacy, ambiguity, assumption, or missing evidence the model recorded in its own primary trace
field — so that a reader given only that field could predict what the question is about.

**Guiding questions:**

1. Which trace field is the question trying to probe?
2. Is that field filled-in and specific, or empty/filler?
3. Can you state a clear path: **trace field → weakness → question**?
4. If not, does the question target a *different* reasoning element, or information lifted straight
   from the argument that no trace field captured?

**Labels (two only):**

- **FOLLOWS** *(trace supports the question)* — the question directly probes the model's filled-in,
  specific primary field. *Eg:* trace `missingComponent` = "no supporting example"; question =
  "What example would show that?" → **FOLLOWS**.
- **CONTRADICTS** *(trace does not support the question)* — any of: **(a)** the question probes a
  different reasoning element than the trace named; **(b)** it relies on argument-surface
  information no trace field captured; **(c)** the primary field is empty/filler. *Eg:* trace
  `coreClaim` = "nothing"; question probes the "wet" definition from the argument's first sentence →
  **CONTRADICTS**.

> The scoring code's `BYPASSES` label is folded into `CONTRADICTS` here (both mean "the trace does
> not support the question").

---

## 4. Evidence Requirements

For every label, cite all four sources:

1. the **argument span** (quote it),
2. the **model trace field** (named by phase, e.g. Analysis/`coreClaim`),
3. the **generated question** (quote it),
4. the **human reference or judge verdict** (the intended weakness it was measured against).

A label missing any of these is incomplete and is re-reviewed.

---

## 5. Annotation Process

### Goal of This Annotation

This annotation focuses primarily on understanding **Case 4 (Correct Question, Unsupported Path)**.

Case 4 represents the key phenomenon studied in CognitiveQG: a model generates a correct Socratic
question, but its own reasoning trace does not fully support that question.

The purpose of annotation is therefore not simply to determine whether a question follows the
trace. Instead, the purpose is to identify:

1. why the question is correct;
2. where the reasoning trace fails to support it;
3. which reasoning field breaks the reasoning path.

### Step 1. Verify the Pre-assigned Case

Each instance already contains a pre-assigned coupling case.

First read:

- Argument
- Model reasoning trace
- Generated question
- Human reference / judge signal

Then verify whether the instance appears to belong to its assigned case. The primary focus of this
annotation is Case 4.

### Step 2. Analyze Trace–Question Consistency

Ask:

> If I only read the model's reasoning trace, could I reasonably predict this question?

There are two outcomes:

**FOLLOWS**

- The trace contains the weakness targeted by the question.
- A clear reasoning path exists from trace to question.

**CONTRADICTS**

- The trace does not contain the weakness targeted by the question.
- The question relies on information absent from the trace.
- The trace contains unrelated or contradictory reasoning.
- The primary field is empty or filler.

### Step 3. Identify the Broken Field

If the question is CONTRADICTS, determine where the reasoning path fails. Inspect all relevant
fields and identify:

- supporting fields;
- contradicting fields;
- missing or filler fields.

Ask:

> Which field should have contained the information needed to generate this question?

This field becomes the primary diagnostic target.

---

## 6. Primary Trace Field Analysis

The **Primary Trace Field** is the field that should have provided the reasoning support for the
generated question.

**Recommended priority:**

> `missingComponent` → `logicalFallacy` → `hasAssumption` → `premise` → `coreClaim` → other relevant
> field

For the selected field, determine:

1. Is the field **present**?
2. Is the field **specific**?
3. Does the field **contain the weakness** targeted by the question?
4. Can the question be **derived from this field alone**?

If the answer to any of these is **"No"**, record the field as a potential reasoning **break point**.
The primary field should explain *why* the question cannot be fully derived from the trace.

---

## 7. Ambiguity & Confidence

- **Empty/filler primary field → CONTRADICTS.**
- **If unsure** whether the question follows → choose **CONTRADICTS** and mark **low confidence**.
- **Never upgrade to FOLLOWS** just because the judge marked the question correct.
- **Correctness and consistency are separate** — a correct question can be CONTRADICTS (Case 4).
- Assign confidence **High / Medium / Low**; route Low items to adjudication.

---

## 8. Annotator Checklist

- ☐ Picked an instance from the given (pre-assigned) case; read the argument, human reference, and
  judge verdict.
- ☐ Identified the trace field the question targets (any phase; default priority); flagged
  empty/filler fields.
- ☐ Labeled each question **FOLLOWS / CONTRADICTS** (majority over the 3).
- ☐ Confirmed the pre-assigned case; flagged any disagreement with it.
- ☐ If Case 4: completed the §2 field-by-field template; identified the broken field (§6).
- ☐ Cited all four evidence sources; set confidence (Low → adjudication); exported the JSON.

---

## 9. (Optional) Human–LLM Field Divergence Analysis

**When to run.** Only when the consistency label is **CONTRADICTS** (§3). This is a secondary
error-analysis layer on top of the 2×2 coupling analysis (§1) and is most informative for **Case 4**.

**Purpose.** A CONTRADICTS label says the model's question does not follow its *own* trace; this
analysis explains *why, relative to the human reasoning* — i.e. which human reasoning component the
model **missed, replaced, or weakened**. It compares the field the **human** used to support the
reference question against the **model's** primary trace field, and assigns a single **divergence
type**. Where §5–§6 locate the break point *inside the model trace*, this section locates the gap
*between the human and the model*.

### Step 1 — Identify the human target field

From the human reference (§4), locate the human annotation field that supports the human reference
question, and the weakness it encodes.

- **Human Target Field:** `<field — e.g. missingComponent>`
- **Human Target Weakness:** `<weakness — e.g. "the definition of 'wet'">`

### Step 2 — Identify the model target field

Locate the model's primary trace field (§6) and the weakness (if any) it encodes.

- **Model Target Field:** `<field — e.g. coreClaim>`
- **Model Weakness:** `<weakness, or "none / empty">`

### Step 3 — Compare human vs. model and assign a divergence type

Decide how the model trace diverges from the human reasoning, and assign **exactly one** type:

| Divergence type | Definition | Example (human → model) |
|---|---|---|
| **Missing Field** | The human target field exists, but the model's corresponding field is empty or absent. | `missingComponent = "definition of wet"` → `coreClaim = "nothing"` |
| **Filler Field** | The model field contains only placeholders or non-informative content. | — → `"none"` / `"nothing"` / `""` / `"[leave blank]"` |
| **Wrong Weakness** | The model identifies a *different* weakness than the human target. | dysphoria counterargument → individual variation |
| **Partial Match** | The model captures part of the human weakness but omits a critical element. | legal rights + moral status → legal rights only |
| **Alternative Weakness** | The model identifies a different but *potentially reasonable* weakness. | slippery slope → definition of censorship |
| **Argument Lift** | The question appears taken directly from the argument text, matching no trace field. | `definition of wet` → no trace field contains "wet"; Q = "What do you mean by wet?" |

### Output template

```
Human Target Field:    <field>
Human Target Weakness: <weakness>
Model Target Field:    <field>
Model Weakness:        <weakness / none>
Divergence Type:       Missing Field | Filler Field | Wrong Weakness |
                       Partial Match | Alternative Weakness | Argument Lift
Explanation:           <one sentence: what the model missed, replaced, or bypassed>
```

### Aggregate analysis (after annotation)

Computed over all **CONTRADICTS** instances:

1. **Divergence-type frequency** — count of each of the six types across all models.
2. **Divergence type × model** — the dominant type(s) for each model. *(Illustrative shape, not
   measured: GPT-5.5 → Alternative Weakness; GPT-4o → Argument Lift; Qwen3 → Filler Field; LLaMA-2 →
   Missing Field; replace with the counts from the annotations.)*
3. **Divergence type × case** — e.g. Case 1 dominated by **Wrong Weakness**; Case 4 dominated by
   **Missing Field + Argument Lift**.

### Research question supported

> When the model trace **CONTRADICTS** the generated question, *which human reasoning component was
> missed, replaced, or bypassed?*

This adds a human-referenced error layer beneath the manuscript's 2×2 coupling analysis,
complementing the model-internal break-point diagnosis of §5–§6.
