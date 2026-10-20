#!/usr/bin/env python3
"""Put span-role F1 rows into the dev3-12 per-annotator tables (Tables 29 and 30).

Replaces the "Span Eval. (span1/2/3)" BERTScore rows (or regenerates existing "Span role:" rows) with
one-vs-rest F1 rows for major claim, minor claim and premise, pooled over the three span slots,
3-seed mean +/- std from results/revision/span_roles_table3.json (batch dev3-12). Bold = best displayed
mean per row, ties included. Adds the span-role sentence to the caption if it is missing.
Idempotent: running it again on its own output changes nothing.

Files (in TABLE_DIR; set CQG_TABLE_DIR to use another folder):
  A1_model-baselinescore-dev3-12.tex  (A1 gold)
  A2-model-baselinescore-dev3-12.tex  (A2 gold)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import OUT, TABLE_DIR

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
ROLES = [("Major claim", "major-claim"), ("Minor claim", "minor-claim"), ("Premise", "premise")]
CAP_ADD = (" Span role rows: one-vs-rest F1 for each argumentative role, pooled over the three span slots"
           " of the reasoning structure.}")
FILES = [("A1_model-baselinescore-dev3-12.tex", "A1"), ("A2-model-baselinescore-dev3-12.tex", "A2")]


def rows(span, gold):
    out = []
    for disp, role in ROLES:
        vals = [span[f"{m}|{gold}|dev3-12"][role] for m in MODELS]
        shown = [f"{v[0]:.1f}" for v in vals]
        best = max(float(s) for s in shown)
        cells = [(r"\textbf{%s}" % s if float(s) == best else s) + r"{\tiny$\pm$%.1f}" % v[1] for s, v in zip(shown, vals)]
        out.append(f"Span role: {disp} & F1 & " + " & ".join(cells) + r" \\")
    return out


def main():
    span = json.load(open(OUT / "span_roles_table3.json"))
    for fname, gold in FILES:
        path = TABLE_DIR / fname
        lines, new, done = path.read_text(encoding="utf-8").split("\n"), [], False
        for l in lines:
            if l.startswith(r"Span Eval.\ (span") or l.startswith("Span role: "):
                if not done:
                    new += rows(span, gold); done = True
                continue
            if l.startswith(r"\caption{Per-field") and "Span role rows" not in l and l.endswith("valid predictions.}"):
                l = l[:-1] + CAP_ADD
            new.append(l)
        assert done, f"no span rows found in {fname}"
        path.write_text("\n".join(new), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
