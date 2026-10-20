"""
LLM-as-a-Judge module for Socratic Question Generation evaluation.

Provides GPT-based scoring of generated Socratic questions on 6 pedagogical
dimensions, with disk-based caching, retry logic, and self-consistency support.

Scoring modes:
  scalar      — rubric-based 1–5 scoring per dimension
  equivalence — binary pedagogical equivalence judgment
  pairwise    — A-vs-B preference comparison

Usage:
    from iaa_core.llm_judge import SocraticJudge

    judge = SocraticJudge(model="gpt-4.5-preview", mode="scalar")
    score = judge.score_pair(argument, model_q, gold_q)
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

JUDGE_DIMENSIONS = [
    "reasoning_depth",
    "contextual_relevance",
    "critical_thinking_stimulation",
    "pedagogical_equivalence",
    "semantic_equivalence",
    "focus_appropriateness",
]

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "llm_judge"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from GPT response, stripping markdown fences."""
    return json.loads(_strip_fences(raw))


def _cache_key(parts: List[str]) -> str:
    combined = "\x00".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class SocraticJudge:
    """
    GPT-based evaluator for Socratic question quality.

    Args:
        model:               OpenAI model name (default: gpt-4o)
        mode:                "scalar" | "equivalence" | "pairwise"
        cache_dir:           Directory for disk-based response caching.
                             Set to None to disable caching.
        temperature:         Sampling temperature (0.0 = deterministic)
        n_consistency_runs:  Score each pair N times and average (scalar mode only).
                             Use 3 for reliability checks; 1 for production runs.
        rate_limit_delay:    Seconds to wait between API calls (default: 1.0)
    """

    def __init__(
        self,
        model: str = "gpt-5.5",
        mode: str = "scalar",
        cache_dir: Optional[str] = ".judge_cache",
        temperature: float = 1.0,
        n_consistency_runs: int = 1,
        rate_limit_delay: float = 1.0,
    ):
        if mode not in ("scalar", "equivalence", "pairwise"):
            raise ValueError(f"mode must be 'scalar', 'equivalence', or 'pairwise'; got {mode!r}")

        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.n_consistency_runs = n_consistency_runs
        self.rate_limit_delay = rate_limit_delay

        # Lazy-load OpenAI client
        self._client = None

        # Disk cache (simple JSON files, no extra dependencies)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load system prompts
        self._system_prompts = {
            "scalar":      _load_prompt("rubric_scalar.txt"),
            "equivalence": _load_prompt("pedagogical_equivalence.txt"),
            "pairwise":    _load_prompt("pairwise_comparison.txt"),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pair(
        self,
        argument: str,
        model_q: str,
        gold_q: str,
        focus_type: Optional[str] = None,
    ) -> dict:
        """
        Score a (model_q, gold_q) pair on 6 pedagogical dimensions.

        Returns dict with keys: reasoning_depth, contextual_relevance,
        critical_thinking_stimulation, pedagogical_equivalence,
        semantic_equivalence, focus_appropriateness, composite,
        rationale, raw_response.

        Empty questions return a zero-score dict without an API call.
        """
        if not model_q.strip() or not gold_q.strip():
            return self._zero_score(reason="empty_question")

        if self.mode != "scalar":
            raise ValueError("score_pair() requires mode='scalar'")

        scores_list = []
        for run_i in range(self.n_consistency_runs):
            raw = self._call_with_cache(
                mode="scalar",
                argument=argument,
                q_a=model_q,
                q_b=gold_q,
                focus_type=focus_type,
                run_index=run_i if self.n_consistency_runs > 1 else 0,
            )
            parsed = self._parse_scalar(raw)
            scores_list.append(parsed)

        return self._average_scalar_runs(scores_list)

    def judge_equivalence(
        self,
        argument: str,
        model_q: str,
        gold_q: str,
    ) -> dict:
        """
        Binary judgment: do model_q and gold_q probe the same reasoning weakness?

        Returns dict with keys: equivalent (bool), confidence (float), explanation (str).
        """
        if not model_q.strip() or not gold_q.strip():
            return {"equivalent": False, "confidence": 0.0, "explanation": "empty_question"}

        if self.mode != "equivalence":
            raise ValueError("judge_equivalence() requires mode='equivalence'")

        raw = self._call_with_cache(
            mode="equivalence",
            argument=argument,
            q_a=model_q,
            q_b=gold_q,
        )
        try:
            result = _parse_json_response(raw)
            return {
                "equivalent":   bool(result.get("equivalent", False)),
                "confidence":   float(result.get("confidence", 0.0)),
                "explanation":  str(result.get("explanation", "")),
                "raw_response": raw,
            }
        except Exception:
            return {"equivalent": False, "confidence": 0.0, "explanation": raw, "raw_response": raw}

    def compare_pair(
        self,
        argument: str,
        q_a: str,
        q_b: str,
    ) -> dict:
        """
        Pairwise comparison: which of q_a (A) or q_b (B) is the better Socratic question?

        Returns dict with keys: preferred ("A" | "B" | "tie"), rationale.
        """
        if self.mode != "pairwise":
            raise ValueError("compare_pair() requires mode='pairwise'")

        raw = self._call_with_cache(
            mode="pairwise",
            argument=argument,
            q_a=q_a,
            q_b=q_b,
        )
        try:
            result = _parse_json_response(raw)
            preferred = str(result.get("preferred", "tie")).upper()
            if preferred not in ("A", "B", "TIE"):
                preferred = "tie"
            return {
                "preferred":    preferred.capitalize() if preferred in ("A", "B") else "tie",
                "rationale":    str(result.get("rationale", "")),
                "raw_response": raw,
            }
        except Exception:
            return {"preferred": "tie", "rationale": raw, "raw_response": raw}

    def score_aligned_pairs(
        self,
        aligned_pairs: List[dict],
        focus_by_arg: Optional[Dict[int, List[dict]]] = None,
    ) -> List[dict]:
        """
        Score a list of aligned (model_q, gold_q) pairs from model_vs_gold_evaluation.

        Each pair dict must have: arg_index, question_position, model_question, gold_question.
        Adds 'llm_judge' key to each pair in-place and returns the updated list.

        focus_by_arg: optional dict mapping arg_index -> list of focus classification records
                      (positional order). When provided, the focus_type for each position is
                      included in the judge prompt.
        """
        total = len(aligned_pairs)
        scored = 0
        skipped = 0

        for pair in aligned_pairs:
            arg_idx  = int(pair["arg_index"])
            pos      = int(pair["question_position"])  # 1-indexed
            argument = pair.get("argument", "")
            model_q  = pair.get("model_question", "")
            gold_q   = pair.get("gold_question",  "")

            # Resolve optional focus type for this position
            focus_type = None
            if focus_by_arg and arg_idx in focus_by_arg:
                focus_list = focus_by_arg[arg_idx]
                zero_pos = pos - 1
                if zero_pos < len(focus_list):
                    focus_type = str(focus_list[zero_pos].get("focus_type", "") or "").strip() or None

            if not model_q.strip() or not gold_q.strip():
                pair["llm_judge"] = self._zero_score(reason="empty_question")
                skipped += 1
                print(f"  [{scored + skipped}/{total}] arg={arg_idx} pos={pos} — skipped (empty)", flush=True)
                continue

            if self.mode == "scalar":
                pair["llm_judge"] = self.score_pair(argument, model_q, gold_q, focus_type)
            elif self.mode == "equivalence":
                pair["llm_judge"] = self.judge_equivalence(argument, model_q, gold_q)
            elif self.mode == "pairwise":
                pair["llm_judge"] = self.compare_pair(argument, model_q, gold_q)

            composite = pair["llm_judge"].get("composite", 0)
            print(f"  [{scored + skipped + 1}/{total}] arg={arg_idx} pos={pos} — composite={composite:.2f}", flush=True)
            scored += 1
            if self.rate_limit_delay > 0 and scored < total:
                time.sleep(self.rate_limit_delay)

        print(f"  LLM judge: scored {scored} pairs, skipped {skipped} (empty)")
        return aligned_pairs

    def rescore_pairs(self, scored_pairs: List[dict]) -> List[dict]:
        """
        Rescore existing scored pairs using the current (updated) prompt.

        Reads pairs from a previously saved judge JSON (the 'scored_pairs' list from
        evaluate_llm_judge.py output), re-runs the full scalar rubric on each non-skipped
        pair, and returns updated pairs. The updated prompt produces new cache keys so
        old scores are preserved and new scores coexist in the cache.

        Input pairs may use either 'judge' or 'llm_judge' as the score key; output
        always uses 'judge'.
        """
        total   = len(scored_pairs)
        scored  = 0
        skipped = 0

        for pair in scored_pairs:
            old_j = pair.get("judge", pair.get("llm_judge", {})) or {}
            if old_j.get("_skipped"):
                skipped += 1
                pair["judge"] = old_j
                continue

            argument = pair.get("argument", "")
            model_q  = str(pair.get("model_question", "") or "").strip()
            gold_q   = str(pair.get("gold_question",  "") or "").strip()

            if not model_q or not gold_q:
                pair["judge"] = self._zero_score(reason="empty_question")
                skipped += 1
                continue

            new_score = self.score_pair(argument, model_q, gold_q)
            pair["judge"] = new_score
            scored += 1

            se_old = old_j.get("semantic_equivalence", "?")
            se_new = new_score.get("semantic_equivalence", "?")
            arg_idx = pair.get("arg_index", "?")
            pos     = pair.get("question_position", "?")
            print(
                f"  [{scored}/{total - skipped}] arg={arg_idx} pos={pos} "
                f"SE: {se_old}→{se_new}  composite={new_score.get('composite', 0):.2f}",
                flush=True,
            )
            if self.rate_limit_delay > 0 and scored < total - skipped:
                time.sleep(self.rate_limit_delay)

        print(f"  Rescore complete: {scored} rescored, {skipped} skipped")
        return scored_pairs

    def aggregate_scores(
        self,
        scored_pairs: List[dict],
        positions: Optional[List[int]] = None,
    ) -> dict:
        """
        Aggregate llm_judge scores from scored_pairs into per-position and overall stats.

        Returns a dict with structure:
          {
            "per_position": {1: {dim: {mean, std}, ...}, 2: ..., 3: ...},
            "overall":      {dim: {mean, std}, ...},
            "model":        self.model,
            "mode":         self.mode,
            "total_pairs_scored": int,
          }

        The per-dimension dicts use the same {mean, std} format as _score_pair_list()
        in model_vs_gold_evaluation.py so ModelVsGoldReporter can consume them directly.
        """
        if positions is None:
            positions = sorted({int(p["question_position"]) for p in scored_pairs})

        result = {
            "per_position": {},
            "overall": {},
            "model": self.model,
            "mode":  self.mode,
            "total_pairs_scored": 0,
        }

        for pos in positions:
            pos_pairs = [p for p in scored_pairs if int(p["question_position"]) == pos]
            result["per_position"][pos] = self._aggregate_pair_list(pos_pairs)

        result["overall"] = self._aggregate_pair_list(scored_pairs)
        result["total_pairs_scored"] = sum(
            1 for p in scored_pairs
            if (p.get("judge") or p.get("llm_judge") or {}).get("_skipped") is None
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai>=1.0.0"
                )
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable not set."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _call_api(self, system_prompt: str, user_message: str, max_retries: int = 3) -> str:
        """Call OpenAI API with exponential backoff. Returns raw response string."""
        client = self._get_client()
        delay = 2.0
        for attempt in range(max_retries):
            try:
                params = dict(
                    model=self.model,
                    messages=[
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_message},
                    ],
                    max_completion_tokens=512,
                    response_format={"type": "json_object"},
                )
                # Some models (e.g. gpt-5.5) only accept the default temperature
                if self.temperature != 1.0:
                    params["temperature"] = self.temperature
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    if "429" in err or "rate_limit" in err.lower():
                        wait = delay * (2 ** attempt)
                        print(f"  Rate limit hit, waiting {wait:.0f}s...")
                        time.sleep(wait)
                    else:
                        time.sleep(delay)
                else:
                    raise

    def _call_with_cache(
        self,
        mode: str,
        argument: str,
        q_a: str,
        q_b: str,
        focus_type: Optional[str] = None,
        run_index: int = 0,
    ) -> str:
        """Return cached response or call API and store result."""
        # Include a short hash of the system prompt so updated prompts produce new cache entries
        prompt_hash = _cache_key([self._system_prompts[mode]])[:16]
        key_parts = [mode, self.model, prompt_hash, argument, q_a, q_b, str(focus_type), str(run_index)]
        key = _cache_key(key_parts)

        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)["raw"]

        system_prompt = self._system_prompts[mode]
        user_message  = self._build_user_message(mode, argument, q_a, q_b, focus_type, run_index)
        raw = self._call_api(system_prompt, user_message)

        if self.cache_dir:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "key_parts": key_parts}, f, ensure_ascii=False)

        return raw

    def _build_user_message(
        self,
        mode: str,
        argument: str,
        q_a: str,
        q_b: str,
        focus_type: Optional[str],
        run_index: int,
    ) -> str:
        if mode == "scalar":
            parts = [
                f"ARGUMENT:\n{argument}",
                f"\nHUMAN REFERENCE QUESTION:\n{q_b}",
                f"\nMODEL-GENERATED QUESTION:\n{q_a}",
            ]
            if focus_type:
                parts.append(f"\nEXPECTED FOCUS TYPE: {focus_type}")
            # Slight perturbation for consistency runs to break identical cache keys
            if run_index > 0:
                parts.append(f"\n[Evaluation run {run_index + 1}]")
            return "\n".join(parts)

        elif mode == "equivalence":
            return (
                f"ARGUMENT:\n{argument}"
                f"\n\nQUESTION A:\n{q_a}"
                f"\n\nQUESTION B:\n{q_b}"
            )

        elif mode == "pairwise":
            return (
                f"ARGUMENT:\n{argument}"
                f"\n\nQUESTION A:\n{q_a}"
                f"\n\nQUESTION B:\n{q_b}"
            )

        raise ValueError(f"Unknown mode: {mode}")

    def _parse_scalar(self, raw: str) -> dict:
        """Parse a scalar rubric response, returning a score dict."""
        try:
            data = _parse_json_response(raw)
            scores = {}
            for dim in JUDGE_DIMENSIONS:
                val = data.get(dim)
                try:
                    scores[dim] = max(1, min(5, int(val)))
                except (TypeError, ValueError):
                    scores[dim] = 1
            scores["composite"] = round(float(np.mean([scores[d] for d in JUDGE_DIMENSIONS])), 4)
            scores["rationale"]    = str(data.get("rationale", ""))
            scores["raw_response"] = raw
            return scores
        except Exception:
            zero = self._zero_score(reason="parse_error")
            zero["raw_response"] = raw
            return zero

    def _average_scalar_runs(self, runs: List[dict]) -> dict:
        """Average scores across multiple consistency runs."""
        if len(runs) == 1:
            return runs[0]
        result = {}
        for dim in JUDGE_DIMENSIONS:
            vals = [r[dim] for r in runs if isinstance(r.get(dim), (int, float))]
            result[dim] = round(float(np.mean(vals)), 4) if vals else 1.0
        result["composite"]    = round(float(np.mean([result[d] for d in JUDGE_DIMENSIONS])), 4)
        result["rationale"]    = runs[0].get("rationale", "")
        result["raw_response"] = runs[0].get("raw_response", "")
        result["n_runs"]       = len(runs)
        return result

    def _zero_score(self, reason: str = "empty") -> dict:
        result = {dim: 0 for dim in JUDGE_DIMENSIONS}
        result["composite"]    = 0.0
        result["rationale"]    = ""
        result["raw_response"] = ""
        result["_skipped"]     = reason
        return result

    def _aggregate_pair_list(self, pairs: List[dict]) -> dict:
        """
        Compute per-dimension {mean, std} stats from a list of scored pairs.
        Mirrors the structure returned by _score_pair_list() in model_vs_gold_evaluation.py.
        Skipped (empty) pairs contribute 0 to all-pairs stats; excluded from non_empty_only.
        """
        all_scores:   Dict[str, List[float]] = {dim: [] for dim in JUDGE_DIMENSIONS + ["composite"]}
        valid_scores: Dict[str, List[float]] = {dim: [] for dim in JUDGE_DIMENSIONS + ["composite"]}

        for p in pairs:
            j = p.get("judge") or p.get("llm_judge") or {}
            is_valid = j.get("_skipped") is None and j.get("composite", 0) > 0
            for dim in JUDGE_DIMENSIONS + ["composite"]:
                v = float(j.get(dim, 0) or 0)
                all_scores[dim].append(v)
                if is_valid:
                    valid_scores[dim].append(v)

        def stat(scores):
            n = len(scores)
            return {
                "mean": float(np.mean(scores)) if n > 0 else 0.0,
                "std":  float(np.std(scores))  if n > 0 else 0.0,
            }

        result = {}
        for dim in JUDGE_DIMENSIONS + ["composite"]:
            result[dim] = stat(all_scores[dim])

        valid_count = len(valid_scores[JUDGE_DIMENSIONS[0]])
        if valid_count > 0:
            result["non_empty_only"] = {dim: stat(valid_scores[dim]) for dim in JUDGE_DIMENSIONS + ["composite"]}
        else:
            result["non_empty_only"] = None

        result["coverage"] = {
            "total_pairs": len(pairs),
            "valid_pairs": valid_count,
        }
        return result
