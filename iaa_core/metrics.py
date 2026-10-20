"""
IAA Metrics Module
Provides categorical and text similarity metrics for Inter-Annotator Agreement evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.metrics import cohen_kappa_score as sklearn_kappa
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CATEGORICAL AGREEMENT METRICS
# ============================================================================

def observed_agreement(y1: List, y2: List) -> float:
    """
    Calculate observed agreement rate between two raters

    Args:
        y1: List of labels from rater 1
        y2: List of labels from rater 2

    Returns:
        float: Agreement rate in range [0, 1]
    """
    if len(y1) != len(y2):
        raise ValueError("Arrays must have same length")

    agreements = sum(1 for a, b in zip(y1, y2) if a == b)
    return agreements / len(y1) if y1 else 0.0


def pabak_score(y1: List, y2: List) -> float:
    """
    Calculate PABAK (Prevalence-Adjusted Bias-Adjusted Kappa)

    Args:
        y1: List of labels from rater 1
        y2: List of labels from rater 2

    Returns:
        float: PABAK score
    """
    if len(y1) != len(y2):
        raise ValueError("Arrays must have same length")

    po = observed_agreement(y1, y2)
    # PABAK assumes equal prevalence and no bias
    return 2 * po - 1


def gwet_ac_score(y1: List, y2: List) -> float:
    """
    Calculate Gwet's Agreement Coefficient (AC)
    More robust alternative to Cohen's Kappa that handles high agreement better

    Args:
        y1: List of labels from rater 1
        y2: List of labels from rater 2

    Returns:
        float: Gwet's AC score
    """
    if len(y1) != len(y2):
        raise ValueError("Arrays must have same length")

    n = len(y1)
    if n == 0:
        return 0.0

    # Convert to numpy arrays
    y1_arr = np.array(y1)
    y2_arr = np.array(y2)

    # Calculate observed agreement
    po = np.mean(y1_arr == y2_arr)

    # Get unique categories
    categories = list(set(y1 + y2))
    k = len(categories)

    if k <= 1:
        return 1.0 if po == 1.0 else 0.0

    # Calculate marginal probabilities for each category
    marginal_probs = []
    for cat in categories:
        prob1 = np.mean(y1_arr == cat)
        prob2 = np.mean(y2_arr == cat)
        marginal_probs.append((prob1 + prob2) / 2)

    # Calculate expected agreement under independence (Gwet's method)
    pe = sum(p * (1 - p) for p in marginal_probs) / (k - 1)

    # Calculate Gwet's AC
    if pe == 1.0:
        return po
    else:
        return (po - pe) / (1 - pe)


def cohen_kappa(y1: List, y2: List) -> float:
    """
    Calculate Cohen's Kappa using sklearn

    Args:
        y1: List of labels from rater 1
        y2: List of labels from rater 2

    Returns:
        float: Cohen's Kappa score
    """
    try:
        return sklearn_kappa(y1, y2)
    except Exception:
        return 0.0


# ============================================================================
# TEXT SIMILARITY METRICS
# ============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts based on token overlap

    Args:
        text1: First text
        text2: Second text

    Returns:
        float: Jaccard similarity in range [0, 1]
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0

    # Tokenize and create sets
    tokens1 = set(str(text1).lower().split())
    tokens2 = set(str(text2).lower().split())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def rouge_l_similarity(text1: str, text2: str, scorer=None) -> float:
    """
    Calculate ROUGE-L similarity

    Args:
        text1: First text
        text2: Second text
        scorer: Optional rouge_scorer instance (for efficiency)

    Returns:
        float: ROUGE-L F-measure in range [0, 1]
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0

    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = scorer.score(str(text1), str(text2))
    return scores['rougeL'].fmeasure


def bert_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """
    Calculate BERT/SentenceTransformer similarity using semantic embeddings

    Args:
        text1: First text
        text2: Second text
        model: SentenceTransformer model instance

    Returns:
        float: Cosine similarity of embeddings in range [0, 1]
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0

    embeddings = model.encode([str(text1), str(text2)])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)


def sent2vec_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """
    Calculate sentence-level embeddings similarity using SentenceTransformers
    (Alias for bert_similarity for backwards compatibility)

    Args:
        text1: First text
        text2: Second text
        model: SentenceTransformer model instance

    Returns:
        float: Cosine similarity in range [0, 1]
    """
    return bert_similarity(text1, text2, model)


def meteor_similarity(text1: str, text2: str) -> float:
    """
    Calculate METEOR score using NLTK
    Falls back to simplified implementation if NLTK METEOR unavailable

    Args:
        text1: First text (hypothesis)
        text2: Second text (reference)

    Returns:
        float: METEOR score in range [0, 1]
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0

    try:
        # Try NLTK METEOR first
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        hypothesis = str(text1).split()
        reference = [str(text2).split()]
        score = nltk_meteor(reference, hypothesis)
        return float(score)
    except Exception:
        # Fallback to simplified METEOR implementation
        return simplified_meteor(text1, text2)


def simplified_meteor(text1: str, text2: str) -> float:
    """
    Simplified METEOR implementation without external dependencies
    Uses unigram matching with fragmentation penalty

    Args:
        text1: First text (hypothesis)
        text2: Second text (reference)

    Returns:
        float: Simplified METEOR score in range [0, 1]
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0

    # Tokenize
    hyp_tokens = str(text1).lower().split()
    ref_tokens = str(text2).lower().split()

    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Find unigram matches
    hyp_matched = [False] * len(hyp_tokens)
    ref_matched = [False] * len(ref_tokens)
    matches = 0

    for i, h_token in enumerate(hyp_tokens):
        for j, r_token in enumerate(ref_tokens):
            if not ref_matched[j] and h_token == r_token:
                hyp_matched[i] = True
                ref_matched[j] = True
                matches += 1
                break

    # Calculate precision and recall
    precision = matches / len(hyp_tokens) if hyp_tokens else 0.0
    recall = matches / len(ref_tokens) if ref_tokens else 0.0

    # Calculate F-mean (harmonic mean with recall weighted 9x more than precision)
    if precision + recall == 0:
        f_mean = 0.0
    else:
        f_mean = (10 * precision * recall) / (recall + 9 * precision)

    # Calculate fragmentation penalty
    chunks = 0
    in_chunk = False
    for matched in hyp_matched:
        if matched:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False

    penalty = 0.5 * (chunks / matches) if matches > 0 else 0.0

    # Final METEOR score
    meteor = f_mean * (1 - penalty)
    return float(meteor)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_delimited_text(text: str, separator: str) -> str:
    """
    Split text by separator, strip each part, and rejoin with a single space.

    This normalises fields that store multiple items as a delimited string so
    that all downstream metrics receive clean, comma/newline-free text.

    Examples:
        "individuality, self-expression"  (sep=',')  → "individuality self-expression"
        "self-expression, individuality"  (sep=',')  → "self-expression individuality"
        "Sentence A.\nSentence B."        (sep='\n') → "Sentence A. Sentence B."

    Args:
        text: Raw field value
        separator: Delimiter character (',' or '\n')

    Returns:
        str: Normalised text with items joined by a single space
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    parts = [p.strip() for p in str(text).split(separator) if p.strip()]
    return ' '.join(parts)


def instruct_score(text1: str, text2: str, model: SentenceTransformer) -> float:
    """
    Simplified InstructScore implementation using semantic similarity
    (Alias for bert_similarity)

    Args:
        text1: First text
        text2: Second text
        model: SentenceTransformer model instance

    Returns:
        float: Semantic similarity score in range [0, 1]
    """
    return bert_similarity(text1, text2, model)


def init_instructscore_backend(repo_path: Optional[str] = None) -> Optional[Any]:
    """
    Initialize InstructScore backend if available.

    This function tries to import and initialize InstructScore from the SEScore3 repo.

    Args:
        repo_path: Optional local path to cloned InstructScore_SEScore3 repo.
                  If None, defaults to '../InstructScore_SEScore3' or './InstructScore_SEScore3'

    Returns:
        Backend scorer object, or None when unavailable.
    """
    # Determine repo path
    if not repo_path:
        # Try common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'InstructScore_SEScore3'),
            os.path.join(os.path.expanduser('~'), 'InstructScore_SEScore3'),
        ]
        for path in possible_paths:
            if os.path.isdir(path):
                repo_path = path
                break

    if not repo_path or not os.path.isdir(repo_path):
        return None

    # Add repo to path
    abs_repo = os.path.abspath(repo_path)
    if abs_repo not in sys.path:
        sys.path.insert(0, abs_repo)

    try:
        from InstructScore import InstructScore
        import torch

        # Initialize with 'caption' task type (most general for text comparison)
        # Use CPU if CUDA not available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  Initializing InstructScore backend (device: {device})...")
        print("  This may take a while as the model is downloaded from HuggingFace...")

        scorer = InstructScore(
            device_id=device,
            task_type='caption',  # General text comparison
            batch_size=4,
            cache_dir=None
        )
        return scorer
    except Exception as e:
        print(f"  Warning: Failed to initialize InstructScore: {e}")
        return None


def _call_instruct_backend(scorer: Any, text1: str, text2: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Call InstructScore backend to score one text pair.

    For InstructScore, text1 is treated as reference and text2 as output.
    Returns:
        Tuple of (score, diagnostic_text) or (None, None) if failed
    """
    if scorer is None:
        return None, None

    try:
        # InstructScore expects: score(ref_ls, out_ls)
        # Returns: (batch_outputs, scores_ls) where
        #   - batch_outputs: list of diagnostic text strings
        #   - scores_ls: list of numerical scores
        batch_outputs, scores_ls = scorer.score(ref_ls=[text1], out_ls=[text2])

        # Extract score (normalize to 0-1 range if needed)
        score = None
        if isinstance(scores_ls, (list, tuple)) and len(scores_ls) > 0:
            raw_score = float(scores_ls[0])
            # InstructScore returns negative scores (0 to -inf where 0 is perfect)
            # Convert to similarity score (0 to 1 where 1 is perfect)
            # Using sigmoid-like transformation
            score = 1.0 / (1.0 + abs(raw_score) / 5.0)

        # Extract diagnostic text
        diagnostic = None
        if isinstance(batch_outputs, (list, tuple)) and len(batch_outputs) > 0:
            diagnostic_raw = batch_outputs[0]
            if diagnostic_raw and isinstance(diagnostic_raw, str):
                diagnostic = diagnostic_raw.strip()

        return score, diagnostic

    except Exception as e:
        # If InstructScore fails, return None to trigger fallback
        return None, None


def instructscore_similarity(
    text1: str,
    text2: str,
    fallback_model: SentenceTransformer,
    backend: Optional[Any] = None,
) -> Tuple[float, Optional[str]]:
    """
    Calculate InstructScore similarity when backend is available.

    Falls back to semantic cosine similarity when backend is unavailable.

    Args:
        text1: First text
        text2: Second text
        fallback_model: SentenceTransformer model for fallback scoring
        backend: Initialized InstructScore backend object/callable

    Returns:
        Tuple of (score, diagnostic_text):
            - score: Similarity score in range [0, 1] when possible
            - diagnostic_text: Human-readable explanation (None if unavailable)
    """
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0.0, None

    score, diagnostic = _call_instruct_backend(backend, str(text1), str(text2))
    if score is not None:
        return float(score), diagnostic

    # Safe fallback to embedding-based similarity (no diagnostic available)
    return bert_similarity(text1, text2, fallback_model), None


def combined_paraphrase_score(text1: str, text2: str, model: SentenceTransformer, scorer=None) -> float:
    """
    Combined score for paraphrasing: 0.5×ROUGE-L + 0.5×SentenceBERT

    Args:
        text1: First text
        text2: Second text
        model: SentenceTransformer model instance
        scorer: Optional rouge_scorer instance

    Returns:
        float: Combined paraphrase score in range [0, 1]
    """
    rouge_score = rouge_l_similarity(text1, text2, scorer)
    bert_score = bert_similarity(text1, text2, model)
    return 0.5 * rouge_score + 0.5 * bert_score
