"""
IAA Evaluator Module
Schema-driven evaluation engine for Inter-Annotator Agreement analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import warnings

from . import metrics
from .schema_config import SchemaConfig

warnings.filterwarnings('ignore')


class FieldAnnotatingEvaluator:
    """
    Schema-driven Inter-Annotator Agreement evaluator
    Supports categorical metrics (Cohen's Kappa, Observed Agreement, PABAK, Gwet AC)
    and text similarity metrics (BERTScore, Jaccard, METEOR)
    """

    def __init__(
        self,
        schema_config: SchemaConfig,
        model_name: str = 'all-MiniLM-L6-v2',
        instructscore_repo_path: Optional[str] = None,
    ):
        """
        Initialize evaluator

        Args:
            schema_config: SchemaConfig instance defining fields to evaluate
            model_name: SentenceTransformer model name for semantic similarity
            instructscore_repo_path: Optional local path to InstructScore_SEScore3 repo
        """
        self.schema = schema_config
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.sentence_model = SentenceTransformer(model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Optional InstructScore backend (falls back safely when unavailable)
        self.instructscore_backend = metrics.init_instructscore_backend(instructscore_repo_path)
        self.instructscore_available = self.instructscore_backend is not None
        if self.instructscore_available:
            print("InstructScore backend: enabled")
        else:
            print("InstructScore backend: not found (fallback to BERT similarity)")

    def load_data(self, file1_path: str, file2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess annotation data from CSV files

        Args:
            file1_path: Path to first annotator's CSV file
            file2_path: Path to second annotator's CSV file

        Returns:
            Tuple of (df1, df2) pandas DataFrames
        """
        print(f"Loading data from {file1_path} and {file2_path}...")
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        print(f"Original data shapes: df1: {df1.shape}, df2: {df2.shape}")

        # Remove rows where argumentIndex is empty or NaN (if column exists)
        if 'argumentIndex' in df1.columns:
            df1 = df1.dropna(subset=['argumentIndex'])
            df2 = df2.dropna(subset=['argumentIndex'])

        # Reset index after dropping rows
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        print(f"After filtering: df1: {df1.shape}, df2: {df2.shape}")

        # Fill NaN values with empty strings for text fields
        for field in self.schema.get_text_fields():
            if field in df1.columns:
                df1[field] = df1[field].fillna('')
                df2[field] = df2[field].fillna('')

        # Validate schema coverage
        validation1 = self.schema.validate_dataframe(df1)
        validation2 = self.schema.validate_dataframe(df2)

        print(f"\nSchema validation for annotator 1:")
        print(f"  Present fields: {len(validation1['present'])}/{len(self.schema.get_all_fields())}")
        if validation1['missing']:
            print(f"  Missing fields: {validation1['missing'][:5]}..." if len(validation1['missing']) > 5 else f"  Missing fields: {validation1['missing']}")

        print(f"\nSchema validation for annotator 2:")
        print(f"  Present fields: {len(validation2['present'])}/{len(self.schema.get_all_fields())}")
        if validation2['missing']:
            print(f"  Missing fields: {validation2['missing'][:5]}..." if len(validation2['missing']) > 5 else f"  Missing fields: {validation2['missing']}")

        return df1, df2

    def evaluate_categorical_field(self, df1: pd.DataFrame, df2: pd.DataFrame, field: str) -> Dict:
        """
        Evaluate categorical field using agreement metrics

        Args:
            df1: First annotator's DataFrame
            df2: Second annotator's DataFrame
            field: Field name to evaluate

        Returns:
            dict: Results with Cohen's Kappa, Observed Agreement, PABAK, Gwet AC
        """
        series1 = df1[field].fillna('missing')
        series2 = df2[field].fillna('missing')

        # Convert to lists for calculations
        y1 = series1.tolist()
        y2 = series2.tolist()

        # Calculate metrics
        kappa = metrics.cohen_kappa(y1, y2)
        observed_agr = metrics.observed_agreement(y1, y2)
        pabak = metrics.pabak_score(y1, y2)
        gwet_ac = metrics.gwet_ac_score(y1, y2)

        # Get confusion details
        agreements = sum(1 for a, b in zip(y1, y2) if a == b)
        disagreements = len(y1) - agreements

        return {
            'cohens_kappa': kappa,
            'observed_agreement': observed_agr,
            'pabak': pabak,
            'gwet_ac': gwet_ac,
            'unique_values_annotator1': len(set(y1)),
            'unique_values_annotator2': len(set(y2)),
            'total_agreements': agreements,
            'total_disagreements': disagreements
        }

    def _normalize(self, text, field: str) -> str:
        """Pre-process a field value using its configured separator, if any."""
        cfg = self.schema.get_field_config(field)
        if cfg and cfg.separator:
            return metrics.normalize_delimited_text(text, cfg.separator)
        return '' if pd.isna(text) else str(text)

    def evaluate_text_field(self, df1: pd.DataFrame, df2: pd.DataFrame, field: str) -> Dict:
        """
        Evaluate text field using similarity metrics

        Args:
            df1: First annotator's DataFrame
            df2: Second annotator's DataFrame
            field: Field name to evaluate

        Returns:
            dict: Results with BERTScore, Jaccard, METEOR metrics
        """
        series1 = df1[field]
        series2 = df2[field]

        # Calculate metrics for ALL pairs (including empty)
        jaccard_scores = []
        rouge_scores = []
        bert_scores = []
        instruct_scores = []
        instruct_diagnostics = []
        meteor_scores = []

        # Track non-empty pairs separately
        jaccard_scores_valid = []
        rouge_scores_valid = []
        bert_scores_valid = []
        instruct_scores_valid = []
        instruct_diagnostics_valid = []
        meteor_scores_valid = []
        valid_pairs = []

        for text1, text2 in zip(series1, series2):
            # Normalise delimiter-separated fields before metric calculation
            text1 = self._normalize(text1, field)
            text2 = self._normalize(text2, field)

            # Calculate for ALL pairs
            jaccard = metrics.jaccard_similarity(text1, text2)
            rouge = metrics.rouge_l_similarity(text1, text2, self.rouge_scorer)
            bert = metrics.bert_similarity(text1, text2, self.sentence_model)
            instruct_score, instruct_diagnostic = metrics.instructscore_similarity(
                text1,
                text2,
                self.sentence_model,
                backend=self.instructscore_backend,
            )
            meteor = metrics.meteor_similarity(text1, text2)

            jaccard_scores.append(jaccard)
            rouge_scores.append(rouge)
            bert_scores.append(bert)
            instruct_scores.append(instruct_score)
            instruct_diagnostics.append(instruct_diagnostic)
            meteor_scores.append(meteor)

            # Check if both texts have content (non-empty pair)
            if (str(text1).strip() and str(text2).strip() and
                    text1 != '' and text2 != '' and
                    not pd.isna(text1) and not pd.isna(text2)):

                valid_pairs.append((text1, text2))
                jaccard_scores_valid.append(jaccard)
                rouge_scores_valid.append(rouge)
                bert_scores_valid.append(bert)
                instruct_scores_valid.append(instruct_score)
                instruct_diagnostics_valid.append(instruct_diagnostic)
                meteor_scores_valid.append(meteor)

        # Prepare results
        total_pairs = len(series1)
        valid_count = len(valid_pairs)

        result = {
            # Original metrics (ALL pairs including empty)
            'jaccard': {
                'mean': np.mean(jaccard_scores),
                'std': np.std(jaccard_scores),
                'scores': jaccard_scores
            },
            'rouge_l': {
                'mean': np.mean(rouge_scores),
                'std': np.std(rouge_scores),
                'scores': rouge_scores
            },
            'bert': {
                'mean': np.mean(bert_scores),
                'std': np.std(bert_scores),
                'scores': bert_scores
            },
            'instructscore': {
                'mean': np.mean(instruct_scores),
                'std': np.std(instruct_scores),
                'scores': instruct_scores,
                'diagnostics': instruct_diagnostics,
                'backend_available': self.instructscore_available,
            },
            'meteor': {
                'mean': np.mean(meteor_scores),
                'std': np.std(meteor_scores),
                'scores': meteor_scores
            },

            # Coverage statistics
            'coverage': {
                'total_pairs': total_pairs,
                'valid_pairs': valid_count,
                'coverage_rate': valid_count / total_pairs if total_pairs > 0 else 0,
                'empty_pairs': total_pairs - valid_count
            }
        }

        # Add non-empty metrics only if valid pairs exist
        if valid_count > 0:
            result['non_empty_only'] = {
                'jaccard': {
                    'mean': np.mean(jaccard_scores_valid),
                    'std': np.std(jaccard_scores_valid) if valid_count > 1 else 0,
                    'scores': jaccard_scores_valid
                },
                'rouge_l': {
                    'mean': np.mean(rouge_scores_valid),
                    'std': np.std(rouge_scores_valid) if valid_count > 1 else 0,
                    'scores': rouge_scores_valid
                },
                'bert': {
                    'mean': np.mean(bert_scores_valid),
                    'std': np.std(bert_scores_valid) if valid_count > 1 else 0,
                    'scores': bert_scores_valid
                },
                'instructscore': {
                    'mean': np.mean(instruct_scores_valid),
                    'std': np.std(instruct_scores_valid) if valid_count > 1 else 0,
                    'scores': instruct_scores_valid,
                    'diagnostics': instruct_diagnostics_valid,
                    'backend_available': self.instructscore_available,
                },
                'meteor': {
                    'mean': np.mean(meteor_scores_valid),
                    'std': np.std(meteor_scores_valid) if valid_count > 1 else 0,
                    'scores': meteor_scores_valid
                }
            }
        else:
            result['non_empty_only'] = None

        return result

    def evaluate_all_fields(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Evaluate all fields defined in schema

        Args:
            df1: First annotator's DataFrame
            df2: Second annotator's DataFrame

        Returns:
            dict: Evaluation results for all fields
        """
        results = {}

        # Evaluate categorical fields
        print("\nEvaluating categorical fields...")
        for field in self.schema.get_categorical_fields():
            if field in df1.columns and field in df2.columns:
                try:
                    results[field] = self.evaluate_categorical_field(df1, df2, field)
                    print(f"  ✓ {field}")
                except Exception as e:
                    print(f"  ✗ {field}: {str(e)}")
            else:
                print(f"  ⊘ {field} (not found in data)")

        # Evaluate text fields
        print("\nEvaluating text similarity fields...")
        for field in self.schema.get_text_fields():
            if field in df1.columns and field in df2.columns:
                try:
                    results[field] = self.evaluate_text_field(df1, df2, field)
                    print(f"  ✓ {field}")
                except Exception as e:
                    print(f"  ✗ {field}: {str(e)}")
            else:
                print(f"  ⊘ {field} (not found in data)")

        return results
