"""
IAA Core - Reusable Inter-Annotator Agreement evaluation framework

This package provides tools for evaluating inter-annotator agreement
with support for categorical metrics (Cohen's Kappa, Observed Agreement,
PABAK, Gwet AC) and text similarity metrics (BERTScore, Jaccard, METEOR).
"""

from .evaluator import FieldAnnotatingEvaluator
from .schema_config import SchemaConfig, create_field_annotating_schema, create_kim_paul_schema
from .reporting import ReportGenerator
from . import metrics

__version__ = '1.0.0'
__author__ = 'CognitiveQG Project'

__all__ = [
    'FieldAnnotatingEvaluator',
    'SchemaConfig',
    'ReportGenerator',
    'create_field_annotating_schema',
    'create_kim_paul_schema',
    'metrics'
]
