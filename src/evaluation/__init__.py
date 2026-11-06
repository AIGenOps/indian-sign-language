"""Evaluation harness and metrics module."""

from src.evaluation.results import EvaluationResult, UNVERIFIED_PREPROCESSING_DISCLAIMER
from src.evaluation.harness import BenchmarkEvaluator, normalize_text

__all__ = [
    "EvaluationResult",
    "UNVERIFIED_PREPROCESSING_DISCLAIMER",
    "BenchmarkEvaluator",
    "normalize_text",
]
