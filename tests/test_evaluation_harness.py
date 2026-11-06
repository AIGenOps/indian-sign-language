"""Unit tests for evaluation harness and metric calculation with immutable disclaimer enforcement."""

import json
import pytest
from src.evaluation import BenchmarkEvaluator, EvaluationResult, UNVERIFIED_PREPROCESSING_DISCLAIMER, normalize_text


def test_normalize_text():
    assert normalize_text("  Hello, World!  ", lowercase=True, strip_punct=True) == "hello world"
    assert normalize_text("Test   sentence.", lowercase=False, strip_punct=False) == "Test sentence."


def test_evaluator_perfect_match():
    evaluator = BenchmarkEvaluator()
    refs = ["the quick brown fox jumps over the lazy dog", "sign language translation test system"]
    hyps = ["the quick brown fox jumps over the lazy dog", "sign language translation test system"]


    result = evaluator.evaluate(refs, hyps, split_name="test_perfect")
    assert pytest.approx(result.wer, abs=1e-3) == 0.0
    assert result.bleu_1 > 95.0
    assert result.bleu_4 > 95.0
    assert pytest.approx(result.rouge_l, abs=1e-3) == 1.0
    assert result.num_samples == 2


def test_evaluator_mismatch():
    evaluator = BenchmarkEvaluator()
    refs = ["the quick brown fox jumps over the lazy dog"]
    hyps = ["completely wrong translation output"]

    result = evaluator.evaluate(refs, hyps, split_name="test_mismatch")
    assert result.wer > 0.5
    assert result.bleu_4 < 10.0
    assert result.rouge_l < 0.5


def test_evaluator_length_mismatch():
    evaluator = BenchmarkEvaluator()
    with pytest.raises(ValueError, match="Mismatch in references count"):
        evaluator.evaluate(["ref1", "ref2"], ["hyp1"])


def test_disclaimer_immutable_labeling():
    result = EvaluationResult(
        wer=0.35,
        bleu_1=45.0,
        bleu_4=20.0,
        rouge_l=0.40,
        num_samples=100,
        split_name="test_disclaimer",
    )

    # 1. Repr and str contain disclaimer
    repr_str = repr(result)
    assert "DISCLAIMER" in repr_str
    assert "measures model+preprocessing combined" in repr_str

    # 2. to_dict contains disclaimer and warning
    res_dict = result.to_dict()
    assert "disclaimer" in res_dict
    assert res_dict["disclaimer"] == UNVERIFIED_PREPROCESSING_DISCLAIMER
    assert res_dict["preprocessing_warning"] == "UNVERIFIED / EXPERIMENTAL (ADR-008)"

    # 3. JSON serialization preserves disclaimer
    json_str = result.to_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["disclaimer"] == UNVERIFIED_PREPROCESSING_DISCLAIMER
    assert parsed_json["preprocessing_warning"] == "UNVERIFIED / EXPERIMENTAL (ADR-008)"
