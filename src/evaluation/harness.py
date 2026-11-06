"""Evaluation harness for calculating WER, BLEU, and ROUGE metrics on text translations."""

import string
import jiwer
import sacrebleu
from rouge_score import rouge_scorer
from typing import List, Optional, Dict, Any

from src.evaluation.results import EvaluationResult, UNVERIFIED_PREPROCESSING_DISCLAIMER


def normalize_text(text: str, lowercase: bool = True, strip_punct: bool = False) -> str:
    """Normalize text for consistent evaluation.

    Args:
        text: Raw input string.
        lowercase: Lowercase string if True.
        strip_punct: Strip punctuation characters if True.

    Returns:
        Cleaned, normalized text.
    """
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    # Standardize whitespace
    return " ".join(text.strip().split())


class BenchmarkEvaluator:
    """Evaluates translation hypotheses against ground-truth references using jiwer, sacrebleu, and rouge-score."""

    def __init__(self, lowercase: bool = True, strip_punct: bool = False):
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.rouge_evaluator = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(
        self,
        references: List[str],
        hypotheses: List[str],
        split_name: str = "eval",
        average_latency_ms: Optional[float] = None,
    ) -> EvaluationResult:
        """Compute WER, BLEU-1, BLEU-4, and ROUGE-L over reference and hypothesis lists.

        Args:
            references: Ground truth target sentences.
            hypotheses: Model predicted sentences.
            split_name: Name of evaluation split.
            average_latency_ms: Optional mean inference latency per sample.

        Returns:
            EvaluationResult dataclass with embedded disclaimer label.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Mismatch in references count ({len(references)}) and hypotheses count ({len(hypotheses)})"
            )
        if not references:
            raise ValueError("Cannot evaluate empty reference list.")

        # Normalize texts
        norm_refs = [normalize_text(r, self.lowercase, self.strip_punct) for r in references]
        norm_hyps = [normalize_text(h, self.lowercase, self.strip_punct) for h in hypotheses]

        # 1. Compute Word Error Rate (WER) via jiwer
        # Handle empty/zero cases safely
        try:
            wer_score = float(jiwer.wer(norm_refs, norm_hyps))
        except Exception:
            wer_score = 1.0

        # 2. Compute BLEU-1 and BLEU-4 via sacrebleu
        # sacrebleu expects hypotheses list and list of reference lists: [references]
        bleu1_metric = sacrebleu.BLEU(max_ngram_order=1)
        bleu4_metric = sacrebleu.BLEU(max_ngram_order=4)
        
        bleu1_res = bleu1_metric.corpus_score(norm_hyps, [norm_refs])
        bleu4_res = bleu4_metric.corpus_score(norm_hyps, [norm_refs])

        bleu_1 = float(bleu1_res.score)
        bleu_4 = float(bleu4_res.score)


        # 3. Compute ROUGE-L F1 via rouge-score
        rouge_scores = []
        for ref, hyp in zip(norm_refs, norm_hyps):
            score = self.rouge_evaluator.score(ref, hyp)
            rouge_scores.append(score["rougeL"].fmeasure)
        rouge_l = float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0

        return EvaluationResult(
            wer=wer_score,
            bleu_1=bleu_1,
            bleu_4=bleu_4,
            rouge_l=rouge_l,
            num_samples=len(references),
            split_name=split_name,
            average_latency_ms=average_latency_ms,
        )
