"""Evaluation results container with immutable benchmark disclaimer label.

Ensures the unverified preprocessing warning (ADR-008) is permanently attached
to all generated metrics in __repr__, __str__, to_dict(), and JSON output.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

UNVERIFIED_PREPROCESSING_DISCLAIMER = (
    "measures model+preprocessing combined; preprocessing is unverified/guessed -- see ADR-008 -- "
    "so this number cannot be attributed to the model alone."
)


@dataclass
class EvaluationResult:
    """Benchmark evaluation result dataclass."""

    wer: float  # Word Error Rate (0.0 to 1.0+)
    bleu_1: float  # BLEU-1 score (0.0 to 100.0)
    bleu_4: float  # BLEU-4 score (0.0 to 100.0)
    rouge_l: float  # ROUGE-L F1 score (0.0 to 1.0)
    num_samples: int
    split_name: str = "eval"
    average_latency_ms: Optional[float] = None
    disclaimer: str = field(default=UNVERIFIED_PREPROCESSING_DISCLAIMER, init=False)
    preprocessing_warning: str = field(default="UNVERIFIED / EXPERIMENTAL (ADR-008)", init=False)

    def __post_init__(self):
        # Force disclaimer integrity
        object.__setattr__(self, "disclaimer", UNVERIFIED_PREPROCESSING_DISCLAIMER)
        object.__setattr__(self, "preprocessing_warning", "UNVERIFIED / EXPERIMENTAL (ADR-008)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dict, ensuring disclaimer keys are preserved."""
        res = asdict(self)
        res["disclaimer"] = UNVERIFIED_PREPROCESSING_DISCLAIMER
        res["preprocessing_warning"] = "UNVERIFIED / EXPERIMENTAL (ADR-008)"
        return res

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to JSON with immutable disclaimer."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult({self.split_name.upper()} | WER={self.wer:.4f}, BLEU-1={self.bleu_1:.2f}, "
            f"BLEU-4={self.bleu_4:.2f}, ROUGE-L={self.rouge_l:.4f}, samples={self.num_samples} | "
            f"DISCLAIMER: '{self.disclaimer}')"
        )

    def __str__(self) -> str:
        return self.__repr__()
