"""Latency Metrics Module for Streaming Inference Pipeline (Prompt 03 Stage C)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyMetrics:
    """Per-stage latency measurements in milliseconds."""

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    model_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "capture_ms": round(self.capture_ms, 2),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "model_ms": round(self.model_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }
