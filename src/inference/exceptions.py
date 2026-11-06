"""Custom Typed Exception Hierarchy for Streaming Inference (Prompt 03 Stage C).

Avoids bare exceptions by providing explicit, domain-specific exception types for streaming operations.
"""

from __future__ import annotations


class StreamingInferenceError(Exception):
    """Base exception for streaming inference failures."""


class CameraConnectionError(StreamingInferenceError):
    """Raised when the camera hardware source drops connection and reconnection fails."""


class BufferError(StreamingInferenceError):
    """Raised for frame buffer operational errors."""


class ModelInferenceError(StreamingInferenceError):
    """Raised when model forward pass or text generation fails during inference."""
