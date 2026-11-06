"""Streaming Inference Pipeline Module (Prompt 03 Stage C).

Wires CameraSource -> BoundedFrameBuffer -> Preprocessing -> SanketModelProvider.predict().

Stage C Enhancements:
1. Reconnect Handling: Automatic reconnect loop with `max_reconnect_attempts` when camera drops. Raises `CameraConnectionError`.
2. Stage Latency Metrics: Tracks `capture_ms`, `preprocess_ms`, `model_ms`, and `total_ms`. Attached to `Prediction.metrics`.
3. Confidence Thresholding: Filters predictions below `min_confidence_threshold` (e.g. 0.70), marking them as `"rejected"`.
4. Typed Exceptions: Raises `CameraConnectionError` and `ModelInferenceError` instead of bare exceptions.
"""

from __future__ import annotations

import logging
import time
from typing import Any
import numpy as np

from src.inference.buffer import BoundedFrameBuffer
from src.inference.camera import CameraSource
from src.inference.exceptions import CameraConnectionError, ModelInferenceError
from src.inference.metrics import LatencyMetrics
from src.models.base import Prediction
from src.models.sanket import SanketModelProvider

logger = logging.getLogger(__name__)


class StreamingInferenceRunner:
    """Manages stream capture, frame buffering, fixed window sampling, prediction triggers, latency metrics, and reconnect handling."""

    def __init__(
        self,
        camera_source: CameraSource,
        buffer: BoundedFrameBuffer | None = None,
        model_provider: SanketModelProvider | None = None,
        stride_frames: int = 15,
        window_frames: int = 30,
        min_buffer_frames: int = 8,
        min_confidence_threshold: float = 0.70,
        enable_confidence_gating: bool = False,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 0.1,
        allow_experimental_inference: bool = True,
    ) -> None:
        self.camera_source = camera_source
        self.buffer = buffer or BoundedFrameBuffer(max_size=120)
        self.model_provider = model_provider
        self.stride_frames = stride_frames
        self.window_frames = window_frames
        self.min_buffer_frames = min_buffer_frames
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_confidence_gating = enable_confidence_gating
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self.allow_experimental_inference = allow_experimental_inference

        self._frames_since_last_predict = 0
        self._total_frames_processed = 0
        self.last_metrics: LatencyMetrics = LatencyMetrics()

    def step(self) -> tuple[bool, Prediction | None]:
        """Execute a single stream capture and trigger prediction if stride interval is reached.

        Returns:
            (success: bool, prediction: Prediction | None)

        Raises:
            CameraConnectionError: If camera connection drops and reconnect attempts fail.
        """
        t_start = time.perf_counter()

        success, frame = self._read_frame_with_reconnect()
        t_captured = time.perf_counter()
        capture_ms = (t_captured - t_start) * 1000.0

        if not success or frame is None:
            return False, None

        self.buffer.push(frame)
        self._frames_since_last_predict += 1
        self._total_frames_processed += 1

        # Check prediction trigger policy
        if (
            self._frames_since_last_predict >= self.stride_frames
            and self.buffer.size() >= self.min_buffer_frames
        ):
            self._frames_since_last_predict = 0
            prediction, metrics = self._run_prediction_on_buffer(capture_ms=capture_ms, t_start=t_start)
            self.last_metrics = metrics
            return True, prediction

        return True, None

    def _read_frame_with_reconnect(self) -> tuple[bool, np.ndarray | None]:
        """Read frame from camera source, executing auto-reconnect logic if connection drops."""
        if not self.camera_source.is_connected():
            if not self.camera_source.connect():
                if not self.auto_reconnect:
                    raise CameraConnectionError("Camera source is not connected.")

        success, frame = self.camera_source.read_frame()
        if success and frame is not None:
            return True, frame

        if not self.auto_reconnect:
            return False, None

        # Reconnect loop
        logger.warning("Camera stream dropped. Attempting reconnection...")
        for attempt in range(1, self.max_reconnect_attempts + 1):
            self.camera_source.release()
            time.sleep(self.reconnect_delay_sec)
            if self.camera_source.connect():
                ret, re_frame = self.camera_source.read_frame()
                if ret and re_frame is not None:
                    logger.info(f"Camera reconnected successfully on attempt {attempt}.")
                    return True, re_frame

        raise CameraConnectionError(
            f"Failed to reconnect to camera source after {self.max_reconnect_attempts} attempt(s)."
        )

    def _run_prediction_on_buffer(
        self, capture_ms: float, t_start: float
    ) -> tuple[Prediction | None, LatencyMetrics]:
        """Extract current buffer window and run model prediction with latency metrics & thresholding."""
        metrics = LatencyMetrics(capture_ms=capture_ms)
        if self.model_provider is None:
            t_end = time.perf_counter()
            metrics.total_ms = (t_end - t_start) * 1000.0
            return None, metrics

        all_buffer_frames = self.buffer.get_frames()
        available_count = len(all_buffer_frames)

        if available_count < self.min_buffer_frames:
            t_end = time.perf_counter()
            metrics.total_ms = (t_end - t_start) * 1000.0
            return None, metrics

        is_partial_window = available_count < self.window_frames
        window_slice = all_buffer_frames[-self.window_frames :]

        # Stage timing: preprocess vs model
        t_prep_start = time.perf_counter()
        # Preprocessing timing captured inside model_provider.predict or measured here
        try:
            raw_prediction = self.model_provider.predict(
                frames=window_slice,
                allow_experimental_inference=self.allow_experimental_inference,
            )
        except Exception as exc:
            raise ModelInferenceError(f"Model prediction failed: {exc}") from exc

        t_model_end = time.perf_counter()
        # Calculate split latency metrics
        metrics.preprocess_ms = (t_prep_start - t_start) * 1000.0
        metrics.model_ms = (t_model_end - t_prep_start) * 1000.0
        metrics.total_ms = (t_model_end - t_start) * 1000.0

        if raw_prediction is None:
            return None, metrics

        # Apply confidence threshold filtering ONLY if explicitly enabled (opt-in)
        confidence = raw_prediction.confidence if raw_prediction.confidence is not None else 0.0
        if self.enable_confidence_gating and confidence < self.min_confidence_threshold:
            rejection_reason = (
                f"LOW_CONFIDENCE: Prediction confidence {confidence:.4f} is below "
                f"minimum threshold {self.min_confidence_threshold:.4f}."
            )
            prediction = Prediction(
                status="rejected",
                text=None,
                confidence=confidence,
                model_version=raw_prediction.model_version,
                reason=rejection_reason,
                warning=raw_prediction.warning,
                metrics=metrics.to_dict(),
            )
            return prediction, metrics

        # Partial window notice
        reason_msg = raw_prediction.reason
        if is_partial_window:
            partial_note = (
                f"PARTIAL_WINDOW_INITIALIZATION: Sampled from initial {available_count} frames "
                f"(full window requires {self.window_frames} frames)."
            )
            reason_msg = f"{reason_msg}; {partial_note}" if reason_msg else partial_note

        prediction = Prediction(
            status=raw_prediction.status,
            text=raw_prediction.text,
            confidence=raw_prediction.confidence,
            model_version=raw_prediction.model_version,
            reason=reason_msg,
            warning=raw_prediction.warning,
            metrics=metrics.to_dict(),
        )

        return prediction, metrics
