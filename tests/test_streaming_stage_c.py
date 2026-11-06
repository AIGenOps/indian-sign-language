from pathlib import Path
import numpy as np
import pytest
from src.inference import (
    BoundedFrameBuffer,
    CameraConnectionError,
    CameraSource,
    FileOrGeneratorCameraSource,
    StreamingInferenceError,
    StreamingInferenceRunner,
)
from src.models.sanket import SanketModelProvider


class FailingCameraSource(CameraSource):
    """Mock camera that fails and cannot reconnect."""

    def __init__(self) -> None:
        self._connected = True

    def connect(self) -> bool:
        return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def release(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


class RecoveringCameraSource(CameraSource):
    """Mock camera that drops on step 2, but succeeds when reconnected."""

    def __init__(self) -> None:
        self._connected = True
        self.step_count = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        self.step_count += 1
        if self.step_count == 2 and self._connected:
            self._connected = False
            return False, None
        return True, np.ones((50, 50, 3), dtype=np.uint8)

    def release(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


def test_reconnect_failure_raises_typed_camera_connection_error():
    failing_source = FailingCameraSource()
    runner = StreamingInferenceRunner(
        camera_source=failing_source,
        max_reconnect_attempts=2,
        reconnect_delay_sec=0.01,
    )

    with pytest.raises(CameraConnectionError, match="Failed to reconnect to camera source"):
        runner.step()


def test_reconnect_success_recovers_stream():
    recovering_source = RecoveringCameraSource()
    runner = StreamingInferenceRunner(
        camera_source=recovering_source,
        max_reconnect_attempts=3,
        reconnect_delay_sec=0.01,
    )

    # Step 1: Normal frame read
    success1, _ = runner.step()
    assert success1 is True

    # Step 2: Stream drops and auto-reconnects
    success2, _ = runner.step()
    assert success2 is True


def test_latency_metrics_and_confidence_score(tmp_path: Path):
    checkpoint_file = Path("artifacts/models/model_best.pth")
    if not checkpoint_file.exists():
        pytest.skip(f"Checkpoint missing at {checkpoint_file}. Run download_sanket.py first.")

    model_provider = SanketModelProvider(str(checkpoint_file))
    model_provider.load()

    synthetic_frames = [
        np.full((100, 100, 3), fill_value=val, dtype=np.uint8) for val in range(20)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=60)

    runner = StreamingInferenceRunner(
        camera_source=camera_source,
        buffer=buffer,
        model_provider=model_provider,
        stride_frames=10,
        min_buffer_frames=8,
        allow_experimental_inference=True,
    )

    assert camera_source.connect() is True

    for _ in range(9):
        runner.step()

    success, prediction = runner.step()
    assert success is True
    assert prediction is not None

    # Verify real computed confidence score in (0.0, 1.0]
    assert prediction.confidence is not None
    assert 0.0 < prediction.confidence <= 1.0

    # Check latency metrics structure
    assert prediction.metrics is not None
    assert "capture_ms" in prediction.metrics
    assert "preprocess_ms" in prediction.metrics
    assert "model_ms" in prediction.metrics
    assert "total_ms" in prediction.metrics
    assert prediction.metrics["total_ms"] >= 0.0

    camera_source.release()


def test_opt_in_confidence_threshold_rejection(tmp_path: Path):
    checkpoint_file = Path("artifacts/models/model_best.pth")
    if not checkpoint_file.exists():
        pytest.skip(f"Checkpoint missing at {checkpoint_file}. Run download_sanket.py first.")

    model_provider = SanketModelProvider(str(checkpoint_file))
    model_provider.load()

    synthetic_frames = [
        np.full((100, 100, 3), fill_value=val, dtype=np.uint8) for val in range(20)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=60)

    # Opt-in to confidence gating with high threshold (1.5) to force rejection
    runner = StreamingInferenceRunner(
        camera_source=camera_source,
        buffer=buffer,
        model_provider=model_provider,
        stride_frames=10,
        min_buffer_frames=8,
        min_confidence_threshold=1.5,
        enable_confidence_gating=True,  # Explicit opt-in
        allow_experimental_inference=True,
    )

    camera_source.connect()
    for _ in range(9):
        runner.step()

    success, prediction = runner.step()
    assert success is True
    assert prediction is not None
    assert prediction.status == "rejected"
    assert prediction.text is None
    assert prediction.reason is not None
    assert "LOW_CONFIDENCE" in prediction.reason

    camera_source.release()
