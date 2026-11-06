from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.inference import (
    BoundedFrameBuffer,
    FileOrGeneratorCameraSource,
    StreamingInferenceRunner,
)
from src.models.sanket import EXPERIMENTAL_PREPROCESSING_WARNING, SanketModelProvider


def test_streaming_pipeline_trigger_and_prediction(tmp_path: Path):
    checkpoint_file = Path("artifacts/models/model_best.pth")
    if not checkpoint_file.exists():
        pytest.skip(f"Checkpoint missing at {checkpoint_file}. Run download_sanket.py first.")

    model_provider = SanketModelProvider(str(checkpoint_file))
    model_provider.load()

    # Generate 30 synthetic RGB frames
    synthetic_frames = [
        np.full((100, 100, 3), fill_value=i * 5, dtype=np.uint8) for i in range(30)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=60)

    runner = StreamingInferenceRunner(
        camera_source=camera_source,
        buffer=buffer,
        model_provider=model_provider,
        stride_frames=15,
        window_frames=30,
        min_buffer_frames=8,
        allow_experimental_inference=True,
    )

    assert camera_source.connect() is True

    predictions_received = []

    for step_num in range(1, 31):
        success, prediction = runner.step()
        assert success is True

        if step_num % 15 == 0:
            assert prediction is not None
            assert prediction.status == "success"
            assert isinstance(prediction.text, str)
            assert prediction.warning == EXPERIMENTAL_PREPROCESSING_WARNING
            predictions_received.append(prediction)
        else:
            assert prediction is None

    assert len(predictions_received) == 2
    camera_source.release()


def test_early_stream_partial_window_flagging(tmp_path: Path):
    """Verify that early stream triggering (< window_frames) flags PARTIAL_WINDOW_INITIALIZATION."""
    checkpoint_file = Path("artifacts/models/model_best.pth")
    if not checkpoint_file.exists():
        pytest.skip(f"Checkpoint missing at {checkpoint_file}. Run download_sanket.py first.")

    model_provider = SanketModelProvider(str(checkpoint_file))
    model_provider.load()

    # Generate 12 synthetic frames (between min_buffer_frames=8 and window_frames=30)
    synthetic_frames = [
        np.full((100, 100, 3), fill_value=val, dtype=np.uint8) for val in range(12)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=60)

    runner = StreamingInferenceRunner(
        camera_source=camera_source,
        buffer=buffer,
        model_provider=model_provider,
        stride_frames=10,
        window_frames=30,
        min_buffer_frames=8,
        allow_experimental_inference=True,
    )

    assert camera_source.connect() is True

    # Step 9 frames (non-trigger)
    for _ in range(9):
        runner.step()

    # 10th frame triggers prediction at stride_frames=10
    success, prediction = runner.step()
    assert success is True
    assert prediction is not None
    assert prediction.reason is not None
    assert "PARTIAL_WINDOW_INITIALIZATION" in prediction.reason
    assert "10 frames" in prediction.reason

    camera_source.release()


def test_fixed_sampling_window_constant_past_window_frames(tmp_path: Path):
    """Verify sampling window stays constant (window_frames=30) even as buffer grows to 60 frames."""
    checkpoint_file = Path("artifacts/models/model_best.pth")
    if not checkpoint_file.exists():
        pytest.skip(f"Checkpoint missing at {checkpoint_file}. Run download_sanket.py first.")

    model_provider = SanketModelProvider(str(checkpoint_file))
    model_provider.load()

    # Spy on predict method to inspect passed frames slice
    original_predict = model_provider.predict
    captured_frame_slices = []

    def spy_predict(frames, **kwargs):
        captured_frame_slices.append(list(frames))
        return original_predict(frames, **kwargs)

    model_provider.predict = spy_predict

    # 60 synthetic frames with distinct values (0 to 59)
    synthetic_frames = [
        np.full((50, 50, 3), fill_value=i, dtype=np.uint8) for i in range(60)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=120)

    runner = StreamingInferenceRunner(
        camera_source=camera_source,
        buffer=buffer,
        model_provider=model_provider,
        stride_frames=30,
        window_frames=30,
        min_buffer_frames=8,
        allow_experimental_inference=True,
    )

    assert camera_source.connect() is True

    for _ in range(60):
        runner.step()

    assert len(captured_frame_slices) == 2
    # Verify both captured slices have length exactly 30 (window_frames)
    assert len(captured_frame_slices[0]) == 30
    assert len(captured_frame_slices[1]) == 30

    # Second prediction slice (frames 30 to 59) should contain pixel values 30..59
    second_slice_vals = [int(f[0, 0, 0]) for f in captured_frame_slices[1]]
    assert second_slice_vals == list(range(30, 60))

    camera_source.release()
