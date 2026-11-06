"""End-to-End Experimental Inference Pipeline Smoke Test.

PLUMBING TEST NOTICE:
This test verifies the software plumbing and tensor dimensional mechanics of the experimental
Sanket inference pipeline (video frames -> uniform 8-frame sampling + ImageNet normalization ->
Timesformer vision encoder + PoseFeatureExtractor MediaPipe pose stream -> Gated Multimodal Fusion ->
DimMapper -> T5 text decoder generation).

It is STRICTLY A PLUMBING / INTEGRATION TEST. It asserts only that the forward pass executes without
crashing and produces a non-empty string output with the UNVERIFIED warning attached.
It does NOT assert correctness or translation quality of the output text.
"""

from pathlib import Path
import numpy as np
import pytest
from src.models.sanket import EXPERIMENTAL_PREPROCESSING_WARNING, SanketModelProvider


def test_experimental_pipeline_smoke_test(tmp_path: Path):
    model_checkpoint = Path("artifacts/models/model_best.pth")
    if not model_checkpoint.exists():
        pytest.skip(f"Model checkpoint not found at {model_checkpoint}. Run download_sanket.py first.")

    provider = SanketModelProvider(str(model_checkpoint))
    provider.load()
    assert provider.loaded is True

    # Synthetic RGB video input (10 frames, 224x224x3 uint8 image arrays)
    synthetic_video_frames = [np.ones((224, 224, 3), dtype=np.uint8) * 128 for _ in range(10)]

    # Execute experimental opt-in inference
    prediction = provider.predict(synthetic_video_frames, allow_experimental_inference=True)

    # Assert plumbing mechanics
    assert prediction.status == "success"
    assert isinstance(prediction.text, str)
    assert len(prediction.text.strip()) > 0
    assert prediction.warning == EXPERIMENTAL_PREPROCESSING_WARNING


def test_default_path_still_refuses_unverified_inference(tmp_path: Path):
    model_checkpoint = Path("artifacts/models/model_best.pth")
    if not model_checkpoint.exists():
        pytest.skip(f"Model checkpoint not found at {model_checkpoint}. Run download_sanket.py first.")

    provider = SanketModelProvider(str(model_checkpoint))
    provider.load()

    synthetic_video_frames = [np.ones((224, 224, 3), dtype=np.uint8) * 128 for _ in range(10)]

    # Default path (allow_experimental_inference=False / omitted) MUST raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Pin the upstream Sanket implementation"):
        provider.predict(synthetic_video_frames)
