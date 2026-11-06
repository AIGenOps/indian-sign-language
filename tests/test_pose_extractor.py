import numpy as np
import pytest
from src.preprocessing.pose import (
    PoseExtractorConfig,
    PoseFeatureExtractor,
    PoseLandmarkScheme,
)


@pytest.mark.parametrize(
    "scheme",
    [
        PoseLandmarkScheme.CANDIDATE_A_UPPER_BODY_HANDS,
        PoseLandmarkScheme.CANDIDATE_B_FULL_POSE_ONE_HAND,
        PoseLandmarkScheme.CANDIDATE_C_HANDS_2DPOSE_VIS,
        PoseLandmarkScheme.CANDIDATE_D_HANDS_POSE_FACE,
    ],
)
def test_pose_extractor_shape_and_dtype(scheme: PoseLandmarkScheme):
    config = PoseExtractorConfig(scheme=scheme, target_dim=195)
    extractor = PoseFeatureExtractor(config=config)

    # Synthetic blank RGB frames (8 frames, 224x224x3)
    dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]
    features = extractor.extract_from_frames(dummy_frames)

    assert isinstance(features, np.ndarray)
    assert features.shape == (8, 195)
    assert features.dtype == np.float32


def test_pose_extractor_empty_frames():
    extractor = PoseFeatureExtractor()
    features = extractor.extract_from_frames([])
    assert features.shape == (0, 195)
    assert features.dtype == np.float32
