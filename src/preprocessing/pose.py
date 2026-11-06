"""Pose Feature Extractor Module (UNVERIFIED / EXPERIMENTAL)

IMPORTANT NOTICE:
This module extracts 195-dimensional MediaPipe landmark feature vectors per frame.
While the 195-dimensional feature width is CONFIRMED FROM WEIGHTS (via `pose_encoder.fc_in.weight` shape (768, 195)),
the exact MediaPipe landmark selection, ordering, and coordinate representation used in the original Sanket training
pipeline are UNKNOWN because no public upstream source code or paper exists for the model.

Therefore, this extractor is labeled UNVERIFIED/EXPERIMENTAL. It must NOT be used to claim reproduction of the
Sanket benchmark results until the upstream preprocessing pipeline is independently confirmed.

Plausible Candidate Landmark Compositions (all summing to 195 float32 features per frame):

1. Candidate A (Default / Leading Candidate): Upper-body pose + Both hands 3D coordinates
   - 23 upper-body pose landmarks (0 to 22) x 3 (x, y, z) = 69 floats
   - 21 left-hand landmarks x 3 (x, y, z)                = 63 floats
   - 21 right-hand landmarks x 3 (x, y, z)               = 63 floats
   - Total: 69 + 63 + 63 = 195 floats

2. Candidate B: Full 33 pose landmarks + Single-hand 3D + Pose visibility scores
   - 33 full-body pose landmarks x 3 (x, y, z)           = 99 floats
   - 21 single-hand landmarks x 3 (x, y, z)              = 63 floats
   - 33 pose landmark visibility scores x 1               = 33 floats
   - Total: 99 + 63 + 33 = 195 floats
   (Note: Asymmetric single-hand treatment is unlikely for two-handed sign languages like ISL).

3. Candidate C: Both hands 3D + Upper-body 2D pose + Pose visibility
   - 21 left-hand landmarks x 3 (x, y, z)                = 63 floats
   - 21 right-hand landmarks x 3 (x, y, z)               = 63 floats
   - 23 upper-body pose landmarks x 2 (x, y)             = 46 floats
   - 23 upper-body pose visibility scores x 1             = 23 floats
   - Total: 63 + 63 + 46 + 23 = 195 floats

4. Candidate D: Both hands 3D + Upper-body pose 3D + Face keypoints 3D
   - 21 left-hand landmarks x 3 (x, y, z)                = 63 floats
   - 21 right-hand landmarks x 3 (x, y, z)               = 63 floats
   - 15 core upper-body pose landmarks x 3 (x, y, z)     = 45 floats
   - 8 key facial landmarks (eyes, lips) x 3 (x, y, z)   = 24 floats
   - Total: 63 + 63 + 45 + 24 = 195 floats
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class PoseLandmarkScheme(str, Enum):
    CANDIDATE_A_UPPER_BODY_HANDS = "candidate_a_upper_body_hands"
    CANDIDATE_B_FULL_POSE_ONE_HAND = "candidate_b_full_pose_one_hand"
    CANDIDATE_C_HANDS_2DPOSE_VIS = "candidate_c_hands_2dpose_vis"
    CANDIDATE_D_HANDS_POSE_FACE = "candidate_d_hands_pose_face"


@dataclass(frozen=True)
class PoseExtractorConfig:
    scheme: PoseLandmarkScheme = PoseLandmarkScheme.CANDIDATE_A_UPPER_BODY_HANDS
    target_dim: int = 195
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseFeatureExtractor:
    """Config-driven MediaPipe pose feature extractor (UNVERIFIED / EXPERIMENTAL)."""

    def __init__(self, config: PoseExtractorConfig | None = None) -> None:
        self.config = config or PoseExtractorConfig()

    def extract_frame_features(self, holistic_results: Any = None) -> np.ndarray:
        """Extract a single-frame 195-dim feature vector from MediaPipe Holistic results."""
        feature_vector = np.zeros(self.config.target_dim, dtype=np.float32)

        if holistic_results is None:
            return feature_vector

        scheme = self.config.scheme

        if scheme == PoseLandmarkScheme.CANDIDATE_A_UPPER_BODY_HANDS:
            # 23 pose (69) + 21 left (63) + 21 right (63) = 195
            idx = 0
            if hasattr(holistic_results, "pose_landmarks") and holistic_results.pose_landmarks:
                for lm in holistic_results.pose_landmarks.landmark[:23]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 69

            if hasattr(holistic_results, "left_hand_landmarks") and holistic_results.left_hand_landmarks:
                for lm in holistic_results.left_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

            if hasattr(holistic_results, "right_hand_landmarks") and holistic_results.right_hand_landmarks:
                for lm in holistic_results.right_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

        elif scheme == PoseLandmarkScheme.CANDIDATE_B_FULL_POSE_ONE_HAND:
            # 33 pose (99) + 21 hand (63) + 33 pose vis (33) = 195
            idx = 0
            if hasattr(holistic_results, "pose_landmarks") and holistic_results.pose_landmarks:
                for lm in holistic_results.pose_landmarks.landmark[:33]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
                for lm in holistic_results.pose_landmarks.landmark[:33]:
                    feature_vector[idx] = getattr(lm, "visibility", 0.0)
                    idx += 1
            else:
                idx += 99 + 33

            hand_lms = getattr(holistic_results, "right_hand_landmarks", None) or getattr(
                holistic_results, "left_hand_landmarks", None
            )
            if hand_lms:
                for lm in hand_lms.landmark[:21]:
                    feature_vector[99 : 99 + 63] = [lm.x, lm.y, lm.z]

        elif scheme == PoseLandmarkScheme.CANDIDATE_C_HANDS_2DPOSE_VIS:
            # 21 left (63) + 21 right (63) + 23 pose 2D (46) + 23 pose vis (23) = 195
            idx = 0
            if hasattr(holistic_results, "left_hand_landmarks") and holistic_results.left_hand_landmarks:
                for lm in holistic_results.left_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

            if hasattr(holistic_results, "right_hand_landmarks") and holistic_results.right_hand_landmarks:
                for lm in holistic_results.right_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

            if hasattr(holistic_results, "pose_landmarks") and holistic_results.pose_landmarks:
                for lm in holistic_results.pose_landmarks.landmark[:23]:
                    feature_vector[idx : idx + 2] = [lm.x, lm.y]
                    idx += 2
                for lm in holistic_results.pose_landmarks.landmark[:23]:
                    feature_vector[idx] = getattr(lm, "visibility", 0.0)
                    idx += 1
            else:
                idx += 46 + 23

        else:  # CANDIDATE_D_HANDS_POSE_FACE
            # 21 left (63) + 21 right (63) + 15 pose (45) + 8 face (24) = 195
            idx = 0
            if hasattr(holistic_results, "left_hand_landmarks") and holistic_results.left_hand_landmarks:
                for lm in holistic_results.left_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

            if hasattr(holistic_results, "right_hand_landmarks") and holistic_results.right_hand_landmarks:
                for lm in holistic_results.right_hand_landmarks.landmark[:21]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 63

            if hasattr(holistic_results, "pose_landmarks") and holistic_results.pose_landmarks:
                for lm in holistic_results.pose_landmarks.landmark[:15]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 45

            if hasattr(holistic_results, "face_landmarks") and holistic_results.face_landmarks:
                for lm in holistic_results.face_landmarks.landmark[:8]:
                    feature_vector[idx : idx + 3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                idx += 24

        return feature_vector[: self.config.target_dim]

    def extract_from_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Extract features across a sequence of RGB video frames.

        Returns:
            np.ndarray of shape (num_frames, 195) and dtype float32.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return np.zeros((0, self.config.target_dim), dtype=np.float32)

        output = np.zeros((num_frames, self.config.target_dim), dtype=np.float32)

        if not HAS_MEDIAPIPE or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "holistic"):
            return output

        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        ) as holistic:
            for i, frame in enumerate(frames):
                results = holistic.process(frame)
                output[i] = self.extract_frame_features(results)

        return output
