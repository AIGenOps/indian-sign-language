"""Video Preprocessing & Temporal Frame Sampling Module (EXPERIMENTAL)

Implements video frame sampling and ImageNet normalization for the experimental ISL translation pipeline.

Named Constants:
- IMAGENET_MEAN: Standard ImageNet RGB channel means [0.485, 0.456, 0.406].
- IMAGENET_STD: Standard ImageNet RGB channel standard deviations [0.229, 0.224, 0.225].
"""

from __future__ import annotations

from typing import Sequence
import cv2
import numpy as np

# Standard ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_uniform_frames(frames: Sequence[np.ndarray], num_samples: int = 8) -> list[np.ndarray]:
    """Sample exactly `num_samples` (default 8) frames uniformly across the input clip's duration.

    TEMPORAL SAMPLING SELECTION:
    - DEFAULT CHOICE: Uniform sampling of 8 frames across the full video clip duration using
      `np.linspace(0, total_frames - 1, num_samples, dtype=int)`. This ensures key gesture phases
      (beginning, peak, end) are captured regardless of clip duration.
    - ALTERNATIVES CONSIDERED:
      1. First 8 consecutive frames: Truncates longer sign video clips and ignores mid/ending sign gestures.
      2. Fixed-FPS sampling (e.g. 15 FPS): Produces variable sequence lengths across videos, requiring
         dynamic padding/masking that conflicts with Sanket's fixed 8-frame temporal embedding tensor
         `video_encoder.embeddings.time_embeddings` (shape (1, 8, 768)).
    """
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Cannot sample frames from an empty sequence.")

    if total_frames <= num_samples:
        # If video has fewer or equal frames than required, repeat last frame as padding
        indices = list(range(total_frames))
        while len(indices) < num_samples:
            indices.append(total_frames - 1)
    else:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

    return [frames[idx] for idx in indices]


def preprocess_video_frames(
    frames: Sequence[np.ndarray],
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Preprocess a list of BGR/RGB image frames into a normalized PyTorch-compatible float32 array.

    Args:
        frames: Sequence of uint8 numpy arrays representing RGB frames.
        target_size: Target spatial dimensions (width, height), default (224, 224).

    Returns:
        np.ndarray of shape (8, 3, 224, 224) and dtype float32, normalized using IMAGENET_MEAN and IMAGENET_STD.
    """
    sampled = sample_uniform_frames(frames, num_samples=8)
    processed = []

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    for frame in sampled:
        # Resize frame if needed
        if (frame.shape[1], frame.shape[0]) != target_size:
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        # Convert to float32 in range [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet mean and std
        normalized = (normalized - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        chw = np.transpose(normalized, (2, 0, 1))
        processed.append(chw)

    return np.stack(processed, axis=0)  # Shape: (8, 3, 224, 224)
