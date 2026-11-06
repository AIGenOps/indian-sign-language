"""Bounded Frame Buffer Module for Live Stream Processing.

Implements a thread-safe, fixed-capacity frame buffer sitting between the camera source
and the sign-recognition preprocessing pipeline.

Drop Policy Rationale:
- DEFAULT POLICY: DROP_OLDEST (FIFO eviction).
- RATIONALE: In live sign language translation, real-time gesture latency is paramount.
  When the frame capture rate outpaces model inference, dropping the oldest (stale) frames
  ensures the buffer retains the signer's current active gestures while bounds-constraining memory.
- ALTERNATIVES:
  1. DROP_NEWEST: Discards current live gestures in favor of older buffered frames (causes severe lag).
  2. BLOCK: Stalls the video capture thread, causing frame drops at the OS/hardware camera layer.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import threading
import numpy as np


class DropPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class BoundedFrameBuffer:
    """Thread-safe bounded buffer for video frame storage and drop policy enforcement."""

    def __init__(
        self,
        max_size: int = 120,
        drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer.")

        self.max_size = max_size
        self.drop_policy = drop_policy
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self.dropped_frame_count = 0

    def push(self, frame: np.ndarray) -> bool:
        """Push a new frame into the buffer.

        Returns:
            bool: True if frame was added, False if frame was dropped due to policy.
        """
        if frame is None:
            return False

        with self._lock:
            if len(self._buffer) >= self.max_size:
                self.dropped_frame_count += 1
                if self.drop_policy == DropPolicy.DROP_OLDEST:
                    self._buffer.popleft()  # Evict oldest frame
                    self._buffer.append(frame)
                    return True
                elif self.drop_policy == DropPolicy.DROP_NEWEST:
                    # Drop incoming new frame
                    return False
                elif self.drop_policy == DropPolicy.BLOCK:
                    # Reject frame when full
                    return False

            self._buffer.append(frame)
            return True

    def get_frames(self) -> list[np.ndarray]:
        """Return a snapshot list of current frames in the buffer without removing them."""
        with self._lock:
            return list(self._buffer)

    def pop_all(self) -> list[np.ndarray]:
        """Drain and return all frames currently in the buffer."""
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
            return frames

    def size(self) -> int:
        """Return current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self.max_size

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
            self.dropped_frame_count = 0
