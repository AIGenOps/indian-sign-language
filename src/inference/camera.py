"""Camera Source Abstraction Layer for Streaming Inference.

Provides an abstract CameraSource interface with:
1. OpenCVCameraSource: Real hardware camera/webcam stream (cv2.VideoCapture).
2. FileOrGeneratorCameraSource: Mock / file camera source for testing without camera hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np


class CameraSource(ABC):
    """Abstract interface for video frame acquisition sources."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the video source."""
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame from the camera source.

        Returns:
            (success: bool, frame: np.ndarray | None)
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Disconnect and release resources."""
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""
        raise NotImplementedError


class OpenCVCameraSource(CameraSource):
    """Real camera stream source backed by OpenCV VideoCapture."""

    def __init__(self, camera_index_or_url: int | str = 0) -> None:
        self.camera_index_or_url = camera_index_or_url
        self.cap: cv2.VideoCapture | None = None
        self._connected = False

    def connect(self) -> bool:
        if self._connected and self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_index_or_url)
        self._connected = self.cap.isOpened()
        return self._connected

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if not self._connected or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self.cap is not None and self.cap.isOpened()


class FileOrGeneratorCameraSource(CameraSource):
    """Mock / file camera source for testing and file-based stream playback without hardware."""

    def __init__(
        self,
        source: Sequence[np.ndarray] | str | Path,
        loop: bool = False,
    ) -> None:
        self.source = source
        self.loop = loop
        self._connected = False
        self._frames: list[np.ndarray] = []
        self._cursor = 0
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> bool:
        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            if not path.exists():
                return False
            self.cap = cv2.VideoCapture(str(path))
            self._connected = self.cap.isOpened()
            return self._connected
        else:
            self._frames = list(self.source)
            self._cursor = 0
            self._connected = True
            return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if not self._connected:
            return False, None

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return False, None
                    return True, frame
                return False, None
            return True, frame

        # Sequence of numpy arrays
        if self._cursor >= len(self._frames):
            if self.loop and len(self._frames) > 0:
                self._cursor = 0
            else:
                return False, None

        frame = self._frames[self._cursor]
        self._cursor += 1
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._connected = False
        self._cursor = 0

    def is_connected(self) -> bool:
        return self._connected
