import numpy as np
import pytest
from src.inference import (
    BoundedFrameBuffer,
    DropPolicy,
    FileOrGeneratorCameraSource,
)


def test_buffer_drop_oldest_policy_under_overflow():
    """Verify BoundedFrameBuffer drops oldest frames when max_size is exceeded."""
    buffer = BoundedFrameBuffer(max_size=5, drop_policy=DropPolicy.DROP_OLDEST)

    # Push 10 frames with distinct pixel values (1 through 10)
    for i in range(1, 11):
        frame = np.full((10, 10, 3), fill_value=i, dtype=np.uint8)
        buffer.push(frame)

    assert buffer.size() == 5
    assert buffer.is_full() is True
    assert buffer.dropped_frame_count == 5

    frames = buffer.get_frames()

    # Verify that oldest frames (1..5) were dropped and buffer contains newest frames (6..10)
    retained_values = [int(f[0, 0, 0]) for f in frames]
    assert retained_values == [6, 7, 8, 9, 10]


def test_buffer_drop_newest_and_block_policies():
    """Verify DROP_NEWEST and BLOCK policies under buffer overflow."""
    buffer_newest = BoundedFrameBuffer(max_size=3, drop_policy=DropPolicy.DROP_NEWEST)
    for i in range(1, 4):
        assert buffer_newest.push(np.full((5, 5, 3), i, dtype=np.uint8)) is True

    # 4th frame should be dropped
    assert buffer_newest.push(np.full((5, 5, 3), 99, dtype=np.uint8)) is False
    retained_newest = [int(f[0, 0, 0]) for f in buffer_newest.get_frames()]
    assert retained_newest == [1, 2, 3]


def test_mock_camera_source_feeding_buffer():
    """Verify FileOrGeneratorCameraSource can feed BoundedFrameBuffer without hardware."""
    # Generate 15 synthetic frames
    synthetic_frames = [
        np.full((100, 100, 3), fill_value=val, dtype=np.uint8) for val in range(15)
    ]
    camera_source = FileOrGeneratorCameraSource(source=synthetic_frames)
    buffer = BoundedFrameBuffer(max_size=10, drop_policy=DropPolicy.DROP_OLDEST)

    assert camera_source.connect() is True
    assert camera_source.is_connected() is True

    pushed_count = 0
    while True:
        success, frame = camera_source.read_frame()
        if not success or frame is None:
            break
        buffer.push(frame)
        pushed_count += 1

    assert pushed_count == 15
    assert buffer.size() == 10
    assert buffer.dropped_frame_count == 5

    # Check retained frames (5 to 14)
    retained_values = [int(f[0, 0, 0]) for f in buffer.get_frames()]
    assert retained_values == list(range(5, 15))

    camera_source.release()
    assert camera_source.is_connected() is False
