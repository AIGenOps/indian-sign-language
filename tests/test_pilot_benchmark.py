"""Unit tests verifying that benchmark/pilot scripts refuse synthetic fallbacks and error loudly when real videos are missing."""

import pytest
from pathlib import Path
from src.dataset.exceptions import MissingVideoDataError
from scripts.run_pilot_benchmark import run_pilot


def test_pilot_benchmark_refuses_missing_video_data(tmp_path: Path):
    """Assert run_pilot raises MissingVideoDataError when real video files are absent."""
    empty_video_dir = tmp_path / "empty_videos"
    empty_video_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(MissingVideoDataError, match=r"real video file\(s\) are missing"):

        run_pilot(num_clips=5, video_dir=empty_video_dir)
