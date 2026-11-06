from pathlib import Path
import pytest
from src.models.sanket import SanketModelProvider, EXPERIMENTAL_PREPROCESSING_WARNING


def test_missing_checkpoint_is_explicit(tmp_path: Path):
    provider = SanketModelProvider(str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError):
        provider.load()


def test_predict_rejects_none_or_empty_frames(tmp_path: Path):
    checkpoint_file = tmp_path / "dummy.pth"
    checkpoint_file.touch()

    provider = SanketModelProvider(str(checkpoint_file))
    provider.loaded = True  # Mock loaded state

    # Test None input raises ValueError (both default and experimental paths)
    with pytest.raises(ValueError, match="Input 'frames' cannot be None or empty"):
        provider.predict(None)

    with pytest.raises(ValueError, match="Input 'frames' cannot be None or empty"):
        provider.predict(None, allow_experimental_inference=True)

    # Test empty sequence raises ValueError
    with pytest.raises(ValueError, match="Input 'frames' cannot be None or empty"):
        provider.predict([])

    with pytest.raises(ValueError, match="Input 'frames' cannot be None or empty"):
        provider.predict([], allow_experimental_inference=True)


def test_predict_default_path_raises_not_implemented(tmp_path: Path):
    checkpoint_file = tmp_path / "dummy.pth"
    checkpoint_file.touch()

    provider = SanketModelProvider(str(checkpoint_file))
    provider.loaded = True  # Mock loaded state

    # Valid non-empty input but without opt-in flag
    with pytest.raises(NotImplementedError, match="Pin the upstream Sanket implementation"):
        provider.predict([1, 2, 3])
