from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "sanket")
    model_checkpoint: str = os.getenv(
        "MODEL_CHECKPOINT", "artifacts/models/model_best.pth"
    )
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    target_fps: int = int(os.getenv("TARGET_FPS", "15"))
    min_translation_confidence: float = float(
        os.getenv("MIN_TRANSLATION_CONFIDENCE", "0.70")
    )
