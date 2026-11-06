from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Prediction:
    status: str
    text: str | None
    confidence: float | None
    model_version: str
    reason: str | None = None
    warning: str | None = None
    metrics: dict[str, float] | None = None

class ModelProvider(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frames: Any) -> Prediction:
        raise NotImplementedError
