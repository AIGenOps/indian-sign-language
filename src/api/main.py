from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import Settings
from src.models.sanket import EXPERIMENTAL_PREPROCESSING_WARNING, SanketModelProvider

app = FastAPI(title="ISL Translator API", version="0.1.0")
settings = Settings()


class PredictRequest(BaseModel):
    allow_experimental_inference: bool = False


class PredictResponse(BaseModel):
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    model_version: str
    warning: Optional[str] = None
    reason: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_provider": settings.model_provider,
        "model_checkpoint": settings.model_checkpoint,
    }


@app.get("/ready")
def ready():
    return {"status": "ready"}
