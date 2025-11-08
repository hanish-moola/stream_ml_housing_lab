"""FastAPI layer for serving predictions from the latest trained model."""

from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import ProjectConfig, load_config
from ..inference_service import InferenceService
from ..logging_utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_CONFIG: ProjectConfig = load_config()
_SERVICE = InferenceService(_CONFIG)

app = FastAPI(
    title="Stream ML Housing Lab API",
    version="0.1.0",
    description="Serves predictions from the latest offline-trained model using FastAPI.",
)


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Partial or complete feature payload for inference.")
    run_id: str | None = Field(
        default=None,
        description="Optional MLflow run id to pin predictions to a specific model.",
    )
    refresh: bool = Field(
        default=False,
        description="Force refresh of the cached model artifacts before prediction.",
    )


class PredictionResponse(BaseModel):
    model_run_id: str
    model_type: str
    prediction: float
    imputed: Dict[str, Any]
    used_features: Dict[str, Any]


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """Lightweight health endpoint."""

    return {"status": "ok", "model_run_id": _SERVICE.cached_run_id() or "unloaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a prediction, imputing missing features from training statistics."""

    try:
        result = _SERVICE.predict(
            request.features,
            run_id=request.run_id,
            force_refresh=request.refresh,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("FastAPI prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictionResponse(
        model_run_id=result.model_run_id,
        model_type=result.model_type,
        prediction=result.prediction,
        imputed=result.imputed,
        used_features=result.used_features,
    )


def main() -> None:
    """Entrypoint for `poetry run serve-api`."""

    host = os.getenv("HOUSING_API_HOST", "0.0.0.0")
    port = int(os.getenv("HOUSING_API_PORT", "8000"))
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
