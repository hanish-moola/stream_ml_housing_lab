"""FastAPI service for real-time housing price prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException

from config import CATEGORICAL_COLUMNS, EXCLUDE_COLUMNS, TARGET_COLUMN
from utils.pipeline_loader import load_pipeline
from .schemas import HousingFeatures, PredictionResponse

app = FastAPI(
    title="Stream ML Housing Predictor",
    description="Real-time inference service backed by the XGBoost pipeline",
    version="0.1.0",
)


class ModelState:
    pipeline = None
    feature_order: List[str] = []
    categorical_columns: List[str] = []
    boolean_categorical: List[str] = []
    metadata: Dict[str, object] = {}
    model_version: Optional[str] = None
    model_path: Optional[Path] = None
    model_uri: Optional[str] = None


state = ModelState()


def _resolve_feature_order(metadata: Dict[str, object], pipeline) -> List[str]:
    if metadata:
        numerical = metadata.get("numerical_columns", [])
        categorical = metadata.get("categorical_columns", [])
        if numerical or categorical:
            return [*numerical, *categorical]

    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    raise RuntimeError("Unable to determine feature order for the loaded pipeline")


def _load_metadata(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def initialise_model() -> None:
    model_path_env = os.getenv("HOUSING_MODEL_PATH")
    model_uri_env = os.getenv("HOUSING_MODEL_URI")

    pipeline, resolved_path, resolved_uri = load_pipeline(
        model_path=Path(model_path_env) if model_path_env else None,
        model_uri=model_uri_env,
    )

    metadata = _load_metadata(resolved_path)
    feature_order = _resolve_feature_order(metadata, pipeline)

    categorical = metadata.get("categorical_columns") or CATEGORICAL_COLUMNS
    boolean_categorical = [col for col in categorical if col != "furnishingstatus"]

    state.pipeline = pipeline
    state.feature_order = feature_order
    state.categorical_columns = categorical
    state.boolean_categorical = boolean_categorical
    state.metadata = metadata
    state.model_path = resolved_path
    state.model_uri = resolved_uri
    state.model_version = (
        metadata.get("run_name")
        or (resolved_path.name if resolved_path else None)
        or resolved_uri
    )


@app.on_event("startup")
async def startup_event() -> None:
    initialise_model()


@app.get("/health")
def health_check() -> Dict[str, object]:
    return {
        "status": "ok" if state.pipeline is not None else "loading",
        "model_version": state.model_version,
        "feature_count": len(state.feature_order),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures) -> PredictionResponse:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        payload_df = features.to_dataframe(state.feature_order, state.boolean_categorical)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing feature: {exc.args[0]}") from exc

    try:
        prediction = float(state.pipeline.predict(payload_df)[0])
    except Exception as exc:  # pragma: no cover - FastAPI handles logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(estimated_price=prediction, model_version=state.model_version)


@app.get("/model-info")
def model_info() -> Dict[str, object]:
    return {
        "model_version": state.model_version,
        "categorical_columns": state.categorical_columns,
        "boolean_categorical": state.boolean_categorical,
        "feature_order": state.feature_order,
        "model_path": str(state.model_path) if state.model_path else None,
        "model_uri": state.model_uri,
        "exclude_columns": EXCLUDE_COLUMNS,
        "target": TARGET_COLUMN,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
