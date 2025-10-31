"""
FastAPI app to serve offline-trained housing price prediction model.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# Project config
from config import RESULTS_DIR, TARGET_COLUMN, CATEGORICAL_COLUMNS


class PredictRequest(BaseModel):
    area: float
    bedrooms: int
    bathrooms: float
    stories: int
    parking: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str

    @field_validator(
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        mode="before",
    )
    @classmethod
    def normalize_yes_no(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"y", "yes", "true", "1"}:  # normalize to yes/no
                return "yes"
            if val in {"n", "no", "false", "0"}:
                return "no"
            return val
        return v


class PredictResponse(BaseModel):
    estimated_price: float
    model_path: Optional[str] = None


def _find_latest_model_dir() -> Path:
    models_root = RESULTS_DIR / "models"
    if not models_root.exists():
        raise FileNotFoundError("No models directory found under results/models")
    candidates = sorted(
        models_root.glob("*/model.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No saved model found (expected results/models/<run>/model.keras)")
    return candidates[0].parent


def _load_artifacts(model_dir: Path) -> Dict[str, Any]:
    model_file = model_dir / "model.keras"
    scaler_file = model_dir / "scaler.pkl"
    feature_names_file = model_dir / "feature_names.txt"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = tf.keras.models.load_model(str(model_file))

    scaler = None
    if scaler_file.exists():
        import pickle

        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

    feature_names: Optional[List[str]] = None
    if feature_names_file.exists():
        with open(feature_names_file, "r") as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]

    return {"model": model, "scaler": scaler, "feature_names": feature_names}


def _build_feature_frame(payload: PredictRequest, feature_names: List[str]) -> pd.DataFrame:
    # Convert request to DataFrame
    df = pd.DataFrame([payload.dict()])
    # One-hot encode categorical columns to mirror training
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    # Align columns to training order
    encoded_aligned = encoded.reindex(columns=feature_names, fill_value=0)
    return encoded_aligned


def _transform_features(X: pd.DataFrame, scaler) -> np.ndarray:
    if scaler is None:
        return X.values
    return scaler.transform(X)


app = FastAPI(title="Housing Price Prediction API", version="0.1.0")


# Load model at startup
_MODEL_DIR: Optional[Path] = None
_MODEL = None
_SCALER = None
_FEATURE_NAMES: Optional[List[str]] = None


@app.on_event("startup")
def load_model_startup():
    global _MODEL_DIR, _MODEL, _SCALER, _FEATURE_NAMES
    model_path_env = os.getenv("HOUSING_MODEL_PATH")
    model_dir = Path(model_path_env) if model_path_env else _find_latest_model_dir()
    artifacts = _load_artifacts(model_dir)

    _MODEL_DIR = model_dir
    _MODEL = artifacts["model"]
    _SCALER = artifacts.get("scaler")
    _FEATURE_NAMES = artifacts.get("feature_names")

    if not _FEATURE_NAMES:
        raise RuntimeError("Feature names not found; required to align inference features with training.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "model_dir": str(_MODEL_DIR) if _MODEL_DIR else None,
    }


@app.get("/model-info")
def model_info():
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_dir": str(_MODEL_DIR),
        "num_features": len(_FEATURE_NAMES) if _FEATURE_NAMES else 0,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target": TARGET_COLUMN,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _MODEL is None or _FEATURE_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        X_frame = _build_feature_frame(req, _FEATURE_NAMES)
        X_scaled = _transform_features(X_frame, _SCALER)
        y_pred = _MODEL.predict(X_scaled, verbose=0).ravel()[0]
        return PredictResponse(estimated_price=float(y_pred), model_path=str(_MODEL_DIR))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

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
