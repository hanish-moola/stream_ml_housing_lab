"""Helpers for locating and loading trained pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import mlflow.sklearn

from config import MLFLOW_CONFIG, RESULTS_DIR


def _latest_local_pipeline() -> Path:
    candidates = sorted(
        (RESULTS_DIR / "models").glob("*/pipeline.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No trained pipeline found. Please run train.py first.")
    return candidates[0]


def load_pipeline(
    model_path: Optional[Path] = None,
    model_uri: Optional[str] = None,
) -> Tuple[object, Optional[Path], Optional[str]]:
    """Load a trained pipeline from disk or MLflow.

    Returns the pipeline along with its resolved local path (if any) and model URI (if any).
    """
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    if model_uri:
        pipeline = mlflow.sklearn.load_model(model_uri)
        return pipeline, None, model_uri

    if model_path:
        pipeline_file = Path(model_path) / "pipeline.joblib"
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline not found at {pipeline_file}")
        return joblib.load(pipeline_file), pipeline_file.parent, None

    latest = _latest_local_pipeline()
    return joblib.load(latest), latest.parent, None
