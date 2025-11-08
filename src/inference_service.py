"""Utilities for loading the latest trained model and serving predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, Tuple, Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .feature_engineering import FeatureMetadata, TrainingStats
from .logging_utils import get_logger
from .registry import download_artifact, get_latest_run_by_stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    run_id: str
    model_type: str
    predict_fn: Callable[[pd.DataFrame], float]
    feature_metadata: FeatureMetadata
    training_stats: TrainingStats


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    used_features: Dict[str, Any]
    imputed: Dict[str, Any]
    model_run_id: str
    model_type: str


class InferenceService:
    """Loads the most recent MLflow training run and performs inference."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._bundle: ModelBundle | None = None
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    def predict(self, payload: Dict[str, Any], *, force_refresh: bool = False) -> PredictionResult:
        """Run inference against the latest model, imputing missing features."""

        run = self._get_latest_training_run()
        bundle = self._ensure_bundle(run, force_refresh=force_refresh)

        features_df, imputed_values, ordered_features = _prepare_feature_frame(
            payload,
            bundle.feature_metadata,
            bundle.training_stats,
        )
        value = float(bundle.predict_fn(features_df))

        return PredictionResult(
            prediction=value,
            used_features=ordered_features,
            imputed=imputed_values,
            model_run_id=bundle.run_id,
            model_type=bundle.model_type,
        )

    def cached_run_id(self) -> str | None:
        """Return the cached run id, if artifacts have been loaded."""

        if self._bundle:
            return self._bundle.run_id
        return None

    def _get_latest_training_run(self) -> mlflow.entities.Run:
        run = get_latest_run_by_stage(self.config.mlflow.experiment_name, "training")
        if run is None:
            raise FileNotFoundError("No training runs tagged with stage 'training' were found.")
        return run

    def _ensure_bundle(
        self,
        run: mlflow.entities.Run,
        *,
        force_refresh: bool = False,
    ) -> ModelBundle:
        run_id = run.info.run_id
        if not force_refresh and self._bundle and self._bundle.run_id == run_id:
            return self._bundle

        with self._lock:
            if not force_refresh and self._bundle and self._bundle.run_id == run_id:
                return self._bundle

            logger.info("Loading artifacts for training run %s", run_id)
            self._bundle = self._load_bundle(run)
            return self._bundle

    def _load_bundle(self, run: mlflow.entities.Run) -> ModelBundle:
        run_id = run.info.run_id
        model_type = run.data.params.get("model_type", self.config.model.type)
        feature_metadata = self._load_feature_metadata(run_id)
        training_stats = self._load_training_stats(run_id)
        predict_fn = self._build_predict_fn(run_id, model_type)

        return ModelBundle(
            run_id=run_id,
            model_type=model_type,
            predict_fn=predict_fn,
            feature_metadata=feature_metadata,
            training_stats=training_stats,
        )

    def _load_feature_metadata(self, run_id: str) -> FeatureMetadata:
        path = download_artifact(run_id, "feature_metadata.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return FeatureMetadata(**payload)

    def _load_training_stats(self, run_id: str) -> TrainingStats:
        path = download_artifact(run_id, "training_stats.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return TrainingStats(**payload)

    def _build_predict_fn(self, run_id: str, model_type: str) -> Callable[[pd.DataFrame], float]:
        model_uri = f"runs:/{run_id}/model"

        if model_type == "neural_network":
            transformer_path = download_artifact(run_id, "transformer/transformer.joblib")
            transformer = joblib.load(transformer_path)
            keras_model = mlflow.keras.load_model(model_uri)

            def predict_fn(features: pd.DataFrame) -> float:
                transformed = transformer.transform(features).astype("float32")
                result = keras_model.predict(transformed, verbose=0)
                return float(np.asarray(result).reshape(-1)[0])

        else:
            pipeline = mlflow.pyfunc.load_model(model_uri)

            def predict_fn(features: pd.DataFrame) -> float:
                result = pipeline.predict(features)
                return float(np.asarray(result).reshape(-1)[0])

        return predict_fn


def _prepare_feature_frame(
    payload: Dict[str, Any],
    metadata: FeatureMetadata,
    stats: TrainingStats,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    expected_features = metadata.numeric + metadata.categorical
    ordered_features: Dict[str, Any] = {}
    imputed: Dict[str, Any] = {}

    for feature in expected_features:
        value = payload.get(feature)
        if _is_missing(value):
            value = _impute_value(feature, metadata, stats)
            imputed[feature] = value
        ordered_features[feature] = value

    frame = pd.DataFrame([ordered_features])
    return frame, imputed, ordered_features


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _impute_value(feature: str, metadata: FeatureMetadata, stats: TrainingStats) -> Any:
    if feature in stats.numeric_means:
        return stats.numeric_means[feature]

    if feature in stats.categorical_levels:
        levels = stats.categorical_levels.get(feature, {})
        if not levels:
            raise ValueError(f"No categorical statistics available to impute '{feature}'.")
        # Deterministically prefer the most frequent category, breaking ties lexicographically
        return max(levels.items(), key=lambda item: (item[1], item[0]))[0]

    # Fallback: determine if feature was expected but training stats are missing
    if feature in metadata.numeric or feature in metadata.categorical:
        raise ValueError(f"No statistics available to impute expected feature '{feature}'.")

    raise ValueError(f"Unexpected feature '{feature}' cannot be imputed.")
