"""Prediction CLI for the housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

from .config import ProjectConfig, load_config
from .logging_utils import configure_logging, get_logger
from .mlflow_utils import ensure_run
from .registry import (
    build_run_name,
    download_artifact,
    get_latest_run_by_stage,
)

logger = get_logger(__name__)


def _get_training_run(run_id: Optional[str], experiment_name: str) -> mlflow.entities.Run:
    client = MlflowClient()
    if run_id:
        return client.get_run(run_id)
    run = get_latest_run_by_stage(experiment_name, "training")
    if run is None:
        raise FileNotFoundError("No training run found. Train a model first.")
    return run


def _load_prediction_pipeline(run: mlflow.entities.Run):
    model_type = run.data.params.get("model_type", "linear_regression")
    model_uri = f"runs:/{run.info.run_id}/model"

    if model_type == "neural_network":
        transformer_path = download_artifact(run.info.run_id, "transformer/transformer.joblib")
        transformer = joblib.load(transformer_path)
        keras_model = mlflow.keras.load_model(model_uri)

        def predict_fn(features: pd.DataFrame) -> float:
            transformed = transformer.transform(features).astype("float32")
            return float(keras_model.predict(transformed, verbose=0).flatten()[0])

        feature_metadata_path = download_artifact(run.info.run_id, "feature_metadata.json")
        with feature_metadata_path.open("r", encoding="utf-8") as handle:
            feature_metadata = json.load(handle)
    else:
        pipeline = mlflow.sklearn.load_model(model_uri)

        def predict_fn(features: pd.DataFrame) -> float:
            return float(pipeline.predict(features)[0])

        feature_metadata_path = download_artifact(run.info.run_id, "feature_metadata.json")
        with feature_metadata_path.open("r", encoding="utf-8") as handle:
            feature_metadata = json.load(handle)

    return predict_fn, feature_metadata, model_type


def _prepare_feature_dataframe(features: Dict[str, object], feature_metadata: Dict[str, List[str]]) -> pd.DataFrame:
    expected = feature_metadata.get("numeric", []) + feature_metadata.get("categorical", [])
    missing = [col for col in expected if col not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    ordered = {col: features[col] for col in expected}
    return pd.DataFrame([ordered])


def run_prediction(
    config: ProjectConfig,
    input_features: Dict[str, object],
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> float:
    configure_logging()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    training_run = _get_training_run(run_id, config.mlflow.experiment_name)
    predict_fn, feature_metadata, model_type = _load_prediction_pipeline(training_run)

    features_df = _prepare_feature_dataframe(input_features, feature_metadata)
    prediction = predict_fn(features_df)

    effective_run_name = run_name or build_run_name("prediction_{timestamp}")
    with ensure_run(effective_run_name) as run:
        mlflow.log_params({f"feature_{k}": v for k, v in input_features.items()})
        mlflow.log_param("source_model_run_id", training_run.info.run_id)
        mlflow.log_param("source_model_type", model_type)
        mlflow.log_metric("prediction", prediction)
        mlflow.set_tag("prediction", True)
        logger.info(
            "Prediction run %s | model_run=%s | model_type=%s | prediction=%.2f",
            run.info.run_id,
            training_run.info.run_id,
            model_type,
            prediction,
        )

    return prediction


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a prediction from the housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--model-run", type=str, help="Optional training run ID to use for predictions")
    parser.add_argument("--features", type=Path, required=True, help="Path to JSON file containing feature values")
    parser.add_argument("--run-name", type=str, help="Optional explicit prediction run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)

    with cli_args.features.open("r", encoding="utf-8") as handle:
        features = json.load(handle)

    prediction = run_prediction(
        config,
        input_features=features,
        run_id=cli_args.model_run,
        run_name=cli_args.run_name,
    )

    print(json.dumps({"prediction": prediction}))


if __name__ == "__main__":  # pragma: no cover
    main()
