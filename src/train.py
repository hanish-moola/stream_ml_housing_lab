"""Training pipeline for the housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import ProjectConfig, load_config
from .data import load_raw_data, split_features_target, stratified_train_test_split
from .feature_engineering import FeatureMetadata, run_feature_engineering
from .logging_utils import configure_logging, get_logger
from .registry import (
    build_run_name,
    load_transformer,
    prepare_run_artifacts,
    resolve_latest_run,
    save_model,
    save_transformer,
    write_metadata,
)

logger = get_logger(__name__)


def _resolve_feature_artifacts(config: ProjectConfig) -> Dict[str, Path]:
    """Ensure a fitted feature transformer exists and return its artifact paths."""

    latest_run = resolve_latest_run(config.artifacts)
    if latest_run is None or not latest_run.transformer_path.exists():
        logger.info("No feature artifacts detected. Triggering feature engineering run.")
        run_feature_engineering(config)
        latest_run = resolve_latest_run(config.artifacts)
        if latest_run is None:
            raise RuntimeError("Feature engineering did not produce artifacts")

    metadata_path = latest_run.run_dir / "feature_metadata.json"
    stats_path = latest_run.run_dir / "training_stats.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected feature metadata at {metadata_path}")

    return {
        "transformer": latest_run.transformer_path,
        "metadata": metadata_path,
        "stats": stats_path,
        "run_dir": latest_run.run_dir,
    }


def _build_model(model_type: str, hyperparameters: Dict[str, Any]) -> Any:
    if model_type == "linear_regression":
        return LinearRegression(**hyperparameters)
    raise ValueError(f"Unsupported model type: {model_type}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _log_training_summary(config: ProjectConfig, metrics: Dict[str, float], run_name: str) -> None:
    mlflow.log_params(
        {
            "model_type": config.model.type,
            "test_size": config.data.test_size,
            "random_state": config.data.random_state,
        }
    )
    for key, value in config.model.hyperparameters.items():
        mlflow.log_param(f"model_{key}", value)
    mlflow.log_metrics(metrics)
    mlflow.set_tag("run_name", run_name)


def run_training(config: ProjectConfig, run_name: Optional[str] = None) -> None:
    configure_logging()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    effective_run_name = run_name or build_run_name(config.mlflow.run_name_template)
    artifacts = prepare_run_artifacts(config.artifacts, effective_run_name)

    feature_artifacts = _resolve_feature_artifacts(config)
    with feature_artifacts["metadata"].open("r", encoding="utf-8") as handle:
        feature_metadata = FeatureMetadata(**json.load(handle))

    with mlflow.start_run(run_name=effective_run_name) as run:
        logger.info("Starting training run: %s", run.info.run_id)

        df = load_raw_data(config.data)
        X, y = split_features_target(df, config.data)
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, config.data)

        transformer = load_transformer(feature_artifacts["transformer"])
        save_transformer(transformer, artifacts.transformer_path)

        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        model = _build_model(config.model.type, config.model.hyperparameters)
        model.fit(X_train_transformed, y_train)
        predictions = model.predict(X_test_transformed)

        metrics = _compute_metrics(y_test.to_numpy(), predictions)
        _log_training_summary(config, metrics, effective_run_name)

        metrics_path = artifacts.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        mlflow.log_artifact(str(metrics_path))

        results_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": predictions,
            "residual": y_test - predictions,
        })
        predictions_path = metrics_path.parent / "predictions.csv"
        results_df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(str(predictions_path))

        inference_pipeline = Pipeline(
            steps=[
                ("preprocess", transformer),
                ("model", model),
            ]
        )
        save_model(inference_pipeline, artifacts.model_path)
        mlflow.sklearn.log_model(inference_pipeline, artifact_path="model")

        run_metadata = {
            "run_name": effective_run_name,
            "mlflow_run_id": run.info.run_id,
            "feature_artifacts": {
                "source_run_dir": str(feature_artifacts["run_dir"]),
                "transformer": str(feature_artifacts["transformer"]),
                "metadata": str(feature_artifacts["metadata"]),
            },
            "artifacts": {
                "model": str(artifacts.model_path),
                "transformer": str(artifacts.transformer_path),
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
            },
            "metrics": metrics,
        }
        write_metadata(run_metadata, artifacts.metadata_path)

        logger.info("Training complete. Model stored at %s", artifacts.model_path)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--run-name", type=str, help="Optional explicit run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    run_training(config, run_name=cli_args.run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
