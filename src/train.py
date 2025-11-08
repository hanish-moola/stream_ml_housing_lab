"""Training pipeline for the housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
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
from .mlflow_utils import ensure_run
from .registry import (
    build_run_name,
    get_latest_run_by_stage,
    download_artifact,
)

logger = get_logger(__name__)


def _resolve_feature_artifacts(config: ProjectConfig) -> Dict[str, Path]:
    """Ensure a fitted feature transformer exists and return downloaded artifacts."""

    feature_run = get_latest_run_by_stage(config.mlflow.experiment_name, "feature_engineering")
    if feature_run is None:
        logger.info("No feature-engineering run found; triggering one now")
        run_feature_engineering(config)
        feature_run = get_latest_run_by_stage(config.mlflow.experiment_name, "feature_engineering")
        if feature_run is None:
            raise RuntimeError("Feature engineering did not produce an MLflow run")

    run_id = feature_run.info.run_id
    transformer_path = download_artifact(run_id, "transformer/transformer.joblib")
    metadata_path = download_artifact(run_id, "feature_metadata.json")
    stats_path = download_artifact(run_id, "training_stats.json")

    return {
        "run_id": run_id,
        "transformer": transformer_path,
        "metadata": metadata_path,
        "stats": stats_path,
    }


def _build_model(model_type: str, hyperparameters: Dict[str, Any], input_dim: Optional[int] = None) -> Any:
    if model_type == "linear_regression":
        return LinearRegression(**hyperparameters)
    if model_type == "neural_network":
        if input_dim is None:
            raise ValueError("input_dim must be provided for neural_network model")
        return _build_keras_regressor(input_dim, hyperparameters)
    raise ValueError(f"Unsupported model type: {model_type}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_keras_regressor(input_dim: int, hyperparameters: Dict[str, Any]):
    from tensorflow import keras
    from tensorflow.keras import layers

    hidden_units = hyperparameters.get("hidden_units", [128, 64])
    activation = hyperparameters.get("activation", "relu")
    dropout = float(hyperparameters.get("dropout", 0.0))
    learning_rate = float(hyperparameters.get("learning_rate", 0.001))

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(int(units), activation=activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
    return model


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

    feature_artifacts = _resolve_feature_artifacts(config)
    with feature_artifacts["metadata"].open("r", encoding="utf-8") as handle:
        feature_metadata = FeatureMetadata(**json.load(handle))

    with ensure_run(effective_run_name) as run:
        logger.info("Starting training run: %s using model type %s", run.info.run_id, config.model.type)
        mlflow.set_tag("stage", "training")

        df = load_raw_data(config.data)
        X, y = split_features_target(df, config.data)
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, config.data)

        transformer = joblib.load(feature_artifacts["transformer"])

        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        model_type = config.model.type
        hyperparameters = config.model.hyperparameters
        mlflow.log_param("model_type", model_type)

        if model_type == "neural_network":
            X_train_transformed = X_train_transformed.astype("float32")
            X_test_transformed = X_test_transformed.astype("float32")
            model = _build_model(model_type, hyperparameters, input_dim=X_train_transformed.shape[1])
            epochs = int(hyperparameters.get("epochs", 200))
            batch_size = int(hyperparameters.get("batch_size", 32))
            validation_split = float(hyperparameters.get("validation_split", 0.1))
            logger.info("Training neural network: %d epochs, batch_size=%d", epochs, batch_size)
            history = model.fit(
                X_train_transformed,
                y_train.to_numpy().astype("float32"),
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,  # Show progress bar
            )
            predictions = model.predict(X_test_transformed, verbose=0).flatten()
            if history.history.get("loss"):
                mlflow.log_metric("train_loss_final", float(history.history["loss"][-1]))
            if history.history.get("mae"):
                mlflow.log_metric("train_mae_final", float(history.history["mae"][-1]))
        else:
            model = _build_model(model_type, hyperparameters)
            model.fit(X_train_transformed, y_train)
            predictions = model.predict(X_test_transformed)

        metrics = _compute_metrics(y_test.to_numpy(), predictions)
        _log_training_summary(config, metrics, effective_run_name)
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(feature_metadata.to_dict(), "feature_metadata.json")

        results_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": predictions,
            "residual": y_test - predictions,
        })
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_path = Path(tmp_dir) / "predictions.csv"
            results_df.to_csv(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

        mlflow.log_artifact(str(feature_artifacts["transformer"]), artifact_path="transformer")

        if model_type == "neural_network":
            mlflow.keras.log_model(model, artifact_path="model")
        else:
            inference_pipeline = Pipeline(
                steps=[
                    ("preprocess", transformer),
                    ("model", model),
                ]
            )
            mlflow.sklearn.log_model(inference_pipeline, artifact_path="model")

        run_metadata = {
            "run_name": effective_run_name,
            "mlflow_run_id": run.info.run_id,
            "model_type": model_type,
            "feature_run_id": feature_artifacts["run_id"],
            "metrics": metrics,
        }
        mlflow.log_dict(run_metadata, "run_metadata.json")

        logger.info("Training complete; model logged to MLflow")


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
