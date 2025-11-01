"""Evaluation pipeline for the trained housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import tempfile
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

from .config import ProjectConfig, load_config
from .data import load_raw_data, split_features_target, stratified_train_test_split
from .logging_utils import configure_logging, get_logger
from .mlflow_utils import ensure_run
from .registry import (
    build_run_name,
    download_artifact,
    get_latest_run_by_stage,
)
from .train import _compute_metrics

logger = get_logger(__name__)


def _get_run(run_id: Optional[str], experiment_name: str) -> mlflow.entities.Run:
    client = MlflowClient()
    if run_id:
        return client.get_run(run_id)
    run = get_latest_run_by_stage(experiment_name, "training")
    if run is None:
        raise FileNotFoundError("No training run found. Train a model first.")
    return run


def _load_model_and_transformer(run: mlflow.entities.Run):
    model_type = run.data.params.get("model_type", "linear_regression")
    run_uri = f"runs:/{run.info.run_id}/model"

    if model_type == "neural_network":
        keras_model = mlflow.keras.load_model(run_uri)
        transformer_path = download_artifact(run.info.run_id, "transformer/transformer.joblib")
        transformer = joblib.load(transformer_path)

        def predict_fn(X: pd.DataFrame) -> pd.Series:
            features = transformer.transform(X).astype("float32")
            preds = keras_model.predict(features, verbose=0).flatten()
            return preds

    else:
        pipeline = mlflow.sklearn.load_model(run_uri)

        def predict_fn(X: pd.DataFrame) -> pd.Series:
            return pipeline.predict(X)

    return predict_fn, model_type


def run_evaluation(
    config: ProjectConfig,
    model_run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    configure_logging()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    base_run_name = run_name or build_run_name(config.mlflow.run_name_template)
    effective_run_name = f"evaluation_{base_run_name}" if not run_name else run_name

    target_run = _get_run(model_run_id, config.mlflow.experiment_name)
    predict_fn, model_type = _load_model_and_transformer(target_run)

    with ensure_run(effective_run_name) as run:
        mlflow.set_tag("stage", "evaluation")
        mlflow.log_param("evaluated_model_run_id", target_run.info.run_id)
        mlflow.log_param("evaluated_model_type", model_type)

        df = load_raw_data(config.data)
        X, y = split_features_target(df, config.data)
        _, X_test, _, y_test = stratified_train_test_split(X, y, config.data)

        predictions = predict_fn(X_test)
        metrics = _compute_metrics(y_test.to_numpy(), predictions)
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        mlflow.log_dict(metrics, "evaluation_metrics.json")

        results_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": predictions,
            "residual": y_test - predictions,
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "evaluation_predictions.csv"
            results_df.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="evaluation")

        metadata = {
            "evaluation_run": effective_run_name,
            "mlflow_run_id": run.info.run_id,
            "evaluated_model_run_id": target_run.info.run_id,
            "metrics": metrics,
        }
        mlflow.log_dict(metadata, "evaluation_metadata.json")

        logger.info("Evaluation complete for model run %s", target_run.info.run_id)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest trained housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--model-run", type=str, help="Optional training run ID to evaluate")
    parser.add_argument("--run-name", type=str, help="Optional explicit evaluation run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    run_evaluation(config, model_run_id=cli_args.model_run, run_name=cli_args.run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
