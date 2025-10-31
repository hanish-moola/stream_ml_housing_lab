"""Offline evaluation for the XGBoost housing price predictor."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    EXCLUDE_COLUMNS,
    MLFLOW_CONFIG,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from utils.data_loader import load_housing_data
from utils.feature_pipeline import split_features_and_target
from utils.metrics import regression_metrics
from utils.pipeline_loader import load_pipeline
from utils.visualization import plot_predictions_vs_actual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(pipeline, df: pd.DataFrame, run_name: str) -> None:
    """Evaluate the pipeline on the configured test split and log artefacts."""
    X, y, _, _ = split_features_and_target(
        df,
        target_column=TARGET_COLUMN,
        exclude_columns=EXCLUDE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    mlflow.log_param("evaluation_rows", len(X_test))

    y_pred = pipeline.predict(X_test)
    metrics = regression_metrics(y_test.values, y_pred)

    for name, value in metrics.items():
        if value is not None:
            mlflow.log_metric(name, float(value))

    eval_dir = RESULTS_DIR / "evaluation" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_path = eval_dir / "predictions_vs_actual.png"
    plot_predictions_vs_actual(y_test.values, y_pred, plot_path)
    mlflow.log_artifact(str(plot_path))

    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items() if v is not None}, f, indent=2)
    mlflow.log_artifact(str(metrics_path))

    logger.info("Evaluation metrics: %s", metrics)


def main(
    model_path: Optional[str] = None,
    model_uri: Optional[str] = None,
    data_path: Optional[str] = None,
) -> None:
    data_path = data_path or DATA_PATH
    df = load_housing_data(data_path)

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    run_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        pipeline, resolved_path, resolved_uri = load_pipeline(
            model_path=Path(model_path) if model_path else None,
            model_uri=model_uri,
        )
        if resolved_path:
            mlflow.log_param("model_path", str(resolved_path))
        if resolved_uri:
            mlflow.log_param("model_uri", resolved_uri)

        evaluate(pipeline, df, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained housing price model")
    parser.add_argument("--model-path", type=str, help="Local directory containing pipeline.joblib")
    parser.add_argument("--model-uri", type=str, help="MLflow model URI to load")
    parser.add_argument("--data-path", type=str, help="Optional override for the dataset path")

    args = parser.parse_args()
    main(args.model_path, args.model_uri, args.data_path)
