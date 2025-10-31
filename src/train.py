"""
Training script for housing price prediction using an XGBoost regression pipeline
with MLflow tracking and shared feature engineering for offline/online parity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    EXCLUDE_COLUMNS,
    MLFLOW_CONFIG,
    MODEL_REGISTRY_NAME,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    XGBOOST_PARAMS,
)
from utils.data_loader import get_data_info, load_housing_data
from utils.feature_pipeline import (
    build_feature_transformer,
    get_feature_names,
    split_features_and_target,
)
from utils.metrics import regression_metrics
from utils.visualization import create_visualizations, plot_predictions_vs_actual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_mlflow(run_name: str | None = None) -> str:
    """Configure MLflow tracking and return the active run name."""
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    if run_name is None:
        run_name = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("MLflow experiment: %s", MLFLOW_CONFIG["experiment_name"])
    logger.info("MLflow run: %s", run_name)

    return run_name


def log_dataset_metadata(data_info: Dict[str, object]) -> None:
    """Record dataset metadata in MLflow for traceability."""
    mlflow.log_param("data_shape_rows", data_info["shape"][0])
    mlflow.log_param("data_shape_cols", data_info["shape"][1])
    mlflow.log_param("memory_usage_mb", round(data_info["memory_usage_mb"], 2))
    mlflow.log_param("feature_columns", ",".join(map(str, data_info["columns"])))


def log_model_artifacts(
    pipeline: Pipeline,
    raw_df: pd.DataFrame,
    encoded_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    feature_names: List[str],
    run_name: str,
) -> None:
    """Generate diagnostic artefacts and store them locally and in MLflow."""
    viz_dir = RESULTS_DIR / "visualizations" / run_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    create_visualizations(raw_df, encoded_df, viz_dir)

    preds_path = viz_dir / "predictions_vs_actual.png"
    plot_predictions_vs_actual(y_test.values, y_pred, preds_path)
    mlflow.log_artifact(str(preds_path))

    booster = pipeline.named_steps["regressor"]
    if hasattr(booster, "feature_importances_"):
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": booster.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_path = viz_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))


def train_model() -> None:
    """Execute the complete training pipeline."""
    run_name = setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        logger.info("Starting training pipeline")

        # Load data
        df = load_housing_data(DATA_PATH)
        data_info = get_data_info(df)
        log_dataset_metadata(data_info)

        # Prepare features
        X, y, numerical_cols, categorical_cols = split_features_and_target(
            df,
            target_column=TARGET_COLUMN,
            exclude_columns=EXCLUDE_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
        )

        if dropped := [col for col in EXCLUDE_COLUMNS if col in df.columns]:
            logger.info("Dropped high-cardinality columns: %s", dropped)

        logger.info(
            "Using %d numerical and %d categorical features",
            len(numerical_cols),
            len(categorical_cols),
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)

        # Build modelling pipeline
        feature_transformer = build_feature_transformer(
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols,
        )
        regressor = XGBRegressor(**XGBOOST_PARAMS)
        pipeline = Pipeline(
            steps=[
                ("features", feature_transformer),
                ("regressor", regressor),
            ]
        )

        logger.info("Training XGBoost pipeline with params: %s", XGBOOST_PARAMS)
        mlflow.log_params({f"model__{key}": value for key, value in XGBOOST_PARAMS.items()})
        pipeline.fit(X_train, y_train)

        # Evaluate on hold-out set
        y_pred = pipeline.predict(X_test)
        metrics = regression_metrics(y_test.values, y_pred)
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, float(metric_value))

        logger.info("Evaluation metrics: %s", metrics)

        # Visual artefacts & diagnostics
        feature_names = get_feature_names(
            pipeline.named_steps["features"], numerical_cols, categorical_cols
        )
        transformed_full = pipeline.named_steps["features"].transform(X)
        encoded_df = pd.DataFrame(transformed_full, columns=feature_names)
        log_model_artifacts(pipeline, df, encoded_df, y_test, y_pred, feature_names, run_name)

        # Persist artefacts locally and to MLflow
        model_dir = RESULTS_DIR / "models" / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = model_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(str(pipeline_path))

        feature_names_path = model_dir / "feature_names.txt"
        feature_names_path.write_text("\n".join(feature_names), encoding="utf-8")
        mlflow.log_artifact(str(feature_names_path))

        input_example = X_test.iloc[:1]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
            input_example=input_example,
        )

        metadata = {
            "run_name": run_name,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "exclude_columns": [col for col in EXCLUDE_COLUMNS if col in df.columns],
            "metrics": {k: float(v) for k, v in metrics.items() if v is not None},
        }
        metadata_path = model_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        mlflow.log_artifact(str(metadata_path))

        logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    train_model()
