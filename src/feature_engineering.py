"""Feature engineering pipeline builder for the housing price project."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ProjectConfig, load_config
from .data import load_raw_data, split_features_target, stratified_train_test_split
from .logging_utils import configure_logging, get_logger
from .mlflow_utils import ensure_run
from .registry import (
    build_run_name,
    prepare_run_artifacts,
    save_transformer,
    write_metadata,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureMetadata:
    numeric: List[str]
    categorical: List[str]
    feature_names: List[str]

    def to_dict(self) -> dict:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "feature_names": self.feature_names,
        }


@dataclass(frozen=True)
class TrainingStats:
    numeric_means: dict
    numeric_stds: dict
    categorical_levels: dict

    def to_dict(self) -> dict:
        return {
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "categorical_levels": self.categorical_levels,
        }


def infer_feature_types(X: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove boolean columns that we treat as categorical (True/False) if encoded as bool
    boolean_as_numeric = [col for col in numeric_cols if X[col].dtype == bool]
    for col in boolean_as_numeric:
        numeric_cols.remove(col)
        categorical_cols.append(col)

    return numeric_cols, categorical_cols


def create_preprocessing_pipeline(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())]) if numeric_columns else "passthrough"
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    ) if categorical_columns else "drop"

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("No features available after type inference")

    return ColumnTransformer(transformers=transformers)


def collect_training_stats(
    X: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> TrainingStats:
    numeric_means = X[numeric_columns].mean(numeric_only=True).fillna(0).to_dict() if numeric_columns else {}
    numeric_stds = X[numeric_columns].std(numeric_only=True).fillna(0).to_dict() if numeric_columns else {}

    categorical_levels = {}
    for col in categorical_columns:
        counts = X[col].value_counts(dropna=False).to_dict()
        categorical_levels[col] = {str(key): int(value) for key, value in counts.items()}

    return TrainingStats(
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_levels=categorical_levels,
    )


def log_metadata_to_mlflow(feature_metadata: FeatureMetadata, stats: TrainingStats) -> None:
    if mlflow.active_run() is None:
        return

    mlflow.log_params(
        {
            "num_numeric_features": len(feature_metadata.numeric),
            "num_categorical_features": len(feature_metadata.categorical),
            "total_transformed_features": len(feature_metadata.feature_names),
        }
    )
    mlflow.log_dict(feature_metadata.to_dict(), "feature_engineering/feature_metadata.json")
    mlflow.log_dict(stats.to_dict(), "feature_engineering/training_stats.json")


def _fit_transformer_and_metadata(
    config: ProjectConfig,
    run_name: str,
) -> tuple[ColumnTransformer, FeatureMetadata, TrainingStats]:
    df = load_raw_data(config.data)
    X, _ = split_features_target(df, config.data)
    X_train, _, _, _ = stratified_train_test_split(X, df[config.data.target_column], config.data)

    numeric_cols, categorical_cols = infer_feature_types(X_train)
    transformer = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    transformer.fit(X_train)

    transformed_feature_names = transformer.get_feature_names_out() if hasattr(transformer, "get_feature_names_out") else []
    metadata = FeatureMetadata(
        numeric=numeric_cols,
        categorical=categorical_cols,
        feature_names=transformed_feature_names.tolist() if isinstance(transformed_feature_names, np.ndarray) else list(transformed_feature_names),
    )
    stats = collect_training_stats(X_train, numeric_cols, categorical_cols)

    return transformer, metadata, stats


def run_feature_engineering(config: ProjectConfig, run_name: Optional[str] = None) -> None:
    configure_logging()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    effective_run_name = run_name or build_run_name(config.mlflow.run_name_template)
    artifacts = prepare_run_artifacts(config.artifacts, effective_run_name)

    with ensure_run(effective_run_name) as run:
        logger.info("Starting feature engineering run: %s", run.info.run_id)

        transformer, feature_metadata, stats = _fit_transformer_and_metadata(config, effective_run_name)

        save_transformer(transformer, artifacts.transformer_path)

        metadata_path = artifacts.run_dir / "feature_metadata.json"
        stats_path = artifacts.run_dir / "training_stats.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(feature_metadata.to_dict(), handle, indent=2)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats.to_dict(), handle, indent=2)

        run_metadata = {
            "run_name": effective_run_name,
            "mlflow_run_id": run.info.run_id,
            "artifacts": {
                "transformer": str(artifacts.transformer_path),
                "feature_metadata": str(metadata_path),
                "training_stats": str(stats_path),
            },
        }
        write_metadata(run_metadata, artifacts.metadata_path)
        log_metadata_to_mlflow(feature_metadata, stats)

        logger.info("Feature engineering complete. Artifacts saved under %s", artifacts.run_dir)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature engineering for the housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--run-name", type=str, help="Optional explicit run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    run_feature_engineering(config, run_name=cli_args.run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
