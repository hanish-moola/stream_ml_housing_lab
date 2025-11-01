"""Data access and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    memory_mb: float
    null_counts: dict


def load_raw_data(config: DataConfig, *, dtype=None) -> pd.DataFrame:
    """Load the raw housing dataset from disk."""

    data_path = config.raw_data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {data_path}")

    df = pd.read_csv(data_path, dtype=dtype)
    summary = summarise_dataframe(df)
    logger.info(
        "Loaded raw dataset with %s rows and %s columns from %s",
        summary.rows,
        summary.columns,
        data_path,
    )
    log_dataset_summary_to_mlflow(summary)
    return df


def split_features_target(
    df: pd.DataFrame,
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target column."""

    if config.target_column not in df.columns:
        raise KeyError(f"Target column '{config.target_column}' missing from dataset")

    y = df[config.target_column]
    X = df.drop(columns=[config.target_column])

    if config.index_column and config.index_column in X.columns:
        X = X.set_index(config.index_column)

    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y,
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets using config values."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    logger.info(
        "Created train/test split with train=%d rows, test=%d rows",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def summarise_dataframe(df: pd.DataFrame) -> DatasetSummary:
    """Generate summary statistics for a dataframe."""

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    null_counts = df.isnull().sum().to_dict()
    return DatasetSummary(
        rows=df.shape[0],
        columns=df.shape[1],
        memory_mb=round(memory_mb, 3),
        null_counts=null_counts,
    )


def log_dataset_summary_to_mlflow(summary: DatasetSummary) -> None:
    """Log dataset metadata to the active MLflow run, if available."""

    if mlflow.active_run() is None:
        return

    mlflow.log_params({
        "data_rows": summary.rows,
        "data_columns": summary.columns,
        "data_memory_mb": summary.memory_mb,
    })
    mlflow.log_dict(summary.null_counts, "dataset/null_counts.json")


def save_json_artifact(content: dict, path: Path) -> None:
    """Persist JSON content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)

    logger.debug("Saved artifact to %s", path)
