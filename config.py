"""Configuration for the Stream-ML-Housing-Lab project."""

from __future__ import annotations

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

for directory in [DATA_DIR, RESULTS_DIR, MLFLOW_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_PATH = os.getenv("HOUSING_DATA_PATH", DATA_DIR / "Housing.csv")
TARGET_COLUMN = "price"

CATEGORICAL_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

# Columns prone to high cardinality that we drop from the baseline pipeline
EXCLUDE_COLUMNS = ["Address"]

# Experiment settings
RANDOM_STATE = 100
TEST_SIZE = 0.2

# Baseline XGBoost configuration for the regression task
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

MODEL_REGISTRY_NAME = "housing_price_predictor"

# MLflow configuration
MLFLOW_CONFIG = {
    "experiment_name": "housing_price_prediction",
    "tracking_uri": f"file:{MLFLOW_DIR}",
}
