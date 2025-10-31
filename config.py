"""
Configuration file for housing price prediction project.
Centralizes all paths, hyperparameters, and experiment settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, MLFLOW_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_PATH = os.getenv("HOUSING_DATA_PATH", "/kaggle/input/housing-prices-dataset/Housing.csv")
TARGET_COLUMN = "price"

# Categorical columns to encode
CATEGORICAL_COLUMNS = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

# Numerical columns for visualization
NUMERICAL_COLUMNS = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Model configuration
RANDOM_STATE = 100
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # For train/val/test split if needed

# Hyperparameter tuning configuration
HYPERPARAMETER_CONFIG = {
    'max_trials': 10,
    'executions_per_trial': 3,
    'epochs': 10,
    'objective': 'val_loss',
    'directory': str(PROJECT_ROOT / 'hyperparameter_tuning'),
    'project_name': 'housing_price_prediction'
}

# Model architecture hyperparameter search space
HYPERPARAMETER_SEARCH_SPACE = {
    'units1': {'min_value': 32, 'max_value': 512, 'step': 32},
    'units2': {'min_value': 32, 'max_value': 512, 'step': 32},
    'activation1': {'values': ['relu', 'tanh', 'sigmoid']},
    'activation2': {'values': ['relu', 'tanh', 'sigmoid']},
    'learning_rate': {'values': [1e-2, 1e-3, 1e-4]}
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'verbose': 1
}

# MLflow configuration
MLFLOW_CONFIG = {
    'experiment_name': 'housing_price_prediction',
    'tracking_uri': f"file:{MLFLOW_DIR}",
    'run_name': None  # Will be set dynamically with timestamp
}

# Evaluation metrics to track
EVALUATION_METRICS = [
    'mean_absolute_error',
    'mean_squared_error',
    'root_mean_squared_error',
    'root_mean_squared_log_error',
    'r2_score'
]

