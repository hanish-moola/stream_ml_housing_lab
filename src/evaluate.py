"""
Evaluation script for housing price prediction with MLflow integration.
"""

import os
import sys
from pathlib import Path
import pickle
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score
)
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    RESULTS_DIR,
    EVALUATION_METRICS
)
from utils.data_loader import load_housing_data
from utils.preprocessing import preprocess_data, scale_features
from utils.visualization import plot_predictions_vs_actual

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metric names and values
    """
    # Ensure predictions are positive for RMSLE (log requires positive values)
    y_pred_positive = np.maximum(y_pred, 1e-8)
    y_true_positive = np.maximum(y_true, 1e-8)
    
    metrics = {
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'root_mean_squared_error': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    # RMSLE calculation (handling negative values)
    try:
        metrics['root_mean_squared_log_error'] = np.sqrt(
            mean_squared_log_error(y_true_positive, y_pred_positive)
        )
    except Exception as e:
        logger.warning(f"Could not calculate RMSLE: {e}")
        metrics['root_mean_squared_log_error'] = None
    
    return metrics


def load_model_and_artifacts(model_uri: str = None, model_path: Path = None) -> dict:
    """
    Load model and associated artifacts (scaler, feature names).
    
    Args:
        model_uri: MLflow model URI
        model_path: Local path to model directory
        
    Returns:
        Dictionary containing model, scaler, and feature_names
    """
    if model_uri:
        logger.info(f"Loading model from MLflow URI: {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        
        # Try to load artifacts from MLflow run
        artifacts = {}
        try:
            # This would require the run_id, so for now we'll use local path
            pass
        except:
            pass
    elif model_path:
        logger.info(f"Loading model from local path: {model_path}")
        model_file = model_path / "model.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = tf.keras.models.load_model(str(model_file))
        
        # Load scaler
        scaler_file = model_path / "scaler.pkl"
        scaler = None
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Loaded scaler from artifacts")
        
        # Load feature names
        feature_names_file = model_path / "feature_names.txt"
        feature_names = None
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        artifacts = {
            'scaler': scaler,
            'feature_names': feature_names
        }
    else:
        raise ValueError("Either model_uri or model_path must be provided")
    
    return {
        'model': model,
        **artifacts
    }


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: pd.Series,
    run_name: str = None
):
    """
    Evaluate model and log metrics to MLflow.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test target
        run_name: Optional run name for MLflow
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Fix bug: Remove boolean conversion (this was in the original notebook)
    # The original had: y_pred = (y_pred > 0.5) which is wrong for regression
    # We'll keep predictions as continuous values
    y_pred = y_pred.flatten()
    y_true = y_test.values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    logger.info("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Log to MLflow
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name or "evaluation_run")
    
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            mlflow.log_metric(metric_name, metric_value)
    
    # Create and save predictions plot
    if run_name:
        plot_dir = RESULTS_DIR / "evaluation" / run_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_predictions_vs_actual(
            y_true, y_pred,
            save_path=plot_dir / "predictions_vs_actual.png"
        )
        mlflow.log_artifact(str(plot_dir / "predictions_vs_actual.png"))
    
    return metrics, y_pred


def main(
    model_path: str = None,
    model_uri: str = None,
    data_path: str = None
):
    """
    Main evaluation pipeline.
    
    Args:
        model_path: Path to local model directory
        model_uri: MLflow model URI
        data_path: Optional custom data path
    """
    data_path = data_path or DATA_PATH
    
    # Setup MLflow
    mlflow.set_experiment("housing_price_prediction")
    
    with mlflow.start_run(run_name="evaluation_run"):
        logger.info("Starting evaluation pipeline...")
        
        # Load model and artifacts
        if model_path:
            artifacts = load_model_and_artifacts(model_path=Path(model_path))
        elif model_uri:
            artifacts = load_model_and_artifacts(model_uri=model_uri)
        else:
            # Try to find latest model in results directory
            model_dirs = sorted(
                (RESULTS_DIR / "models").glob("*/model.keras"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if not model_dirs:
                raise FileNotFoundError(
                    "No model found. Please provide model_path or train a model first."
                )
            model_path = model_dirs[0].parent
            logger.info(f"Using latest model from: {model_path}")
            artifacts = load_model_and_artifacts(model_path=model_path)
        
        model = artifacts['model']
        scaler = artifacts.get('scaler')
        feature_names = artifacts.get('feature_names')
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = load_housing_data(data_path)
        
        # Preprocess data
        preprocessed = preprocess_data(
            df,
            target_column=TARGET_COLUMN,
            categorical_columns=CATEGORICAL_COLUMNS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            scale_features_flag=False  # We'll use the saved scaler
        )
        
        # If we have a saved scaler, use it; otherwise use the one from preprocessing
        if scaler:
            X_test_scaled, _ = scale_features(
                preprocessed['encoded_df'].drop([TARGET_COLUMN], axis=1),
                scaler=scaler,
                fit=False
            )
            y_test = preprocessed['y_test']
        else:
            X_test_scaled = preprocessed['X_test']
            y_test = preprocessed['y_test']
        
        # Evaluate model
        metrics, predictions = evaluate_model(
            model,
            X_test_scaled,
            y_test,
            run_name="evaluation"
        )
        
        # Log model path/uri
        if model_path:
            mlflow.log_param("model_path", str(model_path))
        if model_uri:
            mlflow.log_param("model_uri", model_uri)
        
        logger.info("Evaluation pipeline completed successfully!")
        
        return metrics, predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate housing price prediction model")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local model directory"
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        help="MLflow model URI"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file (optional)"
    )
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        model_uri=args.model_uri,
        data_path=args.data_path
    )

