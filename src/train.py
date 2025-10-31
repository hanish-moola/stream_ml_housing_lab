"""
Training script for housing price prediction with MLflow integration.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    HYPERPARAMETER_CONFIG,
    TRAINING_CONFIG,
    MLFLOW_CONFIG,
    RESULTS_DIR
)
from utils.data_loader import load_housing_data, get_data_info
from utils.preprocessing import preprocess_data
from utils.visualization import (
    plot_price_distribution,
    plot_pairplot,
    plot_countplots,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_histogram
)
from utils.model_builder import create_tuner, build_model_with_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = None, run_name: str = None):
    """
    Setup MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
    """
    mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])
    
    if run_name is None:
        run_name = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"MLflow experiment: {experiment_name or MLFLOW_CONFIG['experiment_name']}")
    logger.info(f"MLflow run: {run_name}")
    
    return run_name


def log_data_info(df: pd.DataFrame, data_info: dict):
    """
    Log data information to MLflow.
    
    Args:
        df: DataFrame
        data_info: Dictionary containing data information
    """
    mlflow.log_param("data_shape_rows", data_info['shape'][0])
    mlflow.log_param("data_shape_cols", data_info['shape'][1])
    mlflow.log_param("memory_usage_mb", round(data_info['memory_usage_mb'], 2))
    mlflow.log_param("num_features", len(data_info['columns']) - 1)  # Excluding target


def create_visualizations(df: pd.DataFrame, encoded_df: pd.DataFrame, save_dir: Path):
    """
    Create and save all visualizations.
    
    Args:
        df: Original DataFrame
        encoded_df: Encoded DataFrame
        save_dir: Directory to save visualizations
    """
    save_dir.mkdir(exist_ok=True)
    
    logger.info("Creating visualizations...")
    
    # Price distribution
    plot_price_distribution(df, save_dir / "price_distribution.png")
    
    # Pairplot for numerical columns
    numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    available_cols = [col for col in numerical_cols if col in df.columns]
    if available_cols:
        plot_pairplot(df, available_cols, save_dir / "pairplot.png")
    
    # Countplots
    plot_countplots(df, CATEGORICAL_COLUMNS, save_dir)
    
    # Boxplots
    plot_boxplots(df, 'bedrooms', 'price', save_dir / "boxplot_bedrooms_price.png")
    plot_boxplots(df, 'mainroad', 'price', save_dir / "boxplot_mainroad_price.png")
    
    # Area distribution
    if 'area' in df.columns:
        plot_histogram(df, 'area', color='green', save_path=save_dir / "area_distribution.png")
    
    # Correlation heatmap
    plot_correlation_heatmap(encoded_df, save_dir / "correlation_heatmap.png")
    
    logger.info(f"Visualizations saved to {save_dir}")
    
    # Log visualizations to MLflow
    for viz_file in save_dir.glob("*.png"):
        mlflow.log_artifact(str(viz_file))


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    input_shape: int
):
    """
    Run hyperparameter tuning with Keras Tuner.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        input_shape: Number of input features
        
    Returns:
        Best model from tuning
    """
    logger.info("Starting hyperparameter tuning...")
    
    tuner = create_tuner(
        input_shape=input_shape,
        max_trials=HYPERPARAMETER_CONFIG['max_trials'],
        executions_per_trial=HYPERPARAMETER_CONFIG['executions_per_trial'],
        objective=HYPERPARAMETER_CONFIG['objective'],
        directory=HYPERPARAMETER_CONFIG['directory'],
        project_name=HYPERPARAMETER_CONFIG['project_name']
    )
    
    # Log hyperparameter search configuration
    mlflow.log_params({
        'max_trials': HYPERPARAMETER_CONFIG['max_trials'],
        'executions_per_trial': HYPERPARAMETER_CONFIG['executions_per_trial'],
        'tuning_epochs': HYPERPARAMETER_CONFIG['epochs']
    })
    
    # Search for best hyperparameters
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=HYPERPARAMETER_CONFIG['epochs'],
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    logger.info(f"Best hyperparameters: {best_hps.values}")
    
    # Log best hyperparameters
    mlflow.log_params(best_hps.values)
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model, best_hps


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    epochs: int = 50
):
    """
    Train the model with callbacks.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    logger.info(f"Training model for {epochs} epochs...")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=TRAINING_CONFIG['verbose']
    )
    
    # Log training history metrics
    best_epoch = len(history.history['loss'])
    mlflow.log_metric("best_epoch", best_epoch)
    mlflow.log_metric("final_train_loss", history.history['loss'][-1])
    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
    mlflow.log_metric("final_train_mae", history.history['mean_absolute_error'][-1])
    mlflow.log_metric("final_val_mae", history.history['val_mean_absolute_error'][-1])
    
    return history


def main():
    """Main training pipeline."""
    # Setup MLflow
    run_name = setup_mlflow()
    
    with mlflow.start_run(run_name=run_name):
        logger.info("Starting training pipeline...")
        
        # Log configuration
        mlflow.log_params({
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'target_column': TARGET_COLUMN,
            'categorical_columns': str(CATEGORICAL_COLUMNS)
        })
        
        # Load data
        logger.info("Loading data...")
        df = load_housing_data(DATA_PATH)
        data_info = get_data_info(df)
        log_data_info(df, data_info)
        
        # Create visualizations
        viz_dir = RESULTS_DIR / "visualizations" / run_name
        create_visualizations(df, df, viz_dir)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessed = preprocess_data(
            df,
            target_column=TARGET_COLUMN,
            categorical_columns=CATEGORICAL_COLUMNS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            scale_features_flag=True
        )
        
        X_train = preprocessed['X_train']
        X_test = preprocessed['X_test']
        y_train = preprocessed['y_train']
        y_test = preprocessed['y_test']
        scaler = preprocessed['scaler']
        encoded_df = preprocessed['encoded_df']
        
        # Log feature information
        mlflow.log_param("num_features_final", X_train.shape[1])
        
        # Update visualizations with encoded data
        create_visualizations(df, encoded_df, viz_dir)
        
        # Hyperparameter tuning
        logger.info("Running hyperparameter tuning...")
        best_model, best_hps = run_hyperparameter_tuning(
            X_train, y_train, X_test, y_test, input_shape=X_train.shape[1]
        )
        
        # Retrain best model
        logger.info("Retraining best model...")
        history = train_model(
            best_model,
            X_train, y_train, X_test, y_test,
            epochs=TRAINING_CONFIG['epochs']
        )
        
        # Save model
        model_path = RESULTS_DIR / "models" / run_name
        model_path.mkdir(parents=True, exist_ok=True)
        model_save_path = model_path / "model.keras"
        best_model.save(str(model_save_path))
        
        logger.info(f"Model saved to {model_save_path}")
        
        # Log model to MLflow
        mlflow.keras.log_model(
            best_model,
            "model",
            registered_model_name="housing_price_predictor"
        )
        
        # Log scaler and preprocessing artifacts
        import pickle
        scaler_path = model_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(str(scaler_path))
        
        # Log feature names
        feature_names_path = model_path / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(preprocessed['feature_names']))
        mlflow.log_artifact(str(feature_names_path))
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()

