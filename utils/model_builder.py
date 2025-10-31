"""
Model building utilities for housing price prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(hp, input_shape: int = 20) -> keras.Model:
    """
    Build a neural network model with hyperparameters.
    
    Args:
        hp: Hyperparameter object from Keras Tuner
        input_shape: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # First hidden layer
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    hp_activation1 = hp.Choice('activation1', values=['relu', 'tanh', 'sigmoid'])
    model.add(layers.Dense(units=hp_units1, activation=hp_activation1))
    
    # Second hidden layer
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    hp_activation2 = hp.Choice('activation2', values=['relu', 'tanh', 'sigmoid'])
    model.add(layers.Dense(units=hp_units2, activation=hp_activation2))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile with hyperparameter learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model


def build_model_with_config(input_shape: int, config: dict) -> keras.Model:
    """
    Build a model with specific configuration (not for hyperparameter tuning).
    
    Args:
        input_shape: Number of input features
        config: Dictionary with model configuration:
            - units1, units2: Number of neurons in hidden layers
            - activation1, activation2: Activation functions
            - learning_rate: Learning rate for optimizer
            
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(
            units=config.get('units1', 256),
            activation=config.get('activation1', 'relu')
        ),
        layers.Dense(
            units=config.get('units2', 128),
            activation=config.get('activation2', 'relu')
        ),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 1e-3)
        ),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    logger.info(f"Built model with config: {config}")
    logger.info(f"Model input shape: {input_shape}")
    
    return model


def create_tuner(
    input_shape: int,
    max_trials: int = 10,
    executions_per_trial: int = 3,
    objective: str = 'val_loss',
    directory: str = 'hyperparameter_tuning',
    project_name: str = 'housing_price_prediction'
) -> RandomSearch:
    """
    Create a Keras Tuner RandomSearch instance.
    
    Args:
        input_shape: Number of input features
        max_trials: Maximum number of hyperparameter trials
        executions_per_trial: Number of model executions per trial
        objective: Objective to optimize
        directory: Directory to save tuning results
        project_name: Name of the tuning project
        
    Returns:
        RandomSearch tuner instance
    """
    def model_builder(hp):
        return build_model(hp, input_shape=input_shape)
    
    tuner = RandomSearch(
        model_builder,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name,
        overwrite=True
    )
    
    logger.info(f"Created RandomSearch tuner with {max_trials} max trials")
    
    return tuner

