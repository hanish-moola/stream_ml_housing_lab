"""
Data preprocessing utilities for housing price prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: list,
    drop_first: bool = False
) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names to encode
        drop_first: Whether to drop first category to avoid multicollinearity
        
    Returns:
        DataFrame with encoded categorical features
    """
    logger.info(f"Encoding {len(categorical_columns)} categorical columns")
    encoded_df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=drop_first
    )
    logger.info(f"Encoded data shape: {encoded_df.shape}")
    logger.info(f"New columns after encoding: {list(encoded_df.columns)}")
    
    return encoded_df


def scale_features(
    X: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X: Feature matrix to scale
        scaler: Pre-fitted scaler (if provided)
        fit: Whether to fit the scaler (if scaler is None)
        
    Returns:
        Tuple of (scaled features, fitted scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        logger.info("Creating new StandardScaler")
    
    if fit:
        logger.info("Fitting scaler and transforming features")
        X_scaled = scaler.fit_transform(X)
    else:
        logger.info("Transforming features with existing scaler")
        X_scaled = scaler.transform(X)
    
    logger.info(f"Scaled features shape: {X_scaled.shape}")
    
    return X_scaled, scaler


def split_data(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 100
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list,
    test_size: float = 0.2,
    random_state: int = 100,
    scale_features_flag: bool = True
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline: encoding, scaling, and splitting.
    
    Args:
        df: Raw DataFrame
        target_column: Name of the target column
        categorical_columns: List of categorical columns to encode
        test_size: Proportion of data for testing
        random_state: Random seed
        scale_features_flag: Whether to scale features
        
    Returns:
        Dictionary containing:
            - X_train, X_test, y_train, y_test
            - scaler (if scaling was performed)
            - encoded_df (for reference)
    """
    logger.info("Starting complete preprocessing pipeline")
    
    # Encode categorical features
    encoded_df = encode_categorical_features(df, categorical_columns)
    
    # Separate features and target
    X = encoded_df.drop([target_column], axis=1)
    y = encoded_df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Scale features
    scaler = None
    if scale_features_flag:
        X_scaled, scaler = scale_features(X)
    else:
        X_scaled = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, test_size, random_state
    )
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoded_df': encoded_df,
        'feature_names': list(X.columns)
    }
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return result

