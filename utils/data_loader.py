"""
Data loading utilities for housing price prediction.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_housing_data(data_path: str) -> pd.DataFrame:
    """
    Load housing dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame containing the housing data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    logger.info(f"Dataset shape: {info['shape']}")
    logger.info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    
    return info

