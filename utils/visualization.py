"""
Visualization utilities for housing price prediction.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_price_distribution(df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Plot the distribution of housing prices.
    
    Args:
        df: DataFrame with 'price' column
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True)
    plt.title('Distribution of Price')
    plt.xlabel('Price')
    plt.ylabel('Count')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    plt.close()


def plot_pairplot(
    df: pd.DataFrame,
    columns: list,
    save_path: Path = None
) -> None:
    """
    Create pairplot for numerical columns.
    
    Args:
        df: DataFrame
        columns: List of column names to include in pairplot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[columns])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pairplot to {save_path}")
    
    plt.close()


def plot_countplots(
    df: pd.DataFrame,
    categorical_columns: list,
    save_dir: Path = None
) -> None:
    """
    Create countplots for categorical columns.
    
    Args:
        df: DataFrame
        categorical_columns: List of categorical column names
        save_dir: Optional directory to save figures
    """
    for col in categorical_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df)
        plt.title(f'Counts of {col.title()}')
        plt.xlabel(col.title())
        plt.ylabel('Count')
        
        if col == 'furnishingstatus':
            plt.xticks(rotation=45)
        
        if save_dir:
            save_path = save_dir / f"countplot_{col}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved countplot to {save_path}")
        
        plt.close()


def plot_boxplots(
    df: pd.DataFrame,
    categorical_column: str,
    target_column: str = 'price',
    save_path: Path = None
) -> None:
    """
    Create boxplot comparing target variable across categories.
    
    Args:
        df: DataFrame
        categorical_column: Column to group by
        target_column: Target variable to plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=categorical_column, y=target_column, data=df)
    plt.title(f'{target_column.title()} by {categorical_column.title()}')
    plt.xlabel(categorical_column.title())
    plt.ylabel(target_column.title())
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved boxplot to {save_path}")
    
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Path = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot correlation heatmap for all features.
    
    Args:
        df: DataFrame
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Correlation Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
    
    plt.close()


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    color: str = 'green',
    save_path: Path = None
) -> None:
    """
    Plot histogram for a numerical column.
    
    Args:
        df: DataFrame
        column: Column name to plot
        color: Color for the histogram
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color=color)
    plt.title(f'Distribution of {column.title()}')
    plt.xlabel(column.title())
    plt.ylabel('Count')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved histogram to {save_path}")
    
    plt.close()


def plot_predictions_vs_actual(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None
) -> None:
    """
    Plot predicted vs actual values.
    
    Args:
        y_actual: Actual target values
        y_pred: Predicted target values
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_pred, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], 
             [y_actual.min(), y_actual.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")
    
    plt.close()

