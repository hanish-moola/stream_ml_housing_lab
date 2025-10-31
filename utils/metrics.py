"""Shared metric utilities for regression models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
    """Calculate core regression metrics and return them as a dictionary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, Optional[float]] = {
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "mean_squared_error": float(mean_squared_error(y_true, y_pred)),
    }
    metrics["root_mean_squared_error"] = float(np.sqrt(metrics["mean_squared_error"]))
    metrics["r2_score"] = float(r2_score(y_true, y_pred))

    try:
        positive_true = np.maximum(y_true, 1e-9)
        positive_pred = np.maximum(y_pred, 1e-9)
        rmsle = np.sqrt(mean_squared_log_error(positive_true, positive_pred))
        metrics["root_mean_squared_log_error"] = float(rmsle)
    except ValueError:
        metrics["root_mean_squared_log_error"] = None

    return metrics
