from __future__ import annotations

import numpy as np
import pytest

from src.train import _build_model, _compute_metrics


def test_build_model_linear_regression():
    model = _build_model("linear_regression", {"fit_intercept": True})
    assert model.__class__.__name__ == "LinearRegression"
    assert model.fit_intercept is True


def test_build_model_unsupported_raises():
    with pytest.raises(ValueError):
        _build_model("unsupported", {})


def test_compute_metrics_outputs_expected_values():
    y_true = np.array([3.0, 5.0, 7.0])
    y_pred = np.array([2.5, 5.5, 7.5])

    metrics = _compute_metrics(y_true, y_pred)

    assert pytest.approx(metrics["mae"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["mse"], rel=1e-6) == 0.25
    assert pytest.approx(metrics["rmse"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["r2"], rel=1e-6) == 0.90625
