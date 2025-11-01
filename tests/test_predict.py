from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.predict import _prepare_feature_dataframe


def test_prepare_feature_dataframe_orders_columns():
    metadata = {
        "numeric": ["area", "bedrooms"],
        "categorical": ["mainroad"],
    }
    df = _prepare_feature_dataframe(
        {"bedrooms": 3, "area": 1000, "mainroad": "yes"},
        metadata,
    )

    assert list(df.columns) == ["area", "bedrooms", "mainroad"]


def test_prepare_feature_dataframe_missing_feature_raises():
    metadata = {"numeric": ["area"], "categorical": ["mainroad"]}
    with pytest.raises(ValueError):
        _prepare_feature_dataframe({"area": 1000}, metadata)
