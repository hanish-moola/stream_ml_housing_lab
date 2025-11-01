from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.feature_engineering import FeatureMetadata
from src.predict import _prepare_feature_dataframe


def test_prepare_feature_dataframe_orders_columns():
    metadata = FeatureMetadata(
        numeric=["area", "bedrooms"],
        categorical=["mainroad"],
        feature_names=["area", "bedrooms", "mainroad_yes"],
    )
    df = _prepare_feature_dataframe(
        {"bedrooms": 3, "area": 1000, "mainroad": "yes"},
        metadata,
    )

    assert list(df.columns) == ["area", "bedrooms", "mainroad"]


def test_prepare_feature_dataframe_missing_feature_raises():
    metadata = FeatureMetadata(numeric=["area"], categorical=["mainroad"], feature_names=["area", "mainroad_yes"])
    with pytest.raises(ValueError):
        _prepare_feature_dataframe({"area": 1000}, metadata)
