from __future__ import annotations

import pandas as pd

from src.feature_engineering import (
    collect_training_stats,
    create_preprocessing_pipeline,
    infer_feature_types,
)


def test_infer_feature_types_recognises_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "area": [1000, 1200],
            "bedrooms": [3, 4],
            "mainroad": ["yes", "no"],
            "has_pool": [True, False],
        }
    )

    numeric, categorical = infer_feature_types(df)

    assert "area" in numeric
    assert "bedrooms" in numeric
    assert "mainroad" in categorical
    assert "has_pool" in categorical  # bool should be treated as categorical


def test_create_preprocessing_pipeline_generates_feature_names():
    df = pd.DataFrame(
        {
            "area": [1000, 1200],
            "bedrooms": [3, 4],
            "mainroad": ["yes", "no"],
        }
    )
    numeric, categorical = infer_feature_types(df)
    pipeline = create_preprocessing_pipeline(numeric, categorical)
    pipeline.fit(df)

    feature_names = pipeline.get_feature_names_out()
    # Expect 2 numeric + 2 categorical (since yes/no)
    assert len(feature_names) == 4


def test_collect_training_stats_returns_expected_keys():
    df = pd.DataFrame(
        {
            "area": [1000, 1200],
            "bedrooms": [3, 4],
            "mainroad": ["yes", "no"],
        }
    )
    numeric, categorical = infer_feature_types(df)
    stats = collect_training_stats(df, numeric, categorical)

    assert set(stats.to_dict().keys()) == {"numeric_means", "numeric_stds", "categorical_levels"}
    assert "area" in stats.numeric_means
    assert "mainroad" in stats.categorical_levels
