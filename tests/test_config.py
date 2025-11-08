from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.config import load_config


def test_load_config_with_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    yaml.safe_dump(
        {
            "project": {"name": "demo"},
            "data": {"raw_data_path": "data.csv", "target_column": "price"},
            "mlflow": {"experiment_name": "exp", "tracking_uri": "mlruns"},
            "model": {"type": "linear", "hyperparameters": {}},
        },
        config_path.open("w", encoding="utf-8"),
    )

    monkeypatch.setenv("HOUSING_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("HOUSING_DATA_PATH", "/tmp/data.csv")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

    cfg = load_config()

    assert cfg.data.raw_data_path == Path("/tmp/data.csv")
    assert cfg.mlflow.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.project_name == "demo"
