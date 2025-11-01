from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config
from src.workflows.offline_train import run_offline_workflow


def _write_config(path: Path, data_path: Path, artifacts_root: Path, mlruns_root: Path) -> None:
    config = {
        "project": {"name": "workflow-test"},
        "data": {
            "raw_data_path": str(data_path),
            "target_column": "price",
            "index_column": None,
            "test_size": 0.2,
            "random_state": 42,
        },
        "artifacts": {
            "root": str(artifacts_root),
            "transformer_subdir": "transformers",
            "model_subdir": "models",
            "metrics_subdir": "metrics",
        },
        "mlflow": {
            "experiment_name": "workflow-test",
            "tracking_uri": str(mlruns_root),
            "run_name_template": "run_{timestamp}",
        },
        "model": {
            "type": "linear_regression",
            "hyperparameters": {},
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _write_dataset(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "price": 100000 + i * 10000,
                "area": 1000 + 50 * i,
                "bedrooms": 3 + (i % 2),
                "bathrooms": 2 + (i % 2),
                "stories": 2,
                "parking": 1,
                "mainroad": "yes" if i % 2 == 0 else "no",
                "guestroom": "no",
                "basement": "no",
                "hotwaterheating": "no",
                "airconditioning": "yes",
                "prefarea": "no",
                "furnishingstatus": "semi-furnished",
            }
            for i in range(6)
        ]
    )
    df.to_csv(path, index=False)


def test_offline_workflow_creates_artifacts(tmp_path):
    data_path = tmp_path / "data.csv"
    artifacts_root = tmp_path / "artifacts"
    mlruns_root = tmp_path / "mlruns"
    config_path = tmp_path / "config.yaml"

    _write_dataset(data_path)
    _write_config(config_path, data_path, artifacts_root, mlruns_root)

    config = load_config(path=config_path)
    summary = run_offline_workflow(config, data_path=data_path, run_name="workflow_test")

    train_model_path = artifacts_root / "workflow_test_train" / "models" / "model.joblib"
    eval_metrics_path = artifacts_root / "workflow_test_eval" / "metrics" / "metrics.json"
    workflow_summary_path = artifacts_root / "workflow_test" / "metadata.json"

    assert train_model_path.exists()
    assert eval_metrics_path.exists()
    assert workflow_summary_path.exists()
    assert summary["workflow_run"] == "workflow_test"
    assert "training" in summary["stages"]
