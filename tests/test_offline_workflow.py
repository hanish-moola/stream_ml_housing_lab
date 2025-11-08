from __future__ import annotations

import pandas as pd
import yaml

from mlflow.tracking import MlflowClient

from src.config import load_config
from src.workflows.offline_train import run_offline_workflow


def _write_config(path, data_path, tracking_uri):
    config = {
        "project": {"name": "workflow-test"},
        "data": {
            "raw_data_path": str(data_path),
            "target_column": "price",
            "test_size": 0.2,
            "random_state": 42,
        },
        "mlflow": {
            "experiment_name": "workflow-test",
            "tracking_uri": str(tracking_uri),
            "run_name_template": "run_{timestamp}",
        },
        "model": {
            "type": "linear_regression",
            "hyperparameters": {},
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _write_dataset(path):
    df = pd.DataFrame(
        [
            {
                "price": 150000 + i * 5000,
                "area": 1200 + i * 50,
                "bedrooms": 3 + (i % 2),
                "bathrooms": 2.0 + (i % 2),
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
            for i in range(8)
        ]
    )
    df.to_csv(path, index=False)


def test_offline_workflow_logs_runs(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    data_path = tmp_path / "housing.csv"
    config_path = tmp_path / "config.yaml"

    _write_dataset(data_path)
    _write_config(config_path, data_path, tracking_uri)

    config = load_config(path=config_path)
    summary = run_offline_workflow(config, data_path=data_path, run_name="workflow_test")

    client = MlflowClient(tracking_uri=str(tracking_uri))
    training_run_id = summary["stages"]["training"]
    evaluation_run_id = summary["stages"]["evaluation"]

    assert training_run_id is not None
    assert evaluation_run_id is not None
    assert client.get_run(training_run_id).data.params["model_type"] == "linear_regression"
    assert client.get_run(evaluation_run_id).data.params["evaluated_model_run_id"] == training_run_id
