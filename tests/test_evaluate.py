from __future__ import annotations

import pandas as pd
import yaml

import mlflow
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.feature_engineering import run_feature_engineering
from src.train import run_training
from src.evaluate import run_evaluation


def _write_config(path, data_path, tracking_uri):
    config = {
        "project": {"name": "eval-test"},
        "data": {
            "raw_data_path": str(data_path),
            "target_column": "price",
            "test_size": 0.2,
            "random_state": 42,
        },
        "mlflow": {
            "experiment_name": "eval-test",
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
        {
            "price": [100000, 120000, 140000, 160000, 180000],
            "area": [1000, 1100, 1200, 1300, 1400],
            "bedrooms": [3, 3, 4, 4, 5],
            "bathrooms": [2, 2, 2, 3, 3],
            "stories": [2, 2, 2, 3, 3],
            "parking": [1, 1, 2, 2, 2],
            "mainroad": ["yes", "no", "yes", "no", "yes"],
            "guestroom": ["no"] * 5,
            "basement": ["no"] * 5,
            "hotwaterheating": ["no"] * 5,
            "airconditioning": ["yes"] * 5,
            "prefarea": ["no"] * 5,
            "furnishingstatus": ["semi-furnished"] * 5,
        }
    )
    df.to_csv(path, index=False)


def test_run_evaluation_logs_metrics(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    data_path = tmp_path / "housing.csv"
    config_path = tmp_path / "config.yaml"

    _write_dataset(data_path)
    _write_config(config_path, data_path, tracking_uri)

    config = load_config(path=config_path)

    run_feature_engineering(config)
    run_training(config)
    training_run = MlflowClient(str(tracking_uri)).search_runs(
        [MlflowClient(str(tracking_uri)).get_experiment_by_name("eval-test").experiment_id],
        "tags.stage = 'training'",
        max_results=1,
    )[0]

    run_evaluation(config, model_run_id=training_run.info.run_id)

    client = MlflowClient(str(tracking_uri))
    eval_run = client.search_runs(
        [client.get_experiment_by_name("eval-test").experiment_id],
        "tags.stage = 'evaluation'",
        max_results=1,
    )[0]

    assert "evaluated_model_run_id" in eval_run.data.params
    assert eval_run.data.params["evaluated_model_run_id"] == training_run.info.run_id
    assert "eval_mae" in eval_run.data.metrics
