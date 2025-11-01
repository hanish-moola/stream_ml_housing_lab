from __future__ import annotations

from pathlib import Path

import mlflow

from src.registry import get_latest_run_by_stage, download_artifact


def test_get_latest_run_by_stage(tmp_path, monkeypatch):
    tracking_uri = tmp_path / "mlruns"
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment("registry-test")

    with mlflow.start_run(run_name="old") as old_run:
        mlflow.set_tag("stage", "feature_engineering")
    with mlflow.start_run(run_name="new") as new_run:
        mlflow.set_tag("stage", "feature_engineering")

    latest = get_latest_run_by_stage("registry-test", "feature_engineering")
    assert latest is not None
    assert latest.info.run_id == new_run.info.run_id


def test_download_artifact(tmp_path, monkeypatch):
    tracking_uri = tmp_path / "mlruns"
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment("registry-download")

    with mlflow.start_run() as run:
        artifact_file = tmp_path / "sample.txt"
        artifact_file.write_text("hello")
        mlflow.log_artifact(str(artifact_file), artifact_path="bundle")

    downloaded = download_artifact(run.info.run_id, "bundle/sample.txt")
    assert downloaded.read_text() == "hello"
