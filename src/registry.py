"""MLflow registry helpers for locating stage runs and artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from .logging_utils import get_logger

logger = get_logger(__name__)


def build_run_name(template: str) -> str:
    from datetime import datetime

    return template.format(timestamp=datetime.utcnow().strftime("%Y%m%d_%H%M%S"))


def _get_client() -> MlflowClient:
    return MlflowClient()


def get_latest_run_by_stage(experiment_name: str, stage: str) -> Optional[Run]:
    client = _get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.stage = '{stage}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0]


def download_artifact(run_id: str, artifact_path: str) -> Path:
    logger.debug("Downloading artifact %s from run %s", artifact_path, run_id)
    dest = Path(tempfile.mkdtemp())
    local_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=str(dest)))
    return local_path
