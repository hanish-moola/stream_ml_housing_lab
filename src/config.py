"""Configuration loader for the Stream-ML Housing Lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: Path
    target_column: str
    index_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ArtifactsConfig:
    root: Path
    transformer_subdir: str = "transformers"
    model_subdir: str = "models"
    metrics_subdir: str = "metrics"


@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str
    tracking_uri: str
    run_name_template: str = "run_{timestamp}"


@dataclass(frozen=True)
class ModelConfig:
    type: str
    hyperparameters: Dict[str, Any]


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    data: DataConfig
    artifacts: ArtifactsConfig
    mlflow: MLflowConfig
    model: ModelConfig

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a serialisable dictionary."""

        return {
            "project_name": self.project_name,
            "data": {
                "raw_data_path": str(self.data.raw_data_path),
                "target_column": self.data.target_column,
                "index_column": self.data.index_column,
                "test_size": self.data.test_size,
                "random_state": self.data.random_state,
            },
            "artifacts": {
                "root": str(self.artifacts.root),
                "transformer_subdir": self.artifacts.transformer_subdir,
                "model_subdir": self.artifacts.model_subdir,
                "metrics_subdir": self.artifacts.metrics_subdir,
            },
            "mlflow": {
                "experiment_name": self.mlflow.experiment_name,
                "tracking_uri": self.mlflow.tracking_uri,
                "run_name_template": self.mlflow.run_name_template,
            },
            "model": {
                "type": self.model.type,
                "hyperparameters": self.model.hyperparameters,
            },
        }


def _resolve_config_path(explicit_path: Optional[Path] = None) -> Path:
    if explicit_path:
        return explicit_path

    env_path = os.getenv("HOUSING_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    return Path("config/config.yaml")


def load_config(path: Optional[Path] = None) -> ProjectConfig:
    """Load project configuration from YAML and apply env overrides."""

    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    project = raw_config.get("project", {})
    data_cfg = raw_config.get("data", {})
    artifacts_cfg = raw_config.get("artifacts", {})
    mlflow_cfg = raw_config.get("mlflow", {})
    model_cfg = raw_config.get("model", {})

    raw_data_path = Path(os.getenv("HOUSING_DATA_PATH", data_cfg.get("raw_data_path")))
    artifacts_root = Path(os.getenv("HOUSING_ARTIFACTS_ROOT", artifacts_cfg.get("root")))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri"))

    project_config = ProjectConfig(
        project_name=project.get("name", "stream-ml-housing-lab"),
        data=DataConfig(
            raw_data_path=raw_data_path,
            target_column=data_cfg.get("target_column", "price"),
            index_column=data_cfg.get("index_column"),
            test_size=float(data_cfg.get("test_size", 0.2)),
            random_state=int(data_cfg.get("random_state", 42)),
        ),
        artifacts=ArtifactsConfig(
            root=artifacts_root,
            transformer_subdir=artifacts_cfg.get("transformer_subdir", "transformers"),
            model_subdir=artifacts_cfg.get("model_subdir", "models"),
            metrics_subdir=artifacts_cfg.get("metrics_subdir", "metrics"),
        ),
        mlflow=MLflowConfig(
            experiment_name=mlflow_cfg.get("experiment_name", "housing_price_workflow"),
            tracking_uri=tracking_uri,
            run_name_template=mlflow_cfg.get("run_name_template", "run_{timestamp}"),
        ),
        model=ModelConfig(
            type=model_cfg.get("type", "linear_regression"),
            hyperparameters=model_cfg.get("hyperparameters", {}),
        ),
    )

    return project_config
