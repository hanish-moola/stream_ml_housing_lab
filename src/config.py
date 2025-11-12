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
class MLflowConfig:
    experiment_name: str
    tracking_uri: str
    run_name_template: str = "run_{timestamp}"


@dataclass(frozen=True)
class ModelConfig:
    type: str
    hyperparameters: Dict[str, Any]


@dataclass(frozen=True)
class StreamingConfig:
    batch_size: int = 32
    sleep_interval_seconds: float = 0.5
    loop_forever: bool = False
    max_rows_per_cycle: Optional[int] = None
    checkpoint_path: Optional[Path] = None
    output_dir: Path = Path("data/stream_outputs")
    output_format: str = "jsonl"
    include_ground_truth: bool = True
    enable_mlflow_logging: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    data: DataConfig
    mlflow: MLflowConfig
    model: ModelConfig
    streaming: StreamingConfig

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
            "mlflow": {
                "experiment_name": self.mlflow.experiment_name,
                "tracking_uri": self.mlflow.tracking_uri,
                "run_name_template": self.mlflow.run_name_template,
            },
            "model": {
                "type": self.model.type,
                "hyperparameters": self.model.hyperparameters,
            },
            "streaming": {
                "batch_size": self.streaming.batch_size,
                "sleep_interval_seconds": self.streaming.sleep_interval_seconds,
                "loop_forever": self.streaming.loop_forever,
                "max_rows_per_cycle": self.streaming.max_rows_per_cycle,
                "checkpoint_path": str(self.streaming.checkpoint_path) if self.streaming.checkpoint_path else None,
                "output_dir": str(self.streaming.output_dir),
                "output_format": self.streaming.output_format,
                "include_ground_truth": self.streaming.include_ground_truth,
                "enable_mlflow_logging": self.streaming.enable_mlflow_logging,
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
    mlflow_cfg = raw_config.get("mlflow", {})
    model_cfg = raw_config.get("model", {})
    streaming_cfg = raw_config.get("streaming", {})

    raw_data_path = Path(os.getenv("HOUSING_DATA_PATH", data_cfg.get("raw_data_path")))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri"))
    checkpoint_path = streaming_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)

    project_config = ProjectConfig(
        project_name=project.get("name", "stream-ml-housing-lab"),
        data=DataConfig(
            raw_data_path=raw_data_path,
            target_column=data_cfg.get("target_column", "price"),
            index_column=data_cfg.get("index_column"),
            test_size=float(data_cfg.get("test_size", 0.2)),
            random_state=int(data_cfg.get("random_state", 42)),
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
        streaming=StreamingConfig(
            batch_size=int(streaming_cfg.get("batch_size", 32)),
            sleep_interval_seconds=float(streaming_cfg.get("sleep_interval_seconds", 0.5)),
            loop_forever=bool(streaming_cfg.get("loop_forever", False)),
            max_rows_per_cycle=(
                int(streaming_cfg["max_rows_per_cycle"]) if streaming_cfg.get("max_rows_per_cycle") is not None else None
            ),
            checkpoint_path=checkpoint_path,
            output_dir=Path(streaming_cfg.get("output_dir", "data/stream_outputs")),
            output_format=streaming_cfg.get("output_format", "jsonl"),
            include_ground_truth=bool(streaming_cfg.get("include_ground_truth", True)),
            enable_mlflow_logging=bool(streaming_cfg.get("enable_mlflow_logging", True)),
        ),
    )

    return project_config
