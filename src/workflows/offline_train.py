"""Offline training workflow orchestrating feature engineering, training, and evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

from ..config import ProjectConfig, load_config
from ..feature_engineering import run_feature_engineering
from ..logging_utils import configure_logging, get_logger
from ..registry import (
    build_run_name,
    load_metadata,
    prepare_run_artifacts,
    write_metadata,
)
from ..train import run_training
from ..evaluate import run_evaluation

logger = get_logger(__name__)


def _apply_data_override(config: ProjectConfig, data_path: Optional[Path]) -> ProjectConfig:
    if data_path is None:
        return config
    data = replace(config.data, raw_data_path=data_path)
    return replace(config, data=data)


def _stage_run_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def run_offline_workflow(
    config: ProjectConfig,
    *,
    data_path: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Run feature engineering, training, and evaluation sequentially.

    Returns a mapping containing metadata for each stage.
    """

    configure_logging()
    config = _apply_data_override(config, data_path)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    base_run = run_name or build_run_name(config.mlflow.run_name_template)
    stage_names = {
        "feature_engineering": _stage_run_name(base_run, "fe"),
        "training": _stage_run_name(base_run, "train"),
        "evaluation": _stage_run_name(base_run, "eval"),
    }

    logger.info("Running offline workflow with base run '%s'", base_run)

    run_feature_engineering(config, run_name=stage_names["feature_engineering"])
    run_training(config, run_name=stage_names["training"])
    training_run_dir = config.artifacts.root / stage_names["training"]
    run_evaluation(
        config,
        model_run_dir=training_run_dir,
        run_name=stage_names["evaluation"]
    )

    stage_metadata: Dict[str, Dict[str, object]] = {}
    for stage, name in stage_names.items():
        metadata_path = config.artifacts.root / name / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Expected metadata for stage '{stage}' at {metadata_path}")
        stage_metadata[stage] = load_metadata(metadata_path)

    with mlflow.start_run(run_name=base_run):
        summary_paths = prepare_run_artifacts(config.artifacts, base_run)
        summary = {
            "workflow_run": base_run,
            "dataset_path": str(config.data.raw_data_path),
            "stages": stage_metadata,
        }
        write_metadata(summary, summary_paths.metadata_path)
        mlflow.log_dict(summary, "workflow/summary.json")
        mlflow.log_params({
            "workflow_dataset_path": str(config.data.raw_data_path),
            "feature_run": stage_names["feature_engineering"],
            "feature_run_id": stage_metadata["feature_engineering"].get("mlflow_run_id"),
            "training_run": stage_names["training"],
            "training_run_id": stage_metadata["training"].get("mlflow_run_id"),
            "evaluation_run": stage_names["evaluation"],
            "evaluation_run_id": stage_metadata["evaluation"].get("mlflow_run_id"),
        })
        mlflow.set_tag("workflow", "offline_train")

    logger.info("Offline workflow complete. Summary stored at %s", summary_paths.metadata_path)
    return summary


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline training workflow")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--data-path", type=Path, help="Override dataset CSV path")
    parser.add_argument("--run-name", type=str, help="Optional base run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    summary = run_offline_workflow(
        config,
        data_path=cli_args.data_path,
        run_name=cli_args.run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
