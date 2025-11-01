"""Evaluation pipeline for the trained housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd

from .config import ArtifactsConfig, ProjectConfig, load_config
from .data import load_raw_data, split_features_target, stratified_train_test_split
from .logging_utils import configure_logging, get_logger
from .registry import (
    build_run_name,
    load_model,
    prepare_run_artifacts,
    resolve_latest_run,
    write_metadata,
)
from .train import _compute_metrics

logger = get_logger(__name__)


def _resolve_model_artifacts(
    artifacts_cfg: ArtifactsConfig,
    model_run_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    if model_run_dir is not None:
        run_dir = model_run_dir
    else:
        latest = resolve_latest_run(artifacts_cfg)
        if latest is None or not latest.model_path.exists():
            raise FileNotFoundError("No trained model artifacts found. Train a model first.")
        run_dir = latest.run_dir

    model_path = run_dir / artifacts_cfg.model_subdir / "model.joblib"
    transformer_path = run_dir / artifacts_cfg.transformer_subdir / "transformer.joblib"
    metadata_path = run_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer artifact missing: {transformer_path}")

    return {
        "run_dir": run_dir,
        "model": model_path,
        "transformer": transformer_path,
        "metadata": metadata_path,
    }


def run_evaluation(
    config: ProjectConfig,
    model_run_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> None:
    configure_logging()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    base_run_name = run_name or build_run_name(config.mlflow.run_name_template)
    effective_run_name = f"evaluation_{base_run_name}" if not run_name else run_name
    artifacts = prepare_run_artifacts(config.artifacts, effective_run_name)

    model_artifacts = _resolve_model_artifacts(config.artifacts, model_run_dir)
    pipeline = load_model(model_artifacts["model"])

    with mlflow.start_run(run_name=effective_run_name) as run:
        logger.info("Evaluating model run %s", model_artifacts["run_dir"])

        df = load_raw_data(config.data)
        X, y = split_features_target(df, config.data)
        _, X_test, _, y_test = stratified_train_test_split(X, y, config.data)

        predictions = pipeline.predict(X_test)
        metrics = _compute_metrics(y_test.to_numpy(), predictions)
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

        metrics_path = artifacts.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        mlflow.log_artifact(str(metrics_path))

        evaluation_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": predictions,
            "residual": y_test - predictions,
        })
        predictions_path = metrics_path.parent / "evaluation_predictions.csv"
        evaluation_df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(str(predictions_path))

        mlflow.log_params(
            {
                "evaluation_model_run": str(model_artifacts["run_dir"]),
                "evaluation_dataset": str(config.data.raw_data_path),
            }
        )
        mlflow.set_tag("evaluation", True)

        evaluation_metadata = {
            "evaluation_run": effective_run_name,
            "mlflow_run_id": run.info.run_id,
            "evaluated_model": str(model_artifacts["model"]),
            "metrics": metrics,
            "artifacts": {
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
            },
        }
        write_metadata(evaluation_metadata, artifacts.metadata_path)

        logger.info("Evaluation complete. Metrics stored at %s", metrics_path)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest trained housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--model-run", type=Path, help="Optional path to a specific training run directory")
    parser.add_argument("--run-name", type=str, help="Optional explicit evaluation run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    run_evaluation(config, model_run_dir=cli_args.model_run, run_name=cli_args.run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
