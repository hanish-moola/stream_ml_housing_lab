"""Prediction CLI for the housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd

from .config import ProjectConfig, load_config
from .feature_engineering import FeatureMetadata
from .logging_utils import configure_logging, get_logger
from .mlflow_utils import ensure_run
from .registry import (
    build_run_name,
    load_metadata,
    load_model,
    resolve_latest_run,
)

logger = get_logger(__name__)


def _resolve_model_artifacts(config: ProjectConfig, run_dir: Optional[Path]) -> Dict[str, Path]:
    if run_dir:
        model_path = run_dir / config.artifacts.model_subdir / "model.joblib"
        metadata_path = run_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Training metadata missing: {metadata_path}")
        return {"model": model_path, "metadata": metadata_path}

    latest = resolve_latest_run(config.artifacts)
    if latest is None or not latest.model_path.exists():
        raise FileNotFoundError("No trained model artifacts found")

    metadata_path = latest.metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata missing: {metadata_path}")

    return {"model": latest.model_path, "metadata": metadata_path}


def _load_feature_metadata(training_metadata: Dict[str, object]) -> FeatureMetadata:
    feature_meta_path = training_metadata.get("feature_artifacts", {}).get("metadata")
    if not feature_meta_path:
        raise RuntimeError("Training metadata does not reference feature metadata path")

    path = Path(feature_meta_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature metadata file missing: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FeatureMetadata(**payload)


def _prepare_feature_dataframe(
    features: Dict[str, object],
    feature_metadata: FeatureMetadata,
) -> pd.DataFrame:
    expected_columns = feature_metadata.numeric + feature_metadata.categorical
    missing = [col for col in expected_columns if col not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    ordered = {col: features[col] for col in expected_columns}
    df = pd.DataFrame([ordered])
    return df


def run_prediction(
    config: ProjectConfig,
    input_features: Dict[str, object],
    run_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> float:
    configure_logging()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    artifacts = _resolve_model_artifacts(config, run_dir)
    training_metadata = load_metadata(artifacts["metadata"])
    feature_metadata = _load_feature_metadata(training_metadata)

    pipeline = load_model(artifacts["model"])

    features_df = _prepare_feature_dataframe(input_features, feature_metadata)
    prediction = float(pipeline.predict(features_df)[0])

    effective_run_name = run_name or build_run_name("prediction_{timestamp}")
    with ensure_run(effective_run_name) as run:
        mlflow.log_params({f"feature_{k}": v for k, v in input_features.items()})
        mlflow.log_metric("prediction", prediction)
        mlflow.set_tag("prediction", True)
        mlflow.set_tag("source_model", str(artifacts["model"]))
        logger.info(
            "Prediction run %s | model=%s | prediction=%.2f",
            run.info.run_id,
            artifacts["model"],
            prediction,
        )

    return prediction


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a prediction from the housing price model")
    parser.add_argument("--config", type=Path, help="Optional path to configuration YAML")
    parser.add_argument("--model-run", type=Path, help="Optional path to a specific training run directory")
    parser.add_argument("--features", type=Path, required=True, help="Path to JSON file containing feature values")
    parser.add_argument("--run-name", type=str, help="Optional explicit prediction run name")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)

    with cli_args.features.open("r", encoding="utf-8") as handle:
        features = json.load(handle)

    prediction = run_prediction(
        config,
        input_features=features,
        run_dir=cli_args.model_run,
        run_name=cli_args.run_name,
    )

    print(json.dumps({"prediction": prediction}))


if __name__ == "__main__":  # pragma: no cover
    main()
