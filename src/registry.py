"""Artifact registry helpers for the housing price pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from .config import ArtifactsConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    model_path: Path
    transformer_path: Path
    metrics_path: Path
    metadata_path: Path


def build_run_name(template: str) -> str:
    return template.format(timestamp=datetime.utcnow().strftime("%Y%m%d_%H%M%S"))


def prepare_run_artifacts(
    artifacts_cfg: ArtifactsConfig,
    run_name: str,
) -> ArtifactPaths:
    """Create directory structure for a new run."""

    run_dir = artifacts_cfg.root / run_name
    transformer_dir = run_dir / artifacts_cfg.transformer_subdir
    model_dir = run_dir / artifacts_cfg.model_subdir
    metrics_dir = run_dir / artifacts_cfg.metrics_subdir

    for directory in (transformer_dir, model_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ArtifactPaths(
        run_dir=run_dir,
        model_path=model_dir / "model.joblib",
        transformer_path=transformer_dir / "transformer.joblib",
        metrics_path=metrics_dir / "metrics.json",
        metadata_path=run_dir / "metadata.json",
    )


def save_model(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    logger.info("Persisted model artifact at %s", destination)


def save_transformer(transformer: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, destination)
    logger.info("Persisted transformer artifact at %s", destination)


def write_metadata(metadata: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.debug("Wrote metadata to %s", destination)


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_latest_run(artifacts_cfg: ArtifactsConfig) -> Optional[ArtifactPaths]:
    """Return artifact paths for the most recent run, if any."""

    if not artifacts_cfg.root.exists():
        return None

    candidate_dirs = [
        path for path in artifacts_cfg.root.iterdir() if path.is_dir()
    ]
    if not candidate_dirs:
        return None

    latest_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
    return ArtifactPaths(
        run_dir=latest_dir,
        model_path=latest_dir / artifacts_cfg.model_subdir / "model.joblib",
        transformer_path=latest_dir / artifacts_cfg.transformer_subdir / "transformer.joblib",
        metrics_path=latest_dir / artifacts_cfg.metrics_subdir / "metrics.json",
        metadata_path=latest_dir / "metadata.json",
    )


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def load_transformer(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Transformer artifact not found: {path}")
    return joblib.load(path)
