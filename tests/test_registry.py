from __future__ import annotations

from pathlib import Path

from src.config import ArtifactsConfig
from src.registry import prepare_run_artifacts, resolve_latest_run


def test_prepare_run_artifacts_creates_structure(tmp_path):
    cfg = ArtifactsConfig(root=tmp_path)
    paths = prepare_run_artifacts(cfg, "test_run")

    assert paths.model_path.parent.exists()
    assert paths.transformer_path.parent.exists()
    assert paths.metrics_path.parent.exists()


def test_resolve_latest_run(tmp_path):
    cfg = ArtifactsConfig(root=tmp_path)
    _ = prepare_run_artifacts(cfg, "run1")
    paths_latest = prepare_run_artifacts(cfg, "run2")

    resolved = resolve_latest_run(cfg)
    assert resolved is not None
    assert resolved.run_dir == paths_latest.run_dir
