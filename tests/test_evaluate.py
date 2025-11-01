from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ArtifactsConfig
from src.evaluate import _resolve_model_artifacts
from src.registry import prepare_run_artifacts, save_model, save_transformer


def _touch_artifacts(paths) -> None:
    save_model({"dummy": True}, paths.model_path)
    save_transformer({"transformer": True}, paths.transformer_path)


def test_resolve_model_artifacts_latest(tmp_path):
    cfg = ArtifactsConfig(root=tmp_path)
    older = prepare_run_artifacts(cfg, "run1")
    _touch_artifacts(older)
    newer = prepare_run_artifacts(cfg, "run2")
    _touch_artifacts(newer)

    resolved = _resolve_model_artifacts(cfg)

    assert resolved["model"] == newer.model_path
    assert resolved["transformer"] == newer.transformer_path


def test_resolve_model_artifacts_explicit_path(tmp_path):
    cfg = ArtifactsConfig(root=tmp_path)
    paths = prepare_run_artifacts(cfg, "explicit_run")
    _touch_artifacts(paths)

    resolved = _resolve_model_artifacts(cfg, model_run_dir=paths.run_dir)
    assert resolved["model"] == paths.model_path


def test_resolve_model_artifacts_missing_model(tmp_path):
    cfg = ArtifactsConfig(root=tmp_path)
    paths = prepare_run_artifacts(cfg, "broken_run")
    # intentionally do not create model file
    with pytest.raises(FileNotFoundError):
        _resolve_model_artifacts(cfg, model_run_dir=paths.run_dir)
