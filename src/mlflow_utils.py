"""Helper utilities for MLflow run management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow


@contextmanager
def ensure_run(run_name: str, *, nested: bool = False) -> Iterator[mlflow.ActiveRun]:
    """Return an MLflow run context, reusing the active run when present.

    Args:
        run_name: Name for the run if a new one is started.
        nested: When True, force creation of a nested run even if one is active.
    """

    active = mlflow.active_run()
    if active and not nested:
        yield active
    else:
        with mlflow.start_run(run_name=run_name, nested=nested) as new_run:
            yield new_run
