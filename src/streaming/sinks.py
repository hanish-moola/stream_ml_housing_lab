"""Sinks for handling streaming prediction events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - mlflow may not be installed in minimal test envs
    import mlflow
except ModuleNotFoundError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

from .types import PredictionEvent


class BaseSink:
    """Base interface for streaming sinks."""

    def write_batch(self, events: Sequence[PredictionEvent]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError


@dataclass
class FileSink(BaseSink):
    """Writes prediction events to newline-delimited JSON."""

    output_dir: Path
    file_prefix: str
    file_extension: str = "jsonl"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.output_dir / f"{self.file_prefix}_{timestamp}.{self.file_extension}"

    def write_batch(self, events: Sequence[PredictionEvent]) -> None:
        if not events:
            return
        with self.file_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.to_dict()))
                handle.write("\n")

    def flush(self) -> None:
        # File writes are flushed per call; nothing to do here.
        return


class MlflowSink(BaseSink):
    """Logs batch-level summaries to MLflow."""

    def __init__(self, log_predictions: bool = False) -> None:
        self._log_predictions = log_predictions
        self._batch_index = 0
        self._total_events = 0

    def write_batch(self, events: Sequence[PredictionEvent]) -> None:
        if not events or mlflow is None or mlflow.active_run() is None:
            return

        self._batch_index += 1
        self._total_events += len(events)
        mlflow.log_metrics(
            {
                "stream_batch_size": len(events),
                "stream_total_predictions": self._total_events,
            }
        )

        if self._log_predictions:
            payload = "\n".join(json.dumps(event.to_dict()) for event in events)
            artifact_path = f"stream/predictions_batch_{self._batch_index}.jsonl"
            mlflow.log_text(payload, artifact_path)

    def flush(self) -> None:
        return


class CompositeSink(BaseSink):
    """Fan-out sink that forwards events to multiple sinks."""

    def __init__(self, sinks: Iterable[BaseSink]):
        self._sinks: List[BaseSink] = list(sinks)

    def write_batch(self, events: Sequence[PredictionEvent]) -> None:
        for sink in self._sinks:
            sink.write_batch(events)

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()
