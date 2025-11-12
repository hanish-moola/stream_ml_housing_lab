"""Streaming worker that performs inference on queued rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Sequence

from ..inference_service import InferenceService
from ..logging_utils import get_logger
from .types import PredictionEvent, StreamingRow

logger = get_logger(__name__)


@dataclass
class StreamingMetrics:
    """Simple container for streaming metrics."""

    rows_received: int = 0
    predictions_succeeded: int = 0
    predictions_failed: int = 0

    def register_success(self) -> None:
        self.rows_received += 1
        self.predictions_succeeded += 1

    def register_failure(self) -> None:
        self.rows_received += 1
        self.predictions_failed += 1


class StreamInferenceWorker:
    """Consumes StreamingRow instances and emits PredictionEvents."""

    def __init__(self, inference_service: InferenceService, include_ground_truth: bool = True) -> None:
        self._service = inference_service
        self._include_ground_truth = include_ground_truth
        self._metrics = StreamingMetrics()

    def score_batch(self, rows: Sequence[StreamingRow]) -> List[PredictionEvent]:
        results: List[PredictionEvent] = []
        for row in rows:
            try:
                prediction = self._service.predict(row.features)
            except Exception:
                logger.exception("Failed to score row_id=%s", row.row_id)
                self._metrics.register_failure()
                continue

            event = PredictionEvent(
                row_id=row.row_id,
                prediction=prediction.prediction,
                model_run_id=prediction.model_run_id,
                model_type=prediction.model_type,
                features=row.features,
                target=row.target if self._include_ground_truth else None,
                source_timestamp=row.timestamp,
                processed_at=datetime.utcnow(),
            )
            results.append(event)
            self._metrics.register_success()
        return results

    def metrics_snapshot(self) -> StreamingMetrics:
        return replace(self._metrics)
