"""Shared streaming dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StreamingRow:
    """Represents a single row pulled from the CSV stream."""

    row_id: int
    features: Dict[str, Any]
    target: Optional[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class PredictionEvent:
    """Represents an inference result emitted by the streaming worker."""

    row_id: int
    prediction: float
    model_run_id: str
    model_type: str
    features: Dict[str, Any]
    target: Optional[float]
    source_timestamp: datetime
    processed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event for downstream sinks."""

        return {
            "row_id": self.row_id,
            "prediction": self.prediction,
            "model_run_id": self.model_run_id,
            "model_type": self.model_type,
            "features": self.features,
            "target": self.target,
            "source_timestamp": self.source_timestamp.isoformat(),
            "processed_at": self.processed_at.isoformat(),
        }
