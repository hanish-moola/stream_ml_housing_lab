"""Schemas for Kafka streaming events."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from ..serving.schemas import HousingFeatures


class HousingEvent(BaseModel):
    event_id: str = Field(description="Unique identifier for the event")
    ingest_ts: datetime = Field(default_factory=datetime.utcnow)
    payload: HousingFeatures


class FeatureVector(BaseModel):
    event_id: str
    ingest_ts: datetime
    features: List[float]
    feature_names: List[str]
    prediction: float
