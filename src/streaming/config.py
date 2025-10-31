"""Streaming configuration and constants."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaSettings:
    broker_url: str = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
    raw_topic: str = os.getenv("HOUSING_RAW_TOPIC", "raw_housing")
    feature_topic: str = os.getenv("HOUSING_FEATURE_TOPIC", "feature_housing")
    predictions_topic: str = os.getenv("HOUSING_PREDICTIONS_TOPIC", "predictions_housing")


SETTINGS = KafkaSettings()
