"""Faust-based feature processor that keeps online features in sync with training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faust
import numpy as np

from config import CATEGORICAL_COLUMNS
from utils.pipeline_loader import load_pipeline
from .config import SETTINGS
from .schemas import FeatureVector, HousingEvent

logger = logging.getLogger(__name__)

app = faust.App(
    "housing-feature-service",
    broker=f"kafka://{SETTINGS.broker_url}",
    value_serializer="raw",
)

raw_topic = app.topic(SETTINGS.raw_topic)
feature_topic = app.topic(SETTINGS.feature_topic)
prediction_topic = app.topic(SETTINGS.predictions_topic)


class PipelineState:
    pipeline = None
    feature_order: List[str] = []
    feature_names: List[str] = []
    boolean_categorical: List[str] = []


state = PipelineState()


def _load_metadata(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def _resolve_feature_order(metadata: Dict[str, object], pipeline) -> List[str]:
    numerical = metadata.get("numerical_columns", [])
    categorical = metadata.get("categorical_columns", [])
    if numerical or categorical:
        return [*numerical, *categorical]
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    raise RuntimeError("Unable to determine feature order for the loaded pipeline")


def _resolve_feature_names(pipeline, metadata: Dict[str, object]) -> List[str]:
    if metadata.get("numerical_columns") and metadata.get("categorical_columns"):
        # Feature names after transformation require the fitted encoder
        transformer = pipeline.named_steps.get("features")
        if transformer is not None and hasattr(transformer, "get_feature_names_out"):
            return transformer.get_feature_names_out().tolist()
    if hasattr(pipeline.named_steps.get("regressor"), "feature_importances_"):
        # Fallback to order of input columns if transformer metadata missing
        return _resolve_feature_order(metadata, pipeline)
    raise RuntimeError("Unable to resolve transformed feature names")


@app.agent(raw_topic)
async def process(events):
    if state.pipeline is None:
        pipeline, resolved_path, _ = load_pipeline()
        metadata = _load_metadata(resolved_path)
        state.pipeline = pipeline
        state.feature_order = _resolve_feature_order(metadata, pipeline)
        transformer = pipeline.named_steps["features"]
        state.feature_names = transformer.get_feature_names_out().tolist()
        categorical = metadata.get("categorical_columns") or CATEGORICAL_COLUMNS
        state.boolean_categorical = [col for col in categorical if col != "furnishingstatus"]
        logger.info("Loaded pipeline for feature processor with %d features", len(state.feature_names))

    async for message in events:
        event = HousingEvent.model_validate_json(message.decode("utf-8"))
        features = event.payload.to_dataframe(state.feature_order, state.boolean_categorical)
        transformed = state.pipeline.named_steps["features"].transform(features)
        prediction = float(state.pipeline.predict(features)[0])

        feature_payload = FeatureVector(
            event_id=event.event_id,
            ingest_ts=event.ingest_ts,
            features=transformed[0].tolist() if isinstance(transformed, np.ndarray) else list(transformed[0]),
            feature_names=state.feature_names,
            prediction=prediction,
        )

        await feature_topic.send(value=feature_payload.model_dump_json().encode("utf-8"))
        await prediction_topic.send(value=feature_payload.model_dump_json().encode("utf-8"))


if __name__ == "__main__":
    app.main()
