"""Async Kafka producer that streams housing events from a static dataset."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from aiokafka import AIOKafkaProducer
except ImportError:  # pragma: no cover - optional dependency during docs builds
    AIOKafkaProducer = None  # type: ignore

from config import CATEGORICAL_COLUMNS, DATA_PATH, EXCLUDE_COLUMNS
from utils.data_loader import load_housing_data
from ..serving.schemas import HousingFeatures
from .config import SETTINGS
from .schemas import HousingEvent


BOOLEAN_CATEGORICAL = [col for col in CATEGORICAL_COLUMNS if col != "furnishingstatus"]


def _row_to_features(row: pd.Series) -> HousingFeatures:
    payload = {}
    for column, value in row.items():
        if column in EXCLUDE_COLUMNS or column == "price":
            continue
        if column in BOOLEAN_CATEGORICAL:
            payload[column] = str(value).lower() in {"yes", "true", "1"}
        elif column == "furnishingstatus":
            payload[column] = str(value).lower()
        else:
            payload[column] = float(value)
    return HousingFeatures(**payload)  # type: ignore[arg-type]


async def stream_events(
    dataframe: pd.DataFrame,
    interval: float,
    topic: str,
    bootstrap_servers: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    if AIOKafkaProducer is None and not dry_run:
        raise RuntimeError("aiokafka is not installed. Install it or use --dry-run")

    producer = None
    if not dry_run:
        producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
        await producer.start()

    try:
        for idx, row in enumerate(dataframe.to_dict(orient="records"), 1):
            features = _row_to_features(pd.Series(row))
            event = HousingEvent(
                event_id=str(uuid.uuid4()),
                ingest_ts=datetime.utcnow(),
                payload=features,
            )
            payload_bytes = event.model_dump_json().encode("utf-8")

            if dry_run:
                print(payload_bytes.decode("utf-8"))
            else:
                assert producer is not None
                await producer.send_and_wait(topic, payload_bytes)

            if limit and idx >= limit:
                break

            await asyncio.sleep(max(interval, 0))
    finally:
        if producer is not None:
            await producer.stop()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream housing events into Kafka")
    parser.add_argument("--source", type=str, default=str(DATA_PATH), help="Path to the housing CSV data")
    parser.add_argument("--topic", type=str, default=SETTINGS.raw_topic, help="Kafka topic to publish raw events")
    parser.add_argument("--bootstrap", type=str, default=SETTINGS.broker_url, help="Kafka bootstrap servers")
    parser.add_argument("--interval", type=float, default=0.5, help="Delay between events in seconds")
    parser.add_argument("--limit", type=int, help="Optional number of events to stream")
    parser.add_argument("--dry-run", action="store_true", help="Print events instead of publishing")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    df = load_housing_data(str(source_path))

    asyncio.run(
        stream_events(
            dataframe=df,
            interval=args.interval,
            topic=args.topic,
            bootstrap_servers=args.bootstrap,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
