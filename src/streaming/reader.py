"""CSV streaming producer that yields rows for inference."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..config import DataConfig, StreamingConfig
from ..logging_utils import get_logger
from .types import StreamingRow

logger = get_logger(__name__)


@dataclass
class _CheckpointStore:
    """Persists the last processed row index."""

    path: Optional[Path]

    def load(self) -> int:
        if self.path is None or not self.path.exists():
            return 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return int(payload.get("offset", 0))
        except (json.JSONDecodeError, ValueError):
            return 0

    def save(self, offset: int) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump({"offset": offset}, handle)

    def reset(self) -> None:
        if self.path and self.path.exists():
            self.path.unlink()


class CsvStreamProducer:
    """Reads the housing CSV and yields StreamingRow objects."""

    def __init__(self, data_config: DataConfig, streaming_config: StreamingConfig) -> None:
        self._data_config = data_config
        self._streaming = streaming_config
        self._checkpoint = _CheckpointStore(streaming_config.checkpoint_path)

    def stream_rows(self) -> Iterator[StreamingRow]:
        csv_path = self._data_config.raw_data_path
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV path not found: {csv_path}")

        max_rows = self._streaming.max_rows_per_cycle
        processed = 0

        while True:
            start_offset = self._checkpoint.load()
            logger.info("Starting CSV stream from offset %s", start_offset)
            for row in self._yield_rows(csv_path, start_offset):
                processed += 1
                yield row
                self._checkpoint.save(row.row_id)
                if max_rows and processed >= max_rows:
                    logger.info("Reached max_rows_per_cycle=%s; stopping stream", max_rows)
                    return

            if not self._streaming.loop_forever:
                break

            logger.info("CSV stream completed; sleeping before restart")
            self._checkpoint.reset()
            time.sleep(self._streaming.sleep_interval_seconds)

    def _yield_rows(self, csv_path: Path, start_offset: int) -> Iterator[StreamingRow]:
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row_index = 0
            for row in reader:
                row_index += 1
                if row_index <= start_offset:
                    continue

                target_raw = row.pop(self._data_config.target_column, None)
                target_value = self._parse_numeric(target_raw)
                features = {key: self._coerce_value(value) for key, value in row.items()}
                yield StreamingRow(row_id=row_index, features=features, target=target_value)

    @staticmethod
    def _parse_numeric(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _coerce_value(value: Optional[str]):
        if value is None:
            return None
        value = value.strip()
        if value == "":
            return None
        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                continue
        return value
