from __future__ import annotations

import json
from datetime import datetime

from src.config import DataConfig, StreamingConfig
from src.inference_service import PredictionResult
from src.streaming.reader import CsvStreamProducer
from src.streaming.sinks import FileSink
from src.streaming.types import PredictionEvent, StreamingRow
from src.streaming.worker import StreamInferenceWorker


def _make_streaming_config(tmp_path):
    return StreamingConfig(
        batch_size=2,
        sleep_interval_seconds=0,
        loop_forever=False,
        max_rows_per_cycle=None,
        checkpoint_path=tmp_path / "state.json",
        output_dir=tmp_path,
        output_format="jsonl",
        include_ground_truth=True,
        enable_mlflow_logging=False,
    )


def test_csv_stream_producer_emits_rows_without_target(tmp_path):
    csv_path = tmp_path / "housing.csv"
    csv_path.write_text("price,area,stories\n100,900,2\n200,1200,3\n", encoding="utf-8")

    data_cfg = DataConfig(
        raw_data_path=csv_path,
        target_column="price",
        index_column=None,
        test_size=0.2,
        random_state=42,
    )
    streaming_cfg = _make_streaming_config(tmp_path)
    producer = CsvStreamProducer(data_cfg, streaming_cfg)

    rows = list(producer.stream_rows())
    assert len(rows) == 2
    assert rows[0].features == {"area": 900, "stories": 2}
    assert rows[0].target == 100.0
    assert rows[1].row_id == 2


class _StubInferenceService:
    def predict(self, payload):
        return PredictionResult(
            prediction=float(payload["area"]),
            used_features=payload,
            imputed={},
            model_run_id="run-123",
            model_type="stub",
        )


def test_stream_worker_produces_prediction_events(tmp_path):
    worker = StreamInferenceWorker(_StubInferenceService(), include_ground_truth=True)

    row = StreamingRow(row_id=1, features={"area": 1500}, target=250000.0, timestamp=datetime.utcnow())
    events = worker.score_batch([row])

    assert len(events) == 1
    event = events[0]
    assert event.prediction == 1500.0
    assert event.target == 250000.0

    metrics = worker.metrics_snapshot()
    assert metrics.rows_received == 1
    assert metrics.predictions_succeeded == 1


def test_file_sink_writes_jsonl(tmp_path):
    sink = FileSink(output_dir=tmp_path, file_prefix="predictions")
    event = PredictionEvent(
        row_id=1,
        prediction=123.4,
        model_run_id="run",
        model_type="stub",
        features={"area": 1000},
        target=120000.0,
        source_timestamp=datetime.utcnow(),
        processed_at=datetime.utcnow(),
    )

    sink.write_batch([event])

    contents = sink.file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["prediction"] == 123.4
