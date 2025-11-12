"""Streaming inference workflow that replays the housing CSV as a continuous feed."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mlflow

from ..config import ProjectConfig, load_config
from ..inference_service import InferenceService
from ..logging_utils import configure_logging, get_logger
from ..registry import build_run_name
from ..streaming.channel import InMemoryChannel
from ..streaming.reader import CsvStreamProducer
from ..streaming.sinks import CompositeSink, FileSink, MlflowSink
from ..streaming.worker import StreamInferenceWorker, StreamingMetrics

logger = get_logger(__name__)


@dataclass
class StreamingSummary:
    rows_received: int
    predictions_succeeded: int
    predictions_failed: int
    output_path: str


def _build_sink(output_dir: Path, file_format: str, enable_mlflow: bool) -> CompositeSink:
    sinks = []
    file_sink = FileSink(output_dir=output_dir, file_prefix="predictions", file_extension=file_format)
    sinks.append(file_sink)
    if enable_mlflow:
        sinks.append(MlflowSink())
    return CompositeSink(sinks), file_sink


def run_streaming_inference(
    config: ProjectConfig,
    *,
    run_name: Optional[str] = None,
) -> StreamingSummary:
    """Execute the streaming inference workflow."""

    configure_logging()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    producer = CsvStreamProducer(config.data, config.streaming)
    channel = InMemoryChannel()
    inference_service = InferenceService(config)
    worker = StreamInferenceWorker(inference_service, include_ground_truth=config.streaming.include_ground_truth)

    effective_run_name = run_name or build_run_name(config.mlflow.run_name_template)
    output_dir = config.streaming.output_dir / effective_run_name
    sink, file_sink = _build_sink(output_dir, config.streaming.output_format, config.streaming.enable_mlflow_logging)

    batch_size = config.streaming.batch_size
    sleep_interval = config.streaming.sleep_interval_seconds
    with mlflow.start_run(run_name=effective_run_name) as run:
        mlflow.set_tag("stage", "streaming_inference")
        mlflow.log_param("stream_batch_size", batch_size)
        mlflow.log_param("stream_sleep_interval", sleep_interval)
        mlflow.log_param("stream_loop_forever", config.streaming.loop_forever)

        for row in producer.stream_rows():
            channel.publish(row)
            batch = channel.consume_batch(batch_size, block=False)
            if not batch:
                continue

            events = worker.score_batch(batch)
            sink.write_batch(events)
            _log_metrics(worker.metrics_snapshot())

            if config.streaming.loop_forever and sleep_interval > 0:
                time.sleep(sleep_interval)

        # Drain any remaining rows
        remaining = channel.drain()
        if remaining:
            events = worker.score_batch(remaining)
            sink.write_batch(events)
            _log_metrics(worker.metrics_snapshot())

        sink.flush()

        metrics = worker.metrics_snapshot()
        summary = StreamingSummary(
            rows_received=metrics.rows_received,
            predictions_succeeded=metrics.predictions_succeeded,
            predictions_failed=metrics.predictions_failed,
            output_path=str(file_sink.file_path),
        )
        mlflow.log_dict(asdict(summary), "streaming_summary.json")
        logger.info("Streaming inference finished: %s", summary)

    return summary


def _log_metrics(metrics: StreamingMetrics) -> None:
    if mlflow.active_run() is None:
        return
    mlflow.log_metrics(
        {
            "stream_rows_received": metrics.rows_received,
            "stream_predictions_succeeded": metrics.predictions_succeeded,
            "stream_predictions_failed": metrics.predictions_failed,
        }
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming inference over the housing CSV.")
    parser.add_argument("--config", type=Path, help="Optional configuration file override.")
    parser.add_argument("--run-name", type=str, help="Optional explicit MLflow run name.")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    cli_args = parse_args(argv)
    config = load_config(path=cli_args.config)
    summary = run_streaming_inference(config, run_name=cli_args.run_name)
    print(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
