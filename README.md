# Stream-ML Housing Lab

A production-minded playground for housing price prediction. The lab now ships with a shared feature
pipeline, an XGBoost regression model tracked in MLflow, a FastAPI inference surface, and Kafka-ready
streaming primitives to move towards near-real-time predictions and agentic stress testing.

## Project Structure

```
stream_ml_housing_lab/
├── config.py                # Central configuration for data paths, model params, MLflow
├── requirements.txt         # Python dependencies
├── src/
│   ├── train.py             # Offline training entrypoint (MLflow + XGBoost pipeline)
│   ├── evaluate.py          # Offline evaluation wired into MLflow
│   ├── serving/             # FastAPI app + request/response schemas
│   └── streaming/           # Kafka producer + Faust feature processor skeleton
├── utils/
│   ├── data_loader.py       # Data ingestion helpers
│   ├── feature_pipeline.py  # Shared feature transformer utilities
│   ├── metrics.py           # Regression metrics helpers
│   ├── pipeline_loader.py   # Centralised pipeline loading helpers
│   └── visualization.py     # Exploratory and diagnostic plots
├── results/                 # Model artefacts, visualisations, evaluation reports
├── mlruns/                  # Local MLflow tracking store
└── data/                    # Default data drop (created on demand)
```

## What's New in This Iteration

- **Unified Feature Pipeline** – ColumnTransformer with scaling + one-hot encoding shared across training,
  evaluation, serving, and streaming to guarantee parity.
- **XGBoost Baseline** – Replaces the Keras network with an XGBoost regressor logged via `mlflow.sklearn`
  and versioned in the registry.
- **FastAPI Inference Service** – `src/serving/app.py` exposes `/predict`, `/health`, and `/model-info`
  endpoints and loads the latest pipeline automatically (local artefact or MLflow URI).
- **Kafka Streaming Skeleton** – `src/streaming/producer.py` streams raw housing events while
  `src/streaming/feature_processor.py` (Faust) converts them into model-ready feature vectors and
  predictions.
- **Lean Dependencies** – Requirements trimmed to the current stack plus the new serving/streaming libs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Most commands rely on the defaults in `config.py`. Override with environment variables when needed
(e.g. `HOUSING_DATA_PATH`, `HOUSING_MODEL_PATH`, `KAFKA_BROKER_URL`).

## Usage

### 1. Train Offline

```bash
python src/train.py
```

- Loads the configured dataset, fits the shared transformer, trains an XGBoost regressor.
- Logs metrics, artefacts, and the pipeline to MLflow and `results/models/<run_name>`.
- Outputs diagnostics (visualisations, feature importance, metadata JSON).

### 2. Evaluate a Model Snapshot

```bash
# Evaluate the latest artefact under results/models
python src/evaluate.py

# Evaluate an explicit folder
python src/evaluate.py --model-path results/models/training_run_20240101_120000

# Evaluate a registered MLflow model version
python src/evaluate.py --model-uri models:/housing_price_predictor/1
```

Evaluation logs metrics back to the same MLflow experiment and produces plots under
`results/evaluation/<run_name>`.

### 3. Serve Predictions via FastAPI

```bash
# Load latest local artefact
uvicorn src.serving.app:app --reload

# Or point to a specific model directory / MLflow URI
HOUSING_MODEL_PATH=results/models/training_run_... uvicorn src.serving.app:app
HOUSING_MODEL_URI=models:/housing_price_predictor/Production uvicorn src.serving.app:app
```

Endpoints:
- `GET /health` – readiness signal with current model version
- `GET /model-info` – feature order, categorical config, source metadata
- `POST /predict` – accepts housing features, returns `estimated_price`

### 4. Stream Raw Events into Kafka (Prototype)

```bash
# Publish dataset rows into a raw Kafka topic (dry run just prints JSON)
python src/streaming/producer.py --dry-run --limit 5
python src/streaming/producer.py --bootstrap localhost:9092 --topic raw_housing
```

The producer converts historic rows into the canonical feature payload defined for serving.

### 5. Online Feature Parity with Faust (Prototype)

```bash
faust -A src.streaming.feature_processor worker -l info
```

The agent consumes the raw topic, applies the shared feature transformer, and publishes both the
feature vector and the associated prediction to downstream topics (`feature_housing`, `predictions_housing`).
This creates the backbone for shadow serving, drift monitoring, and multi-agent stress drills.

## Data Contract

The housing dataset should provide the following columns:

| Column             | Type    | Notes                              |
|--------------------|---------|------------------------------------|
| `price`            | float   | Supervised target (USD)            |
| `area`             | float   | Property area                      |
| `bedrooms`         | int     | Number of bedrooms                 |
| `bathrooms`        | float   | Number of bathrooms                |
| `stories`          | int     | Number of stories                  |
| `parking`          | int     | Parking spots                      |
| `mainroad`         | yes/no  | Categorical (access to main road)  |
| `guestroom`        | yes/no  | Categorical                        |
| `basement`         | yes/no  | Categorical                        |
| `hotwaterheating`  | yes/no  | Categorical                        |
| `airconditioning`  | yes/no  | Categorical                        |
| `prefarea`         | yes/no  | Categorical                        |
| `furnishingstatus` | string  | `furnished`, `semi-furnished`, `unfurnished` |
| `Address`          | string  | High-cardinality (dropped for now) |

Set `HOUSING_DATA_PATH` to point at your CSV if it differs from the default.

## MLflow Quickstart

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Browse experiments, compare runs, and promote the registered `housing_price_predictor` model.

## Next Steps

- Wire Kafka topics into observability and drift detectors
- Add guardrail agents that react to metrics streamed out of the Faust app
- Extend the FastAPI surface with batch prediction and schema validation endpoints
- Deploy the full stack via Docker Compose for reproducible local drills

## License

Apache-2.0 — see `LICENSE` if/when it lands. Current work authored by Hanish Moola.
