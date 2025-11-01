# Stream-ML Housing Lab

A reference implementation of a modular, MLflow-enabled housing price pipeline. The repository
demonstrates how to take a notebook workflow and refactor it into production-friendly scripts with
clear separation of concerns, reproducible configuration, and auditable artifacts.

## Project Structure

```
stream_ml_housing_lab/
├── config/                 # YAML configuration and overrides
├── docs/                   # Pipeline and configuration documentation
├── src/
│   ├── config.py           # Config loader with env overrides
│   ├── data.py             # Dataset loading/splitting helpers
│   ├── feature_engineering.py  # CLI to fit & persist preprocessing transformers
│   ├── train.py            # CLI to train model and log artifacts to MLflow
│   ├── evaluate.py         # CLI to score trained models on hold-out data
│   ├── predict.py          # CLI to run single predictions with saved pipelines
│   ├── logging_utils.py    # Project logging setup
│   └── registry.py         # Artifact persistence and lookup helpers
├── tests/                  # Pytest coverage for utilities and CLIs
├── artifacts/              # Generated artifacts (gitignored, .gitkeep for tree)
├── mlruns/                 # Local MLflow tracking store (gitignored, .gitkeep)
├── IMPLEMENTATION_PLAN.md  # High-level roadmap
├── DECISIONS.md            # Architectural decision log
└── pyproject.toml          # Poetry configuration & dependency manifest
```

## Getting Started

1. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

2. **Activate the environment (optional)**
   ```bash
   poetry shell
   ```

3. **Configure dataset location** (if different from default)
   ```bash
   export HOUSING_DATA_PATH=/path/to/Housing.csv
   ```

4. **Run the pipeline stages**
   ```bash
   poetry run feature-engineering
   poetry run train-model
   poetry run evaluate-model
   poetry run predict-model --features payload.json
   ```

   To execute the entire workflow (feature engineering → training → evaluation) in one command and log an offline model to MLflow, run:

   ```bash
   poetry run offline-train --data-path /path/to/Housing.csv
   ```

Each command accepts `--config` for alternative configuration files and `--run-name` to override the
auto-generated identifier. Evaluation/prediction additionally support `--model-run` to target specific
training artifacts. The offline workflow also accepts these flags and will cascade overrides to each stage.

## Command Cheatsheet

| Stage               | Example |
|---------------------|---------|
| Feature engineering | `poetry run feature-engineering --config config/config.yaml`
| Training            | `poetry run train-model --config config/config.yaml`
| Evaluation          | `poetry run evaluate-model --config config/config.yaml`
| Prediction          | `poetry run predict-model --config config/config.yaml --features payload.json`
| Offline workflow    | `poetry run offline-train --config config/config.yaml --data-path /path/to/Housing.csv`

See `docs/PIPELINE_OVERVIEW.md` for a deeper walkthrough of the stages and artifact layout.

## Configuration

Configuration defaults live in `config/config.yaml` and can be overridden via environment variables or
custom YAML files. A detailed reference is provided in `docs/CONFIG_REFERENCE.md`.

Key overrides:
- `HOUSING_DATA_PATH` – path to the dataset CSV.
- `HOUSING_ARTIFACTS_ROOT` – directory for artifact storage.
- `MLFLOW_TRACKING_URI` – MLflow backend URI (defaults to local `mlruns`).
- `model.type` – switch between `linear_regression` (default) and `neural_network`; adjust `model.hyperparameters` accordingly.

## MLflow Tracking

The pipeline logs parameters, metrics, and artifacts to MLflow:

```bash
poetry run mlflow ui --backend-store-uri file:./mlruns
```

Open `http://127.0.0.1:5000` to inspect runs, compare metrics, and audit artifacts.

## Testing

Run the test suite (once dependencies are installed):

```bash
poetry run pytest
```

The tests cover configuration loading, artifact registry behaviour, feature engineering helpers,
training/evaluation utilities, and prediction payload validation.

## Dataset Contract

The pipeline expects the following columns (defaults sourced from the original Kaggle dataset):

| Column             | Type / Values                        | Notes |
|--------------------|---------------------------------------|-------|
| `price`            | float                                 | Target variable |
| `area`             | float                                 | Property area |
| `bedrooms`         | int                                   | Number of bedrooms |
| `bathrooms`        | float                                 | Number of bathrooms |
| `stories`          | int                                   | Number of stories |
| `parking`          | int                                   | Parking spaces |
| `mainroad`         | {"yes","no"}                        | Access to main road |
| `guestroom`        | {"yes","no"}                        | Guest room availability |
| `basement`         | {"yes","no"}                        | Basement availability |
| `hotwaterheating`  | {"yes","no"}                        | Hot water heating |
| `airconditioning`  | {"yes","no"}                        | Air conditioning |
| `prefarea`         | {"yes","no"}                        | Preferred area indicator |
| `furnishingstatus` | {"furnished","semi-furnished","unfurnished"} | Furnishing category |

Additional columns are ignored unless wired into the feature engineering pipeline.

## Additional Documentation

- `docs/PIPELINE_OVERVIEW.md` – end-to-end flow and CLI reference
- `docs/CONFIG_REFERENCE.md` – configuration key details and overrides
- `DECISIONS.md` – running log of architectural choices

## License

Apache-2.0 (see `LICENSE` when available). Authored by Hanish Moola.
