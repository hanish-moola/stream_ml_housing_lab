# Pipeline Overview

## Stage Summary

0. **Offline Workflow (`poetry run offline-train`)**
   - Orchestrates feature engineering, training, and evaluation in sequence.
   - Accepts dataset overrides via `--data-path` and logs a consolidated MLflow run summarising stage outputs.

1. **Feature Engineering (`poetry run feature-engineering`)**
   - Loads the raw dataset via `src/data.py`.
   - Infers numeric/categorical columns, fits a `ColumnTransformer`, and persists transformer + metadata.
   - Logs feature statistics to MLflow.

2. **Model Training (`poetry run train-model`)**
   - Reuses the persisted transformer; trains the configured estimator; logs metrics and artifacts.
   - Saves an inference-ready sklearn pipeline (`transformer + estimator`).

3. **Model Evaluation (`poetry run evaluate-model`)**
   - Loads the latest (or specified) trained pipeline and evaluates on the hold-out set.
   - Logs evaluation metrics and predictions to MLflow.

4. **Prediction (`poetry run predict-model --features payload.json`)**
   - Validates payloads against saved feature metadata and scores them with the packaged pipeline.
   - Logs inference metadata to MLflow.

## Artifact Layout

```
artifacts/
└── <run_name>/
    ├── metrics/                 # JSON metrics + CSV predictions
    ├── models/                  # model.joblib (sklearn Pipeline)
    ├── transformers/            # transformer.joblib
    ├── feature_metadata.json    # numeric/categorical feature list
    ├── training_stats.json      # numeric means/stds and categorical counts
    └── metadata.json            # links artifacts with MLflow run IDs
```

Evaluation runs use their own `<run_name>` directories with the same structure.

## MLflow Tracking

All stages log to the experiment defined in `config/config.yaml` (default `housing_price_workflow`).
Run the UI locally with:

```bash
poetry run mlflow ui --backend-store-uri file:./mlruns
```

## CLI Reference

| Stage               | Example Command |
|---------------------|-----------------|
| Feature engineering | `poetry run feature-engineering --config config/config.yaml` |
| Training            | `poetry run train-model --config config/config.yaml` |
| Evaluation          | `poetry run evaluate-model --config config/config.yaml` |
| Prediction          | `poetry run predict-model --config config/config.yaml --features payload.json` |

Pass `--run-name` to override generated run identifiers or `--model-run` (where supported) to target
specific training artifacts.
