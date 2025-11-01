# Configuration Reference

Configuration defaults live in `config/config.yaml` and can be overridden via CLI `--config`
or environment variables (`HOUSING_DATA_PATH`, `HOUSING_ARTIFACTS_ROOT`, `MLFLOW_TRACKING_URI`).

| Section      | Key                  | Description |
|--------------|----------------------|-------------|
| `project`    | `name`               | Display name used in metadata. |
| `data`       | `raw_data_path`      | Path to the housing CSV. Override with `HOUSING_DATA_PATH`. |
|              | `target_column`      | Target variable (default `price`). |
|              | `index_column`       | Optional column to set as index. |
|              | `test_size`          | Proportion of data used for the test split. |
|              | `random_state`       | Seed controlling deterministic splits. |
| `artifacts`  | `root`               | Root directory for run artifacts (`artifacts/`). Override via `HOUSING_ARTIFACTS_ROOT`. |
|              | `transformer_subdir` | Subdirectory for transformers (`transformers`). |
|              | `model_subdir`       | Subdirectory for trained models (`models`). |
|              | `metrics_subdir`     | Subdirectory for metrics and predictions (`metrics`). |
| `mlflow`     | `experiment_name`    | MLflow experiment name (default `housing_price_workflow`). |
|              | `tracking_uri`       | Tracking URI (`mlruns`). Override with `MLFLOW_TRACKING_URI`. |
|              | `run_name_template`  | Template used to generate run names (supports `{timestamp}`). |
| `model`      | `type`               | Model identifier (currently `linear_regression`). |
|              | `hyperparameters`    | Dict passed to the estimator on construction. |

To provide overrides, export environment variables before running CLI commands, e.g.:

```bash
export HOUSING_DATA_PATH=/data/Housing.csv
poetry run train-model
```

### Neural Network Hyperparameters

Set `model.type: neural_network` and provide keys in `model.hyperparameters` such as:

| Key | Description |
|-----|-------------|
| `hidden_units` | List of layer sizes (e.g. `[128, 64]`). |
| `activation` | Activation function (default `relu`). |
| `dropout` | Dropout rate applied after each hidden layer. |
| `learning_rate` | Adam optimizer learning rate. |
| `epochs` | Training epochs (default 200). |
| `batch_size` | Batch size (default 32). |
| `validation_split` | Fraction of training data held out for validation. |

