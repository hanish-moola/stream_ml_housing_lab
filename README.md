# Housing Price Prediction with Neural Networks

A machine learning project for predicting housing prices using deep learning, with MLflow integration for experiment tracking and model management.

## Project Structure

```
stream_ml_housing_lab/
├── config.py                 # Configuration parameters
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── notebook_parv_soni.ipynb # Original notebook
├── src/                     # Main execution scripts
│   ├── __init__.py
│   ├── train.py            # Training script with MLflow
│   └── evaluate.py         # Evaluation script with MLflow
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── visualization.py    # Visualization functions
│   └── model_builder.py    # Model architecture and hyperparameter tuning
├── data/                    # Data directory (created automatically)
├── models/                  # Saved models (created automatically)
├── results/                 # Results and visualizations (created automatically)
└── mlruns/                  # MLflow tracking data (created automatically)
```

## Features

- **Modular Design**: Separated concerns with dedicated utility modules
- **MLflow Integration**: Complete experiment tracking, model versioning, and artifact logging
- **Hyperparameter Tuning**: Automated hyperparameter search using Keras Tuner
- **Comprehensive Evaluation**: Multiple regression metrics with detailed analysis
- **Visualization Pipeline**: Automated generation of EDA plots and prediction visualizations
- **Reproducibility**: Fixed random seeds and versioned configurations

## Installation

1. Clone the repository:
```bash
cd stream_ml_housing_lab
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:
- Data paths
- Model hyperparameters
- Training parameters
- MLflow experiment settings

Key configuration options:
- `DATA_PATH`: Path to your housing dataset CSV file
- `HYPERPARAMETER_CONFIG`: Settings for hyperparameter tuning
- `TRAINING_CONFIG`: Training epochs, batch size, etc.
- `MLFLOW_CONFIG`: MLflow experiment and tracking settings

## Usage

### Training

Run the training pipeline:

```bash
python src/train.py
```

This will:
1. Load and preprocess the data
2. Generate exploratory visualizations
3. Perform hyperparameter tuning
4. Train the best model
5. Log everything to MLflow
6. Save the model and artifacts

### Evaluation

Evaluate a trained model:

```bash
# Evaluate using the latest trained model
python src/evaluate.py

# Evaluate using a specific model path
python src/evaluate.py --model-path results/models/training_run_20240101_120000

# Evaluate using an MLflow model URI
python src/evaluate.py --model-uri models:/housing_price_predictor/1
```

### MLflow UI

View experiments and models:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://localhost:5000` in your browser.

## Data Format

The dataset should be a CSV file with the following columns:
- `price`: Target variable (continuous)
- `area`: Numerical feature
- `bedrooms`, `bathrooms`, `stories`, `parking`: Numerical features
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`: Categorical features (yes/no)
- `furnishingstatus`: Categorical feature (furnished/semi-furnished/unfurnished)

## Model Architecture

The neural network consists of:
- Input layer: 20 features (after one-hot encoding)
- Hidden layer 1: 32-512 neurons (tuned)
- Hidden layer 2: 32-512 neurons (tuned)
- Output layer: 1 neuron (linear activation for regression)

Hyperparameters tuned:
- Number of neurons in each hidden layer
- Activation functions (relu, tanh, sigmoid)
- Learning rate (0.01, 0.001, 0.0001)

## Evaluation Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **RMSLE** (Root Mean Squared Logarithmic Error)
- **R² Score** (Coefficient of Determination)

## Best Practices Implemented

### Software Engineering
- ✅ Modular architecture with separation of concerns
- ✅ Configuration management via config file
- ✅ Comprehensive logging
- ✅ Error handling and validation
- ✅ Type hints and documentation
- ✅ Command-line interface with argparse
- ✅ Automated directory creation

### Machine Learning
- ✅ Reproducible experiments (fixed random seeds)
- ✅ Proper train/test splitting
- ✅ Feature scaling for neural networks
- ✅ Hyperparameter tuning
- ✅ Early stopping and learning rate reduction
- ✅ Comprehensive evaluation metrics
- ✅ Model versioning and artifact management
- ✅ Experiment tracking with MLflow

## Fixed Issues

The original notebook had a bug in the evaluation cell:
```python
# BUG: Converting regression predictions to boolean
y_pred = (y_pred > 0.5)
```

This has been fixed in `src/evaluate.py` to properly handle continuous regression predictions.

## Output

After training, you'll find:
- **Models**: Saved in `results/models/[run_name]/model.keras`
- **Scaler**: Saved in `results/models/[run_name]/scaler.pkl`
- **Visualizations**: Saved in `results/visualizations/[run_name]/`
- **MLflow Artifacts**: Logged to MLflow tracking server

## Development

To extend the project:
1. Add new preprocessing steps in `utils/preprocessing.py`
2. Add new visualizations in `utils/visualization.py`
3. Modify model architecture in `utils/model_builder.py`
4. Add new metrics in `src/evaluate.py`

## License

This project was created by Parv Soni for housing price prediction research.

## Notes

- The original notebook path expects data at `/kaggle/input/housing-prices-dataset/Housing.csv`
- Update `DATA_PATH` in `config.py` or set `HOUSING_DATA_PATH` environment variable to use your data location
- MLflow tracking uses local file storage by default (change in `config.py` for remote tracking)
