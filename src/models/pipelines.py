"""Model pipeline utilities for custom estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class KerasRegressionPipeline:
    """Lazy-loading pipeline combining a transformer and a Keras model."""

    def __init__(self, transformer, model_dir: Path):
        self.transformer = transformer
        self.model_dir = str(model_dir)
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from tensorflow import keras

            self._model = keras.models.load_model(self.model_dir)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_model()
        features = self.transformer.transform(X)
        predictions = self._model.predict(features, verbose=0).flatten()
        return predictions

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
